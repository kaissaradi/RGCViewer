from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSplitter, QSizePolicy, QComboBox, QStackedWidget, QSlider,
    QFrame,
)
from qtpy.QtCore import Qt, QTimer
import pyqtgraph as pg

from ..widgets.widgets import MplCanvas
from .cell_tracer_dialog import CellTracerDialog

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..main_window import MainWindow

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def compute_ei_map(ei: np.ndarray, channel_positions: np.ndarray) -> np.ndarray | None:
    """
    Interpolate max-absolute-amplitude per channel onto a 2-D grid.
    Returns None on shape mismatch.
    """
    if ei.shape[0] != channel_positions.shape[0]:
        logger.error(
            "EI channel count mismatch: ei=%d, positions=%d",
            ei.shape[0], channel_positions.shape[0],
        )
        return None

    xrange = (channel_positions[:, 0].min(), channel_positions[:, 0].max())
    yrange = (channel_positions[:, 1].min(), channel_positions[:, 1].max())

    y_dim = 30
    x_dim = max(1, int((xrange[1] - xrange[0]) / max(yrange[1] - yrange[0], 1) * y_dim))

    x_e = np.linspace(xrange[0], xrange[1], x_dim)
    y_e = np.linspace(yrange[0], yrange[1], y_dim)
    grid_x, grid_y = np.meshgrid(x_e, y_e)
    grid_x, grid_y = grid_x.T, grid_y.T

    ei_energy = np.log10(np.max(np.abs(ei), axis=1) + 1e-9)
    ei_map = griddata(
        channel_positions, ei_energy,
        (grid_x, grid_y),
        method='linear',
        fill_value=np.median(ei_energy),
    )
    return ei_map.T


# ---------------------------------------------------------------------------
# 3-D Mountain plot (rendered lazily — only when tab is visible)
# ---------------------------------------------------------------------------

class EIMountainPlotWidget(QWidget):
    """
    Matplotlib 3-D surface of the EI max-projection.
    plot_ei_3d() is only called when the user switches to the 3D view,
    so it never blocks the UI on cluster selection.
    """

    def __init__(self, main_window: MainWindow, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
        layout.addWidget(self.canvas)

        self.ei_data = None
        self.channel_positions = None
        self._grid_res = 60j

    # ------------------------------------------------------------------
    def restyle(self, colors):
        self.canvas.restyle(colors)
        if self.ei_data is not None:
            self.plot_ei_3d(self.ei_data, self.channel_positions)

    def plot_ei_3d(self, ei_data: np.ndarray, channel_positions: np.ndarray):
        self.ei_data = ei_data
        self.channel_positions = channel_positions
        colors = self.main_window.get_current_colors()

        # Deepest trough per channel, inverted so peaks rise upward
        z_values = np.min(self.ei_data, axis=1) * -1.0

        min_x, max_x = channel_positions[:, 0].min(), channel_positions[:, 0].max()
        min_y, max_y = channel_positions[:, 1].min(), channel_positions[:, 1].max()
        grid_x, grid_y = np.mgrid[min_x:max_x:self._grid_res,
                                   min_y:max_y:self._grid_res]
        grid_z = griddata(channel_positions, z_values, (grid_x, grid_y),
                           method='linear', fill_value=0)
        grid_z = np.nan_to_num(grid_z, nan=0.0)

        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111, projection='3d')
        ax.set_facecolor(colors['bg_panel'])
        self.canvas.fig.patch.set_facecolor(colors['bg_panel'])
        ax.plot_surface(grid_x, grid_y, grid_z,
                        cmap='plasma', linewidth=0, antialiased=False,
                        rcount=grid_z.shape[0], ccount=grid_z.shape[1])
        ax.set_axis_off()
        ax.set_title('EI Max Projection', color=colors['text_primary'])
        self.canvas.draw()

    def clear_plot(self):
        self.canvas.fig.clear()
        self.canvas.draw()
        self.ei_data = None
        self.channel_positions = None


# ---------------------------------------------------------------------------
# Main EI Panel
# ---------------------------------------------------------------------------

class EIPanel(QWidget):
    """
    Spatial / EI analysis panel.

    Views
    -----
    Heatmap  — static max-energy imshow (hot cmap), marker overlay for top channels.
               Frame animation available via Play button / frame slider.
    3D       — Matplotlib surface plot (lazy: rendered only on view switch).

    Right panel — pyqtgraph temporal waveforms for the top N channels.
                  Clicking a channel dot on the spatial canvas promotes that
                  channel to the waveform panel.
    """

    # Animation speed — ms per frame
    _ANIM_INTERVAL_MS = 50

    def __init__(self, main_window: MainWindow, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        # ── state ──────────────────────────────────────────────────────────
        self.current_view: str = "Heatmap"
        self.current_ei_data: list[np.ndarray] | None = None
        self.current_ei_map_list: list[np.ndarray] | None = None
        self.current_cluster_ids: list[int] | None = None
        self.current_channels: np.ndarray | None = None
        self.overlay_index: int = 0
        self.n_frames: int = 0

        # per-cluster ei_map cache: {cluster_id: ei_map_ndarray}
        self._ei_map_cache: dict[int, np.ndarray] = {}

        # animation state
        self._anim_frame: int = 0
        self._anim_playing: bool = False
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(self._ANIM_INTERVAL_MS)
        self._anim_timer.timeout.connect(self._anim_step)

        # ── photo overlay state ────────────────────────────────────────────
        self._overlay_enabled: bool = False
        # (H, W, 4) float32 RGBA pre-loaded from the calibration image
        self._overlay_image_rgba: np.ndarray | None = None
        # (x_left_um, x_right_um, y_bottom_um, y_top_um)
        self._overlay_extent_um: tuple[float, float, float, float] | None = None
        self._overlay_alpha: float = 0.45

        # cell tracer dialog reference (prevent GC)
        self._cell_tracer_dlg: CellTracerDialog | None = None

        # waveform overlay state (used when current_view == "Waveform")
        self._waveform_artists: list = []

        # ── build layout ───────────────────────────────────────────────────
        self._build_ui()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([480, 320])

        main_layout.addWidget(splitter)

    # -- left (spatial) ------------------------------------------------------

    def _build_left_panel(self) -> QWidget:
        left = QWidget()
        layout = QVBoxLayout(left)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # top control bar
        layout.addLayout(self._build_top_controls())

        # stacked widget: index 0 = 2-D canvas, index 1 = 3-D mountain
        self.spatial_stack = QStackedWidget()

        # -- 2-D canvas widget -----------------------------------------------
        canvas_widget = QWidget()
        canvas_layout = QVBoxLayout(canvas_widget)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(2)

        self.spatial_canvas = MplCanvas(self, width=10, height=8, dpi=120)
        self.spatial_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.spatial_canvas.fig.canvas.mpl_connect('motion_notify_event', self._on_canvas_hover)
        self.spatial_canvas.fig.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        canvas_layout.addWidget(self.spatial_canvas)

        # animation controls (live inside 2-D widget)
        canvas_layout.addLayout(self._build_anim_controls())

        # overlay nav (shown only when multiple clusters selected)
        self._overlay_nav = self._build_overlay_nav()
        canvas_layout.addLayout(self._overlay_nav['layout'])

        self.spatial_stack.addWidget(canvas_widget)

        # -- 3-D mountain widget ---------------------------------------------
        self.mountain_widget = EIMountainPlotWidget(self.main_window)
        self.spatial_stack.addWidget(self.mountain_widget)

        layout.addWidget(self.spatial_stack)
        return left

    def _build_top_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)

        row.addWidget(QLabel("View:"))
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Heatmap", "3D", "Waveform"])
        self.view_combo.currentTextChanged.connect(self._on_view_changed)
        row.addWidget(self.view_combo)

        row.addStretch()

        # ── Photo overlay toggle ───────────────────────────────────────────
        self.photo_btn = QPushButton("Photo")
        self.photo_btn.setCheckable(True)
        self.photo_btn.setChecked(False)
        self.photo_btn.setFixedHeight(24)
        self.photo_btn.setToolTip(
            "Overlay the calibrated microscope image behind the EI heatmap.\n"
            "Load a transform first via Array → Map Image to Array…"
        )
        self.photo_btn.clicked.connect(self._on_photo_toggled)
        row.addWidget(self.photo_btn)

        # Opacity slider (range 10–95 → alpha 0.10–0.95)
        self.overlay_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.overlay_alpha_slider.setMinimum(10)
        self.overlay_alpha_slider.setMaximum(95)
        self.overlay_alpha_slider.setValue(int(self._overlay_alpha * 100))
        self.overlay_alpha_slider.setFixedWidth(72)
        self.overlay_alpha_slider.setToolTip("Photo overlay opacity")
        self.overlay_alpha_slider.setEnabled(False)  # enabled only when overlay is on
        self.overlay_alpha_slider.valueChanged.connect(self._on_overlay_alpha_changed)
        row.addWidget(self.overlay_alpha_slider)

        # ── separator ─────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setFixedWidth(2)
        row.addWidget(sep)

        # ── Trace Cell button ──────────────────────────────────────────────
        self.trace_btn = QPushButton("Trace Cell…")
        self.trace_btn.setFixedHeight(24)
        self.trace_btn.setToolTip(
            "Open Cell Tracer: draw a freehand outline over a GFP cell\n"
            "and rank all clusters by EI spatial correlation."
        )
        self.trace_btn.clicked.connect(self._open_cell_tracer)
        row.addWidget(self.trace_btn)

        return row

    def _build_anim_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)

        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedWidth(32)
        self.play_btn.setToolTip("Play / Pause EI animation")
        self.play_btn.clicked.connect(self._toggle_animation)
        row.addWidget(self.play_btn)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._on_frame_slider)
        row.addWidget(self.frame_slider)

        self.frame_label = QLabel("t: 0 ms")
        self.frame_label.setMinimumWidth(70)
        row.addWidget(self.frame_label)

        return row

    def _build_overlay_nav(self) -> dict:
        layout = QHBoxLayout()
        layout.setSpacing(4)

        lbl = QLabel("Overlay:")
        left_btn = QPushButton("◀")
        left_btn.setFixedWidth(28)
        right_btn = QPushButton("▶")
        right_btn.setFixedWidth(28)
        dropdown = QComboBox()

        left_btn.clicked.connect(self._on_overlay_left)
        right_btn.clicked.connect(self._on_overlay_right)
        dropdown.currentIndexChanged.connect(self._on_overlay_dropdown)

        layout.addWidget(lbl)
        layout.addWidget(left_btn)
        layout.addWidget(dropdown)
        layout.addWidget(right_btn)

        # store refs
        self.overlay_left_btn = left_btn
        self.overlay_right_btn = right_btn
        self.overlay_dropdown = dropdown

        # hide by default; shown when n_clusters > 1
        for w in (lbl, left_btn, right_btn, dropdown):
            w.hide()
        self._overlay_nav_widgets = (lbl, left_btn, right_btn, dropdown)

        return {'layout': layout}

    # -- right (temporal) ----------------------------------------------------

    def _build_right_panel(self) -> QWidget:
        right = QWidget()
        layout = QVBoxLayout(right)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.temporal_widget = pg.GraphicsLayoutWidget()
        self.temporal_plot = self.temporal_widget.addPlot()
        layout.addWidget(self.temporal_widget)
        return right

    # -----------------------------------------------------------------------
    # Public API (called by main_window / callbacks — interface unchanged)
    # -----------------------------------------------------------------------

    def update_ei(self, cluster_ids):
        if not self.isVisible():
            return

        cluster_ids = np.atleast_1d(np.array(cluster_ids, dtype=int))
        primary_id = int(cluster_ids[0])

        self._stop_animation()
        dm = self.main_window.data_manager
        vision_ids = np.array([dm.get_vision_id_for_cluster(c) for c in cluster_ids])

        try:
            has_vision = (
                self.main_window.data_manager.vision_eis and
                any(vid in self.main_window.data_manager.vision_eis for vid in vision_ids)
            )

            if has_vision:
                self._load_vision_ei(cluster_ids)
            else:
                lw = self.main_window.data_manager.get_lightweight_features(primary_id)
                hw = self.main_window.data_manager.get_heavyweight_features(primary_id)
                if lw is None or hw is None:
                    self._show_message("Loading spatial features…", color='cyan')
                    if self.main_window.spatial_worker:
                        self.main_window.spatial_worker.add_to_queue(primary_id, high_priority=True)
                    return
                self._load_ks_ei(cluster_ids, is_fallback=True)

        except Exception:
            logger.exception("EI update failed for cluster %s", primary_id)
            self._show_message("Error loading EI data", color='red')

    def restyle_plots(self, colors):
        self.spatial_canvas.restyle(colors)
        self.mountain_widget.restyle(colors)
        self.temporal_widget.setBackground(colors['bg_panel'])
        ax = self.temporal_plot.getAxis
        for side in ('bottom', 'left'):
            self.temporal_plot.getAxis(side).setPen(pg.mkPen(colors['border_default']))
            self.temporal_plot.getAxis(side).setTextPen(pg.mkPen(colors['text_secondary']))
        if self.current_ei_data is not None:
            self.update_ei(self.current_cluster_ids)

    def clear(self):
        self._stop_animation()
        self.spatial_canvas.fig.clear()
        self.spatial_canvas.draw()

    # -----------------------------------------------------------------------
    # Key events (←/→ navigate overlay)
    # -----------------------------------------------------------------------

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self._on_overlay_left()
        elif event.key() == Qt.Key_Right:
            self._on_overlay_right()
        else:
            super().keyPressEvent(event)

    # -----------------------------------------------------------------------
    # Internal: data loading
    # -----------------------------------------------------------------------

    def _load_vision_ei(self, cluster_ids: np.ndarray):
        dm = self.main_window.data_manager
        vision_ids = np.array([dm.get_vision_id_for_cluster(c) for c in cluster_ids])
        valid_ei, valid_orig = [], []

        for i, vid in enumerate(vision_ids):
            if not (isinstance(vid, (int, np.integer)) and 0 < vid < 100_000):
                continue
            entry = self.main_window.data_manager.vision_eis.get(int(vid))
            if entry and hasattr(entry, 'ei') and entry.ei is not None and entry.ei.ndim == 2:
                valid_ei.append(entry.ei)
                valid_orig.append(int(cluster_ids[i]))

        if not valid_ei:
            self.clear()
            return

        ch_pos = self._resolve_channel_positions()
        ei_maps, final_ids, final_ei = [], [], []

        for ei_data, orig_id in zip(valid_ei, valid_orig):
            # use cached ei_map when available
            cached = self._ei_map_cache.get(orig_id)
            if cached is not None:
                ei_map = cached
            else:
                ei_map = compute_ei_map(ei_data, ch_pos)
                if ei_map is not None:
                    self._ei_map_cache[orig_id] = ei_map
            if ei_map is not None:
                ei_maps.append(ei_map)
                final_ids.append(orig_id)
                final_ei.append(ei_data)

        if not ei_maps:
            self.clear()
            return

        self.current_ei_map_list = ei_maps
        self.current_cluster_ids = final_ids
        self.current_ei_data = final_ei
        self.n_frames = final_ei[0].shape[1]
        self.overlay_index = 0

        top_ch = self._get_top_electrodes(final_ei[0], n_interval=2, n_markers=3, b_sort=True)
        self.current_channels = top_ch

        self._setup_anim_controls()
        self._update_overlay_nav()
        self._draw_heatmap_frame(self._anim_frame)
        self._draw_temporal(final_ei, final_ids, top_ch)

        # lazy 3-D: only render if that view is currently active
        if self.current_view == "3D":
            self.mountain_widget.plot_ei_3d(final_ei[0], ch_pos)
        elif self.current_view == "Waveform":
            self._draw_waveform_frame()

    def _load_ks_ei(self, cluster_ids: np.ndarray, is_fallback=False):
        primary_id = int(cluster_ids[0])
        lw = self.main_window.data_manager.get_lightweight_features(primary_id)
        hw = self.main_window.data_manager.get_heavyweight_features(primary_id)

        if lw is None or hw is None:
            self._show_message("Error generating features", color='red')
            return

        from .population_panel import plot_rich_ei

        self.spatial_canvas.fig.clear()
        colors = self.main_window.get_current_colors()
        self.spatial_canvas.fig.patch.set_facecolor(colors['bg_panel'])

        plot_rich_ei(
            self.spatial_canvas.fig,
            lw['median_ei'],
            self.main_window.data_manager.channel_positions,
            hw,
            self.main_window.data_manager.sampling_rate,
            _pre_samples=20,
        )
        title = f"Cluster {primary_id}"
        if is_fallback:
            title += " (KS EI)"
        self.spatial_canvas.fig.suptitle(title,
                                          color=colors['text_primary'], fontsize=14)
        self.spatial_canvas.draw()

        # store minimal state so animation / 3-D still work
        ei_data = lw['median_ei']
        ch_pos = self.main_window.data_manager.channel_positions
        self.current_ei_data = [ei_data]
        self.current_cluster_ids = [primary_id]
        self.n_frames = ei_data.shape[1]

        top_ch = self._get_top_electrodes(ei_data, n_interval=2, n_markers=3, b_sort=True)
        self.current_channels = top_ch
        self._setup_anim_controls()
        self._draw_temporal([ei_data], [primary_id], top_ch)

        if self.current_view == "3D":
            self.mountain_widget.plot_ei_3d(ei_data, ch_pos)
        elif self.current_view == "Waveform":
            self._draw_waveform_frame()

    # -----------------------------------------------------------------------
    # Internal: drawing
    # -----------------------------------------------------------------------

    def _draw_heatmap_frame(self, frame: int):
        """
        Render one time frame of the EI on the spatial canvas (imshow).
        Frame -1 means max-projection (used for static heatmap).
        """
        if not self.current_ei_data or not self.current_ei_map_list:
            return

        colors = self.main_window.get_current_colors()
        ch_pos = self._resolve_channel_positions()
        ei_data = self.current_ei_data[self.overlay_index]
        cluster_id = self.current_cluster_ids[self.overlay_index]

        if frame < 0 or frame >= self.n_frames:
            # static max-energy map
            img = self.current_ei_map_list[self.overlay_index]
            title = f"EI — cluster {cluster_id}"
        else:
            # live voltage frame interpolated to the same grid as ei_map
            xrange = (ch_pos[:, 0].min(), ch_pos[:, 0].max())
            yrange = (ch_pos[:, 1].min(), ch_pos[:, 1].max())
            y_dim = 30
            x_dim = max(1, int((xrange[1] - xrange[0]) / max(yrange[1] - yrange[0], 1) * y_dim))
            gx = np.linspace(xrange[0], xrange[1], x_dim)
            gy = np.linspace(yrange[0], yrange[1], y_dim)
            mgx, mgy = np.meshgrid(gx, gy)
            mgx, mgy = mgx.T, mgy.T
            voltage_frame = ei_data[:, frame]
            img = griddata(ch_pos, voltage_frame, (mgx, mgy),
                           method='linear', fill_value=0.0).T
            t_ms = frame / self.main_window.data_manager.sampling_rate * 1000.0
            title = f"EI — cluster {cluster_id}  |  t = {t_ms:.2f} ms"

        self.spatial_canvas.fig.clear()
        ax = self.spatial_canvas.fig.add_subplot(111)
        ax.set_facecolor(colors['bg_panel'])
        self.spatial_canvas.fig.patch.set_facecolor(colors['bg_panel'])

        # ── photo underlay ─────────────────────────────────────────────────
        if (self._overlay_enabled
                and self._overlay_image_rgba is not None
                and self._overlay_extent_um is not None):
            xl, xr_img, yb, yt = self._overlay_extent_um
            ax.imshow(
                self._overlay_image_rgba,
                aspect='auto',
                origin='upper',             # pixel row-0 maps to top of extent rect
                extent=(xl, xr_img, yb, yt),  # (left, right, bottom, top) in microns
                alpha=self._overlay_alpha,
                interpolation='bilinear',
                zorder=0,
            )
        # ── end photo underlay ─────────────────────────────────────────────

        xr = (ch_pos[:, 0].min(), ch_pos[:, 0].max())
        yr = (ch_pos[:, 1].min(), ch_pos[:, 1].max())
        cmap = 'bwr' if frame >= 0 else 'hot'
        ax.imshow(img, cmap=cmap, aspect='auto', origin='lower',
                  extent=(xr[0], xr[1], yr[0], yr[1]),
                  alpha=0.72 if self._overlay_enabled else 1.0,
                  zorder=1)
        ax.axis('off')

        # top-channel markers
        if self.current_channels is not None:
            for j, ch in enumerate(self.current_channels):
                x, y = ch_pos[ch]
                ax.plot(x, y, 'o', markersize=4, markerfacecolor='none',
                        markeredgecolor='cyan', markeredgewidth=1.5,
                        zorder=2)
                ax.text(x, y, str(j), color='cyan', fontsize=6,
                        ha='center', va='center', zorder=2)

        self.spatial_canvas.fig.suptitle(title, color=colors['text_primary'], fontsize=13)
        self.spatial_canvas.fig.tight_layout()
        self.spatial_canvas.draw()

    def _draw_temporal(self, ei_data_list, cluster_ids, channels):
        """Render one waveform row per top channel, one trace per cluster."""
        self.temporal_widget.clear()
        colors = self.main_window.get_current_colors()
        sr = self.main_window.data_manager.sampling_rate

        for i, ch in enumerate(channels):
            pi = pg.PlotItem()
            for side in ('bottom', 'left'):
                pi.getAxis(side).setPen(pg.mkPen(colors['border_default']))
                pi.getAxis(side).setTextPen(pg.mkPen(colors['text_secondary']))
            self.temporal_widget.addItem(pi, i, 0)

            for j, ei in enumerate(ei_data_list):
                t_ms = np.arange(ei.shape[1]) / sr * 1000.0
                color = pg.intColor(j, hues=max(len(cluster_ids), 1))
                pi.plot(t_ms, ei[ch, :],
                        pen=pg.mkPen(color=color, width=2),
                        name=f'C{cluster_ids[j]}')

            pi.setLabel('left', f'ch {ch}')
            pi.setLabel('bottom', 'Time (ms)')
            if len(cluster_ids) > 1:
                pi.addLegend()

    # -----------------------------------------------------------------------
    # Animation
    # -----------------------------------------------------------------------

    def _setup_anim_controls(self):
        """Enable slider based on current n_frames."""
        if self.n_frames > 0:
            self.frame_slider.setMaximum(self.n_frames - 1)
            self.frame_slider.setValue(0)
            self.frame_slider.setEnabled(True)
        else:
            self.frame_slider.setEnabled(False)
        self._anim_frame = 0
        self.play_btn.setText("▶")

    def _toggle_animation(self):
        if self._anim_playing:
            self._stop_animation()
        else:
            self._start_animation()

    def _start_animation(self):
        if self.n_frames < 2:
            return
        self._anim_playing = True
        self.play_btn.setText("⏸")
        self._anim_timer.start()

    def _stop_animation(self):
        self._anim_playing = False
        self._anim_timer.stop()
        self.play_btn.setText("▶")

    def _anim_step(self):
        self._anim_frame = (self._anim_frame + 1) % self.n_frames
        # blockSignals so we don't re-trigger _on_frame_slider redundantly
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self._anim_frame)
        self.frame_slider.blockSignals(False)
        self._render_anim_frame()

    def _on_frame_slider(self, value: int):
        self._anim_frame = value
        sr = self.main_window.data_manager.sampling_rate if self.main_window.data_manager else 20000
        t_ms = value / sr * 1000.0
        self.frame_label.setText(f"t: {t_ms:.2f} ms")
        self._render_anim_frame()

    def _render_anim_frame(self):
        if self.current_view in ("Heatmap", "Waveform"):
            self._draw_heatmap_frame(self._anim_frame)
            sr = self.main_window.data_manager.sampling_rate if self.main_window.data_manager else 20000
            t_ms = self._anim_frame / sr * 1000.0
            self.frame_label.setText(f"t: {t_ms:.2f} ms")

    # -----------------------------------------------------------------------
    # Canvas interaction
    # -----------------------------------------------------------------------

    def _on_canvas_hover(self, event):
        if event.inaxes is None or self.main_window.data_manager is None:
            return
        ch_pos = self._resolve_channel_positions()
        mouse = np.array([[event.xdata, event.ydata]])
        dists = np.linalg.norm(ch_pos - mouse, axis=1)
        nearest = dists.argmin()
        if dists[nearest] < 20:
            ei_amp = ""
            if self.current_ei_data:
                amp = np.abs(self.current_ei_data[0][nearest, :]).max()
                ei_amp = f"  |  amp {amp:.1f} µV"
            self.main_window.status_bar.showMessage(
                f"Channel {nearest}{ei_amp}")
        else:
            self.main_window.status_bar.clearMessage()

    def _on_canvas_click(self, event):
        """Click on spatial map → promote nearest electrode to temporal panel."""
        if event.inaxes is None or self.current_ei_data is None:
            return
        if event.button != 1:
            return
        ch_pos = self._resolve_channel_positions()
        mouse = np.array([[event.xdata, event.ydata]])
        dists = np.linalg.norm(ch_pos - mouse, axis=1)
        nearest = int(dists.argmin())
        if dists[nearest] > 40:
            return

        # Rebuild channel list with clicked channel at index 0
        current = list(self.current_channels) if self.current_channels is not None else []
        if nearest in current:
            current.remove(nearest)
        new_channels = np.array([nearest] + current[:2], dtype=int)
        self.current_channels = new_channels
        self._draw_heatmap_frame(self._anim_frame)
        self._draw_temporal(self.current_ei_data, self.current_cluster_ids, new_channels)

    # -----------------------------------------------------------------------
    # View switching
    # -----------------------------------------------------------------------

    def _on_view_changed(self, text: str):
        self.current_view = text
        if text == "Heatmap":
            self.spatial_stack.setCurrentIndex(0)
            if self.current_ei_data is not None:
                self._draw_heatmap_frame(self._anim_frame)
        elif text == "3D":
            self.spatial_stack.setCurrentIndex(1)
            if self.current_ei_data is not None:
                ch_pos = self._resolve_channel_positions()
                self.mountain_widget.plot_ei_3d(self.current_ei_data[0], ch_pos)
        elif text == "Waveform":
            self.spatial_stack.setCurrentIndex(0)   # reuse the 2-D canvas
            if self.current_ei_data is not None:
                self._draw_waveform_frame()

    # -----------------------------------------------------------------------
    # Waveform view
    # -----------------------------------------------------------------------

    def _clear_waveform_artists(self):
        for artist in self._waveform_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._waveform_artists = []

    def _draw_waveform_frame(self):
        """
        Paint EI waveforms for the currently selected cluster(s) directly
        onto spatial_canvas in micron space, co-registered with the photo
        underlay.  Uses plot_ei_waveforms from analysis_core — the same
        primitive as CellTracerDialog.

        Called when:
          • User switches to Waveform view (via _on_view_changed)
          • A new cluster loads while Waveform is active (_load_vision_ei,
            _load_ks_ei)
        Animation scrubbing stays on _draw_heatmap_frame (the interpolated
        voltage grid) — switching back to Heatmap to animate is intentional
        since per-frame waveform re-renders of 519 traces would be too slow.
        """
        if not self.current_ei_data or not self.current_cluster_ids:
            return

        from ...analysis.analysis_core import plot_ei_waveforms

        ch_pos = self._resolve_channel_positions()
        if ch_pos is None or len(ch_pos) == 0:
            return

        colors_ui = self.main_window.get_current_colors()
        cluster_colors = ['#00e5ff', '#ff9800', '#69ff47', '#ff4081']

        # Full redraw of the canvas so photo underlay and axis limits are set
        # correctly, then paint waveforms on top without clearing again.
        self.spatial_canvas.fig.clear()
        ax = self.spatial_canvas.fig.add_subplot(111)
        ax.set_facecolor(colors_ui['bg_panel'])
        self.spatial_canvas.fig.patch.set_facecolor(colors_ui['bg_panel'])

        # ── photo underlay (same logic as _draw_heatmap_frame) ─────────────
        if (self._overlay_enabled
                and self._overlay_image_rgba is not None
                and self._overlay_extent_um is not None):
            xl, xr_img, yb, yt = self._overlay_extent_um
            ax.imshow(
                self._overlay_image_rgba,
                aspect='auto',
                origin='upper',
                extent=(xl, xr_img, yb, yt),
                alpha=self._overlay_alpha,
                interpolation='bilinear',
                zorder=0,
            )

        # ── set axis limits from electrode positions ────────────────────────
        pad = 50.0
        ax.set_xlim(ch_pos[:, 0].min() - pad, ch_pos[:, 0].max() + pad)
        ax.set_ylim(ch_pos[:, 1].min() - pad, ch_pos[:, 1].max() + pad)
        ax.axis('off')

        # ── electrode scatter (dim, for spatial reference) ──────────────────
        ax.scatter(ch_pos[:, 0], ch_pos[:, 1],
                   s=2, c='#444444', zorder=2, rasterized=True)

        # ── waveforms for each loaded cluster ───────────────────────────────
        self._waveform_artists = []
        for i, (ei_arr, cid) in enumerate(
                zip(self.current_ei_data, self.current_cluster_ids)):
            color = cluster_colors[i % len(cluster_colors)]
            n_ch = min(ei_arr.shape[0], len(ch_pos))
            artists = plot_ei_waveforms(
                ei_arr[:n_ch],
                ch_pos[:n_ch],
                ref_channel=int(self.current_channels[0])
                            if self.current_channels is not None else None,
                ax=ax,
                colors=color,
                alpha=0.80,
                linewidth=0.6,
            )
            self._waveform_artists.extend(artists)

        cluster_str = ", ".join(str(c) for c in self.current_cluster_ids)
        self.spatial_canvas.fig.suptitle(
            f"EI waveforms — cluster{'s' if len(self.current_cluster_ids) > 1 else ''} "
            f"{cluster_str}",
            color=colors_ui['text_primary'], fontsize=13,
        )
        self.spatial_canvas.fig.tight_layout()
        self.spatial_canvas.draw()

    # -----------------------------------------------------------------------
    # Overlay nav (multiple clusters)
    # -----------------------------------------------------------------------

    def _update_overlay_nav(self):
        n = len(self.current_cluster_ids) if self.current_cluster_ids else 0
        show = n > 1
        for w in self._overlay_nav_widgets:
            w.setVisible(show)
        if show:
            self.overlay_dropdown.blockSignals(True)
            self.overlay_dropdown.clear()
            for cid in self.current_cluster_ids:
                self.overlay_dropdown.addItem(str(cid))
            self.overlay_dropdown.setCurrentIndex(
                np.clip(self.overlay_index, 0, n - 1))
            self.overlay_dropdown.blockSignals(False)

    def _on_overlay_dropdown(self, idx: int):
        self.overlay_index = idx
        self._draw_heatmap_frame(self._anim_frame)

    def _on_overlay_left(self):
        if self.overlay_index > 0:
            self.overlay_index -= 1
            self.overlay_dropdown.setCurrentIndex(self.overlay_index)

    def _on_overlay_right(self):
        if self.overlay_index < self.overlay_dropdown.count() - 1:
            self.overlay_index += 1
            self.overlay_dropdown.setCurrentIndex(self.overlay_index)

    # -----------------------------------------------------------------------
    # Photo overlay — public API
    # -----------------------------------------------------------------------

    def refresh_array_image(self, transform_path: str) -> None:
        """
        Pre-load the calibrated microscope image and compute its micron-space
        extent from the transform JSON.  Called by:
          • MainWindow._on_transform_saved()  — after user saves a new transform
          • MainWindow post-load hook          — auto-detected existing transform
          • _try_autoload_transform()          — lazy first-click fallback
        """
        import json
        from pathlib import Path
        from PIL import Image

        self._overlay_image_rgba = None
        self._overlay_extent_um  = None

        try:
            with open(transform_path, 'r') as fh:
                data = json.load(fh)

            img_name = data.get('image_file')
            if not img_name:
                logger.warning("array_transform.json has no image_file field")
                return

            img_path = Path(transform_path).parent / img_name
            if not img_path.exists():
                logger.warning("Array image not found: %s", img_path)
                return

            # Load as RGBA float32 so matplotlib can composite cleanly
            img = Image.open(img_path).convert('RGBA')
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_h, img_w = img_array.shape[:2]

            # Calibration: pixel = scale * micron + offset
            # Invert:      micron = (pixel - offset) / scale
            sx = float(data.get('scale_x', 1.0))
            sy = float(data.get('scale_y', 1.0))
            ox = float(data.get('offset_x', 0.0))
            oy = float(data.get('offset_y', 0.0))

            x_left_um  = (0     - ox) / sx
            x_right_um = (img_w - ox) / sx
            y_at_row0  = (0     - oy) / sy   # micron-Y when pixel row = 0 (top)
            y_at_rowH  = (img_h - oy) / sy   # micron-Y when pixel row = H (bottom)

            # origin='upper' → row-0 sits at the TOP of the bounding box.
            # extent tuple = (left, right, bottom, top)
            bottom_um = min(y_at_row0, y_at_rowH)
            top_um    = max(y_at_row0, y_at_rowH)

            self._overlay_image_rgba = img_array
            self._overlay_extent_um  = (x_left_um, x_right_um, bottom_um, top_um)

            if hasattr(self, 'photo_btn'):
                self.photo_btn.setToolTip(
                    f"Photo: {img_path.name}\n"
                    f"Extent  X {x_left_um:.0f}–{x_right_um:.0f} µm  "
                    f"Y {bottom_um:.0f}–{top_um:.0f} µm"
                )

            logger.info(
                "EI photo overlay ready: %s  (%.0f–%.0f µm, %.0f–%.0f µm)",
                img_path.name, x_left_um, x_right_um, bottom_um, top_um,
            )

        except Exception:
            logger.exception("Failed to load array image for EI panel overlay")
            self._overlay_image_rgba = None
            self._overlay_extent_um  = None

    # -----------------------------------------------------------------------
    # Photo overlay — slot handlers
    # -----------------------------------------------------------------------

    def _on_photo_toggled(self, checked: bool) -> None:
        if checked and self._overlay_image_rgba is None:
            self._try_autoload_transform()

        if self._overlay_image_rgba is None:
            self.photo_btn.setChecked(False)
            if hasattr(self.main_window, 'status_bar'):
                self.main_window.status_bar.showMessage(
                    "No array image loaded — use Array → Map Image to Array…", 4000)
            return

        self._overlay_enabled = checked
        self.overlay_alpha_slider.setEnabled(checked)

        if self.current_ei_data is not None:
            self._draw_heatmap_frame(self._anim_frame)

    def _on_overlay_alpha_changed(self, value: int) -> None:
        self._overlay_alpha = value / 100.0
        if self._overlay_enabled and self.current_ei_data is not None:
            self._draw_heatmap_frame(self._anim_frame)

    def _try_autoload_transform(self) -> None:
        dm = self.main_window.data_manager
        if dm is None:
            return
        from pathlib import Path
        candidate = (
            Path(dm.kilosort_dir).parent / 'transforms' / 'array_transform.json'
        )
        if candidate.exists():
            self.refresh_array_image(str(candidate))
        else:
            logger.debug("No default transform found at %s", candidate)

    # -----------------------------------------------------------------------
    # Cell Tracer
    # -----------------------------------------------------------------------

    def _open_cell_tracer(self) -> None:
        """Open the Cell Tracer dialog."""
        dlg = CellTracerDialog(parent=self.main_window, ei_panel=self)
        dlg.cluster_selected.connect(self._on_tracer_cluster_selected)
        dlg.show()
        self._cell_tracer_dlg = dlg   # prevent GC

    def _on_tracer_cluster_selected(self, cid: int) -> None:
        """Jump to a cluster chosen from the Cell Tracer results table."""
        mw = self.main_window
        if hasattr(mw, '_select_cluster_by_id'):
            mw._select_cluster_by_id(cid)
        elif hasattr(mw, 'select_cluster'):
            mw.select_cluster(cid)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _resolve_channel_positions(self) -> np.ndarray:
        """Single source of truth for channel positions."""
        dm = self.main_window.data_manager
        if dm is None:
            return np.zeros((0, 2))
        if dm.vision_channel_positions is not None:
            return dm.vision_channel_positions
        return dm.channel_positions

    def _get_top_electrodes(self,
                            ei: np.ndarray,
                            n_interval: int = 2,
                            n_markers: int = 5,
                            b_sort: bool = True) -> np.ndarray:
        """Return indices of the top-N electrodes by log-amplitude, optionally sorted by peak latency."""
        ei_map = np.log10(np.max(np.abs(ei), axis=1) + 1e-6)
        top_idx = np.argsort(ei_map.flatten())[::-1][::n_interval][:n_markers]

        if b_sort and len(top_idx) > 0:
            peak_times = np.argmin(ei[top_idx, :], axis=1)
            top_idx = top_idx[np.argsort(peak_times)]

        return top_idx

    def _show_message(self, msg: str, color: str = 'cyan'):
        self.clear()
        self.spatial_canvas.fig.text(
            0.5, 0.5, msg, ha='center', va='center',
            color=color, fontsize=14,
        )
        self.spatial_canvas.draw()