from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_rgba
import matplotlib.cm as mpl_cm

try:
    # matplotlib >= 3.7: mpl_cm.get_cmap is deprecated
    from matplotlib import colormaps as _mpl_colormaps

    def _get_cmap(name):
        return _mpl_colormaps[name]
except Exception:  # older matplotlib
    def _get_cmap(name):
        return mpl_cm.get_cmap(name)

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSplitter,
    QSizePolicy,
    QComboBox,
    QStackedWidget,
    QSlider,
    QFrame,
)
from qtpy.QtCore import Qt, QTimer


class _ComboBoxNoWheel(QComboBox):
    """Ignore wheel events unless the popup is open.

    A QComboBox under the cursor steals wheel events from the canvas and
    changes the current item. That made the EI View combo jump while the
    user scrolled the plot.
    """

    def wheelEvent(self, event):
        if self.view().isVisible():
            super().wheelEvent(event)
        else:
            event.ignore()
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
            ei.shape[0],
            channel_positions.shape[0],
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
        channel_positions,
        ei_energy,
        (grid_x, grid_y),
        method="linear",
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
        grid_x, grid_y = np.mgrid[
            min_x : max_x : self._grid_res, min_y : max_y : self._grid_res
        ]
        grid_z = griddata(
            channel_positions, z_values, (grid_x, grid_y), method="linear", fill_value=0
        )
        grid_z = np.nan_to_num(grid_z, nan=0.0)

        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111, projection="3d")
        ax.set_facecolor(colors["bg_panel"])
        self.canvas.fig.patch.set_facecolor(colors["bg_panel"])
        ax.plot_surface(
            grid_x,
            grid_y,
            grid_z,
            cmap="plasma",
            linewidth=0,
            antialiased=False,
            rcount=grid_z.shape[0],
            ccount=grid_z.shape[1],
        )
        ax.set_axis_off()
        ax.set_title("EI Max Projection", color=colors["text_primary"])
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
        self.current_ei_error: list | None = None
        self.current_ei_map_list: list[np.ndarray] | None = None
        self.current_cluster_ids: list[int] | None = None
        self.current_channels: np.ndarray | None = None
        self.overlay_index: int = 0
        self.n_frames: int = 0
        # sample index of the soma spike (Vision nl_points, or global trough)
        self._soma_frame: int = 0

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
        self.spatial_canvas.fig.canvas.mpl_connect(
            "motion_notify_event", self._on_canvas_hover
        )
        self.spatial_canvas.fig.canvas.mpl_connect(
            "button_press_event", self._on_canvas_click
        )
        canvas_layout.addWidget(self.spatial_canvas)

        # animation controls (live inside 2-D widget)
        canvas_layout.addLayout(self._build_anim_controls())

        # overlay nav (shown only when multiple clusters selected)
        self._overlay_nav = self._build_overlay_nav()
        canvas_layout.addLayout(self._overlay_nav["layout"])

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
        self.view_combo = _ComboBoxNoWheel()
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
        dropdown = _ComboBoxNoWheel()

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

        return {"layout": layout}

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

        # array geometry can change between datasets; recompute pitch lazily
        self._electrode_pitch = None
        # amplitude scale is per-selection; recompute on next draw
        self._global_ei_scale = None

        self._stop_animation()
        dm = self.main_window.data_manager
        vision_ids = np.array([dm.get_vision_id_for_cluster(c) for c in cluster_ids])

        try:
            has_vision = self.main_window.data_manager.vision_eis and any(
                vid in self.main_window.data_manager.vision_eis for vid in vision_ids
            )

            if has_vision:
                self._load_vision_ei(cluster_ids)
            else:
                lw = self.main_window.data_manager.get_lightweight_features(primary_id)
                hw = self.main_window.data_manager.get_heavyweight_features(primary_id)
                if lw is None or hw is None:
                    self._show_message("Loading spatial features…", color="cyan")
                    if self.main_window.spatial_worker:
                        self.main_window.spatial_worker.add_to_queue(
                            primary_id, high_priority=True
                        )
                    return
                self._load_ks_ei(cluster_ids, is_fallback=True)

        except Exception:
            logger.exception("EI update failed for cluster %s", primary_id)
            self._show_message("Error loading EI data", color="red")

    def restyle_plots(self, colors):
        self.spatial_canvas.restyle(colors)
        self.mountain_widget.restyle(colors)
        self.temporal_widget.setBackground(colors["bg_panel"])
        self.temporal_plot.getAxis
        for side in ("bottom", "left"):
            self.temporal_plot.getAxis(side).setPen(pg.mkPen(colors["border_default"]))
            self.temporal_plot.getAxis(side).setTextPen(
                pg.mkPen(colors["text_secondary"])
            )
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
        valid_ei, valid_orig, valid_err = [], [], []
        soma_frame = None

        for i, vid in enumerate(vision_ids):
            if not (isinstance(vid, (int, np.integer)) and 0 < vid < 100_000):
                continue
            entry = self.main_window.data_manager.vision_eis.get(int(vid))
            if (
                entry
                and hasattr(entry, "ei")
                and entry.ei is not None
                and entry.ei.ndim == 2
            ):
                valid_ei.append(entry.ei)
                valid_orig.append(int(cluster_ids[i]))
                # Vision computes a per-sample EI error (SD across spikes);
                # capture it so the footprint view can gate real-signal
                # channels against Vision's own error rather than a guessed
                # noise floor. May be absent on some containers.
                err = getattr(entry, "ei_error", None)
                valid_err.append(err if (err is not None and getattr(err, "ndim", 0) == 2) else None)
                # Vision aligns the spike at sample index nl_points (nl left
                # points, then the trough, then nr right points). Use the
                # primary cluster's alignment as the soma frame for both the
                # default heatmap frame and the animation window.
                if soma_frame is None and hasattr(entry, "nl_points"):
                    try:
                        soma_frame = int(entry.nl_points)
                    except Exception:
                        soma_frame = None

        if not valid_ei:
            self.clear()
            return

        ch_pos = self._resolve_channel_positions()
        ei_maps, final_ids, final_ei, final_err = [], [], [], []

        for ei_data, orig_id, err_data in zip(valid_ei, valid_orig, valid_err):
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
                final_err.append(err_data)

        if not ei_maps:
            self.clear()
            return

        self.current_ei_map_list = ei_maps
        self.current_cluster_ids = final_ids
        self.current_ei_data = final_ei
        self.current_ei_error = final_err
        self.n_frames = final_ei[0].shape[1]
        self.overlay_index = 0

        # Resolve soma frame: prefer Vision nl_points; else fall back to the
        # global trough (deepest negative sample across all channels), which
        # is where the soma spike sits for a baseline-referenced EI.
        if soma_frame is None or not (0 <= soma_frame < self.n_frames):
            soma_frame = int(np.argmin(final_ei[0].min(axis=0)))
        self._soma_frame = soma_frame

        top_ch = self._get_top_electrodes(
            final_ei[0], n_interval=2, n_markers=3, b_sort=True
        )
        self.current_channels = top_ch

        self._setup_anim_controls()
        self._update_overlay_nav()
        # Default the heatmap to MAX-PROJECTION (frame=-1): every electrode the
        # cell ever drives — soma + full axon path — shown at once, so the
        # whole footprint is visible on load. Pressing Play still animates
        # through real time frames (starting at the soma spike). Keep
        # `_anim_frame` at -1 so a View-combo round-trip does not snap to the
        # soma sample that `_setup_anim_controls` parked the slider on.
        self._anim_frame = -1
        self._redraw_current_view()
        self._draw_temporal(final_ei, final_ids, top_ch)

    def _load_ks_ei(self, cluster_ids: np.ndarray, is_fallback=False):
        primary_id = int(cluster_ids[0])
        lw = self.main_window.data_manager.get_lightweight_features(primary_id)
        hw = self.main_window.data_manager.get_heavyweight_features(primary_id)

        if lw is None or hw is None:
            self._show_message("Error generating features", color="red")
            return

        # store minimal state so animation / 3-D / waveform still work
        ei_data = lw["median_ei"]
        ch_pos = self.main_window.data_manager.channel_positions
        self.current_ei_data = [ei_data]
        self.current_ei_error = [None]
        self.current_cluster_ids = [primary_id]
        self.n_frames = ei_data.shape[1]
        # KS EIs use window=(-20, 60): trough ~sample 20. Derive robustly
        # from the global minimum so it tracks whatever alignment is present.
        self._soma_frame = int(np.argmin(ei_data.min(axis=0)))
        self.current_ei_map_list = None

        top_ch = self._get_top_electrodes(
            ei_data, n_interval=2, n_markers=3, b_sort=True
        )
        self.current_channels = top_ch
        self._setup_anim_controls()
        self._draw_temporal([ei_data], [primary_id], top_ch)

        if self.current_view == "3D":
            self.mountain_widget.plot_ei_3d(ei_data, ch_pos)
        elif self.current_view == "Waveform":
            self._draw_waveform_frame()
        else:
            from .population_panel import plot_rich_ei

            self.spatial_canvas.fig.clear()
            colors = self.main_window.get_current_colors()
            self.spatial_canvas.fig.patch.set_facecolor(colors["bg_panel"])

            plot_rich_ei(
                self.spatial_canvas.fig,
                lw["median_ei"],
                ch_pos,
                hw,
                self.main_window.data_manager.sampling_rate,
                _pre_samples=20,
            )
            title = f"Cluster {primary_id}"
            if is_fallback:
                title += " (KS EI)"
            self.spatial_canvas.fig.suptitle(
                title, color=colors["text_primary"], fontsize=14
            )
            self.spatial_canvas.draw()

    # -----------------------------------------------------------------------
    # Internal: drawing
    # -----------------------------------------------------------------------

    # cached electrode pitch so we don't recompute per frame during animation
    _electrode_pitch: float | None = None

    def _get_electrode_pitch(self, ch_pos: np.ndarray) -> float:
        """Median nearest-neighbour spacing of the array (µm). Cached."""
        if self._electrode_pitch is not None:
            return self._electrode_pitch
        pitch = 30.0
        try:
            if len(ch_pos) >= 2:
                tree = cKDTree(ch_pos)
                d, _ = tree.query(ch_pos, k=2)
                nn = d[:, 1]
                nn = nn[np.isfinite(nn) & (nn > 0)]
                if nn.size:
                    pitch = float(np.median(nn))
        except Exception:
            pass
        self._electrode_pitch = pitch
        return pitch

    def _draw_heatmap_frame(self, frame: int):
        """
        Render one time frame of the EI as per-electrode "bubbles": each
        electrode is a soft, semi-transparent disc whose *radius* scales with
        signal amplitude and whose *colour* encodes polarity (live frame) or
        energy (max-projection). Unlike the old griddata→imshow rectangle,
        the bubbles leave the gaps between electrodes empty, so the photo
        underlay reads through and true array geometry (hex pitch) is visible.

        Frame < 0 → max-projection (per-channel peak |amplitude|, sequential
        colormap). Frame ≥ 0 → signed voltage at that time sample (diverging
        colormap, size ∝ |voltage|).
        """
        if not self.current_ei_data:
            return

        colors = self.main_window.get_current_colors()
        ch_pos = self._resolve_channel_positions()
        if ch_pos is None or len(ch_pos) == 0:
            return
        ei_data = self.current_ei_data[self.overlay_index]
        cluster_id = self.current_cluster_ids[self.overlay_index]
        n_ch = min(ei_data.shape[0], len(ch_pos))
        pos = ch_pos[:n_ch]

        pitch = self._get_electrode_pitch(ch_pos)

        if frame < 0 or frame >= self.n_frames:
            # per-channel peak amplitude (always positive) → energy view
            values = np.max(np.abs(ei_data[:n_ch]), axis=1)
            signed = False
            title = f"EI — cluster {cluster_id}"
        else:
            values = ei_data[:n_ch, frame].astype(float)
            signed = True
            t_ms = frame / self.main_window.data_manager.sampling_rate * 1000.0
            title = f"EI — cluster {cluster_id}  |  t = {t_ms:.2f} ms"

        # amplitude used for bubble size — always magnitude
        mag = np.abs(values).astype(float)

        # GLOBAL spike scale: fixed across all frames so the animation is
        # honest — a baseline (pre-spike) frame renders tiny/quiet and the
        # soma-trough frame renders large/saturated. Using a per-frame
        # percentile (the old bug) renormalised every frame to its own noise,
        # so baseline looked identical to the spike. Scale is the 99.5th pct
        # of |amplitude| at the soma frame, cached per selection.
        ceil = getattr(self, "_global_ei_scale", None)
        if ceil is None:
            sf = int(np.clip(self._soma_frame, 0, self.n_frames - 1))
            soma_mag = np.abs(ei_data[:n_ch, sf])
            ceil = float(np.percentile(soma_mag, 99.5)) if soma_mag.size else 1.0
            if ceil <= 0:
                ceil = float(np.max(np.abs(ei_data[:n_ch]))) or 1.0
            self._global_ei_scale = ceil
        norm_mag = np.clip(mag / ceil, 0.0, 1.0)

        self.spatial_canvas.fig.clear()
        ax = self.spatial_canvas.fig.add_subplot(111)
        ax.set_facecolor(colors["bg_panel"])
        self.spatial_canvas.fig.patch.set_facecolor(colors["bg_panel"])

        # ── photo underlay ─────────────────────────────────────────────────
        if (
            self._overlay_enabled
            and self._overlay_image_rgba is not None
            and self._overlay_extent_um is not None
        ):
            xl, xr_img, yb, yt = self._overlay_extent_um
            ax.imshow(
                self._overlay_image_rgba,
                aspect="auto",
                origin="upper",
                extent=(xl, xr_img, yb, yt),
                alpha=self._overlay_alpha,
                interpolation="bilinear",
                zorder=0,
            )
        # ── end photo underlay ─────────────────────────────────────────────

        # colour mapping
        if signed:
            vmax = ceil
            cnorm = Normalize(vmin=-vmax, vmax=vmax)
            cmap = _get_cmap("RdBu_r")
            face_rgba = cmap(cnorm(values))
        else:
            cnorm = Normalize(vmin=0.0, vmax=ceil)
            cmap = _get_cmap("inferno")
            face_rgba = cmap(cnorm(mag))

        # Noise gate: below a fraction of the spike scale a channel is
        # baseline noise, not signal. Clamp those to zero size so the
        # animation isn't a field of jittering mid-size bubbles.
        #
        # In MAX-PROJECTION mode (frame < 0) use the SAME rule as the waveform
        # view — per-channel peak over all time vs the global peak, at
        # MAXPROJ_FRAC=0.06 — so both views show the identical footprint
        # including the low-amplitude axon. In animated single-frame mode keep
        # the soma-frame-scaled gate (a channel quiet *this frame* should be
        # small, which is what makes propagation visible).
        if frame < 0 or frame >= self.n_frames:
            gmax_all = float(np.max(np.abs(ei_data[:n_ch]))) or 1.0
            keep = mag > (0.06 * gmax_all)
            gated = np.where(keep, norm_mag, 0.0)
        else:
            noise_gate = 0.12  # fraction of soma-frame spike scale
            gated = np.where(norm_mag < noise_gate, 0.0, norm_mag)

        # bubble radius in data (µm): silent electrodes are tiny dots, strong
        # ones grow to ~0.42× pitch (disc ~0.84× pitch across) — readable but
        # leaving clear gaps so the photo and neighbours stay visible.
        r_min = pitch * 0.04
        r_max = pitch * 0.42
        radii = r_min + (r_max - r_min) * np.sqrt(gated)

        from matplotlib.collections import EllipseCollection

        # alpha: even the strongest bubble stays semi-transparent so the photo
        # underlay always reads through; faint channels nearly vanish.
        alphas = 0.10 + 0.42 * gated
        face_rgba = np.array(face_rgba)
        face_rgba[:, 3] = alphas

        widths = radii * 2.0
        heights = radii * 2.0

        # soft glow underlay: larger, very transparent copy for a "hot" feel
        glow = EllipseCollection(
            widths=widths * 1.6,
            heights=heights * 1.6,
            angles=np.zeros(n_ch),
            units="xy",
            offsets=pos,
            offset_transform=ax.transData,
            facecolors=np.column_stack(
                [face_rgba[:, :3], alphas * 0.25]
            ),
            edgecolors="none",
            zorder=1.5,
        )
        ax.add_collection(glow)

        bubbles = EllipseCollection(
            widths=widths,
            heights=heights,
            angles=np.zeros(n_ch),
            units="xy",
            offsets=pos,
            offset_transform=ax.transData,
            facecolors=face_rgba,
            edgecolors=to_rgba(colors["bg_panel"], 0.35),
            linewidths=0.4,
            zorder=2,
        )
        ax.add_collection(bubbles)

        # faint reference dots for every electrode so the array is legible
        # even where the cell is silent
        ax.scatter(
            pos[:, 0], pos[:, 1],
            s=1.5, color=to_rgba(colors["text_secondary"], 0.25),
            zorder=1.6, rasterized=True,
        )

        pad = pitch * 1.5
        ax.set_xlim(pos[:, 0].min() - pad, pos[:, 0].max() + pad)
        ax.set_ylim(pos[:, 1].min() - pad, pos[:, 1].max() + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        # top-channel markers (ring + rank label)
        if self.current_channels is not None:
            for j, ch in enumerate(self.current_channels):
                if ch >= len(ch_pos):
                    continue
                x, y = ch_pos[ch]
                ax.add_patch(
                    __import__("matplotlib.patches", fromlist=["Circle"]).Circle(
                        (x, y), pitch * 0.7, fill=False,
                        edgecolor="cyan", linewidth=1.4, zorder=3, alpha=0.9,
                    )
                )
                ax.text(
                    x, y + pitch * 0.9, str(j),
                    color="cyan", fontsize=7, ha="center", va="bottom",
                    zorder=3, fontweight="bold",
                )

        # colourbar for interpretability
        sm = mpl_cm.ScalarMappable(norm=cnorm, cmap=cmap)
        sm.set_array([])
        cbar = self.spatial_canvas.fig.colorbar(
            sm, ax=ax, fraction=0.025, pad=0.01
        )
        cbar.set_label(
            "Voltage (µV)" if signed else "Peak |amplitude| (µV)",
            color=colors["text_secondary"], fontsize=8,
        )
        cbar.ax.tick_params(colors=colors["text_secondary"], labelsize=7)
        cbar.outline.set_edgecolor(colors["border_subtle"])

        self.spatial_canvas.fig.suptitle(
            title, color=colors["text_primary"], fontsize=13
        )
        self.spatial_canvas.fig.tight_layout()
        self.spatial_canvas.draw()

    def _draw_temporal(self, ei_data_list, cluster_ids, channels):
        """Render one waveform row per top channel, one trace per cluster."""
        self.temporal_widget.clear()
        colors = self.main_window.get_current_colors()
        sr = self.main_window.data_manager.sampling_rate

        for i, ch in enumerate(channels):
            pi = pg.PlotItem()
            for side in ("bottom", "left"):
                pi.getAxis(side).setPen(pg.mkPen(colors["border_default"]))
                pi.getAxis(side).setTextPen(pg.mkPen(colors["text_secondary"]))
            self.temporal_widget.addItem(pi, i, 0)

            for j, ei in enumerate(ei_data_list):
                t_ms = np.arange(ei.shape[1]) / sr * 1000.0
                color = pg.intColor(j, hues=max(len(cluster_ids), 1))
                pi.plot(
                    t_ms,
                    ei[ch, :],
                    pen=pg.mkPen(color=color, width=2),
                    name=f"C{cluster_ids[j]}",
                )

            pi.setLabel("left", f"ch {ch}")
            pi.setLabel("bottom", "Time (ms)")
            if len(cluster_ids) > 1:
                pi.addLegend()

    # -----------------------------------------------------------------------
    # Animation
    # -----------------------------------------------------------------------

    def _setup_anim_controls(self):
        """Enable slider based on current n_frames. Start on the soma frame
        (the spike), not sample 0 — sample 0 is pre-spike baseline where the
        EI is essentially flat/noise, which is why a fresh selection used to
        show an empty map."""
        if self.n_frames > 0:
            self.frame_slider.setMaximum(self.n_frames - 1)
            start = int(np.clip(self._soma_frame, 0, self.n_frames - 1))
            self.frame_slider.setValue(start)
            self.frame_slider.setEnabled(True)
            self._anim_frame = start
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
        # Sweep a window bracketing the soma spike rather than the whole
        # 6-7 ms trace — most of which is flat pre/post-spike baseline. The
        # window runs from a few samples before the soma trough (to show the
        # wavefront initiate) through the axon-propagation tail after it.
        sr = (
            self.main_window.data_manager.sampling_rate
            if self.main_window.data_manager
            else 20000
        )
        pre = max(1, int(round(0.4e-3 * sr)))   # ~0.4 ms before trough
        post = max(2, int(round(2.5e-3 * sr)))  # ~2.5 ms after (axon tail)
        lo = int(np.clip(self._soma_frame - pre, 0, self.n_frames - 1))
        hi = int(np.clip(self._soma_frame + post, 0, self.n_frames - 1))
        if hi <= lo:
            lo, hi = 0, self.n_frames - 1

        nxt = self._anim_frame + 1
        if nxt > hi or nxt < lo:
            nxt = lo
        self._anim_frame = nxt
        # blockSignals so we don't re-trigger _on_frame_slider redundantly
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self._anim_frame)
        self.frame_slider.blockSignals(False)
        self._render_anim_frame()

    def _on_frame_slider(self, value: int):
        self._anim_frame = value
        sr = (
            self.main_window.data_manager.sampling_rate
            if self.main_window.data_manager
            else 20000
        )
        t_ms = value / sr * 1000.0
        self.frame_label.setText(f"t: {t_ms:.2f} ms")
        self._render_anim_frame()

    def _active_overlay_index(self) -> int:
        n = len(self.current_ei_data) if self.current_ei_data else 0
        if n <= 0:
            return 0
        return int(np.clip(self.overlay_index, 0, n - 1))

    def _redraw_current_view(self):
        """Redraw whichever view is active. Use this instead of calling
        _draw_heatmap_frame directly from shared handlers (photo toggle,
        alpha, canvas click, overlay dropdown) — otherwise those handlers
        force the canvas back to Heatmap even when the user is in
        Waveform view."""
        if self.current_view == "Waveform":
            self._draw_waveform_frame()
        elif self.current_view == "3D":
            if self.current_ei_data is not None:
                ch_pos = self._resolve_channel_positions()
                ei = self.current_ei_data[self._active_overlay_index()]
                self.mountain_widget.plot_ei_3d(ei, ch_pos)
        else:
            self._draw_heatmap_frame(self._anim_frame)

    def _render_anim_frame(self):
        # Waveform and 3D are not time-frame views. Drawing the heatmap here
        # left the View combo on "Waveform" while the canvas showed Heatmap.
        if self.current_view != "Heatmap":
            return
        self._draw_heatmap_frame(self._anim_frame)
        sr = (
            self.main_window.data_manager.sampling_rate
            if self.main_window.data_manager
            else 20000
        )
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
            self.main_window.status_bar.showMessage(f"Channel {nearest}{ei_amp}")
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
        current = (
            list(self.current_channels) if self.current_channels is not None else []
        )
        if nearest in current:
            current.remove(nearest)
        new_channels = np.array([nearest] + current[:2], dtype=int)
        self.current_channels = new_channels
        self._redraw_current_view()
        self._draw_temporal(
            self.current_ei_data, self.current_cluster_ids, new_channels
        )

    # -----------------------------------------------------------------------
    # View switching
    # -----------------------------------------------------------------------

    def _on_view_changed(self, text: str):
        if text not in ("Heatmap", "3D", "Waveform"):
            return
        self.current_view = text
        self.spatial_stack.setCurrentIndex(1 if text == "3D" else 0)
        if self.current_ei_data is not None:
            self._redraw_current_view()

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
        Paint the full EI waveform footprint for the selected cluster(s) in
        micron space, co-registered with the photo underlay.

        Rendered as vector LineCollections (never rasterized) so traces stay
        crisp at any zoom. Each electrode's trace is drawn in a small box
        sized to the array pitch; per-trace brightness scales with that
        channel's own amplitude so the propagating axon/soma signal reads as
        a bright "comet" against dim, near-silent electrodes. The dominant
        (reference) channel is highlighted thicker and fully opaque. A single
        global normalization keeps relative amplitudes honest across the whole
        array, and a scale bar reports the box size in µV/µm.
        """
        if not self.current_ei_data or not self.current_cluster_ids:
            return

        ch_pos = self._resolve_channel_positions()
        if ch_pos is None or len(ch_pos) == 0:
            return

        colors_ui = self.main_window.get_current_colors()

        self.spatial_canvas.fig.clear()
        ax = self.spatial_canvas.fig.add_subplot(111)
        ax.set_facecolor(colors_ui["bg_panel"])
        self.spatial_canvas.fig.patch.set_facecolor(colors_ui["bg_panel"])

        # ── photo underlay ──────────────────────────────────────────────────
        if (
            self._overlay_enabled
            and self._overlay_image_rgba is not None
            and self._overlay_extent_um is not None
        ):
            xl, xr_img, yb, yt = self._overlay_extent_um
            ax.imshow(
                self._overlay_image_rgba,
                aspect="auto",
                origin="upper",
                extent=(xl, xr_img, yb, yt),
                alpha=self._overlay_alpha,
                interpolation="bilinear",
                zorder=0,
            )

        pitch = self._get_electrode_pitch(ch_pos)
        # box geometry in µm. Box spans ~1.6× pitch so waveform shape is
        # legible; the soma spike is allowed to overflow its box (that reads
        # as intensity and is standard for EI footprints). Horizontal (time)
        # axis kept a bit narrower than vertical so traces don't run into
        # left/right neighbours.
        box_w = pitch * 1.15
        box_h = pitch * 1.85

        pad = pitch * 1.5
        ax.set_xlim(ch_pos[:, 0].min() - pad, ch_pos[:, 0].max() + pad)
        ax.set_ylim(ch_pos[:, 1].min() - pad, ch_pos[:, 1].max() + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        # ── dim electrode grid for spatial reference ────────────────────────
        ax.scatter(
            ch_pos[:, 0], ch_pos[:, 1],
            s=3, color=to_rgba(colors_ui["text_secondary"], 0.30),
            zorder=1.5, rasterized=True,
        )

        self._waveform_artists = []
        ref_ch = (
            int(self.current_channels[0])
            if self.current_channels is not None and len(self.current_channels)
            else None
        )
        sr = (
            self.main_window.data_manager.sampling_rate
            if self.main_window.data_manager else 20000.0
        )
        multi = len(self.current_ei_data) > 1
        # latency colormap: soma/early = magenta→blue, later (axon) = cyan→
        # green→yellow, so a propagating axon reads as a colour sweep away
        # from the soma. Only used in single-cluster mode; multi-cluster uses
        # one solid colour per cluster to keep them distinguishable.
        lat_cmap = _get_cmap("plasma")
        cluster_solid = ["#00e5ff", "#ff9800", "#69ff47", "#ff4081"]

        for i, (ei_arr, cid) in enumerate(
            zip(self.current_ei_data, self.current_cluster_ids)
        ):
            n_ch = min(ei_arr.shape[0], len(ch_pos))
            ei = ei_arr[:n_ch].astype(float)
            pos = ch_pos[:n_ch]
            T = ei.shape[1]
            soma_idx = int(np.clip(self._soma_frame, 0, T - 1))

            # The EI is a spike-triggered average — already smooth and
            # baseline-referenced. We plot ei as-is; the only question is which
            # channels carry real signal (to draw) vs. baseline (to draw
            # faint).
            gmax = float(np.max(np.abs(ei)))
            if gmax <= 0:
                continue
            p2p = ei.max(axis=1) - ei.min(axis=1)
            p2p_max = float(p2p.max()) + 1e-12

            # SIGNAL GATE = the heatmap's MAX-PROJECTION rule, identical here so
            # the two views show exactly the same footprint: a channel is shown
            # iff its peak |amplitude| across ALL time exceeds a fraction of the
            # global peak. Max-projection (not a single-frame window) is what
            # includes the axon — every electrode the cell ever drives — while
            # excluding pure-noise electrodes. This both fixes "waveform shows
            # fewer channels than heatmap" AND "waveform too noisy / too many":
            # one threshold governs both views.
            maxproj = np.max(np.abs(ei), axis=1)          # per-channel, all time
            MAXPROJ_FRAC = 0.06                            # matches heatmap gate
            signal_mask = maxproj > (MAXPROJ_FRAC * gmax)

            # AMPLITUDE SCALING — mild compression to lift dendrites. gamma=0.7
            # maps a 10%-of-soma dendrite to ~20% box height (visible) with
            # minimal distortion of the (already clean) trace shape. Signal
            # channels only; baseline channels draw flat.
            gamma = 0.7
            compressed = np.sign(ei) * np.power(np.abs(ei) / gmax, gamma)
            compressed = compressed * signal_mask[:, None]
            y_scaled = compressed * (box_h * 0.55)

            active = signal_mask

            # per-channel trough latency relative to the soma frame → ms
            trough_idx = np.argmin(ei, axis=1)
            latency_ms = (trough_idx - soma_idx) / sr * 1000.0
            # colour scale spans the observed propagation delays
            if active.any():
                lo_l = float(np.percentile(latency_ms[active], 2))
                hi_l = float(np.percentile(latency_ms[active], 98))
            else:
                lo_l, hi_l = -0.5, 1.5
            if hi_l - lo_l < 1e-3:
                lo_l, hi_l = lo_l - 0.5, hi_l + 0.5
            lat_norm = Normalize(vmin=lo_l, vmax=hi_l)

            # time axis in µm, centred on each electrode
            t = np.linspace(-0.5, 0.5, T) * box_w

            segments = []
            seg_colors = []
            seg_lws = []
            ref_seg = None
            for c in range(n_ch):
                xs = t + pos[c, 0]
                ys = y_scaled[c] + pos[c, 1]
                seg = np.column_stack([xs, ys])
                if ref_ch is not None and c == ref_ch:
                    ref_seg = seg
                    continue
                segments.append(seg)

                amp_w = float(p2p[c] / p2p_max)  # 0..1 relative amplitude
                vis_w = float(np.sqrt(amp_w))
                if not active[c]:
                    # silent: flat faint baseline reference, thin and dim so
                    # the array stays legible without adding a noise carpet
                    rgba = list(to_rgba(colors_ui["text_secondary"]))
                    rgba[3] = 0.18
                    seg_colors.append(rgba)
                    seg_lws.append(0.4)
                    continue
                if multi:
                    rgba = list(to_rgba(cluster_solid[i % len(cluster_solid)]))
                else:
                    rgba = list(lat_cmap(lat_norm(latency_ms[c])))
                # signal channels: opaque and thick enough to read clearly.
                rgba[3] = float(np.clip(0.55 + 0.45 * vis_w, 0.55, 1.0))
                seg_colors.append(rgba)
                seg_lws.append(0.8 + 1.4 * vis_w)

            if segments:
                lc = LineCollection(
                    segments, colors=seg_colors, linewidths=seg_lws,
                    zorder=6, capstyle="round",
                )
                lc.set_antialiased(True)
                ax.add_collection(lc)
                self._waveform_artists.append(lc)

            # reference / soma channel drawn last, bold and bright, on top
            if ref_seg is not None:
                ref_color = (
                    cluster_solid[i % len(cluster_solid)] if multi else "#ffffff"
                )
                (line,) = ax.plot(
                    ref_seg[:, 0], ref_seg[:, 1],
                    color=ref_color, linewidth=2.2, alpha=1.0, zorder=8,
                    solid_capstyle="round",
                )
                self._waveform_artists.append(line)
                from matplotlib.patches import Circle
                ax.add_patch(
                    Circle(
                        (pos[ref_ch, 0], pos[ref_ch, 1]), box_w * 0.5,
                        fill=False, edgecolor=ref_color,
                        linewidth=1.2, alpha=0.8, zorder=7,
                    )
                )

            # latency colourbar (single-cluster only — it's what reveals the
            # axon propagation direction/speed)
            if not multi and active.any():
                sm = mpl_cm.ScalarMappable(norm=lat_norm, cmap=lat_cmap)
                sm.set_array([])
                cbar = self.spatial_canvas.fig.colorbar(
                    sm, ax=ax, fraction=0.025, pad=0.01
                )
                cbar.set_label(
                    "Trough latency re: soma (ms)",
                    color=colors_ui["text_secondary"], fontsize=8,
                )
                cbar.ax.tick_params(colors=colors_ui["text_secondary"], labelsize=7)
                cbar.outline.set_edgecolor(colors_ui["border_subtle"])

        # ── scale bars: one spatial (µm), one amplitude (µV) ────────────────
        try:
            x0 = ch_pos[:, 0].min()
            y0 = ch_pos[:, 1].min() - pitch * 0.9
            # spatial reference = one electrode pitch
            ax.plot(
                [x0, x0 + pitch], [y0, y0],
                color=colors_ui["text_secondary"], linewidth=1.5, zorder=9,
            )
            ax.text(
                x0 + pitch / 2, y0 - pitch * 0.25,
                f"{pitch:.0f} µm",
                color=colors_ui["text_secondary"], fontsize=7,
                ha="center", va="top", zorder=9,
            )
            # amplitude reference. Because the vertical scale is compressed
            # (gamma=0.55), the full box height corresponds to the soma peak
            # while the half-height mark corresponds to gmax*0.5**(1/gamma).
            # Label both so the compression is legible rather than misleading.
            gmax0 = float(np.max(np.abs(self.current_ei_data[0])))
            half_uv = gmax0 * (0.5 ** (1.0 / 0.7))
            x1 = x0 + pitch * 2.5
            ax.plot(
                [x1, x1], [y0, y0 + box_h], color=colors_ui["text_secondary"],
                linewidth=1.5, zorder=9,
            )
            ax.plot(
                [x1 - pitch * 0.06, x1 + pitch * 0.06],
                [y0 + box_h / 2, y0 + box_h / 2],
                color=colors_ui["text_secondary"], linewidth=1.0, zorder=9,
            )
            ax.text(
                x1 + pitch * 0.15, y0 + box_h,
                f"{gmax0:.0f} µV",
                color=colors_ui["text_secondary"], fontsize=7,
                ha="left", va="center", zorder=9,
            )
            ax.text(
                x1 + pitch * 0.15, y0 + box_h / 2,
                f"{half_uv:.0f} µV",
                color=colors_ui["text_secondary"], fontsize=6.5,
                ha="left", va="center", zorder=9, alpha=0.8,
            )
            ax.text(
                x1 - pitch * 0.15, y0 + box_h * 1.12,
                "√-compressed",
                color=colors_ui["text_secondary"], fontsize=6, alpha=0.7,
                ha="left", va="center", zorder=9, style="italic",
            )
        except Exception:
            pass

        cluster_str = ", ".join(str(c) for c in self.current_cluster_ids)
        self.spatial_canvas.fig.suptitle(
            f"EI waveforms — cluster{'s' if len(self.current_cluster_ids) > 1 else ''} "
            f"{cluster_str}",
            color=colors_ui["text_primary"],
            fontsize=13,
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
            self.overlay_dropdown.setCurrentIndex(np.clip(self.overlay_index, 0, n - 1))
            self.overlay_dropdown.blockSignals(False)

    def _on_overlay_dropdown(self, idx: int):
        if idx < 0:
            return
        self.overlay_index = idx
        self._redraw_current_view()

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
        self._overlay_extent_um = None

        try:
            with open(transform_path, "r") as fh:
                data = json.load(fh)

            img_name = data.get("image_file")
            if not img_name:
                logger.warning("array_transform.json has no image_file field")
                return

            img_path = Path(transform_path).parent / img_name
            if not img_path.exists():
                logger.warning("Array image not found: %s", img_path)
                return

            # Load as RGBA float32 so matplotlib can composite cleanly
            img = Image.open(img_path).convert("RGBA")
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_h, img_w = img_array.shape[:2]

            # Calibration: pixel = scale * micron + offset
            # Invert:      micron = (pixel - offset) / scale
            sx = float(data.get("scale_x", 1.0))
            sy = float(data.get("scale_y", 1.0))
            ox = float(data.get("offset_x", 0.0))
            oy = float(data.get("offset_y", 0.0))

            x_left_um = (0 - ox) / sx
            x_right_um = (img_w - ox) / sx
            y_at_row0 = (0 - oy) / sy  # micron-Y when pixel row = 0 (top)
            y_at_rowH = (img_h - oy) / sy  # micron-Y when pixel row = H (bottom)

            # origin='upper' → row-0 sits at the TOP of the bounding box.
            # extent tuple = (left, right, bottom, top)
            bottom_um = min(y_at_row0, y_at_rowH)
            top_um = max(y_at_row0, y_at_rowH)

            self._overlay_image_rgba = img_array
            self._overlay_extent_um = (x_left_um, x_right_um, bottom_um, top_um)

            if hasattr(self, "photo_btn"):
                self.photo_btn.setToolTip(
                    f"Photo: {img_path.name}\n"
                    f"Extent  X {x_left_um:.0f}–{x_right_um:.0f} µm  "
                    f"Y {bottom_um:.0f}–{top_um:.0f} µm"
                )

            logger.info(
                "EI photo overlay ready: %s  (%.0f–%.0f µm, %.0f–%.0f µm)",
                img_path.name,
                x_left_um,
                x_right_um,
                bottom_um,
                top_um,
            )

        except Exception:
            logger.exception("Failed to load array image for EI panel overlay")
            self._overlay_image_rgba = None
            self._overlay_extent_um = None

    # -----------------------------------------------------------------------
    # Photo overlay — slot handlers
    # -----------------------------------------------------------------------

    def _on_photo_toggled(self, checked: bool) -> None:
        if checked and self._overlay_image_rgba is None:
            self._try_autoload_transform()

        if self._overlay_image_rgba is None:
            self.photo_btn.setChecked(False)
            if hasattr(self.main_window, "status_bar"):
                self.main_window.status_bar.showMessage(
                    "No array image loaded — use Array → Map Image to Array…", 4000
                )
            return

        self._overlay_enabled = checked
        self.overlay_alpha_slider.setEnabled(checked)

        if self.current_ei_data is not None:
            self._redraw_current_view()

    def _on_overlay_alpha_changed(self, value: int) -> None:
        self._overlay_alpha = value / 100.0
        if self._overlay_enabled and self.current_ei_data is not None:
            self._redraw_current_view()

    def _try_autoload_transform(self) -> None:
        dm = self.main_window.data_manager
        if dm is None:
            return
        from pathlib import Path

        candidate = Path(dm.kilosort_dir).parent / "transforms" / "array_transform.json"
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
        self._cell_tracer_dlg = dlg  # prevent GC

    def _on_tracer_cluster_selected(self, cid: int) -> None:
        """Jump to a cluster chosen from the Cell Tracer results table."""
        mw = self.main_window
        if hasattr(mw, "_select_cluster_by_id"):
            mw._select_cluster_by_id(cid)
        elif hasattr(mw, "select_cluster"):
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

    def _get_top_electrodes(
        self,
        ei: np.ndarray,
        n_interval: int = 2,
        n_markers: int = 5,
        b_sort: bool = True,
    ) -> np.ndarray:
        """Return indices of the top-N electrodes by log-amplitude, optionally sorted by peak latency."""
        ei_map = np.log10(np.max(np.abs(ei), axis=1) + 1e-6)
        top_idx = np.argsort(ei_map.flatten())[::-1][::n_interval][:n_markers]

        if b_sort and len(top_idx) > 0:
            peak_times = np.argmin(ei[top_idx, :], axis=1)
            top_idx = top_idx[np.argsort(peak_times)]

        return top_idx

    def _show_message(self, msg: str, color: str = "cyan"):
        self.clear()
        self.spatial_canvas.fig.text(
            0.5,
            0.5,
            msg,
            ha="center",
            va="center",
            color=color,
            fontsize=14,
        )
        self.spatial_canvas.draw()