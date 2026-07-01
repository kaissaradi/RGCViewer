"""
STA Panel — spike-triggered average viewer for RGCViewer.

Layout
──────
  Toolbar  (single row: frame nav left | animation controls right)
  ┌──────────────────────────┬─────────────────┐
  │  RF canvas (pyqtgraph)   │ Temporal filter  │  ~70% height
  └──────────────────────────┴─────────────────┘
  ┌──────────────────────────────────────────────┐
  │  Metrics strip (fixed-height pill bar)        │  ~90px
  └──────────────────────────────────────────────┘
"""

import logging

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QSlider, QSplitter,
    QVBoxLayout, QWidget,
)

from ...analysis import analysis_core
from ..theme import DARK_COLORS
from ..widgets.widgets import MplCanvas

logger = logging.getLogger(__name__)

# Matplotlib channel colours — consistent everywhere in this file
_CH_COLORS = ['#e05555', '#55c155', '#5588e0']   # R, G, B (softer than pure)
_CH_NAMES  = ['Red', 'Green', 'Blue']


# ──────────────────────────────────────────────────────────────────────────────
# Metrics pill bar
# ──────────────────────────────────────────────────────────────────────────────

class _MetricsBar(QWidget):
    """
    Fixed-height horizontal bar of metric pills arranged in two labelled groups.

    Each pill is a pair of QLabels: a dim 'name' label and a bright 'value'
    label, painted inside a lightly rounded container frame.  No scrollbar
    ever appears — the 10 metrics fit comfortably at any reasonable width.
    """

    # Ordered display spec: (key_in_metrics_dict, short_display_name)
    _TEMPORAL_FIELDS = [
        ("Polarity",        ""),          # special — coloured badge
        ("Peak (ms)",       "Peak"),
        ("FWHM (ms)",       "FWHM"),
        ("Biphasic Index",  "BI"),
        ("SNR",             "SNR"),
    ]
    _SPATIAL_FIELDS = [
        ("RF σx (stix)",    "σx"),
        ("RF σy (stix)",    "σy"),
        ("RF Area (stix²)", "Area"),
        ("Orientation (°)", "θ"),
        ("Ellipticity",     "ε"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(88)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 4)
        outer.setSpacing(8)

        self._val_labels: dict[str, QLabel] = {}

        for group_name, fields in [("Temporal", self._TEMPORAL_FIELDS),
                                    ("Spatial",  self._SPATIAL_FIELDS)]:
            group = self._make_group(group_name, fields)
            outer.addWidget(group, 1)

    # ── construction helpers ──────────────────────────────────────────────────

    def _make_group(self, title: str, fields: list) -> QFrame:
        frame = QFrame()
        frame.setObjectName("metricsGroup")
        vbox = QVBoxLayout(frame)
        vbox.setContentsMargins(6, 2, 6, 2)
        vbox.setSpacing(2)

        title_lbl = QLabel(title.upper())
        title_lbl.setObjectName("metricsGroupTitle")
        vbox.addWidget(title_lbl)

        pills_row = QHBoxLayout()
        pills_row.setSpacing(4)
        for key, short_name in fields:
            pill = self._make_pill(key, short_name)
            pills_row.addWidget(pill)
        pills_row.addStretch()
        vbox.addLayout(pills_row)

        return frame

    def _make_pill(self, key: str, short_name: str) -> QFrame:
        pill = QFrame()
        pill.setObjectName("metricsPill")
        pill.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        row = QHBoxLayout(pill)
        row.setContentsMargins(7, 3, 7, 3)
        row.setSpacing(4)

        if short_name:                        # normal pill: "FWHM  20.7ms"
            name_lbl = QLabel(short_name)
            name_lbl.setObjectName("pillName")
            row.addWidget(name_lbl)

        val_lbl = QLabel("—")
        val_lbl.setObjectName("pillValue" if short_name else "pillPolarity")
        row.addWidget(val_lbl)

        self._val_labels[key] = val_lbl
        return pill

    # ── public API ────────────────────────────────────────────────────────────

    def update_metrics(self, metrics: dict | None):
        """Populate pills from the metrics dict returned by compute_sta_metrics."""
        if not metrics:
            for lbl in self._val_labels.values():
                lbl.setText("—")
                lbl.setStyleSheet("")
            return

        for key, lbl in self._val_labels.items():
            val = metrics.get(key, "—")

            if key == "Polarity":
                is_off = (val == "OFF")
                color  = "#d95f5f" if is_off else "#5fc45f"
                lbl.setText(val)
                lbl.setStyleSheet(
                    f"color: {color}; font-weight: 700; font-size: 12px;"
                )
            else:
                lbl.setText(str(val))
                lbl.setStyleSheet("")

    def apply_theme(self, colors: dict):
        bg      = colors.get('bg_surface',     '#1e1e1e')
        bg_pill = colors.get('bg_panel',        '#252525')
        border  = colors.get('border_subtle',   '#333333')
        dim     = colors.get('text_secondary',  '#888888')
        bright  = colors.get('text_primary',    '#dddddd')

        self.setStyleSheet(f"""
            QFrame#metricsGroup {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 4px;
            }}
            QLabel#metricsGroupTitle {{
                color: {dim};
                font-size: 9px;
                font-weight: 600;
                letter-spacing: 1px;
            }}
            QFrame#metricsPill {{
                background: {bg_pill};
                border: 1px solid {border};
                border-radius: 3px;
            }}
            QLabel#pillName {{
                color: {dim};
                font-size: 10px;
            }}
            QLabel#pillValue {{
                color: {bright};
                font-size: 11px;
                font-weight: 600;
            }}
            QLabel#pillPolarity {{
                font-size: 12px;
                font-weight: 700;
            }}
        """)


# ──────────────────────────────────────────────────────────────────────────────
# Main panel
# ──────────────────────────────────────────────────────────────────────────────

class STAPanel(QWidget):
    """
    Three-zone STA viewer:
      • RF canvas (pyqtgraph) — left, full height of upper zone
      • Temporal filter plot (matplotlib) — right, full height of upper zone
      • Metrics bar (_MetricsBar) — bottom strip, fixed height
    """

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        # ── animation state ───────────────────────────────────────────────────
        self.current_frame_index    = 0
        self.total_sta_frames       = 0
        self.current_sta_data       = None
        self.current_sta_cluster_id = None
        self.current_stafit         = None
        self.sta_animation_timer    = None

        # ── cached metrics (set by _load_sta_data, read by all draw methods) ─
        self._current_metrics: dict | None = None

        self._setup_ui()

        # Sync button text with initial view state
        if getattr(main_window, 'current_sta_view', 'rf') == 'animation':
            self.sta_animation_button.setText("Pause")

    # ──────────────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        colors = self.main_window.get_current_colors()
        root   = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addLayout(self._build_toolbar(colors), 0)
        root.addWidget(self._build_main_splitter(colors), 1)
        root.addWidget(self._build_metrics_bar(colors), 0)

    # ── toolbar ───────────────────────────────────────────────────────────────

    def _build_toolbar(self, colors) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 3, 4, 3)
        bar.setSpacing(6)

        # Frame navigation
        self.sta_frame_prev_button = QPushButton("‹")
        self.sta_frame_prev_button.setFixedWidth(28)
        self.sta_frame_prev_button.setToolTip("Previous frame")

        self.sta_frame_slider = QSlider(Qt.Horizontal)
        self.sta_frame_slider.setFixedWidth(180)
        self.sta_frame_slider.setMaximumHeight(22)
        self.sta_frame_slider.setToolTip("Scrub STA frames")

        self.sta_frame_next_button = QPushButton("›")
        self.sta_frame_next_button.setFixedWidth(28)
        self.sta_frame_next_button.setToolTip("Next frame")

        self.sta_frame_label = QLabel("—")
        self.sta_frame_label.setFixedWidth(68)
        self.sta_frame_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.sta_frame_label.setStyleSheet(
            f"color: {colors.get('text_secondary','#888')}; font-size: 10px;"
        )

        bar.addWidget(self.sta_frame_prev_button)
        bar.addWidget(self.sta_frame_slider)
        bar.addWidget(self.sta_frame_next_button)
        bar.addWidget(self.sta_frame_label)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"color: {colors.get('border_subtle','#333')};")
        bar.addWidget(sep)

        # Animation controls
        self.sta_animation_button      = QPushButton("▶  Play")
        self.sta_animation_stop_button = QPushButton("■  Stop")
        self.sta_animation_button.setFixedWidth(84)
        self.sta_animation_stop_button.setFixedWidth(72)

        bar.addWidget(self.sta_animation_button)
        bar.addWidget(self.sta_animation_stop_button)
        bar.addStretch()

        # Wire signals
        self.sta_frame_prev_button.clicked.connect(self.prev_sta_frame)
        self.sta_frame_next_button.clicked.connect(self.next_sta_frame)
        self.sta_frame_slider.valueChanged.connect(self.update_sta_frame_manual)
        self.sta_animation_button.clicked.connect(self.toggle_animation)
        self.sta_animation_stop_button.clicked.connect(self.stop_animation)

        return bar

    # ── upper splitter (RF | temporal filter) ─────────────────────────────────

    def _build_main_splitter(self, colors) -> QSplitter:
        self.main_splitter = QSplitter(Qt.Horizontal)

        # ── RF canvas (pyqtgraph) ─────────────────────────────────────────────
        self.rf_canvas = pg.GraphicsLayoutWidget()
        self.rf_canvas.setBackground(colors.get('bg_panel', '#1a1a1a'))
        self.rf_canvas.setToolTip("STA receptive field  —  click Play to animate")

        self.rf_view = self.rf_canvas.addViewBox()
        self.rf_view.setAspectLocked(True)
        self.rf_view.invertY(True)

        self._pg_image_item = pg.ImageItem()
        self.rf_view.addItem(self._pg_image_item)

        # Gaussian ellipse outline
        self.rf_ellipse_item = pg.PlotCurveItem(
            pen=pg.mkPen(colors.get('text_primary', '#ddd'), width=1.5,
                         style=Qt.DashLine)
        )
        self.rf_view.addItem(self.rf_ellipse_item)

        # Gaussian centre dot
        self.rf_center_item = pg.ScatterPlotItem(
            size=7,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(255, 220, 80, 220),
        )
        self.rf_view.addItem(self.rf_center_item)

        # ── Temporal filter (matplotlib) ──────────────────────────────────────
        self.temporal_filter_canvas = MplCanvas(self, width=4, height=5, dpi=110)
        self._draw_temporal_placeholder(colors)

        self.main_splitter.addWidget(self.rf_canvas)
        self.main_splitter.addWidget(self.temporal_filter_canvas)
        self.main_splitter.setSizes([580, 320])
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 2)

        return self.main_splitter

    # ── metrics bar ───────────────────────────────────────────────────────────

    def _build_metrics_bar(self, colors) -> _MetricsBar:
        self.metrics_bar = _MetricsBar(self)
        self.metrics_bar.apply_theme(colors)
        return self.metrics_bar

    # ──────────────────────────────────────────────────────────────────────────
    # Theme
    # ──────────────────────────────────────────────────────────────────────────

    def restyle_plots(self, colors):
        self.rf_canvas.setBackground(colors.get('bg_panel', '#1a1a1a'))
        self.rf_ellipse_item.setPen(
            pg.mkPen(colors.get('text_primary', '#ddd'), width=1.5,
                     style=Qt.DashLine)
        )
        self.temporal_filter_canvas.restyle(colors)
        self.metrics_bar.apply_theme(colors)

        cluster_id = self.main_window._get_selected_cluster_id()
        if cluster_id is not None:
            self.update_view(cluster_id)

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────────

    def update_view(self, cluster_id: int):
        """
        Full refresh for a new cell selection.  Loads data once, then
        dispatches to the three draw methods.
        """
        dm = getattr(self.main_window, 'data_manager', None)
        print(f"[VISION-DEBUG][sta_panel] update_view(cluster_id={cluster_id}) called. "
              f"dm is None: {dm is None}", flush=True)

        if dm is None or not dm.vision_stas:
            print(f"[VISION-DEBUG][sta_panel] GUARD A hit -> clearing. "
                  f"dm.vision_stas = {dm.vision_stas if dm is not None else 'N/A (dm is None)'} "
                  f"(type={type(dm.vision_stas).__name__ if dm is not None and dm.vision_stas is not None else 'NoneType'})",
                  flush=True)
            self._clear_all()
            return

        print(f"[VISION-DEBUG][sta_panel] dm.vision_stas OK: "
              f"type={type(dm.vision_stas).__name__}, len={len(dm.vision_stas)}", flush=True)

        vision_id = dm.get_vision_id_for_cluster(cluster_id)
        print(f"[VISION-DEBUG][sta_panel] get_vision_id_for_cluster({cluster_id}) -> {vision_id} "
              f"(is_vision_only={getattr(dm, 'is_vision_only', 'N/A')})", flush=True)

        # NOTE: There is intentionally no "unshifted ID" fallback here.
        # If the correctly-offset vision_id (per Law 1 / get_vision_id_for_cluster)
        # isn't in dm.vision_stas, it means Vision genuinely has no STA for this
        # cell (e.g. it was excluded as MUA/noise during sorting). Falling back to
        # the raw cluster_id would silently key into a DIFFERENT cell's STA data
        # whenever that raw integer happens to coincide with another cell's vision_id.
        # See AGENTS.md Law 1. Treat "missing" as missing — clear the panel below.
        if vision_id not in dm.vision_stas:
            print(f"[VISION-DEBUG][sta_panel] GUARD B hit -> clearing. "
                  f"vision_id={vision_id} not in dm.vision_stas (len={len(dm.vision_stas)}). "
                  f"Sample of available keys: {list(dm.vision_stas.keys())[:10]}", flush=True)
            self._clear_all()
            return

        print(f"[VISION-DEBUG][sta_panel] Both guards passed for vision_id={vision_id}. "
              f"Calling _load_sta_data...", flush=True)

        # Load & cache
        if not self._load_sta_data(cluster_id, vision_id, dm):
            # vision_stas[vision_id] returned None (corrupt byte offset, or
            # the per-cell read timed out — see LazySTADict.__getitem__).
            # That's an expected, documented possibility, not a crash.
            print(f"[VISION-DEBUG][sta_panel] _load_sta_data returned False for vision_id={vision_id} "
                  f"— the actual disk read failed/timed out.", flush=True)
            self._clear_all(reason="STA unavailable for this cell\n(corrupt data or read timeout)")
            return

        print(f"[VISION-DEBUG][sta_panel] _load_sta_data SUCCEEDED for vision_id={vision_id}. Drawing...", flush=True)

        # Draw
        self._draw_rf_frame()
        self._draw_temporal_filter()
        self._update_metrics_bar()

        # Honour animation state if it was already running
        if getattr(self.main_window, 'current_sta_view', 'rf') == 'animation':
            self._start_animation_timer()

    # ──────────────────────────────────────────────────────────────────────────
    # Data loading  (single call per cell selection)
    # ──────────────────────────────────────────────────────────────────────────

    def _load_sta_data(self, cluster_id: int, vision_id: int, dm) -> bool:
        """
        Fetch sta_data, stafit, and the full metrics dict.  Results cached on
        self so every draw method reads the same objects without re-fetching.

        Returns False (and leaves self.current_sta_data alone) if the STA
        read failed — LazySTADict.__getitem__ deliberately returns None for
        cells with corrupt byte offsets or reads that timed out, per its
        documented contract ("Returning None is safe — all callers check
        hasattr(sta_data, 'red')."). Every other caller in data_manager.py
        already checks this; this one didn't, which is why a single bad
        cell could silently kill the rest of the panel update.
        """
        sta_data = dm.vision_stas[vision_id]

        if not hasattr(sta_data, 'red') or sta_data.red is None:
            logger.warning(
                "STA read returned no data for vision_id=%d (cluster_id=%d); "
                "leaving panel cleared.", vision_id, cluster_id,
            )
            return False

        try:
            stafit = dm.vision_params.get_stafit_for_cell(vision_id)
        except Exception as exc:
            logger.debug("get_stafit_for_cell(%d) failed: %s", vision_id, exc)
            stafit = None

        try:
            metrics = analysis_core.compute_sta_metrics(
                sta_data, stafit, dm.vision_params, vision_id
            )
        except Exception as exc:
            logger.error("compute_sta_metrics failed for cell %d: %s", vision_id, exc)
            metrics = {}

        # Animation state
        n_frames = sta_data.red.shape[2]
        self.current_sta_data       = sta_data
        self.current_stafit         = stafit
        self.current_sta_cluster_id = cluster_id
        self.total_sta_frames       = n_frames
        self._current_metrics       = metrics

        # Park the frame slider on the highest-energy frame
        all_ch        = np.stack([sta_data.red, sta_data.green, sta_data.blue])
        frame_energies = np.max(np.abs(all_ch), axis=(0, 1, 2))
        peak_frame     = int(np.argmax(frame_energies))
        self.current_frame_index = peak_frame

        self.sta_frame_slider.blockSignals(True)
        self.sta_frame_slider.setMinimum(0)
        self.sta_frame_slider.setMaximum(n_frames - 1)
        self.sta_frame_slider.setValue(peak_frame)
        self.sta_frame_slider.blockSignals(False)
        self.sta_frame_slider.setEnabled(True)
        self.sta_frame_label.setText(f"Frame {peak_frame + 1}/{n_frames}")

        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Draw: RF image + ellipse
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_rf_frame(self):
        """Push the current frame to pyqtgraph and update the fit overlay."""
        self._update_pg_image()
        self._update_rf_overlay()

    def _update_pg_image(self):
        """Normalise and push the current frame to pyqtgraph ImageItem."""
        if self.current_sta_data is None:
            self._pg_image_item.clear()
            return

        r = self.current_sta_data.red  [:, :, self.current_frame_index]
        g = self.current_sta_data.green[:, :, self.current_frame_index]
        b = self.current_sta_data.blue [:, :, self.current_frame_index]
        frame = np.stack([r, g, b], axis=-1).astype(np.float32)

        mn, mx = frame.min(), frame.max()
        if mx > mn:
            frame = (frame - mn) / (mx - mn)

        # pyqtgraph ImageItem expects (width, height, 3) with row=x, col=y
        self._pg_image_item.setImage(frame.transpose(1, 0, 2))

    def _update_rf_overlay(self):
        """Draw Gaussian ellipse outline and centre dot from stafit."""
        stafit = self.current_stafit
        if stafit is None:
            self.rf_ellipse_item.setData([], [])
            self.rf_center_item.setData([], [])
            return

        height = self.current_sta_data.red.shape[0]
        cx, cy = stafit.center_x, stafit.center_y
        sx, sy = getattr(stafit, 'std_x', 1), getattr(stafit, 'std_y', 1)
        angle  = getattr(stafit, 'angle', getattr(stafit, 'orientation', 0))

        # Flip Y to match invertY(True) on the ViewBox
        cx_p = cx + 0.5
        cy_p = (height - cy) - 0.5
        ar   = -np.radians(angle)

        t   = np.linspace(0, 2 * np.pi, 120)
        cos_t, sin_t = np.cos(t), np.sin(t)
        x_el = cx_p + sx * cos_t * np.cos(ar) - sy * sin_t * np.sin(ar)
        y_el = cy_p + sx * cos_t * np.sin(ar) + sy * sin_t * np.cos(ar)

        self.rf_ellipse_item.setData(x_el, y_el)
        self.rf_center_item.setData([cx_p], [cy_p])

    # ──────────────────────────────────────────────────────────────────────────
    # Draw: temporal filter
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_temporal_filter(self):
        """
        Plot the normalised dominant-channel temporal filter with:
          • ghost traces for the other channels (alpha 0.18)
          • filled dominant trace
          • FWHM bracket with tick marks
          • peak time annotation
          • ON/OFF badge in the corner
          • spike-time dashed line
        All annotation values come from _raw_temporal in the metrics dict —
        no recomputation here.
        """
        fig    = self.temporal_filter_canvas.fig
        colors = self.main_window.get_current_colors()
        fig.clear()

        metrics = self._current_metrics
        raw     = (metrics or {}).get('_raw_temporal')

        ax = fig.add_subplot(111)
        self._style_ax(ax, colors)

        if raw is None:
            ax.text(0.5, 0.5, "No STA data",
                    transform=ax.transAxes,
                    ha='center', va='center',
                    color=colors.get('text_secondary', '#888'), fontsize=11)
            self.temporal_filter_canvas.draw()
            return

        time_axis   = raw['time_axis']
        norm_trace  = raw['norm_trace']   # smoothed, normalised dom channel
        raw_tc      = raw['raw_tc']       # (n_t, 3) un-normalised all channels
        dom_idx     = raw['dom_idx']
        is_off      = raw['is_off']
        peak_ms     = raw['peak_ms']
        peak_val    = raw['peak_val']
        fwhm_ms     = raw['fwhm_ms']
        fwhm_t0     = raw['fwhm_t_start']
        fwhm_t1     = raw['fwhm_t_end']
        fwhm_h      = raw['fwhm_h']
        source      = raw['source']

        dom_color   = _CH_COLORS[dom_idx]

        # ── ghost traces (other channels, un-normalised, re-scaled) ───────────
        if raw_tc.shape[1] == 3:
            abs_max = np.max(np.abs(raw_tc))
            if abs_max > 0:
                tc_norm = raw_tc / abs_max
            else:
                tc_norm = raw_tc

            for i in range(3):
                if i == dom_idx:
                    continue
                ax.plot(time_axis, tc_norm[:, i],
                        color=_CH_COLORS[i], linewidth=0.9,
                        alpha=0.22, zorder=1)

        # ── dominant trace ────────────────────────────────────────────────────
        ax.plot(time_axis, norm_trace,
                color=dom_color, linewidth=2.0, zorder=3, solid_capstyle='round')
        ax.fill_between(time_axis, norm_trace, 0,
                        color=dom_color, alpha=0.10, zorder=2)

        # ── FWHM bracket ──────────────────────────────────────────────────────
        import math
        if not math.isnan(fwhm_ms) and not math.isnan(fwhm_t0):
            tick_h = 0.06 * np.sign(fwhm_h)           # short vertical tick
            ax.hlines(fwhm_h, fwhm_t0, fwhm_t1,
                      colors='#f0c040', linewidth=1.8, zorder=4)
            ax.vlines([fwhm_t0, fwhm_t1],
                      fwhm_h - tick_h, fwhm_h + tick_h,
                      colors='#f0c040', linewidth=1.8, zorder=4)
            ax.text((fwhm_t0 + fwhm_t1) / 2,
                    fwhm_h + 0.10 * np.sign(fwhm_h),
                    f"{fwhm_ms:.1f} ms",
                    ha='center', va='bottom' if fwhm_h > 0 else 'top',
                    color='#f0c040', fontsize=8, zorder=5)

        # ── peak annotation ───────────────────────────────────────────────────
        ax.scatter([peak_ms], [peak_val],
                   color='#f0c040', s=42, zorder=6, linewidths=0)
        y_offset = 0.13 if peak_val > 0 else -0.13
        ax.text(peak_ms, peak_val + y_offset,
                f"{peak_ms:.0f} ms",
                ha='center', va='bottom' if y_offset > 0 else 'top',
                color='#f0c040', fontsize=8, zorder=6)

        # ── reference lines ───────────────────────────────────────────────────
        ax.axhline(0,   color=colors.get('text_secondary', '#888'),
                   linestyle=':', linewidth=0.8, alpha=0.6, zorder=0)
        ax.axvline(0,   color=colors.get('text_primary', '#ccc'),
                   linestyle='--', linewidth=0.8, alpha=0.55, zorder=0)
        ax.text(0, ax.get_ylim()[1] * 0.88, " spike",
                color=colors.get('text_secondary', '#888'),
                fontsize=7, va='top', zorder=5)

        # ── ON / OFF badge (top-left corner) ─────────────────────────────────
        polarity_str  = "OFF" if is_off else "ON"
        polarity_color = "#d95f5f" if is_off else "#5fc45f"
        ax.text(0.04, 0.96, polarity_str,
                transform=ax.transAxes,
                ha='left', va='top',
                color=polarity_color,
                fontsize=13, fontweight='bold', zorder=7)

        # ── channel label (top-right corner) ─────────────────────────────────
        chan_name = raw['dom_name']
        ax.text(0.97, 0.96, chan_name,
                transform=ax.transAxes,
                ha='right', va='top',
                color=dom_color,
                fontsize=8, alpha=0.8, zorder=7)

        # ── data provenance (bottom-right, tiny) ─────────────────────────────
        src_label = "⚡ precalc" if source == 'precalculated' else "⟳ recomputed"
        ax.text(0.98, 0.03, src_label,
                transform=ax.transAxes,
                ha='right', va='bottom',
                color=colors.get('text_secondary', '#888'),
                fontsize=7, alpha=0.6, zorder=7)

        # ── axis limits ───────────────────────────────────────────────────────
        y_abs = max(np.max(np.abs(norm_trace)), 0.05)
        ax.set_ylim(-y_abs * 1.30, y_abs * 1.30)

        ax.set_xlabel("Time before spike (ms)",
                      color=colors.get('text_secondary', '#888'), fontsize=9)
        ax.tick_params(axis='both', labelsize=8,
                       colors=colors.get('text_secondary', '#888'))

        fig.tight_layout(pad=0.6)
        self.temporal_filter_canvas.draw()

    def _draw_temporal_placeholder(self, colors, message="Select a cell"):
        """Blank placeholder shown before any cell is selected, or to
        surface a reason why the panel is empty (e.g. a failed STA read)."""
        fig = self.temporal_filter_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        self._style_ax(ax, colors)
        ax.text(0.5, 0.5, message,
                transform=ax.transAxes,
                ha='center', va='center',
                color=colors.get('text_secondary', '#888'), fontsize=11)
        self.temporal_filter_canvas.draw()

    @staticmethod
    def _style_ax(ax, colors):
        """Apply dark-theme styling to a matplotlib Axes."""
        bg = colors.get('bg_panel', '#1a1a1a')
        ax.set_facecolor(bg)
        ax.get_figure().patch.set_facecolor(bg)
        ax.tick_params(colors=colors.get('text_secondary', '#888'), labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(colors.get('border_default', '#444'))
        ax.grid(True, color=colors.get('border_subtle', '#333'),
                linewidth=0.5, alpha=0.5)

    # ──────────────────────────────────────────────────────────────────────────
    # Draw: metrics bar
    # ──────────────────────────────────────────────────────────────────────────

    def _update_metrics_bar(self):
        self.metrics_bar.update_metrics(self._current_metrics)

    # ──────────────────────────────────────────────────────────────────────────
    # Clear state
    # ──────────────────────────────────────────────────────────────────────────

    def _clear_all(self, reason: str = "Select a cell"):
        """Reset everything to the no-data state."""
        self._pg_image_item.clear()
        self.rf_ellipse_item.setData([], [])
        self.rf_center_item.setData([], [])
        self.sta_frame_slider.setEnabled(False)
        self.sta_frame_label.setText("—")
        self._current_metrics = None
        self.metrics_bar.update_metrics(None)

        colors = self.main_window.get_current_colors()
        self._draw_temporal_placeholder(colors, message=reason)

    # ──────────────────────────────────────────────────────────────────────────
    # Animation
    # ──────────────────────────────────────────────────────────────────────────

    def toggle_animation(self):
        dm = getattr(self.main_window, 'data_manager', None)
        if not dm or not dm.vision_stas:
            return
        if self.main_window._get_selected_cluster_id() is None:
            return

        if self.sta_animation_timer and self.sta_animation_timer.isActive():
            self.stop_animation()
        else:
            self._start_animation_timer()
            self.sta_animation_button.setText("⏸  Pause")

    def stop_animation(self):
        if self.sta_animation_timer and self.sta_animation_timer.isActive():
            self.sta_animation_timer.stop()
        self.sta_animation_button.setText("▶  Play")

    def _start_animation_timer(self):
        if self.current_sta_data is None:
            return
        if self.sta_animation_timer is None:
            self.sta_animation_timer = QTimer()
            self.sta_animation_timer.timeout.connect(self._advance_frame_internal)
        if not self.sta_animation_timer.isActive():
            self.sta_animation_timer.start(100)   # 10 fps

    def _advance_frame_internal(self):
        """Timer tick — advance one frame without stopping the timer."""
        if self.current_sta_data is None:
            return
        self.current_frame_index = (
            self.current_frame_index + 1) % self.total_sta_frames

        self.sta_frame_slider.blockSignals(True)
        self.sta_frame_slider.setValue(self.current_frame_index)
        self.sta_frame_slider.blockSignals(False)

        self.sta_frame_label.setText(
            f"Frame {self.current_frame_index + 1}/{self.total_sta_frames}")
        self._update_pg_image()

    # ── frame scrubbing ───────────────────────────────────────────────────────

    def update_sta_frame_manual(self, frame_index: int):
        if self.current_sta_data is None:
            return
        self.stop_animation()
        self.current_frame_index = frame_index
        self.sta_frame_label.setText(
            f"Frame {frame_index + 1}/{self.total_sta_frames}")
        self._update_pg_image()

    def prev_sta_frame(self):
        if self.current_sta_data is None:
            return
        self.stop_animation()
        self.current_frame_index = (
            self.current_frame_index - 1) % self.total_sta_frames
        self.sta_frame_slider.setValue(self.current_frame_index)
        self.sta_frame_label.setText(
            f"Frame {self.current_frame_index + 1}/{self.total_sta_frames}")
        self._update_pg_image()

    def next_sta_frame(self):
        if self.current_sta_data is None:
            return
        self.stop_animation()
        self.current_frame_index = (
            self.current_frame_index + 1) % self.total_sta_frames
        self.sta_frame_slider.setValue(self.current_frame_index)
        self.sta_frame_label.setText(
            f"Frame {self.current_frame_index + 1}/{self.total_sta_frames}")
        self._update_pg_image()

    # ── RF canvas click → toggle animation ────────────────────────────────────

    def on_rf_canvas_clicked(self):
        dm = getattr(self.main_window, 'data_manager', None)
        if not dm or not dm.vision_stas:
            return
        if self.main_window._get_selected_cluster_id() is None:
            return

        view = getattr(self.main_window, 'current_sta_view', 'rf')
        if view == 'animation':
            self.stop_animation()
            self.main_window.current_sta_view = 'rf'
            self.main_window.status_bar.showMessage("Animation stopped.", 1500)
        else:
            self._start_animation_timer()
            self.sta_animation_button.setText("⏸  Pause")
            self.main_window.current_sta_view = 'animation'
            self.main_window.status_bar.showMessage(
                "Animating STA  —  click RF or Pause to stop.", 2000)