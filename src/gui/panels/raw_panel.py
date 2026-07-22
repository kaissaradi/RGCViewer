"""
raw_panel.py — Production-quality raw trace viewer panel.

Architecture overview
─────────────────────
Zone 1  Minimap   – thin overview of the full recording: FR density heatmap
                    for the selected unit, viewport rectangle, clickable to
                    jump anywhere.

Zone 2  Main plot – single dominant channel raw trace with:
                    • epoch-style background shading (if symphony markers loaded)
                    • per-unit spike tick strip (all units on dom_chan, colour coded)
                    • selected-unit full-height spike markers (unit colour)
                    • raw LFP trace (white) + optional neighbours (dimmed)

Zone 3  Controls  – smart-jump toolbar | window-size | direct time entry |
                    display toggles

All heavy I/O runs in a QThread worker; the main thread only calls setData()
on pre-allocated PlotDataItems, so redraws are non-blocking.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QThread, QTimer, Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette for per-unit spike colouring (max 16 units on one channel
# before colours wrap).  These are chosen to be distinguishable on a dark
# background and not clash with the white trace.
# ──────────────────────────────────────────────────────────────────────────────
_UNIT_PALETTE: list[tuple[int, int, int]] = [
    (74, 144, 226),  # blue          – primary unit (index 0 = selected unit)
    (240, 180, 40),  # amber
    (80, 200, 120),  # green
    (220, 80, 80),  # red
    (160, 100, 220),  # purple
    (240, 140, 40),  # orange
    (60, 200, 200),  # teal
    (220, 80, 160),  # pink
    (140, 200, 60),  # lime
    (100, 160, 240),  # cornflower
    (240, 220, 80),  # yellow
    (80, 220, 180),  # mint
    (200, 100, 60),  # brown-orange
    (180, 100, 200),  # lavender
    (100, 220, 100),  # light-green
    (220, 160, 100),  # tan
]

# Alpha values for non-selected vs selected units (0-255)
_ALPHA_SELECTED = 220
_ALPHA_OTHER = 100
# Height fraction of the tick strip relative to the total Y range of the plot
_TICK_STRIP_FRAC = 0.07  # tick strip = 7 % of total Y range, below zero line
_TICK_HEIGHT_FRAC = 0.05  # individual tick = 5 %

# Number of FR-density bins in the minimap
_MINIMAP_FR_BINS = 500


def _unit_color(index: int, alpha: int = 255) -> tuple[int, int, int, int]:
    r, g, b = _UNIT_PALETTE[index % len(_UNIT_PALETTE)]
    return (r, g, b, alpha)


# ──────────────────────────────────────────────────────────────────────────────
# Background worker
# ──────────────────────────────────────────────────────────────────────────────


class _RawLoadWorker(pg.QtCore.QObject):
    """
    Loads a raw trace window + ALL-channel spike info on a background thread.

    Emits ``finished`` with:
        cluster_id  : int
        raw_data    : np.ndarray  shape (n_channels, n_samples) float32
        channels    : list[int]   channel indices (dom_chan first)
        start_time  : float
        end_time    : float
    """

    finished = Signal(int, object, object, float, float)
    error = Signal(str)

    def __init__(
        self,
        dm,
        cluster_id: int,
        dom_chan: int,
        neighbour_chans: list[int],
        start_time: float,
        end_time: float,
        load_neighbours: bool = True,
    ):
        super().__init__()
        self._dm = dm
        self._cluster_id = cluster_id
        self._dom_chan = dom_chan
        self._neighbour_chans = neighbour_chans
        self._start_time = start_time
        self._end_time = end_time
        self._load_neighbours = load_neighbours

    def run(self):
        try:
            dm = self._dm
            sr = dm.sampling_rate
            s0 = int(self._start_time * sr)
            s1 = int(self._end_time * sr)

            if self._load_neighbours:
                chans = [self._dom_chan] + [
                    c for c in self._neighbour_chans if c != self._dom_chan
                ]
            else:
                chans = [self._dom_chan]

            raw = dm.get_raw_trace_snippet(chans, s0, s1)

            if raw is None or raw.size == 0:
                self.error.emit("No data returned for requested window.")
                return

            self.finished.emit(
                self._cluster_id,
                raw,
                chans,
                self._start_time,
                self._end_time,
            )
        except Exception as exc:
            logger.exception("_RawLoadWorker error: %s", exc)
            self.error.emit(str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# Main panel
# ──────────────────────────────────────────────────────────────────────────────


class RawPanel(QWidget):
    """
    Full-featured raw trace viewer with multi-unit spike overlay,
    minimap overview, and smart-jump navigation.
    """

    status_message = Signal(str)
    error_message = Signal(str)

    # ── Window-duration presets (seconds) ─────────────────────────────────────
    _WINDOW_PRESETS = [0.5, 1.0, 2.0, 5.0, 10.0]

    def __init__(self, main_window: "MainWindow", parent=None):
        super().__init__(parent)
        self.main_window = main_window

        # ── Core state ────────────────────────────────────────────────────────
        self.current_cluster_id: Optional[int] = None
        self.dom_chan: int = 0
        self.neighbour_chans: list[int] = []
        self.spike_times_sec: np.ndarray = np.empty(0, dtype=np.float64)
        self.spike_amplitudes: np.ndarray = np.empty(0, dtype=np.float64)

        # other-unit spike data on dom_chan: {cluster_id: spike_times_sec}
        self.other_unit_spikes: dict[int, np.ndarray] = {}
        # ordered list of other cluster ids (for colour assignment)
        self.other_unit_ids: list[int] = []

        # view state
        self.window_center: float = 0.0
        self.window_duration: float = 2.0  # seconds

        # navigation state
        self.current_spike_index: int = 0

        # pre-computed jump targets (set in _compute_jump_targets)
        self._jump_targets: dict = {}

        # FR density for the minimap (set in _compute_minimap_fr)
        self._minimap_fr_bins: Optional[np.ndarray] = None
        self._minimap_fr_rate: Optional[np.ndarray] = None
        self._recording_duration: float = 0.0

        # worker thread
        self._worker_thread: Optional[QThread] = None
        self._worker: Optional[_RawLoadWorker] = None

        # debounce timer to avoid hammering the worker on rapid navigation
        self._nav_debounce = QTimer(self)
        self._nav_debounce.setSingleShot(True)
        self._nav_debounce.setInterval(40)  # 40 ms
        self._nav_debounce.timeout.connect(self._execute_load)

        # pending load parameters (set by _request_load)
        self._pending_center: float = 0.0
        self._pending_duration: float = 2.0

        # suppress re-entrant signals while programmatically updating controls
        self._updating_controls: bool = False

        # ── Build UI ──────────────────────────────────────────────────────────
        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API (called from main_window)
    # ─────────────────────────────────────────────────────────────────────────

    def load_data(self, cluster_id: int) -> None:
        """Entry point: called whenever a cluster is selected or the tab is shown."""
        if not self.isVisible():
            return

        dm = self.main_window.data_manager
        if not self._dm_has_raw(dm):
            self.status_message.emit("No raw data file loaded.")
            return

        # ── Resolve dominant channel ──────────────────────────────────────────
        dom_chan, neighbour_chans = self._resolve_channels(dm, cluster_id)

        # ── Load spike times ──────────────────────────────────────────────────
        spike_samples = dm.get_cluster_spikes(cluster_id)
        if len(spike_samples) == 0:
            self.status_message.emit(f"No spikes for cluster {cluster_id}.")
            return

        spike_times_sec = spike_samples.astype(np.float64) / dm.sampling_rate

        # Spike amplitudes (optional – used for largest-amplitude jump target)
        spike_amps = dm.get_cluster_spike_amplitudes(cluster_id)
        spike_amps = np.asarray(spike_amps, dtype=np.float64)

        # ── Other units on same channel ───────────────────────────────────────
        other_unit_spikes, other_unit_ids = self._collect_other_unit_spikes(
            dm, cluster_id, dom_chan
        )

        # ── Recording duration ────────────────────────────────────────────────
        rec_dur = getattr(dm, "n_samples", 0) / max(dm.sampling_rate, 1)

        # ── Commit state ──────────────────────────────────────────────────────
        self.current_cluster_id = cluster_id
        self.dom_chan = dom_chan
        self.neighbour_chans = neighbour_chans
        self.spike_times_sec = spike_times_sec
        self.spike_amplitudes = spike_amps
        self.other_unit_spikes = other_unit_spikes
        self.other_unit_ids = other_unit_ids
        self._recording_duration = rec_dur
        self.current_spike_index = 0

        # ── Pre-computations ─────────────────────────────────────────────────
        self._compute_jump_targets()
        self._compute_minimap_fr()
        self._rebuild_minimap()
        self._populate_jump_combo()

        # ── Navigate to first spike ───────────────────────────────────────────
        first_spike = float(spike_times_sec[0])
        self._request_load(first_spike, self.window_duration)

        self._update_info_label()

    def restyle_plots(self, colors: dict) -> None:
        """Called by MainWindow when the theme changes."""
        bg = colors.get("bg_panel", "#18191C")
        ax_pen = pg.mkPen(colors.get("border_default", "#3D3F48"))
        text_pen = pg.mkPen(colors.get("text_secondary", "#9B9DA6"))

        for pw in (self._minimap_plot, self._main_plot):
            pw.setBackground(bg)
            for axis in ("bottom", "left"):
                ax = pw.getAxis(axis)
                ax.setPen(ax_pen)
                ax.setTextPen(text_pen)

        # Re-render with current cluster if one is loaded
        if self.current_cluster_id is not None:
            self._rebuild_minimap()
            self._request_load(self.window_center, self.window_duration)

    # ─────────────────────────────────────────────────────────────────────────
    # UI Construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── Zone 1: Minimap ───────────────────────────────────────────────────
        self._minimap_plot = pg.PlotWidget()
        self._minimap_plot.setFixedHeight(60)
        self._minimap_plot.setBackground("#18191C")
        self._minimap_plot.hideAxis("left")
        self._minimap_plot.setMenuEnabled(False)
        self._minimap_plot.getViewBox().setMouseEnabled(x=False, y=False)
        self._minimap_plot.setLabel("bottom", "Time (s)", **{"font-size": "9pt"})

        # FR area fill + line
        self._mm_fill = pg.FillBetweenItem(
            pg.PlotDataItem(), pg.PlotDataItem(), brush=pg.mkBrush(_unit_color(0, 45))
        )
        self._mm_line = self._minimap_plot.plot(
            pen=pg.mkPen(_unit_color(0, 180), width=1)
        )
        self._minimap_plot.addItem(self._mm_fill)

        # Viewport rectangle
        self._mm_region = pg.LinearRegionItem(
            movable=True,
            brush=pg.mkBrush(255, 255, 255, 18),
            pen=pg.mkPen(255, 255, 255, 90),
        )
        self._mm_region.sigRegionChangeFinished.connect(self._on_minimap_region_moved)
        self._minimap_plot.addItem(self._mm_region)

        # Current-position marker — bright vertical line showing exact window centre
        self._mm_pos_line = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            movable=False,
            pen=pg.mkPen((255, 200, 60, 220), width=1.5),  # warm amber, always visible
        )
        self._minimap_plot.addItem(self._mm_pos_line)

        # Click-to-jump on minimap
        self._minimap_plot.scene().sigMouseClicked.connect(self._on_minimap_clicked)

        root.addWidget(self._minimap_plot)

        # ── Zone 2: Main plot ─────────────────────────────────────────────────
        self._main_plot = pg.PlotWidget()
        self._main_plot.setBackground("#18191C")
        self._main_plot.setLabel("bottom", "Time (s)")
        self._main_plot.setLabel("left", "Amplitude (µV)")
        self._main_plot.plotItem.getViewBox().setMouseEnabled(y=False)
        self._main_plot.plotItem.getViewBox().setMouseEnabled(x=True)
        self._main_plot.getViewBox().sigXRangeChanged.connect(
            self._on_main_view_scrolled
        )
        # Click on the trace → identify nearest spike and its unit
        self._main_plot.scene().sigMouseClicked.connect(self._on_main_plot_clicked)

        # Floating info label for spike identification (hidden until a click)
        self._spike_info_label = pg.TextItem(
            text="",
            anchor=(0, 1),
            border=pg.mkPen((60, 65, 80), width=1),
            fill=pg.mkBrush(30, 32, 40, 220),
            color=(230, 230, 240),
        )
        self._spike_info_label.setFont(pg.QtGui.QFont("monospace", 9))
        self._main_plot.addItem(self._spike_info_label)
        self._spike_info_label.hide()

        # Timer to auto-hide the info label after 4 s of inactivity
        self._info_hide_timer = QTimer(self)
        self._info_hide_timer.setSingleShot(True)
        self._info_hide_timer.setInterval(4000)
        self._info_hide_timer.timeout.connect(self._spike_info_label.hide)

        # Pre-allocate all plot items (avoids allocating in hot path)

        # Raw trace – dominant channel (bright white, rendered last = on top)
        self._trace_n2 = self._main_plot.plot(
            pen=pg.mkPen((100, 120, 160, 110), width=1)  # neighbour 2 — dimmest, bottom
        )
        self._trace_n1 = self._main_plot.plot(
            pen=pg.mkPen((130, 150, 200, 140), width=1)  # neighbour 1 — mid
        )
        self._trace_dom = self._main_plot.plot(
            pen=pg.mkPen((240, 240, 245, 230), width=1)  # dom channel — brightest
        )
        self._trace_n1.hide()
        self._trace_n2.hide()

        # Channel labels for neighbour traces (shown when neighbours are on)
        self._chan_label_dom = pg.TextItem(
            "", anchor=(0, 1), color=(200, 200, 210, 180)
        )
        self._chan_label_n1 = pg.TextItem("", anchor=(0, 1), color=(130, 150, 200, 160))
        self._chan_label_n2 = pg.TextItem("", anchor=(0, 1), color=(100, 120, 160, 140))
        for lbl in (self._chan_label_dom, self._chan_label_n1, self._chan_label_n2):
            lbl.setFont(pg.QtGui.QFont("monospace", 8))
            self._main_plot.addItem(lbl)
            lbl.hide()

        # Selected-unit spike markers (vertical lines via connect=False trick)
        self._sel_spike_plot = self._main_plot.plot(
            pen=pg.mkPen(_unit_color(0, _ALPHA_SELECTED), width=1.2)
        )

        # Other-unit tick marks – up to 15 other units, each gets its own item
        self._other_spike_plots: list[pg.PlotDataItem] = []
        for i in range(15):
            item = self._main_plot.plot(
                pen=pg.mkPen(_unit_color(i + 1, _ALPHA_OTHER), width=1)
            )
            item.hide()
            self._other_spike_plots.append(item)

        root.addWidget(self._main_plot, stretch=1)

        # ── Zone 3: Controls ─────────────────────────────────────────────────
        root.addWidget(self._build_controls())

    def _build_controls(self) -> QWidget:
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(3)

        # ── Row A: Smart navigation ───────────────────────────────────────────
        row_a = QHBoxLayout()
        row_a.setSpacing(4)

        self._btn_first = self._small_btn("⏮ First")
        self._btn_prev = self._small_btn("◀ Prev")
        self._btn_next = self._small_btn("Next ▶")
        self._btn_last = self._small_btn("Last ⏭")

        self._btn_first.setToolTip("Go to first spike")
        self._btn_prev.setToolTip("Previous spike  (←)")
        self._btn_next.setToolTip("Next spike  (→)")
        self._btn_last.setToolTip("Go to last spike")

        self._btn_first.clicked.connect(self._nav_first)
        self._btn_prev.clicked.connect(self._nav_prev)
        self._btn_next.clicked.connect(self._nav_next)
        self._btn_last.clicked.connect(self._nav_last)

        # Jump combo
        self._jump_label = QLabel("Jump:")
        self._jump_combo = QComboBox()
        self._jump_combo.setFixedWidth(200)
        self._jump_combo.setToolTip("Jump to a pre-computed interesting moment")
        self._jump_combo.currentIndexChanged.connect(self._on_jump_selected)

        row_a.addWidget(self._btn_first)
        row_a.addWidget(self._btn_prev)
        row_a.addWidget(self._btn_next)
        row_a.addWidget(self._btn_last)
        row_a.addSpacing(12)
        row_a.addWidget(self._jump_label)
        row_a.addWidget(self._jump_combo)
        row_a.addStretch()

        # Spike counter label
        self._spike_counter = QLabel("–")
        self._spike_counter.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._spike_counter.setMinimumWidth(140)
        row_a.addWidget(self._spike_counter)

        vbox.addLayout(row_a)

        # ── Row B: Window + time controls ────────────────────────────────────
        row_b = QHBoxLayout()
        row_b.setSpacing(4)

        # Window duration presets
        row_b.addWidget(QLabel("Window:"))
        self._duration_combo = QComboBox()
        for d in self._WINDOW_PRESETS:
            self._duration_combo.addItem(f"{d:g} s", d)
        self._duration_combo.setCurrentIndex(2)  # default: 2 s
        self._duration_combo.setFixedWidth(72)
        self._duration_combo.currentIndexChanged.connect(self._on_duration_changed)
        row_b.addWidget(self._duration_combo)

        row_b.addSpacing(16)

        # Direct time entry
        row_b.addWidget(QLabel("Go to (s):"))
        self._time_spin = QDoubleSpinBox()
        self._time_spin.setRange(0.0, 99999.0)
        self._time_spin.setDecimals(3)
        self._time_spin.setSingleStep(self.window_duration)
        self._time_spin.setFixedWidth(100)
        self._time_spin.setToolTip("Type a time in seconds and press Enter")
        self._time_spin.editingFinished.connect(self._on_time_spin_committed)
        row_b.addWidget(self._time_spin)

        # ± window step buttons
        self._btn_time_back = self._small_btn("← 1×")
        self._btn_time_fwd = self._small_btn("1× →")
        self._btn_time_back.setToolTip("Step back by one window duration")
        self._btn_time_fwd.setToolTip("Step forward by one window duration")
        self._btn_time_back.clicked.connect(self._nav_time_back)
        self._btn_time_fwd.clicked.connect(self._nav_time_fwd)
        row_b.addWidget(self._btn_time_back)
        row_b.addWidget(self._btn_time_fwd)

        row_b.addStretch()

        # Time readout label
        self._time_label = QLabel("—")
        self._time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._time_label.setMinimumWidth(160)
        row_b.addWidget(self._time_label)

        vbox.addLayout(row_b)

        # ── Row C: Display toggles ────────────────────────────────────────────
        row_c = QHBoxLayout()
        row_c.setSpacing(12)

        self._chk_neighbours = QCheckBox("Neighbour ch.")
        self._chk_neighbours.setChecked(
            True
        )  # on by default — key context for spike QC
        self._chk_neighbours.setToolTip(
            "Show the 2 nearest channels stacked above (dimmed)"
        )
        self._chk_neighbours.stateChanged.connect(self._on_toggle_neighbours)

        self._chk_other_units = QCheckBox("Other units")
        self._chk_other_units.setChecked(True)
        self._chk_other_units.setToolTip(
            "Show spike tick marks for all other units on this channel"
        )
        self._chk_other_units.stateChanged.connect(self._on_toggle_other_units)

        row_c.addWidget(self._chk_neighbours)
        row_c.addWidget(self._chk_other_units)
        row_c.addStretch()

        # Colour legend – dynamically rebuilt by _rebuild_unit_legend
        self._legend_widget = QWidget()
        self._legend_layout = QHBoxLayout(self._legend_widget)
        self._legend_layout.setContentsMargins(0, 0, 0, 0)
        self._legend_layout.setSpacing(8)
        row_c.addWidget(self._legend_widget)

        vbox.addLayout(row_c)

        return container

    @staticmethod
    def _small_btn(text: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedHeight(24)
        return btn

    # ─────────────────────────────────────────────────────────────────────────
    # Data helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _dm_has_raw(dm) -> bool:
        if dm is None:
            return False
        return (
            getattr(dm, "raw_reader", None) is not None
            or getattr(dm, "raw_data_memmap", None) is not None
        )

    def _resolve_channels(self, dm, cluster_id: int) -> tuple[int, list[int]]:
        """
        Return (dom_chan, neighbour_chans).

        Priority:
          1. features['median_ei'] peak-to-peak (pre-computed EI)
          2. templates.npy PTP argmax
          3. fall back to channel 0
        """
        features = dm.get_lightweight_features(cluster_id)
        if features is not None:
            try:
                median_ei = features["median_ei"]
                p2p = median_ei.max(axis=1) - median_ei.min(axis=1)
                dom_chan = int(np.argmax(p2p))
                neighbours = [
                    c
                    for c in dm.get_nearest_channels(dom_chan, n_channels=3)
                    if c != dom_chan
                ]
                return dom_chan, neighbours
            except Exception:
                pass

        # Fallback: templates
        try:
            templates = getattr(dm, "templates", None)
            if templates is not None and cluster_id < templates.shape[0]:
                ptp = templates[cluster_id].max(axis=0) - templates[cluster_id].min(
                    axis=0
                )
                dom_chan = int(np.argmax(ptp))
                neighbours = [
                    c
                    for c in dm.get_nearest_channels(dom_chan, n_channels=3)
                    if c != dom_chan
                ]
                return dom_chan, neighbours
        except Exception:
            pass

        return 0, []

    def _collect_other_unit_spikes(
        self,
        dm,
        selected_id: int,
        dom_chan: int,
    ) -> tuple[dict[int, np.ndarray], list[int]]:
        """
        Return spike times (seconds) for every cluster whose Kilosort template
        has non-zero amplitude on *exactly* dom_chan.

        Strictly filters by dom_chan only — units that appear on neighbouring
        channels but NOT on dom_chan are excluded.  This keeps the overlay
        honest: you see only spikes that would realistically be visible on the
        trace you are looking at.
        """
        chan_to_templates = getattr(dm, "channel_to_templates", {})
        if not chan_to_templates:
            return {}, []

        # cluster_ids whose template is active on dom_chan
        candidate_ids = chan_to_templates.get(dom_chan, np.array([]))
        if len(candidate_ids) == 0:
            return {}, []

        cluster_ids_on_chan: list[int] = []
        for tid in candidate_ids:
            cid = int(tid)
            if cid == selected_id:
                continue
            # Verify the cluster actually has spikes in the index
            if (
                dm.cluster_spike_indices.get(cid) is not None
                and len(dm.cluster_spike_indices[cid]) > 0
            ):
                cluster_ids_on_chan.append(cid)

        # Sort by spike count descending so the most-active units get
        # the most visible palette slots (1, 2, 3 …)
        cluster_ids_on_chan.sort(
            key=lambda c: len(dm.cluster_spike_indices.get(c, [])),
            reverse=True,
        )

        other_spikes: dict[int, np.ndarray] = {}
        for cid in cluster_ids_on_chan:
            samps = dm.get_cluster_spikes(cid)
            if len(samps) > 0:
                other_spikes[cid] = samps.astype(np.float64) / dm.sampling_rate

        return other_spikes, list(other_spikes.keys())

    # ─────────────────────────────────────────────────────────────────────────
    # Pre-computation helpers (called once per cluster load)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_jump_targets(self) -> None:
        """
        Pre-compute interesting time points for the jump combo.

        Targets:
          • First / Last spike (trivial)
          • Highest firing-rate window  (10 s sliding bin)
          • Largest amplitude spike     (from spike_amplitudes if available)
          • Most isolated spike         (spike with largest pre-ISI gap)
        """
        st = self.spike_times_sec
        if len(st) == 0:
            self._jump_targets = {}
            return

        targets: dict[str, float] = {}

        # ── First / Last ──────────────────────────────────────────────────────
        targets["First spike"] = float(st[0])
        targets["Last spike"] = float(st[-1])

        # ── Highest FR window (10 s sliding window with 1 s step) ─────────────
        try:
            bin_size = 1.0  # 1 s bins
            window_s = 10.0  # 10 s sliding window
            rec_dur = self._recording_duration
            if rec_dur > window_s:
                bins = np.arange(0.0, rec_dur + bin_size, bin_size)
                counts, _ = np.histogram(st, bins=bins)
                # convolve with window to get 10-bin rolling sum
                kernel = np.ones(int(window_s / bin_size))
                rolling_count = np.convolve(counts, kernel, mode="valid")
                best_bin = int(np.argmax(rolling_count))
                # centre of the best window
                best_t = bins[best_bin] + window_s / 2.0
                best_fr = rolling_count[best_bin] / window_s
                targets[f"Highest FR ({best_fr:.1f} Hz)"] = float(best_t)
        except Exception:
            pass

        # ── Largest amplitude spike ───────────────────────────────────────────
        if self.spike_amplitudes.size == len(st) and self.spike_amplitudes.size > 0:
            try:
                idx_max = int(np.argmax(self.spike_amplitudes))
                targets["Largest amplitude spike"] = float(st[idx_max])
            except Exception:
                pass

        # ── Most isolated spike (largest pre-ISI) ─────────────────────────────
        if len(st) > 1:
            try:
                isis = np.diff(st)
                # Use the geometric mean of pre and post ISI for isolation
                pre_isi = np.concatenate(([isis[0]], isis[:-1]))
                post_isi = np.concatenate((isis[1:], [isis[-1]]))
                isolation = pre_isi + post_isi
                idx_iso = int(np.argmax(isolation))
                targets["Most isolated spike"] = float(st[idx_iso])
            except Exception:
                pass

        self._jump_targets = targets
        logger.debug(
            "Jump targets computed for cluster %s: %s",
            self.current_cluster_id,
            list(targets.keys()),
        )

    def _compute_minimap_fr(self) -> None:
        """Bin spike times into _MINIMAP_FR_BINS evenly spaced bins."""
        st = self.spike_times_sec
        rec_dur = self._recording_duration
        if len(st) == 0 or rec_dur <= 0:
            self._minimap_fr_bins = None
            self._minimap_fr_rate = None
            return

        bins = np.linspace(0.0, rec_dur, _MINIMAP_FR_BINS + 1)
        counts, _ = np.histogram(st, bins=bins)
        bin_width = rec_dur / _MINIMAP_FR_BINS

        # Rate in Hz, lightly smoothed (Gaussian σ = 3 bins)
        from scipy.ndimage import gaussian_filter1d

        rate = gaussian_filter1d(counts.astype(float) / max(bin_width, 1e-9), sigma=3)

        self._minimap_fr_bins = (bins[:-1] + bins[1:]) / 2.0
        self._minimap_fr_rate = rate

    # ─────────────────────────────────────────────────────────────────────────
    # Minimap rendering
    # ─────────────────────────────────────────────────────────────────────────

    def _rebuild_minimap(self) -> None:
        """Rebuild FR density curve and zero baseline fill in the minimap."""
        if self._minimap_fr_bins is None or len(self._minimap_fr_bins) == 0:
            return

        x = self._minimap_fr_bins
        y = self._minimap_fr_rate
        rec = self._recording_duration

        self._mm_line.setData(x, y)

        # Rebuild fill (zero baseline)
        lower = pg.PlotDataItem(x, np.zeros_like(y))
        upper = pg.PlotDataItem(x, y)
        self._mm_fill.setCurves(lower, upper)

        self._minimap_plot.setXRange(0, rec, padding=0.01)
        self._minimap_plot.setYRange(0, float(np.max(y)) * 1.1 + 1e-9, padding=0)

        # Restore viewport rectangle
        self._sync_minimap_region()

    def _sync_minimap_region(self) -> None:
        """Move the minimap viewport rect and position marker to match the current window."""
        half = self.window_duration / 2.0
        lo = self.window_center - half
        hi = self.window_center + half
        self._mm_region.blockSignals(True)
        self._mm_region.setRegion((lo, hi))
        self._mm_region.blockSignals(False)
        self._mm_pos_line.setValue(self.window_center)

    def _on_minimap_region_moved(self) -> None:
        """Dragging the viewport rect in the minimap → navigate main view."""
        lo, hi = self._mm_region.getRegion()
        center = (lo + hi) / 2.0
        self._request_load(center, self.window_duration)

    def _on_minimap_clicked(self, event) -> None:
        """Single click anywhere on the minimap jumps main view to that time."""
        if event.button() != Qt.LeftButton:
            return
        pos = event.scenePos()
        vb = self._minimap_plot.getViewBox()
        if not vb.sceneBoundingRect().contains(pos):
            return
        data_pt = vb.mapSceneToView(pos)
        self._request_load(data_pt.x(), self.window_duration)

    # ─────────────────────────────────────────────────────────────────────────
    # Navigation slots
    # ─────────────────────────────────────────────────────────────────────────

    def _nav_first(self) -> None:
        if len(self.spike_times_sec) == 0:
            return
        self.current_spike_index = 0
        self._request_load(float(self.spike_times_sec[0]), self.window_duration)
        self._update_info_label()

    def _nav_last(self) -> None:
        if len(self.spike_times_sec) == 0:
            return
        self.current_spike_index = len(self.spike_times_sec) - 1
        self._request_load(float(self.spike_times_sec[-1]), self.window_duration)
        self._update_info_label()

    def _nav_prev(self) -> None:
        """
        Go to the spike immediately before the LEFT edge of the current view.
        This ensures Prev always moves backward from what is visible, regardless
        of how the user arrived at the current position (panning, jumping, etc.).
        """
        if len(self.spike_times_sec) == 0:
            return
        left_edge = self.window_center - self.window_duration / 2.0
        # Find the last spike whose time is strictly before the left edge
        candidates = np.where(self.spike_times_sec < left_edge - 1e-9)[0]
        if len(candidates) == 0:
            return  # already at or before the first spike — nothing to do
        self.current_spike_index = int(candidates[-1])
        self._request_load(
            float(self.spike_times_sec[self.current_spike_index]),
            self.window_duration,
        )
        self._update_info_label()

    def _nav_next(self) -> None:
        """
        Go to the spike immediately after the RIGHT edge of the current view.
        This ensures Next always advances forward from what is visible, regardless
        of how the user arrived at the current position.
        """
        if len(self.spike_times_sec) == 0:
            return
        right_edge = self.window_center + self.window_duration / 2.0
        # Find the first spike whose time is strictly after the right edge
        candidates = np.where(self.spike_times_sec > right_edge + 1e-9)[0]
        if len(candidates) == 0:
            return  # already at or past the last spike — nothing to do
        self.current_spike_index = int(candidates[0])
        self._request_load(
            float(self.spike_times_sec[self.current_spike_index]),
            self.window_duration,
        )
        self._update_info_label()

    def _nav_time_back(self) -> None:
        self._request_load(
            self.window_center - self.window_duration, self.window_duration
        )

    def _nav_time_fwd(self) -> None:
        self._request_load(
            self.window_center + self.window_duration, self.window_duration
        )

    def _on_jump_selected(self, index: int) -> None:
        if self._updating_controls:
            return
        data = self._jump_combo.itemData(index)
        if isinstance(data, float):
            self._request_load(data, self.window_duration)
            # Update spike index to nearest spike
            if len(self.spike_times_sec) > 0:
                self.current_spike_index = int(
                    np.argmin(np.abs(self.spike_times_sec - data))
                )
                self._update_info_label()

    def _on_duration_changed(self, _index: int) -> None:
        if self._updating_controls:
            return
        dur = self._duration_combo.currentData()
        if dur and dur != self.window_duration:
            self.window_duration = float(dur)
            self._time_spin.setSingleStep(self.window_duration)
            self._request_load(self.window_center, self.window_duration)

    def _on_time_spin_committed(self) -> None:
        """User typed a time and pressed Enter."""
        t = self._time_spin.value()
        if len(self.spike_times_sec) > 0:
            self.current_spike_index = int(np.argmin(np.abs(self.spike_times_sec - t)))
            self._update_info_label()
        self._request_load(t, self.window_duration)

    def _on_main_view_scrolled(self, _vb, x_range: list) -> None:
        """
        User panned the main plot with the mouse.
        Sync the minimap region and time label.
        """
        lo, hi = x_range
        center = (lo + hi) / 2.0
        self.window_center = center
        self._sync_minimap_region()
        self._update_time_label()

    def _on_toggle_neighbours(self, _state: int) -> None:
        show = self._chk_neighbours.isChecked()
        if not show:
            self._trace_n1.hide()
            self._trace_n2.hide()
        else:
            # Trigger re-load so workers fetch neighbour channels
            if self.current_cluster_id is not None:
                self._request_load(self.window_center, self.window_duration)

    def _on_toggle_other_units(self, _state: int) -> None:
        show = self._chk_other_units.isChecked()
        if not show:
            for item in self._other_spike_plots:
                item.hide()
        else:
            # Re-render the current window (spike data is already in memory)
            self._redraw_spike_overlays_from_cache()

    # ─────────────────────────────────────────────────────────────────────────
    # Spike click identification
    # ─────────────────────────────────────────────────────────────────────────

    def _on_main_plot_clicked(self, event) -> None:
        """
        Click anywhere on the main trace → find the nearest spike within a
        reasonable time tolerance and show a floating info label identifying
        its cluster, KSLabel, total spike count, and ISI violation %.
        """
        if event.button() != Qt.LeftButton:
            return
        if self.current_cluster_id is None:
            return

        pos = event.scenePos()
        vb = self._main_plot.getViewBox()
        if not vb.sceneBoundingRect().contains(pos):
            return

        data_pt = vb.mapSceneToView(pos)
        click_t = float(data_pt.x())

        # tolerance = 2 % of current window width
        tol = self.window_duration * 0.02

        # Search across all units in the current window: selected + others
        half = self.window_duration / 2.0
        start_t = self.window_center - half
        end_t = self.window_center + half

        best_dt = tol
        best_cid = None
        best_t = None

        # Selected unit
        mask = (self.spike_times_sec >= start_t) & (self.spike_times_sec <= end_t)
        win_st = self.spike_times_sec[mask]
        if len(win_st) > 0:
            dt = np.abs(win_st - click_t)
            idx = int(np.argmin(dt))
            if dt[idx] < best_dt:
                best_dt = dt[idx]
                best_cid = self.current_cluster_id
                best_t = float(win_st[idx])

        # Other units
        for other_cid, other_spikes in self.other_unit_spikes.items():
            win_os = other_spikes[(other_spikes >= start_t) & (other_spikes <= end_t)]
            if len(win_os) == 0:
                continue
            dt = np.abs(win_os - click_t)
            idx = int(np.argmin(dt))
            if dt[idx] < best_dt:
                best_dt = dt[idx]
                best_cid = other_cid
                best_t = float(win_os[idx])

        if best_cid is None:
            self._spike_info_label.hide()
            return

        # Gather info from data_manager
        info = self._get_cluster_info(best_cid)
        color_idx = (
            0
            if best_cid == self.current_cluster_id
            else (
                self.other_unit_ids.index(best_cid) + 1
                if best_cid in self.other_unit_ids
                else 1
            )
        )
        r, g, b, _ = _unit_color(color_idx, 255)

        lines = [
            f"C{best_cid}  t={best_t:.4f}s",
            f"label: {info['label']}   n={info['n_spikes']}",
            f"ISI viol: {info['isi']:.2f}%",
        ]
        self._spike_info_label.setText("\n".join(lines))
        self._spike_info_label.setColor((r, g, b))

        # Position label near the click, nudged so it stays inside the view
        x_range = vb.viewRange()[0]
        y_range = vb.viewRange()[1]
        x_frac = (click_t - x_range[0]) / max(x_range[1] - x_range[0], 1e-9)
        anchor_x = 0 if x_frac < 0.7 else 1
        anchor_y = (
            0
            if (data_pt.y() - y_range[0]) / max(y_range[1] - y_range[0], 1e-9) < 0.8
            else 1
        )
        self._spike_info_label.setAnchor((anchor_x, anchor_y))
        self._spike_info_label.setPos(click_t, float(data_pt.y()))
        self._spike_info_label.show()

        # Auto-hide after 4 seconds
        self._info_hide_timer.start()

    def _get_cluster_info(self, cluster_id: int) -> dict:
        """Pull label, n_spikes, ISI from cluster_df. Safe — never raises."""
        dm = self.main_window.data_manager
        out = {"label": "?", "n_spikes": 0, "isi": 0.0}
        try:
            row = dm.cluster_df[dm.cluster_df["cluster_id"] == cluster_id]
            if not row.empty:
                r = row.iloc[0]
                # KSLabel or group column
                for col in ("KSLabel", "group", "status"):
                    if col in dm.cluster_df.columns:
                        out["label"] = str(r.get(col, "?"))
                        break
                out["n_spikes"] = int(r.get("n_spikes", 0))
                out["isi"] = float(r.get("isi_violations_pct", 0.0))
        except Exception:
            pass
        return out

    def _request_load(self, center: float, duration: float) -> None:
        """
        Clamp the requested center to valid recording bounds, then debounce
        before dispatching the worker.
        """
        if self.current_cluster_id is None:
            return

        dm = self.main_window.data_manager
        rec_dur = self._recording_duration or (
            getattr(dm, "n_samples", 0) / max(dm.sampling_rate, 1)
        )
        half = duration / 2.0
        center = float(np.clip(center, half, max(rec_dur - half, half)))

        self._pending_center = center
        self._pending_duration = duration
        self.window_center = center
        self.window_duration = duration

        self._update_time_label()
        self._sync_minimap_region()
        self._nav_debounce.start()

    def _execute_load(self) -> None:
        """Debounce fired – stop any running worker, start a new one."""
        if self.current_cluster_id is None:
            return

        center = self._pending_center
        duration = self._pending_duration
        half = duration / 2.0
        start_t = max(0.0, center - half)
        end_t = center + half

        self._stop_worker()

        dm = self.main_window.data_manager
        load_neighbours = self._chk_neighbours.isChecked()

        worker = _RawLoadWorker(
            dm,
            self.current_cluster_id,
            self.dom_chan,
            self.neighbour_chans,
            start_t,
            end_t,
            load_neighbours=load_neighbours,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        worker.finished.connect(self._on_data_loaded)
        worker.error.connect(self._on_data_error)
        thread.started.connect(worker.run)

        self._worker = worker
        self._worker_thread = thread
        thread.start()

    def _stop_worker(self) -> None:
        if self._worker_thread and self._worker_thread.isRunning():
            self._worker_thread.quit()
            if not self._worker_thread.wait(2000):
                logger.warning("RawPanel worker did not stop cleanly; terminating")
                self._worker_thread.terminate()
                self._worker_thread.wait(500)
        self._worker_thread = None
        self._worker = None

    # ─────────────────────────────────────────────────────────────────────────
    # Worker finished → render
    # ─────────────────────────────────────────────────────────────────────────

    def _on_data_loaded(
        self,
        cluster_id: int,
        raw_data: np.ndarray,
        channels: list,
        start_t: float,
        end_t: float,
    ) -> None:
        """Runs on the main thread (via Qt signal). Renders the loaded data."""
        # Guard: stale result from a previous selection
        if cluster_id != self.current_cluster_id:
            return

        try:
            dm = self.main_window.data_manager
            dm.sampling_rate
            n = raw_data.shape[1]
            t = np.linspace(start_t, end_t, n)

            # ── Compute shared vertical scale from dom trace ───────────────────
            dom_row = 0  # row 0 is always dom_chan (see worker)
            dom_data = raw_data[dom_row]

            # Use robust range: 5th–95th percentile to avoid huge outlier spikes
            # dominating the scale, then add headroom.
            p05, p95 = float(np.percentile(dom_data, 5)), float(
                np.percentile(dom_data, 95)
            )
            trace_range = max(p95 - p05, 10.0)
            offset_step = trace_range * 1.4  # gap between stacked traces

            # ── Dom channel trace ─────────────────────────────────────────────
            self._trace_dom.setData(t, dom_data)

            # ── Neighbour traces ──────────────────────────────────────────────
            show_nb = self._chk_neighbours.isChecked() and len(channels) > 1

            if show_nb and len(channels) > 1:
                nb1_data = raw_data[1] + offset_step  # one step above dom
                self._trace_n1.setData(t, nb1_data)
                self._trace_n1.show()
            else:
                self._trace_n1.hide()

            if show_nb and len(channels) > 2:
                nb2_data = raw_data[2] + offset_step * 2  # two steps above dom
                self._trace_n2.setData(t, nb2_data)
                self._trace_n2.show()
            else:
                self._trace_n2.hide()

            # ── Channel labels (left edge of plot, at each trace baseline) ────
            label_x = start_t + (end_t - start_t) * 0.002  # 0.2% from left
            self._chan_label_dom.setText(f" ch{self.dom_chan}")
            self._chan_label_dom.setPos(label_x, float(np.mean(dom_data)))
            self._chan_label_dom.show()

            if show_nb and len(channels) > 1:
                ch1 = channels[1]
                self._chan_label_n1.setText(f" ch{ch1}")
                self._chan_label_n1.setPos(
                    label_x, float(np.mean(raw_data[1])) + offset_step
                )
                self._chan_label_n1.show()
            else:
                self._chan_label_n1.hide()

            if show_nb and len(channels) > 2:
                ch2 = channels[2]
                self._chan_label_n2.setText(f" ch{ch2}")
                self._chan_label_n2.setPos(
                    label_x, float(np.mean(raw_data[2])) + offset_step * 2
                )
                self._chan_label_n2.show()
            else:
                self._chan_label_n2.hide()

            # ── Determine Y range ─────────────────────────────────────────────
            # Base: dom trace min/max + padding
            y_min = float(np.min(dom_data))
            y_max = float(np.max(dom_data))
            y_pad = (y_max - y_min) * 0.15 or 10.0

            if show_nb:
                n_stacked = min(len(channels) - 1, 2)
                # Top of the highest neighbour trace
                top_trace_max = (
                    float(np.max(raw_data[n_stacked])) + offset_step * n_stacked
                )
                plot_y_max = top_trace_max + y_pad
            else:
                plot_y_max = y_max + y_pad

            # Reserve a strip below y_min for other-unit ticks
            tick_strip_h = (plot_y_max - y_min + 2 * y_pad) * _TICK_STRIP_FRAC
            tick_h = (plot_y_max - y_min + 2 * y_pad) * _TICK_HEIGHT_FRAC
            plot_y_min = y_min - y_pad - tick_strip_h

            self._main_plot.setXRange(start_t, end_t, padding=0)
            self._main_plot.setYRange(plot_y_min, plot_y_max, padding=0)

            # Cache strip coords for spike rendering and click detection
            self._tick_bottom = plot_y_min
            self._tick_top = y_min - y_pad
            self._tick_h = tick_h
            self._y_min_plot = plot_y_min
            self._y_max_plot = plot_y_max
            # Cache raw data for click-identification (dom channel only needed)
            self._last_start_t = start_t
            self._last_end_t = end_t
            self._last_raw_dom = dom_data
            self._last_channels = channels

            # Update title
            nb_str = ""
            if show_nb and len(channels) > 1:
                nb_str = "  +  ch" + ", ch".join(str(c) for c in channels[1:])
            self._main_plot.setTitle(
                f"Cluster {cluster_id}  –  ch {self.dom_chan}{nb_str}  "
                f"  {start_t:.3f}s – {end_t:.3f}s",
                size="10pt",
            )

            # ── Spike overlays ────────────────────────────────────────────────
            self._render_spike_overlays(
                start_t,
                end_t,
                y_min=plot_y_min,
                y_max=plot_y_max,
                tick_bottom=self._tick_bottom,
            )

            # ── Update label ──────────────────────────────────────────────────
            self._update_time_label_range(start_t, end_t)
            self.status_message.emit(
                f"Cluster {cluster_id} · ch {self.dom_chan} · "
                f"{start_t:.3f}s–{end_t:.3f}s"
            )

        except Exception as exc:
            logger.exception("RawPanel render error: %s", exc)
            self.error_message.emit(f"Render error: {exc}")
        finally:
            self._cleanup_worker_thread()

    def _cleanup_worker_thread(self) -> None:
        if self._worker_thread:
            self._worker_thread.quit()
            if not self._worker_thread.wait(2000):
                logger.warning("RawPanel worker cleanup timed out; terminating")
                self._worker_thread.terminate()
                self._worker_thread.wait(500)
            self._worker_thread = None
            self._worker = None

    def _on_data_error(self, msg: str) -> None:
        logger.error("RawPanel worker error: %s", msg)
        self.error_message.emit(msg)
        self._cleanup_worker_thread()

    # ─────────────────────────────────────────────────────────────────────────
    # Spike overlay rendering
    # ─────────────────────────────────────────────────────────────────────────

    def _render_spike_overlays(
        self,
        start_t: float,
        end_t: float,
        y_min: float,
        y_max: float,
        tick_bottom: float,
    ) -> None:
        """
        Draw:
          1. Selected-unit full-height vertical markers
          2. Other-unit tick marks in the strip below the trace
        """
        self.main_window.data_manager

        # ── Selected-unit spikes ──────────────────────────────────────────────
        sel_in_win = self.spike_times_sec[
            (self.spike_times_sec >= start_t) & (self.spike_times_sec <= end_t)
        ]
        self._sel_spike_plot.setData(*self._make_vline_data(sel_in_win, y_min, y_max))
        color = _unit_color(0, _ALPHA_SELECTED)
        self._sel_spike_plot.setPen(pg.mkPen(color, width=1.2))

        # ── Other-unit ticks ──────────────────────────────────────────────────
        show_others = self._chk_other_units.isChecked()
        for slot_idx, plot_item in enumerate(self._other_spike_plots):
            if not show_others or slot_idx >= len(self.other_unit_ids):
                plot_item.hide()
                continue

            other_cid = self.other_unit_ids[slot_idx]
            other_spikes = self.other_unit_spikes.get(other_cid)
            if other_spikes is None or len(other_spikes) == 0:
                plot_item.hide()
                continue

            in_win = other_spikes[(other_spikes >= start_t) & (other_spikes <= end_t)]
            if len(in_win) == 0:
                plot_item.hide()
                continue

            tick_x, tick_y = self._make_vline_data(
                in_win, tick_bottom, tick_bottom + self._tick_h
            )
            color = _unit_color(slot_idx + 1, _ALPHA_OTHER)
            plot_item.setData(tick_x, tick_y)
            plot_item.setPen(pg.mkPen(color, width=1))
            plot_item.show()

        self._rebuild_unit_legend(start_t, end_t)

    def _redraw_spike_overlays_from_cache(self) -> None:
        """Re-render spike overlays using cached window bounds (no I/O)."""
        if not hasattr(self, "_tick_bottom"):
            return
        half = self.window_duration / 2.0
        start_t = self.window_center - half
        end_t = self.window_center + half
        self._render_spike_overlays(
            start_t,
            end_t,
            y_min=self._y_min_plot,
            y_max=self._y_max_plot,
            tick_bottom=self._tick_bottom,
        )

    @staticmethod
    def _make_vline_data(
        times: np.ndarray, y_bot: float, y_top: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build (x, y) arrays for a set of vertical line segments, using NaN
        breaks so a single PlotDataItem renders all lines efficiently.

        Each spike t produces: [(t, y_bot), (t, y_top), (t, NaN)]
        """
        n = len(times)
        if n == 0:
            return np.array([]), np.array([])

        x = np.empty(n * 3)
        y = np.empty(n * 3)
        x[0::3] = times
        x[1::3] = times
        x[2::3] = times
        y[0::3] = y_bot
        y[1::3] = y_top
        y[2::3] = np.nan
        return x, y

    # ─────────────────────────────────────────────────────────────────────────
    # Legend
    # ─────────────────────────────────────────────────────────────────────────

    def _rebuild_unit_legend(self, start_t: float, end_t: float) -> None:
        """
        Compact colour-coded legend for units visible in the current window.

        Layout per unit:  ●  C{id}
        - Dot colour matches the spike-overlay colour for that unit.
        - The selected unit is shown first, bold.
        - Spike count-in-window shown as a tooltip on hover, keeping the
          bar uncluttered.  Total spike count for that unit is also included.
        - Units with zero spikes in the current window are omitted entirely.
        """
        # --- clear previous entries ---
        while self._legend_layout.count():
            item = self._legend_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.main_window.data_manager

        def _ks_badge(label: str) -> str:
            """Return a small inline HTML badge for a KSLabel."""
            label_lc = str(label).lower()
            if label_lc == "good":
                bg, fg = "#1A5C3A", "#6EE7B7"
            elif label_lc == "mua":
                bg, fg = "#5C3A00", "#F0C060"
            elif label_lc in ("noise", "bad"):
                bg, fg = "#5C1A1A", "#F08080"
            else:
                bg, fg = "#2A2D36", "#9B9DA6"
            short = label_lc[:3].upper()
            return (
                f"<span style='background:{bg};color:{fg};"
                f"border-radius:2px;padding:0 2px;font-size:9px;'>"
                f"{short}</span>"
            )

        def _make_swatch(
            color_idx: int,
            cluster_id: int,
            n_in_win: int,
            n_total: int,
            bold: bool = False,
        ) -> QLabel:
            r, g, b, _ = _unit_color(color_idx, 255)
            weight = "bold" if bold else "normal"
            info = self._get_cluster_info(cluster_id)
            badge = _ks_badge(info["label"])
            lbl = QLabel(
                f"<span style='color:rgb({r},{g},{b});'>●</span>"
                f"&nbsp;C{cluster_id}&nbsp;{badge}"
            )
            lbl.setStyleSheet(
                f"font-size: 11px; font-weight: {weight}; color: #C8C9CF;"
            )
            lbl.setToolTip(
                f"Cluster {cluster_id}  ({info['label']})\n"
                f"{n_in_win} spike{'s' if n_in_win != 1 else ''} in view\n"
                f"{n_total} total  ·  ISI viol {info['isi']:.2f}%"
            )
            return lbl

        # --- selected unit ---
        n_sel_win = int(
            np.sum((self.spike_times_sec >= start_t) & (self.spike_times_sec <= end_t))
        )
        if n_sel_win > 0 and self.current_cluster_id is not None:
            n_sel_total = len(self.spike_times_sec)
            self._legend_layout.addWidget(
                _make_swatch(
                    0, self.current_cluster_id, n_sel_win, n_sel_total, bold=True
                )
            )

        # --- other units (only if toggle is on, max 8 shown in legend) ---
        if self._chk_other_units.isChecked():
            shown = 0
            for slot_idx in range(len(self.other_unit_ids)):
                if shown >= 8:
                    break
                other_cid = self.other_unit_ids[slot_idx]
                other_spikes = self.other_unit_spikes.get(other_cid)
                if other_spikes is None:
                    continue
                n_in_win = int(
                    np.sum((other_spikes >= start_t) & (other_spikes <= end_t))
                )
                if n_in_win == 0:
                    continue
                n_total = len(other_spikes)
                self._legend_layout.addWidget(
                    _make_swatch(slot_idx + 1, other_cid, n_in_win, n_total, bold=False)
                )
                shown += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Control population helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _populate_jump_combo(self) -> None:
        self._updating_controls = True
        try:
            self._jump_combo.clear()
            self._jump_combo.addItem("— Jump to —", None)
            for label, t in self._jump_targets.items():
                self._jump_combo.addItem(f"{label}  ({t:.2f} s)", float(t))
        finally:
            self._updating_controls = False

    def _update_info_label(self) -> None:
        """Update spike counter (spike N / total  |  N in view)."""
        n_total = len(self.spike_times_sec)
        if n_total == 0:
            self._spike_counter.setText("No spikes")
            return
        idx = self.current_spike_index + 1
        half = self.window_duration / 2.0
        lo = self.window_center - half
        hi = self.window_center + half
        in_view = int(
            np.sum((self.spike_times_sec >= lo) & (self.spike_times_sec <= hi))
        )
        self._spike_counter.setText(f"Spike {idx} / {n_total}  ({in_view} in view)")

    def _update_time_label(self) -> None:
        if not hasattr(self, "_time_label"):
            return
        half = self.window_duration / 2.0
        lo = self.window_center - half
        hi = self.window_center + half
        self._time_label.setText(f"{lo:.3f} s  –  {hi:.3f} s")
        # Sync spin box without triggering its callback
        self._updating_controls = True
        try:
            self._time_spin.setValue(self.window_center)
        finally:
            self._updating_controls = False

    def _update_time_label_range(self, start_t: float, end_t: float) -> None:
        self._time_label.setText(f"{start_t:.3f} s  –  {end_t:.3f} s")

    # ─────────────────────────────────────────────────────────────────────────
    # Keyboard shortcuts (arrow keys forwarded from MainWindow's key forwarder)
    # ─────────────────────────────────────────────────────────────────────────

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Left:
            self._nav_prev()
        elif event.key() == Qt.Key_Right:
            self._nav_next()
        elif event.key() == Qt.Key_BracketLeft:
            self._nav_time_back()
        elif event.key() == Qt.Key_BracketRight:
            self._nav_time_fwd()
        else:
            super().keyPressEvent(event)

    # ─────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        self._nav_debounce.stop()
        self._info_hide_timer.stop()
        self._stop_worker()

    def closeEvent(self, event) -> None:
        self.cleanup()
        super().closeEvent(event)
