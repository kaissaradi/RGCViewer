"""The Raster tab: every spike of one cell, and the units around it (PLAN.md Q63).

Three views, the same for Kilosort and Vision-only runs:

* Firing over the recording (top): the cell's rate, stimulus blocks shaded.
  Click to bring that moment into the raster below.
* The raster (middle): the whole recording folded into rows, every spike a
  tick. Rows are one trial each when the run's triggers come in trials
  (the Vision .neurons TTLs, MATLAB load_stim's rule), else a fixed length.
  With a Lisp grating loaded, rows can be ordered by condition and a strip
  at the left gives each trial's direction. Double-click opens the Raw tab
  at that moment when a raw file is loaded.
* Nearby units (bottom): lanes for this cell and the units within 100 µm or
  with a similar template, ordered by EI (or template) similarity. The
  table gives distance, similarity and the share of this cell's spikes
  that a neighbour fires within ±0.5 ms (with the chance level): a
  duplicate shares most. Click a row for its cross-correlogram; double-click
  to select that cell.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QLabel,
                            QSplitter, QTableWidget, QTableWidgetItem, QToolTip, QVBoxLayout,
                            QWidget)

from ...analysis import raster_data as rd
from ..theme import apply_plot_theme, categorical, plot_field, resolve_theme_colors
from .raster_view import TICKS_BELOW_S, density, ticks

logger = logging.getLogger(__name__)

ROW_CHOICES = (("Auto", None), ("1 s", 1.0), ("2 s", 2.0), ("5 s", 5.0), ("10 s", 10.0),
               ("30 s", 30.0), ("60 s", 60.0), ("5 min", 300.0))
AUTO_MAX_ROWS = 200
MAX_TICK_SPIKES = 400_000      # above this the raster is drawn as a density image
MAX_GROUP_LANES = 400
DUPLICATE_SHARED = 0.2         # shared ≥ 20 % and ≥ 10 × chance: "possible duplicate"


def lane_colours(n: int, colors) -> List[str]:
    """This cell red (the plot_compare colour), the others the categorical set without red."""
    out = [resolve_theme_colors(colors).get("plot_compare", "#E8564A")]
    k = 0
    while len(out) < n:
        if k % 12 != 2:                        # index 2 is the categorical red
            out.append(categorical(k, colors))
        k += 1
    return out[:n]


class RasterPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._colors = resolve_theme_colors(main_window.get_current_colors())
        self._cid: Optional[int] = None
        self._key = None
        self._times = np.zeros(0)          # this cell's spikes, s
        self._duration = 1.0
        self._rows_start: Optional[np.ndarray] = None   # row start times (s)
        self._row_span = 1.0
        self._row_order: Optional[np.ndarray] = None
        self._trials = None                 # (starts, span) from triggers
        self._lane_ids: List[int] = []
        self._lane_times: List[np.ndarray] = []
        self._neighbours: List[rd.Neighbour] = []
        self._pair_stats: Dict[int, dict] = {}
        self._nb_cache: Dict[tuple, List[rd.Neighbour]] = {}
        self._pending = None
        self._block_items: list = []
        self._signals = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)

        head = QHBoxLayout()
        title = QLabel("RASTER")
        title.setObjectName("mutedLabel")
        head.addWidget(title)
        self.cell_label = QLabel("Select a cell.")
        head.addWidget(self.cell_label, 1)
        head.addWidget(QLabel("Rows:"))
        self.rows_combo = QComboBox()
        self.rows_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.rows_combo.setToolTip("How the recording is folded into rows: one trial per row "
                                   "when the triggers show trials, else a fixed length.")
        self.rows_combo.activated.connect(lambda _i: self._refold())
        head.addWidget(self.rows_combo)
        self.order_combo = QComboBox()
        self.order_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.order_combo.addItems(["In time order", "By condition"])
        self.order_combo.setToolTip("By condition: trials grouped by the Lisp grating's "
                                    "temporal period and direction (File ▸ Load Stimulus File).")
        self.order_combo.activated.connect(lambda _i: self._refold())
        self.order_combo.hide()
        head.addWidget(self.order_combo)
        head.addSpacing(12)
        head.addWidget(QLabel("Lanes:"))
        self.lanes_combo = QComboBox()
        self.lanes_combo.addItems(["Nearby and similar units", "Its group", "All cells"])
        self.lanes_combo.activated.connect(lambda _i: self._fill_lanes())
        head.addWidget(self.lanes_combo)
        outer.addLayout(head)

        split = QSplitter(Qt.Orientation.Vertical)
        outer.addWidget(split, 1)

        # Firing over the recording
        self.rate_plot = pg.PlotWidget()
        self.rate_plot.setMenuEnabled(False)
        self.rate_plot.setLabel("left", "Rate (Hz)")
        self.rate_plot.getViewBox().setMouseEnabled(x=True, y=False)
        self.rate_fill = pg.FillBetweenItem(pg.PlotDataItem(), pg.PlotDataItem())
        self.rate_plot.addItem(self.rate_fill)
        self.rate_curve = self.rate_plot.plot()
        self.view_region = pg.LinearRegionItem(movable=False)
        self.view_region.setZValue(-5)
        self.rate_plot.addItem(self.view_region)
        self.rate_plot.scene().sigMouseClicked.connect(self._on_rate_clicked)
        split.addWidget(self.rate_plot)

        # The folded raster
        self.fold_plot = pg.PlotWidget()
        self.fold_plot.setMenuEnabled(False)
        self.fold_plot.invertY(True)
        self.fold_bg = pg.ImageItem()
        self.fold_bg.setZValue(-20)
        self.fold_plot.addItem(self.fold_bg)
        self.cond_strip = pg.ImageItem()
        self.cond_strip.setZValue(-10)
        self.fold_plot.addItem(self.cond_strip)
        self.fold_ticks = pg.PlotCurveItem(connect="pairs")
        self.fold_plot.addItem(self.fold_ticks)
        self.fold_image = pg.ImageItem()
        self.fold_plot.addItem(self.fold_image)
        self.fold_plot.scene().sigMouseMoved.connect(self._on_fold_hover)
        self.fold_plot.scene().sigMouseClicked.connect(self._on_fold_clicked)
        self.fold_plot.getViewBox().sigYRangeChanged.connect(lambda *_a: self._sync_region())
        split.addWidget(self.fold_plot)

        # Nearby units: lanes | table + CCG
        bottom = QWidget()
        row = QHBoxLayout(bottom)
        row.setContentsMargins(0, 0, 0, 0)
        self.lanes_plot = pg.PlotWidget()
        self.lanes_plot.setMenuEnabled(False)
        self.lanes_plot.invertY(True)
        self.lanes_plot.getViewBox().setMouseEnabled(x=True, y=False)
        self.lanes_plot.setLabel("bottom", "Time in the recording (s)")
        self.lanes_image = pg.ImageItem()
        self.lanes_plot.addItem(self.lanes_image)
        self._lane_curves: List[pg.PlotCurveItem] = []
        self.lanes_plot.scene().sigMouseClicked.connect(self._on_lane_clicked)
        self.lanes_plot.scene().sigMouseMoved.connect(self._on_lane_hover)
        self._lane_debounce = QTimer(self)
        self._lane_debounce.setSingleShot(True)
        self._lane_debounce.setInterval(80)
        self._lane_debounce.timeout.connect(self._render_lanes)
        self.lanes_plot.getViewBox().sigXRangeChanged.connect(lambda *_a: self._lane_debounce.start())
        row.addWidget(self.lanes_plot, 3)

        side = QVBoxLayout()
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Unit", "Distance", "Similarity", "Shared ±0.5 ms"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._on_table_pick)
        self.table.cellDoubleClicked.connect(lambda r, _c: self._select_neighbour(r))
        self.table.setToolTip("Units within 100 µm of this cell or with a similar Kilosort "
                              "template, most similar first. Shared: the share of this cell's "
                              "spikes the unit also fires within ±0.5 ms; chance in brackets. "
                              "Double-click to select it.")
        self.table.setMinimumHeight(130)
        side.addWidget(self.table, 2)
        self.ccg_title = QLabel("")
        self.ccg_title.setObjectName("mutedLabel")
        side.addWidget(self.ccg_title)
        self.ccg_plot = pg.PlotWidget()
        self.ccg_plot.setMinimumHeight(140)
        self.ccg_plot.setToolTip(
            "Cross-correlogram: the neighbour's firing rate around this cell's spikes.\n"
            "A narrow peak at 0 ms: the same spikes twice (a duplicate).\n"
            "A gap at 0 like a refractory period: possibly one cell split in two\n"
            "(sorting also leaves a short gap between units on one electrode).\n"
            "A broad peak of a few ms: cells that fire together.")
        self.ccg_plot.setMenuEnabled(False)
        self.ccg_plot.setLabel("bottom", "Lag from this cell's spikes (ms)")
        self.ccg_plot.setLabel("left", "Rate (Hz)")
        self.ccg_bars = pg.BarGraphItem(x=[], height=[], width=rd.CCG_BIN_S * 1000)
        self.ccg_plot.addItem(self.ccg_bars)
        self.ccg_base = pg.InfiniteLine(angle=0, movable=False)
        self.ccg_plot.addItem(self.ccg_base)
        side.addWidget(self.ccg_plot, 3)
        self.nb_note = QLabel("")
        self.nb_note.setObjectName("mutedLabel")
        self.nb_note.setWordWrap(True)
        side.addWidget(self.nb_note)
        sidew = QWidget()
        sidew.setLayout(side)
        row.addWidget(sidew, 2)
        split.addWidget(bottom)
        split.setSizes([110, 420, 340])

        self.restyle_plots(main_window.get_current_colors())

    # ── entry ─────────────────────────────────────────────────────────────────

    def update_all(self, cluster_id):
        dm = getattr(self.main_window, "data_manager", None)
        if dm is None or cluster_id is None or getattr(dm, "cluster_df", None) is None:
            return
        cid = int(cluster_id)
        key = (id(dm), getattr(dm, "generation", None), cid,
               id(getattr(dm, "grating_raw_data", None)))
        if key == self._key:
            return
        self._key = key
        self._cid = cid
        fs = float(dm.sampling_rate)
        s = dm.get_cluster_spikes(cid)
        self._times = np.sort(np.asarray(s, dtype=np.float64) / fs) if s is not None else np.zeros(0)
        n_samples = getattr(dm, "n_samples", 0) or 0
        spikes = getattr(dm, "spike_times", None)
        last = float(spikes[-1]) / fs if spikes is not None and len(spikes) else 0.0
        self._duration = max(float(n_samples) / fs, last, self._times[-1] if self._times.size else 1.0, 1.0)
        rate = self._times.size / self._duration
        self.cell_label.setText(
            f"Cell {cid} · {self._times.size:,} spikes · {rate:.1f} Hz over {self._duration:.0f} s")
        trig = dm.trigger_times_s() if hasattr(dm, "trigger_times_s") else None
        self._trials = rd.trial_structure(trig) if trig is not None else None
        self._fill_rows_combo()
        self._draw_rate()
        self._refold()
        self._fill_lanes()

    def restyle_plots(self, colors):
        c = resolve_theme_colors(colors)
        self._colors = c
        for p in (self.rate_plot, self.fold_plot, self.lanes_plot, self.ccg_plot):
            apply_plot_theme(p, c)
            p.showGrid(x=False, y=False)
        field, ink = QColor(plot_field(c)), QColor(c["text_primary"])
        self.lanes_image.setColorMap(pg.ColorMap([0.0, 1.0], [field, ink]))
        self.fold_image.setColorMap(pg.ColorMap([0.0, 1.0], [field, ink]))
        region = QColor(c.get("plot_highlight", "#4A82D6"))
        region.setAlpha(40)
        self.view_region.setBrush(pg.mkBrush(region))
        self.view_region.setRegion((0, 0))
        for line in self.view_region.lines:
            line.setPen(pg.mkPen(None))
        self.ccg_base.setPen(pg.mkPen(c["text_secondary"], style=Qt.PenStyle.DashLine))
        if self._cid is not None:
            self._key = None
            cid, self._cid = self._cid, None
            self.update_all(cid)

    # ── rows ──────────────────────────────────────────────────────────────────

    def _fill_rows_combo(self):
        keep = self.rows_combo.currentText()
        self.rows_combo.blockSignals(True)
        self.rows_combo.clear()
        if self._trials is not None:
            starts, span = self._trials
            self.rows_combo.addItem(f"Trials ({starts.size}, {span:.1f} s)", "trials")
        for label, val in ROW_CHOICES:
            self.rows_combo.addItem(label, val)
        i = self.rows_combo.findText(keep)
        if keep.startswith("Trials") and self._trials is not None:
            i = 0
        self.rows_combo.setCurrentIndex(max(0, i))
        self.rows_combo.blockSignals(False)

    def _auto_row_s(self) -> float:
        for _label, val in ROW_CHOICES[1:]:
            if self._duration / val <= AUTO_MAX_ROWS:
                return val
        return ROW_CHOICES[-1][1]

    def _trial_conditions(self):
        """Per-trial (temporal frequency, direction) from a Lisp grating, or None."""
        dm = self.main_window.data_manager
        src = getattr(dm, "grating_source", None)
        raw = getattr(dm, "grating_raw_data", None)
        if self._trials is None or not src or not raw:
            return None
        params = raw.get("trial_parameters") or []
        if not params or len(params) > self._trials[0].size:
            return None
        return [(float(p["temporalFrequency"]), float(p["orientation"])) for p in params]

    def _refold(self):
        mode = self.rows_combo.currentData()
        # "By condition" only means something with trial rows and a Lisp grating.
        self.order_combo.setVisible(mode == "trials" and self._trial_conditions() is not None)
        t = self._times
        conds = None
        if mode == "trials" and self._trials is not None:
            starts, span = self._trials
            x, rows, n = rd.fold_trials(t, starts, span)
            self._rows_start, self._row_span = starts, span
            conds = self._trial_conditions()
            self.fold_plot.setLabel("bottom", "Time from trial start (s)")
        else:
            row_s = float(mode) if mode else self._auto_row_s()
            x, rows, n = rd.fold_fixed(t, row_s, self._duration)
            self._rows_start, self._row_span = np.arange(n) * row_s, row_s
            self.fold_plot.setLabel("bottom", f"Time in the row (s) · rows of {row_s:g} s")
        order = np.arange(n)
        by_cond = conds is not None and self.order_combo.currentIndex() == 1
        if by_cond:
            key = [conds[i] if i < len(conds) else (np.inf, np.inf) for i in range(n)]
            order = np.array(sorted(range(n), key=lambda i: (key[i], i)))
        self._row_order = order
        pos = np.empty(n, dtype=np.int64)
        pos[order] = np.arange(n)
        y = pos[rows] if rows.size else rows
        self._draw_fold(x, y, n, conds, order)

    def _draw_fold(self, x, y, n, conds, order):
        c = self._colors
        span = self._row_span
        if x.size <= MAX_TICK_SPIKES:
            xs = np.repeat(x, 2)
            ys = np.empty(xs.size)
            ys[0::2], ys[1::2] = y + 0.12, y + 0.88
            self.fold_ticks.setData(xs, ys, pen=pg.mkPen(c["text_primary"], width=1))
            self.fold_ticks.show()
            self.fold_image.hide()
        else:
            nx = 1500
            img, _, _ = np.histogram2d(y, x, bins=(n, nx), range=((0, n), (0, span)))
            peak = np.percentile(img[img > 0], 99) if np.any(img > 0) else 1.0
            self.fold_image.setImage(np.clip(img / max(peak, 1.0), 0, 1).T, levels=(0, 1),
                                     autoLevels=False)
            self.fold_image.setRect(pg.QtCore.QRectF(0, 0, span, n))
            self.fold_image.show()
            self.fold_ticks.hide()
        # Row backgrounds: stimulus blocks, alternating shades.
        self._draw_row_blocks(n, order)
        # Condition strip: each trial's direction as a hue, left of zero.
        if conds is not None:
            strip = np.zeros((n, 1, 4), dtype=np.ubyte)
            for r, trial in enumerate(order):
                if trial < len(conds):
                    q = QColor.fromHsvF((conds[trial][1] % 360) / 360.0, 0.55, 0.85)
                    strip[r, 0] = (q.red(), q.green(), q.blue(), 255)
            self.cond_strip.setImage(strip.transpose(1, 0, 2), autoLevels=False)
            w = span * 0.025
            self.cond_strip.setRect(pg.QtCore.QRectF(-w, 0, w, n))
            self.cond_strip.show()
        else:
            self.cond_strip.hide()
        left = self.fold_plot.getAxis("left")
        step = max(1, int(np.ceil(n / 12)))
        if self.rows_combo.currentData() == "trials":
            if conds is not None and self.order_combo.currentIndex() == 1:
                # One label where each condition's block of trials starts.
                ticks_, prev = [], None
                for r, trial in enumerate(order):
                    cond = conds[trial] if trial < len(conds) else None
                    if cond is not None and cond != prev:
                        ticks_.append((r + 0.5, f"{cond[1]:g}° · {cond[0]:.3g} Hz"))
                    prev = cond
            else:
                ticks_ = [(r + 0.5, f"trial {order[r] + 1}") for r in range(0, n, step)]
            left.setLabel("")
        else:
            ticks_ = [(r + 0.5, f"{self._rows_start[order[r]]:.0f} s") for r in range(0, n, step)]
            left.setLabel("Row start")
        left.setTicks([ticks_])
        self.fold_plot.setXRange(-span * 0.03 if conds is not None else 0, span, padding=0)
        self.fold_plot.setYRange(0, n, padding=0)
        self.fold_plot.getViewBox().setLimits(yMin=0, yMax=max(n, 1))

    def _blocks(self):
        dm = self.main_window.data_manager
        try:
            from ...analysis import recording_timeline as rt
            if getattr(dm, "stimulus_manifest", None) is None and hasattr(dm, "load_stimulus_manifest"):
                dm.load_stimulus_manifest()
            spikes = getattr(dm, "spike_times", None)
            ks = getattr(dm, "kilosort_dir", None)
            if ks is None or spikes is None or not len(spikes):
                return []
            return rt.stimulus_blocks(ks.parent.name, dm.stimulus_manifest,
                                      int(spikes[-1]), float(dm.sampling_rate))
        except Exception:
            logger.debug("stimulus blocks unavailable", exc_info=True)
            return []

    def _draw_row_blocks(self, n, order):
        blocks = self._blocks() if self.rows_combo.currentData() != "trials" else []
        if len(blocks) < 2 or self._rows_start is None:
            self.fold_bg.hide()
            return
        shade = QColor(self._colors.get("plot_highlight", "#4A82D6"))
        img = np.zeros((n, 1, 4), dtype=np.ubyte)
        mids = self._rows_start + self._row_span / 2.0
        for i, b in enumerate(blocks):
            alpha = 34 if i % 2 == 0 else 12
            sel = (mids >= b.start_s) & (mids < b.end_s)
            rows = np.flatnonzero(sel[order]) if order is not None else np.flatnonzero(sel)
            img[rows, 0] = (shade.red(), shade.green(), shade.blue(), alpha)
        self.fold_bg.setImage(img.transpose(1, 0, 2), autoLevels=False)
        self.fold_bg.setRect(pg.QtCore.QRectF(0, 0, self._row_span, n))
        self.fold_bg.show()

    # ── rate ──────────────────────────────────────────────────────────────────

    def _draw_rate(self):
        c = self._colors
        tc, rate, _bin = rd.rate_curve(self._times, self._duration)
        colour = c.get("plot_fr", "#E0B000")
        self.rate_curve.setData(tc, rate, pen=pg.mkPen(colour, width=1.2))
        fill = QColor(colour)
        fill.setAlpha(50)
        self.rate_fill.setCurves(pg.PlotDataItem(tc, np.zeros_like(rate)), pg.PlotDataItem(tc, rate))
        self.rate_fill.setBrush(pg.mkBrush(fill))
        for item in self._block_items:
            self.rate_plot.removeItem(item)
        self._block_items = []
        for i, b in enumerate(self._blocks()):
            shade = QColor(c.get("plot_highlight", "#4A82D6"))
            shade.setAlpha(26 if i % 2 == 0 else 8)
            region = pg.LinearRegionItem((b.start_s, b.end_s), movable=False,
                                         brush=pg.mkBrush(shade), pen=pg.mkPen(None))
            region.setZValue(-10)
            label = pg.TextItem(b.protocol, color=c["text_secondary"], anchor=(0, 0))
            label.setPos(b.start_s, float(np.nanmax(rate)) if rate.size else 1.0)
            self.rate_plot.addItem(region)
            self.rate_plot.addItem(label)
            self._block_items += [region, label]
        self.rate_plot.setXRange(0, self._duration, padding=0)
        self.rate_plot.setYRange(0, max(1.0, float(np.nanmax(rate)) * 1.15 if rate.size else 1.0),
                                 padding=0)

    def _sync_region(self):
        """Shade on the rate plot the part of the recording the raster shows."""
        if self._rows_start is None or self._row_order is None or not self._rows_start.size:
            return
        _xr, (y0, y1) = self.fold_plot.getViewBox().viewRange()
        order = self._row_order
        lo, hi = max(0, int(np.floor(y0))), min(order.size, int(np.ceil(y1)))
        if hi <= lo or self.order_combo.currentIndex() == 1 and self.order_combo.isVisible():
            self.view_region.setRegion((0, 0))
            return
        starts = self._rows_start[order[lo:hi]]
        self.view_region.setRegion((float(starts.min()), float(starts.max()) + self._row_span))

    def _on_rate_clicked(self, ev):
        vb = self.rate_plot.getViewBox()
        if not self.rate_plot.sceneBoundingRect().contains(ev.scenePos()) or self._rows_start is None:
            return
        t = float(vb.mapSceneToView(ev.scenePos()).x())
        rows = np.searchsorted(self._rows_start, t, side="right") - 1
        if rows < 0:
            return
        pos = int(np.flatnonzero(self._row_order == rows)[0]) if self._row_order is not None else int(rows)
        _xr, (y0, y1) = self.fold_plot.getViewBox().viewRange()
        half = max(2.0, (y1 - y0) / 2.0)
        self.fold_plot.setYRange(max(0, pos - half), pos + half, padding=0)

    # ── raster interaction ────────────────────────────────────────────────────

    def _time_at(self, scene_pos) -> Optional[float]:
        if self._rows_start is None or not self.fold_plot.sceneBoundingRect().contains(scene_pos):
            return None
        p = self.fold_plot.getViewBox().mapSceneToView(scene_pos)
        r = int(np.floor(p.y()))
        if not 0 <= r < self._row_order.size or not 0 <= p.x() <= self._row_span:
            return None
        return float(self._rows_start[self._row_order[r]] + p.x())

    def _on_fold_hover(self, pos):
        t = self._time_at(pos)
        if t is None:
            return
        raw = "; double-click: Raw tab here" if self._has_raw() else ""
        QToolTip.showText(self.fold_plot.mapToGlobal(self.fold_plot.mapFromScene(pos)),
                          f"{t:.3f} s in the recording{raw}", self.fold_plot)

    def _has_raw(self) -> bool:
        dm = self.main_window.data_manager
        return dm is not None and (getattr(dm, "raw_reader", None) is not None
                                   or getattr(dm, "raw_data_memmap", None) is not None)

    def _on_fold_clicked(self, ev):
        if not ev.double() or not self._has_raw():
            return
        t = self._time_at(ev.scenePos())
        if t is None:
            return
        w = self.main_window
        w.analysis_tabs.setCurrentWidget(w.raw_panel)
        panel = w.raw_panel
        QTimer.singleShot(0, lambda: panel._request_load(t, panel.window_duration))

    # ── lanes ─────────────────────────────────────────────────────────────────

    def _fill_lanes(self):
        """Rows of the lanes plot: neighbours (in the background) or the group / all cells."""
        dm = self.main_window.data_manager
        cid = self._cid
        if dm is None or cid is None:
            return
        mode = self.lanes_combo.currentIndex()
        if mode == 0:
            key = (id(dm), getattr(dm, "generation", None), cid)
            if key in self._nb_cache:
                self._set_neighbours(self._nb_cache[key])
                return
            self._set_lane_ids([cid])
            self.nb_note.setText("Finding the units around this cell…")
            self.table.setRowCount(0)
            self._pending = key

            def work():
                nbs = rd.nearby_units(dm, cid)
                sims = rd.ei_similarities(dm, cid, [n.cluster_id for n in nbs])
                return rd.rank_neighbours(nbs, sims)

            def done(nbs):
                self._signals = None
                if self.main_window.data_manager is not dm:     # another run, or closed
                    return
                self._nb_cache[key] = nbs
                if self._pending == key and self._cid == cid:
                    self._set_neighbours(nbs)

            def failed(exc):
                self._signals = None
                logger.warning("nearby units failed for %s", cid, exc_info=exc)
                self.nb_note.setText(f"Could not find nearby units: {exc}")

            from ..workers.workers import BackgroundCall
            from qtpy.QtCore import QThreadPool
            task = BackgroundCall(work)
            self._signals = task.signals
            task.signals.done.connect(done)
            task.signals.failed.connect(failed)
            QThreadPool.globalInstance().start(task)
            return
        if mode == 1:
            try:
                ids = [int(c) for c in (self.main_window._get_pop_subset_ids() or [])]
            except Exception:
                ids = []
        else:
            ids = []
        if not ids:
            ids = [int(c) for c in dm.cluster_df["cluster_id"].values]
        ids = [cid] + [c for c in ids if c != cid][:MAX_GROUP_LANES - 1]
        self._neighbours = []
        self._pair_stats = {}
        self.table.setRowCount(0)
        self.nb_note.setText(f"{len(ids)} lanes: this cell on top, each lane scaled to its own peak.")
        self._set_lane_ids(ids)

    def _set_neighbours(self, nbs: List[rd.Neighbour]):
        self._neighbours = nbs
        self._set_lane_ids([self._cid] + [n.cluster_id for n in nbs])
        self._compute_pairs()
        self._fill_table()
        if not nbs:
            self.nb_note.setText("No other unit within 100 µm or with a similar template.")
        else:
            src = {n.sim_source for n in nbs if n.sim_source}
            what = " and ".join(sorted(src)) if src else "no"
            self.nb_note.setText(f"{len(nbs)} units within 100 µm or with a similar template; "
                                 f"{what} similarity.")
        if self.table.rowCount():
            self.table.selectRow(0)

    def _set_lane_ids(self, ids: List[int]):
        dm = self.main_window.data_manager
        fs = float(dm.sampling_rate)
        self._lane_ids = list(ids)
        self._lane_times = []
        for c in ids:
            s = dm.get_cluster_spikes(c)
            self._lane_times.append(np.sort(np.asarray(s, dtype=np.float64) / fs)
                                    if s is not None else np.zeros(0))
        for item in self._lane_curves:
            self.lanes_plot.removeItem(item)
        self._lane_curves = []
        colours = lane_colours(len(ids), self._colors) if len(ids) <= 40 else None
        for i in range(len(ids)):
            if colours is None:
                break
            curve = pg.PlotCurveItem(connect="pairs", pen=pg.mkPen(colours[i], width=1))
            self.lanes_plot.addItem(curve)
            self._lane_curves.append(curve)
        left = self.lanes_plot.getAxis("left")
        left.setTicks([[(i + 0.5, str(c)) for i, c in enumerate(ids)]] if len(ids) <= 40 else None)
        self.lanes_plot.setYRange(0, max(1, len(ids)), padding=0)
        self.lanes_plot.setXRange(0, self._duration, padding=0)
        self._render_lanes()

    def _render_lanes(self):
        if not self._lane_ids:
            return
        (x0, x1), _ = self.lanes_plot.getViewBox().viewRange()
        x0, x1 = max(0.0, x0), min(self._duration, x1)
        if x1 <= x0:
            x0, x1 = 0.0, self._duration
        n = len(self._lane_ids)
        if x1 - x0 <= TICKS_BELOW_S * 4 and self._lane_curves:
            ok = True
            for i, curve in enumerate(self._lane_curves):
                xs, ys = ticks([self._lane_times[i]], x0, x1)
                if xs is None:
                    ok = False
                    break
                curve.setData(xs, ys + i)
                curve.show()
            if ok:
                self.lanes_image.hide()
                return
        for curve in self._lane_curves:
            curve.hide()
        width_px = max(200, int(self.lanes_plot.getViewBox().width()) or 1200)
        counts = density(self._lane_times, x0, x1, min(width_px, 3000))
        peak = np.percentile(counts, 99, axis=1, keepdims=True)
        self.lanes_image.setImage(np.clip(counts / np.maximum(peak, 1.0), 0, 1).T,
                                  levels=(0, 1), autoLevels=False)
        self.lanes_image.setRect(pg.QtCore.QRectF(x0, 0, x1 - x0, n))
        self.lanes_image.show()

    def _compute_pairs(self):
        a = self._times
        self._pair_stats = {}
        for i, nb in enumerate(self._neighbours, start=1):
            b = self._lane_times[i]
            rate_b = b.size / self._duration
            self._pair_stats[nb.cluster_id] = {
                "shared": rd.shared_fraction(a, b),
                "chance": rd.chance_shared(rate_b),
                "rate": rate_b,
            }

    def _fill_table(self):
        c = self._colors
        colours = lane_colours(len(self._neighbours) + 1, c)
        self.table.setRowCount(len(self._neighbours))
        for r, nb in enumerate(self._neighbours):
            st = self._pair_stats.get(nb.cluster_id, {})
            shared, chance = st.get("shared", 0.0), st.get("chance", 0.0)
            dup = shared >= DUPLICATE_SHARED and shared >= 10 * max(chance, 1e-6)
            unit = QTableWidgetItem(f"■ {nb.cluster_id}")
            unit.setForeground(QColor(colours[r + 1]))
            dist = QTableWidgetItem("—" if not np.isfinite(nb.distance_um) else f"{nb.distance_um:.0f} µm")
            sim = QTableWidgetItem("—" if not np.isfinite(nb.similarity)
                                   else f"{nb.similarity:.2f} {nb.sim_source}")
            sh = QTableWidgetItem(f"{100 * shared:.0f} % ({100 * chance:.1f} %)"
                                  + ("  possible duplicate" if dup else ""))
            if dup:
                sh.setForeground(QColor(c.get("status_noise_text", "#E8564A")))
            for col, item in enumerate((unit, dist, sim, sh)):
                self.table.setItem(r, col, item)

    def _on_table_pick(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows or not self._neighbours:
            return
        r = rows[0].row()
        nb = self._neighbours[r]
        centres, rate = rd.cross_correlogram(self._times, self._lane_times[r + 1])
        colour = lane_colours(len(self._neighbours) + 1, self._colors)[r + 1]
        self.ccg_bars.setOpts(x=centres, height=rate, width=rd.CCG_BIN_S * 1000,
                              brush=pg.mkBrush(colour), pen=pg.mkPen(None))
        base = self._pair_stats.get(nb.cluster_id, {}).get("rate", 0.0)
        self.ccg_base.setPos(base)
        self.ccg_title.setText(f"Unit {nb.cluster_id} around cell {self._cid}'s spikes "
                               f"(dashed: its mean rate)")
        self.ccg_title.setWordWrap(True)
        self.ccg_plot.setXRange(-rd.CCG_WINDOW_S * 1000, rd.CCG_WINDOW_S * 1000, padding=0.02)
        self.ccg_plot.setYRange(0, max(1.0, float(rate.max()) * 1.1 if rate.size else 1.0, base * 1.3),
                                padding=0)

    def _select_neighbour(self, row):
        if 0 <= row < len(self._neighbours):
            self.main_window._select_cluster_in_tree(self._neighbours[row].cluster_id)

    def _lane_at(self, scene_pos) -> int:
        vb = self.lanes_plot.getViewBox()
        if not self.lanes_plot.sceneBoundingRect().contains(scene_pos):
            return -1
        r = int(np.floor(vb.mapSceneToView(scene_pos).y()))
        return r if 0 <= r < len(self._lane_ids) else -1

    def _on_lane_clicked(self, ev):
        r = self._lane_at(ev.scenePos())
        if r > 0:
            if self._neighbours and r - 1 < self.table.rowCount():
                self.table.selectRow(r - 1)
            if ev.double():
                self.main_window._select_cluster_in_tree(self._lane_ids[r])

    def _on_lane_hover(self, pos):
        r = self._lane_at(pos)
        if r >= 0:
            t = self._lane_times[r]
            what = "this cell" if r == 0 else "double-click to select"
            QToolTip.showText(self.lanes_plot.mapToGlobal(self.lanes_plot.mapFromScene(pos)),
                              f"Cell {self._lane_ids[r]} · {t.size:,} spikes · {what}", self.lanes_plot)
