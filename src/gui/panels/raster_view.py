"""Spike raster over the whole recording, for runs with no raw file (PLAN.md Q38).

The Raw tab used to be disabled without a raw voltage file. Now it shows
this: one row per cell (the selected cell on top, then the rest of its
group), time across, the recording's stimulus blocks shaded (Q41).

Zoomed out, each row is a density image — spike counts per screen pixel,
recomputed only for the visible window (``np.searchsorted`` on each cell's
sorted spike times), so an hour of 300 cells stays quick. Zoomed in below
TICKS_BELOW_S, every spike is a tick. Click a row to select that cell.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QToolTip, QVBoxLayout, QWidget

from ..theme import apply_plot_theme, plot_field, resolve_theme_colors

logger = logging.getLogger(__name__)

MAX_ROWS = 400
TICKS_BELOW_S = 6.0
MAX_TICKS = 60000


def density(spike_times: List[np.ndarray], t0: float, t1: float, n_bins: int) -> np.ndarray:
    """(rows, n_bins) spike counts on [t0, t1); each array must be sorted."""
    edges = np.linspace(t0, t1, n_bins + 1)
    out = np.zeros((len(spike_times), n_bins), dtype=np.float32)
    for i, t in enumerate(spike_times):
        if t.size:
            out[i] = np.diff(np.searchsorted(t, edges))
    return out


def ticks(spike_times: List[np.ndarray], t0: float, t1: float, limit=MAX_TICKS):
    """x, y for PlotCurveItem(connect='pairs'): one short vertical line per spike."""
    xs, ys = [], []
    n = 0
    for i, t in enumerate(spike_times):
        a, b = np.searchsorted(t, [t0, t1])
        sel = t[a:b]
        n += sel.size
        if n > limit:
            return None, None
        xs.append(np.repeat(sel, 2))
        ys.append(np.tile([i + 0.15, i + 0.85], sel.size))
    if not xs:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(xs), np.concatenate(ys)


class SessionRaster(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._cells: List[int] = []
        self._times: List[np.ndarray] = []
        self._duration = 1.0
        self._key = None
        self._block_items = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        head = QHBoxLayout()
        self.note = QLabel(
            "No raw voltage file is loaded, so this tab shows spike rasters: one row per cell, "
            "the selected cell on top (yellow), each row scaled to its own peak. Zoom with the "
            "wheel; below a few seconds every spike is drawn. Hover a row for its ID; click to "
            "select it.")
        self.note.setWordWrap(True)
        self.note.setObjectName("mutedLabel")
        head.addWidget(self.note, 1)
        self.rows_combo = QComboBox()
        self.rows_combo.addItems(["Rows: its group", "Rows: all cells"])
        self.rows_combo.currentIndexChanged.connect(lambda _i: self.show_cell(self._selected, force=True))
        head.addWidget(self.rows_combo)
        self.load_btn = QPushButton("Load raw file…")
        self.load_btn.setToolTip("File ▸ Load Raw Data File: show voltage traces instead")
        self.load_btn.clicked.connect(lambda: main_window.load_raw_data_file())
        head.addWidget(self.load_btn)
        outer.addLayout(head)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Time in the recording (s)")
        self.plot.setMenuEnabled(False)
        self.plot.invertY(True)
        self.plot.getViewBox().setMouseEnabled(x=True, y=False)
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        self.tick_curve = pg.PlotCurveItem(connect="pairs")
        self.plot.addItem(self.tick_curve)
        self.sel_box = pg.PlotCurveItem()
        self.plot.addItem(self.sel_box)
        outer.addWidget(self.plot, 1)
        self._selected = None
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(80)
        self._debounce.timeout.connect(self._render)
        self.plot.getViewBox().sigXRangeChanged.connect(lambda *_a: self._debounce.start())
        self.plot.scene().sigMouseClicked.connect(self._on_click)
        self.plot.scene().sigMouseMoved.connect(self._on_hover)
        self.restyle_plots(main_window.get_current_colors())

    # -- data ------------------------------------------------------------------

    def _rows_for(self, cluster_id) -> List[int]:
        w = self.main_window
        dm = w.data_manager
        if self.rows_combo.currentIndex() == 1:
            ids = [int(c) for c in dm.cluster_df["cluster_id"].values]
        else:
            try:
                ids = [int(c) for c in (w._get_pop_subset_ids() or [])]
            except Exception:
                ids = []
            if not ids:
                ids = [int(c) for c in dm.cluster_df["cluster_id"].values]
        ids = [c for c in ids if c != int(cluster_id)]
        return [int(cluster_id)] + ids[:MAX_ROWS - 1]

    def show_cell(self, cluster_id, force=False):
        dm = getattr(self.main_window, "data_manager", None)
        if dm is None or cluster_id is None:
            return
        self._selected = int(cluster_id)
        rows = self._rows_for(cluster_id)
        key = (getattr(dm, "generation", None), tuple(rows))
        if force or key != self._key:
            self._key = key
            fs = float(dm.sampling_rate)
            self._cells = rows
            self._times = []
            for c in rows:
                s = dm.get_cluster_spikes(c)
                t = np.sort(np.asarray(s, dtype=np.float64) / fs) if s is not None else np.zeros(0)
                self._times.append(t)
            spikes = getattr(dm, "spike_times", None)
            self._duration = float(spikes[-1]) / fs if spikes is not None and len(spikes) else \
                max((t[-1] for t in self._times if t.size), default=1.0)
            self._draw_blocks(dm)
            self.plot.setXRange(0, self._duration, padding=0)
            self.plot.setYRange(0, len(rows), padding=0)
            left = self.plot.getAxis("left")
            left.setTicks([[(i + 0.5, str(c)) for i, c in enumerate(rows)]] if len(rows) <= 40 else None)
            left.setLabel("cell" if len(rows) <= 40 else "cells")
        self._render()

    def _draw_blocks(self, dm):
        for item in self._block_items:
            self.plot.removeItem(item)
        self._block_items = []
        try:
            from ...analysis import recording_timeline as rt
            if getattr(dm, "stimulus_manifest", None) is None and hasattr(dm, "load_stimulus_manifest"):
                dm.load_stimulus_manifest()
            spikes = getattr(dm, "spike_times", None)
            blocks = rt.stimulus_blocks(dm.kilosort_dir.parent.name, dm.stimulus_manifest,
                                        int(spikes[-1]), float(dm.sampling_rate)) \
                if spikes is not None and len(spikes) else []
        except Exception:
            blocks = []
        if len(blocks) < 2:
            return
        c = self._colors
        for i, b in enumerate(blocks):
            shade = QColor(c.get("plot_highlight", "#4A72B8"))
            shade.setAlpha(22 if i % 2 == 0 else 8)
            region = pg.LinearRegionItem((b.start_s, b.end_s), movable=False,
                                         brush=pg.mkBrush(shade), pen=pg.mkPen(None))
            region.setZValue(-10)
            label = pg.TextItem(b.protocol, color=c["text_secondary"], anchor=(0, 1))
            label.setPos(b.start_s, 0)
            self.plot.addItem(region)
            self.plot.addItem(label)
            self._block_items += [region, label]

    # -- drawing ----------------------------------------------------------------

    def _render(self):
        if not self._cells:
            return
        (x0, x1), _ = self.plot.getViewBox().viewRange()
        x0, x1 = max(0.0, x0), min(self._duration, x1)
        if x1 <= x0:
            return
        n = len(self._cells)
        if x1 - x0 <= TICKS_BELOW_S:
            xs, ys = ticks(self._times, x0, x1)
            if xs is not None:
                self.image.hide()
                self.tick_curve.setData(xs, ys, pen=pg.mkPen(self._colors["text_primary"], width=1))
                self.tick_curve.show()
                self._mark_selected(x0, x1)
                return
        width_px = max(200, int(self.plot.getViewBox().width()) or 1200)
        counts = density(self._times, x0, x1, min(width_px, 3000))
        # Each row to its own 99th percentile: when a cell fires, not how much
        # (the rate itself is on the Standard tab).
        peak = np.percentile(counts, 99, axis=1, keepdims=True)
        scaled = np.clip(counts / np.maximum(peak, 1.0), 0, 1)
        self.image.setImage(scaled.T, levels=(0, 1), autoLevels=False)
        self.image.setRect(pg.QtCore.QRectF(x0, 0, x1 - x0, n))
        self.image.show()
        self.tick_curve.hide()
        self._mark_selected(x0, x1)

    def _mark_selected(self, x0, x1):
        pen = pg.mkPen(self._colors.get("plot_peak", "#E0B000"), width=2)
        self.sel_box.setData([x0, x1, x1, x0, x0], [0, 0, 1, 1, 0], pen=pen)

    def restyle_plots(self, colors):
        c = resolve_theme_colors(colors)
        self._colors = c
        apply_plot_theme(self.plot, c)
        self.plot.showGrid(x=False, y=False)
        field, ink = QColor(plot_field(c)), QColor(c["text_primary"])
        cmap = pg.ColorMap([0.0, 1.0], [field, ink])
        self.image.setColorMap(cmap)
        if self._cells:
            self._render()

    # -- interaction --------------------------------------------------------------

    def _row_at(self, scene_pos) -> int:
        vb = self.plot.getViewBox()
        if not self.plot.sceneBoundingRect().contains(scene_pos):
            return -1
        r = int(np.floor(vb.mapSceneToView(scene_pos).y()))
        return r if 0 <= r < len(self._cells) else -1

    def _on_click(self, ev):
        r = self._row_at(ev.scenePos())
        if r > 0:
            self.main_window._select_cluster_in_tree(self._cells[r])

    def _on_hover(self, pos):
        r = self._row_at(pos)
        if r >= 0:
            t = self._times[r]
            QToolTip.showText(self.plot.mapToGlobal(self.plot.mapFromScene(pos)),
                              f"Cell {self._cells[r]} · {t.size} spikes", self.plot)
