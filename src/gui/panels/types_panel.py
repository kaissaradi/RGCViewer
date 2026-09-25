"""Types tab: the whole run by class — barcode and mosaic atlas (PLAN.md Q40, Q42).

Left, the type barcode: one row per cell (STA time course, ACG or chirp),
one band per class, the most typical cell first. A red tick marks a cell
whose response does not match its band.

Right, the mosaic atlas: one small mosaic per class, all at the same scale.
Red outlines: pairs that overlap too much for one type (a split unit, a
duplicate, or a mixed class). Grey rings: unclassified cells of the right
polarity sitting in a gap of a named type's mosaic.

Click a row, an outline or a ring to select that cell. The tab reads only
cached data (``peek_*``); it never computes on the GUI thread, and says
how many cells are still waiting for their caches.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QSplitter, QToolTip, QVBoxLayout, QWidget,
)

from ...analysis import barcode as bc
from ...analysis import mosaic_stats as ms
from ...analysis import rf_geometry
from ...analysis.class_names import canonical_type
from ..theme import apply_plot_theme, plot_field, resolve_theme_colors

logger = logging.getLogger(__name__)

FEATURES = ("STA time course", "Autocorrelation", "Chirp response")
# What the barcode colours mean, per row type (shown under the barcode).
LEGEND = {
    "STA time course": "Red: the screen got brighter before the spike (ON); blue: darker (OFF). "
                       "Time runs from 30 frames before the spike (left) to the spike (right).",
    "Autocorrelation": "Bright: many spikes at that lag after a spike. A bright band at a few ms "
                       "means bursts; a dark start is the refractory period.",
    "Chirp response": "Bright: high firing rate during the chirp (flash, frequency and contrast "
                      "sweeps, left to right).",
}
ATLAS_COLUMNS = 3
MIN_ATLAS_CELLS = 3
HOLE_MIN_R = 0.8        # a gap candidate must also look like the type
MAX_HOLES = 10
_TRASH = "Trash"
# Groups that are bins, not types: no mosaic tile, no "may not belong" ticks.
_BIN = re.compile(r"unclass|weak|huge|big|large|misfit|badfit|\bbad\b|trash|junk|noise|dup"
                  r"|^nc ?\d*$|^(on|off|all)$", re.I)


def is_type_group(name: str) -> bool:
    """A named type, or a group the user made — not a bin like 'unclassified'."""
    if canonical_type(name) is not None:
        return True
    return not any(_BIN.search(part.strip()) for part in name.split("/"))


# --- data -----------------------------------------------------------------------

def tree_groups(main_window) -> Tuple[Dict[int, str], List[str]]:
    """{cluster id: folder path} for every cell in a folder (not Trash), in tree order."""
    from .. import callbacks
    group_of, order = {}, []
    root = main_window.tree_model.invisibleRootItem()

    def walk(item, path):
        for i in range(item.rowCount()):
            child = item.child(i)
            if child is None:
                continue
            if callbacks.is_group_item(child):
                if not path and child.text() == _TRASH:
                    continue
                walk(child, path + [child.text()])
            elif path:
                name = "/".join(p for p in path if p != "All") or path[-1]
                if name not in order:
                    order.append(name)
                cid = child.data(Qt.ItemDataRole.UserRole)
                if cid is not None:
                    group_of[int(cid)] = name
    walk(root, [])
    return group_of, order


def feature_rows(dm, cells, feature) -> Tuple[Dict[int, np.ndarray], bool, str]:
    """(rows, signed, x label) from cached data only."""
    rows = {}
    if feature == "STA time course":
        for cid in cells:
            phys = dm.peek_cell_physics(cid) if hasattr(dm, "peek_cell_physics") else None
            tc = phys.get("timecourse") if phys else None
            if tc is not None:
                rows[cid] = np.asarray(tc, dtype=float)
        return rows, True, "frames before the spike"
    if feature == "Autocorrelation":
        for cid in cells:
            std = dm.peek_standard_plot_data(cid) if hasattr(dm, "peek_standard_plot_data") else None
            lags, acg = (std or {}).get("acg_time_lags"), (std or {}).get("acg_norm")
            if lags is None or acg is None:
                continue
            lags, acg = np.asarray(lags, float), np.asarray(acg, float)
            rows[cid] = acg[(lags >= 0) & (lags <= 100)]
        return rows, False, "lag (ms)"
    for cid in cells:
        d = dm.get_chirp_data_for_cluster(cid) if hasattr(dm, "get_chirp_data_for_cluster") else None
        if d is not None:
            rows[cid] = np.asarray(d["psth_mean"], dtype=float)
    return rows, False, "time (s)"


def can_have_data(dm, cells, feature) -> set:
    """Cells that have (or will have) this kind of data at all."""
    cells = list(cells)
    if feature == "STA time course":
        stas = getattr(dm, "vision_stas", None)
        if not stas:
            return set()
        out = set()
        for c in cells:
            try:
                if dm.get_vision_id_for_cluster(int(c)) in stas:
                    out.add(c)
            except Exception:
                pass
        return out
    if feature == "Chirp response":
        rows = getattr(dm, "chirp_id_to_row", None) or {}
        return {c for c in cells if int(c) in rows} if getattr(dm, "chirp_available", False) else set()
    return set(cells)          # every cell has spikes, so an ACG


def rf_fits(dm, cells) -> Dict[int, Optional[rf_geometry.RFFit]]:
    vp = getattr(dm, "vision_params", None)
    if vp is None:
        return {}
    out = {}
    for cid in cells:
        try:
            out[cid] = rf_geometry.raw_rf_fit(vp, dm.get_vision_id_for_cluster(int(cid)))
        except Exception:
            out[cid] = None
    return out


def sta_polarity(dm, cid) -> Optional[str]:
    """ON / OFF from the lobe nearest the spike (the rule measured in Q36)."""
    phys = dm.peek_cell_physics(cid) if hasattr(dm, "peek_cell_physics") else None
    tc = phys.get("timecourse") if phys else None
    if tc is None:
        return None
    tc = np.asarray(tc, float)
    peak = np.nanmax(np.abs(tc)) if tc.size else 0
    if not np.isfinite(peak) or peak <= 0:
        return None
    for v in tc[::-1]:
        if abs(v) >= 0.5 * peak:
            return "ON" if v > 0 else "OFF"
    return None


# --- the tab ---------------------------------------------------------------------

class TypesPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._signature = None
        self._barcode: Optional[bc.Barcode] = None
        self._tiles: List[dict] = []
        self._selected = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        head = QHBoxLayout()
        self.help = QLabel(
            "Each row is a cell, each band a class. A row unlike its band (red tick) may be "
            "in the wrong class. Mosaics: a real type tiles the retina — red outlines overlap "
            "too much, grey rings are unclassified cells that look like the type and sit in a "
            "gap. NNND ≈ 2 means neighbours touch. Click to select a cell.")
        self.help.setWordWrap(True)
        self.help.setObjectName("mutedLabel")
        head.addWidget(self.help, 1)
        self.feature_combo = QComboBox()
        self.feature_combo.addItems(FEATURES)
        self.feature_combo.setToolTip("What each barcode row shows")
        self.feature_combo.currentTextChanged.connect(lambda _t: self.refresh(force=True))
        head.addWidget(self.feature_combo)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setToolTip("Rebuild from the current tree and caches")
        self.refresh_btn.clicked.connect(lambda: self.refresh(force=True))
        head.addWidget(self.refresh_btn)
        outer.addLayout(head)

        # Suggested classes (PLAN.md Q36): learned from the lab's labelled runs.
        sug = QHBoxLayout()
        self.suggest_btn = QPushButton("Suggest classes")
        self.suggest_btn.setToolTip(
            "Suggest a class for every cell from the cells the lab has already "
            "classified (5 types). The first run reads the lab's .params files (~1–2 min).")
        self.suggest_btn.clicked.connect(self._suggest)
        sug.addWidget(self.suggest_btn)
        self.accept_btn = QPushButton("Accept confident")
        self.accept_btn.setToolTip("Move every unclassified cell with a confident suggestion "
                                   "to its class. Cells you classified are not moved.")
        self.accept_btn.setEnabled(False)
        self.accept_btn.clicked.connect(self._accept_confident)
        sug.addWidget(self.accept_btn)
        self.atlas_btn = QPushButton("Type atlas")
        self.atlas_btn.setToolTip("What each named type looks like across the lab, with this "
                                  "run's cells drawn on top")
        self.atlas_btn.clicked.connect(self._atlas)
        sug.addWidget(self.atlas_btn)
        self.suggest_summary = QLabel("Ctrl+J: next cell to review · Ctrl+Enter: accept its suggestion")
        self.suggest_summary.setObjectName("mutedLabel")
        self.suggest_summary.setWordWrap(True)
        sug.addWidget(self.suggest_summary, 1)
        outer.addLayout(sug)
        self.status = QLabel("")
        self.status.setObjectName("mutedLabel")
        outer.addWidget(self.status)
        self.legend = QLabel(LEGEND[FEATURES[0]])
        self.legend.setObjectName("mutedLabel")
        self.legend.setWordWrap(True)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.bar_widget = pg.GraphicsLayoutWidget()
        self.bar_plot = self.bar_widget.addPlot()
        self.bar_plot.invertY(True)
        self.bar_plot.setMenuEnabled(False)
        self.bar_image = pg.ImageItem()
        self.bar_plot.addItem(self.bar_image)
        self.bar_lines = pg.PlotCurveItem(connect="pairs")
        self.bar_plot.addItem(self.bar_lines)
        self.bar_misfits = pg.ScatterPlotItem(symbol="s", size=6, pxMode=True)
        self.bar_plot.addItem(self.bar_misfits)
        self.bar_selected = pg.PlotCurveItem()
        self.bar_plot.addItem(self.bar_selected)
        self.bar_widget.scene().sigMouseClicked.connect(self._on_bar_click)
        self.bar_widget.scene().sigMouseMoved.connect(self._on_bar_hover)
        split.addWidget(self.bar_widget)

        self.atlas_widget = pg.GraphicsLayoutWidget()
        self.atlas_widget.scene().sigMouseClicked.connect(self._on_atlas_click)
        split.addWidget(self.atlas_widget)
        split.setSizes([600, 500])
        outer.addWidget(split, 1)
        outer.addWidget(self.legend)

        # While caches fill, look again every few seconds (only when shown).
        self._poll = QTimer(self)
        self._poll.setInterval(3000)
        self._poll.timeout.connect(lambda: self.refresh(force=False))
        self.restyle_plots(main_window.get_current_colors())

    def _suggest(self):
        from .. import suggestions
        suggestions.start(self.main_window)

    def _atlas(self):
        from .type_atlas import open_atlas
        open_atlas(self.main_window)

    def _accept_confident(self):
        from .. import suggestions
        if suggestions.accept_confident(self.main_window):
            self.refresh(force=True)

    # -- lifecycle ------------------------------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh(force=False)
        self._poll.start()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._poll.stop()

    def reset_for_new_dataset(self):
        self._signature = None
        self._barcode = None
        self._clear()

    def restyle_plots(self, colors):
        colors = resolve_theme_colors(colors)
        self._colors = colors
        for w in (self.bar_widget, self.atlas_widget):
            apply_plot_theme(w, colors)
        apply_plot_theme(self.bar_plot, colors)
        self.bar_plot.showGrid(x=False, y=False)
        self._signature = None
        if self.isVisible():
            self.refresh(force=True)

    # -- build ---------------------------------------------------------------------------

    def _clear(self):
        self.bar_image.clear()
        self.bar_lines.setData([], [])
        self.bar_misfits.setData([], [])
        self.bar_selected.setData([], [])
        self.atlas_widget.clear()
        self._tiles = []

    def refresh(self, force=False):
        w = self.main_window
        dm = getattr(w, "data_manager", None)
        if dm is None:
            self._clear()
            self.status.setText("Open a run to see its classes.")
            return
        group_of, order = tree_groups(w)
        feature = self.feature_combo.currentText()
        self.legend.setText(LEGEND.get(feature, ""))
        rows, signed, xlabel = feature_rows(dm, list(group_of), feature)
        signature = (getattr(dm, "generation", None), tuple(sorted(group_of.items())),
                     feature, len(rows))
        if not force and signature == self._signature:
            return
        self._signature = signature
        if not group_of:
            self._clear()
            self.status.setText("No cells in folders yet. Put cells in groups (Ctrl+M, Ctrl+G) "
                                "or load a Vision classification, then come back.")
            return
        self._draw_barcode(rows, group_of, order, signed, xlabel, feature)
        self._draw_atlas(dm, group_of, order)
        if feature == "Chirp response" and not getattr(dm, "chirp_available", False):
            self.status.setText("This run has no chirp data. Pick another row type above.")
            return
        possible = can_have_data(dm, group_of, feature)
        waiting = len(possible - set(rows))
        never = len(group_of) - len(possible)
        what = {"STA time course": "no STA (not in the Vision files)",
                "Chirp response": "no chirp response"}.get(feature, "no data")
        self.status.setText(
            f"{len(rows)} of {len(group_of)} grouped cells shown"
            + (f" · {waiting} still computing (updates by itself)" if waiting else "")
            + (f" · {never} with {what}" if never else "")
            + f" · {getattr(self, '_n_misfits', 0)} may not belong")
        if self._selected is not None:
            self.highlight(self._selected)

    def _draw_barcode(self, rows, group_of, order, signed, xlabel, feature):
        colors = self._colors
        code = bc.build_barcode(rows, group_of, order, signed=signed,
                                type_groups=[g for g in order if is_type_group(g)])
        self._barcode = code
        if not code.cells:
            self._clear()
            return
        cmap = pg.colormap.get("CET-D1" if signed else "viridis")
        self.bar_image.setImage(code.matrix.T, levels=(-1, 1) if signed else (0, 1))
        self.bar_image.setColorMap(cmap)
        n_rows, width = code.matrix.shape
        # band separators and labels
        xs, ys, ticks = [], [], []
        min_label = max(3, int(0.025 * n_rows))   # thinner bands: name on hover only
        for name, start, end in code.bands:
            xs += [0, width]
            ys += [start, start]
            if end - start >= min_label:
                ticks.append(((start + end) / 2.0, f"{name} ({end - start})"))
        self.bar_lines.setData(xs, ys, pen=pg.mkPen(colors["text_primary"], width=1))
        self.bar_plot.getAxis("left").setTicks([ticks, []])
        mis = [code.row_of(c) for c in code.misfits()]
        self._n_misfits = len(mis)
        self.bar_misfits.setData([width + 0.8] * len(mis), [r + 0.5 for r in mis],
                                 brush=pg.mkBrush(colors.get("plot_compare", "#c0392b")),
                                 pen=None)
        if feature == "STA time course":
            self.bar_plot.getAxis("bottom").setTicks(
                [[(width - 1 - k + 0.5, str(k)) for k in range(0, width, 5)], []])
        else:
            self.bar_plot.getAxis("bottom").setTicks(None)
        self.bar_plot.setLabel("bottom", xlabel)
        self.bar_plot.setXRange(0, width + 1.5, padding=0)
        self.bar_plot.setYRange(0, n_rows, padding=0)

    def _draw_atlas(self, dm, group_of, order):
        colors = self._colors
        self.atlas_widget.clear()
        self._tiles = []
        fits = rf_fits(dm, list(group_of))
        if not any(f is not None for f in fits.values()):
            self.atlas_widget.addLabel("No RF fits: the atlas needs Vision .params.")
            return
        members_of = {}
        for cid, g in group_of.items():
            members_of.setdefault(g, []).append(cid)
        unclassified = [c for c, g in group_of.items() if canonical_type(g) is None]
        # One scale for every tile (small multiples): where the RFs are.
        pts = np.array([[f.x0, f.y0] for f in fits.values() if f is not None])
        pad = 2.0 * float(np.median([f.std_x for f in fits.values() if f is not None]))
        x_rng = (pts[:, 0].min() - pad, pts[:, 0].max() + pad)
        y_rng = (pts[:, 1].min() - pad, pts[:, 1].max() + pad)
        blue = colors.get("plot_ensemble", colors.get("plot_scatter", "#4A72B8"))
        red = colors.get("plot_compare", "#c0392b")
        grey = colors.get("text_secondary", "#777")
        type_order = sorted((g for g in order if is_type_group(g)),
                            key=lambda g: (canonical_type(g) is None, order.index(g)))
        col = 0
        for g in type_order:
            mem = {c: fits.get(c) for c in members_of.get(g, [])}
            mem = {c: f for c, f in mem.items() if f is not None}
            if len(mem) < MIN_ATLAS_CELLS:
                continue
            stats = ms.class_stats(mem)
            plot = self.atlas_widget.addPlot(row=len(self._tiles) // ATLAS_COLUMNS,
                                             col=col % ATLAS_COLUMNS)
            col += 1
            plot.setTitle(f"<span style='font-size:9pt'>{g} · n={stats.n}<br>{_tile_line(stats)}</span>")
            plot.setAspectLocked(True)
            plot.hideAxis("left")
            plot.hideAxis("bottom")
            plot.setMenuEnabled(False)
            plot.setMouseEnabled(False, False)
            plot.setRange(xRange=x_rng, yRange=y_rng, padding=0.02)
            close = {c for a, b, _v in stats.close_pairs for c in (a, b)}
            plot.addItem(_outline_curve([f for c, f in mem.items() if c not in close],
                                        pg.mkPen(blue, width=1)))
            if close:
                plot.addItem(_outline_curve([mem[c] for c in close], pg.mkPen(red, width=1.6)))
            holes = []
            ct = canonical_type(g)
            if ct is not None:
                polarity = ct[0].split()[0]
                others = {c: fits.get(c) for c in unclassified
                          if sta_polarity(dm, c) == polarity}
                gaps = [c for c, _v in ms.hole_candidates(mem, others)]
                holes = _looks_like(dm, gaps, list(mem))[:MAX_HOLES]
                if holes:
                    plot.addItem(pg.ScatterPlotItem(
                        [fits[c].x0 for c in holes], [fits[c].y0 for c in holes],
                        symbol="o", size=7, pen=pg.mkPen(grey, width=1.2), brush=None))
            selected = pg.PlotCurveItem()
            plot.addItem(selected)
            self._tiles.append({"plot": plot, "group": g, "fits": mem, "holes": holes,
                                "hole_fits": {c: fits[c] for c in holes}, "selected": selected,
                                "stats": stats})
        if not self._tiles:
            self.atlas_widget.addLabel(f"No class has {MIN_ATLAS_CELLS}+ cells with RF fits yet.")

    # -- selection ----------------------------------------------------------------------

    def highlight(self, cluster_id):
        """Mark the selected cell's row and outline (cheap; no rebuild)."""
        self._selected = cluster_id
        colors = getattr(self, "_colors", {})
        pen = pg.mkPen(colors.get("plot_peak", "#E0B000"), width=2)
        code = self._barcode
        r = code.row_of(cluster_id) if code is not None else -1
        if r >= 0:
            wdt = code.matrix.shape[1]
            self.bar_selected.setData([0, wdt, wdt, 0, 0], [r, r, r + 1, r + 1, r], pen=pen)
        else:
            self.bar_selected.setData([], [])
        for tile in self._tiles:
            fit = tile["fits"].get(cluster_id) or tile["hole_fits"].get(cluster_id)
            if fit is None:
                tile["selected"].setData([], [])
                continue
            x, y = rf_geometry.ellipse_outline(*rf_geometry.mosaic_ellipse(fit))
            tile["selected"].setData(x, y, pen=pen)

    def _select(self, cluster_id):
        if cluster_id is None:
            return
        self.main_window._select_cluster_in_tree(int(cluster_id))

    def _row_at(self, scene_pos):
        if self._barcode is None or not self.bar_plot.sceneBoundingRect().contains(scene_pos):
            return -1
        pt = self.bar_plot.vb.mapSceneToView(scene_pos)
        r = int(np.floor(pt.y()))
        return r if 0 <= r < len(self._barcode.cells) else -1

    def _on_bar_click(self, ev):
        r = self._row_at(ev.scenePos())
        if r >= 0:
            self._select(self._barcode.cells[r])

    def _on_bar_hover(self, scene_pos):
        r = self._row_at(scene_pos)
        if r < 0:
            return
        code = self._barcode
        t = code.typicality[r]
        if not np.isfinite(t):
            fit_text = "not judged (a bin, or too few cells)"
        else:
            fit_text = f"r = {t:.2f} with the rest of its class"
            other, r_other = code.elsewhere[r] if code.elsewhere else ("", np.nan)
            if other and np.isfinite(r_other):
                fit_text += f"\nbest other type: {other} (r = {r_other:.2f})"
        QToolTip.showText(self.bar_widget.mapToGlobal(
            self.bar_widget.mapFromScene(scene_pos)),
            f"Cell {code.cells[r]} · {code.groups[r]}\n{fit_text}", self.bar_widget)

    def _on_atlas_click(self, ev):
        pos = ev.scenePos()
        for tile in self._tiles:
            plot = tile["plot"]
            if not plot.sceneBoundingRect().contains(pos):
                continue
            pt = plot.vb.mapSceneToView(pos)
            best, who = np.inf, None
            for group in (tile["fits"], tile["hole_fits"]):
                for cid, f in group.items():
                    d = np.hypot(f.x0 - pt.x(), f.y0 - pt.y())
                    if d < best:
                        best, who = d, cid
            if who is not None:
                self._select(who)
            return


def _tile_line(stats) -> str:
    """NNND and close pairs. Coverage is left out: arrays record only some
    cells, so a low coverage says little (docs/design/rgc_types.md)."""
    if stats.n < 3:
        return "too few cells to judge"
    line = f"NNND {stats.median_nnnd:.1f}"
    if stats.close_pairs:
        k = len(stats.close_pairs)
        line += f" · {k} close pair{'s' if k != 1 else ''}"
    return line


def _unit_tc(dm, cid):
    phys = dm.peek_cell_physics(cid) if hasattr(dm, "peek_cell_physics") else None
    tc = phys.get("timecourse") if phys else None
    if tc is None:
        return None
    tc = np.asarray(tc, float)
    tc = tc - tc.mean()
    n = np.linalg.norm(tc)
    return tc / n if n > 0 else None


def _looks_like(dm, candidates, members) -> List[int]:
    """Candidates whose STA time course correlates ≥ HOLE_MIN_R with the class mean."""
    vecs = [v for v in (_unit_tc(dm, c) for c in members) if v is not None]
    if not vecs:
        return []
    width = min(len(v) for v in vecs)
    mean = np.mean([v[:width] for v in vecs], axis=0)
    mean = mean - mean.mean()
    mean /= np.linalg.norm(mean) or 1.0
    scored = []
    for c in candidates:
        v = _unit_tc(dm, c)
        if v is None or len(v) < width:
            continue
        v = v[:width] - v[:width].mean()
        nv = np.linalg.norm(v)
        r = float(v @ mean / nv) if nv > 0 else 0.0
        if r >= HOLE_MIN_R:
            scored.append((r, c))
    return [c for _r, c in sorted(scored, reverse=True)]


def _outline_curve(fits, pen) -> pg.PlotCurveItem:
    xs, ys = [], []
    for f in fits:
        x, y = rf_geometry.ellipse_outline(*rf_geometry.mosaic_ellipse(f), n=48)
        xs.append(np.append(x, np.nan))
        ys.append(np.append(y, np.nan))
    if not xs:
        return pg.PlotCurveItem()
    return pg.PlotCurveItem(np.concatenate(xs), np.concatenate(ys), pen=pen, connect="finite")
