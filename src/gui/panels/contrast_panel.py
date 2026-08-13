"""Contrast-response across light levels, for one cell and for a tree group.

A concatenated sort that spans several ContrastResponseGrating runs recorded at
different NDF settings is the only way to ask how a cell's contrast response
changes with light level, because every run is otherwise spike-sorted
independently and the IDs do not correspond.

Two things about this data drive the design:

*The interesting result is often "no response".* On 20260715A the fraction of
cells with a measurable response collapses as the light dims — 26.5% at NDF2,
29.1% at NDF3, 2.8% at NDF4, 1.0% at NDF5. A panel that draws four curves per
cell would mostly be drawing noise, so a level with no measurable response is
shown as an explicit absence rather than a flat line, and the per-level
responsive counts sit next to the curves.

*Light level is free text.* Levels are labelled with the experimenter's epoch
group ("NDF5wheel", "NDF3ish"), which is the only record of it anywhere. They
are ordered by acquisition, not parsed into numbers.
"""

import logging

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QStackedLayout, QVBoxLayout, QWidget,
)

from ..theme import apply_plot_theme, plot_grid_alpha, resolve_theme_colors
from ...analysis import contrast_calc

logger = logging.getLogger(__name__)

pg.setConfigOptions(antialias=True)

# Dim -> bright, in acquisition order. Distinguishable in both themes and
# ordered so the ramp itself reads as a light-level axis.
LEVEL_COLORS = [
    (120, 130, 145),
    (90, 140, 220),
    (60, 190, 160),
    (240, 170, 60),
    (235, 110, 90),
    (190, 120, 220),
]


class ContrastPanel(QWidget):
    """Per-cell and per-group contrast-response functions, split by light level."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._group_ids = []
        self._cluster_id = None
        self._curves = []
        self._dirty = False

        colors = resolve_theme_colors(self.main_window.get_current_colors())
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedLayout()
        outer.addLayout(self.stack)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # --- header ---
        header = QHBoxLayout()
        title = QLabel(
            f"<span style='color:{colors['text_tertiary']}; font-size:10px; "
            f"letter-spacing:0.06em;'>CONTRAST RESPONSE ACROSS LIGHT LEVELS</span>")
        header.addWidget(title)
        header.addStretch()

        self.measure_combo = QComboBox()
        self.measure_combo.addItems(["F1 (at drift frequency)", "F0 (mean rate)",
                                     "F2 (frequency doubled)"])
        self.measure_combo.setToolTip(
            "F1 is the modulated response at the grating's drift frequency.\n"
            "F2 picks out frequency-doubling (Y-like) cells.")
        self.measure_combo.currentIndexChanged.connect(self.refresh)
        header.addWidget(QLabel("Measure:"))
        header.addWidget(self.measure_combo)

        self.fit_chk = QCheckBox("Naka-Rushton fit")
        self.fit_chk.setChecked(True)
        self.fit_chk.setToolTip(
            "Overlay the saturation fit and report c50, the contrast at which\n"
            "the cell reaches half its maximum. c50 is flagged when the curve\n"
            "is still rising at the highest contrast shown, where it is an\n"
            "extrapolation rather than a measurement.")
        self.fit_chk.toggled.connect(self.refresh)
        header.addWidget(self.fit_chk)

        self.responsive_only_chk = QCheckBox("Responsive levels only")
        self.responsive_only_chk.setChecked(False)
        self.responsive_only_chk.setToolTip(
            "Hide light levels where this cell has no measurable response\n"
            "(F1 at the top contrast below twice the blank-contrast F1).")
        self.responsive_only_chk.toggled.connect(self.refresh)
        header.addWidget(self.responsive_only_chk)

        self.lbl_summary = QLabel("")
        self.lbl_summary.setStyleSheet(
            f"color: {colors['text_secondary']}; font-size: 11px;")
        header.addSpacing(10)
        header.addWidget(self.lbl_summary)
        layout.addLayout(header)

        # --- plots ---
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(colors["bg_panel"])
        layout.addWidget(self.glw)

        self.cell_plot = self.glw.addPlot(title="Selected cell")
        self.cell_plot.setLabel("bottom", "Contrast")
        self.cell_plot.setLabel("left", "Response (Hz)")
        self.cell_plot.addLegend(offset=(-10, 10), labelTextSize="7pt")

        self.group_plot = self.glw.addPlot(title="Group mean")
        self.group_plot.setLabel("bottom", "Contrast")
        self.group_plot.setLabel("left", "Response (Hz)")

        self.glw.nextRow()
        self.count_plot = self.glw.addPlot(colspan=2, title="Responsive cells per light level")
        self.count_plot.setLabel("left", "Cells")
        self.glw.ci.layout.setRowStretchFactor(0, 3)
        self.glw.ci.layout.setRowStretchFactor(1, 2)

        self.plots = [self.cell_plot, self.group_plot, self.count_plot]
        for p in self.plots:
            self._style_plot(p, colors)

        self.stack.addWidget(page)

        placeholder = QWidget()
        pl = QVBoxLayout(placeholder)
        self.placeholder_label = QLabel("No contrast-response data")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        pl.addWidget(self.placeholder_label)
        self.stack.addWidget(placeholder)
        self.stack.setCurrentIndex(1)

    # ── chrome ───────────────────────────────────────────────────────────────

    def _style_plot(self, p, colors):
        apply_plot_theme(p, colors)
        p.showGrid(x=True, y=True, alpha=plot_grid_alpha(colors))
        if p.titleLabel:
            p.titleLabel.setText(p.titleLabel.text, color=colors["text_secondary"],
                                 size="9pt")

    def restyle_plots(self, colors):
        colors = resolve_theme_colors(colors)
        self.glw.setBackground(colors["bg_panel"])
        for p in self.plots:
            self._style_plot(p, colors)
        self.lbl_summary.setStyleSheet(
            f"color: {colors['text_secondary']}; font-size: 11px;")
        self.refresh()

    @staticmethod
    def _level_color(i):
        return LEVEL_COLORS[i % len(LEVEL_COLORS)]

    def _measure_key(self):
        return ["f1", "f0", "f2"][self.measure_combo.currentIndex()]

    # ── data entry points ────────────────────────────────────────────────────

    def update_all(self, cluster_id):
        """Selection changed to a single cell."""
        self._cluster_id = cluster_id
        self._refresh_if_visible()

    def update_group(self, cluster_ids):
        """A tree folder was selected; its members drive the group curves.

        Deferred while the tab is hidden. A group view costs one contrast-
        response computation per member per light level — about a second for a
        40-cell folder before the cache is warm — and this is called from the
        settled-selection path, so paying it for a tab nobody is looking at
        would be felt as lag in the tree.
        """
        self._group_ids = [int(c) for c in (cluster_ids or [])]
        self._refresh_if_visible()

    def _refresh_if_visible(self):
        if self.isVisible():
            self._dirty = False
            self.refresh()
        else:
            self._dirty = True

    def showEvent(self, event):
        super().showEvent(event)
        if getattr(self, "_dirty", False):
            self._dirty = False
            self.refresh()

    def refresh(self, *_):
        dm = self.main_window.data_manager
        if dm is None or not getattr(dm, "contrast_available", False):
            self.stack.setCurrentIndex(1)
            self.placeholder_label.setText("No contrast-response data loaded")
            return
        self.stack.setCurrentIndex(0)
        self._clear_curves()
        levels = dm.contrast_light_levels()
        self._draw_cell(dm, levels)
        self._draw_group(dm, levels)

    def _clear_curves(self):
        for p in self.plots:
            p.clear()
        if self.cell_plot.legend is not None:
            self.cell_plot.legend.clear()

    # ── per-cell ─────────────────────────────────────────────────────────────

    def _draw_cell(self, dm, levels):
        key = self._measure_key()
        cid = self._cluster_id
        entries = dm.get_contrast_data_for_cluster(cid) if cid is not None else None
        if not entries:
            self.lbl_summary.setText(
                "<b>Cell:</b> no contrast trials" if cid is not None else "")
            return

        bits = []
        for i, (run, label) in enumerate(levels):
            e = entries.get(run)
            if e is None:
                continue
            if self.responsive_only_chk.isChecked() and not e["responsive"]:
                continue
            col = self._level_color(i)
            x, y = e["contrasts"], e[key]
            pen = pg.mkPen(col, width=2 if e["responsive"] else 1,
                           style=Qt.SolidLine if e["responsive"] else Qt.DashLine)
            name = f"{label} ({run})" + ("" if e["responsive"] else " — no response")
            curve = self.cell_plot.plot(x, y, pen=pen, symbol="o", symbolSize=5,
                                        symbolBrush=col, symbolPen=None, name=name)
            self._curves.append(curve)

            # No SEM is accumulated for F2, so don't borrow F0's and label it
            # as this measure's spread.
            sem = {"f1": e["f1_sem"], "f0": e["f0_sem"]}.get(key)
            if sem is not None and np.isfinite(sem).any():
                err = pg.ErrorBarItem(x=x, y=y, height=2 * np.nan_to_num(sem),
                                      pen=pg.mkPen(col, width=1))
                self.cell_plot.addItem(err)

            if self.fit_chk.isChecked() and e["responsive"]:
                fit = contrast_calc.fit_naka_rushton(x, y)
                if fit is not None:
                    xf = np.linspace(float(x.min()), float(x.max()), 100)
                    self.cell_plot.plot(
                        xf, contrast_calc.naka_rushton(xf, fit["r_max"], fit["c50"],
                                                       fit["n"], fit["offset"]),
                        pen=pg.mkPen(col, width=1, style=Qt.DotLine))
                    flag = "*" if fit["c50_at_bound"] else ""
                    bits.append(f"{label}: c50={fit['c50']:.2f}{flag}")

        n_resp = sum(1 for e in entries.values() if e["responsive"])
        summary = f"<b>Cell {cid}:</b> responds at {n_resp}/{len(entries)} levels"
        if bits:
            summary += " &nbsp; " + ", ".join(bits)
        if any("*" in b for b in bits):
            summary += " &nbsp; <i>*still rising at the top contrast</i>"
        self.lbl_summary.setText(summary)
        self.cell_plot.enableAutoRange()

    # ── per-group ────────────────────────────────────────────────────────────

    def _draw_group(self, dm, levels):
        key = self._measure_key()
        ids = self._group_ids
        if not ids:
            self.group_plot.setTitle("Group mean — select a folder in the tree")
            self.count_plot.setTitle("Responsive cells per light level")
            return

        self.group_plot.setTitle(f"Group mean ({len(ids)} cells)")
        per_level_counts, tick_labels = [], []
        for i, (run, label) in enumerate(levels):
            stack, n_resp = [], 0
            for cid in ids:
                entries = dm.get_contrast_data_for_cluster(cid, run=run)
                e = (entries or {}).get(run)
                if e is None:
                    continue
                if e["responsive"]:
                    n_resp += 1
                    stack.append(e[key])
            per_level_counts.append(n_resp)
            tick_labels.append((i, f"{label}\n{n_resp}/{len(ids)}"))

            if not stack:
                continue
            arr = np.vstack(stack)
            mean = np.nanmean(arr, axis=0)
            sem = (np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
                   if arr.shape[0] > 1 else np.zeros_like(mean))
            x = dm.get_contrast_data_for_cluster(ids[0], run=run)
            x = x[run]["contrasts"] if x else np.arange(len(mean))
            col = self._level_color(i)
            self.group_plot.plot(x, mean, pen=pg.mkPen(col, width=2), symbol="o",
                                 symbolSize=5, symbolBrush=col, symbolPen=None)
            if np.isfinite(sem).any():
                self.group_plot.addItem(pg.ErrorBarItem(
                    x=x, y=mean, height=2 * np.nan_to_num(sem),
                    pen=pg.mkPen(col, width=1)))

        # Only cells with a measurable response contribute to the means above,
        # so the counts are needed to read them: a confident-looking curve over
        # 3 of 40 cells is not a property of the group.
        bar = pg.BarGraphItem(
            x=np.arange(len(per_level_counts)), height=per_level_counts, width=0.6,
            brushes=[pg.mkBrush(self._level_color(i))
                     for i in range(len(per_level_counts))])
        self.count_plot.addItem(bar)
        self.count_plot.getAxis("bottom").setTicks([tick_labels])
        self.count_plot.setYRange(0, max(max(per_level_counts, default=1), 1) * 1.15)
        self.group_plot.enableAutoRange()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def cleanup(self):
        for p in self.plots:
            if p:
                p.clear()
        if hasattr(self.glw, "close"):
            self.glw.close()

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)
