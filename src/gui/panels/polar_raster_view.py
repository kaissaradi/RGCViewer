"""Polar tuning plot framed by one spike raster per direction.

Each raster sits at its direction's angle around the polar plot (0° right,
90° up), so the ring shows repeat count and trial-to-trial variance at a
glance. Every direction that ran gets a raster; nothing is dropped.

The polar plot itself is owned by GratingPanel. This widget only places it
in the centre and manages the raster pool.
"""

import math

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QWidget

_MARGIN = 4


def ring_layout(directions_deg, width, height):
    """Rectangles for the polar plot and one raster per direction.

    Returns ``(polar_rect, raster_rects)`` with rects as ``(x, y, w, h)`` in
    widget pixels. Rasters sit on an ellipse at their direction's angle.
    Raster size shrinks until no two rasters overlap.
    """
    dirs = [float(d) for d in directions_deg]
    n = len(dirs)
    cx, cy = width / 2.0, height / 2.0
    if n == 0:
        side = max(min(width, height) - 2 * _MARGIN, 10)
        return (cx - side / 2, cy - side / 2, side, side), []

    rw, rh = width * 0.22, height * 0.20
    for _ in range(30):
        rx = max(width / 2.0 - rw / 2.0 - _MARGIN, 1.0)
        ry = max(height / 2.0 - rh / 2.0 - _MARGIN, 1.0)
        centres = [(cx + rx * math.cos(math.radians(d)),
                    cy - ry * math.sin(math.radians(d))) for d in dirs]
        if not _any_overlap(centres, rw, rh):
            break
        rw *= 0.92
        rh *= 0.92

    rects = [(x - rw / 2.0, y - rh / 2.0, rw, rh) for x, y in centres]
    # Largest centred square that clears every raster (diagonal rasters
    # reach into the middle, so min(width, height) is not enough).
    side = min(width, height) - 2 * _MARGIN
    while side > 40:
        sq = (cx - side / 2.0, cy - side / 2.0, side, side)
        if not any(_rects_overlap(sq, r) for r in rects):
            break
        side *= 0.96
    side = max(side, 40)
    return (cx - side / 2.0, cy - side / 2.0, side, side), rects


def _rects_overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _any_overlap(centres, w, h):
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            if (abs(centres[i][0] - centres[j][0]) < w
                    and abs(centres[i][1] - centres[j][1]) < h):
                return True
    return False


def raster_segments(trials_s):
    """x, y arrays for PlotCurveItem(connect='pairs'): one tick per spike.

    Trial 0 is drawn at the top. Each tick spans 80% of its row.
    """
    n = len(trials_s)
    xs, ys = [], []
    for k, sp in enumerate(trials_s):
        sp = np.asarray(sp, dtype=float)
        if sp.size == 0:
            continue
        row = n - 1 - k
        xs.append(np.repeat(sp, 2))
        ys.append(np.tile([row + 0.1, row + 0.9], sp.size))
    if not xs:
        return np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(ys)


class PolarRasterView(QWidget):
    """Hosts the polar PlotWidget in the centre and a raster per direction."""

    def __init__(self, polar_widget, colors, parent=None):
        super().__init__(parent)
        self.polar = polar_widget
        self.polar.setParent(self)
        self._colors = colors
        self._pool = []          # [(PlotWidget, region, ticks)]
        self._directions = []
        self.note = QLabel("", self)
        self.note.setAlignment(Qt.AlignCenter)
        self.note.setWordWrap(True)
        self.note.hide()
        self.setMinimumSize(320, 260)

    # ── public API ────────────────────────────────────────────────────────

    def set_rasters(self, rasters_by_dir, timing, highlight_dir=None):
        """Show one raster per direction. ``rasters_by_dir``: {deg: [s arrays]}."""
        c = self._colors
        dirs = sorted(rasters_by_dir)
        self._directions = dirs
        pre = float(timing.get("pre_s", 0.0))
        stim = float(timing.get("stim_s", 0.0))
        tail = float(timing.get("tail_s", 0.0))
        x_lo, x_hi = -pre, stim + tail
        for i, d in enumerate(dirs):
            plot, region, ticks = self._raster(i)
            trials = rasters_by_dir[d]
            x, y = raster_segments(trials)
            is_hl = highlight_dir is not None and _same_direction(d, highlight_dir)
            color = c.get("plot_compare", "#E03131") if is_hl else c.get("text_primary", "#ddd")
            ticks.setPen(pg.mkPen(color, width=1))
            ticks.setData(x, y, connect="pairs")
            region.setRegion((0.0, stim))
            plot.setXRange(x_lo, x_hi, padding=0)
            plot.setYRange(0, max(len(trials), 1), padding=0)
            label_color = c.get("text_primary", "#ddd") if is_hl else c.get("text_tertiary", "#888")
            plot.setTitle(
                f"<span style='color:{label_color}; font-size:8px;'>"
                f"{d:g}° · n={len(trials)}</span>")
            plot.show()
        for plot, _r, _t in self._pool[len(dirs):]:
            plot.hide()
        self.note.hide()
        self._relayout()

    def clear_rasters(self, note=""):
        self._directions = []
        for plot, _r, ticks in self._pool:
            ticks.setData([], [])
            plot.hide()
        self.note.setText(note)
        self.note.setVisible(bool(note))
        self._relayout()

    def restyle(self, colors):
        self._colors = colors
        for plot, region, _t in self._pool:
            self._style_raster(plot, region)
        self.note.setStyleSheet(f"color: {colors.get('text_tertiary', '#888')};")

    # ── internals ─────────────────────────────────────────────────────────

    def _raster(self, i):
        while len(self._pool) <= i:
            plot = pg.PlotWidget(parent=self)
            plot.hideAxis("left")
            plot.hideAxis("bottom")
            plot.setMouseEnabled(x=False, y=False)
            plot.setMenuEnabled(False)
            plot.hideButtons()
            region = pg.LinearRegionItem(movable=False)
            for line in region.lines:
                line.setPen(pg.mkPen(None))
            plot.addItem(region)
            ticks = pg.PlotCurveItem()
            plot.addItem(ticks)
            self._style_raster(plot, region)
            self._pool.append((plot, region, ticks))
        return self._pool[i]

    def _style_raster(self, plot, region):
        c = self._colors
        plot.setBackground(c.get("bg_panel", "#1a1a1a"))
        shade = pg.mkColor(c.get("plot_fr", "#FFD43B"))
        shade.setAlpha(28)
        region.setBrush(pg.mkBrush(shade))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout()

    def _relayout(self):
        polar_rect, rects = ring_layout(self._directions, self.width(), self.height())
        x, y, w, h = (int(round(v)) for v in polar_rect)
        self.polar.setGeometry(x, y, w, h)
        for (plot, _r, _t), (rx, ry, rw, rh) in zip(self._pool, rects):
            plot.setGeometry(int(round(rx)), int(round(ry)), int(round(rw)), int(round(rh)))
            plot.raise_()
        self.note.setGeometry(0, self.height() - 40, self.width(), 36)


def _same_direction(a, b, tol=1e-6):
    return abs(((float(a) - float(b) + 180.0) % 360.0) - 180.0) < tol
