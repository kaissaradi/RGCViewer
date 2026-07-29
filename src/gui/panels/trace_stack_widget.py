"""An overlay of one trace per cell, with a live highlight.

The RF mosaic beside the embedding answers "do these cells tile the array?".
This answers the other half of the same question — "do they actually have the
same response?" — for whichever feature blocks the embedding was built from.
Every cell in the run is drawn once as a dim background; selecting a group
repaints just that group on top.

Highlighting is a per-element recolour of a single overlay collection, blitted
over a cached background, exactly as ``RFMapWidget`` does it: matplotlib's
render pipeline runs when the cell set or the theme changes, never on
selection. At the ~700 traces a run holds that is the difference between
brushing and waiting.

Rows that are entirely zero are *not* drawn. In the feature blocks a zero row
is the sentinel for "this cell has no STA / no chirp response", not a flat
measured response (see ``DataManager.get_raw_feature_blocks``), and drawing it
would put a fake line through the middle of every overlay. They are counted in
the caption instead, which incidentally makes visible how many cells in a
cluster are there because they share *missing* data rather than a response.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.ndimage as ndimage
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from qtpy.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)

__all__ = ["TraceStackWidget"]


DEFAULT_COLORS = {
    "bg": "#0f1117",
    "surface": "#171923",
    "border": "#2a2d3e",
    "text_primary": "#e8eaf0",
    "text_muted": "#6b7280",
    "accent": "#4f8ef7",
    "highlight": "#f97316",
}

# Traces are decimated to this many points before drawing. A 1250-bin chirp
# PSTH across 700 cells is ~875k vertices in one collection, which is slow to
# draw and indistinguishable on screen from the decimated version at any
# realistic panel width.
_MAX_POINTS = 500


class TraceStackWidget(QWidget):
    """Overlaid per-cell traces with a blit-updated highlight.

    Usage::

        stack = TraceStackWidget(colors=..., title="Temporal STA")
        stack.set_traces(matrix, cluster_ids, x=lags_ms)   # once, per run
        stack.highlight([12, 44, 91])                      # cheap, per selection
    """

    def __init__(
        self,
        parent=None,
        colors: dict = None,
        title: str = "Traces",
        normalize: str = "peak",
        smooth_bins: float = 0.0,
        regions: list = None,
    ):
        super().__init__(parent)
        self._colors = dict(DEFAULT_COLORS)
        if colors:
            self._colors.update(colors)
        # Panels are ~130 px tall, so there is no room for an axis label on top
        # of tick labels — put the x unit in the title instead.
        self._title = title
        self._normalize = normalize
        self._smooth_bins = float(smooth_bins)
        self._regions = list(regions or [])

        self._cluster_ids: list = []      # cells with a real trace, in draw order
        self._row_of: dict = {}           # cluster_id -> row in the collections
        self._segments: list = []
        self._n_requested = 0             # cells asked for, incl. sentinel rows
        self._overlay = None
        self._blit_bg = None
        self._ax = None

        self._build_ui()

    # ── construction ─────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.caption = QLabel(self._title)
        self.caption.setStyleSheet(
            f"color: {self._colors['text_muted']}; font-size: 11px; padding: 1px 0;"
        )
        layout.addWidget(self.caption)

        self.fig = Figure(figsize=(3.2, 1.6), dpi=100)
        self.fig.patch.set_facecolor(self._colors["bg"])
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(90)
        layout.addWidget(self.canvas, stretch=1)

        self._ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.13, right=0.98, top=0.96, bottom=0.17)
        self._style_axes()
        # Re-snapshot the blit background on every full draw (resize, theme
        # change) — otherwise the next highlight would restore a stale one.
        self.canvas.mpl_connect("draw_event", self._on_draw)

    def _style_axes(self):
        ax = self._ax
        ax.set_facecolor(self._colors["surface"])
        ax.tick_params(
            colors=self._colors["text_muted"], length=2, width=0.6, labelsize=7
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(self._colors["border"])
            spine.set_linewidth(0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── data ─────────────────────────────────────────────────────────────────

    def set_traces(self, matrix, cluster_ids, x=None):
        """Rebuild the overlay from *matrix*, one row per entry of *cluster_ids*.

        Rows that are all zero are treated as missing data and skipped (see the
        module docstring). Clears any highlight.
        """
        matrix = np.asarray(matrix, dtype=float) if matrix is not None else None
        cluster_ids = [int(c) for c in (cluster_ids or [])]
        self._n_requested = len(cluster_ids)

        if (
            matrix is None
            or matrix.ndim != 2
            or matrix.shape[0] != len(cluster_ids)
            or matrix.shape[1] < 2
        ):
            self._cluster_ids = []
            self._row_of = {}
            self._segments = []
            self._redraw_background()
            return

        if x is None:
            x = np.arange(matrix.shape[1], dtype=float)
        x = np.asarray(x, dtype=float)

        # Decide what counts as missing on the FULL trace, before decimation —
        # a cell whose whole response is one narrow transient must not be
        # reclassified as "no data" just because the drawing step skipped it.
        real = ~np.all(np.isclose(matrix, 0.0), axis=1) & np.all(
            np.isfinite(matrix), axis=1
        )
        if self._smooth_bins > 0:
            matrix = ndimage.gaussian_filter1d(
                matrix, sigma=self._smooth_bins, axis=1, mode="nearest"
            )
        matrix, x = _decimate(matrix, x, _MAX_POINTS)
        self._cluster_ids = [c for c, keep in zip(cluster_ids, real) if keep]
        self._row_of = {cid: i for i, cid in enumerate(self._cluster_ids)}

        rows = matrix[real]
        if self._normalize == "peak" and rows.size:
            scale = np.max(np.abs(rows), axis=1, keepdims=True)
            rows = rows / np.where(scale > 0, scale, 1.0)

        self._segments = [np.column_stack([x, row]) for row in rows]
        self._redraw_background()

    def set_smoothing(self, sigma_bins: float):
        """Gaussian sigma (in samples) applied for display only.

        Takes effect on the next ``set_traces``; the caller usually only learns
        the right sigma (from a file's bin size) once data is loaded.
        """
        self._smooth_bins = float(sigma_bins or 0.0)

    def set_regions(self, regions):
        """Shade ``[(x0, x1, rgba), ...]`` behind the traces, e.g. stimulus phases."""
        self._regions = list(regions or [])
        self._redraw_background()

    def _redraw_background(self):
        ax = self._ax
        ax.clear()
        self._style_axes()
        self._overlay = None
        self._blit_bg = None

        if not self._segments:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color=self._colors["text_muted"],
            )
            ax.set_xticks([])
            ax.set_yticks([])
            self._update_caption(0)
            self.canvas.draw_idle()
            return

        for x0, x1, rgba in self._regions:
            ax.axvspan(x0, x1, color=rgba, lw=0, zorder=0)

        n = len(self._segments)
        # Deliberately very faint. Several hundred overlaid traces are a
        # hairball at any alpha; the background's job is to give the highlight
        # something to stand against, not to be read on its own.
        ax.add_collection(
            LineCollection(
                self._segments,
                colors=self._colors.get("accent", DEFAULT_COLORS["accent"]),
                linewidths=0.4,
                alpha=0.13,
                zorder=1,
            )
        )

        # One overlay holding every trace; the highlight is a per-element colour
        # change on this artist, so no artist is created or removed while
        # brushing. animated=True keeps it out of the normal draw pass, which is
        # what makes copy_from_bbox() capture a *clean* background — otherwise
        # the snapshot would bake in the current highlight and smear stale
        # orange traces across later selections.
        self._overlay = LineCollection(
            self._segments,
            colors=np.zeros((n, 4)),
            linewidths=np.zeros(n),
            zorder=3,
            animated=True,
        )
        ax.add_collection(self._overlay)

        all_x = np.concatenate([s[:, 0] for s in self._segments])
        all_y = np.concatenate([s[:, 1] for s in self._segments])
        ax.set_xlim(float(all_x.min()), float(all_x.max()))
        lo, hi = float(all_y.min()), float(all_y.max())
        pad = max((hi - lo) * 0.06, 1e-9)
        ax.set_ylim(lo - pad, hi + pad)
        ax.axhline(0.0, color=self._colors["border"], lw=0.6, zorder=0)

        self._update_caption(0)
        self.canvas.draw()

    # ── highlight ────────────────────────────────────────────────────────────

    def highlight(self, cluster_ids):
        """Paint *cluster_ids* in the highlight colour; everything else dims."""
        if self._overlay is None or not self._segments:
            self._update_caption(0)
            return

        n = len(self._segments)
        rows = [
            self._row_of[int(c)] for c in (cluster_ids or []) if int(c) in self._row_of
        ]

        edge = np.zeros((n, 4))
        widths = np.zeros(n)
        if rows:
            # Slightly translucent: a 30-cell group of dense traces painted
            # opaque reads as one solid block, which hides exactly the
            # within-group spread the overlay exists to show.
            rgba = _to_rgba(self._colors["highlight"])
            rgba[3] = 0.8
            edge[rows] = rgba
            widths[rows] = 1.0

        self._overlay.set_colors(edge)
        self._overlay.set_linewidths(widths)

        self._update_caption(len(rows))
        self._blit()

    def clear_highlight(self):
        self.highlight([])

    def _blit(self):
        if self._blit_bg is None:
            # No usable snapshot yet (first paint, or just after a resize).
            # draw() re-enters _on_draw, which snapshots and re-blits for us.
            self.canvas.draw()
            return
        self.canvas.restore_region(self._blit_bg)
        self._ax.draw_artist(self._overlay)
        self.canvas.blit(self.fig.bbox)

    def _on_draw(self, _event):
        """Cache the freshly drawn background and stamp the highlight back on.

        The overlay is animated, so the draw that just finished holds only the
        background — exactly what should be cached. Restamping means a full
        redraw never silently drops the current selection.
        """
        if self._overlay is None:
            self._blit_bg = None
            return
        self._blit_bg = self.canvas.copy_from_bbox(self.fig.bbox)
        self._ax.draw_artist(self._overlay)
        self.canvas.blit(self.fig.bbox)

    # ── chrome ───────────────────────────────────────────────────────────────

    def _update_caption(self, n_selected: int):
        total = len(self._cluster_ids)
        missing = self._n_requested - total
        text = f"{self._title} — {n_selected} of {total} highlighted"
        if missing > 0:
            text += f"  ({missing} without data)"
        self.caption.setText(text)

    def restyle(self, colors: dict):
        """Adopt a new theme and rebuild (highlight is cleared)."""
        if not colors:
            return
        self._colors.update(colors)
        self.fig.patch.set_facecolor(self._colors["bg"])
        self.caption.setStyleSheet(
            f"color: {self._colors['text_muted']}; font-size: 11px; padding: 1px 0;"
        )
        self._redraw_background()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Geometry changed: the cached background no longer matches the canvas.
        self._blit_bg = None


def _decimate(matrix, x, max_points):
    """Thin *matrix* to at most ``max_points`` columns, keeping peaks.

    Plain ``[:, ::step]`` slicing is wrong for spike PSTHs: RGC responses are
    often a single narrow transient, and a stride that misses it erases the
    cell's entire response from the plot. Keeping the largest-magnitude sample
    in each chunk (signed, so ON and OFF stay distinguishable) is the standard
    peak-preserving reduction and keeps every transient visible. It does render
    a noisy trace as its outer envelope, which is the right trade for judging
    whether a group shares a response shape.
    """
    n = matrix.shape[1]
    if n <= max_points:
        return matrix, x
    step = int(np.ceil(n / max_points))
    pad = (-n) % step
    if pad:
        matrix = np.pad(matrix, ((0, 0), (0, pad)), mode="edge")
        x = np.pad(x, (0, pad), mode="edge")
    chunks = matrix.reshape(matrix.shape[0], -1, step)
    peak_idx = np.argmax(np.abs(chunks), axis=2)[:, :, None]
    return (
        np.take_along_axis(chunks, peak_idx, axis=2)[:, :, 0],
        x.reshape(-1, step).mean(axis=1),
    )


def _to_rgba(color):
    from matplotlib.colors import to_rgba

    try:
        return np.array(to_rgba(color))
    except (ValueError, TypeError):
        return np.array(to_rgba(DEFAULT_COLORS["highlight"]))
