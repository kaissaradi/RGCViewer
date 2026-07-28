"""A compact RF mosaic that highlights a live selection.

Both the Feature Extraction popup and the UMAP panel let you lasso a group of
cells out of a scatter plot; this widget answers the question that immediately
follows — "do those cells actually tile the array?" — without leaving the popup.

Every cell's receptive-field ellipse is drawn once as a dim background. The
highlight is then a per-element recolour of a single overlay collection rather
than a redraw, which keeps brushing responsive at the ~700 cells a run holds:
matplotlib's render pipeline is only invoked when the cell set or the theme
changes, never on selection.
"""

from __future__ import annotations

import logging

import numpy as np
from matplotlib.collections import EllipseCollection
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtWidgets import QVBoxLayout, QLabel, QWidget, QSizePolicy

logger = logging.getLogger(__name__)

__all__ = ["RFMapWidget", "collect_rf_ellipses"]


DEFAULT_COLORS = {
    "bg": "#0f1117",
    "surface": "#171923",
    "border": "#2a2d3e",
    "text_primary": "#e8eaf0",
    "text_muted": "#6b7280",
    "highlight": "#f97316",
}


def collect_rf_ellipses(data_manager, cluster_ids) -> dict:
    """Map ``cluster_id -> (cx, cy, width, height, angle_deg)``.

    Cells with no usable Gaussian fit are simply absent from the result, so the
    caller can tell "not selected" from "has no RF". Geometry matches what the
    population panel draws: full diameters (``std * 2``), degrees, and the same
    ``sta_height - y`` flip so this map is in the same frame as the main mosaic.

    Falls back to a reference-run RF (via ReferenceBridge) when the current run
    has no fit for a cell, mirroring plot_population_rfs_background.
    """
    out: dict = {}
    dm = data_manager
    if dm is None:
        return out

    vision_params = getattr(dm, "vision_params", None)
    sta_height = getattr(dm, "vision_sta_height", None)
    is_vision_only = bool(getattr(dm, "is_vision_only", False))

    borrowed = {}
    bridge = getattr(dm, "reference_bridge", None)
    if bridge is not None and callable(getattr(bridge, "get_all_rf_ellipses", None)):
        try:
            raw = bridge.get_all_rf_ellipses()
            if isinstance(raw, dict):
                borrowed = raw
        except Exception:
            logger.debug("collect_rf_ellipses: bridge RF lookup failed", exc_info=True)

    def _vision_id(cid):
        fn = getattr(dm, "get_vision_id_for_cluster", None)
        if callable(fn):
            try:
                vid = fn(cid)
                if isinstance(vid, (int, np.integer)):
                    return int(vid)
            except Exception:
                pass
        return int(cid) if is_vision_only else int(cid) + 1

    def _flip(y):
        return sta_height - y if sta_height is not None else y

    def _entry(cx, cy, sx, sy, rot):
        if not all(np.isfinite(v) for v in (cx, cy, sx, sy, rot)):
            return None
        if sx <= 0 or sy <= 0:
            return None
        return (float(cx), float(_flip(cy)), float(sx * 2), float(sy * 2),
                float(np.degrees(rot)))

    for cid in cluster_ids:
        try:
            cid_int = int(cid)
        except (TypeError, ValueError):
            continue
        vid = _vision_id(cid_int)
        entry = None

        if vision_params is not None:
            try:
                fit = vision_params.get_stafit_for_cell(vid)
            except Exception:
                fit = None
            if fit is not None:
                entry = _entry(
                    fit.center_x, fit.center_y, fit.std_x, fit.std_y, fit.rot
                )

        if entry is None and vid in borrowed:
            p = borrowed[vid]
            try:
                entry = _entry(p["x0"], p["y0"], p["std_x"], p["std_y"], p["angle"])
            except (KeyError, TypeError):
                entry = None

        if entry is not None:
            out[cid_int] = entry

    return out


class RFMapWidget(QWidget):
    """RF mosaic with a blit-updated highlight overlay.

    Usage::

        rf_map = RFMapWidget(colors=...)
        rf_map.set_cells(data_manager, cluster_ids)   # once, when data loads
        rf_map.highlight([12, 44, 91])                # cheap, per selection
    """

    def __init__(self, parent=None, colors: dict = None, title: str = "RF Mosaic"):
        super().__init__(parent)
        self._colors = dict(DEFAULT_COLORS)
        if colors:
            self._colors.update(colors)
        self._title = title

        self._cluster_ids: list = []        # cells that have an RF, in draw order
        self._row_of: dict = {}             # cluster_id -> row in the collections
        self._geometry: np.ndarray = np.empty((0, 5))
        self._n_requested = 0               # cells asked for, incl. those w/o an RF
        self._overlay = None
        self._blit_bg = None
        self._ax = None

        self._build_ui()

    # ── construction ─────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.caption = QLabel(self._title)
        self.caption.setStyleSheet(
            f"color: {self._colors['text_muted']}; font-size: 11px; padding: 2px 0;"
        )
        layout.addWidget(self.caption)

        self.fig = Figure(figsize=(3.2, 3.2), dpi=100)
        self.fig.patch.set_facecolor(self._colors["bg"])
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumWidth(220)
        layout.addWidget(self.canvas, stretch=1)

        self._ax = self.fig.add_subplot(111)
        self._style_axes()
        # Re-snapshot the blit background whenever matplotlib does a full draw
        # (resize, theme change) — otherwise the first highlight after a resize
        # would restore a stale, wrongly-sized background.
        self.canvas.mpl_connect("draw_event", self._on_draw)

    def _style_axes(self):
        ax = self._ax
        ax.set_facecolor(self._colors["surface"])
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(
            colors=self._colors["text_muted"], length=2, width=0.6, labelsize=7
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(self._colors["border"])
            spine.set_linewidth(0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── data ─────────────────────────────────────────────────────────────────

    def set_cells(self, data_manager, cluster_ids):
        """Rebuild the background mosaic for *cluster_ids*. Clears any highlight."""
        cluster_ids = [int(c) for c in (cluster_ids or [])]
        self._n_requested = len(cluster_ids)

        ellipses = collect_rf_ellipses(data_manager, cluster_ids)
        # Preserve the caller's ordering so row indices are predictable.
        self._cluster_ids = [c for c in cluster_ids if c in ellipses]
        self._row_of = {cid: i for i, cid in enumerate(self._cluster_ids)}
        self._geometry = (
            np.array([ellipses[c] for c in self._cluster_ids], dtype=float)
            if self._cluster_ids
            else np.empty((0, 5))
        )

        self._redraw_background()

    def _redraw_background(self):
        ax = self._ax
        ax.clear()
        self._style_axes()
        self._overlay = None
        self._blit_bg = None

        if self._geometry.shape[0] == 0:
            ax.text(
                0.5,
                0.5,
                "No RF fits available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color=self._colors["text_muted"],
            )
            ax.set_xticks([])
            ax.set_yticks([])
            self._update_caption(0)
            self.canvas.draw_idle()
            return

        geom = self._geometry

        base = EllipseCollection(
            widths=geom[:, 2],
            heights=geom[:, 3],
            angles=geom[:, 4],
            units="x",
            offsets=geom[:, :2],
            offset_transform=ax.transData,
            edgecolors=self._colors["border"],
            facecolors="none",
            linewidths=0.7,
            alpha=0.55,
            zorder=1,
        )
        ax.add_collection(base)

        # One overlay holding every ellipse; the highlight is a per-element
        # alpha change on this artist, so no artist is ever created or removed
        # during brushing.
        #
        # animated=True keeps it out of the normal draw pass, which is what
        # makes copy_from_bbox() capture a *clean* background. Without it the
        # snapshot would bake in the current highlight and restoring it would
        # smear stale orange ellipses across later selections.
        self._overlay = EllipseCollection(
            widths=geom[:, 2],
            heights=geom[:, 3],
            angles=geom[:, 4],
            units="x",
            offsets=geom[:, :2],
            offset_transform=ax.transData,
            edgecolors=np.zeros((geom.shape[0], 4)),  # fully transparent
            facecolors="none",
            linewidths=np.zeros(geom.shape[0]),
            zorder=3,
            animated=True,
        )
        ax.add_collection(self._overlay)

        cx, cy = geom[:, 0], geom[:, 1]
        rx, ry = geom[:, 2] / 2.0, geom[:, 3] / 2.0
        x_lo, x_hi = np.min(cx - rx), np.max(cx + rx)
        y_lo, y_hi = np.min(cy - ry), np.max(cy + ry)
        mx = max((x_hi - x_lo) * 0.05, 5.0)
        my = max((y_hi - y_lo) * 0.05, 5.0)
        ax.set_xlim(x_lo - mx, x_hi + mx)
        ax.set_ylim(y_lo - my, y_hi + my)

        self._update_caption(0)
        self.canvas.draw()

    # ── highlight ────────────────────────────────────────────────────────────

    def highlight(self, cluster_ids):
        """Paint *cluster_ids* orange; everything else returns to the background."""
        if self._overlay is None or self._geometry.shape[0] == 0:
            self._update_caption(len(cluster_ids or []))
            return

        n = self._geometry.shape[0]
        rows = [
            self._row_of[int(c)]
            for c in (cluster_ids or [])
            if int(c) in self._row_of
        ]

        edge = np.zeros((n, 4))
        widths = np.zeros(n)
        if rows:
            rgba = _to_rgba(self._colors["highlight"])
            edge[rows] = rgba
            widths[rows] = 1.6

        self._overlay.set_edgecolors(edge)
        self._overlay.set_linewidths(widths)

        self._update_caption(len(cluster_ids or []))
        self._blit()

    def clear_highlight(self):
        self.highlight([])

    def _blit(self):
        """Restore the clean background and stamp only the overlay onto it."""
        if self._blit_bg is None:
            # No usable snapshot yet (first paint, or just after a resize).
            # draw() re-enters _on_draw, which snapshots and re-blits for us.
            self.canvas.draw()
            return
        self.canvas.restore_region(self._blit_bg)
        self._ax.draw_artist(self._overlay)
        self.canvas.blit(self.fig.bbox)

    def _on_draw(self, _event):
        """Snapshot the freshly drawn background and repaint the highlight.

        The overlay is animated, so the draw that just finished contains only
        the background — exactly what should be cached. The highlight is then
        stamped back on, so a full redraw (resize, theme change, Qt asking the
        widget to repaint) never silently drops the current selection.
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
            text += f"  ({missing} without RF fit)"
        self.caption.setText(text)

    def restyle(self, colors: dict):
        """Adopt a new theme and rebuild (highlight is cleared)."""
        if not colors:
            return
        self._colors.update(colors)
        self.fig.patch.set_facecolor(self._colors["bg"])
        self.caption.setStyleSheet(
            f"color: {self._colors['text_muted']}; font-size: 11px; padding: 2px 0;"
        )
        self._redraw_background()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Geometry changed: the cached background no longer matches the canvas.
        self._blit_bg = None


def _to_rgba(color):
    from matplotlib.colors import to_rgba

    try:
        return np.array(to_rgba(color))
    except (ValueError, TypeError):
        return np.array(to_rgba(DEFAULT_COLORS["highlight"]))
