"""Rectangle/lasso brushing that previews while you drag.

Matplotlib's stock ``RectangleSelector`` and ``LassoSelector`` only report a
selection on mouse *release*, and each keeps its own background snapshot to
blit against. Both are wrong for this app:

* Reporting on release means you cannot nudge a boundary and watch the
  population change — which is the entire point of brushing an embedding
  against an RF mosaic.
* Independent snapshots mean whichever artist painted last wipes the others.
  With several selectors and a highlight overlay sharing one canvas, the
  rubber band vanished the moment a selection landed.

The classes here fix both by routing every repaint through a single owner that
composites the whole canvas in a fixed order. ``FeatureExtractionWindow`` and
``UMAPPanel`` are that owner for their scatter canvases;
``PopulationSelectionMixin`` plays the same role inside the small population
widgets, so a selection can be drawn on the RF mosaic or a trace stack too.

An owner must provide:

``_defer_composite``  bool — suppresses compositing mid-gesture
``_composite()``      restore the cached background, stamp the live layers
``_begin_selection(sel)``   a new gesture started on *sel*
``_on_rect_live(sel)``      the rectangle moved; recompute the selection
``_on_lasso_live(sel)``     the lasso moved; recompute the selection
``_freeze_lasso(sel, verts)``  keep the released lasso path on screen
"""

from __future__ import annotations

import logging

import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector, RectangleSelector

logger = logging.getLogger(__name__)

__all__ = [
    "LiveLassoSelector",
    "LiveRectangleSelector",
    "PopulationSelectionMixin",
    "_axes_ready",
    "make_lasso_selector",
    "make_rect_selector",
    "path_from_extents",
]


class _CompositedSelector:
    """Mixin: hand every repaint to the owning widget's single blit compositor.

    Each matplotlib selector normally keeps its *own* background snapshot and
    blits independently. Routing everything through the owner's ``_composite``
    gives one artist stack with a fixed draw order instead, so no layer can
    wipe another.
    """

    _owner = None

    def update(self):
        owner = self._owner
        if owner is None:
            return super().update()
        if not owner._defer_composite:
            owner._composite()
        return True

    def update_background(self, event):
        # The owner re-snapshots on draw_event. The base implementation forces
        # a full canvas.draw() on every button press, which is the stall this
        # used to have between clicking and seeing the rubber band.
        return

    def _press(self, event):
        if self._owner is not None:
            self._owner._begin_selection(self)
        return super()._press(event)


class LiveRectangleSelector(_CompositedSelector, RectangleSelector):
    """Rectangle whose extents are reported on every move, not just on release.

    That is what makes dragging an edge handle re-filter the population live
    rather than only when the mouse comes up.
    """

    def _onmove(self, event):
        owner = self._owner
        if owner is None:
            return super()._onmove(event)
        # Suppress the intermediate composite the base class triggers; the
        # selection is recomputed first so the shape and the points move
        # together.
        owner._defer_composite = True
        try:
            super()._onmove(event)
        finally:
            owner._defer_composite = False
        owner._on_rect_live(self)


class LiveLassoSelector(_CompositedSelector, LassoSelector):
    """Lasso that previews its contents while you draw and stays on screen after."""

    def _onmove(self, event):
        owner = self._owner
        if owner is None:
            return super()._onmove(event)
        owner._defer_composite = True
        try:
            super()._onmove(event)
        finally:
            owner._defer_composite = False
        owner._on_lasso_live(self)

    def _release(self, event):
        verts = list(self.verts) if self.verts else []
        # The base implementation fires onselect and then throws the path away.
        super()._release(event)
        if self._owner is not None and len(verts) >= 3:
            self._owner._freeze_lasso(self, verts)


# ──────────────────────────────────────────────────────────────────────────────
#  Factories — one place for the look of a selection, so the embedding, the
#  mosaic and the trace stacks all draw the same rubber band.
# ──────────────────────────────────────────────────────────────────────────────


def _axes_ready(ax) -> bool:
    """True when *ax* has a laid-out figure RectangleSelector can measure.

    A hidden or collapsed Qt canvas reports a 0×0 figure. Matplotlib then
    raises ``ValueError: 'box_aspect' and 'fig_aspect' must be positive``
    inside ``RectangleSelector.__init__``.
    """
    if ax is None:
        return False
    fig = getattr(ax, "figure", None)
    if fig is None:
        return False
    try:
        width, height = fig.get_size_inches()
    except Exception:
        return False
    if not (np.isfinite(width) and np.isfinite(height) and width > 0 and height > 0):
        return False
    canvas = getattr(fig, "canvas", None)
    if canvas is None:
        return True
    getter = getattr(canvas, "get_width_height", None)
    if not callable(getter):
        return True
    try:
        pix_w, pix_h = getter()
    except Exception:
        return True
    return pix_w > 0 and pix_h > 0


def make_rect_selector(ax, owner, palette, onselect=None, index=None):
    """An interactive rectangle on *ax*, owned by *owner*.

    Returns ``None`` when the axes are not laid out yet — the caller should
    retry once the canvas has a real size.
    """
    if not _axes_ready(ax):
        return None

    def _noop(_eclick, _erelease):
        # The live handler has already applied the selection; the release
        # callback only has to leave the shape on screen.
        owner._on_rect_live(sel)

    try:
        sel = LiveRectangleSelector(
            ax,
            onselect or _noop,
            useblit=True,
            button=[1],
            interactive=True,  # corner/edge handles, draggable after release
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            drag_from_anywhere=True,
            props=dict(
                edgecolor=palette["text_primary"],
                facecolor=palette["accent"],
                alpha=0.14,
                linewidth=1.2,
                linestyle="--",
            ),
            handle_props=dict(
                marker="s",
                markersize=6,
                markerfacecolor=palette["text_primary"],
                markeredgecolor=palette["bg"],
                alpha=0.95,
            ),
        )
    except ValueError:
        # Figure size can still race to 0 between the check and __init__.
        logger.debug("rectangle selector skipped: axes not laid out yet", exc_info=True)
        return None
    sel._owner = owner
    sel._plot_index = index
    return sel


def make_lasso_selector(ax, owner, palette, onselect=None, index=None):
    """A lasso on *ax* that previews as it is drawn, owned by *owner*.

    Returns ``None`` when the axes are not laid out yet.
    """
    if not _axes_ready(ax):
        return None

    def _noop(verts):
        owner._apply_lasso_verts(index, verts)

    try:
        sel = LiveLassoSelector(
            ax,
            onselect or _noop,
            useblit=True,
            button=[1],
            props=dict(
                color=palette["text_primary"],
                linewidth=1.4,
                linestyle="-",
                alpha=0.85,
            ),
        )
    except ValueError:
        logger.debug("lasso selector skipped: axes not laid out yet", exc_info=True)
        return None
    sel._owner = owner
    sel._plot_index = index
    return sel


def path_from_extents(extents):
    """``(xmin, xmax, ymin, ymax)`` as a closed vertex list."""
    xmin, xmax, ymin, ymax = extents
    return [
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
        (xmin, ymin),
    ]


def retire(sel):
    """Hide a selector's shape and reset it, so its next press starts fresh.

    Hiding alone is not enough: matplotlib decides between "start a new
    rectangle" and "grab an existing handle" from the artist's visibility and
    ``_selection_completed``, so a merely-hidden selector goes on offering
    invisible handles.
    """
    clear = getattr(sel, "_clear_without_update", None)
    if callable(clear):
        try:
            clear()
            return
        except Exception:
            logger.debug("selector clear failed", exc_info=True)
    try:
        sel.set_visible(False)
    except Exception:
        logger.debug("selector hide failed", exc_info=True)


# ──────────────────────────────────────────────────────────────────────────────
#  Selection inside a population view
# ──────────────────────────────────────────────────────────────────────────────


class PopulationSelectionMixin:
    """Brushing support for the small blit-composited population widgets.

    ``RFMapWidget`` and ``TraceStackWidget`` both draw a dim background plus one
    *animated* highlight overlay, and both already restore a cached background
    before stamping that overlay. This bolts a selector onto that same
    pipeline rather than a second one beside it: the rubber band and the frozen
    lasso outline join the stamping order, so the selection shape, the
    highlight and the background cannot wipe each other.

    The host widget must define:

    ``self.fig`` / ``self.canvas`` / ``self._ax``   the matplotlib stack
    ``self._blit_bg``                              cached background, or None
    ``self._overlay``                              animated highlight artist
    ``self._ids_within_path(path)``                hit test, returns cluster IDs

    and must call ``_draw_selection_layers()`` from wherever it stamps the
    overlay, and ``_discard_selection_shapes()`` whenever it rebuilds the axes.
    """

    #: Subclasses that want brushing set these from their own signals.
    selection_changed = None
    context_menu_requested = None

    def _init_selection(self):
        self._defer_composite = False
        self._sel_mode = "off"
        self._rect_sel = None
        self._lasso_sel = None
        self._lasso_poly = None
        self._sel_ids: list = []

    def _maybe_request_context_menu(self, event) -> bool:
        """Forward a right-click to the owning window's context menu.

        The menu itself belongs to the window, not the view: acting on a
        selection means grouping or trashing it, which the small widgets know
        nothing about. Returns True when the click was consumed.
        """
        if event.button != 3:
            return False
        signal = getattr(self, "context_menu_requested", None)
        if signal is None or not hasattr(signal, "emit"):
            return False
        signal.emit()
        return True

    # ── public API ───────────────────────────────────────────────────────────

    def set_selection_mode(self, mode: str):
        """``'off'`` | ``'rect'`` | ``'lasso'`` — which brush is live here.

        Every view in a window shares one mode, so the user picks a tool once
        and it applies wherever they drag.
        """
        if mode not in ("off", "rect", "lasso"):
            mode = "off"
        self._sel_mode = mode
        self._ensure_selectors()

        if self._rect_sel is not None:
            self._rect_sel.set_active(mode == "rect")
            if mode != "rect":
                retire(self._rect_sel)
        if self._lasso_sel is not None:
            self._lasso_sel.set_active(mode == "lasso")
            if mode != "lasso":
                retire(self._lasso_sel)
        if mode != "lasso" and self._lasso_poly is not None:
            self._lasso_poly.set_visible(False)

        if self._rect_sel is not None or self._lasso_sel is not None:
            self._composite()

    def clear_selection_shapes(self):
        """Drop the rubber band without touching the highlight."""
        for sel in (self._rect_sel, self._lasso_sel):
            if sel is not None:
                retire(sel)
        if self._lasso_poly is not None:
            self._lasso_poly.set_visible(False)
        self._composite()

    # ── wiring ───────────────────────────────────────────────────────────────

    def _selection_palette(self):
        c = self._colors
        return {
            "bg": c["bg"],
            "accent": c["accent"],
            "text_primary": c["text_primary"],
        }

    def _ensure_selectors(self):
        """Create the selectors lazily, and only once there are axes to own them.

        Hidden or collapsed views have a 0×0 figure. Creating a
        ``RectangleSelector`` on those raises; we wait for a later resize.
        """
        if self._ax is None or self._sel_mode == "off":
            return
        if not _axes_ready(self._ax):
            return
        palette = self._selection_palette()
        if self._rect_sel is None:
            self._rect_sel = make_rect_selector(self._ax, self, palette)
        if self._lasso_sel is None:
            self._lasso_sel = make_lasso_selector(self._ax, self, palette)
        # Redrawing the background discards the selectors and calls back here,
        # so the armed tool has to be re-applied — otherwise brushing silently
        # stops working after the next set_traces()/set_cells().
        if self._rect_sel is not None:
            self._rect_sel.set_active(self._sel_mode == "rect")
        if self._lasso_sel is not None:
            self._lasso_sel.set_active(self._sel_mode == "lasso")

    def _discard_selection_shapes(self):
        """Forget selectors bound to axes that are about to be cleared.

        ``ax.clear()`` leaves a selector's canvas callbacks connected, so a
        stale one goes on handling events for data that no longer exists.
        """
        for sel in (self._rect_sel, self._lasso_sel):
            if sel is None:
                continue
            try:
                sel.disconnect_events()
            except Exception:
                logger.debug("selector disconnect failed", exc_info=True)
        self._rect_sel = None
        self._lasso_sel = None
        self._lasso_poly = None

    # ── owner protocol ───────────────────────────────────────────────────────

    def _begin_selection(self, sel):
        other = self._lasso_sel if sel is self._rect_sel else self._rect_sel
        if other is not None:
            retire(other)
        if self._lasso_poly is not None:
            self._lasso_poly.set_visible(False)

    def _on_rect_live(self, sel):
        try:
            verts = path_from_extents(sel.extents)
        except Exception:
            logger.debug("rectangle extents unavailable", exc_info=True)
            self._composite()
            return
        self._emit_selection(verts)

    def _on_lasso_live(self, sel):
        verts = sel.verts
        if not verts or len(verts) < 3:
            self._composite()
            return
        self._emit_selection(verts)

    def _apply_lasso_verts(self, _index, verts):
        if not verts or len(verts) < 3:
            self._composite()
            return
        self._emit_selection(verts)

    def _freeze_lasso(self, _sel, verts):
        """Keep the drawn path on screen after release.

        ``LassoSelector`` blanks its own line in ``_release``, so without this
        the outline disappears exactly when you want to look at it.
        """
        if self._ax is None:
            return
        xy = np.asarray(verts, dtype=float)
        if self._lasso_poly is None:
            self._lasso_poly = MplPolygon(
                xy,
                closed=True,
                facecolor=to_rgba(self._colors["accent"], 0.14),
                edgecolor=to_rgba(self._colors["text_primary"], 0.85),
                linewidth=1.4,
                zorder=5,
                animated=True,
            )
            self._ax.add_patch(self._lasso_poly)
        else:
            self._lasso_poly.set_xy(xy)
        self._lasso_poly.set_visible(True)
        self._composite()

    def _emit_selection(self, verts):
        ids = self._ids_within_path(Path(np.asarray(verts, dtype=float)))
        self._sel_ids = ids
        signal = getattr(self, "selection_changed", None)
        if signal is not None and hasattr(signal, "emit"):
            signal.emit(list(ids))
        else:
            # No listener: at least show what was brushed here.
            self.highlight(ids)
        self._composite()

    # ── compositing ──────────────────────────────────────────────────────────

    def _draw_selection_layers(self):
        """Stamp the frozen lasso and the live rubber band. Called from _blit."""
        if self._lasso_poly is not None and self._lasso_poly.get_visible():
            self._ax.draw_artist(self._lasso_poly)
        for sel in (self._rect_sel, self._lasso_sel):
            if sel is None or not sel.active:
                continue
            for artist in sel.artists:
                if artist.get_visible() and artist.axes is not None:
                    artist.axes.draw_artist(artist)

    def _composite(self):
        """Repaint every live layer over the cached background."""
        if self._defer_composite:
            return
        self._blit()
