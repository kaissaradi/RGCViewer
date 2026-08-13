"""
Population Panel for RGC Viewer

This module contains population plotting functions that were previously in the plotting module.
"""

import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.collections import LineCollection, EllipseCollection
from qtpy.QtGui import QColor

from ..theme import (
    DARK_COLORS,
    is_light_theme,
    plot_ensemble_alpha,
    plot_field,
    plot_rf_bg_alpha,
    plot_rf_target_alpha,
    plot_stroke,
    resolve_theme_colors,
)
from ...analysis import grating_calc

# Set matplotlib logging level to WARNING to suppress font debug messages
matplotlib_logger = logging.getLogger("matplotlib.font_manager")
matplotlib_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_group_timecourse_cache = {}
_group_acg_cache = {}
_group_fr_cache = {}
_rf_background_cache = {}
_rf_background_cache_order = []
_RF_CACHE_MAX = 10


def invalidate_population_caches():
    """Clear population-panel caches after source data or membership changes."""
    _group_timecourse_cache.clear()
    _group_acg_cache.clear()
    _group_fr_cache.clear()
    _rf_background_cache.clear()
    _rf_background_cache_order.clear()


def population_group_plots_cached(subset_ids):
    """True when this folder's timecourse and ACG plots are already in RAM.

    Cell-to-cell scroll inside a group used to call both draw functions on
    every selection. Those plots show the group, not the selected cell.
    """
    if not subset_ids:
        return False
    key = frozenset(subset_ids)
    return key in _group_timecourse_cache and key in _group_acg_cache


def _first_plot_artist(plot_result):
    if isinstance(plot_result, (list, tuple)):
        return plot_result[0] if plot_result else None
    try:
        return plot_result[0]
    except Exception:
        return plot_result


def _safe_axis_limits(get_limits):
    try:
        limits = tuple(float(v) for v in get_limits())
    except Exception:
        return None
    return limits if len(limits) == 2 else None


def _show_population_ids(main_window):
    return (
        hasattr(main_window, "pop_show_ids_checkbox")
        and main_window.pop_show_ids_checkbox.isChecked()
    )


def _snapshot_rf_background(ax, colors, show_ids):
    """
    Store the geometry arrays from EllipseCollections/LineCollections on the
    axes, not individual patch objects. Replay via _draw_cached_rf_background
    is then a handful of add_collection calls instead of N add_patch calls.

    Captures LineCollections (DS/OS arrow/tick markers) in addition to the
    original EllipseCollections (RF ellipses) — these were previously
    dropped from the cache entirely, meaning any cache HIT would silently
    omit DS/OS markers even though plot_population_rfs_background drew them
    correctly on the cache MISS that produced the snapshot.
    """
    collections_data = []
    for coll in ax.collections:
        # Only EllipseCollections (the RF ellipses) are cached. DS/OS marker
        # LineCollections are deliberately NOT captured here: they're drawn
        # fresh on every redraw by _draw_dsos_markers (which runs AFTER this
        # snapshot is taken), so they're never present on the axes at
        # snapshot time anyway — and _draw_cached_rf_background only knows
        # how to replay ellipses, so capturing a line entry would KeyError
        # on replay. Ellipse-only keeps capture and replay consistent.
        if isinstance(coll, EllipseCollection):
            offsets = coll.get_offsets()
            widths = coll._widths
            heights = coll._heights
            angles = coll._angles
            collections_data.append(
                {
                    "kind": "ellipse",
                    "offsets": np.array(offsets),
                    "widths": np.array(widths),
                    "heights": np.array(heights),
                    "angles": np.array(angles),
                    "edgecolor": coll.get_edgecolor()[0],
                    "alpha": coll.get_alpha(),
                    "lw": (
                        coll.get_linewidth()[0]
                        if hasattr(coll.get_linewidth(), "__len__")
                        else coll.get_linewidth()
                    ),
                    "zorder": coll.get_zorder(),
                }
            )

    texts = []
    for text in getattr(ax, "texts", []):
        texts.append(
            {
                "position": text.get_position(),
                "text": text.get_text(),
                "color": text.get_color(),
                "fontsize": text.get_fontsize(),
                "ha": text.get_ha(),
                "va": text.get_va(),
                "alpha": text.get_alpha(),
            }
        )

    title = ax.get_title()
    if not isinstance(title, str):
        title = None

    return {
        "colors": dict(colors),
        "show_ids": show_ids,
        "collections": collections_data,
        "texts": texts,
        # Carried so a cache hit stays clickable; the geometry alone cannot say
        # which cell an outline belongs to.
        "hit_ids": np.array(getattr(ax, "_rf_hit_ids", np.empty(0, dtype=int))),
        "hit_geom": np.array(getattr(ax, "_rf_hit_geom", np.empty((0, 5)))),
        "xlim": _safe_axis_limits(ax.get_xlim),
        "ylim": _safe_axis_limits(ax.get_ylim),
        "title": title,
    }


# How close (in screen pixels) a click has to land to an RF outline to count.
_RF_CLICK_TOLERANCE_PX = 8.0


def set_rf_hit_entries(ax, entries):
    """Record ``(vision_id, cx, cy, width, height, angle_deg)`` per drawn RF.

    Stashed on the axes because the ellipses are drawn as EllipseCollections,
    which carry geometry but no identity — there is otherwise nothing to map a
    click back to a cell with.
    """
    ax._rf_hit_ids = np.array([e[0] for e in entries], dtype=int)
    ax._rf_hit_geom = (
        np.array([e[1:] for e in entries], dtype=float)
        if entries
        else np.empty((0, 5))
    )


def rf_vision_id_at(ax, x, y, tolerance_px=_RF_CLICK_TOLERANCE_PX):
    """Vision id of the RF whose outline is nearest ``(x, y)``, or None.

    RFs in a mosaic overlap heavily, so "inside" is ambiguous and the boundary
    is what a user actually aims at. Candidates are ranked by distance from
    *their own* outline (r == 1 in each ellipse's normalised frame), which
    picks the small RF whose edge was clicked over the large one the click
    merely happens to fall inside.
    """
    geom = getattr(ax, "_rf_hit_geom", None)
    ids = getattr(ax, "_rf_hit_ids", None)
    if geom is None or ids is None or geom.shape[0] == 0:
        return None
    if x is None or y is None:
        return None

    cx, cy, w, h, ang = geom.T
    a = np.maximum(w / 2.0, 1e-9)
    b = np.maximum(h / 2.0, 1e-9)
    theta = np.radians(ang)
    dx, dy = x - cx, y - cy
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    u = (dx * cos_t + dy * sin_t) / a
    v = (-dx * sin_t + dy * cos_t) / b
    r = np.hypot(u, v)

    # One pixel tolerance means a different radial tolerance for every ellipse.
    try:
        inv = ax.transData.inverted()
        x0, y0 = inv.transform((0.0, 0.0))
        x1, y1 = inv.transform((tolerance_px, tolerance_px))
        tol_data = max(abs(x1 - x0), abs(y1 - y0))
    except Exception:
        logger.debug("RF click tolerance transform failed", exc_info=True)
        tol_data = 0.0
    tol = tol_data / np.minimum(a, b)

    candidates = np.where(r <= 1.0 + tol)[0]
    if candidates.size == 0:
        return None
    return int(ids[candidates[np.argmin(np.abs(r[candidates] - 1.0))]])


def _apply_rf_axes_style(ax, colors, title=None):
    if title:
        ax.set_title(title, color=colors["text_primary"])
    ax.set_facecolor(plot_field(colors))
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(colors=colors["text_secondary"])
    for spine in ax.spines.values():
        spine.set_edgecolor(colors["border_subtle"])
    ax.grid(False)


def _draw_cached_rf_background(ax, cache_entry, colors):
    """Replay cached EllipseCollections — 2 add_collection calls, not N add_patch."""
    for cd in cache_entry.get("collections", []):
        ec = EllipseCollection(
            widths=cd["widths"] * 2,
            heights=cd["heights"] * 2,
            angles=cd["angles"] * 180 / np.pi,
            units="x",
            offsets=cd["offsets"],
            offset_transform=ax.transData,
            edgecolors=cd["edgecolor"],
            facecolors="none",
            linewidths=cd["lw"],
            alpha=cd["alpha"],
            zorder=cd["zorder"],
        )
        ax.add_collection(ec)

    for text_data in cache_entry["texts"]:
        ax.text(
            text_data["position"][0],
            text_data["position"][1],
            text_data["text"],
            color=text_data["color"],
            fontsize=text_data["fontsize"],
            ha=text_data["ha"],
            va=text_data["va"],
            alpha=text_data["alpha"],
        )

    ax._rf_hit_ids = np.array(cache_entry.get("hit_ids", np.empty(0, dtype=int)))
    ax._rf_hit_geom = np.array(cache_entry.get("hit_geom", np.empty((0, 5))))

    if cache_entry["xlim"] is not None:
        ax.set_xlim(*cache_entry["xlim"])
    if cache_entry["ylim"] is not None:
        ax.set_ylim(*cache_entry["ylim"])
    _apply_rf_axes_style(ax, colors, cache_entry.get("title"))


def _rf_cache_entry_matches(cache_entry, colors, show_ids):
    return (
        isinstance(cache_entry, dict)
        and "collections" in cache_entry
        and cache_entry.get("colors") == colors
        and cache_entry.get("show_ids") == show_ids
    )


def _store_rf_cache_entry(cache_key, cache_entry):
    _rf_background_cache[cache_key] = cache_entry
    if cache_key in _rf_background_cache_order:
        _rf_background_cache_order.remove(cache_key)
    _rf_background_cache_order.append(cache_key)
    while len(_rf_background_cache_order) > _RF_CACHE_MAX:
        evicted = _rf_background_cache_order.pop(0)
        _rf_background_cache.pop(evicted, None)


def draw_population_timecourse_panel(main_window, subset_ids=None):
    """
    Draw population average timecourse with futuristic "shadow traces".
    OPTIMIZATION: Uses Hot-Swap rendering, explicit scaling, and the
    O(1) Physics Cache to guarantee instant scrolling.
    """
    if subset_ids is None:
        try:
            subset_ids = main_window._get_pop_subset_ids()
        except Exception:
            subset_ids = []

    canvas = main_window.pop_timecourse_canvas
    colors = resolve_theme_colors(main_window.get_current_colors())

    # Early exit: nothing selected
    if not subset_ids:
        canvas.fig.clear()
        canvas.fig.set_facecolor(plot_field(colors))
        canvas.fig.text(
            0.5,
            0.5,
            "No cells selected",
            ha="center",
            color=colors["text_secondary"],
            fontsize=10,
        )
        canvas.draw_idle()
        main_window.pop_timecourse_summary.setText("n=0  mean_t2p: N/A  mean_fwhm: N/A")
        if hasattr(canvas, "_timecourse_state"):
            del canvas._timecourse_state
        return

    # --- 1. Fast Data Extraction via Physics Cache ---
    cache_key = frozenset(subset_ids)
    cached = _group_timecourse_cache.get(cache_key)
    if cached is not None:
        arr = cached["arr"]
        mean_tc = cached["mean_tc"]
        t_axis = cached["t_axis"]
        peak_idx = cached["peak_idx"]
        mean_fwhm = cached["mean_fwhm"]
    else:
        traces = []
        for cid in subset_ids:
            physics = main_window.data_manager.get_cell_physics(cid)
            tc = physics.get("timecourse")
            if tc is not None:
                traces.append(tc)

        if not traces:
            canvas.fig.clear()
            canvas.fig.set_facecolor(plot_field(colors))
            canvas.fig.text(
                0.5,
                0.5,
                "No valid timecourses",
                ha="center",
                color=colors["text_secondary"],
                fontsize=10,
            )
            canvas.draw_idle()
            return

        minlen = min(len(t) for t in traces)
        arr = np.vstack([t[:minlen] for t in traces])
        mean_tc = np.nanmean(arr, axis=0)
        t_axis = np.arange(minlen)
        peak_idx = int(np.argmax(np.abs(mean_tc)))

        mean_fwhm = float("nan")
        try:
            from scipy.signal import peak_widths

            widths, *_ = peak_widths(np.abs(mean_tc), [peak_idx], rel_height=0.5)
            if len(widths) > 0:
                mean_fwhm = widths[0]
        except Exception:
            pass

        _group_timecourse_cache[cache_key] = {
            "arr": arr,
            "mean_tc": mean_tc,
            "t_axis": t_axis,
            "peak_idx": peak_idx,
            "mean_fwhm": mean_fwhm,
        }

    peak_time = t_axis[peak_idx]
    peak_val = mean_tc[peak_idx]

    segments = [np.column_stack([t_axis, row]) for row in arr]

    # Robust scaling: a single wild-outlier trace would otherwise stretch the
    # y-axis and flatten the rest of the population into an unreadable band.
    # Use the 1st/99th percentile across all traces as the "typical" envelope —
    # a trace that exceeds it just runs off the top/bottom edge, which is
    # itself a visible signal that it's an outlier.
    finite_vals = arr[np.isfinite(arr)]
    if finite_vals.size == 0:
        y_lo, y_hi = -1.0, 1.0
    else:
        y_lo, y_hi = np.nanpercentile(finite_vals, [1, 99])
    y_range = y_hi - y_lo
    if y_range == 0:
        y_range = 1.0

    y_bottom = y_lo - (0.1 * y_range)
    y_top = y_hi + (0.3 * y_range)  # extra headroom for the "Peak" label

    # --- 2. Hot-Swap Rendering ---
    if (
        hasattr(canvas, "_timecourse_state")
        and canvas._timecourse_state["ax"] in canvas.fig.axes
    ):
        # Fast update
        state = canvas._timecourse_state
        ax = state["ax"]

        # Check if theme changed (background color mismatch)
        current_facecolor = QColor(plot_field(colors)).name().lower()
        # ax.get_facecolor() returns RGBA tuple, need to unpack it
        facecolor_tuple = ax.get_facecolor()
        stored_facecolor = (
            QColor.fromRgbF(
                facecolor_tuple[0],
                facecolor_tuple[1],
                facecolor_tuple[2],
                facecolor_tuple[3],
            )
            .name()
            .lower()
        )

        if current_facecolor != stored_facecolor:
            # Force full rebuild on theme change
            if hasattr(canvas, "_timecourse_state"):
                del canvas._timecourse_state
            draw_population_timecourse_panel(main_window, subset_ids)
            return

        state["mean_line"].set_data(t_axis, mean_tc)
        state["shadow_lines"].set_segments(segments)
        state["peak_marker"].set_data([peak_time], [peak_val])

        state["peak_text"].set_position((peak_time, peak_val + (np.max(mean_tc) * 0.1)))
        state["peak_text"].set_text(f" Peak\n Frame {peak_time}")

        ax.set_xlim(t_axis[0], t_axis[-1])
        ax.set_ylim(y_bottom, y_top)

    else:
        from matplotlib.collections import LineCollection

        # Full Rebuild
        canvas.fig.clear()
        canvas.fig.set_facecolor(plot_field(colors))
        ax = canvas.fig.add_subplot(111)
        ax.set_facecolor(plot_field(colors))

        # 1. Zero Line
        ax.axhline(
            0,
            color=colors["text_primary"],
            linestyle="--",
            linewidth=1.0,
            alpha=0.2,
            zorder=1,
        )

        # 2. Ensemble: a blue wash so the black mean can be read.
        shadow_lines = LineCollection(
            segments,
            color=colors.get("plot_ensemble", colors["plot_shadow"]),
            linewidth=plot_stroke(colors, "thin"),
            alpha=plot_ensemble_alpha(colors),
            zorder=2,
        )
        ax.add_collection(shadow_lines)

        # 3. Mean trace — heavier in light mode so it sits on white.
        mean_line = _first_plot_artist(
            ax.plot(
                t_axis,
                mean_tc,
                color=colors["plot_mean"],
                linewidth=plot_stroke(colors),
                alpha=0.95,
                zorder=4,
            )
        )

        # 4. Highlight the Peak Feature
        peak_marker = _first_plot_artist(
            ax.plot(
                [peak_time],
                [peak_val],
                "o",
                color=colors["plot_peak"],
                markersize=6,
                zorder=5,
            )
        )
        peak_text = ax.text(
            peak_time,
            peak_val + (np.max(mean_tc) * 0.1),
            f" Peak\n Frame {peak_time}",
            color=colors["plot_peak"],
            fontsize=8,
            ha="center",
            va="bottom",
        )

        # Aesthetics
        ax.set_xlabel("Time (frames)", color=colors["text_secondary"], fontsize=9)
        ax.set_ylabel("Response (a.u.)", color=colors["text_secondary"], fontsize=9)

        ax.set_xlim(t_axis[0], t_axis[-1])
        ax.set_ylim(y_bottom, y_top)

        ax.tick_params(colors=colors["text_secondary"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(colors["border_subtle"])

        ax.grid(False)

        # Save to state
        canvas._timecourse_state = {
            "ax": ax,
            "mean_line": mean_line,
            "shadow_lines": shadow_lines,
            "peak_marker": peak_marker,
            "peak_text": peak_text,
        }

    canvas.draw_idle()
    n = arr.shape[0]
    main_window.pop_timecourse_summary.setText(
        f"n={n}  mean_t2p={peak_time:.1f}  mean_fwhm={mean_fwhm:.1f}"
    )


def draw_population_rfs_plot(
    main_window, selected_cell_id=None, subset_cell_ids=None, canvas=None
):
    """
    Draws the population receptive field plot.
    """
    if canvas is None:
        if (
            hasattr(main_window, "population_view_enabled")
            and main_window.population_view_enabled
        ):
            canvas = getattr(main_window, "pop_mosaic_canvas", main_window.rf_canvas)
        else:
            canvas = main_window.rf_canvas

    if subset_cell_ids is None:
        try:
            subset_cell_ids = main_window._get_pop_subset_ids()
        except Exception:
            pass

    colors = resolve_theme_colors(main_window.get_current_colors())
    dm = main_window.data_manager
    vision_params = dm.vision_params
    bridge = getattr(dm, "reference_bridge", None)
    has_borrowed_rfs = bool(bridge and bridge.get_all_rf_ellipses())

    logger.debug(
        f"draw_population_rfs_plot: vision_params={vision_params is not None}, "
        f"borrowed_rfs={has_borrowed_rfs}, selected_cell={selected_cell_id}, "
        f"subset={len(subset_cell_ids) if subset_cell_ids else None}"
    )

    if not vision_params and not has_borrowed_rfs:
        canvas.fig.clear()
        canvas.fig.set_facecolor(plot_field(colors))
        canvas.fig.text(
            0.5,
            0.5,
            "No Vision parameters available",
            ha="center",
            va="center",
            color=colors["text_secondary"],
        )
        canvas.draw_idle()
        return

    current_subset_tuple = (
        tuple(sorted(subset_cell_ids)) if subset_cell_ids is not None else "ALL"
    )
    current_subset_hash = hash(current_subset_tuple)

    # Check if theme changed
    theme_changed = False
    if hasattr(canvas, "_pop_plot_state"):
        stored_colors = canvas._pop_plot_state.get("colors")
        if stored_colors != colors:
            theme_changed = True

    can_hot_swap = (
        not theme_changed
        and hasattr(canvas, "_pop_plot_state")
        and canvas._pop_plot_state["subset_hash"] == current_subset_hash
        and canvas._pop_plot_state["ax"] in canvas.fig.axes
    )

    if can_hot_swap:
        ax = canvas._pop_plot_state["ax"]
        highlight_patch = canvas._pop_plot_state["highlight_artist"]

        # --- FIXED CALL 1 (3 arguments) ---
        _update_highlight_patch(
            highlight_patch, main_window.data_manager, selected_cell_id
        )

        # DS/OS markers are NEVER part of the hot-swap/cache fast paths —
        # see _draw_dsos_markers' docstring for why. Clear any markers this
        # ax already has (from a previous call) and redraw fresh every
        # time, so grating data that landed since the last redraw (e.g.
        # the startup batch-compute finishing) is always reflected, not
        # just when something else happens to also change the cache key.
        _clear_dsos_artists(ax)
        _draw_dsos_markers(
            ax,
            vision_params,
            main_window,
            sta_height=main_window.data_manager.vision_sta_height,
            subset_cell_ids=subset_cell_ids,
            colors=colors,
        )

        canvas.draw_idle()
    else:
        canvas.fig.clear()
        canvas.fig.set_facecolor(plot_field(colors))
        ax = canvas.fig.add_subplot(111)
        ax.set_facecolor(plot_field(colors))
        show_ids = _show_population_ids(main_window)
        cache_entry = _rf_background_cache.get(current_subset_hash)

        if _rf_cache_entry_matches(cache_entry, colors, show_ids):
            _draw_cached_rf_background(ax, cache_entry, colors)
            _store_rf_cache_entry(current_subset_hash, cache_entry)
        else:
            plot_population_rfs_background(
                ax,
                vision_params,
                main_window=main_window,
                sta_height=main_window.data_manager.vision_sta_height,
                subset_cell_ids=subset_cell_ids,
                colors=colors,
            )
            _store_rf_cache_entry(
                current_subset_hash, _snapshot_rf_background(ax, colors, show_ids)
            )

        # Always fresh — see _draw_dsos_markers docstring.
        _draw_dsos_markers(
            ax,
            vision_params,
            main_window,
            sta_height=main_window.data_manager.vision_sta_height,
            subset_cell_ids=subset_cell_ids,
            colors=colors,
        )

        highlight_hex = colors.get("plot_peak", colors["plot_highlight"])
        highlight_rgb = QColor(highlight_hex).getRgbF()[:3]
        highlight_fill = 0.55 if is_light_theme(colors) else 0.48
        highlight_patch = Ellipse(
            xy=(0, 0),
            width=1,
            height=1,
            angle=0,
            edgecolor=colors["plot_line"],
            facecolor=(*highlight_rgb, highlight_fill),
            lw=plot_stroke(colors, "thick"),
            zorder=10,
            visible=False,
        )
        ax.add_patch(highlight_patch)

        # --- FIXED CALL 2 (3 arguments) ---
        _update_highlight_patch(
            highlight_patch, main_window.data_manager, selected_cell_id
        )

        canvas._pop_plot_state = {
            "subset_hash": current_subset_hash,
            "highlight_artist": highlight_patch,
            "ax": ax,
            "colors": colors,
        }
        canvas.draw_idle()


def _update_highlight_patch(patch, data_manager, cell_id):
    if cell_id is None:
        patch.set_visible(False)
        return

    vision_id = data_manager.get_vision_id_for_cluster(cell_id)
    vision_params = data_manager.vision_params
    sta_height = data_manager.vision_sta_height

    try:
        stafit = None
        if vision_params is not None:
            try:
                stafit = vision_params.get_stafit_for_cell(vision_id)
            except Exception:
                stafit = None

        if stafit is not None:
            fit_ok = all(
                np.isfinite(v)
                for v in (
                    stafit.center_x,
                    stafit.center_y,
                    stafit.std_x,
                    stafit.std_y,
                    stafit.rot,
                )
            ) and stafit.std_x > 0 and stafit.std_y > 0
            if fit_ok:
                adjusted_y = (
                    sta_height - stafit.center_y
                    if sta_height is not None
                    else stafit.center_y
                )
                patch.center = (stafit.center_x, adjusted_y)
                patch.width = 2 * stafit.std_x
                patch.height = 2 * stafit.std_y
                patch.angle = np.rad2deg(stafit.rot)
                patch.set_visible(True)
                return

        # Fill-gap: borrowed RF from reference bridge
        bridge = getattr(data_manager, "reference_bridge", None)
        if bridge is not None:
            params = bridge.get_rf_ellipse_params(vision_id)
            if params is not None:
                adjusted_y = (
                    sta_height - params["y0"]
                    if sta_height is not None
                    else params["y0"]
                )
                patch.center = (params["x0"], adjusted_y)
                patch.width = 2 * params["std_x"]
                patch.height = 2 * params["std_y"]
                patch.angle = np.rad2deg(params["angle"])
                patch.set_visible(True)
                return

        patch.set_visible(False)
    except Exception as e:
        logger.debug(f"Failed to get stafit for cell {vision_id}: {e}")
        patch.set_visible(False)


def _build_ellipse_collection(xyw_angle_list, edgecolor, alpha, lw, zorder):
    """
    Build a single EllipseCollection from a list of (x, y, w, h, angle_deg) tuples.
    One draw call replaces N individual add_patch() calls.
    EllipseCollection expects widths/heights as full diameters, angles in degrees.
    """
    if not xyw_angle_list:
        return None
    arr = np.array(xyw_angle_list, dtype=float)  # (N, 5): x y w h angle
    ec = EllipseCollection(
        widths=arr[:, 2],
        heights=arr[:, 3],
        angles=arr[:, 4],
        units="x",
        offsets=arr[:, :2],
        offset_transform=None,  # set below after add_collection
        edgecolors=edgecolor,
        facecolors="none",
        linewidths=lw,
        alpha=alpha,
        zorder=zorder,
    )
    return ec


def _tight_limits(ellipses, frac_margin=0.05):
    """
    Return (xmin, xmax, ymin, ymax) that tightly encloses a list of
    (cx, cy, w, h, angle_deg) ellipse tuples, using the semi-axes as radii.
    A fractional margin (fraction of the span) is added on each side so that
    the outermost ellipses are not clipped.  Returns None when the list is empty.
    """
    if not ellipses:
        return None
    arr = np.array(ellipses)  # (N, 5): cx cy w h angle
    # Defense-in-depth: even though callers should pre-filter degenerate fits,
    # never let a NaN/Inf row reach matplotlib's set_xlim/set_ylim (it raises
    # ValueError and kills the whole panel redraw). Drop bad rows here too.
    finite_mask = np.all(np.isfinite(arr), axis=1)
    if not np.all(finite_mask):
        arr = arr[finite_mask]
    if arr.shape[0] == 0:
        return None
    cx, cy = arr[:, 0], arr[:, 1]
    rx, ry = arr[:, 2] / 2.0, arr[:, 3] / 2.0  # semi-axes (w/h are full diameters)
    x_lo, x_hi = np.min(cx - rx), np.max(cx + rx)
    y_lo, y_hi = np.min(cy - ry), np.max(cy + ry)
    mx = max((x_hi - x_lo) * frac_margin, 5.0)  # never less than 5 µm
    my = max((y_hi - y_lo) * frac_margin, 5.0)
    return x_lo - mx, x_hi + mx, y_lo - my, y_hi + my


def plot_population_rfs_background(
    ax, vision_params, main_window, sta_height, subset_cell_ids, colors
):
    """Draw native RF ellipses plus dashed borrowed RFs from ReferenceBridge."""
    colors = resolve_theme_colors(colors)
    ax.clear()
    show_labels = main_window.pop_show_ids_checkbox.isChecked()
    dm = main_window.data_manager
    is_vision_only = bool(getattr(dm, "is_vision_only", False))

    # reference_bridge may be a MagicMock on test doubles — only use a real bridge
    bridge = getattr(dm, "reference_bridge", None)
    if bridge is not None and not callable(
        getattr(bridge, "get_all_rf_ellipses", None)
    ):
        bridge = None

    def _vision_id(cid):
        """UI cluster_id → Vision id; tolerate incomplete test mocks."""
        fn = getattr(dm, "get_vision_id_for_cluster", None)
        if callable(fn):
            try:
                vid = fn(cid)
                if isinstance(vid, (int, np.integer)):
                    return int(vid)
            except Exception:
                pass
        return int(cid) if is_vision_only else int(cid) + 1

    native_ids = set()
    if vision_params is not None:
        try:
            native_ids = set(vision_params.get_cell_ids())
        except Exception:
            native_ids = set()

    borrowed_map = {}
    if bridge is not None:
        try:
            raw = bridge.get_all_rf_ellipses()
            if isinstance(raw, dict):
                borrowed_map = raw
        except Exception:
            borrowed_map = {}

    # subset_cell_ids are UI/Kilosort ids → Vision ids
    if subset_cell_ids is not None and len(subset_cell_ids) > 0:
        subset_vision_ids = {_vision_id(cid) for cid in subset_cell_ids}
        universe = native_ids | set(borrowed_map.keys())
        has_subset = len(universe) > 0 and len(subset_vision_ids) < len(universe)
    else:
        subset_vision_ids = native_ids | set(borrowed_map.keys())
        has_subset = False

    bg_ellipses = []
    target_ellipses = []
    borrowed_bg = []
    borrowed_target = []
    native_drawn = set()
    # (vision_id, cx, cy, width, height, angle_deg) for every ellipse actually
    # drawn — the EllipseCollections themselves keep no identity, so without
    # this a click on an outline could not be traced back to a cell.
    hit_entries = []

    def _label(cx, cy, vision_id):
        if not show_labels:
            return
        display_id = vision_id if is_vision_only else vision_id - 1
        ax.text(
            cx,
            cy,
            str(display_id),
            color=colors.get("text_secondary", "#9B9DA6"),
            fontsize=8,
            ha="center",
            va="center",
            alpha=0.8,
        )

    if vision_params is not None:
        for cell_id in native_ids:
            try:
                stafit = vision_params.get_stafit_for_cell(cell_id)
            except Exception:
                continue
            fit_vals = (
                stafit.center_x,
                stafit.center_y,
                stafit.std_x,
                stafit.std_y,
                stafit.rot,
            )
            if not all(np.isfinite(v) for v in fit_vals):
                continue
            if stafit.std_x <= 0 or stafit.std_y <= 0:
                continue
            adjusted_y = (
                sta_height - stafit.center_y
                if sta_height is not None
                else stafit.center_y
            )
            entry = (
                stafit.center_x,
                adjusted_y,
                stafit.std_x * 2,
                stafit.std_y * 2,
                np.degrees(stafit.rot),
            )
            native_drawn.add(cell_id)
            hit_entries.append((cell_id,) + entry)
            if cell_id in subset_vision_ids:
                target_ellipses.append(entry)
                _label(stafit.center_x, adjusted_y, cell_id)
            else:
                bg_ellipses.append(entry)

    # Borrowed only when native RF is missing for that Vision id
    for vision_id, params in borrowed_map.items():
        if vision_id in native_drawn:
            continue
        cx = params["x0"]
        cy = params["y0"]
        sx = params["std_x"]
        sy = params["std_y"]
        rot = params["angle"]
        if not all(np.isfinite(v) for v in (cx, cy, sx, sy, rot)):
            continue
        if sx <= 0 or sy <= 0:
            continue
        adjusted_y = sta_height - cy if sta_height is not None else cy
        entry = (cx, adjusted_y, sx * 2, sy * 2, np.degrees(rot))
        hit_entries.append((vision_id,) + entry)
        if vision_id in subset_vision_ids:
            borrowed_target.append(entry)
            _label(cx, adjusted_y, vision_id)
        else:
            borrowed_bg.append(entry)

    is_light = is_light_theme(colors)
    bg_edgecolor = colors.get("plot_shadow", "#3d3d3d")
    bg_alpha = plot_rf_bg_alpha(colors)
    bg_lw = 1.35 if is_light else 0.9
    target_color = colors.get("plot_scatter", colors.get("plot_highlight", "#0d47a1"))
    target_alpha = plot_rf_target_alpha(colors)
    target_lw = 1.8 if is_light else 1.25

    bg_coll = _build_ellipse_collection(
        bg_ellipses, edgecolor=bg_edgecolor, alpha=bg_alpha, lw=bg_lw, zorder=1
    )
    if bg_coll is not None:
        ax.add_collection(bg_coll)
        bg_coll.set_offset_transform(ax.transData)

    target_coll = _build_ellipse_collection(
        target_ellipses,
        edgecolor=target_color,
        alpha=target_alpha,
        lw=target_lw,
        zorder=2,
    )
    if target_coll is not None:
        ax.add_collection(target_coll)
        target_coll.set_offset_transform(ax.transData)

    # Spec D1: dashed + slightly lower alpha for borrowed ellipses
    borrowed_bg_coll = _build_ellipse_collection(
        borrowed_bg,
        edgecolor=bg_edgecolor,
        alpha=max(0.25, bg_alpha * 0.85),
        lw=bg_lw,
        zorder=1,
    )
    if borrowed_bg_coll is not None:
        borrowed_bg_coll.set_linestyle("--")
        ax.add_collection(borrowed_bg_coll)
        borrowed_bg_coll.set_offset_transform(ax.transData)

    borrowed_target_coll = _build_ellipse_collection(
        borrowed_target,
        edgecolor=target_color,
        alpha=max(0.70, target_alpha * 0.85),
        lw=target_lw,
        zorder=2,
    )
    if borrowed_target_coll is not None:
        borrowed_target_coll.set_linestyle("--")
        ax.add_collection(borrowed_target_coll)
        borrowed_target_coll.set_offset_transform(ax.transData)

    zoom_ellipses = (
        target_ellipses + borrowed_target
        if (has_subset and (target_ellipses or borrowed_target))
        else (target_ellipses + bg_ellipses + borrowed_target + borrowed_bg)
    )
    limits = _tight_limits(zoom_ellipses, frac_margin=0.05)
    if limits is not None:
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])

    set_rf_hit_entries(ax, hit_entries)

    n_target = len(target_ellipses) + len(borrowed_target)
    n_borrowed = len(borrowed_target) + len(borrowed_bg)
    title = f"Population Receptive Fields (n={n_target}"
    if n_borrowed:
        title += f", {n_borrowed} borrowed"
    title += ")"
    _apply_rf_axes_style(ax, colors, title=title)



def _clear_dsos_artists(ax):
    """
    Removes DS/OS LineCollections, annotate-arrowhead patches, and the
    DS/OS legend from ax, without touching the RF EllipseCollections/text
    labels underneath. Needed because _draw_dsos_markers is called fresh
    on every redraw (including hot-swap redraws that reuse the same ax
    object across multiple calls) — without clearing first, each redraw
    would stack new markers on top of the previous ones rather than
    replacing them.
    """
    for coll in list(ax.collections):
        if isinstance(coll, LineCollection):
            coll.remove()
    for patch in list(ax.patches):
        # FancyArrowPatch (from ax.annotate) vs. the RF highlight Ellipse —
        # distinguish by type rather than remove all patches, since the
        # selected-cell highlight Ellipse must survive this clear.
        if isinstance(patch, FancyArrowPatch):
            patch.remove()
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def _draw_dsos_markers(
    ax, vision_params, main_window, sta_height, subset_cell_ids, colors
):
    """
    Draws DS/OS arrow/tick markers on the Population RF plot. Deliberately
    NOT part of plot_population_rfs_background / the RF-background cache:
    that cache is keyed on (subset, colors, show_ids) and can legitimately
    be replayed unchanged for many redraws in a row (e.g. selecting a
    different single cell just moves the highlight patch, which is its own
    fast-path — see can_hot_swap in draw_population_rfs_plot). Grating data
    can land in the background (the startup batch-compute, or a cluster
    being visited individually) independent of any of those things
    changing, so a cached/hot-swapped background would show stale or
    entirely absent markers if this were baked into that cache. Calling
    this separately, every time, unconditionally, guarantees the markers
    always reflect whatever's currently in grating_computed_cache — the
    per-cell classification lookup is cheap (dict reads), so recomputing it
    on every redraw is not a real cost.

    Must be called AFTER plot_population_rfs_background (or a cached/
    hot-swapped replay of it) has already populated ax with the RF
    ellipses, since this draws directly on top of the existing ax rather
    than clearing it.
    """
    dm = main_window.data_manager
    is_vision_only = getattr(dm, "is_vision_only", False)

    # Only draw markers for cells whose RF ellipse is actually plotted —
    # i.e. the current subset — NOT every cell in the array. Previously
    # this iterated all_cell_ids regardless of subset, so when a small
    # subset was selected (e.g. one cluster, n=1), DS/OS arrows still
    # appeared for every classified cell across the whole array, floating
    # with no RF ellipse beneath them. This mirrors plot_population_rfs_
    # background's own subset translation (Kilosort IDs -> Vision IDs).
    all_cell_ids = set(vision_params.get_cell_ids())
    if subset_cell_ids is not None and len(subset_cell_ids) > 0:
        subset_vision_ids = {
            dm.get_vision_id_for_cluster(cid) for cid in subset_cell_ids
        }
        cell_ids_to_draw = all_cell_ids & subset_vision_ids
    else:
        # No subset -> whole population (matches background behavior).
        cell_ids_to_draw = all_cell_ids

    ds_lines = []  # each: [(x0,y0), (x1,y1)] arrow shaft; heads drawn separately
    os_lines = []  # each: [(x0,y0), (x1,y1)] double-ended tick

    for cell_id in cell_ids_to_draw:
        try:
            stafit = vision_params.get_stafit_for_cell(cell_id)
        except KeyError:
            continue

        fit_vals = (
            stafit.center_x,
            stafit.center_y,
            stafit.std_x,
            stafit.std_y,
            stafit.rot,
        )
        if not all(np.isfinite(v) for v in fit_vals):
            continue
        if stafit.std_x <= 0 or stafit.std_y <= 0:
            continue

        adjusted_y = (
            sta_height - stafit.center_y if sta_height is not None else stafit.center_y
        )

        # Sized off this cell's own RF so it scales sensibly across cells
        # with very different RF sizes, rather than one fixed pixel length.
        marker_len = max(stafit.std_x, stafit.std_y) * 1.5
        cluster_id = cell_id if is_vision_only else cell_id - 1
        grating_entry = dm.get_grating_data_for_cluster(cluster_id)
        # dsos_threshold: user-adjustable via the population panel's DS/OS
        # threshold slider (MainWindow.dsos_threshold). getattr default
        # matches grating_calc's own DSI_THRESHOLD/OSI_THRESHOLD default —
        # this only changes what counts as "strongly enough tuned to call
        # DS/OS," not the underlying significance/amplitude gate.
        threshold = getattr(main_window, "dsos_threshold", None)
        stats = (
            _best_dsos_condition(
                grating_entry, dsi_threshold=threshold, osi_threshold=threshold
            )
            if grating_entry
            else None
        )
        if stats is None:
            continue
        dsi, osi, pref_dir, pref_ori, classification = stats
        if classification == "DS" and not np.isnan(pref_dir):
            theta = np.deg2rad(pref_dir)
            dx, dy = np.cos(theta) * marker_len, np.sin(theta) * marker_len
            ds_lines.append(
                [(stafit.center_x, adjusted_y), (stafit.center_x + dx, adjusted_y + dy)]
            )
        elif classification == "OS" and not np.isnan(pref_ori):
            theta = np.deg2rad(pref_ori)
            dx, dy = np.cos(theta) * marker_len * 0.6, np.sin(theta) * marker_len * 0.6
            os_lines.append(
                [
                    (stafit.center_x - dx, adjusted_y - dy),
                    (stafit.center_x + dx, adjusted_y + dy),
                ]
            )

    # DS: short arrow along preferred_direction_deg. OS: short double-ended
    # tick along preferred_orientation_deg (axis, not a single direction —
    # matches the DS/OS probe map's existing visual language, before it was
    # removed in favor of this RF-plot overlay).
    ds_color = colors.get("plot_compare", "#E03131")
    os_color = colors.get("plot_overlay", "#1971C2")

    if os_lines:
        os_coll = LineCollection(
            os_lines, colors=os_color, linewidths=1.6, alpha=0.9, zorder=5
        )
        ax.add_collection(os_coll)

    for (x0, y0), (x1, y1) in ds_lines:
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                facecolor=ds_color,
                edgecolor=ds_color,
                width=1.2,
                headwidth=4.5,
                headlength=4.5,
                alpha=0.9,
            ),
            zorder=6,
        )

    # DS/OS legend intentionally omitted — arrows (DS) and double-ended
    # ticks (OS) are drawn directly on the RF plot above; the legend box
    # was redundant and duplicated on redraws.


def plot_rich_ei(
    fig,
    median_ei,
    channel_positions,
    features,
    _sampling_rate,
    _pre_samples=20,
    colors=None,
):
    """
    Plots the electrical image (EI) on the electrode array.
    """
    if colors is None:
        colors = DARK_COLORS
    colors = resolve_theme_colors(colors)

    fig.clear()
    fig.set_facecolor(plot_field(colors))
    ax = fig.add_subplot(111)
    ax.set_facecolor(plot_field(colors))

    if median_ei is not None and channel_positions is not None:
        max_amplitudes = np.max(np.abs(median_ei), axis=1)

        if len(max_amplitudes) == len(channel_positions):
            sc = ax.scatter(
                channel_positions[:, 0],
                channel_positions[:, 1],
                c=max_amplitudes,
                cmap="viridis",
                s=50,
                alpha=0.8,
            )
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("Max Amplitude (µV)", color=colors["text_secondary"])
            cbar.ax.yaxis.set_tick_params(color=colors["text_secondary"])
            plt.setp(
                plt.getp(cbar.ax.axes, "yticklabels"), color=colors["text_secondary"]
            )

            if features:
                com_x = features.get("center_of_mass_x")
                com_y = features.get("center_of_mass_y")
                spread = features.get("spatial_spread")

                if (
                    com_x is not None
                    and not np.isnan(com_x)
                    and com_y is not None
                    and not np.isnan(com_y)
                ):
                    ax.plot(
                        com_x,
                        com_y,
                        "rx",
                        markersize=10,
                        markeredgewidth=2,
                        label="COM",
                    )
                    if spread is not None and spread > 0:
                        from matplotlib.patches import Circle

                        circle = Circle(
                            (com_x, com_y),
                            spread,
                            color="red",
                            fill=False,
                            linestyle="--",
                            linewidth=1,
                            alpha=0.6,
                        )
                        ax.add_patch(circle)
                    ax.legend(
                        loc="upper right",
                        facecolor=plot_field(colors),
                        labelcolor=colors["text_primary"],
                    )
        else:
            ax.text(
                0.5,
                0.5,
                f"Dimension Mismatch: EI={len(max_amplitudes)}, Pos={len(channel_positions)}",
                ha="center",
                va="center",
                color="red",
            )
    else:
        ax.text(
            0.5,
            0.5,
            "No Data",
            ha="center",
            va="center",
            color=colors["text_secondary"],
        )

    ax.set_title("Electrical Image", color=colors["text_primary"])
    ax.set_xlabel("X (µm)", color=colors["text_secondary"])
    ax.set_ylabel("Y (µm)", color=colors["text_secondary"])
    ax.tick_params(colors=colors["text_secondary"])
    for spine in ax.spines.values():
        spine.set_edgecolor(colors["border_subtle"])
    ax.set_aspect("equal")
    fig.tight_layout()


def draw_population_acg_panel(main_window, subset_ids=None):
    if subset_ids is None:
        try:
            subset_ids = main_window._get_pop_subset_ids()
        except Exception:
            subset_ids = []

    canvas = getattr(main_window, "pop_acg_canvas", None)
    if canvas is None:
        return

    colors = resolve_theme_colors(main_window.get_current_colors())

    if not subset_ids:
        canvas.fig.clear()
        canvas.fig.set_facecolor(plot_field(colors))
        canvas.fig.text(
            0.5,
            0.5,
            "No cells selected",
            ha="center",
            color=colors["text_secondary"],
            fontsize=10,
        )
        canvas.draw_idle()
        main_window.pop_acg_summary.setText("n=0")
        if hasattr(canvas, "_acg_state"):
            del canvas._acg_state
        return

    import numpy as np

    cache_key = frozenset(subset_ids)
    cached = _group_acg_cache.get(cache_key)
    if cached is not None:
        arr = cached["arr"]
        mean_acg = cached["mean_acg"]
        t_axis = cached["t_axis"]
    else:
        traces = []
        t_axis = None

        for cid in subset_ids:
            try:
                time_lags, acg_norm = main_window.data_manager.get_acg_data(cid)
                if (
                    time_lags is not None
                    and acg_norm is not None
                    and len(time_lags) > 1
                ):
                    if t_axis is None:
                        t_axis = time_lags
                    if len(acg_norm) == len(t_axis):
                        traces.append(acg_norm)
            except Exception:
                continue

        if not traces:
            canvas.fig.clear()
            canvas.fig.set_facecolor(plot_field(colors))
            canvas.fig.text(
                0.5,
                0.5,
                "No valid ACG data",
                ha="center",
                color=colors["text_secondary"],
                fontsize=10,
            )
            canvas.draw_idle()
            return

        arr = np.vstack(traces)
        mean_acg = np.nanmean(arr, axis=0)
        _group_acg_cache[cache_key] = {
            "arr": arr,
            "mean_acg": mean_acg,
            "t_axis": t_axis,
        }

    t_axis = np.asarray(t_axis)
    arr = np.asarray(arr)

    # Causal lags only — matches the single-cell ACG view in
    # standard_plots_panel.py (mask = time_lags >= 0). The ACG is symmetric,
    # so the negative half adds no information here; dropping it doubles the
    # effective resolution on the refractory/decay structure that matters.
    causal_mask = (t_axis >= 0) & (t_axis <= 50)
    t_axis = t_axis[causal_mask]
    arr = arr[:, causal_mask]
    mean_acg = np.nanmean(arr, axis=0)

    segments = [np.column_stack([t_axis, row]) for row in arr]

    # Scaling, take 3: a 95th-percentile ceiling still let the top 5% of cells
    # run off the visible axis — not acceptable here, every trace needs to
    # stay fully on-screen. Use the true max of per-cell peaks instead. This
    # keeps the take-2 fix (deriving the ceiling from each cell's own peak,
    # not bins pooled across the long flat tail) but drops the percentile
    # cutoff so nothing gets clipped.
    per_cell_peak = np.nanmax(arr, axis=1) if arr.size else np.array([])
    finite_peaks = per_cell_peak[np.isfinite(per_cell_peak)]
    y_hi = float(np.nanmax(finite_peaks)) if finite_peaks.size else 1.0
    if y_hi <= 0:
        y_hi = 1.0
    y_range = y_hi
    y_bottom = -0.02 * y_range
    y_top = y_hi + 0.08 * y_range

    if hasattr(canvas, "_acg_state") and canvas._acg_state["ax"] in canvas.fig.axes:
        state = canvas._acg_state
        ax = state["ax"]

        current_facecolor = QColor(plot_field(colors)).name().lower()
        # ax.get_facecolor() returns RGBA tuple, need to unpack it
        facecolor_tuple = ax.get_facecolor()
        stored_facecolor = (
            QColor.fromRgbF(
                facecolor_tuple[0],
                facecolor_tuple[1],
                facecolor_tuple[2],
                facecolor_tuple[3],
            )
            .name()
            .lower()
        )
        if current_facecolor != stored_facecolor:
            if hasattr(canvas, "_acg_state"):
                del canvas._acg_state
            draw_population_acg_panel(main_window, subset_ids)
            return

        state["mean_line"].set_data(t_axis, mean_acg)
        state["shadow_lines"].set_segments(segments)
        ax.set_xlim(t_axis[0], t_axis[-1])
        ax.set_ylim(y_bottom, y_top)
    else:
        canvas.fig.clear()
        canvas.fig.set_facecolor(plot_field(colors))
        ax = canvas.fig.add_subplot(111)
        ax.set_facecolor(plot_field(colors))

        ax.axhline(
            0,
            color=colors["text_primary"],
            linestyle="--",
            linewidth=1.0,
            alpha=0.2,
            zorder=1,
        )
        ax.axvline(
            0,
            color=colors["text_primary"],
            linestyle="--",
            linewidth=1.0,
            alpha=0.3,
            zorder=1,
        )

        shadow_lines = LineCollection(
            segments,
            color=colors.get("plot_ensemble", colors["plot_acg"]),
            linewidth=plot_stroke(colors, "thin"),
            alpha=plot_ensemble_alpha(colors),
            zorder=2,
        )
        ax.add_collection(shadow_lines)
        mean_line = _first_plot_artist(
            ax.plot(
                t_axis,
                mean_acg,
                color=colors["plot_mean"],
                linewidth=plot_stroke(colors),
                alpha=0.95,
                zorder=4,
            )
        )

        ax.set_xlabel("Time lag (ms)", color=colors["text_secondary"], fontsize=9)
        ax.set_ylabel("Autocorrelation", color=colors["text_secondary"], fontsize=9)
        ax.set_xlim(t_axis[0], t_axis[-1])
        ax.set_ylim(y_bottom, y_top)
        ax.tick_params(colors=colors["text_secondary"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(colors["border_subtle"])

        canvas._acg_state = {
            "ax": ax,
            "mean_line": mean_line,
            "shadow_lines": shadow_lines,
        }

    canvas.draw_idle()
    main_window.pop_acg_summary.setText(f"n={arr.shape[0]}")


def draw_population_fr_panel(main_window, subset_ids=None):
    """
    Population firing rate over the course of the recording for every cell in
    the subset, overlaid. Reads the per-cluster 'fr_bin_centers'/'fr_rate'
    series already computed by DataManager._compute_standard_plots() — no new
    computation, just reusing the standard_plot_cache.

    Primarily a stability/QC view: a cell whose rate trace drifts, drops out,
    or spikes relative to the rest of the subset is a candidate for re-review
    (electrode drift, lost unit, recording artifact), and it's visible here in
    a way that a single static firing-rate number isn't.
    """
    if subset_ids is None:
        try:
            subset_ids = main_window._get_pop_subset_ids()
        except Exception:
            subset_ids = []

    canvas = getattr(main_window, "pop_fr_canvas", None)
    if canvas is None:
        return

    colors = resolve_theme_colors(main_window.get_current_colors())

    if not subset_ids:
        canvas.fig.clear()
        canvas.fig.set_facecolor(plot_field(colors))
        canvas.fig.text(
            0.5,
            0.5,
            "No cells selected",
            ha="center",
            color=colors["text_secondary"],
            fontsize=10,
        )
        canvas.draw_idle()
        if hasattr(main_window, "pop_fr_summary"):
            main_window.pop_fr_summary.setText("n=0")
        if hasattr(canvas, "_fr_state"):
            del canvas._fr_state
        return

    cache_key = frozenset(subset_ids)
    cached = _group_fr_cache.get(cache_key)
    if cached is not None:
        arr = cached["arr"]
        mean_fr = cached["mean_fr"]
        t_axis = cached["t_axis"]
    else:
        traces = []
        t_axis = None

        for cid in subset_ids:
            try:
                std_data = main_window.data_manager.get_standard_plot_data(cid)
                bin_centers = std_data.get("fr_bin_centers") if std_data else None
                rate = std_data.get("fr_rate") if std_data else None
                if (
                    bin_centers is not None
                    and rate is not None
                    and len(bin_centers) > 1
                ):
                    if t_axis is None:
                        t_axis = np.asarray(bin_centers)
                    if len(rate) == len(t_axis):
                        traces.append(np.asarray(rate))
            except Exception:
                continue

        if not traces:
            canvas.fig.clear()
            canvas.fig.set_facecolor(plot_field(colors))
            canvas.fig.text(
                0.5,
                0.5,
                "No valid firing-rate data",
                ha="center",
                color=colors["text_secondary"],
                fontsize=10,
            )
            canvas.draw_idle()
            return

        arr = np.vstack(traces)
        mean_fr = np.nanmean(arr, axis=0)
        _group_fr_cache[cache_key] = {
            "arr": arr,
            "mean_fr": mean_fr,
            "t_axis": t_axis,
        }

    segments = [np.column_stack([t_axis, row]) for row in arr]

    # No-clipping scaling — see draw_population_acg_panel for the full
    # rationale. Ceiling is the true max of per-cell peaks (not a percentile),
    # so no trace ever runs off the visible axis. Floor anchored near zero
    # since firing rate can't go negative.
    per_cell_peak = np.nanmax(arr, axis=1) if arr.size else np.array([])
    finite_peaks = per_cell_peak[np.isfinite(per_cell_peak)]
    y_hi = float(np.nanmax(finite_peaks)) if finite_peaks.size else 1.0
    if y_hi <= 0:
        y_hi = 1.0
    y_range = y_hi
    y_bottom = -0.02 * y_range
    y_top = y_hi + 0.08 * y_range

    if hasattr(canvas, "_fr_state") and canvas._fr_state["ax"] in canvas.fig.axes:
        state = canvas._fr_state
        ax = state["ax"]

        current_facecolor = QColor(plot_field(colors)).name().lower()
        facecolor_tuple = ax.get_facecolor()
        stored_facecolor = (
            QColor.fromRgbF(
                facecolor_tuple[0],
                facecolor_tuple[1],
                facecolor_tuple[2],
                facecolor_tuple[3],
            )
            .name()
            .lower()
        )
        if current_facecolor != stored_facecolor:
            if hasattr(canvas, "_fr_state"):
                del canvas._fr_state
            draw_population_fr_panel(main_window, subset_ids)
            return

        state["mean_line"].set_data(t_axis, mean_fr)
        state["shadow_lines"].set_segments(segments)
        ax.set_xlim(t_axis[0], t_axis[-1])
        ax.set_ylim(y_bottom, y_top)
    else:
        canvas.fig.clear()
        canvas.fig.set_facecolor(plot_field(colors))
        ax = canvas.fig.add_subplot(111)
        ax.set_facecolor(plot_field(colors))

        shadow_lines = LineCollection(
            segments,
            color=colors.get("plot_ensemble", colors["plot_fr"]),
            linewidth=plot_stroke(colors, "thin"),
            alpha=plot_ensemble_alpha(colors),
            zorder=2,
        )
        ax.add_collection(shadow_lines)
        mean_line = _first_plot_artist(
            ax.plot(
                t_axis,
                mean_fr,
                color=colors["plot_mean"],
                linewidth=plot_stroke(colors),
                alpha=0.95,
                zorder=4,
            )
        )

        ax.set_xlabel("Time (s)", color=colors["text_secondary"], fontsize=9)
        ax.set_ylabel("Firing Rate (Hz)", color=colors["text_secondary"], fontsize=9)
        ax.set_xlim(t_axis[0], t_axis[-1])
        ax.set_ylim(y_bottom, y_top)
        ax.tick_params(colors=colors["text_secondary"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(colors["border_subtle"])

        canvas._fr_state = {
            "ax": ax,
            "mean_line": mean_line,
            "shadow_lines": shadow_lines,
        }

    canvas.draw_idle()
    if hasattr(main_window, "pop_fr_summary"):
        main_window.pop_fr_summary.setText(f"n={arr.shape[0]}")


# ---------------------------------------------------------------------------
# DS/OS classification helper
# ---------------------------------------------------------------------------
# The standalone "DS/OS Probe Map" panel (array-space scatter with arrows)
# was removed — DS/OS is now shown directly on the Population Receptive
# Fields plot via _draw_dsos_markers, which is RF-space and doesn't require
# a separate panel to cross-reference. _best_dsos_condition (the shared
# classification lookup) stays, since _draw_dsos_markers still needs it.


def _best_dsos_condition(grating_entry, dsi_threshold=None, osi_threshold=None):
    """
    Delegates to grating_calc.select_best_dsos_condition — the single
    shared, gated selector used by both the DS/OS probe map and
    GratingPanel. See select_best_dsos_condition's docstring for why raw
    max(|DSI|) (the old approach here) was wrong: amplitude-blind,
    significance-blind selection let a near-silent, noisy condition
    outrank a real, strong response.

    dsi_threshold/osi_threshold: optional override for how strong DSI/OSI
    must be (after already passing the significance/amplitude gate — this
    does NOT loosen that gate) to count as DS/OS. None uses grating_calc's
    module-level default (DSI_THRESHOLD/OSI_THRESHOLD = 0.3). Driven by
    the population panel's DS/OS threshold slider — see
    MainWindow.dsos_threshold.

    Returns (dsi, osi, pref_dir_deg, pref_ori_deg, classification) where
    classification is 'DS' | 'OS' | 'none', or None if this cluster has no
    dsos conditions at all. Callers should use `classification` directly
    rather than re-deriving DS/OS from the returned DSI/OSI values against
    local thresholds.
    """
    kwargs = {}
    if dsi_threshold is not None:
        kwargs["dsi_threshold"] = dsi_threshold
    if osi_threshold is not None:
        kwargs["osi_threshold"] = osi_threshold
    selection = grating_calc.select_best_dsos_condition(grating_entry, **kwargs)
    if selection is None:
        return None
    return (
        selection["DSI"],
        selection["OSI"],
        selection["preferred_direction_deg"],
        selection["preferred_orientation_deg"],
        selection["classification"],
    )