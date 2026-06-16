"""
Population Panel for RGC Viewer

This module contains population plotting functions that were previously in the plotting module.
"""

import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import LineCollection, EllipseCollection
from PyQt5.QtGui import QColor

from ..theme import DARK_COLORS

# Set matplotlib logging level to WARNING to suppress font debug messages
matplotlib_logger = logging.getLogger('matplotlib.font_manager')
matplotlib_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_group_timecourse_cache = {}
_group_acg_cache = {}
_rf_background_cache = {}
_rf_background_cache_order = []
_RF_CACHE_MAX = 10


def invalidate_population_caches():
    """Clear population-panel caches after source data or membership changes."""
    _group_timecourse_cache.clear()
    _group_acg_cache.clear()
    _rf_background_cache.clear()
    _rf_background_cache_order.clear()


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
        hasattr(main_window, 'pop_show_ids_checkbox') and
        main_window.pop_show_ids_checkbox.isChecked()
    )


def _snapshot_rf_background(ax, colors, show_ids):
    """
    Store the geometry arrays from EllipseCollections on the axes, not individual
    patch objects. Replay via _draw_cached_rf_background is then two add_collection
    calls instead of N add_patch calls.
    """
    collections_data = []
    for coll in ax.collections:
        if not isinstance(coll, EllipseCollection):
            continue
        offsets = coll.get_offsets()          # (N, 2) array
        widths  = coll._widths                # stored internally
        heights = coll._heights
        angles  = coll._angles
        collections_data.append({
            'offsets': np.array(offsets),
            'widths':  np.array(widths),
            'heights': np.array(heights),
            'angles':  np.array(angles),
            'edgecolor': coll.get_edgecolor()[0],   # RGBA tuple
            'alpha':   coll.get_alpha(),
            'lw':      coll.get_linewidth()[0] if hasattr(coll.get_linewidth(), '__len__') else coll.get_linewidth(),
            'zorder':  coll.get_zorder(),
        })

    texts = []
    for text in getattr(ax, 'texts', []):
        texts.append({
            'position': text.get_position(),
            'text': text.get_text(),
            'color': text.get_color(),
            'fontsize': text.get_fontsize(),
            'ha': text.get_ha(),
            'va': text.get_va(),
            'alpha': text.get_alpha(),
        })

    title = ax.get_title()
    if not isinstance(title, str):
        title = None

    return {
        'colors': dict(colors),
        'show_ids': show_ids,
        'collections': collections_data,
        'texts': texts,
        'xlim': _safe_axis_limits(ax.get_xlim),
        'ylim': _safe_axis_limits(ax.get_ylim),
        'title': title,
    }


def _apply_rf_axes_style(ax, colors, title=None):
    if title:
        ax.set_title(title, color=colors['text_primary'])
    ax.set_facecolor(colors['bg_panel'])
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(colors=colors['text_secondary'])
    for spine in ax.spines.values():
        spine.set_edgecolor(colors['border_subtle'])
    ax.grid(False)


def _draw_cached_rf_background(ax, cache_entry, colors):
    """Replay cached EllipseCollections — 2 add_collection calls, not N add_patch."""
    for cd in cache_entry.get('collections', []):
        ec = EllipseCollection(
            widths=cd['widths'] * 2, heights=cd['heights'] * 2, angles=cd['angles'] * 180 / np.pi,
            units='x', offsets=cd['offsets'], offset_transform=ax.transData,
            edgecolors=cd['edgecolor'], facecolors='none',
            linewidths=cd['lw'], alpha=cd['alpha'], zorder=cd['zorder'],
        )
        ax.add_collection(ec)

    for text_data in cache_entry['texts']:
        ax.text(
            text_data['position'][0],
            text_data['position'][1],
            text_data['text'],
            color=text_data['color'],
            fontsize=text_data['fontsize'],
            ha=text_data['ha'],
            va=text_data['va'],
            alpha=text_data['alpha'],
        )

    if cache_entry['xlim'] is not None:
        ax.set_xlim(*cache_entry['xlim'])
    if cache_entry['ylim'] is not None:
        ax.set_ylim(*cache_entry['ylim'])
    _apply_rf_axes_style(ax, colors, cache_entry.get('title'))


def _rf_cache_entry_matches(cache_entry, colors, show_ids):
    return (
        isinstance(cache_entry, dict) and
        'collections' in cache_entry and
        cache_entry.get('colors') == colors and
        cache_entry.get('show_ids') == show_ids
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
    colors = main_window.get_current_colors()
    
    # Early exit: nothing selected
    if not subset_ids:
        canvas.fig.clear()
        canvas.fig.set_facecolor(colors['bg_panel'])
        canvas.fig.text(0.5, 0.5, "No cells selected", ha='center', color=colors['text_secondary'], fontsize=10)
        canvas.draw_idle()
        main_window.pop_timecourse_summary.setText("n=0  mean_t2p: N/A  mean_fwhm: N/A")
        if hasattr(canvas, '_timecourse_state'):
            del canvas._timecourse_state
        return

    # --- 1. Fast Data Extraction via Physics Cache ---
    cache_key = frozenset(subset_ids)
    cached = _group_timecourse_cache.get(cache_key)
    if cached is not None:
        arr = cached['arr']
        mean_tc = cached['mean_tc']
        t_axis = cached['t_axis']
        peak_idx = cached['peak_idx']
        mean_fwhm = cached['mean_fwhm']
    else:
        traces = []
        for cid in subset_ids:
            physics = main_window.data_manager.get_cell_physics(cid)
            tc = physics.get('timecourse')
            if tc is not None:
                traces.append(tc)

        if not traces:
            canvas.fig.clear()
            canvas.fig.set_facecolor(colors['bg_panel'])
            canvas.fig.text(0.5, 0.5, "No valid timecourses", ha='center', color=colors['text_secondary'], fontsize=10)
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
            'arr': arr,
            'mean_tc': mean_tc,
            't_axis': t_axis,
            'peak_idx': peak_idx,
            'mean_fwhm': mean_fwhm,
        }

    peak_time = t_axis[peak_idx]
    peak_val = mean_tc[peak_idx]

    segments = [np.column_stack([t_axis, row]) for row in arr]

    y_min, y_max = np.min(arr), np.max(arr)
    y_range = y_max - y_min
    if y_range == 0:
        y_range = 1.0
    
    y_bottom = y_min - (0.1 * y_range)
    y_top = y_max + (0.25 * y_range)

    # --- 2. Hot-Swap Rendering ---
    if hasattr(canvas, '_timecourse_state') and canvas._timecourse_state['ax'] in canvas.fig.axes:
        # Fast update
        state = canvas._timecourse_state
        ax = state['ax']

        # Check if theme changed (background color mismatch)
        current_facecolor = QColor(colors['bg_panel']).name().lower()
        # ax.get_facecolor() returns RGBA tuple, need to unpack it
        facecolor_tuple = ax.get_facecolor()
        stored_facecolor = QColor.fromRgbF(
            facecolor_tuple[0], facecolor_tuple[1], facecolor_tuple[2], facecolor_tuple[3]
        ).name().lower()

        if current_facecolor != stored_facecolor:
             # Force full rebuild on theme change
             if hasattr(canvas, '_timecourse_state'):
                 del canvas._timecourse_state
             draw_population_timecourse_panel(main_window, subset_ids)
             return

        state['mean_line'].set_data(t_axis, mean_tc)
        state['shadow_lines'].set_segments(segments)
        state['peak_marker'].set_data([peak_time], [peak_val])
        
        state['peak_text'].set_position((peak_time, peak_val + (np.max(mean_tc)*0.1)))
        state['peak_text'].set_text(f" Peak\n Frame {peak_time}")
        
        ax.set_xlim(t_axis[0], t_axis[-1])
        ax.set_ylim(y_bottom, y_top)
        
    else:
        from matplotlib.collections import LineCollection
        # Full Rebuild
        canvas.fig.clear()
        canvas.fig.set_facecolor(colors['bg_panel'])
        ax = canvas.fig.add_subplot(111)
        ax.set_facecolor(colors['bg_panel'])
        
        # 1. Zero Line
        ax.axhline(0, color=colors['text_primary'], linestyle='--', linewidth=1.0, alpha=0.2, zorder=1)

        # 2. Shadow Traces
        shadow_lines = LineCollection(segments, color=colors['accent'], linewidth=0.8, alpha=0.15, zorder=2)
        ax.add_collection(shadow_lines)

        # 3. Solid Mean Trace
        mean_line = _first_plot_artist(ax.plot(t_axis, mean_tc, color=colors['plot_mean'], linewidth=2.5, zorder=4))

        # 4. Highlight the Peak Feature
        peak_marker = _first_plot_artist(ax.plot([peak_time], [peak_val], 'o', color=colors['plot_peak'], markersize=6, zorder=5))
        peak_text = ax.text(peak_time, peak_val + (np.max(mean_tc)*0.1), 
                            f" Peak\n Frame {peak_time}", color=colors['plot_peak'], 
                            fontsize=8, ha='center', va='bottom')

        # Aesthetics
        ax.set_xlabel("Time (frames)", color=colors['text_secondary'], fontsize=9)
        ax.set_ylabel("Response (a.u.)", color=colors['text_secondary'], fontsize=9)
        
        ax.set_xlim(t_axis[0], t_axis[-1])
        ax.set_ylim(y_bottom, y_top)
        
        ax.tick_params(colors=colors['text_secondary'], labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(colors['border_subtle'])
            
        ax.grid(False)

        # Save to state
        canvas._timecourse_state = {
            'ax': ax, 
            'mean_line': mean_line, 
            'shadow_lines': shadow_lines,
            'peak_marker': peak_marker,
            'peak_text': peak_text
        }

    canvas.draw_idle()
    n = arr.shape[0]
    main_window.pop_timecourse_summary.setText(f"n={n}  mean_t2p={peak_time:.1f}  mean_fwhm={mean_fwhm:.1f}")
    
def draw_population_rfs_plot(
        main_window,
        selected_cell_id=None,
        subset_cell_ids=None,
        canvas=None):
    """
    Draws the population receptive field plot.
    """
    if canvas is None:
        if hasattr(main_window, 'population_view_enabled') and main_window.population_view_enabled:
            canvas = getattr(main_window, 'pop_mosaic_canvas', main_window.rf_canvas)
        else:
            canvas = main_window.rf_canvas

    if subset_cell_ids is None:
        try:
            subset_cell_ids = main_window._get_pop_subset_ids()
        except Exception:
            pass

    colors = main_window.get_current_colors()
    vision_params = main_window.data_manager.vision_params
    
    logger.debug(f"draw_population_rfs_plot: vision_params={vision_params is not None}, selected_cell={selected_cell_id}, subset={len(subset_cell_ids) if subset_cell_ids else None}")
    
    if not vision_params:
        canvas.fig.clear()
        canvas.fig.set_facecolor(colors['bg_panel'])
        canvas.fig.text(0.5, 0.5, "No Vision parameters available", ha='center', va='center', color=colors['text_secondary'])
        canvas.draw_idle()
        return

    current_subset_tuple = tuple(sorted(subset_cell_ids)) if subset_cell_ids is not None else "ALL"
    current_subset_hash = hash(current_subset_tuple)

    # Check if theme changed
    theme_changed = False
    if hasattr(canvas, '_pop_plot_state'):
        stored_colors = canvas._pop_plot_state.get('colors')
        if stored_colors != colors:
            theme_changed = True

    can_hot_swap = (
        not theme_changed and
        hasattr(canvas, '_pop_plot_state') and
        canvas._pop_plot_state['subset_hash'] == current_subset_hash and
        canvas._pop_plot_state['ax'] in canvas.fig.axes
    )

    if can_hot_swap:
        ax = canvas._pop_plot_state['ax']
        highlight_patch = canvas._pop_plot_state['highlight_artist']
        _update_highlight_patch(highlight_patch, vision_params, selected_cell_id, main_window.data_manager.vision_sta_height)
        canvas.draw_idle()
    else:
        canvas.fig.clear()
        canvas.fig.set_facecolor(colors['bg_panel'])
        ax = canvas.fig.add_subplot(111)
        ax.set_facecolor(colors['bg_panel'])
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
                colors=colors
            )
            _store_rf_cache_entry(
                current_subset_hash,
                _snapshot_rf_background(ax, colors, show_ids)
            )

        highlight_rgb = QColor(colors['plot_highlight']).getRgbF()[:3]
        highlight_patch = Ellipse(
            xy=(0, 0), width=1, height=1, angle=0,
            edgecolor=colors['plot_highlight'], facecolor=(*highlight_rgb, 0.42),
            lw=1.75, zorder=10, visible=False
        )
        ax.add_patch(highlight_patch)
        _update_highlight_patch(highlight_patch, vision_params, selected_cell_id, main_window.data_manager.vision_sta_height)

        canvas._pop_plot_state = {
            'subset_hash': current_subset_hash,
            'highlight_artist': highlight_patch,
            'ax': ax,
            'colors': colors
        }
        canvas.draw_idle()


def _update_highlight_patch(patch, vision_params, cell_id, sta_height):
    if cell_id is None:
        patch.set_visible(False)
        return

    vision_id = cell_id + 1
    try:
        stafit = vision_params.get_stafit_for_cell(vision_id)
        adjusted_y = sta_height - stafit.center_y if sta_height is not None else stafit.center_y

        patch.center = (stafit.center_x, adjusted_y)
        patch.width = 2 * stafit.std_x
        patch.height = 2 * stafit.std_y
        patch.angle = np.rad2deg(stafit.rot)
        patch.set_visible(True)
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
        units='x',
        offsets=arr[:, :2],
        offset_transform=None,  # set below after add_collection
        edgecolors=edgecolor,
        facecolors='none',
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
    arr = np.array(ellipses)          # (N, 5): cx cy w h angle
    cx, cy = arr[:, 0], arr[:, 1]
    rx, ry = arr[:, 2] / 2.0, arr[:, 3] / 2.0   # semi-axes (w/h are full diameters)
    x_lo, x_hi = np.min(cx - rx), np.max(cx + rx)
    y_lo, y_hi = np.min(cy - ry), np.max(cy + ry)
    mx = max((x_hi - x_lo) * frac_margin, 5.0)   # never less than 5 µm
    my = max((y_hi - y_lo) * frac_margin, 5.0)
    return x_lo - mx, x_hi + mx, y_lo - my, y_hi + my


def plot_population_rfs_background(ax, vision_params, main_window, sta_height, subset_cell_ids, colors):
    ax.clear()
    show_labels = main_window.pop_show_ids_checkbox.isChecked()
    is_vision_only = getattr(main_window.data_manager, 'is_vision_only', False)

    all_cell_ids = set(vision_params.get_cell_ids())

    # Determine whether a meaningful subset is active.
    # subset_cell_ids uses Kilosort (0-based) IDs; translate to Vision IDs for comparison.
    if subset_cell_ids is not None and len(subset_cell_ids) > 0:
        # Convert subset to Vision IDs for comparison against vision_params IDs
        if is_vision_only:
            subset_vision_ids = set(subset_cell_ids)
        else:
            subset_vision_ids = {cid + 1 for cid in subset_cell_ids}
        has_subset = len(subset_vision_ids) < len(all_cell_ids)
    else:
        subset_vision_ids = all_cell_ids
        has_subset = False

    bg_ellipses = []     # (cx, cy, w, h, angle_deg) for non-subset cells
    target_ellipses = [] # (cx, cy, w, h, angle_deg) for subset cells

    for cell_id in all_cell_ids:
        try:
            stafit = vision_params.get_stafit_for_cell(cell_id)
        except KeyError:
            continue

        adjusted_y = sta_height - stafit.center_y if sta_height is not None else stafit.center_y

        entry = (stafit.center_x, adjusted_y,
                 stafit.std_x * 2, stafit.std_y * 2,
                 np.degrees(stafit.rot))

        if cell_id in subset_vision_ids:
            target_ellipses.append(entry)
            if show_labels:
                ax.text(stafit.center_x, adjusted_y, str(cell_id),
                        color=colors.get('text_secondary', '#9B9DA6'),
                        fontsize=8, ha='center', va='center',
                        alpha=0.8)
        else:
            bg_ellipses.append(entry)

    # Build EllipseCollections (so _snapshot_rf_background can capture them).
    # In light mode 'border_subtle' is very light (#DEE2E6) and alpha=0.15
    # makes it completely invisible against the white panel background.
    # Detect light mode by checking whether bg_panel is white/near-white and
    # use a medium gray with higher opacity instead.
    is_light = colors.get('bg_panel', '').upper() in ('#FFFFFF', '#FAFAFA', '#F8F9FA')
    bg_edgecolor = colors.get('text_tertiary', '#ADB5BD') if is_light else colors.get('border_subtle', '#2E3038')
    bg_alpha = 0.35 if is_light else 0.15

    bg_coll = _build_ellipse_collection(
        bg_ellipses,
        edgecolor=bg_edgecolor,
        alpha=bg_alpha, lw=0.75, zorder=1)
    if bg_coll is not None:
        ax.add_collection(bg_coll)
        bg_coll.set_offset_transform(ax.transData)

    target_coll = _build_ellipse_collection(
        target_ellipses,
        edgecolor=colors.get('plot_highlight', '#00FFFF'),
        alpha=0.55, lw=1.0, zorder=2)
    if target_coll is not None:
        ax.add_collection(target_coll)
        target_coll.set_offset_transform(ax.transData)

    # --- Tight axis limits ---
    # When a non-trivial subset is active, zoom to the subset ellipses only.
    # When showing all cells (or subset == all), zoom to all ellipses.
    # In both cases use ellipse extent (center ± semi-axis) rather than just
    # the center coordinates so that outermost ellipses are never clipped.
    zoom_ellipses = target_ellipses if (has_subset and target_ellipses) else (target_ellipses + bg_ellipses)
    limits = _tight_limits(zoom_ellipses, frac_margin=0.05)
    if limits is not None:
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])

    n_target = len(target_ellipses)
    _apply_rf_axes_style(ax, colors,
                         title=f"Population Receptive Fields (n={n_target})")
def plot_rich_ei(fig, median_ei, channel_positions, features, _sampling_rate, _pre_samples=20, colors=None):
    """
    Plots the electrical image (EI) on the electrode array.
    """
    if colors is None:
        colors = DARK_COLORS

    fig.clear()
    fig.set_facecolor(colors['bg_panel'])
    ax = fig.add_subplot(111)
    ax.set_facecolor(colors['bg_panel'])

    if median_ei is not None and channel_positions is not None:
        max_amplitudes = np.max(np.abs(median_ei), axis=1)

        if len(max_amplitudes) == len(channel_positions):
            sc = ax.scatter(channel_positions[:, 0], channel_positions[:, 1], c=max_amplitudes, cmap='viridis', s=50, alpha=0.8)
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label('Max Amplitude (µV)', color=colors['text_secondary'])
            cbar.ax.yaxis.set_tick_params(color=colors['text_secondary'])
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=colors['text_secondary'])

            if features:
                com_x = features.get('center_of_mass_x')
                com_y = features.get('center_of_mass_y')
                spread = features.get('spatial_spread')

                if com_x is not None and not np.isnan(com_x) and com_y is not None and not np.isnan(com_y):
                    ax.plot(com_x, com_y, 'rx', markersize=10, markeredgewidth=2, label='COM')
                    if spread is not None and spread > 0:
                        from matplotlib.patches import Circle
                        circle = Circle((com_x, com_y), spread, color='red', fill=False, linestyle='--', linewidth=1, alpha=0.6)
                        ax.add_patch(circle)
                    ax.legend(loc='upper right', facecolor=colors['bg_panel'], labelcolor=colors['text_primary'])
        else:
            ax.text(0.5, 0.5, f"Dimension Mismatch: EI={len(max_amplitudes)}, Pos={len(channel_positions)}",
                    ha='center', va='center', color='red')
    else:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center', color=colors['text_secondary'])

    ax.set_title('Electrical Image', color=colors['text_primary'])
    ax.set_xlabel('X (µm)', color=colors['text_secondary'])
    ax.set_ylabel('Y (µm)', color=colors['text_secondary'])
    ax.tick_params(colors=colors['text_secondary'])
    for spine in ax.spines.values():
        spine.set_edgecolor(colors['border_subtle'])
    ax.set_aspect('equal')
    fig.tight_layout()

def draw_population_acg_panel(main_window, subset_ids=None):
    if subset_ids is None:
        try:
            subset_ids = main_window._get_pop_subset_ids()
        except Exception:
            subset_ids = []

    canvas = getattr(main_window, 'pop_acg_canvas', None)
    if canvas is None: return
    
    colors = main_window.get_current_colors()
    
    if not subset_ids:
        canvas.fig.clear()
        canvas.fig.set_facecolor(colors['bg_panel'])
        canvas.fig.text(0.5, 0.5, "No cells selected", ha='center', color=colors['text_secondary'], fontsize=10)
        canvas.draw_idle()
        main_window.pop_acg_summary.setText("n=0")
        if hasattr(canvas, '_acg_state'):
            del canvas._acg_state
        return

    import numpy as np

    cache_key = frozenset(subset_ids)
    cached = _group_acg_cache.get(cache_key)
    if cached is not None:
        arr = cached['arr']
        mean_acg = cached['mean_acg']
        t_axis = cached['t_axis']
    else:
        traces = []
        t_axis = None

        for cid in subset_ids:
            try:
                time_lags, acg_norm = main_window.data_manager.get_acg_data(cid)
                if time_lags is not None and acg_norm is not None and len(time_lags) > 1:
                    if t_axis is None: t_axis = time_lags
                    if len(acg_norm) == len(t_axis): traces.append(acg_norm)
            except Exception: continue

        if not traces:
            canvas.fig.clear()
            canvas.fig.set_facecolor(colors['bg_panel'])
            canvas.fig.text(0.5, 0.5, "No valid ACG data", ha='center', color=colors['text_secondary'], fontsize=10)
            canvas.draw_idle()
            return

        arr = np.vstack(traces)
        mean_acg = np.nanmean(arr, axis=0)
        _group_acg_cache[cache_key] = {
            'arr': arr,
            'mean_acg': mean_acg,
            't_axis': t_axis,
        }

    segments = [np.column_stack([t_axis, row]) for row in arr]

    y_min, y_max = np.min(arr), np.max(arr)
    y_range = y_max - y_min
    if y_range == 0: y_range = 1.0
    y_bottom = y_min - (0.05 * y_range)
    y_top = y_max + (0.05 * y_range)

    if hasattr(canvas, '_acg_state') and canvas._acg_state['ax'] in canvas.fig.axes:
        state = canvas._acg_state
        ax = state['ax']

        current_facecolor = QColor(colors['bg_panel']).name().lower()
        # ax.get_facecolor() returns RGBA tuple, need to unpack it
        facecolor_tuple = ax.get_facecolor()
        stored_facecolor = QColor.fromRgbF(
            facecolor_tuple[0], facecolor_tuple[1], facecolor_tuple[2], facecolor_tuple[3]
        ).name().lower()
        if current_facecolor != stored_facecolor:
             if hasattr(canvas, '_acg_state'): del canvas._acg_state
             draw_population_acg_panel(main_window, subset_ids)
             return

        state['mean_line'].set_data(t_axis, mean_acg)
        state['shadow_lines'].set_segments(segments)
        ax.set_xlim(t_axis[0], t_axis[-1])
        ax.set_ylim(y_bottom, y_top)
    else:
        canvas.fig.clear()
        canvas.fig.set_facecolor(colors['bg_panel'])
        ax = canvas.fig.add_subplot(111)
        ax.set_facecolor(colors['bg_panel'])
        
        ax.axhline(0, color=colors['text_primary'], linestyle='--', linewidth=1.0, alpha=0.2, zorder=1)
        ax.axvline(0, color=colors['text_primary'], linestyle='--', linewidth=1.0, alpha=0.3, zorder=1)

        shadow_lines = LineCollection(segments, color=colors['plot_acg'], linewidth=0.8, alpha=0.18, zorder=2)
        ax.add_collection(shadow_lines)
        mean_line = _first_plot_artist(ax.plot(t_axis, mean_acg, color=colors['plot_compare'], linewidth=2.5, zorder=4))

        ax.set_xlabel("Time lag (ms)", color=colors['text_secondary'], fontsize=9)
        ax.set_ylabel("Autocorrelation", color=colors['text_secondary'], fontsize=9)
        ax.set_xlim(t_axis[0], t_axis[-1])
        ax.set_ylim(y_bottom, y_top)
        ax.tick_params(colors=colors['text_secondary'], labelsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor(colors['border_subtle'])
            
        canvas._acg_state = {'ax': ax, 'mean_line': mean_line, 'shadow_lines': shadow_lines}

    canvas.draw_idle()
    main_window.pop_acg_summary.setText(f"n={arr.shape[0]}")