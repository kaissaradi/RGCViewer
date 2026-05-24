"""
Population Panel for RGC Viewer

This module contains population plotting functions that were previously in the plotting module.
"""

import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import LineCollection
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
    peak_time = t_axis[peak_idx]
    peak_val = mean_tc[peak_idx]
    
    mean_fwhm = float("nan")
    try:
        from scipy.signal import peak_widths
        widths, *_ = peak_widths(np.abs(mean_tc), [peak_idx], rel_height=0.5)
        if len(widths) > 0:
            mean_fwhm = widths[0]
    except Exception:
        pass

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
        mean_line, = ax.plot(t_axis, mean_tc, color=colors['plot_mean'], linewidth=2.5, zorder=4)

        # 4. Highlight the Peak Feature
        peak_marker, = ax.plot([peak_time], [peak_val], 'o', color=colors['plot_peak'], markersize=6, zorder=5)
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

        plot_population_rfs_background(
            ax,
            vision_params,
            main_window=main_window,
            sta_height=main_window.data_manager.vision_sta_height,
            subset_cell_ids=subset_cell_ids,
            colors=colors
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


def plot_population_rfs_background(ax, vision_params, main_window, sta_height=None, subset_cell_ids=None, colors=None):
    if colors is None:
        colors = DARK_COLORS

    try:
        all_cell_ids = vision_params.get_cell_ids()
        logger.debug(f"plot_population_rfs_background: got {len(all_cell_ids) if all_cell_ids else 0} cell IDs from vision_params")
    except Exception as e:
        logger.error(f"Failed to get cell IDs from vision_params: {e}")
        return
    
    vision_subset_ids = [cid + 1 for cid in subset_cell_ids] if subset_cell_ids is not None else None

    x_coords, y_coords = [], []
    target_ids = vision_subset_ids if vision_subset_ids else all_cell_ids

    if vision_subset_ids is not None:
        for cell_id in all_cell_ids:
            if cell_id in vision_subset_ids: continue
            try:
                stafit = vision_params.get_stafit_for_cell(cell_id)
                adjusted_y = sta_height - stafit.center_y if sta_height is not None else stafit.center_y
                e = Ellipse(xy=(stafit.center_x, adjusted_y), width=2*stafit.std_x, height=2*stafit.std_y,
                            angle=np.rad2deg(stafit.rot), edgecolor=colors['text_secondary'], facecolor='none', lw=0.75, alpha=0.15)
                ax.add_patch(e)
            except: continue

    for cell_id in target_ids:
        try:
            stafit = vision_params.get_stafit_for_cell(cell_id)
            adjusted_y = sta_height - stafit.center_y if sta_height is not None else stafit.center_y

            e = Ellipse(xy=(stafit.center_x, adjusted_y), width=2*stafit.std_x, height=2*stafit.std_y,
                        angle=np.rad2deg(stafit.rot), edgecolor=colors['text_primary'], facecolor='none', lw=1.0, alpha=0.55)
            ax.add_patch(e)

            # Add ID labels if enabled via population panel checkbox (AC3)
            if hasattr(main_window, 'pop_show_ids_checkbox') and main_window.pop_show_ids_checkbox.isChecked():
                ax.text(stafit.center_x, adjusted_y, str(cell_id), 
                        color=colors['text_secondary'], fontsize=7, ha='center', va='center', alpha=0.8)

            x_coords.append(stafit.center_x)
            y_coords.append(stafit.center_y)
        except: continue

    if x_coords:
        ax.set_xlim(min(x_coords)-20, max(x_coords)+20)
        ax.set_ylim(max(y_coords)+20, min(y_coords)-20)
    else:
        ax.set_xlim(0, 100); ax.set_ylim(100, 0)

    ax.set_title(f"Population Receptive Fields (n={len(target_ids)})", color=colors['text_primary'])
    ax.set_facecolor(colors['bg_panel'])
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(colors=colors['text_secondary'])
    for spine in ax.spines.values(): spine.set_edgecolor(colors['border_subtle'])
    ax.grid(False)


def plot_population_rfs(fig, vision_params, sta_height=None, selected_cell_id=None, subset_cell_ids=None, colors=None):
    if colors is None:
        colors = DARK_COLORS

    fig.clear()
    fig.set_facecolor(colors['bg_panel'])
    ax = fig.add_subplot(111)
    ax.set_facecolor(colors['bg_panel'])

    try:
        all_cell_ids = vision_params.get_cell_ids()
        logger.debug(f"plot_population_rfs: got {len(all_cell_ids) if all_cell_ids else 0} cell IDs")
    except Exception as e:
        logger.error(f"Failed to get cell IDs: {e}")
        ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', color='red')
        return

    if not all_cell_ids:
        ax.text(0.5, 0.5, "No RF data available", ha='center', va='center', color=colors['text_secondary'])
        ax.set_title("Population Receptive Fields", color=colors['text_primary'])
        return

    vision_cell_id_selected = selected_cell_id + 1 if selected_cell_id is not None else None
    selected_cell_has_rf_data = False
    if vision_cell_id_selected is not None and vision_cell_id_selected in all_cell_ids:
        try:
            vision_params.get_stafit_for_cell(vision_cell_id_selected)
            selected_cell_has_rf_data = True
        except Exception:
            selected_cell_has_rf_data = False

    vision_subset_ids = None
    if subset_cell_ids is not None:
        vision_subset_ids = [cid + 1 for cid in subset_cell_ids]

    x_coords, y_coords = [], []
    target_ids_for_bounds = vision_subset_ids if vision_subset_ids else all_cell_ids

    for cell_id in target_ids_for_bounds:
        if cell_id == vision_cell_id_selected and selected_cell_has_rf_data:
            continue
        try:
            stafit = vision_params.get_stafit_for_cell(cell_id)
            x_coords.append(stafit.center_x)
            y_coords.append(stafit.center_y)
        except Exception:
            continue

    if selected_cell_has_rf_data:
        try:
            stafit = vision_params.get_stafit_for_cell(vision_cell_id_selected)
            x_coords.append(stafit.center_x)
            y_coords.append(stafit.center_y)
        except Exception:
            pass

    if x_coords:
        x_range = (min(x_coords) - 20, max(x_coords) + 20)
        y_range = (min(y_coords) - 20, max(y_coords) + 20)
    else:
        x_range = (0, 100)
        y_range = (0, 100)

    if vision_subset_ids is not None:
        for cell_id in all_cell_ids:
            if cell_id in vision_subset_ids:
                continue
            try:
                stafit = vision_params.get_stafit_for_cell(cell_id)
                adjusted_y = sta_height - stafit.center_y if sta_height is not None else stafit.center_y
                ellipse = Ellipse(xy=(stafit.center_x, adjusted_y), width=2 * stafit.std_x, height=2 * stafit.std_y,
                                  angle=np.rad2deg(stafit.rot), edgecolor=colors['text_secondary'], facecolor='none', lw=0.5, alpha=0.05)
                ax.add_patch(ellipse)
            except Exception:
                continue

    target_ids = vision_subset_ids if vision_subset_ids else all_cell_ids
    valid_target_ids = []
    for cell_id in target_ids:
        if cell_id == vision_cell_id_selected and selected_cell_has_rf_data:
            continue
        try:
            stafit = vision_params.get_stafit_for_cell(cell_id)
            adjusted_y = sta_height - stafit.center_y if sta_height is not None else stafit.center_y
            ellipse = Ellipse(xy=(stafit.center_x, adjusted_y), width=2 * stafit.std_x, height=2 * stafit.std_y,
                              angle=np.rad2deg(stafit.rot), edgecolor=colors['text_primary'], facecolor='none', lw=0.5, alpha=0.3)
            ax.add_patch(ellipse)
            valid_target_ids.append(cell_id)
        except Exception:
            continue

    if selected_cell_has_rf_data:
        try:
            stafit = vision_params.get_stafit_for_cell(vision_cell_id_selected)
            adjusted_y = sta_height - stafit.center_y if sta_height is not None else stafit.center_y
            highlight_rgb = QColor(colors['plot_highlight']).getRgbF()[:3]
            highlight_ellipse = Ellipse(xy=(stafit.center_x, adjusted_y), width=2 * stafit.std_x, height=2 * stafit.std_y,
                                        angle=np.rad2deg(stafit.rot), edgecolor=colors['plot_highlight'], facecolor=(*highlight_rgb, 0.3),
                                        lw=2.0, zorder=10)
            ax.add_patch(highlight_ellipse)
        except Exception as e:
            logger.warning("Could not draw highlighted ellipse for cell %s: %s", vision_cell_id_selected, e)

    target_ids = valid_target_ids
    ax.set_xlim(x_range)
    ax.set_ylim(y_range[1], y_range[0])
    ax.set_title(f"Population Receptive Fields (n={len(target_ids)})", color=colors['text_primary'])
    ax.set_xlabel("X (stixels)", color=colors['text_secondary'])
    ax.set_ylabel("Y (stixels)", color=colors['text_secondary'])
    ax.tick_params(colors=colors['text_secondary'])
    for spine in ax.spines.values():
        spine.set_edgecolor(colors['border_subtle'])
    ax.set_aspect('equal', adjustable='box')


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
        mean_line, = ax.plot(t_axis, mean_acg, color=colors['plot_compare'], linewidth=2.5, zorder=4)

        ax.set_xlabel("Time lag (ms)", color=colors['text_secondary'], fontsize=9)
        ax.set_ylabel("Autocorrelation", color=colors['text_secondary'], fontsize=9)
        ax.set_xlim(t_axis[0], t_axis[-1])
        ax.set_ylim(y_bottom, y_top)
        ax.tick_params(colors=colors['text_secondary'], labelsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor(colors['border_subtle'])
            
        canvas._acg_state = {'ax': ax, 'mean_line': mean_line, 'shadow_lines': shadow_lines}

    canvas.draw_idle()
    main_window.pop_acg_summary.setText(f"n={arr.shape[0]}")
