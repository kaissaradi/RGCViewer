"""
Population Panel for RGC Viewer

This module contains population plotting functions that were previously in the plotting module.
"""

import math
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.signal import peak_widths
from qtpy.QtCore import QTimer
from matplotlib.collections import LineCollection

from ...analysis import analysis_core

# Set matplotlib logging level to WARNING to suppress font debug messages
matplotlib_logger = logging.getLogger('matplotlib.font_manager')
matplotlib_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def draw_population_timecourse_panel(main_window, subset_ids=None):
    """
    Draw population average timecourse with futuristic "shadow traces".
    OPTIMIZATION: Uses Hot-Swap rendering, explicit scaling, Trace Caching, 
    and computes metrics on the mean trace to guarantee instant scrolling.
    """
    if subset_ids is None:
        try:
            subset_ids = main_window._get_pop_subset_ids()
        except Exception:
            subset_ids = []

    canvas = main_window.pop_timecourse_canvas
    
    # Early exit: nothing selected
    if not subset_ids:
        canvas.fig.clear()
        canvas.fig.set_facecolor('#1f1f1f')
        canvas.fig.text(0.5, 0.5, "No cells selected", ha='center', color='gray', fontsize=10)
        canvas.draw_idle()
        main_window.pop_timecourse_summary.setText("n=0  mean_t2p: N/A  mean_fwhm: N/A")
        if hasattr(canvas, '_timecourse_state'):
            del canvas._timecourse_state
        return

    # --- 1. Fast Data Extraction with Caching ---
    # Initialize a lightweight cache on the datamanager if it doesn't exist
    if not hasattr(main_window.data_manager, 'pop_trace_cache'):
        main_window.data_manager.pop_trace_cache = {}
    
    trace_cache = main_window.data_manager.pop_trace_cache
    traces = []

    for cid in subset_ids:
        if cid in trace_cache:
            traces.append(trace_cache[cid])
            continue
            
        vision_id = cid + 1
        tc = None
        
        sta = main_window.data_manager.vision_stas.get(vision_id)
        if not sta:
            continue
            
        stafit = main_window.data_manager.vision_params.get_stafit_for_cell(vision_id)
        t_axis, tc_matrix, _ = analysis_core.get_sta_timecourse_data(
            sta, stafit, main_window.data_manager.vision_params, vision_id
        )

        if tc_matrix is not None:
            energies = np.sum(tc_matrix**2, axis=0)
            dom = int(np.argmax(energies))
            tc = tc_matrix[:, dom]
        else:
            try:
                tc = np.nanmean(sta, axis=(0, 1))
            except Exception:
                continue

        if tc is not None:
            flat_tc = np.asarray(tc).flatten()
            trace_cache[cid] = flat_tc
            traces.append(flat_tc)

    if not traces:
        canvas.fig.clear()
        canvas.fig.set_facecolor('#1f1f1f')
        canvas.fig.text(0.5, 0.5, "No valid timecourses", ha='center', color='gray', fontsize=10)
        canvas.draw_idle()
        return

    minlen = min(len(t) for t in traces)
    arr = np.vstack([t[:minlen] for t in traces])
    mean_tc = np.nanmean(arr, axis=0)
    t_axis = np.arange(minlen)
    
    # Feature Extraction (Peak of the mean trace)
    peak_idx = int(np.argmax(np.abs(mean_tc)))
    peak_time = t_axis[peak_idx]
    peak_val = mean_tc[peak_idx]
    
    # Calculate FWHM directly on the mean trace (Instantaneous)
    mean_fwhm = float("nan")
    try:
        # Use abs so it works for ON and OFF cells
        widths, *_ = peak_widths(np.abs(mean_tc), [peak_idx], rel_height=0.5)
        if len(widths) > 0:
            mean_fwhm = widths[0]
    except Exception:
        pass

    # Prepare shadow traces for LineCollection
    segments = [np.column_stack([t_axis, row]) for row in arr]

    # Explicit Scale Calculation
    y_min, y_max = np.min(arr), np.max(arr)
    y_range = y_max - y_min
    if y_range == 0:
        y_range = 1.0
    
    y_bottom = y_min - (0.1 * y_range)
    y_top = y_max + (0.25 * y_range) # Extra padding at the top for the text marker

    # --- 2. Hot-Swap Rendering ---
    if hasattr(canvas, '_timecourse_state') and canvas._timecourse_state['ax'] in canvas.fig.axes:
        # Fast update
        state = canvas._timecourse_state
        ax = state['ax']
        
        state['mean_line'].set_data(t_axis, mean_tc)
        state['shadow_lines'].set_segments(segments)
        state['peak_marker'].set_data([peak_time], [peak_val])
        
        # Adjust text marker position
        state['peak_text'].set_position((peak_time, peak_val + (np.max(mean_tc)*0.1)))
        state['peak_text'].set_text(f" Peak\n Frame {peak_time}")
        
        # Explicitly set the limits to fix the scale bug
        ax.set_xlim(t_axis[0], t_axis[-1])
        ax.set_ylim(y_bottom, y_top)
        
    else:
        # Full Rebuild (Runs once)
        canvas.fig.clear()
        canvas.fig.set_facecolor('#1f1f1f')
        ax = canvas.fig.add_subplot(111)
        ax.set_facecolor('#1f1f1f')
        
        # 1. Zero Line (Reference)
        ax.axhline(0, color='#ffffff', linestyle='--', linewidth=1.0, alpha=0.2, zorder=1)

        # 2. Shadow Traces (LineCollection for massive performance)
        shadow_lines = LineCollection(segments, color='#4282DA', linewidth=0.8, alpha=0.15, zorder=2)
        ax.add_collection(shadow_lines)

        # 3. Solid Mean Trace
        mean_line, = ax.plot(t_axis, mean_tc, color='#00e6a0', linewidth=2.5, zorder=4)

        # 4. Highlight the Peak Feature
        peak_marker, = ax.plot([peak_time], [peak_val], 'o', color='#ffeb3b', markersize=6, zorder=5)
        peak_text = ax.text(peak_time, peak_val + (np.max(mean_tc)*0.1), 
                            f" Peak\n Frame {peak_time}", color='#ffeb3b', 
                            fontsize=8, ha='center', va='bottom')

        # Aesthetics
        #ax.set_title("Population Dynamics", color='white', fontsize=11, pad=10)
        ax.set_xlabel("Time (frames)", color='gray', fontsize=9)
        ax.set_ylabel("Response (a.u.)", color='gray', fontsize=9)
        
        # Apply Explicit Scales
        ax.set_xlim(t_axis[0], t_axis[-1])
        ax.set_ylim(y_bottom, y_top)
        
        ax.tick_params(colors='gray', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')
            
        ax.grid(True, color='#ffffff', alpha=0.05, linestyle='-', linewidth=0.5)

        # Save to state
        canvas._timecourse_state = {
            'ax': ax, 
            'mean_line': mean_line, 
            'shadow_lines': shadow_lines,
            'peak_marker': peak_marker,
            'peak_text': peak_text
        }

    canvas.draw_idle()

    # Update summary label
    n = arr.shape[0]
    main_window.pop_timecourse_summary.setText(f"n={n}  mean_t2p={peak_time:.1f}  mean_fwhm={mean_fwhm:.1f}")

def draw_population_rfs_plot(
        main_window,
        selected_cell_id=None,
        subset_cell_ids=None,
        canvas=None):
    """
    Draws the population receptive field plot.
    OPTIMIZATION: Uses "Hot-Swap" rendering. It draws the background ghosts once
    and updates only the highlight ellipse geometry on subsequent calls.
    """
    # 1. Determine target canvas
    if canvas is None:
        if hasattr(main_window, 'population_view_enabled') and main_window.population_view_enabled:
            canvas = getattr(main_window, 'pop_mosaic_canvas', main_window.rf_canvas)
        else:
            canvas = main_window.rf_canvas

    # 2. Smart Group Detection
    if selected_cell_id is not None and subset_cell_ids is None:
        try:
            # Use the exact visual state of the tree to guarantee accurate subsets!
            subset_cell_ids = main_window._get_pop_subset_ids()
        except Exception:
            pass

    vision_params = main_window.data_manager.vision_params
    if not vision_params:
        canvas.fig.clear()
        canvas.fig.text(0.5, 0.5, "No Vision parameters available", ha='center', va='center', color='gray')
        canvas.draw_idle()
        return

    # --- STATE MANAGEMENT ---
    # We use a custom attribute on the canvas to track the current state of the plot.
    # State structure: {'subset_hash': hash(tuple(subset_ids)), 'highlight_artist': Patch, 'ax': Axes}

    current_subset_tuple = tuple(sorted(subset_cell_ids)) if subset_cell_ids is not None else "ALL"
    current_subset_hash = hash(current_subset_tuple)

    # Check if we can perform a Fast Update (Hot-Swap)
    # We need: existing state, matching subset, and a valid axes object
    can_hot_swap = (
        hasattr(canvas, '_pop_plot_state') and
        canvas._pop_plot_state['subset_hash'] == current_subset_hash and
        canvas._pop_plot_state['ax'] in canvas.fig.axes
    )

    if can_hot_swap:
        # --- TIER 1: FAST UPDATE ---
        # Update the existing ellipse without clearing the figure
        ax = canvas._pop_plot_state['ax']
        highlight_patch = canvas._pop_plot_state['highlight_artist']

        _update_highlight_patch(highlight_patch, vision_params, selected_cell_id, main_window.data_manager.vision_sta_height)

        # Non-blocking draw
        canvas.draw_idle()

    else:
        # --- TIER 2: FULL REBUILD ---
        # Either first run or the group (subset) changed. Rebuild background.
        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111)

        # 1. Draw Background (Ghosts)
        plot_population_rfs_background(
            ax,
            vision_params,
            sta_width=main_window.data_manager.vision_sta_width,
            sta_height=main_window.data_manager.vision_sta_height,
            subset_cell_ids=subset_cell_ids
        )

        # 2. Create Highlight Artist (Initially hidden or generic)
        # We create it once here so we can update it forever after
        highlight_patch = Ellipse(
            xy=(0, 0), width=1, height=1, angle=0,
            edgecolor='cyan', facecolor=(0.0, 1.0, 1.0, 0.3),
            lw=2.0, zorder=10, visible=False
        )
        ax.add_patch(highlight_patch)

        # 3. Apply initial highlight position
        _update_highlight_patch(highlight_patch, vision_params, selected_cell_id, main_window.data_manager.vision_sta_height)

        # 4. Save State
        canvas._pop_plot_state = {
            'subset_hash': current_subset_hash,
            'highlight_artist': highlight_patch,
            'ax': ax
        }

        canvas.draw_idle()


def _update_highlight_patch(patch, vision_params, cell_id, sta_height):
    """Helper to update the geometry of the persistent highlight ellipse."""
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
    except Exception:
        # Cell has no RF fit data
        patch.set_visible(False)


def plot_population_rfs_background(ax, vision_params, sta_width=None, sta_height=None, subset_cell_ids=None):
    """
    Draws only the static 'ghost' ellipses for the population.
    This is called only when the group changes.
    """
    all_cell_ids = vision_params.get_cell_ids()
    vision_subset_ids = [cid + 1 for cid in subset_cell_ids] if subset_cell_ids is not None else None

    # Auto-scale variables
    x_coords, y_coords = [], []
    target_ids = vision_subset_ids if vision_subset_ids else all_cell_ids

    # --- Draw Ghost Population (Excluded cells) ---
    if vision_subset_ids is not None:
        for cell_id in all_cell_ids:
            if cell_id in vision_subset_ids: continue
            try:
                stafit = vision_params.get_stafit_for_cell(cell_id)
                adjusted_y = sta_height - stafit.center_y if sta_height is not None else stafit.center_y
                e = Ellipse(xy=(stafit.center_x, adjusted_y), width=2*stafit.std_x, height=2*stafit.std_y,
                            angle=np.rad2deg(stafit.rot), edgecolor='gray', facecolor='none', lw=0.5, alpha=0.05)
                ax.add_patch(e)
            except: continue

    # --- Draw Target Population (Included cells) ---
    for cell_id in target_ids:
        try:
            stafit = vision_params.get_stafit_for_cell(cell_id)
            adjusted_y = sta_height - stafit.center_y if sta_height is not None else stafit.center_y

            # Draw standard white ellipse
            e = Ellipse(xy=(stafit.center_x, adjusted_y), width=2*stafit.std_x, height=2*stafit.std_y,
                        angle=np.rad2deg(stafit.rot), edgecolor='white', facecolor='none', lw=0.5, alpha=0.3)
            ax.add_patch(e)

            x_coords.append(stafit.center_x)
            y_coords.append(stafit.center_y)
        except: continue

    # --- Styling ---
    if x_coords:
        ax.set_xlim(min(x_coords)-20, max(x_coords)+20)
        ax.set_ylim(max(y_coords)+20, min(y_coords)-20) # Inverted Y
    else:
        ax.set_xlim(0, 100); ax.set_ylim(100, 0)

    ax.set_title(f"Population Receptive Fields (n={len(target_ids)})", color='white')
    ax.set_facecolor('#1f1f1f')
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(colors='gray')
    for spine in ax.spines.values(): spine.set_edgecolor('gray')









# --- Core Plotting Functions (Moved from analysis_core.py) ---

def plot_population_rfs(fig, vision_params, sta_width=None, sta_height=None, selected_cell_id=None, subset_cell_ids=None):
    """
    Visualizes the receptive fields of all cells, highlighting the selected cell
    by filling its true ellipse shape and making other ellipses more faint. If the selected
    cell has no RF data, no cell is highlighted.

    Args:
        subset_cell_ids (list): List of 0-indexed cluster IDs to include in the population view.
                                If None, all cells are shown.
    """
    fig.clear()
    ax = fig.add_subplot(111)

    all_cell_ids = vision_params.get_cell_ids()

    if not all_cell_ids:
        ax.text(
            0.5,
            0.5,
            "No RF data available",
            ha='center',
            va='center',
            color='gray')
        ax.set_title("Population Receptive Fields", color='white')
        return

    vision_cell_id_selected = selected_cell_id +1 if selected_cell_id is not None else None

    # Check if the selected cell actually has RF data available (not just that
    # it exists as a cell)
    selected_cell_has_rf_data = False
    if vision_cell_id_selected is not None and vision_cell_id_selected in all_cell_ids:
        try:
            vision_params.get_stafit_for_cell(vision_cell_id_selected)
            selected_cell_has_rf_data = True
        except Exception:
            selected_cell_has_rf_data = False

    # Convert subset IDs to Vision IDs (1-based) if provided
    vision_subset_ids = None
    if subset_cell_ids is not None:
        vision_subset_ids = [cid + 1 for cid in subset_cell_ids]

    # --- Auto-determine plot boundaries from data ---
    x_coords, y_coords = [], []

    # Determine which cells to use for boundary calculation
    # If subset is provided, prioritize their boundaries, but maybe keep global context?
    # Let's use the target population (subset or all) for boundaries
    target_ids_for_bounds = vision_subset_ids if vision_subset_ids else all_cell_ids

    for cell_id in target_ids_for_bounds:
        # Only skip the selected cell if it has RF data (to avoid double-processing)
        # If selected cell doesn't have RF data, include it in boundaries
        # calculation normally
        if cell_id == vision_cell_id_selected and selected_cell_has_rf_data:
            continue

        try:
            stafit = vision_params.get_stafit_for_cell(cell_id)
            x_coords.append(stafit.center_x)
            y_coords.append(stafit.center_y)
        except Exception:
            continue

    # Also include the selected cell in boundary calculation if it has RF data
    if selected_cell_has_rf_data:
        try:
            stafit = vision_params.get_stafit_for_cell(vision_cell_id_selected)
            x_coords.append(stafit.center_x)
            y_coords.append(stafit.center_y)
        except Exception:
            pass  # If selected cell has no RF data, it was already handled above

    if x_coords:
        x_range = (min(x_coords) - 20, max(x_coords) + 20)
        y_range = (min(y_coords) - 20, max(y_coords) + 20)
    else:
        x_range = (0, 100)
        y_range = (0, 100)

    # --- STAGE 1: Draw "Ghost" Population (Optional context) ---
    # If a subset is defined, draw the excluded cells very faintly
    if vision_subset_ids is not None:
        for cell_id in all_cell_ids:
            if cell_id in vision_subset_ids:
                continue  # specific drawing later

            try:
                stafit = vision_params.get_stafit_for_cell(cell_id)
                adjusted_y = sta_height - stafit.center_y if sta_height is not None else stafit.center_y

                ellipse = Ellipse(
                    xy=(stafit.center_x, adjusted_y),
                    width=2 * stafit.std_x,
                    height=2 * stafit.std_y,
                    angle=np.rad2deg(stafit.rot),
                    edgecolor='gray',
                    facecolor='none',
                    lw=0.5,
                    alpha=0.05  # Very faint ghost
                )
                ax.add_patch(ellipse)
            except Exception:
                continue

    # --- STAGE 2: Draw the Target Population (Subset or All) ---
    target_ids = vision_subset_ids if vision_subset_ids else all_cell_ids

    # Check if the selected cell actually has RF data available (not just that
    # it exists as a cell)
    selected_cell_has_rf_data = False
    if vision_cell_id_selected is not None and vision_cell_id_selected in all_cell_ids:
        try:
            # Test if selected cell has RF data by attempting to get its STAFit
            test_stafit = vision_params.get_stafit_for_cell(
                vision_cell_id_selected)
            selected_cell_has_rf_data = True
        except Exception:
            selected_cell_has_rf_data = False

    valid_target_ids = []
    for cell_id in target_ids:
        # Skip the selected cell for now if it has RF data; we'll draw it on top.
        # If the selected cell doesn't have RF data, include it in the general
        # population
        if cell_id == vision_cell_id_selected and selected_cell_has_rf_data:
            continue

        try:
            stafit = vision_params.get_stafit_for_cell(cell_id)
            adjusted_y = sta_height - stafit.center_y if sta_height is not None else stafit.center_y

            ellipse = Ellipse(
                xy=(stafit.center_x, adjusted_y),
                width=2 * stafit.std_x,
                height=2 * stafit.std_y,
                angle=np.rad2deg(stafit.rot),
                edgecolor='white',
                facecolor='none',
                lw=0.5,
                alpha=0.3  # Standard visibility
            )
            ax.add_patch(ellipse)
            valid_target_ids.append(cell_id)
        except Exception:
            continue

    # --- STAGE 3: Draw the single, highlighted ellipse on top of everything else ---
    # Only highlight if the cell exists AND has RF data
    if selected_cell_has_rf_data:
        try:
            stafit = vision_params.get_stafit_for_cell(vision_cell_id_selected)
            adjusted_y = sta_height - stafit.center_y if sta_height is not None else stafit.center_y

            # This now correctly uses the selected cell's own parameters for
            # the highlight
            highlight_ellipse = Ellipse(
                xy=(stafit.center_x, adjusted_y),
                width=2 * stafit.std_x,
                height=2 * stafit.std_y,
                angle=np.rad2deg(stafit.rot),
                edgecolor='cyan',  # Changed to cyan as per request
                # Filled with semi-transparent cyan
                facecolor=(0.0, 1.0, 1.0, 0.3),
                lw=2.0,  # Thicker line
                zorder=10  # Ensure it's drawn on top
            )
            ax.add_patch(highlight_ellipse)
        except Exception as e:
            # This will now only be reached for unexpected errors, not for
            # missing cells.
            logger.warning(
                "Could not draw highlighted ellipse for cell %s: %s",
                vision_cell_id_selected,
                e)

    # Update target_ids to only include cells that actually have RF data
    target_ids = valid_target_ids

    # --- Plot styling ---
    ax.set_xlim(x_range)
    ax.set_ylim(y_range[1], y_range[0])
    ax.set_title(
        f"Population Receptive Fields (n={len(target_ids)})",
        color='white')
    ax.set_xlabel("X (stixels)", color='gray')
    ax.set_ylabel("Y (stixels)", color='gray')
    ax.set_facecolor('#1f1f1f')
    ax.tick_params(colors='gray')
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
    ax.set_aspect('equal', adjustable='box')







def plot_rich_ei(fig, median_ei, channel_positions, features, _sampling_rate, _pre_samples=20):
    """
    Plots the electrical image (EI) on the electrode array.
    """
    fig.clear()
    ax = fig.add_subplot(111)

    # Simple scatter plot of channel positions colored by max amplitude
    if median_ei is not None and channel_positions is not None:
        max_amplitudes = np.max(np.abs(median_ei), axis=1)

        # Ensure dimensions match
        if len(max_amplitudes) == len(channel_positions):
            sc = ax.scatter(
                channel_positions[:, 0],
                channel_positions[:, 1],
                c=max_amplitudes,
                cmap='viridis',
                s=50,
                alpha=0.8
            )

            # Add a colorbar
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label('Max Amplitude (µV)', color='gray')
            cbar.ax.yaxis.set_tick_params(color='gray')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='gray')

            # Overlay Center of Mass if available
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

                    ax.legend(loc='upper right', facecolor='#1f1f1f', labelcolor='white')

        else:
            ax.text(0.5, 0.5, f"Dimension Mismatch: EI={len(max_amplitudes)}, Pos={len(channel_positions)}",
                    ha='center', va='center', color='red')
    else:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center', color='gray')

    ax.set_title('Electrical Image', color='white')
    ax.set_xlabel('X (µm)', color='gray')
    ax.set_ylabel('Y (µm)', color='gray')
    ax.set_facecolor('#1f1f1f')
    ax.tick_params(colors='gray')
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
    ax.set_aspect('equal')

    fig.tight_layout()

def draw_population_acg_panel(main_window, subset_ids=None):
    """
    Draw population average ACG with shadow traces.
    OPTIMIZATION: Uses Hot-Swap rendering and LineCollections for instantaneous updates.
    """
    if subset_ids is None:
        try:
            subset_ids = main_window._get_pop_subset_ids()
        except Exception:
            subset_ids = []

    canvas = getattr(main_window, 'pop_acg_canvas', None)
    if canvas is None: return
    
    # Early exit: nothing selected
    if not subset_ids:
        canvas.fig.clear()
        canvas.fig.set_facecolor('#1f1f1f')
        canvas.fig.text(0.5, 0.5, "No cells selected", ha='center', color='gray', fontsize=10)
        canvas.draw_idle()
        main_window.pop_acg_summary.setText("n=0")
        if hasattr(canvas, '_acg_state'):
            del canvas._acg_state
        return

    from matplotlib.collections import LineCollection
    import numpy as np

    traces = []
    t_axis = None

    for cid in subset_ids:
        try:
            time_lags, acg_norm = main_window.data_manager.get_acg_data(cid)
            if time_lags is not None and acg_norm is not None and len(time_lags) > 1:
                if t_axis is None:
                    t_axis = time_lags
                
                # Make sure length matches to avoid broadcast errors
                if len(acg_norm) == len(t_axis):
                    traces.append(acg_norm)
        except Exception:
            continue

    if not traces:
        canvas.fig.clear()
        canvas.fig.set_facecolor('#1f1f1f')
        canvas.fig.text(0.5, 0.5, "No valid ACG data", ha='center', color='gray', fontsize=10)
        canvas.draw_idle()
        return

    arr = np.vstack(traces)
    mean_acg = np.nanmean(arr, axis=0)
    
    # Segments for shadow traces
    segments = [np.column_stack([t_axis, row]) for row in arr]

    # Explicit Scale Calculation
    y_min, y_max = np.min(arr), np.max(arr)
    y_range = y_max - y_min
    if y_range == 0: y_range = 1.0
    y_bottom = y_min - (0.05 * y_range)
    y_top = y_max + (0.05 * y_range)

    # --- Hot-Swap Rendering ---
    if hasattr(canvas, '_acg_state') and canvas._acg_state['ax'] in canvas.fig.axes:
        state = canvas._acg_state
        ax = state['ax']
        
        state['mean_line'].set_data(t_axis, mean_acg)
        state['shadow_lines'].set_segments(segments)
        
        ax.set_xlim(t_axis[0], t_axis[-1])
        ax.set_ylim(y_bottom, y_top)
    else:
        # Full Rebuild
        canvas.fig.clear()
        canvas.fig.set_facecolor('#1f1f1f')
        ax = canvas.fig.add_subplot(111)
        ax.set_facecolor('#1f1f1f')
        
        # Zero Lines (Vertical at lag=0, Horizontal at y=0)
        ax.axhline(0, color='#ffffff', linestyle='--', linewidth=1.0, alpha=0.2, zorder=1)
        ax.axvline(0, color='#ffffff', linestyle='--', linewidth=1.0, alpha=0.3, zorder=1)

        # Shadow Traces (Purple)
        shadow_lines = LineCollection(segments, color='#9b59b6', linewidth=0.8, alpha=0.15, zorder=2)
        ax.add_collection(shadow_lines)

        # Solid Mean Trace (Neon Orange)
        mean_line, = ax.plot(t_axis, mean_acg, color='#ff9800', linewidth=2.5, zorder=4)

        # Aesthetics
        ax.set_xlabel("Time lag (ms)", color='gray', fontsize=9)
        ax.set_ylabel("Autocorrelation", color='gray', fontsize=9)
        
        ax.set_xlim(t_axis[0], t_axis[-1])
        ax.set_ylim(y_bottom, y_top)
        
        ax.tick_params(colors='gray', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')
            
        canvas._acg_state = {
            'ax': ax, 
            'mean_line': mean_line, 
            'shadow_lines': shadow_lines,
        }

    canvas.draw_idle()
    main_window.pop_acg_summary.setText(f"n={arr.shape[0]}")