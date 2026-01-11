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

from ...analysis import analysis_core

# Set matplotlib logging level to WARNING to suppress font debug messages
matplotlib_logger = logging.getLogger('matplotlib.font_manager')
matplotlib_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def draw_population_timecourse_panel(main_window, subset_ids=None):
    """
    Draw population average timecourse and update summary label.
    Expects: main_window.pop_timecourse_canvas, main_window.pop_timecourse_summary
    """
    # determine subset
    if subset_ids is None:
        try:
            subset_ids = main_window._get_pop_subset_ids()
        except Exception:
            subset_ids = []

    # early exit: nothing selected -> clear canvas + summary
    if not subset_ids:
        fig = main_window.pop_timecourse_canvas.fig
        fig.clear()
        fig.text(
            0.5,
            0.5,
            "No cells selected",
            ha='center',
            color='gray',
            fontsize=10)
        main_window.pop_timecourse_canvas.draw()
        main_window.pop_timecourse_summary.setText(
            "n=0  mean_t2p: N/A  mean_fwhm: N/A")
        return

    traces = []
    metrics_t2p = []
    metrics_fwhm = []

    for cid in subset_ids:
        # adapt to your data layout: convert cluster id -> vision id if needed
        vision_id = cid  # change if you use offset e.g., cid+1

        # Attempt to get precomputed timecourse or a simple trace
        tc = None
        try:
            # Example: prefer a matrix TimeCourse stored somewhere (adapt
            # names)
            vision_id = cid + 1

            sta = main_window.data_manager.vision_stas.get(vision_id)
            stafit = main_window.data_manager.vision_params.get_stafit_for_cell(
                vision_id)

            t_axis, tc_matrix, src = analysis_core.get_sta_timecourse_data(
                sta, stafit, main_window.data_manager.vision_params, vision_id
            )

            if tc_matrix is not None:
                # choose dominant channel
                energies = np.sum(tc_matrix**2, axis=0)
                dom = int(np.argmax(energies))
                tc = tc_matrix[:, dom]

        except Exception:
            tc = None

        if tc is None:
            # fallback: try to extract a small vector from STA or skip
            try:
                sta = main_window.data_manager.vision_stas.get(vision_id)
                if sta is not None:
                    # collapse spatial STA to a single timecourse (simple mean)
                    # adjust dims to match your sta shape
                    tc = np.nanmean(sta, axis=(0, 1))
            except Exception:
                tc = None

        if tc is None:
            continue

        # ensure 1D
        tc = np.asarray(tc).flatten()
        traces.append(tc)

        # compute metrics for this cell using your analysis_core helper
        try:
            m = analysis_core.compute_sta_metrics(
                sta, stafit, main_window.data_manager.vision_params, vision_id
            )

            # expect m dict with keys like "Time to Peak (ms)" and "FWHM (ms)" or similar
            # adapt keys as needed
            if m is not None:
                if "Time to Peak (ms)" in m:
                    metrics_t2p.append(float(m["Time to Peak (ms)"]))
                elif "time_to_peak" in m:
                    metrics_t2p.append(float(m["time_to_peak"]))
                if "FWHM (ms)" in m:
                    metrics_fwhm.append(float(m["FWHM (ms)"]))
                elif "fwhm_ms" in m:
                    metrics_fwhm.append(float(m["fwhm_ms"]))
        except Exception:
            pass

    if not traces:
        fig = main_window.pop_timecourse_canvas.fig
        fig.clear()
        fig.text(
            0.5,
            0.5,
            "No valid timecourses",
            ha='center',
            color='gray',
            fontsize=10)
        main_window.pop_timecourse_canvas.draw()
        main_window.pop_timecourse_summary.setText(
            "n=0  mean_t2p: N/A  mean_fwhm: N/A")
        return

    # align traces length: pad or trim to shortest
    minlen = min(len(t) for t in traces)
    arr = np.vstack([t[:minlen] for t in traces])  # n_cells x n_timepoints
    mean_tc = np.nanmean(arr, axis=0)
    sem = np.nanstd(arr, axis=0) / math.sqrt(arr.shape[0])

    # time axis: assume sample indices; if you have ms per frame, multiply
    # accordingly
    t_axis = np.arange(minlen)

    # plot to canvas
    fig = main_window.pop_timecourse_canvas.fig
    fig.clear()
    ax = fig.add_subplot(111)
    ax.plot(t_axis, mean_tc, linewidth=1.6)
    ax.fill_between(t_axis, mean_tc - sem, mean_tc + sem, alpha=0.25)
    ax.set_title("Population mean ± SEM")
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Response (a.u.)")
    ax.grid(True, linewidth=0.2)
    main_window.pop_timecourse_canvas.draw()

    # update summary label (n, mean t2p, mean fwhm)
    n = arr.shape[0]
    mean_t2p = np.nanmean(metrics_t2p) if metrics_t2p else float("nan")
    mean_fwhm = np.nanmean(metrics_fwhm) if metrics_fwhm else float("nan")
    summary_text = f"n={n}  mean_t2p={mean_t2p:.1f}  mean_fwhm={mean_fwhm:.1f}"
    main_window.pop_timecourse_summary.setText(summary_text)








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
        if hasattr(main_window, 'population_view_enabled') and main_window.population_view_enabled:
            df = main_window.data_manager.cluster_df
            if not df.empty and 'cluster_id' in df.columns:
                if selected_cell_id in df['cluster_id'].values:
                    try:
                        row = df[df['cluster_id'] == selected_cell_id].iloc[0]
                        group_label = row.get('KSLabel')
                        if group_label:
                            subset_cell_ids = df[df['KSLabel'] == group_label]['cluster_id'].tolist()
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