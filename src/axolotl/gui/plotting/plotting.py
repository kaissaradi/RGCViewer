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


def draw_sta_metrics(main_window, cluster_id):
    """
    Calculates and displays STA metrics in the text box with improved organization.
    """
    vision_cluster_id = cluster_id + 1
    has_vision_sta = main_window.data_manager.vision_stas and vision_cluster_id in main_window.data_manager.vision_stas

    if has_vision_sta:
        sta_data = main_window.data_manager.vision_stas[vision_cluster_id]
        stafit = main_window.data_manager.vision_params.get_stafit_for_cell(
            vision_cluster_id)

        metrics = analysis_core.compute_sta_metrics(
            sta_data, stafit, main_window.data_manager.vision_params, vision_cluster_id)

        # Format metrics as HTML table with sections
        html = """
        <style>
            table { width: 100%; border-collapse: collapse; color: #e0e0e0; font-family: sans-serif; }
            th { text-align: left; color: #aaa; border-bottom: 1px solid #555; padding: 4px; }
            td { padding: 6px; border-bottom: 1px solid #333; }
            .section { font-weight: bold; color: #4282DA; margin-top: 12px; display: block; font-size: 1.1em; }
            .subsection { font-weight: bold; color: #6AA84F; margin-top: 8px; display: block; }
            .metric-name { font-weight: 600; color: #cccccc; }
            .metric-value { color: #ffffff; }
            .highlight { background-color: rgba(66, 130, 218, 0.1); }
        </style>
        """

        html += "<table>"

        # Temporal Properties
        html += "<tr><th colspan='2' class='section'>Temporal Properties</th></tr>"
        temporal_keys = [
            "Dominant Channel",
            "Polarity",
            "Time to Peak (ms)",
            "Response Duration (ms)",
            "Zero Crossing (ms)",
            "FWHM (Duration)",
            "Biphasic Index",
            "SNR (std ratio)"]
        for k in temporal_keys:
            if k in metrics:
                html += f"<tr><td class='metric-name'>{k}</td><td class='metric-value'>{metrics[k]}</td></tr>"

        # Response Strength
        html += "<tr><th colspan='2' class='subsection'>Response Strength</th></tr>"
        strength_keys = ["Response Integral", "Total Energy"]
        for k in strength_keys:
            if k in metrics:
                html += f"<tr><td class='metric-name'>{k}</td><td class='metric-value'>{metrics[k]}</td></tr>"

        # Color Properties
        if "Color Opponency" in metrics:
            html += "<tr><th colspan='2' class='subsection'>Color Properties</th></tr>"
            html += f"<tr><td class='metric-name'>Color Opponency</td><td class='metric-value'>{metrics['Color Opponency']}</td></tr>"

        # Spatial Properties
        html += "<tr><th colspan='2' class='section'>Spatial Properties</th></tr>"
        spatial_keys = [
            "RF Center X",
            "RF Center Y",
            "RF Sigma X",
            "RF Sigma Y",
            "Orientation",
            "RF Area (sq stix)",
            "RF Ellipticity (σy/σx)",
            "RF Elongation"]
        for k in spatial_keys:
            if k in metrics:
                html += f"<tr><td class='metric-name'>{k}</td><td class='metric-value'>{metrics[k]}</td></tr>"

        # Spatial Asymmetry
        html += "<tr><th colspan='2' class='subsection'>Spatial Asymmetry</th></tr>"
        asymmetry_keys = [
            "Spatial Peak",
            "Spatial Trough",
            "Peak/Trough Ratio",
            "Spatial Skewness"]
        for k in asymmetry_keys:
            if k in metrics:
                html += f"<tr><td class='metric-name'>{k}</td><td class='metric-value'>{metrics[k]}</td></tr>"

        html += "</table>"

        main_window.sta_metrics_text.setHtml(html)
    else:
        main_window.sta_metrics_text.setHtml(
            "<div style='color:gray; text-align:center; padding:20px;'>No STA Data Available</div>")


def draw_temporal_filter_plot(main_window, cluster_id):
    """
    Draws the detailed temporal filter analysis plot.
    """
    vision_cluster_id = cluster_id + 1
    has_vision_sta = main_window.data_manager.vision_stas and vision_cluster_id in main_window.data_manager.vision_stas

    if has_vision_sta:
        sta_data = main_window.data_manager.vision_stas[vision_cluster_id]
        stafit = main_window.data_manager.vision_params.get_stafit_for_cell(
            vision_cluster_id)

        plot_temporal_filter_properties(
            main_window.temporal_filter_canvas.fig,
            sta_data,
            stafit,
            main_window.data_manager.vision_params,
            vision_cluster_id)
        main_window.temporal_filter_canvas.draw()
    else:
        main_window.temporal_filter_canvas.fig.clear()
        main_window.temporal_filter_canvas.fig.text(
            0.5, 0.5, "No Data", ha='center', va='center', color='gray')
        main_window.temporal_filter_canvas.draw()


def draw_sta_plot(main_window, cluster_id):
    """
    MODIFIED: Fetches STAFit data and passes it to the plotting function.
    Now also triggers metrics and temporal filter updates.
    """
    vision_cluster_id = cluster_id + 1
    has_vision_sta = main_window.data_manager.vision_stas and vision_cluster_id in main_window.data_manager.vision_stas

    if has_vision_sta:
        stop_sta_animation(main_window)

        sta_data = main_window.data_manager.vision_stas[vision_cluster_id]
        # --- ADDED: Get STAFit data and store it for other functions to use ---
        stafit = main_window.data_manager.vision_params.get_stafit_for_cell(
            vision_cluster_id)
        main_window.current_sta_data = sta_data
        main_window.current_stafit = stafit  # <-- Store the fit
        main_window.current_sta_cluster_id = vision_cluster_id

        n_frames = sta_data.red.shape[2]
        main_window.total_sta_frames = n_frames
        main_window.sta_frame_slider.setMaximum(n_frames - 1)

        all_channels = np.stack(
            [sta_data.red, sta_data.green, sta_data.blue], axis=0)
        frame_energies = np.max(np.abs(all_channels), axis=(0, 1, 2))
        peak_frame_index = np.argmax(frame_energies)
        main_window.current_frame_index = peak_frame_index

        main_window.sta_frame_slider.setValue(peak_frame_index)
        main_window.sta_frame_label.setText(
            f"Frame: {peak_frame_index + 1}/{n_frames}")
        main_window.sta_frame_slider.setEnabled(True)

        # Use the RF canvas instead of the old sta_canvas
        main_window.rf_canvas.fig.clear()
        animate_sta_movie(
            main_window.rf_canvas.fig,
            sta_data,
            stafit=stafit,  # <-- Pass the fit to the plotting function
            frame_index=peak_frame_index,
            sta_width=main_window.data_manager.vision_sta_width,
            sta_height=main_window.data_manager.vision_sta_height
        )
        main_window.rf_canvas.draw()

        # --- Update New Panels ---
        draw_sta_metrics(main_window, cluster_id)
        draw_temporal_filter_plot(main_window, cluster_id)

    else:
        # No Vision STA data available
        main_window.rf_canvas.fig.clear()
        main_window.rf_canvas.fig.text(
            0.5,
            0.5,
            "No Vision STA data available",
            ha='center',
            va='center',
            color='gray')
        main_window.rf_canvas.draw()
        main_window.sta_frame_slider.setEnabled(False)

        # Clear other panels
        main_window.sta_metrics_text.clear()
        main_window.temporal_filter_canvas.fig.clear()
        main_window.temporal_filter_canvas.draw()


def draw_population_rfs_plot(
        main_window,
        selected_cell_id=None,
        subset_cell_ids=None,
        canvas=None):
    """Draws the population receptive field plot showing all cell RFs."""
    logger.debug(
        "Received selected_cell_id=%s, subset_cell_ids=%s for population RF plot",
        selected_cell_id,
        subset_cell_ids)

    # 1. Determine target canvas
    if canvas is None:
        # If population view is enabled, prefer the dedicated mosaic canvas
        # unless specifically overridden or if we are in a mode that implies
        # the main view
        if hasattr(
                main_window,
                'population_view_enabled') and main_window.population_view_enabled:
            canvas = getattr(
                main_window,
                'pop_mosaic_canvas',
                main_window.rf_canvas)
        else:
            canvas = main_window.rf_canvas

    # 2. Smart Group Detection (if single unit selected but no subset provided)
    # This ensures that when we click a unit in Split View, the Context pane
    # shows its group.
    if selected_cell_id is not None and subset_cell_ids is None:
        # Only do auto-group detection if we are targeting the population canvas
        # (If we are in "Population RFs" main view mode, maybe we want to see ALL cells?)
        # Let's assume for Split View we want the group context.
        if hasattr(
                main_window,
                'population_view_enabled') and main_window.population_view_enabled:
            df = main_window.data_manager.cluster_df
            if not df.empty and 'cluster_id' in df.columns:
                if selected_cell_id in df['cluster_id'].values:
                    # Find the group/label
                    try:
                        row = df[df['cluster_id'] == selected_cell_id].iloc[0]
                        # Try 'KSLabel' first, then 'group' if available
                        # (custom groups)
                        group_label = row.get('KSLabel')

                        if group_label:
                            subset_cell_ids = df[df['KSLabel'] ==
                                                 group_label]['cluster_id'].tolist()
                    except Exception as e:
                        logger.warning(
                            f"Error determining group for cluster {selected_cell_id}: {e}")

    has_vision_params = main_window.data_manager.vision_params

    if has_vision_params:
        canvas.fig.clear()

        plot_population_rfs(
            canvas.fig,
            main_window.data_manager.vision_params,
            sta_width=main_window.data_manager.vision_sta_width,
            sta_height=main_window.data_manager.vision_sta_height,
            selected_cell_id=selected_cell_id,
            # Pass the ID along to the core plotting function
            subset_cell_ids=subset_cell_ids
        )
        canvas.draw()
    else:
        canvas.fig.clear()
        canvas.fig.text(0.5, 0.5, "No Vision parameters available",
                        ha='center', va='center', color='gray')
        canvas.draw()


def draw_sta_timecourse_plot(main_window, cluster_id):
    # Draws the STA timecourse plot for a specific cell.
    vision_cluster_id = cluster_id + 1
    has_vision_sta = main_window.data_manager.vision_stas and vision_cluster_id in main_window.data_manager.vision_stas
    if has_vision_sta:
        sta_data = main_window.data_manager.vision_stas[vision_cluster_id]
        stafit = main_window.data_manager.vision_params.get_stafit_for_cell(
            vision_cluster_id)
        main_window.timecourse_canvas.fig.clear()
        plot_sta_timecourse(
            main_window.timecourse_canvas.fig,
            sta_data,
            stafit,
            main_window.data_manager.vision_params,
            vision_cluster_id
        )
        main_window.timecourse_canvas.draw()
    else:
        main_window.timecourse_canvas.fig.clear()
        main_window.timecourse_canvas.fig.text(
            0.5,
            0.5,
            "No Vision STA data available",
            ha='center',
            va='center',
            color='gray')
        main_window.timecourse_canvas.draw()


def draw_sta_animation_plot(main_window, cluster_id):
    # Draws the STA animation plot for a specific cell.
    vision_cluster_id = cluster_id + 1
    has_vision_sta = main_window.data_manager.vision_stas and vision_cluster_id in main_window.data_manager.vision_stas
    if has_vision_sta:
        main_window.current_sta_data = main_window.data_manager.vision_stas[vision_cluster_id]
        main_window.current_sta_cluster_id = vision_cluster_id
        main_window.current_frame_index = 0
        n_frames = main_window.current_sta_data.red.shape[2]
        main_window.total_sta_frames = n_frames

        main_window.sta_frame_slider.setMinimum(0)
        main_window.sta_frame_slider.setMaximum(n_frames - 1)
        main_window.sta_frame_slider.setValue(0)
        main_window.sta_frame_label.setText(f"Frame: 1/{n_frames}")
        main_window.sta_frame_slider.setEnabled(True)

        # Ensure the animation timer is properly created
        if main_window.sta_animation_timer is None:
            main_window.sta_animation_timer = QTimer()
            main_window.sta_animation_timer.timeout.connect(
                main_window._advance_frame_internal)

        # Stop any currently running animation first to prevent conflicts
        if main_window.sta_animation_timer and main_window.sta_animation_timer.isActive():
            main_window.sta_animation_timer.stop()
        else:
            # Start the animation only if it's not already running
            main_window.sta_animation_timer.start(100)

        # Update the RF canvas with the first frame
        main_window.rf_canvas.fig.clear()
        animate_sta_movie(
            main_window.rf_canvas.fig,
            main_window.current_sta_data,
            stafit=main_window.current_stafit,
            frame_index=main_window.current_frame_index,
            sta_width=main_window.data_manager.vision_sta_width,
            sta_height=main_window.data_manager.vision_sta_height
        )
        main_window.rf_canvas.draw()
    else:
        main_window.rf_canvas.fig.clear()
        main_window.rf_canvas.fig.text(
            0.5,
            0.5,
            "No Vision STA data available",
            ha='center',
            va='center',
            color='gray')
        main_window.rf_canvas.draw()
        main_window.sta_frame_slider.setEnabled(False)


def update_sta_frame(main_window):
    # Updates the STA visualization to the next frame in the animation.
    # This is for external calls (manual frame advance) and will stop the
    # timer.
    if not hasattr(
            main_window,
            'current_sta_data') or main_window.current_sta_data is None:
        # Stop the timer if there's no data to animate
        stop_sta_animation(main_window)
        return

    main_window.current_frame_index = (
        main_window.current_frame_index + 1) % main_window.total_sta_frames
    main_window.sta_frame_slider.setValue(main_window.current_frame_index)
    main_window.sta_frame_label.setText(
        f"Frame: {main_window.current_frame_index + 1}/{main_window.total_sta_frames}")

    main_window.rf_canvas.fig.clear()
    animate_sta_movie(
        main_window.rf_canvas.fig,
        main_window.current_sta_data,
        stafit=main_window.current_stafit,  # <-- Pass the stored fit during animation
        frame_index=main_window.current_frame_index,
        sta_width=main_window.data_manager.vision_sta_width,
        sta_height=main_window.data_manager.vision_sta_height
    )
    main_window.rf_canvas.draw()


def stop_sta_animation(main_window):
    # Stops the STA animation if running.
    if hasattr(
            main_window,
            'sta_animation_timer') and main_window.sta_animation_timer and main_window.sta_animation_timer.isActive():
        main_window.sta_animation_timer.stop()


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


def plot_sta_timecourse(
        fig,
        sta_data,
        stafit,
        vision_params,
        cell_id,
        _sampling_rate=20):
    """
    Visualizes the timecourse of the STA response for a specific cell.
    The x-axis shows time from -500ms to 0ms before the spike.

    Args:
        fig (matplotlib.figure.Figure): The figure object to draw on.
        sta_data (STAContainer): Named tuple containing the raw STA movie.
        stafit (STAFit): Named tuple containing the Gaussian fit parameters.
        vision_params (VisionCellDataTable): Object containing pre-calculated timecourse data.
        cell_id (int): The ID of the cell to plot (1-indexed for vision data).
        sampling_rate (float): Sampling rate in Hz for the STA data (stixels per second).
    """
    fig.clear()

    timecourse_matrix = None
    try:
        red_tc = vision_params.get_data_for_cell(cell_id, 'RedTimeCourse')
        green_tc = vision_params.get_data_for_cell(cell_id, 'GreenTimeCourse')
        blue_tc = vision_params.get_data_for_cell(cell_id, 'BlueTimeCourse')
        if red_tc is not None and green_tc is not None and blue_tc is not None:
            timecourse_matrix = np.stack([red_tc, green_tc, blue_tc], axis=1)
    except Exception:
        timecourse_matrix = None

    if timecourse_matrix is not None and hasattr(timecourse_matrix, 'shape'):
        if len(timecourse_matrix.shape) == 2:
            n_timepoints, n_channels = timecourse_matrix.shape

            # --- DYNAMIC X-AXIS CALCULATION ---
            # Get the refresh time (in ms) from the STA data container.
            # This makes the time axis accurate to the original experiment.
            refresh_ms = getattr(sta_data, 'refresh_time', 1000.0 / 60.0)
            total_duration_ms = (n_timepoints - 1) * refresh_ms

            # Create the accurate time axis based on the data's properties
            time_axis = np.linspace(-total_duration_ms, 0, n_timepoints)

            ax = fig.add_subplot(111)

            n_channels_to_plot = min(n_channels, 3)
            channel_names = ['Red', 'Green', 'Blue'][:n_channels_to_plot]
            colors = ['red', 'green', 'blue'][:n_channels_to_plot]

            for i in range(n_channels_to_plot):
                ax.plot(
                    time_axis,
                    timecourse_matrix[:,
                                      i],
                    color=colors[i],
                    linewidth=1.5,
                    label=channel_names[i])

            ax.set_title("STA Timecourse (Pre-calculated)", color='white')
            ax.set_xlabel("Time (ms)", color='gray')
            ax.set_ylabel("Response", color='gray')
            ax.grid(True, alpha=0.3)
            ax.legend(facecolor='#1f1f1f', labelcolor='white')

            ax.set_facecolor('#1f1f1f')
            ax.tick_params(colors='gray')
            for spine in ax.spines.values():
                spine.set_edgecolor('gray')

            ax.axvline(x=0, color='white', linestyle='--', alpha=0.7)
            ax.axhline(y=0, color='white', linestyle=':',
                       alpha=0.5)  # Add dotted line at y=0

            # --- ACCURATE Y-AXIS SCALING ---
            # This logic fits the axis tightly to the min/max of the saved
            # data.
            if timecourse_matrix.size > 0:
                y_min = timecourse_matrix.min()
                y_max = timecourse_matrix.max()
                y_range = y_max - y_min if y_max > y_min else 1.0
                y_margin = y_range * 0.10  # Add a 10% margin for readability
                ax.set_ylim(y_min - y_margin, y_max + y_margin)

            return

    # Fallback logic remains unchanged
    logger.warning(
        "No precomputed timecourse for cell %s; recomputing",
        cell_id)

    red_channel = sta_data.red
    green_channel = sta_data.green
    blue_channel = sta_data.blue

    n_timepoints = red_channel.shape[2]

    center_x = int(stafit.center_x)
    center_y = int(stafit.center_y)
    std_x = int(max(1, stafit.std_x))
    std_y = int(max(1, stafit.std_y))

    x_min = max(0, center_x - std_x)
    x_max = min(red_channel.shape[1], center_x + std_x + 1)
    y_min = max(0, center_y - std_y)
    y_max = min(red_channel.shape[0], center_y + std_y + 1)

    red_timecourse = np.mean(
        red_channel[y_min:y_max, x_min:x_max], axis=(0, 1))
    green_timecourse = np.mean(
        green_channel[y_min:y_max, x_min:x_max], axis=(0, 1))
    blue_timecourse = np.mean(
        blue_channel[y_min:y_max, x_min:x_max], axis=(0, 1))

    # Fallback uses a hardcoded duration as it has no refresh_time metadata
    total_time_ms = 1500
    time_axis = np.linspace(-total_time_ms, 0, n_timepoints)

    ax = fig.add_subplot(111)

    ax.plot(time_axis, red_timecourse, color='red', linewidth=1.5, label='Red')
    ax.plot(
        time_axis,
        green_timecourse,
        color='green',
        linewidth=1.5,
        label='Green')
    ax.plot(
        time_axis,
        blue_timecourse,
        color='blue',
        linewidth=1.5,
        label='Blue')

    ax.set_title("STA Timecourse (Recalculated)", color='white')
    ax.set_xlabel("Time (ms)", color='gray')
    ax.set_ylabel("Response", color='gray')
    ax.grid(True, alpha=0.3)
    ax.legend(facecolor='#1f1f1f', labelcolor='white')

    ax.set_facecolor('#1f1f1f')
    ax.tick_params(colors='gray')
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')

    ax.axvline(x=0, color='white', linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='white', linestyle=':',
               alpha=0.5)  # Add dotted line at y=0


def animate_sta_movie(
        fig,
        sta_data,
        stafit=None,
        frame_index=0,
        sta_width=None,
        sta_height=None,
        ax=None):
    """
    Animates the STA movie by showing individual frames.
    MODIFIED: Now optionally overlays the STAFit ellipse.
    """
    if ax is None:
        fig.clear()
        ax = fig.add_subplot(111)

    n_frames = sta_data.red.shape[2]
    if frame_index >= n_frames:
        frame_index = 0

    red_frame = sta_data.red[:, :, frame_index]
    green_frame = sta_data.green[:, :, frame_index]
    blue_frame = sta_data.blue[:, :, frame_index]

    sta_rgb = np.stack([red_frame, green_frame, blue_frame], axis=-1)

    min_val, max_val = np.min(sta_rgb), np.max(sta_rgb)
    if max_val != min_val:
        sta_rgb_normalized = (sta_rgb - min_val) / (max_val - min_val)
    else:
        sta_rgb_normalized = np.zeros_like(sta_rgb)

    extent = [
        0,
        sta_width,
        sta_height,
        0] if sta_width is not None else [
        0,
        red_frame.shape[1],
        red_frame.shape[0],
        0]

    ax.imshow(sta_rgb_normalized, origin='upper', extent=extent)

    # --- ADDED: Logic to draw the STAFit ellipse if provided ---
    if stafit:
        if sta_height is not None:
            adjusted_y = sta_height - stafit.center_y
        else:
            image_height = red_frame.shape[0]
            adjusted_y = image_height - stafit.center_y

        ellipse = Ellipse(
            xy=(stafit.center_x, adjusted_y),
            width=2 * stafit.std_x,
            height=2 * stafit.std_y,
            angle=np.rad2deg(stafit.rot),
            edgecolor='cyan',
            facecolor='none',
            lw=2
        )
        ax.add_patch(ellipse)

    ax.set_title(
        f"STA Movie - Frame {frame_index+1}/{n_frames}",
        color='white')
    ax.set_xlabel("X (stixels)", color='gray')
    ax.set_ylabel("Y (stixels)", color='gray')
    ax.set_facecolor('#1f1f1f')
    ax.tick_params(colors='gray')
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')

    fig.tight_layout()


def plot_temporal_filter_properties(fig, sta_data, stafit, vision_params, cell_id):
    """
    Plots the temporal filter with annotations for metrics.
    """
    fig.clear()
    ax = fig.add_subplot(111)

    time_axis, tc_matrix, _ = analysis_core.get_sta_timecourse_data(
        sta_data, stafit, vision_params, cell_id)

    if tc_matrix is None:
        ax.text(
            0.5,
            0.5,
            "No temporal data",
            ha='center',
            va='center',
            color='gray')
        return

    # Dominant channel
    energies = np.sum(tc_matrix**2, axis=0)
    dom_idx = np.argmax(energies)
    dom_trace = tc_matrix[:, dom_idx]
    dom_color = ['red', 'green', 'blue'][dom_idx]

    # Normalize
    abs_max = np.max(np.abs(dom_trace))
    if abs_max == 0:
        abs_max = 1
    norm_trace = dom_trace / abs_max

    ax.plot(
        time_axis,
        norm_trace,
        color=dom_color,
        linewidth=2,
        label='Filter')
    ax.fill_between(time_axis, norm_trace, 0, color=dom_color, alpha=0.1)

    # Find peak
    peak_idx = np.argmax(np.abs(norm_trace))
    peak_time = time_axis[peak_idx]
    peak_val = norm_trace[peak_idx]

    # Annotate Peak
    ax.scatter([peak_time], [peak_val], color='white', s=50, zorder=5)
    ax.annotate(f"Peak: {peak_time:.1f}ms",
                xy=(peak_time, peak_val),
                xytext=(0, 10 if peak_val > 0 else -15),
                textcoords='offset points',
                ha='center', color='white', fontsize=9)

    # FWHM
    try:
        is_off = peak_val < 0
        trace_for_width = -norm_trace if is_off else norm_trace
        widths, width_heights, left_ips, right_ips = peak_widths(
            trace_for_width, [peak_idx], rel_height=0.5
        )
        if len(widths) > 0:
            width = widths[0]
            # Interpolate time points
            sample_interval = abs(time_axis[1] - time_axis[0])
            start_time = time_axis[0] + left_ips[0] * sample_interval
            end_time = time_axis[0] + right_ips[0] * sample_interval

            h = width_heights[0]
            if is_off:
                h = -h

            ax.hlines(
                h,
                start_time,
                end_time,
                colors='yellow',
                linestyles='-',
                linewidth=2)
            ax.annotate(f"FWHM: {width * sample_interval:.1f}ms",
                        xy=((start_time + end_time) / 2, h),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', color='yellow', fontsize=8)
    except Exception:
        pass

    ax.set_title("Temporal Dynamics Analysis", color='white')
    ax.set_xlabel("Time (ms)", color='gray')
    ax.set_facecolor('#1f1f1f')
    ax.tick_params(colors='gray')
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
    ax.grid(True, alpha=0.2)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)

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
