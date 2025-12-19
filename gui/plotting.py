import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, QTimer
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from analysis import analysis_core
import matplotlib
import logging
# Set matplotlib logging level to WARNING to suppress font debug messages
matplotlib_logger = logging.getLogger('matplotlib.font_manager')
matplotlib_logger.setLevel(logging.WARNING)
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)


# Note: The update_raw_trace_plot function has been replaced by background loading
# The functionality is now handled by MainWindow.load_raw_trace_data and related worker
# This stub is maintained for compatibility but does nothing
def update_raw_trace_plot(main_window, cluster_id):
    """
    Draw the raw trace plot with the nearest 3 channels and spike templates overlaid.
    This function has been replaced by background loading for improved performance.
    """
    # Background loading is now handled by MainWindow.load_raw_trace_data
    # This stub is maintained for compatibility but does nothing
    pass

# In gui/plotting.py

# In plotting.py, update the draw_sta_metrics function:

def draw_sta_metrics(main_window, cluster_id):
    """
    Calculates and displays STA metrics in the text box with improved organization.
    """
    vision_cluster_id = cluster_id + 1
    has_vision_sta = main_window.data_manager.vision_stas and vision_cluster_id in main_window.data_manager.vision_stas
    
    if has_vision_sta:
        sta_data = main_window.data_manager.vision_stas[vision_cluster_id]
        stafit = main_window.data_manager.vision_params.get_stafit_for_cell(vision_cluster_id)
        
        metrics = analysis_core.compute_sta_metrics(
            sta_data, stafit, main_window.data_manager.vision_params, vision_cluster_id
        )
        
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
        temporal_keys = ["Dominant Channel", "Polarity", "Time to Peak (ms)", 
                        "Response Duration (ms)", "Zero Crossing (ms)", 
                        "FWHM (Duration)", "Biphasic Index", "SNR (std ratio)"]
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
        spatial_keys = ["RF Center X", "RF Center Y", "RF Sigma X", "RF Sigma Y", 
                       "Orientation", "RF Area (sq stix)", "RF Ellipticity (σy/σx)", 
                       "RF Elongation"]
        for k in spatial_keys:
            if k in metrics:
                html += f"<tr><td class='metric-name'>{k}</td><td class='metric-value'>{metrics[k]}</td></tr>"
        
        # Spatial Asymmetry
        html += "<tr><th colspan='2' class='subsection'>Spatial Asymmetry</th></tr>"
        asymmetry_keys = ["Spatial Peak", "Spatial Trough", "Peak/Trough Ratio", "Spatial Skewness"]
        for k in asymmetry_keys:
            if k in metrics:
                html += f"<tr><td class='metric-name'>{k}</td><td class='metric-value'>{metrics[k]}</td></tr>"
        
        html += "</table>"
        
        main_window.sta_metrics_text.setHtml(html)
    else:
        main_window.sta_metrics_text.setHtml("<div style='color:gray; text-align:center; padding:20px;'>No STA Data Available</div>")

def draw_temporal_filter_plot(main_window, cluster_id):
    """
    Draws the detailed temporal filter analysis plot.
    """
    vision_cluster_id = cluster_id + 1
    has_vision_sta = main_window.data_manager.vision_stas and vision_cluster_id in main_window.data_manager.vision_stas

    if has_vision_sta:
        sta_data = main_window.data_manager.vision_stas[vision_cluster_id]
        stafit = main_window.data_manager.vision_params.get_stafit_for_cell(vision_cluster_id)

        analysis_core.plot_temporal_filter_properties(
            main_window.temporal_filter_canvas.fig,
            sta_data, stafit, main_window.data_manager.vision_params, vision_cluster_id
        )
        main_window.temporal_filter_canvas.draw()
    else:
        main_window.temporal_filter_canvas.fig.clear()
        main_window.temporal_filter_canvas.fig.text(0.5, 0.5, "No Data", ha='center', va='center', color='gray')
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
        stafit = main_window.data_manager.vision_params.get_stafit_for_cell(vision_cluster_id)
        main_window.current_sta_data = sta_data
        main_window.current_stafit = stafit # <-- Store the fit
        main_window.current_sta_cluster_id = vision_cluster_id

        n_frames = sta_data.red.shape[2]
        main_window.total_sta_frames = n_frames
        main_window.sta_frame_slider.setMaximum(n_frames - 1)

        all_channels = np.stack([sta_data.red, sta_data.green, sta_data.blue], axis=0)
        frame_energies = np.max(np.abs(all_channels), axis=(0, 1, 2))
        peak_frame_index = np.argmax(frame_energies)
        main_window.current_frame_index = peak_frame_index

        main_window.sta_frame_slider.setValue(peak_frame_index)
        main_window.sta_frame_label.setText(f"Frame: {peak_frame_index + 1}/{n_frames}")
        main_window.sta_frame_slider.setEnabled(True)

        # Use the RF canvas instead of the old sta_canvas
        main_window.rf_canvas.fig.clear()
        analysis_core.animate_sta_movie(
            main_window.rf_canvas.fig,
            sta_data,
            stafit=stafit, # <-- Pass the fit to the plotting function
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
        main_window.rf_canvas.fig.text(0.5, 0.5, "No Vision STA data available", ha='center', va='center', color='gray')
        main_window.rf_canvas.draw()
        main_window.sta_frame_slider.setEnabled(False)

        # Clear other panels
        main_window.sta_metrics_text.clear()
        main_window.temporal_filter_canvas.fig.clear()
        main_window.temporal_filter_canvas.draw()



def draw_population_rfs_plot(main_window, selected_cell_id=None, subset_cell_ids=None, canvas=None):
    """Draws the population receptive field plot showing all cell RFs."""
    logger.debug("Received selected_cell_id=%s, subset_cell_ids=%s for population RF plot", selected_cell_id, subset_cell_ids)
    
    # 1. Determine target canvas
    if canvas is None:
        # If population view is enabled, prefer the dedicated mosaic canvas
        # unless specifically overridden or if we are in a mode that implies the main view
        if hasattr(main_window, 'population_view_enabled') and main_window.population_view_enabled:
            canvas = getattr(main_window, 'pop_mosaic_canvas', main_window.rf_canvas)
        else:
            canvas = main_window.rf_canvas

    # 2. Smart Group Detection (if single unit selected but no subset provided)
    # This ensures that when we click a unit in Split View, the Context pane shows its group.
    if selected_cell_id is not None and subset_cell_ids is None:
         # Only do auto-group detection if we are targeting the population canvas
         # (If we are in "Population RFs" main view mode, maybe we want to see ALL cells?)
         # Let's assume for Split View we want the group context.
         if hasattr(main_window, 'population_view_enabled') and main_window.population_view_enabled:
             df = main_window.data_manager.cluster_df
             if not df.empty and 'cluster_id' in df.columns:
                 if selected_cell_id in df['cluster_id'].values:
                     # Find the group/label
                     try:
                         row = df[df['cluster_id'] == selected_cell_id].iloc[0]
                         # Try 'KSLabel' first, then 'group' if available (custom groups)
                         group_label = row.get('KSLabel') 
                         
                         if group_label:
                             subset_cell_ids = df[df['KSLabel'] == group_label]['cluster_id'].tolist()
                     except Exception as e:
                         logger.warning(f"Error determining group for cluster {selected_cell_id}: {e}")

    has_vision_params = main_window.data_manager.vision_params

    if has_vision_params:
        canvas.fig.clear()

        analysis_core.plot_population_rfs(
            canvas.fig,
            main_window.data_manager.vision_params,
            sta_width=main_window.data_manager.vision_sta_width,
            sta_height=main_window.data_manager.vision_sta_height,
            selected_cell_id=selected_cell_id, # Pass the ID along to the core plotting function
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
        stafit = main_window.data_manager.vision_params.get_stafit_for_cell(vision_cluster_id)
        main_window.timecourse_canvas.fig.clear()
        analysis_core.plot_sta_timecourse(
            main_window.timecourse_canvas.fig,
            sta_data,
            stafit,
            main_window.data_manager.vision_params,
            vision_cluster_id
        )
        main_window.timecourse_canvas.draw()
    else:
        main_window.timecourse_canvas.fig.clear()
        main_window.timecourse_canvas.fig.text(0.5, 0.5, "No Vision STA data available", ha='center', va='center', color='gray')
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
            main_window.sta_animation_timer.timeout.connect(main_window._advance_frame_internal)

        # Stop any currently running animation first to prevent conflicts
        if main_window.sta_animation_timer and main_window.sta_animation_timer.isActive():
            main_window.sta_animation_timer.stop()
        else:
            # Start the animation only if it's not already running
            main_window.sta_animation_timer.start(100)

        # Update the RF canvas with the first frame
        main_window.rf_canvas.fig.clear()
        analysis_core.animate_sta_movie(
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
        main_window.rf_canvas.fig.text(0.5, 0.5, "No Vision STA data available", ha='center', va='center', color='gray')
        main_window.rf_canvas.draw()
        main_window.sta_frame_slider.setEnabled(False)


def update_sta_frame(main_window):
    # Updates the STA visualization to the next frame in the animation.
    # This is for external calls (manual frame advance) and will stop the timer.
    if not hasattr(main_window, 'current_sta_data') or main_window.current_sta_data is None:
        # Stop the timer if there's no data to animate
        stop_sta_animation(main_window)
        return

    main_window.current_frame_index = (main_window.current_frame_index + 1) % main_window.total_sta_frames
    main_window.sta_frame_slider.setValue(main_window.current_frame_index)
    main_window.sta_frame_label.setText(f"Frame: {main_window.current_frame_index + 1}/{main_window.total_sta_frames}")

    main_window.rf_canvas.fig.clear()
    analysis_core.animate_sta_movie(
        main_window.rf_canvas.fig,
        main_window.current_sta_data,
        stafit=main_window.current_stafit, # <-- Pass the stored fit during animation
        frame_index=main_window.current_frame_index,
        sta_width=main_window.data_manager.vision_sta_width,
        sta_height=main_window.data_manager.vision_sta_height
    )
    main_window.rf_canvas.draw()

def stop_sta_animation(main_window):
    # Stops the STA animation if running.
    if hasattr(main_window, 'sta_animation_timer') and main_window.sta_animation_timer and main_window.sta_animation_timer.isActive():
        main_window.sta_animation_timer.stop()
