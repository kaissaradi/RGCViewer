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

def draw_sta_plot(main_window, cluster_id):
    """
    MODIFIED: Fetches STAFit data and passes it to the plotting function.
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

        main_window.sta_canvas.fig.clear()
        analysis_core.animate_sta_movie(
            main_window.sta_canvas.fig,
            sta_data,
            stafit=stafit, # <-- Pass the fit to the plotting function
            frame_index=peak_frame_index,
            sta_width=main_window.data_manager.vision_sta_width,
            sta_height=main_window.data_manager.vision_sta_height
        )
        main_window.sta_canvas.draw()
    else:
        # No Vision STA data available
        main_window.sta_canvas.fig.clear()
        main_window.sta_canvas.fig.text(0.5, 0.5, "No Vision STA data available", ha='center', va='center', color='gray')
        main_window.sta_canvas.draw()
        main_window.sta_frame_slider.setEnabled(False)



def draw_population_rfs_plot(main_window, selected_cell_id=None):
    """Draws the population receptive field plot showing all cell RFs."""
    logger.debug("Received selected_cell_id=%s for population RF plot", selected_cell_id)
    # MODIFIED: This function now accepts 'selected_cell_id'
    has_vision_params = main_window.data_manager.vision_params

    if has_vision_params:
        main_window.sta_canvas.fig.clear()

        analysis_core.plot_population_rfs(
            main_window.sta_canvas.fig,
            main_window.data_manager.vision_params,
            sta_width=main_window.data_manager.vision_sta_width,
            sta_height=main_window.data_manager.vision_sta_height,
            selected_cell_id=selected_cell_id # Pass the ID along to the core plotting function
        )
        main_window.sta_canvas.draw()
    else:
        main_window.sta_canvas.fig.clear()
        main_window.sta_canvas.fig.text(0.5, 0.5, "No Vision parameters available",
                                       ha='center', va='center', color='gray')
        main_window.sta_canvas.draw()

def draw_sta_timecourse_plot(main_window, cluster_id):
    # Draws the STA timecourse plot for a specific cell.
    vision_cluster_id = cluster_id + 1
    has_vision_sta = main_window.data_manager.vision_stas and vision_cluster_id in main_window.data_manager.vision_stas
    if has_vision_sta:
        sta_data = main_window.data_manager.vision_stas[vision_cluster_id]
        stafit = main_window.data_manager.vision_params.get_stafit_for_cell(vision_cluster_id)
        main_window.sta_canvas.fig.clear()
        analysis_core.plot_sta_timecourse(
            main_window.sta_canvas.fig,
            sta_data,
            stafit,
            main_window.data_manager.vision_params,
            vision_cluster_id
        )
        main_window.sta_canvas.draw()
    else:
        main_window.sta_canvas.fig.clear()
        main_window.sta_canvas.fig.text(0.5, 0.5, "No Vision STA data available", ha='center', va='center', color='gray')
        main_window.sta_canvas.draw()

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

        if main_window.sta_animation_timer is None:
            main_window.sta_animation_timer = QTimer()
            main_window.sta_animation_timer.timeout.connect(lambda: update_sta_frame(main_window))

        if not main_window.sta_animation_timer.isActive():
            main_window.sta_animation_timer.start(100)
    else:
        main_window.sta_canvas.fig.clear()
        main_window.sta_canvas.fig.text(0.5, 0.5, "No Vision STA data available", ha='center', va='center', color='gray')
        main_window.sta_canvas.draw()
        main_window.sta_frame_slider.setEnabled(False)

def update_sta_frame(main_window):
    # Updates the STA visualization to the next frame in the animation.
    if not hasattr(main_window, 'current_sta_data') or main_window.current_sta_data is None:
        return

    main_window.current_frame_index = (main_window.current_frame_index + 1) % main_window.total_sta_frames
    main_window.sta_frame_slider.setValue(main_window.current_frame_index)
    main_window.sta_frame_label.setText(f"Frame: {main_window.current_frame_index + 1}/{main_window.total_sta_frames}")

    main_window.sta_canvas.fig.clear()
    analysis_core.animate_sta_movie(
        main_window.sta_canvas.fig,
        main_window.current_sta_data,
        stafit=main_window.current_stafit, # <-- Pass the stored fit during animation
        frame_index=main_window.current_frame_index,
        sta_width=main_window.data_manager.vision_sta_width,
        sta_height=main_window.data_manager.vision_sta_height
    )
    main_window.sta_canvas.draw()

def stop_sta_animation(main_window):
    # Stops the STA animation if running.
    if hasattr(main_window, 'sta_animation_timer') and main_window.sta_animation_timer and main_window.sta_animation_timer.isActive():
        main_window.sta_animation_timer.stop()

