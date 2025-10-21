import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, QTimer
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from analysis import analysis_core
import matplotlib.pyplot as plt

def draw_summary_EI_plot(main_window, cluster_ids: list):
    # Draws the main spatial analysis plot, switching between custom EI and Vision EI.
    cluster_ids = np.array(cluster_ids)
    # If scalar, add extra dimension for consistent processing
    if cluster_ids.ndim == 0:
        cluster_ids = np.array([cluster_ids])
    vision_cluster_id = cluster_ids + 1
    
    
    
    # Check if we have Vision EI data for any cluster
    has_vision_ei = main_window.data_manager.vision_eis and any(
        cid in main_window.data_manager.vision_eis for cid in vision_cluster_id
    )

    if has_vision_ei:
        if hasattr(main_window, 'ei_animation_timer') and main_window.ei_animation_timer and main_window.ei_animation_timer.isActive():
            main_window.ei_animation_timer.stop()
        draw_vision_ei_animation(main_window, cluster_ids)
        main_window.current_spatial_features = None
    else:
        # Fallback to original Kilosort EI-based spatial plot
        lightweight_features = main_window.data_manager.get_lightweight_features(cluster_ids)
        heavyweight_features = main_window.data_manager.get_heavyweight_features(cluster_ids)
        main_window.current_spatial_features = heavyweight_features
        if lightweight_features is None or heavyweight_features is None:
            main_window.summary_canvas.fig.clear()
            main_window.summary_canvas.fig.text(0.5, 0.5, "Error generating features.", ha='center', va='center', color='red')
            main_window.summary_canvas.draw()
            return
            
        main_window.summary_canvas.fig.clear()
        analysis_core.plot_rich_ei(
            main_window.summary_canvas.fig, lightweight_features['median_ei'], main_window.data_manager.channel_positions,
            heavyweight_features, main_window.data_manager.sampling_rate, pre_samples=20)
        main_window.summary_canvas.fig.suptitle(f"Cluster {cluster_ids} Spatial Analysis", color='white', fontsize=16)
        main_window.summary_canvas.draw()

def on_summary_plot_hover(main_window, event):
    # Handles hover events on the summary plot for tooltips.
    if (event.inaxes is None or main_window.data_manager is None or main_window.current_spatial_features is None):
        return
    if event.inaxes == main_window.summary_canvas.fig.axes[0]:
        positions = main_window.data_manager.channel_positions
        ptp_amps = main_window.current_spatial_features.get('ptp_amps')
        if ptp_amps is None:
            return
        mouse_pos = np.array([[event.xdata, event.ydata]])
        distances = cdist(mouse_pos, positions)[0]
        if distances.min() < 20:
            closest_idx = distances.argmin()
            ptp = ptp_amps[closest_idx]
            main_window.status_bar.showMessage(f"Channel ID {closest_idx}: PTP = {ptp:.2f} µV")

def update_waveform_plot(main_window, cluster_id, lightweight_features):
    # Updates the waveform plot with data for the selected cluster.
    main_window.waveform_plot.clear()
    main_window.waveform_plot.setTitle(f"Cluster {cluster_id} | Waveforms (Sampled)")
    median_ei = lightweight_features['median_ei']
    snippets = lightweight_features['raw_snippets']
    p2p = median_ei.max(axis=1) - median_ei.min(axis=1)
    dom_chan = np.argmax(p2p)
    pre_peak_samples = 20
    time_axis = (np.arange(median_ei.shape[1]) - pre_peak_samples) / main_window.data_manager.sampling_rate * 1000
    for i in range(snippets.shape[2]):
        main_window.waveform_plot.plot(time_axis, snippets[dom_chan, :, i], pen=pg.mkPen(color=(200, 200, 200, 30)))
    main_window.waveform_plot.plot(time_axis, median_ei[dom_chan], pen=pg.mkPen('#00A3E0', width=2.5))
    main_window.waveform_plot.setLabel('bottom', 'Time (ms)')
    main_window.waveform_plot.setLabel('left', 'Amplitude (uV)')
    main_window.waveform_plot.enableAutoRange(axis=pg.ViewBox.XYAxes)

def update_isi_plot(main_window, cluster_id):
    # Updates the ISI histogram plot.
    main_window.isi_plot.clear()
    violation_rate = main_window.data_manager._calculate_isi_violations(cluster_id)
    main_window.data_manager.update_cluster_isi(cluster_id, violation_rate)
    main_window.isi_plot.setTitle(f"Cluster {cluster_id} | ISI | Violations: {violation_rate:.2f}%")
    spikes = main_window.data_manager.get_cluster_spikes(cluster_id)
    if len(spikes) < 2: return
    isis_ms = np.diff(np.sort(spikes)) / main_window.data_manager.sampling_rate * 1000
    y, x = np.histogram(isis_ms, bins=np.linspace(0, 50, 101))
    main_window.isi_plot.plot(x, y, stepMode="center", fillLevel=0, brush=(0, 163, 224, 150))
    main_window.isi_plot.addLine(x=2.0, pen=pg.mkPen('r', style=Qt.PenStyle.DashLine, width=2))
    main_window.isi_plot.setLabel('bottom', 'ISI (ms)')
    main_window.isi_plot.setLabel('left', 'Count')

def update_fr_plot(main_window, cluster_id):
    # Updates the smoothed firing rate plot.
    main_window.fr_plot.clear()
    main_window.fr_plot.setTitle(f"Cluster {cluster_id} | Firing Rate")
    spikes_sec = main_window.data_manager.get_cluster_spikes(cluster_id) / main_window.data_manager.sampling_rate
    if len(spikes_sec) == 0: return
    total_duration = main_window.data_manager.spike_times.max() / main_window.data_manager.sampling_rate
    bins = np.arange(0, total_duration + 1, 1)
    counts, _ = np.histogram(spikes_sec, bins=bins)
    rate = gaussian_filter1d(counts.astype(float), sigma=5)
    main_window.fr_plot.plot(bins[:-1], rate, pen='y')
    main_window.fr_plot.setLabel('bottom', 'Time (s)')
    main_window.fr_plot.setLabel('left', 'Firing Rate (Hz)')

def draw_vision_ei_animation(main_window, cluster_ids: list):
    """Draws an animated visualization of the Vision EI, starting with a peak summary frame."""
    cluster_ids = np.array(cluster_ids)
    vision_cluster_id = cluster_ids + 1
    # if not main_window.data_manager.vision_eis or vision_cluster_id not in main_window.data_manager.vision_eis:
    #     main_window.summary_canvas.fig.clear()
    #     main_window.summary_canvas.fig.text(0.5, 0.5, "No Vision EI data available", ha='center', va='center', color='gray')
    #     main_window.summary_canvas.draw()
    #     main_window.ei_frame_slider.setEnabled(False)
    #     return

    ls_ei_data = []
    for cid in vision_cluster_id:
        if cid in main_window.data_manager.vision_eis:
            ls_ei_data.append(main_window.data_manager.vision_eis[cid].ei)
    # main_window.data_manager.vision_eis[vision_cluster_id].ei
    main_window.current_ei_data = ls_ei_data
    main_window.current_ei_cluster_ids = cluster_ids
    main_window.n_frames = ls_ei_data[0].shape[1]

    # --- CHANGE: Create and draw the peak summary frame initially ---
    ls_summary_frame_data = []
    for ei_data in ls_ei_data:
        sf = _create_peak_summary_frame(ei_data)
        ls_summary_frame_data.append(sf)

    # Update slider properties but don't set a value, as the summary is not a real frame
    main_window.ei_frame_slider.setMinimum(0)
    main_window.ei_frame_slider.setMaximum(main_window.n_frames - 1)
    main_window.ei_frame_slider.setValue(0) # Default slider to start
    main_window.ei_frame_label.setText(f"Frame: Peak Summary")
    main_window.ei_frame_slider.setEnabled(True)
    
    # Draw the summary frame, passing a special index (-1) to identify it
    draw_vision_ei_frame(main_window, ls_summary_frame_data, -1, main_window.n_frames)

def reshape_ei(ei: np.ndarray, sorted_electrodes: np.ndarray,
               n_rows: int=16) -> np.ndarray:
    """
    Reshape the EI matrix from 512 x 201 to 16 x 32 x 201 based on electrode locations.

    Parameters:
    ei (numpy.ndarray): The EI matrix of shape (electrode, frames).
    sorted_electrodes (numpy.ndarray): The sorted indices of the electrodes.
    n_rows (int): The number of rows to reshape the EI matrix into. Default is 16.

    Returns:
    numpy.ndarray: The reshaped EI matrix of shape (16, 32, 201).
    """
    if ei.shape[0] != 512:
        print(f'Warning: Expected EI shape (512, 201), got {ei.shape}')
    n_electrodes = ei.shape[0]
    n_frames = ei.shape[1]
    n_cols = n_electrodes // n_rows  # Assuming 512 electrodes and 16 rows

    if n_cols * n_rows != n_electrodes:
        raise ValueError(f"Number of electrodes {n_electrodes} is not compatible with {n_rows} rows and {n_cols} columns.")

    sorted_ei = ei[sorted_electrodes]

    # Reshape the sorted EI matrix
    reshaped_ei = sorted_ei.reshape(n_rows, n_cols, n_frames)

    return reshaped_ei

def draw_vision_ei_frame(main_window, ls_summary, frame_index, total_frames):
    """Draws a single EI frame for each cluster as a subplot."""
    n_clusters = len(ls_summary)
    main_window.summary_canvas.fig.clear()
    axes = main_window.summary_canvas.fig.subplots(nrows=1, ncols=n_clusters, squeeze=False)[0]


    for i, frame_data in enumerate(ls_summary):
        ax = axes[i]
        if frame_index == -1:
            ax.set_title("Vision EI - Peak Summary")
            ei_map = reshape_ei(
                frame_data[:, np.newaxis],
                main_window.data_manager.sorted_channels
            )
            ei_map = np.log10(ei_map + 1e-6)
            im = ax.imshow(ei_map, cmap='hot', aspect='auto', origin='lower')
            main_window.summary_canvas.fig.colorbar(im, ax=ax, label='Log10 Amplitude (µV)')
        else:
            full_ei = main_window.current_ei_data[i]
            amplitudes = np.abs(full_ei)
            sizes = amplitudes
            scaled_colors = frame_data
            vmax_global = np.max(np.abs(frame_data))
            vmin_global = -vmax_global
            scatter = ax.scatter(
                main_window.data_manager.channel_positions[:, 0],
                main_window.data_manager.channel_positions[:, 1],
                c=scaled_colors, 
                s=sizes,
                cmap='RdBu_r',
                edgecolor='white', 
                linewidth=0.5,
                # vmin=-1,
                # vmax=1
                vmin=vmin_global,
                vmax=vmax_global
            )
            ax.set_title(f"Vision EI - Frame {frame_index + 1}/{total_frames}")
        
            norm = plt.Normalize(vmin=vmin_global, vmax=vmax_global)
            sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
            sm.set_array([])
            cbar = main_window.summary_canvas.fig.colorbar(sm, ax=ax)
            
            cbar.set_label('Amplitude (µV)', color='white')
            cbar.ax.yaxis.set_tick_params(color='white')
            cbar.outline.set_edgecolor('#444444')
            for tick_label in cbar.ax.yaxis.get_ticklabels():
                tick_label.set_color('white')
    
    main_window.summary_canvas.draw()

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

def update_ei_frame(main_window):
    """Updates the EI visualization to the next frame in the animation."""
    if main_window.current_frame >= main_window.n_frames - 1:
        main_window.current_frame = 0
    else:
        main_window.current_frame += 1
    
    frame_data = main_window.current_ei_data[:, main_window.current_frame]
    
    # Use the same drawing function for consistency
    draw_vision_ei_frame(main_window, frame_data, main_window.current_frame, main_window.n_frames)

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
    print(f"--- 2. (Plotting): Received selected_cell_id = {selected_cell_id}. Passing to analysis_core. ---")
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

def _create_peak_summary_frame(ei_data):
    """
    Creates a composite frame showing the peak absolute amplitude for each channel.
    
    Args:
        ei_data (np.ndarray): The full EI data (n_channels, n_frames).
        
    Returns:
        np.ndarray: A 1D array (n_channels,) with the peak value for each channel.
    """
    # Find the index of the max absolute value along the time axis (axis=1) for each channel
    # peak_indices = np.argmax(np.abs(ei_data), axis=1)
    
    # Use these indices to pull the corresponding peak value (maintaining the sign) from the data
    # This uses advanced numpy indexing to get ei_data[channel, peak_index_for_that_channel]
    # summary_frame = ei_data[np.arange(ei_data.shape[0]), peak_indices]

    # return summary_frame

    return np.max(np.abs(ei_data), axis=1)

