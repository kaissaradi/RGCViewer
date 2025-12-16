# --- Standard Library Imports ---
import os
from pathlib import Path
from typing import Tuple

# --- Third-Party Imports ---
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
import networkx as nx
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.interpolate import griddata

# --- Interactive Plotting Imports ---
import ipywidgets as widgets
from ipywidgets import HBox
from IPython.display import display

def compute_per_spike_features(
    snippets: np.ndarray,
    channel_positions: np.ndarray,
    n_pcs: int = 3
) -> np.ndarray:
    """
    Computes waveform and spatial features for each spike in a vectorized manner.
    """
    n_channels, n_samples, n_spikes = snippets.shape
    waveforms_flat = snippets.reshape(n_channels * n_samples, n_spikes).T

    pca = PCA(n_components=n_pcs)
    waveform_features = pca.fit_transform(waveforms_flat)

    ptp_amplitudes = np.ptp(snippets, axis=1)
    sum_of_masses = np.sum(ptp_amplitudes, axis=0)
    sum_of_masses[sum_of_masses == 0] = 1e-9

    weighted_positions = ptp_amplitudes.T @ channel_positions
    spatial_features = weighted_positions / sum_of_masses[:, np.newaxis]

    all_features = np.hstack((waveform_features, spatial_features))
    return all_features

def extract_snippets(dat_path, spike_times, window=(-20, 60), n_channels=512, dtype='int16'):
    """
    Extracts snippets of raw data using memory-mapping for high efficiency.
    """
    snip_len = window[1] - window[0]
    spike_count = len(spike_times)
    
    if spike_count == 0:
        return np.zeros((n_channels, snip_len, 0), dtype=np.float32)

    raw_data = np.memmap(dat_path, dtype=dtype, mode='r').reshape(-1, n_channels)
    total_samples = raw_data.shape[0]

    snips = np.zeros((spike_count, n_channels, snip_len), dtype=np.float32)

    for i, spike_time in enumerate(spike_times):
        start_sample = spike_time.astype(np.int64) + window[0]
        end_sample = start_sample + snip_len

        if start_sample >= 0 and end_sample < total_samples:
            snippet = raw_data[start_sample:end_sample, :]
            snips[i, :, :] = snippet.T
            
    return snips.transpose(1, 2, 0)

def compare_eis(eis, ei_template=None, max_lag=3):
    """
    Compare a list of EIs to each other or to a template.
    """
    k = len(eis)
    if ei_template is not None:
        sim = np.zeros((k, 1))
        for i in range(k):
            ei_i = eis[i]
            dom_chan = np.argmax(np.max(np.abs(ei_i), axis=1))
            trace_i = ei_i[dom_chan, :]
            trace_t = ei_template[dom_chan, :]

            lags = np.arange(-max_lag, max_lag + 1)
            xc = correlate(trace_i, trace_t, mode='full', method='auto')
            center = len(xc) // 2
            xc_window = xc[center - max_lag:center + max_lag + 1]
            shift = lags[np.argmax(xc_window)]

            aligned_t = np.roll(ei_template, shift, axis=1)
            sim[i] = np.dot(ei_i.flatten(), aligned_t.flatten()) / (
                np.linalg.norm(ei_i) * np.linalg.norm(aligned_t))
        return sim

    else:
        sim = np.zeros((k, k))
        for i in range(k):
            ei_i = eis[i]
            dom_chan = np.argmax(np.max(np.abs(ei_i), axis=1))
            trace_i = ei_i[dom_chan, :]

            for j in range(i, k):
                ei_j = eis[j]
                trace_j = ei_j[dom_chan, :]

                lags = np.arange(-max_lag, max_lag + 1)
                xc = correlate(trace_i, trace_j, mode='full', method='auto')
                center = len(xc) // 2
                xc_window = xc[center - max_lag:center + max_lag + 1]
                shift = lags[np.argmax(xc_window)]

                aligned_j = np.roll(ei_j, shift, axis=1)
                val = np.dot(ei_i.flatten(), aligned_j.flatten()) / (
                    np.linalg.norm(ei_i) * np.linalg.norm(aligned_j))
                sim[i, j] = val
                sim[j, i] = val
        return sim

def baseline_correct(snips, pre_samples=20):
    """Corrects for baseline shifts in an EI."""
    if snips.ndim == 3:
        baseline = snips[:, :pre_samples, :].mean(axis=1)
        return snips - baseline[:, np.newaxis, :]
    else:
        return snips - snips[:, :pre_samples].mean(axis=1, keepdims=True)

def compute_ei(snips, pre_samples=20):
    """Computes the Electrical Image (median waveform) from snippets for robustness."""
    snips = baseline_correct(snips, pre_samples=pre_samples)
    snips_torch = torch.from_numpy(snips)
    # Use median for robustness to outlier spikes
    ei = torch.median(snips_torch, dim=2).values.numpy()
    return ei

def select_channels(ei, min_chan=30, max_chan=80, threshold=15):
    """Selects the most significant channels from an EI based on peak-to-peak amplitude."""
    p2p = ei.max(axis=1) - ei.min(axis=1)
    selected = np.where(p2p > threshold)[0]
    if len(selected) > max_chan:
        selected = np.argsort(p2p)[-max_chan:]
    elif len(selected) < min_chan and len(p2p) > min_chan:
        selected = np.argsort(p2p)[-min_chan:]
    return np.sort(selected)

def find_merge_groups(sim, threshold):
    """Finds groups of clusters to merge based on a similarity matrix."""
    G = nx.Graph()
    k = sim.shape[0]
    G.add_nodes_from(range(k))
    for i in range(k):
        for j in range(i + 1, k):
            if sim[i, j] > threshold:
                G.add_edge(i, j)
    return list(nx.connected_components(G))

def calculate_isi_violations(spike_times_samples, sampling_rate, refractory_period_ms=2.0):
    """
    Calculates the rate of refractory period violations for a spike train.
    """
    if len(spike_times_samples) < 2:
        return 0.0
    refractory_period_samples = (refractory_period_ms / 1000.0) * sampling_rate
    sorted_spikes = np.sort(spike_times_samples)
    isis_samples = np.diff(sorted_spikes)
    violation_count = np.sum(isis_samples < refractory_period_samples)
    violation_rate = violation_count / len(isis_samples)
    return violation_rate

def refine_cluster_v2(spike_times, dat_path, channel_positions, params):
    """
    Recursively refines neural spike clusters using PCA+KMeans clustering.
    """
    print(f"\n=== Starting Cluster Refinement ===")
    print(f"Input spikes: {len(spike_times)}")
    
    window = params.get('window', (-20, 60))
    min_spikes = params.get('min_spikes', 500)
    k_start = params.get('k_start', 3)
    k_refine = params.get('k_refine', 2)
    ei_sim_threshold = params.get('ei_sim_threshold', 0.9)
    max_depth = params.get('max_depth', 10)
    
    if isinstance(dat_path, np.ndarray):
        snips = dat_path
    elif isinstance(dat_path, str):
        snips = extract_snippets(dat_path, spike_times, window)
    else:
        return []

    full_inds = np.arange(snips.shape[2])
    cluster_pool = [{'inds': full_inds, 'depth': 0}]
    final_clusters = []

    while cluster_pool:
        cl = cluster_pool.pop(0)
        inds = cl['inds']
        depth = cl['depth']
        
        if depth >= max_depth:
            final_clusters.append({'inds': inds})
            continue

        if len(inds) < min_spikes:
            continue

        k = k_start if depth == 0 else k_refine
        
        snips_cl = snips[:, :, inds]
        ei = compute_ei(snips_cl)
        selected = select_channels(ei)
        snips_sel = snips[np.ix_(selected, np.arange(snips.shape[1]), inds)]
        snips_centered = snips_sel - snips_sel.mean(axis=1, keepdims=True)
        flat = snips_centered.transpose(2, 0, 1).reshape(len(inds), -1)
        pcs = PCA(n_components=5).fit_transform(flat)
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(pcs)

        subclusters = [{'inds': inds[np.where(labels == i)[0]]} for i in range(k)]
        
        large_subclusters = [sc for sc in subclusters if len(sc['inds']) >= min_spikes]
        
        if len(large_subclusters) <= 1:
            final_clusters.append({'inds': inds})
            continue

        eis_large = [compute_ei(snips[:, :, sc['inds']]) for sc in large_subclusters]
        sim = compare_eis(eis_large)
        groups = find_merge_groups(sim, ei_sim_threshold)

        for group in groups:
            all_inds = np.concatenate([large_subclusters[j]['inds'] for j in group])
            if len(all_inds) >= min_spikes:
                cluster_pool.append({'inds': all_inds, 'depth': depth + 1})

    print(f"\n=== Refinement Complete: {len(final_clusters)} final clusters ===")
    return final_clusters

def _spatial_smooth(values, positions, sigma=30):
    """Spatially smooth values based on channel positions."""
    smoothed = np.zeros_like(values)
    for i in range(len(values)):
        distances = np.sqrt(np.sum((positions - positions[i])**2, axis=1))
        weights = np.exp(-distances**2 / (2 * sigma**2))
        smoothed[i] = np.sum(values * weights) / np.sum(weights)
    return smoothed

def compute_spatial_features(ei, channel_positions, sampling_rate=20000, pre_samples=40):
    """
    Computes rich spatial features for the EI.
    """
    peak_negative = ei.min(axis=1)
    peak_times = ei.argmin(axis=1)
    peak_times_ms = (peak_times - pre_samples) / sampling_rate * 1000
    peak_negative_smooth = _spatial_smooth(peak_negative, channel_positions)
    amplitude_threshold = np.percentile(np.abs(peak_negative), 80)
    active_channels = np.abs(peak_negative) > amplitude_threshold
    ptp_amps = np.ptp(ei, axis=1)

    grid_x, grid_y = np.mgrid[
        channel_positions[:, 0].min():channel_positions[:, 0].max():200j,
        channel_positions[:, 1].min():channel_positions[:, 1].max():200j
    ]
    grid_z = griddata(channel_positions, peak_negative_smooth, (grid_x, grid_y), method='cubic')

    return {
        'peak_negative_smooth': peak_negative_smooth,
        'peak_times_ms': peak_times_ms,
        'active_channels': active_channels,
        'ptp_amps': ptp_amps,
        'grid_x': grid_x,
        'grid_y': grid_y,
        'grid_z': grid_z,
    }

# =============================================================================
# Plotting Functions
# =============================================================================

# In analysis_core.py

def plot_rich_ei(fig, median_ei, channel_positions, spatial_features, sampling_rate=20000, pre_samples=20):
    """
    Creates a rich, multi-panel EI visualization with thresholding for clarity.
    """
    fig.clear()
    peak_negative_smooth = spatial_features['peak_negative_smooth']
    ptp_amps = spatial_features['ptp_amps']

    # --- DIAGNOSTIC PRINT ---
    print("\n--- Spatial EI Diagnosis ---")
    print("Peak-to-Peak Amplitudes by Channel:")
    print(ptp_amps)
    print("--------------------------\n")

    grid_x, grid_y, grid_z = spatial_features['grid_x'], spatial_features['grid_y'], spatial_features['grid_z']
    peak_times_ms = spatial_features['peak_times_ms']
    active_channels = spatial_features['active_channels']

    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], width_ratios=[1, 1], hspace=0.4, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    ax1.set_title('Spatial Amplitude', color='white')
    v_min, v_max = np.percentile(peak_negative_smooth, [5, 95])
    contour_fill = ax1.contourf(grid_x, grid_y, grid_z, levels=20, cmap='RdBu_r', alpha=0.7, vmin=v_min, vmax=v_max)
    ax1.contour(grid_x, grid_y, grid_z, levels=20, colors='white', linewidths=0.5, alpha=0.4)
    
    # --- CHANGE START: Thresholding and scaling for scatter plot ---
    # Set a threshold to only show channels with significant peak-to-peak amplitude
    ptp_threshold = np.percentile(ptp_amps, 80) # Only show top 20% of channels
    active_mask = ptp_amps > ptp_threshold
    
    # Create an array for dot sizes, defaulting to 0 (invisible)
    sizes = np.zeros_like(ptp_amps)
    
    # Calculate sizes only for the active channels
    if np.any(active_mask):
        max_active_ptp = ptp_amps[active_mask].max()
        # Scale sizes for active channels, adding a small base size
        sizes[active_mask] = 15 + (ptp_amps[active_mask] / max_active_ptp) * 200

    ax1.scatter(channel_positions[:, 0], channel_positions[:, 1], s=sizes, c=peak_negative_smooth,
                cmap='RdBu_r', edgecolor='black', linewidth=0.7, zorder=2, vmin=v_min, vmax=v_max)
    # --- CHANGE END ---
    
    fig.colorbar(contour_fill, ax=ax1, label='Smoothed Peak Amp (µV)', shrink=0.8)

    ax2.set_title('Spike Propagation', color='white')
    scatter2 = ax2.scatter(channel_positions[active_channels, 0], channel_positions[active_channels, 1],
                           c=peak_times_ms[active_channels], cmap='viridis', s=80, edgecolor='white', linewidth=0.5)
    fig.colorbar(scatter2, ax=ax2, label='Time to Peak (ms)', shrink=0.8)

    ax3.set_title('Waveform Heatmap', color='white')
    time_axis_ms = (np.arange(median_ei.shape[1]) - pre_samples) / sampling_rate * 1000
    if active_channels.sum() > 0:
        active_idx = np.where(active_channels)[0]
        sorted_channel_idx = active_idx[np.argsort(median_ei.argmin(axis=1)[active_idx])]
        waveform_matrix = median_ei[sorted_channel_idx]
        im = ax3.imshow(waveform_matrix, aspect='auto', cmap='RdBu_r',
                        vmin=-np.percentile(np.abs(waveform_matrix), 98),
                        vmax=np.percentile(np.abs(waveform_matrix), 98),
                        extent=[time_axis_ms[0], time_axis_ms[-1], len(sorted_channel_idx), 0])
        ax3.axvline(0, color='black', linestyle='--', alpha=0.8)
        fig.colorbar(im, ax=ax3, label='Amplitude (µV)', shrink=0.8, orientation='horizontal', pad=0.25)

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('#1f1f1f')
        ax.tick_params(colors='gray')
        for spine in ax.spines.values(): spine.set_edgecolor('gray')
    ax1.axis('equal')
    ax2.axis('equal')


def plot_population_rfs(fig, vision_params, sta_width=None, sta_height=None, selected_cell_id=None):
    """
    Visualizes the receptive fields of all cells, highlighting the selected cell
    by filling its true ellipse shape and making other ellipses more faint.
    """
    fig.clear()
    ax = fig.add_subplot(111)
    
    cell_ids = vision_params.get_cell_ids()
    
    if not cell_ids:
        ax.text(0.5, 0.5, "No RF data available", ha='center', va='center', color='gray')
        ax.set_title("Population Receptive Fields", color='white')
        return

    vision_cell_id_selected = selected_cell_id + 1 if selected_cell_id is not None else None
    
    # --- Auto-determine plot boundaries from data ---
    x_coords, y_coords = [], []
    for cell_id in cell_ids:
        try:
            stafit = vision_params.get_stafit_for_cell(cell_id)
            x_coords.append(stafit.center_x)
            y_coords.append(stafit.center_y)
        except Exception:
            continue
    
    x_range = (min(x_coords) - 20, max(x_coords) + 20) if x_coords else (0, 100)
    y_range = (min(y_coords) - 20, max(y_coords) + 20) if y_coords else (0, 100)
    
    # --- STAGE 1: Draw all non-selected ellipses first to make them faint background ---
    for cell_id in cell_ids:
        # Skip the selected cell for now; we'll draw it separately.
        if cell_id == vision_cell_id_selected:
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
                alpha=0.2  # <-- CHANGE: Made even more faint (more transparent)
            )
            ax.add_patch(ellipse)
        except Exception:
            continue

    # --- STAGE 2: Draw the single, highlighted ellipse on top of everything else ---
    if vision_cell_id_selected is not None:
        try:
            stafit = vision_params.get_stafit_for_cell(vision_cell_id_selected)
            adjusted_y = sta_height - stafit.center_y if sta_height is not None else stafit.center_y

            # This now correctly uses the selected cell's own parameters for the highlight
            highlight_ellipse = Ellipse(
                xy=(stafit.center_x, adjusted_y),
                width=2 * stafit.std_x,
                height=2 * stafit.std_y,
                angle=np.rad2deg(stafit.rot),
                edgecolor='red',
                facecolor=(1.0, 0.0, 0.0, 0.4), # Filled with semi-transparent red
                lw=1.5, # A slightly thicker line to stand out
                zorder=10 # Ensure it's drawn on top
            )
            ax.add_patch(highlight_ellipse)
        except Exception as e:
            print(f"Could not draw highlighted ellipse for cell {vision_cell_id_selected}: {e}")

    # --- Plot styling ---
    ax.set_xlim(x_range)
    ax.set_ylim(y_range[1], y_range[0])
    ax.set_title("Population Receptive Fields", color='white')
    ax.set_xlabel("X (stixels)", color='gray')
    ax.set_ylabel("Y (stixels)", color='gray')
    ax.set_facecolor('#1f1f1f')
    ax.tick_params(colors='gray')
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
    ax.set_aspect('equal', adjustable='box')


def plot_sta_timecourse(fig, sta_data, stafit, vision_params, cell_id, sampling_rate=20):
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
                ax.plot(time_axis, timecourse_matrix[:, i], color=colors[i], linewidth=1.5, label=channel_names[i])

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
            
            # --- ACCURATE Y-AXIS SCALING ---
            # This logic fits the axis tightly to the min/max of the saved data.
            if timecourse_matrix.size > 0:
                y_min = timecourse_matrix.min()
                y_max = timecourse_matrix.max()
                y_range = y_max - y_min if y_max > y_min else 1.0
                y_margin = y_range * 0.10  # Add a 10% margin for readability
                ax.set_ylim(y_min - y_margin, y_max + y_margin)
                
            return

    # Fallback logic remains unchanged
    print(f"Warning: Could not access pre-calculated timecourse data for cell {cell_id}, falling back to recalculation.")
    
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
    
    red_timecourse = np.mean(red_channel[y_min:y_max, x_min:x_max], axis=(0, 1))
    green_timecourse = np.mean(green_channel[y_min:y_max, x_min:x_max], axis=(0, 1))
    blue_timecourse = np.mean(blue_channel[y_min:y_max, x_min:x_max], axis=(0, 1))
    
    # Fallback uses a hardcoded duration as it has no refresh_time metadata
    total_time_ms = 1500
    time_axis = np.linspace(-total_time_ms, 0, n_timepoints)
    
    ax = fig.add_subplot(111)
    
    ax.plot(time_axis, red_timecourse, color='red', linewidth=1.5, label='Red')
    ax.plot(time_axis, green_timecourse, color='green', linewidth=1.5, label='Green')
    ax.plot(time_axis, blue_timecourse, color='blue', linewidth=1.5, label='Blue')
    
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


def animate_sta_movie(fig, sta_data, stafit=None, frame_index=0, sta_width=None, sta_height=None, ax=None):
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
    
    extent = [0, sta_width, sta_height, 0] if sta_width is not None else [0, red_frame.shape[1], red_frame.shape[0], 0]
    
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
    
    ax.set_title(f"STA Movie - Frame {frame_index+1}/{n_frames}", color='white')
    ax.set_xlabel("X (stixels)", color='gray')
    ax.set_ylabel("Y (stixels)", color='gray')
    ax.set_facecolor('#1f1f1f')
    ax.tick_params(colors='gray')
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
    
    fig.tight_layout()


# --- New Plotting Function for Vision RF ---
def plot_vision_rf(fig, sta_data, stafit, sta_width=None, sta_height=None, ax=None):
    """
    Visualizes the receptive field from loaded Vision STA and parameter data.
    
    Args:
        fig (matplotlib.figure.Figure): The figure object to draw on.
        sta_data (STAContainer): Named tuple containing the raw STA movie.
        stafit (STAFit): Named tuple containing the Gaussian fit parameters.
        sta_width (int, optional): Width of the stimulus in stixels. If None, inferred from data.
        sta_height (int, optional): Height of the stimulus in stixels. If None, inferred from data.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, creates a new subplot.
    """
    if ax is None:
        fig.clear()
        ax = fig.add_subplot(111)
    
    # Check if the STA has multiple color channels
    red_channel = sta_data.red
    green_channel = sta_data.green
    blue_channel = sta_data.blue
    
    # Determine the peak frame for each channel
    peak_frame_idx_red = np.argmax(np.max(np.abs(red_channel), axis=(0, 1)))
    peak_frame_idx_green = np.argmax(np.max(np.abs(green_channel), axis=(0, 1)))
    peak_frame_idx_blue = np.argmax(np.max(np.abs(blue_channel), axis=(0, 1)))
    
    # Get the peak frames of each channel
    red_frame = red_channel[:, :, peak_frame_idx_red]
    green_frame = green_channel[:, :, peak_frame_idx_green]
    blue_frame = blue_channel[:, :, peak_frame_idx_blue]
    
    # Stack the peak frames to create a multi-channel RGB image
    # Note: The actual protocol for combining channels depends on the stimulus type
    # e.g., for blue-yellow protocol, you might want to show blue/yellow opponent channels
    # For now, we'll create an RGB image from the peak frames
    sta_rgb = np.stack([red_frame, green_frame, blue_frame], axis=-1)
    
    # Normalize the values to [0, 1] for proper color display
    # Find min/max across all channels to maintain relative scaling
    min_val = min(red_frame.min(), green_frame.min(), blue_frame.min())
    max_val = max(red_frame.max(), green_frame.max(), blue_frame.max())
    
    if max_val != min_val:
        sta_rgb_normalized = (sta_rgb - min_val) / (max_val - min_val)
    else:
        sta_rgb_normalized = np.zeros_like(sta_rgb)
    
    # Determine the extent for the imshow based on provided dimensions or data shape
    if sta_width is not None and sta_height is not None:
        # Use the provided dimensions for proper coordinate alignment
        # With origin='upper', the extent is [left, right, top, bottom]
        extent = [0, sta_width, sta_height, 0]
    else:
        # Fallback to the shape of the peak frame
        extent = [0, red_frame.shape[1], red_frame.shape[0], 0]  # [0, width, height, 0]
    
    # Display the colored STA frame with correct orientation
    ax.imshow(sta_rgb_normalized, origin='upper', extent=extent)
    
    # Overlay the Gaussian fit ellipse using Vision coordinates
    # Since we're using origin='upper', the y-coordinate needs to be inverted
    if sta_width is not None and sta_height is not None:
        adjusted_y = sta_height - stafit.center_y
        ellipse = Ellipse(
            xy=(stafit.center_x, adjusted_y),
            width=2 * stafit.std_x,
            height=2 * stafit.std_y,
            angle=np.rad2deg(stafit.rot),
            edgecolor='cyan',
            facecolor='none',
            lw=2
        )
    else:
        # Fallback: try to get dimensions from the image shape
        image_height = red_frame.shape[0]  # height of the original frame
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
    
    # Style the plot
    ax.set_title("Receptive Field (STA + Fit)", color='white')
    ax.set_xlabel("X (stixels)", color='gray')
    ax.set_ylabel("Y (stixels)", color='gray')
    ax.set_facecolor('#1f1f1f')
    ax.tick_params(colors='gray')
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
    
    fig.tight_layout()

