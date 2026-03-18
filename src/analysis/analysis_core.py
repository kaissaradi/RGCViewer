"""
Analysis Core Functions

This module contains all analysis functions for RGC data processing.
"""

import numpy as np
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
from scipy.signal import peak_widths

try:
    from bin2py import PyBinFileReader as _PyBinFileReader
except ImportError:
    _PyBinFileReader = None


def _extract_snippets_from_reader(reader, spike_times, window, n_channels):
    """
    Extract snippets from a PyBinFileReader.

    The Litke .bin format stores a TTL channel at index 0 (row 0 when
    is_row_major=True), so all Kilosort channel indices must be shifted
    by +1 to skip it.  reader.get_data() returns shape
    (N_ELECTRODES_total, num_samples) with is_row_major=True.

    Returns array of shape (n_channels, snip_len, n_spikes), matching
    the contract of extract_snippets.
    """
    snip_len  = int(window[1] - window[0])
    n_spikes  = len(spike_times)
    total_samples = reader.length

    snips = np.zeros((n_spikes, n_channels, snip_len), dtype=np.float32)

    # Litke electrode indices are 1-based (0 is TTL); build the index array once.
    # We want Kilosort channels 0..(n_channels-1)  →  Litke rows 1..n_channels.
    litke_channel_rows = np.arange(1, n_channels + 1, dtype=np.intp)

    spike_times = np.asarray(spike_times, dtype=np.int64)

    for i, spike_time in enumerate(spike_times):
        start_sample = int(spike_time) + int(window[0])
        end_sample   = start_sample + snip_len

        if start_sample < 0 or end_sample > total_samples:
            continue  # out-of-bounds spike — leave row as zeros

        try:
            # raw_block: (N_ELECTRODES_total, snip_len)
            raw_block = reader.get_data(start_sample, snip_len)
            # Select the real electrode rows (skip TTL at row 0)
            snips[i, :, :] = raw_block[litke_channel_rows, :]
        except Exception:
            logger.debug("PyBinFileReader.get_data failed for spike %d at sample %d",
                         i, start_sample, exc_info=True)
            continue

    # (n_spikes, n_channels, snip_len) → (n_channels, snip_len, n_spikes)
    return snips.transpose(1, 2, 0)


def extract_snippets(dat_path_or_memmap, spike_times,
                     window=(-20, 60), n_channels=512, dtype='int16'):
    """Extracts snippets of raw data around each spike time.

    Accepts one of three source types:
    - ``PyBinFileReader``   – native Litke .bin format (preferred when
                              loading from Litke directories).  A +1 channel
                              offset is applied internally to skip the TTL row.
    - ``str`` / ``Path``   – path to a flat binary .dat file; opened as a
                              read-only numpy memmap.
    - ndarray-like          – an already-open memmap or in-memory array with
                              shape ``(n_samples, n_channels)``.

    Returns:
        np.ndarray of shape ``(n_channels, snip_len, n_spikes)``, dtype float32.
    """
    snip_len    = int(window[1] - window[0])
    spike_count = len(spike_times)

    if spike_count == 0:
        return np.zeros((n_channels, snip_len, 0), dtype=np.float32)

    # --- PyBinFileReader branch (Litke native) -----------------------------------
    if _PyBinFileReader is not None and isinstance(dat_path_or_memmap, _PyBinFileReader):
        return _extract_snippets_from_reader(
            dat_path_or_memmap, spike_times, window, n_channels
        )

    # --- flat file path branch ---------------------------------------------------
    if isinstance(dat_path_or_memmap, (str, Path)):
        raw_data = np.memmap(str(dat_path_or_memmap), dtype=dtype, mode='r')
        try:
            raw_data = raw_data.reshape(-1, n_channels)
        except Exception:
            raw_data = raw_data.reshape(-1, n_channels)
    else:
        # --- ndarray / memmap branch ---------------------------------------------
        raw_data = dat_path_or_memmap
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(-1, n_channels)

    total_samples = raw_data.shape[0]

    # Preallocate in spike-major order then transpose for return
    snips = np.zeros((spike_count, n_channels, snip_len), dtype=np.float32)

    spike_times = np.asarray(spike_times, dtype=np.int64)

    for i, spike_time in enumerate(spike_times):
        start_sample = int(spike_time) + int(window[0])
        end_sample   = start_sample + snip_len

        if start_sample < 0 or end_sample > total_samples:
            continue

        snippet = raw_data[start_sample:end_sample, :]
        # (snip_len, n_channels) → (n_channels, snip_len)
        snips[i, :, :] = snippet.T

    return snips.transpose(1, 2, 0)


def baseline_correct(snips, pre_samples=20):
    """Corrects for baseline shifts in an EI."""
    if snips.ndim == 3:
        baseline = snips[:, :pre_samples, :].mean(axis=1)
        return snips - baseline[:, np.newaxis, :]
    else:
        return snips - snips[:, :pre_samples].mean(axis=1, keepdims=True)


def compute_ei(snips, pre_samples=20):
    import torch
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


def get_sta_timecourse_data(sta_data, stafit, vision_params, cell_id):
    """
    Retrieves or calculates the STA timecourse for a cell.
    Returns (time_axis, timecourse_matrix, source)
    """
    timecourse_matrix = None
    source = "precalculated"

    # Try pre-calculated first
    try:
        red_tc = vision_params.get_data_for_cell(cell_id, 'RedTimeCourse')
        green_tc = vision_params.get_data_for_cell(cell_id, 'GreenTimeCourse')
        blue_tc = vision_params.get_data_for_cell(cell_id, 'BlueTimeCourse')
        if red_tc is not None and green_tc is not None and blue_tc is not None:
            timecourse_matrix = np.stack([red_tc, green_tc, blue_tc], axis=1)
    except Exception:
        pass

    # Fallback to calculation from raw STA
    if timecourse_matrix is None and sta_data is not None:
        source = "recalculated"
        red_channel = sta_data.red
        green_channel = sta_data.green
        blue_channel = sta_data.blue

        if stafit:
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
        else:
             # Fallback if no fit: use max pixel
            peak_idx = np.unravel_index(np.argmax(np.abs(red_channel)), red_channel.shape)
            y_idx, x_idx = peak_idx[0], peak_idx[1]
            red_timecourse = red_channel[y_idx, x_idx, :]
            green_timecourse = green_channel[y_idx, x_idx, :]
            blue_timecourse = blue_channel[y_idx, x_idx, :]

        timecourse_matrix = np.stack([red_timecourse, green_timecourse, blue_timecourse], axis=1)

    if timecourse_matrix is None:
        return None, None, None

    n_timepoints = timecourse_matrix.shape[0]

    # Calculate time axis
    if sta_data and hasattr(sta_data, 'refresh_time'):
        refresh_ms = sta_data.refresh_time
    else:
        refresh_ms = 1000.0 / 60.0 # Default approx 60Hz

    total_duration_ms = (n_timepoints - 1) * refresh_ms
    time_axis = np.linspace(-total_duration_ms, 0, n_timepoints)

    return time_axis, timecourse_matrix, source


def compute_sta_metrics(sta_data, stafit, vision_params, cell_id):
    """Compute scalar metrics from STA for RGC classification."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import peak_widths
    from scipy.stats import skew

    metrics = {}
    time_axis, tc_matrix, _ = get_sta_timecourse_data(sta_data, stafit, vision_params, cell_id)

    if tc_matrix is not None:
        # Handle B/W vs color recordings
        if tc_matrix.shape[1] == 1:
            dom_idx = 2  # Blue channel for B/W
            dom_trace = tc_matrix[:, 0]
            red_trace = np.zeros_like(dom_trace)
            green_trace = np.zeros_like(dom_trace)
            blue_trace = dom_trace
        else:
            energies = np.sum(tc_matrix**2, axis=0)
            dom_idx = np.argmax(energies)
            red_trace = tc_matrix[:, 0]
            green_trace = tc_matrix[:, 1]
            blue_trace = tc_matrix[:, 2]
            dom_trace = tc_matrix[:, dom_idx]

        # Normalize for analysis - careful with zero division
        abs_max = np.max(np.abs(dom_trace))
        if abs_max > 0:
            norm_trace = dom_trace / abs_max
            red_trace / abs_max if abs_max > 0 else red_trace
            green_trace / abs_max if abs_max > 0 else green_trace
            blue_trace / abs_max if abs_max > 0 else blue_trace
        else:
            norm_trace = dom_trace

        # Apply Gaussian smoothing for cleaner analysis
        sigma_samples = max(1, int(0.02 * len(norm_trace)))  # 2% of trace length
        smoothed_trace = gaussian_filter1d(norm_trace, sigma=sigma_samples)

        # Polarity determination - use smoothed trace for robustness
        peak_val = np.max(smoothed_trace)
        trough_val = np.min(smoothed_trace)

        # Determine polarity based on which has larger absolute value
        if abs(trough_val) > abs(peak_val):
            is_off = True
            polarity = "OFF"
        else:
            is_off = False
            polarity = "ON"

        # Find primary peak/trough in the appropriate direction
        if is_off:
            primary_idx = np.argmin(smoothed_trace)
            primary_val = smoothed_trace[primary_idx]
        else:
            primary_idx = np.argmax(smoothed_trace)
            primary_val = smoothed_trace[primary_idx]

        time_to_peak = time_axis[primary_idx]

        # Calculate response integral (absolute area)
        response_integral = np.trapz(np.abs(smoothed_trace), time_axis)

        # Calculate SNR (signal std / baseline std)
        baseline_len = int(0.25 * len(smoothed_trace))
        if baseline_len > 5:
            baseline_std = np.std(smoothed_trace[:baseline_len])
            response_std = np.std(smoothed_trace[baseline_len:])
            snr = response_std / baseline_std if baseline_std > 0 else float('inf')
        else:
            snr = float('nan')

        # Find zero crossings (where trace crosses zero)
        zero_crossings = np.where(np.diff(np.signbit(norm_trace)))[0]
        if len(zero_crossings) > 0:
            # Find first zero crossing after primary peak
            post_peak_crossings = zero_crossings[zero_crossings > primary_idx]
            if len(post_peak_crossings) > 0:
                zero_crossing_time = time_axis[post_peak_crossings[0]]
            else:
                zero_crossing_time = None
        else:
            zero_crossing_time = None

        # Calculate response duration (time above/below baseline)
        # Define threshold as 20% of peak absolute value
        threshold = 0.2 * abs(primary_val)

        # Initialize variables to avoid UnboundLocalError
        below_threshold = np.zeros_like(smoothed_trace, dtype=bool)
        above_threshold = np.zeros_like(smoothed_trace, dtype=bool)
        response_duration = 0

        if is_off:
            below_threshold = smoothed_trace < (-threshold)
            mask = below_threshold
        else:
            above_threshold = smoothed_trace > threshold
            mask = above_threshold

        # Find contiguous regions for response duration
        if np.any(mask):
            # Find start and end indices of contiguous region around primary peak
            if mask[primary_idx]:
                # Find start
                start_idx = primary_idx
                while start_idx > 0 and mask[start_idx - 1]:
                    start_idx -= 1
                # Find end
                end_idx = primary_idx
                while end_idx < len(mask) - 1 and mask[end_idx + 1]:
                    end_idx += 1
                response_duration = time_axis[end_idx] - time_axis[start_idx]

        # Biphasic Index (enhanced with proper secondary peak detection)
        # Look for secondary peak in opposite direction AFTER primary peak
        secondary_peak_val = 0
        if is_off:
            # OFF cell: look for ON rebound (positive peak after OFF trough)
            if primary_idx < len(smoothed_trace) - 1:
                post_primary_segment = smoothed_trace[primary_idx + 1:]
                if len(post_primary_segment) > 0:
                    secondary_idx_rel = np.argmax(post_primary_segment)
                    if secondary_idx_rel < len(post_primary_segment):
                        secondary_peak_val = post_primary_segment[secondary_idx_rel]
        else:
            # ON cell: look for OFF rebound (negative peak after ON peak)
            if primary_idx < len(smoothed_trace) - 1:
                post_primary_segment = smoothed_trace[primary_idx + 1:]
                if len(post_primary_segment) > 0:
                    secondary_idx_rel = np.argmin(post_primary_segment)
                    if secondary_idx_rel < len(post_primary_segment):
                        secondary_peak_val = post_primary_segment[secondary_idx_rel]

        # Calculate biphasic index as absolute ratio
        if abs(primary_val) > 0:
            biphasic_index = abs(secondary_peak_val / primary_val)
        else:
            biphasic_index = 0

        # FWHM calculation with robust peak detection
        fwhm_ms = float('nan')
        try:
            # For width calculation, we need positive peaks
            trace_for_width = -smoothed_trace if is_off else smoothed_trace
            peak_idx_for_width = primary_idx

            # Use peak_widths with proper parameters and suppress warnings
            import warnings
            if np.isclose(trace_for_width[peak_idx_for_width], 0.0):
                widths = [0.0]
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)  # Suppress peak property warnings
                    widths, *_ = peak_widths(trace_for_width, peaks=[peak_idx_for_width],
                                            rel_height=0.5, prominence_data=None)

            if len(widths) > 0:
                sample_interval = abs(time_axis[1] - time_axis[0])
                fwhm_ms = widths[0] * sample_interval
        except Exception as e:
            logger.debug(f"FWHM calculation failed: {e}")

        # Color opponency (if color data available)
        if tc_matrix.shape[1] == 3:
            # Calculate RMS of each channel
            channel_rms = np.sqrt(np.mean(tc_matrix**2, axis=0))
            # Normalize by sum
            total_rms = np.sum(channel_rms)
            if total_rms > 0:
                channel_norm = channel_rms / total_rms
                # Color opponency index: difference between strongest and second strongest
                sorted_channels = np.argsort(channel_norm)[::-1]
                if len(sorted_channels) >= 2:
                    color_opponency = channel_norm[sorted_channels[0]] - channel_norm[sorted_channels[1]]
                else:
                    color_opponency = 0
            else:
                color_opponency = 0
        else:
            color_opponency = None

        # Store temporal metrics
        if tc_matrix.shape[1] == 1:
            dominant_channel_name = "Blue (B/W)"
        else:
            channel_names = ["Red", "Green", "Blue"]
            dominant_channel_name = channel_names[dom_idx]

        metrics.update({
            "Dominant Channel": dominant_channel_name,
            "Polarity": polarity,
            "Time to Peak (ms)": f"{time_to_peak:.1f}",
            "Response Duration (ms)": f"{response_duration:.1f}" if response_duration > 0 else "0.0",
            "Zero Crossing (ms)": f"{zero_crossing_time:.1f}" if zero_crossing_time is not None else "N/A",
            "Biphasic Index": f"{biphasic_index:.3f}",
            "FWHM (Duration)": f"{fwhm_ms:.1f} ms" if not np.isnan(fwhm_ms) else "N/A",
            "Response Integral": f"{response_integral:.4f}",
            "SNR (std ratio)": f"{snr:.2f}" if not np.isnan(snr) else "N/A"
        })

        if color_opponency is not None:
            metrics["Color Opponency"] = f"{color_opponency:.3f}"

    # 2. Spatial Metrics
    if stafit:
        # Basic spatial parameters
        metrics["RF Center X"] = f"{stafit.center_x:.1f}"
        metrics["RF Center Y"] = f"{stafit.center_y:.1f}"
        metrics["RF Sigma X"] = f"{stafit.std_x:.2f}"
        metrics["RF Sigma Y"] = f"{stafit.std_y:.2f}"
        metrics["Orientation"] = f"{np.rad2deg(stafit.rot):.1f}°"

        # Effective Area (ellipse area)
        area = np.pi * stafit.std_x * stafit.std_y
        metrics["RF Area (sq stix)"] = f"{area:.1f}"

        # Ellipticity and asymmetry metrics
        if stafit.std_x > 0:
            ellipticity = stafit.std_y / stafit.std_x
        else:
            ellipticity = float('inf')

        if (stafit.std_x + stafit.std_y) > 0:
            elongation = (stafit.std_x - stafit.std_y) / (stafit.std_x + stafit.std_y)
        else:
            elongation = 0

        metrics["RF Ellipticity (σy/σx)"] = f"{ellipticity:.2f}" if not np.isinf(ellipticity) else "∞"
        metrics["RF Elongation"] = f"{elongation:.3f}"

        # Calculate spatial statistics from STA data
        if sta_data is not None and tc_matrix is not None:
            try:
                # Get the frame at time of peak response
                if tc_matrix.shape[1] == 1:
                    # B/W: use blue channel
                    peak_frame_idx = np.argmax(np.abs(blue_trace))
                    spatial_frame = sta_data.blue[:, :, peak_frame_idx]
                else:
                    # Color: use dominant channel
                    peak_frame_idx = np.argmax(np.abs(dom_trace))
                    if dom_idx == 0:
                        spatial_frame = sta_data.red[:, :, peak_frame_idx]
                    elif dom_idx == 1:
                        spatial_frame = sta_data.green[:, :, peak_frame_idx]
                    else:
                        spatial_frame = sta_data.blue[:, :, peak_frame_idx]

                # Flatten and remove NaN/inf
                frame_flat = spatial_frame.flatten()
                frame_flat = frame_flat[np.isfinite(frame_flat)]

                if len(frame_flat) > 0:
                    # Calculate spatial statistics
                    spatial_peak = np.max(frame_flat)
                    spatial_trough = np.min(frame_flat)

                    if spatial_trough != 0:
                        peak_trough_ratio = abs(spatial_peak / spatial_trough)
                    else:
                        peak_trough_ratio = float('inf')

                    # Spatial skewness (asymmetry measure)
                    if len(frame_flat) > 1 and np.std(frame_flat) > 0:
                        spatial_skewness = skew(frame_flat)
                    else:
                        spatial_skewness = 0

                    # Spatial kurtosis (peakedness measure)
                    if len(frame_flat) > 3 and np.std(frame_flat) > 0:
                        from scipy.stats import kurtosis
                        spatial_kurtosis = kurtosis(frame_flat)
                    else:
                        spatial_kurtosis = 0

                    metrics["Spatial Peak"] = f"{spatial_peak:.4f}"
                    metrics["Spatial Trough"] = f"{spatial_trough:.4f}"
                    metrics["Peak/Trough Ratio"] = f"{peak_trough_ratio:.2f}" if not np.isinf(peak_trough_ratio) else "∞"
                    metrics["Spatial Skewness"] = f"{spatial_skewness:.3f}"
                    metrics["Spatial Kurtosis"] = f"{spatial_kurtosis:.3f}"

                    # Additional asymmetry measure: ON/OFF balance
                    if spatial_peak > 0 and spatial_trough < 0:
                        on_off_balance = (spatial_peak + spatial_trough) / (spatial_peak - spatial_trough)
                        metrics["ON/OFF Balance"] = f"{on_off_balance:.3f}"

            except Exception as e:
                logger.debug(f"Failed to calculate spatial statistics: {e}")

    # 3. Overall metrics
    if sta_data is not None:
        try:
            # Calculate total energy across all channels and time
            total_energy = 0
            if hasattr(sta_data, 'red') and sta_data.red is not None:
                total_energy += np.nansum(sta_data.red**2)
            if hasattr(sta_data, 'green') and sta_data.green is not None:
                total_energy += np.nansum(sta_data.green**2)
            if hasattr(sta_data, 'blue') and sta_data.blue is not None:
                total_energy += np.nansum(sta_data.blue**2)

            # Calculate SNR from energy perspective
            # Estimate noise from baseline of timecourse
            if tc_matrix is not None and tc_matrix.size > 0:
                baseline_len = min(10, tc_matrix.shape[0])
                baseline_energy = np.mean(np.sum(tc_matrix[:baseline_len, :]**2, axis=1))
                if baseline_energy > 0:
                    energy_snr = total_energy / (baseline_energy * tc_matrix.shape[0])
                else:
                    energy_snr = float('inf')
            else:
                energy_snr = float('nan')

            metrics["Total Energy"] = f"{total_energy:.2e}"
            if not np.isnan(energy_snr):
                metrics["Energy SNR"] = f"{energy_snr:.1f}"

        except Exception as e:
            logger.debug(f"Failed to calculate energy metrics: {e}")

    return metrics


def compute_spatial_features(ei, channel_positions, _sampling_rate):
    """
    Computes spatial features for an electrical image.

    Args:
        ei (np.ndarray): The electrical image (n_channels, n_samples).
        channel_positions (np.ndarray): Channel coordinates (n_channels, 2).
        sampling_rate (float): Sampling rate in Hz.

    Returns:
        dict: A dictionary of computed features.
    """
    if ei is None or channel_positions is None:
        return {
            "max_amplitude": 0,
            "center_of_mass_x": np.nan,
            "center_of_mass_y": np.nan,
            "spatial_spread": 0
        }

    # 1. Amplitude per channel (Peak-to-Peak)
    # ei shape: (n_channels, n_samples)
    amplitudes = np.ptp(ei, axis=1)  # Peak-to-peak amplitude
    max_amp = np.max(amplitudes)

    # 2. Center of Mass
    # Use amplitudes as weights.
    total_amp = np.sum(amplitudes)
    if total_amp > 0:
        com_x = np.sum(channel_positions[:, 0] * amplitudes) / total_amp
        com_y = np.sum(channel_positions[:, 1] * amplitudes) / total_amp
    else:
        com_x = np.nan
        com_y = np.nan

    # 3. Spatial Spread (weighted std dev)
    if total_amp > 0 and not np.isnan(com_x):
        # Distances from COM
        dx = channel_positions[:, 0] - com_x
        dy = channel_positions[:, 1] - com_y
        dist_sq = dx**2 + dy**2

        # Weighted variance
        spatial_variance = np.sum(dist_sq * amplitudes) / total_amp
        spatial_spread = np.sqrt(spatial_variance)
    else:
        spatial_spread = 0

    return {
        "max_amplitude": max_amp,
        "center_of_mass_x": com_x,
        "center_of_mass_y": com_y,
        "spatial_spread": spatial_spread
    }


__all__ = [
    'extract_snippets',
    'baseline_correct',
    'compute_ei',
    'select_channels',
    'get_sta_timecourse_data',
    'compute_sta_metrics',
    'compute_spatial_features'
]