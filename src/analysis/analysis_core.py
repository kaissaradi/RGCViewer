"""
Analysis Core Functions

This module contains all analysis functions for RGC data processing.
"""

import warnings
import numpy as np
from pathlib import Path
import logging

from scipy.signal import peak_widths
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)

try:
    from bin2py import PyBinFileReader as _PyBinFileReader
except ImportError:
    _PyBinFileReader = None


# ---------------------------------------------------------------------------
# Snippet extraction
# ---------------------------------------------------------------------------

def _extract_snippets_from_reader(reader, spike_times, window, n_channels):
    """
    Extract snippets from a PyBinFileReader.

    The Litke .bin format stores a TTL channel at index 0 (row 0 when
    is_row_major=True), so all Kilosort channel indices must be shifted
    by +1 to skip it.  reader.get_data() returns shape
    (N_ELECTRODES_total, num_samples) with is_row_major=True.

    Returns array of shape (n_channels, snip_len, n_spikes).
    """
    snip_len      = int(window[1] - window[0])
    n_spikes      = len(spike_times)
    total_samples = reader.length

    snips = np.zeros((n_spikes, n_channels, snip_len), dtype=np.float32)

    # Litke electrode indices are 1-based (0 is TTL).
    litke_channel_rows = np.arange(1, n_channels + 1, dtype=np.intp)
    spike_times = np.asarray(spike_times, dtype=np.int64)

    for i, spike_time in enumerate(spike_times):
        start_sample = int(spike_time) + int(window[0])
        end_sample   = start_sample + snip_len

        if start_sample < 0 or end_sample > total_samples:
            continue

        try:
            raw_block = reader.get_data(start_sample, snip_len)
            snips[i] = raw_block[litke_channel_rows, :]
        except Exception:
            logger.debug(
                "PyBinFileReader.get_data failed for spike %d at sample %d",
                i, start_sample, exc_info=True,
            )

    return snips.transpose(1, 2, 0)  # → (n_channels, snip_len, n_spikes)


def extract_snippets(dat_path_or_memmap, spike_times,
                     window=(-20, 60), n_channels=512, dtype='int16'):
    """Extracts snippets of raw data around each spike time.

    Accepts one of three source types:
    - ``PyBinFileReader``  – native Litke .bin format.  A +1 channel offset
                             is applied internally to skip the TTL row.
    - ``str`` / ``Path``  – path to a flat binary .dat file; opened as a
                             read-only numpy memmap.
    - ndarray-like         – an already-open memmap or in-memory array with
                             shape ``(n_samples, n_channels)``.

    Returns:
        np.ndarray of shape ``(n_channels, snip_len, n_spikes)``, dtype float32.
    """
    snip_len    = int(window[1] - window[0])
    spike_count = len(spike_times)

    if spike_count == 0:
        return np.zeros((n_channels, snip_len, 0), dtype=np.float32)

    if _PyBinFileReader is not None and isinstance(dat_path_or_memmap, _PyBinFileReader):
        return _extract_snippets_from_reader(
            dat_path_or_memmap, spike_times, window, n_channels
        )

    if isinstance(dat_path_or_memmap, (str, Path)):
        raw_data = np.memmap(str(dat_path_or_memmap), dtype=dtype, mode='r')
        raw_data = raw_data.reshape(-1, n_channels)
    else:
        raw_data = dat_path_or_memmap
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(-1, n_channels)

    total_samples = raw_data.shape[0]
    snips = np.zeros((spike_count, n_channels, snip_len), dtype=np.float32)
    spike_times = np.asarray(spike_times, dtype=np.int64)

    for i, spike_time in enumerate(spike_times):
        start_sample = int(spike_time) + int(window[0])
        end_sample   = start_sample + snip_len

        if start_sample < 0 or end_sample > total_samples:
            continue

        snips[i] = raw_data[start_sample:end_sample, :].T

    return snips.transpose(1, 2, 0)  # → (n_channels, snip_len, n_spikes)


# ---------------------------------------------------------------------------
# EI utilities
# ---------------------------------------------------------------------------

def baseline_correct(snips, pre_samples=20):
    """Subtract pre-spike baseline mean from each channel."""
    if snips.ndim == 3:
        baseline = snips[:, :pre_samples, :].mean(axis=1)
        return snips - baseline[:, np.newaxis, :]
    return snips - snips[:, :pre_samples].mean(axis=1, keepdims=True)


def compute_ei(snips, pre_samples=20):
    """Compute the Electrical Image (median waveform) from snippets."""
    import torch
    snips = baseline_correct(snips, pre_samples=pre_samples)
    snips_torch = torch.from_numpy(snips)
    return torch.median(snips_torch, dim=2).values.numpy()


def select_channels(ei, min_chan=30, max_chan=80, threshold=15):
    """Select channels from an EI by peak-to-peak amplitude."""
    p2p = ei.max(axis=1) - ei.min(axis=1)
    selected = np.where(p2p > threshold)[0]
    if len(selected) > max_chan:
        selected = np.argsort(p2p)[-max_chan:]
    elif len(selected) < min_chan and len(p2p) > min_chan:
        selected = np.argsort(p2p)[-min_chan:]
    return np.sort(selected)


def compute_spatial_features(ei, channel_positions, _sampling_rate):
    """
    Compute spatial features for an electrical image.

    Args:
        ei (np.ndarray): shape (n_channels, n_samples).
        channel_positions (np.ndarray): shape (n_channels, 2).
        _sampling_rate: unused; retained for API compatibility.

    Returns:
        dict with keys: max_amplitude, center_of_mass_x/y, spatial_spread.
    """
    if ei is None or channel_positions is None:
        return {
            "max_amplitude": 0,
            "center_of_mass_x": np.nan,
            "center_of_mass_y": np.nan,
            "spatial_spread": 0,
        }

    amplitudes = np.ptp(ei, axis=1)
    max_amp    = np.max(amplitudes)
    total_amp  = np.sum(amplitudes)

    if total_amp > 0:
        com_x = np.dot(channel_positions[:, 0], amplitudes) / total_amp
        com_y = np.dot(channel_positions[:, 1], amplitudes) / total_amp
        dx = channel_positions[:, 0] - com_x
        dy = channel_positions[:, 1] - com_y
        spatial_spread = np.sqrt(np.dot(dx**2 + dy**2, amplitudes) / total_amp)
    else:
        com_x = com_y = np.nan
        spatial_spread = 0.0

    return {
        "max_amplitude": max_amp,
        "center_of_mass_x": com_x,
        "center_of_mass_y": com_y,
        "spatial_spread": spatial_spread,
    }


# ---------------------------------------------------------------------------
# STA timecourse retrieval
# ---------------------------------------------------------------------------

def get_sta_timecourse_data(sta_data, stafit, vision_params, cell_id):
    """
    Retrieve or calculate the STA timecourse for a cell.

    Tries pre-calculated Vision params first, falls back to extracting the
    centre-pixel timecourse from the raw STA cube.

    Returns:
        (time_axis, timecourse_matrix, source)
        - time_axis          : 1-D array, ms relative to spike (negative = before)
        - timecourse_matrix  : shape (n_timepoints, 3) — columns [R, G, B]
        - source             : "precalculated" | "recalculated"
        All three are None on failure.
    """
    timecourse_matrix = None
    source = "precalculated"

    try:
        red_tc   = vision_params.get_data_for_cell(cell_id, 'RedTimeCourse')
        green_tc = vision_params.get_data_for_cell(cell_id, 'GreenTimeCourse')
        blue_tc  = vision_params.get_data_for_cell(cell_id, 'BlueTimeCourse')
        if red_tc is not None and green_tc is not None and blue_tc is not None:
            timecourse_matrix = np.stack([red_tc, green_tc, blue_tc], axis=1)
    except Exception:
        pass

    if timecourse_matrix is None and sta_data is not None:
        source = "recalculated"
        r, g, b = sta_data.red, sta_data.green, sta_data.blue

        if stafit:
            cx = int(stafit.center_x)
            cy = int(stafit.center_y)
            sx = int(max(1, stafit.std_x))
            sy = int(max(1, stafit.std_y))
            x0, x1 = max(0, cx - sx), min(r.shape[1], cx + sx + 1)
            y0, y1 = max(0, cy - sy), min(r.shape[0], cy + sy + 1)
            red_tc   = np.mean(r[y0:y1, x0:x1], axis=(0, 1))
            green_tc = np.mean(g[y0:y1, x0:x1], axis=(0, 1))
            blue_tc  = np.mean(b[y0:y1, x0:x1], axis=(0, 1))
        else:
            # No fit: extract from the peak pixel of the red channel
            peak_idx = np.unravel_index(np.argmax(np.abs(r)), r.shape)
            yi, xi   = peak_idx[0], peak_idx[1]
            red_tc, green_tc, blue_tc = r[yi, xi, :], g[yi, xi, :], b[yi, xi, :]

        timecourse_matrix = np.stack([red_tc, green_tc, blue_tc], axis=1)

    if timecourse_matrix is None:
        return None, None, None

    n_timepoints = timecourse_matrix.shape[0]
    refresh_ms   = getattr(sta_data, 'refresh_time', 1000.0 / 60.0) if sta_data else 1000.0 / 60.0
    time_axis    = np.linspace(-(n_timepoints - 1) * refresh_ms, 0, n_timepoints)

    return time_axis, timecourse_matrix, source


# ---------------------------------------------------------------------------
# STA metrics  ← the canonical single source of truth
# ---------------------------------------------------------------------------

def compute_sta_metrics(sta_data, stafit, vision_params, cell_id):
    """
    Compute scalar metrics from STA data for a single RGC.

    All temporal annotations used by the plot (peak time, FWHM, polarity)
    are derived here from a single smoothed trace so the table and the
    plot are guaranteed to show identical numbers.

    Returns a flat dict.  Keys relevant to the temporal filter plot are
    also accessible as raw floats via the '_raw' sub-dict attached to the
    returned object (a plain dict with an extra attribute won't survive
    JSON round-trips, so the raw values are stored under the key
    '_raw_temporal' as a nested dict).

    Kept metrics (display-ready strings)
    ─────────────────────────────────────
    Temporal
        Dominant Channel   – "Red" | "Green" | "Blue" | "Blue (B/W)"
        Polarity           – "ON" | "OFF"
        Peak (ms)          – time of dominant-channel peak, ms pre-spike
        FWHM (ms)          – half-width at half-max of primary peak, ms
        Biphasic Index     – |secondary| / |primary|  (0 = monophasic)
        SNR                – response std / baseline std

    Spatial  (only if stafit is present)
        RF σx / σy         – Gaussian sigma in stixels, both axes
        RF Area (stix²)    – π·σx·σy
        Orientation (°)    – rotation of Gaussian ellipse
        Ellipticity        – σy/σx  (1 = circular)
    """
    metrics: dict = {}

    time_axis, tc_matrix, source = get_sta_timecourse_data(
        sta_data, stafit, vision_params, cell_id
    )

    # ── Temporal block ───────────────────────────────────────────────────────
    if tc_matrix is not None:
        # Dominant channel
        if tc_matrix.shape[1] == 1:
            dom_idx  = 2          # treat B/W as blue
            dom_trace = tc_matrix[:, 0]
        else:
            energies  = np.sum(tc_matrix ** 2, axis=0)
            # Intercept duplicated B/W signal across R/G/B and force Blue
            if np.isclose(energies[0], energies[2], rtol=1e-5):
                dom_idx = 2
            else:
                dom_idx   = int(np.argmax(energies))
            dom_trace = tc_matrix[:, dom_idx]

        channel_names = ["Red", "Green", "Blue"]
        dom_name = "Blue (B/W)" if tc_matrix.shape[1] == 1 else channel_names[dom_idx]

        # Smooth once — every derived quantity uses this trace
        sigma_samples  = max(1, int(0.02 * len(dom_trace)))
        smoothed       = gaussian_filter1d(dom_trace, sigma=sigma_samples)

        abs_max = np.max(np.abs(smoothed))
        if abs_max > 0:
            norm = smoothed / abs_max
        else:
            norm = smoothed.copy()

        # Polarity
        peak_val   = np.max(norm)
        trough_val = np.min(norm)
        is_off     = abs(trough_val) > abs(peak_val)
        polarity   = "OFF" if is_off else "ON"

        # Primary peak index and time
        primary_idx = int(np.argmin(norm) if is_off else np.argmax(norm))
        primary_val = norm[primary_idx]
        peak_ms     = float(time_axis[primary_idx])

        # FWHM  ── single calculation, used by both table and plot annotation
        fwhm_ms = float('nan')
        try:
            trace_for_width = -norm if is_off else norm
            if not np.isclose(trace_for_width[primary_idx], 0.0):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    widths, width_heights, left_ips, right_ips = peak_widths(
                        trace_for_width, peaks=[primary_idx], rel_height=0.5
                    )
                if len(widths) > 0:
                    sample_interval = abs(time_axis[1] - time_axis[0])
                    fwhm_ms         = float(widths[0] * sample_interval)
                    # Compute real-time endpoints for the plot annotation
                    fwhm_t_start = float(time_axis[0] + left_ips[0]  * sample_interval)
                    fwhm_t_end   = float(time_axis[0] + right_ips[0] * sample_interval)
                    fwhm_h       = float(width_heights[0] * (-1 if is_off else 1))
                else:
                    fwhm_t_start = fwhm_t_end = fwhm_h = float('nan')
            else:
                fwhm_t_start = fwhm_t_end = fwhm_h = float('nan')
        except Exception as exc:
            logger.debug("FWHM calculation failed: %s", exc)
            fwhm_t_start = fwhm_t_end = fwhm_h = float('nan')

        # Biphasic index
        post = norm[primary_idx + 1:]
        if len(post) > 0:
            secondary_val = float(np.max(post) if is_off else np.min(post))
            biphasic      = abs(secondary_val / primary_val) if primary_val != 0 else 0.0
        else:
            biphasic = 0.0

        # SNR: response std vs. early-baseline std
        baseline_len = int(0.25 * len(norm))
        if baseline_len > 5:
            baseline_std  = float(np.std(norm[:baseline_len]))
            response_std  = float(np.std(norm[baseline_len:]))
            snr = response_std / baseline_std if baseline_std > 0 else float('inf')
        else:
            snr = float('nan')

        # ── Store display strings ────────────────────────────────────────────
        metrics["Dominant Channel"] = dom_name
        metrics["Polarity"]         = polarity
        metrics["Peak (ms)"]        = f"{peak_ms:.1f}"
        metrics["FWHM (ms)"]        = f"{fwhm_ms:.1f}" if not np.isnan(fwhm_ms) else "N/A"
        metrics["Biphasic Index"]   = f"{biphasic:.3f}"
        metrics["SNR"]              = f"{snr:.2f}"   if not np.isnan(snr) else "N/A"

        # ── Raw floats for plot annotation ───────────────────────────────────
        # The temporal filter plot reads these directly so it never has to
        # recompute — table and annotation are guaranteed identical.
        metrics["_raw_temporal"] = {
            "dom_idx":      dom_idx,
            "dom_name":     dom_name,
            "is_off":       is_off,
            "norm_trace":   norm,           # smoothed, normalised
            "raw_tc":       tc_matrix,      # all channels, un-normalised
            "time_axis":    time_axis,
            "primary_idx":  primary_idx,
            "peak_ms":      peak_ms,
            "peak_val":     float(primary_val),
            "fwhm_ms":      fwhm_ms,
            "fwhm_t_start": fwhm_t_start,
            "fwhm_t_end":   fwhm_t_end,
            "fwhm_h":       fwhm_h,
            "biphasic":     biphasic,
            "snr":          snr,
            "source":       source,
        }

    # ── Spatial block ────────────────────────────────────────────────────────
    if stafit:
        sx, sy = stafit.std_x, stafit.std_y

        orientation_deg = np.rad2deg(stafit.rot) % 180   # keep in [0, 180)
        area            = np.pi * sx * sy
        ellipticity     = (sy / sx) if sx > 0 else float('inf')

        metrics["RF σx (stix)"]    = f"{sx:.2f}"
        metrics["RF σy (stix)"]    = f"{sy:.2f}"
        metrics["RF Area (stix²)"] = f"{area:.1f}"
        metrics["Orientation (°)"] = f"{orientation_deg:.1f}"
        metrics["Ellipticity"]     = f"{ellipticity:.2f}" if not np.isinf(ellipticity) else "∞"

    return metrics


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    'extract_snippets',
    'baseline_correct',
    'compute_ei',
    'select_channels',
    'get_sta_timecourse_data',
    'compute_sta_metrics',
    'compute_spatial_features',
]