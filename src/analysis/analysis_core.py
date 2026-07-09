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
    snip_len = int(window[1] - window[0])
    n_spikes = len(spike_times)
    total_samples = reader.length

    snips = np.zeros((n_spikes, n_channels, snip_len), dtype=np.float32)

    # Litke electrode indices are 1-based (0 is TTL).
    litke_channel_rows = np.arange(1, n_channels + 1, dtype=np.intp)
    spike_times = np.asarray(spike_times, dtype=np.int64)

    for i, spike_time in enumerate(spike_times):
        start_sample = int(spike_time) + int(window[0])
        end_sample = start_sample + snip_len

        if start_sample < 0 or end_sample > total_samples:
            continue

        try:
            raw_block = reader.get_data(start_sample, snip_len)
            snips[i] = raw_block[litke_channel_rows, :]
        except Exception:
            logger.debug(
                "PyBinFileReader.get_data failed for spike %d at sample %d",
                i,
                start_sample,
                exc_info=True,
            )

    return snips.transpose(1, 2, 0)  # → (n_channels, snip_len, n_spikes)


def extract_snippets(
    dat_path_or_memmap, spike_times, window=(-20, 60), n_channels=512, dtype="int16"
):
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
    snip_len = int(window[1] - window[0])
    spike_count = len(spike_times)

    if spike_count == 0:
        return np.zeros((n_channels, snip_len, 0), dtype=np.float32)

    if _PyBinFileReader is not None and isinstance(
        dat_path_or_memmap, _PyBinFileReader
    ):
        return _extract_snippets_from_reader(
            dat_path_or_memmap, spike_times, window, n_channels
        )

    if isinstance(dat_path_or_memmap, (str, Path)):
        raw_data = np.memmap(str(dat_path_or_memmap), dtype=dtype, mode="r")
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
        end_sample = start_sample + snip_len

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
    max_amp = np.max(amplitudes)
    total_amp = np.sum(amplitudes)

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
        red_tc = vision_params.get_data_for_cell(cell_id, "RedTimeCourse")
        green_tc = vision_params.get_data_for_cell(cell_id, "GreenTimeCourse")
        blue_tc = vision_params.get_data_for_cell(cell_id, "BlueTimeCourse")
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
            red_tc = np.mean(r[y0:y1, x0:x1], axis=(0, 1))
            green_tc = np.mean(g[y0:y1, x0:x1], axis=(0, 1))
            blue_tc = np.mean(b[y0:y1, x0:x1], axis=(0, 1))
        else:
            # No fit: extract from the peak pixel of the red channel
            peak_idx = np.unravel_index(np.argmax(np.abs(r)), r.shape)
            yi, xi = peak_idx[0], peak_idx[1]
            red_tc, green_tc, blue_tc = r[yi, xi, :], g[yi, xi, :], b[yi, xi, :]

        timecourse_matrix = np.stack([red_tc, green_tc, blue_tc], axis=1)

    if timecourse_matrix is None:
        return None, None, None

    n_timepoints = timecourse_matrix.shape[0]
    refresh_ms = (
        getattr(sta_data, "refresh_time", 1000.0 / 60.0) if sta_data else 1000.0 / 60.0
    )
    time_axis = np.linspace(-(n_timepoints - 1) * refresh_ms, 0, n_timepoints)

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
            dom_idx = 2  # treat B/W as blue
            dom_trace = tc_matrix[:, 0]
        else:
            energies = np.sum(tc_matrix**2, axis=0)
            # Intercept duplicated B/W signal across R/G/B and force Blue
            if np.isclose(energies[0], energies[2], rtol=1e-5):
                dom_idx = 2
            else:
                dom_idx = int(np.argmax(energies))
            dom_trace = tc_matrix[:, dom_idx]

        channel_names = ["Red", "Green", "Blue"]
        dom_name = "Blue (B/W)" if tc_matrix.shape[1] == 1 else channel_names[dom_idx]

        # Smooth once — every derived quantity uses this trace
        sigma_samples = max(1, int(0.02 * len(dom_trace)))
        smoothed = gaussian_filter1d(dom_trace, sigma=sigma_samples)

        abs_max = np.max(np.abs(smoothed))
        if abs_max > 0:
            norm = smoothed / abs_max
        else:
            norm = smoothed.copy()

        # Polarity
        peak_val = np.max(norm)
        trough_val = np.min(norm)
        is_off = abs(trough_val) > abs(peak_val)
        polarity = "OFF" if is_off else "ON"

        # Primary peak index and time
        primary_idx = int(np.argmin(norm) if is_off else np.argmax(norm))
        primary_val = norm[primary_idx]
        peak_ms = float(time_axis[primary_idx])

        # FWHM  ── single calculation, used by both table and plot annotation
        fwhm_ms = float("nan")
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
                    fwhm_ms = float(widths[0] * sample_interval)
                    # Compute real-time endpoints for the plot annotation
                    fwhm_t_start = float(time_axis[0] + left_ips[0] * sample_interval)
                    fwhm_t_end = float(time_axis[0] + right_ips[0] * sample_interval)
                    fwhm_h = float(width_heights[0] * (-1 if is_off else 1))
                else:
                    fwhm_t_start = fwhm_t_end = fwhm_h = float("nan")
            else:
                fwhm_t_start = fwhm_t_end = fwhm_h = float("nan")
        except Exception as exc:
            logger.debug("FWHM calculation failed: %s", exc)
            fwhm_t_start = fwhm_t_end = fwhm_h = float("nan")

        # Biphasic index
        post = norm[primary_idx + 1 :]
        if len(post) > 0:
            secondary_val = float(np.max(post) if is_off else np.min(post))
            biphasic = abs(secondary_val / primary_val) if primary_val != 0 else 0.0
        else:
            biphasic = 0.0

        # SNR: response std vs. early-baseline std
        baseline_len = int(0.25 * len(norm))
        if baseline_len > 5:
            baseline_std = float(np.std(norm[:baseline_len]))
            response_std = float(np.std(norm[baseline_len:]))
            snr = response_std / baseline_std if baseline_std > 0 else float("inf")
        else:
            snr = float("nan")

        # ── Store display strings ────────────────────────────────────────────
        metrics["Dominant Channel"] = dom_name
        metrics["Polarity"] = polarity
        metrics["Peak (ms)"] = f"{peak_ms:.1f}"
        metrics["FWHM (ms)"] = f"{fwhm_ms:.1f}" if not np.isnan(fwhm_ms) else "N/A"
        metrics["Biphasic Index"] = f"{biphasic:.3f}"
        metrics["SNR"] = f"{snr:.2f}" if not np.isnan(snr) else "N/A"

        # ── Raw floats for plot annotation ───────────────────────────────────
        # The temporal filter plot reads these directly so it never has to
        # recompute — table and annotation are guaranteed identical.
        metrics["_raw_temporal"] = {
            "dom_idx": dom_idx,
            "dom_name": dom_name,
            "is_off": is_off,
            "norm_trace": norm,  # smoothed, normalised
            "raw_tc": tc_matrix,  # all channels, un-normalised
            "time_axis": time_axis,
            "primary_idx": primary_idx,
            "peak_ms": peak_ms,
            "peak_val": float(primary_val),
            "fwhm_ms": fwhm_ms,
            "fwhm_t_start": fwhm_t_start,
            "fwhm_t_end": fwhm_t_end,
            "fwhm_h": fwhm_h,
            "biphasic": biphasic,
            "snr": snr,
            "source": source,
        }

    # ── Spatial block ────────────────────────────────────────────────────────
    if stafit:
        sx, sy = stafit.std_x, stafit.std_y

        orientation_deg = np.rad2deg(stafit.rot) % 180  # keep in [0, 180)
        area = np.pi * sx * sy
        ellipticity = (sy / sx) if sx > 0 else float("inf")

        metrics["RF σx (stix)"] = f"{sx:.2f}"
        metrics["RF σy (stix)"] = f"{sy:.2f}"
        metrics["RF Area (stix²)"] = f"{area:.1f}"
        metrics["Orientation (°)"] = f"{orientation_deg:.1f}"
        metrics["Ellipticity"] = (
            f"{ellipticity:.2f}" if not np.isinf(ellipticity) else "∞"
        )

    return metrics


# ---------------------------------------------------------------------------
# EI spatial waveform plot
# ---------------------------------------------------------------------------


def plot_ei_waveforms(
    ei,
    positions,
    ref_channel=None,
    scale=1.0,
    ax=None,
    colors="white",
    alpha=1.0,
    linewidth=0.5,
    box_height=None,
    box_width=None,
    aspect=1.0,
    min_global_max=None,
):
    """
    Paint one or more EI waveforms onto a matplotlib Axes at their spatial
    electrode positions.  Designed to be called on an existing Axes that
    already shows a photo underlay and electrode scatter — waveforms are
    drawn in data (micron) space so they register correctly.

    Parameters
    ----------
    ei : np.ndarray or list of np.ndarray
        Shape (n_ch, T) per array.  All arrays must share the same n_ch and T.
    positions : np.ndarray
        Shape (n_ch, 2), electrode xy positions in microns.  Row i must
        correspond to ei[i, :].
    ref_channel : int, optional
        Channel index to highlight (thicker, fully opaque trace).
    scale : float
        Vertical scaling relative to box_height.  1.0 = largest waveform
        fills the box exactly.
    ax : matplotlib.axes.Axes, optional
        Target axes.  Defaults to plt.gca().
    colors : str or list of str
        Single colour or one colour per EI array.
    alpha : float or list of float
        Global alpha, or one value per EI array.
    linewidth : float or list of float
        Line width, or one value per EI array.
    box_height : float, optional
        Vertical extent of each waveform box in microns.  Defaults to 80 % of
        the median nearest-neighbour y-spacing so it is derived automatically
        from the actual array geometry rather than hardcoded.
    box_width : float, optional
        Horizontal extent of the waveform time axis in microns.  Defaults to
        the same value as box_height.
    aspect : float
        Axes aspect ratio passed to ax.set_aspect().
    min_global_max : float, optional
        Floor for the global absolute maximum used in normalisation.  Useful
        to prevent tiny-amplitude noise channels from dominating.

    Notes
    -----
    * Pure numpy/matplotlib — no Qt, no I/O.
    * Does not call ax.set_xlim / ax.set_ylim so the caller's axis limits are
      preserved.  The caller should set limits before calling this function.
    * Returns the list of Line2D artists added so the caller can remove them
      later (e.g. on the next selection change).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is required for plot_ei_waveforms")
        return []

    if ax is None:
        ax = plt.gca()

    # ── derive box geometry from array spacing if not supplied ───────────────
    if box_height is None:
        y_vals = np.unique(positions[:, 1])
        if len(y_vals) >= 2:
            spacing = float(np.median(np.diff(np.sort(y_vals))))
        else:
            spacing = 30.0  # safe fallback
        box_height = spacing * 0.8

    if box_width is None:
        box_width = box_height

    # ── normalise inputs ─────────────────────────────────────────────────────
    if isinstance(ei, np.ndarray):
        eis = [ei]
    else:
        eis = list(ei)

    if isinstance(colors, str):
        colors = [colors] * len(eis)
    if not isinstance(alpha, (list, np.ndarray)):
        alpha = [alpha] * len(eis)
    if not isinstance(linewidth, (list, np.ndarray)):
        linewidth = [linewidth] * len(eis)

    # ── global amplitude normalisation ───────────────────────────────────────
    observed_max = max(float(np.max(np.abs(e))) for e in eis)
    if min_global_max is not None:
        global_max = max(observed_max, float(min_global_max))
    else:
        global_max = observed_max

    if global_max <= 0:
        return []

    # time axis in micron units, centred on the electrode x position
    t = np.linspace(-0.5, 0.5, eis[0].shape[1]) * box_width

    added_artists = []

    for ei_array, color, this_alpha, this_lw in zip(eis, colors, alpha, linewidth):
        norm_ei = (ei_array / global_max) * scale * box_height

        # per-channel peak-to-peak for dimming near-silent channels
        p2ps = norm_ei.max(axis=1) - norm_ei.min(axis=1)
        p2p_thresh = 0.05 * p2ps.max()

        for i in range(ei_array.shape[0]):
            if i >= len(positions):
                break  # guard against shape mismatch
            x_offset, y_offset = positions[i]

            if p2ps[i] < p2p_thresh:
                ch_alpha = 0.25
                ch_lw = 0.3
            else:
                ch_alpha = this_alpha
                ch_lw = this_lw

            if ref_channel is not None and int(i) == int(ref_channel):
                ch_alpha = 1.0
                ch_lw = this_lw * 3

            lines = ax.plot(
                t + x_offset,
                norm_ei[i] + y_offset,
                color=color,
                alpha=ch_alpha,
                linewidth=ch_lw,
                zorder=7,  # above photo (0), scatter (2), lasso poly (4)
                rasterized=True,
            )
            added_artists.extend(lines)

    return added_artists


# ---------------------------------------------------------------------------
# Dynamic clustering helpers — pure numpy, no Qt, no I/O
# ---------------------------------------------------------------------------


def peak_align_timecourse(tc):
    """
    Shift a 1-D timecourse so its absolute peak sits at the centre index.

    If the timecourse is flat (``np.std(tc) < DEFAULT_MIN_STA_STD``), a copy
    is returned unchanged — ``np.roll`` is **not** applied because argmax on
    a constant array is arbitrary and would introduce meaningless jitter.

    Parameters
    ----------
    tc : np.ndarray
        1-D array — the temporal STA trace for one cell.

    Returns
    -------
    np.ndarray
        Aligned *copy* of *tc*.  The original is never mutated.
    """
    from .constants import DEFAULT_MIN_STA_STD

    tc = np.asarray(tc, dtype=np.float64).copy()
    if tc.ndim != 1 or len(tc) == 0:
        return tc

    if np.std(tc) < DEFAULT_MIN_STA_STD:
        return tc  # flat — nothing to align

    centre = len(tc) // 2
    peak_idx = int(np.argmax(np.abs(tc)))
    shift = centre - peak_idx
    if shift != 0:
        tc = np.roll(tc, shift)
    return tc


def apply_prefilter(physics_entries, filter_config):
    """
    Partition cluster IDs into valid and discarded based on thresholds.

    A cell is discarded when **any** of the following hold:

    * ``physics['timecourse']`` is not None AND
      ``np.std(physics['timecourse']) < filter_config['min_sta_std']``
      (STA exists but is flat/uninformative — likely a noise unit)
    * ``physics['rf_area'] > 0`` AND
      ``physics['rf_area'] > filter_config['max_rf_area']``

    Note: ``timecourse is None`` is **not** a rejection criterion.
    Cells without STA data (Vision .sta file absent, or cell not fitted)
    are included and embedded using ACG + scalar features only.

    Parameters
    ----------
    physics_entries : dict
        ``{cluster_id: physics_dict}`` where each physics_dict has at least
        keys ``'timecourse'`` (1-D array or None) and ``'rf_area'`` (float).
    filter_config : dict
        ``{'min_sta_std': float, 'max_rf_area': float}``

    Returns
    -------
    (valid_ids, discarded_ids) : (list[int], list[int])
        Both lists are sorted for deterministic ordering.
    """
    min_sta_std = filter_config.get("min_sta_std", 1e-5)
    max_rf_area = filter_config.get("max_rf_area", 300.0)

    valid_ids = []
    discarded_ids = []

    for cid, phys in physics_entries.items():
        tc = phys.get("timecourse")

        # --- reject only if STA exists but is flat/uninformative ---
        # timecourse=None means no STA data, which is fine — cell still
        # contributes ACG + scalar features to the embedding.
        if tc is not None:
            tc = np.asarray(tc, dtype=np.float64)
            if np.std(tc) < min_sta_std:
                discarded_ids.append(cid)
                continue

        # --- reject if RF area is explicitly fitted and too large ---
        # rf_area=0.0 is the default when no fit exists, not a real measurement,
        # so skip the area gate entirely when area is zero.
        rf_area = float(phys.get("rf_area", 0.0))
        if rf_area > 0.0 and rf_area > max_rf_area:
            discarded_ids.append(cid)
            continue

        valid_ids.append(cid)

    return sorted(valid_ids), sorted(discarded_ids)


def build_feature_matrix(raw_blocks, feature_config):
    """
    Transform raw feature arrays into a weighted, PCA-reduced feature matrix.

    Pipeline
    --------
    1. **Temporal STA** (if enabled): peak-align each row via
       :func:`peak_align_timecourse`, PCA to ``TEMPORAL_PCA_COMPONENTS``
       components, multiply by ``w_temporal``.
    2. **ACG** (if enabled): PCA to ``ACG_PCA_COMPONENTS`` components,
       multiply by ``w_acg``.
    3. **Scalars** (each independently toggleable): ``RobustScaler``, multiply
       by the per-feature weight.
    4. ``np.hstack`` all enabled blocks.

    Parameters
    ----------
    raw_blocks : dict
        ``{'temporal': np.ndarray (N, T),
          'acg':       np.ndarray (N, A),
          'scalars':   pd.DataFrame (N rows)}``
    feature_config : dict
        Boolean ``use_*`` flags and float ``w_*`` weights for every feature.

    Returns
    -------
    (matrix, column_labels) : (np.ndarray (N, D), list[str])

    Raises
    ------
    ValueError
        If all features are disabled (D would be 0).
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from .constants import (
        TEMPORAL_PCA_COMPONENTS,
        ACG_PCA_COMPONENTS,
        GRATING_PCA_COMPONENTS,
        CHIRP_PCA_COMPONENTS,
    )

    N = raw_blocks["temporal"].shape[0]  # guaranteed same N across all blocks
    blocks = []
    labels = []

    # ── Temporal STA ─────────────────────────────────────────────────────────
    if feature_config.get("use_temporal", True):
        w = feature_config.get("w_temporal", 3.0)
        tc_matrix = raw_blocks["temporal"].copy()

        # Guard: skip the entire temporal block when no STA data is available.
        # get_raw_feature_blocks fills timecourse=None cells with a zero-length
        # sentinel row; after padding they become all-zero rows of width 1.
        # np.std==0 on such a matrix means PCA and StandardScaler would either
        # produce NaNs (zero-variance columns) or meaningless components.
        sta_available = tc_matrix.shape[1] > 1 and np.std(tc_matrix) > 0
        if sta_available:
            # Peak-align each row
            for i in range(tc_matrix.shape[0]):
                tc_matrix[i] = peak_align_timecourse(tc_matrix[i])

            n_comp = min(TEMPORAL_PCA_COMPONENTS, N - 1, tc_matrix.shape[1])
            if n_comp >= 1:
                tc_pca = PCA(n_components=n_comp).fit_transform(tc_matrix)
                # Z-score each PC independently so weight sliders have a
                # consistent meaning across feature groups (unit variance
                # before the user weight is applied).
                tc_pca = StandardScaler().fit_transform(tc_pca)
                blocks.append(tc_pca * w)
                labels.extend([f"tc_pc{j}" for j in range(n_comp)])
        else:
            logger.debug(
                "build_feature_matrix: skipping temporal STA block — "
                "no STA data available (all timecourses are None)."
            )

    # ── ACG ──────────────────────────────────────────────────────────────────
    if feature_config.get("use_acg", True):
        w = feature_config.get("w_acg", 2.0)
        acg_matrix = raw_blocks["acg"].copy()

        n_comp = min(ACG_PCA_COMPONENTS, N - 1, acg_matrix.shape[1])
        if n_comp >= 1:
            acg_pca = PCA(n_components=n_comp).fit_transform(acg_matrix)
            # Z-score so ACG PCs are on the same scale as temporal PCs
            # and scalar features — weight slider is then interpretable.
            acg_pca = StandardScaler().fit_transform(acg_pca)
            blocks.append(acg_pca * w)
            labels.extend([f"acg_pc{j}" for j in range(n_comp)])

    # ── Grating direction-tuning-curve shape ────────────────────────────────
    # PCA on grating_calc.pooled_direction_tuning_curve — a peak-weighted,
    # shape-normalized tuning curve pooled across every dsos condition per
    # cell (see that function's docstring for the pooling method) — rather
    # than the DSI/OSI scalar summary. DSI/OSI collapse an entire tuning
    # curve to "how concentrated around one harmonic," which can't
    # distinguish a narrow single-peaked curve from a broad lopsided one if
    # they happen to share a DSI value; PCA on the curve shape itself
    # preserves that distinction the same way temporal/ACG PCA preserve STA/
    # autocorrelogram shape rather than collapsing them to summary scalars.
    # Cells with no dsos conditions (or none with usable peak response) get
    # an all-zero sentinel row from get_raw_feature_blocks — same "guard
    # against zero-variance / all-zero matrix" pattern as the temporal
    # block above, not a per-cell special case here.
    if feature_config.get("use_grating_dsos", True):
        w = feature_config.get("w_grating_dsos", 3.0)
        grating_matrix = raw_blocks.get("grating")
        # Defensive: 'grating' is always present from the current
        # get_raw_feature_blocks, but this is a public API (__all__) so
        # guard rather than KeyError if a caller passes an older/partial
        # raw_blocks dict.
        if grating_matrix is None or grating_matrix.size == 0:
            grating_available = False
        else:
            grating_matrix = grating_matrix.copy()
            # Unlike temporal (which collapses to width-1 when no STA),
            # grating is always POOLED_CURVE_N_BINS wide — so the real
            # "no usable data" signal is an all-zero matrix (every cell got
            # the zero sentinel), i.e. np.std == 0. A width check would be
            # meaningless here.
            grating_available = np.std(grating_matrix) > 0

        if grating_available:
            n_comp = min(GRATING_PCA_COMPONENTS, N - 1, grating_matrix.shape[1])
            if n_comp >= 1:
                grating_pca = PCA(n_components=n_comp).fit_transform(grating_matrix)
                grating_pca = StandardScaler().fit_transform(grating_pca)
                blocks.append(grating_pca * w)
                labels.extend([f"grating_pc{j}" for j in range(n_comp)])
        else:
            logger.debug(
                "build_feature_matrix: skipping grating block — no cells "
                "with usable direction-tuning curves (all-zero matrix)."
            )

    # ── Chirp PSTH shape ─────────────────────────────────────────────────────
    # PCA on the L2-normalized chirp response PSTH
    # (get_chirp_data_for_cluster['psth_mean']), gated on availability. Same
    # "shape not scalar" rationale as the grating block above: the chirp QI
    # scalar collapses the whole ON/OFF-step + frequency/contrast-sweep
    # response into one reliability number, which can't tell an ON-transient
    # cell from a sustained-OFF cell that happen to share a QI. Cells with no
    # chirp response (missing, or QI below CHIRP_MIN_QI) get an all-zero
    # sentinel row from get_raw_feature_blocks — same std==0 guard as grating,
    # not a per-cell special case here. Absent entirely when no chirp file is
    # loaded (raw_blocks['chirp'] empty / missing key), so the block simply
    # disappears rather than breaking the embedding.
    if feature_config.get("use_chirp", True):
        w = feature_config.get("w_chirp", 3.0)
        chirp_matrix = raw_blocks.get("chirp")
        if chirp_matrix is None or chirp_matrix.size == 0:
            chirp_available = False
        else:
            chirp_matrix = chirp_matrix.copy()
            chirp_available = np.std(chirp_matrix) > 0

        if chirp_available:
            n_comp = min(CHIRP_PCA_COMPONENTS, N - 1, chirp_matrix.shape[1])
            if n_comp >= 1:
                chirp_pca = PCA(n_components=n_comp).fit_transform(chirp_matrix)
                chirp_pca = StandardScaler().fit_transform(chirp_pca)
                blocks.append(chirp_pca * w)
                labels.extend([f"chirp_pc{j}" for j in range(n_comp)])
        else:
            logger.debug(
                "build_feature_matrix: skipping chirp block — no cells with "
                "usable chirp responses (all-zero/absent matrix)."
            )

    # ── Scalar features ──────────────────────────────────────────────────────
    # Consolidated set: firing_rate, isi_violations, time_to_peak, and
    # ellipticity removed (see constants.py DEFAULT_WEIGHT_* comments).
    # RF area replaced by long/short axis diameters — two independent
    # scalars rather than one derived area+ratio pair, so PCA/UMAP can find
    # whatever size/shape relationship actually matters rather than the
    # relationship being pre-baked into a single ellipticity ratio.
    #
    # NOTE on weight semantics: w_rf_diameter is applied independently to
    # both rf_long_diameter/rf_short_diameter (one checkbox/slider controls
    # both columns at once). Since scaled Euclidean distance sums
    # weight**2 per column, a block's total contribution to distance scales
    # with (n_columns * weight**2), not weight alone.
    scalar_features = [
        ("rf_long_diameter", "use_rf_diameter", "w_rf_diameter", 6.0),
        ("rf_short_diameter", "use_rf_diameter", "w_rf_diameter", 6.0),
    ]

    scalars_df = raw_blocks["scalars"]
    enabled_scalars = []
    scalar_weights = []
    scalar_labels = []

    for col, use_key, w_key, default_w in scalar_features:
        if feature_config.get(use_key, True) and col in scalars_df.columns:
            vals = scalars_df[col].fillna(0.0).values.astype(np.float64)
            enabled_scalars.append(vals.reshape(-1, 1))
            scalar_weights.append(feature_config.get(w_key, default_w))
            scalar_labels.append(col)

    if enabled_scalars:
        scalar_block = np.hstack(enabled_scalars)
        scalar_block = RobustScaler().fit_transform(scalar_block)
        # Apply per-column weights
        for j, w in enumerate(scalar_weights):
            scalar_block[:, j] *= w
        blocks.append(scalar_block)
        labels.extend(scalar_labels)

    # ── Assembly ─────────────────────────────────────────────────────────────
    if not blocks:
        raise ValueError(
            "All features are disabled — cannot build a feature matrix with "
            "zero dimensions. Enable at least one feature."
        )

    matrix = np.hstack(blocks)
    return matrix, labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "extract_snippets",
    "baseline_correct",
    "compute_ei",
    "select_channels",
    "get_sta_timecourse_data",
    "compute_sta_metrics",
    "compute_spatial_features",
    "plot_ei_waveforms",
    "peak_align_timecourse",
    "apply_prefilter",
    "build_feature_matrix",
]
