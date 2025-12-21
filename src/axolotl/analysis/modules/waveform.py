"""Waveform processing utilities."""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from typing import Tuple
from pathlib import Path


def extract_snippets(dat_path_or_memmap, spike_times,
                     window=(-20, 60), n_channels=512, dtype='int16'):
    """Extracts snippets of raw data. Accepts either a file path (str/Path)
    or a memory-mapped array (np.memmap / ndarray-like). Returns array of
    shape (n_channels, snip_len, n_spikes).
    """
    snip_len = int(window[1] - window[0])
    spike_count = len(spike_times)

    if spike_count == 0:
        return np.zeros((n_channels, snip_len, 0), dtype=np.float32)

    # Accept either a path or an existing memmap/ndarray
    if isinstance(dat_path_or_memmap, (str, Path)):
        raw_data = np.memmap(str(dat_path_or_memmap), dtype=dtype, mode='r')
        try:
            raw_data = raw_data.reshape(-1, n_channels)
        except Exception:
            # If reshape fails, try inferring number of channels
            raw_data = raw_data.reshape(-1, n_channels)
    else:
        # Assume it's an ndarray-like (memmap or already shaped array)
        raw_data = dat_path_or_memmap
        # If 1D memmap, reshape to (n_samples, n_channels)
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(-1, n_channels)

    total_samples = raw_data.shape[0]

    # Preallocate in spike-major order then transpose for return
    snips = np.zeros((spike_count, n_channels, snip_len), dtype=np.float32)

    # Ensure integer spike times
    spike_times = np.asarray(spike_times, dtype=np.int64)

    for i, spike_time in enumerate(spike_times):
        start_sample = int(spike_time) + int(window[0])
        end_sample = start_sample + snip_len

        # Skip out-of-bounds spikes
        if start_sample < 0 or end_sample > total_samples:
            continue

        snippet = raw_data[start_sample:end_sample, :]
        # snippet shape: (snip_len, n_channels) -> transpose to (n_channels,
        # snip_len)
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