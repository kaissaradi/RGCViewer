"""Spatial/EI analysis."""
import numpy as np
from scipy.signal import correlate
import torch


def compute_spatial_features(ei, channel_positions, sampling_rate):
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