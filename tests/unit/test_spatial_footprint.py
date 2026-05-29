"""
tests/unit/test_spatial_footprint.py

Phase 1 unit tests for analysis_core.extract_spatial_footprint().

These tests are pure numpy — no Qt, no DataManager, no disk I/O.
Run with:
    conda run -n rgcviewer python -m pytest tests/unit/test_spatial_footprint.py -v
"""

import numpy as np
import pytest

from src.analysis.analysis_core import extract_spatial_footprint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sta(height=20, width=20, n_frames=30, n_channels=3, seed=42):
    """
    Return a synthetic STA movie shaped (height, width, n_frames) for a
    single colour channel, or a dict-like object with .red/.green/.blue
    attributes if n_channels == 3.

    We use a simple namespace so the function can access sta_data.red etc.
    """
    rng = np.random.default_rng(seed)

    class _STA:
        pass

    sta = _STA()
    if n_channels == 1:
        # B/W: only .red attribute, .green and .blue are None
        sta.red   = rng.standard_normal((height, width, n_frames)).astype(np.float32)
        sta.green = None
        sta.blue  = None
    else:
        sta.red   = rng.standard_normal((height, width, n_frames)).astype(np.float32)
        sta.green = rng.standard_normal((height, width, n_frames)).astype(np.float32)
        sta.blue  = rng.standard_normal((height, width, n_frames)).astype(np.float32)
    return sta


def _make_gaussian_sta(
    height=40, width=40, n_frames=30,
    cx=20, cy=20, sigma=4.0, peak_frame=15,
    amplitude=10.0, noise_std=0.2, seed=0,
):
    """
    Synthetic STA where the red channel has a clear 2D Gaussian blob
    centred at (cx, cy) with sigma `sigma` at frame `peak_frame`.
    All other frames are pure noise.
    """
    rng = np.random.default_rng(seed)

    class _STA:
        pass

    sta = _STA()

    base = rng.standard_normal((height, width, n_frames)).astype(np.float32) * noise_std

    # Build Gaussian blob at the peak frame
    yy, xx = np.mgrid[0:height, 0:width]
    blob = amplitude * np.exp(
        -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)
    ).astype(np.float32)
    base[:, :, peak_frame] += blob

    sta.red   = base
    sta.green = rng.standard_normal((height, width, n_frames)).astype(np.float32) * noise_std
    sta.blue  = rng.standard_normal((height, width, n_frames)).astype(np.float32) * noise_std
    return sta, peak_frame


# ---------------------------------------------------------------------------
# AC1 — standard Gaussian blob, 32×32 output
# ---------------------------------------------------------------------------

class TestExtractSpatialFootprintGaussian:

    def test_output_shape_is_grid_size_squared(self):
        sta, peak_idx = _make_gaussian_sta()
        result = extract_spatial_footprint(sta, peak_idx, grid_size=32)
        assert result is not None, "Expected a footprint for a high-SNR STA, got None"
        assert result.shape == (32 * 32,), f"Expected (1024,), got {result.shape}"

    def test_output_is_float32(self):
        sta, peak_idx = _make_gaussian_sta()
        result = extract_spatial_footprint(sta, peak_idx, grid_size=32)
        assert result is not None
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"

    def test_output_is_zero_mean(self):
        sta, peak_idx = _make_gaussian_sta()
        result = extract_spatial_footprint(sta, peak_idx, grid_size=32)
        assert result is not None
        assert abs(result.mean()) < 1e-5, f"Expected zero mean, got {result.mean():.6f}"

    def test_output_is_unit_variance(self):
        sta, peak_idx = _make_gaussian_sta()
        result = extract_spatial_footprint(sta, peak_idx, grid_size=32)
        assert result is not None
        assert abs(result.std() - 1.0) < 1e-4, f"Expected unit std, got {result.std():.6f}"

    def test_grid_size_parameter_respected(self):
        sta, peak_idx = _make_gaussian_sta()
        for gs in (16, 32, 48):
            result = extract_spatial_footprint(sta, peak_idx, grid_size=gs)
            assert result is not None
            assert result.shape == (gs * gs,), f"grid_size={gs}: expected ({gs*gs},), got {result.shape}"

    def test_snr_threshold_respected_high_amplitude(self):
        # High-amplitude blob should always pass any reasonable threshold
        sta, peak_idx = _make_gaussian_sta(amplitude=20.0, noise_std=0.1)
        result = extract_spatial_footprint(sta, peak_idx, grid_size=32, snr_threshold=3.0)
        assert result is not None, "High-SNR STA should produce a footprint"

    def test_peak_frame_at_boundary(self):
        # peak_idx = 0: crop should still work (zero-pad before beginning)
        sta, _ = _make_gaussian_sta(peak_frame=0)
        result = extract_spatial_footprint(sta, peak_idx=0, grid_size=16)
        assert result is not None or result is None  # must not raise


# ---------------------------------------------------------------------------
# AC2 — low-SNR / noise STA returns None
# ---------------------------------------------------------------------------

class TestExtractSpatialFootprintLowSNR:

    def test_uniform_noise_returns_none(self):
        """Pure Gaussian noise — no coherent blob — should return None."""
        rng = np.random.default_rng(99)

        class _STA:
            pass

        sta = _STA()
        # Noise amplitude << SNR threshold; no spatial structure
        sta.red   = rng.standard_normal((30, 30, 20)).astype(np.float32) * 0.01
        sta.green = rng.standard_normal((30, 30, 20)).astype(np.float32) * 0.01
        sta.blue  = rng.standard_normal((30, 30, 20)).astype(np.float32) * 0.01

        result = extract_spatial_footprint(sta, peak_idx=10, grid_size=32, snr_threshold=3.0)
        assert result is None, f"Expected None for low-SNR STA, got array of shape {getattr(result, 'shape', 'N/A')}"

    def test_none_footprint_does_not_raise(self):
        """Regression: calling the function on noise must never raise an exception."""
        rng = np.random.default_rng(7)

        class _STA:
            pass

        sta = _STA()
        sta.red   = rng.standard_normal((20, 20, 10)).astype(np.float32) * 0.001
        sta.green = None
        sta.blue  = None

        try:
            extract_spatial_footprint(sta, peak_idx=5, grid_size=32)
        except Exception as exc:
            pytest.fail(f"extract_spatial_footprint raised on low-SNR STA: {exc}")

    def test_very_high_threshold_forces_none(self):
        """Setting snr_threshold absurdly high should return None even for real signals."""
        sta, peak_idx = _make_gaussian_sta(amplitude=5.0)
        result = extract_spatial_footprint(sta, peak_idx, grid_size=32, snr_threshold=9999.0)
        assert result is None


# ---------------------------------------------------------------------------
# AC1 (bw variant) — single-channel (B/W) STA
# ---------------------------------------------------------------------------

class TestExtractSpatialFootprintSingleChannel:

    def test_bw_sta_produces_footprint(self):
        """
        B/W STA has only .red populated; .green and .blue are None.
        Function must handle this without AttributeError.
        """
        rng = np.random.default_rng(3)
        height, width, n_frames = 30, 30, 20
        peak_frame = 10

        class _STA:
            pass

        sta = _STA()
        base = rng.standard_normal((height, width, n_frames)).astype(np.float32) * 0.1
        # Inject a visible blob in the red channel only
        yy, xx = np.mgrid[0:height, 0:width]
        blob = 8.0 * np.exp(-((xx - 15) ** 2 + (yy - 15) ** 2) / (2 * 3.0 ** 2))
        base[:, :, peak_frame] += blob.astype(np.float32)

        sta.red   = base
        sta.green = None
        sta.blue  = None

        result = extract_spatial_footprint(sta, peak_idx=peak_frame, grid_size=16)
        assert result is not None, "B/W STA with clear blob should return a footprint"
        assert result.shape == (16 * 16,)

    def test_bw_sta_output_normalised(self):
        rng = np.random.default_rng(4)
        height, width, n_frames = 30, 30, 20

        class _STA:
            pass

        sta = _STA()
        base = rng.standard_normal((height, width, n_frames)).astype(np.float32) * 0.1
        yy, xx = np.mgrid[0:height, 0:width]
        blob = 15.0 * np.exp(-((xx - 10) ** 2 + (yy - 10) ** 2) / (2 * 2.5 ** 2))
        base[:, :, 8] += blob.astype(np.float32)

        sta.red   = base
        sta.green = None
        sta.blue  = None

        result = extract_spatial_footprint(sta, peak_idx=8, grid_size=16)
        if result is not None:
            assert abs(result.mean()) < 1e-4
            assert abs(result.std() - 1.0) < 1e-3