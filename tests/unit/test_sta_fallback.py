"""RFs and time courses from the STA when .params has none; empty STA files (PLAN.md Q64)."""

import struct
from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis import analysis_core, rf_geometry as rg
from src.analysis.reference_bridge import ReferenceBridge
from src.analysis.vision_integration import sta_header, sta_is_empty

H, W, D = 20, 24, 30


def _sta(col=15.0, row=6.0, sx=2.5, sy=1.2, phi=0.4, off=True, seed=0):
    """An STA with one Gaussian RF at (col, row) in pixel indices, peak at frame 25."""
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:H, 0:W]
    u = (cols - col) * np.cos(phi) + (rows - row) * np.sin(phi)
    v = -(cols - col) * np.sin(phi) + (rows - row) * np.cos(phi)
    space = np.exp(-0.5 * (u ** 2 / sx ** 2 + v ** 2 / sy ** 2))
    time = np.exp(-0.5 * ((np.arange(D) - 25) / 2.0) ** 2) * (-1 if off else 1)
    cube = 0.05 * space[:, :, None] * time[None, None, :] + rng.normal(0, 0.002, (H, W, D))
    return SimpleNamespace(red=cube, green=cube.copy(), blue=cube.copy())


def test_fit_recovers_the_rf_in_visions_frame():
    fit = rg.fit_from_sta(_sta())
    # x0 = col + 0.5, H − y0 = row + 0.5 (the module's convention).
    assert fit.x0 == pytest.approx(15.5, abs=0.1)
    assert fit.y0 == pytest.approx(H - 6.5, abs=0.1)
    long_, short = max(fit.std_x, fit.std_y), min(fit.std_x, fit.std_y)
    assert long_ == pytest.approx(2.5, rel=0.1) and short == pytest.approx(1.2, rel=0.1)
    axis = fit.theta if fit.std_x >= fit.std_y else fit.theta + np.pi / 2
    assert abs(((np.degrees(axis) - np.degrees(0.4)) + 90) % 180 - 90) < 5
    # image_ellipse puts it back on the pixel it came from
    cx, cy, *_ = rg.image_ellipse(fit, H)
    assert (cx, cy) == pytest.approx((15.5, 6.5), abs=0.1)


def test_noise_and_nan_stas_get_no_fit():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, (H, W, D))
    assert rg.fit_from_sta(SimpleNamespace(red=noise, green=noise, blue=noise)) is None
    nan = np.full((H, W, D), np.nan)
    assert rg.fit_from_sta(SimpleNamespace(red=nan, green=nan, blue=nan)) is None


def test_empty_params_time_course_falls_back_to_the_sta():
    sta = _sta(off=True)
    params = SimpleNamespace(get_data_for_cell=lambda cid, key: np.zeros(0))
    _t, tc, source = analysis_core.get_sta_timecourse_data(sta, None, params, 1)
    assert source == "recalculated" and tc.shape == (D, 3)
    assert int(np.argmin(tc[:, 0])) == 25                     # an OFF cell's trough, at the RF
    nan = SimpleNamespace(red=np.full((H, W, D), np.nan), green=np.full((H, W, D), np.nan),
                          blue=np.full((H, W, D), np.nan))
    assert analysis_core.get_sta_timecourse_data(nan, None, params, 1) == (None, None, None)


def test_matched_cell_borrows_an_sta_fit_when_the_reference_params_has_none():
    nan_fit = SimpleNamespace(center_x=np.nan, center_y=np.nan, std_x=np.nan, std_y=np.nan, rot=np.nan)
    params = SimpleNamespace(get_stafit_for_cell=lambda cid: nan_fit,
                             get_data_for_cell=lambda cid, key: np.zeros(0))
    b = ReferenceBridge({11: 21}, ref_stas={21: _sta()}, ref_params=params,
                        confidence_scores={11: 0.9}, match_statuses={11: "high"})
    assert b.has_rf(11)
    assert b.precompute_rf_fits() == 1
    ell = b.get_rf_ellipse_params(11)
    assert ell["x0"] == pytest.approx(15.5, abs=0.1) and ell["y0"] == pytest.approx(H - 6.5, abs=0.1)
    stafit = b.get_stafit(11)
    assert np.isfinite(stafit.std_x) and stafit.std_x > 0


def _write_sta_header(path, refresh):
    with open(path, "wb") as fh:
        fh.write(struct.pack(">iiiii", 32, 10000, 16, 16, 30) + struct.pack(">dd", 86.0, refresh))


def test_an_sta_written_without_a_stimulus_is_empty(tmp_path):
    _write_sta_header(tmp_path / "data000.sta", float("nan"))
    _write_sta_header(tmp_path / "data002.sta", 16.58)
    assert sta_header(tmp_path / "data002.sta")["refresh_ms"] == pytest.approx(16.58)
    assert sta_is_empty(tmp_path / "data000.sta")
    assert not sta_is_empty(tmp_path / "data002.sta")
