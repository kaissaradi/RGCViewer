"""Vision fits that never left their start value are not fits (PLAN.md Q26).

vision7 ImageFrame.fit starts every (σx, σy) on a 1..3 grid; exactly
(1, 1) means the best fit was the first start, unmoved.
"""

from collections import namedtuple

import numpy as np
import pytest

from src.analysis import analysis_core, rf_geometry

STAFit = namedtuple("STAFit", ["center_x", "center_y", "std_x", "std_y", "rot"])


@pytest.mark.parametrize("sx, sy, unmoved", [
    (1.0, 1.0, True), (-1.0, 1.0, True), (1.0, 1.5, False), (1.0000001, 1.0, False),
    (2.3, 1.7, False), (np.nan, 1.0, False),
])
def test_fit_is_unmoved(sx, sy, unmoved):
    assert rf_geometry.fit_is_unmoved(sx, sy) is unmoved


def test_unmoved_fits_are_not_drawn_or_measured():
    assert rf_geometry.rf_fit_from_stafit(STAFit(10.0, 12.0, 1.0, 1.0, 0.0)) is None
    assert rf_geometry.rf_fit_from_stafit(STAFit(10.0, 12.0, 1.8, 1.2, 0.0)) is not None
    assert rf_geometry.rf_fit_from_params(
        {"x0": 1, "y0": 2, "std_x": 1.0, "std_y": 1.0, "angle": 0}) is None


def test_sta_metrics_say_no_fit():
    sta = type("S", (), {})()
    rng = np.random.default_rng(0)
    for ch in ("red", "green", "blue"):
        setattr(sta, ch, rng.normal(size=(8, 8, 30)))
    sta.refresh_time = 16.6
    m = analysis_core.compute_sta_metrics(sta, STAFit(4.0, 4.0, 1.0, 1.0, 0.0), None, 1)
    assert m["RF σx (stix)"] == "no fit" and "RF Area (stix²)" not in m


def test_physics_rows_saved_from_an_unmoved_fit_are_recomputed():
    from src.analysis.data_manager import DataManager
    dm = DataManager.__new__(DataManager)
    base = {"_computed": True, "timecourse": np.ones(3)}
    assert not dm._physics_entry_is_fresh(1, {**base, "rf_area": float(np.pi)})
    assert dm._physics_entry_is_fresh(1, {**base, "rf_area": 8.9})
    assert dm._physics_entry_is_fresh(1, {**base, "rf_area": 0.0})
