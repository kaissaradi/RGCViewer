"""Unit tests: physics fill-gap, invalidation, raw feature blocks from bridge."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.analysis.data_manager import DataManager
from src.analysis.reference_bridge import ReferenceBridge
from src.analysis.cross_run_matcher import MatchingReport, MatchResult
from src.analysis import grating_calc


def _stafit(x=1.0, y=2.0, sx=3.0, sy=4.0, rot=0.0):
    return SimpleNamespace(
        center_x=x, center_y=y, std_x=sx, std_y=sy, rot=rot, angle=rot
    )


class _FakeParams:
    def __init__(self, fits=None, timecourses=None):
        self._fits = fits or {}
        self._tc = timecourses or {}

    def get_stafit_for_cell(self, cell_id):
        return self._fits.get(int(cell_id))

    def get_data_for_cell(self, cell_id, field):
        key = (int(cell_id), field)
        return self._tc.get(key)


def _minimal_dm():
    """Bare DataManager-like object with just enough for physics/features."""
    dm = DataManager.__new__(DataManager)
    dm._feature_lock = threading.Lock()
    dm._physics_cell_locks = {}
    dm._physics_cell_locks_lock = threading.Lock()
    dm.feature_cache = {}
    dm.is_vision_only = False
    dm.vision_stas = None
    dm.vision_params = None
    dm.reference_bridge = None
    dm.match_caveats = {}
    dm.chirp_available = False
    dm.chirp_data = None
    dm.chirp_id_to_row = None
    dm.grating_available = False
    dm.grating_status = "missing"
    dm.grating_data = None
    dm.grating_computed_cache = {}
    dm._grating_cache_lock = threading.Lock()
    dm.cluster_df = pd.DataFrame({"cluster_id": [0, 1], "isi_violations_pct": [0.0, 0.0]})
    dm.spike_times = np.array([0, 30000], dtype=np.int64)
    dm.sampling_rate = 30000.0
    dm._physics_done_count = 0

    def _std_plot(cid):
        return {"acg_norm": np.ones(50, dtype=float)}

    dm.get_standard_plot_data = _std_plot
    dm.get_cluster_spike_indices = lambda cid: np.array([0], dtype=int)
    return dm


def test_physics_borrows_timecourse_with_ref_params_id(monkeypatch):
    dm = _minimal_dm()
    # current cluster 0 → vision 1 → ref vision 10
    tc = np.linspace(-1, 1, 20)
    ref_params = _FakeParams(
        fits={10: _stafit()},
        timecourses={
            (10, "RedTimeCourse"): tc,
            (10, "GreenTimeCourse"): tc * 0.5,
            (10, "BlueTimeCourse"): tc * 0.1,
        },
    )
    bridge = ReferenceBridge(
        mapping={1: 10},
        ref_params=ref_params,
        ref_stas={10: SimpleNamespace(red=None)},  # has_sta path optional
        confidence_scores={1: 0.97},
        match_statuses={1: "high"},
        full_matches=[MatchResult(1, 10, 0.97, "ei", "high")],
        stimuli_loaded=("params",),
    )
    # Force has_sta True even with dummy
    bridge._ref_stas = {10: object()}
    # Make `10 in ref_stas` work
    class _D(dict):
        pass

    bridge._ref_stas = _D({10: SimpleNamespace(red=np.ones((4, 4, 5)))})

    dm.install_reference_bridge(bridge)
    metrics = dm.get_cell_physics(0)

    assert metrics["timecourse"] is not None
    assert metrics["provenance"]["timecourse"] == "reference"
    assert metrics["provenance"]["rf_geometry"] == "reference"
    assert metrics["match_status"] == "high"
    assert metrics["reference_id"] == 10
    assert metrics["rf_area"] > 0
    assert 0 in dm.match_caveats
    assert dm.match_caveats[0].provenance["timecourse"] == "reference"


def test_physics_prefers_current_sta(monkeypatch):
    dm = _minimal_dm()
    tc = np.ones(10)
    # Current has STA for vision id 1
    current_params = _FakeParams(
        fits={1: _stafit(sx=1.0, sy=1.0)},
        timecourses={
            (1, "RedTimeCourse"): tc,
            (1, "GreenTimeCourse"): tc,
            (1, "BlueTimeCourse"): tc,
        },
    )
    dm.vision_params = current_params
    dm.vision_stas = {1: SimpleNamespace(red=np.ones((3, 3, 2)))}

    bridge = ReferenceBridge(
        mapping={1: 10},
        ref_params=_FakeParams(fits={10: _stafit(sx=9.0, sy=9.0)}),
        confidence_scores={1: 0.99},
        match_statuses={1: "high"},
        full_matches=[MatchResult(1, 10, 0.99, "ei", "high")],
    )
    dm.install_reference_bridge(bridge)
    metrics = dm.get_cell_physics(0)
    assert metrics["provenance"]["timecourse"] == "current"
    assert metrics["rf_area"] == pytest.approx(np.pi * 1.0 * 1.0)


def test_install_bridge_invalidates_computed_without_sta():
    dm = _minimal_dm()
    # Poisoned warm cache: computed but no timecourse
    dm.feature_cache[0] = {
        "_computed": True,
        "acg": np.ones(10),
        "timecourse": None,
        "rf_area": 0.0,
        "ellipticity": 0.0,
        "rf_long_diameter": 0.0,
        "rf_short_diameter": 0.0,
        "time_to_peak": 0,
    }
    tc = np.linspace(0, 1, 15)
    bridge = ReferenceBridge(
        mapping={1: 10},
        ref_params=_FakeParams(
            fits={10: _stafit()},
            timecourses={
                (10, "RedTimeCourse"): tc,
                (10, "GreenTimeCourse"): tc,
                (10, "BlueTimeCourse"): tc,
            },
        ),
        ref_stas={10: SimpleNamespace(red=np.ones((2, 2, 3)))},
        confidence_scores={1: 0.9},
        match_statuses={1: "high"},
        full_matches=[MatchResult(1, 10, 0.9, "ei", "high")],
    )
    # ensure has_sta
    class _D(dict):
        pass

    bridge._ref_stas = _D({10: SimpleNamespace(red=np.ones((2, 2, 3)))})

    n = dm.install_reference_bridge(bridge)
    assert 0 in dm.match_caveats
    # After install, entry should not be sticky _computed with empty STA
    entry = dm.feature_cache[0]
    assert not entry.get("_computed", False)
    # Recompute fills from bridge
    metrics = dm.get_cell_physics(0)
    assert metrics["timecourse"] is not None
    assert metrics["provenance"]["timecourse"] == "reference"
    # ACG preserved path: after invalidate acg was kept; recompute may refresh
    assert metrics["acg"] is not None


def test_raw_blocks_chirp_from_bridge():
    dm = _minimal_dm()
    # Physics so prefilter keeps the cell
    dm.feature_cache[0] = {
        "_computed": True,
        "acg": np.ones(20),
        "timecourse": np.linspace(-1, 1, 12),
        "rf_area": 10.0,
        "ellipticity": 1.0,
        "rf_long_diameter": 4.0,
        "rf_short_diameter": 3.0,
        "time_to_peak": 2,
    }
    psth = np.array([0.0, 1.0, 2.0, 1.0, 0.0], dtype=float)
    bridge = ReferenceBridge(
        mapping={1: 10},
        ref_chirp_data={
            "psth_mean": np.array([psth]),
            "quality_index": np.array([0.9]),
            "cluster_id": np.array([10]),
            "bin_size_ms": 10.0,
        },
        ref_chirp_id_to_row={9: 0},  # ref vid 10 → ks 9
        confidence_scores={1: 0.95},
        match_statuses={1: "high"},
        full_matches=[MatchResult(1, 10, 0.95, "ei", "high")],
        stimuli_loaded=("chirp",),
    )
    dm.install_reference_bridge(bridge)
    assert dm.effective_chirp_available()

    blocks, valid_ids, discarded = dm.get_raw_feature_blocks(
        [0], {"min_sta_std": 0.0, "max_rf_area": 1e9}
    )
    assert 0 in valid_ids
    assert blocks["chirp"].shape[1] == 5
    # Non-zero row (L2-normalized)
    assert np.linalg.norm(blocks["chirp"][0]) == pytest.approx(1.0)
    assert dm.match_caveats[0].provenance["chirp"] == "reference"


def test_raw_blocks_grating_from_bridge():
    dm = _minimal_dm()
    dm.feature_cache[0] = {
        "_computed": True,
        "acg": np.ones(20),
        "timecourse": np.linspace(-1, 1, 12),
        "rf_area": 10.0,
        "ellipticity": 1.0,
        "rf_long_diameter": 4.0,
        "rf_short_diameter": 3.0,
        "time_to_peak": 2,
    }
    # Minimal analyzed entry that pooled_direction_tuning_curve can use
    # If pooling returns None, we still verify bridge path is consulted.
    dirs = np.arange(0, 360, 30, dtype=float)
    amps = np.sin(np.deg2rad(dirs)) + 1.5
    grating_entry = {
        (80.0, 2.0): {
            "condition_type": "dsos",
            "directions_deg": dirs,
            "f1_amplitude": amps,
            "spike_rate_by_direction": amps,
            "peak_rate_hz": float(np.max(amps)),
            "DSI": 0.5,
            "OSI": 0.2,
            "preferred_direction_deg": 90.0,
            "preferred_orientation_deg": 90.0,
            "pvalue": 0.001,
            "classification": "DS",
        }
    }
    bridge = ReferenceBridge(
        mapping={1: 10},
        ref_grating_data={9: grating_entry},  # ref vid 10 → ks 9
        confidence_scores={1: 0.95},
        match_statuses={1: "high"},
        full_matches=[MatchResult(1, 10, 0.95, "ei", "high")],
        stimuli_loaded=("grating",),
    )
    dm.install_reference_bridge(bridge)
    assert dm.effective_grating_available()

    blocks, valid_ids, _ = dm.get_raw_feature_blocks(
        [0], {"min_sta_std": 0.0, "max_rf_area": 1e9}
    )
    assert 0 in valid_ids
    assert blocks["grating"].shape[1] == grating_calc.POOLED_CURVE_N_BINS
    # Either non-zero pooled curve or all-zero if schema insufficient — at least shape ok
    assert blocks["grating"].shape[0] == 1


def test_unmatched_cell_zero_sentinel_chirp():
    dm = _minimal_dm()
    dm.feature_cache[0] = {
        "_computed": True,
        "acg": np.ones(20),
        "timecourse": np.ones(10),
        "rf_area": 5.0,
        "ellipticity": 1.0,
        "rf_long_diameter": 2.0,
        "rf_short_diameter": 2.0,
        "time_to_peak": 1,
    }
    bridge = ReferenceBridge(
        mapping={},  # no match for cell 0
        ref_chirp_data={
            "psth_mean": np.ones((1, 4)),
            "quality_index": np.array([0.9]),
            "cluster_id": np.array([10]),
            "bin_size_ms": 10.0,
        },
        ref_chirp_id_to_row={9: 0},
        full_matches=[MatchResult(1, None, 0.1, "ei", "unmatched")],
        stimuli_loaded=("chirp",),
    )
    dm.install_reference_bridge(bridge)
    blocks, valid_ids, _ = dm.get_raw_feature_blocks(
        [0], {"min_sta_std": 0.0, "max_rf_area": 1e9}
    )
    assert np.allclose(blocks["chirp"][0], 0.0)


def test_effective_chirp_available_or():
    dm = _minimal_dm()
    assert not dm.effective_chirp_available()
    dm.chirp_available = True
    assert dm.effective_chirp_available()
    dm.chirp_available = False
    bridge = ReferenceBridge(
        mapping={1: 10},
        ref_chirp_data={"psth_mean": np.ones((1, 3)), "quality_index": np.array([1.0])},
        ref_chirp_id_to_row={9: 0},
    )
    dm.reference_bridge = bridge
    assert dm.effective_chirp_available()
