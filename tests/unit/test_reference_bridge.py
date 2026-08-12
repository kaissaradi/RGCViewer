"""Unit tests for ReferenceBridge stimulus load, remapping, and caveats."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from src.analysis.reference_bridge import (
    ReferenceBridge,
    CellMatchCaveat,
    vision_id_to_cluster_id,
    cluster_id_to_vision_id,
)
from src.analysis.cross_run_matcher import MatchingReport, MatchResult


def _make_report(matches):
    report = MatchingReport(
        current_run_path="/cur",
        reference_run_path="/ref",
    )
    report.matches = matches
    return report


class _FakeSTADict(dict):
    def __contains__(self, key):
        return dict.__contains__(self, int(key))


class _FakeParams:
    def __init__(self, fits):
        self._fits = fits  # {vision_id: stafit}

    def get_stafit_for_cell(self, cell_id):
        return self._fits.get(int(cell_id))

    def get_data_for_cell(self, cell_id, field):
        return None


def _stafit(x=10.0, y=20.0, sx=2.0, sy=3.0, rot=0.1):
    return SimpleNamespace(
        center_x=x, center_y=y, std_x=sx, std_y=sy, rot=rot, angle=rot
    )


def test_vision_id_cluster_id_roundtrip():
    assert vision_id_to_cluster_id(6, is_vision_only=False) == 5
    assert cluster_id_to_vision_id(5, is_vision_only=False) == 6
    assert vision_id_to_cluster_id(6, is_vision_only=True) == 6
    assert cluster_id_to_vision_id(6, is_vision_only=True) == 6


def test_mapping_keys_are_vision_ids():
    report = _make_report(
        [
            MatchResult(1, 10, 0.95, "ei", "high"),
            MatchResult(2, 20, 0.80, "ei", "marginal"),
            MatchResult(3, None, 0.1, "ei", "unmatched"),
        ]
    )
    bridge = ReferenceBridge(
        mapping=report.mapping,
        match_statuses={1: "high", 2: "marginal"},
        confidence_scores={1: 0.95, 2: 0.80},
        full_matches=report.matches,
    )
    assert bridge.mapping == {1: 10, 2: 20}
    assert bridge.has_match(1)
    assert not bridge.has_match(3)
    assert bridge.get_reference_id(1) == 10


def test_get_sta_uses_reference_id():
    report = _make_report([MatchResult(5, 99, 0.99, "ei", "high")])
    sta_obj = SimpleNamespace(red=np.ones((2, 2, 3)), green=None, blue=None)
    stas = _FakeSTADict({99: sta_obj})
    bridge = ReferenceBridge(
        mapping=report.mapping,
        ref_stas=stas,
        confidence_scores={5: 0.99},
        match_statuses={5: "high"},
        full_matches=report.matches,
    )
    assert bridge.has_sta(5)
    assert bridge.get_sta(5) is sta_obj
    assert bridge.get_sta(99) is None  # wrong key space


def test_get_stafit_and_ellipse():
    report = _make_report([MatchResult(5, 99, 0.99, "ei", "high")])
    fit = _stafit()
    params = _FakeParams({99: fit})
    bridge = ReferenceBridge(
        mapping=report.mapping,
        ref_params=params,
        full_matches=report.matches,
        match_statuses={5: "high"},
        confidence_scores={5: 0.99},
    )
    assert bridge.get_stafit(5) is fit
    ell = bridge.get_rf_ellipse_params(5)
    assert ell is not None
    assert ell["x0"] == 10.0
    assert ell["std_y"] == 3.0
    assert 5 in bridge.get_all_rf_ellipses()


def test_caveats_for_all_report_cells_hybrid():
    report = _make_report(
        [
            MatchResult(6, 16, 0.95, "ei", "high", next_best_corr=0.5, next_best_id=17),
            MatchResult(7, 17, 0.75, "ei", "marginal"),
            MatchResult(8, None, 0.2, "ei", "unmatched"),
            MatchResult(9, 19, 0.9, "ei", "conflict"),
        ]
    )
    bridge = ReferenceBridge(
        mapping=report.mapping,
        full_matches=report.matches,
        match_statuses={6: "high", 7: "marginal"},
        confidence_scores={6: 0.95, 7: 0.75},
        stimuli_loaded=("sta", "chirp"),
        ref_run_path="/ref/run",
    )
    caveats = bridge.build_ui_caveats(is_vision_only=False)
    # UI ids: 5,6,7,8
    assert set(caveats.keys()) == {5, 6, 7, 8}
    assert caveats[5].status == "high"
    assert caveats[5].reference_id == 16
    assert caveats[5].current_vision_id == 6
    assert caveats[7].status == "unmatched"
    assert caveats[7].reference_id is None
    assert caveats[8].status == "conflict"
    assert caveats[5].stimuli_available_on_ref == ("sta", "chirp")


def test_caveats_vision_only_ids_unchanged():
    report = _make_report([MatchResult(6, 16, 0.95, "ei", "high")])
    bridge = ReferenceBridge(
        mapping=report.mapping,
        full_matches=report.matches,
        match_statuses={6: "high"},
        confidence_scores={6: 0.95},
    )
    caveats = bridge.build_ui_caveats(is_vision_only=True)
    assert 6 in caveats
    assert caveats[6].cluster_id == 6


def test_load_chirp_remapped(tmp_path):
    # File IDs are Vision-keyed (1-indexed). After load → KS ids (cid-1).
    mdic = {
        "psth_mean": np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]], dtype=float),
        "cluster_id": np.array([10, 20]),
        "quality_index": np.array([0.8, 0.9]),
        "bin_size_ms": 10.0,
    }
    path = tmp_path / "data_Chirp_analysis.npy"
    np.save(path, mdic, allow_pickle=True)

    report = _make_report(
        [
            MatchResult(1, 10, 0.99, "ei", "high"),  # current vid 1 → ref vid 10
            MatchResult(2, 20, 0.99, "ei", "high"),
        ]
    )
    bridge = ReferenceBridge.from_matching_report(
        report,
        tmp_path,
        "dataset",
        load_stas=False,
        load_params=False,
        load_chirp=True,
        load_grating=False,
        ref_is_vision_only=False,
    )
    assert bridge.has_any_chirp()
    assert "chirp" in bridge.stimuli_loaded
    assert bridge.has_chirp(1)
    row = bridge.get_chirp_row(1)
    assert row is not None
    psth, qi = row
    assert qi == pytest.approx(0.8)
    np.testing.assert_array_equal(psth, [1.0, 2.0, 3.0])


def test_load_grating_remapped(tmp_path):
    # Analyzed grating schema: int keys → condition tuples
    mdic = {
        10: {
            (100, 2.0): {
                "condition_type": "dsos",
                "directions_deg": np.arange(0, 360, 30),
                "f1_amplitude": np.ones(12),
            }
        },
        20: {
            (100, 2.0): {
                "condition_type": "dsos",
                "directions_deg": np.arange(0, 360, 30),
                "f1_amplitude": np.ones(12) * 2,
            }
        },
    }
    path = tmp_path / "data_Grating_combined.npy"
    np.save(path, mdic, allow_pickle=True)

    report = _make_report([MatchResult(1, 10, 0.99, "ei", "high")])
    bridge = ReferenceBridge.from_matching_report(
        report,
        tmp_path,
        "dataset",
        load_stas=False,
        load_params=False,
        load_chirp=False,
        load_grating=True,
        ref_is_vision_only=False,
    )
    assert bridge.has_any_grating()
    assert bridge.has_grating(1)
    entry = bridge.get_grating_entry(1)
    assert entry is not None
    assert (100, 2.0) in entry


def test_missing_stimulus_does_not_fail_bridge(tmp_path):
    report = _make_report([MatchResult(1, 10, 0.99, "ei", "high")])
    # Empty dir — no STA (loader may warn), no chirp/grating
    bridge = ReferenceBridge.from_matching_report(
        report,
        tmp_path,
        "missing_dataset",
        load_stas=False,
        load_params=False,
        load_chirp=True,
        load_grating=True,
    )
    assert bridge.mapping == {1: 10}
    assert not bridge.has_any_chirp()
    assert not bridge.has_any_grating()
    assert bridge.summary()  # does not raise
