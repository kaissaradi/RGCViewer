"""Grating grouping uses whatever (orientation, bar width, TF) was run.

Do not assume a 12-dir crossed grid. A (bw, tf) with enough unique
orientations is DSOS; a 1-dir bar-width sweep stays SF. The old 8-dir
cutoff classified 4- and 6-dir conditions as SF and skipped DSI/OSI.
"""

import pickle
import threading

import numpy as np
import pandas as pd

from src.analysis import cache_persistence, grating_calc
from src.analysis.data_manager import DataManager


def _trial(ori, bw, tf, pre=250.0, stim=1000.0, rep=0):
    return {
        "orientation": float(ori),
        "spatialFrequency": 1.0 / max(bw, 1.0),
        "temporalFrequency": float(tf),
        "barWidth": float(bw),
        "preTime": pre,
        "stimTime": stim,
        "tailTime": 250.0,
        "repetition": rep,
    }


def _split_12_dir_params(n_reps=2):
    """1212A layout: even 60° at 100/2, odd 60° at 400/4."""
    params = []
    for rep in range(n_reps):
        for ori in range(0, 360, 30):
            if ori % 60 == 0:
                bw, tf = 100.0, 2.0
            else:
                bw, tf = 400.0, 4.0
            params.append(_trial(ori, bw, tf, rep=rep))
    return params


def _crossed_12_dir_params(n_reps=1):
    """Usual layout: every ori at both (100, 2) and (400, 4)."""
    params = []
    for rep in range(n_reps):
        for bw, tf in ((100.0, 2.0), (400.0, 4.0)):
            for ori in range(0, 360, 30):
                params.append(_trial(ori, bw, tf, rep=rep))
    return params


def _six_dir_params():
    return [_trial(ori, 200.0, 2.0) for ori in range(0, 360, 60)]


def _four_dir_params():
    return [_trial(ori, 200.0, 2.0) for ori in (0.0, 90.0, 180.0, 270.0)]


def _sf_params():
    """Bar-width sweep at a single orientation."""
    return [_trial(0.0, bw, 2.0) for bw in (10.0, 20.0, 40.0, 80.0, 160.0)]


def _ds_spikes(params, pref=90.0):
    trials = []
    for t in params:
        ori = t["orientation"]
        t0 = t["preTime"]
        t1 = t0 + t["stimTime"]
        delta = abs(((ori - pref + 180.0) % 360.0) - 180.0)
        n = 24 if delta < 20 else (8 if delta < 70 else 2)
        trials.append(np.linspace(t0 + 40.0, t1 - 40.0, n))
    return {0: trials}


def _stale_six_dir_sf_entry():
    """Old cache: 6-dir conditions tagged SF because the cutoff was 8."""
    return {
        (100.0, 2.0): {
            "condition_type": "sf",
            "directions_deg": np.array([0.0, 60.0, 120.0, 180.0, 240.0, 300.0]),
            "DSI": np.nan,
            "OSI": np.nan,
        },
        (400.0, 4.0): {
            "condition_type": "sf",
            "directions_deg": np.array([30.0, 90.0, 150.0, 210.0, 270.0, 330.0]),
            "DSI": np.nan,
            "OSI": np.nan,
        },
        "sf_bar_widths": np.array([100.0, 400.0]),
        "sf_tuning_curve": np.array([1.0, 1.0]),
    }


def _dsos_keys(groups_or_result):
    if isinstance(groups_or_result, list):
        return [g["key"] for g in groups_or_result if g["condition_type"] == "dsos"]
    return [
        k
        for k, v in groups_or_result.items()
        if isinstance(k, tuple) and v.get("condition_type") == "dsos"
    ]


def test_conditions_are_whatever_bw_tf_pairs_ran():
    groups = grating_calc.group_grating_conditions(_split_12_dir_params())
    by_key = {g["key"]: g for g in groups}
    assert set(by_key) == {(100.0, 2.0), (400.0, 4.0)}
    assert by_key[(100.0, 2.0)]["condition_type"] == "dsos"
    assert by_key[(400.0, 4.0)]["condition_type"] == "dsos"
    np.testing.assert_allclose(
        by_key[(100.0, 2.0)]["directions"], [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    )
    np.testing.assert_allclose(
        by_key[(400.0, 4.0)]["directions"], [30.0, 90.0, 150.0, 210.0, 270.0, 330.0]
    )


def test_crossed_protocol_stays_split_by_bar_width_and_tf():
    groups = grating_calc.group_grating_conditions(_crossed_12_dir_params())
    assert len(groups) == 2
    for g in groups:
        assert g["condition_type"] == "dsos"
        assert len(g["directions"]) == 12


def test_six_and_four_dir_conditions_are_dsos():
    six = grating_calc.group_grating_conditions(_six_dir_params())
    four = grating_calc.group_grating_conditions(_four_dir_params())
    assert len(six) == 1 and six[0]["condition_type"] == "dsos"
    assert len(six[0]["directions"]) == 6
    assert len(four) == 1 and four[0]["condition_type"] == "dsos"
    assert len(four[0]["directions"]) == 4


def test_sf_bar_width_sweep_stays_sf():
    groups = grating_calc.group_grating_conditions(_sf_params())
    assert groups
    assert all(g["condition_type"] == "sf" for g in groups)
    assert all(len(g["directions"]) == 1 for g in groups)


def test_untuned_cells_skip_the_shuffle(monkeypatch):
    """Permutation test is the slow part — untuned cells must not run it."""
    calls = []

    def _boom(*_a, **_k):
        calls.append(1)
        raise AssertionError("shuffle_pvalue should be skipped")

    monkeypatch.setattr(grating_calc, "shuffle_pvalue", _boom)
    params = _six_dir_params()
    spikes = {0: [np.array([400.0]) for _ in params]}
    result = grating_calc.compute_grating_response(
        0, spikes, params, n_shuffles=200
    )
    assert calls == []
    dsos = [result[k] for k in _dsos_keys(result)]
    assert dsos
    for entry in dsos:
        assert abs(entry["DSI"]) < grating_calc.SHUFFLE_INDEX_FLOOR
        assert entry["DSI_pvalue"] == 1.0
        assert entry["OSI_pvalue"] == 1.0


def test_tuned_cells_still_run_the_shuffle(monkeypatch):
    calls = []
    real = grating_calc.shuffle_pvalue

    def _count(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(grating_calc, "shuffle_pvalue", _count)
    params = _six_dir_params()
    grating_calc.compute_grating_response(
        0, _ds_spikes(params, pref=90.0), params, n_shuffles=8
    )
    assert calls, "a clearly tuned cell must still get a shuffle p-value"


def test_split_protocol_compute_dsi_per_condition_that_ran():
    params = _split_12_dir_params()
    result = grating_calc.compute_grating_response(
        0, _ds_spikes(params), params, n_shuffles=16
    )
    dsos = _dsos_keys(result)
    assert set(dsos) == {(100.0, 2.0), (400.0, 4.0)}
    for key in dsos:
        entry = result[key]
        assert len(entry["directions_deg"]) == 6
        assert np.isfinite(entry["DSI"])
        assert np.isfinite(entry["OSI"])


def test_stale_six_dir_sf_cache_needs_recompute():
    assert grating_calc.grating_entry_needs_recompute(_stale_six_dir_sf_entry())


def test_real_dsos_cache_is_not_stale():
    entry = {
        (100.0, 2.0): {
            "condition_type": "dsos",
            "directions_deg": np.arange(0.0, 360.0, 30.0),
            "DSI": 0.4,
            "OSI": 0.1,
        }
    }
    assert not grating_calc.grating_entry_needs_recompute(entry)


def test_one_dir_sf_cache_is_not_stale():
    entry = {
        (80.0, 2.0): {
            "condition_type": "sf",
            "directions_deg": np.array([0.0]),
            "DSI": np.nan,
        }
    }
    assert not grating_calc.grating_entry_needs_recompute(entry)


def test_dummy_scalar_cache_is_not_stale():
    assert not grating_calc.grating_entry_needs_recompute({"dsi": 0.4})


def test_format_condition_includes_direction_count():
    label = grating_calc.format_condition_label(
        (100.0, 2.0),
        {"directions_deg": np.array([0.0, 60.0, 120.0, 180.0, 240.0, 300.0])},
    )
    assert "100" in label and "2" in label
    assert "6" in label


def _bare_dm(tmp_path):
    dm = DataManager.__new__(DataManager)
    dm.kilosort_dir = tmp_path
    dm.grating_computed_cache = {}
    dm._grating_cache_lock = threading.Lock()
    dm.cluster_df = pd.DataFrame({"cluster_id": [0, 1]})
    dm.grating_status = "raw_only"
    dm.grating_raw_data = {"spike_times_by_trial": {}, "trial_parameters": []}
    return dm


def test_load_drops_stale_sf_tagged_direction_cache(tmp_path):
    payload = cache_persistence.add_version({0: _stale_six_dir_sf_entry()})
    with open(tmp_path / "grating_computed_cache.pkl", "wb") as f:
        pickle.dump(payload, f)

    dm = _bare_dm(tmp_path)
    restored = dm._load_grating_cache_from_disk()
    assert restored == {}
    assert not (tmp_path / "grating_computed_cache.pkl").exists()


def test_warmup_starts_grating_alongside_physics(monkeypatch):
    from unittest.mock import MagicMock

    from src.gui import callbacks

    order = []
    mw = MagicMock()
    mw.data_manager.cluster_df = pd.DataFrame({"cluster_id": [0]})
    mw._physics_warm_stop = None
    mw.physics_warm_done = MagicMock()

    monkeypatch.setattr(
        callbacks, "maybe_fill_grating_cache", lambda *a, **k: order.append("grating")
    )

    class _Thread:
        def __init__(self, *a, **k):
            order.append("physics-thread")

        def start(self):
            order.append("physics-start")

    monkeypatch.setattr(callbacks.threading, "Thread", _Thread)
    callbacks.start_physics_warmup(mw)
    assert "grating" in order
    assert "physics-start" in order
    # Must not wait for physics to finish before kicking DS/OS.
    assert order.index("grating") != -1

    order.clear()
    callbacks.start_physics_warmup(mw, fill_grating=False)
    assert "grating" not in order


def test_grating_ids_needing_compute_includes_stale_entries():
    dm = _bare_dm("/tmp")
    dm.grating_computed_cache = {
        0: _stale_six_dir_sf_entry(),
        1: {
            (100.0, 2.0): {
                "condition_type": "dsos",
                "directions_deg": np.arange(0.0, 360.0, 30.0),
                "DSI": 0.4,
            }
        },
    }
    assert dm.grating_ids_needing_compute([0, 1]) == [0]
