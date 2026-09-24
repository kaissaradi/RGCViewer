"""Known-answer tests for the grating DSI/OSI math (PLAN.md Q9).

Each test builds spikes whose correct answer is known by hand, so it can
fail for the right reason. See the Conventions block in grating_calc.py.
"""

import numpy as np
import pytest

from src.analysis import grating_calc as gc


def _trial(ori, pre=250.0, stim=1000.0, bw=200.0, tf=2.0):
    return {"orientation": float(ori), "barWidth": bw, "temporalFrequency": tf,
            "preTime": pre, "stimTime": stim, "tailTime": 250.0}


def _modulated_train(onset_ms, stim_ms, tf_hz, n_per_cycle):
    """Spikes packed at the same phase of every cycle: a pure F1 response."""
    period = 1000.0 / tf_hz
    n_cycles = int(stim_ms // period)
    base = np.linspace(0.1 * period, 0.2 * period, n_per_cycle)
    return np.concatenate([onset_ms + k * period + base for k in range(n_cycles)])


# ── vector sum ────────────────────────────────────────────────────────────

def test_vector_sum_known_answer_for_cosine_tuning():
    # r(θ) = 1 + a·cos(θ − 90°) over 8 directions: DSI = a / 2, pref = 90°,
    # and a pure first harmonic has no second-harmonic (orientation) part.
    dirs = np.arange(0.0, 360.0, 45.0)
    a = 0.5
    resp = 1.0 + a * np.cos(np.deg2rad(dirs - 90.0))
    dsi, pref = gc.vector_sum_index(dirs, resp, harmonic=1)
    osi, _ = gc.vector_sum_index(dirs, resp, harmonic=2)
    assert dsi == pytest.approx(a / 2)
    assert pref == pytest.approx(90.0)
    assert osi == pytest.approx(0.0, abs=1e-12)


def test_orientation_tuning_gives_osi_not_dsi():
    dirs = np.arange(0.0, 360.0, 30.0)
    resp = 1.0 + 0.6 * np.cos(2 * np.deg2rad(dirs - 30.0))  # peaks at 30° and 210°
    dsi, _ = gc.vector_sum_index(dirs, resp, harmonic=1)
    osi, pref_ori = gc.vector_sum_index(dirs, resp, harmonic=2)
    assert dsi == pytest.approx(0.0, abs=1e-12)
    assert osi == pytest.approx(0.3)
    assert pref_ori == pytest.approx(30.0)


# ── p-value ───────────────────────────────────────────────────────────────

def test_pvalue_is_never_zero():
    # A cell that beats every shuffle gets 1/(N+1), not 0.
    dirs = [0.0, 90.0, 180.0, 270.0]
    by_dir = {d: np.full(10, 50.0 if d == 90.0 else 0.0) for d in dirs}
    p = gc.shuffle_pvalue(dirs, by_dir, harmonic=1, n_shuffles=199,
                          rng=np.random.default_rng(0))
    assert p == pytest.approx(1.0 / 200.0)


def test_pvalue_is_near_uniform_for_untuned_noise():
    # Calibration: under the null, P(p < 0.05) should be about 0.05.
    rng = np.random.default_rng(1)
    dirs = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    ps = []
    for _ in range(300):
        by_dir = {d: rng.poisson(10, size=8).astype(float) for d in dirs}
        ps.append(gc.shuffle_pvalue(dirs, by_dir, 1, 99, rng))
    frac = np.mean(np.asarray(ps) < 0.05)
    assert 0.01 < frac < 0.10


# ── directions ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [(0, 0.0), (360, 0.0), (-90, 270.0),
                                           (720.5, 0.5), (359.99999999, 0.0)])
def test_normalize_direction(raw, expected):
    assert gc.normalize_direction(raw) == pytest.approx(expected)


def test_zero_and_360_are_one_direction():
    params = [_trial(o) for o in (0.0, 90.0, 180.0, 270.0, 360.0)]
    (group,) = gc.group_grating_conditions(params)
    assert group["directions"] == [0.0, 90.0, 180.0, 270.0]
    assert group["idx_by_dir"][0.0] == [0, 4]


# ── timing ────────────────────────────────────────────────────────────────

def test_each_trial_uses_its_own_pre_time():
    # Same response in both directions, but the 90° trials start later.
    # Taking preTime from trial 0 used to misalign the 90° window.
    params, trials = [], []
    for ori in (0.0, 90.0, 180.0, 270.0):
        pre = 250.0 if ori in (0.0, 180.0) else 750.0
        for _ in range(4):
            params.append(_trial(ori, pre=pre))
            trials.append(_modulated_train(pre, 1000.0, 2.0, 5))
    result = gc.compute_grating_response(0, {0: trials}, params, n_shuffles=0)
    entry = result[(200.0, 2.0)]
    assert np.allclose(entry["mean_response"], entry["mean_response"][0])
    assert entry["mean_response"][0] > 0


def test_f1_window_does_not_run_past_the_stimulus():
    # stimTime 1003 ms is not a multiple of the 5 ms bin. A spike 1 ms after
    # the stimulus ends must not count.
    after = np.array([1004.0, 1004.2, 1004.4])
    assert gc.f1_amplitude(after, (0.0, 1003.0), 2.0) == 0.0


def test_spike_exactly_at_stim_offset_is_outside():
    # [start, end): a spike at the offset belongs to the tail.
    assert gc.window_rates([np.array([1000.0])], [(0.0, 1000.0)])[0] == 0.0
    assert gc.window_rates([np.array([999.9])], [(0.0, 1000.0)])[0] == 1.0


def test_vectorized_f1_matches_one_trial_at_a_time():
    rng = np.random.default_rng(3)
    trials = [np.sort(rng.uniform(250, 1250, rng.integers(0, 60))) for _ in range(12)]
    windows = [(250.0, 1250.0)] * 12
    tfs = [2.0, 4.0] * 6
    vec = gc.f1_amplitudes(trials, windows, tfs)
    one = [gc.f1_amplitude(t, w, f) for t, w, f in zip(trials, windows, tfs)]
    assert np.allclose(vec, one)


# ── output contract ───────────────────────────────────────────────────────

def test_entry_reports_sd_trial_count_and_units():
    params = [_trial(o) for o in (0.0, 90.0, 180.0, 270.0) for _ in range(3)]
    trials = [_modulated_train(250.0, 1000.0, 2.0, 3 + (i % 3)) for i in range(len(params))]
    result = gc.compute_grating_response(0, {0: trials}, params, n_shuffles=0)
    entry = result[(200.0, 2.0)]
    assert result["schema_version"] == gc.GRATING_SCHEMA_VERSION
    assert list(entry["n_trials"]) == [3, 3, 3, 3]
    assert np.allclose(entry["sem_response"], entry["sd_response"] / np.sqrt(3))
    assert entry["response_units"] == "spikes/s"
    assert entry["response_label"] == "F1 amplitude"


def test_unknown_response_metric_is_rejected():
    with pytest.raises(ValueError):
        gc.compute_grating_response(0, {0: []}, [_trial(0)], response_metric="F1")


def test_old_schema_rows_are_recomputed():
    row = {(200.0, 2.0): {"condition_type": "dsos",
                          "directions_deg": np.array([0.0, 90.0, 180.0, 270.0])}}
    assert gc.grating_entry_needs_recompute(row)            # untagged = v1
    row["schema_version"] = gc.GRATING_SCHEMA_VERSION
    assert not gc.grating_entry_needs_recompute(row)


# ── rasters ───────────────────────────────────────────────────────────────

def test_trial_rasters_align_to_each_onset_in_seconds():
    params = [_trial(0.0, pre=250.0), _trial(90.0, pre=500.0), _trial(360.0, pre=250.0),
              _trial(0.0, bw=999.0)]  # other condition: excluded
    trials = [np.array([250.0, 750.0]), np.array([500.0]), np.array([1250.0]), np.array([1.0])]
    r = gc.trial_rasters(trials, params, (200.0, 2.0))
    assert set(k for k in r if k != "_timing") == {0.0, 90.0}
    assert [list(x) for x in r[0.0]] == [[0.0, 0.5], [1.0]]   # 0° and 360° merged
    assert list(r[90.0][0]) == [0.0]
    assert r["_timing"] == {"pre_s": 0.25, "stim_s": 1.0, "tail_s": 0.25}
