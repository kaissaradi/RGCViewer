"""One DS/OS test per cell across all its conditions (PLAN.md Q33).

Testing each (bar width, TF) condition at alpha and taking any pass gave an
untuned cell about 1 - (1 - alpha)^k chances with k conditions. The max
statistic keeps it at alpha.
"""

import numpy as np

from src.analysis import grating_calc as gc

DIRS = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]


def _noise_condition(rng, n_trials=10, rate=5.0):
    return {d: rng.poisson(rate, n_trials).astype(float) for d in DIRS}


def test_one_condition_family_p_equals_the_condition_p():
    rng = np.random.default_rng(3)
    by_dir = _noise_condition(rng)
    obs, null = gc.shuffle_null(DIRS, by_dir, 1, 199, np.random.default_rng(7))
    assert gc.family_pvalues([obs], [null])[0] == gc._permutation_p(null, obs)


def test_family_p_is_never_below_the_condition_p():
    rng = np.random.default_rng(4)
    obs, nulls = [], []
    for _ in range(3):
        o, n = gc.shuffle_null(DIRS, _noise_condition(rng), 1, 199, rng)
        obs.append(o)
        nulls.append(n)
    fam = gc.family_pvalues(obs, nulls)
    for o, n, f in zip(obs, nulls, fam):
        assert f >= gc._permutation_p(n, o)


def test_untuned_cells_pass_at_alpha_not_more():
    rng = np.random.default_rng(0)
    n_cells, any_raw, any_fam = 600, 0, 0
    for _ in range(n_cells):
        obs, nulls = [], []
        for _cond in range(2):
            o, n = gc.shuffle_null(DIRS, _noise_condition(rng), 1, 199, rng)
            obs.append(o)
            nulls.append(n)
        raw = [gc._permutation_p(n, o) for o, n in zip(obs, nulls)]
        any_raw += min(raw) < 0.05
        any_fam += min(gc.family_pvalues(obs, nulls)) < 0.05
    # Two conditions tested separately: ~9.75 %. One test: ~5 %.
    assert 0.07 < any_raw / n_cells < 0.13
    assert 0.03 < any_fam / n_cells < 0.07


def _trial(ori, bw, tf=2.0):
    return {"orientation": ori, "barWidth": bw, "temporalFrequency": tf,
            "preTime": 250.0, "stimTime": 1000.0, "tailTime": 250.0,
            "contrast": 1.0, "repetition": 0}


def _cell(tuned_bw, rng):
    """Two conditions; a 2 Hz modulated response only at 90° of tuned_bw."""
    params, trials = [], []
    for bw in (100.0, 400.0):
        for ori in DIRS + [90.0]:
            for _ in range(8):
                params.append(_trial(ori, bw))
                n = 6 if (bw == tuned_bw and ori == 90.0) else 1
                period = 500.0
                spikes = np.concatenate([250.0 + k * period + rng.uniform(40, 80, n)
                                         for k in range(2)])
                trials.append(np.sort(spikes))
    return params, trials


def test_a_tuned_cell_still_passes_and_every_condition_gets_a_p():
    rng = np.random.default_rng(1)
    params, trials = _cell(400.0, rng)
    res = gc.compute_grating_response(0, {0: trials}, params)
    tuned, flat = res[(400.0, 2.0)], res[(100.0, 2.0)]
    assert tuned["DSI_pvalue_fw"] < 0.05
    assert tuned["DSI_pvalue_fw"] >= tuned["DSI_pvalue"]
    # The flat condition was shuffled too (no 1.0 placeholder), because the
    # family-wise null needs it.
    assert flat["DSI_pvalue"] < 1.0 and np.isfinite(flat["DSI_pvalue_fw"])
    sel = gc.select_best_dsos_condition(res)
    assert sel["classification"] == "DS" and sel["condition"] == (400.0, 2.0)
    assert sel["DSI_pvalue_gate"] == tuned["DSI_pvalue_fw"]


def test_files_without_trials_get_bonferroni_over_conditions():
    entry = {"DSI_pvalue": 0.03, "OSI_pvalue": 0.2}
    assert gc.gate_pvalue(entry, "DSI", 1) == 0.03
    assert gc.gate_pvalue(entry, "DSI", 2) == 0.06
    assert gc.gate_pvalue(entry, "OSI", 9) == 1.0
    assert np.isnan(gc.gate_pvalue({}, "DSI", 2))
    assert gc.gate_pvalue({"DSI_pvalue": 0.5, "DSI_pvalue_fw": 0.01}, "DSI", 3) == 0.01


def test_schema_2_rows_are_recomputed():
    row = {"schema_version": 2,
           (100.0, 2.0): {"condition_type": "dsos", "directions_deg": np.array(DIRS)}}
    assert gc.grating_entry_needs_recompute(row)
