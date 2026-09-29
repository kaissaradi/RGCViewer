"""The Raster tab: folding, trials, nearby units, CCG and shared spikes (PLAN.md Q63)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.analysis import raster_data as rd


def test_fold_fixed_keeps_every_spike_in_its_row():
    x, rows, n = rd.fold_fixed([0.5, 1.2, 2.9], 1.0, 3.0)
    assert n == 3
    np.testing.assert_allclose(x, [0.5, 0.2, 0.9])
    assert rows.tolist() == [0, 1, 2]


def test_fold_trials_times_from_each_start():
    x, rows, n = rd.fold_trials([-1.0, 0.5, 10.2, 25.0, 31.0], [0.0, 10.0, 20.0], 10.0)
    assert n == 3
    assert rows.tolist() == [0, 1, 2]                    # -1 s (before) and 31 s (past the span) dropped
    np.testing.assert_allclose(x, [0.5, 0.2, 5.0])


def test_trial_structure_needs_gaps():
    even = np.arange(200) * 0.833
    assert rd.trial_structure(even) is None                  # white noise: no trials
    jitter = (np.arange(200) * 16655 + np.random.default_rng(0).integers(-1, 2, 200)) / 20000.0
    assert rd.trial_structure(jitter) is None                # one-sample jitter is not a trial
    groups = np.concatenate([k * 12.0 + np.arange(10) * 0.833 for k in range(6)])
    starts, span = rd.trial_structure(groups)
    np.testing.assert_allclose(starts, np.arange(6) * 12.0)
    assert span == pytest.approx(12.0)


def test_shared_fraction_and_chance():
    a = np.arange(1000) * 0.05
    assert rd.shared_fraction(a, a) == 1.0
    assert rd.shared_fraction(a, a + 0.002) == 0.0            # 2 ms apart: not shared
    assert rd.shared_fraction(a, a + 0.0004) == 1.0
    assert rd.chance_shared(20.0) == pytest.approx(1 - np.exp(-0.02))


def test_ccg_peaks_at_the_lag():
    rng = np.random.default_rng(0)
    a = np.sort(rng.uniform(0, 100, 2000))
    lag_ms, rate = rd.cross_correlogram(a, a + 0.0032)
    assert lag_ms[np.argmax(rate)] == pytest.approx(3.25)     # the 3.0–3.5 ms bin
    # Every spike once in that bin (2000 /s), plus the odd chance pair of other spikes.
    assert rate.max() == pytest.approx(1 / rd.CCG_BIN_S, rel=0.02)


def _dm(xy, eis=None):
    ids = list(range(len(xy)))
    df = pd.DataFrame({"cluster_id": ids, "x_um": [p[0] for p in xy], "y_um": [p[1] for p in xy]})
    return SimpleNamespace(cluster_df=df, vision_eis=eis, get_vision_id_for_cluster=lambda c: c + 1)


def test_nearby_units_by_distance_within_the_radius():
    dm = _dm([(0, 0), (30, 0), (0, 60), (500, 0), (np.nan, np.nan)])
    nbs = rd.nearby_units(dm, 0)
    assert [n.cluster_id for n in nbs] == [1, 2]
    assert nbs[0].distance_um == pytest.approx(30.0)


def test_ei_similarity_ranks_the_look_alike_first():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, (64, 40))
    eis = {1: SimpleNamespace(ei=base), 2: SimpleNamespace(ei=rng.normal(0, 1, (64, 40))),
           3: SimpleNamespace(ei=base * 0.9 + rng.normal(0, 0.05, (64, 40)))}
    dm = _dm([(0, 0), (30, 0), (60, 0)], eis)
    nbs = rd.nearby_units(dm, 0)
    sims = rd.ei_similarities(dm, 0, [n.cluster_id for n in nbs])
    assert sims[2] > 0.9 and abs(sims[1]) < 0.3
    ranked = rd.rank_neighbours(nbs, sims)
    assert [n.cluster_id for n in ranked] == [2, 1] and ranked[0].sim_source == "EI"
