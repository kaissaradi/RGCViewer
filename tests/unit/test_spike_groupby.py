"""Stable grouping of spike_clusters for the load-time index map.

``cluster_spike_indices`` must list each cluster's spikes in recording order.
A non-stable sort groups the IDs but scrambles time, which breaks ISI and
every plot that assumes ``get_cluster_spikes()`` is sorted.

The 27 M-spike mergesort on a typical concatenated run is the slow part of
opening a Kilosort folder. Counting sort is O(n) for small integer IDs and
must match ``np.argsort(..., kind="mergesort")`` exactly.
"""

import numpy as np
import pytest

from src.analysis.data_manager import (
    DataManager,
    _group_spike_clusters,
    _runs_from_sorted,
    _stable_groupby_order,
)


def test_stable_groupby_order_matches_mergesort():
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 50, 20_000).astype(np.uint32)
    got = _stable_groupby_order(ids)
    ref = np.argsort(ids, kind="mergesort")
    assert np.array_equal(got, ref)


def test_stable_groupby_order_handles_offset_and_empty_ids():
    assert _stable_groupby_order(np.array([], dtype=np.int32)).size == 0

    ids = np.array([7, 9, 7, 8, 9, 7], dtype=np.int32)
    got = _stable_groupby_order(ids)
    assert np.array_equal(got, np.argsort(ids, kind="mergesort"))
    # Same ID keeps original (time) order.
    assert list(got[ids[got] == 7]) == [0, 2, 5]


def test_group_spike_clusters_matches_unique_counts():
    rng = np.random.default_rng(1)
    ids = rng.integers(3, 12, 5000).astype(np.int32)
    order, unique, starts, counts = _group_spike_clusters(ids)
    assert np.array_equal(order, np.argsort(ids, kind="mergesort"))
    ref_u, ref_c = np.unique(ids, return_counts=True)
    assert np.array_equal(unique, ref_u)
    assert np.array_equal(counts, ref_c)
    assert list(starts) == [0, *np.cumsum(counts[:-1]).tolist()]


def test_runs_from_sorted_split_points():
    sorted_ids = np.array([1, 1, 1, 4, 4, 9], dtype=np.int32)
    unique, starts, counts = _runs_from_sorted(sorted_ids)
    assert list(unique) == [1, 4, 9]
    assert list(starts) == [0, 3, 5]
    assert list(counts) == [3, 2, 1]


def _write_ks(tmp_path, times, clusters):
    ks = tmp_path / "kilosort25" / "data000"
    ks.mkdir(parents=True)
    np.save(ks / "spike_times.npy", np.asarray(times, dtype=np.int64))
    np.save(ks / "spike_clusters.npy", np.asarray(clusters, dtype=np.int32))
    (ks / "params.py").write_text("fs = 20000\nn_channels_dat = 512\n")
    return ks


def test_load_keeps_cluster_spikes_in_time_order(tmp_path, monkeypatch):
    """Interleaved clusters must still come back in recording order."""
    times = np.arange(12, dtype=np.int64) * 10
    clusters = np.array([2, 0, 2, 1, 0, 2, 1, 0, 1, 2, 0, 1], dtype=np.int32)
    ks = _write_ks(tmp_path, times, clusters)

    monkeypatch.setattr(DataManager, "load_stim_timing", lambda self: None)
    dm = DataManager(str(ks))
    ok, msg = dm.load_kilosort_data()
    assert ok, msg
    assert dm.sampling_rate == 20000.0

    for cid in (0, 1, 2):
        spikes = dm.get_cluster_spikes(cid)
        assert spikes.size == 4
        assert np.all(np.diff(spikes) > 0)
        expected = times[clusters == cid]
        assert np.array_equal(spikes, expected)


def test_load_does_not_unique_the_full_spike_array(tmp_path, monkeypatch):
    """A second np.unique on 27 M spikes is the old double-sort."""
    ks = _write_ks(
        tmp_path,
        np.arange(200, dtype=np.int64),
        np.repeat(np.arange(4, dtype=np.int32), 50),
    )
    monkeypatch.setattr(DataManager, "load_stim_timing", lambda self: None)
    dm = DataManager(str(ks))

    calls = {"n": 0}
    real_unique = np.unique

    def _watch(arr, *args, **kwargs):
        data = np.asanyarray(arr)
        if data.size >= 200:
            calls["n"] += 1
        return real_unique(arr, *args, **kwargs)

    monkeypatch.setattr(np, "unique", _watch)
    ok, msg = dm.load_kilosort_data()
    assert ok, msg
    assert calls["n"] == 0
    assert list(dm._spk_unique_cls) == [0, 1, 2, 3]
    assert list(dm._spk_unique_counts) == [50, 50, 50, 50]


def test_second_load_reads_isi_cache_from_params_sampling_rate(tmp_path, monkeypatch):
    """The cache key includes fs. params.py must be read before the key is."""
    ks = _write_ks(
        tmp_path,
        np.cumsum(np.full(80, 40, dtype=np.int64)),
        np.repeat(np.arange(4, dtype=np.int32), 20),
    )
    monkeypatch.setattr(DataManager, "load_stim_timing", lambda self: None)

    first = DataManager(str(ks))
    assert first.load_kilosort_data()[0]
    first.build_cluster_dataframe()
    assert first._cached_isi_pct is None
    assert (ks / DataManager.ISI_CACHE_NAME).exists()

    second = DataManager(str(ks))
    assert second.load_kilosort_data()[0]
    assert second._cached_isi_pct is not None
    assert second._spk_sorted_t is None
