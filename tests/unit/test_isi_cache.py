"""Tests for the persisted ISI-violation cache.

Recomputing the ISI percentages means gathering spike times into cluster order
and running a diff/mask pass over every spike — about 3 s for a one-hour
recording, repeated on every load of the same unchanged run.

The cache is keyed on a hash of the spike array *contents*, not on the file
mtime, because refinement rewrites ``spike_clusters`` in memory only: the file
on disk is byte-identical before and after a split. These tests pin that
distinction, since a stale ISI percentage is a curation error, not a slow load.
"""

import pickle

import numpy as np
import pytest

from src.analysis.data_manager import DataManager

SAMPLING_RATE = 20000.0
N_CLUSTERS = 6


def _write_kilosort_dir(tmp_path, seed=0, n_spikes=4000):
    ks = tmp_path / "kilosort25" / "data000"
    ks.mkdir(parents=True)
    rng = np.random.default_rng(seed)

    # Ascending times with a few deliberately tight pairs so some clusters
    # actually register violations rather than all reading 0.0.
    times = np.cumsum(rng.integers(1, 400, n_spikes)).astype(np.int64)
    clusters = rng.integers(0, N_CLUSTERS, n_spikes).astype(np.int32)

    np.save(ks / "spike_times.npy", times)
    np.save(ks / "spike_clusters.npy", clusters)
    np.save(ks / "amplitudes.npy", np.ones(n_spikes, dtype=np.float32))
    return ks


def _load(ks, monkeypatch):
    monkeypatch.setattr(DataManager, "load_stim_timing", lambda self: None)
    dm = DataManager(str(ks))
    dm.sampling_rate = SAMPLING_RATE
    ok, msg = dm.load_kilosort_data()
    assert ok, msg
    dm.build_cluster_dataframe()
    return dm


def _pcts(dm):
    return dm.cluster_df.set_index("cluster_id")["isi_violations_pct"].to_dict()


@pytest.fixture
def ks_dir(tmp_path):
    return _write_kilosort_dir(tmp_path)


@pytest.fixture
def cache_path(ks_dir):
    return ks_dir / DataManager.ISI_CACHE_NAME


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_first_load_computes_and_writes_the_cache(ks_dir, cache_path, monkeypatch):
    dm = _load(ks_dir, monkeypatch)

    assert dm._cached_isi_pct is None  # nothing on disk to reuse
    assert cache_path.exists()
    assert set(_pcts(dm)) == set(range(N_CLUSTERS))


def test_second_load_reuses_the_cache(ks_dir, monkeypatch):
    first = _pcts(_load(ks_dir, monkeypatch))

    dm = _load(ks_dir, monkeypatch)

    assert dm._cached_isi_pct is not None
    assert _pcts(dm) == first


def test_the_percentages_are_not_all_zero(ks_dir, monkeypatch):
    """A cache that agrees on 0.0 everywhere would prove nothing."""
    assert any(v > 0 for v in _pcts(_load(ks_dir, monkeypatch)).values())


def test_the_isi_cache_dict_is_populated_on_the_warm_path(ks_dir, monkeypatch):
    """_calculate_isi_violations reads this, so a warm load must fill it too."""
    _load(ks_dir, monkeypatch)
    dm = _load(ks_dir, monkeypatch)

    assert dm._cached_isi_pct is not None
    assert len(dm.isi_cache) == N_CLUSTERS


def test_warm_load_skips_the_spike_time_gather(ks_dir, monkeypatch):
    """The gather exists only for this pass; a cache hit must not pay for it."""
    _load(ks_dir, monkeypatch)

    monkeypatch.setattr(DataManager, "load_stim_timing", lambda self: None)
    dm = DataManager(str(ks_dir))
    dm.sampling_rate = SAMPLING_RATE
    assert dm.load_kilosort_data()[0]

    assert dm._spk_sorted_t is None
    assert dm._spk_sorted_cls is not None  # still needed for the index grouping


# ---------------------------------------------------------------------------
# Invalidation — the part that matters
# ---------------------------------------------------------------------------


def test_changed_spike_clusters_invalidate_the_cache(ks_dir, monkeypatch):
    first = _pcts(_load(ks_dir, monkeypatch))

    # A different sort of the same recording.
    clusters = np.load(ks_dir / "spike_clusters.npy")
    clusters[: len(clusters) // 2] = (clusters[: len(clusters) // 2] + 1) % N_CLUSTERS
    np.save(ks_dir / "spike_clusters.npy", clusters)

    dm = _load(ks_dir, monkeypatch)

    assert dm._cached_isi_pct is None
    assert _pcts(dm) != first


def test_in_memory_refinement_is_seen_by_the_key(ks_dir, monkeypatch):
    """The case a file mtime cannot catch.

    spike_clusters is mapped copy-on-write, so a split changes the array but
    never the file. A cache keyed on the file would serve the pre-split
    percentages forever.
    """
    dm = _load(ks_dir, monkeypatch)
    key_before = dm._spike_content_key()

    # A split: move some spikes to a brand-new cluster id, in memory only.
    dm.spike_clusters[:500] = N_CLUSTERS

    assert dm._spike_content_key() != key_before
    assert dm._load_isi_cache() is None


def test_the_file_on_disk_is_untouched_by_refinement(ks_dir, monkeypatch):
    dm = _load(ks_dir, monkeypatch)
    before = (ks_dir / "spike_clusters.npy").read_bytes()

    dm.spike_clusters[:500] = N_CLUSTERS

    assert (ks_dir / "spike_clusters.npy").read_bytes() == before


def test_a_different_sampling_rate_invalidates(ks_dir, monkeypatch):
    """The refractory window is in samples, so the rate changes the answer."""
    dm = _load(ks_dir, monkeypatch)
    key_before = dm._spike_content_key()

    dm.sampling_rate = 30000.0

    assert dm._spike_content_key() != key_before


def test_changed_spike_times_invalidate_the_cache(ks_dir, monkeypatch):
    _load(ks_dir, monkeypatch)

    times = np.load(ks_dir / "spike_times.npy")
    np.save(ks_dir / "spike_times.npy", times * 2)

    dm = _load(ks_dir, monkeypatch)

    assert dm._cached_isi_pct is None


# ---------------------------------------------------------------------------
# Damaged or foreign cache files fall back to a recompute
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b"not a pickle", id="corrupt"),
        pytest.param(pickle.dumps(["wrong", "type"]), id="wrong_type"),
        pytest.param(pickle.dumps({"key": "someone elses key", "isi_pct": {0: 1.0}}),
                     id="foreign_key"),
        pytest.param(pickle.dumps({"isi_pct": {0: 1.0}}), id="no_key"),
        pytest.param(pickle.dumps({"key": "x", "isi_pct": {}}), id="empty_map"),
    ],
)
def test_unusable_cache_falls_back_to_a_recompute(
    ks_dir, cache_path, monkeypatch, payload
):
    reference = _pcts(_load(ks_dir, monkeypatch))
    cache_path.write_bytes(payload)

    dm = _load(ks_dir, monkeypatch)

    assert dm._cached_isi_pct is None
    assert _pcts(dm) == reference


def test_missing_cache_is_not_an_error(ks_dir, cache_path, monkeypatch):
    dm = _load(ks_dir, monkeypatch)
    cache_path.unlink()

    assert dm._load_isi_cache() is None


def test_rebuild_caches_removes_the_isi_cache(ks_dir, cache_path, monkeypatch):
    dm = _load(ks_dir, monkeypatch)
    assert cache_path.exists()

    removed = dm.rebuild_caches()

    assert DataManager.ISI_CACHE_NAME in removed
    assert not cache_path.exists()


def test_key_is_none_without_spike_arrays(ks_dir, monkeypatch):
    dm = _load(ks_dir, monkeypatch)
    dm.spike_clusters = None

    assert dm._spike_content_key() is None
    assert dm._load_isi_cache() is None


def test_close_drops_the_cached_percentages(ks_dir, monkeypatch):
    _load(ks_dir, monkeypatch)
    dm = _load(ks_dir, monkeypatch)
    assert dm._cached_isi_pct is not None

    dm.close()

    assert dm._cached_isi_pct is None
