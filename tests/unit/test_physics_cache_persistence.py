"""Cross-session persistence of the physics, standard-plot and grating caches.

The bug under test: partial ``feature_cache`` rows written by the ACG pass
(``{"acg": ...}``, no ``_computed``) used to be pickled alongside the finished
ones, and the load-time pruner in ``build_cluster_dataframe`` drops every row
lacking ``_computed``. The persisted cache was therefore guaranteed to be thrown
away and the whole physics pass re-ran at every launch — the "double load".

These tests are deliberately Qt-free: they build a DataManager via ``__new__``
(AGENTS.md §9) so they run headless with no dataset, no DataJoint and no Qt
event loop.
"""

import pickle
import threading

import pandas as pd
import pytest

from src.analysis import cache_persistence
from src.analysis.data_manager import DataManager


def _bare_dm(tmp_path):
    """A DataManager with only the cache machinery wired up."""
    dm = DataManager.__new__(DataManager)
    dm.kilosort_dir = tmp_path
    dm.feature_cache = {}
    dm.standard_plot_cache = {}
    dm.grating_computed_cache = {}
    dm._feature_lock = threading.Lock()
    dm._standard_plot_lock = threading.Lock()
    dm._grating_cache_lock = threading.Lock()
    dm._save_lock = threading.Lock()
    dm._save_in_progress = False
    dm._physics_done_count = 0
    dm.cluster_df = pd.DataFrame()
    return dm


def _computed(value):
    return {"acg": [1, 2, 3], "timecourse": value, "_computed": True}


def _partial():
    return {"acg": [1, 2, 3]}


# --------------------------------------------------------------------------
# Pure rules (cache_persistence)
# --------------------------------------------------------------------------


def test_filter_computed_entries_drops_partial_rows():
    cache = {1: _computed(1.0), 2: _partial(), 3: _computed(3.0), 4: None}
    kept = cache_persistence.filter_computed_entries(cache)
    assert set(kept) == {1, 3}


def test_version_roundtrip():
    payload = cache_persistence.add_version({1: _computed(1.0)})
    assert payload[cache_persistence.VERSION_KEY] == cache_persistence.CACHE_VERSION
    entries, version = cache_persistence.strip_version(payload)
    assert set(entries) == {1}
    assert cache_persistence.is_current_version(version)


@pytest.mark.parametrize(
    "payload",
    [
        {1: _computed(1.0)},  # legacy: no version stamp at all
        {1: _computed(1.0), cache_persistence.VERSION_KEY: -1},  # stale version
        ["not", "a", "dict"],
    ],
)
def test_load_versioned_entries_discards_legacy_and_stale(payload):
    assert cache_persistence.load_versioned_entries(payload) == {}


# --------------------------------------------------------------------------
# Fix 1 — the physics cache survives a save/load round trip
# --------------------------------------------------------------------------


def test_only_computed_entries_persist_and_reload(tmp_path):
    dm = _bare_dm(tmp_path)
    dm.feature_cache = {1: _computed(1.0), 2: _partial(), 3: _computed(3.0)}
    dm.save_standard_plot_cache(blocking=True)

    fresh = _bare_dm(tmp_path)
    fresh.load_persisted_caches()

    # The partial row never reached disk, so the pruner has nothing to delete
    # and the cache is genuinely warm on the next launch.
    assert set(fresh.feature_cache) == {1, 3}
    assert fresh._physics_done_count == 2


def test_legacy_unversioned_feature_cache_is_discarded(tmp_path):
    # A pkl written by the old code: unversioned and full of partial rows.
    with open(tmp_path / "feature_cache.pkl", "wb") as f:
        pickle.dump({1: _computed(1.0), 2: _partial()}, f)

    dm = _bare_dm(tmp_path)
    dm.load_persisted_caches()

    assert dm.feature_cache == {}
    assert dm._physics_done_count == 0


def test_empty_snapshot_does_not_clobber_good_cache(tmp_path):
    dm = _bare_dm(tmp_path)
    dm.feature_cache = {1: _computed(1.0)}
    dm.save_standard_plot_cache(blocking=True)

    # A later KS-only session has no computed physics — saving must not wipe
    # the warm cache the previous session built.
    ks_only = _bare_dm(tmp_path)
    ks_only.feature_cache = {2: _partial()}
    ks_only.save_standard_plot_cache(blocking=True)

    fresh = _bare_dm(tmp_path)
    fresh.load_persisted_caches()
    assert set(fresh.feature_cache) == {1}


# --------------------------------------------------------------------------
# Fix 2 — the grating DS/OS batch is persisted
# --------------------------------------------------------------------------


def test_grating_cache_roundtrips(tmp_path):
    dm = _bare_dm(tmp_path)
    dm.grating_computed_cache = {5: {"dsi": 0.4}, 6: {"dsi": 0.1}, 7: None}
    dm.save_standard_plot_cache(blocking=True)

    fresh = _bare_dm(tmp_path)
    restored = fresh._load_grating_cache_from_disk()

    assert restored == {5: {"dsi": 0.4}, 6: {"dsi": 0.1}}


def test_grating_cache_prunes_ids_absent_from_cluster_df(tmp_path):
    dm = _bare_dm(tmp_path)
    dm.grating_computed_cache = {5: {"dsi": 0.4}, 99: {"dsi": 0.9}}
    dm.save_standard_plot_cache(blocking=True)

    fresh = _bare_dm(tmp_path)
    fresh.cluster_df = pd.DataFrame({"cluster_id": [5, 6]})
    assert set(fresh._load_grating_cache_from_disk()) == {5}


def test_grating_cache_stale_version_discarded(tmp_path):
    with open(tmp_path / "grating_computed_cache.pkl", "wb") as f:
        pickle.dump({5: {"dsi": 0.4}}, f)  # legacy, unversioned

    assert _bare_dm(tmp_path)._load_grating_cache_from_disk() == {}


# --------------------------------------------------------------------------
# Fix 3 — the user-facing rebuild
# --------------------------------------------------------------------------


def test_rebuild_caches_clears_ram_and_disk(tmp_path):
    dm = _bare_dm(tmp_path)
    dm.feature_cache = {1: _computed(1.0)}
    dm.standard_plot_cache = {1: {"isi": [1]}}
    dm.grating_computed_cache = {1: {"dsi": 0.4}}
    dm.save_standard_plot_cache(blocking=True)

    for name in (
        "feature_cache.pkl",
        "standard_plot_cache.pkl",
        "grating_computed_cache.pkl",
    ):
        assert (tmp_path / name).exists()

    removed = dm.rebuild_caches()

    assert sorted(removed) == [
        "feature_cache.pkl",
        "grating_computed_cache.pkl",
        "standard_plot_cache.pkl",
    ]
    assert dm.feature_cache == {}
    assert dm.standard_plot_cache == {}
    assert dm.grating_computed_cache == {}
    assert dm._physics_done_count == 0
    for name in (
        "feature_cache.pkl",
        "standard_plot_cache.pkl",
        "grating_computed_cache.pkl",
    ):
        assert not (tmp_path / name).exists()


def test_rebuild_then_save_writes_fresh_files(tmp_path):
    """Rebuild must *overwrite*, not merely delete — a recompute lands on disk."""
    dm = _bare_dm(tmp_path)
    dm.feature_cache = {1: _computed(1.0)}
    dm.save_standard_plot_cache(blocking=True)
    dm.rebuild_caches()

    dm.feature_cache = {1: _computed(9.9), 2: _computed(8.8)}
    dm.save_standard_plot_cache(blocking=True)

    fresh = _bare_dm(tmp_path)
    fresh.load_persisted_caches()
    assert set(fresh.feature_cache) == {1, 2}
    assert fresh.feature_cache[1]["timecourse"] == 9.9
