"""Tests for the memory that a loaded dataset holds, and for its release.

Two related behaviours are covered:

* ``load_kilosort_data`` memory-maps the spike arrays instead of copying them
  into RAM, and ``build_cluster_dataframe`` frees the sort scratch it makes.
* ``DataManager.close`` releases the resident data when the application moves
  to another run.

The DataManager is built against a synthetic Kilosort directory in
``tmp_path``, so these tests need no Qt event loop, no DataJoint and no lab
data. ``load_stim_timing`` is the one part that reaches outside; it is
replaced with a no-op.
"""

import threading

import numpy as np
import pytest

from src.analysis.data_manager import DataManager


SAMPLING_RATE = 20000.0


def _write_kilosort_dir(tmp_path, n_spikes=600, n_clusters=6):
    """Write the two .npy files that load_kilosort_data() requires.

    Spike times ascend, as Kilosort writes them. Cluster assignment cycles so
    that every cluster gets a comparable share of the spikes.
    """
    ks = tmp_path / "kilosort25" / "data000"
    ks.mkdir(parents=True)

    times = np.arange(1, n_spikes + 1, dtype=np.int64) * 20
    clusters = np.arange(n_spikes, dtype=np.int32) % n_clusters

    np.save(ks / "spike_times.npy", times)
    np.save(ks / "spike_clusters.npy", clusters)
    return ks


@pytest.fixture
def loaded_dm(tmp_path, monkeypatch):
    """A DataManager with a synthetic Kilosort run already loaded."""
    ks = _write_kilosort_dir(tmp_path)

    # The real one opens a DataJoint connection.
    monkeypatch.setattr(DataManager, "load_stim_timing", lambda self: None)

    dm = DataManager(str(ks))
    dm.sampling_rate = SAMPLING_RATE
    success, message = dm.load_kilosort_data()
    assert success, message
    return dm


# ---------------------------------------------------------------------------
# Memory mapping of the spike arrays
# ---------------------------------------------------------------------------


def test_spike_arrays_are_memory_mapped(loaded_dm):
    """The spike arrays map from the files; they are not copied into RAM."""
    assert isinstance(loaded_dm.spike_times, np.memmap)
    assert isinstance(loaded_dm.spike_clusters, np.memmap)


def test_spike_times_is_read_only(loaded_dm):
    """Nothing writes to spike_times, so it maps read-only.

    A write attempt must raise rather than pass silently: mode "r" is what
    proves no caller has started modifying the array.
    """
    assert not loaded_dm.spike_times.flags.writeable
    with pytest.raises(ValueError):
        loaded_dm.spike_times[0] = 0


def test_spike_clusters_accepts_the_refinement_write(loaded_dm):
    """update_after_refinement() writes new IDs into spike_clusters.

    The array must therefore be writable. Mode "r" would raise here, which is
    why spike_clusters maps copy-on-write instead.
    """
    assert loaded_dm.spike_clusters.flags.writeable

    new_id = int(loaded_dm.spike_clusters.max()) + 1
    loaded_dm.spike_clusters[:5] = new_id

    assert np.all(loaded_dm.spike_clusters[:5] == new_id)


def test_spike_clusters_write_does_not_reach_the_file(loaded_dm):
    """A copy-on-write map keeps the change private to this process.

    The Kilosort output is the experiment's record. Refinement inside the
    application must never alter it.
    """
    path = loaded_dm.kilosort_dir / "spike_clusters.npy"
    before = np.load(path)

    loaded_dm.spike_clusters[:5] = int(loaded_dm.spike_clusters.max()) + 1

    after = np.load(path)
    assert np.array_equal(before, after)


def test_cluster_spike_indices_still_group_correctly(loaded_dm):
    """Mapping must not change what the per-cluster index returns."""
    for cid, idxs in loaded_dm.cluster_spike_indices.items():
        assert np.all(np.asarray(loaded_dm.spike_clusters)[idxs] == cid)


def test_load_array_falls_back_when_mapping_fails(tmp_path, monkeypatch):
    """Some mounts cannot map a file. Loading it still has to work."""
    from src.analysis import data_manager as dm_mod

    path = tmp_path / "spike_times.npy"
    np.save(path, np.arange(10, dtype=np.int64))

    real_load = np.load

    def no_mapping(p, *args, mmap_mode=None, **kwargs):
        if mmap_mode is not None:
            raise OSError("this filesystem cannot memory-map")
        return real_load(p, *args, **kwargs)

    monkeypatch.setattr(dm_mod.np, "load", no_mapping)

    arr = dm_mod._load_array(path, "r")

    assert np.array_equal(arr, np.arange(10))
    assert not isinstance(arr, np.memmap)


def test_fallback_keeps_read_only_arrays_read_only(tmp_path, monkeypatch):
    """The guarantee must not depend on which machine the app runs on."""
    from src.analysis import data_manager as dm_mod

    path = tmp_path / "spike_times.npy"
    np.save(path, np.arange(10, dtype=np.int64))

    real_load = np.load
    monkeypatch.setattr(
        dm_mod.np,
        "load",
        lambda p, *a, mmap_mode=None, **k: (
            (_ for _ in ()).throw(OSError("no mapping"))
            if mmap_mode is not None
            else real_load(p, *a, **k)
        ),
    )

    arr = dm_mod._load_array(path, "r")

    assert not arr.flags.writeable


def test_fallback_keeps_writable_arrays_writable(tmp_path, monkeypatch):
    """Mode "c" exists because refinement writes to the array."""
    from src.analysis import data_manager as dm_mod

    path = tmp_path / "spike_clusters.npy"
    np.save(path, np.zeros(10, dtype=np.int32))

    real_load = np.load
    monkeypatch.setattr(
        dm_mod.np,
        "load",
        lambda p, *a, mmap_mode=None, **k: (
            (_ for _ in ()).throw(OSError("no mapping"))
            if mmap_mode is not None
            else real_load(p, *a, **k)
        ),
    )

    arr = dm_mod._load_array(path, "c")

    assert arr.flags.writeable
    arr[0] = 7
    assert arr[0] == 7


def test_get_cluster_spikes_returns_a_plain_copy(loaded_dm):
    """Callers get an ordinary writable array, not a view of the map."""
    cid = next(iter(loaded_dm.cluster_spike_indices))
    spikes = loaded_dm.get_cluster_spikes(cid)

    assert len(spikes) > 0
    assert spikes.flags.writeable


# ---------------------------------------------------------------------------
# Release of the sort scratch
# ---------------------------------------------------------------------------


def test_sort_scratch_is_freed_after_the_isi_pass(loaded_dm):
    """The sorted copies are two more full-length spike arrays.

    build_cluster_dataframe() is their only consumer, so it must drop them
    rather than hold them for the rest of the session.
    """
    assert loaded_dm._spk_sorted_cls is not None
    assert loaded_dm._spk_sorted_t is not None

    loaded_dm.build_cluster_dataframe()

    assert loaded_dm._spk_sorted_cls is None
    assert loaded_dm._spk_sorted_t is None


def test_isi_pass_still_runs_without_the_scratch(loaded_dm):
    """Dropping the scratch must not break a second pass.

    build_cluster_dataframe() recomputes the sort when the scratch is absent,
    so calling it twice gives the same ISI figures.
    """
    loaded_dm.build_cluster_dataframe()
    first = loaded_dm.cluster_df.set_index("cluster_id")["isi_violations_pct"]

    loaded_dm.build_cluster_dataframe()
    second = loaded_dm.cluster_df.set_index("cluster_id")["isi_violations_pct"]

    assert first.to_dict() == second.to_dict()


# ---------------------------------------------------------------------------
# DataManager.close
# ---------------------------------------------------------------------------


def _closable_dm(tmp_path):
    """A DataManager carrying stand-ins for the heavy attributes."""
    dm = DataManager.__new__(DataManager)
    dm._closed = False
    dm.kilosort_dir = tmp_path
    dm.raw_reader = None
    dm.main_window = object()

    dm.vision_eis = {1: np.zeros(4)}
    dm.vision_stas = {1: "sta"}
    dm.spike_times = np.arange(4)
    dm.spike_clusters = np.zeros(4)
    dm.feature_cache = {1: {"_computed": True}}
    dm.standard_plot_cache = {1: {"acg": None}}
    dm.grating_computed_cache = {1: {}}
    return dm


def test_close_releases_the_heavy_attributes(tmp_path):
    dm = _closable_dm(tmp_path)

    dm.close()

    assert dm.vision_eis is None
    assert dm.vision_stas is None
    assert dm.spike_times is None
    assert dm.spike_clusters is None
    assert dm.feature_cache == {}
    assert dm.standard_plot_cache == {}
    assert dm.grating_computed_cache == {}


def test_close_drops_the_window_reference(tmp_path):
    """The back-reference is what keeps the object reachable from Qt."""
    dm = _closable_dm(tmp_path)

    dm.close()

    assert dm.main_window is None


def test_close_rebinds_caches_rather_than_clearing_them(tmp_path):
    """A background save holding one of these caches must still finish.

    Rebinding leaves the save's own reference intact. Clearing in place would
    empty the dict underneath it part way through the iteration.
    """
    dm = _closable_dm(tmp_path)
    held = dm.feature_cache

    dm.close()

    assert held == {1: {"_computed": True}}
    assert dm.feature_cache is not held


def test_close_is_repeatable(tmp_path):
    """Teardown can be reached twice — from a run switch and from exit."""
    dm = _closable_dm(tmp_path)

    dm.close()
    dm.feature_cache = {99: {"_computed": True}}
    dm.close()

    # The second call is a no-op, so it must not wipe what came after the first.
    assert dm.feature_cache == {99: {"_computed": True}}


def test_close_survives_a_failing_raw_reader(tmp_path):
    """Teardown must not raise, or the new dataset is left half-loaded."""

    class BadReader:
        def __exit__(self, *args):
            raise OSError("file handle already gone")

    dm = _closable_dm(tmp_path)
    dm.raw_reader = BadReader()

    dm.close()

    assert dm.vision_eis is None


def test_close_reports_itself_closed(tmp_path):
    dm = _closable_dm(tmp_path)
    assert dm._closed is False

    dm.close()

    assert dm._closed is True


def test_close_on_a_fully_loaded_manager(loaded_dm):
    """The real attribute set, not the stand-ins: nothing is missing."""
    loaded_dm.build_cluster_dataframe()

    loaded_dm.close()

    assert loaded_dm.spike_times is None
    assert loaded_dm.spike_clusters is None
    assert loaded_dm.cluster_spike_indices is None
    assert loaded_dm.main_window is None


def test_close_leaves_the_locks_usable(loaded_dm):
    """Panels can still reach a released manager; the locks must survive.

    Releasing them would turn a late access into an AttributeError instead of
    the usual empty-cache path.
    """
    loaded_dm.close()

    assert isinstance(loaded_dm._feature_lock, type(threading.Lock()))
    with loaded_dm._feature_lock:
        assert loaded_dm.feature_cache == {}


# ---------------------------------------------------------------------------
# _release_previous_dataset — when the release actually happens
# ---------------------------------------------------------------------------


class _FakeSignal:
    """Stands in for QThread.finished."""

    def __init__(self):
        self._slots = []

    def connect(self, slot):
        self._slots.append(slot)

    def emit(self):
        for slot in list(self._slots):
            slot()


class _FakeThread:
    def __init__(self):
        self.finished = _FakeSignal()


class _FakeDataManager:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class _FakeWindow:
    def __init__(self, data_manager):
        self.data_manager = data_manager


@pytest.fixture
def release(monkeypatch):
    """_release_previous_dataset with its two collaborators controllable.

    The test drives what _retire_inflight_load reports, which is the only
    input that decides whether the release is immediate or deferred.
    """
    from src.gui import callbacks

    stopped = []
    parked = []

    monkeypatch.setattr(callbacks, "stop_worker", lambda mw: stopped.append(mw))
    monkeypatch.setattr(callbacks, "_retire_inflight_load", lambda mw: list(parked))

    return callbacks._release_previous_dataset, stopped, parked


def test_release_closes_immediately_when_no_load_is_running(release):
    release_fn, stopped, _parked = release
    dm = _FakeDataManager()
    window = _FakeWindow(dm)

    release_fn(window)

    assert dm.close_calls == 1
    assert stopped == [window]


def test_release_waits_for_a_load_still_in_flight(release):
    """A running load owns the manager from another thread.

    Closing it under the worker would free the arrays it is still writing to,
    which crashes the process rather than raising.
    """
    release_fn, _stopped, parked = release
    thread = _FakeThread()
    parked.append(thread)

    dm = _FakeDataManager()
    release_fn(_FakeWindow(dm))

    assert dm.close_calls == 0

    thread.finished.emit()
    assert dm.close_calls == 1


def test_release_waits_for_every_parked_thread(release):
    """Both a Kilosort and a Vision load can be parked at once."""
    release_fn, _stopped, parked = release
    first, second = _FakeThread(), _FakeThread()
    parked.extend([first, second])

    dm = _FakeDataManager()
    release_fn(_FakeWindow(dm))

    first.finished.emit()
    assert dm.close_calls == 0, "released while the second load was still running"

    second.finished.emit()
    assert dm.close_calls == 1


def test_release_closes_once_even_if_a_thread_reports_twice(release):
    release_fn, _stopped, parked = release
    thread = _FakeThread()
    parked.append(thread)

    dm = _FakeDataManager()
    release_fn(_FakeWindow(dm))

    thread.finished.emit()
    thread.finished.emit()

    assert dm.close_calls == 1


def test_release_tolerates_no_manager_yet(release):
    """The first load of a session has nothing to replace."""
    release_fn, stopped, _parked = release
    window = _FakeWindow(None)

    release_fn(window)

    assert stopped == [window]
