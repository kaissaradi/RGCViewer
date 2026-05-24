from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.analysis.data_manager import DataManager
from src.gui import callbacks


class SimilarTemplatesSpy:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)
        self.getitem_calls = 0

    @property
    def shape(self):
        return self._values.shape

    def __getitem__(self, item):
        self.getitem_calls += 1
        return self._values[item]


class FakeSTA:
    def __init__(self, red):
        self.red = np.asarray(red, dtype=float)
        self.green = self.red + 1.0
        self.blue = self.red * 2.0


class FakeLazySTADict:
    def __init__(self, stas_by_id):
        self._stas_by_id = stas_by_id
        self.keys_list = list(stas_by_id)
        self.getitem_calls = 0

    def __bool__(self):
        return True

    def __contains__(self, item):
        return item in self._stas_by_id

    def __getitem__(self, item):
        self.getitem_calls += 1
        return self._stas_by_id[item]
    
    def keys(self):              # ← add this
        return self.keys_list


class FakeVisionLoadThread:
    def __init__(self):
        self.quit_calls = 0
        self.wait_calls = []

    def quit(self):
        self.quit_calls += 1

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        return True


class FakeVisionLoadWorker:
    def __init__(self):
        self.delete_later_calls = 0

    def deleteLater(self):
        self.delete_later_calls += 1


class FakeCentralWidget:
    def __init__(self):
        self.enabled_values = []

    def setEnabled(self, enabled):
        self.enabled_values.append(enabled)


class FakeStatusBar:
    def __init__(self):
        self.messages = []

    def showMessage(self, *args):
        self.messages.append(args)


def test_kilosort_params_uses_literal_eval_without_executing_code(tmp_path):
    marker = tmp_path / "literal_eval_executed.txt"
    params_path = tmp_path / "params.py"
    params_path.write_text(
        "\n".join(
            [
                "fs = 20000",
                "n_channels_dat = 64",
                "dtype = 'int16'",
                "dat_path = ('raw_a.dat', 'raw_b.dat')",
                "channel_map = [0, 1, 2]",
                f"dangerous = __import__('pathlib').Path({str(marker)!r}).write_text('ran')",
            ]
        )
    )

    dm = DataManager(kilosort_dir=str(tmp_path))

    dm._load_kilosort_params()

    assert dm.sampling_rate == 20000
    assert dm.n_channels == 64
    assert not marker.exists()


def test_mea_similarity_table_reuses_cluster_cache(tmp_path):
    dm = DataManager(kilosort_dir=str(tmp_path))
    dm.similar_templates = SimilarTemplatesSpy(
        [
            [1.0, 0.9, 0.3],
            [0.9, 1.0, 0.2],
            [0.3, 0.2, 1.0],
        ]
    )
    dm.cluster_df = pd.DataFrame(
        {
            "cluster_id": [10, 11, 12],
            "n_spikes": [100, 80, 60],
            "status": ["good", "dup", "noise"],
            "set": ["keep", "review", "drop"],
            "x_um": [0.0, 3.0, 4.0],
            "y_um": [0.0, 4.0, 0.0],
        }
    )
    dm.cluster_to_template = {10: 0, 11: 1, 12: 2}
    dm.cluster_id_to_idx = {10: 0, 11: 1, 12: 2}
    dm.mea_sim_cache = {}

    first = dm._get_mea_similarity_table(10, top_n=2)

    assert dm.similar_templates.getitem_calls == 1
    assert 10 in dm.mea_sim_cache
    assert list(first["cluster_id"]) == [11, 12]
    assert list(first["template_sim"]) == [0.9, 0.3]
    assert list(first.columns) == [
        "cluster_id",
        "n_spikes",
        "status",
        "distance_um",
        "template_sim",
        "set",
    ]

    dm.similar_templates.getitem_calls = 0
    second = dm._get_mea_similarity_table(10, top_n=2)

    assert dm.similar_templates.getitem_calls == 0
    pd.testing.assert_frame_equal(second, first)
    assert second is not dm.mea_sim_cache[10]


def test_vision_similarity_table_reuses_cluster_cache(tmp_path):
    dm = DataManager(kilosort_dir=str(tmp_path))
    dm.is_vision_only = True
    dm.vision_stas = FakeLazySTADict(
        {
            10: FakeSTA([[0.0, 1.0], [2.0, 3.0]]),
            11: FakeSTA([[0.0, 2.0], [4.0, 6.0]]),
            12: FakeSTA([[3.0, 2.0], [1.0, 0.0]]),
        }
    )
    dm.vision_params = object()
    dm.cluster_df = pd.DataFrame(
        {
            "cluster_id": [10, 11, 12],
            "n_spikes": [100, 80, 60],
            "status": ["good", "dup", "noise"],
            "set": ["keep", "review", "drop"],
        }
    )
    dm.vision_sim_cache = {}

    first = dm._get_vision_similarity_table(10, top_n=2)

    assert dm.vision_stas.getitem_calls > 0
    assert 10 in dm.vision_sim_cache
    assert list(first["cluster_id"]) == [11, 12]
    assert list(first.columns) == [
        "cluster_id",
        "n_spikes",
        "status",
        "set",
        "template_sim",
    ]

    dm.vision_stas.getitem_calls = 0
    second = dm._get_vision_similarity_table(10, top_n=2)

    assert dm.vision_stas.getitem_calls == 0
    pd.testing.assert_frame_equal(second, first)
    assert second is not dm.vision_sim_cache[10]


@pytest.mark.parametrize(
    ("thread_state", "worker_state"),
    [
        ("missing", "missing"),
        ("cleared", "cleared"),
        ("present", "present"),
    ],
)
def test_on_vision_native_loaded_handles_stale_thread_and_worker_cleanup(
    monkeypatch, thread_state, worker_state
):
    critical_calls = []
    monkeypatch.setattr(
        callbacks.QMessageBox,
        "critical",
        lambda *args: critical_calls.append(args),
    )

    main_window = SimpleNamespace(
        central_widget=FakeCentralWidget(),
        status_bar=FakeStatusBar(),
    )

    thread = None
    if thread_state == "cleared":
        main_window.vision_load_thread = None
    elif thread_state == "present":
        thread = FakeVisionLoadThread()
        main_window.vision_load_thread = thread

    worker = None
    if worker_state == "cleared":
        main_window.vision_load_worker = None
    elif worker_state == "present":
        worker = FakeVisionLoadWorker()
        main_window.vision_load_worker = worker

    callbacks._on_vision_native_loaded(
        main_window,
        success=False,
        message="standalone Vision load failed",
        vision_dir_name="/tmp/fake-vision",
    )

    assert critical_calls
    assert main_window.status_bar.messages == [("Loading failed.", 5000)]
    assert main_window.central_widget.enabled_values == [True]

    if thread is not None:
        assert thread.quit_calls == 1
        assert thread.wait_calls == [2000]
    if worker is not None:
        assert worker.delete_later_calls == 1

    if thread_state != "missing":
        assert main_window.vision_load_thread is None
    if worker_state != "missing":
        assert main_window.vision_load_worker is None


# ---------------------------------------------------------------------------
# AC4 — _load_standard_plot_cache_from_disk() must NOT hold _standard_plot_lock
#        during pickle.load().
# ---------------------------------------------------------------------------

def test_standard_plot_lock_not_held_during_pickle_load(tmp_path, monkeypatch):
    """
    _load_standard_plot_cache_from_disk() must release _standard_plot_lock
    before calling pickle.load() so that concurrent get_standard_plot_data()
    calls are never blocked during the I/O.

    Strategy
    --------
    1. Construct DataManager with an empty tmp_path (no .pkl → early exit in
       __init__, so the monkeypatch fires on the *explicit* second call only).
    2. Write a real .pkl so the existence check passes on the second call.
    3. Monkeypatch pickle.load to:
         a. Set a threading.Event so the spy thread knows I/O has started.
         b. Sleep 200 ms (simulates slow disk).
         c. Return the cache dict.
    4. A spy thread waits for the event, then tries a *non-blocking* acquire of
       _standard_plot_lock and records whether it succeeded.
    5. After the load returns, assert the spy acquired the lock — meaning the
       lock was free during the I/O window.
    """
    import pickle
    import threading
    import time

    # --- Build a minimal DataManager (no .pkl yet → __init__ early-exits) ---
    dm = DataManager(kilosort_dir=str(tmp_path))

    # --- Write a real cache file so the method proceeds past existence check ---
    cache_data = {42: {"acg_norm": None}}
    cache_pkl = tmp_path / "standard_plot_cache.pkl"
    with open(cache_pkl, "wb") as f:
        pickle.dump(cache_data, f)

    # --- Reset the in-memory cache so the method doesn't early-exit again ---
    dm.standard_plot_cache = {}

    # --- Threading primitives ---
    io_started = threading.Event()
    spy_acquired_lock = threading.Event()

    real_pickle_load = pickle.load

    def slow_pickle_load(file_obj):
        io_started.set()          # signal spy: we're inside pickle.load now
        time.sleep(0.2)           # simulate disk latency
        return real_pickle_load(file_obj)

    monkeypatch.setattr("pickle.load", slow_pickle_load)

    def spy():
        io_started.wait(timeout=5.0)   # block until pickle.load has started
        # Non-blocking: succeeds only if the lock is currently FREE
        acquired = dm._standard_plot_lock.acquire(blocking=False)
        if acquired:
            spy_acquired_lock.set()
            dm._standard_plot_lock.release()

    spy_thread = threading.Thread(target=spy, daemon=True)
    spy_thread.start()

    dm._load_standard_plot_cache_from_disk()

    spy_thread.join(timeout=5.0)
    assert not spy_thread.is_alive(), "Spy thread timed out — deadlock?"
    assert spy_acquired_lock.is_set(), (
        "_standard_plot_lock was held during pickle.load(); "
        "concurrent get_standard_plot_data() calls would have been blocked."
    )
    # Sanity: the cache was actually loaded
    assert dm.standard_plot_cache == cache_data


# ---------------------------------------------------------------------------
# AC5 — get_cell_physics() must compute exactly once under concurrency for
#        the same cluster_id.
# ---------------------------------------------------------------------------

def test_get_cell_physics_computes_once_under_concurrency(tmp_path, monkeypatch):
    """
    Two threads calling get_cell_physics() for the same cluster_id at the
    same time must result in vision_stas.__getitem__ being called exactly once.

    The current bug: cell_lock closes *before* vision_stas[vid] is accessed,
    so both threads pass the double-check and both execute the expensive STA
    extraction in parallel.

    Strategy
    --------
    1. Build a minimal DataManager with a SlowFakeLazySTADict whose
       __getitem__ sleeps 100 ms (widens the race window deterministically).
    2. Monkeypatch get_standard_plot_data to return a stub so the test is
       pure-unit and needs no real spike arrays.
    3. Use a threading.Barrier(2) to release both threads simultaneously at
       the call site, maximising the chance of interleaving.
    4. Join both threads, then assert getitem_calls == 1.
    """
    import threading
    import time

    # --- Slow STA spy: sleeps inside __getitem__ to widen the race window ---
    class SlowFakeLazySTADict:
        def __init__(self):
            self.getitem_calls = 0
            self._lock = threading.Lock()   # just for atomic counter increment

        def __bool__(self):
            return True

        def __contains__(self, item):
            return item == 6   # cluster_id=5 → vid=6 (hybrid mode, not vision_only)

        def __getitem__(self, item):
            with self._lock:
                self.getitem_calls += 1
            time.sleep(0.1)    # hold long enough for the second thread to enter
            sta = FakeSTA([[0.0, 1.0], [1.0, 0.0]])
            return sta

        def keys(self):
            return [6]

    # --- Build DataManager with enough state to reach the STA access ---
    dm = DataManager(kilosort_dir=str(tmp_path))
    dm.vision_stas = SlowFakeLazySTADict()
    dm.vision_params = None   # skips stafit / timecourse branches gracefully
    dm.is_vision_only = False # cluster_id=5 → vid=6

    # Stub out get_standard_plot_data so we don't need real spike arrays
    monkeypatch.setattr(
        dm,
        "get_standard_plot_data",
        lambda cid: {"acg_norm": None},
    )

    # Clear feature cache so both threads miss on the fast path
    dm.feature_cache = {}
    dm._physics_done_count = 0

    cluster_id = 5   # vid = 6 (present in SlowFakeLazySTADict)

    barrier = threading.Barrier(2)
    errors = []

    def call_physics():
        try:
            barrier.wait(timeout=5.0)   # release both threads simultaneously
            dm.get_cell_physics(cluster_id)
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=call_physics, daemon=True)
    t2 = threading.Thread(target=call_physics, daemon=True)
    t1.start()
    t2.start()
    t1.join(timeout=10.0)
    t2.join(timeout=10.0)

    assert not t1.is_alive(), "Thread 1 timed out — possible deadlock"
    assert not t2.is_alive(), "Thread 2 timed out — possible deadlock"
    assert not errors, f"Exception(s) in worker threads: {errors}"

    assert dm.vision_stas.getitem_calls == 1, (
        f"Expected vision_stas.__getitem__ called exactly once; "
        f"got {dm.vision_stas.getitem_calls}. "
        f"cell_lock scope does not cover the Vision STA extraction."
    )