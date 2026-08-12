import threading
import pickle
from unittest.mock import MagicMock

import numpy as np

from src.analysis.data_manager import DataManager


def test_standard_plot_cache_computes_same_cluster_once(tmp_path):
    dm = DataManager(kilosort_dir=str(tmp_path))
    compute_started = threading.Event()
    release_compute = threading.Event()
    results = []

    def compute(cluster_id):
        compute_started.set()
        assert release_compute.wait(timeout=2)
        return {"cluster_id": cluster_id}

    dm._compute_standard_plots = MagicMock(side_effect=compute)

    threads = [
        threading.Thread(target=lambda: results.append(dm.get_standard_plot_data(7)))
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()

    assert compute_started.wait(timeout=1)
    release_compute.set()

    for thread in threads:
        thread.join(timeout=2)

    assert len(results) == 2
    assert results[0] == {"cluster_id": 7}
    assert results[1] == {"cluster_id": 7}
    dm._compute_standard_plots.assert_called_once_with(7)


def test_standard_plot_cache_allows_different_clusters_to_compute_concurrently(tmp_path):
    dm = DataManager(kilosort_dir=str(tmp_path))
    entered = []
    entered_lock = threading.Lock()
    both_entered = threading.Event()
    release_compute = threading.Event()

    def compute(cluster_id):
        with entered_lock:
            entered.append(cluster_id)
            if len(entered) == 2:
                both_entered.set()
        assert release_compute.wait(timeout=2)
        return {"cluster_id": cluster_id}

    dm._compute_standard_plots = MagicMock(side_effect=compute)

    threads = [
        threading.Thread(target=dm.get_standard_plot_data, args=(1,)),
        threading.Thread(target=dm.get_standard_plot_data, args=(2,)),
    ]
    for thread in threads:
        thread.start()

    assert both_entered.wait(timeout=1), "Different clusters should not block on one cell lock"
    release_compute.set()

    for thread in threads:
        thread.join(timeout=2)

    assert sorted(entered) == [1, 2]


def test_disk_cache_bypasses_computation(tmp_path):
    """
    AC1: If a valid cache file exists on disk, DataManager loads it directly
    without triggering the background FFT/math logic.
    """
    cluster_id = 99
    fake_cache_data = {"cluster_id": cluster_id, "acg": [1, 2, 3], "from_disk": True}

    # 1. Write the cache to disk BEFORE initializing DataManager
    cache_file = tmp_path / "standard_plot_cache.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump({cluster_id: fake_cache_data}, f)

    # 2. Initialize DataManager. It *should* load the cache from disk.
    dm = DataManager(kilosort_dir=str(tmp_path))

    # Mock the compute function so we can strictly verify it NEVER gets called
    dm._compute_standard_plots = MagicMock(return_value={"cluster_id": cluster_id, "computed_live": True})

    # 3. Fetch the data (DO NOT manually populate dm.standard_plot_cache)
    result = dm.get_standard_plot_data(cluster_id)

    # 4. Assert that it pulled our fake disk data and never touched the math function
    assert result is not None, "DataManager returned None instead of cache data."
    assert result.get("from_disk") is True, "DataManager bypassed the disk cache and computed live data."
    dm._compute_standard_plots.assert_not_called()


def test_cell_physics_marks_cluster_computed_without_vision_sta(tmp_path):
    dm = DataManager(kilosort_dir=str(tmp_path))
    dm.vision_stas = {}
    dm.get_standard_plot_data = MagicMock(
        return_value={"acg_norm": np.array([0.1, 0.2, 0.3])}
    )

    metrics = dm.get_cell_physics(26)

    assert metrics["_computed"] is True
    assert metrics["acg"].tolist() == [0.1, 0.2, 0.3]
    assert metrics["timecourse"] is None
    assert metrics["rf_area"] == 0.0
    assert dm.feature_cache[26] == metrics
    assert dm._physics_done_count == 1


class FakeLitkeReader:
    length = 100

    def get_data(self, start_sample, n_samples):
        rows = np.arange(5, dtype=np.float32)[:, None] * 100
        cols = np.arange(start_sample, start_sample + n_samples, dtype=np.float32)
        return rows + cols


def test_raw_trace_snippet_skips_litke_ttl_row(tmp_path):
    dm = DataManager(kilosort_dir=str(tmp_path))
    dm.raw_reader = FakeLitkeReader()
    dm.raw_data_memmap = None
    dm.n_samples = 100
    dm.n_channels = 4
    dm.uV_per_bit = 1.0

    snippet = dm.get_raw_trace_snippet([0, 2], 10, 13)

    expected = np.array([
        [110.0, 111.0, 112.0],
        [310.0, 311.0, 312.0],
    ], dtype=np.float32)
    np.testing.assert_array_equal(snippet, expected)


def test_apply_ei_updates_keeps_max_dup_r_as_float(tmp_path):
    import pandas as pd

    dm = DataManager(kilosort_dir=str(tmp_path))
    dm.cluster_df = pd.DataFrame(
        {
            "cluster_id": [1, 2, 3],
            "potential_dups": [False, False, False],
            "max_dup_r": [0.0, 0.0, 0.0],
        }
    )
    class _Model:
        def __init__(self):
            self.refreshed = False

        def refresh_view(self):
            self.refreshed = True

    class _MW:
        def __init__(self):
            self.main_cluster_model = _Model()
            self.tree_updated = False

        def _update_tree_view_duplicate_highlight(self):
            self.tree_updated = True

    mw = _MW()
    dm.main_window = mw
    dm._apply_ei_updates({1: True}, {1: 0.864, 2: 0.0})

    assert dm.cluster_df["max_dup_r"].dtype.kind == "f"
    assert float(dm.cluster_df.loc[0, "max_dup_r"]) == 0.864
    assert bool(dm.cluster_df.loc[0, "potential_dups"]) is True
    assert bool(dm.cluster_df.loc[2, "potential_dups"]) is False
    assert mw.main_cluster_model.refreshed is True
    assert mw.tree_updated is True
