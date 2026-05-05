import threading
from unittest.mock import MagicMock

import numpy as np

from src.analysis.data_manager import DataManager


def test_standard_plot_cache_computes_same_cluster_once():
    dm = DataManager(kilosort_dir="/tmp")
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


def test_standard_plot_cache_allows_different_clusters_to_compute_concurrently():
    dm = DataManager(kilosort_dir="/tmp")
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


class FakeLitkeReader:
    length = 100

    def get_data(self, start_sample, n_samples):
        rows = np.arange(5, dtype=np.float32)[:, None] * 100
        cols = np.arange(start_sample, start_sample + n_samples, dtype=np.float32)
        return rows + cols


def test_raw_trace_snippet_skips_litke_ttl_row():
    dm = DataManager(kilosort_dir="/tmp")
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
