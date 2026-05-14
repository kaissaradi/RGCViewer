import pytest
import numpy as np
import pandas as pd
import time
import threading
from src.analysis.data_manager import DataManager
from unittest.mock import MagicMock

@pytest.fixture
def large_mock_dm():
    """Create a DataManager with a large number of spikes for profiling."""
    dm = DataManager(kilosort_dir="/tmp")
    dm.sampling_rate = 30000
    dm.n_channels = 384
    
    n_spikes = 100_000
    n_clusters = 100
    rng = np.random.default_rng(42)
    dm.spike_times = np.sort(rng.integers(0, 3600 * 30000, n_spikes))
    dm.spike_clusters = rng.integers(0, n_clusters, n_spikes)
    dm.templates = rng.normal(0, 1, (n_clusters, 61, 384))
    dm.templates_ind = np.tile(np.arange(384), (n_clusters, 1))
    dm.channel_positions = rng.integers(0, 1000, (384, 2))
    dm.cluster_info = pd.DataFrame({'cluster_id': np.arange(n_clusters), 'group': ['good'] * n_clusters})
    return dm

@pytest.mark.performance
@pytest.mark.skip(reason="Historical reproduction of the old global-lock antipattern; kept for manual reference.")
def test_lock_contention_latency(large_mock_dm):
    """
    REPRODUCTION TEST: Demonstrate how background processing can block UI thread access.
    """
    dm = large_mock_dm
    dm.build_cluster_dataframe()
    
    # 1. Simulate a background worker thread that holds the lock for a long time
    # during a heavy "math" operation.
    def background_worker():
        print("\nBackground worker: starting heavy math and HOLDING LOCK...")
        with dm._standard_plot_lock:
            # Simulate heavy math (e.g. 500ms) while holding the lock
            time.sleep(0.5)
            dm.standard_plot_cache[999] = {'data': 'cached'}
        print("Background worker: finished and released lock.")

    bg_thread = threading.Thread(target=background_worker)
    bg_thread.start()
    
    # Give the bg thread a moment to grab the lock
    time.sleep(0.1)
    
    start_time = time.time()
    print("UI Thread: requesting data for Cluster 0 (should be fast, but BLOCKED)...")
    dm.get_standard_plot_data(0)
    end_time = time.time()
    
    latency = end_time - start_time
    print(f"UI Thread Latency: {latency:.3f} seconds")
    
    bg_thread.join()
    
    # EXPECTATION: Latency should be > 0.4s because of the lock contention
    assert latency > 0.3, "UI thread was blocked by background worker's global lock!"
