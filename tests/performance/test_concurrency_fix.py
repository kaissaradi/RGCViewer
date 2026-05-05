import pytest
import threading
import time
import pandas as pd
from src.analysis.data_manager import DataManager
from unittest.mock import MagicMock
import numpy as np

pytestmark = pytest.mark.performance

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

def test_concurrent_cluster_computation(large_mock_dm):
    """
    VERIFICATION TEST: Prove that calculating one cluster doesn't block another.
    """
    dm = large_mock_dm
    dm.build_cluster_dataframe()
    
    # 1. Background worker calculating Cluster 1
    # To make it take time, we'll temporarily replace _compute_standard_plots with a sleeper
    original_compute = dm._compute_standard_plots
    
    def slow_compute(cluster_id):
        if cluster_id == 1:
            print(f"\nWorker: starting SLOW compute for Cluster {cluster_id}")
            time.sleep(0.5) # Take 500ms
        return original_compute(cluster_id)
    
    dm._compute_standard_plots = slow_compute

    # Start background calculation for Cluster 1
    bg_thread = threading.Thread(target=dm.get_standard_plot_data, args=(1,))
    bg_thread.start()
    
    # Give it a moment to start the "slow" math
    time.sleep(0.1)
    
    # 2. UI thread calculating Cluster 2
    # This should be nearly instant (2ms) and NOT blocked by Cluster 1's slow compute.
    start_time = time.time()
    print("UI Thread: requesting data for Cluster 2...")
    dm.get_standard_plot_data(2)
    end_time = time.time()
    
    latency = end_time - start_time
    print(f"UI Thread Latency: {latency:.3f} seconds")
    
    bg_thread.join()
    
    # EXPECTATION: Latency should be very low (< 0.1s), NOT 0.4s-0.5s.
    assert latency < 0.1, f"UI thread was blocked! Latency was {latency:.3f}s"
    print("SUCCESS: UI thread was not blocked by background worker.")
