import pytest
import numpy as np
import pandas as pd
import threading
from src.analysis.data_manager import DataManager

pytestmark = pytest.mark.performance

@pytest.fixture
def large_mock_dm():
    """Create a DataManager with a large number of spikes."""
    dm = DataManager(kilosort_dir="/tmp")
    dm.sampling_rate = 30000
    dm.n_channels = 384
    
    n_spikes = 500_000
    n_clusters = 500
    rng = np.random.default_rng(42)
    dm.spike_times = np.sort(rng.integers(0, 3600 * 30000, n_spikes))
    dm.spike_clusters = rng.integers(0, n_clusters, n_spikes)
    dm.templates = rng.normal(0, 1, (n_clusters, 61, 384))
    dm.templates_ind = np.tile(np.arange(384), (n_clusters, 1))
    dm.channel_positions = rng.integers(0, 1000, (384, 2))
    dm.cluster_info = pd.DataFrame({'cluster_id': np.arange(n_clusters), 'group': ['good'] * n_clusters})
    return dm

def test_first_click_latency_under_load(benchmark, large_mock_dm):
    """
    Benchmark how long it takes to get plot data for a cell 
    while the background worker is heavily processing other cells.
    """
    dm = large_mock_dm
    dm.build_cluster_dataframe()
    
    # 1. Background worker calculating many clusters
    stop_bg = False
    def background_worker():
        for cid in range(10, 500):
            if stop_bg: break
            dm.get_standard_plot_data(cid)
            
    bg_thread = threading.Thread(target=background_worker)
    bg_thread.start()
    
    # 2. Benchmark the "First Click" (getting data for Cluster 0)
    # This should stay fast regardless of the background worker.
    def get_first_cell():
        return dm.get_standard_plot_data(0)
        
    benchmark(get_first_cell)
    
    stop_bg = True
    bg_thread.join()
    
    assert dm.standard_plot_cache.get(0) is not None
