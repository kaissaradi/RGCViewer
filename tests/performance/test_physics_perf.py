import pytest
import numpy as np
import pandas as pd
from src.analysis.data_manager import DataManager
from unittest.mock import MagicMock
import time

pytestmark = pytest.mark.performance

@pytest.fixture
def mock_dm():
    dm = DataManager(kilosort_dir="/tmp")
    dm.sampling_rate = 30000
    dm.n_channels = 384
    # Mocking external data dependencies
    dm.get_cluster_spike_amplitudes = MagicMock(return_value=np.array([]))
    # Mock get_cluster_spikes to return something fast
    dm.get_cluster_spikes = MagicMock(return_value=np.arange(0, 10000, 30))
    return dm

def test_physics_loading_benchmark_sequential(benchmark, mock_dm):
    """
    Benchmark the time it takes to compute standard plot data for 50 clusters sequentially.
    """
    cluster_ids = np.arange(50)
    
    def run_sequential():
        for cid in cluster_ids:
            mock_dm.get_standard_plot_data(cid)
            
    benchmark(run_sequential)
    assert len(mock_dm.standard_plot_cache) == 50
