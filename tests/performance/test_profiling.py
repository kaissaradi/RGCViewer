import pytest
import numpy as np
import pandas as pd
import time
import cProfile
import pstats
import io
from src.analysis.data_manager import DataManager
from unittest.mock import MagicMock

pytestmark = [
    pytest.mark.performance,
    pytest.mark.profiling,
    pytest.mark.skip(reason="Manual profiling helper; run explicitly while optimizing a hotspot."),
]

def profile_function(func, *args, **kwargs):
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats(20) # Top 20 functions
    return result, s.getvalue()

@pytest.fixture
def large_mock_dm():
    """Create a DataManager with a large number of spikes for profiling."""
    dm = DataManager(kilosort_dir="/tmp")
    dm.sampling_rate = 30000
    dm.n_channels = 384
    
    # 1,000,000 spikes spread across 500 clusters
    n_spikes = 1_000_000
    n_clusters = 500
    rng = np.random.default_rng(42)
    dm.spike_times = np.sort(rng.integers(0, 3600 * 30000, n_spikes))
    dm.spike_clusters = rng.integers(0, n_clusters, n_spikes)
    
    # Mock templates and other data
    dm.templates = rng.normal(0, 1, (n_clusters, 61, 384))
    dm.templates_ind = np.tile(np.arange(384), (n_clusters, 1))
    dm.channel_positions = rng.integers(0, 1000, (384, 2))
    
    # Mock cluster_info
    dm.cluster_info = pd.DataFrame({
        'cluster_id': np.arange(n_clusters),
        'group': ['good'] * n_clusters
    })
    
    return dm

def test_profile_build_cluster_dataframe(large_mock_dm):
    """Profile the initial build_cluster_dataframe method."""
    print("\n" + "="*50)
    print("PROFILING: build_cluster_dataframe")
    print("="*50)
    
    result, profile_output = profile_function(large_mock_dm.build_cluster_dataframe)
    print(profile_output)

def test_profile_compute_standard_plots(large_mock_dm):
    """Profile the math inside _compute_standard_plots for a large cluster."""
    # Find a cluster with many spikes
    cluster_ids, counts = np.unique(large_mock_dm.spike_clusters, return_counts=True)
    target_id = cluster_ids[np.argmax(counts)]
    
    print("\n" + "="*50)
    print(f"PROFILING: _compute_standard_plots for Cluster {target_id} ({counts[np.argmax(counts)]} spikes)")
    print("="*50)
    
    # Pre-cache spike info to isolate the math
    large_mock_dm.build_cluster_dataframe()
    
    result, profile_output = profile_function(large_mock_dm._compute_standard_plots, target_id)
    print(profile_output)

def test_profile_get_cell_physics(large_mock_dm):
    """Profile get_cell_physics overhead."""
    large_mock_dm.build_cluster_dataframe()
    target_id = 0
    
    # Mock Vision STAs
    rng = np.random.default_rng(43)
    large_mock_dm.vision_stas = {target_id + 1: rng.normal(0, 1, (50, 512))}
    large_mock_dm.vision_params = MagicMock()
    
    print("\n" + "="*50)
    print(f"PROFILING: get_cell_physics for Cluster {target_id}")
    print("="*50)
    
    result, profile_output = profile_function(large_mock_dm.get_cell_physics, target_id)
    print(profile_output)
