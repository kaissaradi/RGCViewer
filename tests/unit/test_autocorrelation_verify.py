import pytest
import numpy as np
from src.analysis.data_manager import DataManager
from unittest.mock import MagicMock

@pytest.fixture
def mock_dm():
    """Create a DataManager with minimal mocked attributes for plot data testing."""
    dm = DataManager(kilosort_dir="/tmp")
    dm.sampling_rate = 30000
    dm.n_channels = 384
    # Mocking external data dependencies
    dm.get_cluster_spike_amplitudes = MagicMock(return_value=np.array([]))
    return dm

def test_acg_includes_late_spikes(mock_dm):
    """
    VERIFICATION TEST: Confirm that ACG now includes spikes occurring after 120 seconds.
    """
    cluster_id = 1
    sr = mock_dm.sampling_rate
    
    # Create 100 spikes occurring at 5 minutes (300s)
    # 300s * 30000Hz = 9,000,000 samples
    # Spikes every 300 samples (10ms)
    late_spikes = np.arange(9000000, 9000000 + 100*300, 300) 
    
    mock_dm.get_cluster_spikes = MagicMock(return_value=late_spikes)
    mock_dm.spike_times = late_spikes
    mock_dm.spike_clusters = np.full(late_spikes.shape, cluster_id)
    
    data = mock_dm._compute_standard_plots(cluster_id)
    
    assert data['acg_norm'] is not None
    assert len(data['acg_norm']) == 201
    
    val_at_10ms = data['acg_norm'][110]
    assert val_at_10ms > 0, "ACG should have a peak at 10ms for these late spikes"

def test_acg_integrates_entire_recording(mock_dm):
    """
    VERIFICATION TEST: Confirm that ACG integrates both early and late spikes.
    """
    cluster_id = 2
    sr = mock_dm.sampling_rate
    
    # Early spikes (10s): 100ms apart (10Hz)
    early = np.arange(10*sr, 20*sr, 3000) 
    # Late spikes (500s): 10ms apart (100Hz)
    late = np.arange(500*sr, 510*sr, 300)
    
    all_spikes = np.concatenate([early, late])
    mock_dm.get_cluster_spikes = MagicMock(return_value=all_spikes)
    mock_dm.spike_times = all_spikes
    mock_dm.spike_clusters = np.full(all_spikes.shape, cluster_id)
    
    data = mock_dm._compute_standard_plots(cluster_id)
    
    val_at_10ms = data['acg_norm'][110]
    assert val_at_10ms > 0, "ACG should capture the 100Hz burst from late in the recording"
