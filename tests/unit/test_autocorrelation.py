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

@pytest.mark.parametrize(
    "spikes",
    [
        np.arange(130 * 30000, 131 * 30000, 300),
        np.concatenate([
            np.arange(10 * 30000, 11 * 30000, 3000),
            np.arange(300 * 30000, 301 * 30000, 300),
        ]),
    ],
)
def test_acg_includes_late_spike_trains(mock_dm, spikes):
    """
    SPEC AC1/AC2: ACG includes spikes after 120 seconds and captures a 10 ms peak.
    """
    cluster_id = 1
    mock_dm.spike_times = spikes
    mock_dm.spike_clusters = np.full(spikes.shape, cluster_id)
    mock_dm.get_cluster_spikes = MagicMock(return_value=spikes)

    data = mock_dm._compute_standard_plots(cluster_id)

    assert data['acg_norm'] is not None, "ACG should include late spikes"
    assert len(data['acg_norm']) == 201
    assert data['acg_norm'][110] > 0.5

def test_acg_not_computed_for_too_few_spikes(mock_dm):
    """SPEC AC3: Too-few-spike clusters do not get misleading ACG data."""
    cluster_id = 2
    spikes = np.arange(0, 50 * 300, 300)
    mock_dm.spike_times = spikes
    mock_dm.spike_clusters = np.full(spikes.shape, cluster_id)
    mock_dm.get_cluster_spikes = MagicMock(return_value=spikes)

    data = mock_dm._compute_standard_plots(cluster_id)

    assert data['acg_time_lags'] is None
    assert data['acg_norm'] is None
