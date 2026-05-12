import pytest
from pathlib import Path
from src.analysis.data_manager import DataManager

REAL_DATA_PATH = Path("/home/localadmin/Documents/Development/data/sorted/20260312A/chunk12/kilosort2.5")

@pytest.mark.skipif(not REAL_DATA_PATH.exists(), reason="Real data path not found on this machine")
def test_data_manager_loads_real_kilosort_data():
    """Verify that DataManager can load real Kilosort data from the user's path."""
    dm = DataManager(REAL_DATA_PATH)
    success, message = dm.load_kilosort_data()
    
    assert success, f"Failed to load real Kilosort data: {message}"
    assert not dm.cluster_df.empty, "Cluster DataFrame is empty after loading real data"
    assert "cluster_id" in dm.cluster_df.columns
    assert "n_spikes" in dm.cluster_df.columns

@pytest.mark.skipif(not REAL_DATA_PATH.exists(), reason="Real data path not found on this machine")
def test_data_manager_loads_real_vision_data():
    """Verify that DataManager can load real Vision data from the user's path."""
    dm = DataManager(REAL_DATA_PATH)
    # Kilosort data must be loaded first to initialize some structures
    dm.load_kilosort_data()
    
    success, message = dm.load_vision_data(REAL_DATA_PATH, "kilosort2.5")
    
    assert success, f"Failed to load real Vision data: {message}"
    assert dm.vision_available, "Vision data should be marked as available"
    assert dm.vision_eis is not None, "Vision EIs should not be None"
    assert len(dm.vision_eis) > 0, "Vision EIs should not be empty"

@pytest.mark.skipif(not REAL_DATA_PATH.exists(), reason="Real data path not found on this machine")
def test_data_manager_get_cluster_spikes_real_data():
    """Verify that we can retrieve spikes for a cluster from real data."""
    dm = DataManager(REAL_DATA_PATH)
    dm.load_kilosort_data()
    
    # Get the first cluster ID
    cluster_id = dm.cluster_df["cluster_id"].iloc[0]
    spikes = dm.get_cluster_spikes(cluster_id)
    
    assert len(spikes) > 0, f"No spikes found for cluster {cluster_id} in real data"
    assert len(spikes) == dm.cluster_df.loc[0, "n_spikes"]
