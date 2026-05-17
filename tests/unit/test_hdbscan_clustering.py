import pytest
import numpy as np
from unittest.mock import patch
from PyQt5.QtCore import QEventLoop

# We will import ClusterWorker from umap_panel when we implement it.
# For now, we assume it will exist in src.gui.panels.umap_panel
from src.gui.panels.umap_panel import ClusterWorker, HDBSCAN_AVAILABLE

@pytest.fixture
def synthetic_data_blobs():
    """Create 3 distinct blobs in 2D space."""
    np.random.seed(42)
    blob1 = np.random.normal(loc=[0, 0], scale=0.5, size=(100, 2))
    blob2 = np.random.normal(loc=[10, 10], scale=0.5, size=(100, 2))
    blob3 = np.random.normal(loc=[-10, 10], scale=0.5, size=(100, 2))
    return np.vstack([blob1, blob2, blob3])

@pytest.fixture
def synthetic_data_with_noise():
    """Create a dataset with dense areas and a few extreme outliers."""
    np.random.seed(42)
    blob = np.random.normal(loc=[0, 0], scale=0.5, size=(200, 2))
    noise = np.array([[100, 100], [-100, -100], [100, -100], [-100, 100]])
    return np.vstack([blob, noise])

@pytest.mark.skipif(not HDBSCAN_AVAILABLE, reason="hdbscan not installed")
def test_cluster_worker_hdbscan(synthetic_data_blobs, qtbot):
    """T1: ClusterWorker with HDBSCAN on distinct blobs."""
    worker = ClusterWorker(synthetic_data_blobs, method="HDBSCAN", param=15)
    
    with qtbot.waitSignal(worker.finished, timeout=2000) as blocker:
        worker.run()
        
    labels, method = blocker.args
    assert method == "HDBSCAN"
    assert len(labels) == 300
    
    # -1 is noise. We should have at least 2 distinct clusters.
    non_noise_labels = labels[labels != -1]
    unique_clusters = np.unique(non_noise_labels)
    assert len(unique_clusters) >= 2

def test_cluster_worker_kmeans(synthetic_data_blobs, qtbot):
    """T2: ClusterWorker with K-Means."""
    k = 5
    worker = ClusterWorker(synthetic_data_blobs, method="K-Means", param=k)
    
    with qtbot.waitSignal(worker.finished, timeout=2000) as blocker:
        worker.run()
        
    labels, method = blocker.args
    assert method == "K-Means"
    assert len(labels) == 300
    
    unique_clusters = np.unique(labels)
    assert -1 not in unique_clusters
    assert len(unique_clusters) <= k
    assert np.all((labels >= 0) & (labels < k))

@pytest.mark.skipif(not HDBSCAN_AVAILABLE, reason="hdbscan not installed")
def test_cluster_worker_hdbscan_noise(synthetic_data_with_noise, qtbot):
    """T3: HDBSCAN should label outliers as noise (-1)."""
    worker = ClusterWorker(synthetic_data_with_noise, method="HDBSCAN", param=15)
    
    with qtbot.waitSignal(worker.finished, timeout=2000) as blocker:
        worker.run()
        
    labels, method = blocker.args
    assert method == "HDBSCAN"
    
    # The last 4 points are extreme outliers, they should be noise (-1)
    assert np.all(labels[-4:] == -1)

@patch('src.gui.panels.umap_panel.HDBSCAN_AVAILABLE', False)
def test_cluster_worker_hdbscan_unavailable(synthetic_data_blobs, qtbot):
    """T4: If HDBSCAN is unavailable, emit error instead of crashing."""
    worker = ClusterWorker(synthetic_data_blobs, method="HDBSCAN", param=15)
    
    with qtbot.waitSignal(worker.error, timeout=1000) as blocker:
        worker.run()
        
    error_msg = blocker.args[0]
    assert "not available" in error_msg.lower() or "not installed" in error_msg.lower()
