import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.gui.panels.umap_panel import UMAPPanel, HDBSCAN_AVAILABLE

@pytest.fixture
def mock_main_window():
    class MockMainWindow:
        def __init__(self):
            self.data_manager = MagicMock()
            self.status_bar = MagicMock()
            self.tree_view = MagicMock()
            self.tree_model = MagicMock()
            
        def get_current_colors(self):
            return {
                'bg_window': '#FFFFFF',
                'bg_panel': '#F5F5F5',
                'text_primary': '#000000',
                'text_secondary': '#666666',
                'accent': '#007ACC',
                'accent_positive': '#28A745',
                'accent_negative': '#DC3545',
                'border': '#CCCCCC',
                'plot_highlight': '#FF0000',
            }
    return MockMainWindow()

@pytest.fixture
def initialized_umap_panel(mock_main_window, qtbot):
    panel = UMAPPanel(mock_main_window)
    qtbot.addWidget(panel)
    
    # Inject synthetic embedding and metadata
    np.random.seed(42)
    panel.embedding = np.random.normal(loc=0, scale=1, size=(100, 2))
    panel.cluster_ids = np.arange(100)
    panel.metadata_df = pd.DataFrame({'cluster_id': panel.cluster_ids})
    panel.is_3d = False
    
    return panel

@pytest.mark.skipif(not HDBSCAN_AVAILABLE, reason="hdbscan not installed")
def test_integration_hdbscan_clustering(initialized_umap_panel, qtbot):
    """T5: End-to-end HDBSCAN clustering populates metadata_df."""
    panel = initialized_umap_panel
    
    panel.cluster_method_combo.setCurrentText("HDBSCAN")
    panel.cluster_param_spin.setValue(15)
    
    panel.cluster_btn.click()
    qtbot.waitUntil(
        lambda: 'HDBSCAN' in panel.metadata_df.columns,
        timeout=5000,
    )

    assert 'HDBSCAN' in panel.metadata_df.columns
    labels = panel.metadata_df['HDBSCAN'].values
    assert len(labels) == 100
    assert panel.color_combo.currentText() == "HDBSCAN"

def test_integration_ui_method_swap(initialized_umap_panel, qtbot):
    """T6: Swap method combo and verify spinbox update."""
    panel = initialized_umap_panel
    
    # K-Means should be available
    panel.cluster_method_combo.setCurrentText("K-Means")
    assert panel.cluster_param_spin.prefix() == "k="
    assert panel.cluster_param_spin.minimum() == 2
    assert panel.cluster_param_spin.maximum() == 100
    assert panel.cluster_param_spin.value() == 5
    
    if HDBSCAN_AVAILABLE:
        panel.cluster_method_combo.setCurrentText("HDBSCAN")
        assert panel.cluster_param_spin.prefix() == ""
        assert panel.cluster_param_spin.minimum() == 2
        assert panel.cluster_param_spin.maximum() == 200
        assert panel.cluster_param_spin.value() == 15

def test_integration_kmeans_clustering(initialized_umap_panel, qtbot):
    """T7: K-Means pathway still functions."""
    panel = initialized_umap_panel
    
    panel.cluster_method_combo.setCurrentText("K-Means")
    panel.cluster_param_spin.setValue(4)
    panel.auto_group_chk.setChecked(True)
    
    # Mock auto grouping
    with patch('src.gui.callbacks.group_clusters_in_tree') as mock_group:
        panel.cluster_btn.click()
        qtbot.waitUntil(
            lambda: 'K-Means' in panel.metadata_df.columns,
            timeout=5000,
        )

        assert 'K-Means' in panel.metadata_df.columns
        assert len(np.unique(panel.metadata_df['K-Means'])) <= 4
        assert mock_group.called
