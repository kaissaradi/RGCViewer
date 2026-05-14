import pytest
import psutil
import os
import gc
import numpy as np
import pandas as pd
from qtpy.QtWidgets import QMainWindow
from unittest.mock import MagicMock
from src.gui.panels.umap_panel import UMAPPanel
from src.gui.panels.standard_plots_panel import StandardPlotsPanel

umap = pytest.importorskip("umap")

pytestmark = [pytest.mark.performance, pytest.mark.stress]

class MockMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_manager = MagicMock()
        self.status_bar = MagicMock()
        # Mock cluster_df for get_selected_cluster_ids
        # Make sure cluster_df has at least 100 clusters
        self.data_manager.cluster_df = pd.DataFrame({'cluster_id': range(100)})
        self.tree_view = MagicMock()
        self.tree_model = MagicMock()
        # CRITICAL: Ensure feature_cache has all clusters marked as computed
        # This prevents QMessageBox warning from showing
        self.data_manager.feature_cache = {i: {'_computed': True} for i in range(100)}

    def get_current_colors(self):
        return {
            'bg_panel': '#18191C',
            'accent': '#2E6DD4',
            'accent_positive': '#1A5C3A',
            'text_primary': '#F0F0F2',
            'text_secondary': '#9B9DA6',
            'border_default': '#3D3F48',
            'border_subtle': '#2E3038',
            'text_tertiary': '#5A5C65'
        }

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB

def test_umap_panel_memory_leak(qtbot):
    """
    STRESS TEST: Rapidly create and destroy UMAP panels to check for memory leaks.
    """
    try:
        import objgraph
    except ImportError:
        objgraph = None

    main_window = MockMainWindow()
    
    gc.collect()
    initial_memory = get_process_memory()
    
    if objgraph:
        objgraph.growth() # Baseline

    # 10 iterations is enough to see the trend without being too slow
    rng = np.random.default_rng(42)
    for i in range(10):
        panel = UMAPPanel(main_window)
        qtbot.addWidget(panel)
        
        panel.embedding = rng.random((1000, 2))
        panel.cluster_ids = np.arange(1000)
        panel.metadata_df = pd.DataFrame({'cluster_id': panel.cluster_ids})
        panel.update_plot()
        
        panel.cleanup()
        panel.setParent(None)
        panel.deleteLater()
        del panel
        
        # Force a small wait and GC every 5 panels
        if i % 5 == 0:
            qtbot.wait(50)
            gc.collect()
    
    qtbot.wait(500)
    gc.collect()
    
    final_memory = get_process_memory()
    memory_diff = final_memory - initial_memory
    
    print(f"\n[UMAP] Memory Diff: {memory_diff:.2f} MB")
    
    if objgraph:
        print("Object growth (UMAP):")
        objgraph.show_growth(limit=10)

    # Threshold for 10 panels
    assert memory_diff < 20.0, f"UMAP Memory leak! Grew by {memory_diff:.2f} MB"

def test_standard_plots_memory_leak(qtbot):
    """
    STRESS TEST: Rapidly create and destroy StandardPlots panels to check for memory leaks.
    """
    try:
        import objgraph
    except ImportError:
        objgraph = None

    main_window = MockMainWindow()

    gc.collect()
    initial_memory = get_process_memory()

    for i in range(10):
        panel = StandardPlotsPanel(main_window)
        qtbot.addWidget(panel)

        panel.cleanup()
        panel.setParent(None)
        panel.deleteLater()
        del panel

    qtbot.wait(500)
    gc.collect()

    final_memory = get_process_memory()
    memory_diff = final_memory - initial_memory

    print(f"\n[StandardPlots] Memory Diff: {memory_diff:.2f} MB")
    if objgraph:
        print("Top leaky objects (StandardPlots):")
        objgraph.show_most_common_types(limit=10)

    assert memory_diff < 20.0, f"StandardPlots Memory leak! Grew by {memory_diff:.2f} MB"

def test_frantic_user_stress(qtbot, mocker):
    """
    STRESS TEST: Rapidly trigger UMAP runs to ensure workers are handled correctly.
    Tests that rapid successive calls don't cause hangs or race conditions.
    """
    # Mock umap.UMAP.fit_transform to avoid heavy computation in stress test
    mocker.patch("umap.UMAP.fit_transform", return_value=np.random.rand(100, 2))
    
    # Mock feature extraction to return valid test data
    def mock_extract_features(dm, cluster_ids):
        # Return valid 2D feature matrix for testing
        n_samples = min(len(cluster_ids), 100) if cluster_ids is not None else 100
        features = np.random.rand(n_samples, 10)
        ids = list(range(n_samples))
        metadata = pd.DataFrame({
            'Time to Peak': np.random.rand(n_samples),
            'RF Area': np.random.rand(n_samples),
            'Ellipticity': np.random.rand(n_samples)
        })
        return features, ids, metadata
    
    mocker.patch("src.gui.panels.umap_panel.extract_features_from_datamanager", 
                 side_effect=mock_extract_features)
    
    # Mock any message boxes that might block the test
    mocker.patch("qtpy.QtWidgets.QMessageBox.warning")
    mocker.patch("qtpy.QtWidgets.QMessageBox.critical")
    
    main_window = MockMainWindow()
    panel = UMAPPanel(main_window)
    qtbot.addWidget(panel)
    panel.show()
    
    # Rapidly click "Run UMAP" button
    # Each call should properly clean up the previous worker
    for i in range(5):
        panel.run_umap()
        # Let Qt process events to allow worker initialization and cleanup
        qtbot.wait(50)
        
    # Wait for all pending operations to complete
    qtbot.wait(1000)
    
    # Verify that workers were properly cleaned up (should not hang)
    assert panel.worker is None or panel.worker_thread is None, "Workers not properly cleaned up"
    
    # Buttons should be enabled (run finished) or disabled (still running, but not hung)
    # After our fixes, buttons should be enabled since we clean up properly
    assert panel.run_btn.isEnabled() or not panel.progress.isVisible()
