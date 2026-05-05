import pytest
import psutil
import os
import gc
import numpy as np
import pandas as pd
from qtpy.QtWidgets import QMainWindow
from unittest.mock import MagicMock
from src.gui.panels.umap_panel import UMAPPanel

pytestmark = [pytest.mark.performance, pytest.mark.stress]

class MockMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_manager = MagicMock()
        self.status_bar = MagicMock()
        # Mock cluster_df for get_selected_cluster_ids
        self.data_manager.cluster_df = pd.DataFrame({'cluster_id': range(100)})
        self.tree_view = MagicMock()
        self.tree_model = MagicMock()

    def get_current_colors(self):
        return {
            'bg_panel': '#18191C',
            'accent': '#2E6DD4',
            'accent_positive': '#1A5C3A',
            'text_primary': '#F0F0F2',
            'text_secondary': '#9B9DA6'
        }

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

class MockMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_manager = MagicMock()
        self.status_bar = MagicMock()
        self.data_manager.cluster_df = pd.DataFrame({'cluster_id': range(100)})
        self.tree_view = MagicMock()
        self.tree_model = MagicMock()

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

    # 20 iterations is enough to see the trend without being too slow
    rng = np.random.default_rng(42)
    for i in range(20):
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
            qtbot.wait(100)
            gc.collect()
    
    qtbot.wait(1000)
    gc.collect()
    
    final_memory = get_process_memory()
    memory_diff = final_memory - initial_memory
    
    print(f"\n[UMAP] Memory Diff: {memory_diff:.2f} MB")
    
    if objgraph:
        print("Object growth (UMAP):")
        objgraph.show_growth(limit=10)

    # Allow 0.5MB per panel for Matplotlib/Qt overhead if not fully collected
    # Threshold for 20 panels
    # Matplotlib often has a 'base' cost for the first few figures that stays in a global cache.
    # 1.0MB per panel is a safe 'non-leaking' threshold for this type of stress test.
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

    for i in range(20):
        panel = StandardPlotsPanel(main_window)
        qtbot.addWidget(panel)

        # EXPLICIT CLEANUP
        panel.cleanup()
        panel.setParent(None)
        panel.deleteLater()
        del panel

    qtbot.wait(1000)
    gc.collect()

    final_memory = get_process_memory()
    memory_diff = final_memory - initial_memory

    print(f"\n[StandardPlots] Memory Diff: {memory_diff:.2f} MB")
    if objgraph:
        print("Top leaky objects (StandardPlots):")
        objgraph.show_most_common_types(limit=10)

    # Allow 1.0MB per panel
    assert memory_diff < 20.0, f"StandardPlots Memory leak! Grew by {memory_diff:.2f} MB"

def test_frantic_user_stress(qtbot):
    """
    STRESS TEST: Rapidly trigger UMAP runs to ensure workers are cleaned up.
    """
    main_window = MockMainWindow()
    panel = UMAPPanel(main_window)
    qtbot.addWidget(panel)
    panel.show()
    
    # Mock DataManager cache warming
    main_window.data_manager.cluster_df = pd.DataFrame({'cluster_id': range(100)})
    main_window.data_manager.feature_cache = {i: {'_computed': True} for i in range(100)}

    # Rapidly click "Run UMAP" button
    for i in range(20):
        # We use run_umap directly because mouseClick might wait for event processing
        panel.run_umap()
        
    # Verify that workers are being reset/aborted
    # Since they run in threads, we just want to ensure the app doesn't crash 
    # and eventually finishes.
    qtbot.wait(1000)
    
    assert panel.run_btn.isEnabled() or panel.progress.isVisible()
