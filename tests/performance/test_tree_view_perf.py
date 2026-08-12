import pytest
import pandas as pd
import numpy as np
from qtpy.QtWidgets import QMainWindow, QTreeView
from qtpy.QtGui import QStandardItemModel
from src.gui.callbacks import populate_tree_view
from unittest.mock import MagicMock

pytestmark = pytest.mark.performance

class MockMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_manager = MagicMock()
        self.tree_model = QStandardItemModel()
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.tree_model)
        self.status_bar = MagicMock()
        
    def get_current_colors(self):
        return {'bg_panel': '#000000', 'text_primary': '#ffffff'}
    
    def setup_tree_model(self, model):
        self.tree_view.setModel(model)
        
    def setup_table_model(self, model):
        pass

@pytest.fixture
def main_window(qtbot):
    window = MockMainWindow()
    qtbot.addWidget(window)
    return window

def test_tree_view_population_performance(benchmark, main_window):
    """
    Benchmark the time it takes to populate the tree view with 1000 items.
    """
    # Create dummy data
    n_clusters = 1000
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        'cluster_id': np.arange(n_clusters),
        'KSLabel': rng.choice(['good', 'mua', 'noise'], n_clusters),
        'n_spikes': rng.integers(100, 10000, n_clusters)
    })
    
    # Measure the populate_tree_view function
    benchmark(populate_tree_view, main_window, df=df)
    
    # Assert that we actually populated the tree
    assert main_window.tree_model.rowCount() > 0

def test_tree_view_population_performance_large(benchmark, main_window):
    """
    Benchmark the time it takes to populate the tree view with 5000 items.
    """
    # Create dummy data
    n_clusters = 5000
    rng = np.random.default_rng(43)
    df = pd.DataFrame({
        'cluster_id': np.arange(n_clusters),
        'KSLabel': rng.choice(['good', 'mua', 'noise'], n_clusters),
        'n_spikes': rng.integers(100, 10000, n_clusters)
    })
    
    benchmark(populate_tree_view, main_window, df=df)
    assert main_window.tree_model.rowCount() > 0
