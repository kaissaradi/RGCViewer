import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.gui.panels.umap_panel import UMAPPanel
from unittest.mock import MagicMock
from qtpy.QtWidgets import QMainWindow

pytestmark = pytest.mark.visual

class MockMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_manager = MagicMock()
        self.status_bar = MagicMock()

    def get_current_colors(self):
        return {
            'bg_panel': '#18191C',
            'accent': '#2E6DD4',
            'accent_positive': '#1A5C3A',
            'text_primary': '#F0F0F2',
            'text_secondary': '#9B9DA6'
        }

@pytest.mark.mpl_image_compare(baseline_dir='baselines', tolerance=5)
def test_umap_plot_visual(qtbot):
    """
    VISUAL REGRESSION: Generate a UMAP plot and compare it against the baseline.
    """
    main_window = MockMainWindow()
    panel = UMAPPanel(main_window)
    qtbot.addWidget(panel)
    
    # Deterministic "Data"
    rng = np.random.default_rng(42)
    panel.embedding = rng.standard_normal((100, 2))
    panel.cluster_ids = np.arange(100)
    panel.metadata_df = pd.DataFrame({
        'cluster_id': panel.cluster_ids,
        'KSLabel': ['Good'] * 50 + ['MUA'] * 50
    })
    
    panel.color_combo.setCurrentText("KSLabel")
    panel.update_plot()
    
    # Return the figure for pytest-mpl to capture
    return panel.fig
