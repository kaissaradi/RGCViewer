import pytest
import numpy as np
import pandas as pd
from qtpy.QtWidgets import QMainWindow
from unittest.mock import MagicMock
from src.gui.panels.umap_panel import UMAPPanel
from matplotlib.widgets import LassoSelector, RectangleSelector
from qtpy.QtWidgets import QMessageBox

class MockMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_manager = MagicMock()
        # Mock status bar
        self.status_bar = MagicMock()
        # Mock colors
        self.colors = {
            'bg_panel': '#18191C',
            'accent': '#2E6DD4',
            'accent_positive': '#1A5C3A',
            'text_primary': '#F0F0F2',
            'text_secondary': '#9B9DA6'
        }
        self.tree_view = None
        self.tree_model = None

    def get_current_colors(self):
        return self.colors

@pytest.fixture
def umap_panel(qtbot):
    main_window = MockMainWindow()
    panel = UMAPPanel(main_window)
    qtbot.addWidget(panel)
    return panel

def _metadata(ids):
    return pd.DataFrame({'cluster_id': ids, 'KSLabel': ['Good'] * len(ids)})

def test_umap_selection_tool_persistence(umap_panel):
    """Verify that only ONE selection tool is active at a time."""
    embedding = np.column_stack((np.linspace(0, 1, 10), np.linspace(1, 0, 10)))
    ids = np.arange(10)
    metadata = _metadata(ids)
    
    umap_panel.on_processing_finished(embedding, ids, metadata)
    
    assert umap_panel.selector_combo.currentText() == "Lasso Tool"
    
    lasso_active = umap_panel.selector is not None and umap_panel.selector.active
    current_active = umap_panel.current_selector is not None and umap_panel.current_selector.active
    
    assert not (lasso_active and current_active), "Multiple selectors active simultaneously!"

def test_rectangle_selection_switch(umap_panel):
    """Verify that switching to Rectangle Tool works and disables Lasso."""
    embedding = np.column_stack((np.linspace(0, 1, 10), np.linspace(1, 0, 10)))
    ids = np.arange(10)
    metadata = _metadata(ids)
    umap_panel.on_processing_finished(embedding, ids, metadata)
    
    umap_panel.selector_combo.setCurrentText("Rectangle Tool")
    
    assert isinstance(umap_panel.current_selector, RectangleSelector)
    assert umap_panel.current_selector.active
    
    if umap_panel.selector is not None:
        assert not umap_panel.selector.active, "Old LassoSelector still active after switching to Rectangle!"

def test_on_select_rect_vertices(umap_panel, mocker):
    """Verify that Rectangle selection correctly passes 5 vertices (closed loop) to on_select."""
    mock_on_select = mocker.patch.object(umap_panel, 'on_select')
    
    # Mock event objects
    eclick = MagicMock()
    eclick.xdata, eclick.ydata = 0, 0
    erelease = MagicMock()
    erelease.xdata, erelease.ydata = 1, 1
    
    umap_panel.on_select_rect(eclick, erelease)
    
    # Should be called with 5 vertices forming a rectangle
    mock_on_select.assert_called_once()
    verts = mock_on_select.call_args[0][0]
    assert len(verts) == 5
    assert verts[0] == (0, 0)
    assert verts[2] == (1, 1)
    assert verts[4] == (0, 0) # Closed loop

def test_rectangle_selection_ignores_incomplete_mouse_events(umap_panel, mocker):
    """Rectangle selector can emit None data when released outside the axes."""
    mock_on_select = mocker.patch.object(umap_panel, 'on_select')

    eclick = MagicMock(xdata=0, ydata=None)
    erelease = MagicMock(xdata=1, ydata=1)

    umap_panel.on_select_rect(eclick, erelease)

    mock_on_select.assert_not_called()

def test_lasso_selection_groups_exact_points_inside_bounds(umap_panel, mocker):
    """The selection contract is the actual cluster IDs inside the drawn path."""
    ids = np.array([10, 11, 12, 13])
    umap_panel.embedding = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [2.0, 2.0],
        [0.25, 0.75],
    ])
    umap_panel.cluster_ids = ids
    umap_panel.metadata_df = _metadata(ids)
    umap_panel.is_3d = False

    mocker.patch.object(QMessageBox, 'question', return_value=QMessageBox.Yes)
    mock_create_group = mocker.patch.object(umap_panel, 'create_group')

    umap_panel.on_select([(-0.2, -0.2), (1.1, -0.2), (1.1, 1.1), (-0.2, 1.1)])

    mock_create_group.assert_called_once()
    selected_ids = mock_create_group.call_args.args[0]
    np.testing.assert_array_equal(selected_ids, np.array([10, 11, 13]))

def test_3d_selection_can_be_disabled_without_prompting(umap_panel, mocker):
    ids = np.array([1, 2, 3])
    umap_panel.embedding = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0],
    ])
    umap_panel.cluster_ids = ids
    umap_panel.metadata_df = _metadata(ids)
    umap_panel.is_3d = True
    umap_panel.project_3d_chk.setChecked(False)

    mock_question = mocker.patch.object(QMessageBox, 'question')

    umap_panel.on_select([(-1, -1), (2, -1), (2, 2), (-1, 2)])

    mock_question.assert_not_called()
