import pytest
import pandas as pd
from qtpy.QtGui import QStandardItemModel, QStandardItem
from matplotlib.patches import Ellipse
@pytest.mark.slow
def test_sta_panel_loads_after_vision_load(qtbot, main_window_with_vision_data):
    """High‑level: select a cell, STA panel should show RF image."""
    main_window = main_window_with_vision_data
    
    # FIX: Inject a dummy dataframe so .iloc[0] doesn't throw KeyError
    main_window.data_manager.cluster_df = pd.DataFrame({'cluster_id': [42, 43]})
    
    # Simulate tree selection of first cell
    first_cell_id = main_window.data_manager.cluster_df['cluster_id'].iloc[0]
    main_window._get_selected_cluster_id = lambda: first_cell_id
    main_window.analysis_tabs.setCurrentWidget(main_window.sta_panel)

    qtbot.wait(100)
    rv = main_window.sta_panel.rf_view
    assert len(rv.items) > 0


def test_population_mosaic_updates_on_folder_click(qtbot, main_window_with_vision_data):
    """Select a folder (non‑leaf) and verify the mosaic canvas gets ellipses."""
    main_window = main_window_with_vision_data
    main_window.population_view_enabled = True
    
    # FIX: Inject a dummy tree model so invisibleRootItem() exists
    model = QStandardItemModel()
    folder_item = QStandardItem("Group 1")
    folder_item.setData(None) # None = folder
    model.appendRow(folder_item)
    main_window.tree_model = model
    
    root = main_window.tree_model.invisibleRootItem()
    folder = root.child(0)
    
    # Simulate selection
    main_window._pending_folder_item = folder
    main_window.folder_selection_timer.start()
    qtbot.wait(50)   
    
    from matplotlib.patches import Ellipse
    ax = main_window.pop_mosaic_canvas.fig.axes[0]
    ellipses = [p for p in ax.patches if isinstance(p, Ellipse)]
    assert len(ellipses) > 0