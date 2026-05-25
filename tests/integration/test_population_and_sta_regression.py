import pytest
from qtpy.QtCore import QTimer

@pytest.mark.slow
def test_sta_panel_loads_after_vision_load(qtbot, main_window_with_vision_data):
    """High‑level: select a cell, STA panel should show RF image."""
    main_window = main_window_with_vision_data
    # Simulate tree selection of first cell
    first_cell_id = main_window.data_manager.cluster_df['cluster_id'].iloc[0]
    main_window._get_selected_cluster_id = lambda: first_cell_id
    main_window.analysis_tabs.setCurrentWidget(main_window.sta_panel)

    # Wait for any pending draws
    qtbot.wait(100)

    # Check that the RF canvas has an image item (pyqtgraph ImageItem)
    rv = main_window.sta_panel.rf_view
    assert len(rv.items) > 0   # should contain ImageItem and possibly Ellipse

def test_population_mosaic_updates_on_folder_click(qtbot, main_window_with_vision_data):
    """Select a folder (non‑leaf) and verify the mosaic canvas gets ellipses."""
    main_window = main_window_with_vision_data
    main_window.population_view_enabled = True
    # Get a folder item from the tree model
    root = main_window.tree_model.invisibleRootItem()
    folder_item = None
    for i in range(root.rowCount()):
        child = root.child(i)
        if child.data() is None:  # folder
            folder_item = child
            break
    assert folder_item is not None, "No folder in tree"
    # Simulate selection
    main_window._pending_folder_item = folder_item
    main_window.folder_selection_timer.start()
    qtbot.wait(50)   # enough for timer and draw
    # Check that the mosaic canvas has patches
    ax = main_window.pop_mosaic_canvas.fig.axes[0]
    # There should be at least one Ellipse patch
    ellipses = [p for p in ax.patches if isinstance(p, Ellipse)]
    assert len(ellipses) > 0