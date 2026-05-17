import os
import pytest

from src.gui.main_window import MainWindow

# The path requested by user for this test
TEST_DATA_PATH = "/home/kais/Documents/Development/data/sorted/20251015A/chunk20/kilosort4"

@pytest.fixture
def main_window(qtbot):
    """Fixture to provide a clean MainWindow instance."""
    window = MainWindow()
    qtbot.addWidget(window)
    return window

@pytest.mark.skipif(not os.path.exists(TEST_DATA_PATH), reason=f"Real data not found at {TEST_DATA_PATH}")
def test_population_rf_ids_toggle(main_window, qtbot):
    """
    Integration test using real data to ensure the 'Show IDs' checkbox
    properly invalidates the cache and updates the UI.
    """
    # 1. Load the data using main_window's built-in flow
    main_window.load_directory(kilosort_dir=TEST_DATA_PATH)
    
    # Wait until DataManager is populated and UI tree is built
    qtbot.waitUntil(lambda: main_window.data_manager is not None and not main_window.data_manager.cluster_df.empty, timeout=20000)

    # 2. Open the population view
    main_window.pop_view_btn.setChecked(True)
    
    # Optional: ensure we have an output directory for screenshots
    os.makedirs("tests/test_outputs", exist_ok=True)
    
    # Give the UI a moment to process the split view toggle
    qtbot.wait(100)
    
    # Select the first cluster just to populate something
    first_cluster = main_window.data_manager.cluster_df.iloc[0]['cluster_id']
    main_window._pending_cluster_id = first_cluster
    
    # Mock vision_params so we can draw the plot without actual vision data loaded
    from unittest.mock import MagicMock
    mock_stafit = MagicMock()
    mock_stafit.center_x = 50
    mock_stafit.center_y = 50
    mock_stafit.std_x = 5
    mock_stafit.std_y = 5
    mock_stafit.rot = 0
    main_window.data_manager.vision_params = MagicMock()
    main_window.data_manager.vision_params.get_cell_ids.return_value = [first_cluster + 1] # Vision IDs are 1-indexed
    main_window.data_manager.vision_params.get_stafit_for_cell.return_value = mock_stafit
    main_window.data_manager.vision_sta_height = 100
    
    # Manually trigger drawing to bypass FeatureWorker timing in test
    from src.gui.panels.population_panel import draw_population_rfs_plot
    draw_population_rfs_plot(
        main_window=main_window,
        selected_cell_id=first_cluster,
        canvas=main_window.pop_mosaic_canvas
    )
    
    # Wait for axes to be created
    qtbot.waitUntil(lambda: len(main_window.pop_mosaic_canvas.figure.axes) > 0, timeout=5000)
    qtbot.wait(500) # Give it time to render

    # 3. Test IDs Checkbox Unchecked (Default)
    assert not main_window.pop_show_ids_checkbox.isChecked()
    
    # Capture the figure, verify no text
    canvas = main_window.pop_mosaic_canvas
    ax = canvas.figure.axes[0]
    
    # text instances that are IDs (not the title)
    id_texts_before = [t for t in ax.texts if t.get_text() != ax.get_title()]
    assert len(id_texts_before) == 0, "Expected no ID texts when checkbox is unchecked"
    
    # Save a screenshot
    canvas.figure.savefig("tests/test_outputs/population_mosaic_no_ids.png")

    # 4. Toggle the checkbox
    main_window.pop_show_ids_checkbox.setChecked(True)
    qtbot.wait(500) # Give the _on_pop_show_ids_toggled time to execute and redraw
    
    # Capture the figure, verify text exists
    ax = main_window.pop_mosaic_canvas.figure.axes[0]
    id_texts_after = [t for t in ax.texts if t.get_text() != ax.get_title()]
    assert len(id_texts_after) > 0, "Expected ID texts to appear after checkbox is checked"
    
    # Save a screenshot
    canvas.figure.savefig("tests/test_outputs/population_mosaic_with_ids.png")
