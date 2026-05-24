from unittest.mock import MagicMock, patch

from src.gui.main_window import MainWindow


class _ToggleHarness:
    toggle_population_split_view = MainWindow.toggle_population_split_view
    _draw_population_panel_initial = MainWindow._draw_population_panel_initial


def _make_toggle_harness():
    win = _ToggleHarness()
    win.pop_context_widget = MagicMock()
    win.right_splitter = MagicMock()
    win.right_splitter.sizes.return_value = [900, 0]
    win.pop_mosaic_canvas = MagicMock()
    win._get_selected_cluster_id = MagicMock(return_value=42)
    return win


def test_toggle_population_split_view_defers_initial_draw():
    win = _make_toggle_harness()

    with patch('src.gui.main_window.QTimer.singleShot') as single_shot, \
            patch('src.gui.main_window.draw_population_rfs_plot') as mock_draw_rfs, \
            patch('src.gui.main_window.callbacks.redraw_population_panels') as mock_redraw:
        win.toggle_population_split_view(True)

    assert win.population_view_enabled is True
    win.pop_context_widget.show.assert_called_once()
    win.right_splitter.setSizes.assert_called_once_with([675, 225])
    mock_draw_rfs.assert_not_called()
    mock_redraw.assert_not_called()

    single_shot.assert_called_once()
    delay_ms, callback = single_shot.call_args.args
    assert delay_ms == 0
    assert callback.__self__ is win
    assert callback.__func__ is _ToggleHarness._draw_population_panel_initial


def test_deferred_population_panel_initial_draw_uses_current_selection():
    win = _make_toggle_harness()

    with patch('src.gui.main_window.draw_population_rfs_plot') as mock_draw_rfs, \
            patch('src.gui.main_window.callbacks.redraw_population_panels') as mock_redraw:
        win._draw_population_panel_initial()

    win._get_selected_cluster_id.assert_called_once()
    mock_draw_rfs.assert_called_once_with(
        main_window=win,
        selected_cell_id=42,
        canvas=win.pop_mosaic_canvas,
    )
    mock_redraw.assert_called_once_with(win)
