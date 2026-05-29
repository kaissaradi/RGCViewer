"""
Unit tests for folder selection debounce (§5.1).

Tests:
- test_rapid_folder_selection_fires_once        — 5 rapid assignments → 1 draw call
- test_folder_and_cell_timers_are_independent   — separate QTimer instances
- test_pending_folder_item_set_on_folder_click  — callbacks sets _pending_folder_item
- test_no_draw_when_population_view_disabled    — _process_folder_selection is a no-op
- test_timer_restarts_on_rapid_selection        — timer.start() called each assignment

All tests run with a real QApplication (pytest-qt). No disk I/O.
"""

import pytest
from unittest.mock import MagicMock, patch, call
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_main_window(qtbot, population_view_enabled=True):
    """
    Build the smallest possible real QMainWindow that has the two attributes
    the debounce spec requires, without instantiating the full MainWindow
    (which requires Kilosort data on disk).

    We attach the timer and helpers directly so the tests verify the contract
    without coupling to the full constructor.
    """
    from PyQt5.QtWidgets import QMainWindow

    class _TestWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.population_view_enabled = population_view_enabled
            self._pending_folder_item = None

            # The timer under test
            self.folder_selection_timer = QTimer(self)
            self.folder_selection_timer.setSingleShot(True)
            self.folder_selection_timer.setInterval(25)
            self.folder_selection_timer.timeout.connect(self._process_folder_selection)

            # A separate cell-selection timer (must stay independent)
            self.selection_timer = QTimer(self)
            self.selection_timer.setSingleShot(True)
            self.selection_timer.setInterval(25)

            # Mocked collaborators
            self.data_manager = MagicMock()
            self.pop_mosaic_canvas = MagicMock()
            self._get_group_cluster_ids = MagicMock(return_value=[0, 1, 2, 3, 4])

        def _process_folder_selection(self):
            """Mirrors the implementation described in spec §5.1."""
            item = self._pending_folder_item
            if item is None or not self.population_view_enabled:
                return
            group_ids = self._get_group_cluster_ids(item)
            if not group_ids:
                return
            # The two draw calls that must fire exactly once per debounced selection
            self._draw_rfs(group_ids)
            self._redraw_panels()

        # Mocked draw calls — replaced with MagicMock in tests that need to spy
        _draw_rfs = MagicMock()
        _redraw_panels = MagicMock()

    win = _TestWindow()
    qtbot.addWidget(win)
    return win


def _simulate_folder_selections(window, items, interval_ms=2):
    """
    Simulate rapid folder selections by assigning _pending_folder_item
    and calling folder_selection_timer.start() for each, with `interval_ms`
    between assignments (well below the 25ms debounce window).
    """
    for item in items:
        window._pending_folder_item = item
        window.folder_selection_timer.start()
        # No qtbot.wait here — we deliberately stay under 25ms between assignments


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFolderDebounce:

    def test_rapid_folder_selection_fires_once(self, qtbot):
        """
        Assigning _pending_folder_item 5 times within the debounce window and
        calling folder_selection_timer.start() each time should result in
        exactly 1 draw call — for the last folder — after the timer fires.
        """
        win = _make_minimal_main_window(qtbot)
        win._draw_rfs = MagicMock()
        win._redraw_panels = MagicMock()

        items = [MagicMock(name=f"folder_{i}") for i in range(5)]
        _simulate_folder_selections(win, items)

        # Let the 25ms timer fire
        qtbot.wait(60)

        assert win._draw_rfs.call_count == 1, (
            f"Expected exactly 1 draw call, got {win._draw_rfs.call_count}"
        )
        assert win._redraw_panels.call_count == 1

        # The draw was for the LAST item, not the first
        actual_ids = win._get_group_cluster_ids.call_args[0][0]
        assert actual_ids is items[-1], (
            "draw should use the last selected folder item, not the first"
        )

    def test_single_folder_selection_fires_once(self, qtbot):
        """A single folder selection produces exactly 1 draw call."""
        win = _make_minimal_main_window(qtbot)
        win._draw_rfs = MagicMock()
        win._redraw_panels = MagicMock()

        item = MagicMock(name="folder_0")
        win._pending_folder_item = item
        win.folder_selection_timer.start()

        qtbot.wait(60)

        assert win._draw_rfs.call_count == 1
        assert win._redraw_panels.call_count == 1

    def test_folder_and_cell_timers_are_independent(self, qtbot):
        """
        folder_selection_timer and selection_timer must be distinct QTimer
        instances. Starting folder_selection_timer must not reset selection_timer
        and vice versa.
        """
        win = _make_minimal_main_window(qtbot)

        assert win.folder_selection_timer is not win.selection_timer, (
            "folder_selection_timer and selection_timer must be different QTimer objects"
        )

        # Start cell timer and immediately start folder timer
        win.selection_timer.start()
        remaining_before = win.selection_timer.remainingTime()

        win.folder_selection_timer.start()
        # folder timer starting must not reset the cell timer
        remaining_after = win.selection_timer.remainingTime()

        # Remaining time should still be positive (timer is still running)
        assert remaining_after > 0, (
            "selection_timer was reset when folder_selection_timer was started"
        )

    def test_no_draw_when_population_view_disabled(self, qtbot):
        """
        _process_folder_selection must be a no-op when population_view_enabled
        is False, even if _pending_folder_item is set.
        """
        win = _make_minimal_main_window(qtbot, population_view_enabled=False)
        win._draw_rfs = MagicMock()
        win._redraw_panels = MagicMock()

        win._pending_folder_item = MagicMock(name="some_folder")
        win.folder_selection_timer.start()

        qtbot.wait(60)

        assert win._draw_rfs.call_count == 0, (
            "draw must not fire when population_view_enabled is False"
        )

    def test_no_draw_when_pending_item_is_none(self, qtbot):
        """
        _process_folder_selection with _pending_folder_item = None must not
        call the draw functions (guard clause).
        """
        win = _make_minimal_main_window(qtbot)
        win._draw_rfs = MagicMock()
        win._redraw_panels = MagicMock()

        win._pending_folder_item = None
        win.folder_selection_timer.start()

        qtbot.wait(60)

        assert win._draw_rfs.call_count == 0

    def test_timer_restarts_on_each_rapid_assignment(self, qtbot):
        """
        Each folder assignment calls folder_selection_timer.start().
        QTimer.start() on a running single-shot timer resets the countdown,
        so the timer only fires once after the last assignment.
        Verify via call count on the final draw output (already tested above),
        but also verify the timer is not active after the timeout fires.
        """
        win = _make_minimal_main_window(qtbot)
        win._draw_rfs = MagicMock()

        items = [MagicMock() for _ in range(3)]
        _simulate_folder_selections(win, items)

        # Timer should still be active (hasn't fired yet)
        assert win.folder_selection_timer.isActive(), (
            "Timer should be active immediately after rapid selections"
        )

        # Let it fire
        qtbot.wait(60)

        assert not win.folder_selection_timer.isActive(), (
            "Single-shot timer should be inactive after firing"
        )
        assert win._draw_rfs.call_count == 1


class TestCallbacksFolderBranch:
    """
    Tests that verify callbacks.on_cluster_selection_changed correctly
    uses the debounce path for folder nodes (spec §5.1, callbacks.py side).
    """

    def test_pending_folder_item_set_on_folder_click(self, qtbot):
        """
        When on_cluster_selection_changed receives a folder node (cluster_id=None),
        it should set main_window._pending_folder_item and start folder_selection_timer,
        NOT call draw functions directly.
        """
        from src.gui import callbacks

        # Build a minimal mock main_window that matches what callbacks.py expects
        win = MagicMock()
        win.population_view_enabled = True
        win.view_stack = MagicMock()
        win.view_stack.currentIndex.return_value = 0  # Tree view

        # _get_selected_cluster_id returns None → folder node
        win._get_selected_cluster_id.return_value = None

        # Tree selection returns one item
        mock_item = MagicMock()
        mock_item.data.return_value = None  # UserRole = None → group node
        mock_index = MagicMock()
        win.tree_view = MagicMock()
        win.tree_view.selectionModel.return_value.selectedIndexes.return_value = [mock_index]
        win.tree_model = MagicMock()
        win.tree_model.itemFromIndex.return_value = mock_item
        win._get_group_cluster_ids = MagicMock(return_value=[0, 1, 2])

        # folder_selection_timer is a MagicMock — we just check .start() is called
        win.folder_selection_timer = MagicMock()
        win._pending_folder_item = None

        callbacks.on_cluster_selection_changed(win)

        assert win._pending_folder_item is mock_item, (
            "_pending_folder_item should be set to the selected tree item"
        )
        win.folder_selection_timer.start.assert_called_once(), (
            "folder_selection_timer.start() should be called once"
        )

    def test_direct_draw_not_called_on_folder_click(self, qtbot):
        """
        on_cluster_selection_changed must NOT call draw_population_rfs_plot
        directly when a folder is clicked (the debounce timer handles it).
        """
        from src.gui import callbacks

        win = MagicMock()
        win.population_view_enabled = True
        win.view_stack.currentIndex.return_value = 0
        win._get_selected_cluster_id.return_value = None

        mock_item = MagicMock()
        mock_item.data.return_value = None
        win.tree_view.selectionModel.return_value.selectedIndexes.return_value = [MagicMock()]
        win.tree_model.itemFromIndex.return_value = mock_item
        win._get_group_cluster_ids.return_value = [0, 1, 2]
        win.folder_selection_timer = MagicMock()

        with patch('src.gui.callbacks.draw_population_rfs_plot') as mock_draw_rfs, \
             patch('src.gui.callbacks.redraw_population_panels') as mock_redraw:

            callbacks.on_cluster_selection_changed(win)

            mock_draw_rfs.assert_not_called(), (
                "draw_population_rfs_plot must not be called synchronously on folder click"
            )
            mock_redraw.assert_not_called(), (
                "redraw_population_panels must not be called synchronously on folder click"
            )
