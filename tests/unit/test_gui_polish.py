"""Unit tests for GUI polish: UMAP toolbar layout and sidebar collapse."""

import os
import pytest
from qtpy.QtCore import Qt, QEvent, QRect
from qtpy.QtGui import QStandardItem, QMouseEvent

from src.gui.main_window import MainWindow, SIDEBAR_COLLAPSED_WIDTH
from src.gui.widgets.widgets import ClusterTreeDelegate


@pytest.fixture
def main_window_fixture(qtbot):
    """Create and show a real MainWindow for widget‑level tests."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    return window


class TestUmapLayoutFix:
    def test_umap_layout_on_first_visit(self, qtbot, main_window_fixture):
        """First UMAP visit must not overlap toolbar rows."""
        win = main_window_fixture
        win.analysis_tabs.setCurrentIndex(3)          # UMAP tab
        qtbot.waitUntil(lambda: win.umap_panel.run_btn.isVisible(), timeout=2000)

        panel = win.umap_panel
        panel._refresh_layout()
        qtbot.wait(50)   # brief yield for geometry to settle

        row1_bottom = panel.run_btn.geometry().bottom()
        row2_top = panel.cluster_btn.geometry().top()
        assert row2_top >= row1_bottom, (
            f"Row overlap on first visit: row1 bottom={row1_bottom}, "
            f"row2 top={row2_top}"
        )

    def test_umap_layout_identical_after_tab_switch(self, qtbot, main_window_fixture):
        """
        Geometry must remain stable after switching tabs.
        Small y‑offset differences (≤1 pixel) are tolerated due to offscreen rendering.
        """
        win = main_window_fixture
        win.analysis_tabs.setCurrentIndex(3)
        qtbot.waitUntil(lambda: win.umap_panel.run_btn.isVisible(), timeout=2000)
        geo_before = win.umap_panel.run_btn.geometry()

        win.analysis_tabs.setCurrentIndex(2)   # switch away
        win.analysis_tabs.setCurrentIndex(3)   # back to UMAP
        qtbot.waitUntil(lambda: win.umap_panel.run_btn.isVisible(), timeout=2000)
        geo_after = win.umap_panel.run_btn.geometry()

        # Compare x, width, height exactly, but allow y to differ by at most 1 pixel
        assert geo_before.x() == geo_after.x()
        assert geo_before.width() == geo_after.width()
        assert geo_before.height() == geo_after.height()
        assert abs(geo_before.y() - geo_after.y()) <= 1


class TestSidebarCollapse:
    @pytest.mark.skipif(
        os.environ.get("QT_QPA_PLATFORM") == "offscreen",
        reason="Mouse clicks on buttons are not reliably processed in offscreen mode"
    )
    def test_sidebar_collapse_and_expand(self, qtbot, main_window_fixture):
        """Toggle the sidebar using a simulated click (skip in headless)."""
        win = main_window_fixture
        initial_left = win.main_splitter.sizes()[0]
        assert initial_left > SIDEBAR_COLLAPSED_WIDTH
        assert not win.sidebar_collapsed

        # Collapse
        qtbot.mouseClick(win.sidebar_toggle_btn, Qt.LeftButton)
        qtbot.waitUntil(lambda: win.sidebar_collapsed, timeout=2000)  # increased timeout

        assert win.sidebar_collapsed
        assert win.sidebar_toggle_btn.text() == "+"
        assert not win.left_content.isVisible()
        qtbot.waitUntil(
            lambda: win.main_splitter.sizes()[0] <= SIDEBAR_COLLAPSED_WIDTH + 5,
            timeout=2000,
        )

        # Expand
        qtbot.mouseClick(win.sidebar_toggle_btn, Qt.LeftButton)
        qtbot.waitUntil(lambda: not win.sidebar_collapsed, timeout=2000)

        assert win.sidebar_toggle_btn.text() == "\u2212"
        assert win.left_content.isVisible()
        qtbot.waitUntil(
            lambda: win.main_splitter.sizes()[0] > SIDEBAR_COLLAPSED_WIDTH + 50,
            timeout=2000,
        )

    def test_sidebar_toggle_button_present(self, main_window_fixture):
        """Basic presence test (always runs)."""
        win = main_window_fixture
        assert hasattr(win, "sidebar_toggle_btn")
        assert win.sidebar_toggle_btn.isVisible()
        assert win.sidebar_toggle_btn.text() == "\u2212"


class TestTreeBranchStyling:
    def test_tree_uses_cluster_delegate(self, main_window_fixture):
        win = main_window_fixture
        assert isinstance(win.tree_view.itemDelegate(), ClusterTreeDelegate)

    @pytest.mark.skipif(
        os.environ.get("QT_QPA_PLATFORM") == "offscreen",
        reason="Tree expansion not detectable in offscreen/headless environments"
    )
    def test_delegate_toggle_expands_folder(self, qtbot, main_window_fixture):
        """
        Verify the delegate's editorEvent handles a click on the toggle rect.
        This test is skipped in headless mode because QTreeView.setExpanded()
        has no visual effect offscreen.
        """
        win = main_window_fixture
        model = win.tree_model

        folder = QStandardItem("TestGroup")
        folder.appendRow(QStandardItem("Cluster 1 (n=10)"))
        model.appendRow(folder)
        win._switch_left_view(0)
        qtbot.wait(50)

        index = model.indexFromItem(folder)
        delegate = win.tree_view.itemDelegate()

        # Get the toggle rectangle
        row_rect = win.tree_view.visualRect(index)
        if not row_rect.isValid() or row_rect.isEmpty():
            row_rect = QRect(0, 0, 300, 28)

        toggle = delegate._toggle_rect(index, row_rect)
        assert toggle is not None, "Folder item must produce a toggle rect"

        # Verify toggle is positioned within a plausible x range
        indent = win.tree_view.indentation()
        assert 0 <= toggle.left() < indent * 3, (
            f"Toggle x={toggle.left()} is outside expected indent range"
        )

        # Simulate a mouse release on the toggle
        release = QMouseEvent(
            QEvent.MouseButtonRelease,
            toggle.center(),
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
        result = delegate.editorEvent(release, model, None, index)
        assert result, "editorEvent must return True when a mouse release hits the toggle rect"

    def test_delegate_toggle_miss_returns_false(self, qtbot, main_window_fixture):
        """A click far from the toggle rect must not be handled by the delegate."""
        win = main_window_fixture
        model = win.tree_model

        folder = QStandardItem("MissGroup")
        folder.appendRow(QStandardItem("Cluster 2 (n=5)"))
        model.appendRow(folder)
        win._switch_left_view(0)
        qtbot.wait(50)

        index = model.indexFromItem(folder)
        delegate = win.tree_view.itemDelegate()

        # Click at far right — nowhere near the toggle
        miss = QMouseEvent(
            QEvent.MouseButtonRelease,
            QRect(290, 14, 1, 1).center(),
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
        result = delegate.editorEvent(miss, model, None, index)
        assert not result, "editorEvent must return False for a miss"

    def test_delegate_colors_follow_theme(self, main_window_fixture):
        win = main_window_fixture
        delegate = win._cluster_tree_delegate
        dark = delegate._colors["text_secondary"]
        win.toggle_theme()
        assert delegate._colors["text_secondary"] != dark
        win.toggle_theme()