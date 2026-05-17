"""Unit tests for GUI polish: UMAP toolbar layout and sidebar collapse."""

import pytest
from qtpy.QtCore import Qt

from src.gui.main_window import MainWindow, SIDEBAR_COLLAPSED_WIDTH


@pytest.fixture
def main_window_fixture(qtbot):
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
        win.analysis_tabs.setCurrentIndex(3)
        qtbot.wait(150)
        panel = win.umap_panel
        panel._refresh_layout()
        qtbot.wait(50)
        row1_bottom = panel.run_btn.geometry().bottom()
        row2_top = panel.cluster_btn.geometry().top()
        assert row2_top >= row1_bottom, (
            f"Row overlap on first visit: row1 bottom={row1_bottom}, "
            f"row2 top={row2_top}"
        )

    def test_umap_layout_identical_after_tab_switch(self, qtbot, main_window_fixture):
        """Geometry must not change between first and second UMAP visit."""
        win = main_window_fixture
        win.analysis_tabs.setCurrentIndex(3)
        qtbot.wait(100)
        geo_before = win.umap_panel.run_btn.geometry()
        win.analysis_tabs.setCurrentIndex(2)
        win.analysis_tabs.setCurrentIndex(3)
        qtbot.wait(100)
        geo_after = win.umap_panel.run_btn.geometry()
        assert geo_before == geo_after


class TestSidebarCollapse:
    def test_sidebar_toggle_button_present(self, main_window_fixture):
        win = main_window_fixture
        assert hasattr(win, "sidebar_toggle_btn")
        assert win.sidebar_toggle_btn.isVisible()
        assert win.sidebar_toggle_btn.text() == "\u2212"

    def test_sidebar_collapse_and_expand(self, qtbot, main_window_fixture):
        win = main_window_fixture
        initial_left = win.main_splitter.sizes()[0]
        assert initial_left > SIDEBAR_COLLAPSED_WIDTH
        assert not win.sidebar_collapsed

        qtbot.waitUntil(lambda: win.sidebar_toggle_btn.isVisible(), timeout=2000)
        win.sidebar_toggle_btn.clicked.emit()
        qtbot.wait(200)

        assert win.sidebar_collapsed
        assert win.sidebar_toggle_btn.text() == "+"
        assert not win.left_content.isVisible()
        qtbot.waitUntil(
            lambda: win.main_splitter.sizes()[0] <= SIDEBAR_COLLAPSED_WIDTH + 5,
            timeout=1000,
        )

        win.sidebar_toggle_btn.clicked.emit()
        qtbot.waitUntil(lambda: not win.sidebar_collapsed, timeout=500)

        assert win.sidebar_toggle_btn.text() == "\u2212"
        assert win.left_content.isVisible()
        qtbot.waitUntil(
            lambda: win.main_splitter.sizes()[0] > SIDEBAR_COLLAPSED_WIDTH + 50,
            timeout=1000,
        )
