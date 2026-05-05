import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
import pytest

matplotlib.use("Agg", force=True)


@pytest.fixture
def rgc_colors():
    return {
        "bg_base": "#111214",
        "bg_panel": "#18191C",
        "bg_surface": "#1E2025",
        "border_default": "#3D3F48",
        "border_subtle": "#2E3038",
        "text_primary": "#F0F0F2",
        "text_secondary": "#9B9DA6",
        "text_tertiary": "#5A5C65",
        "text_disabled": "#3A3C44",
        "accent": "#2E6DD4",
        "accent_hover": "#7EB8F7",
        "accent_positive": "#1A5C3A",
    }


@pytest.fixture
def make_main_window(qtbot, rgc_colors):
    from qtpy.QtWidgets import QMainWindow
    from unittest.mock import MagicMock

    class MockMainWindow(QMainWindow):
        def __init__(self, data_manager=None):
            super().__init__()
            self.data_manager = data_manager if data_manager is not None else MagicMock()
            self.status_bar = MagicMock()
            self.tree_view = None
            self.tree_model = None
            self.similarity_panel = None
            self._selected_cluster_id = None

        def get_current_colors(self):
            return dict(rgc_colors)

        def _get_selected_cluster_id(self):
            return self._selected_cluster_id

    def factory(data_manager=None):
        window = MockMainWindow(data_manager=data_manager)
        qtbot.addWidget(window)
        return window

    return factory
