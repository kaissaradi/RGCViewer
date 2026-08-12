import pandas as pd
import pyqtgraph as pg
from qtpy.QtGui import QColor, QStandardItem
from qtpy.QtCore import Qt

from src.gui.main_window import MainWindow
from src.gui.widgets.widgets import HighlightStatusPandasModel


def test_main_window_theme_toggle_updates_stylesheet_and_pyqtgraph(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    window.toggle_theme()

    colors = window.get_current_colors()
    assert window.theme == "light"
    assert colors["text_primary"] in window.styleSheet()
    assert pg.getConfigOption("background") == colors["bg_panel"]
    assert pg.getConfigOption("foreground") == colors["text_secondary"]


def test_dead_good_filter_and_refine_controls_are_removed(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    assert not hasattr(window, "filter_good_btn")
    assert not hasattr(window, "refine_button")


def test_cluster_table_sort_reorders_dataframe(qtbot):
    model = HighlightStatusPandasModel(
        pd.DataFrame(
            {
                "cluster_id": [3, 1, 2],
                "status": ["Clean", "Noise", "Unsure"],
                "firing_rate_hz": [2.5, 9.0, 1.5],
            }
        )
    )

    model.sort(0, Qt.AscendingOrder)
    assert model._dataframe["cluster_id"].tolist() == [1, 2, 3]

    model.sort(2, Qt.DescendingOrder)
    assert model._dataframe["firing_rate_hz"].tolist() == [9.0, 2.5, 1.5]


def test_light_mode_tree_items_use_readable_primary_text(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    group = QStandardItem("good")
    subgroup = QStandardItem("Type_5")
    leaf = QStandardItem("Cluster 1 (n=4915)")
    leaf.setData(1, Qt.UserRole)
    subgroup.appendRow(leaf)
    group.appendRow(subgroup)
    window.tree_model.appendRow(group)
    window.setup_tree_model(window.tree_model)

    window.toggle_theme()
    expected = QColor(window.get_current_colors()["text_primary"]).name()

    assert group.foreground().color().name() == expected
    assert subgroup.foreground().color().name() == expected
    assert leaf.foreground().color().name() == expected


def test_light_mode_table_model_uses_primary_text_for_data_cells():
    model = HighlightStatusPandasModel(
        pd.DataFrame({"cluster_id": [1], "status": ["Clean"], "n_spikes": [4915]})
    )
    light_colors = {
        "text_primary": "#111214",
        "status_good_text": "#065F46",
        "status_mua_text": "#92400E",
        "status_noise_text": "#991B1B",
        "status_unsort_text": "#1E40AF",
    }

    model.update_colors(light_colors)
    regular_cell_color = model.data(model.index(0, 0), Qt.ForegroundRole)

    assert regular_cell_color.name() == QColor(light_colors["text_primary"]).name()
