import numpy as np
import pandas as pd
import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QStandardItemModel
from qtpy.QtWidgets import QMainWindow, QTreeView
from unittest.mock import MagicMock

from src.gui.callbacks import group_clusters_in_tree, populate_tree_view


class MockMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_manager = MagicMock()
        self.tree_model = QStandardItemModel()
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.tree_model)
        self.status_bar = MagicMock()

    def setup_tree_model(self, model):
        self.tree_model = model
        self.tree_view.setModel(model)

    def setup_table_model(self, model):
        self.main_cluster_model = model


@pytest.fixture
def main_window(qtbot):
    window = MockMainWindow()
    qtbot.addWidget(window)
    return window


def _cluster_df(include_label=True):
    data = {
        "cluster_id": np.array([1, 2, 3, 4]),
        "n_spikes": np.array([10, 20, 30, 40]),
    }
    if include_label:
        data["KSLabel"] = ["good", "good", "mua", "noise"]
    return pd.DataFrame(data)


def test_populate_tree_view_groups_clusters_without_mutating_input(main_window):
    df = _cluster_df(include_label=False)
    main_window.data_manager.cluster_df = df

    populate_tree_view(main_window)

    assert "KSLabel" not in df.columns
    assert main_window.tree_model.rowCount() == 1
    group = main_window.tree_model.item(0)
    assert group.text() == "Unknown"
    assert group.rowCount() == 4
    assert group.child(0).data(Qt.ItemDataRole.UserRole) == 1


def test_group_clusters_in_tree_moves_items_and_updates_dataframe(main_window):
    df = _cluster_df(include_label=True)
    main_window.data_manager.cluster_df = df
    populate_tree_view(main_window, df=df)

    group_clusters_in_tree(main_window, [1, 3], "Reviewed")

    matches = main_window.tree_model.match(
        main_window.tree_model.index(0, 0),
        Qt.DisplayRole,
        "Reviewed",
        1,
        Qt.MatchExactly | Qt.MatchRecursive,
    )
    assert matches
    reviewed = main_window.tree_model.itemFromIndex(matches[0])
    moved_ids = [
        reviewed.child(row).data(Qt.ItemDataRole.UserRole)
        for row in range(reviewed.rowCount())
    ]
    assert sorted(moved_ids) == [1, 3]
    assert set(df.loc[df["KSLabel"] == "Reviewed", "cluster_id"]) == {1, 3}
