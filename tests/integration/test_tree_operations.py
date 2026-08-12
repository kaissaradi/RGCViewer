import numpy as np
import pandas as pd
import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QStandardItemModel
from unittest.mock import MagicMock

from src.gui.callbacks import group_clusters_in_tree, populate_tree_view


class _FakeTreeView:
    def setUpdatesEnabled(self, _enabled):
        pass

    def collapseAll(self):
        pass

    def setModel(self, model):
        self._model = model

    def expand(self, _index):
        pass


class MockMainWindow:
    def __init__(self):
        self.data_manager = MagicMock()
        self.tree_model = QStandardItemModel()
        self.tree_view = _FakeTreeView()
        self.status_bar = MagicMock()

    def setup_tree_model(self, model):
        self.tree_model = model
        self.tree_view.setModel(model)

    def setup_table_model(self, model):
        self.main_cluster_model = model


@pytest.fixture
def main_window():
    return MockMainWindow()


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
    assert main_window.tree_model.columnCount() == 3
    assert group.child(0).data(Qt.ItemDataRole.UserRole) == 1
    assert group.child(0).text() == "1"
    assert group.child(0, 1).text() == "10"
    assert group.child(0, 2).text() == "—"


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
    moved = reviewed.child(0)
    assert reviewed.child(0, 1) is not None
    assert moved.data(Qt.ItemDataRole.UserRole) in {1, 3}
    assert set(df.loc[df["KSLabel"] == "Reviewed", "cluster_id"]) == {1, 3}
