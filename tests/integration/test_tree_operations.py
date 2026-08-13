import numpy as np
import pandas as pd
import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QStandardItemModel
from unittest.mock import MagicMock

from src.gui.callbacks import group_clusters_in_tree, populate_tree_view
from src.gui.main_window import MainWindow
from src.gui.widgets.widgets import find_cell_item


class _FakeTreeView:
    def __init__(self):
        self.expanded = []
        self.collapsed = []

    def setUpdatesEnabled(self, _enabled):
        pass

    def collapseAll(self):
        self.collapsed.append("all")

    def setModel(self, model):
        self._model = model

    def expand(self, index):
        self.expanded.append(index)

    def collapse(self, index):
        self.collapsed.append(index)


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
    assert main_window.tree_view.expanded  # default expand_new=True
    assert set(df.loc[df["KSLabel"] == "Reviewed", "cluster_id"]) == {1, 3}


def test_feature_extraction_group_stays_collapsed(main_window):
    df = _cluster_df(include_label=True)
    main_window.data_manager.cluster_df = df
    populate_tree_view(main_window, df=df)
    main_window.tree_view.expanded.clear()
    main_window.tree_view.collapsed.clear()

    group_clusters_in_tree(main_window, [1, 3], "Nc1", expand_new=False)

    matches = main_window.tree_model.match(
        main_window.tree_model.index(0, 0),
        Qt.DisplayRole,
        "Nc1",
        1,
        Qt.MatchExactly | Qt.MatchRecursive,
    )
    assert matches
    nc = main_window.tree_model.itemFromIndex(matches[0])
    assert nc.index() in main_window.tree_view.collapsed
    assert nc.index() not in main_window.tree_view.expanded


def test_pop_subset_uses_innermost_tree_folder_from_table_id(main_window):
    df = _cluster_df(include_label=True)
    main_window.data_manager.cluster_df = df
    populate_tree_view(main_window, df=df)
    group_clusters_in_tree(main_window, [1, 3], "Nc1", expand_new=False)

    item = find_cell_item(main_window.tree_model, np.int64(1))
    assert item is not None
    assert item.parent().text() == "Nc1"

    main_window.view_stack = MagicMock()
    main_window.view_stack.currentIndex.return_value = 1
    main_window._get_selected_cluster_id = lambda: np.int64(1)
    main_window._get_group_cluster_ids = (
        lambda item: MainWindow._get_group_cluster_ids(main_window, item)
    )
    subset = MainWindow._get_pop_subset_ids(main_window)
    assert sorted(int(c) for c in subset) == [1, 3]
