"""Unit tests for the tree sidebar ID / Spikes / Ch row factory."""

from qtpy.QtCore import Qt
from qtpy.QtGui import QStandardItem

from src.gui.widgets.widgets import (
    TREE_COL_CH,
    TREE_COL_SPIKES,
    TREE_HEADERS,
    TREE_SORT_ROLE,
    TreeStandardItem,
    count_tree_leaves,
    format_channel,
    format_spike_count,
    make_cell_row,
    make_group_row,
    refresh_tree_group_counts,
    set_channel_item,
)


def test_headers_are_id_spikes_ch():
    assert TREE_HEADERS == ("ID", "Spikes", "Ch")


def test_cell_row_id_spikes_channel():
    row = make_cell_row(12, 1840, 182)
    assert len(row) == 3
    assert row[0].text() == "12"
    assert row[0].data(Qt.ItemDataRole.UserRole) == 12
    assert row[TREE_COL_SPIKES].text() == "1840"
    assert row[TREE_COL_CH].text() == "182"
    assert row[0].isEditable() is False
    assert row[0].isDropEnabled() is False
    assert row[0].icon().isNull()


def test_missing_channel_is_em_dash():
    assert format_channel(None) == "—"
    assert format_channel(-1) == "—"
    assert format_channel(float("nan")) == "—"
    row = make_cell_row(3, 10, None)
    assert row[TREE_COL_CH].text() == "—"
    assert row[TREE_COL_CH].data(TREE_SORT_ROLE) is None
    row_neg = make_cell_row(4, 10, -1)
    assert row_neg[TREE_COL_CH].text() == "—"


def test_missing_spikes_is_em_dash():
    assert format_spike_count(None) == "—"
    row = make_cell_row(1, None, 5)
    assert row[TREE_COL_SPIKES].text() == "—"


def test_spike_sort_is_numeric():
    low = make_cell_row(1, 999, 1)[TREE_COL_SPIKES]
    high = make_cell_row(2, 1000, 1)[TREE_COL_SPIKES]
    assert low.data(TREE_SORT_ROLE) < high.data(TREE_SORT_ROLE)
    assert low < high
    # Display strings would sort the other way; the role must win.
    assert "1000" < "999"


def test_group_row_has_no_cluster_id_and_no_icon():
    row = make_group_row("good", child_count=42)
    assert row[0].data(Qt.ItemDataRole.UserRole) is None
    assert row[0].text() == "good"
    assert row[0].font().bold()
    assert row[0].isDropEnabled()
    assert row[0].icon().isNull()
    assert row[TREE_COL_SPIKES].text() == "42"
    assert row[TREE_COL_CH].text() == ""
    empty = make_group_row("mua", child_count=0)
    assert empty[TREE_COL_SPIKES].text() == ""


def test_group_items_are_not_hex_painted():
    row = make_group_row("good")
    # No hard-coded fill — theme apply owns group chrome.
    assert row[0].background().style() == Qt.BrushStyle.NoBrush


def test_refresh_group_counts_walks_nested_folders():
    root = QStandardItem("root")
    outer = make_group_row("ON", 0)
    inner = make_group_row("parasol", 0)
    inner[0].appendRow(make_cell_row(1, 10, 2))
    inner[0].appendRow(make_cell_row(2, 20, 3))
    outer[0].appendRow(inner)
    outer[0].appendRow(make_cell_row(3, 5, 4))
    root.appendRow(outer)

    refresh_tree_group_counts(root)
    assert count_tree_leaves(outer[0]) == 3
    assert outer[TREE_COL_SPIKES].text() == "3"
    assert inner[TREE_COL_SPIKES].text() == "2"


def test_set_channel_item_updates_sort_role():
    row = make_cell_row(1, 10, None)
    set_channel_item(row[TREE_COL_CH], 17)
    assert row[TREE_COL_CH].text() == "17"
    assert row[TREE_COL_CH].data(TREE_SORT_ROLE) == 17


def test_tree_standard_item_names_sort_before_ids():
    folder = TreeStandardItem("good")
    cell = TreeStandardItem("12")
    cell.setData(12, TREE_SORT_ROLE)
    assert folder < cell
