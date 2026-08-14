"""Rebuilding the cluster table must not move the user off their cell.

``refresh_table_model`` goes through ``_install_table_proxy``, which builds a
fresh ``HiddenIdFilterProxyModel`` and installs it on the view. Qt gives the
view a new ``QItemSelectionModel`` when the model changes, so the selection is
destroyed — the function preserved column order and sort across that swap but
never the selected row.

It rebuilds whenever a background column arrives: ``sta_snr`` from the Vision
load, ``chirp_qi``/``chirp_onoff`` from the stimulus load. On a local run both
land within ~0.4 s of opening, before anyone has clicked, so the loss was
invisible. Off the lab's CIFS mount they land ~4 s and ~8 s in, by which point
the user is working in the table — and it looked like the app randomly jumping
to a different cell and blanking the panels.
"""

import pandas as pd
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QTableView

from src.gui.main_window import MainWindow
from src.gui.widgets.widgets import HighlightStatusPandasModel


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


class _TableHost:
    """The slice of MainWindow the two selection helpers actually touch."""

    def __init__(self, df):
        self.table_view = QTableView()
        self.main_cluster_model = HighlightStatusPandasModel(df)
        self.table_view.setModel(self.main_cluster_model)

    # Bound straight off MainWindow so the test exercises the shipping code.
    _selected_table_cluster_id = MainWindow._selected_table_cluster_id
    _select_table_cluster_id = MainWindow._select_table_cluster_id

    def rebuild_with(self, df):
        """Stand in for refresh_table_model's model swap."""
        cid = self._selected_table_cluster_id()
        self.main_cluster_model = HighlightStatusPandasModel(df)
        self.table_view.setModel(self.main_cluster_model)
        if cid is not None:
            self._select_table_cluster_id(cid)
        return cid


def _df(with_extra_column=False):
    data = {"cluster_id": [10, 20, 30, 40], "n_spikes": [1, 2, 3, 4]}
    if with_extra_column:
        # What the Vision load adds seconds later on a slow mount.
        data["sta_snr"] = [0.5, 1.5, 2.5, 3.5]
    return pd.DataFrame(data)


def test_selection_survives_a_rebuild(qapp):
    host = _TableHost(_df())
    host.table_view.selectRow(2)
    assert host._selected_table_cluster_id() == 30

    host.rebuild_with(_df(with_extra_column=True))

    assert host._selected_table_cluster_id() == 30, "user was moved off their cell"


def test_selection_follows_the_id_not_the_row(qapp):
    """A reordered rebuild must reselect by cluster_id, not by row number."""
    host = _TableHost(_df())
    host.table_view.selectRow(0)
    assert host._selected_table_cluster_id() == 10

    reordered = _df().iloc[::-1].reset_index(drop=True)  # 40, 30, 20, 10
    host.rebuild_with(reordered)

    assert host._selected_table_cluster_id() == 10
    assert host.table_view.currentIndex().row() == 3


def test_no_selection_stays_no_selection(qapp):
    host = _TableHost(_df())
    assert host._selected_table_cluster_id() is None

    host.rebuild_with(_df(with_extra_column=True))

    assert host._selected_table_cluster_id() is None


def test_vanished_cluster_does_not_force_a_selection(qapp):
    """A cell filtered out or trashed must not drag the selection somewhere else."""
    host = _TableHost(_df())
    host.table_view.selectRow(1)
    assert host._selected_table_cluster_id() == 20

    survivors = _df()[_df()["cluster_id"] != 20].reset_index(drop=True)
    host.rebuild_with(survivors)

    assert host._selected_table_cluster_id() is None
