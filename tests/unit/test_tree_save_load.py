"""The cluster tree survives File ▸ Save and reload, more than once.

save_tree_structure wrote item.data() (UserRole + 1). A tree that had been
loaded once has the id only in UserRole, so its next save wrote every cell
as null and the reload had no cells (Trash included).
"""

import json
from types import SimpleNamespace

import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtGui import QStandardItemModel
from qtpy.QtWidgets import QTreeView

from src.analysis.data_manager import DataManager
from src.gui.widgets.widgets import make_cell_row, make_group_row


def _dm(qapp):
    model = QStandardItemModel()
    view = QTreeView()
    view.setModel(model)
    win = SimpleNamespace(tree_model=model, tree_view=view,
                          setup_tree_model=lambda m: None)
    dm = DataManager.__new__(DataManager)
    dm.main_window = win
    dm.cluster_df = pd.DataFrame({"cluster_id": [0, 1, 2, 3],
                                  "n_spikes": [10, 20, 30, 40],
                                  "best_chan": [5, 6, 7, 8]})
    return dm, model


def _groups(model):
    out = {}

    def walk(item, path):
        for i in range(item.rowCount()):
            c = item.child(i)
            cid = c.data(Qt.ItemDataRole.UserRole)
            if cid is None:
                walk(c, path + [c.text()])
            else:
                out[cid] = "/".join(path)

    walk(model.invisibleRootItem(), [])
    return out


def test_two_round_trips_keep_every_cell(qapp, tmp_path):
    dm, model = _dm(qapp)
    good, trash = make_group_row("good"), make_group_row("Trash")
    sub = make_group_row("ON")
    good[0].appendRow(make_cell_row(0, 10, 5))
    good[0].appendRow(sub)
    sub[0].appendRow(make_cell_row(1, 20, 6))
    trash[0].appendRow(make_cell_row(2, 30, 7))
    model.appendRow(good)
    model.appendRow(trash)
    want = {0: "good", 1: "good/ON", 2: "Trash"}
    assert _groups(model) == want

    p = tmp_path / "t.json"
    for _ in range(2):
        dm.save_tree_structure(p)
        dm.load_tree_structure(p)
        assert _groups(model) == want
    assert model.columnCount() == 3
    assert [c["data"] for c in json.loads(p.read_text())[1]["children"]] == [2]


def test_file_with_null_ids_is_recovered(qapp, tmp_path):
    dm, model = _dm(qapp)
    tree = [{"text": "Trash", "data": None, "child_count": 2, "children": [
                {"text": "3", "data": None, "child_count": 0},
                {"text": "99", "data": None, "child_count": 0}]},
            {"text": "7", "data": None, "child_count": 0}]
    p = tmp_path / "old.json"
    p.write_text(json.dumps(tree))
    dm.load_tree_structure(p)
    # 3 is a cluster of this sort; 99 is not, and "7" is a root folder name.
    assert _groups(model) == {3: "Trash"}
    assert model.item(1).text() == "7"
