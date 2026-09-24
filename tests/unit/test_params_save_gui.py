"""GUI side of Ctrl+S → Vision .params (callbacks.py, shortcuts.py, DataManager)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QKeyEvent, QStandardItem, QStandardItemModel
from qtpy.QtWidgets import QApplication

from src.analysis import params_classification as pc
from src.analysis import vision_sort_check as vsc
from src.analysis.data_manager import DataManager


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


def _dm(vision_only=False):
    dm = DataManager.__new__(DataManager)
    dm.is_vision_only = vision_only
    return dm


def _tree():
    def cell(cid):
        it = QStandardItem(str(cid))
        it.setData(cid, Qt.ItemDataRole.UserRole)
        return it

    def group(name, *children):
        g = QStandardItem(name)
        for c in children:
            g.appendRow(c)
        return g

    model = QStandardItemModel()
    root = model.invisibleRootItem()
    root.appendRow(group("on", group("brisk transient", cell(5)),
                         group("Unclassified", cell(8))))
    root.appendRow(cell(9))
    root.appendRow(group("Unclassified", cell(3), group("x", cell(4))))
    return model


def test_class_ids_for_the_params_file(qapp):
    from src.gui import callbacks
    mw = SimpleNamespace(tree_model=_tree(), data_manager=_dm())
    assert callbacks.vision_class_ids(mw) == {
        6: "All/on/brisk transient",
        9: "All/on/Unclassified",   # a nested "Unclassified" is a real class
        10: "All",                  # root cell
        4: "All",                   # root "Unclassified" group
        5: "All",                   # inside it, any depth
    }


def test_text_export_keeps_nested_unclassified_and_skips_the_root_one(qapp):
    from src.gui import callbacks
    mw = SimpleNamespace(tree_model=_tree(), data_manager=_dm())
    assert callbacks.vision_classification_lines(mw) == [
        "6  All/on/brisk transient/", "9  All/on/Unclassified/", "10  All/"]


def test_classification_text_keeps_names_with_spaces():
    from src.gui import callbacks
    lines = ["12  All/ON/brisk transient/\n", "13  All/\n", "\n", "14  All/OFF/nc 3/"]
    assert callbacks.parse_classification_text(lines) == {
        12: ["ON", "brisk transient"], 13: [], 14: ["OFF", "nc 3"]}


def _diff(**kw):
    base = dict(changed=[1], unclassified=[], not_in_file=[], not_in_encore=[], n_file_rows=10)
    base.update(kw)
    return pc.ClassDiff(**base)


def test_confirm_rules():
    from src.gui import callbacks
    mw = SimpleNamespace()
    path, stamp = Path("/x/data000.params"), (1, 2)
    assert callbacks.params_save_needs_confirm(mw, path, stamp, _diff())      # first save
    mw._params_saved_stamps = {str(path): stamp}
    assert not callbacks.params_save_needs_confirm(mw, path, stamp, _diff())
    assert callbacks.params_save_needs_confirm(mw, path, (1, 3), _diff())     # saved elsewhere
    assert callbacks.params_save_needs_confirm(mw, path, stamp, _diff(unclassified=[1]))
    bad = vsc.SortCheck(200, 0.01, 0.03)
    assert callbacks.params_save_needs_confirm(mw, path, stamp, _diff(), bad)
    assert callbacks.params_save_is_risky(_diff(), bad)
    assert not callbacks.params_save_is_risky(_diff(not_in_encore=[4]))


def test_summary_text():
    from src.gui import callbacks
    path = Path("/x/data000.params")
    text = callbacks.params_save_summary(
        path, _diff(changed=[1, 2, 3], unclassified=[2], not_in_file=[7, 8], not_in_encore=[9]))
    assert "3 cells will change class in data000.params." in text
    assert "1 of them is classified in the file now" in text
    assert "2 Encore cells have no row" in text
    assert "1 row in the file has no cell in this sort" in text
    assert "data000.params.bak" in text
    assert "{" not in text and "WARNING" not in text
    warned = callbacks.params_save_summary(path, _diff(), vsc.SortCheck(240, 0.01, 0.03))
    assert warned.startswith("WARNING:") and "R² = 0.03 over 240 cells" in warned


# ── shortcuts: Ctrl+S saves; Ctrl+Shift+N marks Noisy (ux_ui_redesign AC18) ──

def _forward(key, mods):
    from src.gui.shortcuts import KeyForwarder
    marked = []
    mw = SimpleNamespace(similarity_panel=SimpleNamespace(_mark_status=marked.append))
    consumed = KeyForwarder(mw).eventFilter(None, QKeyEvent(QEvent.Type.KeyPress, key, mods))
    return consumed, marked


def test_ctrl_s_no_longer_marks_noisy(qapp):
    assert _forward(Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier) == (False, [])


def test_ctrl_shift_n_marks_noisy(qapp):
    mods = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
    assert _forward(Qt.Key.Key_N, mods) == (True, ["Noisy"])


def test_other_status_shortcuts_are_unchanged(qapp):
    assert _forward(Qt.Key.Key_D, Qt.KeyboardModifier.ControlModifier) == (True, ["Duplicate"])


# ── DataManager ────────────────────────────────────────────────────────────

def test_params_path_is_found_only_when_the_file_exists(tmp_path):
    (tmp_path / "data000.params").write_bytes(b"x")
    assert DataManager._params_path_for(tmp_path, "data000") == tmp_path / "data000.params"
    assert DataManager._params_path_for(tmp_path, "data001") is None


class _Params:
    def __init__(self, rf):
        self.rf = rf

    def get_cell_ids(self):
        return list(self.rf)

    def get_data_for_cell(self, vid, field):
        return self.rf[vid][0 if field == "x0" else 1]


def _sort_dm(shuffle):
    rng = np.random.default_rng(0)
    pos = rng.uniform(-900, 900, size=(120, 2))
    rf = pos * 0.02 + 25.0
    order = rng.permutation(120) if shuffle else np.arange(120)
    dm = _dm()
    dm.cluster_df = pd.DataFrame({"cluster_id": np.arange(120),
                                  "x_um": pos[:, 0], "y_um": pos[:, 1]})
    # Vision id = cluster + 1 (Law 1)
    dm.vision_params = _Params({int(c) + 1: tuple(rf[order[c]]) for c in range(120)})
    dm._vision_source = "src"
    dm._sort_check_cache = None
    return dm


def test_datamanager_sort_check():
    assert not _sort_dm(shuffle=False).vision_sort_check().mismatch
    assert _sort_dm(shuffle=True).vision_sort_check().mismatch
    dm = _dm()
    dm.cluster_df = None
    dm.vision_params = None
    assert not dm.vision_sort_check().decided
