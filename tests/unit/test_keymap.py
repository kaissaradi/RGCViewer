"""Keyboard workflow on the real window (PLAN.md Q39)."""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from qtpy.QtCore import QEvent, QItemSelectionModel, Qt
from qtpy.QtGui import QKeyEvent

from src.gui import callbacks, keymap
from src.gui.widgets.widgets import make_cell_row, make_group_row

README = Path(__file__).resolve().parents[2] / "README.md"


@pytest.fixture
def win(qtbot):
    from src.gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    w.activateWindow()
    qtbot.waitUntil(lambda: w.isActiveWindow(), timeout=2000)
    root = w.tree_model.invisibleRootItem()
    on = make_group_row("ON brisk sustained")
    on[0].appendRow(make_cell_row(1))
    root.appendRow(on)
    root.appendRow(make_group_row("OFF transient"))
    for cid in (5, 6, 7):
        root.appendRow(make_cell_row(cid))
    w.setup_tree_model(w.tree_model)
    w.central_widget.setEnabled(True)            # as after a run is loaded
    w.tree_view.setFocus()
    return w


def _item(w, cid):
    return callbacks.find_items_by_cluster_ids(w, [cid])[0]


def _select(w, *cids):
    sel = w.tree_view.selectionModel()
    sel.clearSelection()
    for cid in cids:
        index = w.tree_model.indexFromItem(_item(w, cid))
        sel.select(index, QItemSelectionModel.SelectionFlag.Select
                   | QItemSelectionModel.SelectionFlag.Rows)
        sel.setCurrentIndex(index, QItemSelectionModel.SelectionFlag.NoUpdate)


def _group_of(w, cid):
    parent = _item(w, cid).parent()
    return parent.text() if parent is not None else None


def test_every_binding_is_in_the_readme_and_keys_are_unique():
    text = README.read_text()
    table = text[text.index("## Keyboard shortcuts"):]
    table = table[:table.index("\n## ", 5)]
    for b in keymap.BINDINGS:
        assert b.label in table, b.label
    keys = [k for b in keymap.BINDINGS for k in b.keys]
    assert len(keys) == len(set(keys)), "a key is bound twice"


def test_delete_trashes_the_selection_from_any_focus(qtbot, win):
    _select(win, 5)
    win.tree_view.setFocus()
    qtbot.keyClick(win.tree_view, Qt.Key.Key_Delete)
    assert _group_of(win, 5) == callbacks.TRASH_GROUP_NAME
    # From a plot (not the list), too.
    _select(win, 6)
    win.analysis_tabs.currentWidget().setFocus()
    qtbot.keyClick(win.analysis_tabs.currentWidget(), Qt.Key.Key_Delete)
    assert _group_of(win, 6) == callbacks.TRASH_GROUP_NAME


def test_search_bar_keeps_its_keys(qtbot, win):
    _select(win, 7)
    bar = win.cluster_search_bar
    bar.setFocus()
    qtbot.keyClicks(bar, "on bri")               # Space used to be eaten
    assert bar.text() == "on bri"
    qtbot.keyClick(bar, Qt.Key.Key_Left)
    qtbot.keyClick(bar, Qt.Key.Key_Delete)       # deletes a character ...
    assert bar.text() == "on br"
    assert _group_of(win, 7) is None             # ... and trashes nothing
    qtbot.keyClick(bar, Qt.Key.Key_Escape)
    assert bar.text() == "" and not bar.hasFocus()


def test_ctrl_m_moves_and_ctrl_shift_m_repeats(qtbot, win, monkeypatch):
    target = next(item for path, item in callbacks.collect_group_items(win)
                  if path.endswith("OFF transient"))
    shown = []
    monkeypatch.setattr(keymap, "pick_group",
                        lambda parent, groups, preselect=None: shown.append(
                            [p for p, _ in groups]) or target)
    _select(win, 5)
    qtbot.keyClick(win.tree_view, Qt.Key.Key_M, Qt.KeyboardModifier.ControlModifier)
    assert _group_of(win, 5) == "OFF transient"
    assert any(p.endswith("ON brisk sustained") for p in shown[0])
    _select(win, 6)
    qtbot.keyClick(win.tree_view, Qt.Key.Key_M,
                   Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
    assert _group_of(win, 6) == "OFF transient"


def test_repeat_with_no_previous_move_says_so(qtbot, win):
    _select(win, 5)
    qtbot.keyClick(win.tree_view, Qt.Key.Key_M,
                   Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
    assert "Ctrl+M" in win.status_bar.currentMessage()
    assert _group_of(win, 5) is None


def test_tabs_and_views(qtbot, win):
    tabs = win.analysis_tabs
    ctrl = Qt.KeyboardModifier.ControlModifier
    for i in range(tabs.count()):
        tabs.setTabEnabled(i, True)
    qtbot.keyClick(win.tree_view, Qt.Key.Key_3, ctrl)
    assert tabs.currentIndex() == 2
    tabs.setTabEnabled(3, False)
    qtbot.keyClick(win.tree_view, Qt.Key.Key_Tab, ctrl)
    assert tabs.currentIndex() == 4                  # skips the disabled tab
    assert win.app_tab_bar.currentIndex() == 4       # header follows
    qtbot.keyClick(win.tree_view, Qt.Key.Key_T, ctrl)
    assert win.view_stack.currentIndex() == 1
    qtbot.keyClick(win.table_view, Qt.Key.Key_T, ctrl)
    assert win.view_stack.currentIndex() == 0


def test_forwarder_leaves_dialogs_menus_and_text_alone(qtbot, win, monkeypatch):
    from qtpy.QtWidgets import QDialog, QPushButton, QSpinBox
    from src.gui import shortcuts
    fwd = win.key_forwarder
    space = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
    up = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Up, Qt.KeyboardModifier.NoModifier)
    dialog = QDialog(win)
    button = QPushButton("OK", dialog)
    qtbot.addWidget(dialog)
    assert fwd.eventFilter(button, space) is False           # a dialog's own keys
    spin = QSpinBox(win)
    assert fwd.eventFilter(spin, up) is False                # a spin box keeps its arrows
    assert fwd.eventFilter(win.cluster_search_bar, space) is False
    assert fwd.eventFilter(win.cluster_search_bar, up) is True   # search → arrow
    assert fwd.eventFilter(win.tree_view, up) is True
    monkeypatch.setattr(shortcuts, "QApplication", SimpleNamespace(
        instance=lambda: SimpleNamespace(activePopupWidget=lambda: object(),
                                         focusWidget=lambda: None)))
    assert fwd.eventFilter(win.tree_view, up) is False        # an open menu


def test_group_picker_filters_and_picks(qtbot):
    groups = [("All/ON brisk sustained", "on_bs"), ("All/OFF transient", "off_t"),
              ("All/OFF brisk transient", "off_bt")]
    dlg = keymap.GroupPicker(None, groups)
    qtbot.addWidget(dlg)
    qtbot.keyClicks(dlg.filter, "off tr")
    visible = [dlg.list.item(r).text() for r in range(dlg.list.count())
               if not dlg.list.item(r).isHidden()]
    assert visible[:2] == ["All/OFF transient", "All/OFF brisk transient"]
    qtbot.keyClick(dlg.filter, Qt.Key.Key_Down)
    qtbot.keyClick(dlg.filter, Qt.Key.Key_Return)
    assert dlg.picked == "off_bt"


def test_cheat_sheet_lists_every_binding(qtbot):
    from qtpy.QtWidgets import QLabel
    sheet = keymap.CheatSheet(None)
    qtbot.addWidget(sheet)
    texts = {lbl.text() for lbl in sheet.findChildren(QLabel)}
    assert all(b.label in texts for b in keymap.BINDINGS)
    assert re.search(r"Ctrl\+M", " ".join(texts))


def test_question_mark_types_in_search_and_opens_help_elsewhere(qtbot, win, monkeypatch):
    opened = []
    monkeypatch.setattr(keymap.CheatSheet, "exec", lambda self: opened.append(1))
    bar = win.cluster_search_bar
    bar.setFocus()
    qtbot.keyClicks(bar, "?")
    assert bar.text() == "?" and opened == []
    qtbot.keyClick(win.tree_view, Qt.Key.Key_F1)
    assert opened == [1]
