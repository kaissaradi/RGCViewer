"""Every keyboard shortcut in one table (PLAN.md Q39).

BINDINGS drives three things, so they cannot drift apart:
- ``install(window)`` makes a window-level QShortcut for each row with a slot,
- F1 (or ?) shows the table as a cheat sheet,
- tests/unit/test_keymap.py checks README.md's shortcut table against it.

Rows without a slot are handled elsewhere (``shortcuts.KeyForwarder``, the
Ctrl+S QAction, the Ctrl+F QShortcut) and are listed for the cheat sheet.

Window-level shortcuts do not fire while a text field has focus: Qt lets
QLineEdit claim editing keys (Delete, Backspace, arrows, letters) first.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QDialog, QDialogButtonBox, QInputDialog, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QShortcut, QVBoxLayout, QGridLayout, QWidget, QScrollArea,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Binding:
    keys: Tuple[str, ...]      # QKeySequence portable text, e.g. "Ctrl+Shift+M"
    label: str                 # what it does, as the cheat sheet says it
    section: str
    slot: Optional[str] = None  # function name in this module; None = bound elsewhere


SECTIONS = ("Cells", "Groups", "Views", "File")

BINDINGS: List[Binding] = [
    Binding(("Up", "Down"), "Previous / next cell in the list", "Cells"),
    Binding(("Delete", "Backspace"), "Move the selected cells to Trash", "Cells", "trash_selection"),
    Binding(("Ctrl+M",), "Move the selection to a group (type to filter)", "Cells", "move_to_group"),
    Binding(("Ctrl+Shift+M",), "Move the selection to the last group used", "Cells", "move_to_last_group"),
    Binding(("Ctrl+G",), "Put the selected cells in a new group", "Cells", "group_selection"),
    Binding(("Ctrl+D", "Ctrl+C", "Ctrl+E", "Ctrl+W", "Ctrl+X", "Ctrl+A"),
            "Mark Duplicate / Clean / Edge / Unsure / Contaminated / Off Array", "Cells"),
    Binding(("Ctrl+Shift+N",), "Mark Noisy", "Cells"),
    Binding(("Space",), "Next row of the similarity table", "Cells"),
    Binding(("Ctrl+Return", "Ctrl+Enter"), "Accept the suggested class, go to the next cell to review", "Cells", "accept_suggestion"),
    Binding(("Ctrl+J",), "Next cell to review (suggested classes)", "Cells", "next_suggestion"),
    Binding(("F2",), "Rename the selected group", "Groups", "rename_group"),
    Binding(("Ctrl+Shift+F",), "Feature Extraction on the selection", "Groups", "feature_extraction"),
    Binding(("Ctrl+1", "Ctrl+9"), "Go to analysis tab 1 … 9 (Ctrl+0: tab 10)", "Views"),
    Binding(("Ctrl+Tab", "Ctrl+Shift+Tab"), "Next / previous analysis tab", "Views"),
    Binding(("Ctrl+T",), "Switch the cell list between tree and table", "Views", "toggle_left_view"),
    Binding(("Ctrl+P",), "Show / hide the population pane beside the cell", "Views", "toggle_population"),
    Binding(("Ctrl+K",), "Pin / unpin the cell to compare it with others (up to 4)", "Views", "toggle_pin"),
    Binding(("Ctrl+Shift+K",), "Clear the pinned cells", "Views", "clear_pins"),
    Binding(("Ctrl+F",), "Search the cell list (Esc clears)", "Views"),
    Binding(("Left", "Right"), "Previous / next EI overlay cell", "Views"),
    Binding(("Ctrl+O",), "Open a Kilosort run", "File", "open_run"),
    Binding(("Ctrl+S",), "Save the classification to the Vision .params", "File"),
    Binding(("F1", "?"), "Show these shortcuts", "File", "show_cheat_sheet"),
]

# Tab keys are generated, not one row each.
_TAB_KEYS = [(f"Ctrl+{n}", n - 1) for n in range(1, 10)] + [("Ctrl+0", 9)]


# --- selection -------------------------------------------------------------

def _selection(w):
    """(tree items or None, cluster ids) for the active left view."""
    from . import callbacks
    if w.view_stack.currentIndex() == 0:
        items = w._get_selected_tree_items()
        ids = []
        for item in items:
            ids.extend(callbacks.iter_cluster_ids(item))
        return items, list(dict.fromkeys(ids))
    return None, w._get_selected_table_cluster_ids()


def _nothing_selected(w):
    w.status_bar.showMessage("Select one or more cells first.", 3000)


# --- actions ---------------------------------------------------------------

def trash_selection(w):
    from . import callbacks
    items, ids = _selection(w)
    if items:
        callbacks.move_items_to_trash(w, items)
    elif ids:
        callbacks.move_cluster_ids_to_trash(w, ids)


def _move(w, target):
    from . import callbacks
    items, ids = _selection(w)
    n = 0
    if items:
        n = callbacks.move_items_to_group(w, items, target)
    elif ids:
        n = callbacks.move_cluster_ids_to_group(w, ids, target)
    if n:
        w._last_move_target = target
    return n


def move_to_group(w):
    from . import callbacks
    items, ids = _selection(w)
    if not (items or ids):
        return _nothing_selected(w)
    groups = callbacks.collect_group_items(w, exclude=items or [])
    last = getattr(w, "_last_move_target", None)
    picked = pick_group(w, groups, preselect=last)
    if picked is None:
        return
    if picked == NEW_GROUP:
        return group_selection(w)
    _move(w, picked)


def move_to_last_group(w):
    from . import callbacks
    target = getattr(w, "_last_move_target", None)
    alive = target is not None and any(
        item is target for _p, item in callbacks.collect_group_items(w))
    if not alive:
        w.status_bar.showMessage("No group used yet: press Ctrl+M to pick one.", 4000)
        return
    _move(w, target)


def group_selection(w):
    from . import callbacks
    _items, ids = _selection(w)
    if not ids:
        return _nothing_selected(w)
    name, ok = QInputDialog.getText(w, "New Group", f"Name for the group of {len(ids)} cells:")
    if ok and name.strip():
        callbacks.group_clusters_in_tree(w, ids, name.strip())


def rename_group(w):
    from . import callbacks
    items, _ids = _selection(w)
    groups = [i for i in (items or []) if callbacks.is_group_item(i)]
    if len(groups) != 1:
        w.status_bar.showMessage("Select one group in the tree to rename it.", 3000)
        return
    item = groups[0]
    name, ok = QInputDialog.getText(w, "Rename Group", "Group name:", text=item.text())
    if ok and name.strip() and name.strip() != item.text():
        callbacks.rename_class(w, item.text(), name.strip())


def feature_extraction(w):
    from . import callbacks
    _items, ids = _selection(w)
    if not ids:
        return _nothing_selected(w)
    callbacks.feature_extraction(w, ids)


def toggle_left_view(w):
    w._switch_left_view(1 - w.view_stack.currentIndex())


def toggle_population(w):
    w.pop_view_btn.toggle()


def toggle_pin(w):
    from . import callbacks
    from .panels import population_compare as pc
    cid = w._get_selected_cluster_id()
    if cid is None:
        return _nothing_selected(w)
    pins = pc.toggle_pin(w, cid)
    if not w.pop_view_btn.isChecked():
        w.pop_view_btn.setChecked(True)      # the comparison lives in that pane
    callbacks.refresh_population_overlays(w, cid)
    state = "Pinned" if int(cid) in pins else "Unpinned"
    w.status_bar.showMessage(
        f"{state} cell {cid}. Pinned: {', '.join(map(str, pins)) or 'none'} "
        f"(Ctrl+Shift+K clears).", 5000)


def clear_pins(w):
    from . import callbacks
    from .panels import population_compare as pc
    pc.clear_pins(w)
    callbacks.refresh_population_overlays(w)
    w.status_bar.showMessage("Pinned cells cleared.", 3000)


def accept_suggestion(w):
    from . import suggestions
    suggestions.accept(w)


def next_suggestion(w):
    from . import suggestions
    suggestions.next_to_review(w)


def open_run(w):
    w.load_directory()


def show_cheat_sheet(w):
    CheatSheet(w).exec()


def go_to_tab(w, index):
    tabs = w.analysis_tabs
    if 0 <= index < tabs.count() and tabs.isTabEnabled(index):
        tabs.setCurrentIndex(index)


def step_tab(w, step):
    """Next / previous enabled tab, wrapping around."""
    tabs = w.analysis_tabs
    n = tabs.count()
    i = tabs.currentIndex()
    for _ in range(n):
        i = (i + step) % n
        if tabs.isTabEnabled(i):
            tabs.setCurrentIndex(i)
            return


# --- install ---------------------------------------------------------------

def _shortcut(w, key: str, fn: Callable) -> QShortcut:
    sc = QShortcut(QKeySequence(key), w)
    sc.setContext(Qt.ShortcutContext.WindowShortcut)
    sc.activated.connect(fn)
    return sc


def install(w) -> List[QShortcut]:
    """Create the window-level shortcuts on MainWindow ``w``."""
    made = []
    for b in BINDINGS:
        if b.slot is None:
            continue
        fn = globals()[b.slot]
        for key in b.keys:
            made.append(_shortcut(w, key, lambda fn=fn: fn(w)))
    for key, index in _TAB_KEYS:
        made.append(_shortcut(w, key, lambda index=index: go_to_tab(w, index)))
    for key, step in (("Ctrl+Tab", 1), ("Ctrl+PgDown", 1),
                      ("Ctrl+Shift+Tab", -1), ("Ctrl+Backtab", -1), ("Ctrl+PgUp", -1)):
        made.append(_shortcut(w, key, lambda step=step: step_tab(w, step)))
    w._keymap_shortcuts = made
    return made


def key_hint(slot: str) -> str:
    """The first key of a binding, as the menus show it (e.g. 'Ctrl+M')."""
    for b in BINDINGS:
        if b.slot == slot:
            return QKeySequence(b.keys[0]).toString(QKeySequence.SequenceFormat.NativeText)
    return ""


# --- group picker ------------------------------------------------------------

NEW_GROUP = object()


class GroupPicker(QDialog):
    """Type to filter the groups, Enter to pick, Esc to cancel."""

    def __init__(self, parent, groups, preselect=None):
        super().__init__(parent)
        self.setWindowTitle("Move to group")
        self.setMinimumWidth(360)
        self.picked = None
        layout = QVBoxLayout(self)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Type to filter groups…")
        self.filter.setClearButtonEnabled(True)
        layout.addWidget(self.filter)
        self.list = QListWidget()
        layout.addWidget(self.list)
        hint = QLabel("↑ ↓ choose · Enter move · Esc cancel · Ctrl+Shift+M repeats the last move")
        hint.setObjectName("mutedLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        for path, item in groups:
            entry = QListWidgetItem(path)
            entry.setData(Qt.ItemDataRole.UserRole, item)
            self.list.addItem(entry)
            if item is preselect:
                self.list.setCurrentItem(entry)
        new = QListWidgetItem("＋ New group…")
        new.setData(Qt.ItemDataRole.UserRole, NEW_GROUP)
        self.list.addItem(new)
        if self.list.currentRow() < 0:
            self.list.setCurrentRow(0)

        self.filter.textChanged.connect(self._apply_filter)
        self.filter.returnPressed.connect(self._accept_current)
        self.list.itemActivated.connect(lambda _e: self._accept_current())
        self.filter.installEventFilter(self)
        self.filter.setFocus()

    def eventFilter(self, obj, event):
        # Arrow keys in the filter box move the list, so the hands stay put.
        if obj is self.filter and event.type() == QEvent.Type.KeyPress \
                and event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            step = -1 if event.key() == Qt.Key.Key_Up else 1
            row = self.list.currentRow()
            for _ in range(self.list.count()):
                row = (row + step) % self.list.count()
                if not self.list.item(row).isHidden():
                    self.list.setCurrentRow(row)
                    break
            return True
        return super().eventFilter(obj, event)

    def _apply_filter(self, text):
        words = text.lower().split()
        first = None
        for row in range(self.list.count()):
            entry = self.list.item(row)
            show = entry.data(Qt.ItemDataRole.UserRole) is NEW_GROUP or \
                all(wd in entry.text().lower() for wd in words)
            entry.setHidden(not show)
            if show and first is None:
                first = row
        current = self.list.currentItem()
        if (current is None or current.isHidden()) and first is not None:
            self.list.setCurrentRow(first)

    def _accept_current(self):
        entry = self.list.currentItem()
        if entry is not None and not entry.isHidden():
            self.picked = entry.data(Qt.ItemDataRole.UserRole)
            self.accept()


def pick_group(parent, groups, preselect=None):
    """The chosen tree item, NEW_GROUP, or None when cancelled."""
    dlg = GroupPicker(parent, groups, preselect)
    return dlg.picked if dlg.exec() == QDialog.DialogCode.Accepted else None


# --- cheat sheet -------------------------------------------------------------

class CheatSheet(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Keyboard shortcuts")
        self.setMinimumSize(520, 480)
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        grid = QGridLayout(body)
        grid.setColumnStretch(1, 1)
        row = 0
        for section in SECTIONS:
            head = QLabel(section.upper())
            head.setObjectName("sectionLabel")
            grid.addWidget(head, row, 0, 1, 2)
            row += 1
            for b in BINDINGS:
                if b.section != section:
                    continue
                keys = QLabel("  ·  ".join(
                    QKeySequence(k).toString(QKeySequence.SequenceFormat.NativeText) or k
                    for k in b.keys))
                keys.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                grid.addWidget(keys, row, 0, Qt.AlignmentFlag.AlignTop)
                text = QLabel(b.label)
                text.setWordWrap(True)
                grid.addWidget(text, row, 1)
                row += 1
        scroll.setWidget(body)
        outer.addWidget(scroll)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
