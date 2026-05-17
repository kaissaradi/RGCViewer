#!/usr/bin/env python3
"""Capture UMAP, sidebar, and tree-view screenshots for GUI polish verification."""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import Qt, QTimer, QModelIndex
from qtpy.QtWidgets import QApplication

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.gui.main_window import MainWindow  # noqa: E402

DEFAULT_DATA = os.path.expanduser(
    "~/Documents/Development/data/sorted/20251015A/chunk20/kilosort4"
)
OUT_DIR = REPO_ROOT / "tests" / "visual_snapshots" / "gui_polish"


def _save(pixmap, name: str):
    path = OUT_DIR / name
    pixmap.save(str(path))
    print(f"Saved {path}")


def _find_index_by_prefix(model, prefix: str, parent=QModelIndex()):
    """Return first index whose display text starts with prefix."""
    for row in range(model.rowCount(parent)):
        idx = model.index(row, 0, parent)
        text = str(model.data(idx, Qt.DisplayRole) or "")
        if text.startswith(prefix):
            return idx
        child = _find_index_by_prefix(model, prefix, idx)
        if child.isValid():
            return child
    return QModelIndex()


def _expand_nested_folders(tree, model, top_label="good", sub_label="Type_"):
    """Expand top group and first nested Type_* subfolder for screenshot."""
    top = _find_index_by_prefix(model, top_label)
    if top.isValid():
        tree.expand(top)
    sub = QModelIndex()
    if top.isValid():
        for row in range(model.rowCount(top)):
            idx = model.index(row, 0, top)
            text = str(model.data(idx, Qt.DisplayRole) or "")
            if text.startswith(sub_label):
                sub = idx
                break
    if sub.isValid():
        tree.expand(sub)
        for row in range(model.rowCount(sub)):
            child = model.index(row, 0, sub)
            text = str(model.data(child, Qt.DisplayRole) or "")
            if text.startswith(sub_label) and model.rowCount(child) > 0:
                tree.expand(child)
                break
    return top, sub


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1600, 1000)
    win.show()

    state = {"load_attempts": 0}

    def wait_for_tree():
        state["load_attempts"] += 1
        if win.tree_model.rowCount() > 0 or state["load_attempts"] > 60:
            capture_tree_sequence()
        else:
            QTimer.singleShot(500, wait_for_tree)

    def capture_tree_sequence():
        win._switch_left_view(0)
        QApplication.processEvents()

        win.tree_view.collapseAll()
        QApplication.processEvents()
        _save(win.grab(), "tree_collapsed.png")
        _save(win.tree_view.viewport().grab(), "tree_viewport_collapsed.png")

        model = win.tree_model
        top, sub = _expand_nested_folders(win.tree_view, model)
        QApplication.processEvents()
        if top.isValid():
            _save(win.grab(), "tree_good_expanded.png")
            _save(win.tree_view.viewport().grab(), "tree_viewport_good_expanded.png")
        if sub.isValid():
            _save(win.grab(), "tree_nested_folders.png")
            _save(win.tree_view.viewport().grab(), "tree_viewport_nested.png")

        capture_umap_and_sidebar()

    def capture_umap_and_sidebar():
        win.analysis_tabs.setCurrentIndex(3)
        QTimer.singleShot(150, save_umap_first_visit)

    def save_umap_first_visit():
        _save(win.grab(), "umap_first_visit.png")
        win.sidebar_toggle_btn.clicked.emit()
        QTimer.singleShot(200, save_sidebar_collapsed)

    def save_sidebar_collapsed():
        _save(win.grab(), "sidebar_collapsed.png")
        win.sidebar_toggle_btn.clicked.emit()
        QTimer.singleShot(200, save_sidebar_expanded)

    def save_sidebar_expanded():
        _save(win.grab(), "sidebar_expanded.png")
        app.quit()

    def after_show():
        if os.path.isdir(DEFAULT_DATA):
            win.load_directory(DEFAULT_DATA)
            QTimer.singleShot(500, wait_for_tree)
        else:
            print(f"Data not found: {DEFAULT_DATA}")
            capture_tree_sequence()

    QTimer.singleShot(0, after_show)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
