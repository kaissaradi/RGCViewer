"""Crash guard and the small audit fixes (PLAN.md Q21, Q22).

An exception escaping a Qt slot aborts PyQt6 (SIGABRT, exit 134) unless
sys.excepthook is replaced. crash_guard.install() replaces it.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from qtpy.QtWidgets import QApplication, QTableView

REPO = Path(__file__).resolve().parents[2]

_APP = textwrap.dedent("""
    import sys
    sys.path.insert(0, {repo!r})
    from qt_bootstrap import prefer_bundled_qt; prefer_bundled_qt()
    from qtpy.QtCore import QObject, QTimer, Signal
    from qtpy.QtWidgets import QApplication
    app = QApplication([])
    if {guard}:
        from src.gui import crash_guard
        crash_guard.install({log!r})
    class E(QObject):
        sig = Signal()
    e = E()
    def boom():
        raise AttributeError("summary_tab")
    e.sig.connect(boom)
    QTimer.singleShot(20, e.sig.emit)
    QTimer.singleShot(300, lambda: (print("STILL ALIVE"), app.quit()))
    app.exec()
""")


def _run(guard, log):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    code = _APP.format(repo=str(REPO), guard=guard, log=str(log))
    return subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, timeout=60)


def test_slot_exception_aborts_without_the_guard(tmp_path):
    r = _run(False, tmp_path / "e.log")
    assert r.returncode != 0 and "STILL ALIVE" not in r.stdout


def test_crash_guard_keeps_the_app_running_and_logs(tmp_path):
    log = tmp_path / "logs" / "errors.log"
    r = _run(True, log)
    assert r.returncode == 0, r.stderr
    assert "STILL ALIVE" in r.stdout
    assert "AttributeError: summary_tab" in log.read_text()


# ── Q22: spatial result no longer touches the removed summary_tab ─────────

def test_spatial_result_redraws_the_ei_panel_without_summary_tab():
    from src.gui import callbacks

    drawn = []
    ei_panel = SimpleNamespace(update_ei=lambda ids: drawn.append(ids))
    mw = SimpleNamespace(
        _get_selected_cluster_id=lambda: 7,
        ei_panel=ei_panel,
        analysis_tabs=SimpleNamespace(currentWidget=lambda: ei_panel),
        status_bar=SimpleNamespace(showMessage=lambda *a, **k: None),
    )  # no summary_tab attribute, like the real window
    callbacks.on_spatial_data_ready(mw, 7, {})
    callbacks.on_spatial_data_ready(mw, 8, {})   # not the selected cell
    assert drawn == [[7]]


# ── Q22: 30 um 519 arrays get the disconnected-electrode set ──────────────

@pytest.mark.parametrize("array_id", [1501, 1504, 3501])
def test_519_arrays_share_the_disconnected_set(array_id):
    from src.analysis import electrode_map as em
    assert em.get_disconnected_electrode_set_by_array_id(array_id) == {
        0, 129, 258, 259, 388, 389, 518}


def test_512_array_has_no_disconnected_set():
    from src.analysis import electrode_map as em
    assert em.get_disconnected_electrode_set_by_array_id(504) == set()


# ── Q21: selecting a cell in the table view ───────────────────────────────

@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


def test_select_cluster_in_table_finds_the_row(qapp):
    from src.gui.main_window import MainWindow
    from src.gui.widgets.widgets import HighlightStatusPandasModel

    class _Host:
        _select_cluster_in_table = MainWindow._select_cluster_in_table
        _select_table_cluster_id = MainWindow._select_table_cluster_id

        def __init__(self):
            self.table_view = QTableView()
            self.main_cluster_model = HighlightStatusPandasModel(
                pd.DataFrame({"cluster_id": [10, 20, 30], "n_spikes": [1, 2, 3]}))
            self.table_view.setModel(self.main_cluster_model)
            self.tree_fallbacks = []

        def _select_cluster_in_tree(self, cid):
            self.tree_fallbacks.append(cid)
            return False

    host = _Host()
    assert host._select_cluster_in_table(30) is True
    assert host.table_view.currentIndex().row() == 2
    assert host._select_cluster_in_table(99) is False     # not in the table
    assert host.tree_fallbacks == [99]


# ── Q22 / Q24: classification export matches Vision's format ─────────────

def _tree(qapp):
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QStandardItem, QStandardItemModel

    def cell(cid):
        it = QStandardItem(str(cid))
        it.setData(cid, Qt.ItemDataRole.UserRole)
        return it

    model = QStandardItemModel()
    on = QStandardItem("on")
    brisk = QStandardItem("brisk")
    brisk.appendRow(cell(5))
    brisk.appendRow(cell(6))
    on.appendRow(brisk)
    model.invisibleRootItem().appendRow(on)
    model.invisibleRootItem().appendRow(cell(9))
    unc = QStandardItem("Unclassified")
    unc.appendRow(cell(3))
    model.invisibleRootItem().appendRow(unc)
    return model


@pytest.mark.parametrize("vision_only, expected", [
    (False, ["6  All/on/brisk/", "7  All/on/brisk/", "10  All/"]),
    (True, ["5  All/on/brisk/", "6  All/on/brisk/", "9  All/"]),
])
def test_vision_classification_lines(qapp, vision_only, expected):
    from src.analysis.data_manager import DataManager
    from src.gui import callbacks

    dm = DataManager.__new__(DataManager)
    dm.is_vision_only = vision_only
    mw = SimpleNamespace(tree_model=_tree(qapp), data_manager=dm)
    assert callbacks.vision_classification_lines(mw) == expected
