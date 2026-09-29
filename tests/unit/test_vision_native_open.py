"""Opening Vision files with no Kilosort run (2026-09-28 regression and its follow-ups)."""

import numpy as np
import pandas as pd
import pytest


class _Stop(Exception):
    pass


def test_vision_native_open_leaves_the_welcome_screen(qtbot, monkeypatch, tmp_path):
    """The load ran behind the welcome page, so the window looked empty after "loaded"."""
    from qtpy.QtWidgets import QFileDialog
    from src.gui import callbacks, recent_paths
    from src.gui.main_window import MainWindow
    # Never the user's settings file: the load remembers the folder. Stub the
    # two calls rather than hand in a QSettings: one destroyed after the
    # QApplication segfaulted Python 3.13 at exit (CI, empty HOME).
    monkeypatch.setattr(recent_paths, "remember_dir", lambda *a, **k: None)
    monkeypatch.setattr(recent_paths, "last_dir", lambda *a, **k: "")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(tmp_path)))
    seen = {}

    def stop_here(mw):
        seen["on_analysis_view"] = mw.central_stack.currentWidget() is mw.central_widget
        raise _Stop()
    monkeypatch.setattr(callbacks, "_release_previous_dataset", stop_here)
    w = MainWindow()
    try:
        assert w.central_stack.currentWidget() is w.welcome_panel
        with pytest.raises(_Stop):
            callbacks.load_vision_directory(w)
        assert seen["on_analysis_view"]
    finally:
        w.data_manager = None
        w.close()
        w.deleteLater()


def _native_dm():
    from src.analysis.data_manager import DataManager
    dm = DataManager.__new__(DataManager)
    dm.cluster_df = pd.DataFrame({"cluster_id": [1, 2, 3], "n_spikes": [10, 20, 30],
                                  "best_chan": [-1, -1, -1], "x_um": np.nan, "y_um": np.nan})
    dm.native_channels_pending = True
    dm._sort_check_cache = ("old", None)
    return dm


def test_channels_from_the_eis_replace_placeholder_seeds(monkeypatch):
    from src.analysis import axon_bearing
    dm = _native_dm()
    dm._native_vision_source = ("/x/data022", "data022")
    pos = np.array([[0.0, 0.0], [60.0, 0.0], [120.0, 30.0]])
    cells = {1: {"amin": np.array([5.0, 90.0, 3.0])}, 2: {"amin": np.array([1.0, 2.0, 50.0])}}
    monkeypatch.setattr(axon_bearing, "read_run_eis", lambda folder, ds, progress=None: (cells, pos))
    chans = dm.native_channels_from_eis()
    assert chans == {1: (1, 60.0, 0.0), 2: (2, 120.0, 30.0)}
    assert dm.apply_native_channels(chans) == 2
    df = dm.cluster_df.set_index("cluster_id")
    assert df.loc[1, "best_chan"] == 1 and df.loc[2, "x_um"] == 120.0 and df.loc[3, "best_chan"] == -1
    assert not dm.native_channels_pending and dm._sort_check_cache is None   # the check runs again


def test_tree_channel_column_updates_in_place(qtbot):
    from types import SimpleNamespace
    from qtpy.QtGui import QStandardItemModel
    from src.gui import callbacks
    from src.gui.widgets.widgets import TREE_COL_CH, make_cell_row, make_group_row
    model = QStandardItemModel()
    g = make_group_row("ON", 2)
    g[0].appendRow(make_cell_row(1, 10, -1))
    g[0].appendRow(make_cell_row(2, 20, -1))
    model.invisibleRootItem().appendRow(g)
    mw = SimpleNamespace(tree_model=model)
    assert callbacks.update_tree_channels(mw, {1: 7, 2: 352}) == 2
    group = model.invisibleRootItem().child(0, 0)
    assert [group.child(r, TREE_COL_CH).text() for r in range(2)] == ["7", "352"]
    del g, group                              # let the model own and free its items
    model.clear()
