"""Types tab: groups from the tree, bins vs types, cached-only data (PLAN.md Q40/Q42)."""

import numpy as np
import pytest

from src.gui.panels import types_panel as tp
from src.gui.widgets.widgets import make_cell_row, make_group_row


class FakeDM:
    generation = 1
    # MainWindow's debounced tree-change timer reads it (a race on slow CI runners).
    cluster_df = None
    chirp_available = False
    vision_params = None

    def __init__(self):
        t = np.linspace(0, 1, 31)
        self.tc = {c: (1 if c < 10 else -1) * np.sin(np.pi * t) for c in range(20)}

    def peek_cell_physics(self, cid):
        return {"timecourse": self.tc[cid]} if cid in self.tc else None

    def peek_standard_plot_data(self, cid):
        lags = np.arange(-100, 101, 1.0)
        return {"acg_time_lags": lags, "acg_norm": np.exp(-np.abs(lags) / 10.0)}

    def get_cell_physics(self, cid):          # must never be called by the tab
        raise AssertionError("the Types tab computed physics on the GUI thread")


@pytest.mark.parametrize("name, is_type", [
    ("ON brisk sustained", True), ("on/brisk transient", True), ("my DS cells", True),
    ("on/unclassified", False), ("weak", False), ("weak/nc10", False), ("off", False),
    ("Unclassified", False),
])
def test_bins_are_not_types(name, is_type):
    assert tp.is_type_group(name) is is_type


@pytest.fixture
def win(qtbot):
    from src.gui.main_window import MainWindow
    w = MainWindow()
    w.data_manager = FakeDM()
    root = w.tree_model.invisibleRootItem()
    on = make_group_row("All")
    sub = make_group_row("ON brisk sustained")
    for c in range(10):
        sub[0].appendRow(make_cell_row(c))
    on[0].appendRow(sub)
    root.appendRow(on)
    off = make_group_row("OFF transient")
    for c in range(10, 20):
        off[0].appendRow(make_cell_row(c))
    root.appendRow(off)
    trash = make_group_row("Trash")
    trash[0].appendRow(make_cell_row(99))
    root.appendRow(trash)
    root.appendRow(make_cell_row(77))            # not in a folder
    yield w
    w.data_manager = None
    w.close()
    w.deleteLater()


def test_tree_groups_skip_trash_and_loose_cells(win):
    group_of, order = tp.tree_groups(win)
    assert order == ["ON brisk sustained", "OFF transient"]     # 'All/' dropped
    assert 99 not in group_of and 77 not in group_of and len(group_of) == 20


def test_panel_builds_from_cached_data_only(win):
    panel = win.types_panel
    panel.refresh(force=True)
    code = panel._barcode
    assert [b[0] for b in code.bands] == ["ON brisk sustained", "OFF transient"]
    assert "20 of 20 grouped cells shown" in panel.status.text()
    panel.highlight(12)
    assert len(panel.bar_selected.getData()[0]) == 5
    panel.feature_combo.setCurrentText("Chirp response")
    assert "no chirp data" in panel.status.text()
