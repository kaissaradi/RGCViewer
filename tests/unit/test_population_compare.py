"""The selected cell and pinned cells are drawn over their population (PLAN.md Q45)."""

import numpy as np
import pytest

from src.gui import callbacks
from src.gui.panels import population_compare as pc
from src.gui.panels import population_panel as pp


class FakeDM:
    # MainWindow's debounced tree-change timer reads it (a race on slow CI runners).
    cluster_df = None
    def get_cell_physics(self, cid):
        return {"timecourse": np.sin(np.arange(31) / 5.0 + cid)}

    def get_acg_data(self, cid):
        lags = np.arange(-100, 101, 1.0)
        return lags, np.exp(-np.abs(lags) / (10.0 + cid))

    def get_standard_plot_data(self, cid):
        return {"fr_bin_centers": np.arange(100.0), "fr_rate": np.full(100, float(cid))}


@pytest.fixture
def win(qtbot, monkeypatch):
    from src.gui.main_window import MainWindow
    w = MainWindow()
    w.data_manager = FakeDM()
    pp.invalidate_population_caches()
    monkeypatch.setattr(w, "_get_pop_subset_ids", lambda: [1, 2, 3])
    monkeypatch.setattr(w, "_get_selected_cluster_id", lambda: 2)
    callbacks.redraw_population_panels(w, subset=[1, 2, 3])
    yield w
    w.data_manager = None          # closeEvent would try to save a real one
    w.close()
    w.deleteLater()
    pp.invalidate_population_caches()


def _visible(state):
    lines = [ln for ln in state.get("compare_lines", []) if ln.get_visible()]
    keys = [k.get_text() for k in state.get("compare_keys", []) if k.get_visible()]
    return lines, keys


def test_full_redraw_lays_the_selected_cell_on_top(win):
    lines, keys = _visible(win.pop_timecourse_canvas._timecourse_state)
    assert keys == ["Cell 2"]
    np.testing.assert_allclose(lines[0].get_ydata(), np.sin(np.arange(31) / 5.0 + 2))
    lines, keys = _visible(win.pop_acg_canvas._acg_state)
    assert keys == ["Cell 2"] and lines[0].get_xdata().max() <= 50   # causal 0–50 ms, as the pane
    lines, _ = _visible(win.pop_fr_canvas._fr_state)
    assert np.all(lines[0].get_ydata() == 2.0)


def test_pins_join_and_leave(win):
    drawn = pc.update_overlays(win, 2, [3, 1])
    assert drawn == {"timecourse": 3, "acg": 3, "fr": 3}
    lines, keys = _visible(win.pop_fr_canvas._fr_state)
    assert keys == ["Cell 2", "Pinned 3", "Pinned 1"]
    assert len({(ln.get_color(), ln.get_linestyle()) for ln in lines}) == 3   # all distinct
    pc.update_overlays(win, 2, [])
    assert _visible(win.pop_fr_canvas._fr_state)[1] == ["Cell 2"]


def test_same_group_selection_moves_only_the_line(win):
    state = win.pop_timecourse_canvas._timecourse_state
    wash = state["shadow_lines"]
    callbacks.refresh_population_overlays(win, 3)
    assert win.pop_timecourse_canvas._timecourse_state["shadow_lines"] is wash   # group not redrawn
    assert _visible(state)[1] == ["Cell 3"]


def test_pin_toggle_and_limit():
    w = type("W", (), {})()
    for cid in range(1, 7):
        pc.toggle_pin(w, cid)
    assert w._pinned_cells == [3, 4, 5, 6]          # oldest dropped past PIN_LIMIT
    pc.toggle_pin(w, 4)
    assert w._pinned_cells == [3, 5, 6]
    pc.clear_pins(w)
    assert w._pinned_cells == []
