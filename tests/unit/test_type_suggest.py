"""Suggested classes (PLAN.md Q36): features, suggester, GUI helpers."""

import numpy as np
import pytest

from src.analysis import type_features as tf
from src.analysis import type_library as tl
from src.analysis.rf_geometry import RFFit
from src.analysis.type_suggest import Suggester

T = np.arange(31)


def _tc(on=True, peak=26, width=2.0, biphasic=0.0):
    lobe = np.exp(-((T - peak) / width) ** 2) - biphasic * np.exp(-((T - (peak - 6)) / width) ** 2)
    return lobe if on else -lobe


def _cell(tc, burst=0.2, sx=2.0, sy=2.0):
    lags = (np.arange(600) + 0.5) * 0.5
    auto = burst * np.exp(-lags / 5.0) + (1 - burst) * np.exp(-lags / 80.0)
    return tf.CellInput(tc=tc, auto=auto, acf_binning=0.5, sigma_x=sx, sigma_y=sy, stixel=40.0)


def test_polarity_takes_the_lobe_nearest_the_spike():
    tc = _tc(on=True, biphasic=1.2)          # the older lobe is bigger, but OFF-signed
    assert tf.polarity(tc) == 1
    assert tf.polarity(-tc) == -1
    assert tf.polarity(np.zeros(31)) == 0


def test_acg_bins_sum_to_one_and_run_features_shape():
    cells = [_cell(_tc()), _cell(_tc(on=False)), _cell(_tc(), sx=1.0, sy=1.0)]
    X, pol = tf.run_features(cells)
    assert X.shape == (3, tf.N_FEATURES) and list(pol) == [1, -1, 1]
    np.testing.assert_allclose(X[:, 21:41].sum(axis=1), 1.0)
    assert np.isnan(X[2, -1])                # unmoved fit: no RF size


def test_time_is_measured_in_the_runs_own_time_to_peak():
    """The same cell recorded at half the frame rate gives the same tc_run."""
    fast = tf.run_features([_cell(_tc(peak=26, width=1.5)) for _ in range(5)])[0]
    slow = tf.run_features([_cell(_tc(peak=22, width=3.0)) for _ in range(5)])[0]
    np.testing.assert_allclose(fast[0, :21], slow[0, :21], atol=0.08)


def _library(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X, y, pol = [], [], []
    for k, (name, on, burst, bip) in enumerate([
            ("ON brisk sustained", True, 0.1, 0.0), ("ON brisk transient", True, 0.7, 0.8),
            ("OFF brisk sustained", False, 0.1, 0.0), ("OFF transient", False, 0.5, 0.4)]):
        cells = [_cell(_tc(on=on, biphasic=bip) + 0.02 * rng.standard_normal(31), burst=burst)
                 for _ in range(n)]
        f, p = tf.run_features(cells)
        X.append(f); y += [name] * n; pol.append(p)
    X = np.vstack(X)
    return tl.Library(X, np.array(y), np.array(["p%d" % (i % 4) for i in range(len(y))]),
                      np.array(["r"] * len(y)), np.concatenate(pol))


def test_suggester_learns_masks_polarity_and_flags_novel_cells():
    lib = _library()
    sg = Suggester(lib)
    out = sg.suggest(lib.X[:3], lib.polarity[:3])
    assert all(s.best[0] == "ON brisk sustained" for s in out)
    assert all(not n.startswith("OFF") for s in out for n, _p in s.ranked)   # polarity rule
    weird = np.full((1, tf.N_FEATURES), 5.0)
    assert sg.suggest(weird, [1])[0].novel


def test_library_leaves_out_a_run_with_swapped_on_off():
    X, pol = tf.run_features([_cell(_tc(on=True)) for _ in range(12)])
    runs = {"/s/A/k/d1/d1.params": {"stamp": (1, 1), "data": {
                "X": X, "y": np.array(["OFF brisk sustained"] * 12), "polarity": pol}},
            "/s/B/k/d1/d1.params": {"stamp": (1, 1), "data": {
                "X": X, "y": np.array(["ON brisk sustained"] * 12), "polarity": pol}}}
    lib = tl.assemble(runs)
    assert set(lib.run) == {"/s/B/k/d1/d1.params"}
    assert any("opposite ON/OFF" in n for n in lib.notes)
    assert tl.prep_of("/mnt/lab/Array-data/sorted/20260220A/kilosort25/data022/x.params") == "20260220A"
    assert lib.without_prep("B").n == 0


# --- GUI helpers ---------------------------------------------------------------

@pytest.fixture
def win(qtbot):
    from src.gui.main_window import MainWindow
    from src.gui.widgets.widgets import make_cell_row, make_group_row
    w = MainWindow()
    root = w.tree_model.invisibleRootItem()
    on = make_group_row("on")
    bs = make_group_row("brisk sustained")
    bs[0].appendRow(make_cell_row(1))
    on[0].appendRow(bs)
    root.appendRow(on)
    for c in (2, 3, 4):
        root.appendRow(make_cell_row(c))
    yield w
    w.data_manager = None
    w.close()
    w.deleteLater()


def test_class_folder_reuses_and_creates_the_labs_folders(win):
    from src.gui import callbacks, suggestions
    existing = suggestions.class_folder(win, "ON brisk sustained")
    assert existing.text() == "brisk sustained" and existing.parent().text() == "on"
    new = suggestions.class_folder(win, "ON brisk transient")
    assert new.text() == "brisk transient" and new.parent().text() == "on"   # under the same ON
    off = suggestions.class_folder(win, "OFF transient")
    assert off.text() == "transient" and off.parent().text() == "OFF"
    paths = [p for p, _i in callbacks.collect_group_items(win)]
    assert sum(p.endswith("brisk sustained") for p in paths) == 1


def test_mosaic_screen_holds_back_a_cell_on_top_of_a_member(win, monkeypatch):
    from src.gui import suggestions
    from src.gui.panels import types_panel
    from src.analysis.type_suggest import Suggestion
    fits = {1: RFFit(10, 10, 2, 2.0001, 0), 2: RFFit(10.5, 10, 2, 2.0001, 0),
            3: RFFit(30, 10, 2, 2.0001, 0), 4: RFFit(30.4, 10.2, 2, 2.0001, 0)}
    monkeypatch.setattr(types_panel, "rf_fits", lambda dm, cells: {c: fits.get(c) for c in cells})
    win.data_manager = object()
    s = lambda p: Suggestion([("ON brisk sustained", p)], False, 0.1, 1)   # noqa: E731
    win._suggestions = suggestions.RunSuggestions({2: s(0.95), 3: s(0.9), 4: s(0.85)}, 100)
    kept, held = suggestions.mosaic_screen(
        win, {"ON brisk sustained": [2, 3, 4]}, {1: "ON brisk sustained"})
    assert kept == {"ON brisk sustained": [3]} and sorted(held) == [2, 4]
