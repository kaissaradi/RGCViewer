"""DS grating runs in one tissue frame (PLAN.md Q51)."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis import ds_pool as dp


def test_four_lobes_give_their_axis():
    rng = np.random.default_rng(0)
    theta = np.concatenate([a + rng.normal(0, 8, 30) for a in (20, 110, 200, 290)])
    axis, strength = dp.four_fold_axis(theta)
    assert abs(axis - 20) < 4 and strength > 0.8
    assert dp.four_fold_axis(rng.uniform(0, 360, 400))[1] < 0.15
    assert np.isnan(dp.four_fold_axis([])[0])


def test_a_grating_label_moves_the_bars_the_other_way():
    # Stage: θ = 0 drifts the bars left (see MOTION_OFFSET_DEG).
    np.testing.assert_allclose(dp.motion_vectors([0.0, 90.0]), [[-1, 0], [0, -1]], atol=1e-12)
    np.testing.assert_allclose(dp.motion_deg([0.0, 270.0]), [180.0, 90.0])


def test_disc_frame_undoes_the_turn_and_the_bearing():
    # The rig's quarter turn: screen = M · array with M = [[0, 1], [-1, 0]], so
    # array +y (90°) shows as screen +x (0°). The disc lies at array bearing 90°.
    # Label 180° moves the bars to screen +x = array +y = toward the disc: 0°.
    # Label 0° moves them to screen -x = array -y = away: 180°. Label 90° moves
    # them to screen -y = array +x, 90° clockwise of the disc: 270°.
    run = dp.RunDS("p/s/r", theta_deg=np.array([180.0, 0.0, 90.0]), dsi=np.ones(3),
                   bearing_verdict="disc", bearing_deg=90.0, turn=(0, 1, -1, 0))
    np.testing.assert_allclose(run.angles("disc"), [0.0, 180.0, 270.0], atol=1e-9)
    np.testing.assert_allclose(run.angles("screen"), [0.0, 180.0, 270.0])


def test_no_disc_frame_without_a_bearing_or_a_turn():
    run = dp.RunDS("p/s/r", theta_deg=np.array([10.0]), dsi=np.ones(1))
    assert run.angles("disc") is None and not run.can_align
    run.bearing_verdict, run.bearing_deg = "direction", 45.0
    assert run.angles("disc") is None                         # still no turn
    run.turn = (1, 0, 0, 1)
    assert run.can_align
    np.testing.assert_allclose(run.angles("disc"), [(10 + 180 - 45) % 360])
    with pytest.raises(ValueError):
        run.angles("retina")


def test_summary_round_trips_through_json():
    run = dp.RunDS("p/s/r", n_cells=40, has_directions=True, ids=[3, 7],
                   theta_deg=np.array([12.5, 250.0]), dsi=np.array([0.4, 0.8]),
                   bearing_verdict="none", n_axons=4, turn=(0, 1, -1, 0), turn_source="data001 (R² 0.80)")
    back = dp.RunDS.from_json(json.loads(json.dumps(run.to_json())))
    assert back.run == run.run and back.ids == [3, 7] and back.turn == (0, 1, -1, 0)
    np.testing.assert_allclose(back.theta_deg, run.theta_deg)
    assert np.isnan(back.bearing_deg) and back.turn_source == run.turn_source


def _analysed(pref_by_cell):
    """An analysed-file dict: one dsos condition per cell, DS at ``pref``."""
    out = {}
    for cid, (pref, dsi) in pref_by_cell.items():
        dirs = np.arange(0, 360, 45.0)
        resp = 1 + dsi * np.cos(np.radians(dirs - pref))
        out[cid] = {(100.0, 2.0): {"condition_type": "dsos", "directions_deg": dirs,
                                   "mean_response": resp, "sem_response": np.zeros(8),
                                   "DSI": dsi, "preferred_direction_deg": pref, "DSI_pvalue": 0.001,
                                   "OSI": 0.05, "preferred_orientation_deg": 0.0, "OSI_pvalue": 0.5}}
    return out


def test_ds_cells_from_an_analysed_file_uses_the_grating_tab_rule():
    ids, theta, dsi, n, has_dirs = dp.ds_cells(_analysed({5: (30.0, 0.6), 6: (200.0, 0.1)}))
    assert has_dirs and n == 2
    assert ids == [5] and theta.tolist() == [30.0]              # 0.1 is below the DS threshold


def test_ds_cells_from_raw_trials_with_one_direction_says_so():
    tp = [{"orientation": 0.0, "barWidth": bw, "temporalFrequency": 2.0, "preTime": 500.0,
           "stimTime": 1000.0, "tailTime": 500.0} for bw in (40.0, 80.0, 160.0)]
    raw = {"trial_parameters": tp, "spike_times_by_trial": {1: [np.array([600.0, 700.0])] * 3}}
    ids, theta, dsi, n, has_dirs = dp.ds_cells(raw)
    assert not has_dirs and n == 1 and ids == []


def test_find_runs_prefers_the_analysed_file_and_siblings_sort_by_distance(tmp_path):
    for run in ("data002", "data006", "data012"):
        (tmp_path / "20260220A" / "kilosort25" / run).mkdir(parents=True)
    g = tmp_path / "20260220A" / "kilosort25" / "data012"
    (g / "data012_GratingDSOS.npy").write_bytes(b"")
    (g / "data012_GratingDSOS_analyzed.npy").write_bytes(b"")
    for run in ("data002", "data006"):
        d = tmp_path / "20260220A" / "kilosort25" / run
        (d / f"{run}.params").write_bytes(b"")
        (d / f"{run}.ei").write_bytes(b"")
    found = dp.find_grating_runs(tmp_path)
    assert [p.name for p in found] == ["data012_GratingDSOS_analyzed.npy"]
    assert dp.find_grating_runs(tmp_path, prep="20260220A") == found
    assert [d.name for d in dp.sibling_runs(g)] == ["data006", "data002"]


def test_summary_cache_is_used_only_while_its_sources_are_unchanged(tmp_path):
    src = tmp_path / "a.npy"
    src.write_bytes(b"x")
    stamp = dp._stamp([src])
    path = tmp_path / dp.CACHE_NAME
    run = dp.RunDS("p/s/r", n_cells=3)
    dp._write_cache(path, stamp, run)
    assert dp._read_cache(path, stamp).n_cells == 3
    assert not list(tmp_path.glob("*" + dp.TEMP_SUFFIX))
    src.write_bytes(b"xy")                                      # the grating file changed
    assert dp._read_cache(path, dp._stamp([src])) is None


def test_soma_is_the_weighted_centre_of_the_largest_electrodes():
    pos = np.array([[0, 0], [60, 0], [120, 0], [180, 0]], float)
    cells = {1: {"amin": np.array([1.0, 300.0, 100.0, 2.0])}}
    x, y = dp.soma_positions(cells, pos)[1]
    assert 60 < x < 120 and y == 0


def test_one_copy_per_recording_prefers_the_open_sort():
    a = Path("/r/20260220A/kilosort25/data023/data023_GratingDSOS.npy")
    b = Path("/r/20260220A/kilosort40/data023/data023_GratingDSOS.npy")
    c = Path("/r/20260227A/kilosort25/data012/data012_GratingDSOS.npy")
    assert dp.prefer_sorter([a, b, c], "kilosort25") == [a, c]
    assert dp.prefer_sorter([a, b, c], "kilosort40") == [b, c]
    assert dp.prefer_sorter([a, b], "other") == [b]


def _run(name, n, pref=(0, 90, 180, 270), aligned=True):
    rng = np.random.default_rng(len(name))
    theta = np.concatenate([a + rng.normal(0, 10, n // 4) for a in pref])
    return dp.RunDS(name, n_cells=3 * n, has_directions=True, ids=list(range(len(theta))),
                    theta_deg=theta, dsi=np.full(len(theta), 0.5),
                    bearing_verdict="direction" if aligned else "none",
                    bearing_deg=30.0 if aligned else float("nan"),
                    turn=(0, 1, -1, 0) if aligned else None, turn_source="data001 (R² 0.8)")


def test_one_run_per_prep_is_pooled_by_default():
    from src.gui.panels.ds_compare_dialog import one_run_per_prep
    runs = [_run("A/ks/data001", 20), _run("A/ks/data002", 40), _run("B/ks/data005", 12),
            dp.RunDS("C/ks/data009", n_cells=50, has_directions=False)]
    assert one_run_per_prep(runs) == {"A/ks/data001": False, "A/ks/data002": True,
                                      "B/ks/data005": True, "C/ks/data009": False}


def test_current_run_dir_from_params_or_ksfiles(tmp_path):
    from types import SimpleNamespace
    from src.gui.panels.ds_compare_dialog import current_run_dir
    run = tmp_path / "20260220A" / "kilosort25" / "data022"
    assert current_run_dir(SimpleNamespace(vision_params_path=run / "data022.params")) == run
    assert current_run_dir(SimpleNamespace(vision_params_path=None,
                                           kilosort_dir=run / "ksfiles")) == run
    assert current_run_dir(SimpleNamespace()) is None


def test_dialog_lists_runs_and_draws_both_frames(qtbot, tmp_path):
    from src.gui.main_window import MainWindow
    from src.gui.panels import ds_compare_dialog as dc
    w = MainWindow()
    try:
        dlg = dc.DSCompareDialog(w, tmp_path, "A", "A/ks/data002")
        qtbot.addWidget(dlg)
        qtbot.waitUntil(lambda: dlg._state["finished"], timeout=5000)   # empty root: no runs
        dlg._poll()
        assert "No grating runs" in dlg.status.text()
        runs = [_run("A/ks/data001", 20), _run("A/ks/data002", 40), _run("B/ks/data005", 12, aligned=False)]
        dlg._state = {"paths": runs, "done": runs, "i": 3, "stage": "", "finished": True,
                      "error": None, "stop": dc.threading.Event()}
        dlg._poll()
        assert dlg.table.rowCount() == 3 and "◀ open" in dlg.table.item(1, 1).text()
        assert dlg.table.item(0, 0).checkState() == dc.Qt.CheckState.Unchecked   # same prep, fewer DS
        assert "2 of those with both" in dlg.status.text()
        for i in range(dlg.frame_combo.count()):
            dlg.frame_combo.setCurrentIndex(i)
            dlg._fill_table()
            dlg.redraw()
        assert dlg.frame_combo.currentData() == "disc"
        assert dlg.table.item(2, 3).text() == ""              # no disc frame for the unaligned run
        pooled_ax = dlg.fig.axes[0]
        assert "ventral (to disc)" in [t.get_text() for t in pooled_ax.get_xticklabels()]
        dlg.dorsal_box.setChecked(False)                      # the assumption can be dropped
        assert "to disc" in [t.get_text() for t in dlg.fig.axes[0].get_xticklabels()]
        assert dlg.fig.axes[0].get_title().startswith("Pooled: ")
        dlg.table.item(0, 0).setCheckState(dc.Qt.CheckState.Checked)
        assert dlg.pooled["A/ks/data001"]
    finally:
        w.data_manager = None
        w.close()
        w.deleteLater()


def test_array_menu_needs_an_open_run(qtbot, monkeypatch):
    from qtpy.QtWidgets import QMessageBox
    from src.gui.main_window import MainWindow
    said = []
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: said.append(a[2])))
    w = MainWindow()
    try:
        w.ds_compare_action.trigger()
        assert said and "Open a run first" in said[0]
    finally:
        w.close()
        w.deleteLater()


def test_a_turn_needs_a_good_pairing(monkeypatch, tmp_path):
    rng = np.random.default_rng(3)
    somas = {v: tuple(rng.uniform(-900, 900, 2)) for v in range(1, 201)}
    m = np.array([[0, 1], [-1, 0]], float)                   # the rig's quarter turn
    good = {v: tuple(m @ np.array(p) / 30 + 20 + rng.normal(0, 1.5, 2)) for v, p in somas.items()}
    monkeypatch.setattr(dp, "rf_centres", lambda folder, dataset: good)
    turn, r2 = dp.turn_from(tmp_path / "a", "a", somas)
    assert turn == (0, 1, -1, 0) and r2 > 0.9
    shuffled = dict(zip(good, rng.permutation(list(good.values()))))   # another sort's IDs
    monkeypatch.setattr(dp, "rf_centres", lambda folder, dataset: shuffled)
    turn, r2 = dp.turn_from(tmp_path / "b", "b", somas)
    assert turn is None and r2 < dp.TURN_MIN_R2
