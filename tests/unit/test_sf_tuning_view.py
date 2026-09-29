"""Spatial-frequency protocols get their own Grating-tab view (PLAN.md Q65)."""

import numpy as np

from src.analysis import grating_calc as gc
from src.gui.panels import sf_tuning_view as sv


def _sf_run():
    """One direction, three periods, two TFs, 3 repeats; the cell prefers period 25."""
    rng = np.random.default_rng(0)
    params, trials = [], []
    for _rep in range(3):
        for sp in (6.0, 25.0, 400.0):
            for tf in (1.0, 2.0):
                params.append({"barWidth": sp, "temporalFrequency": tf, "orientation": 0.0,
                               "preTime": 500.0, "stimTime": 2000.0, "tailTime": 500.0})
                n = 40 if sp == 25.0 else 5
                phase = rng.uniform(0, 1.0 / tf, n)
                t = 500 + (np.floor(rng.uniform(0, 2 * tf, n)) / tf + 0 * phase) * 1000 + rng.normal(50, 10, n)
                trials.append(np.sort(np.clip(t, 500, 2499)))
    return {7: trials}, params


def test_curves_rasters_and_the_grating_tab_view(qtbot):
    by_cell, params = _sf_run()
    data = gc.compute_grating_response(7, by_cell, params, n_shuffles=20)
    curves = sv.sf_curves(data)
    assert sorted(curves) == [1.0, 2.0]
    assert [p[0] for p in curves[2.0]] == [6.0, 25.0, 400.0]
    best = max(curves[2.0], key=lambda p: p[1])
    assert best[0] == 25.0
    rows, keys, timing = sv.raster_rows(by_cell[7], params)
    assert [k[0] for k in keys] == sorted(k[0] for k in keys)          # grouped by period
    assert timing == {"pre_s": 0.5, "stim_s": 2.0, "tail_s": 0.5}
    assert min(float(r.min()) for r in rows if r.size) >= 0.0          # spikes from onset

    from types import SimpleNamespace
    from src.gui.main_window import MainWindow
    w = MainWindow()
    try:
        w.data_manager = SimpleNamespace(
            grating_available=True, grating_status="raw_only", grating_source=None,
            grating_spatial_label=("period", "px"),
            grating_raw_data={"spike_times_by_trial": by_cell, "trial_parameters": params},
            get_grating_data_for_cluster=lambda cid: data, get_borrowed_grating=lambda cid: None,
            vision_neurons_source=lambda: None)
        panel = w.grating_panel
        panel.update_all(7)
        assert panel.sf_view.isVisibleTo(panel) and not panel.plots_box.isVisibleTo(panel)
        assert "25px" in panel.sf_view.stats.text()
        assert "SPATIAL TUNING" in panel.title_label.text()
    finally:
        w.data_manager = None
        w.close()
        w.deleteLater()
