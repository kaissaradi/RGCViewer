"""The Raster tab draws one cell's spikes and finds its neighbours (PLAN.md Q63)."""

import numpy as np
import pandas as pd



def _dm(triggers=None):
    """A stand-in with what the Raster tab reads (Kilosort-like IDs, no Vision files)."""
    from types import SimpleNamespace
    rng = np.random.default_rng(0)
    fs = 20000.0
    spikes = {c: np.sort(rng.uniform(0, 600 * fs, 3000)).astype(np.int64) for c in range(4)}
    spikes[1] = np.sort(np.r_[spikes[0][::2] + 4, spikes[1][:500]])   # shares half of cell 0's spikes
    df = pd.DataFrame({"cluster_id": [0, 1, 2, 3], "n_spikes": [3000] * 4,
                       "x_um": [0.0, 20.0, 60.0, 900.0], "y_um": [0.0] * 4})
    dm = SimpleNamespace(
        cluster_df=df, sampling_rate=fs, n_samples=int(600 * fs), generation=1,
        spike_times=np.sort(np.concatenate(list(spikes.values()))),
        get_cluster_spikes=lambda c: spikes[int(c)], get_vision_id_for_cluster=lambda c: c + 1,
        vision_eis=None, kilosort_dir=None, grating_source=None, grating_raw_data=None,
        raw_reader=None, raw_data_memmap=None, stimulus_manifest=None)
    dm.trigger_times_s = lambda: None if triggers is None else np.asarray(triggers) / fs
    return dm


def test_raster_tab_folds_the_recording_and_flags_a_duplicate(qtbot):
    from src.gui.main_window import MainWindow
    w = MainWindow()
    try:
        w.data_manager = _dm()
        p = w.raster_panel
        p.update_all(0)
        assert p._trials is None and p.rows_combo.currentText() == "Auto"
        assert p._row_order.size == 120                          # 600 s in rows of 5 s
        qtbot.waitUntil(lambda: bool(p._neighbours), timeout=5000)
        assert [n.cluster_id for n in p._neighbours] == [1, 2]    # cell 3 is 900 µm away
        assert "possible duplicate" in p.table.item(0, 3).text()
        assert p._pair_stats[1]["shared"] > 0.45
    finally:
        w.data_manager = None
        w.close()
        w.deleteLater()


def test_raster_tab_rows_are_trials_when_the_triggers_say_so(qtbot):
    from src.gui.main_window import MainWindow
    fs = 20000.0
    trig = np.concatenate([(k * 12.0 + np.arange(10) * 0.833) * fs for k in range(50)]).astype(np.int64)
    w = MainWindow()
    try:
        dm = _dm(trig)
        w.data_manager = dm
        p = w.raster_panel
        p.update_all(0)
        assert p._trials is not None and p.rows_combo.currentData() == "trials"
        assert p._row_order.size == 50 and abs(p._row_span - 12.0) < 1e-6
    finally:
        w.data_manager = None
        w.close()
        w.deleteLater()
