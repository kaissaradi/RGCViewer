"""A cell with no chirp here shows its matched reference cell's chirp (PLAN.md Q37)."""

import numpy as np

from src.analysis.data_manager import DataManager
from src.analysis.reference_bridge import ReferenceBridge


def _bridge():
    from src.analysis.chirp_calc import timing_from
    chirp = {"quality_index": np.array([0.1, 0.62]), "bin_size_ms": 50.0,
             "phase_step_on": (0, 20), "phase_step_off": (20, 40), "phase_freq_sweep": (60, 140),
             "phase_contrast": (150, 200), "freq_min_hz": 0.5, "freq_max_hz": 8.0}
    n_bins = int(np.ceil(timing_from(None, chirp).total * 1000.0 / 50.0))   # a real file's length
    chirp["psth_mean"] = np.vstack([np.zeros(n_bins), np.sin(np.linspace(0, 20, n_bins)) + 1.0])
    rng = np.random.default_rng(0)
    rate = 0.5 * (np.sin(np.linspace(0, 20, n_bins)) + 1.0)          # a reproducible response
    chirp["spikes_binned"] = np.stack([np.zeros((10, n_bins)), rng.poisson(rate, (10, n_bins))])
    chirp["quality_index"] = np.array([250.0, 1900.0])             # the file's own scale
    # current Vision 11 (cluster 10) matches reference Vision 21 (cluster 20, chirp row 1)
    return ReferenceBridge({11: 21}, ref_run_path="/x/20260529A/kilosort25/data014",
                           confidence_scores={11: 0.93}, match_statuses={11: "high"},
                           ref_chirp_data=chirp, ref_chirp_id_to_row={20: 1})


def _dm(bridge):
    dm = DataManager.__new__(DataManager)
    dm.reference_bridge = bridge
    dm.is_vision_only = False
    return dm


def test_borrowed_chirp_is_the_matched_cells_row():
    dm = _dm(_bridge())
    d = dm.get_borrowed_chirp(10)
    assert d is not None and 0 < d["quality_index"] <= 1                # rescored, not 1900
    assert d["trials"]["counts"].shape[0] == 10
    assert d["psth_mean"].max() > 1.5 and d["psth_mean"].min() >= 0
    assert d["borrowed"]["reference_id"] == 21 and d["borrowed"]["confidence"] == 0.93
    assert d["bin_size_ms"] == 50.0 and d["phase_contrast"] == (150, 200)


def test_no_borrowing_without_a_match_or_a_bridge():
    assert _dm(_bridge()).get_borrowed_chirp(99) is None
    assert _dm(None).get_borrowed_chirp(10) is None


def test_chirp_tab_says_where_the_trace_comes_from(qtbot):
    from src.gui.main_window import MainWindow
    dm = _dm(_bridge())
    dm.get_chirp_data_for_cluster = lambda cid: None
    dm.chirp_timing = lambda: None
    dm.get_chirp_trials_for_cluster = lambda cid: None
    dm.chirp_qi_details = lambda cid: None
    w = MainWindow()
    try:
        w.data_manager = dm
        panel = w.chirp_panel
        panel.update_all(10)
        assert panel.stack.currentIndex() == 0                    # drawn, not the empty page
        assert "Borrowed from data014" in panel.borrow_note.text()
        assert "0.93" in panel.borrow_note.text()
    finally:
        w.data_manager = None
        w.close()
        w.deleteLater()


# ---------------------------------------------------------------- grating (Q52)

def _raw_grating_bridge():
    """A reference run with raw grating trials: Vision 21 (cluster 20) prefers label 90°."""
    rng = np.random.default_rng(1)
    tp, trials = [], []
    for rep_ in range(3):
        for ori in range(0, 360, 45):
            tp.append({"orientation": float(ori), "barWidth": 100.0, "temporalFrequency": 2.0,
                       "preTime": 500.0, "stimTime": 2000.0, "tailTime": 500.0})
            rate = 5 + 40 * max(0.0, np.cos(np.radians(ori - 90)))     # spikes/s in the window
            n = rng.poisson(rate * 2.0)
            # phase-locked at 2 Hz so the F1 sees the drive
            t = 500 + (np.floor(rng.uniform(0, 4, n)) * 500 + rng.normal(100, 30, n)).clip(0, 1999)
            trials.append(np.sort(t))
    raw = {"spike_times_by_trial": {20: trials}, "trial_parameters": tp, "source": "data023_GratingDSOS.npy"}
    return ReferenceBridge({11: 21}, ref_run_path="/x/20260220A/kilosort25/data023",
                           confidence_scores={11: 0.91}, match_statuses={11: "high"},
                           ref_grating_raw=raw)


def test_borrowed_grating_is_scored_from_the_matched_cells_trials():
    bridge = _raw_grating_bridge()
    dm = _dm(bridge)
    assert bridge.has_any_grating() and bridge.has_grating(11)
    got = dm.get_borrowed_grating(10)                   # current cluster 10 = Vision 11
    assert got is not None and got["borrowed"]["reference_id"] == 21
    from src.analysis import grating_calc as gc
    sel = gc.select_best_dsos_condition(got["data"])
    assert sel["classification"] == "DS"
    assert abs(((sel["preferred_direction_deg"] - 90 + 180) % 360) - 180) < 25
    trials, params = got["trials"]
    assert len(trials) == len(params) == 24
    assert bridge.get_grating_entry(11) is got["data"]    # scored once, then kept
    assert dm.get_borrowed_grating(99) is None and _dm(None).get_borrowed_grating(10) is None


def test_grating_tab_shows_the_borrowed_cell_and_its_rasters(qtbot):
    from src.gui.main_window import MainWindow
    dm = _dm(_raw_grating_bridge())
    dm.grating_available, dm.grating_status = False, "missing"
    w = MainWindow()
    try:
        w.data_manager = dm
        panel = w.grating_panel
        panel.update_all(10)
        assert panel.stack.currentIndex() == 0                    # drawn, not the empty page
        assert "Borrowed from data023" in panel.borrow_note.text() and "0.91" in panel.borrow_note.text()
        panel.update_all(99)                                      # no match: the empty page again
        assert panel.stack.currentIndex() == 1 and not panel.borrow_note.isVisibleTo(panel)
    finally:
        w.data_manager = None
        w.close()
        w.deleteLater()


# ---------------------------------------------------------------- population and table (Q58)

def _dm_with_cells(bridge):
    import pandas as pd
    dm = _dm(bridge)
    dm.grating_available = False                      # a white-noise run: no grating file
    dm.cluster_df = pd.DataFrame({"cluster_id": [10, 11], "n_spikes": [100, 100]})
    return dm


def test_borrowed_grating_is_shown_only_once_scored_then_everywhere():
    bridge = _raw_grating_bridge()
    dm = _dm_with_cells(bridge)
    assert dm.grating_entry_for_display(10) == (None, None)       # not scored: nothing computed here
    assert bridge.precompute_gratings() == 1
    entry, source = dm.grating_entry_for_display(10)
    assert source == "borrowed" and entry is bridge.get_grating_entry(11)
    assert dm.grating_entry_for_display(11) == (None, None)       # cluster 11 has no match

    from types import SimpleNamespace
    from src.gui.panels import population_panel as pp
    mw = SimpleNamespace(data_manager=dm, dsos_threshold=None)
    rows = list(pp._iter_dsos_population(mw, None))
    assert [cid for cid, _sel in rows] == [10] and rows[0][1]["classification"] == "DS"

    assert dm.attach_grating_columns() is True                    # used to return False here
    row = dm.cluster_df.set_index("cluster_id").loc[10]
    assert row["dsos"] == "DS" and row["dsi"] > 0.3


def test_the_summary_says_what_was_borrowed_and_where():
    from src.gui import callbacks
    bridge = _raw_grating_bridge()
    dm = _dm_with_cells(bridge)
    bridge.precompute_gratings()
    c = callbacks.borrow_counts(dm, bridge)
    assert (c["cells"], c["matched"], c["high"], c["grating"], c["ds"]) == (2, 1, 1, 1, 1)
    text = callbacks.borrow_summary(c, "data023")
    assert "Matched 1 of 2 cells to data023" in text
    assert "Drifting gratings: 1 cells" in text and "1 DS" in text and "arrows on the population" in text
    assert "Chirp: none" in text
