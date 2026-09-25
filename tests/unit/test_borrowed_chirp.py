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
