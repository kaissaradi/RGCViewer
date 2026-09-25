"""The Waveforms-tab PCA reads real spikes only (2026-09-25 diagnosis, PLAN.md Q56)."""

import numpy as np
import pandas as pd

from src.analysis import analysis_core
from src.analysis.data_manager import DataManager


def _dm(raw=True):
    """3 cells on a line of 6 electrodes 30 µm apart; cell 2's template channel is 3."""
    rng = np.random.default_rng(0)
    n_samples, n_ch = 20000, 6
    data = rng.normal(0, 2, (n_samples, n_ch)).astype(np.float32)
    times = {0: np.arange(200, 19000, 400), 1: np.arange(350, 19000, 500), 2: np.arange(500, 19000, 700)}
    shape = -40 * np.exp(-((np.arange(80) - 20) / 4.0) ** 2)
    amp = {0: (1.0, 0.5), 1: (0.3, 1.0), 2: (1.5, 0.8)}          # (ch 0, ch 1) gain per cell
    for c, ts in times.items():
        for t in ts:
            data[t - 20:t + 60, 0] += amp[c][0] * shape
            data[t - 20:t + 60, 1] += amp[c][1] * shape
    st = np.concatenate([times[c] for c in (0, 1, 2)])
    owner = np.concatenate([np.full(len(times[c]), c) for c in (0, 1, 2)])
    dm = DataManager.__new__(DataManager)
    dm.spike_times = st
    dm.cluster_df = pd.DataFrame({"cluster_id": [0, 1, 2], "best_chan": [0, 1, 3]})
    dm.channel_positions = np.column_stack([np.arange(n_ch) * 30.0, np.zeros(n_ch)])
    dm.get_cluster_spike_indices = lambda cid: np.flatnonzero(owner == cid)
    dm.raw_reader = None
    dm.raw_data_memmap = data if raw else None
    dm.uV_per_bit = 1.0
    dm.sampling_rate = 20000.0
    return dm


def test_channel_snippets_keep_only_the_asked_channels_and_drop_edges():
    data = np.arange(100 * 4, dtype=np.float32).reshape(100, 4)
    out, kept = analysis_core.extract_channel_snippets(data, [5, 50, 99], [2, 0], window=(-2, 3))
    assert out.shape == (3, 2, 5) and kept.tolist() == [True, True, False]
    np.testing.assert_array_equal(out[1, 0], data[48:53, 2])
    out, kept = analysis_core.extract_channel_snippets(data, [5, 50], [0], window=(-2, 3),
                                                       cancelled=lambda: True)
    assert not kept.any()


def test_pca_reads_the_cell_even_off_its_template_channel_and_neighbours_by_distance():
    dm = _dm()
    assert dm.pca_channels(0) == [0, 1, 2, 3]
    assert dm.pca_neighbours(0, 0) == [1]                 # 30 µm away; cell 2 sits at 90 µm
    r = dm.get_channel_all_snippets(0, 2)                  # cell 2's template channel is 3
    assert r["source"] == "raw" and len(r["unit_waves"]) == len(dm.get_cluster_spike_indices(2))
    assert r["unit_waves"].shape[1] == 4 * 80              # 4 channels × 80 samples, µV
    assert sorted(r["bg_waves_by_cid"]) == [0, 1]
    trough = r["unit_waves"][:, 20].mean()                 # channel 0 at the spike time
    assert trough < -40                                    # real amplitude, not z-scored


def test_no_raw_file_gives_no_made_up_waveforms():
    r = _dm(raw=False).get_channel_all_snippets(0, 0)
    assert r["source"] == "none" and len(r["unit_waves"]) == 0 and len(r["bg_waves"]) == 0


def test_pca_ellipse_is_turned_to_its_major_axis(qtbot):
    from src.gui.panels import waveforms_panel as wp
    rng = np.random.default_rng(1)
    x = rng.normal(0, 3, 500)
    pts = np.column_stack([x, x + rng.normal(0, 0.3, 500)])   # along the 45° diagonal
    e = wp._compute_ellipse(pts)
    item = wp._ellipse_item(e, wp.pg.mkPen("r"))
    assert abs(item.rotation() - 45) < 5 or abs(item.rotation() + 135) < 5
    assert abs(item.pos().x() - e["cx"]) < 1e-9


def test_blocks_are_merged_and_a_sparse_cell_is_topped_up_by_seeks():
    dm = _dm()
    dm.sampling_rate = 400.0             # 1 s blocks = 400 samples: few spikes per block
    dm.PCA_BLOCKS = 2
    r = dm.get_channel_all_snippets(0, 2)
    n_all = len(dm.get_cluster_spike_indices(2))
    assert len(r["unit_waves"]) == min(n_all, dm.PCA_MIN_UNIT)   # topped up to the minimum
    assert len(set(r["unit_indices"].tolist())) == len(r["unit_indices"])   # no spike twice


def test_pair_dprime_is_the_separation_in_standard_deviations():
    from src.gui.panels import waveforms_panel as wp
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, (4000, 3))
    b = rng.normal(0, 1, (4000, 3)) + np.array([4.0, 0.0, 0.0])
    assert abs(wp._pair_dprime(a, b) - 4.0) < 0.15
    assert wp._pair_dprime(a, a + 0) == 0.0
