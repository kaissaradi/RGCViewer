"""Lisp stimulus sequence files → grating trials (PLAN.md Q61).

The rules under test are MATLAB load_stim.m's (gdfield/matlab_base): header
skipped, trial starts at interval changes > 2 SD, an early-ended run keeps
whole repeats, spikes in [start, start + duration).
"""
from pathlib import Path

import numpy as np
import pytest

from src.analysis import lisp_stimulus as ls

FS = 20000.0
SEQ = """(:TYPE :DRIFTING-SINUSOID :RGB #(0.48 0.48 0.48) :X-START 0 :X-END 640 :FRAMES 960) \
(:SPATIAL-PERIOD 64 :TEMPORAL-PERIOD 64 :DIRECTION 0) \
(:SPATIAL-PERIOD 64 :TEMPORAL-PERIOD 64 :DIRECTION 180) \
(:SPATIAL-PERIOD 64 :TEMPORAL-PERIOD 64 :DIRECTION 180) \
(:SPATIAL-PERIOD 64 :TEMPORAL-PERIOD 64 :DIRECTION 0) """


def _seq(tmp_path, text=SEQ, name="s02"):
    p = tmp_path / name
    p.write_text(text)
    return ls.read_sequence(p)


def _triggers(n_trials, per_trial=10, period_s=12.0, step_s=100 / 120.0, t0=0.01):
    """``per_trial`` triggers every 100 frames, trials ``period_s`` apart (samples)."""
    out = [t0 + k * period_s + j * step_s for k in range(n_trials) for j in range(per_trial)]
    return np.round(np.asarray(out) * FS)


def test_read_forms_vectors_comments_nesting():
    forms = ls.read_forms("; a comment\n(:A 1 :B #(1 2.5) (:C :D)) (:E -3)")
    assert forms == [[":A", 1, ":B", (1, 2.5), [":C", ":D"]], [":E", -3]]


def test_read_forms_unbalanced():
    with pytest.raises(ls.LispStimulusError):
        ls.read_forms("(:A 1")
    with pytest.raises(ls.LispStimulusError):
        ls.read_forms(":A 1)")


def test_read_sequence_header_trials_combinations(tmp_path):
    seq = _seq(tmp_path)
    assert seq.stim_type == "DRIFTING-SINUSOID"
    assert seq.header["FRAMES"] == 960 and seq.header["RGB"] == (0.48, 0.48, 0.48)
    assert len(seq.trials) == 4
    assert seq.trials[1] == {"SPATIAL_PERIOD": 64, "TEMPORAL_PERIOD": 64, "DIRECTION": 180}
    assert len(seq.combinations) == 2 and seq.trial_list == [0, 1, 1, 0]
    assert seq.repetitions == 2


def test_script_file_is_refused_with_a_hint(tmp_path):
    with pytest.raises(ls.LispStimulusError, match="script"):
        _seq(tmp_path, "(let ((rgb (mul 0.48 #(1 1 1)))) (run-stimulus *display*))", "stimuli.lisp")


def test_trial_starts_are_first_trigger_after_each_gap():
    trig = _triggers(5) / FS
    starts = ls.trial_starts(trig)
    np.testing.assert_allclose(starts, trig[::10])


def test_build_counts_match_the_matlab_window(tmp_path):
    seq = _seq(tmp_path)
    trig = _triggers(4)
    rng = np.random.default_rng(0)
    spikes = np.sort(rng.uniform(0, 4 * 12.0, 3000)) * FS
    g = ls.build_grating_trials(seq, trig, {7: spikes}, FS)
    starts = trig[::10] / FS
    for i, t0 in enumerate(starts):
        p = g.trial_parameters[i]
        x = g.spike_times_by_trial[7][i]
        mine = np.sum((x >= p["preTime"]) & (x < p["preTime"] + p["stimTime"]))
        s = spikes / FS
        matlab = np.sum((s >= t0) & (s < t0 + 8.0))
        assert mine == matlab
    p0 = g.trial_parameters[0]
    assert p0["stimTime"] == pytest.approx(8000.0)
    assert p0["temporalFrequency"] == pytest.approx(120.0 / 64)
    assert p0["barWidth"] == 64 and p0["orientation"] == 0
    # The first trial starts 10 ms into the recording: its baseline is 10 ms.
    assert p0["preTime"] == pytest.approx(10.0, abs=0.1)
    assert g.trial_parameters[1]["preTime"] == pytest.approx(1000.0)


def test_early_end_keeps_whole_repeats(tmp_path):
    seq = _seq(tmp_path)                       # 2 conditions × 2 repeats
    trig = np.r_[_triggers(3)]                 # only 3 trials started
    g = ls.build_grating_trials(seq, trig, {1: np.zeros(0)}, FS)
    assert len(g.trial_parameters) == 2
    assert g.summary["repetitions"] == 1 and g.summary["n_trial_starts"] == 3
    assert "ended early" in ls.describe(g.summary)


def test_more_starts_than_trials_is_the_wrong_file(tmp_path):
    seq = _seq(tmp_path)
    with pytest.raises(ls.LispStimulusError, match="does not describe"):
        ls.build_grating_trials(seq, _triggers(6), {1: np.zeros(0)}, FS)


def test_one_trigger_per_trial(tmp_path):
    seq = _seq(tmp_path)
    trig = _triggers(4, per_trial=1)
    g = ls.build_grating_trials(seq, trig, {1: np.zeros(0)}, FS)
    assert len(g.trial_parameters) == 4


def test_non_grating_type_is_refused(tmp_path):
    seq = _seq(tmp_path, "(:TYPE :PULSE :FRAMES 360) (:RGB #(1 1 1)) (:RGB #(-1 -1 -1))")
    with pytest.raises(ls.LispStimulusError, match="drifting gratings"):
        ls.build_grating_trials(seq, _triggers(2), {1: np.zeros(0)}, FS)


def test_sequence_file_names_and_search(tmp_path):
    assert ls.sequence_file_names("data002")[:2] == ["s02", "s02.txt"]
    assert ls.sequence_file_names("data000-3600-7200s")[0] == "s00"
    assert ls.sequence_file_names("chunk1") == []
    run = tmp_path / "2012-10-15-0" / "data002" / "data002-map"
    run.mkdir(parents=True)
    stim = tmp_path / "2012-10-15-0" / "stimuli"
    stim.mkdir()
    (stim / "s02").write_text(SEQ)
    (stim / "s03").write_text(SEQ)
    assert ls.find_sequence_files(run, "data002") == [stim / "s02"]
    assert ls.stimulus_dirs(run) == [stim]


def test_datamanager_reads_through_the_neurons_file(tmp_path, monkeypatch):
    """Triggers and spikes both from .neurons; Vision IDs map to cluster IDs (Law 1)."""
    import pandas as pd
    import src.analysis.visionloader as vl
    from src.analysis.data_manager import DataManager

    (tmp_path / "s02").write_text(SEQ)
    trig = _triggers(4)

    class FakeReader:
        sample_freq = FS

        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_TTL_times(self):
            return trig

        def get_spike_sample_nums_for_all_real_neurons(self):
            return {5: np.array([FS * 1.0]), 9: np.array([FS * 13.0])}

    monkeypatch.setattr(vl, "NeuronsReader", FakeReader)
    dm = DataManager.__new__(DataManager)
    dm.cluster_df = pd.DataFrame({"cluster_id": [4, 8]})
    dm.is_vision_only = False
    dm._vision_neurons_source = (str(tmp_path), "data002")
    g = dm.read_lisp_grating(tmp_path / "s02")
    assert sorted(g.spike_times_by_trial) == [4, 8]          # Vision 5, 9 → Kilosort 4, 8
    dm.apply_lisp_grating(g)
    assert dm.grating_status == "raw_only" and dm.grating_available
    assert dm.grating_spatial_label == ("period", "px")
    assert dm.grating_conditions == [(64, 120.0 / 64)]
    assert dm.grating_source["n_trials"] == 4


def test_needs_vision_neurons(tmp_path):
    import pandas as pd
    from src.analysis.data_manager import DataManager
    dm = DataManager.__new__(DataManager)
    dm.cluster_df = pd.DataFrame({"cluster_id": [1]})
    with pytest.raises(ls.LispStimulusError, match="Vision"):
        dm.read_lisp_grating(Path(tmp_path) / "s02")
