"""Stimulus blocks inside a sort, and the stability verdict (PLAN.md Q41)."""

from types import SimpleNamespace

import numpy as np

from src.analysis import recording_timeline as rt


class Manifest:
    def __init__(self, runs):
        self.runs = runs

    def get(self, run):
        return self.runs.get(run)


def _m():
    info = lambda proto, n: SimpleNamespace(protocol=proto, n_samples=n)  # noqa: E731
    return Manifest({"data007": info("manookinlab.protocols.ChirpStimulus", 6_040_000),
                     "data008": info("manookinlab.protocols.ContrastResponseGrating", 10_160_000),
                     "data009": info("fieldlab.protocols.GratingDSOS_ks", 36_080_000),
                     "data010": info("manookinlab.protocols.SpatialNoise", 37_280_000)})


def test_folder_names():
    assert rt.runs_in_folder("data007-010") == ["data007", "data008", "data009", "data010"]
    assert rt.runs_in_folder("data000_data002_data004") == ["data000", "data002", "data004"]
    assert rt.runs_in_folder("data022") == ["data022"]
    assert rt.runs_in_folder("ksfiles") == [] and rt.runs_in_folder("data018-") == []


def test_blocks_of_the_real_20260715A_data007_010():
    """Last spike of that sort: sample 89,559,992 (checked 2026-09-25)."""
    blocks = rt.stimulus_blocks("data007-010", _m(), 89_559_992, 20000.0)
    assert [b.protocol for b in blocks] == ["Chirp", "Contrast", "Gratings", "Noise"]
    assert blocks[0].start_s == 0 and np.isclose(blocks[-1].end_s, 89_560_000 / 20000)


def test_blocks_are_refused_when_lengths_do_not_add_up():
    assert rt.stimulus_blocks("data007-010", _m(), 99_000_000, 20000.0) == []   # longer than the runs
    assert rt.stimulus_blocks("data007-010", _m(), 40_000_000, 20000.0) == []   # ends before the last run
    assert rt.stimulus_blocks("data007-011", _m(), 89_559_992, 20000.0) == []   # a run with no entry


def test_stability_names_the_block_where_the_cell_went_quiet():
    blocks = rt.stimulus_blocks("data007-010", _m(), 89_559_992, 20000.0)
    t = np.arange(0, 4478, 10.0)
    rate = np.full(t.size, 20.0)
    grating = (t >= blocks[2].start_s) & (t < blocks[2].end_s)
    rate[grating] = 2.0
    amp = np.linspace(100, 50, t.size)
    st = rt.stability(t, rate, amp, blocks)
    assert "fires little during Gratings" in st.verdict
    assert "amplitude falls" in st.verdict
    assert rt.stability(t, np.full(t.size, 20.0), np.full(t.size, 80.0), blocks).verdict == "stable"
