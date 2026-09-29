"""File ▸ Match Runs lists this experiment's runs (PLAN.md Q52, Q62)."""

from qtpy.QtCore import Qt

from src.gui.panels import run_picker as rp


def _prep(tmp_path):
    prep = tmp_path / "20260220A"
    layout = {
        ("kilosort25", "data022"): ["data022.ei", "data022.sta", "data022.params"],
        ("kilosort25", "data023"): ["data023.ei", "data023_GratingDSOS.npy"],
        ("kilosort25", "data024"): ["data024.ei", "data024_Chirp.npy"],
        ("kilosort25", "data025"): ["notes.txt"],                     # no .ei: not listed
        ("kilosort40", "data023"): ["data023.ei", "data023_GratingDSOS.npy"],
    }
    for (sorter, run), files in layout.items():
        d = prep / sorter / run
        d.mkdir(parents=True)
        for f in files:
            (d / f).write_bytes(b"")
    return prep


def _old_layout(tmp_path):
    """<prep>/<run>/[<run>-map]/…, gratings in <prep>/stimuli/sNN."""
    prep = tmp_path / "2012-10-15-0"
    for rel, files in {
        "data000": ["data000.ei", "data000.sta"],
        "data002/data002-map": ["data002.ei", "data002.params"],
        "data002/data000-map": ["data000-map.ei", "data000-map.sta"],
        "Yass/data000": ["data000.ei", "data000.sta"],
    }.items():
        d = prep / rel
        d.mkdir(parents=True)
        for f in files:
            (d / f).write_bytes(b"")
    (prep / "stimuli").mkdir()
    (prep / "stimuli" / "s02").write_text("(:TYPE :DRIFTING-SINUSOID :FRAMES 960)")
    return prep


def test_runs_of_the_prep_with_what_they_hold(tmp_path):
    prep = _prep(tmp_path)
    runs = rp.list_runs(prep, prep / "kilosort25" / "data022")
    assert [(r.sorter, r.name) for r in runs] == [
        ("kilosort25", "data022"), ("kilosort25", "data023"), ("kilosort25", "data024"),
        ("kilosort40", "data023")]
    assert runs[0].is_current and runs[0].stimuli == ["white noise"]
    assert runs[1].stimuli == ["grating"] and runs[2].stimuli == ["chirp"]
    # The open run has no grating: the grating run is ticked; with one, the chirp run.
    assert rp.preferred_indices(runs, ["white noise"]) == [1, 2]
    assert rp.preferred_indices(runs, ["white noise", "grating"]) == [2]
    assert rp.preferred_index(runs, ["white noise", "grating", "chirp"]) == 1   # any other run


def test_old_layout_runs_and_lisp_gratings(tmp_path):
    prep = _old_layout(tmp_path)
    cur = prep / "data002" / "data002-map"
    assert rp.prep_dir_for(cur) == prep
    assert rp.prep_dir_for(prep / "data000") == prep            # a run right under the prep
    runs = rp.list_runs(prep, cur)
    by_label = {r.label: r for r in runs}
    assert set(by_label) == {"data000", "data002/data002-map", "data002/data000-map", "Yass/data000"}
    assert by_label["data002/data002-map"].is_current
    assert by_label["data002/data002-map"].stimuli == ["grating (Lisp s02)"]
    assert by_label["data002/data000-map"].dataset == "data000-map"
    # White noise from the folder next to the open run (the same sort) first.
    ticked = [runs[i].label for i in rp.preferred_indices(runs, ["grating"])]
    assert ticked == ["data002/data000-map"]


def test_picker_returns_the_ticked_runs_never_the_open_one(qtbot, tmp_path):
    prep = _prep(tmp_path)
    runs = rp.list_runs(prep, prep / "kilosort25" / "data022")
    dlg = rp.RunPicker(None, runs, prep.name, rp.preferred_indices(runs, ["white noise"]))
    qtbot.addWidget(dlg)
    assert dlg.match_btn.text() == "Match 2 runs"
    dlg._accept_checked()
    assert dlg.chosen == [str(prep / "kilosort25" / "data023"), str(prep / "kilosort25" / "data024")]
    # The open run has no checkbox.
    assert not (dlg.table.item(0, 0).flags() & Qt.ItemFlag.ItemIsUserCheckable)
    dlg.set_checked([])
    assert not dlg.match_btn.isEnabled()
