"""File ▸ Map Reference Run lists this experiment's runs (PLAN.md Q52)."""

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


def test_runs_of_the_prep_with_what_they_hold(tmp_path):
    prep = _prep(tmp_path)
    runs = rp.list_runs(prep, prep / "kilosort25" / "data022")
    assert [(r.sorter, r.name) for r in runs] == [
        ("kilosort25", "data022"), ("kilosort25", "data023"), ("kilosort25", "data024"),
        ("kilosort40", "data023")]
    assert runs[0].is_current and runs[0].stimuli == ["white noise"]
    assert runs[1].stimuli == ["grating"] and runs[2].stimuli == ["chirp"]
    # The open run has no grating: the grating run is preselected; with one, the chirp run.
    assert rp.preferred_index(runs, []) == 1
    assert rp.preferred_index(runs, ["grating"]) == 2
    assert rp.preferred_index(runs, ["grating", "chirp", "contrast"]) == 1   # any other run


def test_picker_returns_the_selected_run_and_never_the_open_one(qtbot, tmp_path):
    prep = _prep(tmp_path)
    runs = rp.list_runs(prep, prep / "kilosort25" / "data022")
    dlg = rp.RunPicker(None, runs, prep.name, rp.preferred_index(runs, []))
    qtbot.addWidget(dlg)
    dlg._accept_selected()
    assert dlg.chosen == str(prep / "kilosort25" / "data023")
    dlg.chosen = None
    dlg.table.selectRow(0)                                           # the open run
    dlg._accept_selected()
    assert dlg.chosen is None
