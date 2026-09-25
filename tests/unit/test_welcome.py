"""The welcome screen and the recent-runs list (PLAN.md Q49)."""

import pytest
from qtpy.QtCore import QSettings

from src.gui import recent_paths


@pytest.fixture
def settings(tmp_path, monkeypatch):
    s = QSettings(str(tmp_path / "t.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(recent_paths, "_settings", lambda: s)
    return s


def test_recent_runs_newest_first_unique_capped_and_existing(tmp_path, settings):
    dirs = []
    for i in range(8):
        d = tmp_path / f"run{i}" / "ksfiles"
        d.mkdir(parents=True)
        dirs.append(d)
        recent_paths.remember_recent(d)
    recent_paths.remember_recent(dirs[3])                      # reopened: moves to the top
    runs = recent_paths.recent_runs()
    assert runs[0] == str(dirs[3]) and len(runs) == recent_paths.RECENT_MAX
    assert len(set(runs)) == len(runs)
    dirs[7].rmdir()                                            # gone from disk: not offered
    assert str(dirs[7]) not in recent_paths.recent_runs()


def test_window_opens_on_the_welcome_page_and_leaves_it_for_a_load(qtbot, settings, tmp_path):
    from src.gui.main_window import MainWindow
    run = tmp_path / "20260220A" / "kilosort25" / "data022" / "ksfiles"
    run.mkdir(parents=True)
    recent_paths.remember_recent(run)
    w = MainWindow()
    try:
        assert w.central_stack.currentWidget() is w.welcome_panel
        assert [b.text() for b in w.welcome_panel.recent_buttons] == ["20260220A / kilosort25 / data022"]
        w.show_analysis_view()
        assert w.central_stack.currentWidget() is w.central_widget
    finally:
        w.data_manager = None
        w.close()
        w.deleteLater()
