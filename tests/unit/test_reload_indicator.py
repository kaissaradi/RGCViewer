"""The load indicator on a dataset switch mid-warm-up (PLAN.md Q19).

The old run's progress poll kept running into the next load. Once the new
run's Kilosort files were in, it saw a warm cache, hid "Loading dataset..."
and announced "Physics Cache Ready" before the new run was on screen.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication, QProgressBar

from src.gui import callbacks


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


def _mw(revealed):
    bar = QProgressBar()
    bar.setRange(0, 0)                       # the busy "Loading dataset..." state
    bar.setFormat("Loading dataset...")
    bar.show()
    dm = SimpleNamespace(cluster_df=pd.DataFrame({"cluster_id": [0, 1]}),
                         standard_plot_cache={0: {}, 1: {}}, _physics_done_count=2,
                         vision_stas=None, save_standard_plot_cache=MagicMock())
    return SimpleNamespace(data_manager=dm, cache_progress=bar, _dataset_revealed=revealed,
                           _expect_physics=False, _cache_save_triggered=False,
                           status_bar=MagicMock())


def test_progress_poll_leaves_the_load_indicator_alone_during_a_load(qapp):
    mw = _mw(revealed=False)
    callbacks.update_cache_progress(mw)
    assert mw.cache_progress.isVisible()
    assert (mw.cache_progress.minimum(), mw.cache_progress.maximum()) == (0, 0)
    mw.data_manager.save_standard_plot_cache.assert_not_called()
    mw.status_bar.showMessage.assert_not_called()


def test_after_reveal_a_warm_cache_hides_the_bar_once(qapp):
    mw = _mw(revealed=True)
    callbacks.update_cache_progress(mw)
    assert not mw.cache_progress.isVisible()
    mw.data_manager.save_standard_plot_cache.assert_called_once()


def test_releasing_the_old_run_stops_its_progress_poll(qapp, monkeypatch):
    monkeypatch.setattr(callbacks, "stop_worker", lambda mw: None)
    monkeypatch.setattr(callbacks, "_retire_inflight_load", lambda mw: False)
    timer = QTimer()
    timer.start(250)
    mw = SimpleNamespace(_cache_progress_timer=timer, data_manager=None)
    callbacks._release_previous_dataset(mw)
    assert not timer.isActive()
