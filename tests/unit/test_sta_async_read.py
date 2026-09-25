"""A cold STA read runs off the GUI thread (PLAN.md Q5)."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication

from src.analysis.vision_integration import LazySTADict


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


def test_cached_never_reads():
    d = LazySTADict.__new__(LazySTADict)
    d._cache_lock = threading.Lock()
    d._cache = {7: "sta"}
    assert d.cached(7) == "sta" and d.cached(8) is None and d.cached("x") is None


class _SlowStas:
    """vision_stas whose reads take a while and record their thread."""

    def __init__(self, delay=0.3):
        self.delay, self.read_threads, self._cache = delay, [], {}

    def __contains__(self, vid):
        return True

    def cached(self, vid):
        return self._cache.get(vid)

    def __getitem__(self, vid):
        self.read_threads.append(threading.current_thread())
        time.sleep(self.delay)
        self._cache[vid] = SimpleNamespace(red=None)   # "read failed" is enough here
        return self._cache[vid]


def _panel(qapp, stas):
    from src.gui.panels.sta_panel import STAPanel
    dm = SimpleNamespace(vision_stas=stas, get_vision_id_for_cluster=lambda c: c + 1,
                         vision_sort_check=lambda: None)
    selected = {"cid": 4}
    mw = SimpleNamespace(data_manager=dm, get_current_colors=lambda: {},
                         _get_selected_cluster_id=lambda: selected["cid"],
                         current_sta_view="rf")
    return STAPanel(mw), selected


def _pump(pred, secs=5.0):
    end = time.time() + secs
    while time.time() < end and not pred():
        QApplication.processEvents()
        time.sleep(0.01)


def test_update_view_returns_at_once_and_reads_in_the_background(qapp):
    stas = _SlowStas()
    panel, _ = _panel(qapp, stas)
    shown = []
    panel._show_sta = lambda cid, vid, dm, sta: shown.append((cid, vid))
    t0 = time.time()
    panel.update_view(4)
    assert time.time() - t0 < 0.2                      # did not wait for the read
    _pump(lambda: shown)
    assert shown == [(4, 5)]
    assert threading.main_thread() not in stas.read_threads


def test_only_the_cell_the_user_stops_on_is_read(qapp):
    stas = _SlowStas(delay=0.3)
    panel, selected = _panel(qapp, stas)
    shown = []
    panel._show_sta = lambda cid, vid, dm, sta: shown.append(cid)
    for cid in (4, 5, 6, 7):                          # scroll while the first read runs
        selected["cid"] = cid
        panel.update_view(cid)
    _pump(lambda: shown, secs=5)
    _pump(lambda: False, secs=0.8)
    assert shown == [7]
    assert len(stas.read_threads) == 2                # the first, then the last one wanted
