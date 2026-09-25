"""A new selection must redraw the active tab even while FeatureWorker runs.

With a raw file attached, an uncached cell sends ``_process_selection`` down
the FeatureWorker path. ``on_features_ready`` redraws only the EI, Waveforms
and Standard tabs, so the STA, Grating, Chirp and Contrast tabs used to stay
on the previous cell until the user switched tabs. On 20251212A/data018 with
the raw .bin loaded, 12 of 12 fresh STA selections failed to draw.
"""

from types import SimpleNamespace

import pytest

from src.gui import main_window as mw_module
from src.gui.main_window import MainWindow


class _FakeSignal:
    def connect(self, *_args, **_kwargs):
        pass


class _FakeThread:
    def __init__(self):
        self.started = _FakeSignal()

    def start(self):
        pass


class _FakeWorker:
    def __init__(self, *_args):
        self.features_ready = _FakeSignal()
        self.error = _FakeSignal()
        self.run = None

    def moveToThread(self, _thread):
        pass


class _Waveforms:
    def __init__(self):
        self.reading = []

    def show_reading(self, cluster_id):
        self.reading.append(cluster_id)


class _Host:
    """The slice of MainWindow that _process_selection touches."""

    _process_selection = MainWindow._process_selection
    _panels_needing_features = MainWindow._panels_needing_features
    _start_feature_worker = MainWindow._start_feature_worker
    _feature_thread_alive = MainWindow._feature_thread_alive
    _feature_worker_done = MainWindow._feature_worker_done

    def __init__(self, current_tab_name):
        self.ei_panel = object()
        self.waveforms_panel = _Waveforms()
        self.standard_plots_panel = object()
        self.sta_panel = object()
        self.grating_panel = object()
        current = getattr(self, current_tab_name)
        self.analysis_tabs = SimpleNamespace(currentWidget=lambda: current)
        self.status_bar = SimpleNamespace(showMessage=lambda *a, **k: None)
        self.data_manager = SimpleNamespace(
            get_lightweight_features=lambda cid: None,  # uncached
            dat_path="/raw/data018",                    # raw file attached
        )
        self._pending_cluster_id = 7
        self.drawn = []

    def _cleanup_thread(self, _attr):
        pass

    def on_features_ready(self, cluster_id, features):
        pass

    def _draw_plots(self, cluster_id, features):
        self.drawn.append((cluster_id, features))


@pytest.fixture(autouse=True)
def _no_real_threads(monkeypatch):
    monkeypatch.setattr(mw_module, "QThread", _FakeThread)
    monkeypatch.setattr(mw_module, "FeatureWorker", _FakeWorker)


@pytest.mark.parametrize("tab", ["sta_panel", "grating_panel"])
def test_tabs_without_features_draw_immediately(tab):
    host = _Host(tab)
    host._process_selection()
    assert host.drawn == [(7, None)]


@pytest.mark.parametrize("tab", ["ei_panel", "waveforms_panel", "standard_plots_panel"])
def test_feature_tabs_wait_for_the_worker(tab):
    host = _Host(tab)
    host._process_selection()
    # on_features_ready draws these; drawing now would paint them twice.
    assert host.drawn == []
    # The Waveforms tab clears to "reading…" at once instead of keeping the last cell.
    assert host.waveforms_panel.reading == ([7] if tab == "waveforms_panel" else [])


def test_debounced_selection_after_the_run_closed_does_nothing():
    host = _Host("sta_panel")
    host.data_manager = None                 # the run closed before the timer fired
    host._process_selection()
    assert host.drawn == []
