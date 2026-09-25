"""Nothing from the previous run is drawn on the next one (PLAN.md Q13).

Cluster IDs repeat between runs. Caches that live outside the DataManager
and key by cluster ID alone drew the previous run's cell 5 on this run's
cell 5: the Waveforms PCA cache, the EI map cache, the array photo.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication, QPushButton, QSlider

from src.analysis.data_manager import DataManager


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


def test_every_datamanager_gets_a_new_generation(tmp_path):
    a = DataManager(str(tmp_path))
    b = DataManager(str(tmp_path))
    assert isinstance(a.generation, int) and b.generation > a.generation


def test_waveform_results_from_the_previous_run_are_stale():
    from src.gui.panels.waveforms_panel import WaveformPanel
    host = SimpleNamespace(_current_cluster_id=5,
                           main_window=SimpleNamespace(data_manager=SimpleNamespace(generation=2)))
    host._generation = lambda: WaveformPanel._generation(host)
    stale = lambda cid, payload: WaveformPanel._is_stale(host, cid, payload)  # noqa: E731
    assert stale(5, {"_generation": 1})          # same cell ID, previous run
    assert not stale(5, {"_generation": 2})
    assert stale(6, {"_generation": 2})          # another cell
    assert not stale(5, {})                      # untagged payload: current run


def test_ei_panel_forgets_the_previous_runs_photo_and_maps(qapp):
    from src.gui.panels.ei_panel import EIPanel
    btn = QPushButton()
    btn.setCheckable(True)
    btn.setChecked(True)
    host = SimpleNamespace(_ei_map_cache={5: np.zeros(3)}, _ei_map_cache_key=(1, "x"),
                           _overlay_image_rgba=np.zeros((2, 2, 4)),
                           _overlay_extent_um=(0, 1, 0, 1), _overlay_enabled=True,
                           photo_btn=btn, overlay_alpha_slider=QSlider())
    EIPanel.reset_for_new_dataset(host)
    assert host._ei_map_cache == {} and host._ei_map_cache_key is None
    assert host._overlay_image_rgba is None and host._overlay_extent_um is None
    assert host._overlay_enabled is False and not btn.isChecked()


def test_a_new_load_clears_the_view_caches(qapp):
    from src.gui import callbacks
    from src.gui.panels import waveforms_panel
    waveforms_panel._PCA_CACHE[(1, 5, 7)] = {"x": 1}
    ei_panel = MagicMock()
    callbacks._forget_previous_dataset_views(SimpleNamespace(ei_panel=ei_panel))
    assert not waveforms_panel._PCA_CACHE
    ei_panel.reset_for_new_dataset.assert_called_once()
