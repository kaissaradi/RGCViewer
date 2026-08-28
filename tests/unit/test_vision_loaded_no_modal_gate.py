"""A dataset is revealed once, whole, and never behind a dialog.

Two separate regressions live here.

**The dialog.** ``_on_vision_loaded`` used to raise an "STA file does not match
this sort" warning with ``QMessageBox.warning``, which runs a nested event loop
and does not return until the user clicks OK. The physics warm-up was kicked off
after it, so on any run with partial STA coverage the warm-up never started:
``_physics_done_count`` stayed 0, ``update_cache_progress`` never saw ``ready``,
and ``save_standard_plot_cache`` was never reached — so ``standard_plot_cache``,
``feature_cache`` and ``ei_corr_dict`` were never written and *every* open of
that dataset was a cold one. The dialog is gone entirely: partial overlap
between the .sta id set and the sort is the ordinary case, not a stale file.

**The order.** Opening a run is three loads — Kilosort, Vision, and the stimulus
analyses — and the last two land whenever they land. The UI used to unlock as
soon as Kilosort finished, and Vision and stimulus each rebuilt the cluster
table on arrival. Locally all three finish inside a second and nobody notices.
On the lab's CIFS mount Vision takes ~8 s and the stimulus load ~4 s, so the
table rebuilt itself twice, seconds apart, while the user was already clicking
around in it — and each rebuild installs a fresh proxy, which drops the
selection. These tests pin the fix: nothing is revealed until every phase has
reported in, and then the table is built exactly once.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import src.gui.callbacks as callbacks


def _main_window(sta_consistent=True, n_clusters=3):
    """A MagicMock main window mid-load, with both secondary loads pending."""
    mw = MagicMock()
    dm = mw.data_manager
    dm.cluster_df = pd.DataFrame({"cluster_id": list(range(n_clusters))})
    dm.vision_stas = {1: object()}
    dm.vision_eis = {1: object()}
    dm.vision_available = True
    dm.sta_ids_consistent = sta_consistent
    dm.sta_consistency_message = "53 of 731 cells in the .sta have no unit"
    dm.attach_sta_quality_column.return_value = True
    dm.attach_chirp_qi_column.return_value = True
    dm.attach_chirp_onoff_column.return_value = True
    dm.standard_plot_cache = {}
    dm._physics_done_count = 0
    mw._get_selected_cluster_id.return_value = None
    mw._expect_physics = True
    mw._dataset_revealed = False
    mw._vision_load_succeeded = False
    # Explicit: on a MagicMock an unset flag reads back as a truthy auto-attribute,
    # which would send _on_vision_loaded down the network-deferral path and start
    # a real QThread against a mock worker.
    mw._deferred_stimulus_load = False
    mw._load_phases_pending = {"vision", "stimulus"}
    return mw


def _run_load(mw, vision_success=True):
    """Finish both secondary loads and return the patched collaborators."""
    with patch.object(callbacks, "start_physics_warmup") as warmup, \
            patch.object(callbacks, "start_worker") as worker, \
            patch.object(callbacks, "start_cache_progress_polling"), \
            patch.object(callbacks, "invalidate_population_caches"), \
            patch.object(callbacks, "on_cluster_selection_changed"), \
            patch.object(callbacks, "QMessageBox") as msgbox:
        callbacks._on_vision_loaded(mw, vision_success, "ok", False)
        vision_only = {
            "revealed": mw._dataset_revealed,
            "enabled": mw.central_widget.setEnabled.called,
            "rebuilds": mw.refresh_table_model.call_count,
        }
        callbacks._on_stimulus_analyses_loaded(mw, True, "ok")
        return warmup, worker, msgbox, vision_only


@pytest.mark.parametrize("sta_consistent", [False, True])
def test_no_dialog_on_vision_loaded(sta_consistent):
    """Partial STA coverage is normal — nothing may raise a box over it."""
    mw = _main_window(sta_consistent)
    _, _, msgbox, _ = _run_load(mw)

    assert not msgbox.warning.called
    assert not msgbox.critical.called
    assert not msgbox.called


@pytest.mark.parametrize("sta_consistent", [False, True])
def test_physics_warmup_starts_regardless_of_sta_coverage(sta_consistent):
    """The warm-up runs either way — it is what fills the on-disk cache."""
    mw = _main_window(sta_consistent)
    warmup, _, _, _ = _run_load(mw)

    warmup.assert_called_once_with(mw)
    mw.data_manager.precompute_ei_correlations_background.assert_called_once()


def test_nothing_is_revealed_until_every_phase_reports():
    """Vision finishing first must not unlock a window the stimuli have not reached."""
    mw = _main_window()
    _, worker, _, vision_only = _run_load(mw)

    # State captured after Vision but before the stimulus load.
    assert vision_only["revealed"] is False
    assert vision_only["enabled"] is False
    assert vision_only["rebuilds"] == 0

    # ...and only once the last phase lands.
    assert mw._dataset_revealed is True
    mw.central_widget.setEnabled.assert_called_once_with(True)
    worker.assert_called_once_with(mw)


def test_table_is_built_exactly_once():
    """Every derived column is attached before the single rebuild."""
    mw = _main_window()
    _run_load(mw)

    assert mw.refresh_table_model.call_count == 1
    mw.data_manager.attach_sta_quality_column.assert_called_once()
    mw.data_manager.attach_chirp_qi_column.assert_called_once()
    mw.data_manager.attach_chirp_onoff_column.assert_called_once()
    mw.sta_panel.show.assert_called_once()


def test_failed_stimulus_load_still_reveals():
    """A phase that fails has to release its claim, or the window stays locked."""
    mw = _main_window()
    with patch.object(callbacks, "start_physics_warmup"), \
            patch.object(callbacks, "start_worker"), \
            patch.object(callbacks, "start_cache_progress_polling"), \
            patch.object(callbacks, "invalidate_population_caches"), \
            patch.object(callbacks, "on_cluster_selection_changed"), \
            patch.object(callbacks, "QMessageBox"):
        callbacks._on_vision_loaded(mw, True, "ok", False)
        callbacks._on_stimulus_analyses_loaded(mw, False, "no chirp file")

    assert mw._dataset_revealed is True
    mw.central_widget.setEnabled.assert_called_once_with(True)


def test_failed_vision_load_still_reveals_without_warmup():
    """Vision failing leaves a usable Kilosort-only window, not a locked one."""
    mw = _main_window()
    warmup, _, _, _ = _run_load(mw, vision_success=False)

    assert mw._dataset_revealed is True
    mw.central_widget.setEnabled.assert_called_once_with(True)
    assert not warmup.called


class TestNetworkStimulusDeferral:
    """On a network mount the stimulus load waits for Vision to finish.

    The two need ~5.4 s of link time on the lab's CIFS share but took 11.2 s
    when overlapped — one saturated 1 GbE link, and interleaved seeks cost more
    than the concurrency returns. Serialising them cut a server open from
    19.5 s to 12.8 s. Local disk keeps the overlap.
    """

    def test_deferred_worker_starts_only_after_vision(self):
        mw = _main_window()
        mw._deferred_stimulus_load = True
        mw._load_phases_pending = {"vision", "stimulus"}

        with patch.object(callbacks, "_start_stimulus_analysis_load") as launch, \
                patch.object(callbacks, "start_physics_warmup"), \
                patch.object(callbacks, "start_worker"), \
                patch.object(callbacks, "start_cache_progress_polling"), \
                patch.object(callbacks, "invalidate_population_caches"), \
                patch.object(callbacks, "on_cluster_selection_changed"), \
                patch.object(callbacks, "QMessageBox"):
            callbacks._on_vision_loaded(mw, True, "ok", False)

        launch.assert_called_once_with(mw, phase_already_claimed=True)
        assert mw._deferred_stimulus_load is False
        # Its phase was claimed up front, so the barrier has not fired.
        assert mw._dataset_revealed is False

    def test_claimed_phase_is_released_when_there_is_nothing_to_load(self):
        """A claim made on assumption must not outlive the assumption."""
        mw = _main_window()
        mw.data_manager._analysis_candidates = {}
        mw._load_phases_pending = {"stimulus"}

        with patch.object(callbacks, "start_physics_warmup"), \
                patch.object(callbacks, "start_worker"), \
                patch.object(callbacks, "start_cache_progress_polling"), \
                patch.object(callbacks, "on_cluster_selection_changed"):
            callbacks._start_stimulus_analysis_load(mw, phase_already_claimed=True)

        assert mw._load_phases_pending == set()
        assert mw._dataset_revealed is True
