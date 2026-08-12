from __future__ import annotations
import os
import threading
from pathlib import Path
from qtpy.QtWidgets import QFileDialog, QMessageBox, QApplication, QStyle
from qtpy.QtCore import QThread, Qt, QTimer, QEventLoop
from qtpy.QtGui import QStandardItem, QColor

from ..analysis.data_manager import DataManager
from .workers.workers import (
    RefinementWorker,
    SpatialWorker,
    StandardPlotsWorker,
    KilosortLoadWorker,
    VisionLoadWorker,
    StimulusAnalysisLoadWorker,
)
from .widgets.widgets import HighlightStatusPandasModel
from .panels.population_panel import (
    draw_population_timecourse_panel,
    invalidate_population_caches,
)
from .panels.feature_extraction import FeatureExtractionWindow
from . import recent_paths
from typing import TYPE_CHECKING
import logging

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from gui.main_window import MainWindow


def _show_load_progress(main_window, msg):
    """Paint a load status line now. Safe to call from a worker via BlockingQueuedConnection."""
    bar = getattr(main_window, "status_bar", None)
    if bar is not None:
        bar.showMessage(msg)
    QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)


def _ensure_cache_progress_timer(main_window):
    """Main-thread timer: physics warm-up no longer emits finished_cluster."""
    timer = getattr(main_window, "_cache_progress_timer", None)
    if timer is None:
        timer = QTimer(main_window)
        timer.setInterval(250)
        timer.timeout.connect(lambda: update_cache_progress(main_window))
        main_window._cache_progress_timer = timer
    return timer


def start_cache_progress_polling(main_window):
    """Poll cache counters on the UI thread while the progress bar is visible."""
    _ensure_cache_progress_timer(main_window).start()


def stop_cache_progress_polling(main_window):
    timer = getattr(main_window, "_cache_progress_timer", None)
    if timer is not None:
        timer.stop()


def update_cache_progress(main_window):
    """Updates the main window progress bar based on actual cache state."""
    if not main_window.data_manager or main_window.data_manager.cluster_df.empty:
        return

    dm = main_window.data_manager
    total = len(dm.cluster_df)
    if total <= 0:
        return

    physics_done = getattr(dm, "_physics_done_count", 0)
    std_done = len(dm.standard_plot_cache)
    vision_loaded = bool(getattr(dm, "vision_stas", None))

    # StandardPlotsWorker drives finished_cluster; physics warm-up does not.
    if vision_loaded:
        val = int(physics_done / total * 100)
        ready = physics_done >= total
    else:
        val = int(std_done / total * 100)
        ready = std_done >= total

    main_window.cache_progress.setValue(min(val, 100))

    if ready and not getattr(main_window, "_cache_save_triggered", False):
        # Marks the warm-up milestone only. closeEvent saves again on exit, so
        # work done after this point is no longer lost.
        main_window._cache_save_triggered = True
        stop_cache_progress_polling(main_window)
        main_window.cache_progress.hide()
        main_window.status_bar.showMessage(
            "Physics Cache Ready: UMAP and Population panels optimized.", 8000
        )
        dm.save_standard_plot_cache()


def _retire_inflight_load(main_window):
    """Detach a running dataset load so a new one can start safely.

    The Kilosort worker's ``run()`` is one long blocking call, so it cannot be
    interrupted and waiting for it would freeze the UI for the length of a
    load. Instead its signals are disconnected — its results would otherwise
    land on the DataManager the new load is about to replace — and both the
    thread and the worker are parked until it finishes on its own.

    **Both** references matter, and this is the actual crash. ``moveToThread``
    does not give the thread a Python reference to its worker, so
    ``main_window.ks_load_worker`` is the only one. The old code reassigned
    that attribute for the new load, which garbage-collected a worker whose
    ``run()`` was still executing on another thread — a segfault, not an
    exception, which is why the app vanished with no traceback.

    Returns the threads it parked, so the caller can wait for them before it
    touches anything the retired load still owns.
    """
    parked = []
    retired = getattr(main_window, "_retired_load_threads", None)
    if retired is None:
        retired = main_window._retired_load_threads = []

    # Physics warm is a plain threading.Thread; flip its stop event so
    # ensure_physics_cache returns and does not emit physics_warm_done
    # against the dataset we are about to replace.
    stop = getattr(main_window, "_physics_warm_stop", None)
    if stop is not None:
        stop.set()

    for thread_attr, worker_attr in (
        ("ks_load_thread", "ks_load_worker"),
        ("vision_load_thread", "vision_load_worker"),
        ("analysis_load_thread", "analysis_load_worker"),
        ("_grating_batch_thread", "_grating_batch_worker"),
        ("feature_worker_thread", "feature_worker"),
    ):
        thread = getattr(main_window, thread_attr, None)
        worker = getattr(main_window, worker_attr, None)
        try:
            running = thread is not None and thread.isRunning()
        except RuntimeError:
            running = False  # wrapper already deleted; nothing to retire
        if not running:
            continue

        logger.info("Retiring an in-flight load before starting a new one (%s)",
                    thread_attr)
        if worker is not None:
            if hasattr(worker, "stop"):
                try:
                    worker.stop()
                except RuntimeError:
                    pass
            for signal_name in ("finished", "progress", "error"):
                signal = getattr(worker, signal_name, None)
                if signal is None:
                    continue
                try:
                    signal.disconnect()
                except (TypeError, RuntimeError):
                    pass  # nothing connected, or already gone

        # Park the pair together and only release it once the thread reports
        # it has actually finished.
        entry = (thread, worker)
        retired.append(entry)

        def _drop(e=entry):
            if e in retired:
                retired.remove(e)
            try:
                e[0].deleteLater()
            except RuntimeError:
                pass

        try:
            thread.finished.connect(_drop)
            thread.quit()  # takes effect once run() returns
            parked.append(thread)
        except RuntimeError:
            pass
        setattr(main_window, thread_attr, None)
        setattr(main_window, worker_attr, None)

    return parked


def _release_previous_dataset(main_window):
    """Free the dataset that the next load replaces.

    Assigning a new DataManager over ``main_window.data_manager`` does not
    free the old one — the panels and the worker threads still reference it —
    so without this both datasets are resident at once. The Vision EI table
    alone is more than 500 MB, and that peak is what makes the app run out of
    memory when a user moves between runs during a session.

    Stopping the workers first is also correct on its own: a spatial or
    standard-plots worker left running against the old dataset goes on
    emitting results keyed by the old run's cluster IDs, which the panels
    would apply to the new run. Cell IDs do not carry between runs, so those
    results are meaningless.

    A load that is still in flight owns the old DataManager from another
    thread, so it cannot be released here. ``_retire_inflight_load`` parks
    that thread and reports it, and the release then waits for it to finish.
    """
    stop_worker(main_window)
    parked = _retire_inflight_load(main_window)

    old_dm = getattr(main_window, "data_manager", None)
    if old_dm is None or not hasattr(old_dm, "close"):
        return

    if not parked:
        old_dm.close()
        return

    # Still running. Release it once every parked thread has returned from
    # run(); until then the load worker is free to keep writing to it.
    #
    # ``released`` guards the case where a thread reports finished more than
    # once: the emptied list would otherwise read as "all done" a second time.
    pending = list(parked)
    released = []

    def _release_when_idle(thread):
        if thread in pending:
            pending.remove(thread)
        if pending or released:
            return
        released.append(True)
        old_dm.close()

    for thread in parked:
        try:
            thread.finished.connect(lambda t=thread: _release_when_idle(t))
        except RuntimeError:
            _release_when_idle(thread)  # wrapper gone; the thread is done


def load_directory(main_window, kilosort_dir=None, dat_file=None):
    """Triggers the background loading of a Kilosort directory."""
    if kilosort_dir is None:
        ks_dir_name = QFileDialog.getExistingDirectory(
            main_window,
            "Select Kilosort Output Directory",
            recent_paths.last_dir(main_window, "kilosort"),
        )
    else:
        ks_dir_name = kilosort_dir

    if not ks_dir_name:
        return

    # Remember the *parent*: run folders sit side by side (data006, data007…),
    # so reopening the dialog inside the run you just picked would show only
    # its contents. The parent is where the next run actually lives.
    recent_paths.remember_dir(Path(ks_dir_name).parent, "kilosort")

    # A load already in flight has to be retired before its attributes are
    # overwritten below. Reassigning ks_load_thread while the old one is still
    # running drops the last reference to a live QThread, so Qt destroys it
    # mid-run — "QThread: Destroyed while thread is still running", followed by
    # the process aborting. This was always latent; it only became reachable
    # when the app started auto-loading the remembered dataset at startup,
    # because until then a session only ever performed one load.
    #
    # The same step also stops the background workers and frees the dataset
    # being replaced, so the old run does not stay resident behind the new one.
    _release_previous_dataset(main_window)

    # 1. Lock UI and Prep DataManager
    main_window.central_widget.setEnabled(False)
    main_window.status_bar.showMessage("Initializing loader...")
    main_window.data_manager = DataManager(ks_dir_name, main_window)
    main_window.setWindowTitle(f"RGC Viewer - {ks_dir_name}")

    # 2. Setup Thread and Worker
    main_window.ks_load_thread = QThread()
    main_window.ks_load_worker = KilosortLoadWorker(
        main_window.data_manager, ks_dir_name, dat_file
    )
    main_window.ks_load_worker.moveToThread(main_window.ks_load_thread)

    # 3. Connect Signals
    main_window.ks_load_thread.started.connect(main_window.ks_load_worker.run)
    # Blocking so each "Checking for…" actually paints before the worker
    # continues. Queued progress arrives in a burst after the scan, which
    # is why those messages never flashed.
    main_window.ks_load_worker.progress.connect(
        lambda msg: _show_load_progress(main_window, msg),
        Qt.BlockingQueuedConnection,
    )

    # Connect the finished signal to our new UI-thread cleanup function
    main_window.ks_load_worker.finished.connect(
        lambda success, msg: _on_kilosort_loaded(
            main_window, success, msg, ks_dir_name, dat_file
        )
    )

    # 4. Fire!
    main_window.ks_load_thread.start()


def _on_kilosort_loaded(main_window, success, message, ks_dir_name, dat_file):
    """Callback triggered on the UI thread when KilosortLoadWorker finishes."""
    if getattr(main_window, "ks_load_thread", None):
        main_window.ks_load_thread.quit()
        main_window.ks_load_thread.wait()
        main_window.ks_load_thread = None

    if not success:
        # A remembered dataset that no longer loads must not be reoffered on
        # every launch — that would leave the app unable to start cleanly until
        # the user found the setting.
        if getattr(main_window, "_reopening_remembered", False):
            recent_paths.forget_dataset()
            main_window._reopening_remembered = False
        main_window.status_bar.showMessage("Loading failed.", 5000)
        main_window.central_widget.setEnabled(True)
        # Deferred, NOT called here. This function runs inside the worker's
        # finished-signal emission, and a modal dialog spins a nested event
        # loop that processes the worker's and thread's pending deleteLater
        # while that emission is still on the stack — an access violation, so
        # the app vanishes with no traceback. Handing the dialog to the event
        # loop lets the signal unwind first.
        QTimer.singleShot(
            0, lambda: QMessageBox.critical(main_window, "Loading Error", message)
        )
        return

    # Only a load that actually worked is worth reopening next launch.
    main_window._reopening_remembered = False
    recent_paths.remember_dataset(ks_dir_name, dat_file)

    n_clusters = len(main_window.data_manager.cluster_df)
    # Take "Checking for…" off the bar before tree / UMAP setup. That work
    # used to leave the last check message on screen and look like the
    # check itself was still running.
    main_window.status_bar.showMessage(
        f"Loaded {n_clusters} clusters. Preparing views..."
    )
    QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)

    # --- GUI UPDATES (Safe because we are back on the main thread) ---

    # 1. Raw Tab Management
    if dat_file is not None:
        main_window.analysis_tabs.setTabEnabled(
            main_window.analysis_tabs.indexOf(main_window.raw_panel), True
        )
    else:
        main_window.analysis_tabs.setTabEnabled(
            main_window.analysis_tabs.indexOf(main_window.raw_panel), False
        )

    # 2. Tree/Table Population
    tree_file_path = os.path.join(ks_dir_name, "cluster_group_refined_tree.json")
    if os.path.exists(tree_file_path):
        main_window.data_manager.load_tree_structure(tree_file_path)
    else:
        populate_tree_view(main_window)

    main_window._update_table_view_duplicate_highlight()
    main_window._update_tree_view_duplicate_highlight()

    # 3. Enable UI & Workers
    main_window.cache_progress_count = 0
    main_window.cache_progress.setValue(0)
    main_window.cache_progress.show()
    start_cache_progress_polling(main_window)

    # Start the worker — handles signal wiring AND queueing all clusters internally
    start_worker(main_window)
    main_window.central_widget.setEnabled(True)

    main_window.load_raw_action.setEnabled(True)
    main_window.load_vision_action.setEnabled(True)
    main_window.load_classification_action.setEnabled(True)
    main_window.save_action.setEnabled(True)
    main_window.save_classification_action.setEnabled(True)
    main_window.calibrate_array_action.setEnabled(True)
    if hasattr(main_window, 'map_reference_action'):
        main_window.map_reference_action.setEnabled(True)
    # A cache can only be rebuilt once there is a run to rebuild it for.
    if hasattr(main_window, 'rebuild_cache_action'):
        main_window.rebuild_cache_action.setEnabled(True)

    # 4. Handle Vision specific UI updates
    if main_window.data_manager.vision_stas:
        main_window.sta_panel.show()
    else:
        main_window.sta_panel.hide()

    if (
        hasattr(main_window, "similarity_panel")
        and main_window.data_manager.vision_available
    ):
        main_window.similarity_panel.on_vision_loaded()

    # Drop everything the UMAP panel derived from the *previous* preparation
    # and re-gate its feature checkboxes against this one. Both matter here:
    # runs are sorted independently, so a stale embedding and mosaic describe
    # cells that no longer exist under those IDs, and the panel was constructed
    # back in MainWindow.__init__ when data_manager was still None, so its
    # chirp row came up disabled regardless of the dataset. Chirp/grating
    # files are only globbed here; they load after the UI unlocks, and
    # refresh_feature_availability runs again then.
    if hasattr(main_window, "umap_panel"):
        main_window.umap_panel.reset_for_new_dataset()

    # 5. Array Transform check
    transform_path = (
        main_window.data_manager.kilosort_dir.parent
        / "transforms"
        / "array_transform.json"
    )
    if transform_path.exists() and hasattr(main_window, "standard_plots_panel"):
        main_window.standard_plots_panel.refresh_array_image(str(transform_path))
        cid = main_window._get_selected_cluster_id()
        if cid is not None:
            main_window.standard_plots_panel.update_all(cid)

    # Auto-start Vision loading if the KS worker found .ei files in the directory.
    # This runs as a SEPARATE VisionLoadWorker so the UI is already unlocked
    # and responsive while Vision data streams in the background.
    auto_dir = getattr(main_window.data_manager, "_auto_vision_dir", None)
    auto_dataset = getattr(main_window.data_manager, "_auto_vision_dataset", None)
    if auto_dir and auto_dataset:
        main_window.status_bar.showMessage(
            f"Loaded {n_clusters} clusters. Auto-loading Vision data in background...",
            6000,
        )
        # Reuse the existing load_vision_directory plumbing but bypass the dialog
        # VisionLoadWorker is already imported at the top of this module
        main_window.vision_load_thread = QThread()
        main_window.vision_load_worker = VisionLoadWorker(
            main_window.data_manager, auto_dir
        )
        main_window.vision_load_worker.moveToThread(main_window.vision_load_thread)
        main_window.vision_load_thread.started.connect(
            main_window.vision_load_worker.run
        )
        main_window.vision_load_worker.progress.connect(
            lambda msg: _show_load_progress(main_window, msg),
            Qt.BlockingQueuedConnection,
        )
        main_window.vision_load_worker.finished.connect(
            lambda success, msg, is_partial: _on_vision_loaded(
                main_window, success, msg, is_partial
            )
        )
        main_window.vision_load_thread.start()
    else:
        main_window.status_bar.showMessage(
            f"Successfully loaded {n_clusters} clusters.", 5000
        )

    _start_stimulus_analysis_load(main_window)


def _start_stimulus_analysis_load(main_window):
    """Unpickle chirp/contrast/grating after the cluster table is already up."""
    dm = main_window.data_manager
    cands = getattr(dm, "_analysis_candidates", None) or {}
    if not any(cands.get(k) for k in ("chirp", "contrast", "grating")):
        return

    main_window.analysis_load_thread = QThread()
    main_window.analysis_load_worker = StimulusAnalysisLoadWorker(dm)
    main_window.analysis_load_worker.moveToThread(main_window.analysis_load_thread)
    main_window.analysis_load_thread.started.connect(
        main_window.analysis_load_worker.run
    )
    main_window.analysis_load_worker.progress.connect(
        lambda msg: _show_load_progress(main_window, msg),
        Qt.BlockingQueuedConnection,
    )
    main_window.analysis_load_worker.finished.connect(
        lambda success, msg: _on_stimulus_analyses_loaded(main_window, success, msg)
    )
    main_window.analysis_load_thread.start()


def _on_stimulus_analyses_loaded(main_window, success, message):
    """Attach chirp columns and re-gate UMAP once the .npy files are in RAM."""
    if hasattr(main_window, "analysis_load_thread") and main_window.analysis_load_thread:
        main_window.analysis_load_thread.quit()
        if not main_window.analysis_load_thread.wait(2000):
            logger.warning("Analysis load thread didn't exit cleanly, terminating")
            main_window.analysis_load_thread.terminate()
            main_window.analysis_load_thread.wait(500)
        main_window.analysis_load_thread = None

    if hasattr(main_window, "analysis_load_worker") and main_window.analysis_load_worker:
        main_window.analysis_load_worker.deleteLater()
        main_window.analysis_load_worker = None

    if not success:
        main_window.status_bar.showMessage(
            f"Stimulus analysis load failed: {message}", 5000
        )
        return

    dm = main_window.data_manager
    changed = False
    if dm.attach_chirp_qi_column():
        changed = True
    if dm.attach_chirp_onoff_column():
        changed = True
    if changed and hasattr(main_window, "refresh_table_model"):
        main_window.refresh_table_model()
    if hasattr(main_window, "umap_panel"):
        main_window.umap_panel.refresh_feature_availability()
    if main_window._get_selected_cluster_id() is not None:
        on_cluster_selection_changed(main_window)
    main_window.status_bar.showMessage(message, 4000)


def _on_vision_native_loaded(main_window, success, message, vision_dir_name):
    """Cleanup and GUI update after StandaloneVisionWorker finishes."""

    # --- Safe thread/worker cleanup (mirrors _on_vision_loaded pattern) ---
    if hasattr(main_window, "vision_load_thread") and main_window.vision_load_thread:
        main_window.vision_load_thread.quit()
        if not main_window.vision_load_thread.wait(2000):
            logger.warning("Vision native load thread didn't exit cleanly, terminating")
            main_window.vision_load_thread.terminate()
            main_window.vision_load_thread.wait(500)
        main_window.vision_load_thread = None

    if hasattr(main_window, "vision_load_worker") and main_window.vision_load_worker:
        main_window.vision_load_worker.deleteLater()
        main_window.vision_load_worker = None

    # --- Always re-enable UI, even on failure ---
    main_window.central_widget.setEnabled(True)

    if not success:
        QMessageBox.critical(main_window, "Loading Error", message)
        main_window.status_bar.showMessage("Loading failed.", 5000)
        return

    # --- Disable raw tab: no .bin data in Vision-native mode ---
    main_window.analysis_tabs.setTabEnabled(
        main_window.analysis_tabs.indexOf(main_window.raw_panel), False
    )

    # --- Populate tree and tables ---
    populate_tree_view(main_window)
    main_window._update_table_view_duplicate_highlight()
    main_window._update_tree_view_duplicate_highlight()

    # --- Drop the previous dataset's UMAP results ---
    # Path A of load_vision_directory builds a brand-new DataManager, so this is
    # a different preparation just as much as a Kilosort load is. Same reasoning
    # as in _on_kilosort_loaded. reset_for_new_dataset() re-gates the feature
    # checkboxes itself, which this path was not doing at all before.
    if hasattr(main_window, "umap_panel"):
        main_window.umap_panel.reset_for_new_dataset()

    # --- Enable actions ---
    main_window.load_raw_action.setEnabled(True)
    main_window.save_action.setEnabled(True)
    main_window.save_classification_action.setEnabled(True)
    if hasattr(main_window, 'map_reference_action'):
        main_window.map_reference_action.setEnabled(True)
    # A cache can only be rebuilt once there is a run to rebuild it for.
    if hasattr(main_window, 'rebuild_cache_action'):
        main_window.rebuild_cache_action.setEnabled(True)

    # --- Show STA panel if data is available ---
    if main_window.data_manager.vision_stas:
        main_window.sta_panel.show()

    # --- Start standard plots background worker ---
    start_worker(main_window)

    # --- Kick off EI correlation precompute on background thread ---
    if main_window.data_manager.vision_eis is not None:
        main_window.data_manager.precompute_ei_correlations_background()
        main_window.status_bar.showMessage(
            "Vision loaded. Computing EI correlations in background...", 4000
        )

    if success:
        start_physics_warmup(main_window)


def rebuild_physics_cache(main_window):
    """Delete the persisted physics caches and recompute them from source.

    The safety valve behind the File ▸ Rebuild Physics Cache action: if a run's
    ``feature_cache.pkl`` is stale or corrupt there is otherwise no way out of
    it from inside the app. Deletes the pkls, clears RAM, then re-runs the
    normal warm-up — which writes fresh files at the end, so the cache is
    *overwritten*, not merely dropped.
    """
    dm = main_window.data_manager
    if dm is None or dm.cluster_df is None or dm.cluster_df.empty:
        QMessageBox.information(
            main_window,
            "Rebuild Physics Cache",
            "Load a dataset first — there is no cache to rebuild.",
        )
        return

    reply = QMessageBox.question(
        main_window,
        "Rebuild Physics Cache",
        "Recompute and overwrite the cached physics for this run?\n\n"
        "The cached ISI/ACG, physics and grating DS/OS files are deleted and "
        "rebuilt from source. This takes as long as a first-time load.",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        QMessageBox.StandardButton.Cancel,
    )
    if reply != QMessageBox.StandardButton.Yes:
        return

    removed = dm.rebuild_caches()
    logger.info("Rebuild requested by user; removed %s", removed or "nothing")

    # start_worker() resets _cache_save_triggered and re-queues every cluster
    # (the caches it checks are now empty), so the save fires again at the end.
    start_worker(main_window)
    start_physics_warmup(main_window)
    main_window.status_bar.showMessage("Rebuilding physics cache...", 5000)


def start_physics_warmup(main_window, all_ids=None):
    """Queue the physics warm-up, then chain the grating DS/OS batch after it.

    Called once per dataset load and again by the Rebuild Physics Cache
    action. Both Vision-load paths used to carry byte-identical copies of
    this chain (complete with a leftover ``REPLACE the _warm_physics block``
    editing note); they now share this one, so the rebuild path cannot drift
    from the load path.

    Nothing here computes on the GUI thread: ``_warm_physics`` runs on a
    plain daemon thread and hops back via ``physics_warm_done`` before any
    QThread is started (AGENTS.md Law 2).
    """
    if all_ids is None:
        all_ids = (
            main_window.data_manager.cluster_df["cluster_id"].astype(int).tolist()
        )
    _all_ids = all_ids

    def _start_grating_batch():
        # Batch-precompute grating DSI/OSI for every cluster so the DS/OS
        # population views (probe map, RF-plot markers) reflect the whole
        # dataset immediately rather than only clusters the user happens
        # to have opened individually. This deliberately overrides
        # GratingComputeWorker's documented "only compute on demand"
        # design — an accepted tradeoff (startup time vs. complete
        # population views) made explicitly for this feature, not an
        # oversight of that design note.
        #
        # Runs on a real QThread + GratingBatchWorker (see workers.py),
        # NOT a plain threading.Thread — its progress/finished signals
        # need to safely reach GUI-thread code (status bar text, a
        # matplotlib redraw), and only a real QThread with Qt's
        # signal/slot queuing guarantees that reliably across
        # platforms. A bare Python thread has no Qt event loop, so
        # QTimer.singleShot() from inside one is not reliable (Qt
        # requires timers to be started on the thread they fire on),
        # and calling widget methods (e.g. status_bar.showMessage)
        # directly from a non-GUI thread is unsafe regardless.
        from .workers.workers import GratingBatchWorker

        main_window._grating_batch_thread = QThread()
        main_window._grating_batch_worker = GratingBatchWorker(
            main_window.data_manager, _all_ids
        )
        main_window._grating_batch_worker.moveToThread(
            main_window._grating_batch_thread
        )
        main_window._grating_batch_thread.started.connect(
            main_window._grating_batch_worker.run
        )

        def _on_progress(done, total):
            main_window.status_bar.showMessage(
                f"Computing grating DS/OS tuning... ({done}/{total})"
            )

        def _on_batch_finished(thread=main_window._grating_batch_thread,
                               worker=main_window._grating_batch_worker):
            # A newer load may have retired this pair. Do not quit/delete
            # whatever now sits on the attributes.
            if getattr(main_window, "_grating_batch_thread", None) is not thread:
                return
            main_window.status_bar.showMessage(
                "Grating DS/OS tuning computed for all clusters.", 4000
            )
            # _draw_population_panel_initial (not redraw_population_
            # panels directly) because it also redraws the Population
            # Receptive Fields plot, which now carries the DS/OS
            # markers — redraw_population_panels alone omits that plot.
            if getattr(main_window, "population_view_enabled", False):
                main_window._draw_population_panel_initial()
            thread.quit()
            thread.wait(2000)
            try:
                worker.deleteLater()
                thread.deleteLater()
            except RuntimeError:
                pass
            if getattr(main_window, "_grating_batch_thread", None) is thread:
                main_window._grating_batch_thread = None
                main_window._grating_batch_worker = None

        main_window._grating_batch_worker.progress.connect(_on_progress)
        main_window._grating_batch_worker.finished.connect(_on_batch_finished)
        main_window._grating_batch_thread.start()

    def _on_physics_warm_done():
        # Self-disconnect so repeated Vision loads in one session don't
        # stack duplicate connections — same idiom as
        # _on_standard_plots_done just below.
        try:
            main_window.physics_warm_done.disconnect(_on_physics_warm_done)
        except (RuntimeError, TypeError):
            pass
        if getattr(main_window, "_physics_warm_stop", None) is not None:
            if main_window._physics_warm_stop.is_set():
                return
        _start_grating_batch()

    main_window.physics_warm_done.connect(_on_physics_warm_done)

    warm_stop = threading.Event()
    prev_stop = getattr(main_window, "_physics_warm_stop", None)
    if prev_stop is not None:
        prev_stop.set()
    main_window._physics_warm_stop = warm_stop
    dm_for_warm = main_window.data_manager

    def _warm_physics():
        if warm_stop.is_set():
            return
        dm_for_warm.ensure_physics_cache(
            _all_ids, max_workers=1, stop_event=warm_stop
        )
        if warm_stop.is_set():
            return
        # Chain the grating batch after physics warm-up completes.
        # _warm_physics itself runs on a plain threading.Thread (see
        # below) — that's fine for ensure_physics_cache, which does no
        # Qt/widget work of its own — but starting a QThread must
        # happen from a thread with a Qt event loop for its signals to
        # be delivered correctly, so hop back onto the GUI thread via a
        # Qt-native mechanism (Signal, not QTimer.singleShot) rather
        # than starting the QThread directly from this bare thread.
        main_window.physics_warm_done.emit()

    def _on_standard_plots_done():
        # Disconnect immediately so it only fires once
        try:
            main_window.standard_plots_worker.all_done.disconnect(
                _on_standard_plots_done
            )
        except RuntimeError:
            pass

        # StandardPlotsWorker has finished — feature_cache has all ACGs.
        # Now physics warm-up only needs to do STA seeks, no redundant ACG work.
        main_window.cache_progress_count = 0
        main_window.cache_progress.setValue(0)
        main_window.cache_progress.show()
        start_cache_progress_polling(main_window)
        warm_thread = threading.Thread(
            target=_warm_physics, daemon=True, name="physics-warm"
        )
        main_window._physics_warm_thread = warm_thread
        warm_thread.start()

    main_window.standard_plots_worker.all_done.connect(_on_standard_plots_done)


def load_vision_directory(main_window):
    """Smart router: Loads Vision-only if no KS exists, otherwise appends Vision to KS."""
    vision_dir_name = QFileDialog.getExistingDirectory(
        main_window,
        "Select Vision Analysis Directory",
        recent_paths.last_dir(main_window, "vision"),
    )
    if not vision_dir_name:
        return

    recent_paths.remember_dir(Path(vision_dir_name).parent, "vision")

    main_window.central_widget.setEnabled(False)

    # PATH A: Standalone Vision Load (No Kilosort loaded, or already in Vision-only mode)
    if not main_window.data_manager or getattr(
        main_window.data_manager, "is_vision_only", False
    ):
        main_window.status_bar.showMessage("Initializing Vision-native loader...")

        # Retire any load still in flight, stop the workers, and free the
        # dataset this one replaces — see _release_previous_dataset().
        _release_previous_dataset(main_window)

        # Initialize a fresh DataManager and set the flag immediately
        main_window.data_manager = DataManager(vision_dir_name, main_window)
        main_window.data_manager.is_vision_only = True
        main_window.setWindowTitle(
            f"RGC Viewer - {Path(vision_dir_name).name} (Vision Native)"
        )

        from .workers.workers import StandaloneVisionWorker

        main_window.vision_load_thread = QThread()
        main_window.vision_load_worker = StandaloneVisionWorker(
            main_window.data_manager, vision_dir_name
        )
        main_window.vision_load_worker.moveToThread(main_window.vision_load_thread)

        main_window.vision_load_worker.finished.connect(
            lambda success, msg: _on_vision_native_loaded(
                main_window, success, msg, vision_dir_name
            )
        )

    # PATH B: Append Vision to Kilosort (Standard flow)
    else:
        main_window.status_bar.showMessage(
            "Appending Vision data to Kilosort dataset..."
        )
        main_window.vision_load_thread = QThread()
        main_window.vision_load_worker = VisionLoadWorker(
            main_window.data_manager, vision_dir_name
        )
        main_window.vision_load_worker.moveToThread(main_window.vision_load_thread)

        main_window.vision_load_worker.finished.connect(
            lambda success, msg, is_partial: _on_vision_loaded(
                main_window, success, msg, is_partial
            )
        )

    main_window.vision_load_thread.started.connect(main_window.vision_load_worker.run)
    main_window.vision_load_worker.progress.connect(
        lambda msg: _show_load_progress(main_window, msg),
        Qt.BlockingQueuedConnection,
    )
    main_window.vision_load_thread.start()


def _on_vision_loaded(main_window, success, message, is_partial):
    """Callback triggered on the UI thread when VisionLoadWorker finishes."""
    # Properly cleanup the thread and worker
    if hasattr(main_window, "vision_load_thread") and main_window.vision_load_thread:
        main_window.vision_load_thread.quit()
        if not main_window.vision_load_thread.wait(2000):
            logger.warning("Vision load thread didn't exit cleanly, terminating")
            main_window.vision_load_thread.terminate()
            main_window.vision_load_thread.wait(500)
        main_window.vision_load_thread = None

    if hasattr(main_window, "vision_load_worker") and main_window.vision_load_worker:
        main_window.vision_load_worker.deleteLater()
        main_window.vision_load_worker = None

    main_window.central_widget.setEnabled(True)

    if success and not is_partial:
        invalidate_population_caches()
        main_window.status_bar.showMessage(message, 5000)
    elif is_partial:
        invalidate_population_caches()
        main_window.status_bar.showMessage(
            f"Loaded partial Vision data: {message}", 5000
        )
        QMessageBox.warning(
            main_window,
            "Vision Loading Warning",
            f"Could not load all Vision data, but some files were found.\n{message}",
        )
    else:
        QMessageBox.critical(main_window, "Vision Loading Error", message)
        main_window.status_bar.showMessage("Vision loading failed.", 5000)

    # Show STA panel if data is now available
    if main_window.data_manager.vision_stas:
        main_window.sta_panel.show()
        # Surface STA quality as a sortable column, and check that the .sta
        # actually belongs to this sort — a stale one silently attaches
        # receptive fields to the wrong units.
        if main_window.data_manager.attach_sta_quality_column():
            main_window.refresh_table_model()
        if not main_window.data_manager.sta_ids_consistent:
            QMessageBox.warning(
                main_window,
                "STA file does not match this sort",
                main_window.data_manager.sta_consistency_message
                + "\n\nThe STA panel, the RF mosaic and the temporal STA "
                  "features in the UMAP are all affected. Spike-derived "
                  "analyses (chirp, grating, contrast) are not.",
            )

    # BUG-6 fix: kick off EI correlation precompute on a background thread
    # so it's ready before the user opens the similarity panel.
    # If a pickle cache exists this returns almost instantly; otherwise it
    # runs the heavy computation without ever touching the UI thread.
    if success and main_window.data_manager.vision_eis is not None:
        main_window.data_manager.precompute_ei_correlations_background()
        main_window.status_bar.showMessage(
            "Vision loaded. Computing EI correlations in background...", 4000
        )

    # Now that Vision STA data is loaded, compute get_cell_physics for all
    # clusters. The StandardPlotsWorker deliberately skips this so it can
    # never produce stale timecourse=None entries before Vision is ready.
    if success:
        start_physics_warmup(main_window)

    # Trigger a refresh of the currently selected cluster to show new data
    if main_window._get_selected_cluster_id() is not None:
        on_cluster_selection_changed(main_window)


def redraw_population_panels(main_window: MainWindow, subset=None):
    # draws the middle panel (and optionally clears bottom)
    # If an explicit subset is provided by the caller, use it; otherwise derive it.
    if subset is None:
        subset = main_window._get_pop_subset_ids()

    # Import and call all population plots. draw_population_dsos_plot
    # (the standalone array-space DS/OS probe map) was removed — DS/OS is
    # now drawn directly on the Population Receptive Fields plot via
    # _draw_dsos_markers, called from draw_population_rfs_plot itself.
    from .panels.population_panel import draw_population_acg_panel

    draw_population_timecourse_panel(main_window, subset_ids=subset)
    draw_population_acg_panel(main_window, subset_ids=subset)


def on_cluster_selection_changed(main_window: MainWindow):
    """
    Handles a cluster selection by delegating to the main window controller.

    OPTIMIZED: Direct plotting calls removed.
    main_window.update_cluster_views() now orchestrates:
      - Tier 1: Immediate updates (RFs, Histograms, Cached Data)
      - Tier 2: Debounced updates (Raw Data, Heavy Feature Extraction)
    """
    cluster_id = main_window._get_selected_cluster_id()

    if cluster_id is not None:
        # Single unit selected
        # Delegate to the controller to handle immediate vs delayed updates
        main_window.update_cluster_views(cluster_id)

    else:
        # Group Folder selected (Tree View)
        if main_window.view_stack.currentIndex() == 0:  # Tree View
            selection = main_window.tree_view.selectionModel().selectedIndexes()
            if selection:
                index = selection[0]
                item = main_window.tree_model.itemFromIndex(index)

                # Check if it is a group (groups store None in UserRole)
                if item and item.data(Qt.ItemDataRole.UserRole) is None:
                    main_window._pending_folder_item = item
                    main_window.folder_selection_timer.start()
                    return


def on_spatial_data_ready(main_window: MainWindow, cluster_id: int, _features: dict):
    """Callback for when heavyweight spatial features are ready from the worker."""
    current_id = main_window._get_selected_cluster_id()
    current_tab_widget = main_window.analysis_tabs.currentWidget()
    if cluster_id == current_id and current_tab_widget == main_window.summary_tab:
        # Since draw_summary_EI_plot is not imported and apparently not used,
        # we'll comment out this line for now until it's properly implemented
        # plotting.draw_summary_EI_plot(main_window, cluster_id)
        main_window.status_bar.showMessage("Spatial analysis complete.", 2000)


def on_refine_cluster(main_window: MainWindow):
    """Starts the cluster refinement process in a background thread."""
    cluster_id = main_window._get_selected_cluster_id()
    if cluster_id is None:
        QMessageBox.warning(
            main_window,
            "No Cluster Selected",
            "Please select a cluster from the table to refine.",
        )
        return

    main_window.refine_button.setEnabled(False)
    main_window.status_bar.showMessage(
        f"Starting refinement for Cluster {cluster_id}..."
    )

    main_window.refine_thread = QThread()
    main_window.refinement_worker = RefinementWorker(
        main_window.data_manager, cluster_id
    )
    main_window.refinement_worker.moveToThread(main_window.refine_thread)
    main_window.refinement_worker.finished.connect(
        main_window.handle_refinement_results
    )
    main_window.refinement_worker.error.connect(main_window.handle_refinement_error)
    main_window.refinement_worker.progress.connect(
        lambda msg: main_window.status_bar.showMessage(msg, 3000)
    )
    main_window.refine_thread.started.connect(main_window.refinement_worker.run)
    main_window.refine_thread.start()


def handle_refinement_results(
    main_window: MainWindow, parent_id: int, new_clusters: list[int]
):
    """Handles the results from a successful refinement operation."""
    main_window.status_bar.showMessage(
        f"Refinement of C{parent_id} complete. Found {len(new_clusters)} sub-clusters.",
        5000,
    )
    main_window.data_manager.update_after_refinement(parent_id, new_clusters)
    invalidate_population_caches()
    # Refresh the tree view to show updated cluster information
    populate_tree_view(main_window)
    main_window.refine_button.setEnabled(True)
    main_window.save_action.setEnabled(True)
    if hasattr(main_window, 'map_reference_action'):
        main_window.map_reference_action.setEnabled(True)
    main_window.setWindowTitle("*RGC Viewer (unsaved changes)")
    main_window.refine_thread.quit()
    main_window.refine_thread.wait()


def handle_refinement_error(main_window: MainWindow, error_message: str):
    """Handles an error from the refinement worker."""
    QMessageBox.critical(main_window, "Refinement Error", error_message)
    main_window.status_bar.showMessage("Refinement failed.", 5000)
    main_window.refine_button.setEnabled(True)
    main_window.refine_thread.quit()
    main_window.refine_thread.wait()


# In callbacks.py
def on_save_action(main_window: MainWindow):
    """Handles the save action from the menu or UMAP button."""
    if not main_window.data_manager:
        QMessageBox.warning(
            main_window, "No Data", "No DataManager found. Load data first."
        )
        return

    # Default beside the data it describes; fall back to wherever the user last
    # saved when no dataset path is available.
    if main_window.data_manager.info_path:
        original_path = main_window.data_manager.info_path
        suggested = recent_paths.suggested_path(
            main_window,
            "export",
            f"{original_path.stem}_refined.tsv",
            preferred_dir=original_path.parent,
        )
    else:
        suggested = recent_paths.suggested_path(
            main_window,
            "export",
            "classification_export.tsv",
            preferred_dir=main_window.data_manager.kilosort_dir,
        )

    # This dialog MUST appear if the function is reached
    save_path, _ = QFileDialog.getSaveFileName(
        main_window, "Save Classification Results", suggested, "TSV Files (*.tsv)"
    )

    if save_path:
        recent_paths.remember_dir(save_path, "export")
        # This calls the actual file-writing logic
        save_results(main_window, save_path)


def save_results(main_window, output_path):
    """Saves internal data AND the Vision-compatible text file."""
    try:
        # 1. Save your internal TSV and JSON tree as before
        final_df = main_window.data_manager.cluster_df[["cluster_id", "KSLabel"]].copy()
        final_df.to_csv(output_path, sep="\t", index=False)
        tree_save_path = output_path.replace(".tsv", "_tree.json")
        main_window.data_manager.save_tree_structure(tree_save_path)

        # 2. NEW: Export the Vision-compatible .txt classification
        txt_output_path = output_path.replace(".tsv", ".txt")
        lines_to_write = []

        def recurse_extract(item, current_path):
            for i in range(item.rowCount()):
                child = item.child(i)
                cluster_id = child.data(Qt.ItemDataRole.UserRole)
                if cluster_id is not None:
                    # Format: "VisionID Path/" (Note: VisionID is cluster_id + 1)
                    lines_to_write.append(f"{cluster_id + 1} {current_path}")
                else:
                    if child.text() != "Unclassified":
                        recurse_extract(child, f"{current_path}{child.text()}/")

        recurse_extract(main_window.tree_model.invisibleRootItem(), "")

        with open(txt_output_path, "w") as f:
            f.write("\n".join(lines_to_write))

        main_window.status_bar.showMessage("Saved: .tsv, .json, and Vision .txt", 5000)
    except Exception as e:
        QMessageBox.critical(main_window, "Save Error", f"Could not save files: {e}")


def apply_good_filter(main_window: MainWindow):
    """Filters the tree view to show only 'good' clusters."""
    if main_window.data_manager is None:
        return

    df = main_window.data_manager.cluster_df[
        main_window.data_manager.cluster_df["KSLabel"] == "good"
    ].copy()

    populate_tree_view(main_window, df)


def reset_views(main_window: MainWindow):
    """Resets the views to their original, unfiltered state."""
    if main_window.data_manager is None:
        return
    # Repopulate the tree with original data
    populate_tree_view(main_window)


def start_worker(main_window: MainWindow):
    """Starts the background worker threads (spatial features + standard plots)."""
    main_window._cache_save_triggered = False
    if (
        main_window.worker_thread is not None
        or getattr(main_window, "standard_worker_thread", None) is not None
    ):
        stop_worker(main_window)

    # --- Spatial features worker (existing behaviour) ---
    main_window.worker_thread = QThread()
    main_window.spatial_worker = SpatialWorker(main_window.data_manager)
    main_window.spatial_worker.moveToThread(main_window.worker_thread)
    main_window.worker_thread.started.connect(main_window.spatial_worker.run)
    main_window.spatial_worker.result_ready.connect(main_window.on_spatial_data_ready)
    main_window.worker_thread.start()

    # --- NEW: Standard plots worker for ISI/ACG/FR caching ---
    main_window.standard_worker_thread = QThread()
    main_window.standard_plots_worker = StandardPlotsWorker(main_window.data_manager)
    main_window.standard_plots_worker.moveToThread(main_window.standard_worker_thread)

    # Connect the progress bar signal BEFORE starting or queueing
    main_window.standard_plots_worker.finished_cluster.connect(
        lambda cid: update_cache_progress(main_window)
    )
    # Auto-refresh the Standard Plots panel when the viewed cluster becomes ready
    main_window.standard_plots_worker.finished_cluster.connect(
        main_window.on_standard_plot_ready
    )

    main_window.standard_worker_thread.started.connect(
        main_window.standard_plots_worker.run
    )
    main_window.standard_worker_thread.start()

    # Kick off low-priority background caching for all clusters
    dm = main_window.data_manager
    if dm is not None and not dm.cluster_df.empty:
        main_window.cache_progress_count = 0

        for cid in dm.cluster_df["cluster_id"]:
            cid_int = int(cid)

            # Check if it's already cached. feature_cache presence is not
            # enough — the ACG pass leaves half-built {"acg": ...} rows behind,
            # and treating those as done would skip the cell's physics.
            has_std = cid_int in getattr(dm, "standard_plot_cache", {})
            has_feat = bool(
                getattr(dm, "feature_cache", {}).get(cid_int, {}).get("_computed")
            )

            if not (has_std and has_feat):
                # Only queue cells that actually need math done
                main_window.standard_plots_worker.add_to_queue(cid_int)

        # Update once after queueing/checking all
        update_cache_progress(main_window)


def populate_tree_view(main_window: MainWindow, df=None):
    """
    Builds the tree and table from the cluster data.

    Args:
        main_window: The main window instance.
        df: Optional dataframe to use. If None, uses main_window.data_manager.cluster_df.
    """
    if df is None:
        df = main_window.data_manager.cluster_df

    # --- Populate Table View ---
    main_window.main_cluster_model = HighlightStatusPandasModel(df)
    main_window.setup_table_model(main_window.main_cluster_model)

    # --- Populate Tree View ---
    main_window.tree_view.setUpdatesEnabled(False)
    model = main_window.tree_model
    model.clear()  # Clear any previous data

    # Use a copy for tree construction to ensure we have KSLabel without warning
    df_tree = df.copy()
    if "KSLabel" not in df_tree.columns:
        df_tree["KSLabel"] = "Unknown"

    # Pre-compute string conversions ONCE (O(n) total, not O(n²))
    df_tree["KSLabel_str"] = df_tree["KSLabel"].astype(str)
    unique_labels = sorted(df_tree["KSLabel_str"].unique())

    # Pre-fetch icons and font to avoid redundant Qt calls in the loop
    dir_icon = main_window.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)
    file_icon = main_window.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon)

    # Create top-level nodes for each unique KSLabel
    groups = {}
    group_cell_items = {}  # To store lists of items for batch append

    for label in unique_labels:
        group_item = QStandardItem(label)
        group_item.setEditable(False)
        group_item.setDropEnabled(True)  # Can drop cells into it

        # Style group items differently from cells
        font = group_item.font()
        font.setBold(True)
        group_item.setFont(font)

        # Set different background color for groups
        group_item.setBackground(QColor("#3C3C3C"))
        group_item.setIcon(dir_icon)

        groups[label] = group_item
        group_cell_items[label] = []  # Initialize list for batching

    # Add each cluster as a child item to its group
    # itertuples is significantly faster than iterrows
    for row in df_tree.itertuples():
        cluster_id = row.cluster_id
        label = row.KSLabel_str

        # The text displayed will be e.g., "Cluster 123 (n=456 spikes)"
        n_spikes = getattr(row, "n_spikes", "?")
        item_text = f"Cluster {cluster_id} (n={n_spikes})"
        cell_item = QStandardItem(item_text)
        cell_item.setEditable(False)
        cell_item.setIcon(file_icon)

        # IMPORTANT: Store the actual cluster ID in the item's data role.
        cell_item.setData(cluster_id, Qt.ItemDataRole.UserRole)

        # Prevent dropping items onto cells
        cell_item.setDropEnabled(False)

        if label in group_cell_items:
            group_cell_items[label].append(cell_item)

    # Now add all populated groups to the model in one pass
    for label in unique_labels:
        group_item = groups[label]
        # Batch append items to the group
        if group_cell_items[label]:
            group_item.appendRows(group_cell_items[label])
        model.appendRow(group_item)

    main_window.setup_tree_model(model)
    main_window.tree_view.collapseAll()
    main_window.tree_view.setUpdatesEnabled(True)


def add_new_group(main_window, name: str, parent_item=None):
    """Adds a new group, safely nesting it at the top of the selected folder or root."""
    from qtpy.QtWidgets import QApplication, QStyle
    from qtpy.QtGui import QStandardItem, QColor
    from qtpy.QtCore import Qt

    item = QStandardItem(name)
    item.setEditable(False)
    item.setDropEnabled(True)

    font = item.font()
    font.setBold(True)
    item.setFont(font)
    item.setBackground(QColor("#3C3C3C"))

    app = QApplication.instance()
    if app:
        item.setIcon(app.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon))

    # If a valid folder was clicked, insert inside it at the top
    if parent_item is not None and parent_item.data(Qt.ItemDataRole.UserRole) is None:
        parent_item.insertRow(0, item)
        main_window.tree_view.expand(parent_item.index())
    else:
        # Safe insertion at the absolute root of the tree
        main_window.tree_model.invisibleRootItem().insertRow(0, item)


def delete_group(main_window, item):
    """Deletes a group folder, moving all its children up one level to its parent."""
    from qtpy.QtCore import Qt

    if item is None or item.data(Qt.ItemDataRole.UserRole) is not None:
        return  # Safety check: not a group

    model = main_window.tree_model
    parent_item = item.parent()
    if parent_item is None:
        parent_item = model.invisibleRootItem()

    parent_name = parent_item.text() if hasattr(parent_item, "text") else "Unclassified"

    # Track clusters for DataFrame update
    cluster_ids = []

    def extract_cids(node):
        for i in range(node.rowCount()):
            child = node.child(i)
            if child:
                cid = child.data(Qt.ItemDataRole.UserRole)
                if cid is not None:
                    cluster_ids.append(cid)
                if child.hasChildren():
                    extract_cids(child)

    extract_cids(item)

    # Move children up to the parent layer safely
    target_row = item.row()
    while item.rowCount() > 0:
        row_items = item.takeRow(0)
        parent_item.insertRow(target_row + 1, row_items)
        target_row += 1

    # Remove the empty folder
    parent_item.removeRow(item.row())

    # Sync DataFrame in the background
    if cluster_ids:
        df = main_window.data_manager.cluster_df
        df.loc[df["cluster_id"].isin(cluster_ids), "KSLabel"] = parent_name


def flatten_group(main_window, item):
    """Removes all sub-folders within a group, pulling all units directly into this group."""
    from qtpy.QtCore import Qt

    if item is None or item.data(Qt.ItemDataRole.UserRole) is not None:
        return

    cluster_items = []
    direct_subfolders = []  # Changed: We only need to delete top-level subfolders

    def gather_children(node, is_root=False):
        for i in range(node.rowCount()):
            child = node.child(i)
            if not child:
                continue

            if child.data(Qt.ItemDataRole.UserRole) is not None:
                if not is_root:
                    cluster_items.append(child)
            else:
                if is_root:
                    # Only track folders directly under the main item for deletion
                    direct_subfolders.append(child)
                # Keep digging for cells
                gather_children(child)

    gather_children(item, is_root=True)

    # Extract all deeply nested cells and pull them up to this folder
    for cell in cluster_items:
        parent = cell.parent()
        if parent:
            row_items = parent.takeRow(cell.row())
            item.appendRow(row_items)

    # Delete the direct sub-folders (this safely destroys all nested folders within them)
    for folder in direct_subfolders:
        # We know the parent is 'item', so we can bypass folder.parent() safely
        item.removeRow(folder.row())

    # Sync DataFrame in the background
    cluster_ids = [
        c.data(Qt.ItemDataRole.UserRole)
        for c in cluster_items
        if c.data(Qt.ItemDataRole.UserRole) is not None
    ]
    if cluster_ids:
        df = main_window.data_manager.cluster_df
        df.loc[df["cluster_id"].isin(cluster_ids), "KSLabel"] = item.text()


def group_clusters_in_tree(main_window, cluster_ids, group_name):
    """
    Dynamically groups selected clusters into a new folder exactly where they are,
    without destroying the rest of the tree. Safe against numpy/Qt type issues.
    """
    model = main_window.tree_model

    # Safely cast all incoming IDs (especially from NumPy) to Python ints
    target_ids = set(int(c) for c in cluster_ids)

    # 1. Manually and safely find the actual cell items in the tree
    items_to_move = []

    def find_cells(parent_item):
        for i in range(parent_item.rowCount()):
            child = parent_item.child(i)
            if child is None:
                continue

            # Only check items that have actual cluster IDs
            cid_data = child.data(
                Qt.ItemDataRole.UserRole if hasattr(Qt, "ItemDataRole") else Qt.UserRole
            )
            if cid_data is not None:
                try:
                    if int(cid_data) in target_ids:
                        items_to_move.append(child)
                except (ValueError, TypeError):
                    pass

            # Recurse into folders
            if child.hasChildren():
                find_cells(child)

    find_cells(model.invisibleRootItem())

    if not items_to_move:
        return

    # 2. Determine where the new folder should go (use the parent of the first item)
    parent_item = items_to_move[0].parent()
    if parent_item is None:
        parent_item = model.invisibleRootItem()

    # 3. Create the new folder item
    group_item = QStandardItem(group_name)
    group_item.setEditable(False)
    group_item.setDropEnabled(True)

    font = group_item.font()
    font.setBold(True)
    group_item.setFont(font)
    group_item.setBackground(QColor("#3C3C3C"))

    app = QApplication.instance()
    if app:
        group_item.setIcon(app.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon))

    # 4. Insert the new folder at the TOP of the parent (Row 0)
    parent_item.insertRow(0, group_item)

    # 5. Move the clusters into the new folder safely
    for item in items_to_move:
        current_parent = item.parent()
        if current_parent is None:
            current_parent = model.invisibleRootItem()

        row_idx = item.row()
        if row_idx >= 0:
            row_items = current_parent.takeRow(row_idx)
            group_item.appendRow(row_items)

    # 6. Update the DataFrame so saving still works
    df = main_window.data_manager.cluster_df
    df.loc[df["cluster_id"].isin(list(target_ids)), "KSLabel"] = group_name

    # 7. Expand only the new folder to show the user it workedload_vision_directory
    main_window.tree_view.expand(group_item.index())
    if hasattr(parent_item, "index"):
        main_window.tree_view.expand(parent_item.index())


# ─────────────────────────────────────────────────────────────────────────────
#  Moving items between groups (multi-select aware)
# ─────────────────────────────────────────────────────────────────────────────

TRASH_GROUP_NAME = "Trash"


def is_group_item(item) -> bool:
    """True when *item* is a folder rather than a cell.

    Cells carry their cluster_id in UserRole; folders carry None. Note that
    ``hasChildren()`` is NOT a reliable test on its own — an empty folder has
    no children but is still a folder.
    """
    return item is not None and item.data(Qt.ItemDataRole.UserRole) is None


#: Name of the cluster_df column carrying each cell's innermost folder.
GROUP_COLUMN = "group"


def build_cluster_group_map(main_window) -> dict:
    """``{cluster_id: innermost folder title}`` for every cell in the tree.

    The *innermost* folder is the one that actually names a cell — a cell in
    ``All/ON/OnParasol`` is an OnParasol, and reporting the whole path or the
    outermost folder would say nothing useful in a table column. Cells sitting
    at the top level, outside any folder, are absent from the map.
    """
    model = getattr(main_window, "tree_model", None)
    if model is None:
        return {}

    out: dict = {}

    def walk(item, folder_title):
        for i in range(item.rowCount()):
            child = item.child(i)
            if child is None:
                continue
            if is_group_item(child):
                # Folders carry None in UserRole, so this is the reliable test;
                # an empty folder still has to be recursed into for correctness
                # even though it contributes nothing.
                walk(child, child.text())
            elif folder_title is not None:
                cid = child.data(Qt.ItemDataRole.UserRole)
                if cid is not None:
                    out[int(cid)] = folder_title

    walk(model.invisibleRootItem(), None)
    return out


def iter_cluster_ids(item) -> list:
    """Every cluster_id at or below *item*, in tree order."""
    out = []

    def recurse(node):
        cid = node.data(Qt.ItemDataRole.UserRole)
        if cid is not None:
            out.append(int(cid))
        for i in range(node.rowCount()):
            child = node.child(i)
            if child is not None:
                recurse(child)

    recurse(item)
    return out


def _is_descendant_of(item, possible_ancestor) -> bool:
    """True when *item* sits anywhere beneath *possible_ancestor*."""
    node = item.parent()
    while node is not None:
        if node is possible_ancestor:
            return True
        node = node.parent()
    return False


def prune_redundant_items(items: list) -> list:
    """Drop any item that already sits inside another item in the list.

    Moving both a folder and one of its children would move the child twice —
    the second take would pull it out of the folder it just travelled with.
    Keeping only the topmost items makes the move idempotent.
    """
    kept = []
    for item in items:
        if any(_is_descendant_of(item, other) for other in items if other is not item):
            continue
        kept.append(item)
    return kept


def move_items_to_group(main_window, items: list, target_item) -> int:
    """Move *items* (cells and/or folders) into *target_item*.

    ``target_item`` may be a folder item or the model's invisible root. Returns
    the number of top-level items actually moved.

    Items already sitting directly in the target are skipped, and a folder can
    never be moved into itself or one of its own descendants — that would
    detach the whole subtree from the model.
    """
    model = main_window.tree_model
    root = model.invisibleRootItem()

    if target_item is None:
        target_item = root
    if target_item is not root and not is_group_item(target_item):
        return 0  # cells cannot contain other items

    movable = []
    for item in prune_redundant_items([i for i in items if i is not None]):
        if item is target_item or _is_descendant_of(target_item, item):
            continue  # would re-parent a folder beneath itself
        parent = item.parent() or root
        if parent is target_item:
            continue  # already there
        movable.append(item)

    if not movable:
        return 0

    moved_cluster_ids = []
    for item in movable:
        moved_cluster_ids.extend(iter_cluster_ids(item))
        parent = item.parent() or root
        row = item.row()
        if row < 0:
            continue
        # Re-read item.row() per iteration: taking a row shifts its siblings.
        target_item.appendRow(parent.takeRow(row))

    # Keep cluster_df in step so saving and the table view reflect the move.
    group_name = target_item.text() if target_item is not root else "Unclassified"
    if moved_cluster_ids and main_window.data_manager is not None:
        df = main_window.data_manager.cluster_df
        df.loc[df["cluster_id"].isin(moved_cluster_ids), "KSLabel"] = group_name

    if target_item is not root:
        main_window.tree_view.expand(target_item.index())
    return len(movable)


def get_or_create_trash(main_window):
    """Return the Trash folder, creating it pinned to the bottom of the root."""
    model = main_window.tree_model
    root = model.invisibleRootItem()

    for i in range(root.rowCount()):
        child = root.child(i)
        if child is not None and is_group_item(child):
            if child.text() == TRASH_GROUP_NAME:
                return child

    trash = QStandardItem(TRASH_GROUP_NAME)
    trash.setEditable(False)
    trash.setDropEnabled(True)

    font = trash.font()
    font.setBold(True)
    trash.setFont(font)
    # Muted red so discarded units read as set-aside at a glance.
    trash.setBackground(QColor("#4A2C2C"))

    app = QApplication.instance()
    if app:
        trash.setIcon(app.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon))

    root.appendRow(trash)
    return trash


def find_trash(main_window):
    """Return the Trash folder if it exists, else None (never creates it)."""
    root = main_window.tree_model.invisibleRootItem()
    for i in range(root.rowCount()):
        child = root.child(i)
        if child is not None and is_group_item(child):
            if child.text() == TRASH_GROUP_NAME:
                return child
    return None


def get_trashed_cluster_ids(main_window) -> set:
    """Cluster IDs currently sitting in Trash — used to exclude them from analyses."""
    trash = find_trash(main_window)
    if trash is None:
        return set()
    return set(iter_cluster_ids(trash))


def items_are_in_trash(main_window, items: list) -> bool:
    """True when every item in *items* sits strictly inside the Trash folder.

    The Trash folder itself does not count as being in the trash — otherwise
    right-clicking it would offer to restore it out of itself.
    """
    trash = find_trash(main_window)
    if trash is None or not items:
        return False
    return all(_is_descendant_of(item, trash) for item in items)


def move_items_to_trash(main_window, items: list) -> int:
    """Move *items* into the Trash folder, creating it on first use."""
    trash = get_or_create_trash(main_window)
    n = move_items_to_group(main_window, items, trash)
    if n:
        main_window.status_bar.showMessage(
            f"Moved {n} item{'s' if n != 1 else ''} to {TRASH_GROUP_NAME}", 4000
        )
    return n


def find_items_by_cluster_ids(main_window, cluster_ids) -> list:
    """Locate the tree cell items for *cluster_ids*, wherever they currently sit.

    Lets the table view — which knows only cluster IDs, not tree items — drive
    the same move operations the tree context menu uses.
    """
    targets = {int(c) for c in cluster_ids}
    found = []

    def recurse(node):
        for i in range(node.rowCount()):
            child = node.child(i)
            if child is None:
                continue
            cid = child.data(Qt.ItemDataRole.UserRole)
            if cid is not None:
                try:
                    if int(cid) in targets:
                        found.append(child)
                except (TypeError, ValueError):
                    pass
            elif child.hasChildren():
                recurse(child)

    recurse(main_window.tree_model.invisibleRootItem())
    return found


def move_cluster_ids_to_group(main_window, cluster_ids, target_item) -> int:
    """Move cells named by ID into *target_item*."""
    items = find_items_by_cluster_ids(main_window, cluster_ids)
    if not items:
        return 0
    return move_items_to_group(main_window, items, target_item)


def move_cluster_ids_to_trash(main_window, cluster_ids) -> int:
    """Move cells named by ID into Trash — the table view's entry point."""
    items = find_items_by_cluster_ids(main_window, cluster_ids)
    if not items:
        main_window.status_bar.showMessage(
            "Those cells were not found in the tree — nothing moved.", 4000
        )
        return 0
    return move_items_to_trash(main_window, items)


def restore_items_from_trash(main_window, items: list) -> int:
    """Move *items* out of Trash and back to the top level of the tree."""
    root = main_window.tree_model.invisibleRootItem()
    n = move_items_to_group(main_window, items, root)
    if n:
        main_window.status_bar.showMessage(
            f"Restored {n} item{'s' if n != 1 else ''} from {TRASH_GROUP_NAME}", 4000
        )
    return n


def collect_group_items(main_window, exclude: list = None) -> list:
    """All folders in the tree as ``(display_path, item)``, in tree order.

    ``exclude`` suppresses a folder *and its whole subtree* — used to keep the
    "Move to group" menu from offering a destination inside the very items the
    user is moving.
    """
    exclude = exclude or []
    out = []

    def recurse(node, path):
        for i in range(node.rowCount()):
            child = node.child(i)
            if child is None or not is_group_item(child):
                continue
            if any(
                child is ex or _is_descendant_of(child, ex) for ex in exclude
            ):
                continue
            child_path = f"{path}/{child.text()}" if path else child.text()
            out.append((child_path, child))
            recurse(child, child_path)

    recurse(main_window.tree_model.invisibleRootItem(), "")
    return out


def rename_class(
    main_window: MainWindow, original_group_name: str, new_group_name: str
):
    """Renames a group non-destructively without rebuilding the tree."""
    df = main_window.data_manager.cluster_df
    df.loc[df["KSLabel"] == original_group_name, "KSLabel"] = new_group_name

    # Safely find the folder in the tree and update its text in place
    model = main_window.tree_model
    matches = model.match(
        model.index(0, 0),
        Qt.DisplayRole,
        original_group_name,
        1,
        Qt.MatchExactly | Qt.MatchRecursive,
    )
    if matches:
        item = model.itemFromIndex(matches[0])
        item.setText(new_group_name)


def feature_extraction(main_window: MainWindow, cluster_ids):
    """Feature extraction for selected clusters."""
    logger.info(f"feature extraction for clusters: {cluster_ids}")
    dlg = FeatureExtractionWindow(main_window, cluster_ids)
    dlg.exec()


def stop_worker(main_window: MainWindow):
    """Stops the background worker threads (spatial + standard plots)."""
    # Spatial worker
    if (
        getattr(main_window, "worker_thread", None)
        and main_window.worker_thread.isRunning()
    ):
        if getattr(main_window, "spatial_worker", None) is not None:
            main_window.spatial_worker.stop()
        main_window.worker_thread.quit()
        main_window.worker_thread.wait()
        main_window.worker_thread = None
        main_window.spatial_worker = None

    # Standard plots worker
    if (
        getattr(main_window, "standard_worker_thread", None)
        and main_window.standard_worker_thread.isRunning()
    ):
        if getattr(main_window, "standard_plots_worker", None) is not None:
            main_window.standard_plots_worker.stop()
        main_window.standard_worker_thread.quit()
        main_window.standard_worker_thread.wait()
        main_window.standard_worker_thread = None
        main_window.standard_plots_worker = None


def load_classification_file(main_window: MainWindow):
    """Load a Vision classification file and populate the tree structure based on it."""
    if not main_window.data_manager:
        QMessageBox.warning(
            main_window, "No Data Loaded", "Please load a Kilosort directory first."
        )
        return

    file_path, _ = QFileDialog.getOpenFileName(
        main_window,
        "Select Classification File",
        recent_paths.last_dir(main_window, "classification"),
        "Text Files (*.txt);;All Files (*)",
    )

    if not file_path:
        return

    recent_paths.remember_dir(file_path, "classification")

    try:
        main_window.status_bar.showMessage("Loading classification file...")
        QApplication.processEvents()

        # Parse the classification file
        with open(file_path, "r") as f:
            lines = f.readlines()

        # Dictionary to store cluster_id -> classification path
        classifications = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                vision_id = int(parts[0])
                classification_path = parts[1]

                # --- FIX: Skip the standard "All/" root folder ---
                if classification_path.startswith("All/"):
                    classification_path = classification_path[4:]
                elif classification_path == "All":
                    classification_path = ""

                # Convert vision_id (1-indexed) to cluster_id (0-indexed)
                cluster_id = vision_id - 1
                classifications[cluster_id] = classification_path

        # Update the tree view based on classifications
        main_window.tree_model.clear()

        # Group clusters by their classifications
        classification_groups = {}
        for cluster_id, path in classifications.items():
            if path not in classification_groups:
                classification_groups[path] = []
            classification_groups[path].append(cluster_id)

        # Also need to maintain clusters that weren't in the classification
        # file
        all_cluster_ids = set(main_window.data_manager.cluster_df["cluster_id"])
        classified_cluster_ids = set(classifications.keys())
        unclassified_cluster_ids = all_cluster_ids - classified_cluster_ids

        # Add clusters that were in the classification file
        for path, cluster_ids in classification_groups.items():
            # Create hierarchical structure from path (e.g., "All/OFF/brisk
            # sustained/" -> nested groups)
            path_parts = [
                part for part in path.split("/") if part
            ]  # Remove empty parts

            # Navigate/create the hierarchical structure
            current_parent = main_window.tree_model
            for _, part in enumerate(path_parts):
                # Look for existing item with this name in current parent
                found_item = None
                for row in range(current_parent.rowCount()):
                    # For QStandardItemModel (like tree_model), use item(row)
                    # For QStandardItem (nested items), use child(row)
                    if type(current_parent).__name__ == "QStandardItemModel":
                        item = current_parent.item(row)
                    else:
                        item = current_parent.child(row)
                    if item and item.text() == part:
                        found_item = item
                        break

                if found_item is None:
                    # Create new group item
                    group_item = QStandardItem(part)
                    group_item.setEditable(False)
                    group_item.setDropEnabled(True)

                    # Style group items differently from cells
                    font = group_item.font()
                    font.setBold(True)
                    group_item.setFont(font)

                    # Set different background color for groups
                    # Dark gray background for groups
                    group_item.setBackground(QColor("#3C3C3C"))

                    # Add folder icon for groups
                    group_item.setIcon(
                        main_window.style().standardIcon(
                            QStyle.StandardPixmap.SP_DirIcon
                        )
                    )

                    current_parent.appendRow(group_item)
                    current_parent = group_item
                else:
                    current_parent = found_item

            # Add cluster items to the final group
            df = main_window.data_manager.cluster_df
            for cluster_id in cluster_ids:
                if cluster_id in all_cluster_ids:
                    # Sync the dataframe so background plots know the cell's new group
                    leaf_name = path_parts[-1] if path_parts else "Unclassified"
                    df.loc[df["cluster_id"] == cluster_id, "KSLabel"] = leaf_name

                    cluster_info = df[df["cluster_id"] == cluster_id].iloc[0]

                    # Create cluster item with n_spikes and ISI info
                    item_text = f"Cluster {cluster_id} (n={cluster_info['n_spikes']}, ISI={cluster_info['isi_violations_pct']:.2f}%)"
                    cell_item = QStandardItem(item_text)
                    cell_item.setEditable(False)

                    # Add file icon for cells
                    cell_item.setIcon(
                        main_window.style().standardIcon(
                            QStyle.StandardPixmap.SP_FileIcon
                        )
                    )

                    # Store the cluster ID in the item's data role
                    cell_item.setData(cluster_id, Qt.ItemDataRole.UserRole)

                    # Prevent dropping items onto cells
                    cell_item.setDropEnabled(False)

                    current_parent.appendRow(cell_item)

        # Add unclassified clusters under an 'Unclassified' group
        if unclassified_cluster_ids:
            unclassified_group = QStandardItem("Unclassified")
            unclassified_group.setEditable(False)
            unclassified_group.setDropEnabled(True)

            # Style group items differently from cells
            font = unclassified_group.font()
            font.setBold(True)
            unclassified_group.setFont(font)

            # Set different background color for groups
            unclassified_group.setBackground(
                QColor("#3C3C3C")
            )  # Dark gray background for groups

            # Add folder icon for groups
            unclassified_group.setIcon(
                main_window.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)
            )

            main_window.tree_model.appendRow(unclassified_group)

            for cluster_id in unclassified_cluster_ids:
                cluster_info = main_window.data_manager.cluster_df[
                    main_window.data_manager.cluster_df["cluster_id"] == cluster_id
                ].iloc[0]

                # Create cluster item with n_spikes and ISI info
                item_text = f"Cluster {cluster_id} (n={cluster_info['n_spikes']}, ISI={cluster_info['isi_violations_pct']:.2f}%)"
                cell_item = QStandardItem(item_text)
                cell_item.setEditable(False)

                # Add file icon for cells
                cell_item.setIcon(
                    main_window.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon)
                )

                # Store the cluster ID in the item's data role
                cell_item.setData(cluster_id, Qt.ItemDataRole.UserRole)

                # Prevent dropping items onto cells
                cell_item.setDropEnabled(False)

                unclassified_group.appendRow(cell_item)

        # Set up the tree model and expand all
        main_window.setup_tree_model(main_window.tree_model)
        main_window.tree_view.collapseAll()

        main_window.status_bar.showMessage(
            f"Loaded classification file with {len(classifications)} classified clusters.",
            5000,
        )
        main_window.load_classification_action.setEnabled(True)

    except Exception as e:
        QMessageBox.critical(
            main_window, "Loading Error", f"Error loading classification file: {e}"
        )
        main_window.status_bar.showMessage("Classification file loading failed.", 5000)


def save_classification_to_file(main_window: MainWindow):
    """
    Saves the current Tree View structure to a text file in the Vision format.
    Format: [VisionID]  [Path/To/Group/]
    Excludes the root 'Unclassified' group.
    """
    if not main_window.data_manager:
        return

    # 1. Dialog for user to choose location and name
    suggested_name = recent_paths.suggested_path(
        main_window,
        "classification",
        "classification.txt",
        preferred_dir=main_window.data_manager.kilosort_dir,
    )

    file_path, _ = QFileDialog.getSaveFileName(
        main_window,
        "Save Classification Text File",
        suggested_name,
        "Text Files (*.txt)",
    )

    if not file_path:
        return  # User cancelled

    recent_paths.remember_dir(file_path, "classification")

    lines_to_write = []

    # Recursive helper to walk the tree
    def recurse_tree(item, current_path):
        for i in range(item.rowCount()):
            child = item.child(i)
            cluster_id = child.data(Qt.ItemDataRole.UserRole)

            if cluster_id is not None:
                # Leaf Node (Cluster) -> Write ID and Path
                # Vision ID is cluster_id + 1
                vision_id = (
                    cluster_id
                    if getattr(main_window.data_manager, "is_vision_only", False)
                    else cluster_id + 1
                )

                # Ensure path starts with "All/"
                final_path = current_path
                if not final_path.startswith("All/"):
                    if final_path:  # if there's already a path, prepend All/
                        final_path = f"All/{final_path}"
                    else:  # if path is completely empty, just make it All/
                        final_path = "All/"

                # Format requires TWO spaces between ID and the path
                lines_to_write.append(f"{vision_id}  {final_path}")

            else:
                # Group Node -> Recurse deeper
                group_name = child.text()

                # Skip root 'Unclassified' group if desired
                if group_name == "Unclassified":
                    continue

                # Build path (e.g., "ON" -> "ON/")
                new_path = f"{current_path}{group_name}/"
                recurse_tree(child, new_path)

    # Start recursion from root
    recurse_tree(main_window.tree_model.invisibleRootItem(), "")

    # Write to file
    try:
        with open(file_path, "w") as f:
            f.write("\n".join(lines_to_write) + "\n")

        main_window.status_bar.showMessage(f"Classification saved to {file_path}", 5000)
    except Exception as e:
        QMessageBox.critical(main_window, "Save Error", f"Could not save file:\n{e}")


def load_raw_data(main_window):
    """Handles manually loading raw data after Kilosort is loaded.

    Presents a two-option dialog so the user can choose between:
    - **Litke directory** (folder containing chunked data000.bin, data001.bin …)
      → opens via PyBinFileReader, which handles the multi-file format natively.
    - **Flat binary file** (.dat / .bin, Kilosort-concatenated)
      → opened as a numpy memmap (legacy path).

    ``DataManager.set_dat_path`` routes to the correct backend automatically
    based on whether the supplied path is a directory or a file.
    """
    if not main_window.data_manager:
        QMessageBox.warning(
            main_window, "No Data", "Please load a Kilosort directory first."
        )
        return

    start_dir = recent_paths.last_dir(main_window, "raw")

    # Ask the user which format they are loading.
    choice = QMessageBox(main_window)
    choice.setWindowTitle("Select Raw Data Format")
    choice.setText("Which raw data format would you like to load?")
    btn_litke = choice.addButton("Litke .bin folder", QMessageBox.ButtonRole.AcceptRole)
    choice.addButton("Flat .dat / .bin file", QMessageBox.ButtonRole.AcceptRole)
    btn_cancel = choice.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
    choice.exec()

    clicked = choice.clickedButton()
    if clicked is btn_cancel or clicked is None:
        return

    raw_path = None

    if clicked is btn_litke:
        # Litke directory: let PyBinFileReader find data000.bin, data001.bin …
        selected = QFileDialog.getExistingDirectory(
            main_window,
            "Select Litke Dataset Folder (containing data*.bin files)",
            start_dir,
        )
        if selected:
            raw_path = Path(selected)
    else:
        # Legacy flat file
        selected, _ = QFileDialog.getOpenFileName(
            main_window,
            "Select Raw Data File (.dat or .bin)",
            start_dir,
            "Binary Files (*.dat *.bin);;All Files (*)",
        )
        if selected:
            raw_path = Path(selected)

    if not raw_path:
        return  # user cancelled the file/folder dialog

    # Directory pick: remember its parent (sibling runs live there). File pick:
    # remember_dir already stores the containing folder.
    recent_paths.remember_dir(
        raw_path.parent if raw_path.is_dir() else raw_path, "raw"
    )

    main_window.status_bar.showMessage("Loading raw data...")
    QApplication.processEvents()

    main_window.data_manager.set_dat_path(raw_path)

    # Unlock the Raw Trace tab
    main_window.analysis_tabs.setTabEnabled(
        main_window.analysis_tabs.indexOf(main_window.raw_panel), True
    )

    display_name = raw_path.name or str(raw_path)
    main_window.status_bar.showMessage(f"Raw data loaded: {display_name}", 5000)

    # Refresh immediately if the raw tab is currently visible
    if main_window.analysis_tabs.currentWidget() == main_window.raw_panel:
        cluster_id = main_window._get_selected_cluster_id()
        if cluster_id is not None:
            main_window.raw_panel.load_data(cluster_id)


def map_reference_run(main_window):
    """
    Menu action: Map Reference Run.

    Opens a directory picker for the reference Vision analysis directory,
    runs EI-based cross-run matching, shows a summary dialog, and installs
    the ReferenceBridge on the DataManager.
    """
    from ..analysis.cross_run_matcher import (
        CrossRunMatcher,
        MatchingReport,
        get_mapping_path,
    )
    from ..analysis.reference_bridge import ReferenceBridge

    dm = main_window.data_manager
    if dm is None:
        QMessageBox.warning(
            main_window, "No Data", "Load a dataset first."
        )
        return

    # --- Check for existing EIs in current run ---
    if dm.vision_eis is None:
        QMessageBox.warning(
            main_window,
            "No EI Data",
            "The current run has no EI data loaded.\n"
            "Load Vision files for the current run first, or ensure the .ei file exists.",
        )
        return

    # --- Pick reference directory ---
    ref_dir = QFileDialog.getExistingDirectory(
        main_window,
        "Select Reference Run (Vision Analysis Directory)",
        recent_paths.last_dir(main_window, "reference"),
    )
    if not ref_dir:
        return

    recent_paths.remember_dir(Path(ref_dir).parent, "reference")

    ref_path = Path(ref_dir)

    # Detect dataset name (Vision convention: files named <dataset_name>.ei, etc.)
    ei_files = list(ref_path.glob("*.ei"))
    if not ei_files:
        QMessageBox.warning(
            main_window,
            "No EI File",
            f"No .ei file found in:\n{ref_dir}\n\n"
            "Select a directory containing Vision analysis output.",
        )
        return

    ref_dataset = ei_files[0].stem

    # --- Check for existing mapping JSON ---
    current_run_path = str(dm.kilosort_dir) if hasattr(dm, 'kilosort_dir') else str(ref_path)
    mapping_path = get_mapping_path(current_run_path, ref_dir)

    report = None
    if mapping_path.exists():
        reply = QMessageBox.question(
            main_window,
            "Existing Mapping Found",
            f"A saved mapping to this reference run already exists:\n"
            f"{mapping_path.name}\n\n"
            "Use the existing mapping, or re-run matching?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
        )
        if reply == QMessageBox.Cancel:
            return
        if reply == QMessageBox.Yes:
            try:
                report = MatchingReport.load(mapping_path)
                main_window.status_bar.showMessage(
                    f"Loaded existing mapping: {report.summary()}", 5000
                )
            except Exception as e:
                QMessageBox.warning(
                    main_window, "Load Failed",
                    f"Could not load saved mapping:\n{e}\n\nRe-running matching.",
                )
                report = None

    # --- Run matching if needed ---
    if report is None:
        main_window.status_bar.showMessage("Running cross-run EI matching...")
        QApplication.processEvents()

        try:
            # Detect current dataset name
            current_dataset = None
            if hasattr(dm, 'vision_dataset_name'):
                current_dataset = dm.vision_dataset_name
            else:
                # Try to detect from existing vision dir
                current_vision_dir = getattr(dm, 'vision_dir', None)
                if current_vision_dir:
                    current_ei_files = list(Path(current_vision_dir).glob("*.ei"))
                    if current_ei_files:
                        current_dataset = current_ei_files[0].stem

            # Build matcher from already-loaded EIs + reference directory
            from ..analysis.vision_integration import load_ei_data
            ref_ei_bundle = load_ei_data(ref_path, ref_dataset)
            ref_eis = ref_ei_bundle.get("ei_data") if ref_ei_bundle else None

            if ref_eis is None:
                QMessageBox.warning(
                    main_window, "EI Load Failed",
                    "Could not load EI data from the reference run.",
                )
                return

            matcher = CrossRunMatcher(
                current_eis=dm.vision_eis,
                reference_eis=ref_eis,
                current_run_path=current_run_path,
                reference_run_path=ref_dir,
            )
            report = matcher.run()

            # Save mapping JSON
            try:
                report.save(mapping_path)
            except Exception as e:
                main_window.status_bar.showMessage(
                    f"Warning: could not save mapping JSON: {e}", 5000
                )

        except Exception as e:
            QMessageBox.critical(
                main_window, "Matching Failed",
                f"Cross-run matching failed:\n{e}",
            )
            return

    # --- Show summary dialog ---
    summary = report.summary()
    n_high = len(report.high_confidence)
    n_marginal = len(report.marginal)
    n_unmatched = len(report.unmatched)
    n_conflicts = len(report.conflicts)

    detail_lines = [summary, ""]
    if n_marginal > 0:
        detail_lines.append(f"Marginal matches (review recommended):")
        for m in report.marginal[:20]:  # show first 20
            detail_lines.append(
                f"  {m.current_id} → {m.reference_id}  "
                f"(corr={m.confidence:.3f}, next={m.next_best_corr:.3f})"
            )
        if n_marginal > 20:
            detail_lines.append(f"  ... and {n_marginal - 20} more")

    msg = QMessageBox(main_window)
    msg.setWindowTitle("Cross-Run Matching Results")
    msg.setText(summary)
    msg.setDetailedText("\n".join(detail_lines))
    msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    msg.setDefaultButton(QMessageBox.Ok)
    msg.button(QMessageBox.Ok).setText("Accept && Load RFs")
    result = msg.exec_()

    if result != QMessageBox.Ok:
        return

    # --- Install ReferenceBridge (STA/params + chirp/grating if present) ---
    main_window.status_bar.showMessage(
        "Loading reference stimuli (STA/RF, chirp, grating)..."
    )
    QApplication.processEvents()

    try:
        bridge = ReferenceBridge.from_matching_report(
            report,
            ref_path,
            ref_dataset,
            load_stas=True,
            load_params=True,
            load_chirp=True,
            load_grating=True,
            ref_is_vision_only=bool(getattr(dm, "is_vision_only", False)),
        )
        dm.install_reference_bridge(bridge, report=report)
        main_window.status_bar.showMessage(
            f"Reference mapped: {bridge.summary()}", 8000
        )
    except Exception as e:
        QMessageBox.critical(
            main_window, "Bridge Failed",
            f"Could not load reference data:\n{e}",
        )
        return

    # --- Re-gate UMAP feature checkboxes (effective chirp/grating) ---
    try:
        if hasattr(main_window, "umap_panel") and main_window.umap_panel is not None:
            main_window.umap_panel.refresh_feature_availability()
    except Exception:
        logger.debug("UMAP feature re-gate after map failed", exc_info=True)

    # --- Refresh population panel (native + borrowed RFs) ---
    try:
        from .panels.population_panel import (
            invalidate_population_caches,
            draw_population_rfs_plot,
        )
        invalidate_population_caches()
        draw_population_rfs_plot(main_window)
    except Exception:
        logger.debug("Population RF redraw after map failed", exc_info=True)