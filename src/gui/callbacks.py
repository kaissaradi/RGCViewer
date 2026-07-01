from __future__ import annotations
import os
import threading
from pathlib import Path
from qtpy.QtWidgets import QFileDialog, QMessageBox, QApplication, QStyle
from qtpy.QtCore import QThread, Qt, QObject, QTimer
from qtpy.QtGui import QStandardItem, QColor

from ..analysis.data_manager import DataManager
from .workers.workers import (
    RefinementWorker, SpatialWorker, StandardPlotsWorker, 
    KilosortLoadWorker, VisionLoadWorker
)
from .widgets.widgets import HighlightStatusPandasModel
from .panels.population_panel import (
    draw_population_timecourse_panel,
    draw_population_rfs_plot,
    invalidate_population_caches,
)
from .panels.feature_extraction import FeatureExtractionWindow
from typing import TYPE_CHECKING
import logging

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from gui.main_window import MainWindow

def _ensure_cache_progress_timer(main_window):
    """Main-thread timer: physics warm-up no longer emits finished_cluster."""
    timer = getattr(main_window, '_cache_progress_timer', None)
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
    timer = getattr(main_window, '_cache_progress_timer', None)
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

    physics_done = getattr(dm, '_physics_done_count', 0)
    std_done = len(dm.standard_plot_cache)
    vision_loaded = bool(getattr(dm, 'vision_stas', None))

    # StandardPlotsWorker drives finished_cluster; physics warm-up does not.
    if vision_loaded:
        done = physics_done
        val = int(physics_done / total * 100)
        ready = physics_done >= total
    else:
        done = std_done
        val = int(std_done / total * 100)
        ready = std_done >= total

    main_window.cache_progress.setValue(min(val, 100))

    if ready and not getattr(main_window, '_cache_save_triggered', False):
        main_window._cache_save_triggered = True
        stop_cache_progress_polling(main_window)
        main_window.cache_progress.hide()
        main_window.status_bar.showMessage(
            "Physics Cache Ready: UMAP and Population panels optimized.", 8000)
        dm.save_standard_plot_cache()

def load_directory(main_window, kilosort_dir=None, dat_file=None):
    """Triggers the background loading of a Kilosort directory."""
    default_dir = Path("/home/localadmin/Documents/Development/data/sorted")
    if not default_dir.exists():
        default_dir = Path.home()
        
    if kilosort_dir is None:
        ks_dir_name = QFileDialog.getExistingDirectory(
            main_window, "Select Kilosort Output Directory", str(default_dir))
    else:
        ks_dir_name = kilosort_dir
        
    if not ks_dir_name:
        return

    # 1. Lock UI and Prep DataManager
    main_window.central_widget.setEnabled(False)
    main_window.status_bar.showMessage("Initializing loader...")
    main_window.data_manager = DataManager(ks_dir_name, main_window)
    main_window.setWindowTitle(f"RGC Viewer - {ks_dir_name}")

    # 2. Setup Thread and Worker
    main_window.ks_load_thread = QThread()
    main_window.ks_load_worker = KilosortLoadWorker(main_window.data_manager, ks_dir_name, dat_file)
    main_window.ks_load_worker.moveToThread(main_window.ks_load_thread)

    # 3. Connect Signals
    main_window.ks_load_thread.started.connect(main_window.ks_load_worker.run)
    main_window.ks_load_worker.progress.connect(lambda msg: main_window.status_bar.showMessage(msg))
    
    # Connect the finished signal to our new UI-thread cleanup function
    main_window.ks_load_worker.finished.connect(
        lambda success, msg: _on_kilosort_loaded(main_window, success, msg, ks_dir_name, dat_file)
    )

    # 4. Fire!
    main_window.ks_load_thread.start()


def _on_kilosort_loaded(main_window, success, message, ks_dir_name, dat_file):
    """Callback triggered on the UI thread when KilosortLoadWorker finishes."""
    if getattr(main_window, 'ks_load_thread', None):
        main_window.ks_load_thread.quit()
        main_window.ks_load_thread.wait()
        main_window.ks_load_thread = None

    if not success:
        QMessageBox.critical(main_window, "Loading Error", message)
        main_window.status_bar.showMessage("Loading failed.", 5000)
        main_window.central_widget.setEnabled(True)
        return

    # --- GUI UPDATES (Safe because we are back on the main thread) ---
    
    # 1. Raw Tab Management
    if dat_file is not None:
        main_window.analysis_tabs.setTabEnabled(
            main_window.analysis_tabs.indexOf(main_window.raw_panel), True)
    else:
        main_window.analysis_tabs.setTabEnabled(
            main_window.analysis_tabs.indexOf(main_window.raw_panel), False)

    # 2. Tree/Table Population
    tree_file_path = os.path.join(ks_dir_name, 'cluster_group_refined_tree.json')
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

    # 4. Handle Vision specific UI updates
    if main_window.data_manager.vision_stas:
        main_window.sta_panel.show()
    else:
        main_window.sta_panel.hide()

    if hasattr(main_window, 'similarity_panel') and main_window.data_manager.vision_available:
        main_window.similarity_panel.on_vision_loaded()

    # 5. Array Transform check
    transform_path = main_window.data_manager.kilosort_dir.parent / 'transforms' / 'array_transform.json'
    if transform_path.exists() and hasattr(main_window, 'standard_plots_panel'):
        main_window.standard_plots_panel.refresh_array_image(str(transform_path))
        cid = main_window._get_selected_cluster_id()
        if cid is not None:
            main_window.standard_plots_panel.update_all(cid)

    n_clusters = len(main_window.data_manager.cluster_df)

    # Auto-start Vision loading if the KS worker found .ei files in the directory.
    # This runs as a SEPARATE VisionLoadWorker so the UI is already unlocked
    # and responsive while Vision data streams in the background.
    auto_dir     = getattr(main_window.data_manager, '_auto_vision_dir', None)
    auto_dataset = getattr(main_window.data_manager, '_auto_vision_dataset', None)
    if auto_dir and auto_dataset:
        main_window.status_bar.showMessage(
            f"Loaded {n_clusters} clusters. Auto-loading Vision data in background...", 6000)
        # Reuse the existing load_vision_directory plumbing but bypass the dialog
        # VisionLoadWorker is already imported at the top of this module
        main_window.vision_load_thread = QThread()
        main_window.vision_load_worker = VisionLoadWorker(main_window.data_manager, auto_dir)
        main_window.vision_load_worker.moveToThread(main_window.vision_load_thread)
        main_window.vision_load_thread.started.connect(main_window.vision_load_worker.run)
        main_window.vision_load_worker.progress.connect(
            lambda msg: main_window.status_bar.showMessage(msg))
        main_window.vision_load_worker.finished.connect(
            lambda success, msg, is_partial: _on_vision_loaded(main_window, success, msg, is_partial))
        main_window.vision_load_thread.start()
    else:
        main_window.status_bar.showMessage(
            f"Successfully loaded {n_clusters} clusters.", 5000)
        
def _on_vision_native_loaded(main_window, success, message, vision_dir_name):
    """Cleanup and GUI update after StandaloneVisionWorker finishes."""

    # --- Safe thread/worker cleanup (mirrors _on_vision_loaded pattern) ---
    if hasattr(main_window, 'vision_load_thread') and main_window.vision_load_thread:
        main_window.vision_load_thread.quit()
        if not main_window.vision_load_thread.wait(2000):
            logger.warning("Vision native load thread didn't exit cleanly, terminating")
            main_window.vision_load_thread.terminate()
            main_window.vision_load_thread.wait(500)
        main_window.vision_load_thread = None

    if hasattr(main_window, 'vision_load_worker') and main_window.vision_load_worker:
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
        main_window.analysis_tabs.indexOf(main_window.raw_panel), False)

    # --- Populate tree and tables ---
    populate_tree_view(main_window)
    main_window._update_table_view_duplicate_highlight()
    main_window._update_tree_view_duplicate_highlight()

    # --- Enable actions ---
    main_window.load_raw_action.setEnabled(True)
    main_window.save_action.setEnabled(True)
    main_window.save_classification_action.setEnabled(True)

    # --- Show STA panel if data is available ---
    if main_window.data_manager.vision_stas:
        main_window.sta_panel.show()

    # --- Start standard plots background worker ---
    start_worker(main_window)

    # --- Kick off EI correlation precompute on background thread ---
    if main_window.data_manager.vision_eis is not None:
        main_window.data_manager.precompute_ei_correlations_background()
        main_window.status_bar.showMessage(
            "Vision loaded. Computing EI correlations in background...", 4000)

    # In _on_vision_loaded, REPLACE the _warm_physics block with:
    if success:
        _all_ids = main_window.data_manager.cluster_df['cluster_id'].astype(int).tolist()

        def _warm_physics():
            main_window.data_manager.ensure_physics_cache(_all_ids, max_workers=1)

        def _on_standard_plots_done():
            # Disconnect immediately so it only fires once
            try:
                main_window.standard_plots_worker.all_done.disconnect(_on_standard_plots_done)
            except RuntimeError:
                pass
                
            # StandardPlotsWorker has finished — feature_cache has all ACGs.
            # Now physics warm-up only needs to do STA seeks, no redundant ACG work.
            main_window.cache_progress_count = 0
            main_window.cache_progress.setValue(0)
            main_window.cache_progress.show()
            start_cache_progress_polling(main_window)
            threading.Thread(target=_warm_physics, daemon=True).start()

        main_window.standard_plots_worker.all_done.connect(_on_standard_plots_done)

def load_vision_directory(main_window):
    """Smart router: Loads Vision-only if no KS exists, otherwise appends Vision to KS."""
    vision_dir_name = QFileDialog.getExistingDirectory(
        main_window, "Select Vision Analysis Directory")
    if not vision_dir_name:
        return

    main_window.central_widget.setEnabled(False)

    # PATH A: Standalone Vision Load (No Kilosort loaded, or already in Vision-only mode)
    if not main_window.data_manager or getattr(main_window.data_manager, 'is_vision_only', False):
        main_window.status_bar.showMessage("Initializing Vision-native loader...")
        
        # Initialize a fresh DataManager and set the flag immediately
        main_window.data_manager = DataManager(vision_dir_name, main_window)
        main_window.data_manager.is_vision_only = True
        main_window.setWindowTitle(f"RGC Viewer - {Path(vision_dir_name).name} (Vision Native)")

        from .workers.workers import StandaloneVisionWorker
        main_window.vision_load_thread = QThread()
        main_window.vision_load_worker = StandaloneVisionWorker(main_window.data_manager, vision_dir_name)
        main_window.vision_load_worker.moveToThread(main_window.vision_load_thread)
        
        main_window.vision_load_worker.finished.connect(
            lambda success, msg: _on_vision_native_loaded(main_window, success, msg, vision_dir_name)
        )
        
    # PATH B: Append Vision to Kilosort (Standard flow)
    else:
        main_window.status_bar.showMessage("Appending Vision data to Kilosort dataset...")
        main_window.vision_load_thread = QThread()
        main_window.vision_load_worker = VisionLoadWorker(main_window.data_manager, vision_dir_name)
        main_window.vision_load_worker.moveToThread(main_window.vision_load_thread)
        
        main_window.vision_load_worker.finished.connect(
            lambda success, msg, is_partial: _on_vision_loaded(main_window, success, msg, is_partial)
        )

    main_window.vision_load_thread.started.connect(main_window.vision_load_worker.run)
    main_window.vision_load_worker.progress.connect(lambda msg: main_window.status_bar.showMessage(msg))
    main_window.vision_load_thread.start()


def _on_vision_loaded(main_window, success, message, is_partial):
    """Callback triggered on the UI thread when VisionLoadWorker finishes."""
    # Properly cleanup the thread and worker
    if hasattr(main_window, 'vision_load_thread') and main_window.vision_load_thread:
        main_window.vision_load_thread.quit()
        if not main_window.vision_load_thread.wait(2000):
            logger.warning("Vision load thread didn't exit cleanly, terminating")
            main_window.vision_load_thread.terminate()
            main_window.vision_load_thread.wait(500)
        main_window.vision_load_thread = None
    
    if hasattr(main_window, 'vision_load_worker') and main_window.vision_load_worker:
        main_window.vision_load_worker.deleteLater()
        main_window.vision_load_worker = None

    main_window.central_widget.setEnabled(True)

    if success and not is_partial:
        invalidate_population_caches()
        main_window.status_bar.showMessage(message, 5000)
    elif is_partial:
        invalidate_population_caches()
        main_window.status_bar.showMessage(f"Loaded partial Vision data: {message}", 5000)
        QMessageBox.warning(main_window, "Vision Loading Warning",
                            f"Could not load all Vision data, but some files were found.\n{message}")
    else:
        QMessageBox.critical(main_window, "Vision Loading Error", message)
        main_window.status_bar.showMessage("Vision loading failed.", 5000)

    # Show STA panel if data is now available
    if main_window.data_manager.vision_stas:
        main_window.sta_panel.show()

    # BUG-6 fix: kick off EI correlation precompute on a background thread
    # so it's ready before the user opens the similarity panel.
    # If a pickle cache exists this returns almost instantly; otherwise it
    # runs the heavy computation without ever touching the UI thread.
    if success and main_window.data_manager.vision_eis is not None:
        main_window.data_manager.precompute_ei_correlations_background()
        main_window.status_bar.showMessage("Vision loaded. Computing EI correlations in background...", 4000)

    # Now that Vision STA data is loaded, compute get_cell_physics for all
    # clusters. The StandardPlotsWorker deliberately skips this so it can
    # never produce stale timecourse=None entries before Vision is ready.
    # In _on_vision_loaded, REPLACE the _warm_physics block with:
    if success:
        _all_ids = main_window.data_manager.cluster_df['cluster_id'].astype(int).tolist()

        def _warm_physics():
            main_window.data_manager.ensure_physics_cache(_all_ids, max_workers=1)

        def _on_standard_plots_done():
            # Disconnect immediately so Vision reload in the same session
            # doesn't fire a second _warm_physics thread.
            try:
                main_window.standard_plots_worker.all_done.disconnect(_on_standard_plots_done)
            except RuntimeError:
                pass  # already disconnected, fine
            main_window.cache_progress_count = 0
            main_window.cache_progress.setValue(0)
            main_window.cache_progress.show()
            start_cache_progress_polling(main_window)
            threading.Thread(target=_warm_physics, daemon=True).start()

        main_window.standard_plots_worker.all_done.connect(_on_standard_plots_done)

    # Trigger a refresh of the currently selected cluster to show new data
    if main_window._get_selected_cluster_id() is not None:
        on_cluster_selection_changed(main_window)


def redraw_population_panels(main_window: MainWindow, subset=None):
    # draws the middle panel (and optionally clears bottom)
    # If an explicit subset is provided by the caller, use it; otherwise derive it.
    if subset is None:
        subset = main_window._get_pop_subset_ids()

    # Import and call all three population plots
    from .panels.population_panel import draw_population_acg_panel, draw_population_fr_panel
    draw_population_timecourse_panel(main_window, subset_ids=subset)
    draw_population_acg_panel(main_window, subset_ids=subset)
    draw_population_fr_panel(main_window, subset_ids=subset)


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


def on_spatial_data_ready(
        main_window: MainWindow,
        cluster_id: int,
        _features: dict):
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
            "Please select a cluster from the table to refine.")
        return

    main_window.refine_button.setEnabled(False)
    main_window.status_bar.showMessage(
        f"Starting refinement for Cluster {cluster_id}...")

    main_window.refine_thread = QThread()
    main_window.refinement_worker = RefinementWorker(
        main_window.data_manager, cluster_id)
    main_window.refinement_worker.moveToThread(main_window.refine_thread)
    main_window.refinement_worker.finished.connect(
        main_window.handle_refinement_results)
    main_window.refinement_worker.error.connect(
        main_window.handle_refinement_error)
    main_window.refinement_worker.progress.connect(
        lambda msg: main_window.status_bar.showMessage(msg, 3000))
    main_window.refine_thread.started.connect(
        main_window.refinement_worker.run)
    main_window.refine_thread.start()


def handle_refinement_results(
        main_window: MainWindow,
        parent_id: int,
        new_clusters: list[int]):
    """Handles the results from a successful refinement operation."""
    main_window.status_bar.showMessage(
        f"Refinement of C{parent_id} complete. Found {len(new_clusters)} sub-clusters.",
        5000)
    main_window.data_manager.update_after_refinement(parent_id, new_clusters)
    invalidate_population_caches()
    # Refresh the tree view to show updated cluster information
    populate_tree_view(main_window)
    main_window.refine_button.setEnabled(True)
    main_window.save_action.setEnabled(True)
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
        QMessageBox.warning(main_window, "No Data", "No DataManager found. Load data first.")
        return

    # Create a safe default path regardless of whether info_path exists
    if main_window.data_manager.info_path:
        original_path = main_window.data_manager.info_path
        suggested_path = str(original_path.parent / f"{original_path.stem}_refined.tsv")
    else:
        # Fallback to the Kilosort directory if info_path is missing
        suggested_path = str(main_window.data_manager.kilosort_dir / "classification_export.tsv")

    # This dialog MUST appear if the function is reached
    save_path, _ = QFileDialog.getSaveFileName(
        main_window, "Save Classification Results", suggested_path, "TSV Files (*.tsv)"
    )

    if save_path:
        # This calls the actual file-writing logic
        save_results(main_window, save_path)


def save_results(main_window, output_path):
    """Saves internal data AND the Vision-compatible text file."""
    try:
        # 1. Save your internal TSV and JSON tree as before
        final_df = main_window.data_manager.cluster_df[['cluster_id', 'KSLabel']].copy()
        final_df.to_csv(output_path, sep='\t', index=False)
        tree_save_path = output_path.replace('.tsv', '_tree.json')
        main_window.data_manager.save_tree_structure(tree_save_path)

        # 2. NEW: Export the Vision-compatible .txt classification
        txt_output_path = output_path.replace('.tsv', '.txt')
        lines_to_write = []

        def recurse_extract(item, current_path):
            for i in range(item.rowCount()):
                child = item.child(i)
                cluster_id = child.data(Qt.ItemDataRole.UserRole)
                if cluster_id is not None:
                    vision_id = main_window.data_manager.get_vision_id_for_cluster(cluster_id)
                    lines_to_write.append(f"{vision_id} {current_path}")
                else:
                    if child.text() != "Unclassified":
                        recurse_extract(child, f"{current_path}{child.text()}/")

        recurse_extract(main_window.tree_model.invisibleRootItem(), "")

        with open(txt_output_path, 'w') as f:
            f.write("\n".join(lines_to_write))

        main_window.status_bar.showMessage("Saved: .tsv, .json, and Vision .txt", 5000)
    except Exception as e:
        QMessageBox.critical(main_window, "Save Error", f"Could not save files: {e}")


def apply_good_filter(main_window: MainWindow):
    """Filters the tree view to show only 'good' clusters."""
    if main_window.data_manager is None:
        return

    df = main_window.data_manager.cluster_df[
        main_window.data_manager.cluster_df['KSLabel'] == 'good'
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
    if main_window.worker_thread is not None or getattr(main_window, "standard_worker_thread", None) is not None:
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
    
    main_window.standard_worker_thread.started.connect(main_window.standard_plots_worker.run)
    main_window.standard_worker_thread.start()
    
    # Kick off low-priority background caching for all clusters
    dm = main_window.data_manager
    if dm is not None and not dm.cluster_df.empty:
        main_window.cache_progress_count = 0
        
        for cid in dm.cluster_df['cluster_id']:
            cid_int = int(cid)
            
            # Check if it's already cached
            has_std = cid_int in getattr(dm, 'standard_plot_cache', {})
            has_feat = cid_int in getattr(dm, 'feature_cache', {})
            
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
    if 'KSLabel' not in df_tree.columns:
        df_tree['KSLabel'] = 'Unknown'

    # Pre-compute string conversions ONCE (O(n) total, not O(n²))
    df_tree['KSLabel_str'] = df_tree['KSLabel'].astype(str)
    unique_labels = sorted(df_tree['KSLabel_str'].unique())

    # Pre-fetch icons and font to avoid redundant Qt calls in the loop
    dir_icon = main_window.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)
    file_icon = main_window.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon)
    
    # Create top-level nodes for each unique KSLabel
    groups = {}
    group_cell_items = {} # To store lists of items for batch append

    for label in unique_labels:
        group_item = QStandardItem(label)
        group_item.setEditable(False)
        group_item.setDropEnabled(True)  # Can drop cells into it

        # Style group items differently from cells
        font = group_item.font()
        font.setBold(True)
        group_item.setFont(font)

        # Set different background color for groups
        group_item.setBackground(QColor('#3C3C3C'))
        group_item.setIcon(dir_icon)

        groups[label] = group_item
        group_cell_items[label] = [] # Initialize list for batching

    # Add each cluster as a child item to its group
    # itertuples is significantly faster than iterrows
    for row in df_tree.itertuples():
        cluster_id = row.cluster_id
        label = row.KSLabel_str

        # The text displayed will be e.g., "Cluster 123 (n=456 spikes)"
        n_spikes = getattr(row, 'n_spikes', '?')
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
    item.setBackground(QColor('#3C3C3C'))
    
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
    
    parent_name = parent_item.text() if hasattr(parent_item, 'text') else "Unclassified"

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
        df.loc[df['cluster_id'].isin(cluster_ids), 'KSLabel'] = parent_name


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
            if not child: continue
            
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
    cluster_ids = [c.data(Qt.ItemDataRole.UserRole) for c in cluster_items if c.data(Qt.ItemDataRole.UserRole) is not None]
    if cluster_ids:
        df = main_window.data_manager.cluster_df
        df.loc[df['cluster_id'].isin(cluster_ids), 'KSLabel'] = item.text()
        
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
            if child is None: continue
            
            # Only check items that have actual cluster IDs
            cid_data = child.data(Qt.ItemDataRole.UserRole if hasattr(Qt, 'ItemDataRole') else Qt.UserRole)
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
    group_item.setBackground(QColor('#3C3C3C'))
    
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
    df.loc[df['cluster_id'].isin(list(target_ids)), 'KSLabel'] = group_name

    # 7. Expand only the new folder to show the user it workedload_vision_directory
    main_window.tree_view.expand(group_item.index())
    if hasattr(parent_item, 'index'):
        main_window.tree_view.expand(parent_item.index())

def rename_class(main_window: MainWindow, original_group_name: str, new_group_name: str):
    """Renames a group non-destructively without rebuilding the tree."""
    df = main_window.data_manager.cluster_df
    df.loc[df["KSLabel"] == original_group_name, 'KSLabel'] = new_group_name
    
    # Safely find the folder in the tree and update its text in place
    model = main_window.tree_model
    matches = model.match(model.index(0, 0), Qt.DisplayRole, original_group_name, 1, Qt.MatchExactly | Qt.MatchRecursive)
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
    if getattr(
        main_window,
        "worker_thread",
            None) and main_window.worker_thread.isRunning():
        if getattr(main_window, "spatial_worker", None) is not None:
            main_window.spatial_worker.stop()
        main_window.worker_thread.quit()
        main_window.worker_thread.wait()
        main_window.worker_thread = None
        main_window.spatial_worker = None

    # Standard plots worker
    if getattr(main_window, "standard_worker_thread",
               None) and main_window.standard_worker_thread.isRunning():
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
            main_window,
            "No Data Loaded",
            "Please load a Kilosort directory first.")
        return

    file_path, _ = QFileDialog.getOpenFileName(
        main_window,
        "Select Classification File",
        str(Path.home()),
        "Text Files (*.txt);;All Files (*)"
    )

    if not file_path:
        return

    try:
        main_window.status_bar.showMessage("Loading classification file...")
        QApplication.processEvents()

        # Parse the classification file
        with open(file_path, 'r') as f:
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
                cluster_id = vision_id if getattr(main_window.data_manager, 'is_vision_only', False) else vision_id - 1
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
        all_cluster_ids = set(
            main_window.data_manager.cluster_df['cluster_id'])
        classified_cluster_ids = set(classifications.keys())
        unclassified_cluster_ids = all_cluster_ids - classified_cluster_ids

        # Add clusters that were in the classification file
        for path, cluster_ids in classification_groups.items():
            # Create hierarchical structure from path (e.g., "All/OFF/brisk
            # sustained/" -> nested groups)
            path_parts = [part for part in path.split(
                '/') if part]  # Remove empty parts

            # Navigate/create the hierarchical structure
            current_parent = main_window.tree_model
            for _, part in enumerate(path_parts):
                # Look for existing item with this name in current parent
                found_item = None
                for row in range(current_parent.rowCount()):
                    # For QStandardItemModel (like tree_model), use item(row)
                    # For QStandardItem (nested items), use child(row)
                    if type(current_parent).__name__ == 'QStandardItemModel':
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
                    group_item.setBackground(QColor('#3C3C3C'))

                    # Add folder icon for groups
                    group_item.setIcon(
                        main_window.style().standardIcon(
                            QStyle.StandardPixmap.SP_DirIcon))

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
                    df.loc[df['cluster_id'] == cluster_id, 'KSLabel'] = leaf_name
                    
                    cluster_info = df[df['cluster_id'] == cluster_id].iloc[0]

                    # Create cluster item with n_spikes and ISI info
                    item_text = f"Cluster {cluster_id} (n={cluster_info['n_spikes']}, ISI={cluster_info['isi_violations_pct']:.2f}%)"
                    cell_item = QStandardItem(item_text)
                    cell_item.setEditable(False)

                    # Add file icon for cells
                    cell_item.setIcon(
                        main_window.style().standardIcon(
                            QStyle.StandardPixmap.SP_FileIcon))

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
                QColor('#3C3C3C'))  # Dark gray background for groups

            # Add folder icon for groups
            unclassified_group.setIcon(
                main_window.style().standardIcon(
                    QStyle.StandardPixmap.SP_DirIcon))

            main_window.tree_model.appendRow(unclassified_group)

            for cluster_id in unclassified_cluster_ids:
                cluster_info = main_window.data_manager.cluster_df[
                    main_window.data_manager.cluster_df['cluster_id'] == cluster_id].iloc[0]

                # Create cluster item with n_spikes and ISI info
                item_text = f"Cluster {cluster_id} (n={cluster_info['n_spikes']}, ISI={cluster_info['isi_violations_pct']:.2f}%)"
                cell_item = QStandardItem(item_text)
                cell_item.setEditable(False)

                # Add file icon for cells
                cell_item.setIcon(
                    main_window.style().standardIcon(
                        QStyle.StandardPixmap.SP_FileIcon))

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
            5000)
        main_window.load_classification_action.setEnabled(True)

    except Exception as e:
        QMessageBox.critical(
            main_window,
            "Loading Error",
            f"Error loading classification file: {e}")
        main_window.status_bar.showMessage(
            "Classification file loading failed.", 5000)

def save_classification_to_file(main_window: MainWindow):
    """
    Saves the current Tree View structure to a text file in the Vision format.
    Format: [VisionID]  [Path/To/Group/]
    Excludes the root 'Unclassified' group.
    """
    if not main_window.data_manager:
        return

    # 1. Dialog for user to choose location and name
    suggested_name = "classification.txt"
    if main_window.data_manager.kilosort_dir:
        suggested_name = str(main_window.data_manager.kilosort_dir / "classification.txt")

    file_path, _ = QFileDialog.getSaveFileName(
        main_window,
        "Save Classification Text File",
        suggested_name,
        "Text Files (*.txt)"
    )

    if not file_path:
        return  # User cancelled

    lines_to_write = []

    # Recursive helper to walk the tree
    def recurse_tree(item, current_path):
        for i in range(item.rowCount()):
            child = item.child(i)
            cluster_id = child.data(Qt.ItemDataRole.UserRole)

            if cluster_id is not None:
                # Leaf Node (Cluster) -> Write ID and Path
                # Vision ID is cluster_id + 1
                vision_id = cluster_id if getattr(main_window.data_manager, 'is_vision_only', False) else cluster_id + 1

                # Ensure path starts with "All/"
                final_path = current_path
                if not final_path.startswith("All/"):
                    if final_path:  # if there's already a path, prepend All/
                        final_path = f"All/{final_path}"
                    else:           # if path is completely empty, just make it All/
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
        with open(file_path, 'w') as f:
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
        QMessageBox.warning(main_window, "No Data", "Please load a Kilosort directory first.")
        return

    start_dir = (
        str(main_window.data_manager.kilosort_dir)
        if main_window.data_manager.kilosort_dir
        else str(Path.home())
    )

    # Ask the user which format they are loading.
    choice = QMessageBox(main_window)
    choice.setWindowTitle("Select Raw Data Format")
    choice.setText("Which raw data format would you like to load?")
    btn_litke  = choice.addButton("Litke .bin folder",  QMessageBox.ButtonRole.AcceptRole)
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