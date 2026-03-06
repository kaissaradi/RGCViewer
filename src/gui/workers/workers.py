from qtpy.QtCore import QObject, QThread, Signal
from collections import deque
from ...analysis import analysis_core
import numpy as np
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class KilosortLoadWorker(QObject):
    """Background worker to handle heavy Kilosort and auto-Vision I/O."""
    finished = Signal(bool, str)
    progress = Signal(str)

    def __init__(self, data_manager, ks_dir_name, dat_file):
        super().__init__()
        self.dm = data_manager
        self.ks_dir_name = ks_dir_name
        self.dat_file = dat_file

    def run(self):
        try:
            self.progress.emit("Loading Kilosort files...")
            success, message = self.dm.load_kilosort_data()
            if not success:
                self.finished.emit(False, message)
                return

            if self.dat_file is not None:
                self.dm.set_dat_path(Path(self.dat_file))

            self.progress.emit("Building cluster dataframe (this may take a moment)...")
            self.dm.build_cluster_dataframe()

            # Auto-load Vision Data
            vision_dir = Path(self.ks_dir_name)
            dataset_name = None
            for ei_file in vision_dir.glob('*.ei'):
                dataset_name = ei_file.stem
                break 

            if dataset_name:
                self.progress.emit(f"Found Vision EI file ('{dataset_name}') - loading automatically...")
                self.dm.load_vision_data(vision_dir, dataset_name)
            else:
                logger.debug("No complete set of Vision files found; skipping automatic loading")

            # Load cell type file
            ls_txt = list(vision_dir.glob('*.txt'))
            txt_file = ls_txt[0] if ls_txt else None
            self.dm.load_cell_type_file(txt_file)

            self.finished.emit(True, "Kilosort data loaded successfully.")
        except Exception as e:
            logger.exception("Error in KilosortLoadWorker")
            self.finished.emit(False, str(e))


class VisionLoadWorker(QObject):
    """Background worker to handle explicit Vision directory loading."""
    finished = Signal(bool, str, bool) # success, message, is_partial
    progress = Signal(str)

    def __init__(self, data_manager, vision_dir_name):
        super().__init__()
        self.dm = data_manager
        self.vision_dir_name = vision_dir_name

    def run(self):
        try:
            vision_dir = Path(self.vision_dir_name)
            self.progress.emit(f"Loading Vision files from {vision_dir.name}...")
            
            # Find dataset name dynamically
            dataset_name = None
            for ei_file in vision_dir.glob('*.ei'):
                dataset_name = ei_file.stem
                break
                
            success, message = self.dm.load_vision_data(vision_dir, dataset_name)
            
            is_partial = False
            if not success:
                params_path = vision_dir / 'sta_params.params'
                sta_path = vision_dir / 'sta_container.sta'
                if params_path.exists() or sta_path.exists():
                    success_partial, message_partial = self.dm.load_vision_data(vision_dir, dataset_name)
                    if success_partial:
                        success = True
                        is_partial = True
                        message = message_partial
            
            self.finished.emit(success, message, is_partial)
        except Exception as e:
            logger.exception("Error in VisionLoadWorker")
            self.finished.emit(False, str(e), False)

class SpatialWorker(QObject):
    """
    Runs in a separate thread to compute heavyweight features without freezing the UI.
    """
    result_ready = Signal(int, dict)

    def __init__(self, data_manager):
        super().__init__()
        self.data_manager = data_manager
        self.is_running = True
        self.queue = deque()

    def run(self):
        while self.is_running:
            if self.queue:
                cluster_id = self.queue.popleft()
                # Use DataManager API which handles cache locking internally
                features = self.data_manager.get_heavyweight_features(
                    cluster_id)
                if features:
                    self.result_ready.emit(cluster_id, features)
            else:
                QThread.msleep(100)

    def add_to_queue(self, cluster_id, high_priority=False):
        if cluster_id in self.queue:
            return
        if high_priority:
            self.queue.appendleft(cluster_id)
        else:
            self.queue.append(cluster_id)

    def stop(self):
        self.is_running = False


class RefinementWorker(QObject):
    """
    Runs the `refine_cluster_v2` function in a background thread.
    """
    finished = Signal(int, list)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, data_manager, cluster_id):
        super().__init__()
        self.data_manager = data_manager
        self.cluster_id = cluster_id

    def run(self):
        try:
            spike_times_cluster = self.data_manager.get_cluster_spikes(
                self.cluster_id)
            params = {'min_spikes': 500, 'ei_sim_threshold': 0.90}
            # Prefer passing the memmap if available to avoid reopening the dat
            # file
            dat_source = self.data_manager.raw_data_memmap if getattr(
                self.data_manager, 'raw_data_memmap', None) is not None else str(
                self.data_manager.dat_path)
            refined_clusters = analysis_core.refine_cluster_v2(
                spike_times_cluster,
                dat_source,
                self.data_manager.channel_positions,
                params
            )
            self.finished.emit(self.cluster_id, refined_clusters)
        except Exception as e:
            self.error.emit(
                f"Refinement failed for cluster {self.cluster_id}: {str(e)}")


class RawTraceWorker(QObject):
    """
    Runs in a separate thread to load raw trace data without freezing the UI.
    """
    data_loaded = Signal(
        int,
        object,
        float,
        float)  # cluster_id, raw_trace_data, start_time, end_time
    error = Signal(str)

    def __init__(
            self,
            data_manager,
            cluster_id,
            nearest_channels,
            start_time,
            end_time):
        super().__init__()
        self.data_manager = data_manager
        self.cluster_id = cluster_id
        self.nearest_channels = nearest_channels
        self.start_time = start_time
        self.end_time = end_time

    def run(self):
        try:
            # Convert time range from seconds to samples
            start_sample = int(
                self.start_time *
                self.data_manager.sampling_rate)
            end_sample = int(self.end_time * self.data_manager.sampling_rate)

            # Ensure we stay within bounds
            start_sample = max(0, start_sample)
            end_sample = min(self.data_manager.n_samples, end_sample)

            # Get the raw trace data for the nearest channels
            raw_trace_data = self.data_manager.get_raw_trace_snippet(
                self.nearest_channels, start_sample, end_sample
            )

            if raw_trace_data is not None and raw_trace_data.size > 0:
                # Emit the loaded data
                self.data_loaded.emit(
                    self.cluster_id,
                    raw_trace_data,
                    self.start_time,
                    self.end_time)
            else:
                self.error.emit(
                    f"No raw trace data available for cluster {self.cluster_id}")
        except Exception as e:
            self.error.emit(
                f"Raw trace loading failed for cluster {self.cluster_id}: {str(e)}")

# Add this new class to gui/workers.py


class FeatureWorker(QObject):
    """
    Worker to calculate features (EI, snippets) in the background.
    This moves the slowest part of the cluster selection process off the main thread.
    """
    features_ready = Signal(
        int, dict)  # Emits cluster_id and the features dictionary
    error = Signal(str)

    def __init__(self, data_manager, cluster_id):
        super().__init__()
        self.data_manager = data_manager
        self.cluster_id = cluster_id

    # In gui/workers.py

# REPLACE the run method in the FeatureWorker class with this:
    # In gui/workers.py -> class FeatureWorker

    def run(self):
        """
        Calculates features by taking a small, fixed sample of the first spikes,
        providing a consistently fast result for all clusters.
        """
        try:
            # 1. Get all spike times for the selected cluster.
            all_spikes = self.data_manager.get_cluster_spikes(self.cluster_id)

            if len(all_spikes) == 0:
                self.error.emit(f"Cluster {self.cluster_id} has no spikes.")
                return

            # 2. Take a small sample of the *first* spikes for speed.
            sample_size = min(len(all_spikes), 100)
            spike_sample = all_spikes[:sample_size]

            # 3. Perform the disk I/O for the small sample. Prefer memmap if
            # set.
            if getattr(self.data_manager, 'raw_data_memmap', None) is not None:
                dat_source = self.data_manager.raw_data_memmap
            else:
                dat_source = str(self.data_manager.dat_path)

            snippets_raw = analysis_core.extract_snippets(
                dat_source, spike_sample.astype(int), n_channels=self.data_manager.n_channels)

            # 4. Perform the rest of the feature calculation.
            snippets_uV = snippets_raw.astype(
                np.float32) * self.data_manager.uV_per_bit
            snippets_bc = analysis_core.baseline_correct(
                snippets_uV, pre_samples=20)
            median_ei = analysis_core.compute_ei(snippets_bc, pre_samples=20)

            features = {
                'median_ei': median_ei,
                'raw_snippets': snippets_bc[:, :, :min(30, snippets_bc.shape[2])]
            }

            # 5. Emit the results back to the main thread.
            self.features_ready.emit(self.cluster_id, features)

        except Exception as e:
            self.error.emit(
                f"Feature extraction failed for cluster {self.cluster_id}: {str(e)}")


class StandardPlotsWorker(QObject):
    """
    Background worker that pre-computes and caches standard plots (ISI/ACG/FR)
    for clusters so that UI tab switches are nearly instant.
    """
    finished_cluster = Signal(int)
    error = Signal(str)

    def __init__(self, data_manager):
        super().__init__()
        self.data_manager = data_manager
        self.queue = deque()
        self.is_running = True

    def run(self):
        """
        Main worker loop. Pulls cluster IDs off a queue and asks the DataManager
        to compute standard-plot data and assemble core physics for each one.
        """
        while self.is_running:
            if self.queue:
                cluster_id = self.queue.popleft()
                try:
                    # 1. Pre-calculate and cache standard plots
                    self.data_manager.get_standard_plot_data(cluster_id)
                    
                    # 2. Pre-assemble and cache physical properties
                    if hasattr(self.data_manager, 'get_cell_physics'):
                        self.data_manager.get_cell_physics(cluster_id)
                        
                    self.finished_cluster.emit(int(cluster_id))
                except Exception as e:
                    self.error.emit(
                        f"Background precompute failed for cluster {cluster_id}: {e}")
            else:
                QThread.msleep(100)

    def add_to_queue(self, cluster_id, high_priority=False):
        """
        Enqueue a cluster for background caching. Duplicate IDs are ignored.
        """
        if cluster_id in self.queue:
            return
        if high_priority:
            self.queue.appendleft(cluster_id)
        else:
            self.queue.append(cluster_id)

    def stop(self):
        self.is_running = False
