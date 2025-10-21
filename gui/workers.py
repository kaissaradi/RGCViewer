from qtpy.QtCore import QObject, QThread, Signal
from collections import deque
from analysis import analysis_core
import numpy as np

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
                if cluster_id not in self.data_manager.heavyweight_cache:
                    features = self.data_manager.get_heavyweight_features(cluster_id)
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
            spike_times_cluster = self.data_manager.get_cluster_spikes(self.cluster_id)
            params = {'min_spikes': 500, 'ei_sim_threshold': 0.90}
            refined_clusters = analysis_core.refine_cluster_v2(
                spike_times_cluster,
                str(self.data_manager.dat_path),
                self.data_manager.channel_positions,
                params
            )
            self.finished.emit(self.cluster_id, refined_clusters)
        except Exception as e:
            self.error.emit(f"Refinement failed for cluster {self.cluster_id}: {str(e)}")

class RawTraceWorker(QObject):
    """
    Runs in a separate thread to load raw trace data without freezing the UI.
    """
    data_loaded = Signal(int, object, float, float)  # cluster_id, raw_trace_data, start_time, end_time
    error = Signal(str)

    def __init__(self, data_manager, cluster_id, nearest_channels, start_time, end_time):
        super().__init__()
        self.data_manager = data_manager
        self.cluster_id = cluster_id
        self.nearest_channels = nearest_channels
        self.start_time = start_time
        self.end_time = end_time

    def run(self):
        try:
            # Convert time range from seconds to samples
            start_sample = int(self.start_time * self.data_manager.sampling_rate)
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
                self.data_loaded.emit(self.cluster_id, raw_trace_data, self.start_time, self.end_time)
            else:
                self.error.emit(f"No raw trace data available for cluster {self.cluster_id}")
        except Exception as e:
            self.error.emit(f"Raw trace loading failed for cluster {self.cluster_id}: {str(e)}")

# Add this new class to gui/workers.py

class FeatureWorker(QObject):
    """
    Worker to calculate features (EI, snippets) in the background.
    This moves the slowest part of the cluster selection process off the main thread.
    """
    features_ready = Signal(int, dict)  # Emits cluster_id and the features dictionary
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

            # 3. Perform the disk I/O for the small sample.
            snippets_raw = analysis_core.extract_snippets(
                str(self.data_manager.dat_path), spike_sample.astype(int), n_channels=self.data_manager.n_channels
            )
            
            # 4. Perform the rest of the feature calculation.
            snippets_uV = snippets_raw.astype(np.float32) * self.data_manager.uV_per_bit
            snippets_bc = analysis_core.baseline_correct(snippets_uV, pre_samples=20)
            median_ei = analysis_core.compute_ei(snippets_bc, pre_samples=20)

            features = {
                'median_ei': median_ei, 
                'raw_snippets': snippets_bc[:, :, :min(30, snippets_bc.shape[2])]
            }
            
            # 5. Emit the results back to the main thread.
            self.features_ready.emit(self.cluster_id, features)

        except Exception as e:
            self.error.emit(f"Feature extraction failed for cluster {self.cluster_id}: {str(e)}")