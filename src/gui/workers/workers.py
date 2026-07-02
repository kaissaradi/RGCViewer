from qtpy.QtCore import QObject, QThread, Signal
from collections import deque
from ...analysis import analysis_core
import numpy as np
import pandas as pd
import sklearn.cluster
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


# Vision files for a single dataset all share one stem, e.g.:
#   data000.ei, data000.sta, data000.params, data000.neurons
# Every worker below needs to recover that stem from a directory listing.
# Preference order matters only when a directory somehow contains stems from
# more than one dataset; .ei is checked first purely to preserve prior
# behavior, but a dataset missing .ei (a perfectly normal, supported case —
# see vision_integration.load_ei_data's FileNotFoundError handling) must
# still resolve via .sta/.params/.neurons instead of silently giving up.
_DATASET_NAME_GLOB_PRIORITY = ('*.ei', '*.sta', '*.params', '*.neurons')


def _resolve_vision_dataset_name(vision_dir: Path):
    """
    Finds the Vision dataset name (file stem shared by .ei/.sta/.params/
    .neurons) in `vision_dir`. Returns None if no Vision files are found at
    all. Does NOT require an .ei file — that was the bug.
    """
    for pattern in _DATASET_NAME_GLOB_PRIORITY:
        for f in vision_dir.glob(pattern):
            return f.stem
    return None


class KilosortLoadWorker(QObject):
    """Background worker to handle Kilosort and Vision I/O synchronously."""
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

            # --- SYNCHRONOUS VISION LOADING ---
            vision_dir = Path(self.ks_dir_name)
            dataset_name = _resolve_vision_dataset_name(vision_dir)
            print(f"[VISION-DEBUG][workers] KilosortLoadWorker resolved dataset_name="
                  f"{dataset_name!r} in {vision_dir}", flush=True)

            if dataset_name:
                self.progress.emit("Found Vision data. Queueing background load...")
                # Tell the DataManager where the data is so callbacks.py can spawn the VisionLoadWorker
                self.dm._auto_vision_dir = str(vision_dir)
                self.dm._auto_vision_dataset = dataset_name
            else:
                print(f"[VISION-DEBUG][workers] No .ei/.sta/.params/.neurons files found in "
                      f"{vision_dir} — Vision auto-load will NOT trigger.", flush=True)

            # Load cell type file
            ls_txt = list(vision_dir.glob('*.txt'))
            txt_file = ls_txt[0] if ls_txt else None
            self.dm.load_cell_type_file(txt_file)

            # --- CHIRP DATA (optional) ---
            # chirp_analysis.py writes directly into this same directory, so
            # no path configuration is needed. Missing file is not an error —
            # load_chirp_data() handles that internally and just leaves
            # chirp_available False.
            self.progress.emit("Checking for chirp analysis data...")
            chirp_success, chirp_msg = self.dm.load_chirp_data()
            if chirp_success:
                logger.debug(chirp_msg)

            # --- GRATING DATA (optional) ---
            # Same colocated-with-kilosort_dir convention as chirp. Unlike
            # chirp, this may only find a RAW file — DSI/OSI then get
            # computed later, on demand, per cluster (see GratingComputeWorker).
            self.progress.emit("Checking for grating analysis data...")
            grating_success, grating_msg = self.dm.load_grating_data()
            if grating_success:
                logger.debug(grating_msg)

            self.finished.emit(True, "Kilosort and Vision data loaded successfully.")
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
            dataset_name = _resolve_vision_dataset_name(vision_dir)
            if dataset_name is None:
                self.finished.emit(
                    False,
                    f"No Vision files (.ei/.sta/.params/.neurons) found in {vision_dir}.",
                    False,
                )
                return

            # load_vision_data handles partial loading internally — one call is enough.
            # The old code called it a second time on failure which was pure wasted work.
            success, message = self.dm.load_vision_data(vision_dir, dataset_name)

            has_ei  = self.dm.vision_eis  is not None
            has_sta = self.dm.vision_stas is not None
            is_partial = success and has_sta and not has_ei

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
            # Prefer PyBinFileReader if available, then memmap, then path string.
            if getattr(self.data_manager, 'raw_reader', None) is not None:
                dat_source = self.data_manager.raw_reader
            elif getattr(self.data_manager, 'raw_data_memmap', None) is not None:
                dat_source = self.data_manager.raw_data_memmap
            else:
                dat_source = str(self.data_manager.dat_path)
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

            # 3. Perform the disk I/O for the small sample.
            # Priority: PyBinFileReader > memmap > path string.
            # RefinementWorker uses the same priority order for consistency.
            if getattr(self.data_manager, 'raw_reader', None) is not None:
                dat_source = self.data_manager.raw_reader
            elif getattr(self.data_manager, 'raw_data_memmap', None) is not None:
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
    finished_cluster = Signal(int)
    all_done = Signal()          # ← NEW: fires once when queue is empty
    error = Signal(str)

    def __init__(self, data_manager):
        super().__init__()
        self.data_manager = data_manager
        self.queue = deque()
        self.is_running = True
        self._all_done_emitted = False   # ← NEW

    def run(self):
        if hasattr(self.data_manager, 'load_persisted_caches'):
            self.data_manager.load_persisted_caches()

        while self.is_running:
            if self.queue:
                self._all_done_emitted = False   # ← reset if new work arrives
                cluster_id = self.queue.popleft()
                try:
                    self.data_manager.get_standard_plot_data(cluster_id)
                except Exception as e:
                    # Added missing logger call for test verification
                    logger.error(f"Failed to compute standard plots for cluster {cluster_id}")
                    self.error.emit(f"Background precompute failed for cluster {cluster_id}: {str(e)}")
                finally:
                    self.finished_cluster.emit(int(cluster_id))
                    QThread.msleep(20)
            else:
                if not self._all_done_emitted:   # ← NEW
                    self.all_done.emit()          # ← NEW
                    self._all_done_emitted = True # ← NEW
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


class GratingComputeWorker(QObject):
    """
    One-shot DSI/OSI compute for a single cluster, from raw grating data.

    Deliberately NOT a persistent queue-worker like StandardPlotsWorker.
    Grating DSI/OSI (FFT + 1000-shuffle permutation test per condition) is
    only ever needed for clusters the user actually views — batch
    precomputing all ~900 clusters at dataset-load time would be wasted
    work for the vast majority never opened in a session. Spawned on demand
    by GratingPanel with its own throwaway QThread; caches into
    DataManager.grating_computed_cache so repeat views of the same cluster
    don't recompute.
    """
    finished = Signal(int, bool, str)   # cluster_id, success, message

    def __init__(self, data_manager, cluster_id):
        super().__init__()
        self.dm = data_manager
        self.cluster_id = int(cluster_id)

    def run(self):
        try:
            result = self.dm.compute_grating_data_for_cluster(self.cluster_id)
            if result is None:
                self.finished.emit(self.cluster_id, False,
                                    f"No grating trials for cluster {self.cluster_id}")
            else:
                self.finished.emit(self.cluster_id, True, "")
        except Exception as e:
            logger.exception("GratingComputeWorker failed for cluster %s", self.cluster_id)
            self.finished.emit(self.cluster_id, False, str(e))


class StandaloneVisionWorker(QObject):
    """Background worker to handle loading pure Vision datasets."""
    finished = Signal(bool, str)
    progress = Signal(str)

    def __init__(self, data_manager, vision_dir_name):
        super().__init__()
        self.dm = data_manager
        self.vision_dir_name = vision_dir_name

    def run(self):
        try:
            vision_dir = Path(self.vision_dir_name)
            self.progress.emit(f"Loading Vision-native dataset from {vision_dir.name}...")
            
            # Find dataset name dynamically
            dataset_name = _resolve_vision_dataset_name(vision_dir)

            if not dataset_name:
                self.finished.emit(
                    False,
                    f"No Vision files (.ei/.sta/.params/.neurons) found in {vision_dir}.",
                )
                return

            # Execute our new native loader
            success, message = self.dm.load_vision_native_data(vision_dir, dataset_name)
            
            if success:
                pass

            self.finished.emit(success, message)
        except Exception as e:
            logger.exception("Error in StandaloneVisionWorker")
            self.finished.emit(False, str(e))


class UMAPWorker(QObject):
    """Background worker to compute features and run UMAP."""
    finished = Signal(object, object, object, object, object)  # embedding, matrix, valid_ids, discarded_ids, metadata_df
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, data_manager, selected_cluster_ids=None, n_components=2, feature_config=None, filter_config=None):
        super().__init__()
        self.dm = data_manager
        self.selected_cluster_ids = selected_cluster_ids
        self.n_components = n_components
        self.feature_config = feature_config or {}
        self.filter_config = filter_config or {}

    def run(self):
        try:
            try:
                import umap
            except ImportError:
                self.error.emit("umap-learn library is not installed.")
                return

            target_ids = self.selected_cluster_ids
            if target_ids is None:
                if not self.dm.cluster_df.empty:
                    target_ids = self.dm.cluster_df['cluster_id'].values
                else:
                    target_ids = []

            if len(target_ids) == 0:
                self.error.emit("No clusters to run UMAP on.")
                return

            self.progress.emit("Ensuring physics cache...")
            self.dm.ensure_physics_cache(target_ids)

            self.progress.emit("Extracting raw features...")
            raw_blocks, valid_ids, discarded_ids = self.dm.get_raw_feature_blocks(target_ids, self.filter_config)

            if len(valid_ids) == 0:
                self.error.emit("No valid features could be extracted (all cells filtered out).")
                return

            self.progress.emit("Assembling feature matrix...")
            matrix, col_labels = analysis_core.build_feature_matrix(raw_blocks, self.feature_config)

            self.progress.emit(f"Running UMAP on {len(valid_ids)} cells...")
            reducer = umap.UMAP(
                n_neighbors=min(15, len(valid_ids) - 1),
                min_dist=0.1,
                metric='euclidean',
                low_memory=True,
                n_jobs=-1,
                n_components=self.n_components,
                verbose=False
            )
            embedding = reducer.fit_transform(matrix)

            # Reconstruct the metadata DataFrame
            meta_df = pd.DataFrame(index=range(len(valid_ids)))
            meta_df['cluster_id'] = valid_ids

            # Get KSLabel
            if not self.dm.cluster_df.empty and 'KSLabel' in self.dm.cluster_df.columns:
                label_map = dict(zip(self.dm.cluster_df['cluster_id'], self.dm.cluster_df['KSLabel']))
                meta_df['KSLabel'] = [label_map.get(cid, 'unsorted') for cid in valid_ids]
            else:
                meta_df['KSLabel'] = 'unsorted'

            # Get Polarity
            polarities = []
            for i, cid in enumerate(valid_ids):
                tc = raw_blocks['temporal'][i]
                if tc is not None and len(tc) > 0:
                    peak_val = np.max(tc)
                    trough_val = np.min(tc)
                    is_off = abs(trough_val) > abs(peak_val)
                    polarities.append("OFF" if is_off else "ON")
                else:
                    polarities.append("ON")
            meta_df['Polarity'] = polarities

            # Firing Rate
            meta_df['Firing Rate'] = raw_blocks['scalars']['firing_rate'].values
            # isi_violations (lowercase!)
            meta_df['isi_violations'] = raw_blocks['scalars']['isi_violations'].values
            # Time to Peak
            meta_df['Time to Peak'] = raw_blocks['scalars']['time_to_peak'].values
            # RF Area
            meta_df['RF Area'] = raw_blocks['scalars']['rf_area'].values
            # Ellipticity
            meta_df['Ellipticity'] = raw_blocks['scalars']['ellipticity'].values
            # Color Opponency
            meta_df['Color Opponency'] = 0.0

            self.progress.emit(f"UMAP complete for {len(valid_ids)} cells")
            self.finished.emit(embedding, matrix, valid_ids, discarded_ids, meta_df)

        except Exception as e:
            logger.exception("UMAP Worker failed")
            self.error.emit(str(e))


class ClusterWorker(QObject):
    """Background worker for clustering (Ward hierarchical or K-Means).

    Ward agglomerative clustering is the preferred method for RGC data.
    It operates in the weighted feature space (temporal STA PCs, ACG PCs,
    scalar metrics) and produces a dendrogram whose top-level splits
    naturally recover the dominant biological axes (ON/OFF, then
    transient/sustained, then RF size) without any user guidance.

    K-Means is kept as a fast alternative when the user wants a flat
    partition with a specific k.
    """
    finished = Signal(object, str)  # labels, method_name
    error = Signal(str)

    def __init__(self, feature_matrix, method, param):
        super().__init__()
        self.feature_matrix = np.array(feature_matrix, copy=True)
        self.method = method
        self.param = param  # n_clusters for both methods

    def run(self):
        try:
            if self.method == "Ward":
                clusterer = sklearn.cluster.AgglomerativeClustering(
                    n_clusters=self.param,
                    linkage='ward',
                )
                labels = clusterer.fit_predict(self.feature_matrix)
                self.finished.emit(labels, "Ward")
            else:
                # K-Means — kept as fast flat-partition fallback
                kmeans = sklearn.cluster.KMeans(
                    n_clusters=self.param, random_state=42, n_init=10)
                labels = kmeans.fit_predict(self.feature_matrix)
                self.finished.emit(labels, "K-Means")
        except Exception as e:
            logger.exception("Clustering failed")
            self.error.emit(str(e))