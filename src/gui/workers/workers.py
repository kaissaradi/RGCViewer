from qtpy.QtCore import QObject, QThread, Signal
from collections import deque
from ...analysis import analysis_core
import numpy as np
import pandas as pd
import sklearn.cluster
import logging
import os
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
_VISION_SUFFIXES = (".ei", ".sta", ".params", ".neurons")


def _vision_stems_in(vision_dir: Path):
    """Stems that have at least one Vision file. One scandir, names only."""
    by_suffix = {suf: [] for suf in _VISION_SUFFIXES}
    try:
        with os.scandir(vision_dir) as it:
            for entry in it:
                lower = entry.name.lower()
                for suf in _VISION_SUFFIXES:
                    if lower.endswith(suf):
                        by_suffix[suf].append(Path(entry.name).stem)
                        break
    except OSError:
        return []
    ordered = []
    seen = set()
    for suf in _VISION_SUFFIXES:
        for stem in by_suffix[suf]:
            if stem in seen:
                continue
            seen.add(stem)
            ordered.append(stem)
    return ordered


def _resolve_vision_dataset_name(vision_dir: Path, preferred_stem=None):
    """
    Finds the Vision dataset name (file stem shared by .ei/.sta/.params/
    .neurons) in `vision_dir`. Returns None if no Vision files are found at
    all. Does NOT require an .ei file — that was the bug.

    When several datasets share a folder, prefer the stem that matches the
    sort directory name (``data006`` next to ``data006.sta``).
    """
    stems = _vision_stems_in(vision_dir)
    if not stems:
        return None
    if preferred_stem:
        if preferred_stem in stems:
            return preferred_stem
        for stem in stems:
            if stem.startswith(preferred_stem) or preferred_stem.startswith(stem):
                return stem
    return stems[0]


def _locate_vision_dataset(ks_dir: Path):
    """
    Finds the directory holding the Vision files for a Kilosort output at
    `ks_dir`, returning (vision_dir, dataset_name) or (None, None).

    Vision files sit in the folder the user picked or one level up. The
    colocated case is checked first so an experiment that has both keeps
    the files that belong to this sort.
    """
    preferred = ks_dir.name
    for candidate in (ks_dir, ks_dir.parent):
        dataset_name = _resolve_vision_dataset_name(
            candidate, preferred_stem=preferred
        )
        if dataset_name:
            return candidate, dataset_name
    return None, None


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
            ks_dir = Path(self.ks_dir_name)
            vision_dir, dataset_name = _locate_vision_dataset(ks_dir)
            logger.debug(
                "KilosortLoadWorker resolved dataset_name=%r in %s",
                dataset_name,
                vision_dir,
            )

            if dataset_name:
                self.progress.emit("Found Vision data. Queueing background load...")
                # Tell the DataManager where the data is so callbacks.py can spawn the VisionLoadWorker
                self.dm._auto_vision_dir = str(vision_dir)
                self.dm._auto_vision_dataset = dataset_name
            else:
                logger.info(
                    "No .ei/.sta/.params/.neurons files found in %s or %s — "
                    "Vision auto-load will NOT trigger.",
                    ks_dir,
                    ks_dir.parent,
                )

            # Load cell type file. It travels with the Vision files, but a
            # classification written back into the sort directory wins.
            # Always also look one level up — same rule as stim / Vision.
            txt_search_dirs = [ks_dir]
            parent = ks_dir.parent
            if parent != ks_dir:
                txt_search_dirs.append(parent)
            if vision_dir is not None and vision_dir not in txt_search_dirs:
                txt_search_dirs.append(vision_dir)
            txt_file = next(
                (f for d in txt_search_dirs for f in sorted(d.glob("*.txt"))), None
            )
            self.dm.load_cell_type_file(txt_file)

            # The manifest is what turns a source run into a light level, so
            # it has to be read before anything that labels runs.
            self.progress.emit("Reading stimulus manifest...")
            self.dm.load_stimulus_manifest()

            # Presence only. One scandir per search root (this folder, its
            # ksfiles/, the parent, the parent's ksfiles/). Contrast and
            # grating then filter that listing in memory so they flash.
            self.progress.emit("Checking for chirp analysis data...")
            found = self.dm.probe_stimulus_analyses()
            self.progress.emit("Checking for contrast-response data...")
            self.progress.emit("Checking for grating analysis data...")
            present = [name for name, paths in found.items() if paths]
            if present:
                self.progress.emit(
                    "Found " + ", ".join(present) + " analysis file(s)."
                )
            logger.info(
                "Analysis files: %s",
                {name: len(paths) for name, paths in found.items()},
            )

            self.finished.emit(True, "Kilosort and Vision data loaded successfully.")
        except Exception as e:
            logger.exception("Error in KilosortLoadWorker")
            self.finished.emit(False, str(e))


class VisionLoadWorker(QObject):
    """Background worker to handle explicit Vision directory loading."""

    finished = Signal(bool, str, bool)  # success, message, is_partial
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

            has_ei = self.dm.vision_eis is not None
            has_sta = self.dm.vision_stas is not None
            is_partial = success and has_sta and not has_ei

            # STA SNR reads the movies. Must not run on the GUI thread
            # (attach_sta_quality_column only writes the prepared column).
            if success and has_sta:
                self.progress.emit("Computing STA quality...")
                self.dm.prepare_sta_quality_column()

            self.finished.emit(success, message, is_partial)
        except Exception as e:
            logger.exception("Error in VisionLoadWorker")
            self.finished.emit(False, str(e), False)


class StimulusAnalysisLoadWorker(QObject):
    """Load chirp / contrast / grating .npy after the cluster table is up.

    The Kilosort worker only globs. This worker unpickles the chosen file
    for each stimulus. Chirp table columns are attached on the main thread
    when ``finished`` fires — ``cluster_df`` is main-thread-owned by then.
    """

    finished = Signal(bool, str)
    progress = Signal(str)

    def __init__(self, data_manager):
        super().__init__()
        self.dm = data_manager
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            cands = getattr(self.dm, "_analysis_candidates", None)
            if not cands:
                cands = self.dm.probe_stimulus_analyses()

            if self._stop:
                self.finished.emit(False, "Analysis load cancelled.")
                return

            if cands.get("chirp"):
                self.progress.emit("Loading chirp analysis...")
                ok, msg = self.dm.load_chirp_data(cands["chirp"][0])
                if ok:
                    logger.debug(msg)
                elif msg:
                    logger.info(msg)

            if self._stop:
                self.finished.emit(False, "Analysis load cancelled.")
                return

            if cands.get("contrast"):
                self.progress.emit("Loading contrast-response data...")
                contrast_path = max(cands["contrast"], key=lambda p: p.stat().st_size)
                ok, msg = self.dm.load_contrast_data(contrast_path)
                if ok:
                    logger.debug(msg)
                elif msg:
                    logger.info(msg)

            if self._stop:
                self.finished.emit(False, "Analysis load cancelled.")
                return

            if cands.get("grating"):
                self.progress.emit("Loading grating analysis...")
                ok, msg = self.dm.load_grating_data(cands["grating"])
                if ok:
                    logger.debug(msg)
                elif msg:
                    logger.info(msg)

            self.finished.emit(True, "Stimulus analyses ready.")
        except Exception as e:
            logger.exception("Error in StimulusAnalysisLoadWorker")
            self.finished.emit(False, str(e))


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
            params = {"min_spikes": 500, "ei_sim_threshold": 0.90}
            # Prefer PyBinFileReader if available, then memmap, then path string.
            if getattr(self.data_manager, "raw_reader", None) is not None:
                dat_source = self.data_manager.raw_reader
            elif getattr(self.data_manager, "raw_data_memmap", None) is not None:
                dat_source = self.data_manager.raw_data_memmap
            else:
                dat_source = str(self.data_manager.dat_path)
            refined_clusters = analysis_core.refine_cluster_v2(
                spike_times_cluster,
                dat_source,
                self.data_manager.channel_positions,
                params,
            )
            self.finished.emit(self.cluster_id, refined_clusters)
        except Exception as e:
            self.error.emit(
                f"Refinement failed for cluster {self.cluster_id}: {str(e)}"
            )


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
            # Priority: PyBinFileReader > memmap > path string.
            # RefinementWorker uses the same priority order for consistency.
            if getattr(self.data_manager, "raw_reader", None) is not None:
                dat_source = self.data_manager.raw_reader
            elif getattr(self.data_manager, "raw_data_memmap", None) is not None:
                dat_source = self.data_manager.raw_data_memmap
            else:
                dat_source = str(self.data_manager.dat_path)

            snippets_raw = analysis_core.extract_snippets(
                dat_source,
                spike_sample.astype(int),
                n_channels=self.data_manager.n_channels,
            )

            # 4. Perform the rest of the feature calculation.
            snippets_uV = snippets_raw.astype(np.float32) * self.data_manager.uV_per_bit
            snippets_bc = analysis_core.baseline_correct(snippets_uV, pre_samples=20)
            median_ei = analysis_core.compute_ei(snippets_bc, pre_samples=20)

            features = {
                "median_ei": median_ei,
                "raw_snippets": snippets_bc[:, :, : min(30, snippets_bc.shape[2])],
            }

            # 5. Emit the results back to the main thread.
            self.features_ready.emit(self.cluster_id, features)

        except Exception as e:
            self.error.emit(
                f"Feature extraction failed for cluster {self.cluster_id}: {str(e)}"
            )


class StandardPlotsWorker(QObject):
    finished_cluster = Signal(int)
    all_done = Signal()  # ← NEW: fires once when queue is empty
    error = Signal(str)

    def __init__(self, data_manager):
        super().__init__()
        self.data_manager = data_manager
        self.queue = deque()
        self.is_running = True
        self._all_done_emitted = False  # ← NEW

    def run(self):
        if hasattr(self.data_manager, "load_persisted_caches"):
            self.data_manager.load_persisted_caches()

        while self.is_running:
            if self.queue:
                self._all_done_emitted = False  # ← reset if new work arrives
                cluster_id = self.queue.popleft()
                try:
                    self.data_manager.get_standard_plot_data(cluster_id)
                except Exception as e:
                    # Added missing logger call for test verification
                    logger.error(
                        f"Failed to compute standard plots for cluster {cluster_id}"
                    )
                    self.error.emit(
                        f"Background precompute failed for cluster {cluster_id}: {str(e)}"
                    )
                finally:
                    self.finished_cluster.emit(int(cluster_id))
                    QThread.msleep(20)
            else:
                if not self._all_done_emitted:  # ← NEW
                    self.all_done.emit()  # ← NEW
                    self._all_done_emitted = True  # ← NEW
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

    finished = Signal(int, bool, str)  # cluster_id, success, message

    def __init__(self, data_manager, cluster_id):
        super().__init__()
        self.dm = data_manager
        self.cluster_id = int(cluster_id)

    def run(self):
        try:
            result = self.dm.compute_grating_data_for_cluster(self.cluster_id)
            if result is None:
                self.finished.emit(
                    self.cluster_id,
                    False,
                    f"No grating trials for cluster {self.cluster_id}",
                )
            else:
                self.finished.emit(self.cluster_id, True, "")
        except Exception as e:
            logger.exception(
                "GratingComputeWorker failed for cluster %s", self.cluster_id
            )
            self.finished.emit(self.cluster_id, False, str(e))


class GratingBatchWorker(QObject):
    """
    One-time, sequential DSI/OSI compute for every cluster in the dataset,
    run once at startup (chained after physics-cache warm-up) so
    population-level DS/OS views (probe map, RF-plot markers) reflect the
    whole dataset immediately, rather than only clusters the user has
    individually opened in GratingPanel.

    This intentionally overrides the "don't batch-precompute" design
    documented on GratingComputeWorker above — that tradeoff (startup time
    vs. complete population views without manual per-cluster visits) was a
    deliberate choice, not an oversight; see the ADR-style justification in
    callbacks.py's _on_vision_loaded where this is wired up.

    Runs on a real QThread (not a plain threading.Thread) specifically so
    its `progress`/`finished` signals reliably marshal back onto the GUI
    thread via Qt's normal cross-thread signal/slot queuing — this is
    NOT interchangeable with QTimer.singleShot() called from a bare Python
    thread, which requires the calling thread to already have a running
    Qt event loop and is not reliable cross-platform (confirmed broken on
    Windows in practice; Qt's own docs state a QTimer must be started on
    the thread that has the event loop it needs to fire on).

    Sequential by design (one cluster at a time, not a worker pool) — see
    _on_vision_loaded for the reasoning: this can take a while for a large
    dataset, which is an accepted, deliberate tradeoff.
    """

    progress = Signal(int, int)  # (done_count, total_count)
    finished = Signal()

    def __init__(self, data_manager, cluster_ids):
        super().__init__()
        self.dm = data_manager
        self.cluster_ids = [int(c) for c in cluster_ids]
        self._stop_requested = False

    def stop(self):
        """Cooperative cancellation — checked between clusters, not mid-compute."""
        self._stop_requested = True

    def run(self):
        total = len(self.cluster_ids)
        for i, cid in enumerate(self.cluster_ids):
            if self._stop_requested:
                break
            # Unlocked membership check: worst case under a race with a
            # concurrent GratingPanel-triggered single-cluster compute is a
            # harmless redundant recompute for that one cluster — the
            # actual cache write in compute_grating_data_for_cluster is
            # properly locked, so this can't corrupt the shared cache.
            if cid in self.dm.grating_computed_cache:
                continue
            try:
                self.dm.compute_grating_data_for_cluster(cid)
            except Exception:
                logger.exception(
                    "GratingBatchWorker failed for cluster %s; skipping.", cid
                )
            if i % 25 == 0 or i == total - 1:
                self.progress.emit(i + 1, total)
        self.finished.emit()


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
            self.progress.emit(
                f"Loading Vision-native dataset from {vision_dir.name}..."
            )

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

    # raw_blocks rides along so the panel can overlay the feature traces the
    # embedding was actually built from (temporal STA, ACG, chirp PSTH) without
    # re-deriving them on the GUI thread.
    finished = Signal(
        object, object, object, object, object, object, object
    )  # embedding, matrix, valid_ids, discarded_ids, metadata_df, raw_blocks,
       # col_labels
    error = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        data_manager,
        selected_cluster_ids=None,
        n_components=2,
        feature_config=None,
        filter_config=None,
    ):
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
                    target_ids = self.dm.cluster_df["cluster_id"].values
                else:
                    target_ids = []

            if len(target_ids) == 0:
                self.error.emit("No clusters to run UMAP on.")
                return

            self.progress.emit("Ensuring physics cache...")
            self.dm.ensure_physics_cache(target_ids)

            self.progress.emit("Extracting raw features...")
            raw_blocks, valid_ids, discarded_ids = self.dm.get_raw_feature_blocks(
                target_ids, self.filter_config
            )

            if len(valid_ids) == 0:
                self.error.emit(
                    "No valid features could be extracted (all cells filtered out)."
                )
                return

            self.progress.emit("Assembling feature matrix...")
            matrix, col_labels = analysis_core.build_feature_matrix(
                raw_blocks, self.feature_config
            )

            self.progress.emit(f"Running UMAP on {len(valid_ids)} cells...")
            reducer = umap.UMAP(
                n_neighbors=min(15, len(valid_ids) - 1),
                min_dist=0.1,
                metric="euclidean",
                low_memory=True,
                n_jobs=-1,
                n_components=self.n_components,
                verbose=False,
            )
            embedding = reducer.fit_transform(matrix)

            # Reconstruct the metadata DataFrame
            meta_df = pd.DataFrame(index=range(len(valid_ids)))
            meta_df["cluster_id"] = valid_ids

            # Get KSLabel
            if not self.dm.cluster_df.empty and "KSLabel" in self.dm.cluster_df.columns:
                label_map = dict(
                    zip(self.dm.cluster_df["cluster_id"], self.dm.cluster_df["KSLabel"])
                )
                meta_df["KSLabel"] = [
                    label_map.get(cid, "unsorted") for cid in valid_ids
                ]
            else:
                meta_df["KSLabel"] = "unsorted"

            # Get Polarity
            polarities = []
            for i, cid in enumerate(valid_ids):
                tc = raw_blocks["temporal"][i]
                if tc is not None and len(tc) > 0:
                    peak_val = np.max(tc)
                    trough_val = np.min(tc)
                    is_off = abs(trough_val) > abs(peak_val)
                    polarities.append("OFF" if is_off else "ON")
                else:
                    polarities.append("ON")
            meta_df["Polarity"] = polarities

            # Firing Rate
            meta_df["Firing Rate"] = raw_blocks["scalars"]["firing_rate"].values
            # isi_violations (lowercase!)
            meta_df["isi_violations"] = raw_blocks["scalars"]["isi_violations"].values
            # Time to Peak
            meta_df["Time to Peak"] = raw_blocks["scalars"]["time_to_peak"].values
            # RF Area
            meta_df["RF Area"] = raw_blocks["scalars"]["rf_area"].values
            # Ellipticity
            meta_df["Ellipticity"] = raw_blocks["scalars"]["ellipticity"].values
            # Grating DSI/OSI/peak rate — metadata-only (the embedding
            # itself uses the pooled tuning-curve-shape PCA block instead,
            # see analysis_core.py's build_feature_matrix), kept here so
            # they're available for scatter-plot coloring/hover, same as
            # the other metadata-only scalars above.
            if "grating_dsi" in raw_blocks["scalars"].columns:
                meta_df["Grating DSI"] = raw_blocks["scalars"]["grating_dsi"].values
                meta_df["Grating OSI"] = raw_blocks["scalars"]["grating_osi"].values
                meta_df["Grating Peak Rate (Hz)"] = raw_blocks["scalars"][
                    "grating_peak_rate_hz"
                ].values
            # Per-unit quality indices. These live on cluster_df rather than in
            # raw_blocks['scalars'] — they are attached at load time from the
            # chirp file and the .sta — so they are looked up by cluster_id
            # rather than sliced by row. Colouring the embedding by them answers
            # the question the scatter cannot: whether a blob is a cell type or
            # a pile of units whose features are mostly noise.
            cdf = getattr(self.dm, "cluster_df", None)
            if cdf is not None and not cdf.empty and "cluster_id" in cdf.columns:
                for col, label in (("sta_snr", "STA SNR"), ("chirp_qi", "Chirp QI"),
                                   ("chirp_onoff", "Chirp ON/OFF")):
                    if col not in cdf.columns:
                        continue
                    lookup = dict(zip(cdf["cluster_id"], cdf[col]))
                    meta_df[label] = [
                        float(lookup.get(cid, np.nan)) for cid in valid_ids
                    ]

            # Color Opponency
            meta_df["Color Opponency"] = 0.0

            self.progress.emit(f"UMAP complete for {len(valid_ids)} cells")
            self.finished.emit(
                embedding, matrix, valid_ids, discarded_ids, meta_df, raw_blocks,
                col_labels,
            )

        except Exception as e:
            logger.exception("UMAP Worker failed")
            self.error.emit(str(e))


class ClusterWorker(QObject):
    """Background worker for clustering (Ward hierarchical or K-Means).

    Operates on the 2D/3D UMAP embedding itself (whatever array the caller
    passes in as feature_matrix — see umap_panel.py's run_clustering, which
    passes self.embedding, not the pre-UMAP weighted feature matrix).
    Clustering on the embedding, rather than the original high-dimensional
    weighted feature space, ensures the cluster boundaries the user sees
    match the boundaries actually computed — clustering pre-UMAP and only
    painting labels onto the embedding for display can make visually
    adjacent points land in different clusters, since UMAP's projection
    doesn't preserve high-dimensional distances exactly.

    Ward agglomerative clustering is the preferred method for RGC data.
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
                    linkage="ward",
                )
                labels = clusterer.fit_predict(self.feature_matrix)
                self.finished.emit(labels, "Ward")
            else:
                # K-Means — kept as fast flat-partition fallback
                kmeans = sklearn.cluster.KMeans(
                    n_clusters=self.param, random_state=42, n_init=10
                )
                labels = kmeans.fit_predict(self.feature_matrix)
                self.finished.emit(labels, "K-Means")
        except Exception as e:
            logger.exception("Clustering failed")
            self.error.emit(str(e))
