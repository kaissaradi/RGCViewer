from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import numpy as np
import pandas as pd
from pathlib import Path
from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtGui import QStandardItem
from . import analysis_core
from . import vision_integration
from .constants import (
    ISI_REFRACTORY_PERIOD_MS,
    EI_CORR_THRESHOLD,
    LS_CELL_TYPE_LABELS,
    CHIRP_MIN_REPEATS_FOR_QI,
    STA_SNR_GOOD,
)
import pickle
import os
import tempfile
import logging
import ast
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

try:
    import bin2py

    _BIN2PY_AVAILABLE = True
except ImportError:
    _BIN2PY_AVAILABLE = False
logger = logging.getLogger(__name__)


def get_channel_template_mappings(templates: np.ndarray) -> dict:
    channel_to_templates = {}
    template_to_channels = {}

    n_templates = templates.shape[0]
    n_channels = templates.shape[2]

    # Amplitude of [clusters, channels]
    amplitudes = templates.max(axis=1) - templates.min(axis=1)
    # Non-zero amplitudes
    cls, chs = np.where(amplitudes > 0)
    for ch in range(n_channels):
        channel_to_templates[ch] = cls[chs == ch]
    for cid in range(n_templates):
        template_to_channels[cid] = chs[cls == cid]

    d_out = {
        "channel_to_templates": channel_to_templates,
        "template_to_channels": template_to_channels,
    }
    return d_out


def ei_corr(
    ref_ei_dict, test_ei_dict, method: str = "full", n_removed_channels: int = 1
) -> np.ndarray:
    # Courtesy of @DRezeanu
    # Basic validation: handle None or empty inputs gracefully
    if not ref_ei_dict or not test_ei_dict:
        return np.array([])

    # Pull reference eis, filtering out invalid entries
    ref_ids = list(ref_ei_dict.keys())
    ref_eis = []
    for cell in ref_ids:
        entry = ref_ei_dict.get(cell)
        if entry is None:
            continue
        ei_arr = getattr(entry, "ei", None)
        if isinstance(ei_arr, np.ndarray) and ei_arr.size > 0:
            ref_eis.append(ei_arr)

    if n_removed_channels > 0:
        max_ref_vals = [np.array(np.max(ei, axis=1)) for ei in ref_eis]
        ref_to_remove = [np.argsort(val)[-n_removed_channels:] for val in max_ref_vals]
        ref_eis = [
            np.delete(ei, ref_to_remove[idx], axis=0) for idx, ei in enumerate(ref_eis)
        ]

    # Set any EI value where the ei is less than 1.5* its standard deviation to 0
    # Added check for std to avoid division by zero (when std is 0)
    for idx, ei in enumerate(ref_eis):
        ei_std = ei.std()
        if ei_std > 0:
            ref_eis[idx][abs(ei) < (ei_std * 1.5)] = 0
        else:
            # If std is 0, all values are the same, set all to 0
            ref_eis[idx][:] = 0

    # For 'full' method: flatten each 512 x 201 ei array into a vector
    # and stack flattened eis into a numpy array
    if "full" in method:
        ref_eis_flat = [ei.flatten() for ei in ref_eis]
        ref_eis = np.array(ref_eis_flat)
    # For 'time' method, take max of absolute value over time and
    # stack the resulting 512 x 1 vectors into a numpy array
    elif "space" in method:
        ref_eis_mean = [np.max(np.abs(ei), axis=1) for ei in ref_eis]
        ref_eis = np.array(ref_eis_mean)
    # For 'power' method, square each 512 x 201 ei array, take the mean over time,
    # and stack the resulting 512 x 1 vectors into a numpy array
    elif "power" in method:
        ref_eis_mean = [np.mean(ei**2, axis=1) for ei in ref_eis]
        ref_eis = np.array(ref_eis_mean)
    else:
        raise NameError("Method poperty must be 'full', 'time', or 'power'.")

    # Pull test eis, filtering out invalid entries
    test_ids = list(test_ei_dict.keys())
    test_eis = []
    for cell in test_ids:
        entry = test_ei_dict.get(cell)
        if entry is None:
            continue
        ei_arr = getattr(entry, "ei", None)
        if isinstance(ei_arr, np.ndarray) and ei_arr.size > 0:
            test_eis.append(ei_arr)

    if n_removed_channels > 0:
        max_test_vals = [np.array(np.max(ei, axis=1)) for ei in test_eis]
        test_to_remove = [
            np.argsort(val)[-n_removed_channels:] for val in max_test_vals
        ]
        test_eis = [
            np.delete(ei, test_to_remove[idx], axis=0)
            for idx, ei in enumerate(test_eis)
        ]

    # Set the EI value where the EI is less than 1.5* its standard deviation
    # to 0
    for idx, ei in enumerate(test_eis):
        ei_std = ei.std()
        if ei_std > 0:
            test_eis[idx][abs(ei) < (ei_std * 1.5)] = 0
        else:
            # If std is 0, all values are the same, set all to 0
            test_eis[idx][:] = 0

    # For 'full' method: flatten each 512 x 201 ei array into a vector
    # and stack flattened eis into a numpy array
    if "full" in method:
        test_eis_flat = [ei.flatten() for ei in test_eis]
        test_eis = np.array(test_eis_flat)
    # For 'time' method, take max of absolute value over time and
    # stack the resulting 512 x 1 vectors into a numpy array
    elif "space" in method:
        test_eis_mean = [np.max(np.abs(ei), axis=1) for ei in test_eis]
        test_eis = np.array(test_eis_mean)
    # For 'power' method, square each 512 x 201 ei array, take the mean over time,
    # and stack the resulting 512 x 1 vectors into a numpy array
    elif "power" in method:
        test_eis_mean = [np.mean(ei**2, axis=1) for ei in test_eis]
        test_eis = np.array(test_eis_mean)
    else:
        raise NameError("Method poperty must be 'full', 'space', or 'power'.")

    # If after filtering we have no valid EIs, return empty array
    if len(ref_eis) == 0 or len(test_eis) == 0:
        return np.array([])

    num_pts = ref_eis.shape[1]

    # Calculate covariance and correlation
    c = test_eis @ ref_eis.T / num_pts
    d = np.mean(test_eis, axis=1)[:, None] * np.mean(ref_eis, axis=1)[:, None].T
    covs = c - d

    std_calc = np.std(test_eis, axis=1)[:, None] * np.std(ref_eis, axis=1)[:, None].T
    # Avoid division by zero - set to 0 if std calculation is 0
    corr = np.divide(covs, std_calc, out=np.zeros_like(covs), where=std_calc != 0)

    # Set nan values and infinite values to 0
    np.nan_to_num(corr, copy=False, nan=0, posinf=0, neginf=0)

    return corr.T


def sort_electrode_map(electrode_map: np.ndarray) -> np.ndarray:
    """
    Sort electrodes by their x, y locations.

    This uses lexsort to sort electrodes by their x, y locations
    First sort by rows, break ties by columns.
    As each row is jittered but within row the electrodes have exact same y location.

    Parameters:
    electrode_map (numpy.ndarray): The electrode locations of shape (512, 2).

    Returns:
    numpy.ndarray: Sorted indices of the electrodes (512,).
    """
    sorted_indices = np.lexsort((electrode_map[:, 0], electrode_map[:, 1]))
    return sorted_indices


class DataManager(QObject):
    """
    Manages all data loading, processing, and caching.
    """

    is_dirty = False
    ei_updates_ready = Signal(dict, dict)

    def __init__(self, kilosort_dir, main_window=None):
        super().__init__()
        self.ei_updates_ready.connect(self._apply_ei_updates)
        self.kilosort_dir = Path(kilosort_dir)
        self.exp_name = self.kilosort_dir.parent.parent.name
        self.datafile_name = self.kilosort_dir.parent.name
        logger.debug(
            f"Initializing DataManager for experiment={self.exp_name}, datafile={self.datafile_name}"
        )  # NOTE: load_stim_timing() is intentionally NOT called here.
        # It makes a live DataJoint DB connection which can block for 30-120s
        # on a timeout. It is called inside load_kilosort_data() instead,
        # which runs on a background QThread.

        self.ei_cache = {}
        self.heavyweight_cache = {}
        self.feature_cache = {}  # Cache for feature extraction panel (PCA, ACG, etc.)
        self._physics_done_count = 0
        # Lock to protect accesses to heavyweight_cache from multiple threads
        self._feature_lock = threading.Lock()
        self._heavyweight_lock = threading.Lock()
        self.isi_cache = {}  # Cache for ISI violation calculations

        # Cache + lock for standard plots (ISI / ACG / FR)
        self.standard_plot_cache = {}
        self._standard_plot_lock = threading.Lock()
        self._load_standard_plot_cache_from_disk()

        # Per-cell in-flight locks for get_cell_physics.
        # Prevents two concurrent workers from both missing the cache and
        # redundantly recomputing the same cell at the same time.
        # _physics_cell_locks: dict[cluster_id -> threading.Lock]
        # _physics_cell_locks_lock: guards the dict itself.
        self._physics_cell_locks = {}
        self._physics_cell_locks_lock = threading.Lock()

        # Background save protection
        self._save_lock = threading.Lock()
        self._save_in_progress = False

        self.dat_path = None

        self.new_class_id = 0
        self.cluster_df = pd.DataFrame()
        self.original_cluster_df = pd.DataFrame()
        self.info_path = None
        self.uV_per_bit = 0.195
        self.main_window = main_window  # Reference to main window for tree operations

        # status df
        self.status_df = pd.DataFrame(columns=["cluster_id", "status", "set"])
        self.status_df["set"] = self.status_df["set"].astype(object)
        self.status_csv = self.kilosort_dir / "status.csv"

        self.cluster_id_to_idx = None  # Map cluster_id -> row index
        # --- Vision Data ---
        self.vision_eis = None
        self.vision_stas = None
        self.vision_params = None
        self.vision_channel_positions = None  # Store channel positions from vision data
        self.vision_sta_width = None  # Store stimulus width for coordinate alignment
        self.vision_sta_height = None  # Store stimulus height for coordinate alignment
        self.ei_corr_dict = (
            None  # Initialize to None, will be set when vision data is loaded
        )

        # --- Cross-Run Reference Bridge ---
        self.reference_bridge = None  # Optional[ReferenceBridge]
        # Per original-run UI cluster_id → CellMatchCaveat (match quality + provenance)
        self.match_caveats = {}

        # --- MEA Similarity Data ---
        # (n_templates, n_templates) from Kilosort
        self.similar_templates = None
        self.cluster_to_template = None  # dict[int -> int]
        self.mea_sim_cache = {}  # cluster_id -> DataFrame
        self.vision_sim_cache = {}  # cluster_id -> DataFrame
        self.vision_available = False

        # Initialize raw data memmap attribute (kept for legacy/fallback paths)
        self.raw_data_memmap = None
        # PyBinFileReader instance — used when loading native Litke .bin files.
        # Takes priority over raw_data_memmap when set.
        self.raw_reader = None

        # Initialize refractory period (default from constants)
        self.refractory_period_ms = ISI_REFRACTORY_PERIOD_MS

        # --- Chirp Data ---
        # Precomputed offline by chirp_analysis.py and saved as a .npy in
        # kilosort_dir. cluster_id inside the file is Kilosort space (same
        # space as cluster_df['cluster_id']) — no ID translation needed.
        self.chirp_data = None  # raw loaded dict (mdic) or None
        self.chirp_id_to_row = None  # dict[int cluster_id -> int row index]
        self.chirp_n_repeats = None  # stimulus repeats behind quality_index
        self.chirp_qi_is_baden = False  # True when chirp_qi was recomputed
        self.sta_ids_consistent = True   # .sta matches this sort?
        self.sta_consistency_message = ""
        self.chirp_available = False

        # --- Grating (DSOS / SF) Data ---
        # Unlike chirp, this is NOT always precomputed offline. A raw file
        # (spike_times_by_trial + trial_parameters) is guaranteed; an
        # analyzed file (DSI/OSI already computed) is not. Whichever is
        # found determines grating_status; DSI/OSI get computed on demand
        # per cluster via GratingComputeWorker when only raw data exists.
        self.grating_data = None  # pre-analyzed dict[cluster_id -> ...] or None
        self.grating_raw_data = (
            None  # {'spike_times_by_trial':..., 'trial_parameters':...} or None
        )
        self.grating_available = False
        self.grating_status = "missing"  # 'ok' | 'raw_only' | 'missing'
        self.grating_conditions = None  # sorted list[(barWidth, temporalFrequency)]
        self.grating_computed_cache = (
            {}
        )  # dict[cluster_id -> per-condition dict], on-demand results
        self._grating_cache_lock = threading.Lock()

    def set_refractory_period(self, new_period_ms):
        """
        Set the refractory period for ISI analysis.
        """
        self.refractory_period_ms = float(new_period_ms)

    def get_refractory_period(self):
        """
        Get the current refractory period.
        """
        return self.refractory_period_ms

    def _apply_ei_updates(self, potential_dups_map, max_dup_r_map):
        """Safely apply background-computed EI updates to the DataFrame on the main thread."""
        if self.cluster_df.empty:
            return

        # .infer_objects(copy=False) after fillna silences the pandas
        # "Downcasting object dtype arrays on .fillna is deprecated"
        # FutureWarning — the trailing .astype() alone does NOT (the
        # warning fires during fillna itself). Matches the pattern already
        # used at the KSLabel / isi_violations_pct fillna sites below.
        self.cluster_df["potential_dups"] = (
            self.cluster_df["cluster_id"]
            .map(potential_dups_map)
            .fillna(False)
            .infer_objects(copy=False)
            .astype(bool)
        )
        self.cluster_df["max_dup_r"] = (
            self.cluster_df["cluster_id"]
            .map(max_dup_r_map)
            .fillna(0.0)
            .infer_objects(copy=False)
            .astype(float)
        )
        self.cluster_df["max_dup_r"] = self.cluster_df["max_dup_r"].map(
            lambda x: f"{x:.2f}"
        )
        logger.debug(
            "Successfully updated cluster_df with duplicates on the Main Thread."
        )

    def _save_pickle_with_fallback(self, data, path):
        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path))
        os.close(tmp_fd)
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, path)  # atomic move
            return path
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise

    def _sanitize_ei_dict(self, ei_dict):
        """
        Return a sanitized copy of the EI dict containing only entries with a
        valid numpy `ei` array. Keys are converted to ints when possible.
        """
        if not ei_dict:
            return {}
        out = {}
        for k, v in ei_dict.items():
            try:
                key = int(k)
            except Exception:
                key = k
            if v is None:
                continue
            ei_arr = getattr(v, "ei", None)
            if isinstance(ei_arr, np.ndarray) and ei_arr.size > 0:
                out[key] = v
            else:
                logger.warning("Skipping EI for key %s: invalid or empty EI data", k)
        return out

    def load_stim_timing(self):
        try:
            import retinanalysis.utils.datajoint_utils as dju

            self.block_id = dju.get_block_id_from_datafile(
                self.exp_name, self.datafile_name
            )
            self.d_timing = dju.get_epochblock_timing(self.exp_name, self.block_id)
            logger.debug("Loaded stimulus timing data successfully")
        except ImportError:
            logger.warning(
                "retinanalysis module not available, skipping stimulus timing data loading"
            )
            return
        except Exception:
            logger.warning("Failed to load stimulus timing data (optional module)")
            return

    def update_and_export_status(self, selected_ids, status):
        """
        Batch update status_df efficiently in O(n) time.
        """
        selected_ids = set(selected_ids)
        logger.debug("Marking %s: %s", status, selected_ids)

        if not selected_ids:
            return

        # Build all updates in memory first (O(n))
        updates = []
        for cid in selected_ids:
            set_ids = selected_ids if status == "Duplicate" else {cid}
            updates.append({"cluster_id": cid, "status": status, "set": set_ids})

        updates_df = pd.DataFrame(updates)

        # Single batch operation: remove old + add new (O(n) total)
        self.status_df = pd.concat(
            [
                self.status_df[~self.status_df["cluster_id"].isin(selected_ids)],
                updates_df,
            ],
            ignore_index=True,
        )

        self.update_cluster_df_with_status()
        self.export_status()

    def update_cluster_df_with_status(self):
        """
        Update the cluster_df 'status' column based on current status_df.
        Uses vectorized operations for efficiency.
        """
        if self.cluster_df.empty or self.status_df.empty:
            return

        # Reset all statuses to 'Original'
        self.cluster_df["status"] = "Original"
        self.cluster_df["set"] = None

        # Use vectorized merge for efficient update
        status_dict = self.status_df.set_index("cluster_id")[["status", "set"]].to_dict(
            "index"
        )
        for cluster_id, row_data in status_dict.items():
            if cluster_id in self.cluster_df["cluster_id"].values:
                idx = self.cluster_df[
                    self.cluster_df["cluster_id"] == cluster_id
                ].index[0]
                self.cluster_df.at[idx, "status"] = row_data["status"]
                self.cluster_df.at[idx, "set"] = row_data["set"]

    def export_status(self):
        """
        Export duplicate_sets to a JSON file in the Kilosort directory.
        """

        if self.status_df.empty:
            # Nothing to save
            return

        try:
            self.status_df.to_csv(self.status_csv, index=False)
            logger.debug(
                "Exported %d status entries to %s", len(self.status_df), self.status_csv
            )
        except Exception:
            logger.exception("Failed to export status entries")

    def load_status(self):
        """
        Load status df from a csv file in the Kilosort directory.
        Returns True if file was found and loaded, False otherwise.
        """

        if not self.status_csv.exists():
            return False

        try:
            status_df = pd.read_csv(self.status_csv)
            # Convert string representation of sets back to actual sets
            status_df["set"] = status_df["set"].apply(
                lambda x: set(map(int, x.strip("{}").split(",")))
            )
            self.status_df = status_df

            logger.debug("Loaded status csv: %s", self.status_csv)
            logger.debug(
                "Status counts: %s", self.status_df["status"].value_counts().to_dict()
            )

            self.update_cluster_df_with_status()

            return True
        except Exception:
            logger.exception("Failed to load duplicate sets")
            return False

    def load_kilosort_data(self):
        """
        Load Kilosort outputs using memory-mapped NumPy arrays and build a
        minimal cluster_df used by downstream code. Avoids eager copies and
        defers heavy operations (e.g. unwhitening) to later functions.
        Returns: (success: bool, message: str)
        """
        import numpy as np
        import pandas as pd

        ks = self.kilosort_dir

        # --- stimulus timing (DataJoint) — safe here because we're on a background thread ---
        import threading

        threading.Thread(target=self.load_stim_timing, daemon=True).start()

        try:
            # --- core spike arrays (memmap, view) --------------------------------
            st_path = ks / "spike_times.npy"
            sc_path = ks / "spike_clusters.npy"
            if not st_path.exists() or not sc_path.exists():
                return (
                    False,
                    "Missing spike_times.npy or spike_clusters.npy in kilosort dir.",
                )

            # mmap_mode="r" avoids loading the full arrays into RAM upfront
            # (~200–800 MB for a 1-hour recording). Per-cluster access via
            # cluster_spike_indices does a single fancy-index copy of only that
            # cluster's spikes, so UI scrolling is unaffected; the OS page
            # cache keeps hot pages warm across accesses.
            self.spike_times = np.load(st_path).ravel()
            self.spike_clusters = np.load(sc_path).ravel()

            # --- channel positions / map -----------------------------------------
            chan_pos_path = ks / "channel_positions.npy"
            if chan_pos_path.exists():
                self.channel_positions = np.load(chan_pos_path, mmap_mode="r")
            else:
                self.channel_positions = None

            channel_map_path = ks / "channel_map.npy"
            if channel_map_path.exists():
                self.channel_map = np.load(channel_map_path, mmap_mode="r")
            elif self.channel_positions is not None:
                self.channel_map = np.arange(self.channel_positions.shape[0], dtype=int)
            else:
                self.channel_map = None

            # sorted channels (safe if channel_positions is None)
            try:
                self.sorted_channels = (
                    sort_electrode_map(self.channel_positions)
                    if self.channel_positions is not None
                    else None
                )
            except Exception:
                self.sorted_channels = None

            # --- templates and related small-index files (memmap) ----------------
            templates_path = ks / "templates.npy"
            self.templates = (
                np.load(templates_path, mmap_mode="r")
                if templates_path.exists()
                else None
            )

            templates_ind_path = ks / "templates_ind.npy"
            self.templates_ind = (
                np.load(templates_ind_path, mmap_mode="r")
                if templates_ind_path.exists()
                else None
            )

            # --- spike amplitudes (optional) ------------------------------------
            amplitudes_path = ks / "amplitudes.npy"
            self.spike_amplitudes = (
                np.load(amplitudes_path).ravel() if amplitudes_path.exists() else None
            )

            # --- cluster info / fallback ----------------------------------------
            info_path = ks / "cluster_info.tsv"
            group_path = ks / "cluster_group.tsv"
            if info_path.exists():
                self.info_path = info_path
                self.cluster_info = pd.read_csv(info_path, sep="\t")
            elif group_path.exists():
                self.info_path = group_path
                self.cluster_info = pd.read_csv(group_path, sep="\t")
            else:
                self.info_path = None
                # _unique_ids/_counts computed once below — reuse here
                self.cluster_info = None  # filled after the single np.unique call

            # --- single np.unique scan — reused everywhere below ----------------
            # Previously called 3 separate times (lines ~487, ~501, ~504).
            # np.unique on a multi-million element array is O(N log N); doing it
            # once and threading the results through saves 2 full sort passes.
            _unique_ids, _counts = np.unique(self.spike_clusters, return_counts=True)

            # Fill in the fallback cluster_info now that we have _unique_ids
            if self.cluster_info is None:
                self.cluster_info = pd.DataFrame(
                    {
                        "cluster_id": _unique_ids.astype(int),
                        "group": ["unsorted"] * len(_unique_ids),
                    }
                )

            # --- build minimal cluster_df (used heavily elsewhere) ---------------
            if "cluster_id" in self.cluster_info.columns:
                cluster_ids = self.cluster_info["cluster_id"].astype(int).values
                if "group" in self.cluster_info.columns:
                    status_vals = self.cluster_info["group"].astype(str).values
                else:
                    status_vals = (
                        self.cluster_info.get(
                            "status", pd.Series(["unsorted"] * len(cluster_ids))
                        )
                        .astype(str)
                        .values
                    )
            else:
                cluster_ids = _unique_ids.astype(int)
                status_vals = np.array(["unsorted"] * len(cluster_ids), dtype=object)

            counts_map = dict(
                zip(_unique_ids.astype(int).tolist(), _counts.astype(int).tolist())
            )
            n_spikes = np.array(
                [int(counts_map.get(int(cid), 0)) for cid in cluster_ids], dtype=int
            )

            self.cluster_df = pd.DataFrame(
                {
                    "cluster_id": cluster_ids.astype(int),
                    "n_spikes": n_spikes,
                    "status": status_vals,
                    "x_um": np.nan,
                    "y_um": np.nan,
                }
            )

            self.cluster_id_to_idx = {
                int(cid): int(i)
                for i, cid in enumerate(self.cluster_df["cluster_id"].values)
            }

            # --- optional: channel-template mappings ----------------------------
            try:
                if self.templates is not None:
                    d_mappings = get_channel_template_mappings(self.templates)
                    self.channel_to_templates = d_mappings.get(
                        "channel_to_templates", {}
                    )
                    self.template_to_channels = d_mappings.get(
                        "template_to_channels", {}
                    )
                else:
                    self.channel_to_templates = {}
                    self.template_to_channels = {}
            except Exception:
                self.channel_to_templates = {}
                self.template_to_channels = {}

            # --- load params and similarity -------------------------------------
            try:
                self._load_kilosort_params()
            except Exception:
                logger.debug(
                    "Failed to load kilosort params (non-fatal).", exc_info=True
                )

            try:
                self._load_kilosort_similarity()
            except Exception:
                logger.debug(
                    "Failed to load kilosort similarity (non-fatal).", exc_info=True
                )

            # NOTE: Caches (standard_plot_cache, feature_cache) are intentionally
            # NOT loaded here. They are loaded at the end of build_cluster_dataframe(),
            # after cluster_df has been finalized, so stale keys from a previous
            # session can be pruned against the real cluster population.

            # --- Build O(1) spike index lookup + cache sort order for reuse ---
            logger.debug("Building cluster index mapping...")
            order = np.argsort(self.spike_clusters, kind="mergesort")
            sorted_cls = self.spike_clusters[order]
            sorted_t = self.spike_times[order]

            self._spk_sorted_cls = sorted_cls
            self._spk_sorted_t = sorted_t
            _, split_idxs = np.unique(sorted_cls, return_index=True)
            grouped_indices = np.split(order, split_idxs[1:])
            unique_cls = sorted_cls[split_idxs]

            self.cluster_spike_indices = {
                int(cid): idxs for cid, idxs in zip(unique_cls, grouped_indices)
            }

            # _unique_ids / _counts already computed above — no repeat scan needed.
            # unique_cls is derived from sorted_cls which covers the same set,
            # so we reuse _counts directly (order matches since both come from
            # the same np.unique on spike_clusters).
            self._spk_unique_cls = unique_cls
            self._spk_unique_counts = _counts

            return True, "Successfully loaded Kilosort data."

        except Exception as e:
            return False, f"Error during Kilosort data loading: {e}"

    def load_vision_data(self, vision_dir, dataset_name):
        """
        Loads EI, STA, and params data from a specified Vision directory.

        IMPORTANT: Heavy EI correlation computations are now deferred and
        run lazily the first time Vision similarity is requested, so this
        call returns much faster.
        """
        logger.debug("Starting vision data load from %s", vision_dir)
        vision_path = Path(vision_dir)

        # Use the high-level helper in vision_integration
        logger.debug("Calling vision_integration.load_vision_data")
        vision_data = vision_integration.load_vision_data(vision_path, dataset_name)
        logger.debug("Completed vision_integration.load_vision_data call")

        success = False

        if vision_data:
            # --- Full load path (EI + STA + params) ---
            ei_bundle = vision_data.get("ei")
            if ei_bundle:
                self.vision_eis = ei_bundle.get("ei_data")
                _raw_elec_map = ei_bundle.get("electrode_map")
                if _raw_elec_map is not None and isinstance(_raw_elec_map, np.ndarray):
                    if not np.all(np.isfinite(_raw_elec_map)):
                        logger.warning(
                            "EI electrode_map contains non-finite coordinates; "
                            "discarding vision_channel_positions to prevent UI crash."
                        )
                    elif np.max(np.abs(_raw_elec_map)) > 100_000:
                        logger.warning(
                            "EI electrode_map coordinates unreasonably large "
                            "(>100 000 µm); discarding vision_channel_positions."
                        )
                    else:
                        self.vision_channel_positions = _raw_elec_map
                        logger.debug(
                            "vision_channel_positions set: shape=%s",
                            self.vision_channel_positions.shape,
                        )
                if self.vision_eis:
                    logger.debug(
                        "Available Vision EI IDs (sample): %s",
                        list(self.vision_eis.keys())[:10],
                    )

            self.vision_stas = vision_data.get("sta")
            self.vision_params = vision_data.get("params")

            # Extract and store stimulus dimensions for coordinate alignment
            if self.vision_stas and len(self.vision_stas) > 0:
                # Get the first available STA to extract dimensions
                first_cell_id = next(iter(self.vision_stas))
                first_sta = self.vision_stas[first_cell_id]

                # The STA structure is likely a container with red, green, blue
                # channels
                if hasattr(first_sta, "red") and first_sta.red is not None:
                    sta_shape = first_sta.red.shape
                    if len(sta_shape) >= 2:
                        # Dimensions are [height, width, timepoints]
                        self.vision_sta_height = sta_shape[0]
                        self.vision_sta_width = sta_shape[1]
                    else:
                        # If we only have 2 dimensions, they are likely
                        # [height, width]
                        self.vision_sta_height = sta_shape[0]
                        self.vision_sta_width = sta_shape[1]
                else:
                    # Fallback if red channel is not available
                    logger.warning(
                        "Could not extract dimensions from STA data, using defaults."
                    )
                    self.vision_sta_width = 100
                    self.vision_sta_height = 100
            else:
                # Fallback if no STA data is available
                logger.warning(
                    "No STA data available to extract dimensions, using defaults."
                )
                self.vision_sta_width = 100
                self.vision_sta_height = 100

            # NOTE: EI correlation matrices and duplicate detection
            # are now computed lazily in _compute_ei_correlations_if_needed().
            logger.info(
                "Vision data loaded; STA dimensions: %sx%s",
                self.vision_sta_width,
                self.vision_sta_height,
            )
            success = True

        else:
            # --- Partial load path (if the combined loader failed) ---
            logger.debug("Full vision loading failed; attempting partial load")

            # Check if params or STA files exist even if the full loading
            # failed
            params_path = vision_path / "sta_params.params"
            sta_path = vision_path / "sta_container.sta"

            if params_path.exists() or sta_path.exists():
                logger.debug("Found params/sta files; attempting partial load")

                # Try to load the existing files one by one using the available
                # functions
                vision_data = {}

                if params_path.exists():
                    logger.debug("Loading params data")
                    try:
                        params_data = vision_integration.load_params_data(
                            vision_path, dataset_name
                        )
                        vision_data["params"] = params_data
                        logger.info("Loaded Vision params data")
                    except Exception:
                        logger.exception("Error loading params")

                if sta_path.exists():
                    logger.debug("Loading STA data")
                    try:
                        sta_data = vision_integration.load_sta_data(
                            vision_path, dataset_name
                        )
                        vision_data["sta"] = sta_data
                        logger.info("Loaded Vision STA data")

                        # Extract dimensions from STA if it was loaded
                        if sta_data:
                            first_cell_id = next(iter(sta_data))
                            first_sta = sta_data[first_cell_id]
                            if hasattr(first_sta, "red") and first_sta.red is not None:
                                sta_shape = first_sta.red.shape
                                if len(sta_shape) >= 2:
                                    self.vision_sta_height = sta_shape[0]
                                    self.vision_sta_width = sta_shape[1]
                                else:
                                    self.vision_sta_height = sta_shape[0]
                                    self.vision_sta_width = sta_shape[1]
                    except Exception:
                        logger.exception("Error loading STA data")

                # Update instance variables with any loaded data
                ei_bundle = vision_data.get("ei")
                if ei_bundle:
                    self.vision_eis = ei_bundle.get("ei_data")
                    _raw_elec_map = ei_bundle.get("electrode_map")
                    if _raw_elec_map is not None and isinstance(
                        _raw_elec_map, np.ndarray
                    ):
                        if not np.all(np.isfinite(_raw_elec_map)):
                            logger.warning(
                                "EI electrode_map (partial load) contains non-finite "
                                "coordinates; discarding vision_channel_positions."
                            )
                        elif np.max(np.abs(_raw_elec_map)) > 100_000:
                            logger.warning(
                                "EI electrode_map (partial load) coordinates unreasonably "
                                "large (>100 000 µm); discarding vision_channel_positions."
                            )
                        else:
                            self.vision_channel_positions = _raw_elec_map
                            logger.debug(
                                "vision_channel_positions set (partial load): shape=%s",
                                self.vision_channel_positions.shape,
                            )

                self.vision_stas = vision_data.get("sta")
                self.vision_params = vision_data.get("params")

                if vision_data:  # If we loaded any data
                    logger.info(
                        "Partial Vision data loaded; STA dimensions: %sx%s",
                        self.vision_sta_width,
                        self.vision_sta_height,
                    )
                    success = True
                else:
                    logger.debug("No vision data could be loaded")
            else:
                logger.debug("No vision files found in directory")

        # Mark vision as available if we have the required data
        self.vision_available = success and (
            self.vision_eis is not None or self.vision_stas is not None
        )

        logger.info(
            "Vision data load done — success=%s vision_available=%s "
            "vision_stas=%s vision_params=%s vision_eis=%s",
            success,
            self.vision_available,
            "None" if self.vision_stas is None else f"{len(self.vision_stas)} cells",
            "None" if self.vision_params is None else "present",
            "None" if self.vision_eis is None else f"{len(self.vision_eis)} cells",
        )

        return (
            success,
            f"{'Successfully' if success else 'Failed to'} load Vision data for {dataset_name}.",
        )

    def load_vision_native_data(self, vision_dir, dataset_name):
        """Fallback loader: builds master spike arrays purely from Vision data."""
        from pathlib import Path
        import numpy as np
        import pandas as pd
        from . import vision_integration
        import logging

        vision_path = Path(vision_dir)
        self.is_vision_only = True  # Flag this session!

        # 1. Load the Vision data
        vision_data = vision_integration.load_vision_data(vision_path, dataset_name)

        neurons_bundle = vision_data.get("neurons")
        if not neurons_bundle:
            return (
                False,
                "No .neurons file found. Cannot build standalone Vision dataset.",
            )

        # 2. Extract Spikes and Seed Electrodes
        spikes_dict = neurons_bundle["spikes_by_id"]
        seed_electrodes = neurons_bundle["seed_electrodes"]
        self.sampling_rate = float(neurons_bundle["sampling_rate"])

        # 3. Concatenate and Sort Spikes
        all_times = []
        all_clusters = []
        for vid, times in spikes_dict.items():
            all_times.append(times)
            all_clusters.append(np.full(len(times), vid))

        if not all_times:
            return False, "Neurons file is empty."

        self.spike_times = np.concatenate(all_times)
        self.spike_clusters = np.concatenate(all_clusters)

        sort_order = np.argsort(self.spike_times, kind="mergesort")
        self.spike_times = self.spike_times[sort_order]
        self.spike_clusters = self.spike_clusters[sort_order]

        if len(self.spike_times) > 0:
            self.n_samples = int(self.spike_times[-1]) + int(self.sampling_rate)
        else:
            self.n_samples = 0

        # ---------------------------------------------------------
        # 4. Load Geometry DIRECTLY from .globals
        # ---------------------------------------------------------
        self.channel_positions = None
        self.n_channels = 512
        try:
            import src.analysis.visionloader as vl

            with vl.GlobalsFileReader(str(vision_path), dataset_name) as gbfr:
                raw_positions, _ = gbfr.get_electrode_map()

                # OOM Killer Protection
                if not np.all(np.isfinite(raw_positions)):
                    logging.warning(
                        "Globals file contains NaN coordinates! Discarding to prevent UI crash."
                    )
                elif np.max(np.abs(raw_positions)) > 100000:
                    logging.warning(
                        "Globals file coordinates are unreasonably large! Discarding."
                    )
                else:
                    self.channel_positions = raw_positions
                    self.n_channels = self.channel_positions.shape[0]
                    # Mirror into vision_channel_positions so _resolve_channel_positions()
                    # returns consistent positions in vision-only mode.
                    self.vision_channel_positions = self.channel_positions
                    logger.debug(
                        "vision_channel_positions mirrored from .globals: shape=%s",
                        self.vision_channel_positions.shape,
                    )
        except Exception as e:
            logging.warning(f"Could not load .globals file for array geometry: {e}")

        # ---------------------------------------------------------
        # 5. Build Base Cluster DataFrame
        # ---------------------------------------------------------
        unique_ids, counts = np.unique(self.spike_clusters, return_counts=True)
        self.cluster_df = pd.DataFrame(
            {
                "cluster_id": unique_ids,
                "n_spikes": counts,
                "status": "Original",
                "x_um": np.nan,
                "y_um": np.nan,
                "best_chan": -1,
            }
        )

        self.cluster_id_to_idx = {
            int(cid): i for i, cid in enumerate(self.cluster_df["cluster_id"].values)
        }

        # Map the seed electrode to 0-indexed channel positions
        if self.channel_positions is not None:
            best_chans = {}
            x_dict = {}
            y_dict = {}

            for vid, seed_elec in seed_electrodes.items():
                cid = vid
                ch_idx = seed_elec - 1  # Seed electrodes are 1-indexed

                if 0 <= ch_idx < self.n_channels:
                    best_chans[cid] = ch_idx
                    x_dict[cid] = self.channel_positions[ch_idx, 0]
                    y_dict[cid] = self.channel_positions[ch_idx, 1]

            self.cluster_df["best_chan"] = (
                self.cluster_df["cluster_id"].map(best_chans).fillna(-1).astype(int)
            )
            self.cluster_df["x_um"] = self.cluster_df["cluster_id"].map(x_dict)
            self.cluster_df["y_um"] = self.cluster_df["cluster_id"].map(y_dict)

        # 6. Populate standard Vision attributes
        ei_bundle = vision_data.get("ei")
        self.vision_eis = ei_bundle.get("ei_data") if ei_bundle else None
        self.vision_stas = vision_data.get("sta")
        self.vision_params = vision_data.get("params")
        # vision_available mirrors the hybrid-path contract: True only when we have
        # at least EI or STA data.  vision_eis may legitimately be None (no .ei file
        # present) — that must not block loading; it just disables duplicate detection.
        self.vision_available = (
            self.vision_eis is not None or self.vision_stas is not None
        )

        self.templates = None
        self.templates_ind = None
        self.raw_data_memmap = None
        self.raw_reader = None
        self.cluster_info = pd.DataFrame()

        return True, "Successfully loaded Vision-native dataset."

    def precompute_ei_correlations_background(self):
        """
        BUG-6 fix: launch _compute_ei_correlations_if_needed() on a daemon
        thread so the UI is never blocked when Vision data is first loaded.
        Safe to call multiple times — the internal lock prevents double work.
        """
        if not hasattr(self, "_ei_corr_lock"):
            self._ei_corr_lock = threading.Lock()

        def _run():
            if not self._ei_corr_lock.acquire(blocking=False):
                return  # already running
            try:
                self._compute_ei_correlations_if_needed()
            finally:
                self._ei_corr_lock.release()

        t = threading.Thread(target=_run, daemon=True, name="EICorrelationWorker")
        t.start()

    def _compute_ei_correlations_if_needed(self):
        """
        Lazily compute EI correlation matrices and duplicate flags.

        This used to run inside load_vision_data() and block Vision loading.
        Now called from precompute_ei_correlations_background() (background
        thread) so the UI is never blocked.
        """
        if getattr(self, "is_vision_only", False):
            logger.info(
                "Vision dataset detected. Skipping duplicate EI correlation to prevent RAM explosion."
            )
            if hasattr(self, "ei_updates_ready"):
                self.ei_updates_ready.emit({}, {})
            return

        # Already computed / loaded?
        if self.ei_corr_dict is not None:
            return

        # Must have Vision EIs to do anything
        if self.vision_eis is None:
            logger.warning("Cannot compute EI correlations: vision_eis is None")
            # Thread-safe fallback: emit empty dicts so the main thread applies defaults
            if hasattr(self, "ei_updates_ready"):
                self.ei_updates_ready.emit({}, {})
            return

        str_corr_pkl = os.path.join(self.kilosort_dir, "ei_corr_dict.pkl")

        # Sanitize loaded EIs before any numeric operations
        sanitized_eis = self._sanitize_ei_dict(self.vision_eis)

        if len(sanitized_eis) < 2:
            logger.warning(
                "Not enough Vision EIs to compute correlations; skipping duplicate detection"
            )
            # Thread-safe fallback: emit empty dicts so the main thread applies defaults
            if hasattr(self, "ei_updates_ready"):
                self.ei_updates_ready.emit({}, {})
            return

        # Try to load existing correlations from disk
        if os.path.exists(str_corr_pkl):
            try:
                logger.debug(
                    "Loading precomputed EI correlations from %s", str_corr_pkl
                )
                with open(str_corr_pkl, "rb") as f:
                    self.ei_corr_dict = pickle.load(f)
                logger.debug("Loaded EI correlations successfully")
                full_corr = self.ei_corr_dict.get("full")
                space_corr = self.ei_corr_dict.get("space")
                power_corr = self.ei_corr_dict.get("power")
            except Exception as e:
                logger.warning(
                    "Failed to load EI correlation pickle: %s; recomputing", e
                )
                full_corr = ei_corr(
                    sanitized_eis, sanitized_eis, method="full", n_removed_channels=1
                )
                space_corr = ei_corr(
                    sanitized_eis, sanitized_eis, method="space", n_removed_channels=1
                )
                power_corr = ei_corr(
                    sanitized_eis, sanitized_eis, method="power", n_removed_channels=1
                )
                self.ei_corr_dict = {
                    "full": full_corr,
                    "space": space_corr,
                    "power": power_corr,
                }
                saved_path = self._save_pickle_with_fallback(
                    self.ei_corr_dict, str_corr_pkl
                )
                logger.debug("EI correlations recomputed and saved to %s", saved_path)
        else:
            # Compute from scratch
            logger.debug("Computing EI correlations")
            full_corr = ei_corr(
                sanitized_eis, sanitized_eis, method="full", n_removed_channels=1
            )
            space_corr = ei_corr(
                sanitized_eis, sanitized_eis, method="space", n_removed_channels=1
            )
            power_corr = ei_corr(
                sanitized_eis, sanitized_eis, method="power", n_removed_channels=1
            )
            self.ei_corr_dict = {
                "full": full_corr,
                "space": space_corr,
                "power": power_corr,
            }
            saved_path = self._save_pickle_with_fallback(
                self.ei_corr_dict, str_corr_pkl
            )
            logger.debug("EI correlations computed and saved to %s", saved_path)

        # With correlation matrices available, update cluster_df duplicate-related columns
        if not self.cluster_df.empty:
            cluster_ids = list(sanitized_eis.keys())
            # Convert to 0-based ONLY if Kilosort is the boss
            if not getattr(self, "is_vision_only", False):
                cluster_ids = np.array(cluster_ids) - 1
            else:
                cluster_ids = np.array(cluster_ids)

            potential_dups_map = {}
            max_dup_r_map = {}

            for i, cid in enumerate(cluster_ids):
                # Exclude self-comparison by masking the diagonal
                full_mask = np.delete(self.ei_corr_dict["full"][i, :], i)
                space_mask = np.delete(self.ei_corr_dict["space"][i, :], i)
                power_mask = np.delete(self.ei_corr_dict["power"][i, :], i)

                if (
                    np.any(full_mask > EI_CORR_THRESHOLD)
                    or np.any(space_mask > EI_CORR_THRESHOLD)
                    or np.any(power_mask > EI_CORR_THRESHOLD)
                ):
                    potential_dups_map[cid] = True

                    max_r = max(
                        np.max(full_mask) if full_mask.size > 0 else 0,
                        np.max(space_mask) if space_mask.size > 0 else 0,
                        np.max(power_mask) if power_mask.size > 0 else 0,
                    )
                    max_dup_r_map[cid] = max_r

            # --- EMIT SIGNAL TO MAIN THREAD ---
            if hasattr(self, "ei_updates_ready"):
                self.ei_updates_ready.emit(potential_dups_map, max_dup_r_map)
                logger.debug("Emitted EI correlations to main thread.")
            else:
                logger.error("Signal ei_updates_ready is missing! Cannot update UI.")

        else:
            logger.warning("cluster_df is empty; cannot update duplicate columns")

    def load_cell_type_file(self, txt_file: str = None):
        logger.debug("Loading cell type file: %s", txt_file)
        if txt_file is None:
            logger.debug("No cell type file provided; setting cell types to Unknown")
            # Drop existing cell_type column if exists
            if "cell_type" in self.cluster_df.columns:
                self.cluster_df.drop(columns=["cell_type"], inplace=True)
            return

        try:
            d_result = {}
            with open(txt_file, "r") as file:
                for line in file:
                    # Skip empty lines to prevent crashes
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        # Split each line into key and value using the specified delimiter
                        key, value = map(str.strip, line.split(" ", 1))
                        sub_values = value.split("/")

                        # --- NEW LOGIC: Respect Native Vision IDs ---
                        if getattr(self, "is_vision_only", False):
                            ks_id = int(key)  # Native Vision ID
                        else:
                            ks_id = int(key) - 1  # KS to Vision offset
                        # --------------------------------------------

                        for str_label in LS_CELL_TYPE_LABELS:
                            if str_label in sub_values:
                                d_result[ks_id] = str_label
                                break
                    except ValueError:
                        # Catch lines that don't split perfectly so it doesn't crash
                        logger.debug(f"Skipped unparseable line in text file: {line}")
                        continue

            # Add to cluster_df
            self.cluster_df["cell_type"] = (
                self.cluster_df["cluster_id"]
                .map(d_result)
                .fillna("Unknown")
                .infer_objects(copy=False)
            )
            logger.debug("Loaded cell type file: %s", txt_file)

            # If all are unknown, delete column and print
            if all(ct == "Unknown" for ct in self.cluster_df["cell_type"]):
                self.cluster_df.drop(columns=["cell_type"], inplace=True)
                logger.debug(
                    "All loaded cell types are Unknown; dropping cell_type column"
                )
        except Exception:
            logger.exception("Error loading cell type file")

    def _load_kilosort_params(self):
        params_path = self.kilosort_dir / "params.py"
        if not params_path.exists():
            raise FileNotFoundError("params.py not found.")
        params = {}
        with open(params_path, "r") as f:
            for line in f:
                if "=" in line:
                    key, val = map(str.strip, line.split("=", 1))
                    try:
                        # Use ast.literal_eval for safe parsing of literals
                        params[key] = ast.literal_eval(val)
                    except (ValueError, SyntaxError):
                        # Fallback: treat as a plain string (remove surrounding quotes if present)
                        params[key] = val.strip("'\"")
        self.sampling_rate = params.get("fs", 30000)
        self.n_channels = params.get("n_channels_dat", 512)
        dat_path_str = params.get("dat_path", "")
        if isinstance(dat_path_str, (list, tuple)) and dat_path_str:
            dat_path_str = dat_path_str[0]

    def set_dat_path(self, dat_path):
        """
        Set the path to the raw data source and prepare it for efficient access.

        Accepts either:
        - A directory path containing chunked Litke .bin files  →  opens a
          PyBinFileReader (preferred, native Litke format).
        - A single flat .dat/.bin file  →  falls back to a numpy memmap
          (legacy Kilosort-concatenated raw data).

        In both cases self.n_samples is updated so that downstream panels
        (e.g. RawPanel) continue to know the total recording duration.
        """
        dat_path = Path(dat_path)

        # --- close any previously opened reader to avoid file-handle leaks ---
        self._close_raw_reader()

        # --- Litke directory path: use PyBinFileReader ---
        if dat_path.is_dir() and _BIN2PY_AVAILABLE:
            logger.info(
                "set_dat_path: opening Litke bin directory via PyBinFileReader: %s",
                dat_path,
            )
            self.dat_path = dat_path
            self.raw_reader = bin2py.PyBinFileReader(str(dat_path), is_row_major=True)
            self.n_samples = self.raw_reader.length
            # raw_data_memmap is intentionally left None — get_raw_trace_snippet
            # will route through raw_reader instead.
            self.raw_data_memmap = None
            return

        # --- single flat file: legacy numpy memmap ---
        if not dat_path.exists():
            logger.warning("set_dat_path: path does not exist: %s", dat_path)
            return

        logger.info("set_dat_path: opening flat dat file via memmap: %s", dat_path)
        self.dat_path = dat_path
        file_size = dat_path.stat().st_size
        # Assuming int16 (2 bytes) per sample
        self.n_samples = file_size // (self.n_channels * 2)
        self.raw_data_memmap = np.memmap(
            dat_path, dtype=np.int16, mode="r", shape=(self.n_samples, self.n_channels)
        )

    def build_cluster_dataframe(self):
        """
        Build the cluster dataframe (backwards-compatible single-call).

        Optimizations:
        - ISI calculation is done in a vectorized pass when spike_times exist.
        """
        logger.debug("Starting build_cluster_dataframe")

        # --- basic counts — reuse cached results from load_kilosort_data ---
        # spike_clusters was already scanned + sorted there; don't repeat it.
        if hasattr(self, "_spk_unique_cls") and self._spk_unique_cls is not None:
            cluster_ids = self._spk_unique_cls
            n_spikes = self._spk_unique_counts
        else:
            cluster_ids, n_spikes = np.unique(self.spike_clusters, return_counts=True)
        logger.debug("Found %d clusters", len(cluster_ids))

        df = pd.DataFrame({"cluster_id": cluster_ids, "n_spikes": n_spikes})

        # Initialize quick columns (same as before)
        df["potential_dups"] = False
        df["max_dup_r"] = 0.0
        df["cell_type"] = "Unknown"
        df["isi_violations_pct"] = 0.0

        col = "KSLabel" if "KSLabel" in self.cluster_info.columns else "group"
        if col not in self.cluster_info.columns:
            self.cluster_info[col] = "unsorted"
        info_subset = self.cluster_info[["cluster_id", col]].rename(
            columns={col: "KSLabel"}
        )
        df = pd.merge(df, info_subset, on="cluster_id", how="left")
        # Any cluster present in spike_clusters but absent from cluster_info.tsv
        # gets KSLabel=NaN from the left-merge. Fill with 'unsorted' so they are
        # never silently excluded when downstream code filters by KSLabel.
        df["KSLabel"] = df["KSLabel"].fillna("unsorted").infer_objects(copy=False)

        df["status"] = "Original"
        df["set"] = [set([cid]) for cid in df["cluster_id"]]
        df["set"] = df["set"].astype(object)

        self.cluster_df = df[
            [
                "cluster_id",
                "cell_type",
                "n_spikes",
                "isi_violations_pct",
                "max_dup_r",
                "potential_dups",
                "status",
                "set",
                "KSLabel",
            ]
        ]
        self.cluster_df["cluster_id"] = self.cluster_df["cluster_id"].astype(int)
        self.original_cluster_df = self.cluster_df.copy()
        logger.debug("build_cluster_dataframe basic structure complete")

        # Reset ISI cache (BUG-5 fix: was reset twice here; now just once)
        self.isi_cache = {}

        # --- Vectorized ISI Computation (BUG-4 fix) ---
        # Old code: Python for-loop calling _calculate_isi_violations() per cluster.
        # New code: single argsort + diff pass over the full spike array, then
        # group-count violations — O(N log N) total instead of O(N*C) with loop overhead.
        logger.debug("Computing ISI violations per cluster (vectorized)...")

        refractory_samples = (ISI_REFRACTORY_PERIOD_MS / 1000.0) * self.sampling_rate

        # Reuse the sort order already computed in load_kilosort_data.
        # This avoids two more full scans + fancy-index reads of spike arrays.
        if hasattr(self, "_spk_sorted_cls") and self._spk_sorted_cls is not None:
            sorted_cls = self._spk_sorted_cls
            sorted_t = self._spk_sorted_t
        else:
            _order = np.argsort(self.spike_clusters, kind="stable")
            sorted_cls = self.spike_clusters[_order]
            sorted_t = self.spike_times[_order]

        isis = np.diff(sorted_t.astype(np.int64))
        same_cls = sorted_cls[:-1] == sorted_cls[1:]  # mask out cluster boundaries
        violations = (isis < refractory_samples) & same_cls
        spike_count = same_cls  # denominator: pairs within same cluster

        # Count violations and total pairs per cluster
        unique_ids, first_idx = np.unique(sorted_cls, return_index=True)
        # split_indices mark where each cluster starts in the sorted arrays
        viol_counts = np.array(
            [
                violations[i:j].sum()
                for i, j in zip(first_idx, np.append(first_idx[1:], len(violations)))
            ]
        )
        pair_counts = np.array(
            [
                spike_count[i:j].sum()
                for i, j in zip(first_idx, np.append(first_idx[1:], len(spike_count)))
            ]
        )

        isi_pct_map = {}
        for cid, viol, pairs in zip(unique_ids, viol_counts, pair_counts):
            val = float(viol / pairs * 100) if pairs > 0 else 0.0
            isi_pct_map[int(cid)] = val
            self.isi_cache[(int(cid), float(ISI_REFRACTORY_PERIOD_MS))] = val

        self.cluster_df["isi_violations_pct"] = (
            self.cluster_df["cluster_id"]
            .map(isi_pct_map)
            .fillna(0.0)
            .infer_objects(copy=False)
        )

        # --- Load status & compute remaining metrics ---
        self.load_status()

        # compute geometry and merge TSV metrics (keep existing code)
        try:
            self._compute_cluster_geometry()
        except Exception as e:
            logger.exception("Error computing cluster geometry: %s", e)

        try:
            self._merge_cluster_tsvs()
        except Exception as e:
            logger.exception("Error merging TSVs: %s", e)

        # NOTE: similar_templates.npy and mea_sorted_indices are loaded by
        # _load_kilosort_similarity(), called from load_kilosort_data().

        # --- Load persisted caches now that cluster_df is fully finalized ---
        # This is the correct place (not load_kilosort_data) because we can
        # now validate cache keys against the real cluster population and
        # prune any stale entries from a previous session (e.g. after refinement
        # changed cluster IDs).
        self.load_persisted_caches()

        valid_ids = set(self.cluster_df["cluster_id"].astype(int).tolist())

        with self._standard_plot_lock:
            stale = [k for k in list(self.standard_plot_cache) if k not in valid_ids]
            for k in stale:
                del self.standard_plot_cache[k]
            if stale:
                logger.debug("Pruned %d stale standard_plot_cache entries", len(stale))

        with self._feature_lock:
            stale = [
                k
                for k in list(self.feature_cache)
                if k not in valid_ids or not self.feature_cache[k].get("_computed")
            ]
            for k in stale:
                del self.feature_cache[k]
            self._physics_done_count = sum(
                1 for v in self.feature_cache.values() if v.get("_computed")
            )
            if stale:
                logger.debug("Pruned %d stale feature_cache entries", len(stale))

        logger.debug("build_cluster_dataframe complete")

    def get_cluster_spike_indices(self, cluster_id):
        """Return the indices into the master spike arrays for spikes of a cluster in O(1) time."""
        if not hasattr(self, "cluster_spike_indices"):
            # Fallback if somehow not initialized
            return np.where(self.spike_clusters == cluster_id)[0]
        return self.cluster_spike_indices.get(int(cluster_id), np.array([], dtype=int))

    def get_cluster_spikes(self, cluster_id):
        """Instantly fetch spike times using the precomputed indices."""
        inds = self.get_cluster_spike_indices(cluster_id)
        if len(inds) == 0:
            return np.array([])
        return self.spike_times[inds]

    def get_cluster_spike_amplitudes(self, cluster_id):
        """Return the per-spike amplitudes for a cluster (empty array if not available)."""
        if not hasattr(self, "spike_amplitudes") or self.spike_amplitudes is None:
            return np.array([])
        inds = self.get_cluster_spike_indices(cluster_id)
        return self.spike_amplitudes[inds]

    def get_standard_plot_data(self, cluster_id):
        """Return cached standard-plot data (ISI/ACG/FR) for a cluster.

        Heavy computations are done once per cluster and stored in
        self.standard_plot_cache, protected by self._standard_plot_lock.
        """
        # Ensure the cache attributes exist (for older sessions)
        if not hasattr(self, "standard_plot_cache"):
            self.standard_plot_cache = {}
            self._standard_plot_lock = threading.Lock()

        # 1. Fast path: check cache under lock
        with self._standard_plot_lock:
            cached = self.standard_plot_cache.get(cluster_id)
            if cached is not None:
                return cached

        # 2. Per-cell in-flight lock to prevent redundant computation
        # but ALLOW other clusters to be computed in parallel.
        if not hasattr(self, "_std_plot_cell_locks"):
            self._std_plot_cell_locks = {}
            self._std_plot_cell_locks_lock = threading.Lock()

        with self._std_plot_cell_locks_lock:
            if cluster_id not in self._std_plot_cell_locks:
                self._std_plot_cell_locks[cluster_id] = threading.Lock()
            cell_lock = self._std_plot_cell_locks[cluster_id]

        with cell_lock:
            # Re-check cache now that we have the per-cell lock
            with self._standard_plot_lock:
                cached = self.standard_plot_cache.get(cluster_id)
                if cached is not None:
                    return cached

            # Compute OUTSIDE the global lock (expensive)
            data = self._compute_standard_plots(cluster_id)

            # Store back under global lock
            with self._standard_plot_lock:
                self.standard_plot_cache[cluster_id] = data

        return data

    def try_get_standard_plot_data(self, cluster_id):
        """Non-blocking cache lookup — returns cached data or None.

        Unlike get_standard_plot_data(), this NEVER triggers computation.
        Safe to call from the UI/main thread without risking a freeze.
        """
        if not hasattr(self, "standard_plot_cache"):
            return None
        with self._standard_plot_lock:
            return self.standard_plot_cache.get(cluster_id)

    def save_standard_plot_cache(self):
        """Persist the full standard-plot and feature caches to disk in a background thread."""
        if not self.kilosort_dir:
            return

        with self._save_lock:
            if self._save_in_progress:
                return
            self._save_in_progress = True

        save_path = str(self.kilosort_dir / "standard_plot_cache.pkl")
        feature_save_path = str(self.kilosort_dir / "feature_cache.pkl")

        with self._standard_plot_lock:
            snapshot = dict(self.standard_plot_cache)

        with self._feature_lock:
            feature_snapshot = dict(getattr(self, "feature_cache", {}))

        def _save():
            try:
                self._save_pickle_with_fallback(snapshot, save_path)
                self._save_pickle_with_fallback(feature_snapshot, feature_save_path)
                logger.debug(
                    "Saved caches (%d std, %d feat) to disk",
                    len(snapshot),
                    len(feature_snapshot),
                )
            except Exception:
                logger.exception("Failed to persist standard_plot_cache")
            finally:
                with self._save_lock:
                    self._save_in_progress = False

        t = threading.Thread(target=_save, daemon=True)
        t.start()

    def _load_standard_plot_cache_from_disk(self):
        """Restore standard plot cache from disk, falling back to an empty cache.

        The lock is NOT held during pickle.load() so that concurrent
        get_standard_plot_data() callers are never blocked on disk I/O.
        Double-checked locking pattern:
          1. Acquire → guard check → release.
          2. Deserialise outside the lock.
          3. Acquire → re-check → assign → release.
        """
        if not self.kilosort_dir:
            return

        cache_pkl = self.kilosort_dir / "standard_plot_cache.pkl"

        # --- Step 1: cheap guard under lock ---
        with self._standard_plot_lock:
            if not cache_pkl.exists() or self.standard_plot_cache:
                return

        # --- Step 2: slow I/O outside the lock ---
        try:
            with open(cache_pkl, "rb") as f:
                cache = pickle.load(f)
            if not isinstance(cache, dict):
                raise TypeError("standard_plot_cache.pkl did not contain a dict")

            # Bloat guard: pre-fix caches stored spike-length raw arrays
            # ('spikes', 'spikes_sec', etc.) which caused OOM kills at ~68%
            # warm-up on large datasets.  Detect by inspecting the first entry
            # and discard the whole file if any bloated key is present.
            _BLOATED_KEYS = (
                "spikes",
                "spikes_sec",
                "spikes_ms",
                "isi_ms",
                "isi_vs_amp_valid_isi",
                "isi_vs_amp_valid_amplitudes",
                "fr_overlay_x",
                "fr_overlay_y",
            )
            first_entry = next(iter(cache.values()), {}) if cache else {}
            if any(
                k in first_entry and first_entry[k] is not None for k in _BLOATED_KEYS
            ):
                logger.warning(
                    "standard_plot_cache.pkl contains pre-fix bloated arrays — "
                    "discarding to prevent OOM. It will be rebuilt automatically."
                )
                cache = {}

            # FR downsampling guard: caches built before Fix 2 store one point
            # per second of recording (up to 3600 elements for a 1-hour session).
            # Detect by checking if fr_bin_centers exceeds the new cap and discard
            # so the cache is rebuilt with the compact representation.
            if cache:
                first_entry = next(iter(cache.values()), {})
                fr_centers = first_entry.get("fr_bin_centers")
                if (
                    fr_centers is not None
                    and hasattr(fr_centers, "__len__")
                    and len(fr_centers) > 300
                ):
                    logger.warning(
                        "standard_plot_cache.pkl has fat FR arrays (%d bins) — "
                        "discarding and rebuilding with compact representation.",
                        len(fr_centers),
                    )
                    cache = {}
        except (EOFError, pickle.UnpicklingError, OSError, AttributeError, TypeError):
            logger.warning("Could not load standard_plot_cache.pkl", exc_info=True)
            cache = {}

        # --- Step 3: assign under lock, double-checking no one beat us ---
        with self._standard_plot_lock:
            if not self.standard_plot_cache:  # another thread may have loaded first
                self.standard_plot_cache = cache
                logger.debug(
                    "Restored standard_plot_cache (%d entries) from disk",
                    len(self.standard_plot_cache),
                )

    def load_persisted_caches(self):
        """Loads both standard plot and feature caches from disk.
        Designed to be called safely from a background thread."""
        if not self.kilosort_dir:
            return

        self._load_standard_plot_cache_from_disk()

        feat_pkl = self.kilosort_dir / "feature_cache.pkl"
        with self._feature_lock:
            if feat_pkl.exists() and not self.feature_cache:
                try:
                    with open(feat_pkl, "rb") as f:
                        self.feature_cache = pickle.load(f)
                    self._physics_done_count = sum(
                        1 for v in self.feature_cache.values() if v.get("_computed")
                    )
                    logger.debug(
                        "Restored feature_cache (%d entries, %d computed) from disk",
                        len(self.feature_cache),
                        self._physics_done_count,
                    )
                except Exception:
                    logger.warning("Could not load feature_cache.pkl", exc_info=True)
                    self.feature_cache = {}
                    self._physics_done_count = 0
            else:
                self._physics_done_count = sum(
                    1 for v in self.feature_cache.values() if v.get("_computed")
                )

    def get_cell_physics(self, cluster_id):
        """
        Single Source of Truth for a cell's physical metrics.
        Assembles pre-calculated Vision data and caches it into a flat,
        instantly accessible dictionary to prevent redundant UI processing loops.
        """
        # Ensure thread safety for background loading
        if not hasattr(self, "_feature_lock"):
            self._feature_lock = threading.Lock()
        if not hasattr(self, "_physics_cell_locks"):
            self._physics_cell_locks = {}
            self._physics_cell_locks_lock = threading.Lock()

        # 1. Fast path: check feature cache under lock.
        with self._feature_lock:
            cached = self.feature_cache.get(cluster_id)
            if cached is not None and cached.get("_computed"):
                return cached

        # 2. Per-cell in-flight lock to prevent redundant computation
        with self._physics_cell_locks_lock:
            if cluster_id not in self._physics_cell_locks:
                self._physics_cell_locks[cluster_id] = threading.Lock()
            cell_lock = self._physics_cell_locks[cluster_id]

        with cell_lock:
            # 3. Re-check cache now that we hold the per-cell lock (double-check idiom)
            with self._feature_lock:
                cached = self.feature_cache.get(cluster_id)
                if cached is not None and cached.get("_computed"):
                    return cached

            # --- Everything below stays inside cell_lock so a second thread that
            #     also passed the fast-path miss cannot enter this block until the
            #     first thread has written to feature_cache. ---

            with self._feature_lock:
                _partial = self.feature_cache.get(cluster_id, {})
                acg_norm = _partial.get("acg")

            if acg_norm is None:
                std_data = self.get_standard_plot_data(cluster_id)
                acg_norm = std_data.get("acg_norm") if std_data else None

            timecourse = None
            rf_area = 0.0
            ellipticity = 0.0
            rf_long_diameter = 0.0
            rf_short_diameter = 0.0
            time_to_peak = 0

            vid = (
                cluster_id if getattr(self, "is_vision_only", False) else cluster_id + 1
            )
            if self.vision_stas and vid in self.vision_stas:
                sta_data = self.vision_stas[vid]
                stafit = None

                # Geometry (Extracting from Vision's pre-computed Gaussian fits)
                try:
                    if self.vision_params:
                        stafit = self.vision_params.get_stafit_for_cell(vid)
                        if stafit:
                            rf_area = np.pi * stafit.std_x * stafit.std_y
                            if stafit.std_x > 0:
                                ellipticity = stafit.std_y / stafit.std_x
                            # Long/short axis diameters (not just area+ratio):
                            # *2 matches the convention already used when
                            # drawing RF ellipses elsewhere (population_panel.py
                            # uses std_x*2/std_y*2 as ellipse width/height).
                            # max/min rather than x/y directly, since the
                            # fit's x/y axis assignment is arbitrary relative
                            # to which axis is actually longer — without this,
                            # "long diameter" could silently mean std_x for one
                            # cell and std_y for another.
                            axis_a = stafit.std_x * 2
                            axis_b = stafit.std_y * 2
                            rf_long_diameter = max(axis_a, axis_b)
                            rf_short_diameter = min(axis_a, axis_b)
                except Exception:
                    logger.debug(
                        "Failed to extract STA geometry for cluster %s",
                        cluster_id,
                        exc_info=True,
                    )

                # Timecourse (Pulls pre-computed 1D arrays from Vision params)
                try:
                    time_axis, tc_matrix, _ = analysis_core.get_sta_timecourse_data(
                        sta_data, stafit, self.vision_params, vid
                    )

                    if tc_matrix is not None and tc_matrix.size > 0:
                        energies = np.sum(tc_matrix**2, axis=0)
                        dom_idx = np.argmax(energies)
                        dom_trace = tc_matrix[:, dom_idx]

                        abs_max = np.max(np.abs(dom_trace))
                        if abs_max > 0:
                            timecourse = dom_trace / abs_max
                        else:
                            timecourse = dom_trace

                        time_to_peak = int(np.argmax(np.abs(timecourse)))
                except Exception:
                    logger.debug(
                        "Failed to extract STA timecourse for cluster %s",
                        cluster_id,
                        exc_info=True,
                    )

            # --- Fallback: borrow STA/RF from reference bridge ---
            # Mapping keys are Vision IDs. Params timecourse lookup MUST use
            # the reference Vision ID (not current vid) — Law 1 on ref table.
            elif self.reference_bridge and self.reference_bridge.has_match(vid):
                bridge = self.reference_bridge
                ref_id = bridge.get_reference_id(vid)
                sta_data = bridge.get_sta(vid) if bridge.has_sta(vid) else None
                stafit = bridge.get_stafit(vid)

                if stafit is not None:
                    try:
                        rf_area = np.pi * stafit.std_x * stafit.std_y
                        if stafit.std_x > 0:
                            ellipticity = stafit.std_y / stafit.std_x
                        axis_a = stafit.std_x * 2
                        axis_b = stafit.std_y * 2
                        rf_long_diameter = max(axis_a, axis_b)
                        rf_short_diameter = min(axis_a, axis_b)
                    except Exception:
                        logger.debug(
                            "Failed to extract borrowed RF geometry for cluster %s",
                            cluster_id,
                            exc_info=True,
                        )

                if sta_data is not None or stafit is not None:
                    try:
                        time_axis, tc_matrix, _ = analysis_core.get_sta_timecourse_data(
                            sta_data,
                            stafit,
                            bridge._ref_params,
                            ref_id if ref_id is not None else vid,
                        )
                        if tc_matrix is not None and tc_matrix.size > 0:
                            energies = np.sum(tc_matrix**2, axis=0)
                            dom_idx = np.argmax(energies)
                            dom_trace = tc_matrix[:, dom_idx]
                            abs_max = np.max(np.abs(dom_trace))
                            if abs_max > 0:
                                timecourse = dom_trace / abs_max
                            else:
                                timecourse = dom_trace
                            time_to_peak = int(np.argmax(np.abs(timecourse)))
                    except Exception:
                        logger.debug(
                            "Failed to extract borrowed STA timecourse for cluster %s",
                            cluster_id,
                            exc_info=True,
                        )

            # Provenance: which source supplied each STA-derived field
            used_current_sta = bool(
                self.vision_stas and vid in self.vision_stas
            )
            provenance = {
                "timecourse": None,
                "rf_geometry": None,
                "chirp": None,
                "grating": None,
            }
            if timecourse is not None:
                provenance["timecourse"] = "current" if used_current_sta else "reference"
            if rf_area and float(rf_area) > 0:
                provenance["rf_geometry"] = (
                    "current" if used_current_sta else "reference"
                )

            # Match metadata for this cell (UI id space)
            match_status = ""
            match_confidence = 0.0
            reference_id = None
            caveat = getattr(self, "match_caveats", {}).get(int(cluster_id))
            if caveat is not None:
                match_status = caveat.status or ""
                match_confidence = float(caveat.confidence or 0.0)
                reference_id = caveat.reference_id
                caveat.provenance["timecourse"] = provenance["timecourse"]
                caveat.provenance["rf_geometry"] = provenance["rf_geometry"]
            elif self.reference_bridge and self.reference_bridge.has_match(vid):
                match_status = self.reference_bridge.get_status(vid)
                match_confidence = self.reference_bridge.get_confidence(vid)
                reference_id = self.reference_bridge.get_reference_id(vid)

            metrics = {
                "_computed": True,
                "acg": acg_norm,
                "timecourse": timecourse,
                "rf_area": rf_area,
                "ellipticity": ellipticity,
                "rf_long_diameter": rf_long_diameter,
                "rf_short_diameter": rf_short_diameter,
                "time_to_peak": time_to_peak,
                "provenance": provenance,
                "match_status": match_status,
                "match_confidence": match_confidence,
                "reference_id": reference_id,
            }

            # 4. Store in global cache safely, even when Vision STA is missing.
            with self._feature_lock:
                already_computed = self.feature_cache.get(cluster_id, {}).get(
                    "_computed"
                )
                self.feature_cache[cluster_id] = metrics
                if not already_computed:
                    self._physics_done_count = (
                        getattr(self, "_physics_done_count", 0) + 1
                    )

            return metrics

    def get_physics_feature_matrix(
        self, cluster_ids, w_shape=2.0, w_pattern=1.5, w_geometry=1.0
    ):
        """
        Build the weighted PCA feature matrix used by UMAPWorker and FeatureAnalysisWorker.

        Reads pre-computed physics from feature_cache via get_cell_physics() — O(1)
        cache hit if the background pass has run; triggers on-demand computation otherwise.
        """
        valid_ids = []
        tc_list = []
        acg_list = []
        scalars_list = []
        metadata = {"Time to Peak": [], "RF Area": [], "Ellipticity": []}

        for cid in cluster_ids:
            metrics = self.get_cell_physics(int(cid))
            tc = metrics.get("timecourse")
            acg = metrics.get("acg")
            if tc is None or acg is None:
                continue

            valid_ids.append(cid)
            tc_list.append(tc)
            acg_list.append(acg)

            area = metrics.get("rf_area") or 0.0
            ellip = metrics.get("ellipticity") or 0.0
            t2p = metrics.get("time_to_peak") or 0
            scalars_list.append([area, ellip])
            metadata["Time to Peak"].append(t2p)
            metadata["RF Area"].append(area)
            metadata["Ellipticity"].append(ellip)

        if not valid_ids:
            return np.array([]), [], {}

        max_tc = max(len(t) for t in tc_list)
        tc_mat = np.array(
            [
                np.pad(t, (0, max_tc - len(t))) if len(t) < max_tc else t[:max_tc]
                for t in tc_list
            ]
        )
        max_acg = max(len(a) for a in acg_list)
        acg_mat = np.array(
            [
                np.pad(a, (0, max_acg - len(a))) if len(a) < max_acg else a[:max_acg]
                for a in acg_list
            ]
        )
        scalars_mat = np.array(scalars_list, dtype=np.float64)

        nan_mask = (
            np.any(np.isnan(tc_mat), axis=1)
            | np.any(np.isnan(acg_mat), axis=1)
            | np.any(np.isnan(scalars_mat), axis=1)
        )
        if np.any(nan_mask):
            logger.warning(
                "get_physics_feature_matrix: dropping %d unit(s) with NaN features",
                int(nan_mask.sum()),
            )
            keep = ~nan_mask
            valid_ids = [v for v, k in zip(valid_ids, keep) if k]
            tc_mat = tc_mat[keep]
            acg_mat = acg_mat[keep]
            scalars_mat = scalars_mat[keep]
            for key in metadata:
                metadata[key] = [v for v, k in zip(metadata[key], keep) if k]

        if not valid_ids:
            return np.array([]), [], {}

        if scalars_mat.shape[0] > 0:
            scalars_mat = RobustScaler().fit_transform(scalars_mat)

        n_comp_tc = min(3, len(valid_ids), tc_mat.shape[1])
        n_comp_acg = min(3, len(valid_ids), acg_mat.shape[1])

        tc_pca = (
            PCA(n_components=n_comp_tc).fit_transform(tc_mat)
            if n_comp_tc > 0
            else np.zeros((len(valid_ids), 0))
        )
        acg_pca = (
            PCA(n_components=n_comp_acg).fit_transform(acg_mat)
            if n_comp_acg > 0
            else np.zeros((len(valid_ids), 0))
        )

        feature_matrix = np.hstack(
            [
                tc_pca * w_shape,
                acg_pca * w_pattern,
                scalars_mat * w_geometry,
            ]
        )

        return feature_matrix, valid_ids, metadata

    def ensure_physics_cache(self, cluster_ids, max_workers=1):
        """
        Guarantee that all cluster_ids have a fully-computed feature_cache entry
        (_computed == True) before returning.
        """
        with self._feature_lock:
            missing = [
                int(cid)
                for cid in cluster_ids
                if not self.feature_cache.get(int(cid), {}).get("_computed")
            ]

        if not missing:
            return

        logger.debug(
            "ensure_physics_cache: %d cells not yet computed, filling now",
            len(missing),
        )

        # Per-cell timeout: the STA read inside get_cell_physics can hang on a
        # corrupt byte offset in the .sta file. LazySTADict.__getitem__ already has
        # an 8-second inner timeout; we add a 15-second outer ceiling here so even
        # if multiple slow paths compound (ACG compute + STA seek), one bad cell
        # never stalls the entire 1600-cell warm-up queue indefinitely.
        _CELL_TIMEOUT_S = 15

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.get_cell_physics, cid): cid for cid in missing
            }
            for future in as_completed(futures):
                cid = futures[future]
                try:
                    future.result(timeout=_CELL_TIMEOUT_S)
                except TimeoutError:
                    future.cancel()
                    logger.warning(
                        "ensure_physics_cache: timed out after %ds for cluster %d; "
                        "writing empty entry so UMAP gate can pass.",
                        _CELL_TIMEOUT_S,
                        cid,
                    )
                    # Write a minimal _computed entry so valid_cached count advances
                    # and the UMAP "Please wait" gate doesn't block forever.
                    with self._feature_lock:
                        if not self.feature_cache.get(cid, {}).get("_computed"):
                            self.feature_cache[cid] = {
                                "_computed": True,
                                "acg": None,
                                "timecourse": None,
                                "rf_area": 0.0,
                                "ellipticity": 0.0,
                                "time_to_peak": 0,
                            }
                            self._physics_done_count = (
                                getattr(self, "_physics_done_count", 0) + 1
                            )
                except Exception:
                    logger.warning(
                        "ensure_physics_cache: failed for cluster %d",
                        cid,
                        exc_info=True,
                    )

    def get_raw_feature_blocks(self, cluster_ids, filter_config):
        """
        Extract raw feature arrays for the dynamic clustering pipeline.

        This is the data-access counterpart to analysis_core.build_feature_matrix().
        It calls get_cell_physics() per cluster (Law 1 handled internally), applies
        the pre-filter, and returns raw arrays as *copies* — never cache references.

        Parameters
        ----------
        cluster_ids : list[int]
            Cluster IDs to process.
        filter_config : dict
            ``{'min_sta_std': float, 'max_rf_area': float}``

        Returns
        -------
        raw_blocks : dict
            ``{'temporal': np.ndarray (N, T), 'acg': np.ndarray (N, A),
              'grating': np.ndarray (N, grating_calc.POOLED_CURVE_N_BINS),
              'scalars': pd.DataFrame (N rows)}``
        valid_ids : list[int]
            Cluster IDs that passed pre-filtering.
        discarded_ids : list[int]
            Cluster IDs rejected by pre-filter.

        Thread safety
        -------------
        Acquires _feature_lock for cache reads via get_cell_physics().
        Safe to call from worker threads.
        """
        # 1. Collect physics for every cluster (triggers on-demand computation)
        physics_entries = {}
        for cid in cluster_ids:
            cid = int(cid)
            physics_entries[cid] = self.get_cell_physics(cid)

        # 2. Pre-filter via pure function in analysis_core
        valid_ids, discarded_ids = analysis_core.apply_prefilter(
            physics_entries, filter_config
        )

        if not valid_ids:
            empty_blocks = {
                "temporal": np.empty((0, 0)),
                "acg": np.empty((0, 0)),
                "grating": np.empty((0, 0)),
                "chirp": np.empty((0, 0)),
                "scalars": pd.DataFrame(),
            }
            return empty_blocks, valid_ids, discarded_ids

        # 3. Collect raw arrays for valid cells (always copy — Law 3)
        tc_list = []
        acg_list = []
        grating_list = []  # pooled direction-tuning curve shapes, for PCA
        chirp_list = []  # L2-normalized chirp PSTH shapes, for PCA
        scalar_rows = []

        # Chirp availability + fixed PSTH width (see
        # docs/specs/chirp_umap_feature_spec.md). chirp_id_to_row already has
        # the Vision→Kilosort offset baked in at load time (load_chirp_data),
        # so indexing it here is Law-1-safe and never re-derives the offset.
        # psth_mean is one fixed-width (n_cells, n_bins) array, so the sentinel
        # width is known up front — unlike grating, no per-cell interpolation.
        # Effective availability also includes reference-bridge chirp (fill-gap).
        from .constants import CHIRP_MIN_QI

        local_chirp = (
            getattr(self, "chirp_available", False)
            and getattr(self, "chirp_data", None) is not None
            and getattr(self, "chirp_id_to_row", None) is not None
        )
        bridge = getattr(self, "reference_bridge", None)
        bridge_chirp = bool(bridge and bridge.has_any_chirp())
        chirp_available = local_chirp or bridge_chirp
        if local_chirp:
            chirp_n_bins = int(np.asarray(self.chirp_data["psth_mean"]).shape[1])
        elif bridge_chirp:
            chirp_n_bins = int(
                np.asarray(bridge._ref_chirp_data["psth_mean"]).shape[1]
            )
        else:
            chirp_n_bins = 0

        # Recording duration, needed for the firing_rate metadata column
        # below. firing_rate/isi_violations are NOT embedding features
        # (removed from the weighted feature space — see constants.py
        # DEFAULT_WEIGHT_* comments), but UMAPWorker still surfaces them as
        # display/hover metadata on the scatter plot, so they need to keep
        # existing in raw_blocks['scalars'] even though build_feature_
        # matrix's scalar_features table no longer reads them for
        # clustering distance.
        recording_duration_sec = 0.0
        if (
            hasattr(self, "spike_times")
            and self.spike_times is not None
            and len(self.spike_times) > 1
        ):
            sr = float(getattr(self, "sampling_rate", 30000.0))
            recording_duration_sec = (
                float(self.spike_times[-1] - self.spike_times[0]) / sr
            )

        from . import grating_calc

        for cid in valid_ids:
            phys = physics_entries[cid]

            # Timecourse — must copy to prevent np.roll from mutating cache.
            # timecourse=None means no STA was available for this cell; use a
            # zero-length sentinel so the cell still contributes ACG + scalars.
            # build_feature_matrix detects all-zero tc_mat and skips the
            # temporal PCA block rather than producing NaN components.
            tc = phys.get("timecourse")
            if tc is not None:
                tc = np.array(tc, dtype=np.float64).copy()
            else:
                tc = np.zeros(0, dtype=np.float64)
            tc_list.append(tc)

            # ACG — copy for safety
            acg = phys.get("acg")
            if acg is not None:
                acg = np.array(acg, dtype=np.float64).copy()
            else:
                acg = np.zeros(1, dtype=np.float64)
            acg_list.append(acg)

            # Grating direction-tuning curve shape — for the EMBEDDING (PCA
            # block in build_feature_matrix, see GRATING_PCA_COMPONENTS).
            # Fill-gap: current analyzed/computed entry first, else bridge.
            grating_entry = None
            grating_source = None
            grating_status = getattr(self, "grating_status", "missing")
            grating_data = getattr(self, "grating_data", None)
            grating_cache = getattr(self, "grating_computed_cache", None) or {}
            grating_lock = getattr(self, "_grating_cache_lock", None)
            if grating_status == "ok" and grating_data is not None:
                grating_entry = grating_data.get(cid)
                if grating_entry is not None:
                    grating_source = "current"
            if grating_entry is None:
                if grating_lock is not None:
                    with grating_lock:
                        grating_entry = grating_cache.get(cid)
                else:
                    grating_entry = grating_cache.get(cid)
                if grating_entry is not None:
                    grating_source = "current"
            if grating_entry is None and bridge is not None:
                vid = self.get_vision_id_for_cluster(cid)
                if bridge.has_match(vid):
                    grating_entry = bridge.get_grating_entry(vid)
                    if grating_entry is not None:
                        grating_source = "reference"

            pooled_curve = (
                grating_calc.pooled_direction_tuning_curve(grating_entry)
                if grating_entry
                else None
            )
            if pooled_curve is not None:
                grating_list.append(np.asarray(pooled_curve, dtype=np.float64).copy())
            else:
                grating_list.append(
                    np.zeros(grating_calc.POOLED_CURVE_N_BINS, dtype=np.float64)
                )
                grating_source = None

            # Chirp PSTH SHAPE — fill-gap: current file first, else bridge.
            # L2-normalize; QI gate via CHIRP_MIN_QI; zero sentinel if missing.
            chirp_source = None
            if chirp_available:
                sentinel = np.zeros(chirp_n_bins, dtype=np.float64)
                used = False
                if local_chirp:
                    row = self.chirp_id_to_row.get(int(cid))
                    if row is not None:
                        qi = float(self.chirp_data["quality_index"][row])
                        if np.isfinite(qi) and qi >= CHIRP_MIN_QI:
                            psth = np.asarray(
                                self.chirp_data["psth_mean"][row], dtype=np.float64
                            ).copy()
                            norm = np.linalg.norm(psth)
                            chirp_list.append(psth / norm if norm > 0 else sentinel)
                            chirp_source = "current" if norm > 0 else None
                            used = True
                if not used and bridge_chirp:
                    vid = self.get_vision_id_for_cluster(cid)
                    row_pack = bridge.get_chirp_row(vid) if bridge.has_match(vid) else None
                    if row_pack is not None:
                        psth, qi = row_pack
                        if np.isfinite(qi) and qi >= CHIRP_MIN_QI:
                            psth = np.asarray(psth, dtype=np.float64).copy()
                            # Align width if reference bins differ (pad/truncate)
                            if len(psth) < chirp_n_bins:
                                psth = np.pad(psth, (0, chirp_n_bins - len(psth)))
                            elif len(psth) > chirp_n_bins:
                                psth = psth[:chirp_n_bins]
                            norm = np.linalg.norm(psth)
                            chirp_list.append(psth / norm if norm > 0 else sentinel)
                            chirp_source = "reference" if norm > 0 else None
                            used = True
                if not used:
                    chirp_list.append(sentinel)
            else:
                chirp_list.append(np.zeros(chirp_n_bins, dtype=np.float64))

            # Sync chirp/grating provenance onto match_caveats when present
            caveat = getattr(self, "match_caveats", {}).get(int(cid))
            if caveat is not None:
                caveat.provenance["chirp"] = chirp_source
                caveat.provenance["grating"] = grating_source

            # Scalar features for the EMBEDDING (fed into build_feature_matrix
            # via analysis_core.py's scalar_features table): RF long/short
            # diameter only now — DSI/OSI/peak_rate/angle moved to
            # metadata-only below (the grating PCA block above is what
            # feeds the embedding for direction/orientation tuning now).
            rf_long_diameter = float(phys.get("rf_long_diameter", 0.0))
            rf_short_diameter = float(phys.get("rf_short_diameter", 0.0))

            # Metadata-only scalars — NOT read by build_feature_matrix's
            # scalar_features table, but still needed here because
            # UMAPWorker.run() surfaces these as display/hover columns on
            # the scatter plot. rf_area/ellipticity/time_to_peak are
            # already computed and cached in get_cell_physics(); firing_rate/
            # isi_violations are computed fresh here since their prior
            # computation block was removed along with their embedding use.
            # grating_dsi/osi/peak_rate_hz/pref_angle_* are gated summary
            # scalars (from select_best_dsos_condition) kept for scatter-
            # plot coloring/hover ("this point is DSI=0.6") even though the
            # embedding itself now uses the pooled curve SHAPE, not these
            # scalars — see the block above.
            rf_area = float(phys.get("rf_area", 0.0))
            ellipticity = float(phys.get("ellipticity", 0.0))
            time_to_peak_meta = float(phys.get("time_to_peak", 0))

            firing_rate = 0.0
            if recording_duration_sec > 0:
                n_spikes = len(self.get_cluster_spike_indices(cid))
                firing_rate = n_spikes / recording_duration_sec

            isi_violations = 0.0
            if (
                not self.cluster_df.empty
                and "isi_violations_pct" in self.cluster_df.columns
            ):
                row = self.cluster_df.loc[
                    self.cluster_df["cluster_id"] == cid, "isi_violations_pct"
                ]
                if not row.empty:
                    isi_violations = float(row.iloc[0])

            # cos/sin rather than a raw degree value avoids the 359°/1°
            # discontinuity a linear angle feature would introduce — kept
            # here purely for display purposes (e.g. a future "color by
            # preferred angle" option), not read by the embedding.
            dsi = osi = peak_rate = 0.0
            angle_cos = angle_sin = 0.0
            if grating_entry:
                selection = grating_calc.select_best_dsos_condition(grating_entry)
                if selection is not None and selection["condition"] is not None:
                    classification = selection["classification"]
                    dsi = (
                        float(selection["DSI"])
                        if np.isfinite(selection["DSI"])
                        else 0.0
                    )
                    osi = (
                        float(selection["OSI"])
                        if np.isfinite(selection["OSI"])
                        else 0.0
                    )
                    peak_rate = float(selection.get("peak_rate_hz", 0.0) or 0.0)
                    pref_angle = (
                        selection["preferred_direction_deg"]
                        if classification == "DS"
                        else selection["preferred_orientation_deg"]
                    )
                    if np.isfinite(pref_angle):
                        angle_cos = float(np.cos(np.deg2rad(pref_angle)))
                        angle_sin = float(np.sin(np.deg2rad(pref_angle)))

            scalar_rows.append(
                {
                    "rf_long_diameter": rf_long_diameter,
                    "rf_short_diameter": rf_short_diameter,
                    # Metadata-only — not embedding features, see comments above.
                    "firing_rate": firing_rate,
                    "isi_violations": isi_violations,
                    "time_to_peak": time_to_peak_meta,
                    "rf_area": rf_area,
                    "ellipticity": ellipticity,
                    "grating_dsi": dsi,
                    "grating_osi": osi,
                    "grating_peak_rate_hz": peak_rate,
                    "grating_pref_angle_cos": angle_cos,
                    "grating_pref_angle_sin": angle_sin,
                }
            )

        # 4. Pad/truncate to uniform length
        max_tc = max((len(t) for t in tc_list), default=0)
        if max_tc == 0:
            # All cells have timecourse=None (no STA data at all).
            # Fill with a single zero column so tc_mat has a consistent shape;
            # build_feature_matrix will detect np.std==0 and skip temporal PCA.
            tc_mat = np.zeros((len(valid_ids), 1), dtype=np.float64)
        else:
            tc_mat = np.array(
                [
                    np.pad(t, (0, max_tc - len(t))) if len(t) < max_tc else t[:max_tc]
                    for t in tc_list
                ]
            )

        max_acg = max(len(a) for a in acg_list)
        acg_mat = np.array(
            [
                np.pad(a, (0, max_acg - len(a))) if len(a) < max_acg else a[:max_acg]
                for a in acg_list
            ]
        )

        # grating_list is already uniform-width (every row is exactly
        # grating_calc.POOLED_CURVE_N_BINS long by construction — either a
        # real pooled curve of that width, or a same-width zero sentinel),
        # unlike tc_list/acg_list which can have ragged per-cell lengths —
        # so no pad/truncate step is needed here.
        grating_mat = np.array(grating_list, dtype=np.float64)

        # chirp_list rows are all exactly chirp_n_bins wide (real normalized
        # PSTH or same-width zero sentinel), so np.array gives a clean
        # (N, chirp_n_bins) matrix — or (N, 0) when no chirp data is loaded,
        # which build_feature_matrix skips via its size==0 guard.
        chirp_mat = (
            np.array(chirp_list, dtype=np.float64)
            if chirp_n_bins > 0
            else np.empty((len(valid_ids), 0), dtype=np.float64)
        )

        scalars_df = pd.DataFrame(scalar_rows, index=range(len(valid_ids)))

        raw_blocks = {
            "temporal": tc_mat,
            "acg": acg_mat,
            "grating": grating_mat,
            "chirp": chirp_mat,
            "scalars": scalars_df,
        }
        return raw_blocks, valid_ids, discarded_ids

    def get_acg_data(self, cluster_id):
        """Convenience wrapper: return (time_lags_ms, acg_values)."""
        data = self.get_standard_plot_data(cluster_id)
        return data.get("acg_time_lags"), data.get("acg_norm")

    def _compute_standard_plots(self, cluster_id):
        """Internal helper that actually computes all standard-plot data.

        This mirrors the logic in StandardPlotsPanel.update_all for:
        - autocorrelation (ACG)
        - ISI histogram
        - firing rate + amplitude
        and packs the numeric results into a dict.
        """
        data = {
            "isi_hist_x": None,
            "isi_hist_y": None,
            "acg_time_lags": None,
            "acg_norm": None,
            "fr_bin_centers": None,
            "fr_rate": None,
            "fr_amp_x": None,
            "fr_amp_y": None,
            "fr_amp_ymax": None,
        }

        # Basic safety checks
        if not hasattr(self, "spike_times") or self.spike_times is None:
            return data
        if not hasattr(self, "spike_clusters") or self.spike_clusters is None:
            return data
        if getattr(self, "sampling_rate", 0) <= 0:
            return data

        # --- Gather spikes & amplitudes for this cluster ---
        spikes = self.get_cluster_spikes(cluster_id)
        spikes = np.asarray(spikes)

        if spikes.size == 0:
            return data

        sr = float(self.sampling_rate)

        # spikes_sec — full resolution, used for ISI and FR (both need the real timestamps).
        spikes_sec = spikes / sr
        # NOTE: spikes_ms is NOT computed here for the full array.
        # The ACG subsamples first (Fix 3: avoid allocating a 500k-element ms array
        # only to immediately throw 98% of it away for a 10k-spike subsample).

        # ISI vector (ms) — local only, not cached.
        # get_cluster_spikes() returns spikes in time order (Kilosort guarantee).
        if spikes.size > 1:
            isi_ms = np.diff(spikes) / sr * 1000.0

            if isi_ms.size > 0:
                hist_y, hist_x = np.histogram(isi_ms, bins=np.linspace(0, 50, 101))
                data["isi_hist_x"] = hist_x
                data["isi_hist_y"] = hist_y

        # Get per-spike amplitudes (may be empty)
        all_amplitudes = self.get_cluster_spike_amplitudes(cluster_id)
        all_amplitudes = np.asarray(all_amplitudes)

        # --- Autocorrelation (ACG) — Direct Lag Histogram on entire recording ---
        # Instead of a memory-heavy FFT on a dense array, we use a sliding window
        # approach with bincount. This is O(N * spikes_per_window) and uses the
        # FULL recording duration while remaining extremely fast for sparse spike trains.
        MIN_SPIKES = 50
        if spikes.size > MIN_SPIKES:
            MAX_LAG = 100  # ms — ±100 ms window, giving 201-sample ACG

            # Fix 3: subsample in sample-count space first, then convert to ms.
            # Old code converted all N spikes to ms and then subsampled, allocating
            # a full N-spike int array (up to ~4 MB for a 500k-spike cluster) that
            # was immediately discarded. Now we subsample the raw int64 sample counts
            # and do one small ms conversion on 10k elements instead.
            # Seeded on cluster_id → deterministic; same ACG every run, no cache surprises.
            t_raw = spikes  # already int64, already sorted (Kilosort guarantee)
            if len(t_raw) > 10_000:
                rng = np.random.default_rng(seed=int(cluster_id))
                idx = np.sort(rng.choice(len(t_raw), size=10_000, replace=False))
                t_raw = t_raw[idx]
            t = (t_raw / sr * 1000.0).astype(np.int64)

            # Compute lags: for each spike, find neighbors within MAX_LAG.
            # We iterate k neighbors away until the distance > MAX_LAG for all spikes.
            lags_sum = np.zeros(MAX_LAG + 1, dtype=np.float64)
            for k in range(1, 1000):  # support up to 10kHz firing rate in window
                if k >= len(t):
                    break
                diffs = t[k:] - t[:-k]
                valid = diffs <= MAX_LAG
                if not np.any(valid):
                    break
                lags_sum += np.bincount(diffs[valid], minlength=MAX_LAG + 1)

            # Construct symmetric ACG: [reverse(lags), 0, lags]
            # Index 0 of lags_sum is self-coincidence (distance 0), which we ignore for lag 0.
            acg = np.concatenate([lags_sum[:0:-1], [0], lags_sum[1:]])

            time_lags = np.arange(-MAX_LAG, MAX_LAG + 1, dtype=float)
            n_spikes_f = float(len(t))
            acg_norm = acg / n_spikes_f if n_spikes_f > 0 else acg

            data["acg_time_lags"] = time_lags
            data["acg_norm"] = acg_norm

            # Write ACG into feature_cache immediately so get_cell_physics()
            # does not need to recompute it. Partial entry — no _computed key.
            with self._feature_lock:
                existing = self.feature_cache.get(cluster_id)
                if existing is None:
                    self.feature_cache[cluster_id] = {"acg": acg_norm}
                elif not existing.get("_computed"):
                    existing["acg"] = acg_norm

        # --- Firing rate & amplitude over time ---
        if spikes_sec.size > 0:
            max_t = float(spikes_sec.max())
            if max_t <= 0:
                bins = np.array([0.0, 1.0], dtype=float)
            else:
                # 1-second bins from 0 to floor(max_t)+1
                bins = np.arange(0.0, max_t + 1.0, 1.0, dtype=float)

            counts, bin_edges = np.histogram(spikes_sec, bins=bins)
            bin_centers = bin_edges[:-1]

            if counts.size > 0:
                rate = gaussian_filter1d(counts.astype(float), sigma=5)
            else:
                rate = np.zeros_like(bin_centers, dtype=float)

            # Amplitude per bin — O(N) with bincount instead of the old
            # O(N × B) loop that did `all_amplitudes[bin_indices == b]` for
            # every bin b.  For a 1-hour recording with 3600 bins and 100k
            # spikes that loop was doing 360M element comparisons.
            amplitude_smoothed = None
            if all_amplitudes.size > 0 and bin_centers.size > 0:
                bin_indices = np.searchsorted(bins[1:], spikes_sec, side="left")
                bin_indices = np.clip(bin_indices, 0, len(bin_centers) - 1)

                amp_sums = np.bincount(
                    bin_indices,
                    weights=all_amplitudes.astype(float),
                    minlength=len(bin_centers),
                )
                amp_counts = np.bincount(bin_indices, minlength=len(bin_centers))
                with np.errstate(invalid="ignore"):
                    amplitude_binned = np.where(
                        amp_counts > 0, amp_sums / amp_counts, np.nan
                    )

                amplitude_binned = np.asarray(amplitude_binned, dtype=float)

                # Interpolate NaNs if needed
                if np.any(np.isnan(amplitude_binned)):
                    valid_idx = ~np.isnan(amplitude_binned)
                    if np.sum(valid_idx) > 1:
                        f = interp1d(
                            bin_centers[valid_idx],
                            amplitude_binned[valid_idx],
                            kind="linear",
                            bounds_error=False,
                            fill_value="extrapolate",
                        )
                        amplitude_binned = f(bin_centers)
                    elif np.sum(valid_idx) == 1:
                        amplitude_binned = np.full_like(
                            amplitude_binned, amplitude_binned[valid_idx][0]
                        )
                    else:
                        amplitude_binned = None

                if amplitude_binned is not None:
                    amplitude_smoothed = gaussian_filter1d(amplitude_binned, sigma=5)

            # Fix 2: All computation above runs at full 1-second resolution
            # (scientifically correct — Gaussian smoothing shape is preserved).
            # Before caching, downsample to at most FR_CACHE_MAX_BINS points.
            # A 1-hour recording has 3600 bins; the plot canvas is ~800px wide,
            # so storing 3600 float64 values per cluster per array wastes ~72 MB
            # across a 500-cluster dataset for zero visual benefit.
            # Downsampling with a uniform stride after smoothing keeps the curve
            # shape exact and reduces per-cluster FR cache size ~12×.
            FR_CACHE_MAX_BINS = 300
            if len(bin_centers) > FR_CACHE_MAX_BINS:
                step = max(1, len(bin_centers) // FR_CACHE_MAX_BINS)
                cache_centers = bin_centers[::step]
                cache_rate = rate[::step]
            else:
                step = 1
                cache_centers = bin_centers
                cache_rate = rate

            data["fr_bin_centers"] = cache_centers
            data["fr_rate"] = cache_rate

            if amplitude_smoothed is not None:
                data["fr_amp_x"] = cache_centers
                data["fr_amp_y"] = amplitude_smoothed[::step]

                # Use template PTP to set a sensible right-axis scale when available
                max_ptp = 1.0
                templates = getattr(self, "templates", None)
                try:
                    if templates is not None and cluster_id < templates.shape[0]:
                        ptp = templates[cluster_id].max(axis=0) - templates[
                            cluster_id
                        ].min(axis=0)
                        if ptp.size > 0:
                            max_ptp = float(ptp.max())
                    # Fallback: use amplitude range
                    if not np.isfinite(max_ptp) or max_ptp <= 0:
                        max_ptp = (
                            float(np.nanmax(amplitude_smoothed))
                            if np.nanmax(amplitude_smoothed) > 0
                            else 1.0
                        )
                except Exception:
                    max_ptp = (
                        float(np.nanmax(amplitude_smoothed))
                        if np.nanmax(amplitude_smoothed) > 0
                        else 1.0
                    )

                data["fr_amp_ymax"] = max_ptp * 1.1

        return data

    def get_cluster_mean_amplitude(self, cluster_id, method="mean"):
        """Return a scalar amplitude for the cluster (mean or median)."""
        amps = self.get_cluster_spike_amplitudes(cluster_id)
        if amps.size == 0:
            return 0.0
        if method == "median":
            return float(np.median(amps))
        return float(np.mean(amps))

    def get_lightweight_features(self, cluster_id):
        """
        Non-blocking cache check for lightweight features.

        This function NO LONGER calculates features. It only checks if they
        have already been computed and cached. The actual calculation is now
        handled by the FeatureWorker.
        """
        return self.ei_cache.get(cluster_id, None)

    # ──────────────────────────────────────────────────────────────────────────────
    # Paste this method into the DataManager class in data_manager.py,
    # directly after get_lightweight_features() (~line 1716).
    # It is a regular instance method — indented 4 spaces like all other methods.
    # ──────────────────────────────────────────────────────────────────────────────

    def get_channel_all_snippets(
        self,
        dom_chan: int,
        target_cluster_id: int,
        max_bg_spikes: int = 1500,
        snippet_window=(-20, 60),
    ) -> dict:
        """
        Return all spike waveforms on *dom_chan*, labelled by cluster.

        Two backends:
          1. Raw data (memmap / PyBinFileReader) — real snippets, preferred.
          2. Template fallback — synthetic waveforms, no disk I/O.

        Returns
        -------
        dict:
            "unit_waves"      : np.ndarray (n_unit, n_time) float32
            "bg_waves"        : np.ndarray (n_bg,   n_time) float32
            "bg_waves_by_cid" : dict[int, np.ndarray]  — per-cluster bg waves
            "unit_indices"    : np.ndarray (n_unit,) int64  — global spike idx
            "source"          : "raw" | "template" | "none"
        """
        import numpy as np
        from . import analysis_core

        snip_len_fallback = int(snippet_window[1] - snippet_window[0])

        empty = {
            "unit_waves": np.empty((0, snip_len_fallback), dtype=np.float32),
            "bg_waves": np.empty((0, snip_len_fallback), dtype=np.float32),
            "bg_waves_by_cid": {},
            "unit_indices": np.empty((0,), dtype=np.int64),
            "source": "none",
        }

        # ── 1. Find clusters that live on dom_chan ─────────────────────────────
        if "best_chan" not in self.cluster_df.columns:
            return empty

        chan_df = self.cluster_df[self.cluster_df["best_chan"] == dom_chan]
        if chan_df.empty:
            return empty

        # ── 2a. Raw data path ──────────────────────────────────────────────────
        raw_available = self.raw_reader is not None or self.raw_data_memmap is not None

        if raw_available:
            try:
                snip_len = int(snippet_window[1] - snippet_window[0])
                unit_waves_list = []
                unit_indices_list = []
                bg_waves_by_cid = {}  # {cid: ndarray (n, snip_len)}

                n_bg_clusters = max(1, len(chan_df) - 1)
                quota_per_bg = max(1, max_bg_spikes // n_bg_clusters)

                source = (
                    self.raw_reader
                    if self.raw_reader is not None
                    else self.raw_data_memmap
                )

                for row in chan_df.itertuples():
                    cid = int(row.cluster_id)
                    spike_indices = self.get_cluster_spike_indices(cid)
                    spike_times = self.spike_times[spike_indices]
                    n = len(spike_times)

                    if n == 0:
                        continue

                    if cid == target_cluster_id:
                        # Unit: subsample up to 1500, keep global indices
                        if n > 1500:
                            chosen = np.random.choice(n, 1500, replace=False)
                        else:
                            chosen = np.arange(n, dtype=np.intp)
                        st_chosen = spike_times[chosen].astype(np.int64)
                        idx_chosen = spike_indices[chosen]

                        raw = analysis_core.extract_snippets(
                            source,
                            st_chosen,
                            window=snippet_window,
                            n_channels=self.n_channels,
                        )  # (n_ch, snip_len, n_chosen)
                        if raw.shape[2] == 0:
                            continue
                        waves = raw[dom_chan, :, :].T  # (n_chosen, snip_len)
                        unit_waves_list.append(
                            waves.astype(np.float32) * self.uV_per_bit
                        )
                        unit_indices_list.append(idx_chosen)

                    else:
                        # Background cluster — subsample to quota
                        if n > quota_per_bg:
                            chosen = np.random.choice(n, quota_per_bg, replace=False)
                            st_chosen = spike_times[chosen].astype(np.int64)
                        else:
                            st_chosen = spike_times.astype(np.int64)

                        raw = analysis_core.extract_snippets(
                            source,
                            st_chosen,
                            window=snippet_window,
                            n_channels=self.n_channels,
                        )
                        if raw.shape[2] == 0:
                            continue
                        waves = (
                            raw[dom_chan, :, :].T.astype(np.float32) * self.uV_per_bit
                        )
                        bg_waves_by_cid[cid] = waves

                unit_waves = (
                    np.vstack(unit_waves_list)
                    if unit_waves_list
                    else np.empty((0, snip_len), dtype=np.float32)
                )
                unit_indices = (
                    np.concatenate(unit_indices_list)
                    if unit_indices_list
                    else np.empty((0,), dtype=np.int64)
                )
                bg_waves_all = (
                    np.vstack(list(bg_waves_by_cid.values()))
                    if bg_waves_by_cid
                    else np.empty((0, snip_len), dtype=np.float32)
                )

                return {
                    "unit_waves": unit_waves,
                    "bg_waves": bg_waves_all,
                    "bg_waves_by_cid": bg_waves_by_cid,
                    "unit_indices": unit_indices,
                    "source": "raw",
                }

            except Exception:
                logger.exception(
                    "get_channel_all_snippets raw path failed cid=%d chan=%d",
                    target_cluster_id,
                    dom_chan,
                )
                # fall through to template path

        # ── 2b. Template fallback (no disk I/O) ────────────────────────────────
        try:
            templates = getattr(self, "templates", None)  # (n_tpl, n_time, n_tCh)
            tmpl_ind = getattr(self, "templates_ind", None)
            if templates is None:
                return empty

            snip_len = templates.shape[1]
            unit_waves_list = []
            bg_waves_by_cid = {}

            n_bg_clusters = max(1, len(chan_df) - 1)
            quota_per_bg = max(5, max_bg_spikes // n_bg_clusters)

            for row in chan_df.itertuples():
                cid = int(row.cluster_id)
                if cid >= templates.shape[0]:
                    continue

                tpl = np.asarray(templates[cid], dtype=np.float32)  # (n_time, n_tCh)

                # Map dom_chan → local template channel index
                if tmpl_ind is not None and cid < tmpl_ind.shape[0]:
                    local = np.where(tmpl_ind[cid] == dom_chan)[0]
                    if len(local) == 0:
                        continue
                    wave = tpl[:, int(local[0])]
                else:
                    if dom_chan >= tpl.shape[1]:
                        continue
                    wave = tpl[:, dom_chan]

                n_spikes_row = int(row.n_spikes)
                noise_scale = float(np.std(wave)) or 1e-6

                if cid == target_cluster_id:
                    n_rep = min(500, max(20, n_spikes_row // 50))
                    noise = np.random.normal(
                        0, noise_scale * 0.12, (n_rep, snip_len)
                    ).astype(np.float32)
                    unit_waves_list.append(np.tile(wave, (n_rep, 1)) + noise)
                else:
                    n_rep = min(quota_per_bg, max(5, n_spikes_row // 100))
                    noise = np.random.normal(
                        0, noise_scale * 0.05, (n_rep, snip_len)
                    ).astype(np.float32)
                    bg_waves_by_cid[cid] = np.tile(wave, (n_rep, 1)) + noise

            unit_waves = (
                np.vstack(unit_waves_list)
                if unit_waves_list
                else np.empty((0, snip_len), dtype=np.float32)
            )
            bg_waves_all = (
                np.vstack(list(bg_waves_by_cid.values()))
                if bg_waves_by_cid
                else np.empty((0, snip_len), dtype=np.float32)
            )

            return {
                "unit_waves": unit_waves,
                "bg_waves": bg_waves_all,
                "bg_waves_by_cid": bg_waves_by_cid,
                "unit_indices": np.empty((0,), dtype=np.int64),
                "source": "template",
            }

        except Exception:
            logger.exception(
                "get_channel_all_snippets template path failed cid=%d chan=%d",
                target_cluster_id,
                dom_chan,
            )
            return empty

    def get_heavyweight_features(self, cluster_id):
        # Fast-path: check cache under lock
        with self._heavyweight_lock:
            if cluster_id in self.heavyweight_cache:
                return self.heavyweight_cache[cluster_id]

        # If not cached, compute without holding the lock (expensive op)
        lightweight_data = self.get_lightweight_features(cluster_id)
        if not lightweight_data:
            return None

        features = analysis_core.compute_spatial_features(
            lightweight_data["median_ei"], self.channel_positions, self.sampling_rate
        )

        # Store computed features under lock
        with self._heavyweight_lock:
            # Another thread may have computed it while we were working; prefer
            # existing
            if cluster_id not in self.heavyweight_cache:
                self.heavyweight_cache[cluster_id] = features

        return self.heavyweight_cache.get(cluster_id, features)

    def get_nearest_channels(self, central_channel_idx, n_channels=3):
        """
        Find the n_channels nearest channels to the central_channel_idx based on physical positions.
        Returns the indices of the nearest channels, ordered so the dominant channel can be placed in the center
        (e.g., [neighbor_1, dominant_channel, neighbor_2]).
        """
        if self.channel_positions is None:
            # If no channel positions are available, return consecutive
            # channels
            start_idx = max(0, central_channel_idx)
            end_idx = min(self.n_channels, start_idx + n_channels)
            return list(range(start_idx, end_idx))

        if central_channel_idx >= len(self.channel_positions):
            # Use the last available channel
            central_channel_idx = len(self.channel_positions) - 1

        # Calculate Euclidean distance from the central channel to all other
        # channels
        central_pos = self.channel_positions[central_channel_idx]
        distances = np.linalg.norm(self.channel_positions - central_pos, axis=1)

        # Get the indices of the n_channels closest channels (excluding the
        # central channel itself)
        # Exclude central channel at index 0
        nearest_indices = np.argsort(distances)[1 : min(n_channels + 1, len(distances))]

        # Create the list [neighbor_1, dominant_channel, neighbor_2] with the
        # dominant channel in the middle
        result = nearest_indices.tolist()
        # Insert the central channel in the middle
        result.insert(1, central_channel_idx)

        # Make sure we only return n_channels (default 3) total
        if len(result) > n_channels:
            result = result[:n_channels]

        return result

    def get_raw_trace_snippet(self, channel_indices, start_sample, end_sample):
        """
        Get a snippet of raw trace data for specified channels and time range,
        returned in microvolts with shape (n_channels, n_samples).

        Supports two backends:
        - PyBinFileReader (self.raw_reader): native Litke .bin format.  Channel
          index 0 in the Litke stream is the TTL channel, so a +1 offset is
          applied automatically so that callers use the same 0-based Kilosort
          channel numbering in both backends.
        - numpy memmap (self.raw_data_memmap): legacy flat .dat file.
        """
        # --- bounds / validation -------------------------------------------------
        start_sample = max(0, int(start_sample))
        end_sample = min(self.n_samples, int(end_sample))
        valid_channel_indices = [
            int(idx) for idx in channel_indices if 0 <= int(idx) < self.n_channels
        ]
        if start_sample >= end_sample or not valid_channel_indices:
            return np.array([]).reshape(0, 0)

        # --- PyBinFileReader backend (Litke native) ------------------------------
        if self.raw_reader is not None:
            num_samples = end_sample - start_sample
            try:
                # get_data returns (n_electrodes_with_ttl, num_samples) when
                # is_row_major=True.  Row 0 is the TTL channel, rows 1..N are
                # the real electrodes, so we add 1 to convert Kilosort 0-based
                # channel indices to Litke 1-based electrode indices.
                raw_block = self.raw_reader.get_data(start_sample, num_samples)
                # raw_block shape: (N_ELECTRODES_total, num_samples)
                litke_indices = [idx + 1 for idx in valid_channel_indices]
                raw_snippet = raw_block[litke_indices, :]  # (n_ch, n_samples)
            except Exception:
                logger.exception(
                    "PyBinFileReader.get_data failed for samples %d:%d",
                    start_sample,
                    end_sample,
                )
                return None

            return raw_snippet.astype(np.float32) * self.uV_per_bit

        # --- legacy memmap backend -----------------------------------------------
        if self.raw_data_memmap is None:
            return None

        raw_snippet = self.raw_data_memmap[
            start_sample:end_sample, valid_channel_indices
        ]
        uv_snippet = raw_snippet.astype(np.float32) * self.uV_per_bit
        return uv_snippet.T  # → (n_channels, n_samples)

    def _close_raw_reader(self):
        """Safely close the PyBinFileReader file handles, if any are open."""
        if self.raw_reader is not None:
            try:
                # PyBinFileReader supports the context-manager protocol;
                # calling __exit__ directly is the safest teardown path.
                self.raw_reader.__exit__(None, None, None)
            except Exception:
                logger.warning("Error closing PyBinFileReader", exc_info=True)
            finally:
                self.raw_reader = None

    def update_after_refinement(self, parent_id, new_clusters_data):
        self.is_dirty = True
        parent_indices = np.where(self.spike_clusters == parent_id)[0]
        self.cluster_df.loc[self.cluster_df["cluster_id"] == parent_id, "status"] = (
            "Refined (Parent)"
        )
        max_id = self.spike_clusters.max()
        new_rows = []
        for i, new_cluster in enumerate(new_clusters_data):
            new_id = max_id + 1 + i
            sub_indices = parent_indices[new_cluster["inds"]]
            self.spike_clusters[sub_indices] = new_id
            isi_violations = self._calculate_isi_violations(new_id)
            new_row = {
                "cluster_id": new_id,
                "KSLabel": "good",
                "n_spikes": len(sub_indices),
                "isi_violations_pct": isi_violations,
                "status": f"Refined (from C{parent_id})",
            }
            new_rows.append(new_row)
        self.cluster_df = pd.concat(
            [self.cluster_df, pd.DataFrame(new_rows)], ignore_index=True
        )
        # Refinement changes spike assignments; cached standard plots are now
        # stale.
        if hasattr(self, "standard_plot_cache"):
            with getattr(self, "_standard_plot_lock", threading.Lock()):
                self.standard_plot_cache.clear()

    def _calculate_isi_violations(
        self, cluster_id, refractory_period_ms=ISI_REFRACTORY_PERIOD_MS
    ):
        # Check if we already have the ISI calculation for this cluster in
        # cache
        cache_key = (cluster_id, refractory_period_ms)
        if cache_key in self.isi_cache:
            return self.isi_cache[cache_key]

        spike_times_cluster = self.get_cluster_spikes(cluster_id)
        if len(spike_times_cluster) < 2:
            isi_value = 0.0
        else:
            # spike_times_cluster comes from get_cluster_spikes() which reads the sorted Kilosort memmap.
            isis = np.diff(spike_times_cluster)
            refractory_period_samples = (
                refractory_period_ms / 1000.0
            ) * self.sampling_rate
            violations = np.sum(isis < refractory_period_samples)
            isi_value = (violations / (len(spike_times_cluster) - 1)) * 100

        # Cache the result
        self.isi_cache[cache_key] = isi_value
        return isi_value

    def save_tree_structure(self, file_path):
        """
        Save the current tree structure to a JSON file.
        """
        import json

        def serialize_item(item):
            """Recursively serialize a QStandardItem and its children."""
            item_data = {
                "text": item.text(),
                "data": item.data(),  # cluster_id for cells, None for groups
                "child_count": item.rowCount(),
            }

            if item.rowCount() > 0:
                item_data["children"] = []
                for i in range(item.rowCount()):
                    child_item = item.child(i)
                    item_data["children"].append(serialize_item(child_item))

            return item_data

        tree_data = []
        root_model = self.main_window.tree_model  # Access the main window's tree model

        for i in range(root_model.rowCount()):
            item = root_model.item(i)
            tree_data.append(serialize_item(item))

        with open(file_path, "w") as f:
            json.dump(tree_data, f, indent=2)

    def load_tree_structure(self, file_path):
        """
        Load the tree structure from a JSON file.
        """
        import json

        with open(file_path, "r") as f:
            tree_data = json.load(f)

        def deserialize_item(item_data):
            """Recursively deserialize an item and its children."""
            item = QStandardItem(item_data["text"])
            item.setEditable(False)

            # Set data (cluster_id for cells)
            item.setData(item_data["data"], Qt.ItemDataRole.UserRole)

            # For groups, enable drop
            if item_data["data"] is None:  # This is a group
                item.setDropEnabled(True)
            else:  # This is a cell
                item.setDropEnabled(False)

            # Add children if they exist
            if "children" in item_data and item_data["children"]:
                for child_data in item_data["children"]:
                    child_item = deserialize_item(child_data)
                    item.appendRow(child_item)

            return item

        # Clear the current tree
        self.main_window.tree_model.clear()

        # Populate the tree with loaded data
        for item_data in tree_data:
            item = deserialize_item(item_data)
            self.main_window.tree_model.appendRow(item)

        # Set the model to the tree view
        self.main_window.setup_tree_model(self.main_window.tree_model)
        self.main_window.tree_view.expandAll()

    def _compute_cluster_geometry(self):
        """
        Populate x_um, y_um, best_chan, and template_amp in cluster_df.

        Fast path (KS4): load cluster_positions.npy directly — one memmap
        read, zero computation.

        Fallback (KS2/3): vectorized PTP argmax across all templates at once —
        no Python loop, no unwhitening needed (whitened PTP gives the same
        argmax as unwhitened for finding the dominant channel).
        """
        ks_dir = self.kilosort_dir

        # Use already-loaded arrays — no redundant disk reads
        templates = getattr(self, "templates", None)
        templates_ind = getattr(self, "templates_ind", None)
        chan_pos = getattr(self, "channel_positions", None)
        cids = self.cluster_df["cluster_id"].values

        # ── best_chan + template_amp from self.templates (always computed first) ─
        if templates is not None:
            ptp = templates.max(axis=1) - templates.min(axis=1)  # (n_tpl, n_tCh)
            n_tpl = ptp.shape[0]
            best_local = ptp.argmax(axis=1)  # (n_tpl,) local index
            if templates_ind is not None:
                best_global = templates_ind[np.arange(n_tpl), best_local].astype(int)
            else:
                best_global = best_local  # dense layout: local == global channel
            ptp_at_best = ptp[np.arange(n_tpl), best_local]

            valid_tpl = (cids >= 0) & (cids < n_tpl)
            safe_cids = np.where(valid_tpl, cids, 0)
            best_chans = np.where(valid_tpl, best_global[safe_cids], -1)
            self.cluster_df["best_chan"] = best_chans
            self.cluster_df["template_amp"] = np.where(
                valid_tpl, ptp_at_best[safe_cids], np.nan
            )
            logger.debug("Computed best_chan + template_amp from self.templates")
        else:
            logger.warning("templates not loaded; best_chan unavailable")
            return

        # ── x_um / y_um: KS4 cluster_positions.npy fast path ─────────────────
        cluster_pos_path = ks_dir / "cluster_positions.npy"
        if cluster_pos_path.exists():
            try:
                cluster_pos = np.load(cluster_pos_path)  # (n_clusters, 2)
                valid = (cids >= 0) & (cids < len(cluster_pos))
                safe_cids2 = np.where(valid, cids, 0)
                self.cluster_df["x_um"] = np.where(
                    valid, cluster_pos[safe_cids2, 0], np.nan
                )
                self.cluster_df["y_um"] = np.where(
                    valid, cluster_pos[safe_cids2, 1], np.nan
                )
                logger.debug("Loaded x_um/y_um from cluster_positions.npy")
                return
            except Exception as e:
                logger.warning(
                    "Failed to load cluster_positions.npy (%s); falling back to channel_positions",
                    e,
                )

        # ── x_um / y_um fallback: look up best_chan in self.channel_positions ──
        if chan_pos is not None:
            valid_ch = (best_chans >= 0) & (best_chans < len(chan_pos))
            safe_chs = np.where(valid_ch, best_chans, 0)
            self.cluster_df["x_um"] = np.where(valid_ch, chan_pos[safe_chs, 0], np.nan)
            self.cluster_df["y_um"] = np.where(valid_ch, chan_pos[safe_chs, 1], np.nan)
            logger.debug("Computed x_um/y_um from channel_positions[best_chan]")
        else:
            logger.warning("channel_positions not loaded; x_um/y_um unavailable")

    def _build_cluster_to_template_map(self):
        """
        Returns dict: cluster_id -> template_index.
        For KS4, cluster ids and template indices are 0..n_clusters-1.
        For KS2/3 you can use spike_templates for spikes in each cluster.
        """
        # For KS4, assume cluster_id == template index
        # If you have a more accurate mapping already, keep that instead.
        n_templates = self.templates.shape[0] if hasattr(self, "templates") else None
        mapping = {}
        for cid in self.cluster_df["cluster_id"]:
            # For Kilosort4, cluster_id is the same as template index
            # For other versions, this mapping might be different
            mapping[cid] = int(cid) if 0 <= int(cid) < n_templates else -1

        return mapping

    def _merge_cluster_tsvs(self):
        """
        Merge standard Phy/Kilosort cluster tsvs into self.cluster_df.
        Do NOT rename existing columns; only add if missing.
        """
        import pandas as pd

        ks_dir = self.kilosort_dir

        tsvs = {
            "firing_rate_hz": "cluster_firing_rate.tsv",
            "contam_pct": "cluster_ContamPct.tsv",
            "amp_median": "cluster_Amplitude.tsv",  # or whatever name you like
        }

        for col, fname in tsvs.items():
            path = ks_dir / fname
            if path.exists() and col not in self.cluster_df.columns:
                try:
                    df = pd.read_csv(path, sep="\t")
                    # Assume df has columns ['cluster_id', 'value'] or similar
                    # adapt if your files have different schema
                    value_col = [c for c in df.columns if c != "cluster_id"][0]
                    cluster_id_to_value = df.set_index("cluster_id")[
                        value_col
                    ].to_dict()
                    self.cluster_df[col] = self.cluster_df["cluster_id"].map(
                        cluster_id_to_value
                    )
                except Exception as e:
                    logger.warning(f"Could not load {fname}: {e}")

    def _load_kilosort_similarity(self):
        """Load similar_templates.npy as a memmap without blocking the main thread."""
        import numpy as np

        ks_dir = self.kilosort_dir

        try:
            sim_path = ks_dir / "similar_templates.npy"
            if not sim_path.exists():
                raise FileNotFoundError

            # memmap for zero-copy access
            self.similar_templates = np.load(sim_path, mmap_mode="r")

            # cluster_id -> index lookup
            cluster_ids = self.cluster_df["cluster_id"].values
            self.cluster_id_to_idx = {
                int(cid): int(i) for i, cid in enumerate(cluster_ids)
            }

            # DELETED the massive argsort here. We will do it lazily.

            logger.debug(
                "Loaded similar_templates.npy shape=%s", self.similar_templates.shape
            )
        except FileNotFoundError:
            logger.warning("similar_templates.npy not found; MEA similarity disabled")
            self.similar_templates = None
            self.cluster_id_to_idx = None

    def get_similarity_table(self, cluster_id: int, source: str = "MEA"):
        """
        Get similarity table for a cluster from the specified source.
        """
        if source == "MEA":
            return self._get_mea_similarity_table(cluster_id)
        elif source == "vision":
            return self._get_vision_similarity_table(cluster_id)
        else:
            raise ValueError(f"Unknown sim source: {source}")

    def _get_mea_similarity_table(self, cluster_id: int, top_n: int = 50):
        """Fast, vectorized retrieval of MEA similarity table for cluster_id on the fly."""
        import numpy as np
        import pandas as pd

        cache_key = int(cluster_id)
        cached = self.mea_sim_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        if self.similar_templates is None:
            return pd.DataFrame([])

        # 1. Map requested cluster_id to its original Kilosort template index
        tpl_map = getattr(self, "cluster_to_template", None)
        if tpl_map is None:
            tpl_map = self._build_cluster_to_template_map()
            self.cluster_to_template = tpl_map

        t1 = int(tpl_map.get(int(cluster_id), -1))

        if t1 < 0 or t1 >= self.similar_templates.shape[0]:
            return pd.DataFrame([])

        # 2. Get the similarities between t1 and ALL OTHER templates
        sim_row = self.similar_templates[t1]

        # 3. Extract similarities for valid templates only
        cluster_ids = self.cluster_df["cluster_id"].values
        n_clusters = len(cluster_ids)
        t2_array = np.array([int(tpl_map.get(int(cid), -1)) for cid in cluster_ids])

        tpl_sims = np.zeros(n_clusters, dtype=float)
        valid_mask = (t2_array >= 0) & (t2_array < len(sim_row))
        tpl_sims[valid_mask] = sim_row[t2_array[valid_mask]]

        # Exclude the cluster itself
        self_mask = cluster_ids == cluster_id
        tpl_sims[self_mask] = -1.0

        # 4. Fast partial sort (argpartition) instead of full argsort
        actual_top_n = min(top_n, n_clusters)
        if actual_top_n == 0:
            return pd.DataFrame([])

        top_idx = np.argsort(-tpl_sims)[:actual_top_n]

        # 5. Build output DataFrame
        other_ids = cluster_ids[top_idx]
        other_sims = tpl_sims[top_idx]

        n_spikes_arr = (
            self.cluster_df["n_spikes"].values
            if "n_spikes" in self.cluster_df.columns
            else np.zeros(n_clusters, dtype=int)
        )
        status_arr = (
            self.cluster_df["status"].astype(str).values
            if "status" in self.cluster_df.columns
            else np.array([""] * n_clusters)
        )
        set_col = self.cluster_df["set"] if "set" in self.cluster_df.columns else None

        distances = np.full(actual_top_n, np.nan)
        if "x_um" in self.cluster_df.columns and "y_um" in self.cluster_df.columns:
            x = self.cluster_df["x_um"].values.astype(float)
            y = self.cluster_df["y_um"].values.astype(float)
            target_row = self.cluster_id_to_idx.get(int(cluster_id))
            if target_row is not None:
                dx = x[top_idx] - float(x[target_row])
                dy = y[top_idx] - float(y[target_row])
                distances = np.sqrt(dx * dx + dy * dy)

        rows = {
            "cluster_id": other_ids.astype(int),
            "n_spikes": n_spikes_arr[top_idx].astype(int),
            "status": status_arr[top_idx],
            "distance_um": distances,
            "template_sim": other_sims,
        }

        if set_col is not None:
            set_vals = set_col.values
            rows["set"] = [set_vals[i] for i in top_idx]

        result_df = pd.DataFrame(rows)
        self.mea_sim_cache[cache_key] = result_df
        return result_df.copy()

    def _get_vision_similarity_table(self, cluster_id: int, top_n: int = 50):
        """Get vision-based similarity table for cluster_id using STA correlations."""
        import numpy as np
        import pandas as pd

        cache_key = int(cluster_id)
        cached = self.vision_sim_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        if not self.vision_stas or not self.vision_params:
            return pd.DataFrame([])

        vision_cluster_id = (
            cluster_id if getattr(self, "is_vision_only", False) else cluster_id + 1
        )

        if vision_cluster_id not in self.vision_stas:
            return pd.DataFrame([])

        source_sta = self.vision_stas[vision_cluster_id]
        if not hasattr(source_sta, "red") or source_sta.red is None:
            return pd.DataFrame([])

        source_flat = np.stack(
            [source_sta.red, source_sta.green, source_sta.blue], axis=0
        ).flatten()
        source_std = np.std(source_flat)
        if source_std == 0:
            return pd.DataFrame([])

        similarities = {}
        for vid in self.vision_stas.keys():
            if vid == vision_cluster_id:
                continue
            sta_data = self.vision_stas[vid]
            if not hasattr(sta_data, "red") or sta_data.red is None:
                continue
            target_flat = np.stack(
                [sta_data.red, sta_data.green, sta_data.blue], axis=0
            ).flatten()
            if np.std(target_flat) == 0:
                continue
            corr = np.corrcoef(source_flat, target_flat)[0, 1]
            if not np.isnan(corr):
                similarities[vid] = corr

        if not similarities:
            return pd.DataFrame([])

        sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]

        cluster_ids, sim_values, n_spikes_list, status_list, set_list = (
            [],
            [],
            [],
            [],
            [],
        )
        for vid, sim in sorted_sims:
            cid = vid if getattr(self, "is_vision_only", False) else vid - 1
            cluster_ids.append(cid)
            sim_values.append(sim)
            row = self.cluster_df[self.cluster_df["cluster_id"] == cid]
            if not row.empty:
                n_spikes_list.append(
                    row["n_spikes"].values[0] if "n_spikes" in row.columns else 0
                )
                status_list.append(
                    row["status"].values[0] if "status" in row.columns else ""
                )
                set_list.append(row["set"].values[0] if "set" in row.columns else "")
            else:
                n_spikes_list.append(0)
                status_list.append("")
                set_list.append("")

        result_df = pd.DataFrame(
            {
                "cluster_id": cluster_ids,
                "n_spikes": n_spikes_list,
                "status": status_list,
                "set": set_list,
                "template_sim": sim_values,
            }
        )

        self.vision_sim_cache[cache_key] = result_df
        return result_df.copy()

    def load_chirp_data(self, chirp_path=None):
        """
        Load a chirp_analysis.py output .npy from kilosort_dir.

        chirp_analysis.py writes its output into the exact same directory
        DataManager treats as kilosort_dir (SORT_PATH/exp/chunk/algorithm),
        so no separate path needs to be configured — we just glob for it.

        cluster_id inside the file is Kilosort space (same as
        cluster_df['cluster_id']); no Vision ID translation is applied.

        Returns (success: bool, message: str). Never raises — on any failure
        chirp_available is set False and chirp_data/chirp_id_to_row are None,
        so callers can proceed without chirp data.
        """
        try:
            if chirp_path is None:
                candidates = sorted(self.kilosort_dir.glob("*Chirp*.npy"))
                if not candidates:
                    self.chirp_available = False
                    self.chirp_data = None
                    self.chirp_id_to_row = None
                    self.chirp_n_repeats = None
                    logger.info("No chirp analysis file found in %s", self.kilosort_dir)
                    return False, "No chirp analysis file found."
                chirp_path = candidates[0]

            mdic = np.load(chirp_path, allow_pickle=True).item()

            required_keys = ("psth_mean", "cluster_id", "quality_index", "bin_size_ms")
            missing = [k for k in required_keys if k not in mdic]
            if missing:
                self.chirp_available = False
                self.chirp_data = None
                self.chirp_id_to_row = None
                self.chirp_n_repeats = None
                logger.warning(
                    "Chirp file %s missing required keys: %s", chirp_path.name, missing
                )
                return False, f"Chirp file missing required keys: {missing}"

            self.chirp_data = mdic
            # Repeat count backs the trustworthiness of quality_index. It is
            # not stored as a scalar in the file, but spikes_binned is
            # (n_cells, n_trials, n_bins), so read it off there. Absent that
            # array the count is unknown, which is treated as untrustworthy
            # rather than assumed fine.
            binned = mdic.get("spikes_binned")
            self.chirp_n_repeats = (
                int(binned.shape[1])
                if binned is not None and np.asarray(binned).ndim == 3
                else None
            )
            self.chirp_id_to_row = {}
            for i, cid in enumerate(mdic["cluster_id"]):
                # Translate incoming Vision IDs (1-indexed) to Kilosort IDs (0-indexed)
                ks_id = cid if getattr(self, "is_vision_only", False) else cid - 1
                self.chirp_id_to_row[int(ks_id)] = i
            self.chirp_available = True
            n_cells = len(self.chirp_id_to_row)
            logger.debug(
                "Loaded chirp data from %s (%d cells)", chirp_path.name, n_cells
            )
            return True, f"Loaded chirp data from {chirp_path.name} ({n_cells} cells)"

        except Exception as e:
            logger.warning("Could not load chirp data: %s", e, exc_info=True)
            self.chirp_available = False
            self.chirp_data = None
            self.chirp_id_to_row = None
            self.chirp_n_repeats = None
            return False, str(e)

    def chirp_qi_is_trustworthy(self) -> bool:
        """Whether the loaded chirp run has enough repeats for its QI to mean
        anything. Unknown repeat count counts as untrustworthy."""
        return (
            self.chirp_n_repeats is not None
            and self.chirp_n_repeats >= CHIRP_MIN_REPEATS_FOR_QI
        )

    @staticmethod
    def _baden_quality_index(spikes_binned) -> np.ndarray:
        """Baden et al. (2016) response quality: Var_t(<C>_r) / <Var_t(C)>_r.

        ``spikes_binned`` is (n_cells, n_trials, n_bins). 1 means a perfectly
        reproducible response across repeats, 0 means noise.
        """
        x = np.asarray(spikes_binned, dtype=float)
        across = x.mean(axis=1).var(axis=1)     # variance of the trial-average
        within = x.var(axis=1).mean(axis=1)     # mean within-bin variance
        with np.errstate(divide="ignore", invalid="ignore"):
            qi = np.where(within > 0, across / within, 0.0)
        return np.nan_to_num(qi, nan=0.0, posinf=0.0)

    def attach_chirp_qi_column(self) -> bool:
        """Write the chirp quality index into ``cluster_df`` as ``chirp_qi``.

        The QI already existed in the loaded file and was used to gate the
        UMAP chirp block, but it never reached the table, so it could not be
        sorted on or eyeballed against the other per-unit quality metrics.

        **The value here is recomputed, not the file's ``quality_index``.**
        That stored field is on an unknown scale — across the chirp files on
        this drive it ranges roughly 200-2000 with a minimum near 217, and no
        linear transform maps it onto Baden's index (best fit R^2 ~0.87). It
        ranks almost identically (Spearman +0.999), so it is fine for
        ordering, but its absolute values cannot be compared against the
        literature, where the usual chirp cut is 0.45. Since ``spikes_binned``
        (n_cells, n_trials, n_bins) is present in every chirp file checked,
        the index is recomputed from it so the column means what its name
        says. Files lacking that array fall back to the stored value.

        Cells absent from the chirp file get NaN, which the table renders
        blank and sorts into its own group. Returns False (leaving cluster_df
        untouched) when there is no chirp data to attach.
        """
        if not self.chirp_available or self.chirp_data is None:
            return False
        if self.cluster_df is None or self.cluster_df.empty:
            return False

        binned = self.chirp_data.get("spikes_binned")
        if binned is not None and np.asarray(binned).ndim == 3:
            qi_all = self._baden_quality_index(binned)
            self.chirp_qi_is_baden = True
        else:
            qi_all = np.asarray(self.chirp_data["quality_index"], dtype=float)
            self.chirp_qi_is_baden = False
            logger.warning(
                "Chirp file has no spikes_binned; falling back to its stored "
                "quality_index, which is on an unknown scale — rank order is "
                "usable but absolute thresholds are not."
            )

        values = np.full(len(self.cluster_df), np.nan, dtype=float)
        for i, cid in enumerate(self.cluster_df["cluster_id"].to_numpy()):
            row = self.chirp_id_to_row.get(int(cid))
            if row is not None and 0 <= row < len(qi_all):
                values[i] = qi_all[row]

        self.cluster_df["chirp_qi"] = values
        logger.debug(
            "Attached chirp_qi to cluster_df: %d/%d cells, %d repeats (%s), "
            "source=%s",
            int(np.isfinite(values).sum()),
            len(values),
            self.chirp_n_repeats or -1,
            "trustworthy" if self.chirp_qi_is_trustworthy() else "TOO FEW REPEATS",
            "recomputed Baden" if self.chirp_qi_is_baden else "stored (unknown scale)",
        )
        return True

    # ------------------------------------------------------------------
    # STA quality
    # ------------------------------------------------------------------

    @staticmethod
    def sta_peak_to_rms(vol) -> float:
        """Peak-to-RMS ratio of an STA volume — how far the strongest stixel
        stands above the volume's own noise.

        A real receptive field is a compact blob in one or two frames and
        near-zero everywhere else, so its peak towers over the RMS. A noise
        STA has comparable energy in every stixel of every frame, so the two
        are close. On 20260715A/data010 the measure is sharply bimodal:
        noise cells sit near 4.5, cells with a receptive field near 35, with
        an empty valley between roughly 10 and 25.

        Scale-free, so Vision's peak-normalisation of the stored volumes does
        not affect it.
        """
        v = np.asarray(vol, dtype=np.float64)
        if v.size == 0:
            return float("nan")
        v = v - v.mean()
        rms = float(np.sqrt((v * v).mean()))
        if rms <= 0 or not np.isfinite(rms):
            return float("nan")
        return float(np.abs(v).max() / rms)

    def check_sta_consistency(self):
        """Does the .sta describe the cells in this sort?

        Returns ``(ok, message)``. The decisive check is the id sets: an STA
        for a cell the sort does not contain can only mean the STA file was
        written against an earlier version of that sort. This is not
        hypothetical — 20260715A/data007-010 ships an .sta holding 53 cells
        absent from its own .neurons, and its receptive fields are attached
        to the wrong units as a result.
        """
        if not self.vision_stas or self.cluster_df is None:
            return True, ""
        try:
            sta_ids = {int(i) for i in self.vision_stas.keys()}
        except Exception:
            return True, ""
        if not sta_ids:
            return True, ""

        vision_ids = {
            self.get_vision_id_for_cluster(int(c))
            for c in self.cluster_df["cluster_id"].to_numpy()
        }
        orphan = sta_ids - vision_ids
        if orphan:
            return False, (
                f"{len(orphan)} of {len(sta_ids)} cells in the .sta do not exist "
                f"in this sort. The STA file predates the spike sort, so "
                f"receptive fields are attached to the wrong units. Read STAs "
                f"from the noise run's own folder instead."
            )
        return True, ""

    def attach_sta_quality_column(self) -> bool:
        """Write ``sta_snr`` into ``cluster_df`` and check the .sta's provenance.

        Separates cells with a real receptive field from cells whose STA is
        noise — the two are indistinguishable from the KSLabel, which only
        reports spike-sorting quality and knows nothing about visual response.
        """
        if not self.vision_stas:
            return False
        if self.cluster_df is None or self.cluster_df.empty:
            return False

        ok, msg = self.check_sta_consistency()
        self.sta_ids_consistent = ok
        self.sta_consistency_message = msg
        if not ok:
            logger.warning("STA provenance: %s", msg)

        values = np.full(len(self.cluster_df), np.nan, dtype=float)
        for i, cid in enumerate(self.cluster_df["cluster_id"].to_numpy()):
            vid = self.get_vision_id_for_cluster(int(cid))
            try:
                sta = self.vision_stas[vid]
            except (KeyError, TypeError):
                continue
            if sta is None:
                continue
            # Green carries the luminance signal for the achromatic noise used
            # here; fall back if a file stores only one channel.
            vol = getattr(sta, "green", None)
            if vol is None:
                vol = getattr(sta, "red", None)
            if vol is None:
                continue
            values[i] = self.sta_peak_to_rms(vol)

        self.cluster_df["sta_snr"] = values
        n = int(np.isfinite(values).sum())
        logger.debug(
            "Attached sta_snr to cluster_df: %d/%d cells, %d above %.0f, "
            "provenance %s",
            n, len(values), int(np.nansum(values >= STA_SNR_GOOD)),
            STA_SNR_GOOD, "OK" if ok else "SUSPECT",
        )
        return True

    def get_chirp_data_for_cluster(self, cluster_id):
        """
        Return chirp PSTH data for a single cluster, or None if unavailable.

        Read-only dict/row lookup — no lock, no disk I/O. Safe to call from
        the main thread, but per AGENTS.md Law 2 this still must be invoked
        from Tier 2 (panel.update_all()), never Tier 1, since it backs a
        panel rebuild.

        Returns a dict with:
            psth_mean        : (n_bins,) ndarray, Hz
            quality_index    : float (NaN if silent)
            bin_size_ms      : float
            phase_step_on    : (start, end) bin indices
            phase_step_off   : (start, end) bin indices
            phase_freq_sweep : (start, end) bin indices
            phase_contrast   : (start, end) bin indices
            freq_min_hz      : float
            freq_max_hz      : float
        or None if chirp data isn't loaded or this cluster isn't in it.
        """
        if (
            not self.chirp_available
            or self.chirp_data is None
            or self.chirp_id_to_row is None
        ):
            return None

        row = self.chirp_id_to_row.get(int(cluster_id))
        if row is None:
            return None

        d = self.chirp_data
        return {
            "psth_mean": d["psth_mean"][row],
            "quality_index": float(d["quality_index"][row]),
            "bin_size_ms": float(d["bin_size_ms"]),
            "phase_step_on": tuple(d["phase_step_on"]),
            "phase_step_off": tuple(d["phase_step_off"]),
            "phase_freq_sweep": tuple(d["phase_freq_sweep"]),
            "phase_contrast": tuple(d["phase_contrast"]),
            "freq_min_hz": float(d["freq_min_hz"]),
            "freq_max_hz": float(d["freq_max_hz"]),
        }

    def _classify_grating_npy(self, mdic):
        """Returns 'analyzed', 'raw', or 'unknown'. Schema-based, since
        filenames don't reliably distinguish raw vs analyzed grating files."""
        if self._grating_looks_analyzed(mdic):
            return "analyzed"
        if "trial_parameters" in mdic and "spike_times_by_trial" in mdic:
            return "raw"
        return "unknown"

    @staticmethod
    def _grating_looks_analyzed(mdic):
        sample_vals = [v for k, v in mdic.items() if isinstance(k, (int, np.integer))]
        if not sample_vals:
            return False
        first = sample_vals[0]
        if not isinstance(first, dict):
            return False
        return any(
            isinstance(k, tuple) and isinstance(v, dict) and "condition_type" in v
            for k, v in first.items()
        )

    def load_grating_data(self, grating_path=None):
        """
        Load grating (DSOS/SF) data from kilosort_dir.

        Handles both cases: a precomputed analyzed file (DSI/OSI already
        run, e.g. combined_grating_analysis.py output), or only a raw file
        (spike_times_by_trial + trial_parameters) — DSI/OSI then get
        computed on demand per cluster via GratingComputeWorker.

        Never raises. Always leaves grating_available/grating_status in a
        well-defined state.
        """
        try:
            candidates = grating_path
            if candidates is None:
                candidates = sorted(self.kilosort_dir.glob("*Grating*.npy")) + sorted(
                    self.kilosort_dir.glob("*DSOS*.npy")
                )
                candidates = sorted(set(candidates))
                if not candidates:
                    self.grating_available = False
                    self.grating_status = "missing"
                    logger.info(
                        "No grating analysis file found in %s", self.kilosort_dir
                    )
                    return False, "No grating file found."
            else:
                candidates = [candidates]

            analyzed_path = None
            raw_path = None
            for p in candidates:
                try:
                    mdic = np.load(p, allow_pickle=True).item()
                except Exception:
                    continue
                kind = self._classify_grating_npy(mdic)
                if kind == "analyzed":
                    if analyzed_path is None or "_combined" in p.name:
                        analyzed_path = p
                elif kind == "raw" and raw_path is None:
                    raw_path = p

            if analyzed_path is not None:
                mdic = np.load(analyzed_path, allow_pickle=True).item()
                self.grating_data = {}
                for k, v in mdic.items():
                    if isinstance(k, (int, np.integer)):
                        ks_id = k if getattr(self, "is_vision_only", False) else k - 1
                        self.grating_data[int(ks_id)] = v
                self.grating_raw_data = None
                self.grating_available = True
                self.grating_status = "ok"
                self.grating_conditions = self._collect_grating_conditions(
                    self.grating_data
                )
                logger.debug(
                    "Loaded analyzed grating data from %s (%d cells)",
                    analyzed_path.name,
                    len(self.grating_data),
                )
                return True, f"Loaded analyzed grating data from {analyzed_path.name}"

            if raw_path is not None:
                mdic = np.load(raw_path, allow_pickle=True).item()
                raw_spikes = mdic["spike_times_by_trial"]

                # Re-key the raw trial dictionary to use Kilosort IDs
                ks_spikes = {}
                for k, v in raw_spikes.items():
                    ks_id = k if getattr(self, "is_vision_only", False) else k - 1
                    ks_spikes[int(ks_id)] = v

                self.grating_raw_data = {
                    "spike_times_by_trial": ks_spikes,
                    "trial_parameters": mdic["trial_parameters"],
                }
                self.grating_data = None
                self.grating_available = True
                self.grating_status = "raw_only"
                self.grating_computed_cache = {}
                self.grating_conditions = sorted(
                    set(
                        (t["barWidth"], t["temporalFrequency"])
                        for t in mdic["trial_parameters"]
                    )
                )
                logger.debug(
                    "Loaded raw grating data from %s (will compute DSI/OSI on demand)",
                    raw_path.name,
                )
                return (
                    True,
                    f"Loaded raw grating data from {raw_path.name} (unanalyzed)",
                )

            self.grating_available = False
            self.grating_status = "missing"
            logger.info(
                "Grating file(s) found in %s but none matched a known schema",
                self.kilosort_dir,
            )
            return False, "Grating file(s) found but schema not recognized."

        except Exception as e:
            logger.warning("Could not load grating data: %s", e, exc_info=True)
            self.grating_available = False
            self.grating_status = "missing"
            self.grating_data = None
            self.grating_raw_data = None
            return False, str(e)

    @staticmethod
    def _collect_grating_conditions(grating_data):
        conditions = set()
        for cid, per_cond in grating_data.items():
            for k in per_cond:
                if isinstance(k, tuple):
                    conditions.add(k)
        return sorted(conditions)

    def get_grating_data_for_cluster(self, cluster_id):
        """
        Tier 1-safe dict lookup ONLY — never computes. Checks the
        pre-analyzed file first, then the on-demand compute cache.

        Returns None if unavailable in either. If grating_status ==
        'raw_only' and this returns None, the caller (GratingPanel) is
        responsible for spawning a GratingComputeWorker — this method does
        not do that itself, to keep it non-blocking and side-effect-free.
        """
        cluster_id = int(cluster_id)

        if self.grating_status == "ok" and self.grating_data is not None:
            return self.grating_data.get(cluster_id)

        if self.grating_status == "raw_only":
            with self._grating_cache_lock:
                return self.grating_computed_cache.get(cluster_id)

        return None

    # Alias matching try_get_standard_plot_data's non-blocking-read naming.
    try_get_grating_data_for_cluster = get_grating_data_for_cluster

    def compute_grating_data_for_cluster(self, cluster_id):
        """
        Runs grating_calc.compute_grating_response() for one cluster and
        caches the result. Expensive (FFT + 1000-shuffle permutation test
        per DSOS condition) — must only be called from a background thread
        (GratingComputeWorker.run()), never from the main/GUI thread.
        """
        cluster_id = int(cluster_id)
        if self.grating_raw_data is None:
            return None

        from . import grating_calc

        result = grating_calc.compute_grating_response(
            cluster_id,
            self.grating_raw_data["spike_times_by_trial"],
            self.grating_raw_data["trial_parameters"],
        )

        with self._grating_cache_lock:
            self.grating_computed_cache[cluster_id] = result

        return result

    def get_vision_id_for_cluster(self, cluster_id: int) -> int:
        """Translate UI cluster_id → Vision file key, respecting is_vision_only."""
        if getattr(self, "is_vision_only", False):
            return int(cluster_id)
        return int(cluster_id) + 1

    def get_cluster_id_for_vision(self, vision_id: int) -> int:
        """Translate Vision file key → UI cluster_id.

        The exact inverse of get_vision_id_for_cluster. Anything reading an id
        back out of Vision data (a picked RF, a parsed classification file)
        must come through here rather than subtracting one by hand — the offset
        does not apply in vision-only mode and getting it wrong is silent.
        """
        if getattr(self, "is_vision_only", False):
            return int(vision_id)
        return int(vision_id) - 1

    # ------------------------------------------------------------------
    # Cross-run reference bridge install / availability
    # ------------------------------------------------------------------

    def install_reference_bridge(self, bridge, report=None):
        """
        Install a ReferenceBridge, build per-UI-id match caveats, and
        invalidate physics entries that can now fill-gap from the reference.

        Safe to call from the main thread after Map Reference Run accepts.
        """
        self.reference_bridge = bridge
        is_vo = bool(getattr(self, "is_vision_only", False))
        if bridge is not None:
            self.match_caveats = bridge.build_ui_caveats(is_vision_only=is_vo)
        else:
            self.match_caveats = {}
        self.invalidate_physics_for_reference_bridge()
        logger.info(
            "Installed reference bridge: %s (%d caveats)",
            bridge.summary() if bridge else "None",
            len(self.match_caveats),
        )
        return self.match_caveats

    def invalidate_physics_for_reference_bridge(self):
        """
        Clear sticky `_computed` physics for cells that lack STA-derived
        fields but can now borrow them from the reference bridge.

        Preserves ACG when present so we do not recompute pure spike metrics.
        """
        bridge = getattr(self, "reference_bridge", None)
        if bridge is None:
            return 0

        ids = set()
        with self._feature_lock:
            ids.update(int(k) for k in self.feature_cache.keys())
        if (
            hasattr(self, "cluster_df")
            and self.cluster_df is not None
            and not self.cluster_df.empty
            and "cluster_id" in self.cluster_df.columns
        ):
            ids.update(int(x) for x in self.cluster_df["cluster_id"].tolist())

        n_invalidated = 0
        with self._feature_lock:
            for cid in ids:
                vid = self.get_vision_id_for_cluster(cid)
                if not bridge.has_match(vid):
                    continue
                can_borrow = bridge.has_sta(vid) or bridge.has_rf(vid)
                if not can_borrow:
                    continue
                entry = self.feature_cache.get(cid)
                if entry is None:
                    continue
                if not entry.get("_computed"):
                    continue
                missing_tc = entry.get("timecourse") is None
                missing_rf = float(entry.get("rf_area") or 0.0) == 0.0
                # Only recompute when STA-derived fields are empty and the
                # bridge can supply them. Cells that already have native STA
                # products are left alone.
                if not (missing_tc or missing_rf):
                    continue
                acg = entry.get("acg")
                self.feature_cache[cid] = (
                    {"acg": acg, "_computed": False} if acg is not None else {}
                )
                n_invalidated += 1

        if n_invalidated:
            logger.info(
                "Invalidated %d physics cache entries for reference fill-gap",
                n_invalidated,
            )
        return n_invalidated

    def effective_chirp_available(self) -> bool:
        """True if current run or reference bridge has chirp product."""
        if getattr(self, "chirp_available", False):
            return True
        bridge = getattr(self, "reference_bridge", None)
        return bool(bridge and bridge.has_any_chirp())

    def effective_grating_available(self) -> bool:
        """True if current run or reference bridge has grating product."""
        if getattr(self, "grating_available", False):
            return True
        bridge = getattr(self, "reference_bridge", None)
        return bool(bridge and bridge.has_any_grating())

    def get_match_caveat(self, cluster_id: int):
        """Return CellMatchCaveat for a UI cluster_id, or None."""
        return getattr(self, "match_caveats", {}).get(int(cluster_id))