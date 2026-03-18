from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import threading
import numpy as np
import pandas as pd
from pathlib import Path
from qtpy.QtCore import QObject, Qt
from qtpy.QtGui import QStandardItem
from . import analysis_core
from . import vision_integration
from .constants import ISI_REFRACTORY_PERIOD_MS, EI_CORR_THRESHOLD, LS_CELL_TYPE_LABELS
import pickle
import os
import tempfile
import logging
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
        'channel_to_templates': channel_to_templates,
        'template_to_channels': template_to_channels
    }
    return d_out


def ei_corr(ref_ei_dict, test_ei_dict,
            method: str = 'full', n_removed_channels: int = 1) -> np.ndarray:
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
        ei_arr = getattr(entry, 'ei', None)
        if isinstance(ei_arr, np.ndarray) and ei_arr.size > 0:
            ref_eis.append(ei_arr)

    if n_removed_channels > 0:
        max_ref_vals = [np.array(np.max(ei, axis=1)) for ei in ref_eis]
        ref_to_remove = [np.argsort(val)[-n_removed_channels:]
                         for val in max_ref_vals]
        ref_eis = [np.delete(ei, ref_to_remove[idx], axis=0)
                   for idx, ei in enumerate(ref_eis)]

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
    if 'full' in method:
        ref_eis_flat = [ei.flatten() for ei in ref_eis]
        ref_eis = np.array(ref_eis_flat)
    # For 'time' method, take max of absolute value over time and
    # stack the resulting 512 x 1 vectors into a numpy array
    elif 'space' in method:
        ref_eis_mean = [np.max(np.abs(ei), axis=1) for ei in ref_eis]
        ref_eis = np.array(ref_eis_mean)
    # For 'power' method, square each 512 x 201 ei array, take the mean over time,
    # and stack the resulting 512 x 1 vectors into a numpy array
    elif 'power' in method:
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
        ei_arr = getattr(entry, 'ei', None)
        if isinstance(ei_arr, np.ndarray) and ei_arr.size > 0:
            test_eis.append(ei_arr)

    if n_removed_channels > 0:
        max_test_vals = [np.array(np.max(ei, axis=1)) for ei in test_eis]
        test_to_remove = [np.argsort(val)[-n_removed_channels:]
                          for val in max_test_vals]
        test_eis = [np.delete(ei, test_to_remove[idx], axis=0)
                    for idx, ei in enumerate(test_eis)]

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
    if 'full' in method:
        test_eis_flat = [ei.flatten() for ei in test_eis]
        test_eis = np.array(test_eis_flat)
    # For 'time' method, take max of absolute value over time and
    # stack the resulting 512 x 1 vectors into a numpy array
    elif 'space' in method:
        test_eis_mean = [np.max(np.abs(ei), axis=1) for ei in test_eis]
        test_eis = np.array(test_eis_mean)
    # For 'power' method, square each 512 x 201 ei array, take the mean over time,
    # and stack the resulting 512 x 1 vectors into a numpy array
    elif 'power' in method:
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
    d = np.mean(test_eis, axis=1)[:, None] * \
        np.mean(ref_eis, axis=1)[:, None].T
    covs = c - d

    std_calc = np.std(test_eis, axis=1)[
        :, None] * np.std(ref_eis, axis=1)[:, None].T
    # Avoid division by zero - set to 0 if std calculation is 0
    corr = np.divide(
        covs,
        std_calc,
        out=np.zeros_like(covs),
        where=std_calc != 0)

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

    def __init__(self, kilosort_dir, main_window=None):
        super().__init__()
        self.kilosort_dir = Path(kilosort_dir)
        self.exp_name = self.kilosort_dir.parent.parent.name
        self.datafile_name = self.kilosort_dir.parent.name
        self.d_timing = {}
        logger.debug(
            f"Initializing DataManager for experiment={self.exp_name}, datafile={self.datafile_name}")
        # NOTE: load_stim_timing() is intentionally NOT called here.
        # It makes a live DataJoint DB connection which can block for 30-120s
        # on a timeout. It is called inside load_kilosort_data() instead,
        # which runs on a background QThread.

        self.ei_cache = {}
        self.heavyweight_cache = {}
        self.feature_cache = {} # Cache for feature extraction panel (PCA, ACG, etc.)
        # Lock to protect accesses to heavyweight_cache from multiple threads
        self._feature_lock = threading.Lock()
        self._heavyweight_lock = threading.Lock()
        self.isi_cache = {}  # Cache for ISI violation calculations

        # Cache + lock for standard plots (ISI / ACG / FR)
        self.standard_plot_cache = {}
        self._standard_plot_lock = threading.Lock()

        self.dat_path = None

        self.new_class_id = 0
        self.cluster_df = pd.DataFrame()
        self.original_cluster_df = pd.DataFrame()
        self.info_path = None
        self.uV_per_bit = 0.195
        self.main_window = main_window  # Reference to main window for tree operations

        # status df
        self.status_df = pd.DataFrame(columns=['cluster_id', 'status', 'set'])
        self.status_df['set'] = self.status_df['set'].astype(object)
        self.status_csv = self.kilosort_dir / 'status.csv'

        self.mea_sorted_indices = None  # Pre-sorted indices for each cluster (set by _load_kilosort_similarity)
        self.cluster_id_to_idx = None  # Map cluster_id -> row index

        # --- Vision Data ---
        self.vision_eis = None
        self.vision_stas = None
        self.vision_params = None
        self.vision_channel_positions = None  # Store channel positions from vision data
        self.vision_sta_width = None  # Store stimulus width for coordinate alignment
        self.vision_sta_height = None  # Store stimulus height for coordinate alignment
        self.ei_corr_dict = None  # Initialize to None, will be set when vision data is loaded

        # --- MEA Similarity Data ---
        # (n_templates, n_templates) from Kilosort
        self.similar_templates = None
        self.cluster_to_template = None        # dict[int -> int]
        self.mea_sim_cache = {}                # cluster_id -> DataFrame
        self.vision_sim_cache = {}             # cluster_id -> DataFrame
        self.vision_available = False

        # Initialize raw data memmap attribute (kept for legacy/fallback paths)
        self.raw_data_memmap = None
        # PyBinFileReader instance — used when loading native Litke .bin files.
        # Takes priority over raw_data_memmap when set.
        self.raw_reader = None

        # Initialize refractory period (default from constants)
        self.refractory_period_ms = ISI_REFRACTORY_PERIOD_MS

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


    def _save_pickle_with_fallback(self, data, path):
        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path))
        os.close(tmp_fd)
        try:
            with open(tmp_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, path)   # atomic move
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
            ei_arr = getattr(v, 'ei', None)
            if isinstance(ei_arr, np.ndarray) and ei_arr.size > 0:
                out[key] = v
            else:
                logger.warning(
                    "Skipping EI for key %s: invalid or empty EI data", k)
        return out

    def load_stim_timing(self):
        try:
            import retinanalysis.utils.datajoint_utils as dju
            self.block_id = dju.get_block_id_from_datafile(
                self.exp_name, self.datafile_name)
            self.d_timing = dju.get_epochblock_timing(
                self.exp_name, self.block_id)
            logger.debug("Loaded stimulus timing data successfully")
        except ImportError:
            logger.warning("retinanalysis module not available, skipping stimulus timing data loading")
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
            set_ids = selected_ids if status == 'Duplicate' else {cid}
            updates.append({
                'cluster_id': cid,
                'status': status,
                'set': set_ids
            })

        updates_df = pd.DataFrame(updates)

        # Single batch operation: remove old + add new (O(n) total)
        self.status_df = pd.concat([
            self.status_df[~self.status_df['cluster_id'].isin(selected_ids)],
            updates_df
        ], ignore_index=True)

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
        self.cluster_df['status'] = 'Original'
        self.cluster_df['set'] = None

        # Use vectorized merge for efficient update
        status_dict = self.status_df.set_index('cluster_id')[['status', 'set']].to_dict('index')
        for cluster_id, row_data in status_dict.items():
            if cluster_id in self.cluster_df['cluster_id'].values:
                idx = self.cluster_df[self.cluster_df['cluster_id'] == cluster_id].index[0]
                self.cluster_df.at[idx, 'status'] = row_data['status']
                self.cluster_df.at[idx, 'set'] = row_data['set']

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
                "Exported %d status entries to %s", len(
                    self.status_df), self.status_csv)
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
            status_df['set'] = status_df['set'].apply(
                lambda x: set(map(int, x.strip("{}").split(","))))
            self.status_df = status_df

            logger.debug("Loaded status csv: %s", self.status_csv)
            logger.debug(
                "Status counts: %s",
                self.status_df['status'].value_counts().to_dict())

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
                return False, "Missing spike_times.npy or spike_clusters.npy in kilosort dir."

            # mmap_mode="r" avoids loading the full arrays into RAM upfront
            # (~200–800 MB for a 1-hour recording). Per-cluster access via
            # cluster_spike_indices does a single fancy-index copy of only that
            # cluster's spikes, so UI scrolling is unaffected; the OS page
            # cache keeps hot pages warm across accesses.
            self.spike_times    = np.load(st_path,  mmap_mode="r").ravel()
            self.spike_clusters = np.load(sc_path, mmap_mode="r").ravel()

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
                self.sorted_channels = sort_electrode_map(self.channel_positions) if self.channel_positions is not None else None
            except Exception:
                self.sorted_channels = None

            # --- templates and related small-index files (memmap) ----------------
            templates_path = ks / "templates.npy"
            self.templates = np.load(templates_path, mmap_mode="r") if templates_path.exists() else None

            templates_ind_path = ks / "templates_ind.npy"
            self.templates_ind = np.load(templates_ind_path, mmap_mode="r") if templates_ind_path.exists() else None

            whitening_path = ks / "whitening_mat_inv.npy"
            self.whitening_mat_inv = np.load(whitening_path, mmap_mode="r") if whitening_path.exists() else None

            # --- spike amplitudes (optional) ------------------------------------
            amplitudes_path = ks / "amplitudes.npy"
            self.spike_amplitudes = np.load(amplitudes_path, mmap_mode="r").ravel() if amplitudes_path.exists() else None

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
                self.cluster_info = pd.DataFrame({
                    "cluster_id": _unique_ids.astype(int),
                    "group": ["unsorted"] * len(_unique_ids)
                })

            # --- build minimal cluster_df (used heavily elsewhere) ---------------
            if "cluster_id" in self.cluster_info.columns:
                cluster_ids = self.cluster_info["cluster_id"].astype(int).values
                if "group" in self.cluster_info.columns:
                    status_vals = self.cluster_info["group"].astype(str).values
                else:
                    status_vals = self.cluster_info.get("status", pd.Series(["unsorted"] * len(cluster_ids))).astype(str).values
            else:
                cluster_ids = _unique_ids.astype(int)
                status_vals = np.array(["unsorted"] * len(cluster_ids), dtype=object)

            counts_map = dict(zip(_unique_ids.astype(int).tolist(), _counts.astype(int).tolist()))
            n_spikes = np.array([int(counts_map.get(int(cid), 0)) for cid in cluster_ids], dtype=int)

            self.cluster_df = pd.DataFrame({
                "cluster_id": cluster_ids.astype(int),
                "n_spikes": n_spikes,
                "status": status_vals,
                "x_um": np.nan,
                "y_um": np.nan
            })

            self.cluster_id_to_idx = {int(cid): int(i) for i, cid in enumerate(self.cluster_df["cluster_id"].values)}

            # --- optional: channel-template mappings ----------------------------
            try:
                if self.templates is not None:
                    d_mappings = get_channel_template_mappings(self.templates)
                    self.channel_to_templates = d_mappings.get("channel_to_templates", {})
                    self.template_to_channels = d_mappings.get("template_to_channels", {})
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
                logger.debug("Failed to load kilosort params (non-fatal).", exc_info=True)

            try:
                self._load_kilosort_similarity()
            except Exception:
                logger.debug("Failed to load kilosort similarity (non-fatal).", exc_info=True)

            # NOTE: Caches (standard_plot_cache, feature_cache) are intentionally
            # NOT loaded here. They are loaded at the end of build_cluster_dataframe(),
            # after cluster_df has been finalized, so stale keys from a previous
            # session can be pruned against the real cluster population.

            # --- Build O(1) spike index lookup + cache sort order for reuse ---
            logger.debug("Building cluster index mapping...")
            order      = np.argsort(self.spike_clusters, kind='mergesort')
            sorted_cls = self.spike_clusters[order]
            sorted_t   = self.spike_times[order]

            self._spk_sort_order = order
            self._spk_sorted_cls = sorted_cls
            self._spk_sorted_t   = sorted_t

            _, split_idxs   = np.unique(sorted_cls, return_index=True)
            grouped_indices = np.split(order, split_idxs[1:])
            unique_cls      = sorted_cls[split_idxs]

            self.cluster_spike_indices = {
                int(cid): idxs for cid, idxs in zip(unique_cls, grouped_indices)
            }

            # _unique_ids / _counts already computed above — no repeat scan needed.
            # unique_cls is derived from sorted_cls which covers the same set,
            # so we reuse _counts directly (order matches since both come from
            # the same np.unique on spike_clusters).
            self._spk_unique_cls    = unique_cls
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
        vision_data = vision_integration.load_vision_data(
            vision_path, dataset_name)
        logger.debug("Completed vision_integration.load_vision_data call")

        success = False

        if vision_data:
            # --- Full load path (EI + STA + params) ---
            ei_bundle = vision_data.get('ei')
            if ei_bundle:
                self.vision_eis = ei_bundle.get('ei_data')
                self.vision_channel_positions = ei_bundle.get('electrode_map')
                if self.vision_eis:
                    logger.debug(
                        "Available Vision EI IDs (sample): %s",
                        list(self.vision_eis.keys())[:10],
                    )

            self.vision_stas = vision_data.get('sta')
            self.vision_params = vision_data.get('params')

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
                            if hasattr(
                                    first_sta, "red") and first_sta.red is not None:
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
                    self.vision_channel_positions = ei_bundle.get(
                        "electrode_map")

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

        return success, f"{'Successfully' if success else 'Failed to'} load Vision data for {dataset_name}."

    def precompute_ei_correlations_background(self):
        """
        BUG-6 fix: launch _compute_ei_correlations_if_needed() on a daemon
        thread so the UI is never blocked when Vision data is first loaded.
        Safe to call multiple times — the internal lock prevents double work.
        """
        if not hasattr(self, '_ei_corr_lock'):
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
        # Already computed / loaded?
        if self.ei_corr_dict is not None:
            return

        # Must have Vision EIs to do anything
        if self.vision_eis is None:
            logger.warning(
                "Cannot compute EI correlations: vision_eis is None")
            if not self.cluster_df.empty:
                if "potential_dups" not in self.cluster_df.columns:
                    self.cluster_df["potential_dups"] = False
                if "max_dup_r" not in self.cluster_df.columns:
                    self.cluster_df["max_dup_r"] = 0.0
            return

        str_corr_pkl = os.path.join(self.kilosort_dir, "ei_corr_dict.pkl")

        # Sanitize loaded EIs before any numeric operations
        sanitized_eis = self._sanitize_ei_dict(self.vision_eis)

        if len(sanitized_eis) < 2:
            logger.warning(
                "Not enough Vision EIs to compute correlations; skipping duplicate detection"
            )
            if not self.cluster_df.empty:
                if "potential_dups" not in self.cluster_df.columns:
                    self.cluster_df["potential_dups"] = False
                if "max_dup_r" not in self.cluster_df.columns:
                    self.cluster_df["max_dup_r"] = 0.0
            return

        # Try to load existing correlations from disk
        if os.path.exists(str_corr_pkl):
            try:
                logger.debug(
                    "Loading precomputed EI correlations from %s",
                    str_corr_pkl)
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
                    sanitized_eis,
                    sanitized_eis,
                    method="full",
                    n_removed_channels=1)
                space_corr = ei_corr(
                    sanitized_eis,
                    sanitized_eis,
                    method="space",
                    n_removed_channels=1)
                power_corr = ei_corr(
                    sanitized_eis,
                    sanitized_eis,
                    method="power",
                    n_removed_channels=1)
                self.ei_corr_dict = {
                    "full": full_corr,
                    "space": space_corr,
                    "power": power_corr,
                }
                saved_path = self._save_pickle_with_fallback(
                    self.ei_corr_dict, str_corr_pkl
                )
                logger.debug(
                    "EI correlations recomputed and saved to %s",
                    saved_path)
        else:
            # Compute from scratch
            logger.debug("Computing EI correlations")
            full_corr = ei_corr(
                sanitized_eis,
                sanitized_eis,
                method="full",
                n_removed_channels=1)
            space_corr = ei_corr(
                sanitized_eis,
                sanitized_eis,
                method="space",
                n_removed_channels=1)
            power_corr = ei_corr(
                sanitized_eis,
                sanitized_eis,
                method="power",
                n_removed_channels=1)
            self.ei_corr_dict = {
                "full": full_corr,
                "space": space_corr,
                "power": power_corr,
            }
            saved_path = self._save_pickle_with_fallback(
                self.ei_corr_dict, str_corr_pkl)
            logger.debug(
                "EI correlations computed and saved to %s",
                saved_path)

        # With correlation matrices available, update cluster_df
        # duplicate-related columns
        if not self.cluster_df.empty:
            cluster_ids = list(sanitized_eis.keys())
            # Vision IDs are 1-based; convert to 0-based Kilosort cluster IDs
            cluster_ids = np.array(cluster_ids) - 1
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

            self.cluster_df["potential_dups"] = (
                self.cluster_df["cluster_id"]
                .map(potential_dups_map)
                .fillna(False)
                .infer_objects(copy=False)
            )
            self.cluster_df["max_dup_r"] = (
                self.cluster_df["cluster_id"]
                .map(max_dup_r_map)
                .fillna(0.0)
                .infer_objects(copy=False)
            )

            # Sort in-place by max_dup_r
            self.cluster_df = (
                self.cluster_df.sort_values(by="max_dup_r", ascending=False)
                .reset_index(drop=True)
            )
            # Format to 2 decimal places for display
            self.cluster_df["max_dup_r"] = self.cluster_df["max_dup_r"].map(
                lambda x: f"{x:.2f}"
            )

            logger.debug(
                "Updated cluster_df with potential duplicates based on EI correlations"
            )
        else:
            logger.warning(
                "cluster_df is empty; cannot update duplicate columns")

    def load_cell_type_file(self, txt_file: str = None):
        logger.debug("Loading cell type file: %s", txt_file)
        if txt_file is None:
            logger.debug(
                "No cell type file provided; setting cell types to Unknown")
            # Drop existing cell_type column if exists
            if 'cell_type' in self.cluster_df.columns:
                self.cluster_df.drop(columns=['cell_type'], inplace=True)
            return

        try:
            d_result = {}
            with open(txt_file, 'r') as file:
                for line in file:
                    # Split each line into key and value using the specified
                    # delimiter
                    key, value = map(str.strip, line.split(' ', 1))
                    sub_values = value.split('/')

                    # -1 for vision to KS IDs.
                    ks_id = int(key) - 1

                    for str_label in LS_CELL_TYPE_LABELS:
                        if str_label in sub_values:
                            d_result[ks_id] = str_label
                            break

            # Add to cluster_df
            self.cluster_df['cell_type'] = self.cluster_df['cluster_id'].map(
                d_result).fillna('Unknown')
            logger.debug("Loaded cell type file: %s", txt_file)

            # If all are unknown, delete column and print
            if all(ct == 'Unknown' for ct in self.cluster_df['cell_type']):
                self.cluster_df.drop(columns=['cell_type'], inplace=True)
                logger.debug(
                    "All loaded cell types are Unknown; dropping cell_type column")
        except Exception:
            logger.exception("Error loading cell type file")

    def _load_kilosort_params(self):
        params_path = self.kilosort_dir / 'params.py'
        if not params_path.exists():
            raise FileNotFoundError("params.py not found.")
        params = {}
        with open(params_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, val = map(str.strip, line.split('=', 1))
                    try:
                        params[key] = eval(val)
                    except (NameError, SyntaxError):
                        params[key] = val.strip("'\"")
        self.sampling_rate = params.get('fs', 30000)
        self.n_channels = params.get('n_channels_dat', 512)
        dat_path_str = params.get('dat_path', '')
        if isinstance(dat_path_str, (list, tuple)) and dat_path_str:
            dat_path_str = dat_path_str[0]
        suggested_path = Path(dat_path_str)
        if not suggested_path.is_absolute():
            self.dat_path_suggestion = self.kilosort_dir.parent / suggested_path
        else:
            self.dat_path_suggestion = suggested_path

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
            logger.info("set_dat_path: opening Litke bin directory via PyBinFileReader: %s", dat_path)
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
            dat_path, dtype=np.int16, mode='r',
            shape=(self.n_samples, self.n_channels)
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
        if hasattr(self, '_spk_unique_cls') and self._spk_unique_cls is not None:
            cluster_ids = self._spk_unique_cls
            n_spikes    = self._spk_unique_counts
        else:
            cluster_ids, n_spikes = np.unique(self.spike_clusters, return_counts=True)
        logger.debug("Found %d clusters", len(cluster_ids))

        df = pd.DataFrame({'cluster_id': cluster_ids, 'n_spikes': n_spikes})

        # Initialize quick columns (same as before)
        df['potential_dups'] = False
        df['max_dup_r'] = 0.0
        df['cell_type'] = 'Unknown'
        df['isi_violations_pct'] = 0.0

        col = 'KSLabel' if 'KSLabel' in self.cluster_info.columns else 'group'
        if col not in self.cluster_info.columns:
            self.cluster_info[col] = 'unsorted'
        info_subset = self.cluster_info[['cluster_id', col]].rename(columns={
                                                                    col: 'KSLabel'})
        df = pd.merge(df, info_subset, on='cluster_id', how='left')
        # Any cluster present in spike_clusters but absent from cluster_info.tsv
        # gets KSLabel=NaN from the left-merge. Fill with 'unsorted' so they are
        # never silently excluded when downstream code filters by KSLabel.
        df['KSLabel'] = df['KSLabel'].fillna('unsorted')

        df['status'] = 'Original'
        df['set'] = [set([cid]) for cid in df['cluster_id']]
        df['set'] = df['set'].astype(object)

        self.cluster_df = df[
            [
                'cluster_id',
                'cell_type',
                'n_spikes',
                'isi_violations_pct',
                'max_dup_r',
                'potential_dups',
                'status',
                'set',
                'KSLabel',
            ]
        ]
        self.cluster_df['cluster_id'] = self.cluster_df['cluster_id'].astype(
            int)
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
        if hasattr(self, '_spk_sorted_cls') and self._spk_sorted_cls is not None:
            sorted_cls = self._spk_sorted_cls
            sorted_t   = self._spk_sorted_t
        else:
            _order     = np.argsort(self.spike_clusters, kind='stable')
            sorted_cls = self.spike_clusters[_order]
            sorted_t   = self.spike_times[_order]

        isis        = np.diff(sorted_t.astype(np.int64))
        same_cls    = sorted_cls[:-1] == sorted_cls[1:]   # mask out cluster boundaries
        violations  = (isis < refractory_samples) & same_cls
        spike_count = same_cls  # denominator: pairs within same cluster

        # Count violations and total pairs per cluster
        unique_ids, first_idx = np.unique(sorted_cls, return_index=True)
        # split_indices mark where each cluster starts in the sorted arrays
        viol_counts  = np.array([violations [i:j].sum()    for i, j in zip(first_idx, np.append(first_idx[1:], len(violations)))])
        pair_counts  = np.array([spike_count[i:j].sum()    for i, j in zip(first_idx, np.append(first_idx[1:], len(spike_count)))])

        isi_pct_map = {}
        for cid, viol, pairs in zip(unique_ids, viol_counts, pair_counts):
            val = float(viol / pairs * 100) if pairs > 0 else 0.0
            isi_pct_map[int(cid)] = val
            self.isi_cache[(int(cid), float(ISI_REFRACTORY_PERIOD_MS))] = val

        self.cluster_df['isi_violations_pct'] = (
            self.cluster_df['cluster_id'].map(isi_pct_map).fillna(0.0)
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

        valid_ids = set(self.cluster_df['cluster_id'].astype(int).tolist())

        with self._standard_plot_lock:
            stale = [k for k in list(self.standard_plot_cache) if k not in valid_ids]
            for k in stale:
                del self.standard_plot_cache[k]
            if stale:
                logger.debug("Pruned %d stale standard_plot_cache entries", len(stale))

        with self._feature_lock:
            stale = [k for k in list(self.feature_cache)
                     if k not in valid_ids or not self.feature_cache[k].get('_computed')]
            for k in stale:
                del self.feature_cache[k]
            if stale:
                logger.debug("Pruned %d stale feature_cache entries", len(stale))

        logger.debug("build_cluster_dataframe complete")

    def get_cluster_spike_indices(self, cluster_id):
        """Return the indices into the master spike arrays for spikes of a cluster in O(1) time."""
        if not hasattr(self, 'cluster_spike_indices'):
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
        if not hasattr(
                self,
                'spike_amplitudes') or self.spike_amplitudes is None:
            return np.array([])
        inds = self.get_cluster_spike_indices(cluster_id)
        return self.spike_amplitudes[inds]

    def get_standard_plot_data(self, cluster_id):
        """Return cached standard-plot data (ISI/ACG/FR) for a cluster.

        Heavy computations are done once per cluster and stored in
        self.standard_plot_cache, protected by self._standard_plot_lock.
        """
        # Ensure the cache attributes exist (for older sessions)
        if not hasattr(self, 'standard_plot_cache'):
            self.standard_plot_cache = {}
            self._standard_plot_lock = threading.Lock()

        # Fast path: check cache under lock
        with self._standard_plot_lock:
            cached = self.standard_plot_cache.get(cluster_id)
        if cached is not None:
            return cached

        # Compute outside the lock (expensive)
        data = self._compute_standard_plots(cluster_id)

        # Store back under lock
        with self._standard_plot_lock:
            self.standard_plot_cache[cluster_id] = data

        return data

    def save_standard_plot_cache(self):
        """Persist the full standard-plot and feature caches to disk in a background thread."""
        save_path = str(self.kilosort_dir / 'standard_plot_cache.pkl')
        feature_save_path = str(self.kilosort_dir / 'feature_cache.pkl')

        with self._standard_plot_lock:
            snapshot = dict(self.standard_plot_cache)
            
        with getattr(self, '_feature_lock', threading.Lock()):
            feature_snapshot = dict(getattr(self, 'feature_cache', {}))

        def _save():
            try:
                self._save_pickle_with_fallback(snapshot, save_path)
                self._save_pickle_with_fallback(feature_snapshot, feature_save_path)
                logger.debug("Saved caches (%d std, %d feat) to disk", len(snapshot), len(feature_snapshot))
            except Exception:
                logger.exception("Failed to persist standard_plot_cache")

        t = threading.Thread(target=_save, daemon=True)
        t.start()
    
    def load_persisted_caches(self):
        """Loads both standard plot and feature caches from disk.
        Designed to be called safely from a background thread."""
        if not self.kilosort_dir:
            return

        cache_pkl = self.kilosort_dir / 'standard_plot_cache.pkl'
        if cache_pkl.exists() and not getattr(self, 'standard_plot_cache', {}):
            try:
                with open(cache_pkl, 'rb') as f:
                    self.standard_plot_cache = pickle.load(f)
                logger.debug("Restored standard_plot_cache (%d entries) from disk", len(self.standard_plot_cache))
            except Exception:
                logger.warning("Could not load standard_plot_cache.pkl", exc_info=True)
                self.standard_plot_cache = {}

        feat_pkl = self.kilosort_dir / 'feature_cache.pkl'
        if feat_pkl.exists() and not getattr(self, 'feature_cache', {}):
            try:
                with open(feat_pkl, 'rb') as f:
                    self.feature_cache = pickle.load(f)
                logger.debug("Restored feature_cache (%d entries) from disk", len(self.feature_cache))
            except Exception:
                logger.warning("Could not load feature_cache.pkl", exc_info=True)
                self.feature_cache = {}

    def get_cell_physics(self, cluster_id):
        """
        Single Source of Truth for a cell's physical metrics.
        Assembles pre-calculated Vision data and caches it into a flat, 
        instantly accessible dictionary to prevent redundant UI processing loops.
        """
        # Ensure thread safety for background loading
        if not hasattr(self, '_feature_lock'):
            self._feature_lock = threading.Lock()


        # 1. Fast path: check feature cache under lock.
        # We use a '_computed' sentinel key to distinguish "cached and done"
        # from "never cached" — this correctly handles cells where acg is
        # legitimately None (too few spikes for ACG to compute).
        with self._feature_lock:
            cached = self.feature_cache.get(cluster_id)
            if cached is not None and cached.get('_computed'):
                return cached

        # 2. Get standard plot data — use cache if warm, compute inline if cold.
        # NOTE: we intentionally do NOT return None on a cold cache. UMAP and other
        # callers iterate every cluster and cannot handle None in their feature matrix.
        # get_standard_plot_data() checks the cache first and only computes if needed.
        std_data = self.get_standard_plot_data(cluster_id)
        acg_norm = std_data.get('acg_norm') if std_data else None

        # 3. Assemble Pre-Calculated Vision/STA Features
        timecourse = None
        rf_area = 0.0
        ellipticity = 0.0
        time_to_peak = 0

        vid = cluster_id + 1  # Vision IDs are 1-indexed
        if self.vision_stas and vid in self.vision_stas:
            sta_data = self.vision_stas[vid]
            
            # Geometry (Extracting from Vision's pre-computed Gaussian fits)
            try:
                stafit = self.vision_params.get_stafit_for_cell(vid)
                if stafit:
                    rf_area = np.pi * stafit.std_x * stafit.std_y
                    if stafit.std_x > 0:
                        ellipticity = stafit.std_y / stafit.std_x
            except Exception:
                stafit = None

            # Timecourse (Pulls pre-computed 1D arrays from Vision params)
            time_axis, tc_matrix, _ = analysis_core.get_sta_timecourse_data(
                sta_data, stafit, self.vision_params, vid
            )

            if tc_matrix is not None and tc_matrix.size > 0:
                # Stack, find dominant channel by energy, and normalize just once
                energies = np.sum(tc_matrix**2, axis=0)
                dom_idx = np.argmax(energies)
                dom_trace = tc_matrix[:, dom_idx]
                
                # Normalize the trace to -1 to 1 bounds for UI rendering
                abs_max = np.max(np.abs(dom_trace))
                if abs_max > 0:
                    timecourse = dom_trace / abs_max
                else:
                    timecourse = dom_trace
                
                time_to_peak = int(np.argmax(np.abs(timecourse)))

        # 4. Package into our immutable physics dictionary
        metrics = {
            '_computed': True,   # sentinel: marks entry as fully computed, not stale
            'acg': acg_norm,
            'timecourse': timecourse,
            'rf_area': rf_area,
            'ellipticity': ellipticity,
            'time_to_peak': time_to_peak
        }

        # 5. Store in global cache safely
        with self._feature_lock:
            self.feature_cache[cluster_id] = metrics

        return metrics

    def get_acg_data(self, cluster_id):
        """Convenience wrapper: return (time_lags_ms, acg_values)."""
        data = self.get_standard_plot_data(cluster_id)
        return data.get('acg_time_lags'), data.get('acg_norm')

    def get_isi_data(self, cluster_id):
        """Convenience wrapper: return (isi_ms, hist_x, hist_y)."""
        data = self.get_standard_plot_data(cluster_id)
        return data.get('isi_ms'), data.get(
            'isi_hist_x'), data.get('isi_hist_y')

    def get_isi_vs_amplitude_data(self, cluster_id):
        """Convenience wrapper for ISI vs amplitude scatter/density.

        Returns (valid_isi_ms, valid_amplitudes_uV) or (None, None) if unavailable.
        """
        data = self.get_standard_plot_data(cluster_id)
        return data.get('isi_vs_amp_valid_isi'), data.get(
            'isi_vs_amp_valid_amplitudes')

    def get_firing_rate_data(self, cluster_id):
        """Convenience wrapper for firing-rate / amplitude plot.

        Returns:
            bin_centers_sec, rate_hz, amp_x_sec, amp_y_uV, amp_ymax, overlay_x_sec, overlay_y
        """
        data = self.get_standard_plot_data(cluster_id)
        return (
            data.get('fr_bin_centers'),
            data.get('fr_rate'),
            data.get('fr_amp_x'),
            data.get('fr_amp_y'),
            data.get('fr_amp_ymax'),
            data.get('fr_overlay_x'),
            data.get('fr_overlay_y'),
        )

    def _compute_standard_plots(self, cluster_id):
        """Internal helper that actually computes all standard-plot data.

        This mirrors the logic in StandardPlotsPanel.update_all for:
        - autocorrelation (ACG)
        - ISI histogram
        - firing rate + amplitude
        and packs the numeric results into a dict.
        """
        data = {
            'spikes': None,
            'spikes_sec': None,
            'spikes_ms': None,
            'isi_ms': None,
            'isi_hist_x': None,
            'isi_hist_y': None,
            'acg_time_lags': None,
            'acg_norm': None,
            'fr_bin_centers': None,
            'fr_rate': None,
            'fr_amp_x': None,
            'fr_amp_y': None,
            'fr_amp_ymax': None,
            'fr_overlay_x': None,
            'fr_overlay_y': None,
            'isi_vs_amp_valid_isi': None,
            'isi_vs_amp_valid_amplitudes': None,
        }

        # Basic safety checks
        if not hasattr(self, 'spike_times') or self.spike_times is None:
            return data
        if not hasattr(self, 'spike_clusters') or self.spike_clusters is None:
            return data
        if getattr(self, 'sampling_rate', 0) <= 0:
            return data

        # --- Gather spikes & amplitudes for this cluster ---
        spikes = self.get_cluster_spikes(cluster_id)
        spikes = np.asarray(spikes)
        data['spikes'] = spikes

        if spikes.size == 0:
            return data

        sr = float(self.sampling_rate)

        # Convert to seconds and milliseconds
        spikes_sec = spikes / sr
        spikes_ms = (spikes_sec * 1000.0).astype(int)
        data['spikes_sec'] = spikes_sec
        data['spikes_ms'] = spikes_ms

        # ISI vector (ms) and histogram.
        # get_cluster_spikes() returns spike_times[cluster_spike_indices[cid]].
        # cluster_spike_indices was built from argsort(spike_clusters) which
        # preserves the original spike_times order — and spike_times is already
        # monotonically increasing (Kilosort writes spikes in time order).
        # So `spikes` is already sorted; np.sort() is unnecessary.
        if spikes.size > 1:
            isi_ms = np.diff(spikes) / sr * 1000.0
            data['isi_ms'] = isi_ms

            if isi_ms.size > 0:
                hist_y, hist_x = np.histogram(
                    isi_ms, bins=np.linspace(0, 50, 101))
                data['isi_hist_x'] = hist_x
                data['isi_hist_y'] = hist_y

        # Get per-spike amplitudes (may be empty)
        all_amplitudes = self.get_cluster_spike_amplitudes(cluster_id)
        all_amplitudes = np.asarray(all_amplitudes)

        # ISI vs amplitude alignment (for scatter/density)
        if data['isi_ms'] is not None and all_amplitudes.size > 1:
            isi_ms = data['isi_ms']
            min_len = min(len(isi_ms), all_amplitudes.size - 1)
            if min_len > 0:
                valid_isi = isi_ms[:min_len]
                valid_amplitudes = all_amplitudes[1:min_len + 1]
                data['isi_vs_amp_valid_isi'] = valid_isi
                data['isi_vs_amp_valid_amplitudes'] = valid_amplitudes

        # --- Autocorrelation (ACG) — FFT on capped-duration spike train ---
        # The original bug: np.arange(first_ms, last_ms) → 3.6M bins for 1hr recording.
        # Our previous fix (Python loop) was actually SLOWER for large clusters.
        # Correct fix: cap the spike train at MAX_DURATION_MS, then use the fast
        # FFT-based correlate on a small fixed-size array (~120K bins max = ~480KB).
        # ACG statistics are stable after ~2 minutes of spikes, so capping is valid.
        if spikes_ms.size > 1:
            from scipy.signal import fftconvolve
            MAX_LAG       = 100    # ms
            MAX_DURATION  = 120_000  # ms — 2 minutes max, enough for stable ACG stats

            t = np.sort(spikes_ms).astype(np.int64)
            t = t - t[0]  # shift to 0-based

            # Cap at MAX_DURATION to keep the FFT array small regardless of recording length
            if t[-1] > MAX_DURATION:
                t = t[t <= MAX_DURATION]

            if t.size > 1:
                duration = int(t[-1]) + 1
                bins_arr = np.zeros(duration, dtype=np.float32)
                np.add.at(bins_arr, t, 1)

                # FFT cross-correlation: O(N log N), pure C, <1ms for 120K bins
                acg_full = fftconvolve(bins_arr, bins_arr[::-1], mode='full')
                center   = len(acg_full) // 2
                acg      = acg_full[center - MAX_LAG : center + MAX_LAG + 1].copy()
                acg[MAX_LAG] = 0.0  # zero self-coincidence at lag 0

                time_lags  = np.arange(-MAX_LAG, MAX_LAG + 1, dtype=float)
                n_spikes_f = float(spikes_ms.size)
                acg_norm   = acg / n_spikes_f if n_spikes_f > 0 else acg

                data['acg_time_lags'] = time_lags
                data['acg_norm']      = acg_norm

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
            data['fr_bin_centers'] = bin_centers

            if counts.size > 0:
                rate = gaussian_filter1d(counts.astype(float), sigma=5)
            else:
                rate = np.zeros_like(bin_centers, dtype=float)
            data['fr_rate'] = rate

            # Amplitude per bin — O(N) with bincount instead of the old
            # O(N × B) loop that did `all_amplitudes[bin_indices == b]` for
            # every bin b.  For a 1-hour recording with 3600 bins and 100k
            # spikes that loop was doing 360M element comparisons.
            if all_amplitudes.size > 0 and bin_centers.size > 0:
                bin_indices = np.searchsorted(bins[1:], spikes_sec, side='left')
                bin_indices = np.clip(bin_indices, 0, len(bin_centers) - 1)

                amp_sums   = np.bincount(bin_indices, weights=all_amplitudes.astype(float),
                                         minlength=len(bin_centers))
                amp_counts = np.bincount(bin_indices, minlength=len(bin_centers))
                with np.errstate(invalid='ignore'):
                    amplitude_binned = np.where(amp_counts > 0,
                                                amp_sums / amp_counts,
                                                np.nan)

                amplitude_binned = np.asarray(amplitude_binned, dtype=float)

                # Interpolate NaNs if needed
                if np.any(np.isnan(amplitude_binned)):
                    valid_idx = ~np.isnan(amplitude_binned)
                    if np.sum(valid_idx) > 1:
                        f = interp1d(
                            bin_centers[valid_idx],
                            amplitude_binned[valid_idx],
                            kind='linear',
                            bounds_error=False,
                            fill_value='extrapolate',
                        )
                        amplitude_binned = f(bin_centers)
                    elif np.sum(valid_idx) == 1:
                        amplitude_binned = np.full_like(
                            amplitude_binned, amplitude_binned[valid_idx][0])
                    else:
                        amplitude_binned = None

                if amplitude_binned is not None:
                    amplitude_smoothed = gaussian_filter1d(
                        amplitude_binned, sigma=5)
                    data['fr_amp_x'] = bin_centers
                    data['fr_amp_y'] = amplitude_smoothed

                    # Use template PTP to set a sensible right-axis scale when
                    # available
                    max_ptp = 1.0
                    templates = getattr(self, 'templates', None)
                    try:
                        if templates is not None and cluster_id < templates.shape[0]:
                            ptp = templates[cluster_id].max(
                                axis=0) - templates[cluster_id].min(axis=0)
                            if ptp.size > 0:
                                max_ptp = float(ptp.max())
                        # Fallback: use amplitude range
                        if not np.isfinite(max_ptp) or max_ptp <= 0:
                            max_ptp = float(np.nanmax(amplitude_smoothed)) if np.nanmax(
                                amplitude_smoothed) > 0 else 1.0
                    except Exception:
                        max_ptp = float(np.nanmax(amplitude_smoothed)) if np.nanmax(
                            amplitude_smoothed) > 0 else 1.0

                    data['fr_amp_ymax'] = max_ptp * 1.1

            # Overlay averaged amplitude on firing-rate trace (left axis)
            if all_amplitudes.size > 0 and rate.size > 0 and spikes_sec.size > 10:
                max_amp = float(np.max(all_amplitudes))
                if max_amp > 0:
                    normalized_amplitudes = all_amplitudes / max_amp
                else:
                    normalized_amplitudes = all_amplitudes.astype(float)

                if normalized_amplitudes.size > 10:
                    avg_amplitude = np.convolve(
                        normalized_amplitudes, np.ones(10) / 10.0, mode='valid')
                    scaled_amplitude = avg_amplitude * \
                        0.8 * float(np.max(rate))

                    overlay_len = min(len(scaled_amplitude), len(spikes_sec))
                    if overlay_len > 0:
                        data['fr_overlay_x'] = spikes_sec[:overlay_len]
                        data['fr_overlay_y'] = scaled_amplitude[:overlay_len]

        return data

    def get_cluster_mean_amplitude(self, cluster_id, method='mean'):
        """Return a scalar amplitude for the cluster (mean or median)."""
        amps = self.get_cluster_spike_amplitudes(cluster_id)
        if amps.size == 0:
            return 0.0
        if method == 'median':
            return float(np.median(amps))
        return float(np.mean(amps))

    def get_cluster_spikes_in_window(self, cluster_id, start_time, end_time):
        """
        Efficiently get spikes for a cluster within a specific time window.

        This optimized version first finds the time window in the master spike_times
        array (which is sorted) and only then filters that small slice by cluster_id.
        This avoids loading all spikes for a high-firing cluster into memory.
        """
        # Convert the time window (in seconds) to sample indices.
        start_sample = int(start_time * self.sampling_rate)
        end_sample = int(end_time * self.sampling_rate)

        # Use np.searchsorted to find the start and end indices of our time window.
        # This is extremely fast because spike_times is sorted.
        start_idx = np.searchsorted(
            self.spike_times, start_sample, side='left')
        end_idx = np.searchsorted(self.spike_times, end_sample, side='right')

        # If the window is empty or invalid, return an empty array.
        if start_idx >= end_idx:
            return np.array([])

        # Get the small slice of cluster IDs corresponding to our time window.
        window_cluster_ids = self.spike_clusters[start_idx:end_idx]

        # Get the small slice of spike times for that same window.
        window_spike_times = self.spike_times[start_idx:end_idx]

        # Now, perform the final, fast filter on the small slice.
        cluster_spikes_in_window = window_spike_times[window_cluster_ids == cluster_id]

        return cluster_spikes_in_window

    def get_lightweight_features(self, cluster_id):
        """
        Non-blocking cache check for lightweight features.

        This function NO LONGER calculates features. It only checks if they
        have already been computed and cached. The actual calculation is now
        handled by the FeatureWorker.
        """
        return self.ei_cache.get(cluster_id, None)

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
            lightweight_data['median_ei'], self.channel_positions, self.sampling_rate)

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
        distances = np.linalg.norm(
            self.channel_positions - central_pos, axis=1)

        # Get the indices of the n_channels closest channels (excluding the
        # central channel itself)
        # Exclude central channel at index 0
        nearest_indices = np.argsort(
            distances)[1:min(n_channels + 1, len(distances))]

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
        end_sample   = min(self.n_samples, int(end_sample))
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
                raw_snippet = raw_block[litke_indices, :]   # (n_ch, n_samples)
            except Exception:
                logger.exception("PyBinFileReader.get_data failed for samples %d:%d",
                                 start_sample, end_sample)
                return None

            return raw_snippet.astype(np.float32) * self.uV_per_bit

        # --- legacy memmap backend -----------------------------------------------
        if self.raw_data_memmap is None:
            return None

        raw_snippet = self.raw_data_memmap[start_sample:end_sample,
                                           valid_channel_indices]
        uv_snippet = raw_snippet.astype(np.float32) * self.uV_per_bit
        return uv_snippet.T   # → (n_channels, n_samples)

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

    def clear_caches(self):
        """Clear large caches to free memory. Thread-safe for heavyweight_cache."""
        try:
            with self._heavyweight_lock:
                self.heavyweight_cache.clear()
        except Exception:
            # If lock isn't present for any reason, fall back to clearing
            # without lock
            try:
                self.heavyweight_cache.clear()
            except Exception:
                pass

        # Clear similarity precomputation
        self.mea_sorted_indices = None
        self.cluster_id_to_idx = None
        self.cluster_idx_to_id = None  # Added for reverse lookup

        # Clear old caches
        try:
            self.mea_sim_cache.clear()
        except Exception:
            pass
        try:
            self.vision_sim_cache.clear()
        except Exception:
            pass

        # Clear other caches
        try:
            self.ei_cache.clear()
        except Exception:
            pass
        try:
            self.isi_cache.clear()
        except Exception:
            pass

        # Clear standard plots cache
        if hasattr(self, 'standard_plot_cache'):
            try:
                with getattr(self, '_standard_plot_lock', threading.Lock()):
                    self.standard_plot_cache.clear()
            except Exception:
                pass

    def update_after_refinement(self, parent_id, new_clusters_data):
        self.is_dirty = True
        parent_indices = np.where(self.spike_clusters == parent_id)[0]
        self.cluster_df.loc[self.cluster_df['cluster_id']
                            == parent_id, 'status'] = 'Refined (Parent)'
        max_id = self.spike_clusters.max()
        new_rows = []
        for i, new_cluster in enumerate(new_clusters_data):
            new_id = max_id + 1 + i
            sub_indices = parent_indices[new_cluster['inds']]
            self.spike_clusters[sub_indices] = new_id
            isi_violations = self._calculate_isi_violations(new_id)
            new_row = {
                'cluster_id': new_id,
                'KSLabel': 'good',
                'n_spikes': len(sub_indices),
                'isi_violations_pct': isi_violations,
                'status': f'Refined (from C{parent_id})'}
            new_rows.append(new_row)
        self.cluster_df = pd.concat(
            [self.cluster_df, pd.DataFrame(new_rows)], ignore_index=True)
        # Refinement changes spike assignments; cached standard plots are now
        # stale.
        if hasattr(self, 'standard_plot_cache'):
            with getattr(self, '_standard_plot_lock', threading.Lock()):
                self.standard_plot_cache.clear()

    def _calculate_isi_violations(
            self,
            cluster_id,
            refractory_period_ms=ISI_REFRACTORY_PERIOD_MS):
        # Check if we already have the ISI calculation for this cluster in
        # cache
        cache_key = (cluster_id, refractory_period_ms)
        if cache_key in self.isi_cache:
            return self.isi_cache[cache_key]

        spike_times_cluster = self.get_cluster_spikes(cluster_id)
        if len(spike_times_cluster) < 2:
            isi_value = 0.0
        else:
            isis = np.diff(np.sort(spike_times_cluster))
            refractory_period_samples = (
                refractory_period_ms / 1000.0) * self.sampling_rate
            violations = np.sum(isis < refractory_period_samples)
            isi_value = (violations / (len(spike_times_cluster) - 1)) * 100

        # Cache the result
        self.isi_cache[cache_key] = isi_value
        return isi_value

    def update_cluster_isi(self, cluster_id, isi_value):
        """Update the ISI value for a single cluster in both dataframes."""
        # Update the current cluster dataframe
        mask = self.cluster_df['cluster_id'] == cluster_id
        if mask.any():
            self.cluster_df.loc[mask, 'isi_violations_pct'] = isi_value

        # Update the original cluster dataframe
        mask_orig = self.original_cluster_df['cluster_id'] == cluster_id
        if mask_orig.any():
            self.original_cluster_df.loc[mask_orig,
                                         'isi_violations_pct'] = isi_value

    def save_tree_structure(self, file_path):
        """
        Save the current tree structure to a JSON file.
        """
        import json

        def serialize_item(item):
            """Recursively serialize a QStandardItem and its children."""
            item_data = {
                'text': item.text(),
                'data': item.data(),  # cluster_id for cells, None for groups
                'child_count': item.rowCount()
            }

            if item.rowCount() > 0:
                item_data['children'] = []
                for i in range(item.rowCount()):
                    child_item = item.child(i)
                    item_data['children'].append(serialize_item(child_item))

            return item_data

        tree_data = []
        root_model = self.main_window.tree_model  # Access the main window's tree model

        for i in range(root_model.rowCount()):
            item = root_model.item(i)
            tree_data.append(serialize_item(item))

        with open(file_path, 'w') as f:
            json.dump(tree_data, f, indent=2)

    def load_tree_structure(self, file_path):
        """
        Load the tree structure from a JSON file.
        """
        import json

        with open(file_path, 'r') as f:
            tree_data = json.load(f)

        def deserialize_item(item_data):
            """Recursively deserialize an item and its children."""
            item = QStandardItem(item_data['text'])
            item.setEditable(False)

            # Set data (cluster_id for cells)
            item.setData(item_data['data'], Qt.ItemDataRole.UserRole)

            # For groups, enable drop
            if item_data['data'] is None:  # This is a group
                item.setDropEnabled(True)
            else:  # This is a cell
                item.setDropEnabled(False)

            # Add children if they exist
            if 'children' in item_data and item_data['children']:
                for child_data in item_data['children']:
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

    def get_first_spike_time(self, cluster_id):
        """
        Efficiently finds the time of the very first spike for a given cluster.

        Uses numpy.argmax for a highly optimized search, which is orders of
        magnitude faster than iterating or filtering the entire spike array.

        Returns:
            float: The time of the first spike in seconds, or None if the cluster has no spikes.
        """
        try:
            # Create a boolean mask for the selected cluster.
            cluster_mask = (self.spike_clusters == cluster_id)

            # Check if the cluster has any spikes at all.
            if not np.any(cluster_mask):
                return None

            # np.argmax returns the index of the *first* True value. This is
            # extremely fast.
            first_spike_index = np.argmax(cluster_mask)

            # Use that index to get the spike time (in samples) from the sorted
            # times array.
            first_spike_sample = self.spike_times[first_spike_index]

            # Convert to seconds and return.
            return first_spike_sample / self.sampling_rate
        except (IndexError, TypeError):
            # Return None if any error occurs (e.g., empty arrays).
            return None

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

        # ── Fast path: KS4 writes per-cluster positions directly ──────────────
        cluster_pos_path = ks_dir / "cluster_positions.npy"
        if cluster_pos_path.exists():
            try:
                cluster_pos = np.load(cluster_pos_path)  # (n_clusters, 2)
                cids = self.cluster_df["cluster_id"].values
                valid = (cids >= 0) & (cids < len(cluster_pos))
                safe_cids = np.where(valid, cids, 0)
                self.cluster_df["x_um"] = np.where(valid, cluster_pos[safe_cids, 0], np.nan)
                self.cluster_df["y_um"] = np.where(valid, cluster_pos[safe_cids, 1], np.nan)
                logger.debug("Loaded cluster geometry from cluster_positions.npy")
                return
            except Exception as e:
                logger.warning("Failed to load cluster_positions.npy (%s); falling back", e)

        # ── Fallback: derive best channel from templates.npy (fully vectorized) ─
        try:
            chan_pos = np.load(ks_dir / "channel_positions.npy")               # (n_ch, 2)
            templates = np.load(ks_dir / "templates.npy", mmap_mode="r")      # (n_tpl, nt, n_tCh)
            templates_ind = np.load(ks_dir / "templates_ind.npy", mmap_mode="r")  # (n_tpl, n_tCh)
        except FileNotFoundError:
            logger.warning("Required files for cluster geometry not found, skipping.")
            return

        # Compute PTP directly on the memmap — no full copy into RAM.
        # NumPy streams the memmap in blocks, so peak memory is a single row
        # rather than the entire (n_tpl, nt, n_tCh) array (~84 MB for 512-ch).
        ptp = templates.max(axis=1) - templates.min(axis=1)  # (n_tpl, n_tCh)

        n_tpl = ptp.shape[0]
        best_local = ptp.argmax(axis=1)                                        # (n_tpl,)
        best_global = templates_ind[np.arange(n_tpl), best_local].astype(int) # (n_tpl,)
        ptp_at_best = ptp[np.arange(n_tpl), best_local]                       # (n_tpl,)

        cids = self.cluster_df["cluster_id"].values
        valid_tpl = (cids >= 0) & (cids < n_tpl)
        safe_cids = np.where(valid_tpl, cids, 0)

        best_chans = np.where(valid_tpl, best_global[safe_cids], -1)
        self.cluster_df["best_chan"] = best_chans
        self.cluster_df["template_amp"] = np.where(valid_tpl, ptp_at_best[safe_cids], np.nan)

        valid_ch = (best_chans >= 0) & (best_chans < len(chan_pos))
        safe_chans = np.where(valid_ch, best_chans, 0)
        self.cluster_df["x_um"] = np.where(valid_ch, chan_pos[safe_chans, 0], np.nan)
        self.cluster_df["y_um"] = np.where(valid_ch, chan_pos[safe_chans, 1], np.nan)

        logger.debug("Computed cluster geometry via vectorized template PTP")

    def _build_cluster_to_template_map(self):
        """
        Returns dict: cluster_id -> template_index.
        For KS4, cluster ids and template indices are 0..n_clusters-1.
        For KS2/3 you can use spike_templates for spikes in each cluster.
        """
        # For KS4, assume cluster_id == template index
        # If you have a more accurate mapping already, keep that instead.
        n_templates = self.templates.shape[0] if hasattr(
            self, 'templates') else None
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
            "amp_median": "cluster_Amplitude.tsv",    # or whatever name you like
        }

        for col, fname in tsvs.items():
            path = ks_dir / fname
            if path.exists() and col not in self.cluster_df.columns:
                try:
                    df = pd.read_csv(path, sep="\t")
                    # Assume df has columns ['cluster_id', 'value'] or similar
                    # adapt if your files have different schema
                    value_col = [c for c in df.columns if c != "cluster_id"][0]
                    cluster_id_to_value = df.set_index(
                        "cluster_id")[value_col].to_dict()
                    self.cluster_df[col] = self.cluster_df["cluster_id"].map(
                        cluster_id_to_value)
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
            self.cluster_id_to_idx = {int(cid): int(i) for i, cid in enumerate(cluster_ids)}

            # DELETED the massive argsort here. We will do it lazily.
            self.mea_sorted_indices = None

            logger.debug("Loaded similar_templates.npy shape=%s", self.similar_templates.shape)
        except FileNotFoundError:
            logger.warning("similar_templates.npy not found; MEA similarity disabled")
            self.similar_templates = None
            self.mea_sorted_indices = None
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
        self_mask = (cluster_ids == cluster_id)
        tpl_sims[self_mask] = -1.0

        # 4. Fast partial sort (argpartition) instead of full argsort
        actual_top_n = min(top_n, n_clusters)
        if actual_top_n == 0:
            return pd.DataFrame([])

        top_idx = np.argsort(-tpl_sims)[:actual_top_n]

        # 5. Build output DataFrame
        other_ids = cluster_ids[top_idx]
        other_sims = tpl_sims[top_idx]

        n_spikes_arr = self.cluster_df["n_spikes"].values if "n_spikes" in self.cluster_df.columns else np.zeros(n_clusters, dtype=int)
        status_arr = self.cluster_df["status"].astype(str).values if "status" in self.cluster_df.columns else np.array([""] * n_clusters)
        set_col = self.cluster_df["set"] if "set" in self.cluster_df.columns else None

        distances = np.full(actual_top_n, np.nan)
        if "x_um" in self.cluster_df.columns and "y_um" in self.cluster_df.columns:
            x = self.cluster_df["x_um"].values.astype(float)
            y = self.cluster_df["y_um"].values.astype(float)
            target_row = self.cluster_id_to_idx.get(int(cluster_id))
            if target_row is not None:
                dx = x[top_idx] - float(x[target_row])
                dy = y[top_idx] - float(y[target_row])
                distances = np.sqrt(dx*dx + dy*dy)

        rows = {
            "cluster_id": other_ids.astype(int),
            "n_spikes": n_spikes_arr[top_idx].astype(int),
            "status": status_arr[top_idx],
            "distance_um": distances,
            "template_sim": other_sims
        }

        if set_col is not None:
            set_vals = set_col.values
            rows["set"] = [set_vals[i] for i in top_idx]

        return pd.DataFrame(rows)

    def _get_vision_similarity_table(self, cluster_id: int, top_n: int = 50):
        """Get vision-based similarity table for cluster_id using STA correlations."""
        import numpy as np
        import pandas as pd

        if not self.vision_stas or not self.vision_params:
            return pd.DataFrame([])

        vision_cluster_id = cluster_id + 1  # Convert to vision's 1-based indexing

        if vision_cluster_id not in self.vision_stas:
            return pd.DataFrame([])

        # Get the source STA data
        source_sta = self.vision_stas[vision_cluster_id]
        if not hasattr(source_sta, 'red') or source_sta.red is None:
            return pd.DataFrame([])

        # Compute similarity between source and all other vision clusters
        similarities = {}
        for vid, sta_data in self.vision_stas.items():
            if vid == vision_cluster_id:
                continue  # Skip self

            if not hasattr(sta_data, 'red') or sta_data.red is None:
                continue

            # Compute correlation-based similarity using flattened STA frames
            # Stack channels and compute correlation
            source_flat = np.stack([source_sta.red, source_sta.green, source_sta.blue], axis=0).flatten()
            target_flat = np.stack([sta_data.red, sta_data.green, sta_data.blue], axis=0).flatten()

            # Pearson correlation
            if np.std(source_flat) > 0 and np.std(target_flat) > 0:
                corr = np.corrcoef(source_flat, target_flat)[0, 1]
                if not np.isnan(corr):
                    similarities[vid] = corr

        if not similarities:
            return pd.DataFrame([])

        # Sort by similarity and get top N
        sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Build DataFrame
        cluster_ids = []
        sim_values = []
        n_spikes_list = []
        status_list = []
        set_list = []

        for vid, sim in sorted_sims:
            # Convert vision ID back to cluster ID
            cid = vid - 1
            cluster_ids.append(cid)
            sim_values.append(sim)

            # Get metadata from cluster_df if available
            row = self.cluster_df[self.cluster_df['cluster_id'] == cid]
            if not row.empty:
                n_spikes_list.append(row['n_spikes'].values[0] if 'n_spikes' in row.columns else 0)
                status_list.append(row['status'].values[0] if 'status' in row.columns else '')
                set_list.append(row['set'].values[0] if 'set' in row.columns else '')
            else:
                n_spikes_list.append(0)
                status_list.append('')
                set_list.append('')

        return pd.DataFrame({
            'cluster_id': cluster_ids,
            'n_spikes': n_spikes_list,
            'status': status_list,
            'set': set_list,
            'template_sim': sim_values
        })