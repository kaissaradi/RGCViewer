import numpy as np
import pandas as pd
import json
from pathlib import Path
from qtpy.QtCore import QObject, Qt
from qtpy.QtGui import QStandardItem
from analysis import analysis_core
from analysis import vision_integration
from analysis.constants import ISI_REFRACTORY_PERIOD_MS, EI_CORR_THRESHOLD, LS_CELL_TYPE_LABELS
import pickle
import os
import tempfile
import logging
logger = logging.getLogger(__name__)
import threading
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate
from scipy.interpolate import interp1d


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
        channel_to_templates[ch] = cls[chs==ch]
    for cid in range(n_templates):
        template_to_channels[cid] = chs[cls==cid]

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
        max_ref_vals = [np.array(np.max(ei, axis = 1)) for ei in ref_eis]
        ref_to_remove = [np.argsort(val)[-n_removed_channels:] for val in max_ref_vals]
        ref_eis = [np.delete(ei, ref_to_remove[idx], axis = 0) for idx, ei in enumerate(ref_eis)]

    # Set any EI value where the ei is less than 1.5* its standard deviation to 0
    # Added check for std to avoid division by zero (when std is 0)
    for idx, ei in enumerate(ref_eis):
        ei_std = ei.std()
        if ei_std > 0:
            ref_eis[idx][abs(ei) < (ei_std*1.5)] = 0
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
        ref_eis_mean = [np.max(np.abs(ei), axis = 1) for ei in ref_eis]
        ref_eis = np.array(ref_eis_mean)
    # For 'power' method, square each 512 x 201 ei array, take the mean over time,
    # and stack the resulting 512 x 1 vectors into a numpy array
    elif 'power' in method:
        ref_eis_mean = [np.mean(ei**2, axis = 1) for ei in ref_eis]
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
        max_test_vals = [np.array(np.max(ei, axis = 1)) for ei in test_eis]
        test_to_remove = [np.argsort(val)[-n_removed_channels:] for val in max_test_vals]
        test_eis = [np.delete(ei, test_to_remove[idx], axis = 0) for idx, ei in enumerate(test_eis)]

    # Set the EI value where the EI is less than 1.5* its standard deviation to 0
    for idx, ei in enumerate(test_eis):
        ei_std = ei.std()
        if ei_std > 0:
            test_eis[idx][abs(ei) < (ei_std*1.5)] = 0
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
        test_eis_mean = [np.max(np.abs(ei), axis = 1) for ei in test_eis]
        test_eis = np.array(test_eis_mean)
    # For 'power' method, square each 512 x 201 ei array, take the mean over time,
    # and stack the resulting 512 x 1 vectors into a numpy array
    elif 'power' in method:
        test_eis_mean = [np.mean(ei**2, axis = 1) for ei in test_eis]
        test_eis = np.array(test_eis_mean)
    else:
        raise NameError("Method poperty must be 'full', 'space', or 'power'.")


    # If after filtering we have no valid EIs, return empty array
    if len(ref_eis) == 0 or len(test_eis) == 0:
        return np.array([])

    num_pts = ref_eis.shape[1]

    # Calculate covariance and correlation
    c = test_eis @ ref_eis.T / num_pts
    d = np.mean(test_eis, axis = 1)[:,None] * np.mean(ref_eis, axis = 1)[:,None].T
    covs = c - d

    std_calc = np.std(test_eis, axis = 1)[:,None] * np.std(ref_eis, axis = 1)[:, None].T
    # Avoid division by zero - set to 0 if std calculation is 0
    corr = np.divide(covs, std_calc, out=np.zeros_like(covs), where=std_calc!=0)

    # Set nan values and infinite values to 0
    np.nan_to_num(corr, copy=False, nan = 0, posinf = 0, neginf = 0)

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
        logger.debug(f"Initializing DataManager for experiment={self.exp_name}, datafile={self.datafile_name}")
        self.load_stim_timing()

        self.ei_cache = {}
        self.heavyweight_cache = {}
        # Lock to protect accesses to heavyweight_cache from multiple threads
        self._heavyweight_lock = threading.Lock()
        self.isi_cache = {}  # Cache for ISI violation calculations

        # Cache + lock for standard plots (ISI / ACG / FR)
        self.standard_plot_cache = {}
        self._standard_plot_lock = threading.Lock()

        self.dat_path = None

        self.cluster_df = pd.DataFrame()
        self.original_cluster_df = pd.DataFrame()
        self.info_path = None
        self.uV_per_bit = 0.195
        self.main_window = main_window  # Reference to main window for tree operations

        # status df
        self.status_df = pd.DataFrame(columns=['cluster_id', 'status', 'set'])
        self.status_df['set'] = self.status_df['set'].astype(object)
        self.status_csv = self.kilosort_dir / 'status.csv'

        self.mea_similarity_matrix = None  # Full (n_clusters x n_clusters) similarity matrix
        self.mea_sorted_indices = None  # Pre-sorted indices for each cluster
        self.cluster_id_to_idx = None  # Map cluster_id -> row index

        # --- Vision Data ---
        self.vision_eis = None
        self.vision_stas = None
        self.vision_params = None
        self.vision_channel_positions = None # Store channel positions from vision data
        self.vision_sta_width = None  # Store stimulus width for coordinate alignment
        self.vision_sta_height = None  # Store stimulus height for coordinate alignment
        self.ei_corr_dict = None  # Initialize to None, will be set when vision data is loaded

        # --- MEA Similarity Data ---
        self.similar_templates = None          # (n_templates, n_templates) from Kilosort
        self.cluster_to_template = None        # dict[int -> int]
        self.mea_sim_cache = {}                # cluster_id -> DataFrame
        self.vision_sim_cache = {}             # cluster_id -> DataFrame
        self.vision_available = False

        # Initialize raw data memmap attribute (will hold memmap object)
        self.raw_data_memmap = None

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

    def _save_pickle_with_fallback(self, data, filepath):
        """
        Save pickle data to the original filepath. If permission is denied,
        save to a temporary location instead.

        Returns the path where the data was actually saved.
        """
        try:
            # Try to save to the original location first
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            logger.debug("Successfully saved pickle to %s", filepath)
            return filepath
        except PermissionError:
            # If permission denied, save to a temporary location
            temp_dir = Path(tempfile.gettempdir())
            filename = Path(filepath).name
            temp_path = temp_dir / filename

            try:
                with open(temp_path, 'wb') as f:
                    pickle.dump(data, f)
                logger.debug("Saved pickle to temporary location: %s (original not writable)", temp_path)
                return temp_path
            except Exception as e:
                logger.exception("Failed to save pickle to both original and temporary locations")
                raise e
        except Exception as e:
            logger.exception("Failed to save pickle")
            raise e

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
                logger.warning("Skipping EI for key %s: invalid or empty EI data", k)
        return out

    def load_stim_timing(self):
        try:
            import retinanalysis.utils.datajoint_utils as dju
            self.block_id = dju.get_block_id_from_datafile(self.exp_name, self.datafile_name)
            self.d_timing = dju.get_epochblock_timing(self.exp_name, self.block_id)
            logger.debug("Loaded stimulus timing data successfully")
        except Exception as e:
            logger.exception("Failed to load stimulus timing data")
            return

    def update_and_export_status(self, selected_ids, status):
        selected_ids = set(selected_ids)
        logger.debug("Marking %s: %s", status, selected_ids)
        # Update status_df
        for cid in selected_ids:
            # If Duplicate, all selected ids are a 'set', else only self
            if status == 'Duplicate':
                set_ids = selected_ids
            else:
                set_ids = set([cid])

            if cid in self.status_df['cluster_id'].values:
                idx = self.status_df[self.status_df['cluster_id'] == cid].index[0]

                # Update existing entry
                self.status_df.at[idx, 'status'] = status
                self.status_df.at[idx, 'set'] = set_ids
            else:
                # Create new entry
                self.status_df = pd.concat([self.status_df, pd.DataFrame({
                    'cluster_id': [cid],
                    'status': [status],
                    'set': [set_ids]
                })], ignore_index=True)


        # Update cluster_df status and export status csv
        self.update_cluster_df_with_status()
        self.export_status()


    def update_cluster_df_with_status(self):
        """
        Update the cluster_df 'status' column based on current status_df.
        """
        if self.cluster_df.empty:
            return

        # Reset all statuses to 'Original'
        self.cluster_df['status'] = 'Original'

        for _, row in self.status_df.iterrows():
            cluster_id = row['cluster_id']
            status = row['status']
            idx = self.cluster_df[self.cluster_df['cluster_id'] == cluster_id].index[0]
            self.cluster_df.at[idx, 'status'] = status
            self.cluster_df.at[idx, 'set'] = row['set']

    def export_status(self):
        """
        Export duplicate_sets to a JSON file in the Kilosort directory.
        """

        if self.status_df.empty:
            # Nothing to save
            return

            try:
                self.status_df.to_csv(self.status_csv, index=False)
                logger.debug("Exported %d status entries to %s", len(self.status_df), self.status_csv)
            except Exception as e:
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
            status_df['set'] = status_df['set'].apply(lambda x: set(map(int, x.strip("{}").split(","))))
            self.status_df = status_df

            logger.debug("Loaded status csv: %s", self.status_csv)
            logger.debug("Status counts: %s", self.status_df['status'].value_counts().to_dict())

            self.update_cluster_df_with_status()

            return True
        except Exception as e:
            logger.exception("Failed to load duplicate sets")
            return False

    def load_kilosort_data(self):
        try:
            self.spike_times = np.load(self.kilosort_dir / 'spike_times.npy').flatten()
            self.spike_clusters = np.load(self.kilosort_dir / 'spike_clusters.npy').flatten()
            self.channel_positions = np.load(self.kilosort_dir / 'channel_positions.npy')
            self.sorted_channels = sort_electrode_map(self.channel_positions)

            # Load whitening matrix inverse to unwhiten templates (templates are stored in whitened space)
            self.templates = np.load(self.kilosort_dir / 'templates.npy')

            # Check if whitening_mat_inv exists and apply it to unwhiten templates
            whitening_mat_inv_path = self.kilosort_dir / 'whitening_mat_inv.npy'
            if whitening_mat_inv_path.exists():
                whitening_mat_inv = np.load(whitening_mat_inv_path)
                # Apply inverse whitening matrix to convert templates from whitened to unwhitened space
                # Templates shape: (n_clusters, n_timepoints, n_channels)
                # Whitening matrix shape: (n_channels, n_channels)
                # For each template, multiply each timepoint by whitening_mat_inv along channel dimension
                unwhitened_templates = np.einsum('ijk,kl->ijl', self.templates, whitening_mat_inv)
                self.templates = unwhitened_templates
                logger.debug("Applied whitening_mat_inv to templates for proper unwhitened visualization")
            else:
                logger.warning("whitening_mat_inv.npy not found; templates may be displayed in whitened space")

            d_mappings = get_channel_template_mappings(self.templates)
            self.channel_to_templates = d_mappings['channel_to_templates']
            self.template_to_channels = d_mappings['template_to_channels']

            self.spike_amplitudes = np.load(self.kilosort_dir / 'amplitudes.npy').flatten()

            info_path = self.kilosort_dir / 'cluster_info.tsv'
            group_path = self.kilosort_dir / 'cluster_group.tsv'

            if info_path.exists():
                self.info_path = info_path
                self.cluster_info = pd.read_csv(info_path, sep='\t')
            elif group_path.exists():
                self.info_path = group_path
                self.cluster_info = pd.read_csv(group_path, sep='\t')
            else:
                logger.info("No 'cluster_info.tsv' or 'cluster_group.tsv' found; labeling clusters as 'unsorted'")
                self.info_path = None
                all_cluster_ids = np.unique(self.spike_clusters)
                self.cluster_info = pd.DataFrame({
                    'cluster_id': all_cluster_ids,
                    'group': ['unsorted'] * len(all_cluster_ids)
                })

            self._load_kilosort_params()

            # Load Kilosort similarity data
            self._load_kilosort_similarity()

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

                # The STA structure is likely a container with red, green, blue channels
                if hasattr(first_sta, "red") and first_sta.red is not None:
                    sta_shape = first_sta.red.shape
                    if len(sta_shape) >= 2:
                        # Dimensions are [height, width, timepoints]
                        self.vision_sta_height = sta_shape[0]
                        self.vision_sta_width = sta_shape[1]
                    else:
                        # If we only have 2 dimensions, they are likely [height, width]
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

            # Check if params or STA files exist even if the full loading failed
            params_path = vision_path / "sta_params.params"
            sta_path = vision_path / "sta_container.sta"

            if params_path.exists() or sta_path.exists():
                logger.debug("Found params/sta files; attempting partial load")

                # Try to load the existing files one by one using the available functions
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
                    self.vision_channel_positions = ei_bundle.get("electrode_map")

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

    def _compute_ei_correlations_if_needed(self):
        """
        Lazily compute EI correlation matrices and duplicate flags.

        This used to run inside load_vision_data() and block Vision loading.
        Now it is called on-demand (e.g. from the Vision similarity table).
        """
        # Already computed / loaded?
        if self.ei_corr_dict is not None:
            return

        # Must have Vision EIs to do anything
        if self.vision_eis is None:
            logger.warning("Cannot compute EI correlations: vision_eis is None")
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
                logger.debug("Loading precomputed EI correlations from %s", str_corr_pkl)
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
            saved_path = self._save_pickle_with_fallback(self.ei_corr_dict, str_corr_pkl)
            logger.debug("EI correlations computed and saved to %s", saved_path)

        # With correlation matrices available, update cluster_df duplicate-related columns
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
            )
            self.cluster_df["max_dup_r"] = (
                self.cluster_df["cluster_id"]
                .map(max_dup_r_map)
                .fillna(0.0)
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
            logger.warning("cluster_df is empty; cannot update duplicate columns")


    def load_cell_type_file(self, txt_file: str=None):
        logger.debug("Loading cell type file: %s", txt_file)
        if txt_file is None:
            logger.debug("No cell type file provided; setting cell types to Unknown")
            # Drop existing cell_type column if exists
            if 'cell_type' in self.cluster_df.columns:
                self.cluster_df.drop(columns=['cell_type'], inplace=True)
            return

        try:
            d_result = {}
            with open(txt_file, 'r') as file:
                for line in file:
                    # Split each line into key and value using the specified delimiter
                    key, value = map(str.strip, line.split(' ', 1))
                    sub_values = value.split('/')

                    # -1 for vision to KS IDs.
                    ks_id = int(key) - 1

                    for str_label in LS_CELL_TYPE_LABELS:
                        if str_label in sub_values:
                            d_result[ks_id] = str_label
                            break


            # Add to cluster_df
            self.cluster_df['cell_type'] = self.cluster_df['cluster_id'].map(d_result).fillna('Unknown')
            logger.debug("Loaded cell type file: %s", txt_file)

            # If all are unknown, delete column and print
            if all(ct == 'Unknown' for ct in self.cluster_df['cell_type']):
                self.cluster_df.drop(columns=['cell_type'], inplace=True)
                logger.debug("All loaded cell types are Unknown; dropping cell_type column")
        except Exception as e:
            logger.exception("Error loading cell type file")



    def _load_kilosort_params(self):
        params_path = self.kilosort_dir / 'params.py'
        if not params_path.exists(): raise FileNotFoundError("params.py not found.")
        params = {}
        with open(params_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, val = map(str.strip, line.split('=', 1))
                    try: params[key] = eval(val)
                    except (NameError, SyntaxError): params[key] = val.strip("'\"")
        self.sampling_rate = params.get('fs', 30000)
        self.n_channels = params.get('n_channels_dat', 512)
        dat_path_str = params.get('dat_path', '')
        if isinstance(dat_path_str, (list, tuple)) and dat_path_str: dat_path_str = dat_path_str[0]
        suggested_path = Path(dat_path_str)
        if not suggested_path.is_absolute():
            self.dat_path_suggestion = self.kilosort_dir.parent / suggested_path
        else:
            self.dat_path_suggestion = suggested_path

    def set_dat_path(self, dat_path):
        """
        Set the path to the raw data file and create a memory map for efficient access.
        """
        self.dat_path = Path(dat_path)
        # Calculate the number of samples in the file based on its size
        file_size = self.dat_path.stat().st_size
        self.n_samples = file_size // (self.n_channels * 2)  # Assuming int16 (2 bytes) per sample
        # Create a memory map to efficiently access the raw data without loading it all into RAM
        self.raw_data_memmap = np.memmap(self.dat_path, dtype=np.int16, mode='r',
                                         shape=(self.n_samples, self.n_channels))

    def build_cluster_dataframe(self):
        """
        Build the cluster dataframe (backwards-compatible single-call).

        Optimizations:
        - ISI calculation is done in a vectorized pass when spike_times exist.
        - If self.defer_mea_precompute is True, skip the MEA precompute step
            (useful to avoid the long O(N^2) stage during load).
        """
        logger.debug("Starting build_cluster_dataframe")

        # --- basic counts / fast table ---
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
        info_subset = self.cluster_info[['cluster_id', col]].rename(columns={col: 'KSLabel'})
        df = pd.merge(df, info_subset, on='cluster_id', how='left')

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
        self.cluster_df['cluster_id'] = self.cluster_df['cluster_id'].astype(int)
        self.original_cluster_df = self.cluster_df.copy()
        logger.debug("build_cluster_dataframe basic structure complete")

        # Initialize / reset the ISI cache
        self.isi_cache = {}

        # --- Fast vectorized ISI computation when possible ---
        try:
            spikes = np.asarray(self.spike_times)
            clusters = np.asarray(self.spike_clusters)
            if spikes.size >= 2 and spikes.shape[0] == clusters.shape[0]:
                logger.debug("Computing ISI violations with vectorized routine")
                # Sort by cluster id while preserving per-cluster time order (mergesort)
                order = np.argsort(clusters, kind="mergesort")
                clusters_sorted = clusters[order]
                spikes_sorted = spikes[order]

                # Within-cluster adjacent diffs
                same_cluster = clusters_sorted[1:] == clusters_sorted[:-1]
                if np.any(same_cluster):
                    dt = spikes_sorted[1:] - spikes_sorted[:-1]
                    dt = dt[same_cluster]
                    pair_clusters = clusters_sorted[1:][same_cluster]

                    refractory_samples = (ISI_REFRACTORY_PERIOD_MS / 1000.0) * float(self.sampling_rate)
                    violations_mask = dt < refractory_samples

                    # Aggregate counts per cluster id
                    max_cid = int(max(pair_clusters.max(), int(self.cluster_df['cluster_id'].max())))
                    total_pairs = np.bincount(pair_clusters.astype(int), minlength=max_cid + 1)
                    violation_counts = np.bincount(pair_clusters.astype(int), weights=violations_mask.astype(np.int64), minlength=max_cid + 1)

                    isi_vals = []
                    for cid in self.cluster_df['cluster_id'].astype(int).values:
                        if cid <= max_cid:
                            pairs = int(total_pairs[cid])
                            viol = int(violation_counts[cid])
                        else:
                            pairs = 0
                            viol = 0
                        pct = 0.0 if pairs == 0 else (viol / pairs) * 100.0
                        isi_vals.append(pct)
                        # cache it
                        try:
                            self.isi_cache[(int(cid), float(ISI_REFRACTORY_PERIOD_MS))] = pct
                        except Exception:
                            pass

                    self.cluster_df['isi_violations_pct'] = isi_vals
                    logger.debug("Vectorized ISI computation complete")
                else:
                    # no within-cluster adjacent pairs => zeros
                    logger.debug("No within-cluster pairs found; ISI values set to 0")
                    self.cluster_df['isi_violations_pct'] = 0.0
                    for cid in self.cluster_df['cluster_id'].astype(int).values:
                        self.isi_cache[(int(cid), float(ISI_REFRACTORY_PERIOD_MS))] = 0.0
            else:
                # fallback to per-cluster loop below
                raise ValueError("spike_times/spike_clusters shapes not compatible for vectorized path")
        except Exception as e:
            # Fallback: compute per-cluster to preserve behavior if vectorized path fails
            logger.debug("Vectorized ISI compute failed or unavailable (%s); falling back to per-cluster loop", str(e))
            cluster_ids_list = self.cluster_df['cluster_id'].values
            isi_values = []
            total_clusters = len(cluster_ids_list)
            for i, cluster_id in enumerate(cluster_ids_list):
                isi_value = self._calculate_isi_violations(cluster_id, refractory_period_ms=ISI_REFRACTORY_PERIOD_MS)
                isi_values.append(isi_value)
                try:
                    self.isi_cache[(int(cluster_id), float(ISI_REFRACTORY_PERIOD_MS))] = isi_value
                except Exception:
                    pass
                if (i + 1) % 50 == 0 or i == total_clusters - 1:
                    logger.debug("Calculated ISI for %d/%d clusters", i + 1, total_clusters)
            self.cluster_df['isi_violations_pct'] = isi_values
            logger.debug("Per-cluster ISI computation complete")

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

        # MEA precompute: skip if user set self.defer_mea_precompute = True
        try:
            if getattr(self, 'defer_mea_precompute', False):
                logger.debug("Skipping MEA similarity precompute (defer_mea_precompute=True)")
            else:
                logger.debug("Starting MEA similarity precomputation")
                # prefer vectorized if available, fallback to previous
                if hasattr(self, '_precompute_mea_similarity_vectorized'):
                    self._precompute_mea_similarity()
                else:
                    self._precompute_mea_similarity()
                logger.debug("MEA similarity precomputation complete")
        except Exception as e:
            logger.exception("MEA similarity precomputation failed: %s", e)

        logger.debug("build_cluster_dataframe complete")




    def get_cluster_spikes(self, cluster_id):
        return self.spike_times[self.spike_clusters == cluster_id]

    def get_cluster_spike_indices(self, cluster_id):
        """Return the indices into the master spike arrays for spikes of a cluster."""
        return np.where(self.spike_clusters == cluster_id)[0]

    def get_cluster_spike_amplitudes(self, cluster_id):
        """Return the per-spike amplitudes for a cluster (empty array if not available)."""
        if not hasattr(self, 'spike_amplitudes') or self.spike_amplitudes is None:
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

    def get_acg_data(self, cluster_id):
        """Convenience wrapper: return (time_lags_ms, acg_values)."""
        data = self.get_standard_plot_data(cluster_id)
        return data.get('acg_time_lags'), data.get('acg_norm')

    def get_isi_data(self, cluster_id):
        """Convenience wrapper: return (isi_ms, hist_x, hist_y)."""
        data = self.get_standard_plot_data(cluster_id)
        return data.get('isi_ms'), data.get('isi_hist_x'), data.get('isi_hist_y')

    def get_isi_vs_amplitude_data(self, cluster_id):
        """Convenience wrapper for ISI vs amplitude scatter/density.

        Returns (valid_isi_ms, valid_amplitudes_uV) or (None, None) if unavailable.
        """
        data = self.get_standard_plot_data(cluster_id)
        return data.get('isi_vs_amp_valid_isi'), data.get('isi_vs_amp_valid_amplitudes')

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

        # ISI vector (ms) and histogram
        if spikes.size > 1:
            sorted_spikes = np.sort(spikes)
            isi_ms = np.diff(sorted_spikes) / sr * 1000.0
            data['isi_ms'] = isi_ms

            if isi_ms.size > 0:
                hist_y, hist_x = np.histogram(isi_ms, bins=np.linspace(0, 50, 101))
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

        # --- Autocorrelation (ACG) ---
        if spikes_ms.size > 1:
            duration = int(spikes_ms[-1])
            if duration > 0:
                bin_width_ms = 1
                bins = np.arange(0, duration + bin_width_ms, bin_width_ms)
                binned_spikes, _ = np.histogram(spikes_ms, bins=bins)

                if binned_spikes.size > 0:
                    centered = binned_spikes - np.mean(binned_spikes)
                    acg_full = correlate(centered, centered, mode='full')

                    zero_lag_idx = len(acg_full) // 2
                    max_lag_ms = 100
                    num_bins = int(max_lag_ms / bin_width_ms)
                    lag_range = min(num_bins, zero_lag_idx)

                    if lag_range > 0:
                        acg_symmetric = acg_full[zero_lag_idx - lag_range : zero_lag_idx + lag_range + 1]
                        time_lags = np.arange(-lag_range, lag_range + 1) * bin_width_ms

                        # Zero out the central peak so refractory effects are visible
                        zero_idx = np.where(time_lags == 0)[0]
                        if zero_idx.size > 0:
                            acg_symmetric[zero_idx[0]] = 0

                        # Normalize by variance and length
                        spike_variance = np.var(binned_spikes)
                        if spike_variance != 0:
                            acg_norm = acg_symmetric / spike_variance / len(binned_spikes)
                        else:
                            acg_norm = acg_symmetric.astype(float)

                        data['acg_time_lags'] = time_lags.astype(float)
                        data['acg_norm'] = acg_norm.astype(float)

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

            # Amplitude on the right axis (gold line)
            if all_amplitudes.size > 0 and bin_centers.size > 0:
                amplitude_binned = []
                for bin_start in bin_centers:
                    bin_end = bin_start + 1.0
                    mask = (spikes_sec >= bin_start) & (spikes_sec < bin_end)
                    if np.any(mask):
                        amplitude_binned.append(float(np.mean(all_amplitudes[mask])))
                    else:
                        amplitude_binned.append(np.nan)

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
                        amplitude_binned = np.full_like(amplitude_binned, amplitude_binned[valid_idx][0])
                    else:
                        amplitude_binned = None

                if amplitude_binned is not None:
                    amplitude_smoothed = gaussian_filter1d(amplitude_binned, sigma=5)
                    data['fr_amp_x'] = bin_centers
                    data['fr_amp_y'] = amplitude_smoothed

                    # Use template PTP to set a sensible right-axis scale when available
                    max_ptp = 1.0
                    templates = getattr(self, 'templates', None)
                    try:
                        if templates is not None and cluster_id < templates.shape[0]:
                            ptp = templates[cluster_id].max(axis=0) - templates[cluster_id].min(axis=0)
                            if ptp.size > 0:
                                max_ptp = float(ptp.max())
                        # Fallback: use amplitude range
                        if not np.isfinite(max_ptp) or max_ptp <= 0:
                            max_ptp = float(np.nanmax(amplitude_smoothed)) if np.nanmax(amplitude_smoothed) > 0 else 1.0
                    except Exception:
                        max_ptp = float(np.nanmax(amplitude_smoothed)) if np.nanmax(amplitude_smoothed) > 0 else 1.0

                    data['fr_amp_ymax'] = max_ptp * 1.1

            # Overlay averaged amplitude on firing-rate trace (left axis)
            if all_amplitudes.size > 0 and rate.size > 0 and spikes_sec.size > 10:
                max_amp = float(np.max(all_amplitudes))
                if max_amp > 0:
                    normalized_amplitudes = all_amplitudes / max_amp
                else:
                    normalized_amplitudes = all_amplitudes.astype(float)

                if normalized_amplitudes.size > 10:
                    avg_amplitude = np.convolve(normalized_amplitudes, np.ones(10) / 10.0, mode='valid')
                    scaled_amplitude = avg_amplitude * 0.8 * float(np.max(rate))

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
        start_idx = np.searchsorted(self.spike_times, start_sample, side='left')
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
            # Another thread may have computed it while we were working; prefer existing
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
            # If no channel positions are available, return consecutive channels
            start_idx = max(0, central_channel_idx)
            end_idx = min(self.n_channels, start_idx + n_channels)
            return list(range(start_idx, end_idx))

        if central_channel_idx >= len(self.channel_positions):
            # Use the last available channel
            central_channel_idx = len(self.channel_positions) - 1

        # Calculate Euclidean distance from the central channel to all other channels
        central_pos = self.channel_positions[central_channel_idx]
        distances = np.linalg.norm(self.channel_positions - central_pos, axis=1)

        # Get the indices of the n_channels closest channels (excluding the central channel itself)
        nearest_indices = np.argsort(distances)[1:min(n_channels + 1, len(distances))]  # Exclude central channel at index 0

        # Create the list [neighbor_1, dominant_channel, neighbor_2] with the dominant channel in the middle
        result = nearest_indices.tolist()
        result.insert(1, central_channel_idx)  # Insert the central channel in the middle

        # Make sure we only return n_channels (default 3) total
        if len(result) > n_channels:
            result = result[:n_channels]

        return result

    def get_raw_trace_snippet(self, channel_indices, start_sample, end_sample):
        """
        Get a snippet of raw trace data for specified channels and time range.
        Apply the uV_per_bit conversion factor.
        """
        if self.raw_data_memmap is None:
            return None

        # Ensure indices are within bounds
        valid_channel_indices = [idx for idx in channel_indices if 0 <= idx < self.n_channels]

        # Ensure sample range is within bounds
        start_sample = max(0, start_sample)
        end_sample = min(self.n_samples, end_sample)

        if start_sample >= end_sample or len(valid_channel_indices) == 0:
            # Return empty array with proper shape if no valid data
            return np.array([]).reshape(0, 0)

        # Extract the requested data
        raw_snippet = self.raw_data_memmap[start_sample:end_sample, valid_channel_indices]

        # Convert to microvolts using the conversion factor
        uv_snippet = raw_snippet.astype(np.float32) * self.uV_per_bit

        return uv_snippet.T  # Transpose to have channels as rows and time as columns

    def clear_caches(self):
        """Clear large caches to free memory. Thread-safe for heavyweight_cache."""
        try:
            with self._heavyweight_lock:
                self.heavyweight_cache.clear()
        except Exception:
            # If lock isn't present for any reason, fall back to clearing without lock
            try:
                self.heavyweight_cache.clear()
            except Exception:
                pass

        # Clear similarity precomputation
        self.mea_similarity_matrix = None
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
        self.cluster_df.loc[self.cluster_df['cluster_id'] == parent_id, 'status'] = 'Refined (Parent)'
        max_id = self.spike_clusters.max()
        new_rows = []
        for i, new_cluster in enumerate(new_clusters_data):
            new_id = max_id + 1 + i
            sub_indices = parent_indices[new_cluster['inds']]
            self.spike_clusters[sub_indices] = new_id
            isi_violations = self._calculate_isi_violations(new_id)
            new_row = {
                'cluster_id': new_id, 'KSLabel': 'good', 'n_spikes': len(sub_indices),
                'isi_violations_pct': isi_violations, 'status': f'Refined (from C{parent_id})'
            }
            new_rows.append(new_row)
        self.cluster_df = pd.concat([self.cluster_df, pd.DataFrame(new_rows)], ignore_index=True)
        # Refinement changes spike assignments; cached standard plots are now stale.
        if hasattr(self, 'standard_plot_cache'):
            with getattr(self, '_standard_plot_lock', threading.Lock()):
                self.standard_plot_cache.clear()


    def _calculate_isi_violations(self, cluster_id, refractory_period_ms=ISI_REFRACTORY_PERIOD_MS):
        # Check if we already have the ISI calculation for this cluster in cache
        cache_key = (cluster_id, refractory_period_ms)
        if cache_key in self.isi_cache:
            return self.isi_cache[cache_key]

        spike_times_cluster = self.get_cluster_spikes(cluster_id)
        if len(spike_times_cluster) < 2:
            isi_value = 0.0
        else:
            isis = np.diff(np.sort(spike_times_cluster))
            refractory_period_samples = (refractory_period_ms / 1000.0) * self.sampling_rate
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
            self.original_cluster_df.loc[mask_orig, 'isi_violations_pct'] = isi_value

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

            # np.argmax returns the index of the *first* True value. This is extremely fast.
            first_spike_index = np.argmax(cluster_mask)

            # Use that index to get the spike time (in samples) from the sorted times array.
            first_spike_sample = self.spike_times[first_spike_index]

            # Convert to seconds and return.
            return first_spike_sample / self.sampling_rate
        except (IndexError, TypeError):
            # Return None if any error occurs (e.g., empty arrays).
            return None

    def _compute_cluster_geometry(self):
        """
        Compute best_chan, x_um, y_um, and template_amp for each cluster.
        Uses templates.npy, templates_ind.npy, channel_positions.npy.
        """
        import numpy as np
        ks_dir = self.kilosort_dir

        # Load required files
        try:
            templates = np.load(ks_dir / "templates.npy", mmap_mode="r")        # (n_templates, nt, n_tempCh)
            templates_ind = np.load(ks_dir / "templates_ind.npy", mmap_mode="r")  # (n_templates, n_tempCh)
            chan_pos = np.load(ks_dir / "channel_positions.npy")               # (n_channels, 2)
        except FileNotFoundError:
            logger.warning("Required files for cluster geometry computation not found, skipping...")
            return

        # optional but nice: unwhiten before computing PTP
        try:
            W_inv = np.load(ks_dir / "whitening_mat_inv.npy", mmap_mode="r")
        except FileNotFoundError:
            W_inv = None

        n_templates = templates.shape[0]

        best_chan_per_template = np.zeros(n_templates, dtype=int)
        ptp_per_template = np.zeros(n_templates, dtype=float)

        for k in range(n_templates):
            T_white = templates[k]  # (nt, n_tempCh)

            if W_inv is not None:
                # Expand to full channel set, then unwhiten along channel axis
                chans = templates_ind[k].astype(int)
                nt, n_tempCh = T_white.shape
                n_channels = W_inv.shape[0]
                T_full_white = np.zeros((nt, n_channels), dtype=T_white.dtype)
                T_full_white[:, chans] = T_white
                T = T_full_white @ W_inv        # unwhitened (nt, n_channels)
            else:
                # Use whitened template directly; still OK for relative PTP
                chans = templates_ind[k].astype(int)
                T = np.zeros((T_white.shape[0], chan_pos.shape[0]), dtype=T_white.dtype)
                T[:, chans] = T_white

            ptp = T.max(axis=0) - T.min(axis=0)   # (n_channels,)
            best_chan = int(ptp.argmax())
            best_chan_per_template[k] = best_chan
            ptp_per_template[k] = float(ptp[best_chan])

        # Map cluster_id -> template index (for KS4, cluster == template; for KS2/3 use spike_templates)
        # You likely already have something like this; keep your existing logic if so.
        cluster_to_template = self._build_cluster_to_template_map()

        # Add new columns to cluster_df (without changing existing ones)
        if "best_chan" not in self.cluster_df.columns:
            cluster_to_best_chan = {}
            for cid in self.cluster_df["cluster_id"]:
                if cid in cluster_to_template:
                    template_idx = cluster_to_template[cid]
                    if template_idx < len(best_chan_per_template):
                        cluster_to_best_chan[cid] = best_chan_per_template[template_idx]
            self.cluster_df["best_chan"] = self.cluster_df["cluster_id"].map(cluster_to_best_chan)

        if "x_um" not in self.cluster_df.columns or "y_um" not in self.cluster_df.columns:
            self.cluster_df["x_um"] = self.cluster_df["best_chan"].map(lambda ch: chan_pos[ch, 0] if ch < len(chan_pos) else np.nan)
            self.cluster_df["y_um"] = self.cluster_df["best_chan"].map(lambda ch: chan_pos[ch, 1] if ch < len(chan_pos) else np.nan)

        if "template_amp" not in self.cluster_df.columns:
            cluster_to_template_amp = {}
            for cid in self.cluster_df["cluster_id"]:
                if cid in cluster_to_template:
                    template_idx = cluster_to_template[cid]
                    if template_idx < len(ptp_per_template):
                        cluster_to_template_amp[cid] = ptp_per_template[template_idx]
            self.cluster_df["template_amp"] = self.cluster_df["cluster_id"].map(cluster_to_template_amp)

    def _build_cluster_to_template_map(self):
        """
        Returns dict: cluster_id -> template_index.
        For KS4, cluster ids and template indices are 0..n_clusters-1.
        For KS2/3 you can use spike_templates for spikes in each cluster.
        """
        # For KS4, assume cluster_id == template index
        # If you have a more accurate mapping already, keep that instead.
        n_templates = self.templates.shape[0] if hasattr(self, 'templates') else None
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
                    cluster_id_to_value = df.set_index("cluster_id")[value_col].to_dict()
                    self.cluster_df[col] = self.cluster_df["cluster_id"].map(cluster_id_to_value)
                except Exception as e:
                    logger.warning(f"Could not load {fname}: {e}")

    def _load_kilosort_similarity(self):
        """
        Load Kilosort similarity matrix (similar_templates.npy).
        """
        ks_dir = self.kilosort_dir
        try:
            self.similar_templates = np.load(ks_dir / "similar_templates.npy", mmap_mode="r")
            logger.debug("Loaded similar_templates.npy with shape: %s", self.similar_templates.shape)
        except FileNotFoundError:
            logger.warning("similar_templates.npy not found; MEA-based similarity will not be available")
            self.similar_templates = None

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

    def _get_mea_similarity_table(self, cluster_id: int):
        """
        Fast retrieval of precomputed similarity table.
        """
        import numpy as np
        import pandas as pd
        
        # Check if precomputed
        if (self.mea_similarity_matrix is None or 
            self.mea_sorted_indices is None or 
            self.cluster_id_to_idx is None):
            # Fall back to original method if not precomputed
            return self._get_mea_similarity_table_legacy(cluster_id)
        
        # Get index
        idx = self.cluster_id_to_idx.get(cluster_id)
        if idx is None:
            return pd.DataFrame()
        
        # Get pre-sorted indices
        sorted_indices = self.mea_sorted_indices[idx]
        
        # Get cluster IDs from indices
        all_cluster_ids = self.cluster_df['cluster_id'].values
        
        # Build DataFrame quickly
        result_ids = all_cluster_ids[sorted_indices]
        
        # Get template similarity and distance values
        template_sim = self.similar_templates if self.similar_templates is not None else None
        
        # Build DataFrame
        rows = []
        for j, other_idx in enumerate(sorted_indices[:50]):  # Limit to top 50
            other_id = all_cluster_ids[other_idx]
            
            # Get values from cluster_df
            other_row = self.cluster_df.iloc[other_idx]
            
            # Calculate distance
            x0, y0 = self.cluster_df.iloc[idx][['x_um', 'y_um']]
            x1, y1 = other_row[['x_um', 'y_um']]
            distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            
            # Get template similarity if available
            if template_sim is not None:
                cluster_to_template = self._build_cluster_to_template_map()
                t1 = cluster_to_template[cluster_id]
                t2 = cluster_to_template[other_id]
                tpl_sim = float(template_sim[t1, t2]) if t1 < template_sim.shape[0] and t2 < template_sim.shape[1] else 0.0
            else:
                tpl_sim = 0.0
            
            rows.append({
                'cluster_id': int(other_id),
                'n_spikes': int(other_row['n_spikes']),
                'status': str(other_row['status']),
                'set': other_row.get('set', set()),
                'distance_um': float(distance),
                'template_sim': tpl_sim
            })
        
        return pd.DataFrame(rows)

    def _get_vision_similarity_table(self, cluster_id: int):
        """
        Get vision-based similarity table for a cluster.

        EI correlation matrices are computed lazily the first time this
        is called (via _compute_ei_correlations_if_needed).
        """
        import pandas as pd

        if cluster_id in self.vision_sim_cache:
            return self.vision_sim_cache[cluster_id]

        # Ensure correlations exist (may trigger a one-time heavy compute)
        self._compute_ei_correlations_if_needed()

        # Check again if vision data is available
        if self.ei_corr_dict is None or self.vision_eis is None:
            logger.error("Vision data not available for similarity table")
            return pd.DataFrame()  # Return empty DataFrame

        # Get vision correlation values for the main cluster
        vision_cluster_ids = np.array(list(self.vision_eis.keys()))
        kilosort_cluster_ids = vision_cluster_ids - 1  # Convert to 0-indexed

        # Check if the selected cluster exists in vision data
        cluster_match_idx = np.where(kilosort_cluster_ids == cluster_id)[0]
        if len(cluster_match_idx) == 0:
            logger.warning("Cluster ID %s not found in Vision data", cluster_id)
            return pd.DataFrame()

        main_idx = cluster_match_idx[0]  # Get the first match index
        other_idx = np.where(kilosort_cluster_ids != cluster_id)[0]
        other_ids = kilosort_cluster_ids[other_idx]

        # Only include other_ids that actually exist in the main cluster dataframe
        valid_cluster_df_ids = set(self.cluster_df["cluster_id"].values)
        valid_other_ids = [oid for oid in other_ids if oid in valid_cluster_df_ids]

        if not valid_other_ids:
            logger.warning(
                "No valid other cluster IDs found in Vision data for main cluster %s",
                cluster_id,
            )
            return pd.DataFrame()

        # Map valid other ids back to their Vision index positions
        valid_other_idx = []
        for oid in valid_other_ids:
            matches = np.where(kilosort_cluster_ids == oid)[0]
            if len(matches) > 0:
                valid_other_idx.append(matches[0])

        valid_other_idx = np.array(valid_other_idx, dtype=int)
        valid_other_ids = np.array(valid_other_ids, dtype=int)

        # Build DataFrame with EI correlation values
        d_df = {
            "cluster_id": valid_other_ids,
            "space_ei_corr": self.ei_corr_dict["space"][main_idx, valid_other_idx],
            "full_ei_corr": self.ei_corr_dict["full"][main_idx, valid_other_idx],
            "power_ei_corr": self.ei_corr_dict["power"][main_idx, valid_other_idx],
        }
        df = pd.DataFrame(d_df)

        # Add n_spikes, status, set from main cluster_df
        cluster_df = self.cluster_df
        n_spikes_map = dict(zip(cluster_df["cluster_id"], cluster_df["n_spikes"]))
        df["n_spikes"] = df["cluster_id"].map(n_spikes_map)

        status_map = dict(zip(cluster_df["cluster_id"], cluster_df["status"]))
        df["status"] = df["cluster_id"].map(status_map)

        set_map = dict(zip(cluster_df["cluster_id"], cluster_df["set"]))
        df["set"] = df["cluster_id"].map(set_map)

        # Sort by EI correlation
        df = df.sort_values(by="space_ei_corr", ascending=False).reset_index(drop=True)

        # Potential duplicate flag (Vision-side)
        df["potential_dups"] = (
            (df["full_ei_corr"].astype(float) > EI_CORR_THRESHOLD)
            | (df["space_ei_corr"].astype(float) > EI_CORR_THRESHOLD)
            | (df["power_ei_corr"].astype(float) > EI_CORR_THRESHOLD)
        )

        # Format correlation columns to 2 decimal places
        for col in ["full_ei_corr", "space_ei_corr", "power_ei_corr"]:
            df[col] = df[col].map(lambda x: f"{x:.2f}")

        df["potential_dups"] = df["potential_dups"].map(lambda x: "Yes" if x else "")

        self.vision_sim_cache[cluster_id] = df
        return df

    def _precompute_mea_similarity(self):
        """
        Precompute the complete MEA similarity matrix and sorted indices.
        Called once during data loading.
        """
        if self.similar_templates is None:
            return
        
        n_clusters = len(self.cluster_df)
        
        # Create mapping from cluster_id to index
        self.cluster_id_to_idx = {
            cid: idx for idx, cid in enumerate(self.cluster_df['cluster_id'])
        }
        
        # Get positions
        x_pos = self.cluster_df['x_um'].values
        y_pos = self.cluster_df['y_um'].values
        
        # Build template similarity matrix
        template_matrix = np.zeros((n_clusters, n_clusters))
        distance_matrix = np.zeros((n_clusters, n_clusters))
        
        # Precompute template indices
        cluster_to_template = self._build_cluster_to_template_map()
        
        # Fill matrices
        for i, cid_i in enumerate(self.cluster_df['cluster_id']):
            template_i = cluster_to_template[cid_i]
            
            for j, cid_j in enumerate(self.cluster_df['cluster_id']):
                if i == j:
                    continue
                    
                template_j = cluster_to_template[cid_j]
                
                # Template similarity
                template_matrix[i, j] = self.similar_templates[template_i, template_j]
                
                # Euclidean distance
                dx = x_pos[i] - x_pos[j]
                dy = y_pos[i] - y_pos[j]
                distance_matrix[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Combine into single similarity score (weight distance more)
        # Formula: similarity = template_sim - (distance / max_distance) * distance_weight
        max_dist = distance_matrix.max() if distance_matrix.max() > 0 else 1
        distance_weight = 0.7  # Weight distance vs template similarity
        
        self.mea_similarity_matrix = template_matrix - (distance_matrix / max_dist) * distance_weight
        
        # Pre-sort indices for each cluster
        self.mea_sorted_indices = []
        for i in range(n_clusters):
            # Get indices sorted by similarity (descending)
            sorted_idx = np.argsort(-self.mea_similarity_matrix[i])
            # Remove self
            sorted_idx = sorted_idx[sorted_idx != i]
            self.mea_sorted_indices.append(sorted_idx)

    def _precompute_mea_similarity_vectorized(self):
        """
        Vectorized precomputation of MEA similarity matrix.
        Called once during data loading for instant similarity table access.
        """
        if self.similar_templates is None:
            logger.debug("similar_templates.npy not available, skipping MEA similarity precomputation")
            return
        
        n_clusters = len(self.cluster_df)
        logger.debug(f"Precomputing MEA similarity matrix for {n_clusters} clusters")
        
        # Create bidirectional mappings between cluster_id and index
        self.cluster_id_to_idx = {}
        self.cluster_idx_to_id = []
        
        for idx, cid in enumerate(self.cluster_df['cluster_id']):
            self.cluster_id_to_idx[cid] = idx
            self.cluster_idx_to_id.append(cid)
        
        # Get positions
        x_pos = self.cluster_df['x_um'].fillna(0).values.astype(np.float32)
        y_pos = self.cluster_df['y_um'].fillna(0).values.astype(np.float32)
        
        # Precompute template indices
        cluster_to_template = self._build_cluster_to_template_map()
        
        # Build template index array
        template_indices = np.zeros(n_clusters, dtype=np.int32)
        for i, cid in enumerate(self.cluster_df['cluster_id']):
            template_indices[i] = cluster_to_template.get(cid, -1)
        
        # Calculate distance matrix using vectorized operations
        # Broadcast x and y positions to create all pairwise differences
        x_diff = x_pos[:, np.newaxis] - x_pos[np.newaxis, :]
        y_diff = y_pos[:, np.newaxis] - y_pos[np.newaxis, :]
        distance_matrix = np.sqrt(x_diff**2 + y_diff**2).astype(np.float32)
        
        # Calculate template similarity matrix
        n_templates = self.similar_templates.shape[0]
        template_matrix = np.zeros((n_clusters, n_clusters), dtype=np.float32)
        
        # Create mask for valid template indices
        valid_i = template_indices >= 0
        valid_i_idx = np.where(valid_i)[0]
        
        # Only compute similarities for clusters with valid template indices
        for i in valid_i_idx:
            t_i = template_indices[i]
            if t_i >= n_templates:
                continue
                
            # Get template similarities for this template
            for j in valid_i_idx:
                if i == j:
                    continue
                t_j = template_indices[j]
                if t_j >= n_templates:
                    continue
                template_matrix[i, j] = self.similar_templates[t_i, t_j]
        
        # Normalize distance matrix (0-1 range)
        max_dist = distance_matrix.max()
        if max_dist > 0:
            norm_distance = distance_matrix / max_dist
        else:
            norm_distance = np.zeros_like(distance_matrix)
        
        # Set diagonal to 0
        np.fill_diagonal(norm_distance, 0)
        
        # Combine into similarity score (distance is negative influence)
        # Weight factors can be adjusted
        template_weight = 0.3
        distance_weight = 0.7
        
        self.mea_similarity_matrix = (
            template_matrix * template_weight - 
            norm_distance * distance_weight
        ).astype(np.float32)
        
        # Pre-sort indices for each cluster (descending similarity)
        self.mea_sorted_indices = []
        for i in range(n_clusters):
            # Get all similarities for this cluster
            similarities = self.mea_similarity_matrix[i]
            
            # Sort indices by similarity (descending)
            sorted_idx = np.argsort(-similarities)
            
            # Remove self from the list
            sorted_idx = sorted_idx[sorted_idx != i]
            
            # Limit to top N (e.g., 100) for memory efficiency
            if len(sorted_idx) > 100:
                sorted_idx = sorted_idx[:100]
            
            self.mea_sorted_indices.append(sorted_idx.astype(np.int32))
        
        logger.debug(f"MEA similarity precomputation complete. Matrix shape: {self.mea_similarity_matrix.shape}")


