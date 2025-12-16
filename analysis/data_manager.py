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
        print(f"[DEBUG] Initializing DataManager for experiment: {self.exp_name}, datafile: {self.datafile_name}")
        self.load_stim_timing()
        self.ei_cache = {}
        self.heavyweight_cache = {}
        self.isi_cache = {}  # Cache for ISI violation calculations
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

        # --- Vision Data ---
        self.vision_eis = None
        self.vision_stas = None
        self.vision_params = None
        self.vision_channel_positions = None # Store channel positions from vision data
        self.vision_sta_width = None  # Store stimulus width for coordinate alignment
        self.vision_sta_height = None  # Store stimulus height for coordinate alignment
        self.ei_corr_dict = None  # Initialize to None, will be set when vision data is loaded

        # Initialize raw data memmap attribute (will hold memmap object)
        self.raw_data_memmap = None

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
            print(f"[DEBUG] Successfully saved pickle to {filepath}")
            return filepath
        except PermissionError:
            # If permission denied, save to a temporary location
            temp_dir = Path(tempfile.gettempdir())
            filename = Path(filepath).name
            temp_path = temp_dir / filename

            try:
                with open(temp_path, 'wb') as f:
                    pickle.dump(data, f)
                print(f"[DEBUG] Saved pickle to temporary location: {temp_path} (original location was not writable)")
                return temp_path
            except Exception as e:
                print(f"[ERROR] Failed to save pickle to both original and temporary locations: {e}")
                raise e
        except Exception as e:
            print(f"[ERROR] Failed to save pickle: {e}")
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
                print(f"[WARN] Skipping EI for key {k}: invalid or empty EI data")
        return out

    def load_stim_timing(self):
        try:
            import retinanalysis.utils.datajoint_utils as dju
            self.block_id = dju.get_block_id_from_datafile(self.exp_name, self.datafile_name)
            self.d_timing = dju.get_epochblock_timing(self.exp_name, self.block_id)
            print("[DEBUG] Loaded stimulus timing data successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load stimulus timing data: {e}")
            return

    def update_and_export_status(self, selected_ids, status):
        selected_ids = set(selected_ids)
        print(f'[DEBUG] Marking {status}: {selected_ids}')
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
            print(f"[DEBUG] Exported {len(self.status_df)} status entries to {self.status_csv}")
        except Exception as e:
            print(f"[ERROR] Failed to export status entries: {e}")

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

            print(f"[DEBUG] Loaded {self.status_csv}")
            print(f"[DEBUG] {self.status_df['status'].value_counts().to_dict()}")

            self.update_cluster_df_with_status()

            return True
        except Exception as e:
            print(f"[ERROR] Failed to load duplicate sets: {e}")
            return False

    def load_kilosort_data(self):
        try:
            self.spike_times = np.load(self.kilosort_dir / 'spike_times.npy').flatten()
            self.spike_clusters = np.load(self.kilosort_dir / 'spike_clusters.npy').flatten()
            self.channel_positions = np.load(self.kilosort_dir / 'channel_positions.npy')
            self.sorted_channels = sort_electrode_map(self.channel_positions)

            # templates is (n_clusters, n_timepoints, n_channels)
            self.templates = np.load(self.kilosort_dir / 'templates.npy')
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
                print("Info: 'cluster_info.tsv' or 'cluster_group.tsv' not found. Labeling all clusters as 'unsorted'.")
                self.info_path = None
                all_cluster_ids = np.unique(self.spike_clusters)
                self.cluster_info = pd.DataFrame({
                    'cluster_id': all_cluster_ids,
                    'group': ['unsorted'] * len(all_cluster_ids)
                })

            self._load_kilosort_params()
            return True, "Successfully loaded Kilosort data."
        except Exception as e:
            return False, f"Error during Kilosort data loading: {e}"

    def load_vision_data(self, vision_dir, dataset_name):
        """
        Loads EI, STA, and params data from a specified Vision directory.
        """
        print(f"[DEBUG] Starting to load vision data from {vision_dir}")  # Debug
        vision_path = Path(vision_dir)

        # To get the STA dimensions, we need to access them directly from the STAReader
        # The STAReader has width and height attributes that represent the stimulus dimensions
        print(f"[DEBUG] About to call vision_integration.load_vision_data")  # Debug
        vision_data = vision_integration.load_vision_data(vision_path, dataset_name)
        print(f"[DEBUG] Completed vision_integration.load_vision_data call")  # Debug

        if vision_data:
            ei_bundle = vision_data.get('ei')
            if ei_bundle:
                self.vision_eis = ei_bundle.get('ei_data')
                self.vision_channel_positions = ei_bundle.get('electrode_map')
                if self.vision_eis:
                    print(f"[DEBUG] Available Vision EI IDs (sample): {list(self.vision_eis.keys())[:10]}")

            self.vision_stas = vision_data.get('sta')
            self.vision_params = vision_data.get('params')

            # Extract and store stimulus dimensions for coordinate alignment
            # Get dimensions from the STA data structure if available
            if self.vision_stas and len(self.vision_stas) > 0:
                # Get the first available STA to extract dimensions
                first_cell_id = next(iter(self.vision_stas))
                first_sta = self.vision_stas[first_cell_id]

                # The STA structure is likely a container with red, green, blue channels
                # Extract spatial dimensions from the shape of one of the channels
                if hasattr(first_sta, 'red') and first_sta.red is not None:
                    # Get the dimensions from the STA container
                    # Use the shape of the red channel to extract width and height
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
                    print("Warning: Could not extract dimensions from STA data, using defaults.")
                    self.vision_sta_width = 100
                    self.vision_sta_height = 100
            else:
                # Fallback if no STA data is available
                print("Warning: No STA data available to extract dimensions, using defaults.")
                self.vision_sta_width = 100
                self.vision_sta_height = 100

            # Compute correlation matrices for EIs
            str_corr_pkl = os.path.join(self.kilosort_dir, 'ei_corr_dict.pkl')

            # Sanitize loaded EIs before any numeric operations
            sanitized_eis = self._sanitize_ei_dict(self.vision_eis)

            if len(sanitized_eis) >= 2:
                if os.path.exists(str_corr_pkl):
                    try:
                        print(f"[DEBUG] Loading precomputed EI correlations from {str_corr_pkl}")
                        with open(str_corr_pkl, 'rb') as f:
                            self.ei_corr_dict = pickle.load(f)
                        print(f"[DEBUG] Loaded EI correlations successfully")
                        full_corr = self.ei_corr_dict.get('full')
                        space_corr = self.ei_corr_dict.get('space')
                        power_corr = self.ei_corr_dict.get('power')
                    except Exception as e:
                        print(f"[WARN] Failed to load EI correlation pickle: {e}. Recomputing.")
                        full_corr = ei_corr(sanitized_eis, sanitized_eis, method='full', n_removed_channels=1)
                        space_corr = ei_corr(sanitized_eis, sanitized_eis, method='space', n_removed_channels=1)
                        power_corr = ei_corr(sanitized_eis, sanitized_eis, method='power', n_removed_channels=1)
                        self.ei_corr_dict = {
                            'full': full_corr,
                            'space': space_corr,
                            'power': power_corr
                        }
                        saved_path = self._save_pickle_with_fallback(self.ei_corr_dict, str_corr_pkl)
                        print(f"[DEBUG] EI correlations recomputed and saved to {saved_path}")
                else:
                    print(f'[DEBUG] Computing EI correlations')
                    full_corr = ei_corr(sanitized_eis, sanitized_eis, method='full', n_removed_channels=1)
                    space_corr = ei_corr(sanitized_eis, sanitized_eis, method='space', n_removed_channels=1)
                    power_corr = ei_corr(sanitized_eis, sanitized_eis, method='power', n_removed_channels=1)
                    self.ei_corr_dict = {
                        'full': full_corr,
                        'space': space_corr,
                        'power': power_corr
                    }
                    saved_path = self._save_pickle_with_fallback(self.ei_corr_dict, str_corr_pkl)
                    print(f"[DEBUG] EI correlations computed successfully and saved to {saved_path}")

                # Update cluster_df to mark potential duplicates based on any EI correlation > threshold
                if not self.cluster_df.empty:
                    # For each cluster, check if any other cluster has a correlation > threshold
                    cluster_ids = list(sanitized_eis.keys())
                    cluster_ids = np.array(cluster_ids) - 1 # Vision to ks IDs
                    potential_dups_map = {}
                    max_dup_r_map = {}
                    for i, cid in enumerate(cluster_ids):
                        # Exclude self-comparison by masking the diagonal
                        full_mask = np.delete(full_corr[i, :], i)
                        space_mask = np.delete(space_corr[i, :], i)
                        power_mask = np.delete(power_corr[i, :], i)
                        if (
                            np.any(full_mask > EI_CORR_THRESHOLD) or
                            np.any(space_mask > EI_CORR_THRESHOLD) or
                            np.any(power_mask > EI_CORR_THRESHOLD)
                        ):
                            potential_dups_map[cid] = True
                        max_r = max(
                            np.max(full_mask),
                            np.max(space_mask),
                            # np.max(power_mask)
                        )
                        max_dup_r_map[cid] = max_r

                    self.cluster_df['potential_dups'] = self.cluster_df['cluster_id'].map(potential_dups_map).fillna(False)
                    self.cluster_df['max_dup_r'] = self.cluster_df['cluster_id'].map(max_dup_r_map).fillna(0.0)
                    # Sort cluster_df by max_dup_r descending
                    self.cluster_df = self.cluster_df.sort_values(by='max_dup_r', ascending=False).reset_index(drop=True)

                    # Format to 2 decimal places
                    self.cluster_df['max_dup_r'] = self.cluster_df['max_dup_r'].map(lambda x: f"{x:.2f}")

                    print(f"[DEBUG] Updated cluster_df with potential duplicates based on EI correlations")
            else:
                print("[WARN] Vision EIs are empty or not loaded. Skipping duplicate detection.")
                # Initialize columns so the GUI doesn't crash later if it expects them
                if not self.cluster_df.empty:
                    self.cluster_df['potential_dups'] = False
                    self.cluster_df['max_dup_r'] = 0.0

            # (Duplicate correlation-to-cluster_df update removed - handled above)

            print(f"Vision data has been loaded into the DataManager. STA dimensions: {self.vision_sta_width}x{self.vision_sta_height}")
            return True, f"Successfully loaded Vision data for {dataset_name}."
        else:
            print(f"[DEBUG] Full vision loading failed, trying partial loading") # Debug
            # Check if params or STA files exist even if the full loading failed
            params_path = vision_path / 'sta_params.params'
            sta_path = vision_path / 'sta_container.sta'

            if params_path.exists() or sta_path.exists():
                print(f"[DEBUG] Found params or sta files, attempting partial load") # Debug
                # Try to load the existing files one by one using the available functions
                vision_data = {}
                if params_path.exists():
                    print(f"[DEBUG] Loading params data...") # Debug
                    try:
                        # Using the actual function name from vision_integration.py
                        params_data = vision_integration.load_params_data(vision_path, dataset_name)
                        vision_data['params'] = params_data
                        print("Loaded Vision params data.")
                    except Exception as e:
                        print(f"Error loading params: {e}")

                if sta_path.exists():
                    print(f"[DEBUG] Loading STA data...") # Debug
                    try:
                        # Using the actual function name from vision_integration.py
                        sta_data = vision_integration.load_sta_data(vision_path, dataset_name)
                        vision_data['sta'] = sta_data
                        print("Loaded Vision STA data.")

                        # Extract dimensions from STA if it was loaded
                        if sta_data:
                            first_cell_id = next(iter(sta_data))
                            first_sta = sta_data[first_cell_id]
                            if hasattr(first_sta, 'red') and first_sta.red is not None:
                                sta_shape = first_sta.red.shape
                                if len(sta_shape) >= 2:
                                    self.vision_sta_height = sta_shape[0]
                                    self.vision_sta_width = sta_shape[1]
                                else:
                                    self.vision_sta_height = sta_shape[0]
                                    self.vision_sta_width = sta_shape[1]
                    except Exception as e:
                        print(f"Error loading STA: {e}")

                # Update the instance variables with any loaded data
                ei_bundle = vision_data.get('ei')
                if ei_bundle:
                    self.vision_eis = ei_bundle.get('ei_data')
                    self.vision_channel_positions = ei_bundle.get('electrode_map')

                self.vision_stas = vision_data.get('sta')
                self.vision_params = vision_data.get('params')

                if vision_data:  # If we loaded any data
                    print(f"Partial Vision data has been loaded into the DataManager. STA dimensions: {self.vision_sta_width}x{self.vision_sta_height}")
                    return True, f"Successfully loaded partial Vision data for {dataset_name}."
                else:
                    print(f"[DEBUG] No vision data could be loaded") # Debug
                    return False, "Failed to load Vision data but files were found."
            else:
                print(f"[DEBUG] No vision files found in directory") # Debug
                return False, "No Vision data files found in the directory."
    # --- End New Method ---

    def load_cell_type_file(self, txt_file: str=None):
        print(f'[DEBUG] Loading cell type file: {txt_file}')
        if txt_file is None:
            print(f'[DEBUG] No cell type file provided, Unknown for all IDs.')
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
            print(f'[DEBUG] Loaded cell type file {txt_file}.')

            # If all are unknown, delete column and print
            if all(ct == 'Unknown' for ct in self.cluster_df['cell_type']):
                self.cluster_df.drop(columns=['cell_type'], inplace=True)
                print(f'[DEBUG] All cell type entries are Unknown, dropping cell_type column.')
        except Exception as e:
            print(f"Error loading cell type file: {e}")



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
        print(f"[DEBUG] Starting build_cluster_dataframe")  # Debug
        cluster_ids, n_spikes = np.unique(self.spike_clusters, return_counts=True)
        print(f"[DEBUG] Found {len(cluster_ids)} clusters")  # Debug

        # Create initial dataframe without ISI violations for faster loading
        df = pd.DataFrame({'cluster_id': cluster_ids, 'n_spikes': n_spikes})

        # Initialize potential_dups with False
        df['potential_dups'] = False

        # Initialize max_dup_r to 0
        df['max_dup_r'] = 0.0

        # Initialize cell types with 'Unknown'
        df['cell_type'] = 'Unknown'

        # Initialize ISI violations column with zeros for now
        df['isi_violations_pct'] = 0.0

        col = 'KSLabel' if 'KSLabel' in self.cluster_info.columns else 'group'
        if col not in self.cluster_info.columns: self.cluster_info[col] = 'unsorted'
        info_subset = self.cluster_info[['cluster_id', col]].rename(columns={col: 'KSLabel'})
        df = pd.merge(df, info_subset, on='cluster_id', how='left')

        # Status and set columns
        df['status'] = 'Original'
        df['set'] = [set([cid]) for cid in df['cluster_id']]
        df['set'] = df['set'].astype(object)

        self.cluster_df = df[['cluster_id', 'cell_type', 'n_spikes', 'isi_violations_pct', 'max_dup_r', 'potential_dups', 'status', 'set', 'KSLabel']]
        self.cluster_df['cluster_id'] = self.cluster_df['cluster_id'].astype(int)
        self.original_cluster_df = self.cluster_df.copy()
        print(f"[DEBUG] build_cluster_dataframe complete")

        # Initialize an empty cache for ISI violations
        self.isi_cache = {}
        # Calculate ISI violations for all clusters with optimized approach
        cluster_ids = self.cluster_df['cluster_id'].values
        isi_values = []

        # Calculate ISI for each cluster but show progress
        total_clusters = len(cluster_ids)
        for i, cluster_id in enumerate(cluster_ids):
            isi_value = self._calculate_isi_violations(cluster_id, refractory_period_ms=ISI_REFRACTORY_PERIOD_MS)
            isi_values.append(isi_value)

            # Print progress every 50 clusters to avoid too much output
            if (i + 1) % 50 == 0 or i == total_clusters - 1:
                print(f"[DEBUG] Calculated ISI for {i + 1}/{total_clusters} clusters")

        self.cluster_df['isi_violations_pct'] = isi_values
        print(f"[DEBUG] ISI violations calculated and updated in cluster_df")

        # Load status
        self.load_status()



    def get_cluster_spikes(self, cluster_id):
        return self.spike_times[self.spike_clusters == cluster_id]

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
        if cluster_id in self.heavyweight_cache: return self.heavyweight_cache[cluster_id]
        lightweight_data = self.get_lightweight_features(cluster_id)
        if not lightweight_data: return None
        features = analysis_core.compute_spatial_features(
            lightweight_data['median_ei'], self.channel_positions, self.sampling_rate)
        self.heavyweight_cache[cluster_id] = features
        return features

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
