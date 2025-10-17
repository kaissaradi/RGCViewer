import numpy as np
import pandas as pd
import json
from pathlib import Path
from qtpy.QtCore import QObject, Qt
from qtpy.QtGui import QStandardItem
import analysis_core
import vision_integration
from constants import ISI_REFRACTORY_PERIOD_MS, EI_CORR_THRESHOLD, LS_CELL_TYPE_LABELS
import os
import pickle

def ei_corr(ref_ei_dict, test_ei_dict, 
            method: str = 'full', n_removed_channels: int = 1) -> np.ndarray:
    # Courtesy of @DRezeanu
    # Pull reference eis
    ref_ids = ref_ei_dict.keys()
    ref_eis = [ref_ei_dict[cell].ei for cell in ref_ids]

    if n_removed_channels > 0:
        max_ref_vals = [np.array(np.max(ei, axis = 1)) for ei in ref_eis]
        ref_to_remove = [np.argsort(val)[-n_removed_channels:] for val in max_ref_vals]
        ref_eis = [np.delete(ei, ref_to_remove[idx], axis = 0) for idx, ei in enumerate(ref_eis)]

    # Set any EI value where the ei is less than 1.5* its standard deviation to 0
    for idx, ei in enumerate(ref_eis):
        ref_eis[idx][abs(ei) < (ei.std()*1.5)] = 0

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


    # Pull test eis
    test_ids = test_ei_dict.keys()
    test_eis = [test_ei_dict[cell].ei for cell in test_ids]

    if n_removed_channels > 0:
        max_test_vals = [np.array(np.max(ei, axis = 1)) for ei in test_eis]
        test_to_remove = [np.argsort(val)[-n_removed_channels:] for val in max_test_vals]
        test_eis = [np.delete(ei, test_to_remove[idx], axis = 0) for idx, ei in enumerate(test_eis)]

    # Set the EI value where the EI is less than 1.5* its standard deviation to 0
    for idx, ei in enumerate(test_eis):
        test_eis[idx][abs(ei) < (ei.std()*1.5)] = 0

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


    num_pts = ref_eis.shape[1]

    # Calculate covariance and correlation
    c = test_eis @ ref_eis.T / num_pts
    d = np.mean(test_eis, axis = 1)[:,None] * np.mean(ref_eis, axis = 1)[:,None].T
    covs = c - d

    std_calc = np.std(test_eis, axis = 1)[:,None] * np.std(ref_eis, axis = 1)[:, None].T
    corr = covs / std_calc

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
        self.ei_cache = {}
        self.heavyweight_cache = {}
        self.isi_cache = {}  # Cache for ISI violation calculations
        self.dat_path = None
        self.cluster_df = pd.DataFrame()
        self.original_cluster_df = pd.DataFrame()
        self.info_path = None
        self.uV_per_bit = 0.195
        self.main_window = main_window  # Reference to main window for tree operations
        
        # List of duplicate sets
        self.duplicate_sets = []
        self.dup_json = self.kilosort_dir / 'duplicate_sets.json'
        
        # --- Vision Data ---
        self.vision_eis = None
        self.vision_stas = None
        self.vision_params = None
        self.vision_sta_width = None  # Store stimulus width for coordinate alignment
        self.vision_sta_height = None  # Store stimulus height for coordinate alignment
        
        # Initialize raw data memmap attribute (will hold memmap object)
        self.raw_data_memmap = None

    def export_duplicate_sets(self):
        """
        Export duplicate_sets to a JSON file in the Kilosort directory.
        """
        
        if not self.duplicate_sets:
            # Nothing to save
            return
        
        # Convert sets to lists for JSON serialization
        duplicate_sets_as_lists = [
            [int(cluster_id) for cluster_id in s] 
            for s in self.duplicate_sets
        ]
        
        try:
            with open(self.dup_json, 'w') as f:
                json.dump(duplicate_sets_as_lists, f, indent=2)
            print(f"[DEBUG] Exported {len(duplicate_sets_as_lists)} duplicate set(s) to {self.dup_json}")
        except Exception as e:
            print(f"[ERROR] Failed to export duplicate sets: {e}")
    
    def load_duplicate_sets(self):
        """
        Load duplicate_sets from a JSON file in the Kilosort directory.
        Returns True if file was found and loaded, False otherwise.
        """

        if not self.dup_json.exists():
            return False
        
        try:
            with open(self.dup_json, 'r') as f:
                duplicate_sets_as_lists = json.load(f)
            
            # Convert lists back to sets
            self.duplicate_sets = [set(s) for s in duplicate_sets_as_lists]

            print(f"[DEBUG] Loaded {len(self.duplicate_sets)} duplicate set(s) from {self.dup_json}")
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

    # --- New Method for Vision Data ---
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
            self.vision_eis = vision_data.get('ei')
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
            if os.path.exists(str_corr_pkl):
                print(f"[DEBUG] Loading precomputed EI correlations from {str_corr_pkl}")
                with open(str_corr_pkl, 'rb') as f:
                    self.ei_corr_dict = pickle.load(f)
                print(f"[DEBUG] Loaded EI correlations successfully")
                full_corr = self.ei_corr_dict.get('full')
                space_corr = self.ei_corr_dict.get('space')
                power_corr = self.ei_corr_dict.get('power')

            else:
                print(f'[DEBUG] Computing EI correlations')
                full_corr = ei_corr(self.vision_eis, self.vision_eis, method='full', n_removed_channels=1)
                space_corr = ei_corr(self.vision_eis, self.vision_eis, method='space', n_removed_channels=1)
                power_corr = ei_corr(self.vision_eis, self.vision_eis, method='power', n_removed_channels=1)
                self.ei_corr_dict = {
                    'full': full_corr,
                    'space': space_corr,
                    'power': power_corr
                }
                print(f"[DEBUG] EI correlations computed successfully")
                # Save to pickle for future use
                with open(str_corr_pkl, 'wb') as f:
                    pickle.dump(self.ei_corr_dict, f)
                print(f"[DEBUG] EI correlations saved to {str_corr_pkl}")

            # Update cluster_df to mark potential duplicates based on any EI correlation > threshold
            if not self.cluster_df.empty:
            # For each cluster, check if any other cluster has a correlation > threshold
                cluster_ids = list(self.vision_eis.keys())
                cluster_ids = np.array(cluster_ids) - 1 # Vision to ks IDs
                potential_dups_map = {}
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

                self.cluster_df['potential_dups'] = self.cluster_df['cluster_id'].map(potential_dups_map).fillna(False)

                print(f"[DEBUG] Updated cluster_df with potential duplicates based on EI correlations")

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
                self.vision_eis = vision_data.get('ei')
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
        if txt_file is None:
            print(f'[DEBUG] No cell type file provided, Unknown for all IDs.')
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

        # Initialize cell types with 'Unknown'
        df['cell_type'] = 'Unknown'
        
        # Initialize ISI violations column with zeros for now
        df['isi_violations_pct'] = 0.0
        
        col = 'KSLabel' if 'KSLabel' in self.cluster_info.columns else 'group'
        if col not in self.cluster_info.columns: self.cluster_info[col] = 'unsorted'
        info_subset = self.cluster_info[['cluster_id', col]].rename(columns={col: 'KSLabel'})
        df = pd.merge(df, info_subset, on='cluster_id', how='left')
        df['status'] = 'Original'
        self.cluster_df = df[['cluster_id', 'cell_type', 'n_spikes', 'isi_violations_pct', 'potential_dups', 'status', 'KSLabel']]
        self.cluster_df['cluster_id'] = self.cluster_df['cluster_id'].astype(int)
        self.original_cluster_df = self.cluster_df.copy()
        print(f"[DEBUG] build_cluster_dataframe complete")  # Debug
        
        # Initialize an empty cache for ISI violations that will be populated as clusters are selected
        self.isi_cache = {}

    

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
