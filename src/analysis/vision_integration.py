from pathlib import Path
import logging
logger = logging.getLogger(__name__)

# Guard import of visionloader so the app can run without it installed
try:
    import visionloader as vl
    VISION_LOADER_AVAILABLE = True
except Exception:
    vl = None
    VISION_LOADER_AVAILABLE = False
    logger.info("'visionloader' not available; vision integration disabled")


class LazySTADict:
    """
    A dictionary-like wrapper that keeps the .sta file pointer open 
    and reads STA movies directly from disk on-demand. 
    Includes a lightweight FIFO cache for smooth UI scrolling.
    """
    def __init__(self, vision_dir, dataset_name):
        self.reader = vl.STAReader(str(vision_dir), dataset_name)
        self.keys_list = list(self.reader.cell_id_to_byte_offset.keys())
        
        # --- NEW: LRU Cache to fix UI Scrolling lag ---
        self._cache = {}
        self._cache_keys = []
        self._max_cache = 40  # Keep the last 40 STAs in RAM
        
    def __contains__(self, key):
        return key in self.keys_list
    
    def get(self, key, default=None):
        """Emulate dict.get() using our internal cache and SSD reader."""
        if key in self:
            return self[key]  # This safely triggers __getitem__ and the RAM cache
        return default
        
    def __getitem__(self, key):
        # 1. Check RAM cache first (Instant)
        if key in self._cache:
            return self._cache[key]
            
        # 2. Read from SSD (Slow)
        sta_data = self.reader.get_sta_for_cell_id(key)
        
        # 3. Store in Cache
        self._cache[key] = sta_data
        self._cache_keys.append(key)
        
        # 4. Evict oldest if we exceed limit to save RAM
        if len(self._cache_keys) > self._max_cache:
            oldest_key = self._cache_keys.pop(0)
            if oldest_key in self._cache:
                del self._cache[oldest_key]
                
        return sta_data
        
    def __iter__(self):
        return iter(self.keys_list)
        
    def __len__(self):
        return len(self.keys_list)
        
    def keys(self):
        return self.keys_list
        
    def __del__(self):
        if hasattr(self, 'reader'):
            try:
                self.reader.close()
            except Exception:
                pass


def load_vision_data(vision_dir: Path, dataset_name: str):
    """
    Loads all relevant data from Vision files (.ei, .sta, .params, .neurons).
    """
    logger.debug(f"Loading Vision files from: {vision_dir}")

    if not VISION_LOADER_AVAILABLE:
        logger.warning(
            f"visionloader is not available; skipping vision load for {vision_dir}")
        return {'ei': None, 'sta': None, 'params': None, 'neurons': None}

    ei_data = None
    sta_data = None
    params_data = None
    neurons_data = None

    try:
        ei_data = load_ei_data(vision_dir, dataset_name)
    except Exception as e:
        logger.warning(f"Could not load EI data: {e}")

    try:
        sta_data = load_sta_data(vision_dir, dataset_name)
    except Exception as e:
        logger.warning(f"Could not load STA data: {e}")

    try:
        params_data = load_params_data(vision_dir, dataset_name)
    except Exception as e:
        logger.warning(f"Could not load Params data: {e}")
        
    try:
        neurons_data = load_neurons_data(vision_dir, dataset_name)
    except Exception as e:
        logger.warning(f"Could not load Neurons data: {e}")

    return {
        'ei': ei_data,
        'sta': sta_data,
        'params': params_data,
        'neurons': neurons_data
    }


def load_ei_data(vision_dir: Path, dataset_name: str):
    """Loads Electrical Image (EI) data from a .ei file."""
    if not VISION_LOADER_AVAILABLE:
        return None

    try:
        with vl.EIReader(str(vision_dir), dataset_name) as eir:
            eis_by_cell_id = eir.get_all_eis_by_cell_id()
            electrode_map = eir.get_electrode_map()
            logger.info(f"Loaded EIs for {len(eis_by_cell_id)} cells")
            return {'ei_data': eis_by_cell_id, 'electrode_map': electrode_map}
    except FileNotFoundError:
        logger.error(f"EI file not found in {vision_dir}")
        return None
    except Exception:
        logger.exception("Unexpected error loading EI data")
        return None


def load_sta_data(vision_dir: Path, dataset_name: str):
    """
    Initializes a Lazy Reader for STA data instead of loading into RAM.
    """
    if not VISION_LOADER_AVAILABLE:
        return None

    try:
        # We return the lazy dictionary wrapper instead of executing chunked_load_all_stas()
        lazy_sta_dict = LazySTADict(vision_dir, dataset_name)
        logger.info(f"Initialized Lazy STA Reader for {len(lazy_sta_dict)} cells")
        return lazy_sta_dict
    except FileNotFoundError:
        logger.error(f"STA file not found in {vision_dir}")
        return None
    except Exception:
        logger.exception("Unexpected error initializing Lazy STA Reader")
        return None


def load_params_data(vision_dir: Path, dataset_name: str):
    """Loads receptive field parameters from a .params file."""
    if not VISION_LOADER_AVAILABLE:
        return None

    try:
        vcd = vl.VisionCellDataTable()
        with vl.ParametersFileReader(str(vision_dir), dataset_name) as pfr:
            pfr.update_visioncelldata_obj(vcd)
            logger.info(f"Loaded params for {len(vcd.get_cell_ids())} cells")
            return vcd
    except FileNotFoundError:
        logger.error(f"Params file not found in {vision_dir}")
        return None
    except Exception:
        logger.exception("Unexpected error loading params data")
        return None


def load_neurons_data(vision_dir: Path, dataset_name: str):
    """Loads spike times, seed electrodes, and sampling rate from a .neurons file."""
    if not VISION_LOADER_AVAILABLE:
        return None

    try:
        with vl.NeuronsReader(str(vision_dir), dataset_name) as nr:
            spikes_by_id = nr.get_spike_sample_nums_for_all_real_neurons()
            seed_electrodes = nr.get_identifier_electrodes_for_all_real_neurons()
            sampling_rate = nr.sample_freq
            logger.info(f"Loaded .neurons for {len(spikes_by_id)} cells")
            return {
                'spikes_by_id': spikes_by_id, 
                'seed_electrodes': seed_electrodes,
                'sampling_rate': sampling_rate
            }
    except FileNotFoundError:
        logger.error(f"Neurons file not found in {vision_dir}")
        return None
    except Exception:
        logger.exception("Unexpected error loading Neurons data")
        return None