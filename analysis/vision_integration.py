import os
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

def load_vision_data(vision_dir: Path, dataset_name: str):
    """
    Loads all relevant data from Vision files (.ei, .sta, .params).

    Args:
        vision_dir (Path): The directory containing the Vision files.
        dataset_name (str): The base name of the dataset (e.g., 'data000').

    Returns:
        dict: A dictionary containing the loaded EI, STA, and parameters data.
              Returns None for a data type if it fails to load.
    """
    logger.debug(f"Loading Vision files from: {vision_dir}")

    # If visionloader isn't available, return safe empty values so callers can continue
    if not VISION_LOADER_AVAILABLE:
        logger.warning(f"visionloader is not available; skipping vision load for {vision_dir}")
        return {'ei': None, 'sta': None, 'params': None}

    ei_data = None
    sta_data = None
    params_data = None

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

    return {
        'ei': ei_data,
        'sta': sta_data,
        'params': params_data
    }

def load_ei_data(vision_dir: Path, dataset_name: str):
    """
    Loads Electrical Image (EI) data from a .ei file.

    Args:
        vision_dir (Path): The directory containing the .ei file.
        dataset_name (str): The base name of the dataset.

    Returns:
        dict: A dictionary containing 'ei_data' (a dict of EIs keyed by cluster_id)
              and 'electrode_map' (the numpy array of channel positions).
    """
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
    except Exception as e:
        logger.exception("Unexpected error loading EI data")
        return None


def load_sta_data(vision_dir: Path, dataset_name: str):
    """
    Loads Spike-Triggered Average (STA) data from a .sta file.

    Args:
        vision_dir (Path): The directory containing the .sta file.
        dataset_name (str): The base name of the dataset.

    Returns:
        dict: A dictionary of STA data keyed by cluster_id.
    """
    if not VISION_LOADER_AVAILABLE:
        return None

    try:
        with vl.STAReader(str(vision_dir), dataset_name) as star:
            stas_by_cell_id = star.chunked_load_all_stas()
            logger.info(f"Loaded STAs for {len(stas_by_cell_id)} cells")
            return stas_by_cell_id
    except FileNotFoundError:
        logger.error(f"STA file not found in {vision_dir}")
        return None
    except Exception as e:
        logger.exception("Unexpected error loading STA data")
        return None

def load_params_data(vision_dir: Path, dataset_name: str):
    """
    Loads receptive field parameters from a .params file.

    Args:
        vision_dir (Path): The directory containing the .params file.
        dataset_name (str): The base name of the dataset.

    Returns:
        VisionCellDataTable: An object containing all the parameter data.
    """
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
    except Exception as e:
        logger.exception("Unexpected error loading params data")
        return None
