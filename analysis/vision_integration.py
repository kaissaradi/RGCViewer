import os
from pathlib import Path

# Guard import of visionloader so the app can run without it installed
try:
    import visionloader as vl
    VISION_LOADER_AVAILABLE = True
except Exception:
    vl = None
    VISION_LOADER_AVAILABLE = False
    print("[INFO] 'visionloader' not available. Vision integration features will be disabled.")

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
    print(f"Loading Vision files from: {vision_dir}")

    # If visionloader isn't available, return safe empty values so callers can continue
    if not VISION_LOADER_AVAILABLE:
        print(f"[WARN] visionloader is not available; skipping vision load for {vision_dir}")
        return {'ei': None, 'sta': None, 'params': None}

    ei_data = None
    sta_data = None
    params_data = None

    try:
        ei_data = load_ei_data(vision_dir, dataset_name)
    except Exception as e:
        print(f"Warning: Could not load EI data. {e}")

    try:
        sta_data = load_sta_data(vision_dir, dataset_name)
    except Exception as e:
        print(f"Warning: Could not load STA data. {e}")

    try:
        params_data = load_params_data(vision_dir, dataset_name)
    except Exception as e:
        print(f"Warning: Could not load Params data. {e}")

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
        dict: A dictionary of EI data keyed by cluster_id.
    """
    if not VISION_LOADER_AVAILABLE:
        return None

    try:
        with vl.EIReader(str(vision_dir), dataset_name) as eir:
            eis_by_cell_id = eir.get_all_eis_by_cell_id()
            print(f"Successfully loaded EIs for {len(eis_by_cell_id)} cells.")
            return eis_by_cell_id
    except FileNotFoundError:
        print(f"Error: EI file not found in {vision_dir}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading EI data: {e}")
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
            print(f"Successfully loaded STAs for {len(stas_by_cell_id)} cells.")
            return stas_by_cell_id
    except FileNotFoundError:
        print(f"Error: STA file not found in {vision_dir}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading STA data: {e}")
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
            print(f"Successfully loaded params for {len(vcd.get_cell_ids())} cells.")
            return vcd
    except FileNotFoundError:
        print(f"Error: Params file not found in {vision_dir}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading params data: {e}")
        return None
