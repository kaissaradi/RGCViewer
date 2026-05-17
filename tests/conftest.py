import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib
import pytest

# -----------------------------------------------------------------------------
# Headless environment for CI and headless runs
# -----------------------------------------------------------------------------
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

# Force matplotlib to use the Agg backend (no GUI)
matplotlib.use("Agg", force=True)

# -----------------------------------------------------------------------------
# Common fixtures
# -----------------------------------------------------------------------------
from src.analysis.data_manager import DataManager


@pytest.fixture
def rgc_colors():
    """Standard dark theme colors for testing UI components."""
    return {
        "bg_base": "#111214",
        "bg_panel": "#18191C",
        "bg_surface": "#1E2025",
        "border_default": "#3D3F48",
        "border_subtle": "#2E3038",
        "text_primary": "#F0F0F2",
        "text_secondary": "#9B9DA6",
        "text_tertiary": "#5A5C65",
        "text_disabled": "#3A3C44",
        "accent": "#2E6DD4",
        "accent_hover": "#7EB8F7",
        "accent_positive": "#1A5C3A",
    }


@pytest.fixture
def mock_dm():
    """
    Minimal DataManager mock for unit tests that need DM but not actual data.
    Use this for logic tests that don't require disk I/O.
    """
    dm = DataManager(kilosort_dir="/tmp")
    dm.sampling_rate = 30000
    dm.n_channels = 384
    # Mock expensive or external dependencies
    dm.get_cluster_spike_amplitudes = MagicMock(return_value=[])
    dm.get_cluster_spikes = MagicMock(return_value=[])
    return dm


@pytest.fixture
def make_main_window(qtbot, rgc_colors):
    """
    Factory fixture to create a minimal mock QMainWindow with essential attributes.
    Use this when you need a main_window but don't want a full instantiation.
    """
    from qtpy.QtWidgets import QMainWindow

    class MockMainWindow(QMainWindow):
        def __init__(self, data_manager=None):
            super().__init__()
            self.data_manager = data_manager if data_manager is not None else MagicMock()
            self.status_bar = MagicMock()
            self.tree_view = None
            self.tree_model = None
            self.similarity_panel = None
            self._selected_cluster_id = None
            self.update_cluster_views = MagicMock()

    return MockMainWindow


@pytest.fixture
def real_data_manager():
    """
    Provides a DataManager pointed at the real lab dataset.
    Automatically skips the test if the network drive is unmounted or missing.
    """
    target_path = Path("/mnt/lab/Array-data/sorted/20260506A/chunk10/kilosort2.5")
    if not target_path.exists():
        pytest.skip(f"Real data path {target_path} not found. Skipping integration test.")
    dm = DataManager(kilosort_dir=str(target_path))
    return dm


@pytest.fixture
def cache_cleared_data_manager(tmp_path):
    """
    Copies a tiny subset of real data to a temporary directory AND deletes the .pkl
    files to guarantee that physics math is actually re‑calculated during the test.
    """
    source_path = Path("/mnt/lab/Array-data/sorted/20260506A/chunk10/kilosort2.5")
    if not source_path.exists():
        pytest.skip("Real data path missing. Cannot test cache invalidation.")

    # Copy files to an isolated temporary directory for safe mutation
    shutil.copytree(source_path, tmp_path, dirs_exist_ok=True)

    # Force deletion of any pre-existing cache files so we know math runs
    cache_files = ["standard_plot_cache.pkl", "feature_cache.pkl", "ei_corr_dict.pkl"]
    for cache_name in cache_files:
        file_to_remove = tmp_path / cache_name
        if file_to_remove.exists():
            file_to_remove.unlink()

    dm = DataManager(kilosort_dir=str(tmp_path))
    return dm