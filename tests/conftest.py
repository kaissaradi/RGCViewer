import os
import sys
import shutil
import threading
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib
import numpy as np
import pytest

# -----------------------------------------------------------------------------
# Headless environment for CI and headless runs
# -----------------------------------------------------------------------------
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg", force=True)

# -----------------------------------------------------------------------------
# Stub Qt BEFORE any src.* import touches it.
# data_manager.py has `from qtpy.QtCore import QObject` at module level —
# without this every test collection spin-initialises a Qt application.
# -----------------------------------------------------------------------------
for _mod in (
    "qtpy",
    "qtpy.QtCore",
    "qtpy.QtGui",
    "qtpy.QtWidgets",
):
    sys.modules.setdefault(_mod, MagicMock())

# QObject needs to behave like a real base class (DataManager inherits it).
# A plain MagicMock is not usable as a base class, so we give it a real one.
_QtCore = sys.modules["qtpy.QtCore"]
if not isinstance(_QtCore.QObject, type):
    class _QObjectStub:
        def __init__(self, *a, **kw): pass
        def __init_subclass__(cls, **kw): super().__init_subclass__(**kw)
    _QtCore.QObject = _QObjectStub
    _QtCore.Signal  = lambda *a, **kw: None
    _QtCore.Qt      = MagicMock()


# -----------------------------------------------------------------------------
# Common fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def rgc_colors():
    """Standard dark-theme color map for UI component tests."""
    return {
        "bg_base":        "#111214",
        "bg_panel":       "#18191C",
        "bg_surface":     "#1E2025",
        "border_default": "#3D3F48",
        "border_subtle":  "#2E3038",
        "text_primary":   "#F0F0F2",
        "text_secondary": "#9B9DA6",
        "text_tertiary":  "#5A5C65",
        "text_disabled":  "#3A3C44",
        "accent":         "#2E6DD4",
        "accent_hover":   "#7EB8F7",
        "accent_positive":"#1A5C3A",
    }


@pytest.fixture
def mock_dm():
    """
    Minimal DataManager stub for unit tests that need a DM but no disk I/O.
    Uses __new__ to skip __init__ entirely — no Qt, no DataJoint, no files.
    """
    from src.analysis.data_manager import DataManager

    dm = DataManager.__new__(DataManager)
    dm.sampling_rate   = 30000.0
    dm.n_channels      = 384
    dm.kilosort_dir    = Path("/tmp")
    dm.feature_cache   = {}
    dm.standard_plot_cache = {}
    dm.heavyweight_cache   = {}
    dm._feature_lock           = threading.Lock()
    dm._standard_plot_lock     = threading.Lock()
    dm._heavyweight_lock       = threading.Lock()
    dm._physics_cell_locks     = {}
    dm._physics_cell_locks_lock = threading.Lock()
    dm._physics_done_count     = 0
    dm.vision_stas   = None
    dm.vision_eis    = None
    dm.vision_params = None
    dm.cluster_df    = __import__("pandas").DataFrame()
    dm.get_cluster_spike_amplitudes = MagicMock(return_value=np.array([]))
    dm.get_cluster_spikes           = MagicMock(return_value=np.array([]))
    return dm


@pytest.fixture
def make_main_window(qtbot, rgc_colors):
    """
    Factory fixture for a minimal mock QMainWindow.
    Only use when you genuinely need a Qt event loop (pytest-qt tests).
    """
    from qtpy.QtWidgets import QMainWindow

    class MockMainWindow(QMainWindow):
        def __init__(self, data_manager=None):
            super().__init__()
            self.data_manager      = data_manager if data_manager is not None else MagicMock()
            self.status_bar        = MagicMock()
            self.tree_view         = None
            self.tree_model        = None
            self.similarity_panel  = None
            self._selected_cluster_id = None
            self.update_cluster_views = MagicMock()

    return MockMainWindow


@pytest.fixture
def real_data_manager():
    """
    DataManager pointed at the real lab dataset.
    Auto-skips if the network drive is unmounted.
    """
    from src.analysis.data_manager import DataManager

    target_path = Path("/mnt/lab/Array-data/sorted/20260506A/chunk10/kilosort2.5")
    if not target_path.exists():
        pytest.skip(f"Real data path {target_path} not found.")
    return DataManager(kilosort_dir=str(target_path))


@pytest.fixture
def cache_cleared_data_manager(tmp_path):
    """
    Copies a tiny subset of real data to tmp_path and deletes all .pkl caches
    so physics math is guaranteed to re-run during the test.
    """
    from src.analysis.data_manager import DataManager

    source_path = Path("/mnt/lab/Array-data/sorted/20260506A/chunk10/kilosort2.5")
    if not source_path.exists():
        pytest.skip("Real data path missing.")

    shutil.copytree(source_path, tmp_path, dirs_exist_ok=True)
    for name in ("standard_plot_cache.pkl", "feature_cache.pkl", "ei_corr_dict.pkl"):
        p = tmp_path / name
        if p.exists():
            p.unlink()

    return DataManager(kilosort_dir=str(tmp_path))