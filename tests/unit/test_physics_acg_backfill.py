"""A warm open must not lose the ACG block (physics entries saved without one).

The ACG reached feature_cache only when _compute_standard_plots ran. With a
warm standard_plot_cache.pkl it does not run, and a physics entry saved
earlier without "acg" counted as fresh forever: every warm-opened run had no
ACG PCs in Feature Extraction or the UMAP (found on 20251212A/data018:
324/324 cells had acg_norm in the standard cache, 0/324 in physics).
"""

import threading

import numpy as np

from src.analysis.data_manager import DataManager


def _dm():
    dm = DataManager.__new__(DataManager)
    dm.is_vision_only = False
    dm._feature_lock = threading.Lock()
    dm._standard_plot_lock = threading.Lock()
    dm._physics_cell_locks = {}
    dm._physics_cell_locks_lock = threading.Lock()
    dm._vision_source = "src"
    dm.feature_cache = {4: {"_computed": True, "timecourse": np.ones(30),
                            "rf_area": 12.0, "_vision_source": "src"}}
    dm.standard_plot_cache = {4: {"acg_norm": np.arange(201.0)}}
    return dm


def test_fresh_physics_entry_gets_the_acg_from_the_standard_cache():
    dm = _dm()
    phys = dm.get_cell_physics(4)
    np.testing.assert_array_equal(phys["acg"], np.arange(201.0))
    assert dm.feature_cache[4]["acg"] is phys["acg"]      # stored, so it is saved


def test_no_standard_cache_entry_leaves_the_physics_entry_alone():
    dm = _dm()
    dm.standard_plot_cache = {}
    phys = dm.get_cell_physics(4)
    assert phys.get("acg") is None and phys["rf_area"] == 12.0
