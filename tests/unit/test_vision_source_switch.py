"""Attaching different Vision files must not reuse the old ones' results.

PLAN.md Q11. Every Vision-derived cache is keyed by cluster id only, so
File > Load Vision with another folder kept the old STA timecourses, RF
fits and EI duplicate flags. Entries now carry the source they came from.
"""

import pickle
import threading

import numpy as np

from src.analysis.data_manager import DataManager

A = "/runs/A::data018"
B = "/runs/B::data018"


def _dm(source):
    dm = DataManager.__new__(DataManager)
    dm._vision_source = source
    dm._feature_lock = threading.Lock()
    dm._heavyweight_lock = threading.Lock()
    dm.feature_cache = {}
    dm.heavyweight_cache = {}
    dm.vision_sim_cache = {}
    dm.vision_params = None
    dm.vision_stas = None
    dm.vision_eis = None
    dm.reference_bridge = None
    dm.is_vision_only = False
    dm.ei_corr_dict = None
    return dm


def _entry(source):
    e = {"_computed": True, "timecourse": np.ones(5)}
    if source is not None:
        e["_vision_source"] = source
    return e


def test_entry_from_other_vision_files_is_stale():
    dm = _dm(B)
    assert not dm._physics_entry_is_fresh(0, _entry(A))
    assert dm._physics_entry_is_fresh(0, _entry(B))


def test_untagged_legacy_entry_is_kept():
    assert _dm(B)._physics_entry_is_fresh(0, _entry(None))


def test_source_change_closes_readers_and_clears_caches():
    dm = _dm(A)
    closed = []

    class _Reader:
        def __init__(self, name):
            self.name = name

        def close(self):
            closed.append(self.name)

    dm.vision_stas, dm.vision_eis = _Reader("sta"), _Reader("ei")
    dm.feature_cache[0] = _entry(A)
    dm.heavyweight_cache[0] = {"x": 1}
    dm.vision_sim_cache[0] = "rows"
    dm.ei_corr_dict = {"full": np.eye(2)}
    dm._physics_done_count = 5

    dm._forget_vision_derived(A, B)

    assert sorted(closed) == ["ei", "sta"]
    assert dm.vision_stas is None and dm.vision_eis is None
    assert dm.feature_cache == {} and dm.heavyweight_cache == {}
    assert dm.vision_sim_cache == {} and dm.ei_corr_dict is None
    assert dm._physics_done_count == 0


def test_ei_corr_pickle_from_other_vision_files_is_refused(tmp_path):
    path = tmp_path / "ei_corr_dict.pkl"
    m = np.eye(3)
    with open(path, "wb") as f:
        pickle.dump({"full": m, "space": m, "power": m, "ids": [1, 2, 3],
                     "vision_source": A}, f)
    assert _dm(B)._load_cached_ei_corr(str(path)) == (None, None)
    cached, ids = _dm(A)._load_cached_ei_corr(str(path))
    assert ids == [1, 2, 3]
