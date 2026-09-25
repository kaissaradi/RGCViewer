"""The Kilosort sample rate comes from params.py ``sample_rate`` (PLAN.md Q48).

Every lab params.py (1,000 files, 2026-09-25) says ``sample_rate = 20000``;
reading only ``fs`` made Encore use 30000 for all of them.
"""

import pickle
import threading
from pathlib import Path

import pytest

from src.analysis import data_manager as dmod
from src.analysis.data_manager import DataManager


def _dm(tmp_path, text):
    (tmp_path / "params.py").write_text(text)
    dm = DataManager.__new__(DataManager)
    dm.kilosort_dir = Path(tmp_path)
    return dm


@pytest.mark.parametrize("text, rate", [
    ("dat_path = 'x.dat'\nn_channels_dat = 512\ndtype = 'int16'\noffset = 0\nsample_rate = 20000.\nhp_filtered = True", 20000.0),
    ("sample_rate = 20000\n", 20000.0),
    ("fs = 25000\n", 25000.0),
    ("n_channels_dat = 512\n", 30000.0),
])
def test_params_rate(tmp_path, text, rate):
    dm = _dm(tmp_path, text)
    dm._load_kilosort_params()
    assert dm.sampling_rate == rate


def _cache_dm(tmp_path, rate):
    dm = DataManager.__new__(DataManager)
    dm.kilosort_dir = Path(tmp_path)
    dm.sampling_rate = rate
    dm.standard_plot_cache = {}
    dm._standard_plot_lock = threading.Lock()
    return dm


def test_standard_cache_made_at_another_rate_is_rebuilt(tmp_path):
    old = {5: {"acg_norm": [1, 2]}, dmod.SAMPLE_RATE_KEY: 30000.0}
    (tmp_path / "standard_plot_cache.pkl").write_bytes(pickle.dumps(old))
    dm = _cache_dm(tmp_path, 20000.0)
    dm.feature_cache = {5: {"acg": [9], "_computed": True}}
    dm._feature_lock = threading.Lock()
    dm._load_standard_plot_cache_from_disk()
    assert dm.standard_plot_cache == {} and dm._std_cache_rate_changed
    assert "acg" not in dm.feature_cache[5]                     # refilled from the rebuilt cache


def test_cache_restored_before_params_is_judged_when_the_rate_is_known(tmp_path):
    """DataManager.__init__ restores the cache before load_kilosort_data reads params.py."""
    (tmp_path / "standard_plot_cache.pkl").write_bytes(
        pickle.dumps({5: {"acg_norm": [1]}, dmod.SAMPLE_RATE_KEY: 20000.0}))
    (tmp_path / "params.py").write_text("sample_rate = 20000.\n")
    dm = _cache_dm(tmp_path, 0)
    del dm.sampling_rate
    dm._load_standard_plot_cache_from_disk()
    assert dm.standard_plot_cache                                # rate unknown: kept for now
    dm._load_kilosort_params()
    assert dm.standard_plot_cache == {5: {"acg_norm": [1]}}      # matching rate: kept


def test_unstamped_legacy_cache_is_rebuilt_and_a_matching_one_kept(tmp_path):
    (tmp_path / "standard_plot_cache.pkl").write_bytes(pickle.dumps({5: {"acg_norm": [1]}}))
    dm = _cache_dm(tmp_path, 20000.0)
    dm._load_standard_plot_cache_from_disk()
    assert dm.standard_plot_cache == {}
    good = {5: {"acg_norm": [1]}, dmod.SAMPLE_RATE_KEY: 20000.0}
    (tmp_path / "standard_plot_cache.pkl").write_bytes(pickle.dumps(good))
    dm = _cache_dm(tmp_path, 20000.0)
    dm._load_standard_plot_cache_from_disk()
    assert dm.standard_plot_cache == {5: {"acg_norm": [1]}}       # stamp not left in memory


def test_restore_path_with_an_old_rate_cache_does_not_hang(tmp_path):
    """The feature-cache restore held _feature_lock and re-took it: the load froze (Q48)."""
    from src.analysis import cache_persistence
    (tmp_path / "standard_plot_cache.pkl").write_bytes(
        pickle.dumps({5: {"acg_norm": [1]}}))                      # legacy: no stamp
    (tmp_path / "feature_cache.pkl").write_bytes(pickle.dumps(
        cache_persistence.add_version({5: {"acg": [9], "_computed": True}})))
    dm = _cache_dm(tmp_path, 20000.0)
    dm.feature_cache = {}
    dm._feature_lock = threading.Lock()
    done = threading.Event()

    def run():
        dm.load_persisted_caches()
        dm._load_standard_plot_cache_from_disk()                  # a second call must not re-read
        done.set()
    threading.Thread(target=run, daemon=True).start()
    assert done.wait(5), "load_persisted_caches deadlocked"
    assert dm.standard_plot_cache == {}
    assert dm.feature_cache[5].get("acg") is None and dm.feature_cache[5]["_computed"]
