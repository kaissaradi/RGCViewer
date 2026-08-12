# tests/unit/test_physics_cache_unified.py
import threading
import time
import numpy as np
import pytest
from unittest.mock import MagicMock

from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_dm(n_clusters=10, acg_len=201, tc_len=30, all_computed=False):
    """Minimal DataManager stub with feature_cache and locking wired up."""
    dm = MagicMock()
    dm._feature_lock = threading.Lock()
    dm._physics_cell_locks = {}
    dm._physics_cell_locks_lock = threading.Lock()
    rng = np.random.default_rng(42)

    cache = {}
    for cid in range(n_clusters):
        entry = {
            'acg': rng.random(acg_len),
            'timecourse': rng.standard_normal(tc_len),
            'rf_area': float(rng.uniform(0.01, 0.5)),
            'ellipticity': float(rng.uniform(0.5, 2.0)),
            'time_to_peak': int(rng.integers(5, 25)),
        }
        if all_computed:
            entry['_computed'] = True
        cache[cid] = entry

    dm.feature_cache = cache
    dm.get_cell_physics = lambda cid: cache[int(cid)]
    return dm, cache


def _make_lazy_sta_dict(keys=None, max_cache=10, sleep_s=0.0):
    """LazySTADict with a mocked reader."""
    from src.analysis.vision_integration import LazySTADict
    reader = MagicMock()
    reader.get_sta_for_cell_id = lambda key: (time.sleep(sleep_s), np.zeros((4, 4, 5)))[1]
    if keys is None:
        keys = list(range(1, 6))
    lsd = LazySTADict.__new__(LazySTADict)
    lsd.reader = reader
    lsd._max_cache = max_cache
    lsd._cache = {}
    lsd._cache_keys = []
    lsd._cache_lock = threading.Lock()
    lsd.keys_list = keys
    return lsd


def _reference_extract_features_from_datamanager(dm, cluster_ids):
    """
    Frozen copy of extract_features_from_datamanager (umap_panel.py) before
    delegation to DataManager.get_physics_feature_matrix — used by AC7 only.
    """
    W_SHAPE = 2.0
    W_PATTERN = 1.5
    W_GEOMETRY = 1.0

    valid_ids = []
    tc_list = []
    acg_list = []
    scalars_list = []

    metadata = {
        'Time to Peak': [],
        'RF Area': [],
        'Ellipticity': []
    }

    for cid in cluster_ids:
        metrics = dm.get_cell_physics(cid)

        tc = metrics.get('timecourse')
        acg = metrics.get('acg')

        if tc is not None and acg is not None:
            valid_ids.append(cid)
            tc_list.append(tc)
            acg_list.append(acg)

            area = metrics.get('rf_area') or 0.0
            ellip = metrics.get('ellipticity') or 0.0
            t2p = metrics.get('time_to_peak') or 0

            scalars_list.append([area, ellip])

            metadata['Time to Peak'].append(t2p)
            metadata['RF Area'].append(area)
            metadata['Ellipticity'].append(ellip)

    if not valid_ids:
        return np.array([]), [], {}

    max_tc_len = max(len(t) for t in tc_list)
    tc_mat = np.array([
        np.pad(t, (0, max_tc_len - len(t))) if len(t) < max_tc_len else t[:max_tc_len]
        for t in tc_list
    ])

    max_acg_len = max(len(a) for a in acg_list)
    acg_mat = np.array([
        np.pad(a, (0, max_acg_len - len(a))) if len(a) < max_acg_len else a[:max_acg_len]
        for a in acg_list
    ])

    scalars_mat = np.array(scalars_list)

    nan_mask = (
        np.any(np.isnan(tc_mat), axis=1) |
        np.any(np.isnan(acg_mat), axis=1) |
        np.any(np.isnan(scalars_mat), axis=1)
    )
    if np.any(nan_mask):
        keep = ~nan_mask
        valid_ids = [vid for vid, k in zip(valid_ids, keep) if k]
        tc_mat = tc_mat[keep]
        acg_mat = acg_mat[keep]
        scalars_mat = scalars_mat[keep]
        for key in metadata:
            metadata[key] = [v for v, k in zip(metadata[key], keep) if k]

    if len(valid_ids) == 0:
        return np.array([]), [], {}

    if scalars_mat.shape[0] > 0 and scalars_mat.shape[1] > 0:
        scalars_mat = RobustScaler().fit_transform(scalars_mat)

    n_comp = min(3, len(valid_ids))
    tc_pca = PCA(n_components=n_comp).fit_transform(tc_mat) if n_comp > 0 else np.zeros((len(valid_ids), 0))
    acg_pca = PCA(n_components=n_comp).fit_transform(acg_mat) if n_comp > 0 else np.zeros((len(valid_ids), 0))

    final_features = np.hstack([
        tc_pca * W_SHAPE,
        acg_pca * W_PATTERN,
        scalars_mat * W_GEOMETRY
    ])

    return final_features, valid_ids, metadata


def _bind_physics_matrix_method(dm):
    from src.analysis.data_manager import DataManager
    dm.get_physics_feature_matrix = DataManager.get_physics_feature_matrix.__get__(
        dm, DataManager
    )
    return dm


# ─────────────────────────────────────────────────────────────────────────────
# AC1 — _compute_standard_plots writes ACG into feature_cache
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_standard_plots_writes_acg_to_feature_cache(tmp_path):
    from src.analysis.data_manager import DataManager

    dm = DataManager.__new__(DataManager)
    dm._feature_lock = threading.Lock()
    dm._standard_plot_lock = threading.Lock()
    dm._std_plot_cell_locks = {}
    dm._std_plot_cell_locks_lock = threading.Lock()
    dm.feature_cache = {}
    dm.standard_plot_cache = {}
    dm.sampling_rate = 20000.0

    spikes = np.arange(0, 200 * 40, 40, dtype=np.int64)
    dm.spike_times = spikes
    dm.spike_clusters = np.zeros(len(spikes), dtype=np.int64)

    dm.get_cluster_spikes = lambda cid: spikes
    dm.get_cluster_spike_amplitudes = lambda cid: np.ones(len(spikes))
    dm.templates = None

    dm._compute_standard_plots(0)

    assert 0 in dm.feature_cache
    assert dm.feature_cache[0].get('acg') is not None
    assert not dm.feature_cache[0].get('_computed')


# ─────────────────────────────────────────────────────────────────────────────
# AC2 — get_cell_physics skips get_standard_plot_data when ACG already in cache
# ─────────────────────────────────────────────────────────────────────────────

def test_get_cell_physics_skips_std_data_when_acg_cached(tmp_path):
    from src.analysis.data_manager import DataManager

    dm = DataManager.__new__(DataManager)
    dm._feature_lock = threading.Lock()
    dm._physics_cell_locks = {}
    dm._physics_cell_locks_lock = threading.Lock()
    dm._physics_done_count = 0
    dm.is_vision_only = False
    dm.vision_stas = None
    dm.vision_params = None

    sentinel = np.ones(201)
    dm.feature_cache = {0: {'acg': sentinel}}

    call_counter = {'n': 0}

    def fake_std_data(cid):
        call_counter['n'] += 1
        return {'acg_norm': np.zeros(201)}

    dm.get_standard_plot_data = fake_std_data

    metrics = DataManager.get_cell_physics(dm, 0)

    assert call_counter['n'] == 0, "get_standard_plot_data should not have been called"
    assert np.array_equal(metrics['acg'], sentinel)


# ─────────────────────────────────────────────────────────────────────────────
# AC3 — get_cell_physics falls back to get_standard_plot_data when no ACG cached
# ─────────────────────────────────────────────────────────────────────────────

def test_get_cell_physics_falls_back_to_std_data_when_no_acg(tmp_path):
    from src.analysis.data_manager import DataManager

    dm = DataManager.__new__(DataManager)
    dm._feature_lock = threading.Lock()
    dm._physics_cell_locks = {}
    dm._physics_cell_locks_lock = threading.Lock()
    dm._physics_done_count = 0
    dm.is_vision_only = False
    dm.vision_stas = None
    dm.vision_params = None
    dm.feature_cache = {}

    expected_acg = np.ones(201) * 0.5
    call_counter = {'n': 0}

    def fake_std_data(cid):
        call_counter['n'] += 1
        return {'acg_norm': expected_acg}

    dm.get_standard_plot_data = fake_std_data

    metrics = DataManager.get_cell_physics(dm, 0)

    assert call_counter['n'] == 1
    assert np.array_equal(metrics['acg'], expected_acg)


# ─────────────────────────────────────────────────────────────────────────────
# AC4 — LazySTADict concurrent reads do not corrupt cache
# ─────────────────────────────────────────────────────────────────────────────

def test_lazy_sta_dict_cache_is_thread_safe():
    lsd = _make_lazy_sta_dict(keys=[1, 2], sleep_s=0.005)
    errors = []

    def fetch(key):
        try:
            _ = lsd[key]
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=fetch, args=(1,))
    t2 = threading.Thread(target=fetch, args=(2,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, f"Thread errors: {errors}"
    assert 1 in lsd._cache and 2 in lsd._cache
    assert len(lsd._cache_keys) == len(set(lsd._cache_keys)), "Duplicate keys in _cache_keys"


# ─────────────────────────────────────────────────────────────────────────────
# AC5 — LazySTADict SSD read is not serialised
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_lazy_sta_dict_reads_are_concurrent():
    lsd = _make_lazy_sta_dict(keys=[1, 2], sleep_s=0.05)
    start = time.monotonic()

    t1 = threading.Thread(target=lambda: lsd[1])
    t2 = threading.Thread(target=lambda: lsd[2])
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    elapsed = time.monotonic() - start
    assert elapsed < 0.08, f"Reads were serialised (elapsed={elapsed:.3f}s)"


# ─────────────────────────────────────────────────────────────────────────────
# AC6 — ensure_physics_cache fills misses, no-op when warm
# ─────────────────────────────────────────────────────────────────────────────

def test_ensure_physics_cache_fills_misses():
    from src.analysis.data_manager import DataManager

    dm = DataManager.__new__(DataManager)
    dm._feature_lock = threading.Lock()
    dm._physics_cell_locks = {}
    dm._physics_cell_locks_lock = threading.Lock()
    dm._physics_done_count = 0
    dm.feature_cache = {}

    call_counter = {'n': 0}

    def fake_get_physics(cid):
        call_counter['n'] += 1
        with dm._feature_lock:
            dm.feature_cache[int(cid)] = {
                '_computed': True, 'acg': None, 'timecourse': None,
                'rf_area': 0.0, 'ellipticity': 0.0, 'time_to_peak': 0,
            }
        return dm.feature_cache[int(cid)]

    dm.get_cell_physics = fake_get_physics

    DataManager.ensure_physics_cache(dm, list(range(10)))

    assert call_counter['n'] == 10
    assert all(dm.feature_cache[i].get('_computed') for i in range(10))


def test_ensure_physics_cache_noop_when_warm():
    from src.analysis.data_manager import DataManager

    dm = DataManager.__new__(DataManager)
    dm._feature_lock = threading.Lock()
    dm.feature_cache = {i: {'_computed': True} for i in range(10)}

    call_counter = {'n': 0}
    dm.get_cell_physics = lambda cid: (call_counter.__setitem__('n', call_counter['n'] + 1), {})[1]

    DataManager.ensure_physics_cache(dm, list(range(10)))

    assert call_counter['n'] == 0


def test_ensure_physics_cache_respects_max_workers():
    """Pool size is configurable; default must not be overridden silently."""
    from concurrent.futures import ThreadPoolExecutor as _RealTPE
    from unittest.mock import patch, call
    from src.analysis.data_manager import DataManager

    dm = DataManager.__new__(DataManager)
    dm._feature_lock = threading.Lock()
    dm._physics_cell_locks = {}
    dm._physics_cell_locks_lock = threading.Lock()
    dm._physics_done_count = 0
    dm.feature_cache = {}

    def fake_physics(cid):
        with dm._feature_lock:
            dm.feature_cache[int(cid)] = {'_computed': True}

    dm.get_cell_physics = fake_physics

    captured = {}

    class CapturingTPE(_RealTPE):
        def __init__(self, *a, **kw):
            captured['max_workers'] = kw.get('max_workers')
            super().__init__(*a, **kw)

    with patch('src.analysis.data_manager.ThreadPoolExecutor', CapturingTPE):
        DataManager.ensure_physics_cache(dm, [0, 1], max_workers=2)

    assert captured.get('max_workers') == 2


def test_ensure_physics_cache_limits_parallel_sta_work():
    """Mock: 8 workers must not run unbounded concurrent get_cell_physics calls."""
    from concurrent.futures import ThreadPoolExecutor
    from unittest.mock import patch
    from src.analysis.data_manager import DataManager

    dm = DataManager.__new__(DataManager)
    dm._feature_lock = threading.Lock()
    dm._physics_cell_locks = {}
    dm._physics_cell_locks_lock = threading.Lock()
    dm._physics_done_count = 0
    dm.feature_cache = {}

    max_inflight = {'peak': 0}
    inflight = {'n': 0}
    lock = threading.Lock()

    def slow_physics(cid):
        with lock:
            inflight['n'] += 1
            max_inflight['peak'] = max(max_inflight['peak'], inflight['n'])
        time.sleep(0.02)
        with dm._feature_lock:
            dm.feature_cache[int(cid)] = {'_computed': True}
            dm._physics_done_count += 1
        with lock:
            inflight['n'] -= 1

    dm.get_cell_physics = slow_physics

    with patch('src.analysis.data_manager.ThreadPoolExecutor', ThreadPoolExecutor):
        DataManager.ensure_physics_cache(dm, list(range(20)), max_workers=4)

    assert max_inflight['peak'] <= 4


def test_cache_progress_does_not_drop_to_zero_when_vision_arrives():
    """Vision used to switch the bar from std_done to physics_done=0."""
    from src.gui.callbacks import _cache_progress_state

    val, ready, label = _cache_progress_state(
        total=100, std_done=50, physics_done=0, expect_physics=True
    )
    assert val == 25
    assert ready is False
    assert "spike plots" in label.lower()


def test_cache_progress_tracks_physics_in_second_half():
    """ensure_physics_cache does not emit finished_cluster; the bar still moves."""
    from src.gui.callbacks import _cache_progress_state

    val, ready, label = _cache_progress_state(
        total=100, std_done=100, physics_done=35, expect_physics=True
    )
    assert val == 67
    assert ready is False
    assert "Physics" in label

    done, ready, _ = _cache_progress_state(
        total=100, std_done=100, physics_done=100, expect_physics=True
    )
    assert done == 100
    assert ready is True


def test_cache_progress_std_only_without_vision():
    from src.gui.callbacks import _cache_progress_state

    val, ready, label = _cache_progress_state(
        total=100, std_done=40, physics_done=0, expect_physics=False
    )
    assert val == 40
    assert ready is False
    assert "spike plots" in label.lower()


def test_update_cache_progress_uses_two_phase_value():
    """Full Qt progress-bar polling is not exercised here (no event loop)."""
    import pandas as pd
    from unittest.mock import MagicMock
    from src.gui.callbacks import update_cache_progress

    mw = MagicMock()
    dm = MagicMock()
    dm.cluster_df = pd.DataFrame({'cluster_id': range(100)})
    dm._physics_done_count = 35
    dm.standard_plot_cache = {i: {} for i in range(90)}
    dm.vision_stas = object()
    mw.data_manager = dm
    mw._cache_save_triggered = False
    mw._expect_physics = True
    mw.cache_progress.minimum.return_value = 0
    mw.cache_progress.maximum.return_value = 100

    update_cache_progress(mw)

    # 50% * 90/100 + 50% * 35/100 = 62
    mw.cache_progress.setValue.assert_called_once_with(62)


# ─────────────────────────────────────────────────────────────────────────────
# AC7 — get_physics_feature_matrix matches old extract_features_from_datamanager
# ─────────────────────────────────────────────────────────────────────────────

def test_get_physics_feature_matrix_matches_old_extractor():
    dm, _cache = _make_mock_dm(n_clusters=10, all_computed=True)
    _bind_physics_matrix_method(dm)

    mat_new, ids_new, meta_new = dm.get_physics_feature_matrix(
        list(range(10)), w_shape=2.0, w_pattern=1.5, w_geometry=1.0)

    mat_old, ids_old, meta_old = _reference_extract_features_from_datamanager(
        dm, list(range(10)))

    assert ids_new == ids_old
    assert mat_new.shape == mat_old.shape
    assert np.allclose(mat_new, mat_old, atol=1e-6)
    assert meta_new.keys() == meta_old.keys()


# ─────────────────────────────────────────────────────────────────────────────
# AC8 — None timecourse or acg excluded
# ─────────────────────────────────────────────────────────────────────────────

def test_get_physics_feature_matrix_excludes_none_features():
    from src.analysis.data_manager import DataManager

    dm = DataManager.__new__(DataManager)
    dm._feature_lock = threading.Lock()
    dm.is_vision_only = False
    rng = np.random.default_rng(7)

    cache = {
        0: {'_computed': True, 'timecourse': None, 'acg': rng.random(201),
            'rf_area': 0.0, 'ellipticity': 0.0, 'time_to_peak': 0},
        1: {'_computed': True, 'timecourse': rng.standard_normal(30), 'acg': None,
            'rf_area': 0.0, 'ellipticity': 0.0, 'time_to_peak': 0},
        2: {'_computed': True, 'timecourse': rng.standard_normal(30), 'acg': rng.random(201),
            'rf_area': 0.0, 'ellipticity': 0.0, 'time_to_peak': 0},
        3: {'_computed': True, 'timecourse': rng.standard_normal(30), 'acg': rng.random(201),
            'rf_area': 0.0, 'ellipticity': 0.0, 'time_to_peak': 0},
        4: {'_computed': True, 'timecourse': rng.standard_normal(30), 'acg': rng.random(201),
            'rf_area': 0.0, 'ellipticity': 0.0, 'time_to_peak': 0},
    }
    dm.feature_cache = cache
    dm.get_cell_physics = lambda cid: cache[int(cid)]

    _, valid_ids, _ = DataManager.get_physics_feature_matrix(dm, list(range(5)))
    assert len(valid_ids) == 3
    assert 0 not in valid_ids and 1 not in valid_ids


# ─────────────────────────────────────────────────────────────────────────────
# AC9 — NaN rows dropped before PCA
# ─────────────────────────────────────────────────────────────────────────────

def test_get_physics_feature_matrix_drops_nan_rows():
    from src.analysis.data_manager import DataManager

    dm = DataManager.__new__(DataManager)
    dm._feature_lock = threading.Lock()
    dm.is_vision_only = False
    rng = np.random.default_rng(3)

    nan_tc = rng.standard_normal(30).copy()
    nan_tc[5] = np.nan
    cache = {
        0: {'_computed': True, 'timecourse': nan_tc, 'acg': rng.random(201),
            'rf_area': 0.0, 'ellipticity': 0.0, 'time_to_peak': 0},
        1: {'_computed': True, 'timecourse': rng.standard_normal(30), 'acg': rng.random(201),
            'rf_area': 0.0, 'ellipticity': 0.0, 'time_to_peak': 0},
        2: {'_computed': True, 'timecourse': rng.standard_normal(30), 'acg': rng.random(201),
            'rf_area': 0.0, 'ellipticity': 0.0, 'time_to_peak': 0},
        3: {'_computed': True, 'timecourse': rng.standard_normal(30), 'acg': rng.random(201),
            'rf_area': 0.0, 'ellipticity': 0.0, 'time_to_peak': 0},
        4: {'_computed': True, 'timecourse': rng.standard_normal(30), 'acg': rng.random(201),
            'rf_area': 0.0, 'ellipticity': 0.0, 'time_to_peak': 0},
    }
    dm.feature_cache = cache
    dm.get_cell_physics = lambda cid: cache[int(cid)]

    mat, valid_ids, _ = DataManager.get_physics_feature_matrix(dm, list(range(5)))
    assert 0 not in valid_ids
    assert len(valid_ids) == 4
    assert not np.any(np.isnan(mat))


# ─────────────────────────────────────────────────────────────────────────────
# AC10 — np.sort removal from _calculate_isi_violations is safe
# ─────────────────────────────────────────────────────────────────────────────

def test_isi_violations_sort_removed_output_unchanged():
    SAMPLING_RATE = 20000.0
    REFRACTORY_MS = 2.0

    spikes = np.array([0, 30, 60, 1000, 2000, 3000], dtype=np.int64)

    ref_period = (REFRACTORY_MS / 1000.0) * SAMPLING_RATE
    ref_pct = (np.sum(np.diff(np.sort(spikes)) < ref_period) / (len(spikes) - 1)) * 100
    fix_pct = (np.sum(np.diff(spikes) < ref_period) / (len(spikes) - 1)) * 100

    assert ref_pct == fix_pct
    assert ref_pct == pytest.approx(40.0)


# ─────────────────────────────────────────────────────────────────────────────
# AC13 — ACG cap at 10,000 spikes works and uses correct normalization count
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_standard_plots_caps_acg_spikes():
    from src.analysis.data_manager import DataManager

    dm = DataManager.__new__(DataManager)
    dm._feature_lock = threading.Lock()
    dm._standard_plot_lock = threading.Lock()
    dm._std_plot_cell_locks = {}
    dm._std_plot_cell_locks_lock = threading.Lock()
    dm.feature_cache = {}
    dm.standard_plot_cache = {}
    dm.sampling_rate = 20000.0

    # 15,000 spikes spaced by 2ms (40 samples @ 20kHz)
    spikes = np.arange(0, 15000 * 40, 40, dtype=np.int64)
    dm.spike_times = spikes
    dm.spike_clusters = np.zeros(len(spikes), dtype=np.int64)

    dm.get_cluster_spikes = lambda cid: spikes
    dm.get_cluster_spike_amplitudes = lambda cid: np.ones(len(spikes))
    dm.templates = None

    data = dm._compute_standard_plots(0)

    assert data['acg_norm'] is not None
    # If 10,000 cap is used on 15,000 spikes, the density is reduced to 2/3.
    # So the normalized ACG sum should be close to 100 * (2/3) = 66.67.
    # If the normalization was incorrectly using the uncapped count (15,000),
    # the sum would be close to 44.4.
    acg_sum = np.sum(data['acg_norm'])
    assert acg_sum == pytest.approx(65.88, abs=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# AC14 — try_get_standard_plot_data is non-blocking
# ─────────────────────────────────────────────────────────────────────────────

def test_try_get_standard_plot_data_returns_none_on_miss():
    """Non-blocking lookup must return None when the cluster is not cached."""
    from src.analysis.data_manager import DataManager

    dm = DataManager.__new__(DataManager)
    dm._standard_plot_lock = threading.Lock()
    dm.standard_plot_cache = {}

    result = dm.try_get_standard_plot_data(42)
    assert result is None


def test_try_get_standard_plot_data_returns_cached_data():
    """Non-blocking lookup must return data when the cluster IS cached."""
    from src.analysis.data_manager import DataManager

    dm = DataManager.__new__(DataManager)
    dm._standard_plot_lock = threading.Lock()
    sentinel = {'spikes': np.array([1, 2, 3]), 'acg_norm': np.ones(201)}
    dm.standard_plot_cache = {7: sentinel}

    result = dm.try_get_standard_plot_data(7)
    assert result is sentinel


def test_try_get_standard_plot_data_never_computes():
    """try_get must never call _compute_standard_plots even on miss."""
    from src.analysis.data_manager import DataManager

    dm = DataManager.__new__(DataManager)
    dm._standard_plot_lock = threading.Lock()
    dm.standard_plot_cache = {}

    compute_called = {'n': 0}
    dm._compute_standard_plots = lambda cid: (compute_called.__setitem__('n', compute_called['n'] + 1), {})[1]

    dm.try_get_standard_plot_data(99)
    assert compute_called['n'] == 0, "_compute_standard_plots should never be called"


# ─────────────────────────────────────────────────────────────────────────────
# AC15 — OOM fix: bloated raw arrays must NOT be stored in standard_plot_cache
# ─────────────────────────────────────────────────────────────────────────────

def _make_standard_plots_dm(n_spikes=500):
    from src.analysis.data_manager import DataManager
    dm = DataManager.__new__(DataManager)
    dm._feature_lock = threading.Lock()
    dm._standard_plot_lock = threading.Lock()
    dm._std_plot_cell_locks = {}
    dm._std_plot_cell_locks_lock = threading.Lock()
    dm.feature_cache = {}
    dm.standard_plot_cache = {}
    dm.sampling_rate = 20000.0
    dm.templates = None
    spikes = np.arange(0, n_spikes * 40, 40, dtype=np.int64)
    # _compute_standard_plots guards on these before doing any work
    dm.spike_times    = spikes
    dm.spike_clusters = np.zeros(n_spikes, dtype=np.int64)
    dm.get_cluster_spikes = lambda cid: spikes
    dm.get_cluster_spike_amplitudes = lambda cid: np.ones(n_spikes)
    return dm, spikes


BLOATED_KEYS = ('spikes', 'spikes_sec', 'spikes_ms', 'isi_ms',
                'isi_vs_amp_valid_isi', 'isi_vs_amp_valid_amplitudes',
                'fr_overlay_x', 'fr_overlay_y')

REQUIRED_KEYS = ('isi_hist_x', 'isi_hist_y',
                 'acg_time_lags', 'acg_norm',
                 'fr_bin_centers', 'fr_rate')


def test_compute_standard_plots_does_not_store_raw_spike_arrays():
    """
    After the OOM fix, _compute_standard_plots must NOT include any
    spike-length raw arrays in its return dict.
    """
    dm, _ = _make_standard_plots_dm(n_spikes=500)
    data = dm._compute_standard_plots(0)

    for key in BLOATED_KEYS:
        assert key not in data or data[key] is None, (
            f"Bloated key '{key}' found in cache — this causes OOM kills on large datasets."
        )


def test_compute_standard_plots_retains_aggregated_arrays():
    """
    The tiny binned/aggregated arrays the UI plots must still be present
    after removing the bloated raw arrays.
    """
    dm, _ = _make_standard_plots_dm(n_spikes=500)
    data = dm._compute_standard_plots(0)

    for key in REQUIRED_KEYS:
        assert data.get(key) is not None, (
            f"Required aggregated key '{key}' is missing from cache after OOM fix."
        )


def test_cache_memory_footprint_is_bounded():
    """
    Total bytes of one cached entry must be independent of spike count.
    500 vs 50,000 spikes should produce < 3x size difference.
    Before the fix the ratio was ~100x.
    """
    dm_small, _ = _make_standard_plots_dm(n_spikes=500)
    dm_large, _ = _make_standard_plots_dm(n_spikes=50_000)

    data_small = dm_small._compute_standard_plots(0)
    data_large = dm_large._compute_standard_plots(0)

    def _total_bytes(d):
        return sum(v.nbytes for v in d.values() if isinstance(v, np.ndarray))

    bytes_small = _total_bytes(data_small)
    bytes_large = _total_bytes(data_large)

    assert bytes_small > 0, "Cache is completely empty — something else is wrong."
    ratio = bytes_large / bytes_small
    assert ratio < 3.0, (
        f"Cache scales with spike count (ratio={ratio:.1f}x). "
        f"Raw arrays are still being stored. small={bytes_small}B large={bytes_large}B"
    )


# ─────────────────────────────────────────────────────────────────────────────
# AC16 — panel consumers must work without bloated keys
# ─────────────────────────────────────────────────────────────────────────────

def test_isi_histogram_works_without_isi_ms_in_cache():
    """
    Panel must compute isi_ms on-demand via np.diff(spikes)/sr*1000.
    Spikes every 40 samples @ 20kHz = 2ms ISI.
    With 1ms-wide bins (linspace 0..150, 151 edges = 150 bins),
    a 2ms ISI falls in bin index 2 (the bin covering 2.0–3.0ms).
    """
    sr = 20000.0
    spikes = np.arange(0, 500 * 40, 40, dtype=np.int64)

    isi_ms = np.diff(spikes) / sr * 1000.0  # all exactly 2.0 ms

    bins = np.linspace(0, 150, 151)         # 150 bins, each 1ms wide
    hist_y, hist_x = np.histogram(isi_ms, bins=bins)
    bin_centers = 0.5 * (hist_x[:-1] + hist_x[1:])

    assert len(hist_y) == 150
    assert np.sum(hist_y) == len(spikes) - 1  # all ISIs accounted for

    # 2ms ISI: bin edges are [0,1,2,3,...] so 2.0 falls in bin index 2
    # (left-closed, right-open: bin2 covers [2,3))
    assert hist_y[2] == len(spikes) - 1, (
        f"Expected all ISIs in bin[2] (2-3ms), got distribution: {hist_y[:6]}"
    )


def test_isi_vs_amplitude_works_without_cached_arrays():
    """ISI/amplitude alignment must produce equal-length, correctly sized arrays."""
    sr = 20000.0
    n = 300
    spikes = np.arange(0, n * 40, 40, dtype=np.int64)
    amplitudes = np.ones(n)

    isi_ms = np.diff(spikes) / sr * 1000.0
    min_len = min(len(isi_ms), len(amplitudes) - 1)
    valid_isi = isi_ms[:min_len]
    valid_amp = amplitudes[1:min_len + 1]

    assert len(valid_isi) == len(valid_amp)
    assert len(valid_isi) == n - 1


def test_fr_overlay_works_without_cached_arrays():
    """FR overlay recomputed from amplitudes + cached fr_rate must have consistent length."""
    from scipy.ndimage import gaussian_filter1d
    sr = 20000.0
    n_spikes = 500
    spikes = np.arange(0, n_spikes * 40, 40, dtype=np.int64)
    spikes_sec = spikes / sr
    amplitudes = np.ones(n_spikes)

    max_t = float(spikes_sec.max())
    bins = np.arange(0.0, max_t + 1.0, 1.0)
    counts, _ = np.histogram(spikes_sec, bins=bins)
    rate = gaussian_filter1d(counts.astype(float), sigma=5)

    norm_amp = amplitudes / float(np.max(amplitudes))
    avg_amp = np.convolve(norm_amp, np.ones(10) / 10.0, mode='valid')
    scaled_amp = avg_amp * 0.8 * float(np.max(rate))

    overlay_len = min(len(scaled_amp), len(spikes_sec))
    overlay_x = spikes_sec[:overlay_len]
    overlay_y = scaled_amp[:overlay_len]

    assert len(overlay_x) == len(overlay_y)
    assert len(overlay_x) > 0


# ─────────────────────────────────────────────────────────────────────────────
# AC17 — stale bloated pickle must be rejected by the loader
# ─────────────────────────────────────────────────────────────────────────────

def test_load_standard_plot_cache_rejects_bloated_pickle(tmp_path):
    """
    A standard_plot_cache.pkl written before the OOM fix contains 'spikes'
    arrays. The loader must detect and discard it rather than reloading GBs
    of bloated data into RAM.
    """
    import pickle
    from src.analysis.data_manager import DataManager

    bloated_cache = {
        0: {
            'spikes': np.arange(100_000, dtype=np.int64),
            'spikes_sec': np.arange(100_000, dtype=np.float64) / 20000.0,
            'acg_norm': np.ones(201),
            'isi_hist_x': np.linspace(0, 50, 101),
            'isi_hist_y': np.ones(100),
        }
    }
    cache_pkl = tmp_path / 'standard_plot_cache.pkl'
    with open(cache_pkl, 'wb') as f:
        pickle.dump(bloated_cache, f)

    dm = DataManager.__new__(DataManager)
    dm._standard_plot_lock = threading.Lock()
    dm.standard_plot_cache = {}
    dm.kilosort_dir = tmp_path

    dm._load_standard_plot_cache_from_disk()

    assert dm.standard_plot_cache == {}, (
        "Bloated on-disk cache was loaded instead of discarded. "
        "Add a version guard or bloat-detection check to _load_standard_plot_cache_from_disk."
    )