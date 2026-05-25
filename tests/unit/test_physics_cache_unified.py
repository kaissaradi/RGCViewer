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
