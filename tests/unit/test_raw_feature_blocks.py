"""
Unit tests for DataManager.get_raw_feature_blocks().

Uses a lightweight mock DataManager (no Qt event loop, no disk I/O)
to verify pre-filtering, array copying, uniform padding, and scalar
extraction.
"""

import numpy as np
import pandas as pd
import pytest
import threading


# ---------------------------------------------------------------------------
# Lightweight mock DataManager
# ---------------------------------------------------------------------------

class _MockDataManager:
    """
    Minimal stand-in for DataManager, providing just enough to exercise
    get_raw_feature_blocks() without Qt, disk, or real spike data.
    """

    def __init__(self, physics_map, cluster_df=None, spike_times=None,
                 spike_indices=None, sampling_rate=30000.0, is_vision_only=False,
                 chirp_data=None, chirp_id_to_row=None):
        """
        Parameters
        ----------
        physics_map : dict
            {cluster_id: physics_dict} — what get_cell_physics() returns.
        cluster_df : pd.DataFrame, optional
            Must have columns 'cluster_id' and 'isi_violations_pct'.
        spike_times : np.ndarray, optional
        spike_indices : dict, optional
            {cluster_id: np.ndarray of spike indices}
        """
        self.feature_cache = {}
        self._feature_lock = threading.Lock()
        self._physics_cell_locks = {}
        self._physics_cell_locks_lock = threading.Lock()
        self._physics_done_count = 0
        self.is_vision_only = is_vision_only
        self.sampling_rate = sampling_rate
        self.vision_stas = None
        self.vision_params = None

        self._physics_map = physics_map
        self.cluster_df = cluster_df if cluster_df is not None else pd.DataFrame()
        self.spike_times = spike_times
        self._spike_indices = spike_indices or {}

        # Standard plot cache (not used directly, but the method may check)
        self.standard_plot_cache = {}
        self._standard_plot_lock = threading.Lock()

        # Grating computed cache — the real DataManager populates this when
        # GratingComputeWorker finishes for each cluster.  Empty dict means
        # "no grating data computed yet" → the get_raw_feature_blocks loop
        # falls through to the zero-sentinel path, which is correct for a
        # mock that provides no grating stimulus data.
        self.grating_computed_cache = {}

        # Chirp — mirrors the real DataManager attrs set by load_chirp_data().
        # chirp_id_to_row already has the Vision→Kilosort offset baked in, so
        # get_raw_feature_blocks indexes it directly (never re-derives Law 1).
        self.chirp_data = chirp_data
        self.chirp_id_to_row = chirp_id_to_row
        self.chirp_available = (
            chirp_data is not None and chirp_id_to_row is not None
        )

    def get_cell_physics(self, cluster_id):
        """Return pre-configured physics dict for the given cluster."""
        return self._physics_map.get(int(cluster_id), {
            '_computed': True,
            'timecourse': None,
            'acg': None,
            'rf_area': 0.0,
            'ellipticity': 0.0,
            'time_to_peak': 0,
        })

    def get_cluster_spike_indices(self, cluster_id):
        """Return pre-configured spike indices for the given cluster."""
        return self._spike_indices.get(int(cluster_id), np.array([]))


# Bind the real method to the mock class so we test the actual implementation
from src.analysis.data_manager import DataManager
_MockDataManager.get_raw_feature_blocks = DataManager.get_raw_feature_blocks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SENTINEL = object()


def _make_physics(tc_len=20, acg_len=50, rf_area=50.0, ellipticity=1.0,
                  time_to_peak=10, tc_value=_SENTINEL, acg_value=None):
    """Helper to build a physics dict matching get_cell_physics() output."""
    if tc_value is _SENTINEL:
        tc = np.sin(np.linspace(0, np.pi, tc_len))
    else:
        tc = tc_value  # can be None or a custom array

    if acg_value is not None:
        acg = acg_value
    else:
        acg = np.random.RandomState(42).randn(acg_len)

    return {
        '_computed': True,
        'timecourse': tc,
        'acg': acg,
        'rf_area': rf_area,
        'ellipticity': ellipticity,
        'time_to_peak': time_to_peak,
    }


@pytest.fixture
def filter_config():
    return {'min_sta_std': 1e-5, 'max_rf_area': 300.0}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetRawFeatureBlocks:

    def test_valid_cells_returned(self, filter_config):
        """Good cells pass the filter and appear in raw_blocks."""
        physics = {
            0: _make_physics(rf_area=50.0),
            1: _make_physics(rf_area=100.0),
        }
        dm = _MockDataManager(physics)
        raw_blocks, valid_ids, discarded_ids = dm.get_raw_feature_blocks([0, 1], filter_config)

        assert valid_ids == [0, 1]
        assert discarded_ids == []
        assert raw_blocks['temporal'].shape[0] == 2
        assert raw_blocks['acg'].shape[0] == 2
        assert len(raw_blocks['scalars']) == 2

    def test_discards_flat_sta(self, filter_config):
        """Cell with flat timecourse is discarded."""
        physics = {
            0: _make_physics(),                                        # good
            1: _make_physics(tc_value=np.full(20, 0.5)),              # flat
        }
        dm = _MockDataManager(physics)
        raw_blocks, valid_ids, discarded_ids = dm.get_raw_feature_blocks([0, 1], filter_config)

        assert valid_ids == [0]
        assert discarded_ids == [1]
        assert raw_blocks['temporal'].shape[0] == 1

    def test_discards_large_rf(self, filter_config):
        """Cell with oversized RF is discarded."""
        physics = {
            0: _make_physics(rf_area=50.0),    # good
            1: _make_physics(rf_area=500.0),   # too large
        }
        dm = _MockDataManager(physics)
        raw_blocks, valid_ids, discarded_ids = dm.get_raw_feature_blocks([0, 1], filter_config)

        assert valid_ids == [0]
        assert discarded_ids == [1]

    def test_all_filtered_returns_empty(self, filter_config):
        """Oversized RF is discarded. Missing STA is not — that cell stays."""
        physics = {
            0: _make_physics(tc_value=None),
            1: _make_physics(rf_area=999.0),
        }
        dm = _MockDataManager(physics)
        raw_blocks, valid_ids, discarded_ids = dm.get_raw_feature_blocks([0, 1], filter_config)

        # apply_prefilter keeps timecourse=None (ACG + scalars still useful).
        # Only the oversized fitted RF is discarded.
        assert valid_ids == [0]
        assert discarded_ids == [1]
        assert raw_blocks['temporal'].shape[0] == 1

    def test_all_oversized_rf_returns_empty(self, filter_config):
        """Every cell over max_rf_area → empty blocks, no crash."""
        physics = {
            0: _make_physics(rf_area=999.0),
            1: _make_physics(rf_area=500.0),
        }
        dm = _MockDataManager(physics)
        raw_blocks, valid_ids, discarded_ids = dm.get_raw_feature_blocks([0, 1], filter_config)

        assert valid_ids == []
        assert discarded_ids == [0, 1]
        assert raw_blocks['temporal'].shape[0] == 0

    def test_copies_arrays_not_cache_refs(self, filter_config):
        """Returned arrays must be copies, not references to cached originals."""
        tc_original = np.sin(np.linspace(0, np.pi, 20))
        physics = {0: _make_physics(tc_value=tc_original.copy())}
        dm = _MockDataManager(physics)
        raw_blocks, _, _ = dm.get_raw_feature_blocks([0], filter_config)

        # Mutate the returned array
        raw_blocks['temporal'][0, 0] = 999.0

        # Original in physics_map must be unaffected
        assert physics[0]['timecourse'][0] != 999.0

    def test_uniform_padding(self, filter_config):
        """All TC rows and ACG rows have the same length after padding."""
        physics = {
            0: _make_physics(tc_len=15, acg_len=40),
            1: _make_physics(tc_len=20, acg_len=50),
            2: _make_physics(tc_len=18, acg_len=45),
        }
        dm = _MockDataManager(physics)
        raw_blocks, valid_ids, _ = dm.get_raw_feature_blocks([0, 1, 2], filter_config)

        tc = raw_blocks['temporal']
        acg = raw_blocks['acg']

        assert tc.shape[0] == 3
        assert tc.shape[1] == 20    # max tc_len
        assert acg.shape[0] == 3
        assert acg.shape[1] == 50   # max acg_len

    def test_scalars_include_firing_rate(self, filter_config):
        """Scalars DataFrame includes firing_rate column."""
        physics = {0: _make_physics()}
        spike_times = np.array([0, 30000, 60000, 90000])  # 4 spikes over 3 seconds
        spike_indices = {0: np.array([0, 1, 2, 3])}
        dm = _MockDataManager(
            physics,
            spike_times=spike_times,
            spike_indices=spike_indices,
        )
        raw_blocks, _, _ = dm.get_raw_feature_blocks([0], filter_config)

        scalars = raw_blocks['scalars']
        assert 'firing_rate' in scalars.columns
        # 4 spikes / 3.0 sec ≈ 1.33 Hz
        assert scalars['firing_rate'].iloc[0] == pytest.approx(4.0 / 3.0, rel=0.01)

    def test_scalars_include_isi_violations(self, filter_config):
        """Scalars DataFrame includes isi_violations from cluster_df."""
        physics = {0: _make_physics()}
        cluster_df = pd.DataFrame({
            'cluster_id': [0],
            'isi_violations_pct': [2.5],
        })
        dm = _MockDataManager(physics, cluster_df=cluster_df)
        raw_blocks, _, _ = dm.get_raw_feature_blocks([0], filter_config)

        scalars = raw_blocks['scalars']
        assert 'isi_violations' in scalars.columns
        assert scalars['isi_violations'].iloc[0] == pytest.approx(2.5)

    def test_scalars_all_columns_present(self, filter_config):
        """Scalars DataFrame keeps the display/hover columns plus RF diameters."""
        physics = {0: _make_physics()}
        dm = _MockDataManager(physics)
        raw_blocks, _, _ = dm.get_raw_feature_blocks([0], filter_config)

        expected_cols = {
            'rf_long_diameter', 'rf_short_diameter',
            'firing_rate', 'isi_violations', 'time_to_peak',
            'rf_area', 'ellipticity',
            'grating_dsi', 'grating_osi', 'grating_peak_rate_hz',
            'grating_pref_angle_cos', 'grating_pref_angle_sin',
        }
        assert set(raw_blocks['scalars'].columns) == expected_cols

    def test_row_count_matches_valid_ids(self, filter_config):
        """All blocks have exactly len(valid_ids) rows."""
        physics = {
            0: _make_physics(),
            1: _make_physics(tc_value=None),   # filtered out
            2: _make_physics(),
        }
        dm = _MockDataManager(physics)
        raw_blocks, valid_ids, _ = dm.get_raw_feature_blocks([0, 1, 2], filter_config)

        n = len(valid_ids)
        assert raw_blocks['temporal'].shape[0] == n
        assert raw_blocks['acg'].shape[0] == n
        assert len(raw_blocks['scalars']) == n


# ---------------------------------------------------------------------------
# Chirp block (see docs/specs/chirp_umap_feature_spec.md, Stage 2)
# ---------------------------------------------------------------------------

class TestGetRawFeatureBlocksChirp:

    def _chirp(self, n_cells, n_bins=40, qi=None):
        """Build a chirp_data dict + identity id→row map for n_cells."""
        rng = np.random.RandomState(3)
        psth = rng.rand(n_cells, n_bins) * 20.0
        if qi is None:
            qi = np.ones(n_cells)
        return {'psth_mean': psth, 'quality_index': np.asarray(qi, dtype=float)}, \
               {i: i for i in range(n_cells)}

    def test_chirp_block_present_and_l2_normalized(self, filter_config):
        physics = {0: _make_physics(), 1: _make_physics()}
        cdata, cmap = self._chirp(2, n_bins=40)
        dm = _MockDataManager(physics, chirp_data=cdata, chirp_id_to_row=cmap)
        raw_blocks, valid_ids, _ = dm.get_raw_feature_blocks([0, 1], filter_config)

        assert 'chirp' in raw_blocks
        assert raw_blocks['chirp'].shape == (2, 40)
        # Each present row is L2-normalized (shape, not amplitude).
        norms = np.linalg.norm(raw_blocks['chirp'], axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_chirp_missing_cell_is_zero_sentinel(self, filter_config):
        physics = {0: _make_physics(), 1: _make_physics()}
        # Only cell 0 present in the chirp file.
        cdata, _ = self._chirp(1, n_bins=40)
        dm = _MockDataManager(physics, chirp_data=cdata, chirp_id_to_row={0: 0})
        raw_blocks, valid_ids, _ = dm.get_raw_feature_blocks([0, 1], filter_config)

        row_for_1 = valid_ids.index(1)
        assert np.all(raw_blocks['chirp'][row_for_1] == 0.0)
        # Cell 0 still has a real normalized row.
        assert np.isclose(np.linalg.norm(raw_blocks['chirp'][valid_ids.index(0)]), 1.0)

    def test_chirp_nan_qi_is_zero_sentinel(self, filter_config):
        physics = {0: _make_physics(), 1: _make_physics()}
        cdata, cmap = self._chirp(2, n_bins=40, qi=[1.0, np.nan])
        dm = _MockDataManager(physics, chirp_data=cdata, chirp_id_to_row=cmap)
        raw_blocks, valid_ids, _ = dm.get_raw_feature_blocks([0, 1], filter_config)

        assert np.all(raw_blocks['chirp'][valid_ids.index(1)] == 0.0)
        assert np.isclose(np.linalg.norm(raw_blocks['chirp'][valid_ids.index(0)]), 1.0)

    def test_chirp_absent_is_empty_block(self, filter_config):
        """No chirp data loaded → 'chirp' present but width 0 (skipped downstream)."""
        physics = {0: _make_physics(), 1: _make_physics()}
        dm = _MockDataManager(physics)  # no chirp
        raw_blocks, valid_ids, _ = dm.get_raw_feature_blocks([0, 1], filter_config)

        assert 'chirp' in raw_blocks
        assert raw_blocks['chirp'].shape == (2, 0)

    def test_chirp_key_in_empty_blocks(self, filter_config):
        """When all cells are filtered out, the 'chirp' key still exists."""
        physics = {0: _make_physics(rf_area=999.0), 1: _make_physics(rf_area=999.0)}
        cdata, cmap = self._chirp(2)
        dm = _MockDataManager(physics, chirp_data=cdata, chirp_id_to_row=cmap)
        raw_blocks, valid_ids, _ = dm.get_raw_feature_blocks([0, 1], filter_config)

        assert valid_ids == []
        assert 'chirp' in raw_blocks
