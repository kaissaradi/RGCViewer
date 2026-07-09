"""
Unit tests for the dynamic clustering pure-math functions in analysis_core.

These test peak_align_timecourse, apply_prefilter, and build_feature_matrix
without any Qt, DataManager, or disk I/O dependencies.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.analysis_core import (
    peak_align_timecourse,
    apply_prefilter,
    build_feature_matrix,
)


# ---------------------------------------------------------------------------
# peak_align_timecourse
# ---------------------------------------------------------------------------

class TestPeakAlignTimecourse:
    """Tests for peak_align_timecourse."""

    def test_centers_peak(self):
        """Peak should land at len//2 after alignment."""
        tc = np.zeros(20)
        tc[3] = 5.0          # peak far from centre (index 10)
        aligned = peak_align_timecourse(tc)
        assert np.argmax(np.abs(aligned)) == 10

    def test_two_shifted_traces_align(self):
        """Two identical shapes shifted by 4 frames both align to centre."""
        base = np.sin(np.linspace(0, np.pi, 20))
        tc_a = np.roll(base, 2)
        tc_b = np.roll(base, 6)

        aligned_a = peak_align_timecourse(tc_a)
        aligned_b = peak_align_timecourse(tc_b)

        centre = len(base) // 2
        assert np.argmax(np.abs(aligned_a)) == centre
        assert np.argmax(np.abs(aligned_b)) == centre

    def test_skips_flatline(self):
        """Flat trace (std < threshold) is returned unchanged — no roll."""
        tc = np.full(20, 0.5)     # constant — std == 0
        aligned = peak_align_timecourse(tc)
        np.testing.assert_array_equal(aligned, tc)

    def test_returns_copy(self):
        """Original array must never be mutated."""
        tc = np.zeros(20)
        tc[3] = 5.0
        original = tc.copy()
        _ = peak_align_timecourse(tc)
        np.testing.assert_array_equal(tc, original)

    def test_negative_peak(self):
        """Negative peak (OFF cell) is also centred via abs()."""
        tc = np.zeros(20)
        tc[2] = -8.0
        aligned = peak_align_timecourse(tc)
        assert np.argmax(np.abs(aligned)) == 10

    def test_empty_array(self):
        """Empty input returns empty output without error."""
        tc = np.array([])
        aligned = peak_align_timecourse(tc)
        assert len(aligned) == 0

    def test_already_centred(self):
        """If peak is already at centre, array values are preserved."""
        tc = np.zeros(20)
        tc[10] = 3.0
        aligned = peak_align_timecourse(tc)
        np.testing.assert_array_almost_equal(aligned, tc)


# ---------------------------------------------------------------------------
# apply_prefilter
# ---------------------------------------------------------------------------

class TestApplyPrefilter:
    """Tests for apply_prefilter."""

    @pytest.fixture
    def filter_config(self):
        return {'min_sta_std': 1e-5, 'max_rf_area': 300.0}

    def test_passes_good_cell(self, filter_config):
        """A normal biological cell should pass the filter."""
        physics = {
            0: {'timecourse': np.sin(np.linspace(0, np.pi, 20)), 'rf_area': 50.0},
        }
        valid, discarded = apply_prefilter(physics, filter_config)
        assert valid == [0]
        assert discarded == []

    def test_passes_none_timecourse(self, filter_config):
        """Cell with timecourse=None is accepted (KS-only support)."""
        physics = {
            0: {'timecourse': None, 'rf_area': 50.0},
        }
        valid, discarded = apply_prefilter(physics, filter_config)
        assert valid == [0]
        assert discarded == []

    def test_rejects_flat_sta(self, filter_config):
        """Cell with flat STA (std < threshold) is discarded."""
        physics = {
            0: {'timecourse': np.full(20, 1.0), 'rf_area': 50.0},
        }
        valid, discarded = apply_prefilter(physics, filter_config)
        assert valid == []
        assert discarded == [0]

    def test_rejects_large_rf(self, filter_config):
        """Cell with rf_area > threshold is discarded."""
        physics = {
            0: {'timecourse': np.sin(np.linspace(0, np.pi, 20)), 'rf_area': 500.0},
        }
        valid, discarded = apply_prefilter(physics, filter_config)
        assert valid == []
        assert discarded == [0]

    def test_mixed_cells(self, filter_config):
        """Correctly partitions a mix of good, flat, oversized, and None-tc cells."""
        physics = {
            1: {'timecourse': np.sin(np.linspace(0, np.pi, 20)), 'rf_area': 50.0},   # good
            2: {'timecourse': np.sin(np.linspace(0, np.pi, 20)), 'rf_area': 500.0},   # big RF
            3: {'timecourse': np.full(20, 0.0), 'rf_area': 50.0},                     # flat
            4: {'timecourse': None, 'rf_area': 50.0},                                  # None tc → accepted (KS-only)
            5: {'timecourse': np.sin(np.linspace(0, np.pi, 20)), 'rf_area': 100.0},   # good
        }
        valid, discarded = apply_prefilter(physics, filter_config)
        assert valid == [1, 4, 5]
        assert sorted(discarded) == [2, 3]

    def test_sorted_output(self, filter_config):
        """Output lists are sorted regardless of input dict order."""
        physics = {
            9: {'timecourse': np.sin(np.linspace(0, np.pi, 20)), 'rf_area': 50.0},
            3: {'timecourse': np.sin(np.linspace(0, np.pi, 20)), 'rf_area': 50.0},
            7: {'timecourse': None, 'rf_area': 50.0},  # None tc → accepted
        }
        valid, discarded = apply_prefilter(physics, filter_config)
        assert valid == [3, 7, 9]  # all pass, sorted
        assert valid == sorted(valid)
        assert discarded == sorted(discarded)


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    """Tests for build_feature_matrix."""

    @pytest.fixture
    def raw_blocks_10_cells(self):
        """10 cells with 20-sample timecourses, 50-sample ACGs, and 5 scalars."""
        rng = np.random.RandomState(42)
        return {
            'temporal': rng.randn(10, 20),
            'acg': rng.randn(10, 50),
            'scalars': pd.DataFrame({
                'firing_rate': rng.rand(10) * 20,
                'isi_violations': rng.rand(10) * 0.1,
                'time_to_peak': rng.randint(0, 20, size=10).astype(float),
                'rf_area': rng.rand(10) * 100,
                'ellipticity': rng.rand(10) * 2,
            }),
        }

    @pytest.fixture
    def all_enabled_config(self):
        return {
            'use_temporal': True, 'w_temporal': 3.0,
            'use_acg': True, 'w_acg': 2.0,
            'use_firing_rate': True, 'w_firing_rate': 1.5,
            'use_isi_violations': True, 'w_isi_violations': 1.0,
            'use_time_to_peak': True, 'w_time_to_peak': 1.0,
            'use_rf_area': True, 'w_rf_area': 1.0,
            'use_ellipticity': True, 'w_ellipticity': 1.0,
        }

    def test_output_shape(self, raw_blocks_10_cells, all_enabled_config):
        """Matrix has N rows and labels match column count."""
        matrix, labels = build_feature_matrix(raw_blocks_10_cells, all_enabled_config)
        assert matrix.shape[0] == 10
        assert matrix.shape[1] == len(labels)

    def test_zero_weight_omits_acg(self, raw_blocks_10_cells, all_enabled_config):
        """Disabling ACG reduces matrix width by ACG_PCA_COMPONENTS."""
        matrix_full, _ = build_feature_matrix(raw_blocks_10_cells, all_enabled_config)

        config_no_acg = {**all_enabled_config, 'use_acg': False}
        matrix_no_acg, _ = build_feature_matrix(raw_blocks_10_cells, config_no_acg)

        from src.analysis.constants import ACG_PCA_COMPONENTS
        expected_diff = min(ACG_PCA_COMPONENTS, 10 - 1, 50)
        assert matrix_full.shape[1] - matrix_no_acg.shape[1] == expected_diff

    def test_zero_weight_omits_temporal(self, raw_blocks_10_cells, all_enabled_config):
        """Disabling temporal reduces matrix width by TEMPORAL_PCA_COMPONENTS."""
        matrix_full, _ = build_feature_matrix(raw_blocks_10_cells, all_enabled_config)

        config_no_tc = {**all_enabled_config, 'use_temporal': False}
        matrix_no_tc, _ = build_feature_matrix(raw_blocks_10_cells, config_no_tc)

        from src.analysis.constants import TEMPORAL_PCA_COMPONENTS
        expected_diff = min(TEMPORAL_PCA_COMPONENTS, 10 - 1, 20)
        assert matrix_full.shape[1] - matrix_no_tc.shape[1] == expected_diff

    def test_all_disabled_raises(self, raw_blocks_10_cells):
        """All features disabled raises ValueError."""
        config = {
            'use_temporal': False, 'use_acg': False,
            'use_firing_rate': False, 'use_isi_violations': False,
            'use_time_to_peak': False, 'use_rf_area': False,
            'use_ellipticity': False,
        }
        with pytest.raises(ValueError, match="All features are disabled"):
            build_feature_matrix(raw_blocks_10_cells, config)

    def test_row_alignment(self, raw_blocks_10_cells, all_enabled_config):
        """Matrix has exactly N rows matching the input block row count."""
        matrix, _ = build_feature_matrix(raw_blocks_10_cells, all_enabled_config)
        assert matrix.shape[0] == raw_blocks_10_cells['temporal'].shape[0]

    def test_pca_clamps_components(self, all_enabled_config):
        """With only 3 samples, PCA clamps to min(configured, N-1)."""
        rng = np.random.RandomState(42)
        raw_blocks = {
            'temporal': rng.randn(3, 20),
            'acg': rng.randn(3, 50),
            'scalars': pd.DataFrame({
                'firing_rate': [1.0, 2.0, 3.0],
                'isi_violations': [0.01, 0.02, 0.03],
                'time_to_peak': [5.0, 10.0, 15.0],
                'rf_area': [50.0, 60.0, 70.0],
                'ellipticity': [1.0, 1.5, 0.8],
            }),
        }
        matrix, labels = build_feature_matrix(raw_blocks, all_enabled_config)
        assert matrix.shape[0] == 3
        # With 3 samples: temporal PCA → min(5,2,20)=2, ACG PCA → min(3,2,50)=2
        # + 5 scalars = 9 columns
        tc_cols = [l for l in labels if l.startswith('tc_pc')]
        acg_cols = [l for l in labels if l.startswith('acg_pc')]
        assert len(tc_cols) == 2    # clamped from 5 to 2
        assert len(acg_cols) == 2   # clamped from 3 to 2

    def test_no_nan_in_output(self, raw_blocks_10_cells, all_enabled_config):
        """Output matrix should contain no NaN values."""
        matrix, _ = build_feature_matrix(raw_blocks_10_cells, all_enabled_config)
        assert not np.any(np.isnan(matrix))

    def test_missing_scalar_imputed(self, all_enabled_config):
        """Missing scalar values (NaN) are imputed as 0.0."""
        rng = np.random.RandomState(42)
        raw_blocks = {
            'temporal': rng.randn(5, 20),
            'acg': rng.randn(5, 50),
            'scalars': pd.DataFrame({
                'firing_rate': [1.0, np.nan, 3.0, np.nan, 5.0],
                'isi_violations': [0.01, 0.02, np.nan, 0.04, 0.05],
                'time_to_peak': [5.0, 10.0, 15.0, 20.0, 25.0],
                'rf_area': [50.0, 60.0, 70.0, 80.0, 90.0],
                'ellipticity': [1.0, 1.5, 0.8, 1.2, 1.1],
            }),
        }
        matrix, _ = build_feature_matrix(raw_blocks, all_enabled_config)
        assert not np.any(np.isnan(matrix))

    def test_scalars_only(self):
        """Works with only scalar features enabled (no PCA blocks).

        build_feature_matrix's scalar_features table was consolidated to
        rf_long_diameter / rf_short_diameter only (see analysis_core.py
        and constants.py DEFAULT_WEIGHT_* comments).  Old scalars like
        firing_rate, isi_violations, etc. are still present in
        raw_blocks['scalars'] for metadata/hover but are NOT used as
        embedding features.
        """
        rng = np.random.RandomState(42)
        raw_blocks = {
            'temporal': rng.randn(5, 20),
            'acg': rng.randn(5, 50),
            'grating': np.zeros((5, 12)),  # zero sentinel → skipped
            'scalars': pd.DataFrame({
                'rf_long_diameter': rng.rand(5) * 50,
                'rf_short_diameter': rng.rand(5) * 30,
                'firing_rate': rng.rand(5) * 20,
                'isi_violations': rng.rand(5) * 0.1,
                'time_to_peak': rng.randint(0, 20, size=5).astype(float),
                'rf_area': rng.rand(5) * 100,
                'ellipticity': rng.rand(5) * 2,
            }),
        }
        config = {
            'use_temporal': False, 'use_acg': False,
            'use_grating_dsos': False,
            'use_rf_diameter': True, 'w_rf_diameter': 6.0,
        }
        matrix, labels = build_feature_matrix(raw_blocks, config)
        assert matrix.shape == (5, 2)
        assert labels == ['rf_long_diameter', 'rf_short_diameter']


# ---------------------------------------------------------------------------
# build_feature_matrix — chirp PSTH shape block
# (see docs/specs/chirp_umap_feature_spec.md)
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrixChirp:
    """Chirp PSTH-shape PCA block behaves like the grating block:
    PCA'd when present, skipped (no NaN) when all-zero/absent/disabled."""

    @pytest.fixture
    def raw_blocks_chirp(self):
        """10 cells; chirp = 40-bin PSTH shape rows with real variance."""
        rng = np.random.RandomState(7)
        return {
            'temporal': rng.randn(10, 20),
            'acg': rng.randn(10, 50),
            'grating': np.zeros((10, 12)),  # zero sentinel → skipped
            'chirp': rng.randn(10, 40),
            'scalars': pd.DataFrame({
                'rf_long_diameter': rng.rand(10) * 50,
                'rf_short_diameter': rng.rand(10) * 30,
            }),
        }

    @pytest.fixture
    def chirp_only_config(self):
        # Only the chirp block active, so width == chirp PC count exactly.
        return {
            'use_temporal': False, 'use_acg': False,
            'use_grating_dsos': False, 'use_rf_diameter': False,
            'use_chirp': True, 'w_chirp': 3.0,
        }

    def test_chirp_block_present_adds_pcs(self, raw_blocks_chirp, chirp_only_config):
        from src.analysis.constants import CHIRP_PCA_COMPONENTS
        matrix, labels = build_feature_matrix(raw_blocks_chirp, chirp_only_config)
        n_comp = min(CHIRP_PCA_COMPONENTS, 10 - 1, 40)
        chirp_cols = [l for l in labels if l.startswith('chirp_pc')]
        assert len(chirp_cols) == n_comp
        assert matrix.shape == (10, n_comp)
        assert not np.any(np.isnan(matrix))

    def test_chirp_disabled_omits_block(self, raw_blocks_chirp, chirp_only_config):
        full, _ = build_feature_matrix(
            raw_blocks_chirp, {**chirp_only_config, 'use_rf_diameter': True,
                               'w_rf_diameter': 6.0})
        no_chirp, labels = build_feature_matrix(
            raw_blocks_chirp, {**chirp_only_config, 'use_chirp': False,
                               'use_rf_diameter': True, 'w_rf_diameter': 6.0})
        from src.analysis.constants import CHIRP_PCA_COMPONENTS
        diff = min(CHIRP_PCA_COMPONENTS, 10 - 1, 40)
        assert full.shape[1] - no_chirp.shape[1] == diff
        assert not any(l.startswith('chirp_pc') for l in labels)

    def test_chirp_all_zero_skipped(self, raw_blocks_chirp, chirp_only_config):
        """All-zero chirp matrix (every cell sentinel) is skipped, no NaN."""
        raw_blocks_chirp['chirp'] = np.zeros((10, 40))
        # Need at least one other enabled block so the matrix isn't empty.
        cfg = {**chirp_only_config, 'use_rf_diameter': True, 'w_rf_diameter': 6.0}
        matrix, labels = build_feature_matrix(raw_blocks_chirp, cfg)
        assert not any(l.startswith('chirp_pc') for l in labels)
        assert not np.any(np.isnan(matrix))

    def test_chirp_absent_key_no_error(self, raw_blocks_chirp, chirp_only_config):
        """A raw_blocks dict with no 'chirp' key (older caller) is tolerated."""
        del raw_blocks_chirp['chirp']
        cfg = {**chirp_only_config, 'use_rf_diameter': True, 'w_rf_diameter': 6.0}
        matrix, labels = build_feature_matrix(raw_blocks_chirp, cfg)
        assert not any(l.startswith('chirp_pc') for l in labels)
        assert matrix.shape[0] == 10

    def test_chirp_pca_clamps_components(self, chirp_only_config):
        """With only 3 cells, chirp PCA clamps to N-1."""
        rng = np.random.RandomState(1)
        raw_blocks = {
            'temporal': rng.randn(3, 20),
            'acg': rng.randn(3, 50),
            'chirp': rng.randn(3, 40),
            'scalars': pd.DataFrame({
                'rf_long_diameter': [10.0, 20.0, 30.0],
                'rf_short_diameter': [5.0, 10.0, 15.0],
            }),
        }
        matrix, labels = build_feature_matrix(raw_blocks, chirp_only_config)
        chirp_cols = [l for l in labels if l.startswith('chirp_pc')]
        assert len(chirp_cols) == 2  # min(4, 3-1, 40)
