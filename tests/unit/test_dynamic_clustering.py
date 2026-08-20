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
    non_sentinel_mask,
    observed_euclidean_distances,
    drop_empty_feature_rows,
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


# ---------------------------------------------------------------------------
# Mixed missing blocks: keep the cell, do not invent a "no STA" cluster
# ---------------------------------------------------------------------------

class TestMixedMissingBlocks:
    """Cells without an STA (or grating/chirp/RF) stay in the matrix.

    Their missing-block columns are NaN. PCA is fit only on rows that
    have the block, so they do not collapse onto one temporal point.
    """

    def _cfg(self, **overrides):
        cfg = {
            "use_temporal": True,
            "w_temporal": 10.0,
            "use_acg": True,
            "w_acg": 10.0,
            "use_rf_diameter": True,
            "w_rf_diameter": 10.0,
            "use_grating_dsos": False,
            "use_chirp": False,
        }
        cfg.update(overrides)
        return cfg

    def test_no_sta_rows_kept_with_nan_temporal(self):
        rng = np.random.RandomState(0)
        temporal = rng.randn(6, 20)
        temporal[[1, 4]] = 0.0  # no-STA sentinels
        raw = {
            "temporal": temporal,
            "acg": rng.randn(6, 50),
            "scalars": pd.DataFrame(
                {
                    "rf_long_diameter": [20.0, 0.0, 25.0, 18.0, 0.0, 22.0],
                    "rf_short_diameter": [10.0, 0.0, 12.0, 9.0, 0.0, 11.0],
                }
            ),
        }
        matrix, labels = build_feature_matrix(raw, self._cfg())
        assert matrix.shape[0] == 6
        tc_idx = [i for i, lab in enumerate(labels) if lab.startswith("tc_pc")]
        acg_idx = [i for i, lab in enumerate(labels) if lab.startswith("acg_pc")]
        assert tc_idx, "STA checked and some cells have STAs — keep the block"
        assert np.all(np.isnan(matrix[np.ix_([1, 4], tc_idx)]))
        assert np.all(np.isfinite(matrix[np.ix_([0, 2, 3, 5], tc_idx)]))
        # ACG is present for everyone, including the no-STA cells.
        assert np.all(np.isfinite(matrix[:, acg_idx]))

    def test_no_sta_cells_are_not_identical(self):
        """Two no-STA cells with different ACGs must not share a matrix row."""
        rng = np.random.RandomState(1)
        temporal = rng.randn(4, 20)
        temporal[[2, 3]] = 0.0
        acg = rng.randn(4, 50)
        acg[2] = np.linspace(0, 1, 50)
        acg[3] = np.linspace(1, 0, 50)
        raw = {
            "temporal": temporal,
            "acg": acg,
            "scalars": pd.DataFrame(
                {
                    "rf_long_diameter": [20.0, 22.0, 0.0, 0.0],
                    "rf_short_diameter": [10.0, 11.0, 0.0, 0.0],
                }
            ),
        }
        matrix, _ = build_feature_matrix(raw, self._cfg())
        assert not np.allclose(matrix[2], matrix[3], equal_nan=True)

    def test_all_valid_stays_finite(self):
        rng = np.random.RandomState(8)
        raw = {
            "temporal": rng.randn(8, 20),
            "acg": rng.randn(8, 50),
            "scalars": pd.DataFrame(
                {
                    "rf_long_diameter": rng.rand(8) * 20 + 8,
                    "rf_short_diameter": rng.rand(8) * 10 + 4,
                }
            ),
        }
        matrix, _ = build_feature_matrix(raw, self._cfg())
        assert np.isfinite(matrix).all()

    def test_all_missing_temporal_omits_block(self):
        rng = np.random.RandomState(2)
        raw = {
            "temporal": np.zeros((5, 1)),
            "acg": rng.randn(5, 50),
            "scalars": pd.DataFrame(
                {
                    "rf_long_diameter": rng.rand(5) * 20 + 5,
                    "rf_short_diameter": rng.rand(5) * 10 + 3,
                }
            ),
        }
        matrix, labels = build_feature_matrix(raw, self._cfg())
        assert not any(lab.startswith("tc_pc") for lab in labels)
        assert matrix.shape[0] == 5
        assert np.isfinite(matrix).all()

    def test_mixed_grating_is_nan_not_zero(self):
        rng = np.random.RandomState(3)
        grating = rng.randn(5, 12)
        grating[1] = 0.0
        grating[3] = 0.0
        raw = {
            "temporal": rng.randn(5, 20),
            "acg": rng.randn(5, 50),
            "grating": grating,
            "scalars": pd.DataFrame(
                {
                    "rf_long_diameter": np.linspace(10, 30, 5),
                    "rf_short_diameter": np.linspace(5, 15, 5),
                }
            ),
        }
        matrix, labels = build_feature_matrix(
            raw, self._cfg(use_grating_dsos=True, w_grating_dsos=10.0)
        )
        g_idx = [i for i, lab in enumerate(labels) if lab.startswith("grating_pc")]
        assert g_idx
        assert np.all(np.isnan(matrix[np.ix_([1, 3], g_idx)]))
        assert np.all(np.isfinite(matrix[np.ix_([0, 2, 4], g_idx)]))

    def test_zero_rf_diameter_is_nan(self):
        rng = np.random.RandomState(4)
        raw = {
            "temporal": rng.randn(4, 20),
            "acg": rng.randn(4, 50),
            "scalars": pd.DataFrame(
                {
                    "rf_long_diameter": [20.0, 0.0, 25.0, 18.0],
                    "rf_short_diameter": [10.0, 0.0, 12.0, 9.0],
                }
            ),
        }
        matrix, labels = build_feature_matrix(
            raw, self._cfg(use_temporal=False, use_acg=False)
        )
        assert labels == ["rf_long_diameter", "rf_short_diameter"]
        assert np.all(np.isnan(matrix[1]))
        assert np.all(np.isfinite(matrix[[0, 2, 3]]))


class TestObservedEuclidean:
    def test_matches_euclidean_when_complete(self):
        rng = np.random.RandomState(5)
        X = rng.randn(6, 4)
        got = observed_euclidean_distances(X)
        expected = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
        np.testing.assert_allclose(got, expected, atol=1e-10)

    def test_missing_block_does_not_collapse_or_push_apart(self):
        """No-STA cells (NaN temporal) are compared on ACG only.

        If those NaNs were filled with 0 after a weight-10 temporal block,
        the two incomplete cells would be distance 0 from each other and
        far from every cell that has an STA.
        """
        # cols 0-1 = temporal (weight already applied), cols 2-3 = ACG
        X = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [np.nan, np.nan, 0.0, 0.0],
                [np.nan, np.nan, 3.0, 4.0],
            ]
        )
        dist = observed_euclidean_distances(X)
        # Incomplete pair: only ACG (0,0) vs (3,4) → 5
        assert dist[2, 3] == pytest.approx(5.0)
        # Incomplete vs complete, shared ACG only: (0,0) vs (0,0) → 0
        assert dist[2, 0] == pytest.approx(0.0)
        # They are not glued to each other.
        assert dist[2, 3] > dist[2, 0]

    def test_no_shared_features_are_not_identical(self):
        """Complementary missingness is not distance 0.

        A cell with only STA and a cell with only RF share no coordinates.
        Treating that as identity turns them into hubs and collapses UMAP
        into a circle when the user runs one feature at a time.
        """
        X = np.array(
            [
                [1.0, np.nan],
                [np.nan, 1.0],
                [2.0, np.nan],
            ]
        )
        dist = observed_euclidean_distances(X)
        assert dist[0, 2] == pytest.approx(1.0)
        assert dist[0, 1] > 0.0
        assert dist[0, 1] >= dist[0, 2]
        np.testing.assert_array_equal(np.diag(dist), 0.0)

    def test_all_nan_row_is_not_zero_distance_to_everyone(self):
        """An empty row must not become a neighbour of every cell."""
        X = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [np.nan, np.nan],
            ]
        )
        dist = observed_euclidean_distances(X)
        assert dist[0, 1] == pytest.approx(np.sqrt(2.0))
        assert dist[2, 0] > 0.0
        assert dist[2, 1] > 0.0
        assert dist[2, 2] == 0.0

    def test_non_sentinel_mask(self):
        m = np.vstack([np.ones((2, 4)), np.zeros((2, 4))])
        np.testing.assert_array_equal(non_sentinel_mask(m), [True, True, False, False])


class TestDropEmptyFeatureRows:
    def test_keeps_complete_matrix(self):
        matrix = np.arange(8, dtype=float).reshape(4, 2)
        raw = {
            "temporal": np.ones((4, 3)),
            "scalars": pd.DataFrame({"rf_long_diameter": [1.0, 2.0, 3.0, 4.0]}),
        }
        out, ids, discarded, blocks = drop_empty_feature_rows(
            matrix, [10, 11, 12, 13], [], raw
        )
        np.testing.assert_array_equal(out, matrix)
        assert ids == [10, 11, 12, 13]
        assert discarded == []
        assert blocks["temporal"].shape == (4, 3)

    def test_moves_all_nan_rows_to_discarded(self):
        matrix = np.array(
            [
                [1.0, 2.0],
                [np.nan, np.nan],
                [3.0, 4.0],
                [np.nan, np.nan],
            ]
        )
        raw = {
            "temporal": np.arange(8, dtype=float).reshape(4, 2),
            "scalars": pd.DataFrame({"rf_long_diameter": [10.0, 0.0, 20.0, 0.0]}),
        }
        out, ids, discarded, blocks = drop_empty_feature_rows(
            matrix, [0, 1, 2, 3], [99], raw
        )
        np.testing.assert_array_equal(out, [[1.0, 2.0], [3.0, 4.0]])
        assert ids == [0, 2]
        assert discarded == [99, 1, 3]
        np.testing.assert_array_equal(blocks["temporal"], [[0.0, 1.0], [4.0, 5.0]])
        assert list(blocks["scalars"]["rf_long_diameter"]) == [10.0, 20.0]


class TestDefaultFeatureFlags:
    def test_defaults_are_temporal_acg_rf_at_ten(self):
        from src.analysis.constants import (
            DEFAULT_USE_TEMPORAL,
            DEFAULT_USE_ACG,
            DEFAULT_USE_RF_DIAMETER,
            DEFAULT_USE_GRATING_DSOS,
            DEFAULT_USE_CHIRP,
            DEFAULT_WEIGHT_TEMPORAL,
            DEFAULT_WEIGHT_ACG,
            DEFAULT_WEIGHT_RF_DIAMETER,
        )

        assert DEFAULT_USE_TEMPORAL is True
        assert DEFAULT_USE_ACG is True
        assert DEFAULT_USE_RF_DIAMETER is True
        assert DEFAULT_USE_GRATING_DSOS is False
        assert DEFAULT_USE_CHIRP is False
        assert DEFAULT_WEIGHT_TEMPORAL == 10.0
        assert DEFAULT_WEIGHT_ACG == 10.0
        assert DEFAULT_WEIGHT_RF_DIAMETER == 10.0
