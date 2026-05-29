"""
tests/unit/test_data_manager_spatial_pca.py

Phase 2 & 4 unit tests covering:
  - get_physics_feature_matrix() with spatial PCA (Vision present)
  - get_physics_feature_matrix() in Kilosort-only mode (vision_stas=None)
  - load_persisted_caches() discarding a legacy feature_cache.pkl

All tests use mock_dm (no disk I/O) or tmp_path (isolated pkl round-trip).
Cache bypass is guaranteed: mock_dm.feature_cache starts empty, tmp_path has no real data.

Run with:
    conda run -n rgcviewer python -m pytest tests/unit/test_data_manager_spatial_pca.py -v
"""

import pickle
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Shared synthetic data helpers
# ---------------------------------------------------------------------------

_N_FRAMES   = 20
_HEIGHT     = 30
_WIDTH      = 30
_PEAK_FRAME = 10
_N_CLUSTERS = 8
_ACG_LEN    = 100  # length of a synthetic ACG vector


def _make_acg(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.random(_ACG_LEN).astype(np.float32)
    a /= a.max() + 1e-9
    return a


def _make_timecourse(seed=0, length=20):
    rng = np.random.default_rng(seed)
    tc = rng.standard_normal(length).astype(np.float32)
    tc /= np.abs(tc).max() + 1e-9
    return tc


def _make_footprint(seed=0, grid_size=32):
    rng = np.random.default_rng(seed)
    fp = rng.standard_normal(grid_size * grid_size).astype(np.float32)
    fp = (fp - fp.mean()) / (fp.std() + 1e-9)
    return fp


def _make_blob_sta(cx=15, cy=15, sigma=3.0, amplitude=10.0, noise_std=0.1, seed=0):
    """Return a minimal STA-like object with .red/.green/.blue arrays."""
    rng = np.random.default_rng(seed)

    class _STA:
        pass

    sta = _STA()
    base = rng.standard_normal((_HEIGHT, _WIDTH, _N_FRAMES)).astype(np.float32) * noise_std
    yy, xx = np.mgrid[0:_HEIGHT, 0:_WIDTH]
    blob = amplitude * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)
    base[:, :, _PEAK_FRAME] += blob

    sta.red   = base.copy()
    sta.green = rng.standard_normal((_HEIGHT, _WIDTH, _N_FRAMES)).astype(np.float32) * noise_std
    sta.blue  = rng.standard_normal((_HEIGHT, _WIDTH, _N_FRAMES)).astype(np.float32) * noise_std
    return sta


def _build_cluster_df(cluster_ids):
    """Minimal cluster_df matching the guaranteed columns from build_cluster_dataframe()."""
    n = len(cluster_ids)
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "cluster_id":        cluster_ids,
        "n_spikes":          rng.integers(100, 5000, size=n),
        "isi_violations_pct": rng.uniform(0, 5, size=n),
        "KSLabel":           ["good"] * n,
        "status":            ["Original"] * n,
        "cell_type":         ["Unknown"] * n,
        "max_dup_r":         [0.0] * n,
        "potential_dups":    [False] * n,
        "set":               [{cid} for cid in cluster_ids],
    })


def _populate_feature_cache_vision(dm, cluster_ids, grid_size=32):
    """
    Fill dm.feature_cache with fully-computed entries that include
    timecourse, acg, and spatial_footprint (Vision-present scenario).
    """
    for i, cid in enumerate(cluster_ids):
        dm.feature_cache[int(cid)] = {
            "_computed":        True,
            "acg":              _make_acg(seed=i),
            "timecourse":       _make_timecourse(seed=i),
            "spatial_footprint": _make_footprint(seed=i, grid_size=grid_size),
            "rf_area":          float(np.pi * 3.0 * 3.0),
            "ellipticity":      1.0,
            "time_to_peak":     _PEAK_FRAME,
        }


def _populate_feature_cache_ks_only(dm, cluster_ids):
    """
    Fill dm.feature_cache with entries that have acg but no timecourse
    and no spatial_footprint (Kilosort-only scenario).
    """
    for i, cid in enumerate(cluster_ids):
        dm.feature_cache[int(cid)] = {
            "_computed":        True,
            "acg":              _make_acg(seed=i),
            "timecourse":       None,
            "spatial_footprint": None,
            "rf_area":          0.0,
            "ellipticity":      0.0,
            "time_to_peak":     0,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_dm_vision(mock_dm):
    """mock_dm pre-loaded with Vision-present feature cache entries."""
    cids = list(range(_N_CLUSTERS))
    mock_dm.cluster_df = _build_cluster_df(cids)
    mock_dm.vision_stas = {cid + 1: _make_blob_sta(seed=cid) for cid in cids}  # 1-indexed
    mock_dm.is_vision_only = False
    _populate_feature_cache_vision(mock_dm, cids)
    return mock_dm, cids


@pytest.fixture
def mock_dm_ks_only(mock_dm):
    """mock_dm pre-loaded with Kilosort-only feature cache entries (no Vision)."""
    cids = list(range(_N_CLUSTERS))
    mock_dm.cluster_df = _build_cluster_df(cids)
    mock_dm.vision_stas = None
    mock_dm.is_vision_only = False
    _populate_feature_cache_ks_only(mock_dm, cids)
    return mock_dm, cids


# ---------------------------------------------------------------------------
# test_feature_matrix_spatial_pca
# Spec: AC1 / test plan row "test_feature_matrix_includes_spatial_pca"
# ---------------------------------------------------------------------------

class TestFeatureMatrixWithSpatialPCA:
    """
    When Vision STA data is present, get_physics_feature_matrix() must
    produce a matrix with 11 columns: 3 TC-PCA + 3 ACG-PCA + 3 spatial-PCA + 2 KS scalars.

    NOTE: This test verifies the *contract* (11 columns). It will FAIL until
    the implementation is complete — that is expected.
    """

    def test_matrix_shape_n_rows_equals_n_valid_cells(self, mock_dm_vision):
        dm, cids = mock_dm_vision
        matrix, valid_ids, metadata = dm.get_physics_feature_matrix(cids)
        assert matrix.shape[0] == len(cids), (
            f"Expected {len(cids)} rows, got {matrix.shape[0]}"
        )

    def test_matrix_has_9_columns_when_all_blocks_present(self, mock_dm_vision):
        dm, cids = mock_dm_vision
        matrix, valid_ids, metadata = dm.get_physics_feature_matrix(cids)
        assert matrix.shape[1] == 11, (
            f"Expected 11 columns (3 TC + 3 ACG + 3 spatial + 2 KS scalars), got {matrix.shape[1]}"
        )

    def test_valid_ids_match_cluster_ids(self, mock_dm_vision):
        dm, cids = mock_dm_vision
        _, valid_ids, _ = dm.get_physics_feature_matrix(cids)
        assert set(valid_ids) == set(cids)

    def test_no_nans_in_matrix(self, mock_dm_vision):
        dm, cids = mock_dm_vision
        matrix, valid_ids, _ = dm.get_physics_feature_matrix(cids)
        assert not np.any(np.isnan(matrix)), "Feature matrix must contain no NaNs"

    def test_metadata_keys_present(self, mock_dm_vision):
        dm, cids = mock_dm_vision
        _, _, metadata = dm.get_physics_feature_matrix(cids)
        for key in ("Time to Peak", "RF Area", "Ellipticity"):
            assert key in metadata, f"Missing metadata key: {key}"

    def test_weights_scale_matrix(self, mock_dm_vision):
        """Doubling w_shape must not leave the matrix identical."""
        dm, cids = mock_dm_vision
        m1, _, _ = dm.get_physics_feature_matrix(cids, w_shape=1.0, w_pattern=1.0, w_geometry=1.0)
        m2, _, _ = dm.get_physics_feature_matrix(cids, w_shape=2.0, w_pattern=1.0, w_geometry=1.0)
        assert not np.allclose(m1, m2), "w_shape scaling had no effect on matrix"


# ---------------------------------------------------------------------------
# test_feature_matrix_kilosort_only
# Spec: AC3 / test plan row "test_feature_matrix_kilosort_only_mode"
# ---------------------------------------------------------------------------

class TestFeatureMatrixKilosortOnly:
    """
    When vision_stas is None, get_physics_feature_matrix() must succeed
    using only ACG-PCA (3 cols) + Kilosort scalars (2 cols) = 5 columns.

    NOTE: Currently the function returns an empty matrix for this case because
    of the `if tc is None or acg is None: continue` gate. These tests FAIL
    until the gate is relaxed.
    """

    def test_returns_nonempty_matrix_without_vision(self, mock_dm_ks_only):
        dm, cids = mock_dm_ks_only
        matrix, valid_ids, metadata = dm.get_physics_feature_matrix(cids)
        assert len(valid_ids) > 0, (
            "Expected valid cells in Kilosort-only mode, got empty valid_ids. "
            "The tc/acg gate must be relaxed to allow ACG-only rows."
        )
        assert matrix.shape[0] == len(valid_ids)

    def test_matrix_has_5_columns_in_ks_only_mode(self, mock_dm_ks_only):
        dm, cids = mock_dm_ks_only
        matrix, valid_ids, _ = dm.get_physics_feature_matrix(cids)
        assert len(valid_ids) > 0, "Precondition: must have valid cells"
        assert matrix.shape[1] == 5, (
            f"Expected 5 columns (3 ACG-PCA + 2 KS scalars) in Kilosort-only mode, "
            f"got {matrix.shape[1]}"
        )

    def test_no_nans_in_ks_only_matrix(self, mock_dm_ks_only):
        dm, cids = mock_dm_ks_only
        matrix, valid_ids, _ = dm.get_physics_feature_matrix(cids)
        if len(valid_ids) > 0:
            assert not np.any(np.isnan(matrix)), "Kilosort-only matrix must contain no NaNs"

    def test_all_cells_included_in_ks_only_mode(self, mock_dm_ks_only):
        """
        Every cell has an ACG — none should be dropped when tc is None.
        """
        dm, cids = mock_dm_ks_only
        _, valid_ids, _ = dm.get_physics_feature_matrix(cids)
        assert set(valid_ids) == set(cids), (
            f"Expected all {len(cids)} cells; got {len(valid_ids)}. "
            f"Missing: {set(cids) - set(valid_ids)}"
        )

    def test_empty_result_if_acg_also_missing(self, mock_dm):
        """
        ACG is the last required feature. If it's also None, the cell is excluded.
        """
        cids = [0, 1, 2]
        mock_dm.cluster_df = _build_cluster_df(cids)
        mock_dm.vision_stas = None
        for cid in cids:
            mock_dm.feature_cache[cid] = {
                "_computed": True,
                "acg": None,          # no ACG either
                "timecourse": None,
                "spatial_footprint": None,
                "rf_area": 0.0,
                "ellipticity": 0.0,
                "time_to_peak": 0,
            }
        matrix, valid_ids, _ = mock_dm.get_physics_feature_matrix(cids)
        assert len(valid_ids) == 0, "Cells with no ACG must still be excluded"
        assert matrix.shape[0] == 0


# ---------------------------------------------------------------------------
# test_cache_migration_prunes_legacy
# Spec: AC4 / test plan row "test_cache_migration_prunes_legacy"
# ---------------------------------------------------------------------------

class TestCacheMigrationPrunesLegacy:
    """
    load_persisted_caches() must detect a pre-spatial-footprint cache
    (entries that have _computed=True but no spatial_footprint key) and
    discard it entirely so physics is recomputed on next ensure_physics_cache().
    """

    def _write_legacy_cache(self, path: Path):
        """Write a feature_cache.pkl that lacks the spatial_footprint key."""
        legacy = {
            0: {"_computed": True, "acg": _make_acg(), "timecourse": _make_timecourse(),
                "rf_area": 10.0, "ellipticity": 1.1, "time_to_peak": 5},
            1: {"_computed": True, "acg": _make_acg(1), "timecourse": _make_timecourse(1),
                "rf_area": 8.0, "ellipticity": 0.9, "time_to_peak": 7},
        }
        with open(path / "feature_cache.pkl", "wb") as f:
            pickle.dump(legacy, f)

    def _write_current_cache(self, path: Path):
        """Write a feature_cache.pkl that already has the spatial_footprint key."""
        current = {
            0: {"_computed": True, "acg": _make_acg(), "timecourse": _make_timecourse(),
                "spatial_footprint": _make_footprint(),
                "rf_area": 10.0, "ellipticity": 1.1, "time_to_peak": 5},
        }
        with open(path / "feature_cache.pkl", "wb") as f:
            pickle.dump(current, f)

    def test_legacy_cache_is_discarded(self, tmp_path, mock_dm):
        """A pre-footprint pkl must be discarded; feature_cache ends up empty."""
        self._write_legacy_cache(tmp_path)
        mock_dm.kilosort_dir = tmp_path

        mock_dm.load_persisted_caches()

        assert len(mock_dm.feature_cache) == 0, (
            f"Expected empty feature_cache after legacy migration, "
            f"got {len(mock_dm.feature_cache)} entries"
        )

    def test_legacy_cache_resets_physics_done_count(self, tmp_path, mock_dm):
        self._write_legacy_cache(tmp_path)
        mock_dm.kilosort_dir = tmp_path

        mock_dm.load_persisted_caches()

        assert mock_dm._physics_done_count == 0, (
            f"Expected _physics_done_count=0 after legacy migration, "
            f"got {mock_dm._physics_done_count}"
        )

    def test_current_cache_is_kept(self, tmp_path, mock_dm):
        """A cache that already has spatial_footprint keys must NOT be discarded."""
        self._write_current_cache(tmp_path)
        mock_dm.kilosort_dir = tmp_path

        mock_dm.load_persisted_caches()

        assert len(mock_dm.feature_cache) == 1, (
            f"Expected current cache to be preserved (1 entry), "
            f"got {len(mock_dm.feature_cache)}"
        )

    def test_empty_cache_file_does_not_crash(self, tmp_path, mock_dm):
        """An empty (but valid) feature_cache.pkl must load without error."""
        with open(tmp_path / "feature_cache.pkl", "wb") as f:
            pickle.dump({}, f)
        mock_dm.kilosort_dir = tmp_path

        try:
            mock_dm.load_persisted_caches()
        except Exception as exc:
            pytest.fail(f"load_persisted_caches() raised on empty cache: {exc}")

        assert mock_dm.feature_cache == {}

    def test_corrupt_cache_file_does_not_crash(self, tmp_path, mock_dm):
        """A corrupt pkl must be silently discarded, not propagate an exception."""
        (tmp_path / "feature_cache.pkl").write_bytes(b"not a pickle at all")
        mock_dm.kilosort_dir = tmp_path

        try:
            mock_dm.load_persisted_caches()
        except Exception as exc:
            pytest.fail(f"load_persisted_caches() raised on corrupt cache: {exc}")

        assert mock_dm.feature_cache == {}

    def test_missing_cache_file_is_fine(self, tmp_path, mock_dm):
        """No pkl on disk must leave feature_cache empty and not raise."""
        mock_dm.kilosort_dir = tmp_path
        assert not (tmp_path / "feature_cache.pkl").exists()

        try:
            mock_dm.load_persisted_caches()
        except Exception as exc:
            pytest.fail(f"load_persisted_caches() raised when pkl absent: {exc}")