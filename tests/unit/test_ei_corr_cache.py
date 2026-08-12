"""Tests for the EI-correlation cache on the dataset load path.

The correlation matrices are expensive enough to be cached in
``ei_corr_dict.pkl``, but reading that file used to happen *after* every EI in
the dataset had already been walked — so a warm run paid the full cost anyway.
These tests pin the two properties that fixes it:

1. A warm cache touches no EI array at all.
2. The row order of the matrices comes from the cache, not from re-walking the
   EIs, so a cache that no longer describes this dataset is rejected instead of
   silently mismatched.
"""

import pickle
from collections import namedtuple

import numpy as np
import pandas as pd
import pytest
from qtpy.QtCore import QObject

from src.analysis.data_manager import DataManager

N_ELECTRODES = 8
N_SAMPLES = 20

# Only the ``ei`` field is read by _sanitize_ei_dict and ei_corr.
_EI = namedtuple("_EI", ["ei"])


def _make_eis(n_cells, seed=0):
    """Vision-ID-keyed EI entries, 1-indexed as the real loader produces."""
    rng = np.random.default_rng(seed)
    return {
        vid: _EI(ei=rng.normal(size=(N_ELECTRODES, N_SAMPLES)).astype(np.float32))
        for vid in range(1, n_cells + 1)
    }


class _ExplodingEIs(dict):
    """An EI mapping that fails if anything reads a cell's arrays.

    Stands in for the lazy reader that replaces the eager dict: ``keys()`` and
    ``len()`` are free there because they come from the seek table, but every
    other access is a disk read. Anything that trips this has proven the warm
    path is still paying for EIs it does not use.
    """

    _MSG = "warm path must not read EI arrays"

    def items(self):
        raise AssertionError(self._MSG)

    def values(self):
        raise AssertionError(self._MSG)

    def get(self, *_args, **_kwargs):
        raise AssertionError(self._MSG)

    def __getitem__(self, _key):
        raise AssertionError(self._MSG)

    def __iter__(self):
        raise AssertionError(self._MSG)


@pytest.fixture
def corr_dm(tmp_path):
    """A DataManager wired up for just the EI-correlation path.

    Built with __new__ + QObject.__init__ so the signal works without the real
    __init__ touching disk, Qt widgets or DataJoint.
    """
    dm = DataManager.__new__(DataManager)
    QObject.__init__(dm)
    dm.kilosort_dir = tmp_path
    dm.is_vision_only = False
    dm.ei_corr_dict = None
    dm.vision_eis = _make_eis(4)
    dm.cluster_df = pd.DataFrame({"cluster_id": [0, 1, 2, 3]})
    dm.emitted = []
    dm.ei_updates_ready.connect(lambda a, b: dm.emitted.append((a, b)))
    return dm


def _pkl(dm):
    return dm.kilosort_dir / "ei_corr_dict.pkl"


def _read_pkl(dm):
    with open(_pkl(dm), "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Cold path
# ---------------------------------------------------------------------------


def test_cold_run_computes_and_stores_row_ids(corr_dm):
    corr_dm._compute_ei_correlations_if_needed()

    saved = _read_pkl(corr_dm)
    assert set(saved) == {"full", "space", "power", "ids"}
    # Vision IDs, in the order ei_corr built the matrix rows.
    assert saved["ids"] == [1, 2, 3, 4]
    assert saved["full"].shape[0] == 4


def test_cold_run_emits_duplicate_maps(corr_dm):
    corr_dm._compute_ei_correlations_if_needed()
    assert len(corr_dm.emitted) == 1


def test_fewer_than_two_eis_emits_empty_and_writes_nothing(corr_dm):
    corr_dm.vision_eis = _make_eis(1)

    corr_dm._compute_ei_correlations_if_needed()

    assert corr_dm.emitted == [({}, {})]
    assert not _pkl(corr_dm).exists()


# ---------------------------------------------------------------------------
# Warm path — the point of the change
# ---------------------------------------------------------------------------


def test_warm_run_never_touches_the_ei_arrays(corr_dm):
    corr_dm._compute_ei_correlations_if_needed()
    expected = corr_dm.ei_corr_dict["full"].copy()

    # Second load of the same run: any read of an EI array now raises.
    corr_dm.ei_corr_dict = None
    corr_dm.vision_eis = _ExplodingEIs(corr_dm.vision_eis)
    corr_dm.emitted.clear()

    corr_dm._compute_ei_correlations_if_needed()

    np.testing.assert_array_equal(corr_dm.ei_corr_dict["full"], expected)
    assert len(corr_dm.emitted) == 1


def test_warm_run_gives_the_same_duplicate_flags_as_a_cold_one(corr_dm):
    corr_dm._compute_ei_correlations_if_needed()
    cold = corr_dm.emitted[0]

    corr_dm.ei_corr_dict = None
    corr_dm.emitted.clear()
    corr_dm._compute_ei_correlations_if_needed()

    assert corr_dm.emitted[0] == cold


def test_already_loaded_correlations_short_circuit(corr_dm):
    corr_dm.ei_corr_dict = {"full": np.zeros((2, 2))}
    corr_dm.vision_eis = _ExplodingEIs(corr_dm.vision_eis)

    corr_dm._compute_ei_correlations_if_needed()

    assert corr_dm.emitted == []


# ---------------------------------------------------------------------------
# Legacy files — written before row ids were stored
# ---------------------------------------------------------------------------


def test_legacy_cache_is_adopted_and_stamped_with_ids(corr_dm):
    corr_dm._compute_ei_correlations_if_needed()
    legacy = {k: v for k, v in _read_pkl(corr_dm).items() if k != "ids"}
    expected = legacy["full"].copy()
    corr_dm._save_pickle_with_fallback(legacy, str(_pkl(corr_dm)))

    corr_dm.ei_corr_dict = None
    corr_dm._compute_ei_correlations_if_needed()

    # Matrices reused, not recomputed, and the file is upgraded in place.
    np.testing.assert_array_equal(corr_dm.ei_corr_dict["full"], expected)
    assert _read_pkl(corr_dm)["ids"] == [1, 2, 3, 4]


def test_upgraded_legacy_cache_is_warm_on_the_next_load(corr_dm):
    corr_dm._compute_ei_correlations_if_needed()
    legacy = {k: v for k, v in _read_pkl(corr_dm).items() if k != "ids"}
    corr_dm._save_pickle_with_fallback(legacy, str(_pkl(corr_dm)))

    corr_dm.ei_corr_dict = None
    corr_dm._compute_ei_correlations_if_needed()  # upgrades

    corr_dm.ei_corr_dict = None
    corr_dm.vision_eis = _ExplodingEIs(corr_dm.vision_eis)
    corr_dm._compute_ei_correlations_if_needed()  # must now be free

    assert corr_dm.ei_corr_dict["ids"] == [1, 2, 3, 4]


def test_legacy_cache_for_a_different_cell_count_is_rejected(corr_dm):
    """The silent-corruption case: matrices that describe another .ei file.

    Row i would be read as a different cell than it was computed for, marking
    the wrong units as duplicates with no error anywhere.
    """
    corr_dm._compute_ei_correlations_if_needed()
    legacy = {k: v for k, v in _read_pkl(corr_dm).items() if k != "ids"}
    stale_full = legacy["full"].copy()
    corr_dm._save_pickle_with_fallback(legacy, str(_pkl(corr_dm)))

    # The .ei file now holds six cells, not four.
    corr_dm.vision_eis = _make_eis(6, seed=1)
    corr_dm.cluster_df = pd.DataFrame({"cluster_id": [0, 1, 2, 3, 4, 5]})
    corr_dm.ei_corr_dict = None

    corr_dm._compute_ei_correlations_if_needed()

    assert corr_dm.ei_corr_dict["full"].shape[0] == 6
    assert corr_dm.ei_corr_dict["ids"] == [1, 2, 3, 4, 5, 6]
    assert corr_dm.ei_corr_dict["full"].shape != stale_full.shape


# ---------------------------------------------------------------------------
# Unusable files fall back to a recompute rather than raising
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b"not a pickle at all", id="corrupt_bytes"),
        pytest.param(pickle.dumps(["not", "a", "dict"]), id="wrong_type"),
        pytest.param(pickle.dumps({"full": np.zeros((4, 4))}), id="missing_matrices"),
        pytest.param(
            pickle.dumps(
                {
                    "full": np.zeros((4, 4)),
                    "space": np.zeros((3, 3)),
                    "power": np.zeros((4, 4)),
                }
            ),
            id="mismatched_shapes",
        ),
        pytest.param(
            pickle.dumps(
                {
                    "full": np.zeros((4, 4)),
                    "space": np.zeros((4, 4)),
                    "power": np.zeros((4, 4)),
                    "ids": [1, 2],
                }
            ),
            id="ids_do_not_match_rows",
        ),
        pytest.param(
            pickle.dumps({"full": None, "space": None, "power": None}),
            id="matrices_are_none",
        ),
    ],
)
def test_unusable_cache_falls_back_to_a_recompute(corr_dm, payload):
    _pkl(corr_dm).write_bytes(payload)

    corr_dm._compute_ei_correlations_if_needed()

    assert corr_dm.ei_corr_dict["ids"] == [1, 2, 3, 4]
    assert corr_dm.ei_corr_dict["full"].shape[0] == 4
    assert _read_pkl(corr_dm)["ids"] == [1, 2, 3, 4]


def test_missing_file_reports_nothing_cached(corr_dm):
    assert corr_dm._load_cached_ei_corr(str(_pkl(corr_dm))) == (None, None)


def test_vision_only_datasets_skip_the_whole_pass(corr_dm):
    corr_dm.is_vision_only = True
    corr_dm.vision_eis = _ExplodingEIs(corr_dm.vision_eis)

    corr_dm._compute_ei_correlations_if_needed()

    assert corr_dm.emitted == [({}, {})]
    assert not _pkl(corr_dm).exists()


def test_no_vision_eis_emits_empty(corr_dm):
    corr_dm.vision_eis = None

    corr_dm._compute_ei_correlations_if_needed()

    assert corr_dm.emitted == [({}, {})]
