"""Tests for LazyEIDict, the read-on-demand .ei mapping.

The .ei file is the largest in a preparation (~740 MB for 900 cells on the 512
array) and used to be read in full at every load. LazyEIDict replaces that with
a seek table plus per-cell reads.

These tests write a real .ei file with the repo's own EIFileWriter, so the
comparison against the eager reader is byte-exact rather than approximate.
"""

import threading

import numpy as np
import pytest

from src.analysis import vision_integration as vi
from src.analysis.vision_integration import LazyEIDict, load_ei_data

pytest.importorskip("src.analysis.visionloader")

import src.analysis.visionloader as vl  # noqa: E402

DATASET = "data000"
N_ELECTRODES = 8          # excluding the TTL row
N_LEFT, N_RIGHT = 3, 4
N_SAMPLES = N_LEFT + N_RIGHT + 1
CELL_IDS = list(range(1, 7))
ARRAY_ID = 9999           # reconfigurable: coordinates live in the .globals


def _write_dataset(folder, cell_ids=CELL_IDS, seed=0):
    """Write a real .ei + .globals pair and return the raw arrays written."""
    rng = np.random.default_rng(seed)

    # The .globals map carries the TTL row; the reader strips it back off.
    coords = np.column_stack(
        [
            np.arange(N_ELECTRODES + 1, dtype=np.float64),
            np.arange(N_ELECTRODES + 1, dtype=np.float64) * 2.0,
        ]
    )
    vl.GlobalsFileWriter(str(folder), DATASET).write(ARRAY_ID, coords)

    written = {}
    payload = {}
    for cid in cell_ids:
        # Rows INCLUDE the TTL row at index 0, per EIFileWriter's contract.
        ei = rng.normal(size=(N_ELECTRODES + 1, N_SAMPLES)).astype(np.float32)
        err = rng.normal(size=(N_ELECTRODES + 1, N_SAMPLES)).astype(np.float32)
        payload[cid] = (ei, err, 100 + cid)
        written[cid] = (ei, err)

    vl.EIFileWriter(str(folder), DATASET, N_LEFT, N_RIGHT, ARRAY_ID).write(payload)
    return written


@pytest.fixture
def ei_dir(tmp_path):
    _write_dataset(tmp_path)
    return tmp_path


@pytest.fixture
def lazy(ei_dir):
    d = LazyEIDict(ei_dir, DATASET)
    yield d
    d.close()


# ---------------------------------------------------------------------------
# Equivalence with the eager reader — the property that matters most
# ---------------------------------------------------------------------------


def test_every_cell_matches_the_eager_reader_exactly(ei_dir, lazy):
    with vl.EIReader(str(ei_dir), DATASET) as eager_reader:
        eager = eager_reader.get_all_eis_by_cell_id()

    assert sorted(lazy.keys()) == sorted(eager)
    for cid, expected in eager.items():
        got = lazy[cid]
        np.testing.assert_array_equal(got.ei, expected.ei)
        np.testing.assert_array_equal(got.ei_error, expected.ei_error)
        assert got.n_spikes == expected.n_spikes
        assert got.nl_points == expected.nl_points
        assert got.nr_points == expected.nr_points


def test_ttl_row_is_dropped_like_the_eager_reader(lazy):
    assert lazy[CELL_IDS[0]].ei.shape == (N_ELECTRODES, N_SAMPLES)


def test_electrode_map_excludes_the_ttl_row(lazy):
    assert lazy.get_electrode_map().shape == (N_ELECTRODES, 2)


# ---------------------------------------------------------------------------
# Mapping protocol — every shape the existing consumers use
# ---------------------------------------------------------------------------


def test_len_and_keys(lazy):
    assert len(lazy) == len(CELL_IDS)
    assert sorted(lazy.keys()) == CELL_IDS


def test_truthiness(lazy):
    """ei_panel guards with a bare `data_manager.vision_eis and ...`."""
    assert bool(lazy) is True


def test_contains(lazy):
    assert CELL_IDS[0] in lazy
    assert 10_000 not in lazy
    assert "not an id" not in lazy


def test_get_returns_default_for_an_unknown_cell(lazy):
    assert lazy.get(10_000) is None
    assert lazy.get(10_000, "fallback") == "fallback"


def test_get_accepts_the_int_cast_callers_use(lazy):
    assert lazy.get(int(CELL_IDS[0])) is not None


def test_iteration_yields_the_cell_ids(lazy):
    assert sorted(iter(lazy)) == CELL_IDS


def test_items_streams_rather_than_materialising(lazy):
    """Cell Tracer walks .items(); rebuilding the eager dict there would undo
    the whole point of this class."""
    items = lazy.items()
    assert not isinstance(items, (dict, list))

    seen = dict(items)
    assert sorted(seen) == CELL_IDS
    assert all(v.ei.shape == (N_ELECTRODES, N_SAMPLES) for v in seen.values())


def test_values_streams_too(lazy):
    assert len(list(lazy.values())) == len(CELL_IDS)


def test_unknown_cell_id_reads_as_none_rather_than_raising(lazy):
    assert lazy[10_000] is None


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------


def test_repeat_reads_come_from_the_cache(lazy):
    first = lazy[CELL_IDS[0]]
    assert lazy[CELL_IDS[0]] is first


def test_cache_is_bounded(ei_dir, monkeypatch):
    """The budget is what keeps a full scan from rebuilding the eager dict."""
    monkeypatch.setattr(vi, "MAX_EI_CACHE_CELLS", 2)
    d = LazyEIDict(ei_dir, DATASET)
    try:
        for cid in CELL_IDS:
            assert d[cid] is not None
        assert len(d._cache) <= 2
    finally:
        d.close()


def test_a_full_scan_does_not_retain_every_cell(ei_dir, monkeypatch):
    monkeypatch.setattr(vi, "MAX_EI_CACHE_CELLS", 2)
    d = LazyEIDict(ei_dir, DATASET)
    try:
        list(d.items())
        assert len(d._cache) <= 2
    finally:
        d.close()


# ---------------------------------------------------------------------------
# Threads — the physics warm-up and the panels read from several at once
# ---------------------------------------------------------------------------


def test_concurrent_reads_from_several_threads_agree(ei_dir, lazy):
    with vl.EIReader(str(ei_dir), DATASET) as eager_reader:
        expected = eager_reader.get_all_eis_by_cell_id()

    errors = []

    def worker():
        try:
            for _ in range(5):
                for cid in CELL_IDS:
                    got = lazy[cid]
                    np.testing.assert_array_equal(got.ei, expected[cid].ei)
        except Exception as exc:  # pragma: no cover - only on a real failure
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []


def test_each_thread_gets_its_own_reader(ei_dir, lazy):
    lazy[CELL_IDS[0]]  # creator thread uses the shared reader

    def worker():
        lazy[CELL_IDS[1]]

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert len(lazy._all_readers) == 2


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


def test_close_releases_every_reader(ei_dir):
    d = LazyEIDict(ei_dir, DATASET)
    d[CELL_IDS[0]]

    def worker():
        d[CELL_IDS[1]]

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    readers = list(d._all_readers)
    assert len(readers) == 2

    d.close()

    assert d._all_readers == []
    assert all(r.ei_fp.closed for r in readers)


def test_close_is_repeatable(lazy):
    lazy.close()
    lazy.close()


def test_close_drops_the_cached_containers(lazy):
    lazy[CELL_IDS[0]]
    assert lazy._cache

    lazy.close()

    assert lazy._cache == {}


# ---------------------------------------------------------------------------
# load_ei_data wiring
# ---------------------------------------------------------------------------


def test_load_ei_data_returns_a_lazy_mapping(ei_dir):
    bundle = load_ei_data(ei_dir, DATASET)
    try:
        assert isinstance(bundle["ei_data"], LazyEIDict)
        assert bundle["electrode_map"].shape == (N_ELECTRODES, 2)
    finally:
        bundle["ei_data"].close()


def test_load_ei_data_does_not_read_the_cells(ei_dir, monkeypatch):
    """Opening a dataset must cost the seek table, not the whole file."""
    def _boom(self, cell_id):
        raise AssertionError("opening a dataset must not read any EI")

    monkeypatch.setattr(vl.EIReader, "get_ei_for_cell_id", _boom)

    bundle = load_ei_data(ei_dir, DATASET)
    try:
        assert len(bundle["ei_data"]) == len(CELL_IDS)
    finally:
        bundle["ei_data"].close()


def test_missing_ei_file_still_yields_none(tmp_path):
    assert load_ei_data(tmp_path, DATASET) is None
