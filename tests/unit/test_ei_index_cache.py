"""The .ei seek table is cached, and never trusted when it might be wrong.

``EIReader`` locates cells by seeking to each record and reading an 8-byte
header. Records are one EI apart (~825 KB), so it is ~700 scattered reads over
a 600 MB file — free on NVMe, but ~9 ms each over CIFS, which made the scan
6.5 s of a 19.5 s open. Cached, that becomes 0.01 s.

Correctness matters more than the speed here: a wrong offset does not fail
loudly, it hands back another cell's EI. So the cache is keyed on the .ei
file's size and mtime, and every failure mode falls back to the scan.
"""

import os
import pickle

import pytest

from src.analysis import ei_index_cache


@pytest.fixture
def ei_file(tmp_path):
    path = tmp_path / "data000.ei"
    path.write_bytes(b"\x00" * 4096)
    return path


OFFSETS = {1: 12, 2: 837, 3: 1662}
NSPIKES = {1: 100, 2: 200, 3: 300}


def test_roundtrip(ei_file):
    assert ei_index_cache.save(ei_file, OFFSETS, NSPIKES)
    assert ei_index_cache.load(ei_file) == (OFFSETS, NSPIKES)


def test_missing_cache_returns_none(ei_file):
    assert ei_index_cache.load(ei_file) is None


def test_index_sits_beside_the_ei_file(ei_file):
    ei_index_cache.save(ei_file, OFFSETS, NSPIKES)
    assert ei_index_cache.cache_path(ei_file).parent == ei_file.parent
    assert ei_index_cache.cache_path(ei_file).exists()


def test_index_is_small(ei_file):
    """A few hundred int pairs — this must not become a large artifact."""
    big = {i: i * 837 + 12 for i in range(1, 1001)}
    ei_index_cache.save(ei_file, big, {i: i for i in big})
    assert ei_index_cache.cache_path(ei_file).stat().st_size < 64 * 1024


def test_changed_ei_invalidates(ei_file):
    """A rewritten .ei makes every cached offset meaningless."""
    ei_index_cache.save(ei_file, OFFSETS, NSPIKES)
    ei_file.write_bytes(b"\x01" * 8192)
    assert ei_index_cache.load(ei_file) is None


def test_touched_ei_invalidates(ei_file):
    """Same size, new mtime — still not to be trusted."""
    ei_index_cache.save(ei_file, OFFSETS, NSPIKES)
    st = os.stat(ei_file)
    os.utime(ei_file, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert ei_index_cache.load(ei_file) is None


def test_version_bump_invalidates(ei_file, monkeypatch):
    ei_index_cache.save(ei_file, OFFSETS, NSPIKES)
    monkeypatch.setattr(ei_index_cache, "INDEX_VERSION",
                        ei_index_cache.INDEX_VERSION + 1)
    assert ei_index_cache.load(ei_file) is None


def test_corrupt_cache_falls_back_to_scan(ei_file):
    ei_index_cache.cache_path(ei_file).write_bytes(b"not a pickle at all")
    assert ei_index_cache.load(ei_file) is None


def test_wrong_payload_shape_falls_back(ei_file):
    payload = {
        ei_index_cache.VERSION_KEY: ei_index_cache.INDEX_VERSION,
        "ei_identity": (ei_file.stat().st_size, ei_file.stat().st_mtime_ns),
        "cell_id_to_offset": "not a dict",
        "cell_id_to_nspikes": {},
    }
    ei_index_cache.cache_path(ei_file).write_bytes(pickle.dumps(payload))
    assert ei_index_cache.load(ei_file) is None


def test_empty_table_is_never_written(ei_file):
    """An empty scan is a failed read, not a result worth remembering."""
    assert ei_index_cache.save(ei_file, {}, {}) is False
    assert not ei_index_cache.cache_path(ei_file).exists()


def test_missing_ei_file_is_survivable(tmp_path):
    absent = tmp_path / "gone.ei"
    assert ei_index_cache.load(absent) is None
    assert ei_index_cache.save(absent, OFFSETS, NSPIKES) is False


def test_unwritable_directory_is_survivable(ei_file):
    """A read-only share is ordinary; it must cost a rescan, not a crash."""
    os.chmod(ei_file.parent, 0o500)
    try:
        assert ei_index_cache.save(ei_file, OFFSETS, NSPIKES) is False
    finally:
        os.chmod(ei_file.parent, 0o700)


def test_no_temp_files_left_behind(ei_file):
    ei_index_cache.save(ei_file, OFFSETS, NSPIKES)
    leftovers = [p for p in ei_file.parent.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []
