"""Reading a run off the lab server must not corrupt it or thrash the link.

Datasets open either from local NVMe or from ``//bach/Fieldlab`` over CIFS.
Two things about the network case are load-bearing:

* Concurrency is worthless there. Measured on 20260715A/data007-010 with the
  page cache evicted, reading 48 STA cells took 0.81 s on one thread and 0.76 s
  on eight — while median per-read latency went 3.2 ms -> 118.8 ms. The link
  serializes regardless, so extra readers only queue behind each other and push
  ordinary reads toward the 8 s timeout meant to catch a corrupt byte offset.
* When that timeout does fire, the placeholder it writes must stay in RAM. It
  is marked ``_computed`` so the progress bar and UMAP gate can move past a slow
  cell, which makes it indistinguishable from a real result — persisting one
  bakes zeroed physics into the run's feature_cache.pkl for good.
"""

from unittest.mock import mock_open, patch

from src.analysis import cache_persistence, storage


class TestNetworkDetection:
    """/proc/mounts decides, and the longest matching mountpoint wins."""

    MOUNTS = (
        "/dev/nvme0n1p5 / ext4 rw,relatime 0 0\n"
        "//bach/Fieldlab /mnt/lab cifs rw,relatime 0 0\n"
        "server:/export /mnt/nfs nfs4 rw 0 0\n"
    )

    def _patched(self, path, fn):
        with patch("builtins.open", mock_open(read_data=self.MOUNTS)), \
                patch("src.analysis.storage.Path.resolve", lambda self: self):
            return fn(path)

    def test_cifs_mount_is_network(self):
        assert self._patched("/mnt/lab/Array-data/sorted/x", storage.is_network_path)

    def test_nfs_mount_is_network(self):
        assert self._patched("/mnt/nfs/data", storage.is_network_path)

    def test_local_disk_is_not_network(self):
        assert not self._patched("/home/fieldlab/data", storage.is_network_path)

    def test_longest_mountpoint_wins(self):
        """/mnt/lab/... must not be answered by the / entry."""
        assert self._patched("/mnt/lab/x/y/z", storage.filesystem_type) == "cifs"

    def test_single_reader_on_network(self):
        assert self._patched("/mnt/lab/x", storage.io_workers) == 1

    def test_fan_out_kept_on_local_disk(self):
        assert self._patched("/home/fieldlab/x", storage.io_workers) == 4

    def test_unreadable_mount_table_assumes_local(self):
        """A wrong 'local' costs latency; a wrong 'network' serializes a fast disk."""
        with patch("builtins.open", side_effect=OSError):
            assert storage.is_network_path("/anything") is False
            assert storage.io_workers("/anything") == 4


class TestTimedOutEntriesNeverPersist:
    """The placeholder is a session-local concession, not a result."""

    def _timed_out_entry(self):
        return {
            "_computed": True,
            "_timed_out": True,
            "acg": None,
            "timecourse": None,
            "rf_area": 0.0,
            "ellipticity": 0.0,
            "time_to_peak": 0,
        }

    def test_timed_out_row_is_dropped_on_save(self):
        cache = {1: {"_computed": True, "rf_area": 12.0}, 2: self._timed_out_entry()}
        kept = cache_persistence.filter_computed_entries(cache)
        assert 1 in kept
        assert 2 not in kept, "zeroed physics would be read back as truth next session"

    def test_real_rows_still_persist(self):
        cache = {1: {"_computed": True, "rf_area": 12.0, "ellipticity": 0.4}}
        assert cache_persistence.filter_computed_entries(cache) == cache

    def test_partial_rows_still_dropped(self):
        """The original contract: ACG-only rows never reach disk."""
        assert cache_persistence.filter_computed_entries({1: {"acg": [1, 2]}}) == {}


class TestTimedOutEntriesAreRetried:
    """In-session, a placeholder must not satisfy a later request."""

    def test_placeholder_is_not_a_cache_hit(self):
        from src.analysis.data_manager import DataManager

        dm = DataManager.__new__(DataManager)
        entry = {"_computed": True, "_timed_out": True, "timecourse": None}
        assert dm._physics_entry_is_fresh(1, entry) is False

    def test_real_entry_is_a_cache_hit(self):
        from src.analysis.data_manager import DataManager

        dm = DataManager.__new__(DataManager)
        entry = {"_computed": True, "timecourse": [0.1, 0.2]}
        assert dm._physics_entry_is_fresh(1, entry) is True
