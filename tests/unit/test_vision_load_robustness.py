"""Older kilosort4-converted Vision files: missing pieces and EI stride mismatch.

kilosort4 converters sometimes write a 512-style .globals next to a 519-wide
.ei (header array_id 1551). EIReader used to take payload width from the
globals electrode map, walk the file at the wrong stride, and treat waveform
floats as cell ids. Missing .sta/.params is also a normal older-run state.
"""

import logging
import threading

import numpy as np
import pytest

from src.analysis.electrode_map import get_litke_array_coordinates_by_array_id
from src.analysis.vision_integration import load_params_data, load_sta_data
from src.analysis import visionloader as vl


DATASET = "kilosort4"
# Litke 519 (30 µm) vs Litke 512 — the pair seen on 20250421B.
EI_ARRAY_ID = 1551
GLOBALS_ARRAY_ID = 504
N_LEFT, N_RIGHT = 3, 4
N_SAMPLES = N_LEFT + N_RIGHT + 1
CELL_IDS = [2, 6, 10, 14]


def _n_payload_for_array(array_id):
    """Channels written into the .ei, including the TTL row."""
    return get_litke_array_coordinates_by_array_id(array_id).shape[0] + 1


def _write_mismatched_ei(folder, cell_ids=CELL_IDS, seed=0):
    """519-wide .ei + 512-wide .globals, no .sta / .params."""
    n_payload = _n_payload_for_array(EI_ARRAY_ID)
    rng = np.random.default_rng(seed)

    # Globals claims the 512 array and carries no electrode map, so the
    # reader would previously take n_electrodes from that 512-row table.
    vl.GlobalsFileWriter(str(folder), DATASET).write(GLOBALS_ARRAY_ID)

    written = {}
    payload = {}
    for cid in cell_ids:
        ei = rng.normal(size=(n_payload, N_SAMPLES)).astype(np.float32)
        err = rng.normal(size=(n_payload, N_SAMPLES)).astype(np.float32)
        payload[cid] = (ei, err, 100 + cid)
        written[cid] = (ei, err)
    vl.EIFileWriter(str(folder), DATASET, N_LEFT, N_RIGHT, EI_ARRAY_ID).write(payload)
    return written, n_payload


@pytest.fixture
def mismatched_dir(tmp_path):
    _write_mismatched_ei(tmp_path)
    return tmp_path


def test_ei_reader_uses_ei_header_array_id_not_globals_map(tmp_path):
    written, n_payload = _write_mismatched_ei(tmp_path)
    n_elec = n_payload - 1
    mismatched_dir = tmp_path

    with vl.EIReader(str(mismatched_dir), DATASET) as reader:
        assert reader.n_electrodes == n_elec
        assert sorted(reader.cell_id_to_offset) == CELL_IDS
        assert max(reader.cell_id_to_offset) < 100_000

        for cid, (ei, err) in written.items():
            got = reader.get_ei_for_cell_id(cid)
            # TTL row is dropped, same as a matched file.
            np.testing.assert_array_equal(got.ei, ei[1:])
            np.testing.assert_array_equal(got.ei_error, err[1:])
            assert got.ei.shape == (n_elec, N_SAMPLES)

        # Plot map has to match the payload, not the 512-row .globals.
        assert reader.get_electrode_map().shape[0] == n_elec


def test_ei_reader_file_size_divides_exactly_for_519_and_512(tmp_path):
    for array_id, n_cells in ((EI_ARRAY_ID, 5), (GLOBALS_ARRAY_ID, 5)):
        folder = tmp_path / str(array_id)
        folder.mkdir()
        n_payload = _n_payload_for_array(array_id)
        vl.GlobalsFileWriter(str(folder), DATASET).write(array_id)
        payload = {}
        for cid in range(1, n_cells + 1):
            ei = np.zeros((n_payload, N_SAMPLES), dtype=np.float32)
            err = np.zeros((n_payload, N_SAMPLES), dtype=np.float32)
            payload[cid] = (ei, err, cid)
        vl.EIFileWriter(str(folder), DATASET, N_LEFT, N_RIGHT, array_id).write(payload)

        with vl.EIReader(str(folder), DATASET) as reader:
            assert len(reader.cell_id_to_offset) == n_cells
            rec = reader.num_bytes_per_ei + 8
            body = (folder / f"{DATASET}.ei").stat().st_size - reader.header_size
            assert body % rec == 0
            assert body // rec == n_cells


def test_load_sta_data_missing_file_returns_none(mismatched_dir, caplog):
    caplog.set_level(logging.ERROR)
    assert load_sta_data(mismatched_dir, DATASET) is None
    assert not any(
        rec.levelno >= logging.ERROR for rec in caplog.records
    ), "a missing .sta is expected on older runs, not an unexpected error"


def test_load_params_data_missing_file_returns_none(mismatched_dir, caplog):
    caplog.set_level(logging.ERROR)
    assert load_params_data(mismatched_dir, DATASET) is None
    assert not any(
        rec.levelno >= logging.ERROR for rec in caplog.records
    ), "a missing .params is expected on older runs, not an unexpected error"


def test_lazy_sta_getitem_missing_key_does_not_call_reader(tmp_path):
    from unittest.mock import MagicMock, patch

    from src.analysis.vision_integration import LazySTADict

    mock_reader = MagicMock()
    mock_reader.cell_id_to_byte_offset = {2: 0, 4: 100}
    mock_reader.get_sta_for_cell_id.side_effect = AssertionError("should not read")

    with patch("src.analysis.vision_integration.vl") as mock_vl:
        mock_vl.STAReader.return_value = mock_reader
        lazy = LazySTADict(tmp_path, "test")

    assert 3 not in lazy
    assert lazy[3] is None
    mock_reader.get_sta_for_cell_id.assert_not_called()
    assert lazy.get(3) is None
    mock_reader.get_sta_for_cell_id.assert_not_called()


def test_attach_sta_quality_only_touches_ids_in_sta():
    import pandas as pd

    from src.analysis.data_manager import DataManager

    reads = []

    class _FakeSTA:
        def __init__(self, vid):
            reads.append(vid)
            vol = np.zeros((4, 4, 2), dtype=float)
            vol[1, 1, 0] = 10.0
            self.green = vol
            self.red = None

    class _FakeSTADict:
        def __init__(self):
            self._ids = {2, 6}

        def __contains__(self, key):
            return int(key) in self._ids

        def __iter__(self):
            return iter(self._ids)

        def keys(self):
            return list(self._ids)

        def __getitem__(self, key):
            key = int(key)
            if key not in self._ids:
                raise AssertionError("must not probe missing STA ids")
            return _FakeSTA(key)

        def __len__(self):
            return len(self._ids)

    dm = DataManager.__new__(DataManager)
    dm.vision_stas = _FakeSTADict()
    dm.is_vision_only = False
    dm.cluster_df = pd.DataFrame({"cluster_id": [1, 2, 5, 6]})
    dm.sta_ids_consistent = True
    dm.sta_consistency_message = ""

    assert dm.prepare_sta_quality_column() is True
    # hybrid mode: cluster_id + 1 → vision id. Only 1→2 and 5→6 exist.
    assert reads == [2, 6]
    assert dm.attach_sta_quality_column() is True
    assert list(np.isfinite(dm.cluster_df["sta_snr"])) == [True, False, True, False]


def test_retire_includes_grating_batch_and_stops_physics_warm():
    from src.gui.callbacks import _retire_inflight_load

    class _FakeSignal:
        def __init__(self):
            self.slots = []

        def connect(self, slot):
            self.slots.append(slot)

        def disconnect(self):
            self.slots.clear()

    class _FakeThread:
        def __init__(self):
            self._running = True
            self.quit_called = False
            self.finished = _FakeSignal()

        def isRunning(self):
            return self._running

        def quit(self):
            self.quit_called = True

        def deleteLater(self):
            pass

    class _FakeWorker:
        def __init__(self):
            self.stop_called = False
            self.finished = _FakeSignal()
            self.progress = _FakeSignal()
            self.error = _FakeSignal()

        def stop(self):
            self.stop_called = True

    class _FakeWindow:
        pass

    win = _FakeWindow()
    win._physics_warm_stop = threading.Event()
    win._grating_batch_thread = _FakeThread()
    win._grating_batch_worker = _FakeWorker()
    win.ks_load_thread = None
    win.ks_load_worker = None
    win.vision_load_thread = None
    win.vision_load_worker = None
    win.feature_worker_thread = None
    win.feature_worker = None

    batch_worker = win._grating_batch_worker
    parked = _retire_inflight_load(win)
    assert win._physics_warm_stop.is_set()
    assert batch_worker.stop_called
    assert len(parked) == 1
    assert parked[0].quit_called
    assert win._grating_batch_thread is None
    assert win._grating_batch_worker is None
