"""Presence checks for chirp/contrast/grating must be glob-only.

The Kilosort load worker used to np.load every matching .npy under
"Checking for grating analysis data...". Those files are pickled object
dicts and can be hundreds of MB. Checking whether they exist is a glob.
Loading them happens later, and only for the one file we will use.
"""

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.analysis.data_manager import DataManager


def _analysis_dm(tmp_path):
    dm = DataManager.__new__(DataManager)
    dm.kilosort_dir = Path(tmp_path)
    dm.is_vision_only = False
    dm.chirp_available = False
    dm.chirp_data = None
    dm.chirp_id_to_row = None
    dm.contrast_available = False
    dm.contrast_status = "missing"
    dm.contrast_data = None
    dm.grating_available = False
    dm.grating_status = "missing"
    dm.grating_data = None
    dm.grating_raw_data = None
    dm.grating_conditions = None
    dm.grating_computed_cache = {}
    dm._grating_cache_lock = threading.Lock()
    dm._analysis_candidates = {}
    dm.cluster_df = pd.DataFrame({"cluster_id": [0, 1]})
    dm.vision_stas = None
    return dm


def _write_npy_dict(path, payload):
    np.save(path, payload, allow_pickle=True)


def _analyzed_grating_dict():
    return {
        1: {
            (200.0, 2.0): {"condition_type": "dsos", "dsi": 0.4},
        }
    }


def _raw_grating_dict():
    return {
        "spike_times_by_trial": {1: [[0.1], [0.2]]},
        "trial_parameters": [
            {"barWidth": 200.0, "temporalFrequency": 2.0},
        ],
    }


def test_probe_stimulus_analyses_does_not_unpickle(tmp_path):
    """A file that would explode on np.load must still count as present."""
    bomb = tmp_path / "data000_ChirpStimulus.npy"
    bomb.write_bytes(b"not a numpy file")
    (tmp_path / "data000_contrastResponse_unified.npy").write_bytes(b"nope")
    (tmp_path / "data000_GratingDSOS.npy").write_bytes(b"nope")

    dm = _analysis_dm(tmp_path)
    found = dm.probe_stimulus_analyses()

    assert found["chirp"] == [bomb]
    assert len(found["contrast"]) == 1
    assert len(found["grating"]) == 1
    assert dm.chirp_available is False
    assert dm.grating_status == "missing"


def test_probe_finds_ksfiles_layout(tmp_path):
    ksfiles = tmp_path / "ksfiles"
    ksfiles.mkdir()
    chirp = ksfiles / "data010-013_ChirpStimulus.npy"
    chirp.write_bytes(b"x")

    dm = _analysis_dm(tmp_path)
    found = dm.probe_stimulus_analyses()
    assert found["chirp"] == [chirp]


def test_probe_finds_parent_dir(tmp_path):
    run = tmp_path / "data006"
    run.mkdir()
    chirp = tmp_path / "data006_ChirpStimulus.npy"
    grating = tmp_path / "data006_GratingDSOS.npy"
    chirp.write_bytes(b"x")
    grating.write_bytes(b"x")

    dm = _analysis_dm(run)
    found = dm.probe_stimulus_analyses()
    assert found["chirp"] == [chirp]
    assert found["grating"] == [grating]


def test_probe_prefers_selected_dir_over_parent(tmp_path):
    run = tmp_path / "data006"
    run.mkdir()
    here = run / "data006_ChirpStimulus.npy"
    above = tmp_path / "data006_ChirpStimulus.npy"
    here.write_bytes(b"x")
    above.write_bytes(b"x")

    dm = _analysis_dm(run)
    found = dm.probe_stimulus_analyses()
    assert found["chirp"][0] == here
    assert above in found["chirp"]


def test_probe_parent_prefers_this_run_name(tmp_path):
    run = tmp_path / "data006"
    run.mkdir()
    other = tmp_path / "data000_ChirpStimulus.npy"
    mine = tmp_path / "data006_ChirpStimulus.npy"
    other.write_bytes(b"x")
    mine.write_bytes(b"x")

    dm = _analysis_dm(run)
    found = dm.probe_stimulus_analyses()
    assert found["chirp"][0] == mine


def test_probe_lists_each_root_once(tmp_path, monkeypatch):
    import os

    run = tmp_path / "data006"
    (run / "ksfiles").mkdir(parents=True)
    (tmp_path / "ksfiles").mkdir()
    (run / "x_ChirpStimulus.npy").write_bytes(b"x")

    calls = []
    real = os.scandir

    def counting(path):
        calls.append(str(Path(path).resolve()))
        return real(path)

    monkeypatch.setattr(os, "scandir", counting)
    dm = _analysis_dm(run)
    dm.probe_stimulus_analyses()
    resolved = [str(Path(p).resolve()) for p in calls]
    assert resolved.count(str(run.resolve())) == 1
    assert resolved.count(str((run / "ksfiles").resolve())) == 1
    assert resolved.count(str(tmp_path.resolve())) == 1
    assert resolved.count(str((tmp_path / "ksfiles").resolve())) == 1
    n = len(calls)
    dm.find_analysis_files("*Chirp*.npy")
    assert len(calls) == n


def test_locate_vision_in_parent(tmp_path):
    from src.gui.workers.workers import _locate_vision_dataset

    run = tmp_path / "data006"
    run.mkdir()
    (tmp_path / "data006.sta").write_bytes(b"x")
    vision_dir, name = _locate_vision_dataset(run)
    assert vision_dir == tmp_path
    assert name == "data006"


def test_locate_vision_prefers_matching_stem(tmp_path):
    from src.gui.workers.workers import _locate_vision_dataset

    run = tmp_path / "data006"
    run.mkdir()
    (tmp_path / "data000.ei").write_bytes(b"x")
    (tmp_path / "data006.sta").write_bytes(b"x")
    vision_dir, name = _locate_vision_dataset(run)
    assert vision_dir == tmp_path
    assert name == "data006"


def test_locate_vision_prefers_files_in_selected_dir(tmp_path):
    from src.gui.workers.workers import _locate_vision_dataset

    run = tmp_path / "data006"
    run.mkdir()
    (run / "data006.sta").write_bytes(b"x")
    (tmp_path / "data000.ei").write_bytes(b"x")
    vision_dir, name = _locate_vision_dataset(run)
    assert vision_dir == run
    assert name == "data006"


def test_load_grating_prefers_combined_and_loads_it_once(tmp_path, monkeypatch):
    raw_path = tmp_path / "data000_GratingDSOS.npy"
    combined = tmp_path / "data000_combined_GratingDSOS.npy"
    leftover = tmp_path / "data000_old_GratingDSOS.npy"
    _write_npy_dict(raw_path, _raw_grating_dict())
    _write_npy_dict(combined, _analyzed_grating_dict())
    _write_npy_dict(leftover, _raw_grating_dict())

    loads = []
    real_load = np.load

    def _counting_load(path, *args, **kwargs):
        loads.append(Path(path).name)
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", _counting_load)

    dm = _analysis_dm(tmp_path)
    ok, msg = dm.load_grating_data()

    assert ok
    assert dm.grating_status == "ok"
    assert 0 in dm.grating_data  # Vision 1 → KS 0
    assert loads == ["data000_combined_GratingDSOS.npy"]
    assert "combined" in msg


def test_load_grating_raw_only_when_no_analyzed(tmp_path):
    raw_path = tmp_path / "data000_GratingDSOS.npy"
    _write_npy_dict(raw_path, _raw_grating_dict())

    dm = _analysis_dm(tmp_path)
    dm._load_grating_cache_from_disk = lambda: {}
    ok, _msg = dm.load_grating_data()

    assert ok
    assert dm.grating_status == "raw_only"
    assert dm.grating_raw_data is not None
    assert 0 in dm.grating_raw_data["spike_times_by_trial"]


def test_load_grating_does_not_unpickle_a_second_raw_file(tmp_path, monkeypatch):
    """Two raw GratingDSOS.npy files (sort dir + parent) must not both load.

    Each is ~80 MB of pickled trials. Opening the second looking for an
    analyzed file is what made 'Checking for grating' hang on every load.
    """
    _write_npy_dict(tmp_path / "a_GratingDSOS.npy", _raw_grating_dict())
    _write_npy_dict(tmp_path / "b_GratingDSOS.npy", _raw_grating_dict())

    loads = []
    real_load = np.load

    def _counting_load(path, *args, **kwargs):
        loads.append(Path(path).name)
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", _counting_load)

    dm = _analysis_dm(tmp_path)
    dm._load_grating_cache_from_disk = lambda: {}
    ok, _msg = dm.load_grating_data()

    assert ok
    assert dm.grating_status == "raw_only"
    assert loads == ["a_GratingDSOS.npy"]


def test_grating_ids_needing_compute_skips_cached_and_analyzed():
    dm = _analysis_dm("/tmp")
    dm.grating_status = "raw_only"
    dm.grating_raw_data = {"spike_times_by_trial": {}, "trial_parameters": []}
    dm.grating_computed_cache = {1: {"dsi": 0.2}, 2: None}
    assert dm.grating_ids_needing_compute([1, 2, 3]) == [2, 3]

    dm.grating_status = "ok"
    assert dm.grating_ids_needing_compute([1, 2, 3]) == []


def test_analysis_worker_loads_probed_chirp(tmp_path):
    from src.gui.workers.workers import StimulusAnalysisLoadWorker

    chirp = tmp_path / "data000_ChirpStimulus.npy"
    _write_npy_dict(
        chirp,
        {
            "psth_mean": np.zeros((1, 8)),
            "cluster_id": np.array([1]),
            "quality_index": np.array([0.5]),
            "bin_size_ms": 20.0,
        },
    )
    dm = _analysis_dm(tmp_path)
    dm.probe_stimulus_analyses()
    worker = StimulusAnalysisLoadWorker(dm)
    results = []
    worker.finished.connect(lambda ok, msg: results.append((ok, msg)))
    worker.run()

    assert results == [(True, "Stimulus analyses ready.")]
    assert dm.chirp_available is True
    assert 0 in dm.chirp_id_to_row


def test_kilosort_worker_probes_without_loading(tmp_path, monkeypatch):
    from src.gui.workers import workers as W

    monkeypatch.setattr(W, "_locate_vision_dataset", lambda _ks: (None, None))

    dm = _analysis_dm(tmp_path)
    (tmp_path / "x_ChirpStimulus.npy").write_bytes(b"x")
    (tmp_path / "x_GratingDSOS.npy").write_bytes(b"x")
    dm.load_kilosort_data = lambda: (True, "ok")
    dm.build_cluster_dataframe = lambda: None
    dm.load_cell_type_file = lambda _p: None
    dm.load_stimulus_manifest = lambda: None
    dm.load_chirp_data = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("check must not load chirp")
    )
    dm.load_contrast_data = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("check must not load contrast")
    )
    dm.load_grating_data = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("check must not load grating")
    )

    worker = W.KilosortLoadWorker(dm, str(tmp_path), None)
    messages = []
    results = []
    worker.progress.connect(messages.append)
    worker.finished.connect(lambda ok, msg: results.append((ok, msg)))
    worker.run()

    assert results == [(True, "Kilosort and Vision data loaded successfully.")]
    assert any("Checking for chirp" in m for m in messages)
    assert any("Checking for grating" in m for m in messages)
    assert dm._analysis_candidates["chirp"]
    assert dm._analysis_candidates["grating"]


def test_prepare_sta_quality_uses_reader_snr_not_full_movie():
    dm = DataManager.__new__(DataManager)
    dm.is_vision_only = False
    dm.cluster_df = pd.DataFrame({"cluster_id": [1, 2]})
    dm.sta_ids_consistent = True
    dm.sta_consistency_message = ""
    dm.kilosort_dir = Path("/tmp")

    class _SnrDict:
        def __init__(self):
            self.ids = {2, 3}
            self.getitem_calls = []
            self.snr_calls = []

        def __contains__(self, key):
            return int(key) in self.ids

        def keys(self):
            return list(self.ids)

        def __getitem__(self, key):
            self.getitem_calls.append(int(key))
            raise AssertionError("quality sweep must not unpack the STA movie")

        def sta_peak_to_rms(self, key):
            self.snr_calls.append(int(key))
            return 12.5

    stas = _SnrDict()
    dm.vision_stas = stas

    assert dm.prepare_sta_quality_column() is True
    assert stas.snr_calls == [2, 3]
    assert stas.getitem_calls == []
    assert list(dm._sta_snr_values) == [12.5, 12.5]


def test_sta_snr_cache_roundtrip(tmp_path):
    dm = DataManager.__new__(DataManager)
    dm.is_vision_only = False
    dm.cluster_df = pd.DataFrame({"cluster_id": [1, 2]})
    dm.kilosort_dir = tmp_path
    dm.sta_ids_consistent = True
    dm.sta_consistency_message = ""
    dm._save_lock = threading.Lock()
    dm._save_in_progress = False

    sta_path = tmp_path / "data000.sta"
    sta_path.write_bytes(b"sta")

    class _Cached:
        vision_dir = tmp_path
        dataset_name = "data000"

        def __contains__(self, key):
            return True

        def keys(self):
            return [2, 3]

        def sta_peak_to_rms(self, key):
            return float(key)

    dm.vision_stas = _Cached()
    assert dm.prepare_sta_quality_column() is True
    assert list(dm._sta_snr_values) == [2.0, 3.0]
    cache = tmp_path / "sta_snr_cache.pkl"
    assert cache.is_file()

    dm2 = DataManager.__new__(DataManager)
    dm2.is_vision_only = False
    dm2.cluster_df = pd.DataFrame({"cluster_id": [1, 2]})
    dm2.kilosort_dir = tmp_path
    dm2.sta_ids_consistent = True
    dm2.sta_consistency_message = ""

    class _MustNotRead:
        vision_dir = tmp_path
        dataset_name = "data000"

        def __contains__(self, key):
            return True

        def keys(self):
            return [2, 3]

        def sta_peak_to_rms(self, key):
            raise AssertionError("cache hit must not reread STAs")

        def __getitem__(self, key):
            raise AssertionError("cache hit must not reread STAs")

    dm2.vision_stas = _MustNotRead()
    assert dm2.prepare_sta_quality_column() is True
    assert list(dm2._sta_snr_values) == [2.0, 3.0]


def test_lazy_sta_peak_to_rms_uses_green_channel_helper(tmp_path):
    from src.analysis.vision_integration import LazySTADict

    mock_reader = MagicMock()
    mock_reader.cell_id_to_byte_offset = {6: 0}
    mock_reader.green_peak_to_rms.return_value = 31.0

    with patch("src.analysis.vision_integration.vl") as mock_vl:
        mock_vl.STAReader.return_value = mock_reader
        lazy = LazySTADict(tmp_path, "data000")

    assert lazy.sta_peak_to_rms(6) == 31.0
    mock_reader.green_peak_to_rms.assert_called_once_with(6)
    mock_reader.get_sta_for_cell_id.assert_not_called()


def test_retire_includes_analysis_load_worker():
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
    win._physics_warm_stop = None
    win.analysis_load_thread = _FakeThread()
    win.analysis_load_worker = _FakeWorker()
    win.ks_load_thread = None
    win.ks_load_worker = None
    win.vision_load_thread = None
    win.vision_load_worker = None
    win._grating_batch_thread = None
    win._grating_batch_worker = None
    win.feature_worker_thread = None
    win.feature_worker = None

    worker = win.analysis_load_worker
    parked = _retire_inflight_load(win)
    assert worker.stop_called
    assert len(parked) == 1
    assert win.analysis_load_thread is None
    assert win.analysis_load_worker is None
