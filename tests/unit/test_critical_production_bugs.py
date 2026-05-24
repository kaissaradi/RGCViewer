from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.analysis.data_manager import DataManager
from src.gui import callbacks


class SimilarTemplatesSpy:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)
        self.getitem_calls = 0

    @property
    def shape(self):
        return self._values.shape

    def __getitem__(self, item):
        self.getitem_calls += 1
        return self._values[item]


class FakeSTA:
    def __init__(self, red):
        self.red = np.asarray(red, dtype=float)
        self.green = self.red + 1.0
        self.blue = self.red * 2.0


class FakeLazySTADict:
    def __init__(self, stas_by_id):
        self._stas_by_id = stas_by_id
        self.keys_list = list(stas_by_id)
        self.getitem_calls = 0

    def __bool__(self):
        return True

    def __contains__(self, item):
        return item in self._stas_by_id

    def __getitem__(self, item):
        self.getitem_calls += 1
        return self._stas_by_id[item]
    
    def keys(self):              # ← add this
        return self.keys_list


class FakeVisionLoadThread:
    def __init__(self):
        self.quit_calls = 0
        self.wait_calls = []

    def quit(self):
        self.quit_calls += 1

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        return True


class FakeVisionLoadWorker:
    def __init__(self):
        self.delete_later_calls = 0

    def deleteLater(self):
        self.delete_later_calls += 1


class FakeCentralWidget:
    def __init__(self):
        self.enabled_values = []

    def setEnabled(self, enabled):
        self.enabled_values.append(enabled)


class FakeStatusBar:
    def __init__(self):
        self.messages = []

    def showMessage(self, *args):
        self.messages.append(args)


def test_kilosort_params_uses_literal_eval_without_executing_code(tmp_path):
    marker = tmp_path / "literal_eval_executed.txt"
    params_path = tmp_path / "params.py"
    params_path.write_text(
        "\n".join(
            [
                "fs = 20000",
                "n_channels_dat = 64",
                "dtype = 'int16'",
                "dat_path = ('raw_a.dat', 'raw_b.dat')",
                "channel_map = [0, 1, 2]",
                f"dangerous = __import__('pathlib').Path({str(marker)!r}).write_text('ran')",
            ]
        )
    )

    dm = DataManager(kilosort_dir=str(tmp_path))

    dm._load_kilosort_params()

    assert dm.sampling_rate == 20000
    assert dm.n_channels == 64
    assert not marker.exists()


def test_mea_similarity_table_reuses_cluster_cache(tmp_path):
    dm = DataManager(kilosort_dir=str(tmp_path))
    dm.similar_templates = SimilarTemplatesSpy(
        [
            [1.0, 0.9, 0.3],
            [0.9, 1.0, 0.2],
            [0.3, 0.2, 1.0],
        ]
    )
    dm.cluster_df = pd.DataFrame(
        {
            "cluster_id": [10, 11, 12],
            "n_spikes": [100, 80, 60],
            "status": ["good", "dup", "noise"],
            "set": ["keep", "review", "drop"],
            "x_um": [0.0, 3.0, 4.0],
            "y_um": [0.0, 4.0, 0.0],
        }
    )
    dm.cluster_to_template = {10: 0, 11: 1, 12: 2}
    dm.cluster_id_to_idx = {10: 0, 11: 1, 12: 2}
    dm.mea_sim_cache = {}

    first = dm._get_mea_similarity_table(10, top_n=2)

    assert dm.similar_templates.getitem_calls == 1
    assert 10 in dm.mea_sim_cache
    assert list(first["cluster_id"]) == [11, 12]
    assert list(first["template_sim"]) == [0.9, 0.3]
    assert list(first.columns) == [
        "cluster_id",
        "n_spikes",
        "status",
        "distance_um",
        "template_sim",
        "set",
    ]

    dm.similar_templates.getitem_calls = 0
    second = dm._get_mea_similarity_table(10, top_n=2)

    assert dm.similar_templates.getitem_calls == 0
    pd.testing.assert_frame_equal(second, first)
    assert second is not dm.mea_sim_cache[10]


def test_vision_similarity_table_reuses_cluster_cache(tmp_path):
    dm = DataManager(kilosort_dir=str(tmp_path))
    dm.is_vision_only = True
    dm.vision_stas = FakeLazySTADict(
        {
            10: FakeSTA([[0.0, 1.0], [2.0, 3.0]]),
            11: FakeSTA([[0.0, 2.0], [4.0, 6.0]]),
            12: FakeSTA([[3.0, 2.0], [1.0, 0.0]]),
        }
    )
    dm.vision_params = object()
    dm.cluster_df = pd.DataFrame(
        {
            "cluster_id": [10, 11, 12],
            "n_spikes": [100, 80, 60],
            "status": ["good", "dup", "noise"],
            "set": ["keep", "review", "drop"],
        }
    )
    dm.vision_sim_cache = {}

    first = dm._get_vision_similarity_table(10, top_n=2)

    assert dm.vision_stas.getitem_calls > 0
    assert 10 in dm.vision_sim_cache
    assert list(first["cluster_id"]) == [11, 12]
    assert list(first.columns) == [
        "cluster_id",
        "n_spikes",
        "status",
        "set",
        "template_sim",
    ]

    dm.vision_stas.getitem_calls = 0
    second = dm._get_vision_similarity_table(10, top_n=2)

    assert dm.vision_stas.getitem_calls == 0
    pd.testing.assert_frame_equal(second, first)
    assert second is not dm.vision_sim_cache[10]


@pytest.mark.parametrize(
    ("thread_state", "worker_state"),
    [
        ("missing", "missing"),
        ("cleared", "cleared"),
        ("present", "present"),
    ],
)
def test_on_vision_native_loaded_handles_stale_thread_and_worker_cleanup(
    monkeypatch, thread_state, worker_state
):
    critical_calls = []
    monkeypatch.setattr(
        callbacks.QMessageBox,
        "critical",
        lambda *args: critical_calls.append(args),
    )

    main_window = SimpleNamespace(
        central_widget=FakeCentralWidget(),
        status_bar=FakeStatusBar(),
    )

    thread = None
    if thread_state == "cleared":
        main_window.vision_load_thread = None
    elif thread_state == "present":
        thread = FakeVisionLoadThread()
        main_window.vision_load_thread = thread

    worker = None
    if worker_state == "cleared":
        main_window.vision_load_worker = None
    elif worker_state == "present":
        worker = FakeVisionLoadWorker()
        main_window.vision_load_worker = worker

    callbacks._on_vision_native_loaded(
        main_window,
        success=False,
        message="standalone Vision load failed",
        vision_dir_name="/tmp/fake-vision",
    )

    assert critical_calls
    assert main_window.status_bar.messages == [("Loading failed.", 5000)]
    assert main_window.central_widget.enabled_values == [True]

    if thread is not None:
        assert thread.quit_calls == 1
        assert thread.wait_calls == [2000]
    if worker is not None:
        assert worker.delete_later_calls == 1

    if thread_state != "missing":
        assert main_window.vision_load_thread is None
    if worker_state != "missing":
        assert main_window.vision_load_worker is None
