import numpy as np
import pandas as pd

from src.analysis.data_manager import DataManager


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
