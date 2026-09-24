"""DS/OS, DSI and OSI in the cluster table (PLAN.md Q7).

The lab meeting asked for DSI/OSI in the table. They were never there.
The table uses the Grating tab's Auto pick, so both show the same numbers.
"""

import threading

import numpy as np
import pandas as pd

from src.analysis.data_manager import DataManager


def _entry(dsi, osi, p=0.01, peak=10.0):
    return {
        "condition_type": "dsos",
        "directions_deg": np.array([0.0, 90.0, 180.0, 270.0]),
        "mean_response": np.array([peak, 1.0, 1.0, 1.0]),
        "DSI": dsi, "OSI": osi, "DSI_pvalue": p, "OSI_pvalue": p,
    }


def _dm(grating_data, ids):
    dm = DataManager.__new__(DataManager)
    dm.cluster_df = pd.DataFrame({"cluster_id": ids})
    dm.grating_available = True
    dm.grating_status = "ok"
    dm.grating_data = grating_data
    dm.grating_computed_cache = {}
    dm._grating_cache_lock = threading.Lock()
    return dm


def test_columns_follow_the_auto_pick_and_threshold():
    dm = _dm({
        0: {(100.0, 2.0): _entry(0.6, 0.2)},               # DS
        1: {(100.0, 2.0): _entry(0.1, 0.5)},               # OS
        2: {(100.0, 2.0): _entry(0.4, 0.1, p=0.5)},        # not significant
    }, ids=[0, 1, 2, 3])                                    # 3: no grating data
    assert dm.attach_grating_columns(0.3)
    df = dm.cluster_df
    assert list(df["dsos"]) == ["DS", "OS", "", ""]
    assert np.allclose(df["dsi"][:3], [0.6, 0.1, 0.4])     # shown even when not significant
    assert np.isnan(df["dsi"][3])

    dm.attach_grating_columns(0.55)                         # stricter slider
    assert list(dm.cluster_df["dsos"]) == ["DS", "", "", ""]


def test_no_grating_leaves_the_table_alone():
    dm = _dm({}, ids=[0])
    dm.grating_available = False
    assert not dm.attach_grating_columns()
    assert "dsos" not in dm.cluster_df.columns
