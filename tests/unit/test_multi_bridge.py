"""File ▸ Match Runs with several runs, and the reference ID shift (PLAN.md Q62)."""

import numpy as np
import pandas as pd

from src.analysis.data_manager import DataManager
from src.analysis.reference_bridge import MultiBridge, ReferenceBridge
from tests.unit.test_borrowed_chirp import _bridge as chirp_bridge
from tests.unit.test_borrowed_chirp import _raw_grating_bridge as grating_bridge


def _dm(bridge):
    dm = DataManager.__new__(DataManager)
    dm.reference_bridge = bridge
    dm.is_vision_only = False
    dm.grating_available = False
    dm.cluster_df = pd.DataFrame({"cluster_id": [10, 11], "n_spikes": [100, 100]})
    return dm


def test_vision_only_reference_serves_the_matched_cell_not_its_neighbour(tmp_path):
    """Keys loaded with shift 0 must be looked up with shift 0 (was ref Vision ID − 1)."""
    np.save(tmp_path / "x_Chirp.npy", {
        "psth_mean": np.array([[1.0, 1.0], [2.0, 2.0]]), "cluster_id": np.array([10, 11]),
        "quality_index": np.array([0.1, 0.2]), "bin_size_ms": 10}, allow_pickle=True)
    data, id_to_row = ReferenceBridge._try_load_chirp(tmp_path, ref_is_vision_only=True)
    b = ReferenceBridge({5: 11}, ref_chirp_data=data, ref_chirp_id_to_row=id_to_row, ref_id_shift=0)
    psth, _qi = b.get_chirp_row(5)
    assert psth[0] == 2.0                                   # Vision 11's row, not Vision 10's
    data, id_to_row = ReferenceBridge._try_load_chirp(tmp_path, ref_is_vision_only=False)
    b = ReferenceBridge({5: 11}, ref_chirp_data=data, ref_chirp_id_to_row=id_to_row)
    assert b.get_chirp_row(5)[0][0] == 2.0                  # Kilosort keys: shift 1 both ways


def test_each_response_comes_from_the_run_that_has_it():
    chirp, grating = chirp_bridge(), grating_bridge()      # both match current Vision 11
    multi = MultiBridge([chirp, grating])
    assert multi.pick(11, "chirp") is chirp and multi.pick(11, "grating") is grating
    assert multi.pick(99, "match") is None
    assert multi.has_any_chirp() and multi.has_any_grating()
    dm = _dm(multi)
    g = dm.get_borrowed_grating(10)
    assert g is not None and g["borrowed"]["run"].endswith("data023")
    c = dm.get_borrowed_chirp(10)
    assert c is not None and c["borrowed"]["run"].endswith("data014")
    # Order decides: a run listed first wins for a kind both have.
    assert MultiBridge([grating, chirp]).pick(11, "match") is grating


def test_summary_names_the_run_behind_each_kind():
    from src.gui import callbacks
    chirp, grating = chirp_bridge(), grating_bridge()
    multi = MultiBridge([chirp, grating])
    multi.precompute_gratings()
    dm = _dm(multi)
    c = callbacks.borrow_counts(dm, multi)
    assert c["matched"] == 1 and c["grating"] == 1 and c["chirp"] == 1
    assert c["by_run"]["data023"]["grating"] == 1 and c["by_run"]["data014"]["chirp"] == 1
    text = callbacks.borrow_summary(c, "data014 + data023")
    assert "Drifting gratings: 1 cells" in text and "Chirp: 1 cells" in text


def test_caveats_prefer_a_real_match():
    chirp, grating = chirp_bridge(), grating_bridge()
    multi = MultiBridge([chirp, grating])
    cav = multi.build_ui_caveats()
    assert set(cav) == {10}
    assert cav[10].status == "high"


def test_cached_borrowed_physics_is_good_only_while_its_run_is_matched():
    """A saved session's borrowed time course must not come back as this run's own."""
    import threading
    from types import SimpleNamespace
    dm = DataManager.__new__(DataManager)
    dm.is_vision_only = False
    dm._feature_lock = threading.Lock()
    entry = {"_computed": True, "timecourse": np.ones(30), "rf_area": 5.0,
             "provenance": {"timecourse": "reference", "rf_geometry": "reference"},
             "_reference_run": "/x/data022"}

    def bridge(path):
        b = SimpleNamespace(ref_run_path=path, has_match=lambda vid: vid == 11)
        b.pick = lambda vid, kind: b if vid == 11 and kind in ("rf", "match") else None
        return b

    dm.reference_bridge = None
    assert not dm._physics_entry_is_fresh(10, entry)          # reopened, nothing matched
    dm.reference_bridge = bridge("/x/data022")
    assert dm._physics_entry_is_fresh(10, entry)              # the same run matched again
    dm.reference_bridge = bridge("/x/data006")
    assert not dm._physics_entry_is_fresh(10, entry)          # matched to another run
    own = dict(entry, provenance={"timecourse": "current", "rf_geometry": "current"})
    dm.reference_bridge = None
    assert dm._physics_entry_is_fresh(10, own)                # the run's own data: unaffected
