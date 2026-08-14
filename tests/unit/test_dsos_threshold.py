"""DS/OS slider must drive the grating-panel classification, not just the mosaic."""

import numpy as np

from src.analysis.grating_calc import select_best_dsos_condition
from src.gui.panels.grating_panel import select_dsos_for_display


def _dsos_entry(dsi, osi=0.1, peak=10.0, p=0.001):
    return {
        "condition_type": "dsos",
        "DSI": dsi,
        "OSI": osi,
        "DSI_pvalue": p,
        "OSI_pvalue": p,
        "peak_rate_hz": peak,
        "preferred_direction_deg": 90.0,
        "preferred_orientation_deg": 45.0,
    }


def test_select_dsos_for_display_honors_slider_threshold():
    data = {(100.0, 2.0): _dsos_entry(0.4)}

    default = select_dsos_for_display(data, None)
    assert default["classification"] == "DS"

    raised = select_dsos_for_display(data, 0.5)
    assert raised["classification"] == "none"

    lowered = select_dsos_for_display(data, 0.3)
    assert lowered["classification"] == "DS"


def test_select_dsos_for_display_matches_grating_calc_override():
    data = {(80.0, 1.0): _dsos_entry(0.22, osi=0.45)}
    helper = select_dsos_for_display(data, 0.2)
    direct = select_best_dsos_condition(
        data, dsi_threshold=0.2, osi_threshold=0.2
    )
    assert helper["classification"] == direct["classification"]
    assert helper["classification"] in ("DS", "OS")
    np.testing.assert_allclose(helper["DSI"], direct["DSI"])


def test_compass_cells_are_unique_for_twelve_directions():
    """12×30° gratings used to map two dirs onto each corner cell."""
    from src.gui.panels.grating_panel import assign_directions_to_compass

    dirs = np.arange(0.0, 360.0, 30.0)
    assigned = assign_directions_to_compass(dirs)
    cells = [cell for cell, _d in assigned]
    assert len(cells) == len(set(cells))
    assert len(assigned) == 8
