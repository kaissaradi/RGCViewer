"""DS/OS population view without STA RFs, and preferred-orientation polar."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from unittest.mock import MagicMock

from src.gui.panels.population_panel import (
    _draw_preferred_orientation_polar,
    _iter_dsos_population,
    draw_population_rfs_plot,
    pop_canvas_can_hot_swap,
)
from src.gui.theme import DARK_COLORS


def _entry(dsi=0.1, osi=0.1, pref_dir=90.0, pref_ori=45.0, peak=10.0, p=0.001):
    return {
        (100.0, 2.0): {
            "condition_type": "dsos",
            "DSI": dsi,
            "OSI": osi,
            "DSI_pvalue": p,
            "OSI_pvalue": p,
            "peak_rate_hz": peak,
            "preferred_direction_deg": pref_dir,
            "preferred_orientation_deg": pref_ori,
            "mean_response": np.array([peak, 0.2, 0.1, 0.2]),
        }
    }


def test_iter_dsos_population_does_not_need_sta():
    mw = MagicMock()
    mw.dsos_threshold = 0.3
    mw.data_manager.cluster_df = pd.DataFrame({"cluster_id": [0, 1, 2]})

    def grating(cid):
        if cid == 0:
            return _entry(dsi=0.55)
        if cid == 1:
            return _entry(dsi=0.05, osi=0.62)
        return None

    mw.data_manager.get_grating_data_for_cluster.side_effect = grating
    rows = list(_iter_dsos_population(mw, None))
    by_id = {cid: sel["classification"] for cid, sel in rows}
    assert by_id == {0: "DS", 1: "OS"}


def test_polar_draws_ds_arrows_and_os_bars():
    fig = Figure()
    ax = fig.add_subplot(111)
    rows = [
        (
            0,
            {
                "classification": "DS",
                "DSI": 0.5,
                "OSI": 0.1,
                "preferred_direction_deg": 90.0,
                "preferred_orientation_deg": 0.0,
            },
        ),
        (
            1,
            {
                "classification": "OS",
                "DSI": 0.1,
                "OSI": 0.6,
                "preferred_direction_deg": 0.0,
                "preferred_orientation_deg": 45.0,
            },
        ),
    ]
    n_ds, n_os = _draw_preferred_orientation_polar(ax, rows, DARK_COLORS)
    assert n_ds == 1 and n_os == 1
    assert ax.findobj(FancyArrowPatch)
    assert any(isinstance(c, LineCollection) for c in ax.collections)


def test_population_without_sta_uses_orientation_polar():
    fig = Figure()
    canvas = MagicMock()
    canvas.fig = fig
    canvas._pop_plot_state = None
    assert pop_canvas_can_hot_swap(canvas) is False

    mw = MagicMock()
    mw.dsos_threshold = 0.3
    mw.population_view_enabled = False
    mw.rf_canvas = canvas
    mw.get_current_colors.return_value = DARK_COLORS
    mw.pop_show_ids_checkbox.isChecked.return_value = False
    mw.data_manager.vision_params = None
    mw.data_manager.reference_bridge = None
    mw.data_manager.grating_available = True
    mw.data_manager.grating_status = "raw_only"
    mw.data_manager.is_vision_only = False
    mw.data_manager.cluster_df = pd.DataFrame({"cluster_id": [4]})
    mw.data_manager.get_grating_data_for_cluster.return_value = _entry(dsi=0.5)

    draw_population_rfs_plot(mw, subset_cell_ids=[4], canvas=canvas)

    assert fig.axes, "expected a polar axes instead of the no-Vision placeholder"
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("0°" in t or "90°" in t for t in texts)
    assert fig.axes[0].findobj(FancyArrowPatch)
    assert isinstance(canvas._pop_plot_state, dict)
    assert pop_canvas_can_hot_swap(canvas) is True

    # A second draw (cluster click) must not wipe state to None or rebuild.
    draw_population_rfs_plot(mw, subset_cell_ids=[4], canvas=canvas)
    assert isinstance(canvas._pop_plot_state, dict)
    assert pop_canvas_can_hot_swap(canvas) is True


def test_none_pop_plot_state_is_not_hot_swappable():
    canvas = MagicMock()
    canvas._pop_plot_state = None
    canvas.fig.axes = []
    assert pop_canvas_can_hot_swap(canvas) is False
    if hasattr(canvas, "_pop_plot_state"):
        state = canvas._pop_plot_state
    assert state is None or isinstance(state, dict)
