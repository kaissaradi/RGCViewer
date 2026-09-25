"""UMAP waits for every selected input (PLAN.md Q28)."""

from types import SimpleNamespace

import pandas as pd

from src.gui.panels.umap_panel import umap_inputs_waiting


def _dm(physics, spike_plots, n=4):
    return SimpleNamespace(
        cluster_df=pd.DataFrame({"cluster_id": range(n)}),
        feature_cache={i: {"_computed": True} for i in range(physics)},
        standard_plot_cache={i: {} for i in range(spike_plots)})


def test_waits_for_physics():
    assert "2 / 4" in umap_inputs_waiting(_dm(2, 4), {"use_acg": True})


def test_waits_for_the_acg_pass_only_when_acg_is_used():
    assert "Autocorrelograms ready: 1 / 4" in umap_inputs_waiting(_dm(4, 1), {"use_acg": True})
    assert umap_inputs_waiting(_dm(4, 1), {"use_acg": False}) == ""


def test_ready():
    assert umap_inputs_waiting(_dm(4, 4), {"use_acg": True}) == ""
