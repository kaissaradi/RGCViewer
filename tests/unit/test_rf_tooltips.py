"""RF short vs long is explained where users pick it (PLAN.md Q25)."""

from src.analysis import feature_catalog as fc
from src.gui.panels import feature_extraction as fe
from src.gui.panels import umap_panel


def test_umap_rf_checkbox_explains_long_and_short():
    tip = umap_panel.FEATURE_TOOLTIPS["use_rf_diameter"]
    assert "longer axis" in tip and "shorter axis" in tip and "stixels" in tip


def test_fixed_panels_do_not_claim_microns():
    labels = [t for meta in fe.FeatureExtractionWindow._PLOT_META for t in meta]
    assert not any("µm" in t for t in labels)
