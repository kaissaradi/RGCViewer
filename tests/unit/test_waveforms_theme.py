"""The Waveforms panel follows the light / dark theme (PLAN.md Q16)."""

from src.gui.panels import waveforms_panel as wp
from src.gui.theme import DARK_COLORS, LIGHT_COLORS


def test_theme_palette_covers_every_panel_colour():
    for colors in (LIGHT_COLORS, DARK_COLORS):
        assert set(wp.palette_from_theme(colors)) == set(wp._C)


def test_light_palette_is_light():
    pal = wp.palette_from_theme(LIGHT_COLORS)
    assert pal["bg_main"] == LIGHT_COLORS["plot_bg"] == "#FFFFFF"
    assert pal["median"] == LIGHT_COLORS["plot_mean"]
    assert pal["cloud_4"][3] > pal["cloud_0"][3]          # ramp kept
