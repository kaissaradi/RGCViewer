def test_theme_palettes_have_identical_semantic_keys():
    from src.gui.theme import DARK_COLORS, LIGHT_COLORS

    assert set(DARK_COLORS) == set(LIGHT_COLORS)


def test_theme_palettes_include_required_plot_roles():
    from src.gui.theme import DARK_COLORS, LIGHT_COLORS

    required_roles = {
        "bg_base",
        "bg_panel",
        "bg_surface",
        "text_primary",
        "text_secondary",
        "text_tertiary",
        "border_default",
        "border_subtle",
        "plot_line",
        "plot_scatter",
        "plot_shadow",
        "plot_mean",
        "plot_peak",
        "plot_highlight",
    }

    assert required_roles.issubset(DARK_COLORS)
    assert required_roles.issubset(LIGHT_COLORS)
