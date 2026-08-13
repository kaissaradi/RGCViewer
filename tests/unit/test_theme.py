from matplotlib.figure import Figure

from src.gui.theme import (
    DARK_COLORS,
    LIGHT_COLORS,
    PLOT_CATEGORICAL,
    SP_1,
    SP_2,
    SP_3,
    SP_4,
    SP_5,
    apply_plot_theme,
    contrast_ratio,
    feature_palette,
    get_theme_colors,
    is_light_theme,
    plot_grid_alpha,
    plot_stroke,
)


NEW_TOKENS = {
    "accent_pressed",
    "accent_muted",
    "accent_text",
    "border_focus",
    "bg_overlay",
    "bg_tooltip",
    "text_tooltip",
}


def test_theme_palettes_have_identical_semantic_keys():
    assert set(DARK_COLORS) == set(LIGHT_COLORS)


def test_theme_palettes_include_required_plot_roles():
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
        "plot_bg",
        "plot_ensemble",
        "plot_fill",
    }

    assert required_roles.issubset(DARK_COLORS)
    assert required_roles.issubset(LIGHT_COLORS)


def test_new_tokens_in_both_themes():
    assert NEW_TOKENS.issubset(DARK_COLORS)
    assert NEW_TOKENS.issubset(LIGHT_COLORS)
    for key in NEW_TOKENS:
        assert isinstance(DARK_COLORS[key], str) and DARK_COLORS[key]
        assert isinstance(LIGHT_COLORS[key], str) and LIGHT_COLORS[key]


def test_categorical_palette_length():
    assert len(PLOT_CATEGORICAL) == 12
    for pair in PLOT_CATEGORICAL:
        assert len(pair) == 2


def test_categorical_contrast_ratio():
    for dark_variant, light_variant in PLOT_CATEGORICAL:
        assert contrast_ratio(dark_variant, DARK_COLORS["bg_panel"]) >= 4.5
        assert contrast_ratio(light_variant, LIGHT_COLORS["bg_panel"]) >= 4.5


def test_spacing_constants():
    assert (SP_1, SP_2, SP_3, SP_4, SP_5) == (4, 8, 12, 16, 24)


def test_apply_plot_theme_matplotlib():
    fig = Figure()
    ax = fig.add_subplot(111)
    apply_plot_theme(fig, DARK_COLORS)
    assert _mpl_hex(fig.get_facecolor()) == DARK_COLORS["plot_bg"].lower()
    assert _mpl_hex(ax.get_facecolor()) == DARK_COLORS["plot_bg"].lower()

    apply_plot_theme(fig, LIGHT_COLORS)
    assert _mpl_hex(fig.get_facecolor()) == LIGHT_COLORS["plot_bg"].lower()
    assert _mpl_hex(ax.get_facecolor()) == LIGHT_COLORS["plot_bg"].lower()


def test_feature_palette_tracks_theme_text():
    dark = feature_palette(DARK_COLORS)
    light = feature_palette(LIGHT_COLORS)
    assert dark["text_primary"] == DARK_COLORS["text_primary"]
    assert light["text_primary"] == LIGHT_COLORS["text_primary"]
    assert dark["bg"] == DARK_COLORS["bg_base"]
    assert light["surface"] == LIGHT_COLORS["bg_panel"]


def test_get_theme_colors_returns_copy():
    colors = get_theme_colors("light")
    colors["accent"] = "#000000"
    assert LIGHT_COLORS["accent"] != "#000000"


def test_accent_is_blue_not_bauhaus_red():
    assert DARK_COLORS["accent"].lower() != "#e30613"
    assert LIGHT_COLORS["accent"].lower() != "#e30613"
    assert contrast_ratio(DARK_COLORS["accent_text"], DARK_COLORS["accent"]) >= 4.5
    assert contrast_ratio(LIGHT_COLORS["accent_text"], LIGHT_COLORS["accent"]) >= 4.5


def test_light_and_dark_plot_roles_are_designed_separately():
    assert LIGHT_COLORS["plot_line"] != DARK_COLORS["plot_line"]
    assert LIGHT_COLORS["plot_fr"] != DARK_COLORS["plot_fr"]
    assert LIGHT_COLORS["plot_acg"] != DARK_COLORS["plot_acg"]
    assert LIGHT_COLORS["accent"] != "#e30613"


def test_is_light_theme_and_plot_stroke():
    assert is_light_theme(LIGHT_COLORS) is True
    assert is_light_theme(DARK_COLORS) is False
    assert plot_stroke(LIGHT_COLORS) > plot_stroke(DARK_COLORS)
    assert plot_grid_alpha(LIGHT_COLORS) > plot_grid_alpha(DARK_COLORS)


def test_light_plot_ink_contrasts_against_paper():
    paper = LIGHT_COLORS["plot_bg"]
    for key in ("plot_line", "plot_mean", "plot_acg", "plot_isi", "plot_ensemble"):
        assert contrast_ratio(LIGHT_COLORS[key], paper) >= 4.5, key


def test_light_body_and_status_text_contrast():
    bg = LIGHT_COLORS["bg_panel"]
    assert contrast_ratio(LIGHT_COLORS["text_primary"], bg) >= 4.5
    assert contrast_ratio(LIGHT_COLORS["text_secondary"], bg) >= 4.5
    assert contrast_ratio(LIGHT_COLORS["status_good_text"], bg) >= 4.5
    assert contrast_ratio(LIGHT_COLORS["status_mua_text"], bg) >= 4.5


def _mpl_hex(rgba) -> str:
    r, g, b = (int(round(c * 255)) for c in rgba[:3])
    return f"#{r:02x}{g:02x}{b:02x}"
