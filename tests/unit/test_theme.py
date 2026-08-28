from matplotlib.figure import Figure

from src.gui.theme import (
    APP_NAME,
    DARK_COLORS,
    LIGHT_COLORS,
    PALETTE_DARK,
    PALETTE_LIGHT,
    PLOT_CATEGORICAL,
    SP_1,
    SP_2,
    SP_3,
    SP_4,
    SP_5,
    apply_plot_theme,
    contrast_ratio,
    feature_palette,
    format_run_meta,
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


def test_locked_palette_tokens():
    assert PALETTE_LIGHT == {
        "bg": "#F2EFE6",
        "surface": "#FFFFFF",
        "ink": "#1B1B1B",
        "muted": "#6E6A61",
        "rule": "#D9D4C7",
        "red": "#C8322B",
        "yellow": "#E9B520",
        "blue": "#1B4E9B",
    }
    assert PALETTE_DARK == {
        "bg": "#1A1917",
        "surface": "#232220",
        "ink": "#F2EFE6",
        "muted": "#A19C91",
        "rule": "#35332E",
        "red": "#E8564A",
        "yellow": "#F5C842",
        "blue": "#4A82D6",
    }
    assert APP_NAME == "ENCORE"


def test_semantic_roles_map_onto_locked_palette():
    assert LIGHT_COLORS["bg_base"] == PALETTE_LIGHT["bg"]
    assert LIGHT_COLORS["bg_panel"] == PALETTE_LIGHT["surface"]
    assert LIGHT_COLORS["text_primary"] == PALETTE_LIGHT["ink"]
    assert LIGHT_COLORS["text_secondary"] == PALETTE_LIGHT["muted"]
    assert LIGHT_COLORS["border_default"] == PALETTE_LIGHT["rule"]
    assert LIGHT_COLORS["accent"] == PALETTE_LIGHT["blue"]
    assert LIGHT_COLORS["plot_fr"] == PALETTE_LIGHT["yellow"]
    assert LIGHT_COLORS["plot_acg"] == PALETTE_LIGHT["blue"]
    assert LIGHT_COLORS["plot_line"] == PALETTE_LIGHT["ink"]
    assert DARK_COLORS["bg_base"] == PALETTE_DARK["bg"]
    assert DARK_COLORS["plot_fr"] == PALETTE_DARK["yellow"]
    assert DARK_COLORS["accent"] != "#e30613"


def test_ink_is_warm_black_never_pure_black():
    assert LIGHT_COLORS["text_primary"].lower() != "#000000"
    assert LIGHT_COLORS["plot_line"].lower() != "#000000"
    assert LIGHT_COLORS["plot_mean"].lower() != "#000000"
    assert LIGHT_COLORS["text_primary"] == "#1B1B1B"


def test_accent_is_palette_blue_and_readable():
    assert LIGHT_COLORS["accent"] == PALETTE_LIGHT["blue"]
    assert DARK_COLORS["accent"] == "#1B4E9B"
    assert LIGHT_COLORS["accent"].lower() != "#e30613"
    assert DARK_COLORS["accent"].lower() != "#e30613"
    assert contrast_ratio(DARK_COLORS["accent_text"], DARK_COLORS["accent"]) >= 4.5
    assert contrast_ratio(LIGHT_COLORS["accent_text"], LIGHT_COLORS["accent"]) >= 4.5
    assert contrast_ratio(DARK_COLORS["accent_text"], DARK_COLORS["accent_hover"]) >= 4.5
    assert contrast_ratio(LIGHT_COLORS["accent_text"], LIGHT_COLORS["accent_hover"]) >= 4.5


def test_light_and_dark_plot_roles_are_designed_separately():
    assert LIGHT_COLORS["plot_line"] != DARK_COLORS["plot_line"]
    assert LIGHT_COLORS["plot_fr"] != DARK_COLORS["plot_fr"]
    assert LIGHT_COLORS["plot_acg"] != DARK_COLORS["plot_acg"]
    assert LIGHT_COLORS["accent"] != "#e30613"


def test_light_plots_use_bauhaus_primaries_not_only_black():
    assert LIGHT_COLORS["plot_acg"] == PALETTE_LIGHT["blue"]
    assert LIGHT_COLORS["plot_fr"] == PALETTE_LIGHT["yellow"]
    assert LIGHT_COLORS["plot_ensemble"] == PALETTE_LIGHT["blue"]
    assert LIGHT_COLORS["plot_isi"] != LIGHT_COLORS["plot_fr"]


def test_format_run_meta_breadcrumb():
    assert format_run_meta(None, None, None, 0) == "No run loaded"
    assert (
        format_run_meta("20251015A", "chunk20", "kilosort4", 312)
        == "20251015A / chunk20 / kilosort4  ·  312 cells"
    )
    assert format_run_meta("20251015A", "chunk20", "kilosort4", 0) == (
        "20251015A / chunk20 / kilosort4"
    )


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
    assert contrast_ratio(LIGHT_COLORS["text_tertiary"], bg) >= 4.5
    assert contrast_ratio(LIGHT_COLORS["status_good_text"], bg) >= 4.5
    assert contrast_ratio(LIGHT_COLORS["status_mua_text"], bg) >= 4.5


def _mpl_hex(rgba) -> str:
    r, g, b = (int(round(c * 255)) for c in rgba[:3])
    return f"#{r:02x}{g:02x}{b:02x}"
