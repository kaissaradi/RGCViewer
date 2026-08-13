"""Semantic color palettes and shared UI constants for RGCViewer.

The palettes follow the Bauhaus / Swiss tokens in encore_mockup.html:
red accent, black/white surfaces, no decoration. Existing semantic key
names are kept so QSS and restyle_plots do not break on rename.
"""

import pyqtgraph as pg

DARK_COLORS = {
    "bg_base": "#0d0d0d",
    "bg_panel": "#1a1a1a",
    "bg_surface": "#1a1a1a",
    "bg_elevated": "#242424",
    "bg_overlay": "rgba(0,0,0,0.55)",
    "bg_tooltip": "#242424",
    "accent": "#e30613",
    "accent_hover": "#ff2a36",
    "accent_pressed": "#b0050f",
    "accent_muted": "rgba(227,6,19,0.14)",
    "accent_text": "#ffffff",
    "accent_positive": "#00c853",
    "accent_pos_text": "#00c853",
    "text_primary": "#ffffff",
    "text_secondary": "#b0b0b0",
    "text_tertiary": "#707070",
    "text_disabled": "#555555",
    "text_tooltip": "#ffffff",
    "border_subtle": "#2a2a2a",
    "border_default": "#333333",
    "border_strong": "#555555",
    "border_focus": "#e30613",
    "status_good_bg": "rgba(0, 200, 83, 0.18)",
    "status_good_text": "#00c853",
    "status_mua_bg": "rgba(255, 193, 7, 0.18)",
    "status_mua_text": "#ffc107",
    "status_noise_bg": "rgba(227, 6, 19, 0.18)",
    "status_noise_text": "#ff6b6b",
    "status_unsort_bg": "rgba(10, 132, 255, 0.18)",
    "status_unsort_text": "#64b5ff",
    "selection_bg": "rgba(227, 6, 19, 0.16)",
    "selection_bg_strong": "rgba(227, 6, 19, 0.28)",
    "plot_line": "#ffffff",
    "plot_scatter": "#ff6b6b",
    "plot_shadow": "#707070",
    "plot_mean": "#ffffff",
    "plot_peak": "#ffc107",
    "plot_highlight": "#18ffff",
    "plot_acg": "#d1b3ff",
    "plot_isi": "#64b5ff",
    "plot_fr": "#ffc107",
    "plot_overlay": "#00e676",
    "plot_compare": "#ffb74d",
    "plot_waveform_shadow": "#333333",
}


LIGHT_COLORS = {
    "bg_base": "#f2f2f2",
    "bg_panel": "#ffffff",
    "bg_surface": "#ffffff",
    "bg_elevated": "#eaeaea",
    "bg_overlay": "rgba(0,0,0,0.25)",
    "bg_tooltip": "#ffffff",
    "accent": "#e30613",
    "accent_hover": "#b0050f",
    "accent_pressed": "#8a040c",
    "accent_muted": "rgba(227,6,19,0.08)",
    "accent_text": "#ffffff",
    "accent_positive": "#00703a",
    "accent_pos_text": "#00703a",
    "text_primary": "#000000",
    "text_secondary": "#333333",
    "text_tertiary": "#666666",
    "text_disabled": "#a0a0a0",
    "text_tooltip": "#000000",
    "border_subtle": "#e0e0e0",
    "border_default": "#d0d0d0",
    "border_strong": "#a0a0a0",
    "border_focus": "#e30613",
    "status_good_bg": "rgba(0, 112, 58, 0.12)",
    "status_good_text": "#00703a",
    "status_mua_bg": "rgba(245, 166, 35, 0.18)",
    "status_mua_text": "#8a5a00",
    "status_noise_bg": "rgba(227, 6, 19, 0.12)",
    "status_noise_text": "#b0050f",
    "status_unsort_bg": "rgba(10, 132, 255, 0.12)",
    "status_unsort_text": "#0050b4",
    "selection_bg": "rgba(227, 6, 19, 0.10)",
    "selection_bg_strong": "rgba(227, 6, 19, 0.18)",
    "plot_line": "#000000",
    "plot_scatter": "#c00000",
    "plot_shadow": "#888888",
    "plot_mean": "#000000",
    "plot_peak": "#8a5a00",
    "plot_highlight": "#006f7a",
    "plot_acg": "#5b2bb3",
    "plot_isi": "#0050b4",
    "plot_fr": "#8a5a00",
    "plot_overlay": "#00703a",
    "plot_compare": "#b34a00",
    "plot_waveform_shadow": "#e0e0e0",
}


# 12 (dark, light) pairs. Each variant is WCAG AA against that theme's bg_panel.
PLOT_CATEGORICAL = [
    ("#ff6b6b", "#c00000"),
    ("#ffd54f", "#8a6a00"),
    ("#64b5ff", "#0050b4"),
    ("#ffffff", "#000000"),
    ("#00e676", "#00703a"),
    ("#ffb74d", "#b34a00"),
    ("#18ffff", "#006f7a"),
    ("#d1b3ff", "#5b2bb3"),
    ("#ff80ab", "#ad1457"),
    ("#d4ff4d", "#5a7000"),
    ("#1de9b6", "#00695c"),
    ("#ffe082", "#7a4f00"),
]


SP_1 = 4
SP_2 = 8
SP_3 = 12
SP_4 = 16
SP_5 = 24

PANEL_PADDING = SP_2
CTRL_SPACING = 6
ROW_HEIGHT = 28

TYPE_HEADING = 13
TYPE_BODY = 12
TYPE_CAPTION = 11
TYPE_MONO = 11

UI_FONT_FAMILY = "Helvetica Neue, Helvetica, Arial, sans-serif"


def get_theme_colors(theme_name: str) -> dict:
    """Return a copy of the requested semantic palette."""
    return dict(LIGHT_COLORS if theme_name == "light" else DARK_COLORS)


def resolve_theme_colors(colors: dict = None, theme_name: str = "dark") -> dict:
    """Merge a partial color mapping onto the complete semantic palette."""
    resolved = get_theme_colors(theme_name)
    if colors:
        resolved.update(colors)
    return resolved


def configure_pyqtgraph_theme(colors: dict) -> None:
    """Apply global pyqtgraph colors for newly-created widgets."""
    pg.setConfigOption("background", colors["bg_panel"])
    pg.setConfigOption("foreground", colors["text_secondary"])
    pg.setConfigOptions(antialias=True)


def _as_mpl_color(value: str):
    """Matplotlib accepts #hex; leave rgba strings for Qt."""
    return value


def apply_plot_theme(target, colors: dict) -> None:
    """Paint a pyqtgraph or matplotlib plot from the semantic palette.

    Accepts PlotWidget, PlotItem, GraphicsLayoutWidget, matplotlib Figure,
    or matplotlib Axes. Unknown objects are ignored.
    """
    colors = resolve_theme_colors(colors)
    bg = colors["bg_panel"]
    spine = colors["border_default"]
    tick = colors["text_secondary"]

    if pg is not None:
        plot_widget = getattr(pg, "PlotWidget", None)
        layout_widget = getattr(pg, "GraphicsLayoutWidget", None)
        plot_item_cls = getattr(pg, "PlotItem", None)
        if plot_widget is not None and isinstance(target, plot_widget):
            target.setBackground(bg)
            _style_pg_plot_item(target.getPlotItem(), colors)
            return
        if layout_widget is not None and isinstance(target, layout_widget):
            target.setBackground(bg)
            return
        if plot_item_cls is not None and isinstance(target, plot_item_cls):
            _style_pg_plot_item(target, colors)
            return

    fig = getattr(target, "figure", None)
    axes = getattr(target, "axes", None)
    if fig is None and hasattr(target, "patch") and hasattr(target, "axes"):
        fig = target
        axes = target.axes
    if fig is not None and hasattr(fig, "patch"):
        fig.patch.set_facecolor(_as_mpl_color(bg))
        for ax in list(axes or []):
            _style_mpl_axes(ax, colors)
        return
    if hasattr(target, "set_facecolor") and hasattr(target, "spines"):
        _style_mpl_axes(target, colors)


def _style_pg_plot_item(plot_item, colors: dict) -> None:
    spine = pg.mkPen(colors["border_default"])
    tick = pg.mkPen(colors["text_secondary"])
    for name in ("bottom", "left"):
        try:
            axis = plot_item.getAxis(name)
        except Exception:
            continue
        axis.setPen(spine)
        axis.setTextPen(tick)
    try:
        plot_item.showAxis("top", False)
        plot_item.showAxis("right", False)
        plot_item.showGrid(x=True, y=True, alpha=0.08)
    except Exception:
        pass


def _style_mpl_axes(ax, colors: dict) -> None:
    bg = colors["bg_panel"]
    spine = colors["border_default"]
    tick = colors["text_secondary"]
    ax.set_facecolor(bg)
    for side in ax.spines.values():
        side.set_color(spine)
        side.set_linewidth(0.8)
    ax.tick_params(colors=tick, labelsize=8)
    ax.xaxis.label.set_color(tick)
    ax.yaxis.label.set_color(tick)
    title = ax.title
    if title is not None:
        title.set_color(colors["text_primary"])
    ax.grid(True, color=colors["border_subtle"], linewidth=0.6, alpha=0.8)


def feature_palette(colors: dict = None) -> dict:
    """Map app theme tokens onto the Feature Extraction dialog's local keys."""
    colors = resolve_theme_colors(colors)
    return {
        "bg": colors["bg_base"],
        "surface": colors["bg_panel"],
        "border": colors["border_default"],
        "text_primary": colors["text_primary"],
        "text_muted": colors["text_secondary"],
        "accent": colors["accent"],
        "highlight": colors["plot_compare"],
        "grid": colors["border_subtle"],
        "progress_fg": colors["accent"],
        "progress_bg": colors["bg_elevated"],
        "btn_active": colors["accent"],
        "btn_inactive": colors["bg_elevated"],
    }


def contrast_ratio(hex_fg: str, hex_bg: str) -> float:
    """WCAG contrast ratio between two #rrggbb colors."""

    def _lin(channel: float) -> float:
        return channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4

    def _lum(value: str) -> float:
        raw = value.lstrip("#")
        r, g, b = (int(raw[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)

    lighter, darker = sorted((_lum(hex_fg), _lum(hex_bg)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)
