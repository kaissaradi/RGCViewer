"""Semantic color palettes and shared UI constants for RGCViewer.

Canonical Bauhaus tokens live in ``PALETTE_LIGHT`` / ``PALETTE_DARK``.
Do not invent extra primaries. Ink is warm black (``#1B1B1B``), never
``#000000``. See ``docs/design/palette.md``.

Semantic keys (``bg_base``, ``accent``, ``plot_acg``, …) are derived from
those eight tokens so QSS and ``restyle_plots`` keep stable names.
"""

import pyqtgraph as pg

# Product mark in the Swiss header. Window title stays "RGC Viewer".
APP_NAME = "ENCORE"

# Locked 2026-08-12. These eight names are the only primaries.
PALETTE_LIGHT = {
    "bg": "#F2EFE6",  # warm paper, not white
    "surface": "#FFFFFF",
    "ink": "#1B1B1B",  # warm black, never #000
    "muted": "#6E6A61",
    "rule": "#D9D4C7",
    "red": "#C8322B",
    "yellow": "#E9B520",
    "blue": "#1B4E9B",
}

PALETTE_DARK = {
    "bg": "#1A1917",
    "surface": "#232220",
    "ink": "#F2EFE6",
    "muted": "#A19C91",
    "rule": "#35332E",
    "red": "#E8564A",
    "yellow": "#F5C842",
    "blue": "#4A82D6",
}

# Functional extras (not Bauhaus primaries). Used for status / AA fills.
_GOOD_LIGHT = "#1F7A4D"
_GOOD_DARK = "#5DCAA0"
_YELLOW_TEXT_LIGHT = "#8A6500"  # #E9B520 fails AA as 12px text on white
_BLUE_PLOT_DARK = "#6B9BE0"  # #4A82D6 is 4.13:1 on #232220
_BLUE_FILL_DARK = "#1B4E9B"  # #4A82D6 + white is 3.85:1
_RED_TEXT_DARK = "#F28A82"  # #E8564A is 4.44:1 on #232220


DARK_COLORS = {
    "bg_base": PALETTE_DARK["bg"],
    "bg_panel": PALETTE_DARK["surface"],
    "bg_surface": "#2A2926",
    "bg_elevated": "#312F2B",
    "bg_overlay": "rgba(0,0,0,0.55)",
    "bg_tooltip": "#312F2B",
    "accent": _BLUE_FILL_DARK,
    "accent_hover": "#2260B5",
    "accent_pressed": "#163E80",
    "accent_muted": "rgba(74,130,214,0.18)",
    "accent_text": "#FFFFFF",
    "accent_positive": _GOOD_DARK,
    "accent_pos_text": _GOOD_DARK,
    "text_primary": PALETTE_DARK["ink"],
    "text_secondary": PALETTE_DARK["muted"],
    "text_tertiary": PALETTE_DARK["muted"],
    "text_disabled": "#6B675F",
    "text_tooltip": PALETTE_DARK["ink"],
    "border_subtle": PALETTE_DARK["rule"],
    "border_default": PALETTE_DARK["rule"],
    "border_strong": "#4A4740",
    "border_focus": PALETTE_DARK["blue"],
    "status_good_bg": "rgba(93, 202, 160, 0.16)",
    "status_good_text": _GOOD_DARK,
    "status_mua_bg": "rgba(245, 200, 66, 0.16)",
    "status_mua_text": PALETTE_DARK["yellow"],
    "status_noise_bg": "rgba(232, 86, 74, 0.16)",
    "status_noise_text": _RED_TEXT_DARK,
    "status_unsort_bg": "rgba(74, 130, 214, 0.16)",
    "status_unsort_text": _BLUE_PLOT_DARK,
    "selection_bg": "rgba(74, 130, 214, 0.22)",
    "selection_bg_strong": "rgba(74, 130, 214, 0.36)",
    "plot_bg": PALETTE_DARK["surface"],
    "plot_line": PALETTE_DARK["ink"],
    "plot_scatter": _BLUE_PLOT_DARK,
    "plot_shadow": PALETTE_DARK["muted"],
    "plot_ensemble": _BLUE_PLOT_DARK,
    "plot_fill": _BLUE_PLOT_DARK,
    "plot_mean": PALETTE_DARK["ink"],
    "plot_peak": PALETTE_DARK["yellow"],
    "plot_highlight": _BLUE_PLOT_DARK,
    "plot_acg": _BLUE_PLOT_DARK,
    "plot_isi": _BLUE_PLOT_DARK,
    "plot_fr": PALETTE_DARK["yellow"],
    "plot_overlay": _GOOD_DARK,
    "plot_compare": PALETTE_DARK["red"],
    "plot_waveform_shadow": PALETTE_DARK["rule"],
}


LIGHT_COLORS = {
    "bg_base": PALETTE_LIGHT["bg"],
    "bg_panel": PALETTE_LIGHT["surface"],
    "bg_surface": PALETTE_LIGHT["bg"],
    "bg_elevated": "#E8E4D8",
    "bg_overlay": "rgba(27,27,27,0.32)",
    "bg_tooltip": PALETTE_LIGHT["surface"],
    "accent": PALETTE_LIGHT["blue"],
    "accent_hover": "#15448A",
    "accent_pressed": "#12356F",
    "accent_muted": "rgba(27,78,155,0.12)",
    "accent_text": "#FFFFFF",
    "accent_positive": _GOOD_LIGHT,
    "accent_pos_text": _GOOD_LIGHT,
    "text_primary": PALETTE_LIGHT["ink"],
    "text_secondary": PALETTE_LIGHT["muted"],
    "text_tertiary": PALETTE_LIGHT["muted"],
    "text_disabled": "#A39E94",
    "text_tooltip": PALETTE_LIGHT["ink"],
    "border_subtle": PALETTE_LIGHT["rule"],
    "border_default": PALETTE_LIGHT["rule"],
    "border_strong": "#B8B2A4",
    "border_focus": PALETTE_LIGHT["blue"],
    "status_good_bg": "rgba(31, 122, 77, 0.12)",
    "status_good_text": _GOOD_LIGHT,
    "status_mua_bg": "rgba(138, 101, 0, 0.14)",
    "status_mua_text": _YELLOW_TEXT_LIGHT,
    "status_noise_bg": "rgba(200, 50, 43, 0.12)",
    "status_noise_text": PALETTE_LIGHT["red"],
    "status_unsort_bg": "rgba(27, 78, 155, 0.12)",
    "status_unsort_text": PALETTE_LIGHT["blue"],
    "selection_bg": "rgba(27, 78, 155, 0.16)",
    "selection_bg_strong": "rgba(27, 78, 155, 0.28)",
    "plot_bg": PALETTE_LIGHT["surface"],
    "plot_line": PALETTE_LIGHT["ink"],
    "plot_scatter": PALETTE_LIGHT["blue"],
    "plot_shadow": PALETTE_LIGHT["muted"],
    "plot_ensemble": PALETTE_LIGHT["blue"],
    "plot_fill": PALETTE_LIGHT["blue"],
    "plot_mean": PALETTE_LIGHT["ink"],
    "plot_peak": PALETTE_LIGHT["yellow"],
    "plot_highlight": PALETTE_LIGHT["blue"],
    "plot_acg": PALETTE_LIGHT["blue"],
    "plot_isi": PALETTE_LIGHT["blue"],
    "plot_fr": PALETTE_LIGHT["yellow"],
    "plot_overlay": PALETTE_LIGHT["blue"],
    "plot_compare": PALETTE_LIGHT["red"],
    "plot_waveform_shadow": PALETTE_LIGHT["rule"],
}


# 12 (dark, light) pairs. Each variant is WCAG AA against that theme's bg_panel.
PLOT_CATEGORICAL = [
    (_BLUE_PLOT_DARK, PALETTE_LIGHT["blue"]),
    (PALETTE_DARK["yellow"], _YELLOW_TEXT_LIGHT),
    (_RED_TEXT_DARK, PALETTE_LIGHT["red"]),
    (PALETTE_DARK["ink"], PALETTE_LIGHT["ink"]),
    (_GOOD_DARK, _GOOD_LIGHT),
    ("#F0A050", "#B45309"),
    ("#B794F4", "#6D28D9"),
    ("#F472B6", "#BE185D"),
    ("#A3E635", "#3F6212"),
    ("#67C4E8", "#0E7490"),
    ("#E9B520", "#A16207"),
    ("#FB7185", "#9F1239"),
]


SP_1 = 4
SP_2 = 8
SP_3 = 12
SP_4 = 16
SP_5 = 24

PANEL_PADDING = SP_2
CTRL_SPACING = SP_2
ROW_HEIGHT = 28

TYPE_HEADING = 13
TYPE_BODY = 12
TYPE_CAPTION = 11
TYPE_MONO = 11

UI_FONT_FAMILY = "Helvetica Neue, Helvetica, Arial, sans-serif"


def format_run_meta(exp_name, datafile_name, sorter_name, n_cells) -> str:
    """Swiss header breadcrumb: ``20251015A / chunk20 / kilosort4  ·  312 cells``."""
    parts = [str(p) for p in (exp_name, datafile_name, sorter_name) if p]
    if not parts:
        return "No run loaded"
    try:
        n = int(n_cells)
    except (TypeError, ValueError):
        n = 0
    crumb = " / ".join(parts)
    if n > 0:
        return f"{crumb}  ·  {n} cells"
    return crumb


def get_theme_colors(theme_name: str) -> dict:
    """Return a copy of the requested semantic palette."""
    return dict(LIGHT_COLORS if theme_name == "light" else DARK_COLORS)


def resolve_theme_colors(colors: dict = None, theme_name: str = "dark") -> dict:
    """Merge a partial color mapping onto the complete semantic palette."""
    resolved = get_theme_colors(theme_name)
    if colors:
        resolved.update(colors)
    return resolved


def is_light_theme(colors: dict = None, theme_name: str = "dark") -> bool:
    """True when *bg_panel* is closer to white than to black."""
    colors = resolve_theme_colors(colors, theme_name)
    bg = colors.get("bg_panel", "#000000")
    if not isinstance(bg, str) or not bg.startswith("#") or len(bg) < 7:
        return False
    hex_bg = bg[:7]
    return contrast_ratio("#000000", hex_bg) >= contrast_ratio("#ffffff", hex_bg)


def plot_stroke(colors: dict = None, weight: str = "line") -> float:
    """Line width that stays readable on the current plot field.

    Light-mode traces sit on white, so they are drawn heavier than the same
    role on a dark field.
    """
    light = is_light_theme(colors)
    strokes = {
        "thin": (1.6, 1.0),
        "line": (2.6, 2.0),
        "thick": (3.2, 2.4),
    }
    lo, dk = strokes.get(weight, strokes["line"])
    return lo if light else dk


def plot_grid_alpha(colors: dict = None) -> float:
    """Hairline grid. The mockup panes are almost gridless."""
    return 0.10 if is_light_theme(colors) else 0.08


def plot_field(colors: dict = None, theme_name: str = "dark") -> str:
    """Background of a plot, which may differ from chrome ``bg_panel``."""
    colors = resolve_theme_colors(colors, theme_name)
    return colors.get("plot_bg", colors["bg_panel"])


def plot_ensemble_alpha(colors: dict = None) -> float:
    """Many overlapping traces: saturated enough to read on paper/surface."""
    return 0.65 if is_light_theme(colors) else 0.45


def plot_rf_bg_alpha(colors: dict = None) -> float:
    return 0.90 if is_light_theme(colors) else 0.32


def plot_rf_target_alpha(colors: dict = None) -> float:
    return 1.0 if is_light_theme(colors) else 0.75


def opaque_brush(color_hex: str, alpha: int = 255):
    """Fully-specified pyqtgraph brush so fills do not inherit a wash."""
    c = pg.mkColor(color_hex)
    c.setAlpha(int(alpha))
    return pg.mkBrush(c)


def configure_pyqtgraph_theme(colors: dict) -> None:
    """Apply global pyqtgraph colors for newly-created widgets."""
    pg.setConfigOption("background", plot_field(colors))
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
    bg = plot_field(colors)

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
        plot_item.showGrid(x=True, y=True, alpha=plot_grid_alpha(colors))
        plot_item.setContentsMargins(SP_2, SP_2, SP_2, SP_2)
    except Exception:
        pass


def _style_mpl_axes(ax, colors: dict) -> None:
    bg = plot_field(colors)
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
    grid_alpha = 0.55 if is_light_theme(colors) else 0.35
    ax.grid(True, color=colors["border_subtle"], linewidth=0.7, alpha=grid_alpha)


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
        "highlight": colors["plot_peak"],
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
