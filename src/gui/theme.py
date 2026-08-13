"""Semantic color palettes and shared UI constants for RGCViewer.

Swiss layout (hairline panes, caption titles, black ink on white) with a
blue chrome accent instead of the mockup red. Population ensembles use a
chromatic wash (`plot_ensemble`), not grey. Existing semantic key names
are kept so QSS and restyle_plots do not break on rename.
"""

import pyqtgraph as pg

DARK_COLORS = {
    "bg_base": "#111214",
    "bg_panel": "#181b22",
    "bg_surface": "#1e222b",
    "bg_elevated": "#272c37",
    "bg_overlay": "rgba(0,0,0,0.55)",
    "bg_tooltip": "#272c37",
    "accent": "#2563eb",
    "accent_hover": "#3b82f6",
    "accent_pressed": "#1d4ed8",
    "accent_muted": "rgba(37,99,235,0.18)",
    "accent_text": "#ffffff",
    "accent_positive": "#34d399",
    "accent_pos_text": "#34d399",
    "text_primary": "#f8fafc",
    "text_secondary": "#94a3b8",
    "text_tertiary": "#74859b",
    "text_disabled": "#475569",
    "text_tooltip": "#f8fafc",
    "border_subtle": "#2a303c",
    "border_default": "#3a4252",
    "border_strong": "#5b6578",
    "border_focus": "#3b82f6",
    "status_good_bg": "rgba(52, 211, 153, 0.16)",
    "status_good_text": "#34d399",
    "status_mua_bg": "rgba(251, 191, 36, 0.16)",
    "status_mua_text": "#fbbf24",
    "status_noise_bg": "rgba(251, 113, 133, 0.16)",
    "status_noise_text": "#fb7185",
    "status_unsort_bg": "rgba(56, 189, 248, 0.16)",
    "status_unsort_text": "#38bdf8",
    "selection_bg": "rgba(37, 99, 235, 0.18)",
    "selection_bg_strong": "rgba(37, 99, 235, 0.30)",
    "plot_bg": "#181b22",
    "plot_line": "#f8fafc",
    "plot_scatter": "#60a5fa",
    "plot_shadow": "#94a3b8",
    "plot_ensemble": "#7eb8f7",
    "plot_fill": "#334155",
    "plot_mean": "#f8fafc",
    "plot_peak": "#fbbf24",
    "plot_highlight": "#22d3ee",
    "plot_acg": "#c4b5fd",
    "plot_isi": "#38bdf8",
    "plot_fr": "#fbbf24",
    "plot_overlay": "#34d399",
    "plot_compare": "#fb923c",
    "plot_waveform_shadow": "#334155",
}


LIGHT_COLORS = {
    "bg_base": "#f2f2f2",
    "bg_panel": "#ffffff",
    "bg_surface": "#f2f2f2",
    "bg_elevated": "#eaeaea",
    "bg_overlay": "rgba(15,23,42,0.28)",
    "bg_tooltip": "#ffffff",
    "accent": "#1d4ed8",
    "accent_hover": "#1e40af",
    "accent_pressed": "#1e3a8a",
    "accent_muted": "rgba(29,78,216,0.10)",
    "accent_text": "#ffffff",
    "accent_positive": "#047857",
    "accent_pos_text": "#047857",
    "text_primary": "#000000",
    "text_secondary": "#333333",
    "text_tertiary": "#666666",
    "text_disabled": "#a0a0a0",
    "text_tooltip": "#000000",
    "border_subtle": "#e0e0e0",
    "border_default": "#d0d0d0",
    "border_strong": "#a0a0a0",
    "border_focus": "#1d4ed8",
    "status_good_bg": "rgba(4, 120, 87, 0.12)",
    "status_good_text": "#047857",
    "status_mua_bg": "rgba(180, 83, 9, 0.14)",
    "status_mua_text": "#b45309",
    "status_noise_bg": "rgba(190, 18, 60, 0.12)",
    "status_noise_text": "#be123c",
    "status_unsort_bg": "rgba(3, 105, 161, 0.12)",
    "status_unsort_text": "#0369a1",
    "selection_bg": "rgba(29, 78, 216, 0.14)",
    "selection_bg_strong": "rgba(29, 78, 216, 0.24)",
    "plot_bg": "#ffffff",
    "plot_line": "#000000",
    "plot_scatter": "#1d4ed8",
    "plot_shadow": "#888888",
    "plot_ensemble": "#2f5aa0",
    "plot_fill": "#d4d4d4",
    "plot_mean": "#000000",
    "plot_peak": "#c47f00",
    "plot_highlight": "#1d4ed8",
    "plot_acg": "#000000",
    "plot_isi": "#000000",
    "plot_fr": "#000000",
    "plot_overlay": "#666666",
    "plot_compare": "#c2410c",
    "plot_waveform_shadow": "#e0e0e0",
}


# 12 (dark, light) pairs. Each variant is WCAG AA against that theme's bg_panel.
PLOT_CATEGORICAL = [
    ("#60a5fa", "#1d4ed8"),
    ("#fbbf24", "#b45309"),
    ("#34d399", "#047857"),
    ("#f8fafc", "#0f172a"),
    ("#fb923c", "#c2410c"),
    ("#22d3ee", "#0e7490"),
    ("#c4b5fd", "#6d28d9"),
    ("#f472b6", "#be185d"),
    ("#a3e635", "#3f6212"),
    ("#38bdf8", "#0369a1"),
    ("#fde047", "#a16207"),
    ("#fb7185", "#be123c"),
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
        "thin": (1.2, 0.9),
        "line": (2.2, 1.8),
        "thick": (2.8, 2.3),
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
    """Many overlapping traces: a chromatic wash, not a grey slab."""
    return 0.38 if is_light_theme(colors) else 0.36


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
