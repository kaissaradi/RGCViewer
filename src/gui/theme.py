"""Semantic color palettes and shared UI constants for RGCViewer."""

import pyqtgraph as pg


DARK_COLORS = {
    "bg_base": "#111214",
    "bg_panel": "#18191C",
    "bg_surface": "#1E2025",
    "bg_elevated": "#282A30",
    "accent": "#2E6DD4",
    "accent_hover": "#4A8BEF",
    "accent_positive": "#1A5C3A",
    "accent_pos_text": "#6EE7B7",
    "text_primary": "#F0F0F2",
    "text_secondary": "#9B9DA6",
    "text_tertiary": "#5A5C65",
    "text_disabled": "#3A3C44",
    "border_subtle": "#2E3038",
    "border_default": "#3D3F48",
    "border_strong": "#5A5C65",
    "status_good_bg": "rgba(26, 92, 58, 0.20)",
    "status_good_text": "#6EE7B7",
    "status_mua_bg": "rgba(186, 117, 23, 0.20)",
    "status_mua_text": "#F0C060",
    "status_noise_bg": "rgba(163, 45, 45, 0.20)",
    "status_noise_text": "#F08080",
    "status_unsort_bg": "rgba(46, 109, 212, 0.20)",
    "status_unsort_text": "#7EB8F7",
    "selection_bg": "rgba(46, 109, 212, 0.18)",
    "selection_bg_strong": "rgba(46, 109, 212, 0.25)",
    "plot_line": "#00E6A0",
    "plot_scatter": "#33B5E5",
    "plot_shadow": "#2E6DD4",
    "plot_mean": "#00E6A0",
    "plot_peak": "#FFEB3B",
    "plot_highlight": "#00FFFF",
    "plot_acg": "#9B59B6",
    "plot_isi": "#33B5E5",
    "plot_fr": "#FFEB3B",
    "plot_overlay": "#00FF00",
    "plot_compare": "#FF9800",
    "plot_waveform_shadow": "#00331F",
}


LIGHT_COLORS = {
    "bg_base": "#F0F2F5",
    "bg_panel": "#FFFFFF",
    "bg_surface": "#F8F9FA",
    "bg_elevated": "#E9ECEF",
    "accent": "#2E6DD4",
    "accent_hover": "#1A4A9E",
    "accent_positive": "#DFF8EC",
    "accent_pos_text": "#065F46",
    "text_primary": "#111214",
    "text_secondary": "#495057",
    "text_tertiary": "#6C757D",
    "text_disabled": "#ADB5BD",
    "border_subtle": "#DEE2E6",
    "border_default": "#CED4DA",
    "border_strong": "#ADB5BD",
    "status_good_bg": "rgba(110, 231, 183, 0.25)",
    "status_good_text": "#065F46",
    "status_mua_bg": "rgba(240, 192, 96, 0.25)",
    "status_mua_text": "#92400E",
    "status_noise_bg": "rgba(240, 128, 128, 0.25)",
    "status_noise_text": "#991B1B",
    "status_unsort_bg": "rgba(46, 109, 212, 0.18)",
    "status_unsort_text": "#1E40AF",
    "selection_bg": "rgba(46, 109, 212, 0.16)",
    "selection_bg_strong": "rgba(46, 109, 212, 0.24)",
    "plot_line": "#007F5F",
    "plot_scatter": "#006D9C",
    "plot_shadow": "#2E6DD4",
    "plot_mean": "#007F5F",
    "plot_peak": "#9A6700",
    "plot_highlight": "#007A85",
    "plot_acg": "#6F2DA8",
    "plot_isi": "#006D9C",
    "plot_fr": "#9A6700",
    "plot_overlay": "#157A2F",
    "plot_compare": "#B45309",
    "plot_waveform_shadow": "#8ED9C0",
}


PANEL_PADDING = 8
CTRL_SPACING = 6
ROW_HEIGHT = 28


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
