"""The selected cell, and pinned cells, drawn over their population (PLAN.md Q45).

The population pane (Ctrl+P) shows a group's time courses, ACGs and firing
rates as a blue wash with the mean in ink. This module lays the selected cell
on top in red, and up to PIN_LIMIT pinned cells (Ctrl+K) in yellow / ink with
distinct dashes, each named in a key at the top left. Only the overlay lines
change on a new selection; the group drawing and its caches stay as they are.

It runs in Tier 2 (after the selection settles), because a cell's time course
can come from a cold physics read.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PIN_LIMIT = 4
# Never ink: the group mean is ink. Yellow / red, told apart by the dash.
_PIN_STYLES = (("plot_peak", "-"), ("plot_peak", "--"), ("plot_compare", "--"), ("plot_peak", ":"))

# pane -> (canvas attribute, state attribute)
PANES = {
    "timecourse": ("pop_timecourse_canvas", "_timecourse_state"),
    "acg": ("pop_acg_canvas", "_acg_state"),
    "fr": ("pop_fr_canvas", "_fr_state"),
}


def cell_trace(dm, cluster_id, pane) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(x, y) of one cell as the given pane draws its population, or None."""
    try:
        if pane == "timecourse":
            tc = dm.get_cell_physics(cluster_id).get("timecourse")
            if tc is None:
                return None
            y = np.asarray(tc, dtype=float)
            return np.arange(len(y)), y
        if pane == "acg":
            lags, acg = dm.get_acg_data(cluster_id)
            if lags is None or acg is None:
                return None
            lags, acg = np.asarray(lags, dtype=float), np.asarray(acg, dtype=float)
            keep = (lags >= 0) & (lags <= 50)          # as draw_population_acg_panel
            return lags[keep], acg[keep]
        if pane == "fr":
            std = dm.get_standard_plot_data(cluster_id) or {}
            x, y = std.get("fr_bin_centers"), std.get("fr_rate")
            if x is None or y is None or len(x) != len(y):
                return None
            return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    except Exception:
        logger.debug("no %s trace for cell %s", pane, cluster_id, exc_info=True)
    return None


def _styles(colors, n_pins) -> List[Tuple[str, str, float]]:
    """(colour, linestyle, width) for the selected cell, then each pin."""
    out = [(colors.get("plot_compare", "#c0392b"), "-", 1.8)]
    for key, dash in _PIN_STYLES[:n_pins]:
        out.append((colors.get(key, "#d4a017"), dash, 1.4))
    return out


def _overlay_artists(state, ax, count, colors):
    """The pane's overlay lines and their key labels, made once per full redraw."""
    from ..theme import plot_field
    lines = state.setdefault("compare_lines", [])
    keys = state.setdefault("compare_keys", [])
    while len(lines) < count:
        (line,) = ax.plot([], [], zorder=6, solid_capstyle="round")
        lines.append(line)
        # A paper backing keeps the key readable over the traces.
        keys.append(ax.text(0.01, 0.97 - 0.1 * len(keys), "", transform=ax.transAxes,
                            ha="left", va="top", fontsize=8, zorder=7,
                            bbox=dict(boxstyle="round,pad=0.15", linewidth=0, alpha=0.85,
                                      facecolor=plot_field(colors))))
    return lines, keys


def update_overlays(main_window, selected_id, pinned_ids: Sequence[int] = ()) -> Dict[str, int]:
    """Draw the selected + pinned cells over each visible population pane.

    Returns {pane: cells drawn}. Panes with no population drawing are skipped.
    """
    from ..theme import resolve_theme_colors
    dm = getattr(main_window, "data_manager", None)
    drawn = {}
    if dm is None:
        return drawn
    colors = resolve_theme_colors(main_window.get_current_colors())
    pins = [int(c) for c in pinned_ids if c is not None and c != selected_id][:PIN_LIMIT]
    cells = ([int(selected_id)] if selected_id is not None else []) + pins
    styles = _styles(colors, len(pins))
    if selected_id is None:
        styles = styles[1:]
    for pane, (canvas_attr, state_attr) in PANES.items():
        canvas = getattr(main_window, canvas_attr, None)
        state = getattr(canvas, state_attr, None) if canvas is not None else None
        if not state or state.get("ax") not in canvas.fig.axes:
            continue
        ax = state["ax"]
        lines, keys = _overlay_artists(state, ax, len(cells), colors)
        n = 0
        for i, (line, key) in enumerate(zip(lines, keys)):
            trace = cell_trace(dm, cells[i], pane) if i < len(cells) else None
            if trace is None:
                line.set_visible(False)
                key.set_visible(False)
                continue
            x, y = trace
            if pane == "timecourse":                # the pane truncates to its shortest
                width = len(state["mean_line"].get_xdata())
                x, y = x[:width], y[:width]
            colour, dash, lw = styles[i]
            line.set_data(x, y)
            line.set_color(colour)
            line.set_linestyle(dash)
            line.set_linewidth(lw)
            line.set_visible(True)
            selected = i == 0 and selected_id is not None
            key.set_text(f"{'Cell' if selected else 'Pinned'} {cells[i]}")
            key.set_color(colour)
            key.set_position((0.01, 0.97 - 0.1 * n))
            key.set_visible(True)
            n += 1
        canvas.draw_idle()
        drawn[pane] = n
    return drawn


# --- pins ---------------------------------------------------------------------

def toggle_pin(main_window, cluster_id) -> List[int]:
    """Pin / unpin a cell; the oldest pin drops out past PIN_LIMIT."""
    pins = list(getattr(main_window, "_pinned_cells", []))
    cid = int(cluster_id)
    if cid in pins:
        pins.remove(cid)
    else:
        pins.append(cid)
        pins = pins[-PIN_LIMIT:]
    main_window._pinned_cells = pins
    return pins


def clear_pins(main_window) -> None:
    main_window._pinned_cells = []
