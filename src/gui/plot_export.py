"""Export a tree group's plots to image files.

Reached by right-clicking a folder in the tree. Two shapes of output, because
they answer different questions:

*Per-cell files* — one image per cell in the group, named
``<Group>_c<cluster>_v<vision>.<ext>``. This is the contact sheet: you flip
through it looking for the one cell that does not belong.

*Group mean overlay* — a single SVG holding every cell's trace plus their
mean, with a legend naming each ID. This is the type summary: it shows whether
the group has one shared response or two populations that were merged.

Both IDs go in every filename deliberately. Vision ID = cluster_id + 1 outside
vision-only mode, and a folder of files carrying just one of them is a folder
that silently means something different depending on which one it was — the
kind of error that survives all the way into a figure. See CLAUDE.md.

Figures are rendered white-on-black-text rather than in the app's dark theme:
these are export products headed for a paper or a slide, not screen widgets.
SVG text is saved as text (``svg.fonttype='none'``) so it stays editable in
Illustrator or Inkscape.

Rendering runs on a worker thread. It is safe there because nothing here
touches a Qt canvas — figures are built with the bare ``Figure`` class and
handed straight to an Agg or SVG canvas — and because LazySTADict already
hands out one reader per thread.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from ..analysis import analysis_core
from . import recent_paths
from .panels.rf_map_widget import collect_rf_ellipses

logger = logging.getLogger(__name__)

__all__ = ["open_plot_export_dialog"]


# ── plot kinds ───────────────────────────────────────────────────────────────
# Registry order is menu order. `per_cell` and `group_mean` gate which mode a
# kind appears in: an STA is an image with no meaningful mean, and a mosaic is
# only a statement about a whole group, so neither exists in both modes.

class PlotKind:
    def __init__(
        self,
        key,
        label,
        ext,
        per_cell=True,
        group_mean=True,
        x_label="",
        y_label="",
        requires="",
    ):
        self.key = key
        self.label = label
        self.ext = ext
        self.per_cell = per_cell
        self.group_mean = group_mean
        self.x_label = x_label
        self.y_label = y_label
        # Human-readable name of the data this kind needs, for the "why is
        # this greyed out" tooltip.
        self.requires = requires


PLOT_KINDS = [
    PlotKind(
        "sta",
        "STA (spatial receptive field)",
        "png",
        group_mean=False,
        requires="Vision STA data",
    ),
    PlotKind(
        "timecourse",
        "Time course (temporal STA)",
        "svg",
        x_label="Time before spike (ms)",
        y_label="Normalised response",
        requires="Vision STA data",
    ),
    PlotKind(
        "acg",
        "Autocorrelogram",
        "svg",
        x_label="Lag (ms)",
        y_label="Coincidences per spike",
        requires="spike times",
    ),
    PlotKind(
        "chirp",
        "Chirp PSTH",
        "svg",
        x_label="Time (s)",
        y_label="Firing rate (Hz)",
        requires="a chirp analysis file",
    ),
    PlotKind(
        "grating",
        "Grating direction tuning",
        "svg",
        x_label="Direction (°)",
        y_label="Normalised response",
        requires="grating data",
    ),
    PlotKind(
        "mosaic",
        "RF mosaic",
        "svg",
        per_cell=False,
        x_label="x (stixels)",
        y_label="y (stixels)",
        requires="Vision RF fits",
    ),
]

PLOT_KINDS_BY_KEY = {k.key: k for k in PLOT_KINDS}

# Matplotlib defaults for everything drawn here. Deliberately not the app
# theme — see the module docstring.
#
# This has to name every key it cares about rather than just the obvious few.
# panels/feature_extraction.py updates the GLOBAL mpl.rcParams with the dark
# palette at import time, and main_window imports it, so by the time an export
# runs the ambient defaults are dark-on-dark with a grid enabled. Anything left
# unset here inherits that and lands in the file — which is how the first
# version produced white figures with pale grey titles and a black grid.
_EXPORT_RC = {
    "figure.facecolor": "white",
    "figure.edgecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "black",
    "axes.titlecolor": "black",
    "axes.titleweight": "normal",
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "axes.grid": False,
    "text.color": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.labelcolor": "black",
    "font.size": 9,
    # Don't depend on the app's display font being installed wherever these
    # figures are opened next.
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    # Keep SVG text as text so labels stay editable downstream.
    "svg.fonttype": "none",
}

_CH_COLORS = ["#c23b3b", "#2f9e44", "#3b6fc2"]  # R, G, B
_CH_NAMES = ["Red", "Green", "Blue"]
_MEAN_COLOR = "#111111"


def kind_available(dm, key) -> bool:
    """Whether *dm* holds the data this kind needs, before anything is drawn.

    Cheap checks only — this gates checkboxes as the dialog opens, so it must
    not read the .sta file or compute anything.
    """
    if dm is None:
        return False
    if key in ("sta", "timecourse", "mosaic"):
        return bool(getattr(dm, "vision_stas", None))
    if key == "acg":
        return True
    if key == "chirp":
        if hasattr(dm, "effective_chirp_available"):
            return bool(dm.effective_chirp_available())
        return bool(getattr(dm, "chirp_available", False))
    if key == "grating":
        return getattr(dm, "grating_status", None) in ("ok", "raw_only")
    return False


# ── filesystem naming ────────────────────────────────────────────────────────

_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_name(text: str) -> str:
    """A group name that survives being a file or folder name on Windows.

    Group names come from the user and routinely contain '/' (they read as
    paths in the classification format), which would otherwise silently
    scatter the export across directories.
    """
    cleaned = _UNSAFE.sub("_", str(text or "group")).strip(" .")
    return cleaned or "group"


def cell_stem(group_name: str, cluster_id: int, vision_id: int) -> str:
    """``<Group>_c<cluster>_v<vision>`` — the filename minus its extension."""
    return f"{safe_name(group_name)}_c{int(cluster_id)}_v{int(vision_id)}"


# ── data access ──────────────────────────────────────────────────────────────
# Every fetcher returns None when the cell simply has no such data. That is an
# ordinary outcome — most runs are one protocol, so most cells lack most of
# these — and the caller counts them rather than failing the export.


def _vision_id(dm, cluster_id):
    return int(dm.get_vision_id_for_cluster(int(cluster_id)))


def _fetch_sta(dm, cluster_id):
    """(sta_data, stafit) for a cell, or None."""
    stas = getattr(dm, "vision_stas", None)
    if not stas:
        return None
    vid = _vision_id(dm, cluster_id)
    if vid not in stas:
        return None
    sta_data = stas[vid]
    # LazySTADict returns None for a corrupt byte offset or a timed-out read;
    # its documented contract is that callers check for .red.
    if not hasattr(sta_data, "red") or sta_data.red is None:
        return None
    try:
        stafit = dm.vision_params.get_stafit_for_cell(vid)
    except Exception:
        logger.debug("get_stafit_for_cell failed for vision %s", vid, exc_info=True)
        stafit = None
    return sta_data, stafit


def _fetch_timecourse(dm, cluster_id):
    """The annotated per-cell temporal filter, straight from the STA SSOT.

    Returns compute_sta_metrics' ``_raw_temporal`` block — the same numbers
    the STA panel plots, so an exported time course and the one on screen can
    never disagree.
    """
    payload = _fetch_sta(dm, cluster_id)
    if payload is None:
        return None
    sta_data, stafit = payload
    try:
        metrics = analysis_core.compute_sta_metrics(
            sta_data, stafit, dm.vision_params, _vision_id(dm, cluster_id)
        )
    except Exception:
        logger.debug("compute_sta_metrics failed for %s", cluster_id, exc_info=True)
        return None
    return (metrics or {}).get("_raw_temporal")


def _fetch_acg(dm, cluster_id):
    lags, acg = dm.get_acg_data(int(cluster_id))
    if lags is None or acg is None:
        return None
    lags = np.asarray(lags, dtype=float)
    acg = np.asarray(acg, dtype=float)
    if lags.size < 2 or lags.size != acg.size:
        return None
    return lags, acg


def _fetch_chirp(dm, cluster_id):
    entry = dm.get_chirp_data_for_cluster(int(cluster_id))
    if not entry:
        return None
    psth = np.asarray(entry.get("psth_mean"), dtype=float)
    bin_ms = float(entry.get("bin_size_ms") or 0.0)
    if psth.ndim != 1 or psth.size < 2 or bin_ms <= 0:
        return None
    t = np.arange(psth.size, dtype=float) * bin_ms / 1000.0
    return t, psth, entry


def _fetch_grating(dm, cluster_id):
    entry = dm.get_grating_data_for_cluster(int(cluster_id))
    if not entry:
        return None
    from ..analysis import grating_calc

    curve = grating_calc.pooled_direction_tuning_curve(entry)
    if curve is None:
        return None
    curve = np.asarray(curve, dtype=float)
    angles = np.linspace(0, 360, curve.size, endpoint=False)
    return angles, curve


def _sta_refresh_ms(dm, cluster_ids):
    """Frame interval of this run's STA, read from the first cell that has one.

    One .sta read for the whole export. Returns None when there is no STA at
    all, in which case the time-course x axis stays in frames rather than
    inventing a millisecond scale.
    """
    for cid in cluster_ids:
        payload = _fetch_sta(dm, cid)
        if payload is None:
            continue
        refresh = getattr(payload[0], "refresh_time", None)
        try:
            refresh = float(refresh)
        except (TypeError, ValueError):
            return None
        return refresh if refresh > 0 else None
    return None


def group_trace(dm, key, cluster_id, refresh_ms=None):
    """(x, y) for one cell's contribution to a group-mean overlay, or None.

    The time course deliberately comes from ``get_cell_physics`` rather than
    from the STA read the per-cell plot uses: that array — dominant channel,
    peak-normalised — is the one ``build_feature_matrix`` actually clusters
    on. A group-mean overlay exists to answer "do these cells share the
    response that grouped them", so it has to plot the quantity that grouped
    them, not a cosmetically similar cousin.
    """
    if key == "timecourse":
        tc = (dm.get_cell_physics(int(cluster_id)) or {}).get("timecourse")
        if tc is None:
            return None
        tc = np.asarray(tc, dtype=float)
        if tc.ndim != 1 or tc.size < 2:
            return None
        # get_cell_physics stores the filter forwards in frames; the x axis is
        # time *before* the spike, so frame 0 is the oldest sample.
        idx = np.arange(tc.size, dtype=float) - (tc.size - 1)
        return (idx * refresh_ms if refresh_ms else idx), tc

    if key == "acg":
        return _fetch_acg(dm, cluster_id)

    if key == "chirp":
        got = _fetch_chirp(dm, cluster_id)
        return (got[0], got[1]) if got else None

    if key == "grating":
        return _fetch_grating(dm, cluster_id)

    return None


# ── figure construction ──────────────────────────────────────────────────────


def _new_figure(width=4.2, height=3.0):
    fig = Figure(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor("white")
    return fig


def _style(ax, kind=None, title=None):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if kind is not None:
        if kind.x_label:
            ax.set_xlabel(kind.x_label)
        if kind.y_label:
            ax.set_ylabel(kind.y_label)
    if title:
        ax.set_title(title, fontsize=9)


def _save(fig, path: Path):
    """Write *fig* to *path*, choosing the canvas from the extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = (
        FigureCanvasAgg(fig)
        if path.suffix.lower() == ".png"
        else FigureCanvasSVG(fig)
    )
    fig.set_canvas(canvas)
    fig.savefig(str(path), format=path.suffix.lower().lstrip("."), bbox_inches="tight")


def _cell_title(group_name, cluster_id, vision_id, extra=""):
    base = f"{group_name} · cluster {cluster_id} (vision {vision_id})"
    return f"{base}\n{extra}" if extra else base


# --- per-cell figures ---


def figure_sta(dm, cluster_id, group_name):
    """Peak-energy STA frame with the Gaussian fit drawn over it."""
    payload = _fetch_sta(dm, cluster_id)
    if payload is None:
        return None
    sta_data, stafit = payload

    r, g, b = sta_data.red, sta_data.green, sta_data.blue
    # Same frame the STA panel parks its slider on: the one carrying the most
    # signal across all three channels.
    energies = np.max(np.abs(np.stack([r, g, b])), axis=(0, 1, 2))
    frame_idx = int(np.argmax(energies))

    frame = np.stack(
        [r[:, :, frame_idx], g[:, :, frame_idx], b[:, :, frame_idx]], axis=-1
    ).astype(np.float32)
    mn, mx = float(frame.min()), float(frame.max())
    if mx > mn:
        frame = (frame - mn) / (mx - mn)
    else:
        frame = np.zeros_like(frame)

    fig = _new_figure(3.6, 3.4)
    ax = fig.add_subplot(111)
    # origin='upper' puts array row 0 at the top, which is the frame the
    # stafit centre coordinates are expressed in — so the ellipse needs no
    # flip here, unlike the pyqtgraph panel with its inverted ViewBox.
    ax.imshow(frame, origin="upper", interpolation="nearest")

    if stafit is not None:
        # STAFit.rot is radians (visionloader.STAFit), and get_stafit_for_cell
        # has already flipped center_y into array-row space — the same frame
        # get_sta_timecourse_data indexes with, and the frame imshow draws in
        # under origin='upper'. Only the rotation needs adjusting: y runs
        # downward here, so a positive mathematical angle reads as clockwise.
        angle_deg = np.degrees(float(getattr(stafit, "rot", 0.0) or 0.0))
        ax.add_patch(
            Ellipse(
                (float(stafit.center_x), float(stafit.center_y)),
                width=2.0 * float(getattr(stafit, "std_x", 1.0)),
                height=2.0 * float(getattr(stafit, "std_y", 1.0)),
                angle=-angle_deg,
                fill=False,
                edgecolor="#f0c040",
                linewidth=1.4,
            )
        )
        ax.plot(
            [float(stafit.center_x)], [float(stafit.center_y)],
            marker="+", color="#f0c040", markersize=6, markeredgewidth=1.2,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        _cell_title(
            group_name, cluster_id, _vision_id(dm, cluster_id),
            extra=f"STA frame {frame_idx + 1}/{r.shape[2]}",
        ),
        fontsize=8,
    )
    fig.tight_layout(pad=0.4)
    return fig


def figure_timecourse(dm, cluster_id, group_name):
    """All three colour channels, dominant one emphasised."""
    raw = _fetch_timecourse(dm, cluster_id)
    if raw is None:
        return None

    time_axis = np.asarray(raw["time_axis"], dtype=float)
    raw_tc = np.asarray(raw["raw_tc"], dtype=float)
    norm_trace = np.asarray(raw["norm_trace"], dtype=float)
    dom_idx = int(raw["dom_idx"])

    fig = _new_figure()
    ax = fig.add_subplot(111)
    _style(ax, PLOT_KINDS_BY_KEY["timecourse"])

    if raw_tc.ndim == 2 and raw_tc.shape[1] == 3:
        abs_max = float(np.max(np.abs(raw_tc)))
        scaled = raw_tc / abs_max if abs_max > 0 else raw_tc
        for i in range(3):
            if i == dom_idx:
                continue
            ax.plot(
                time_axis, scaled[:, i],
                color=_CH_COLORS[i], linewidth=0.9, alpha=0.45,
                label=_CH_NAMES[i],
            )

    ax.plot(
        time_axis, norm_trace,
        color=_CH_COLORS[dom_idx], linewidth=2.0,
        label=f"{_CH_NAMES[dom_idx]} (dominant)",
    )
    ax.axhline(0, color="#999999", linestyle=":", linewidth=0.8)
    ax.axvline(0, color="#555555", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.legend(fontsize=6, frameon=False, loc="best")

    polarity = "OFF" if raw.get("is_off") else "ON"
    ax.set_title(
        _cell_title(
            group_name, cluster_id, _vision_id(dm, cluster_id),
            extra=f"{polarity} · peak {raw.get('peak_ms', float('nan')):.0f} ms",
        ),
        fontsize=8,
    )
    fig.tight_layout(pad=0.4)
    return fig


def figure_acg(dm, cluster_id, group_name):
    got = _fetch_acg(dm, cluster_id)
    if got is None:
        return None
    lags, acg = got

    fig = _new_figure()
    ax = fig.add_subplot(111)
    _style(ax, PLOT_KINDS_BY_KEY["acg"])
    ax.fill_between(lags, acg, 0, color="#4f8ef7", alpha=0.35, linewidth=0)
    ax.plot(lags, acg, color="#2a5fb0", linewidth=1.0)
    ax.set_xlim(lags[0], lags[-1])
    ax.set_title(
        _cell_title(group_name, cluster_id, _vision_id(dm, cluster_id)), fontsize=8
    )
    fig.tight_layout(pad=0.4)
    return fig


# Same phase tints the chirp dashboard and the UMAP overlay use, so all three
# read alike.
_CHIRP_PHASE_TINTS = {
    "phase_step_on": ("#787878", 0.16),
    "phase_step_off": ("#787878", 0.16),
    "phase_freq_sweep": ("#4f8ddb", 0.14),
    "phase_contrast": ("#db8d4f", 0.14),
}


def _shade_chirp_phases(ax, entry):
    bin_ms = float(entry.get("bin_size_ms") or 0.0)
    if bin_ms <= 0:
        return
    for phase, (color, alpha) in _CHIRP_PHASE_TINTS.items():
        bounds = entry.get(phase)
        if bounds is None or len(bounds) != 2:
            continue
        start, end = float(bounds[0]), float(bounds[1])
        if end > start:
            ax.axvspan(
                start * bin_ms / 1000.0, end * bin_ms / 1000.0,
                color=color, alpha=alpha, linewidth=0, zorder=0,
            )


def figure_chirp(dm, cluster_id, group_name):
    got = _fetch_chirp(dm, cluster_id)
    if got is None:
        return None
    t, psth, entry = got

    fig = _new_figure(5.2, 2.6)
    ax = fig.add_subplot(111)
    _style(ax, PLOT_KINDS_BY_KEY["chirp"])
    _shade_chirp_phases(ax, entry)
    ax.plot(t, psth, color="#2a5fb0", linewidth=0.9)
    ax.set_xlim(float(t[0]), float(t[-1]))

    qi = entry.get("quality_index")
    extra = "" if qi is None or not np.isfinite(qi) else f"QI {qi:.3f}"
    ax.set_title(
        _cell_title(group_name, cluster_id, _vision_id(dm, cluster_id), extra=extra),
        fontsize=8,
    )
    fig.tight_layout(pad=0.4)
    return fig


def figure_grating(dm, cluster_id, group_name):
    got = _fetch_grating(dm, cluster_id)
    if got is None:
        return None
    angles, curve = got

    fig = _new_figure()
    ax = fig.add_subplot(111)
    _style(ax, PLOT_KINDS_BY_KEY["grating"])
    # Close the circle so the curve reads as the periodic thing it is.
    ax.plot(
        np.append(angles, 360.0), np.append(curve, curve[0]),
        color="#2a5fb0", linewidth=1.4, marker="o", markersize=3,
    )
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_title(
        _cell_title(group_name, cluster_id, _vision_id(dm, cluster_id)), fontsize=8
    )
    fig.tight_layout(pad=0.4)
    return fig


PER_CELL_FIGURES = {
    "sta": figure_sta,
    "timecourse": figure_timecourse,
    "acg": figure_acg,
    "chirp": figure_chirp,
    "grating": figure_grating,
}


# --- group figures ---


def _legend_columns(n):
    """Keep a per-cell legend readable as the group grows."""
    if n <= 12:
        return 1
    if n <= 30:
        return 2
    return 3


def figure_group_overlay(dm, key, cluster_ids, group_name, refresh_ms=None):
    """Every cell's trace plus their mean, one colour and legend entry each.

    Returns ``(figure, n_used, n_skipped)``. Traces whose length disagrees
    with the majority are dropped rather than resampled: within one run they
    should all share a grid, so a mismatch means the cell's data came from
    somewhere else (a bridged reference run, a different chirp file) and
    averaging it in would quietly fabricate a trace.
    """
    kind = PLOT_KINDS_BY_KEY[key]

    collected = []
    for cid in cluster_ids:
        try:
            got = group_trace(dm, key, cid, refresh_ms=refresh_ms)
        except Exception:
            logger.debug("group_trace failed for %s/%s", key, cid, exc_info=True)
            got = None
        if got is None:
            continue
        x, y = np.asarray(got[0], dtype=float), np.asarray(got[1], dtype=float)
        if x.size == y.size and x.size >= 2 and np.all(np.isfinite(y)):
            collected.append((int(cid), x, y))

    if not collected:
        return None, 0, len(cluster_ids)

    lengths = [len(y) for _, _, y in collected]
    modal = max(set(lengths), key=lengths.count)
    usable = [entry for entry in collected if len(entry[2]) == modal]
    n_skipped = len(cluster_ids) - len(usable)

    x = usable[0][1]
    stack = np.vstack([y for _, _, y in usable])
    mean = stack.mean(axis=0)

    fig = _new_figure(6.4 if key == "chirp" else 5.4, 3.4)
    ax = fig.add_subplot(111)
    _style(ax, kind)

    if key == "chirp":
        entry = dm.get_chirp_data_for_cluster(usable[0][0])
        if entry:
            _shade_chirp_phases(ax, entry)

    cmap = matplotlib.colormaps["viridis"]
    denom = max(len(usable) - 1, 1)
    for i, (cid, _xi, y) in enumerate(usable):
        ax.plot(
            x, y,
            color=cmap(i / denom), linewidth=0.7, alpha=0.55,
            label=f"c{cid} / v{_vision_id(dm, cid)}",
            zorder=2,
        )

    ax.plot(
        x, mean,
        color=_MEAN_COLOR, linewidth=2.2, zorder=3,
        label=f"mean (n={len(usable)})",
    )

    if key == "timecourse":
        ax.axhline(0, color="#999999", linestyle=":", linewidth=0.8, zorder=1)
        ax.axvline(0, color="#555555", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)
        if refresh_ms is None:
            ax.set_xlabel("Frames before spike")
    if key == "grating":
        ax.set_xlim(0, 330)
        ax.set_xticks([0, 90, 180, 270])
    if key in ("acg", "chirp"):
        ax.set_xlim(float(x[0]), float(x[-1]))

    ax.set_title(f"{group_name} — {kind.label}", fontsize=9)
    # Outside the axes: with 40 traces an inside legend covers the data it is
    # supposed to explain.
    ax.legend(
        fontsize=5.5,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=_legend_columns(len(usable)),
        handlelength=1.2,
        borderaxespad=0.0,
    )
    fig.tight_layout(pad=0.4)
    return fig, len(usable), n_skipped


def figure_mosaic(dm, cluster_ids, group_name):
    """The tiling check: every RF in the group, each labelled with its IDs.

    Returns ``(figure, n_used, n_skipped)``. Cells with no usable Gaussian fit
    are absent from collect_rf_ellipses and counted as skipped, which is the
    honest reading — an RF that was never fit is not an RF that does not
    overlap.
    """
    ellipses = collect_rf_ellipses(dm, list(cluster_ids))
    ordered = [c for c in cluster_ids if int(c) in ellipses]
    n_skipped = len(cluster_ids) - len(ordered)
    if not ordered:
        return None, 0, len(cluster_ids)

    geom = np.array([ellipses[int(c)] for c in ordered], dtype=float)

    fig = _new_figure(5.6, 4.4)
    ax = fig.add_subplot(111)
    _style(ax, PLOT_KINDS_BY_KEY["mosaic"])

    cmap = matplotlib.colormaps["viridis"]
    denom = max(len(ordered) - 1, 1)
    for i, cid in enumerate(ordered):
        cx, cy, w, h, angle = geom[i]
        color = cmap(i / denom)
        ax.add_patch(
            Ellipse(
                (cx, cy), width=w, height=h, angle=angle,
                fill=False, edgecolor=color, linewidth=1.0,
            )
        )
        ax.annotate(
            f"c{int(cid)}/v{_vision_id(dm, cid)}",
            (cx, cy), fontsize=4.5, color=color, ha="center", va="center",
        )

    cx, cy = geom[:, 0], geom[:, 1]
    rx, ry = geom[:, 2] / 2.0, geom[:, 3] / 2.0
    x_lo, x_hi = float(np.min(cx - rx)), float(np.max(cx + rx))
    y_lo, y_hi = float(np.min(cy - ry)), float(np.max(cy + ry))
    mx = max((x_hi - x_lo) * 0.05, 5.0)
    my = max((y_hi - y_lo) * 0.05, 5.0)
    ax.set_xlim(x_lo - mx, x_hi + mx)
    ax.set_ylim(y_lo - my, y_hi + my)
    ax.set_aspect("equal", adjustable="box")

    subtitle = f"{len(ordered)} RF fits"
    if n_skipped:
        subtitle += f" · {n_skipped} cell{'s' if n_skipped != 1 else ''} without a fit"
    ax.set_title(f"{group_name} — RF mosaic\n{subtitle}", fontsize=9)
    fig.tight_layout(pad=0.4)
    return fig, len(ordered), n_skipped


# ── worker ───────────────────────────────────────────────────────────────────


class PlotExportWorker(QThread):
    """Renders and writes every requested file off the GUI thread.

    Per-cell mode is the slow one — an STA read per cell, and an ACG compute
    for any cell the background warm-up has not reached yet — so progress is
    reported per file and cancellation is checked between them.
    """

    progress = Signal(int, int, str)  # done, total, current label
    done = Signal(dict)  # summary
    failed = Signal(str)

    def __init__(self, dm, mode, keys, cluster_ids, group_name, out_dir):
        super().__init__()
        self.dm = dm
        self.mode = mode  # "per_cell" | "group_mean"
        self.keys = list(keys)
        self.cluster_ids = [int(c) for c in cluster_ids]
        self.group_name = group_name
        self.out_dir = Path(out_dir)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            with matplotlib.rc_context(_EXPORT_RC):
                summary = (
                    self._run_per_cell()
                    if self.mode == "per_cell"
                    else self._run_group_mean()
                )
            if not self._cancelled:
                self.done.emit(summary)
            else:
                summary["cancelled"] = True
                self.done.emit(summary)
        except Exception as exc:
            logger.exception("Plot export failed")
            self.failed.emit(str(exc))

    def _run_per_cell(self):
        written, skipped, paths = 0, 0, []
        total = len(self.keys) * len(self.cluster_ids)
        step = 0
        safe_group = safe_name(self.group_name)

        for key in self.keys:
            kind = PLOT_KINDS_BY_KEY[key]
            # One directory per plot type, so the filename itself can stay
            # exactly <Group>_c<n>_v<m> even when several types are exported
            # together.
            kind_dir = self.out_dir / f"{safe_group}_{key}"
            for cid in self.cluster_ids:
                if self._cancelled:
                    return self._summary(written, skipped, paths)
                step += 1
                self.progress.emit(step, total, f"{kind.label} — cluster {cid}")
                try:
                    fig = PER_CELL_FIGURES[key](self.dm, cid, self.group_name)
                except Exception:
                    logger.debug(
                        "figure for %s/%s failed", key, cid, exc_info=True
                    )
                    fig = None
                if fig is None:
                    skipped += 1
                    continue
                path = kind_dir / (
                    cell_stem(self.group_name, cid, _vision_id(self.dm, cid))
                    + f".{kind.ext}"
                )
                _save(fig, path)
                written += 1
                if len(paths) < 4:
                    paths.append(path)

        return self._summary(written, skipped, paths)

    def _run_group_mean(self):
        written, skipped, paths = 0, 0, []
        total = len(self.keys)
        # Only paid for when a time course is actually being exported.
        refresh_ms = (
            _sta_refresh_ms(self.dm, self.cluster_ids)
            if "timecourse" in self.keys
            else None
        )

        for step, key in enumerate(self.keys, start=1):
            if self._cancelled:
                break
            kind = PLOT_KINDS_BY_KEY[key]
            self.progress.emit(step, total, kind.label)
            if key == "mosaic":
                fig, n_used, n_skipped = figure_mosaic(
                    self.dm, self.cluster_ids, self.group_name
                )
            else:
                fig, n_used, n_skipped = figure_group_overlay(
                    self.dm, key, self.cluster_ids, self.group_name,
                    refresh_ms=refresh_ms,
                )
            if fig is None:
                skipped += 1
                continue
            path = self.out_dir / f"{safe_name(self.group_name)}_{key}_mean.svg"
            _save(fig, path)
            written += 1
            paths.append(path)

        return self._summary(written, skipped, paths)

    def _summary(self, written, skipped, paths):
        return {
            "written": written,
            "skipped": skipped,
            "paths": [str(p) for p in paths],
            "out_dir": str(self.out_dir),
            "cancelled": self._cancelled,
        }


# ── dialog ───────────────────────────────────────────────────────────────────


class PlotExportDialog(QDialog):
    """Pick a mode, the plot types, and where the files go."""

    def __init__(self, main_window, group_name, cluster_ids, mode="per_cell"):
        super().__init__(main_window)
        self.main_window = main_window
        self.group_name = group_name
        self.cluster_ids = list(cluster_ids)
        self.worker = None
        self.progress_dialog = None

        self.setWindowTitle(f"Export plots — {group_name}")
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)

        n = len(self.cluster_ids)
        layout.addWidget(
            QLabel(f"<b>{group_name}</b> — {n} cell{'s' if n != 1 else ''}")
        )

        # ── mode ──
        mode_box = QGroupBox("What to export")
        mode_layout = QVBoxLayout(mode_box)
        self.per_cell_radio = QRadioButton("One file per cell")
        self.per_cell_radio.setToolTip(
            "A folder per plot type, each holding one file per cell named "
            "<Group>_c<cluster>_v<vision>."
        )
        self.group_radio = QRadioButton("Group mean, with every cell overlaid")
        self.group_radio.setToolTip(
            "One SVG per plot type: each cell's trace plus the group mean, "
            "with a legend naming every ID."
        )
        group = QButtonGroup(self)
        group.addButton(self.per_cell_radio)
        group.addButton(self.group_radio)
        (self.group_radio if mode == "group_mean" else self.per_cell_radio).setChecked(
            True
        )
        self.per_cell_radio.toggled.connect(self._refresh_kind_boxes)
        mode_layout.addWidget(self.per_cell_radio)
        mode_layout.addWidget(self.group_radio)
        layout.addWidget(mode_box)

        # ── plot types ──
        self.kinds_box = QGroupBox("Plot types")
        kinds_layout = QVBoxLayout(self.kinds_box)
        dm = getattr(main_window, "data_manager", None)
        self.kind_boxes = {}
        for kind in PLOT_KINDS:
            box = QCheckBox(f"{kind.label}  ({kind.ext.upper()})")
            available = kind_available(dm, kind.key)
            box.setEnabled(available)
            if not available:
                box.setToolTip(f"This dataset has no {kind.requires}.")
            self.kind_boxes[kind.key] = box
            kinds_layout.addWidget(box)
        layout.addWidget(self.kinds_box)

        # ── destination ──
        dest_box = QGroupBox("Destination folder")
        dest_layout = QHBoxLayout(dest_box)
        self.dest_edit = QLineEdit(recent_paths.last_dir(main_window, "plot_export"))
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        dest_layout.addWidget(self.dest_edit)
        dest_layout.addWidget(browse_btn)
        layout.addWidget(dest_box)

        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: palette(mid);")
        layout.addWidget(self.hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        buttons.button(QDialogButtonBox.Ok).setText("Export")
        buttons.accepted.connect(self._start)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._refresh_kind_boxes()

    # -- ui state --

    def _refresh_kind_boxes(self):
        """Grey out kinds the current mode has no meaning for.

        An STA has no group mean and a mosaic has no per-cell form, so rather
        than let the user tick something that would silently produce nothing,
        the box goes away with the reason attached.
        """
        per_cell = self.per_cell_radio.isChecked()
        dm = getattr(self.main_window, "data_manager", None)
        for kind in PLOT_KINDS:
            box = self.kind_boxes[kind.key]
            supported = kind.per_cell if per_cell else kind.group_mean
            available = kind_available(dm, kind.key)
            box.setEnabled(supported and available)
            if not available:
                box.setChecked(False)
                box.setToolTip(f"This dataset has no {kind.requires}.")
            elif not supported:
                box.setChecked(False)
                box.setToolTip(
                    "No per-cell form — this is a whole-group plot."
                    if per_cell
                    else "No group mean — this is a per-cell image."
                )
            else:
                box.setToolTip("")

        stem = cell_stem(self.group_name, 0, 1).replace("_c0_v1", "")
        if per_cell:
            self.hint.setText(
                f"Writes <i>{stem}_&lt;type&gt;/</i> folders, one file per cell named "
                f"<i>{stem}_c&lt;cluster&gt;_v&lt;vision&gt;</i>."
            )
        else:
            self.hint.setText(
                f"Writes one SVG per plot type, named "
                f"<i>{stem}_&lt;type&gt;_mean.svg</i>."
            )

    def _browse(self):
        chosen = QFileDialog.getExistingDirectory(
            self, "Export plots to", self.dest_edit.text() or ""
        )
        if chosen:
            self.dest_edit.setText(chosen)

    def selected_keys(self):
        return [
            k.key
            for k in PLOT_KINDS
            if self.kind_boxes[k.key].isEnabled() and self.kind_boxes[k.key].isChecked()
        ]

    # -- run --

    def _start(self):
        keys = self.selected_keys()
        if not keys:
            QMessageBox.warning(
                self, "Nothing selected", "Pick at least one plot type to export."
            )
            return

        out_dir = Path(self.dest_edit.text().strip() or ".")
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.critical(
                self, "Cannot write there", f"{out_dir}\n\n{exc}"
            )
            return
        recent_paths.remember_dir(out_dir, "plot_export")

        mode = "per_cell" if self.per_cell_radio.isChecked() else "group_mean"
        total = len(keys) * len(self.cluster_ids) if mode == "per_cell" else len(keys)

        self.progress_dialog = QProgressDialog(
            "Rendering…", "Cancel", 0, max(total, 1), self
        )
        self.progress_dialog.setWindowTitle("Exporting plots")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)

        self.worker = PlotExportWorker(
            self.main_window.data_manager,
            mode,
            keys,
            self.cluster_ids,
            self.group_name,
            out_dir,
        )
        self.progress_dialog.canceled.connect(self.worker.cancel)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, done, total, label):
        if self.progress_dialog is None:
            return
        self.progress_dialog.setMaximum(max(total, 1))
        self.progress_dialog.setValue(done)
        self.progress_dialog.setLabelText(label)

    def _close_progress(self):
        if self.progress_dialog is not None:
            self.progress_dialog.reset()
            self.progress_dialog = None

    def _on_done(self, summary):
        self._close_progress()
        self._finish_worker()

        written = summary.get("written", 0)
        skipped = summary.get("skipped", 0)
        lines = [f"Wrote {written} file{'s' if written != 1 else ''} to:",
                 summary.get("out_dir", "")]
        if skipped:
            lines.append(
                f"\n{skipped} skipped — no data of that type for those cells."
            )
        if summary.get("cancelled"):
            lines.append("\nCancelled before finishing.")

        if hasattr(self.main_window, "status_bar"):
            self.main_window.status_bar.showMessage(
                f"Exported {written} plot file{'s' if written != 1 else ''}"
                f"{f' ({skipped} skipped)' if skipped else ''}.",
                6000,
            )

        if written == 0 and not summary.get("cancelled"):
            QMessageBox.warning(
                self,
                "Nothing exported",
                "None of the selected cells had data for the chosen plot types.",
            )
            return

        QMessageBox.information(self, "Export complete", "\n".join(lines))
        self.accept()

    def _on_failed(self, message):
        self._close_progress()
        self._finish_worker()
        QMessageBox.critical(self, "Export failed", message)

    def _finish_worker(self):
        if self.worker is not None:
            self.worker.quit()
            self.worker.wait(2000)
            self.worker = None

    def reject(self):
        # A running render holds the DataManager; let it stop before the
        # dialog that owns it goes away.
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(2000)
        self._close_progress()
        self._finish_worker()
        super().reject()


def open_plot_export_dialog(main_window, group_item, mode="per_cell"):
    """Entry point for the tree context menu."""
    from .callbacks import iter_cluster_ids

    dm = getattr(main_window, "data_manager", None)
    if dm is None:
        return

    cluster_ids = iter_cluster_ids(group_item)
    if not cluster_ids:
        QMessageBox.information(
            main_window, "Empty group", "This folder has no cells to export."
        )
        return

    dlg = PlotExportDialog(main_window, group_item.text(), cluster_ids, mode=mode)
    dlg.exec()
