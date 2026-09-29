"""Spatial-frequency protocols on the Grating tab (PLAN.md Q65).

A grating run that varies the spatial period at one or two directions (the
"drifting grating for RF sizes" runs of 2026-05-14-0) has no direction
tuning to show. This view shows what it does measure:

* the response against spatial period, log axis, one curve per temporal
  frequency, ±1 SD across trials, the peak marked;
* every trial's spikes, grouped by spatial period (then temporal
  frequency), with the stimulus window shaded.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ..theme import apply_plot_theme, categorical, resolve_theme_colors


def sf_curves(data: dict) -> Dict[float, List[Tuple[float, float, float, int]]]:
    """{tf: [(spatial value, mean, sd, n), ...] sorted} from a cell's grating dict."""
    out: Dict[float, list] = {}
    for key, entry in data.items():
        if not isinstance(key, tuple) or not isinstance(entry, dict):
            continue
        if entry.get("condition_type") != "sf":
            continue
        bw, tf = float(key[0]), float(key[1])
        mean = np.asarray(entry.get("mean_response", [np.nan]), dtype=float)
        sd = np.asarray(entry.get("sd_response", [np.nan]) if entry.get("sd_response") is not None
                        else [np.nan], dtype=float)
        n = np.asarray(entry.get("n_trials", [0]))
        # One or two directions: average them (the protocol varies the period).
        out.setdefault(tf, []).append((bw, float(np.nanmean(mean)) if mean.size else np.nan,
                                       float(np.nanmean(sd)) if sd.size else np.nan,
                                       int(np.sum(n)) if n.size else 0))
    return {tf: sorted(v) for tf, v in sorted(out.items())}


def raster_rows(trials, trial_parameters) -> Tuple[List[np.ndarray], List[Tuple[float, float]], dict]:
    """Trials sorted by (spatial value, TF, order): spike times in s from onset, keys, timing."""
    order = sorted(range(len(trial_parameters)),
                   key=lambda i: (float(trial_parameters[i]["barWidth"]),
                                  float(trial_parameters[i]["temporalFrequency"]), i))
    rows, keys, pres, stims, tails = [], [], [], [], []
    for i in order:
        p = trial_parameters[i]
        pre = float(p.get("preTime", 0.0))
        pres.append(pre)
        stims.append(float(p.get("stimTime", 0.0)))
        tails.append(float(p.get("tailTime", 0.0) or 0.0))
        rows.append((np.asarray(trials[i], dtype=float) - pre) / 1000.0)
        keys.append((float(p["barWidth"]), float(p["temporalFrequency"])))
    timing = {"pre_s": min(pres) / 1000.0 if pres else 0.0,
              "stim_s": min(stims) / 1000.0 if stims else 0.0,
              "tail_s": min(tails) / 1000.0 if tails else 0.0}
    return rows, keys, timing


class SpatialTuningView(QWidget):
    def __init__(self, colors, parent=None):
        super().__init__(parent)
        self._colors = resolve_theme_colors(colors)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.stats = QLabel("")
        self.stats.setWordWrap(True)
        lay.addWidget(self.stats)
        row = QHBoxLayout()
        self.curve_plot = pg.PlotWidget()
        self.curve_plot.setLogMode(x=True, y=False)
        self.curve_plot.setMenuEnabled(False)
        self.curve_plot.addLegend(offset=(-10, 10))
        row.addWidget(self.curve_plot, 2)
        self.raster_plot = pg.PlotWidget()
        self.raster_plot.setMenuEnabled(False)
        self.raster_plot.invertY(True)
        self.raster_plot.setLabel("bottom", "Time from stimulus onset (s)")
        self.raster_plot.getViewBox().setMouseEnabled(x=True, y=True)
        row.addWidget(self.raster_plot, 3)
        lay.addLayout(row, 1)
        self.note = QLabel("")
        self.note.setObjectName("mutedLabel")
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        self.restyle(colors)

    def restyle(self, colors):
        self._colors = resolve_theme_colors(colors)
        for p in (self.curve_plot, self.raster_plot):
            apply_plot_theme(p, self._colors)

    def show_cell(self, data: dict, trials=None, trial_parameters=None, spatial=None,
                  response_label="Response (spikes/s)"):
        c = self._colors
        name, unit = spatial if spatial else ("bar width", "")
        curves = sf_curves(data)
        self.curve_plot.clear()
        legend = self.curve_plot.plotItem.legend
        if legend is not None:
            legend.clear()
        self.curve_plot.setLabel("bottom", f"Spatial {name}" + (f" ({unit})" if unit else ""))
        self.curve_plot.setLabel("left", response_label)
        best = None
        periods = sorted({p[0] for pts in curves.values() for p in pts if p[0] > 0})
        if periods:
            # Label the periods that ran, not log decades and their minor ticks.
            self.curve_plot.getAxis("bottom").setTicks(
                [[(float(np.log10(v)), f"{v:g}") for v in periods], []])
        for k, (tf, pts) in enumerate(curves.items()):
            colour = categorical(k if k % 12 != 2 else k + 1, c)
            x = np.array([p[0] for p in pts])
            y = np.array([p[1] for p in pts])
            sd = np.array([p[2] for p in pts])
            ok = np.isfinite(y) & (x > 0)
            if not ok.any():
                continue
            self.curve_plot.plot(x[ok], y[ok], pen=pg.mkPen(colour, width=2), symbol="o",
                                 symbolSize=6, symbolBrush=colour, name=f"{tf:.3g} Hz")
            good_sd = ok & np.isfinite(sd)
            if good_sd.any():
                err = pg.ErrorBarItem(x=np.log10(x[good_sd]), y=y[good_sd], height=2 * sd[good_sd],
                                      pen=pg.mkPen(colour, width=1))
                self.curve_plot.addItem(err)
            j = int(np.nanargmax(np.where(ok, y, -np.inf)))
            if best is None or y[j] > best[2]:
                best = (tf, x[j], y[j])
        if best is not None:
            tf, xb, yb = best
            self.curve_plot.plot([xb], [yb], pen=None, symbol="star", symbolSize=16,
                                 symbolBrush=c.get("plot_peak", "#E0B000"))
            per_tf = "; ".join(
                f"{tf_:.3g} Hz: {pts[int(np.nanargmax([p[1] if np.isfinite(p[1]) else -np.inf for p in pts]))][0]:g}{unit}"
                for tf_, pts in curves.items() if pts)
            self.stats.setText(f"[spatial tuning]  Peak {yb:.1f} at {name} {xb:g}{unit}, "
                               f"{tf:.3g} Hz.   Preferred {name} by temporal frequency: {per_tf}")
        else:
            self.stats.setText("[spatial tuning]  No response to any spatial period.")
        self._draw_rasters(trials, trial_parameters, name, unit)

    def _draw_rasters(self, trials, trial_parameters, name, unit):
        c = self._colors
        self.raster_plot.clear()
        if trials is None or not trial_parameters:
            self.note.setText("Rasters need the per-trial spikes (a raw grating file).")
            return
        rows, keys, timing = raster_rows(trials, trial_parameters)
        n = len(rows)
        pre, stim, tail = timing["pre_s"], timing["stim_s"], timing["tail_s"]
        shade = QColor(c.get("plot_highlight", "#4A82D6"))
        shade.setAlpha(34)
        region = pg.LinearRegionItem((0.0, stim), movable=False, brush=pg.mkBrush(shade),
                                     pen=pg.mkPen(None))
        region.setZValue(-10)
        self.raster_plot.addItem(region)
        tfs = sorted({k[1] for k in keys})
        colour_of = {tf: categorical(i if i % 12 != 2 else i + 1, c) for i, tf in enumerate(tfs)}
        for tf in tfs:
            xs, ys = [], []
            for r, (t, key) in enumerate(zip(rows, keys)):
                if key[1] != tf:
                    continue
                t = t[(t >= -pre) & (t < stim + tail)]
                xs.append(np.repeat(t, 2))
                ys.append(np.tile([r + 0.1, r + 0.9], t.size))
            if xs:
                curve = pg.PlotCurveItem(np.concatenate(xs), np.concatenate(ys), connect="pairs",
                                         pen=pg.mkPen(colour_of[tf], width=1))
                self.raster_plot.addItem(curve)
        # One label at the middle of each spatial period's block of trials.
        ticks, start = [], 0
        for r in range(1, n + 1):
            if r == n or keys[r][0] != keys[start][0]:
                ticks.append(((start + r) / 2.0, f"{keys[start][0]:g}{unit}"))
                if r < n:
                    line = pg.InfiniteLine(pos=r, angle=0, pen=pg.mkPen(c["text_secondary"], width=0.5))
                    self.raster_plot.addItem(line)
                start = r
        left = self.raster_plot.getAxis("left")
        left.setTicks([ticks])
        left.setLabel(f"Spatial {name}")
        self.raster_plot.setXRange(-pre, stim + tail, padding=0.01)
        self.raster_plot.setYRange(0, max(n, 1), padding=0)
        legend = ", ".join(f"{tf:.3g} Hz" for tf in tfs)
        self.note.setText(f"Rasters: one row per trial ({n}), grouped by spatial {name}; tick colour = "
                          f"temporal frequency ({legend}); shading = stimulus on.")
