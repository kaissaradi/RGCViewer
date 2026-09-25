"""Per-cell features for suggested classes (PLAN.md Q36).

The definitions that scored 0.90 leave-one-prep-out on the lab's labelled
runs (2,767 cells, 31 preps; PLAN.md Q36). Everything comes from Vision
``.params`` columns plus medians over the run's own cells — no labels — so
the same code runs on the lab library and on a new, unlabelled run.

* polarity: sign of the time-course lobe nearest the spike (scan back from
  the last frame; first sample with |tc| >= half the peak). It matched 100 %
  of the lab's ON/OFF labels.
* tc_run: the time course resampled on 21 points from 0 to 2.5 x the run's
  median time-to-peak. Runs differ in frame duration although every .sta
  header says 8.333 ms, so time is measured in the run's own time-to-peak.
* acg: the ``Auto`` autocorrelation summed into 20 log-spaced bins
  (0.5–300 ms), as fractions of its total.
* rf_rel: log RF radius (sqrt(SigmaX * SigmaY) * stixel size) minus the
  run median; missing for unmoved fits (SigmaX = SigmaY = 1, Q26).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .rf_geometry import fit_is_unmoved

S_GRID = np.linspace(0.0, 2.5, 21)            # time in units of time-to-peak
ACG_EDGES = np.geomspace(0.5, 300.0, 21)      # 20 log-spaced bins, ms
N_FEATURES = len(S_GRID) + (len(ACG_EDGES) - 1) + 1


@dataclass
class CellInput:
    """What one cell contributes: .params columns and the run's stixel size."""
    tc: np.ndarray            # time course, oldest -> newest (last = frame at the spike)
    auto: np.ndarray          # Vision 'Auto'
    acf_binning: float        # ms per 'Auto' bin
    sigma_x: float
    sigma_y: float
    stixel: float


def polarity(tc) -> int:
    """+1 ON, -1 OFF, 0 when the time course is flat."""
    tc = np.asarray(tc, dtype=float)
    peak = np.max(np.abs(tc)) if tc.size else 0.0
    if not np.isfinite(peak) or peak <= 0:
        return 0
    for v in tc[::-1]:
        if abs(v) >= 0.5 * peak:
            return 1 if v > 0 else -1
    return 0


def _peak_frames_before_spike(tcn) -> float:
    """Frames from the |peak| to the last sample, parabolic sub-frame refinement."""
    a = np.abs(tcn)
    p = int(a.argmax())
    off = 0.0
    if 0 < p < len(a) - 1:
        den = a[p - 1] - 2 * a[p] + a[p + 1]
        off = 0.5 * (a[p - 1] - a[p + 1]) / den if den != 0 else 0.0
    return max((len(a) - 1) - (p + float(np.clip(off, -0.5, 0.5))), 0.5)


def _resample(tcn, frames_to_peak) -> np.ndarray:
    n = len(tcn)
    tau_axis = (n - 1) - np.arange(n)          # frames before the spike
    return np.interp(S_GRID * frames_to_peak, tau_axis[::-1], tcn[::-1],
                     left=tcn[-1], right=0.0)


def acg_logbins(auto, binning=0.5) -> np.ndarray:
    auto = np.asarray(auto, dtype=float)
    centres = (np.arange(len(auto)) + 0.5) * float(binning)
    idx = np.clip(np.searchsorted(ACG_EDGES, centres, side="right") - 1, 0, len(ACG_EDGES) - 2)
    out = np.bincount(idx, weights=np.nan_to_num(auto), minlength=len(ACG_EDGES) - 1)
    total = out.sum()
    return out / total if total > 0 else out


def run_features(cells: Sequence[CellInput]):
    """(features (n, N_FEATURES), polarity (n,)) for the cells of ONE run.

    Run medians (time-to-peak, RF size) come from these cells, so pass the
    whole run, not a subset. The RF column is NaN where there is no fit.
    """
    n = len(cells)
    if n == 0:
        return np.zeros((0, N_FEATURES)), np.zeros(0, dtype=int)
    tcn, frames, pol = [], np.zeros(n), np.zeros(n, dtype=int)
    for i, c in enumerate(cells):
        g = np.nan_to_num(np.asarray(c.tc, dtype=float))
        m = np.abs(g).max() if g.size else 0.0
        g = g / m if m > 0 else g
        tcn.append(g)
        frames[i] = _peak_frames_before_spike(g) if m > 0 else np.nan
        pol[i] = polarity(g)
    # Run medians from cells with a real RF fit: unmoved fits go with weak STAs.
    fitted = np.array([not fit_is_unmoved(c.sigma_x, c.sigma_y) for c in cells])
    basis = frames[fitted & np.isfinite(frames)]
    if basis.size == 0:
        basis = frames[np.isfinite(frames)]
    run_frames = float(np.median(basis)) if basis.size else 1.0
    tc_run = np.array([_resample(t, run_frames) for t in tcn])
    acg = np.array([acg_logbins(c.auto, c.acf_binning) for c in cells])
    rf = np.full(n, np.nan)
    for i, c in enumerate(cells):
        if fit_is_unmoved(c.sigma_x, c.sigma_y):
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.log(np.sqrt(abs(c.sigma_x * c.sigma_y)) * c.stixel)
        rf[i] = v if np.isfinite(v) else np.nan
    if np.isfinite(rf).any():
        rf = rf - np.nanmedian(rf)
    return np.hstack([tc_run, acg, rf[:, None]]), pol


def cells_from_vision(vision_params, vision_ids: Sequence[int], stixel: float) -> Dict[int, CellInput]:
    """CellInput for each Vision ID with a time course and an ACG in the loaded .params."""
    out = {}
    for vid in vision_ids:
        try:
            get = lambda name: vision_params.get_data_for_cell(int(vid), name)  # noqa: E731
            tc = get("GreenTimeCourse")
            if tc is None:
                tc = get("RedTimeCourse")
            auto = get("Auto")
            if tc is None or auto is None:
                continue
            out[int(vid)] = CellInput(
                tc=np.asarray(tc, float), auto=np.asarray(auto, float),
                acf_binning=float(get("acfBinning") or 0.5),
                sigma_x=float(get("SigmaX") or np.nan), sigma_y=float(get("SigmaY") or np.nan),
                stixel=float(stixel))
        except Exception:
            continue
    return out
