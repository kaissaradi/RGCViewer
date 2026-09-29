"""Many spike trains over a long window, fast (PLAN.md Q38, Q63).

Used by the Raster tab's lanes. Zoomed out, each row is a density image:
spike counts per screen pixel, recomputed only for the visible window
(``np.searchsorted`` on each cell's sorted spike times), so an hour of 300
cells stays quick. Zoomed in below TICKS_BELOW_S, every spike is a tick.
"""

from __future__ import annotations

from typing import List

import numpy as np

MAX_ROWS = 400
TICKS_BELOW_S = 6.0
MAX_TICKS = 60000


def density(spike_times: List[np.ndarray], t0: float, t1: float, n_bins: int) -> np.ndarray:
    """(rows, n_bins) spike counts on [t0, t1); each array must be sorted."""
    edges = np.linspace(t0, t1, n_bins + 1)
    out = np.zeros((len(spike_times), n_bins), dtype=np.float32)
    for i, t in enumerate(spike_times):
        if t.size:
            out[i] = np.diff(np.searchsorted(t, edges))
    return out


def ticks(spike_times: List[np.ndarray], t0: float, t1: float, limit=MAX_TICKS):
    """x, y for PlotCurveItem(connect='pairs'): one short vertical line per spike."""
    xs, ys = [], []
    n = 0
    for i, t in enumerate(spike_times):
        a, b = np.searchsorted(t, [t0, t1])
        sel = t[a:b]
        n += sel.size
        if n > limit:
            return None, None
        xs.append(np.repeat(sel, 2))
        ys.append(np.tile([i + 0.15, i + 0.85], sel.size))
    if not xs:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(xs), np.concatenate(ys)
