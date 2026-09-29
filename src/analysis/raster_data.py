"""The Raster tab's numbers: folded rasters, nearby units, CCGs (PLAN.md Q63).

Nothing here touches Qt. The same functions serve Kilosort and Vision-only
sessions: positions come from ``cluster_df`` (x_um, y_um), similarity from
the Vision EIs when loaded, else from Kilosort's template similarity.

* ``fold_fixed`` / ``fold_trials``: a whole recording as rows, every spike
  kept. Rows are fixed-length segments, or one row per trial when the
  run's triggers show trials (``lisp_stimulus.trial_starts``, the MATLAB
  load_stim rule).
* ``nearby_units``: the cells within NEAR_UM of this one, plus Kilosort's
  most similar templates, ordered by similarity.
* ``cross_correlogram`` / ``shared_fraction``: whether a neighbour fires
  with this cell. A duplicate shares most spikes within ±0.5 ms; chance is
  1 − exp(−2 · rate · 0.5 ms), about 2 % at 20 Hz.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

NEAR_UM = 100.0            # neighbours: cells within this of the selected cell
MAX_NEIGHBOURS = 8
TEMPLATE_SIM_MIN = 0.5     # Kilosort template similarity worth showing beyond NEAR_UM
SHARED_TOL_S = 0.0005      # a shared spike: within ±0.5 ms
CCG_WINDOW_S = 0.025
CCG_BIN_S = 0.0005
MAX_CCG_PAIRS = 2_000_000


@dataclass
class Neighbour:
    cluster_id: int
    distance_um: float = float("nan")
    similarity: float = float("nan")
    sim_source: str = ""          # "EI", "template" or ""


# ── folding ───────────────────────────────────────────────────────────────────

def fold_fixed(times_s, row_s: float, duration_s: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """(x in row, row index, number of rows) for rows of ``row_s`` seconds from 0."""
    t = np.asarray(times_s, dtype=np.float64)
    row_s = float(row_s)
    n_rows = max(1, int(np.ceil(max(float(duration_s), 1e-9) / row_s)))
    rows = np.floor(t / row_s).astype(np.int64)
    keep = (rows >= 0) & (rows < n_rows)
    return t[keep] - rows[keep] * row_s, rows[keep], n_rows


def fold_trials(times_s, starts_s, span_s: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """(x from trial start, trial index, number of trials): spikes in [start, start + span)."""
    t = np.asarray(times_s, dtype=np.float64)
    starts = np.asarray(starts_s, dtype=np.float64)
    if starts.size == 0:
        return np.zeros(0), np.zeros(0, dtype=np.int64), 0
    i = np.searchsorted(starts, t, side="right") - 1
    ok = i >= 0
    x = np.full(t.shape, np.inf)
    x[ok] = t[ok] - starts[i[ok]]
    keep = ok & (x < span_s)
    return x[keep], i[keep].astype(np.int64), int(starts.size)


def trial_structure(ttl_s) -> Optional[Tuple[np.ndarray, float]]:
    """(trial starts, row span) when the triggers come in trials, else None.

    The starts are load_stim's (lisp_stimulus.trial_starts). A white-noise
    run's evenly spaced triggers give no trials. The span is the median
    start-to-start time, so a row runs up to the next trial.
    """
    from . import lisp_stimulus
    ttl = np.asarray(ttl_s, dtype=np.float64)
    if ttl.size < 6:
        return None
    starts = lisp_stimulus.trial_starts(ttl)
    # load_stim's threshold is 2 SD of the intervals: with evenly spaced
    # triggers (white noise) a one-sample jitter passes it. A trial start
    # here also needs a real gap before it: > 1.5 × the median interval.
    gap_min = 1.5 * float(np.median(np.diff(ttl)))
    idx = np.searchsorted(ttl, starts)
    prev = np.r_[np.inf, ttl[np.clip(idx[1:] - 1, 0, ttl.size - 1)]]
    starts = starts[(starts - prev >= gap_min) | (np.arange(starts.size) == 0)]
    if starts.size < 3 or starts.size >= ttl.size:
        return None
    span = float(np.median(np.diff(starts)))
    if not np.isfinite(span) or span <= 0:
        return None
    return starts, span


def rate_curve(times_s, duration_s: float, n_bins: int = 1200) -> Tuple[np.ndarray, np.ndarray, float]:
    """(bin centres s, rate spikes/s, bin width s) over the recording."""
    duration_s = max(float(duration_s), 1e-6)
    bin_s = max(0.25, duration_s / n_bins)
    edges = np.arange(0.0, duration_s + bin_s, bin_s)
    counts, _ = np.histogram(np.asarray(times_s, dtype=np.float64), bins=edges)
    return (edges[:-1] + edges[1:]) / 2.0, counts / bin_s, bin_s


# ── pairs ─────────────────────────────────────────────────────────────────────

def _pairs(a, b, window_s):
    """Differences b − a for every pair within ±window_s (a subsampled if huge)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros(0), a.size
    lo = np.searchsorted(b, a - window_s, side="left")
    hi = np.searchsorted(b, a + window_s, side="right")
    counts = hi - lo
    total = int(counts.sum())
    n_used = a.size
    if total > MAX_CCG_PAIRS:
        keep = np.linspace(0, a.size - 1, int(a.size * MAX_CCG_PAIRS / total)).astype(np.int64)
        a, lo, counts = a[keep], lo[keep], counts[keep]
        total = int(counts.sum())
        n_used = a.size
    if total == 0:
        return np.zeros(0), n_used
    ia = np.repeat(np.arange(a.size), counts)
    first = np.repeat(np.cumsum(counts) - counts, counts)
    ib = np.repeat(lo, counts) + (np.arange(total) - first)
    return b[ib] - a[ia], n_used


def cross_correlogram(a, b, window_s: float = CCG_WINDOW_S, bin_s: float = CCG_BIN_S):
    """(bin centres ms, rate of b around a's spikes in spikes/s)."""
    d, n_a = _pairs(a, b, window_s)
    n = int(round(window_s / bin_s))
    edges = np.linspace(-n * bin_s, n * bin_s, 2 * n + 1)
    counts, _ = np.histogram(d, bins=edges)
    rate = counts / max(1, n_a) / bin_s
    return (edges[:-1] + edges[1:]) / 2.0 * 1000.0, rate


def shared_fraction(a, b, tol_s: float = SHARED_TOL_S) -> float:
    """Fraction of a's spikes with a spike of b within ±tol_s."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 0.0
    i = np.searchsorted(b, a)
    right = np.abs(b[np.clip(i, 0, b.size - 1)] - a)
    left = np.abs(b[np.clip(i - 1, 0, b.size - 1)] - a)
    return float(np.mean(np.minimum(left, right) <= tol_s))


def chance_shared(rate_b_hz: float, tol_s: float = SHARED_TOL_S) -> float:
    """shared_fraction expected from two independent trains (Poisson b)."""
    return float(1.0 - np.exp(-2.0 * tol_s * max(0.0, float(rate_b_hz))))


# ── neighbours ────────────────────────────────────────────────────────────────

def _positions(dm):
    df = dm.cluster_df
    if df is None or "x_um" not in df.columns or "y_um" not in df.columns:
        return None, None
    return df["cluster_id"].to_numpy().astype(np.int64), \
        np.c_[df["x_um"].to_numpy(dtype=float), df["y_um"].to_numpy(dtype=float)]


def nearby_units(dm, cluster_id: int, radius_um: float = NEAR_UM,
                 max_n: int = MAX_NEIGHBOURS) -> List[Neighbour]:
    """Cells within ``radius_um`` (nearest first), plus similar Kilosort templates."""
    cid = int(cluster_id)
    ids, xy = _positions(dm)
    out: Dict[int, Neighbour] = {}
    if ids is not None:
        here = np.flatnonzero(ids == cid)
        if here.size and np.all(np.isfinite(xy[here[0]])):
            d = np.hypot(*(xy - xy[here[0]]).T)
            order = np.argsort(np.where(np.isfinite(d), d, np.inf))
            for j in order:
                if ids[j] == cid or not np.isfinite(d[j]) or d[j] > radius_um:
                    continue
                out[int(ids[j])] = Neighbour(int(ids[j]), float(d[j]))
                if len(out) >= 3 * max_n:
                    break
    # Kilosort's template similarity (Kilosort sessions only).
    table = None
    fn = getattr(dm, "_get_mea_similarity_table", None)
    if callable(fn) and getattr(dm, "_optional_attr", lambda *_: None)("similar_templates") is not None:
        try:
            table = fn(cid)
        except Exception:
            logger.debug("template similarity failed for %s", cid, exc_info=True)
    if table is not None and len(table) and "template_sim" in table.columns:
        for row in table.itertuples(index=False):
            other = int(row.cluster_id)
            sim = float(row.template_sim)
            if other == cid or not np.isfinite(sim):
                continue
            if other in out:
                out[other].similarity, out[other].sim_source = sim, "template"
            elif sim >= TEMPLATE_SIM_MIN:
                dist = float(getattr(row, "distance_um", np.nan))
                out[other] = Neighbour(other, dist, sim, "template")
    return list(out.values())


def ei_similarities(dm, cluster_id: int, others: Sequence[int]) -> Dict[int, float]:
    """Correlation of each other cell's Vision EI with this one's ({} without EIs).

    The lab's ``ei_corr`` (full EI, largest channel dropped, values below
    1.5 SD zeroed), one read of each EI.
    """
    eis = getattr(dm, "vision_eis", None)
    if not eis or not others:
        return {}
    from .data_manager import ei_corr
    vid = dm.get_vision_id_for_cluster
    try:
        ref = eis.get(vid(int(cluster_id))) if hasattr(eis, "get") else eis[vid(int(cluster_id))]
    except Exception:
        ref = None
    if ref is None:
        return {}
    test, keys = {}, []
    for c in others:
        try:
            e = eis.get(vid(int(c))) if hasattr(eis, "get") else eis[vid(int(c))]
        except Exception:
            e = None
        if e is not None and getattr(e, "ei", None) is not None:
            test[int(c)] = e
            keys.append(int(c))
    if not test:
        return {}
    corr = ei_corr({0: ref}, test)          # (n_ref = 1, n_test)
    if corr.size != len(keys):
        return {}
    return {k: float(v) for k, v in zip(keys, np.asarray(corr).reshape(-1))}


def rank_neighbours(neighbours: List[Neighbour], ei_sims: Dict[int, float],
                    max_n: int = MAX_NEIGHBOURS) -> List[Neighbour]:
    """EI similarity where known (it overrides template similarity), most similar first."""
    for n in neighbours:
        if n.cluster_id in ei_sims:
            n.similarity, n.sim_source = ei_sims[n.cluster_id], "EI"

    def key(n):
        sim = n.similarity if np.isfinite(n.similarity) else -np.inf
        dist = n.distance_um if np.isfinite(n.distance_um) else np.inf
        return (-sim, dist)

    return sorted(neighbours, key=key)[:max_n]
