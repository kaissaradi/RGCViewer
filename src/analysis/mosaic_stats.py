"""Does a class tile the retina? Mosaic numbers per class (PLAN.md Q42).

A real RGC type tiles: its receptive fields sit about one RF apart with
little same-type overlap (docs/design/rgc_types.md). Numbers here:

* NNND (Ravi et al. 2018): for each cell and its nearest same-class
  neighbour, 2·d / (S1 + S2), with d the centre distance and S each RF's
  1-SD radius along the line between them. About 2 when the RFs touch at
  1 SD. Below CLOSE_NNND the pair overlaps heavily: a split unit, a
  duplicate, or two types in one class. Arrays record only some of the
  cells, so a gap does not disprove a type; a close pair is the stronger sign.
* coverage: summed 1-SD RF area over the area the centres span (convex
  hull). Mouse types are often ~2–3 (Baden et al. 2016); well above that
  suggests a mixture.
* holes: cells outside the class whose RF centre sits inside the class's
  footprint but far from every member — where a missing member would be.

Inputs are ``rf_geometry.RFFit`` in Vision's y-up stixel frame. Pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

CLOSE_NNND = 1.0     # below: heavy overlap, worth a look
HOLE_NNND = 2.5      # a non-member this far from every member sits in a hole


def radius_toward(fit, angle):
    """1-SD radius of the fitted ellipse along direction ``angle`` (y-up, rad).

    In the y-up frame the SigmaX axis lies at -Theta (rf_geometry). Works on
    arrays of angles.
    """
    psi = np.asarray(angle) + float(fit.theta)
    a, b = float(fit.std_x), float(fit.std_y)
    return a * b / np.hypot(b * np.cos(psi), a * np.sin(psi))


def _arrays(fits):
    """(x, y, sx, sy, theta) columns for a list of fits."""
    if not fits:
        return (np.zeros(0),) * 5
    m = np.array([[f.x0, f.y0, f.std_x, f.std_y, f.theta] for f in fits], dtype=float)
    return m[:, 0], m[:, 1], m[:, 2], m[:, 3], m[:, 4]


def _radius(sx, sy, theta, angle):
    psi = angle + theta
    return sx * sy / np.hypot(sy * np.cos(psi), sx * np.sin(psi))


def nnnd_matrix(a_fits, b_fits) -> np.ndarray:
    """NNND 2·d / (S_a + S_b) for every pair (rows a, columns b); vectorised."""
    ax, ay, asx, asy, ath = _arrays(a_fits)
    bx, by, bsx, bsy, bth = _arrays(b_fits)
    dx = bx[None, :] - ax[:, None]
    dy = by[None, :] - ay[:, None]
    d = np.hypot(dx, dy)
    ang = np.arctan2(dy, dx)
    sa = _radius(asx[:, None], asy[:, None], ath[:, None], ang)
    sb = _radius(bsx[None, :], bsy[None, :], bth[None, :], ang + np.pi)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d == 0, 0.0, 2.0 * d / (sa + sb))


def _cross(u, v) -> float:
    return float(u[0] * v[1] - u[1] * v[0])


def _hull(points: np.ndarray) -> np.ndarray:
    """Convex-hull vertices, counter-clockwise (monotone chain)."""
    pts = np.unique(points, axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and _cross(out[-1] - out[-2], p - out[-2]) <= 0:
                out.pop()
            out.append(p)
        return out
    return np.array(half(pts)[:-1] + half(pts[::-1])[:-1])


def _hull_area(points: np.ndarray) -> float:
    hull = _hull(points)
    if len(hull) < 3:
        return 0.0
    x, y = hull[:, 0], hull[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _inside(points: np.ndarray, hull: np.ndarray) -> np.ndarray:
    """Which points lie inside (or on) a counter-clockwise convex hull."""
    if len(hull) < 3 or len(points) == 0:
        return np.zeros(len(points), dtype=bool)
    e0, e1 = hull, np.roll(hull, -1, axis=0)
    ex, ey = (e1 - e0)[:, 0], (e1 - e0)[:, 1]
    px = points[:, 0][:, None] - e0[:, 0][None, :]
    py = points[:, 1][:, None] - e0[:, 1][None, :]
    return np.all(ex[None, :] * py - ey[None, :] * px >= -1e-9, axis=1)


@dataclass
class MosaicStats:
    n: int
    nnnd: Dict[int, Tuple[float, int]]     # cell -> (NNND, nearest same-class cell)
    median_nnnd: float
    close_pairs: List[Tuple[int, int, float]]
    coverage: float

    @property
    def verdict(self) -> str:
        """One line for the atlas tile."""
        if self.n < 3:
            return "too few cells to judge"
        parts = [f"NNND {self.median_nnnd:.1f}"]
        if np.isfinite(self.coverage) and self.coverage > 0:
            parts.append(f"coverage {self.coverage:.1f}")
        if self.close_pairs:
            parts.append(f"{len(self.close_pairs)} close pair{'s' if len(self.close_pairs) != 1 else ''}")
        return " · ".join(parts)


def class_stats(fits: Dict[int, "object"]) -> MosaicStats:
    items = [(int(c), f) for c, f in fits.items() if f is not None]
    ids = [c for c, _f in items]
    flist = [f for _c, f in items]
    nnnd = {}
    if len(items) >= 2:
        m = nnnd_matrix(flist, flist)
        np.fill_diagonal(m, np.inf)
        j = np.argmin(m, axis=1)
        for i, c in enumerate(ids):
            nnnd[c] = (float(m[i, j[i]]), ids[j[i]])
    close = sorted({(min(c, j), max(c, j), v) for c, (v, j) in nnnd.items() if v < CLOSE_NNND},
                   key=lambda t: t[2])
    centres = np.array([[f.x0, f.y0] for f in flist]) if flist else np.zeros((0, 2))
    area = _hull_area(centres) if len(centres) else 0.0
    rf_area = sum(np.pi * f.std_x * f.std_y for f in flist)
    vals = [v for v, _j in nnnd.values() if np.isfinite(v)]
    return MosaicStats(
        n=len(items), nnnd=nnnd,
        median_nnnd=float(np.median(vals)) if vals else float("nan"),
        close_pairs=close,
        coverage=float(rf_area / area) if area > 0 else float("nan"))


def hole_candidates(members: Dict[int, "object"], others: Dict[int, "object"],
                    min_nnnd: float = HOLE_NNND) -> List[Tuple[int, float]]:
    """Non-members inside the class footprint and far from every member.

    Sorted farthest first: (cell, NNND to the nearest member).
    """
    mem = [f for f in members.values() if f is not None]
    if len(mem) < 3:
        return []
    cand = [(int(c), f) for c, f in others.items() if f is not None and int(c) not in members]
    if not cand:
        return []
    hull = _hull(np.array([[f.x0, f.y0] for f in mem]))
    pts = np.array([[f.x0, f.y0] for _c, f in cand])
    inside = _inside(pts, hull)
    if not inside.any():
        return []
    kept = [cand[i] for i in np.flatnonzero(inside)]
    near = nnnd_matrix([f for _c, f in kept], mem).min(axis=1)
    out = [(c, float(v)) for (c, _f), v in zip(kept, near) if v >= min_nnnd]
    return sorted(out, key=lambda t: -t[1])
