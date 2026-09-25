"""Type barcode: one row per cell, grouped by class (PLAN.md Q40).

Each row is one cell's response (STA time course, ACG, chirp PSTH), scaled
to its own peak so shape — not spike count — sets the colour. Rows are
grouped in tree order; within a group the most typical cell comes first.
Typicality is the correlation with the mean of the *other* cells in the
group, so one cell does not vote for itself. A row far below its band is a
cell that may be in the wrong group.

Pure numpy; the GUI is ``src/gui/panels/types_panel.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

# A cell "may not belong" when another type's mean fits it better than its own
# group's by this margin, or when it hardly matches its own group at all.
BETTER_ELSEWHERE = 0.05
MISFIT_R = 0.5


@dataclass
class Barcode:
    matrix: np.ndarray                    # (n_rows, n_cols), each row scaled to its peak
    cells: List[int]                      # cluster id per row
    groups: List[str]                     # group per row
    typicality: np.ndarray                # r with the rest of its group (nan: < 3 cells)
    bands: List[Tuple[str, int, int]] = field(default_factory=list)  # (group, first, end)
    # best other type group and its r, per row ('' / nan when none)
    elsewhere: List[Tuple[str, float]] = field(default_factory=list)

    def misfits(self) -> List[int]:
        out = []
        for i, (c, r) in enumerate(zip(self.cells, self.typicality)):
            if not np.isfinite(r):
                continue
            other_r = self.elsewhere[i][1] if self.elsewhere else np.nan
            if r < MISFIT_R or (np.isfinite(other_r) and other_r > r + BETTER_ELSEWHERE):
                out.append(c)
        return out

    def row_of(self, cluster_id) -> int:
        try:
            return self.cells.index(int(cluster_id))
        except ValueError:
            return -1


def _scale(v: np.ndarray, signed: bool) -> np.ndarray:
    peak = np.nanmax(np.abs(v)) if signed else np.nanmax(v)
    return v / peak if np.isfinite(peak) and peak > 0 else np.zeros_like(v)


def _leave_one_out_r(block: np.ndarray) -> np.ndarray:
    n = len(block)
    if n < 3:
        return np.full(n, np.nan)
    total = np.nansum(block, axis=0)
    out = np.empty(n)
    for i in range(n):
        rest = (total - block[i]) / (n - 1)
        a, b = block[i] - block[i].mean(), rest - rest.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        out[i] = (a * b).sum() / den if den > 0 else np.nan
    return out


def _r(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def build_barcode(rows: Dict[int, np.ndarray], group_of: Dict[int, str],
                  group_order: Sequence[str], signed: bool = True,
                  sort_by_typicality: bool = True, type_groups=None) -> Barcode:
    """Rows for every cell that has a vector and a group, bands in group order.

    Vectors of different lengths are cut to the shortest. ``type_groups``:
    the groups that are types (not bins); only their means are offered as
    "fits better elsewhere", and only their cells can be flagged.
    """
    usable = {int(c): np.asarray(v, dtype=float) for c, v in rows.items()
              if v is not None and len(v) > 1 and int(c) in group_of}
    if not usable:
        return Barcode(np.zeros((0, 0)), [], [], np.zeros(0))
    width = min(len(v) for v in usable.values())
    order = list(dict.fromkeys(list(group_order) + sorted(set(group_of.values()))))
    matrix, cells, groups, typ, bands = [], [], [], [], []
    for g in order:
        members = [c for c in usable if group_of[c] == g]
        if not members:
            continue
        block = np.vstack([_scale(np.nan_to_num(usable[c][:width]), signed) for c in members])
        r = _leave_one_out_r(block)
        idx = np.argsort(-np.nan_to_num(r, nan=-2)) if sort_by_typicality else np.arange(len(members))
        start = len(cells)
        for i in idx:
            matrix.append(block[i])
            cells.append(members[i])
            groups.append(g)
            typ.append(r[i])
        bands.append((g, start, len(cells)))
    matrix = np.vstack(matrix)
    typ = np.asarray(typ)
    types = set(type_groups) if type_groups is not None else set(groups)
    means = {g: matrix[b:e].mean(axis=0) for g, b, e in bands if g in types and e - b >= 3}
    elsewhere = []
    for i, g in enumerate(groups):
        if g not in types:
            elsewhere.append(("", np.nan))
            continue
        best = ("", np.nan)
        for h, m in means.items():
            if h == g:
                continue
            r = _r(matrix[i], m)
            if np.isfinite(r) and not (r <= best[1]):
                best = (h, r)
        elsewhere.append(best)
    for i, g in enumerate(groups):
        if g not in types:
            typ[i] = np.nan               # bins are not judged
    return Barcode(matrix, cells, groups, typ, bands, elsewhere)
