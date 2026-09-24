"""Do the loaded Vision files describe the cells of the loaded sort?

Encore pairs Vision cell ``v`` with Kilosort cluster ``v - 1`` (AGENTS.md
Law 1). The pairing is only right if the Vision files (.sta, .params) were
made from this sort. A .sta/.params made from another sort still loads, and
every STA, RF fit and Vision class is then shown on the wrong cell.

Test: a cell's receptive field sits over its soma, so across cells the RF
centre (``x0, y0`` in .params, stixels) is an affine function of the cell's
position on the array (µm). Fit that map by least squares over the paired
cells, drop the worst 10 % (bad RF fits), fit again, and report R².

Measured 2026-09-24 (template-centroid positions, robust R²):

    20260220A/kilosort25/data022        0.86   matched folder
    20260529A/kilosort25/data017        0.60   matched folder
    20260511A/kilosort40/data007        0.52   244 stale .params rows
    20260715A/kilosort25/data007-010    0.38   53 stale .params rows
    20251212A/kilosort40/data018        0.03   .sta/.params from another sort
    any folder, ids shuffled            < 0.01

Stale rows (clusters deleted after the STA run) do not break the pairing of
the rest, so the id sets alone cannot tell the two cases apart. That is why
DataManager.check_sta_consistency (id sets only) is informational, and why
this check looks at the content instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

MIN_CELLS = 30          # fewer paired cells: no verdict
MISMATCH_R2 = 0.15      # robust R² below this: the pairing is wrong
KEEP_FRACTION = 0.9     # fraction of cells kept for the robust refit


@dataclass(frozen=True)
class SortCheck:
    n_cells: int            # cells with both an RF centre and an array position
    r2: float               # all paired cells
    r2_robust: float        # after dropping the worst 10 %

    @property
    def decided(self) -> bool:
        return self.n_cells >= MIN_CELLS and np.isfinite(self.r2_robust)

    @property
    def mismatch(self) -> bool:
        return self.decided and self.r2_robust < MISMATCH_R2


NO_CHECK = SortCheck(0, float("nan"), float("nan"))


def _r2(p: np.ndarray, q: np.ndarray) -> Tuple[float, np.ndarray]:
    x = np.c_[p, np.ones(len(p))]
    coef, *_ = np.linalg.lstsq(x, q, rcond=None)
    res = q - x @ coef
    ss_tot = ((q - q.mean(0)) ** 2).sum()
    return (1.0 - (res ** 2).sum() / ss_tot if ss_tot > 0 else float("nan"),
            np.linalg.norm(res, axis=1))


def check_pairing(rf_centres: Dict[int, Tuple[float, float]],
                  cell_positions: Dict[int, Tuple[float, float]]) -> SortCheck:
    """R² of RF centre against array position, keyed by the same (Vision) id.

    Pairs with a non-finite value, or an RF centre of exactly (0, 0) (Vision's
    "no fit"), are left out.
    """
    ids = [v for v in rf_centres if v in cell_positions]
    q = np.array([rf_centres[v] for v in ids], dtype=float).reshape(-1, 2)
    p = np.array([cell_positions[v] for v in ids], dtype=float).reshape(-1, 2)
    ok = np.isfinite(q).all(1) & np.isfinite(p).all(1) & ~(q == 0).all(1)
    p, q = p[ok], q[ok]
    if len(p) < MIN_CELLS:
        return SortCheck(len(p), float("nan"), float("nan"))
    r2_all, err = _r2(p, q)
    keep = err <= np.quantile(err, KEEP_FRACTION)
    r2_keep, _ = _r2(p[keep], q[keep])
    return SortCheck(len(p), float(r2_all), float(r2_keep))


def describe(check: SortCheck) -> str:
    """One sentence for the log, the status bar and the Ctrl+S dialog."""
    return (f"The Vision RF centres do not follow the cells' array positions "
            f"(R² = {check.r2_robust:.2f} over {check.n_cells} cells; matched folders "
            f"give 0.4–0.9). The .sta/.params files probably come from a different "
            f"sort, so their IDs point at other cells.")
