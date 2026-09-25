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
from typing import Dict, Optional, Tuple

import numpy as np

MIN_CELLS = 30          # fewer paired cells: no verdict
MISMATCH_R2 = 0.15      # robust R² below this: the pairing is wrong
# Below this (but above MISMATCH_R2): doubtful. Every matched folder so far
# scored 0.38 or more; 20260715A/data003 scored 0.22 and its RF centres do
# not agree with the same cells in data010 (PLAN.md Q37).
DOUBTFUL_R2 = 0.30
# Median STA peak/RMS below this: the STAs are mostly noise (20260721A
# data006 / data007: 4.0 / 4.5; good runs 7–12), so no RF follows its cell
# whatever the pairing.
NOISY_STA_SNR = 5.5
KEEP_FRACTION = 0.9     # fraction of cells kept for the robust refit


@dataclass(frozen=True)
class SortCheck:
    n_cells: int            # cells with both an RF centre and an array position
    r2: float               # all paired cells
    r2_robust: float        # after dropping the worst 10 %
    # Array → screen orientation from the robust fit, snapped to the nearest
    # quarter turn / mirror: (m00, m01, m10, m11), screen ≈ M · array. None
    # when there is no fit. On the lab rig it is a −90° turn (PLAN.md Q20).
    screen_matrix: Optional[Tuple[int, int, int, int]] = None
    sta_snr_median: Optional[float] = None   # median STA peak/RMS of the run, when known

    @property
    def screen_turn(self) -> Optional[Tuple[int, int, int, int]]:
        """screen_matrix, but only when the pairing itself can be trusted."""
        return self.screen_matrix if self.decided and not self.mismatch else None

    @property
    def decided(self) -> bool:
        return self.n_cells >= MIN_CELLS and np.isfinite(self.r2_robust)

    @property
    def mismatch(self) -> bool:
        return self.decided and self.r2_robust < MISMATCH_R2

    @property
    def doubtful(self) -> bool:
        return self.decided and MISMATCH_R2 <= self.r2_robust < DOUBTFUL_R2

    @property
    def noisy_stas(self) -> bool:
        return self.sta_snr_median is not None and self.sta_snr_median < NOISY_STA_SNR

    @property
    def warn(self) -> bool:
        """Anything worth a warning: a mismatch, a doubtful pairing, or noise STAs."""
        return self.mismatch or self.doubtful or (self.decided and self.noisy_stas
                                                  and self.r2_robust < DOUBTFUL_R2)

    @property
    def short(self) -> str:
        """A few words for the status bar."""
        if not self.warn:
            return ""
        if self.noisy_stas:
            return "⚠ STAs look like noise"
        return "⚠ Vision files may be from another sort" if self.mismatch else "⚠ Weak Vision/sort pairing"


NO_CHECK = SortCheck(0, float("nan"), float("nan"))


def _r2(p: np.ndarray, q: np.ndarray, return_coef=False):
    x = np.c_[p, np.ones(len(p))]
    coef, *_ = np.linalg.lstsq(x, q, rcond=None)
    res = q - x @ coef
    ss_tot = ((q - q.mean(0)) ** 2).sum()
    out = (1.0 - (res ** 2).sum() / ss_tot if ss_tot > 0 else float("nan"),
           np.linalg.norm(res, axis=1))
    return out + (coef,) if return_coef else out


_SIGNED_PERMUTATIONS = [(a, 0, 0, b) for a in (1, -1) for b in (1, -1)] + \
                       [(0, a, b, 0) for a in (1, -1) for b in (1, -1)]


def nearest_quarter_turn(a_matrix) -> Tuple[int, int, int, int]:
    """The rotation / mirror by quarter turns closest to the 2×2 map ``a_matrix``.

    The polar factor R = U·Vᵀ of A removes scale and shear; the signed
    permutation M that maximises trace(Mᵀ·R) is the closest of the eight.
    """
    u, _s, vt = np.linalg.svd(np.asarray(a_matrix, dtype=float))
    r = u @ vt
    return max(_SIGNED_PERMUTATIONS,
               key=lambda m: m[0] * r[0, 0] + m[1] * r[0, 1] + m[2] * r[1, 0] + m[3] * r[1, 1])


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
    r2_keep, _, coef = _r2(p[keep], q[keep], return_coef=True)
    a_matrix = coef[:2].T                       # q ≈ A · p + b
    return SortCheck(len(p), float(r2_all), float(r2_keep), nearest_quarter_turn(a_matrix))


def describe(check: SortCheck) -> str:
    """One or two sentences for the log, the status bar and the Ctrl+S dialog."""
    fit = (f"R² = {check.r2_robust:.2f} over {check.n_cells} cells; matched folders "
           f"give 0.4–0.9")
    if check.noisy_stas and check.r2_robust < DOUBTFUL_R2:
        return (f"The STAs are mostly noise (median peak/RMS {check.sta_snr_median:.1f}; good "
                f"runs 7–12), so their RF centres do not follow the cells ({fit}). The white-"
                f"noise analysis may have used the wrong movie or frame timing.")
    if check.mismatch:
        return (f"The Vision RF centres do not follow the cells' array positions ({fit}). "
                f"The .sta/.params files probably come from a different sort, so their IDs "
                f"point at other cells.")
    return (f"The Vision RF centres follow the cells' array positions only weakly ({fit}). "
            f"Some STAs and classes may belong to other cells: compare a few with their EIs.")
