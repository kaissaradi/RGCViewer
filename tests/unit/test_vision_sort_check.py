"""Doubtful pairing and noise STAs in the Vision sort check (PLAN.md Q32, Q37)."""

import numpy as np

from src.analysis import vision_sort_check as vsc


def _check(r2, snr=None):
    return vsc.SortCheck(300, r2, r2, (0, 1, -1, 0), snr)


def test_three_bands():
    assert _check(0.03).mismatch and _check(0.03).warn
    doubtful = _check(0.22)                              # 20260715A/data003
    assert doubtful.doubtful and doubtful.warn and not doubtful.mismatch
    assert "only weakly" in vsc.describe(doubtful)
    good = _check(0.48)                                  # 20260715A/data010
    assert not good.warn and good.short == ""


def test_noise_stas_are_named_as_the_cause():
    noisy = _check(0.01, snr=4.0)                        # 20260721A/data006
    assert noisy.noisy_stas and noisy.warn
    assert noisy.short == "⚠ STAs look like noise"
    assert "mostly noise" in vsc.describe(noisy) and "different sort" not in vsc.describe(noisy)
    # good STAs and a bad fit: still the sort-pairing message
    assert "different sort" in vsc.describe(_check(0.03, snr=10.0))
    # noisy STAs but a pairing that works: no warning from this check
    assert not _check(0.6, snr=5.0).warn


def test_check_pairing_carries_no_snr_by_default():
    rng = np.random.default_rng(0)
    pos = {i: tuple(rng.uniform(0, 1000, 2)) for i in range(60)}
    rf = {i: (p[1] / 20.0, -p[0] / 20.0) for i, p in pos.items()}
    chk = vsc.check_pairing(rf, pos)
    assert chk.r2_robust > 0.99 and chk.sta_snr_median is None and not chk.warn
