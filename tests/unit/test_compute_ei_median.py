"""compute_ei no longer needs torch, and gives the same numbers (PLAN.md Q27)."""

import numpy as np
import pytest

from src.analysis import analysis_core


@pytest.mark.parametrize("n_spikes", [1, 2, 7, 30])
def test_median_is_the_lower_middle_value(n_spikes):
    rng = np.random.default_rng(n_spikes)
    snips = rng.normal(size=(5, 40, n_spikes)).astype(np.float32)
    ei = analysis_core.compute_ei(snips)
    base = analysis_core.baseline_correct(snips)
    expect = np.sort(base, axis=2)[:, :, (n_spikes - 1) // 2]
    np.testing.assert_array_equal(ei, expect)


def test_matches_torch_exactly_when_torch_is_installed():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    for n in (4, 9):
        snips = rng.normal(size=(8, 50, n)).astype(np.float32)
        base = analysis_core.baseline_correct(snips)
        ref = torch.median(torch.from_numpy(base), dim=2).values.numpy()
        np.testing.assert_array_equal(analysis_core.compute_ei(snips), ref)
