"""Spike raster maths (PLAN.md Q38)."""

import numpy as np

from src.gui.panels import raster_view as rv


def test_density_counts_per_bin():
    t = [np.array([0.1, 0.2, 0.25, 1.5]), np.array([]), np.array([1.9])]
    d = rv.density(t, 0.0, 2.0, 2)
    assert d.tolist() == [[3, 1], [0, 0], [0, 1]]


def test_ticks_two_points_per_spike_and_a_limit():
    t = [np.array([0.5, 1.0, 3.0]), np.array([0.7])]
    xs, ys = rv.ticks(t, 0.0, 2.0)
    assert xs.tolist() == [0.5, 0.5, 1.0, 1.0, 0.7, 0.7]
    assert ys[:2].tolist() == [0.15, 0.85] and ys[-1] == 1.85
    assert rv.ticks(t, 0.0, 2.0, limit=2) == (None, None)     # too many: stay on the density view
