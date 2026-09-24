"""Raster ring around the grating polar plot (PLAN.md Q8).

The old 3x3 PSTH grid held at most 8 directions, so a 12-direction run lost
four. The ring gives every direction its own raster at its own angle.
"""

import numpy as np
import pytest

from src.gui.panels.polar_raster_view import raster_segments, ring_layout


def _overlaps(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


@pytest.mark.parametrize("n_dirs", [4, 6, 8, 12, 16])
@pytest.mark.parametrize("size", [(900, 500), (600, 600), (420, 300)])
def test_every_direction_gets_a_raster_and_none_overlap(n_dirs, size):
    dirs = np.arange(n_dirs) * 360.0 / n_dirs
    polar, rects = ring_layout(dirs, *size)
    assert len(rects) == n_dirs
    for i in range(n_dirs):
        for j in range(i + 1, n_dirs):
            assert not _overlaps(rects[i], rects[j]), (i, j)
    for r in rects:
        assert r[0] >= 0 and r[1] >= 0
        assert r[0] + r[2] <= size[0] + 1e-6 and r[1] + r[3] <= size[1] + 1e-6


def test_rasters_sit_at_their_direction():
    polar, rects = ring_layout([0.0, 90.0, 180.0, 270.0], 800, 600)
    centre = [(x + w / 2, y + h / 2) for x, y, w, h in rects]
    assert centre[0][0] > 400 and abs(centre[0][1] - 300) < 1   # 0° right
    assert centre[1][1] < 300 and abs(centre[1][0] - 400) < 1   # 90° up
    assert centre[2][0] < 400                                   # 180° left
    assert centre[3][1] > 300                                   # 270° down


def test_raster_segments_one_tick_per_spike_trial_zero_on_top():
    x, y = raster_segments([np.array([0.1, 0.2]), np.array([]), np.array([0.3])])
    assert list(x) == [0.1, 0.1, 0.2, 0.2, 0.3, 0.3]
    # 3 trials: trial 0 in row 2 (top), trial 2 in row 0.
    assert list(y[:2]) == [2.1, 2.9]
    assert list(y[-2:]) == [0.1, 0.9]


@pytest.mark.parametrize("n_dirs", [4, 6, 8, 12])
def test_polar_plot_clears_every_raster(n_dirs):
    dirs = np.arange(n_dirs) * 360.0 / n_dirs
    polar, rects = ring_layout(dirs, 900, 800)
    assert polar[2] >= 40
    for r in rects:
        assert not _overlaps(polar, r)
