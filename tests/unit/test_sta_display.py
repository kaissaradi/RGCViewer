"""STA display scale and space–time slices.

Vision STAs are zero-mean with peak |value| = 1. The panel used to stretch
every frame to its own min/max, so "gray" moved between frames and a pure
noise frame filled the full contrast range. It now uses one symmetric scale
for the whole movie, with 0 at mid-gray.
"""

import numpy as np

from src.gui.panels.sta_panel import space_time_slices, stimulus_frame


def test_zero_maps_to_mid_gray():
    frame = np.zeros((4, 5, 3), dtype=np.float32)
    assert np.allclose(stimulus_frame(frame, absmax=1.0), 0.5)


def test_noise_frame_stays_near_gray_under_the_movie_scale():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.05, size=(10, 10, 3))
    out = stimulus_frame(noise, absmax=1.0)   # the movie peak is elsewhere
    # A per-frame min/max stretch would put this at 0..1; the fixed scale
    # keeps it within ±0.05 * 0.5 * a few sigma of gray.
    assert out.min() > 0.35 and out.max() < 0.65


def test_peak_maps_to_the_ends_and_sign_is_kept():
    frame = np.array([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]])
    out = stimulus_frame(frame, absmax=1.0)
    assert np.allclose(out[0, 0], 0.0)   # darkest = negative peak
    assert np.allclose(out[0, 1], 1.0)   # brightest = positive peak


def test_empty_movie_is_flat_gray():
    frame = np.zeros((2, 2, 3))
    assert np.allclose(stimulus_frame(frame, absmax=0.0), 0.5)


def test_space_time_slices_pick_the_row_and_column():
    h, w, t = 3, 4, 5
    ch = np.arange(h * w * t, dtype=float).reshape(h, w, t)
    xt, yt = space_time_slices(ch, row=1, col=2)
    assert xt.shape == (t, w)
    assert yt.shape == (t, h)
    # x–t: time on axis 0, column on axis 1, taken from row 1
    assert np.array_equal(xt[:, 3], ch[1, 3, :])
    # y–t: time on axis 0, row on axis 1, taken from column 2
    assert np.array_equal(yt[:, 0], ch[0, 2, :])


def test_space_time_slices_clip_out_of_range_centres():
    ch = np.zeros((3, 4, 2))
    xt, yt = space_time_slices(ch, row=99, col=-5)
    assert xt.shape == (2, 4) and yt.shape == (2, 3)
