"""Array views line up with the screen (PLAN.md Q20).

On the lab rig the array is a quarter turn from the screen: the affine fit
array position → RF centre has a −89° / −88° / −83° rotation on three
matched runs. Array-space drawings are turned by that run's snapped turn.
"""

import numpy as np
import pytest
from qtpy.QtCore import QSettings

from src.analysis import vision_sort_check as vsc
from src.gui import array_orientation as ao


def _rot(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


@pytest.mark.parametrize("deg, expect", [(-89, (0, 1, -1, 0)), (-83, (0, 1, -1, 0)),
                                         (2, (1, 0, 0, 1)), (91, (0, -1, 1, 0)), (179, (-1, 0, 0, -1))])
def test_snaps_to_the_nearest_quarter_turn(deg, expect):
    a = _rot(deg) @ np.diag([0.056, 0.042])          # anisotropic stixel/µm scale
    assert vsc.nearest_quarter_turn(a) == expect


def test_a_mirror_is_kept():
    assert vsc.nearest_quarter_turn(np.diag([-0.05, 0.05])) == (-1, 0, 0, 1)


def _run(deg=-89, n=150, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-900, 900, size=(n, 2))
    rf = pos @ (_rot(deg) @ np.diag([0.05, 0.04])).T + [70, 44] + rng.normal(0, 0.3, (n, 2))
    return ({i: tuple(r) for i, r in enumerate(rf)}, {i: tuple(p) for i, p in enumerate(pos)}, pos, rf)


def test_turned_positions_line_up_with_the_screen():
    rf, cells, pos, rf_arr = _run()
    check = vsc.check_pairing(rf, cells)
    assert check.screen_turn == (0, 1, -1, 0)
    shown = ao.to_display(pos, check.screen_turn)
    x = np.c_[shown, np.ones(len(shown))]
    coef, *_ = np.linalg.lstsq(x, rf_arr, rcond=None)
    assert vsc.nearest_quarter_turn(coef[:2].T) == ao.IDENTITY   # no turn left
    assert ao.describe(check.screen_turn) == "turned 90° clockwise"


def test_a_mismatched_run_is_not_turned():
    rf, cells, _, _ = _run()
    shuffled = dict(zip(cells, [cells[k] for k in np.random.default_rng(1).permutation(list(cells))]))
    assert vsc.check_pairing(rf, shuffled).screen_turn is None


def test_photo_transform_matches_the_point_turn():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    m = (0, 1, -1, 0)
    p = np.array([[300.0, -120.0]])
    via_image = ao.mpl_transform(ax, m).transform(p)
    via_points = ax.transData.transform(ao.to_display(p, m))
    np.testing.assert_allclose(via_image, via_points)
    plt.close(fig)


def test_setting_turns_it_off(tmp_path):
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)

    class _DM:
        def vision_sort_check(self):
            return vsc.SortCheck(200, 0.8, 0.9, (0, 1, -1, 0))

    assert ao.display_matrix(_DM(), settings) == (0, 1, -1, 0)
    ao.set_enabled(False, settings)
    assert ao.display_matrix(_DM(), settings) == ao.IDENTITY
    assert ao.display_matrix(None, settings) == ao.IDENTITY
