"""Draw the electrode array the way the screen is oriented (PLAN.md Q20).

The STA movie and the RF mosaic are in screen coordinates. Everything drawn
from electrode positions (EI map, spatial template, cell tracer, array
photo) is in array coordinates. On the lab rig the array is a quarter turn
from the screen (−89° / −88° / −83° on three matched runs, no mirror), so
those views looked rotated 90° against the STA.

The turn comes from each run's own data: vision_sort_check fits array
position → RF centre and snaps it to the nearest quarter turn or mirror.
It is applied only when that pairing passes, only for drawing, and only
while "Align array views to the screen" (Array menu) is on. Analysis and
array calibration always use the raw array coordinates.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

SETTING_KEY = "view/align_array_to_screen"
IDENTITY = (1, 0, 0, 1)


def _settings():
    from . import recent_paths
    return recent_paths._settings()


def enabled(settings=None) -> bool:
    value = (settings or _settings()).value(SETTING_KEY, True)
    return str(value).lower() not in ("false", "0", "no")


def set_enabled(on: bool, settings=None) -> None:
    (settings or _settings()).setValue(SETTING_KEY, bool(on))


def display_matrix(dm, settings=None):
    """(m00, m01, m10, m11) applied to positions for drawing; IDENTITY when off/unknown."""
    if dm is None or not enabled(settings):
        return IDENTITY
    try:
        turn = dm.vision_sort_check().screen_turn
    except Exception:
        logger.debug("no array/screen orientation for this run", exc_info=True)
        return IDENTITY
    return tuple(int(v) for v in turn) if isinstance(turn, tuple) and len(turn) == 4 else IDENTITY


def to_display(positions, matrix):
    """(N, 2) positions turned by ``matrix``; the input itself when it is IDENTITY."""
    if positions is None or tuple(matrix) == IDENTITY:
        return positions
    pos = np.asarray(positions, dtype=float)
    m = np.array(matrix, dtype=float).reshape(2, 2)
    return pos @ m.T


def mpl_transform(ax, matrix):
    """A matplotlib transform that draws array-µm artists (the photo) turned."""
    from matplotlib.transforms import Affine2D
    m00, m01, m10, m11 = matrix
    # Affine2D.from_values(a, b, c, d, e, f) is [[a, c, e], [b, d, f]].
    return Affine2D.from_values(m00, m10, m01, m11, 0.0, 0.0) + ax.transData


def describe(matrix) -> str:
    """Plain words for a quarter-turn matrix (screen = M · array)."""
    m00, m01, m10, m11 = matrix
    if tuple(matrix) == IDENTITY:
        return "not turned"
    det = m00 * m11 - m01 * m10
    angle = int(round(np.degrees(np.arctan2(m10, m00))))
    if det > 0:
        return {90: "turned 90° anticlockwise", -90: "turned 90° clockwise",
                180: "turned 180°", -180: "turned 180°"}.get(angle, f"turned {angle}°")
    return "mirrored" + ("" if angle == 0 else f" and turned {angle}°")
