"""The one place a Vision RF fit becomes drawable ellipse geometry.

Vision's convention, measured 2026-09-23 against STA image moments on a
Java-written .params (20260511A/data007) and an MEA-written one
(20260603A/data002), elongated cells only:

* ``x0``/``y0`` are stixels with the pixel EDGES on integers and y pointing
  UP: ``x0 = col + 0.5`` and ``H - y0 = row + 0.5`` (median offsets +0.49 /
  +0.51 on the Java file), where ``row`` indexes the STA array's first axis.
* ``Theta`` is radians. In that y-up frame the SigmaX axis lies at
  ``-Theta``; in image (x, row) coordinates it lies at ``+Theta``. That fits
  the STA spot to a 3.5 deg median error on the Java file (the opposite sign
  gives 39 deg).

Every panel should draw through these helpers so the STA view, the mosaic,
the highlight, the RF map and the exports cannot disagree again. Two frames:

* the MOSAIC frame is Vision's own y-up frame, drawn on normal (ascending)
  axes. It looks the same way round as the STA image.
* the IMAGE frame is (x, row) on axes whose y runs downward (pyqtgraph
  ``invertY(True)``, or matplotlib ``imshow(origin='upper')``).
"""

from collections import namedtuple

import numpy as np

RFFit = namedtuple("RFFit", ["x0", "y0", "std_x", "std_y", "theta"])

# vision7 ImageFrame.fit starts the Gaussian at every (σx, σy) on a grid
# from 1 to 3 in steps of 0.5 and keeps the best result. A fit that never
# moved from its first start stays at exactly (1, 1); it is not a
# measurement. 2026-09-24: 23/303 cells on 20251212A/data018, 25/578 on
# 20260220A/data022, 351/584 on 20260529A/data017 (PLAN.md Q26).
UNMOVED_SIGMA = 1.0


def fit_is_unmoved(std_x, std_y):
    """True when σx and σy are both exactly Vision's first start value."""
    try:
        return (abs(abs(float(std_x)) - UNMOVED_SIGMA) < 1e-9
                and abs(abs(float(std_y)) - UNMOVED_SIGMA) < 1e-9)
    except (TypeError, ValueError):
        return False


def raw_rf_fit(vision_params, cell_id):
    """Return Vision's stored fit as an ``RFFit``, or None if unusable.

    ``get_stafit_for_cell`` moves the centre into a different frame when the
    table carries runtime-movie params, so undo that here: every caller gets
    the stored, y-up values whatever the table was loaded with.
    """
    if vision_params is None:
        return None
    try:
        fit = vision_params.get_stafit_for_cell(cell_id)
    except Exception:
        return None
    return rf_fit_from_stafit(fit, getattr(vision_params, "runtimemovie_params", None))


def rf_fit_from_stafit(fit, runtimemovie_params=None):
    """``STAFit`` (or anything with center_x/center_y/std_x/std_y/rot) -> RFFit."""
    if fit is None:
        return None
    try:
        x, y = float(fit.center_x), float(fit.center_y)
        sx, sy = float(fit.std_x), float(fit.std_y)
        rot = float(getattr(fit, "rot", getattr(fit, "angle", np.nan)))
    except (AttributeError, TypeError, ValueError):
        return None
    if runtimemovie_params is not None:
        # visionloader returned (x + 0.5, height - y + 0.5); invert it.
        x = x - 0.5
        y = float(runtimemovie_params.height) - (y - 0.5)
    if not np.all(np.isfinite([x, y, sx, sy, rot])) or sx <= 0 or sy <= 0:
        return None
    if fit_is_unmoved(sx, sy):
        return None
    return RFFit(x, y, sx, sy, rot)


def rf_fit_from_params(params):
    """A ``{x0, y0, std_x, std_y, angle}`` dict (reference bridge) -> RFFit."""
    if not params:
        return None
    try:
        vals = [float(params[k]) for k in ("x0", "y0", "std_x", "std_y", "angle")]
    except (KeyError, TypeError, ValueError):
        return None
    if not np.all(np.isfinite(vals)) or vals[2] <= 0 or vals[3] <= 0:
        return None
    if fit_is_unmoved(vals[2], vals[3]):
        return None
    return RFFit(*vals)


def mosaic_ellipse(fit):
    """``(cx, cy, width, height, angle_deg)`` in the y-up mosaic frame.

    For matplotlib ``Ellipse``/``EllipseCollection`` on ascending axes with
    an equal aspect ratio.
    """
    return (fit.x0, fit.y0, 2.0 * fit.std_x, 2.0 * fit.std_y,
            -float(np.degrees(fit.theta)))


def image_ellipse(fit, height, pixel_centres_on_integers=False):
    """``(cx, cy, width, height, angle_deg)`` in the (x, row) image frame.

    ``pixel_centres_on_integers`` is False for pyqtgraph's ImageItem (pixel i
    spans [i, i + 1]) and True for matplotlib ``imshow`` with its default
    extent (pixel i spans [i - 0.5, i + 0.5]). The angle is for axes whose y
    runs downward, where a data-space rotation of +Theta is correct.
    """
    shift = 0.5 if pixel_centres_on_integers else 0.0
    return (fit.x0 - shift, float(height) - fit.y0 - shift,
            2.0 * fit.std_x, 2.0 * fit.std_y, float(np.degrees(fit.theta)))


def ellipse_outline(cx, cy, width, height, angle_deg, n=120):
    """Outline points of an ellipse rotated by ``angle_deg`` in DATA space."""
    t = np.linspace(0.0, 2.0 * np.pi, n)
    a = np.radians(angle_deg)
    ex, ey = 0.5 * width * np.cos(t), 0.5 * height * np.sin(t)
    return (cx + ex * np.cos(a) - ey * np.sin(a),
            cy + ex * np.sin(a) + ey * np.cos(a))
