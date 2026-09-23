"""Pin the Vision RF-ellipse convention for every drawing path.

The convention (analysis/rf_geometry.py) was measured on real files: x0 =
col + 0.5, H - y0 = row + 0.5, and the SigmaX axis lies at +Theta in image
(x, row) coordinates, i.e. at -Theta in Vision's y-up frame. These tests
build a synthetic elongated STA spot that obeys it and check that each path
draws its ellipse along the spot, in the frame that path displays.
"""

import math
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

from src.analysis import rf_geometry
from src.analysis.visionloader import STAFit

H, W = 40, 60
ROW, COL = 17, 29          # spot centre pixel
SX, SY = 6.0, 2.0          # SigmaX is the long axis


def _spot(theta):
    """Image (rows, cols) with a Gaussian whose long axis is +theta in (x, row)."""
    r, c = np.indices((H, W)) + 0.5            # pixel centres, edges on integers
    dx, dy = c - (COL + 0.5), r - (ROW + 0.5)
    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)
    return np.exp(-0.5 * ((u / SX) ** 2 + (v / SY) ** 2))


def _stafit(theta):
    # What Vision stores for that spot.
    return STAFit(COL + 0.5, H - (ROW + 0.5), SX, SY, theta)


def _moments(x, y, w=None):
    w = np.ones_like(x) if w is None else w
    mx, my = (w * x).sum() / w.sum(), (w * y).sum() / w.sum()
    X, Y = x - mx, y - my
    phi = 0.5 * math.atan2(2 * (w * X * Y).sum(), (w * X * X).sum() - (w * Y * Y).sum())
    return phi, mx, my


def _axis_err(a, b):
    e = abs((a - b) % math.pi)
    return math.degrees(min(e, math.pi - e))


def _spot_moments(theta, frame):
    img = _spot(theta)
    r, c = np.indices(img.shape)
    x, y = c + 0.5, r + 0.5                    # (x, row), edges on integers
    if frame == "rows_centre":
        x, y = x - 0.5, y - 0.5
    elif frame == "yup":
        y = H - y
    return _moments(x.ravel(), y.ravel(), img.ravel())


THETAS = [math.radians(a) for a in (20.0, 70.0, 125.0, 160.0)]


@pytest.mark.parametrize("theta", THETAS)
def test_sta_panel_outline_lies_along_the_spot(theta):
    """STA panel: pyqtgraph ImageItem, invertY, pixel edges on integers."""
    fit = rf_geometry.rf_fit_from_stafit(_stafit(theta))
    xe, ye = rf_geometry.ellipse_outline(*rf_geometry.image_ellipse(fit, H))
    phi, mx, my = _spot_moments(theta, "rows_edge")
    th, cx, cy = _moments(xe[:-1], ye[:-1])
    assert _axis_err(th, phi) < 2.0
    assert math.hypot(cx - mx, cy - my) < 0.05


@pytest.mark.parametrize("theta", THETAS)
def test_mirrored_sign_would_fail(theta):
    """Guard the test itself: the old negated angle must NOT pass."""
    fit = rf_geometry.rf_fit_from_stafit(_stafit(theta))
    cx, cy, w, h, ang = rf_geometry.image_ellipse(fit, H)
    xe, ye = rf_geometry.ellipse_outline(cx, cy, w, h, -ang)
    phi, _, _ = _spot_moments(theta, "rows_edge")
    th, _, _ = _moments(xe[:-1], ye[:-1])
    assert _axis_err(th, phi) > 20.0


def test_imshow_frame_is_half_a_pixel_up_left():
    fit = rf_geometry.rf_fit_from_stafit(_stafit(0.3))
    a = rf_geometry.image_ellipse(fit, H)
    b = rf_geometry.image_ellipse(fit, H, pixel_centres_on_integers=True)
    assert np.allclose(np.subtract(a[:2], b[:2]), 0.5)
    assert b[:2] == (COL, ROW)


def _render_pixels(draw):
    fig = Figure(figsize=(4, 4))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    draw(ax)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(int)
    rr, cc = np.nonzero(buf.sum(-1) < 200)
    xy = ax.transData.inverted().transform(np.c_[cc + 0.5, buf.shape[0] - rr - 0.5])
    return _moments(xy[:, 0], xy[:, 1])


@pytest.mark.parametrize("theta", THETAS)
def test_mosaic_collection_lies_along_the_spot(theta):
    """Mosaic: EllipseCollection on ascending axes, Vision's y-up frame."""
    from src.gui.panels.population_panel import _build_ellipse_collection

    fit = rf_geometry.rf_fit_from_stafit(_stafit(theta))

    def draw(ax):
        ec = _build_ellipse_collection([rf_geometry.mosaic_ellipse(fit)], "none", 1, 1, 1)
        ax.add_collection(ec)
        ec.set_offset_transform(ax.transData)
        ec.set_facecolor("k")

    th, cx, cy = _render_pixels(draw)
    phi, mx, my = _spot_moments(theta, "yup")
    assert _axis_err(th, phi) < 3.0
    assert math.hypot(cx - mx, cy - my) < 0.3


def _fake_dm(theta):
    vp = MagicMock()
    vp.runtimemovie_params = None
    vp.get_stafit_for_cell.side_effect = lambda vid: _stafit(theta)
    vp.get_cell_ids.return_value = [1]
    img = _spot(theta)[:, :, None].repeat(3, axis=2)
    sta = SimpleNamespace(red=img, green=img, blue=img)
    return SimpleNamespace(
        vision_params=vp, vision_stas={1: sta}, vision_sta_height=H,
        reference_bridge=None, is_vision_only=False,
        get_vision_id_for_cluster=lambda cid: int(cid) + 1,
    )


@pytest.mark.parametrize("theta", THETAS)
def test_highlight_patch_lies_along_the_spot(theta):
    from src.gui.panels.population_panel import _update_highlight_patch

    dm = _fake_dm(theta)

    def draw(ax):
        patch = Ellipse(xy=(0, 0), width=1, height=1, angle=0, facecolor="k", edgecolor="none")
        ax.add_patch(patch)
        _update_highlight_patch(patch, dm, 0)
        assert patch.get_visible()

    th, cx, cy = _render_pixels(draw)
    phi, mx, my = _spot_moments(theta, "yup")
    assert _axis_err(th, phi) < 3.0
    assert math.hypot(cx - mx, cy - my) < 0.3


def test_rf_map_widget_uses_the_mosaic_frame():
    from src.gui.panels.rf_map_widget import collect_rf_ellipses

    theta = THETAS[1]
    got = collect_rf_ellipses(_fake_dm(theta), [0])[0]
    want = rf_geometry.mosaic_ellipse(rf_geometry.rf_fit_from_stafit(_stafit(theta)))
    assert np.allclose(got, want)


@pytest.mark.parametrize("theta", THETAS)
def test_export_sta_figure_lies_along_the_spot(theta):
    """Export: imshow(origin='upper'), pixel centres on integers."""
    from src.gui.plot_export import figure_sta

    out = figure_sta(_fake_dm(theta), 0, "g")
    fig = out[0] if isinstance(out, tuple) else out
    FigureCanvasAgg(fig)
    fig.canvas.draw()
    ax = fig.axes[0]
    el = [p for p in ax.patches if isinstance(p, Ellipse)][0]
    xy = ax.transData.inverted().transform(el.get_verts())
    th, cx, cy = _moments(xy[:-1, 0], xy[:-1, 1])
    phi, mx, my = _spot_moments(theta, "rows_centre")
    assert _axis_err(th, phi) < 2.0
    assert math.hypot(cx - mx, cy - my) < 0.1


def test_raw_fit_undoes_the_runtimemovie_flip():
    """get_stafit_for_cell moves y when runtime-movie params are loaded."""
    stored = _stafit(0.4)
    rtmp = SimpleNamespace(height=H)
    loaded = STAFit(stored.center_x + 0.5, H - stored.center_y + 0.5,
                    stored.std_x, stored.std_y, stored.rot)
    fit = rf_geometry.rf_fit_from_stafit(loaded, rtmp)
    assert np.allclose(fit, (stored.center_x, stored.center_y, SX, SY, 0.4))


def test_unusable_fits_are_none():
    assert rf_geometry.rf_fit_from_stafit(STAFit(1, 2, 0, 1, 0.1)) is None
    assert rf_geometry.rf_fit_from_stafit(STAFit(1, np.nan, 1, 1, 0.1)) is None
    assert rf_geometry.rf_fit_from_params({"x0": 1, "y0": 2}) is None
