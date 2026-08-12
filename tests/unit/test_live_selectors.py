"""Selection tools must not crash when a canvas has not been laid out yet."""

from matplotlib.figure import Figure

from src.gui.panels.live_selectors import _axes_ready, make_lasso_selector, make_rect_selector

_PALETTE = {"text_primary": "k", "accent": "r", "bg": "w"}


def test_axes_ready_rejects_zero_size_figure():
    fig = Figure(figsize=(0.0, 0.0))
    ax = fig.add_subplot(111)
    assert _axes_ready(ax) is False


def test_axes_ready_accepts_positive_size_without_canvas():
    fig = Figure(figsize=(4.0, 3.0))
    ax = fig.add_subplot(111)
    # No Qt canvas attached — size alone is enough to treat the axes as ready.
    assert _axes_ready(ax) is True


def test_make_selectors_return_none_on_zero_size_figure():
    fig = Figure(figsize=(0.0, 0.0))
    ax = fig.add_subplot(111)
    assert make_rect_selector(ax, owner=None, palette=_PALETTE) is None
    assert make_lasso_selector(ax, owner=None, palette=_PALETTE) is None
