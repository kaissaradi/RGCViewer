"""A redraw must not blit (PLAN.md open defects: macOS segfault on a tree drag).

The draw_event handler ran canvas.blit(), a synchronous QWidget.repaint().
A draw can run inside the canvas's own paintEvent, so that nested a paint in
a paint: "QPainter::begin: Paint device returned engine == 0, type: 1" and,
on macOS, a segfault. umap_panel was fixed this way in f883d6c. The highlight
must still be in the redrawn image.
"""

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


def _rf(monkeypatch):
    from src.gui.panels import rf_map_widget as m
    monkeypatch.setattr(m, "collect_rf_ellipses", lambda dm, ids: {
        c: (10.0 * i, 0.0, 8.0, 6.0, 0.0) for i, c in enumerate(ids)})
    view = m.RFMapWidget()
    view.resize(500, 400)
    view.show()
    view.set_cells(object(), [1, 2, 3])
    return view


def _trace(monkeypatch):
    from src.gui.panels.trace_stack_widget import TraceStackWidget
    view = TraceStackWidget()
    view.resize(500, 400)
    view.show()
    t = np.linspace(0, 6, 200)
    view.set_traces(np.vstack([np.sin(t + k) for k in range(3)]), [1, 2, 3])
    return view


def _orange_pixels(canvas):
    buf = np.asarray(canvas.buffer_rgba())
    r, g, b = buf[..., 0].astype(int), buf[..., 1].astype(int), buf[..., 2].astype(int)
    return int(((r - b > 100) & (r > 150) & (g < r - 40)).sum())


@pytest.mark.parametrize("make", [_rf, _trace])
def test_redraw_stamps_highlight_without_blit(qapp, monkeypatch, make):
    view = make(monkeypatch)
    qapp.processEvents()
    view.canvas.draw()
    assert _orange_pixels(view.canvas) == 0
    view.highlight([2])
    qapp.processEvents()

    blits = []
    real = view.canvas.blit
    monkeypatch.setattr(view.canvas, "blit", lambda *a, **k: blits.append(1) or real(*a, **k))
    view.canvas.draw()
    assert blits == []
    assert _orange_pixels(view.canvas) > 0
    view.close()
