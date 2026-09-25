"""A hidden (0×0) RF map must not raise on draw; it draws once it has a size.

errors.log 2026-09-25: the UMAP's RF map drew while hidden and raised
"'box_aspect' and 'fig_aspect' must be positive", which stopped the UMAP
finishing and the theme toggle (every panel after UMAP kept the old colours).
"""

from src.gui.panels import rf_map_widget as rmw


def test_hidden_map_defers_its_draw(qtbot, monkeypatch):
    ellipses = {c: (10.0 * c, 5.0 * c, 8.0, 6.0, 0.0) for c in range(1, 6)}
    monkeypatch.setattr(rmw, "collect_rf_ellipses", lambda dm, ids: {c: ellipses[c] for c in ids})
    w = rmw.RFMapWidget()
    qtbot.addWidget(w)
    w.canvas.resize(0, 0)                       # hidden / not laid out
    w.fig.set_size_inches(0, 0)
    w.set_cells(None, list(ellipses))           # used to raise ValueError
    assert w._pending_draw
    w.restyle({"bg": "#ffffff"})                # the theme toggle path, same draw
    w.show()
    w.resize(400, 300)
    qtbot.waitUntil(lambda: not w._pending_draw, timeout=3000)
