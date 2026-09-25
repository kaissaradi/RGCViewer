"""Axon directions and the optic-disc estimate on synthetic data with known answers (PLAN.md Q43)."""

import numpy as np

from src.analysis import axon_bearing as ab


def _array(pitch=60.0, nx=32, ny=16):
    """A 512-like hexagonal-ish grid, µm, centred on 0."""
    xs, ys = np.meshgrid(np.arange(nx) * pitch, np.arange(ny) * pitch * 0.87)
    xs = xs + (np.arange(ny)[:, None] % 2) * pitch / 2
    pos = np.column_stack([xs.ravel(), ys.ravel()])
    return pos - pos.mean(0)


def _cell(pos, soma, u, rng, speed_um_ms=1000.0):
    """amin / tmin of one cell with an axon leaving ``soma`` along ``u``."""
    n = len(pos)
    V = pos - soma
    along = V @ u
    perp = np.abs(V[:, 0] * u[1] - V[:, 1] * u[0])
    amin = np.abs(rng.normal(0.3, 0.15, n))
    tmin = rng.uniform(0, 200, n)
    t0 = 67.0
    on = (along > 0) & (perp < 35)
    amin[on] = 15 * np.exp(-along[on] / 2000) + rng.normal(0, 1, on.sum()).clip(-3, 3)
    tmin[on] = t0 + along[on] / speed_um_ms * ab.FS_KHZ + rng.normal(0, 0.4, on.sum())
    ds = np.hypot(*V.T)
    near = ds < 70
    amin[near] = 300 * np.exp(-ds[near] / 40)
    tmin[near] = t0
    amin[np.argmin(ds)] = 400
    tmin[np.argmin(ds)] = t0
    return {"amin": amin, "tmin": tmin, "noise": np.ones(n), "n_spikes": 5000}


def test_one_cell_direction_and_speed():
    rng = np.random.default_rng(0)
    pos = _array()
    u = np.array([np.cos(0.4), np.sin(0.4)])
    c = _cell(pos, pos[np.argmin(np.hypot(*(pos - [-600, -200]).T))], u, rng, speed_um_ms=800)
    fit = ab.cell_axon(c["amin"], c["tmin"], pos, ab.pitch_of(pos))
    assert fit["found"] and fit["r2"] > 0.9
    assert np.degrees(np.arccos(np.clip(fit["u"] @ u, -1, 1))) < 5
    assert abs(fit["speed_m_s"] - 0.8) < 0.08


def _run(pos, directions_for, n=40, seed=1):
    rng = np.random.default_rng(seed)
    cells = {}
    for i in range(n):
        soma = pos[rng.integers(len(pos))]
        cells[i] = _cell(pos, soma, directions_for(soma, rng), rng)
    return cells


def test_axons_toward_one_point_give_the_disc():
    pos = _array()
    disc = np.array([0.0, 1500.0])                       # off the array, "up"

    def toward(soma, rng):
        v = disc - soma
        th = np.arctan2(v[1], v[0]) + np.radians(rng.normal(0, 5))
        return np.array([np.cos(th), np.sin(th)])
    b = ab.estimate(_run(pos, toward), pos, n_null=40, n_boot=40)
    assert b.verdict == "disc", b.sentence()
    assert np.hypot(b.disc_xy[0] - disc[0], b.disc_xy[1] - disc[1]) < 400
    assert 60 < b.bearing_deg < 120 and b.within_tol > 0.8


def test_random_axons_give_no_disc():
    pos = _array()

    def anywhere(_soma, rng):
        th = rng.uniform(0, 2 * np.pi)
        return np.array([np.cos(th), np.sin(th)])
    b = ab.estimate(_run(pos, anywhere, seed=2), pos, n_null=40, n_boot=40)
    assert b.verdict != "disc"
    assert "No reliable" in b.sentence() or "Direction only" in b.sentence()


def test_too_few_axons_says_so():
    pos = _array()
    b = ab.estimate({}, pos)
    assert b.verdict == "none" and b.reason


def test_dialog_turns_the_array_like_the_ei_panel(qtbot):
    from src.gui.main_window import MainWindow
    from src.gui.panels import optic_disc_dialog as od
    assert od._screen_words([0, 1]) == "up" and od._screen_words([1, 0.1]) == "to the right"
    b = ab.Bearing("disc", 100, 60, 30, speed_m_s=0.6, bearing_deg=90.0, bearing_ci=(85, 95),
                   disc_xy=(0.0, 1000.0), distance_um=1000.0, within_tol=0.85,
                   cells=[{"soma_xy": (0.0, 0.0), "u": (0.0, 1.0), "r2": 0.9}])
    w = MainWindow()
    try:
        dlg = od.OpticDiscDialog(w, b, _array(), (0, 1, -1, 0))   # the rig's quarter turn
        qtbot.addWidget(dlg)
        text = dlg.findChildren(od.QLabel)[0].text()
        assert "to the right" in text and "Dorsal vs ventral" in text   # array "up" = screen right
    finally:
        w.data_manager = None
        w.close()
        w.deleteLater()
