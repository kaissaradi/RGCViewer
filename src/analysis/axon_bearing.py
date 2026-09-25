"""Which way the retina lies: axon directions from EIs point to the optic disc (PLAN.md Q43).

An RGC's electrical image shows its spike leaving the soma and travelling
down the axon. RGC axons run to the optic disc, so the axons of many cells
point at one place. Measured 2026-09-25 (PLAN.md Q43): on 20260220A/data022
(512 array) 85 % of well-fitted axons point within ±20° of one point about
1 mm off the array (edge-aware null 36 %), and a second run of the same piece
puts it 70 µm away; on a 519 array and on another prep the directions agree
but no point can be placed. So the result is graded:

* "disc"      — a point, when enough cells pass, the held-out fit beats the
                edge-aware null and the bootstrap distance is bounded;
* "direction" — only a bearing, when the directions agree more than the
                edge-aware null allows;
* "none"      — neither.

It gives the direction of the optic disc, not dorsal / ventral: that needs
where the piece came from in the eye.

The per-cell method is the one measured: soma = largest trough; axon
electrodes ≥ 2 % of the soma amplitude and above background, connected to
the soma, > 100 µm away, lagging it by 0.1–5 ms; a robust fit of position
against latency gives direction and speed; quality = R² of that fit.
Positions are in array µm (the Vision electrode map); ``array_orientation``
turns a bearing into screen terms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

FS_KHZ = 20.0            # Vision EI samples per ms
BG_FAR_UM = 300.0        # background: electrodes farther than this from the soma
BG_K = 6.0               # threshold = background median + BG_K * MAD
MIN_FRAC_SOMA = 0.02     # and at least 2 % of the soma trough
LINK_PITCH = 1.6         # neighbours within 1.6 electrode pitches are connected
AXON_MIN_D_UM = 100.0
LAT_MIN_MS, LAT_MAX_MS = 0.1, 5.0
MIN_AXON_EL = 5
MIN_EXTENT_UM = 150.0
GOOD_R2 = 0.7            # a cell's axon counts toward the bearing above this fit quality
MIN_GOOD_CELLS = 20
TOL_DEG = 20.0
MIN_SPIKES = 100
BASELINE = slice(0, 41)  # 2 ms before the trigger


# --- per cell ----------------------------------------------------------------------

def ei_features(ei: np.ndarray):
    """(amin, tmin, noise) per electrode: trough depth, sub-sample trough time, baseline SD."""
    ei = np.asarray(ei, dtype=np.float64)
    n_el, n_t = ei.shape
    x = ei - np.median(ei[:, BASELINE], axis=1, keepdims=True)
    noise = 1.4826 * np.median(np.abs(x[:, BASELINE]), axis=1)
    i = np.argmin(x, axis=1)
    ar = np.arange(n_el)
    ic = np.clip(i, 1, n_t - 2)
    y0, y1, y2 = x[ar, ic - 1], x[ar, ic], x[ar, ic + 1]
    den = y0 - 2 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        off = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0)
    off = np.clip(off, -0.5, 0.5)
    off[(i == 0) | (i == n_t - 1)] = 0.0
    return -x[ar, i], i + off, noise


def pitch_of(pos) -> float:
    from scipy.spatial import cKDTree
    d, _ = cKDTree(pos).query(pos, k=2)
    return float(np.median(d[:, 1]))


def _tukey_fit(X, Y, n_iter=20):
    w = np.ones(len(X))
    beta = None
    for _ in range(n_iter):
        sw = np.sqrt(w)[:, None]
        beta, *_ = np.linalg.lstsq(X * sw, (Y if Y.ndim == 2 else Y[:, None]) * sw, rcond=None)
        res = np.linalg.norm((Y if Y.ndim == 2 else Y[:, None]) - X @ beta, axis=1)
        u = res / (4.685 * (1.4826 * np.median(res) + 1e-6))
        new = np.where(u < 1, (1 - u ** 2) ** 2, 0.0)
        if new.sum() < 3:
            break
        w = new
    return beta, w


def cell_axon(amin, tmin, pos, pitch, bad=None) -> dict:
    """Axon estimate for one cell: ``found`` with direction ``u``, speed, ``r2``."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    a = np.asarray(amin, float).copy()
    if bad is not None:
        a[bad] = 0.0
    s = int(np.argmax(a))
    ps = pos[s]
    d = np.hypot(*(pos - ps).T)
    far = (d > BG_FAR_UM) & (a > 0 if bad is None else ~bad)
    out = {"found": False, "soma_xy": ps}
    if far.sum() < 5:
        out["reason"] = "no background"
        return out
    med = np.median(a[far])
    thr = max(med + BG_K * 1.4826 * np.median(np.abs(a[far] - med)), MIN_FRAC_SOMA * a[s])
    lat = (np.asarray(tmin, float) - tmin[s]) / FS_KHZ
    idx = np.where(a > thr)[0]
    if s not in idx:
        out["reason"] = "soma below threshold"
        return out
    pairs = cKDTree(pos[idx]).query_pairs(LINK_PITCH * pitch, output_type="ndarray")
    n = len(idx)
    if len(pairs):
        g = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        _, lab = connected_components(g, directed=False)
    else:
        lab = np.arange(n)
    comp = idx[lab == lab[np.where(idx == s)[0][0]]]
    ax = comp[(d[comp] > AXON_MIN_D_UM) & (lat[comp] > LAT_MIN_MS) & (lat[comp] < LAT_MAX_MS)]
    out["n_axon"] = len(ax)
    if len(ax) < MIN_AXON_EL:
        out["reason"] = f"only {len(ax)} axon electrodes"
        return out
    extent = d[ax].max() - d[ax].min()
    if extent < MIN_EXTENT_UM:
        out["reason"] = f"axon only {extent:.0f} µm long"
        return out
    P, L = pos[ax] - ps, lat[ax]
    X = np.column_stack([np.ones_like(L), L])
    beta, _w = _tukey_fit(X, P)
    r0, v = beta[0], beta[1]
    speed = float(np.linalg.norm(v))                      # µm / ms = mm / s
    pred = r0[None, :] + L[:, None] * v[None, :]
    r2 = 1 - np.sum((P - pred) ** 2) / np.sum((P - P.mean(0)) ** 2)
    out.update(found=True, u=v / (speed + 1e-12), speed_m_s=speed / 1000.0, r2=float(r2),
               reach=float(d[ax].max()))
    return out


def dedupe(amin_rows: np.ndarray, cos_thr=0.9) -> np.ndarray:
    """Same soma electrode and near-identical amplitude profile: keep the larger."""
    amp = amin_rows.max(1)
    soma = amin_rows.argmax(1)
    keep = np.ones(len(amp), bool)
    order = np.argsort(-amp)
    for k, i in enumerate(order):
        if not keep[i]:
            continue
        for j in order[k + 1:]:
            if keep[j] and soma[j] == soma[i]:
                a, b = amin_rows[i].clip(0), amin_rows[j].clip(0)
                if a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12) > cos_thr:
                    keep[j] = False
    return keep


# --- many cells: one point or one bearing -----------------------------------------------

def _angles_to_points(P, U, X):
    V = X[:, None, :] - P[None, :, :]
    cos = (V[..., 0] * U[None, :, 0] + V[..., 1] * U[None, :, 1]) / (np.linalg.norm(V, axis=2) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _grid_point(P, U, center, half=8000, step=200):
    g = np.arange(-half, half + 1, step)
    gx, gy = np.meshgrid(g + center[0], g + center[1])
    G = np.column_stack([gx.ravel(), gy.ravel()])
    ang = _angles_to_points(P, U, G)
    inl = ang < TOL_DEG
    score = inl.sum(1) - (np.where(inl, ang, 0).sum(1) / np.maximum(inl.sum(1), 1)) / (10 * TOL_DEG)
    k = int(np.argmax(score))
    return G[k], float(inl[k].mean())


def _angular_point(P, U, X0, c_deg=30.0):
    from scipy.optimize import minimize

    def f(x):
        a = _angles_to_points(P, U, x[None])[0] / c_deg
        return np.sum(np.where(a < 1, 1 - (1 - a ** 2) ** 3, 1.0))
    X0 = np.asarray(X0, float)
    simplex = np.array([X0, X0 + [150.0, 0.0], X0 + [0.0, 150.0]])
    return minimize(f, X0, method="Nelder-Mead",
                    options={"xatol": 1.0, "fatol": 1e-4, "maxiter": 2000,
                             "initial_simplex": simplex}).x


def _feasible(P, reach, pos, pitch, n_ang=360):
    """Directions along which an axon of that reach would stay on the array."""
    from scipy.spatial import cKDTree
    tree = cKDTree(pos)
    phis = np.radians(np.arange(n_ang) * 360.0 / n_ang)
    D = np.column_stack([np.cos(phis), np.sin(phis)])
    M = np.zeros((len(P), n_ang), bool)
    for i, (p, r) in enumerate(zip(P, reach)):
        steps = np.arange(pitch / 2, r + 1e-6, pitch / 2)
        pts = p[None, None, :] + steps[None, :, None] * D[:, None, :]
        dd, _ = tree.query(pts.reshape(-1, 2))
        M[i] = np.all(dd.reshape(n_ang, len(steps)) <= 0.8 * pitch, axis=1)
    return M, phis


def _sample(M, phis, rng):
    out = np.zeros((M.shape[0], 2))
    for i in range(M.shape[0]):
        ok = np.where(M[i])[0]
        j = rng.choice(ok) if len(ok) else rng.integers(len(phis))
        out[i] = [np.cos(phis[j]), np.sin(phis[j])]
    return out


def _resultant(U):
    th = np.arctan2(U[:, 1], U[:, 0])
    return float(np.hypot(np.cos(th).mean(), np.sin(th).mean())), float(np.degrees(np.arctan2(
        np.sin(th).mean(), np.cos(th).mean())))


@dataclass
class Bearing:
    verdict: str                          # "disc" / "direction" / "none"
    n_cells: int
    n_axons: int
    n_good: int
    speed_m_s: float = float("nan")
    bearing_deg: float = float("nan")     # array frame, from the array centre (disc) or mean direction
    bearing_ci: tuple = (float("nan"), float("nan"))
    disc_xy: Optional[tuple] = None       # array µm, only for "disc"
    distance_um: float = float("nan")
    distance_ci: tuple = (float("nan"), float("nan"))
    within_tol: float = float("nan")      # share of good axons pointing within ±20°
    heldout: float = float("nan")
    null_edge_97: float = float("nan")
    resultant: float = float("nan")
    resultant_null_p: float = float("nan")
    cells: List[dict] = field(default_factory=list, repr=False)   # good cells: soma_xy, u, r2
    reason: str = ""

    def sentence(self) -> str:
        if self.verdict == "disc":
            return (f"The optic disc lies {self.distance_um / 1000:.1f} mm from the array centre, "
                    f"bearing {self.bearing_deg:.0f}° (95 %: {self.bearing_ci[0]:.0f}–"
                    f"{self.bearing_ci[1]:.0f}°), from {self.n_good} axons "
                    f"({self.within_tol:.0%} point within ±20°).")
        if self.verdict == "direction":
            return (f"The axons run toward bearing {self.bearing_deg:.0f}° "
                    f"(95 %: {self.bearing_ci[0]:.0f}–{self.bearing_ci[1]:.0f}°, {self.n_good} axons), "
                    f"but no disc point can be placed. Direction only, low confidence.")
        return f"No reliable direction: {self.reason}"


def estimate(cells: Dict[int, dict], pos, n_null=100, n_boot=100, seed=0,
             progress: Optional[Callable[[str], None]] = None) -> Bearing:
    """``cells``: {id: {"amin", "tmin", "n_spikes"}} from ei_features. Positions: array µm."""
    rng = np.random.default_rng(seed)
    pos = np.asarray(pos, float)
    pitch = pitch_of(pos)
    center = 0.5 * (pos.min(0) + pos.max(0))
    ids = sorted(cells)
    if not ids:
        return Bearing("none", 0, 0, 0, reason="no EIs")
    amin = np.array([cells[i]["amin"] for i in ids])
    noise = np.array([cells[i].get("noise", np.ones(len(pos))) for i in ids])
    bad = np.median(noise, 0) < 1e-6                       # flat, disconnected electrodes
    keep = dedupe(np.where(bad[None, :], 0, amin)) & \
        np.array([cells[i].get("n_spikes", MIN_SPIKES) >= MIN_SPIKES for i in ids])
    fits = [cell_axon(cells[i]["amin"], cells[i]["tmin"], pos, pitch, bad)
            for i, k in zip(ids, keep) if k]
    found = [f for f in fits if f["found"]]
    good = [f for f in found if f["r2"] >= GOOD_R2]
    b = Bearing("none", int(keep.sum()), len(found), len(good),
                speed_m_s=float(np.median([f["speed_m_s"] for f in good])) if good else float("nan"),
                cells=[{"soma_xy": tuple(f["soma_xy"]), "u": tuple(f["u"]), "r2": f["r2"]} for f in good])
    if len(good) < MIN_GOOD_CELLS:
        b.reason = f"only {len(good)} axons pass the quality cut (need {MIN_GOOD_CELLS})."
        return b
    if progress:
        progress("fitting the disc point")
    P = np.array([f["soma_xy"] for f in good], float)
    U = np.array([f["u"] for f in good], float)
    n = len(P)
    Xg, _frac = _grid_point(P, U, center)
    X = _angular_point(P, U, Xg)
    b.within_tol = float(np.mean(_angles_to_points(P, U, X[None])[0] < TOL_DEG))
    # edge-aware null: directions an axon of that reach could take on this array
    M, phis = _feasible(P, np.array([f["reach"] for f in good]), pos, pitch)
    heldout, null_heldout, null_R = [], [], []
    b.resultant, mean_dir = _resultant(U)
    for _ in range(n_null):
        idx = rng.permutation(n)
        a, c = idx[: n // 2], idx[n // 2:]
        Xa, _ = _grid_point(P[a], U[a], center)
        heldout.append(np.mean(_angles_to_points(P[c], U[c], Xa[None])[0] < TOL_DEG))
        Ue = _sample(M, phis, rng)
        Xe, _ = _grid_point(P[a], Ue[a], center)
        null_heldout.append(np.mean(_angles_to_points(P[c], Ue[c], Xe[None])[0] < TOL_DEG))
        null_R.append(_resultant(Ue)[0])
    b.heldout = float(np.mean(heldout))
    b.null_edge_97 = float(np.percentile(null_heldout, 97.5))
    b.resultant_null_p = float((1 + np.sum(np.array(null_R) >= b.resultant)) / (1 + n_null))
    if progress:
        progress("bootstrap")
    bear, dist = [], []
    for _ in range(n_boot):
        idx = rng.integers(n, size=n)
        Xb0, _ = _grid_point(P[idx], U[idx], center)
        v = _angular_point(P[idx], U[idx], Xb0) - center
        bear.append(np.degrees(np.arctan2(v[1], v[0])))
        dist.append(np.linalg.norm(v))
    v = X - center
    b.bearing_deg = float(np.degrees(np.arctan2(v[1], v[0])))
    wrapped = (np.array(bear) - b.bearing_deg + 180) % 360 - 180 + b.bearing_deg
    b.bearing_ci = (float(np.percentile(wrapped, 2.5)), float(np.percentile(wrapped, 97.5)))
    b.distance_um = float(np.linalg.norm(v))
    b.distance_ci = (float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5)))
    bounded = (b.distance_ci[1] - b.distance_ci[0]) < 0.6 * b.distance_um
    if b.heldout > b.null_edge_97 and bounded:
        b.verdict, b.disc_xy = "disc", (float(X[0]), float(X[1]))
        return b
    if b.resultant_null_p < 0.05:
        b.verdict = "direction"
        b.bearing_deg = mean_dir
        th = np.degrees(np.arctan2(U[:, 1], U[:, 0]))
        boot = [_resultant(U[rng.integers(n, size=n)])[1] for _ in range(n_boot)]
        w = (np.array(boot) - mean_dir + 180) % 360 - 180 + mean_dir
        b.bearing_ci = (float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5)))
        del th
        return b
    b.reason = "the axon directions agree no more than random ones on this array would."
    return b


def read_run_eis(folder, dataset, progress: Optional[Callable[[int, int], None]] = None):
    """{vision id: features} and electrode positions, reading the .ei once in file order."""
    from . import visionloader as vl
    r = vl.EIReader(str(folder), dataset)
    try:
        ids = sorted(r.cell_id_to_offset, key=lambda c: r.cell_id_to_offset[c])
        out = {}
        for k, cid in enumerate(ids):
            e = r.get_ei_for_cell_id(cid)
            amin, tmin, noise = ei_features(e.ei)
            out[int(cid)] = {"amin": amin, "tmin": tmin, "noise": noise, "n_spikes": int(e.n_spikes)}
            if progress and k % 50 == 0:
                progress(k, len(ids))
        return out, np.asarray(r.electrode_map, float)
    finally:
        r.close()
