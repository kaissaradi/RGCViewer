"""
waveforms_panel.py  —  Encore Waveform & Isolation Panel
=============================================================
Architecture
------------
Three-tier progressive rendering so the user always sees *something* within
one frame of clicking a cluster:

  Tier 1 (< 5 ms, synchronous):
      Draw median trace from ei_cache (already warm from background worker).
      Populate stats from pre-computed values.

  Tier 2 (< 80 ms, QRunnable):
      Render shadow-cloud from raw_snippets (cached, no disk I/O, just math).
      5 amplitude-bucket CurveItems — 5 draw calls total regardless of n_spikes.

  Tier 3 (< 400 ms, QRunnable):
      Fetch channel-context data (templates or raw), run PCA, populate right panel.
      Computes isolation metrics (L-ratio, d-prime proxy).

Cancellation: every async result carries the cluster_id it was computed for.
If the user has moved on, the result is silently dropped.

Future hooks
------------
- PCA scatter stores global spike indices in item.data() → ready for lasso-split
- _on_lasso_selected(spike_indices) stub present → wire to DataManager.split_cluster()
- Neighbor-channel sparklines slot exists → fill with median_ei neighbour channels
- UMAP toggle slot → swap PCA worker for UMAP worker without touching UI
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, QRunnable, QThreadPool, QObject, Signal, Slot
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QFrame,
    QGridLayout,
    QPushButton,
)

if TYPE_CHECKING:
    from ..analysis.data_manager import DataManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Design tokens — single source of truth for all colours / sizes
# ---------------------------------------------------------------------------
_C = {
    # Backgrounds
    "bg_main": "#13151A",  # near-black, warm tint
    "bg_right": "#0F1116",  # slightly darker for right panel
    "bg_card": "#1C1F27",  # stat card surface
    "bg_card_hover": "#242830",
    # Borders
    "border_subtle": "#252830",
    "border_default": "#2E3240",
    # Text
    "text_primary": "#E8EAF0",
    "text_secondary": "#7A7E8E",
    "text_dim": "#4A4E5E",
    # Accent colours for waveform
    "median": "#FFD166",  # warm amber — the star of the show
    "envelope": "#3A4060",  # muted navy for 10-90 ribbon
    # Shadow-cloud amplitude colour ramp (5 buckets, low→high)
    "cloud_0": (60, 80, 160, 35),  # dim periwinkle
    "cloud_1": (80, 110, 200, 45),
    "cloud_2": (120, 150, 240, 55),
    "cloud_3": (170, 190, 255, 65),
    "cloud_4": (220, 230, 255, 80),  # bright near-white
    # PCA
    "pca_bg": (100, 105, 120, 55),  # background cloud
    "pca_unit": "#FFD166",  # selected unit (same amber)
    "pca_ellipse": "#FFD166",
    "pca_compare": "#FF6B9D",  # coral — comparison cluster
    "pca_cmp_ell": "#FF6B9D",
    # Badge colours
    "badge_good": "#2ECC71",
    "badge_warn": "#F39C12",
    "badge_bad": "#E74C3C",
    "badge_neutral": "#7A7E8E",
    # Zero-line
    "zero_line": "#2A2E3A",
    # Axis
    "axis_pen": "#3A3E50",
    "axis_text": "#5A5E70",
}


def _rgba(hex_color, alpha):
    h = str(hex_color).lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)


def palette_from_theme(colors):
    """``_C`` from the theme tokens (docs/design/palette.md).

    The panel used its own fixed dark palette, so light mode left the whole
    Waveforms tab near-black (PLAN.md Q16). Mean traces are ink, ensembles
    blue, the compared cell red, as on every other panel.
    """
    ens = colors["plot_ensemble"]
    return {
        "bg_main": colors["plot_bg"],
        "bg_right": colors["bg_panel"],
        "bg_card": colors["bg_elevated"],
        "bg_card_hover": colors["bg_elevated"],
        "border_subtle": colors["border_subtle"],
        "border_default": colors["border_default"],
        "text_primary": colors["text_primary"],
        "text_secondary": colors["text_secondary"],
        "text_dim": colors["text_disabled"],
        "median": colors["plot_mean"],
        "envelope": _rgba(ens, 30),
        "cloud_0": _rgba(ens, 35),
        "cloud_1": _rgba(ens, 45),
        "cloud_2": _rgba(ens, 55),
        "cloud_3": _rgba(ens, 65),
        "cloud_4": _rgba(ens, 80),
        "pca_bg": _rgba(colors["plot_shadow"], 120),   # unsorted crossings: visible, recessive
        "pca_unit": colors["plot_compare"],   # this cell red, as the selected cell everywhere
        "pca_ellipse": colors["plot_highlight"],
        "pca_compare": colors["plot_compare"],
        "pca_cmp_ell": colors["plot_compare"],
        "badge_good": colors["accent_positive"],
        "badge_warn": colors["status_mua_text"],
        "badge_bad": colors["status_noise_text"],
        "badge_neutral": colors["text_secondary"],
        "zero_line": colors["border_strong"],
        "axis_pen": colors["border_strong"],
        "axis_text": colors["text_secondary"],
    }

# Snippet window pre-spike samples (must match extract_snippets call in analysis_core)
_PRE_SAMPLES = 20

# PCA cache — LRU, max 30 entries
_PCA_CACHE: OrderedDict = OrderedDict()
_PCA_CACHE_MAX = 30


# ---------------------------------------------------------------------------
# Worker signals (must live on a QObject for Qt signal/slot to work)
# ---------------------------------------------------------------------------
class _WorkerSignals(QObject):
    cloud_ready = Signal(int, object)  # cluster_id, payload dict
    pca_ready = Signal(int, object)  # cluster_id, payload dict
    error = Signal(int, str)  # cluster_id, message


# ---------------------------------------------------------------------------
# Tier-2 worker: fetch raw spikes + build shadow cloud
# ---------------------------------------------------------------------------
# How many shadow traces to render and fetch from raw data
_N_CLOUD_SPIKES = 150


class _CloudWorker(QRunnable):
    """
    Fetches _N_CLOUD_SPIKES raw spike waveforms directly from the data source
    (bypassing ei_cache which may have been subsampled to ~30 for EI purposes),
    then builds the amplitude-bucketed shadow cloud.

    Falls back to whatever snippets were passed in if raw data is unavailable.
    """

    def __init__(
        self,
        cluster_id: int,
        dm,
        median_trace: np.ndarray,
        t_ms: np.ndarray,
        dom_chan: int,
        sr: float,
        fallback_snippets: Optional[np.ndarray],
        signals: _WorkerSignals,
    ):
        super().__init__()
        self.setAutoDelete(True)
        self.cluster_id = cluster_id
        self.dm = dm
        self.median_trace = median_trace
        self.t_ms = t_ms
        self.dom_chan = dom_chan
        self.sr = sr
        self.fallback_snippets = fallback_snippets  # (n_spikes, n_time) from ei_cache
        self.signals = signals

    @Slot()
    def run(self):
        try:
            waveforms = self._fetch_waveforms()
            if waveforms is None or len(waveforms) == 0:
                return
            payload = _build_cloud_payload(
                self.cluster_id, waveforms, self.median_trace, self.t_ms
            )
            payload["_generation"] = getattr(self.dm, "generation", None)
            self.signals.cloud_ready.emit(self.cluster_id, payload)
        except Exception as exc:
            logger.exception("CloudWorker error cid=%d", self.cluster_id)
            self.signals.error.emit(self.cluster_id, str(exc))

    def _fetch_waveforms(self) -> Optional[np.ndarray]:
        """
        Try to pull _N_CLOUD_SPIKES real spikes from raw data.
        Returns (n_spikes, n_time) float32 in µV, or falls back to cached snippets.
        """
        dm = self.dm
        raw_available = (
            getattr(dm, "raw_reader", None) is not None
            or getattr(dm, "raw_data_memmap", None) is not None
        )

        if raw_available:
            try:
                # Grab analysis_core from sys.modules — it's already imported
                # by data_manager.py, so this never fails and avoids the
                # relative-import error that occurs inside a bare QRunnable.
                import sys as _sys

                _ac_key = next(
                    (
                        k
                        for k in _sys.modules
                        if k.endswith("analysis_core") and "analysis_core" in k
                    ),
                    None,
                )
                if _ac_key is None:
                    raise ImportError("analysis_core not found in sys.modules")
                _ac = _sys.modules[_ac_key]
                spike_indices = dm.get_cluster_spike_indices(self.cluster_id)
                n = len(spike_indices)
                if n == 0:
                    return self.fallback_snippets

                # Random subsample — stratified over recording time for visual richness
                n_want = min(_N_CLOUD_SPIKES, n)
                if n > n_want:
                    # Uniform stride subsample (representative across session)
                    step = n // n_want
                    chosen = spike_indices[::step][:n_want]
                else:
                    chosen = spike_indices

                spike_times = dm.spike_times[chosen].astype(np.int64)
                n_time = len(self.t_ms)
                pre = _PRE_SAMPLES
                window = (-pre, n_time - pre)

                source = (
                    dm.raw_reader if dm.raw_reader is not None else dm.raw_data_memmap
                )
                raw = _ac.extract_snippets(
                    source,
                    spike_times,
                    window=window,
                    n_channels=dm.n_channels,
                )  # (n_ch, n_time, n_chosen)

                if raw.shape[2] == 0:
                    return self.fallback_snippets

                waves = raw[self.dom_chan, :, :].T  # (n_chosen, n_time)
                waves = waves.astype(np.float32) * dm.uV_per_bit

                # Baseline-correct each spike (subtract mean of pre-spike samples)
                baseline = waves[:, :pre].mean(axis=1, keepdims=True)
                waves -= baseline
                return waves

            except Exception:
                logger.debug(
                    "CloudWorker raw fetch failed, using fallback", exc_info=True
                )

        return self.fallback_snippets


def _build_cloud_payload(cluster_id, waveforms, median_trace, t_ms):
    """Pure numpy — safe to run off the GUI thread."""
    n_spikes, n_time = waveforms.shape
    MAX_SHADOWS = 200  # hard cap; renders ~5 ms

    # Compute per-spike amplitude (peak-to-trough) for colour bucketing
    spike_ptp = waveforms.max(axis=1) - waveforms.min(axis=1)
    ptp_min, ptp_max = spike_ptp.min(), spike_ptp.max()
    ptp_range = max(ptp_max - ptp_min, 1e-6)
    norm_ptp = (spike_ptp - ptp_min) / ptp_range  # 0..1

    # Stratified subsample: equal reps from each amplitude quintile
    n_buckets = 5
    bucket_ids = np.floor(norm_ptp * (n_buckets - 1e-9)).astype(int)
    per_bucket = max(1, MAX_SHADOWS // n_buckets)

    buckets_xy = []  # list of (x_flat, y_flat) per bucket
    for b in range(n_buckets):
        mask = bucket_ids == b
        indices = np.where(mask)[0]
        if len(indices) == 0:
            buckets_xy.append((np.array([]), np.array([])))
            continue
        chosen = (
            indices
            if len(indices) <= per_bucket
            else np.random.choice(indices, per_bucket, replace=False)
        )
        subset = waveforms[chosen]  # (k, n_time)
        k = len(chosen)
        nan_col = np.full((k, 1), np.nan, dtype=np.float32)
        x_con = np.tile(t_ms, (k, 1))  # (k, n_time)
        x_flat = np.column_stack([x_con, nan_col]).ravel()
        y_flat = np.column_stack([subset, nan_col]).ravel()
        buckets_xy.append((x_flat.astype(np.float32), y_flat.astype(np.float32)))

    # Percentile envelope
    if n_spikes >= 10:
        p10 = np.percentile(waveforms, 10, axis=0).astype(np.float32)
        p90 = np.percentile(waveforms, 90, axis=0).astype(np.float32)
    else:
        p10 = p90 = None

    return {
        "cluster_id": cluster_id,
        "buckets_xy": buckets_xy,
        "p10": p10,
        "p90": p90,
        "n_spikes": n_spikes,
    }


# ---------------------------------------------------------------------------
# Tier-3 worker: PCA + isolation metrics
# ---------------------------------------------------------------------------
class _PCAWorker(QRunnable):
    def __init__(
        self,
        cluster_id: int,
        unit_waves: np.ndarray,
        bg_waves: Optional[np.ndarray],
        unit_spike_indices: Optional[np.ndarray],
        signals: _WorkerSignals,
        bg_waves_by_cid: Optional[dict] = None,
        generation=None,
        unsorted_waves: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.setAutoDelete(True)
        self.unsorted_waves = unsorted_waves
        self.cluster_id = cluster_id
        self.generation = generation  # DataManager.generation of the request
        self.unit_waves = unit_waves  # (n_unit, n_time)
        self.bg_waves = bg_waves  # (n_bg, n_time) or None
        self.unit_spike_indices = unit_spike_indices  # global indices for future lasso
        self.signals = signals
        self.bg_waves_by_cid = bg_waves_by_cid or {}  # {cid: ndarray}

    @Slot()
    def run(self):
        try:
            payload = _build_pca_payload(
                self.cluster_id,
                self.unit_waves,
                self.bg_waves,
                self.unit_spike_indices,
                self.bg_waves_by_cid,
                unsorted_waves=self.unsorted_waves,
            )
            payload["_generation"] = self.generation
            self.signals.pca_ready.emit(self.cluster_id, payload)
        except Exception as exc:
            logger.exception("PCAWorker error cid=%d", self.cluster_id)
            self.signals.error.emit(self.cluster_id, str(exc))


def _build_pca_payload(
    cluster_id, unit_waves, bg_waves, unit_spike_indices, bg_waves_by_cid=None,
    unsorted_waves=None,
):
    """
    Pure numpy/sklearn — safe off GUI thread.

    bg_waves_by_cid : dict[int, ndarray] — per-cluster background waveforms.
        When present the payload includes bg_coords_by_cid so _draw_pca can
        colour individual comparison clusters differently.
    """
    from sklearn.decomposition import PCA

    MAX_UNIT = 1500
    MAX_BG = 1500

    # --- subsample unit -------------------------------------------------------
    n_unit = unit_waves.shape[0]
    if n_unit > MAX_UNIT:
        idx = np.random.choice(n_unit, MAX_UNIT, replace=False)
        unit_sub = unit_waves[idx].astype(np.float32)
        unit_sidx = unit_spike_indices[idx] if unit_spike_indices is not None else None
    else:
        unit_sub = unit_waves.astype(np.float32)
        unit_sidx = unit_spike_indices

    has_bg = bg_waves is not None and len(bg_waves) > 0
    if has_bg:
        n_bg = bg_waves.shape[0]
        if n_bg > MAX_BG:
            bg_sub = bg_waves[np.random.choice(n_bg, MAX_BG, replace=False)].astype(
                np.float32
            )
        else:
            bg_sub = bg_waves.astype(np.float32)

        # Templates and raw snippets can differ by a few samples — truncate both
        # to the shorter length so vstack never raises.
        n_time = min(unit_sub.shape[1], bg_sub.shape[1])
        unit_sub = unit_sub[:, :n_time]
        bg_sub = bg_sub[:, :n_time]

        combined = np.vstack([bg_sub, unit_sub])
        labels = np.concatenate([np.zeros(len(bg_sub)), np.ones(len(unit_sub))])
    else:
        combined = unit_sub
        labels = np.ones(len(unit_sub))
        bg_sub = None
        n_time = unit_sub.shape[1]

    # Threshold crossings no unit claims are part of the fit too: the axes
    # should describe everything on the channel (label 2).
    has_unsorted = unsorted_waves is not None and len(unsorted_waves) > 0
    if has_unsorted:
        uns = unsorted_waves.astype(np.float32)[:, :n_time]
        combined = np.vstack([combined, uns])
        labels = np.concatenate([labels, np.full(len(uns), 2.0)])

    if len(combined) < 4:
        return {"cluster_id": cluster_id, "error": "too_few_spikes"}

    # --- PCA on the waveforms in µV -----------------------------------------
    # Rows are real spikes on the dominant channel and its nearest channels,
    # baseline-subtracted (get_channel_all_snippets). Amplitude is kept: it
    # is most of what separates two cells on one electrode. (They were
    # z-scored only to mix in synthetic template waves, now removed.)
    # 3 components: the plot shows the first 2, the isolation score uses all 3.
    pca = PCA(n_components=min(3, combined.shape[1], len(combined)))
    coords3 = pca.fit_transform(combined)
    coords = coords3[:, :2]
    var = pca.explained_variance_ratio_

    bg_mask = labels == 0
    unit_mask = labels == 1
    bg_coords = coords[bg_mask] if has_bg else np.empty((0, 2))
    unit_coords = coords[unit_mask]
    unit3 = coords3[unit_mask]
    unsorted_coords = coords[labels == 2]

    # Unsorted crossings inside the cell's own 95 % region (Mahalanobis in
    # the 3 PCs, χ²₃ = 7.81): spikes it may be missing.
    n_unsorted_inside = 0
    if has_unsorted and len(unit3) >= 10 and coords3.shape[1] == 3:
        try:
            mu = unit3.mean(axis=0)
            icov = np.linalg.inv(np.cov(unit3.T) + np.eye(3) * 1e-6)
            dx = coords3[labels == 2] - mu
            n_unsorted_inside = int(np.sum(np.einsum("ij,jk,ik->i", dx, icov, dx) < 7.81))
        except np.linalg.LinAlgError:
            pass

    # --- Project each per-cluster bg set into the SAME PCA space -----------
    # This is the key step for the compare feature: we project bg_waves_by_cid
    # through the already-fitted PCA so all clouds share the same axes.
    bg_coords_by_cid: dict = {}
    bg3_by_cid: dict = {}
    if bg_waves_by_cid:
        for cid, cid_waves in bg_waves_by_cid.items():
            if cid_waves is None or len(cid_waves) == 0:
                continue
            try:
                cid_sub = cid_waves.astype(np.float32)
                # Subsample if large
                if len(cid_sub) > MAX_BG:
                    cid_sub = cid_sub[
                        np.random.choice(len(cid_sub), MAX_BG, replace=False)
                    ]
                # Align time dimension
                cid_sub = cid_sub[:, :n_time]
                c3 = pca.transform(cid_sub)
                bg3_by_cid[cid] = c3
                bg_coords_by_cid[cid] = c3[:, :2]
            except Exception:
                pass

    # --- Isolation: d′ to the closest neighbouring cell ---------------------
    # Each neighbour on its own, along the line joining the two means in the
    # first 3 PCs: d′ = |Δmean| / sqrt((var_cell + var_neighbour) / 2). Against
    # all neighbours pooled, as before, the score mixed several cells' spread
    # and moved from 0.64 to 0.32 between two samples of one cell (2026-09-25).
    isolation_label = "N/A (no other unit fires on this channel)"
    dprime = None
    closest = None
    dprime_by_cid = {}
    if len(unit3) >= 10:
        for cid, c3 in bg3_by_cid.items():
            if len(c3) < 10:
                continue
            d = _pair_dprime(unit3, c3)
            dprime_by_cid[cid] = d
            if dprime is None or d < dprime:
                dprime, closest = d, cid
    if dprime is not None:
        verdict = ("✓ well isolated" if dprime > 3.0 else
                   "~ marginal" if dprime > 1.5 else "✗ poor")
        isolation_label = f"closest neighbour: cell {closest}, d' = {dprime:.1f}  {verdict}"

    # --- Ellipse (unit, 2-sigma) --------------------------------------------
    ellipse = _compute_ellipse(unit_coords)

    return {
        "cluster_id": cluster_id,
        "bg_coords": bg_coords,  # flat, all bg — for gray cloud
        "bg_coords_by_cid": bg_coords_by_cid,  # per-cluster — for compare feature
        "unit_coords": unit_coords,
        "unit_spike_idx": unit_sidx,
        "var": var,
        "isolation_label": isolation_label,
        "dprime": dprime,
        "ellipse": ellipse,
        "has_bg": has_bg,
        "unsorted_coords": unsorted_coords,
        "n_unsorted_inside": n_unsorted_inside,
        "dprime_by_cid": dprime_by_cid,
    }


def _pair_dprime(a: np.ndarray, b: np.ndarray) -> float:
    """d′ between two point clouds along the line joining their means."""
    v = b.mean(axis=0) - a.mean(axis=0)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return 0.0
    v /= n
    pa, pb = a @ v, b @ v
    return float(abs(pb.mean() - pa.mean()) / np.sqrt((pa.var() + pb.var()) / 2.0 + 1e-12))


def _ellipse_item(e: dict, pen):
    """A 2-σ ellipse item, turned to its major axis (data coordinates, y up).

    The angle was computed but never applied, so every ellipse was drawn
    axis-aligned (2026-09-25).
    """
    w, h = e["width"], e["height"]
    ell = pg.QtWidgets.QGraphicsEllipseItem(-w / 2, -h / 2, w, h)
    ell.setPos(e["cx"], e["cy"])
    ell.setRotation(e["angle"])
    ell.setPen(pen)
    ell.setBrush(pg.mkBrush(0, 0, 0, 0))
    return ell


def _compute_ellipse(coords: np.ndarray) -> Optional[dict]:
    """Return 2-sigma ellipse params for a 2-D point cloud, or None."""
    if len(coords) < 4:
        return None
    try:
        cov = np.cov(coords.T)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        angle_deg = float(np.degrees(np.arctan2(*vecs[:, order[0]][::-1])))
        width = float(2 * 2 * np.sqrt(max(vals[0], 0)))
        height = float(2 * 2 * np.sqrt(max(vals[1], 0)))
        return {
            "cx": float(coords[:, 0].mean()),
            "cy": float(coords[:, 1].mean()),
            "width": width,
            "height": height,
            "angle": angle_deg,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Stage-A worker: fetch channel snippets, then fire PCA worker
# ---------------------------------------------------------------------------
class _ChannelSnippetsWorker(QRunnable):
    """
    Reads the cell's and its neighbours' real spikes around the dominant
    channel (dm.get_channel_all_snippets), then starts a _PCAWorker.

    Raw data only. The old fallback drew templates plus noise as if they
    were spikes and scored a d′ on them (2026-09-25). Stops early when the
    user has moved to another cell (``cancelled``).
    """

    def __init__(
        self,
        cluster_id: int,
        dom_chan: int,
        dm,
        signals: _WorkerSignals,
        cancelled=None,
    ):
        super().__init__()
        self.setAutoDelete(True)
        self.cluster_id = cluster_id
        self.dom_chan = dom_chan
        self.dm = dm
        self.signals = signals
        self.cancelled = cancelled or (lambda: False)

    @Slot()
    def run(self):
        try:
            r = self.dm.get_channel_all_snippets(
                self.dom_chan, self.cluster_id, cancelled=self.cancelled)
            if r["source"] == "cancelled":
                return
            if r["source"] != "raw" or len(r["unit_waves"]) == 0:
                self.signals.error.emit(self.cluster_id, "no raw spikes read for PCA")
                return
            pca_worker = _PCAWorker(
                self.cluster_id,
                r["unit_waves"],
                r["bg_waves"],
                r["unit_indices"],
                self.signals,
                bg_waves_by_cid=r.get("bg_waves_by_cid", {}),
                generation=getattr(self.dm, "generation", None),
                unsorted_waves=r.get("unsorted_waves"),
            )
            QThreadPool.globalInstance().start(pca_worker)
        except Exception as exc:
            logger.exception("ChannelSnippetsWorker failed cid=%d", self.cluster_id)
            self.signals.error.emit(self.cluster_id, str(exc))

# ---------------------------------------------------------------------------
# Stat badge helpers
# ---------------------------------------------------------------------------
def _badge_color(key: str, value) -> str:
    """Return a CSS colour string for a metric badge."""
    thresholds = {
        "isi_viol": [(0.5, _C["badge_good"]), (2.0, _C["badge_warn"])],
        "snr": [(10, _C["badge_good"]), (4.0, _C["badge_warn"])],
        "amp_uv": [(15, _C["badge_good"]), (5.0, _C["badge_warn"])],
        "amp_cv": [(0.25, _C["badge_good"]), (0.5, _C["badge_warn"])],
        "dprime": [(3.0, _C["badge_good"]), (1.5, _C["badge_warn"])],
    }
    if key not in thresholds or value is None:
        return _C["badge_neutral"]
    rules = thresholds[key]
    # For isi_viol and amp_cv: lower is better
    if key in ("isi_viol", "amp_cv"):
        good_thresh, warn_thresh = rules[0][0], rules[1][0]
        if value <= good_thresh:
            return rules[0][1]
        if value <= warn_thresh:
            return rules[1][1]
        return _C["badge_bad"]
    else:  # higher is better
        good_thresh, warn_thresh = rules[0][0], rules[1][0]
        if value >= good_thresh:
            return rules[0][1]
        if value >= warn_thresh:
            return rules[1][1]
        return _C["badge_bad"]


def _make_badge(color: str) -> QLabel:
    lbl = QLabel("●")
    lbl.setStyleSheet(f"color: {color}; font-size: 10px;")
    lbl.setFixedWidth(14)
    return lbl


def _plot_style(plot: pg.PlotWidget, title=""):
    """Apply consistent styling to a PlotWidget."""
    plot.setBackground(_C["bg_main"])
    if title:
        plot.setTitle(title, color=_C["text_secondary"], size="9pt")
    for axis in ("bottom", "left", "top", "right"):
        ax = plot.getAxis(axis)
        if axis in ("top", "right"):
            ax.hide()
        else:
            ax.setPen(pg.mkPen(_C["axis_pen"], width=1))
            ax.setTextPen(pg.mkPen(_C["axis_text"]))
            ax.setStyle(tickTextOffset=4, tickLength=-4)
    plot.showGrid(x=False, y=False)


# ---------------------------------------------------------------------------
# Main Panel
# ---------------------------------------------------------------------------
class WaveformPanel(QWidget):
    """
    Production-grade Waveform & Isolation panel.

    Public API (called by MainWindow / callbacks):
        update_all(cluster_id)   — trigger full refresh for new selection
        restyle_plots(colors)    — theme change hook (legacy compat)

    Internal rendering pipeline:
        _render_tier1()          — synchronous, median + stats
        _render_tier2()          — async shadow cloud
        _render_tier3()          — async PCA + isolation
    """

    # -- Future hook: emitted when user lasso-selects spikes in PCA view
    # spike_selection_changed = Signal(int, np.ndarray)  # cluster_id, spike_indices

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        # Build in the current theme's colours (a test double may not have them).
        try:
            _C.update(palette_from_theme(main_window.get_current_colors()))
        except (AttributeError, KeyError, TypeError, ValueError):
            logger.debug("theme colours unavailable; using the built-in palette")

        # Threading
        self._pool = QThreadPool.globalInstance()
        self._signals = _WorkerSignals()
        self._signals.cloud_ready.connect(self._on_cloud_ready)
        self._signals.pca_ready.connect(self._on_pca_ready)
        self._signals.error.connect(self._on_worker_error)

        # State
        self._current_cluster_id: Optional[int] = None
        self._current_dom_chan: Optional[int] = None
        self._current_t_ms: Optional[np.ndarray] = None
        self._compare_cluster_id: Optional[int] = None  # compare feature
        self._last_pca_payload: Optional[dict] = None  # for re-draw on compare change

        self._setup_ui()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------
    def _setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        self._splitter = splitter
        splitter.setHandleWidth(3)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{ background: {_C['border_subtle']}; }}
            QSplitter::handle:hover {{ background: {_C['border_default']}; }}
        """)
        root.addWidget(splitter)

        # ── LEFT: waveform ─────────────────────────────────────────────────
        left = QWidget()
        self._left_pane = left
        left.setStyleSheet(f"background: {_C['bg_main']};")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(8, 8, 4, 8)
        lv.setSpacing(4)

        self._wave_plot = self._make_wave_plot()
        lv.addWidget(self._wave_plot)

        # Neighbour sparklines (Phase 2 — placeholder row)
        self._sparkline_row = QHBoxLayout()
        self._sparkline_row.setSpacing(4)
        self._sparkline_labels: list[pg.PlotWidget] = []
        lv.addLayout(self._sparkline_row)

        splitter.addWidget(left)

        # ── RIGHT: PCA + stats ─────────────────────────────────────────────
        right = QFrame()
        self._right_pane = right
        right.setStyleSheet(f"""
            QFrame {{
                background: {_C['bg_right']};
                border-left: 1px solid {_C['border_subtle']};
            }}
        """)
        right.setFixedWidth(420)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(12, 12, 12, 12)
        rv.setSpacing(10)

        # Header label for cluster identity
        self._cluster_header = QLabel("—")
        self._cluster_header.setStyleSheet(
            f"color: {_C['text_primary']}; font-size: 13px; font-weight: 600;"
        )
        rv.addWidget(self._cluster_header)

        # PCA plot: every event on the channel, coloured by unit (QA view)
        self._pca_plot = self._make_pca_plot()
        rv.addWidget(self._pca_plot, stretch=3)

        # Isolation metric label (wraps: it names a neighbour and a count)
        self._isolation_label = QLabel("Isolation: —")
        self._isolation_label.setStyleSheet(
            f"color: {_C['text_secondary']}; font-size: 10px;"
        )
        self._isolation_label.setWordWrap(True)
        self._isolation_label.setMinimumHeight(30)          # two lines: neighbour, then misses
        self._isolation_label.setAlignment(Qt.AlignCenter)
        rv.addWidget(self._isolation_label)

        # Legend: one chip per unit on this channel, in its colour; click to
        # highlight it. A grid of two columns (a single row overflowed the
        # narrow column with 10–16 units and hid every chip).
        self._compare_strip_container = QWidget()
        self._compare_strip_container.setStyleSheet("background: transparent;")
        self._compare_strip_layout = QGridLayout(self._compare_strip_container)
        self._compare_strip_layout.setContentsMargins(0, 0, 0, 0)
        self._compare_strip_layout.setHorizontalSpacing(4)
        self._compare_strip_layout.setVerticalSpacing(2)
        self._cmp_label = QLabel("")
        self._compare_buttons: dict = {}
        rv.addWidget(self._compare_strip_container)

        # Divider
        div = QFrame()
        self._divider = div
        div.setFrameShape(QFrame.HLine)
        div.setStyleSheet(f"border: none; border-top: 1px solid {_C['border_subtle']};")
        div.setFixedHeight(1)
        rv.addWidget(div)

        # Stats grid
        rv.addWidget(self._make_stats_panel())

        splitter.addWidget(right)
        splitter.setSizes([1000, 420])

    def _make_wave_plot(self) -> pg.PlotWidget:
        pw = pg.PlotWidget()
        _plot_style(pw)
        pw.setLabel("bottom", "Time (ms)", **{"color": _C["axis_text"], "font-size": "9pt"})
        pw.setLabel("left", "Amplitude (µV)", **{"color": _C["axis_text"], "font-size": "9pt"})

        # Persistent zero-line
        self._zero_line = pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen(_C["zero_line"], width=1, style=Qt.DashLine)
        )
        pw.addItem(self._zero_line)

        # Pre-allocated plot items so we never recreate them per frame
        # Envelope (fill between)
        self._env_curve_lo = pg.PlotCurveItem(pen=None)
        self._env_curve_hi = pg.PlotCurveItem(pen=None)
        self._env_fill = pg.FillBetweenItem(
            self._env_curve_lo,
            self._env_curve_hi,
            brush=pg.mkBrush(*_C["envelope"]),
        )
        pw.addItem(self._env_curve_lo)
        pw.addItem(self._env_curve_hi)
        pw.addItem(self._env_fill)

        # 5 shadow-cloud CurveItems (one per amplitude bucket)
        cloud_colors = [_C[f"cloud_{i}"] for i in range(5)]
        self._cloud_items: list[pg.PlotCurveItem] = []
        for rgba in cloud_colors:
            item = pg.PlotCurveItem(
                pen=pg.mkPen(color=rgba, width=1), connect="finite", antialias=False
            )  # antialias=False is critical for speed
            pw.addItem(item)
            self._cloud_items.append(item)

        # Median trace (on top of everything)
        self._median_item = pg.PlotCurveItem(
            pen=pg.mkPen(_C["median"], width=2.5), antialias=True
        )
        pw.addItem(self._median_item)

        # Title text item
        self._wave_title = pg.TextItem("", anchor=(0, 0), color=_C["text_secondary"])
        self._wave_title.setFont(QFont("", 9))
        pw.addItem(self._wave_title)

        return pw

    def _make_pca_plot(self) -> pg.PlotWidget:
        pw = pg.PlotWidget()
        pw.setBackground(_C["bg_main"])
        pw.setAspectLocked(False)
        for axis in ("top", "right"):
            pw.getAxis(axis).hide()
        for axis in ("bottom", "left"):
            ax = pw.getAxis(axis)
            ax.setPen(pg.mkPen(_C["axis_pen"], width=1))
            ax.setTextPen(pg.mkPen(_C["axis_text"]))
            ax.setStyle(tickLength=-4)
        pw.setLabel("bottom", "PC1", **{"color": _C["axis_text"], "font-size": "9pt"})
        pw.setLabel("left", "PC2", **{"color": _C["axis_text"], "font-size": "9pt"})

        # Pre-allocated scatter items
        self._pca_bg_scatter = pg.ScatterPlotItem(
            size=4, pen=None, brush=pg.mkBrush(*_C["pca_bg"]), hoverable=False
        )
        self._pca_unit_scatter = pg.ScatterPlotItem(
            size=5,
            pen=None,
            brush=pg.mkBrush(_C["pca_unit"]),
            hoverable=True,
            tip=lambda x, y, data: f"spike idx: {data}" if data is not None else "",
        )

        # Compare cluster scatter (coral, shown only when compare active)
        self._pca_cmp_scatter = pg.ScatterPlotItem(
            size=5, pen=None, brush=pg.mkBrush(_C["pca_compare"]), hoverable=False
        )

        pw.addItem(self._pca_bg_scatter)
        pw.addItem(self._pca_cmp_scatter)  # below unit so unit stays on top
        pw.addItem(self._pca_unit_scatter)

        # Ellipse placeholders (added/removed dynamically)
        self._pca_ellipse_item: Optional[pg.QtWidgets.QGraphicsEllipseItem] = None
        self._pca_cmp_ellipse_item: Optional[pg.QtWidgets.QGraphicsEllipseItem] = None

        return pw

    def _make_stats_panel(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 4, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        # --- metric definitions -----------------------------------------
        # (key, display_label, badge_key, format_fn)
        self._stat_defs = [
            ("n_spikes", "Spikes", None, lambda v: f"{v:,}"),
            ("mean_fr", "Mean FR", None, lambda v: f"{v:.1f} Hz"),
            ("isi_viol", "ISI violations", "isi_viol", lambda v: f"{v:.2f} %"),
            ("amp_uv", "Amplitude", "amp_uv", lambda v: f"{v:.1f} µV"),
            ("snr", "SNR", "snr", lambda v: f"{v:.1f}"),
            ("width_ms", "Spike width", None, lambda v: f"{v:.3f} ms"),
            ("peak_lat", "Peak latency", None, lambda v: f"{v:.3f} ms"),
            ("amp_cv", "Amp stability", "amp_cv", lambda v: f"CV {v:.3f}"),
        ]

        self._stat_badges: dict[str, QLabel] = {}
        self._stat_values: dict[str, QLabel] = {}
        self._stat_names: list[QLabel] = []

        for i, (key, label, badge_key, _) in enumerate(self._stat_defs):
            row, col_offset = divmod(i, 2)
            col = col_offset * 3  # badge | label | value

            badge = _make_badge(_C["badge_neutral"])
            self._stat_badges[key] = badge

            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {_C['text_secondary']}; font-size: 10px;")
            self._stat_names.append(lbl)

            val = QLabel("—")
            val.setStyleSheet(
                f"color: {_C['text_primary']}; font-size: 12px; font-weight: 600;"
            )
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._stat_values[key] = val

            grid.addWidget(badge, row, col)
            grid.addWidget(lbl, row, col + 1)
            grid.addWidget(val, row, col + 2)

        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(4, 1)
        return container

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def update_all(self, cluster_id: int):
        """Entry point called by MainWindow on cluster selection change."""
        if not self.isVisible():
            return
        self._current_cluster_id = cluster_id
        self._compare_cluster_id = None  # reset on new selection
        self._last_pca_payload = None
        self._clear_compare_buttons()  # remove stale buttons immediately
        self._render_tier1(cluster_id)

    def show_reading(self, cluster_id: int):
        """The raw read for a new cell has started: clear the last cell's plots.

        The tab used to keep the previous cell's waveforms and title until the
        read finished (5–16 s on the share), so it looked stuck.
        """
        if not self.isVisible():
            return
        self._current_cluster_id = cluster_id        # also stops the last cell's PCA reads
        self._compare_cluster_id = None
        self._last_pca_payload = None
        self._clear_compare_buttons()
        self._clear_plots()
        self._cluster_header.setText(f"Cluster {cluster_id} — reading its spikes from the raw file…")
        self._wave_title.setText("")
        self._isolation_label.setText("PCA: waiting for the spikes…")

    def restyle_plots(self, colors: dict):
        """Re-colour the panel for a theme switch (PLAN.md Q16)."""
        _C.update(palette_from_theme(colors))
        self._splitter.setStyleSheet(f"""
            QSplitter::handle {{ background: {_C['border_subtle']}; }}
            QSplitter::handle:hover {{ background: {_C['border_default']}; }}
        """)
        self._left_pane.setStyleSheet(f"background: {_C['bg_main']};")
        self._right_pane.setStyleSheet(f"""
            QFrame {{
                background: {_C['bg_right']};
                border-left: 1px solid {_C['border_subtle']};
            }}
        """)
        self._cluster_header.setStyleSheet(
            f"color: {_C['text_primary']}; font-size: 13px; font-weight: 600;")
        self._cmp_label.setStyleSheet(
            f"color: {_C['text_secondary']}; font-size: 9px; font-weight: 600;")
        self._isolation_label.setStyleSheet(
            f"color: {_C['text_secondary']}; font-size: 10px;")
        self._divider.setStyleSheet(
            f"border: none; border-top: 1px solid {_C['border_subtle']};")
        for lbl in self._stat_names:
            lbl.setStyleSheet(f"color: {_C['text_secondary']}; font-size: 10px;")
        for val in self._stat_values.values():
            val.setStyleSheet(
                f"color: {_C['text_primary']}; font-size: 12px; font-weight: 600;")
        for badge in self._stat_badges.values():
            badge.setStyleSheet(f"color: {_C['badge_neutral']}; font-size: 10px;")

        _plot_style(self._wave_plot)
        self._pca_plot.setBackground(_C["bg_main"])
        for plot, (bottom, left) in ((self._wave_plot, ("Time", "Amplitude")),
                                     (self._pca_plot, ("PC1", "PC2"))):
            for axis in ("bottom", "left"):
                ax = plot.getAxis(axis)
                ax.setPen(pg.mkPen(_C["axis_pen"], width=1))
                ax.setTextPen(pg.mkPen(_C["axis_text"]))
        style = {"color": _C["axis_text"], "font-size": "9pt"}
        # Units in the text: with units= pyqtgraph added its own prefix ("mms", "mµV").
        self._wave_plot.setLabel("bottom", "Time (ms)", **style)
        self._wave_plot.setLabel("left", "Amplitude (µV)", **style)
        self._pca_plot.setLabel("bottom", "PC1", **style)
        self._pca_plot.setLabel("left", "PC2", **style)
        self._zero_line.setPen(pg.mkPen(_C["zero_line"], width=1, style=Qt.DashLine))
        self._env_fill.setBrush(pg.mkBrush(*_C["envelope"]))
        for i, item in enumerate(self._cloud_items):
            item.setPen(pg.mkPen(color=_C[f"cloud_{i}"], width=1))
        self._median_item.setPen(pg.mkPen(_C["median"], width=2.5))
        self._wave_title.setColor(_C["text_secondary"])
        self._pca_bg_scatter.setBrush(pg.mkBrush(*_C["pca_bg"]))
        self._pca_unit_scatter.setBrush(pg.mkBrush(_C["pca_unit"]))
        self._pca_cmp_scatter.setBrush(pg.mkBrush(_C["pca_compare"]))

    # -----------------------------------------------------------------------
    # Tier 1 — synchronous, median + basic stats (< 5 ms)
    # -----------------------------------------------------------------------
    def _render_tier1(self, cluster_id: int):
        self._clear_plots()

        dm: DataManager = self.main_window.data_manager
        features = dm.get_lightweight_features(cluster_id)

        if not features or "median_ei" not in features:
            self._wave_title.setText("")
            if getattr(dm, "dat_path", None) is None:
                # This used to say "waiting for cache…" forever.
                self._cluster_header.setText(
                    f"Cluster {cluster_id} — no raw recording is loaded. "
                    "File ▸ Load Raw Data File shows its spikes and their PCA.")
                self._isolation_label.setText("PCA: needs the raw recording")
            else:
                self._cluster_header.setText(
                    f"Cluster {cluster_id} — reading its spikes from the raw file…")
            return

        median_ei = features["median_ei"]  # (n_ch, n_time)
        raw_snippets = features.get("raw_snippets")  # (n_ch, n_time, n_spikes) or None

        # --- Dominant channel --------------------------------------------
        ptp = median_ei.max(axis=1) - median_ei.min(axis=1)
        dom_chan = int(np.argmax(ptp))
        self._current_dom_chan = dom_chan

        median_trace = median_ei[dom_chan].astype(np.float32)  # (n_time,)
        n_time = median_trace.shape[0]

        sr = float(getattr(dm, "sampling_rate", 30_000))
        t_ms = ((np.arange(n_time) - _PRE_SAMPLES) / sr * 1_000).astype(np.float32)
        self._current_t_ms = t_ms

        # --- Draw median immediately (Tier 1) ----------------------------
        self._median_item.setData(t_ms, median_trace)

        # Title. The cell's own spike count, not the number of cached
        # snippets: that was 30 for every cell, and the mean rate came out
        # as 30 spikes over the whole recording, "0.0 Hz" (2026-09-25).
        row = dm.cluster_df.loc[dm.cluster_df.cluster_id == cluster_id, "n_spikes"]
        n_spikes = int(row.values[0]) if len(row) else 0
        n_drawn = int(raw_snippets.shape[2]) if raw_snippets is not None else 0
        self._wave_title.setText(
            f"Cluster {cluster_id}  ·  Ch {dom_chan}  ·  {n_spikes:,} spikes"
            + (f" ({n_drawn} drawn)" if 0 < n_drawn < n_spikes else "")
        )
        self._wave_title.setPos(t_ms[0], float(median_trace.max()))

        self._cluster_header.setText(
            f"Cluster {cluster_id}   ·   Ch {dom_chan}   ·   {n_spikes:,} spikes"
        )

        # --- Tier-1 stats (no raw data needed) ---------------------------
        self._populate_stats_tier1(cluster_id, median_trace, t_ms, n_spikes, dm)

        # --- Launch Tier 2 & 3 ─────────────────────────────────────────
        # For the cloud: always attempt to fetch real raw spikes.
        # Pass cached snippets only as a fallback if raw data isn't loaded.
        fallback = raw_snippets[dom_chan, :, :].T if raw_snippets is not None else None
        self._launch_cloud_worker(cluster_id, fallback, median_trace, t_ms, sr, dm)

        # For PCA: pass real unit waveforms from cache if available.
        # _ChannelSnippetsWorker will also fetch real raw snippets for bg.
        self._launch_pca_worker(cluster_id, fallback, dom_chan, dm)

    # -----------------------------------------------------------------------
    # Tier 1 stats
    # -----------------------------------------------------------------------
    def _populate_stats_tier1(
        self,
        cluster_id: int,
        median_trace: np.ndarray,
        t_ms: np.ndarray,
        n_spikes: int,
        dm,
    ):
        sr = float(getattr(dm, "sampling_rate", 30_000))
        float(t_ms[1] - t_ms[0]) if len(t_ms) > 1 else 1.0 / sr * 1000

        # Spike count
        self._set_stat("n_spikes", n_spikes, None)

        # Mean firing rate — best estimate of recording duration:
        #   1. n_samples / sr  (most accurate, requires raw data loaded)
        #   2. spike_times[-1] / sr  (always available after KS load)
        # We deliberately do NOT use first-to-last spike of THIS cluster
        # because a small cluster spanning a short window would show an
        # inflated FR.
        try:
            n_total = int(getattr(dm, "n_samples", 0))
            recording_s = n_total / sr if n_total > 0 else 0.0
            if recording_s <= 0:
                # Use the last spike time across ALL clusters as proxy
                all_spikes = getattr(dm, "spike_times", None)
                if all_spikes is not None and len(all_spikes) > 0:
                    recording_s = float(all_spikes[-1]) / sr
            mean_fr = float(n_spikes / recording_s) if recording_s > 0 else 0.0
        except Exception:
            mean_fr = 0.0
        self._set_stat("mean_fr", mean_fr, None)

        # ISI violations from cluster_df
        row = dm.cluster_df[dm.cluster_df.cluster_id == cluster_id]
        isi_viol = (
            float(row["isi_violations_pct"].values[0])
            if not row.empty and "isi_violations_pct" in row.columns
            else None
        )
        self._set_stat("isi_viol", isi_viol, "isi_viol")

        # Amplitude p2p from median trace
        amp_uv = float(np.ptp(median_trace))
        self._set_stat("amp_uv", amp_uv, "amp_uv")

        # SNR: median peak-to-trough amplitude / MAD of per-spike amplitudes.
        # This is the most meaningful single-number quality metric for a
        # sorted unit: how many 'noise widths' tall is the average spike?
        # Uses Kilosort spike_amplitudes (already stored per-spike).
        # Falls back to median / std if amplitudes unavailable.
        try:
            amps = dm.get_cluster_spike_amplitudes(cluster_id)
            if amps is not None and len(amps) > 4:
                amps_f = np.abs(amps.astype(np.float32))
                med_amp = float(np.median(amps_f))
                mad_amp = float(np.median(np.abs(amps_f - med_amp))) * 1.4826
                snr = med_amp / mad_amp if mad_amp > 1e-9 else 0.0
            else:
                # Fallback: P2P of median / std of pre-spike region
                amp_uv_val = float(np.ptp(median_trace))
                noise_std = float(np.std(median_trace[:_PRE_SAMPLES]))
                snr = amp_uv_val / noise_std if noise_std > 1e-9 else 0.0
        except Exception:
            snr = 0.0
        self._set_stat("snr", snr, "snr")

        # Peak latency
        trough_idx = int(np.argmin(median_trace))
        peak_lat_ms = float(t_ms[trough_idx])
        self._set_stat("peak_lat", peak_lat_ms, None)

        # Waveform width at half-minimum (ms)
        try:
            trough_val = median_trace[trough_idx]
            half_val = trough_val / 2.0
            # Find crossings left and right of trough
            left_cross = np.where(median_trace[:trough_idx] >= half_val)[0]
            right_cross = np.where(median_trace[trough_idx:] >= half_val)[0]
            if len(left_cross) > 0 and len(right_cross) > 0:
                left_idx = left_cross[-1]
                right_idx = trough_idx + right_cross[0]
                width_ms = float(t_ms[right_idx] - t_ms[left_idx])
            else:
                width_ms = float("nan")
        except Exception:
            width_ms = float("nan")
        self._set_stat("width_ms", width_ms if np.isfinite(width_ms) else None, None)

        # Amplitude stability (CV) — requires spike amplitudes
        amps = dm.get_cluster_spike_amplitudes(cluster_id)
        if amps is not None and len(amps) > 1:
            mean_amp = float(np.mean(amps))
            amp_cv = float(np.std(amps) / mean_amp) if mean_amp > 0 else 0.0
        else:
            amp_cv = None
        self._set_stat("amp_cv", amp_cv, "amp_cv")

    def _set_stat(self, key: str, value, badge_key: Optional[str]):
        fmt_fn = {d[0]: d[3] for d in self._stat_defs}.get(key, str)
        val_lbl = self._stat_values.get(key)
        badge_lbl = self._stat_badges.get(key)
        if val_lbl:
            val_lbl.setText(fmt_fn(value) if value is not None else "—")
        if badge_lbl:
            color = _badge_color(badge_key, value) if badge_key else _C["badge_neutral"]
            badge_lbl.setStyleSheet(f"color: {color}; font-size: 10px;")

    # -----------------------------------------------------------------------
    # Tier 2 — shadow cloud
    # -----------------------------------------------------------------------
    def _launch_cloud_worker(
        self,
        cluster_id: int,
        fallback_snippets: Optional[np.ndarray],
        median_trace: np.ndarray,
        t_ms: np.ndarray,
        sr: float,
        dm,
    ):
        worker = _CloudWorker(
            cluster_id,
            dm,
            median_trace,
            t_ms,
            self._current_dom_chan,
            sr,
            fallback_snippets,
            self._signals,
        )
        self._pool.start(worker)

    @Slot(int, object)
    def _generation(self):
        return getattr(getattr(self.main_window, "data_manager", None), "generation", None)

    def _is_stale(self, cluster_id, payload):
        """Another cell, or the same cell ID in the previous run (PLAN.md Q13)."""
        if cluster_id != self._current_cluster_id:
            return True
        return isinstance(payload, dict) and payload.get("_generation", self._generation()) \
            != self._generation()

    def _on_cloud_ready(self, cluster_id: int, payload: dict):
        if self._is_stale(cluster_id, payload):
            return  # stale — user moved on, or the run changed
        self._draw_cloud(payload)

    def _draw_cloud(self, payload: dict):
        if payload is None:
            return
        t_ms = self._current_t_ms
        if t_ms is None:
            return

        buckets_xy = payload["buckets_xy"]
        p10 = payload["p10"]
        p90 = payload["p90"]

        # Shadow traces — 5 CurveItems, 5 draw calls
        for i, (xf, yf) in enumerate(buckets_xy):
            if len(xf) > 0:
                self._cloud_items[i].setData(xf, yf)

        # Envelope fill
        if p10 is not None and p90 is not None:
            self._env_curve_lo.setData(t_ms, p10)
            self._env_curve_hi.setData(t_ms, p90)

    # -----------------------------------------------------------------------
    # Tier 3 — PCA
    # -----------------------------------------------------------------------
    def _launch_pca_worker(
        self,
        cluster_id: int,
        unit_waves_from_cache: Optional[np.ndarray],
        dom_chan: int,
        dm,
    ):
        """
        Two-stage PCA pipeline:
          Stage A: _ChannelSnippetsWorker  → fetches unit + bg waveforms
                   (uses dm.get_channel_all_snippets if available, else
                    falls back to an inline template synthesiser)
          Stage B: _PCAWorker              → runs PCA + computes isolation

        LRU cache keyed by (DataManager.generation, cluster_id, dom_chan)
        to avoid recomputing when the user re-visits a cluster. The generation
        keeps another run's cluster 5 from answering for this run's.
        """
        cache_key = (getattr(dm, "generation", None), cluster_id, dom_chan)
        if cache_key in _PCA_CACHE:
            _PCA_CACHE.move_to_end(cache_key)
            self._on_pca_ready(cluster_id, _PCA_CACHE[cache_key])
            return

        # Fire a snippets-fetch worker; on completion it will fire the PCA worker.
        # It stops reading once the user has moved on (it used to compete
        # with the next cell's reads).
        worker = _ChannelSnippetsWorker(
            cluster_id=cluster_id,
            dom_chan=dom_chan,
            dm=dm,
            signals=self._signals,
            cancelled=lambda cid=cluster_id: self._current_cluster_id != cid,
        )
        self._isolation_label.setText("PCA: reading spikes from the raw file…")
        self._pool.start(worker)

    @Slot(int, object)
    def _on_pca_ready(self, cluster_id: int, payload: dict):
        if self._is_stale(cluster_id, payload):
            return
        if payload is None or payload.get("error"):
            self._isolation_label.setText("PCA: insufficient data")
            return

        # Update LRU cache
        cache_key = (payload.get("_generation"), cluster_id, self._current_dom_chan)
        _PCA_CACHE[cache_key] = payload
        _PCA_CACHE.move_to_end(cache_key)
        while len(_PCA_CACHE) > _PCA_CACHE_MAX:
            _PCA_CACHE.popitem(last=False)

        self._last_pca_payload = payload
        self._rebuild_compare_strip(payload)
        self._draw_pca(payload)

    def _rebuild_compare_strip(self, payload: dict):
        """Legend chips: every unit with events on this channel, then the unsorted count.

        Each chip is in its unit's colour and shows its events here and its d′
        to this cell; clicking one highlights that unit in the PCA.
        """
        self._clear_compare_buttons()
        by_cid = payload.get("bg_coords_by_cid", {}) or {}
        colours = self._unit_colours(by_cid)
        dmap = payload.get("dprime_by_cid", {}) or {}
        chips = [(None, f"this cell · {len(payload['unit_coords'])}", _C["pca_unit"])]
        for cid in sorted(by_cid, key=lambda c: dmap.get(c, 99.0)):   # closest first
            d = f" · d′ {dmap[cid]:.1f}" if cid in dmap else ""
            chips.append((cid, f"cell {cid} · {len(by_cid[cid])}{d}", colours[cid]))
        uns = payload.get("unsorted_coords")
        if uns is not None and len(uns):
            chips.append(("unsorted", f"unsorted · {len(uns)}", _C["text_secondary"]))
        for key, label, colour in chips:
            btn = QPushButton(f"● {label}")
            btn.setCheckable(key not in (None, "unsorted"))
            btn.setChecked(self._compare_cluster_id == key)
            btn.setFixedHeight(20)
            btn.setToolTip("Highlight this unit in the PCA" if btn.isCheckable() else "")
            btn.setStyleSheet(
                f"QPushButton {{ background: {_C['bg_card']}; color: {colour};"
                f" border: 1px solid {_C['border_default']}; border-radius: 3px;"
                f" font-size: 9px; padding: 0 5px; }}"
                f"QPushButton:checked {{ border-color: {colour}; font-weight: 600; }}")
            if btn.isCheckable():
                btn.clicked.connect(lambda checked, c=key: self._on_compare_selected(c))
            n = len(self._compare_buttons)
            self._compare_strip_layout.addWidget(btn, n // 2, n % 2)
            btn.show()
            self._compare_buttons[key] = btn

    def _on_compare_selected(self, cid: int):
        """Toggle compare cluster. Clicking the active one deselects it."""
        if self._compare_cluster_id == cid:
            self._compare_cluster_id = None
        else:
            self._compare_cluster_id = cid

        # Update button checked states
        for c, btn in self._compare_buttons.items():
            btn.setChecked(c == self._compare_cluster_id)

        # Redraw PCA with current payload — no new worker needed
        if self._last_pca_payload is not None:
            self._draw_pca(self._last_pca_payload)

    def _unit_colours(self, cids) -> dict:
        """One theme colour per other unit, stable by ID order; red stays this cell's."""
        from ..theme import categorical
        colors = self.main_window.get_current_colors()
        out, k = {}, 0
        for cid in sorted(cids):
            if k % 12 == 2:                       # the categorical red: kept for this cell
                k += 1
            out[cid] = categorical(k, colors)
            k += 1
        return out

    def _draw_pca(self, payload: dict):
        """Every event on the channel, coloured by who owns it (QA view).

        This cell red, each other unit its own colour, threshold crossings no
        unit claims small and grey. A chip in the strip above highlights one
        unit (the rest fade) and gives its d′ to this cell.
        """
        self._pca_bg_scatter.clear()
        self._pca_cmp_scatter.clear()
        self._pca_unit_scatter.clear()
        for attr in ("_pca_ellipse_item", "_pca_cmp_ellipse_item"):
            item = getattr(self, attr, None)
            if item is not None:
                try:
                    self._pca_plot.removeItem(item)
                except Exception:
                    pass
                setattr(self, attr, None)

        by_cid = payload.get("bg_coords_by_cid", {}) or {}
        unit_coords = payload["unit_coords"]
        unsorted = payload.get("unsorted_coords")
        var = payload["var"]
        unit_sidx = payload.get("unit_spike_idx")
        focus = self._compare_cluster_id
        colours = self._unit_colours(by_cid)

        # Unsorted crossings: small and grey, underneath everything.
        if unsorted is not None and len(unsorted):
            self._pca_bg_scatter.addPoints(x=unsorted[:, 0].tolist(), y=unsorted[:, 1].tolist())

        # Other units, one colour each; faded when another unit is in focus.
        spots = []
        for cid, c in by_cid.items():
            if not len(c):
                continue
            col = pg.mkColor(colours[cid])
            if focus is not None and focus != cid:
                col.setAlpha(60)
            brush = pg.mkBrush(col)
            size = 7 if focus == cid else 5
            spots += [{"pos": (float(x), float(y)), "brush": brush, "size": size, "data": int(cid)}
                      for x, y in c[:, :2]]
        if spots:
            self._pca_cmp_scatter.addPoints(spots)
        if focus is not None and focus in by_cid and len(by_cid[focus]) >= 4:
            e = _compute_ellipse(by_cid[focus][:, :2])
            if e is not None:
                ell = _ellipse_item(e, pg.mkPen(colours[focus], width=1.5, style=Qt.DashLine))
                self._pca_plot.addItem(ell)
                self._pca_cmp_ellipse_item = ell

        # This cell, on top.
        if len(unit_coords) > 0:
            if unit_sidx is not None and len(unit_sidx) == len(unit_coords):
                self._pca_unit_scatter.addPoints([
                    {"pos": (float(unit_coords[i, 0]), float(unit_coords[i, 1])),
                     "data": int(unit_sidx[i])} for i in range(len(unit_coords))])
            else:
                self._pca_unit_scatter.addPoints(
                    x=unit_coords[:, 0].tolist(), y=unit_coords[:, 1].tolist())
        unit_ellipse = payload.get("ellipse")
        if unit_ellipse is not None:
            ell = _ellipse_item(unit_ellipse, pg.mkPen(_C["pca_unit"], width=1.5, style=Qt.DashLine))
            self._pca_plot.addItem(ell)
            self._pca_ellipse_item = ell

        if len(var) >= 2:
            style = {"color": _C["axis_text"], "font-size": "9pt"}
            self._pca_plot.setLabel("bottom", f"PC1  ({var[0]:.1%})", **style)
            self._pca_plot.setLabel("left", f"PC2  ({var[1]:.1%})", **style)

        # Isolation line: the focused unit, else the closest one; plus misses.
        dmap = payload.get("dprime_by_cid", {}) or {}
        if focus is not None and focus in dmap:
            d = dmap[focus]
            verdict = "✓ well isolated" if d > 3 else "~ marginal" if d > 1.5 else "✗ poor"
            text = f"this cell vs cell {focus}: d' = {d:.1f}  {verdict}"
        else:
            text = payload.get("isolation_label", "")
        n_in = int(payload.get("n_unsorted_inside", 0) or 0)
        n_uns = 0 if unsorted is None else len(unsorted)
        if n_uns:
            text += (f"\n{n_in} of {n_uns} unsorted crossings fall inside this cell's "
                     f"cluster (95 %)" if n_in else
                     f"\n{n_uns} unsorted crossings, none inside this cell's cluster")
        self._isolation_label.setText(text)

    def _on_worker_error(self, cluster_id: int, msg: str):
        if cluster_id != self._current_cluster_id:
            return
        logger.info("Waveform worker cid=%d: %s", cluster_id, msg)
        if msg == "no raw spikes read for PCA":
            self._isolation_label.setText(
                "PCA: no spikes of this cell could be read from the raw file")
        else:
            self._isolation_label.setText(f"PCA failed: {msg}")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def _clear_compare_buttons(self):
        """Remove all compare toggle buttons without touching PCA plots."""
        for btn in list(self._compare_buttons.values()):
            self._compare_strip_layout.removeWidget(btn)
            btn.deleteLater()
        self._compare_buttons.clear()

    def _clear_plots(self):
        """Reset all pre-allocated plot items to empty data."""
        empty = np.array([], dtype=np.float32)
        self._median_item.setData(empty, empty)
        self._env_curve_lo.setData(empty, empty)
        self._env_curve_hi.setData(empty, empty)
        for item in self._cloud_items:
            item.setData(empty, empty)
        self._pca_bg_scatter.clear()
        self._pca_cmp_scatter.clear()
        self._pca_unit_scatter.clear()
        for attr in ("_pca_ellipse_item", "_pca_cmp_ellipse_item"):
            item = getattr(self, attr, None)
            if item is not None:
                try:
                    self._pca_plot.removeItem(item)
                except Exception:
                    pass
                setattr(self, attr, None)
        self._isolation_label.setText("Isolation: —")
        for key in self._stat_values:
            self._stat_values[key].setText("—")
        for key in self._stat_badges:
            self._stat_badges[key].setStyleSheet(
                f"color: {_C['badge_neutral']}; font-size: 10px;"
            )

    # -----------------------------------------------------------------------
    # Future hooks (stubs — fill in when DataManager.split_cluster is ready)
    # -----------------------------------------------------------------------
    def _on_spike_clicked(self, _scatter_item, points):
        """
        HOOK: Called when user clicks a spike in the PCA scatter.
        Each point carries its global spike index in .data().
        Wire this to a lasso-selection mode in Phase 2.
        """
        for pt in points:
            spike_idx = pt.data()
            logger.debug("Spike clicked: global index %s", spike_idx)
        # TODO: accumulate selected indices → emit spike_selection_changed

    def _on_lasso_selected(self, spike_indices: np.ndarray):
        """
        HOOK: Called with the set of global spike indices inside the lasso.
        Phase 2: call DataManager.split_cluster(cluster_id, spike_indices).
        """
        logger.debug(
            "Lasso selected %d spikes for cluster %s",
            len(spike_indices),
            self._current_cluster_id,
        )
        # TODO: wire to split/merge dialog

    def _render_neighbour_sparklines(
        self, cluster_id: int, median_ei: np.ndarray, dom_chan: int, dm
    ):
        """
        HOOK (Phase 2): Populate self._sparkline_row with tiny PlotWidgets
        showing the median waveform on the ±1 nearest-neighbour channels.
        Called at the end of _render_tier1 when median_ei is available.
        """
        pass  # TODO: get_nearest_channels → draw sparklines
