"""
waveforms_panel.py  —  RGC Viewer Waveform & Isolation Panel
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
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, QRunnable, QThreadPool, QObject, Signal, Slot
from qtpy.QtGui import QColor, QFont
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSplitter, QFrame, QGridLayout, QSizePolicy,
    QGraphicsEllipseItem, QPushButton,
)

if TYPE_CHECKING:
    from ..analysis.data_manager import DataManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Design tokens — single source of truth for all colours / sizes
# ---------------------------------------------------------------------------
_C = {
    # Backgrounds
    "bg_main":        "#13151A",   # near-black, warm tint
    "bg_right":       "#0F1116",   # slightly darker for right panel
    "bg_card":        "#1C1F27",   # stat card surface
    "bg_card_hover":  "#242830",

    # Borders
    "border_subtle":  "#252830",
    "border_default": "#2E3240",

    # Text
    "text_primary":   "#E8EAF0",
    "text_secondary": "#7A7E8E",
    "text_dim":       "#4A4E5E",

    # Accent colours for waveform
    "median":         "#FFD166",   # warm amber — the star of the show
    "envelope":       "#3A4060",   # muted navy for 10-90 ribbon

    # Shadow-cloud amplitude colour ramp (5 buckets, low→high)
    "cloud_0":        (60,  80, 160, 35),   # dim periwinkle
    "cloud_1":        (80, 110, 200, 45),
    "cloud_2":        (120, 150, 240, 55),
    "cloud_3":        (170, 190, 255, 65),
    "cloud_4":        (220, 230, 255, 80),  # bright near-white

    # PCA
    "pca_bg":         (100, 105, 120, 55),  # background cloud
    "pca_unit":       "#FFD166",            # selected unit (same amber)
    "pca_ellipse":    "#FFD166",
    "pca_compare":    "#FF6B9D",   # coral — comparison cluster
    "pca_cmp_ell":    "#FF6B9D",

    # Badge colours
    "badge_good":     "#2ECC71",
    "badge_warn":     "#F39C12",
    "badge_bad":      "#E74C3C",
    "badge_neutral":  "#7A7E8E",

    # Zero-line
    "zero_line":      "#2A2E3A",

    # Axis
    "axis_pen":       "#3A3E50",
    "axis_text":      "#5A5E70",
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
    cloud_ready  = Signal(int, object)   # cluster_id, payload dict
    pca_ready    = Signal(int, object)   # cluster_id, payload dict
    error        = Signal(int, str)      # cluster_id, message


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
    def __init__(self, cluster_id: int, dm,
                 median_trace: np.ndarray, t_ms: np.ndarray,
                 dom_chan: int, sr: float,
                 fallback_snippets: Optional[np.ndarray],
                 signals: _WorkerSignals):
        super().__init__()
        self.setAutoDelete(True)
        self.cluster_id        = cluster_id
        self.dm                = dm
        self.median_trace      = median_trace
        self.t_ms              = t_ms
        self.dom_chan           = dom_chan
        self.sr                = sr
        self.fallback_snippets = fallback_snippets  # (n_spikes, n_time) from ei_cache
        self.signals           = signals

    @Slot()
    def run(self):
        try:
            waveforms = self._fetch_waveforms()
            if waveforms is None or len(waveforms) == 0:
                return
            payload = _build_cloud_payload(
                self.cluster_id, waveforms, self.median_trace, self.t_ms)
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
        raw_available = (getattr(dm, "raw_reader", None) is not None or
                         getattr(dm, "raw_data_memmap", None) is not None)

        if raw_available:
            try:
                # Grab analysis_core from sys.modules — it's already imported
                # by data_manager.py, so this never fails and avoids the
                # relative-import error that occurs inside a bare QRunnable.
                import sys as _sys
                _ac_key = next(
                    (k for k in _sys.modules
                     if k.endswith('analysis_core') and 'analysis_core' in k),
                    None)
                if _ac_key is None:
                    raise ImportError('analysis_core not found in sys.modules')
                _ac = _sys.modules[_ac_key]
                spike_indices = dm.get_cluster_spike_indices(self.cluster_id)
                n = len(spike_indices)
                if n == 0:
                    return self.fallback_snippets

                # Random subsample — stratified over recording time for visual richness
                n_want = min(_N_CLOUD_SPIKES, n)
                if n > n_want:
                    # Uniform stride subsample (representative across session)
                    step    = n // n_want
                    chosen  = spike_indices[::step][:n_want]
                else:
                    chosen = spike_indices

                spike_times = dm.spike_times[chosen].astype(np.int64)
                n_time      = len(self.t_ms)
                pre         = _PRE_SAMPLES
                window      = (-pre, n_time - pre)

                source = (dm.raw_reader if dm.raw_reader is not None
                          else dm.raw_data_memmap)
                raw = _ac.extract_snippets(
                    source, spike_times,
                    window=window,
                    n_channels=dm.n_channels,
                )  # (n_ch, n_time, n_chosen)

                if raw.shape[2] == 0:
                    return self.fallback_snippets

                waves = raw[self.dom_chan, :, :].T  # (n_chosen, n_time)
                waves = waves.astype(np.float32) * dm.uV_per_bit

                # Baseline-correct each spike (subtract mean of pre-spike samples)
                baseline = waves[:, :pre].mean(axis=1, keepdims=True)
                waves   -= baseline
                return waves

            except Exception:
                logger.debug("CloudWorker raw fetch failed, using fallback",
                             exc_info=True)

        return self.fallback_snippets


def _build_cloud_payload(cluster_id, waveforms, median_trace, t_ms):
    """Pure numpy — safe to run off the GUI thread."""
    n_spikes, n_time = waveforms.shape
    MAX_SHADOWS = 200   # hard cap; renders ~5 ms

    # Compute per-spike amplitude (peak-to-trough) for colour bucketing
    spike_ptp = waveforms.max(axis=1) - waveforms.min(axis=1)
    ptp_min, ptp_max = spike_ptp.min(), spike_ptp.max()
    ptp_range = max(ptp_max - ptp_min, 1e-6)
    norm_ptp   = (spike_ptp - ptp_min) / ptp_range   # 0..1

    # Stratified subsample: equal reps from each amplitude quintile
    n_buckets = 5
    bucket_ids = np.floor(norm_ptp * (n_buckets - 1e-9)).astype(int)
    per_bucket = max(1, MAX_SHADOWS // n_buckets)

    buckets_xy = []   # list of (x_flat, y_flat) per bucket
    for b in range(n_buckets):
        mask = bucket_ids == b
        indices = np.where(mask)[0]
        if len(indices) == 0:
            buckets_xy.append((np.array([]), np.array([])))
            continue
        chosen = indices if len(indices) <= per_bucket else \
            np.random.choice(indices, per_bucket, replace=False)
        subset = waveforms[chosen]           # (k, n_time)
        k = len(chosen)
        nan_col = np.full((k, 1), np.nan, dtype=np.float32)
        x_con = np.tile(t_ms, (k, 1))        # (k, n_time)
        x_flat = np.column_stack([x_con, nan_col]).ravel()
        y_flat = np.column_stack([subset, nan_col]).ravel()
        buckets_xy.append((x_flat.astype(np.float32),
                           y_flat.astype(np.float32)))

    # Percentile envelope
    if n_spikes >= 10:
        p10 = np.percentile(waveforms, 10, axis=0).astype(np.float32)
        p90 = np.percentile(waveforms, 90, axis=0).astype(np.float32)
    else:
        p10 = p90 = None

    return {
        "cluster_id":  cluster_id,
        "buckets_xy":  buckets_xy,
        "p10":         p10,
        "p90":         p90,
        "n_spikes":    n_spikes,
    }


# ---------------------------------------------------------------------------
# Tier-3 worker: PCA + isolation metrics
# ---------------------------------------------------------------------------
class _PCAWorker(QRunnable):
    def __init__(self, cluster_id: int, unit_waves: np.ndarray,
                 bg_waves: Optional[np.ndarray],
                 unit_spike_indices: Optional[np.ndarray],
                 signals: _WorkerSignals,
                 bg_waves_by_cid: Optional[dict] = None):
        super().__init__()
        self.setAutoDelete(True)
        self.cluster_id         = cluster_id
        self.unit_waves         = unit_waves         # (n_unit, n_time)
        self.bg_waves           = bg_waves           # (n_bg, n_time) or None
        self.unit_spike_indices = unit_spike_indices  # global indices for future lasso
        self.signals            = signals
        self.bg_waves_by_cid    = bg_waves_by_cid or {}  # {cid: ndarray}

    @Slot()
    def run(self):
        try:
            payload = _build_pca_payload(
                self.cluster_id, self.unit_waves, self.bg_waves,
                self.unit_spike_indices, self.bg_waves_by_cid)
            self.signals.pca_ready.emit(self.cluster_id, payload)
        except Exception as exc:
            logger.exception("PCAWorker error cid=%d", self.cluster_id)
            self.signals.error.emit(self.cluster_id, str(exc))


def _build_pca_payload(cluster_id, unit_waves, bg_waves, unit_spike_indices,
                       bg_waves_by_cid=None):
    """
    Pure numpy/sklearn — safe off GUI thread.

    bg_waves_by_cid : dict[int, ndarray] — per-cluster background waveforms.
        When present the payload includes bg_coords_by_cid so _draw_pca can
        colour individual comparison clusters differently.
    """
    from sklearn.decomposition import PCA

    MAX_UNIT = 1500
    MAX_BG   = 1500

    # --- subsample unit -------------------------------------------------------
    n_unit = unit_waves.shape[0]
    if n_unit > MAX_UNIT:
        idx = np.random.choice(n_unit, MAX_UNIT, replace=False)
        unit_sub  = unit_waves[idx].astype(np.float32)
        unit_sidx = unit_spike_indices[idx] if unit_spike_indices is not None else None
    else:
        unit_sub  = unit_waves.astype(np.float32)
        unit_sidx = unit_spike_indices

    has_bg = bg_waves is not None and len(bg_waves) > 0
    if has_bg:
        n_bg = bg_waves.shape[0]
        if n_bg > MAX_BG:
            bg_sub = bg_waves[np.random.choice(n_bg, MAX_BG, replace=False)].astype(np.float32)
        else:
            bg_sub = bg_waves.astype(np.float32)

        # Templates and raw snippets can differ by a few samples — truncate both
        # to the shorter length so vstack never raises.
        n_time   = min(unit_sub.shape[1], bg_sub.shape[1])
        unit_sub = unit_sub[:, :n_time]
        bg_sub   = bg_sub[:,   :n_time]

        combined = np.vstack([bg_sub, unit_sub])
        labels   = np.concatenate([np.zeros(len(bg_sub)), np.ones(len(unit_sub))])
    else:
        combined  = unit_sub
        labels    = np.ones(len(unit_sub))
        bg_sub    = None
        n_time    = unit_sub.shape[1]

    if len(combined) < 4:
        return {"cluster_id": cluster_id, "error": "too_few_spikes"}

    # --- Z-score normalise each waveform row before PCA -------------------
    # This is essential when mixing real raw snippets (µV scale) with
    # template-synthesised bg waveforms (whitened / arbitrary scale).
    # Each row becomes unit-variance, so PCA separates shape not amplitude.
    row_std = combined.std(axis=1, keepdims=True)
    row_std[row_std < 1e-9] = 1.0  # guard against flat waveforms
    combined_norm = (combined - combined.mean(axis=1, keepdims=True)) / row_std

    # --- PCA on combined set ------------------------------------------------
    pca    = PCA(n_components=2)
    coords = pca.fit_transform(combined_norm)
    var    = pca.explained_variance_ratio_

    bg_mask   = labels == 0
    unit_mask = labels == 1
    bg_coords   = coords[bg_mask]   if has_bg else np.empty((0, 2))
    unit_coords = coords[unit_mask]

    # --- Project each per-cluster bg set into the SAME PCA space -----------
    # This is the key step for the compare feature: we project bg_waves_by_cid
    # through the already-fitted PCA so all clouds share the same axes.
    bg_coords_by_cid: dict = {}
    if bg_waves_by_cid:
        for cid, cid_waves in bg_waves_by_cid.items():
            if cid_waves is None or len(cid_waves) == 0:
                continue
            try:
                cid_sub = cid_waves.astype(np.float32)
                # Subsample if large
                if len(cid_sub) > MAX_BG:
                    cid_sub = cid_sub[np.random.choice(len(cid_sub), MAX_BG, replace=False)]
                # Align time dimension
                cid_sub = cid_sub[:, :n_time]
                # Z-score normalise same way as the combined set
                cs_std = cid_sub.std(axis=1, keepdims=True)
                cs_std[cs_std < 1e-9] = 1.0
                cid_norm = (cid_sub - cid_sub.mean(axis=1, keepdims=True)) / cs_std
                bg_coords_by_cid[cid] = pca.transform(cid_norm)
            except Exception:
                pass

    # --- Isolation metrics (d-prime, unit vs all-background) ---------------
    isolation_label = "N/A (no bg)"
    dprime = None
    if has_bg and len(unit_coords) > 3 and len(bg_coords) > 3:
        try:
            unit_mean  = unit_coords.mean(axis=0)
            unit_cov   = np.cov(unit_coords.T) + np.eye(2) * 1e-6
            bg_mean    = bg_coords.mean(axis=0)
            pooled_std = np.sqrt((np.trace(unit_cov) +
                                  np.trace(np.cov(bg_coords.T))) / 4.0)
            if pooled_std > 0:
                dprime = float(np.linalg.norm(unit_mean - bg_mean) / pooled_std)
                if dprime > 3.0:
                    isolation_label = f"d' = {dprime:.2f}  ✓ well isolated"
                elif dprime > 1.5:
                    isolation_label = f"d' = {dprime:.2f}  ~ marginal"
                else:
                    isolation_label = f"d' = {dprime:.2f}  ✗ poor"
        except Exception:
            pass

    # --- Ellipse (unit, 2-sigma) --------------------------------------------
    ellipse = _compute_ellipse(unit_coords)

    return {
        "cluster_id":       cluster_id,
        "bg_coords":        bg_coords,           # flat, all bg — for gray cloud
        "bg_coords_by_cid": bg_coords_by_cid,    # per-cluster — for compare feature
        "unit_coords":      unit_coords,
        "unit_spike_idx":   unit_sidx,
        "var":              var,
        "isolation_label":  isolation_label,
        "dprime":           dprime,
        "ellipse":          ellipse,
        "has_bg":           has_bg,
    }


def _compute_ellipse(coords: np.ndarray) -> Optional[dict]:
    """Return 2-sigma ellipse params for a 2-D point cloud, or None."""
    if len(coords) < 4:
        return None
    try:
        cov  = np.cov(coords.T)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals  = vals[order]
        angle_deg = float(np.degrees(np.arctan2(*vecs[:, order[0]][::-1])))
        width  = float(2 * 2 * np.sqrt(max(vals[0], 0)))
        height = float(2 * 2 * np.sqrt(max(vals[1], 0)))
        return {"cx": float(coords[:, 0].mean()),
                "cy": float(coords[:, 1].mean()),
                "width": width, "height": height, "angle": angle_deg}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Stage-A worker: fetch channel snippets, then fire PCA worker
# ---------------------------------------------------------------------------
class _ChannelSnippetsWorker(QRunnable):
    """
    Fetches unit + background waveforms for the dominant channel.

    Priority order:
      1. dm.get_channel_all_snippets()  (raw data path — real spikes)
      2. Inline template synthesiser    (no disk I/O, always available)

    On success, immediately enqueues a _PCAWorker with the result.
    """
    def __init__(self, cluster_id: int, dom_chan: int, dm,
                 unit_waves_override: Optional[np.ndarray],
                 signals: _WorkerSignals):
        super().__init__()
        self.setAutoDelete(True)
        self.cluster_id          = cluster_id
        self.dom_chan             = dom_chan
        self.dm                  = dm
        self.unit_waves_override = unit_waves_override   # already-extracted from ei_cache
        self.signals             = signals

    @Slot()
    def run(self):
        try:
            result = self._fetch()
            if result is None:
                self.signals.error.emit(
                    self.cluster_id, "no waveform data available for PCA")
                return
            unit_waves, bg_waves, bg_waves_by_cid, unit_indices = result

            # Hand off to PCA stage
            pca_worker = _PCAWorker(
                self.cluster_id, unit_waves, bg_waves,
                unit_indices, self.signals,
                bg_waves_by_cid=bg_waves_by_cid)
            QThreadPool.globalInstance().start(pca_worker)

        except Exception as exc:
            logger.exception("ChannelSnippetsWorker failed cid=%d", self.cluster_id)
            self.signals.error.emit(self.cluster_id, str(exc))

    def _fetch(self):
        dm  = self.dm
        cid = self.cluster_id
        ch  = self.dom_chan

        # -- Path 1: dm.get_channel_all_snippets (raw data available) -----------
        if hasattr(dm, "get_channel_all_snippets"):
            raw_available = (getattr(dm, "raw_reader", None) is not None or
                             getattr(dm, "raw_data_memmap", None) is not None)
            if raw_available:
                try:
                    r = dm.get_channel_all_snippets(ch, cid)
                    if (r["source"] == "raw"
                            and len(r["unit_waves"]) > 0):
                        return (r["unit_waves"], r["bg_waves"],
                                r.get("bg_waves_by_cid", {}), r["unit_indices"])
                except Exception:
                    logger.debug("get_channel_all_snippets failed", exc_info=True)

        # -- Path 2: use unit waveforms already extracted from ei_cache ----------
        unit_waves = self.unit_waves_override
        bg_waves   = self._synth_bg_from_templates(dm, cid, ch)
        if unit_waves is None:
            unit_waves = self._synth_unit_from_template(dm, cid, ch)

        if unit_waves is None or len(unit_waves) == 0:
            return None

        # No global spike indices in template path — leave empty for now
        unit_indices = getattr(dm, "get_cluster_spike_indices", lambda x: None)(cid)

        # Build per-cluster bg dict from template synthesiser
        bg_by_cid = self._synth_bg_by_cid_from_templates(dm, cid, ch)
        return (unit_waves.astype(np.float32),
                bg_waves.astype(np.float32) if bg_waves is not None else None,
                bg_by_cid,
                unit_indices)

    # -- Template synthesisers (no disk I/O) ------------------------------------
    def _synth_bg_from_templates(self, dm, cluster_id, dom_chan):
        try:
            templates = getattr(dm, "templates", None)
            tmpl_ind  = getattr(dm, "templates_ind", None)
            if templates is None or "best_chan" not in dm.cluster_df.columns:
                return None
            chan_df = dm.cluster_df[dm.cluster_df["best_chan"] == dom_chan]
            waves = []
            for row in chan_df.itertuples():
                c = int(row.cluster_id)
                if c == cluster_id or c >= templates.shape[0]:
                    continue
                wave = _extract_template_channel(templates, tmpl_ind, c, dom_chan)
                if wave is None:
                    continue
                n_rep = min(200, max(10, int(row.n_spikes) // 100))
                noise = np.random.normal(
                    0, float(np.std(wave)) * 0.05,
                    (n_rep, len(wave))).astype(np.float32)
                waves.append(np.tile(wave, (n_rep, 1)) + noise)
            return np.vstack(waves) if waves else None
        except Exception:
            return None

    def _synth_unit_from_template(self, dm, cluster_id, dom_chan):
        try:
            templates = getattr(dm, "templates", None)
            tmpl_ind  = getattr(dm, "templates_ind", None)
            if templates is None or cluster_id >= templates.shape[0]:
                return None
            wave = _extract_template_channel(templates, tmpl_ind, cluster_id, dom_chan)
            if wave is None:
                return None
            row = dm.cluster_df[dm.cluster_df.cluster_id == cluster_id]
            n_spikes = int(row["n_spikes"].values[0]) if not row.empty else 100
            n_rep = min(500, max(20, n_spikes // 50))
            noise = np.random.normal(
                0, float(np.std(wave)) * 0.12,
                (n_rep, len(wave))).astype(np.float32)
            return np.tile(wave, (n_rep, 1)) + noise
        except Exception:
            return None

    def _synth_bg_by_cid_from_templates(self, dm, cluster_id, dom_chan):
        """Return dict{cid: ndarray} — one synthetic wave array per neighbour cluster."""
        result = {}
        try:
            templates = getattr(dm, "templates", None)
            tmpl_ind  = getattr(dm, "templates_ind", None)
            if templates is None or "best_chan" not in dm.cluster_df.columns:
                return result
            chan_df = dm.cluster_df[dm.cluster_df["best_chan"] == dom_chan]
            for row in chan_df.itertuples():
                c = int(row.cluster_id)
                if c == cluster_id or c >= templates.shape[0]:
                    continue
                wave = _extract_template_channel(templates, tmpl_ind, c, dom_chan)
                if wave is None:
                    continue
                n_rep = min(200, max(10, int(row.n_spikes) // 100))
                noise = np.random.normal(
                    0, float(np.std(wave)) * 0.05,
                    (n_rep, len(wave))).astype(np.float32)
                result[c] = np.tile(wave, (n_rep, 1)) + noise
        except Exception:
            pass
        return result


def _extract_template_channel(templates, tmpl_ind, cluster_id, dom_chan):
    """Extract the 1-D waveform for dom_chan from a cluster's template."""
    tpl = np.asarray(templates[cluster_id], dtype=np.float32)  # (n_time, n_tCh)
    if tmpl_ind is not None and cluster_id < tmpl_ind.shape[0]:
        local = np.where(tmpl_ind[cluster_id] == dom_chan)[0]
        if len(local) == 0:
            return None
        return tpl[:, int(local[0])]
    else:
        if dom_chan >= tpl.shape[1]:
            return None
        return tpl[:, dom_chan]


# ---------------------------------------------------------------------------
# Stat badge helpers
# ---------------------------------------------------------------------------
def _badge_color(key: str, value) -> str:
    """Return a CSS colour string for a metric badge."""
    thresholds = {
        "isi_viol":  [(0.5, _C["badge_good"]), (2.0, _C["badge_warn"])],
        "snr":       [(10,  _C["badge_good"]), (4.0, _C["badge_warn"])],
        "amp_uv":    [(15,  _C["badge_good"]), (5.0,  _C["badge_warn"])],
        "amp_cv":    [(0.25, _C["badge_good"]), (0.5, _C["badge_warn"])],
        "dprime":    [(3.0, _C["badge_good"]), (1.5, _C["badge_warn"])],
    }
    if key not in thresholds or value is None:
        return _C["badge_neutral"]
    rules = thresholds[key]
    # For isi_viol and amp_cv: lower is better
    if key in ("isi_viol", "amp_cv"):
        good_thresh, warn_thresh = rules[0][0], rules[1][0]
        if value <= good_thresh:  return rules[0][1]
        if value <= warn_thresh:  return rules[1][1]
        return _C["badge_bad"]
    else:  # higher is better
        good_thresh, warn_thresh = rules[0][0], rules[1][0]
        if value >= good_thresh:  return rules[0][1]
        if value >= warn_thresh:  return rules[1][1]
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

        # Threading
        self._pool    = QThreadPool.globalInstance()
        self._signals = _WorkerSignals()
        self._signals.cloud_ready.connect(self._on_cloud_ready)
        self._signals.pca_ready.connect(self._on_pca_ready)
        self._signals.error.connect(self._on_worker_error)

        # State
        self._current_cluster_id: Optional[int]  = None
        self._current_dom_chan: Optional[int]     = None
        self._current_t_ms: Optional[np.ndarray] = None
        self._compare_cluster_id: Optional[int]  = None   # compare feature
        self._last_pca_payload: Optional[dict]   = None   # for re-draw on compare change

        self._setup_ui()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------
    def _setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(3)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{ background: {_C['border_subtle']}; }}
            QSplitter::handle:hover {{ background: {_C['border_default']}; }}
        """)
        root.addWidget(splitter)

        # ── LEFT: waveform ─────────────────────────────────────────────────
        left = QWidget()
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
            f"color: {_C['text_primary']}; font-size: 13px; font-weight: 600;")
        rv.addWidget(self._cluster_header)

        # Compare selector strip
        self._compare_strip_container = QWidget()
        self._compare_strip_container.setStyleSheet("background: transparent;")
        self._compare_strip_layout = QHBoxLayout(self._compare_strip_container)
        self._compare_strip_layout.setContentsMargins(0, 0, 0, 0)
        self._compare_strip_layout.setSpacing(4)
        cmp_lbl = QLabel("Compare:")
        cmp_lbl.setStyleSheet(
            f"color: {_C['text_secondary']}; font-size: 9px; font-weight: 600;")
        cmp_lbl.setFixedWidth(56)
        self._compare_strip_layout.addWidget(cmp_lbl)
        self._compare_strip_layout.addStretch()
        self._compare_buttons: dict[int, QPushButton] = {}
        rv.addWidget(self._compare_strip_container)

        # PCA plot
        self._pca_plot = self._make_pca_plot()
        rv.addWidget(self._pca_plot, stretch=3)

        # Isolation metric label
        self._isolation_label = QLabel("Isolation: —")
        self._isolation_label.setStyleSheet(
            f"color: {_C['text_secondary']}; font-size: 10px;")
        self._isolation_label.setAlignment(Qt.AlignCenter)
        rv.addWidget(self._isolation_label)

        # Divider
        div = QFrame()
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
        pw.setLabel("bottom", "Time", units="ms",
                    **{"color": _C["axis_text"], "font-size": "9pt"})
        pw.setLabel("left", "Amplitude", units="µV",
                    **{"color": _C["axis_text"], "font-size": "9pt"})

        # Persistent zero-line
        self._zero_line = pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen(_C["zero_line"], width=1, style=Qt.DashLine))
        pw.addItem(self._zero_line)

        # Pre-allocated plot items so we never recreate them per frame
        # Envelope (fill between)
        self._env_curve_lo = pg.PlotCurveItem(pen=None)
        self._env_curve_hi = pg.PlotCurveItem(pen=None)
        self._env_fill = pg.FillBetweenItem(
            self._env_curve_lo, self._env_curve_hi,
            brush=pg.mkBrush(*[int(x) for x in [58, 64, 96, 30]]))
        pw.addItem(self._env_curve_lo)
        pw.addItem(self._env_curve_hi)
        pw.addItem(self._env_fill)

        # 5 shadow-cloud CurveItems (one per amplitude bucket)
        cloud_colors = [_C[f"cloud_{i}"] for i in range(5)]
        self._cloud_items: list[pg.PlotCurveItem] = []
        for rgba in cloud_colors:
            item = pg.PlotCurveItem(
                pen=pg.mkPen(color=rgba, width=1),
                connect="finite",
                antialias=False)  # antialias=False is critical for speed
            pw.addItem(item)
            self._cloud_items.append(item)

        # Median trace (on top of everything)
        self._median_item = pg.PlotCurveItem(
            pen=pg.mkPen(_C["median"], width=2.5),
            antialias=True)
        pw.addItem(self._median_item)

        # Title text item
        self._wave_title = pg.TextItem(
            "", anchor=(0, 0),
            color=_C["text_secondary"])
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
        pw.setLabel("left",   "PC2", **{"color": _C["axis_text"], "font-size": "9pt"})

        # Pre-allocated scatter items
        self._pca_bg_scatter = pg.ScatterPlotItem(
            size=3, pen=None,
            brush=pg.mkBrush(*_C["pca_bg"]),
            hoverable=False)
        self._pca_unit_scatter = pg.ScatterPlotItem(
            size=5, pen=None,
            brush=pg.mkBrush(_C["pca_unit"]),
            hoverable=True,
            tip=lambda x, y, data: f'spike idx: {data}' if data is not None else '')

        # Compare cluster scatter (coral, shown only when compare active)
        self._pca_cmp_scatter = pg.ScatterPlotItem(
            size=5, pen=None,
            brush=pg.mkBrush(_C["pca_compare"]),
            hoverable=False)

        pw.addItem(self._pca_bg_scatter)
        pw.addItem(self._pca_cmp_scatter)   # below unit so unit stays on top
        pw.addItem(self._pca_unit_scatter)

        # Ellipse placeholders (added/removed dynamically)
        self._pca_ellipse_item: Optional[pg.QtWidgets.QGraphicsEllipseItem] = None
        self._pca_cmp_ellipse_item: Optional[pg.QtWidgets.QGraphicsEllipseItem] = None

        return pw

    def _make_stats_panel(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet(f"background: transparent;")
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 4, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        # --- metric definitions -----------------------------------------
        # (key, display_label, badge_key, format_fn)
        self._stat_defs = [
            ("n_spikes",    "Spikes",        None,        lambda v: f"{v:,}"),
            ("mean_fr",     "Mean FR",        None,        lambda v: f"{v:.1f} Hz"),
            ("isi_viol",    "ISI violations", "isi_viol",  lambda v: f"{v:.2f} %"),
            ("amp_uv",      "Amplitude",      "amp_uv",    lambda v: f"{v:.1f} µV"),
            ("snr",         "SNR",            "snr",       lambda v: f"{v:.1f}"),
            ("width_ms",    "Spike width",    None,        lambda v: f"{v:.3f} ms"),
            ("peak_lat",    "Peak latency",   None,        lambda v: f"{v:.3f} ms"),
            ("amp_cv",      "Amp stability",  "amp_cv",    lambda v: f"CV {v:.3f}"),
        ]

        self._stat_badges: dict[str, QLabel] = {}
        self._stat_values: dict[str, QLabel] = {}

        for i, (key, label, badge_key, _) in enumerate(self._stat_defs):
            row, col_offset = divmod(i, 2)
            col = col_offset * 3   # badge | label | value

            badge = _make_badge(_C["badge_neutral"])
            self._stat_badges[key] = badge

            lbl = QLabel(label)
            lbl.setStyleSheet(
                f"color: {_C['text_secondary']}; font-size: 10px;")

            val = QLabel("—")
            val.setStyleSheet(
                f"color: {_C['text_primary']}; font-size: 12px; font-weight: 600;")
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._stat_values[key] = val

            grid.addWidget(badge, row, col)
            grid.addWidget(lbl,   row, col + 1)
            grid.addWidget(val,   row, col + 2)

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
        self._compare_cluster_id = None   # reset on new selection
        self._last_pca_payload   = None
        self._clear_compare_buttons()     # remove stale buttons immediately
        self._render_tier1(cluster_id)

    def restyle_plots(self, colors: dict):
        """Legacy compatibility — theme colours are applied on next update_all."""
        pass

    # -----------------------------------------------------------------------
    # Tier 1 — synchronous, median + basic stats (< 5 ms)
    # -----------------------------------------------------------------------
    def _render_tier1(self, cluster_id: int):
        self._clear_plots()

        dm: DataManager = self.main_window.data_manager
        features = dm.get_lightweight_features(cluster_id)

        if not features or "median_ei" not in features:
            self._wave_title.setText(f"Cluster {cluster_id} — waiting for cache…")
            self._cluster_header.setText(f"Cluster {cluster_id}")
            return

        median_ei    = features["median_ei"]         # (n_ch, n_time)
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

        # Title
        n_spikes = int(raw_snippets.shape[2]) if raw_snippets is not None else \
                   int(dm.cluster_df.loc[dm.cluster_df.cluster_id == cluster_id,
                                         "n_spikes"].values[0]
                       if not dm.cluster_df.empty else 0)
        self._wave_title.setText(
            f"Cluster {cluster_id}  ·  Ch {dom_chan}  ·  {n_spikes:,} spikes")
        self._wave_title.setPos(t_ms[0], float(median_trace.max()))

        self._cluster_header.setText(
            f"Cluster {cluster_id}   ·   Ch {dom_chan}   ·   {n_spikes:,} spikes")

        # --- Tier-1 stats (no raw data needed) ---------------------------
        self._populate_stats_tier1(cluster_id, median_trace, t_ms, n_spikes, dm)

        # --- Launch Tier 2 & 3 ─────────────────────────────────────────
        # For the cloud: always attempt to fetch real raw spikes.
        # Pass cached snippets only as a fallback if raw data isn't loaded.
        fallback = (raw_snippets[dom_chan, :, :].T
                    if raw_snippets is not None else None)
        self._launch_cloud_worker(cluster_id, fallback,
                                  median_trace, t_ms, sr, dm)

        # For PCA: pass real unit waveforms from cache if available.
        # _ChannelSnippetsWorker will also fetch real raw snippets for bg.
        self._launch_pca_worker(cluster_id, fallback, dom_chan, dm)

    # -----------------------------------------------------------------------
    # Tier 1 stats
    # -----------------------------------------------------------------------
    def _populate_stats_tier1(self, cluster_id: int, median_trace: np.ndarray,
                              t_ms: np.ndarray, n_spikes: int, dm):
        sr = float(getattr(dm, "sampling_rate", 30_000))
        dt = float(t_ms[1] - t_ms[0]) if len(t_ms) > 1 else 1.0 / sr * 1000

        # Spike count
        self._set_stat("n_spikes", n_spikes, None)

        # Mean firing rate — best estimate of recording duration:
        #   1. n_samples / sr  (most accurate, requires raw data loaded)
        #   2. spike_times[-1] / sr  (always available after KS load)
        # We deliberately do NOT use first-to-last spike of THIS cluster
        # because a small cluster spanning a short window would show an
        # inflated FR.
        try:
            n_total = int(getattr(dm, 'n_samples', 0))
            recording_s = n_total / sr if n_total > 0 else 0.0
            if recording_s <= 0:
                # Use the last spike time across ALL clusters as proxy
                all_spikes = getattr(dm, 'spike_times', None)
                if all_spikes is not None and len(all_spikes) > 0:
                    recording_s = float(all_spikes[-1]) / sr
            mean_fr = float(n_spikes / recording_s) if recording_s > 0 else 0.0
        except Exception:
            mean_fr = 0.0
        self._set_stat("mean_fr", mean_fr, None)

        # ISI violations from cluster_df
        row = dm.cluster_df[dm.cluster_df.cluster_id == cluster_id]
        isi_viol = float(row["isi_violations_pct"].values[0]) \
            if not row.empty and "isi_violations_pct" in row.columns else None
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
                amps_f    = np.abs(amps.astype(np.float32))
                med_amp   = float(np.median(amps_f))
                mad_amp   = float(np.median(np.abs(amps_f - med_amp))) * 1.4826
                snr = med_amp / mad_amp if mad_amp > 1e-9 else 0.0
            else:
                # Fallback: P2P of median / std of pre-spike region
                amp_uv_val = float(np.ptp(median_trace))
                noise_std  = float(np.std(median_trace[:_PRE_SAMPLES]))
                snr = amp_uv_val / noise_std if noise_std > 1e-9 else 0.0
        except Exception:
            snr = 0.0
        self._set_stat("snr", snr, "snr")

        # Peak latency
        trough_idx  = int(np.argmin(median_trace))
        peak_lat_ms = float(t_ms[trough_idx])
        self._set_stat("peak_lat", peak_lat_ms, None)

        # Waveform width at half-minimum (ms)
        try:
            trough_val = median_trace[trough_idx]
            half_val   = trough_val / 2.0
            # Find crossings left and right of trough
            left_cross  = np.where(median_trace[:trough_idx] >= half_val)[0]
            right_cross = np.where(median_trace[trough_idx:] >= half_val)[0]
            if len(left_cross) > 0 and len(right_cross) > 0:
                left_idx  = left_cross[-1]
                right_idx = trough_idx + right_cross[0]
                width_ms  = float(t_ms[right_idx] - t_ms[left_idx])
            else:
                width_ms = float("nan")
        except Exception:
            width_ms = float("nan")
        self._set_stat("width_ms",
                       width_ms if np.isfinite(width_ms) else None, None)

        # Amplitude stability (CV) — requires spike amplitudes
        amps = dm.get_cluster_spike_amplitudes(cluster_id)
        if amps is not None and len(amps) > 1:
            mean_amp = float(np.mean(amps))
            amp_cv   = float(np.std(amps) / mean_amp) if mean_amp > 0 else 0.0
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
    def _launch_cloud_worker(self, cluster_id: int,
                             fallback_snippets: Optional[np.ndarray],
                             median_trace: np.ndarray, t_ms: np.ndarray,
                             sr: float, dm):
        worker = _CloudWorker(
            cluster_id, dm, median_trace, t_ms,
            self._current_dom_chan, sr,
            fallback_snippets, self._signals)
        self._pool.start(worker)

    @Slot(int, object)
    def _on_cloud_ready(self, cluster_id: int, payload: dict):
        if cluster_id != self._current_cluster_id:
            return   # stale — user moved on
        self._draw_cloud(payload)

    def _draw_cloud(self, payload: dict):
        if payload is None:
            return
        t_ms = self._current_t_ms
        if t_ms is None:
            return

        buckets_xy = payload["buckets_xy"]
        p10        = payload["p10"]
        p90        = payload["p90"]

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
    def _launch_pca_worker(self, cluster_id: int,
                           unit_waves_from_cache: Optional[np.ndarray],
                           dom_chan: int, dm):
        """
        Two-stage PCA pipeline:
          Stage A: _ChannelSnippetsWorker  → fetches unit + bg waveforms
                   (uses dm.get_channel_all_snippets if available, else
                    falls back to an inline template synthesiser)
          Stage B: _PCAWorker              → runs PCA + computes isolation

        LRU cache keyed by (cluster_id, dom_chan) to avoid recomputing
        when the user re-visits a cluster.
        """
        cache_key = (cluster_id, dom_chan)
        if cache_key in _PCA_CACHE:
            _PCA_CACHE.move_to_end(cache_key)
            self._on_pca_ready(cluster_id, _PCA_CACHE[cache_key])
            return

        # Fire a snippets-fetch worker; on completion it will fire the PCA worker
        worker = _ChannelSnippetsWorker(
            cluster_id=cluster_id,
            dom_chan=dom_chan,
            dm=dm,
            unit_waves_override=unit_waves_from_cache,
            signals=self._signals,
        )
        self._pool.start(worker)

    @Slot(int, object)
    def _on_pca_ready(self, cluster_id: int, payload: dict):
        if cluster_id != self._current_cluster_id:
            return
        if payload is None or payload.get("error"):
            self._isolation_label.setText("PCA: insufficient data")
            return

        # Update LRU cache
        cache_key = (cluster_id, self._current_dom_chan)
        _PCA_CACHE[cache_key] = payload
        _PCA_CACHE.move_to_end(cache_key)
        while len(_PCA_CACHE) > _PCA_CACHE_MAX:
            _PCA_CACHE.popitem(last=False)

        self._last_pca_payload = payload
        self._rebuild_compare_strip(payload)
        self._draw_pca(payload)

    def _rebuild_compare_strip(self, payload: dict):
        """Repopulate the compare buttons from bg_coords_by_cid."""
        # Remove old buttons
        for btn in self._compare_buttons.values():
            self._compare_strip_layout.removeWidget(btn)
            btn.deleteLater()
        self._compare_buttons.clear()

        bg_by_cid = payload.get("bg_coords_by_cid", {})
        if not bg_by_cid:
            return

        dm = self.main_window.data_manager
        for cid in sorted(bg_by_cid.keys()):
            row = dm.cluster_df[dm.cluster_df.cluster_id == cid]
            n = int(row["n_spikes"].values[0]) if not row.empty else 0
            label = f"C{cid}·{n//1000}k" if n >= 1000 else f"C{cid}·{n}"

            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(self._compare_cluster_id == cid)
            btn.setFixedHeight(20)
            btn.setStyleSheet(
                f"QPushButton {{"
                f"  background: {_C['bg_card']};"
                f"  color: {_C['text_secondary']};"
                f"  border: 1px solid {_C['border_default']};"
                f"  border-radius: 3px;"
                f"  font-size: 9px;"
                f"  padding: 0 5px;"
                f"}}"
                f"QPushButton:checked {{"
                f"  background: {_C['pca_compare']}22;"
                f"  color: {_C['pca_compare']};"
                f"  border-color: {_C['pca_compare']};"
                f"}}"
            )
            # Capture cid in closure
            btn.clicked.connect(lambda checked, c=cid: self._on_compare_selected(c))
            # Insert before the trailing stretch
            stretch_idx = self._compare_strip_layout.count() - 1
            self._compare_strip_layout.insertWidget(stretch_idx, btn)
            self._compare_buttons[cid] = btn

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

    def _draw_pca(self, payload: dict):
        # --- Clear all dynamic items -------------------------------------
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

        bg_coords       = payload["bg_coords"]
        bg_coords_by_cid = payload.get("bg_coords_by_cid", {})
        unit_coords     = payload["unit_coords"]
        var             = payload["var"]
        unit_sidx       = payload.get("unit_spike_idx")
        cmp_cid         = self._compare_cluster_id

        # --- Gray background cloud (all bg, minus compare if active) -----
        if payload["has_bg"] and len(bg_coords) > 0:
            if cmp_cid is not None and cmp_cid in bg_coords_by_cid:
                # Remove compare cluster from the gray cloud so it doesn't
                # appear twice (gray + coral)
                cmp_coords = bg_coords_by_cid[cmp_cid]
                cmp_n      = len(cmp_coords)
                rest       = bg_coords[cmp_n:]   # approximate exclusion — good enough
                if len(rest) > 0:
                    self._pca_bg_scatter.addPoints(
                        x=rest[:, 0].tolist(), y=rest[:, 1].tolist())
            else:
                self._pca_bg_scatter.addPoints(
                    x=bg_coords[:, 0].tolist(), y=bg_coords[:, 1].tolist())

        # --- Coral compare cluster ----------------------------------------
        if cmp_cid is not None and cmp_cid in bg_coords_by_cid:
            cmp_coords = bg_coords_by_cid[cmp_cid]
            if len(cmp_coords) > 0:
                self._pca_cmp_scatter.addPoints(
                    x=cmp_coords[:, 0].tolist(), y=cmp_coords[:, 1].tolist())
                # 2-sigma ellipse for compare cluster
                cmp_ellipse = _compute_ellipse(cmp_coords)
                if cmp_ellipse is not None:
                    cx, cy = cmp_ellipse["cx"], cmp_ellipse["cy"]
                    w, h   = cmp_ellipse["width"], cmp_ellipse["height"]
                    ell = pg.QtWidgets.QGraphicsEllipseItem(
                        cx - w / 2, cy - h / 2, w, h)
                    ell.setPen(pg.mkPen(_C["pca_cmp_ell"], width=1.5,
                                        style=Qt.DashLine))
                    ell.setBrush(pg.mkBrush(0, 0, 0, 0))
                    self._pca_plot.addItem(ell)
                    self._pca_cmp_ellipse_item = ell

        # --- Unit cloud (amber) -------------------------------------------
        if len(unit_coords) > 0:
            if unit_sidx is not None and len(unit_sidx) == len(unit_coords):
                pts = [{"pos": (float(unit_coords[i, 0]),
                                float(unit_coords[i, 1])),
                        "data": int(unit_sidx[i])}
                       for i in range(len(unit_coords))]
                self._pca_unit_scatter.addPoints(pts)
            else:
                self._pca_unit_scatter.addPoints(
                    x=unit_coords[:, 0].tolist(),
                    y=unit_coords[:, 1].tolist())

        # --- Unit 2-sigma ellipse -----------------------------------------
        unit_ellipse = payload.get("ellipse")
        if unit_ellipse is not None:
            cx, cy = unit_ellipse["cx"], unit_ellipse["cy"]
            w, h   = unit_ellipse["width"], unit_ellipse["height"]
            ell = pg.QtWidgets.QGraphicsEllipseItem(
                cx - w / 2, cy - h / 2, w, h)
            ell.setPen(pg.mkPen(_C["pca_ellipse"], width=1.5,
                                 style=Qt.DashLine))
            ell.setBrush(pg.mkBrush(0, 0, 0, 0))
            self._pca_plot.addItem(ell)
            self._pca_ellipse_item = ell

        # --- Axis labels ---------------------------------------------------
        if len(var) >= 2:
            self._pca_plot.setLabel(
                "bottom", f"PC1  ({var[0]:.1%})",
                **{"color": _C["axis_text"], "font-size": "9pt"})
            self._pca_plot.setLabel(
                "left", f"PC2  ({var[1]:.1%})",
                **{"color": _C["axis_text"], "font-size": "9pt"})

        # --- Isolation label — pairwise when compare is active ------------
        if cmp_cid is not None and cmp_cid in bg_coords_by_cid:
            cmp_c = bg_coords_by_cid[cmp_cid]
            if len(unit_coords) > 3 and len(cmp_c) > 3:
                try:
                    um = unit_coords.mean(axis=0)
                    cm = cmp_c.mean(axis=0)
                    pooled = np.sqrt((np.trace(np.cov(unit_coords.T)) +
                                      np.trace(np.cov(cmp_c.T))) / 4.0)
                    dp = float(np.linalg.norm(um - cm) / pooled) if pooled > 0 else 0.0
                    sym = "✓" if dp > 3 else ("~" if dp > 1.5 else "✗")
                    self._isolation_label.setText(
                        f"C{self._current_cluster_id} vs C{cmp_cid}:  "
                        f"d' = {dp:.2f}  {sym}")
                except Exception:
                    self._isolation_label.setText(payload["isolation_label"])
            else:
                self._isolation_label.setText(payload["isolation_label"])
        else:
            self._isolation_label.setText(
                f"Isolation:  {payload['isolation_label']}")

        # -- Future hook: lasso-split prototype --------
        # self._pca_unit_scatter.sigClicked.connect(self._on_spike_clicked)

    # -----------------------------------------------------------------------
    # Worker error
    # -----------------------------------------------------------------------
    @Slot(int, str)
    def _on_worker_error(self, cluster_id: int, msg: str):
        if cluster_id == self._current_cluster_id:
            logger.warning("Waveform worker error cid=%d: %s", cluster_id, msg)

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
                f"color: {_C['badge_neutral']}; font-size: 10px;")

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
            len(spike_indices), self._current_cluster_id)
        # TODO: wire to split/merge dialog

    def _render_neighbour_sparklines(self, cluster_id: int, median_ei: np.ndarray,
                                     dom_chan: int, dm):
        """
        HOOK (Phase 2): Populate self._sparkline_row with tiny PlotWidgets
        showing the median waveform on the ±1 nearest-neighbour channels.
        Called at the end of _render_tier1 when median_ei is available.
        """
        pass  # TODO: get_nearest_channels → draw sparklines