"""Named, plottable per-cell features for the Feature Extraction scatters.

The Feature Extraction window used to show six hardcoded panels built from two
PCA blocks and two scalars. Everything else the app computes — chirp response
shape, grating tuning, contrast response, the quality metrics — was unavailable
to it, even though those are exactly the axes that separate RGC types.

This module assembles one flat, named catalogue of per-cell features so the
window can offer them all in a picker. Every entry is a 1-D float array aligned
to the same ``valid_ids``, so any two can be plotted against each other.

Two deliberate choices:

*PCA is computed on the same blocks the embedding uses*, via
``get_raw_feature_blocks``, so a PC shown here is the PC that drove the UMAP —
including its zero sentinels. A cell with no STA contributes an all-zero
temporal row and therefore lands at the same point as every other cell with no
STA. That is a real property of the embedding, not an artefact of this module,
and hiding it here would make the scatter disagree with the map it explains.

*Contrast-response features are opt-in.* They cost an FFT per trial per light
level — roughly 8 ms per cell per run, so ~30 s for 900 cells across four
levels on a first pass — against microseconds for everything else. They are
worth it when classifying by contrast gain and pure overhead otherwise.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["FeatureEntry", "build_catalog", "resolve_panels", "DEFAULT_PANELS"]

N_PCS = 3

# What the six panels show before the user picks anything. Keeps the original
# layout's meaning while naming the axes in the new vocabulary.
DEFAULT_PANELS = [
    ("Temporal STA PC1", "Temporal STA PC2"),
    ("RF diameter (long)", "Temporal STA PC1"),
    ("RF diameter (long)", "Time to peak"),
    ("ACG PC1", "ACG PC2"),
    ("RF diameter (long)", "ACG PC1"),
    ("Temporal STA PC1", "ACG PC1"),
]


class FeatureEntry:
    """One plottable feature: a name, the group it belongs to, and its values."""

    __slots__ = ("name", "group", "values", "description")

    def __init__(self, name, group, values, description=""):
        self.name = name
        self.group = group
        self.values = np.asarray(values, dtype=float)
        self.description = description

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        return f"FeatureEntry({self.name!r}, {self.group!r}, n={len(self.values)})"


def reindex_catalog(catalog, from_ids, to_ids):
    """Re-key every feature from *from_ids* row order onto *to_ids*.

    Callers hold other per-cell arrays (traces, mosaics, selection indices)
    keyed to their own id order. Rather than requiring the catalogue to match,
    it is mapped onto theirs; cells the catalogue lacks become NaN and drop out
    of the scatters. Silently allowing two row orders to coexist is how one
    cell's features end up plotted under another cell's identity.
    """
    from_index = {int(c): i for i, c in enumerate(from_ids)}
    rows = np.array([from_index.get(int(c), -1) for c in to_ids])
    found = rows >= 0
    safe = np.clip(rows, 0, max(len(from_ids) - 1, 0))
    out = OrderedDict()
    for name, entry in catalog.items():
        values = np.full(len(to_ids), np.nan)
        if len(entry.values):
            values = np.where(found, entry.values[safe], np.nan)
        out[name] = FeatureEntry(name, entry.group, values, entry.description)
    return out


def resolve_panels(catalog, panels=None):
    """Fit *panels* to whatever the catalogue actually holds.

    The defaults name temporal-STA and ACG components, and those are simply
    absent when a dataset has no STAs, no spike times, or a single degenerate
    block — which is common enough (a vision-only load, a run with no noise
    stimulus) that falling back has to be automatic rather than leaving six
    empty axes. Missing names are replaced with the first features that do
    exist, keeping the requested pairing where possible.
    """
    names = list(catalog.keys())
    if not names:
        return []
    wanted = list(panels or DEFAULT_PANELS)
    fallback = (names * 2)[: 2 * len(wanted)]
    out, cursor = [], 0
    for x, y in wanted:
        if x not in catalog:
            x = fallback[cursor % len(fallback)]
            cursor += 1
        if y not in catalog:
            y = fallback[cursor % len(fallback)]
            cursor += 1
        out.append((x, y))
    return out


def _pca_columns(matrix, n_pcs=N_PCS):
    """First *n_pcs* principal components of *matrix*, or None if not usable."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return None
    if not np.isfinite(matrix).all():
        matrix = np.nan_to_num(matrix)
    if np.allclose(matrix.std(axis=0), 0.0):
        return None  # every row identical: PCA would return zeros or NaN
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        logger.warning("scikit-learn unavailable; PCA features disabled")
        return None
    n = int(min(n_pcs, matrix.shape[0], matrix.shape[1]))
    if n < 1:
        return None
    try:
        return PCA(n_components=n).fit_transform(matrix)
    except Exception:
        logger.debug("PCA failed for a feature block", exc_info=True)
        return None


def _add_block_pcs(catalog, label, group, block, n_pcs=N_PCS):
    scores = _pca_columns(block, n_pcs)
    if scores is None:
        return
    n_zero = int(np.all(np.isclose(np.asarray(block, dtype=float), 0.0), axis=1).sum())
    note = (f" — {n_zero} cells have no data and share one point"
            if n_zero else "")
    for i in range(scores.shape[1]):
        catalog[f"{label} PC{i + 1}"] = FeatureEntry(
            f"{label} PC{i + 1}", group, scores[:, i],
            f"Principal component {i + 1} of the {label} block{note}")


def _column(df, name):
    try:
        if name in df.columns:
            return np.asarray(df[name].to_numpy(), dtype=float)
    except Exception:
        pass
    return None


def build_catalog(dm, cluster_ids, filter_config=None, include_contrast=False,
                  progress=None):
    """``(valid_ids, OrderedDict[name -> FeatureEntry])`` for *cluster_ids*.

    *progress* is an optional ``callable(message, percent)``. Safe to call from
    a worker thread — it only reads DataManager caches.
    """
    def report(msg, pct):
        if progress:
            try:
                progress(msg, pct)
            except Exception:
                pass

    filter_config = filter_config or {"min_sta_std": 0.0, "max_rf_area": 1e9}
    report("Extracting feature blocks...", 10)
    raw_blocks, valid_ids, _discarded = dm.get_raw_feature_blocks(
        cluster_ids, filter_config)
    if not valid_ids:
        return [], OrderedDict()

    catalog: OrderedDict = OrderedDict()

    report("Reducing blocks...", 30)
    for key, label, group in (
        ("temporal", "Temporal STA", "Shape (PCA)"),
        ("acg", "ACG", "Shape (PCA)"),
        ("grating", "Grating tuning", "Shape (PCA)"),
        ("chirp", "Chirp PSTH", "Shape (PCA)"),
    ):
        block = raw_blocks.get(key)
        if block is not None and np.asarray(block).size:
            _add_block_pcs(catalog, label, group, block)

    report("Collecting scalars...", 45)
    scalars = raw_blocks.get("scalars")
    if scalars is not None and len(scalars):
        for col, name, group, desc in (
            ("firing_rate", "Firing rate", "Spiking", "Mean rate over the recording (Hz)"),
            ("isi_violations", "ISI violations", "Spiking", "Refractory-period violation fraction"),
            ("time_to_peak", "Time to peak", "Receptive field", "STA time course peak (frames)"),
            ("rf_area", "RF area", "Receptive field", "Gaussian fit area"),
            ("rf_long_diameter", "RF diameter (long)", "Receptive field", "Long axis of the RF fit"),
            ("rf_short_diameter", "RF diameter (short)", "Receptive field", "Short axis of the RF fit"),
            ("ellipticity", "RF ellipticity", "Receptive field", "Long/short axis ratio"),
            ("grating_dsi", "Grating DSI", "Grating", "Direction selectivity index"),
            ("grating_osi", "Grating OSI", "Grating", "Orientation selectivity index"),
            ("grating_peak_rate_hz", "Grating peak rate", "Grating", "Peak evoked rate (Hz)"),
        ):
            values = _column(scalars, col)
            if values is not None and np.isfinite(values).any():
                catalog[name] = FeatureEntry(name, group, values, desc)

    # Quality columns live on cluster_df, keyed by cluster_id rather than by
    # row, so they have to be looked up rather than sliced.
    report("Collecting quality metrics...", 55)
    cdf = getattr(dm, "cluster_df", None)
    if cdf is not None and not cdf.empty and "cluster_id" in cdf.columns:
        index = {int(c): i for i, c in enumerate(cdf["cluster_id"].to_numpy())}
        rows = np.array([index.get(int(c), -1) for c in valid_ids])
        for col, name, group, desc in (
            ("chirp_qi", "Chirp QI", "Quality",
             "Response quality index, per light level, at a fixed bin width"),
            ("sta_snr", "STA SNR", "Quality",
             "STA peak-to-RMS: separates a real receptive field from noise"),
            ("n_spikes", "Spike count", "Spiking", "Total spikes in the recording"),
        ):
            if col not in cdf.columns:
                continue
            source = np.asarray(cdf[col].to_numpy(), dtype=float)
            values = np.where(rows >= 0, source[np.clip(rows, 0, len(source) - 1)],
                              np.nan)
            if np.isfinite(values).any():
                catalog[name] = FeatureEntry(name, group, values, desc)

    report("Deriving chirp response features...", 65)
    _add_chirp_features(dm, valid_ids, catalog)

    if include_contrast:
        report("Computing contrast responses (slow)...", 75)
        _add_contrast_features(dm, valid_ids, catalog, report)

    report("Catalogue ready.", 90)
    logger.debug("Feature catalogue: %d features for %d cells",
                 len(catalog), len(valid_ids))
    return valid_ids, catalog


def _add_chirp_features(dm, valid_ids, catalog):
    """ON/OFF index and frequency-following cutoff, per cell.

    Both come from the trial-averaged chirp PSTH, so they are cheap — no
    per-trial work — and they are the two chirp numbers that a PCA of the PSTH
    shape does not hand you directly.
    """
    if not getattr(dm, "chirp_available", False) or dm.chirp_data is None:
        return
    try:
        from . import chirp_calc
        timing = dm.chirp_timing()
    except Exception:
        logger.debug("chirp timing unavailable", exc_info=True)
        return

    n = len(valid_ids)
    onoff = np.full(n, np.nan)
    fcut = np.full(n, np.nan)
    peak_freq = np.full(n, np.nan)
    for i, cid in enumerate(valid_ids):
        entry = dm.get_chirp_data_for_cluster(int(cid))
        if entry is None:
            continue
        psth = np.asarray(entry["psth_mean"], dtype=float)
        bin_ms = float(entry["bin_size_ms"])
        try:
            onoff[i] = chirp_calc.on_off_index(psth, timing, bin_ms)[0]
            freqs, amps, snr = chirp_calc.frequency_tuning(psth, timing, bin_ms)
            if len(freqs):
                good = np.where(snr > 3.0)[0]
                if len(good):
                    fcut[i] = float(freqs[good[-1]])
                if np.isfinite(amps).any():
                    peak_freq[i] = float(freqs[int(np.nanargmax(amps))])
        except Exception:
            logger.debug("chirp features failed for cell %s", cid, exc_info=True)

    for values, name, desc in (
        (onoff, "Chirp ON/OFF index",
         "+1 pure increment-driven, -1 pure decrement-driven; pools both "
         "increments (0 s, 7 s) against both decrements (3 s, 4 s)"),
        (fcut, "Chirp freq cutoff",
         "Highest sweep frequency the cell still follows above noise (Hz)"),
        (peak_freq, "Chirp preferred freq",
         "Sweep frequency of the cell's strongest response (Hz)"),
    ):
        if np.isfinite(values).any():
            catalog[name] = FeatureEntry(name, "Chirp response", values, desc)


def _add_contrast_features(dm, valid_ids, catalog, report=None):
    """Per-light-level contrast gain, saturation and responsiveness."""
    if not getattr(dm, "contrast_available", False):
        return
    levels = dm.contrast_light_levels()
    if not levels:
        return
    try:
        from .contrast_calc import fit_naka_rushton
    except Exception:
        return

    n = len(valid_ids)
    n_responsive = np.zeros(n)
    for li, (run, label) in enumerate(levels):
        top = np.full(n, np.nan)
        c50 = np.full(n, np.nan)
        for i, cid in enumerate(valid_ids):
            entries = dm.get_contrast_data_for_cluster(int(cid), run=run)
            e = (entries or {}).get(run)
            if e is None:
                continue
            top[i] = e["f1"][-1]
            if e["responsive"]:
                n_responsive[i] += 1
                fit = fit_naka_rushton(e["contrasts"], e["f1"])
                # A c50 pinned at the highest contrast tested is an
                # extrapolation, not a measurement, so it is not reported.
                if fit is not None and not fit["c50_at_bound"]:
                    c50[i] = fit["c50"]
        if report:
            report(f"Contrast: {label} ({li + 1}/{len(levels)})",
                   75 + int(10 * (li + 1) / len(levels)))
        if np.isfinite(top).any():
            catalog[f"Contrast F1 @ {label}"] = FeatureEntry(
                f"Contrast F1 @ {label}", "Contrast response", top,
                f"Response at the highest contrast, {label} ({run})")
        if np.isfinite(c50).any():
            catalog[f"Contrast c50 @ {label}"] = FeatureEntry(
                f"Contrast c50 @ {label}", "Contrast response", c50,
                f"Half-saturating contrast at {label} ({run})")

    catalog["Contrast levels responsive"] = FeatureEntry(
        "Contrast levels responsive", "Contrast response", n_responsive,
        "How many light levels this cell responds at — the light level where "
        "a response appears is often more informative than its gain")
