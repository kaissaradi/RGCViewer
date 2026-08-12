"""Contrast-response maths for ``*_contrastResponse_unified.npy``.

The file holds a drifting sinewave grating shown at a series of contrasts:
``spike_times_by_trial[vision_id][trial]`` in ms relative to trial start, plus a
``trial_parameters`` list carrying contrast, temporal frequency, bar width and
the pre/stim/tail times. Schema is identical to the raw grating file, so the F1
machinery here mirrors ``grating_calc.f1_amplitude`` deliberately rather than
inventing a second convention.

**Trials from several runs are concatenated with no per-trial source tag.** A
file whose ``source_files`` lists four runs holds 4 x 80 trials in source order
and nothing marks the boundaries. ``split_trials_by_run`` applies the only
available rule — equal blocks in ``source_files`` order — which was verified on
20260715A by checking that response strength tracks light level across the
blocks (1.0%, 2.8%, 29.1%, 26.5% responsive for NDF5, NDF4, NDF3, NDF2) and
matches what the independently sorted per-run files give. It is still an
inference, so ``split_trials_by_run`` refuses rather than guesses when the trial
count is not divisible by the run count.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "PSTH_BIN_MS", "split_trials_by_run", "cycle_metrics", "contrast_response",
    "naka_rushton", "fit_naka_rushton",
]

PSTH_BIN_MS = 25.0

# A cell counts as responding at a light level when its F1 at the highest
# contrast clears the blank-contrast F1 by this much. The blank trials are a
# measured noise floor, which is why the test is multiplicative on them plus a
# small absolute floor for cells that are silent in both.
RESPONSE_RATIO = 2.0
RESPONSE_FLOOR_HZ = 1.0


def split_trials_by_run(n_trials, source_files):
    """``{run: (start, stop)}`` trial index ranges, or None if not inferable.

    See the module docstring: the file carries no per-trial provenance, so this
    is the equal-blocks-in-order rule and nothing better is available.
    """
    runs = list(source_files or [])
    if not runs:
        return None
    if len(runs) == 1:
        return {runs[0]: (0, int(n_trials))}
    if n_trials % len(runs) != 0:
        logger.warning(
            "Contrast file has %d trials across %d source runs, which does not "
            "divide evenly — cannot attribute trials to runs.", n_trials, len(runs)
        )
        return None
    per = n_trials // len(runs)
    return {run: (i * per, (i + 1) * per) for i, run in enumerate(runs)}


def cycle_metrics(spike_times_ms, window, tf_hz, bin_ms=PSTH_BIN_MS):
    """``(f0, f1, f2)`` in Hz for one trial over *window* ``(t0, t1)`` ms.

    f0 is the mean rate, f1 the amplitude at the drift frequency (the linear
    response) and f2 at twice it (the frequency-doubled response that separates
    Y-like from X-like cells).
    """
    t0, t1 = window
    edges = np.arange(t0, t1 + bin_ms, bin_ms)
    counts, _ = np.histogram(np.asarray(spike_times_ms, dtype=float), bins=edges)
    rate = counts / (bin_ms / 1000.0)
    n = len(rate)
    if n < 4:
        return np.nan, np.nan, np.nan
    f0 = float(rate.mean())
    spec = np.fft.rfft(rate - f0)
    freqs = np.fft.rfftfreq(n, d=bin_ms / 1000.0)
    k1 = int(np.argmin(np.abs(freqs - tf_hz)))
    k2 = int(np.argmin(np.abs(freqs - 2.0 * tf_hz)))
    return f0, float(2 * np.abs(spec[k1]) / n), float(2 * np.abs(spec[k2]) / n)


def contrast_response(trials, trial_parameters, trial_range=None, bin_ms=PSTH_BIN_MS):
    """Contrast-response function for one cell over one run's trials.

    ``trials`` is that cell's per-trial spike-time list; ``trial_range`` an
    optional ``(start, stop)`` restricting to one source run.

    Returns a dict with ``contrasts`` and, per contrast, the mean and SEM of
    F0/F1/F2 across repeats, plus ``responsive`` and the repeat count. Returns
    None when the range holds no usable trials.
    """
    if trials is None or not len(trial_parameters):
        return None
    start, stop = trial_range if trial_range else (0, len(trial_parameters))
    idx = [i for i in range(start, min(stop, len(trial_parameters), len(trials)))]
    if not idx:
        return None

    p0 = trial_parameters[idx[0]]
    pre = float(p0.get("preTime", 0.0))
    stim = float(p0.get("stimTime", 0.0))
    tf = float(p0.get("temporalFrequency", 1.0) or 1.0)
    window = (pre, pre + stim)

    by_c = {}
    for i in idx:
        c = float(trial_parameters[i].get("contrast", np.nan))
        by_c.setdefault(c, []).append(i)
    contrasts = sorted(by_c)
    if not contrasts:
        return None

    n_c = len(contrasts)
    f0 = np.full(n_c, np.nan); f1 = np.full(n_c, np.nan); f2 = np.full(n_c, np.nan)
    f1_sem = np.full(n_c, np.nan); f0_sem = np.full(n_c, np.nan)
    n_reps = np.zeros(n_c, dtype=int)
    for j, c in enumerate(contrasts):
        vals = np.array([cycle_metrics(trials[i], window, tf, bin_ms) for i in by_c[c]])
        vals = vals[np.all(np.isfinite(vals), axis=1)] if vals.size else vals
        if not len(vals):
            continue
        n_reps[j] = len(vals)
        f0[j], f1[j], f2[j] = vals.mean(axis=0)
        if len(vals) > 1:
            f0_sem[j] = vals[:, 0].std(ddof=1) / np.sqrt(len(vals))
            f1_sem[j] = vals[:, 1].std(ddof=1) / np.sqrt(len(vals))

    blank = f1[0] if contrasts[0] == 0.0 else np.nan
    top = f1[-1]
    if np.isfinite(blank):
        responsive = bool(top > RESPONSE_RATIO * blank + RESPONSE_FLOOR_HZ)
    else:
        # No blank condition in this run — fall back to an absolute floor.
        responsive = bool(np.isfinite(top) and top > RESPONSE_FLOOR_HZ)

    return {
        "contrasts": np.array(contrasts, dtype=float),
        "f0": f0, "f1": f1, "f2": f2,
        "f0_sem": f0_sem, "f1_sem": f1_sem,
        "n_repeats": n_reps,
        "responsive": responsive,
        "blank_f1": float(blank) if np.isfinite(blank) else np.nan,
        "temporal_frequency": tf,
        "bar_width": p0.get("barWidth"),
        "n_trials": len(idx),
    }


def naka_rushton(c, r_max, c50, n, offset):
    """``r_max * c^n / (c^n + c50^n) + offset`` — the standard contrast-response
    saturation model."""
    c = np.asarray(c, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.power(c, n)
        den = num + np.power(c50, n)
        out = np.where(den > 0, r_max * num / den, 0.0) + offset
    return np.nan_to_num(out)


def fit_naka_rushton(contrasts, response):
    """Fit the saturation model; returns a dict or None if it cannot be fit.

    ``c50`` — the contrast at which the cell reaches half its maximum — is the
    parameter worth comparing across light levels, so it is bounded to the
    tested contrast range: a c50 extrapolated past the highest contrast shown
    is not measured, it is invented.
    """
    c = np.asarray(contrasts, dtype=float)
    r = np.asarray(response, dtype=float)
    good = np.isfinite(c) & np.isfinite(r)
    c, r = c[good], r[good]
    if c.size < 4 or np.ptp(r) <= 0:
        return None
    try:
        from scipy.optimize import curve_fit
        c_max = float(c.max()) or 1.0
        p0 = [float(np.ptp(r)), max(float(c_max) / 2, 1e-3), 2.0, float(r.min())]
        popt, _ = curve_fit(
            naka_rushton, c, r, p0=p0, maxfev=20000,
            bounds=([0.0, 1e-3, 0.5, -np.inf], [np.inf, c_max, 10.0, np.inf]),
        )
    except Exception:
        return None
    fit = naka_rushton(c, *popt)
    ss_res = float(((r - fit) ** 2).sum())
    ss_tot = float(((r - r.mean()) ** 2).sum())
    return {
        "r_max": float(popt[0]), "c50": float(popt[1]),
        "n": float(popt[2]), "offset": float(popt[3]),
        "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "c50_at_bound": bool(popt[1] >= float(c.max()) * 0.999),
    }
