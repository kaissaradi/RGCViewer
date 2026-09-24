"""
grating_calc.py

Single-cluster DSI/OSI + bar-width tuning from a raw grating npy
(spike_times_by_trial + trial_parameters). Math is the same as
combined_grating_analysis.py (f1_amplitude, vector_sum_index,
shuffle_pvalue).

Conditions are the (barWidth, temporalFrequency) pairs that actually ran.
A pair with MIN_DIRECTIONS_FOR_DSOS (4) or more unique orientations is
DSOS. Fewer orientations is SF. Do not assume a 12-dir crossed grid.

select_best_dsos_condition classifies each DSOS pair, then picks the
strongest significant response. GratingPanel, the population RF overlay,
and the preferred-orientation polar all use that pick.

pooled_direction_tuning_curve is a peak-weighted shape across every DSOS
pair. UMAP uses that curve (GRATING_PCA_COMPONENTS), not the DSI/OSI
scalars.

N_SHUFFLES is 200. Conditions with |DSI| and |OSI| both below
SHUFFLE_INDEX_FLOOR skip the permutation test. GratingBatchWorker fills
the cache in parallel with physics warm-up.

Conventions
-----------
* Spike times are ms from trial start. Each trial uses its OWN preTime and
  stimTime; the stimulus window is [preTime, preTime + stimTime).
* Bins are half-open. A partial last bin is dropped, never extended past
  the end of the stimulus.
* Directions are normalized to [0, 360), so 0° and 360° are one direction.
* Shuffle p-values are (k + 1) / (N + 1): never 0, smallest 1/(N+1).
* Responses are in spikes/s ("F1 amplitude" of the rate, or "Δ rate").
* GRATING_SCHEMA_VERSION tags each computed result. Cached rows from an
  older version are recomputed once.
"""

from collections import defaultdict

import numpy as np

PSTH_BIN_MS = 5.0
# 200 shuffles resolves p to 0.005, enough for a 0.05 gate. 1000 was the
# offline-script default and is why a 700-cell batch felt like it hung.
N_SHUFFLES = 200
RNG_SEED = 0
# Population slider floor is 0.10. Below that, p-values are never consulted,
# so the permutation test is skipped and p is stored as 1.0.
SHUFFLE_INDEX_FLOOR = 0.10
# Vector-sum DSI/OSI is defined for a circular set of directions. Four
# is the usual minimum (every 90°). One- or two-direction bar-width
# sweeps stay SF. Do not assume 12 directions or a crossed bw×TF grid —
# use however many orientations were actually presented at each (bw, tf).
MIN_DIRECTIONS_FOR_DSOS = 4

POOLED_CURVE_N_BINS = 12  # direction-bin count for pooled_direction_tuning_
# curve's output — a module constant (not just
# that function's default parameter) so callers
# building a zero-sentinel row for cells with no
# dsos data (see data_manager.py's
# get_raw_feature_blocks) always match the real
# function's output width without needing to
# duplicate the number or import the function
# just to call it with no data.

# --- "Best condition" / DS-OS classification -------------------------------
# Selection previously used max(|DSI|) alone, which is amplitude-blind and
# significance-blind: a condition with a handful of spikes that happened to
# land in one direction by chance can produce a higher |DSI| than a
# condition with a large, clearly time-locked, less-perfectly-concentrated
# response. Gate on both a minimum response amplitude AND a significance
# test before ranking by |DSI|/|OSI|, so "best" means "reliable," not just
# "numerically largest."
# Amplitude is used to RANK conditions (pick the run where the cell
# actually responded), not to veto membership. A 2 Hz floor was dropping
# sparse but significantly tuned DS/OS cells. Callers can still pass a
# positive min_response_hz to restore a floor.
MIN_RESPONSE_HZ = 0.0
ALPHA = 0.05  # shuffle-test significance threshold
DSI_THRESHOLD = 0.3  # DS classification cutoff, applied AFTER gating
OSI_THRESHOLD = 0.3  # OS classification cutoff, applied AFTER gating

# Bump when compute_grating_response's output changes meaning.
#   1 (untagged): timing from trial 0, p = k/N, raw orientation labels.
#   2: per-trial timing, p = (k+1)/(N+1), directions mod 360,
#      sd_response / n_trials / response_label added.
GRATING_SCHEMA_VERSION = 2
RESPONSE_METRICS = {"f1": "F1 amplitude", "delta": "Δ firing rate"}
RESPONSE_UNITS = "spikes/s"
PSTH_DISPLAY_BIN_MS = 50.0


def normalize_direction(deg):
    """Map a direction to [0, 360). 0° and 360° (or -90° and 270°) match."""
    d = round(float(deg) % 360.0, 6)
    return d % 360.0  # 359.9999999 rounds to 360.0; wrap it to 0.0


def stim_window(trial_params):
    """(start, end) of the stimulus in ms from trial start, for one trial."""
    pre = float(trial_params["preTime"])
    return pre, pre + float(trial_params["stimTime"])


def _n_full_bins(duration_ms, bin_ms):
    return int(np.floor(duration_ms / bin_ms + 1e-9))


def binned_counts(trials, onsets_ms, n_bins, bin_ms):
    """Spike counts per bin, one row per trial, aligned to each trial's onset.

    Bin k of trial i is [onset_i + k*bin_ms, onset_i + (k+1)*bin_ms).
    One bincount for all trials; this replaced a histogram per trial.
    """
    n = len(trials)
    counts = np.zeros((n, max(n_bins, 0)), dtype=np.float64)
    if n == 0 or n_bins <= 0:
        return counts
    lens = np.fromiter((len(t) for t in trials), dtype=np.int64, count=n)
    if lens.sum() == 0:
        return counts
    spikes = np.concatenate([np.asarray(t, dtype=np.float64).ravel() for t in trials])
    rows = np.repeat(np.arange(n), lens)
    onsets = np.repeat(np.asarray(onsets_ms, dtype=np.float64), lens)
    bins = np.floor((spikes - onsets) / bin_ms).astype(np.int64)
    ok = (bins >= 0) & (bins < n_bins)
    flat = np.bincount(rows[ok] * n_bins + bins[ok], minlength=n * n_bins)
    return flat.reshape(n, n_bins).astype(np.float64)


def f1_amplitudes(trials, windows, tfs_hz, bin_ms=PSTH_BIN_MS):
    """F1 amplitude (spikes/s) of every trial's rate at its own temporal frequency.

    ``windows`` holds one (t0, t1) per trial. Trials are grouped by window
    length so each group shares one binned matrix and one FFT.
    """
    out = np.full(len(trials), np.nan)
    if not len(trials):
        return out
    windows = np.asarray(windows, dtype=np.float64).reshape(-1, 2)
    tfs_hz = np.asarray(tfs_hz, dtype=np.float64)
    durations = windows[:, 1] - windows[:, 0]
    for dur in np.unique(durations):
        sel = np.flatnonzero(durations == dur)
        n_bins = _n_full_bins(dur, bin_ms)
        if n_bins < 4:
            continue
        counts = binned_counts([trials[i] for i in sel], windows[sel, 0], n_bins, bin_ms)
        rate = counts / (bin_ms / 1000.0)
        spec = np.fft.rfft(rate - rate.mean(axis=1, keepdims=True), axis=1)
        freqs = np.fft.rfftfreq(n_bins, d=bin_ms / 1000.0)
        f1_idx = np.argmin(np.abs(freqs[None, :] - tfs_hz[sel][:, None]), axis=1)
        out[sel] = 2.0 * np.abs(spec[np.arange(sel.size), f1_idx]) / n_bins
    return out


def f1_amplitude(spike_times_ms, window, tf_hz, bin_ms=PSTH_BIN_MS):
    """F1 amplitude (spikes/s) for one trial. See f1_amplitudes."""
    return float(f1_amplitudes([spike_times_ms], [window], [tf_hz], bin_ms)[0])


def window_rates(trials, windows):
    """Mean rate (spikes/s) of each trial in its own [t0, t1) window."""
    out = np.full(len(trials), np.nan)
    if not len(trials):
        return out
    windows = np.asarray(windows, dtype=np.float64).reshape(-1, 2)
    durations = windows[:, 1] - windows[:, 0]
    for dur in np.unique(durations):
        if dur <= 0:
            continue
        sel = np.flatnonzero(durations == dur)
        counts = binned_counts([trials[i] for i in sel], windows[sel, 0], 1, dur)
        out[sel] = counts[:, 0] / (dur / 1000.0)
    return out


def firing_rate_in_window(spike_times_ms, window):
    t0, t1 = window
    n = np.sum((spike_times_ms >= t0) & (spike_times_ms < t1))
    dur_sec = (t1 - t0) / 1000.0
    return n / dur_sec if dur_sec > 0 else np.nan


def vector_sum_index(thetas_deg, responses, harmonic=1):
    thetas_rad = np.deg2rad(thetas_deg) * harmonic
    vec = np.sum(responses * np.exp(1j * thetas_rad))
    denom = np.sum(responses)
    if denom <= 0 or not np.isfinite(denom):
        return np.nan, np.nan
    index = np.abs(vec) / denom
    pref_angle = np.rad2deg(np.angle(vec)) / harmonic
    pref_angle = pref_angle % (360.0 / harmonic)
    return index, pref_angle


def _permutation_p(null_indices, observed_index):
    """(k + 1) / (N + 1) over the finite nulls. Never 0 for a finite N."""
    null_indices = np.asarray(null_indices, dtype=float)
    valid = np.isfinite(null_indices)
    n = int(np.count_nonzero(valid))
    if n == 0:
        return np.nan
    k = int(np.count_nonzero(null_indices[valid] >= observed_index))
    return (k + 1.0) / (n + 1.0)


def shuffle_pvalue(directions, trial_responses_by_dir, harmonic, n_shuffles, rng):
    """Permutation test: shuffle trial responses across directions.

    Same shuffles as combined_grating_analysis.py. The p-value is
    (k + 1) / (N + 1), not k / N: the observed labelling is itself one of
    the permutations, and k / N reports p = 0 for any cell that beats all
    200 shuffles.
    """
    directions = sorted(directions)
    sizes = [len(trial_responses_by_dir[d]) for d in directions]
    n_per_dir = sizes[0]
    if any(s != n_per_dir for s in sizes):
        all_resp = np.concatenate([trial_responses_by_dir[d] for d in directions])
        boundaries = np.cumsum([0] + sizes)
        observed_means = np.array(
            [np.nanmean(trial_responses_by_dir[d]) for d in directions]
        )
        observed_index, _ = vector_sum_index(
            np.array(directions), observed_means, harmonic
        )
        if not np.isfinite(observed_index):
            return np.nan
        null_indices = np.empty(n_shuffles)
        for s in range(n_shuffles):
            shuffled = rng.permutation(all_resp)
            means = np.array(
                [
                    np.nanmean(shuffled[boundaries[i] : boundaries[i + 1]])
                    for i in range(len(directions))
                ]
            )
            null_indices[s], _ = vector_sum_index(np.array(directions), means, harmonic)
        return _permutation_p(null_indices, observed_index)

    n_dir = len(directions)
    all_resp = np.concatenate([trial_responses_by_dir[d] for d in directions])
    total_n = all_resp.shape[0]

    observed_means = np.array([np.mean(trial_responses_by_dir[d]) for d in directions])
    observed_index, _ = vector_sum_index(np.array(directions), observed_means, harmonic)
    if not np.isfinite(observed_index):
        return np.nan

    theta = np.deg2rad(np.array(directions)) * harmonic
    unit_vecs = np.exp(1j * theta)

    shuffle_idx = np.argsort(rng.random((n_shuffles, total_n)), axis=1)
    shuffled_resp = all_resp[shuffle_idx]
    shuffled_resp = shuffled_resp.reshape(n_shuffles, n_dir, n_per_dir)
    means = shuffled_resp.mean(axis=2)

    vec = means @ unit_vecs
    denom = means.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        null_indices = np.abs(vec) / denom
    return _permutation_p(null_indices, observed_index)


def direction_psth(spike_times_by_direction_ms, window, bin_ms=PSTH_DISPLAY_BIN_MS,
                   onsets_ms=None):
    """
    Mean PSTH (spikes/s) across trials for one direction over the stimulus.

    ``window`` is (t0, t1) and fixes the duration. ``onsets_ms`` gives each
    trial's own stimulus onset; by default every trial starts at t0.
    Returns (bin centres in ms from onset, rate).
    """
    t0, t1 = window
    n_bins = _n_full_bins(t1 - t0, bin_ms)
    if n_bins <= 0:
        return np.array([]), np.array([])
    trials = list(spike_times_by_direction_ms)
    if onsets_ms is None:
        onsets_ms = [t0] * len(trials)
    counts = binned_counts(trials, onsets_ms, n_bins, bin_ms).sum(axis=0)
    n_trials = max(len(trials), 1)
    rate = (counts / n_trials) / (bin_ms / 1000.0)
    t = np.arange(n_bins) * bin_ms + bin_ms / 2.0
    return t, rate


def trial_rasters(spike_times_by_trial_for_cell, trial_parameters, condition):
    """Per-direction spike rasters for one (barWidth, temporalFrequency).

    Returns ``{direction_deg: [spike times in s from stimulus onset, ...]}``
    with one array per trial, in presentation order, or ``{}`` if the
    condition did not run. Directions are normalized to [0, 360).
    ``pre_s`` / ``stim_s`` / ``tail_s`` in the returned ``"_timing"`` entry
    give the shared pre-stimulus, stimulus and tail durations (the shortest
    across trials; tail is 0 when the protocol does not record it).
    """
    bw, tf = (float(condition[0]), float(condition[1]))
    by_dir = defaultdict(list)
    pres, stims, tails = [], [], []
    for i, t in enumerate(trial_parameters):
        if float(t["barWidth"]) != bw or float(t["temporalFrequency"]) != tf:
            continue
        pre = float(t["preTime"])
        pres.append(pre)
        stims.append(float(t["stimTime"]))
        tails.append(float(t.get("tailTime", 0.0) or 0.0))
        sp = np.asarray(spike_times_by_trial_for_cell[i], dtype=np.float64)
        by_dir[normalize_direction(t["orientation"])].append((sp - pre) / 1000.0)
    if not by_dir:
        return {}
    out = dict(sorted(by_dir.items()))
    out["_timing"] = {"pre_s": min(pres) / 1000.0, "stim_s": min(stims) / 1000.0,
                      "tail_s": min(tails) / 1000.0}
    return out


def group_grating_conditions(
    trial_parameters, min_directions_for_dsos=MIN_DIRECTIONS_FOR_DSOS
):
    """Partition trials by the (barWidth, temporalFrequency) pairs that ran.

    No assumed grid: a (bw, tf) with enough unique orientations is DSOS
    (DSI/OSI + polar); fewer orientations is SF (bar-width tuning). Four
    directions is enough for a vector-sum index; a 1-dir bar-width sweep
    stays SF.

    Returns a list of dicts with ``key``, ``condition_type``, ``directions``,
    ``idx_by_dir``.
    """
    if not trial_parameters:
        return []

    conditions = sorted(
        set(
            (float(t["barWidth"]), float(t["temporalFrequency"]))
            for t in trial_parameters
        )
    )
    groups = []
    for bw, tf in conditions:
        idx_by_dir = defaultdict(list)
        for i, t in enumerate(trial_parameters):
            if float(t["barWidth"]) == bw and float(t["temporalFrequency"]) == tf:
                idx_by_dir[normalize_direction(t["orientation"])].append(i)
        directions = sorted(idx_by_dir)
        typ = "dsos" if len(directions) >= min_directions_for_dsos else "sf"
        groups.append(
            {
                "key": (bw, tf),
                "condition_type": typ,
                "directions": directions,
                "idx_by_dir": idx_by_dir,
            }
        )
    return groups


def grating_entry_needs_recompute(
    entry, min_directions_for_dsos=MIN_DIRECTIONS_FOR_DSOS
):
    """True when a persisted result must be recomputed.

    Two reasons: its DSOS/SF tags don't match the direction counts, or it
    was computed by an older GRATING_SCHEMA_VERSION. Use this on rows from
    compute_grating_response only; analyzed files carry no version.

    Dummy cache rows without per-condition tuples (used by persistence
    tests) are left alone.
    """
    if not isinstance(entry, dict):
        return False
    conds = [
        v
        for k, v in entry.items()
        if isinstance(k, tuple) and isinstance(v, dict)
    ]
    if not conds:
        return False
    if entry.get("schema_version", 1) < GRATING_SCHEMA_VERSION:
        return True
    for v in conds:
        dirs = v.get("directions_deg")
        n = 0 if dirs is None else len(np.asarray(dirs))
        should_be_dsos = n >= min_directions_for_dsos
        is_dsos = v.get("condition_type") == "dsos"
        if should_be_dsos != is_dsos:
            return True
    return False


def format_condition_label(cond, entry=None):
    """Legend / stats text: the (bw, tf) that ran, plus how many directions."""
    bw, tf = cond
    label = f"bw={bw:g} tf={tf:g}Hz"
    dirs = (entry or {}).get("directions_deg")
    if dirs is not None:
        n = len(np.asarray(dirs))
        if n:
            label += f" ({n:g} dir)"
    return label


def compute_grating_response(
    cluster_id,
    spike_times_by_trial,
    trial_parameters,
    n_shuffles=N_SHUFFLES,
    min_directions_for_dsos=MIN_DIRECTIONS_FOR_DSOS,
    response_metric="f1",
    rng_seed=RNG_SEED,
):
    """
    Compute DSI/OSI (or bar-width tuning point) for ONE cluster, across
    every (barWidth, temporalFrequency) condition present in the raw file.

    Uses the (bw, tf, orientation) combinations that were actually run —
    see :func:`group_grating_conditions`. F1 is taken at each trial's own
    temporal frequency.

    Returns the same per-condition dict shape combined_grating_analysis.py
    produces for results[cluster_id], so panel rendering code doesn't need
    to know whether data came from disk or was computed live:

        {
            (bw, tf): {
                'condition_type': 'dsos' | 'sf',
                'directions_deg': ndarray,
                'mean_response': ndarray,
                'sem_response': ndarray,
                'DSI', 'preferred_direction_deg', 'DSI_pvalue',
                'OSI', 'preferred_orientation_deg', 'OSI_pvalue',
                # 'sf' conditions additionally/instead carry:
                'bw_tuning_point', 'bw_tuning_point_sem',
            },
            ...
            'sf_bar_widths': ndarray,      # only if any 'sf' conditions exist
            'sf_tuning_curve': ndarray,    # only if any 'sf' conditions exist
        }

    Returns None if cluster_id has no trials in spike_times_by_trial.
    """
    if response_metric not in RESPONSE_METRICS:
        raise ValueError(
            f"response_metric must be one of {sorted(RESPONSE_METRICS)}, "
            f"got {response_metric!r}"
        )
    if cluster_id not in spike_times_by_trial:
        return None

    rng = np.random.default_rng(rng_seed)
    trials = spike_times_by_trial[cluster_id]
    # Every trial's own stimulus window; a protocol can mix durations.
    windows = [stim_window(t) for t in trial_parameters]

    groups = group_grating_conditions(
        trial_parameters, min_directions_for_dsos=min_directions_for_dsos
    )
    condition_type = {g["key"]: g["condition_type"] for g in groups}

    result = {"schema_version": GRATING_SCHEMA_VERSION}
    for group in groups:
        bw, tf = group["key"]
        local_dirs = group["directions"]
        typ = group["condition_type"]
        idx_by_dir = group["idx_by_dir"]
        group_idxs = [i for d in local_dirs for i in idx_by_dir[d]]

        # One vectorized pass over every trial of this condition.
        g_trials = [trials[i] for i in group_idxs]
        g_windows = [windows[i] for i in group_idxs]
        if response_metric == "f1":
            g_tfs = [float(trial_parameters[i]["temporalFrequency"]) for i in group_idxs]
            g_resp = f1_amplitudes(g_trials, g_windows, g_tfs)
        else:
            # Δ rate can be negative; vector_sum_index is only meaningful
            # for non-negative responses, so treat this metric's DSI/OSI
            # as descriptive.
            evoked = window_rates(g_trials, g_windows)
            baseline = window_rates(g_trials, [(0.0, w[0]) for w in g_windows])
            g_resp = evoked - baseline
        g_rates = window_rates(g_trials, g_windows)

        trial_resp_by_dir = {}
        pos = 0
        for direction in local_dirs:
            n = len(idx_by_dir[direction])
            trial_resp_by_dir[direction] = g_resp[pos:pos + n]
            pos += n

        n_trials = np.array([np.count_nonzero(np.isfinite(trial_resp_by_dir[d]))
                             for d in local_dirs])
        with np.errstate(invalid="ignore"):
            mean_resp = np.array([np.nanmean(trial_resp_by_dir[d])
                                  if n_trials[j] else np.nan
                                  for j, d in enumerate(local_dirs)])
            sd_resp = np.array([np.nanstd(trial_resp_by_dir[d], ddof=1)
                                if n_trials[j] > 1 else np.nan
                                for j, d in enumerate(local_dirs)])
            sem_resp = sd_resp / np.sqrt(np.maximum(n_trials, 1))

        entry = {
            "condition_type": typ,
            "directions_deg": np.array(local_dirs),
            "mean_response": mean_resp,
            "sd_response": sd_resp,
            "sem_response": sem_resp,
            "n_trials": n_trials,
            "response_label": RESPONSE_METRICS[response_metric],
            "response_units": RESPONSE_UNITS,
        }

        if typ == "dsos":
            # Per-direction firing-rate PSTHs, aligned to each trial's own
            # onset and cut to the shortest stimulus in the condition. One
            # binned matrix for the condition, then a mean per direction.
            bin_ms = PSTH_DISPLAY_BIN_MS
            dur = min(w[1] - w[0] for w in g_windows) if g_windows else 0.0
            n_bins = _n_full_bins(dur, bin_ms)
            counts = binned_counts(g_trials, [w[0] for w in g_windows], n_bins, bin_ms)
            psth_by_dir = {}
            pos = 0
            for direction in local_dirs:
                n = len(idx_by_dir[direction])
                psth_by_dir[direction] = (
                    counts[pos:pos + n].sum(axis=0) / max(n, 1) / (bin_ms / 1000.0)
                )
                pos += n
            t = np.arange(n_bins) * bin_ms + bin_ms / 2.0
            entry["psth_time_s"] = (t / 1000.0) if local_dirs else np.array([])
            entry["psth_by_direction"] = psth_by_dir

            dsi, pref_dir = vector_sum_index(
                np.array(local_dirs), mean_resp, harmonic=1
            )
            osi, pref_ori = vector_sum_index(
                np.array(local_dirs), mean_resp, harmonic=2
            )
            # Most cells are untuned. The shuffle is the expensive step and
            # is only consumed when |DSI| or |OSI| could pass the slider.
            need_shuffle = n_shuffles > 0 and (
                (np.isfinite(dsi) and abs(dsi) >= SHUFFLE_INDEX_FLOOR)
                or (np.isfinite(osi) and abs(osi) >= SHUFFLE_INDEX_FLOOR)
            )
            if need_shuffle:
                dsi_p = shuffle_pvalue(
                    local_dirs,
                    trial_resp_by_dir,
                    harmonic=1,
                    n_shuffles=n_shuffles,
                    rng=rng,
                )
                osi_p = shuffle_pvalue(
                    local_dirs,
                    trial_resp_by_dir,
                    harmonic=2,
                    n_shuffles=n_shuffles,
                    rng=rng,
                )
            else:
                dsi_p = 1.0
                osi_p = 1.0

            # peak_rate_hz: real evoked firing rate (Hz), independent of
            # response_metric ('f1' amplitude / 'delta' aren't in Hz units
            # and aren't comparable across conditions run with different
            # tf). This is the amplitude-floor gate for best-condition
            # selection — see select_best_dsos_condition — so a condition
            # with a handful of noisy spikes can't out-rank a condition
            # with a real, strong response just because its DSI happens
            # to be numerically higher.
            peak_rate_hz = np.nanmax(g_rates) if g_rates.size else np.nan

            entry.update(
                {
                    "DSI": dsi,
                    "preferred_direction_deg": pref_dir,
                    "DSI_pvalue": dsi_p,
                    "OSI": osi,
                    "preferred_orientation_deg": pref_ori,
                    "OSI_pvalue": osi_p,
                    "peak_rate_hz": (
                        float(peak_rate_hz) if np.isfinite(peak_rate_hz) else np.nan
                    ),
                }
            )
        else:
            entry.update(
                {
                    "DSI": np.nan,
                    "preferred_direction_deg": np.nan,
                    "DSI_pvalue": np.nan,
                    "OSI": np.nan,
                    "preferred_orientation_deg": np.nan,
                    "OSI_pvalue": np.nan,
                    "bw_tuning_point": np.nanmean(mean_resp),
                    "bw_tuning_point_sem": (
                        np.nanstd(mean_resp, ddof=1) / np.sqrt(len(local_dirs))
                        if len(local_dirs) > 1
                        else np.nan
                    ),
                }
            )

        result[(bw, tf)] = entry

    sf_bar_widths = sorted(
        set(bw for (bw, tf), typ in condition_type.items() if typ == "sf")
    )
    if sf_bar_widths:
        curve = np.full(len(sf_bar_widths), np.nan)
        for j, bw in enumerate(sf_bar_widths):
            vals = [
                result[(bw2, tf)]["bw_tuning_point"]
                for (bw2, tf), typ in condition_type.items()
                if bw2 == bw and typ == "sf"
            ]
            if vals:
                curve[j] = np.nanmean(vals)
        result["sf_bar_widths"] = np.array(sf_bar_widths)
        result["sf_tuning_curve"] = curve

    return result


def condition_amplitude(entry):
    """How strongly this (bw, tf) actually drove the cell.

    Prefer peak of the trial-averaged tuning curve (F1 / mean_response) —
    that is the same metric DSI/OSI were computed from. Fall back to
    peak_rate_hz when the curve is missing (legacy analyzed files).
    """
    resp = entry.get("mean_response")
    if resp is not None:
        arr = np.asarray(resp, dtype=float)
        if arr.size:
            peak = np.nanmax(arr)
            if np.isfinite(peak) and peak > 0:
                return float(peak)
    rate = entry.get("peak_rate_hz", np.nan)
    if np.isfinite(rate) and rate > 0:
        return float(rate)
    return 0.0


def _pvalue_passes(entry, key, alpha):
    """Shuffle p < alpha. Missing p (legacy files) does not veto."""
    pval = entry.get(key, np.nan)
    if not np.isfinite(pval):
        return True
    return pval < alpha


def _none_dsos_selection():
    return {
        "condition": None,
        "classification": "none",
        "DSI": np.nan,
        "OSI": np.nan,
        "preferred_direction_deg": np.nan,
        "preferred_orientation_deg": np.nan,
        "DSI_pvalue": np.nan,
        "OSI_pvalue": np.nan,
        "peak_rate_hz": np.nan,
    }


def _selection_from_entry(cond, classification, entry):
    return {
        "condition": cond,
        "classification": classification,
        "DSI": entry.get("DSI", np.nan),
        "OSI": entry.get("OSI", np.nan),
        "preferred_direction_deg": entry.get("preferred_direction_deg", np.nan),
        "preferred_orientation_deg": entry.get("preferred_orientation_deg", np.nan),
        "DSI_pvalue": entry.get("DSI_pvalue", np.nan),
        "OSI_pvalue": entry.get("OSI_pvalue", np.nan),
        "peak_rate_hz": entry.get("peak_rate_hz", np.nan),
    }


def select_best_dsos_condition(
    data,
    min_response_hz=MIN_RESPONSE_HZ,
    alpha=ALPHA,
    dsi_threshold=DSI_THRESHOLD,
    osi_threshold=OSI_THRESHOLD,
):
    """
    Picks the single 'best' (barWidth, temporalFrequency) condition and a
    DS/OS classification for one cluster.

    `data` is the per-cluster dict returned by compute_grating_response
    (or the equivalent pre-analyzed-file entry) — i.e. data[cluster_id].

    Per (bw, tf) that was actually run:
      1. GATE: shuffle p-value < alpha (missing p does not veto). A
         positive min_response_hz, if the caller sets one, is an extra
         amplitude floor; the default is 0 so sparse cells are not dropped.
      2. CLASSIFY that condition: DS if |DSI| > dsi_threshold (DSI-first
         at the same condition, because a single lobe lifts both
         harmonics); else OS if |OSI| > osi_threshold.
      3. RANK across conditions: pick the classified condition with the
         strongest response (peak of mean_response). A noisy high-DSI
         run at 2 Hz must not beat a real DS/OS run at 20 Hz, and a weak
         DS at one bar width must not hide a strong OS at another.

    Returns a dict:
        {
            'condition': (bw, tf) or None,
            'classification': 'DS' | 'OS' | 'none',
            'DSI', 'OSI', 'preferred_direction_deg', 'preferred_orientation_deg',
            'DSI_pvalue', 'OSI_pvalue', 'peak_rate_hz',
        }
    or None if the cluster has no 'dsos' conditions at all (as opposed to
    having conditions that just didn't pass the gate — that case returns
    classification='none' with condition=None, which callers should render
    as an explicit "not significantly tuned" state, not silently omit).
    """
    dsos_conditions = [
        c
        for c in data
        if isinstance(c, tuple) and data[c].get("condition_type") == "dsos"
    ]
    if not dsos_conditions:
        return None

    classified = []
    for cond in dsos_conditions:
        entry = data[cond]
        amp = condition_amplitude(entry)
        if min_response_hz > 0 and amp <= min_response_hz:
            continue
        dsi = entry.get("DSI", np.nan)
        osi = entry.get("OSI", np.nan)
        is_ds = (
            _pvalue_passes(entry, "DSI_pvalue", alpha)
            and np.isfinite(dsi)
            and abs(dsi) > dsi_threshold
        )
        is_os = (
            _pvalue_passes(entry, "OSI_pvalue", alpha)
            and np.isfinite(osi)
            and abs(osi) > osi_threshold
        )
        if is_ds:
            classified.append((cond, "DS", amp, abs(dsi)))
        elif is_os:
            classified.append((cond, "OS", amp, abs(osi)))

    if not classified:
        return _none_dsos_selection()

    # Strongest response wins. Tie-break: DS before OS, then larger index.
    best_cond, best_cls, _amp, _idx = max(
        classified, key=lambda item: (item[2], 1 if item[1] == "DS" else 0, item[3])
    )
    return _selection_from_entry(best_cond, best_cls, data[best_cond])


def pooled_direction_tuning_curve(data, n_bins=POOLED_CURVE_N_BINS):
    """
    Peak-weighted, shape-normalized direction tuning curve for one cluster,
    pooled across every 'dsos' condition present (not just the single best
    condition select_best_dsos_condition would pick).

    Motivation: DSI/OSI are each a single scalar summarizing an entire
    tuning curve into "how concentrated is the response around one
    harmonic." Two very differently-shaped curves (a narrow single peak vs.
    a broad lopsided hump) can produce the same DSI — that shape
    information is lost before it ever reaches the embedding. This
    function instead returns the curve SHAPE itself (interpolated onto a
    fixed n_bins-point grid), meant to be PCA'd (see
    GRATING_PCA_COMPONENTS in constants.py) the same way temporal STA and
    ACG shapes already are, rather than collapsed to DSI/OSI scalars.

    Pooling method (peak-weighted average of per-condition normalized
    curves):
      1. For each dsos condition, normalize mean_response by its own peak
         — isolates SHAPE from amplitude on a per-condition basis, since a
         cell can be genuinely tuned at one (barWidth, TF) and untuned at
         another (real spatiotemporal tuning, not noise), so a flat
         unweighted average across conditions would blur "untuned here"
         into "tuned there" and produce a muddier shape than either alone.
      2. Interpolate onto a common n_bins-point direction grid (handles any
         condition-to-condition variation in which exact directions were
         tested).
      3. Average the normalized curves across conditions, weighted by each
         condition's own peak response — a condition the cell barely
         responds to contributes little to the pooled shape, a condition
         with a strong response dominates. This is deliberately NOT
         additionally gated by DSI_pvalue/significance the way
         select_best_dsos_condition's WINNER is — an untuned condition's
         low peak already suppresses its own contribution via the
         weighting itself, so a separate significance filter here would
         just throw away real partial signal from conditions that didn't
         individually clear p<0.05 but still meaningfully shape the pooled
         curve.

    Returns an (n_bins,) array (direction bins spanning 0-360°, uniformly
    spaced), or None if this cluster has no dsos conditions, or all of them
    have zero/non-finite peak response (nothing to weight by).
    """
    dsos_conditions = [
        c
        for c in data
        if isinstance(c, tuple) and data[c].get("condition_type") == "dsos"
    ]
    if not dsos_conditions:
        return None

    target_angles = np.linspace(0, 360, n_bins, endpoint=False)

    weighted_sum = np.zeros(n_bins, dtype=np.float64)
    total_weight = 0.0

    for cond in dsos_conditions:
        entry = data[cond]
        dirs = np.asarray(entry.get("directions_deg", []), dtype=float)
        resp = np.asarray(entry.get("mean_response", []), dtype=float)
        if dirs.size < 2 or resp.size != dirs.size:
            continue

        peak = np.nanmax(resp) if resp.size else np.nan
        if not np.isfinite(peak) or peak <= 0:
            continue  # nothing to weight by, and normalizing would divide by ~0

        normalized = resp / peak

        # Interpolate onto the common grid. Directions are circular (0deg
        # and 360deg are the same point), so pad BOTH ends before
        # np.interp: prepend the last sample shifted -360, append the first
        # shifted +360. Padding only the top (append first+360) is not
        # enough — if the directions don't start at 0 (e.g. 30,75,...,345),
        # target angles below the lowest direction (0..30 here) would fall
        # off the bottom of the data and np.interp would CLAMP them to the
        # endpoint value instead of wrapping around from 345. Both-end
        # padding makes the wrap correct at both edges. (np.interp also
        # requires strictly increasing x, which this preserves.)
        order = np.argsort(dirs)
        dirs_sorted = dirs[order]
        normalized_sorted = normalized[order]
        dirs_wrapped = np.concatenate(
            [dirs_sorted[-1:] - 360.0, dirs_sorted, dirs_sorted[:1] + 360.0]
        )
        normalized_wrapped = np.concatenate(
            [normalized_sorted[-1:], normalized_sorted, normalized_sorted[:1]]
        )
        interp_curve = np.interp(target_angles, dirs_wrapped, normalized_wrapped)

        weighted_sum += interp_curve * peak
        total_weight += peak

    if total_weight <= 0:
        return None

    return weighted_sum / total_weight
