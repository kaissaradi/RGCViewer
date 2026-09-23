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


def f1_amplitude(spike_times_ms, window, tf_hz, bin_ms=PSTH_BIN_MS):
    t0, t1 = window
    edges = np.arange(t0, t1 + bin_ms, bin_ms)
    counts, _ = np.histogram(spike_times_ms, bins=edges)
    rate = counts / (bin_ms / 1000.0)
    n = len(rate)
    if n < 4:
        return np.nan
    fft_vals = np.fft.rfft(rate - rate.mean())
    freqs = np.fft.rfftfreq(n, d=bin_ms / 1000.0)
    f1_idx = int(np.argmin(np.abs(freqs - tf_hz)))
    return 2.0 * np.abs(fft_vals[f1_idx]) / n


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


def shuffle_pvalue(directions, trial_responses_by_dir, harmonic, n_shuffles, rng):
    """Permutation test — identical to combined_grating_analysis.py."""
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
        return np.mean(null_indices >= observed_index)

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

    valid = np.isfinite(null_indices)
    if not np.any(valid):
        return np.nan
    p = np.mean(null_indices[valid] >= observed_index)
    return p


def direction_psth(spike_times_by_direction_ms, window, bin_ms=50.0):
    """
    Mean PSTH (Hz) across trials for one direction, binned over the stim
    window. Cheap — just a histogram, unlike f1_amplitude/shuffle_pvalue —
    safe to compute for every direction alongside the DSI/OSI math without
    materially adding to per-cluster compute cost.
    """
    t0, t1 = window
    edges = np.arange(0, (t1 - t0) + bin_ms, bin_ms)
    n_bins = len(edges) - 1
    if n_bins <= 0:
        return np.array([]), np.array([])
    counts = np.zeros(n_bins, dtype=np.float64)
    n_trials = max(len(spike_times_by_direction_ms), 1)
    for sp in spike_times_by_direction_ms:
        sp_rel = np.asarray(sp) - t0
        mask = (sp_rel >= 0) & (sp_rel < (t1 - t0))
        c, _ = np.histogram(sp_rel[mask], bins=edges)
        counts += c
    rate = (counts / n_trials) / (bin_ms / 1000.0)
    t = edges[:-1] + bin_ms / 2.0
    return t, rate


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
                idx_by_dir[float(t["orientation"])].append(i)
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
    """True when a persisted result's DSOS/SF tags don't match the data.

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
    if cluster_id not in spike_times_by_trial:
        return None

    rng = np.random.default_rng(rng_seed)
    trials = spike_times_by_trial[cluster_id]

    pre_time_ms = trial_parameters[0]["preTime"]
    stim_time_ms = trial_parameters[0]["stimTime"]
    stim_window = (pre_time_ms, pre_time_ms + stim_time_ms)

    groups = group_grating_conditions(
        trial_parameters, min_directions_for_dsos=min_directions_for_dsos
    )
    condition_type = {g["key"]: g["condition_type"] for g in groups}

    result = {}
    for group in groups:
        bw, tf = group["key"]
        local_dirs = group["directions"]
        typ = group["condition_type"]
        idx_by_dir = group["idx_by_dir"]

        trial_resp_by_dir = {}
        for direction in local_dirs:
            idxs = idx_by_dir[direction]
            if response_metric == "f1":
                resp = np.array(
                    [
                        f1_amplitude(
                            trials[i],
                            stim_window,
                            float(trial_parameters[i]["temporalFrequency"]),
                        )
                        for i in idxs
                    ]
                )
            else:
                baseline = np.array(
                    [firing_rate_in_window(trials[i], (0.0, pre_time_ms)) for i in idxs]
                )
                evoked = np.array(
                    [firing_rate_in_window(trials[i], stim_window) for i in idxs]
                )
                resp = evoked - baseline
            trial_resp_by_dir[direction] = resp

        mean_resp = np.array([np.nanmean(trial_resp_by_dir[dd]) for dd in local_dirs])
        sem_resp = np.array(
            [
                np.nanstd(trial_resp_by_dir[dd], ddof=1)
                / np.sqrt(len(trial_resp_by_dir[dd]))
                for dd in local_dirs
            ]
        )

        entry = {
            "condition_type": typ,
            "directions_deg": np.array(local_dirs),
            "mean_response": mean_resp,
            "sem_response": sem_resp,
        }

        if typ == "dsos":
            # Per-direction firing-rate PSTHs — cheap, used by the GUI's
            # sanity-check strip so it can show a real time-resolved trace
            # at the preferred direction, not just the scalar mean_response.
            psth_by_dir = {}
            for direction in local_dirs:
                idxs = idx_by_dir[direction]
                t, rate = direction_psth([trials[i] for i in idxs], stim_window)
                psth_by_dir[direction] = rate
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
            group_idxs = [i for idxs in idx_by_dir.values() for i in idxs]
            peak_rate_hz = (
                np.nanmax(
                    [
                        firing_rate_in_window(trials[i], stim_window)
                        for i in group_idxs
                    ]
                )
                if group_idxs
                else np.nan
            )

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
