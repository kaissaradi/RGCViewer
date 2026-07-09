"""
grating_calc.py

Single-cluster DSI/OSI + bar-width tuning computation, ported from
combined_grating_analysis.py's per-file batch logic (f1_amplitude,
vector_sum_index, shuffle_pvalue — copied verbatim, not reimplemented, so
GUI-computed results match what the offline script would produce).

Used when only a raw grating .npy (spike_times_by_trial + trial_parameters)
is on disk and no precomputed analyzed file exists. Scoped to run for ONE
cluster at a time — this is what makes it cheap enough to run synchronously
inside a background worker per cluster-selection, instead of needing to
batch-precompute the whole dataset (890 clusters x N conditions x 1000
shuffles) at load time.
"""
from collections import defaultdict

import numpy as np

PSTH_BIN_MS = 5.0
N_SHUFFLES = 1000
RNG_SEED = 0
MIN_DIRECTIONS_FOR_DSOS = 8

# --- "Best condition" / DS-OS classification -------------------------------
# Selection previously used max(|DSI|) alone, which is amplitude-blind and
# significance-blind: a condition with a handful of spikes that happened to
# land in one direction by chance can produce a higher |DSI| than a
# condition with a large, clearly time-locked, less-perfectly-concentrated
# response. Gate on both a minimum response amplitude AND a significance
# test before ranking by |DSI|/|OSI|, so "best" means "reliable," not just
# "numerically largest."
MIN_RESPONSE_HZ = 2.0      # peak-condition response floor (Hz-equivalent for f1/delta metrics)
ALPHA = 0.05                # shuffle-test significance threshold
DSI_THRESHOLD = 0.3         # DS classification cutoff, applied AFTER gating
OSI_THRESHOLD = 0.3         # OS classification cutoff, applied AFTER gating


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
        observed_means = np.array([np.nanmean(trial_responses_by_dir[d]) for d in directions])
        observed_index, _ = vector_sum_index(np.array(directions), observed_means, harmonic)
        if not np.isfinite(observed_index):
            return np.nan
        null_indices = np.empty(n_shuffles)
        for s in range(n_shuffles):
            shuffled = rng.permutation(all_resp)
            means = np.array([
                np.nanmean(shuffled[boundaries[i]:boundaries[i + 1]])
                for i in range(len(directions))
            ])
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
    with np.errstate(invalid='ignore', divide='ignore'):
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


def compute_grating_response(cluster_id, spike_times_by_trial, trial_parameters,
                              n_shuffles=N_SHUFFLES,
                              min_directions_for_dsos=MIN_DIRECTIONS_FOR_DSOS,
                              response_metric='f1', rng_seed=RNG_SEED):
    """
    Compute DSI/OSI (or bar-width tuning point) for ONE cluster, across
    every (barWidth, temporalFrequency) condition present in the raw file.

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

    pre_time_ms = trial_parameters[0]['preTime']
    stim_time_ms = trial_parameters[0]['stimTime']
    stim_window = (pre_time_ms, pre_time_ms + stim_time_ms)

    conditions = sorted(set((t['barWidth'], t['temporalFrequency']) for t in trial_parameters))
    dirs_by_condition = {}
    for (bw, tf) in conditions:
        dirs_here = sorted(set(t['orientation'] for t in trial_parameters
                                if t['barWidth'] == bw and t['temporalFrequency'] == tf))
        dirs_by_condition[(bw, tf)] = dirs_here

    condition_type = {
        cond: ('dsos' if len(dirs) >= min_directions_for_dsos else 'sf')
        for cond, dirs in dirs_by_condition.items()
    }

    result = {}
    for (bw, tf) in conditions:
        local_dirs = dirs_by_condition[(bw, tf)]
        typ = condition_type[(bw, tf)]

        idx_by_dir = defaultdict(list)
        for i, t in enumerate(trial_parameters):
            if t['barWidth'] == bw and t['temporalFrequency'] == tf:
                idx_by_dir[t['orientation']].append(i)

        trial_resp_by_dir = {}
        for direction in local_dirs:
            idxs = idx_by_dir[direction]
            if response_metric == 'f1':
                resp = np.array([f1_amplitude(trials[i], stim_window, tf) for i in idxs])
            else:
                baseline = np.array([
                    firing_rate_in_window(trials[i], (0.0, pre_time_ms)) for i in idxs
                ])
                evoked = np.array([firing_rate_in_window(trials[i], stim_window) for i in idxs])
                resp = evoked - baseline
            trial_resp_by_dir[direction] = resp

        mean_resp = np.array([np.nanmean(trial_resp_by_dir[dd]) for dd in local_dirs])
        sem_resp = np.array([
            np.nanstd(trial_resp_by_dir[dd], ddof=1) / np.sqrt(len(trial_resp_by_dir[dd]))
            for dd in local_dirs
        ])

        entry = {
            'condition_type': typ,
            'directions_deg': np.array(local_dirs),
            'mean_response': mean_resp,
            'sem_response': sem_resp,
        }

        if typ == 'dsos':
            # Per-direction firing-rate PSTHs — cheap, used by the GUI's
            # sanity-check strip so it can show a real time-resolved trace
            # at the preferred direction, not just the scalar mean_response.
            psth_by_dir = {}
            for direction in local_dirs:
                idxs = idx_by_dir[direction]
                t, rate = direction_psth([trials[i] for i in idxs], stim_window)
                psth_by_dir[direction] = rate
            entry['psth_time_s'] = (t / 1000.0) if local_dirs else np.array([])
            entry['psth_by_direction'] = psth_by_dir

            dsi, pref_dir = vector_sum_index(np.array(local_dirs), mean_resp, harmonic=1)
            osi, pref_ori = vector_sum_index(np.array(local_dirs), mean_resp, harmonic=2)
            dsi_p = shuffle_pvalue(local_dirs, trial_resp_by_dir, harmonic=1,
                                    n_shuffles=n_shuffles, rng=rng)
            osi_p = shuffle_pvalue(local_dirs, trial_resp_by_dir, harmonic=2,
                                    n_shuffles=n_shuffles, rng=rng)

            # peak_rate_hz: real evoked firing rate (Hz), independent of
            # response_metric ('f1' amplitude / 'delta' aren't in Hz units
            # and aren't comparable across conditions run with different
            # tf). This is the amplitude-floor gate for best-condition
            # selection — see select_best_dsos_condition — so a condition
            # with a handful of noisy spikes can't out-rank a condition
            # with a real, strong response just because its DSI happens
            # to be numerically higher.
            peak_rate_hz = np.nanmax([
                firing_rate_in_window(trials[i], stim_window)
                for i in range(len(trials))
                if trial_parameters[i]['barWidth'] == bw
                and trial_parameters[i]['temporalFrequency'] == tf
            ]) if local_dirs else np.nan

            entry.update({
                'DSI': dsi, 'preferred_direction_deg': pref_dir, 'DSI_pvalue': dsi_p,
                'OSI': osi, 'preferred_orientation_deg': pref_ori, 'OSI_pvalue': osi_p,
                'peak_rate_hz': float(peak_rate_hz) if np.isfinite(peak_rate_hz) else np.nan,
            })
        else:
            entry.update({
                'DSI': np.nan, 'preferred_direction_deg': np.nan, 'DSI_pvalue': np.nan,
                'OSI': np.nan, 'preferred_orientation_deg': np.nan, 'OSI_pvalue': np.nan,
                'bw_tuning_point': np.nanmean(mean_resp),
                'bw_tuning_point_sem': (np.nanstd(mean_resp, ddof=1) / np.sqrt(len(local_dirs))
                                        if len(local_dirs) > 1 else np.nan),
            })

        result[(bw, tf)] = entry

    sf_bar_widths = sorted(set(bw for (bw, tf), typ in condition_type.items() if typ == 'sf'))
    if sf_bar_widths:
        curve = np.full(len(sf_bar_widths), np.nan)
        for j, bw in enumerate(sf_bar_widths):
            vals = [result[(bw2, tf)]['bw_tuning_point']
                    for (bw2, tf) in conditions if bw2 == bw and condition_type[(bw2, tf)] == 'sf']
            if vals:
                curve[j] = np.nanmean(vals)
        result['sf_bar_widths'] = np.array(sf_bar_widths)
        result['sf_tuning_curve'] = curve

    return result


def select_best_dsos_condition(data, min_response_hz=MIN_RESPONSE_HZ, alpha=ALPHA,
                                dsi_threshold=DSI_THRESHOLD, osi_threshold=OSI_THRESHOLD):
    """
    Picks the single 'best' (barWidth, temporalFrequency) condition and a
    DS/OS classification for one cluster, replacing the old max(|DSI|)
    selection used independently (and inconsistently) in GratingPanel and
    the population DS/OS probe map.

    `data` is the per-cluster dict returned by compute_grating_response
    (or the equivalent pre-analyzed-file entry) — i.e. data[cluster_id].

    Selection logic, in order:
      1. GATE: a condition is only eligible if peak_rate_hz > min_response_hz
         AND its relevant p-value < alpha. This is the actual bug fix — the
         old code ranked by raw |DSI| with no amplitude or significance
         floor, so a near-silent condition with a few spikes that landed in
         one direction by chance could out-rank a condition with a real,
         strong, time-locked response.
      2. CLASSIFY: DS takes priority over OS, matching the reference
         notebook's convention (DSI > threshold => DS, regardless of OSI;
         only cells that clear the OSI bar *without* clearing the DSI bar
         are called OS). A single-lobed tuning curve mathematically pushes
         up both DSI and OSI together, so DSI-first is the standard/non-
         arbitrary way to split them, rather than comparing DSI vs OSI
         head-to-head (they're different harmonics and not on a shared
         scale — see harmonic=1 vs harmonic=2 in vector_sum_index).
      3. RANK: among gated conditions of the winning class (DS or OS),
         pick the highest |DSI| (or |OSI| for OS) — same idea as before,
         just restricted to conditions that already passed the gate.

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
        c for c in data
        if isinstance(c, tuple) and data[c].get('condition_type') == 'dsos'
    ]
    if not dsos_conditions:
        return None

    def _passes_gate(entry, pvalue_key):
        # NOTE: pre-analyzed files loaded from disk (combined_grating_
        # analysis.py output, predating this gate) won't have
        # 'peak_rate_hz' in their per-condition dict. That's a schema gap,
        # not a "this condition is weak" signal — but np.isfinite(nan) is
        # False either way, so it fails the gate and this cluster reports
        # classification='none' until re-run through compute_grating_
        # response. This is intentionally conservative (silently passing
        # ungated legacy data through would reintroduce the exact bug this
        # function exists to fix) but it does mean old analyzed files go
        # dark on first load — flag this to the user rather than treating
        # it as a quiet edge case.
        peak_hz = entry.get('peak_rate_hz', np.nan)
        pval = entry.get(pvalue_key, np.nan)
        if not np.isfinite(peak_hz) or peak_hz <= min_response_hz:
            return False
        if not np.isfinite(pval) or pval >= alpha:
            return False
        return True

    ds_candidates = [c for c in dsos_conditions if _passes_gate(data[c], 'DSI_pvalue')]
    os_candidates = [c for c in dsos_conditions if _passes_gate(data[c], 'OSI_pvalue')]

    def _abs_or_neg1(val):
        return abs(val) if np.isfinite(val) else -1.0

    best_ds = max(ds_candidates, key=lambda c: _abs_or_neg1(data[c].get('DSI', np.nan))) \
        if ds_candidates else None
    if best_ds is not None and abs(data[best_ds].get('DSI', 0.0)) > dsi_threshold:
        entry = data[best_ds]
        return {
            'condition': best_ds,
            'classification': 'DS',
            'DSI': entry.get('DSI', np.nan), 'OSI': entry.get('OSI', np.nan),
            'preferred_direction_deg': entry.get('preferred_direction_deg', np.nan),
            'preferred_orientation_deg': entry.get('preferred_orientation_deg', np.nan),
            'DSI_pvalue': entry.get('DSI_pvalue', np.nan), 'OSI_pvalue': entry.get('OSI_pvalue', np.nan),
            'peak_rate_hz': entry.get('peak_rate_hz', np.nan),
        }

    best_os = max(os_candidates, key=lambda c: _abs_or_neg1(data[c].get('OSI', np.nan))) \
        if os_candidates else None
    if best_os is not None and abs(data[best_os].get('OSI', 0.0)) > osi_threshold:
        entry = data[best_os]
        return {
            'condition': best_os,
            'classification': 'OS',
            'DSI': entry.get('DSI', np.nan), 'OSI': entry.get('OSI', np.nan),
            'preferred_direction_deg': entry.get('preferred_direction_deg', np.nan),
            'preferred_orientation_deg': entry.get('preferred_orientation_deg', np.nan),
            'DSI_pvalue': entry.get('DSI_pvalue', np.nan), 'OSI_pvalue': entry.get('OSI_pvalue', np.nan),
            'peak_rate_hz': entry.get('peak_rate_hz', np.nan),
        }

    return {
        'condition': None, 'classification': 'none',
        'DSI': np.nan, 'OSI': np.nan,
        'preferred_direction_deg': np.nan, 'preferred_orientation_deg': np.nan,
        'DSI_pvalue': np.nan, 'OSI_pvalue': np.nan, 'peak_rate_hz': np.nan,
    }