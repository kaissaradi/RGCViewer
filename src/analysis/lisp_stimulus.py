"""Grating runs described by a Lisp stimulus file (PLAN.md Q61).

Older rigs were driven by a Lisp stimulus program. For a list of stimuli it
writes the sequence it showed to a file named after the run (``s02``,
``s04.txt``): the first form is the stimulus every trial shares
(``:TYPE :DRIFTING-SINUSOID :FRAMES 960 ...``), each later form one trial's
own values (``:SPATIAL-PERIOD 64 :TEMPORAL-PERIOD 256 :DIRECTION 45``), in the
order shown. The trial start times are the TTL triggers in the Vision
``.neurons`` file.

The rules are those of the lab's MATLAB code (gdfield/matlab_base on GitHub):

* ``code/lab/load_stim.m``: the header form is skipped; a trial starts at
  the first trigger and at the trigger after every change of the
  inter-trigger interval larger than 2 SD of the intervals
  (``find_trigger``, ``trigger_iti_thr``). With fewer trial starts than
  trials the run ended early, and only whole repeats are kept
  (``correction_incomplet_run``).
* ``private/gfield/get_grating_spike_times.m``: a trial's spikes are those in
  [start, start + stimulus duration), timed from the start.

The stimulus duration is ``:FRAMES`` at REFRESH_HZ. The lab's scripts write
8 s as ``:frames (* 8 120)``, and the MATLAB analysis uses 8 s for 960 frames.
The 100-frame trigger interval measured on 2012-10-15-0/data002 is 832.8 ms,
which is 120.1 Hz.

Encore keeps up to 1 s before and after each trial, never past the
neighbouring trial, so rasters show the onset and the Δ-rate metric has a
baseline. The DS/OS statistics use the stimulus window only, as MATLAB does.

What ``:DIRECTION`` means on the retina (which way the bars move, and in
which screen frame) is not verified here. Encore shows the value as it is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REFRESH_HZ = 120.0
TRIGGER_ITI_THR = 2.0          # load_stim.m: trigger_iti_thr
PAD_MS = 1000.0                # context kept before / after each trial
GRATING_TYPES = ("DRIFTING-SINUSOID", "DRIFTING-SQUAREWAVE")


class LispStimulusError(ValueError):
    """The file is not a stimulus sequence Encore can use for this run."""


# ── reading the file ──────────────────────────────────────────────────────────

_TOKEN = re.compile(r"#\(|\(|\)|[^\s()]+")


def _atom(tok: str):
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        return tok


def read_forms(text: str) -> list:
    """Top-level forms of Lisp data: lists as ``list``, ``#(...)`` as ``tuple``."""
    text = re.sub(r";[^\n]*", " ", text)
    stack: list = [[]]
    kinds: list = []
    for tok in _TOKEN.findall(text):
        if tok in ("(", "#("):
            stack.append([])
            kinds.append(tok)
        elif tok == ")":
            if not kinds:
                raise LispStimulusError("Unbalanced ')' in the stimulus file.")
            items = stack.pop()
            stack[-1].append(tuple(items) if kinds.pop() == "#(" else items)
        else:
            stack[-1].append(_atom(tok))
    if kinds:
        raise LispStimulusError("The stimulus file ends inside a form (missing ')').")
    return stack[0]


def _key(tok) -> Optional[str]:
    """``:SPATIAL-PERIOD`` → ``SPATIAL_PERIOD`` (load_stim.m's field names)."""
    if isinstance(tok, str) and tok.startswith(":") and len(tok) > 1:
        return tok[1:].upper().replace("-", "_")
    return None


def _plist(form) -> Dict[str, object]:
    if not isinstance(form, list):
        raise LispStimulusError("Expected a (:KEY value ...) list.")
    out = {}
    i = 0
    while i < len(form):
        k = _key(form[i])
        if k is None or i + 1 >= len(form):
            raise LispStimulusError(f"Expected :KEY value pairs, got {form[i]!r}.")
        v = form[i + 1]
        if isinstance(v, str) and v.startswith(":"):
            v = v[1:].upper()
        out[k] = v
        i += 2
    return out


@dataclass
class StimulusSequence:
    path: Path
    header: Dict[str, object]
    trials: List[Dict[str, object]]
    combinations: List[Dict[str, object]] = field(default_factory=list)
    trial_list: List[int] = field(default_factory=list)   # index into combinations

    @property
    def stim_type(self) -> str:
        return str(self.header.get("TYPE", "")).upper()

    @property
    def repetitions(self) -> float:
        return len(self.trials) / max(1, len(self.combinations))


def read_sequence(path) -> StimulusSequence:
    """The header and trials of a stimulus sequence file (``s02``)."""
    path = Path(path)
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        raise LispStimulusError(f"Could not read {path}: {exc}") from exc
    forms = read_forms(text)
    if not forms or not isinstance(forms[0], list) or _key(forms[0][0] if forms[0] else None) != "TYPE":
        raise LispStimulusError(
            f"{path.name} is not a stimulus sequence. A sequence file starts with "
            "(:TYPE ...); a file that starts with (let ...) is the script that ran it. "
            "Pick the file the run wrote, named after the run (s02 for data002).")
    header = _plist(forms[0])
    trials = [_plist(f) for f in forms[1:]]
    if not trials:
        raise LispStimulusError(f"{path.name} lists no trials after the header.")
    combos: List[Dict[str, object]] = []
    trial_list = []
    for t in trials:
        try:
            trial_list.append(combos.index(t))
        except ValueError:
            combos.append(t)
            trial_list.append(len(combos) - 1)
    return StimulusSequence(path, header, trials, combos, trial_list)


# ── triggers → trials ─────────────────────────────────────────────────────────

def trial_starts(triggers: Sequence[float], thr_factor: float = TRIGGER_ITI_THR) -> np.ndarray:
    """load_stim.m ``find_trigger``: first trigger, then each one after an interval change."""
    trig = np.asarray(triggers, dtype=np.float64)
    if trig.size < 3:
        return trig.copy()
    tt = np.diff(trig)
    thr = np.std(tt, ddof=1) * thr_factor
    starts = [trig[0]]
    alt = tt[0]
    for i in range(1, tt.size):
        j = i
        if abs(tt[i] - alt) > thr:
            starts.append(trig[i + 1])
            if i < tt.size - 1:
                j = i + 1
        alt = tt[j]
    return np.asarray(starts)


@dataclass
class GratingTrials:
    """Encore's raw grating schema plus what was read and why trials were dropped."""
    spike_times_by_trial: Dict[int, List[np.ndarray]]
    trial_parameters: List[dict]
    summary: Dict[str, object]

    def as_raw(self) -> dict:
        return {"spike_times_by_trial": self.spike_times_by_trial,
                "trial_parameters": self.trial_parameters}


def _trial_value(trial, header, key, default=None):
    return trial.get(key, header.get(key, default))


def build_grating_trials(seq: StimulusSequence, triggers_samples, spikes_by_id: Dict[int, np.ndarray],
                         fs: float, refresh_hz: float = REFRESH_HZ,
                         pad_ms: float = PAD_MS, recording_end_samples=None) -> GratingTrials:
    """Per-trial spike times (ms, from each trial's window start) for every cell.

    ``spikes_by_id`` is sample numbers per cell, on the same clock as the triggers.
    """
    if seq.stim_type not in GRATING_TYPES:
        raise LispStimulusError(
            f"{seq.path.name} is a {seq.stim_type.lower() or 'unknown'} stimulus. "
            "Encore reads drifting gratings (drifting-sinusoid, drifting-squarewave) from Lisp files.")
    missing = [k for k in ("SPATIAL_PERIOD", "TEMPORAL_PERIOD", "DIRECTION")
               if any(_trial_value(t, seq.header, k) is None for t in seq.trials)]
    if missing:
        raise LispStimulusError(f"{seq.path.name}: trials without {', '.join(missing)}.")
    fs = float(fs)
    trig_s = np.asarray(triggers_samples, dtype=np.float64) / fs
    if trig_s.size == 0:
        raise LispStimulusError("The .neurons file has no trigger times.")
    n_file = len(seq.trials)
    # One trigger per trial is unambiguous; otherwise load_stim's rule.
    starts = trig_s.copy() if trig_s.size == n_file else trial_starts(trig_s)
    n_combo = len(seq.combinations)
    n_starts = int(starts.size)
    notes = []
    trials = list(seq.trials)
    if starts.size > n_file:
        raise LispStimulusError(
            f"{starts.size} trial starts in the triggers but {n_file} trials in "
            f"{seq.path.name}. This file does not describe this run.")
    if starts.size < n_file:
        reps = starts.size // n_combo
        if reps < 1:
            raise LispStimulusError(
                f"{starts.size} trial starts in the triggers, fewer than one repeat of the "
                f"{n_combo} conditions in {seq.path.name}.")
        keep = reps * n_combo
        notes.append(f"The run ended early: {starts.size} of {n_file} trials started; "
                     f"kept {reps} whole repeats ({keep} trials), as MATLAB load_stim does.")
        trials = trials[:keep]
        starts = starts[:keep]

    stim_ms = []
    for t in trials:
        frames = _trial_value(t, seq.header, "FRAMES")
        if frames is None:
            raise LispStimulusError(f"{seq.path.name}: no :FRAMES, so the trial length is unknown.")
        stim_ms.append(float(frames) / refresh_hz * 1000.0)
    stim_ms = np.asarray(stim_ms)
    start_ms = starts * 1000.0
    end_ms = start_ms + stim_ms
    rec_end_ms = (float(recording_end_samples) / fs * 1000.0
                  if recording_end_samples is not None else np.inf)
    prev_end = np.r_[0.0, end_ms[:-1]]
    next_start = np.r_[start_ms[1:], rec_end_ms]
    pre_ms = np.clip(np.minimum(pad_ms, start_ms - prev_end), 0.0, None)
    tail_ms = np.clip(np.minimum(pad_ms, next_start - end_ms), 0.0, None)
    short = int(np.sum(end_ms > rec_end_ms))
    if short:
        notes.append(f"{short} trial(s) run past the end of the recording.")

    params = []
    for t, pre, stim, tail in zip(trials, pre_ms, stim_ms, tail_ms):
        sp = float(_trial_value(t, seq.header, "SPATIAL_PERIOD"))
        tp = float(_trial_value(t, seq.header, "TEMPORAL_PERIOD"))
        params.append({
            "barWidth": sp,
            "temporalFrequency": refresh_hz / tp,
            "orientation": float(_trial_value(t, seq.header, "DIRECTION")),
            "preTime": float(pre), "stimTime": float(stim), "tailTime": float(tail),
            "spatialPeriod_px": sp, "temporalPeriod_frames": tp,
            "stimulusType": seq.stim_type.lower(), "source": "lisp",
        })

    win0 = (start_ms - pre_ms) / 1000.0 * fs      # window edges in samples
    win1 = (end_ms + tail_ms) / 1000.0 * fs
    by_cell: Dict[int, List[np.ndarray]] = {}
    for cid, samples in spikes_by_id.items():
        s = np.sort(np.asarray(samples, dtype=np.float64))
        a = np.searchsorted(s, win0, side="left")
        b = np.searchsorted(s, win1, side="left")
        by_cell[int(cid)] = [(s[i:j] - w) / fs * 1000.0 for i, j, w in zip(a, b, win0)]

    intra = np.diff(trig_s)
    intra = intra[intra < np.median(intra) * 1.5] if intra.size else intra
    tps = sorted({p["temporalPeriod_frames"] for p in params})
    sps = sorted({p["spatialPeriod_px"] for p in params})
    dirs = sorted({p["orientation"] for p in params})
    summary = {
        "path": str(seq.path), "type": seq.stim_type.lower(),
        "n_trials_file": n_file, "n_trials": len(params), "n_conditions": n_combo,
        "repetitions": len(params) // max(1, n_combo),
        "repetitions_file": seq.repetitions,
        "n_triggers": int(trig_s.size), "n_trial_starts": n_starts,
        "trigger_interval_ms": float(np.median(intra) * 1000.0) if intra.size else float("nan"),
        "stim_ms": float(np.median(stim_ms)), "refresh_hz": refresh_hz,
        "spatial_periods_px": sps, "temporal_periods_frames": tps, "directions_deg": dirs,
        "notes": notes,
    }
    return GratingTrials(by_cell, params, summary)


def describe(summary: Dict[str, object]) -> str:
    """One paragraph for the status bar / dialog."""
    s = summary
    tps = ", ".join(f"{t:g}" for t in s["temporal_periods_frames"])
    sps = ", ".join(f"{p:g}" for p in s["spatial_periods_px"])
    text = (f"{Path(str(s['path'])).name}: {s['type'].replace('-', ' ')}, "
            f"{s['n_conditions']} conditions (spatial period {sps} px; temporal period {tps} "
            f"frames; {len(s['directions_deg'])} directions), {s['repetitions']} repeats, "
            f"{s['n_trials']} trials of {s['stim_ms'] / 1000:.2f} s.")
    for n in s.get("notes") or []:
        text += " " + n
    return text


# ── finding the file ──────────────────────────────────────────────────────────

_RUN_NUMBER = re.compile(r"data(\d{3})", re.IGNORECASE)


def sequence_file_names(dataset_name: str) -> List[str]:
    """``data002`` → ``["s02", "s02.txt", "s002", "s002.txt"]``."""
    m = _RUN_NUMBER.search(str(dataset_name or ""))
    if not m:
        return []
    n = int(m.group(1))
    return [f"s{n:02d}", f"s{n:02d}.txt", f"s{n:03d}", f"s{n:03d}.txt"]


def find_sequence_files(run_dir, dataset_name, levels: int = 4) -> List[Path]:
    """Likely sequence files for a run: ``stimuli/`` or ``Visual/`` above it. One listing each."""
    names = {n.lower() for n in sequence_file_names(dataset_name)}
    if not names:
        return []
    run_dir = Path(run_dir)
    out, seen = [], set()
    for anc in [run_dir, *list(run_dir.parents)[:levels]]:
        for d in (anc / "stimuli", anc / "Visual", anc):
            key = str(d)
            if key in seen:
                continue
            seen.add(key)
            try:
                entries = [e for e in d.iterdir() if e.is_file()]
            except OSError:
                continue
            out.extend(e for e in entries if e.name.lower() in names)
    return out


def stimulus_dirs(run_dir, levels: int = 4) -> List[Path]:
    """``stimuli/`` and ``Visual/`` folders above a run, nearest first (dialog start)."""
    run_dir = Path(run_dir)
    out = []
    for anc in [run_dir, *list(run_dir.parents)[:levels]]:
        for d in (anc / "stimuli", anc / "Visual"):
            if d.is_dir():
                out.append(d)
    return out
