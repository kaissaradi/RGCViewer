"""Chirp stimulus reconstruction and per-phase tuning measures.

The ChirpStimulus PSTH is 25 s of four concatenated stimulus phases, and almost
every useful number about a cell comes from treating them separately. This
module rebuilds the stimulus from the recorded parameters (never hardcoded) and
turns each phase into something readable:

    bright step   0-3 s     -> transition responses, ON/OFF index
    grey          3-4 s
    dark step     4-7 s     -> transition responses
    grey          7-8 s
    frequency     8-16 s    -> frequency tuning curve (response vs Hz)
    grey          16-17 s   -> the oscillation control: no stimulus is on
    contrast ramp 17-25 s   -> contrast tuning curve (response vs contrast)

**The stimulus has two increments and two decrements.** Light steps up at 0 s
and again at 7 s (the *offset* of the dark step); it steps down at 3 s and 4 s.
The large transient most cells show near 7 s is therefore ON drive, and reading
it as a second bright step — or as an OFF response — inverts the cell's
polarity. This matters especially in the rd10 + Grm6-waChR preparations, where
excitation enters at the ON bipolar cells so every response is increment-driven
by construction.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "ChirpTiming", "timing_from", "stimulus_intensity", "frequency_tuning",
    "contrast_tuning", "transition_responses", "on_off_index",
]

# Response to a step is measured over this window after the transition.
TRANSITION_WINDOW_S = 0.35


class ChirpTiming:
    """Phase boundaries in seconds, plus the parameters that shape each phase."""

    __slots__ = ("step", "inter", "freq", "contrast", "f_min", "f_max",
                 "c_min", "c_max", "c_freq", "step_contrast", "background",
                 "total")

    def __init__(self, step=3.0, inter=1.0, freq=8.0, contrast=8.0,
                 f_min=0.0, f_max=15.0, c_min=0.02, c_max=1.0, c_freq=2.0,
                 step_contrast=1.0, background=0.5):
        self.step = step
        self.inter = inter
        self.freq = freq
        self.contrast = contrast
        self.f_min, self.f_max = f_min, f_max
        self.c_min, self.c_max = c_min, c_max
        self.c_freq = c_freq
        self.step_contrast = step_contrast
        self.background = background
        self.total = 2 * step + 3 * inter + freq + contrast

    # Phase edges, all derived so a preparation with different timings works.
    @property
    def bright(self):
        return (0.0, self.step)

    @property
    def dark(self):
        return (self.step + self.inter, 2 * self.step + self.inter)

    @property
    def sweep(self):
        t0 = 2 * self.step + 2 * self.inter
        return (t0, t0 + self.freq)

    @property
    def gap(self):
        """The grey interval after the sweep — no stimulus, so whatever the
        cell does here is intrinsic (in rd10 retina, 5-15 Hz oscillation)."""
        t0 = self.sweep[1]
        return (t0, t0 + self.inter)

    @property
    def ramp(self):
        t0 = self.gap[1]
        return (t0, t0 + self.contrast)

    @property
    def transitions(self):
        """``[(label, time_s, +1 increment / -1 decrement), ...]``."""
        return [
            ("light on", self.bright[0], +1),
            ("light off", self.bright[1], -1),
            ("dark on", self.dark[0], -1),
            ("dark off", self.dark[1], +1),
        ]

    def phases(self):
        return {"bright step": self.bright, "dark step": self.dark,
                "frequency sweep": self.sweep, "grey gap": self.gap,
                "contrast ramp": self.ramp}


def timing_from(params=None, chirp_data=None) -> ChirpTiming:
    """Build timing from the Symphony manifest parameters, falling back to the
    chirp file's own fields.

    The manifest is preferred because it is the only source for the contrast
    ramp's depth and carrier frequency; the analysis file records the phase
    durations but not those.
    """
    p = dict(params or {})
    d = dict(chirp_data or {})

    def pick(pkey, dkey, default):
        for source, key in ((p, pkey), (d, dkey)):
            v = source.get(key)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return default

    return ChirpTiming(
        step=pick("stepTime", "step_ms", 3000.0) / 1000.0,
        inter=pick("interTime", "inter_ms", 1000.0) / 1000.0,
        freq=pick("frequencyTime", "freq_ms", 8000.0) / 1000.0,
        contrast=pick("contrastTime", "contrast_ms", 8000.0) / 1000.0,
        f_min=pick("frequencyMin", "freq_min_hz", 0.0),
        f_max=pick("frequencyMax", "freq_max_hz", 15.0),
        c_min=pick("contrastMin", "__", 0.02),
        c_max=pick("contrastMax", "__", 1.0),
        c_freq=pick("contrastFrequency", "__", 2.0),
        step_contrast=pick("stepContrast", "__", 1.0),
        background=pick("backgroundIntensity", "__", 0.5),
    )


def stimulus_intensity(timing: ChirpTiming, n_points=4000):
    """``(t_seconds, intensity)`` for the stimulus as a figure would draw it.

    Intensity is in projector units where *background* is mid-grey, so a full-
    contrast bright step reaches 1.0 and the dark step 0.0 — the convention
    every published chirp figure uses.
    """
    t = np.linspace(0.0, timing.total, int(n_points))
    bg = timing.background
    y = np.full_like(t, bg)

    b0, b1 = timing.bright
    y[(t >= b0) & (t < b1)] = bg * (1.0 + timing.step_contrast)
    d0, d1 = timing.dark
    y[(t >= d0) & (t < d1)] = bg * (1.0 - timing.step_contrast)

    s0, s1 = timing.sweep
    m = (t >= s0) & (t < s1)
    ts = t[m] - s0
    # Linear frequency sweep: instantaneous f = f_min + (f_max-f_min)*ts/T, so
    # the phase is its integral. Using f*t directly would sweep twice as fast.
    k = (timing.f_max - timing.f_min) / max(timing.freq, 1e-9)
    phase = 2 * np.pi * (timing.f_min * ts + 0.5 * k * ts ** 2)
    y[m] = bg * (1.0 + np.sin(phase))

    r0, r1 = timing.ramp
    m = (t >= r0) & (t < r1)
    tr = t[m] - r0
    amp = timing.c_min + (timing.c_max - timing.c_min) * tr / max(timing.contrast, 1e-9)
    y[m] = bg * (1.0 + amp * np.sin(2 * np.pi * timing.c_freq * tr))
    return t, y


def _windowed_amplitude(psth, bin_ms, t0, t1, target_hz, win_s, step_s):
    """Sliding-window amplitude at *target_hz* (a callable of window centre).

    Returns ``(centres_s, amplitude_hz, snr)``. The noise floor is the median
    amplitude across the rest of the spectrum with the fundamental and its
    second harmonic excluded, so ``snr`` has an expected value near 1 when the
    cell is not following the stimulus.
    """
    dt = bin_ms / 1000.0
    a, b = int(round(t0 / dt)), int(round(t1 / dt))
    w, st = max(int(round(win_s / dt)), 8), max(int(round(step_s / dt)), 1)
    if b - a < w:
        return np.array([]), np.array([]), np.array([])
    freqs = np.fft.rfftfreq(w, d=dt)
    centres, amps, snrs = [], [], []
    for s0 in range(0, (b - a) - w + 1, st):
        seg = np.asarray(psth[a + s0:a + s0 + w], dtype=float)
        seg = seg - seg.mean()
        spec = np.abs(np.fft.rfft(seg))
        centre = (s0 + w / 2.0) * dt
        f = target_hz(centre)
        if not np.isfinite(f) or f <= 0:
            continue
        k = int(np.argmin(np.abs(freqs - f)))
        if k == 0:
            continue
        keep = np.ones(len(freqs), bool)
        for kk in (k, 2 * k):
            keep[max(kk - 1, 0):kk + 2] = False
        keep[freqs < 1.0] = False
        floor = np.median(spec[keep]) if keep.any() else 0.0
        amp = 2.0 * spec[k] / w
        centres.append(centre)
        amps.append(amp)
        snrs.append(spec[k] / floor if floor > 0 else np.nan)
    return np.array(centres), np.array(amps), np.array(snrs)


def frequency_tuning(psth, timing: ChirpTiming, bin_ms, win_s=1.0, step_s=0.25):
    """Response amplitude against the sweep's instantaneous frequency.

    A real tuning curve rather than the raw sweep trace: within each window the
    amplitude is taken at the frequency the stimulus is *currently* at, which
    is phase-invariant and so unaffected by the cell's response latency.

    The window sets the frequency resolution (1 s -> 1 Hz), so frequencies
    below ~1.5 Hz are not resolvable and are dropped.
    """
    s0, s1 = timing.sweep
    k = (timing.f_max - timing.f_min) / max(timing.freq, 1e-9)

    def target(centre_in_window):
        return timing.f_min + k * centre_in_window

    centres, amps, snrs = _windowed_amplitude(
        psth, bin_ms, s0, s1, target, win_s, step_s)
    if not len(centres):
        return np.array([]), np.array([]), np.array([])
    freqs = timing.f_min + k * centres
    keep = freqs >= 1.5
    return freqs[keep], amps[keep], snrs[keep]


def contrast_tuning(psth, timing: ChirpTiming, bin_ms, win_s=1.0, step_s=0.25):
    """Response amplitude at the ramp's carrier against instantaneous contrast."""
    r0, r1 = timing.ramp
    centres, amps, snrs = _windowed_amplitude(
        psth, bin_ms, r0, r1, lambda _c: timing.c_freq, win_s, step_s)
    if not len(centres):
        return np.array([]), np.array([]), np.array([])
    contrasts = timing.c_min + (timing.c_max - timing.c_min) * centres / max(
        timing.contrast, 1e-9)
    return contrasts, amps, snrs


def transition_responses(psth, timing: ChirpTiming, bin_ms,
                         window_s=TRANSITION_WINDOW_S):
    """Peak rate in the window after each of the four stimulus transitions.

    Returns ``[{label, time_s, polarity, peak_hz, baseline_hz, delta_hz}, ...]``.
    Baseline is the 500 ms of grey before the transition where there is one, so
    ``delta_hz`` is the evoked change rather than the absolute rate.
    """
    dt = bin_ms / 1000.0
    psth = np.asarray(psth, dtype=float)
    n = len(psth)
    out = []
    for label, t0, polarity in timing.transitions:
        a = int(round(t0 / dt))
        b = min(int(round((t0 + window_s) / dt)), n)
        if a >= n or b <= a:
            continue
        pre_a = max(int(round((t0 - 0.5) / dt)), 0)
        baseline = float(psth[pre_a:a].mean()) if a > pre_a else float("nan")
        peak = float(psth[a:b].max())
        out.append({
            "label": label, "time_s": t0, "polarity": polarity,
            "peak_hz": peak, "baseline_hz": baseline,
            "delta_hz": peak - baseline if np.isfinite(baseline) else peak,
        })
    return out


def on_off_index(psth, timing: ChirpTiming, bin_ms):
    """``(index, increment_hz, decrement_hz)`` in [-1, 1]; +1 is purely ON.

    Computed over the *transitions*, pooling the two increments (light on at
    0 s, dark off at 7 s) against the two decrements (light off at 3 s, dark on
    at 4 s). Pooling both of each is what makes this robust on chirp: a cell
    that responds to the 7 s dark-step offset is ON-driven, and an index built
    from the bright step alone would miss it entirely.
    """
    resp = transition_responses(psth, timing, bin_ms)
    if not resp:
        return float("nan"), float("nan"), float("nan")
    inc = [max(r["delta_hz"], 0.0) for r in resp if r["polarity"] > 0]
    dec = [max(r["delta_hz"], 0.0) for r in resp if r["polarity"] < 0]
    on = float(np.mean(inc)) if inc else 0.0
    off = float(np.mean(dec)) if dec else 0.0
    total = on + off
    if total <= 0:
        return float("nan"), on, off
    return (on - off) / total, on, off
