"""Which stimulus ran when, within one sorted recording (PLAN.md Q41).

A sort covers one run (``data022``) or several recorded back to back
(``data007-010``, ``data000_data001_data004``). The stimulus manifest
(``stimuli/<exp>.json``) gives each run's protocol and length in samples.
Laid end to end in folder order, the runs give the stimulus blocks of the
recording. Checked on 20260715A/kilosort25/data007-010: the four runs sum to
89,560,000 samples and the last spike is at 89,559,992.

The blocks are shown only when the lengths add up to the recording (the
last spike falls in the final run); otherwise none, rather than a guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

_RANGE = re.compile(r"^data(\d+)-(\d+)$")
_ONE = re.compile(r"^data\d+$")


@dataclass(frozen=True)
class Block:
    run: str
    protocol: str          # short name, e.g. "ChirpStimulus"
    start_s: float
    end_s: float


def runs_in_folder(name: str) -> List[str]:
    """``data007-010`` → data007..data010; ``data000_data004`` → both; ``data022`` → itself."""
    name = name.strip()
    m = _RANGE.match(name)
    if m:
        a, b = m.group(1), m.group(2)
        width = len(a)
        lo, hi = int(a), int(b)
        if hi < lo:
            return []
        return [f"data{i:0{width}d}" for i in range(lo, hi + 1)]
    parts = name.split("_")
    if parts and all(_ONE.match(p) for p in parts):
        return parts
    return []


def short_protocol(label: Optional[str]) -> str:
    """'manookinlab.protocols.ChirpStimulus' → 'ChirpStimulus'; '..._ks' dropped."""
    if not label:
        return "?"
    name = str(label).rsplit(".", 1)[-1]
    return name[:-3] if name.endswith("_ks") else name


def stimulus_blocks(folder_name: str, manifest, last_spike_sample: int,
                    fs: float) -> List[Block]:
    """Blocks in seconds from the recording start, or [] when they cannot be trusted."""
    runs = runs_in_folder(folder_name)
    if not runs or manifest is None or fs <= 0:
        return []
    lengths = []
    for r in runs:
        info = manifest.get(r)
        n = getattr(info, "n_samples", None) if info is not None else None
        if not n:
            return []
        lengths.append((r, int(n), short_protocol(getattr(info, "protocol", None))))
    total = sum(n for _r, n, _p in lengths)
    final_start = total - lengths[-1][1]
    if not (final_start <= int(last_spike_sample) < total):
        return []
    out, t = [], 0
    for r, n, proto in lengths:
        out.append(Block(r, proto, t / fs, (t + n) / fs))
        t += n
    return out


# --- is the cell stable over the recording? ----------------------------------------

LOW_RATE = 0.25        # a block whose median rate is under this share of the cell's: "fires little"
DRIFT = 0.30           # first vs last tenth of the amplitude trace, relative


@dataclass(frozen=True)
class Stability:
    block_rate: dict            # protocol/run -> median rate / whole-recording median rate
    amp_drift: Optional[float]  # (last tenth - first tenth) / median amplitude
    verdict: str                # one line for the plot title


def stability(t, rate, amp=None, blocks: Sequence[Block] = ()) -> Stability:
    import numpy as np
    t = np.asarray(t, float)
    rate = np.asarray(rate, float)
    overall = float(np.nanmedian(rate)) if rate.size else float("nan")
    per_block = {}
    notes = []
    if blocks and overall > 0:
        for b in blocks:
            m = (t >= b.start_s) & (t < b.end_s)
            if m.sum() >= 3:
                rel = float(np.nanmedian(rate[m]) / overall)
                per_block[f"{b.protocol} ({b.run})"] = rel
                if rel < LOW_RATE:
                    notes.append(f"fires little during {b.protocol} ({rel:.0%} of its rate)")
    drift = None
    if amp is not None:
        a = np.asarray(amp, float)
        med = float(np.nanmedian(a)) if a.size else float("nan")
        k = max(1, a.size // 10)
        if a.size >= 10 and np.isfinite(med) and med > 0:
            drift = float((np.nanmean(a[-k:]) - np.nanmean(a[:k])) / med)
            if abs(drift) > DRIFT:
                notes.append(f"amplitude {'falls' if drift < 0 else 'rises'} {abs(drift):.0%} "
                             f"over the recording")
    return Stability(per_block, drift, "; ".join(notes) if notes else "stable")
