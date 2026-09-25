"""Compare DS grating runs in one tissue frame (PLAN.md Q51).

A preferred direction from the Grating tab is a direction on the screen. The
piece lies on the array at a different angle in every prep, so screen
directions from two preps cannot be pooled. Three things fix that, each
measured from the run's own files:

- DS cells: the run's ``*GratingDSOS.npy`` through ``grating_calc``, the same
  shuffle test and thresholds as the Grating tab (Q33).
- The optic-disc direction in array coordinates: the run's own ``.ei``,
  through ``axon_bearing`` (Q43).
- The array → screen turn: a white-noise run of the same prep, from its RF
  centres (``.params``) against its somas (``.ei``), through
  ``vision_sort_check`` (Q32). A run with its own ``.params`` uses those.

With the disc direction and the turn, each preferred direction becomes an
angle from the direction to the optic disc, which means the same thing in
every prep. It does not say which side is nasal: that needs the eye and the
piece's place in it.

Reads are sequential (the lab share is CIFS). One small summary per run is
kept next to the grating file's Kilosort output (``ksfiles/ds_pool_summary.json``),
stamped with the sizes and times of the files it came from.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA = 2
# A turn is used only from a pairing this good (robust R², vision_sort_check).
# Measured 2026-09-25 on 50 white-noise runs of the grating preps: every
# pairing at ≥ 0.43 gave its prep's usual turn; every one ≤ 0.34 was noise or
# another sort (a mirror on 20251212A/data018 at 0.33, identity at 0.001).
TURN_MIN_R2 = 0.4
CACHE_NAME = "ds_pool_summary.json"
TEMP_SUFFIX = ".encore-tmp"          # as DataManager.TEMP_SUFFIX, so its sweep finds leftovers
GRATING_GLOB = "*GratingDSOS*.npy"
FRAMES = ("screen", "disc")


# --------------------------------------------------------------------- stimulus frame

# A grating labelled θ (the ``orientation`` in the file, and the Grating tab's
# angle) moves its bars toward θ + 180°. From the lab's protocol and Stage:
# GratingDSOS.m passes θ to Stage unchanged (lines 120, 219); Stage's Grating
# rotates it anticlockwise by θ on a y-up canvas (Grating.m performDraw,
# Canvas.m orthographic(0, w, 0, h)) and samples s = phase/360 + n(x'+1)/2
# (Grating.m updateVertexBuffer), so as the phase grows each crest moves to
# -x'. The white-noise frame is drawn row 1 at the top (TextureObject.m
# flipdim, Image.m) and Vision's y0 counts up from the bottom, so the canvas
# and the RF-fit frame have the same orientation. Checked 2026-09-25 from
# the source in ~/Downloads (GratingDSOS.m, stage-master); a physical check
# on the rig display (θ = 0 should drift left) is still to do.
MOTION_OFFSET_DEG = 180.0


def motion_deg(theta_deg) -> np.ndarray:
    """Direction the bars moved, degrees anticlockwise from screen right (y up)."""
    return (np.asarray(theta_deg, float) + MOTION_OFFSET_DEG) % 360


def motion_vectors(theta_deg) -> np.ndarray:
    """(N, 2) unit vectors of bar motion in the Vision RF-fit frame (x right, y up)."""
    th = np.radians(motion_deg(theta_deg))
    return np.column_stack([np.cos(th), np.sin(th)])


# --------------------------------------------------------------------- DS cells

@dataclass
class RunDS:
    run: str                                  # "<prep>/<sorter>/<run>"
    n_cells: int = 0
    has_directions: bool = False              # False: the protocol varied bar width only
    ids: List[int] = field(default_factory=list)          # Vision IDs of the DS cells
    theta_deg: np.ndarray = field(default_factory=lambda: np.zeros(0))   # the Grating tab's angle
    dsi: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bearing_verdict: str = "none"             # axon_bearing verdict of the run's own .ei
    bearing_deg: float = float("nan")         # array frame
    bearing_ci: Tuple[float, float] = (float("nan"), float("nan"))
    n_axons: int = 0                          # axons with a clean fit
    turn: Optional[Tuple[int, int, int, int]] = None   # screen (RF-fit frame) ≈ turn · array
    turn_source: str = ""                     # which run's .params gave the turn, and its R²
    note: str = ""

    @property
    def n_ds(self) -> int:
        return len(self.theta_deg)

    @property
    def prep(self) -> str:
        return self.run.split("/")[0]

    def lobes(self) -> Tuple[float, float]:
        """(axis°, strength) of a four-lobe pattern: 4-fold circular mean; axis in [0, 90)."""
        return four_fold_axis(self.theta_deg)

    @property
    def can_align(self) -> bool:
        return (self.bearing_verdict in ("disc", "direction") and self.turn is not None
                and np.isfinite(self.bearing_deg))

    def angles(self, frame: str) -> Optional[np.ndarray]:
        """Preferred motion directions in ``frame`` (see MOTION_OFFSET_DEG).

        'screen': degrees anticlockwise from screen right. 'disc': degrees
        anticlockwise (in the array's own frame, as the EI panel draws it
        unturned) from the direction to the optic disc, so 0° = toward the
        disc. None when the run has no disc direction or no turn.
        """
        if frame == "screen":
            return motion_deg(self.theta_deg)
        if frame != "disc":
            raise ValueError(f"unknown frame {frame!r}")
        if not self.can_align:
            return None
        v_fit = motion_vectors(self.theta_deg)
        m = np.array(self.turn, float).reshape(2, 2)
        v_array = v_fit @ m                    # Mᵀ · v for each row: M is orthogonal
        return (np.degrees(np.arctan2(v_array[:, 1], v_array[:, 0])) - self.bearing_deg) % 360

    def to_json(self) -> dict:
        return {"run": self.run, "n_cells": self.n_cells, "has_directions": self.has_directions,
                "ids": [int(i) for i in self.ids],
                "theta_deg": [float(x) for x in self.theta_deg],
                "dsi": [float(x) for x in self.dsi],
                "bearing_verdict": self.bearing_verdict, "bearing_deg": _num(self.bearing_deg),
                "bearing_ci": [_num(x) for x in self.bearing_ci], "n_axons": self.n_axons,
                "turn": list(self.turn) if self.turn else None, "turn_source": self.turn_source,
                "note": self.note}

    @classmethod
    def from_json(cls, d: dict) -> "RunDS":
        return cls(run=d["run"], n_cells=int(d["n_cells"]), has_directions=bool(d["has_directions"]),
                   ids=[int(i) for i in d["ids"]], theta_deg=np.array(d["theta_deg"], float),
                   dsi=np.array(d["dsi"], float), bearing_verdict=d["bearing_verdict"],
                   bearing_deg=_nan(d["bearing_deg"]),
                   bearing_ci=tuple(_nan(x) for x in d["bearing_ci"]), n_axons=int(d["n_axons"]),
                   turn=tuple(int(v) for v in d["turn"]) if d.get("turn") else None,
                   turn_source=d.get("turn_source", ""), note=d.get("note", ""))


def _num(x):
    return None if x is None or not np.isfinite(x) else float(x)


def _nan(x):
    return float("nan") if x is None else float(x)


def four_fold_axis(theta_deg) -> Tuple[float, float]:
    th = np.radians(np.asarray(theta_deg, float))
    if th.size == 0:
        return float("nan"), 0.0
    m = np.exp(4j * th).mean()
    return float((np.degrees(np.angle(m)) / 4) % 90), float(abs(m))


def ds_cells(grating: dict):
    """(ids, theta°, DSI, n_cells, has_directions) for one grating file's dict.

    ids are the file's own keys: Vision IDs (CLAUDE.md trap 1). DataManager's
    grating cache is not reused: it is keyed by Kilosort ID, or by Vision ID
    on a Vision-only open, and the file does not say which.
    """
    from . import grating_calc as gc
    ids, theta, dsi = [], [], []
    if "trial_parameters" in grating and "spike_times_by_trial" in grating:
        tp = grating["trial_parameters"]
        groups = gc.group_grating_conditions(tp)
        has_dirs = any(g["condition_type"] == "dsos" for g in groups)
        cells = list(grating["spike_times_by_trial"])
        if has_dirs:
            for cid in cells:
                e = gc.compute_grating_response(cid, grating["spike_times_by_trial"], tp)
                _take(gc, cid, e, ids, theta, dsi)
    else:
        cells = [k for k in grating if isinstance(k, (int, np.integer))]
        has_dirs = any(isinstance(v, dict) and v.get("condition_type") == "dsos"
                       for c in cells for v in grating[c].values() if isinstance(v, dict))
        for cid in cells:
            _take(gc, cid, grating[cid], ids, theta, dsi)
    return ids, np.array(theta, float), np.array(dsi, float), len(cells), has_dirs


def _take(gc, cid, entry, ids, theta, dsi):
    if entry is None:
        return
    s = gc.select_best_dsos_condition(entry)
    if s and s["classification"] == "DS" and np.isfinite(s["preferred_direction_deg"]):
        ids.append(int(cid))
        theta.append(float(s["preferred_direction_deg"]))
        dsi.append(float(s["DSI"]))


# --------------------------------------------------------------------- somas and the turn

def soma_positions(cells: Dict[int, dict], pos) -> Dict[int, Tuple[float, float]]:
    """{id: (x, y)} µm: amplitude-weighted centre of each EI's three largest electrodes."""
    pos = np.asarray(pos, float)
    out = {}
    for cid, c in cells.items():
        a = np.asarray(c["amin"], float)
        top = np.argsort(a)[-3:]
        w = np.clip(a[top], 0, None)
        if w.sum() > 0:
            out[int(cid)] = tuple((pos[top] * w[:, None]).sum(0) / w.sum())
    return out


def rf_centres(folder: Path, dataset: str) -> Dict[int, Tuple[float, float]]:
    from . import visionloader as vl
    table = vl.VisionCellDataTable()
    with vl.ParametersFileReader(str(folder), dataset) as r:
        r.update_visioncelldata_obj(table)
    out = {}
    for vid in table.get_cell_ids():
        try:
            out[int(vid)] = (float(table.get_data_for_cell(vid, "x0")),
                             float(table.get_data_for_cell(vid, "y0")))
        except (KeyError, TypeError, ValueError):
            continue
    return out


_TURN_MEMO: Dict[tuple, tuple] = {}     # (folder, stamp) -> (turn, R²): two grating runs share a sibling


def turn_from(folder: Path, dataset: str, somas=None):
    """(turn, R²) from one run's .params against its .ei somas; turn None when not trusted."""
    from . import axon_bearing as ab, vision_sort_check as vsc
    folder = Path(folder)
    key = (str(folder), dataset, tuple(map(tuple, _stamp([folder / f"{dataset}.params",
                                                          folder / f"{dataset}.ei"]))))
    if somas is None and key in _TURN_MEMO:
        return _TURN_MEMO[key]
    if somas is None:
        cells, pos = ab.read_run_eis(folder, dataset)
        somas = soma_positions(cells, pos)
    check = vsc.check_pairing(rf_centres(folder, dataset), somas)
    good = check.screen_turn is not None and check.r2_robust >= TURN_MIN_R2
    _TURN_MEMO[key] = ((check.screen_turn if good else None), check.r2_robust)
    return _TURN_MEMO[key]


def _run_number(name: str) -> int:
    m = re.search(r"data(\d+)", name)
    return int(m.group(1)) if m else 10 ** 6


def sibling_runs(run_dir: Path) -> List[Path]:
    """White-noise candidates for the turn: runs of the same sort with .params and .ei, nearest first."""
    me = _run_number(run_dir.name)
    out = []
    for d in sorted(p for p in run_dir.parent.iterdir() if p.is_dir() and p != run_dir):
        if (d / f"{d.name}.params").is_file() and (d / f"{d.name}.ei").is_file():
            out.append(d)
    return sorted(out, key=lambda d: (abs(_run_number(d.name) - me), d.name))


# --------------------------------------------------------------------- one run, cached

def _stamp(paths) -> list:
    out = []
    for p in paths:
        try:
            st = os.stat(p)
            out.append([str(p), st.st_size, int(st.st_mtime)])
        except OSError:
            out.append([str(p), None, None])
    return out


def _read_cache(path: Path, stamp) -> Optional[RunDS]:
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    if d.get("schema") != SCHEMA or d.get("stamp") != stamp:
        return None
    try:
        return RunDS.from_json(d["run"])
    except (KeyError, TypeError, ValueError):
        return None


def _write_cache(path: Path, stamp, run: RunDS) -> None:
    if not path.parent.is_dir():
        return
    try:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=TEMP_SUFFIX)
        with os.fdopen(fd, "w") as f:
            json.dump({"schema": SCHEMA, "stamp": stamp, "run": run.to_json()}, f)
        os.replace(tmp, path)
    except OSError:
        logger.info("could not write %s", path, exc_info=True)
        try:
            os.remove(tmp)
        except (OSError, UnboundLocalError):
            pass


def summarize_run(grating_npy: Path, root: Path, progress: Optional[Callable[[str], None]] = None,
                  use_cache: bool = True) -> RunDS:
    """DS cells, disc direction and turn for one grating run, from its cache when fresh."""
    from . import axon_bearing as ab
    grating_npy = Path(grating_npy)
    run_dir = grating_npy.parent
    ei = run_dir / f"{run_dir.name}.ei"
    params = run_dir / f"{run_dir.name}.params"
    siblings = sibling_runs(run_dir)[:3]
    stamp = _stamp([grating_npy, ei, params] + [s / f"{s.name}.params" for s in siblings])
    cache_path = run_dir / "ksfiles" / CACHE_NAME
    if use_cache:
        hit = _read_cache(cache_path, stamp)
        if hit is not None:
            return hit
    say = progress or (lambda s: None)
    rel = str(run_dir.relative_to(root)) if root in run_dir.parents else run_dir.name
    run = RunDS(run=rel)

    say("DS cells")
    grating = np.load(grating_npy, allow_pickle=True).item()
    run.ids, run.theta_deg, run.dsi, run.n_cells, run.has_directions = ds_cells(grating)
    del grating
    if not run.has_directions:
        run.note = "one direction only: a bar-width sweep"
        _write_cache(cache_path, stamp, run)
        return run

    somas = None
    if ei.is_file():
        say("axons")
        try:
            cells, pos = ab.read_run_eis(run_dir, run_dir.name)
        except Exception as exc:              # e.g. an .ei whose array id Vision cannot place
            logger.info("unreadable %s", ei, exc_info=True)
            run.note = f"the .ei could not be read ({exc})"
        else:
            b = ab.estimate(cells, pos)
            run.bearing_verdict, run.bearing_deg = b.verdict, float(b.bearing_deg)
            run.bearing_ci, run.n_axons = tuple(float(x) for x in b.bearing_ci), int(b.n_good)
            somas = soma_positions(cells, pos)
            del cells
    else:
        run.note = "no .ei"

    say("array turn")
    if params.is_file() and somas is not None:
        try:
            turn, r2 = turn_from(run_dir, run_dir.name, somas)
        except Exception:
            logger.info("no turn from %s", run_dir, exc_info=True)
            turn = None
        if turn is not None:
            run.turn, run.turn_source = turn, f"{run_dir.name} · R² {r2:.2f}"
    for sib in siblings:
        if run.turn is not None:
            break
        try:
            turn, r2 = turn_from(sib, sib.name)
        except Exception:
            logger.info("no turn from %s", sib, exc_info=True)
            continue
        if turn is not None:
            run.turn, run.turn_source = turn, f"{sib.name} · R² {r2:.2f}"
    _write_cache(cache_path, stamp, run)
    return run


def find_grating_runs(root: Path, prep: Optional[str] = None) -> List[Path]:
    """Grating files under ``root/<prep>/<sorter>/<run>/``, one per run (analysed file first)."""
    root = Path(root)
    pattern = f"{prep}/*/*/{GRATING_GLOB}" if prep else f"*/*/*/{GRATING_GLOB}"
    by_run: Dict[Path, Path] = {}
    for p in sorted(root.glob(pattern)):
        best = by_run.get(p.parent)
        if best is None or _priority(p) < _priority(best):
            by_run[p.parent] = p
    return [by_run[k] for k in sorted(by_run)]


def _priority(path: Path) -> int:
    name = path.name.lower()
    return 0 if ("_combined" in name or "analyzed" in name) else (2 if "raw" in name else 1)


def prefer_sorter(paths: List[Path], sorter: str) -> List[Path]:
    """One grating file per (prep, run): the same recording sorted twice counts once.

    Keeps ``sorter``'s copy when there is one, else the last sorter by name.
    """
    by_key: Dict[Tuple[str, str], Path] = {}
    for p in paths:
        run_dir = Path(p).parent
        key = (run_dir.parent.parent.name, run_dir.name)
        old = by_key.get(key)
        if old is None:
            by_key[key] = p
            continue
        mine, theirs = run_dir.parent.name, old.parent.parent.name
        if theirs != sorter and (mine == sorter or mine > theirs):
            by_key[key] = p
    return sorted(by_key.values())
