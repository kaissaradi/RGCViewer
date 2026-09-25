"""The lab's labelled cells as a training set for suggested classes (PLAN.md Q36).

Scans a folder tree of Vision runs (default ``/mnt/lab/Array-data/sorted``)
for ``.params`` files with named classes, reads each run once, and keeps
the features of its labelled cells (``type_features``). Run medians use the
run's own cells with a real fit, the same way a new unlabelled run is
treated. The result is cached in ``~/.encore/type_library.npz`` with each
file's size and mtime; a later build reads only files that changed.

Kept out, and reported:
* uncertain labels ('?', 'maybe', under 'unclassified');
* types outside TRAIN_CLASSES (too few cells to learn);
* exact duplicate cells (concatenated runs repeat one run's Vision rows);
* runs whose labels contradict the STA polarity for most cells (e.g. a
  run with ON and OFF swapped).

Nothing is written next to the lab data. No model is saved or committed.
"""

from __future__ import annotations

import glob
import logging
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .class_names import canonical_type
from .type_features import CellInput, N_FEATURES, run_features

logger = logging.getLogger(__name__)

DEFAULT_ROOT = "/mnt/lab/Array-data/sorted"
CACHE_PATH = Path.home() / ".encore" / "type_library.npz"
CACHE_VERSION = 1
TRAIN_CLASSES = ("OFF brisk sustained", "OFF brisk transient", "OFF transient",
                 "ON brisk sustained", "ON brisk transient")
MIN_NAMED_CELLS = 10          # a run with fewer named cells adds nothing
MAX_POLARITY_DISAGREE = 0.5   # more than this: the run's ON/OFF labels are suspect


@dataclass
class Library:
    X: np.ndarray                         # (n, N_FEATURES)
    y: np.ndarray                         # class names
    prep: np.ndarray
    run: np.ndarray
    polarity: np.ndarray                  # +1 / -1 from the STA
    notes: List[str] = field(default_factory=list)   # what was left out, in words

    @property
    def n(self) -> int:
        return len(self.y)

    def without_prep(self, prep: Optional[str]) -> "Library":
        """The library minus one prep: never learn from the run being checked."""
        if not prep:
            return self
        keep = self.prep != prep
        return Library(self.X[keep], self.y[keep], self.prep[keep], self.run[keep],
                       self.polarity[keep], self.notes)


def prep_of(path) -> Optional[str]:
    """``<prep>`` of ``.../sorted/<prep>/<sorter>/<run>/...``, or None."""
    parts = Path(path).parts
    if "sorted" in parts:
        i = parts.index("sorted")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def _sta_stixel(params_path: str) -> Optional[float]:
    sta = params_path[:-len(".params")] + ".sta"
    try:
        with open(sta, "rb") as fp:
            fp.read(20)
            stixel, _refresh, _off = struct.unpack(">ddi", fp.read(20))
        return float(stixel)
    except (OSError, struct.error):
        return None


def read_run(params_path: str) -> Optional[dict]:
    """Features + labels of one run, or None when it has no named cells."""
    from .visionloader import ParametersFileReader
    folder, name = os.path.dirname(params_path), os.path.basename(params_path)[:-len(".params")]
    stixel = _sta_stixel(params_path)
    if stixel is None:
        return None
    try:
        reader = ParametersFileReader(folder, name)
    except Exception as exc:                    # damaged or unreadable file
        logger.info("type library: skip %s (%s)", params_path, exc)
        return None
    try:
        col = {c: k for k, c in enumerate(reader.column_names)}
        need = ("classID", "GreenTimeCourse", "Auto", "SigmaX", "SigmaY")
        if any(k not in col for k in need):
            return None
        data = reader.col_row_to_arbitrary_data
        labels, cells = [], []
        for j in range(reader.n_rows):
            tc, auto = data[(col["GreenTimeCourse"], j)], data[(col["Auto"], j)]
            if tc is None or auto is None:
                continue
            ct = canonical_type(data[(col["classID"], j)])
            labels.append(ct)
            cells.append(CellInput(
                tc=np.asarray(tc, float), auto=np.asarray(auto, float),
                acf_binning=float(data.get((col.get("acfBinning", -1), j)) or 0.5),
                sigma_x=float(data[(col["SigmaX"], j)] or np.nan),
                sigma_y=float(data[(col["SigmaY"], j)] or np.nan), stixel=stixel))
    finally:
        reader.close()
    named = [i for i, ct in enumerate(labels) if ct is not None and not ct[1]
             and ct[0] in TRAIN_CLASSES]
    if len(named) < MIN_NAMED_CELLS:
        return None
    X, pol = run_features(cells)
    return {"X": X[named], "y": np.array([labels[i][0] for i in named]),
            "polarity": pol[named]}


def _stamp(path) -> Tuple[int, int]:
    st = os.stat(path)
    return int(st.st_size), int(st.st_mtime_ns)


def _load_cache(path: Path) -> Dict[str, dict]:
    try:
        with np.load(path, allow_pickle=True) as z:
            if int(z["version"]) != CACHE_VERSION:
                return {}
            return z["runs"].item()
    except Exception:
        return {}


def _save_cache(path: Path, runs: Dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, version=CACHE_VERSION, runs=np.array(runs, dtype=object))
    os.replace(tmp, path)


def build_library(root: str = DEFAULT_ROOT, cache_path: Optional[Path] = None,
                  progress: Optional[Callable[[int, int], None]] = None,
                  exclude_runs: Tuple[str, ...] = ()) -> Library:
    """Scan ``root`` (sequentially: the lab share is slow in parallel)."""
    cache_path = Path(cache_path) if cache_path is not None else CACHE_PATH
    files = sorted(glob.glob(os.path.join(root, "*", "*", "data*", "*.params")))
    cached = _load_cache(cache_path)
    runs: Dict[str, dict] = {}
    t0 = time.time()
    for i, f in enumerate(files):
        if progress is not None:
            progress(i, len(files))
        try:
            stamp = _stamp(f)
        except OSError:
            continue
        hit = cached.get(f)
        if hit is not None and tuple(hit["stamp"]) == stamp:
            runs[f] = hit
            continue
        entry = read_run(f)
        runs[f] = {"stamp": stamp, "data": entry}
    _save_cache(cache_path, runs)
    logger.info("type library: %d files in %.0f s", len(files), time.time() - t0)
    return assemble(runs, exclude_runs)


def assemble(runs: Dict[str, dict], exclude_runs=()) -> Library:
    X, y, prep, run, pol, notes = [], [], [], [], [], []
    seen = set()
    n_dup = 0
    for f, entry in sorted(runs.items()):
        d = entry.get("data")
        if d is None or f in exclude_runs:
            continue
        label_pol = np.where(np.char.startswith(d["y"].astype(str), "ON"), 1, -1)
        disagree = float(np.mean(label_pol != d["polarity"]))
        if disagree > MAX_POLARITY_DISAGREE:
            notes.append(f"{f}: {disagree:.0%} of its labelled cells have the opposite ON/OFF "
                         f"of their STA — left out (ON/OFF labels probably swapped).")
            continue
        p = f.split(os.sep)[-4] if len(f.split(os.sep)) >= 4 else f
        for i in range(len(d["y"])):
            key = (p, np.round(d["X"][i], 6).tobytes())
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            X.append(d["X"][i]); y.append(d["y"][i]); prep.append(p); run.append(f)
            pol.append(d["polarity"][i])
    if n_dup:
        notes.append(f"{n_dup} duplicate cells left out (runs that repeat another run's rows).")
    if not X:
        return Library(np.zeros((0, N_FEATURES)), np.array([]), np.array([]), np.array([]),
                       np.array([]), notes)
    return Library(np.vstack(X), np.array(y), np.array(prep), np.array(run), np.array(pol), notes)


# --- the type atlas (PLAN.md Q44) ---------------------------------------------------

@dataclass
class TypeProfile:
    name: str
    n_cells: int
    n_preps: int
    tc_mean: np.ndarray        # run-scaled time course (type_features.S_GRID)
    tc_sd: np.ndarray
    acg_mean: np.ndarray       # 20 log bins (type_features.ACG_EDGES)
    acg_sd: np.ndarray


def atlas_profiles(lib: Library) -> Dict[str, TypeProfile]:
    """Mean and spread of each class's time course and ACG across the library."""
    from .type_features import S_GRID
    n_tc = len(S_GRID)
    out = {}
    for name in sorted(set(lib.y)):
        m = lib.y == name
        X = lib.X[m]
        out[name] = TypeProfile(
            name=name, n_cells=int(m.sum()), n_preps=len(set(lib.prep[m])),
            tc_mean=X[:, :n_tc].mean(axis=0), tc_sd=X[:, :n_tc].std(axis=0),
            acg_mean=X[:, n_tc:n_tc + 20].mean(axis=0), acg_sd=X[:, n_tc:n_tc + 20].std(axis=0))
    return out
