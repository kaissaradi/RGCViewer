"""Pick the runs for File ▸ Match Runs from this experiment (PLAN.md Q52, Q62).

Match Runs matches cells to other runs of the same retina by their EIs,
then borrows those runs' responses (chirp, gratings, RFs) for cells this run
lacks. This lists the runs of the open run's prep that have an ``.ei``, with
the stimuli each one holds, and ticks the ones that fill a gap in the open
run. Any number can be ticked; a cell takes each response from the first
ticked run (top to bottom) that has it.

Two folder layouts are walked, one directory listing per folder (the share
is CIFS): ``<prep>/<sorter>/<run>/<run>.ei`` (Array-data) and
``<prep>/<run>/[<run>-map|<sorter>]/<name>.ei`` (older Chichilnisky-lab
analyses, whose grating trials are in ``<prep>/stimuli/sNN``).
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QHeaderView,
                            QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from ...analysis import lisp_stimulus

# The patterns DataManager._ANALYSIS_GLOBS uses to find stimulus files.
STIMULUS_GLOBS = {"chirp": ("*Chirp*.npy",), "grating": ("*Grating*.npy", "*DSOS*.npy"),
                  "contrast": ("*ontrast*.npy",)}
# What a matched run can fill in (ReferenceBridge: STAs/RFs, gratings, chirp).
GAP_ORDER = ("white noise", "grating", "chirp")


@dataclass
class RunInfo:
    path: Path                     # the folder holding the .ei
    dataset: str                   # the .ei's stem (Vision dataset name)
    label: str                     # path relative to the prep, for the table
    stimuli: List[str] = field(default_factory=list)   # "white noise", "chirp", "grating", …
    is_current: bool = False
    lisp_file: Optional[str] = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def sorter(self) -> str:
        return self.path.parent.name


def current_run_dir(dm):
    """The loaded run's folder (the one with its Vision files), or None."""
    p = getattr(dm, "vision_params_path", None)
    if p:
        return Path(str(p)).parent
    k = getattr(dm, "kilosort_dir", None)
    if not k:
        return None
    k = Path(str(k))
    return k.parent if k.name == "ksfiles" else k


_PREP = re.compile(r"^(\d{8}[A-Z]?|\d{4}-\d{2}-\d{2}-\d+)$")


def prep_dir_for(run_dir) -> Optional[Path]:
    """The prep folder above a run: 20260220A or 2012-10-15-0 by name, else two levels up."""
    run_dir = Path(run_dir)
    for anc in [run_dir, *list(run_dir.parents)[:4]]:
        if _PREP.match(anc.name):
            return anc
    return run_dir.parents[1] if len(run_dir.parents) >= 2 else None


def _listdir(d: Path) -> List[str]:
    try:
        return os.listdir(d)
    except OSError:
        return []


def _run_at(d: Path, names: Sequence[str], prep_dir: Path, seq_names, current) -> Optional[RunInfo]:
    eis = sorted(n for n in names if n.endswith(".ei") and not n.startswith("."))
    if not eis:
        return None
    stem = eis[0][:-3]
    stimuli = []
    if f"{stem}.sta" in names:
        from ...analysis.vision_integration import sta_is_empty
        # One 36-byte read: a run recorded without a stimulus has an empty
        # (NaN) STA, and matching to it borrows nothing (20260514A/data000).
        stimuli.append("STA empty" if sta_is_empty(d / f"{stem}.sta") else "white noise")
    for kind, globs in STIMULUS_GLOBS.items():
        if any(fnmatch.fnmatch(n, g) for n in names for g in globs):
            stimuli.append(kind)
    lisp = None
    if "grating" not in stimuli:
        lisp = next((n for n in lisp_stimulus.sequence_file_names(stem) if n.lower() in seq_names), None)
        if lisp:
            stimuli.append(f"grating (Lisp {lisp})")
    try:
        label = str(d.relative_to(prep_dir))
    except ValueError:
        label = d.name
    is_current = current is not None and d.resolve() == current
    return RunInfo(d, stem, label, stimuli, is_current, lisp)


def list_runs(prep_dir: Path, current: Optional[Path] = None) -> List[RunInfo]:
    """Folders one or two levels under ``prep_dir`` that hold a Vision ``.ei``."""
    prep_dir = Path(prep_dir)
    current = Path(current).resolve() if current else None
    seq_names = set()
    for d in ("stimuli", "Visual"):
        seq_names |= {n.lower() for n in _listdir(prep_dir / d)}
    out = []
    try:
        level1 = sorted(d for d in prep_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
    except OSError:
        return out
    for d1 in level1:
        if d1.name in ("stimuli", "Visual"):
            continue
        names = _listdir(d1)
        run = _run_at(d1, names, prep_dir, seq_names, current)
        if run is not None:
            out.append(run)
        for n in sorted(names):
            d2 = d1 / n
            if n.startswith(".") or "." in n or not d2.is_dir():
                continue
            run = _run_at(d2, _listdir(d2), prep_dir, seq_names, current)
            if run is not None:
                out.append(run)
    return out


def _has(run: RunInfo, want: str) -> bool:
    return any(s == want or s.startswith(want + " ") for s in run.stimuli)


def preferred_indices(runs: List[RunInfo], current_stimuli) -> List[int]:
    """Runs to tick: for each stimulus the open run lacks, the nearest run that has it.

    Nearest: a folder next to the open run first (the same sort; in the old
    layout ``data002/data000-map`` is data000 mapped with data002's sort),
    then the run recorded closest to it (cells drift over an experiment, so
    the EIs match best nearby), then the shortest path.
    """
    cur = next((r for r in runs if r.is_current), None)

    def run_number(r):
        m = re.search(r"data(\d{3})", r.dataset or r.name)
        return int(m.group(1)) if m else None

    here = run_number(cur) if cur is not None else None

    def rank(i):
        r = runs[i]
        sibling = cur is not None and r.path.parent == cur.path.parent
        n = run_number(r)
        gap = abs(n - here) if n is not None and here is not None else 10_000
        return (0 if sibling else 1, gap, r.label.count("/"), len(r.label), r.label)

    order = sorted((i for i, r in enumerate(runs) if not r.is_current), key=rank)
    picks = []
    for want in GAP_ORDER:
        if want in current_stimuli:
            continue
        i = next((i for i in order if _has(runs[i], want)), None)
        if i is not None and i not in picks:
            picks.append(i)
    if not picks and order:
        picks.append(order[0])
    return sorted(picks)


def preferred_index(runs: List[RunInfo], current_stimuli) -> int:
    """The first run preferred_indices would tick, or -1."""
    picks = preferred_indices(runs, current_stimuli)
    return picks[0] if picks else -1


class RunPicker(QDialog):
    def __init__(self, parent, runs: List[RunInfo], prep: str, preselect, start_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Match Runs")
        self.resize(700, 480)
        self.runs, self.chosen, self._start_dir = list(runs), [], start_dir
        if isinstance(preselect, int):
            preselect = [preselect] if preselect >= 0 else []
        layout = QVBoxLayout(self)
        intro = QLabel(
            f"Tick the runs of {prep} to match to this one. Encore matches the cells by their "
            "electrical images; a cell that has no chirp, grating or receptive field here then "
            "shows its matched cell's, with a note saying which run it came from. With several "
            "runs ticked, each response comes from the first run (top down) that has it.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Run", "Has"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        for i, r in enumerate(self.runs):
            self._add_row(r, i in preselect)
        self.table.itemChanged.connect(lambda _item: self._update_button())
        layout.addWidget(self.table, 1)
        if not self.runs:
            layout.insertWidget(1, QLabel("No other run of this prep has an .ei file."))
        self.rematch = QCheckBox("Match again, even where a saved match exists")
        self.rematch.setToolTip("Saved matches (_mapping_to_*.json next to this run) are reused otherwise.")
        layout.addWidget(self.rematch)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.match_btn = buttons.addButton("Match", QDialogButtonBox.ButtonRole.AcceptRole)
        other = QPushButton("Add another folder…")
        other.setToolTip("A run outside this list (another prep, or a folder without the usual layout)")
        buttons.addButton(other, QDialogButtonBox.ButtonRole.ActionRole)
        other.clicked.connect(self._pick_folder)
        buttons.accepted.connect(self._accept_checked)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._update_button()

    def _add_row(self, r: RunInfo, checked: bool):
        row = self.table.rowCount()
        self.table.insertRow(row)
        name = QTableWidgetItem(r.label + ("  (open now)" if r.is_current else ""))
        has = QTableWidgetItem(", ".join(r.stimuli) or "—")
        for col, item in enumerate((name, has)):
            item.setToolTip(str(r.path))
            self.table.setItem(row, col, item)
        # Items are user-checkable by default; only a run you may match gets a box.
        for item in (name, has):
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
        if r.is_current:
            for item in (name, has):
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
        else:
            name.setFlags(name.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            name.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)

    def checked_rows(self) -> List[int]:
        out = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None and item.flags() & Qt.ItemFlag.ItemIsUserCheckable \
                    and item.checkState() == Qt.CheckState.Checked:
                out.append(row)
        return out

    def set_checked(self, rows):
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None and item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(Qt.CheckState.Checked if row in rows else Qt.CheckState.Unchecked)

    def _update_button(self):
        n = len(self.checked_rows())
        self.match_btn.setEnabled(n > 0)
        self.match_btn.setText("Match" if n <= 1 else f"Match {n} runs")

    def _accept_checked(self):
        rows = self.checked_rows()
        if not rows:
            return
        self.chosen = [str(self.runs[r].path) for r in rows]
        self.accept()

    def _pick_folder(self):
        d = QFileDialog.getExistingDirectory(
            self, "A run's Vision folder (with its .ei)", self._start_dir)
        if not d:
            return
        names = _listdir(Path(d))
        run = _run_at(Path(d), names, Path(d).parent, set(), None) or \
            RunInfo(Path(d), Path(d).name, Path(d).name, [])
        self.runs.append(run)
        self._add_row(run, True)
        self._update_button()


def pick_runs(main_window, run_dir: Optional[Path], current_stimuli, start_dir="",
              already: Sequence[str] = ()) -> Tuple[List[str], bool]:
    """(run folders the user ticked, match again?) or ([], False) on cancel."""
    prep_dir = prep_dir_for(run_dir) if run_dir is not None else None
    if prep_dir is None:
        d = QFileDialog.getExistingDirectory(
            main_window, "A run's Vision folder (with its .ei)", start_dir)
        return ([d] if d else []), False
    from qtpy.QtWidgets import QApplication
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        runs = list_runs(prep_dir, run_dir)
    finally:
        QApplication.restoreOverrideCursor()
    held = {str(Path(p).resolve()) for p in already}
    pre = [i for i, r in enumerate(runs) if str(r.path.resolve()) in held] or \
        preferred_indices(runs, current_stimuli)
    dlg = RunPicker(main_window, runs, prep_dir.name, pre, start_dir)
    if not dlg.exec():
        return [], False
    return dlg.chosen, dlg.rematch.isChecked()
