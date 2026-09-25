"""Pick the reference run for File ▸ Map Reference Run from this experiment (PLAN.md Q52).

Map Reference Run matches cells to another run of the same retina by their
EIs, then borrows that run's responses (chirp, gratings, RFs) for cells this
run lacks. The user used to find the folder in a file dialog. This lists the
runs of the open run's prep (every sorter) that have an ``.ei``, with the
stimuli each one holds, and preselects one that fills a gap in the open run.
One directory listing per run: the share is CIFS.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QDialog, QDialogButtonBox, QFileDialog, QHeaderView, QLabel,
                            QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

# The patterns DataManager._ANALYSIS_GLOBS uses to find stimulus files.
STIMULUS_GLOBS = {"chirp": ("*Chirp*.npy",), "grating": ("*Grating*.npy", "*DSOS*.npy"),
                  "contrast": ("*ontrast*.npy",)}


@dataclass
class RunInfo:
    path: Path
    sorter: str
    stimuli: List[str] = field(default_factory=list)   # "white noise", "chirp", "grating", "contrast"
    is_current: bool = False

    @property
    def name(self) -> str:
        return self.path.name


def list_runs(prep_dir: Path, current: Optional[Path] = None) -> List[RunInfo]:
    """Runs under ``prep_dir/<sorter>/<run>/`` that have ``<run>.ei``."""
    out = []
    current = Path(current).resolve() if current else None
    try:
        sorters = sorted(d for d in Path(prep_dir).iterdir() if d.is_dir())
    except OSError:
        return out
    for sorter in sorters:
        try:
            runs = sorted(d for d in sorter.iterdir() if d.is_dir())
        except OSError:
            continue
        for run in runs:
            try:
                names = os.listdir(run)
            except OSError:
                continue
            if f"{run.name}.ei" not in names:
                continue
            stimuli = []
            if f"{run.name}.sta" in names:
                stimuli.append("white noise")
            for kind, globs in STIMULUS_GLOBS.items():
                if any(fnmatch.fnmatch(n, g) for n in names for g in globs):
                    stimuli.append(kind)
            out.append(RunInfo(run, sorter.name, stimuli,
                               is_current=current is not None and run.resolve() == current))
    return out


def preferred_index(runs: List[RunInfo], current_stimuli) -> int:
    """The run to preselect: one with a stimulus the open run lacks (grating, then chirp)."""
    for want in ("grating", "chirp", "contrast"):
        if want in current_stimuli:
            continue
        for i, r in enumerate(runs):
            if not r.is_current and want in r.stimuli:
                return i
    for i, r in enumerate(runs):
        if not r.is_current:
            return i
    return -1


class RunPicker(QDialog):
    def __init__(self, parent, runs: List[RunInfo], prep: str, preselect: int, start_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Map Reference Run")
        self.resize(640, 460)
        self.runs, self.chosen, self._start_dir = runs, None, start_dir
        layout = QVBoxLayout(self)
        intro = QLabel(
            f"Pick another run of {prep}. Encore matches its cells to this run's by their "
            "electrical images; a cell with no chirp or grating here then shows its matched "
            "cell's, with a note saying so.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.table = QTableWidget(len(runs), 3)
        self.table.setHorizontalHeaderLabels(["Run", "Sort", "Has"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        for row, r in enumerate(runs):
            name = QTableWidgetItem(r.name + ("  (open now)" if r.is_current else ""))
            items = [name, QTableWidgetItem(r.sorter), QTableWidgetItem(", ".join(r.stimuli) or "—")]
            for col, item in enumerate(items):
                item.setToolTip(str(r.path))
                if r.is_current:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable & ~Qt.ItemFlag.ItemIsEnabled)
                self.table.setItem(row, col, item)
        if preselect >= 0:
            self.table.selectRow(preselect)
        self.table.doubleClicked.connect(lambda _i: self._accept_selected())
        layout.addWidget(self.table, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.match_btn = buttons.addButton("Match cells", QDialogButtonBox.ButtonRole.AcceptRole)
        other = QPushButton("Other folder…")
        other.setToolTip("A run outside this list (another prep, or a folder without the usual layout)")
        buttons.addButton(other, QDialogButtonBox.ButtonRole.ActionRole)
        other.clicked.connect(self._pick_folder)
        buttons.accepted.connect(self._accept_selected)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        if not runs:
            layout.insertWidget(1, QLabel("No other run of this prep has an .ei file."))

    def _accept_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        r = self.runs[rows[0].row()]
        if r.is_current:
            return
        self.chosen = str(r.path)
        self.accept()

    def _pick_folder(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Reference Run (Vision Analysis Directory)", self._start_dir)
        if d:
            self.chosen = d
            self.accept()


def pick_reference_run(main_window, run_dir: Optional[Path], current_stimuli, start_dir="") -> Optional[str]:
    """The reference run folder the user picks, or None. Falls back to a folder dialog."""
    if run_dir is None or len(Path(run_dir).parents) < 2:
        d = QFileDialog.getExistingDirectory(
            main_window, "Select Reference Run (Vision Analysis Directory)", start_dir)
        return d or None
    prep_dir = Path(run_dir).parents[1]
    runs = list_runs(prep_dir, run_dir)
    dlg = RunPicker(main_window, runs, prep_dir.name, preferred_index(runs, current_stimuli), start_dir)
    return dlg.chosen if dlg.exec() else None
