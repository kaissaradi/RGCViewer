"""Array ▸ Compare DS Runs: preferred directions of every DS grating run (PLAN.md Q51).

One row per grating run: its DS cells, the four-lobe axis, the optic-disc
direction from its own EIs and the array → screen turn (``ds_pool``). The
figure pools the ticked runs and shows each run on its own, either as the
bars moved on the screen or as an angle from the direction to the optic
disc, which is the same in every prep. Runs are read one at a time in the
background (the share is CIFS) and kept in a small summary per run, so the
second look is instant.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (QCheckBox, QComboBox, QDialog, QHBoxLayout, QHeaderView, QLabel,
                            QMessageBox, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
                            QVBoxLayout, QWidget)

from ...analysis import ds_pool

logger = logging.getLogger(__name__)

N_BINS = 24
MAX_SMALL = 24                     # small multiples drawn; the table lists every run
FRAME_LABELS = {"screen": "As the bars moved on the screen",
                "disc": "From the direction to the optic disc"}
COLUMNS = ["Pool", "Run", "DS cells", "Four lobes", "Optic disc", "Turn from"]


def current_run_dir(dm):
    """The loaded run's folder (<prep>/<sorter>/<run>), or None."""
    p = getattr(dm, "vision_params_path", None)
    if p:
        return Path(str(p)).parent
    k = getattr(dm, "kilosort_dir", None)
    if not k:
        return None
    k = Path(str(k))
    return k.parent if k.name == "ksfiles" else k


def one_run_per_prep(runs):
    """{run: pooled?}: in each prep only the run with the most DS cells.

    Two runs of one piece record the same cells; pooling both counts them twice.
    """
    best = {}
    for r in runs:
        if r.n_ds and (r.prep not in best or r.n_ds > best[r.prep].n_ds):
            best[r.prep] = r
    return {r.run: best.get(r.prep) is r for r in runs}


def _disc_text(r) -> str:
    if r.bearing_verdict == "none" or not np.isfinite(r.bearing_deg):
        return f"none ({r.n_axons} axons)" if r.has_directions else ""
    lo, hi = r.bearing_ci
    ci = f" ± {abs(hi - lo) / 2:.0f}°" if np.isfinite(lo) and np.isfinite(hi) else ""
    word = "point" if r.bearing_verdict == "disc" else "direction"
    return f"{word} {r.bearing_deg % 360:.0f}°{ci}"


class DSCompareDialog(QDialog):
    def __init__(self, main_window, root: Path, prep: str, current_run: str):
        super().__init__(main_window)
        self.setWindowTitle("Compare DS runs")
        self.resize(1180, 760)
        self.main_window = main_window
        self.root, self.prep, self.current_run = Path(root), prep, current_run
        self.runs = []                        # RunDS, in table order
        self.pooled = {}                      # run -> bool
        self._state = None                    # background scan state
        from ..theme import resolve_theme_colors
        self.c = resolve_theme_colors(main_window.get_current_colors())

        layout = QVBoxLayout(self)
        self.intro = QLabel(
            "Each DS cell's preferred direction, from the same test as the Grating tab. Angles "
            "are the way the bars moved: the Grating tab's angle + 180° (the lab's protocol "
            "drifts a grating labelled θ toward θ + 180°). “From the optic disc” turns each "
            "run by its array turn and its axons' direction, so 0° means toward the disc in "
            "every prep; for a dorsal piece that is roughly ventral. It cannot tell nasal from "
            "temporal: that needs which eye.")
        self.intro.setWordWrap(True)
        layout.addWidget(self.intro)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Runs:"))
        self.scope_combo = QComboBox()
        self.scope_combo.addItem(f"This prep ({prep})", "prep")
        self.scope_combo.addItem("Every prep on the share", "all")
        self.scope_combo.setToolTip("Every prep reads each run once (about 10–40 s each); "
                                    "later looks use the saved summaries.")
        self.scope_combo.activated.connect(self._on_scope_picked)
        controls.addWidget(self.scope_combo)
        controls.addSpacing(16)
        controls.addWidget(QLabel("Directions:"))
        self.frame_combo = QComboBox()
        for key, text in FRAME_LABELS.items():
            self.frame_combo.addItem(text, key)
        self.frame_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.frame_combo.activated.connect(lambda _i: (self._fill_table(), self.redraw()))
        controls.addWidget(self.frame_combo)
        # The lab cuts its pieces from dorsal retina (user, 2026-09-25: "we
        # always dissect from the dorsal side", the side with few UV cones;
        # S opsin is repressed in dorsal mouse retina, Applebury et al. 2000,
        # PMID 11055434). From a dorsal piece the optic disc lies ventral, so
        # "toward the disc" is roughly ventral: exact only on the vertical
        # meridian, off by the piece's angle from it elsewhere.
        self.dorsal_box = QCheckBox("Pieces are dorsal")
        self.dorsal_box.setChecked(True)
        self.dorsal_box.setToolTip(
            "The lab cuts dorsal pieces (the side with few UV cones). Then the optic disc lies\n"
            "ventral of the piece, so “to disc” ≈ ventral, up to the piece's angle from the\n"
            "vertical meridian. Nasal vs temporal still needs which eye.")
        self.dorsal_box.toggled.connect(lambda _on: self.redraw())
        controls.addWidget(self.dorsal_box)
        self.weight_box = QCheckBox("Weight by DSI")
        self.weight_box.toggled.connect(lambda _on: self.redraw())
        controls.addWidget(self.weight_box)
        controls.addStretch()
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setToolTip("Stop after the run being read")
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setVisible(False)
        controls.addWidget(self.stop_btn)
        layout.addLayout(controls)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemChanged.connect(self._on_item_changed)
        split.addWidget(self.table)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self.fig = Figure(figsize=(7, 6), facecolor=self.c["bg_panel"])
        self.canvas = FigureCanvasQTAgg(self.fig)
        holder = QWidget()
        hl = QVBoxLayout(holder)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(self.canvas)
        split.addWidget(holder)
        split.setSizes([620, 660])
        layout.addWidget(split, 1)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.timer = QTimer(self)
        self.timer.setInterval(400)
        self.timer.timeout.connect(self._poll)
        self.scan()

    # ----------------------------------------------------------------- scanning

    def scan(self):
        self.stop()
        scope = self.scope_combo.currentData()
        state = {"paths": None, "done": [], "i": 0, "stage": "finding grating runs",
                 "stop": threading.Event(), "error": None, "finished": False}
        self._state = state
        self.runs, self.pooled = [], {}
        self.table.setRowCount(0)
        root, prep = self.root, (self.prep if scope == "prep" else None)
        sorter = Path(self.current_run).parent.name

        def work():
            try:
                paths = ds_pool.prefer_sorter(ds_pool.find_grating_runs(root, prep), sorter)
                state["paths"] = paths
                for i, p in enumerate(paths):
                    if state["stop"].is_set():
                        break
                    state["i"], state["stage"] = i, str(p.parent.relative_to(root))
                    try:
                        state["done"].append(ds_pool.summarize_run(p, root))
                    except Exception as exc:          # one bad run must not end the scan
                        logger.warning("DS summary failed for %s", p, exc_info=True)
                        state["done"].append(ds_pool.RunDS(str(p.parent.relative_to(root)),
                                                           note=f"could not read: {exc}"))
            except Exception as exc:
                state["error"] = exc
            finally:
                state["finished"] = True

        threading.Thread(target=work, name="ds-compare", daemon=True).start()
        self.stop_btn.setVisible(True)
        self.timer.start()
        self._poll()

    def _on_scope_picked(self, _index):
        # Screen directions of two preps do not line up (each piece lies at
        # its own angle), so a pool across preps is shown from the optic disc.
        if self.scope_combo.currentData() == "all":
            self.frame_combo.setCurrentIndex(self.frame_combo.findData("disc"))
        self.scan()

    def stop(self):
        if self._state is not None:
            self._state["stop"].set()

    def _poll(self):
        st = self._state
        if st is None:
            return
        new = st["done"][len(self.runs):]
        if new:
            self.runs.extend(new)
            self.pooled = one_run_per_prep(self.runs)
            self._fill_table()
            self.redraw()
        n = len(st["paths"]) if st["paths"] is not None else "?"
        if st["finished"]:
            self.timer.stop()
            self.stop_btn.setVisible(False)
            if st["error"] is not None:
                self.status.setText(f"Could not list the grating runs: {st['error']}")
            elif not self.runs:
                self.status.setText("No grating runs found here.")
            else:
                self.status.setText(self._summary_text())
        else:
            self.status.setText(f"Reading {st['stage']} ({len(self.runs)}/{n})…  "
                                "Each run is read once and remembered.")

    def closeEvent(self, event):
        self.stop()
        self.timer.stop()
        super().closeEvent(event)

    def reject(self):
        self.stop()
        super().reject()

    # ----------------------------------------------------------------- table

    def _fill_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.runs))
        for row, r in enumerate(self.runs):
            pool = QTableWidgetItem()
            if r.n_ds:
                pool.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                pool.setCheckState(Qt.CheckState.Checked if self.pooled.get(r.run)
                                   else Qt.CheckState.Unchecked)
            else:
                pool.setFlags(Qt.ItemFlag.ItemIsEnabled)
            frame = self.frame_combo.currentData()
            angles = r.angles(frame) if r.n_ds >= 8 else None
            if angles is not None:
                axis, strength = ds_pool.four_fold_axis(angles)
                lobes = f"{axis:.0f}° ({strength:.2f})"
            else:
                lobes = ""
            ds = f"{r.n_ds} / {r.n_cells}" if r.has_directions else (r.note or "no DS test")
            name = f"{r.prep} {Path(r.run).name}"
            cells = [pool, QTableWidgetItem(name + ("  ◀ open" if r.run == self.current_run else "")),
                     QTableWidgetItem(ds), QTableWidgetItem(lobes),
                     QTableWidgetItem(_disc_text(r)), QTableWidgetItem(r.turn_source or
                                                                      ("" if not r.has_directions
                                                                       else "none found"))]
            tips = ["Pooled in the left-most rose. One run per prep by default: two runs of one "
                    "piece record the same cells.", r.run + (f" · {r.note}" if r.note else ""),
                    "Cells classed DS by the Grating tab's test / cells in the file",
                    "Axis of a four-lobe pattern (0–90°, in the chosen directions) and how "
                    "strong it is (0 = none, 1 = four sharp lobes)",
                    "Where this run's axons point, in array coordinates (Array ▸ Find the Optic Disc)",
                    "The white-noise run whose RF centres against its somas give the array → "
                    "screen turn"]
            for col, (item, tip) in enumerate(zip(cells, tips)):
                item.setToolTip(tip)
                self.table.setItem(row, col, item)
        self.table.blockSignals(False)

    def _on_item_changed(self, item):
        if item.column() != 0:
            return
        r = self.runs[item.row()]
        self.pooled[r.run] = item.checkState() == Qt.CheckState.Checked
        self.redraw()

    def _summary_text(self) -> str:
        with_ds = [r for r in self.runs if r.n_ds]
        aligned = [r for r in with_ds if r.can_align]
        return (f"{len(self.runs)} grating runs, {len(with_ds)} with DS cells, "
                f"{len(aligned)} of those with both an optic-disc direction and an array turn. "
                "The DS test and thresholds are the Grating tab's.")

    # ----------------------------------------------------------------- figure

    def redraw(self):
        frame = self.frame_combo.currentData()
        c = self.c
        self.fig.clear()
        chosen = [r for r in self.runs if r.n_ds and self.pooled.get(r.run)]
        shown = [r for r in self.runs if r.n_ds][:MAX_SMALL]
        if frame == "disc":
            chosen = [r for r in chosen if r.can_align]
            shown = [r for r in shown if r.can_align]
        n = 1 + len(shown)
        cols = int(np.ceil(np.sqrt(n * 1.3)))
        rows = int(np.ceil(n / cols))
        gs = self.fig.add_gridspec(rows, cols, hspace=0.55, wspace=0.35)
        weight = self.weight_box.isChecked()
        pooled_angles = [r.angles(frame) for r in chosen]
        pooled_w = [r.dsi for r in chosen]
        self._rose(self.fig.add_subplot(gs[0, 0], projection="polar"),
                   np.concatenate(pooled_angles) if pooled_angles else np.zeros(0),
                   np.concatenate(pooled_w) if pooled_w and weight else None, frame,
                   f"Pooled: {sum(len(a) for a in pooled_angles)} cells, {len(chosen)} "
                   f"run{'' if len(chosen) == 1 else 's'}",
                   bold=True)
        for k, r in enumerate(shown, start=1):
            ax = self.fig.add_subplot(gs[k // cols, k % cols], projection="polar")
            self._rose(ax, r.angles(frame), r.dsi if weight else None, frame,
                       f"{r.prep} {Path(r.run).name}\n{r.n_ds} DS", dim=not self.pooled.get(r.run))
            if frame == "screen" and r.can_align:
                self._disc_marker(ax, r)
        if frame == "disc" and not shown:
            self.fig.text(0.5, 0.5, "No run has both an optic-disc direction and an array turn yet.",
                          ha="center", va="center", color=c["text_secondary"])
        self.canvas.draw_idle()

    def _rose(self, ax, angles, weights, frame, title, bold=False, dim=False):
        c = self.c
        ax.set_facecolor(c["bg_panel"])
        if frame == "disc":
            ax.set_theta_zero_location("N")           # toward the disc at the top
        edges = np.linspace(0, 2 * np.pi, N_BINS + 1)
        counts = np.histogram(np.radians(angles % 360), bins=edges,
                              weights=weights)[0] if len(angles) else np.zeros(N_BINS)
        ink = c.get("plot_highlight", "#4A72B8")
        ax.bar(edges[:-1], counts, width=2 * np.pi / N_BINS, align="edge",
               color=ink, alpha=0.35 if dim else 0.85, edgecolor=c["bg_panel"], linewidth=0.5)
        ax.set_yticks([])
        ax.set_xticks(np.radians([0, 90, 180, 270]))
        if frame == "disc":
            names = (["ventral (to disc)", "90°", "dorsal", "270°"] if self.dorsal_box.isChecked()
                     else ["to disc", "90°", "180°", "270°"])
            ax.set_xticklabels(names if bold else ["", "", "", ""])
        else:
            ax.set_xticklabels(["0°", "90°", "180°", "270°"] if bold else ["", "", "", ""])
        ax.tick_params(colors=c["text_secondary"], labelsize=8, pad=0)
        ax.grid(color=c["border_subtle"], linewidth=0.5)
        ax.spines["polar"].set_edgecolor(c["border_subtle"])
        ax.set_title(title, color=c["text_primary"] if bold else c["text_secondary"],
                     fontsize=9 if bold else 7, fontweight="bold" if bold else "normal",
                     pad=14 if bold else 4)

    def _disc_marker(self, ax, r):
        """On the screen frame: where this run's optic disc lies, as a thin line."""
        m = np.array(r.turn, float).reshape(2, 2)
        b = np.radians(r.bearing_deg)
        v = m @ np.array([np.cos(b), np.sin(b)])
        th = np.arctan2(v[1], v[0])
        top = ax.get_ylim()[1] or 1.0
        ax.plot([th, th], [0, top], color=self.c.get("plot_peak", "#E0B000"), linewidth=1.2)


def compare_ds_runs(main_window):
    dm = getattr(main_window, "data_manager", None)
    run_dir = current_run_dir(dm) if dm is not None else None
    if run_dir is None or len(run_dir.parents) < 3:
        QMessageBox.information(main_window, "Compare DS runs",
                                "Open a run first: the comparison starts from its prep.")
        return
    root = run_dir.parents[2]
    prep = run_dir.parents[1].name
    rel = f"{prep}/{run_dir.parent.name}/{run_dir.name}"
    DSCompareDialog(main_window, root, prep, rel).exec()
