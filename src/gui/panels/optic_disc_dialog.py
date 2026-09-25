"""Array ▸ Find the Optic Disc: where the axons point (PLAN.md Q43).

Reads the run's EIs once in the background, fits each cell's axon, and asks
whether the good axons converge (``axon_bearing``). The figure is drawn the
way the EI panel draws the array (turned to the screen when that is on,
Q20), so "up" means the same in both. The result is kept on the
DataManager (``axon_bearing``) for pooling DS runs later.
"""

from __future__ import annotations

import logging

import numpy as np
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QDialog, QDialogButtonBox, QLabel, QMessageBox, QVBoxLayout

from .. import array_orientation

logger = logging.getLogger(__name__)


def _screen_words(vec) -> str:
    """'up and to the right' for a display-frame vector (y up)."""
    ang = np.degrees(np.arctan2(vec[1], vec[0])) % 360
    names = ["to the right", "up and to the right", "up", "up and to the left",
             "to the left", "down and to the left", "down", "down and to the right"]
    return names[int(((ang + 22.5) % 360) // 45)]


class OpticDiscDialog(QDialog):
    def __init__(self, main_window, bearing, positions, matrix):
        super().__init__(main_window)
        self.setWindowTitle("Optic disc direction")
        self.resize(760, 640)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from ..theme import resolve_theme_colors
        c = resolve_theme_colors(main_window.get_current_colors())
        layout = QVBoxLayout(self)
        text = bearing.sentence()
        M = np.array(matrix, float).reshape(2, 2)
        if bearing.verdict in ("disc", "direction"):
            th = np.radians(bearing.bearing_deg)
            v = M @ np.array([np.cos(th), np.sin(th)])
            where = "the array as the EI panel draws it" if tuple(matrix) != array_orientation.IDENTITY \
                else "the array's own frame"
            text += f" In {where}, the disc is {_screen_words(v)}."
        text += ("\n\nThis gives the optic-disc direction only. Dorsal vs ventral also needs "
                 "where the piece came from in the eye.")
        label = QLabel(text)
        label.setWordWrap(True)
        layout.addWidget(label)
        fig = Figure(figsize=(7, 5.4), facecolor=c["bg_panel"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(c["bg_panel"])
        pos = array_orientation.to_display(np.asarray(positions, float), matrix)
        ax.scatter(pos[:, 0], pos[:, 1], s=4, color=c["border_default"], zorder=1)
        ink = c.get("plot_highlight", "#4A72B8")
        for cell in bearing.cells:
            p = M @ np.array(cell["soma_xy"])
            u = M @ np.array(cell["u"])
            ax.annotate("", xy=p + 180 * u, xytext=p,
                        arrowprops=dict(arrowstyle="->", color=ink, lw=1.2), zorder=3)
            ax.plot(*p, "o", color=c["text_primary"], ms=2.5, zorder=4)
        if bearing.disc_xy is not None:
            d = M @ np.array(bearing.disc_xy)
            ax.plot(*d, marker="*", ms=18, color=c.get("plot_peak", "#E0B000"), zorder=5,
                    markeredgecolor=c["text_primary"])
            ax.annotate("optic disc", d, xytext=(8, 8), textcoords="offset points",
                        color=c["text_primary"], fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlabel("µm", color=c["text_secondary"])
        ax.tick_params(colors=c.get("plot_tick", c["text_secondary"]), labelsize=8)
        for s in ax.spines.values():
            s.set_edgecolor(c["border_subtle"])
        ax.set_title(f"{bearing.n_good} axons with a clean fit (of {bearing.n_axons} found in "
                     f"{bearing.n_cells} cells) · median speed {bearing.speed_m_s:.2f} m/s",
                     color=c["text_secondary"], fontsize=9)
        fig.tight_layout()
        layout.addWidget(FigureCanvasQTAgg(fig), 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def find_optic_disc(main_window):
    from ...analysis import axon_bearing as ab
    from .. import callbacks
    dm = getattr(main_window, "data_manager", None)
    path = getattr(dm, "vision_params_path", None) if dm is not None else None
    ei_path = None
    if path is not None:
        from pathlib import Path
        p = Path(str(path))
        ei_path = p.with_suffix(".ei")
    if ei_path is None or not ei_path.exists():
        QMessageBox.information(main_window, "Optic disc direction",
                                "This needs the run's Vision .ei file (electrical images).")
        return
    state = {"i": 0, "n": 0, "stage": "reading the EIs"}
    timer = QTimer(main_window)
    timer.setInterval(400)
    timer.timeout.connect(lambda: main_window.status_bar.showMessage(
        f"Optic disc: {state['stage']} ({state['i']}/{state['n'] or '?'})…"))
    timer.start()
    generation = getattr(dm, "generation", None)

    def work():
        cells, pos = ab.read_run_eis(ei_path.parent, ei_path.stem,
                                     progress=lambda i, n: state.update(i=i, n=n))
        state["stage"] = "fitting"
        return ab.estimate(cells, pos, progress=lambda s: state.update(stage=s)), pos

    def done(result):
        timer.stop()
        if getattr(main_window.data_manager, "generation", None) != generation:
            return
        bearing, pos = result
        main_window.data_manager.axon_bearing = bearing
        main_window.status_bar.showMessage(bearing.sentence(), 12000)
        OpticDiscDialog(main_window, bearing, pos,
                        array_orientation.display_matrix(main_window.data_manager)).exec()

    def failed(exc):
        timer.stop()
        logger.warning("optic disc estimate failed", exc_info=exc)
        QMessageBox.warning(main_window, "Optic disc direction", f"Could not estimate it: {exc}")

    callbacks._run_in_background(main_window, work, done, failed)
