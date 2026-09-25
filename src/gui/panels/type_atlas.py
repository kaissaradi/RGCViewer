"""Type atlas: what each named type looks like across the lab (PLAN.md Q44).

One row per type the suggester knows: its time course and autocorrelation
as the mean ± 1 SD over every classified cell in the lab library (all
preps), with this run's cells in that folder drawn on top. Time is in the
run's own time-to-peak (runs differ in frame duration, PLAN.md Q36). The
one-line signature comes from docs/design/rgc_types.md.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from qtpy.QtWidgets import QDialog, QDialogButtonBox, QLabel, QScrollArea, QVBoxLayout, QWidget

from ...analysis.type_features import ACG_EDGES, S_GRID
from ..theme import apply_plot_theme, resolve_theme_colors

# docs/design/rgc_types.md (literature checked 2026-09-25; species in brackets).
SIGNATURE = {
    "ON brisk sustained": "Weakly biphasic time course, highest firing rate, RF larger than "
                          "OFF brisk sustained [rat; Ravi et al. 2018]. Likely the sustained ON alpha.",
    "OFF brisk sustained": "Weakly biphasic, but more than ON; longer integration than ON brisk "
                           "sustained [rat; Ravi 2018]. Likely the sustained OFF alpha.",
    "ON brisk transient": "Large RF, brief integration, rectified [rat; Ravi 2018]. "
                          "Possibly the transient ON alpha (weak evidence).",
    "OFF brisk transient": "Largest RF of the OFF brisk types, brief integration; bursty ACG "
                           "[mouse tOFFα; van Wyk et al. 2009].",
    "OFF transient": "Not one of Ravi's six types; may be a mixture of OFF-transient types "
                     "[Goetz et al. 2022] — check its mosaic.",
}
MAX_RUN_TRACES = 40


class TypeAtlas(QDialog):
    def __init__(self, parent, profiles, run_rows=None):
        """``profiles``: type_library.atlas_profiles; ``run_rows``: {class: [feature rows]}."""
        super().__init__(parent)
        self.setWindowTitle("Type atlas — the lab's named types")
        self.resize(980, 820)
        c = resolve_theme_colors(parent.get_current_colors() if parent else None)
        outer = QVBoxLayout(self)
        intro = QLabel(
            "Shaded: mean ± 1 SD over every cell the lab classified as that type (all preps). "
            "Red lines: this run's cells in that folder. Time runs from the spike (0) back to "
            "2.5 time-to-peaks; the ACG is spike share per log-spaced lag bin.")
        intro.setWordWrap(True)
        outer.addWidget(intro)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        col = QVBoxLayout(body)
        run_rows = run_rows or {}
        centres = np.sqrt(ACG_EDGES[:-1] * ACG_EDGES[1:])
        self.plots = {}
        for name, prof in profiles.items():
            head = QLabel(f"<b>{name}</b> — {prof.n_cells} cells from {prof.n_preps} preps. "
                          f"{SIGNATURE.get(name, '')}")
            head.setWordWrap(True)
            col.addWidget(head)
            glw = pg.GraphicsLayoutWidget()
            glw.setMinimumHeight(170)
            apply_plot_theme(glw, c)
            tc_plot = glw.addPlot(title="time course")
            acg_plot = glw.addPlot(title="autocorrelation")
            for plot, x, mean, sd in ((tc_plot, S_GRID, prof.tc_mean, prof.tc_sd),
                                      (acg_plot, np.log10(centres), prof.acg_mean, prof.acg_sd)):
                apply_plot_theme(plot, c)
                band_colour = pg.mkColor(c.get("plot_ensemble", "#4A72B8"))
                band_colour.setAlpha(70)
                lo = pg.PlotCurveItem(x, mean - sd, pen=None)
                hi = pg.PlotCurveItem(x, mean + sd, pen=None)
                plot.addItem(pg.FillBetweenItem(lo, hi, brush=pg.mkBrush(band_colour)))
                plot.plot(x, mean, pen=pg.mkPen(c["plot_mean"], width=2))
            rows = run_rows.get(name, [])[:MAX_RUN_TRACES]
            red = pg.mkPen(c.get("plot_compare", "#c0392b"), width=1)
            for r in rows:
                tc_plot.plot(S_GRID, r[:len(S_GRID)], pen=red)
                acg_plot.plot(np.log10(centres), r[len(S_GRID):len(S_GRID) + 20], pen=red)
            tc_plot.setLabel("bottom", "time before the spike (× time-to-peak)")
            acg_plot.setLabel("bottom", "lag, log10 ms")
            col.addWidget(glw)
            self.plots[name] = (tc_plot, acg_plot, len(rows))
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)


def open_atlas(main_window):
    """Show the atlas; builds the lab library in the background the first time."""
    from ...analysis import type_library as tl
    from .. import suggestions
    st = getattr(main_window, "_suggestions", None)
    if st is None or st.library is None:
        main_window.status_bar.showMessage(
            "The atlas needs the lab library: running Suggest classes first…", 6000)
        main_window._open_atlas_after_suggest = True
        suggestions.start(main_window)
        return
    classes = suggestions.folder_classes(main_window)
    run_rows = {}
    for cid, row in st.run_features.items():
        cls = classes.get(cid)
        if cls:
            run_rows.setdefault(cls, []).append(row)
    TypeAtlas(main_window, tl.atlas_profiles(st.library), run_rows).exec()
