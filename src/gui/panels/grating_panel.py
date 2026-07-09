import logging
import math

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QThread, Qt
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from ..theme import resolve_theme_colors
from ..workers.workers import GratingComputeWorker
from ...analysis import grating_calc

logger = logging.getLogger(__name__)

pg.setConfigOptions(antialias=True)

# Fixed categorical palette for overlaying multiple (barWidth, TF) conditions
# at once. The "best" condition (highest |DSI|) is drawn at full opacity in
# its assigned color; every other condition is drawn as a low-alpha "shadow"
# trace in the same color family, so you can see whether tuning is
# consistent across conditions without clicking through a dropdown.
_CONDITION_COLORS = [
    "#4DABF7",
    "#FF6B6B",
    "#69DB7C",
    "#FFD43B",
    "#B197FC",
    "#FF922B",
    "#66D9E8",
]
_SHADOW_ALPHA = 70  # 0-255, applied to non-best condition traces
_BEST_ALPHA = 255


class GratingPanel(QWidget):
    """
    Displays direction/orientation tuning for the selected cluster: a polar
    plot and a linear bar-chart histogram, both overlaying every (barWidth,
    temporalFrequency) 'dsos' condition at once — the strongest-DSI
    condition drawn bold, the rest as faint shadow traces — plus a
    firing-rate sanity strip at the best condition's preferred direction,
    and an aggregate bar-width/SF tuning curve when present.

    Unlike ChirpPanel, grating data is NOT always precomputed offline. If
    DataManager.grating_status == 'raw_only', DSI/OSI for a not-yet-seen
    cluster is computed on demand via GratingComputeWorker (spawned here,
    not a persistent queue — see workers.py). Panels never reference other
    panels (AGENTS.md §2) — reads DataManager only.
    """

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        colors = resolve_theme_colors(self.main_window.get_current_colors())

        # cluster_id -> (QThread, GratingComputeWorker) for in-flight computes,
        # so we never spawn a second worker for the same cluster while one
        # is already running, and can tell a stale result from a live one.
        self._pending_workers = {}
        self._current_cluster_id = None
        self._current_data = None

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedLayout()
        outer_layout.addLayout(self.stack)

        # ---------------------------------------------------------
        # Page 0: data view
        # ---------------------------------------------------------
        data_page = QWidget()
        data_layout = QVBoxLayout(data_page)
        data_layout.setContentsMargins(4, 4, 4, 4)

        header = QHBoxLayout()
        title = QLabel(
            f"<span style='color:{colors['text_tertiary']}; font-size:10px; "
            f"letter-spacing:0.06em;'>DIRECTION TUNING</span>"
        )
        header.addWidget(title)
        header.addStretch()
        data_layout.addLayout(header)

        # Condition legend — replaces the old dropdown. All conditions are
        # rendered simultaneously (see _CONDITION_COLORS), this just labels
        # which color is which (barWidth, TF) pair and which one is "best".
        self.legend_label = QLabel("")
        self.legend_label.setWordWrap(True)
        data_layout.addWidget(self.legend_label)

        stats_row = QHBoxLayout()
        self.stats_label = QLabel("")
        stats_row.addWidget(self.stats_label)
        stats_row.addStretch()
        data_layout.addLayout(stats_row)

        # --- Polar plot + linear histogram, side by side ---
        plots_row = QHBoxLayout()

        self.polar_plot = pg.PlotWidget()
        self.polar_plot.setAspectLocked(True)
        self.polar_plot.hideAxis("bottom")
        self.polar_plot.hideAxis("left")
        self._style_plot(self.polar_plot, colors)
        self._polar_grid_items = []  # persistent ring items, rebuilt per cluster
        self._polar_curves = []  # one PlotCurveItem per condition, rebuilt per cluster
        self._polar_pref_line = pg.PlotCurveItem(
            pen=pg.mkPen(colors.get("plot_compare", "r"), width=2, style=Qt.DashLine)
        )
        self.polar_plot.addItem(self._polar_pref_line)
        plots_row.addWidget(self.polar_plot, stretch=1)

        self.hist_plot = pg.PlotWidget()
        self.hist_plot.setLabel("bottom", "Direction (deg)")
        self.hist_plot.setLabel("left", "Response")
        self._style_plot(self.hist_plot, colors)
        self._hist_bar_item = (
            None  # BarGraphItem for the best condition, rebuilt per cluster
        )
        self._hist_shadow_curves = (
            []
        )  # PlotCurveItems for non-best conditions, rebuilt per cluster
        plots_row.addWidget(self.hist_plot, stretch=1)

        data_layout.addLayout(plots_row, stretch=3)

        # Per-direction PSTH grid — replaces the old single-trace "firing
        # rate @ preferred direction" strip. That strip only ever showed
        # ONE direction and threw away psth_by_direction for every other
        # direction, which was already being computed. This shows all of
        # them at once, arranged compass-style (direction angle -> grid
        # position) so the spatial layout itself communicates the tuning
        # shape, not just a single scalar-adjacent curve. Shared y-axis
        # across all subplots is required — without it a flat noisy
        # direction can look as "big" as a real response, which is the
        # same amplitude-blind illusion that caused the DSI selection bug.
        psth_label = QLabel(
            f"<span style='color:{colors['text_tertiary']}; font-size:9px;'>"
            f"PER-DIRECTION PSTH (best condition)</span>"
        )
        data_layout.addWidget(psth_label)
        self._data_layout = (
            data_layout  # kept so _clear_psth_grid can swap the widget in place
        )
        self.psth_grid_widget = pg.GraphicsLayoutWidget()
        self.psth_grid_widget.setBackground(colors["bg_panel"])
        self._psth_grid_plots = {}  # direction_deg -> PlotItem, rebuilt per cluster
        data_layout.addWidget(self.psth_grid_widget, stretch=2)

        # Bar-width / SF tuning curve — always visible when present,
        # independent of the dsos overlay (it's an aggregate view across
        # all 'sf' conditions, not tied to one).
        self.sf_label = QLabel(
            f"<span style='color:{colors['text_tertiary']}; font-size:9px;'>"
            f"BAR-WIDTH / SF TUNING</span>"
        )
        self.sf_label.setVisible(False)
        data_layout.addWidget(self.sf_label)
        self.sf_plot = pg.PlotWidget()
        self.sf_plot.setLabel("bottom", "Bar width")
        self.sf_plot.setLabel("left", "Response")
        self._style_plot(self.sf_plot, colors)
        self._sf_curve = self.sf_plot.plot(
            [],
            [],
            pen=pg.mkPen(colors.get("plot_overlay", "c"), width=2),
            symbol="o",
            symbolSize=6,
        )
        self.sf_plot.setVisible(False)
        data_layout.addWidget(self.sf_plot, stretch=1)

        self.stack.addWidget(data_page)

        # ---------------------------------------------------------
        # Page 1: placeholder (missing / computing / no-response-for-cell)
        # ---------------------------------------------------------
        placeholder_page = QWidget()
        placeholder_layout = QVBoxLayout(placeholder_page)
        self.placeholder_label = QLabel("No grating data")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        placeholder_layout.addWidget(self.placeholder_label)
        self.stack.addWidget(placeholder_page)

        self.stack.setCurrentIndex(1)

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------
    def _style_plot(self, plot_widget, colors=None):
        if colors is None:
            colors = resolve_theme_colors(self.main_window.get_current_colors())
        else:
            colors = resolve_theme_colors(colors)
        plot_item = plot_widget.getPlotItem()
        plot_widget.setBackground(colors["bg_panel"])
        plot_item.getAxis("bottom").setPen(pg.mkPen(colors["border_default"]))
        plot_item.getAxis("left").setPen(pg.mkPen(colors["border_default"]))
        plot_item.getAxis("bottom").setTextPen(pg.mkPen(colors["text_secondary"]))
        plot_item.getAxis("left").setTextPen(pg.mkPen(colors["text_secondary"]))
        plot_item.showAxis("top", False)
        plot_item.showAxis("right", False)
        plot_item.showGrid(x=True, y=True, alpha=0.08)
        plot_item.setContentsMargins(8, 8, 8, 8)

    def restyle_plots(self, colors):
        colors = resolve_theme_colors(colors)
        self._style_plot(self.polar_plot, colors)
        self._style_plot(self.hist_plot, colors)
        self._style_plot(self.sf_plot, colors)
        self.psth_grid_widget.setBackground(colors["bg_panel"])
        self._polar_pref_line.setPen(
            pg.mkPen(colors.get("plot_compare", "r"), width=2, style=Qt.DashLine)
        )
        self._sf_curve.setPen(pg.mkPen(colors.get("plot_overlay", "c"), width=2))
        if self._current_cluster_id is not None:
            self.update_all(self._current_cluster_id)

    # ------------------------------------------------------------------
    # Main entry point — Tier 2 only, same rule as ChirpPanel.
    # ------------------------------------------------------------------
    def update_all(self, cluster_id):
        if cluster_id is None:
            return
        cluster_id = int(cluster_id)
        self._current_cluster_id = cluster_id

        dm = self.main_window.data_manager
        if dm is None:
            return

        if (
            not getattr(dm, "grating_available", False)
            or dm.grating_status == "missing"
        ):
            self._show_placeholder("No grating data")
            return

        data = dm.get_grating_data_for_cluster(cluster_id)

        if data is None:
            if dm.grating_status == "raw_only":
                self._ensure_computing(cluster_id)
            else:
                self._show_placeholder(f"No grating response for cluster {cluster_id}")
            return

        self._render_cluster_data(cluster_id, data)

    def _show_placeholder(self, text):
        self.stack.setCurrentIndex(1)
        self.placeholder_label.setText(text)

    # ------------------------------------------------------------------
    # On-demand compute lifecycle
    # ------------------------------------------------------------------
    def _ensure_computing(self, cluster_id):
        self._show_placeholder(f"Computing DSI/OSI for cluster {cluster_id}...")

        if cluster_id in self._pending_workers:
            return  # already in flight, just wait for its finished signal

        dm = self.main_window.data_manager
        thread = QThread()
        worker = GratingComputeWorker(dm, cluster_id)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_grating_computed)
        worker.finished.connect(thread.quit)
        # IMPORTANT: only drop the Python reference to `thread` (and delete
        # it) once thread.finished fires — i.e. once the QThread has
        # actually stopped running, not merely once the worker's business
        # logic (worker.finished) has completed. thread.quit() only
        # *requests* the thread's event loop stop; popping/deleting the
        # QThread wrapper before isRunning() goes False is what causes
        # "QThread: Destroyed while thread is still running" and can abort
        # the process, as opposed to just leaking a warning.
        thread.finished.connect(
            lambda cid=cluster_id: self._pending_workers.pop(cid, None)
        )
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._pending_workers[cluster_id] = (thread, worker)
        thread.start()

    def _on_grating_computed(self, cluster_id, success, message):
        # Note: do NOT pop _pending_workers here. That happens on
        # thread.finished (see _ensure_computing) so we never delete a
        # QThread wrapper while the thread itself is still winding down.

        # Stale-result guard: only repaint if the user is still looking at
        # this cluster (they may have clicked to a different one while the
        # permutation test was running).
        if self._current_cluster_id != cluster_id:
            return

        if not success:
            self._show_placeholder(f"No grating response for cluster {cluster_id}")
            return

        self.update_all(cluster_id)

    def cleanup(self):
        """Stop any in-flight compute threads before the panel is destroyed."""
        for thread, worker in list(self._pending_workers.values()):
            thread.quit()
            thread.wait(2000)
        self._pending_workers.clear()
        for plot in (self.polar_plot, self.hist_plot, self.sf_plot):
            plot.clear()
            if hasattr(plot, "close"):
                plot.close()
        # Not calling .clear() here — see _clear_psth_grid for why
        # GraphicsLayout.clear()/removeItem() are unreliable with this
        # grid's reuse pattern. At teardown we're destroying the whole
        # widget anyway, so just close it directly; nothing needs to
        # survive this call.
        self.psth_grid_widget.close()

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render_cluster_data(self, cluster_id, data):
        self.stack.setCurrentIndex(0)
        self._current_data = data

        dsos_conditions = sorted(
            (
                k
                for k in data
                if isinstance(k, tuple) and data[k].get("condition_type") == "dsos"
            ),
        )

        # SF tuning curve (aggregate, independent of the dsos overlay).
        if "sf_tuning_curve" in data and "sf_bar_widths" in data:
            bar_widths = np.asarray(data["sf_bar_widths"], dtype=float)
            curve = np.asarray(data["sf_tuning_curve"], dtype=float)
            self._sf_curve.setData(bar_widths, curve)
            self.sf_plot.setVisible(True)
            self.sf_label.setVisible(True)
            self.sf_plot.enableAutoRange()
        else:
            self.sf_plot.setVisible(False)
            self.sf_label.setVisible(False)

        # "Best" condition + DS/OS classification comes from the shared
        # gated selector in grating_calc.py — see select_best_dsos_condition
        # for why raw max(|DSI|) was wrong for CLASSIFICATION (amplitude-
        # blind, significance-blind). But the gate should only decide
        # whether we call a cell DS/OS — it should never hide the actual
        # data. Even when nothing passes the gate, every condition still
        # gets drawn (overlay, histogram, PSTH grid) so the person can look
        # at the real response and judge for themselves; only the stats
        # label reflects "not significant."
        selection = grating_calc.select_best_dsos_condition(data)

        if selection is None:
            # No dsos conditions at all — genuinely nothing to plot.
            self.stats_label.setText(
                "No direction-tuning conditions found for this cluster."
            )
            self.legend_label.setText("")
            self._clear_polar()
            self._clear_hist()
            self._clear_psth_grid()
            return

        if selection["condition"] is not None:
            # Something passed the gate — use it, and its own classification.
            display_cond = selection["condition"]
            classification = selection["classification"]
            dsi = selection["DSI"]
            osi = selection["OSI"]
            pref_dir = selection["preferred_direction_deg"]
            pref_ori = selection["preferred_orientation_deg"]
            dsi_p = selection["DSI_pvalue"]
            osi_p = selection["OSI_pvalue"]
        else:
            # Nothing passed the gate. Still show something real: pick the
            # condition with the highest raw |DSI| purely so the PSTH grid
            # and stats line have a condition to key off of — this is
            # display-only and does NOT feed DS/OS classification or the
            # population probe map, which stay gated. The person can look
            # at the actual PSTH/tuning curve and judge for themselves
            # whether the response looks real despite not clearing the
            # gate, rather than the panel deciding that for them by hiding
            # the data.
            def _abs_or_neg1(v):
                return abs(v) if np.isfinite(v) else -1.0

            display_cond = max(
                dsos_conditions, key=lambda c: _abs_or_neg1(data[c].get("DSI", np.nan))
            )
            classification = "none"
            dsi = data[display_cond].get("DSI", np.nan)
            osi = data[display_cond].get("OSI", np.nan)
            pref_dir = data[display_cond].get("preferred_direction_deg", np.nan)
            pref_ori = data[display_cond].get("preferred_orientation_deg", np.nan)
            dsi_p = data[display_cond].get("DSI_pvalue", np.nan)
            osi_p = data[display_cond].get("OSI_pvalue", np.nan)

        self._render_legend(dsos_conditions, display_cond, data)
        self._render_dsos_overlay(dsos_conditions, display_cond, data)
        self._render_hist(dsos_conditions, display_cond, data)

        display_entry = data[display_cond]
        bw, tf = display_cond
        pref_angle = pref_dir if classification == "DS" else pref_ori
        if classification == "none":
            label = "[not significant]"
        else:
            label = f"[{classification}]"
        self.stats_label.setText(
            f"{label}  Shown: bw={bw:g} tf={tf:g}Hz   "
            f"DSI: {self._fmt(dsi)} (p={self._fmt(dsi_p, 3)})   "
            f"OSI: {self._fmt(osi)} (p={self._fmt(osi_p, 3)})   "
            f"Pref. {'dir' if classification == 'DS' else 'ori'}: {self._fmt(pref_angle, 0)}°"
        )
        self._draw_direction_psth_grid(display_entry, pref_dir)

    def _render_legend(self, conditions, best_cond, data):
        parts = []
        for i, cond in enumerate(conditions):
            color = _CONDITION_COLORS[i % len(_CONDITION_COLORS)]
            bw, tf = cond
            is_best = cond == best_cond
            weight = "bold" if is_best else "normal"
            marker = "●" if is_best else "○"
            parts.append(
                f"<span style='color:{color}; font-weight:{weight};'>"
                f"{marker} bw={bw:g} tf={tf:g}Hz</span>"
            )
        self.legend_label.setText("&nbsp;&nbsp;".join(parts))

    @staticmethod
    def _fmt(val, decimals=2):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "n/a"
        return f"{val:.{decimals}f}"

    # -- Polar overlay --------------------------------------------------
    def _clear_polar(self):
        for item in self._polar_grid_items + self._polar_curves:
            self.polar_plot.removeItem(item)
        self._polar_grid_items = []
        self._polar_curves = []
        self._polar_pref_line.setData([], [])

    def _render_dsos_overlay(self, conditions, best_cond, data):
        self._clear_polar()

        r_max = 0.0
        for cond in conditions:
            resp = np.asarray(data[cond].get("mean_response", []), dtype=float)
            if resp.size and np.any(np.isfinite(resp)):
                r_max = max(r_max, float(np.nanmax(resp)))
        r_max = r_max if r_max > 0 else 1.0

        colors = resolve_theme_colors(self.main_window.get_current_colors())
        ring_pen = pg.mkPen(colors["border_subtle"], width=1)
        for frac in (0.5, 1.0):
            theta = np.linspace(0, 2 * np.pi, 100)
            ring = pg.PlotCurveItem(
                (r_max * frac) * np.cos(theta),
                (r_max * frac) * np.sin(theta),
                pen=ring_pen,
            )
            self.polar_plot.addItem(ring)
            self._polar_grid_items.append(ring)

        for i, cond in enumerate(conditions):
            entry = data[cond]
            directions = np.asarray(entry["directions_deg"], dtype=float)
            responses = np.asarray(entry["mean_response"], dtype=float)
            if directions.size == 0:
                continue

            color = _CONDITION_COLORS[i % len(_CONDITION_COLORS)]
            is_best = cond == best_cond
            alpha = _BEST_ALPHA if is_best else _SHADOW_ALPHA
            width = 2.5 if is_best else 1.0
            qcolor = pg.mkColor(color)
            qcolor.setAlpha(alpha)

            order = np.argsort(directions)
            thetas = np.deg2rad(directions[order])
            rs = np.clip(responses[order], 0, None)
            thetas = np.append(thetas, thetas[0])
            rs = np.append(rs, rs[0])
            xs = rs * np.cos(thetas)
            ys = rs * np.sin(thetas)

            curve = pg.PlotCurveItem(xs, ys, pen=pg.mkPen(qcolor, width=width))
            self.polar_plot.addItem(curve)
            self._polar_curves.append(curve)

        self.polar_plot.setXRange(-r_max * 1.15, r_max * 1.15)
        self.polar_plot.setYRange(-r_max * 1.15, r_max * 1.15)

    # -- Linear histogram overlay ----------------------------------------
    def _clear_hist(self):
        if self._hist_bar_item is not None:
            self.hist_plot.removeItem(self._hist_bar_item)
            self._hist_bar_item = None
        for c in self._hist_shadow_curves:
            self.hist_plot.removeItem(c)
        self._hist_shadow_curves = []

    def _render_hist(self, conditions, best_cond, data):
        """
        Real bar-chart histogram for the best condition (mean_response vs
        direction), with every other condition overlaid as a thin low-alpha
        line trace connecting its own direction/response points — the
        "shadow traces" of the other bar widths/temporal frequencies,
        alongside an actual histogram rather than only a polar summary.
        """
        self._clear_hist()

        best_entry = data[best_cond]
        best_dirs = np.asarray(best_entry["directions_deg"], dtype=float)
        best_resp = np.asarray(best_entry["mean_response"], dtype=float)

        if best_dirs.size == 0:
            return

        order = np.argsort(best_dirs)
        best_dirs_sorted = best_dirs[order]
        best_resp_sorted = np.clip(best_resp[order], 0, None)

        # Bar width: fraction of the smallest gap between adjacent directions,
        # falling back to a fixed width if there's only one direction.
        if len(best_dirs_sorted) > 1:
            min_gap = np.min(np.diff(best_dirs_sorted))
            bar_width = max(min_gap * 0.6, 1.0)
        else:
            bar_width = 10.0

        best_color = _CONDITION_COLORS[
            conditions.index(best_cond) % len(_CONDITION_COLORS)
        ]
        bar_brush = pg.mkColor(best_color)
        bar_brush.setAlpha(_BEST_ALPHA)

        self._hist_bar_item = pg.BarGraphItem(
            x=best_dirs_sorted,
            height=best_resp_sorted,
            width=bar_width,
            brush=bar_brush,
            pen=pg.mkPen(None),
        )
        self.hist_plot.addItem(self._hist_bar_item)

        for i, cond in enumerate(conditions):
            if cond == best_cond:
                continue
            entry = data[cond]
            dirs = np.asarray(entry["directions_deg"], dtype=float)
            resp = np.asarray(entry["mean_response"], dtype=float)
            if dirs.size == 0:
                continue
            o = np.argsort(dirs)
            color = _CONDITION_COLORS[i % len(_CONDITION_COLORS)]
            qcolor = pg.mkColor(color)
            qcolor.setAlpha(_SHADOW_ALPHA)
            curve = pg.PlotCurveItem(
                dirs[o],
                np.clip(resp[o], 0, None),
                pen=pg.mkPen(qcolor, width=1.5),
                symbol=None,
            )
            self.hist_plot.addItem(curve)
            self._hist_shadow_curves.append(curve)

        self.hist_plot.enableAutoRange()

    def _clear_psth_grid(self):
        # GraphicsLayout's removeItem()/clear() have confirmed upstream
        # bugs around cell bookkeeping and orphaned border QGraphicsRectItems
        # (pyqtgraph#2173, pyqtgraph#3085) that leave Qt's real
        # QGraphicsGridLayout believing a cell is still occupied even after
        # a "successful" removeItem() call — which is what produced the
        # "Cell (0, 2) already taken" warnings and visibly broken/misplaced
        # subplots on the next addPlot(row=, col=) at that same cell. We
        # tried removeItem()-based approaches twice; both hit the same
        # underlying pyqtgraph fragility since this grid reuses fixed
        # (row, col) positions every redraw. Rather than continue patching
        # around a confirmed library bug, replace the whole widget each
        # cluster switch — cheap for a handful of small plots, and there is
        # no shared internal state left over for Qt's grid layout to get
        # confused about, since the old widget (and its QGraphicsGridLayout)
        # is discarded wholesale rather than mutated in place.
        colors = resolve_theme_colors(self.main_window.get_current_colors())
        old_widget = self.psth_grid_widget

        new_widget = pg.GraphicsLayoutWidget()
        new_widget.setBackground(colors["bg_panel"])

        self._data_layout.replaceWidget(old_widget, new_widget)
        old_widget.setParent(None)
        old_widget.deleteLater()

        self.psth_grid_widget = new_widget
        self._psth_grid_plots = {}
        self._polar_pref_line.setData([], [])

    def _draw_direction_psth_grid(self, entry, pref_dir_deg):
        """
        Draws one small PSTH per direction for the best condition, arranged
        compass-style: grid position is chosen by each direction's angle
        (0 deg at the right/center-column-top, going counterclockwise, 3x3
        layout), so the *spatial arrangement* of the grid communicates the
        tuning shape at a glance, not just a single scalar-adjacent curve.

        Replaces the old single-direction firing-rate strip, which only
        ever plotted the direction closest to preferred and discarded
        psth_by_direction for every other direction despite it already
        being computed by grating_calc. Answers "is there a real,
        time-locked response, in which directions, or is DSI/OSI picking
        out noise on a near-silent cell" — visible across the whole tuning
        curve, not just at the winning direction.

        All subplots share one y-axis range (max across all directions'
        PSTHs) — this is required, not cosmetic: without a shared scale, a
        flat noisy direction can visually look as "big" as a real evoked
        response, which is the same amplitude-blind illusion that caused
        the original DSI-selection bug.
        """
        self._clear_psth_grid()

        directions_deg = np.asarray(entry.get("directions_deg", []), dtype=float)
        psth_by_dir = entry.get("psth_by_direction")
        t_s = entry.get("psth_time_s")

        # Pre-analyzed files loaded from disk (predating psth_by_direction)
        # won't have per-bin PSTHs. Consistent with the peak_rate_hz gating
        # decision elsewhere in this fix: no scalar-value fallback here
        # either — an empty grid is a more honest signal that this
        # cluster's data needs re-running through compute_grating_response
        # than silently substituting a flat reference line.
        if directions_deg.size == 0 or not psth_by_dir or t_s is None:
            return

        # Preferred-direction spoke on the polar plot (kept from the old
        # sanity strip — cheap, and still the clearest single indicator of
        # "where is preferred pointing" at a glance).
        if not (isinstance(pref_dir_deg, float) and math.isnan(pref_dir_deg)):
            mean_resp = np.asarray(entry.get("mean_response", []), dtype=float)
            r_max = np.nanmax(mean_resp) if mean_resp.size else 1.0
            r_max = r_max if r_max > 0 else 1.0
            theta = math.radians(pref_dir_deg)
            self._polar_pref_line.setData(
                [0, r_max * math.cos(theta)], [0, r_max * math.sin(theta)]
            )

        # Shared y-range across every subplot in the grid — see docstring.
        y_max = 0.0
        for d in directions_deg:
            rate = np.asarray(psth_by_dir.get(d, []), dtype=float)
            if rate.size and np.any(np.isfinite(rate)):
                y_max = max(y_max, float(np.nanmax(rate)))
        y_max = y_max if y_max > 0 else 1.0

        colors = resolve_theme_colors(self.main_window.get_current_colors())
        t = np.asarray(t_s)

        closest_idx = None
        if directions_deg.size and not (
            isinstance(pref_dir_deg, float) and math.isnan(pref_dir_deg)
        ):
            closest_idx = int(
                np.argmin(np.abs(((directions_deg - pref_dir_deg + 180) % 360) - 180))
            )

        # Map each direction onto a 3x3 compass grid cell by angle.
        # 0 deg -> right-center, 90 -> top-center, 180 -> left-center, etc.
        # (row, col), row 0 = top, col 0 = left, center cell (1,1) unused.
        compass_cells = {
            0: (1, 2),
            45: (0, 2),
            90: (0, 1),
            135: (0, 0),
            180: (1, 0),
            225: (2, 0),
            270: (2, 1),
            315: (2, 2),
        }

        def _nearest_cell(deg):
            keys = np.array(list(compass_cells.keys()))
            nearest = keys[np.argmin(np.abs(((keys - deg + 180) % 360) - 180))]
            return compass_cells[nearest]

        order = np.argsort(directions_deg)
        for i in order:
            d = directions_deg[i]
            rate = np.asarray(psth_by_dir.get(d, []), dtype=float)
            row, col = _nearest_cell(d)

            plot_item = self.psth_grid_widget.addPlot(row=row, col=col)
            plot_item.hideAxis("bottom")
            plot_item.hideAxis("left")
            plot_item.setYRange(0, y_max * 1.05, padding=0)
            plot_item.setMouseEnabled(x=False, y=False)
            plot_item.setMenuEnabled(False)

            is_pref = closest_idx is not None and i == closest_idx
            pen_color = (
                colors.get("plot_compare", "#E03131")
                if is_pref
                else colors.get("plot_fr", "#FFD43B")
            )
            plot_item.plot(
                t, rate, pen=pg.mkPen(pen_color, width=2 if is_pref else 1.5)
            )

            label_color = colors["text_primary"] if is_pref else colors["text_tertiary"]
            title_html = (
                f"<span style='color:{label_color}; font-size:8px;'>{d:g}°</span>"
            )
            plot_item.setTitle(title_html)

            self._psth_grid_plots[float(d)] = plot_item
