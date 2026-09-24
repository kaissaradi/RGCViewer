import logging
import math

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QThread, Qt
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from ..theme import apply_plot_theme, plot_grid_alpha, plot_stroke, resolve_theme_colors
from ..workers.workers import GratingComputeWorker
from ...analysis import grating_calc
from .polar_raster_view import PolarRasterView

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


def select_dsos_for_display(data, dsos_threshold=None):
    """Classify one cluster with the same threshold the population slider uses.

    ``None`` keeps grating_calc's module default (0.3). A float from
    ``MainWindow.dsos_threshold`` is applied to both DSI and OSI.
    """
    if dsos_threshold is None:
        return grating_calc.select_best_dsos_condition(data)
    threshold = float(dsos_threshold)
    return grating_calc.select_best_dsos_condition(
        data, dsi_threshold=threshold, osi_threshold=threshold
    )


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
        data_layout.setContentsMargins(8, 8, 8, 8)
        data_layout.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel(
            f"<span style='color:{colors['text_tertiary']}; font-size:10px; "
            f"letter-spacing:0.06em;'>DIRECTION TUNING</span>"
        )
        header.addWidget(title)
        header.addStretch()
        # Which (bw, tf) the stats, rasters and error bars describe. "Auto"
        # follows select_best_dsos_condition. A manual pick sticks across
        # cells while that condition exists, so one condition can be scanned.
        header.addWidget(QLabel("Condition:"))
        self.condition_combo = QComboBox()
        self.condition_combo.setToolTip(
            "Condition shown in the stats line, rasters and error bars.\n"
            "Auto = the strongest significant condition for this cell."
        )
        self.condition_combo.activated.connect(self._on_condition_picked)
        header.addWidget(self.condition_combo)
        self._condition_override = None
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
        # Polar plot framed by one spike raster per direction (replaces the
        # 3x3 PSTH grid, which could hold only 8 directions and dropped the
        # rest of a 12-direction run).
        self.raster_view = PolarRasterView(self.polar_plot, colors)
        plots_row.addWidget(self.raster_view, stretch=3)

        self.hist_plot = pg.PlotWidget()
        self.hist_plot.setLabel("bottom", "Direction (deg)")
        self.hist_plot.setLabel("left", "Response")
        self._style_plot(self.hist_plot, colors)
        self._hist_bar_item = (
            None  # BarGraphItem for the best condition, rebuilt per cluster
        )
        self._hist_error_item = None  # ±1 SD for the shown condition
        self._hist_shadow_curves = (
            []
        )  # PlotCurveItems for non-best conditions, rebuilt per cluster
        plots_row.addWidget(self.hist_plot, stretch=2)

        data_layout.addLayout(plots_row, stretch=4)

        # What the rings, bars and error bars mean, in words and units.
        self.units_label = QLabel("")
        self.units_label.setWordWrap(True)
        data_layout.addWidget(self.units_label)

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
        apply_plot_theme(plot_widget, colors)
        plot_item = plot_widget.getPlotItem()
        plot_item.showGrid(x=True, y=True, alpha=plot_grid_alpha(colors))

    def restyle_plots(self, colors):
        colors = resolve_theme_colors(colors)
        self._style_plot(self.polar_plot, colors)
        self._style_plot(self.hist_plot, colors)
        self._style_plot(self.sf_plot, colors)
        self.raster_view.restyle(colors)
        self._polar_pref_line.setPen(
            pg.mkPen(
                colors.get("plot_compare", "r"),
                width=plot_stroke(colors),
                style=Qt.DashLine,
            )
        )
        self._sf_curve.setPen(
            pg.mkPen(colors.get("plot_overlay", "c"), width=plot_stroke(colors))
        )
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
        # The population DS/OS slider writes MainWindow.dsos_threshold.
        # Passing it here is what makes that slider change this label.
        selection = select_dsos_for_display(
            data, getattr(self.main_window, "dsos_threshold", None)
        )

        if selection is None:
            # No dsos conditions at all — genuinely nothing to plot.
            self.stats_label.setText(
                "No direction-tuning conditions found for this cluster."
            )
            self.legend_label.setText("")
            self.units_label.setText("")
            self._populate_condition_combo([], None, data)
            self._clear_polar()
            self._clear_hist()
            self.raster_view.clear_rasters()
            return

        if selection["condition"] is not None:
            # Something passed the gate — use it, and its own classification.
            auto_cond = selection["condition"]
            classification = selection["classification"]
        else:
            # Nothing passed the gate. Still show something real: pick the
            # condition with the highest raw |DSI| purely so the rasters
            # and stats line have a condition to key off of — this is
            # display-only and does NOT feed DS/OS classification or the
            # population probe map, which stay gated. The person can look
            # at the actual rasters/tuning curve and judge for themselves
            # whether the response looks real despite not clearing the
            # gate, rather than the panel deciding that for them by hiding
            # the data.
            def _abs_or_neg1(v):
                return abs(v) if np.isfinite(v) else -1.0

            auto_cond = max(
                dsos_conditions, key=lambda c: _abs_or_neg1(data[c].get("DSI", np.nan))
            )
            classification = "none"

        self._populate_condition_combo(dsos_conditions, auto_cond, data)
        display_cond = (
            self._condition_override
            if self._condition_override in dsos_conditions
            else auto_cond
        )
        display_entry = data[display_cond]
        dsi = display_entry.get("DSI", np.nan)
        osi = display_entry.get("OSI", np.nan)
        pref_dir = display_entry.get("preferred_direction_deg", np.nan)
        pref_ori = display_entry.get("preferred_orientation_deg", np.nan)
        dsi_p = display_entry.get("DSI_pvalue", np.nan)
        osi_p = display_entry.get("OSI_pvalue", np.nan)

        self._render_legend(dsos_conditions, display_cond, data)
        self._render_dsos_overlay(dsos_conditions, display_cond, data)
        self._render_hist(dsos_conditions, display_cond, data)

        cond_label = grating_calc.format_condition_label(display_cond, display_entry)
        pref_angle = pref_dir if classification == "DS" else pref_ori
        if classification == "none":
            label = "[not significant]"
        else:
            label = f"[{classification}]"
        if display_cond != auto_cond:
            # The DS/OS label is the cell's, decided at its best condition.
            best = grating_calc.format_condition_label(auto_cond, data[auto_cond])
            label += f" (at {best})"
        self.stats_label.setText(
            f"{label}  Shown: {cond_label}   "
            f"DSI: {self._fmt(dsi)} (p={self._fmt(dsi_p, 3)})   "
            f"OSI: {self._fmt(osi)} (p={self._fmt(osi_p, 3)})   "
            f"Pref. {'dir' if classification == 'DS' else 'ori'}: {self._fmt(pref_angle, 0)}°"
        )
        self._draw_pref_spoke(display_entry, pref_dir)
        self._draw_rasters(cluster_id, display_cond, display_entry, pref_dir)
        self._update_units_label(display_entry)

    # -- Condition selector ----------------------------------------------
    def _populate_condition_combo(self, conditions, auto_cond, data):
        combo = self.condition_combo
        combo.blockSignals(True)
        combo.clear()
        if auto_cond is not None:
            auto_label = grating_calc.format_condition_label(auto_cond, data.get(auto_cond))
            combo.addItem(f"Auto ({auto_label})", None)
        for cond in conditions:
            combo.addItem(grating_calc.format_condition_label(cond, data.get(cond)), cond)
        index = 0
        if self._condition_override in conditions:
            index = 1 + conditions.index(self._condition_override)
        combo.setCurrentIndex(index)
        combo.setEnabled(bool(conditions))
        combo.blockSignals(False)

    def _on_condition_picked(self, index):
        self._condition_override = self.condition_combo.itemData(index)
        if self._current_cluster_id is not None and self._current_data is not None:
            self._render_cluster_data(self._current_cluster_id, self._current_data)

    # -- Error bars and units --------------------------------------------
    @staticmethod
    def _spread(entry):
        """(±values, 'SD' | 'SEM') for the tuning-curve error bars, or (None, None).

        SD across trials when the entry has it. Older analyzed files and v1
        caches only stored SEM; say so rather than label SEM as SD.
        """
        sd = entry.get("sd_response")
        if sd is not None:
            return np.asarray(sd, dtype=float), "SD"
        sem = entry.get("sem_response")
        if sem is not None:
            return np.asarray(sem, dtype=float), "SEM"
        return None, None

    @staticmethod
    def _response_axis_label(entry):
        label = entry.get("response_label") or "F1 amplitude"
        units = entry.get("response_units") or "spikes/s"
        return f"{label} ({units})"

    def _update_units_label(self, entry):
        colors = resolve_theme_colors(self.main_window.get_current_colors())
        spread, kind = self._spread(entry)
        n = entry.get("n_trials")
        n_txt = ""
        if n is not None and len(n):
            lo, hi = int(np.min(n)), int(np.max(n))
            n_txt = f", n = {lo}" if lo == hi else f", n = {lo}–{hi}"
        bars = (f"error bars ±1 {kind} across trials{n_txt}"
                if kind else "no error bars (file has no trial spread)")
        self.units_label.setText(
            f"<span style='color:{colors['text_tertiary']}; font-size:9px;'>"
            f"Radius and bars: {self._response_axis_label(entry)}, trial-averaged; "
            f"{bars}. Rings mark ½ and 1× the largest response. "
            f"Rasters: one row per trial, shading = stimulus on.</span>"
        )

    # -- Preferred-direction spoke and rasters ----------------------------
    def _draw_pref_spoke(self, entry, pref_dir_deg):
        if pref_dir_deg is None or not np.isfinite(pref_dir_deg):
            self._polar_pref_line.setData([], [])
            return
        mean_resp = np.asarray(entry.get("mean_response", []), dtype=float)
        r_max = np.nanmax(mean_resp) if mean_resp.size else 1.0
        r_max = r_max if np.isfinite(r_max) and r_max > 0 else 1.0
        theta = math.radians(pref_dir_deg)
        self._polar_pref_line.setData(
            [0, r_max * math.cos(theta)], [0, r_max * math.sin(theta)]
        )

    def _draw_rasters(self, cluster_id, cond, entry, pref_dir_deg):
        dm = self.main_window.data_manager
        raw = getattr(dm, "grating_raw_data", None)
        trials = None
        if raw is not None:
            trials = raw.get("spike_times_by_trial", {}).get(int(cluster_id))
        if trials is None:
            self.raster_view.clear_rasters(
                "Rasters need per-trial spikes (the raw grating file). "
                "This run only has analysed DS/OS values."
            )
            return
        rasters = grating_calc.trial_rasters(trials, raw["trial_parameters"], cond)
        timing = rasters.pop("_timing", {})
        highlight = None
        dirs = np.asarray(entry.get("directions_deg", []), dtype=float)
        if dirs.size and pref_dir_deg is not None and np.isfinite(pref_dir_deg):
            gap = np.abs(((dirs - pref_dir_deg + 180.0) % 360.0) - 180.0)
            highlight = float(dirs[int(np.argmin(gap))])
        self.raster_view.set_rasters(rasters, timing, highlight_dir=highlight)

    def _render_legend(self, conditions, best_cond, data):
        parts = []
        for i, cond in enumerate(conditions):
            color = _CONDITION_COLORS[i % len(_CONDITION_COLORS)]
            cond_label = grating_calc.format_condition_label(cond, data.get(cond))
            is_best = cond == best_cond
            weight = "bold" if is_best else "normal"
            marker = "●" if is_best else "○"
            parts.append(
                f"<span style='color:{color}; font-weight:{weight};'>"
                f"{marker} {cond_label}</span>"
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
        best_entry = data[best_cond]
        best_resp = np.asarray(best_entry.get("mean_response", []), dtype=float)
        spread, _kind = self._spread(best_entry)
        if spread is not None and spread.size == best_resp.size and best_resp.size:
            tops = np.clip(best_resp, 0, None) + np.nan_to_num(spread)
            if np.any(np.isfinite(tops)):
                r_max = max(r_max, float(np.nanmax(tops)))
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
            # Ring value, so the radius has a scale. Anchored to the left of
            # its point so it stays inside the view; the unit is the title.
            ring_label = pg.TextItem(
                f"{r_max * frac:.3g}", color=colors["text_tertiary"], anchor=(1, 1))
            ang = math.radians(22.5)
            ring_label.setPos(r_max * frac * math.cos(ang), r_max * frac * math.sin(ang))
            self.polar_plot.addItem(ring_label)
            self._polar_grid_items.append(ring_label)

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

        # ±spread along each spoke of the shown condition.
        dirs = np.asarray(best_entry.get("directions_deg", []), dtype=float)
        if spread is not None and spread.size == dirs.size == best_resp.size and dirs.size:
            th = np.deg2rad(dirs)
            r = np.clip(best_resp, 0, None)
            lo = np.clip(r - spread, 0, None)
            hi = r + spread
            ok = np.isfinite(lo) & np.isfinite(hi)
            xs = np.column_stack([lo * np.cos(th), hi * np.cos(th)])[ok].ravel()
            ys = np.column_stack([lo * np.sin(th), hi * np.sin(th)])[ok].ravel()
            color = _CONDITION_COLORS[conditions.index(best_cond) % len(_CONDITION_COLORS)]
            bars = pg.PlotCurveItem(xs, ys, connect="pairs", pen=pg.mkPen(color, width=1.5))
            self.polar_plot.addItem(bars)
            self._polar_grid_items.append(bars)

        self.polar_plot.setTitle(
            f"<span style='color:{colors['text_tertiary']}; font-size:9px;'>"
            f"{self._response_axis_label(best_entry)}</span>")
        self.polar_plot.setXRange(-r_max * 1.15, r_max * 1.15)
        self.polar_plot.setYRange(-r_max * 1.15, r_max * 1.15)

    # -- Linear histogram overlay ----------------------------------------
    def _clear_hist(self):
        if self._hist_bar_item is not None:
            self.hist_plot.removeItem(self._hist_bar_item)
            self._hist_bar_item = None
        if self._hist_error_item is not None:
            self.hist_plot.removeItem(self._hist_error_item)
            self._hist_error_item = None
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
        self.hist_plot.setLabel("left", self._response_axis_label(best_entry))

        spread, _kind = self._spread(best_entry)
        if spread is not None and spread.size == best_resp.size:
            s_sorted = np.nan_to_num(spread[order])
            self._hist_error_item = pg.ErrorBarItem(
                x=best_dirs_sorted,
                y=best_resp_sorted,
                top=s_sorted,
                bottom=np.minimum(s_sorted, best_resp_sorted),  # never below 0
                beam=bar_width * 0.4,
                pen=pg.mkPen(resolve_theme_colors(
                    self.main_window.get_current_colors())["text_primary"], width=1.2),
            )
            self.hist_plot.addItem(self._hist_error_item)

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
