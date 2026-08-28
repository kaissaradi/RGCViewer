"""Chirp dashboard: stimulus, raster, PSTH and per-phase tuning for one cell.

Laid out the way a chirp figure is laid out in a paper, top to bottom on a
shared time axis:

    stimulus model   rebuilt from the recorded parameters, never hardcoded
    raster           every trial, grouped and labelled by light level
    PSTH             trial-averaged firing rate

and beneath, the three things the phases actually measure: how ON or OFF the
cell is (left), how fast it can follow (middle), and how its response grows
with contrast (right).

The polarity panel pools the stimulus's *two* increments (light on at 0 s, dark
off at 7 s) against its *two* decrements (light off at 3 s, dark on at 4 s).
The big transient near 7 s is the offset of the dark step, so it is ON drive;
an index built from the bright step alone misses it, and reading it as a second
bright step inverts the cell's polarity.
"""

import logging

import numpy as np
import pyqtgraph as pg
import scipy.ndimage as ndimage
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QStackedLayout, QVBoxLayout, QWidget,
)

from ..theme import apply_plot_theme, plot_grid_alpha, plot_stroke, resolve_theme_colors
from ...analysis import chirp_calc

logger = logging.getLogger(__name__)

pg.setConfigOptions(antialias=True)

SMOOTH_MS = 25.0
# Light levels in the raster, dim to bright, matching the Contrast tab.
LEVEL_COLORS = [(120, 130, 145), (90, 140, 220), (60, 190, 160),
                (240, 170, 60), (235, 110, 90), (190, 120, 220)]


class ChirpPanel(QWidget):
    """Stimulus / raster / PSTH / tuning dashboard for the selected cluster."""

    PHASE_COLORS = {
        "bright step": (120, 120, 120, 40),
        "dark step": (120, 120, 120, 40),
        "frequency sweep": (80, 140, 220, 35),
        "contrast ramp": (220, 140, 80, 35),
    }

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._cluster_id = None
        colors = resolve_theme_colors(self.main_window.get_current_colors())

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedLayout()
        outer.addLayout(self.stack)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # --- header ---
        self.header_layout = QHBoxLayout()
        self.header_layout.addWidget(QLabel(
            f"<span style='color:{colors['text_tertiary']}; font-size:10px; "
            f"letter-spacing:0.06em;'>CHIRP DASHBOARD</span>"))
        self.header_layout.addStretch()
        self.lbl_qi = QLabel("")
        self.lbl_rate = QLabel("")
        self.lbl_onoff = QLabel("")
        self.lbl_max_fr = QLabel("")
        for lbl in (self.lbl_max_fr, self.lbl_rate, self.lbl_onoff, self.lbl_qi):
            lbl.setStyleSheet(f"color: {colors['text_secondary']}; font-size: 11px;")
            self.header_layout.addWidget(lbl)
            self.header_layout.addSpacing(10)

        self.split_chk = QCheckBox("Split PSTH by light level")
        self.split_chk.setToolTip(
            "The stored PSTH averages every trial together, including trials\n"
            "recorded at different light levels. Splitting shows one trace per\n"
            "run, which is the honest view when a chirp file spans NDF steps.")
        self.split_chk.toggled.connect(lambda: self.update_all(self._cluster_id))
        self.header_layout.addWidget(self.split_chk)

        from ..widgets.widgets import make_reset_button

        def _reset_chirp_views():
            for plot in getattr(self, "plots", ()):
                try:
                    plot.enableAutoRange()
                except Exception:
                    pass

        self.reset_views_btn = make_reset_button(_reset_chirp_views, page)
        self.header_layout.addWidget(self.reset_views_btn)
        layout.addLayout(self.header_layout)

        # --- graphics ---
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(colors["bg_panel"])
        layout.addWidget(self.glw)

        self.stim_plot = self.glw.addPlot(colspan=3)
        self.stim_plot.setLabel("left", "Stim")
        self.stim_plot.setMouseEnabled(x=True, y=False)
        self.stim_plot.hideAxis("bottom")

        self.glw.nextRow()
        self.raster_plot = self.glw.addPlot(colspan=3)
        self.raster_plot.setLabel("left", "Trial")
        self.raster_plot.hideAxis("bottom")

        self.glw.nextRow()
        self.psth_plot = self.glw.addPlot(colspan=3)
        self.psth_plot.setLabel("bottom", "Time (s)")
        self.psth_plot.setLabel("left", "Rate (Hz)")

        self.glw.nextRow()
        self.polarity_plot = self.glw.addPlot(title="Polarity")
        self.polarity_plot.setLabel("left", "Δ rate (Hz)")
        self.freq_plot = self.glw.addPlot(title="Frequency tuning")
        self.freq_plot.setLabel("bottom", "Frequency (Hz)")
        self.freq_plot.setLabel("left", "Amplitude (Hz)")
        self.contrast_plot = self.glw.addPlot(title="Contrast tuning")
        self.contrast_plot.setLabel("bottom", "Contrast")
        self.contrast_plot.setLabel("left", "Amplitude (Hz)")

        # One time axis for the three stacked traces: zooming the PSTH must
        # carry the raster and the stimulus with it or they stop lining up.
        for p in (self.stim_plot, self.raster_plot):
            p.setXLink(self.psth_plot)

        # The stimulus strip is a legend, not data; the tuning row needs room
        # for its axes.
        self.glw.ci.layout.setRowStretchFactor(0, 1)
        self.glw.ci.layout.setRowStretchFactor(1, 4)
        self.glw.ci.layout.setRowStretchFactor(2, 3)
        self.glw.ci.layout.setRowStretchFactor(3, 3)

        self.plots = [self.stim_plot, self.raster_plot, self.psth_plot,
                      self.polarity_plot, self.freq_plot, self.contrast_plot]
        for p in self.plots:
            self._style_plot(p, colors)

        self._phase_regions = {}
        self.stack.addWidget(page)

        placeholder = QWidget()
        pl = QVBoxLayout(placeholder)
        self.placeholder_label = QLabel("No chirp data")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        pl.addWidget(self.placeholder_label)
        self.stack.addWidget(placeholder)
        self.stack.setCurrentIndex(1)

    # ── chrome ───────────────────────────────────────────────────────────────

    def _style_plot(self, p, colors):
        apply_plot_theme(p, colors)
        p.showGrid(x=True, y=True, alpha=plot_grid_alpha(colors))
        if p.titleLabel:
            p.titleLabel.setText(p.titleLabel.text, color=colors["text_secondary"],
                                 size="9pt")

    def restyle_plots(self, colors):
        colors = resolve_theme_colors(colors)
        self.glw.setBackground(colors["bg_panel"])
        for p in self.plots:
            self._style_plot(p, colors)
        for lbl in (self.lbl_max_fr, self.lbl_rate, self.lbl_onoff, self.lbl_qi):
            lbl.setStyleSheet(f"color: {colors['text_secondary']}; font-size: 11px;")
        self.update_all(self._cluster_id)

    @staticmethod
    def _level_color(i):
        return LEVEL_COLORS[i % len(LEVEL_COLORS)]

    # ── update ───────────────────────────────────────────────────────────────

    def update_all(self, cluster_id):
        if cluster_id is None:
            return
        self._cluster_id = cluster_id
        dm = self.main_window.data_manager
        if dm is None:
            return

        data = dm.get_chirp_data_for_cluster(cluster_id)
        if data is None:
            self.stack.setCurrentIndex(1)
            self.placeholder_label.setText(
                f"No chirp response for cluster {cluster_id}"
                if getattr(dm, "chirp_available", False) else "No chirp data")
            return

        self.stack.setCurrentIndex(0)
        colors = resolve_theme_colors(self.main_window.get_current_colors())
        timing = dm.chirp_timing()
        bin_ms = float(data["bin_size_ms"])
        trials = dm.get_chirp_trials_for_cluster(cluster_id)

        for p in (self.stim_plot, self.raster_plot, self.psth_plot,
                  self.polarity_plot, self.freq_plot, self.contrast_plot):
            p.clear()

        self._draw_stimulus(timing, colors)
        self._add_phase_regions(self.raster_plot, timing)
        self._add_phase_regions(self.psth_plot, timing)
        self._draw_raster(trials, bin_ms, colors)
        psth = self._draw_psth(data, trials, bin_ms, colors)
        self._draw_polarity(psth, timing, bin_ms, data, dm, cluster_id, colors)
        self._draw_frequency(psth, timing, bin_ms, colors)
        self._draw_contrast(psth, timing, bin_ms, colors)
        self.psth_plot.setXRange(0, timing.total, padding=0.01)

    # ── panels ───────────────────────────────────────────────────────────────

    def _add_phase_regions(self, plot, timing):
        """Shade the stimulus phases behind *plot*.

        Drawn on all three time-aligned plots, not just the stimulus strip, so
        a burst in the raster or a bump in the PSTH can be attributed to a
        phase by looking straight down rather than by tracking a time value up
        to the legend.
        """
        for name, (a, b) in timing.phases().items():
            rgba = self.PHASE_COLORS.get(name)
            if rgba is None:
                continue
            region = pg.LinearRegionItem(values=(a, b), brush=pg.mkBrush(*rgba),
                                         movable=False)
            region.setZValue(-10)
            for line in region.lines:
                line.setPen(pg.mkPen(None))
            # The bands are chrome: without this they answer mouse events and
            # a drag over the PSTH grabs a band instead of panning the axis.
            region.setAcceptedMouseButtons(Qt.NoButton)
            plot.addItem(region, ignoreBounds=True)

    def _draw_stimulus(self, timing, colors):
        t, y = chirp_calc.stimulus_intensity(timing)
        self.stim_plot.plot(t, y, pen=pg.mkPen(colors["text_secondary"], width=1))
        self.stim_plot.setYRange(-0.05, 1.05, padding=0)
        self.stim_plot.getAxis("left").setTicks([[(0, "0"), (0.5, "½"), (1, "1")]])
        self._add_phase_regions(self.stim_plot, timing)

    def _draw_raster(self, trials, bin_ms, colors):
        if trials is None:
            return
        counts = trials["counts"]
        n_trials, n_bins = counts.shape
        dt = bin_ms / 1000.0
        for i, (run, level, start, stop) in enumerate(trials["blocks"]):
            col = self._level_color(i)
            block = counts[start:stop]
            rows, cols = np.nonzero(block)
            if len(cols):
                self.raster_plot.addItem(pg.ScatterPlotItem(
                    x=(cols + 0.5) * dt, y=start + rows + 0.5,
                    pen=None, brush=pg.mkBrush(*col), size=2.0, pxMode=True))
            if start > 0:
                line = pg.InfiniteLine(pos=start, angle=0,
                                       pen=pg.mkPen(colors["border_default"], width=1))
                self.raster_plot.addItem(line)
            text = pg.TextItem(f"{level}", color=col, anchor=(0, 0))
            text.setPos(0.1, start)
            self.raster_plot.addItem(text)
        self.raster_plot.setYRange(0, n_trials, padding=0.02)
        self.raster_plot.invertY(True)

    def _draw_psth(self, data, trials, bin_ms, colors):
        """Draws the PSTH and returns the (smoothed) trace the tuning uses."""
        sigma = SMOOTH_MS / bin_ms
        psth = np.asarray(data["psth_mean"], dtype=float)
        smoothed = ndimage.gaussian_filter1d(psth, sigma=sigma, mode="nearest")
        t = np.arange(len(smoothed)) * bin_ms / 1000.0

        split = self.split_chk.isChecked() and trials is not None and \
            len(trials["blocks"]) > 1
        if split:
            for i, (run, level, start, stop) in enumerate(trials["blocks"]):
                block = trials["counts"][start:stop].mean(axis=0) / (bin_ms / 1000.0)
                self.psth_plot.plot(
                    t[:len(block)],
                    ndimage.gaussian_filter1d(block, sigma=sigma, mode="nearest"),
                    pen=pg.mkPen(self._level_color(i), width=plot_stroke(colors)))
        else:
            pen = pg.mkPen(colors.get("plot_fr", "y"), width=plot_stroke(colors))
            fill = pg.mkColor(colors.get("plot_fr", "y"))
            fill.setAlpha(40)
            self.psth_plot.plot(t, smoothed, pen=pen, fillLevel=0,
                                fillBrush=pg.mkBrush(fill))
        return smoothed

    def _draw_polarity(self, psth, timing, bin_ms, data, dm, cluster_id, colors):
        resp = chirp_calc.transition_responses(psth, timing, bin_ms)
        index, on, off = chirp_calc.on_off_index(psth, timing, bin_ms)

        if resp:
            heights = [r["delta_hz"] for r in resp]
            # Increments and decrements in different colours: the point of the
            # panel is that two of the four transitions are increments.
            brushes = [pg.mkBrush(90, 190, 240) if r["polarity"] > 0
                       else pg.mkBrush(230, 130, 90) for r in resp]
            self.polarity_plot.addItem(pg.BarGraphItem(
                x=np.arange(len(resp)), height=heights, width=0.62, brushes=brushes))
            self.polarity_plot.getAxis("bottom").setTicks([[
                (i, f"{r['label']}\n{'inc' if r['polarity'] > 0 else 'dec'}")
                for i, r in enumerate(resp)]])
            self.polarity_plot.addItem(pg.InfiniteLine(
                pos=0, angle=0, pen=pg.mkPen(colors["border_default"], width=1)))

        qi = float(data.get("quality_index", float("nan")))
        details = dm.chirp_qi_details(cluster_id) if hasattr(dm, "chirp_qi_details") else None
        qi_txt = "—"
        if details and details["per_run"]:
            shown = [r for r in details["per_run"] if not r.get("gated")]
            if shown:
                best = max(shown, key=lambda r: r["qi"])
                qi_txt = f"{best['qi']:.2f}"
            else:
                qi_txt = "too few spikes"
        elif np.isfinite(qi):
            qi_txt = f"{qi:.2f}"

        mean_rate = float(np.mean(psth))
        self.polarity_plot.setTitle(
            f"ON/OFF {index:+.2f}" if np.isfinite(index) else "ON/OFF —",
            color=colors["text_secondary"], size="9pt")

        self.lbl_qi.setText(f"<b>QI:</b> {qi_txt}")
        self.lbl_rate.setText(f"<b>Mean:</b> {mean_rate:.1f}Hz")
        self.lbl_max_fr.setText(f"<b>Max:</b> {np.max(psth):.1f}Hz")
        self.lbl_onoff.setText(
            f"<b>ON/OFF:</b> {index:+.2f}" if np.isfinite(index)
            else "<b>ON/OFF:</b> —")

    def _draw_frequency(self, psth, timing, bin_ms, colors):
        freqs, amps, snr = chirp_calc.frequency_tuning(psth, timing, bin_ms)
        if not len(freqs):
            return
        self.freq_plot.plot(freqs, amps, pen=pg.mkPen(colors.get("plot_scatter", "c"),
                                                      width=2),
                            symbol="o", symbolSize=4,
                            symbolBrush=pg.mkBrush(colors.get("plot_scatter", "c")),
                            symbolPen=None)
        # Where the cell stops following: the highest frequency still clearing
        # the local noise floor. Marked rather than merely implied, because the
        # curve alone does not say which part of it is signal.
        good = np.where(snr > 3.0)[0]
        if len(good):
            self.freq_plot.addItem(pg.InfiniteLine(
                pos=float(freqs[good[-1]]), angle=90,
                pen=pg.mkPen(colors.get("plot_peak", "y"), width=1,
                             style=Qt.DashLine),
                label=f"{freqs[good[-1]]:.1f} Hz",
                labelOpts={"position": 0.9, "color": colors.get("plot_peak", "y")}))

    def _draw_contrast(self, psth, timing, bin_ms, colors):
        contrasts, amps, snr = chirp_calc.contrast_tuning(psth, timing, bin_ms)
        if not len(contrasts):
            return
        col = colors.get("plot_compare", "#FF9800")
        self.contrast_plot.plot(contrasts, amps, pen=pg.mkPen(col, width=2),
                                symbol="o", symbolSize=4,
                                symbolBrush=pg.mkBrush(col), symbolPen=None)
        # Same saturation model the Contrast tab fits, so c50 means the same
        # thing in both places.
        from ...analysis.contrast_calc import fit_naka_rushton, naka_rushton
        fit = fit_naka_rushton(contrasts, amps)
        if fit is not None and fit["r_squared"] > 0.5:
            xf = np.linspace(float(contrasts.min()), float(contrasts.max()), 100)
            self.contrast_plot.plot(
                xf, naka_rushton(xf, fit["r_max"], fit["c50"], fit["n"],
                                 fit["offset"]),
                pen=pg.mkPen(col, width=1, style=Qt.DotLine))
            flag = "*" if fit["c50_at_bound"] else ""
            self.contrast_plot.setTitle(
                f"Contrast tuning — c50 {fit['c50']:.2f}{flag}",
                color=colors["text_secondary"], size="9pt")

    # ── lifecycle ────────────────────────────────────────────────────────────

    def cleanup(self):
        for p in self.plots:
            if p:
                p.clear()
        if hasattr(self.glw, "close"):
            self.glw.close()

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)
