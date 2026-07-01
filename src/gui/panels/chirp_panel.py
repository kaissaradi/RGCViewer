import logging
import numpy as np
import pyqtgraph as pg
import scipy.ndimage as ndimage
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QStackedLayout, QVBoxLayout, QWidget

from ..theme import resolve_theme_colors

logger = logging.getLogger(__name__)

pg.setConfigOptions(antialias=True)

def naka_rushton(c, r_max, c50, n, offset):
    """Naka-Rushton model for contrast sensitivity."""
    return r_max * (c**n / (c**n + c50**n)) + offset


class ChirpPanel(QWidget):
    """
    Displays the chirp-stimulus PSTH and extracted temporal features for the selected cluster.
    Data is loaded via DataManager.get_chirp_data_for_cluster().
    Metrics (Transience, ON/OFF bias) and smoothing are computed on-the-fly.
    """

    PHASE_COLORS = {
        'phase_step_on': (120, 120, 120, 40),
        'phase_step_off': (120, 120, 120, 40),
        'phase_freq_sweep': (80, 140, 220, 35),
        'phase_contrast': (220, 140, 80, 35),
    }

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        colors = resolve_theme_colors(self.main_window.get_current_colors())

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedLayout()
        outer_layout.addLayout(self.stack)

        # ---------------------------------------------------------
        # Page 0: Dashboard View
        # ---------------------------------------------------------
        data_page = QWidget()
        data_layout = QVBoxLayout(data_page)
        data_layout.setContentsMargins(4, 4, 4, 4)

        # --- Header / Metrics Bar ---
        self.header_layout = QHBoxLayout()
        title = QLabel(
            f"<span style='color:{colors['text_tertiary']}; font-size:10px; "
            f"letter-spacing:0.06em;'>CHIRP DASHBOARD</span>"
        )
        self.header_layout.addWidget(title)
        self.header_layout.addStretch()

        # Dynamic metric labels
        self.lbl_qi = QLabel("")
        self.lbl_bias = QLabel("")
        self.lbl_transience = QLabel("")
        self.lbl_max_fr = QLabel("")
        
        for lbl in (self.lbl_max_fr, self.lbl_bias, self.lbl_transience, self.lbl_qi):
            lbl.setStyleSheet(f"color: {colors['text_secondary']}; font-size: 11px;")
            self.header_layout.addWidget(lbl)
            self.header_layout.addSpacing(10)
            
        data_layout.addLayout(self.header_layout)

        # --- Graphics Layout (Grid) ---
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(colors['bg_panel'])
        data_layout.addWidget(self.glw)

        # Top: Full Spanning PSTH
        self.psth_plot = self.glw.addPlot(colspan=3)
        self.psth_plot.setLabel('bottom', "Time (s)")
        self.psth_plot.setLabel('left', "Firing Rate (Hz)")
        
        # Bottom Row: Subplots
        self.glw.nextRow()
        self.step_plot = self.glw.addPlot(title="Step Response (Zoom)")
        self.step_plot.setLabel('bottom', "Time (s)")
        
        self.freq_plot = self.glw.addPlot(title="Freq Tuning")
        self.freq_plot.setLabel('bottom', "Frequency (Hz)")
        
        self.contrast_plot = self.glw.addPlot(title="Contrast Tuning")
        self.contrast_plot.setLabel('bottom', "Contrast (%)")

        self.plots = [self.psth_plot, self.step_plot, self.freq_plot, self.contrast_plot]
        for p in self.plots:
            self._style_plot(p, colors)

        # --- Curve Initialization ---
        pen_style = pg.mkPen(colors.get('plot_fr', 'y'), width=1.5)
        fill_color = pg.mkColor(colors.get('plot_fr', 'y'))
        fill_color.setAlpha(40)
        brush_style = pg.mkBrush(fill_color)

        self._psth_curve = self.psth_plot.plot([], [], pen=pen_style, fillLevel=0, fillBrush=brush_style)
        self._step_curve = self.step_plot.plot([], [], pen=pen_style, fillLevel=0, fillBrush=brush_style)
        self._freq_curve = self.freq_plot.plot([], [], pen=pen_style, fillLevel=0, fillBrush=brush_style)
        self._contrast_curve = self.contrast_plot.plot([], [], pen=pen_style, fillLevel=0, fillBrush=brush_style)

        # Phase Regions (Only on the main PSTH)
        self._phase_regions = {}
        for key, rgba in self.PHASE_COLORS.items():
            region = pg.LinearRegionItem(values=(0, 0), brush=pg.mkBrush(*rgba), movable=False)
            region.setZValue(-10)
            for line in region.lines:
                line.setPen(pg.mkPen(None))
            self.psth_plot.addItem(region)
            self._phase_regions[key] = region

        self.stack.addWidget(data_page)

        # ---------------------------------------------------------
        # Page 1: Placeholder
        # ---------------------------------------------------------
        placeholder_page = QWidget()
        placeholder_layout = QVBoxLayout(placeholder_page)
        self.placeholder_label = QLabel("No chirp data")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        placeholder_layout.addWidget(self.placeholder_label)
        self.stack.addWidget(placeholder_page)

        self.stack.setCurrentIndex(1)

    def _style_plot(self, plot_item, colors):
        plot_item.getAxis('bottom').setPen(pg.mkPen(colors['border_default']))
        plot_item.getAxis('left').setPen(pg.mkPen(colors['border_default']))
        plot_item.getAxis('bottom').setTextPen(pg.mkPen(colors['text_secondary']))
        plot_item.getAxis('left').setTextPen(pg.mkPen(colors['text_secondary']))
        plot_item.showAxis('top', False)
        plot_item.showAxis('right', False)
        plot_item.showGrid(x=True, y=True, alpha=0.08)
        
        # Style subplot titles if they exist
        title_label = plot_item.titleLabel
        if title_label:
            title_label.setText(title_label.text, color=colors['text_secondary'], size='9pt')

    def restyle_plots(self, colors):
        colors = resolve_theme_colors(colors)
        self.glw.setBackground(colors['bg_panel'])
        
        pen_style = pg.mkPen(colors.get('plot_fr', 'y'), width=1.5)
        fill_color = pg.mkColor(colors.get('plot_fr', 'y'))
        fill_color.setAlpha(40)
        brush_style = pg.mkBrush(fill_color)

        for p in self.plots:
            self._style_plot(p, colors)

        for curve in [self._psth_curve, self._step_curve, self._freq_curve, self._contrast_curve]:
            curve.setPen(pen_style)
            curve.setBrush(brush_style)

    def update_all(self, cluster_id):
        if cluster_id is None: return
        dm = self.main_window.data_manager
        if dm is None: return

        data = dm.get_chirp_data_for_cluster(cluster_id)
        if data is None:
            self.stack.setCurrentIndex(1)
            self.placeholder_label.setText(
                f"No chirp response for cluster {cluster_id}" if getattr(dm, 'chirp_available', False) else "No chirp data"
            )
            return

        self.stack.setCurrentIndex(0)
        psth = np.asarray(data['psth_mean'])
        bin_size_ms = data['bin_size_ms']
        
        # Smoothing
        sigma_bins = 25.0 / bin_size_ms
        smoothed = ndimage.gaussian_filter1d(psth, sigma=sigma_bins, mode='constant')
        t = (np.arange(len(smoothed)) * bin_size_ms) / 1000.0

        # Update Main Plot
        self._psth_curve.setData(t, smoothed)
        for key, region in self._phase_regions.items():
            s, e = data.get(key, (0, 0))
            region.setRegion(((s * bin_size_ms) / 1000.0, (e * bin_size_ms) / 1000.0))

        # Metrics
        on_s, on_e = data.get('phase_step_on', (0,0))
        off_s, off_e = data.get('phase_step_off', (0,0))
        on_trace, off_trace = smoothed[on_s:on_e], smoothed[off_s:off_e]
        on_pk, off_pk = np.max(on_trace) if len(on_trace)>0 else 0, np.max(off_trace) if len(off_trace)>0 else 0
        
        bias = (on_pk - off_pk) / (on_pk + off_pk) if (on_pk + off_pk) > 0 else 0.0
        dom = on_trace if on_pk > off_pk else off_trace
        ti = 0.0
        if len(dom) > 0:
            sustained_bins = int(200.0 / bin_size_ms)
            if len(dom) > sustained_bins:
                ti = (np.max(dom) - np.mean(dom[-sustained_bins:])) / np.max(dom) if np.max(dom) > 0 else 0.0

        self.lbl_qi.setText(f"<b>QI:</b> {data.get('quality_index', 0):.2f}")
        self.lbl_max_fr.setText(f"<b>Max:</b> {np.max(smoothed):.1f}Hz")
        self.lbl_bias.setText(f"<b>Bias:</b> {bias:.2f}")
        self.lbl_transience.setText(f"<b>TI:</b> {ti:.2f}")

        # Subplot A: Step Zoom
        self._step_curve.setData(t[:max(on_e, off_e)+100], smoothed[:max(on_e, off_e)+100])

        # Subplot B: Freq Tuning
        f_s, f_e = data.get('phase_freq_sweep', (0,0))
        if f_e > f_s:
            self._freq_curve.setData(np.linspace(data.get('freq_min_hz', 0.5), data.get('freq_max_hz', 8.0), f_e-f_s), smoothed[f_s:f_e])
        else: self._freq_curve.setData([], [])

        # Subplot C: Contrast Response Function (CRF)
        c_s, c_e = data.get('phase_contrast', (0,0))
        if c_e > c_s:
            y = smoothed[c_s:c_e]
            x = np.linspace(0, 1, len(y)) # Normalized contrast
            
            # CRF Fitting (Naka-Rushton)
            try:
                from scipy.optimize import curve_fit
                popt, _ = curve_fit(naka_rushton, x, y, p0=[np.max(y), 0.5, 2.0, np.min(y)], bounds=(0, [np.inf, 1, 10, np.inf]))
                self._contrast_curve.setData(x * 100, naka_rushton(x, *popt)) # Show fit
            except:
                self._contrast_curve.setData(x * 100, y) # Fallback to raw if fit fails
        else: self._contrast_curve.setData([], [])

        for p in self.plots: p.enableAutoRange()

    def cleanup(self):
        for p in self.plots:
            if p:
                p.clear()
        if hasattr(self.glw, 'close'):
            self.glw.close()

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)