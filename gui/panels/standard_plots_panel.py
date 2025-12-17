import numpy as np
import pyqtgraph as pg
from qtpy.QtWidgets import QWidget, QVBoxLayout, QSplitter, QHBoxLayout, QCheckBox, QComboBox, QLabel
from qtpy.QtCore import Qt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate
from analysis.constants import ISI_REFRACTORY_PERIOD_MS

class StandardPlotsPanel(QWidget):
    """
    Standard Dashboard:
    [ Template Grid ] [ Autocorrelation ]
    [ ISI Hist      ] [ Firing Rate     ]
    """
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        # Controls: heatmap toggle, weighting mode, and ISI view toggle
        ctrl_bar = QHBoxLayout()
        self.heatmap_checkbox = QCheckBox('Show Amplitude Heatmap')
        self.heatmap_checkbox.setChecked(True)
        ctrl_bar.addWidget(self.heatmap_checkbox)

        ctrl_bar.addWidget(QLabel('Weighting:'))
        self.weighting_combo = QComboBox()
        self.weighting_combo.addItems(['mean', 'median'])
        ctrl_bar.addWidget(self.weighting_combo)

        self.isi_view_combo = QComboBox()
        self.isi_view_combo.addItems(['ISI Histogram', 'ISI vs Amplitude'])
        ctrl_bar.addWidget(QLabel('ISI View:'))
        ctrl_bar.addWidget(self.isi_view_combo)

        self.main_channel_checkbox = QCheckBox('Show Main Channel Only')
        self.main_channel_checkbox.setChecked(False)
        ctrl_bar.addWidget(self.main_channel_checkbox)

        # connect controls to refresh the panel immediately
        self.heatmap_checkbox.toggled.connect(self._on_control_changed)
        self.weighting_combo.currentIndexChanged.connect(self._on_control_changed)
        self.isi_view_combo.currentTextChanged.connect(self._on_control_changed)
        self.main_channel_checkbox.toggled.connect(self._on_control_changed)

        layout.addLayout(ctrl_bar)

        # 2x2 Layout using Splitters
        self.vert_splitter = QSplitter(Qt.Vertical)
        layout.addWidget(self.vert_splitter)

        self.top_splitter = QSplitter(Qt.Horizontal)
        self.bottom_splitter = QSplitter(Qt.Horizontal)
        self.vert_splitter.addWidget(self.top_splitter)
        self.vert_splitter.addWidget(self.bottom_splitter)

        # 1. Template Grid (Top Left)
        self.grid_widget = pg.GraphicsLayoutWidget()
        self.grid_plot = self.grid_widget.addPlot(title="Spatial Template")
        self.grid_plot.setAspectLocked(True)
        self.grid_plot.hideAxis('bottom')
        self.grid_plot.hideAxis('left')
        self.top_splitter.addWidget(self.grid_widget)

        # 2. Autocorrelation (Top Right)
        self.acg_plot = pg.PlotWidget(title="Autocorrelation")
        self.acg_plot.setLabel('bottom', "Lag (ms)")
        self.top_splitter.addWidget(self.acg_plot)

        # 3. ISI (Bottom Left)
        self.isi_plot = pg.PlotWidget(title="ISI Distribution")
        self.isi_plot.setLabel('bottom', "ISI (ms)")
        self.bottom_splitter.addWidget(self.isi_plot)

        # 4. Firing Rate (Bottom Right)
        self.fr_plot = pg.PlotWidget(title="Firing Rate")
        self.fr_plot.setLabel('bottom', "Time (s)")
        self.fr_plot.setLabel('left', "Hz")
        self.bottom_splitter.addWidget(self.fr_plot)

        self.vert_splitter.setSizes([500, 300])
        # ensure controls reflect current selection on init
        try:
            self._on_control_changed()
        except Exception:
            pass

    def _on_control_changed(self):
        """Called when a control changes; refresh the panel for the current selection."""
        try:
            cluster_id = self.main_window._get_selected_cluster_id()
        except Exception:
            cluster_id = None
        if cluster_id is not None:
            try:
                self.update_all(cluster_id)
            except Exception:
                # avoid bubbling UI errors during control toggles
                pass

    def update_all(self, cluster_id):
        if cluster_id is None: return
        dm = self.main_window.data_manager

        # --- 1. Template Grid (Spatial) ---
        self.grid_plot.clear()
        if hasattr(dm, 'templates') and cluster_id < dm.templates.shape[0]:
            template = dm.templates[cluster_id] # (n_time, n_chan)
            pos = dm.channel_positions
            x_scale, y_scale = 1.5, 1.0

            # Per-channel peak-to-peak
            ptp = template.max(axis=0) - template.min(axis=0)
            max_ptp = ptp.max() if ptp.size>0 else 1.0
            # Cluster amplitude (mean or median)
            weighting_mode = self.weighting_combo.currentText() if hasattr(self, 'weighting_combo') else 'mean'
            cluster_amp = dm.get_cluster_mean_amplitude(cluster_id, method=weighting_mode)

            # Channel values = normalized ptp * cluster amplitude
            if max_ptp == 0:
                norm_ptp = np.zeros_like(ptp)
            else:
                norm_ptp = ptp / max_ptp
            channel_values = norm_ptp * cluster_amp

            # Optionally draw amplitude-weighted scatter under traces
            if self.heatmap_checkbox.isChecked():
                # Build scatter spots
                spots = []
                vmin, vmax = channel_values.min(), channel_values.max()
                vrange = max(vmax - vmin, 1e-6)
                for ch in range(len(channel_values)):
                    x, y = pos[ch]
                    val = (channel_values[ch] - vmin) / vrange
                    size = 6 + val * 20
                    # color from cool->hot
                    r = int(255 * val)
                    g = int(120 * (1 - val))
                    b = int(255 * (1 - val))
                    brush = pg.mkBrush(r, g, b, 180)
                    spots.append({'pos': (x * x_scale, y * y_scale), 'size': size, 'brush': brush, 'pen': pg.mkPen(None)})
                scatter = pg.ScatterPlotItem(size=8)
                scatter.addPoints(spots)
                scatter.setZValue(-10)
                self.grid_plot.addItem(scatter)

            # Plot traces on top for relevant channels
            relevant_chans = np.where(ptp > 0.05 * max_ptp)[0]
            for ch in relevant_chans:
                x, y = pos[ch]
                trace = template[:, ch]
                trace_scaled = (trace / max_ptp) * 20
                t_offset = np.linspace(-10, 10, len(trace))
                self.grid_plot.plot(x * x_scale + t_offset, y * y_scale + trace_scaled,
                                    pen=pg.mkPen('#00e6a0', width=1.2))

        # --- Data Prep for Metrics ---
        spikes = dm.get_cluster_spikes(cluster_id)
        if len(spikes) < 2: return
        sr = dm.sampling_rate

        # --- 2. Autocorrelation (Purple) ---
        self.acg_plot.clear()
        # Bin spikes at 1ms
        spikes_ms = (spikes / sr * 1000).astype(int)
        if len(spikes_ms) > 0:
            duration = spikes_ms[-1]
            bins = np.arange(0, duration + 1, 1)
            binned, _ = np.histogram(spikes_ms, bins=bins)

            # Compute ACG via FFT (Fast)
            # Pad to power of 2 for speed
            n = 1 << (len(binned) * 2 - 1).bit_length()
            ft = np.fft.rfft(binned, n)
            acg = np.fft.irfft(ft * np.conj(ft))
            # Keep center 100ms
            acg = acg[:100]
            acg[0] = 0 # Remove zero-lag peak

            self.acg_plot.plot(np.arange(101), acg, fillLevel=0, stepMode=True,
                               brush=(170, 0, 255, 100), pen='#aa00ff')

        # --- 3. ISI (Blue Step + Filled) ---
        self.isi_plot.clear()
        isi_ms = np.diff(np.sort(spikes)) / sr * 1000

        # Determine which view to show based on the combo box
        current_isi_view = self.isi_view_combo.currentText()

        if current_isi_view == 'ISI Histogram':
            # Original histogram view
            y, x = np.histogram(isi_ms, bins=np.linspace(0, 50, 101))

            self.isi_plot.plot(x, y, stepMode="center", fillLevel=0,
                               brush=(0, 163, 224, 150),
                               pen=pg.mkPen('#33b5e5', width=2))
            self.isi_plot.addItem(pg.InfiniteLine(ISI_REFRACTORY_PERIOD_MS, angle=90,
                                                  pen=pg.mkPen('r', style=Qt.DashLine)))
            self.isi_plot.setLabel('bottom', "ISI (ms)")
            self.isi_plot.setLabel('left', "Count")
        elif current_isi_view == 'ISI vs Amplitude':
            # ISI vs amplitude scatter plot for attenuation analysis
            spike_indices = dm.get_cluster_spike_indices(cluster_id)
            all_amplitudes = dm.get_cluster_spike_amplitudes(cluster_id)

            if len(isi_ms) > 0 and len(all_amplitudes) > 1:
                # The ISI[i] corresponds to the time between spike[i] and spike[i+1]
                # To correlate with amplitudes, we'll use the amplitude of the second spike in each ISI pair
                if len(all_amplitudes) > len(isi_ms):
                    # Use amplitudes from index 1 to len(isi_ms)+1 to match ISI pairs
                    amplitude_values = all_amplitudes[1:len(isi_ms)+1] if len(isi_ms) < len(all_amplitudes) else all_amplitudes[1:]
                else:
                    # If we don't have enough amplitudes for all ISI pairs, truncate ISI
                    min_len = min(len(isi_ms), len(all_amplitudes)-1)
                    isi_ms = isi_ms[:min_len]
                    amplitude_values = all_amplitudes[1:min_len+1] if min_len > 0 else np.array([])

                if len(amplitude_values) > 0 and len(isi_ms) > 0:
                    # Create scatter plot
                    self.isi_plot.plot(isi_ms, amplitude_values,
                                     pen=None,
                                     symbol='o',
                                     symbolSize=5,
                                     symbolPen=None,
                                     symbolBrush=pg.mkBrush(255, 165, 0, 150))  # Orange color
                    self.isi_plot.setLabel('bottom', "ISI (ms)")
                    self.isi_plot.setLabel('left', "Amplitude")
                else:
                    # If we can't match ISI and amplitude data, plot empty
                    self.isi_plot.plot([], [], pen=None, symbol='o', symbolSize=5)
                    self.isi_plot.setLabel('bottom', "ISI (ms)")
                    self.isi_plot.setLabel('left', "Amplitude")
            else:
                # If not enough data, plot empty
                self.isi_plot.plot([], [], pen=None, symbol='o', symbolSize=5)
                self.isi_plot.setLabel('bottom', "ISI (ms)")
                self.isi_plot.setLabel('left', "Amplitude")

        # --- 4. Firing Rate (Yellow Smooth) ---
        self.fr_plot.clear()
        spikes_sec = spikes / sr
        bins = np.arange(0, spikes_sec.max() + 1, 1)
        counts, _ = np.histogram(spikes_sec, bins=bins)
        rate = gaussian_filter1d(counts.astype(float), sigma=5) # Smooth

        self.fr_plot.plot(bins[:-1], rate, pen=pg.mkPen('#ffeb3b', width=2))
        self.fr_plot.setLabel('bottom', "Time (s)")
        self.fr_plot.setLabel('left', "Hz")