import numpy as np
import pyqtgraph as pg
from qtpy.QtWidgets import QWidget, QVBoxLayout, QSplitter, QHBoxLayout, QCheckBox, QComboBox, QLabel
from qtpy.QtCore import Qt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate
from scipy.interpolate import interp1d
from analysis.constants import ISI_REFRACTORY_PERIOD_MS

# Configure pyqtgraph for antialiasing
pg.setConfigOptions(antialias=True)

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
        
        # Controls: main channel toggle
        ctrl_bar = QHBoxLayout()
        # Set fixed height for control bar for consistency during rapid scrolling
        ctrl_bar_widget = QWidget()
        ctrl_bar_widget.setMaximumHeight(35)
        ctrl_bar_widget.setLayout(ctrl_bar)
        
        self.main_channel_checkbox = QCheckBox('Show Main Channel Only')
        self.main_channel_checkbox.setChecked(False)
        ctrl_bar.addWidget(self.main_channel_checkbox)
        
        ctrl_bar.addStretch()

        # connect controls to refresh the panel immediately
        self.main_channel_checkbox.toggled.connect(self._on_control_changed)

        layout.addWidget(ctrl_bar_widget)

        # 2x2 Layout using Splitters
        self.vert_splitter = QSplitter(Qt.Vertical)
        layout.addWidget(self.vert_splitter)

        self.top_splitter = QSplitter(Qt.Horizontal)
        self.bottom_splitter = QSplitter(Qt.Horizontal)
        self.vert_splitter.addWidget(self.top_splitter)
        self.vert_splitter.addWidget(self.bottom_splitter)

        # 1. Template Grid (Top Left) with checkbox
        template_container = QWidget()
        template_layout = QVBoxLayout(template_container)
        template_layout.setContentsMargins(0, 0, 0, 0)
        
        template_controls = QHBoxLayout()
        self.heatmap_checkbox = QCheckBox('Show Channels Up Close')
        self.heatmap_checkbox.setChecked(True)
        template_controls.addWidget(self.heatmap_checkbox)
        template_controls.addStretch()
        template_layout.addLayout(template_controls)
        
        self.grid_widget = pg.GraphicsLayoutWidget()
        self.grid_plot = self.grid_widget.addPlot(title="Spatial Template")
        self.grid_plot.setAspectLocked(True)
        self.grid_plot.hideAxis('bottom')
        self.grid_plot.hideAxis('left')
        template_layout.addWidget(self.grid_widget)
        self.top_splitter.addWidget(template_container)
        
        self.heatmap_checkbox.toggled.connect(self._on_control_changed)

        # 2. Autocorrelation (Top Right)
        self.acg_plot = pg.PlotWidget(title="Autocorrelation")
        self.acg_plot.setLabel('bottom', "Lag (ms)")
        self._style_plot(self.acg_plot)
        self.top_splitter.addWidget(self.acg_plot)

        # 3. ISI (Bottom Left) with dropdown
        isi_container = QWidget()
        isi_layout = QVBoxLayout(isi_container)
        isi_layout.setContentsMargins(0, 0, 0, 0)
        
        isi_controls = QHBoxLayout()
        self.isi_view_combo = QComboBox()
        self.isi_view_combo.addItems(['ISI Histogram', 'ISI vs Amplitude'])
        isi_controls.addWidget(QLabel('View:'))
        isi_controls.addWidget(self.isi_view_combo)
        isi_controls.addStretch()
        isi_layout.addLayout(isi_controls)
        
        self.isi_plot = pg.PlotWidget(title="ISI Distribution")
        self.isi_plot.setLabel('bottom', "ISI (ms)")
        self._style_plot(self.isi_plot)
        isi_layout.addWidget(self.isi_plot)
        self.bottom_splitter.addWidget(isi_container)
        
        self.isi_view_combo.currentTextChanged.connect(self._on_control_changed)

        # 4. Firing Rate (Bottom Right) with dual-axis for amplitude
        self.fr_plot = pg.PlotWidget(title="Signal Health")
        self.fr_plot.setLabel('bottom', "Time (s)")
        self.fr_plot.setLabel('left', "Firing Rate (Hz)", color='#ffeb3b')
        self._style_plot(self.fr_plot)
        
        # Create a secondary ViewBox for amplitude on the right axis
        self.fr_viewbox = self.fr_plot.plotItem.getViewBox()
        self.fr_viewbox_right = pg.ViewBox()
        self.fr_plot.plotItem.scene().addItem(self.fr_viewbox_right)
        # Link the right viewbox to the main one for synchronized panning/zooming on X-axis
        self.fr_viewbox_right.linkView(pg.ViewBox.XAxis, self.fr_viewbox)
        
        # Create right axis
        self.fr_axis_right = pg.AxisItem(orientation='right')
        self.fr_axis_right.linkToView(self.fr_viewbox_right)
        self.fr_plot.plotItem.layout.addItem(self.fr_axis_right, 2, 3)
        self.fr_axis_right.setLabel('Amplitude (µV)', color='#ffd700')
        
        # Handle axis label color
        self.fr_axis_right.setPen(pg.mkPen('#888888'))
        self.fr_axis_right.setTextPen(pg.mkPen('#888888'))
        
        self.bottom_splitter.addWidget(self.fr_plot)

        self.vert_splitter.setSizes([500, 300])
        # ensure controls reflect current selection on init
        try:
            self._on_control_changed()
        except Exception:
            pass
    
    def _style_plot(self, plot):
        """Apply consistent styling to plots: grid lines and axis colors."""
        plot.showGrid(x=True, y=True, alpha=0.2)
        
        # Style axis colors
        plot.getAxis('bottom').setPen(pg.mkPen('#888888'))
        plot.getAxis('bottom').setTextPen(pg.mkPen('#888888'))
        plot.getAxis('left').setPen(pg.mkPen('#888888'))
        plot.getAxis('left').setTextPen(pg.mkPen('#888888'))
    
    def _create_hot_colormap(self):
        """Create a 'hot'-like colormap: black -> red -> yellow -> white."""
        # Define the colormap: black -> red -> yellow -> white
        colors = [
            (0, 0, 0),        # black
            (255, 0, 0),      # red
            (255, 255, 0),    # yellow
            (255, 255, 255)   # white
        ]
        positions = [0, 0.33, 0.66, 1.0]
        cmap = pg.ColorMap(pos=positions, color=colors)
        return cmap.getLookupTable(start=0.0, stop=1.0, nPts=256)

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
            # Add subtle dark border/shadow to green traces for visibility pop
            relevant_chans = np.where(ptp > 0.05 * max_ptp)[0]
            for ch in relevant_chans:
                x, y = pos[ch]
                trace = template[:, ch]
                trace_scaled = (trace / max_ptp) * 20
                t_offset = np.linspace(-10, 10, len(trace))
                
                # Draw shadow/border with darker color behind the main trace
                self.grid_plot.plot(x * x_scale + t_offset, y * y_scale + trace_scaled,
                                    pen=pg.mkPen('#00331f', width=2.5), alpha=0.6)
                # Draw main bright green trace on top
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
            # ISI vs amplitude as scatter plot with mean line
            spike_indices = dm.get_cluster_spike_indices(cluster_id)
            all_amplitudes = dm.get_cluster_spike_amplitudes(cluster_id)

            if len(isi_ms) > 0 and len(all_amplitudes) > 1:
                # Align ISI and amplitude data
                if len(all_amplitudes) > len(isi_ms):
                    amplitude_values = all_amplitudes[1:len(isi_ms)+1]
                else:
                    min_len = min(len(isi_ms), len(all_amplitudes)-1)
                    isi_ms = isi_ms[:min_len]
                    amplitude_values = all_amplitudes[1:min_len+1]

                if len(amplitude_values) > 10 and len(isi_ms) > 10:
                    # Scatter plot for ISI vs amplitude
                    self.isi_plot.plot(
                        isi_ms, amplitude_values,
                        pen=None, symbol='o', symbolSize=5, symbolBrush=(255, 165, 0, 150),
                        name="ISI vs Amplitude Scatter"
                    )

                    # Update labels
                    self.isi_plot.setLabel('bottom', "ISI (ms)")
                    self.isi_plot.setLabel('left', "Amplitude (µV)")
                else:
                    # Not enough data for scatter plot
                    self.isi_plot.plot([], [], pen=None, symbol='o', symbolSize=5)
                    self.isi_plot.setLabel('bottom', "ISI (ms)")
                    self.isi_plot.setLabel('left', "Amplitude (µV)")
            else:
                # If not enough data, plot empty
                self.isi_plot.plot([], [], pen=None, symbol='o', symbolSize=5)
                self.isi_plot.setLabel('bottom', "ISI (ms)")
                self.isi_plot.setLabel('left', "Amplitude (µV)")

        # --- 4. Firing Rate (Yellow) + Amplitude (Gold) Dual-Axis ---
        self.fr_plot.clear()
        spikes_sec = spikes / sr
        bins = np.arange(0, spikes_sec.max() + 1, 1)
        counts, bin_edges = np.histogram(spikes_sec, bins=bins)
        rate = gaussian_filter1d(counts.astype(float), sigma=5)  # Smooth for firing rate
        bin_centers = bin_edges[:-1]
        
        # Plot yellow firing rate line on left axis
        self.fr_plot.plot(bin_centers, rate, pen=pg.mkPen('#ffeb3b', width=2))
        
        # --- Compute and plot amplitude on dual-axis (Gold line) ---
        all_amplitudes = dm.get_cluster_spike_amplitudes(cluster_id)
        
        if len(all_amplitudes) > 0 and len(spikes) > 0:
            # Bin amplitudes into same 1-second bins as firing rate
            amplitude_binned = []
            
            for bin_idx in range(len(bin_centers)):
                bin_start = bin_centers[bin_idx]
                bin_end = bin_start + 1.0  # 1-second bin
                # Use spikes_sec (already in seconds) for binning
                mask = (spikes_sec >= bin_start) & (spikes_sec < bin_end)
                
                if np.any(mask):
                    # Average amplitude in this bin
                    amplitude_binned.append(np.mean(all_amplitudes[mask]))
                else:
                    # Empty bin - use NaN
                    amplitude_binned.append(np.nan)
            
            amplitude_binned = np.array(amplitude_binned)
            
            # Interpolate NaN values
            if np.any(np.isnan(amplitude_binned)):
                valid_idx = ~np.isnan(amplitude_binned)
                if np.sum(valid_idx) > 1:
                    f = interp1d(bin_centers[valid_idx], amplitude_binned[valid_idx],
                                kind='linear', bounds_error=False, fill_value='extrapolate')
                    amplitude_binned = f(bin_centers)
                elif np.sum(valid_idx) == 1:
                    # Only one valid point, use it for all bins
                    amplitude_binned = np.full_like(amplitude_binned, amplitude_binned[valid_idx][0])
                else:
                    # No valid data
                    amplitude_binned = None
            
            if amplitude_binned is not None:
                # Apply gaussian smoothing to match firing rate smoothness
                amplitude_smoothed = gaussian_filter1d(amplitude_binned, sigma=5)
                
                # Sync Y-range: scale amplitude to match visual prominence of firing rate
                # Use the spatial template's max PTP as reference
                if cluster_id < dm.templates.shape[0]:
                    ptp = dm.templates[cluster_id].max(axis=0) - dm.templates[cluster_id].min(axis=0)
                    max_ptp = ptp.max() if ptp.size > 0 else 1.0
                else:
                    max_ptp = amplitude_smoothed.max() if amplitude_smoothed.max() > 0 else 1.0
                
                # Set the right-axis range to match the amplitude data
                self.fr_viewbox_right.setYRange(0, max_ptp * 1.1, padding=0)
                
                # Create a PlotDataItem for the amplitude line
                amp_curve = pg.PlotCurveItem(bin_centers, amplitude_smoothed,
                                            pen=pg.mkPen('#ffd700', width=3))  # Gold
                self.fr_viewbox_right.addItem(amp_curve)

            # Overlay averaged amplitude line on mean firing rate plot
            normalized_amplitudes = all_amplitudes / np.max(all_amplitudes) if np.max(all_amplitudes) > 0 else all_amplitudes

            if len(spikes) > 10:
                avg_amplitude = np.convolve(normalized_amplitudes, np.ones(10)/10, mode='valid')
                scaled_amplitude = avg_amplitude * 0.8 * np.max(rate)  # Scale amplitude below firing rate

                self.fr_plot.plot(
                    spikes_sec[:len(scaled_amplitude)], scaled_amplitude,
                    pen=pg.mkPen('#00FF00', width=2),  # Green line for averaged amplitude
                    name="Averaged Amplitude"
                )
        
        self.fr_plot.setLabel('bottom', "Time (s)")
        self.fr_plot.setLabel('left', "Firing Rate (Hz)", color='#ffeb3b')