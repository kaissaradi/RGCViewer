import numpy as np
import pyqtgraph as pg
from qtpy.QtWidgets import QWidget, QVBoxLayout, QSplitter, QHBoxLayout, QCheckBox, QComboBox, QLabel, QPushButton, QDoubleSpinBox
from qtpy.QtCore import Qt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate
from scipy.interpolate import interp1d
from analysis.constants import ISI_REFRACTORY_PERIOD_MS
ISI_DENSITY_THRESHOLD = 5000  # switch to density view when > this many ISIs


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

        # Controls: top control bar
        ctrl_bar = QHBoxLayout()
        # Set fixed height for control bar for consistency during rapid scrolling
        ctrl_bar_widget = QWidget()
        ctrl_bar_widget.setMaximumHeight(35)
        ctrl_bar_widget.setLayout(ctrl_bar)

        # Channel display mode
        ctrl_bar.addWidget(QLabel('Channel Display:'))
        self.channel_mode_combo = QComboBox()
        self.channel_mode_combo.addItems(['Main Channel', 'Top Channels', 'Whole Array'])
        ctrl_bar.addWidget(self.channel_mode_combo)

        ctrl_bar.addStretch()

        # connect controls to refresh the panel immediately
        self.channel_mode_combo.currentTextChanged.connect(self._on_control_changed)

        layout.addWidget(ctrl_bar_widget)

        # 2x2 Layout using Splitters
        self.vert_splitter = QSplitter(Qt.Vertical)
        layout.addWidget(self.vert_splitter)

        self.top_splitter = QSplitter(Qt.Horizontal)
        self.bottom_splitter = QSplitter(Qt.Horizontal)
        self.vert_splitter.addWidget(self.top_splitter)
        self.vert_splitter.addWidget(self.bottom_splitter)

        # 1. Template Grid (Top Left) - no more checkbox
        template_container = QWidget()
        template_layout = QVBoxLayout(template_container)
        template_layout.setContentsMargins(0, 0, 0, 0)

        self.grid_widget = pg.GraphicsLayoutWidget()
        self.grid_plot = self.grid_widget.addPlot(title="Spatial Template")
        self.grid_plot.setAspectLocked(True)
        self.grid_plot.hideAxis('bottom')
        self.grid_plot.hideAxis('left')
        template_layout.addWidget(self.grid_widget)
        self.top_splitter.addWidget(template_container)

        # 2. Autocorrelation (Top Right)
        self.acg_plot = pg.PlotWidget(title="Autocorrelation")
        self.acg_plot.setLabel('bottom', "Time lag (ms)")
        self.acg_plot.setLabel('left', "Autocorrelation")
        self._style_plot(self.acg_plot)
        self.top_splitter.addWidget(self.acg_plot)
        # 3. ISI (Bottom Left) with dropdown
        isi_container = QWidget()
        isi_layout = QVBoxLayout(isi_container)
        isi_layout.setContentsMargins(0, 0, 0, 0)

        isi_controls = QHBoxLayout()

        # --- View selector ---
        isi_controls.addWidget(QLabel('View:'))
        self.isi_view_combo = QComboBox()
        self.isi_view_combo.addItems(['ISI Histogram', 'ISI vs Amplitude'])
        isi_controls.addWidget(self.isi_view_combo)

        # --- Refractory controls (short labels to keep bar compact) ---
        self.show_refractory_line_checkbox = QCheckBox('Refr line')
        self.show_refractory_line_checkbox.setChecked(True)
        isi_controls.addWidget(self.show_refractory_line_checkbox)

        isi_controls.addWidget(QLabel('Ref (ms):'))
        self.refractory_spinbox = QDoubleSpinBox()
        self.refractory_spinbox.setRange(0.1, 10.0)
        self.refractory_spinbox.setDecimals(2)
        self.refractory_spinbox.setSingleStep(0.1)
        self.refractory_spinbox.setValue(ISI_REFRACTORY_PERIOD_MS)
        isi_controls.addWidget(self.refractory_spinbox)

        self.update_refractory_btn = QPushButton('Set')
        isi_controls.addWidget(self.update_refractory_btn)

        # --- ISI display mode + X-range presets ---
        isi_controls.addWidget(QLabel('Plot:'))
        self.isi_display_combo = QComboBox()
        self.isi_display_combo.addItems(['Scatter', 'Density'])
        self.isi_display_combo.setCurrentText('Scatter')
        isi_controls.addWidget(self.isi_display_combo)

        isi_controls.addWidget(QLabel('X:'))
        self.isi_range_combo = QComboBox()
        self.isi_range_combo.addItems(['0–50 ms', '0–500 ms', 'Full'])
        self.isi_range_combo.setCurrentText('0–500 ms')
        isi_controls.addWidget(self.isi_range_combo)

        isi_controls.addStretch()
        isi_layout.addLayout(isi_controls)

        self.isi_plot = pg.PlotWidget(title="ISI Distribution")
        self.isi_plot.setLabel('bottom', "ISI (ms)")
        self._style_plot(self.isi_plot)
        isi_layout.addWidget(self.isi_plot)
        self.bottom_splitter.addWidget(isi_container)

        # ISI controls update the plot
        self.isi_view_combo.currentTextChanged.connect(self._on_control_changed)
        self.show_refractory_line_checkbox.stateChanged.connect(self._on_control_changed)
        self.update_refractory_btn.clicked.connect(self._update_refractory_period)
        self.isi_display_combo.currentTextChanged.connect(self._on_control_changed)
        self.isi_range_combo.currentTextChanged.connect(self._on_control_changed)

        # Colormap for density view
        self._hot_lut = self._create_hot_colormap()


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

    def _update_refractory_period(self):
        """Update the refractory period in the data manager and refresh the plot."""
        new_period = self.refractory_spinbox.value()
        self.main_window.data_manager.set_refractory_period(new_period)
        # Refresh the current plot to show the updated refractory line
        cluster_id = self.main_window._get_selected_cluster_id()
        if cluster_id is not None:
            self.update_all(cluster_id)

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
    
    def _apply_isi_range_preset(self, isi_ms):
        """Apply the X-range preset for the ISI plot."""
        if len(isi_ms) == 0:
            return

        preset = self.isi_range_combo.currentText()
        if preset == '0–50 ms':
            self.isi_plot.setXRange(0.0, 50.0, padding=0)
        elif preset == '0–500 ms':
            self.isi_plot.setXRange(0.0, 500.0, padding=0)
        else:  # 'Full'
            x_max = float(isi_ms.max())
            self.isi_plot.setXRange(0.0, x_max * 1.05, padding=0)


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
        if not self.isVisible(): return
        if cluster_id is None: return
        dm = self.main_window.data_manager

        # --- Clear similarity panel selection if main cluster changed ---
        sim_panel = getattr(self.main_window, 'similarity_panel', None)
        if sim_panel is not None:
            # Check if this is a different main cluster than before
            if hasattr(sim_panel, 'main_cluster_id') and sim_panel.main_cluster_id != cluster_id:
                # Clear selection in similarity table
                if sim_panel.table and sim_panel.table.selectionModel():
                    sim_panel.table.selectionModel().clearSelection()

        # --- 1. Template Grid (Spatial) ---
        self.grid_plot.clear()
        if hasattr(dm, 'templates') and cluster_id < dm.templates.shape[0]:
            template = dm.templates[cluster_id] # (n_time, n_chan)
            pos = dm.channel_positions
            x_scale, y_scale = 1.5, 1.0

            # Per-channel peak-to-peak
            ptp = template.max(axis=0) - template.min(axis=0)
            max_ptp = ptp.max() if ptp.size>0 else 1.0

            # Find the main/dominant channel (highest PTP)
            main_channel_idx = np.argmax(ptp) if ptp.size > 0 else 0

            # Determine channels to show based on channel mode selection
            current_mode = self.channel_mode_combo.currentText()
            if current_mode == 'Main Channel':
                # Show ONLY the main/dominant channel
                relevant_channels = [main_channel_idx]
                # For single channel, show waveform but no dots (not meaningful)
                show_waveforms = True
                show_dots = False
                waveform_channels = [main_channel_idx]  # Waveforms for this channel
            elif current_mode == 'Top Channels':
                # Show top 3 channels with highest PTP
                top_channel_indices = np.argsort(ptp)[::-1][:3]  # Get indices of top 3 channels
                relevant_channels = top_channel_indices
                # Show waveforms and dots for these top channels
                show_waveforms = True
                show_dots = True
                waveform_channels = top_channel_indices  # Waveforms for these channels
            else:  # 'Whole Array'
                # Show ALL channels as dots (for spatial distribution visualization)
                # Show waveforms for top 6 channels with highest PTP
                relevant_channels = np.arange(len(ptp))  # All channels for dots
                top_waveform_channels = np.argsort(ptp)[::-1][:6]  # Top 6 for waveforms
                waveform_channels = top_waveform_channels  # Channels to show waveforms for
                show_waveforms = True  # Show waveforms for top channels
                show_dots = True  # And show dots for all channels

            # Cluster amplitude (using mean as default since mean/median was removed)
            cluster_amp = dm.get_cluster_mean_amplitude(cluster_id, method='mean')

            # Channel values = normalized ptp * cluster amplitude
            if max_ptp == 0:
                norm_ptp = np.zeros_like(ptp)
            else:
                norm_ptp = ptp / max_ptp
            channel_values = norm_ptp * cluster_amp

            # Draw amplitude-weighted scatter under traces based on mode
            if show_dots:
                # Build scatter spots
                spots = []
                vmin, vmax = channel_values.min(), channel_values.max()
                vrange = max(vmax - vmin, 1e-6)
                for ch in relevant_channels:  # Only show relevant channels
                    if ch >= len(channel_values): continue  # Safety check
                    x, y = pos[ch]
                    val = (channel_values[ch] - vmin) / vrange
                    size = 6 + val * 20
                    # color from cool->hot
                    r = int(255 * val)
                    g = int(120 * (1 - val))
                    b = int(255 * (1 - val))
                    brush = pg.mkBrush(r, g, b, 180)
                    spots.append({'pos': (x * x_scale, y * y_scale), 'size': size, 'brush': brush, 'pen': pg.mkPen(None)})
                if spots:  # Only add scatter if there are spots to plot
                    scatter = pg.ScatterPlotItem(size=8)
                    scatter.addPoints(spots)
                    scatter.setZValue(-10)
                    self.grid_plot.addItem(scatter)

            # Plot traces on top for waveform channels (not necessarily all channels)
            # Add subtle dark border/shadow to green traces for visibility pop
            for ch in waveform_channels:  # Only plot waveforms for specified channels
                if ch >= len(pos): continue  # Safety check
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

            # Add horizontal line at y=0 for the main channel template view
            if current_mode == 'Main Channel' and len(waveform_channels) > 0:
                # Create a horizontal line at y=0 voltage level
                # Since templates are centered around 0 voltage, this is a horizontal line at y=0 in the plot
                # We need to account for the y position offset we added when plotting the template
                main_x, main_y = pos[main_channel_idx]
                # The template is plotted at y * y_scale + trace_scaled, where trace_scaled has the voltage values
                # So the y=0 voltage line should be at y * y_scale level
                h_line = pg.InfiniteLine(pos=main_y * y_scale, angle=0,
                                         pen=pg.mkPen('#ffffff', width=1, style=Qt.DashLine))
                self.grid_plot.addItem(h_line)

            # After plotting main cluster template, add similar cluster overlays (if similarity panel exists)
            sim_panel = getattr(self.main_window, 'similarity_panel', None)
            selected_similar = []
            if sim_panel is not None and getattr(sim_panel, 'table', None) is not None:
                sel_model = sim_panel.table.selectionModel()
                if sel_model is not None:
                    selected_similar = sel_model.selectedRows()

            if selected_similar and hasattr(sim_panel, 'similarity_model'):
                sim_model = sim_panel.similarity_model
                for idx in selected_similar[:3]:  # Limit to 3 similar clusters
                    similar_id = sim_model._dataframe.iloc[idx.row()]['cluster_id']
                    if similar_id < dm.templates.shape[0]:
                        sim_template = dm.templates[similar_id]
                        sim_ptp = sim_template.max(axis=0) - sim_template.min(axis=0)
                        # Plot in semi-transparent orange/yellow
                        for ch in waveform_channels:
                            if ch >= len(pos): continue
                            x, y = pos[ch]
                            trace = sim_template[:, ch]
                            trace_scaled = (trace / max_ptp) * 20
                            t_offset = np.linspace(-10, 10, len(trace))
                            self.grid_plot.plot(x * x_scale + t_offset, y * y_scale + trace_scaled,
                                                pen=pg.mkPen('#ff9800', width=1.5))


        # --- Data Prep for Metrics ---
        spikes = dm.get_cluster_spikes(cluster_id)
        if len(spikes) < 2: return
        sr = dm.sampling_rate

        # --- 2. Cross-Correlogram or Autocorrelation ---
        self.acg_plot.clear()

        # Safely get selected similar clusters from the similarity panel (if present)
        sim_panel = getattr(self.main_window, 'similarity_panel', None)
        selected_similar = []
        valid_similar_id = None

        if sim_panel is not None and getattr(sim_panel, 'table', None) is not None:
            sel_model = sim_panel.table.selectionModel()
            if sel_model is not None:
                selected_similar = sel_model.selectedRows()

        # If a similar cluster is selected, validate it belongs to current main cluster's similarity table
        if selected_similar and hasattr(sim_panel, 'similarity_model'):
            sim_model = sim_panel.similarity_model
            
            # Check if the model exists and the selected row is valid
            if sim_model is not None and selected_similar[0].row() < len(sim_model._dataframe):
                similar_id = sim_model._dataframe.iloc[selected_similar[0].row()]['cluster_id']
                
                # CRITICAL FIX: Check if similar_id is in the current similarity table
                # Also check it's not the main cluster itself and is a valid cluster
                if (similar_id != cluster_id and 
                    similar_id < dm.templates.shape[0] and
                    hasattr(sim_panel, 'main_cluster_id') and 
                    sim_panel.main_cluster_id == cluster_id):  # This ensures we're looking at the right table
                    
                    # Get spike trains for both clusters
                    spikes1 = dm.get_cluster_spikes(cluster_id)
                    spikes2 = dm.get_cluster_spikes(similar_id)

                    # Only proceed if both clusters have >1 spike
                    if len(spikes1) > 1 and len(spikes2) > 1:
                        # Convert spike times to ms
                        spikes1_ms = (spikes1 / sr * 1000).astype(int)
                        spikes2_ms = (spikes2 / sr * 1000).astype(int)

                        duration = max(spikes1_ms[-1], spikes2_ms[-1]) if len(spikes1_ms) > 0 and len(spikes2_ms) > 0 else 0

                        if duration > 0:
                            bin_width_ms = 1
                            bins = np.arange(0, duration + bin_width_ms, bin_width_ms)

                            binned1, _ = np.histogram(spikes1_ms, bins=bins)
                            binned2, _ = np.histogram(spikes2_ms, bins=bins)

                            centered1 = binned1 - np.mean(binned1)
                            centered2 = binned2 - np.mean(binned2)

                            ccg_full = correlate(centered1, centered2, mode='full')
                            zero_lag_idx = len(ccg_full) // 2
                            max_lag_ms = 100
                            num_bins = int(max_lag_ms / bin_width_ms)
                            lag_range = min(num_bins, zero_lag_idx)

                            ccg_symmetric = ccg_full[zero_lag_idx - lag_range : zero_lag_idx + lag_range + 1]
                            time_lags = np.arange(-lag_range, lag_range + 1) * bin_width_ms

                            # Normalize if possible
                            variance = np.sqrt(np.var(binned1) * np.var(binned2))
                            if variance != 0:
                                ccg_norm = ccg_symmetric / variance / len(binned1)
                            else:
                                ccg_norm = ccg_symmetric

                            # Draw CCG as bars
                            bar_graph = pg.BarGraphItem(
                                x=time_lags,
                                height=ccg_norm,
                                width=0.8,
                                brush=(255, 152, 0, 100),
                                pen=pg.mkPen('#ff9800', width=1),
                            )
                            self.acg_plot.addItem(bar_graph)

                            self.acg_plot.setTitle(f"CCG: {cluster_id} vs {similar_id}")

                            # Add a vertical zero-lag line
                            zero_line = pg.InfiniteLine(
                                pos=0, angle=90,
                                pen=pg.mkPen('#ffffff', width=2, style=Qt.DashLine)
                            )
                            self.acg_plot.addItem(zero_line)
                            
                            # Skip the autocorrelation code below
                            return

        # If we get here, either no valid similar cluster is selected or CCG couldn't be computed
        # So draw the autocorrelation instead

        # Autocorrelation for a single cluster (original logic)
        # Convert spike times to ms
        spikes_ms = (spikes / sr * 1000).astype(int)

        if len(spikes_ms) > 1:
            duration = spikes_ms[-1]
            if duration > 0:
                bin_width_ms = 1
                bins = np.arange(0, duration + bin_width_ms, bin_width_ms)
                binned_spikes, _ = np.histogram(spikes_ms, bins=bins)

                centered = binned_spikes - np.mean(binned_spikes)
                acg_full = correlate(centered, centered, mode='full')

                zero_lag_idx = len(acg_full) // 2
                max_lag_ms = 100
                num_bins = int(max_lag_ms / bin_width_ms)
                lag_range = min(num_bins, zero_lag_idx)

                acg_symmetric = acg_full[zero_lag_idx - lag_range : zero_lag_idx + lag_range + 1]
                time_lags = np.arange(-lag_range, lag_range + 1) * bin_width_ms

                # Zero out the central peak so refractory effects are visible
                zero_idx = np.where(time_lags == 0)[0]
                if len(zero_idx) > 0:
                    acg_symmetric[zero_idx[0]] = 0

                # Normalize by variance and length
                spike_variance = np.var(binned_spikes)
                if spike_variance != 0:
                    acg_norm = acg_symmetric / spike_variance / len(binned_spikes)
                else:
                    acg_norm = acg_symmetric

                # Draw ACG as bars
                bar_graph = pg.BarGraphItem(
                    x=time_lags,
                    height=acg_norm,
                    width=0.8,
                    brush=(170, 0, 255, 100),
                    pen=pg.mkPen('#aa00ff', width=1),
                )
                self.acg_plot.addItem(bar_graph)

                # Vertical zero-lag line
                zero_line = pg.InfiniteLine(
                    pos=0, angle=90,
                    pen=pg.mkPen('#ffffff', width=2, style=Qt.DashLine)
                )
                self.acg_plot.addItem(zero_line)

        # --- 3. ISI (Blue Step + Filled) ---
        self.isi_plot.clear()
        isi_ms = np.diff(np.sort(spikes)) / sr * 1000

        # Determine which view to show based on the combo box
        current_isi_view = self.isi_view_combo.currentText()

        if current_isi_view == 'ISI Histogram':
            # Original histogram view with toggleable refractory line
            y, x = np.histogram(isi_ms, bins=np.linspace(0, 50, 101))

            self.isi_plot.plot(x, y, stepMode="center", fillLevel=0,
                               brush=(0, 163, 224, 150),
                               pen=pg.mkPen('#33b5e5', width=2))

            # Get current refractory period from data manager
            refractory_period = dm.get_refractory_period()

            # Conditionally add refractory period line based on checkbox
            if self.show_refractory_line_checkbox.isChecked():
                self.isi_plot.addItem(pg.InfiniteLine(refractory_period, angle=90,
                                                      pen=pg.mkPen('r', style=Qt.DashLine)))
            self.isi_plot.setLabel('bottom', "ISI (ms)")
            self.isi_plot.setLabel('left', "Count")
        elif current_isi_view == 'ISI vs Amplitude':
            # ISI vs amplitude view (scatter or 2D density)
            
            # Get amplitudes for this cluster
            all_amplitudes = dm.get_cluster_spike_amplitudes(cluster_id)
            
            if len(isi_ms) > 0 and len(all_amplitudes) > 1:
                # FIX: Align ISI and amplitude data correctly
                # ISI[i] is the interval between spike i and spike i+1
                # This should be paired with amplitude of spike i+1 (the second spike in the pair)
                # So we need amplitudes[1:] to match with isi_ms
                
                # Make sure we have compatible lengths
                min_len = min(len(isi_ms), len(all_amplitudes) - 1)
                
                if min_len > 0:
                    # Take only the valid pairs
                    valid_isi = isi_ms[:min_len]
                    # Amplitudes for spikes 1 through min_len+1 (since spike 0 doesn't have a preceding ISI)
                    valid_amplitudes = all_amplitudes[1:min_len + 1]
                    
                    # Auto-switch to density view when there are many ISIs
                    if len(valid_isi) > ISI_DENSITY_THRESHOLD and self.isi_display_combo.currentText() == 'Scatter':
                        self.isi_display_combo.setCurrentText('Density')
                    
                    display_mode = self.isi_display_combo.currentText()

                    if display_mode == 'Scatter':
                        # Classic scatter plot
                        self.isi_plot.plot(
                            valid_isi, valid_amplitudes,
                            pen=None,
                            symbol='o',
                            symbolSize=5,
                            symbolBrush=(255, 165, 0, 150),
                            name="ISI vs Amplitude Scatter"
                        )
                    else:  # 'Density'
                        # 2D histogram / heatmap
                        nbins_isi = 100
                        nbins_amp = 80

                        isi_min, isi_max = float(valid_isi.min()), float(valid_isi.max())
                        amp_min, amp_max = float(valid_amplitudes.min()), float(valid_amplitudes.max())

                        # Add small padding to avoid zero range
                        if isi_max <= isi_min:
                            isi_max = isi_min + 1e-3
                        if amp_max <= amp_min:
                            amp_max = amp_min + 1e-3

                        H, xedges, yedges = np.histogram2d(
                            valid_isi,
                            valid_amplitudes,
                            bins=[nbins_isi, nbins_amp],
                            range=[[isi_min, isi_max], [amp_min, amp_max]]
                        )

                        # Log-compress to keep dynamic range reasonable
                        H = np.log1p(H)
                        H = H.T                      # (y, x)
                        H = np.flipud(H)            # flip vertically so low amps at bottom, high at top

                        img = pg.ImageItem(H)
                        if hasattr(self, "_hot_lut") and self._hot_lut is not None:
                            img.setLookupTable(self._hot_lut)

                        rect = pg.QtCore.QRectF(
                            xedges[0],
                            yedges[0],
                            xedges[-1] - xedges[0],
                            yedges[-1] - yedges[0]
                        )
                        img.setRect(rect)
                        self.isi_plot.addItem(img)

                    # Common axis labels
                    self.isi_plot.setLabel('bottom', "ISI (ms)")
                    self.isi_plot.setLabel('left', "Amplitude (µV)")

                    # Apply X-range preset for this view
                    self._apply_isi_range_preset(valid_isi)
                else:
                    # Not enough valid pairs
                    self.isi_plot.plot([], [], pen=None, symbol='o', symbolSize=5)
                    self.isi_plot.setLabel('bottom', "ISI (ms)")
                    self.isi_plot.setLabel('left', "Amplitude (µV)")
            else:
                # If not enough data, plot empty
                self.isi_plot.plot([], [], pen=None, symbolSize=5)
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