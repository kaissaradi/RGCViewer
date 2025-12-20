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

        # --- persistent ACG bar item (reuse each update) ---
        self._acg_bar = None      # will be a BarGraphItem (or None if no data)
        self._acg_zero_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('#ffffff', width=2, style=Qt.DashLine))

        # --- persistent ISI objects ---
        self._isi_hist_item = None     # step histogram as PlotDataItem (stepMode implemented via x,y)
        self._isi_refractory_line = pg.InfiniteLine(0, angle=90, pen=pg.mkPen('r', style=Qt.DashLine))

        # --- persistent ISI vs Amp items ---
        self._isi_scatter = None
        self._isi_image = None

        # --- persistent FR items ---
        self._fr_rate_curve = self.fr_plot.plot([], [], pen=pg.mkPen('#ffeb3b', width=2), name='fr')
        self._fr_amp_curve = self.fr_plot.plot([], [], pen=pg.mkPen('#ffd700', width=1), name='amp')

        # keep a ref to image item LUT if needed
        if hasattr(self, '_hot_lut') and self._hot_lut is not None:
            self._hot_lut_local = self._hot_lut
        else:
            self._hot_lut_local = None


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
        """Update all standard plots for the given cluster.

        Uses DataManager.get_standard_plot_data so that heavy numeric work
        happens in the background worker, and the UI thread mostly just draws
        precomputed arrays. Behaviour stays compatible with the previous
        implementation.
        """
        # --- Basic guards ---
        if cluster_id is None:
            return

        dm = self.main_window.data_manager
        if dm is None:
            return

        # --- Clear similarity panel selection if main cluster changed ---
        sim_panel = getattr(self.main_window, 'similarity_panel', None)
        if sim_panel is not None:
            try:
                if hasattr(sim_panel, 'main_cluster_id') and sim_panel.main_cluster_id != cluster_id:
                    table = getattr(sim_panel, 'table', None)
                    sel_model = table.selectionModel() if table is not None else None
                    if sel_model is not None:
                        sel_model.clearSelection()
            except Exception:
                # Never let similarity panel errors break the main UI
                pass

        # ------------------------------------------------------------------
        # 1. TEMPLATE GRID (SPATIAL)
        # ------------------------------------------------------------------
        self.grid_plot.clear()
        try:
            if hasattr(dm, 'templates') and dm.templates is not None and cluster_id < dm.templates.shape[0]:
                template = dm.templates[cluster_id]  # (n_time, n_chan)
                pos = dm.channel_positions
                x_scale, y_scale = 1.5, 1.0

                # Per-channel peak-to-peak
                ptp = template.max(axis=0) - template.min(axis=0)
                max_ptp = ptp.max() if ptp.size > 0 else 1.0

                # Main / dominant channel
                main_channel_idx = int(np.argmax(ptp)) if ptp.size > 0 else 0

                # Channel mode selection
                current_mode = self.channel_mode_combo.currentText()
                if current_mode == 'Main Channel':
                    # Only dominant channel
                    relevant_channels = [main_channel_idx]
                    show_waveforms = True
                    show_dots = False
                    waveform_channels = [main_channel_idx]
                elif current_mode == 'Top Channels':
                    # Top 3 channels by PTP
                    top_channel_indices = np.argsort(ptp)[::-1][:3]
                    relevant_channels = top_channel_indices
                    show_waveforms = True
                    show_dots = True
                    waveform_channels = top_channel_indices
                else:  # 'Whole Array'
                    # All channels as dots; waveforms for top N
                    relevant_channels = np.arange(len(ptp))
                    top_waveform_channels = np.argsort(ptp)[::-1][:6]
                    waveform_channels = top_waveform_channels
                    show_waveforms = True
                    show_dots = True

                # Cluster amplitude (mean)
                cluster_amp = dm.get_cluster_mean_amplitude(cluster_id, method='mean')

                # Channel values = normalized PTP * amplitude
                if max_ptp == 0:
                    norm_ptp = np.zeros_like(ptp)
                else:
                    norm_ptp = ptp / max_ptp
                channel_values = norm_ptp * cluster_amp

                # Dots: amplitude-weighted scatter
                if show_dots:
                    spots = []
                    vmin, vmax = channel_values.min(), channel_values.max()
                    vrange = max(vmax - vmin, 1e-6)
                    for ch in relevant_channels:
                        if ch >= len(channel_values) or ch >= len(pos):
                            continue
                        x, y = pos[ch]
                        val = (channel_values[ch] - vmin) / vrange
                        size = 6 + val * 20
                        # Cool → hot
                        r = int(255 * val)
                        g = int(120 * (1 - val))
                        b = int(255 * (1 - val))
                        brush = pg.mkBrush(r, g, b, 180)
                        spots.append(
                            {
                                'pos': (x * x_scale, y * y_scale),
                                'size': size,
                                'brush': brush,
                                'pen': pg.mkPen(None),
                            }
                        )
                    if spots:
                        scatter = pg.ScatterPlotItem(size=8)
                        scatter.addPoints(spots)
                        scatter.setZValue(-10)
                        self.grid_plot.addItem(scatter)

                # Waveforms on top
                if show_waveforms:
                    for ch in waveform_channels:
                        if ch >= len(pos):
                            continue
                        x, y = pos[ch]
                        trace = template[:, ch]
                        trace_scaled = (trace / max_ptp) * 20 if max_ptp > 0 else trace
                        t_offset = np.linspace(-10, 10, len(trace))

                        # Dark border
                        self.grid_plot.plot(
                            x * x_scale + t_offset,
                            y * y_scale + trace_scaled,
                            pen=pg.mkPen('#00331f', width=2.5),
                            alpha=0.6,
                        )
                        # Bright green waveform
                        self.grid_plot.plot(
                            x * x_scale + t_offset,
                            y * y_scale + trace_scaled,
                            pen=pg.mkPen('#00e6a0', width=1.2),
                        )

                    # Zero line on main channel
                    if current_mode == 'Main Channel' and waveform_channels and main_channel_idx < len(pos):
                        _, main_y = pos[main_channel_idx]
                        h_line = pg.InfiniteLine(
                            pos=main_y * y_scale,
                            angle=0,
                            pen=pg.mkPen('#ffffff', width=1, style=Qt.DashLine),
                        )
                        self.grid_plot.addItem(h_line)

                # Similar-cluster overlays
                sim_panel = getattr(self.main_window, 'similarity_panel', None)
                selected_similar = []
                if sim_panel is not None and getattr(sim_panel, 'table', None) is not None:
                    sel_model = sim_panel.table.selectionModel()
                    if sel_model is not None:
                        selected_similar = sel_model.selectedRows()

                if selected_similar and hasattr(sim_panel, 'similarity_model'):
                    sim_model = sim_panel.similarity_model
                    if sim_model is not None and hasattr(sim_model, '_dataframe'):
                        for idx in selected_similar[:3]:  # up to 3 overlays
                            row = idx.row()
                            if row >= len(sim_model._dataframe):
                                continue
                            similar_id = sim_model._dataframe.iloc[row]['cluster_id']
                            if similar_id < dm.templates.shape[0]:
                                sim_template = dm.templates[similar_id]
                                for ch in waveform_channels:
                                    if ch >= len(pos):
                                        continue
                                    x, y = pos[ch]
                                    trace = sim_template[:, ch]
                                    trace_scaled = (trace / max_ptp) * 20 if max_ptp > 0 else trace
                                    t_offset = np.linspace(-10, 10, len(trace))
                                    self.grid_plot.plot(
                                        x * x_scale + t_offset,
                                        y * y_scale + trace_scaled,
                                        pen=pg.mkPen('#ff9800', width=1.5),
                                    )
        except Exception:
            # Never allow template-grid issues to break the rest of the panel
            pass

        # ------------------------------------------------------------------
        # 2. DATA PREP FOR METRICS (ACG / ISI / FR) – USE CACHED DATA
        # ------------------------------------------------------------------
        try:
            data = dm.get_standard_plot_data(cluster_id)
        except Exception:
            data = None

        spikes = None
        spikes_ms = None
        if data is not None:
            spikes = data.get('spikes')
            spikes_ms = data.get('spikes_ms')

        # Fallback in case spikes weren't cached (or older DM)
        if spikes is None:
            try:
                spikes = dm.get_cluster_spikes(cluster_id)
            except Exception:
                spikes = None

        sr = getattr(dm, 'sampling_rate', None)
        if spikes is None or len(spikes) < 2 or sr is None or sr <= 0:
            # Not enough data to show ACG/ISI/FR – leave previous plots alone.
            return

        if spikes_ms is None and spikes is not None:
            spikes_ms = (np.asarray(spikes) / float(sr) * 1000.0).astype(int)

        # ------------------------------------------------------------------
        # 3. CCG (IF SIMILAR CLUSTER SELECTED) ELSE ACG
        # ------------------------------------------------------------------
        self.acg_plot.clear()
        similar_id_valid = False

        sim_panel = getattr(self.main_window, 'similarity_panel', None)
        selected_similar = []
        try:
            if sim_panel is not None and getattr(sim_panel, 'table', None) is not None:
                sel_model = sim_panel.table.selectionModel()
                if sel_model is not None:
                    selected_similar = sel_model.selectedRows()
        except Exception:
            selected_similar = []

        if selected_similar and hasattr(sim_panel, 'similarity_model'):
            sim_model = sim_panel.similarity_model
            if sim_model is not None and hasattr(sim_model, '_dataframe'):
                first_idx = selected_similar[0]
                row = first_idx.row()
                if 0 <= row < len(sim_model._dataframe):
                    similar_id = int(sim_model._dataframe.iloc[row]['cluster_id'])

                    templates = getattr(dm, 'templates', None)
                    templates_n = templates.shape[0] if templates is not None else 0
                    if (
                        similar_id != cluster_id
                        and similar_id < templates_n
                        and hasattr(sim_panel, 'main_cluster_id')
                        and sim_panel.main_cluster_id == cluster_id
                    ):
                        # Get cached spikes for similar cluster, if available
                        try:
                            data2 = dm.get_standard_plot_data(similar_id)
                        except Exception:
                            data2 = None

                        spikes2_ms = None
                        if data2 is not None:
                            spikes2_ms = data2.get('spikes_ms')
                            if spikes2_ms is None:
                                spikes2 = data2.get('spikes')
                                if spikes2 is not None:
                                    spikes2_ms = (np.asarray(spikes2) / float(sr) * 1000.0).astype(int)

                        if spikes2_ms is None:
                            # Fallback: direct cluster spikes
                            try:
                                spikes2 = dm.get_cluster_spikes(similar_id)
                            except Exception:
                                spikes2 = None
                            if spikes2 is not None and len(spikes2) > 0:
                                spikes2_ms = (np.asarray(spikes2) / float(sr) * 1000.0).astype(int)

                        if (
                            spikes_ms is not None and spikes2_ms is not None
                            and len(spikes_ms) > 1 and len(spikes2_ms) > 1
                        ):
                            spikes1_ms = np.asarray(spikes_ms, dtype=int)
                            spikes2_ms = np.asarray(spikes2_ms, dtype=int)

                            duration = int(max(spikes1_ms[-1], spikes2_ms[-1]))
                            if duration > 0:
                                bin_width_ms = 1
                                bins = np.arange(0, duration + bin_width_ms, bin_width_ms, dtype=int)

                                binned1, _ = np.histogram(spikes1_ms, bins=bins)
                                binned2, _ = np.histogram(spikes2_ms, bins=bins)

                                if len(binned1) > 0 and len(binned2) > 0:
                                    centered1 = binned1 - np.mean(binned1)
                                    centered2 = binned2 - np.mean(binned2)

                                    ccg_full = correlate(centered1, centered2, mode='full')
                                    zero_lag_idx = len(ccg_full) // 2
                                    max_lag_ms = 100
                                    num_bins = int(max_lag_ms / bin_width_ms)
                                    lag_range = min(num_bins, zero_lag_idx)

                                    if lag_range > 0:
                                        ccg_symmetric = ccg_full[
                                            zero_lag_idx - lag_range : zero_lag_idx + lag_range + 1
                                        ]
                                        time_lags = np.arange(-lag_range, lag_range + 1) * bin_width_ms

                                        variance = np.sqrt(np.var(binned1) * np.var(binned2))
                                        if variance != 0:
                                            ccg_norm = ccg_symmetric / variance / len(binned1)
                                        else:
                                            ccg_norm = ccg_symmetric.astype(float)

                                        bar_graph = pg.BarGraphItem(
                                            x=time_lags,
                                            height=ccg_norm,
                                            width=0.8,
                                            brush=(255, 152, 0, 100),
                                            pen=pg.mkPen('#ff9800', width=1),
                                        )
                                        self.acg_plot.addItem(bar_graph)

                                        zero_line = pg.InfiniteLine(
                                            pos=0,
                                            angle=90,
                                            pen=pg.mkPen('#ffffff', width=2, style=Qt.DashLine),
                                        )
                                        self.acg_plot.addItem(zero_line)

                                        self.acg_plot.setTitle(f"CCG: {cluster_id} vs {similar_id}")
                                        similar_id_valid = True

        # Fallback: ACG from cache (or recompute once)
        if not similar_id_valid:
            time_lags = acg_norm = None
            if data is not None:
                time_lags = data.get('acg_time_lags')
                acg_norm = data.get('acg_norm')

            # Safety: recompute if cache missing
            if (time_lags is None or acg_norm is None) and spikes_ms is not None and len(spikes_ms) > 1:
                duration = int(spikes_ms[-1])
                if duration > 0:
                    bin_width_ms = 1
                    bins = np.arange(0, duration + bin_width_ms, bin_width_ms, dtype=int)
                    binned_spikes, _ = np.histogram(spikes_ms, bins=bins)

                    if len(binned_spikes) > 0:
                        centered = binned_spikes - np.mean(binned_spikes)
                        acg_full = correlate(centered, centered, mode='full')

                        zero_lag_idx = len(acg_full) // 2
                        max_lag_ms = 100
                        num_bins = int(max_lag_ms / bin_width_ms)
                        lag_range = min(num_bins, zero_lag_idx)

                        if lag_range > 0:
                            acg_symmetric = acg_full[
                                zero_lag_idx - lag_range : zero_lag_idx + lag_range + 1
                            ]
                            time_lags = np.arange(-lag_range, lag_range + 1) * bin_width_ms

                            # Zero out central peak
                            zero_idx = np.where(time_lags == 0)[0]
                            if len(zero_idx) > 0:
                                acg_symmetric[zero_idx[0]] = 0

                            spike_variance = np.var(binned_spikes)
                            if spike_variance != 0:
                                acg_norm = acg_symmetric / spike_variance / len(binned_spikes)
                            else:
                                acg_norm = acg_symmetric.astype(float)

            if time_lags is not None and acg_norm is not None and len(time_lags) > 1:
                bar_graph = pg.BarGraphItem(
                    x=time_lags,
                    height=acg_norm,
                    width=0.8,
                    brush=(170, 0, 255, 100),
                    pen=pg.mkPen('#aa00ff', width=1),
                )
                self.acg_plot.addItem(bar_graph)

                zero_line = pg.InfiniteLine(
                    pos=0,
                    angle=90,
                    pen=pg.mkPen('#ffffff', width=2, style=Qt.DashLine),
                )
                self.acg_plot.addItem(zero_line)

        # ------------------------------------------------------------------
        # 4. ISI (HISTOGRAM / ISI vs AMPLITUDE)
        # ------------------------------------------------------------------
        self.isi_plot.clear()
        try:
            isi_ms = hist_x = hist_y = None
            if data is not None:
                isi_ms = data.get('isi_ms')
                hist_x = data.get('isi_hist_x')
                hist_y = data.get('isi_hist_y')

            if isi_ms is None and spikes is not None:
                isi_ms = np.diff(np.sort(spikes)) / float(sr) * 1000.0

            current_isi_view = self.isi_view_combo.currentText()

            if current_isi_view == 'ISI Histogram':
                if (hist_x is None or hist_y is None) and isi_ms is not None and len(isi_ms) > 0:
                    hist_y, hist_x = np.histogram(isi_ms, bins=np.linspace(0, 50, 101))

                if hist_x is not None and hist_y is not None:
                    self.isi_plot.plot(
                        hist_x,
                        hist_y,
                        stepMode='center',
                        fillLevel=0,
                        brush=(0, 163, 224, 150),
                        pen=pg.mkPen('#33b5e5', width=2),
                    )

                    refractory_period = dm.get_refractory_period()
                    if self.show_refractory_line_checkbox.isChecked():
                        self.isi_plot.addItem(
                            pg.InfiniteLine(
                                refractory_period,
                                angle=90,
                                pen=pg.mkPen('r', style=Qt.DashLine),
                            )
                        )

                self.isi_plot.setLabel('bottom', 'ISI (ms)')
                self.isi_plot.setLabel('left', 'Count')

            elif current_isi_view == 'ISI vs Amplitude':
                # ISI vs amplitude view (scatter or 2D density)
                valid_isi = valid_amplitudes = None
                if data is not None:
                    valid_isi = data.get('isi_vs_amp_valid_isi')
                    valid_amplitudes = data.get('isi_vs_amp_valid_amplitudes')

                if valid_isi is None or valid_amplitudes is None:
                    try:
                        valid_isi, valid_amplitudes = dm.get_isi_vs_amplitude_data(cluster_id)
                    except AttributeError:
                        # Older DataManager without caching: compute locally
                        all_amplitudes = dm.get_cluster_spike_amplitudes(cluster_id)
                        if isi_ms is not None and len(isi_ms) > 0 and len(all_amplitudes) > 1:
                            min_len = min(len(isi_ms), len(all_amplitudes) - 1)
                            if min_len > 0:
                                valid_isi = isi_ms[:min_len]
                                valid_amplitudes = all_amplitudes[1 : min_len + 1]

                if (
                    valid_isi is not None
                    and valid_amplitudes is not None
                    and len(valid_isi) > 0
                ):
                    # Auto-switch to density view for many ISIs
                    if (
                        len(valid_isi) > ISI_DENSITY_THRESHOLD
                        and self.isi_display_combo.currentText() == 'Scatter'
                    ):
                        self.isi_display_combo.setCurrentText('Density')

                    display_mode = self.isi_display_combo.currentText()

                    if display_mode == 'Scatter':
                        self.isi_plot.plot(
                            valid_isi,
                            valid_amplitudes,
                            pen=None,
                            symbol='o',
                            symbolSize=5,
                            symbolBrush=(255, 165, 0, 150),
                            name='ISI vs Amplitude Scatter',
                        )
                    else:
                        # Density (2D histogram)
                        nbins_isi = 100
                        nbins_amp = 80

                        isi_min, isi_max = float(valid_isi.min()), float(valid_isi.max())
                        amp_min, amp_max = float(valid_amplitudes.min()), float(
                            valid_amplitudes.max()
                        )

                        if isi_max <= isi_min:
                            isi_max = isi_min + 1e-3
                        if amp_max <= amp_min:
                            amp_max = amp_min + 1e-3

                        H, xedges, yedges = np.histogram2d(
                            valid_isi,
                            valid_amplitudes,
                            bins=[nbins_isi, nbins_amp],
                            range=[[isi_min, isi_max], [amp_min, amp_max]],
                        )

                        H = np.log1p(H)
                        H = H.T
                        H = np.flipud(H)

                        img = pg.ImageItem(H)
                        if hasattr(self, '_hot_lut') and self._hot_lut is not None:
                            img.setLookupTable(self._hot_lut)

                        rect = pg.QtCore.QRectF(
                            xedges[0],
                            yedges[0],
                            xedges[-1] - xedges[0],
                            yedges[-1] - yedges[0],
                        )
                        img.setRect(rect)
                        self.isi_plot.addItem(img)

                    self.isi_plot.setLabel('bottom', 'ISI (ms)')
                    self.isi_plot.setLabel('left', 'Amplitude (µV)')

                    # Apply X-range preset for this view
                    self._apply_isi_range_preset(valid_isi)
                else:
                    # Not enough data → empty scatter
                    self.isi_plot.plot([], [], pen=None, symbol='o', symbolSize=5)
                    self.isi_plot.setLabel('bottom', 'ISI (ms)')
                    self.isi_plot.setLabel('left', 'Amplitude (µV)')

        except Exception:
            # Never allow ISI failures to break FR plot
            pass

        # ------------------------------------------------------------------
        # 5. FIRING RATE (LEFT AXIS) + AMPLITUDE (RIGHT AXIS)
        # ------------------------------------------------------------------
        # Use persistent curves so we don't keep allocating new PlotDataItems.
        try:
            # Lazily create helper attributes
            if not hasattr(self, '_fr_rate_curve'):
                self._fr_rate_curve = None
            if not hasattr(self, '_fr_amp_curve'):
                self._fr_amp_curve = None
            if not hasattr(self, '_fr_overlay_curve'):
                self._fr_overlay_curve = None

            fr_bin_centers = fr_rate = None
            amp_x = amp_y = amp_ymax = None
            overlay_x = overlay_y = None

            if data is not None:
                fr_bin_centers = data.get('fr_bin_centers')
                fr_rate = data.get('fr_rate')
                amp_x = data.get('fr_amp_x')
                amp_y = data.get('fr_amp_y')
                amp_ymax = data.get('fr_amp_ymax')
                overlay_x = data.get('fr_overlay_x')
                overlay_y = data.get('fr_overlay_y')

            # Backwards-compatible FR computation if cache missing
            if (fr_bin_centers is None or fr_rate is None) and spikes is not None:
                spikes_sec = np.asarray(spikes) / float(sr)
                if spikes_sec.size > 0:
                    bins = np.arange(0, spikes_sec.max() + 1.0, 1.0)
                    counts, bin_edges = np.histogram(spikes_sec, bins=bins)
                    fr_rate = gaussian_filter1d(counts.astype(float), sigma=5)
                    fr_bin_centers = bin_edges[:-1]

                    # Bin amplitudes by time (same binning as FR)
                    all_amplitudes = dm.get_cluster_spike_amplitudes(cluster_id)
                    if len(all_amplitudes) > 0:
                        amplitude_binned = []
                        for i in range(len(bin_edges) - 1):
                            bin_start = bin_edges[i]
                            bin_end = bin_edges[i + 1]
                            mask = (spikes_sec >= bin_start) & (spikes_sec < bin_end)
                            if np.any(mask):
                                amplitude_binned.append(np.mean(all_amplitudes[mask]))
                            else:
                                amplitude_binned.append(np.nan)
                        amplitude_binned = np.asarray(amplitude_binned)

                        if np.any(~np.isnan(amplitude_binned)):
                            amp_x = fr_bin_centers
                            amp_y = amplitude_binned.copy()
                            # Simple interpolation over NaNs
                            valid_idx = ~np.isnan(amp_y)
                            if np.any(valid_idx):
                                try:
                                    f_interp = interp1d(
                                        amp_x[valid_idx],
                                        amp_y[valid_idx],
                                        kind='linear',
                                        fill_value='extrapolate',
                                    )
                                    amp_y = f_interp(amp_x)
                                except Exception:
                                    pass
                            amp_ymax = float(np.nanmax(amp_y)) if np.any(~np.isnan(amp_y)) else None

                        # Overlay averaged amplitude (normalized) on FR axis
                        if np.max(all_amplitudes) > 0 and np.max(fr_rate) > 0:
                            normalized_amplitudes = all_amplitudes / np.max(all_amplitudes)
                            if len(normalized_amplitudes) > 10:
                                avg_amp = np.convolve(
                                    normalized_amplitudes, np.ones(10) / 10.0, mode='valid'
                                )
                                overlay_y = avg_amp * 0.8 * np.max(fr_rate)
                                overlay_x = spikes_sec[: len(overlay_y)]

            # Left axis: firing rate
            if fr_bin_centers is not None and fr_rate is not None and len(fr_bin_centers) > 0:
                if self._fr_rate_curve is None:
                    self._fr_rate_curve = self.fr_plot.plot(
                        fr_bin_centers,
                        fr_rate,
                        pen=pg.mkPen('#ffeb3b', width=2),
                    )
                else:
                    self._fr_rate_curve.setData(fr_bin_centers, fr_rate)
            elif self._fr_rate_curve is not None:
                self._fr_rate_curve.setData([], [])
            # Overlay averaged amplitude (green) on FR axis
            if overlay_x is not None and overlay_y is not None and len(overlay_x) > 0:
                if self._fr_overlay_curve is None:
                    self._fr_overlay_curve = self.fr_plot.plot(
                        overlay_x,
                        overlay_y,
                        pen=pg.mkPen('#00FF00', width=0.5),
                        name='Averaged Amplitude',
                    )
                else:
                    self._fr_overlay_curve.setData(overlay_x, overlay_y)
            elif self._fr_overlay_curve is not None:
                self._fr_overlay_curve.setData([], [])

        except Exception:
            # If FR plotting fails, keep axes but avoid crashing the UI
            pass

        # Final axis labels (kept consistent even if something failed above)
        self.fr_plot.setLabel('bottom', 'Time (s)')
        self.fr_plot.setLabel('left', 'Firing Rate (Hz)', color='#ffeb3b')
