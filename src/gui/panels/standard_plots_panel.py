import numpy as np
import pyqtgraph as pg
from qtpy.QtWidgets import QWidget, QVBoxLayout, QSplitter, QHBoxLayout, QCheckBox, QComboBox, QLabel, QPushButton, QDoubleSpinBox
from qtpy.QtCore import Qt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate
from scipy.interpolate import interp1d
from ...analysis.constants import ISI_REFRACTORY_PERIOD_MS
import logging

logger = logging.getLogger(__name__)
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
        layout.setContentsMargins(0, 0, 0, 0)

        # Background image from array calibration
        self._array_bg_image = None
        self._array_transform_data = None
        self._array_image_path = None
        self._has_valid_array_transform = False

        # Controls: top control bar
        ctrl_bar = QHBoxLayout()
        ctrl_bar_widget = QWidget()
        ctrl_bar_widget.setMaximumHeight(35)
        ctrl_bar_widget.setLayout(ctrl_bar)

        # Channel display mode
        ctrl_bar.addWidget(QLabel('Channel Display:'))
        self.channel_mode_combo = QComboBox()
        self.channel_mode_combo.addItems(
            ['Main Channel', 'Top Channels', 'Whole Array', 'Array Image'])
        ctrl_bar.addWidget(self.channel_mode_combo)

        ctrl_bar.addStretch()

        self.channel_mode_combo.currentTextChanged.connect(
            self._on_control_changed)

        layout.addWidget(ctrl_bar_widget)

        # 2x2 Layout using Splitters
        self.vert_splitter = QSplitter(Qt.Vertical)
        layout.addWidget(self.vert_splitter)

        self.top_splitter = QSplitter(Qt.Horizontal)
        self.bottom_splitter = QSplitter(Qt.Horizontal)
        self.vert_splitter.addWidget(self.top_splitter)
        self.vert_splitter.addWidget(self.bottom_splitter)

        # ---------------------------------------------------------
        # 1. Template Grid (Top Left)
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # 2. Autocorrelation (Top Right)
        # ---------------------------------------------------------
        self.acg_plot = pg.PlotWidget(title="Autocorrelation")
        self.acg_plot.setLabel('bottom', "Time lag (ms)")
        self.acg_plot.setLabel('left', "Autocorrelation")
        self._style_plot(self.acg_plot)
        
        # --- PERSISTENT ACG/CCG ITEMS ---
        # 1. ACG Bar (Purple) - Default
        self._acg_bar = pg.BarGraphItem(x=[], height=[], width=0.8, brush=(170, 0, 255, 100), pen=pg.mkPen('#aa00ff', width=1))
        self.acg_plot.addItem(self._acg_bar)
        
        # 2. CCG Bar (Orange) - Hidden by default
        self._ccg_bar = pg.BarGraphItem(x=[], height=[], width=0.8, brush=(255, 152, 0, 100), pen=pg.mkPen('#ff9800', width=1))
        self.acg_plot.addItem(self._ccg_bar)
        self._ccg_bar.setVisible(False)

        # 3. Zero Line
        self._acg_zero_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('#ffffff', width=2, style=Qt.DashLine))
        self.acg_plot.addItem(self._acg_zero_line)

        self.top_splitter.addWidget(self.acg_plot)

        # ---------------------------------------------------------
        # 3. ISI (Bottom Left)
        # ---------------------------------------------------------
        isi_container = QWidget()
        isi_layout = QVBoxLayout(isi_container)
        isi_layout.setContentsMargins(0, 0, 0, 0)
        isi_controls = QHBoxLayout()

        isi_controls.addWidget(QLabel('View:'))
        self.isi_view_combo = QComboBox()
        self.isi_view_combo.addItems(['ISI Histogram', 'ISI vs Amplitude'])
        isi_controls.addWidget(self.isi_view_combo)

        self.show_refractory_line_checkbox = QCheckBox('Refr line')
        self.show_refractory_line_checkbox.setChecked(True)
        isi_controls.addWidget(self.show_refractory_line_checkbox)

        isi_controls.addWidget(QLabel('Ref (ms):'))
        self.refractory_spinbox = QDoubleSpinBox()
        self.refractory_spinbox.setRange(0.1, 10.0)
        self.refractory_spinbox.setDecimals(2)
        self.refractory_spinbox.setSingleStep(0.1)
        self.refractory_spinbox.setValue(1.0)
        isi_controls.addWidget(self.refractory_spinbox)

        self.update_refractory_btn = QPushButton('Set')
        isi_controls.addWidget(self.update_refractory_btn)

        isi_controls.addWidget(QLabel('Plot:'))
        self.isi_display_combo = QComboBox()
        self.isi_display_combo.addItems(['Scatter', 'Density'])
        self.isi_display_combo.setCurrentText('Scatter')
        isi_controls.addWidget(self.isi_display_combo)

        isi_controls.addWidget(QLabel('X:'))
        self.isi_range_combo = QComboBox()
        self.isi_range_combo.addItems(['0–50 ms', '0–500 ms', '0–1000 ms', 'Full'])
        self.isi_range_combo.setCurrentText('0–50 ms')
        isi_controls.addWidget(self.isi_range_combo)

        isi_controls.addStretch()
        isi_layout.addLayout(isi_controls)

        self.isi_plot = pg.PlotWidget(title="ISI Distribution")
        self.isi_plot.setLabel('bottom', "ISI (ms)")
        self._style_plot(self.isi_plot)
        
        # --- PERSISTENT ISI ITEMS ---
        # 1. Histogram (Step Mode)
        self._isi_curve = self.isi_plot.plot([], [], stepMode=True, fillLevel=0, 
                                             brush=(0, 163, 224, 150), pen=pg.mkPen('#33b5e5', width=2))
        
        # 2. Scatter
        self._isi_scatter = pg.ScatterPlotItem(size=5, pen=None, brush=pg.mkBrush(255, 165, 0, 150))
        self.isi_plot.addItem(self._isi_scatter)
        self._isi_scatter.setVisible(False)
        
        # 3. Density
        self._isi_image = pg.ImageItem()
        self._hot_lut = self._create_hot_colormap()
        self._isi_image.setLookupTable(self._hot_lut)
        self.isi_plot.addItem(self._isi_image)
        self._isi_image.setVisible(False)

        # 4. Refractory Line
        self._isi_ref_line = pg.InfiniteLine(angle=90, pen=pg.mkPen('r', style=Qt.DashLine))
        self.isi_plot.addItem(self._isi_ref_line)

        isi_layout.addWidget(self.isi_plot)
        self.bottom_splitter.addWidget(isi_container)

        self.isi_view_combo.currentTextChanged.connect(self._on_control_changed)
        self.show_refractory_line_checkbox.stateChanged.connect(self._on_control_changed)
        self.update_refractory_btn.clicked.connect(self._update_refractory_period)
        self.isi_display_combo.currentTextChanged.connect(self._on_control_changed)
        self.isi_range_combo.currentTextChanged.connect(self._on_control_changed)

        # ---------------------------------------------------------
        # 4. Firing Rate (Bottom Right)
        # ---------------------------------------------------------
        self.fr_plot = pg.PlotWidget(title="Signal Health")
        self.fr_plot.setLabel('bottom', "Time (s)")
        self.fr_plot.setLabel('left', "Firing Rate (Hz)", color='#ffeb3b')
        self._style_plot(self.fr_plot)

        # --- PERSISTENT FR ITEMS ---
        # 1. Yellow Rate Curve
        self._fr_rate_curve = self.fr_plot.plot([], [], pen=pg.mkPen('#ffeb3b', width=2), name='fr')
        
        # 2. Green Overlay Curve
        self._fr_overlay_curve = self.fr_plot.plot([], [], pen=pg.mkPen('#00FF00', width=0.5), name='Averaged Amplitude')

        self.bottom_splitter.addWidget(self.fr_plot)

        self.vert_splitter.setSizes([500, 300])
        self._hot_lut_local = self._hot_lut

    def _update_refractory_period(self):
        new_period = self.refractory_spinbox.value()
        self.main_window.data_manager.set_refractory_period(new_period)
        cluster_id = self.main_window._get_selected_cluster_id()
        if cluster_id is not None:
            self.update_all(cluster_id)

    def _style_plot(self, plot):
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.getAxis('bottom').setPen(pg.mkPen('#888888'))
        plot.getAxis('bottom').setTextPen(pg.mkPen('#888888'))
        plot.getAxis('left').setPen(pg.mkPen('#888888'))
        plot.getAxis('left').setTextPen(pg.mkPen('#888888'))

    def _create_hot_colormap(self):
        colors = [(0, 0, 0), (255, 0, 0), (255, 255, 0), (255, 255, 255)]
        positions = [0, 0.33, 0.66, 1.0]
        cmap = pg.ColorMap(pos=positions, color=colors)
        return cmap.getLookupTable(start=0.0, stop=1.0, nPts=256)

    def _apply_isi_range_preset(self, isi_ms):
        """Apply X-range preset safely, enforcing positive start for Log stability."""
        x_min = 0.01 
        preset = self.isi_range_combo.currentText()
        if preset == '0–50 ms':
            self.isi_plot.setXRange(x_min, 50.0, padding=0)
        elif preset == '0–500 ms':
            self.isi_plot.setXRange(x_min, 500.0, padding=0)
        elif preset == '0–1000 ms':
            self.isi_plot.setXRange(x_min, 1000.0, padding=0)
        else: # Full
            if isi_ms is not None and len(isi_ms) > 0:
                current_max = float(isi_ms.max())
            else:
                current_max = 50.0
            if current_max <= x_min: current_max = 50.0
            self.isi_plot.setXRange(x_min, current_max * 1.05, padding=0)
            
        self.isi_plot.plotItem.disableAutoRange(pg.ViewBox.XAxis)
    
    def _on_control_changed(self):
        try:
            cluster_id = self.main_window._get_selected_cluster_id()
            if cluster_id is not None:
                self.update_all(cluster_id)
        except Exception:
            pass

    def refresh_array_image(self, transform_path: str):
        """Loads and aligns the microscope image behind the template grid."""
        import json
        from pathlib import Path
        from PIL import Image
        import numpy as np
        from qtpy.QtGui import QTransform
        import logging
        logger = logging.getLogger(__name__)

        try:
            with open(transform_path, 'r') as f:
                data = json.load(f)

            img_name = data.get('image_file')
            if not img_name: 
                return

            img_path = Path(transform_path).parent / img_name
            if not img_path.exists():
                return

            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img, dtype=np.uint8)
            # Transpose to match pyqtgraph's (Width, Height, Color) expectation
            img_pg = img_array.transpose(1, 0, 2)

            if self._array_bg_image is None:
                self._array_bg_image = pg.ImageItem()
                self._array_bg_image.setZValue(-20)  # Push firmly to background
                self.grid_plot.addItem(self._array_bg_image)

            self._array_bg_image.setImage(img_pg)

            # Store transform data
            self._array_transform_data = data
            self._array_image_path = img_path
            self._has_valid_array_transform = True

            # --- Proper QTransform Mapping ---
            # Math: pixels = scale * microns + offset
            # Ergo: microns = (pixels - offset) / scale
            sx = float(data.get('scale_x', 1.0))
            sy = float(data.get('scale_y', 1.0))
            ox = float(data.get('offset_x', 0.0))
            oy = float(data.get('offset_y', 0.0))

            # Since the grid distortion (x_scale) turns off when the image is active,
            # we do a pure 1:1 physical translation matrix.
            tr = QTransform()
            tr.setMatrix(1/sx, 0, 0,
                         0, 1/sy, 0,
                         -ox/sx, -oy/sy, 1)
            self._array_bg_image.setTransform(tr)

        except Exception as e:
            logger.warning(f"Failed to load array image: {e}")
            self._has_valid_array_transform = False


    def update_all(self, cluster_id):
        """
        Update all standard plots for the given cluster.
        Uses batch rendering with disabled auto-range for single render pass.
        """
        if cluster_id is None:
            return

        dm = self.main_window.data_manager
        if dm is None:
            return

        # Disable auto-range for batch rendering (prevents multiple render passes)
        plots_to_update = [self.grid_plot, self.acg_plot, self.isi_plot, self.fr_plot]
        for plot in plots_to_update:
            plot.disableAutoRange()

        # If user requests array image but none is loaded, prompt for calibration
        current_mode = self.channel_mode_combo.currentText()
        if current_mode == 'Array Image' and not self._has_valid_array_transform:
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "No Array Image",
                "No calibrated array image is available. "
                "Please map an image to the array using File → Array → Map Image to Array...",
            )
            try:
                self.main_window._open_array_calibration()
            except Exception:
                pass
            return

        # ------------------------------------------------------------------
        # 1. TEMPLATE GRID
        # ------------------------------------------------------------------
        self.grid_plot.clear() 
        
        # Only show background when in Array Image mode
        if self._array_bg_image is not None and current_mode == 'Array Image':
            self.grid_plot.addItem(self._array_bg_image)
        
        try:
            if hasattr(dm, 'templates') and dm.templates is not None and cluster_id < dm.templates.shape[0]:
                template = dm.templates[cluster_id]
                pos = np.array(dm.channel_positions)
                
                # --- Dynamic Stretch: 1.0 (true physical) for Image, 1.5 (distorted) for traces ---
                x_scale = 1.0 if current_mode == 'Array Image' else 1.5
                y_scale = 1.0
                
                ptp = template.max(axis=0) - template.min(axis=0)
                max_ptp = ptp.max() if ptp.size > 0 else 1.0
                main_channel_idx = int(np.argmax(ptp)) if ptp.size > 0 else 0
                
                if current_mode == 'Main Channel':
                    relevant_channels = [main_channel_idx]
                    show_waveforms = True; show_dots = False; waveform_channels = [main_channel_idx]
                elif current_mode == 'Top Channels':
                    top_channel_indices = np.argsort(ptp)[::-1][:3]
                    relevant_channels = top_channel_indices
                    show_waveforms = True; show_dots = True; waveform_channels = top_channel_indices
                elif current_mode in ('Whole Array', 'Array Image'):
                    relevant_channels = np.arange(len(ptp))
                    waveform_channels = np.argsort(ptp)[::-1][:6]
                    show_waveforms = True; show_dots = True
                else:
                    relevant_channels = np.arange(len(ptp))
                    waveform_channels = np.argsort(ptp)[::-1][:6]
                    show_waveforms = True; show_dots = True

                cluster_amp = dm.get_cluster_mean_amplitude(cluster_id, method='mean')
                norm_ptp = ptp / max_ptp if max_ptp > 0 else np.zeros_like(ptp)
                channel_values = norm_ptp * cluster_amp

                if show_dots:
                    spots = []
                    vmin, vmax = channel_values.min(), channel_values.max()
                    vrange = max(vmax - vmin, 1e-6)
                    
                    # --- Lower opacity for Array Image mode so tissue shows through ---
                    alpha = 80 if current_mode == 'Array Image' else 180
                    
                    for ch in relevant_channels:
                        if ch >= len(channel_values) or ch >= len(pos): continue
                        x, y = pos[ch]
                        val = (channel_values[ch] - vmin) / vrange
                        size = 6 + val * 20
                        r = int(255 * val); g = int(120 * (1 - val)); b = int(255 * (1 - val))
                        
                        spots.append({'pos': (x * x_scale, y * y_scale), 'size': size, 
                                      'brush': pg.mkBrush(r, g, b, alpha), 'pen': pg.mkPen(None)})
                        
                    if spots:
                        scatter = pg.ScatterPlotItem(size=8)
                        scatter.addPoints(spots)
                        scatter.setZValue(-10)
                        self.grid_plot.addItem(scatter)

                if show_waveforms:
                    for ch in waveform_channels:
                        if ch >= len(pos): continue
                        x, y = pos[ch]
                        trace = template[:, ch]
                        trace_scaled = (trace / max_ptp) * 20 if max_ptp > 0 else trace
                        t_offset = np.linspace(-10, 10, len(trace))
                        self.grid_plot.plot(x * x_scale + t_offset, y * y_scale + trace_scaled, pen=pg.mkPen('#00331f', width=2.5), alpha=0.6)
                        self.grid_plot.plot(x * x_scale + t_offset, y * y_scale + trace_scaled, pen=pg.mkPen('#00e6a0', width=1.2))

                    if current_mode == 'Main Channel' and waveform_channels and main_channel_idx < len(pos):
                        _, main_y = pos[main_channel_idx]
                        self.grid_plot.addItem(pg.InfiniteLine(pos=main_y * y_scale, angle=0, pen=pg.mkPen('#ffffff', width=1, style=Qt.DashLine)))

                sim_panel = getattr(self.main_window, 'similarity_panel', None)
                selected_similar = []
                try:
                    if sim_panel and getattr(sim_panel, 'table', None):
                        sel_model = sim_panel.table.selectionModel()
                        if sel_model: selected_similar = sel_model.selectedRows()
                except Exception: pass

                if selected_similar and hasattr(sim_panel, 'similarity_model'):
                    sim_model = sim_panel.similarity_model
                    if sim_model and hasattr(sim_model, '_dataframe'):
                        for idx in selected_similar[:3]:
                            row = idx.row()
                            if row >= len(sim_model._dataframe): continue
                            similar_id = sim_model._dataframe.iloc[row]['cluster_id']
                            if similar_id < dm.templates.shape[0]:
                                sim_template = dm.templates[similar_id]
                                for ch in waveform_channels:
                                    if ch >= len(pos): continue
                                    x, y = pos[ch]
                                    trace = sim_template[:, ch]
                                    trace_scaled = (trace / max_ptp) * 20 if max_ptp > 0 else trace
                                    t_offset = np.linspace(-10, 10, len(trace))
                                    self.grid_plot.plot(x * x_scale + t_offset, y * y_scale + trace_scaled, pen=pg.mkPen('#ff9800', width=1.5))

                # --- Lock the zoom to perfectly frame the electrodes ---
                if current_mode in ('Whole Array', 'Array Image'):
                    min_x, min_y = np.min(pos, axis=0)
                    max_x, max_y = np.max(pos, axis=0)
                    
                    # Account for the scale stretch
                    min_x *= x_scale; max_x *= x_scale
                    min_y *= y_scale; max_y *= y_scale
                    
                    span_x = max_x - min_x
                    span_y = max_y - min_y
                    
                    pad_x = span_x * 0.9
                    pad_y = span_y * 0.2
                    
                    self.grid_plot.setXRange(min_x - pad_x, max_x + pad_x, padding=0)
                    self.grid_plot.setYRange(min_y - pad_y, max_y + pad_y, padding=0)

        except Exception as e: 
            import logging
            logging.getLogger(__name__).warning(f"Error drawing template grid: {e}")

        # ------------------------------------------------------------------
        # 2. FETCH DATA
        # ------------------------------------------------------------------
        try:
            data = dm.get_standard_plot_data(cluster_id)
        except Exception:
            data = None
        
        spikes = data.get('spikes') if data else None
        if spikes is None:
             self._acg_bar.setOpts(height=[])
             self._ccg_bar.setVisible(False)
             self._isi_curve.setData([0.01, 1], [0])
             self._fr_rate_curve.setData([], [])
             return

        # ------------------------------------------------------------------
        # 3. ACG / CCG
        # ------------------------------------------------------------------
        sim_panel = getattr(self.main_window, 'similarity_panel', None)
        selected_similar_rows = []
        try:
            if sim_panel and getattr(sim_panel, 'table', None):
                sel = sim_panel.table.selectionModel()
                if sel: selected_similar_rows = sel.selectedRows()
        except: pass

        showing_ccg = False

        if selected_similar_rows and hasattr(sim_panel, 'similarity_model'):
            sim_model = sim_panel.similarity_model
            if sim_model and hasattr(sim_model, '_dataframe'):
                row_idx = selected_similar_rows[0].row()
                if 0 <= row_idx < len(sim_model._dataframe):
                    similar_id = int(sim_model._dataframe.iloc[row_idx]['cluster_id'])
                    
                    if similar_id != cluster_id:
                        try:
                            spikes2 = None
                            data2 = dm.get_standard_plot_data(similar_id)
                            if data2: spikes2 = data2.get('spikes')
                            if spikes2 is None: spikes2 = dm.get_cluster_spikes(similar_id)

                            if spikes is not None and spikes2 is not None and len(spikes)>1 and len(spikes2)>1:
                                sr = float(dm.sampling_rate)
                                s1 = (np.asarray(spikes)/sr*1000).astype(int)
                                s2 = (np.asarray(spikes2)/sr*1000).astype(int)
                                duration = int(max(np.max(s1), np.max(s2)))
                                
                                if duration > 0:
                                    from scipy.signal import correlate
                                    bins = np.arange(0, duration + 1, 1, dtype=int)
                                    b1, _ = np.histogram(s1, bins=bins)
                                    b2, _ = np.histogram(s2, bins=bins)
                                    
                                    if len(b1) > 0 and len(b2) > 0:
                                        c1 = b1 - np.mean(b1)
                                        c2 = b2 - np.mean(b2)
                                        ccg_full = correlate(c1, c2, mode='full')
                                        
                                        zero = len(ccg_full) // 2
                                        lag = 100
                                        ccg_sym = ccg_full[zero - lag : zero + lag + 1]
                                        lags = np.arange(-lag, lag + 1)
                                        
                                        var = np.sqrt(np.var(b1) * np.var(b2))
                                        norm = ccg_sym / var / len(b1) if var != 0 else ccg_sym.astype(float)
                                        
                                        self._ccg_bar.setOpts(x=lags, height=norm)
                                        self._ccg_bar.setVisible(True)
                                        self._acg_bar.setVisible(False)
                                        self.acg_plot.setTitle(f"CCG: {cluster_id} vs {similar_id}")
                                        showing_ccg = True
                        except Exception: pass

        if not showing_ccg:
            time_lags = data.get('acg_time_lags')
            acg_norm = data.get('acg_norm')

            if time_lags is not None and acg_norm is not None and len(time_lags) > 1:
                self._acg_bar.setOpts(x=time_lags, height=acg_norm)
                self._acg_bar.setVisible(True)
                self._ccg_bar.setVisible(False)
                self.acg_plot.setTitle("Autocorrelation")
            else:
                self._acg_bar.setOpts(height=[])
                self._ccg_bar.setVisible(False)

        # ------------------------------------------------------------------
        # 4. ISI PLOT
        # ------------------------------------------------------------------
        isi_view = self.isi_view_combo.currentText()
        raw_isi_ms = data.get('isi_ms') if data else None
        if raw_isi_ms is None: raw_isi_ms = np.array([])

        if isi_view == 'ISI Histogram':
            self._isi_curve.setVisible(True)
            self._isi_scatter.setVisible(False)
            self._isi_image.setVisible(False)
            
            if len(raw_isi_ms) > 0:
                bins = np.linspace(0.01, 1000, 1001)
                hist_y, hist_x = np.histogram(raw_isi_ms, bins=bins)
                self._isi_curve.setData(hist_x, hist_y)
            else:
                self._isi_curve.setData([0.01, 1], [0])

            if self.show_refractory_line_checkbox.isChecked():
                self._isi_ref_line.setValue(dm.get_refractory_period())
                self._isi_ref_line.setVisible(True)
            else:
                self._isi_ref_line.setVisible(False)
            
            self._apply_isi_range_preset(raw_isi_ms)
            self.isi_plot.setLabel('bottom', 'ISI (ms)')
            self.isi_plot.setLabel('left', 'Count')

        elif isi_view == 'ISI vs Amplitude':
            self._isi_curve.setVisible(False)
            self._isi_ref_line.setVisible(False)
            
            valid_isi = data.get('isi_vs_amp_valid_isi')
            valid_amp = data.get('isi_vs_amp_valid_amplitudes')
            
            if valid_isi is not None and len(valid_isi) > 0:
                if len(valid_isi) > 5000 and self.isi_display_combo.currentText() == 'Scatter':
                    self.isi_display_combo.setCurrentText('Density')

                if self.isi_display_combo.currentText() == 'Scatter':
                    self._isi_scatter.setData(valid_isi, valid_amp)
                    self._isi_scatter.setVisible(True)
                    self._isi_image.setVisible(False)
                else:
                    self._isi_scatter.setVisible(False)
                    self._isi_image.setVisible(True)
                    isi_min, isi_max = float(valid_isi.min()), float(valid_isi.max())
                    amp_min, amp_max = float(valid_amp.min()), float(valid_amp.max())
                    if isi_max <= isi_min: isi_max = isi_min + 1e-3
                    if amp_max <= amp_min: amp_max = amp_min + 1e-3
                    H, xedges, yedges = np.histogram2d(valid_isi, valid_amp, bins=[100, 80], range=[[isi_min, isi_max], [amp_min, amp_max]])
                    self._isi_image.setImage(np.log1p(H.T))
                    self._isi_image.setRect(pg.QtCore.QRectF(xedges[0], yedges[0], xedges[-1]-xedges[0], yedges[-1]-yedges[0]))

                self._apply_isi_range_preset(valid_isi)
                self.isi_plot.setLabel('bottom', 'ISI (ms)')
                self.isi_plot.setLabel('left', 'Amplitude (µV)')
            else:
                self._isi_scatter.setData([], [])
                self._isi_image.clear()

        # ------------------------------------------------------------------
        # 5. FIRING RATE
        # ------------------------------------------------------------------
        fr_bin_centers = data.get('fr_bin_centers')
        fr_rate = data.get('fr_rate')
        overlay_x = data.get('fr_overlay_x')
        overlay_y = data.get('fr_overlay_y')

        if fr_bin_centers is not None and fr_rate is not None:
            self._fr_rate_curve.setData(fr_bin_centers, fr_rate)
        else:
            self._fr_rate_curve.setData([], [])

        if overlay_x is not None and overlay_y is not None:
            self._fr_overlay_curve.setData(overlay_x, overlay_y)
        else:
            self._fr_overlay_curve.setData([], [])

        # Re-enable auto-range after batch updates
        for plot in plots_to_update:
            plot.enableAutoRange()