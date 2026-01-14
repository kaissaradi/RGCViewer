import numpy as np
import pyqtgraph as pg
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QSplitter, QFrame, QGridLayout, QTabWidget
)
from qtpy.QtCore import Qt
from sklearn.decomposition import PCA

class WaveformPanel(QWidget):
    """
    Advanced Waveform Panel v2:
    - Left: Raw snippets cloud + Median + Percentile Envelope
    - Right (Tab 1): PCA (Unit vs Background Spikes)
    - Right (Tab 2): Template Match (Correlation Dist + Residuals)
    - Right (Bottom): Quick Stats
    """

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._setup_ui()

    def _setup_ui(self):
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Splitter to resize between Waveforms (Left) and Analytics (Right)
        self.splitter = QSplitter(Qt.Horizontal)
        self.layout.addWidget(self.splitter)

        # --- LEFT: Main Waveform Plot ---
        self.wave_container = QWidget()
        wave_layout = QVBoxLayout(self.wave_container)
        wave_layout.setContentsMargins(0,0,0,0)

        self.plot_widget = pg.PlotWidget(title="Waveform Cloud (Median + 10-90% Envelope)")
        self.plot_widget.setLabel('bottom', "Time (ms)")
        self.plot_widget.setLabel('left', "Amplitude (µV)")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend(offset=(10, 10))
        wave_layout.addWidget(self.plot_widget)

        # --- RIGHT: Analytics Panel ---
        self.stats_container = QFrame()
        self.stats_container.setStyleSheet("background-color: #252525; border-left: 1px solid #3d3d3d;")
        self.stats_container.setFixedWidth(400) 
        stats_layout = QVBoxLayout(self.stats_container)
        stats_layout.setContentsMargins(10, 10, 10, 10)
        stats_layout.setSpacing(10)

        # Tabs for Plots
        self.tabs = QTabWidget()
        stats_layout.addWidget(self.tabs, stretch=2)

        # Tab 1: PCA
        self.pca_tab = QWidget()
        pca_layout = QVBoxLayout(self.pca_tab)
        pca_layout.setContentsMargins(5,5,5,5)
        self.pca_plot = pg.PlotWidget()
        self.pca_plot.setBackground(None)
        self.pca_plot.hideAxis('bottom')
        self.pca_plot.hideAxis('left')
        self.pca_plot.showGrid(x=True, y=True, alpha=0.2)
        self.pca_plot.setAspectLocked(False)
        pca_layout.addWidget(QLabel("<b>Principal Component Analysis</b><br>(Gray=Ch. Background, Red=Unit)"))
        pca_layout.addWidget(self.pca_plot)
        self.tabs.addTab(self.pca_tab, "PCA (Isolation)")

        # Tab 2: Template Match / Residuals
        self.tmpl_tab = QWidget()
        tmpl_layout = QVBoxLayout(self.tmpl_tab)
        tmpl_layout.setContentsMargins(5,5,5,5)
        
        self.hist_plot = pg.PlotWidget()
        self.hist_plot.setBackground(None)
        self.hist_plot.setLabel('bottom', "Correlation with Median")
        self.hist_plot.setLabel('left', "Count")
        self.hist_plot.showGrid(x=True, y=True, alpha=0.2)
        
        tmpl_layout.addWidget(QLabel("<b>Template Match Score</b> (Dot Product)"))
        tmpl_layout.addWidget(self.hist_plot)
        self.tabs.addTab(self.tmpl_tab, "Template Match")

        # Bottom: Statistics Grid
        stats_layout.addWidget(QLabel("<b>Cluster Statistics</b>"))
        self.stats_grid = QGridLayout()
        self.stats_labels = {}
        
        metrics = [
            ("n_spikes", "Spike Count"), 
            ("p2p", "Peak-to-Peak (µV)"),
            ("snr", "SNR (Est)"),
            ("jitter", "Jitter (ms)"),
            ("resid_rms", "Resid. RMS (µV)"), # New metric
            ("match_mean", "Mean Match Score") # New metric
        ]
        
        for i, (key, label) in enumerate(metrics):
            lbl_title = QLabel(label)
            lbl_title.setStyleSheet("color: #888;")
            lbl_value = QLabel("--")
            lbl_value.setStyleSheet("font-size: 13px; font-weight: bold; color: #ddd;")
            
            self.stats_labels[key] = lbl_value
            self.stats_grid.addWidget(lbl_title, i // 2, (i % 2) * 2)
            self.stats_grid.addWidget(lbl_value, i // 2, (i % 2) * 2 + 1)

        stats_layout.addLayout(self.stats_grid)
        stats_layout.addStretch(0)

        self.splitter.addWidget(self.wave_container)
        self.splitter.addWidget(self.stats_container)
        self.splitter.setSizes([900, 400])

    def update_all(self, cluster_id):
        if not self.isVisible():
            return
            
        self.plot_widget.clear()
        self.pca_plot.clear()
        self.hist_plot.clear()
        
        if cluster_id is None:
            return

        # Fetch Features
        features = self.main_window.data_manager.get_lightweight_features(cluster_id)

        if not features or 'raw_snippets' not in features:
            self.plot_widget.setTitle(f"Cluster {cluster_id} - Waiting for data...")
            self._reset_stats()
            return

        # Data retrieval
        snippets_raw = features['raw_snippets'] # (n_channels, n_time, n_spikes)
        mean_ei = features['median_ei']         # (n_channels, n_time)
        
        # Identify Dominant Channel
        ptp = mean_ei.max(axis=1) - mean_ei.min(axis=1)
        dom_chan = np.argmax(ptp)

        # Extract dominant channel data -> (n_spikes, n_time)
        waveforms = snippets_raw[dom_chan, :, :].T 
        n_spikes, n_time = waveforms.shape
        
        # Time Axis
        sr = self.main_window.data_manager.sampling_rate
        t_ms = (np.arange(n_time) - 20) / sr * 1000 

        # --- PLOT 1: Waveforms (Cloud + Median + Envelope) ---
        if n_spikes > 5:
            p10 = np.percentile(waveforms, 10, axis=0)
            p90 = np.percentile(waveforms, 90, axis=0)
            
            curve1 = pg.PlotCurveItem(t_ms, p10, pen=None)
            curve2 = pg.PlotCurveItem(t_ms, p90, pen=None)
            fill = pg.FillBetweenItem(curve1, curve2, brush=pg.mkBrush(50, 50, 255, 40))
            self.plot_widget.addItem(fill)

        # Shadows
        n_shadows = min(n_spikes, 100)
        indices = np.random.choice(n_spikes, n_shadows, replace=False)
        subset_waves = waveforms[indices, :]
        
        x_connected = np.tile(t_ms, (n_shadows, 1))
        x_flat = np.column_stack((x_connected, np.full(n_shadows, np.nan))).flatten()
        y_flat = np.column_stack((subset_waves, np.full(n_shadows, np.nan))).flatten()

        self.plot_widget.plot(x_flat, y_flat, pen=pg.mkPen(color=(255, 255, 255, 30), width=1))

        # Median
        median_trace = np.median(waveforms, axis=0)
        self.plot_widget.plot(t_ms, median_trace, pen=pg.mkPen(color='#00e6a0', width=3), name="Median")
        self.plot_widget.setTitle(f"Cluster {cluster_id} - Ch {dom_chan} ({n_spikes} spikes)")

        # --- PLOT 2: PCA (Unit vs Background) ---
        # Note: Ideally, get_lightweight_features returns 'background_snippets' on the same channel
        # If not available, we just plot the unit's spikes (self-consistency check)
        bg_waves = features.get('background_snippets', None) # Expected: (n_bg, n_time) or None
        
        self._update_pca(waveforms, bg_waves)

        # --- PLOT 3: Template Match & Residuals ---
        self._update_template_metrics(waveforms, median_trace)

        # --- STATISTICS ---
        self._update_stats(n_spikes, median_trace, t_ms, waveforms)

    def _update_pca(self, unit_waves, bg_waves=None):
        """
        Computes PCA on combined (Unit + Background) spikes to show isolation.
        """
        n_unit = unit_waves.shape[0]
        
        # Subsample for PCA speed (max 500 unit spikes)
        if n_unit > 500:
            idx = np.random.choice(n_unit, 500, replace=False)
            unit_subset = unit_waves[idx]
        else:
            unit_subset = unit_waves

        combined_waves = unit_subset
        labels = np.ones(len(unit_subset)) # 1 = Unit

        # Add background if available
        if bg_waves is not None and len(bg_waves) > 0:
            n_bg = bg_waves.shape[0]
            # Subsample background
            if n_bg > 500:
                idx_bg = np.random.choice(n_bg, 500, replace=False)
                bg_subset = bg_waves[idx_bg]
            else:
                bg_subset = bg_waves
                
            combined_waves = np.vstack([bg_subset, unit_subset])
            labels = np.concatenate([np.zeros(len(bg_subset)), np.ones(len(unit_subset))]) # 0 = Bg, 1 = Unit

        if len(combined_waves) < 3:
            return

        try:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(combined_waves)
            
            # Plot Background (Gray)
            bg_mask = (labels == 0)
            if np.any(bg_mask):
                scatter_bg = pg.ScatterPlotItem(
                    size=5, pen=None, brush=pg.mkBrush(100, 100, 100, 100), hoverable=False
                )
                scatter_bg.addPoints(x=coords[bg_mask, 0], y=coords[bg_mask, 1])
                self.pca_plot.addItem(scatter_bg)

            # Plot Unit (Red)
            unit_mask = (labels == 1)
            scatter_unit = pg.ScatterPlotItem(
                size=6, pen=None, brush=pg.mkBrush(255, 50, 50, 180), hoverable=True
            )
            scatter_unit.addPoints(x=coords[unit_mask, 0], y=coords[unit_mask, 1])
            self.pca_plot.addItem(scatter_unit)

            var = pca.explained_variance_ratio_
            self.pca_plot.setLabel('bottom', f"PC1 ({var[0]:.1%})")
            self.pca_plot.setLabel('left', f"PC2 ({var[1]:.1%})")
            
        except Exception as e:
            print(f"PCA Error: {e}")

    def _update_template_metrics(self, waveforms, median_trace):
        """
        Computes 1. Correlation of each spike with Median. 2. Residual RMS.
        """
        if len(waveforms) == 0:
            return

        # A. Template Match (Correlation Coefficient)
        # Normalize template (median)
        norm_template = np.linalg.norm(median_trace)
        if norm_template == 0:
            return
            
        unit_vec = median_trace / norm_template
        
        # Dot product of normalized waveforms with normalized template
        # (This is cosine similarity / correlation for zero-centered data)
        # Normalize each waveform
        wave_norms = np.linalg.norm(waveforms, axis=1)
        valid_idx = wave_norms > 0
        
        correlations = np.zeros(len(waveforms))
        correlations[valid_idx] = np.dot(waveforms[valid_idx], unit_vec) / wave_norms[valid_idx]
        
        # Plot Histogram
        y, x = np.histogram(correlations, bins=np.linspace(0, 1.0, 50))
        bg = pg.BarGraphItem(x=x[:-1], height=y, width=0.02, brush='b')
        self.hist_plot.addItem(bg)
        
        # B. Residuals (Spike - Median)
        residuals = waveforms - median_trace
        # RMS of the residuals
        resid_rms = np.sqrt(np.mean(residuals**2))
        
        # Store for stats
        self.current_resid_rms = resid_rms
        self.current_match_mean = np.mean(correlations)

    def _update_stats(self, n_spikes, median_trace, t_ms, all_waves):
        p2p = np.ptp(median_trace)
        baseline_std = np.std(median_trace[:15])
        snr = (np.max(np.abs(median_trace)) / baseline_std) if baseline_std > 0 else 0.0

        peak_indices = np.argmin(all_waves, axis=1)
        jitter_ms = np.std(peak_indices) * (t_ms[1] - t_ms[0])

        self.stats_labels['n_spikes'].setText(str(n_spikes))
        self.stats_labels['p2p'].setText(f"{p2p:.1f}")
        self.stats_labels['snr'].setText(f"{snr:.1f}")
        self.stats_labels['jitter'].setText(f"{jitter_ms:.3f}")
        
        # New Metrics
        if hasattr(self, 'current_resid_rms'):
            self.stats_labels['resid_rms'].setText(f"{self.current_resid_rms:.1f}")
        if hasattr(self, 'current_match_mean'):
            self.stats_labels['match_mean'].setText(f"{self.current_match_mean:.3f}")

    def _reset_stats(self):
        for lbl in self.stats_labels.values():
            lbl.setText("--")