import numpy as np
import pyqtgraph as pg
from qtpy.QtWidgets import QWidget, QVBoxLayout
from qtpy.QtCore import Qt

class WaveformPanel(QWidget):
    """
    Displays the 'Cloud' of raw waveform snippets for the selected cluster.
    """
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Styling
        pg.setConfigOption('background', '#1e1e1e')
        pg.setConfigOption('foreground', '#d0d0d0')
        pg.setConfigOptions(antialias=True)

        # Plot Widget
        self.plot_widget = pg.PlotWidget(title="Raw Spike Snippets")
        self.plot_widget.setLabel('bottom', "Time (ms)")
        self.plot_widget.setLabel('left', "Amplitude (µV)")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.layout.addWidget(self.plot_widget)

    def update_all(self, cluster_id):
        if not self.isVisible(): return
        self.plot_widget.clear()
        if cluster_id is None:
            return

        # 1. Get Pre-Calculated Features (from FeatureWorker)
        # This dictionary contains {'median_ei': ..., 'raw_snippets': ...}
        features = self.main_window.data_manager.get_lightweight_features(cluster_id)
        
        if not features or 'raw_snippets' not in features:
            self.plot_widget.setTitle(f"Cluster {cluster_id} - Waiting for data...")
            return

        snippets = features['raw_snippets'] # Shape: (n_channels, n_time, n_spikes)
        mean_ei = features['median_ei']     # Shape: (n_channels, n_time)

        # 2. Find Dominant Channel
        # We only plot the snippets for the channel with the biggest spike
        ptp = mean_ei.max(axis=1) - mean_ei.min(axis=1)
        dom_chan = np.argmax(ptp)

        # 3. Prepare Data for "Cloud" Plotting
        # Extract snippets for dominant channel: (n_time, n_spikes)
        raw_data = snippets[dom_chan, :, :]
        
        # Create Time Axis
        n_samples = raw_data.shape[0]
        sr = self.main_window.data_manager.sampling_rate
        t_ms = (np.arange(n_samples) - 20) / sr * 1000  # Center roughly on peak

        # 4. The Performance Trick (Connect with NaN)
        # We flatten all snippets into one single line separated by NaNs
        # This allows PyQtGraph to draw 100 spikes in 1 draw call instead of 100.
        n_spikes = raw_data.shape[1]
        
        if n_spikes > 0:
            # Create array with space for NaN at end of each trace
            x_connected = np.empty((n_samples + 1, n_spikes))
            y_connected = np.empty((n_samples + 1, n_spikes))
            
            x_connected[:-1, :] = t_ms[:, np.newaxis]
            x_connected[-1, :] = np.nan
            
            y_connected[:-1, :] = raw_data
            y_connected[-1, :] = np.nan
            
            # Flatten
            x_flat = x_connected.flatten()
            y_flat = y_connected.flatten()
            
            # Plot the "Cloud" (White, Transparent)
            self.plot_widget.plot(x_flat, y_flat, pen=pg.mkPen(color=(255, 255, 255, 30), width=1))

        # 5. Plot the Mean Waveform on top (Thick, Cyan)
        mean_trace = mean_ei[dom_chan, :]
        self.plot_widget.plot(t_ms, mean_trace, pen=pg.mkPen(color='#00e6a0', width=3))
        
        self.plot_widget.setTitle(f"Cluster {cluster_id} - Channel {dom_chan} ({n_spikes} snippets)")