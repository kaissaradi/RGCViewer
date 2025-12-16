from __future__ import annotations
from qtpy.QtWidgets import QWidget, QVBoxLayout, QSplitter
import pyqtgraph as pg
import numpy as np
from qtpy.QtCore import Qt
from scipy.ndimage import gaussian_filter1d
from analysis.constants import ISI_REFRACTORY_PERIOD_MS
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gui.main_window import MainWindow
import logging
logger = logging.getLogger(__name__)

class StandardPlotsPanel(QWidget):
    """
    New "Home" tab containing quick diagnostic plots in a 2x2 grid:
    - Template (Top Left): Kilosort template for cluster's dominant channel
    - Autocorrelation (Top Right): New histogram-based ACG plot
    - ISI Distribution (Bottom Left): Replicates old plotting aesthetics
    - Firing Rate (Bottom Right): Replicates old plotting aesthetics
    """
    def __init__(self, main_window: MainWindow, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.sampling_rate = self.main_window.data_manager.sampling_rate if self.main_window.data_manager else 20000.0
        
        # Create main layout
        layout = QVBoxLayout(self)
        
        # Create main splitter (vertical) to contain two horizontal splitters
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Top horizontal splitter for Template and Autocorrelation
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Template plot (Top Left)
        self.template_plot = pg.PlotWidget(title="Template (Dominant Channel)")
        top_splitter.addWidget(self.template_plot)
        
        # Autocorrelation plot (Top Right) 
        self.acg_plot = pg.PlotWidget(title="Autocorrelation")
        top_splitter.addWidget(self.acg_plot)
        
        # Add top splitter to main splitter
        main_splitter.addWidget(top_splitter)
        
        # Bottom horizontal splitter for ISI and Firing Rate
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # ISI Distribution plot (Bottom Left)
        self.isi_plot = pg.PlotWidget(title="Inter-Spike Interval (ISI) Histogram")
        bottom_splitter.addWidget(self.isi_plot)
        
        # Firing Rate plot (Bottom Right)
        self.fr_plot = pg.PlotWidget(title="Firing Rate")
        bottom_splitter.addWidget(self.fr_plot)
        
        # Add bottom splitter to main splitter
        main_splitter.addWidget(bottom_splitter)
        
        # Set sizes to make the layout approximately even
        top_splitter.setSizes([400, 400])
        bottom_splitter.setSizes([400, 400])
        main_splitter.setSizes([400, 400])
        
        # Add main splitter to the layout
        layout.addWidget(main_splitter)

    def update_all(self, cluster_id):
        """
        Update all plots for the given cluster ID
        """
        if self.main_window.data_manager is None:
            logger.warning("No data manager available for StandardPlotsPanel")
            return
            
        data_manager = self.main_window.data_manager
        
        # Update each plot
        self._update_template_plot(cluster_id, data_manager)
        self._update_acg_plot(cluster_id, data_manager)
        self._update_isi_plot(cluster_id, data_manager)
        self._update_firing_rate_plot(cluster_id, data_manager)

    def _update_template_plot(self, cluster_id, data_manager):
        """Update the template plot for the dominant channel."""
        self.template_plot.clear()
        
        if hasattr(data_manager, 'templates') and cluster_id < data_manager.templates.shape[0]:
            # Get the template for this cluster: shape is (n_clusters, n_timepoints, n_channels)
            template = data_manager.templates[cluster_id]
            
            # Find the dominant channel (largest amplitude across time)
            ptp_amplitudes = template.max(axis=0) - template.min(axis=0)  # amplitude per channel
            dominant_channel = np.argmax(ptp_amplitudes)
            
            # Plot the template for the dominant channel
            dominant_template = template[:, dominant_channel]
            
            # Create time axis in ms
            time_axis = np.arange(len(dominant_template))
            time_axis = (time_axis - len(time_axis)//2) / self.sampling_rate * 1000  # Convert to ms
            
            self.template_plot.plot(time_axis, dominant_template, pen=pg.mkPen('w', width=2))
            self.template_plot.setTitle(f"Template (Cluster {cluster_id}, Ch {dominant_channel})")
        else:
            self.template_plot.setTitle(f"Template (Cluster {cluster_id} - No data)")

    def _update_acg_plot(self, cluster_id, data_manager):
        """Update the autocorrelation plot."""
        self.acg_plot.clear()
        
        spike_times = data_manager.get_cluster_spikes(cluster_id)
        if spike_times is None or len(spike_times) < 2:
            self.acg_plot.setTitle(f"Autocorrelation (Cluster {cluster_id} - No data)")
            return
            
        # Calculate all pairwise differences for spikes within a reasonable window
        max_lag = int(0.1 * self.sampling_rate)  # 100ms window
        spike_times = np.sort(spike_times)
        
        # Calculate differences efficiently
        diffs = []
        for i, st in enumerate(spike_times):
            # Look at following spikes within the lag window
            future_spikes = spike_times[i+1:]
            lag_diffs = future_spikes - st
            lag_diffs = lag_diffs[lag_diffs <= max_lag]
            diffs.extend(lag_diffs)
            # Stop if we're getting too far
            if len(lag_diffs) == 0:
                continue
                
        if len(diffs) > 0:
            diffs = np.concatenate([np.array(diffs), -np.array(diffs)])  # Include negative lags
            bins = np.linspace(-max_lag, max_lag, 101)
            counts, bin_edges = np.histogram(diffs, bins=bins)
            
            # Convert bins to ms
            bins_ms = bin_edges / self.sampling_rate * 1000
            
            # Plot as step histogram
            self.acg_plot.plot(bins_ms, counts, stepMode=True, fillLevel=0, 
                              brush=(100, 100, 200, 100), pen=pg.mkPen('b', width=1))
            self.acg_plot.setTitle(f"Autocorrelation (Cluster {cluster_id})")
        else:
            self.acg_plot.setTitle(f"Autocorrelation (Cluster {cluster_id})")

    def _update_isi_plot(self, cluster_id, data_manager):
        """Update the ISI plot with the old aesthetic."""
        self.isi_plot.clear()
        
        spike_times = data_manager.get_cluster_spikes(cluster_id)
        if spike_times is None or len(spike_times) < 2:
            self.isi_plot.setTitle(f"ISI Histogram (Cluster {cluster_id} - No data)")
            return
            
        # Calculate ISIs in ms
        isis_ms = np.diff(np.sort(spike_times)) / self.sampling_rate * 1000
        
        # Create histogram
        bins = np.linspace(0, 50, 101)  # 0 to 50 ms in 100 bins
        y, x = np.histogram(isis_ms, bins=bins)
        
        # Plot with old aesthetic: step mode, filled, specific blue color
        self.isi_plot.plot(x, y, stepMode="center", fillLevel=0,
                          brush=(0, 163, 224, 150),  # The specific "Old" Blue
                          pen=pg.mkPen(color='#33b5e5', width=2))
        
        # Add vertical dashed red line at refractory period (1.5 or 2.0 ms)
        self.isi_plot.addLine(x=ISI_REFRACTORY_PERIOD_MS, pen=pg.mkPen('r', style=Qt.PenStyle.DashLine, width=2))
        
        self.isi_plot.setTitle(f"ISI Histogram (Cluster {cluster_id})")
        self.isi_plot.setLabel('bottom', 'ISI (ms)')
        self.isi_plot.setLabel('left', 'Count')

    def _update_firing_rate_plot(self, cluster_id, data_manager):
        """Update the firing rate plot with the old aesthetic."""
        self.fr_plot.clear()
        
        spike_times = data_manager.get_cluster_spikes(cluster_id)
        if spike_times is None or len(spike_times) < 2:
            self.fr_plot.setTitle(f"Firing Rate (Cluster {cluster_id} - No data)")
            return
            
        # Convert spike times to seconds
        spike_times_sec = spike_times / self.sampling_rate
        
        # Calculate total duration and create 1-second bins
        duration = spike_times_sec.max() - spike_times_sec.min()
        if duration <= 1:  # If less than 1 second, use total duration as bin
            bins = np.array([spike_times_sec.min(), spike_times_sec.max()])
        else:
            bins = np.arange(spike_times_sec.min(), spike_times_sec.max() + 1, 1)
            
        # Calculate counts per bin
        counts, _ = np.histogram(spike_times_sec, bins=bins)
        
        # Apply Gaussian smoothing with sigma=5 as in the old code
        if len(counts) > 1:
            rate = gaussian_filter1d(counts.astype(float), sigma=5)
            
            # Convert bin edges to bin centers for plotting
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # Plot with yellow line as in old aesthetic
            self.fr_plot.plot(bin_centers, rate, pen=pg.mkPen('y', width=2))  # 'y' = Yellow
            self.fr_plot.setTitle(f"Firing Rate (Cluster {cluster_id})")
            self.fr_plot.setLabel('bottom', 'Time (s)')
            self.fr_plot.setLabel('left', 'Rate (Hz)')
        else:
            self.fr_plot.setTitle(f"Firing Rate (Cluster {cluster_id})")