import numpy as np
import pyqtgraph as pg
from qtpy.QtWidgets import QWidget, QVBoxLayout, QSplitter
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

    def update_all(self, cluster_id):
        if cluster_id is None: return
        dm = self.main_window.data_manager
        
        # --- 1. Template Grid (Spatial) ---
        self.grid_plot.clear()
        if hasattr(dm, 'templates') and cluster_id < dm.templates.shape[0]:
            template = dm.templates[cluster_id] # (n_time, n_chan)
            # Use channel positions to plot geometric layout
            # (Simplified logic: Loop channels, offset by position, plot line)
            pos = dm.channel_positions
            # Scale factors for visualization
            x_scale, y_scale = 1.5, 1.0 
            
            # Find relevant channels (thresholding)
            ptp = template.max(axis=0) - template.min(axis=0)
            max_ptp = ptp.max()
            relevant_chans = np.where(ptp > 0.1 * max_ptp)[0]
            
            for ch in relevant_chans:
                x, y = pos[ch]
                trace = template[:, ch]
                # Normalize and scale trace to fit grid
                trace_scaled = (trace / max_ptp) * 20 
                t_offset = np.linspace(-10, 10, len(trace))
                
                # Plot
                self.grid_plot.plot(x * x_scale + t_offset, y * y_scale + trace_scaled, 
                                    pen=pg.mkPen('#00e6a0', width=1))

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
        y, x = np.histogram(isi_ms, bins=np.linspace(0, 50, 101))
        
        self.isi_plot.plot(x, y, stepMode="center", fillLevel=0,
                           brush=(0, 163, 224, 150), 
                           pen=pg.mkPen('#33b5e5', width=2))
        self.isi_plot.addItem(pg.InfiniteLine(ISI_REFRACTORY_PERIOD_MS, angle=90, 
                                              pen=pg.mkPen('r', style=Qt.DashLine)))

        # --- 4. Firing Rate (Yellow Smooth) ---
        self.fr_plot.clear()
        spikes_sec = spikes / sr
        bins = np.arange(0, spikes_sec.max() + 1, 1)
        counts, _ = np.histogram(spikes_sec, bins=bins)
        rate = gaussian_filter1d(counts.astype(float), sigma=5) # Smooth
        
        self.fr_plot.plot(bins[:-1], rate, pen=pg.mkPen('#ffeb3b', width=2))