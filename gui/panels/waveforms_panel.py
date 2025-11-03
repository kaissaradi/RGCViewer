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

class WaveformPanel(QWidget):
    """
    Panel for displaying cluster waveforms, ISI histogram, and firing rate.
    """
    def __init__(self, main_window: MainWindow, parent=None):
        super().__init__(parent)
        self.main_window = main_window  # Needed for data access and callbacks
        layout = QVBoxLayout(self)
        splitter = QSplitter()
        splitter.setOrientation(pg.QtCore.Qt.Vertical)

        # sampling rate
        self.sampling_rate = 20000.0  # Default
        if hasattr(self.main_window.data_manager, 'sampling_rate'):
            self.sampling_rate = self.main_window.data_manager.sampling_rate

        # --- Top: Waveform grid plot ---
        self.waveform_grid_widget = pg.GraphicsLayoutWidget()
        self.waveform_grid_plot = self.waveform_grid_widget.addPlot()
        self.waveform_grid_plot.setAspectLocked(True)
        self.waveform_grid_plot.hideAxis('bottom')
        self.waveform_grid_plot.hideAxis('left')
        splitter.addWidget(self.waveform_grid_widget)

        # --- Bottom: ISI and FR side by side ---
        bottom_splitter = QSplitter()
        bottom_splitter.setOrientation(pg.QtCore.Qt.Horizontal)
        self.isi_plot = pg.PlotWidget(title="Inter-Spike Interval (ISI) Histogram")
        self.fr_plot = pg.PlotWidget(title="Smoothed Firing Rate")
        bottom_splitter.addWidget(self.isi_plot)
        bottom_splitter.addWidget(self.fr_plot)
        splitter.addWidget(bottom_splitter)

        splitter.setSizes([600, 400])
        layout.addWidget(splitter)

        # --- Interactive parameters ---
        self.wf_scale = 100.0
        self.t_scale = 20.0
        self.x_ch_sep = 1.5
        self.y_ch_sep = 1.0
        self.overlap = True
        self.wf_norm = False

        # For key toggles 
        self.setFocusPolicy(Qt.NoFocus)
        self.waveform_grid_widget.setFocusPolicy(Qt.NoFocus)
        self.waveform_grid_widget.installEventFilter(self)
        self.waveform_grid_plot.setFocusPolicy(Qt.NoFocus)

        # Mouse/keyboard interaction
        self.waveform_grid_widget.scene().sigMouseClicked.connect(self._focus_waveform_plot)
        self.waveform_grid_widget.keyPressEvent = self._on_key_press
        # self.waveform_grid_widget.wheelEvent = self._on_mouse_wheel

        # Store last data for redraw
        self._last_templates = None
        self._last_channel_positions = None
        self._last_cluster_ids = None
        self._last_colors = None

    def _focus_waveform_plot(self, ev):
        self.waveform_grid_widget.setFocus()
    
    def _on_mouse_wheel(self, ev):
        # For Qt5/6, angleDelta().y() gives the wheel delta
        delta = ev.angleDelta().y()
        modifiers = ev.modifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            self.t_scale *= 1.1 if delta > 0 else 1/1.1
        elif modifiers & Qt.KeyboardModifier.ShiftModifier:
            self.x_ch_sep *= 1.1 if delta > 0 else 1/1.1
        else:
            self.wf_scale *= 1.1 if delta > 0 else 1/1.1
        self._redraw_waveform_grid()
        ev.accept()
        print(f'[DEBUG] Mouse wheel delta: {delta}')

    def _on_key_press(self, ev):
        key = ev.key()
        if key == Qt.Key.Key_O:
            self.overlap = not self.overlap
        elif key == Qt.Key.Key_N:
            self.wf_norm = not self.wf_norm
            if self.wf_norm:
                self.wf_scale = 10.0
            else:
                self.wf_scale = 100.0
        self._redraw_waveform_grid()

    def _redraw_waveform_grid(self):
        if self._last_templates is not None and self._last_channel_positions is not None:
            self._plot_waveforms_on_grid(
                self._last_templates,
                self._last_channel_positions,
                cluster_ids=self._last_cluster_ids,
                colors=self._last_colors
            )

    def update_all(self, cluster_ids, color_map=None, colors=None):
        # If single cluster_id, make it a list
        if np.isscalar(cluster_ids):
            cluster_ids = [cluster_ids]
        cluster_ids = np.array(cluster_ids, dtype=int)
        n_clusters = len(cluster_ids)
        self._last_cluster_ids = cluster_ids

        if colors is None:
            if color_map and cluster_ids:
                colors = [color_map.get(cid, (200, 200, 255)) for cid in cluster_ids]
            else:
                colors = [pg.intColor(i, hues=n_clusters) for i in range(n_clusters)]
        self.update_waveforms(cluster_ids, colors)
        self.update_isi(cluster_ids, colors)
        self.update_fr(cluster_ids, colors)
    
    def update_waveforms(self, cluster_ids, colors):
        """
        Update the waveform grid plot for the given cluster(s).
        Args:
            cluster_ids: list of cluster IDs to plot
            color_map: optional dict mapping cluster_id -> color (tuple or string)
        """
        data_manager = self.main_window.data_manager

        if data_manager is None or not hasattr(data_manager, 'templates') or not hasattr(data_manager, 'channel_positions'):
            self.waveform_grid_plot.clear()
            self.waveform_grid_plot.setTitle("Waveforms (No data)")
            return

        templates = []
       
        for cid in cluster_ids:
            # Use the template for this cluster
            if hasattr(data_manager, 'templates'):
                # templates shape: (n_clusters, n_timepoints, n_channels)
                if cid < data_manager.templates.shape[0]:
                    templates.append(data_manager.templates[cid])
        if not templates:
            self.waveform_grid_plot.clear()
            self.waveform_grid_plot.setTitle("Waveforms (No data)")
            return

        templates = np.stack(templates, axis=0)
        channel_positions = data_manager.channel_positions

        self._last_templates = templates
        self._last_channel_positions = channel_positions
        self._last_cluster_ids = cluster_ids
        self._last_colors = colors

        self._plot_waveforms_on_grid(templates, channel_positions, cluster_ids=cluster_ids, colors=colors)

    def _plot_waveforms_on_grid(
        self, templates, channel_positions, colors, cluster_ids, wf_scale=None, t_scale=None, amplitude_threshold=0.00,
        overlap=None, x_ch_sep=None, y_ch_sep=None, wf_norm=None,
    ):
        # Use current settings if not provided
        wf_scale = wf_scale if wf_scale is not None else self.wf_scale
        t_scale = t_scale if t_scale is not None else self.t_scale
        x_ch_sep = x_ch_sep if x_ch_sep is not None else self.x_ch_sep
        y_ch_sep = y_ch_sep if y_ch_sep is not None else self.y_ch_sep
        overlap = overlap if overlap is not None else self.overlap
        wf_norm = wf_norm if wf_norm is not None else self.wf_norm

        n_templates, n_timepoints, n_channels = templates.shape

        # Scale channel positions
        ch_pos = channel_positions.copy()
        ch_pos[:, 0] = ch_pos[:, 0] * x_ch_sep
        ch_pos[:, 1] = ch_pos[:, 1] * y_ch_sep

        t = np.arange(n_timepoints)
        t = t - t.mean()
        t = t / t.max()
        t = t * t_scale

        # For each channel, find which templates have significant amplitude
        # channel_to_templates = {}
        # for ch in range(n_channels):
        #     channel_to_templates[ch] = []
        #     for i in range(n_templates):
        #         template = templates[i]
        #         amplitudes = template.max(axis=0) - template.min(axis=0)
        #         best_channel = np.argmax(amplitudes)
        #         max_amp = amplitudes[best_channel]
        #         if amplitudes[ch] > amplitude_threshold * max_amp:
        #             channel_to_templates[ch].append(i)

        self.waveform_grid_plot.clear()
        for ch in range(n_channels):
            templates_here = self.main_window.data_manager.channel_to_templates[ch]
            templates_here = [i for i in templates_here if i in cluster_ids]
            n_here = len(templates_here)
            if n_here == 0:
                continue
            x, y = ch_pos[ch]
            # Draw zero line and label only once per channel
            self.waveform_grid_plot.plot([x - t_scale, x + t_scale], [y, y], pen=pg.mkPen('k', width=1))
            text_item = pg.TextItem(text=str(ch), color='k', anchor=(1, 1))
            text_item.setPos(x - t_scale, y)
            self.waveform_grid_plot.addItem(text_item)
            for idx, i in enumerate(templates_here):
                # Get index into input templates, which is index into cluster_ids
                t_idx = np.where(cluster_ids == i)[0][0]
                template = templates[t_idx]
                wf = template[:, ch]
                if wf_norm:
                    wf = wf / np.max(np.abs(wf)) if np.max(np.abs(wf)) > 0 else wf
                wf = wf * wf_scale
                # Offset for non-overlap
                if overlap or n_here == 1:
                    x_offset = 0
                    opacity = 0.8
                else:
                    x_offset = ((idx - (n_here-1)/2) * t_scale)*2
                    opacity = 1.0
                amplitudes = template.max(axis=0) - template.min(axis=0)
                best_channel = np.argmax(amplitudes)
                lw = 2 if ch == best_channel else 1
                color = colors[t_idx] if isinstance(colors[t_idx], (tuple, list)) else pg.mkColor(colors[t_idx])
                pen = pg.mkPen(color=color, width=lw)
                self.waveform_grid_plot.plot(x + t + x_offset, y + wf, pen=pen)

        self.waveform_grid_plot.setTitle("Waveforms on Electrode Grid")
        self.waveform_grid_plot.setAspectLocked(True)
        self.waveform_grid_plot.hideAxis('bottom')
        self.waveform_grid_plot.hideAxis('left')

    def update_isi(self, cluster_ids, colors):

        data_manager = self.main_window.data_manager
        self.isi_plot.clear()
        
        for idx, cid in enumerate(cluster_ids):
            sts = data_manager.get_cluster_spikes(cid)

        
            if sts is None or len(sts) < 2:
                print(f'[DEBUG] No spike times for cluster {cid}')
                continue
            
            isi_ms = np.diff(sts) / self.sampling_rate * 1000
            bins = np.linspace(0, 50, 101)
            y, x = np.histogram(isi_ms, bins=bins)

            color = colors[idx] if isinstance(colors[idx], (tuple, list)) else pg.mkColor(colors[idx])
            pen = pg.mkPen(color=color, width=2)
            # self.isi_plot.plot(x, y, stepMode="center", fillLevel=0, brush=(0, 163, 224, 150))
            self.isi_plot.plot(x, y, stepMode="center", pen=pen)
        
        self.isi_plot.addLine(x=ISI_REFRACTORY_PERIOD_MS, pen=pg.mkPen('r', style=Qt.PenStyle.DashLine, width=2))
        self.isi_plot.setTitle(f"ISI Histogram (Cluster {cluster_ids})")
        self.isi_plot.setLabel('bottom', 'ISI (ms)')
        self.isi_plot.setLabel('left', 'Count')

    def update_fr(self, cluster_ids, colors):
        self.fr_plot.clear()
        data_manager = self.main_window.data_manager

        for idx, cluster_id in enumerate(cluster_ids):
            sts = data_manager.get_cluster_spikes(cluster_id)

            if sts is None or len(sts) < 2:
                print(f'[DEBUG] No spike times for cluster {cluster_id}')
                continue

            spike_times_sec = sts / self.sampling_rate
            duration = spike_times_sec[-1] - spike_times_sec[0]
            if duration <= 0:
                print(f'[DEBUG] No valid duration for cluster {cluster_id}')
                continue

            bins = np.arange(0, duration+1, 1)
            counts, _ = np.histogram(spike_times_sec, bins=bins)
            rate = gaussian_filter1d(counts.astype(float), sigma=5)
            color = colors[idx] if isinstance(colors[idx], (tuple, list)) else pg.mkColor(colors[idx])
            pen = pg.mkPen(color=color, width=2)
            self.fr_plot.plot(bins[:-1], rate, pen=pen)

        self.fr_plot.setTitle(f"Firing Rate (Cluster {cluster_ids})")
        self.fr_plot.setLabel('bottom', 'Time (s)')
        self.fr_plot.setLabel('left', 'Rate (Hz)')

    def clear(self):
        self.waveform_grid_plot.clear()
        self.isi_plot.clear()
        self.fr_plot.clear()
        self.waveform_grid_plot.setTitle("Waveforms (Sampled)")
        self.isi_plot.setTitle("ISI Histogram")
        self.fr_plot.setTitle("Firing Rate")