from __future__ import annotations
import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel, QSlider
)
from qtpy.QtCore import Qt, QThread, QTimer, Signal
import pyqtgraph as pg
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gui.main_window import MainWindow

class RawPanel(QWidget):
    # Signals for status updates (optional, connect to main window if desired)
    status_message = Signal(str)
    error_message = Signal(str)

    def __init__(self, main_window: MainWindow, parent=None):
        super().__init__(parent)
        self.main_window = main_window  # For access to data_manager, etc.

        # --- State ---
        self.buffer_start_time = 0.0
        self.buffer_end_time = 0.0
        self.current_cluster_id = None
        self._manual_load = False
        self.worker_thread = None
        self.worker = None

        # --- UI ---
        layout = QVBoxLayout(self)
        time_control_layout = QHBoxLayout()
        self.time_input_hours = QLineEdit()
        self.time_input_hours.setPlaceholderText("HH")
        self.time_input_hours.setFixedWidth(40)
        self.time_input_minutes = QLineEdit()
        self.time_input_minutes.setPlaceholderText("MM")
        self.time_input_minutes.setFixedWidth(40)
        self.time_input_seconds = QLineEdit()
        self.time_input_seconds.setPlaceholderText("SS")
        self.time_input_seconds.setFixedWidth(40)
        time_control_layout.addWidget(QLabel("Time:"))
        time_control_layout.addWidget(self.time_input_hours)
        time_control_layout.addWidget(QLabel(":"))
        time_control_layout.addWidget(self.time_input_minutes)
        time_control_layout.addWidget(QLabel(":"))
        time_control_layout.addWidget(self.time_input_seconds)
        self.go_button = QPushButton("Go")
        time_control_layout.addWidget(self.go_button)
        self.load_prev_10s_button = QPushButton("Load Prev 10s")
        self.load_next_10s_button = QPushButton("Load Next 10s")
        time_control_layout.addWidget(self.load_prev_10s_button)
        time_control_layout.addWidget(self.load_next_10s_button)
        time_control_layout.addStretch()
        layout.addLayout(time_control_layout)

        self.plot = pg.PlotWidget(title="Raw Traces with Spike Templates")
        self.plot.plotItem.getViewBox().setMouseEnabled(y=False)
        layout.addWidget(self.plot)

        # --- Persistent Plot Items ---
        self.trace_plots = [self.plot.plot(pen=pg.mkPen(color=(150, 150, 150, 150), width=1)) for _ in range(3)]
        self.template_plot = self.plot.plot(pen=pg.mkPen('#FFA500', width=2))
        self.spike_marker_plot = self.plot.plot(pen=pg.mkPen('#FFFF00', width=1))
        self.template_plot.hide()
        self.spike_marker_plot.hide()

        # --- Connections ---
        self.go_button.clicked.connect(self._on_go_to_time)
        self.load_next_10s_button.clicked.connect(self.load_next_10s_data)
        self.load_prev_10s_button.clicked.connect(self.load_prev_10s_data)
        self.plot.sigXRangeChanged.connect(self._on_zoom)

        # --- Timer for async UI flag reset ---
        self._reset_manual_timer = QTimer(self)
        self._reset_manual_timer.setSingleShot(True)
        self._reset_manual_timer.timeout.connect(self._reset_manual_flag)

        # --- Internal update lock ---
        self._updating = False

    def load_data(self, cluster_id):
        """Load and display raw trace data for a cluster (called by main window)."""
        dm = self.main_window.data_manager
        if dm.raw_data_memmap is None:
            self.status_message.emit("No raw data file loaded.")
            return

        first_spike_time = dm.get_first_spike_time(cluster_id)
        start_time = max(0, (first_spike_time or 0) - 5.0)
        end_time = min(start_time + 30.0, dm.n_samples / dm.sampling_rate)
        self.buffer_start_time = start_time
        self.buffer_end_time = end_time
        self.current_cluster_id = cluster_id
        self._start_worker(cluster_id, start_time, end_time)

    def load_next_10s_data(self):
        """Load the next 10 seconds of data."""
        dm = self.main_window.data_manager
        if dm.raw_data_memmap is None or self.current_cluster_id is None:
            self.status_message.emit("No raw data file or cluster selected.")
            return
        self._manual_load = True
        buffer_start = self.buffer_end_time
        buffer_end = min(buffer_start + 10, dm.n_samples / dm.sampling_rate)
        self.buffer_start_time = buffer_start
        self.buffer_end_time = buffer_end
        self._start_worker(self.current_cluster_id, buffer_start, buffer_end)
        self.plot.setXRange(buffer_start, buffer_end, padding=0)
        self._reset_manual_timer.start(100)

    def load_prev_10s_data(self):
        """Load the previous 10 seconds of data."""
        dm = self.main_window.data_manager
        if dm.raw_data_memmap is None or self.current_cluster_id is None:
            self.status_message.emit("No raw data file or cluster selected.")
            return
        self._manual_load = True
        buffer_end = self.buffer_start_time
        buffer_start = max(0, buffer_end - 10)
        self.buffer_start_time = buffer_start
        self.buffer_end_time = buffer_end
        self._start_worker(self.current_cluster_id, buffer_start, buffer_end)
        self.plot.setXRange(buffer_start, buffer_end, padding=0)
        self._reset_manual_timer.start(100)

    def _on_go_to_time(self):
        """Jump to a specific time (not fully implemented here)."""
        # You can implement this as needed, similar to your callbacks.on_go_to_time
        pass

    def _on_zoom(self):
        """Handle seamless infinite scroll when user pans/zooms."""
        if self._updating or self._manual_load:
            return
        self._updating = True
        try:
            if self.current_cluster_id is None:
                return
            view_range = self.plot.viewRange()
            x_min, x_max = view_range[0]
            buffer_duration = 60.0
            buffer_threshold = 30.0
            dm = self.main_window.data_manager
            max_time = dm.n_samples / dm.sampling_rate
            load_new = False
            if self.current_cluster_id != self.current_cluster_id:
                load_new = True
            elif x_max > (self.buffer_end_time - buffer_threshold) and self.buffer_end_time < max_time:
                load_new = True
            elif x_min < (self.buffer_start_time + buffer_threshold) and self.buffer_start_time > 0:
                load_new = True
            if load_new:
                center_time = (x_min + x_max) / 2.0
                new_start = max(0, center_time - buffer_duration / 2.0)
                new_end = min(new_start + buffer_duration, max_time)
                if new_end > max_time:
                    new_end = max_time
                    new_start = max(0, new_end - buffer_duration)
                self.buffer_start_time = new_start
                self.buffer_end_time = new_end
                self._start_worker(self.current_cluster_id, new_start, new_end)
        finally:
            self._updating = False

    def _start_worker(self, cluster_id, start_time, end_time):
        """Start background worker to load data and plot."""
        # Stop any existing worker
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()
        # Get channel info
        dm = self.main_window.data_manager
        features = dm.get_lightweight_features(cluster_id)
        if not features:
            self.status_message.emit("No features for cluster.")
            return
        p2p = features['median_ei'].max(axis=1) - features['median_ei'].min(axis=1)
        dom_chan = np.argmax(p2p)
        nearest_channels = dm.get_nearest_channels(dom_chan, n_channels=3)
        # Import worker here to avoid circular import
        from gui.workers import RawTraceWorker
        self.worker = RawTraceWorker(dm, cluster_id, nearest_channels, start_time, end_time)
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker.data_loaded.connect(self._on_data_loaded)
        self.worker.error.connect(self._on_data_error)
        self.worker_thread.started.connect(self.worker.run)
        self.worker_thread.start()
        self.status_message.emit(f"Loading raw trace for cluster {cluster_id}...")

    def _on_data_loaded(self, cluster_id, raw_trace_data, start_time, end_time):
        """Plot loaded data using persistent PlotDataItems."""
        try:
            dm = self.main_window.data_manager
            features = dm.get_lightweight_features(cluster_id)
            if not features:
                self.status_message.emit(f"Could not retrieve features for cluster {cluster_id}")
                return

            median_ei = features['median_ei']
            if median_ei.size == 0 or len(median_ei.shape) < 2:
                self.status_message.emit(f"Invalid data for cluster {cluster_id}")
                return

            p2p = median_ei.max(axis=1) - median_ei.min(axis=1)
            dom_chan = np.argmax(p2p)
            nearest_channels = dm.get_nearest_channels(dom_chan, n_channels=3)

            time_axis = np.linspace(start_time, end_time, raw_trace_data.shape[1]) if raw_trace_data.shape[1] > 0 else np.array([])
            vertical_offset = max(np.max(np.abs(raw_trace_data)), 1) * 2.0 if raw_trace_data.size > 0 else 100

            self.plot.setTitle(f"Raw Traces for Cluster {cluster_id} - Dominant: {dom_chan}")
            self.plot.setLabel('bottom', 'Time (s)')
            self.plot.setLabel('left', 'Amplitude (µV)')

            # Update trace plots
            num_traces = len(raw_trace_data)
            for i in range(len(self.trace_plots)):
                if i < num_traces:
                    chan_idx, trace = nearest_channels[i], raw_trace_data[i]
                    offset_trace = trace + i * vertical_offset
                    pen_color = (150, 150, 150, 150) if chan_idx == dom_chan else (120, 120, 150, 100)
                    self.trace_plots[i].setData(time_axis, offset_trace)
                    self.trace_plots[i].setPen(pg.mkPen(color=pen_color, width=1))
                    self.trace_plots[i].show()
                else:
                    self.trace_plots[i].hide()

            # Plot spikes/templates
            all_spikes = dm.get_cluster_spikes_in_window(cluster_id, start_time, end_time)
            if len(all_spikes) > 0:
                window_spikes_sec = all_spikes / dm.sampling_rate
                visible_range = self.plot.viewRange()[0]
                visible_duration = visible_range[1] - visible_range[0]

                if visible_duration < 0.1:  # Zoomed in: show templates
                    self.spike_marker_plot.hide()
                    try:
                        dom_idx_in_list = nearest_channels.index(dom_chan)
                    except ValueError:
                        dom_idx_in_list = 0
                    
                    template_waveform = median_ei[dom_chan]
                    template_len = len(template_waveform)
                    trough_idx = np.argmin(template_waveform)
                    time_to_trough = trough_idx / dm.sampling_rate
                    
                    dominant_trace = raw_trace_data[dom_idx_in_list] if dom_idx_in_list < len(raw_trace_data) else np.array([0])
                    baseline_offset = np.median(dominant_trace)
                    
                    all_template_points = np.empty((len(window_spikes_sec) * (template_len + 1), 2))
                    all_template_points.fill(np.nan)

                    for i, spike_time in enumerate(window_spikes_sec):
                        start_idx = i * (template_len + 1)
                        end_idx = start_idx + template_len
                        start_time_template = spike_time - time_to_trough
                        end_time_template = start_time_template + (template_len - 1) / dm.sampling_rate
                        
                        all_template_points[start_idx:end_idx, 0] = np.linspace(start_time_template, end_time_template, template_len)
                        all_template_points[start_idx:end_idx, 1] = template_waveform + baseline_offset + (dom_idx_in_list * vertical_offset)
                    
                    self.template_plot.setData(all_template_points[:, 0], all_template_points[:, 1])
                    self.template_plot.show()
                else:  # Zoomed out: show vertical lines
                    self.template_plot.hide()
                    view_box = self.plot.getViewBox()
                    y_range = view_box.viewRange()[1]
                    y_min, y_max = y_range[0], y_range[1]
                    
                    line_points = np.empty((len(window_spikes_sec) * 3, 2))
                    line_points.fill(np.nan)

                    for i, spike_time in enumerate(window_spikes_sec):
                        idx = i * 3
                        line_points[idx] = (spike_time, y_min)
                        line_points[idx + 1] = (spike_time, y_max)
                    
                    self.spike_marker_plot.setData(line_points[:, 0], line_points[:, 1])
                    self.spike_marker_plot.show()
            else:
                self.template_plot.hide()
                self.spike_marker_plot.hide()

        except Exception as e:
            # In case of error, ensure plots are cleared to avoid showing stale data
            for plot_item in self.trace_plots:
                plot_item.clear()
            self.template_plot.clear()
            self.spike_marker_plot.clear()
            self.status_message.emit(f"Error plotting raw trace: {str(e)}")
        finally:
            self.status_message.emit(f"Raw trace loaded for cluster {cluster_id}")
            if self.worker_thread:
                self.worker_thread.quit()
                self.worker_thread.wait()
                self.worker_thread = None
                self.worker = None

    def _on_data_error(self, msg):
        self.error_message.emit(msg)
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
            self.worker = None

    def _reset_manual_flag(self):
        self._manual_load = False