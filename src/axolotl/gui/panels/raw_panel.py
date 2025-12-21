from __future__ import annotations
import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
)
from qtpy.QtCore import QThread, QTimer, Signal, Qt
import pyqtgraph as pg
from typing import TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


class RawPanel(QWidget):
    status_message = Signal(str)
    error_message = Signal(str)

    def __init__(self, main_window: MainWindow, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        # --- State ---
        self.buffer_start_time = 0.0
        self.buffer_end_time = 0.0
        self.current_cluster_id = None
        self._manual_load = False
        self.worker_thread = None
        self.worker = None
        self.window_duration = 2.0
        self.current_spike_index = 0
        self.spike_times_sec = np.array([])  # Initialize empty array

        # --- DEBUG: Check data manager ---
        dm = self.main_window.data_manager
        if dm and dm.raw_data_memmap is not None:
            logger.info("RawPanel: Data manager has raw data loaded")
        else:
            logger.warning("RawPanel: Data manager has NO raw data loaded!")
        
        # --- UI ---
        layout = QVBoxLayout(self)
        
        # Navigation controls
        nav_layout = QHBoxLayout()
        
        self.prev_spike_button = QPushButton("← Prev Spike")
        self.next_spike_button = QPushButton("Next Spike →")
        self.prev_time_button = QPushButton("← Prev 2s")
        self.next_time_button = QPushButton("Next 2s →")
        
        self.time_label = QLabel("Time: 0.0s - 2.0s")
        self.spike_label = QLabel("Spike: 0 / 0")
        
        nav_layout.addWidget(self.prev_spike_button)
        nav_layout.addWidget(self.next_spike_button)
        nav_layout.addWidget(self.prev_time_button)
        nav_layout.addWidget(self.next_time_button)
        nav_layout.addStretch()
        nav_layout.addWidget(self.spike_label)
        nav_layout.addWidget(self.time_label)
        
        layout.addLayout(nav_layout)

        # Plot widget
        self.plot = pg.PlotWidget(title="Raw Traces with Spike Markers")
        self.plot.plotItem.getViewBox().setMouseEnabled(y=False)
        self.plot.setLabel('bottom', 'Time (s)')
        self.plot.setLabel('left', 'Amplitude (µV)')
        layout.addWidget(self.plot)

        # --- Persistent Plot Items ---
        self.trace_plots = [
            self.plot.plot(
                pen=pg.mkPen(
                    color=(150, 150, 150, 150),
                    width=1
                )
            ) for _ in range(3)
        ]
        
        # Spike markers - thin, transparent dotted lines
        self.spike_marker_plot = self.plot.plot(
            pen=pg.mkPen(
                color=(255, 255, 0, 80),  # Yellow, low alpha
                width=1,
                style=Qt.DotLine
            )
        )
        
        for plot_item in self.trace_plots:
            plot_item.hide()
        self.spike_marker_plot.hide()

        # --- Connections ---
        self.prev_spike_button.clicked.connect(self._on_prev_spike)
        self.next_spike_button.clicked.connect(self._on_next_spike)
        self.prev_time_button.clicked.connect(self._on_prev_time)
        self.next_time_button.clicked.connect(self._on_next_time)
        
        # --- Timer for async UI flag reset ---
        self._reset_manual_timer = QTimer(self)
        self._reset_manual_timer.setSingleShot(True)
        self._reset_manual_timer.timeout.connect(self._reset_manual_flag)

        # --- Internal update lock ---
        self._updating = False
        
        logger.info("RawPanel initialized successfully")

    def load_data(self, cluster_id):
        """Load and display raw trace data for a cluster."""
        logger.info(f"RawPanel.load_data called for cluster {cluster_id}")
        
        if not self.isVisible():
            logger.debug("RawPanel is not visible, skipping load")
            return
            
        dm = self.main_window.data_manager
        if dm.raw_data_memmap is None:
            logger.error("No raw data file loaded in data manager!")
            self.status_message.emit("No raw data file loaded.")
            return

        # Get spike times for this cluster
        spike_times = dm.get_cluster_spikes(cluster_id)
        logger.info(f"Found {len(spike_times)} spikes for cluster {cluster_id}")
        
        if len(spike_times) == 0:
            self.status_message.emit(f"No spikes found for cluster {cluster_id}")
            return
            
        # Convert to seconds
        spike_times_sec = spike_times / dm.sampling_rate
        
        # Store spike times and reset index
        self.spike_times_sec = spike_times_sec
        self.current_spike_index = 0
        
        # Start at first spike
        first_spike_time = spike_times_sec[0]
        start_time = max(0, first_spike_time - self.window_duration / 2)
        end_time = start_time + self.window_duration
        
        self.buffer_start_time = start_time
        self.buffer_end_time = end_time
        self.current_cluster_id = cluster_id
        
        self._update_spike_label()
        self._update_time_label()
        self._start_worker(cluster_id, start_time, end_time)

    def _update_spike_label(self):
        """Update the spike counter label."""
        if hasattr(self, 'spike_times_sec') and len(self.spike_times_sec) > 0:
            total_spikes = len(self.spike_times_sec)
            # Find how many spikes are in current window
            window_spikes = self.spike_times_sec[
                (self.spike_times_sec >= self.buffer_start_time) & 
                (self.spike_times_sec <= self.buffer_end_time)
            ]
            current_spike_in_window = len(window_spikes)
            self.spike_label.setText(
                f"Spike: {self.current_spike_index + 1} / {total_spikes} "
                f"(+{current_spike_in_window - 1} in view)"
            )
        else:
            self.spike_label.setText("Spike: 0 / 0")

    def _on_prev_spike(self):
        """Navigate to previous spike."""
        logger.debug("Prev spike button clicked")
        if self.current_cluster_id is None or not hasattr(self, 'spike_times_sec') or len(self.spike_times_sec) == 0:
            logger.warning("Cannot navigate: no cluster or spikes")
            return
            
        if self.current_spike_index > 0:
            self.current_spike_index -= 1
            spike_time = self.spike_times_sec[self.current_spike_index]
            start_time = max(0, spike_time - self.window_duration / 2)
            logger.info(f"Navigating to spike {self.current_spike_index} at time {spike_time:.3f}s")
            self._manual_load = True
            self._load_window(start_time, self.window_duration)
            self._reset_manual_timer.start(100)

    def _on_next_spike(self):
        """Navigate to next spike."""
        logger.debug("Next spike button clicked")
        if self.current_cluster_id is None or not hasattr(self, 'spike_times_sec') or len(self.spike_times_sec) == 0:
            logger.warning("Cannot navigate: no cluster or spikes")
            return
            
        if self.current_spike_index < len(self.spike_times_sec) - 1:
            self.current_spike_index += 1
            spike_time = self.spike_times_sec[self.current_spike_index]
            start_time = max(0, spike_time - self.window_duration / 2)
            logger.info(f"Navigating to spike {self.current_spike_index} at time {spike_time:.3f}s")
            self._manual_load = True
            self._load_window(start_time, self.window_duration)
            self._reset_manual_timer.start(100)

    def _on_prev_time(self):
        """Move window back by 2 seconds."""
        logger.debug("Prev time button clicked")
        if self.current_cluster_id is None:
            return
            
        new_start = max(0, self.buffer_start_time - self.window_duration)
        logger.info(f"Navigating to time window: {new_start:.1f}s - {new_start + self.window_duration:.1f}s")
        self._manual_load = True
        self._load_window(new_start, self.window_duration)
        self._reset_manual_timer.start(100)

    def _on_next_time(self):
        """Move window forward by 2 seconds."""
        logger.debug("Next time button clicked")
        if self.current_cluster_id is None:
            return
            
        dm = self.main_window.data_manager
        max_time = dm.n_samples / dm.sampling_rate
        
        new_start = min(
            self.buffer_start_time + self.window_duration,
            max_time - self.window_duration
        )
        logger.info(f"Navigating to time window: {new_start:.1f}s - {new_start + self.window_duration:.1f}s")
        self._manual_load = True
        self._load_window(new_start, self.window_duration)
        self._reset_manual_timer.start(100)

    def _load_window(self, start_time, duration):
        """Helper to load a specific time window."""
        if self.current_cluster_id is None:
            logger.warning("Cannot load window: no cluster selected")
            return
            
        dm = self.main_window.data_manager
        end_time = min(
            start_time + duration,
            dm.n_samples / dm.sampling_rate
        )
        
        self.buffer_start_time = start_time
        self.buffer_end_time = end_time
        
        self._update_time_label()
        self._start_worker(self.current_cluster_id, start_time, end_time)
        
        # Update view range
        self.plot.setXRange(start_time, end_time, padding=0)

    def _update_time_label(self):
        """Update the time range label."""
        self.time_label.setText(
            f"Time: {self.buffer_start_time:.1f}s - {self.buffer_end_time:.1f}s"
        )

    def _start_worker(self, cluster_id, start_time, end_time):
        """Start background worker to load data."""
        # Stop any existing worker
        if self.worker_thread and self.worker_thread.isRunning():
            logger.debug("Stopping existing worker thread")
            self.worker_thread.quit()
            self.worker_thread.wait()
            
        # Get channel info
        dm = self.main_window.data_manager
        features = dm.get_lightweight_features(cluster_id)
        if not features:
            logger.warning(f"No features for cluster {cluster_id}")
            self.status_message.emit("No features for cluster.")
            return
            
        p2p = features['median_ei'].max(axis=1) - features['median_ei'].min(axis=1)
        dom_chan = np.argmax(p2p)
        nearest_channels = dm.get_nearest_channels(dom_chan, n_channels=3)
        
        # Import worker - FIXED IMPORT PATH
        try:
            from ..workers.workers import RawTraceWorker
            logger.info("Successfully imported RawTraceWorker")
        except ImportError as e:
            logger.error(f"FAILED TO IMPORT RawTraceWorker: {e}")
            self.status_message.emit(f"Import error: {e}")
            return
        
        logger.info(f"Starting worker for cluster {cluster_id}, time {start_time:.3f}-{end_time:.3f}s")
        self.worker = RawTraceWorker(
            dm, cluster_id, nearest_channels, start_time, end_time
        )
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker.data_loaded.connect(self._on_data_loaded)
        self.worker.error.connect(self._on_data_error)
        self.worker_thread.started.connect(self.worker.run)
        self.worker_thread.start()
        
        self.status_message.emit(
            f"Loading raw trace for cluster {cluster_id}..."
        )

    def _on_data_loaded(
            self,
            cluster_id,
            raw_trace_data,
            start_time,
            end_time
        ):
        """Plot loaded data."""
        logger.info(f"Worker finished loading data for cluster {cluster_id}")
        
        try:
            dm = self.main_window.data_manager
            features = dm.get_lightweight_features(cluster_id)
            if not features:
                logger.warning(f"Could not retrieve features for cluster {cluster_id}")
                self.status_message.emit(
                    f"Could not retrieve features for cluster {cluster_id}"
                )
                return

            median_ei = features['median_ei']
            if median_ei.size == 0 or len(median_ei.shape) < 2:
                logger.warning(f"Invalid data for cluster {cluster_id}")
                self.status_message.emit(
                    f"Invalid data for cluster {cluster_id}"
                )
                return

            p2p = median_ei.max(axis=1) - median_ei.min(axis=1)
            dom_chan = np.argmax(p2p)
            nearest_channels = dm.get_nearest_channels(dom_chan, n_channels=3)

            # Create time axis
            if raw_trace_data.size == 0:
                logger.warning("Raw trace data is empty!")
                self.status_message.emit("No data returned for time window")
                return
                
            time_axis = np.linspace(start_time, end_time, raw_trace_data.shape[1])
            
            # Calculate vertical offset for stacked traces
            vertical_offset = max(
                np.max(np.abs(raw_trace_data)),
                1
            ) * 2.0

            # Update title
            self.plot.setTitle(
                f"Raw Traces for Cluster {cluster_id} - Channel {dom_chan}"
            )

            # Update trace plots
            num_traces = len(raw_trace_data)
            logger.info(f"Plotting {num_traces} traces with {len(time_axis)} samples each")
            
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

            # Plot spike markers (faint dotted lines)
            all_spikes = dm.get_cluster_spikes_in_window(
                cluster_id, start_time, end_time
            )
            
            logger.info(f"Found {len(all_spikes)} spikes in window {start_time:.3f}-{end_time:.3f}s")
            
            if len(all_spikes) > 0:
                window_spikes_sec = all_spikes / dm.sampling_rate
                
                # Get current Y range
                view_box = self.plot.getViewBox()
                y_range = view_box.viewRange()[1]
                y_min, y_max = y_range
                
                # Create line segments for each spike
                line_points = np.empty((len(window_spikes_sec) * 3, 2))
                line_points.fill(np.nan)
                
                for i, spike_time in enumerate(window_spikes_sec):
                    idx = i * 3
                    line_points[idx] = (spike_time, y_min)
                    line_points[idx + 1] = (spike_time, y_max)
                    # NaN at idx + 2 to break the line between spikes
                
                self.spike_marker_plot.setData(
                    line_points[:, 0],
                    line_points[:, 1]
                )
                self.spike_marker_plot.show()
            else:
                self.spike_marker_plot.hide()

            # Update spike count in window
            self._update_spike_label()
            
            logger.info("Plotting completed successfully")

        except Exception as e:
            logger.exception(f"Error plotting raw trace: {e}")
            # Clear plots on error
            for plot_item in self.trace_plots:
                plot_item.clear()
            self.spike_marker_plot.clear()
            self.status_message.emit(f"Error plotting raw trace: {str(e)}")
        finally:
            self.status_message.emit(
                f"Raw trace loaded for cluster {cluster_id}"
            )
            if self.worker_thread:
                self.worker_thread.quit()
                self.worker_thread.wait()
                self.worker_thread = None
                self.worker = None
            logger.debug("Worker thread cleaned up")

    def _on_data_error(self, msg):
        logger.error(f"Worker error: {msg}")
        self.error_message.emit(msg)
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
            self.worker = None

    def _reset_manual_flag(self):
        self._manual_load = False