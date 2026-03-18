from __future__ import annotations
import matplotlib.pyplot as plt
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSplitter
from qtpy.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
import numpy as np
from ..widgets.widgets import MplCanvas
from qtpy.QtWidgets import QSizePolicy, QComboBox, QStackedWidget
import pyqtgraph as pg
from scipy.interpolate import griddata
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..main_window import MainWindow
import logging
logger = logging.getLogger(__name__)


class EIMountainPlotWidget(QWidget):
    """
    Matplotlib 3D static mountain plot for EI max-projection.
    This provides a stable, non-OpenGL 3D surface that can be rotated
    with the mouse and works across platforms.
    """

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        # Main Layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Matplotlib canvas
        self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
        self.layout.addWidget(self.canvas)

        self.ei_data = None
        self.channel_positions = None
        # grid resolution for interpolation
        self.grid_res = 60j

    def restyle(self, colors):
        """Updates the plot styling based on the provided color scheme."""
        self.canvas.restyle(colors)
        if self.ei_data is not None:
            self.plot_ei_3d(self.ei_data, self.channel_positions)

    def plot_ei_3d(self, ei_data, channel_positions):
        """
        Plot the Max-Projection (min over time, inverted) as a 3D surface.
        """
        self.ei_data = ei_data
        self.channel_positions = channel_positions
        colors = self.main_window.get_current_colors()

        # Compute spatial footprint (deepest negative trough per channel)
        spatial_footprint = np.min(self.ei_data, axis=1)
        # Invert so mountains rise up
        z_values = spatial_footprint * -1.0

        # Interpolate to grid
        min_x, max_x = self.channel_positions[:, 0].min(
        ), self.channel_positions[:, 0].max()
        min_y, max_y = self.channel_positions[:, 1].min(
        ), self.channel_positions[:, 1].max()

        grid_x, grid_y = np.mgrid[min_x:max_x:self.grid_res,
                                  min_y:max_y:self.grid_res]

        grid_z = griddata(
            self.channel_positions,
            z_values,
            (grid_x, grid_y),
            method='linear',
            fill_value=0
        )
        grid_z = np.nan_to_num(grid_z, nan=0.0)

        # Render using Matplotlib 3D
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111, projection='3d')
        ax.set_facecolor(colors['bg_panel'])
        self.canvas.fig.patch.set_facecolor(colors['bg_panel'])

        _ = ax.plot_surface(
            grid_x, grid_y, grid_z,
            cmap='plasma',
            linewidth=0,
            antialiased=False,
            rcount=grid_z.shape[0],
            ccount=grid_z.shape[1]
        )

        # Styling: hide axes for cleaner view
        ax.set_axis_off()
        ax.set_title('EI Max Projection (Voltage)', color=colors['text_primary'])

        # Optional subtle floor grid
        # Draw and finish
        self.canvas.draw()

    def clear_plot(self):
        self.canvas.fig.clear()
        self.canvas.draw()
        self.ei_data = None
        self.channel_positions = None


def plot_latency_map(fig, ei_data, channel_positions, sampling_rate, colors=None):
    """
    Plots the propagation latency.
    Color = Time to Peak (Blue -> Red = Start -> End).
    Size = Amplitude of signal (bigger dots = larger amplitude).
    """
    if colors is None:
        # Fallback to dark defaults if no colors provided
        colors = {'bg_panel': '#111214', 'bg_surface': '#1E2025', 'text_primary': '#F0F0F2', 'text_secondary': '#9B9DA6'}

    fig.clear()
    ax = fig.add_subplot(111)

    # 1. Calculate Time to First Significant Deflection (Latency) for every channel
    baseline = ei_data[:, :10].mean(axis=1)  # Use first 10 samples as baseline
    baseline_corrected = ei_data - baseline[:, np.newaxis]

    # Calculate amplitude-based threshold (20% of the min amplitude)
    min_amplitudes = np.min(baseline_corrected, axis=1)
    threshold = min_amplitudes * 0.2  # 20% of minimum amplitude

    # Find the first sample where the signal crosses the threshold
    peak_indices = np.zeros(ei_data.shape[0], dtype=int)
    for i in range(ei_data.shape[0]):
        # Look for first crossing of the threshold
        crossing_samples = np.where(
            baseline_corrected[i, :] <= threshold[i])[0]
        if len(crossing_samples) > 0:
            peak_indices[i] = crossing_samples[0]  # First crossing
        else:
            # If no crossing found, use argmin as fallback
            peak_indices[i] = np.argmin(baseline_corrected[i, :])

    peak_times_ms = (peak_indices / sampling_rate) * 1000.0

    # Calculate amplitudes for sizing the dots
    amplitudes = np.abs(np.min(ei_data, axis=1))
    min_amp, max_amp = amplitudes.min(), amplitudes.max()
    if max_amp > min_amp:  # Avoid division by zero
        normalized_sizes = 10 + 140 * \
            (amplitudes - min_amp) / (max_amp - min_amp)
    else:
        # Default size if all amplitudes are the same
        normalized_sizes = np.full_like(amplitudes, 80.0)

    # 2. Plot all electrode positions with latency (color) and amplitude (size)
    sc = ax.scatter(
        channel_positions[:, 0],
        channel_positions[:, 1],
        c=peak_times_ms,
        s=normalized_sizes,
        cmap='turbo',
        edgecolor='white' if colors['bg_panel'] in ['#111214', '#18191C'] else 'black',
        linewidth=0.4,
        alpha=0.8  # Slight transparency to handle overlapping dots
    )

    ax.set_facecolor(colors['bg_panel'])
    fig.patch.set_facecolor(colors['bg_panel'])
    ax.set_title('Signal Propagation Latency', color=colors['text_primary'])
    ax.set_xlabel('X (µm)', color=colors['text_secondary'])
    ax.set_ylabel('Y (µm)', color=colors['text_secondary'])
    ax.tick_params(colors=colors['text_secondary'])
    ax.set_aspect('equal')

    # Colorbar for latency
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Time to Peak (ms)', color=colors['text_secondary'])
    cbar.ax.yaxis.set_tick_params(color=colors['text_secondary'])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=colors['text_secondary'])

    # Create a second legend-like axis for amplitude sizes
    axins = ax.inset_axes([0.02, 0.02, 0.4, 0.1])
    # Semi-transparent background
    bg_color = QColor(colors.get('bg_surface', '#1E2025'))
    axins.set_facecolor((bg_color.red()/255, bg_color.green()/255, bg_color.blue()/255, 0.7))

    # Show a few representative dot sizes
    example_sizes = [20, 80, 140]  # Representing small, medium, large
    example_amps = []
    for size in example_sizes:
        amp_val = ((size - 10) / 140.0) * (max_amp - min_amp) + min_amp
        example_amps.append(amp_val)

    axins.scatter([0.2, 0.5, 0.8], [0.5, 0.5, 0.5],
                  s=example_sizes, c=colors['text_primary'], edgecolor=colors['text_primary'], alpha=0.8)
    axins.text(0.2, 0.7, f'{example_amps[0]:.1f} µV', color=colors['text_primary'], fontsize=8, ha='center')
    axins.text(0.5, 0.7, f'{example_amps[1]:.1f} µV', color=colors['text_primary'], fontsize=8, ha='center')
    axins.text(0.8, 0.7, f'{example_amps[2]:.1f} µV', color=colors['text_primary'], fontsize=8, ha='center')
    axins.set_xlim(0, 1)
    axins.set_ylim(0, 1)
    axins.axis('off')

    fig.tight_layout()
    fig.canvas.draw()


def compute_ei_map(
        ei: np.ndarray,
        channel_positions: np.ndarray) -> np.ndarray:

    if ei.shape[0] != channel_positions.shape[0]:
        logger.error(
            "EI channel count mismatch: ei=%d, positions=%d",
            ei.shape[0],
            channel_positions.shape[0])
        return None

    if ei.shape[0] != 512:
        if ei.shape[0] not in [512, 519]:  # Common sizes in vision data
            logger.warning('Unexpected EI shape: %s', ei.shape)

    xrange = (np.min(channel_positions[:, 0]), np.max(channel_positions[:, 0]))
    yrange = (np.min(channel_positions[:, 1]), np.max(channel_positions[:, 1]))

    y_dim = 30  # Fixed y dimension for scaling
    x_dim = int((xrange[1] - xrange[0]) / (yrange[1] - yrange[0]) * y_dim)

    x_e = np.linspace(xrange[0], xrange[1], x_dim)
    y_e = np.linspace(yrange[0], yrange[1], y_dim)

    grid_x, grid_y = np.meshgrid(x_e, y_e)
    grid_x = grid_x.T
    grid_y = grid_y.T

    ei_energy = np.log10(np.max(np.abs(ei), axis=1) + 1e-9)
    ei_energy_grid = griddata(
        channel_positions, ei_energy,
        (grid_x, grid_y), method='linear',
        fill_value=np.median(ei_energy))

    return ei_energy_grid.T


class EIPanel(QWidget):
    """
    Panel for spatial/EI analysis, including controls and matplotlib canvas.
    """

    def __init__(self, main_window: MainWindow, parent=None):
        super().__init__(parent)
        self.main_window = main_window  # Needed for data access and callbacks

        # Track the current view to properly update when cluster changes
        self.current_view = "2D Heatmap"

        # --- Create stacked widget for switching between 2D and 3D views ---
        self.spatial_stack_widget = QStackedWidget()

        # --- 2D Matplotlib view ---
        self.spatial_2d_widget = QWidget()
        self.spatial_2d_layout = QVBoxLayout(self.spatial_2d_widget)
        self.spatial_2d_layout.setContentsMargins(0, 0, 0, 0)
        self.spatial_2d_layout.setSpacing(0)

        # --- 2D Spatial EI Canvas ---
        self.spatial_canvas = MplCanvas(self, width=10, height=8, dpi=120)
        self.spatial_2d_layout.addWidget(self.spatial_canvas)
        self.spatial_canvas.fig.canvas.mpl_connect(
            'motion_notify_event', self.on_canvas_hover)

        # Overlay controls for 2D view
        overlay_control_layout = QHBoxLayout()
        self.overlay_left_btn = QPushButton("◀")
        self.overlay_right_btn = QPushButton("▶")
        self.overlay_dropdown = QComboBox()
        overlay_control_layout.addWidget(QLabel("Overlay:"))
        overlay_control_layout.addWidget(self.overlay_left_btn)
        overlay_control_layout.addWidget(self.overlay_dropdown)
        overlay_control_layout.addWidget(self.overlay_right_btn)
        self.spatial_2d_layout.addLayout(overlay_control_layout)

        self.overlay_index = 0
        self.overlay_dropdown.currentIndexChanged.connect(
            self._on_overlay_dropdown_changed)
        self.overlay_left_btn.clicked.connect(self._on_overlay_left)
        self.overlay_right_btn.clicked.connect(self._on_overlay_right)

        # Add 2D view to the stacked widget
        self.spatial_stack_widget.addWidget(self.spatial_2d_widget)

        # --- 3D Mountain Plot View ---
        self.spatial_3d_widget = QWidget()
        self.spatial_3d_layout = QVBoxLayout(self.spatial_3d_widget)
        self.spatial_3d_layout.setContentsMargins(0, 0, 0, 0)
        self.spatial_3d_layout.setSpacing(0)

        # Create 3D mountain plot widget
        self.mountain_plot_widget = EIMountainPlotWidget(self.main_window)
        self.spatial_3d_layout.addWidget(self.mountain_plot_widget)

        # Add 3D view to the stacked widget
        self.spatial_stack_widget.addWidget(self.spatial_3d_widget)

        # --- Dropdown for switching between 2D and 3D views ---
        view_selection_layout = QHBoxLayout()
        view_selection_layout.addWidget(QLabel("View:"))
        self.view_dropdown = QComboBox()
        self.view_dropdown.addItems(
            ["2D Heatmap", "3D Mountain Plot", "Latency Map"])
        self.view_dropdown.currentTextChanged.connect(self._on_view_changed)
        view_selection_layout.addWidget(self.view_dropdown)
        view_selection_layout.addStretch()  # Add stretch to push dropdown to the left

        # --- Splitter for spatial (left) and temporal (right) EI ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        # Add view selection dropdown to the left widget
        left_layout.addLayout(view_selection_layout)

        # Add the stacked widget to the left widget
        left_layout.addWidget(self.spatial_stack_widget)
        splitter.addWidget(left_widget)

        # --- Temporal EI Canvas (right) ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        self.temporal_widget = pg.GraphicsLayoutWidget()
        self.temporal_plot = self.temporal_widget.addPlot()
        right_layout.addWidget(self.temporal_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 400])
        self.spatial_canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.temporal_plot.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)

        QTimer.singleShot(0, self.spatial_canvas.draw)

        # --- Main Layout ---
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(splitter)

        self.current_ei_data = None
        self.current_cluster_ids = None
        self.n_frames = 0

    def restyle_plots(self, colors):
        """Updates plot styling based on the provided color scheme."""
        self.spatial_canvas.restyle(colors)
        self.mountain_plot_widget.restyle(colors)
        self.temporal_widget.setBackground(colors['bg_panel'])
        
        # Style the temporal plot
        self.temporal_plot.getAxis('bottom').setPen(pg.mkPen(colors['border_default']))
        self.temporal_plot.getAxis('left').setPen(pg.mkPen(colors['border_default']))
        self.temporal_plot.getAxis('bottom').setTextPen(pg.mkPen(colors['text_secondary']))
        self.temporal_plot.getAxis('left').setTextPen(pg.mkPen(colors['text_secondary']))
        
        # Re-plot current data if available to update title colors etc.
        if self.current_ei_data is not None:
             self.update_ei(self.current_cluster_ids)

    def on_canvas_hover(self, event):
        if event.inaxes is None or self.main_window.data_manager is None or self.current_ei_data is None:
            return
        if event.inaxes in self.spatial_canvas.fig.axes:
            positions = self.main_window.data_manager.channel_positions
            mouse_pos = np.array([[event.xdata, event.ydata]])
            distances = np.linalg.norm(positions - mouse_pos, axis=1)
            if distances.min() < 20:
                closest_idx = distances.argmin()
                self.main_window.status_bar.showMessage(
                    f"Channel ID {closest_idx}")
            else:
                self.main_window.status_bar.clearMessage()

    def update_ei(self, cluster_ids):
        if not self.isVisible():
            return
        cluster_ids = np.array(cluster_ids, dtype=int)
        if cluster_ids.ndim == 0:
            cluster_ids = np.array([cluster_ids], dtype=int)

        primary_cluster_id = cluster_ids[0]
        vision_cluster_ids = cluster_ids + 1

        try:
            has_vision_ei = self.main_window.data_manager.vision_eis and any(
                cid in self.main_window.data_manager.vision_eis for cid in vision_cluster_ids)

            if has_vision_ei:
                self._load_and_draw_vision_ei(cluster_ids)
            else:
                lightweight = self.main_window.data_manager.get_lightweight_features(primary_cluster_id)
                heavyweight = self.main_window.data_manager.get_heavyweight_features(primary_cluster_id)

                if lightweight is None or heavyweight is None:
                    self._show_loading_state("Loading spatial features...")
                    if self.main_window.spatial_worker:
                        self.main_window.spatial_worker.add_to_queue(primary_cluster_id, high_priority=True)
                    return

                self._load_and_draw_ks_ei(cluster_ids, is_fallback=True)

            if self.current_view == "Latency Map":
                if self.current_ei_data is not None and len(self.current_ei_data) > 0:
                    ei_data = self.current_ei_data[0]
                    channel_positions = self.main_window.data_manager.channel_positions
                    if self.main_window.data_manager.vision_channel_positions is not None:
                        channel_positions = self.main_window.data_manager.vision_channel_positions
                    plot_latency_map(
                        self.spatial_canvas.fig,
                        ei_data,
                        channel_positions,
                        self.main_window.data_manager.sampling_rate,
                        colors=self.main_window.get_current_colors())

        except Exception as e:
            logger.exception(f"EI update failed for cluster {primary_cluster_id}")
            self._show_error_state(f"Error: {str(e)[:100]}")

    def _show_loading_state(self, message="Loading..."):
        self.clear()
        colors = self.main_window.get_current_colors()
        self.spatial_canvas.fig.text(
            0.5, 0.5, message,
            ha='center', va='center', color='cyan', fontsize=14
        )
        self.spatial_canvas.draw()

    def _show_error_state(self, message="Error"):
        self.clear()
        self.spatial_canvas.fig.text(
            0.5, 0.5, message,
            ha='center', va='center', color='red', fontsize=12
        )
        self.spatial_canvas.draw()

    def clear(self):
        self.spatial_canvas.fig.clear()
        self.spatial_canvas.draw()

    def _load_and_draw_vision_ei(self, cluster_ids):
        vision_cluster_ids = cluster_ids + 1
        valid_ei_data_list = []
        valid_original_ids = []

        for i, cid in enumerate(vision_cluster_ids):
            original_id = cluster_ids[i]
            if isinstance(cid, (int, np.integer)) and 0 < cid < 100000:
                if cid in self.main_window.data_manager.vision_eis:
                    ei_entry = self.main_window.data_manager.vision_eis[cid]
                    if hasattr(ei_entry, 'ei') and ei_entry.ei is not None and ei_entry.ei.ndim == 2:
                        valid_ei_data_list.append(ei_entry.ei)
                        valid_original_ids.append(original_id)

        if not valid_ei_data_list:
            self.clear()
            return

        ei_map_list = []
        final_valid_ids = []
        final_valid_ei_data = []

        for i, ei_data in enumerate(valid_ei_data_list):
            original_id = valid_original_ids[i]
            try:
                if self.main_window.data_manager.vision_channel_positions is not None:
                    channel_positions = self.main_window.data_manager.vision_channel_positions
                else:
                    channel_positions = self.main_window.data_manager.channel_positions

                ei_map = compute_ei_map(ei_data, channel_positions)
                if ei_map is not None:
                    ei_map_list.append(ei_map)
                    final_valid_ids.append(original_id)
                    final_valid_ei_data.append(ei_data)
            except Exception:
                logger.exception("Failed to compute EI map for cluster %s", original_id)

        if not ei_map_list:
            self.clear()
            return

        self.current_ei_map_list = ei_map_list
        self.current_cluster_ids = final_valid_ids
        self.current_ei_data = final_valid_ei_data
        self.n_frames = self.current_ei_data[0].shape[1] if self.current_ei_data else 0

        try:
            top_channels = self._get_top_electrodes(self.current_ei_data[0], n_interval=2, n_markers=3, b_sort=True)
        except Exception:
            top_channels = []

        self.current_channels = top_channels
        self.overlay_index = 0

        self._draw_vision_ei_spatial_overlay_only(self.current_ei_map_list, self.current_cluster_ids, top_channels)
        self._draw_vision_ei_temporal(self.current_ei_data, self.current_cluster_ids, top_channels)

        if self.current_ei_data and len(self.current_ei_data) > 0:
            ei_data = self.current_ei_data[0]
            channel_positions = self.main_window.data_manager.channel_positions
            if self.main_window.data_manager.vision_channel_positions is not None:
                channel_positions = self.main_window.data_manager.vision_channel_positions
            self.mountain_plot_widget.plot_ei_3d(ei_data, channel_positions)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self._on_overlay_left()
        elif event.key() == Qt.Key_Right:
            self._on_overlay_right()
        else:
            super().keyPressEvent(event)

    def _draw_vision_ei_spatial_overlay_only(self, ei_map_list, cluster_ids, channels=None):
        n_clusters = len(ei_map_list)
        self.spatial_canvas.fig.clear()
        colors = self.main_window.get_current_colors()

        if n_clusters > 1:
            self.overlay_left_btn.show()
            self.overlay_right_btn.show()
            self.overlay_dropdown.show()
        else:
            self.overlay_left_btn.hide()
            self.overlay_right_btn.hide()
            self.overlay_dropdown.hide()

        ax = self.spatial_canvas.fig.add_subplot(111)
        ax.set_facecolor(colors['bg_panel'])
        self.spatial_canvas.fig.patch.set_facecolor(colors['bg_panel'])
        
        overlay_idx = getattr(self, "overlay_index", 0)
        overlay_idx = np.clip(overlay_idx, 0, n_clusters - 1)

        channel_positions = self.main_window.data_manager.channel_positions
        xrange = (np.min(channel_positions[:, 0]), np.max(channel_positions[:, 0]))
        yrange = (np.min(channel_positions[:, 1]), np.max(channel_positions[:, 1]))

        ax.imshow(ei_map_list[overlay_idx], cmap='hot', aspect='auto', origin='lower',
                  extent=(xrange[0], xrange[1], yrange[0], yrange[1]))
        ax.axis('off')

        if channels is not None:
            for j, ch in enumerate(channels):
                x, y = channel_positions[ch]
                ax.plot(x, y, 'go', markersize=3, markerfacecolor='none', markeredgewidth=2)
                ax.text(x, y, str(j), color='cyan', fontsize=6, ha='center', va='center')

        self.spatial_canvas.fig.suptitle(f"Spatial EI {cluster_ids[overlay_idx]}", color=colors['text_primary'], fontsize=16)
        self.spatial_canvas.fig.tight_layout()
        self.spatial_canvas.draw()

        self.overlay_dropdown.blockSignals(True)
        self.overlay_dropdown.clear()
        for cid in cluster_ids:
            self.overlay_dropdown.addItem(str(cid))
        self.overlay_dropdown.setCurrentIndex(overlay_idx)
        self.overlay_dropdown.blockSignals(False)

    def _draw_vision_ei_temporal(self, ei_data_list, cluster_ids, channels):
        self.temporal_widget.clear()
        colors = self.main_window.get_current_colors()

        for i, ch in enumerate(channels):
            plot_item = pg.PlotItem()
            # Style plot item
            plot_item.getAxis('bottom').setPen(pg.mkPen(colors['border_default']))
            plot_item.getAxis('left').setPen(pg.mkPen(colors['border_default']))
            plot_item.getAxis('bottom').setTextPen(pg.mkPen(colors['text_secondary']))
            plot_item.getAxis('left').setTextPen(pg.mkPen(colors['text_secondary']))
            
            self.temporal_widget.addItem(plot_item, i, 0)

            for j, ei_data in enumerate(ei_data_list):
                time = np.arange(ei_data.shape[1]) / self.main_window.data_manager.sampling_rate * 1000  # ms
                plot_item.plot(time, ei_data[ch, :], pen=pg.mkPen(color=pg.intColor(
                    j, hues=len(cluster_ids)), width=2), name=f'Cluster {cluster_ids[j]}')

            plot_item.setLabel('left', f'{i}: {ch}')
            plot_item.setLabel('bottom', 'Time (ms)')
            plot_item.addLegend()

    def _on_overlay_dropdown_changed(self, idx):
        self.overlay_index = idx
        self._draw_vision_ei_spatial_overlay_only(self.current_ei_map_list, self.current_cluster_ids, self.current_channels)

    def _on_overlay_left(self):
        if self.overlay_index > 0:
            self.overlay_index -= 1
            self.overlay_dropdown.setCurrentIndex(self.overlay_index)

    def _on_overlay_right(self):
        if self.overlay_index < self.overlay_dropdown.count() - 1:
            self.overlay_index += 1
            self.overlay_dropdown.setCurrentIndex(self.overlay_index)

    def _on_view_changed(self, text):
        self.current_view = text
        if text == "2D Heatmap":
            self.spatial_stack_widget.setCurrentIndex(0)
        elif text == "3D Mountain Plot":
            self.spatial_stack_widget.setCurrentIndex(1)
            if self.current_ei_data is not None:
                ei_data = self.current_ei_data[0]
                channel_positions = self.main_window.data_manager.channel_positions
                if self.main_window.data_manager.vision_channel_positions is not None:
                    channel_positions = self.main_window.data_manager.vision_channel_positions
                self.mountain_plot_widget.plot_ei_3d(ei_data, channel_positions)
        elif text == "Latency Map":
            self.spatial_stack_widget.setCurrentIndex(0)
            if self.current_ei_data is not None:
                ei_data = self.current_ei_data[0]
                channel_positions = self.main_window.data_manager.channel_positions
                if self.main_window.data_manager.vision_channel_positions is not None:
                    channel_positions = self.main_window.data_manager.vision_channel_positions
                plot_latency_map(
                    self.spatial_canvas.fig,
                    ei_data,
                    channel_positions,
                    self.main_window.data_manager.sampling_rate,
                    colors=self.main_window.get_current_colors())

    def _get_top_electrodes(self, ei, n_interval=2, n_markers=5, b_sort=True):
        ei_map = np.max(np.abs(ei), axis=1)
        ei_map = np.log10(ei_map + 1e-6)
        ei_sidx = np.argsort(ei_map.flatten())[::-1]
        top_idx = ei_sidx[::n_interval][:n_markers]

        if b_sort:
            amin_ei_ts = np.zeros(n_markers)
            for i in range(n_markers):
                ei_ts = ei[top_idx[i], :]
                amin_ei_ts[i] = np.argmin(ei_ts)
            top_idx = top_idx[np.argsort(amin_ei_ts)]

        return top_idx

    def _load_and_draw_ks_ei(self, cluster_ids, is_fallback=False):
        if not hasattr(cluster_ids, '__len__') or len(cluster_ids) == 0:
            self.clear()
            self.spatial_canvas.fig.text(0.5, 0.5, "No cluster ID provided.", ha='center', va='center', color='orange')
            self.spatial_canvas.draw()
            return

        cluster_id = cluster_ids[0]
        lightweight_features = self.main_window.data_manager.get_lightweight_features(cluster_id)
        heavyweight_features = self.main_window.data_manager.get_heavyweight_features(cluster_id)
        
        if lightweight_features is None or heavyweight_features is None:
            self.clear()
            self.spatial_canvas.fig.text(0.5, 0.5, "Error generating features.", ha='center', va='center', color='red')
            self.spatial_canvas.draw()
            return

        self.spatial_canvas.fig.clear()
        colors = self.main_window.get_current_colors()
        self.spatial_canvas.fig.patch.set_facecolor(colors['bg_panel'])
        
        from .population_panel import plot_rich_ei
        plot_rich_ei(
            self.spatial_canvas.fig,
            lightweight_features['median_ei'],
            self.main_window.data_manager.channel_positions,
            heavyweight_features,
            self.main_window.data_manager.sampling_rate,
            _pre_samples=20)

        title = f"Cluster {cluster_id} Spatial Analysis"
        if is_fallback:
            title += " (Vision EI not found)"
        self.spatial_canvas.fig.suptitle(title, color=colors['text_primary'], fontsize=16)
        self.spatial_canvas.draw()

        if lightweight_features and 'median_ei' in lightweight_features:
            ei_data = lightweight_features['median_ei']
            channel_positions = self.main_window.data_manager.channel_positions
            self.mountain_plot_widget.plot_ei_3d(ei_data, channel_positions)
