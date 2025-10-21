from __future__ import annotations
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QSplitter
from qtpy.QtCore import Qt, QTimer
import numpy as np
from gui.widgets import MplCanvas
from qtpy.QtWidgets import QSizePolicy, QComboBox, QScrollArea
import pyqtgraph as pg
from scipy.interpolate import griddata
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gui.main_window import MainWindow


class EIPanel(QWidget):
    """
    Panel for spatial/EI analysis, including controls and matplotlib canvas.
    """
    def __init__(self, main_window: MainWindow, parent=None):
        super().__init__(parent)
        self.main_window = main_window  # Needed for data access and callbacks

         # --- Splitter for spatial (left) and temporal (right) EI ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        # --- Spatial EI Canvas ---
        self.spatial_canvas = MplCanvas(self, width=10, height=8, dpi=120)
        left_layout.addWidget(self.spatial_canvas)
        self.spatial_canvas.fig.canvas.mpl_connect('motion_notify_event', self.on_canvas_hover)
        splitter.addWidget(left_widget)

        # Overlay controls
        overlay_control_layout = QHBoxLayout()
        self.overlay_left_btn = QPushButton("◀")
        self.overlay_right_btn = QPushButton("▶")
        self.overlay_dropdown = QComboBox()
        overlay_control_layout.addWidget(QLabel("Overlay:"))
        overlay_control_layout.addWidget(self.overlay_left_btn)
        overlay_control_layout.addWidget(self.overlay_dropdown)
        overlay_control_layout.addWidget(self.overlay_right_btn)
        left_layout.addLayout(overlay_control_layout)

        self.overlay_index = 0
        self.overlay_dropdown.currentIndexChanged.connect(self._on_overlay_dropdown_changed)
        self.overlay_left_btn.clicked.connect(self._on_overlay_left)
        self.overlay_right_btn.clicked.connect(self._on_overlay_right)

        # Key toggles for overlay navigation
        
        # --- Temporal EI Canvas (right) ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        # self.temporal_canvas = MplCanvas(self, width=7, height=6, dpi=120)
        self.temporal_widget = pg.GraphicsLayoutWidget()
        self.temporal_plot = self.temporal_widget.addPlot()
        right_layout.addWidget(self.temporal_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 400])
        self.spatial_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.temporal_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
 
        QTimer.singleShot(0, self.spatial_canvas.draw)
        # QTimer.singleShot(0, self.temporal_plot.draw)

        # --- Main Layout ---
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(splitter)

        self.current_ei_data = None
        self.current_cluster_ids = None
        self.n_frames = 0

    def on_canvas_hover(self, event):
        # Handles hover events on the summary plot for tooltips.
        if event.inaxes is None or self.main_window.data_manager is None or self.current_ei_data is None:
            return
        if event.inaxes in self.spatial_canvas.fig.axes:
            positions = self.main_window.data_manager.channel_positions
            mouse_pos = np.array([[event.xdata, event.ydata]])
            distances = np.linalg.norm(positions - mouse_pos, axis=1)
            if distances.min() < 20:
                closest_idx = distances.argmin()
                self.main_window.status_bar.showMessage(f"Channel ID {closest_idx}")
            else:
                self.main_window.status_bar.clearMessage()
    
    def update_ei(self, cluster_ids):
        """
        Main entry point: update the EI panel for one or more clusters.
        """
        cluster_ids = np.array(cluster_ids, dtype=int)
        if cluster_ids.ndim == 0:
            cluster_ids = np.array([cluster_ids], dtype=int)
        vision_cluster_ids = cluster_ids + 1

        # Check for Vision EI
        has_vision_ei = self.main_window.data_manager.vision_eis and any(
            cid in self.main_window.data_manager.vision_eis for cid in vision_cluster_ids
        )

        if has_vision_ei:
            self._load_and_draw_vision_ei(cluster_ids)
        else:
            self._load_and_draw_ks_ei(cluster_ids)

    def clear(self):
        self.spatial_canvas.fig.clear()
        self.spatial_canvas.draw()

    # --- Internal: Vision EI ---

    def _load_and_draw_vision_ei(self, cluster_ids):
        vision_cluster_ids = cluster_ids + 1
        ei_data_list = []
        for cid in vision_cluster_ids:
            if cid in self.main_window.data_manager.vision_eis:
                ei_data_list.append(self.main_window.data_manager.vision_eis[cid].ei)
        if not ei_data_list:
            self.clear()
            return

        self.current_ei_data = ei_data_list
        self.current_cluster_ids = cluster_ids
        self.n_frames = ei_data_list[0].shape[1]

        # Make EI maps
        ei_map_list = []
        for ei_data in ei_data_list:
            ei_map = self._compute_ei_map(
                ei_data,
                self.main_window.data_manager.channel_positions
            )
            ei_map_list.append(ei_map)

        # Get top electrode for first cluster
        top_channels = self._get_top_electrodes(
            ei_data_list[0], 
            n_interval=2, n_markers=3, b_sort=True
        ) 

        self.current_ei_map_list = ei_map_list
        self.current_cluster_ids = cluster_ids
        self.current_channels = top_channels
        self.overlay_index = 0
            
        # Draw spatial and temporal EI
        # self._draw_vision_ei_spatial(ei_map_list, cluster_ids, top_channels)
        self._draw_vision_ei_spatial_overlay_only(ei_map_list, cluster_ids, top_channels)
        self._draw_vision_ei_temporal(ei_data_list, cluster_ids, top_channels)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self._on_overlay_left()
        elif event.key() == Qt.Key_Right:
            self._on_overlay_right()
        else:
            super().keyPressEvent(event)
    
    def _draw_vision_ei_spatial_overlay_only(self, ei_map_list, cluster_ids, channels=None):
            """
            Draw only the overlay axis, even for multiple clusters.
            """
            n_clusters = len(ei_map_list)
            self.spatial_canvas.fig.clear()

            # Always show overlay controls if multiple clusters
            if n_clusters > 1:
                self.overlay_left_btn.show()
                self.overlay_right_btn.show()
                self.overlay_dropdown.show()
            else:
                self.overlay_left_btn.hide()
                self.overlay_right_btn.hide()
                self.overlay_dropdown.hide()

            # Draw only one axis: the overlay
            ax = self.spatial_canvas.fig.add_subplot(111)
            overlay_idx = getattr(self, "overlay_index", 0)
            overlay_idx = np.clip(overlay_idx, 0, n_clusters - 1)
            # ax.set_title(f"Overlay: Cluster {cluster_ids[overlay_idx]}")

            channel_positions = self.main_window.data_manager.channel_positions
            xrange = (np.min(channel_positions[:, 0]), np.max(channel_positions[:, 0]))
            yrange = (np.min(channel_positions[:, 1]), np.max(channel_positions[:, 1]))

            im = ax.imshow(
                ei_map_list[overlay_idx], cmap='hot', aspect='auto', origin='lower',
                extent = (xrange[0], xrange[1], yrange[0], yrange[1])
            )
            # cbar
            # self.spatial_canvas.fig.colorbar(im, ax=ax, label='log10(abs(EI amplitude))')
            ax.axis('off')

            if channels is not None:
                for j, ch in enumerate(channels):
                    x, y = channel_positions[ch]
                    ax.plot(x, y, 'go', markersize=3, markerfacecolor='none', markeredgewidth=2)
                    ax.text(x, y, str(j), color='cyan', fontsize=6, ha='center', va='center')

            self.spatial_canvas.fig.suptitle(f"Spatial EI {cluster_ids[overlay_idx]}", color='white', fontsize=16)
            self.spatial_canvas.fig.tight_layout()
            self.spatial_canvas.draw()

            # Update dropdown
            self.overlay_dropdown.blockSignals(True)
            self.overlay_dropdown.clear()
            for cid in cluster_ids:
                self.overlay_dropdown.addItem(str(cid))
            self.overlay_dropdown.setCurrentIndex(overlay_idx)
            self.overlay_dropdown.blockSignals(False)
    
    def _draw_vision_ei_temporal(self, ei_data_list, cluster_ids, channels):
        """
        Plot temporal EI traces for the given clusters using pyqtgraph.
        """
        self.temporal_widget.clear()  # Clear previous plots

        for i, ch in enumerate(channels):
            plot_item = pg.PlotItem()
            self.temporal_widget.addItem(plot_item, i, 0)  # Add to grid layout

            for j, ei_data in enumerate(ei_data_list):
                time = np.arange(ei_data.shape[1]) / self.main_window.data_manager.sampling_rate * 1000  # ms
                plot_item.plot(time, ei_data[ch, :], pen=pg.mkPen(color=pg.intColor(j, hues=len(cluster_ids)), width=2), name=f'Cluster {cluster_ids[j]}')

            plot_item.setLabel('left', f'{i}: {ch}')
            plot_item.setLabel('bottom', 'Time (ms)')
            plot_item.addLegend()

        self.temporal_plot.setTitle("Temporal EI")
        self.temporal_plot.setLabel('left', 'Amplitude (µV)')
        self.temporal_plot.setLabel('bottom', 'Time (ms)')
        # self.temporal_plot.setAspectLocked(True)
        # self.temporal_plot.showGrid(x=True, y=True)
    
    # def _draw_vision_ei_temporal(self, ei_data_list, cluster_ids, channels):
    #     """
    #     Example: Plot temporal EI traces for the given clusters.
    #     """
    #     self.temporal_canvas.fig.clear()
    #     n_channels = len(channels)
    #     n_rows = n_channels
    #     n_cols = 1
    #     axes = self.temporal_canvas.fig.subplots(nrows=n_rows, ncols=n_cols, sharex=True, sharey=True)
    #     axes = axes.flatten() if n_channels > 1 else [axes]

    #     for i, ch in enumerate(channels):
    #         ax = axes[i]
    #         for j, ei_data in enumerate(ei_data_list):
    #             time = np.arange(ei_data.shape[1]) / self.main_window.data_manager.sampling_rate * 1000  # ms
    #             ax.plot(time, ei_data[ch, :], alpha=0.7, label=f'Cluster {cluster_ids[j]}')
    #         ax.set_title(f"{i} Chan {ch}")
    #         ax.set_xlabel("Time (ms)")
    #         # ax.set_ylabel("Amplitude (µV)")
    #         # ax.legend()
    #         ax.grid(True)
    #         ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        
    #     # Turn off any unused axes
    #     for k in range(i+1, len(axes)):
    #         axes[k].axis('off')

    #     # Legend over top of all subplots shared across top row
    #     handles, labels = axes[0].get_legend_handles_labels()
    #     if handles:
    #         self.temporal_canvas.fig.legend(handles, labels, loc='upper center', ncol=len(cluster_ids), bbox_to_anchor=(0.5, 1.02))
        
    #     self.temporal_canvas.fig.suptitle("Temporal EI", color='white', fontsize=16)
    #     self.temporal_canvas.fig.tight_layout()
    #     self.temporal_canvas.draw()
    
    # def _draw_vision_ei_spatial(self, ei_map_list, cluster_ids, channels=None):
    #     n_clusters = len(ei_map_list)
    #     self.spatial_canvas.fig.clear()

    #     # Show/hide overlay controls based on number of clusters
    #     if n_clusters > 1:
    #         self.overlay_left_btn.show()
    #         self.overlay_right_btn.show()
    #         self.overlay_dropdown.show()
    #         n_cols = min(n_clusters, self.n_max_cols)
    #         n_rows = (n_clusters + n_cols - 2) // n_cols + 1  # +1 for overlay axis
    #         axes = self.spatial_canvas.fig.subplots(nrows=n_rows, ncols=n_cols)
    #         axes = axes.flatten()

    #         # --- Overlay axis ---
    #         overlay_ax = axes[0]
    #         overlay_idx = getattr(self, "overlay_index", 0)
    #         overlay_idx = np.clip(overlay_idx, 0, n_clusters - 1)
    #         overlay_ax.set_title(f"Overlay: Cluster {cluster_ids[overlay_idx]}")
    #         im = overlay_ax.imshow(ei_map_list[overlay_idx], cmap='hot', aspect='equal', origin='lower')
    #         if channels is not None:
    #             for j, ch in enumerate(channels):
    #                 y, x = np.unravel_index(ch, ei_map_list[overlay_idx].shape)
    #                 overlay_ax.plot(x, y, 'go', markersize=3, markerfacecolor='none', markeredgewidth=2)
    #                 overlay_ax.text(x, y, str(j), color='cyan', fontsize=6, ha='center', va='center')

    #         # --- Rest of the grid ---
    #         for i, ei_map in enumerate(ei_map_list):
    #             ax = axes[i+1]
    #             ax.set_title(f"Cluster {cluster_ids[i]} EI")
    #             im = ax.imshow(ei_map, cmap='hot', aspect='equal', origin='lower')
    #             if channels is not None:
    #                 for j, ch in enumerate(channels):
    #                     y, x = np.unravel_index(ch, ei_map.shape)
    #                     ax.plot(x, y, 'go', markersize=3, markerfacecolor='none', markeredgewidth=2)
    #                     ax.text(x, y, str(j), color='cyan', fontsize=6, ha='center', va='center')
    #         # Turn off any unused axes
    #         for k in range(i+2, len(axes)):
    #             axes[k].axis('off')

    #         # Update dropdown
    #         self.overlay_dropdown.blockSignals(True)
    #         self.overlay_dropdown.clear()
    #         for cid in cluster_ids:
    #             self.overlay_dropdown.addItem(str(cid))
    #         self.overlay_dropdown.setCurrentIndex(overlay_idx)
    #         self.overlay_dropdown.blockSignals(False)

    #     else:
    #         # Only one cluster: hide overlay controls, show only one axis
    #         self.overlay_left_btn.hide()
    #         self.overlay_right_btn.hide()
    #         self.overlay_dropdown.hide()
    #         ax = self.spatial_canvas.fig.add_subplot(111)
    #         ax.set_title(f"Cluster {cluster_ids[0]} EI")
    #         im = ax.imshow(ei_map_list[0], cmap='hot', aspect='equal', origin='lower')
    #         if channels is not None:
    #             for j, ch in enumerate(channels):
    #                 y, x = np.unravel_index(ch, ei_map_list[0].shape)
    #                 ax.plot(x, y, 'go', markersize=3, markerfacecolor='none', markeredgewidth=2)
    #                 ax.text(x, y, str(j), color='cyan', fontsize=6, ha='center', va='center')

    #     self.spatial_canvas.fig.suptitle("Spatial EI", color='white', fontsize=16)
    #     self.spatial_canvas.fig.tight_layout()
    #     self.spatial_canvas.draw()

    def _on_overlay_dropdown_changed(self, idx):
        self.overlay_index = idx
        # Redraw overlay axis only
        # self._draw_vision_ei_spatial(self.current_ei_map_list, self.current_cluster_ids, self.current_channels)
        self._draw_vision_ei_spatial_overlay_only(self.current_ei_map_list, self.current_cluster_ids, self.current_channels)

    def _on_overlay_left(self):
        if self.overlay_index > 0:
            self.overlay_index -= 1
            self.overlay_dropdown.setCurrentIndex(self.overlay_index)

    def _on_overlay_right(self):
        if self.overlay_index < self.overlay_dropdown.count() - 1:
            self.overlay_index += 1
            self.overlay_dropdown.setCurrentIndex(self.overlay_index)
    
    def _get_top_electrodes(self, ei, n_interval=2, n_markers=5, b_sort=True):
        ## Label top n_markers pixels spaced by n_interval in the heatmap
        
        # Compute simple EI map in channel space
        ei_map = np.max(np.abs(ei), axis=1)
        ei_map = np.log10(ei_map + 1e-6)
        # Sorted index of channels
        ei_sidx = np.argsort(ei_map.flatten())[::-1]
        top_idx = ei_sidx[::n_interval][:n_markers]

        # Sort top_idx by argmin of EI time series
        if b_sort:
            amin_ei_ts = np.zeros(n_markers)
            for i in range(n_markers):
                ei_ts = ei[top_idx[i], :]
                amin_ei_ts[i] = np.argmin(ei_ts)
            top_idx = top_idx[np.argsort(amin_ei_ts)]

        return top_idx

    # --- Internal: Kilosort EI ---
    def _load_and_draw_ks_ei(self, cluster_ids):
        lightweight_features = self.main_window.data_manager.get_lightweight_features(cluster_ids)
        heavyweight_features = self.main_window.data_manager.get_heavyweight_features(cluster_ids)
        if lightweight_features is None or heavyweight_features is None:
            self.clear()
            self.spatial_canvas.fig.text(0.5, 0.5, "Error generating features.", ha='center', va='center', color='red')
            self.spatial_canvas.draw()
            return

        self.spatial_canvas.fig.clear()
        from analysis import analysis_core
        analysis_core.plot_rich_ei(
            self.spatial_canvas.fig, lightweight_features['median_ei'],
            self.main_window.data_manager.channel_positions,
            heavyweight_features, self.main_window.data_manager.sampling_rate, pre_samples=20
        )
        self.spatial_canvas.fig.suptitle(f"Cluster {cluster_ids} Spatial Analysis", color='white', fontsize=16)
        self.spatial_canvas.draw()
    
    def _reshape_ei(
        self, ei: np.ndarray,
        sorted_electrodes: np.ndarray, n_rows: int=16) -> np.ndarray:

        if ei.shape[0] != 512:
            print(f'Warning: Expected EI shape (512, 201), got {ei.shape}')
        n_electrodes = ei.shape[0]
        n_frames = ei.shape[1]
        n_cols = n_electrodes // n_rows
        if n_cols * n_rows != n_electrodes:
            raise ValueError(f"Number of electrodes {n_electrodes} is not compatible with {n_rows} rows and {n_cols} columns.")
        sorted_ei = ei[sorted_electrodes]
        reshaped_ei = sorted_ei.reshape(n_rows, n_cols, n_frames)
        return reshaped_ei
    
    def _compute_ei_map(
        self, ei: np.ndarray,
        channel_positions: np.ndarray) -> np.ndarray:

        if ei.shape[0] != 512:
            print(f'Warning: Expected EI shape (512, n_timepoints), got {ei.shape}')

        xrange = (np.min(channel_positions[:, 0]), np.max(channel_positions[:, 0]))
        yrange = (np.min(channel_positions[:, 1]), np.max(channel_positions[:, 1]))

        y_dim = 30 # Fixed y dimension for scaling
        x_dim = int((xrange[1] - xrange[0])/(yrange[1] - yrange[0]) * y_dim) # x dim is proportional to y dim

        x_e = np.linspace(xrange[0], xrange[1], x_dim)
        y_e = np.linspace(yrange[0], yrange[1], y_dim)

        grid_x, grid_y = np.meshgrid(x_e, y_e)
        grid_x = grid_x.T
        grid_y = grid_y.T

        # ei_energy = np.log10(np.mean(np.power(ei, 2), axis=1) + .000000001)
        ei_energy = np.log10(np.max(np.abs(ei), axis=1) + 1e-9)
        ei_energy_grid = griddata(
            channel_positions, ei_energy, 
            (grid_x, grid_y), method='linear', 
            fill_value=np.median(ei_energy))

        return ei_energy_grid.T