"""
Visualization widgets for the RGC Viewer application.
This module contains widgets for displaying EI data as contour plots (replacing 3D visualization).
"""
import numpy as np
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel
from qtpy.QtCore import Qt, QTimer
from scipy.interpolate import griddata
from gui.widgets import MplCanvas
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class EIMountainPlotWidget(QWidget):
    """
    Matplotlib 3D static mountain plot for EI max-projection.
    This provides a stable, non-OpenGL 3D surface that can be rotated
    with the mouse and works across platforms.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

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

    def plot_ei_3d(self, ei_data, channel_positions):
        """
        Plot the Max-Projection (min over time, inverted) as a 3D surface.
        """
        self.ei_data = ei_data
        self.channel_positions = channel_positions

        # Compute spatial footprint (deepest negative trough per channel)
        spatial_footprint = np.min(self.ei_data, axis=1)
        # Invert so mountains rise up
        z_values = spatial_footprint * -1.0

        # Interpolate to grid
        min_x, max_x = self.channel_positions[:, 0].min(), self.channel_positions[:, 0].max()
        min_y, max_y = self.channel_positions[:, 1].min(), self.channel_positions[:, 1].max()

        grid_x, grid_y = np.mgrid[min_x:max_x:self.grid_res, min_y:max_y:self.grid_res]

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
        ax.set_facecolor('#1f1f1f')
        self.canvas.fig.patch.set_facecolor('#1f1f1f')

        surf = ax.plot_surface(
            grid_x, grid_y, grid_z,
            cmap='plasma',
            linewidth=0,
            antialiased=False,
            rcount=grid_z.shape[0],
            ccount=grid_z.shape[1]
        )

        # Styling: hide axes for cleaner view
        ax.set_axis_off()
        ax.set_title('EI Max Projection (Voltage)', color='white')

        # Optional subtle floor grid
        # Draw and finish
        self.canvas.draw()

    def clear_plot(self):
        self.canvas.fig.clear()
        self.canvas.draw()
        self.ei_data = None
        self.channel_positions = None