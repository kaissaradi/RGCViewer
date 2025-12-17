"""
3D visualization widgets for the RGC Viewer application.
"""
import numpy as np
import pyqtgraph.opengl as gl
from qtpy.QtWidgets import QWidget, QVBoxLayout
from analysis import analysis_core
from collections import OrderedDict


class EIMountainPlotWidget(QWidget):
    """
    3D visualization widget for displaying EI data as a mountain/surface plot.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        # Create layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create the GL View Widget for 3D plotting
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setBackgroundColor('k')  # Black background

        # Add to layout
        layout.addWidget(self.gl_view)

        # Initialize surfaces
        self.surface_items = []

    def clear_plot(self):
        """Clear all surfaces from the 3D plot."""
        for item in self.surface_items:
            self.gl_view.removeItem(item)
        self.surface_items = []

    def plot_ei_3d(self, ei_data, channel_positions):
        """
        Plot EI data in 3D surface view.

        Args:
            ei_data: EI data array (n_channels, time_points)
            channel_positions: Channel position array (n_channels, 2)
        """
        # Clear previous surfaces
        self.clear_plot()

        # Prepare data for 3D visualization
        grid_x, grid_y, grid_z = analysis_core.get_ei_surface_data(
            ei_data,
            channel_positions,
            grid_resolution=50j  # Lower resolution for faster rendering
        )

        if grid_x is None or grid_y is None or grid_z is None:
            print("Error: Could not compute surface data for 3D plot")
            return

        # Replace NaN values with zeros
        grid_z = np.nan_to_num(grid_z, nan=0.0)

        # Create a surface plot item with color gradient based on height
        surface = gl.GLSurfacePlotItem(
            x=grid_x[:, 0],  # x coordinates (first column)
            y=grid_y[0, :],  # y coordinates (first row)
            z=grid_z,        # z values (height)
            shader='shaded',
            smooth=True
        )

        # Add the surface to the view
        self.gl_view.addItem(surface)
        self.surface_items.append(surface)

        # Add axes
        axis = gl.GLAxisItem()
        self.gl_view.addItem(axis)

        # Add grid
        grid = gl.GLGridItem()
        grid.setSize(x=100, y=100, z=100)
        grid.setSpacing(10, 10, 10)
        self.gl_view.addItem(grid)

        # Set initial viewing angle
        self.gl_view.opts['distance'] = 150
        self.gl_view.setCameraPosition(distance=150, elevation=30, azimuth=45)