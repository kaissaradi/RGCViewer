import logging
import pyqtgraph as pg
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSplitter, QSlider, QLabel,
    QTextEdit
)
from qtpy.QtCore import Qt, QTimer
import numpy as np
from ...analysis import analysis_core
from ..widgets.widgets import MplCanvas
from ..theme import DARK_COLORS

logger = logging.getLogger(__name__)


class STAPanel(QWidget):
    def restyle_plots(self, colors):
        """Updates plot styling based on the provided color scheme."""
        self.rf_canvas.setBackground(colors['bg_panel'])
        self.rf_ellipse_item.setPen(pg.mkPen(colors['text_primary'], width=2, style=Qt.DashLine))
        self.timecourse_canvas.restyle(colors)
        self.temporal_filter_canvas.restyle(colors)

        # Update text edit style
        self.sta_metrics_text.setStyleSheet(f"background-color: {colors['bg_surface']}; color: {colors['text_primary']}; border: 1px solid {colors['border_subtle']};")

        # Force a redraw of current cluster if it exists
        cluster_id = self.main_window._get_selected_cluster_id()
        if cluster_id is not None:
             self.update_view(cluster_id)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        # --- STA Animation State ---
        self.current_frame_index = 0
        self.total_sta_frames = 0
        self.current_sta_data = None
        self.current_sta_cluster_id = None
        self.current_stafit = None  # Added to store STAFit data
        self.sta_animation_timer = None

        # --- UI Setup ---
        self._setup_ui()

        # Initialize button text based on current view
        if hasattr(main_window, 'current_sta_view'):
            if main_window.current_sta_view == "animation":
                self.sta_animation_button.setText("Pause Animation")
            else:
                self.sta_animation_button.setText("Play Animation")

    def _setup_ui(self):
        """Initializes and lays out all the UI widgets for STA panel."""
        colors = self.main_window.get_current_colors()
        layout = QVBoxLayout(self)

        # Control buttons layout
        sta_control_layout = QHBoxLayout()
        self.sta_population_rfs_button = QPushButton("Population RFs")
        self.sta_animation_button = QPushButton("Play Animation")
        self.sta_animation_stop_button = QPushButton("Stop Animation")

        self.sta_population_rfs_button.clicked.connect(self._toggle_population_rfs_view)
        self.sta_animation_button.clicked.connect(self.toggle_animation)
        self.sta_animation_stop_button.clicked.connect(self.stop_animation)
        sta_control_layout.addWidget(self.sta_population_rfs_button)
        sta_control_layout.addWidget(self.sta_animation_button)
        sta_control_layout.addWidget(self.sta_animation_stop_button)

        sta_control_layout.addStretch()  # Push buttons to left

        # --- Add Frame Slider and Label for STA Animation ---
        self.sta_frame_controls_layout = QHBoxLayout()
        self.sta_frame_controls_layout.setSpacing(5)
        self.sta_frame_controls_layout.setContentsMargins(0, 0, 0, 0)

        self.sta_frame_prev_button = QPushButton("Previous Frame")
        self.sta_frame_slider = QSlider(Qt.Horizontal)
        self.sta_frame_slider.setFixedWidth(200)
        self.sta_frame_slider.setMaximumHeight(30)
        self.sta_frame_next_button = QPushButton("Next Frame")
        self.sta_frame_label = QLabel("Frame: 0/0")

        self.sta_frame_prev_button.clicked.connect(self.prev_sta_frame)
        self.sta_frame_next_button.clicked.connect(self.next_sta_frame)
        self.sta_frame_slider.valueChanged.connect(
            self.update_sta_frame_manual)

        self.sta_frame_controls_layout.addWidget(self.sta_frame_prev_button)
        self.sta_frame_controls_layout.addWidget(self.sta_frame_slider)
        self.sta_frame_controls_layout.addWidget(self.sta_frame_next_button)
        self.sta_frame_controls_layout.addWidget(self.sta_frame_label)
        self.sta_frame_controls_layout.addStretch()

        # --- Create 4 Quadrants for STA Analysis ---
        # Replaced MplCanvas with pyqtgraph components for high-fps animation
        self.rf_canvas = pg.GraphicsLayoutWidget(self)
        self.rf_view = self.rf_canvas.addViewBox()
        self.rf_view.setAspectLocked(True)
        self.rf_view.invertY(True) # Optional: match typical image coordinate orientation
        self._pg_image_item = pg.ImageItem()
        self.rf_view.addItem(self._pg_image_item)
        self.rf_ellipse_item = pg.PlotCurveItem(pen=pg.mkPen(colors['text_primary'], width=2, style=Qt.DashLine))
        self.rf_view.addItem(self.rf_ellipse_item)
        self.rf_canvas.setToolTip("RF view and animation")

        self.timecourse_canvas = MplCanvas(self, width=5, height=4, dpi=120)
        self.timecourse_canvas.fig.text(
            0.5,
            0.5,
            "No STA data selected",
            ha='center',
            va='center',
            color=colors['text_secondary'])
        self.timecourse_canvas.draw()

        self.sta_metrics_text = QTextEdit()
        self.sta_metrics_text.setReadOnly(True)
        # Use dark theme colors by default - will be updated by restyle_plots()
        self.sta_metrics_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {DARK_COLORS['bg_surface']};
                color: {DARK_COLORS['text_primary']};
                font-family: Consolas, "Courier New", monospace;
                font-size: 11pt;
                border: 1px solid {DARK_COLORS['border_subtle']};
                padding: 10px;
            }}
        """)
        self.sta_metrics_text.setPlaceholderText(
            "Select a cell to view STA metrics...")

        self.temporal_filter_canvas = MplCanvas(
            self, width=5, height=4, dpi=120)
        self.temporal_filter_canvas.fig.text(
            0.5,
            0.5,
            "Temporal Analysis",
            ha='center',
            va='center',
            color=colors['text_secondary'])
        self.temporal_filter_canvas.draw()

        # --- Layout Assembly ---
        self.top_splitter = QSplitter(Qt.Horizontal)
        self.top_splitter.addWidget(self.rf_canvas)
        self.top_splitter.addWidget(self.timecourse_canvas)
        self.top_splitter.setSizes([400, 400])

        self.bottom_splitter = QSplitter(Qt.Horizontal)
        self.bottom_splitter.addWidget(self.sta_metrics_text)
        self.bottom_splitter.addWidget(self.temporal_filter_canvas)
        self.bottom_splitter.setSizes([300, 500])

        self.sta_splitter = QSplitter(Qt.Vertical)
        self.sta_splitter.addWidget(self.top_splitter)
        self.sta_splitter.addWidget(self.bottom_splitter)
        self.sta_splitter.setSizes([400, 300])

        layout.addLayout(sta_control_layout, 0)
        layout.addLayout(self.sta_frame_controls_layout, 0)
        layout.addWidget(self.sta_splitter, 1)

    def _update_pg_image(self):
        """Helper to extract, normalize, and push the current frame to pyqtgraph."""
        if self.current_sta_data is None:
            self._pg_image_item.clear()
            return
            
        red = self.current_sta_data.red[:, :, self.current_frame_index]
        green = self.current_sta_data.green[:, :, self.current_frame_index]
        blue = self.current_sta_data.blue[:, :, self.current_frame_index]
        frame = np.stack([red, green, blue], axis=-1)  # (height, width, 3)

        mn, mx = frame.min(), frame.max()
        if mx != mn:
            frame = (frame - mn) / (mx - mn)

        self._pg_image_item.setImage(frame.transpose(1, 0, 2))

    def _toggle_population_rfs_view(self):
        """Toggle between population RFs and single-cell STA view."""
        if hasattr(self.main_window, 'current_sta_view'):
            new_view = "rf" if self.main_window.current_sta_view == "population_rfs" else "population_rfs"

            # Update the main window's current view
            self.main_window.current_sta_view = new_view

            # Update button text based on current view
            if new_view == "rf":
                self.sta_animation_button.setText("Play Animation")
            elif new_view == "animation":
                self.sta_animation_button.setText("Pause Animation")
            elif new_view == "population_rfs":
                self.sta_animation_button.setText("Play Animation")

            cluster_id = self.main_window._get_selected_cluster_id()
            if cluster_id is not None:
                # Call update_view to refresh the display with the new view
                self.update_view(cluster_id)

    def update_view(self, cluster_id):
        """Update the STA view for the given cluster ID."""
        if (hasattr(self.main_window, 'data_manager') and
            self.main_window.data_manager and
            self.main_window.data_manager.vision_stas):

            # Draw the single-cell plots for the STA quad-view.
            self.plot_sta(cluster_id)
            self.plot_sta_timecourse(cluster_id)

            # Handle specific view-type overrides for the main RF canvas
            if hasattr(self.main_window, 'current_sta_view'):
                if self.main_window.current_sta_view == "population_rfs":
                    # MplCanvas removed for RF view, so dropping the population mode override
                    pass
                elif self.main_window.current_sta_view == "animation":
                    # Animation should only affect the RF plot
                    self.plot_sta_animation(cluster_id)
        else:
            # No Vision STA data available
            colors = self.main_window.get_current_colors()
            self._pg_image_item.clear()
            self.timecourse_canvas.fig.clear()
            self.timecourse_canvas.fig.text(
                0.5,
                0.5,
                "No Vision STA data available",
                ha='center',
                va='center',
                color=colors['text_secondary'])
            self.timecourse_canvas.draw()
            self.sta_frame_slider.setEnabled(False)

            # Clear other panels
            self.sta_metrics_text.clear()
            self.temporal_filter_canvas.fig.clear()
            self.temporal_filter_canvas.draw()

    def update_sta_frame_manual(self, frame_index):
        """Updates the STA visualization to a specific frame manually."""
        if self.current_sta_data is not None:
            self.stop_animation()
            self.current_frame_index = frame_index
            self.sta_frame_label.setText(
                f"Frame: {frame_index+1}/{self.total_sta_frames}")
            self._update_pg_image()

    def _advance_frame_internal(self):
        """Internal method for the timer to call without stopping itself."""
        if self.current_sta_data is not None:
            self.current_frame_index = (
                self.current_frame_index + 1) % self.total_sta_frames

            # Block signals so we don't trigger update_sta_frame_manual
            self.sta_frame_slider.blockSignals(True)
            self.sta_frame_slider.setValue(self.current_frame_index)
            self.sta_frame_slider.blockSignals(False)

            self.sta_frame_label.setText(
                f"Frame: {self.current_frame_index+1}/{self.total_sta_frames}")

            self._update_pg_image()

    def prev_sta_frame(self):
        """Go to the previous frame in the STA animation."""
        if self.current_sta_data is not None:
            self.stop_animation()
            self.current_frame_index = (
                self.current_frame_index - 1) % self.total_sta_frames
            self.sta_frame_slider.setValue(self.current_frame_index)
            self.sta_frame_label.setText(
                f"Frame: {self.current_frame_index+1}/{self.total_sta_frames}")
                
            self._update_pg_image()

    def next_sta_frame(self):
        """Go to the next frame in the STA animation."""
        if self.current_sta_data is not None:
            self.stop_animation()
            self.current_frame_index = (
                self.current_frame_index + 1) % self.total_sta_frames
            self.sta_frame_slider.setValue(self.current_frame_index)
            self.sta_frame_label.setText(
                f"Frame: {self.current_frame_index+1}/{self.total_sta_frames}")
                
            self._update_pg_image()

    def toggle_animation(self):
        """Toggle the animation between play and pause."""
        if not self.main_window.data_manager or not self.main_window.data_manager.vision_stas:
            # No data available
            return

        cluster_id = self.main_window._get_selected_cluster_id()
        if cluster_id is None:
            return

        # Update the animation button text based on current state
        if self.sta_animation_timer and self.sta_animation_timer.isActive():
            # Currently playing, so stop it
            self.stop_animation()
            self.sta_animation_button.setText("Play Animation")
        else:
            # Currently paused or stopped, so start it
            self.plot_sta_animation(cluster_id)
            self.sta_animation_button.setText("Pause Animation")

    def on_rf_canvas_clicked(self):
        """Handle clicks on the RF canvas in STA tab - toggle between RF and animation."""
        if not self.main_window.data_manager or not self.main_window.data_manager.vision_stas:
            return

        cluster_id = self.main_window._get_selected_cluster_id()
        if cluster_id is None:
            return

        # Toggle between RF and animation views
        if self.main_window.current_sta_view == "rf":
            # Start animation
            self.main_window.current_sta_view = "animation"
            self.main_window.select_sta_view("animation", force_animation=True)
            self.main_window.status_bar.showMessage(
                "Started animation. Click again to stop.", 2000)
        elif self.main_window.current_sta_view == "animation":
            # Stop animation and go back to RF
            self.stop_animation()
            self.main_window.current_sta_view = "rf"
            self.main_window.select_sta_view("rf")
            self.main_window.status_bar.showMessage("Stopped animation.", 2000)
        elif self.main_window.current_sta_view == "population_rfs":
            # From population view, go to RF view
            self.main_window.current_sta_view = "rf"
            self.main_window.select_sta_view("rf")
            self.main_window.status_bar.showMessage(
                "Switched to single-cell RF view.", 2000)

    def stop_animation(self):
        """Stop the animation completely."""
        if self.sta_animation_timer and self.sta_animation_timer.isActive():
            self.sta_animation_timer.stop()
        self.sta_animation_button.setText(
            "Play Animation")  # Reset button to Play

    def plot_sta(self, cluster_id):
        """
        Fetches STAFit data and passes it to the plotting function.
        Now also triggers metrics and temporal filter updates.
        """
        vision_cluster_id = cluster_id + 1
        has_vision_sta = self.main_window.data_manager.vision_stas and vision_cluster_id in self.main_window.data_manager.vision_stas
        
        logger.debug(f"plot_sta: cluster_id={cluster_id}, vision_cluster_id={vision_cluster_id}, has_vision_sta={has_vision_sta}")
        logger.debug(f"vision_stas keys: {list(self.main_window.data_manager.vision_stas.keys())[:5] if self.main_window.data_manager.vision_stas else 'None'}")

        if has_vision_sta:
            self.stop_animation()

            sta_data = self.main_window.data_manager.vision_stas[vision_cluster_id]
            try:
                stafit = self.main_window.data_manager.vision_params.get_stafit_for_cell(
                    vision_cluster_id)
                logger.debug(f"Got stafit for {vision_cluster_id}: {stafit is not None}")
            except Exception as e:
                logger.error(f"Failed to get stafit for {vision_cluster_id}: {e}")
                stafit = None
            self.current_sta_data = sta_data
            self.current_stafit = stafit
            self.current_sta_cluster_id = vision_cluster_id

            n_frames = sta_data.red.shape[2]
            self.total_sta_frames = n_frames
            self.sta_frame_slider.setMaximum(n_frames - 1)

            all_channels = np.stack(
                [sta_data.red, sta_data.green, sta_data.blue], axis=0)
            frame_energies = np.max(np.abs(all_channels), axis=(0, 1, 2))
            peak_frame_index = np.argmax(frame_energies)
            self.current_frame_index = peak_frame_index

            self.sta_frame_slider.setValue(peak_frame_index)
            self.sta_frame_label.setText(
                f"Frame: {peak_frame_index + 1}/{n_frames}")
            self.sta_frame_slider.setEnabled(True)

            self._update_pg_image()

            if stafit:
                cx, cy = stafit.center_x, stafit.center_y
                sx, sy = getattr(stafit, 'std_x', 1), getattr(stafit, 'std_y', 1)
                
                # Get the height of the STA to flip the Y-axis
                height = self.current_sta_data.red.shape[0]
                
                # 1. Convert angle and negate it because of the Y-axis flip
                angle = getattr(stafit, 'angle', getattr(stafit, 'orientation', 0))
                angle_rad = -np.radians(angle) 
                
                # 2. Flip Vision's bottom-left Y coordinate to top-left, 
                # and adjust by 0.5 for Pyqtgraph's pixel anchors
                cx_plot = cx + 0.5
                cy_plot = (height - cy) - 0.5 
                
                # Generate ellipse points
                t = np.linspace(0, 2*np.pi, 100)
                x_el = cx_plot + sx * np.cos(t) * np.cos(angle_rad) - sy * np.sin(t) * np.sin(angle_rad)
                y_el = cy_plot + sx * np.cos(t) * np.sin(angle_rad) + sy * np.sin(t) * np.cos(angle_rad)
                
                self.rf_ellipse_item.setData(x_el, y_el)
                self.rf_ellipse_item.setVisible(True)
            else:
                self.rf_ellipse_item.setVisible(False)
            # --- Update New Panels ---
            self.plot_sta_metrics(cluster_id)
            self.plot_temporal_filter(cluster_id)

        else:
            # No Vision STA data available
            self._pg_image_item.clear()
            self.sta_frame_slider.setEnabled(False)

            # Clear other panels
            self.sta_metrics_text.clear()
            self.temporal_filter_canvas.fig.clear()
            self.temporal_filter_canvas.draw()

    def plot_sta_timecourse(self, cluster_id):
        # Draws the STA timecourse plot for a specific cell.
        vision_cluster_id = cluster_id + 1
        has_vision_sta = self.main_window.data_manager.vision_stas and vision_cluster_id in self.main_window.data_manager.vision_stas
        if has_vision_sta:
            sta_data = self.main_window.data_manager.vision_stas[vision_cluster_id]
            stafit = self.main_window.data_manager.vision_params.get_stafit_for_cell(
                vision_cluster_id)
            self.timecourse_canvas.fig.clear()
            self.plot_sta_timecourse_internal(
                self.timecourse_canvas.fig,
                sta_data,
                stafit,
                self.main_window.data_manager.vision_params,
                vision_cluster_id
            )
            self.timecourse_canvas.draw()
        else:
            colors = self.main_window.get_current_colors()
            self.timecourse_canvas.fig.clear()
            self.timecourse_canvas.fig.text(
                0.5,
                0.5,
                "No Vision STA data available",
                ha='center',
                va='center',
                color=colors['text_secondary'])
            self.timecourse_canvas.draw()

    def plot_sta_animation(self, cluster_id):
        # Draws the STA animation plot for a specific cell.
        vision_cluster_id = cluster_id + 1
        has_vision_sta = self.main_window.data_manager.vision_stas and vision_cluster_id in self.main_window.data_manager.vision_stas
        if has_vision_sta:
            self.current_sta_data = self.main_window.data_manager.vision_stas[vision_cluster_id]
            self.current_sta_cluster_id = vision_cluster_id
            self.current_frame_index = 0
            n_frames = self.current_sta_data.red.shape[2]
            self.total_sta_frames = n_frames

            self.sta_frame_slider.setMinimum(0)
            self.sta_frame_slider.setMaximum(n_frames - 1)
            self.sta_frame_slider.setValue(0)
            self.sta_frame_label.setText(f"Frame: 1/{n_frames}")
            self.sta_frame_slider.setEnabled(True)

            # Ensure the animation timer is properly created
            if self.sta_animation_timer is None:
                self.sta_animation_timer = QTimer()
                self.sta_animation_timer.timeout.connect(
                    self._advance_frame_internal)

            # Stop any currently running animation first to prevent conflicts
            if self.sta_animation_timer and self.sta_animation_timer.isActive():
                self.sta_animation_timer.stop()
            else:
                # Start the animation only if it's not already running
                self.sta_animation_timer.start(100)

            self._update_pg_image()
        else:
            self._pg_image_item.clear()
            self.sta_frame_slider.setEnabled(False)

    def plot_sta_metrics(self, cluster_id):
        """
        Calculates and displays STA metrics in the text box with improved organization.
        """
        vision_cluster_id = cluster_id + 1
        has_vision_sta = self.main_window.data_manager.vision_stas and vision_cluster_id in self.main_window.data_manager.vision_stas

        if has_vision_sta:
            sta_data = self.main_window.data_manager.vision_stas[vision_cluster_id]
            stafit = self.main_window.data_manager.vision_params.get_stafit_for_cell(
                vision_cluster_id)

            metrics = analysis_core.compute_sta_metrics(
                sta_data, stafit, self.main_window.data_manager.vision_params, vision_cluster_id)

            # Get current theme colors
            colors = self.main_window.get_current_colors()

            # Format metrics as HTML table with sections - using theme colors
            html = f"""
            <style>
                table {{ width: 100%; border-collapse: collapse; color: {colors['text_primary']}; font-family: sans-serif; }}
                th {{ text-align: left; color: {colors['text_secondary']}; border-bottom: 1px solid {colors['border_default']}; padding: 4px; }}
                td {{ padding: 6px; border-bottom: 1px solid {colors['border_subtle']}; }}
                .section {{ font-weight: bold; color: {colors['accent']}; margin-top: 12px; display: block; font-size: 1.1em; }}
                .subsection {{ font-weight: bold; color: {colors['accent_positive']}; margin-top: 8px; display: block; }}
                .metric-name {{ font-weight: 600; color: {colors['text_secondary']}; }}
                .metric-value {{ color: {colors['text_primary']}; }}
                .highlight {{ background-color: {colors['selection_bg']}; }}
            </style>
            """

            html += "<table>"

            # Temporal Properties
            html += "<tr><th colspan='2' class='section'>Temporal Properties</th></tr>"
            temporal_keys = [
                "Dominant Channel",
                "Polarity",
                "Time to Peak (ms)",
                "Response Duration (ms)",
                "Zero Crossing (ms)",
                "FWHM (Duration)",
                "Biphasic Index",
                "SNR (std ratio)"]
            for k in temporal_keys:
                if k in metrics:
                    html += f"<tr><td class='metric-name'>{k}</td><td class='metric-value'>{metrics[k]}</td></tr>"

            # Response Strength
            html += "<tr><th colspan='2' class='subsection'>Response Strength</th></tr>"
            strength_keys = ["Response Integral", "Total Energy"]
            for k in strength_keys:
                if k in metrics:
                    html += f"<tr><td class='metric-name'>{k}</td><td class='metric-value'>{metrics[k]}</td></tr>"

            # Color Properties
            if "Color Opponency" in metrics:
                html += "<tr><th colspan='2' class='subsection'>Color Properties</th></tr>"
                html += f"<tr><td class='metric-name'>Color Opponency</td><td class='metric-value'>{metrics['Color Opponency']}</td></tr>"

            # Spatial Properties
            html += "<tr><th colspan='2' class='section'>Spatial Properties</th></tr>"
            spatial_keys = [
                "RF Center X",
                "RF Center Y",
                "RF Sigma X",
                "RF Sigma Y",
                "Orientation",
                "RF Area (sq stix)",
                "RF Ellipticity (σy/σx)",
                "RF Elongation"]
            for k in spatial_keys:
                if k in metrics:
                    html += f"<tr><td class='metric-name'>{k}</td><td class='metric-value'>{metrics[k]}</td></tr>"

            # Spatial Asymmetry
            html += "<tr><th colspan='2' class='subsection'>Spatial Asymmetry</th></tr>"
            asymmetry_keys = [
                "Spatial Peak",
                "Spatial Trough",
                "Peak/Trough Ratio",
                "Spatial Skewness"]
            for k in asymmetry_keys:
                if k in metrics:
                    html += f"<tr><td class='metric-name'>{k}</td><td class='metric-value'>{metrics[k]}</td></tr>"

            html += "</table>"

            self.sta_metrics_text.setHtml(html)
        else:
            colors = self.main_window.get_current_colors()
            self.sta_metrics_text.setHtml(
                f"<div style='color:{colors['text_secondary']}; text-align:center; padding:20px;'>No STA Data Available</div>")

    def plot_temporal_filter(self, cluster_id):
        """
        Draws the detailed temporal filter analysis plot.
        """
        vision_cluster_id = cluster_id + 1
        has_vision_sta = self.main_window.data_manager.vision_stas and vision_cluster_id in self.main_window.data_manager.vision_stas

        if has_vision_sta:
            sta_data = self.main_window.data_manager.vision_stas[vision_cluster_id]
            stafit = self.main_window.data_manager.vision_params.get_stafit_for_cell(
                vision_cluster_id)

            self.plot_temporal_filter_properties(
                self.temporal_filter_canvas.fig,
                sta_data,
                stafit,
                self.main_window.data_manager.vision_params,
                vision_cluster_id)
            self.temporal_filter_canvas.draw()
        else:
            colors = self.main_window.get_current_colors()
            self.temporal_filter_canvas.fig.clear()
            self.temporal_filter_canvas.fig.text(
                0.5, 0.5, "No Data", ha='center', va='center', color=colors['text_secondary'])
            self.temporal_filter_canvas.draw()

    def plot_sta_timecourse_internal(
            self,
            fig,
            sta_data,
            stafit,
            vision_params,
            cell_id,
            _sampling_rate=20):
        """
        Visualizes the timecourse of the STA response for a specific cell.
        The x-axis shows time from -500ms to 0ms before the spike.

        Args:
            fig (matplotlib.figure.Figure): The figure object to draw on.
            sta_data (STAContainer): Named tuple containing the raw STA movie.
            stafit (STAFit): Named tuple containing the Gaussian fit parameters.
            vision_params (VisionCellDataTable): Object containing pre-calculated timecourse data.
            cell_id (int): The ID of the cell to plot (1-indexed for vision data).
            sampling_rate (float): Sampling rate in Hz for the STA data (stixels per second).
        """
        fig.clear()

        timecourse_matrix = None
        try:
            red_tc = vision_params.get_data_for_cell(cell_id, 'RedTimeCourse')
            green_tc = vision_params.get_data_for_cell(cell_id, 'GreenTimeCourse')
            blue_tc = vision_params.get_data_for_cell(cell_id, 'BlueTimeCourse')
            if red_tc is not None and green_tc is not None and blue_tc is not None:
                timecourse_matrix = np.stack([red_tc, green_tc, blue_tc], axis=1)
        except Exception:
            timecourse_matrix = None

        if timecourse_matrix is not None and hasattr(timecourse_matrix, 'shape'):
            if len(timecourse_matrix.shape) == 2:
                n_timepoints, n_channels = timecourse_matrix.shape

                # --- DYNAMIC X-AXIS CALCULATION ---
                # Get the refresh time (in ms) from the STA data container.
                # This makes the time axis accurate to the original experiment.
                refresh_ms = getattr(sta_data, 'refresh_time', 1000.0 / 60.0)
                total_duration_ms = (n_timepoints - 1) * refresh_ms

                # Create the accurate time axis based on the data's properties
                time_axis = np.linspace(-total_duration_ms, 0, n_timepoints)

                ax = fig.add_subplot(111)

                n_channels_to_plot = min(n_channels, 3)
                channel_names = ['Red', 'Green', 'Blue'][:n_channels_to_plot]
                channel_colors = ['red', 'green', 'blue'][:n_channels_to_plot]
                
                # Get theme colors from main window
                theme_colors = getattr(self.main_window, 'get_current_colors', lambda: {})()
                if not theme_colors:
                    theme_colors = DARK_COLORS

                for i in range(n_channels_to_plot):
                    ax.plot(
                        time_axis,
                        timecourse_matrix[:,
                                          i],
                        color=channel_colors[i],
                        linewidth=1.5,
                        label=channel_names[i])

                ax.set_title("STA Timecourse (Pre-calculated)", color=theme_colors['text_primary'])
                ax.set_xlabel("Time (ms)", color=theme_colors['text_secondary'])
                ax.set_ylabel("Response", color=theme_colors['text_secondary'])
                ax.grid(True, alpha=0.3)
                ax.legend(facecolor=theme_colors['bg_panel'], labelcolor=theme_colors['text_primary'])

                ax.set_facecolor(theme_colors['bg_panel'])
                ax.tick_params(colors=theme_colors['text_secondary'])
                for spine in ax.spines.values():
                    spine.set_edgecolor(theme_colors['border_default'])

                ax.axvline(x=0, color=theme_colors['text_primary'], linestyle='--', alpha=0.7)
                ax.axhline(y=0, color=theme_colors['text_secondary'], linestyle=':',
                           alpha=0.5)  # Add dotted line at y=0

                # --- ACCURATE Y-AXIS SCALING ---
                # This logic fits the axis tightly to the min/max of the saved
                # data.
                if timecourse_matrix.size > 0:
                    y_min = timecourse_matrix.min()
                    y_max = timecourse_matrix.max()
                    y_range = y_max - y_min if y_max > y_min else 1.0
                    y_margin = y_range * 0.10  # Add a 10% margin for readability
                    ax.set_ylim(y_min - y_margin, y_max + y_margin)

                return

        # Fallback logic remains unchanged
        logger.warning(
            "No precomputed timecourse for cell %s; recomputing",
            cell_id)

        red_channel = sta_data.red
        green_channel = sta_data.green
        blue_channel = sta_data.blue

        n_timepoints = red_channel.shape[2]

        center_x = int(stafit.center_x)
        center_y = int(stafit.center_y)
        std_x = int(max(1, stafit.std_x))
        std_y = int(max(1, stafit.std_y))

        x_min = max(0, center_x - std_x)
        x_max = min(red_channel.shape[1], center_x + std_x + 1)
        y_min = max(0, center_y - std_y)
        y_max = min(red_channel.shape[0], center_y + std_y + 1)

        red_timecourse = np.mean(
            red_channel[y_min:y_max, x_min:x_max], axis=(0, 1))
        green_timecourse = np.mean(
            green_channel[y_min:y_max, x_min:x_max], axis=(0, 1))
        blue_timecourse = np.mean(
            blue_channel[y_min:y_max, x_min:x_max], axis=(0, 1))

        # Fallback uses a hardcoded duration as it has no refresh_time metadata
        total_time_ms = 1500
        time_axis = np.linspace(-total_time_ms, 0, n_timepoints)

        ax = fig.add_subplot(111)

        ax.plot(time_axis, red_timecourse, color='red', linewidth=1.5, label='Red')
        ax.plot(
            time_axis,
            green_timecourse,
            color='green',
            linewidth=1.5,
            label='Green')
        ax.plot(
            time_axis,
            blue_timecourse,
            color='blue',
            linewidth=1.5,
            label='Blue')

        # Get theme colors from main window
        theme_colors = getattr(self.main_window, 'get_current_colors', lambda: {})()
        if not theme_colors:
            theme_colors = DARK_COLORS

        ax.set_title("STA Timecourse (Recalculated)", color=theme_colors['text_primary'])
        ax.set_xlabel("Time (ms)", color=theme_colors['text_secondary'])
        ax.set_ylabel("Response", color=theme_colors['text_secondary'])
        ax.grid(True, alpha=0.3)
        ax.legend(facecolor=theme_colors['bg_panel'], labelcolor=theme_colors['text_primary'])

        ax.set_facecolor(theme_colors['bg_panel'])
        ax.tick_params(colors=theme_colors['text_secondary'])
        for spine in ax.spines.values():
            spine.set_edgecolor(theme_colors['border_default'])

        ax.axvline(x=0, color=theme_colors['text_primary'], linestyle='--', alpha=0.7)
        ax.axhline(y=0, color=theme_colors['text_secondary'], linestyle=':',
                   alpha=0.5)  # Add dotted line at y=0

    def plot_temporal_filter_properties(self, fig, sta_data, stafit, vision_params, cell_id):
        """
        Plots the temporal filter with annotations for metrics.
        """
        fig.clear()
        ax = fig.add_subplot(111)
        theme_colors = getattr(self.main_window, 'get_current_colors', lambda: {})()
        if not theme_colors:
            theme_colors = DARK_COLORS

        time_axis, tc_matrix, _ = analysis_core.get_sta_timecourse_data(
            sta_data, stafit, vision_params, cell_id)

        if tc_matrix is None:
            ax.text(
                0.5,
                0.5,
                "No temporal data",
                ha='center',
                va='center',
                color=theme_colors['text_secondary'])
            return

        # Dominant channel
        energies = np.sum(tc_matrix**2, axis=0)
        dom_idx = np.argmax(energies)
        dom_trace = tc_matrix[:, dom_idx]
        dom_color = ['red', 'green', 'blue'][dom_idx]

        # Normalize
        abs_max = np.max(np.abs(dom_trace))
        if abs_max == 0:
            abs_max = 1
        norm_trace = dom_trace / abs_max

        ax.plot(
            time_axis,
            norm_trace,
            color=dom_color,
            linewidth=2,
            label='Filter')
        ax.fill_between(time_axis, norm_trace, 0, color=dom_color, alpha=0.1)

        # Find peak
        peak_idx = np.argmax(np.abs(norm_trace))
        peak_time = time_axis[peak_idx]
        peak_val = norm_trace[peak_idx]

        # Annotate Peak
        ax.scatter([peak_time], [peak_val], color=theme_colors['plot_peak'], s=50, zorder=5)
        ax.annotate(f"Peak: {peak_time:.1f}ms",
                    xy=(peak_time, peak_val),
                    xytext=(0, 10 if peak_val > 0 else -15),
                    textcoords='offset points',
                    ha='center', color=theme_colors['plot_peak'], fontsize=9)

        # FWHM
        try:
            is_off = peak_val < 0
            trace_for_width = -norm_trace if is_off else norm_trace
            from scipy.signal import peak_widths
            widths, width_heights, left_ips, right_ips = peak_widths(
                trace_for_width, [peak_idx], rel_height=0.5
            )
            if len(widths) > 0:
                width = widths[0]
                # Interpolate time points
                sample_interval = abs(time_axis[1] - time_axis[0])
                start_time = time_axis[0] + left_ips[0] * sample_interval
                end_time = time_axis[0] + right_ips[0] * sample_interval

                h = width_heights[0]
                if is_off:
                    h = -h

                ax.hlines(
                    h,
                    start_time,
                    end_time,
                    colors=theme_colors['plot_peak'],
                    linestyles='-',
                    linewidth=2)
                ax.annotate(f"FWHM: {width * sample_interval:.1f}ms",
                            xy=((start_time + end_time) / 2, h),
                            xytext=(0, 5), textcoords='offset points',
                            ha='center', color=theme_colors['plot_peak'], fontsize=8)
        except Exception:
            pass

        ax.set_title("Temporal Dynamics Analysis", color=theme_colors['text_primary'])
        ax.set_xlabel("Time (ms)", color=theme_colors['text_secondary'])
        ax.set_facecolor(theme_colors['bg_panel'])
        ax.tick_params(colors=theme_colors['text_secondary'])
        for spine in ax.spines.values():
            spine.set_edgecolor(theme_colors['border_default'])
        ax.grid(True, alpha=0.2)
        ax.axhline(0, color=theme_colors['text_secondary'], linestyle=':', alpha=0.5)
