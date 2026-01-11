import logging
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSplitter, QSlider, QLabel,
    QTextEdit
)
from qtpy.QtCore import Qt, QTimer
import numpy as np
from ...analysis import analysis_core
from ..widgets.widgets import MplCanvas

logger = logging.getLogger(__name__)


class STAPanel(QWidget):
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
        self.rf_canvas = MplCanvas(self, width=5, height=4, dpi=120)
        self.rf_canvas.fig.text(
            0.5,
            0.5,
            "No STA data selected",
            ha='center',
            va='center',
            color='gray')
        self.rf_canvas.draw()
        self.rf_canvas.clicked.connect(self.on_rf_canvas_clicked)
        self.rf_canvas.setToolTip(
            "Click to toggle between RF view and animation")

        self.timecourse_canvas = MplCanvas(self, width=5, height=4, dpi=120)
        self.timecourse_canvas.fig.text(
            0.5,
            0.5,
            "No STA data selected",
            ha='center',
            va='center',
            color='gray')
        self.timecourse_canvas.draw()

        self.sta_metrics_text = QTextEdit()
        self.sta_metrics_text.setReadOnly(True)
        self.sta_metrics_text.setStyleSheet("""
            QTextEdit {
                background-color: #1f1f1f;
                color: #e0e0e0;
                font-family: Consolas, "Courier New", monospace;
                font-size: 11pt;
                border: 1px solid #333;
                padding: 10px;
            }
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
            color='gray')
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
            # Use the STAPanel's own methods instead of the main window's
            self.plot_sta(cluster_id)
            self.plot_sta_timecourse(cluster_id)

            # Handle specific view-type overrides for the main RF canvas
            if hasattr(self.main_window, 'current_sta_view'):
                if self.main_window.current_sta_view == "population_rfs":
                    # This button press should always draw the population plot in the MAIN STA view (rf_canvas),
                    # overriding the single-cell RF plot drawn by draw_sta_plot above.
                    from ..plotting import plotting
                    plotting.draw_population_rfs_plot(
                        main_window=self.main_window, selected_cell_id=cluster_id, canvas=self.rf_canvas)
                elif self.main_window.current_sta_view == "animation":
                    # Animation should only affect the RF plot
                    self.plot_sta_animation(cluster_id)
        else:
            # No Vision STA data available
            self.rf_canvas.fig.clear()
            self.rf_canvas.fig.text(
                0.5,
                0.5,
                "No Vision STA data available",
                ha='center',
                va='center',
                color='gray')
            self.timecourse_canvas.fig.clear()
            self.timecourse_canvas.fig.text(
                0.5,
                0.5,
                "No Vision STA data available",
                ha='center',
                va='center',
                color='gray')
            self.rf_canvas.draw()
            self.timecourse_canvas.draw()
            self.sta_frame_slider.setEnabled(False)

            # Clear other panels
            self.sta_metrics_text.clear()
            self.temporal_filter_canvas.fig.clear()
            self.temporal_filter_canvas.draw()

    def update_sta_frame_manual(self, frame_index):
        """Updates the STA visualization to a specific frame manually."""
        if self.current_sta_data is not None:
            # Stop any running animation
            self.stop_animation()

            # Update the frame index
            self.current_frame_index = frame_index

            # Update the label
            self.sta_frame_label.setText(
                f"Frame: {frame_index+1}/{self.total_sta_frames}")
            # Update the STA canvas with the new frame - use RF canvas for
            # animation
            self.rf_canvas.fig.clear()
            self.animate_sta_movie(
                self.rf_canvas.fig,
                self.current_sta_data,
                frame_index=frame_index,
                sta_width=self.main_window.data_manager.vision_sta_width,
                sta_height=self.main_window.data_manager.vision_sta_height
            )
            cluster_id = self.current_sta_cluster_id - 1  # Convert back to 0-indexed
            self.rf_canvas.fig.suptitle(
                f"Cluster {cluster_id} - STA Frame {frame_index+1}/{self.total_sta_frames}",
                color='white',
                fontsize=16)  # this overlaps with self.sta_frame_label
            self.rf_canvas.draw()

    def _advance_frame_internal(self):
        """Internal method for the timer to call without stopping itself."""
        if self.current_sta_data is not None:
            # Increment frame and loop back to 0 if at the end
            self.current_frame_index = (
                self.current_frame_index + 1) % self.total_sta_frames

            # --- FIX: Block signals so we don't trigger update_sta_frame_manual ---
            self.sta_frame_slider.blockSignals(True)
            self.sta_frame_slider.setValue(self.current_frame_index)
            self.sta_frame_slider.blockSignals(False)
            # ---------------------------------------------------------------------

            self.sta_frame_label.setText(
                f"Frame: {self.current_frame_index+1}/{self.total_sta_frames}")

            # Redraw the RF canvas
            self.rf_canvas.fig.clear()
            self.animate_sta_movie(
                self.rf_canvas.fig,
                self.current_sta_data,
                stafit=self.current_stafit,
                frame_index=self.current_frame_index,
                sta_width=self.main_window.data_manager.vision_sta_width,
                sta_height=self.main_window.data_manager.vision_sta_height
            )
            self.rf_canvas.draw()

    def prev_sta_frame(self):
        """Go to the previous frame in the STA animation."""
        if self.current_sta_data is not None:
            # Stop the animation when manually navigating
            self.stop_animation()
            self.current_frame_index = (
                self.current_frame_index - 1) % self.total_sta_frames
            self.sta_frame_slider.setValue(self.current_frame_index)
            self.sta_frame_label.setText(
                f"Frame: {self.current_frame_index+1}/{self.total_sta_frames}")
            self.rf_canvas.fig.clear()
            self.animate_sta_movie(
                self.rf_canvas.fig,
                self.current_sta_data,
                stafit=self.current_stafit,  # <-- Pass the stored fit
                frame_index=self.current_frame_index,
                sta_width=self.main_window.data_manager.vision_sta_width,
                sta_height=self.main_window.data_manager.vision_sta_height
            )
            self.rf_canvas.draw()

    def next_sta_frame(self):
        """Go to the next frame in the STA animation."""
        if self.current_sta_data is not None:
            # Stop the animation when manually navigating
            self.stop_animation()
            self.current_frame_index = (
                self.current_frame_index + 1) % self.total_sta_frames
            self.sta_frame_slider.setValue(self.current_frame_index)
            self.sta_frame_label.setText(
                f"Frame: {self.current_frame_index+1}/{self.total_sta_frames}")
            self.rf_canvas.fig.clear()
            self.animate_sta_movie(
                self.rf_canvas.fig,
                self.current_sta_data,
                stafit=self.current_stafit,  # <-- Pass the stored fit
                frame_index=self.current_frame_index,
                sta_width=self.main_window.data_manager.vision_sta_width,
                sta_height=self.main_window.data_manager.vision_sta_height
            )
            self.rf_canvas.draw()

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

        if has_vision_sta:
            self.stop_animation()

            sta_data = self.main_window.data_manager.vision_stas[vision_cluster_id]
            # --- ADDED: Get STAFit data and store it for other functions to use ---
            stafit = self.main_window.data_manager.vision_params.get_stafit_for_cell(
                vision_cluster_id)
            self.current_sta_data = sta_data
            self.current_stafit = stafit  # <-- Store the fit
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

            # Use the RF canvas instead of the old sta_canvas
            self.rf_canvas.fig.clear()
            self.animate_sta_movie(
                self.rf_canvas.fig,
                sta_data,
                stafit=stafit,  # <-- Pass the fit to the plotting function
                frame_index=peak_frame_index,
                sta_width=self.main_window.data_manager.vision_sta_width,
                sta_height=self.main_window.data_manager.vision_sta_height
            )
            self.rf_canvas.draw()

            # --- Update New Panels ---
            self.plot_sta_metrics(cluster_id)
            self.plot_temporal_filter(cluster_id)

        else:
            # No Vision STA data available
            self.rf_canvas.fig.clear()
            self.rf_canvas.fig.text(
                0.5,
                0.5,
                "No Vision STA data available",
                ha='center',
                va='center',
                color='gray')
            self.rf_canvas.draw()
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
            self.timecourse_canvas.fig.clear()
            self.timecourse_canvas.fig.text(
                0.5,
                0.5,
                "No Vision STA data available",
                ha='center',
                va='center',
                color='gray')
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

            # Update the RF canvas with the first frame
            self.rf_canvas.fig.clear()
            self.animate_sta_movie(
                self.rf_canvas.fig,
                self.current_sta_data,
                stafit=self.current_stafit,
                frame_index=self.current_frame_index,
                sta_width=self.main_window.data_manager.vision_sta_width,
                sta_height=self.main_window.data_manager.vision_sta_height
            )
            self.rf_canvas.draw()
        else:
            self.rf_canvas.fig.clear()
            self.rf_canvas.fig.text(
                0.5,
                0.5,
                "No Vision STA data available",
                ha='center',
                va='center',
                color='gray')
            self.rf_canvas.draw()
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

            # Format metrics as HTML table with sections
            html = """
            <style>
                table { width: 100%; border-collapse: collapse; color: #e0e0e0; font-family: sans-serif; }
                th { text-align: left; color: #aaa; border-bottom: 1px solid #555; padding: 4px; }
                td { padding: 6px; border-bottom: 1px solid #333; }
                .section { font-weight: bold; color: #4282DA; margin-top: 12px; display: block; font-size: 1.1em; }
                .subsection { font-weight: bold; color: #6AA84F; margin-top: 8px; display: block; }
                .metric-name { font-weight: 600; color: #cccccc; }
                .metric-value { color: #ffffff; }
                .highlight { background-color: rgba(66, 130, 218, 0.1); }
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
            self.sta_metrics_text.setHtml(
                "<div style='color:gray; text-align:center; padding:20px;'>No STA Data Available</div>")

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
            self.temporal_filter_canvas.fig.clear()
            self.temporal_filter_canvas.fig.text(
                0.5, 0.5, "No Data", ha='center', va='center', color='gray')
            self.temporal_filter_canvas.draw()

    def animate_sta_movie(
            self,
            fig,
            sta_data,
            stafit=None,
            frame_index=0,
            sta_width=None,
            sta_height=None,
            ax=None):
        """
        Animates the STA movie by showing individual frames.
        MODIFIED: Now optionally overlays the STAFit ellipse.
        """
        if ax is None:
            fig.clear()
            ax = fig.add_subplot(111)

        n_frames = sta_data.red.shape[2]
        if frame_index >= n_frames:
            frame_index = 0

        red_frame = sta_data.red[:, :, frame_index]
        green_frame = sta_data.green[:, :, frame_index]
        blue_frame = sta_data.blue[:, :, frame_index]

        sta_rgb = np.stack([red_frame, green_frame, blue_frame], axis=-1)

        min_val, max_val = np.min(sta_rgb), np.max(sta_rgb)
        if max_val != min_val:
            sta_rgb_normalized = (sta_rgb - min_val) / (max_val - min_val)
        else:
            sta_rgb_normalized = np.zeros_like(sta_rgb)

        extent = [
            0,
            sta_width,
            sta_height,
            0] if sta_width is not None else [
            0,
            red_frame.shape[1],
            red_frame.shape[0],
            0]

        ax.imshow(sta_rgb_normalized, origin='upper', extent=extent)

        # --- ADDED: Logic to draw the STAFit ellipse if provided ---
        if stafit:
            if sta_height is not None:
                adjusted_y = sta_height - stafit.center_y
            else:
                image_height = red_frame.shape[0]
                adjusted_y = image_height - stafit.center_y

            from matplotlib.patches import Ellipse
            ellipse = Ellipse(
                xy=(stafit.center_x, adjusted_y),
                width=2 * stafit.std_x,
                height=2 * stafit.std_y,
                angle=np.rad2deg(stafit.rot),
                edgecolor='cyan',
                facecolor='none',
                lw=2
            )
            ax.add_patch(ellipse)

        ax.set_title(
            f"STA Movie - Frame {frame_index+1}/{n_frames}",
            color='white')
        ax.set_xlabel("X (stixels)", color='gray')
        ax.set_ylabel("Y (stixels)", color='gray')
        ax.set_facecolor('#1f1f1f')
        ax.tick_params(colors='gray')
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')

        fig.tight_layout()

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
                colors = ['red', 'green', 'blue'][:n_channels_to_plot]

                for i in range(n_channels_to_plot):
                    ax.plot(
                        time_axis,
                        timecourse_matrix[:,
                                          i],
                        color=colors[i],
                        linewidth=1.5,
                        label=channel_names[i])

                ax.set_title("STA Timecourse (Pre-calculated)", color='white')
                ax.set_xlabel("Time (ms)", color='gray')
                ax.set_ylabel("Response", color='gray')
                ax.grid(True, alpha=0.3)
                ax.legend(facecolor='#1f1f1f', labelcolor='white')

                ax.set_facecolor('#1f1f1f')
                ax.tick_params(colors='gray')
                for spine in ax.spines.values():
                    spine.set_edgecolor('gray')

                ax.axvline(x=0, color='white', linestyle='--', alpha=0.7)
                ax.axhline(y=0, color='white', linestyle=':',
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

        ax.set_title("STA Timecourse (Recalculated)", color='white')
        ax.set_xlabel("Time (ms)", color='gray')
        ax.set_ylabel("Response", color='gray')
        ax.grid(True, alpha=0.3)
        ax.legend(facecolor='#1f1f1f', labelcolor='white')

        ax.set_facecolor('#1f1f1f')
        ax.tick_params(colors='gray')
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')

        ax.axvline(x=0, color='white', linestyle='--', alpha=0.7)
        ax.axhline(y=0, color='white', linestyle=':',
                   alpha=0.5)  # Add dotted line at y=0

    def plot_temporal_filter_properties(self, fig, sta_data, stafit, vision_params, cell_id):
        """
        Plots the temporal filter with annotations for metrics.
        """
        fig.clear()
        ax = fig.add_subplot(111)

        time_axis, tc_matrix, _ = analysis_core.get_sta_timecourse_data(
            sta_data, stafit, vision_params, cell_id)

        if tc_matrix is None:
            ax.text(
                0.5,
                0.5,
                "No temporal data",
                ha='center',
                va='center',
                color='gray')
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
        ax.scatter([peak_time], [peak_val], color='white', s=50, zorder=5)
        ax.annotate(f"Peak: {peak_time:.1f}ms",
                    xy=(peak_time, peak_val),
                    xytext=(0, 10 if peak_val > 0 else -15),
                    textcoords='offset points',
                    ha='center', color='white', fontsize=9)

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
                    colors='yellow',
                    linestyles='-',
                    linewidth=2)
                ax.annotate(f"FWHM: {width * sample_interval:.1f}ms",
                            xy=((start_time + end_time) / 2, h),
                            xytext=(0, 5), textcoords='offset points',
                            ha='center', color='yellow', fontsize=8)
        except Exception:
            pass

        ax.set_title("Temporal Dynamics Analysis", color='white')
        ax.set_xlabel("Time (ms)", color='gray')
        ax.set_facecolor('#1f1f1f')
        ax.tick_params(colors='gray')
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
        ax.grid(True, alpha=0.2)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)