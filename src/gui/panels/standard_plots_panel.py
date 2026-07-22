import numpy as np
import pyqtgraph as pg
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QSplitter,
    QHBoxLayout,
    QCheckBox,
    QComboBox,
    QLabel,
    QPushButton,
    QDoubleSpinBox,
    QFrame,
)
from qtpy.QtCore import Qt
from ..theme import resolve_theme_colors
import logging

logger = logging.getLogger(__name__)
ISI_DENSITY_THRESHOLD = 5000  # switch to density view when > this many ISIs


# Configure pyqtgraph for antialiasing
pg.setConfigOptions(antialias=True)


class StandardPlotsPanel(QWidget):
    """
    Standard Dashboard:
    [ Template Grid ] [ Autocorrelation ]
    [ ISI Hist      ] [ Firing Rate     ]
    """

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        colors = resolve_theme_colors(self.main_window.get_current_colors())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Background image from array calibration
        self._array_bg_image = None
        self._array_transform_data = None
        self._array_image_path = None
        self._has_valid_array_transform = False

        # Controls: top control bar
        ctrl_bar = QHBoxLayout()
        ctrl_bar_widget = QWidget()
        ctrl_bar_widget.setFixedHeight(32)
        ctrl_bar_widget.setLayout(ctrl_bar)

        # Channel display mode
        ctrl_bar.addWidget(QLabel("Channel Display:"))
        self.channel_mode_combo = QComboBox()
        self.channel_mode_combo.addItems(
            ["Main Channel", "Top Channels", "Whole Array", "Array Image"]
        )
        ctrl_bar.addWidget(self.channel_mode_combo)

        ctrl_bar.addStretch()

        self.channel_mode_combo.currentTextChanged.connect(self._on_control_changed)

        layout.addWidget(ctrl_bar_widget)

        # 2x2 Layout using Splitters
        self.vert_splitter = QSplitter(Qt.Vertical)
        layout.addWidget(self.vert_splitter)

        self.top_splitter = QSplitter(Qt.Horizontal)
        self.bottom_splitter = QSplitter(Qt.Horizontal)
        self.vert_splitter.addWidget(self.top_splitter)
        self.vert_splitter.addWidget(self.bottom_splitter)

        # ---------------------------------------------------------
        # 1. Template Grid (Top Left)
        # ---------------------------------------------------------
        template_container = QWidget()
        template_layout = QVBoxLayout(template_container)
        template_layout.setContentsMargins(0, 0, 0, 0)

        self.grid_widget = pg.GraphicsLayoutWidget()
        self.grid_plot = self.grid_widget.addPlot()
        self.grid_plot.setTitle(
            f"<span style='color:{colors['text_tertiary']}; font-size:10px; letter-spacing:0.06em;'>SPATIAL TEMPLATE</span>"
        )
        self.grid_plot.setAspectLocked(True)
        self.grid_plot.hideAxis("bottom")
        self.grid_plot.hideAxis("left")

        template_layout.addWidget(self.grid_widget)
        self.top_splitter.addWidget(template_container)

        # ---------------------------------------------------------
        # 2. Autocorrelation (Top Right)
        # ---------------------------------------------------------
        self.acg_plot = pg.PlotWidget()
        self.acg_plot.setTitle(
            f"<span style='color:{colors['text_tertiary']}; font-size:10px; letter-spacing:0.06em;'>AUTOCORRELATION</span>"
        )
        self.acg_plot.setLabel("bottom", "Time lag (ms)")
        self.acg_plot.setLabel("left", "Autocorrelation")
        self._style_plot(self.acg_plot)

        # --- PERSISTENT ACG/CCG ITEMS ---
        # 1. ACG Line (Purple) - positive lags only
        self._acg_line = self.acg_plot.plot(
            [], [], pen=pg.mkPen(colors["plot_acg"], width=2)
        )

        # 2. CCG Bar (Orange) - Hidden by default
        self._ccg_bar = pg.BarGraphItem(
            x=[],
            height=[],
            width=0.8,
            brush=pg.mkBrush(colors["plot_compare"]),
            pen=pg.mkPen(colors["plot_compare"], width=1),
        )
        self.acg_plot.addItem(self._ccg_bar)
        self._ccg_bar.setVisible(False)

        # 3. Zero Line (only shown during CCG)
        self._acg_zero_line = pg.InfiniteLine(
            pos=0,
            angle=90,
            pen=pg.mkPen(colors["text_primary"], width=2, style=Qt.DashLine),
        )
        self.acg_plot.addItem(self._acg_zero_line)
        self._acg_zero_line.setVisible(False)

        self.top_splitter.addWidget(self.acg_plot)

        # ---------------------------------------------------------
        # 3. ISI (Bottom Left)
        # ---------------------------------------------------------
        isi_container = QWidget()
        isi_layout = QVBoxLayout(isi_container)
        isi_layout.setContentsMargins(0, 0, 0, 0)

        isi_controls = QHBoxLayout()
        isi_controls.setSpacing(4)

        # Group 1: View type
        self.isi_view_combo = QComboBox()
        self.isi_view_combo.addItems(["ISI Histogram", "ISI vs Amplitude"])
        self.isi_view_combo.setFixedHeight(24)
        isi_controls.addWidget(self.isi_view_combo)

        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet(f"color: {colors['border_subtle']};")
        sep1.setFixedWidth(1)
        isi_controls.addWidget(sep1)

        # Group 2: Refractory line
        self.show_refractory_line_checkbox = QCheckBox("Refr.")
        self.show_refractory_line_checkbox.setChecked(True)
        self.refractory_spinbox = QDoubleSpinBox()
        self.refractory_spinbox.setRange(0.1, 10.0)
        self.refractory_spinbox.setDecimals(1)
        self.refractory_spinbox.setSingleStep(0.1)
        self.refractory_spinbox.setValue(1.0)
        self.refractory_spinbox.setFixedWidth(52)
        self.refractory_spinbox.setFixedHeight(24)
        self.refractory_spinbox.setSuffix(" ms")
        self.update_refractory_btn = QPushButton("Set")
        self.update_refractory_btn.setFixedHeight(24)
        self.update_refractory_btn.setFixedWidth(44)

        isi_controls.addWidget(self.show_refractory_line_checkbox)
        isi_controls.addWidget(self.refractory_spinbox)
        isi_controls.addWidget(self.update_refractory_btn)

        isi_controls.addStretch()

        isi_layout.addLayout(isi_controls)

        self.isi_plot = pg.PlotWidget()
        self.isi_plot.setTitle(
            f"<span style='color:{colors['text_tertiary']}; font-size:10px; letter-spacing:0.06em;'>ISI DISTRIBUTION</span>"
        )
        self.isi_plot.setLabel("bottom", "ISI (ms)")
        self._style_plot(self.isi_plot)

        # --- PERSISTENT ISI ITEMS ---
        # 1. Histogram (filled line, bin centers)
        self._isi_curve = self.isi_plot.plot(
            [],
            [],
            stepMode=False,
            fillLevel=0,
            brush=pg.mkBrush(colors["plot_isi"]),
            pen=pg.mkPen(colors["plot_isi"], width=2),
        )

        # 2. Scatter
        self._isi_scatter = pg.ScatterPlotItem(
            size=5, pen=None, brush=pg.mkBrush(255, 165, 0, 150)
        )
        self.isi_plot.addItem(self._isi_scatter)
        self._isi_scatter.setVisible(False)

        # 3. Density
        self._isi_image = pg.ImageItem()
        self._hot_lut = self._create_hot_colormap()
        self._isi_image.setLookupTable(self._hot_lut)
        self.isi_plot.addItem(self._isi_image)
        self._isi_image.setVisible(False)

        # 4. Refractory Line
        self._isi_ref_line = pg.InfiniteLine(
            angle=90, pen=pg.mkPen("r", style=Qt.DashLine)
        )
        self.isi_plot.addItem(self._isi_ref_line)

        isi_layout.addWidget(self.isi_plot)
        self.bottom_splitter.addWidget(isi_container)

        self.isi_view_combo.currentTextChanged.connect(self._on_control_changed)
        self.show_refractory_line_checkbox.stateChanged.connect(
            self._on_control_changed
        )
        self.update_refractory_btn.clicked.connect(self._update_refractory_period)

        # ---------------------------------------------------------
        # 4. Firing Rate (Bottom Right)
        # ---------------------------------------------------------
        self.fr_plot = pg.PlotWidget()
        self.fr_plot.setTitle(
            f"<span style='color:{colors['text_tertiary']}; font-size:10px; letter-spacing:0.06em;'>SIGNAL HEALTH</span>"
        )
        self.fr_plot.setLabel("bottom", "Time (s)")
        self.fr_plot.setLabel("left", "Firing Rate (Hz)", color=colors["plot_fr"])
        self._style_plot(self.fr_plot)

        # --- PERSISTENT FR ITEMS ---
        # 1. Yellow Rate Curve
        self._fr_rate_curve = self.fr_plot.plot(
            [], [], pen=pg.mkPen(colors["plot_fr"], width=2), name="fr"
        )

        # 2. Green Overlay Curve
        self._fr_overlay_curve = self.fr_plot.plot(
            [],
            [],
            pen=pg.mkPen(colors["plot_overlay"], width=0.5),
            name="Averaged Amplitude",
        )

        self.bottom_splitter.addWidget(self.fr_plot)

        self.vert_splitter.setSizes([500, 300])
        self._hot_lut_local = self._hot_lut

    def _update_refractory_period(self):
        new_period = self.refractory_spinbox.value()
        self.main_window.data_manager.set_refractory_period(new_period)
        cluster_id = self.main_window._get_selected_cluster_id()
        if cluster_id is not None:
            self.update_all(cluster_id)

    def restyle_plots(self, colors):
        """Updates plot styling based on the provided color scheme."""
        self._style_plot(self.grid_plot, colors)
        self._style_plot(self.acg_plot, colors)
        self._style_plot(self.isi_plot, colors)
        self._style_plot(self.fr_plot, colors)

        # Update Titles
        self.grid_plot.setTitle(
            f"<span style='color:{colors['text_tertiary']}; font-size:10px; letter-spacing:0.06em;'>SPATIAL TEMPLATE</span>"
        )
        self.acg_plot.setTitle(
            f"<span style='color:{colors['text_tertiary']}; font-size:10px; letter-spacing:0.06em;'>AUTOCORRELATION</span>"
        )
        self.isi_plot.setTitle(
            f"<span style='color:{colors['text_tertiary']}; font-size:10px; letter-spacing:0.06em;'>INTER-SPIKE INTERVAL</span>"
        )
        self.fr_plot.setTitle(
            f"<span style='color:{colors['text_tertiary']}; font-size:10px; letter-spacing:0.06em;'>FIRING RATE (OVER TIME)</span>"
        )

        # Update specific items
        self._acg_line.setPen(pg.mkPen(colors["plot_acg"], width=2))
        self._ccg_bar.setOpts(
            brush=pg.mkBrush(colors["plot_compare"]),
            pen=pg.mkPen(colors["plot_compare"], width=1),
        )
        self._acg_zero_line.setPen(
            pg.mkPen(colors["text_primary"], width=2, style=Qt.DashLine)
        )
        self._isi_curve.setPen(pg.mkPen(colors["plot_isi"], width=2))
        if hasattr(self._isi_curve, "curve") and hasattr(
            self._isi_curve.curve, "setBrush"
        ):
            self._isi_curve.curve.setBrush(pg.mkBrush(colors["plot_isi"]))
        self._fr_rate_curve.setPen(pg.mkPen(colors["plot_fr"], width=2))
        self._fr_overlay_curve.setPen(pg.mkPen(colors["plot_overlay"], width=0.5))
        self.fr_plot.setLabel("left", "Firing Rate (Hz)", color=colors["plot_fr"])

        # Refresh widgets
        self.grid_widget.setBackground(colors["bg_panel"])

    def _style_plot(self, plot_widget, colors=None):
        if colors is None:
            colors = resolve_theme_colors(self.main_window.get_current_colors())
        else:
            colors = resolve_theme_colors(colors)

        # Handle both PlotWidget and PlotItem (for GraphicsLayoutWidget)
        if isinstance(plot_widget, pg.PlotWidget):
            plot_item = plot_widget.getPlotItem()
            plot_widget.setBackground(colors["bg_panel"])
        else:
            plot_item = plot_widget

        plot_item.getAxis("bottom").setPen(pg.mkPen(colors["border_default"]))
        plot_item.getAxis("left").setPen(pg.mkPen(colors["border_default"]))
        plot_item.getAxis("bottom").setTextPen(pg.mkPen(colors["text_secondary"]))
        plot_item.getAxis("left").setTextPen(pg.mkPen(colors["text_secondary"]))

        # Hide top and right spines
        plot_item.showAxis("top", False)
        plot_item.showAxis("right", False)

        # Subtle grid
        plot_item.showGrid(x=True, y=True, alpha=0.08)

        # Remove the default blue border pyqtgraph adds
        plot_item.setContentsMargins(8, 8, 8, 8)

    def _create_hot_colormap(self):
        colors = [(0, 0, 0), (255, 0, 0), (255, 255, 0), (255, 255, 255)]
        positions = [0, 0.33, 0.66, 1.0]
        cmap = pg.ColorMap(pos=positions, color=colors)
        return cmap.getLookupTable(start=0.0, stop=1.0, nPts=256)

    def _apply_isi_range_preset(self, isi_ms):
        """Hardcoded 0–100 ms x-range for ISI plot."""
        self.isi_plot.setXRange(0, 100.0, padding=0)
        self.isi_plot.plotItem.disableAutoRange(pg.ViewBox.XAxis)

    def _on_control_changed(self):
        try:
            cluster_id = self.main_window._get_selected_cluster_id()
            if cluster_id is not None:
                self.update_all(cluster_id)
        except Exception:
            pass

    def refresh_array_image(self, transform_path: str):
        """Loads and aligns the microscope image behind the template grid."""
        import json
        from pathlib import Path
        from PIL import Image
        import numpy as np
        from qtpy.QtCore import QRectF
        import logging

        logger = logging.getLogger(__name__)

        try:
            with open(transform_path, "r") as f:
                data = json.load(f)

            img_name = data.get("image_file")
            if not img_name:
                return

            img_path = Path(transform_path).parent / img_name
            if not img_path.exists():
                return

            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img, dtype=np.uint8)
            img_h, img_w = img_array.shape[:2]

            # Calibration was done with invertY=True (pixel y=0 at top).
            # grid_plot is Y-up, so we flip the image vertically before display.
            img_pg = img_array[::-1, :, :].transpose(1, 0, 2)  # flip rows → (W, H, C)

            if self._array_bg_image is None:
                self._array_bg_image = pg.ImageItem()
                self._array_bg_image.setZValue(-20)
                self.grid_plot.addItem(self._array_bg_image)

            self._array_bg_image.setImage(img_pg)

            self._array_transform_data = data
            self._array_image_path = img_path
            self._has_valid_array_transform = True

            # Transform (calibrated): pixel = scale * micron + offset
            # Invert to get micron corners from pixel corners.
            sx = float(data.get("scale_x", 1.0))
            sy = float(data.get("scale_y", 1.0))
            ox = float(data.get("offset_x", 0.0))
            oy = float(data.get("offset_y", 0.0))

            x0 = (0 - ox) / sx  # left micron edge   (pixel x=0)
            x1 = (img_w - ox) / sx  # right micron edge  (pixel x=W)
            y_top = (0 - oy) / sy  # micron Y of pixel row 0 (top of image)
            y_bottom = (img_h - oy) / sy  # micron Y of pixel row H (bottom of image)

            # In a Y-up plot, y_top > y_bottom (larger micron = higher on screen).
            # setRect(left, bottom, width, height) — height is positive upward.
            bottom = min(y_top, y_bottom)
            height = abs(y_top - y_bottom)
            self._array_bg_image.setRect(QRectF(x0, bottom, x1 - x0, height))

        except Exception as e:
            logger.warning(f"Failed to load array image: {e}")
            self._has_valid_array_transform = False

    def cleanup(self):
        """Explicitly cleanup pyqtgraph resources to prevent memory leaks."""
        plots = [self.grid_plot, self.acg_plot, self.isi_plot, self.fr_plot]
        widgets = [self.grid_widget]

        for plot in plots:
            if plot:
                plot.clear()
                # If it's a PlotWidget, we can close it
                if hasattr(plot, "close"):
                    plot.close()

        for widget in widgets:
            if widget:
                widget.close()
                widget.deleteLater()

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)

    def update_all(self, cluster_id):
        """
        Update all standard plots for the given cluster.
        Uses batch rendering with disabled auto-range for single render pass.
        """
        if cluster_id is None:
            return

        dm = self.main_window.data_manager
        if dm is None:
            return
        colors = resolve_theme_colors(self.main_window.get_current_colors())

        # Disable auto-range for batch rendering (prevents multiple render passes)
        plots_to_update = [self.grid_plot, self.acg_plot, self.isi_plot, self.fr_plot]
        for plot in plots_to_update:
            plot.disableAutoRange()

        # If user requests array image but none is loaded, prompt for calibration
        current_mode = self.channel_mode_combo.currentText()
        if current_mode == "Array Image" and not self._has_valid_array_transform:
            from qtpy.QtWidgets import QMessageBox

            QMessageBox.information(
                self,
                "No Array Image",
                "No calibrated array image is available. "
                "Please map an image to the array using File → Array → Map Image to Array...",
            )
            try:
                self.main_window._open_array_calibration()
            except Exception:
                pass
            return

        # ------------------------------------------------------------------
        # 1. TEMPLATE GRID
        # ------------------------------------------------------------------
        self.grid_plot.clear()

        # Only show background when in Array Image mode
        if self._array_bg_image is not None and current_mode == "Array Image":
            self.grid_plot.addItem(self._array_bg_image)

        try:
            if (
                hasattr(dm, "templates")
                and dm.templates is not None
                and cluster_id < dm.templates.shape[0]
            ):
                template = dm.templates[cluster_id]
                pos = np.array(dm.channel_positions)

                # --- Dynamic Stretch: 1.0 (true physical) for Image, 1.5 (distorted) for traces ---
                x_scale = 1.0 if current_mode == "Array Image" else 1.5
                y_scale = 1.0

                ptp = template.max(axis=0) - template.min(axis=0)
                max_ptp = ptp.max() if ptp.size > 0 else 1.0
                main_channel_idx = int(np.argmax(ptp)) if ptp.size > 0 else 0

                if current_mode == "Main Channel":
                    relevant_channels = [main_channel_idx]
                    show_waveforms = True
                    show_dots = False
                    waveform_channels = [main_channel_idx]
                elif current_mode == "Top Channels":
                    top_channel_indices = np.argsort(ptp)[::-1][:3]
                    relevant_channels = top_channel_indices
                    show_waveforms = True
                    show_dots = True
                    waveform_channels = top_channel_indices
                elif current_mode in ("Whole Array", "Array Image"):
                    relevant_channels = np.arange(len(ptp))
                    waveform_channels = np.argsort(ptp)[::-1][:6]
                    show_waveforms = True
                    show_dots = True
                else:
                    relevant_channels = np.arange(len(ptp))
                    waveform_channels = np.argsort(ptp)[::-1][:6]
                    show_waveforms = True
                    show_dots = True

                cluster_amp = dm.get_cluster_mean_amplitude(cluster_id, method="mean")
                norm_ptp = ptp / max_ptp if max_ptp > 0 else np.zeros_like(ptp)
                channel_values = norm_ptp * cluster_amp

                if show_dots:
                    spots = []
                    vmin, vmax = channel_values.min(), channel_values.max()
                    vrange = max(vmax - vmin, 1e-6)

                    # --- Lower opacity for Array Image mode so tissue shows through ---
                    alpha = 80 if current_mode == "Array Image" else 180

                    for ch in relevant_channels:
                        if ch >= len(channel_values) or ch >= len(pos):
                            continue
                        x, y = pos[ch]
                        val = (channel_values[ch] - vmin) / vrange
                        size = 6 + val * 20
                        r = int(255 * val)
                        g = int(120 * (1 - val))
                        b = int(255 * (1 - val))

                        spots.append(
                            {
                                "pos": (x * x_scale, y * y_scale),
                                "size": size,
                                "brush": pg.mkBrush(r, g, b, alpha),
                                "pen": pg.mkPen(None),
                            }
                        )

                    if spots:
                        scatter = pg.ScatterPlotItem(size=8)
                        scatter.addPoints(spots)
                        scatter.setZValue(-10)
                        self.grid_plot.addItem(scatter)

                if show_waveforms:
                    for ch in waveform_channels:
                        if ch >= len(pos):
                            continue
                        x, y = pos[ch]
                        trace = template[:, ch]
                        trace_scaled = (trace / max_ptp) * 20 if max_ptp > 0 else trace
                        t_offset = np.linspace(-10, 10, len(trace))
                        self.grid_plot.plot(
                            x * x_scale + t_offset,
                            y * y_scale + trace_scaled,
                            pen=pg.mkPen(colors["plot_waveform_shadow"], width=2.5),
                            alpha=0.6,
                        )
                        self.grid_plot.plot(
                            x * x_scale + t_offset,
                            y * y_scale + trace_scaled,
                            pen=pg.mkPen(colors["plot_line"], width=1.2),
                        )

                    if (
                        current_mode == "Main Channel"
                        and waveform_channels
                        and main_channel_idx < len(pos)
                    ):
                        _, main_y = pos[main_channel_idx]
                        self.grid_plot.addItem(
                            pg.InfiniteLine(
                                pos=main_y * y_scale,
                                angle=0,
                                pen=pg.mkPen(
                                    colors["text_primary"], width=1, style=Qt.DashLine
                                ),
                            )
                        )

                sim_panel = getattr(self.main_window, "similarity_panel", None)
                selected_similar = []
                try:
                    if sim_panel and getattr(sim_panel, "table", None):
                        sel_model = sim_panel.table.selectionModel()
                        if sel_model:
                            selected_similar = sel_model.selectedRows()
                except Exception:
                    pass

                if selected_similar and hasattr(sim_panel, "similarity_model"):
                    sim_model = sim_panel.similarity_model
                    if sim_model and hasattr(sim_model, "_dataframe"):
                        for idx in selected_similar[:3]:
                            row = idx.row()
                            if row >= len(sim_model._dataframe):
                                continue
                            similar_id = sim_model._dataframe.iloc[row]["cluster_id"]
                            if similar_id < dm.templates.shape[0]:
                                sim_template = dm.templates[similar_id]
                                for ch in waveform_channels:
                                    if ch >= len(pos):
                                        continue
                                    x, y = pos[ch]
                                    trace = sim_template[:, ch]
                                    trace_scaled = (
                                        (trace / max_ptp) * 20 if max_ptp > 0 else trace
                                    )
                                    t_offset = np.linspace(-10, 10, len(trace))
                                    self.grid_plot.plot(
                                        x * x_scale + t_offset,
                                        y * y_scale + trace_scaled,
                                        pen=pg.mkPen(colors["plot_compare"], width=1.5),
                                    )

                # --- Lock the zoom to perfectly frame the electrodes ---
                if current_mode in ("Whole Array", "Array Image"):
                    min_x, min_y = np.min(pos, axis=0)
                    max_x, max_y = np.max(pos, axis=0)

                    # Account for the scale stretch
                    min_x *= x_scale
                    max_x *= x_scale
                    min_y *= y_scale
                    max_y *= y_scale

                    span_x = max_x - min_x
                    span_y = max_y - min_y

                    pad_x = span_x * 0.9
                    pad_y = span_y * 0.2

                    self.grid_plot.setXRange(min_x - pad_x, max_x + pad_x, padding=0)
                    self.grid_plot.setYRange(min_y - pad_y, max_y + pad_y, padding=0)

        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Error drawing template grid: {e}")

        # ------------------------------------------------------------------
        # 2. FETCH DATA (non-blocking — never compute on UI thread)
        # ------------------------------------------------------------------
        try:
            data = dm.try_get_standard_plot_data(cluster_id)
        except Exception:
            data = None

        if data is None:
            # Cache miss — queue for background computation, don't block
            worker = getattr(self.main_window, "standard_plots_worker", None)
            if worker is not None:
                worker.add_to_queue(cluster_id, high_priority=True)
            # Clear plots so the user sees them as "pending"
            self._acg_line.setData([], [])
            self._ccg_bar.setVisible(False)
            self._isi_curve.setData([], [])
            self._isi_scatter.setVisible(False)
            self._fr_rate_curve.setData([], [])
            self._fr_overlay_curve.setData([], [])
            self.acg_plot.setTitle(
                f"<span style='color:{colors['text_tertiary']}; font-size:10px;'>AUTOCORRELATION — computing…</span>"
            )
            for plot in plots_to_update:
                plot.enableAutoRange()
            return

        # Guard: if acg_norm is absent the cluster has no usable data yet
        if data is None or data.get("acg_norm") is None:
            self._acg_line.setData([], [])
            self._ccg_bar.setVisible(False)
            self._isi_curve.setData([], [])
            self._fr_rate_curve.setData([], [])
            return

        # ------------------------------------------------------------------
        # 3. ACG / CCG
        # ------------------------------------------------------------------
        sim_panel = getattr(self.main_window, "similarity_panel", None)
        selected_similar_rows = []
        try:
            if sim_panel and getattr(sim_panel, "table", None):
                sel = sim_panel.table.selectionModel()
                if sel:
                    selected_similar_rows = sel.selectedRows()
        except:
            pass

        showing_ccg = False

        if selected_similar_rows and hasattr(sim_panel, "similarity_model"):
            sim_model = sim_panel.similarity_model
            if sim_model and hasattr(sim_model, "_dataframe"):
                row_idx = selected_similar_rows[0].row()
                if 0 <= row_idx < len(sim_model._dataframe):
                    similar_id = int(sim_model._dataframe.iloc[row_idx]["cluster_id"])

                    if similar_id != cluster_id:
                        try:
                            # Fetch both spike trains on-demand — memmap slices, fast
                            spikes1 = dm.get_cluster_spikes(cluster_id)
                            spikes2 = dm.get_cluster_spikes(similar_id)

                            if (
                                spikes1 is not None
                                and spikes2 is not None
                                and len(spikes1) > 1
                                and len(spikes2) > 1
                            ):
                                sr = float(dm.sampling_rate)
                                s1 = (np.asarray(spikes1) / sr * 1000).astype(int)
                                s2 = (np.asarray(spikes2) / sr * 1000).astype(int)
                                duration = int(max(np.max(s1), np.max(s2)))

                                if duration > 0:
                                    from scipy.signal import correlate

                                    bins = np.arange(0, duration + 1, 1, dtype=int)
                                    b1, _ = np.histogram(s1, bins=bins)
                                    b2, _ = np.histogram(s2, bins=bins)

                                    if len(b1) > 0 and len(b2) > 0:
                                        c1 = b1 - np.mean(b1)
                                        c2 = b2 - np.mean(b2)
                                        ccg_full = correlate(c1, c2, mode="full")

                                        zero = len(ccg_full) // 2
                                        lag = 100
                                        ccg_sym = ccg_full[zero - lag : zero + lag + 1]
                                        lags = np.arange(-lag, lag + 1)

                                        var = np.sqrt(np.var(b1) * np.var(b2))
                                        norm = (
                                            ccg_sym / var / len(b1)
                                            if var != 0
                                            else ccg_sym.astype(float)
                                        )

                                        self._ccg_bar.setOpts(x=lags, height=norm)
                                        self._ccg_bar.setVisible(True)
                                        self._acg_line.setVisible(False)
                                        self._acg_zero_line.setVisible(True)
                                        self.acg_plot.setTitle(
                                            f"CCG: {cluster_id} vs {similar_id}"
                                        )
                                        showing_ccg = True
                        except Exception:
                            pass

        if not showing_ccg:
            time_lags = data.get("acg_time_lags")
            acg_norm = data.get("acg_norm")

            if time_lags is not None and acg_norm is not None and len(time_lags) > 1:
                # Positive lags only, rendered as a line
                mask = np.asarray(time_lags) >= 0
                self._acg_line.setData(time_lags[mask], acg_norm[mask])
                self._acg_line.setVisible(True)
                self._ccg_bar.setVisible(False)
                self._acg_zero_line.setVisible(False)
                self.acg_plot.setXRange(0, float(time_lags[mask].max()), padding=0.02)
                self.acg_plot.setTitle("Autocorrelation")
            else:
                self._acg_line.setData([], [])
                self._ccg_bar.setVisible(False)
                self._acg_zero_line.setVisible(False)

        # ------------------------------------------------------------------
        # 4. ISI PLOT
        # ------------------------------------------------------------------
        isi_view = self.isi_view_combo.currentText()

        # Fetch spikes on-demand — memmap slice, GC'd after this block
        _spikes_raw = dm.get_cluster_spikes(cluster_id)
        _spikes_raw = (
            np.asarray(_spikes_raw) if _spikes_raw is not None else np.array([])
        )
        if _spikes_raw.size > 1:
            raw_isi_ms = np.diff(_spikes_raw) / float(dm.sampling_rate) * 1000.0
        else:
            raw_isi_ms = np.array([])

        if isi_view == "ISI Histogram":
            self._isi_curve.setVisible(True)
            self._isi_scatter.setVisible(False)
            self._isi_image.setVisible(False)

            if len(raw_isi_ms) > 0:
                bins = np.linspace(
                    0, 150, 151
                )  # 1 ms bins, headroom past 100 ms display
                hist_y, hist_x = np.histogram(raw_isi_ms, bins=bins)
                bin_centers = 0.5 * (hist_x[:-1] + hist_x[1:])
                self._isi_curve.setData(bin_centers, hist_y)
            else:
                self._isi_curve.setData([], [])

            if self.show_refractory_line_checkbox.isChecked():
                self._isi_ref_line.setValue(dm.get_refractory_period())
                self._isi_ref_line.setVisible(True)
            else:
                self._isi_ref_line.setVisible(False)

            self._apply_isi_range_preset(raw_isi_ms)
            self.isi_plot.setLabel("bottom", "ISI (ms)")
            self.isi_plot.setLabel("left", "Count")

        elif isi_view == "ISI vs Amplitude":
            self._isi_curve.setVisible(False)
            self._isi_ref_line.setVisible(False)
            self._isi_image.setVisible(False)

            valid_isi = None
            valid_amp = None
            _spikes_isi = dm.get_cluster_spikes(cluster_id)
            _amps = dm.get_cluster_spike_amplitudes(cluster_id)
            _spikes_isi = (
                np.asarray(_spikes_isi) if _spikes_isi is not None else np.array([])
            )
            _amps = np.asarray(_amps)
            if _spikes_isi.size > 1 and _amps.size > 1:
                _isi = np.diff(_spikes_isi) / float(dm.sampling_rate) * 1000.0
                _min_len = min(len(_isi), _amps.size - 1)
                if _min_len > 0:
                    valid_isi = _isi[:_min_len]
                    valid_amp = _amps[1 : _min_len + 1]

            if valid_isi is not None and len(valid_isi) > 0:
                self._isi_scatter.setData(valid_isi, valid_amp)
                self._isi_scatter.setVisible(True)
                self._apply_isi_range_preset(valid_isi)
                self.isi_plot.setLabel("bottom", "ISI (ms)")
                self.isi_plot.setLabel("left", "Amplitude (µV)")
            else:
                self._isi_scatter.setData([], [])
                self._isi_scatter.setVisible(False)

        # ------------------------------------------------------------------
        # 5. FIRING RATE
        # ------------------------------------------------------------------
        fr_bin_centers = data.get("fr_bin_centers")
        fr_rate = data.get("fr_rate")

        if fr_bin_centers is not None and fr_rate is not None:
            self._fr_rate_curve.setData(fr_bin_centers, fr_rate)
        else:
            self._fr_rate_curve.setData([], [])

        # FR overlay — recompute on-demand from amplitudes; spike-length arrays
        # are never cached (OOM fix). Memory is reclaimed after this block.
        overlay_x = None
        overlay_y = None
        _amps_ov = dm.get_cluster_spike_amplitudes(cluster_id)
        _amps_ov = np.asarray(_amps_ov) if _amps_ov is not None else np.array([])
        if (
            _amps_ov.size > 10
            and fr_rate is not None
            and fr_rate.size > 0
            and _spikes_raw.size > 10
        ):
            _max_amp = float(np.max(_amps_ov))
            if _max_amp > 0:
                _norm_amp = _amps_ov / _max_amp
            else:
                _norm_amp = _amps_ov.astype(float)
            _avg_amp = np.convolve(_norm_amp, np.ones(10) / 10.0, mode="valid")
            _scaled = _avg_amp * 0.8 * float(np.max(fr_rate))
            _spikes_sec_ov = _spikes_raw / float(dm.sampling_rate)
            _ov_len = min(len(_scaled), len(_spikes_sec_ov))
            if _ov_len > 0:
                overlay_x = _spikes_sec_ov[:_ov_len]
                overlay_y = _scaled[:_ov_len]

        if overlay_x is not None and overlay_y is not None:
            self._fr_overlay_curve.setData(overlay_x, overlay_y)
        else:
            self._fr_overlay_curve.setData([], [])

        # Re-enable auto-range after batch updates
        for plot in plots_to_update:
            plot.enableAutoRange()
