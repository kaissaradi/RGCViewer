import os
import logging
from qtpy.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSplitter, QStatusBar,
    QHeaderView, QMessageBox, QTabWidget,
    QTreeView, QAbstractItemView, QSlider, QLabel,
    QMenu, QInputDialog, QStackedWidget, QApplication,
    QTextEdit, QCheckBox
)
from qtpy.QtCore import Qt, QItemSelectionModel, QThread, QTimer
from qtpy.QtGui import QFont, QStandardItemModel
import pyqtgraph as pg
from ..analysis.data_manager import DataManager
from typing import Optional
# Custom GUI Modules
from .widgets.widgets import MplCanvas, HighlightStatusPandasModel, CustomTableView
from . import callbacks
from .plotting import plotting
from .panels.similarity_panel import SimilarityPanel
from .panels.waveforms_panel import WaveformPanel
from .panels.standard_plots_panel import StandardPlotsPanel
from .panels.ei_panel import EIPanel
from .panels.raw_panel import RawPanel
from .workers.workers import FeatureWorker
from .shortcuts import KeyForwarder
from PyQt5.QtGui import QColor
from .panels.umap_panel import UMAPPanel

# Global pyqtgraph configuration
pg.setConfigOption('background', '#1f1f1f')
pg.setConfigOption('foreground', 'd')

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, default_kilosort_dir=None, default_dat_file=None):
        super().__init__()
        self.setWindowTitle("axolotl")
        self.setGeometry(50, 50, 1800, 1000)

        # --- Application State ---
        self.data_manager: Optional[DataManager] = None
        self.main_cluster_model = None

        self.tree_model = QStandardItemModel()
        self.refine_thread = None
        self.refinement_worker = None

        # Spatial (EI) worker
        self.worker_thread = None
        self.spatial_worker = None

        # NEW: standard-plots (ISI/ACG/FR) worker
        self.standard_worker_thread = None
        self.standard_plots_worker = None

        self.spatial_plot_dirty = False

        self.current_spatial_features = None
        # --- Timer for EI Animation ---
        self.ei_animation_timer = None  # To prevent garbage collection
        # --- Current STA View ---
        self.current_sta_view = "rf"  # Default to RF plot
        # --- STA Animation State ---
        self.current_sta_data = None
        self.current_sta_cluster_id = None
        self.current_frame_index = 0
        self.total_sta_frames = 0
        self.current_stafit = None  # Added to store STAFit data
        self.sta_animation_timer = None
        self._is_syncing = False
        self.last_left_width = 450
        self.feature_worker_thread = None
        self.population_view_enabled = False

        # --- UI Setup ---
        self._setup_style()
        self._setup_ui()
        self.analysis_tabs.currentChanged.connect(self.on_tab_changed)
        self.central_widget.setEnabled(False)
        self.status_bar.showMessage(
            "Welcome to axolotl. Please load a Kilosort directory to begin.")

        # selection timer for debouncing rapid selections
        self.selection_timer = QTimer(self)
        self.selection_timer.setSingleShot(True)
        self.selection_timer.setInterval(150)  # 150ms delay
        self.selection_timer.timeout.connect(self._process_selection)
        self._pending_cluster_id = None

        # Auto-load if default paths are provided
        if default_kilosort_dir and os.path.isdir(default_kilosort_dir):
            self.load_directory(default_kilosort_dir, default_dat_file)

        # key forwarder
        self.key_forwarder = KeyForwarder(self)
        QApplication.instance().installEventFilter(self.key_forwarder)

    def _move_selection_in_view(self, view, key):
        sel_model = view.selectionModel()
        model = view.model()
        if not sel_model or not model:
            return

        if view is self.tree_view:
            # Build a flat list of all leaf indices (clusters)
            leaf_indices = []
            for group_row in range(model.rowCount()):
                group_item = model.item(group_row)
                for child_row in range(group_item.rowCount()):
                    child_item = group_item.child(child_row)
                    index = model.indexFromItem(child_item)
                    leaf_indices.append(index)
            if not leaf_indices:
                return
            # Find the currently selected leaf
            selected = sel_model.selectedIndexes()
            if not selected or selected[0] not in leaf_indices:
                # Select first leaf if nothing is selected or selection is not
                # a leaf
                sel_model.select(
                    leaf_indices[0],
                    QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
                view.scrollTo(leaf_indices[0])
                return
            current_idx = leaf_indices.index(selected[0])
            if key == Qt.Key_Up:
                new_idx = max(0, current_idx - 1)
            else:
                new_idx = min(len(leaf_indices) - 1, current_idx + 1)
            sel_model.select(
                leaf_indices[new_idx],
                QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
            view.scrollTo(leaf_indices[new_idx])
        else:
            # Table view logic
            current = view.currentIndex()
            if not current.isValid():
                # Select first row if nothing is selected
                index = model.index(0, 0)
                sel_model.setCurrentIndex(
                    index, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
                view.scrollTo(index)
                return
            current_row = current.row()
            if key == Qt.Key_Up:
                new_row = max(0, current_row - 1)
            else:
                new_row = min(model.rowCount() - 1, current_row + 1)
            index = model.index(new_row, 0)
            sel_model.setCurrentIndex(
                index, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
            view.scrollTo(index)

    def _setup_style(self):
        """Sets the application's stylesheet."""
        self.setFont(QFont("Segoe UI", 9))
        self.setStyleSheet("""
            QWidget { color: white; background-color: #2D2D2D; }
            QTableView { background-color: #191919; alternate-background-color: #252525; gridline-color: #454545; }
            QHeaderView::section { background-color: #353535; padding: 4px; border: 1px solid #555555; }
            QPushButton { background-color: #353535; border: 1px solid #555555; padding: 5px; border-radius: 4px; }
            QPushButton:hover { background-color: #454545; }
            QPushButton:pressed { background-color: #252525; }
            QTabWidget::pane { border: 1px solid #4282DA; }
            QTabBar::tab { color: white; background: #353535; padding: 8px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #4282DA; }
            QStatusBar { color: white; }
        """)

    def update_cluster_views(self, cluster_id):
        """
        Receives a selection event, stores the cluster_id, and restarts the
        selection timer. This 'debounces' rapid selections.
        """
        self._pending_cluster_id = cluster_id
        self.selection_timer.start()

    def _process_selection(self):
        """
        This method is called by the timer ONLY after the user has stopped
        scrolling. It performs the actual data loading for the last selected cluster.
        """
        cluster_id = self._pending_cluster_id
        if cluster_id is None:
            return

        self.status_bar.showMessage(
            f"Loading data for Cluster ID: {cluster_id}...")

        cached_features = self.data_manager.get_lightweight_features(
            cluster_id)
        if cached_features:
            self._draw_plots(cluster_id, cached_features)
            return

        # Only run FeatureWorker if dat_path is available
        if self.data_manager.dat_path is not None:
            # --- FIX: Ensure the previous worker is fully terminated before starting a new one.
            if self.feature_worker_thread and self.feature_worker_thread.isRunning():
                self.feature_worker_thread.quit()
                # ensure previous thread fully stops before starting a new one
                self.feature_worker_thread.wait()

            self.feature_worker_thread = QThread()
            self.feature_worker = FeatureWorker(self.data_manager, cluster_id)
            self.feature_worker.moveToThread(self.feature_worker_thread)
            self.feature_worker.features_ready.connect(self.on_features_ready)
            self.feature_worker.error.connect(
                lambda msg: self.status_bar.showMessage(msg, 4000))
            self.feature_worker_thread.started.connect(self.feature_worker.run)
            self.feature_worker_thread.start()
        else:
            self.status_bar.showMessage(
                "Raw data file not loaded: waveform plot disabled.", 4000)
            self._draw_plots(cluster_id, None)

    def on_features_ready(self, cluster_id, features):
        """
        Slot that receives the features from the background worker.
        """
        # Cache the newly computed features.
        self.data_manager.ei_cache[cluster_id] = features

        # VERY IMPORTANT: Only draw if the returned data is for the currently selected cluster.
        # This prevents a slow, old request from overwriting a new, quick one.
        if cluster_id == self._get_selected_cluster_id():
            self._draw_plots(cluster_id, features)

        self.feature_worker_thread.quit()
        try:
            # Wait for the thread to finish teardown to avoid races
            if self.feature_worker_thread and self.feature_worker_thread.isRunning():
                self.feature_worker_thread.wait()
        except Exception:
            pass

    def on_tab_changed(self, index):
        """
        Handles updates when the user switches tabs OR when the active tab
        is refreshed after a cluster change.

        Only the active panel is updated.
        """
        cluster_id = self._get_selected_cluster_id()
        if cluster_id is None:
            return

        current_panel = self.analysis_tabs.widget(index)

        if current_panel == self.standard_plots_panel:
            # Only compute standard plots when this tab is actually visible
            self.standard_plots_panel.update_all(cluster_id)

        elif current_panel == self.ei_panel:
            self.ei_panel.update_ei([cluster_id])

        elif current_panel == self.waveforms_panel:
            self.waveforms_panel.update_all(cluster_id)

        elif current_panel == self.raw_panel:
            self.raw_panel.load_data(cluster_id)

        elif current_panel == self.sta_panel:
            if self.data_manager and self.data_manager.vision_stas:
                self.select_sta_view(self.current_sta_view)
            else:
                # Use the appropriate canvas based on current view - use RF
                # canvas as default
                canvas_to_use = self.rf_canvas
                canvas_to_use.fig.clear()
                canvas_to_use.fig.text(
                    0.5,
                    0.5,
                    "No Vision STA data available",
                    ha='center',
                    va='center',
                    color='gray',
                )
                canvas_to_use.draw()

    def _draw_plots(self, cluster_id, features):
        """Only update what's actually visible."""

        current_tab = self.analysis_tabs.currentWidget()

        # --- ONLY UPDATE STANDARD PLOTS WHEN THAT TAB IS VISIBLE ---
        if current_tab == self.standard_plots_panel:
            self.standard_plots_panel.update_all(cluster_id)
            self.similarity_panel.update_main_cluster_id(cluster_id)

        # --- UPDATE ONLY THE ACTIVE TAB ---
        if current_tab == self.ei_panel:
            self.ei_panel.update_ei([cluster_id])
            self.similarity_panel.update_main_cluster_id(cluster_id)

        elif current_tab == self.waveforms_panel:
            self.waveforms_panel.update_all(cluster_id)

        elif current_tab == self.raw_panel:
            self.raw_panel.load_data(cluster_id)

        elif current_tab == self.sta_panel:
            # STA tab must be FAST — no standard plots, no recompute
            if self.data_manager and self.data_manager.vision_stas:
                self.select_sta_view(self.current_sta_view)

        self.status_bar.showMessage("Ready.", 2000)

    def _setup_ui(self):
        """Initializes and lays out all the UI widgets."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QHBoxLayout(self.central_widget)

        # --- Left Pane ---
        self.left_pane = QWidget()
        left_layout = QVBoxLayout(self.left_pane)

        # Add a toggle button for collapsing/expanding the sidebar
        self.sidebar_toggle_button = QPushButton("◀")
        self.sidebar_toggle_button.setFixedWidth(20)
        self.sidebar_toggle_button.clicked.connect(self.toggle_sidebar)
        self.sidebar_collapsed = False

        # Create a widget to contain the filter box and views
        left_content = QWidget()
        left_content_layout = QVBoxLayout(left_content)

        # Add filter controls
        filter_box = QHBoxLayout()
        self.filter_button = QPushButton("Filter 'Good'")
        self.reset_button = QPushButton("Reset View")
        filter_box.addWidget(self.filter_button)
        filter_box.addWidget(self.reset_button)

        # --- View Switcher ---
        view_switch_layout = QHBoxLayout()
        self.tree_view_button = QPushButton("Tree View")
        self.table_view_button = QPushButton("Table View")
        self.tree_view_button.clicked.connect(
            lambda: self._switch_left_view(0))
        self.table_view_button.clicked.connect(
            lambda: self._switch_left_view(1))
        view_switch_layout.addWidget(self.tree_view_button)
        view_switch_layout.addWidget(self.table_view_button)

        # --- View Stack (Tree and Table) ---
        self.view_stack = QStackedWidget()

        # Tree View
        self.tree_view = QTreeView()
        self.tree_view.setHeaderHidden(True)
        self.tree_view.setDragEnabled(True)
        self.tree_view.setAcceptDrops(True)
        self.tree_view.setDropIndicatorShown(True)
        self.tree_view.setDragDropMode(
            QAbstractItemView.DragDropMode.InternalMove)
        self.tree_view.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(
            self.open_tree_context_menu)

        # Table View
        self.table_view = CustomTableView()
        self.table_view.setSortingEnabled(True)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive)

        self.view_stack.addWidget(self.tree_view)
        self.view_stack.addWidget(self.table_view)
        # Default to table view
        self.view_stack.setCurrentIndex(1)

        self.refine_button = QPushButton("Refine Selected Cluster")
        self.refine_button.setFixedHeight(40)
        self.refine_button.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #aeffe3; background-color: #005230;")

        left_content_layout.addLayout(filter_box)
        left_content_layout.addLayout(view_switch_layout)
        left_content_layout.addWidget(self.view_stack)
        left_content_layout.addWidget(self.refine_button)

        # --- Similarity Panel ---
        self.similarity_panel = SimilarityPanel(self)
        left_content_layout.addWidget(self.similarity_panel)
        self.similarity_panel.selection_changed.connect(
            self.on_similarity_selection_changed)

        # Add the toggle button and content to the left pane
        left_layout.addWidget(self.sidebar_toggle_button)
        left_layout.addWidget(left_content)
        # Store reference to content widget for collapsing/expanding
        self.left_content = left_content

        # --- Right Pane (Tabbed Interface) ---
        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Create Tab Widget
        self.analysis_tabs = QTabWidget()

        # New Checkbox for Population Split View (Moved to Top Right of Tabs)
        self.pop_view_checkbox = QCheckBox("Population Split View")
        self.pop_view_checkbox.setFocusPolicy(Qt.NoFocus)
        self.pop_view_checkbox.toggled.connect(
            self.toggle_population_split_view)

        # Place checkbox in the top-right corner of the tab bar
        self.analysis_tabs.setCornerWidget(
            self.pop_view_checkbox, Qt.TopRightCorner)

        # --- NEW: population context widget (right side) ---
        self.pop_context_widget = QWidget()
        # ---- inside _setup_ui(): build pop_context_widget contents ----
        pop_layout = QVBoxLayout(self.pop_context_widget)
        pop_layout.setContentsMargins(4, 4, 4, 4)
        pop_layout.setSpacing(6)

        # Top: Population RF (existing)
        self.pop_mosaic_canvas = MplCanvas(width=6, height=4, dpi=100)
        pop_layout.addWidget(self.pop_mosaic_canvas, stretch=3)

        # Middle + Bottom: make a vertical splitter with two panels
        # (middle=timecourse, bottom=placeholder)
        self.pop_timecourse_splitter = QSplitter(Qt.Orientation.Vertical)
        # Middle panel widget
        self.pop_timecourse_widget = QWidget()
        mid_layout = QVBoxLayout(self.pop_timecourse_widget)
        mid_layout.setContentsMargins(2, 2, 2, 2)

        # Header row for middle panel: title + summary
        hdr = QHBoxLayout()
        hdr_label = QLabel("Population Average Timecourse")
        hdr_label.setStyleSheet("font-weight:bold;")
        self.pop_timecourse_summary = QLabel(
            "n=0  mean_t2p: N/A  mean_fwhm: N/A")
        hdr.addWidget(hdr_label)
        hdr.addStretch()
        hdr.addWidget(self.pop_timecourse_summary)
        mid_layout.addLayout(hdr)

        # Middle canvas (timecourse)
        self.pop_timecourse_canvas = MplCanvas(width=6, height=2, dpi=100)
        mid_layout.addWidget(self.pop_timecourse_canvas)

        self.pop_timecourse_splitter.addWidget(self.pop_timecourse_widget)

        # Bottom placeholder panel
        self.pop_bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(self.pop_bottom_widget)
        bottom_layout.setContentsMargins(2, 2, 2, 2)
        bottom_label = QLabel("Population - Reserved")
        bottom_layout.addWidget(bottom_label)
        self.pop_bottom_canvas = MplCanvas(width=6, height=2, dpi=100)
        bottom_layout.addWidget(self.pop_bottom_canvas)
        self.pop_timecourse_splitter.addWidget(self.pop_bottom_widget)

        # add the splitter into the pop_layout
        pop_layout.addWidget(self.pop_timecourse_splitter, stretch=2)

        # initial sizes (you can tweak)
        self.pop_timecourse_splitter.setSizes([200, 120])

        # --- NEW: right-side splitter containing tabs and pop widget ---
        self.right_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.right_splitter.addWidget(self.analysis_tabs)
        self.right_splitter.addWidget(self.pop_context_widget)

        # Start hidden by default
        self.pop_context_widget.hide()
        # initial ratio: all space to tabs
        self.right_splitter.setSizes([1200, 0])

        right_layout.addWidget(self.right_splitter)

        # --- Panels ---
        self.standard_plots_panel = StandardPlotsPanel(self)
        self.ei_panel = EIPanel(self)
        self.waveforms_panel = WaveformPanel(self)
        self.raw_panel = RawPanel(self)

        self.umap_panel = UMAPPanel(self)

        # --- STA Analysis Panel (Re-parented) ---
        self.sta_panel = QWidget()
        sta_layout = QVBoxLayout(self.sta_panel)

        # Control buttons layout
        sta_control_layout = QHBoxLayout()
        self.sta_population_rfs_button = QPushButton("Population RFs")
        self.sta_animation_button = QPushButton("Play Animation")
        self.sta_animation_stop_button = QPushButton("Stop Animation")

        self.sta_population_rfs_button.clicked.connect(lambda: self.select_sta_view(
            "rf" if self.current_sta_view == "population_rfs" else "population_rfs"))
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

        sta_layout.setStretch(1, 0)

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

        self.sta_canvas = MplCanvas(self, width=10, height=8, dpi=120)
        self.sta_canvas.hide()

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

        sta_layout.addLayout(sta_control_layout, 0)
        sta_layout.addLayout(self.sta_frame_controls_layout, 0)
        sta_layout.addWidget(self.sta_splitter, 1)

        # --- Tab Order ---
        self.analysis_tabs.addTab(self.standard_plots_panel, "Standard Plots")
        self.analysis_tabs.addTab(self.ei_panel, "EI Analysis")
        self.analysis_tabs.addTab(self.sta_panel, "STA Analysis")
        self.analysis_tabs.addTab(self.umap_panel, "Class Discovery (UMAP)")
        self.analysis_tabs.addTab(self.waveforms_panel, "Raw Waveforms")
        self.analysis_tabs.addTab(self.raw_panel, "Raw Trace")

        # --- Main Splitter and Layout ---
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.addWidget(self.left_pane)
        self.main_splitter.addWidget(right_pane)
        self.main_splitter.setSizes([450, 1350])  # Adjusted initial size
        main_layout.addWidget(self.main_splitter)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # --- Menu Bar ---
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        load_ks_action = file_menu.addAction("&Load Kilosort Directory...")
        self.load_vision_action = file_menu.addAction("&Load Vision Files...")
        self.load_vision_action.setEnabled(False)
        self.load_classification_action = file_menu.addAction(
            "&Load Classification File...")
        self.load_classification_action.setEnabled(False)
        self.save_action = file_menu.addAction("&Save Results...")
        self.save_action.setEnabled(False)

        self.save_classification_action = file_menu.addAction("Save Classification Text File...")
        self.save_classification_action.setEnabled(False)  # Disabled until data loads
        # -----------------------

        self.save_action = file_menu.addAction("&Save Results...")
        self.save_action.setEnabled(False)

        # Connect Signals
        load_ks_action.triggered.connect(lambda: self.load_directory())
        self.load_vision_action.triggered.connect(self.load_vision_directory)
        self.load_classification_action.triggered.connect(self.load_classification_file)
        
        # --- NEW CONNECTION ---
        self.save_classification_action.triggered.connect(self.on_save_classification_action)
        # ----------------------
        
        self.save_action.triggered.connect(self.on_save_action)

        # Connect Signals to Callback Functions ---
        load_ks_action.triggered.connect(lambda: self.load_directory())
        self.load_vision_action.triggered.connect(self.load_vision_directory)
        self.load_classification_action.triggered.connect(
            self.load_classification_file)
        self.save_action.triggered.connect(self.on_save_action)
        self.filter_button.clicked.connect(self.apply_good_filter)
        self.reset_button.clicked.connect(self.reset_views)
        self.refine_button.clicked.connect(self.on_refine_cluster)
        self.analysis_tabs.currentChanged.connect(self.on_tab_changed)

        # Connect the raw panel's status and error messages to the status bar
        self.raw_panel.status_message.connect(
            lambda msg: self.status_bar.showMessage(msg, 3000))
        self.raw_panel.error_message.connect(
            lambda msg: self.status_bar.showMessage(msg, 4000))

    def toggle_population_split_view(self, checked: bool):
        """Toggles the global population context pane (right side)."""
        self.population_view_enabled = bool(checked)

        if checked:
            # show the right-hand population widget
            self.pop_context_widget.show()

            # Expand it to a sensible size (give about 20-30% to the right
            # pane)
            total = sum(self.right_splitter.sizes()) or 1400
            left_size = max(int(total * 0.75), 400)
            right_size = total - left_size
            self.right_splitter.setSizes([left_size, right_size])

            # If a cluster/cell is selected, draw its population mosaic
            # immediately
            selected = None
            try:
                selected = self._get_selected_cluster_id()  # adapt to your selector fun
            except Exception:
                selected = None

            # Call plotting routine with explicit canvas
            plotting.draw_population_rfs_plot(
                main_window=self,
                selected_cell_id=selected,
                canvas=self.pop_mosaic_canvas)
            callbacks.redraw_population_panels(self)
        else:
            # hide it
            self.pop_context_widget.hide()
            # collapse the right column completely
            self.right_splitter.setSizes([sum(self.right_splitter.sizes()), 0])

    def _switch_left_view(self, index):
        """Switches between the tree and table views in the left pane."""
        self.view_stack.setCurrentIndex(index)

    # --- Helper Method ---
    def _get_selected_cluster_id(self):
        """Returns the cluster_id of the currently selected item from the active view."""
        current_view_index = self.view_stack.currentIndex()

        # Case 1: Tree View is active
        if current_view_index == 0:
            if not self.tree_view.selectionModel().hasSelection():
                return None

            index = self.tree_view.selectionModel().selectedIndexes()[0]
            item = self.tree_model.itemFromIndex(index)

            # Only leaf nodes (cells) have a cluster ID stored. Groups will
            # return None.
            cluster_id = item.data(Qt.ItemDataRole.UserRole)
            return cluster_id

        # Case 2: Table View is active
        elif current_view_index == 1:
            if not self.table_view.selectionModel(
            ).hasSelection() or self.main_cluster_model is None:
                return None

            selected_row = self.table_view.selectionModel().selectedIndexes()[
                0].row()

            # Check if the model has mapToSource method (for proxy models)
            model = self.table_view.model()
            if hasattr(model, 'mapToSource'):
                # The pandas model can be sorted, so we must map the view's row
                # to the model's row
                source_index = model.mapToSource(model.index(selected_row, 0))
                cluster_id = model._dataframe.iloc[source_index.row(
                )]['cluster_id']
            else:
                # If no proxy model, use the row directly
                cluster_id = model._dataframe.iloc[selected_row]['cluster_id']
            return cluster_id

        return None
    
    def on_save_classification_action(self):
        """Wrapper to call the callback function."""
        callbacks.save_classification_to_file(self)

    def _get_group_cluster_ids(self, item):
        cluster_ids = []
        for i in range(item.rowCount()):
            child = item.child(i)
            cid = child.data(Qt.ItemDataRole.UserRole)
            if cid is not None:
                cluster_ids.append(cid)

        return cluster_ids

    def _get_pop_subset_ids(self):
        """
        Gets the list of cluster IDs for the currently selected population subset.
        If a single cell is selected, it finds its group and returns all cells in that group.
        If a group is selected, it returns all cells in that group.
        """
        cluster_id = self._get_selected_cluster_id()

        # Case 1: A single cluster is selected. Find its group.
        if cluster_id is not None:
            df = self.data_manager.cluster_df
            if not df.empty and 'cluster_id' in df.columns and 'KSLabel' in df.columns:
                if cluster_id in df['cluster_id'].values:
                    try:
                        row = df[df['cluster_id'] == cluster_id].iloc[0]
                        group_label = row.get('KSLabel')
                        if group_label:
                            return df[df['KSLabel'] ==
                                      group_label]['cluster_id'].tolist()
                    except Exception as e:
                        logger.warning(
                            f"Could not determine group for cluster {cluster_id}: {e}")
            return [cluster_id]  # Fallback to just the selected cluster

        # Case 2: A group/folder is selected in the Tree View
        elif self.view_stack.currentIndex() == 0:  # Tree View
            selection = self.tree_view.selectionModel().selectedIndexes()
            if selection:
                index = selection[0]
                item = self.tree_model.itemFromIndex(index)
                if item and item.data(
                        Qt.ItemDataRole.UserRole) is None:  # It's a group
                    return self._get_group_cluster_ids(item)

        return []  # Return empty list if no valid selection

    def setup_tree_model(self, model):
        """Sets up the tree view model and connects the selection changed signal."""
        self.tree_view.setModel(model)
        try:
            self.tree_view.selectionModel().selectionChanged.disconnect(
                self.on_view_selection_changed)
        except (TypeError, RuntimeError):
            pass
        self.tree_view.selectionModel().selectionChanged.connect(
            self.on_view_selection_changed)

    def setup_table_model(self, model):
        """Sets up the table view model and connects the selection changed signal."""
        self.table_view.setModel(model)
        try:
            self.table_view.selectionModel().selectionChanged.disconnect(
                self.on_view_selection_changed)
        except (TypeError, RuntimeError):
            pass
        self.table_view.selectionModel().selectionChanged.connect(
            self.on_view_selection_changed)

    # --- Methods to bridge UI signals to callback functions ---
    def load_directory(self, kilosort_dir=None, dat_file=None):
        callbacks.load_directory(self, kilosort_dir, dat_file)

    def load_vision_directory(self):
        callbacks.load_vision_directory(self)

    def on_view_selection_changed(self, _selected, _deselected):
        """
        Handles a selection change in either view, synchronizes the other view,
        and then triggers the main plot update callback.
        """
        if self._is_syncing:
            return

        self._is_syncing = True

        cluster_id = self._get_selected_cluster_id()
        sender = self.sender()

        if cluster_id is not None:
            # Sync from Tree to Table
            if sender == self.tree_view.selectionModel():
                model = self.table_view.model()
                if hasattr(model, '_data'):
                    df = model._data
                    if cluster_id in df['cluster_id'].values:
                        row_indices = df.index[df['cluster_id']
                                               == cluster_id].tolist()
                        if row_indices:
                            model_row = df.index.get_loc(row_indices[0])
                            source_index = model.index(model_row, 0)
                            # This assumes the model is a proxy model if
                            # sorting is enabled
                            view_index = model.mapFromSource(source_index) if hasattr(
                                model, 'mapFromSource') else source_index
                            if view_index.isValid():
                                self.table_view.selectionModel().select(
                                    view_index, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
                                self.table_view.scrollTo(
                                    view_index, QAbstractItemView.ScrollHint.PositionAtCenter)

            # Sync from Table to Tree
            elif sender == self.table_view.selectionModel():
                for row in range(self.tree_model.rowCount()):
                    group_item = self.tree_model.item(row)
                    if not group_item:
                        continue
                    for child_row in range(group_item.rowCount()):
                        child_item = group_item.child(child_row)
                        if child_item and child_item.data(
                                Qt.ItemDataRole.UserRole) == cluster_id:
                            index = self.tree_model.indexFromItem(child_item)
                            self.tree_view.selectionModel().select(
                                index, QItemSelectionModel.ClearAndSelect)
                            self.tree_view.scrollTo(
                                index, QAbstractItemView.ScrollHint.PositionAtCenter)
                            break
                    else:
                        continue
                    break

        # Now that views are synced, trigger the update callbacks
        callbacks.on_cluster_selection_changed(self)
        self._is_syncing = False

        self.similarity_panel.reset_spacebar_counter()

    def on_similarity_selection_changed(self, selected_cluster_ids):
        # Always include the main selected cluster if there is one selected
        main_cluster = self._get_selected_cluster_id()
        if main_cluster is not None and len(selected_cluster_ids) > 0:
            clusters_to_plot = [main_cluster] + selected_cluster_ids
        elif main_cluster is not None and len(selected_cluster_ids) == 0:
            # If no similar clusters are selected, just plot the main cluster
            clusters_to_plot = [main_cluster]
        else:
            # If no main cluster is selected, just plot the selected similar
            # clusters
            clusters_to_plot = selected_cluster_ids
        logger.debug(
            f'on_similarity_selection_changed: main_cluster = {main_cluster}')
        logger.debug(
            f'on_similarity_selection_changed: clusters_to_plot = {clusters_to_plot}')

        self.ei_panel.update_ei(clusters_to_plot)
        self.waveforms_panel.update_all(main_cluster)

        if main_cluster is not None:
            self.standard_plots_panel.update_all(main_cluster)

    def _update_table_view_duplicate_highlight(self):
        df = self.data_manager.cluster_df
        self.main_cluster_model = HighlightStatusPandasModel(df)
        self.setup_table_model(self.main_cluster_model)

    def _update_tree_view_duplicate_highlight(self):
        # Collect all duplicate IDs
        sdf = self.data_manager.status_df
        duplicate_ids = sdf[sdf['status'] ==
                            'Duplicate']['cluster_id'].tolist()
        duplicate_ids = set(duplicate_ids)
        for row in range(self.tree_model.rowCount()):
            group_item = self.tree_model.item(row)
            for child_row in range(group_item.rowCount()):
                child_item = group_item.child(child_row)
                if child_item.data(Qt.ItemDataRole.UserRole) in duplicate_ids:
                    child_item.setForeground(QColor('#FF2222'))  # Red text
                else:
                    child_item.setForeground(QColor('white'))

    def on_cluster_selection_changed(self, *args):
        callbacks.on_cluster_selection_changed(self)

    def on_spatial_data_ready(self, cluster_id, features):
        callbacks.on_spatial_data_ready(self, cluster_id, features)

    def on_refine_cluster(self):
        callbacks.on_refine_cluster(self)

    def handle_refinement_results(self, parent_id, new_clusters):
        callbacks.handle_refinement_results(self, parent_id, new_clusters)

    def handle_refinement_error(self, error_message):
        callbacks.handle_refinement_error(self, error_message)

    def on_save_action(self):
        callbacks.on_save_action(self)

    def apply_good_filter(self):
        callbacks.apply_good_filter(self)

    def reset_views(self):
        callbacks.reset_views(self)

    def select_sta_view(self, view_type, force_animation=False):
        """Select the STA view to display."""
        self.current_sta_view = view_type
        cluster_id = self._get_selected_cluster_id()
        if cluster_id is None:
            return

        # Only proceed if vision STA data is available
        if not self.data_manager or not self.data_manager.vision_stas:
            # Clear both canvases
            self.rf_canvas.fig.clear()
            self.timecourse_canvas.fig.clear()
            self.rf_canvas.fig.text(
                0.5,
                0.5,
                "No Vision STA data available",
                ha='center',
                va='center',
                color='gray')
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
            return

        # Update button text based on current view
        if view_type == "rf":
            self.sta_animation_button.setText("Play Animation")
        elif view_type == "animation":
            self.sta_animation_button.setText("Pause Animation")
        elif view_type == "population_rfs":
            self.sta_animation_button.setText("Play Animation")

        # Draw the single-cell plots for the STA quad-view.
        # `draw_sta_plot` handles the main RF canvas and metrics.
        plotting.draw_sta_plot(self, cluster_id)
        plotting.draw_sta_timecourse_plot(self, cluster_id)

        # Handle specific view-type overrides for the main RF canvas
        if view_type == "population_rfs":
            # This button press should always draw the population plot in the MAIN STA view (rf_canvas),
            # overriding the single-cell RF plot drawn by draw_sta_plot above.
            plotting.draw_population_rfs_plot(
                self, selected_cell_id=cluster_id, canvas=self.rf_canvas)
        elif view_type == "animation" or force_animation:
            # Animation should only affect the RF plot
            plotting.draw_sta_animation_plot(self, cluster_id)

    def update_sta_frame_manual(self, frame_index):
        """Updates the STA visualization to a specific frame manually."""
        if hasattr(
                self,
                'current_sta_data') and self.current_sta_data is not None:
            # Stop any running animation
            plotting.stop_sta_animation(self)

            # Update the frame index
            self.current_frame_index = frame_index

            # Update the label
            self.sta_frame_label.setText(
                f"Frame: {frame_index+1}/{self.total_sta_frames}")
            # Update the STA canvas with the new frame - use RF canvas for
            # animation
            self.rf_canvas.fig.clear()
            plotting.animate_sta_movie(
                self.rf_canvas.fig,
                self.current_sta_data,
                frame_index=frame_index,
                sta_width=self.data_manager.vision_sta_width,
                sta_height=self.data_manager.vision_sta_height
            )
            cluster_id = self.current_sta_cluster_id - 1  # Convert back to 0-indexed
            self.rf_canvas.fig.suptitle(
                f"Cluster {cluster_id} - STA Frame {frame_index+1}/{self.total_sta_frames}",
                color='white',
                fontsize=16)  # this overlaps with self.sta_frame_label
            self.rf_canvas.draw()

    def _advance_frame_internal(self):
        """Internal method for the timer to call without stopping itself."""
        if hasattr(
                self,
                'current_sta_data') and self.current_sta_data is not None:
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
            plotting.animate_sta_movie(
                self.rf_canvas.fig,
                self.current_sta_data,
                stafit=self.current_stafit,
                frame_index=self.current_frame_index,
                sta_width=self.data_manager.vision_sta_width,
                sta_height=self.data_manager.vision_sta_height
            )
            self.rf_canvas.draw()

    def prev_sta_frame(self):
        """Go to the previous frame in the STA animation."""
        if hasattr(
                self,
                'current_sta_data') and self.current_sta_data is not None:
            # Stop the animation when manually navigating
            plotting.stop_sta_animation(self)
            self.current_frame_index = (
                self.current_frame_index - 1) % self.total_sta_frames
            self.sta_frame_slider.setValue(self.current_frame_index)
            self.sta_frame_label.setText(
                f"Frame: {self.current_frame_index+1}/{self.total_sta_frames}")
            self.rf_canvas.fig.clear()
            plotting.animate_sta_movie(
                self.rf_canvas.fig,
                self.current_sta_data,
                stafit=self.current_stafit,  # <-- Pass the stored fit
                frame_index=self.current_frame_index,
                sta_width=self.data_manager.vision_sta_width,
                sta_height=self.data_manager.vision_sta_height
            )
            self.rf_canvas.draw()

    def next_sta_frame(self):
        """Go to the next frame in the STA animation."""
        if hasattr(
                self,
                'current_sta_data') and self.current_sta_data is not None:
            # Stop the animation when manually navigating
            plotting.stop_sta_animation(self)
            self.current_frame_index = (
                self.current_frame_index + 1) % self.total_sta_frames
            self.sta_frame_slider.setValue(self.current_frame_index)
            self.sta_frame_label.setText(
                f"Frame: {self.current_frame_index+1}/{self.total_sta_frames}")
            self.rf_canvas.fig.clear()
            plotting.animate_sta_movie(
                self.rf_canvas.fig,
                self.current_sta_data,
                stafit=self.current_stafit,  # <-- Pass the stored fit
                frame_index=self.current_frame_index,
                sta_width=self.data_manager.vision_sta_width,
                sta_height=self.data_manager.vision_sta_height
            )
            self.rf_canvas.draw()

    def load_classification_file(self):
        callbacks.load_classification_file(self)

    def open_tree_context_menu(self, position):
        menu = QMenu()
        index = self.tree_view.indexAt(position)
        item = self.tree_model.itemFromIndex(index)
        add_group_action = menu.addAction("Add New Group")

        if item.hasChildren():  # when clicking the group item
            feature_extraction_action = menu.addAction("Feature Extraction")

        action = menu.exec(self.tree_view.viewport().mapToGlobal(position))

        if action == add_group_action:
            text, ok = QInputDialog.getText(
                self, 'New Group', 'Enter group name:')
            if ok and text:
                callbacks.add_new_group(self, text)
        elif action == feature_extraction_action:
            cluster_ids = self._get_group_cluster_ids(item)
            callbacks.feature_extraction(self, cluster_ids)

    def toggle_animation(self):
        """Toggle the animation between play and pause."""
        if not self.data_manager or not self.data_manager.vision_stas:
            # No data available
            return

        cluster_id = self._get_selected_cluster_id()
        if cluster_id is None:
            return

        # Update the animation button text based on current state
        if hasattr(
                self,
                'sta_animation_timer') and self.sta_animation_timer and self.sta_animation_timer.isActive():
            # Currently playing, so stop it
            plotting.stop_sta_animation(self)
            self.sta_animation_button.setText("Play Animation")
        else:
            # Currently paused or stopped, so start it
            plotting.draw_sta_animation_plot(self, cluster_id)
            self.sta_animation_button.setText("Pause Animation")

    def on_rf_canvas_clicked(self):
        """Handle clicks on the RF canvas in STA tab - toggle between RF and animation."""
        if not self.data_manager or not self.data_manager.vision_stas:
            return

        cluster_id = self._get_selected_cluster_id()
        if cluster_id is None:
            return

        # Toggle between RF and animation views
        if self.current_sta_view == "rf":
            # Start animation
            self.current_sta_view = "animation"
            self.select_sta_view("animation", force_animation=True)
            self.status_bar.showMessage(
                "Started animation. Click again to stop.", 2000)
        elif self.current_sta_view == "animation":
            # Stop animation and go back to RF
            self.stop_animation()
            self.current_sta_view = "rf"
            self.select_sta_view("rf")
            self.status_bar.showMessage("Stopped animation.", 2000)
        elif self.current_sta_view == "population_rfs":
            # From population view, go to RF view
            self.current_sta_view = "rf"
            self.select_sta_view("rf")
            self.status_bar.showMessage(
                "Switched to single-cell RF view.", 2000)

    def stop_animation(self):
        """Stop the animation completely."""
        plotting.stop_sta_animation(self)
        self.sta_animation_button.setText(
            "Play Animation")  # Reset button to Play

    def toggle_sidebar(self):
        """Collapses or expands the left sidebar by manipulating the main splitter."""
        if self.sidebar_collapsed:
            # --- EXPAND ---
            self.sidebar_toggle_button.setText("◀")
            widths = self.main_splitter.sizes()
            total_width = sum(widths)
            self.main_splitter.setSizes(
                [self.last_left_width, total_width - self.last_left_width])
            self.sidebar_collapsed = False
        else:
            # --- COLLAPSE ---
            self.sidebar_toggle_button.setText("▶")
            widths = self.main_splitter.sizes()
            # Save the current width if it's not already collapsed
            if widths[0] > 35:
                self.last_left_width = widths[0]
            total_width = sum(widths)
            self.main_splitter.setSizes([35, total_width - 35])
            self.sidebar_collapsed = True

    def closeEvent(self, event):
        """Handles the window close event."""
        if self.data_manager and self.data_manager.is_dirty:
            reply = QMessageBox.question(
                self,
                'Unsaved Changes',
                "You have unsaved refinement changes. Do you want to save before exiting?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Save:
                self.on_save_action()
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
        callbacks.stop_worker(self)
        # Stop any running raw trace worker
        if hasattr(
                self,
                'raw_panel') and self.raw_panel.worker_thread and self.raw_panel.worker_thread.isRunning():
            self.raw_panel.worker_thread.quit()
            self.raw_panel.worker_thread.wait()
        event.accept()
