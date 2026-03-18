import os
import logging
from qtpy.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSplitter, QStatusBar,
    QHeaderView, QMessageBox, QTabWidget,
    QTreeView, QAbstractItemView, QSlider, QLabel,
    QMenu, QInputDialog, QStackedWidget, QApplication,
    QTextEdit, QCheckBox, QProgressBar
)
from qtpy.QtCore import Qt, QItemSelectionModel, QThread, QTimer
from qtpy.QtGui import QFont, QStandardItemModel
import pyqtgraph as pg
from ..analysis.data_manager import DataManager
from typing import Optional
# Custom GUI Modules
from .widgets.widgets import MplCanvas, HighlightStatusPandasModel, CustomTableView
from . import callbacks
from .panels.population_panel import (
    draw_population_timecourse_panel,
    draw_population_rfs_plot,
    plot_population_rfs,
    plot_rich_ei
)
from .panels.similarity_panel import SimilarityPanel
from .panels.waveforms_panel import WaveformPanel
from .panels.standard_plots_panel import StandardPlotsPanel
from .panels.ei_panel import EIPanel
from .panels.raw_panel import RawPanel
from .panels.sta_panel import STAPanel
from .workers.workers import FeatureWorker
from .shortcuts import KeyForwarder
from PyQt5.QtGui import QColor
from .panels.umap_panel import UMAPPanel
# Array calibration dialog
from .panels.array_calibration_panel import ArrayCalibrationDialog

# Global pyqtgraph configuration
pg.setConfigOption('background', '#1f1f1f')
pg.setConfigOption('foreground', 'd')

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, default_kilosort_dir=None, default_dat_file=None):
        super().__init__()
        self.setWindowTitle("RGC Viewer")
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

        # Additional thread references for proper cleanup
        self.ks_load_thread = None
        self.vision_load_thread = None

        self.spatial_plot_dirty = False

        self.current_spatial_features = None
        # --- Timer for EI Animation ---
        self.ei_animation_timer = None  # To prevent garbage collection
        # --- Current STA View ---
        self.current_sta_view = "rf"  # Default to RF plot
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
            "Welcome to RGC Viewer. Please load a Kilosort directory to begin.")

        # selection timer for debouncing rapid selections
        self.selection_timer = QTimer(self)
        self.selection_timer.setSingleShot(True)
        self.selection_timer.setInterval(25)  # 25ms - minimal delay for responsive feel
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

        current = view.currentIndex()
        
        # If nothing is selected, select the first visible row
        if not current.isValid():
            index = model.index(0, 0)
            sel_model.setCurrentIndex(
                index, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
            view.scrollTo(index)
            return

        # Tree View Logic (Respects nested/collapsed visual states)
        if view is self.tree_view:
            if key == Qt.Key_Up:
                new_idx = view.indexAbove(current)
            else:
                new_idx = view.indexBelow(current)
                
            # Only move if the new index is valid (prevents scrolling off the edge)
            if new_idx.isValid():
                sel_model.setCurrentIndex(
                    new_idx, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
                view.scrollTo(new_idx)

        # Table View Logic (Flat list)
        else:
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
        Receives a selection event.

        TIER 1 (Immediate): Updates lightweight plots (RF, Standard Plots, Cached EI).
        TIER 2 (Debounced): Starts timer for heavy tasks.
        """
        self._pending_cluster_id = cluster_id

        # --- TIER 1: IMMEDIATE UPDATES (Hot-Swap / Cached Data) ---

        # 1. Population RF Updates - Only if hot-swap is possible (fast path)
        # Full rebuild is deferred to Tier 2 to avoid first-click freeze
        if self.population_view_enabled:
            # Quick check if we can hot-swap (check if canvas already has state)
            canvas = self.pop_mosaic_canvas
            can_hot_swap = (
                hasattr(canvas, '_pop_plot_state') and
                canvas._pop_plot_state.get('ax') in canvas.fig.axes
            )
            if can_hot_swap:
                # Fast update - update existing ellipse geometry only
                try:
                    draw_population_rfs_plot(
                        main_window=self,
                        selected_cell_id=cluster_id,
                        canvas=self.pop_mosaic_canvas
                    )
                except Exception as e:
                    logger.error(f"Tier 1 Pop Split update failed: {e}")
            # If can't hot-swap, defer to Tier 2 for full rebuild

        # B. STA Tab (Main Center Pane) - Only update if explicitly in 'Population' mode
        if self.analysis_tabs.currentWidget() == self.sta_panel:
            if self.current_sta_view == 'population_rfs':
                # Population RF in STA tab - also defer to Tier 2
                pass  # Will be handled in _draw_plots

        # 2. Standard Plots Panel (ACG, ISI, Firing Rate) - handled in _draw_plots to avoid duplicate updates
        # Standard plots are updated only in _draw_plots to prevent redundant redraws

        # 3. Electrical Image (EI) - Only if cached
        has_cached_ei = False
        if self.data_manager:
            if hasattr(self.data_manager, 'has_cached_ei'):
                has_cached_ei = self.data_manager.has_cached_ei(cluster_id)
            elif hasattr(self.data_manager, 'ei_cache'):
                has_cached_ei = cluster_id in self.data_manager.ei_cache

        if self.ei_panel.isVisible() and has_cached_ei:
            try:
                self.ei_panel.update_ei([cluster_id])
            except Exception as e:
                logger.error(f"Tier 1 EI update failed: {e}")

        # --- TIER 2: HEAVY LIFTING (Queued) ---
        # Start/Restart the timer.
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
            # Cleanup previous worker before starting a new one
            self._cleanup_thread('feature_worker_thread')

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
        Cache features and update UI ONLY if still the current selection.
        Prevents stale data from overwriting fresh results.
        """
        current_selection = self._get_selected_cluster_id()

        # CRITICAL: Discard stale results BEFORE caching
        if cluster_id != current_selection:
            logger.debug(f"Discarding stale features for C{cluster_id} (now viewing C{current_selection})")
            return

        # Cache the newly computed features
        self.data_manager.ei_cache[cluster_id] = features

        # Only draw if still on a tab that needs these features
        current_tab = self.analysis_tabs.currentWidget()
        if current_tab in (self.ei_panel, self.waveforms_panel, self.standard_plots_panel):
            self._draw_plots(cluster_id, features)

        # Cleanup with timeout to prevent hangs
        self._cleanup_thread('feature_worker_thread')

    def _cleanup_thread(self, thread_attr: str, timeout_ms: int = 2000):
        """
        Safely cleanup a QThread and its worker with timeout.
        Prevents memory leaks and application hangs.
        """
        thread = getattr(self, thread_attr, None)
        if thread and thread.isRunning():
            thread.quit()
            if not thread.wait(timeout_ms):  # Timeout prevents infinite waits
                logger.warning(f"Thread {thread_attr} didn't exit cleanly, terminating")
                thread.terminate()
                thread.wait(1000)
        setattr(self, thread_attr, None)

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
            try:
                self.ei_panel.update_ei([cluster_id])
            except Exception as e:
                logger.error(f"Tab change EI update failed: {e}")

        elif current_panel == self.waveforms_panel:
            self.waveforms_panel.update_all(cluster_id)

        elif current_panel == self.raw_panel:
            self.raw_panel.load_data(cluster_id)

        elif current_panel == self.sta_panel:
            self.sta_panel.update_view(cluster_id)

    def _draw_plots(self, cluster_id, features):
        """Only update what's actually visible."""

        current_tab = self.analysis_tabs.currentWidget()

        # --- TIER 2: POPULATION RF (full rebuild) ---
        # Only when hot-swap wasn't possible in Tier 1
        if self.population_view_enabled:
            canvas = self.pop_mosaic_canvas
            can_hot_swap = (
                hasattr(canvas, '_pop_plot_state') and
                canvas._pop_plot_state.get('ax') in canvas.fig.axes
            )
            if not can_hot_swap:
                # Full rebuild needed - do it in Tier 2
                try:
                    draw_population_rfs_plot(
                        main_window=self,
                        selected_cell_id=cluster_id,
                        canvas=self.pop_mosaic_canvas
                    )
                except Exception as e:
                    logger.error(f"Tier 2 Pop Split rebuild failed: {e}")

        # --- ONLY UPDATE STANDARD PLOTS WHEN THAT TAB IS VISIBLE ---
        if current_tab == self.standard_plots_panel:
            self.standard_plots_panel.update_all(cluster_id)
            self.similarity_panel.update_main_cluster_id(cluster_id)

        # --- UPDATE ONLY THE ACTIVE TAB ---
        if current_tab == self.ei_panel:
            try:
                self.ei_panel.update_ei([cluster_id])
            except Exception as e:
                logger.error(f"Draw plots EI update failed: {e}")
            self.similarity_panel.update_main_cluster_id(cluster_id)

        elif current_tab == self.waveforms_panel:
            self.waveforms_panel.update_all(cluster_id)

        elif current_tab == self.raw_panel:
            self.raw_panel.load_data(cluster_id)

        elif current_tab == self.sta_panel:
            # STA tab - update single-cell or population view
            if self.current_sta_view == 'population_rfs':
                # Population RF in STA tab - full rebuild in Tier 2
                try:
                    draw_population_rfs_plot(
                        main_window=self,
                        selected_cell_id=cluster_id,
                        canvas=self.sta_panel.rf_canvas
                    )
                except Exception as e:
                    logger.error(f"Tier 2 STA Pop RF rebuild failed: {e}")
            else:
                # Single-cell STA view
                self.sta_panel.update_view(cluster_id)

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
        pop_layout = QVBoxLayout(self.pop_context_widget)
        pop_layout.setContentsMargins(4, 4, 4, 4)
        pop_layout.setSpacing(6)

        # Top Control Bar (with Expand Button)
        pop_ctrl_layout = QHBoxLayout()
        self.pop_expand_btn = QPushButton("⛶ Full Screen")
        self.pop_expand_btn.setToolTip("Toggle Full Screen Population View")
        self.pop_expand_btn.setCheckable(True)
        self.pop_expand_btn.clicked.connect(self.toggle_population_fullscreen)
        self.pop_expand_btn.setStyleSheet("font-weight: bold; background-color: #4282DA; padding: 4px 10px;")
        pop_ctrl_layout.addStretch()
        pop_ctrl_layout.addWidget(self.pop_expand_btn)
        pop_layout.addLayout(pop_ctrl_layout)

        # Master Vertical Splitter for all 3 canvases (makes them height-adjustable)
        self.pop_master_splitter = QSplitter(Qt.Orientation.Vertical)

        # 1. RF Mosaic Panel
        self.pop_mosaic_widget = QWidget()
        mosaic_layout = QVBoxLayout(self.pop_mosaic_widget)
        mosaic_layout.setContentsMargins(0, 0, 0, 0)
        self.pop_mosaic_canvas = MplCanvas(width=6, height=4, dpi=100)
        mosaic_layout.addWidget(self.pop_mosaic_canvas)
        self.pop_master_splitter.addWidget(self.pop_mosaic_widget)

        # 2. Timecourse Panel
        self.pop_timecourse_widget = QWidget()
        tc_layout = QVBoxLayout(self.pop_timecourse_widget)
        tc_layout.setContentsMargins(0, 0, 0, 0)
        tc_hdr = QHBoxLayout()
        tc_label = QLabel("Population Dynamics")
        tc_label.setStyleSheet("font-weight:bold; color: white;")
        self.pop_timecourse_summary = QLabel("n=0  mean_t2p: N/A  mean_fwhm: N/A")
        tc_hdr.addWidget(tc_label)
        tc_hdr.addStretch()
        tc_hdr.addWidget(self.pop_timecourse_summary)
        tc_layout.addLayout(tc_hdr)
        self.pop_timecourse_canvas = MplCanvas(width=6, height=2, dpi=100)
        tc_layout.addWidget(self.pop_timecourse_canvas)
        self.pop_master_splitter.addWidget(self.pop_timecourse_widget)

        # 3. ACG Panel
        self.pop_acg_widget = QWidget()
        acg_layout = QVBoxLayout(self.pop_acg_widget)
        acg_layout.setContentsMargins(0, 0, 0, 0)
        acg_hdr = QHBoxLayout()
        acg_label = QLabel("Population Autocorrelation")
        acg_label.setStyleSheet("font-weight:bold; color: white;")
        self.pop_acg_summary = QLabel("n=0")
        acg_hdr.addWidget(acg_label)
        acg_hdr.addStretch()
        acg_hdr.addWidget(self.pop_acg_summary)
        acg_layout.addLayout(acg_hdr)
        self.pop_acg_canvas = MplCanvas(width=6, height=2, dpi=100)
        acg_layout.addWidget(self.pop_acg_canvas)
        self.pop_master_splitter.addWidget(self.pop_acg_widget)

        # Add master splitter to layout
        pop_layout.addWidget(self.pop_master_splitter, stretch=1)
        self.pop_master_splitter.setSizes([400, 200, 200])

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
        self.sta_panel = STAPanel(self)
        self.umap_panel = UMAPPanel(self)

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

        self.cache_progress = QProgressBar()
        self.cache_progress.setMaximumWidth(250)
        self.cache_progress.setFormat("Pre-computing Physics Cache: %p%")
        self.cache_progress.hide()
        self.status_bar.addPermanentWidget(self.cache_progress)
        self.cache_progress_count = 0

        # --- Menu Bar ---
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        
        load_ks_action = file_menu.addAction("&Load Kilosort Directory...")
        
        # NEW: Separate Raw Data Loader
        self.load_raw_action = file_menu.addAction("Load &Raw Data File...")
        self.load_raw_action.setEnabled(False) # Disabled until KS is loaded
        
        self.load_vision_action = file_menu.addAction("&Load Vision Files...")
        self.load_vision_action.setEnabled(False)
        
        self.load_classification_action = file_menu.addAction("&Load Classification File...")
        self.load_classification_action.setEnabled(False)
        
        self.save_classification_action = file_menu.addAction("Save Classification Text File...")
        self.save_classification_action.setEnabled(False)

        self.save_action = file_menu.addAction("&Save Results...")
        self.save_action.setEnabled(False)

        # --- Array Menu ---
        array_menu = menu.addMenu("&Array")
        self.calibrate_array_action = array_menu.addAction("Map Image to Array...")
        self.calibrate_array_action.setEnabled(False)  # Enabled after data loads
        self.calibrate_array_action.triggered.connect(self._open_array_calibration)

        # Connect Signals
        load_ks_action.triggered.connect(lambda: self.load_directory())
        self.load_raw_action.triggered.connect(self.load_raw_data_file)
        self.load_vision_action.triggered.connect(self.load_vision_directory)
        self.load_classification_action.triggered.connect(self.load_classification_file)
        self.save_classification_action.triggered.connect(self.on_save_classification_action)
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

    def _open_array_calibration(self):
        """Open the Map Image to Array calibration dialog."""
        dlg = ArrayCalibrationDialog(self, data_manager=self.data_manager)
        dlg.transformSaved.connect(self._on_transform_saved)
        dlg.show()

    def _on_transform_saved(self, transform_path: str):
        """Called after user saves a new transform — refresh image overlay."""
        if hasattr(self, 'standard_plots_panel'):
            self.standard_plots_panel.refresh_array_image(transform_path=transform_path)
            # re-draw the grid for the currently selected cluster (if any)
            cid = self._get_selected_cluster_id()
            if cid is not None:
                self.standard_plots_panel.update_all(cid)
        self.status_bar.showMessage(f"Array transform saved: {transform_path}", 4000)

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
            draw_population_rfs_plot(
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
        """Recursively gets all cluster IDs from a folder and all its sub-folders."""
        cluster_ids = []
        def recurse(node):
            for i in range(node.rowCount()):
                child = node.child(i)
                if not child:
                    continue
                cid = child.data(Qt.ItemDataRole.UserRole)
                if cid is not None:
                    cluster_ids.append(cid)
                # Keep digging if it's a sub-folder
                if child.hasChildren():
                    recurse(child)
        recurse(item)
        return cluster_ids

    def _get_pop_subset_ids(self):
        """
        Gets the list of cluster IDs for the currently selected population subset.
        Uses the visual Tree Model to ensure perfect matching with the GUI state.
        """
        cluster_id = self._get_selected_cluster_id()

        # Case 1: A group/folder is selected in the Tree View
        if cluster_id is None and self.view_stack.currentIndex() == 0:
            selection = self.tree_view.selectionModel().selectedIndexes()
            if selection:
                index = selection[0]
                item = self.tree_model.itemFromIndex(index)
                if item and item.data(Qt.ItemDataRole.UserRole) is None:  # It's a group
                    return self._get_group_cluster_ids(item)

        # Case 2: A single cell is selected. Find its immediate parent folder.
        if cluster_id is not None:
            # Always trust the Tree Model first, as it perfectly reflects nested folders
            model = self.tree_model
            matches = model.match(model.index(0, 0), Qt.ItemDataRole.UserRole, cluster_id, 1, Qt.MatchExactly | Qt.MatchRecursive)
            
            if matches:
                item = model.itemFromIndex(matches[0])
                parent_item = item.parent()
                if parent_item is None:
                    parent_item = model.invisibleRootItem()
                return self._get_group_cluster_ids(parent_item)
                
            return [cluster_id]  # Fallback

        return []

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

    # --- Methods to bridge UI signals to callback functions ---
    def load_directory(self, kilosort_dir=None, dat_file=None):
        callbacks.load_directory(self, kilosort_dir, dat_file)

    def load_vision_directory(self):
        callbacks.load_vision_directory(self)

    def load_raw_data_file(self):
        callbacks.load_raw_data(self)

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
                # Use Qt's highly optimized, built-in recursive match
                start_index = self.tree_model.index(0, 0)
                matches = self.tree_model.match(
                    start_index,
                    Qt.ItemDataRole.UserRole,           # What role to search (Cluster ID)
                    cluster_id,                         # What value to look for
                    1,                                  # Stop after 1 match is found
                    Qt.MatchExactly | Qt.MatchRecursive # Tell it to search sub-folders
                )
                
                if matches:
                    index = matches[0]
                    self.tree_view.selectionModel().select(
                        index, QItemSelectionModel.ClearAndSelect)
                    self.tree_view.scrollTo(
                        index, QAbstractItemView.ScrollHint.PositionAtCenter)

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

        # Update button text based on current view
        if hasattr(self.sta_panel, 'sta_animation_button'):  # Check if STAPanel is initialized
            if view_type == "rf":
                self.sta_panel.sta_animation_button.setText("Play Animation")
            elif view_type == "animation":
                self.sta_panel.sta_animation_button.setText("Pause Animation")
            elif view_type == "population_rfs":
                self.sta_panel.sta_animation_button.setText("Play Animation")

        # Delegate to the STAPanel
        cluster_id = self._get_selected_cluster_id()
        if cluster_id is not None:
            self.sta_panel.update_view(cluster_id)

    def update_sta_frame_manual(self, frame_index):
        """Updates the STA visualization to a specific frame manually."""
        # Delegate to the STAPanel
        self.sta_panel.update_sta_frame_manual(frame_index)

    def _advance_frame_internal(self):
        """Internal method for the timer to call without stopping itself."""
        # Delegate to the STAPanel
        self.sta_panel._advance_frame_internal()

    def prev_sta_frame(self):
        """Go to the previous frame in the STA animation."""
        # Delegate to the STAPanel
        self.sta_panel.prev_sta_frame()

    def next_sta_frame(self):
        """Go to the next frame in the STA animation."""
        # Delegate to the STAPanel
        self.sta_panel.next_sta_frame()

    def load_classification_file(self):
        callbacks.load_classification_file(self)

    def open_tree_context_menu(self, position):
        menu = QMenu()
        index = self.tree_view.indexAt(position)
        item = self.tree_model.itemFromIndex(index)
        if not item: return

        add_group_action = menu.addAction("Add New Group")

        # Only show folder options if clicking a folder (hasChildren or no UserRole)
        if item.hasChildren() or item.data(Qt.ItemDataRole.UserRole) is None:  
            rename_action = menu.addAction("Rename")
            feature_extraction_action = menu.addAction("Feature Extraction")
            menu.addSeparator()
            flatten_action = menu.addAction("Flatten Group (Remove Sub-folders)")
            delete_action = menu.addAction("Delete Group (Keep Units)")

        action = menu.exec(self.tree_view.viewport().mapToGlobal(position))

        if action == add_group_action:
            text, ok = QInputDialog.getText(
                self, 'New Group', 'Enter group name:')
            if ok and text:
                callbacks.add_new_group(self, text, parent_item=item)
                
        elif item.hasChildren() or item.data(Qt.ItemDataRole.UserRole) is None:  
            if action == rename_action:
                new_group_name, ok = QInputDialog.getText(
                    self, 'Rename Group', 'Enter group name:', text=item.text())
                if ok and new_group_name:
                    original_group_name = item.text()
                    callbacks.rename_class(self, original_group_name, new_group_name)
            elif action == feature_extraction_action:
                cluster_ids = self._get_group_cluster_ids(item)
                callbacks.feature_extraction(self, cluster_ids)
            elif action == flatten_action:
                callbacks.flatten_group(self, item)
            elif action == delete_action:
                callbacks.delete_group(self, item)

    def toggle_animation(self):
        """Toggle the animation between play and pause."""
        # Delegate to the STAPanel
        self.sta_panel.toggle_animation()

    def on_rf_canvas_clicked(self):
        """Handle clicks on the RF canvas in STA tab - toggle between RF and animation."""
        # Delegate to the STAPanel
        self.sta_panel.on_rf_canvas_clicked()

    def stop_animation(self):
        """Stop the animation completely."""
        # Delegate to the STAPanel
        self.sta_panel.stop_animation()

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

    def toggle_population_fullscreen(self, checked):
        """Toggles the Population panel to take up 100% of the right pane."""
        if checked:
            # Full screen mode: Collapse the main tabs completely
            self.right_splitter.setSizes([0, 1000])
            self.pop_expand_btn.setText("🗗 Restore")
            self.pop_expand_btn.setStyleSheet("font-weight: bold; background-color: #2D6A4F; padding: 4px 10px;")
        else:
            # Restore mode: 75/25 split
            total = sum(self.right_splitter.sizes()) or 1400
            left_size = max(int(total * 0.75), 400)
            right_size = total - left_size
            self.right_splitter.setSizes([left_size, right_size])
            self.pop_expand_btn.setText("⛶ Full Screen")
            self.pop_expand_btn.setStyleSheet("font-weight: bold; background-color: #4282DA; padding: 4px 10px;")

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

        # Stop background workers first
        callbacks.stop_worker(self)

        # Cleanup all threads with timeout to prevent hangs
        for thread_attr in [
            'feature_worker_thread', 'worker_thread',
            'standard_worker_thread', 'ks_load_thread',
            'vision_load_thread', 'refine_thread'
        ]:
            self._cleanup_thread(thread_attr, timeout_ms=1000)

        # Stop any running raw trace worker
        if hasattr(self, 'raw_panel') and self.raw_panel.worker_thread and self.raw_panel.worker_thread.isRunning():
            self.raw_panel.worker_thread.stop()
            self.raw_panel.worker_thread.wait()

        # Close any open PyBinFileReader file handles to avoid leaking OS-level
        # file descriptors.  _close_raw_reader() is a no-op when raw_reader is
        # already None, so it is always safe to call.
        if self.data_manager is not None:
            self.data_manager._close_raw_reader()

        # --- NEW: Automatic Temp File Cleanup Sweep ---
        try:
            if self.data_manager and self.data_manager.kilosort_dir:
                import glob
                import os
                ks_dir = str(self.data_manager.kilosort_dir)
                # Find all .tmp files in the Kilosort directory
                tmp_files = glob.glob(os.path.join(ks_dir, '*.tmp'))
                for tmp_file in tmp_files:
                    try:
                        os.remove(tmp_file)
                        logging.info(f"Cleaned up orphaned temp file: {tmp_file}")
                    except OSError:
                        pass # File might be locked, just skip it
        except Exception as e:
            logging.error(f"Error during temp file cleanup: {e}")
        # ----------------------------------------------
        
        event.accept()