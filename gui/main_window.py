import os
from qtpy.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSplitter, QStatusBar,
    QHeaderView, QMessageBox, QTabWidget,
    QTreeView, QAbstractItemView, QSlider, QLabel, 
    QMenu, QInputDialog, QStackedWidget, QLineEdit,
    QApplication
)
from qtpy.QtCore import Qt, QItemSelectionModel, QThread, QTimer
from qtpy.QtGui import QFont, QStandardItemModel
import pyqtgraph as pg
from analysis import analysis_core
from analysis.data_manager import DataManager
from typing import Optional
# Custom GUI Modules
from gui.widgets import MplCanvas, HighlightStatusPandasModel, CustomTableView
import gui.callbacks as callbacks
import gui.plotting as plotting
from gui.panels.similarity_panel import SimilarityPanel
from gui.panels.waveforms_panel import WaveformPanel
from gui.panels.ei_panel import EIPanel
from gui.panels.raw_panel import RawPanel
from gui.workers import FeatureWorker
from gui.shortcuts import KeyForwarder
from PyQt5.QtGui import QColor

# Global pyqtgraph configuration
pg.setConfigOption('background', '#1f1f1f')
pg.setConfigOption('foreground', 'd')


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
        self.worker_thread = None
        self.spatial_worker = None
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
        self.sta_animation_timer = None
        self._is_syncing = False
        self.last_left_width = 450
        self.feature_worker_thread = None

        # --- UI Setup ---
        self._setup_style()
        self._setup_ui()
        self.central_widget.setEnabled(False)
        self.status_bar.showMessage("Welcome to axolotl. Please load a Kilosort directory to begin.")

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
                # Select first leaf if nothing is selected or selection is not a leaf
                sel_model.select(leaf_indices[0], QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
                view.scrollTo(leaf_indices[0])
                return
            current_idx = leaf_indices.index(selected[0])
            if key == Qt.Key_Up:
                new_idx = max(0, current_idx - 1)
            else:
                new_idx = min(len(leaf_indices) - 1, current_idx + 1)
            sel_model.select(leaf_indices[new_idx], QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
            view.scrollTo(leaf_indices[new_idx])
        else:
            # Table view logic
            current = view.currentIndex()
            if not current.isValid():
                # Select first row if nothing is selected
                index = model.index(0, 0)
                sel_model.setCurrentIndex(index, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
                view.scrollTo(index)
                return
            current_row = current.row()
            if key == Qt.Key_Up:
                new_row = max(0, current_row - 1)
            else:
                new_row = min(model.rowCount() - 1, current_row + 1)
            index = model.index(new_row, 0)
            sel_model.setCurrentIndex(index, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
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

# In main_window.py, add these new methods to the MainWindow class.
# Also, add 'self.feature_worker_thread = None' to the __init__ method.
# Make sure to add `from gui.workers import FeatureWorker` to your imports.

    # In MainWindow, ensure you have these two methods.
    # update_cluster_views catches the rapid clicks.
    def update_cluster_views(self, cluster_id):
        """
        Receives a selection event, stores the cluster_id, and restarts the
        selection timer. This 'debounces' rapid selections.
        """
        self._pending_cluster_id = cluster_id
        self.selection_timer.start()

    # _process_selection runs only after the user pauses.
    # In gui/main_window.py

    def _process_selection(self):
        """
        This method is called by the timer ONLY after the user has stopped
        scrolling. It performs the actual data loading for the last selected cluster.
        """
        cluster_id = self._pending_cluster_id
        if cluster_id is None:
            return

        self.status_bar.showMessage(f"Loading data for Cluster ID: {cluster_id}...")
        
        cached_features = self.data_manager.get_lightweight_features(cluster_id)
        if cached_features:
            self._draw_plots(cluster_id, cached_features)
            return

        # Only run FeatureWorker if dat_path is available
        if self.data_manager.dat_path is not None:
            # --- FIX: Ensure the previous worker is fully terminated before starting a new one.
            if self.feature_worker_thread and self.feature_worker_thread.isRunning():
                self.feature_worker_thread.quit()
                self.feature_worker_thread.wait() # This is the critical addition

            self.feature_worker_thread = QThread()
            self.feature_worker = FeatureWorker(self.data_manager, cluster_id)
            self.feature_worker.moveToThread(self.feature_worker_thread)
            self.feature_worker.features_ready.connect(self.on_features_ready)
            self.feature_worker.error.connect(lambda msg: self.status_bar.showMessage(msg, 4000))
            self.feature_worker_thread.started.connect(self.feature_worker.run)
            self.feature_worker_thread.start()
        else:
            self.status_bar.showMessage("Raw data file not loaded: waveform plot disabled.", 4000)
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

    def _draw_plots(self, cluster_id, features):
        """A single, centralized function to update all plots."""
        # Update the standard plots.
        # if features is not None:
            # self.waveforms_panel.update_waveforms(cluster_id, {cluster_id: features})
        # else:
            # self.waveforms_panel.clear()
            # self.waveforms_panel.waveform_plot.setTitle("Waveforms (Raw data not loaded)")
        
        self.similarity_panel.update_main_cluster_id(cluster_id)
        # Select the top row in similarity panel by default. This updates waveforms and EI panel.
        self.similarity_panel.select_top_n_rows(1)
        
        # self.waveforms_panel.update_all(cluster_id)
        # self.ei_panel.update_ei(cluster_id)

        # Update the tab-specific plot (Raw Trace).
        if self.analysis_tabs.currentWidget() == self.raw_panel:
            self.raw_panel.load_data(cluster_id)
        else:
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
        self.tree_view_button.clicked.connect(lambda: self._switch_left_view(0))
        self.table_view_button.clicked.connect(lambda: self._switch_left_view(1))
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
        self.tree_view.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.tree_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.open_tree_context_menu)
        
        # Table View
        self.table_view = CustomTableView()
        self.table_view.setSortingEnabled(True)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

        self.view_stack.addWidget(self.tree_view)
        self.view_stack.addWidget(self.table_view)
        # Default to table view
        self.view_stack.setCurrentIndex(1)

        self.refine_button = QPushButton("Refine Selected Cluster")
        self.refine_button.setFixedHeight(40)
        self.refine_button.setStyleSheet("font-size: 14px; font-weight: bold; color: #aeffe3; background-color: #005230;")

        left_content_layout.addLayout(filter_box)
        left_content_layout.addLayout(view_switch_layout)
        left_content_layout.addWidget(self.view_stack)
        left_content_layout.addWidget(self.refine_button)

        # --- Similarity Panel ---
        self.similarity_panel = SimilarityPanel(self)
        left_content_layout.addWidget(self.similarity_panel)
        self.similarity_panel.selection_changed.connect(self.on_similarity_selection_changed)

        # Add the toggle button and content to the left pane
        left_layout.addWidget(self.sidebar_toggle_button)
        left_layout.addWidget(left_content)
        # Store reference to content widget for collapsing/expanding
        self.left_content = left_content

        # --- Right Pane ---
        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # --- Panels ---
        self.waveforms_panel = WaveformPanel(self)
        self.ei_panel = EIPanel(self)
        self.raw_panel = RawPanel(self)

        # --- STA Analysis Panel ---
        self.sta_panel = QWidget()
        sta_layout = QVBoxLayout(self.sta_panel)
        
        # Add buttons to select different STA views
        sta_control_layout = QHBoxLayout()
        self.sta_rf_button = QPushButton("RF Plot")
        self.sta_population_rfs_button = QPushButton("Population RFs")
        self.sta_timecourse_button = QPushButton("Timecourse")
        self.sta_animation_button = QPushButton("Animate STA")
        self.sta_animation_stop_button = QPushButton("Stop Animation")
        
        # Set up button functionality
        self.sta_rf_button.clicked.connect(lambda: self.select_sta_view("rf"))
        self.sta_population_rfs_button.clicked.connect(lambda: self.select_sta_view("population_rfs"))
        self.sta_timecourse_button.clicked.connect(lambda: self.select_sta_view("timecourse"))
        self.sta_animation_button.clicked.connect(lambda: self.select_sta_view("animation"))
        self.sta_animation_stop_button.clicked.connect(lambda: plotting.stop_sta_animation(self))
        
        # Add buttons to layout
        sta_control_layout.addWidget(self.sta_rf_button)
        sta_control_layout.addWidget(self.sta_population_rfs_button)
        sta_control_layout.addWidget(self.sta_timecourse_button)
        sta_control_layout.addWidget(self.sta_animation_button)
        sta_control_layout.addWidget(self.sta_animation_stop_button)
        
        # Add frame control elements
        sta_frame_layout = QHBoxLayout()
        self.sta_frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.sta_frame_slider.setMinimum(0)
        self.sta_frame_slider.setMaximum(29)  # Default to 30 frames
        self.sta_frame_slider.setValue(0)
        self.sta_frame_slider.setEnabled(False)  # Only enabled during manual animation
        self.sta_frame_label = QLabel("Frame: 0/30")
        self.sta_frame_prev_button = QPushButton("<< Prev")
        self.sta_frame_next_button = QPushButton("Next >>")
        
        # Connect slider and buttons
        self.sta_frame_slider.valueChanged.connect(self.update_sta_frame_manual)
        self.sta_frame_prev_button.clicked.connect(self.prev_sta_frame)
        self.sta_frame_next_button.clicked.connect(self.next_sta_frame)
        
        # Add frame controls to layout
        sta_frame_layout.addWidget(self.sta_frame_prev_button)
        sta_frame_layout.addWidget(self.sta_frame_slider)
        sta_frame_layout.addWidget(self.sta_frame_next_button)
        sta_frame_layout.addWidget(self.sta_frame_label)
        sta_control_layout.addLayout(sta_frame_layout)
        
        # Add controls and canvas to layout
        sta_layout.addLayout(sta_control_layout)
        self.sta_canvas = MplCanvas(self, width=10, height=8, dpi=120)
        sta_layout.addWidget(self.sta_canvas)

        # --- Top Splitter: Waveforms and STA side by side ---
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.addWidget(self.waveforms_panel)
        top_splitter.addWidget(self.sta_panel)
        top_splitter.setSizes([800, 200])

        # --- Main Right Splitter: Top (waveforms+STA), Bottom (EI spatial and temporal) ---
        main_right_splitter = QSplitter(Qt.Orientation.Vertical)
        main_right_splitter.addWidget(top_splitter)
        main_right_splitter.addWidget(self.ei_panel)
        main_right_splitter.setSizes([600, 400])

        # --- Tab Widget for Raw Trace ---
        self.analysis_tabs = QTabWidget()
        self.analysis_tabs.addTab(main_right_splitter, "Main Analysis")
        self.analysis_tabs.addTab(self.raw_panel, "Raw Trace")
        right_layout.addWidget(self.analysis_tabs)

        # --- Main Splitter and Layout ---
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.addWidget(self.left_pane)
        self.main_splitter.addWidget(right_pane)
        self.main_splitter.setSizes([800, 600])
        main_layout.addWidget(self.main_splitter)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # --- Menu Bar ---
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        load_ks_action = file_menu.addAction("&Load Kilosort Directory...")
        self.load_vision_action = file_menu.addAction("&Load Vision Files...")
        self.load_vision_action.setEnabled(False)
        self.load_classification_action = file_menu.addAction("&Load Classification File...")
        self.load_classification_action.setEnabled(False)
        self.save_action = file_menu.addAction("&Save Results...")
        self.save_action.setEnabled(False)
        
        # Connect Signals to Callback Functions ---
        load_ks_action.triggered.connect(self.load_directory)
        self.load_vision_action.triggered.connect(self.load_vision_directory)
        self.load_classification_action.triggered.connect(self.load_classification_file)
        self.save_action.triggered.connect(self.on_save_action)
        self.filter_button.clicked.connect(self.apply_good_filter)
        self.reset_button.clicked.connect(self.reset_views)
        self.refine_button.clicked.connect(self.on_refine_cluster)

        # Connect the raw panel's status and error messages to the status bar
        self.raw_panel.status_message.connect(lambda msg: self.status_bar.showMessage(msg, 3000))
        self.raw_panel.error_message.connect(lambda msg: self.status_bar.showMessage(msg, 4000))


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
            
            # Only leaf nodes (cells) have a cluster ID stored. Groups will return None.
            cluster_id = item.data(Qt.ItemDataRole.UserRole)
            return cluster_id

        # Case 2: Table View is active
        elif current_view_index == 1:
            if not self.table_view.selectionModel().hasSelection() or self.main_cluster_model is None:
                return None
            
            selected_row = self.table_view.selectionModel().selectedIndexes()[0].row()
            
            # Check if the model has mapToSource method (for proxy models)
            model = self.table_view.model()
            if hasattr(model, 'mapToSource'):
                # The pandas model can be sorted, so we must map the view's row to the model's row
                source_index = model.mapToSource(model.index(selected_row, 0))
                cluster_id = model._dataframe.iloc[source_index.row()]['cluster_id']
            else:
                # If no proxy model, use the row directly
                cluster_id = model._dataframe.iloc[selected_row]['cluster_id']
            return cluster_id
        
        return None

    def setup_tree_model(self, model):
        """Sets up the tree view model and connects the selection changed signal."""
        self.tree_view.setModel(model)
        try:
            self.tree_view.selectionModel().selectionChanged.disconnect(self.on_view_selection_changed)
        except (TypeError, RuntimeError):
            pass
        self.tree_view.selectionModel().selectionChanged.connect(self.on_view_selection_changed)

    def setup_table_model(self, model):
        """Sets up the table view model and connects the selection changed signal."""
        self.table_view.setModel(model)
        try:
            self.table_view.selectionModel().selectionChanged.disconnect(self.on_view_selection_changed)
        except (TypeError, RuntimeError):
            pass
        self.table_view.selectionModel().selectionChanged.connect(self.on_view_selection_changed)
        
    # --- Methods to bridge UI signals to callback functions ---
    def load_directory(self, kilosort_dir=None, dat_file=None):
        callbacks.load_directory(self, kilosort_dir, dat_file)

    def load_vision_directory(self):
        callbacks.load_vision_directory(self)

    def on_view_selection_changed(self, selected, deselected):
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
                        row_indices = df.index[df['cluster_id'] == cluster_id].tolist()
                        if row_indices:
                            model_row = df.index.get_loc(row_indices[0])
                            source_index = model.index(model_row, 0)
                            # This assumes the model is a proxy model if sorting is enabled
                            view_index = model.mapFromSource(source_index) if hasattr(model, 'mapFromSource') else source_index
                            if view_index.isValid():
                                self.table_view.selectionModel().select(view_index, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
                                self.table_view.scrollTo(view_index, QAbstractItemView.ScrollHint.PositionAtCenter)

            # Sync from Table to Tree
            elif sender == self.table_view.selectionModel():
                for row in range(self.tree_model.rowCount()):
                    group_item = self.tree_model.item(row)
                    if not group_item: continue
                    for child_row in range(group_item.rowCount()):
                        child_item = group_item.child(child_row)
                        if child_item and child_item.data(Qt.ItemDataRole.UserRole) == cluster_id:
                            index = self.tree_model.indexFromItem(child_item)
                            self.tree_view.selectionModel().select(index, QItemSelectionModel.ClearAndSelect)
                            self.tree_view.scrollTo(index, QAbstractItemView.ScrollHint.PositionAtCenter)
                            break
                    else:
                        continue
                    break
        
        # Now that views are synced, trigger the update callbacks
        callbacks.on_cluster_selection_changed(self)
        self._is_syncing = False

        self.similarity_panel.reset_spacebar_counter()

    def on_similarity_selection_changed(self, selected_cluster_ids):
        # Always include the main selected cluster
        main_cluster = self._get_selected_cluster_id()
        clusters_to_plot = [main_cluster] + selected_cluster_ids
        print(f'[DEBUG] on_similarity_selection_changed: main_cluster = {main_cluster}')
        print(f'[DEBUG] on_similarity_selection_changed: clusters_to_plot = {clusters_to_plot}')

        self.ei_panel.update_ei(clusters_to_plot)
        # self.waveforms_panel.update_all(clusters_to_plot)

    def _update_table_view_duplicate_highlight(self):
        df = self.data_manager.cluster_df
        self.main_cluster_model = HighlightStatusPandasModel(df)
        self.setup_table_model(self.main_cluster_model)

    def _update_tree_view_duplicate_highlight(self):
        # Collect all duplicate IDs
        sdf = self.data_manager.status_df
        duplicate_ids = sdf[sdf['status'] == 'Duplicate']['cluster_id'].tolist()
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

    def select_sta_view(self, view_type):
        """Select the STA view to display."""
        self.current_sta_view = view_type
        cluster_id = self._get_selected_cluster_id()
        if cluster_id is None:
            return
        
        # Call the appropriate plotting function based on the selected view
        if view_type == "rf":
            plotting.draw_sta_plot(self, cluster_id)
        elif view_type == "population_rfs":
            print(f"--- 1. DEBUG (MainWindow): Got selected_cell_id = {cluster_id}. Passing to plotting function. ---")
            plotting.draw_population_rfs_plot(self, selected_cell_id=cluster_id)
        elif view_type == "timecourse":
            plotting.draw_sta_timecourse_plot(self, cluster_id)
        elif view_type == "animation":
            plotting.draw_sta_animation_plot(self, cluster_id)

    def update_sta_frame_manual(self, frame_index):
        """Updates the STA visualization to a specific frame manually."""
        if hasattr(self, 'current_sta_data') and self.current_sta_data is not None:
            # Stop any running animation
            plotting.stop_sta_animation(self)
            
            # Update the frame index
            self.current_frame_index = frame_index
            
            # Update the label
            self.sta_frame_label.setText(f"Frame: {frame_index+1}/{self.total_sta_frames}")
            
            # Update the STA canvas with the new frame
            self.sta_canvas.fig.clear()
            analysis_core.animate_sta_movie(
                self.sta_canvas.fig,
                self.current_sta_data,
                frame_index=frame_index,
                sta_width=self.data_manager.vision_sta_width,
                sta_height=self.data_manager.vision_sta_height
            )
            cluster_id = self.current_sta_cluster_id - 1  # Convert back to 0-indexed
            self.sta_canvas.fig.suptitle(f"Cluster {cluster_id} - STA Frame {frame_index+1}/{self.total_sta_frames}", color='white', fontsize=16)
            self.sta_canvas.draw()

    def prev_sta_frame(self):
        """Go to the previous frame in the STA animation."""
        if hasattr(self, 'current_sta_data') and self.current_sta_data is not None:
            plotting.stop_sta_animation(self)
            self.current_frame_index = (self.current_frame_index - 1) % self.total_sta_frames
            self.sta_frame_slider.setValue(self.current_frame_index)
            self.sta_frame_label.setText(f"Frame: {self.current_frame_index+1}/{self.total_sta_frames}")
            self.sta_canvas.fig.clear()
            analysis_core.animate_sta_movie(
                self.sta_canvas.fig,
                self.current_sta_data,
                stafit=self.current_stafit, # <-- Pass the stored fit
                frame_index=self.current_frame_index,
                sta_width=self.data_manager.vision_sta_width,
                sta_height=self.data_manager.vision_sta_height
            )
            self.sta_canvas.draw()

    def next_sta_frame(self):
        """Go to the next frame in the STA animation."""
        if hasattr(self, 'current_sta_data') and self.current_sta_data is not None:
            plotting.stop_sta_animation(self)
            self.current_frame_index = (self.current_frame_index + 1) % self.total_sta_frames
            self.sta_frame_slider.setValue(self.current_frame_index)
            self.sta_frame_label.setText(f"Frame: {self.current_frame_index+1}/{self.total_sta_frames}")
            self.sta_canvas.fig.clear()
            analysis_core.animate_sta_movie(
                self.sta_canvas.fig,
                self.current_sta_data,
                stafit=self.current_stafit, # <-- Pass the stored fit
                frame_index=self.current_frame_index,
                sta_width=self.data_manager.vision_sta_width,
                sta_height=self.data_manager.vision_sta_height
            )
            self.sta_canvas.draw()
    
    def load_classification_file(self):
        callbacks.load_classification_file(self)
        
    def open_tree_context_menu(self, position):
        menu = QMenu()
        add_group_action = menu.addAction("Add New Group")
        
        action = menu.exec(self.tree_view.viewport().mapToGlobal(position))
        
        if action == add_group_action:
            text, ok = QInputDialog.getText(self, 'New Group', 'Enter group name:')
            if ok and text:
                callbacks.add_new_group(self, text)
    
    def toggle_sidebar(self):
        """Collapses or expands the left sidebar by manipulating the main splitter."""
        if self.sidebar_collapsed:
            # --- EXPAND --- 
            self.sidebar_toggle_button.setText("◀")
            widths = self.main_splitter.sizes()
            total_width = sum(widths)
            self.main_splitter.setSizes([self.last_left_width, total_width - self.last_left_width])
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
            reply = QMessageBox.question(self, 'Unsaved Changes',
                "You have unsaved refinement changes. Do you want to save before exiting?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Save:
                self.on_save_action()
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
        callbacks.stop_worker(self)
        # Stop any running raw trace worker
        if hasattr(self, 'raw_panel') and self.raw_panel.worker_thread and self.raw_panel.worker_thread.isRunning():
            self.raw_panel.worker_thread.quit()
            self.raw_panel.worker_thread.wait()
        event.accept()
