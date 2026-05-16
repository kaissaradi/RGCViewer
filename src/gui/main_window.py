import os
import logging
from qtpy.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSplitter, QStatusBar,
    QHeaderView, QMessageBox, QTabWidget,
    QTreeView, QAbstractItemView, QSlider, QLabel,
    QMenu, QInputDialog, QStackedWidget, QApplication,
    QTextEdit, QCheckBox, QProgressBar, QButtonGroup, QFrame
)
from qtpy.QtCore import Qt, QItemSelectionModel, QThread, QTimer
from qtpy.QtGui import QFont, QStandardItemModel
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
from .theme import (
    DARK_COLORS,
    PANEL_PADDING,
    CTRL_SPACING,
    ROW_HEIGHT,
    configure_pyqtgraph_theme,
    get_theme_colors,
)
# Array calibration dialog
from .panels.array_calibration_panel import ArrayCalibrationDialog

# Global pyqtgraph configuration
configure_pyqtgraph_theme(DARK_COLORS)

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, default_kilosort_dir=None, default_dat_file=None):
        super().__init__()
        self.setWindowTitle("RGC Viewer")
        self.setGeometry(50, 50, 1800, 1000)

        # --- Application State ---
        self.theme = "dark"
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
        self._setup_style(DARK_COLORS)
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

    def _setup_style(self, colors):
        self.setFont(QFont("Inter", 11))

        self.setStyleSheet(f"""
            /* ── Base ───────────────────────────── */
            QWidget {{
                color: {colors['text_primary']};
                background-color: {colors['bg_base']};
                font-family: 'Inter', 'Segoe UI', sans-serif;
                font-size: 12px;
            }}
            QMainWindow, QDialog {{
                background-color: {colors['bg_base']};
            }}

            /* ── Splitter handles ────────────────── */
            QSplitter::handle {{
                background: {colors['border_subtle']};
            }}
            QSplitter::handle:horizontal {{
                width: 5px;
            }}
            QSplitter::handle:vertical {{
                height: 5px;
            }}
            QSplitter::handle:horizontal:hover,
            QSplitter::handle:vertical:hover {{
                background: {colors['accent_hover']};
            }}

            /* ── Tables ──────────────────────────── */
            QTableView {{
                background-color: {colors['bg_panel']};
                alternate-background-color: {colors['bg_surface']};
                color: {colors['text_primary']};
                gridline-color: transparent;
                border: none;
                selection-background-color: {colors['selection_bg']};
                selection-color: {colors['text_primary']};
            }}
            QTableView::item {{
                border-bottom: 1px solid {colors['border_subtle']};
                padding: 0 8px;
            }}
            QTableView::item:selected {{
                background-color: {colors['selection_bg']};
            }}
            QHeaderView::section {{
                background-color: {colors['bg_panel']};
                color: {colors['text_tertiary']};
                padding: 4px 8px;
                border: none;
                border-bottom: 1px solid {colors['border_default']};
                font-size: 10px;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}
            QHeaderView::section:hover {{
                background-color: {colors['bg_surface']};
                color: {colors['text_secondary']};
            }}
            QHeaderView {{
                background-color: {colors['bg_panel']};
            }}

            /* ── Buttons ─────────────────────────── */
            QPushButton {{
                background-color: transparent;
                border: 0.5px solid {colors['border_default']};
                color: {colors['text_secondary']};
                padding: 4px 10px;
                border-radius: 5px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {colors['bg_surface']};
                border-color: {colors['border_strong']};
                color: {colors['text_primary']};
            }}
            QPushButton:pressed {{
                background-color: {colors['bg_elevated']};
            }}
            QPushButton:checked {{
                background-color: {colors['status_unsort_bg']};
                border-color: {colors['accent']};
                color: {colors['accent_hover']};
            }}
            QPushButton:disabled {{
                color: {colors['text_disabled']};
                border-color: {colors['border_subtle']};
            }}

            /* ── Tabs ────────────────────────────── */
            QTabWidget::pane {{
                border: none;
                border-top: 1px solid {colors['border_subtle']};
            }}
            QTabBar::tab {{
                color: {colors['text_secondary']};
                background: transparent;
                padding: 6px 16px;
                font-size: 12px;
                border-bottom: 2px solid transparent;
                margin-right: 2px;
                min-width: 40px;
            }}
            QTabBar::tab:selected {{
                color: {colors['text_primary']};
                border-bottom: 2px solid {colors['accent_hover']};
            }}
            QTabBar::tab:hover:!selected {{
                color: {colors['text_primary']};
                background: {colors['bg_surface']};
            }}
            QTabBar::scroller {{
                width: 24px;
            }}

            /* ── Inputs ──────────────────────────── */
            QComboBox {{
                background-color: {colors['bg_panel']};
                border: 0.5px solid {colors['border_default']};
                border-radius: 4px;
                padding: 3px 8px;
                color: {colors['text_primary']};
                font-size: 12px;
                min-height: 22px;
            }}
            QComboBox:hover {{ border-color: {colors['border_strong']}; }}
            QComboBox::drop-down {{
                border: none;
                width: 18px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {colors['bg_surface']};
                border: 0.5px solid {colors['border_default']};
                selection-background-color: {colors['selection_bg_strong']};
                color: {colors['text_primary']};
            }}
            QDoubleSpinBox, QSpinBox {{
                background-color: {colors['bg_panel']};
                border: 0.5px solid {colors['border_default']};
                border-radius: 4px;
                padding: 3px 6px;
                color: {colors['text_primary']};
                font-size: 12px;
            }}
            QDoubleSpinBox:hover, QSpinBox:hover {{
                border-color: {colors['border_strong']};
            }}

            /* ── Checkboxes ──────────────────────── */
            QCheckBox {{
                color: {colors['text_secondary']};
                spacing: 5px;
                font-size: 12px;
            }}
            QCheckBox:hover {{ color: {colors['text_primary']}; }}
            QCheckBox::indicator {{
                width: 14px;
                height: 14px;
                border: 0.5px solid {colors['border_default']};
                border-radius: 3px;
                background: {colors['bg_panel']};
            }}
            QCheckBox::indicator:checked {{
                background: {colors['accent']};
                border-color: {colors['accent']};
            }}

            /* ── Radio buttons ───────────────────── */
            QRadioButton {{
                color: {colors['text_secondary']};
                spacing: 5px;
                font-size: 12px;
            }}
            QRadioButton:hover {{ color: {colors['text_primary']}; }}

            /* ── Labels ──────────────────────────── */
            QLabel {{
                color: {colors['text_secondary']};
                font-size: 12px;
            }}

            /* ── Scrollbars ──────────────────────── */
            QScrollBar:vertical {{
                background: {colors['bg_panel']};
                width: 6px;
                border-radius: 3px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: {colors['border_default']};
                border-radius: 3px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{ background: {colors['border_strong']}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QScrollBar:horizontal {{
                background: {colors['bg_panel']};
                height: 6px;
                border-radius: 3px;
                margin: 0;
            }}
            QScrollBar::handle:horizontal {{
                background: {colors['border_default']};
                border-radius: 3px;
                min-width: 20px;
            }}
            QScrollBar::handle:horizontal:hover {{ background: {colors['border_strong']}; }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

            /* ── Tree View ───────────────────────── */
            QTreeView {{
                background-color: {colors['bg_panel']};
                color: {colors['text_primary']};
                border: none;
                alternate-background-color: {colors['bg_surface']};
                selection-background-color: {colors['selection_bg']};
            }}
            QTreeView::item {{ color: {colors['text_primary']}; }}
            QTreeView::item:hover {{ background: {colors['bg_surface']}; color: {colors['text_primary']}; }}
            QTreeView::item:selected {{ background: {colors['selection_bg']}; color: {colors['text_primary']}; }}
            QTreeView::branch {{ background: {colors['bg_panel']}; }}

            /* ── Status bar ──────────────────────── */
            QStatusBar {{
                color: {colors['text_tertiary']};
                font-size: 11px;
                border-top: 0.5px solid {colors['border_subtle']};
                background: {colors['bg_base']};
                padding: 2px 8px;
            }}

            /* ── Menu bar ────────────────────────── */
            QMenuBar {{
                background-color: {colors['bg_base']};
                color: {colors['text_secondary']};
                border-bottom: 0.5px solid {colors['border_subtle']};
                font-size: 12px;
            }}
            QMenuBar::item:selected {{ background: {colors['bg_surface']}; color: {colors['text_primary']}; }}
            QMenu {{
                background-color: {colors['bg_surface']};
                border: 0.5px solid {colors['border_default']};
                color: {colors['text_primary']};
                font-size: 12px;
            }}
            QMenu::item:selected {{ background: {colors['selection_bg_strong']}; }}
            QMenu::separator {{
                height: 1px;
                background: {colors['border_subtle']};
                margin: 3px 0;
            }}

            /* ── Progress bar ────────────────────── */
            QProgressBar {{
                background-color: {colors['bg_panel']};
                border: 0.5px solid {colors['border_default']};
                border-radius: 4px;
                text-align: center;
                color: {colors['text_secondary']};
                font-size: 11px;
                height: 8px;
            }}
            QProgressBar::chunk {{
                background-color: {colors['accent']};
                border-radius: 3px;
            }}

            /* ── Tooltip ─────────────────────────── */
            QToolTip {{
                background-color: {colors['bg_surface']};
                border: 0.5px solid {colors['border_default']};
                color: {colors['text_primary']};
                font-size: 11px;
                padding: 4px 8px;
                border-radius: 4px;
            }}
        """)

    def get_current_colors(self):
        """Returns the color dictionary for the current theme."""
        return get_theme_colors(self.theme)

    def toggle_theme(self):
        """Toggles between light and dark themes."""
        self.theme = "light" if self.theme == "dark" else "dark"
        colors = self.get_current_colors()
        
        # 1. Update Application-wide Stylesheet
        self._setup_style(colors)
        self._apply_theme_widget_styles(colors)
        
        # 2. Update Global pyqtgraph options
        configure_pyqtgraph_theme(colors)
        
        # 3. Notify all panels to restyle their internal plots
        panels = [
            self.standard_plots_panel,
            self.ei_panel,
            self.waveforms_panel,
            self.raw_panel,
            self.sta_panel,
            self.umap_panel
        ]
        
        for panel in panels:
            if hasattr(panel, 'restyle_plots'):
                panel.restyle_plots(colors)
        
        # 4. Refresh similarity panel
        self.similarity_panel.restyle_plots(colors)

        # 4b. Refresh population canvases
        self.pop_mosaic_canvas.restyle(colors)
        self.pop_timecourse_canvas.restyle(colors)
        self.pop_acg_canvas.restyle(colors)

        # Update population header styles
        self.pop_tc_label.setStyleSheet(f"font-weight:bold; color: {colors['text_primary']};")
        self.pop_acg_label.setStyleSheet(f"font-weight:bold; color: {colors['text_primary']};")
        
        # 5. Refresh data models if they use custom colors
        if self.table_view.model() and hasattr(self.table_view.model(), 'update_colors'):
            self.table_view.model().update_colors(colors)
        
        # 6. Re-load current cluster to ensure plot colors update
        cluster_id = self._get_selected_cluster_id()
        if cluster_id is not None:
            self.on_tab_changed(self.analysis_tabs.currentIndex())
            
        self.status_bar.showMessage(f"Switched to {self.theme} mode.")

    def _apply_theme_widget_styles(self, colors):
        """Refresh inline styles that cannot be fully expressed in global QSS."""
        if hasattr(self, "reset_button"):
            self.reset_button.setStyleSheet(f"""
                QPushButton {{ border: none; color: {colors['text_tertiary']}; font-size: 14px; }}
                QPushButton:hover {{ color: {colors['text_primary']}; }}
            """)

        if hasattr(self, "pop_view_btn"):
            self.pop_view_btn.setStyleSheet(f"""
                QPushButton {{
                    font-size: 11px;
                    padding: 0 10px;
                    border: 0.5px solid {colors['border_default']};
                    border-radius: 5px;
                    color: {colors['text_secondary']};
                    background: transparent;
                }}
                QPushButton:checked {{
                    background: {colors['status_unsort_bg']};
                    border-color: {colors['accent']};
                    color: {colors['accent_hover']};
                }}
                QPushButton:hover:!checked {{
                    background: {colors['bg_surface']};
                    color: {colors['text_primary']};
                }}
            """)

        if hasattr(self, "pop_expand_btn"):
            bg = colors['accent_positive'] if self.pop_expand_btn.isChecked() else colors['accent']
            self.pop_expand_btn.setStyleSheet(
                f"font-weight: bold; background-color: {bg}; padding: 4px 10px;"
            )

        if hasattr(self, "pop_tc_label"):
            self.pop_tc_label.setStyleSheet(f"font-weight:bold; color: {colors['text_primary']};")
        if hasattr(self, "pop_acg_label"):
            self.pop_acg_label.setStyleSheet(f"font-weight:bold; color: {colors['text_primary']};")

        if hasattr(self, "tree_model"):
            self._apply_tree_item_theme(colors)

    def _apply_tree_item_theme(self, colors):
        """Apply readable item brushes for the current palette across the tree."""
        if self.tree_model is None:
            return

        group_bg = QColor(colors['bg_elevated'])
        group_fg = QColor(colors['text_primary'])
        cell_fg = QColor(colors['text_primary'])

        def visit(item):
            is_group = item.data(Qt.ItemDataRole.UserRole) is None
            item.setForeground(group_fg if is_group else cell_fg)
            if is_group:
                item.setBackground(group_bg)
            else:
                item.setBackground(QColor(colors['bg_panel']))

            for row in range(item.rowCount()):
                child = item.child(row)
                if child is not None:
                    visit(child)

        root = self.tree_model.invisibleRootItem()
        for row in range(root.rowCount()):
            item = root.child(row)
            if item is not None:
                visit(item)

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
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        # Create a widget to contain the filter box and views
        left_content = QWidget()
        left_content_layout = QVBoxLayout(left_content)
        left_content_layout.setContentsMargins(PANEL_PADDING, PANEL_PADDING, PANEL_PADDING, PANEL_PADDING)
        left_content_layout.setSpacing(CTRL_SPACING)

        # --- Filter + View Toggle Row ---
        top_ctrl_layout = QHBoxLayout()
        top_ctrl_layout.setSpacing(4)

        colors = self.get_current_colors()

        self.filter_all_btn = QPushButton("All")
        self.filter_all_btn.setCheckable(True)
        self.filter_all_btn.setFixedHeight(26)
        self.filter_all_btn.setChecked(True)
        self.filter_all_btn.setStyleSheet(
            "border-radius: 5px;"
        )

        # Segmented view toggle
        self.view_group = QButtonGroup(self)
        self.table_view_button = QPushButton("Table")
        self.tree_view_button  = QPushButton("Tree")
        self.view_group.addButton(self.table_view_button)
        self.view_group.addButton(self.tree_view_button)
        self.view_group.setExclusive(True)
        
        for btn in (self.table_view_button, self.tree_view_button):
            btn.setCheckable(True)
            btn.setFixedHeight(26)
        self.table_view_button.setChecked(True)

        # Reset as ghost link
        self.reset_button = QPushButton("↺")
        self.reset_button.setToolTip("Reset View")
        self.reset_button.setFixedSize(26, 26)
        self.reset_button.setStyleSheet(f"""
            QPushButton {{ border: none; color: {colors['text_tertiary']}; font-size: 14px; }}
            QPushButton:hover {{ color: {colors['text_primary']}; }}
        """)

        top_ctrl_layout.addWidget(self.filter_all_btn)
        top_ctrl_layout.addSpacing(8)
        top_ctrl_layout.addWidget(self.table_view_button)
        top_ctrl_layout.addWidget(self.tree_view_button)
        top_ctrl_layout.addStretch()
        top_ctrl_layout.addWidget(self.reset_button)

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

        left_content_layout.addLayout(top_ctrl_layout)
        left_content_layout.addWidget(self.view_stack)

        # --- Similarity Panel ---
        self.similarity_panel = SimilarityPanel(self)
        left_content_layout.addWidget(self.similarity_panel)
        self.similarity_panel.selection_changed.connect(
            self.on_similarity_selection_changed)

        # Add the content to the left pane
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
        self.analysis_tabs.tabBar().setUsesScrollButtons(True)
        self.analysis_tabs.tabBar().setElideMode(Qt.ElideNone)

        # Replace corner widget checkbox with a compact icon button:
        self.pop_view_btn = QPushButton("⊞  Population")
        self.pop_view_btn.setCheckable(True)
        self.pop_view_btn.setFixedHeight(28)
        self.pop_view_btn.setStyleSheet(f"""
                QPushButton {{
                    font-size: 11px;
                    padding: 0 10px;
                    border: 0.5px solid {colors['border_default']};
                border-radius: 5px;
                    color: {colors['text_secondary']};
                    background: transparent;
                }}
                QPushButton:checked {{
                    background: {colors['status_unsort_bg']};
                    border-color: {colors['accent']};
                    color: {colors['accent_hover']};
                }}
                QPushButton:hover:!checked {{
                    background: {colors['bg_surface']};
                    color: {colors['text_primary']};
                }}
        """)
        self.pop_view_btn.toggled.connect(self.toggle_population_split_view)
        self.analysis_tabs.setCornerWidget(self.pop_view_btn, Qt.TopRightCorner)

        # --- NEW: population context widget (right side) ---
        self.pop_context_widget = QWidget()
        pop_layout = QVBoxLayout(self.pop_context_widget)
        pop_layout.setContentsMargins(4, 4, 4, 4)
        pop_layout.setSpacing(6)

        # Top Control Bar (with Expand Button)
        pop_ctrl_layout = QHBoxLayout()
        self.pop_show_ids_checkbox = QCheckBox("Show IDs")
        self.pop_show_ids_checkbox.setChecked(False)
        pop_ctrl_layout.addWidget(self.pop_show_ids_checkbox)
        self.pop_expand_btn = QPushButton("⛶ Full Screen")
        self.pop_expand_btn.setToolTip("Toggle Full Screen Population View")
        self.pop_expand_btn.setCheckable(True)
        self.pop_expand_btn.clicked.connect(self.toggle_population_fullscreen)
        self.pop_expand_btn.setStyleSheet(f"font-weight: bold; background-color: {colors['accent']}; padding: 4px 10px;")
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
        # AC4: Zoom & Pan toolbar for the RF mosaic
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        self.pop_mosaic_toolbar = NavigationToolbar2QT(self.pop_mosaic_canvas, self.pop_mosaic_widget)
        self.pop_mosaic_toolbar.setMaximumHeight(28)
        mosaic_layout.addWidget(self.pop_mosaic_toolbar)
        self.pop_master_splitter.addWidget(self.pop_mosaic_widget)

        # 2. Timecourse Panel
        self.pop_timecourse_widget = QWidget()
        tc_layout = QVBoxLayout(self.pop_timecourse_widget)
        tc_layout.setContentsMargins(0, 0, 0, 0)
        tc_hdr = QHBoxLayout()
        self.pop_tc_label = QLabel("Population Dynamics")
        self.pop_tc_label.setStyleSheet(f"font-weight:bold; color: {colors['text_primary']};")
        self.pop_timecourse_summary = QLabel("n=0  mean_t2p: N/A  mean_fwhm: N/A")
        tc_hdr.addWidget(self.pop_tc_label)
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
        self.pop_acg_label = QLabel("Population Autocorrelation")
        self.pop_acg_label.setStyleSheet(f"font-weight:bold; color: {colors['text_primary']};")
        self.pop_acg_summary = QLabel("n=0")
        acg_hdr.addWidget(self.pop_acg_label)
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

        # --- Tab Order (Short Labels) ---
        self.analysis_tabs.addTab(self.standard_plots_panel, "Standard")
        self.analysis_tabs.addTab(self.ei_panel, "EI")
        self.analysis_tabs.addTab(self.sta_panel, "STA")
        self.analysis_tabs.addTab(self.umap_panel, "UMAP")
        self.analysis_tabs.addTab(self.waveforms_panel, "Waveforms")
        self.analysis_tabs.addTab(self.raw_panel, "Raw")

        # --- Main Splitter and Layout ---
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.addWidget(self.left_pane)
        self.main_splitter.addWidget(right_pane)
        self.main_splitter.setSizes([220, 1800 - 220])
        self.main_splitter.setStretchFactor(0, 0)  # Left panel doesn't stretch
        self.main_splitter.setStretchFactor(1, 1)  # Right panel takes remaining space
        self.main_splitter.setHandleWidth(5)
        self.main_splitter.handle(1).mouseDoubleClickEvent = lambda e: self.toggle_sidebar()
        
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
        self.load_vision_action.setEnabled(True)
        
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

        # --- View Menu ---
        view_menu = menu.addMenu("&View")
        self.toggle_theme_action = view_menu.addAction("Toggle Light/Dark Mode")
        self.toggle_theme_action.triggered.connect(self.toggle_theme)

        # Connect Signals
        load_ks_action.triggered.connect(lambda: self.load_directory())
        self.load_raw_action.triggered.connect(self.load_raw_data_file)
        self.load_vision_action.triggered.connect(self.load_vision_directory)
        self.load_classification_action.triggered.connect(self.load_classification_file)
        self.save_classification_action.triggered.connect(self.on_save_classification_action)
        self.save_action.triggered.connect(self.on_save_action)

        # Connect New Left Panel Buttons
        self.filter_all_btn.clicked.connect(self.reset_views)
        self.tree_view_button.clicked.connect(lambda: self._switch_left_view(0))
        self.table_view_button.clicked.connect(lambda: self._switch_left_view(1))
        
        self.reset_button.clicked.connect(self.reset_views)
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
            selection_model = self.tree_view.selectionModel()
            if selection_model is None or not selection_model.hasSelection():
                return None

            index = selection_model.selectedIndexes()[0]
            item = self.tree_model.itemFromIndex(index)
            if item is None:
                return None

            # Only leaf nodes (cells) have a cluster ID stored. Groups will
            # return None.
            cluster_id = item.data(Qt.ItemDataRole.UserRole)
            return cluster_id

        # Case 2: Table View is active
        elif current_view_index == 1:
            selection_model = self.table_view.selectionModel()
            if selection_model is None or not selection_model.hasSelection() or self.main_cluster_model is None:
                return None

            selected_row = selection_model.selectedIndexes()[0].row()

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
        if hasattr(model, 'update_colors'):
            model.update_colors(self.get_current_colors())
        self.table_view.setModel(model)
        self.table_view.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
        self.table_view.verticalHeader().setVisible(False)
        try:
            self.table_view.selectionModel().selectionChanged.disconnect(
                self.on_view_selection_changed)
        except (TypeError, RuntimeError):
            pass
        self.table_view.selectionModel().selectionChanged.connect(
            self.on_view_selection_changed)

        # Column header labels override (keeps internal df names intact)
        HEADER_LABELS = {
            'cluster_id':        'ID',
            'n_spikes':          '# Spikes',
            'best_chan':         'Ch',
            'KSLabel':           'KS Label',
            'isi_violations_pct':'ISI Viol%',
            'contam_pct':        'Contam%',
            'amp_median':        'Amp (µV)',
            'firing_rate_hz':    'FR (Hz)',
            'template_amp':      'Tpl Amp',
            'max_dup_r':         'Max Dup R',
            'potential_dups':    'Dup?',
            'cell_type':         'Type',
            'status':            'Status',
            'x_um':              'X (µm)',
            'y_um':              'Y (µm)',
            'set':               'Set',
        }

        df_cols = list(model._dataframe.columns)
        col_index = {name: idx for idx, name in enumerate(df_cols)}

        # Monkey-patch pretty header labels onto the model
        model._header_overrides = {}
        _orig_headerData = model.headerData

        def _make_patched(orig, overrides, cols):
            def patched(section, orientation, role=Qt.DisplayRole):
                if orientation == Qt.Horizontal and role == Qt.DisplayRole:
                    col_name = cols[section] if section < len(cols) else None
                    if col_name and col_name in overrides:
                        return overrides[col_name]
                return orig(section, orientation, role)
            return patched

        model.headerData = _make_patched(
            _orig_headerData, model._header_overrides, df_cols)

        for col_name, label in HEADER_LABELS.items():
            model._header_overrides[col_name] = label

        # Allow drag-to-reorder; only apply the default ordering on first load
        header = self.table_view.horizontalHeader()
        header.setSectionsMovable(True)

        if not getattr(self, '_table_columns_initialized', False):
            self._apply_default_column_order(header, df_cols, col_index)
            self._table_columns_initialized = True

        self.table_view.resizeColumnsToContents()

    def _apply_default_column_order(self, header, df_cols, col_index):
        """Apply the desired default visual column order. Called only once."""
        ORDERED_COLS = [
            'cluster_id',       # shown as "ID" — thin
            'n_spikes',
            'best_chan',
            'KSLabel',
            'isi_violations_pct',
            'contam_pct',
            'amp_median',
            'firing_rate_hz',
            'template_amp',
            'max_dup_r',
            'potential_dups',
            'cell_type',
            'status',
            'x_um',
            'y_um',
            'set',
        ]
        visual_order = [c for c in ORDERED_COLS if c in col_index]
        listed = set(ORDERED_COLS)
        for c in df_cols:
            if c not in listed:
                visual_order.append(c)

        for target_visual, col_name in enumerate(visual_order):
            logical = col_index[col_name]
            current_visual = header.visualIndex(logical)
            if current_visual != target_visual:
                header.moveSection(current_visual, target_visual)

    def refresh_table_model(self):
        """Rebuild the main cluster table from the current cluster_df.
        Call this after async column additions (best_chan, amp_median, etc.)
        to make newly added columns visible without resetting user column order."""
        if self.data_manager is None:
            return
        df = self.data_manager.cluster_df
        # Preserve the user's current visual column order across the rebuild
        header = self.table_view.horizontalHeader()
        old_model = self.table_view.model()
        if old_model is not None and hasattr(old_model, '_dataframe'):
            old_cols = list(old_model._dataframe.columns)
            # Build logical→visual map from the old header state
            old_visual_order = [
                old_cols[header.logicalIndex(v)]
                for v in range(header.count())
                if header.logicalIndex(v) < len(old_cols)
            ]
        else:
            old_visual_order = None

        model = HighlightStatusPandasModel(df)
        model.update_colors(self.get_current_colors())
        self.main_cluster_model = model
        self.table_view.setModel(model)
        self.table_view.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
        self.table_view.verticalHeader().setVisible(False)
        try:
            self.table_view.selectionModel().selectionChanged.disconnect(
                self.on_view_selection_changed)
        except (TypeError, RuntimeError):
            pass
        self.table_view.selectionModel().selectionChanged.connect(
            self.on_view_selection_changed)

        # Re-apply header labels
        new_df_cols = list(df.columns)
        HEADER_LABELS = {
            'cluster_id': 'ID', 'n_spikes': '# Spikes', 'best_chan': 'Ch',
            'KSLabel': 'KS Label', 'isi_violations_pct': 'ISI Viol%',
            'contam_pct': 'Contam%', 'amp_median': 'Amp (µV)',
            'firing_rate_hz': 'FR (Hz)', 'template_amp': 'Tpl Amp',
            'max_dup_r': 'Max Dup R', 'potential_dups': 'Dup?',
            'cell_type': 'Type', 'status': 'Status',
            'x_um': 'X (µm)', 'y_um': 'Y (µm)', 'set': 'Set',
        }
        model._header_overrides = {}
        _orig = model.headerData

        def _make_patched(orig, overrides, cols):
            def patched(section, orientation, role=Qt.DisplayRole):
                if orientation == Qt.Horizontal and role == Qt.DisplayRole:
                    col_name = cols[section] if section < len(cols) else None
                    if col_name and col_name in overrides:
                        return overrides[col_name]
                return orig(section, orientation, role)
            return patched

        model.headerData = _make_patched(_orig, model._header_overrides, new_df_cols)
        for col_name, label in HEADER_LABELS.items():
            model._header_overrides[col_name] = label

        new_header = self.table_view.horizontalHeader()
        new_header.setSectionsMovable(True)
        new_col_index = {name: idx for idx, name in enumerate(new_df_cols)}

        if old_visual_order:
            # Restore the user's previous order for columns that still exist;
            # new columns (e.g. best_chan just added) go after
            ordered = [c for c in old_visual_order if c in new_col_index]
            for c in new_df_cols:
                if c not in ordered:
                    ordered.append(c)
        else:
            ordered = None

        if ordered:
            for target_visual, col_name in enumerate(ordered):
                if col_name not in new_col_index:
                    continue
                logical = new_col_index[col_name]
                current_visual = new_header.visualIndex(logical)
                if current_visual != target_visual:
                    new_header.moveSection(current_visual, target_visual)
        else:
            self._apply_default_column_order(
                new_header, new_df_cols, new_col_index)

        self.table_view.resizeColumnsToContents()

    
    def setup_tree_model(self, model):
        """Sets up the tree view model and connects the selection changed signal."""
        self.tree_view.setModel(model)
        self._apply_tree_item_theme(self.get_current_colors())
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
        self.refresh_table_model()

    def _update_tree_view_duplicate_highlight(self):
        # Collect all duplicate IDs
        colors = self.get_current_colors()
        sdf = self.data_manager.status_df
        duplicate_ids = sdf[sdf['status'] ==
                            'Duplicate']['cluster_id'].tolist()
        duplicate_ids = set(duplicate_ids)
        self._apply_tree_item_theme(colors)

        def visit(item):
            cluster_id = item.data(Qt.ItemDataRole.UserRole)
            if cluster_id in duplicate_ids:
                item.setForeground(QColor(colors['status_noise_text']))
            for child_row in range(item.rowCount()):
                child_item = item.child(child_row)
                if child_item is not None:
                    visit(child_item)

        for row in range(self.tree_model.rowCount()):
            group_item = self.tree_model.item(row)
            if group_item is not None:
                visit(group_item)

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

    def select_sta_view(self, view_type, _force_animation=False):
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
        widths = self.main_splitter.sizes()
        if widths[0] < 50:
            # --- EXPAND ---
            total_width = sum(widths)
            self.main_splitter.setSizes(
                [self.last_left_width, total_width - self.last_left_width])
            self.sidebar_collapsed = False
        else:
            # --- COLLAPSE ---
            self.last_left_width = widths[0]
            total_width = sum(widths)
            self.main_splitter.setSizes([0, total_width])
            self.sidebar_collapsed = True

    def toggle_population_fullscreen(self, checked):
        """Toggles the Population panel to take up 100% of the right pane."""
        colors = self.get_current_colors()
        if checked:
            # Full screen mode: Collapse the main tabs completely
            self.right_splitter.setSizes([0, 1000])
            self.pop_expand_btn.setText("🗗 Restore")
            self.pop_expand_btn.setStyleSheet(f"font-weight: bold; background-color: {colors['accent_positive']}; padding: 4px 10px;")
        else:
            # Restore mode: 75/25 split
            total = sum(self.right_splitter.sizes()) or 1400
            left_size = max(int(total * 0.75), 400)
            right_size = total - left_size
            self.right_splitter.setSizes([left_size, right_size])
            self.pop_expand_btn.setText("⛶ Full Screen")
            self.pop_expand_btn.setStyleSheet(f"font-weight: bold; background-color: {colors['accent']}; padding: 4px 10px;")

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
        
        # Also cleanup vision_load_worker if it exists
        if hasattr(self, 'vision_load_worker') and self.vision_load_worker:
            self.vision_load_worker.deleteLater()
            self.vision_load_worker = None

        # Stop any running raw trace worker
        if hasattr(self, 'raw_panel'):
            self.raw_panel._stop_worker()

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
