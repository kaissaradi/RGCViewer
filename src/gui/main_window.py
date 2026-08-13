import os
import logging
from qtpy.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSplitter,
    QStatusBar,
    QHeaderView,
    QMessageBox,
    QTabWidget,
    QTreeView,
    QAbstractItemView,
    QLabel,
    QMenu,
    QInputDialog,
    QStackedWidget,
    QApplication,
    QCheckBox,
    QProgressBar,
    QButtonGroup,
    QLineEdit,
    QShortcut,
    QSlider,
)
from qtpy.QtCore import (
    Qt,
    QItemSelectionModel,
    QThread,
    QTimer,
    Signal,
)
from qtpy.QtGui import QFont, QStandardItemModel, QKeySequence
from ..analysis.data_manager import DataManager
from typing import Optional

# Custom GUI Modules
from .widgets.widgets import (
    MplCanvas,
    HiddenIdFilterProxyModel,
    HighlightStatusPandasModel,
    CustomTableView,
    ClusterTreeDelegate,
    PandasModel,
    TREE_CH_WIDTH,
    TREE_COL_CH,
    TREE_COL_ID,
    TREE_HEADERS,
    TREE_SPIKES_WIDTH,
    refresh_tree_group_counts,
    find_cell_item,
    set_channel_item,
    tree_item_from_index,
    make_nav_toolbar,
)
from . import callbacks
from . import plot_export
from .panels.population_panel import (
    draw_population_rfs_plot,
    population_group_plots_cached,
    rf_vision_id_at,
)
from .panels.similarity_panel import SimilarityPanel
from .panels.waveforms_panel import WaveformPanel
from .panels.standard_plots_panel import StandardPlotsPanel
from .panels.chirp_panel import ChirpPanel
from .panels.contrast_panel import ContrastPanel
from .panels.grating_panel import GratingPanel
from .panels.ei_panel import EIPanel
from .panels.raw_panel import RawPanel
from .panels.sta_panel import STAPanel
from .workers.workers import FeatureWorker
from .shortcuts import KeyForwarder
from qtpy.QtGui import QColor
from .panels.umap_panel import UMAPPanel
from .theme import (
    DARK_COLORS,
    PANEL_PADDING,
    CTRL_SPACING,
    ROW_HEIGHT,
    UI_FONT_FAMILY,
    configure_pyqtgraph_theme,
    get_theme_colors,
)

# Array calibration dialog
from .panels.array_calibration_panel import ArrayCalibrationDialog

# Global pyqtgraph configuration
configure_pyqtgraph_theme(DARK_COLORS)

logger = logging.getLogger(__name__)

SIDEBAR_COLLAPSED_WIDTH = 22


class MainWindow(QMainWindow):
    # Emitted from the _warm_physics background thread (a plain
    # threading.Thread, not a QThread) once physics-cache warming
    # completes, to safely hop back onto the GUI thread and start the
    # grating batch QThread there. A Qt Signal, unlike QTimer.singleShot(),
    # can be emitted from any thread and is guaranteed to be delivered on
    # the thread of the connected slot's receiver (here, MainWindow itself,
    # which lives on the GUI thread) via Qt's normal queued-connection
    # mechanism — this is the actually-reliable cross-thread marshaling
    # primitive, not a timer that needs an event loop on the emitting side.
    physics_warm_done = Signal()

    def __init__(self, default_kilosort_dir=None, default_dat_file=None):
        super().__init__()
        self.setWindowTitle("RGC Viewer")
        self._place_window()

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
        self._expect_physics = False

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
        self.last_left_width = 280
        self.sidebar_collapsed = False
        self.feature_worker_thread = None
        self.population_view_enabled = False

        # --- UI Setup ---
        self._setup_style(DARK_COLORS)
        self._setup_ui()
        self.analysis_tabs.currentChanged.connect(self.on_tab_changed)
        self.central_widget.setEnabled(False)
        self.status_bar.showMessage(
            "Welcome to RGC Viewer. Please load a Kilosort directory to begin."
        )

        # selection timer for debouncing rapid selections
        self.selection_timer = QTimer(self)
        self.selection_timer.setSingleShot(True)
        self.selection_timer.setInterval(25)  # 25ms - minimal delay for responsive feel
        self.selection_timer.timeout.connect(self._process_selection)
        self._pending_cluster_id = None

        self.folder_selection_timer = QTimer(self)
        self.folder_selection_timer.setSingleShot(True)
        self.folder_selection_timer.setInterval(25)
        self.folder_selection_timer.timeout.connect(self._process_folder_selection)
        self._pending_folder_item = None

        # The table's group column follows the tree. Driven off the model's own
        # signals rather than from each of the ~10 places that mutate the tree,
        # so a new grouping path cannot forget to refresh it.
        self.group_column_timer = QTimer(self)
        self.group_column_timer.setSingleShot(True)
        self.group_column_timer.setInterval(150)
        self.group_column_timer.timeout.connect(self._refresh_group_column)
        self.tree_model.rowsInserted.connect(self._schedule_group_column_refresh)
        self.tree_model.rowsRemoved.connect(self._schedule_group_column_refresh)
        self.tree_model.itemChanged.connect(self._schedule_group_column_refresh)

        # Only open a dataset when the caller asked (--kilosort-dir / tests).
        # File dialogs still remember the last folder via recent_paths.
        # Auto-reopening the last run made every launch look hung, and a
        # broken remembered path made the app unusable until QSettings
        # was cleared.
        if default_kilosort_dir and os.path.isdir(default_kilosort_dir):
            ks_dir, dat_path = default_kilosort_dir, default_dat_file
            QTimer.singleShot(0, lambda: self.load_directory(ks_dir, dat_path))

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
                index, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows
            )
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
                    new_idx,
                    QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows,
                )
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
                index, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows
            )
            view.scrollTo(index)

    def _place_window(self):
        """Open at 1800×1000, but never larger than the usable screen."""
        app = QApplication.instance()
        screen = app.primaryScreen() if app is not None else None
        if screen is None:
            self.setGeometry(50, 50, 1800, 1000)
            return
        avail = screen.availableGeometry()
        width = min(1800, avail.width())
        height = min(1000, avail.height())
        self.setGeometry(avail.x(), avail.y(), width, height)

    def _setup_style(self, colors):
        self.setFont(QFont("Helvetica Neue", 11))

        self.setStyleSheet(f"""
            /* ── Base ───────────────────────────── */
            QWidget {{
                color: {colors['text_primary']};
                background-color: {colors['bg_base']};
                font-family: {UI_FONT_FAMILY};
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
                background-color: {colors['accent']};
                border-color: {colors['accent']};
                color: {colors['accent_text']};
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
                color: {colors['accent']};
                border-bottom: 2px solid {colors['accent']};
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
                color: {colors['text_primary']};
                font-size: 12px;
            }}

            /* ── Group boxes ─────────────────────── */
            QGroupBox {{
                color: {colors['text_primary']};
                border: 1px solid {colors['border_default']};
                border-radius: 3px;
                margin-top: 12px;
                padding-top: 8px;
                font-size: 12px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: {colors['text_tertiary']};
            }}

            /* ── Line edits ──────────────────────── */
            QLineEdit {{
                background-color: {colors['bg_panel']};
                border: 1px solid {colors['border_default']};
                border-radius: 3px;
                padding: 3px 8px;
                color: {colors['text_primary']};
                font-size: 12px;
                min-height: 22px;
            }}
            QLineEdit:focus {{
                border-color: {colors['border_focus']};
            }}
            QLineEdit::placeholder {{
                color: {colors['text_tertiary']};
            }}

            /* ── Plot toolbars ───────────────────── */
            QToolBar {{
                background: {colors['bg_panel']};
                border: none;
                spacing: 2px;
                padding: 0;
            }}
            QToolButton {{
                background: transparent;
                color: {colors['text_secondary']};
                border: none;
                padding: 2px;
            }}
            QToolButton:hover {{
                background: {colors['bg_elevated']};
                color: {colors['text_primary']};
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
            /* Tree item colors handled on tree_view + ClusterTreeDelegate */

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
                background-color: {colors['bg_tooltip']};
                border: 0.5px solid {colors['border_default']};
                color: {colors['text_tooltip']};
                font-size: 11px;
                padding: 4px 8px;
                border-radius: 3px;
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
            self.chirp_panel,
            self.contrast_panel,
            self.grating_panel,
            self.ei_panel,
            self.waveforms_panel,
            self.raw_panel,
            self.sta_panel,
            self.umap_panel,
        ]

        for panel in panels:
            if hasattr(panel, "restyle_plots"):
                panel.restyle_plots(colors)

        # 4. Refresh similarity panel
        self.similarity_panel.restyle_plots(colors)

        # 4b. Refresh population canvases
        self.pop_mosaic_canvas.restyle(colors)
        self.pop_timecourse_canvas.restyle(colors)
        self.pop_acg_canvas.restyle(colors)

        # Update population header styles
        self.pop_tc_label.setStyleSheet(
            f"font-weight:bold; color: {colors['text_primary']};"
        )
        self.pop_acg_label.setStyleSheet(
            f"font-weight:bold; color: {colors['text_primary']};"
        )

        if hasattr(self, "sidebar_toggle_btn"):
            self._style_sidebar_toggle_btn(colors)

        # 5. Refresh data models if they use custom colors
        if self.table_view.model() and hasattr(
            self.table_view.model(), "update_colors"
        ):
            self.table_view.model().update_colors(colors)

        # 6. Re-load current cluster to ensure plot colors update
        cluster_id = self._get_selected_cluster_id()
        if cluster_id is not None:
            self.on_tab_changed(self.analysis_tabs.currentIndex())

        self.status_bar.showMessage(f"Switched to {self.theme} mode.")

    def _apply_theme_widget_styles(self, colors):
        """Refresh inline styles that cannot be fully expressed in global QSS."""
        ghost = f"""
            QPushButton {{ border: none; color: {colors['text_tertiary']}; font-size: 14px; }}
            QPushButton:hover {{ color: {colors['text_primary']}; }}
        """
        if hasattr(self, "reset_button"):
            self.reset_button.setStyleSheet(ghost)
        if hasattr(self, "collapse_all_btn"):
            self.collapse_all_btn.setStyleSheet(ghost)

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
                    background: {colors['accent']};
                    border-color: {colors['accent']};
                    color: {colors['accent_text']};
                }}
                QPushButton:hover:!checked {{
                    background: {colors['bg_surface']};
                    color: {colors['text_primary']};
                }}
            """)

        if hasattr(self, "pop_expand_btn"):
            bg = (
                colors["accent_positive"]
                if self.pop_expand_btn.isChecked()
                else colors["accent"]
            )
            self.pop_expand_btn.setStyleSheet(
                f"font-weight: bold; background-color: {bg}; "
                f"color: {colors['accent_text']}; padding: 4px 10px;"
            )

        if hasattr(self, "pop_tc_label"):
            self.pop_tc_label.setStyleSheet(
                f"font-weight:bold; color: {colors['text_primary']};"
            )
        if hasattr(self, "pop_acg_label"):
            self.pop_acg_label.setStyleSheet(
                f"font-weight:bold; color: {colors['text_primary']};"
            )

        if hasattr(self, "tree_model"):
            self._apply_tree_item_theme(colors)

    def _apply_tree_view_style(self, colors):
        """Per-tree stylesheet; branch chrome is drawn by ClusterTreeDelegate."""
        self.tree_view.setStyleSheet(f"""
            QTreeView {{
                background-color: {colors['bg_panel']};
                color: {colors['text_primary']};
                border: none;
                alternate-background-color: {colors['bg_surface']};
                selection-background-color: {colors['selection_bg']};
            }}
            QTreeView::item {{
                color: {colors['text_primary']};
                padding: 2px 0;
            }}
            QTreeView::item:hover {{
                background: {colors['bg_surface']};
            }}
            QTreeView::item:selected {{
                background: {colors['selection_bg']};
            }}
            QTreeView::branch {{
                background: transparent;
                border: none;
                image: none;
            }}
            QHeaderView::section {{
                background-color: {colors['bg_panel']};
                color: {colors['text_tertiary']};
                border: none;
                border-bottom: 1px solid {colors['border_subtle']};
                padding: 2px 4px;
                font-size: 10px;
            }}
        """)
        if hasattr(self, "_cluster_tree_delegate"):
            self._cluster_tree_delegate.update_colors(colors)

    def _apply_tree_item_theme(self, colors):
        """Apply readable item brushes for the current palette across the tree."""
        self._apply_tree_view_style(colors)
        if self.tree_model is None:
            return

        group_bg = QColor(colors["bg_elevated"])
        group_fg = QColor(colors["text_primary"])
        cell_fg = QColor(colors["text_primary"])
        secondary = QColor(colors["text_secondary"])

        def paint_row(item):
            is_group = item.data(Qt.ItemDataRole.UserRole) is None
            item.setForeground(group_fg if is_group else cell_fg)
            item.setBackground(group_bg if is_group else QColor(colors["bg_panel"]))
            model = item.model()
            if model is None:
                return
            idx = item.index()
            for col in range(1, model.columnCount()):
                sib = model.itemFromIndex(idx.sibling(idx.row(), col))
                if sib is None:
                    continue
                if is_group:
                    sib.setBackground(group_bg)
                    sib.setForeground(secondary)
                else:
                    sib.setForeground(secondary)

        def visit(item):
            paint_row(item)
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

        # 0. UMAP side panels — show where this one cell sits among the cells
        # it was embedded with. Blit-only, so it costs nothing on the fast path.
        self._highlight_selection_in_umap([cluster_id])

        # 1. Population RF Updates - Only if hot-swap is possible (fast path)
        # Full rebuild is deferred to Tier 2 to avoid first-click freeze
        if self.population_view_enabled:
            # Quick check if we can hot-swap (check if canvas already has state)
            canvas = self.pop_mosaic_canvas
            can_hot_swap = (
                hasattr(canvas, "_pop_plot_state")
                and canvas._pop_plot_state.get("ax") in canvas.fig.axes
            )
            if can_hot_swap:
                # Fast update - update existing ellipse geometry only
                try:
                    draw_population_rfs_plot(
                        main_window=self,
                        selected_cell_id=cluster_id,
                        canvas=self.pop_mosaic_canvas,
                    )
                except Exception as e:
                    logger.error(f"Tier 1 Pop Split update failed: {e}")
            # If can't hot-swap, defer to Tier 2 for full rebuild

        # 2. Standard Plots Panel (ACG, ISI, Firing Rate) - handled in _draw_plots to avoid duplicate updates
        # Standard plots are updated only in _draw_plots to prevent redundant redraws

        # 3. Electrical Image (EI) - Only if cached
        has_cached_ei = False
        if self.data_manager:
            if hasattr(self.data_manager, "has_cached_ei"):
                has_cached_ei = self.data_manager.has_cached_ei(cluster_id)
            elif hasattr(self.data_manager, "ei_cache"):
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

        self.status_bar.showMessage(f"Loading data for Cluster ID: {cluster_id}...")

        cached_features = self.data_manager.get_lightweight_features(cluster_id)
        if cached_features:
            self._draw_plots(cluster_id, cached_features)
            return

        # Only run FeatureWorker if dat_path is available
        if self.data_manager.dat_path is not None:
            # Cleanup previous worker before starting a new one
            self._cleanup_thread("feature_worker_thread")

            self.feature_worker_thread = QThread()
            self.feature_worker = FeatureWorker(self.data_manager, cluster_id)
            self.feature_worker.moveToThread(self.feature_worker_thread)
            self.feature_worker.features_ready.connect(self.on_features_ready)
            self.feature_worker.error.connect(
                lambda msg: self.status_bar.showMessage(msg, 4000)
            )
            self.feature_worker_thread.started.connect(self.feature_worker.run)
            self.feature_worker_thread.start()
        else:
            self.status_bar.showMessage(
                "Raw data file not loaded: waveform plot disabled.", 4000
            )
            self._draw_plots(cluster_id, None)

    def _schedule_group_column_refresh(self, *_args):
        """Debounce tree edits into one group-column update.

        Grouping a lasso selection fires a row-inserted signal per cell, so the
        column is rebuilt once after the burst settles rather than hundreds of
        times during it.
        """
        timer = getattr(self, "group_column_timer", None)
        if timer is not None:
            timer.start()

    def _refresh_group_column(self):
        """Write each cell's innermost folder into the table's group column.

        Kept in sync from the tree rather than stored on the cell, because the
        tree is the authority on grouping — the same cluster_df is reused across
        regroupings, and anything cached would go stale the moment a cell moved.
        """
        dm = self.data_manager
        if dm is None or dm.cluster_df is None or dm.cluster_df.empty:
            return
        if "cluster_id" not in dm.cluster_df.columns:
            return

        refresh_tree_group_counts(self.tree_model.invisibleRootItem())
        mapping = callbacks.build_cluster_group_map(self)
        values = [mapping.get(int(c), "") for c in dm.cluster_df["cluster_id"]]
        col = callbacks.GROUP_COLUMN
        is_new = col not in dm.cluster_df.columns

        if is_new:
            # Sits immediately before KS Label, which is where the eye lands
            # when scanning a row for what a cell is.
            dm.cluster_df[col] = values
            cols = list(dm.cluster_df.columns)
            cols.remove(col)
            anchor = cols.index("KSLabel") if "KSLabel" in cols else len(cols)
            cols.insert(anchor, col)
            dm.cluster_df = dm.cluster_df[cols]
            self.refresh_table_model()
            return

        dm.cluster_df[col] = values

        # Update the live model in place. refresh_table_model rebuilds the
        # whole table and reinstalls the proxy, which is far too heavy to run
        # on every tree edit.
        model = self.table_view.model()
        source = model.sourceModel() if hasattr(model, "sourceModel") else model
        frame = getattr(source, "_dataframe", None)
        if frame is None or col not in frame.columns:
            self.refresh_table_model()
            return
        frame[col] = values
        if hasattr(source, "refresh_view"):
            source.refresh_view()
        else:
            top = source.index(0, 0)
            bottom = source.index(source.rowCount() - 1, source.columnCount() - 1)
            source.dataChanged.emit(top, bottom)

    def _highlight_selection_in_umap(self, cluster_ids):
        """Mirror the tree selection onto the UMAP tab's side panels.

        Cheap by construction — the RF mosaic and trace overlays repaint by
        blitting an overlay, not by redrawing — so this runs on every settled
        selection without a visibility check. Never allowed to break selection
        handling if the panel is mid-rebuild.
        """
        panel = getattr(self, "umap_panel", None)
        if panel is None:
            return
        try:
            panel.highlight_cells(cluster_ids)
        except Exception:
            logger.debug("UMAP side-panel highlight failed", exc_info=True)

    def _process_folder_selection(self):
        """Draw the selected population folder after folder navigation settles."""
        item = self._pending_folder_item
        if item is None:
            return

        group_ids = self._get_group_cluster_ids(item)
        if not group_ids:
            return

        # Highlight the folder on the UMAP tab before the population-view gate
        # below: those panels live in the analysis tab, not the population pane,
        # so they should track the tree whether or not the split view is open.
        self._highlight_selection_in_umap(group_ids)

        # Group contrast-response curves follow the tree folder too. Cheap
        # enough only because the panel recomputes lazily when it is the
        # visible tab; see ContrastPanel.update_group.
        panel = getattr(self, "contrast_panel", None)
        if panel is not None:
            try:
                panel.update_group(group_ids)
            except Exception:
                logger.debug("Contrast group update failed", exc_info=True)

        if not self.population_view_enabled:
            return

        # Skip if Qt fired a duplicate selectionChanged for the same folder
        # (happens on focus changes and arrow-key repeats landing on the same item).
        current_sig = frozenset(group_ids)
        if getattr(self, "_last_pop_folder_sig", None) == current_sig:
            return
        self._last_pop_folder_sig = current_sig

        draw_population_rfs_plot(
            main_window=self,
            subset_cell_ids=group_ids,
            canvas=self.pop_mosaic_canvas,
        )
        callbacks.redraw_population_panels(self, subset=group_ids)

    def on_features_ready(self, cluster_id, features):
        """
        Cache features and update UI ONLY if still the current selection.
        Prevents stale data from overwriting fresh results.
        """
        current_selection = self._get_selected_cluster_id()

        # CRITICAL: Discard stale results BEFORE caching
        if cluster_id != current_selection:
            logger.debug(
                f"Discarding stale features for C{cluster_id} (now viewing C{current_selection})"
            )
            return

        # Cache the newly computed features
        self.data_manager.ei_cache[cluster_id] = features

        # Only draw if still on a tab that needs these features
        current_tab = self.analysis_tabs.currentWidget()
        if current_tab in (
            self.ei_panel,
            self.waveforms_panel,
            self.standard_plots_panel,
        ):
            self._draw_plots(cluster_id, features)

        # Cleanup with timeout to prevent hangs
        self._cleanup_thread("feature_worker_thread")

    def _cleanup_thread(self, thread_attr: str, timeout_ms: int = 2000):
        """
        Safely cleanup a QThread and its worker with timeout.
        Prevents memory leaks and application hangs.

        Workers that tidy up after themselves call ``deleteLater()`` on their
        thread, which destroys the C++ object while this Python attribute still
        references its wrapper. Touching it then raises RuntimeError — and
        because this runs in a loop from closeEvent, one dead wrapper aborted
        the cleanup of every thread after it, which is what produced Qt's
        "QThread: Destroyed while thread is still running" on exit. A dead
        wrapper means the thread is already gone, so it is simply cleared.
        """
        thread = getattr(self, thread_attr, None)
        try:
            if thread and thread.isRunning():
                thread.quit()
                if not thread.wait(timeout_ms):  # Timeout prevents infinite waits
                    logger.warning(
                        f"Thread {thread_attr} didn't exit cleanly, terminating")
                    thread.terminate()
                    thread.wait(1000)
        except RuntimeError:
            logger.debug("%s was already deleted; nothing to stop", thread_attr)
        setattr(self, thread_attr, None)

    def on_tab_changed(self, index):
        """
        Handles updates when the user switches tabs OR when the active tab
        is refreshed after a cluster change.

        Only the active panel is updated.
        """
        current_panel = self.analysis_tabs.widget(index)
        # Hidden tabs skip layout until first show. Force the UMAP toolbar
        # to commit sizes on every visit so the first open matches STA-and-back.
        if current_panel is self.umap_panel and hasattr(
            self.umap_panel, "_schedule_layout"
        ):
            self.umap_panel._schedule_layout()

        cluster_id = self._get_selected_cluster_id()
        if cluster_id is None:
            return

        if current_panel == self.standard_plots_panel:
            # Only compute standard plots when this tab is actually visible
            self.standard_plots_panel.update_all(cluster_id)

        elif current_panel == self.ei_panel:
            try:
                self.ei_panel.update_ei([cluster_id])
            except Exception as e:
                logger.error(f"Tab change EI update failed: {e}")

        elif current_panel == self.chirp_panel:
            self.chirp_panel.update_all(cluster_id)

        elif current_panel == self.contrast_panel:
            self.contrast_panel.update_all(cluster_id)

        elif current_panel == self.grating_panel:
            self.grating_panel.update_all(cluster_id)

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
                hasattr(canvas, "_pop_plot_state")
                and canvas._pop_plot_state.get("ax") in canvas.fig.axes
            )
            if not can_hot_swap:
                # Full rebuild needed - do it in Tier 2
                try:
                    draw_population_rfs_plot(
                        main_window=self,
                        selected_cell_id=cluster_id,
                        canvas=self.pop_mosaic_canvas,
                    )
                except Exception as e:
                    logger.error(f"Tier 2 Pop Split rebuild failed: {e}")

            # Keep Average Timecourse & ACG live when a cell is selected.
            # Those plots show the parent folder, not the cell. Same-group
            # scroll after the first draw is a cache hit; skip the matplotlib
            # rebuild so chirp-tab scroll does not redraw them every time.
            try:
                subset = self._get_pop_subset_ids()
                if not population_group_plots_cached(subset):
                    callbacks.redraw_population_panels(self, subset=subset)
            except Exception as e:
                logger.error(
                    f"Failed to update population panels on cell selection: {e}"
                )

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

        elif current_tab == self.chirp_panel:
            self.chirp_panel.update_all(cluster_id)

        elif current_tab == self.contrast_panel:
            self.contrast_panel.update_all(cluster_id)

        elif current_tab == self.grating_panel:
            self.grating_panel.update_all(cluster_id)

        elif current_tab == self.waveforms_panel:
            self.waveforms_panel.update_all(cluster_id)

        elif current_tab == self.raw_panel:
            self.raw_panel.load_data(cluster_id)

        elif current_tab == self.sta_panel:
            self.sta_panel.update_view(cluster_id)

        self.status_bar.showMessage("Ready.", 2000)

    def on_standard_plot_ready(self, cluster_id):
        """Called when StandardPlotsWorker finishes caching a cluster.

        If the user is still viewing this cluster on the Standard tab,
        refresh the panel so the plots appear without needing another click.
        """
        current_id = self._get_selected_cluster_id()
        if current_id is None or current_id != cluster_id:
            return
        current_tab = self.analysis_tabs.currentWidget()
        if current_tab == self.standard_plots_panel:
            self.standard_plots_panel.update_all(cluster_id)

    def _setup_ui(self):
        """Initializes and lays out all the UI widgets."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QHBoxLayout(self.central_widget)

        # --- Left Pane ---
        self.left_pane = QWidget()
        left_pane_layout = QHBoxLayout(self.left_pane)
        left_pane_layout.setContentsMargins(0, 0, 0, 0)
        left_pane_layout.setSpacing(0)

        # Create a widget to contain the filter box and views
        left_content = QWidget()
        left_content_layout = QVBoxLayout(left_content)
        left_content_layout.setContentsMargins(
            PANEL_PADDING, PANEL_PADDING, PANEL_PADDING, PANEL_PADDING
        )
        left_content_layout.setSpacing(CTRL_SPACING)

        # --- Filter + View Toggle Row ---
        top_ctrl_layout = QHBoxLayout()
        top_ctrl_layout.setSpacing(4)

        colors = self.get_current_colors()

        self.filter_all_btn = QPushButton("All")
        self.filter_all_btn.setCheckable(True)
        self.filter_all_btn.setFixedHeight(26)
        self.filter_all_btn.setChecked(True)
        self.filter_all_btn.setStyleSheet("border-radius: 5px;")

        # Segmented view toggle
        self.view_group = QButtonGroup(self)
        self.table_view_button = QPushButton("Table")
        self.tree_view_button = QPushButton("Tree")
        self.view_group.addButton(self.table_view_button)
        self.view_group.addButton(self.tree_view_button)
        self.view_group.setExclusive(True)

        for btn in (self.table_view_button, self.tree_view_button):
            btn.setCheckable(True)
            btn.setFixedHeight(26)
        self.tree_view_button.setChecked(True)

        ghost = f"""
            QPushButton {{ border: none; color: {colors['text_tertiary']}; font-size: 14px; }}
            QPushButton:hover {{ color: {colors['text_primary']}; }}
        """
        self.collapse_all_btn = QPushButton("⊟")
        self.collapse_all_btn.setToolTip("Collapse all folders")
        self.collapse_all_btn.setFixedSize(26, 26)
        self.collapse_all_btn.setStyleSheet(ghost)

        self.reset_button = QPushButton("↺")
        self.reset_button.setToolTip("Reset View")
        self.reset_button.setFixedSize(26, 26)
        self.reset_button.setStyleSheet(ghost)

        top_ctrl_layout.addWidget(self.filter_all_btn)
        top_ctrl_layout.addSpacing(8)
        top_ctrl_layout.addWidget(self.table_view_button)
        top_ctrl_layout.addWidget(self.tree_view_button)
        top_ctrl_layout.addStretch()
        top_ctrl_layout.addWidget(self.collapse_all_btn)
        top_ctrl_layout.addWidget(self.reset_button)

        # --- View Stack (Tree and Table) ---
        self.view_stack = QStackedWidget()

        # Tree View
        self.tree_view = QTreeView()
        self.tree_view.setHeaderHidden(False)
        self.tree_view.setRootIsDecorated(True)
        self.tree_view.setIndentation(16)
        self.tree_view.setAnimated(True)
        self.tree_view.setUniformRowHeights(True)
        self.tree_view.setAlternatingRowColors(True)
        self._cluster_tree_delegate = ClusterTreeDelegate(
            self.tree_view, self.get_current_colors()
        )
        self.tree_view.setItemDelegate(self._cluster_tree_delegate)
        self._apply_tree_view_style(self.get_current_colors())
        self.tree_view.setDragEnabled(True)
        self.tree_view.setAcceptDrops(True)
        self.tree_view.setDropIndicatorShown(True)
        self.tree_view.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        # ExtendedSelection gives shift-click ranges and ctrl-click toggles for
        # free, and lets a drag carry the whole selection rather than the one
        # row under the cursor.
        self.tree_view.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.tree_view.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.tree_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.open_tree_context_menu)

        # Delete sends the selection to Trash. Scoped to the tree rather than
        # the window so Delete keeps its normal meaning in text fields.
        self._trash_shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self.tree_view)
        self._trash_shortcut.setContext(Qt.WidgetShortcut)
        self._trash_shortcut.activated.connect(self.move_tree_selection_to_trash)

        # Table View
        self.table_view = CustomTableView()
        self.table_view.setSortingEnabled(True)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        # Same shift/ctrl selection as the tree, and select whole rows so a
        # click anywhere in the row picks the cell it describes.
        self.table_view.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.table_view.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table_view.customContextMenuRequested.connect(
            self.open_table_context_menu
        )

        self._table_trash_shortcut = QShortcut(
            QKeySequence(Qt.Key_Delete), self.table_view
        )
        self._table_trash_shortcut.setContext(Qt.WidgetShortcut)
        self._table_trash_shortcut.activated.connect(
            self.move_table_selection_to_trash
        )

        self.view_stack.addWidget(self.tree_view)
        self.view_stack.addWidget(self.table_view)
        # Default to tree — grouping is the daily path
        self.view_stack.setCurrentIndex(0)

        left_content_layout.addLayout(top_ctrl_layout)

        # --- Sidebar Search Bar ---
        self.cluster_search_bar = QLineEdit()
        self.cluster_search_bar.setPlaceholderText("Search clusters...")
        self.cluster_search_bar.setClearButtonEnabled(True)
        self.cluster_search_bar.setFixedHeight(28)
        self.cluster_search_bar.textChanged.connect(self._filter_sidebar)
        left_content_layout.addWidget(self.cluster_search_bar)

        left_content_layout.addWidget(self.view_stack)

        # --- Similarity Panel ---
        self.similarity_panel = SimilarityPanel(self)
        left_content_layout.addWidget(self.similarity_panel)
        self.similarity_panel.selection_changed.connect(
            self.on_similarity_selection_changed
        )

        left_pane_layout.addWidget(left_content, stretch=1)

        # Collapse/expand control on the right edge of the cluster sidebar
        self.sidebar_toggle_btn = QPushButton("\u2212")
        self.sidebar_toggle_btn.setToolTip("Collapse cluster sidebar")
        self.sidebar_toggle_btn.setFixedWidth(20)
        self.sidebar_toggle_btn.setFlat(True)
        self.sidebar_toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.sidebar_toggle_btn.clicked.connect(self.toggle_sidebar)
        self._style_sidebar_toggle_btn(colors)
        left_pane_layout.addWidget(self.sidebar_toggle_btn)

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
                    background: {colors['accent']};
                    border-color: {colors['accent']};
                    color: {colors['accent_text']};
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
        self.pop_show_ids_checkbox.stateChanged.connect(self._on_pop_show_ids_toggled)
        pop_ctrl_layout.addWidget(self.pop_show_ids_checkbox)

        # DS/OS classification threshold slider. Controls how strong DSI/OSI
        # must be (after already passing grating_calc's significance/
        # amplitude gate — this does NOT loosen that gate, only how
        # strongly-tuned a cell must be to get drawn as DS/OS) — see
        # population_panel.py's _best_dsos_condition. Range 0.1-0.9 in
        # integer-slider units of 0.01 (Qt's QSlider is integer-only);
        # default 0.3 matches grating_calc.DSI_THRESHOLD/OSI_THRESHOLD.
        self.dsos_threshold = 0.3
        pop_ctrl_layout.addWidget(QLabel("DS/OS threshold:"))
        self.pop_dsos_threshold_slider = QSlider(Qt.Horizontal)
        self.pop_dsos_threshold_slider.setRange(10, 90)
        self.pop_dsos_threshold_slider.setValue(30)
        self.pop_dsos_threshold_slider.setFixedWidth(100)
        self.pop_dsos_threshold_slider.setToolTip(
            "How strong DSI/OSI must be to count as DS/OS (0.10-0.90)"
        )
        pop_ctrl_layout.addWidget(self.pop_dsos_threshold_slider)
        self.pop_dsos_threshold_label = QLabel("0.30")
        self.pop_dsos_threshold_label.setFixedWidth(32)
        pop_ctrl_layout.addWidget(self.pop_dsos_threshold_label)
        self.pop_dsos_threshold_slider.valueChanged.connect(
            self._on_dsos_threshold_changed
        )

        self.pop_expand_btn = QPushButton("⛶ Full Screen")
        self.pop_expand_btn.setToolTip("Toggle Full Screen Population View")
        self.pop_expand_btn.setCheckable(True)
        self.pop_expand_btn.clicked.connect(self.toggle_population_fullscreen)
        self.pop_expand_btn.setStyleSheet(
            f"font-weight: bold; background-color: {colors['accent']}; padding: 4px 10px;"
        )
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

        self.pop_mosaic_toolbar = NavigationToolbar2QT(
            self.pop_mosaic_canvas, self.pop_mosaic_widget
        )
        self.pop_mosaic_toolbar.setMaximumHeight(28)
        mosaic_layout.addWidget(self.pop_mosaic_toolbar)
        self.pop_master_splitter.addWidget(self.pop_mosaic_widget)

        # Default to Pan tool and implement mouse-wheel zoom
        self.pop_mosaic_toolbar.pan()

        def on_mosaic_scroll(event):
            if not event.inaxes:
                return
            ax = event.inaxes
            base_scale = 1.2
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            xdata, ydata = event.xdata, event.ydata
            if event.button == "up":
                scale_factor = 1 / base_scale
            elif event.button == "down":
                scale_factor = base_scale
            else:
                scale_factor = 1
            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
            ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * (relx)])
            ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * (rely)])
            ax.figure.canvas.draw_idle()

        self.pop_mosaic_canvas.mpl_connect("scroll_event", on_mosaic_scroll)

        # Click an RF outline to bring that cell up. The pan tool owns the left
        # button here, so the click is resolved on release and only when the
        # mouse has not travelled — otherwise every pan would end in a
        # selection jump.
        self._mosaic_press_xy = None

        def on_mosaic_press(event):
            self._mosaic_press_xy = (
                (event.x, event.y) if event.button == 1 else None
            )

        def on_mosaic_release(event):
            press = self._mosaic_press_xy
            self._mosaic_press_xy = None
            if press is None or event.button != 1 or event.inaxes is None:
                return
            if abs(event.x - press[0]) > 3 or abs(event.y - press[1]) > 3:
                return  # that was a pan, not a click
            self._on_mosaic_rf_clicked(event)

        self.pop_mosaic_canvas.mpl_connect("button_press_event", on_mosaic_press)
        self.pop_mosaic_canvas.mpl_connect("button_release_event", on_mosaic_release)
        self.pop_mosaic_canvas.setToolTip(
            "Click an RF outline to select that cell"
        )

        # 2. Timecourse Panel
        self.pop_timecourse_widget = QWidget()
        tc_layout = QVBoxLayout(self.pop_timecourse_widget)
        tc_layout.setContentsMargins(0, 0, 0, 0)
        tc_hdr = QHBoxLayout()
        self.pop_tc_label = QLabel("Population Dynamics")
        self.pop_tc_label.setStyleSheet(
            f"font-weight:bold; color: {colors['text_primary']};"
        )
        self.pop_timecourse_summary = QLabel("n=0  mean_t2p: N/A  mean_fwhm: N/A")
        tc_hdr.addWidget(self.pop_tc_label)
        tc_hdr.addStretch()
        tc_hdr.addWidget(self.pop_timecourse_summary)
        tc_layout.addLayout(tc_hdr)
        self.pop_timecourse_canvas = MplCanvas(width=6, height=2, dpi=100)
        tc_layout.addWidget(self.pop_timecourse_canvas)
        self.pop_tc_toolbar = make_nav_toolbar(
            self.pop_timecourse_canvas, self.pop_timecourse_widget
        )
        tc_layout.addWidget(self.pop_tc_toolbar)
        self.pop_master_splitter.addWidget(self.pop_timecourse_widget)

        # 3. ACG Panel
        self.pop_acg_widget = QWidget()
        acg_layout = QVBoxLayout(self.pop_acg_widget)
        acg_layout.setContentsMargins(0, 0, 0, 0)
        acg_hdr = QHBoxLayout()
        self.pop_acg_label = QLabel("Population Autocorrelation")
        self.pop_acg_label.setStyleSheet(
            f"font-weight:bold; color: {colors['text_primary']};"
        )
        self.pop_acg_summary = QLabel("n=0")
        acg_hdr.addWidget(self.pop_acg_label)
        acg_hdr.addStretch()
        acg_hdr.addWidget(self.pop_acg_summary)
        acg_layout.addLayout(acg_hdr)
        self.pop_acg_canvas = MplCanvas(width=6, height=2, dpi=100)
        acg_layout.addWidget(self.pop_acg_canvas)
        self.pop_acg_toolbar = make_nav_toolbar(
            self.pop_acg_canvas, self.pop_acg_widget
        )
        acg_layout.addWidget(self.pop_acg_toolbar)
        self.pop_master_splitter.addWidget(self.pop_acg_widget)

        # Standalone "DS/OS Probe Map" panel removed — DS/OS is now shown
        # directly on the Population Receptive Fields plot (arrows/ticks at
        # each classified cell's RF center via _draw_dsos_markers), which
        # doesn't require cross-referencing a separate array-space panel.

        # Add master splitter to layout
        pop_layout.addWidget(self.pop_master_splitter, stretch=1)
        # 3 panes now (RF mosaic, timecourse, ACG) — was 4 before the
        # standalone DS/OS Probe Map pane was removed. Same proportions as
        # before (RF mosaic gets double weight), just one fewer entry.
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
        self.chirp_panel = ChirpPanel(self)
        self.contrast_panel = ContrastPanel(self)
        self.grating_panel = GratingPanel(self)
        self.ei_panel = EIPanel(self)
        self.waveforms_panel = WaveformPanel(self)
        self.raw_panel = RawPanel(self)
        self.sta_panel = STAPanel(self)
        self.umap_panel = UMAPPanel(self)

        # --- Tab Order (Short Labels) ---
        self.analysis_tabs.addTab(self.standard_plots_panel, "Standard")
        self.analysis_tabs.addTab(self.chirp_panel, "Chirp")
        self.analysis_tabs.addTab(self.contrast_panel, "Contrast")
        self.analysis_tabs.addTab(self.grating_panel, "Grating")
        self.analysis_tabs.addTab(self.ei_panel, "EI")
        self.analysis_tabs.addTab(self.sta_panel, "STA")
        self.analysis_tabs.addTab(self.umap_panel, "UMAP")
        self.analysis_tabs.addTab(self.waveforms_panel, "Waveforms")
        self.analysis_tabs.addTab(self.raw_panel, "Raw")

        # --- Main Splitter and Layout ---
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.addWidget(self.left_pane)
        self.main_splitter.addWidget(right_pane)
        self.main_splitter.setSizes([280, 1800 - 280])
        self.last_left_width = 280
        self.main_splitter.setStretchFactor(0, 0)  # Left panel doesn't stretch
        self.main_splitter.setStretchFactor(1, 1)  # Right panel takes remaining space
        self.main_splitter.setHandleWidth(5)
        self.main_splitter.handle(1).mouseDoubleClickEvent = (
            lambda e: self.toggle_sidebar()
        )

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
        self.load_raw_action.setEnabled(False)  # Disabled until KS is loaded

        self.load_vision_action = file_menu.addAction("&Load Vision Files...")
        self.load_vision_action.setEnabled(True)

        self.load_classification_action = file_menu.addAction(
            "&Load Classification File..."
        )
        self.load_classification_action.setEnabled(True)

        self.save_classification_action = file_menu.addAction(
            "Save Classification Text File..."
        )
        self.save_classification_action.setEnabled(False)

        self.save_action = file_menu.addAction("&Save Results...")
        self.save_action.setEnabled(False)

        file_menu.addSeparator()
        self.map_reference_action = file_menu.addAction("Map &Reference Run...")
        self.map_reference_action.setEnabled(False)

        file_menu.addSeparator()
        # Escape hatch for a stale or corrupt physics cache: without it a bad
        # feature_cache.pkl can only be cleared by deleting files by hand.
        self.rebuild_cache_action = file_menu.addAction("Re&build Physics Cache...")
        self.rebuild_cache_action.setEnabled(False)

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
        self.save_classification_action.triggered.connect(
            self.on_save_classification_action
        )
        self.save_action.triggered.connect(self.on_save_action)
        self.map_reference_action.triggered.connect(self.map_reference_run)
        self.rebuild_cache_action.triggered.connect(
            lambda: callbacks.rebuild_physics_cache(self)
        )

        # Connect New Left Panel Buttons
        self.filter_all_btn.clicked.connect(self.reset_views)
        self.tree_view_button.clicked.connect(lambda: self._switch_left_view(0))
        self.table_view_button.clicked.connect(lambda: self._switch_left_view(1))

        self.collapse_all_btn.clicked.connect(self.collapse_tree)
        self.reset_button.clicked.connect(self.reset_views)
        self.analysis_tabs.currentChanged.connect(self.on_tab_changed)

        # Ctrl+F: focus the sidebar search bar
        self.search_shortcut = QShortcut(QKeySequence("Ctrl+F"), self)
        self.search_shortcut.activated.connect(self._focus_search_bar)

        # Connect the raw panel's status and error messages to the status bar
        self.raw_panel.status_message.connect(
            lambda msg: self.status_bar.showMessage(msg, 3000)
        )
        self.raw_panel.error_message.connect(
            lambda msg: self.status_bar.showMessage(msg, 4000)
        )

    def _open_array_calibration(self):
        """Open the Map Image to Array calibration dialog."""
        dlg = ArrayCalibrationDialog(self, data_manager=self.data_manager)
        dlg.transformSaved.connect(self._on_transform_saved)
        dlg.show()

    def _on_transform_saved(self, transform_path: str):
        """Called after user saves a new transform — refresh image overlay."""
        if hasattr(self, "standard_plots_panel"):
            self.standard_plots_panel.refresh_array_image(transform_path=transform_path)
            # re-draw the grid for the currently selected cluster (if any)
            cid = self._get_selected_cluster_id()
            if cid is not None:
                self.standard_plots_panel.update_all(cid)

        # Also push to the EI panel so Photo overlay is immediately available
        if hasattr(self, "ei_panel"):
            self.ei_panel.refresh_array_image(transform_path)

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

            QTimer.singleShot(0, self._draw_population_panel_initial)
        else:
            # hide it
            self.pop_context_widget.hide()
            # collapse the right column completely
            self.right_splitter.setSizes([sum(self.right_splitter.sizes()), 0])

    def _draw_population_panel_initial(self):
        """Draw population context after Qt has laid out the newly shown pane."""
        selected = None
        try:
            selected = self._get_selected_cluster_id()
        except Exception:
            selected = None

        draw_population_rfs_plot(
            main_window=self, selected_cell_id=selected, canvas=self.pop_mosaic_canvas
        )
        callbacks.redraw_population_panels(self)

    def _on_pop_show_ids_toggled(self, state):
        """Force a redraw of the population mosaic when Show IDs is toggled."""
        if hasattr(self, "pop_mosaic_canvas"):
            # Invalidate the hot-swap cache to force a full background rebuild
            if hasattr(self.pop_mosaic_canvas, "_pop_plot_state"):
                del self.pop_mosaic_canvas._pop_plot_state

            selected = None
            try:
                selected = self._get_selected_cluster_id()
            except Exception:
                pass

            draw_population_rfs_plot(
                main_window=self,
                selected_cell_id=selected,
                canvas=self.pop_mosaic_canvas,
            )

    def _on_dsos_threshold_changed(self, slider_value):
        """
        DS/OS threshold slider moved. Updates self.dsos_threshold (read by
        population_panel.py's _draw_dsos_markers on every redraw) and
        forces a full redraw so the change is visible immediately, using
        the same cache-invalidation approach as _on_pop_show_ids_toggled —
        DS/OS markers are already always redrawn fresh regardless of cache
        state (see _draw_dsos_markers docstring), but invalidating
        _pop_plot_state here too guarantees a clean full redraw rather than
        depending on the hot-swap path's DS/OS-always-fresh behavior for
        this specific interaction.
        """
        self.dsos_threshold = slider_value / 100.0
        self.pop_dsos_threshold_label.setText(f"{self.dsos_threshold:.2f}")

        # The grating panel label used to stay at the 0.3 default until the
        # user re-clicked the cell. Refresh it if a cluster is already up.
        gp = getattr(self, "grating_panel", None)
        cid = getattr(gp, "_current_cluster_id", None) if gp is not None else None
        if gp is not None and cid is not None:
            gp.update_all(cid)

        if hasattr(self, "pop_mosaic_canvas"):
            if hasattr(self.pop_mosaic_canvas, "_pop_plot_state"):
                del self.pop_mosaic_canvas._pop_plot_state

            selected = None
            try:
                selected = self._get_selected_cluster_id()
            except Exception:
                pass

            draw_population_rfs_plot(
                main_window=self,
                selected_cell_id=selected,
                canvas=self.pop_mosaic_canvas,
            )

    def _on_mosaic_rf_clicked(self, event):
        """Resolve a click on the population mosaic to a cell and select it."""
        if self.data_manager is None:
            return
        vision_id = rf_vision_id_at(event.inaxes, event.xdata, event.ydata)
        if vision_id is None:
            return

        cluster_id = self.data_manager.get_cluster_id_for_vision(vision_id)
        if self.focus_cluster(cluster_id):
            self.status_bar.showMessage(
                f"Selected cell {cluster_id} (Vision {vision_id}) from the RF mosaic.",
                4000,
            )
        else:
            self.status_bar.showMessage(
                f"Vision {vision_id} has an RF here but is not in the cell list.",
                4000,
            )

    def focus_cluster(self, cluster_id):
        """Select *cluster_id* as if the user had clicked it in the sidebar.

        The entry point for plots that can identify a cell but do not own the
        selection — clicking an RF outline in a mosaic, for one. It drives
        whichever view is currently active, because the selection handler reads
        the active view; that same handler then mirrors the choice into the
        other view and refreshes every detail panel through its normal path.

        Returns True if the cell was found.
        """
        if cluster_id is None:
            return False
        cluster_id = int(cluster_id)

        if self.view_stack.currentIndex() == 1:
            return self._select_cluster_in_table(cluster_id)
        return self._select_cluster_in_tree(cluster_id)

    def _select_cluster_in_tree(self, cluster_id) -> bool:
        if self.tree_model is None:
            return False
        matches = self.tree_model.match(
            self.tree_model.index(0, 0),
            Qt.ItemDataRole.UserRole,
            cluster_id,
            1,
            Qt.MatchExactly | Qt.MatchRecursive,
        )
        if not matches:
            logger.debug("focus_cluster: cluster %s not in the tree", cluster_id)
            return False

        index = matches[0]
        self.tree_view.setCurrentIndex(index)
        self.tree_view.selectionModel().select(
            index, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows
        )
        self.tree_view.scrollTo(index, QAbstractItemView.ScrollHint.PositionAtCenter)
        return True

    def _select_cluster_in_table(self, cluster_id) -> bool:
        view_model = self.table_view.model()
        if view_model is None:
            return False
        # The view holds the sort/filter proxy; the frame lives on the source.
        source = (
            view_model.sourceModel()
            if hasattr(view_model, "sourceModel")
            else view_model
        )
        if source is None or not hasattr(source, "_data"):
            return False

        df = source._data
        if "cluster_id" not in df or cluster_id not in df["cluster_id"].values:
            return False

        row_labels = df.index[df["cluster_id"] == cluster_id].tolist()
        if not row_labels:
            return False
        source_index = source.index(df.index.get_loc(row_labels[0]), 0)
        view_index = (
            view_model.mapFromSource(source_index)
            if view_model is not source
            else source_index
        )
        if not view_index.isValid():
            # Filtered out of the table right now — fall back to the tree.
            return self._select_cluster_in_tree(cluster_id)

        self.table_view.setCurrentIndex(view_index)
        self.table_view.selectionModel().select(
            view_index, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows
        )
        self.table_view.scrollTo(
            view_index, QAbstractItemView.ScrollHint.PositionAtCenter
        )
        return True

    def _switch_left_view(self, index):
        """Switches between the tree (0) and table (1) views in the left pane."""
        self.view_stack.setCurrentIndex(index)
        # Re-apply any active search query to whichever view just became active
        if hasattr(self, "cluster_search_bar"):
            self._filter_sidebar(self.cluster_search_bar.text())

    # --- Helper Method ---
    def _get_selected_cluster_id(self):
        """Returns the cluster_id of the currently selected item from the active view."""
        current_view_index = self.view_stack.currentIndex()

        # Case 1: Tree View is active
        if current_view_index == 0:
            selection_model = self.tree_view.selectionModel()
            if selection_model is None or not selection_model.hasSelection():
                return None

            # Prefer the focused row over the topmost selected one: under
            # ExtendedSelection those differ, and the detail panels should
            # follow the cell the user just clicked, not whichever sits
            # highest in the tree.
            index = self.tree_view.currentIndex()
            if not index.isValid() or not selection_model.isSelected(index):
                index = selection_model.selectedIndexes()[0]
            item = tree_item_from_index(self.tree_model, index)
            if item is None:
                return None

            # Only leaf nodes (cells) have a cluster ID stored. Groups will
            # return None.
            cluster_id = item.data(Qt.ItemDataRole.UserRole)
            if cluster_id is None:
                return None
            return int(cluster_id)

        # Case 2: Table View is active
        elif current_view_index == 1:
            selection_model = self.table_view.selectionModel()
            if (
                selection_model is None
                or not selection_model.hasSelection()
                or self.main_cluster_model is None
            ):
                return None

            selected_row = selection_model.selectedIndexes()[0].row()

            # Check if the model has mapToSource method (for proxy models)
            model = self.table_view.model()
            if hasattr(model, "mapToSource"):
                # Map the view's sorted/filtered row back to the source model row
                source_index = model.mapToSource(model.index(selected_row, 0))
                cluster_id = model.sourceModel()._dataframe.iloc[source_index.row()][
                    "cluster_id"
                ]
            else:
                # Bare model (no proxy), use the row directly
                cluster_id = model._dataframe.iloc[selected_row]["cluster_id"]
            try:
                return int(cluster_id)
            except (TypeError, ValueError):
                return None

        return None

    def on_save_classification_action(self):
        """Wrapper to call the callback function."""
        callbacks.save_classification_to_file(self)

    def _get_selected_tree_items(self):
        """Every distinct item currently selected in the tree, in tree order.

        ``selectedIndexes()`` yields one index per column; the tree is single
        -column so that is already one per row, but de-duplicating by index
        keeps this correct if a column is ever added.
        """
        selection_model = self.tree_view.selectionModel()
        if selection_model is None:
            return []

        seen = set()
        items = []
        for index in selection_model.selectedIndexes():
            if index.column() != 0:
                continue
            key = (index.row(), index.internalId())
            if key in seen:
                continue
            seen.add(key)
            item = self.tree_model.itemFromIndex(index)
            if item is not None:
                items.append(item)
        return items

    def _tree_action_items(self, clicked_item=None):
        """Items a tree action should apply to.

        Follows the file-manager convention: act on the whole selection when
        the right-clicked row is part of it, otherwise act on just that row.
        """
        selected = self._get_selected_tree_items()
        if clicked_item is None:
            return selected
        if any(item is clicked_item for item in selected):
            return selected
        return [clicked_item]

    def move_tree_selection_to_trash(self):
        """Send the current tree selection to Trash (also bound to Delete)."""
        if self.view_stack.currentIndex() != 0:
            return
        items = self._get_selected_tree_items()
        if not items:
            return
        callbacks.move_items_to_trash(self, items)

    def _get_selected_table_cluster_ids(self):
        """Cluster IDs for every row selected in the table view.

        Rows must be mapped back through the sort/filter proxy — the view's row
        order is not the DataFrame's once a column has been sorted.
        """
        selection_model = self.table_view.selectionModel()
        model = self.table_view.model()
        if selection_model is None or model is None:
            return []

        source = model.sourceModel() if hasattr(model, "sourceModel") else model
        df = getattr(source, "_dataframe", None)
        if df is None or "cluster_id" not in df.columns:
            return []

        col = df.columns.get_loc("cluster_id")
        ids = []
        seen = set()
        for index in selection_model.selectedRows():
            src = (
                model.mapToSource(index) if hasattr(model, "mapToSource") else index
            )
            row = src.row()
            if row < 0 or row >= len(df):
                continue
            cid = int(df.iloc[row, col])
            if cid not in seen:
                seen.add(cid)
                ids.append(cid)
        return ids

    def move_table_selection_to_trash(self):
        """Send the current table selection to Trash (also bound to Delete)."""
        if self.view_stack.currentIndex() != 1:
            return
        cluster_ids = self._get_selected_table_cluster_ids()
        if not cluster_ids:
            return
        callbacks.move_cluster_ids_to_trash(self, cluster_ids)

    def _clicked_table_column_name(self, position):
        """Name of the dataframe column under *position*, or None."""
        index = self.table_view.indexAt(position)
        if not index.isValid():
            return None
        model = self.table_view.model()
        src = model.mapToSource(index) if hasattr(model, "mapToSource") else index
        source = model.sourceModel() if hasattr(model, "sourceModel") else model
        df = getattr(source, "_dataframe", None)
        if df is None or src.column() >= len(df.columns):
            return None
        return str(df.columns[src.column()])

    def show_chirp_qi_details(self, cluster_id):
        """Explain one cell's chirp QI: how many trials, and at which levels.

        The single number in the column hides two things that decide whether it
        can be read at all — the repeat count behind it, and the fact that a
        concatenated sort's chirp file spans several runs at different light
        levels, which are scored separately here.
        """
        dm = self.data_manager
        details = dm.chirp_qi_details(cluster_id) if dm is not None else None
        if not details:
            QMessageBox.information(
                self, "Chirp QI",
                f"Cell {cluster_id} has no chirp response in the loaded file.")
            return

        n = details["n_trials_per_run"]
        ok = details["trustworthy"]
        lines = [
            f"<b>Cell {cluster_id} — chirp quality index</b>",
            "",
            f"Trials behind each QI: <b>{n}</b> "
            f"({'enough' if ok else 'TOO FEW'}; "
            f"{details['min_repeats']} needed)",
        ]
        if details["n_trials_total"] and details["n_trials_total"] != n:
            lines.append(
                f"The file holds {details['n_trials_total']} trials in total, but they "
                f"span {len(details['source_runs'])} runs at different light levels, "
                f"so each run is scored on its own {n}.")
        lines.append(
            f"Computed at {details['qi_bin_ms']:.0f} ms bins"
            + (f" (file is binned at {details['file_bin_ms']:.0f} ms)"
               if details["file_bin_ms"] and details["file_bin_ms"] != details["qi_bin_ms"]
               else "")
            + ". The QI moves with bin width, so it is rebinned to a fixed "
              "width to stay comparable between files.")
        if not ok:
            lines.append(
                "<span style='color:#F08080'>With this few repeats the index is a "
                "ratio of two variances estimated from almost nothing — it stops "
                "tracking anything real and should not be used as a "
                "criterion.</span>")

        lines += ["", "<b>Per light level</b> (QI, and rho = mean pairwise trial "
                      "correlation, which is 0 for noise at any repeat count):"]
        for r in details["per_run"]:
            mark = " &nbsp;&larr; reported" if r["run"] == details["best_run"] else ""
            level = r["light_level"] or "light level unknown"
            if r.get("gated"):
                lines.append(
                    f"&nbsp;&nbsp;{r['run']} ({level}): "
                    f"<span style='color:#F0C060'>not measured</span> — "
                    f"{r['spikes_per_trial']:.1f} spikes/trial, under the "
                    f"{details['min_spikes_per_trial']:.0f} needed")
            else:
                lines.append(f"&nbsp;&nbsp;{r['run']} ({level}): QI "
                             f"<b>{r['qi']:.3f}</b>, rho {r['rho']:+.3f}, "
                             f"{r['spikes_per_trial']:.0f} spikes/trial{mark}")
        if len(details["per_run"]) > 1:
            lines.append("")
            lines.append("The column reports the best level, so a cell that is "
                         "reliable at one light level and silent at another is "
                         "not hidden by averaging.")

        box = QMessageBox(self)
        box.setWindowTitle(f"Chirp QI — cell {cluster_id}")
        box.setTextFormat(Qt.RichText)
        box.setText("<br>".join(lines))
        box.exec_()

    def open_table_context_menu(self, position):
        clicked_column = self._clicked_table_column_name(position)
        cluster_ids = self._get_selected_table_cluster_ids()
        if not cluster_ids:
            index = self.table_view.indexAt(position)
            if not index.isValid():
                return
            self.table_view.selectRow(index.row())
            cluster_ids = self._get_selected_table_cluster_ids()
            if not cluster_ids:
                return

        n = len(cluster_ids)
        noun = f"{n} cell{'s' if n != 1 else ''}"

        menu = QMenu()
        move_menu = menu.addMenu(f"Move {noun} to")
        move_actions = {}
        for path, group_item in callbacks.collect_group_items(self):
            act = move_menu.addAction(path)
            move_actions[act] = group_item
        if move_actions:
            move_menu.addSeparator()
        move_new_action = move_menu.addAction("New group…")

        trash_action = menu.addAction(f"Move {noun} to Trash")
        menu.addSeparator()
        group_action = menu.addAction(f"Group {noun} into new group…")
        feature_action = menu.addAction("Feature Extraction")

        # Offered on the QI cell itself, where the question "how many trials is
        # this actually based on?" comes up.
        qi_action = None
        if clicked_column == "chirp_qi":
            menu.addSeparator()
            qi_action = menu.addAction("Chirp QI details…")

        action = menu.exec(self.table_view.viewport().mapToGlobal(position))
        if action is None:
            return

        if qi_action is not None and action == qi_action:
            self.show_chirp_qi_details(cluster_ids[0])
        elif action in move_actions:
            callbacks.move_cluster_ids_to_group(
                self, cluster_ids, move_actions[action]
            )
        elif action == move_new_action:
            name, ok = QInputDialog.getText(self, "New Group", "Enter group name:")
            if ok and name:
                callbacks.group_clusters_in_tree(self, cluster_ids, name)
        elif action == trash_action:
            callbacks.move_cluster_ids_to_trash(self, cluster_ids)
        elif action == group_action:
            name, ok = QInputDialog.getText(self, "New Group", "Enter group name:")
            if ok and name:
                callbacks.group_clusters_in_tree(self, cluster_ids, name)
        elif action == feature_action:
            callbacks.feature_extraction(self, cluster_ids)

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
                item = tree_item_from_index(self.tree_model, selection[0])
                if item and item.data(Qt.ItemDataRole.UserRole) is None:  # It's a group
                    return self._get_group_cluster_ids(item)

        # Case 2: A cell is selected (tree or table). Population is the
        # innermost tree folder that cell sits in — not KSLabel, not the
        # whole run. Table mode uses the same walk; Qt match misses numpy ids.
        if cluster_id is not None:
            item = find_cell_item(self.tree_model, cluster_id)
            if item is None:
                return [int(cluster_id)]
            parent_item = item.parent()
            if parent_item is None or parent_item is self.tree_model.invisibleRootItem():
                return [int(cluster_id)]
            return self._get_group_cluster_ids(parent_item)

        return []

    def _install_table_proxy(self, model):
        """Wrap *model* in the search/sort proxy and install it on the table.

        Single owner of the proxy's configuration. setup_table_model and
        refresh_table_model both go through here — they used to configure a
        proxy each, which is how refresh_table_model ended up without the sort
        role and silently reverted "# Spikes" to string ordering.

        The proxy also drops anything sitting in Trash, so discarded units stop
        appearing in the view curation is done in while staying recoverable
        from the tree.
        """
        proxy = HiddenIdFilterProxyModel(
            lambda: callbacks.get_trashed_cluster_ids(self), self
        )
        proxy.setSourceModel(model)
        proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        proxy.setFilterKeyColumn(-1)  # search all columns
        # Sort on the raw values, not the formatted text — otherwise "# Spikes"
        # sorts lexicographically and 1000 lands before 999. Filtering stays on
        # DisplayRole so the search bar still matches what is on screen.
        proxy.setSortRole(PandasModel.SORT_ROLE)
        proxy.refresh_hidden()

        self.table_view.setModel(proxy)
        self.table_view.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
        self.table_view.verticalHeader().setVisible(False)
        try:
            self.table_view.selectionModel().selectionChanged.disconnect(
                self.on_view_selection_changed
            )
        except (TypeError, RuntimeError):
            pass
        self.table_view.selectionModel().selectionChanged.connect(
            self.on_view_selection_changed
        )

        # Re-apply any active search query to the new proxy
        if hasattr(self, "cluster_search_bar") and self.cluster_search_bar.text():
            proxy.setFilterFixedString(self.cluster_search_bar.text())

        return proxy

    def _apply_chirp_qi_context(self, model, header_labels):
        """Label and flag the chirp QI column according to its repeat count.

        Both table-building paths call this, so the warning cannot end up on
        one and not the other. When the run has too few repeats the header
        carries the count and the model renders the column in the warning
        colour — see HighlightStatusPandasModel.set_chirp_qi_context.
        """
        dm = self.data_manager
        n_repeats = getattr(dm, "chirp_n_repeats", None) if dm is not None else None
        trustworthy = True
        if dm is not None and hasattr(dm, "chirp_qi_is_trustworthy"):
            trustworthy = dm.chirp_qi_is_trustworthy()

        header_labels["chirp_qi"] = (
            f"Chirp QI (n={n_repeats})" if trustworthy and n_repeats
            else "Chirp QI" if trustworthy else f"Chirp QI (n={n_repeats or '?'})"
        )
        if hasattr(model, "set_chirp_qi_context"):
            from ..analysis.constants import CHIRP_MIN_REPEATS_FOR_QI
            model.set_chirp_qi_context(
                n_repeats, trustworthy, CHIRP_MIN_REPEATS_FOR_QI)

    def setup_table_model(self, model):
        """Sets up the table view model, wrapping it in a search proxy."""
        if hasattr(model, "update_colors"):
            model.update_colors(self.get_current_colors())

        proxy = self._install_table_proxy(model)

        # Column header labels override (keeps internal df names intact)
        HEADER_LABELS = {
            "cluster_id": "ID",
            "n_spikes": "# Spikes",
            "best_chan": "Ch",
            "group": "Folder",
            "chirp_onoff": "ON/OFF",
            "sta_snr": "STA SNR",
            "KSLabel": "KS Label",
            "isi_violations_pct": "ISI Viol%",
            "contam_pct": "Contam%",
            "amp_median": "Amp (µV)",
            "firing_rate_hz": "FR (Hz)",
            "template_amp": "Tpl Amp",
            "max_dup_r": "Max Dup R",
            "potential_dups": "Dup?",
            "cell_type": "Type",
            "status": "Status",
            "x_um": "X (µm)",
            "y_um": "Y (µm)",
            "set": "Set",
        }
        self._apply_chirp_qi_context(model, HEADER_LABELS)

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
            _orig_headerData, model._header_overrides, df_cols
        )

        for col_name, label in HEADER_LABELS.items():
            model._header_overrides[col_name] = label

        # Allow drag-to-reorder; only apply the default ordering on first load
        header = self.table_view.horizontalHeader()
        header.setSectionsMovable(True)

        if not getattr(self, "_table_columns_initialized", False):
            self._apply_default_column_order(header, df_cols, col_index)
            self._table_columns_initialized = True

        self._resize_table_columns_fast()

    def _resize_table_columns_fast(self, sample_rows=24):
        """Size columns from the header and a small row sample.

        ``QTableView.resizeColumnsToContents`` asks the model for every
        cell. On a 300+ cluster run that is several thousand pandas
        lookups and a multi-second hitch on the GUI thread after
        Kilosort has already finished.
        """
        view = self.table_view
        model = view.model()
        if model is None:
            return
        fm = view.fontMetrics()
        n_rows = min(int(model.rowCount()), int(sample_rows))
        pad = 20
        max_w = 240
        min_w = 36
        for col in range(int(model.columnCount())):
            header = model.headerData(col, Qt.Horizontal, Qt.DisplayRole)
            width = fm.horizontalAdvance(str(header if header is not None else "")) + pad
            for row in range(n_rows):
                text = model.data(model.index(row, col), Qt.DisplayRole)
                if text is None:
                    continue
                width = max(width, fm.horizontalAdvance(str(text)) + pad)
            view.setColumnWidth(col, max(min_w, min(width, max_w)))

    def _apply_default_column_order(self, header, df_cols, col_index):
        """Apply the desired default visual column order. Called only once."""
        ORDERED_COLS = [
            "cluster_id",  # shown as "ID" — thin
            "n_spikes",
            "best_chan",
            "group",  # innermost folder, immediately before the KS label
            "KSLabel",
            "isi_violations_pct",
            "contam_pct",
            "chirp_qi",  # sits with the other per-unit quality metrics
            "chirp_onoff",  # polarity, beside the quality index it pairs with
            "sta_snr",      # the other per-unit quality metric
            "amp_median",
            "firing_rate_hz",
            "template_amp",
            "max_dup_r",
            "potential_dups",
            "cell_type",
            "status",
            "x_um",
            "y_um",
            "set",
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
        # Preserve the user's current visual column order across the rebuild.
        # The view may have a proxy installed, so unwrap to the source model.
        header = self.table_view.horizontalHeader()
        # Installing a new proxy drops the active sort, so a background refresh
        # would quietly reshuffle rows out from under the user. Capture the sort
        # by *column name* — the rebuild can change column indices.
        old_sort_col = None
        old_sort_order = Qt.AscendingOrder
        old_view_model = self.table_view.model()
        if old_view_model is not None:
            # Unwrap proxy if present
            old_source = (
                old_view_model.sourceModel()
                if hasattr(old_view_model, "sourceModel")
                else old_view_model
            )
            if hasattr(old_source, "_dataframe"):
                old_cols = list(old_source._dataframe.columns)
                old_visual_order = [
                    old_cols[header.logicalIndex(v)]
                    for v in range(header.count())
                    if header.logicalIndex(v) < len(old_cols)
                ]
                section = header.sortIndicatorSection()
                if 0 <= section < len(old_cols):
                    old_sort_col = old_cols[section]
                    old_sort_order = header.sortIndicatorOrder()
            else:
                old_visual_order = None
        else:
            old_visual_order = None

        model = HighlightStatusPandasModel(df)
        model.update_colors(self.get_current_colors())
        self.main_cluster_model = model

        # Wrap in proxy for search filtering, then install on the view
        proxy = self._install_table_proxy(model)

        # Re-apply any active search filter to the new proxy
        if hasattr(self, "cluster_search_bar") and self.cluster_search_bar.text():
            proxy.setFilterFixedString(self.cluster_search_bar.text())

        # Re-apply header labels
        new_df_cols = list(df.columns)
        HEADER_LABELS = {
            "cluster_id": "ID",
            "n_spikes": "# Spikes",
            "best_chan": "Ch",
            "group": "Folder",
            "chirp_onoff": "ON/OFF",
            "sta_snr": "STA SNR",
            "KSLabel": "KS Label",
            "isi_violations_pct": "ISI Viol%",
            "contam_pct": "Contam%",
            "amp_median": "Amp (µV)",
            "firing_rate_hz": "FR (Hz)",
            "template_amp": "Tpl Amp",
            "max_dup_r": "Max Dup R",
            "potential_dups": "Dup?",
            "cell_type": "Type",
            "status": "Status",
            "x_um": "X (µm)",
            "y_um": "Y (µm)",
            "set": "Set",
        }
        self._apply_chirp_qi_context(model, HEADER_LABELS)
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
            self._apply_default_column_order(new_header, new_df_cols, new_col_index)

        # Restore the sort the user had before the rebuild, by column name so a
        # changed column layout cannot silently sort a different column.
        if old_sort_col is not None and old_sort_col in new_col_index:
            self.table_view.sortByColumn(
                new_col_index[old_sort_col], old_sort_order
            )

        self._resize_table_columns_fast()
        self._refresh_tree_channels()

    def _refresh_tree_channels(self):
        """Write best_chan into tree column 2 without rebuilding the tree."""
        dm = self.data_manager
        if dm is None or dm.cluster_df is None or "best_chan" not in dm.cluster_df.columns:
            return
        model = self.tree_model
        if model is None:
            return
        chan_map = dict(
            zip(
                dm.cluster_df["cluster_id"].astype(int),
                dm.cluster_df["best_chan"],
            )
        )

        def visit(parent):
            for row in range(parent.rowCount()):
                item = parent.child(row, TREE_COL_ID)
                if item is None:
                    continue
                cid = item.data(Qt.ItemDataRole.UserRole)
                if cid is not None:
                    ch_item = parent.child(row, TREE_COL_CH)
                    if ch_item is not None:
                        set_channel_item(ch_item, chan_map.get(int(cid)))
                if item.hasChildren():
                    visit(item)

        visit(model.invisibleRootItem())

    def _configure_tree_columns(self):
        model = self.tree_view.model()
        if model is None:
            return
        if model.columnCount() < 3:
            model.setColumnCount(3)
        model.setHorizontalHeaderLabels(list(TREE_HEADERS))
        header = self.tree_view.header()
        header.setVisible(True)
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(1, TREE_SPIKES_WIDTH)
        header.resizeSection(2, TREE_CH_WIDTH)
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        # Manual sort on header click. setSortingEnabled(True) re-sorts on
        # every insert and fights drag-and-drop.
        self.tree_view.setSortingEnabled(False)
        if not getattr(self, "_tree_header_sort_wired", False):
            header.sectionClicked.connect(self._on_tree_header_clicked)
            self._tree_header_sort_wired = True
            self._tree_sort_section = -1
            self._tree_sort_order = Qt.SortOrder.AscendingOrder

    def _on_tree_header_clicked(self, section):
        header = self.tree_view.header()
        if getattr(self, "_tree_sort_section", -1) == section:
            order = (
                Qt.SortOrder.DescendingOrder
                if self._tree_sort_order == Qt.SortOrder.AscendingOrder
                else Qt.SortOrder.AscendingOrder
            )
        else:
            order = Qt.SortOrder.AscendingOrder
        self._tree_sort_section = section
        self._tree_sort_order = order
        header.setSortIndicator(section, order)
        if self.tree_model is not None:
            self.tree_model.sort(section, order)

    def setup_tree_model(self, model):
        """Sets up the tree view model and connects the selection changed signal."""
        self.tree_view.setModel(model)
        self._configure_tree_columns()
        self._apply_tree_item_theme(self.get_current_colors())
        try:
            self.tree_view.selectionModel().selectionChanged.disconnect(
                self.on_view_selection_changed
            )
        except (TypeError, RuntimeError):
            pass
        self.tree_view.selectionModel().selectionChanged.connect(
            self.on_view_selection_changed
        )
        self._watch_tree_for_trash_changes(model)

    def _watch_tree_for_trash_changes(self, model):
        """Keep the table's Trash filter in step with the tree.

        Hooked to the model's structural signals rather than to the individual
        move helpers: cells reach Trash by drag-and-drop, by the Delete key, by
        either context menu and by loading a saved tree, and a filter that
        misses one of those routes leaves discarded cells visible with no way to
        tell why.
        """
        for signal in (model.rowsInserted, model.rowsRemoved, model.modelReset):
            try:
                signal.disconnect(self._schedule_trash_filter_refresh)
            except (TypeError, RuntimeError):
                pass
            signal.connect(self._schedule_trash_filter_refresh)

    def _schedule_trash_filter_refresh(self, *_args):
        """Coalesce a burst of tree edits into one filter refresh.

        Moving a 40-cell folder emits a signal per row, and each refresh walks
        the whole Trash subtree; collapsing them onto the next event-loop turn
        makes the cost independent of how many cells moved.
        """
        if getattr(self, "_trash_refresh_pending", False):
            return
        self._trash_refresh_pending = True
        QTimer.singleShot(0, self._refresh_trash_filter)

    def _refresh_trash_filter(self):
        self._trash_refresh_pending = False
        proxy = self.table_view.model()
        if isinstance(proxy, HiddenIdFilterProxyModel):
            proxy.refresh_hidden()

    # ── Sidebar Search ────────────────────────────────────────────────────────

    def _focus_search_bar(self):
        """Ctrl+F: move focus to the sidebar search bar and select all text."""
        self.cluster_search_bar.setFocus()
        self.cluster_search_bar.selectAll()

    def _filter_sidebar(self, text: str):
        """
        Dispatch the current search query to whichever view is active.
        Also called from _switch_left_view so a pending query is applied
        immediately when the user toggles between Tree and Table.
        """
        query = text.strip()
        if self.view_stack.currentIndex() == 0:
            self._filter_tree(query)
        else:
            self._filter_table(query)

    def _filter_tree(self, query: str):
        """
        Show only tree leaf items whose display text contains `query`
        (case-insensitive substring). Clears to collapsed default when empty.
        """
        root = self.tree_model.invisibleRootItem()
        self._apply_tree_filter_recursive(root, query.lower())
        if not query:
            # Empty query: restore all items then collapse to default state
            self.tree_view.collapseAll()

    def _apply_tree_filter_recursive(self, parent_item, query: str) -> bool:
        """
        Recursively show/hide tree items. Returns True if any child is visible.
        Group nodes are visible iff at least one descendant leaf matches.
        """
        any_visible = False
        for row in range(parent_item.rowCount()):
            child = parent_item.child(row)
            index = self.tree_model.indexFromItem(child)

            if callbacks.is_group_item(child):
                child_matched = self._apply_tree_filter_recursive(child, query)
                self.tree_view.setRowHidden(
                    index.row(), index.parent(), not child_matched
                )
                if child_matched:
                    self.tree_view.setExpanded(index, True)
                    any_visible = True
            else:
                texts = [child.text().lower()]
                for col in (1, 2):
                    sib = parent_item.child(row, col)
                    if sib is not None:
                        texts.append(sib.text().lower())
                visible = (not query) or any(query in text for text in texts)
                self.tree_view.setRowHidden(index.row(), index.parent(), not visible)
                if visible:
                    any_visible = True

        return any_visible

    def _filter_table(self, query: str):
        """
        Apply a substring filter to the table's QSortFilterProxyModel.
        filterKeyColumn=-1 means all columns are searched.
        setFilterFixedString does a case-insensitive substring match
        (respecting the proxy's filterCaseSensitivity setting).
        """
        model = self.table_view.model()
        if model is None or not hasattr(model, "setFilterFixedString"):
            # Proxy not yet installed (before first data load) — nothing to do
            return
        model.setFilterFixedString(query)

    # ── End Sidebar Search ────────────────────────────────────────────────────

    # --- Methods to bridge UI signals to callback functions ---
    def load_directory(self, kilosort_dir=None, dat_file=None):
        callbacks.load_directory(self, kilosort_dir, dat_file)

    def load_vision_directory(self):
        callbacks.load_vision_directory(self)

    def map_reference_run(self):
        callbacks.map_reference_run(self)

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
                if hasattr(model, "_data"):
                    df = model._data
                    if cluster_id in df["cluster_id"].values:
                        row_indices = df.index[df["cluster_id"] == cluster_id].tolist()
                        if row_indices:
                            model_row = df.index.get_loc(row_indices[0])
                            source_index = model.index(model_row, 0)
                            # This assumes the model is a proxy model if
                            # sorting is enabled
                            view_index = (
                                model.mapFromSource(source_index)
                                if hasattr(model, "mapFromSource")
                                else source_index
                            )
                            if view_index.isValid():
                                self.table_view.selectionModel().select(
                                    view_index,
                                    QItemSelectionModel.ClearAndSelect
                                    | QItemSelectionModel.Rows,
                                )
                                self.table_view.scrollTo(
                                    view_index,
                                    QAbstractItemView.ScrollHint.PositionAtCenter,
                                )

            # Sync from Table to Tree
            elif sender == self.table_view.selectionModel():
                # Use Qt's highly optimized, built-in recursive match
                start_index = self.tree_model.index(0, 0)
                matches = self.tree_model.match(
                    start_index,
                    Qt.ItemDataRole.UserRole,  # What role to search (Cluster ID)
                    cluster_id,  # What value to look for
                    1,  # Stop after 1 match is found
                    Qt.MatchExactly
                    | Qt.MatchRecursive,  # Tell it to search sub-folders
                )

                if matches:
                    index = matches[0]
                    self.tree_view.selectionModel().select(
                        index,
                        QItemSelectionModel.ClearAndSelect
                        | QItemSelectionModel.Rows,
                    )
                    self.tree_view.scrollTo(
                        index, QAbstractItemView.ScrollHint.PositionAtCenter
                    )

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
        logger.debug(f"on_similarity_selection_changed: main_cluster = {main_cluster}")
        logger.debug(
            f"on_similarity_selection_changed: clusters_to_plot = {clusters_to_plot}"
        )

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
        duplicate_ids = sdf[sdf["status"] == "Duplicate"]["cluster_id"].tolist()
        duplicate_ids = set(duplicate_ids)
        self._apply_tree_item_theme(colors)

        def visit(item):
            cluster_id = item.data(Qt.ItemDataRole.UserRole)
            if cluster_id in duplicate_ids:
                item.setForeground(QColor(colors["status_noise_text"]))
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

    def collapse_tree(self):
        if getattr(self, "tree_view", None) is not None:
            self.tree_view.collapseAll()

    def reset_views(self):
        callbacks.reset_views(self)

    def select_sta_view(self, view_type, _force_animation=False):
        """Select the STA view to display."""
        self.current_sta_view = view_type

        # Update button text based on current view
        if hasattr(
            self.sta_panel, "sta_animation_button"
        ):  # Check if STAPanel is initialized
            if view_type == "rf":
                self.sta_panel.sta_animation_button.setText("Play Animation")
            elif view_type == "animation":
                self.sta_panel.sta_animation_button.setText("Pause Animation")

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
        index = self.tree_view.indexAt(position)
        item = tree_item_from_index(self.tree_model, index)
        if not item:
            return

        # Act on the whole selection when the clicked row is part of it.
        targets = self._tree_action_items(item)
        n_cells = sum(1 for t in targets if not callbacks.is_group_item(t))
        n_groups = len(targets) - n_cells
        cluster_ids = []
        for t in targets:
            cluster_ids.extend(callbacks.iter_cluster_ids(t))

        in_trash = callbacks.items_are_in_trash(self, targets)

        if len(targets) == 1:
            noun = "group" if n_groups else "cell"
        else:
            parts = []
            if n_cells:
                parts.append(f"{n_cells} cell{'s' if n_cells != 1 else ''}")
            if n_groups:
                parts.append(f"{n_groups} group{'s' if n_groups != 1 else ''}")
            noun = " + ".join(parts)

        menu = QMenu()

        # ── Move to… ─────────────────────────────────────────────────────────
        move_menu = menu.addMenu(f"Move {noun} to")
        move_actions = {}
        for path, group_item in callbacks.collect_group_items(self, exclude=targets):
            act = move_menu.addAction(path)
            move_actions[act] = group_item
        if move_actions:
            move_menu.addSeparator()
        move_top_action = move_menu.addAction("Top level")
        move_new_action = move_menu.addAction("New group…")

        # ── Trash ────────────────────────────────────────────────────────────
        if in_trash:
            trash_action = None
            restore_action = menu.addAction(f"Restore {noun} from Trash")
        else:
            trash_action = menu.addAction(f"Move {noun} to Trash")
            restore_action = None

        menu.addSeparator()

        # ── Actions on the selected cells ────────────────────────────────────
        group_action = feature_action = None
        if cluster_ids:
            group_action = menu.addAction(
                f"Group {len(cluster_ids)} cell"
                f"{'s' if len(cluster_ids) != 1 else ''} into new group…"
            )
            feature_action = menu.addAction("Feature Extraction")

        # ── Folder-only actions (single folder clicked) ──────────────────────
        add_group_action = rename_action = flatten_action = delete_action = None
        export_cells_action = export_mean_action = None
        if len(targets) == 1 and callbacks.is_group_item(item):
            menu.addSeparator()

            # Export is folder-scoped because both output shapes are named
            # after the folder: the per-cell filenames carry it, and the
            # overlay is a statement about that group as a whole.
            export_menu = menu.addMenu("Export plots")
            export_cells_action = export_menu.addAction("One file per cell…")
            export_mean_action = export_menu.addAction("Group mean overlay…")
            if not cluster_ids:
                export_menu.setEnabled(False)

            add_group_action = menu.addAction("Add New Group")
            rename_action = menu.addAction("Rename")
            flatten_action = menu.addAction("Flatten Group (Remove Sub-folders)")
            delete_action = menu.addAction("Delete Group (Keep Units)")

        action = menu.exec(self.tree_view.viewport().mapToGlobal(position))
        if action is None:
            return

        if action in move_actions:
            callbacks.move_items_to_group(self, targets, move_actions[action])
        elif action == move_top_action:
            callbacks.move_items_to_group(
                self, targets, self.tree_model.invisibleRootItem()
            )
        elif action == move_new_action:
            name, ok = QInputDialog.getText(self, "New Group", "Enter group name:")
            if ok and name:
                callbacks.add_new_group(self, name)
                new_group = self.tree_model.invisibleRootItem().child(0)
                callbacks.move_items_to_group(self, targets, new_group)
        elif trash_action is not None and action == trash_action:
            callbacks.move_items_to_trash(self, targets)
        elif restore_action is not None and action == restore_action:
            callbacks.restore_items_from_trash(self, targets)
        elif group_action is not None and action == group_action:
            name, ok = QInputDialog.getText(self, "New Group", "Enter group name:")
            if ok and name:
                callbacks.group_clusters_in_tree(self, cluster_ids, name)
        elif feature_action is not None and action == feature_action:
            callbacks.feature_extraction(self, cluster_ids)
        elif export_cells_action is not None and action == export_cells_action:
            plot_export.open_plot_export_dialog(self, item, mode="per_cell")
        elif export_mean_action is not None and action == export_mean_action:
            plot_export.open_plot_export_dialog(self, item, mode="group_mean")
        elif add_group_action is not None and action == add_group_action:
            text, ok = QInputDialog.getText(self, "New Group", "Enter group name:")
            if ok and text:
                callbacks.add_new_group(self, text, parent_item=item)
        elif rename_action is not None and action == rename_action:
            new_group_name, ok = QInputDialog.getText(
                self, "Rename Group", "Enter group name:", text=item.text()
            )
            if ok and new_group_name:
                callbacks.rename_class(self, item.text(), new_group_name)
        elif flatten_action is not None and action == flatten_action:
            callbacks.flatten_group(self, item)
        elif delete_action is not None and action == delete_action:
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

    def _style_sidebar_toggle_btn(self, colors):
        """Theme-aware styling for the left sidebar collapse control."""
        self.sidebar_toggle_btn.setStyleSheet(f"""
            QPushButton {{
                border: none;
                border-left: 0.5px solid {colors['border_subtle']};
                color: {colors['text_secondary']};
                background: {colors['bg_panel']};
                font-size: 14px;
                font-weight: bold;
                padding: 0;
            }}
            QPushButton:hover {{
                color: {colors['text_primary']};
                background: {colors['bg_surface']};
            }}
        """)

    def _update_sidebar_toggle_btn(self):
        """Sync −/+ label and tooltip with collapsed state."""
        if self.sidebar_collapsed:
            self.sidebar_toggle_btn.setText("+")
            self.sidebar_toggle_btn.setToolTip("Expand cluster sidebar")
        else:
            self.sidebar_toggle_btn.setText("\u2212")
            self.sidebar_toggle_btn.setToolTip("Collapse cluster sidebar")

    def _animate_splitter_sizes(self, target_left, steps=5, interval_ms=25):
        """Step main_splitter sizes toward target for a snappy collapse/expand."""
        widths = self.main_splitter.sizes()
        total = sum(widths) or 1
        start_left = widths[0]
        delta = (target_left - start_left) / steps
        step = [0]

        def tick():
            import sip

            if sip.isdeleted(self.main_splitter):
                return
            step[0] += 1
            left = int(start_left + delta * step[0])
            if step[0] >= steps:
                left = target_left
            self.main_splitter.setSizes([left, total - left])
            if step[0] < steps:
                QTimer.singleShot(interval_ms, tick)

        tick()

    def toggle_sidebar(self):
        """Collapses or expands the left cluster sidebar (tree/table pane)."""
        widths = self.main_splitter.sizes()
        sum(widths) or self.width()

        if self.sidebar_collapsed:
            self.left_content.show()
            target_left = self.last_left_width
            self.sidebar_collapsed = False
        else:
            if widths[0] >= SIDEBAR_COLLAPSED_WIDTH:
                self.last_left_width = max(widths[0], 180)
            self.left_content.hide()
            target_left = SIDEBAR_COLLAPSED_WIDTH
            self.sidebar_collapsed = True

        self._update_sidebar_toggle_btn()
        self._animate_splitter_sizes(target_left)

    def toggle_population_fullscreen(self, checked):
        """Toggles the Population panel to take up 100% of the right pane."""
        colors = self.get_current_colors()
        if checked:
            # Full screen mode: Collapse the main tabs completely
            self.right_splitter.setSizes([0, 1000])
            self.pop_expand_btn.setText("🗗 Restore")
            self.pop_expand_btn.setStyleSheet(
                f"font-weight: bold; background-color: {colors['accent_positive']}; padding: 4px 10px;"
            )
        else:
            # Restore mode: 75/25 split
            total = sum(self.right_splitter.sizes()) or 1400
            left_size = max(int(total * 0.75), 400)
            right_size = total - left_size
            self.right_splitter.setSizes([left_size, right_size])
            self.pop_expand_btn.setText("⛶ Full Screen")
            self.pop_expand_btn.setStyleSheet(
                f"font-weight: bold; background-color: {colors['accent']}; padding: 4px 10px;"
            )

    def closeEvent(self, event):
        """Handles the window close event."""
        if self.data_manager and self.data_manager.is_dirty:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved refinement changes. Do you want to save before exiting?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Save:
                self.on_save_action()
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return

        # Persist the derived caches before tearing anything down. Without
        # this they were only ever written once, at the moment the physics
        # warm-up completed: quitting before that finished threw the whole
        # session's work away, and anything computed afterwards was lost too.
        # Blocking on purpose — the background saver is a daemon thread and
        # would be killed mid-write by interpreter shutdown.
        if self.data_manager is not None:
            try:
                self.data_manager.save_standard_plot_cache(blocking=True)
            except Exception:
                logger.warning("Could not persist caches on exit", exc_info=True)

        # The grating batch pass walks every cluster sequentially and so is
        # usually the thread still running when the window closes early. It was
        # absent from the cleanup list below, which is what produced
        # "QThread: Destroyed while thread is still running" on exit — and an
        # unclean teardown is why the cache save above could be lost. Ask it to
        # stop cooperatively before quitting its thread; quit() alone cannot
        # interrupt a worker sitting in a long compute loop.
        batch_worker = getattr(self, "_grating_batch_worker", None)
        if batch_worker is not None and hasattr(batch_worker, "stop"):
            try:
                batch_worker.stop()
            except RuntimeError:
                pass  # already deleted by its own finished handler

        # Stop background workers first
        callbacks.stop_worker(self)

        # Cleanup all threads with timeout to prevent hangs
        for thread_attr in [
            "feature_worker_thread",
            "worker_thread",
            "standard_worker_thread",
            "ks_load_thread",
            "vision_load_thread",
            "refine_thread",
            "_grating_batch_thread",
        ]:
            self._cleanup_thread(thread_attr, timeout_ms=1000)

        # Also cleanup vision_load_worker if it exists. Same deleted-wrapper
        # hazard as the threads above: its own finished handler may already
        # have destroyed it.
        if hasattr(self, "vision_load_worker") and self.vision_load_worker:
            try:
                self.vision_load_worker.deleteLater()
            except RuntimeError:
                pass
            self.vision_load_worker = None

        # Stop any running raw trace worker
        if hasattr(self, "raw_panel"):
            try:
                self.raw_panel._stop_worker()
            except (RuntimeError, AttributeError):
                logger.debug("raw panel worker already gone", exc_info=True)

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
                tmp_files = glob.glob(os.path.join(ks_dir, "*.tmp"))
                for tmp_file in tmp_files:
                    try:
                        os.remove(tmp_file)
                        logging.info(f"Cleaned up orphaned temp file: {tmp_file}")
                    except OSError:
                        pass  # File might be locked, just skip it
        except Exception as e:
            logging.error(f"Error during temp file cleanup: {e}")
        # ----------------------------------------------

        event.accept()