from __future__ import annotations
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QAbstractItemView, QComboBox
from qtpy.QtCore import Signal, QItemSelectionModel, Qt
from ..widgets.widgets import HighlightStatusPandasModel, CustomTableView
import pandas as pd
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..main_window import MainWindow
import logging
logger = logging.getLogger(__name__)

PANEL_PADDING  = 8   # px — inner padding on all panels
CTRL_SPACING   = 6   # px — gap between controls in a row
ROW_HEIGHT     = 28  # px — standard table row height (was ~32px)


class SimilarityPanel(QWidget):
    # Signal emitted when the selection changes; sends list of selected
    # cluster IDs
    selection_changed = Signal(list)

    def restyle_plots(self, colors):
        """Updates UI elements based on the provided color scheme."""
        # Update segmented buttons
        for i, btn in enumerate([self.mea_btn, self.vision_btn]):
            btn.setStyleSheet(f"""
                QPushButton {{
                    font-size: 11px;
                    padding: 0 10px;
                    border: 0.5px solid {colors['border_default']};
                    border-{'right' if i == 0 else 'left'}-width: 0;
                    border-radius: 0;
                    {'border-top-left-radius: 4px; border-bottom-left-radius: 4px;' if i == 0 else
                    'border-top-right-radius: 4px; border-bottom-right-radius: 4px; border-left: none;'}
                    color: {colors['text_secondary']};
                    background: transparent;
                }}
                QPushButton:checked {{
                    background: rgba(46, 109, 212, 0.20);
                    border-color: {colors['accent']};
                    color: {colors['accent_hover']};
                }}
            """)
        
        # Update model if it exists
        if self.table.model() and hasattr(self.table.model(), 'update_colors'):
            self.table.model().update_colors(colors)

    def __init__(self, main_window: MainWindow, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.main_cluster_id = None
        self._spacebar_select_count = 1
        self.current_source = "MEA"  # Default to MEA-based similarity
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, PANEL_PADDING, 0, 0)
        layout.setSpacing(CTRL_SPACING)

        self.label = QLabel("Similar Clusters")
        layout.addWidget(self.label)

        # Segmented source toggle (MEA / Vision)
        source_row = QHBoxLayout()
        source_row.setSpacing(0)

        self.mea_btn    = QPushButton("MEA")
        self.vision_btn = QPushButton("Vision")

        # Get initial colors from main window
        colors = self.main_window.get_current_colors()

        for i, btn in enumerate([self.mea_btn, self.vision_btn]):
            btn.setCheckable(True)
            btn.setFixedHeight(24)
            btn.setStyleSheet(f"""
                QPushButton {{
                    font-size: 11px;
                    padding: 0 10px;
                    border: 0.5px solid {colors['border_default']};
                    border-{'right' if i == 0 else 'left'}-width: 0;
                    border-radius: 0;
                    {'border-top-left-radius: 4px; border-bottom-left-radius: 4px;' if i == 0 else
                    'border-top-right-radius: 4px; border-bottom-right-radius: 4px; border-left: none;'}
                    color: {colors['text_secondary']};
                    background: transparent;
                }}
                QPushButton:checked {{
                    background: rgba(46, 109, 212, 0.20);
                    border-color: {colors['accent']};
                    color: {colors['accent_hover']};
                }}
            """)

        self.mea_btn.setChecked(True)
        self.mea_btn.clicked.connect(lambda: self._set_source("MEA"))
        self.vision_btn.clicked.connect(lambda: self._set_source("vision"))

        source_row.addWidget(self.mea_btn)
        source_row.addWidget(self.vision_btn)
        source_row.addStretch()
        layout.addLayout(source_row)

        self.table = CustomTableView()
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        status_layout = QHBoxLayout()
        self.status_combo = QComboBox()
        self.status_combo.addItems([
            "Clean", "Edge", "Duplicate", "Unsure", "Noisy", "Contaminated", "Off Array"
        ])
        self.status_combo.setFixedHeight(26)
        self.status_combo.setFixedWidth(110)
        
        self.mark_button = QPushButton("Mark")
        self.mark_button.setFixedHeight(26)
        
        status_layout.addWidget(self.status_combo)
        status_layout.addWidget(self.mark_button)
        status_layout.addStretch()
        layout.addLayout(status_layout)

        self.mark_button.clicked.connect(self._mark_selected_status)

        self.similarity_model = None

        # Connect selection change after model is set (see set_data)
        self.table_selection_connected = False

    def _set_source(self, source):
        """Update source selection and refresh table."""
        self.current_source = source
        if source == "MEA":
            self.mea_btn.setChecked(True)
            self.vision_btn.setChecked(False)
        else:
            self.mea_btn.setChecked(False)
            self.vision_btn.setChecked(True)
        
        if self.main_cluster_id is not None:
            self.update_main_cluster_id(self.main_cluster_id)

    def set_data(self, similarity_df):
        """Set the DataFrame for the similarity table and format it."""
        self.similarity_model = HighlightStatusPandasModel(similarity_df)
        self.similarity_model.update_colors(self.main_window.get_current_colors())
        
        # --- Monkey-patch pretty header labels onto the model ---
        HEADER_LABELS = {
            'cluster_id':        'ID',
            'best_chan':         'Ch',
            'n_spikes':          '# Spikes',
            'status':            'Status',
            'set':               'Set',
            'similarity':        'Sim.',
            'distance':          'Dist.',
            'potential_dups':    'Dup?',
            'KSLabel':           'KS Label',
        }

        df_cols = list(self.similarity_model._dataframe.columns)
        self.similarity_model._header_overrides = HEADER_LABELS.copy()
        _orig_headerData = self.similarity_model.headerData

        def _make_patched(orig, overrides, cols):
            def patched(section, orientation, role=Qt.DisplayRole):
                if orientation == Qt.Horizontal and role == Qt.DisplayRole:
                    col_name = cols[section] if section < len(cols) else None
                    if col_name and col_name in overrides:
                        return overrides[col_name]
                return orig(section, orientation, role)
            return patched

        self.similarity_model.headerData = _make_patched(
            _orig_headerData, self.similarity_model._header_overrides, df_cols)

        self.table.setModel(self.similarity_model)
        
        # --- Allow drag-to-reorder ---
        header = self.table.horizontalHeader()
        header.setSectionsMovable(True)
        
        # --- Set Default Visual Column Order ---
        col_index = {name: idx for idx, name in enumerate(df_cols)}
        
        # Define the exact order you want from left to right
        ORDERED_COLS = [
            'cluster_id', 
            'best_chan', 
            'similarity', 
            'distance', 
            'potential_dups', 
            'n_spikes', 
            'status', 
            'set'
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

        # Compress columns to minimum width based on contents
        self.table.resizeColumnsToContents()
        
        # Manage selection connections
        try:
            self.table.selectionModel().selectionChanged.disconnect(self._on_selection_changed)
        except (TypeError, RuntimeError):
            pass
        self.table.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self.table_selection_connected = True

    def _on_selection_changed(self):
        """Emit the list of selected cluster IDs."""
        indexes = self.table.selectionModel().selectedRows()
        if self.similarity_model is not None:
            selected_ids = [self.similarity_model._dataframe.iloc[idx.row()]['cluster_id']
                            for idx in indexes]
            self.selection_changed.emit(selected_ids)
        else:
            # If model is not set, emit empty list to handle deselection
            # properly
            self.selection_changed.emit([])

    def select_top_n_rows(self, n):
        """Select the top n rows in the table."""
        model = self.table.model()
        if model is None or model.rowCount() == 0:
            return
        selection_model = self.table.selectionModel()
        selection_model.clearSelection()
        for row in range(min(n, model.rowCount())):
            index = model.index(row, 0)
            selection_model.select(
                index, QItemSelectionModel.Select | QItemSelectionModel.Rows)
        # Optionally scroll to the last selected row
        if n > 0:
            self.table.scrollTo(model.index(n - 1, 0))

    def handle_spacebar(self):
        """Call this on each spacebar press."""
        model = self.table.model()
        if model is None or model.rowCount() == 0:
            return

        # Find the currently selected row
        selection_model = self.table.selectionModel()
        row_count = model.rowCount()
        selected_rows = selection_model.selectedRows()
        if selected_rows:
            current_row = selected_rows[0].row()
            next_row = (current_row + 1) % row_count
        else:
            next_row = 0  # Start from the top if nothing is selected

        selection_model.clearSelection()
        index = model.index(next_row, 0)
        selection_model.select(
            index, QItemSelectionModel.Select | QItemSelectionModel.Rows)
        self.table.scrollTo(index)

    def reset_spacebar_counter(self):
        self._spacebar_select_count = 1

    def _mark_status(self, status):
        """Emit the selected clusters along with main cluster ID as a duplicate group."""
        indexes = self.table.selectionModel().selectedRows()
        if self.similarity_model is None:
            logger.error("Similarity model is not set")
            return

        # If status is Clean, only apply to main cluster.
        if status == 'Clean':
            selected_ids = [self.main_cluster_id]
        else:
            selected_ids = [self.similarity_model._dataframe.iloc[idx.row()]['cluster_id']
                            for idx in indexes]
            selected_ids.append(self.main_cluster_id)

        # Update data manager and export
        dm = self.main_window.data_manager
        dm.update_and_export_status(selected_ids, status=status)

        # Refresh both similarity and main views
        self.main_window.main_cluster_model.refresh_view()
        self.similarity_model.refresh_view()

        self.main_window.status_bar.showMessage(
            f"Marked {len(selected_ids)} clusters as {status} and saved to file.", 3000)

    def _mark_selected_status(self):
        status = self.status_combo.currentText()
        self._mark_status(status)

    def clear(self):
        """Clear the table."""
        self.table.setModel(None)
        self.similarity_model = None

    def on_vision_loaded(self):
        """Called when vision data is loaded - enables vision similarity option."""
        # Enable the vision radio button
        self.vision_btn.setEnabled(True)

        # Update the table if vision similarity is currently selected
        if self.current_source == "vision" and self.main_cluster_id is not None:
            self.update_main_cluster_id(self.main_cluster_id)

    def update_main_cluster_id(self, cluster_id):
        """Update the similarity table for the given cluster_id."""
        dm = self.main_window.data_manager
        if dm is None:
            logger.error("DataManager not available")
            self.clear()
            return

        # Clear selection in the table when switching main cluster ID
        if self.table and self.table.selectionModel():
            self.table.selectionModel().clearSelection()

        self.main_cluster_id = cluster_id

        # Determine which source to use and get appropriate similarity data
        try:
            if self.current_source == "MEA":
                # Get MEA-based similarity data
                similarity_df = dm.get_similarity_table(
                    cluster_id, source="MEA")

                # Add any custom columns that are useful for display if not
                # already present
                if similarity_df is not None and not similarity_df.empty:
                    # Add n_spikes, status, and best_chan from main cluster_df for
                    # consistency
                    cluster_df = dm.cluster_df
                    if 'n_spikes' not in similarity_df.columns:
                        n_spikes_map = dict(
                            zip(cluster_df['cluster_id'], cluster_df['n_spikes']))
                        similarity_df['n_spikes'] = similarity_df['cluster_id'].map(
                            n_spikes_map)

                    if 'status' not in similarity_df.columns:
                        status_map = dict(
                            zip(cluster_df['cluster_id'], cluster_df['status']))
                        similarity_df['status'] = similarity_df['cluster_id'].map(
                            status_map)

                    if 'set' not in similarity_df.columns:
                        set_map = dict(
                            zip(cluster_df['cluster_id'], cluster_df['set']))
                        similarity_df['set'] = similarity_df['cluster_id'].map(
                            set_map)

                    # --- NEW: Inject Best Channel ---
                    if 'best_chan' not in similarity_df.columns:
                        chan_map = dict(
                            zip(cluster_df['cluster_id'], cluster_df['best_chan']))
                        similarity_df['best_chan'] = similarity_df['cluster_id'].map(
                            chan_map)

            elif self.current_source == "vision":
                # Check if vision data is available
                if not dm.vision_available:
                    logger.warning(
                        "Vision data not available for similarity table")
                    # Show empty table or placeholder data
                    similarity_df = pd.DataFrame(
                        columns=['cluster_id', 'n_spikes', 'status', 'best_chan'])
                else:
                    # Get vision-based similarity data
                    similarity_df = dm.get_similarity_table(
                        cluster_id, source="vision")
                    
                    # Ensure best_chan is there for Vision too
                    if similarity_df is not None and not similarity_df.empty:
                        cluster_df = dm.cluster_df
                        if 'best_chan' not in similarity_df.columns:
                            chan_map = dict(
                                zip(cluster_df['cluster_id'], cluster_df['best_chan']))
                            similarity_df['best_chan'] = similarity_df['cluster_id'].map(
                                chan_map)
            else:
                logger.error(f"Unknown source: {self.current_source}")
                self.clear()
                return

        except Exception as e:
            logger.error(
                f"Error getting similarity table for source {self.current_source}: {e}")
            self.clear()
            return

        if similarity_df is not None and not similarity_df.empty:
            # Add potential_dups based on relevant threshold if not already
            # present
            if 'potential_dups' not in similarity_df.columns:
                # Default to no potential duplicates if not specifically
                # computed
                similarity_df['potential_dups'] = ''

            self.set_data(similarity_df)
        else:
            self.clear()
