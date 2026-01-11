from __future__ import annotations
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QAbstractItemView, QComboBox, QButtonGroup, QRadioButton
from qtpy.QtCore import Signal, QItemSelectionModel
from ..widgets.widgets import HighlightStatusPandasModel, CustomTableView
import pandas as pd
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..main_window import MainWindow
import logging
logger = logging.getLogger(__name__)


class SimilarityPanel(QWidget):
    # Signal emitted when the selection changes; sends list of selected
    # cluster IDs
    selection_changed = Signal(list)

    def __init__(self, main_window: MainWindow, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.main_cluster_id = None
        self._spacebar_select_count = 1
        self.current_source = "MEA"  # Default to MEA-based similarity
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel("Similar Clusters")
        layout.addWidget(self.label)

        # Add source selection buttons (MEA / Vision toggle)
        source_layout = QHBoxLayout()
        self.source_button_group = QButtonGroup()

        self.mea_radio = QRadioButton("MEA Similarity")
        self.vision_radio = QRadioButton("Vision Similarity")
        self.mea_radio.setChecked(True)  # Default to MEA

        self.source_button_group.addButton(self.mea_radio)
        self.source_button_group.addButton(self.vision_radio)

        source_layout.addWidget(self.mea_radio)
        source_layout.addWidget(self.vision_radio)
        source_layout.addStretch()
        layout.addLayout(source_layout)

        # Connect radio buttons to source change handler
        self.mea_radio.toggled.connect(self._on_source_toggled)

        self.table = CustomTableView()
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.table)

        # Button row
        # button_layout = QHBoxLayout()
        # self.clean_button = QPushButton("Mark Clean")
        # self.edge_button = QPushButton("Mark Edge")
        # self.duplicate_button = QPushButton("Mark as Duplicates")
        # self.unsure_button = QPushButton("Mark Unsure")
        # self.duplicate_button.setToolTip("Mark selected clusters as duplicates (Cmd+D / Ctrl+D)")
        # self.clean_button.setToolTip("Mark selected clusters as clean (Cmd+C / Ctrl+C)")
        # self.edge_button.setToolTip("Mark selected clusters as edge (Cmd+E / Ctrl+E)")
        # self.unsure_button.setToolTip("Mark selected clusters as unsure like What? (Cmd+W / Ctrl+W)")
        # button_layout.addWidget(self.duplicate_button)
        # button_layout.addWidget(self.clean_button)
        # button_layout.addWidget(self.edge_button)
        # button_layout.addWidget(self.unsure_button)
        # button_layout.addStretch()
        # layout.addLayout(button_layout)

        status_layout = QHBoxLayout()
        self.status_combo = QComboBox()
        self.status_combo.addItems([
            "Clean", "Edge", "Duplicate", "Unsure", "Noisy", "Contaminated", "Off Array"
        ])
        self.mark_button = QPushButton("Mark Status")
        status_layout.addWidget(self.status_combo)
        status_layout.addWidget(self.mark_button)
        status_layout.addStretch()
        layout.addLayout(status_layout)

        # self.duplicate_button.clicked.connect(lambda: self._mark_status('Duplicate'))
        # self.clean_button.clicked.connect(lambda: self._mark_status('Clean'))
        # self.edge_button.clicked.connect(lambda: self._mark_status('Edge'))
        # self.unsure_button.clicked.connect(lambda: self._mark_status('Unsure'))
        self.mark_button.clicked.connect(self._mark_selected_status)

        self.similarity_model = None

        # Connect selection change after model is set (see set_data)
        self.table_selection_connected = False

    def set_data(self, similarity_df):
        """Set the DataFrame for the similarity table."""
        self.similarity_model = HighlightStatusPandasModel(similarity_df)
        self.table.setModel(self.similarity_model)
        self.table.resizeColumnsToContents()
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
        # self._spacebar_select_count += 1
        # if self._spacebar_select_count > model.rowCount():
        #     self._spacebar_select_count = 1  # wrap around
        # self.select_top_n_rows(self._spacebar_select_count)

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
        self.vision_radio.setEnabled(True)

        # Update the table if vision similarity is currently selected
        if self.current_source == "vision" and self.main_cluster_id is not None:
            self.update_main_cluster_id(self.main_cluster_id)

    def _on_source_toggled(self):
        """Handle when the user toggles between MEA and Vision similarity sources."""
        if self.mea_radio.isChecked():
            self.current_source = "MEA"
        else:
            self.current_source = "vision"

        # Update the table with the new source if a main cluster is selected
        if self.main_cluster_id is not None:
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
                    # Add n_spikes and status from main cluster_df for
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

            elif self.current_source == "vision":
                # Check if vision data is available
                if not dm.vision_available:
                    logger.warning(
                        "Vision data not available for similarity table")
                    # Show empty table or placeholder data
                    similarity_df = pd.DataFrame(
                        columns=['cluster_id', 'n_spikes', 'status'])
                else:
                    # Get vision-based similarity data
                    similarity_df = dm.get_similarity_table(
                        cluster_id, source="vision")
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
