from __future__ import annotations
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QTableView, QPushButton, QHBoxLayout, QAbstractItemView
from qtpy.QtCore import Signal, QItemSelectionModel
from gui.widgets import PandasModel
import numpy as np
import pandas as pd
from analysis.constants import EI_CORR_THRESHOLD
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gui.main_window import MainWindow

class SimilarityPanel(QWidget):
    # Signal emitted when the selection changes; sends list of selected cluster IDs
    selection_changed = Signal(list)

    def __init__(self, main_window: MainWindow, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.main_cluster_id = None
        self._spacebar_select_count = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel("Similar Clusters")
        layout.addWidget(self.label)

        self.table = QTableView()
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.table)

        # Button row
        button_layout = QHBoxLayout()
        self.duplicate_button = QPushButton("Mark as Duplicates")
        self.duplicate_button.setToolTip("Mark selected clusters as duplicates (Cmd+D / Ctrl+D)")
        button_layout.addWidget(self.duplicate_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        self.duplicate_button.clicked.connect(self._on_mark_duplicates)

        self.similarity_model = None

        # Connect selection change after model is set (see set_data)
        self.table_selection_connected = False
    
    def set_data(self, similarity_df):
        """Set the DataFrame for the similarity table."""
        self.similarity_model = PandasModel(similarity_df)
        self.table.setModel(self.similarity_model)
        self.table.resizeColumnsToContents()
        self.table.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self.table_selection_connected = True

    def _on_selection_changed(self):
        """Emit the list of selected cluster IDs."""
        indexes = self.table.selectionModel().selectedRows()
        if self.similarity_model is not None:
            selected_ids = [self.similarity_model._dataframe.iloc[idx.row()]['cluster_id'] for idx in indexes]
            self.selection_changed.emit(selected_ids)

    def select_top_n_rows(self, n):
        """Select the top n rows in the table."""
        model = self.table.model()
        if model is None or model.rowCount() == 0:
            return
        selection_model = self.table.selectionModel()
        selection_model.clearSelection()
        for row in range(min(n, model.rowCount())):
            index = model.index(row, 0)
            selection_model.select(index, QItemSelectionModel.Select | QItemSelectionModel.Rows)
        # Optionally scroll to the last selected row
        if n > 0:
            self.table.scrollTo(model.index(n-1, 0))

    def handle_spacebar(self):
        """Call this on each spacebar press."""
        model = self.table.model()
        if model is None or model.rowCount() == 0:
            return
        self._spacebar_select_count += 1
        if self._spacebar_select_count > model.rowCount():
            self._spacebar_select_count = 1  # wrap around
        self.select_top_n_rows(self._spacebar_select_count)

    def reset_spacebar_counter(self):
        self._spacebar_select_count = 0
    
    def _on_mark_duplicates(self):
        """Emit the selected clusters along with main cluster ID as a duplicate group."""
        indexes = self.table.selectionModel().selectedRows()
        if self.similarity_model is None:
            print("[ERROR] Similarity model is not set.")
            return
        
        dup_ids = [self.similarity_model._dataframe.iloc[idx.row()]['cluster_id'] for idx in indexes]
        dup_ids.append(self.main_cluster_id)
        # Ensure uniqueness
        dup_ids = set(dup_ids)

        # Add to data_manager
        dm = self.main_window.data_manager
        dm.mark_duplicates(dup_ids)

        self.main_window.status_bar.showMessage(f"Marked {len(dup_ids)} clusters as duplicates and saved to file.", 3000)

        



    def clear(self):
        """Clear the table."""
        self.table.setModel(None)
        self.similarity_model = None

    def update_main_cluster_id(self, cluster_id):
        # Get EI correlation values from data_manager
        if self.main_window.data_manager is None or self.main_window.data_manager.ei_corr_dict is None:
            print("Error: DataManager or EI correlation data not available.")
            self.clear()
            return

        self.main_cluster_id = cluster_id
        
        ei_corr_dict = self.main_window.data_manager.ei_corr_dict
        cluster_ids = np.array(list(self.main_window.data_manager.vision_eis.keys())) - 1
        main_idx = np.where(cluster_ids == cluster_id)[0][0]
        other_idx = np.where(cluster_ids != cluster_id)[0]
        other_ids = cluster_ids[other_idx]
        d_df = {
            'cluster_id': other_ids,
            'space_ei_corr': ei_corr_dict['space'][main_idx, other_idx],
            'full_ei_corr': ei_corr_dict['full'][main_idx, other_idx],
            'power_ei_corr': ei_corr_dict['power'][main_idx, other_idx]
        }
        df = pd.DataFrame(d_df)
        # Add n_spikes column from data_manager.cluster_df
        cluster_df = self.main_window.data_manager.cluster_df
        n_spikes_map = dict(zip(cluster_df['cluster_id'], cluster_df['n_spikes']))
        df['n_spikes'] = df['cluster_id'].map(n_spikes_map)
        
        # Sort by space_ei_corr descending
        df = df.sort_values(by='space_ei_corr', ascending=False).reset_index(drop=True)

        df['potential_dups'] = (
            (df['full_ei_corr'].astype(float) > EI_CORR_THRESHOLD) |
            (df['space_ei_corr'].astype(float) > EI_CORR_THRESHOLD) |
            (df['power_ei_corr'].astype(float) > EI_CORR_THRESHOLD)
        )

        # Format correlation columns to 2 decimal places
        for col in ['full_ei_corr', 'space_ei_corr', 'power_ei_corr']:
            df[col] = df[col].map(lambda x: f"{x:.2f}")

        df['potential_dups'] = df['potential_dups'].map(lambda x: 'Yes' if x else '')
        
        self.set_data(df)