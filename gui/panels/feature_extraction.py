from __future__ import annotations
from qtpy.QtWidgets import QDialog, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QSplitter, QMenu, QStyle
from qtpy.QtGui import QCursor, QStandardItem, QColor
from qtpy.QtCore import Qt
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gui.main_window import MainWindow
from gui.widgets import HighlightStatusPandasModel
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from sklearn.decomposition import PCA

class FeatureExtractionWindow(QDialog):
    """
    Pop up window for feature extraction.
    """
    def __init__(self, main_window: MainWindow, cluster_ids, parent=None):
        super().__init__(parent)
        self.main_window = main_window # for data access
        self.cluster_ids = cluster_ids
        self.setWindowTitle('Feature Extraction')
        self.setGeometry(200, 200, 900, 600)

        self.main_layout = QVBoxLayout()
        self.fig = plt.figure()
        self.canvas = FigureCanvas(self.fig)
        self.axes = self.fig.subplots(2, 3)
        self.main_layout.addWidget(self.canvas)
        self.setLayout(self.main_layout)

        self.temporal_traces = self.get_temporal_traces()
        self.pca_scores = self.pca_analysis()
        self.draw_subplots(0) # draw the first subplot
        self.draw_subplots(1) # draw the second subplot

    def get_temporal_traces(self):
        temporal_traces = []
        for cluster_index in self.cluster_ids:
            vision_cluster_index = cluster_index + 1
            red_tc = self.main_window.data_manager.vision_params.get_data_for_cell(vision_cluster_index, 'RedTimeCourse')
            temporal_traces.append(red_tc)
        temporal_traces = np.array(temporal_traces)
        
        return temporal_traces

    def pca_analysis(self):
        pca = PCA(n_components=3)
        pca_scores = pca.fit_transform(self.temporal_traces)
        return pca_scores
    
    def draw_subplots(self, subplot_idx):
        if subplot_idx == 0:
            ax = self.axes[0][0]
            scatter_vals = self.pca_scores[:, 0:2]
        if subplot_idx == 1:
            ax = self.axes[0][1]
            scatter_vals = self.pca_scores[:, 1:]

        ax.scatter(scatter_vals[:, 0], scatter_vals[:, 1])
        self.canvas.draw()
        
        def make_callback(subplot_idx, scatter_vals):
            def _callback(eclick, erelease):
                self._on_select_subplot(eclick, erelease, subplot_idx, scatter_vals)
            return _callback

        if not hasattr(self, "selectors"):
            self.selectors = {}

        self.selectors[subplot_idx] = RectangleSelector(ax, make_callback(subplot_idx, scatter_vals),
                                        useblit=False,
                                        minspanx=0.01,
                                        minspany=0.01,
                                        spancoords='data',
                                        button=[1],
                                        interactive=True)



    def _on_select_subplot(self, eclick, erelease, subplot_idx, scatter_vals):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])

        flag_ROI = ((scatter_vals[:, 0] >= xmin) & 
                    (scatter_vals[:, 0] <= xmax) & 
                    (scatter_vals[:, 1] >= ymin) &
                    (scatter_vals[:, 1] <= ymax))

        selected_ids = [cid for cid, flag in zip(self.cluster_ids, flag_ROI) if flag]

        menu = QMenu(self)
        add_new_class_action = menu.addAction("New class")
        action = menu.exec_(QCursor.pos())
        if action == add_new_class_action:
            selector = self.selectors[subplot_idx]
            for artist in selector.artists:
                artist.set_visible(False)
            self.canvas.draw_idle()
            print("Selected cluster IDs:", selected_ids)
            self.create_new_class(selected_ids)
    
    def create_new_class(self, selected_ids):
        if self.main_window.data_manager is None:
            return

        model = self.main_window.tree_model
        model.clear()

        df = self.main_window.data_manager.cluster_df[
        self.main_window.data_manager.cluster_df['KSLabel'] != 'mua'
        ].copy()

        group_name = "Nc1"
        df.loc[df['cluster_id'].isin(selected_ids), 'KSLabel'] = group_name

        # Update table view with filtered data
        self.main_window.main_cluster_model = HighlightStatusPandasModel(df)
        self.main_window.setup_table_model(self.main_window.main_cluster_model)

        # Create top-level nodes for each unique KSLabel
        groups = {}
        for label in df['KSLabel'].unique():
            group_item = QStandardItem(label)
            # group_item.setEditable(False)
            group_item.setDropEnabled(True)  # Can drop cells into it
            
            # Style group items differently from cells
            font = group_item.font()
            font.setBold(True)
            group_item.setFont(font)
            
            # Set different background color for groups
            group_item.setBackground(QColor('#3C3C3C'))  # Dark gray background for groups
            
            # Add folder icon for groups
            group_item.setIcon(self.main_window.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon))
            
            groups[label] = group_item
            model.appendRow(group_item)
            
        # Add each cluster as a child item to its group
        for _, row in df.iterrows():
            cluster_id = row['cluster_id']
            label = row['KSLabel']
            
            # The text displayed will be e.g., "Cluster 123 (n=456 spikes)"
            item_text = f"Cluster {cluster_id} (n={row['n_spikes']})"
            cell_item = QStandardItem(item_text)
            cell_item.setEditable(False)
            
            # Add a special icon or style for cells to distinguish them
            font = cell_item.font()
            font.setItalic(False)
            cell_item.setFont(font)
            
            # Add file icon for cells
            cell_item.setIcon(self.main_window.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))
            
            # IMPORTANT: Store the actual cluster ID in the item's data role.
            # This is how we'll retrieve it when the item is clicked.
            cell_item.setData(cluster_id, Qt.ItemDataRole.UserRole)
            
            # Prevent dropping items onto cells
            cell_item.setDropEnabled(False)
            
            groups[label].appendRow(cell_item)
            
        self.main_window.setup_tree_model(model)
        self.main_window.tree_view.expandAll()