from __future__ import annotations
from qtpy.QtWidgets import QDialog, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QSplitter
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gui.main_window import MainWindow
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
        print(self.pca_scores)
        print(self.pca_scores.shape)
        self.draw_subplots(0) # draw the first subplot

    def get_temporal_traces(self):
        temporal_traces = []
        for cluster_index in self.cluster_ids:
            vision_cluster_index = cluster_index + 1
            if vision_cluster_index in self.main_window.data_manager.vision_params.main_datatable:
                red_tc = self.main_window.data_manager.vision_params.get_data_for_cell(vision_cluster_index, 'RedTimeCourse')
                temporal_traces.append(red_tc)
            else:
                print(f"Warning: Cluster ID {cluster_index} (Vision ID {vision_cluster_index}) not found in Vision data. Skipping.")
        temporal_traces = np.array(temporal_traces)
        
        return temporal_traces

    def pca_analysis(self):
        pca = PCA(n_components=3)
        pca_scores = pca.fit_transform(self.temporal_traces)
        return pca_scores
    
    def draw_subplots(self, index_subplot):
        if index_subplot == 0:
            ax = self.axes[0][0]
            ax.scatter(self.pca_scores[:, 0], self.pca_scores[:, 1])
            self.canvas.draw()
            
            selected_ids = []

            def onselect(eclick, erelease):
                x1, y1 = eclick.xdata, eclick.ydata
                x2, y2 = erelease.xdata, erelease.ydata

                xmin, xmax = sorted([x1, x2])
                ymin, ymax = sorted([y1, y2])

                flag_ROI = ((self.pca_scores[:, 0] >= xmin) & 
                            (self.pca_scores[:, 0] <= xmax) & 
                            (self.pca_scores[:, 1] >= ymin) &
                            (self.pca_scores[:, 1] <= ymax))

                sel_ids = [cid for cid, flag in zip(self.cluster_ids, flag_ROI) if flag]
                selected_ids.clear()
                selected_ids.extend(sel_ids)
                print("Selected cluster IDs:", sel_ids)

            self.selector = RectangleSelector(ax, onselect,
                            useblit=False,
                            minspanx=0.01,
                            minspany=0.01,
                            spancoords='data',
                            button=[1],
                            interactive=True)

            return selected_ids
