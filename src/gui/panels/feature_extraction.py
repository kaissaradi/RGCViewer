from __future__ import annotations
from qtpy.QtWidgets import QDialog, QVBoxLayout, QMenu
from qtpy.QtGui import QCursor
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..main_window import MainWindow
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from sklearn.decomposition import PCA
import logging
logger = logging.getLogger(__name__)


class FeatureExtractionWindow(QDialog):
    """
    Pop up window for feature extraction.
    """

    def __init__(self, main_window: MainWindow, cluster_ids, parent=None):
        super().__init__(parent)
        self.main_window = main_window  # for data access
        self.cluster_ids = cluster_ids
        self.setWindowTitle('Feature Extraction')
        self.setGeometry(200, 200, 900, 600)

        self.main_layout = QVBoxLayout()
        self.fig = plt.figure()
        self.canvas = FigureCanvas(self.fig)
        self.axes = self.fig.subplots(2, 3)
        self.fig.subplots_adjust(wspace=0.45, hspace=0.3)  # increase gaps
        
        self.main_layout.addWidget(self.canvas)
        self.setLayout(self.main_layout)

        # temporal features
        self.temporal_traces = self.get_temporal_traces()
        self.ACG = self.get_autocorrelogram()
        # pca analysis
        self.temporal_traces_pca_scores = self.temporal_traces_pca_analysis()
        self.ACG_pca_scores = self.ACG_pca_analysis()
        # spatial features
        self.stafit = self.get_stafit()
        self.RF_diameter = np.sqrt(self.stafit[:, 0] * self.stafit[:, 1])

        logger.debug(
            'PCA scores computed for temporal traces; shape=%s',
            getattr(
                self.temporal_traces_pca_scores,
                'shape',
                None))
        logger.debug(
            'PCA scores computed for ACG; shape=%s',
            getattr(
                self.ACG_pca_scores,
                'shape',
                None))

        self.draw_subplots(0)  # draw the first subplot
        self.draw_subplots(1)  # draw the second subplot
        self.draw_subplots(2)  # draw the third subplot
        self.draw_subplots(3)  # draw the forth subplot
        self.draw_subplots(4)  # draw the third subplot
        self.draw_subplots(5)  # draw the forth subplot

    def get_temporal_traces(self):
        temporal_traces = []
        for cluster_index in self.cluster_ids:
            vision_cluster_index = cluster_index + 1
            if vision_cluster_index in self.main_window.data_manager.vision_params.main_datatable:
                red_tc = self.main_window.data_manager.vision_params.get_data_for_cell(
                    vision_cluster_index, 'RedTimeCourse')
                temporal_traces.append(red_tc)
            else:
                logger.warning(
                    'Cluster ID %s (Vision ID %s) not found in Vision data; skipping',
                    cluster_index,
                    vision_cluster_index)
        if len(temporal_traces) == 0:
            return np.empty((0,))

        temporal_traces = np.array(temporal_traces)
        return temporal_traces

    def get_autocorrelogram(self):
        ACG = []
        for cluster_index in self.cluster_ids:
            vision_cluster_index = cluster_index + 1
            if vision_cluster_index in self.main_window.data_manager.vision_params.main_datatable:
                ACG_current = self.main_window.data_manager.get_acg_data(cluster_index)
                ACG.append(ACG_current[1])
            else:
                logger.warning(
                    'Cluster ID %s (Vision ID %s) not found in Vision data; skipping',
                    cluster_index,
                    vision_cluster_index)
        if len(ACG) == 0:
            return np.empty((0,))

        ACG = np.array(ACG)
        return ACG
    
    def get_stafit(self):
        stafit = []
        for cluster_index in self.cluster_ids:
            vision_cluster_index = cluster_index + 1
            if vision_cluster_index in self.main_window.data_manager.vision_params.main_datatable:
                stafit_current = self.main_window.data_manager.vision_params.get_stafit_for_cell(vision_cluster_index)
                stafit.append([stafit_current.std_x, stafit_current.std_y])
            else:
                logger.warning(
                    'Cluster ID %s (Vision ID %s) not found in Vision data; skipping',
                    cluster_index,
                    vision_cluster_index)
        if len(stafit) == 0:
            return np.empty((0,))

        stafit = np.array(stafit)
        return stafit

    def temporal_traces_pca_analysis(self):
        pca = PCA(n_components=3)
        pca_scores = pca.fit_transform(self.temporal_traces)
        return pca_scores
    
    def ACG_pca_analysis(self):
        pca = PCA(n_components=3)
        pca_scores = pca.fit_transform(self.ACG)
        return pca_scores

    def draw_subplots(self, subplot_idx):
        subplots_lw = 0.6
        subplots_marker_size = 15
        if subplot_idx == 0:
            ax = self.axes[0][0]
            ax.scatter(self.temporal_traces_pca_scores[:, 0], self.temporal_traces_pca_scores[:, 1],
                       marker='o', facecolors='none', edgecolors='k',
                       linewidths=subplots_lw, s = subplots_marker_size)
            ax.set_xlabel('TF 1')
            ax.set_ylabel('TF 2')
            self.canvas.draw()

            def onselect(eclick, erelease):
                x1, y1 = eclick.xdata, eclick.ydata
                x2, y2 = erelease.xdata, erelease.ydata

                xmin, xmax = sorted([x1, x2])
                ymin, ymax = sorted([y1, y2])

                flag_ROI = ((self.temporal_traces_pca_scores[:, 0] >= xmin) &
                            (self.temporal_traces_pca_scores[:, 0] <= xmax) &
                            (self.temporal_traces_pca_scores[:, 1] >= ymin) &
                            (self.temporal_traces_pca_scores[:, 1] <= ymax))

                selected_ids = [
                    cid for cid, flag in zip(
                        self.cluster_ids, flag_ROI) if flag]

                # Trigger menu immediately when selection is made
                if selected_ids:
                    menu = QMenu(self)
                    create_action = menu.addAction(
                        f"Create Group from {len(selected_ids)} clusters")
                    action = menu.exec_(QCursor.pos())
                    if action == create_action:
                        self.create_new_class(selected_ids)

                logger.debug('Selected cluster IDs: %s', selected_ids)

            self.selector = RectangleSelector(ax, onselect,
                                              useblit=False,
                                              minspanx=0.01,
                                              minspany=0.01,
                                              spancoords='data',
                                              button=[1],
                                              interactive=True)

        elif subplot_idx == 1:
            ax = self.axes[0][1]
            ax.scatter(self.RF_diameter, self.temporal_traces_pca_scores[:, 0],
                       marker='o', facecolors='none', edgecolors='k',
                       linewidths=subplots_lw, s = subplots_marker_size)
            ax.set_xlabel('RF Diameter')
            ax.set_ylabel('TF 1')
            self.canvas.draw()

            def onselect(eclick, erelease):
                x1, y1 = eclick.xdata, eclick.ydata
                x2, y2 = erelease.xdata, erelease.ydata

                xmin, xmax = sorted([x1, x2])
                ymin, ymax = sorted([y1, y2])

                flag_ROI = ((self.RF_diameter >= xmin) &
                            (self.RF_diameter <= xmax) &
                            (self.temporal_traces_pca_scores[:, 0] >= ymin) &
                            (self.temporal_traces_pca_scores[:, 0] <= ymax))

                selected_ids = [
                    cid for cid, flag in zip(
                        self.cluster_ids, flag_ROI) if flag]

                # Trigger menu immediately when selection is made
                if selected_ids:
                    menu = QMenu(self)
                    create_action = menu.addAction(
                        f"Create Group from {len(selected_ids)} clusters")
                    action = menu.exec_(QCursor.pos())
                    if action == create_action:
                        self.create_new_class(selected_ids)

                logger.debug('Selected cluster IDs: %s', selected_ids)

            self.selector2 = RectangleSelector(ax, onselect,
                                               useblit=False,
                                               minspanx=0.01,
                                               minspany=0.01,
                                               spancoords='data',
                                               button=[1],
                                               interactive=True)
        elif subplot_idx == 2:
            ax = self.axes[0][2]
            markerline, stemlines, baseline = ax.stem(self.RF_diameter, np.ones(self.RF_diameter.shape))
            markerline.set_marker('None')
            stemlines.set_color('k')
            stemlines.set_linewidth(0.6)
            baseline.set_visible(False)
            ax.set_xlabel("RF Diameter")
            self.canvas.draw()

            def onselect(eclick, erelease):
                x1, y1 = eclick.xdata, eclick.ydata
                x2, y2 = erelease.xdata, erelease.ydata

                xmin, xmax = sorted([x1, x2])
                ymin, ymax = sorted([y1, y2])

                flag_ROI = ((self.RF_diameter >= xmin) &
                            (self.RF_diameter <= xmax))

                selected_ids = [
                    cid for cid, flag in zip(
                        self.cluster_ids, flag_ROI) if flag]

                # Trigger menu immediately when selection is made
                if selected_ids:
                    menu = QMenu(self)
                    create_action = menu.addAction(
                        f"Create Group from {len(selected_ids)} clusters")
                    action = menu.exec_(QCursor.pos())
                    if action == create_action:
                        self.create_new_class(selected_ids)

                logger.debug('Selected cluster IDs: %s', selected_ids)

            self.selector3 = RectangleSelector(ax, onselect,
                                               useblit=False,
                                               minspanx=0.01,
                                               minspany=0.01,
                                               spancoords='data',
                                               button=[1],
                                               interactive=True)

        elif subplot_idx == 3:
            ax = self.axes[1][0]
            ax.scatter(self.ACG_pca_scores[:, 0], self.ACG_pca_scores[:, 1],
                       marker='o', facecolors='none', edgecolors='k',
                       linewidths=subplots_lw, s = subplots_marker_size)
            ax.set_xlabel("Autocorrelation 1")
            ax.set_ylabel("Autocorrelation 2")
            self.canvas.draw()

            def onselect(eclick, erelease):
                x1, y1 = eclick.xdata, eclick.ydata
                x2, y2 = erelease.xdata, erelease.ydata

                xmin, xmax = sorted([x1, x2])
                ymin, ymax = sorted([y1, y2])

                flag_ROI = ((self.ACG_pca_scores[:, 0] >= xmin) &
                            (self.ACG_pca_scores[:, 0] <= xmax) &
                            (self.ACG_pca_scores[:, 1] >= ymin) &
                            (self.ACG_pca_scores[:, 1] <= ymax))

                selected_ids = [
                    cid for cid, flag in zip(
                        self.cluster_ids, flag_ROI) if flag]

                # Trigger menu immediately when selection is made
                if selected_ids:
                    menu = QMenu(self)
                    create_action = menu.addAction(
                        f"Create Group from {len(selected_ids)} clusters")
                    action = menu.exec_(QCursor.pos())
                    if action == create_action:
                        self.create_new_class(selected_ids)

                logger.debug('Selected cluster IDs: %s', selected_ids)

            self.selector4 = RectangleSelector(ax, onselect,
                                               useblit=False,
                                               minspanx=0.01,
                                               minspany=0.01,
                                               spancoords='data',
                                               button=[1],
                                               interactive=True)
        elif subplot_idx == 4:
            ax = self.axes[1][1]
            ax.scatter(self.RF_diameter, self.ACG_pca_scores[:, 0],
                       marker='o', facecolors='none', edgecolors='k',
                       linewidths=subplots_lw, s = subplots_marker_size)
            ax.set_xlabel("RF Diameter")
            ax.set_ylabel("Autocorrelation 1")
            self.canvas.draw()

            def onselect(eclick, erelease):
                x1, y1 = eclick.xdata, eclick.ydata
                x2, y2 = erelease.xdata, erelease.ydata

                xmin, xmax = sorted([x1, x2])
                ymin, ymax = sorted([y1, y2])

                flag_ROI = ((self.RF_diameter >= xmin) &
                            (self.RF_diameter <= xmax) &
                            (self.ACG_pca_scores[:, 0] >= ymin) &
                            (self.ACG_pca_scores[:, 0] <= ymax))

                selected_ids = [
                    cid for cid, flag in zip(
                        self.cluster_ids, flag_ROI) if flag]

                # Trigger menu immediately when selection is made
                if selected_ids:
                    menu = QMenu(self)
                    create_action = menu.addAction(
                        f"Create Group from {len(selected_ids)} clusters")
                    action = menu.exec_(QCursor.pos())
                    if action == create_action:
                        self.create_new_class(selected_ids)

                logger.debug('Selected cluster IDs: %s', selected_ids)

            self.selector5 = RectangleSelector(ax, onselect,
                                               useblit=False,
                                               minspanx=0.01,
                                               minspany=0.01,
                                               spancoords='data',
                                               button=[1],
                                               interactive=True)
        
        elif subplot_idx == 5:
            ax = self.axes[1][2]
            ax.scatter(self.temporal_traces_pca_scores[:, 0], self.ACG_pca_scores[:, 0],
                       marker='o', facecolors='none', edgecolors='k',
                       linewidths=subplots_lw, s = subplots_marker_size)
            ax.set_xlabel("TF 1")
            ax.set_ylabel("Autocorrelation 1")
            self.canvas.draw()

            def onselect(eclick, erelease):
                x1, y1 = eclick.xdata, eclick.ydata
                x2, y2 = erelease.xdata, erelease.ydata

                xmin, xmax = sorted([x1, x2])
                ymin, ymax = sorted([y1, y2])

                flag_ROI = ((self.temporal_traces_pca_scores[:, 0] >= xmin) &
                            (self.temporal_traces_pca_scores[:, 0] <= xmax) &
                            (self.ACG_pca_scores[:, 0] >= ymin) &
                            (self.ACG_pca_scores[:, 0] <= ymax))

                selected_ids = [
                    cid for cid, flag in zip(
                        self.cluster_ids, flag_ROI) if flag]

                # Trigger menu immediately when selection is made
                if selected_ids:
                    menu = QMenu(self)
                    create_action = menu.addAction(
                        f"Create Group from {len(selected_ids)} clusters")
                    action = menu.exec_(QCursor.pos())
                    if action == create_action:
                        self.create_new_class(selected_ids)

                logger.debug('Selected cluster IDs: %s', selected_ids)

            self.selector6 = RectangleSelector(ax, onselect,
                                               useblit=False,
                                               minspanx=0.01,
                                               minspany=0.01,
                                               spancoords='data',
                                               button=[1],
                                               interactive=True)

    def create_new_class(self, selected_ids):
        if self.main_window.data_manager is None:
            return

        from ..callbacks import populate_tree_view

        df = self.main_window.data_manager.cluster_df[
            self.main_window.data_manager.cluster_df['KSLabel'] != 'mua'
        ].copy()

        current_new_class_id = self.main_window.data_manager.new_class_id
        group_name = f"Nc{current_new_class_id}"
        df.loc[df['cluster_id'].isin(selected_ids), 'KSLabel'] = group_name
        self.main_window.data_manager.new_class_id += 1
        self.main_window.data_manager.cluster_df = df

        populate_tree_view(self.main_window, df)
