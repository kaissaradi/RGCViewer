import numpy as np
import pandas as pd
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                            QComboBox, QLabel, QProgressBar, QMessageBox,
                            QSpinBox, QDialog, QTextEdit, QCheckBox)
from qtpy.QtCore import QThread, Signal, QObject
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import LassoSelector, RectangleSelector
from matplotlib.path import Path as MplPath
from mpl_toolkits.mplot3d import Axes3D, art3d, proj3d  # noqa: F401
import logging
import sklearn.cluster

# --- Scientific Computing Imports ---
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.ndimage import gaussian_filter1d

from ...analysis import analysis_core

logger = logging.getLogger(__name__)

# --- WEIGHTING CONSTANTS ---
# Tuned for RGC Classification: 
# Shape (Polarity/Kinetics) > Pattern (Burstiness) > Geometry (Area)
W_SHAPE = 2.0       
W_PATTERN = 1.5     
W_GEOMETRY = 1.0    


def robust_polarity(trace):
    """
    Robust check for ON vs OFF based on absolute magnitude of peaks/troughs.
    """
    if trace is None or len(trace) == 0:
        return "Unknown"
    peak = np.max(trace)
    trough = np.min(trace)
    return "OFF" if abs(trough) > abs(peak) else "ON"

def extract_features_from_datamanager(dm, cluster_ids):
    """
    O(1) Feature Extraction.
    Pulls pre-calculated physics directly from DataManager cache, applies 
    RobustScaler to geometry, and PCA-compresses high-dimensional arrays.
    """
    valid_ids = []
    tc_list = []
    acg_list = []
    scalars_list = []
    
    # We must build a metadata dictionary so the UI can color the UMAP dots
    metadata = {
        'Time to Peak': [],
        'RF Area': [],
        'Ellipticity': []
    }

    for cid in cluster_ids:
        # 1. INSTANT O(1) CACHE LOOKUP
        metrics = dm.get_cell_physics(cid)
        
        tc = metrics.get('timecourse')
        acg = metrics.get('acg')
        
        if tc is not None and acg is not None:
            valid_ids.append(cid)
            tc_list.append(tc)
            acg_list.append(acg)
            
            area = metrics.get('rf_area') or 0.0
            ellip = metrics.get('ellipticity') or 0.0
            t2p = metrics.get('time_to_peak') or 0
            
            scalars_list.append([area, ellip])
            
            metadata['Time to Peak'].append(t2p)
            metadata['RF Area'].append(area)
            metadata['Ellipticity'].append(ellip)

    if not valid_ids:
        return np.array([]), [], {}

    # 2. Standardize Array Lengths
    max_tc_len = max(len(t) for t in tc_list)
    tc_mat = np.array([np.pad(t, (0, max_tc_len - len(t))) if len(t) < max_tc_len else t[:max_tc_len] for t in tc_list])

    max_acg_len = max(len(a) for a in acg_list)
    acg_mat = np.array([np.pad(a, (0, max_acg_len - len(a))) if len(a) < max_acg_len else a[:max_acg_len] for a in acg_list])

    scalars_mat = np.array(scalars_list)

    # 3a. Drop rows that contain NaN in any matrix (skip those units)
    nan_mask = (
        np.any(np.isnan(tc_mat), axis=1) |
        np.any(np.isnan(acg_mat), axis=1) |
        np.any(np.isnan(scalars_mat), axis=1)
    )
    if np.any(nan_mask):
        n_dropped = int(nan_mask.sum())
        logger.warning(f"Dropping {n_dropped} unit(s) with NaN features before UMAP")
        keep = ~nan_mask
        valid_ids   = [vid for vid, k in zip(valid_ids, keep) if k]
        tc_mat      = tc_mat[keep]
        acg_mat     = acg_mat[keep]
        scalars_mat = scalars_mat[keep]
        for key in metadata:
            metadata[key] = [v for v, k in zip(metadata[key], keep) if k]

    if len(valid_ids) == 0:
        return np.array([]), [], {}

    # 3b. Robust Normalization 
    if scalars_mat.shape[0] > 0 and scalars_mat.shape[1] > 0:
        scalars_mat = RobustScaler().fit_transform(scalars_mat)

    # 4. Pre-PCA Compression 
    n_comp = min(3, len(valid_ids))
    tc_pca = PCA(n_components=n_comp).fit_transform(tc_mat) if n_comp > 0 else np.zeros((len(valid_ids), 0))
    acg_pca = PCA(n_components=n_comp).fit_transform(acg_mat) if n_comp > 0 else np.zeros((len(valid_ids), 0))

    # 5. Final Weighted Concatenation
    final_features = np.hstack([
        tc_pca * W_SHAPE,
        acg_pca * W_PATTERN,
        scalars_mat * W_GEOMETRY
    ])

    return final_features, valid_ids, metadata

class KMeansWorker(QObject):
    """Background worker for K-Means clustering."""
    finished = Signal(object)  # labels
    error = Signal(str)

    def __init__(self, embedding, k):
        super().__init__()
        # Ensure contiguous array for thread safety
        self.embedding = np.array(embedding, copy=True)
        self.k = k

    def run(self):
        try:
            # Run K-Means
            kmeans = sklearn.cluster.KMeans(
                n_clusters=self.k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.embedding)
            self.finished.emit(labels)
        except Exception as e:
            logger.exception("K-Means failed")
            self.error.emit(str(e))


class UMAPWorker(QObject):
    """Background worker to compute features and run UMAP."""
    finished = Signal(object, object, object)  # embedding, cluster_ids, metadata_df
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, data_manager, selected_cluster_ids=None, n_components=2):
        super().__init__()
        self.dm = data_manager
        self.selected_cluster_ids = selected_cluster_ids
        self.n_components = n_components

    def run(self):
        try:
            try:
                import umap
            except ImportError:
                self.error.emit("umap-learn library is not installed.")
                return

            self.progress.emit("Extracting features...")
            
            # --- THE FIX: Intercept 'None' and swap it for ALL cluster IDs ---
            target_ids = self.selected_cluster_ids
            if target_ids is None:
                # If no specific cells are selected, run on the entire dataset
                target_ids = self.dm.cluster_df['cluster_id'].values

            features, cluster_ids, metadata = extract_features_from_datamanager(
                self.dm, target_ids)

            if len(features) == 0:
                self.error.emit("No valid features could be extracted for the selected cells.")
                return

            self.progress.emit(f"Running UMAP on {len(features)} cells...")

            reducer = umap.UMAP(
                n_neighbors=min(15, len(features) - 1),
                min_dist=0.1,
                metric='euclidean',
                low_memory=True,
                n_jobs=-1,
                n_components=self.n_components,
                verbose=False
            )

            embedding = reducer.fit_transform(features)

            # Reconstruct the metadata DataFrame
            meta_df = pd.DataFrame(metadata)
            meta_df['cluster_id'] = cluster_ids
            
            self.progress.emit(f"UMAP complete for {len(cluster_ids)} cells")
            self.finished.emit(embedding, cluster_ids, meta_df)

        except Exception as e:
            import logging
            logging.getLogger(__name__).exception("UMAP Worker failed")
            self.error.emit(str(e))


class UMAPPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.embedding = None
        self.cluster_ids = None
        self.metadata_df = None
        self.cbar = None
        self.is_3d = False
        self.selector = None

        self.layout = QVBoxLayout(self)

        # Define dark theme colors for initialization (will be updated by restyle_plots)
        self._umap_colors = {
            "accent": "#2E6DD4",
            "accent_positive": "#1A5C3A",
            "bg_panel": "#18191C",
        }

        # --- Controls Row 1 ---
        ctrl_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run UMAP (2D)")
        self.run_btn.clicked.connect(self.run_umap)
        self.run_btn.setStyleSheet(
            f"background-color: {self._umap_colors['accent']}; font-weight: bold;")

        self.run_3d_btn = QPushButton("Run UMAP (3D)")
        self.run_3d_btn.clicked.connect(self.run_umap_3d)
        self.run_3d_btn.setStyleSheet(
            f"background-color: {self._umap_colors['accent_positive']}; font-weight: bold;")

        self.color_combo = QComboBox()
        self.color_combo.addItems(
            ["KSLabel", "Polarity", "K-Means", "Firing Rate", "ISI Violations", "Time to Peak", "RF Area", "Color Opponency"])
        self.color_combo.currentTextChanged.connect(
            lambda: self.update_plot())

        # NEW: Selection Tool Toggle
        self.selector_combo = QComboBox()
        self.selector_combo.addItems(["Lasso Tool", "Rectangle Tool"])
        self.selector_combo.currentIndexChanged.connect(self.update_selector)

        self.progress = QProgressBar()
        self.progress.hide()

        ctrl_layout.addWidget(self.run_btn)
        ctrl_layout.addWidget(self.run_3d_btn)
        ctrl_layout.addWidget(QLabel("Color:"))
        ctrl_layout.addWidget(self.color_combo)
        ctrl_layout.addWidget(QLabel("Tool:"))
        ctrl_layout.addWidget(self.selector_combo)
        ctrl_layout.addWidget(self.progress)
        ctrl_layout.addStretch()

        # --- Controls Row 2 (Clustering & Options) ---
        cluster_layout = QHBoxLayout()

        # K-Means controls
        self.k_spin = QSpinBox()
        self.k_spin.setRange(2, 20)
        self.k_spin.setValue(5)
        self.k_spin.setPrefix("k=")
        self.k_spin.setToolTip("Number of clusters for K-Means")

        self.kmeans_btn = QPushButton("Run K-Means")
        self.kmeans_btn.clicked.connect(self.run_kmeans)

        # Auto-Group Checkbox
        self.auto_group_chk = QCheckBox("Auto-Group Tree")
        self.auto_group_chk.setToolTip(
            "If checked, finishing clustering will automatically create/overwrite "
            "groups in the main Tree View (e.g., 'Type_1', 'Type_2')."
        )
        self.auto_group_chk.setChecked(False)

        self.show_ids_btn = QPushButton("Show IDs")
        self.show_ids_btn.clicked.connect(self.show_group_ids)
        self.show_ids_btn.setEnabled(False)

        self.project_3d_chk = QCheckBox("Lasso 3D Proj")
        self.project_3d_chk.setToolTip("Allows Lasso selection in 3D mode via screen projection.")
        self.project_3d_chk.setChecked(True)

        cluster_layout.addWidget(QLabel("Clustering:"))
        cluster_layout.addWidget(self.k_spin)
        cluster_layout.addWidget(self.kmeans_btn)
        cluster_layout.addWidget(self.auto_group_chk)
        cluster_layout.addWidget(self.show_ids_btn)
        cluster_layout.addWidget(self.project_3d_chk)
        cluster_layout.addStretch()

        self.layout.addLayout(ctrl_layout)
        self.layout.addLayout(cluster_layout)

        # Plot Initialization
        self.fig = Figure(facecolor=self._umap_colors['bg_panel'])
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

        # Initialize default 2D axes
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self._umap_colors['bg_panel'])

        # NEW: Initialize empty selector state
        self.current_selector = None

        # Worker refs
        self.worker_thread = None
        self.worker = None
        self.kmeans_worker_thread = None
        self.kmeans_worker = None

    def _reset_workers(self):
        """Clean up any running workers."""
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
            self.worker = None

    def get_selected_cluster_ids(self):
        """Get currently selected cluster IDs from the main window tree view."""
        from qtpy.QtCore import Qt

        try:
            tree_view = self.main_window.tree_view
            model = self.main_window.tree_model
            if not tree_view or not model or not tree_view.selectionModel():
                return None

            selected_indexes = tree_view.selectionModel().selectedIndexes()
            if not selected_indexes:
                logger.info("No cells selected, using all clusters")
                return None

            # Use a set to prevent duplicates if a user selects both a parent AND its child
            selected_ids = set()

            # --- RECURSIVE HELPER ---
            def extract_cids_recursively(item):
                # Check if the current item is a cluster (leaf)
                cid = item.data(Qt.UserRole)
                if cid is not None:
                    selected_ids.add(cid)
                
                # Dig into children/sub-folders
                for i in range(item.rowCount()):
                    child = item.child(i)
                    if child:
                        extract_cids_recursively(child)

            for index in selected_indexes:
                item = model.itemFromIndex(index)
                if item:
                    extract_cids_recursively(item)

            if not selected_ids:
                logger.info("No valid cluster IDs found in selection, using all clusters")
                return None

            result_list = list(selected_ids)
            logger.info(f"Found {len(result_list)} selected cluster IDs")
            return result_list

        except Exception as e:
            logger.error(f"Error getting selected cluster IDs: {e}")
            return None
    def run_umap(self):
        dm = self.main_window.data_manager
        total_clusters = len(dm.cluster_df)
        feature_cache  = getattr(dm, 'feature_cache', {})
        valid_cached = sum(1 for v in feature_cache.values() if v.get('_computed'))
        if valid_cached < total_clusters:
            QMessageBox.warning(self, "Cache Warming Up",
                                f"Please wait for background caching to finish.\n\n"
                                f"Ready: {valid_cached} / {total_clusters} cells.\n"
                                f"Check the progress bar in the bottom right.")
            return
        self._reset_workers()
        self.run_btn.setEnabled(False)
        self.run_3d_btn.setEnabled(False)
        self.progress.show()
        self.progress.setRange(0, 0)

        # Get selected cluster IDs
        selected_cluster_ids = self.get_selected_cluster_ids()
        
        if selected_cluster_ids is not None and len(selected_cluster_ids) < 5:
            self.run_btn.setEnabled(True)
            self.run_3d_btn.setEnabled(True)
            self.progress.hide()
            QMessageBox.warning(self, "Insufficient Selection", 
                              f"Need at least 5 selected cells for UMAP. Only {len(selected_cluster_ids)} selected.")
            return

        self.worker_thread = QThread()
        self.worker = UMAPWorker(self.main_window.data_manager, 
                                selected_cluster_ids=selected_cluster_ids,
                                n_components=2)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_status)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_processing_finished)

        self.worker_thread.start()

    def run_umap_3d(self):
        dm = self.main_window.data_manager
        total_clusters = len(dm.cluster_df)
        feature_cache  = getattr(dm, 'feature_cache', {})
        valid_cached = sum(1 for v in feature_cache.values() if v.get('_computed'))
        if valid_cached < total_clusters:
            QMessageBox.warning(self, "Cache Warming Up",
                                f"Please wait for background caching to finish.\n\n"
                                f"Ready: {valid_cached} / {total_clusters} cells.\n"
                                f"Check the progress bar in the bottom right.")
            return
        self._reset_workers()
        self.run_btn.setEnabled(False)
        self.run_3d_btn.setEnabled(False)
        self.progress.show()
        self.progress.setRange(0, 0)

        # Get selected cluster IDs
        selected_cluster_ids = self.get_selected_cluster_ids()
        
        if selected_cluster_ids is not None and len(selected_cluster_ids) < 5:
            self.run_btn.setEnabled(True)
            self.run_3d_btn.setEnabled(True)
            self.progress.hide()
            QMessageBox.warning(self, "Insufficient Selection", 
                              f"Need at least 5 selected cells for UMAP. Only {len(selected_cluster_ids)} selected.")
            return

        self.worker_thread = QThread()
        self.worker = UMAPWorker(self.main_window.data_manager, 
                                selected_cluster_ids=selected_cluster_ids,
                                n_components=3)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_status)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_processing_finished)

        self.worker_thread.start()

    def run_kmeans(self):
        if self.embedding is None:
            QMessageBox.warning(self, "No Data", "Please run UMAP first.")
            return

        self.kmeans_btn.setEnabled(False)
        self.progress.show()
        self.progress.setRange(0, 0)

        k = self.k_spin.value()

        self.kmeans_worker_thread = QThread()
        self.kmeans_worker = KMeansWorker(self.embedding, k)
        self.kmeans_worker.moveToThread(self.kmeans_worker_thread)

        self.kmeans_worker_thread.started.connect(self.kmeans_worker.run)
        self.kmeans_worker.error.connect(self.on_kmeans_error)
        self.kmeans_worker.finished.connect(self.on_kmeans_finished)

        self.kmeans_worker_thread.start()

    def update_status(self, msg):
        self.main_window.status_bar.showMessage(msg)

    def on_error(self, msg):
        self.run_btn.setEnabled(True)
        self.run_3d_btn.setEnabled(True)
        self.progress.hide()
        QMessageBox.critical(self, "Processing Error", msg)
        self._reset_workers()

    def on_kmeans_error(self, msg):
        self.kmeans_btn.setEnabled(True)
        self.progress.hide()
        QMessageBox.critical(self, "K-Means Error", msg)
        if self.kmeans_worker_thread:
            self.kmeans_worker_thread.quit()
            self.kmeans_worker_thread.wait()
            self.kmeans_worker_thread = None

    def on_processing_finished(self, embedding, ids, metadata):
        self.embedding = np.asarray(embedding)
        self.cluster_ids = np.array(ids)
        self.metadata_df = metadata

        # Determine if result is 3D
        self.is_3d = (self.embedding.shape[1] == 3)
        self.project_3d_chk.setVisible(self.is_3d)

        self.run_btn.setEnabled(True)
        self.run_3d_btn.setEnabled(True)
        self.progress.hide()
        self.show_ids_btn.setEnabled(True)
        self.kmeans_btn.setEnabled(True)

        self._reset_workers()

        self.update_plot()  # already calls update_selector() at the end

        mode_str = "3D UMAP" if self.is_3d else "2D UMAP"
        selection_info = "selected" if self.get_selected_cluster_ids() is not None else "all"
        self.main_window.status_bar.showMessage(
            f"{mode_str} Complete. {len(self.cluster_ids)} {selection_info} cells. (Shape={W_SHAPE}, Pattern={W_PATTERN}, Geo={W_GEOMETRY})")

    def on_kmeans_finished(self, labels):
        self.metadata_df['K-Means'] = labels
        self.kmeans_btn.setEnabled(True)
        self.progress.hide()
        if self.kmeans_worker_thread:
            self.kmeans_worker_thread.quit()
            self.kmeans_worker_thread.wait()
            self.kmeans_worker_thread = None

        self.color_combo.setCurrentText("K-Means")
        self.update_plot()
        self.main_window.status_bar.showMessage("K-Means clustering complete.")

        # --- AUTO-GROUP LOGIC ---
        if self.auto_group_chk.isChecked():
            self.apply_kmeans_grouping(labels)

    def apply_kmeans_grouping(self, labels):
        try:
            from ..callbacks import group_clusters_in_tree
            unique_labels = np.unique(labels)
            count = 0
            
            for lbl in unique_labels:
                subset_indices = np.where(labels == lbl)[0]
                group_cluster_ids = self.cluster_ids[subset_indices]
                group_name = f"Type_{lbl+1}"
                
                # Use our safe, in-place tree modifier!
                group_clusters_in_tree(self.main_window, group_cluster_ids, group_name)
                count += 1
            
            self.main_window.status_bar.showMessage(
                f"Auto-Group: created {count} groups (Type_1 … Type_{count})."
            )
            
            # Force the main tree view to collapse all groups after K-Means auto-grouping
            if hasattr(self.main_window, 'tree_view'):
                self.main_window.tree_view.collapseAll()
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to auto-group: {e}")
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Auto-Group Error", str(e))

    def restyle_plots(self, colors):
        """Updates plot styling based on the provided color scheme."""
        self.fig.patch.set_facecolor(colors['bg_panel'])
        self.ax.set_facecolor(colors['bg_panel'])
        
        # Update button colors
        self.run_btn.setStyleSheet(
            f"background-color: {colors['accent']}; font-weight: bold;")
        self.run_3d_btn.setStyleSheet(
            f"background-color: {colors['accent_positive']}; font-weight: bold;")
        
        # Update stored colors
        self._umap_colors = {
            "accent": colors['accent'],
            "accent_positive": colors['accent_positive'],
            "bg_panel": colors['bg_panel'],
        }
        
        self.update_plot()

    def update_plot(self, _color_mode=None):
        if self.embedding is None:
            return
            
        colors = self.main_window.get_current_colors()

        # Clean up old plot
        if getattr(self, "cbar", None):
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None

        # Re-create axes if dimension changed (2D vs 3D)
        current_is_3d = hasattr(self.ax, 'zaxis')
        
        if self.is_3d != current_is_3d:
            self.fig.clear()
            if self.is_3d:
                self.ax = self.fig.add_subplot(111, projection='3d')
            else:
                self.ax = self.fig.add_subplot(111)
            self.ax.set_facecolor(colors['bg_panel'])
            self.fig.patch.set_facecolor(colors['bg_panel'])
        else:
            self.ax.clear()
            self.ax.set_facecolor(colors['bg_panel'])

        # ... (rest of update_plot, updating hardcoded colors) ...
        # Determine Colors
        mode = self.color_combo.currentText()
        c = 'cyan'
        cmap = None
        is_discrete = False

        if mode == "KSLabel":
            if 'KSLabel' in self.metadata_df:
                labels = self.metadata_df['KSLabel'].values
                unique_labels = np.unique(labels)
                label_map = {l: i for i, l in enumerate(unique_labels)}
                c = [label_map.get(l, 0) for l in labels]
                cmap = 'tab10'
                is_discrete = True
        elif mode == "Polarity":
            if 'Polarity' in self.metadata_df:
                labels = self.metadata_df['Polarity'].values
                unique_labels = np.unique(labels)
                label_map = {l: i for i, l in enumerate(unique_labels)}
                c = [label_map.get(l, 0) for l in labels]
                cmap = 'coolwarm' 
                is_discrete = True
        elif mode == "Firing Rate":
            if 'Firing Rate' in self.metadata_df:
                c = self.metadata_df['Firing Rate'].values
                cmap = 'plasma'
        elif mode == "ISI Violations":
            if 'isi_violations' in self.metadata_df:
                c = self.metadata_df['isi_violations'].values
                cmap = 'magma_r'
        elif mode == "Time to Peak":
            if 'Time to Peak' in self.metadata_df:
                c = self.metadata_df['Time to Peak'].values
                cmap = 'viridis'
        elif mode == "RF Area":
            if 'RF Area' in self.metadata_df:
                c = self.metadata_df['RF Area'].values
                cmap = 'viridis'
        elif mode == "Color Opponency":
            if 'Color Opponency' in self.metadata_df:
                c = self.metadata_df['Color Opponency'].values
                cmap = 'cool'
        elif mode == "K-Means":
            if 'K-Means' in self.metadata_df:
                c = self.metadata_df['K-Means'].values
                cmap = 'tab20'
                is_discrete = True
            else:
                c = 'gray'

        # Draw Scatter
        if self.is_3d:
            scatter = self.ax.scatter(
                self.embedding[:, 0],
                self.embedding[:, 1],
                self.embedding[:, 2],
                c=c,
                cmap=cmap,
                s=20,
                alpha=0.8,
                edgecolors='none'
            )
            self.ax.set_xlabel('Dim 1', color=colors['text_secondary'])
            self.ax.set_ylabel('Dim 2', color=colors['text_secondary'])
            self.ax.set_zlabel('Dim 3', color=colors['text_secondary'])
            self.ax.xaxis.pane.fill = False
            self.ax.yaxis.pane.fill = False
            self.ax.zaxis.pane.fill = False
            self.ax.grid(False)
        else:
            scatter = self.ax.scatter(
                self.embedding[:, 0],
                self.embedding[:, 1],
                c=c,
                cmap=cmap,
                s=15,
                alpha=0.8,
                edgecolors='none'
            )

        if mode != "KSLabel" and not (mode == "K-Means" and is_discrete) and not (mode == "Polarity" and is_discrete):
            self.cbar = self.fig.colorbar(scatter, ax=self.ax, pad=0.1 if self.is_3d else 0.05)
            # Style colorbar ticks
            if self.cbar:
                self.cbar.ax.yaxis.set_tick_params(color=colors['text_secondary'])
                for label in self.cbar.ax.get_yticklabels():
                    label.set_color(colors['text_secondary'])
        
        # Add selection info to title
        selection_info = "selected" if self.get_selected_cluster_ids() is not None else "all"
        title_prefix = "UMAP (3D)" if self.is_3d else "UMAP (2D)"
        self.ax.set_title(
            f"{title_prefix} - {len(self.cluster_ids)} {selection_info} cells - Color: {mode}",
            color=colors['text_primary'])
        self.ax.tick_params(colors=colors['text_secondary'])

        # NEW: Re-attach the active selection tool to the fresh plot
        self.update_selector()

        self.canvas.draw()

    def show_group_ids(self):
        if self.metadata_df is None:
            return

        mode = self.color_combo.currentText()
        if mode not in ["KSLabel", "K-Means", "Polarity"]:
            QMessageBox.information(
                self,
                "Info",
                "Group IDs only available for discrete categories (KSLabel, K-Means, Polarity).")
            return

        if mode not in self.metadata_df:
            return

        groups = self.metadata_df.groupby(mode)['cluster_id'].apply(list)
        text_output = ""
        for group_name, ids in groups.items():
            text_output += f"=== Group {group_name} ({len(ids)} cells) ===\n"
            id_strs = [str(x) for x in sorted(ids)]
            chunked = [", ".join(id_strs[i:i + 10])
                       for i in range(0, len(id_strs), 10)]
            text_output += "\n".join(chunked)
            text_output += "\n\n"

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Cluster IDs ({mode})")
        dlg.resize(600, 400)
        l = QVBoxLayout(dlg)
        t = QTextEdit()
        t.setReadOnly(True)
        t.setText(text_output)
        l.addWidget(t)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        l.addWidget(close_btn)
        dlg.exec_()

    def on_select(self, verts):
        if self.embedding is None:
            return

        path = MplPath(verts)
        selected_ids = []

        if self.is_3d:
            if self.project_3d_chk.isChecked():
                try:
                    proj = self.ax.get_proj()
                    xs, ys, zs = self.embedding[:, 0], self.embedding[:, 1], self.embedding[:, 2]
                    x2, y2, _ = proj3d.proj_transform(xs, ys, zs, proj)
                    points_2d = np.column_stack((x2, y2))
                    mask = path.contains_points(points_2d)
                    selected_ids = self.cluster_ids[mask]
                except Exception as e:
                    logger.warning(f"3D selection projection failed: {e}")
                    return
            else:
                return
        else:
            mask = path.contains_points(self.embedding)
            selected_ids = self.cluster_ids[mask]

        if len(selected_ids) > 0:
            reply = QMessageBox.question(
                self,
                "Selection",
                f"Selected {len(selected_ids)} clusters.\nCreate a new Group?",
                QMessageBox.Yes | QMessageBox.No)

            if reply == QMessageBox.Yes:
                self.create_group(selected_ids)

    def create_group(self, ids):
        from qtpy.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(
            self, "Group Name", "Enter name for this cluster group:")
        if ok and name:
            from ..callbacks import group_clusters_in_tree
            group_clusters_in_tree(self.main_window, ids, name)

    def update_selector(self):
        """Hot-swaps between Lasso and Rectangle selection tools."""
        if not hasattr(self, 'ax'): 
            return
            
        # Clear existing selector safely
        if hasattr(self, 'current_selector') and self.current_selector is not None:
            self.current_selector.set_active(False)
            self.current_selector = None
            
        if self.selector_combo.currentText() == "Lasso Tool":
            self.current_selector = LassoSelector(self.ax, onselect=self.on_select)
        else:
            self.current_selector = RectangleSelector(
                self.ax, onselect=self.on_select_rect,
                useblit=True, button=[1], minspanx=5, minspany=5,
                spancoords='pixels', interactive=True
            )

    def on_select_rect(self, eclick, erelease):
        """Bridges RectangleSelector output into existing Lasso selection pipeline."""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        # Convert bounding box coordinates to a vertex path
        verts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
        
        # Pass directly into your existing lasso logic
        if hasattr(self, 'on_select'):
            self.on_select(verts)