import numpy as np
import pandas as pd
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                            QComboBox, QLabel, QProgressBar, QMessageBox,
                            QSpinBox, QDialog, QTextEdit, QCheckBox)
from qtpy.QtCore import QThread, Signal, QObject
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path as MplPath
from mpl_toolkits.mplot3d import Axes3D, art3d, proj3d  # noqa: F401
import logging
import sklearn.cluster

from ...analysis import analysis_core

logger = logging.getLogger(__name__)


def extract_features_from_datamanager(dm, selected_cluster_ids=None, progress_signal=None):
    """
    Simplified and fast feature extraction based on old version.
    Returns features, cluster_ids, and metadata for SELECTED cells only.
    """
    if dm is None:
        raise ValueError("DataManager is not available (None).")

    if not hasattr(dm, "cluster_df") or dm.cluster_df is None:
        raise ValueError("cluster_df is not available on DataManager.")

    if not getattr(dm, "vision_available", False):
        raise ValueError("Vision data (STA/Params) is required for these metrics.")

    if progress_signal:
        progress_signal.emit("Gathering metrics for selected clusters...")

    features = []
    cluster_ids = []
    metadata = []

    # STA metrics we want to use as features (if available) - Fixed set like old version
    sta_feature_keys = [
        "Time to Peak (ms)",
        "Response Duration (ms)",
        "Zero Crossing (ms)",
        "FWHM (Duration)",
        "Biphasic Index",
        "SNR (std ratio)",
        "Response Integral",
        "Total Energy",
        "RF Area (sq stix)",
        "RF Ellipticity (σy/σx)",
    ]

    # If no selection provided, use all clusters
    if selected_cluster_ids is None:
        selected_cluster_ids = dm.cluster_df['cluster_id'].values
    
    total = len(selected_cluster_ids)

    for i, cid in enumerate(selected_cluster_ids):
        # Yield progress every 10 items
        if progress_signal and i % 10 == 0:
            progress_signal.emit(f"Processing cluster {i}/{total}...")

        vid = int(cid) + 1  # Vision ID

        # Skip if no STA data
        if not dm.vision_stas or vid not in dm.vision_stas:
            continue

        # Get STA data - simplified like old version
        sta_data = dm.vision_stas[vid]
        try:
            stafit = dm.vision_params.get_stafit_for_cell(vid)
        except Exception:
            stafit = None

        # ---- Get STA metrics via compute_sta_metrics - SIMPLIFIED ----
        metrics = None
        try:
            metrics = analysis_core.compute_sta_metrics(
                sta_data, stafit, dm.vision_params, vid
            )
        except Exception:
            metrics = None

        # Build STA feature vector from metrics - fill NaNs with 0
        sta_vals = []
        for key in sta_feature_keys:
            val = 0.0  # Default to 0 instead of NaN
            if metrics is not None and key in metrics:
                try:
                    val = float(metrics[key])
                    if np.isnan(val) or np.isinf(val):
                        val = 0.0
                except Exception:
                    val = 0.0
            sta_vals.append(val)

        # ---- Kilosort / cluster_df extras ----
        try:
            row = dm.cluster_df[
                dm.cluster_df['cluster_id'] == cid
            ].iloc[0]
        except Exception:
            continue

        isi_viol = float(row.get('isi_violations_pct', 0.0) or 0.0)
        n_spikes = int(row.get('n_spikes', 0) or 0)
        firing_rate = float(row.get('firing_rate_hz', 0.0) or 0.0)
        log_n_spikes = float(np.log1p(max(n_spikes, 0)))

        # Simple log energy estimation from STA if available
        log_energy = 0.0
        try:
            # Quick energy calculation from STA
            if sta_data is not None:
                energy = np.sum(sta_data**2)
                log_energy = float(np.log1p(energy))
        except Exception:
            log_energy = 0.0

        # ---- Final feature vector - like old version ----
        feat_vec = sta_vals + [log_energy, log_n_spikes, firing_rate, isi_viol]
        features.append(feat_vec)
        cluster_ids.append(cid)

        # ---- Simplified metadata ----
        kslabel = row.get('KSLabel', row.get('group', 'unsorted'))
        
        # Simple time_to_peak from metrics or fallback
        time_to_peak = sta_vals[0] if len(sta_vals) > 0 else 0.0
        biphasic_index = sta_vals[4] if len(sta_vals) > 4 else 0.0

        metadata.append({
            'KSLabel': kslabel,
            'isi_violations': isi_viol,
            'n_spikes': n_spikes,
            'firing_rate': firing_rate,
            'time_to_peak': time_to_peak,
            'biphasic_index': biphasic_index
        })

    if len(features) < 5:
        raise ValueError(f"Not enough valid clusters (only {len(features)} found, need > 5).")

    logger.info(f"Extracted features for {len(features)} clusters")
    return features, cluster_ids, metadata


class KMeansWorker(QObject):
    """Background worker for K-Means clustering."""
    finished = Signal(object)  # labels
    error = Signal(str)

    def __init__(self, embedding, k):
        super().__init__()
        self.embedding = embedding
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

            from sklearn.preprocessing import StandardScaler

            # Extract features using simplified helper (fast version) - FOR SELECTED CELLS
            self.progress.emit("Extracting features for selected cells...")
            features, cluster_ids, metadata = extract_features_from_datamanager(
                self.dm, self.selected_cluster_ids, self.progress)

            self.progress.emit(f"Running UMAP on {len(features)} selected cells...")

            # Standardization & UMAP - with optimized parameters
            X = np.array(features, dtype=float)
            X = np.nan_to_num(X)  # Replace any remaining NaNs/infs with 0

            # UMAP parameters optimized for speed
            reducer = umap.UMAP(
                n_neighbors=min(15, len(features) - 1),  # Adjust based on sample size
                min_dist=0.1,
                metric='euclidean',
                low_memory=True,
                n_jobs=-1,  # Use all cores for parallel processing
                n_components=self.n_components,
                verbose=False  # Disable verbose output for speed
            )

            # Scale data
            scaled_data = StandardScaler().fit_transform(X)
            
            # Fit UMAP
            embedding = reducer.fit_transform(scaled_data)

            meta_df = pd.DataFrame(metadata)
            meta_df['cluster_id'] = cluster_ids
            
            self.progress.emit(f"UMAP complete for {len(cluster_ids)} cells")
            self.finished.emit(embedding, cluster_ids, meta_df)

        except Exception as e:
            logger.exception("UMAP Worker failed")
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

        self.layout = QVBoxLayout(self)

        # --- Controls Row 1 ---
        ctrl_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run UMAP (2D)")
        self.run_btn.clicked.connect(self.run_umap)
        self.run_btn.setStyleSheet(
            "background-color: #4282DA; font-weight: bold;")

        self.run_3d_btn = QPushButton("Run UMAP (3D)")
        self.run_3d_btn.clicked.connect(self.run_umap_3d)
        self.run_3d_btn.setStyleSheet(
            "background-color: #2D6A4F; font-weight: bold;")

        self.color_combo = QComboBox()
        self.color_combo.addItems(
            ["KSLabel", "Firing Rate", "ISI Violations", "Time to Peak", "K-Means"])
        self.color_combo.currentTextChanged.connect(
            lambda: self.update_plot())

        self.progress = QProgressBar()
        self.progress.hide()

        ctrl_layout.addWidget(self.run_btn)
        ctrl_layout.addWidget(self.run_3d_btn)
        ctrl_layout.addWidget(QLabel("Color:"))
        ctrl_layout.addWidget(self.color_combo)
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
        self.auto_group_chk.setChecked(False)  # Safer to let user opt-in

        self.show_ids_btn = QPushButton("Show IDs")
        self.show_ids_btn.clicked.connect(self.show_group_ids)
        self.show_ids_btn.setEnabled(False)

        self.project_3d_chk = QCheckBox("Lasso 3D Proj")
        self.project_3d_chk.setToolTip("Allows Lasso selection in 3D mode via screen projection.")
        self.project_3d_chk.setChecked(True)

        cluster_layout.addWidget(QLabel("Clustering:"))
        cluster_layout.addWidget(self.k_spin)
        cluster_layout.addWidget(self.kmeans_btn)
        cluster_layout.addWidget(self.auto_group_chk)  # Added Checkbox
        cluster_layout.addWidget(self.show_ids_btn)
        cluster_layout.addWidget(self.project_3d_chk)
        cluster_layout.addStretch()

        self.layout.addLayout(ctrl_layout)
        self.layout.addLayout(cluster_layout)

        # Plot Initialization
        self.fig = Figure(facecolor='#1f1f1f')
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

        # Initialize default 2D axes
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1f1f1f')

        # Interaction state
        self.selector = LassoSelector(self.ax, self.on_select)
        self.selector.set_active(False)  # Enable only after plot

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
            # Get selected indexes from tree view (QTreeView with QStandardItemModel)
            tree_view = self.main_window.tree_view
            model = self.main_window.tree_model
            if not tree_view or not model or not tree_view.selectionModel():
                return None

            selected_indexes = tree_view.selectionModel().selectedIndexes()
            if not selected_indexes:
                # If nothing is selected, use all clusters
                logger.info("No cells selected, using all clusters")
                return None

            selected_ids = []
            for index in selected_indexes:
                item = model.itemFromIndex(index)
                if item:
                    # Check if item has cluster_id data stored in UserRole
                    cid = item.data(Qt.UserRole)
                    if cid is not None:
                        selected_ids.append(cid)
                    elif item.hasChildren():
                        # If it's a group item, get all children
                        for i in range(item.rowCount()):
                            child = item.child(i)
                            if child:
                                child_cid = child.data(Qt.UserRole)
                                if child_cid is not None:
                                    selected_ids.append(child_cid)

            if not selected_ids:
                logger.info("No valid cluster IDs found in selection, using all clusters")
                return None

            logger.info(f"Found {len(selected_ids)} selected cluster IDs")
            return selected_ids

        except Exception as e:
            logger.error(f"Error getting selected cluster IDs: {e}")
            return None

    def run_umap(self):
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

        self.run_btn.setEnabled(True)
        self.run_3d_btn.setEnabled(True)
        self.progress.hide()
        self.show_ids_btn.setEnabled(True)
        self.kmeans_btn.setEnabled(True)

        self._reset_workers()

        self.update_plot()
        
        # Re-initialize selector (needs to be attached to the new axes)
        if self.selector:
            self.selector.disconnect_events()
        self.selector = LassoSelector(self.ax, self.on_select)
        self.selector.set_active(True)

        mode_str = "3D UMAP" if self.is_3d else "2D UMAP"
        selection_info = "selected" if self.get_selected_cluster_ids() is not None else "all"
        self.main_window.status_bar.showMessage(
            f"{mode_str} Complete. {len(self.cluster_ids)} {selection_info} cells.")

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
        """
        Takes the K-Means labels and creates groups in the main data manager.
        Only affects the cells that were included in the UMAP (selected cells).
        """
        try:
            df = self.main_window.data_manager.cluster_df
            
            unique_labels = np.unique(labels)
            count = 0
            
            for lbl in unique_labels:
                # Find cluster IDs for this label (from the subset used in UMAP)
                subset_indices = np.where(labels == lbl)[0]
                group_cluster_ids = self.cluster_ids[subset_indices]
                
                # Create a group name
                group_name = f"Type_{lbl+1}"
                
                # Update DataFrame - ONLY for these specific cluster IDs
                df.loc[df['cluster_id'].isin(group_cluster_ids), 'KSLabel'] = group_name
                count += 1
            
            from ..callbacks import populate_tree_view
            populate_tree_view(self.main_window)
            
            QMessageBox.information(
                self, 
                "Auto-Group", 
                f"Successfully created {count} groups (Type_1...Type_{count}) for the selected cells."
            )
            
        except Exception as e:
            logger.error(f"Failed to auto-group: {e}")
            QMessageBox.warning(self, "Auto-Group Error", str(e))

    def update_plot(self, _color_mode=None):
        if self.embedding is None:
            return

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
            self.ax.set_facecolor('#1f1f1f')
            
            if self.selector:
                self.selector.disconnect_events()
            self.selector = LassoSelector(self.ax, self.on_select)
            self.selector.set_active(True)
        else:
            self.ax.clear()

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
        elif mode == "Firing Rate":
            if 'firing_rate' in self.metadata_df:
                c = self.metadata_df['firing_rate'].values
                cmap = 'plasma'
        elif mode == "ISI Violations":
            if 'isi_violations' in self.metadata_df:
                c = self.metadata_df['isi_violations'].values
                cmap = 'magma_r'
        elif mode == "Time to Peak":
            if 'time_to_peak' in self.metadata_df:
                c = self.metadata_df['time_to_peak'].values
                cmap = 'viridis'
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
            self.ax.set_xlabel('Dim 1', color='gray')
            self.ax.set_ylabel('Dim 2', color='gray')
            self.ax.set_zlabel('Dim 3', color='gray')
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

        if mode != "KSLabel" and not (mode == "K-Means" and is_discrete):
            self.cbar = self.fig.colorbar(scatter, ax=self.ax, pad=0.1 if self.is_3d else 0.05)
        
        # Add selection info to title
        selection_info = "selected" if self.get_selected_cluster_ids() is not None else "all"
        title_prefix = "UMAP (3D)" if self.is_3d else "UMAP (2D)"
        self.ax.set_title(
            f"{title_prefix} - {len(self.cluster_ids)} {selection_info} cells - Color: {mode}",
            color='white')
        self.ax.tick_params(colors='gray')
        self.canvas.draw()

    def show_group_ids(self):
        if self.metadata_df is None:
            return

        mode = self.color_combo.currentText()
        if mode not in ["KSLabel", "K-Means"]:
            QMessageBox.information(
                self,
                "Info",
                "Group IDs only available for discrete categories (KSLabel, K-Means).")
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
            df = self.main_window.data_manager.cluster_df
            df.loc[df['cluster_id'].isin(ids), 'KSLabel'] = name

            from ..callbacks import populate_tree_view
            populate_tree_view(self.main_window)