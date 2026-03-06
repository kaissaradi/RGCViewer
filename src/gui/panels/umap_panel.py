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


def extract_features_from_datamanager(dm, selected_cluster_ids=None, progress_signal=None):
    """
    Advanced Feature Extraction Strategy (Hybrid):
    1. Collect Raw Timecourses -> Run PCA -> Top 3 Components (Shape/Polarity/Kinetics)
    2. Collect Raw ACGs -> Run PCA -> Top 2 Components (Burstiness/Pattern)
    3. Collect Scalars -> Area, Ellipticity, Color, Energy.
    4. Weight and Concatenate.
    """
    if dm is None:
        raise ValueError("DataManager is not available (None).")

    # Check for Vision Data availability
    if not getattr(dm, "vision_stas", None):
        raise ValueError(
            "Vision STA data is not loaded. \n\n"
            "This panel requires STA shapes to classify RGCs. "
            "Please ensure 'Vision Integration' succeeded."
        )

    if selected_cluster_ids is None:
        if hasattr(dm, "cluster_df") and dm.cluster_df is not None:
            selected_cluster_ids = dm.cluster_df['cluster_id'].values
        else:
            raise ValueError("No clusters available in DataManager.")

    if progress_signal:
        progress_signal.emit("Gathering hybrid features (Shape + Pattern + Geometry)...")

    # Containers for raw arrays (for PCA)
    raw_timecourses = []
    raw_acgs = []
    
    # Containers for scalars
    scalar_features = []  # [Area, Ellipticity, ColorOpponency, LogEnergy]
    
    # Metadata for plotting
    metadata = []
    valid_cluster_ids = []

    total = len(selected_cluster_ids)

    # --- 1. GATHER DATA ---
    for i, cid in enumerate(selected_cluster_ids):
        # Yield progress every 10 items
        if progress_signal and i % 10 == 0:
            progress_signal.emit(f"Processing cluster {i}/{total}...")

        vid = int(cid) + 1  # Vision ID (1-based)

        # SKIP cells with no Vision data
        if vid not in dm.vision_stas:
            continue

        # A. GET TIMECOURSE (Shape)
        sta_data = dm.vision_stas[vid]
        try:
            stafit = dm.vision_params.get_stafit_for_cell(vid)
        except Exception:
            stafit = None

        _, tc_matrix, _ = analysis_core.get_sta_timecourse_data(
            sta_data, stafit, dm.vision_params, vid
        )

        if tc_matrix is None or tc_matrix.size == 0:
            continue

        # Normalize dominant channel
        energies = np.sum(tc_matrix**2, axis=0)
        dom_idx = np.argmax(energies)
        dom_trace = tc_matrix[:, dom_idx]
        
        abs_max = np.max(np.abs(dom_trace))
        if abs_max == 0: continue
        norm_trace = dom_trace / abs_max
        
        # Apply slight smoothing for PCA robustness
        smooth_trace = gaussian_filter1d(norm_trace, sigma=1)
        
        # Resample/Pad trace to fixed length (e.g. 30 points)
        target_len = 30
        if len(smooth_trace) > target_len:
            trace_feat = smooth_trace[:target_len]
        else:
            trace_feat = np.pad(smooth_trace, (0, target_len - len(smooth_trace)))
            
        raw_timecourses.append(trace_feat)

        # B. GET ACG (Pattern)
        try:
            lags, acg_norm = dm.get_acg_data(cid)
            if acg_norm is not None and len(acg_norm) > 0:
                # Normalize length to fixed bins (e.g. center +/- 25 points)
                center = len(acg_norm) // 2
                half_width = 25
                if center > half_width:
                    acg_feat = acg_norm[center-half_width : center+half_width]
                else:
                    acg_feat = np.pad(acg_norm, (0, 50-len(acg_norm)))
            else:
                acg_feat = np.zeros(50)
        except Exception:
            acg_feat = np.zeros(50)
            
        raw_acgs.append(acg_feat)

        # C. GET SCALARS (Geometry)
        area = np.pi * stafit.std_x * stafit.std_y if stafit else 0
        
        # Safer Ellipticity Calculation
        ellipticity = 0.0
        if stafit and stafit.std_x > 0:
            ellipticity = stafit.std_y / stafit.std_x
        
        # Safer Color Opponency Proxy
        color_opp = 0.0
        if tc_matrix.shape[1] == 3:
             sorted_e = np.sort(energies)[::-1]
             if sorted_e[0] > 0:
                 color_opp = sorted_e[1] / sorted_e[0]

        log_energy = np.log1p(np.sum(energies))

        scalar_features.append([area, ellipticity, color_opp, log_energy])
        
        # D. METADATA (For Coloring)
        polarity = robust_polarity(smooth_trace)
        
        # Estimate Time to Peak for display
        is_off = polarity == "OFF"
        primary_idx = np.argmin(smooth_trace) if is_off else np.argmax(smooth_trace)
        time_to_peak_display = primary_idx * (1000/60) / 30 * 30 # Approx ms
        
        try:
            row = dm.cluster_df[dm.cluster_df['cluster_id'] == cid].iloc[0]
            kslabel = row.get('KSLabel', row.get('group', 'unsorted'))
            fr = float(row.get('firing_rate_hz', 0.0))
            isi = float(row.get('isi_violations_pct', 0.0))
        except Exception:
            kslabel = 'unsorted'
            fr = 0.0
            isi = 0.0

        metadata.append({
            'KSLabel': kslabel,
            'Polarity': polarity,
            'Time to Peak': time_to_peak_display,
            'Firing Rate': fr,
            'isi_violations': isi,
            'Color Opponency': color_opp,
            'RF Area': area
        })
        
        valid_cluster_ids.append(cid)

    # Check sufficiency
    if len(valid_cluster_ids) < 5:
        raise ValueError(f"Not enough valid Vision clusters (found {len(valid_cluster_ids)}). Need at least 5.")

    # --- 2. PCA TRANSFORMATIONS ---
    if progress_signal: progress_signal.emit("Running PCA on Timecourses and ACGs...")
    
    # A. Timecourse PCA (Shape)
    X_tc = np.array(raw_timecourses)
    X_tc = np.nan_to_num(X_tc) # Safety 1
    
    pca_tc = PCA(n_components=min(3, X_tc.shape[0], X_tc.shape[1]))
    X_tc_pca = pca_tc.fit_transform(X_tc)
    
    # B. ACG PCA (Pattern)
    X_acg = np.array(raw_acgs)
    X_acg = np.nan_to_num(X_acg) # Safety 2
    
    norms = np.linalg.norm(X_acg, axis=1, keepdims=True)
    norms[norms==0] = 1
    X_acg = X_acg / norms
    
    pca_acg = PCA(n_components=min(2, X_acg.shape[0], X_acg.shape[1]))
    X_acg_pca = pca_acg.fit_transform(X_acg)

    # C. Prepare Scalars - THIS WAS THE LIKELY CULPRIT
    X_scalars = np.array(scalar_features)
    X_scalars = np.nan_to_num(X_scalars) # <--- ADDED CRITICAL NAN FIX HERE
    
    scaler_geo = RobustScaler()
    X_scalars_scaled = scaler_geo.fit_transform(X_scalars)

    # --- 3. APPLY WEIGHTS & CONCATENATE ---
    if progress_signal: progress_signal.emit("Applying feature weights...")

    X_tc_weighted = X_tc_pca * W_SHAPE
    X_acg_weighted = X_acg_pca * W_PATTERN
    X_scalars_weighted = X_scalars_scaled * W_GEOMETRY
    
    X_final = np.hstack([X_tc_weighted, X_acg_weighted, X_scalars_weighted])
    
    # Final safety check before UMAP
    X_final = np.nan_to_num(X_final) 
    
    logger.info(f"Extracted features for {len(X_final)} clusters")
    return X_final, valid_cluster_ids, metadata


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

            # Extract features using HYBRID helper
            self.progress.emit("Extracting features for selected cells...")
            features, cluster_ids, metadata = extract_features_from_datamanager(
                self.dm, self.selected_cluster_ids, self.progress)

            self.progress.emit(f"Running UMAP on {len(features)} selected cells...")

            # UMAP parameters optimized for speed and structure preservation
            # Note: We do NOT use StandardScaler here because features are already scaled/weighted
            reducer = umap.UMAP(
                n_neighbors=min(15, len(features) - 1),  # Adjust based on sample size
                min_dist=0.1,
                metric='euclidean',
                low_memory=True,
                n_jobs=-1,  # Use all cores for parallel processing
                n_components=self.n_components,
                verbose=False
            )

            # Fit UMAP directly on weighted feature matrix
            embedding = reducer.fit_transform(features)

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
        # Updated list of color options including new metrics
        self.color_combo.addItems(
            ["KSLabel", "Polarity", "K-Means", "Firing Rate", "ISI Violations", "Time to Peak", "RF Area", "Color Opponency"])
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
            
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.information(
                self, 
                "Auto-Group", 
                f"Successfully created {count} groups (Type_1...Type_{count}) for the selected cells."
            )
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to auto-group: {e}")
            from qtpy.QtWidgets import QMessageBox
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

        if mode != "KSLabel" and not (mode == "K-Means" and is_discrete) and not (mode == "Polarity" and is_discrete):
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