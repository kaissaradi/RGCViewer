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

# --- IMPORT IAN & EMBED_UTILS ---
IAN_AVAILABLE = False
IAN_IMPORT_ERROR = ""

# try:
#     # Try importing the installed package
#     from ian.ian import *
#     from ian.utils import *
#     from ian.dset_utils import *
#     from ian.embed_utils import *
#     IAN_AVAILABLE = True
#     # Check if the function we need exists
# except ImportError:
#     # Fallback: Try importing local ian.py file in this directory
#     IAN_IMPORT_ERROR = str(e)

logger = logging.getLogger(__name__)


def extract_features_from_datamanager(dm, progress_signal=None):
    """
    Helper function to extract feature vectors from the DataManager.
    Used by both UMAPWorker and IANWorker to ensure consistent input data.
    """
    if dm is None:
        raise ValueError("DataManager is not available (None).")

    if not hasattr(dm, "cluster_df") or dm.cluster_df is None:
        raise ValueError("cluster_df is not available on DataManager.")

    if not getattr(dm, "vision_available", False):
        raise ValueError("Vision data (STA/Params) is required for these metrics.")

    if progress_signal:
        progress_signal.emit("Gathering metrics for all clusters...")

    features = []
    cluster_ids = []
    metadata = []

    # STA metrics we want to use as features (if available)
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

    from scipy.ndimage import gaussian_filter1d

    # Get list of clusters
    all_ids = dm.cluster_df['cluster_id'].values
    total = len(all_ids)

    for i, cid in enumerate(all_ids):
        # Yield progress every 10 items to keep UI responsive
        if progress_signal and i % 10 == 0:
            progress_signal.emit(f"Processing cluster {i}/{total}...")

        vid = int(cid) + 1  # Vision ID

        if not dm.vision_stas or vid not in dm.vision_stas:
            continue

        # Grab STA + STAFit
        sta_data = dm.vision_stas[vid]
        try:
            stafit = dm.vision_params.get_stafit_for_cell(vid)
        except Exception:
            stafit = None

        # Timecourse metrics from analysis_core
        try:
            time_axis, tc_matrix, _ = analysis_core.get_sta_timecourse_data(
                sta_data, stafit, dm.vision_params, vid
            )
        except Exception:
            time_axis, tc_matrix = None, None

        if tc_matrix is None:
            continue

        # Dominant channel trace
        energies = np.sum(tc_matrix**2, axis=0)
        dom_idx = int(np.argmax(energies))
        dom_trace = tc_matrix[:, dom_idx]

        # Normalize & Smooth
        abs_max = float(np.max(np.abs(dom_trace))) if dom_trace.size > 0 else 0.0
        if abs_max == 0:
            continue
        norm_trace = dom_trace / abs_max
        smooth_trace = gaussian_filter1d(norm_trace, sigma=1)

        # Fallback temporal features from timecourse
        peak_val = float(np.max(smooth_trace))
        trough_val = float(np.min(smooth_trace))
        is_off = abs(trough_val) > abs(peak_val)

        if is_off:
            primary_idx = int(np.argmin(smooth_trace))
        else:
            primary_idx = int(np.argmax(smooth_trace))

        if time_axis is not None and len(time_axis) > primary_idx:
            time_to_peak_fallback = float(time_axis[primary_idx])
        else:
            time_to_peak_fallback = float(primary_idx)

        # Biphasic index fallback
        secondary_val = 0.0
        if primary_idx < len(smooth_trace) - 1:
            post = smooth_trace[primary_idx:]
            if post.size > 0:
                secondary_val = float(np.max(post) if is_off else np.min(post))

        denom = (trough_val if is_off else peak_val)
        if denom != 0:
            biphasic_idx_fallback = float(abs(secondary_val / denom))
        else:
            biphasic_idx_fallback = 0.0

        # Log energy from dominant channel
        log_energy = float(np.log1p(np.sum(energies)))

        # ---- STA metrics via compute_sta_metrics ----
        metrics = None
        try:
            metrics = analysis_core.compute_sta_metrics(
                sta_data, stafit, dm.vision_params, vid
            )
        except Exception:
            metrics = None

        # Build STA feature vector from metrics
        sta_vals = []
        for key in sta_feature_keys:
            val = np.nan
            if metrics is not None and key in metrics:
                try:
                    val = float(metrics[key])
                except Exception:
                    val = np.nan
            sta_vals.append(val)

        # Derive metadata time_to_peak & biphasic_index from metrics if present
        if metrics is not None:
            ttp_meta = metrics.get("Time to Peak (ms)", time_to_peak_fallback)
            bi_meta = metrics.get("Biphasic Index", biphasic_idx_fallback)
        else:
            ttp_meta = time_to_peak_fallback
            bi_meta = biphasic_idx_fallback

        try:
            ttp_meta = float(ttp_meta)
        except Exception:
            ttp_meta = float(time_to_peak_fallback)

        try:
            bi_meta = float(bi_meta)
        except Exception:
            bi_meta = float(biphasic_idx_fallback)

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

        # ---- Final feature vector ----
        feat_vec = sta_vals + [log_energy, log_n_spikes, firing_rate, isi_viol]
        features.append(feat_vec)
        cluster_ids.append(cid)

        # ---- Metadata ----
        kslabel = row.get('KSLabel', row.get('group', 'unsorted'))

        metadata.append({
            'KSLabel': kslabel,
            'isi_violations': isi_viol,
            'n_spikes': n_spikes,
            'firing_rate': firing_rate,
            'time_to_peak': ttp_meta,
            'biphasic_index': bi_meta
        })

    if len(features) < 5:
        raise ValueError("Not enough valid clusters (need > 5).")

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

    def __init__(self, data_manager):
        super().__init__()
        self.dm = data_manager

    def run(self):
        try:
            try:
                import umap
            except ImportError:
                self.error.emit("umap-learn library is not installed.")
                return

            from sklearn.preprocessing import StandardScaler

            # Extract features using shared helper
            features, cluster_ids, metadata = extract_features_from_datamanager(
                self.dm, self.progress)

            self.progress.emit(f"Running UMAP on {len(features)} cells...")

            # Standardization & UMAP
            X = np.array(features, dtype=float)
            X = np.nan_to_num(X)  # Replace NaNs/infs with 0

            reducer = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean',
                low_memory=True,
                n_jobs=1
            )

            scaled_data = StandardScaler().fit_transform(X)
            embedding = reducer.fit_transform(scaled_data)

            meta_df = pd.DataFrame(metadata)
            self.finished.emit(embedding, cluster_ids, meta_df)

        except Exception as e:
            logger.exception("UMAP Worker failed")
            self.error.emit(str(e))


class IANWorker(QObject):
    """Background worker to run IAN and generate a 3D Diffusion Map."""
    finished = Signal(object, object, object)  # embedding (3D), cluster_ids, metadata_df
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, data_manager):
        super().__init__()
        self.dm = data_manager

    def run(self):
        try:
            # Check availability
            if not globals().get('IAN_AVAILABLE', False):
                self.error.emit(f"IAN libraries missing/error: {IAN_IMPORT_ERROR}")
                return

            import matplotlib.pyplot as plt
            from sklearn.preprocessing import StandardScaler

            # 1. Extract features
            features, cluster_ids, metadata = extract_features_from_datamanager(
                self.dm, self.progress)

            X = np.array(features, dtype=float)
            X = np.nan_to_num(X)
            X_scaled = StandardScaler().fit_transform(X)

            self.progress.emit(f"Running IAN (Adaptive Pruning) on {len(X)} cells...")

            # --- FIX: Monkeypatch plt.show to prevent thread crash ---
            original_show = plt.show
            plt.show = lambda *args, **kwargs: None 
            
            try:
                # 2. Run IAN
                # G: unweighted graph, wG: weighted kernel, sigmas: local scales
                # disc_pts: indices of disconnected points (outliers)
                G, wG, sigmas, disc_pts = IAN(
                    method="exact",
                    X=X_scaled,
                    metric="correlation",
                    verbose=1
                )
            finally:
                plt.show = original_show
            # ---------------------------------------------------------

            # 2b. Remove Outliers (Disconnected Points)
            # This cleans up the "floating points" the user saw
            if len(disc_pts) > 0:
                self.progress.emit(f"Removing {len(disc_pts)} outlier(s)...")
                
                # Create a mask of valid points
                n_samples = wG.shape[0]
                valid_mask = np.ones(n_samples, dtype=bool)
                valid_mask[list(disc_pts)] = False
                
                # Filter the Kernel Matrix (wG)
                # Slice rows and columns
                wG_clean = wG[valid_mask, :][:, valid_mask]
                
                # Filter metadata arrays
                cluster_ids = [cid for i, cid in enumerate(cluster_ids) if valid_mask[i]]
                metadata = [m for i, m in enumerate(metadata) if valid_mask[i]]
            else:
                wG_clean = wG

            self.progress.emit("Generating 3D Diffusion Map...")

            # 3. Diffusion Map Embedding
            embedding, eigenvals = diffusionMapFromK(
                wG_clean, 
                n_components=3, 
                alpha=1.0, 
                t=1
            )

            meta_df = pd.DataFrame(metadata)
            self.finished.emit(embedding, cluster_ids, meta_df)

        except Exception as e:
            logger.exception("IAN Worker failed")
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

        # IAN Button
        self.ian_btn = QPushButton("Run IAN (3D)")
        self.ian_btn.clicked.connect(self.run_ian)
        self.ian_btn.setStyleSheet(
            "background-color: #2D6A4F; font-weight: bold;")
        if not IAN_AVAILABLE:
            self.ian_btn.setEnabled(False)
            self.ian_btn.setToolTip("IAN dependencies missing")

        self.color_combo = QComboBox()
        self.color_combo.addItems(
            ["KSLabel", "Firing Rate", "ISI Violations", "Time to Peak", "K-Means"])
        self.color_combo.currentTextChanged.connect(
            lambda: self.update_plot())

        self.progress = QProgressBar()
        self.progress.hide()

        ctrl_layout.addWidget(self.run_btn)
        ctrl_layout.addWidget(self.ian_btn)
        ctrl_layout.addWidget(QLabel("Color:"))
        ctrl_layout.addWidget(self.color_combo)
        ctrl_layout.addWidget(self.progress)
        ctrl_layout.addStretch()

        # --- Controls Row 2 (Clustering & Options) ---
        cluster_layout = QHBoxLayout()
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
            "If checked, finishing K-Means will automatically create/overwrite "
            "groups in the main Tree View (e.g., 'Type_1', 'Type_2')."
        )
        self.auto_group_chk.setChecked(False) # Safer to let user opt-in

        self.show_ids_btn = QPushButton("Show IDs")
        self.show_ids_btn.clicked.connect(self.show_group_ids)
        self.show_ids_btn.setEnabled(False)

        self.project_3d_chk = QCheckBox("Lasso 3D Proj")
        self.project_3d_chk.setToolTip("Allows Lasso selection in 3D mode via screen projection.")
        self.project_3d_chk.setChecked(True)

        cluster_layout.addWidget(QLabel("Clustering:"))
        cluster_layout.addWidget(self.k_spin)
        cluster_layout.addWidget(self.kmeans_btn)
        cluster_layout.addWidget(self.auto_group_chk) # Added Checkbox
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

    def run_umap(self):
        self._reset_workers()
        self.run_btn.setEnabled(False)
        self.ian_btn.setEnabled(False)
        self.progress.show()
        self.progress.setRange(0, 0)

        self.worker_thread = QThread()
        self.worker = UMAPWorker(self.main_window.data_manager)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_status)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_processing_finished)

        self.worker_thread.start()

    def run_ian(self):
        self._reset_workers()
        self.run_btn.setEnabled(False)
        self.ian_btn.setEnabled(False)
        self.progress.show()
        self.progress.setRange(0, 0)

        self.worker_thread = QThread()
        self.worker = IANWorker(self.main_window.data_manager)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_status)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_processing_finished)

        self.worker_thread.start()

    def run_kmeans(self):
        if self.embedding is None:
            QMessageBox.warning(self, "No Data", "Please run UMAP or IAN first.")
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
        if IAN_AVAILABLE:
            self.ian_btn.setEnabled(True)
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

        # Add cluster IDs to metadata for easier grouping later
        self.metadata_df['cluster_id'] = self.cluster_ids

        self.run_btn.setEnabled(True)
        if IAN_AVAILABLE:
            self.ian_btn.setEnabled(True)
        self.progress.hide()
        self.show_ids_btn.setEnabled(True)

        self._reset_workers()

        self.update_plot()
        
        # Re-initialize selector (needs to be attached to the new axes)
        if self.selector:
            self.selector.disconnect_events()
        self.selector = LassoSelector(self.ax, self.on_select)
        self.selector.set_active(True)

        mode_str = "3D IAN Manifold" if self.is_3d else "2D UMAP"
        self.main_window.status_bar.showMessage(
            f"{mode_str} Complete. {len(self.cluster_ids)} cells.")

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
        else:
            # Optional: Ask user if they didn't check the box but might want to?
            pass

    def apply_kmeans_grouping(self, labels):
        """
        Takes the K-Means labels and creates groups in the main data manager.
        """
        try:
            df = self.main_window.data_manager.cluster_df
            
            # Reset groups to 'unsorted' or keep existing?
            # Strategy: Overwrite only cells present in this analysis.
            
            unique_labels = np.unique(labels)
            count = 0
            
            for lbl in unique_labels:
                # Find cluster IDs for this label
                # We need to map back from the subset used in IAN to the global DF
                # self.cluster_ids corresponds to labels indices
                
                # Get indices in the analysis subset
                subset_indices = np.where(labels == lbl)[0]
                
                # Get actual cluster IDs
                group_cluster_ids = self.cluster_ids[subset_indices]
                
                # Create a group name
                group_name = f"Type_{lbl+1}"
                
                # Update DataFrame
                df.loc[df['cluster_id'].isin(group_cluster_ids), 'KSLabel'] = group_name
                count += 1
            
            from ..callbacks import populate_tree_view
            populate_tree_view(self.main_window)
            
            QMessageBox.information(
                self, 
                "Auto-Group", 
                f"Successfully created {count} groups (Type_1...Type_{count}) in the Tree View."
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
            labels = self.metadata_df['KSLabel'].values
            unique_labels = np.unique(labels)
            label_map = {l: i for i, l in enumerate(unique_labels)}
            c = [label_map[l] for l in labels]
            cmap = 'tab10'
            is_discrete = True
        elif mode == "Firing Rate":
            c = self.metadata_df['firing_rate'].values
            cmap = 'plasma'
        elif mode == "ISI Violations":
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
        
        title_prefix = "IAN Diffusion Manifold (3D)" if self.is_3d else "UMAP Projection (2D)"
        self.ax.set_title(
            f"{title_prefix} (n={len(self.cluster_ids)}) - Color: {mode}",
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