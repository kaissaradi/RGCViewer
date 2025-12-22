import numpy as np
import pandas as pd
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                            QComboBox, QLabel, QProgressBar, QMessageBox,
                            QSpinBox, QDialog, QTextEdit)
from qtpy.QtCore import QThread, Signal, QObject
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path as MplPath
import logging
import sklearn.cluster

from ...analysis import analysis_core

logger = logging.getLogger(__name__)


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
            # Local imports to avoid global import issues
            try:
                import umap
            except ImportError:
                self.error.emit("umap-learn library is not installed.")
                return

            from scipy.ndimage import gaussian_filter1d
            from sklearn.preprocessing import StandardScaler

            # Basic sanity checks
            if self.dm is None:
                self.error.emit("DataManager is not available (None).")
                return

            if not hasattr(self.dm, "cluster_df") or self.dm.cluster_df is None:
                self.error.emit("cluster_df is not available on DataManager.")
                return

            if not getattr(self.dm, "vision_available", False):
                self.error.emit(
                    "Vision data (STA/Params) is required for these metrics.")
                return

            self.progress.emit("Gathering metrics for all clusters...")

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

            # Get list of clusters
            all_ids = self.dm.cluster_df['cluster_id'].values
            total = len(all_ids)

            for i, cid in enumerate(all_ids):
                # Yield progress every 10 items to keep UI responsive
                if i % 10 == 0:
                    self.progress.emit(f"Processing cluster {i}/{total}...")

                vid = int(cid) + 1  # Vision ID

                if not self.dm.vision_stas or vid not in self.dm.vision_stas:
                    continue

                # Grab STA + STAFit
                sta_data = self.dm.vision_stas[vid]
                try:
                    stafit = self.dm.vision_params.get_stafit_for_cell(vid)
                except Exception:
                    stafit = None

                # Timecourse metrics from analysis_core
                try:
                    time_axis, tc_matrix, _ = analysis_core.get_sta_timecourse_data(
                        sta_data, stafit, self.dm.vision_params, vid
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

                # Zero-cross fallback (time)
                zero_cross_fallback = 0.0
                if primary_idx < len(smooth_trace) - 1 and time_axis is not None:
                    post = smooth_trace[primary_idx:]
                    zc_idx = np.where(np.diff(np.signbit(post)))[0]
                    if len(zc_idx) > 0:
                        zero_cross_fallback = float(
                            time_axis[primary_idx + zc_idx[0]])

                # Log energy from dominant channel
                log_energy = float(np.log1p(np.sum(energies)))

                # ---- STA metrics via compute_sta_metrics ----
                metrics = None
                try:
                    metrics = analysis_core.compute_sta_metrics(
                        sta_data, stafit, self.dm.vision_params, vid
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
                    row = self.dm.cluster_df[
                        self.dm.cluster_df['cluster_id'] == cid
                    ].iloc[0]
                except Exception:
                    # If something is weird with cluster_df, skip this cluster
                    continue

                # These are used as metadata AND as additional features
                isi_viol = float(row.get('isi_violations_pct', 0.0) or 0.0)
                n_spikes = int(row.get('n_spikes', 0) or 0)
                firing_rate = float(row.get('firing_rate_hz', 0.0) or 0.0)
                log_n_spikes = float(np.log1p(max(n_spikes, 0)))

                # ---- Final feature vector for UMAP ----
                # [STA metrics..., log_energy, log_n_spikes, firing_rate, isi_violations]
                feat_vec = sta_vals + [log_energy, log_n_spikes, firing_rate, isi_viol]
                features.append(feat_vec)
                cluster_ids.append(cid)

                # ---- Metadata for coloring / dialogs ----
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
                self.error.emit(
                    "Not enough valid clusters for UMAP (need > 5).")
                return

            self.progress.emit(f"Running UMAP on {len(features)} cells...")

            # 3. Standardization & UMAP
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


class UMAPPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.embedding = None
        self.cluster_ids = None
        self.metadata_df = None
        self.cbar = None  # Track colorbar

        self.layout = QVBoxLayout(self)

        # --- Controls Row 1 ---
        ctrl_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run UMAP")
        self.run_btn.clicked.connect(self.run_umap)
        self.run_btn.setStyleSheet(
            "background-color: #4282DA; font-weight: bold;")

        self.color_combo = QComboBox()
        self.color_combo.addItems(
            ["KSLabel", "Firing Rate", "ISI Violations", "Time to Peak", "K-Means"])
        self.color_combo.currentTextChanged.connect(
            lambda: self.update_plot())

        self.progress = QProgressBar()
        self.progress.hide()

        ctrl_layout.addWidget(self.run_btn)
        ctrl_layout.addWidget(QLabel("Color:"))
        ctrl_layout.addWidget(self.color_combo)
        ctrl_layout.addWidget(self.progress)
        ctrl_layout.addStretch()

        # --- Controls Row 2 (Clustering) ---
        cluster_layout = QHBoxLayout()
        self.k_spin = QSpinBox()
        self.k_spin.setRange(2, 20)
        self.k_spin.setValue(5)
        self.k_spin.setPrefix("k=")
        self.k_spin.setToolTip("Number of clusters for K-Means")

        self.kmeans_btn = QPushButton("Run K-Means")
        self.kmeans_btn.clicked.connect(self.run_kmeans)

        self.show_ids_btn = QPushButton("Show Cluster IDs")
        self.show_ids_btn.clicked.connect(self.show_group_ids)
        self.show_ids_btn.setEnabled(False)  # Enable only when data exists

        cluster_layout.addWidget(QLabel("Clustering:"))
        cluster_layout.addWidget(self.k_spin)
        cluster_layout.addWidget(self.kmeans_btn)
        cluster_layout.addWidget(self.show_ids_btn)
        cluster_layout.addStretch()

        self.layout.addLayout(ctrl_layout)
        self.layout.addLayout(cluster_layout)

        # Plot
        self.fig = Figure(facecolor='#1f1f1f')
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1f1f1f')
        self.layout.addWidget(self.canvas)

        # Interaction state
        self.selector = LassoSelector(self.ax, self.on_select)
        self.selector.set_active(False)  # Enable only after plot

        # Store worker references to prevent garbage collection
        self.umap_worker_thread = None
        self.umap_worker = None
        self.kmeans_worker_thread = None
        self.kmeans_worker = None

    def run_umap(self):
        self.run_btn.setEnabled(False)
        self.progress.show()
        self.progress.setRange(0, 0)  # indeterminate

        self.umap_worker_thread = QThread()
        self.umap_worker = UMAPWorker(self.main_window.data_manager)
        self.umap_worker.moveToThread(self.umap_worker_thread)

        self.umap_worker_thread.started.connect(self.umap_worker.run)
        self.umap_worker.progress.connect(self.update_status)
        self.umap_worker.error.connect(self.on_error)
        self.umap_worker.finished.connect(self.on_umap_finished)

        self.umap_worker_thread.start()

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
        self.progress.hide()
        QMessageBox.critical(self, "UMAP Error", msg)
        self.clean_umap_thread()

    def on_kmeans_error(self, msg):
        self.kmeans_btn.setEnabled(True)
        self.progress.hide()
        QMessageBox.critical(self, "K-Means Error", msg)
        self.clean_kmeans_thread()

    def on_umap_finished(self, embedding, ids, metadata):
        self.embedding = np.asarray(embedding)
        self.cluster_ids = np.array(ids)
        self.metadata_df = metadata

        # Add cluster IDs to metadata for easier grouping later
        self.metadata_df['cluster_id'] = self.cluster_ids

        self.run_btn.setEnabled(True)
        self.progress.hide()
        self.show_ids_btn.setEnabled(True)

        self.clean_umap_thread()

        self.update_plot()
        self.selector.set_active(True)
        self.main_window.status_bar.showMessage(
            "UMAP Complete. Use Lasso (Mouse Drag) to select groups.")

    def on_kmeans_finished(self, labels):
        # Store labels
        self.metadata_df['K-Means'] = labels

        self.kmeans_btn.setEnabled(True)
        self.progress.hide()
        self.clean_kmeans_thread()

        # Update view
        self.color_combo.setCurrentText("K-Means")
        self.update_plot()
        self.main_window.status_bar.showMessage(
            "K-Means clustering complete.")

    def clean_umap_thread(self):
        if self.umap_worker_thread:
            self.umap_worker_thread.quit()
            self.umap_worker_thread.wait()
            self.umap_worker_thread = None
            self.umap_worker = None

    def clean_kmeans_thread(self):
        if self.kmeans_worker_thread:
            self.kmeans_worker_thread.quit()
            self.kmeans_worker_thread.wait()
            self.kmeans_worker_thread = None
            self.kmeans_worker = None

    def update_plot(self, _color_mode=None):
        if self.embedding is None:
            return

        # Clear colorbar if it exists to prevent duplication
        if getattr(self, "cbar", None):
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None

        self.ax.clear()
        mode = self.color_combo.currentText()

        c = 'cyan'  # Default
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
                # If selected but not run yet
                c = 'gray'

        scatter = self.ax.scatter(self.embedding[:, 0],
                                  self.embedding[:, 1],
                                  c=c,
                                  cmap=cmap,
                                  s=15,
                                  alpha=0.8,
                                  edgecolors='none')

        # Add colorbar only for continuous or if desired for discrete
        if mode != "KSLabel" and not (mode == "K-Means" and is_discrete):
            self.cbar = self.fig.colorbar(scatter, ax=self.ax)
        elif mode == "K-Means" and is_discrete:
            # No colorbar for K-Means to keep it clean
            pass

        self.ax.set_title(
            f"UMAP Projection (n={len(self.cluster_ids)}) - Color: {mode}",
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

        # Group by the current mode
        groups = self.metadata_df.groupby(mode)['cluster_id'].apply(list)

        text_output = ""
        for group_name, ids in groups.items():
            text_output += f"=== Group {group_name} ({len(ids)} cells) ===\n"
            # Format IDs nicely (e.g., 10 per line)
            id_strs = [str(x) for x in sorted(ids)]
            chunked = [", ".join(id_strs[i:i + 10])
                       for i in range(0, len(id_strs), 10)]
            text_output += "\n".join(chunked)
            text_output += "\n\n"

        # Show in Dialog
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

            # Refresh Tree View (use package-relative import)
            from ..callbacks import populate_tree_view
            populate_tree_view(self.main_window)
