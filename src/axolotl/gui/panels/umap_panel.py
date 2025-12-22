import numpy as np
import pandas as pd
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                            QComboBox, QLabel, QProgressBar, QMessageBox)
from qtpy.QtCore import QThread, Signal, QObject
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path as MplPath
import logging

from ...analysis import analysis_core

logger = logging.getLogger(__name__)


class UMAPWorker(QObject):
    """Background worker to compute features and run UMAP."""
    finished = Signal(
        object,
        object,
        object)  # embedding, cluster_ids, metadata_df
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, data_manager):
        super().__init__()
        self.dm = data_manager
        
    def run(self):
        try:
            from scipy.ndimage import gaussian_filter1d
            from sklearn.preprocessing import StandardScaler
            import umap
            if not self.dm.vision_available:
                self.error.emit(
                    "Vision data (STA/Params) is required for these metrics.")
                return

            self.progress.emit("Gathering metrics for all clusters...")

            features = []
            cluster_ids = []
            metadata = []

            # Get list of good/valid clusters
            # Filter for 'good' or 'mua' if you prefer, currently taking all
            all_ids = self.dm.cluster_df['cluster_id'].values

            total = len(all_ids)
            for i, cid in enumerate(all_ids):
                if i % 10 == 0:
                    self.progress.emit(f"Processing cluster {i}/{total}...")

                # 1. Get Basic Info
                vid = cid + 1  # Vision ID
                if vid not in self.dm.vision_stas:
                    continue

                # 2. Compute Numeric Metrics (Re-implementing simplified core logic for speed/floats)
                # We need raw floats, not the formatted strings from
                # analysis_core
                sta_data = self.dm.vision_stas[vid]
                stafit = self.dm.vision_params.get_stafit_for_cell(vid)

                # Timecourse metrics
                time_axis, tc_matrix, _ = analysis_core.get_sta_timecourse_data(
                    sta_data, stafit, self.dm.vision_params, vid)

                if tc_matrix is None:
                    continue

                # Find dominant channel
                energies = np.sum(tc_matrix**2, axis=0)
                dom_idx = np.argmax(energies)
                dom_trace = tc_matrix[:, dom_idx]

                # Normalize & Smooth
                abs_max = np.max(np.abs(dom_trace))
                if abs_max == 0:
                    continue
                norm_trace = dom_trace / abs_max
                smooth_trace = gaussian_filter1d(norm_trace, sigma=1)

                # Features
                peak_val = np.max(smooth_trace)
                trough_val = np.min(smooth_trace)
                is_off = abs(trough_val) > abs(peak_val)

                primary_idx = np.argmin(
                    smooth_trace) if is_off else np.argmax(smooth_trace)
                time_to_peak = time_axis[primary_idx]

                # Biphasic Index
                secondary_val = 0
                if is_off:
                    if primary_idx < len(smooth_trace) - 1:
                        secondary_val = np.max(smooth_trace[primary_idx:])
                else:
                    if primary_idx < len(smooth_trace) - 1:
                        secondary_val = np.min(smooth_trace[primary_idx:])

                biphasic_idx = abs(secondary_val /
                                   (trough_val if is_off else peak_val))

                # Zero Crossing
                # Simple check after peak
                zero_cross = 0
                if primary_idx < len(time_axis) - 1:
                    post_peak = smooth_trace[primary_idx:]
                    # Find sign change
                    zc_indices = np.where(np.diff(np.signbit(post_peak)))[0]
                    if len(zc_indices) > 0:
                        zero_cross = time_axis[primary_idx + zc_indices[0]]

                # Spatial (from fit)
                area = np.pi * stafit.std_x * stafit.std_y if stafit else 0
                ellipticity = (
                    stafit.std_y /
                    stafit.std_x) if (
                    stafit and stafit.std_x > 0) else 0

                # Add to lists
                features.append([
                    time_to_peak,
                    biphasic_idx,
                    zero_cross,
                    area,
                    ellipticity,
                    # Add total energy log
                    np.log1p(np.sum(energies))
                ])
                cluster_ids.append(cid)

                # Metadata for coloring
                row = self.dm.cluster_df[self.dm.cluster_df['cluster_id']
                                         == cid].iloc[0]
                metadata.append({
                    'KSLabel': row['KSLabel'],
                    'isi_violations': row['isi_violations_pct'],
                    'n_spikes': row['n_spikes'],
                    'firing_rate': row.get('firing_rate_hz', 0)
                })

            if len(features) < 5:
                self.error.emit(
                    "Not enough valid clusters for UMAP (need > 5).")
                return

            self.progress.emit(f"Running UMAP on {len(features)} cells...")

            # 3. Standardization & UMAP
            X = np.array(features)
            # Handle NaNs if any
            X = np.nan_to_num(X)

            reducer = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean')
            scaled_data = StandardScaler().fit_transform(X)
            embedding = reducer.fit_transform(scaled_data)

            self.finished.emit(embedding, cluster_ids, pd.DataFrame(metadata))

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

        self.layout = QVBoxLayout(self)

        # Controls
        ctrl_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run UMAP Analysis")
        self.run_btn.clicked.connect(self.run_umap)
        self.run_btn.setStyleSheet(
            "background-color: #4282DA; font-weight: bold;")

        self.color_combo = QComboBox()
        self.color_combo.addItems(
            ["KSLabel", "Firing Rate", "ISI Violations", "Time to Peak (proxy)"])
        self.color_combo.currentTextChanged.connect(self.update_plot)

        self.progress = QProgressBar()
        self.progress.hide()

        ctrl_layout.addWidget(self.run_btn)
        ctrl_layout.addWidget(QLabel("Color By:"))
        ctrl_layout.addWidget(self.color_combo)
        ctrl_layout.addWidget(self.progress)
        ctrl_layout.addStretch()
        self.layout.addLayout(ctrl_layout)

        # Plot
        self.fig = Figure(facecolor='#1f1f1f')
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1f1f1f')
        self.layout.addWidget(self.canvas)

        # Interaction state
        self.selector = LassoSelector(self.ax, self.on_select)
        self.selector.set_active(False)  # Enable only after plot

    def run_umap(self):
        self.run_btn.setEnabled(False)
        self.progress.show()
        self.progress.setRange(0, 0)  # indeterminate

        self.thread = QThread()
        self.worker = UMAPWorker(self.main_window.data_manager)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_status)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_finished)

        self.thread.start()

    def update_status(self, msg):
        self.main_window.status_bar.showMessage(msg)

    def on_error(self, msg):
        self.run_btn.setEnabled(True)
        self.progress.hide()
        QMessageBox.critical(self, "UMAP Error", msg)
        self.thread.quit()

    def on_finished(self, embedding, ids, metadata):
        self.embedding = embedding
        self.cluster_ids = np.array(ids)
        self.metadata_df = metadata
        self.run_btn.setEnabled(True)
        self.progress.hide()
        self.thread.quit()
        self.update_plot()
        self.selector.set_active(True)
        self.main_window.status_bar.showMessage(
            "UMAP Complete. Use Lasso (Mouse Drag) to select groups.")

    def update_plot(self, _color_mode=None):
        if self.embedding is None:
            return

        self.ax.clear()
        mode = self.color_combo.currentText()

        c = 'cyan'  # Default
        cmap = None

        if mode == "KSLabel":
            # Map labels to integers for coloring
            labels = self.metadata_df['KSLabel'].values
            unique_labels = np.unique(labels)
            label_map = {l: i for i, l in enumerate(unique_labels)}
            c = [label_map[l] for l in labels]
            cmap = 'tab10'
        elif mode == "Firing Rate":
            c = self.metadata_df['firing_rate'].values
            cmap = 'plasma'
        elif mode == "ISI Violations":
            c = self.metadata_df['isi_violations'].values
            cmap = 'magma_r'

        scatter = self.ax.scatter(self.embedding[:, 0], self.embedding[:, 1],
                                  c=c, cmap=cmap, s=15, alpha=0.8, edgecolors='none')

        if mode != "KSLabel":
            self.fig.colorbar(scatter, ax=self.ax)

        self.ax.set_title(
            f"UMAP Projection (n={len(self.cluster_ids)})",
            color='white')
        self.ax.tick_params(colors='gray')
        self.canvas.draw()

    def on_select(self, verts):
        if self.embedding is None:
            return

        path = MplPath(verts)
        mask = path.contains_points(self.embedding)
        selected_ids = self.cluster_ids[mask]

        if len(selected_ids) > 0:
            # Trigger the standard refinement/selection logic
            # For now, just print or show a dialog
            reply = QMessageBox.question(
                self,
                "Selection",
                f"Selected {len(selected_ids)} clusters.\nCreate a new Group?",
                QMessageBox.Yes | QMessageBox.No)

            if reply == QMessageBox.Yes:
                pass
                # You could call feature_extraction(self.main_window, selected_ids)
                # Or directly create a group:
                self.create_group(selected_ids)

    def create_group(self, ids):
        # reuse the logic from FeatureExtractionWindow.create_new_class
        # This requires importing QInputDialog to name it
        from qtpy.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(
            self, "Group Name", "Enter name for this cluster group:")
        if ok and name:
            df = self.main_window.data_manager.cluster_df
            df.loc[df['cluster_id'].isin(ids), 'KSLabel'] = name

            # Refresh Tree View
            from gui.callbacks import populate_tree_view
            populate_tree_view(self.main_window)
