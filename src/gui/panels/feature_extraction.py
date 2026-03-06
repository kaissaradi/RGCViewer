from __future__ import annotations
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from sklearn.decomposition import PCA
from scipy.signal import correlate
from qtpy.QtWidgets import QDialog, QVBoxLayout, QMenu, QLabel, QApplication, QProgressBar
from qtpy.QtGui import QCursor
from qtpy.QtCore import QThread, Signal, Qt
from typing import TYPE_CHECKING, Optional, List

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


class FeatureAnalysisWorker(QThread):
    """
    Background worker to compute features and PCA scores.
    Now utilizes caching in DataManager to avoid re-computation.
    """
    finished = Signal(dict)
    progress = Signal(str, int)

    def __init__(self, data_manager, cluster_ids):
        super().__init__()
        self.data_manager = data_manager
        self.cluster_ids = cluster_ids
        self.is_running = True

    def run(self):
        try:
            # Check cache first
            # We need to see which cluster_ids are NOT in cache
            uncached_ids = [cid for cid in self.cluster_ids if cid not in self.data_manager.feature_cache]
            
            # Only compute for uncached
            if uncached_ids:
                total_steps = len(uncached_ids) * 3
                current_step = 0

                # 1. Temporal Traces
                temporal_traces = {} # Map cid -> trace
                
                if (self.data_manager.vision_params and 
                    hasattr(self.data_manager.vision_params, 'main_datatable') and
                    self.data_manager.vision_params.main_datatable is not None):
                    
                    for i, cluster_index in enumerate(uncached_ids):
                        if not self.is_running: return
                        current_step += 1
                        if i % 10 == 0:
                            self.progress.emit(f"Loading Traces ({i+1}/{len(uncached_ids)})...", 
                                             int(current_step / total_steps * 100))
                        
                        vision_cluster_index = cluster_index + 1
                        if vision_cluster_index in self.data_manager.vision_params.main_datatable:
                            red_tc = self.data_manager.vision_params.get_data_for_cell(
                                vision_cluster_index, 'RedTimeCourse')
                            if red_tc is not None:
                                temporal_traces[cluster_index] = red_tc

                # 2. ACG
                ACG = {} # Map cid -> acg
                
                for i, cluster_index in enumerate(uncached_ids):
                    if not self.is_running: return
                    current_step += 1
                    if i % 10 == 0:
                        self.progress.emit(f"Loading ACG ({i+1}/{len(uncached_ids)})...", 
                                         int(current_step / total_steps * 100))

                    # Try local compute first
                    acg_computed = False
                    try:
                        spikes = self.data_manager.get_cluster_spikes(cluster_index)
                        sr = self.data_manager.sampling_rate
                        if len(spikes) > 1:
                            limit_samples = int(300 * sr) 
                            if spikes[-1] > limit_samples:
                                spikes = spikes[spikes < limit_samples]
                            
                            if len(spikes) > 1:
                                spikes_ms = (spikes / sr * 1000.0).astype(int)
                                duration = int(spikes_ms[-1])
                                bin_width_ms = 1
                                bins = np.arange(0, duration + bin_width_ms, bin_width_ms)
                                binned_spikes, _ = np.histogram(spikes_ms, bins=bins)
                                
                                if binned_spikes.size > 0:
                                    centered = binned_spikes - np.mean(binned_spikes)
                                    acg_full = correlate(centered, centered, mode='full')
                                    zero_lag_idx = len(acg_full) // 2
                                    max_lag_ms = 100
                                    lag_range = min(int(max_lag_ms / bin_width_ms), zero_lag_idx)
                                    
                                    if lag_range > 0:
                                        acg_symmetric = acg_full[zero_lag_idx - lag_range: zero_lag_idx + lag_range + 1]
                                        acg_symmetric[lag_range] = 0 
                                        
                                        spike_variance = np.var(binned_spikes)
                                        if spike_variance != 0:
                                            acg_norm = acg_symmetric / spike_variance / len(binned_spikes)
                                        else:
                                            acg_norm = acg_symmetric.astype(float)
                                        
                                        ACG[cluster_index] = acg_norm
                                        acg_computed = True
                    except Exception:
                        pass

                    if not acg_computed:
                        ACG_current = self.data_manager.get_acg_data(cluster_index)
                        if ACG_current is not None and len(ACG_current) > 1 and ACG_current[1] is not None:
                            ACG[cluster_index] = ACG_current[1]

                # 3. STA Fit (RF Diameter)
                stafit = {}
                
                if (self.data_manager.vision_params and 
                    hasattr(self.data_manager.vision_params, 'main_datatable') and
                    self.data_manager.vision_params.main_datatable is not None):
                    
                    for i, cluster_index in enumerate(uncached_ids):
                        if not self.is_running: return
                        current_step += 1
                        if i % 10 == 0:
                            self.progress.emit(f"Loading STA Fits ({i+1}/{len(uncached_ids)})...", 
                                             int(current_step / total_steps * 100))
                        
                        vision_cluster_index = cluster_index + 1
                        if vision_cluster_index in self.data_manager.vision_params.main_datatable:
                            stafit_current = self.data_manager.vision_params.get_stafit_for_cell(vision_cluster_index)
                            if stafit_current is not None:
                                stafit[cluster_index] = [stafit_current.std_x, stafit_current.std_y]
                
                # Update Cache
                for cid in uncached_ids:
                    self.data_manager.feature_cache[cid] = {
                        'temporal_trace': temporal_traces.get(cid),
                        'acg': ACG.get(cid),
                        'stafit': stafit.get(cid)
                    }

            # --- Consolidate Data for Analysis ---
            # Now all requested IDs should be in cache (or have None if data missing)
            
            valid_ids = []
            final_traces = []
            final_acg = []
            final_rf_diam = []
            final_time_to_peak = []

            for cid in self.cluster_ids:
                cached = self.data_manager.feature_cache.get(cid)
                if cached:
                    trace = cached.get('temporal_trace')
                    acg = cached.get('acg')
                    stafit_val = cached.get('stafit')
                    
                    if trace is not None and acg is not None and stafit_val is not None:
                        valid_ids.append(cid)
                        final_traces.append(trace)
                        final_acg.append(acg)
                        final_rf_diam.append(np.sqrt(stafit_val[0] * stafit_val[1]))
                        # Time to Peak (index of max absolute value) * sampling interval
                        # Assuming Vision trace 
                        final_time_to_peak.append(np.argmax(np.abs(trace))) 

            final_traces = np.array(final_traces)
            final_acg = np.array(final_acg)
            final_rf_diam = np.array(final_rf_diam)
            final_time_to_peak = np.array(final_time_to_peak) # Units: samples/frames

            # PCA
            if len(valid_ids) > 0:
                pca_traces = PCA(n_components=3).fit_transform(final_traces) if final_traces.size > 0 else np.empty((0,3))
                pca_acg = PCA(n_components=3).fit_transform(final_acg) if final_acg.size > 0 else np.empty((0,3))
            else:
                pca_traces = np.empty((0,3))
                pca_acg = np.empty((0,3))

            results = {
                'cluster_ids': valid_ids,
                'temporal_pca': pca_traces,
                'acg_pca': pca_acg,
                'rf_diameter': final_rf_diam,
                'time_to_peak': final_time_to_peak
            }
            
            self.progress.emit("Finalizing...", 100)
            self.finished.emit(results)
            
        except Exception as e:
            logger.error(f"Error in FeatureAnalysisWorker: {e}", exc_info=True)
            self.finished.emit({}) 

    def stop(self):
        self.is_running = False


class FeatureExtractionWindow(QDialog):
    """
    Pop up window for feature extraction with linked brushing and lazy loading.
    """

    def __init__(self, main_window: MainWindow, cluster_ids, parent=None):
        logger.debug(f"Initializing FeatureExtractionWindow with {len(cluster_ids)} clusters")
        super().__init__(parent)
        self.main_window = main_window
        self.initial_cluster_ids = cluster_ids
        self.cluster_ids = [] # Will be updated with valid ones
        
        self.setWindowTitle('Feature Extraction')
        self.resize(1100, 700) 

        # Layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Status/Loading Bar
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.main_layout.addWidget(self.progress_bar)

        # Canvas
        self.fig = plt.figure()
        self.canvas = FigureCanvas(self.fig)
        self.axes = self.fig.subplots(2, 3)
        self.fig.subplots_adjust(wspace=0.3, hspace=0.3, left=0.08, right=0.95, bottom=0.08, top=0.95)
        self.main_layout.addWidget(self.canvas)
        self.canvas.hide() 

        self.scatter_artists: List[Optional[any]] = [None] * 6
        self.selectors = []

        # Start Worker
        self.worker = FeatureAnalysisWorker(self.main_window.data_manager, self.initial_cluster_ids)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_progress(self, msg, value):
        self.status_label.setText(msg)
        self.progress_bar.setValue(value)

    def on_worker_finished(self, results):
        self.progress_bar.hide()
        self.status_label.hide()
        self.canvas.show()
        
        if not results:
            self.status_label.setText("Analysis failed or no data found.")
            self.status_label.show()
            return

        self.cluster_ids = results.get('cluster_ids', [])
        self.temporal_pca = results.get('temporal_pca', np.empty((0,3)))
        self.acg_pca = results.get('acg_pca', np.empty((0,3)))
        self.rf_diameter = results.get('rf_diameter', np.empty((0,)))
        self.time_to_peak = results.get('time_to_peak', np.empty((0,)))
        
        if len(self.cluster_ids) == 0:
            self.status_label.setText("No valid data found for selected clusters.")
            self.status_label.show()
            return
            
        self.draw_plots()

    def draw_plots(self):
        # Common aesthetics
        scatter_kwargs = {
            'marker': 'o',
            'facecolors': 'none',
            'edgecolors': 'k', 
            'linewidths': 0.5,
            's': 20,
            'alpha': 0.7,
            'picker': 5
        }
        
        # Clear axes
        for ax_row in self.axes:
            for ax in ax_row:
                ax.clear()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

        # --- Plot 0: Temporal PC 1 vs 2 ---
        ax = self.axes[0, 0]
        if len(self.temporal_pca) > 0:
            self.scatter_artists[0] = ax.scatter(self.temporal_pca[:, 0], self.temporal_pca[:, 1], **scatter_kwargs)
            ax.autoscale(tight=True)
        ax.set_xlabel('Temporal PC 1')
        ax.set_ylabel('Temporal PC 2')
        self.setup_selector(ax, 0, self.temporal_pca[:, 0], self.temporal_pca[:, 1])

        # --- Plot 1: RF Diameter vs Temporal PC 1 ---
        ax = self.axes[0, 1]
        if len(self.rf_diameter) > 0 and len(self.temporal_pca) > 0:
            self.scatter_artists[1] = ax.scatter(self.rf_diameter, self.temporal_pca[:, 0], **scatter_kwargs)
            ax.autoscale(tight=True)
        ax.set_xlabel('RF Diameter (µm)')
        ax.set_ylabel('Temporal PC 1')
        self.setup_selector(ax, 1, self.rf_diameter, self.temporal_pca[:, 0])

        # --- Plot 2: Time to Peak vs RF Diameter (REPLACES HISTOGRAM) ---
        ax = self.axes[0, 2]
        if len(self.rf_diameter) > 0 and len(self.time_to_peak) > 0:
            # Note: time_to_peak is in frames/samples. 
            self.scatter_artists[2] = ax.scatter(self.rf_diameter, self.time_to_peak, **scatter_kwargs)
            ax.autoscale(tight=True)
        ax.set_xlabel("RF Diameter (µm)")
        ax.set_ylabel("Time to Peak (frames)")
        self.setup_selector(ax, 2, self.rf_diameter, self.time_to_peak)

        # --- Plot 3: ACG PC 1 vs 2 ---
        ax = self.axes[1, 0]
        if len(self.acg_pca) > 0:
            self.scatter_artists[3] = ax.scatter(self.acg_pca[:, 0], self.acg_pca[:, 1], **scatter_kwargs)
            ax.autoscale(tight=True)
        ax.set_xlabel("ACG PC 1")
        ax.set_ylabel("ACG PC 2")
        self.setup_selector(ax, 3, self.acg_pca[:, 0], self.acg_pca[:, 1])

        # --- Plot 4: RF vs ACG PC 1 ---
        ax = self.axes[1, 1]
        if len(self.rf_diameter) > 0 and len(self.acg_pca) > 0:
            self.scatter_artists[4] = ax.scatter(self.rf_diameter, self.acg_pca[:, 0], **scatter_kwargs)
            ax.autoscale(tight=True)
        ax.set_xlabel("RF Diameter (µm)")
        ax.set_ylabel("ACG PC 1")
        self.setup_selector(ax, 4, self.rf_diameter, self.acg_pca[:, 0])

        # --- Plot 5: Temporal PC 1 vs ACG PC 1 ---
        ax = self.axes[1, 2]
        if len(self.temporal_pca) > 0 and len(self.acg_pca) > 0:
            self.scatter_artists[5] = ax.scatter(self.temporal_pca[:, 0], self.acg_pca[:, 0], **scatter_kwargs)
            ax.autoscale(tight=True)
        ax.set_xlabel("Temporal PC 1")
        ax.set_ylabel("ACG PC 1")
        self.setup_selector(ax, 5, self.temporal_pca[:, 0], self.acg_pca[:, 0])

        self.canvas.draw()

    def setup_selector(self, ax, index, x_data, y_data):
        """
        Sets up a RectangleSelector for a given axes.
        """
        def onselect(eclick, erelease):
            if len(self.cluster_ids) == 0: return

            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])

            mask = (x_data >= xmin) & (x_data <= xmax) & (y_data >= ymin) & (y_data <= ymax)
            selected_indices = np.where(mask)[0]
            
            self.highlight_selection(selected_indices)
            self.show_context_menu(selected_indices)

        rect = RectangleSelector(ax, onselect,
                                useblit=False, 
                                button=[1],
                                interactive=True,
                                minspanx=5, minspany=5,
                                spancoords='pixels')
        self.selectors.append(rect)

    def highlight_selection(self, indices):
        """
        Highlights the selected indices in ALL plots.
        """
        default_color = 'black'
        selected_color = 'red'
        
        n_points = len(self.cluster_ids)
        colors = np.array([default_color] * n_points)
        if len(indices) > 0:
            colors[indices] = selected_color
            
        for i, scatter in enumerate(self.scatter_artists):
            if scatter is not None:
                scatter.set_edgecolors(colors)
        
        self.canvas.draw()

    def show_context_menu(self, selected_indices):
        if len(selected_indices) == 0: return
        
        selected_ids = [self.cluster_ids[i] for i in selected_indices]
        
        menu = QMenu(self)
        create_action = menu.addAction(f"Create Group from {len(selected_ids)} clusters")
        
        action = menu.exec(QCursor.pos())
        
        if action == create_action:
            self.create_new_class(selected_ids)

    def create_new_class(self, selected_ids):
        if self.main_window.data_manager is None: return

        current_new_class_id = self.main_window.data_manager.new_class_id
        group_name = f"Nc{current_new_class_id}"
        self.main_window.data_manager.new_class_id += 1
        
        # Use our safe, in-place tree modifier!
        from ..callbacks import group_clusters_in_tree
        group_clusters_in_tree(self.main_window, selected_ids, group_name)
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Created new group {group_name} with {len(selected_ids)} clusters")
        self.close()

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        super().closeEvent(event)
