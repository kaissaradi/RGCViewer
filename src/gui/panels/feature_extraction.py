from __future__ import annotations
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import RectangleSelector
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from sklearn.decomposition import PCA
from scipy.signal import correlate
from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QMenu, QLabel,
    QApplication, QProgressBar, QSizePolicy, QFrame, QWidget
)
from qtpy.QtGui import QCursor, QFont, QColor, QPalette
from qtpy.QtCore import QThread, Signal, Qt
from typing import TYPE_CHECKING, Optional, List

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  Design Tokens
# ──────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "bg":           "#0f1117",   # Near-black background
    "surface":      "#171923",   # Slightly lighter surface
    "border":       "#2a2d3e",   # Subtle border
    "text_primary": "#e8eaf0",   # Near-white text
    "text_muted":   "#6b7280",   # Muted labels
    "accent":       "#4f8ef7",   # Blue accent (default points)
    "highlight":    "#f97316",   # Orange highlight (selected points)
    "grid":         "#1e2130",   # Very subtle grid lines
    "progress_fg":  "#4f8ef7",
    "progress_bg":  "#1e2130",
}

# Matplotlib RC overrides — applied once at module import
_MPL_RC = {
    "figure.facecolor":      PALETTE["bg"],
    "axes.facecolor":        PALETTE["surface"],
    "axes.edgecolor":        PALETTE["border"],
    "axes.labelcolor":       PALETTE["text_muted"],
    "axes.titlecolor":       PALETTE["text_primary"],
    "axes.labelsize":        9,
    "axes.titlesize":        10,
    "axes.titleweight":      "semibold",
    "axes.grid":             True,
    "axes.axisbelow":        True,
    "grid.color":            PALETTE["grid"],
    "grid.linewidth":        0.8,
    "xtick.color":           PALETTE["text_muted"],
    "ytick.color":           PALETTE["text_muted"],
    "xtick.labelsize":       8,
    "ytick.labelsize":       8,
    "xtick.major.size":      3,
    "ytick.major.size":      3,
    "xtick.major.pad":       4,
    "ytick.major.pad":       4,
    "text.color":            PALETTE["text_primary"],
    "font.family":           "sans-serif",
    "font.sans-serif":       ["IBM Plex Sans", "Helvetica Neue", "DejaVu Sans"],
    "savefig.facecolor":     PALETTE["bg"],
}
mpl.rcParams.update(_MPL_RC)


# ──────────────────────────────────────────────────────────────────────────────
#  Stylesheet helpers
# ──────────────────────────────────────────────────────────────────────────────

DIALOG_STYLESHEET = f"""
QDialog {{
    background-color: {PALETTE["bg"]};
    color: {PALETTE["text_primary"]};
}}

QLabel#status_label {{
    color: {PALETTE["text_muted"]};
    font-size: 13px;
    letter-spacing: 0.04em;
    padding: 6px 0;
}}

QProgressBar {{
    background-color: {PALETTE["progress_bg"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 4px;
    height: 6px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {PALETTE["progress_fg"]};
    border-radius: 4px;
}}

QMenu {{
    background-color: {PALETTE["surface"]};
    color: {PALETTE["text_primary"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 6px;
    padding: 4px 0;
    font-size: 12px;
}}
QMenu::item {{
    padding: 7px 20px;
    border-radius: 3px;
}}
QMenu::item:selected {{
    background-color: {PALETTE["accent"]};
    color: #ffffff;
}}
"""


# ──────────────────────────────────────────────────────────────────────────────
#  Background worker (unchanged logic, cleaner progress messages)
# ──────────────────────────────────────────────────────────────────────────────

class FeatureAnalysisWorker(QThread):
    """
    Background worker to compute features and PCA scores.
    Utilises caching in DataManager to avoid re-computation.
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
            uncached_ids = [cid for cid in self.cluster_ids
                            if cid not in self.data_manager.feature_cache]

            if uncached_ids:
                total_steps = len(uncached_ids) * 3
                current_step = 0

                # 1. Temporal Traces
                temporal_traces = {}
                if (self.data_manager.vision_params and
                        hasattr(self.data_manager.vision_params, 'main_datatable') and
                        self.data_manager.vision_params.main_datatable is not None):

                    for i, cluster_index in enumerate(uncached_ids):
                        if not self.is_running:
                            return
                        current_step += 1
                        if i % 10 == 0:
                            self.progress.emit(
                                f"Loading temporal traces  {i + 1}/{len(uncached_ids)}",
                                int(current_step / total_steps * 100))
                        vision_cluster_index = cluster_index + 1
                        if vision_cluster_index in self.data_manager.vision_params.main_datatable:
                            red_tc = self.data_manager.vision_params.get_data_for_cell(
                                vision_cluster_index, 'RedTimeCourse')
                            if red_tc is not None:
                                temporal_traces[cluster_index] = red_tc

                # 2. ACG
                ACG = {}
                for i, cluster_index in enumerate(uncached_ids):
                    if not self.is_running:
                        return
                    current_step += 1
                    if i % 10 == 0:
                        self.progress.emit(
                            f"Computing autocorrelograms  {i + 1}/{len(uncached_ids)}",
                            int(current_step / total_steps * 100))

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
                                        acg_symmetric = acg_full[
                                            zero_lag_idx - lag_range:
                                            zero_lag_idx + lag_range + 1]
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
                        if (ACG_current is not None and len(ACG_current) > 1
                                and ACG_current[1] is not None):
                            ACG[cluster_index] = ACG_current[1]

                # 3. STA Fit (RF Diameter)
                stafit = {}
                if (self.data_manager.vision_params and
                        hasattr(self.data_manager.vision_params, 'main_datatable') and
                        self.data_manager.vision_params.main_datatable is not None):

                    for i, cluster_index in enumerate(uncached_ids):
                        if not self.is_running:
                            return
                        current_step += 1
                        if i % 10 == 0:
                            self.progress.emit(
                                f"Loading STA fits  {i + 1}/{len(uncached_ids)}",
                                int(current_step / total_steps * 100))
                        vision_cluster_index = cluster_index + 1
                        if vision_cluster_index in self.data_manager.vision_params.main_datatable:
                            stafit_current = self.data_manager.vision_params.get_stafit_for_cell(
                                vision_cluster_index)
                            if stafit_current is not None:
                                stafit[cluster_index] = [stafit_current.std_x, stafit_current.std_y]

                # Update cache
                for cid in uncached_ids:
                    self.data_manager.feature_cache[cid] = {
                        'temporal_trace': temporal_traces.get(cid),
                        'acg':            ACG.get(cid),
                        'stafit':         stafit.get(cid),
                    }

            # Consolidate
            valid_ids = []
            final_traces = []
            final_acg = []
            final_rf_diam = []
            final_time_to_peak = []

            for cid in self.cluster_ids:
                cached = self.data_manager.feature_cache.get(cid)
                if cached:
                    trace     = cached.get('temporal_trace')
                    acg       = cached.get('acg')
                    stafit_val = cached.get('stafit')
                    if trace is not None and acg is not None and stafit_val is not None:
                        valid_ids.append(cid)
                        final_traces.append(trace)
                        final_acg.append(acg)
                        final_rf_diam.append(np.sqrt(stafit_val[0] * stafit_val[1]))
                        final_time_to_peak.append(np.argmax(np.abs(trace)))

            final_traces      = np.array(final_traces)
            final_acg         = np.array(final_acg)
            final_rf_diam     = np.array(final_rf_diam)
            final_time_to_peak = np.array(final_time_to_peak)

            if len(valid_ids) > 0:
                pca_traces = (PCA(n_components=3).fit_transform(final_traces)
                              if final_traces.size > 0 else np.empty((0, 3)))
                pca_acg    = (PCA(n_components=3).fit_transform(final_acg)
                              if final_acg.size > 0 else np.empty((0, 3)))
            else:
                pca_traces = np.empty((0, 3))
                pca_acg    = np.empty((0, 3))

            results = {
                'cluster_ids':   valid_ids,
                'temporal_pca':  pca_traces,
                'acg_pca':       pca_acg,
                'rf_diameter':   final_rf_diam,
                'time_to_peak':  final_time_to_peak,
            }

            self.progress.emit("Finalizing…", 100)
            self.finished.emit(results)

        except Exception as e:
            logger.error(f"Error in FeatureAnalysisWorker: {e}", exc_info=True)
            self.finished.emit({})

    def stop(self):
        self.is_running = False


# ──────────────────────────────────────────────────────────────────────────────
#  Main Window
# ──────────────────────────────────────────────────────────────────────────────

class FeatureExtractionWindow(QDialog):
    """
    Pop-up window for feature extraction with linked brushing and lazy loading.
    Redesigned with a clean dark-scientific aesthetic.
    """

    # Plot descriptors: (title, x_label, y_label)
    _PLOT_META = [
        ("Temporal PCA",          "PC 1",          "PC 2"),
        ("RF vs Temporal PC1",    "RF Diameter (µm)", "Temporal PC 1"),
        ("Time to Peak vs RF",    "RF Diameter (µm)", "Time to Peak (frames)"),
        ("ACG PCA",               "PC 1",          "PC 2"),
        ("RF vs ACG PC1",         "RF Diameter (µm)", "ACG PC 1"),
        ("Temporal vs ACG PC1",   "Temporal PC 1", "ACG PC 1"),
    ]

    def __init__(self, main_window: "MainWindow", cluster_ids, parent=None):
        logger.debug(f"Initializing FeatureExtractionWindow with {len(cluster_ids)} clusters")
        super().__init__(parent)
        self.main_window = main_window
        self.initial_cluster_ids = cluster_ids
        self.cluster_ids: list = []

        self._build_window()
        self._build_loading_ui()
        self._build_canvas()
        self._start_worker()

    # ── Window shell ──────────────────────────────────────────────────────────

    def _build_window(self):
        self.setWindowTitle("Feature Extraction")
        self.setMinimumSize(900, 600)
        self.resize(1200, 780)
        self.setStyleSheet(DIALOG_STYLESHEET)

        # Dark window background via palette so Qt chrome matches
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(PALETTE["bg"]))
        self.setPalette(pal)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(16, 14, 16, 14)
        self.main_layout.setSpacing(10)
        self.setLayout(self.main_layout)

    # ── Loading UI ────────────────────────────────────────────────────────────

    def _build_loading_ui(self):
        self.loading_widget = QWidget()
        loading_layout = QVBoxLayout(self.loading_widget)
        loading_layout.setContentsMargins(0, 0, 0, 0)
        loading_layout.setSpacing(8)

        self.status_label = QLabel("Initializing…")
        self.status_label.setObjectName("status_label")
        self.status_label.setAlignment(Qt.AlignCenter)
        loading_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        loading_layout.addWidget(self.progress_bar)

        self.main_layout.addWidget(self.loading_widget)

    # ── Canvas ────────────────────────────────────────────────────────────────

    def _build_canvas(self):
        # Use constrained_layout (set via rcParams above) + explicit figsize
        self.fig, self.axes = plt.subplots(
            2, 3,
            figsize=(12, 7),
            dpi=100,
        )
        self.fig.set_layout_engine('constrained')

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.hide()

        self.main_layout.addWidget(self.canvas, stretch=1)

        self.scatter_artists: List[Optional[any]] = [None] * 6
        self.selectors: list = []

    # ── Worker ────────────────────────────────────────────────────────────────

    def _start_worker(self):
        self.worker = FeatureAnalysisWorker(
            self.main_window.data_manager, self.initial_cluster_ids)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    # ── Slots ─────────────────────────────────────────────────────────────────

    def on_progress(self, msg: str, value: int):
        self.status_label.setText(msg)
        self.progress_bar.setValue(value)

    def on_worker_finished(self, results: dict):
        self.loading_widget.hide()

        if not results:
            self.status_label.setText("Analysis failed — no data found.")
            self.loading_widget.show()
            return

        self.cluster_ids      = results.get("cluster_ids",   [])
        self.temporal_pca     = results.get("temporal_pca",  np.empty((0, 3)))
        self.acg_pca          = results.get("acg_pca",       np.empty((0, 3)))
        self.rf_diameter      = results.get("rf_diameter",   np.empty((0,)))
        self.time_to_peak     = results.get("time_to_peak",  np.empty((0,)))

        if len(self.cluster_ids) == 0:
            self.status_label.setText("No valid data found for the selected clusters.")
            self.loading_widget.show()
            return

        self.canvas.show()
        self.draw_plots()

    # ── Plotting ──────────────────────────────────────────────────────────────

    def draw_plots(self):
        """Render all six scatter plots with modern dark-scientific aesthetics."""

        n = len(self.cluster_ids)

        # Base scatter style — filled translucent circles, accent-coloured
        scatter_kwargs = dict(
            marker="o",
            s=28,
            linewidths=0,
            alpha=0.65,
            color=PALETTE["accent"],
            picker=5,
            zorder=3,
        )

        # Data pairs for each subplot
        pca_t = self.temporal_pca
        pca_a = self.acg_pca
        rf    = self.rf_diameter
        ttp   = self.time_to_peak

        data_pairs = [
            (pca_t[:, 0] if len(pca_t) > 0 else np.empty(0),
             pca_t[:, 1] if len(pca_t) > 0 else np.empty(0)),
            (rf,   pca_t[:, 0] if len(pca_t) > 0 else np.empty(0)),
            (rf,   ttp),
            (pca_a[:, 0] if len(pca_a) > 0 else np.empty(0),
             pca_a[:, 1] if len(pca_a) > 0 else np.empty(0)),
            (rf,   pca_a[:, 0] if len(pca_a) > 0 else np.empty(0)),
            (pca_t[:, 0] if len(pca_t) > 0 else np.empty(0),
             pca_a[:, 0] if len(pca_a) > 0 else np.empty(0)),
        ]

        self.selectors = []

        for idx, ax in enumerate(self.axes.flat):
            ax.clear()
            title, xlabel, ylabel = self._PLOT_META[idx]
            x_data, y_data = data_pairs[idx]

            # Style the axes
            ax.set_facecolor(PALETTE["surface"])
            for spine in ax.spines.values():
                spine.set_edgecolor(PALETTE["border"])
                spine.set_linewidth(0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.set_title(title, pad=8, fontsize=10, fontweight="semibold",
                         color=PALETTE["text_primary"])
            ax.set_xlabel(xlabel, fontsize=8.5, color=PALETTE["text_muted"], labelpad=5)
            ax.set_ylabel(ylabel, fontsize=8.5, color=PALETTE["text_muted"], labelpad=5)

            ax.tick_params(colors=PALETTE["text_muted"], length=3, width=0.7)
            ax.grid(True, color=PALETTE["grid"], linewidth=0.8, linestyle="-", zorder=0)

            if len(x_data) > 0 and len(y_data) > 0:
                artist = ax.scatter(x_data, y_data, **scatter_kwargs)
                self.scatter_artists[idx] = artist

                # Add a 5% margin so edge points aren't clipped
                x_pad = (x_data.max() - x_data.min()) * 0.05 or 0.1
                y_pad = (y_data.max() - y_data.min()) * 0.05 or 0.1
                ax.set_xlim(x_data.min() - x_pad, x_data.max() + x_pad)
                ax.set_ylim(y_data.min() - y_pad, y_data.max() + y_pad)

                self._setup_selector(ax, idx, x_data, y_data)
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9,
                        color=PALETTE["text_muted"])

        self.fig.patch.set_facecolor(PALETTE["bg"])
        self.canvas.draw()

    # ── Selection ─────────────────────────────────────────────────────────────

    def _setup_selector(self, ax, index: int, x_data: np.ndarray, y_data: np.ndarray):
        """Attach a RectangleSelector to an axes."""

        def onselect(eclick, erelease):
            if len(self.cluster_ids) == 0:
                return
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])
            mask = ((x_data >= xmin) & (x_data <= xmax) &
                    (y_data >= ymin) & (y_data <= ymax))
            selected_indices = np.where(mask)[0]
            self.highlight_selection(selected_indices)
            self.show_context_menu(selected_indices)

        rect = RectangleSelector(
            ax, onselect,
            useblit=False,
            button=[1],
            interactive=True,
            minspanx=5, minspany=5,
            spancoords="pixels",
            props=dict(
                edgecolor=PALETTE["accent"],
                facecolor=PALETTE["accent"],
                alpha=0.12,
                linewidth=1.2,
                linestyle="--",
            ),
        )
        self.selectors.append(rect)

    # Keep old name as alias so any external callers still work
    def setup_selector(self, ax, index, x_data, y_data):
        self._setup_selector(ax, index, x_data, y_data)

    def highlight_selection(self, indices: np.ndarray):
        """Highlight selected indices across ALL plots."""
        n = len(self.cluster_ids)
        base_color    = np.array([*mpl.colors.to_rgb(PALETTE["accent"]),    0.60])
        sel_color     = np.array([*mpl.colors.to_rgb(PALETTE["highlight"]), 0.95])

        colors = np.tile(base_color, (n, 1))
        sizes  = np.full(n, 28.0)

        if len(indices) > 0:
            colors[indices] = sel_color
            sizes[indices]  = 55.0   # Slightly larger for selected points

        for scatter in self.scatter_artists:
            if scatter is not None:
                scatter.set_facecolor(colors)
                scatter.set_sizes(sizes)

        self.canvas.draw()

    def show_context_menu(self, selected_indices: np.ndarray):
        if len(selected_indices) == 0:
            return
        selected_ids = [self.cluster_ids[i] for i in selected_indices]
        menu = QMenu(self)
        action_label = f"Create group from {len(selected_ids)} cluster{'s' if len(selected_ids) != 1 else ''}"
        create_action = menu.addAction(action_label)
        action = menu.exec(QCursor.pos())
        if action == create_action:
            self._create_new_class(selected_ids)

    # Keep old name as alias
    def create_new_class(self, selected_ids):
        self._create_new_class(selected_ids)

    def _create_new_class(self, selected_ids: list):
        if self.main_window.data_manager is None:
            return
        current_new_class_id = self.main_window.data_manager.new_class_id
        group_name = f"Nc{current_new_class_id}"
        self.main_window.data_manager.new_class_id += 1
        from ..callbacks import group_clusters_in_tree
        group_clusters_in_tree(self.main_window, selected_ids, group_name)
        logger.info(f"Created new group {group_name} with {len(selected_ids)} clusters")
        self.close()

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        super().closeEvent(event)



        