from __future__ import annotations
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import RectangleSelector, LassoSelector
from matplotlib.path import Path
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from qtpy.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QMenu,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QSplitter,
    QWidget,
    QPushButton,
)
from qtpy.QtGui import QCursor, QColor, QPalette
from qtpy.QtCore import QThread, Signal, Qt
from typing import TYPE_CHECKING, Optional, List

from .rf_map_widget import RFMapWidget

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  Design Tokens
# ──────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "bg": "#0f1117",
    "surface": "#171923",
    "border": "#2a2d3e",
    "text_primary": "#e8eaf0",
    "text_muted": "#6b7280",
    "accent": "#4f8ef7",
    "highlight": "#f97316",
    "grid": "#1e2130",
    "progress_fg": "#4f8ef7",
    "progress_bg": "#1e2130",
    "btn_active": "#4f8ef7",
    "btn_inactive": "#1e2130",
}

_MPL_RC = {
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor": PALETTE["surface"],
    "axes.edgecolor": PALETTE["border"],
    "axes.labelcolor": PALETTE["text_muted"],
    "axes.titlecolor": PALETTE["text_primary"],
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "semibold",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": PALETTE["grid"],
    "grid.linewidth": 0.8,
    "xtick.color": PALETTE["text_muted"],
    "ytick.color": PALETTE["text_muted"],
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.pad": 4,
    "ytick.major.pad": 4,
    "text.color": PALETTE["text_primary"],
    "font.family": "sans-serif",
    "font.sans-serif": ["IBM Plex Sans", "Helvetica Neue", "DejaVu Sans"],
    "savefig.facecolor": PALETTE["bg"],
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

QPushButton#tool_btn {{
    background-color: {PALETTE["btn_inactive"]};
    color: {PALETTE["text_muted"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 5px;
    padding: 4px 14px;
    font-size: 12px;
    font-weight: 500;
    min-width: 80px;
}}
QPushButton#tool_btn:hover {{
    background-color: {PALETTE["border"]};
    color: {PALETTE["text_primary"]};
}}
QPushButton#tool_btn[active="true"] {{
    background-color: {PALETTE["btn_active"]};
    color: #ffffff;
    border-color: {PALETTE["btn_active"]};
}}
"""


# ──────────────────────────────────────────────────────────────────────────────
#  Background worker
# ──────────────────────────────────────────────────────────────────────────────


class FeatureAnalysisWorker(QThread):
    """
    Background worker to compute features and PCA scores.
    Utilises the DataManager's robust get_cell_physics cache as SSOT.
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
            total = len(self.cluster_ids)
            self.progress.emit(f"Ensuring physics cache for {total} clusters...", 0)

            self.data_manager.ensure_physics_cache(self.cluster_ids)

            self.progress.emit(f"Extracting features for {total} clusters...", 10)

            feature_matrix, valid_ids, metadata = (
                self.data_manager.get_physics_feature_matrix(self.cluster_ids)
            )

            if len(valid_ids) == 0:
                self.progress.emit("No valid features found.", 100)
                self.finished.emit({})
                return

            n = len(valid_ids)
            n_comp = min(3, n)

            def _pad3(arr):
                if arr.shape[1] < 3:
                    return np.pad(arr, ((0, 0), (0, 3 - arr.shape[1])), mode="constant")
                return arr

            tc_pca_block = feature_matrix[:, :n_comp]
            acg_pca_block = feature_matrix[:, n_comp : 2 * n_comp]

            results = {
                "cluster_ids": valid_ids,
                "temporal_pca": _pad3(tc_pca_block),
                "acg_pca": _pad3(acg_pca_block),
                "rf_diameter": np.sqrt(np.array(metadata["RF Area"]) / np.pi),
                "time_to_peak": np.array(metadata["Time to Peak"]),
            }

            self.progress.emit("Done.", 100)
            self.finished.emit(results)

        except Exception as e:
            logger.error("Error in FeatureAnalysisWorker: %s", e, exc_info=True)
            self.finished.emit({})

    def stop(self):
        self.is_running = False


# ──────────────────────────────────────────────────────────────────────────────
#  Main Window
# ──────────────────────────────────────────────────────────────────────────────


class FeatureExtractionWindow(QDialog):
    """
    Pop-up window for feature extraction with linked brushing.

    Improvements over the original:
    • useblit=True on all selectors → rubber-band draws via XOR/blit, near-instant
    • draw_idle() instead of draw() when updating point colours → no blocking redraws
    • LassoSelector added alongside RectangleSelector; toggle with toolbar buttons
    • Window stays open after creating a new cluster group
    """

    _PLOT_META = [
        ("Temporal PCA", "PC 1", "PC 2"),
        ("RF vs Temporal PC1", "RF Diameter (µm)", "Temporal PC 1"),
        ("Time to Peak vs RF", "RF Diameter (µm)", "Time to Peak (frames)"),
        ("ACG PCA", "PC 1", "PC 2"),
        ("RF vs ACG PC1", "RF Diameter (µm)", "ACG PC 1"),
        ("Temporal vs ACG PC1", "Temporal PC 1", "ACG PC 1"),
    ]

    def __init__(self, main_window: "MainWindow", cluster_ids, parent=None):
        logger.debug(
            f"Initializing FeatureExtractionWindow with {len(cluster_ids)} clusters"
        )
        super().__init__(parent)
        self.main_window = main_window
        self.initial_cluster_ids = cluster_ids
        self.cluster_ids: list = []

        # Selection tool state: 'rect' | 'lasso'
        self._selection_mode = "rect"

        self._build_window()
        self._build_toolbar()
        self._build_loading_ui()
        self._build_canvas()
        self._start_worker()

    # ── Window shell ──────────────────────────────────────────────────────────

    def _build_window(self):
        self.setWindowTitle("Feature Extraction")
        self.setMinimumSize(900, 640)
        self.resize(1200, 820)
        self.setStyleSheet(DIALOG_STYLESHEET)

        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(PALETTE["bg"]))
        self.setPalette(pal)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(16, 14, 16, 14)
        self.main_layout.setSpacing(8)
        self.setLayout(self.main_layout)

    # ── Toolbar ───────────────────────────────────────────────────────────────

    def _build_toolbar(self):
        """Thin toolbar with Rectangle / Lasso toggle buttons."""
        toolbar = QWidget()
        toolbar.setFixedHeight(36)
        row = QHBoxLayout(toolbar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        label = QLabel("Selection tool:")
        label.setStyleSheet(
            f"color: {PALETTE['text_muted']}; font-size: 12px; padding-right: 4px;"
        )
        row.addWidget(label)

        self._btn_rect = self._make_tool_btn("▭  Rectangle", "rect")
        self._btn_lasso = self._make_tool_btn("⌇  Lasso", "lasso")

        row.addWidget(self._btn_rect)
        row.addWidget(self._btn_lasso)
        row.addStretch()

        hint = QLabel("Drag to select · right-click selection to create cluster")
        hint.setStyleSheet(
            f"color: {PALETTE['text_muted']}; font-size: 11px; font-style: italic;"
        )
        row.addWidget(hint)

        self.main_layout.addWidget(toolbar)
        self._update_tool_buttons()

    def _make_tool_btn(self, text: str, mode: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("tool_btn")
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(lambda _, m=mode: self._set_selection_mode(m))
        return btn

    def _update_tool_buttons(self):
        for btn, mode in [(self._btn_rect, "rect"), (self._btn_lasso, "lasso")]:
            active = self._selection_mode == mode
            btn.setProperty("active", "true" if active else "false")
            # Force stylesheet re-evaluation
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _set_selection_mode(self, mode: str):
        if mode == self._selection_mode:
            return
        self._selection_mode = mode
        self._update_tool_buttons()

        # Enable only the matching selector family; disable the other
        for sel in self._rect_selectors:
            sel.set_active(mode == "rect")
        for sel in self._lasso_selectors:
            sel.set_active(mode == "lasso")

        self.canvas.draw_idle()

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
        self.fig, self.axes = plt.subplots(2, 3, figsize=(12, 7), dpi=100)
        self.fig.set_layout_engine("constrained")

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # RF mosaic alongside the scatters: a selection that is a real cell type
        # should tile the array, and that is only checkable while brushing.
        self.rf_map = RFMapWidget(colors=PALETTE, title="RF Mosaic")

        self.plot_splitter = QSplitter(Qt.Horizontal)
        self.plot_splitter.addWidget(self.canvas)
        self.plot_splitter.addWidget(self.rf_map)
        self.plot_splitter.setStretchFactor(0, 3)
        self.plot_splitter.setStretchFactor(1, 1)
        self.plot_splitter.setSizes([900, 300])
        self.plot_splitter.hide()

        self.main_layout.addWidget(self.plot_splitter, stretch=1)

        self.scatter_artists: List[Optional[any]] = [None] * 6
        # Overlay scatters: one per axes, holds ONLY selected points (orange, larger)
        # We update these via set_offsets() — much faster than recoloring the base scatter
        self._overlay_artists: List[Optional[any]] = [None] * 6
        # Per-axes (x, y) arrays stored so highlight_selection can look up coordinates
        self._plot_data: List[Optional[tuple]] = [None] * 6

        # Separate lists so we can toggle them independently
        self._rect_selectors: list = []
        self._lasso_selectors: list = []

    # ── Worker ────────────────────────────────────────────────────────────────

    def _start_worker(self):
        self.worker = FeatureAnalysisWorker(
            self.main_window.data_manager, self.initial_cluster_ids
        )
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

        self.cluster_ids = results.get("cluster_ids", [])
        self.temporal_pca = results.get("temporal_pca", np.empty((0, 3)))
        self.acg_pca = results.get("acg_pca", np.empty((0, 3)))
        self.rf_diameter = results.get("rf_diameter", np.empty((0,)))
        self.time_to_peak = results.get("time_to_peak", np.empty((0,)))

        if len(self.cluster_ids) == 0:
            self.status_label.setText("No valid data found for the selected clusters.")
            self.loading_widget.show()
            return

        self.plot_splitter.show()
        self.rf_map.set_cells(self.main_window.data_manager, self.cluster_ids)
        self.draw_plots()

    # ── Plotting ──────────────────────────────────────────────────────────────

    def draw_plots(self):
        """Render all six scatter plots."""
        scatter_kwargs = dict(
            marker="o",
            s=28,
            linewidths=0,
            alpha=0.65,
            color=PALETTE["accent"],
            picker=5,
            zorder=3,
        )

        pca_t = self.temporal_pca
        pca_a = self.acg_pca
        rf = self.rf_diameter
        ttp = self.time_to_peak

        data_pairs = [
            (
                pca_t[:, 0] if len(pca_t) > 0 else np.empty(0),
                pca_t[:, 1] if len(pca_t) > 0 else np.empty(0),
            ),
            (rf, pca_t[:, 0] if len(pca_t) > 0 else np.empty(0)),
            (rf, ttp),
            (
                pca_a[:, 0] if len(pca_a) > 0 else np.empty(0),
                pca_a[:, 1] if len(pca_a) > 0 else np.empty(0),
            ),
            (rf, pca_a[:, 0] if len(pca_a) > 0 else np.empty(0)),
            (
                pca_t[:, 0] if len(pca_t) > 0 else np.empty(0),
                pca_a[:, 0] if len(pca_a) > 0 else np.empty(0),
            ),
        ]

        self._rect_selectors.clear()
        self._lasso_selectors.clear()

        for idx, ax in enumerate(self.axes.flat):
            ax.clear()
            title, xlabel, ylabel = self._PLOT_META[idx]
            x_data, y_data = data_pairs[idx]

            ax.set_facecolor(PALETTE["surface"])
            for spine in ax.spines.values():
                spine.set_edgecolor(PALETTE["border"])
                spine.set_linewidth(0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.set_title(
                title,
                pad=8,
                fontsize=10,
                fontweight="semibold",
                color=PALETTE["text_primary"],
            )
            ax.set_xlabel(xlabel, fontsize=8.5, color=PALETTE["text_muted"], labelpad=5)
            ax.set_ylabel(ylabel, fontsize=8.5, color=PALETTE["text_muted"], labelpad=5)
            ax.tick_params(colors=PALETTE["text_muted"], length=3, width=0.7)
            ax.grid(True, color=PALETTE["grid"], linewidth=0.8, linestyle="-", zorder=0)

            if len(x_data) > 0 and len(y_data) > 0:
                # Base scatter — drawn once, NEVER mutated again
                artist = ax.scatter(x_data, y_data, **scatter_kwargs)
                self.scatter_artists[idx] = artist

                # Overlay scatter — starts empty, updated cheaply via set_offsets()
                overlay = ax.scatter(
                    [],
                    [],
                    marker="o",
                    s=55,
                    linewidths=0.6,
                    edgecolors=PALETTE["bg"],
                    color=PALETTE["highlight"],
                    alpha=0.95,
                    zorder=4,
                )
                self._overlay_artists[idx] = overlay

                # Store data for coordinate lookup in highlight_selection
                self._plot_data[idx] = (x_data, y_data)

                x_pad = (x_data.max() - x_data.min()) * 0.05 or 0.1
                y_pad = (y_data.max() - y_data.min()) * 0.05 or 0.1
                ax.set_xlim(x_data.min() - x_pad, x_data.max() + x_pad)
                ax.set_ylim(y_data.min() - y_pad, y_data.max() + y_pad)

                self._attach_rect_selector(ax, x_data, y_data)
                self._attach_lasso_selector(ax, x_data, y_data)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=PALETTE["text_muted"],
                )

        self.fig.patch.set_facecolor(PALETTE["bg"])

        # Apply initial active state
        for sel in self._lasso_selectors:
            sel.set_active(self._selection_mode == "lasso")
        for sel in self._rect_selectors:
            sel.set_active(self._selection_mode == "rect")

        # Full render once — then snapshot the clean background for blitting
        self.canvas.draw()
        self._blit_bg = self.canvas.copy_from_bbox(self.fig.bbox)

    # ── Selection: Rectangle ──────────────────────────────────────────────────

    def _attach_rect_selector(self, ax, x_data: np.ndarray, y_data: np.ndarray):
        """
        Attach a RectangleSelector with useblit=True for instant rubber-banding.
        The selector fires only on mouse release; no blocking work happens during drag.
        """

        def onselect(eclick, erelease):
            if not self.cluster_ids:
                return
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            if None in (x1, y1, x2, y2):
                return
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])
            mask = (
                (x_data >= xmin)
                & (x_data <= xmax)
                & (y_data >= ymin)
                & (y_data <= ymax)
            )
            selected = np.where(mask)[0]
            self.highlight_selection(selected)
            self.show_context_menu(selected)

        rect = RectangleSelector(
            ax,
            onselect,
            useblit=True,  # ← key speed-up: XOR/blit rubber band
            button=[1],
            interactive=True,
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            props=dict(
                edgecolor=PALETTE["accent"],
                facecolor=PALETTE["accent"],
                alpha=0.12,
                linewidth=1.2,
                linestyle="--",
            ),
        )
        self._rect_selectors.append(rect)

    # ── Selection: Lasso ──────────────────────────────────────────────────────

    def _attach_lasso_selector(self, ax, x_data: np.ndarray, y_data: np.ndarray):
        """
        Attach a LassoSelector with useblit=True.
        Points inside the closed lasso path are selected on mouse release.
        """
        # Pre-build the (N, 2) array of point coordinates for this subplot
        pts = np.column_stack([x_data, y_data])

        def onselect(verts):
            if not self.cluster_ids or len(verts) < 3:
                return
            path = Path(verts)
            mask = path.contains_points(pts)
            selected = np.where(mask)[0]
            self.highlight_selection(selected)
            self.show_context_menu(selected)

        lasso = LassoSelector(
            ax,
            onselect,
            useblit=True,  # ← blitting keeps the lasso line fast
            button=[1],
            props=dict(
                color=PALETTE["highlight"],
                linewidth=1.5,
                linestyle="-",
                alpha=0.80,
            ),
        )
        lasso.set_active(False)  # starts inactive; activated by toolbar
        self._lasso_selectors.append(lasso)

    # ── Public aliases (keep external callers working) ────────────────────────

    def _setup_selector(self, ax, index, x_data, y_data):
        self._attach_rect_selector(ax, x_data, y_data)
        self._attach_lasso_selector(ax, x_data, y_data)

    def setup_selector(self, ax, index, x_data, y_data):
        self._setup_selector(ax, index, x_data, y_data)

    # ── Highlight ─────────────────────────────────────────────────────────────

    def highlight_selection(self, indices: np.ndarray):
        """
        True blit highlight — genuinely instant regardless of N.

        Pattern:
          1. Restore the clean background snapshot (no redraw)
          2. Update overlay scatter offsets (O(k), k = selected points)
          3. draw_artist() each overlay directly onto the canvas pixels
          4. blit() — pushes only the dirty rectangle to the screen

        The matplotlib render pipeline is never invoked.
        """
        for idx, overlay in enumerate(self._overlay_artists):
            if overlay is None:
                continue
            if len(indices) == 0:
                overlay.set_offsets(np.empty((0, 2)))
            else:
                plot_xy = self._plot_data[idx]
                if plot_xy is None:
                    continue
                x_data, y_data = plot_xy
                overlay.set_offsets(np.column_stack([x_data[indices], y_data[indices]]))

        # Restore clean background — wipes previous orange dots in one memcpy
        self.canvas.restore_region(self._blit_bg)

        # Stamp each overlay artist directly onto the canvas buffer
        for idx, overlay in enumerate(self._overlay_artists):
            if overlay is not None:
                self.axes.flat[idx].draw_artist(overlay)

        # Push the updated buffer to the screen — single GPU/X11 flush
        self.canvas.blit(self.fig.bbox)

        # Mirror the selection onto the RF mosaic. Indices are positions in
        # self.cluster_ids, so map them back to cluster IDs first.
        if getattr(self, "rf_map", None) is not None:
            selected_ids = [
                self.cluster_ids[i] for i in indices if 0 <= i < len(self.cluster_ids)
            ]
            self.rf_map.highlight(selected_ids)

    # ── Context menu ──────────────────────────────────────────────────────────

    def show_context_menu(self, selected_indices: np.ndarray):
        if len(selected_indices) == 0:
            return
        selected_ids = [self.cluster_ids[i] for i in selected_indices]
        menu = QMenu(self)
        n = len(selected_ids)
        label = f"Create group from {n} cluster{'s' if n != 1 else ''}"
        create_action = menu.addAction(label)

        # Add a "Clear selection" option as a convenience
        clear_action = menu.addAction("Clear selection")

        action = menu.exec(QCursor.pos())
        if action == create_action:
            self._create_new_class(selected_ids)
        elif action == clear_action:
            self.highlight_selection(np.array([], dtype=int))

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
        # ← self.close() removed: window stays open so you can keep selecting

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def resizeEvent(self, event):
        """Invalidate the blit background cache whenever the window is resized."""
        super().resizeEvent(event)
        # After resize matplotlib will redraw; re-snapshot on the next draw event
        if hasattr(self, "_blit_bg") and self._blit_bg is not None:
            self._blit_bg = None
            self.canvas.mpl_connect("draw_event", self._on_canvas_draw)

    def _on_canvas_draw(self, event):
        """Re-snapshot the background after a full redraw (e.g. after resize)."""
        self._blit_bg = self.canvas.copy_from_bbox(self.fig.bbox)

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        super().closeEvent(event)
