from __future__ import annotations
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import RectangleSelector, LassoSelector
from matplotlib.path import Path
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.colors import to_rgba
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

QWidget#sel_panel {{
    background-color: {PALETTE["surface"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 6px;
}}
QWidget#sel_panel[armed="true"] {{
    border-color: {PALETTE["highlight"]};
}}

QLabel#sel_heading {{
    color: {PALETTE["text_muted"]};
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
}}
QLabel#sel_count {{
    color: {PALETTE["text_muted"]};
    font-size: 12px;
}}
QLabel#sel_count[armed="true"] {{
    color: {PALETTE["highlight"]};
    font-size: 14px;
    font-weight: 600;
}}

QPushButton#primary_btn {{
    background-color: {PALETTE["accent"]};
    color: #ffffff;
    border: 1px solid {PALETTE["accent"]};
    border-radius: 5px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 600;
}}
QPushButton#primary_btn:hover {{
    background-color: #6ba1f9;
}}
QPushButton#primary_btn:disabled {{
    background-color: {PALETTE["btn_inactive"]};
    border-color: {PALETTE["border"]};
    color: {PALETTE["text_muted"]};
}}

QPushButton#ghost_btn {{
    background-color: transparent;
    color: {PALETTE["text_muted"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 5px;
    padding: 6px 12px;
    font-size: 12px;
}}
QPushButton#ghost_btn:hover {{
    color: {PALETTE["text_primary"]};
    border-color: {PALETTE["text_muted"]};
}}
QPushButton#ghost_btn:disabled {{
    color: {PALETTE["border"]};
}}
"""


# ──────────────────────────────────────────────────────────────────────────────
#  Selector plumbing
# ──────────────────────────────────────────────────────────────────────────────


class _CompositedSelector:
    """Mixin: hand every repaint to the owning window's single blit compositor.

    Each matplotlib selector normally keeps its *own* background snapshot and
    blits independently. With six rectangle selectors, six lassos and the
    highlight overlay all sharing one canvas, whichever one painted last wiped
    the others — which is why the rubber band vanished the moment a selection
    landed. Routing everything through ``FeatureExtractionWindow._composite``
    gives one artist stack with a fixed draw order instead.
    """

    _owner = None

    def update(self):
        owner = self._owner
        if owner is None:
            return super().update()
        if not owner._defer_composite:
            owner._composite()
        return True

    def update_background(self, event):
        # The owner re-snapshots on draw_event. The base implementation forces
        # a full canvas.draw() on every button press, which is the stall this
        # panel used to have between clicking and seeing the rubber band.
        return

    def _press(self, event):
        if self._owner is not None:
            self._owner._begin_selection(self)
        return super()._press(event)


class LiveRectangleSelector(_CompositedSelector, RectangleSelector):
    """Rectangle whose extents are reported on every move, not just on release.

    That is what makes dragging an edge handle re-filter the population live
    rather than only when the mouse comes up.
    """

    def _onmove(self, event):
        owner = self._owner
        if owner is None:
            return super()._onmove(event)
        # Suppress the intermediate composite the base class triggers; the
        # selection is recomputed first so the shape and the points move together.
        owner._defer_composite = True
        try:
            super()._onmove(event)
        finally:
            owner._defer_composite = False
        owner._on_rect_live(self)


class LiveLassoSelector(_CompositedSelector, LassoSelector):
    """Lasso that previews its contents while you draw and stays on screen after."""

    def _onmove(self, event):
        owner = self._owner
        if owner is None:
            return super()._onmove(event)
        owner._defer_composite = True
        try:
            super()._onmove(event)
        finally:
            owner._defer_composite = False
        owner._on_lasso_live(self)

    def _release(self, event):
        verts = list(self.verts) if self.verts else []
        # The base implementation fires onselect and then throws the path away.
        super()._release(event)
        if self._owner is not None and len(verts) >= 3:
            self._owner._freeze_lasso(self, verts)


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

        # Single-compositor state. Everything that paints on the canvas goes
        # through _composite(); _defer_composite suppresses it while a selector
        # is mid-update so one mouse move produces exactly one blit.
        self._blit_bg = None
        self._defer_composite = False
        self._building = False
        self._selection = np.empty(0, dtype=int)

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

        hint = QLabel("Drag to select · drag the handles to refine · group it on the right")
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

        # Enable only the matching selector family; disable the other. The
        # outgoing family's shape is hidden so two selection outlines can never
        # be on screen at once, but the selected cells themselves are kept.
        for sel in self._rect_selectors:
            sel.set_active(mode == "rect")
            if mode != "rect":
                self._retire(sel)
        for sel in self._lasso_selectors:
            sel.set_active(mode == "lasso")
            if mode != "lasso":
                self._retire(sel)
        if mode != "lasso":
            for poly in self._lasso_polys:
                if poly is not None:
                    poly.set_visible(False)

        self._composite()

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

        # A full redraw invalidates the cached background. Snapshotting from
        # the draw_event (rather than only after resize) means the highlight
        # and the selection shape survive every repaint Qt asks for.
        self.canvas.mpl_connect("draw_event", self._on_canvas_draw)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_button)

        # RF mosaic alongside the scatters: a selection that is a real cell type
        # should tile the array, and that is only checkable while brushing.
        self.rf_map = RFMapWidget(colors=PALETTE, title="RF Mosaic")
        self.rf_map.cell_clicked.connect(self._on_rf_cell_clicked)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(8)
        side_layout.addWidget(self.rf_map, stretch=1)
        side_layout.addWidget(self._build_selection_panel())

        self.plot_splitter = QSplitter(Qt.Horizontal)
        self.plot_splitter.addWidget(self.canvas)
        self.plot_splitter.addWidget(side)
        self.plot_splitter.setStretchFactor(0, 3)
        self.plot_splitter.setStretchFactor(1, 1)
        self.plot_splitter.setSizes([900, 300])
        self.plot_splitter.hide()

        self.main_layout.addWidget(self.plot_splitter, stretch=1)

        self.scatter_artists: List[Optional[any]] = [None] * 6
        # Overlay scatters: one per axes, holds ONLY selected points (orange, larger).
        # animated=True keeps them out of the normal draw pass, so the cached
        # background stays clean instead of baking in a stale selection.
        self._overlay_artists: List[Optional[any]] = [None] * 6
        # Per-axes (x, y) arrays stored so highlight_selection can look up coordinates
        self._plot_data: List[Optional[tuple]] = [None] * 6
        # Frozen lasso outlines, one per axes — LassoSelector throws its own
        # path away on release, so we keep a copy to draw.
        self._lasso_polys: List[Optional[MplPolygon]] = [None] * 6

        # Separate lists so we can toggle them independently
        self._rect_selectors: list = []
        self._lasso_selectors: list = []

    # ── Selection panel ───────────────────────────────────────────────────────

    def _build_selection_panel(self) -> QWidget:
        """The 'make a group out of this' prompt, as a panel instead of a popup.

        It used to be a modal QMenu at the cursor, which froze the plot: you
        could not nudge the rectangle without dismissing it first. As a panel
        it stays out of the way and updates live while you drag.
        """
        panel = QWidget()
        panel.setObjectName("sel_panel")
        panel.setProperty("armed", "false")
        self._sel_panel = panel

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(6)

        heading = QLabel("SELECTION")
        heading.setObjectName("sel_heading")
        layout.addWidget(heading)

        self.sel_count_label = QLabel("Drag on a plot to select cells")
        self.sel_count_label.setObjectName("sel_count")
        self.sel_count_label.setProperty("armed", "false")
        self.sel_count_label.setWordWrap(True)
        layout.addWidget(self.sel_count_label)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        self.btn_create_group = QPushButton("Create group")
        self.btn_create_group.setObjectName("primary_btn")
        self.btn_create_group.setCursor(Qt.PointingHandCursor)
        self.btn_create_group.setEnabled(False)
        self.btn_create_group.clicked.connect(self._create_group_from_selection)

        self.btn_clear_selection = QPushButton("Clear")
        self.btn_clear_selection.setObjectName("ghost_btn")
        self.btn_clear_selection.setCursor(Qt.PointingHandCursor)
        self.btn_clear_selection.setEnabled(False)
        self.btn_clear_selection.clicked.connect(self.clear_selection)

        buttons.addWidget(self.btn_create_group, stretch=2)
        buttons.addWidget(self.btn_clear_selection, stretch=1)
        layout.addLayout(buttons)

        return panel

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

        # Old selectors keep their canvas callbacks alive even after ax.clear(),
        # so they must be disconnected or they go on handling events for axes
        # that no longer hold their data.
        for sel in self._rect_selectors + self._lasso_selectors:
            try:
                sel.disconnect_events()
            except Exception:
                logger.debug("selector disconnect failed", exc_info=True)
        self._rect_selectors.clear()
        self._lasso_selectors.clear()
        self._lasso_polys = [None] * 6
        self._selection = np.empty(0, dtype=int)

        self._building = True
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
                    animated=True,
                )
                self._overlay_artists[idx] = overlay

                # Store data for coordinate lookup in highlight_selection
                self._plot_data[idx] = (x_data, y_data)

                x_pad = (x_data.max() - x_data.min()) * 0.05 or 0.1
                y_pad = (y_data.max() - y_data.min()) * 0.05 or 0.1
                ax.set_xlim(x_data.min() - x_pad, x_data.max() + x_pad)
                ax.set_ylim(y_data.min() - y_pad, y_data.max() + y_pad)

                self._attach_rect_selector(ax, idx, x_data, y_data)
                self._attach_lasso_selector(ax, idx, x_data, y_data)
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
            self._retire(sel)
        for sel in self._rect_selectors:
            sel.set_active(self._selection_mode == "rect")
            self._retire(sel)

        self._building = False
        self._sync_selection_panel()
        # Full render once; _on_canvas_draw snapshots the clean background.
        self.canvas.draw()

    # ── Selection: Rectangle ──────────────────────────────────────────────────

    def _attach_rect_selector(self, ax, idx: int, x_data: np.ndarray, y_data: np.ndarray):
        """Attach an interactive rectangle whose extents drive the selection live."""

        def onselect(eclick, erelease):
            # The live handler has already applied this selection; the release
            # callback only has to leave the shape on screen.
            self._on_rect_live(rect)

        rect = LiveRectangleSelector(
            ax,
            onselect,
            useblit=True,
            button=[1],
            interactive=True,      # corner/edge handles, draggable after release
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            drag_from_anywhere=True,
            props=dict(
                edgecolor=PALETTE["text_primary"],
                facecolor=PALETTE["accent"],
                alpha=0.14,
                linewidth=1.2,
                linestyle="--",
            ),
            handle_props=dict(
                marker="s",
                markersize=6,
                markerfacecolor=PALETTE["text_primary"],
                markeredgecolor=PALETTE["bg"],
                alpha=0.95,
            ),
        )
        rect._owner = self
        rect._plot_index = idx
        self._rect_selectors.append(rect)

    # ── Selection: Lasso ──────────────────────────────────────────────────────

    def _attach_lasso_selector(self, ax, idx: int, x_data: np.ndarray, y_data: np.ndarray):
        """Attach a lasso that previews as you draw and leaves its outline behind."""

        def onselect(verts):
            self._apply_lasso_verts(idx, verts)

        lasso = LiveLassoSelector(
            ax,
            onselect,
            useblit=True,
            button=[1],
            props=dict(
                color=PALETTE["text_primary"],
                linewidth=1.4,
                linestyle="-",
                alpha=0.85,
            ),
        )
        lasso._owner = self
        lasso._plot_index = idx
        lasso.set_active(False)  # starts inactive; activated by toolbar
        self._lasso_selectors.append(lasso)

    # ── Public aliases (keep external callers working) ────────────────────────

    def _setup_selector(self, ax, index, x_data, y_data):
        self._attach_rect_selector(ax, index, x_data, y_data)
        self._attach_lasso_selector(ax, index, x_data, y_data)

    def setup_selector(self, ax, index, x_data, y_data):
        self._setup_selector(ax, index, x_data, y_data)

    # ── Live selection ────────────────────────────────────────────────────────

    @staticmethod
    def _retire(sel):
        """Hide a selector's shape and reset it, so its next press starts fresh.

        Hiding alone is not enough: matplotlib decides between "start a new
        rectangle" and "grab an existing handle" from the artist's visibility
        and ``_selection_completed``, so a merely-hidden selector goes on
        offering invisible handles.
        """
        clear = getattr(sel, "_clear_without_update", None)
        if callable(clear):
            clear()
        else:
            sel.set_visible(False)

    def _begin_selection(self, sel):
        """A new gesture started: retire every other shape so only one is shown.

        The pressed selector is deliberately left alone — matplotlib's own
        _press reads its current visibility to tell a fresh drag from a handle
        grab, and showing it here would make every new drag look like a grab.
        """
        for other in self._rect_selectors + self._lasso_selectors:
            if other is not sel:
                self._retire(other)
        for poly in self._lasso_polys:
            if poly is not None:
                poly.set_visible(False)

    def _on_rect_live(self, sel):
        """Recompute the selection from the rectangle's current extents.

        Called on every mouse move, so dragging an edge in or out re-filters
        the population as you go instead of only on release.
        """
        idx = getattr(sel, "_plot_index", None)
        plot_xy = self._plot_data[idx] if idx is not None else None
        if plot_xy is None or not self.cluster_ids:
            self._composite()
            return
        x_data, y_data = plot_xy
        xmin, xmax, ymin, ymax = sel.extents
        mask = (
            (x_data >= xmin) & (x_data <= xmax) & (y_data >= ymin) & (y_data <= ymax)
        )
        self._set_selection(np.where(mask)[0])

    def _on_lasso_live(self, sel):
        """Preview the enclosed points while the lasso is still being drawn."""
        idx = getattr(sel, "_plot_index", None)
        verts = sel.verts
        if idx is None or not verts or len(verts) < 3:
            self._composite()
            return
        self._apply_lasso_verts(idx, verts)

    def _apply_lasso_verts(self, idx: int, verts):
        plot_xy = self._plot_data[idx]
        if plot_xy is None or not self.cluster_ids or len(verts) < 3:
            self._composite()
            return
        x_data, y_data = plot_xy
        mask = Path(verts).contains_points(np.column_stack([x_data, y_data]))
        self._set_selection(np.where(mask)[0])

    def _freeze_lasso(self, sel, verts):
        """Keep the drawn path on screen after release.

        LassoSelector blanks its own line in ``_release``, so without this the
        outline disappeared exactly when you wanted to look at it.
        """
        idx = getattr(sel, "_plot_index", None)
        if idx is None:
            return
        xy = np.asarray(verts, dtype=float)
        poly = self._lasso_polys[idx]
        if poly is None:
            poly = MplPolygon(
                xy,
                closed=True,
                facecolor=to_rgba(PALETTE["accent"], 0.14),
                edgecolor=to_rgba(PALETTE["text_primary"], 0.85),
                linewidth=1.4,
                zorder=5,
                animated=True,
            )
            self.axes.flat[idx].add_patch(poly)
            self._lasso_polys[idx] = poly
        else:
            poly.set_xy(xy)
        poly.set_visible(True)
        self._composite()

    # ── Highlight / compositing ───────────────────────────────────────────────

    def _set_selection(self, indices):
        """Adopt *indices* as the selection and repaint everything that shows it."""
        indices = np.asarray(indices, dtype=int)
        if not np.array_equal(indices, self._selection):
            self._selection = indices
            self._sync_overlays(indices)
            self._sync_selection_panel()
            if getattr(self, "rf_map", None) is not None:
                self.rf_map.highlight(
                    [
                        self.cluster_ids[i]
                        for i in indices
                        if 0 <= i < len(self.cluster_ids)
                    ]
                )
        self._composite()

    def highlight_selection(self, indices: np.ndarray):
        """Public entry point kept for external callers."""
        self._set_selection(indices)

    def _sync_overlays(self, indices: np.ndarray):
        for idx, overlay in enumerate(self._overlay_artists):
            if overlay is None:
                continue
            plot_xy = self._plot_data[idx]
            if plot_xy is None or len(indices) == 0:
                overlay.set_offsets(np.empty((0, 2)))
                continue
            x_data, y_data = plot_xy
            overlay.set_offsets(np.column_stack([x_data[indices], y_data[indices]]))

    def _composite(self):
        """The single painter for this canvas.

        Restore the clean background, then stamp the animated layers back on in
        a fixed order: selected points, frozen lasso outlines, then whatever
        the live selector is drawing. Because every repaint goes through here,
        no layer can wipe another — which is what the selectors did to the
        highlight (and to each other) when they each blitted on their own.
        """
        if self._building or not hasattr(self, "canvas"):
            return
        if self._blit_bg is None:
            # No usable snapshot (first paint or post-resize). draw() re-enters
            # _on_canvas_draw, which snapshots and composites for us.
            self.canvas.draw()
            return

        self.canvas.restore_region(self._blit_bg)

        for idx, ax in enumerate(self.axes.flat):
            overlay = self._overlay_artists[idx]
            if overlay is not None:
                ax.draw_artist(overlay)
            poly = self._lasso_polys[idx]
            if poly is not None and poly.get_visible():
                ax.draw_artist(poly)

        for sel in self._rect_selectors + self._lasso_selectors:
            if not sel.active:
                continue
            for artist in sel.artists:
                if artist.get_visible() and artist.axes is not None:
                    artist.axes.draw_artist(artist)

        self.canvas.blit(self.fig.bbox)

    def _on_canvas_draw(self, event):
        """Cache the freshly rendered background, then repaint the live layers.

        Every animated artist is excluded from the draw that just finished, so
        the snapshot is clean by construction.
        """
        if self._building:
            return
        self._blit_bg = self.canvas.copy_from_bbox(self.fig.bbox)
        self._composite()

    # ── Selection panel ───────────────────────────────────────────────────────

    def _next_group_name(self) -> str:
        dm = self.main_window.data_manager
        if dm is None:
            return "group"
        return f"Nc{dm.new_class_id}"

    def _sync_selection_panel(self):
        n = len(self._selection)
        armed = "true" if n > 0 else "false"

        if n:
            plural = "s" if n != 1 else ""
            self.sel_count_label.setText(f"{n} cell{plural} selected")
            self.btn_create_group.setText(f"Create {self._next_group_name()}")
        else:
            self.sel_count_label.setText("Drag on a plot to select cells")
            self.btn_create_group.setText("Create group")

        self.btn_create_group.setEnabled(n > 0)
        self.btn_clear_selection.setEnabled(n > 0)

        for widget in (self._sel_panel, self.sel_count_label):
            if widget.property("armed") != armed:
                widget.setProperty("armed", armed)
                widget.style().unpolish(widget)
                widget.style().polish(widget)

    def clear_selection(self):
        """Drop the selection and every selection shape."""
        for sel in self._rect_selectors + self._lasso_selectors:
            self._retire(sel)
        for poly in self._lasso_polys:
            if poly is not None:
                poly.set_visible(False)
        self._set_selection(np.empty(0, dtype=int))

    def _create_group_from_selection(self):
        indices = self._selection
        if len(indices) == 0:
            return
        selected_ids = [
            self.cluster_ids[i] for i in indices if 0 <= i < len(self.cluster_ids)
        ]
        if not selected_ids:
            return
        name = self._create_new_class(selected_ids)
        # Keep the shape and the selection: refining and regrouping is the
        # normal workflow, so nothing is torn down here.
        self.sel_count_label.setText(f"{name} created — {len(selected_ids)} cells")
        self.btn_create_group.setText(f"Create {self._next_group_name()}")

    # ── Context menu ──────────────────────────────────────────────────────────

    def _on_canvas_button(self, event):
        """Right-click offers the same actions as the panel, at the cursor."""
        if event.button == 3 and len(self._selection):
            self.show_context_menu(self._selection)

    def show_context_menu(self, selected_indices: np.ndarray):
        if len(selected_indices) == 0:
            return
        selected_ids = [
            self.cluster_ids[i]
            for i in selected_indices
            if 0 <= i < len(self.cluster_ids)
        ]
        if not selected_ids:
            return
        menu = QMenu(self)
        n = len(selected_ids)
        plural = "s" if n != 1 else ""
        create_action = menu.addAction(f"Create group from {n} cluster{plural}")
        clear_action = menu.addAction("Clear selection")

        action = menu.exec(QCursor.pos())
        if action == create_action:
            self._create_group_from_selection()
        elif action == clear_action:
            self.clear_selection()

    # Keep old name as alias
    def create_new_class(self, selected_ids):
        self._create_new_class(selected_ids)

    def _create_new_class(self, selected_ids: list) -> str:
        if self.main_window.data_manager is None:
            return ""
        current_new_class_id = self.main_window.data_manager.new_class_id
        group_name = f"Nc{current_new_class_id}"
        self.main_window.data_manager.new_class_id += 1
        from ..callbacks import group_clusters_in_tree

        group_clusters_in_tree(self.main_window, selected_ids, group_name)
        logger.info(f"Created new group {group_name} with {len(selected_ids)} clusters")
        # ← self.close() removed: window stays open so you can keep selecting
        return group_name

    # ── RF mosaic click-through ───────────────────────────────────────────────

    def _on_rf_cell_clicked(self, cluster_id: int):
        """Clicking an RF in the mosaic selects that one cell everywhere."""
        try:
            idx = list(self.cluster_ids).index(int(cluster_id))
        except (ValueError, TypeError):
            return
        for sel in self._rect_selectors + self._lasso_selectors:
            self._retire(sel)
        for poly in self._lasso_polys:
            if poly is not None:
                poly.set_visible(False)
        self._set_selection(np.array([idx], dtype=int))
        focus = getattr(self.main_window, "focus_cluster", None)
        if callable(focus):
            focus(int(cluster_id))

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def resizeEvent(self, event):
        """Invalidate the blit cache; the following draw_event re-snapshots it."""
        super().resizeEvent(event)
        self._blit_bg = None

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        super().closeEvent(event)
