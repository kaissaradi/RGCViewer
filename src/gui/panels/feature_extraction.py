from __future__ import annotations
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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
    QComboBox,
    QCheckBox,
    QApplication,
)
from qtpy.QtGui import QCursor, QColor, QPalette
from qtpy.QtCore import QThread, Signal, Qt
from typing import TYPE_CHECKING, Optional, List

from .live_selectors import (
    LiveLassoSelector,
    LiveRectangleSelector,
    make_lasso_selector,
    make_rect_selector,
    retire,
)
from .rf_map_widget import RFMapWidget
from .trace_stack_widget import TraceStackWidget
from .view_toggles import PopulationViewToggles
from ...analysis import feature_catalog

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

/* The axis pickers and the n selector are the only combos in this dialog and
   would otherwise render in the platform's light default against a dark
   panel. */
QComboBox {{
    background-color: {PALETTE["btn_inactive"]};
    color: {PALETTE["text_primary"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 11px;
}}
QComboBox:hover {{
    border-color: {PALETTE["text_muted"]};
}}
QComboBox::drop-down {{
    border: none;
    width: 16px;
}}
QComboBox QAbstractItemView {{
    background-color: {PALETTE["surface"]};
    color: {PALETTE["text_primary"]};
    border: 1px solid {PALETTE["border"]};
    selection-background-color: {PALETTE["accent"]};
    selection-color: #ffffff;
}}
QCheckBox {{
    color: {PALETTE["text_muted"]};
    font-size: 11px;
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


# The selector classes this window is built on now live in live_selectors.py,
# so the UMAP panel can brush the same way. LiveRectangleSelector and
# LiveLassoSelector are re-exported above for anything still importing them
# from here.


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

    def __init__(self, data_manager, cluster_ids, include_contrast=False):
        super().__init__()
        self.data_manager = data_manager
        self.cluster_ids = cluster_ids
        self.include_contrast = include_contrast
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

            # The catalogue is built first and independently, because the
            # legacy matrix above requires BOTH an STA time course and an ACG
            # for every cell. A dataset that has neither — vision-only, or one
            # carrying only chirp and contrast analyses — used to fail here
            # outright, even though its chirp, contrast and quality features
            # are perfectly usable.
            self.progress.emit("Building feature catalogue...", 40)
            catalog, cat_ids = {}, []
            try:
                cat_ids, catalog = feature_catalog.build_catalog(
                    self.data_manager, valid_ids or self.cluster_ids,
                    include_contrast=self.include_contrast,
                    progress=lambda m, p: self.progress.emit(m, min(40 + p // 3, 78)),
                )
            except Exception:
                logger.warning("Feature catalogue failed; falling back to the "
                               "fixed panels", exc_info=True)

            if len(valid_ids) == 0:
                if not catalog or not cat_ids:
                    self.progress.emit("No valid features found.", 100)
                    self.finished.emit({})
                    return
                # Stand on the catalogue alone. The fixed-panel arrays below
                # become empty, which draw_plots already handles by preferring
                # the catalogue.
                valid_ids = list(cat_ids)
                feature_matrix = np.zeros((len(valid_ids), 0))
                metadata = {k: [0.0] * len(valid_ids)
                            for k in ("Time to Peak", "RF Area", "Ellipticity")}

            n = len(valid_ids)
            n_comp = min(3, n, feature_matrix.shape[1] // 2 if feature_matrix.size else 0)

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

            # build_catalog applies its own prefilter, so it can return a
            # different — usually shorter — row order than valid_ids. Everything
            # else in results (the trace stacks, the RF mosaic, the selection
            # indices) is keyed to valid_ids, so the catalogue is reindexed onto
            # that order rather than trusted to match. Cells it dropped become
            # NaN and simply do not plot; letting two row orders coexist would
            # silently attach one cell's features to another cell's identity.
            if catalog:
                results["catalog"] = feature_catalog.reindex_catalog(
                    catalog, cat_ids, valid_ids)

            # The scatters plot PCA scores; the population views plot the traces
            # those scores came from. Fetching them here keeps the whole cost on
            # the worker thread — get_raw_feature_blocks reads the same physics
            # cache ensure_physics_cache just warmed, so this is cheap.
            self.progress.emit("Collecting population traces...", 80)
            results.update(self._collect_traces(valid_ids))

            self.progress.emit("Done.", 100)
            self.finished.emit(results)

        except Exception as e:
            logger.error("Error in FeatureAnalysisWorker: %s", e, exc_info=True)
            self.finished.emit({})

    def _collect_traces(self, valid_ids):
        """Temporal STA and chirp PSTH rows aligned to *valid_ids*.

        The prefilter is deliberately wide open here: ``valid_ids`` has already
        been decided by ``get_physics_feature_matrix``, and letting
        ``get_raw_feature_blocks`` drop a second, differently-filtered set would
        silently misalign the traces against the scatter points.

        Returns an empty dict on any failure — the population views are an
        extra, and losing them must not cost the user the whole analysis.
        """
        try:
            blocks, block_ids, _discarded = self.data_manager.get_raw_feature_blocks(
                list(valid_ids),
                {"min_sta_std": 0.0, "max_rf_area": float("inf")},
            )
        except Exception:
            logger.warning("population traces unavailable", exc_info=True)
            return {}

        order = {int(c): i for i, c in enumerate(block_ids)}
        rows = [order.get(int(c)) for c in valid_ids]
        if any(r is None for r in rows):
            logger.warning(
                "raw feature blocks cover %d of %d cells — skipping population views",
                sum(r is not None for r in rows),
                len(rows),
            )
            return {}

        out = {}
        for key in ("temporal", "acg", "chirp"):
            mat = blocks.get(key)
            if mat is None:
                continue
            mat = np.asarray(mat, dtype=float)
            if mat.ndim != 2 or mat.shape[1] < 2:
                continue
            out[key] = mat[rows]
        return out

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

        # Named per-cell features the axis pickers offer, and the (x, y) pair
        # each of the six panels currently shows. Empty until the worker
        # returns; draw_plots falls back to the fixed panels until then.
        self.catalog: dict = {}
        self._panels = None
        self._include_contrast = False
        self._building_combos = False
        # Feature-selection mode: 'manual' names every axis, 'shuffle' draws
        # n features and plots every pair of them.
        self._feature_mode = "manual"
        self._shuffled_once = False
        self._rng = np.random.default_rng()

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
        # Six scatters plus the mosaic and trace column need room: at the old
        # 1200x820 the panel titles ran into each other. Take most of the
        # screen but never exceed it, so this still behaves on a laptop.
        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            self.resize(min(1700, int(avail.width() * 0.92)),
                        min(1050, int(avail.height() * 0.90)))
        else:
            self.resize(1400, 900)
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

        fs_label = QLabel("Feature selection:")
        fs_label.setStyleSheet(
            f"color: {PALETTE['text_muted']}; font-size: 12px; "
            f"padding-left: 18px; padding-right: 4px;")
        row.addWidget(fs_label)
        self._btn_manual = self._make_feature_mode_btn("Manual", "manual")
        self._btn_shuffle = self._make_feature_mode_btn("Shuffle", "shuffle")
        row.addWidget(self._btn_manual)
        row.addWidget(self._btn_shuffle)
        row.addStretch()
        # Kept so _build_axis_picker can drop the contrast toggle here: it
        # applies to both modes, so it cannot live on the manual-only bar.
        self._toolbar_row = row

        hint = QLabel(
            "Drag on any plot — or on the mosaic and traces — to select · "
            "drag the handles to refine · right-click to group or trash"
        )
        hint.setStyleSheet(
            f"color: {PALETTE['text_muted']}; font-size: 11px; font-style: italic;"
        )
        row.addWidget(hint)

        self.main_layout.addWidget(toolbar)
        self._update_tool_buttons()
        self._build_axis_picker()

    def _make_feature_mode_btn(self, text, mode):
        btn = QPushButton(text)
        btn.setObjectName("tool_btn")
        btn.setCursor(QCursor(Qt.PointingHandCursor))
        btn.setProperty("active", self._feature_mode == mode)
        btn.clicked.connect(lambda: self._set_feature_mode(mode))
        return btn

    def _set_feature_mode(self, mode):
        """Switch between naming every axis and shuffling through pairings."""
        self._feature_mode = mode
        for btn, name in ((self._btn_manual, "manual"), (self._btn_shuffle, "shuffle")):
            btn.setProperty("active", mode == name)
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        manual = mode == "manual"
        self.axis_bar.setVisible(manual)
        self.axis_bar2.setVisible(manual)
        self.shuffle_bar.setVisible(not manual)
        if not manual and self.catalog and not self._shuffled_once:
            self._do_shuffle()

    def _build_axis_picker(self):
        """Per-panel X/Y feature selectors.

        The six panels were fixed to temporal/ACG PCs and RF diameter, which
        are the axes that separate cells by receptive field but say nothing
        about how a cell responds. Everything the app computes is now
        selectable here — chirp shape and polarity, grating tuning, contrast
        gain per light level, firing rate, the quality metrics — because which
        pair separates two candidate types is not knowable in advance.
        """
        self.axis_bar = QWidget()
        row = QHBoxLayout(self.axis_bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        label = QLabel("Panels:")
        label.setStyleSheet(
            f"color: {PALETTE['text_muted']}; font-size: 12px; padding-right: 4px;")
        row.addWidget(label)

        self.axis_combos = []
        for i in range(6):
            cx, cy = QComboBox(), QComboBox()
            for combo in (cx, cy):
                combo.setMinimumWidth(120)
                combo.setStyleSheet("font-size: 11px;")
                combo.currentIndexChanged.connect(self._on_axis_changed)
            sep = QLabel(f"{i + 1}:")
            sep.setStyleSheet(f"color: {PALETTE['text_muted']}; font-size: 11px;")
            row.addWidget(sep)
            row.addWidget(cx)
            vs = QLabel("vs")
            vs.setStyleSheet(f"color: {PALETTE['text_muted']}; font-size: 10px;")
            row.addWidget(vs)
            row.addWidget(cy)
            self.axis_combos.append((cx, cy))
            if i == 2:
                row.addStretch()
                self.main_layout.addWidget(self.axis_bar)
                self.axis_bar2 = QWidget()
                row = QHBoxLayout(self.axis_bar2)
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(4)
                pad = QLabel("          ")
                row.addWidget(pad)

        self._build_shuffle_bar()

        self.contrast_chk = QCheckBox("Contrast features")  # placed on the toolbar
        self.contrast_chk.setToolTip(
            "Adds per-light-level contrast gain and c50 to the feature list.\n"
            "Off by default: it costs an FFT per trial per level, roughly\n"
            "30 s for a whole dataset on the first pass (cached afterwards).")
        self.contrast_chk.toggled.connect(self._on_contrast_toggled)
        self._toolbar_row.addWidget(self.contrast_chk)
        row.addStretch()
        self.main_layout.addWidget(self.axis_bar2)
        self.axis_bar.setEnabled(False)
        self.axis_bar2.setEnabled(False)

    def _build_shuffle_bar(self):
        """Pick some feature families, draw n of their features, plot every pair.

        Six panels hold C(n,2) = n(n-1)/2 distinct pairs, and C(n,2) = 6 solves
        to n = 4 — four features fill the grid exactly, every pairing shown once
        with none repeated and none left out. Smaller n underfills; larger n has
        more pairs than panels, so Shuffle samples six of them and re-rolls.
        """
        self.shuffle_bar = QWidget()
        row = QHBoxLayout(self.shuffle_bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        lbl = QLabel("Features:")
        lbl.setStyleSheet(f"color: {PALETTE['text_muted']}; font-size: 12px;")
        row.addWidget(lbl)

        # Families rather than individual features: picking "Chirp PSTH" should
        # bring all of its PCs into play, which is how you ask "does anything
        # about the chirp separate these cells?"
        self.family_btn = QPushButton("Choose families ▾")
        self.family_btn.setObjectName("tool_btn")
        self.family_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.family_menu = QMenu(self.family_btn)
        self.family_btn.setMenu(self.family_menu)
        row.addWidget(self.family_btn)

        n_lbl = QLabel("n =")
        n_lbl.setStyleSheet(f"color: {PALETTE['text_muted']}; font-size: 12px;")
        row.addWidget(n_lbl)
        self.n_combo = QComboBox()
        for n in range(3, 8):
            pairs = n * (n - 1) // 2
            if pairs == 6:
                note = "exactly fills the 6 panels"
            elif pairs < 6:
                note = f"{pairs} pairs — fills {pairs} of 6"
            else:
                note = f"{pairs} pairs — 6 sampled"
            self.n_combo.addItem(f"{n}  ({note})", n)
        self.n_combo.setCurrentIndex(1)  # n = 4
        self.n_combo.setMinimumWidth(210)
        self.n_combo.currentIndexChanged.connect(self._do_shuffle)
        row.addWidget(self.n_combo)

        self.shuffle_btn = QPushButton("🎲  Shuffle")
        self.shuffle_btn.setObjectName("primary_btn")
        self.shuffle_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.shuffle_btn.setToolTip(
            "Draw a fresh set of features from the chosen families and plot "
            "every pair. Press again for another draw.")
        self.shuffle_btn.clicked.connect(self._do_shuffle)
        row.addWidget(self.shuffle_btn)

        self.shuffle_status = QLabel("")
        self.shuffle_status.setStyleSheet(
            f"color: {PALETTE['text_muted']}; font-size: 11px; font-style: italic;")
        row.addWidget(self.shuffle_status)
        row.addStretch()

        self.main_layout.addWidget(self.shuffle_bar)
        self.shuffle_bar.setVisible(False)

    def _populate_family_menu(self):
        """One checkable entry per family present in the catalogue."""
        self.family_menu.clear()
        self._family_actions = {}
        families = []
        for entry in self.catalog.values():
            if entry.group not in families:
                families.append(entry.group)
        for fam in families:
            act = self.family_menu.addAction(fam)
            act.setCheckable(True)
            # Shape PCs are the point of the shuffle, so they start on; the
            # scalars are available but would otherwise dominate by count.
            act.setChecked(fam == "Shape (PCA)")
            act.toggled.connect(self._on_family_toggled)
            self._family_actions[fam] = act
        self._update_family_button()

    def _selected_families(self):
        return [f for f, a in getattr(self, "_family_actions", {}).items()
                if a.isChecked()]

    def _update_family_button(self):
        chosen = self._selected_families()
        if not chosen:
            self.family_btn.setText("Choose families ▾")
        elif len(chosen) == 1:
            self.family_btn.setText(f"{chosen[0]} ▾")
        else:
            self.family_btn.setText(f"{len(chosen)} families ▾")

    def _on_family_toggled(self, _checked):
        self._update_family_button()
        self._do_shuffle()

    def _shuffle_pool(self):
        """Feature names eligible for the draw, given the chosen families."""
        chosen = set(self._selected_families())
        pool = [n for n, e in self.catalog.items()
                if not chosen or e.group in chosen]
        # A feature that is NaN everywhere plots nothing; drawing one would
        # waste a panel.
        return [n for n in pool if np.isfinite(self.catalog[n].values).any()]

    def _do_shuffle(self, *_):
        """Draw n features and lay every pair of them across the panels."""
        if not self.catalog or self._feature_mode != "shuffle":
            return
        pool = self._shuffle_pool()
        n_panels = len(self.axes.flat)
        if len(pool) < 2:
            self.shuffle_status.setText("not enough features in these families")
            return

        n = self.n_combo.currentData() or 4
        n = min(int(n), len(pool))
        picked = list(self._rng.choice(pool, size=n, replace=False))
        pairs = [(a, b) for i, a in enumerate(picked) for b in picked[i + 1:]]
        self._rng.shuffle(pairs)

        if len(pairs) > n_panels:
            pairs = pairs[:n_panels]
        elif len(pairs) < n_panels:
            # Underfilled: repeat the pairs rather than leaving dead axes, so
            # the grid always says something.
            pairs = [pairs[i % len(pairs)] for i in range(n_panels)]

        self._panels = pairs
        self._shuffled_once = True
        self.shuffle_status.setText(
            f"{n} features → {n * (n - 1) // 2} pairs: " + ", ".join(picked[:4])
            + ("…" if n > 4 else ""))
        self.draw_plots()

    def _populate_axis_combos(self):
        """Fill the selectors from the catalogue, grouping by feature family."""
        if not self.catalog:
            return
        self._building_combos = True
        names = list(self.catalog.keys())
        panels = feature_catalog.resolve_panels(self.catalog, self._panels)
        self._panels = panels
        for (cx, cy), (xname, yname) in zip(self.axis_combos, panels):
            for combo, chosen in ((cx, xname), (cy, yname)):
                combo.blockSignals(True)
                combo.clear()
                last_group = None
                for name in names:
                    group = self.catalog[name].group
                    if group != last_group:
                        # A non-selectable header keeps ~30 features readable.
                        combo.addItem(f"— {group} —")
                        combo.model().item(combo.count() - 1).setEnabled(False)
                        last_group = group
                    combo.addItem(name)
                    combo.setItemData(combo.count() - 1,
                                      self.catalog[name].description, Qt.ToolTipRole)
                idx = combo.findText(chosen)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                combo.blockSignals(False)
                self._sync_combo_tooltip(combo)
        self.axis_bar.setEnabled(True)
        self.axis_bar2.setEnabled(True)
        self._building_combos = False

    def _sync_combo_tooltip(self, combo):
        """Explain the selected feature on hover, not just in the dropdown.

        The per-item tooltips only appear once the list is open, so a closed
        combo showing "Chirp PSTH PC2" said nothing about what that component
        actually weights.
        """
        entry = self.catalog.get(combo.currentText())
        combo.setToolTip(entry.description if entry else "")

    def _on_axis_changed(self, *_):
        if getattr(self, "_building_combos", False) or not self.catalog:
            return
        panels = []
        for cx, cy in self.axis_combos:
            x, y = cx.currentText(), cy.currentText()
            if x not in self.catalog or y not in self.catalog:
                return  # a group header, not a feature
            panels.append((x, y))
        for cx, cy in self.axis_combos:
            self._sync_combo_tooltip(cx)
            self._sync_combo_tooltip(cy)
        self._panels = panels
        self.draw_plots()

    def _on_contrast_toggled(self, checked):
        """Recompute the catalogue with or without the contrast features."""
        if not self.cluster_ids:
            return
        self._include_contrast = bool(checked)
        self.loading_widget.show()
        self.status_label.setText(
            "Computing contrast responses…" if checked else "Rebuilding features…")
        self._start_worker()

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

        self._sync_population_selection_mode()
        self._composite()

    def _sync_population_selection_mode(self):
        """Arm the RF mosaic and trace stacks with the same tool as the scatters.

        One tool choice for the whole window: the user picks Rectangle or Lasso
        once and can then drag in a feature plot or in a population view
        interchangeably.
        """
        for view in getattr(self, "_population_views", []):
            view.set_selection_mode(self._selection_mode)

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

        # Under it, the raw traces the PCA scores were computed from. The
        # mosaic says whether a selection is a plausible single type; these say
        # whether its members actually share a response — which the PC-score
        # scatters cannot show, since two different shapes can land on the same
        # score.
        self.trace_stacks = {
            "temporal": TraceStackWidget(
                colors=PALETTE, title="Temporal STA (frames)"
            ),
            "acg": TraceStackWidget(colors=PALETTE, title="Autocorrelogram (ms lag)"),
            "chirp": TraceStackWidget(colors=PALETTE, title="Chirp PSTH (s)"),
        }

        #: Every view you can brush in directly, in side-panel order.
        self._population_views = [self.rf_map] + list(self.trace_stacks.values())
        for view in self._population_views:
            view.selection_changed.connect(
                lambda ids, src=view: self._on_population_selection(src, ids)
            )
            view.context_menu_requested.connect(self._on_view_context_menu)

        side_stack = QSplitter(Qt.Vertical)
        side_stack.addWidget(self.rf_map)
        for key in ("temporal", "acg", "chirp"):
            side_stack.addWidget(self.trace_stacks[key])
        side_stack.setSizes([300, 140, 140, 140])

        # The ACG starts closed for the same reason as in the UMAP panel: it is
        # the block whose overlay least often settles a curation decision, and
        # four panels in this column leaves none of them tall enough to read.
        self.view_toggles = PopulationViewToggles(colors=PALETTE)
        self.view_toggles.add_view("rf", "RF Mosaic", self.rf_map, checked=True)
        for key, label, checked in (
            ("temporal", "Temporal STA", True),
            ("acg", "ACG", False),
            ("chirp", "Chirp", True),
        ):
            self.view_toggles.add_view(
                key, label, self.trace_stacks[key], checked=checked
            )
            self.view_toggles.set_available(key, False, "Still computing…")
        self.view_toggles.view_toggled.connect(self._on_view_toggled)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(8)
        side_layout.addWidget(self.view_toggles)
        side_layout.addWidget(side_stack, stretch=1)
        side_layout.addWidget(self._build_selection_panel())

        self.plot_splitter = QSplitter(Qt.Horizontal)
        self.plot_splitter.addWidget(self.canvas)
        self.plot_splitter.addWidget(side)
        self.plot_splitter.setStretchFactor(0, 3)
        self.plot_splitter.setStretchFactor(1, 1)
        self.plot_splitter.setSizes([880, 360])
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
        # Stop any in-flight run first: toggling the contrast features
        # restarts the worker, and two of them would race to emit results.
        old = getattr(self, "worker", None)
        if old is not None and old.isRunning():
            old.stop()
            old.wait(2000)
        self.worker = FeatureAnalysisWorker(
            self.main_window.data_manager, self.initial_cluster_ids,
            include_contrast=getattr(self, "_include_contrast", False),
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
        self.catalog = results.get("catalog") or {}
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
        self.view_toggles.set_available("rf", True)
        self._populate_trace_stacks(results)
        self._populate_axis_combos()
        self._populate_family_menu()
        self.draw_plots()
        # Arm the population views with whichever tool the toolbar shows.
        self._sync_population_selection_mode()

    def _populate_trace_stacks(self, results: dict):
        """Fill the trace overlays, greying out any block with no data.

        Chirp is precomputed offline and often simply absent, so its stack is
        marked unavailable rather than showing an empty axes — see the "if
        relevant" in the panel's brief.
        """
        ids = list(self.cluster_ids)
        for key, stack in self.trace_stacks.items():
            block = results.get(key)
            block = np.asarray(block) if block is not None else None
            if block is None or block.ndim != 2 or block.shape[1] < 2:
                stack.set_traces(None, [])
                self.view_toggles.set_available(
                    key, False, "This run has no data for that feature."
                )
                continue

            x = np.arange(block.shape[1], dtype=float)
            if key == "acg":
                # get_standard_plot_data builds the ACG symmetric about zero in
                # 1 ms bins, so the half-width follows from the column count.
                x = x - (block.shape[1] - 1) / 2.0
            elif key == "chirp":
                bin_ms = self._chirp_bin_ms()
                if bin_ms:
                    # Match the Chirp dashboard's 25 ms Gaussian: unsmoothed,
                    # hundreds of 25 s PSTHs overlay into solid hash.
                    stack.set_smoothing(25.0 / bin_ms)
                    x = x * bin_ms / 1000.0
            stack.set_traces(block, ids, x=x)
            self.view_toggles.set_available(key, True)

    def _on_view_toggled(self, key, is_open):
        """Bring a reopened view back up to date with the current selection.

        Closed views are skipped by _set_selection, so without this one would
        come back showing whatever was selected when it was closed.
        """
        if not is_open:
            return
        view = self.view_toggles.widget_for(key)
        if view is None:
            return
        view.highlight(self._selected_cluster_ids())
        view.set_selection_mode(self._selection_mode)

    def _selected_cluster_ids(self):
        return [
            self.cluster_ids[i]
            for i in self._selection
            if 0 <= i < len(self.cluster_ids)
        ]

    def _chirp_bin_ms(self):
        dm = self.main_window.data_manager
        data = getattr(dm, "chirp_data", None) if dm is not None else None
        if not isinstance(data, dict):
            return None
        try:
            return float(data["bin_size_ms"])
        except (KeyError, TypeError, ValueError):
            return None

    # ── Plotting ──────────────────────────────────────────────────────────────

    @staticmethod
    def _set_finite_limits(setter, values):
        """Scale an axis from the finite values only; skip if there are none."""
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return
        lo, hi = float(finite.min()), float(finite.max())
        pad = (hi - lo) * 0.05 or 0.1
        setter(lo - pad, hi + pad)

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

        if self.catalog and self._panels:
            data_pairs = [(self.catalog[x].values, self.catalog[y].values)
                          for x, y in self._panels]
            # No title: it would repeat the two axis labels verbatim, and with
            # names like "Temporal STA PC3" the titles collided with each
            # other across the grid. The axes already say what is plotted.
            plot_meta = [("", x, y) for x, y in self._panels]
        else:
            # No catalogue (a dataset with no usable feature blocks, or the
            # build failed) — the original fixed panels still work.
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
            plot_meta = list(self._PLOT_META)

        # Old selectors keep their canvas callbacks alive even after ax.clear(),
        # so they must be disconnected or they go on handling events for axes
        # that no longer hold their data.
        for sel in self._rect_selectors + self._lasso_selectors:
            if sel is None:
                continue
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
            title, xlabel, ylabel = plot_meta[idx]
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

                # A feature can be entirely NaN — an RF measure with no STAs, a
                # contrast level no cell responded at — and min()/max() then
                # hand matplotlib NaN limits, which raises. Scale from the
                # finite values only, and leave the axis alone when there are
                # none.
                self._set_finite_limits(ax.set_xlim, x_data)
                self._set_finite_limits(ax.set_ylim, y_data)

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
            if sel is None:
                continue
            sel.set_active(self._selection_mode == "lasso")
            self._retire(sel)
        for sel in self._rect_selectors:
            if sel is None:
                continue
            sel.set_active(self._selection_mode == "rect")
            self._retire(sel)

        self._building = False
        self._sync_selection_panel()
        # Full render once; _on_canvas_draw snapshots the clean background.
        self.canvas.draw()

    # ── Selection: Rectangle ──────────────────────────────────────────────────

    def _attach_rect_selector(self, ax, idx: int, x_data: np.ndarray, y_data: np.ndarray):
        """Attach an interactive rectangle whose extents drive the selection live."""
        rect = make_rect_selector(ax, self, PALETTE, index=idx)
        if rect is not None:
            self._rect_selectors.append(rect)

    # ── Selection: Lasso ──────────────────────────────────────────────────────

    def _attach_lasso_selector(self, ax, idx: int, x_data: np.ndarray, y_data: np.ndarray):
        """Attach a lasso that previews as you draw and leaves its outline behind."""
        lasso = make_lasso_selector(ax, self, PALETTE, index=idx)
        if lasso is None:
            return
        lasso.set_active(False)  # starts inactive; activated by toolbar
        self._lasso_selectors.append(lasso)

    # ── Public aliases (keep external callers working) ────────────────────────

    def _setup_selector(self, ax, index, x_data, y_data):
        self._attach_rect_selector(ax, index, x_data, y_data)
        self._attach_lasso_selector(ax, index, x_data, y_data)

    def setup_selector(self, ax, index, x_data, y_data):
        self._setup_selector(ax, index, x_data, y_data)

    # ── Live selection ────────────────────────────────────────────────────────

    _retire = staticmethod(retire)

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
            ids = [
                self.cluster_ids[i] for i in indices if 0 <= i < len(self.cluster_ids)
            ]
            # Every *open* population view repaints from here, so the mosaic and
            # the trace overlays can never end up describing different
            # selections. Closed ones are caught up in _on_view_toggled rather
            # than painted invisibly on every drag.
            for view in self._open_views():
                view.highlight(ids)
        self._composite()

    def _open_views(self):
        """The population views currently on screen, in side-panel order."""
        toggles = getattr(self, "view_toggles", None)
        if toggles is None:
            return list(getattr(self, "_population_views", []))
        out = []
        for key in ("rf", "temporal", "acg", "chirp"):
            if toggles.is_open(key):
                view = toggles.widget_for(key)
                if view is not None:
                    out.append(view)
        return out

    def _on_population_selection(self, source, cluster_ids):
        """A brush landed in the RF mosaic or a trace stack.

        Selecting cells this way is the point of the population views: "these
        five RFs sit on top of each other" and "these traces have the early
        transient" are questions you can only ask in the view that shows them,
        and the answer should become the selection everywhere else.
        """
        wanted = {int(c) for c in (cluster_ids or [])}
        indices = [i for i, c in enumerate(self.cluster_ids) if int(c) in wanted]

        # The gesture that produced this selection is the only shape that should
        # remain on screen.
        for sel in self._rect_selectors + self._lasso_selectors:
            self._retire(sel)
        for poly in self._lasso_polys:
            if poly is not None:
                poly.set_visible(False)
        for view in self._population_views:
            if view is not source:
                view.clear_selection_shapes()

        self._set_selection(np.asarray(indices, dtype=int))

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
        for view in getattr(self, "_population_views", []):
            view.clear_selection_shapes()
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

    def _on_view_context_menu(self):
        """Right-click in the mosaic or a trace stack, same menu."""
        if len(self._selection):
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
        trash_action = menu.addAction(f"Move {n} cell{plural} to Trash")
        menu.addSeparator()
        clear_action = menu.addAction("Clear selection")

        action = menu.exec(QCursor.pos())
        if action == create_action:
            self._create_group_from_selection()
        elif action == trash_action:
            self._trash_selection()
        elif action == clear_action:
            self.clear_selection()

    def _trash_selection(self):
        """Send the selected cells to the tree's Trash folder.

        The scatters and the population views deliberately keep showing them:
        this window's analysis was computed for a fixed cell set, and quietly
        dropping rows out of it would leave the PCA scores describing a
        population that is no longer on screen. They are gone from the table and
        from the next UMAP run, which is where it matters.
        """
        ids = self._selected_cluster_ids()
        if not ids:
            return
        from ..callbacks import move_cluster_ids_to_trash

        n = move_cluster_ids_to_trash(self.main_window, ids)
        self.clear_selection()
        plural = "s" if n != 1 else ""
        self.sel_count_label.setText(
            f"{n} cell{plural} moved to Trash"
            if n
            else "Nothing moved — those cells are not in the tree"
        )

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
        for view in self._population_views:
            view.clear_selection_shapes()
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
