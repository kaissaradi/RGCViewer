import numpy as np
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QLabel,
    QProgressBar,
    QMessageBox,
    QSpinBox,
    QDialog,
    QTextEdit,
    QCheckBox,
    QSizePolicy,
    QGroupBox,
    QDoubleSpinBox,
    QGridLayout,
    QMenu,
    QSlider,
    QSplitter,
)
from qtpy.QtCore import QThread, QTimer, Qt
from qtpy.QtGui import QCursor
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath
from mpl_toolkits.mplot3d import Axes3D, art3d, proj3d  # noqa: F401
import logging

from ..theme import resolve_theme_colors
from ..workers.workers import UMAPWorker, ClusterWorker
from .live_selectors import make_lasso_selector, make_rect_selector, retire
from .rf_map_widget import RFMapWidget
from .trace_stack_widget import TraceStackWidget
from .view_toggles import PopulationViewToggles
from ...analysis.constants import (
    DEFAULT_MIN_STA_STD,
    DEFAULT_MAX_RF_AREA,
    DEFAULT_WEIGHT_TEMPORAL,
    DEFAULT_WEIGHT_ACG,
    DEFAULT_WEIGHT_RF_DIAMETER,
    DEFAULT_WEIGHT_GRATING_DSOS,
    DEFAULT_WEIGHT_CHIRP,
)

logger = logging.getLogger(__name__)


class UMAPPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.embedding = None
        self.cluster_ids = None
        self.metadata_df = None
        self.raw_blocks = None
        self._active_stacks = set()
        self._pending_highlight = None
        self.cbar = None
        self.is_3d = False
        self.selector = None

        # Single-compositor state, mirroring the Feature Extraction window.
        # Every artist that paints over the embedding goes through _composite();
        # _defer_composite suppresses it while a selector is mid-update so one
        # mouse move produces exactly one blit. Without this the rubber band,
        # the selection overlay and the scatter wiped each other in turn.
        self._blit_bg = None
        self._defer_composite = False
        self._building = False
        self._selection = np.empty(0, dtype=int)
        self._selection_mode = "lasso"
        self._rect_selector = None
        self._lasso_selector = None
        self._lasso_poly = None
        self._overlay_artist = None
        # Feature config of the run currently on screen — not the live widget
        # state, which the user may have changed since. The side panels must
        # describe the embedding that exists, not the one the controls would
        # produce if run again.
        self._run_feature_config = {}

        self.layout = QVBoxLayout(self)

        colors = resolve_theme_colors(self.main_window.get_current_colors())
        self._umap_colors = {
            "accent": colors["accent"],
            "accent_positive": colors["accent_positive"],
            "bg_panel": colors["bg_panel"],
        }

        self._build_control_panel(colors)

        # Plot Initialization
        self.fig = Figure(facecolor=self._umap_colors["bg_panel"])
        self.canvas = FigureCanvas(self.fig)

        # RF mosaic beside the embedding: lassoing a blob is only half the
        # question — whether those cells tile the array is the other half.
        self.rf_map = RFMapWidget(colors=self._rf_map_colors(colors), title="RF Mosaic")
        self.rf_map.cell_clicked.connect(self._on_rf_cell_clicked)

        # Under it, one overlay per feature block the embedding was built from.
        # The mosaic says whether a selection is a plausible single type; these
        # say whether its members share the response that put them together.
        # Keyed by the raw_blocks key so populating them is a plain loop.
        self.trace_stacks = {
            "temporal": TraceStackWidget(
                colors=self._rf_map_colors(colors), title="Temporal STA (frames)"
            ),
            "acg": TraceStackWidget(
                colors=self._rf_map_colors(colors), title="Autocorrelogram (ms lag)"
            ),
            "chirp": TraceStackWidget(
                colors=self._rf_map_colors(colors), title="Chirp PSTH (s)"
            ),
        }
        # Which feature keys drive which stack, for gating against the config
        # the run actually used.
        self._stack_feature_keys = {
            "temporal": "use_temporal",
            "acg": "use_acg",
            "chirp": "use_chirp",
        }

        #: Every view a selection can be drawn in or read off, mosaic first.
        self._population_views = [self.rf_map] + [
            self.trace_stacks[k] for k in ("temporal", "acg", "chirp")
        ]
        for view in self._population_views:
            view.selection_changed.connect(
                lambda ids, src=view: self._on_population_selection(src, ids)
            )
            view.context_menu_requested.connect(self.show_context_menu)

        self.side_splitter = QSplitter(Qt.Vertical)
        self.side_splitter.addWidget(self.rf_map)
        for key in ("temporal", "acg", "chirp"):
            self.side_splitter.addWidget(self.trace_stacks[key])
        self.side_splitter.setSizes([300, 130, 130, 130])

        # The ACG starts closed: it is the one block whose overlay rarely
        # settles a curation decision, and four stacked panels in this column
        # leaves none of them tall enough to read.
        self.view_toggles = PopulationViewToggles(colors=self._rf_map_colors(colors))
        self.view_toggles.add_view("rf", "RF Mosaic", self.rf_map, checked=True)
        for key, label, checked in (
            ("temporal", "Temporal STA", True),
            ("acg", "ACG", False),
            ("chirp", "Chirp", True),
        ):
            self.view_toggles.add_view(
                key, label, self.trace_stacks[key], checked=checked
            )
        self.view_toggles.view_toggled.connect(self._on_view_toggled)
        # Nothing is populated until a run, and an empty axes reading "No data"
        # is worse than no panel at all.
        for key in ("temporal", "acg", "chirp"):
            self.view_toggles.set_available(key, False, "Run UMAP to fill this view.")

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(6)
        side_layout.addWidget(self.view_toggles)
        side_layout.addWidget(self.side_splitter, stretch=1)
        side_layout.addWidget(self._build_selection_panel(colors))

        self.plot_splitter = QSplitter(Qt.Horizontal)
        self.plot_splitter.addWidget(self.canvas)
        self.plot_splitter.addWidget(side)
        self.plot_splitter.setStretchFactor(0, 3)
        self.plot_splitter.setStretchFactor(1, 1)
        self.plot_splitter.setSizes([900, 340])
        self.layout.addWidget(self.plot_splitter)

        # Initialize default 2D axes
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self._umap_colors["bg_panel"])

        # NEW: Initialize empty selector state
        self.current_selector = None

        # Right-click picks a point (and its cluster) for the RF mosaic.
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        # A full redraw invalidates the cached background; snapshotting from
        # draw_event means the selection survives every repaint Qt asks for.
        self.canvas.mpl_connect("draw_event", self._on_canvas_draw)

        # Worker refs
        self.worker_thread = None
        self.worker = None
        self.cluster_worker_thread = None
        self.cluster_worker = None

    def _build_selection_panel(self, colors):
        """The "make a group out of this" prompt, as a panel instead of a popup.

        It used to be a modal QMessageBox fired the instant a lasso closed,
        followed by a modal name prompt: you could not nudge the shape, look at
        the mosaic or change your mind without dismissing it first, and every
        stray drag interrupted you. As a panel it stays out of the way and
        updates live while you drag — the same flow as Feature Extraction.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        self.sel_count_label = QLabel("Drag on the embedding to select cells")
        self.sel_count_label.setWordWrap(True)
        self.sel_count_label.setStyleSheet(
            f"color: {colors['text_secondary']}; font-size: 11px;"
        )
        layout.addWidget(self.sel_count_label)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        self.btn_create_group = QPushButton("Create group")
        self.btn_create_group.setEnabled(False)
        self.btn_create_group.clicked.connect(self._create_group_from_selection)
        self.btn_clear_selection = QPushButton("Clear")
        self.btn_clear_selection.setEnabled(False)
        self.btn_clear_selection.clicked.connect(self.clear_selection)
        buttons.addWidget(self.btn_create_group, stretch=2)
        buttons.addWidget(self.btn_clear_selection, stretch=1)
        layout.addLayout(buttons)

        return panel

    @staticmethod
    def _rf_map_colors(colors):
        """Translate the app theme into the RF map's colour keys."""
        return {
            "bg": colors["bg_panel"],
            "surface": colors["bg_panel"],
            "border": colors.get("border_subtle", "#2a2d3e"),
            "text_primary": colors.get("text_primary", "#e8eaf0"),
            "text_muted": colors.get("text_secondary", "#6b7280"),
            # Blue is the resting state of the mosaic; orange means selected.
            "accent": colors.get("plot_scatter", "#4f8ef7"),
            "highlight": "#f97316",
        }

    def _build_control_panel(self, colors):
        # --- Controls Row 1 ---
        ctrl_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run UMAP (2D)")
        self.run_btn.clicked.connect(self.run_umap)
        self.run_btn.setStyleSheet(
            f"background-color: {self._umap_colors['accent']}; font-weight: bold;"
        )

        self.run_3d_btn = QPushButton("Run UMAP (3D)")
        self.run_3d_btn.clicked.connect(self.run_umap_3d)
        self.run_3d_btn.setStyleSheet(
            f"background-color: {self._umap_colors['accent_positive']}; font-weight: bold;"
        )

        self.color_combo = QComboBox()
        self.color_combo.addItems(
            [
                "KSLabel",
                "Polarity",
                "Ward",
                "K-Means",
                "Firing Rate",
                "ISI Violations",
                "Time to Peak",
                "RF Area",
                "Color Opponency",
            ]
        )
        self.color_combo.currentTextChanged.connect(lambda: self.update_plot())

        # NEW: Selection Tool Toggle
        self.selector_combo = QComboBox()
        self.selector_combo.addItems(["Lasso Tool", "Rectangle Tool"])
        self.selector_combo.currentIndexChanged.connect(self.update_selector)

        self.progress = QProgressBar()
        self.progress.hide()

        ctrl_layout.addWidget(self.run_btn)
        ctrl_layout.addWidget(self.run_3d_btn)
        ctrl_layout.addWidget(QLabel("Color:"))
        ctrl_layout.addWidget(self.color_combo)
        ctrl_layout.addWidget(QLabel("Tool:"))
        ctrl_layout.addWidget(self.selector_combo)
        ctrl_layout.addWidget(self.progress)
        ctrl_layout.addStretch()

        # --- Controls Row 2 (Clustering & Options) ---
        cluster_layout = QHBoxLayout()

        # Clustering controls
        self.cluster_method_combo = QComboBox()
        self.cluster_method_combo.addItems(["Ward", "K-Means"])

        self.cluster_param_spin = QSpinBox()
        self.cluster_param_spin.setRange(2, 100)
        self.cluster_param_spin.setValue(9)
        self.cluster_param_spin.setPrefix("# Types: ")

        self.cluster_method_combo.currentIndexChanged.connect(
            self._on_cluster_method_changed
        )

        self.cluster_btn = QPushButton("Run Clustering")
        self.cluster_btn.clicked.connect(self.run_clustering)

        # Auto-Group Checkbox
        self.auto_group_chk = QCheckBox("Auto-Group Tree")
        self.auto_group_chk.setToolTip(
            "If checked, finishing clustering will automatically create/overwrite "
            "groups in the main Tree View (e.g., 'Type_1', 'Type_2')."
        )
        self.auto_group_chk.setChecked(False)

        self.show_ids_btn = QPushButton("Show IDs")
        self.show_ids_btn.clicked.connect(self.show_group_ids)
        self.show_ids_btn.setEnabled(False)

        self.project_3d_chk = QCheckBox("Lasso 3D Proj")
        self.project_3d_chk.setToolTip(
            "Allows Lasso selection in 3D mode via screen projection."
        )
        self.project_3d_chk.setChecked(True)

        cluster_layout.addWidget(QLabel("Clustering:"))
        cluster_layout.addWidget(self.cluster_method_combo)
        cluster_layout.addWidget(self.cluster_param_spin)
        cluster_layout.addWidget(self.cluster_btn)
        cluster_layout.addWidget(self.auto_group_chk)
        cluster_layout.addWidget(self.show_ids_btn)
        cluster_layout.addWidget(self.project_3d_chk)
        cluster_layout.addStretch()

        pick_hint = QLabel("Right-click a point to select its cluster, again to act →")
        pick_hint.setStyleSheet(
            f"color: {colors['text_secondary']}; font-size: 11px; font-style: italic;"
        )
        pick_hint.setToolTip(
            "Right-clicking a point selects every cell in its cluster and shows "
            "them on the RF mosaic. Before clustering has been run, it selects "
            "just that cell. Right-clicking a point that is already selected "
            "opens a menu to group it or move it to Trash — the same menu the "
            "mosaic and trace overlays offer. Left-drag still lassoes."
        )
        cluster_layout.addWidget(pick_hint)

        # --- Controls Row 3 (Feature Selection & Weights Grid) ---
        weights_group = QGroupBox("Feature Selection & Weights")
        weights_grid = QGridLayout(weights_group)
        weights_grid.setContentsMargins(6, 6, 6, 6)
        weights_grid.setSpacing(6)

        self.feature_widgets = {}
        # Config map for features. Must stay in sync with analysis_core.py's
        # scalar_features table (temporal/acg/grating are handled as PCA
        # blocks, not scalar columns, but share the same use/weight-key
        # convention). "RF Diameter" covers rf_long_diameter +
        # rf_short_diameter as two scalar columns under one checkbox+slider.
        # "Grating DS/OS" drives PCA on the pooled direction-tuning curve
        # SHAPE (grating_calc.pooled_direction_tuning_curve), not a raw
        # DSI/OSI scalar summary — see build_feature_matrix's grating block
        # for why (DSI/OSI can't distinguish differently-shaped curves that
        # share a value). DSI/OSI/peak_rate/angle still exist as metadata-
        # only columns (UMAP scatter coloring/hover), just not read by the
        # embedding anymore.
        features_info = [
            ("Temporal STA", "use_temporal", "w_temporal", DEFAULT_WEIGHT_TEMPORAL),
            ("ACG (Autocorrelogram)", "use_acg", "w_acg", DEFAULT_WEIGHT_ACG),
            (
                "RF Diameter (long + short)",
                "use_rf_diameter",
                "w_rf_diameter",
                DEFAULT_WEIGHT_RF_DIAMETER,
            ),
            (
                "Grating DS/OS (tuning shape)",
                "use_grating_dsos",
                "w_grating_dsos",
                DEFAULT_WEIGHT_GRATING_DSOS,
            ),
            (
                "Chirp PSTH (response shape)",
                "use_chirp",
                "w_chirp",
                DEFAULT_WEIGHT_CHIRP,
            ),
        ]

        for idx, (label, use_key, w_key, default_w) in enumerate(features_info):
            chk = QCheckBox(label)
            chk.setChecked(True)

            # Slider (0-100 internal range) + read-only numeric readout,
            # replacing the old bare QDoubleSpinBox. Qt's QSlider is
            # integer-only, so the slider's internal range is 10x the
            # actual weight range (0-10.0); divide by 10 wherever the real
            # float value is read (see get_feature_config, the status-bar
            # summary below, and _on_slider_changed here). The readout
            # label makes the exact value always visible (needed for
            # reproducibility — knowing a UMAP run used weight 3.0 vs 3.2
            # later matters), which a bare slider alone would lose.
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(int(round(default_w * 10)))
            slider.setFixedWidth(110)
            slider.setToolTip(f"{label} weight (0-10)")

            value_label = QLabel(f"{default_w:.1f} / 10")
            value_label.setFixedWidth(48)

            def _make_slider_handler(lbl):
                def _on_slider_changed(v):
                    lbl.setText(f"{v / 10.0:.1f} / 10")

                return _on_slider_changed

            slider.valueChanged.connect(_make_slider_handler(value_label))

            # Connect checkbox to enable/disable the slider + readout
            chk.toggled.connect(slider.setEnabled)
            chk.toggled.connect(value_label.setEnabled)

            # Store widget references — get_feature_config() reads
            # slider.value()/10.0 rather than spin.value() directly now.
            # value_label is kept too so refresh_feature_availability() can
            # drive the whole row without re-deriving it from the layout.
            self.feature_widgets[use_key] = (chk, slider, w_key, value_label)

            # Add to grid layout: checkbox, slider, readout per feature
            row = idx // 3
            col = (idx % 3) * 3
            weights_grid.addWidget(chk, row, col)
            weights_grid.addWidget(slider, row, col + 1)
            weights_grid.addWidget(value_label, row, col + 2)

        # Set the initial gate state. data_manager is still None at construction
        # (MainWindow.__init__ builds the panels before any dataset is picked),
        # so this lands on "chirp disabled" — correct for the pre-load state.
        # callbacks._on_kilosort_loaded() calls this again once a dataset is in.
        self.refresh_feature_availability()

        # --- Controls Row 4 (Pre-Filter Thresholds) ---
        filter_group = QGroupBox("Quality Pre-Filtering")
        filter_layout = QHBoxLayout(filter_group)
        filter_layout.setContentsMargins(6, 6, 6, 6)
        filter_layout.setSpacing(10)

        # Min STA Std
        filter_layout.addWidget(QLabel("Min STA Std:"))
        self.min_sta_std_spin = QDoubleSpinBox()
        self.min_sta_std_spin.setRange(0.0, 1.0)
        self.min_sta_std_spin.setSingleStep(0.00001)
        self.min_sta_std_spin.setValue(DEFAULT_MIN_STA_STD)
        self.min_sta_std_spin.setDecimals(6)
        self.min_sta_std_spin.setFixedWidth(90)
        filter_layout.addWidget(self.min_sta_std_spin)

        # Max RF Area
        filter_layout.addWidget(QLabel("Max RF Area:"))
        self.max_rf_area_spin = QDoubleSpinBox()
        self.max_rf_area_spin.setRange(0.0, 9999.0)
        self.max_rf_area_spin.setSingleStep(10.0)
        self.max_rf_area_spin.setValue(DEFAULT_MAX_RF_AREA)
        self.max_rf_area_spin.setDecimals(1)
        self.max_rf_area_spin.setFixedWidth(80)
        filter_layout.addWidget(self.max_rf_area_spin)
        filter_layout.addStretch()

        # Build final controls layout
        self.controls_widget = QWidget(self)
        controls_layout = QVBoxLayout(self.controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)
        controls_layout.addLayout(ctrl_layout)
        controls_layout.addLayout(cluster_layout)
        controls_layout.addWidget(weights_group)
        controls_layout.addWidget(filter_group)

        self.controls_widget.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        self.layout.addWidget(self.controls_widget)

    def refresh_feature_availability(self):
        """Re-gate data-dependent feature rows against the CURRENT DataManager.

        Chirp has no in-GUI fallback (it's precomputed offline, unlike grating),
        so the row is only live when a chirp file was actually loaded.

        Called once at construction — when main_window.data_manager is still
        None, so chirp starts disabled — and again from
        callbacks._on_kilosort_loaded() once loading is done and
        chirp_available is final (load_chirp_data runs inside
        KilosortLoadWorker.run, so the flag is settled by the time that slot
        fires). Without the second call the checkbox stays dead on every
        dataset, chirp or not.

        Bidirectional on purpose: panels are built once and persist, while
        data_manager is replaced on every load, so a chirp-less dataset
        following a chirp-bearing one has to re-disable the row.
        """
        dm = getattr(self.main_window, "data_manager", None)
        # Prefer effective_* so a reference-bridge-only chirp still enables
        # the row (docs/specs/cross_run_stimulus_bridge.md D5).
        if dm is not None and hasattr(dm, "effective_chirp_available"):
            chirp_ok = bool(dm.effective_chirp_available())
        else:
            chirp_ok = bool(getattr(dm, "chirp_available", False))
        self._set_feature_enabled(
            "use_chirp",
            chirp_ok,
            "No chirp data loaded for this dataset (or mapped reference)",
        )

    def _set_feature_enabled(self, use_key, enabled, disabled_tooltip=""):
        """Enable/disable one feature row: checkbox, weight slider, readout."""
        entry = self.feature_widgets.get(use_key)
        if entry is None:
            return
        chk, slider, _w_key, value_label = entry
        chk.setEnabled(enabled)
        chk.setChecked(enabled)
        slider.setEnabled(enabled)
        value_label.setEnabled(enabled)
        chk.setToolTip("" if enabled else disabled_tooltip)

    def get_feature_config(self):
        config = {}
        for use_key, (chk, slider, w_key, _lbl) in self.feature_widgets.items():
            config[use_key] = chk.isChecked()
            # slider is an integer QSlider on a 0-100 range representing a
            # 0.0-10.0 float weight — see the slider construction above.
            config[w_key] = slider.value() / 10.0
        return config

    def get_filter_config(self):
        return {
            "min_sta_std": float(self.min_sta_std_spin.value()),
            "max_rf_area": float(self.max_rf_area_spin.value()),
        }

    def showEvent(self, event):
        """
        Qt defers geometry computation for widgets inside QTabWidget until they
        are first shown. On the initial visit the first paint can fire before
        the layout pass has committed sizes, causing the two toolbar rows to
        overlap. singleShot(0) defers the layout activation until after the
        event loop processes the show, guaranteeing geometry is committed
        before first paint.
        """
        super().showEvent(event)
        QTimer.singleShot(0, self._refresh_layout)
        QTimer.singleShot(50, self._refresh_layout)
        # Catch up on any tree selection that landed while this tab was hidden
        # (see highlight_cells). Deferred past the layout pass so the blit
        # background is snapshotted at the final geometry.
        if self._pending_highlight is not None and self.embedding is not None:
            QTimer.singleShot(60, lambda: self._highlight(self._pending_highlight))

    def _refresh_layout(self):
        self.controls_widget.adjustSize()
        hint_h = self.controls_widget.sizeHint().height()
        if hint_h > 0:
            self.controls_widget.setMinimumHeight(hint_h)
        self.layout.activate()
        self.updateGeometry()

    def _reset_workers(self):
        """Clean up any running workers."""
        # --- UMAP Worker Cleanup ---
        if self.worker:
            # Disconnect all signals first to avoid deadlocks
            try:
                self.worker.finished.disconnect()
                self.worker.error.disconnect()
                self.worker.progress.disconnect()
            except (TypeError, RuntimeError):
                # Already disconnected or never connected
                pass
            self.worker = None

        if self.worker_thread:
            self.worker_thread.quit()
            # Use timeout to prevent indefinite hangs (positional argument, not keyword)
            if not self.worker_thread.wait(1000):  # 1 second timeout in ms
                logger.warning(
                    "UMAP worker thread did not stop gracefully, forcing termination"
                )
                self.worker_thread.terminate()
                self.worker_thread.wait(500)
            self.worker_thread = None

        # --- Cluster Worker Cleanup ---
        if self.cluster_worker:
            # Disconnect all signals first
            try:
                self.cluster_worker.finished.disconnect()
                self.cluster_worker.error.disconnect()
            except (TypeError, RuntimeError):
                pass
            self.cluster_worker = None

        if self.cluster_worker_thread:
            self.cluster_worker_thread.quit()
            # Use timeout to prevent indefinite hangs (positional argument, not keyword)
            if not self.cluster_worker_thread.wait(1000):  # 1 second timeout in ms
                logger.warning(
                    "Cluster worker thread did not stop gracefully, forcing termination"
                )
                self.cluster_worker_thread.terminate()
                self.cluster_worker_thread.wait(500)
            self.cluster_worker_thread = None

    def reset_for_new_dataset(self):
        """Throw away everything that describes the previous preparation.

        Runs are spike-sorted independently, so a cluster ID from the old
        dataset names a different cell in the new one — and thanks to the
        Vision-ID offset it usually still resolves to *something*. Left alone,
        the panel would go on showing the previous embedding, its mosaic and
        its cluster colours, all of which look entirely plausible against the
        newly loaded prep. Everything derived from a run is dropped here, and
        the panel returns to its pre-run state until UMAP is run again.
        """
        self._reset_workers()

        self.embedding = None
        self.feature_matrix = None
        self.cluster_ids = None
        self.metadata_df = None
        self.raw_blocks = None
        self.discarded_ids = None
        self._run_feature_config = {}
        self._pending_highlight = None
        self._selection = np.empty(0, dtype=int)
        self._blit_bg = None
        self._overlay_artist = None
        self._lasso_poly = None

        self._clear_shapes()
        # Disconnect rather than merely hide: the axes below are about to be
        # thrown away, and a selector left wired to the canvas goes on handling
        # drags for an embedding that no longer exists.
        for attr in ("current_selector", "selector", "_rect_selector", "_lasso_selector"):
            sel = getattr(self, attr, None)
            if sel is not None:
                try:
                    sel.set_active(False)
                    sel.disconnect_events()
                except Exception:
                    logger.debug("selector teardown failed", exc_info=True)
                setattr(self, attr, None)
        for view in self._population_views:
            view.set_selection_mode("off")

        self.rf_map.set_cells(None, [])
        for view in self._population_views:
            view.set_group_colors(None)
        for key, stack in self.trace_stacks.items():
            stack.set_traces(None, [])
        # Greyed out rather than unchecked, so the user's own open/closed
        # choices survive the dataset swap and come back on the next run.
        for key in ("rf", "temporal", "acg", "chirp"):
            self.view_toggles.set_available(key, False, "Run UMAP to fill this view.")
        self._active_stacks = set()

        # Cluster labels lived in metadata_df, which is gone; leaving the combo
        # on "Ward" would show an empty plot titled as a clustering.
        self.color_combo.setCurrentText("KSLabel")
        self.show_ids_btn.setEnabled(False)

        self._building = True
        try:
            self.fig.clear()
            self.ax = self.fig.add_subplot(111)
            self.ax.set_facecolor(self._umap_colors["bg_panel"])
            self.is_3d = False
        finally:
            self._building = False
        self.canvas.draw_idle()

        self._sync_selection_panel()
        self.refresh_feature_availability()

    def cleanup(self):
        """Explicitly cleanup resources to prevent memory leaks."""
        self._reset_workers()

        # Explicitly clear and delete selector
        for attr in ("current_selector", "selector", "_rect_selector", "_lasso_selector"):
            sel = getattr(self, attr, None)
            if sel is not None:
                try:
                    sel.set_active(False)
                    sel.disconnect_events()
                except Exception:
                    logger.debug("selector teardown failed", exc_info=True)
                setattr(self, attr, None)

        # Explicitly clear the Matplotlib figure
        if hasattr(self, "fig"):
            self.fig.clf()

        for stack in getattr(self, "trace_stacks", {}).values():
            stack.fig.clf()

        # Delete the canvas
        if hasattr(self, "canvas"):
            self.canvas.setParent(None)
            self.canvas.deleteLater()
            # self.canvas = None # Do not set to None here if we need to check it later,
            # but usually it's better to let GC handle it after deleteLater

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)

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
                # Nothing selected means "use everything" — but everything the
                # user explicitly threw away is not part of everything. When
                # Trash is empty this returns None exactly as before.
                from .. import callbacks

                trashed = callbacks.get_trashed_cluster_ids(self.main_window)
                if not trashed:
                    logger.info("No cells selected, using all clusters")
                    return None

                dm = self.main_window.data_manager
                all_ids = [int(c) for c in dm.cluster_df["cluster_id"].values]
                kept = [c for c in all_ids if c not in trashed]
                logger.info(
                    "No cells selected, using all clusters except %d in Trash",
                    len(all_ids) - len(kept),
                )
                return kept

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
                logger.info(
                    "No valid cluster IDs found in selection, using all clusters"
                )
                return None

            result_list = list(selected_ids)
            logger.info(f"Found {len(result_list)} selected cluster IDs")
            return result_list

        except Exception as e:
            logger.error(f"Error getting selected cluster IDs: {e}")
            return None

    def run_umap(self):
        dm = self.main_window.data_manager
        total_clusters = len(dm.cluster_df)
        feature_cache = getattr(dm, "feature_cache", {})
        valid_cached = sum(1 for v in feature_cache.values() if v.get("_computed"))
        if valid_cached < total_clusters:
            QMessageBox.warning(
                self,
                "Cache Warming Up",
                f"Please wait for background caching to finish.\n\n"
                f"Ready: {valid_cached} / {total_clusters} cells.\n"
                f"Check the progress bar in the bottom right.",
            )
            return
        self._reset_workers()

        # Verify at least one feature is enabled
        feature_config = self.get_feature_config()
        if not any(feature_config.get(k, False) for k in self.feature_widgets.keys()):
            QMessageBox.warning(
                self,
                "No Features Selected",
                "Please select at least one feature to run UMAP.",
            )
            return

        self.run_btn.setEnabled(False)
        self.run_3d_btn.setEnabled(False)
        self.progress.show()
        self.progress.setRange(0, 0)

        # Get selected cluster IDs
        selected_cluster_ids = self.get_selected_cluster_ids()
        target_ids = selected_cluster_ids
        if target_ids is None:
            target_ids = list(dm.cluster_df["cluster_id"].values)

        if len(target_ids) < 5:
            self.run_btn.setEnabled(True)
            self.run_3d_btn.setEnabled(True)
            self.progress.hide()
            QMessageBox.warning(
                self,
                "Insufficient Selection",
                f"Need at least 5 cells for UMAP. Only {len(target_ids)} available.",
            )
            return

        self._run_feature_config = dict(feature_config)
        self.worker_thread = QThread()
        self.worker = UMAPWorker(
            self.main_window.data_manager,
            selected_cluster_ids=selected_cluster_ids,
            n_components=2,
            feature_config=feature_config,
            filter_config=self.get_filter_config(),
        )
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_status)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_processing_finished)

        self.worker_thread.start()

    def run_umap_3d(self):
        dm = self.main_window.data_manager
        total_clusters = len(dm.cluster_df)
        feature_cache = getattr(dm, "feature_cache", {})
        valid_cached = sum(1 for v in feature_cache.values() if v.get("_computed"))
        if valid_cached < total_clusters:
            QMessageBox.warning(
                self,
                "Cache Warming Up",
                f"Please wait for background caching to finish.\n\n"
                f"Ready: {valid_cached} / {total_clusters} cells.\n"
                f"Check the progress bar in the bottom right.",
            )
            return
        self._reset_workers()

        # Verify at least one feature is enabled
        feature_config = self.get_feature_config()
        if not any(feature_config.get(k, False) for k in self.feature_widgets.keys()):
            QMessageBox.warning(
                self,
                "No Features Selected",
                "Please select at least one feature to run UMAP.",
            )
            return

        self.run_btn.setEnabled(False)
        self.run_3d_btn.setEnabled(False)
        self.progress.show()
        self.progress.setRange(0, 0)

        # Get selected cluster IDs
        selected_cluster_ids = self.get_selected_cluster_ids()
        target_ids = selected_cluster_ids
        if target_ids is None:
            target_ids = list(dm.cluster_df["cluster_id"].values)

        if len(target_ids) < 5:
            self.run_btn.setEnabled(True)
            self.run_3d_btn.setEnabled(True)
            self.progress.hide()
            QMessageBox.warning(
                self,
                "Insufficient Selection",
                f"Need at least 5 cells for UMAP. Only {len(target_ids)} available.",
            )
            return

        self._run_feature_config = dict(feature_config)
        self.worker_thread = QThread()
        self.worker = UMAPWorker(
            self.main_window.data_manager,
            selected_cluster_ids=selected_cluster_ids,
            n_components=3,
            feature_config=feature_config,
            filter_config=self.get_filter_config(),
        )
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_status)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_processing_finished)

        self.worker_thread.start()

    def _on_cluster_method_changed(self):
        # Both Ward and K-Means use the same n_clusters parameter.
        self.cluster_param_spin.setRange(2, 100)
        self.cluster_param_spin.setPrefix("# Types: ")

    def run_clustering(self):
        if getattr(self, "embedding", None) is None:
            QMessageBox.warning(self, "No Data", "Please run UMAP first.")
            return

        self.cluster_btn.setEnabled(False)
        self.progress.show()
        self.progress.setRange(0, 0)

        method = self.cluster_method_combo.currentText()
        param = self.cluster_param_spin.value()

        self.cluster_worker_thread = QThread()
        # Cluster on self.embedding (the 2D/3D UMAP projection actually
        # rendered on screen), NOT self.feature_matrix (the pre-UMAP
        # high-dimensional weighted feature space). Clustering in the
        # high-dim space and only painting the resulting labels onto the
        # embedding for display is exactly what produced the "scattered
        # clusters" symptom: UMAP's projection can fold/curve the original
        # space in ways that don't preserve "close in full feature space"
        # as "close on screen," so a pair of points adjacent in the plotted
        # embedding could get different labels from a boundary that only
        # made sense in the original >10-dimensional space. Clustering
        # directly on the embedding guarantees the boundaries the user sees
        # are the boundaries that were actually computed.
        self.cluster_worker = ClusterWorker(self.embedding, method, param)
        self.cluster_worker.moveToThread(self.cluster_worker_thread)

        self.cluster_worker_thread.started.connect(self.cluster_worker.run)
        self.cluster_worker.error.connect(self.on_cluster_error)
        self.cluster_worker.finished.connect(self.on_cluster_finished)

        self.cluster_worker_thread.start()

    def update_status(self, msg):
        self.main_window.status_bar.showMessage(msg)

    def on_error(self, msg):
        self.run_btn.setEnabled(True)
        self.run_3d_btn.setEnabled(True)
        self.progress.hide()
        QMessageBox.critical(self, "Processing Error", msg)
        # Don't call _reset_workers() here since we're already in error state
        # Just clean up the thread refs to be safe
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait(500)
            self.worker_thread = None
        self.worker = None

    def on_cluster_error(self, msg):
        self.cluster_btn.setEnabled(True)
        self.progress.hide()
        QMessageBox.critical(self, "Clustering Error", msg)
        if self.cluster_worker_thread:
            self.cluster_worker_thread.quit()
            self.cluster_worker_thread.wait(2000)
            self.cluster_worker_thread = None
        self.cluster_worker = None

    def on_processing_finished(
        self, embedding, matrix, valid_ids, discarded_ids, metadata, raw_blocks=None
    ):
        self.embedding = np.asarray(embedding)
        self.feature_matrix = np.asarray(matrix)
        self.cluster_ids = np.array(valid_ids)
        self.metadata_df = metadata
        self.discarded_ids = np.array(discarded_ids)
        self.raw_blocks = raw_blocks

        # Determine if result is 3D
        self.is_3d = self.embedding.shape[1] == 3
        self.project_3d_chk.setVisible(self.is_3d)

        self.run_btn.setEnabled(True)
        self.run_3d_btn.setEnabled(True)
        self.progress.hide()
        self.show_ids_btn.setEnabled(True)
        self.cluster_btn.setEnabled(True)

        # Clean up worker thread gracefully
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait(500)
            self.worker_thread = None
        self.worker = None

        # A fresh embedding invalidates the old selection and the old cluster
        # colours: both are indexed by position in cluster_ids, which has just
        # changed. Silently keeping them would paint plausible-looking nonsense.
        self._selection = np.empty(0, dtype=int)
        self._clear_shapes()
        for view in self._population_views:
            view.set_group_colors(None)

        # Rebuild the RF mosaic for exactly the cells that made it into this
        # embedding, so the two panels always describe the same population.
        self.rf_map.set_cells(self.main_window.data_manager, list(self.cluster_ids))
        self.view_toggles.set_available("rf", True)
        self._populate_trace_stacks()

        self.update_plot()

        mode_str = "3D UMAP" if self.is_3d else "2D UMAP"
        selection_info = (
            "selected" if self.get_selected_cluster_ids() is not None else "all"
        )

        # Display weights used in status bar
        weight_strs = []
        for use_key, (chk, slider, w_key, _lbl) in self.feature_widgets.items():
            if chk.isChecked():
                name_short = use_key.replace("use_", "")
                weight_strs.append(f"{name_short}={slider.value() / 10.0:.1f}")
        weights_summary = ", ".join(weight_strs)

        self.main_window.status_bar.showMessage(
            f"{mode_str} Complete. {len(self.cluster_ids)} {selection_info} cells (discarded {len(self.discarded_ids)}). Features: {weights_summary}"
        )

    # ── side-panel trace overlays ────────────────────────────────────────────

    def _populate_trace_stacks(self):
        """Fill the per-feature trace overlays for the run just finished.

        A stack is only shown when its feature was actually enabled for this
        run and the block holds real data. Showing the chirp overlay after a
        run that excluded chirp would invite reading structure into the
        embedding that never contributed to it — and chirp in particular is
        often unavailable, since it is precomputed offline.
        """
        blocks = self.raw_blocks or {}
        ids = list(self.cluster_ids) if self.cluster_ids is not None else []
        # Tracked explicitly rather than via isVisible(): a widget inside a
        # non-current QTabWidget page reports isVisible() False even though it
        # is populated, which would silently skip its highlight whenever the
        # user is looking at another tab. This set means "holds data for this
        # run" — whether it is *on screen* is the toggle bar's business.
        self._active_stacks = set()

        for key, stack in self.trace_stacks.items():
            used = bool(self._run_feature_config.get(self._stack_feature_keys[key]))
            block = blocks.get(key)
            block = np.asarray(block) if block is not None else None
            has_data = block is not None and block.ndim == 2 and block.shape[1] >= 2

            if not (used and has_data):
                stack.set_traces(None, [])
                self.view_toggles.set_available(
                    key,
                    False,
                    "This feature was not enabled for the run on screen."
                    if has_data
                    else "This run has no data for that feature.",
                )
                continue

            if key == "chirp":
                # Match the Chirp dashboard's 25 ms Gaussian. Unsmoothed, 400
                # normalised 25 s PSTHs overlay into solid hash in a 130 px
                # strip and nothing is legible.
                bin_ms = self._chirp_bin_ms()
                stack.set_smoothing((25.0 / bin_ms) if bin_ms else 0.0)
            stack.set_traces(block, ids, x=self._trace_x_axis(key, block.shape[1]))
            self.view_toggles.set_available(key, True)
            self._active_stacks.add(key)

        self._apply_chirp_regions()

    def _on_view_toggled(self, key, is_open):
        """Bring a reopened view back up to date with the current selection.

        A closed view is skipped by _highlight — repainting a mosaic and three
        overlays costs real time on held arrow keys — so without this it would
        come back showing whatever was selected when it was closed.
        """
        if not is_open:
            return
        view = self.view_toggles.widget_for(key)
        if view is None:
            return
        view.highlight(self._pending_highlight or [])
        view.set_selection_mode(self._selection_mode if self.embedding is not None else "off")

    def _trace_x_axis(self, key, n_cols):
        """Real units for a block's x axis where they are recoverable."""
        if key == "acg":
            # get_standard_plot_data builds the ACG symmetric about zero in
            # 1 ms bins, so the half-width follows from the column count and
            # stays correct if MAX_LAG ever changes.
            return np.arange(n_cols, dtype=float) - (n_cols - 1) / 2.0
        if key == "chirp":
            bin_ms = self._chirp_bin_ms()
            if bin_ms:
                return np.arange(n_cols, dtype=float) * bin_ms / 1000.0
        return np.arange(n_cols, dtype=float)

    def _chirp_bin_ms(self):
        dm = getattr(self.main_window, "data_manager", None)
        data = getattr(dm, "chirp_data", None) if dm is not None else None
        if not isinstance(data, dict):
            return None
        try:
            return float(data["bin_size_ms"])
        except (KeyError, TypeError, ValueError):
            return None

    # Same phases the Chirp dashboard shades, so the two panels read alike.
    _CHIRP_PHASE_TINTS = {
        "phase_step_on": (0.47, 0.47, 0.47, 0.16),
        "phase_step_off": (0.47, 0.47, 0.47, 0.16),
        "phase_freq_sweep": (0.31, 0.55, 0.86, 0.14),
        "phase_contrast": (0.86, 0.55, 0.31, 0.14),
    }

    def _apply_chirp_regions(self):
        """Shade the chirp overlay by stimulus phase.

        Without this the overlay is 25 s of unlabelled wiggle; with it you can
        see *which* phase a candidate group agrees or disagrees in, which is
        the whole reason to look at the chirp rather than its PCA scores.
        """
        if "chirp" not in self._active_stacks:
            return
        stack = self.trace_stacks["chirp"]
        dm = getattr(self.main_window, "data_manager", None)
        data = getattr(dm, "chirp_data", None) if dm is not None else None
        bin_ms = self._chirp_bin_ms()
        if not isinstance(data, dict) or not bin_ms:
            return

        regions = []
        for phase, tint in self._CHIRP_PHASE_TINTS.items():
            bounds = data.get(phase)
            if bounds is None or len(bounds) != 2:
                continue
            start, end = float(bounds[0]), float(bounds[1])
            if end > start:
                regions.append(
                    (start * bin_ms / 1000.0, end * bin_ms / 1000.0, tint)
                )
        if regions:
            stack.set_regions(regions)

    def _highlight(self, cluster_ids):
        """Single entry point for "these cells are the current selection".

        Every side panel repaints from here, so the mosaic and the trace
        overlays can never end up describing different selections — which is
        exactly what would happen if each call site painted the panels it
        happened to know about.
        """
        ids = [int(c) for c in (cluster_ids or [])]
        # Remembered whatever the source was — a lasso, a right-click pick or the
        # tree — so showEvent's catch-up can never resurrect an older selection
        # over a newer one made in this panel. It is also what a view closed at
        # the moment of selection is caught up from when it reopens.
        self._pending_highlight = ids
        # Closed views are skipped rather than painted invisibly: the mosaic and
        # three overlays cost ~56 ms for a 35-cell folder at 700 cells, which is
        # felt on held arrow keys and entirely wasted on a hidden canvas.
        if self.view_toggles.is_open("rf"):
            self.rf_map.highlight(ids)
        for key in self._active_stacks:
            if self.view_toggles.is_open(key):
                self.trace_stacks[key].highlight(ids)

    def highlight_cells(self, cluster_ids):
        """Highlight *cluster_ids* from outside the panel (e.g. a tree group).

        Deferred while the tab is hidden. This is called from the Tier 1
        selection path, where it must stay cheap: painting the mosaic and three
        overlays costs ~12 ms for one cell and ~56 ms for a 35-cell folder at
        700 cells, which is real lag on held arrow keys and entirely wasted
        when the UMAP tab is not the one on screen. The pending selection is
        applied in showEvent instead, so the panel is correct the moment it
        becomes visible.

        Silently does nothing before a run: there is no population to highlight
        within yet.
        """
        if self.embedding is None or self.cluster_ids is None:
            return
        if self.isVisible():
            self._highlight(cluster_ids)
        else:
            self._pending_highlight = [int(c) for c in (cluster_ids or [])]

    # Discrete colormap the embedding uses for cluster labels. The population
    # views read from the same one so a group is the same colour in both.
    _CLUSTER_CMAP = "tab20"

    def cluster_label_colors(self, labels):
        """``cluster_id -> RGBA`` for *labels*, in the scatter's own colours.

        Matplotlib maps integer labels through tab20 by normalising them over
        the range present, so reproducing that normalisation here is what keeps
        "cluster 3 is green" true on the mosaic as well as the embedding.
        """
        labels = np.asarray(labels)
        ids = np.asarray(self.cluster_ids)
        if labels.size == 0 or labels.size != ids.size:
            return {}

        # matplotlib.colormaps replaced cm.get_cmap, which is gone in 3.9+.
        try:
            cmap = matplotlib.colormaps[self._CLUSTER_CMAP]
        except (AttributeError, KeyError):
            cmap = matplotlib.cm.get_cmap(self._CLUSTER_CMAP)
        lo, hi = float(np.min(labels)), float(np.max(labels))
        span = (hi - lo) or 1.0

        out = {}
        for cid, lbl in zip(ids, labels):
            if lbl < 0:  # -1 is noise, not a group
                continue
            out[int(cid)] = cmap((float(lbl) - lo) / span)
        return out

    def _apply_cluster_colors(self, labels):
        """Paint the mosaic and trace overlays by cluster membership.

        Highlighting shows one group at a time, which answers "does this group
        tile?" but not "did the clustering carve up the array sensibly?".
        Colouring every resting outline by its group answers the second, which
        is the question you actually have the moment a clustering run lands.
        """
        colors = self.cluster_label_colors(labels)
        for view in self._population_views:
            view.set_group_colors(colors)

    def on_cluster_finished(self, labels, method_name):
        self.metadata_df[method_name] = labels
        self.cluster_btn.setEnabled(True)
        self.progress.hide()
        if self.cluster_worker_thread:
            self.cluster_worker_thread.quit()
            self.cluster_worker_thread.wait(2000)
            self.cluster_worker_thread = None
        self.cluster_worker = None

        self.color_combo.setCurrentText(method_name)
        self._apply_cluster_colors(labels)
        self.update_plot()
        self.main_window.status_bar.showMessage(f"{method_name} clustering complete.")

        # --- AUTO-GROUP LOGIC ---
        if self.auto_group_chk.isChecked():
            # Skip noise points (-1)
            valid_labels = labels[labels >= 0]
            if len(valid_labels) > 0:
                self.apply_grouping(labels, method_name)

    def apply_grouping(self, labels, method_name):
        try:
            from ..callbacks import group_clusters_in_tree

            unique_labels = np.unique(labels)
            count = 0

            for lbl in unique_labels:
                if lbl == -1:
                    continue  # Skip noise points
                subset_indices = np.where(labels == lbl)[0]
                group_cluster_ids = self.cluster_ids[subset_indices]
                group_name = (
                    f"Type_{lbl+1}"
                    if method_name in ("K-Means", "Ward")
                    else f"Cluster_{lbl}"
                )

                # Use our safe, in-place tree modifier!
                group_clusters_in_tree(self.main_window, group_cluster_ids, group_name)
                count += 1

            self.main_window.status_bar.showMessage(
                f"Auto-Group: created {count} groups."
            )

            # Force the main tree view to collapse all groups after auto-grouping
            if hasattr(self.main_window, "tree_view"):
                self.main_window.tree_view.collapseAll()

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Failed to auto-group: {e}")
            from qtpy.QtWidgets import QMessageBox

            QMessageBox.warning(self, "Auto-Group Error", str(e))

    def restyle_plots(self, colors):
        """Updates plot styling based on the provided color scheme."""
        self.fig.patch.set_facecolor(colors["bg_panel"])
        self.ax.set_facecolor(colors["bg_panel"])

        # Update button colors
        self.run_btn.setStyleSheet(
            f"background-color: {colors['accent']}; font-weight: bold;"
        )
        self.run_3d_btn.setStyleSheet(
            f"background-color: {colors['accent_positive']}; font-weight: bold;"
        )

        # Update stored colors
        self._umap_colors = {
            "accent": colors["accent"],
            "accent_positive": colors["accent_positive"],
            "bg_panel": colors["bg_panel"],
        }

        if getattr(self, "rf_map", None) is not None:
            self.rf_map.restyle(self._rf_map_colors(colors))

        for stack in getattr(self, "trace_stacks", {}).values():
            stack.restyle(self._rf_map_colors(colors))
        # restyle rebuilds each background from scratch, which drops the phase
        # shading with it.
        self._apply_chirp_regions()

        self.update_plot()

    def update_plot(self, _color_mode=None):
        if self.embedding is None:
            return

        colors = resolve_theme_colors(self.main_window.get_current_colors())

        # Suppress compositing for the duration of the rebuild: every artist is
        # about to be recreated, so any blit against the old background would
        # smear the previous embedding across the new one.
        self._building = True
        self._blit_bg = None
        self._overlay_artist = None
        self._lasso_poly = None

        # Clean up old plot
        if getattr(self, "cbar", None):
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None

        # Re-create axes if dimension changed (2D vs 3D)
        current_is_3d = hasattr(self.ax, "zaxis")

        if self.is_3d != current_is_3d:
            self.fig.clear()
            if self.is_3d:
                self.ax = self.fig.add_subplot(111, projection="3d")
            else:
                self.ax = self.fig.add_subplot(111)
            self.ax.set_facecolor(colors["bg_panel"])
            self.fig.patch.set_facecolor(colors["bg_panel"])
        else:
            self.ax.clear()
            self.ax.set_facecolor(colors["bg_panel"])

        # ... (rest of update_plot, updating hardcoded colors) ...
        # Determine Colors
        mode = self.color_combo.currentText()
        c = colors["plot_highlight"]
        cmap = None
        is_discrete = False

        if mode == "KSLabel":
            if "KSLabel" in self.metadata_df:
                labels = self.metadata_df["KSLabel"].values
                unique_labels = np.unique(labels)
                label_map = {l: i for i, l in enumerate(unique_labels)}
                c = [label_map.get(l, 0) for l in labels]
                cmap = "tab10"
                is_discrete = True
        elif mode == "Polarity":
            if "Polarity" in self.metadata_df:
                labels = self.metadata_df["Polarity"].values
                unique_labels = np.unique(labels)
                label_map = {l: i for i, l in enumerate(unique_labels)}
                c = [label_map.get(l, 0) for l in labels]
                cmap = "coolwarm"
                is_discrete = True
        elif mode == "Firing Rate":
            if "Firing Rate" in self.metadata_df:
                c = self.metadata_df["Firing Rate"].values
                cmap = "plasma"
        elif mode == "ISI Violations":
            if "isi_violations" in self.metadata_df:
                c = self.metadata_df["isi_violations"].values
                cmap = "magma_r"
        elif mode == "Time to Peak":
            if "Time to Peak" in self.metadata_df:
                c = self.metadata_df["Time to Peak"].values
                cmap = "viridis"
        elif mode == "RF Area":
            if "RF Area" in self.metadata_df:
                c = self.metadata_df["RF Area"].values
                cmap = "viridis"
        elif mode == "Color Opponency":
            if "Color Opponency" in self.metadata_df:
                c = self.metadata_df["Color Opponency"].values
                cmap = "cool"
        elif mode == "K-Means":
            if "K-Means" in self.metadata_df:
                c = self.metadata_df["K-Means"].values
                cmap = "tab20"
                is_discrete = True
            else:
                c = colors["text_secondary"]
        elif mode == "Ward":
            if "Ward" in self.metadata_df:
                # Ward produces clean integer labels 0..n_clusters-1, no noise.
                c = self.metadata_df["Ward"].values
                cmap = "tab20"
                is_discrete = True
            else:
                c = colors["text_secondary"]

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
                edgecolors="none",
            )
            self.ax.set_xlabel("Dim 1", color=colors["text_secondary"])
            self.ax.set_ylabel("Dim 2", color=colors["text_secondary"])
            self.ax.set_zlabel("Dim 3", color=colors["text_secondary"])
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
                edgecolors="none",
            )
            # --- 2D axis polish ---
            self.ax.set_xlabel(
                "UMAP Dimension 1", color=colors["text_secondary"], fontsize=9
            )
            self.ax.set_ylabel(
                "UMAP Dimension 2", color=colors["text_secondary"], fontsize=9
            )
            self.ax.tick_params(colors=colors["text_secondary"], labelsize=8)
            # Hide the harsh default box; keep only left + bottom as thin guides
            self.ax.spines["top"].set_visible(False)
            self.ax.spines["right"].set_visible(False)
            self.ax.spines["left"].set_edgecolor(colors["border_subtle"])
            self.ax.spines["left"].set_linewidth(0.8)
            self.ax.spines["bottom"].set_edgecolor(colors["border_subtle"])
            self.ax.spines["bottom"].set_linewidth(0.8)
            # Soft dotted grid for spatial reference without visual noise
            self.ax.grid(
                True, color=colors["border_subtle"], linestyle=":", alpha=0.5, zorder=0
            )

            # Overlay scatter holding ONLY the selected points. animated=True
            # keeps it out of the normal draw pass, so the cached background
            # stays clean instead of baking in a stale selection.
            self._overlay_artist = self.ax.scatter(
                [],
                [],
                s=46,
                linewidths=0.6,
                edgecolors=colors["bg_panel"],
                color=colors.get("plot_highlight", "#f97316"),
                alpha=0.95,
                zorder=6,
                animated=True,
            )

        if (
            mode != "KSLabel"
            and not (mode == "K-Means" and is_discrete)
            and not (mode == "Polarity" and is_discrete)
            and not (mode == "Ward" and is_discrete)
        ):
            self.cbar = self.fig.colorbar(
                scatter, ax=self.ax, pad=0.1 if self.is_3d else 0.05
            )
            # Style colorbar ticks
            if self.cbar:
                self.cbar.ax.yaxis.set_tick_params(color=colors["text_secondary"])
                for label in self.cbar.ax.get_yticklabels():
                    label.set_color(colors["text_secondary"])

        # Add selection info to title
        selection_info = (
            "selected" if self.get_selected_cluster_ids() is not None else "all"
        )
        title_prefix = "UMAP (3D)" if self.is_3d else "UMAP (2D)"
        self.ax.set_title(
            f"{title_prefix} - {len(self.cluster_ids)} {selection_info} cells - Color: {mode}",
            color=colors["text_primary"],
        )
        self.ax.tick_params(colors=colors["text_secondary"])

        # Re-attach the selection tools to the fresh axes, then restore the
        # selection onto them — a colour-mode change must not silently drop it.
        self.update_selector()
        self._sync_overlay(self._selection)
        self._sync_selection_panel()

        self._building = False
        # Full render once; _on_canvas_draw snapshots the clean background.
        self.canvas.draw()

    def show_group_ids(self):
        if self.metadata_df is None:
            return

        mode = self.color_combo.currentText()
        if mode not in ["KSLabel", "K-Means", "Polarity", "Ward"]:
            QMessageBox.information(
                self,
                "Info",
                "Group IDs only available for discrete categories (KSLabel, K-Means, Polarity, Ward).",
            )
            return

        if mode not in self.metadata_df:
            return

        groups = self.metadata_df.groupby(mode)["cluster_id"].apply(list)
        text_output = ""
        for group_name, ids in groups.items():
            text_output += f"=== Group {group_name} ({len(ids)} cells) ===\n"
            id_strs = [str(x) for x in sorted(ids)]
            chunked = [
                ", ".join(id_strs[i : i + 10]) for i in range(0, len(id_strs), 10)
            ]
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

    # ── selection ────────────────────────────────────────────────────────────

    def on_select(self, verts):
        """Adopt the cells inside *verts* as the selection.

        Kept as the public entry point (the 3D projection path and external
        callers still use it), but it no longer prompts: the selection panel
        beside the mosaic offers the group instead, so refining a shape and
        watching the RFs re-tile is possible without dismissing a dialog first.
        """
        if self.embedding is None or verts is None or len(verts) < 3:
            return

        pts = self._selection_points()
        if pts is None:
            return

        mask = MplPath(np.asarray(verts, dtype=float)).contains_points(pts)
        self._set_selection(np.where(mask)[0])

    def _selection_points(self):
        """Screen-plane coordinates to hit-test against, or None.

        In 3D this is the projection, which is only meaningful while "Lasso 3D
        Proj" is on — without it there is no defensible mapping from a flat
        shape to points in a rotated volume, so selection is simply refused.
        """
        if self.embedding is None:
            return None
        if not self.is_3d:
            return self.embedding[:, :2]
        if not self.project_3d_chk.isChecked():
            self.main_window.status_bar.showMessage(
                "Enable 'Lasso 3D Proj' to select in a 3D embedding.", 4000
            )
            return None
        return self._embedding_2d()

    def _set_selection(self, indices):
        """Adopt *indices* as the selection and repaint everything showing it."""
        indices = np.asarray(indices, dtype=int)
        if not np.array_equal(indices, self._selection):
            self._selection = indices
            ids = [
                int(self.cluster_ids[i])
                for i in indices
                if 0 <= i < len(self.cluster_ids)
            ]
            self._sync_overlay(indices)
            self._sync_selection_panel()
            self._highlight(ids)
        self._composite()

    def _selected_cluster_ids(self):
        return [
            int(self.cluster_ids[i])
            for i in self._selection
            if 0 <= i < len(self.cluster_ids)
        ]

    def _sync_overlay(self, indices):
        """Move the animated overlay scatter onto the selected points.

        3D has no overlay — matplotlib's 3D collections cannot be repositioned
        cheaply, and the side panels carry the selection there instead.
        """
        if self._overlay_artist is None:
            return
        pts = self.embedding[:, :2] if self.embedding is not None else None
        if pts is None or len(indices) == 0:
            self._overlay_artist.set_offsets(np.empty((0, 2)))
        else:
            self._overlay_artist.set_offsets(pts[indices])

    def _sync_selection_panel(self):
        n = len(self._selection)
        if n:
            plural = "s" if n != 1 else ""
            self.sel_count_label.setText(f"{n} cell{plural} selected")
            self.btn_create_group.setText(f"Create {self._next_group_name()}")
        else:
            self.sel_count_label.setText("Drag on the embedding to select cells")
            self.btn_create_group.setText("Create group")
        self.btn_create_group.setEnabled(n > 0)
        self.btn_clear_selection.setEnabled(n > 0)

    def _next_group_name(self):
        dm = self.main_window.data_manager
        if dm is None:
            return "group"
        return f"Nc{dm.new_class_id}"

    def clear_selection(self):
        """Drop the selection and every selection shape."""
        self._clear_shapes()
        self._set_selection(np.empty(0, dtype=int))

    def _clear_shapes(self, keep=None):
        for sel in (self._rect_selector, self._lasso_selector):
            if sel is not None:
                retire(sel)
        if self._lasso_poly is not None:
            self._lasso_poly.set_visible(False)
        for view in self._population_views:
            if view is not keep:
                view.clear_selection_shapes()

    def _create_group_from_selection(self):
        ids = self._selected_cluster_ids()
        if not ids:
            return
        dm = self.main_window.data_manager
        if dm is None:
            return
        name = f"Nc{dm.new_class_id}"
        dm.new_class_id += 1

        from ..callbacks import group_clusters_in_tree

        group_clusters_in_tree(self.main_window, ids, name)
        # Shape and selection are deliberately kept: refining a boundary and
        # regrouping is the normal workflow.
        self.sel_count_label.setText(f"{name} created — {len(ids)} cells")
        self.btn_create_group.setText(f"Create {self._next_group_name()}")
        self.main_window.status_bar.showMessage(
            f"Created {name} with {len(ids)} cells.", 4000
        )

    def show_context_menu(self):
        """The selection panel's actions, raised at the cursor."""
        ids = self._selected_cluster_ids()
        if not ids:
            return

        menu = QMenu(self)
        n = len(ids)
        plural = "s" if n != 1 else ""
        create_action = menu.addAction(f"Create group from {n} cluster{plural}")
        trash_action = menu.addAction(f"Move {n} cell{plural} to Trash")
        menu.addSeparator()
        clear_action = menu.addAction("Clear selection")

        action = menu.exec_(QCursor.pos())
        if action == create_action:
            self._create_group_from_selection()
        elif action == trash_action:
            self._trash_selection()
        elif action == clear_action:
            self.clear_selection()

    def _trash_selection(self):
        """Send the selected cells to the tree's Trash folder.

        They stay in the embedding and the mosaic: this run's UMAP was fitted to
        a fixed cell set, and dropping points out of it would leave the layout
        describing a population that is no longer shown. The next run excludes
        them — get_selected_cluster_ids already filters Trash — and they are
        gone from the table immediately.
        """
        ids = self._selected_cluster_ids()
        if not ids:
            return
        from ..callbacks import move_cluster_ids_to_trash

        n = move_cluster_ids_to_trash(self.main_window, ids)
        self.clear_selection()
        if n:
            plural = "s" if n != 1 else ""
            self.sel_count_label.setText(f"{n} cell{plural} moved to Trash")
            self.main_window.status_bar.showMessage(
                f"Moved {n} cell{plural} to Trash — they leave the embedding "
                "on the next UMAP run.",
                5000,
            )
        else:
            self.sel_count_label.setText(
                "Nothing moved — those cells are not in the tree"
            )

    def _on_population_selection(self, source, cluster_ids):
        """A brush landed in the RF mosaic or one of the trace overlays.

        The mosaic answers "do these tile?" and the overlays answer "do these
        share a response?" — being able to select *from* those views closes the
        loop, so a group spotted there becomes the selection on the embedding
        too.
        """
        wanted = {int(c) for c in (cluster_ids or [])}
        ids = list(self.cluster_ids) if self.cluster_ids is not None else []
        indices = [i for i, c in enumerate(ids) if int(c) in wanted]
        self._clear_shapes(keep=source)
        self._set_selection(np.asarray(indices, dtype=int))

    # ── selector plumbing (owner protocol for live_selectors) ────────────────

    def _begin_selection(self, sel):
        other = self._lasso_selector if sel is self._rect_selector else self._rect_selector
        if other is not None:
            retire(other)
        if self._lasso_poly is not None:
            self._lasso_poly.set_visible(False)
        for view in self._population_views:
            view.clear_selection_shapes()

    def _apply_rect(self, sel):
        try:
            xmin, xmax, ymin, ymax = sel.extents
        except Exception:
            logger.debug("rectangle extents unavailable", exc_info=True)
            self._composite()
            return
        pts = self._selection_points()
        if pts is None:
            self._composite()
            return
        mask = (
            (pts[:, 0] >= xmin)
            & (pts[:, 0] <= xmax)
            & (pts[:, 1] >= ymin)
            & (pts[:, 1] <= ymax)
        )
        self._set_selection(np.where(mask)[0])

    def _on_rect_live(self, sel):
        # 3D has no cached background to blit against (the axes repaint on every
        # rotation), so previewing per mouse-move would force a full redraw each
        # time. There the selection lands on release instead.
        if self.is_3d:
            self._composite()
            return
        self._apply_rect(sel)

    def _on_lasso_live(self, sel):
        if self.is_3d:
            self._composite()
            return
        verts = sel.verts
        if not verts or len(verts) < 3:
            self._composite()
            return
        self.on_select(verts)

    def _apply_lasso_verts(self, _index, verts):
        self.on_select(verts)

    def _freeze_lasso(self, _sel, verts):
        """Keep the drawn path on screen after release.

        LassoSelector blanks its own line in _release, so without this the
        outline disappeared exactly when you wanted to look at it.
        """
        if self.is_3d or self.ax is None:
            return
        xy = np.asarray(verts, dtype=float)
        if self._lasso_poly is None:
            colors = resolve_theme_colors(self.main_window.get_current_colors())
            self._lasso_poly = MplPolygon(
                xy,
                closed=True,
                facecolor=to_rgba(colors["accent"], 0.14),
                edgecolor=to_rgba(colors["text_primary"], 0.85),
                linewidth=1.4,
                zorder=5,
                animated=True,
            )
            self.ax.add_patch(self._lasso_poly)
        else:
            self._lasso_poly.set_xy(xy)
        self._lasso_poly.set_visible(True)
        self._composite()

    def _composite(self):
        """The single painter for this canvas.

        Restore the clean background, then stamp the animated layers back on in
        a fixed order: selected points, the frozen lasso outline, then whatever
        the live selector is drawing. Because every repaint goes through here,
        no layer can wipe another.
        """
        if self._building or self._defer_composite:
            return
        if self.is_3d or self._blit_bg is None:
            # 3D repaints wholesale on every rotation, so there is no stable
            # background to cache. draw_idle re-enters _on_canvas_draw.
            self.canvas.draw_idle()
            return

        self.canvas.restore_region(self._blit_bg)
        if self._overlay_artist is not None:
            self.ax.draw_artist(self._overlay_artist)
        if self._lasso_poly is not None and self._lasso_poly.get_visible():
            self.ax.draw_artist(self._lasso_poly)
        for sel in (self._rect_selector, self._lasso_selector):
            if sel is None or not sel.active:
                continue
            for artist in sel.artists:
                if artist.get_visible() and artist.axes is not None:
                    artist.axes.draw_artist(artist)
        self.canvas.blit(self.fig.bbox)

    def _on_canvas_draw(self, _event):
        """Cache the freshly rendered background, then repaint the live layers."""
        if self._building or self.is_3d:
            self._blit_bg = None
            return
        self._blit_bg = self.canvas.copy_from_bbox(self.fig.bbox)
        self._composite()

    def resizeEvent(self, event):
        """Invalidate the blit cache; the following draw_event re-snapshots it."""
        super().resizeEvent(event)
        self._blit_bg = None

    # Colour modes that hold integer cluster assignments rather than a
    # continuous metric — only these support "highlight this whole cluster".
    _CLUSTER_COLOR_MODES = ("Ward", "K-Means")

    def _current_cluster_labels(self):
        """Labels for the clustering currently being displayed, or None.

        Keyed off the colour mode rather than the last run, so the pick always
        matches the grouping actually on screen.
        """
        mode = self.color_combo.currentText()
        if mode not in self._CLUSTER_COLOR_MODES:
            return None
        if self.metadata_df is None or mode not in self.metadata_df:
            return None
        return np.asarray(self.metadata_df[mode])

    def _embedding_2d(self):
        """Screen-plane coordinates for every point, or None if unavailable."""
        if self.embedding is None:
            return None
        if not self.is_3d:
            return self.embedding[:, :2]
        try:
            xs, ys, zs = (
                self.embedding[:, 0],
                self.embedding[:, 1],
                self.embedding[:, 2],
            )
            x2, y2, _ = proj3d.proj_transform(xs, ys, zs, self.ax.get_proj())
            return np.column_stack((x2, y2))
        except Exception:
            logger.debug("3D projection for click-pick failed", exc_info=True)
            return None

    def _on_canvas_click(self, event):
        """Right-click a point to select its whole cluster; again to act on it.

        The first right-click picks — which is the fast path the panel is built
        around — and because the pick *is* the selection, right-clicking a point
        that is already selected can safely mean "do something with this"
        instead. That keeps one-motion picking while making group and trash
        reachable where you would expect them, without a modifier key.

        Falls back to the single nearest cell when no clustering is displayed,
        so the pick is still useful straight after a UMAP run.
        """
        if event.button != 3 or event.inaxes is not self.ax:
            return
        if self.embedding is None or event.xdata is None or event.ydata is None:
            return

        pts = self._embedding_2d()
        if pts is None or len(pts) == 0:
            return

        d2 = (pts[:, 0] - event.xdata) ** 2 + (pts[:, 1] - event.ydata) ** 2
        idx = int(np.argmin(d2))

        # Ignore clicks in empty space: without this, a click anywhere on the
        # canvas would light up whichever cluster happened to be least far away.
        # Checked before the menu, so a stray click far from the data clears the
        # selection rather than offering to trash it.
        span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
        if span > 0 and np.sqrt(d2[idx]) > 0.05 * span:
            self.clear_selection()
            self.main_window.status_bar.showMessage(
                "No cell near that point — highlight cleared.", 3000
            )
            return

        if len(self._selection) and idx in set(self._selection.tolist()):
            self.show_context_menu()
            return

        # The pick becomes the selection, not just a highlight, so the panel can
        # offer to group it the same way a lasso can.
        self._clear_shapes()
        labels = self._current_cluster_labels()
        if labels is not None and len(labels) == len(self.cluster_ids):
            label = labels[idx]
            if label >= 0:  # -1 is noise, which is not a cluster
                members = np.where(labels == label)[0]
                self._set_selection(members)
                self.main_window.status_bar.showMessage(
                    f"Selected {len(members)} cells in "
                    f"{self.color_combo.currentText()} cluster {label}.",
                    5000,
                )
                return

        self._set_selection(np.array([idx], dtype=int))
        self.main_window.status_bar.showMessage(
            f"Selected cell {int(np.asarray(self.cluster_ids)[idx])} "
            "(run clustering to pick whole clusters).",
            5000,
        )

    def _on_rf_cell_clicked(self, cluster_id):
        """Clicking an RF outline in the mosaic brings that cell up in the app."""
        cluster_id = int(cluster_id)
        self._highlight([cluster_id])
        focus = getattr(self.main_window, "focus_cluster", None)
        if callable(focus) and focus(cluster_id):
            self.main_window.status_bar.showMessage(
                f"Selected cell {cluster_id} from the RF mosaic.", 4000
            )

    def create_group(self, ids):
        """Group *ids* under an auto-numbered name (kept for external callers)."""
        ids = [int(c) for c in (ids or [])]
        if not ids:
            return
        dm = self.main_window.data_manager
        name = f"Nc{dm.new_class_id}" if dm is not None else "group"
        if dm is not None:
            dm.new_class_id += 1
        from ..callbacks import group_clusters_in_tree

        group_clusters_in_tree(self.main_window, ids, name)

    def update_selector(self):
        """Rebuild the live selectors on the current axes and arm the chosen one.

        Called after every plot rebuild: ax.clear() leaves a selector's canvas
        callbacks connected, so a stale one would go on handling events for an
        embedding that no longer exists.
        """
        if not hasattr(self, "ax"):
            return

        self._selection_mode = (
            "lasso" if self.selector_combo.currentText() == "Lasso Tool" else "rect"
        )

        for attr in ("current_selector", "selector", "_rect_selector", "_lasso_selector"):
            sel = getattr(self, attr, None)
            if sel is not None:
                try:
                    sel.set_active(False)
                    sel.disconnect_events()
                except Exception:
                    logger.debug("selector teardown failed", exc_info=True)
                setattr(self, attr, None)
        self._lasso_poly = None

        colors = resolve_theme_colors(self.main_window.get_current_colors())
        palette = {
            "bg": colors["bg_panel"],
            "accent": colors["accent"],
            "text_primary": colors["text_primary"],
        }
        # button=[1] so right-click stays free for _on_canvas_click's
        # "highlight this point's cluster" pick; the selectors otherwise grab
        # every mouse button and would swallow it.
        self._rect_selector = make_rect_selector(
            self.ax, self, palette, onselect=lambda *_: self._apply_rect(self._rect_selector)
        )
        self._lasso_selector = make_lasso_selector(self.ax, self, palette)

        self._rect_selector.set_active(self._selection_mode == "rect")
        self._lasso_selector.set_active(self._selection_mode == "lasso")
        retire(self._rect_selector)
        retire(self._lasso_selector)

        # Same tool in the mosaic and the trace overlays, so one choice covers
        # every view a selection can be drawn in.
        for view in self._population_views:
            view.set_selection_mode(self._selection_mode)

        # Kept pointing at whichever tool is live for anything still reading it.
        self.current_selector = (
            self._lasso_selector
            if self._selection_mode == "lasso"
            else self._rect_selector
        )

    def on_select_rect(self, eclick, erelease):
        """Bridge a bare RectangleSelector callback into the selection pipeline."""
        if (
            eclick.xdata is None
            or eclick.ydata is None
            or erelease.xdata is None
            or erelease.ydata is None
        ):
            return

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.on_select([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)])
