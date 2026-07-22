"""
cell_tracer_dialog.py
=====================
Cell Tracer — draw a freehand lasso over a GFP cell in the EI + photo overlay,
auto-select the electrodes underneath, and rank every cluster by how well its
spatial EI footprint matches the selected electrode subset.

Usage (from EIPanel toolbar):
    from .cell_tracer_dialog import CellTracerDialog
    dlg = CellTracerDialog(parent=self, ei_panel=self)
    dlg.show()

Public API consumed by EIPanel:
    CellTracerDialog(parent, ei_panel)   — construct and show
    dlg.cluster_selected                 — Signal(int) emitted when user picks a
                                          result row; EIPanel connects this to
                                          its own update_cluster() method.
"""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path as MplPath  # lasso polygon hit-test
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from qtpy.QtCore import Qt, Signal, QThread, QObject
from qtpy.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSplitter,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QComboBox,
    QSlider,
    QProgressBar,
    QFrame,
)
from qtpy.QtGui import QColor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ei_panel import EIPanel

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker — correlation computation off the UI thread
# ---------------------------------------------------------------------------


class _CorrelationWorker(QObject):
    """
    Score every cluster by how much of its EI energy falls inside the lasso.

    Algorithm
    ---------
    For each cluster:
        spatial[ch] = max(|EI[ch, :]|)          # peak amplitude per channel
        score = L2_norm(spatial[lasso_chs])
                  / L2_norm(spatial[all_chs])    # fraction of energy in lasso

    A cluster whose soma sits under the lasso concentrates its signal on the
    lasso electrodes → score near 1.0.
    A cluster whose soma is elsewhere has most energy outside → score near 0.

    Crucially, no reference cluster is used.  The score is a pure geometric
    property of each cluster's EI relative to the drawn region.

    This runs in < 100 ms for 512-channel / 500-cluster datasets.
    """

    finished = Signal(list)  # list of (ks_cluster_id, score, best_chan)
    error = Signal(str)

    def __init__(self, dm, selected_ch_indices: np.ndarray):
        """
        dm                  — DataManager instance
        selected_ch_indices — 1-D int array of electrode indices inside lasso
        """
        super().__init__()
        self._dm = dm
        self._sel = selected_ch_indices

    def run(self):
        try:
            results = self._compute()
            self.finished.emit(results)
        except Exception as exc:
            logger.exception("CellTracer correlation failed")
            self.error.emit(str(exc))

    def _compute(self) -> list[tuple[int, float, int]]:
        """
        Returns list of (ks_cluster_id, score, best_chan), sorted by score desc.

        vision_eis keys are Vision IDs (1-indexed when paired with KS).
        We convert: ks_id = vision_id - 1   (or vision_id if is_vision_only).
        best_chan is looked up from cluster_df to match the CH column in the
        main sidebar table.
        """
        dm = self._dm
        sel = self._sel
        is_vision_only = getattr(dm, "is_vision_only", False)

        if dm.vision_eis is None:
            return []

        # Pre-build vision_id → best_chan from cluster_df
        best_chan_map: dict[int, int] = {}
        if hasattr(dm, "cluster_df") and dm.cluster_df is not None:
            df = dm.cluster_df
            id_col = "cluster_id" if "cluster_id" in df.columns else None
            chan_col = "best_chan" if "best_chan" in df.columns else None
            if id_col and chan_col:
                for _, row in df[[id_col, chan_col]].iterrows():
                    ks_id = int(row[id_col])
                    vid = ks_id if is_vision_only else ks_id + 1
                    best_chan_map[vid] = int(row[chan_col])

        results = []
        for vision_id, entry in dm.vision_eis.items():
            try:
                ei_arr = getattr(entry, "ei", None)
                if not isinstance(ei_arr, np.ndarray) or ei_arr.size == 0:
                    continue

                vision_id_int = int(vision_id)
                ks_id = vision_id_int if is_vision_only else vision_id_int - 1

                # Peak amplitude per channel (n_ch,)
                spatial = np.max(np.abs(ei_arr), axis=1)

                # Clamp lasso indices to valid channel range
                valid_sel = sel[sel < len(spatial)]

                total_norm = np.linalg.norm(spatial)
                if total_norm == 0 or len(valid_sel) == 0:
                    score = 0.0
                else:
                    lasso_norm = np.linalg.norm(spatial[valid_sel])
                    score = float(lasso_norm / total_norm)

                best_chan = best_chan_map.get(vision_id_int, -1)
                results.append((ks_id, round(score, 4), best_chan))

            except Exception:
                continue

        results.sort(key=lambda x: x[1], reverse=True)
        return results


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------


class CellTracerDialog(QDialog):
    """
    Floating tool dialog.

    Left pane  — Photo underlay + electrode scatter (no EI heatmap).
                 User draws a freehand lasso with the left mouse button.
                 Electrodes inside the lasso are highlighted in gold.

    Right pane — Results table ranked by EI spatial energy concentration:
                 score = L2_norm(ei_amplitude[lasso_electrodes])
                           / L2_norm(ei_amplitude[all_electrodes])
                 Double-click a row to jump to that cluster in the main viewer.
    """

    #: Emitted when user double-clicks a result row
    cluster_selected = Signal(int)

    def __init__(self, parent=None, ei_panel: "EIPanel | None" = None):
        super().__init__(parent, Qt.Window)
        self.ei_panel = ei_panel
        self.setWindowTitle("Cell Tracer — EI Correlation Search")
        self.setMinimumSize(1050, 640)
        self.resize(1200, 720)

        # ── state ────────────────────────────────────────────────────────
        self._lasso_verts: list[tuple[float, float]] = []
        self._drawing = False
        self._selected_ch_indices: np.ndarray = np.array([], dtype=int)
        self._worker_thread: QThread | None = None
        self._worker: _CorrelationWorker | None = None

        # EI waveform overlay state
        # _cached_ch_pos is captured once in _initial_draw() so the lasso
        # hit-test and waveform overlay always use the exact same positions
        # array — immune to any mid-session reload of dm.vision_channel_positions.
        self._cached_ch_pos: np.ndarray | None = None
        self._ei_waveform_artists: list = []  # Line2D objects on self._ax
        self._ei_overlay_ks_id: int | None = None  # which cluster is displayed

        # ── UI ───────────────────────────────────────────────────────────
        self._build_ui()
        self._connect_canvas_events()
        self._initial_draw()

    # ===================================================================
    # UI construction
    # ===================================================================

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── top instruction bar ─────────────────────────────────────────
        info = QLabel(
            "Draw a freehand outline around a GFP cell (left-click drag). "
            "Release to search. Double-click a result to jump to that cluster."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        root.addWidget(info)

        # ── main splitter ───────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(6)

        # LEFT — canvas pane
        left_frame = QFrame()
        left_frame.setFrameShape(QFrame.StyledPanel)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        # canvas toolbar
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        self._clear_btn = QPushButton("Clear lasso")
        self._clear_btn.setFixedHeight(24)
        self._clear_btn.clicked.connect(self._clear_lasso)
        ctrl_row.addWidget(self._clear_btn)

        ctrl_row.addWidget(QLabel("Photo α:"))
        self._alpha_slider = QSlider(Qt.Horizontal)
        self._alpha_slider.setMinimum(10)
        self._alpha_slider.setMaximum(95)
        self._alpha_slider.setValue(50)
        self._alpha_slider.setFixedWidth(80)
        self._alpha_slider.valueChanged.connect(self._on_alpha_changed)
        ctrl_row.addWidget(self._alpha_slider)

        ctrl_row.addStretch()

        self._status_lbl = QLabel("No selection")
        self._status_lbl.setStyleSheet("color: #888; font-size: 11px;")
        ctrl_row.addWidget(self._status_lbl)

        left_layout.addLayout(ctrl_row)

        # matplotlib canvas
        self._fig = Figure(figsize=(5, 5), dpi=100, facecolor="#1e1e2e")
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self._canvas)

        splitter.addWidget(left_frame)

        # RIGHT — results pane
        right_frame = QFrame()
        right_frame.setFrameShape(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(6)

        res_hdr = QHBoxLayout()
        res_hdr_lbl = QLabel("Top matches")
        res_hdr_lbl.setStyleSheet("font-weight: bold; font-size: 12px;")
        res_hdr.addWidget(res_hdr_lbl)

        res_hdr.addStretch()

        res_hdr.addWidget(QLabel("Show:"))
        self._top_n_combo = QComboBox()
        self._top_n_combo.addItems(["10", "25", "50", "100", "All"])
        self._top_n_combo.setFixedWidth(60)
        self._top_n_combo.currentIndexChanged.connect(self._apply_top_n)
        res_hdr.addWidget(self._top_n_combo)

        right_layout.addLayout(res_hdr)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setFixedHeight(4)
        self._progress.setVisible(False)
        right_layout.addWidget(self._progress)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Cluster", "Score", "Best CH"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSortingEnabled(True)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet("""
            QTableWidget {
                background: #1a1a2e;
                alternate-background-color: #16213e;
                color: #e0e0e0;
                gridline-color: #2d2d44;
                font-size: 12px;
            }
            QHeaderView::section {
                background: #0f3460;
                color: #e0e0e0;
                font-weight: bold;
                padding: 4px;
                border: none;
            }
            QTableWidget::item:selected {
                background: #e94560;
                color: white;
            }
        """)
        self._table.doubleClicked.connect(self._on_table_double_click)
        self._table.itemSelectionChanged.connect(self._on_table_selection_changed)
        right_layout.addWidget(self._table)

        self._result_cache: list[tuple[int, float, int]] = []

        splitter.addWidget(right_frame)
        splitter.setSizes([620, 430])

        root.addWidget(splitter, stretch=1)

        # ── bottom row ──────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(90)
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    # ===================================================================
    # Canvas initialisation
    # ===================================================================

    def _initial_draw(self):
        """Draw photo underlay + electrode scatter into the local canvas.

        No EI heatmap is shown here — the heatmap lives in the EI panel and
        its coordinate frame cannot be guaranteed to match the photo/scatter
        space, which would make the lasso hit-test unreliable.  The user
        traces from the photo; correlation runs from the electrode selection.

        Coordinate-frame contract
        ─────────────────────────
        Both the photo imshow and the electrode scatter are drawn in the same
        micron space (electrode array coordinates).  After drawing we
        explicitly lock the axes limits to the electrode-array bounding box
        (with a small margin) so that:
          • matplotlib event xdata/ydata are in micron space
          • MplPath.contains_points(ch_pos) uses the same micron space
          • The photo is rendered with aspect='equal' so there is no
            visual-vs-data distortion that could mislead the user
        """
        self._fig.clear()
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor("#1e1e2e")
        self._fig.patch.set_facecolor("#1e1e2e")

        ei_panel = self.ei_panel
        if ei_panel is None:
            self._ax.text(
                0.5,
                0.5,
                "No EI data",
                ha="center",
                va="center",
                color="#aaa",
                transform=self._ax.transAxes,
            )
            self._canvas.draw()
            return

        # ── electrode positions (needed for axis limits) ─────────────────
        ch_pos = self._ch_pos()
        has_electrodes = ch_pos is not None and len(ch_pos) > 0

        # Cache positions for this dialog instance.  Every downstream
        # operation (lasso hit-test, EI overlay) uses self._cached_ch_pos
        # so they are guaranteed to be in the same coordinate frame.
        self._cached_ch_pos = ch_pos

        # Compute array bounding box in microns (used to lock axes limits)
        if has_electrodes:
            x_min, x_max = ch_pos[:, 0].min(), ch_pos[:, 0].max()
            y_min, y_max = ch_pos[:, 1].min(), ch_pos[:, 1].max()
            margin_x = max((x_max - x_min) * 0.05, 20.0)
            margin_y = max((y_max - y_min) * 0.05, 20.0)
            ax_xlim = (x_min - margin_x, x_max + margin_x)
            ax_ylim = (y_min - margin_y, y_max + margin_y)
        else:
            ax_xlim = ax_ylim = None

        # ── photo underlay ───────────────────────────────────────────────
        # aspect='equal' keeps the photo spatially honest so the user can
        # trust that what they see corresponds to what the lasso selects.
        alpha = self._alpha_slider.value() / 100.0
        if (
            ei_panel._overlay_image_rgba is not None
            and ei_panel._overlay_extent_um is not None
        ):
            xl, xr, yb, yt = ei_panel._overlay_extent_um
            self._ax.imshow(
                ei_panel._overlay_image_rgba,
                aspect="equal",
                origin="upper",
                extent=(xl, xr, yb, yt),
                alpha=alpha,
                interpolation="bilinear",
                zorder=0,
            )
        else:
            self._ax.text(
                0.5,
                0.97,
                "No photo loaded — load transform first",
                ha="center",
                va="top",
                color="#e94560",
                fontsize=9,
                transform=self._ax.transAxes,
            )

        # ── electrode scatter ────────────────────────────────────────────
        if has_electrodes:
            self._electrode_scatter = self._ax.scatter(
                ch_pos[:, 0],
                ch_pos[:, 1],
                s=6,
                c="#4488ff",
                alpha=0.5,
                zorder=2,
                linewidths=0,
            )
        else:
            self._electrode_scatter = None

        # ── lock axes to electrode bounding box ──────────────────────────
        # This is the critical step: it guarantees that lasso xdata/ydata
        # and ch_pos are in the same data-coordinate frame regardless of
        # how matplotlib auto-scaled due to the photo extent.
        if ax_xlim is not None:
            self._ax.set_xlim(ax_xlim)
            self._ax.set_ylim(ax_ylim)
            self._ax.set_aspect("equal", adjustable="datalim")

        # ── lasso line (empty at start) ──────────────────────────────────
        (self._lasso_line,) = self._ax.plot(
            [], [], "-", color="#00ffcc", linewidth=1.5, zorder=5
        )
        self._lasso_poly_patch = None
        self._sel_scatter = None

        self._ax.axis("off")
        self._fig.tight_layout(pad=0.2)
        self._canvas.draw()

        # fig.clear() above destroyed all previous artists.  Reset the list
        # so _clear_ei_overlay() doesn't try to .remove() dead objects.
        self._ei_waveform_artists = []
        self._ei_overlay_ks_id = None

    # ===================================================================
    # Mouse interaction — freehand lasso
    # ===================================================================

    def _connect_canvas_events(self):
        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._canvas.mpl_connect("button_release_event", self._on_release)

    def _on_press(self, event):
        if event.inaxes != self._ax or event.button != 1:
            return
        self._drawing = True
        self._lasso_verts = [(event.xdata, event.ydata)]
        self._lasso_line.set_data([], [])
        self._canvas.draw_idle()

    def _on_motion(self, event):
        if not self._drawing or event.inaxes != self._ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._lasso_verts.append((event.xdata, event.ydata))
        xs = [v[0] for v in self._lasso_verts]
        ys = [v[1] for v in self._lasso_verts]
        self._lasso_line.set_data(xs, ys)
        self._canvas.draw_idle()

    def _on_release(self, event):
        if not self._drawing or event.button != 1:
            return
        self._drawing = False
        if len(self._lasso_verts) < 3:
            return

        # Close the polygon
        self._lasso_verts.append(self._lasso_verts[0])
        self._redraw_lasso_polygon()
        self._compute_selected_electrodes()
        self._run_correlation()

    # ===================================================================
    # Lasso polygon drawing
    # ===================================================================

    def _redraw_lasso_polygon(self):
        """Replace freehand line with a filled polygon patch."""
        if self._lasso_poly_patch is not None:
            self._lasso_poly_patch.remove()
            self._lasso_poly_patch = None

        verts = np.array(self._lasso_verts)
        patch = MplPolygon(
            verts,
            closed=True,
            facecolor="#00ffcc",
            alpha=0.15,
            edgecolor="#00ffcc",
            linewidth=1.5,
            zorder=4,
        )
        self._ax.add_patch(patch)
        self._lasso_poly_patch = patch
        self._lasso_line.set_data([], [])  # hide raw line
        self._canvas.draw_idle()

    def _compute_selected_electrodes(self):
        """Find all electrode indices whose positions fall inside the lasso."""
        ch_pos = self._ch_pos()
        if ch_pos is None or len(ch_pos) == 0 or len(self._lasso_verts) < 3:
            self._selected_ch_indices = np.array([], dtype=int)
            self._status_lbl.setText("No electrodes selected")
            return

        poly_path = MplPath(np.array(self._lasso_verts))
        inside = poly_path.contains_points(ch_pos)
        self._selected_ch_indices = np.where(inside)[0]

        n = len(self._selected_ch_indices)
        self._status_lbl.setText(f"{n} electrode{'s' if n != 1 else ''} selected")

        # Highlight selected electrodes
        self._redraw_electrode_highlights()

    def _redraw_electrode_highlights(self):
        ch_pos = self._ch_pos()
        if ch_pos is None:
            return

        if self._sel_scatter is not None:
            self._sel_scatter.remove()
            self._sel_scatter = None

        if len(self._selected_ch_indices) > 0:
            sel_pos = ch_pos[self._selected_ch_indices]
            self._sel_scatter = self._ax.scatter(
                sel_pos[:, 0],
                sel_pos[:, 1],
                s=28,
                c="#ffd700",
                alpha=0.9,
                zorder=6,
                linewidths=1.2,
                edgecolors="#ffffff",
                marker="o",
            )

        self._canvas.draw_idle()

    def _clear_lasso(self):
        self._lasso_verts = []
        self._selected_ch_indices = np.array([], dtype=int)
        self._lasso_line.set_data([], [])

        if self._lasso_poly_patch is not None:
            self._lasso_poly_patch.remove()
            self._lasso_poly_patch = None

        if self._sel_scatter is not None:
            self._sel_scatter.remove()
            self._sel_scatter = None

        self._clear_ei_overlay()
        self._status_lbl.setText("No selection")
        self._table.setRowCount(0)
        self._result_cache = []
        self._canvas.draw_idle()

    # ===================================================================
    # Correlation computation (background thread)
    # ===================================================================

    def _run_correlation(self):
        if len(self._selected_ch_indices) == 0:
            return

        dm = self._get_dm()
        if dm is None or dm.vision_eis is None:
            self._status_lbl.setText("No Vision EIs loaded")
            return

        # Cancel any previous run
        if self._worker_thread is not None and self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait(500)

        self._progress.setVisible(True)
        self._table.setRowCount(0)

        self._worker = _CorrelationWorker(dm, self._selected_ch_indices.copy())
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_results)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker_thread.start()

    def _on_results(self, results: list[tuple[int, float, int]]):
        self._progress.setVisible(False)
        self._result_cache = results
        self._apply_top_n()

    def _on_error(self, msg: str):
        self._progress.setVisible(False)
        self._status_lbl.setText(f"Error: {msg}")

    def _apply_top_n(self):
        """Slice _result_cache to the selected top-N and populate table."""
        text = self._top_n_combo.currentText()
        if text == "All":
            rows = self._result_cache
        else:
            n = int(text)
            rows = self._result_cache[:n]

        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(rows))

        for i, (cid, score, best_chan) in enumerate(rows):
            self._table.setItem(i, 0, self._make_cell(str(cid)))
            score_item = self._make_cell(f"{score:.4f}")
            # colour-code by score: green → yellow → red via HSV hue
            h = int(score * 120)  # 0 = red, 120 = green
            score_item.setForeground(QColor.fromHsv(h, 200, 220))
            self._table.setItem(i, 1, score_item)
            chan_str = str(best_chan) if best_chan >= 0 else "—"
            self._table.setItem(i, 2, self._make_cell(chan_str))

        self._table.setSortingEnabled(True)
        n_total = len(self._result_cache)
        self._status_lbl.setText(
            f"{len(self._selected_ch_indices)} electrodes — "
            f"showing {len(rows)} / {n_total} clusters"
        )

    @staticmethod
    def _make_cell(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        return item

    # ===================================================================
    # Table interaction
    # ===================================================================

    def _on_table_double_click(self, index):
        row = index.row()
        cid_item = self._table.item(row, 0)
        if cid_item is None:
            return
        try:
            cid = int(cid_item.text())
        except ValueError:
            return
        self.cluster_selected.emit(cid)
        # Also reflect in the EI panel directly if accessible
        if self.ei_panel is not None and hasattr(self.ei_panel, "main_window"):
            mw = self.ei_panel.main_window
            if hasattr(mw, "_select_cluster_by_id"):
                mw._select_cluster_by_id(cid)

    def _on_table_selection_changed(self):
        """Single-click: paint the selected cluster's EI waveforms on the canvas."""
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        cid_item = self._table.item(rows[0].row(), 0)
        if cid_item is None:
            return
        try:
            ks_id = int(cid_item.text())
        except ValueError:
            return
        self._draw_ei_overlay(ks_id)

    # ===================================================================
    # EI waveform overlay
    # ===================================================================

    def _clear_ei_overlay(self):
        """Remove all EI waveform Line2D artists from the canvas."""
        for artist in self._ei_waveform_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._ei_waveform_artists = []
        self._ei_overlay_ks_id = None

    def _draw_ei_overlay(self, ks_id: int):
        """
        Fetch the EI for ks_id and paint its waveforms directly onto self._ax
        in micron space, co-registered with the photo underlay and electrode
        scatter.

        Uses self._cached_ch_pos (captured at _initial_draw time) so the
        positions are guaranteed to match the lasso coordinate frame.

        Single-click calls this; alpha slider calls _initial_draw then this.
        """
        dm = self._get_dm()
        if dm is None or dm.vision_eis is None:
            return

        ch_pos = self._cached_ch_pos
        if ch_pos is None:
            return

        # Translate ks_id → vision_id (LAW 1)
        is_vision_only = getattr(dm, "is_vision_only", False)
        vid = ks_id if is_vision_only else ks_id + 1

        entry = dm.vision_eis.get(int(vid))
        if entry is None:
            return
        ei_arr = getattr(entry, "ei", None)
        if not isinstance(ei_arr, np.ndarray) or ei_arr.ndim != 2:
            return

        # Guard: EI channel count must not exceed positions length
        n_ch = min(ei_arr.shape[0], len(ch_pos))
        if n_ch == 0:
            return
        ei_arr = ei_arr[:n_ch]
        positions = ch_pos[:n_ch]

        # Best channel from result cache for ref highlight
        best_chan: int | None = None
        for cid, _score, bchan in self._result_cache:
            if cid == ks_id:
                best_chan = bchan if bchan >= 0 else None
                break

        # Clear previous overlay then draw the new one
        self._clear_ei_overlay()

        from ...analysis.analysis_core import plot_ei_waveforms

        artists = plot_ei_waveforms(
            ei_arr,
            positions,
            ref_channel=best_chan,
            ax=self._ax,
            colors="#00e5ff",
            alpha=0.75,
            linewidth=0.6,
            # box geometry auto-derived from array spacing in plot_ei_waveforms
        )

        self._ei_waveform_artists = artists
        self._ei_overlay_ks_id = ks_id
        self._canvas.draw_idle()

    # ===================================================================
    # Alpha slider
    # ===================================================================

    def _on_alpha_changed(self, value: int):
        """Redraws only the photo imshow with updated alpha."""
        # Remember which cluster was overlaid before the full redraw wipes it.
        prev_ks_id = self._ei_overlay_ks_id

        # Quick path: just re-run initial draw (fast since it's a local canvas)
        self._initial_draw()
        # Restore lasso state on top
        if self._lasso_verts:
            self._redraw_lasso_polygon()
            self._redraw_electrode_highlights()

        # Restore EI overlay if one was active before the redraw.
        if prev_ks_id is not None:
            self._draw_ei_overlay(prev_ks_id)

    # ===================================================================
    # Helpers
    # ===================================================================

    def _ch_pos(self) -> np.ndarray | None:
        dm = self._get_dm()
        if dm is None:
            return None
        if dm.vision_channel_positions is not None:
            return dm.vision_channel_positions
        return dm.channel_positions

    def _get_dm(self):
        if self.ei_panel is None:
            return None
        mw = getattr(self.ei_panel, "main_window", None)
        if mw is None:
            return None
        return getattr(mw, "data_manager", None)

    def closeEvent(self, event):
        if self._worker_thread is not None and self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait(300)
        super().closeEvent(event)
