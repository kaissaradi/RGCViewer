"""
array_calibration_panel.py
==========================
A Qt dialog for mapping an array photo to electrode coordinates.
"""

from __future__ import annotations

import json
import logging
import numpy as np
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, List, Tuple

import pyqtgraph as pg
from qtpy.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QSplitter,
    QHeaderView,
    QMessageBox,
    QWidget,
    QGroupBox,
    QCheckBox,
    QAbstractItemView,
    QStatusBar,
)
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QFont

from .. import recent_paths

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Coordinate transform
# ─────────────────────────────────────────────────────────────────────────────


def fit_transform(pixel_pts: np.ndarray, micron_pts: np.ndarray) -> Optional[dict]:
    """Fit per-axis linear transform: pixel = scale * micron + offset."""
    if len(pixel_pts) < 2:
        return None
    pixel_pts = np.array(pixel_pts, dtype=float)
    micron_pts = np.array(micron_pts, dtype=float)

    def fit_axis(micro, pix):
        A = np.column_stack([micro, np.ones(len(micro))])
        res, _, _, _ = np.linalg.lstsq(A, pix, rcond=None)
        rmse = float(np.sqrt(np.mean((pix - (res[0] * micro + res[1])) ** 2)))
        return float(res[0]), float(res[1]), rmse

    sx, ox, rmse_x = fit_axis(micron_pts[:, 0], pixel_pts[:, 0])
    sy, oy, rmse_y = fit_axis(micron_pts[:, 1], pixel_pts[:, 1])
    return dict(
        scale_x=sx,
        offset_x=ox,
        scale_y=sy,
        offset_y=oy,
        rmse_x_px=rmse_x,
        rmse_y_px=rmse_y,
        n_points=len(pixel_pts),
    )


def microns_to_pixels(xy_microns: np.ndarray, t: dict) -> np.ndarray:
    scalar = np.ndim(xy_microns) == 1
    xy = np.atleast_2d(xy_microns).astype(float)
    result = np.column_stack(
        [
            t["scale_x"] * xy[:, 0] + t["offset_x"],
            t["scale_y"] * xy[:, 1] + t["offset_y"],
        ]
    )
    return result[0] if scalar else result


# ─────────────────────────────────────────────────────────────────────────────
# Custom pyqtgraph ViewBox
# ─────────────────────────────────────────────────────────────────────────────


class ClickableViewBox(pg.ViewBox):
    """ViewBox that emits image pixel coords on left-click."""

    sigImageClicked = Signal(float, float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._click_mode = False

    def set_click_mode(self, enabled: bool):
        self._click_mode = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def mousePressEvent(self, ev):
        if self._click_mode and ev.button() == Qt.LeftButton:
            pos = self.mapSceneToView(ev.scenePos())
            self.sigImageClicked.emit(pos.x(), pos.y())
            ev.accept()
        else:
            super().mousePressEvent(ev)


# ─────────────────────────────────────────────────────────────────────────────
# The main dialog
# ─────────────────────────────────────────────────────────────────────────────


class ArrayCalibrationDialog(QDialog):
    transformSaved = Signal(str)

    def __init__(self, parent=None, data_manager=None):
        super().__init__(parent)
        self.setWindowTitle("Map Image to Array")
        self.resize(1100, 700)
        self.setModal(False)

        self.dm = data_manager

        self._image_array: Optional[np.ndarray] = None
        self._image_path: Optional[Path] = None
        self._img_h = 0
        self._img_w = 0
        self._adding_point = False

        self._pixel_pts: List[Tuple[float, float]] = []
        self._micron_pts: List[Tuple[float, float]] = []
        self._elec_nums: List[int] = []

        self._marker_items: List[pg.ScatterPlotItem] = []
        self._label_items: List[pg.TextItem] = []
        self._preview_scatter: Optional[pg.ScatterPlotItem] = None

        self._setup_ui()
        self._connect_signals()
        self._refresh_state()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        toolbar = QHBoxLayout()
        self.btn_load_image = QPushButton("📂  Load Image")
        self.btn_load_image.setMinimumHeight(30)
        toolbar.addWidget(self.btn_load_image)

        self.btn_load_transform = QPushButton("📋  Load Existing Transform")
        self.btn_load_transform.setMinimumHeight(30)
        toolbar.addWidget(self.btn_load_transform)

        toolbar.addStretch()

        self.btn_add_point = QPushButton("➕  Add Point")
        self.btn_add_point.setCheckable(True)
        self.btn_add_point.setMinimumHeight(30)
        self.btn_add_point.setStyleSheet(
            "QPushButton:checked { background:#2a6f2a; border:1px solid #5aaf5a; }"
        )
        toolbar.addWidget(self.btn_add_point)

        self.btn_clear_all = QPushButton("🗑  Clear All")
        self.btn_clear_all.setMinimumHeight(30)
        toolbar.addWidget(self.btn_clear_all)

        root.addLayout(toolbar)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, stretch=1)

        self.gfx = pg.GraphicsLayoutWidget()
        self.gfx.setBackground("#1a1a1a")
        self.vb = ClickableViewBox(lockAspect=True, invertY=True)
        self.plot = self.gfx.addPlot(viewBox=self.vb)
        self.plot.hideAxis("bottom")
        self.plot.hideAxis("left")

        self.img_item = pg.ImageItem()
        self.plot.addItem(self.img_item)

        self.placeholder = pg.TextItem(
            "Click  'Load Image'  to begin", color=(120, 120, 120), anchor=(0.5, 0.5)
        )
        self.placeholder.setFont(QFont("Arial", 14))
        self.plot.addItem(self.placeholder)

        splitter.addWidget(self.gfx)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(6)

        pts_group = QGroupBox("Calibration Points")
        pts_layout = QVBoxLayout(pts_group)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Pixel X", "Pixel Y", "Electrode #"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setMaximumHeight(200)
        pts_layout.addWidget(self.table)

        tbl_btn_row = QHBoxLayout()
        self.btn_delete_row = QPushButton("Delete Selected")
        tbl_btn_row.addStretch()
        tbl_btn_row.addWidget(self.btn_delete_row)
        pts_layout.addLayout(tbl_btn_row)
        right_layout.addWidget(pts_group)

        quality_group = QGroupBox("Transform Quality")
        q_layout = QVBoxLayout(quality_group)
        self.lbl_rmse = QLabel("RMSE: — (need ≥ 2 points)")
        q_layout.addWidget(self.lbl_rmse)
        self.lbl_npts = QLabel("Points: 0")
        q_layout.addWidget(self.lbl_npts)
        self.chk_show_preview = QCheckBox("Show all-channel preview overlay")
        self.chk_show_preview.setChecked(True)
        q_layout.addWidget(self.chk_show_preview)
        right_layout.addWidget(quality_group)

        tips = QLabel(
            "<b>Tips:</b><br>• Click <i>Add Point</i>, then click an electrode<br>• Enter its electrode number when prompted<br>• Add 3+ points for best accuracy"
        )
        tips.setWordWrap(True)
        right_layout.addWidget(tips)
        right_layout.addStretch()

        save_group = QGroupBox("Save Transform")
        save_layout = QVBoxLayout(save_group)
        self.lbl_save_path = QLabel("No path set")
        save_layout.addWidget(self.lbl_save_path)
        save_btn_row = QHBoxLayout()
        self.btn_set_save_path = QPushButton("Set Path…")
        self.btn_save = QPushButton("💾  Save Transform")
        self.btn_save.setEnabled(False)
        save_btn_row.addWidget(self.btn_set_save_path)
        save_btn_row.addWidget(self.btn_save, stretch=1)
        save_layout.addLayout(save_btn_row)
        right_layout.addWidget(save_group)

        right_panel.setMinimumWidth(280)
        right_panel.setMaximumWidth(340)
        splitter.addWidget(right_panel)
        splitter.setSizes([760, 300])

        self.status = QStatusBar()
        root.addWidget(self.status)
        self._set_status("Load an image to begin")

        self._save_path: Optional[Path] = None
        if self.dm is not None and hasattr(self.dm, "kilosort_dir"):
            default = (
                Path(self.dm.kilosort_dir).parent
                / "transforms"
                / "array_transform.json"
            )
            self._set_save_path(default)

    def _connect_signals(self):
        self.btn_load_image.clicked.connect(self._on_load_image)
        self.btn_load_transform.clicked.connect(self._on_load_transform)
        self.btn_add_point.toggled.connect(self._on_add_point_toggled)
        self.btn_clear_all.clicked.connect(self._on_clear_all)
        self.btn_delete_row.clicked.connect(self._on_delete_selected_row)
        self.btn_set_save_path.clicked.connect(self._on_set_save_path)
        self.btn_save.clicked.connect(self._on_save)
        self.chk_show_preview.toggled.connect(self._refresh_preview)
        self.vb.sigImageClicked.connect(self._on_image_clicked)
        self.table.itemChanged.connect(self._on_table_item_changed)

    def _main_window(self):
        """Adapter for recent_paths, which reads ``.data_manager``.

        This dialog holds the manager as ``self.dm`` and has no main-window
        reference, so hand over a stand-in exposing the expected attribute
        rather than widening recent_paths' contract for one caller.
        """
        return SimpleNamespace(data_manager=self.dm)

    def _on_load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Array Image",
            recent_paths.last_dir(self._main_window(), "array_image"),
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All Files (*)",
        )
        if not path:
            return
        recent_paths.remember_dir(path, "array_image")
        self._load_image_from_path(Path(path))

    def _load_image_from_path(self, path: Path):
        try:
            from PIL import Image

            img = Image.open(path).convert("RGB")
            self._image_array = np.array(img, dtype=np.uint8)
        except Exception as e:
            QMessageBox.critical(self, "Image Load Error", str(e))
            return

        self._image_path = path
        self._img_h, self._img_w = self._image_array.shape[:2]
        img_pg = self._image_array.transpose(1, 0, 2)
        self.img_item.setImage(img_pg)
        self.placeholder.setVisible(False)
        self.plot.autoRange()

        if self._save_path is None:
            self._set_save_path(path.parent / "array_transform.json")

        self._set_status(f"Loaded: {path.name}  ({self._img_w}×{self._img_h} px)")
        self._refresh_state()

    def _on_load_transform(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Transform",
            recent_paths.last_dir(self._main_window(), "transform"),
            "JSON (*.json);;All Files (*)",
        )
        if not path:
            return
        recent_paths.remember_dir(path, "transform")
        try:
            with open(path) as f:
                data = json.load(f)
            pixel_pts = data.get("pixel_pts", [])
            micron_pts = data.get("micron_pts", [])
            elec_nums = data.get("electrode_nums_used", [None] * len(pixel_pts))
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return

        self._on_clear_all(confirm=False)
        for ppt, mpt, en in zip(pixel_pts, micron_pts, elec_nums):
            self._add_calibration_point(ppt[0], ppt[1], en or 0, mpt[0], mpt[1])

        self._set_save_path(Path(path))
        self._set_status(f"Loaded transform from {Path(path).name}")
        self._refresh_state()

    def _on_add_point_toggled(self, checked: bool):
        self._adding_point = checked
        self.vb.set_click_mode(checked)

    def _on_image_clicked(self, px_x: float, px_y: float):
        if not self._adding_point or self._image_array is None:
            return
        px_x = max(0.0, min(float(self._img_w - 1), px_x))
        px_y = max(0.0, min(float(self._img_h - 1), px_y))

        elec_num = self._ask_electrode_number(px_x, px_y)
        if elec_num is None:
            return

        um_x, um_y = self._electrode_to_microns(elec_num)
        if um_x is None:
            QMessageBox.warning(
                self,
                "Electrode Not Found",
                f"Electrode {elec_num} not found in channel_map.",
            )
            return

        self._add_calibration_point(px_x, px_y, elec_num, um_x, um_y)
        self._refresh_state()

    def _ask_electrode_number(self, px_x: float, px_y: float) -> Optional[int]:
        from qtpy.QtWidgets import QInputDialog

        val, ok = QInputDialog.getInt(
            self,
            "Electrode Number",
            f"Enter electrode number for point ({px_x:.0f}, {px_y:.0f}):",
            value=1,
            min=0,
            max=10000,
        )
        return val if ok else None

    def _electrode_to_microns(
        self, elec_num: int
    ) -> Tuple[Optional[float], Optional[float]]:
        if self.dm is None:
            return None, None
        try:
            chan_map = np.array(self.dm.channel_map).flatten()
            chan_pos = np.array(self.dm.channel_positions)
            matches = np.where(chan_map == elec_num)[0]
            if len(matches) == 0:
                return None, None
            idx = matches[0]
            return float(chan_pos[idx, 0]), float(chan_pos[idx, 1])
        except Exception as e:
            logger.warning(f"Could not look up electrode {elec_num}: {e}")
            return None, None

    def _add_calibration_point(
        self, px_x: float, px_y: float, elec_num: int, um_x: float, um_y: float
    ):
        self._pixel_pts.append((px_x, px_y))
        self._micron_pts.append((um_x, um_y))
        self._elec_nums.append(elec_num)

        marker = pg.ScatterPlotItem(
            [px_x],
            [px_y],
            symbol="o",
            size=14,
            brush=pg.mkBrush(255, 80, 80, 200),
            pen=pg.mkPen("w", width=1.5),
        )
        self.plot.addItem(marker)
        self._marker_items.append(marker)

        label = pg.TextItem(str(elec_num), color=(255, 255, 100), anchor=(0, 1))
        label.setFont(QFont("Arial", 9, QFont.Bold))
        label.setPos(px_x + 8, px_y - 8)
        self.plot.addItem(label)
        self._label_items.append(label)

        self.table.blockSignals(True)
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(f"{px_x:.1f}"))
        self.table.setItem(row, 1, QTableWidgetItem(f"{px_y:.1f}"))
        elec_item = QTableWidgetItem(str(elec_num))
        elec_item.setForeground(QColor(255, 220, 80))
        self.table.setItem(row, 2, elec_item)
        for col in (0, 1):
            self.table.item(row, col).setFlags(
                self.table.item(row, col).flags() & ~Qt.ItemIsEditable
            )
        self.table.blockSignals(False)

    def _on_table_item_changed(self, item: QTableWidgetItem):
        if item.column() != 2:
            return
        row = item.row()
        try:
            new_num = int(item.text())
        except ValueError:
            item.setText(str(self._elec_nums[row]))
            return

        um_x, um_y = self._electrode_to_microns(new_num)
        if um_x is None:
            QMessageBox.warning(
                self, "Not Found", f"Electrode {new_num} not in channel_map."
            )
            item.setText(str(self._elec_nums[row]))
            return

        self._elec_nums[row] = new_num
        self._micron_pts[row] = (um_x, um_y)
        self._label_items[row].setText(str(new_num))
        self._refresh_state()

    def _on_delete_selected_row(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.plot.removeItem(self._marker_items[row])
            self.plot.removeItem(self._label_items[row])
            del self._marker_items[row]
            del self._label_items[row]
            del self._pixel_pts[row]
            del self._micron_pts[row]
            del self._elec_nums[row]
            self.table.removeRow(row)
        self._refresh_state()

    def _on_clear_all(self, confirm=True):
        if confirm and self._pixel_pts:
            if (
                QMessageBox.question(
                    self,
                    "Clear All",
                    "Remove all calibration points?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                != QMessageBox.Yes
            ):
                return
        for item in self._marker_items + self._label_items:
            self.plot.removeItem(item)
        self._marker_items.clear()
        self._label_items.clear()
        self._pixel_pts.clear()
        self._micron_pts.clear()
        self._elec_nums.clear()
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        self.table.blockSignals(False)
        self._remove_preview()
        self._refresh_state()

    def _refresh_preview(self):
        self._remove_preview()
        if (
            not self.chk_show_preview.isChecked()
            or len(self._pixel_pts) < 2
            or self.dm is None
        ):
            return
        t = fit_transform(np.array(self._pixel_pts), np.array(self._micron_pts))
        if t is None:
            return
        try:
            chan_pos = np.array(self.dm.channel_positions)
        except Exception:
            return
        px = microns_to_pixels(chan_pos, t)
        self._preview_scatter = pg.ScatterPlotItem(
            px[:, 0],
            px[:, 1],
            symbol="o",
            size=6,
            brush=pg.mkBrush(0, 200, 255, 140),
            pen=pg.mkPen(None),
        )
        self._preview_scatter.setZValue(-5)
        self.plot.addItem(self._preview_scatter)

    def _remove_preview(self):
        if self._preview_scatter is not None:
            self.plot.removeItem(self._preview_scatter)
            self._preview_scatter = None

    def _refresh_state(self):
        n = len(self._pixel_pts)
        self.lbl_npts.setText(f"Points: {n}")
        if n >= 2:
            t = fit_transform(np.array(self._pixel_pts), np.array(self._micron_pts))
            if t:
                rx, ry = t["rmse_x_px"], t["rmse_y_px"]
                color = (
                    "#5af55a"
                    if max(rx, ry) < 5
                    else "#f5a84a" if max(rx, ry) < 15 else "#f55a5a"
                )
                self.lbl_rmse.setText(
                    f"<span style='color:{color}'>RMSE X: {rx:.2f} px | Y: {ry:.2f} px</span>"
                )
        else:
            self.lbl_rmse.setText("RMSE: — (need ≥ 2 points)")

        self.btn_save.setEnabled(n >= 2 and self._save_path is not None)
        self.btn_clear_all.setEnabled(n > 0)
        self.btn_delete_row.setEnabled(n > 0)
        self.btn_add_point.setEnabled(self._image_array is not None)
        self._refresh_preview()

    def _on_set_save_path(self):
        default = (
            str(self._save_path)
            if self._save_path
            else recent_paths.suggested_path(
                self._main_window(), "transform", "array_transform.json"
            )
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Transform As", default, "JSON (*.json);;All Files (*)"
        )
        if path:
            recent_paths.remember_dir(path, "transform")
            self._set_save_path(Path(path))

    def _set_save_path(self, path: Path):
        self._save_path = path
        self.lbl_save_path.setText(str(path))
        self._refresh_state()

    def _on_save(self):
        if not self._save_path or len(self._pixel_pts) < 2:
            return
        t = fit_transform(np.array(self._pixel_pts), np.array(self._micron_pts))
        if t is None:
            return

        self._save_path.parent.mkdir(parents=True, exist_ok=True)

        # --- NEW: Copy the image to the transforms folder ---
        if self._image_path and self._image_path.exists():
            dest_img_path = self._save_path.parent / self._image_path.name
            if self._image_path != dest_img_path:
                shutil.copy2(self._image_path, dest_img_path)
        # ----------------------------------------------------

        data = {
            "image_file": self._image_path.name if self._image_path else None,
            "image_size_wh": [self._img_w, self._img_h],
            "n_channels": int(len(self.dm.channel_positions)) if self.dm else None,
            "electrode_nums_used": self._elec_nums,
            "pixel_pts": [list(p) for p in self._pixel_pts],
            "micron_pts": [list(p) for p in self._micron_pts],
            **t,
        }
        try:
            with open(self._save_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))
            return

        self._set_status(f"✓ Saved → {self._save_path}")
        self.transformSaved.emit(str(self._save_path))
        QMessageBox.information(
            self,
            "Saved",
            f"Transform saved to:\n{self._save_path}\n\nRMSE X: {t['rmse_x_px']:.2f} px Y: {t['rmse_y_px']:.2f} px",
        )

    def _set_status(self, msg: str):
        self.status.showMessage(msg)
