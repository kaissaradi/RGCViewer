import pandas as pd
from PyQt5.QtGui import QColor, QPainter, QPen
from qtpy.QtCore import QAbstractTableModel, Qt, QModelIndex, Signal, QRect, QEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtWidgets import QTableView, QStyledItemDelegate, QTreeView, QStyleOptionViewItem

from ..theme import DARK_COLORS

import logging
logger = logging.getLogger(__name__)


class CustomTableView(QTableView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PandasModel(QAbstractTableModel):
    """A model to interface a pandas DataFrame with a QTableView."""

    def __init__(self, dataframe: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.set_dataframe(dataframe)

    def set_dataframe(self, dataframe):
        self.beginResetModel()
        self._dataframe = dataframe
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self._dataframe)

    def columnCount(self, parent=QModelIndex()):
        return len(self._dataframe.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._dataframe.iloc[index.row(), index.column()]
            if pd.isna(value):
                return ""
            if isinstance(value, float):
                return f"{value:.2f}"
            return str(value)
        return None

    def headerData(self, section, orientation, role):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._dataframe.columns[section])
            if orientation == Qt.Orientation.Vertical:
                return str(self._dataframe.index[section])
        return None

    def sort(self, column, order):
        self.layoutAboutToBeChanged.emit()
        colname = self._dataframe.columns[column]
        self._dataframe.sort_values(colname, ascending=(
            order == Qt.SortOrder.AscendingOrder), inplace=True)
        self._dataframe.reset_index(inplace=True, drop=True)
        if hasattr(self, "refresh_view"):
            self.refresh_view()
        self.layoutChanged.emit()


class MplCanvas(FigureCanvas):
    """A canvas that displays a matplotlib figure."""
    clicked = Signal()

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(
            figsize=(width, height),
            dpi=dpi,
            facecolor=DARK_COLORS['bg_panel'])
        super().__init__(self.fig)
        self.setCursor(Qt.PointingHandCursor)
        self.mpl_connect('button_press_event', self._on_click)

    def restyle(self, colors):
        """Updates the canvas background based on the provided color scheme."""
        self.fig.patch.set_facecolor(colors['bg_panel'])
        self.draw()

    def _on_click(self, _event):
        """Handle matplotlib mouse click events."""
        self.clicked.emit()


class HighlightStatusPandasModel(PandasModel):
    """Optimized model with role-based caching for faster scrolling."""

    STATUS_COLORS = {
        "Clean": DARK_COLORS["status_good_text"],
        "Edge": DARK_COLORS["status_mua_text"],
        "Duplicate": DARK_COLORS["status_noise_text"],
        "Noise": DARK_COLORS["status_noise_text"],
        "Unsure": DARK_COLORS["status_unsort_text"],
        "Original": DARK_COLORS["text_primary"],
    }

    def __init__(self, dataframe: pd.DataFrame, parent=None):
        super().__init__(dataframe, parent)
        self._background_cache = {}
        self._foreground_cache = {}
        self._display_cache = {}

    def update_colors(self, colors):
        """Updates status colors based on the current theme."""
        self.STATUS_COLORS = {
            "Clean": colors.get("status_good_text", DARK_COLORS["status_good_text"]),
            "Edge": colors.get("status_mua_text", DARK_COLORS["status_mua_text"]),
            "Duplicate": colors.get("status_noise_text", DARK_COLORS["status_noise_text"]),
            "Noise": colors.get("status_noise_text", DARK_COLORS["status_noise_text"]),
            "Unsure": colors.get("status_unsort_text", DARK_COLORS["status_unsort_text"]),
            "Original": colors.get("text_primary", DARK_COLORS["text_primary"]),
        }
        self.refresh_view()

    def refresh_view(self, row_indices=None):
        """Invalidate cache on data change."""
        if row_indices is None:
            self._background_cache.clear()
            self._foreground_cache.clear()
            self._display_cache.clear()
        else:
            for row in row_indices:
                for cache in [self._background_cache, self._foreground_cache, self._display_cache]:
                    keys_to_remove = [k for k in cache if k[0] == row]
                    for k in keys_to_remove:
                        cache.pop(k, None)

        if row_indices is None:
            row_indices = range(len(self._dataframe))
        row_indices = list(row_indices)
        if not row_indices:
            return
        top_left = self.index(min(row_indices), 0)
        bottom_right = self.index(max(row_indices), self.columnCount() - 1)
        self.dataChanged.emit(
            top_left, bottom_right,
            [Qt.BackgroundRole, Qt.ForegroundRole, Qt.DisplayRole])

    def data(self, index, role=Qt.DisplayRole):
        """Return cached data if available, otherwise compute and cache."""
        if not index.isValid():
            return None

        row = index.row()
        cache_key = (row, index.column())

        if role == Qt.BackgroundRole:
            if cache_key in self._background_cache:
                return self._background_cache[cache_key]
        elif role == Qt.ForegroundRole:
            if cache_key in self._foreground_cache:
                return self._foreground_cache[cache_key]
        elif role == Qt.DisplayRole:
            if cache_key in self._display_cache:
                return self._display_cache[cache_key]

        result = self._compute_data(index, role)

        if role == Qt.BackgroundRole:
            self._background_cache[cache_key] = result
        elif role == Qt.ForegroundRole:
            self._foreground_cache[cache_key] = result
        elif role == Qt.DisplayRole:
            self._display_cache[cache_key] = result

        return result

    def _compute_data(self, index, role):
        """Original data() logic moved here."""
        value = super().data(index, role)
        if not index.isValid():
            return value

        try:
            if 'status' not in self._dataframe.columns:
                return value

            col_name = self._dataframe.columns[index.column()]
            status_col_idx = self._dataframe.columns.get_loc('status')
            status_value = str(self._dataframe.iloc[index.row(), status_col_idx])

            if role == Qt.ForegroundRole:
                if col_name == 'status':
                    color = self.STATUS_COLORS.get(status_value, DARK_COLORS["text_primary"])
                    return QColor(color)
                return QColor(self.STATUS_COLORS.get("Original", DARK_COLORS["text_primary"]))

            if role == Qt.BackgroundRole:
                return None

        except Exception:
            logger.exception("HighlightStatusPandasModel.data error")

        return value


class ClusterTreeDelegate(QStyledItemDelegate):
    """
    Paints +/- expand toggles and hierarchy guide lines for folder rows in the
    cluster tree. Qt stylesheet ::branch rules are unreliable across platforms,
    so this delegate draws directly in the tree indent/branch area.
    """

    TOGGLE_SIZE = 14

    def __init__(self, tree_view: QTreeView, colors=None):
        super().__init__(tree_view)
        self._tree = tree_view
        self._colors = dict(colors or DARK_COLORS)

    def update_colors(self, colors):
        self._colors = dict(colors)

    @staticmethod
    def _depth(index: QModelIndex) -> int:
        depth = 0
        parent = index.parent()
        while parent.isValid():
            depth += 1
            parent = parent.parent()
        return depth

    def _ancestor_at_level(self, index: QModelIndex, level: int) -> QModelIndex:
        depth = self._depth(index)
        anc = index
        for _ in range(depth - level):
            anc = anc.parent()
        return anc

    def _toggle_rect(self, index: QModelIndex, row_rect: QRect):
        if not index.model().hasChildren(index):
            return None
        depth = self._depth(index)
        indent = self._tree.indentation()
        x = depth * indent + (indent - self.TOGGLE_SIZE) // 2
        y = row_rect.center().y() - self.TOGGLE_SIZE // 2
        return QRect(x, y, self.TOGGLE_SIZE, self.TOGGLE_SIZE)

    def _draw_guides(self, painter: QPainter, index: QModelIndex, row_rect: QRect):
        depth = self._depth(index)
        if depth == 0:
            return

        indent = self._tree.indentation()
        model = index.model()
        line = QColor(self._colors['border_default'])
        painter.setPen(QPen(line, 1))

        for level in range(depth):
            anc = self._ancestor_at_level(index, level)
            parent = anc.parent()
            last_child = anc.row() == model.rowCount(parent) - 1
            x = level * indent + indent // 2
            if last_child:
                painter.drawLine(x, row_rect.top(), x, row_rect.center().y())
            else:
                painter.drawLine(x, row_rect.top(), x, row_rect.bottom())

        parent_level = depth - 1
        x = parent_level * indent + indent // 2
        painter.drawLine(x, row_rect.center().y(), row_rect.left(), row_rect.center().y())

    def _draw_toggle(self, painter: QPainter, rect: QRect, expanded: bool):
        accent = QColor(self._colors['text_secondary'])
        painter.setPen(QPen(accent, 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect, 2, 2)
        cx, cy = rect.center().x(), rect.center().y()
        painter.setPen(QPen(accent, 1.5))
        painter.drawLine(cx - 3, cy, cx + 3, cy)
        if not expanded:
            painter.drawLine(cx, cy - 3, cx, cy + 3)

    def paint(self, painter, option, index):
        tree = self._tree
        row_rect = tree.visualRect(index)
        if row_rect.isValid():
            painter.save()
            painter.setRenderHint(QPainter.Antialiasing)
            self._draw_guides(painter, index, row_rect)
            toggle = self._toggle_rect(index, row_rect)
            if toggle is not None:
                self._draw_toggle(painter, toggle, tree.isExpanded(index))
            painter.restore()
        super().paint(painter, option, index)

    def editorEvent(self, event, model, option, index):
        if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            # In headless/test environments visualRect may return a null rect.
            # Fall back to a synthetic row_rect derived from the event position
            # so that _toggle_rect produces coords in the same space as event.pos().
            row_rect = self._tree.visualRect(index)
            if not row_rect.isValid() or row_rect.isEmpty():
                pos = event.pos() if not hasattr(event, "position") else event.position().toPoint()
                row_rect = QRect(0, pos.y() - 14, 300, 28)
            toggle = self._toggle_rect(index, row_rect)
            if toggle is not None:
                pos = event.pos()
                if hasattr(event, "position"):
                    pos = event.position().toPoint()
                hit = toggle.adjusted(-2, -2, 2, 2)
                if hit.contains(pos):
                    self._tree.setExpanded(index, not self._tree.isExpanded(index))
                    return True
        return super().editorEvent(
            event, model,
            option if option is not None else QStyleOptionViewItem(),
            index
        )