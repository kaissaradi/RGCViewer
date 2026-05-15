import pandas as pd
from PyQt5.QtGui import QColor
from qtpy.QtCore import QAbstractTableModel, Qt, QModelIndex, Signal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtWidgets import QTableView

from ..theme import DARK_COLORS

import logging
logger = logging.getLogger(__name__)


class CustomTableView(QTableView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def paintEvent(self, event):
    #     painter = QPainter(self.viewport())
    #     for row in range(self.model().rowCount()):
    #         for column in range(self.model().columnCount()):
    #             index = self.model().index(row, column)
    #             if index.isValid():
    #                 # Draw the background color
    #                 background_color = self.model().data(index, Qt.BackgroundRole)
    #                 if background_color:
    #                     painter.fillRect(self.visualRect(index), background_color)

    #     # Call the base class implementation to handle the default painting
    #     super().paintEvent(event)


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
    # Add a signal for mouse clicks
    clicked = Signal()

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # Default to dark theme background - will be updated by restyle() if needed
        self.fig = Figure(
            figsize=(
                width,
                height),
            dpi=dpi,
            facecolor=DARK_COLORS['bg_panel'])
        super().__init__(self.fig)

        # Enable mouse tracking and set cursor
        self.setCursor(Qt.PointingHandCursor)

        # Connect matplotlib mouse press event
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
        # Caching dictionaries for faster data() lookups
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
            # Full refresh
            self._background_cache.clear()
            self._foreground_cache.clear()
            self._display_cache.clear()
        else:
            # Partial refresh
            for row in row_indices:
                for cache in [self._background_cache, self._foreground_cache, self._display_cache]:
                    keys_to_remove = [k for k in cache if k[0] == row]
                    for k in keys_to_remove:
                        cache.pop(k, None)

        # Notify views
        if row_indices is None:
            row_indices = range(len(self._dataframe))
        row_indices = list(row_indices)
        if not row_indices:
            return
        top_left = self.index(min(row_indices), 0)
        bottom_right = self.index(max(row_indices), self.columnCount() - 1)
        self.dataChanged.emit(
            top_left, bottom_right, [
                Qt.BackgroundRole, Qt.ForegroundRole, Qt.DisplayRole])

    def data(self, index, role=Qt.DisplayRole):
        """Return cached data if available, otherwise compute and cache."""
        if not index.isValid():
            return None

        row = index.row()
        cache_key = (row, index.column())

        # Check appropriate cache first
        if role == Qt.BackgroundRole:
            if cache_key in self._background_cache:
                return self._background_cache[cache_key]
        elif role == Qt.ForegroundRole:
            if cache_key in self._foreground_cache:
                return self._foreground_cache[cache_key]
        elif role == Qt.DisplayRole:
            if cache_key in self._display_cache:
                return self._display_cache[cache_key]

        # Compute if not cached
        result = self._compute_data(index, role)

        # Cache the result
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
                # Background handled by alternatingRowColors and QSS
                return None

        except Exception:
            logger.exception("HighlightStatusPandasModel.data error")

        return value
