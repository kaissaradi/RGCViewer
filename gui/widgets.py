import pandas as pd
from qtpy.QtGui import QPainter
from PyQt5.QtGui import QColor
from qtpy.QtCore import QAbstractTableModel, Qt, QModelIndex, Signal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtWidgets import QTableView
from qtpy.QtGui import QPainter

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
        self._dataframe.sort_values(colname, ascending=(order == Qt.SortOrder.AscendingOrder), inplace=True)
        self._dataframe.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()

class MplCanvas(FigureCanvas):
    """A canvas that displays a matplotlib figure."""
    # Add a signal for mouse clicks
    clicked = Signal()
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#1f1f1f')
        super().__init__(self.fig)
        
        # Enable mouse tracking and set cursor
        self.setCursor(Qt.PointingHandCursor)
        
        # Connect matplotlib mouse press event
        self.mpl_connect('button_press_event', self._on_click)
    
    def _on_click(self, event):
        """Handle matplotlib mouse click events."""
        self.clicked.emit()

class HighlightStatusPandasModel(PandasModel):
    STATUS_COLORS = {
        'Duplicate': QColor('#FFDDDD'),      # Light red
        'Clean': QColor('#DDFFDD'),          # Light green
        'Edge': QColor('#FFFFDD'),           # Light yellow
        'Unsure': QColor('#DDDDFF'),         # Light blue
        'Noisy': QColor('#FFDDEE'),          # Pink
        'Contaminated': QColor('#FFCCFF'),   # Purple
        'Off Array': QColor('#CCCCCC'),      # Gray
    }
    def refresh_view(self, row_indices=None):
        # for row in row_indices:
        #     self._dataframe.at[row, 'status'] = status
        # Notify the view that the data has changed for these rows
        if row_indices is None:
            row_indices = range(len(self._dataframe))
        top_left = self.index(min(row_indices), 0)
        bottom_right = self.index(max(row_indices), self.columnCount()-1)
        self.dataChanged.emit(top_left, bottom_right, [Qt.BackgroundRole, Qt.ForegroundRole, Qt.DisplayRole])
    
    def data(self, index, role=Qt.DisplayRole):
        value = super().data(index, role)
        if not index.isValid():
            return value
            
        try:
            # Use lowercase 'status' to match your cluster_df column name
            if 'status' not in self._dataframe.columns:
                return value
                
            status_col_idx = self._dataframe.columns.get_loc('status')
            status_value = self._dataframe.iloc[index.row(), status_col_idx]
            
            if role == Qt.BackgroundRole:
                color = self.STATUS_COLORS.get(status_value)
                if color:
                    return color
                    
            if role == Qt.ForegroundRole:
                cluster_id_col_idx = self._dataframe.columns.get_loc('cluster_id')
                
                # Set text color for highlighted statuses
                if status_value in ['Clean', 'Edge', 'Unsure', 'Duplicate']:
                    if index.column() == cluster_id_col_idx:
                        return QColor('#FF2222')  # Red text for cluster_id
                    else:
                        return QColor('#000000')  # Black text

        except Exception as e:
            # If any error occurs, log and return the default value
            logger.exception("HighlightDuplicatesPandasModel.data error")
            pass
            
        return value
