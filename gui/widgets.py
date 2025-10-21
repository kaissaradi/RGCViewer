import pandas as pd
from qtpy.QtWidgets import QTableView
from PyQt5.QtGui import QColor
from qtpy.QtCore import QAbstractTableModel, Qt, QModelIndex
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#1f1f1f')
        super().__init__(self.fig)

class HighlightDuplicatesPandasModel(PandasModel):
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
                if status_value == 'Duplicate':
                    return QColor('#FFDDDD')  # Light red background
                    
            if role == Qt.ForegroundRole:
                cluster_id_col_idx = self._dataframe.columns.get_loc('cluster_id')
                if index.column() == cluster_id_col_idx and status_value == 'Duplicate':
                    return QColor('#FF2222')  # Red text for cluster_id
        except Exception as e:
            # If any error occurs, just return the default value
            print(f"[ERROR] HighlightDuplicatesPandasModel.data error: {e}")
            pass
            
        return value
