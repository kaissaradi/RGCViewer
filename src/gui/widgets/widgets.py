import numpy as np
import pandas as pd
from qtpy.QtGui import QColor, QPainter, QPen, QStandardItem
from qtpy.QtCore import (
    QAbstractTableModel,
    Qt,
    QModelIndex,
    QSortFilterProxyModel,
    Signal,
    QRect,
    QEvent,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtWidgets import (
    QTableView,
    QStyledItemDelegate,
    QTreeView,
    QStyleOptionViewItem,
)

from ..theme import DARK_COLORS

import logging

logger = logging.getLogger(__name__)


class CustomTableView(QTableView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PandasModel(QAbstractTableModel):
    """A model to interface a pandas DataFrame with a QTableView."""

    # DisplayRole is formatted text, so a proxy sorting on it compares strings
    # and puts "1000" before "999". SORT_ROLE hands back the underlying value
    # with its real type, so numeric columns sort numerically. The view's proxy
    # opts in via setSortRole() — see MainWindow.setup_table_model.
    SORT_ROLE = Qt.ItemDataRole.UserRole + 100

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
        if role == PandasModel.SORT_ROLE:
            value = self._dataframe.iloc[index.row(), index.column()]
            if pd.isna(value):
                return None  # invalid QVariant — Qt groups these together
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            if isinstance(value, (int, np.integer)):
                return int(value)
            if isinstance(value, (float, np.floating)):
                return float(value)
            return str(value)
        return None

    def cluster_id_at_row(self, row: int):
        """The cluster_id in *row*, or None when the frame has no such column.

        Lets a proxy filter rows by identity without knowing which column index
        cluster_id currently occupies — the table's columns are user-reorderable
        and are rebuilt whenever a background pass adds one.
        """
        df = self._dataframe
        if "cluster_id" not in df.columns or not (0 <= row < len(df)):
            return None
        try:
            return int(df["cluster_id"].iloc[row])
        except (TypeError, ValueError):
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
        self._dataframe.sort_values(
            colname, ascending=(order == Qt.SortOrder.AscendingOrder), inplace=True
        )
        self._dataframe.reset_index(inplace=True, drop=True)
        if hasattr(self, "refresh_view"):
            self.refresh_view()
        self.layoutChanged.emit()


class HiddenIdFilterProxyModel(QSortFilterProxyModel):
    """The table's search/sort proxy, with certain cluster IDs filtered out.

    Cells the user has thrown away live on in the tree's Trash folder — that is
    what makes the decision reversible — but they should be gone from the table,
    which is where curation actually happens. Scrolling past discarded units, or
    worse re-selecting one, is the whole reason to trash them.

    ``hidden_ids_fn`` is re-consulted on ``refresh_hidden()`` rather than cached
    at construction: rows move in and out of Trash constantly, and the proxy has
    no way to observe that on its own.
    """

    def __init__(self, hidden_ids_fn, parent=None):
        super().__init__(parent)
        self._hidden_ids_fn = hidden_ids_fn
        self._hidden = frozenset()

    def refresh_hidden(self):
        """Re-read the hidden set; re-filter only if it actually changed."""
        try:
            hidden = frozenset(int(c) for c in (self._hidden_ids_fn() or ()))
        except Exception:
            logger.debug("hidden-id lookup failed", exc_info=True)
            return
        if hidden != self._hidden:
            self._hidden = hidden
            self.invalidateFilter()

    @property
    def hidden_count(self) -> int:
        return len(self._hidden)

    def filterAcceptsRow(self, source_row, source_parent):
        if self._hidden:
            model = self.sourceModel()
            getter = getattr(model, "cluster_id_at_row", None)
            if callable(getter):
                cid = getter(source_row)
                if cid is not None and cid in self._hidden:
                    return False
        return super().filterAcceptsRow(source_row, source_parent)


class MplCanvas(FigureCanvas):
    """A canvas that displays a matplotlib figure."""

    clicked = Signal()

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(
            figsize=(width, height), dpi=dpi, facecolor=DARK_COLORS["bg_panel"]
        )
        super().__init__(self.fig)
        self.setCursor(Qt.PointingHandCursor)
        self.mpl_connect("button_press_event", self._on_click)

    def restyle(self, colors):
        """Updates the canvas background based on the provided color scheme."""
        self.fig.patch.set_facecolor(colors["bg_panel"])
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

    # Sort keys for chirp_qi values whose repeat count is too low to trust are
    # shifted below this, so a descending sort puts every trustworthy cell
    # first regardless of the raw numbers. Well clear of any real QI, which is
    # a variance ratio and in practice stays under ~50.
    UNTRUSTED_SORT_OFFSET = 1e6

    def __init__(self, dataframe: pd.DataFrame, parent=None):
        super().__init__(dataframe, parent)
        self._background_cache = {}
        self._foreground_cache = {}
        self._display_cache = {}
        # Set by MainWindow once the chirp file's repeat count is known.
        self._chirp_qi_trustworthy = True
        self._chirp_n_repeats = None
        self._chirp_min_repeats = None

    def update_colors(self, colors):
        """Updates status colors based on the current theme."""
        self.STATUS_COLORS = {
            "Clean": colors.get("status_good_text", DARK_COLORS["status_good_text"]),
            "Edge": colors.get("status_mua_text", DARK_COLORS["status_mua_text"]),
            "Duplicate": colors.get(
                "status_noise_text", DARK_COLORS["status_noise_text"]
            ),
            "Noise": colors.get("status_noise_text", DARK_COLORS["status_noise_text"]),
            "Unsure": colors.get(
                "status_unsort_text", DARK_COLORS["status_unsort_text"]
            ),
            "Original": colors.get("text_primary", DARK_COLORS["text_primary"]),
        }
        self.refresh_view()

    def set_chirp_qi_context(self, n_repeats, trustworthy, min_repeats=None):
        """Tell the model how many stimulus repeats back the chirp QI column.

        Above the threshold the column is drawn green, below it red, demoted in
        sort order and tooltipped with the count — the QI's floor is
        ``1/n_repeats``, so a low-repeat run yields respectable-looking numbers
        that predict nothing, and the colour is the standing reminder.
        """
        self._chirp_n_repeats = n_repeats
        self._chirp_qi_trustworthy = bool(trustworthy)
        if min_repeats is not None:
            self._chirp_min_repeats = min_repeats
        self.refresh_view()

    def refresh_view(self, row_indices=None):
        """Invalidate cache on data change."""
        if row_indices is None:
            self._background_cache.clear()
            self._foreground_cache.clear()
            self._display_cache.clear()
        else:
            for row in row_indices:
                for cache in [
                    self._background_cache,
                    self._foreground_cache,
                    self._display_cache,
                ]:
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
            top_left,
            bottom_right,
            [Qt.BackgroundRole, Qt.ForegroundRole, Qt.DisplayRole],
        )

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
            col_name = self._dataframe.columns[index.column()]

            # --- chirp QI: flag and demote when too few repeats back it ---
            # Handled before the status lookup because it applies whether or
            # not a status column exists.
            if col_name == "chirp_qi" and self._chirp_qi_trustworthy:
                # Green says the repeat count behind this number is enough for
                # it to mean anything — not that the cell is good. Paired with
                # the red below, the column reads at a glance as "can I trust
                # this column at all", which is the question that actually
                # bites: the QI's floor is 1/n_trials, so a low-repeat run
                # produces respectable-looking values that predict nothing.
                if role == Qt.ForegroundRole and value is not None:
                    return QColor(self.STATUS_COLORS.get("Good", "#6EE7B7"))
                if role == Qt.ToolTipRole:
                    return (f"Chirp QI from {self._chirp_n_repeats} repeats per run "
                            f"(>= {self._chirp_min_repeats} needed).\n"
                            f"Right-click for the per-light-level breakdown.")

            if col_name == "chirp_qi" and not self._chirp_qi_trustworthy:
                if role == Qt.ForegroundRole:
                    return QColor(self.STATUS_COLORS.get("Noise", "#F08080"))
                if role == PandasModel.SORT_ROLE and value is not None:
                    return float(value) - self.UNTRUSTED_SORT_OFFSET
                if role == Qt.ToolTipRole:
                    n = self._chirp_n_repeats
                    return (
                        f"Quality index from only {n} stimulus repeats"
                        if n is not None
                        else "Quality index from an unknown number of repeats"
                    ) + (
                        ".\nThe QI is a ratio of two variances estimated from "
                        "the repeats; below ~8 it stops tracking anything real "
                        "and does not reproduce on a split-half test.\n"
                        "Treat these values as unusable, not merely noisy."
                    )

            if "status" not in self._dataframe.columns:
                return value

            status_col_idx = self._dataframe.columns.get_loc("status")
            status_value = str(self._dataframe.iloc[index.row(), status_col_idx])

            if role == Qt.ForegroundRole:
                if col_name == "status":
                    color = self.STATUS_COLORS.get(
                        status_value, DARK_COLORS["text_primary"]
                    )
                    return QColor(color)
                return QColor(
                    self.STATUS_COLORS.get("Original", DARK_COLORS["text_primary"])
                )

            if role == Qt.BackgroundRole:
                return None

        except Exception:
            logger.exception("HighlightStatusPandasModel.data error")

        return value


# Tree sidebar: one row is [ID | Spikes | Ch]. Cluster identity lives on
# column 0 UserRole (int for a cell, None for a folder). TREE_SORT_ROLE is a
# numeric key so header-click sort does not put "1000" before "999".
TREE_COL_ID = 0
TREE_COL_SPIKES = 1
TREE_COL_CH = 2
TREE_HEADERS = ("ID", "Spikes", "Ch")
TREE_SORT_ROLE = Qt.ItemDataRole.UserRole + 1
TREE_SPIKES_WIDTH = 56
TREE_CH_WIDTH = 40


def format_spike_count(n_spikes) -> str:
    if n_spikes is None:
        return "—"
    try:
        if isinstance(n_spikes, float) and np.isnan(n_spikes):
            return "—"
        return f"{int(n_spikes)}"
    except (TypeError, ValueError):
        return "—"


def format_channel(best_chan) -> str:
    if best_chan is None:
        return "—"
    try:
        if isinstance(best_chan, float) and np.isnan(best_chan):
            return "—"
        ch = int(best_chan)
    except (TypeError, ValueError):
        return "—"
    if ch < 0:
        return "—"
    return str(ch)


def _sort_int(value):
    if value is None:
        return None
    try:
        if isinstance(value, float) and np.isnan(value):
            return None
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    return number


class TreeStandardItem(QStandardItem):
    """QStandardItem that compares TREE_SORT_ROLE numerically when both sides have it."""

    def __lt__(self, other):
        if not isinstance(other, QStandardItem):
            return super().__lt__(other)
        left = self.data(TREE_SORT_ROLE)
        right = other.data(TREE_SORT_ROLE)
        if left is not None and right is not None:
            try:
                return left < right
            except TypeError:
                pass
        if left is not None:
            return False
        if right is not None:
            return True
        return (self.text() or "") < (other.text() or "")


def make_group_row(name, child_count=0):
    """Folder row: name, leaf count, empty Ch. UserRole on col 0 stays None."""
    name_item = TreeStandardItem(str(name))
    name_item.setEditable(False)
    name_item.setDropEnabled(True)
    font = name_item.font()
    font.setBold(True)
    name_item.setFont(font)

    count_item = TreeStandardItem()
    count_item.setEditable(False)
    count_item.setDropEnabled(False)
    count_item.setTextAlignment(
        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
    )
    set_count_item(count_item, child_count)

    ch_item = TreeStandardItem("")
    ch_item.setEditable(False)
    ch_item.setDropEnabled(False)

    return [name_item, count_item, ch_item]


def make_cell_row(cluster_id, n_spikes=None, best_chan=None):
    """Cell row: ID, spike count, channel. UserRole on col 0 is the cluster id."""
    cid = int(cluster_id)
    id_item = TreeStandardItem(str(cid))
    id_item.setEditable(False)
    id_item.setDropEnabled(False)
    id_item.setData(cid, Qt.ItemDataRole.UserRole)
    id_item.setData(cid, TREE_SORT_ROLE)

    spikes_item = TreeStandardItem(format_spike_count(n_spikes))
    spikes_item.setEditable(False)
    spikes_item.setDropEnabled(False)
    spikes_item.setTextAlignment(
        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
    )
    spikes_sort = _sort_int(n_spikes)
    if spikes_sort is not None:
        spikes_item.setData(spikes_sort, TREE_SORT_ROLE)

    ch_item = TreeStandardItem(format_channel(best_chan))
    ch_item.setEditable(False)
    ch_item.setDropEnabled(False)
    ch_item.setTextAlignment(
        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
    )
    ch_sort = _sort_int(best_chan)
    if ch_sort is not None:
        ch_item.setData(ch_sort, TREE_SORT_ROLE)

    return [id_item, spikes_item, ch_item]


def set_count_item(count_item, n):
    try:
        count = int(n)
    except (TypeError, ValueError):
        count = 0
    text = str(count) if count else ""
    if count_item.text() != text:
        count_item.setText(text)
    if count_item.data(TREE_SORT_ROLE) != count:
        count_item.setData(count, TREE_SORT_ROLE)


def set_channel_item(ch_item, best_chan):
    text = format_channel(best_chan)
    if ch_item.text() != text:
        ch_item.setText(text)
    ch_sort = _sort_int(best_chan)
    if ch_item.data(TREE_SORT_ROLE) != ch_sort:
        ch_item.setData(ch_sort, TREE_SORT_ROLE)


def tree_item_from_index(model, index):
    """Column-0 item for a tree index. Spikes/Ch cells are not identity items."""
    if model is None or index is None or not index.isValid():
        return None
    if index.column() != TREE_COL_ID:
        index = index.sibling(index.row(), TREE_COL_ID)
    return model.itemFromIndex(index)


def count_tree_leaves(item) -> int:
    """Recursive cell count under a folder (col-0 walk)."""
    if item is None:
        return 0
    total = 0
    for row in range(item.rowCount()):
        child = item.child(row, TREE_COL_ID)
        if child is None:
            continue
        if child.data(Qt.ItemDataRole.UserRole) is None:
            total += count_tree_leaves(child)
        else:
            total += 1
    return total


def refresh_tree_group_counts(root_item):
    """Write leaf counts into every folder's Spikes cell."""
    if root_item is None:
        return

    def visit(item):
        for row in range(item.rowCount()):
            child = item.child(row, TREE_COL_ID)
            if child is None:
                continue
            if child.data(Qt.ItemDataRole.UserRole) is None:
                visit(child)
                count_item = item.child(row, TREE_COL_SPIKES)
                if count_item is not None:
                    set_count_item(count_item, count_tree_leaves(child))

    visit(root_item)


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
        line = QColor(self._colors["border_default"])
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
        painter.drawLine(
            x, row_rect.center().y(), row_rect.left(), row_rect.center().y()
        )

    def _draw_toggle(self, painter: QPainter, rect: QRect, expanded: bool):
        accent = QColor(self._colors["text_secondary"])
        painter.setPen(QPen(accent, 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect, 2, 2)
        cx, cy = rect.center().x(), rect.center().y()
        painter.setPen(QPen(accent, 1.5))
        painter.drawLine(cx - 3, cy, cx + 3, cy)
        if not expanded:
            painter.drawLine(cx, cy - 3, cx, cy + 3)

    def paint(self, painter, option, index):
        if index.column() != TREE_COL_ID:
            super().paint(painter, option, index)
            return
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
        if index.column() != TREE_COL_ID:
            return super().editorEvent(
                event,
                model,
                option if option is not None else QStyleOptionViewItem(),
                index,
            )
        if (
            event.type() == QEvent.MouseButtonRelease
            and event.button() == Qt.LeftButton
        ):
            # In headless/test environments visualRect may return a null rect.
            # Fall back to a synthetic row_rect derived from the event position
            # so that _toggle_rect produces coords in the same space as event.pos().
            row_rect = self._tree.visualRect(index)
            if not row_rect.isValid() or row_rect.isEmpty():
                pos = (
                    event.pos()
                    if not hasattr(event, "position")
                    else event.position().toPoint()
                )
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
            event,
            model,
            option if option is not None else QStyleOptionViewItem(),
            index,
        )
