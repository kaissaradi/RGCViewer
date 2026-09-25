from __future__ import annotations
from qtpy.QtCore import QEvent, QObject
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractSpinBox, QApplication, QLineEdit, QPlainTextEdit, QTextEdit, QWidget,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gui.main_window import MainWindow


_TEXT_INPUTS = (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)


class KeyForwarder(QObject):
    def __init__(self, main_window: MainWindow):
        super().__init__()
        self.main_window = main_window

    def _is_main_window_key(self, obj, key) -> bool:
        """Only keys meant for the main window's plots and lists (PLAN.md Q39).

        A dialog (Feature Extraction, a message box, the group picker), an
        open menu, and a text field keep their own keys; only Up/Down in the
        sidebar search bar still move the cell list, so search → arrow works.
        """
        app = QApplication.instance()
        if app is not None and app.activePopupWidget() is not None:
            return False
        target = obj if isinstance(obj, QWidget) else (app.focusWidget() if app else None)
        if target is not None and target.window() is not self.main_window:
            return False
        if isinstance(target, _TEXT_INPUTS):
            search = getattr(self.main_window, "cluster_search_bar", None)
            return target is search and key in (Qt.Key_Up, Qt.Key_Down)
        return True

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            if not self._is_main_window_key(obj, event.key()):
                return False
            if event.key() == Qt.Key_Space:
                self.main_window.similarity_panel.handle_spacebar()
                return True
            elif event.key() in (Qt.Key_Left, Qt.Key_Right):
                self.main_window.ei_panel.keyPressEvent(event)
                return True
            elif event.key() in (Qt.Key_Up, Qt.Key_Down):
                current_view = self.main_window.view_stack.currentWidget()
                if current_view is self.main_window.tree_view:
                    self.main_window._move_selection_in_view(
                        self.main_window.tree_view, event.key()
                    )
                elif current_view is self.main_window.table_view:
                    self.main_window._move_selection_in_view(
                        self.main_window.table_view, event.key()
                    )
                return True
            # Ctrl+<letter> marks the status. Noisy is Ctrl+Shift+N: Ctrl+S
            # saves the classification (docs/specs/ux_ui_redesign.md AC18).
            elif event.modifiers() & Qt.ControlModifier:
                if event.key() == Qt.Key_N and event.modifiers() & Qt.ShiftModifier:
                    status = "Noisy"
                elif event.key() == Qt.Key_D:
                    status = "Duplicate"
                elif event.key() == Qt.Key_C:
                    status = "Clean"
                elif event.key() == Qt.Key_E:
                    status = "Edge"
                elif event.key() == Qt.Key_W:
                    status = "Unsure"
                elif event.key() == Qt.Key_X:
                    status = "Contaminated"
                elif event.key() == Qt.Key_A:
                    status = "Off Array"
                else:
                    return False
                self.main_window.similarity_panel._mark_status(status)

                return True
        return False
