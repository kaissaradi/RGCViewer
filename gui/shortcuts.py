from __future__ import annotations
from qtpy.QtCore import QEvent, QObject
from qtpy.QtCore import Qt
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gui.main_window import MainWindow

class KeyForwarder(QObject):
    def __init__(self, main_window: MainWindow):
        super().__init__()
        self.main_window = main_window

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Space:
                self.main_window.similarity_panel.handle_spacebar()
                return True
            elif event.key() in (Qt.Key_Left, Qt.Key_Right):
                self.main_window.ei_panel.keyPressEvent(event)
                return True
            elif event.key() in (Qt.Key_Up, Qt.Key_Down):
                current_view = self.main_window.view_stack.currentWidget()
                if current_view is self.main_window.tree_view:
                    self.main_window._move_selection_in_view(self.main_window.tree_view, event.key())
                elif current_view is self.main_window.table_view:
                    self.main_window._move_selection_in_view(self.main_window.table_view, event.key())
                return True
            # Add Cmd+D / Ctrl+D shortcut for marking duplicates
            elif (event.modifiers() & Qt.ControlModifier):
                if event.key() == Qt.Key_D: 
                    status = 'Duplicate'
                elif event.key() == Qt.Key_C:
                    status = 'Clean'
                elif event.key() == Qt.Key_E:
                    status = 'Edge'
                elif event.key() == Qt.Key_W:
                    status = 'Unsure'
                elif event.key() == Qt.Key_S:
                    status = 'Noisy'
                elif event.key() == Qt.Key_X:
                    status = 'Contaminated'
                elif event.key() == Qt.Key_A:
                    status = 'Off Array'
                else:
                    return False
                self.main_window.similarity_panel._mark_status(status)
                
                return True
        return False