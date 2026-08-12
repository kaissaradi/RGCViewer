from qtpy import QtCore
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout

def test_qt_setup(qtbot):
    """Sanity check to ensure pytest-qt and qtbot are working."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    button = QPushButton("Click Me")
    layout.addWidget(button)
    widget.show()
    
    # Use qtbot to simulate a click
    qtbot.mouseClick(button, QtCore.Qt.LeftButton)
    
    assert button.text() == "Click Me"
