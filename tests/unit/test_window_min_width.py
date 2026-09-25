"""Wide rows must not set the window's minimum width (PLAN.md Q17).

A QStackedWidget takes the widest page's minimum, so one wide control row
in any tab held the whole window at ~1400 px.
"""

import pytest
from qtpy.QtWidgets import QApplication, QLabel, QWidget

from src.gui.panels.umap_panel import _HScrollArea


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


def test_a_wide_row_in_the_scroll_area_does_not_widen_its_parent(qapp):
    wide = QLabel("x" * 400)                      # ~2000+ px of text
    area = _HScrollArea(wide)
    assert wide.minimumSizeHint().width() > 1500
    assert area.minimumSizeHint().width() < 200


def test_the_bar_adds_height_only_when_needed(qapp):
    content = QLabel("x" * 400)
    content.setFixedHeight(40)
    area = _HScrollArea(content)
    area.resize(3000, 60)
    area.show()
    QApplication.processEvents()
    area.sync_height()
    assert area.height() == 40
    area.resize(300, 60)
    QApplication.processEvents()
    area.sync_height()
    assert area.height() > 40
