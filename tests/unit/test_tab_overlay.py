"""A tab page shown by hand overlays the current tab (PLAN.md Q29)."""

from types import SimpleNamespace

import pytest
from qtpy.QtWidgets import QApplication, QLabel, QTabWidget


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


def _tabs():
    tabs = QTabWidget()
    a, b = QLabel("standard"), QLabel("sta")
    tabs.addTab(a, "Standard")
    tabs.addTab(b, "STA")
    tabs.resize(300, 200)
    tabs.show()
    tabs.setCurrentIndex(0)
    return tabs, a, b


def test_showing_a_page_by_hand_puts_it_over_the_current_tab(qapp):
    # The mechanism behind the bug: this is what _finalize_dataset_load did.
    tabs, a, b = _tabs()
    b.show()
    assert a.isVisible() and b.isVisible()


def test_set_tab_available_never_overlays(qapp):
    from src.gui import callbacks
    tabs, a, b = _tabs()
    mw = SimpleNamespace(analysis_tabs=tabs)
    callbacks.set_tab_available(mw, b, True)
    assert a.isVisible() and not b.isVisible()
    assert tabs.isTabEnabled(1)
    callbacks.set_tab_available(mw, b, False)
    assert not tabs.isTabEnabled(1) and not b.isVisible()
