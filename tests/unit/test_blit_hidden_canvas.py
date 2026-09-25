"""A hidden 0x0 canvas must not be drawn (AGENTS.md Law 5, PLAN.md Q13).

umap_panel.reset_for_new_dataset clears the selection shapes, which blits.
On the second dataset load the UMAP tab was hidden, its figure 0x0, and the
draw raised "'box_aspect' and 'fig_aspect' must be positive" inside
_on_kilosort_loaded, which then stopped half way.
"""

import pytest
from qtpy.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.mark.parametrize("widget", ["rf", "trace"])
def test_clearing_shapes_on_a_hidden_zero_size_view_does_not_raise(qapp, widget):
    if widget == "rf":
        from src.gui.panels.rf_map_widget import RFMapWidget
        view = RFMapWidget()
    else:
        from src.gui.panels.trace_stack_widget import TraceStackWidget
        view = TraceStackWidget()
    view._ax.set_aspect("equal")
    view.fig.set_size_inches(0, 0)
    view._blit_bg = None
    view.clear_selection_shapes()        # used to raise ValueError
