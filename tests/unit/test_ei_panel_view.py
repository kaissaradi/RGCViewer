"""EI View / Overlay combo routing.

These tests bind the real EIPanel methods onto a stub so they do not need
a displayed MainWindow. They lock the contract that shared handlers must
not force the heatmap when the View combo is on Waveform or 3D.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from qtpy.QtWidgets import QComboBox

from src.gui.panels.ei_panel import EIPanel, _ComboBoxNoWheel


def _make_panel(view="Heatmap", n_clusters=2):
    panel = EIPanel.__new__(EIPanel)
    panel.current_view = view
    panel.current_ei_data = [np.zeros((4, 10)), np.zeros((4, 10))][:n_clusters]
    panel.current_cluster_ids = [11, 22][:n_clusters]
    panel.overlay_index = 0
    panel._anim_frame = -1
    panel.n_frames = 10
    panel.spatial_stack = MagicMock()
    panel.mountain_widget = MagicMock()
    panel.frame_label = MagicMock()
    panel.main_window = SimpleNamespace(
        data_manager=SimpleNamespace(sampling_rate=20000.0)
    )
    panel.heatmap_frames = []
    panel.waveform_calls = 0

    def _heat(frame):
        panel.heatmap_frames.append(frame)

    def _wave():
        panel.waveform_calls += 1

    panel._draw_heatmap_frame = _heat
    panel._draw_waveform_frame = _wave
    panel._resolve_channel_positions = lambda: np.zeros((4, 2))
    return panel


class TestViewComboRouting:
    def test_switch_to_waveform_does_not_draw_heatmap(self):
        panel = _make_panel("Heatmap")
        EIPanel._on_view_changed(panel, "Waveform")
        assert panel.current_view == "Waveform"
        assert panel.waveform_calls == 1
        assert panel.heatmap_frames == []
        panel.spatial_stack.setCurrentIndex.assert_called_with(0)

    def test_switch_to_3d_uses_overlay_cluster(self):
        panel = _make_panel("Heatmap")
        panel.overlay_index = 1
        EIPanel._on_view_changed(panel, "3D")
        assert panel.current_view == "3D"
        panel.spatial_stack.setCurrentIndex.assert_called_with(1)
        ei_arg = panel.mountain_widget.plot_ei_3d.call_args[0][0]
        assert ei_arg is panel.current_ei_data[1]
        assert panel.heatmap_frames == []

    def test_empty_combo_text_is_ignored(self):
        panel = _make_panel("Waveform")
        EIPanel._on_view_changed(panel, "")
        assert panel.current_view == "Waveform"
        assert panel.waveform_calls == 0
        assert panel.heatmap_frames == []

    def test_return_to_heatmap_keeps_max_projection_frame(self):
        panel = _make_panel("Waveform")
        panel._anim_frame = -1
        EIPanel._on_view_changed(panel, "Heatmap")
        assert panel.heatmap_frames == [-1]


class TestOverlayDropdown:
    def test_overlay_change_in_waveform_redraws_waveform(self):
        panel = _make_panel("Waveform")
        EIPanel._on_overlay_dropdown(panel, 1)
        assert panel.overlay_index == 1
        assert panel.waveform_calls == 1
        assert panel.heatmap_frames == []

    def test_overlay_change_in_heatmap_redraws_heatmap(self):
        panel = _make_panel("Heatmap")
        EIPanel._on_overlay_dropdown(panel, 1)
        assert panel.overlay_index == 1
        assert panel.heatmap_frames == [-1]

    def test_negative_index_is_ignored(self):
        panel = _make_panel("Heatmap")
        EIPanel._on_overlay_dropdown(panel, -1)
        assert panel.overlay_index == 0
        assert panel.heatmap_frames == []


class TestAnimDoesNotStealView:
    def test_anim_tick_in_waveform_does_not_draw_heatmap(self):
        panel = _make_panel("Waveform")
        panel._anim_frame = 4
        EIPanel._render_anim_frame(panel)
        assert panel.heatmap_frames == []
        assert panel.waveform_calls == 0

    def test_anim_tick_in_heatmap_draws_that_frame(self):
        panel = _make_panel("Heatmap")
        panel._anim_frame = 4
        EIPanel._render_anim_frame(panel)
        assert panel.heatmap_frames == [4]


class TestComboBoxNoWheel:
    def test_subclass_is_a_combo(self):
        assert issubclass(_ComboBoxNoWheel, QComboBox)
