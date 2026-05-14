import pytest
import numpy as np
import pyqtgraph as pg
from qtpy.QtWidgets import QMainWindow, QCheckBox
from src.gui.panels.standard_plots_panel import StandardPlotsPanel
from unittest.mock import MagicMock

class SyntheticStandardPlotDataManager:
    def __init__(self):
        self.sampling_rate = 30000
        self.templates = np.zeros((1, 21, 4), dtype=np.float32)
        self.templates[0, :, 0] = np.linspace(-1, 1, 21)
        self.templates[0, :, 1] = np.linspace(1, -1, 21)
        self.channel_positions = np.array([
            [0.0, 0.0],
            [20.0, 0.0],
            [0.0, 20.0],
            [20.0, 20.0],
        ])
        self._refractory_period = 1.0

    def get_cluster_mean_amplitude(self, _cluster_id, method='mean'):
        return 100.0

    def get_refractory_period(self):
        return self._refractory_period

    def set_refractory_period(self, value):
        self._refractory_period = value

    def get_standard_plot_data(self, _cluster_id):
        return {
            'spikes': np.array([0, 3000, 6000]),
            'acg_time_lags': np.array([-1.0, 0.0, 1.0]),
            'acg_norm': np.array([0.25, 0.0, 0.25]),
            'isi_ms': np.array([100.0, 100.0]),
            'isi_vs_amp_valid_isi': np.array([100.0, 100.0]),
            'isi_vs_amp_valid_amplitudes': np.array([-100.0, -105.0]),
            'fr_bin_centers': np.array([0.0, 1.0]),
            'fr_rate': np.array([1.0, 2.0]),
            'fr_overlay_x': None,
            'fr_overlay_y': None,
        }

class MockMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_manager = SyntheticStandardPlotDataManager()
        self.similarity_panel = None
        self._selected_cluster_id = 0
        
    def get_current_colors(self):
        return {
            'bg_base': '#111214',
            'bg_panel': '#18191C',
            'bg_surface': '#1E2025',
            'border_default': '#3D3F48',
            'border_subtle': '#2E3038',
            'text_primary': '#F0F0F2',
            'text_secondary': '#9B9DA6',
            'text_tertiary': '#5A5C65',
            'text_disabled': '#3A3C44'
        }

    def _get_selected_cluster_id(self):
        return self._selected_cluster_id

@pytest.fixture
def standard_plots_panel(qtbot):
    main_window = MockMainWindow()
    panel = StandardPlotsPanel(main_window)
    qtbot.addWidget(panel)
    return panel

def test_cell_id_toggle_exists(standard_plots_panel):
    """
    SPEC: The StandardPlotsPanel should have a 'Show IDs' checkbox.
    """
    # This test will fail initially because the checkbox doesn't exist yet
    assert hasattr(standard_plots_panel, 'show_ids_checkbox')
    assert isinstance(standard_plots_panel.show_ids_checkbox, QCheckBox)
    assert standard_plots_panel.show_ids_checkbox.text() == "Show IDs"

def test_cell_id_toggle_defaults_to_unchecked(standard_plots_panel):
    """
    SPEC: The 'Show IDs' toggle should be unchecked by default.
    """
    assert standard_plots_panel.show_ids_checkbox.isChecked() is False

def test_show_ids_toggle_draws_labels_above_channel_dots(standard_plots_panel):
    """
    SPEC: When enabled, channel labels are added above the array scatter items.
    """
    standard_plots_panel.channel_mode_combo.setCurrentText("Whole Array")
    standard_plots_panel.show_ids_checkbox.setChecked(True)

    standard_plots_panel.update_all(0)

    labels = [
        item for item in standard_plots_panel.grid_plot.items
        if isinstance(item, pg.TextItem)
    ]
    assert labels, "Expected channel ID labels to be drawn"
    assert all(label.zValue() == 10 for label in labels)
