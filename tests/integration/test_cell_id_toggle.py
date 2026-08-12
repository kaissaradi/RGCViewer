import pytest
import numpy as np
import pyqtgraph as pg
from qtpy.QtWidgets import QMainWindow
from src.gui.panels.standard_plots_panel import StandardPlotsPanel

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

def test_standard_plots_panel_has_no_show_ids_toggle(standard_plots_panel):
    """
    SPEC: The StandardPlotsPanel should NO LONGER have a 'Show IDs' checkbox.
    """
    assert not hasattr(standard_plots_panel, 'show_ids_checkbox')

def test_standard_plots_panel_does_not_draw_channel_labels(standard_plots_panel):
    """
    SPEC: The StandardPlotsPanel template grid no longer displays channel-number labels.
    """
    standard_plots_panel.channel_mode_combo.setCurrentText("Whole Array")
    
    # We shouldn't need to check any checkbox since it doesn't exist.
    # Just update the plot and verify no pg.TextItem are added for labels.
    standard_plots_panel.update_all(0)

    labels = [
        item for item in standard_plots_panel.grid_plot.items
        if isinstance(item, pg.TextItem)
    ]
    assert not labels, "Expected NO channel ID labels to be drawn on the grid plot"
