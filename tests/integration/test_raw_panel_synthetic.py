import numpy as np
import pandas as pd
from qtpy.QtWidgets import QMainWindow
from unittest.mock import MagicMock
from src.gui.panels.raw_panel import RawPanel

class SyntheticRawDataManager:
    def __init__(self):
        self.sampling_rate = 30000
        self.n_channels = 512
        self.n_samples = 300000
        self.uV_per_bit = 1.0
        self.raw_reader = object()
        self.raw_data_memmap = None
        self.cluster_df = pd.DataFrame({
            'cluster_id': [1],
            'KSLabel': ['Good'],
            'n_spikes': [3],
            'isi_violations_pct': [0.0],
        })
        self.templates = np.zeros((2, 61, self.n_channels), dtype=np.float32)
        self.templates[1, :, 7] = np.sin(np.linspace(0, np.pi, 61))
        self.channel_to_templates = {}
        self.cluster_spike_indices = {1: np.array([0, 1, 2])}
        self.raw_calls = 0

    def get_lightweight_features(self, _cluster_id):
        return None

    def get_nearest_channels(self, central_channel_idx, n_channels=3):
        return [central_channel_idx, central_channel_idx + 1, central_channel_idx + 2][:n_channels]

    def get_cluster_spikes(self, _cluster_id):
        return np.array([1000, 5000, 10000])

    def get_cluster_spike_amplitudes(self, _cluster_id):
        return np.array([-100, -110, -105])

    def get_raw_trace_snippet(self, channel_indices, start_sample, end_sample):
        self.raw_calls += 1
        end_sample - start_sample
        t = np.arange(start_sample, end_sample, dtype=np.float32)
        traces = []
        for channel in channel_indices:
            freq = 10.0 + (int(channel) % 7)
            traces.append(50.0 * np.sin(2 * np.pi * freq * t / self.sampling_rate))
        return np.asarray(traces, dtype=np.float32)

class MockMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_manager = SyntheticRawDataManager()
        self.status_bar = MagicMock()

    def get_current_colors(self):
        return {
            'bg_panel': '#18191C',
            'text_primary': '#F0F0F2',
            'text_secondary': '#9B9DA6',
            'accent': '#2E6DD4',
            'border_default': '#3D3F48'
        }

def test_raw_panel_synthetic_data(qtbot):
    """
    INTEGRATION TEST: Verify RawPanel (Trace Viewer) works with synthetic data.
    """
    main_window = MockMainWindow()
    panel = RawPanel(main_window)
    qtbot.addWidget(panel)
    panel.show()
    
    panel.load_data(1)
    
    qtbot.waitUntil(
        lambda: panel._trace_dom.yData is not None and len(panel._trace_dom.yData) > 0,
        timeout=3000,
    )
    
    y_data = panel._trace_dom.yData
    assert y_data is not None, "Dominant trace yData is None!"
    assert np.any(y_data != 0), "Trace data is empty/zeros!"

    initial_calls = main_window.data_manager.raw_calls
    panel._request_load(2.0, panel.window_duration)
    qtbot.waitUntil(
        lambda: main_window.data_manager.raw_calls > initial_calls and panel._worker_thread is None,
        timeout=3000,
    )
    
    panel.cleanup() if hasattr(panel, 'cleanup') else None
    panel.setParent(None)
    panel.deleteLater()
