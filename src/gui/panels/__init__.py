"""
Panels package initialization.

This package contains the various GUI panels for different views.
"""

# Import all panels for convenience
from .similarity_panel import SimilarityPanel
from .waveforms_panel import WaveformPanel
from .standard_plots_panel import StandardPlotsPanel
from .ei_panel import EIPanel
from .raw_panel import RawPanel
from .umap_panel import UMAPPanel
from .feature_extraction import FeatureExtractionWindow

__all__ = [
    'SimilarityPanel',
    'WaveformPanel',
    'StandardPlotsPanel',
    'EIPanel',
    'RawPanel',
    'UMAPPanel',
    'FeatureExtractionWindow'
]