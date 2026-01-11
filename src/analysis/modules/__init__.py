"""
Analysis modules package initialization.

This package contains specialized analysis modules for waveform, STA, and spatial analysis.
"""

# Import key functions for easy access
from .waveform import (
    extract_snippets,
    baseline_correct,
    compute_ei,
    select_channels
)

from .sta import (
    get_sta_timecourse_data,
    compute_sta_metrics
)

from .spatial import compute_spatial_features

__all__ = [
    'extract_snippets',
    'baseline_correct',
    'compute_ei',
    'select_channels',
    'get_sta_timecourse_data',
    'compute_sta_metrics',
    'compute_spatial_features'
]