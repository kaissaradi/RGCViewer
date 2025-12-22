"""
Analysis package initialization.

This package provides core analysis functionality for RGC data processing.
"""

# Import key functions for easy access
from .analysis_core import (
    extract_snippets,
    baseline_correct,
    compute_ei,
    select_channels,
    get_sta_timecourse_data,
    compute_sta_metrics,
    compute_spatial_features
)

from .data_manager import DataManager
from .vision_integration import load_vision_data
from .constants import ISI_REFRACTORY_PERIOD_MS, EI_CORR_THRESHOLD

__all__ = [
    'DataManager',
    'load_vision_data',
    'extract_snippets',
    'baseline_correct',
    'compute_ei',
    'select_channels',
    'get_sta_timecourse_data',
    'compute_sta_metrics',
    'compute_spatial_features',
    'ISI_REFRACTORY_PERIOD_MS',
    'EI_CORR_THRESHOLD'
]