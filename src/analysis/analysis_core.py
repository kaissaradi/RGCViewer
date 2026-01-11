"""
Analysis Core Facade

This module provides a single import point for all analysis functions.
It aggregates functionality from submodules to simplify imports elsewhere.
"""

# Re-export functions from submodules for convenient access
from .modules.waveform import (
    extract_snippets,
    baseline_correct,
    compute_ei,
    select_channels
)

from .modules.sta import (
    get_sta_timecourse_data,
    compute_sta_metrics
)

from .modules.spatial import compute_spatial_features

__all__ = [
    'extract_snippets',
    'baseline_correct',
    'compute_ei',
    'select_channels',
    'get_sta_timecourse_data',
    'compute_sta_metrics',
    'compute_spatial_features'
]
