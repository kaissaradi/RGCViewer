"""
Unit tests for UMAPWorker and ClusterWorker.
These verify signal emissions, valid/discarded partitions, error handling,
and clustering pathways on the high-D matrix.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from src.gui.workers.workers import UMAPWorker, ClusterWorker

class MockDataManager:
    def __init__(self, cluster_df=None, raw_blocks=None, valid_ids=None, discarded_ids=None):
        self.cluster_df = cluster_df if cluster_df is not None else pd.DataFrame()
        self.raw_blocks = raw_blocks
        self.valid_ids = valid_ids or []
        self.discarded_ids = discarded_ids or []
        self.ensure_physics_cache_calls = []

    def ensure_physics_cache(self, cluster_ids):
        self.ensure_physics_cache_calls.append(cluster_ids)

    def get_raw_feature_blocks(self, cluster_ids, filter_config):
        return self.raw_blocks, self.valid_ids, self.discarded_ids


def test_worker_emits_valid_and_discarded():
    """Verify that UMAPWorker signals emit correct arguments and valid/discarded partitions."""
    cluster_df = pd.DataFrame({'cluster_id': [0, 1, 2, 3, 4, 5], 'KSLabel': ['good'] * 5 + ['mua']})
    
    # 6 cells. 1 discarded, 5 valid.
    # temporal: (5, 20), acg: (5, 50), scalars: 5 rows
    raw_blocks = {
        'temporal': np.random.randn(5, 20),
        'acg': np.random.randn(5, 50),
        'scalars': pd.DataFrame({
            'firing_rate': [5.0, 10.0, 7.5, 8.0, 9.2],
            'isi_violations': [0.1, 0.2, 0.15, 0.05, 0.12],
            'time_to_peak': [12.0, 15.0, 10.0, 14.0, 11.0],
            'rf_area': [45.0, 60.0, 50.0, 55.0, 48.0],
            'ellipticity': [0.9, 1.1, 1.0, 0.95, 1.05],
        })
    }
    
    dm = MockDataManager(cluster_df, raw_blocks, valid_ids=[0, 1, 2, 3, 4], discarded_ids=[5])
    
    feature_config = {
        'use_temporal': True, 'w_temporal': 1.0,
        'use_acg': True, 'w_acg': 1.0,
        'use_firing_rate': True, 'w_firing_rate': 1.0,
        'use_isi_violations': True, 'w_isi_violations': 1.0,
        'use_time_to_peak': True, 'w_time_to_peak': 1.0,
        'use_rf_area': True, 'w_rf_area': 1.0,
        'use_ellipticity': True, 'w_ellipticity': 1.0,
    }
    filter_config = {'min_sta_std': 1e-5, 'max_rf_area': 300.0}

    worker = UMAPWorker(dm, selected_cluster_ids=[0, 1, 2, 3, 4, 5], n_components=2,
                        feature_config=feature_config, filter_config=filter_config)
    
    finished_args = []
    errors = []
    
    worker.finished.connect(lambda *args: finished_args.append(args))
    worker.error.connect(errors.append)
    
    worker.run()
    
    assert not errors
    assert len(finished_args) == 1
    
    embedding, matrix, valid_ids, discarded_ids, meta_df = finished_args[0]
    
    assert embedding.shape == (5, 2)
    assert valid_ids == [0, 1, 2, 3, 4]
    assert discarded_ids == [5]
    assert len(meta_df) == 5
    assert list(meta_df['cluster_id']) == [0, 1, 2, 3, 4]
    assert 'KSLabel' in meta_df.columns
    assert 'Polarity' in meta_df.columns
    assert 'Firing Rate' in meta_df.columns
    assert 'isi_violations' in meta_df.columns


def test_worker_all_filtered_out():
    """All cells fail filter -> graceful empty result/error emitted, no crash."""
    cluster_df = pd.DataFrame({'cluster_id': [0, 1], 'KSLabel': ['good', 'good']})
    
    # 0 valid cells, 2 discarded
    raw_blocks = {
        'temporal': np.empty((0, 20)),
        'acg': np.empty((0, 50)),
        'scalars': pd.DataFrame()
    }
    
    dm = MockDataManager(cluster_df, raw_blocks, valid_ids=[], discarded_ids=[0, 1])
    
    worker = UMAPWorker(dm, selected_cluster_ids=[0, 1], n_components=2)
    
    finished_args = []
    errors = []
    
    worker.finished.connect(lambda *args: finished_args.append(args))
    worker.error.connect(errors.append)
    
    worker.run()
    
    assert not finished_args
    assert len(errors) == 1
    assert "filtered out" in errors[0]


def test_worker_single_cell_passes():
    """Only 1 cell passes filter -> UMAP fails (requires neighbors >= 2), gracefully handled."""
    cluster_df = pd.DataFrame({'cluster_id': [0], 'KSLabel': ['good']})
    
    raw_blocks = {
        'temporal': np.random.randn(1, 20),
        'acg': np.random.randn(1, 50),
        'scalars': pd.DataFrame({
            'firing_rate': [5.0],
            'isi_violations': [0.1],
            'time_to_peak': [12.0],
            'rf_area': [45.0],
            'ellipticity': [0.9],
        })
    }
    
    dm = MockDataManager(cluster_df, raw_blocks, valid_ids=[0], discarded_ids=[])
    
    worker = UMAPWorker(dm, selected_cluster_ids=[0], n_components=2)
    
    finished_args = []
    errors = []
    
    worker.finished.connect(lambda *args: finished_args.append(args))
    worker.error.connect(errors.append)
    
    worker.run()
    
    assert not finished_args
    assert len(errors) == 1
    assert any(term in errors[0] for term in ("n_neighbors", "ValueError", "neighborhood"))


def test_cluster_worker_uses_highd_matrix():
    """ClusterWorker receives high-D matrix, not 2D embedding, and clusters correctly."""
    matrix = np.random.randn(10, 8)
    
    from src.gui.workers.workers import HDBSCAN_AVAILABLE
    if HDBSCAN_AVAILABLE:
        worker = ClusterWorker(matrix, method="HDBSCAN", param=3)
        finished_args = []
        errors = []
        worker.finished.connect(lambda *args: finished_args.append(args))
        worker.error.connect(errors.append)
        
        worker.run()
        assert not errors
        assert len(finished_args) == 1
        labels, method = finished_args[0]
        assert len(labels) == 10
        assert method == "HDBSCAN"
        
    worker_km = ClusterWorker(matrix, method="K-Means", param=3)
    finished_km = []
    errors_km = []
    worker_km.finished.connect(lambda *args: finished_km.append(args))
    worker_km.error.connect(errors_km.append)
    
    worker_km.run()
    assert not errors_km
    assert len(finished_km) == 1
    labels, method = finished_km[0]
    assert len(labels) == 10
    assert method == "K-Means"
    assert len(np.unique(labels)) <= 3
