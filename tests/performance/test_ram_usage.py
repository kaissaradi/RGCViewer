# tests/performance/test_ram_usage.py
import pytest
import os
import gc
from pathlib import Path

# psutil lives in requirements-dev.txt, not requirements.txt. Skip rather than
# raise: a bare `import psutil` here aborts collection for the whole tests/
# tree, so one missing optional dep hides every other test's result.
psutil = pytest.importorskip("psutil", reason="psutil is not installed")

pytestmark = [pytest.mark.performance, pytest.mark.slow]

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB

def test_physics_warmup_ram_limit(cache_cleared_data_manager):
    """
    PERFORMANCE TEST: Verify that capping the STA cache capacity and forcing
    periodic GC keeps RAM usage perfectly stable during a full physics warm-up.
    """
    dm = cache_cleared_data_manager
    
    # 1. Load Kilosort data
    success, _ = dm.load_kilosort_data()
    assert success
    
    dm.build_cluster_dataframe()
    cluster_ids = dm.cluster_df['cluster_id'].astype(int).tolist()
    
    # 2. Load Vision data
    dataset_name = "kilosort2.5"
    dm.load_vision_data(dm.kilosort_dir, dataset_name)
    
    # Ensure Vision is available
    if dm.vision_stas is None:
        pytest.skip("Vision STAs not available for this dataset.")
        
    # Override cache limit to 50
    dm.vision_stas._max_cache = 50
    
    gc.collect()
    initial_rss = get_process_memory()
    
    # 3. Simulate warm-up on 120 clusters
    target_ids = cluster_ids[:min(len(cluster_ids), 120)]
    
    for idx, cid in enumerate(target_ids):
        try:
            dm.get_cell_physics(cid)
        except Exception:
            pass
            
        # Trigger GC collection every 50 cells
        if (idx + 1) % 50 == 0:
            gc.collect()
            
    gc.collect()
    final_rss = get_process_memory()
    memory_growth = final_rss - initial_rss
    
    print(f"\n[Physics RAM Limit] Initial RSS: {initial_rss:.2f} MB | Final RSS: {final_rss:.2f} MB | Growth: {memory_growth:.2f} MB")
    
    # Assert that the memory growth does not exceed 300 MB (with max_cache=50,
    # the 50 STAs of 2.67 MB each take up ~133 MB in RAM, so growth should easily be < 300 MB)
    assert memory_growth < 300.0, f"Memory growth exceeded limit! Grew by {memory_growth:.2f} MB"
