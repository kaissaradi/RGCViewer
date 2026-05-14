# Specification: Physics Cache & Threading Fix

## Metadata

* **Status:** Ready for Dev
* **Target Release:** v1.1
* **Primary Developer/Agent:** Agent 1

## Objective

Fix the "double loading" bug where the application ignores existing pre-computed physics caches (`.pkl` files) and redundantly recalculates ACG, ISI, and RF data on startup, freezing the background workers.

## User Story

"As a user loading a previously analyzed dataset, I want the physics cache to load from disk instantly so that the progress bar skips the calculation phase and I can view my UMAP and Standard Plots immediately."

## Acceptance Criteria (Definition of Done)

* **AC1:** If a valid cache file exists for a cluster, `DataManager` must load it into RAM without dispatching a computation task to the `StandardPlotsWorker`.
* **AC2:** The UI progress bar must reflect the instantly loaded cache (e.g., jumping to 100% if all clusters are cached).
* **AC3:** If the cache is missing or corrupt, it gracefully falls back to computing it via the background thread.
* **AC4 (Regression):** Thread locks must continue to prevent two workers from computing the same cluster simultaneously (as proven in `test_data_manager_cache.py`).
* **AC5:** Clusters without Vision STA data must still be marked `_computed` in `feature_cache` with safe defaults (so UMAP cache readiness can reach `N/N`).

## Architecture & Technical Constraints

* **Files Modified:** * `src/analysis/data_manager.py` (specifically `get_standard_plot_data` and disk I/O logic).
  * `src/gui/workers/workers.py` (ensure worker emits progress even on per-cluster errors).
* **Data Contracts:** Persisted caches must be loadable as dictionaries keyed by `cluster_id`. After `cluster_df` is finalized, stale keys are pruned against the current cluster population.
* **UI/Threading Rules:** Disk I/O for the cache must not block the main Qt thread.

## Test Plan (TDD Requirements)

* **Unit:** * Add `test_disk_cache_bypasses_computation()` to `test_data_manager_cache.py`.
  * Create a fake `.pkl` file, call the fetch method, and assert that the FFT/math logic is never triggered (e.g., by mocking `_compute_standard_plots` and asserting it is not called).
* **Unit:** Add a regression proving `get_cell_physics()` marks `_computed` even when Vision STA is missing (defaults for `timecourse`/geometry).
* **Integration:** Load a subset of real data from `/mnt/lab/Array-data/` that already has a cache, and assert via `qtbot` that the UI panels populate in under 500ms.

## Out Of Scope

* Changing the mathematical formulas for ACG or ISI.
