```markdown
# Specification: Population Panel Performance & Tree View Smoothness

## Metadata

| Field | Value |
|---|---|
| **Date Created** | 2025-04-12 |
| **Last Updated** | 2025-04-12 |
| **Spec Written Against Commit** | `b3a2f1e` (current main) |
| **Branch** | `feat/population-panel-caching` |
| **Author** | AI Agent |

## Objective

Eliminate all perceptible lag (> 200 ms) when selecting a folder containing 50–100 cells while **Population View** is enabled, and when performing drag‑and‑drop reorganisation in the cluster tree view. Currently these operations freeze the UI for several seconds because:

1. **Timecourse panel** reads STA movies from disk for each cell in the folder (`LazySTADict` cache miss).
2. **RF mosaic** redraws the entire figure (background ellipses for *all* cells) on every selection.
3. **Tree view drag/drop** triggers the same slow redraws even when the selected folder hasn’t changed.

After this fix, all population plots shall refresh in < 50 ms for cached folders and < 200 ms for the first folder after pre‑completion. Tree reorganisation shall complete in < 100 ms.

## User Story

> *As a scientist curating hundreds of retinal ganglion cells, I want to click on any group folder (e.g., “OFF sustained”, n=100) and instantly see the population RF mosaic, average timecourse, and average autocorrelation, so that I can quickly compare functional properties across groups without waiting for disk I/O or rendering delays.*

> *As a user organising clusters into hierarchical groups, I want to drag and drop folders anywhere in the tree view and have the operation complete immediately, without the population panels freezing or recalculating data that hasn’t changed.*

## Vision ID Contract

| Question | Answer |
|---|---|
| **ID space this spec operates in** | Both (hybrid and vision‑only) |
| **Reads is_vision_only?** | Yes — `get_cell_physics()` already respects the flag |
| **Translation used** | Calls `get_cell_physics()` which internally uses `vid = cluster_id if is_vision_only else cluster_id + 1` |

The timecourse panel retrieves data via `data_manager.get_cell_physics(cid)`, which already contains the correct ID translation. No new direct Vision ID access is introduced.

## Affected Files

| File | Role | Change Type |
|---|---|---|
| `src/analysis/vision_integration.py` | Increase `LazySTADict._max_cache` to ≥200 | Modify constant |
| `src/gui/workers/workers.py` | Add `PhysicsWorker` for background pre‑computation | Add class |
| `src/gui/callbacks.py` | Start `PhysicsWorker` after Vision loads; add debounce for tree view selection | Modify |
| `src/gui/panels/population_panel.py` | Add module‑level caches, refactor RF background drawing | Modify |
| `src/gui/main_window.py` | Return early in `on_view_selection_changed` if same folder selected | Minor modification |
| `tests/unit/test_population_panel.py` | Unit tests for group caching and RF background drawing | Add |
| `tests/unit/test_physics_worker.py` | Unit tests for `PhysicsWorker` | Add |
| `tests/integration/test_population_panel.py` | Integration tests with `qtbot` for latency | Add |

> **DataManager Bottleneck:** This spec touches `population_panel.py` (which reads from `DataManager` but does not modify it) and adds `PhysicsWorker` which only calls existing `get_cell_physics()`. No changes to `data_manager.py` internals. However, because `callbacks.py` and `workers.py` are modified, rebase from `main` before pushing.

## Qt Threading Contract

| Concern | Answer |
|---|---|
| **Runs on thread** | `PhysicsWorker` runs on a `QThread` (background). Population panel updates (group caching, RF background drawing) run on the main thread (they are pure numpy + matplotlib, no I/O). |
| **Emits signal** | `PhysicsWorker` emits `progress(current, total)` and `finished()`. |
| **Connected slot** | Main window's `update_cache_progress` and `_on_physics_finished`. |
| **Tier 1 or Tier 2** | Population panel redraws are triggered from `on_cluster_selection_changed` (Tier 2, after debounce). Tree view drag/drop redraws are also Tier 2. |

**Stale result guard:** Not applicable for population panel (no async worker per folder). However, the `PhysicsWorker` result is only used to update the progress bar; no data is returned.

## Cache Contract

| Concern | Answer |
|---|---|
| **Reads cache?** | Yes — `feature_cache` (for timecourses) and `standard_plot_cache` (for ACG) are read via `get_cell_physics()` and `get_acg_data()`. |
| **Writes cache?** | Yes — `get_cell_physics()` already writes to `feature_cache`. No new cache writes. |
| **Invalidates cache?** | Yes — a new function `invalidate_population_caches()` will clear the module‑level group caches (`_group_timecourse_cache`, `_group_acg_cache`, `_rf_background_state`). Called when `DataManager` signals `ei_updates_ready` (after refinement or new Vision load). |
| **Tests must bypass cache?** | For group cache logic tests: use fresh `mock_dm` (no disk cache). For verifying that `PhysicsWorker` actually calls `get_cell_physics`: use `cache_cleared_data_manager` fixture to ensure no existing `feature_cache.pkl` interferes. |

## DataManager Attributes Used

| Attribute | Type | Can be None? | Set by |
|---|---|---|---|
| `feature_cache` | `dict[int, dict]` | No (empty dict before population) | `get_cell_physics()` |
| `vision_stas` | `LazySTADict` | Yes | `load_vision_data()` |
| `cluster_df` | `pd.DataFrame` | Yes (empty before load) | `build_cluster_dataframe()` |
| `is_vision_only` | `bool` | No | `load_vision_native_data()` |

> All accesses go through existing safe methods (`get_cell_physics`, `get_acg_data`). No direct attribute access.

## Acceptance Criteria

### AC1: Instant folder selection after pre‑computation

- **Condition:** All Vision data loaded, `PhysicsWorker` has completed, user clicks a folder with 100 cells.
- **Expected:** RF mosaic, timecourse panel, ACG panel update in **< 50 ms** (measured from `selectionChanged` signal to last canvas idle draw).
- **Test type:** Integration (`test_folder_selection_latency_after_precompute`).

### AC2: First‑folder latency bound

- **Condition:** Immediately after Vision load and pre‑completion, user clicks a folder that has never been selected before.
- **Expected:** Update completes in **< 200 ms** (dominated by numpy mean/segments, no STA disk reads).
- **Test type:** Performance benchmark (`test_first_folder_latency`).

### AC3: Tree drag‑and‑drop responsiveness

- **Condition:** Population view enabled, user drags a folder to a different parent and drops.
- **Expected:** Drop completes in **< 100 ms** after mouse release. Population panels reflect new hierarchy without stutter.
- **Test type:** Integration (`test_tree_drag_drop_latency`).

### AC4: No redundant RF mosaic rebuild

- **Condition:** User selects folder A, then folder B, then folder A again.
- **Expected:** On the second selection of folder A, the RF mosaic does **not** recreate background ellipses. Only the subset ellipses and highlight are updated (or no redraw if nothing changed). Time to update < 10 ms.
- **Test type:** Unit (`test_rf_mosaic_background_drawn_once` + mock spy on `ax.add_patch`).

### AC5: Group caches invalidated correctly

- **Condition:** After cluster refinement (splitting a cluster), or after loading new Vision data (which may change timecourses), the affected folder is selected.
- **Expected:** The group cache for that folder is recomputed (cache miss). No stale data shown.
- **Test type:** Unit (`test_group_cache_invalidation_on_refinement`).

### AC6: Cache progress bar reflects physics pre‑computation

- **Condition:** Vision data is loaded. `PhysicsWorker` runs in background.
- **Expected:** The cache progress bar (already used by `StandardPlotsWorker`) shows progress from 0% to 100% and disappears when done. User can interact with UI immediately (non‑blocking).
- **Test type:** Integration (`test_physics_worker_progress_and_nonblocking`).

### AC7: STA reader cache size increased

- **Condition:** After loading Vision data, `LazySTADict` is instantiated.
- **Expected:** `_max_cache` ≥ 200.
- **Test type:** Unit (`test_lazy_sta_dict_cache_size`).

### AC8: No duplicate redraws during rapid folder toggling

- **Condition:** User clicks two different folders in quick succession (< 100 ms apart).
- **Expected:** Only the final folder triggers a redraw. Intermediate clicks are debounced (25 ms timer).
- **Test type:** Integration (`test_rapid_folder_selection_debounce`).

### AC-VISUAL: Population mosaic with many cells

- **State to reproduce:**
  1. Launch app with dataset containing ≥ 200 cells (e.g., `/mnt/lab/Array-data/sorted/20260506A/chunk10/kilosort2.5`).
  2. Enable Population View (⊞ Population button).
  3. Select a folder with ≥ 100 cells (or the root “All” group).
- **Expected appearance:** All ellipses rendered without overlaps, gridlines absent, toolbar visible with Pan/zoom working. No lag when panning.
- **Check both:** Dark mode AND Light mode.
- **Screenshot filename:** `tests/screenshots/ac_visual_population_mosaic_dark.png` and `_light.png`.
- **Verified by:** [ ] Author  [ ] Reviewer

### AC-VISUAL: Show IDs checkbox toggles IDs instantly

- **State to reproduce:** Same as above, check “Show IDs”.
- **Expected:** Cell IDs appear inside ellipses immediately (cache invalidated, background redrawn). Unchecking removes them.
- **Screenshot filename:** `tests/screenshots/ac_visual_population_show_ids.png`
- **Verified by:** [ ] Author  [ ] Reviewer

### AC-VISUAL: Tree view drag/drop result

- **State to reproduce:** Drag a folder (e.g., “OFF”) into another folder (“ON”).
- **Expected:** Folder moves in tree, population view updates to show the moved folder’s cells (if it became the new selection). No crash, no visible delay.
- **Screenshot filename:** `tests/screenshots/ac_visual_tree_drag_drop.png`
- **Verified by:** [ ] Author  [ ] Reviewer

## Regression Guard

| Bug | Fixed in | Regression test |
|---|---|---|
| ACG only using first 2 minutes | `fix/acg-full-recording` | `test_acg_includes_late_spike_trains` |
| Physics cache double‑load / missing cells | `fix/physics-cache` | `test_cell_physics_marks_cluster_computed_without_vision_sta` |
| UMAP subset‑of‑cells bug in vision‑only | `feat/vision-standalone` | `real_data_manager` fixture (all cells in `cluster_df`) |
| Population mosaic gridlines causing clutter | `fix/population-mosaic` | Visual AC (no gridlines) |
| HDBSCAN clustering default | `feat/hdbscan-clustering` | `tests/unit/test_hdbscan_clustering.py` (7 tests) |
| `LazySTADict` file handle leak | None (to be added in this spec) | `test_lazy_sta_dict_closes_handle_on_del` (new) |

## Test Plan

### Unit Tests

| Test | File | Fixture | Description |
|---|---|---|---|
| `test_physics_worker_caches_all_cells` | `tests/unit/test_physics_worker.py` | `mock_dm` | Spy on `dm.get_cell_physics`, assert called once per cluster ID. |
| `test_physics_worker_skips_duplicate_calls` | `tests/unit/test_physics_worker.py` | `mock_dm`, `threading.Event` | Run two workers concurrently; ensure per‑cell lock prevents double computation. |
| `test_group_timecourse_cache_hit` | `tests/unit/test_population_panel.py` | `mock_dm`, monkeypatch | Call `draw_population_timecourse_panel` twice with same `subset_ids`. Assert heavy computation runs only once. |
| `test_group_acg_cache_hit` | `tests/unit/test_population_panel.py` | `mock_dm`, monkeypatch | Same for ACG panel. |
| `test_rf_mosaic_background_drawn_once` | `tests/unit/test_population_panel.py` | `mock_dm`, dummy `Figure` | After first draw, count `ax.add_patch` calls for background. On second draw (different subset), new patches count = subset size, not total cells. |
| `test_group_cache_invalidation` | `tests/unit/test_population_panel.py` | `mock_dm` | Populate cache, call `invalidate_population_caches()`, assert dicts empty. |
| `test_lazy_sta_dict_cache_size` | `tests/unit/test_vision_integration.py` | `tmp_path`, dummy files | Instantiate `LazySTADict`, assert `_max_cache >= 200`. |
| `test_lazy_sta_dict_closes_handle_on_del` | `tests/unit/test_vision_integration.py` | `tmp_path`, mock reader | Create dict, delete it, assert reader close called. |

### Integration Tests (with `qtbot`)

| Test | Fixture | Steps | Pass condition |
|---|---|---|---|
| `test_folder_selection_latency_after_precompute` | `real_data_manager` (or `mock_dm` with 100 fake cells) | 1. Load dataset. 2. Wait for `PhysicsWorker.finished`. 3. Click folder. 4. Measure time to canvas idle. | < 50 ms |
| `test_first_folder_latency` | `real_data_manager` | Same as above but run **before** folder has been selected (must be the first selection). | < 200 ms |
| `test_tree_drag_drop_latency` | `real_data_manager` | 1. Enable population view. 2. Simulate drag/drop of a folder. 3. Measure time from drop to canvas idle. | < 100 ms |
| `test_tree_drag_drop_no_redundant_redraws` | `mock_main_window` | Spy on `draw_population_rfs_plot`. Perform drop that does not change selected folder. Assert no call. | 0 calls |
| `test_rapid_folder_selection_debounce` | `mock_main_window` | Simulate two folder selections within 20 ms. Use `QSignalSpy` on `canvas.draw_idle`. Assert exactly 1 redraw. | 1 redraw |
| `test_physics_worker_progress_and_nonblocking` | `mock_main_window`, `qtbot` | Start `PhysicsWorker` with 10 ms delay per cell. While running, click a button. Assert UI responsive (button click handled). | Progress updates, click handled |

### Performance Benchmarks (optional)

Add to `tests/benchmark/test_population_panel_perf.py` (requires `pytest-benchmark`):

- `bench_first_folder_selection` – record first‑folder latency.
- `bench_cached_folder_selection` – record second‑folder latency.
- `bench_tree_drag_drop` – record drop latency.

No hard thresholds in CI, but regressions flagged.

### Screenshot Verification

Use `pytest-mpl` for visual regression or manual inspection. Temp screenshots go to `tests/screenshots/` (gitignored). Baselines stored in `tests/baseline_images/` (git‑tracked). Generate with:

bash
conda run -n rgcviewer python -m pytest --mpl-generate-path tests/baseline_images/ tests/integration/test_population_panel.py


## Out of Scope

- **Optimising the initial loading of Vision data** (`.ei`, `.sta` files) – this is a one‑time cost and will be addressed separately if needed.
- **Redesigning the Matplotlib backend** – we keep using `FigureCanvasQTAgg`; the performance gains come from caching and reduced drawing, not from replacing the renderer.
- **Optimising the ACG panel** – it already uses `standard_plot_cache` and is fast; group‑level caching for ACG is added for completeness but not required.
- **Adding progress bars for group cache computation** – group caches are computed synchronously on the first click; they are cheap (numpy only) and do not warrant a progress bar.
- **Extending group caching to other panels** (e.g., similarity, standard plots) – not needed for this spec.
- **Changing the population view enable/disable logic** – only the drawing performance.
- **Refactoring the entire population panel into pyqtgraph** – remains Matplotlib; we only add caching.
```
