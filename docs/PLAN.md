```markdown
# PLAN.md — RGCViewer Master Development Plan

> Read AGENTS.md before this document.
> This is a **snapshot of the current codebase state**, not a roadmap narrative.
> Update this file every time a spec completes or a new untested behavior is discovered.
> Last updated: 2026-06-09 | Branch: main

---

## 1. Fragile Zones

These files and functions will silently break something if modified carelessly.
Read the "failure mechanism" column before touching anything in the "what" column.

| What | Failure mechanism | Required action before modifying |
|---|---|---|
| `data_manager.py` (whole file) | Every panel and cache depends on it. Concurrent write bugs are timing-dependent and hard to reproduce. | `git fetch && git rebase origin/main` before every push. Run full test suite after. |
| `get_cell_physics()` | Vision ID offset lives here. `vision_id = cluster_id + 1` in hybrid mode. Accessing wrong key returns wrong cell's STA silently. | Run `test_get_cell_physics_vision_id_offset` — both parametrize branches — after any change. |
| `build_cluster_dataframe()` | Runs a single `np.unique` scan that produces `_spk_unique_cls`, `_counts`. Three downstream functions consume these. A second scan breaks count assumptions and doubles load time. | Never add a second `np.unique` or `np.argsort` on the full spike arrays inside this function. |
| `update_cluster_views()` / `_process_selection()` | The Tier 1 / Tier 2 boundary. Any heavy operation added to Tier 1 freezes the UI during keypress scrolling. | Classify every new operation as Tier 1 or Tier 2 explicitly before writing code. See AGENTS.md §1 Law 2. |
| `LazySTADict` in `vision_integration.py` | Thread-local STAReader instances are spawned for background threads to ensure thread-safe, lock-free parallel reads from the SSD. Worker readers are tracked in `_all_readers` and closed in `__del__` to prevent handle leaks. | Run `test_lazy_sta_dict_reads_are_concurrent` and `test_lazy_sta_dict_cache_is_thread_safe` after any refactoring. |
| `_save_pickle_with_fallback()` | Uses `tempfile + os.replace()` for atomicity. Replacing with a direct `pickle.dump(open(path))` would leave a corrupt truncated file if the process crashes mid-write. | Never simplify this function. The verbosity is intentional. |
| `_compute_ei_correlations_if_needed()` | The `is_vision_only` guard prevents building a 512×512 correlation matrix that exhausts RAM on large Vision-native datasets. | The guard must never be removed or conditioned on anything else. |
| `get_cluster_spike_indices()` | Returns pre-built index arrays from `_cluster_spike_indices` dict. Callers assume O(1). Replacing with `np.where(spike_clusters == id)` inside any loop causes O(N × n_clusters) runtime. | Never bypass this method. Never call `np.where` on the full spike arrays in a hot path. |

---

## 2. Test Coverage Map

### Tested — do not regress

| Behavior | Test function | File |
|---|---|---|
| ACG uses full recording, not first 2 min | `test_acg_includes_late_spike_trains` | `tests/unit/test_autocorrelation.py` |
| Too-few-spike clusters return `None` ACG | `test_acg_not_computed_for_too_few_spikes` | `tests/unit/test_autocorrelation.py` |
| Same cluster computed exactly once under concurrency | `test_standard_plot_cache_computes_same_cluster_once` | `tests/unit/test_data_manager_cache.py` |
| Different clusters can compute in parallel (no cross-lock) | `test_standard_plot_cache_allows_different_clusters_to_compute_concurrently` | `tests/unit/test_data_manager_cache.py` |
| Disk `.pkl` cache bypasses `_compute_standard_plots()` | `test_disk_cache_bypasses_computation` | `tests/unit/test_data_manager_cache.py` |
| Cell without Vision STA marked `_computed` with safe defaults | `test_cell_physics_marks_cluster_computed_without_vision_sta` | `tests/unit/test_data_manager_cache.py` |
| Raw trace snippet skips Litke TTL row | `test_raw_trace_snippet_skips_litke_ttl_row` | `tests/unit/test_data_manager_cache.py` |
| HDBSCAN runs as default, K-means as fallback (7 tests) | `tests/unit/test_hdbscan_clustering.py` | `tests/unit/test_hdbscan_clustering.py` |
| LazySTADict concurrent reads do not corrupt cache | `test_lazy_sta_dict_cache_is_thread_safe` | `tests/unit/test_physics_cache_unified.py` |
| LazySTADict SSD reads are concurrent (not serialised) | `test_lazy_sta_dict_reads_are_concurrent` | `tests/unit/test_physics_cache_unified.py` |

### Untested — add these before touching the corresponding code paths

| Behavior | Where to add | Priority | Notes |
|---|---|---|---|
| Vision ID offset — hybrid branch (`cluster_id + 1`) | `tests/unit/test_data_manager_cache.py` | **HIGH** | Parametrize over `is_vision_only=True/False` |
| Vision ID offset — vision-only branch (`cluster_id`) | `tests/unit/test_data_manager_cache.py` | **HIGH** | Same test, both branches |
| `on_features_ready()` discards result when cluster changed | `tests/integration/test_main_window.py` | **HIGH** | Use `qtbot` + `threading.Event` |
| Stale `standard_plot_cache` entries pruned after cluster refinement | `tests/unit/test_data_manager_cache.py` | **HIGH** | Inject stale key, call rebuild, assert key gone |
| Atomic pkl write: failure leaves original file intact | `tests/unit/test_data_manager_cache.py` | **HIGH** | `patch('os.replace', side_effect=OSError)` |
| `ei_corr()` with zero-std EI returns zeros, not `NaN` | `tests/unit/test_data_manager_cache.py` | MEDIUM | `np.ones((512, 201))` input |
| `_apply_ei_updates()` is called via signal, never directly | `tests/integration/test_data_manager_signals.py` | MEDIUM | Emit signal, assert `cluster_df` updated on main thread |
| Population mosaic `Show IDs` toggle invalidates hot-swap cache | `tests/integration/test_population_panel.py` | LOW | |

---

## 3. Completed Fix Registry

Every completed fix is listed here with the exact test that would catch a regression.
Before modifying any file in the "changed files" column, run the corresponding test first.

| Fix | Changed files | Regression test | Regression risk |
|---|---|---|---|
| ACG uses full recording (not first 2 min) | `data_manager.py::_compute_standard_plots()` | `test_acg_includes_late_spike_trains` | HIGH — any change to `_compute_standard_plots` |
| Physics cache double-load on init | `data_manager.py::__init__`, `_load_standard_plot_cache_from_disk()` | `test_standard_plot_cache_computes_same_cluster_once` | MEDIUM |
| Vision-only subset-of-cells UMAP bug | `data_manager.py::load_vision_native_data()` | `real_data_manager` fixture — assert all cells in `cluster_df` | HIGH — any change to vision loading |
| `vision_channel_positions` unguarded assignment (full + partial load paths) | `data_manager.py::load_vision_data()`, `_partial_vision_reload()` | Add: assert `vision_channel_positions` is None when `electrode_map` contains NaN or values >100 000 µm | MEDIUM |
| `vision_channel_positions` never set in vision-only path | `data_manager.py::load_vision_native_data()` | Add: assert `vision_channel_positions is not None` after a vision-only load with valid `.globals` | MEDIUM |
| `plot_ei_waveforms` added to `analysis_core.py` | `analysis_core.py` | Smoke test: call with synthetic (519, 60) EI + positions, assert 519 artists returned | LOW |
| Cell Tracer EI waveform overlay on single-click | `cell_tracer_dialog.py` | Manual AC: open tracer, draw lasso, single-click row → waveforms appear on canvas; alpha slider → waveforms persist; Clear lasso → waveforms removed | MEDIUM |
| UMAP toolbar overlap on first render | `panels/umap_panel.py` | Visual AC — navigate directly to UMAP tab on cold launch | LOW |
| UMAP lasso/rect selector conflict | `panels/umap_panel.py` | Manual — activate both selectors in sequence | LOW |
| HDBSCAN as default clustering | `panels/umap_panel.py`, `workers/workers.py` | `tests/unit/test_hdbscan_clustering.py` (7 tests) | LOW |
| Population mosaic gridlines, zoom/pan, Show IDs cache | `main_window.py`, `panels/population_panel.py` | Integration tests on real dataset | MEDIUM |
| Light mode theme system | `src/gui/theme.py`, `main_window.py::_setup_style()` | Visual AC — toggle theme, check all panels | LOW |
| Sidebar live search (`Ctrl+F`) | `main_window.py` | Manual | LOW |
| Physics cache warm-up freeze on large datasets | `src/analysis/vision_integration.py` | `test_lazy_sta_dict_cache_is_thread_safe`, `test_lazy_sta_dict_reads_are_concurrent` | HIGH — any changes to LazySTADict concurrency, dict caching, or contains checking |

---

## 4. Active Work

### Priority 1 — Standalone Vision Integration
- **Spec:** `docs/specs/vision_standalone.md`
- **Branch:** `feat/vision-standalone`
- **Status:** In progress
- **What is done:** Basic `load_vision_native_data()` path exists. `is_vision_only` flag set correctly.
- **What remains:**
  - Subset-of-cells UMAP bug — not all Vision cells appearing. Root cause not yet confirmed.
  - EI waveform templates from `.ei` + `.globals` not yet integrated into Standard Plots panel.
  - Graceful fallback when `.sta` is missing — `STAPanel` currently crashes.
  - Graceful fallback when `.ei` is missing — `EIPanel` currently crashes.
- **Open architectural question:** When Vision-only, `cluster_id` is used directly as `vision_id`. The `cluster_df` index must be built from `neurons_data['spikes_by_id'].keys()` — confirm this is happening in `build_cluster_dataframe()` before adding any panel code.
- **Fragile zone overlap:** Touches `data_manager.py`. Rebase from main before every push.

---

### Priority 2 — EI Panel Waveform View Mode
- **Status:** Planned — not started
- **What it is:** A third view option in the EI panel (`View: Heatmap | Photo | Waveform`) that replaces the heatmap `ImageItem` with a matplotlib canvas rendering `plot_ei_waveforms` co-registered to electrode positions. Triggered via the existing `View:` toggle pattern. Photo underlay optional (toggle). Scrolling to a new cluster re-renders via the existing `_update_ei` path.
- **What remains (all of it):**
  - Add `"Waveform"` to the `View:` button group in `ei_panel.py`
  - Add a `FigureCanvas` widget that is shown/hidden based on mode (hides the `pyqtgraph ImageItem`, shows the mpl canvas)
  - Wire cluster selection → `_draw_ei_waveform_mode()` which calls `plot_ei_waveforms` onto the panel's axes
  - Implement artist cleanup between cluster changes (same `_ei_waveform_artists` list pattern as `CellTracerDialog`)
  - Optional: expose `Photo α` slider in the panel when waveform mode is active, using `ei_panel._overlay_image_rgba` if available
- **Key constraint:** Must not touch the existing heatmap render path — mode switch is purely additive. `_load_vision_ei()` and `_load_ks_ei()` stay unchanged.
- **Uses:** `plot_ei_waveforms` from `analysis_core.py` and `_resolve_channel_positions()` already in `ei_panel.py`.
- **Fragile zone overlap:** Touches `ei_panel.py` render path. Classify any new operation as Tier 1 or Tier 2 before writing (LAW 2).

---

### Running tests

```bash
# Full suite — always run before pushing
conda run -n rgcviewer python -m pytest tests/ -v

# Unit tests only — fast, no real data
conda run -n rgcviewer python -m pytest tests/unit/ -v

# Single test
conda run -n rgcviewer python -m pytest tests/unit/test_autocorrelation.py::test_acg_includes_late_spike_trains -v

# Stop on first failure
conda run -n rgcviewer python -m pytest tests/ -x -v
```

### Real data paths

```
Raw Litke:     /mnt/lab/Array-data/raw/20260506A/data009
Sorted/Vision: /mnt/lab/Array-data/sorted/20260506A/chunk10/kilosort2.5
```

Tests using `real_data_manager` automatically skip if these paths are unmounted.

### Cache invalidation rule

Any test that verifies computation logic (ACG, ISI, physics, EI correlation) must use `tmp_path` or `cache_cleared_data_manager`. Using a real data path risks loading a warm `.pkl` and skipping all math. See AGENTS.md §1 Law 3.

### Screenshot storage

| Type | Location | Git-tracked? |
|---|---|---|
| Visual AC verification | `tests/screenshots/` | No — gitignored, auto-deleted |
| Visual regression baselines | `tests/baseline_images/` | Yes — generate with `--mpl-generate-path` |

Filename format: `ac{N}_{feature_name}_{dark|light}.png`
Default capture window size: 1800 × 1000px
Always capture both light and dark mode for any visual AC.

```
