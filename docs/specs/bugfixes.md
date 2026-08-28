# Specification: Critical Production Bug Fixes

**Status (2026-08-12):** Historical. This spec is not the active queue.
See `docs/PLAN.md`.

## Metadata

* **Status:** Historical — not the active queue
* **Target Release:** v1.1
* **Primary Developer:** Kais Saradi
* **Branch:** `fix/critical-production-bugs`
* **Spec File:** `docs/specs/critical-production-bugs.md`
* **Last Updated:** 2026-05-23

---

## Objective

Fix verified production bugs that cause crashes, UI freezes, severe repeated
work, or unsafe code execution. Cosmetic, speculative, or unsupported items
are explicitly excluded and listed under Out Of Scope.

---

## User Story

"As a scientist using Encore on large Vision/Kilosort datasets, I want
dataset loading, similarity lookup, and physics cache computation to avoid
crashes, redundant work, and UI stalls so that I can safely inspect cells
without restarting the app."

---

## Acceptance Criteria & Status

| # | Criterion | Status | Test | Commit |
|---|---|---|---|---|
| AC1 | `load_neurons_data()` broad exception returns `None`; no `NameError` | ✅ Done | *(no regression test — pre-existing correct behavior)* | `1d7caae` |
| AC2 | Repeated MEA similarity requests return from `mea_sim_cache` without rebuilding | ✅ Done | `test_mea_similarity_table_reuses_cluster_cache` | `141e82d` |
| AC3 | Repeated Vision similarity requests return from `vision_sim_cache` without rereading all lazy STAs | ✅ Done | `test_vision_similarity_table_reuses_cluster_cache` | `124e41a` |
| AC4 | `_load_standard_plot_cache_from_disk()` does not hold `_standard_plot_lock` during `pickle.load()` | ❌ Not started | `test_standard_plot_lock_not_held_during_pickle_load` *(not written)* | — |
| AC5 | Concurrent `get_cell_physics()` calls for same cluster compute exactly once | ❌ Not started | `test_get_cell_physics_computes_once_under_concurrency` *(not written)* | — |
| AC6 | `_load_kilosort_params()` uses `ast.literal_eval()`; no arbitrary code execution | ✅ Done | `test_kilosort_params_uses_literal_eval_without_executing_code` | `1d7caae` |
| AC7 | `_on_vision_native_loaded()` does not crash if thread/worker missing, cleared, or already stopped | ✅ Done | `test_on_vision_native_loaded_handles_stale_thread_and_worker_cleanup` (3 parametrize branches) | `124e41a` |
| AC8 | Public DataFrame columns, cache filenames, UI actions, and panel behavior unchanged | ✅ Ongoing | Verified by full regression run after each commit | — |

**Additional bugs found during audit (not in original spec):**

| # | Bug | Status | Notes |
|---|---|---|---|
| Bug A | `_get_vision_similarity_table()` accessed `.keys_list` directly instead of `.keys()` | ✅ Fixed as part of AC3 | `FakeLazySTADict` spy also updated to expose `.keys()` |
| Bug B | `save_classification_to_file()` always applied `vision_id = cluster_id + 1`, corrupting output in Vision-native sessions | ✅ Fixed | One-line `is_vision_only` guard added |

---

## Fragile Zones Touched

Per PLAN.md §1 — these files were modified on this branch and carry
regression risk:

| File | Risk | Required check before re-touching |
|---|---|---|
| `src/analysis/data_manager.py` | Every panel and cache depends on it | Run full test suite; rebase from main first |
| `src/analysis/vision_integration.py` | `LazySTADict` holds open file handle; singleton assumption | Do not instantiate more than once per Vision dir |
| `src/gui/callbacks.py` | Thread lifetime and Qt signal wiring | Run `tests/integration/test_gui_sanity.py` |

---

## Remaining Work

### AC4 — `_load_standard_plot_cache_from_disk()` lock contention

**Root cause:** `pickle.load()` runs inside `with self._standard_plot_lock`,
blocking any concurrent `get_standard_plot_data()` call for 100–500 ms on
a warm cache file.

**Fix pattern:**

1. Acquire lock, check existence + empty cache, release lock.
2. Load pickle outside lock.
3. Acquire lock again, re-check empty (double-check idiom), assign.

**Test to write first:**

```python
# Monkeypatch pickle.load to block for 200ms.
# Spawn thread that calls get_standard_plot_data() during the load.
# Assert _standard_plot_lock is not held during the sleep.
```

---

### AC5 — `get_cell_physics()` per-cell lock escape

**Root cause:** The `with cell_lock:` block closes before the Vision STA
extraction, timecourse computation, and `metrics` dict assembly. Two
concurrent threads can both pass the double-check and both compute the
expensive STA work.

**Fix:** Extend `cell_lock` scope to cover everything through the
`feature_cache` write. The global `_feature_lock` is still used for the
fast-path cache reads and the final write — `cell_lock` just prevents
redundant parallel computation for the same cluster.

**Test to write first:**

```python
# Patch vision_stas.__getitem__ with a slow mock (time.sleep).
# Run two threads through get_cell_physics() for same cluster_id.
# Assert __getitem__ called exactly once.
```

---

## Architecture & Technical Constraints

* **Threading rules:**
  * Do not hold global locks during disk deserialization (AC4).
  * Per-cell lock must cover the full computation including Vision STA
    extraction and final cache write (AC5).
  * Never update Qt UI from background threads.
* **Data contracts:**
  * Similarity table columns and sort order unchanged (AC8).
  * `mea_sim_cache` and `vision_sim_cache` keyed by `int(cluster_id)`.
  * `standard_plot_cache.pkl` schema unchanged.
* **Scope discipline:** This branch is reliability only. No UI polish,
  no schema changes, no new features.

---

## Regression Commands

```bash
# Fast — unit only
conda run -n encore python -m pytest tests/unit/test_critical_production_bugs.py tests/unit/test_data_manager_cache.py -v

# Full suite before pushing
conda run -n encore python -m pytest tests/ -v

# GUI sanity (if mounted)
conda run -n encore python -m pytest tests/integration/test_gui_sanity.py -v
```

---

## Out Of Scope

* RGB normalization expression cleanup in `ei_corr()`
* `torch` import placement or docstring cleanup in `compute_ei()`
* `extract_snippets()` corrupt-file reshape behavior
* `QApplication.processEvents()` in raw data loading
* Sidebar toggle no-op expression
* Garbled comments, root logger calls, logging cleanup
* Any visual redesign, panel behavior change, or new feature work

---

## Commit Log (this branch)

| Hash | Message |
|---|---|
| `124e41a` | fix AC3 — vision sim cache + AC7 safe thread cleanup |
| `141e82d` | fix: add failing AC3/AC7 tests; lock AC6; cache MEA similarity (AC2) |
| `1d7caae` | fix AC1 and AC6, add test for AC2 |
