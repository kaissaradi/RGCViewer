# Specification: Critical Production Bug Fixes

## Metadata

* **Status:** Draft
* **Target Release:** v1.1
* **Primary Developer/Agent:** TBD
* **Target Branch:** `fix/critical-production-bugs`
* **Spec File:** `docs/specs/critical-production-bugs.md`

## Objective

Fix the verified production bugs that can cause crashes, UI freezes, severe repeated work, or unsafe execution. This spec intentionally excludes bug-report items that are only cosmetic, speculative, or not supported by the current code.

## Validation Summary

After re-checking the code:

* **Confirmed critical / must fix:** `.neurons` crash typo, unused similarity caches, standard-plot cache lock contention, `get_cell_physics()` per-cell lock escape, standalone Vision callback missing thread/worker guards.
* **Confirmed hardening / should fix in same branch:** `eval()` in Kilosort params parsing.
* **Not included as critical:** discarded RGB normalization expressions, `compute_ei` docstring/import placement, duplicate `extract_snippets()` reshape except block, raw-load `processEvents()`, sidebar no-op expression, garbled comment, root logger calls.

The excluded items may still be cleaned up later, but they should not be treated as proven critical production bugs.

## User Story

"As a scientist using RGCViewer on large Vision/Kilosort datasets, I want dataset loading, similarity lookup, and physics cache computation to avoid crashes, redundant work, and UI stalls so that I can safely inspect cells without restarting the app."

## Acceptance Criteria

* **AC1:** If `.neurons` loading raises an unexpected exception, `load_neurons_data()` logs it and returns `None`; it must not raise `NameError`.
* **AC2:** Repeated MEA similarity requests for the same cluster return from `mea_sim_cache` without rebuilding the table.
* **AC3:** Repeated Vision similarity requests for the same cluster return from `vision_sim_cache` without rereading all lazy STAs.
* **AC4:** `_load_standard_plot_cache_from_disk()` does not hold `_standard_plot_lock` while executing `pickle.load()`.
* **AC5:** Concurrent calls to `get_cell_physics()` for the same cluster perform the full physics computation once and return the same cached result to all callers.
* **AC6:** `_load_kilosort_params()` uses `ast.literal_eval()` instead of `eval()` and still supports normal Kilosort literals such as numbers, strings, lists, and tuples.
* **AC7:** `_on_vision_native_loaded()` does not crash if `vision_load_thread` or `vision_load_worker` is missing, already stopped, or already cleared.
* **AC8:** Existing public DataFrame columns, cache file names, UI actions, and panel behavior remain unchanged.

## Current Progress

* **AC1:** Implemented in `load_neurons_data()`; no regression test added in this slice by request.
* **AC2:** Implemented and covered by `test_mea_similarity_table_reuses_cluster_cache`.
* **AC3:** Failing regression test added (`test_vision_similarity_table_reuses_cluster_cache`); production fix is still pending.
* **AC6:** Implemented and covered by `test_kilosort_params_uses_literal_eval_without_executing_code`.

## Architecture & Technical Constraints

* **Files Modified:**
  * `src/analysis/vision_integration.py`
  * `src/analysis/data_manager.py`
  * `src/gui/callbacks.py`
* **Data Contracts:**
  * Similarity tables must keep the existing columns and sort order.
  * `mea_sim_cache` and `vision_sim_cache` should be keyed by `cluster_id`, matching the current cache comments and current `get_similarity_table()` API.
  * `standard_plot_cache.pkl` remains a pickle dictionary keyed by cluster ID.
* **Threading Rules:**
  * Do not hold global locks during disk deserialization.
  * Keep the per-cell physics lock around the full same-cluster computation, including Vision STA extraction and final cache write.
  * Do not update Qt UI from background threads.
* **Safety Rules:**
  * Do not broaden this branch into UI polish or unrelated cleanup.
  * Tests that validate cache computation must use temporary directories or fresh in-memory `DataManager` instances.

## Implementation Plan

* Fix `load_neurons_data()` so the broad exception path returns `None`.
* Add early cache checks and final cache writes in both similarity table helpers.
* Move standard plot pickle loading outside `_standard_plot_lock`, then re-check under lock before assignment.
* Restructure `get_cell_physics()` so the per-cell lock covers the double-check, standard plot lookup, Vision STA/timecourse extraction, metrics creation, and `feature_cache` write.
* Replace `eval()` with `ast.literal_eval()` in Kilosort params parsing; fall back to stripped strings for non-literal values.
* Mirror the safe cleanup pattern from `_on_vision_loaded()` in `_on_vision_native_loaded()`, including worker cleanup and nulling references.

## Test Plan

* **Unit:** Add `tests/unit/test_critical_production_bugs.py`.
  * Mock a failing `NeuronsReader`; assert `load_neurons_data()` returns `None`.
  * Create a minimal MEA similarity setup; call twice and assert the second call returns cached data without rebuilding. **Done.**
  * Create a lazy fake `vision_stas`; call twice and assert the second call does not access every STA again. **Test written; currently failing until AC3 is implemented.**
  * Monkeypatch pickle loading to block while another thread probes `_standard_plot_lock`; assert the lock is not held during load.
  * Run two threads through `get_cell_physics()` for the same cluster; assert expensive mocked dependencies are called once.
  * Parse temp `params.py` files containing normal literals and a malicious/non-literal expression; assert no code execution. **Done.**
* **Qt Callback Unit:** Add or extend callback tests with a mock main window.
  * `_on_vision_native_loaded()` handles missing thread and worker attributes without raising.
  * Successful and failed load paths re-enable the central widget as before.
* **Regression Run:**
  * `conda run -n rgcviewer python -m pytest tests/unit/test_critical_production_bugs.py tests/unit/test_data_manager_cache.py`
  * If available, run the relevant GUI sanity tests: `conda run -n rgcviewer python -m pytest tests/integration/test_gui_sanity.py`

## Out Of Scope

* Changing STA color metric formulas.
* Moving `torch` import or docstring cleanup in `compute_ei()`.
* Changing `extract_snippets()` corrupt-file behavior.
* Removing `QApplication.processEvents()` from raw data loading.
* Sidebar toggle cleanup, comment cleanup, or logging cleanup.
* Any visual redesign, panel behavior change, cache schema change, or new feature work.

## Notes For Implementation

Write failing tests first, per `docs/AGENTS.md`. Create the branch before editing code. Keep the branch focused: this spec is about production-critical reliability only, not the full original bug-report cleanup list.
