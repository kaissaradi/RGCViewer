# Phase 1 Handoff — Physics Cache Fix

**Branch:** `claude/physics-cache-loading-98b5op`  
**Base:** `anushka_dev` + `dev-testing` test harness  
**Status:** Pushed, ready for in-app testing

## What changed

Two bugs made every app launch recompute all physics from scratch.

**Bug 1 — Feature cache discarded on reload.**  
Half-built rows (ACG-only, no `_computed` flag) were saved to
`feature_cache.pkl`. The load-time filter deleted every row without
`_computed`. The cache was thrown away every session.

**Fix:** Save only `_computed` rows. Add a version stamp. Reject
unstamped or stale files on load. Guard against an empty session
overwriting a good cache.

**Bug 2 — Grating cache never saved.**  
`grating_computed_cache = {}` reset on every raw-grating load. The
FFT + 1000-shuffle permutation test re-ran over ~900 cells each launch.

**Fix:** Persist `grating_computed_cache.pkl` alongside the other caches.
Load it from disk instead of resetting to empty. Compute only missing
clusters.

**Rebuild button:** File menu → "Rebuild Physics Cache..." clears all
three caches (RAM + disk) and re-runs the warm-up. Overwrites, not just
deletes. Greyed out until a run is loaded.

## Commits (9)

| Hash | Summary |
|------|---------|
| `07b93df` | Change Qt imports to qtpy for Qt6 |
| `3b44262` | Add pure rules for cache persistence |
| `72ad0f5` | Stop loss of physics cache at each start |
| `2f78583` | Keep grating DS/OS results between sessions |
| `16b0ff6` | Put physics warm-up chain in one function |
| `7415c6d` | Add "Rebuild Physics Cache" to File menu |
| `5e8bbcc` | Add tests for physics-cache persistence |
| `113004d` | Correct cache-migration tests for version stamp |
| `15bb25a` | Let full test suite collect without dev-only tools |

## Files touched

| File | What |
|------|------|
| `src/analysis/cache_persistence.py` | NEW — version stamp, filter, round-trip rules |
| `src/analysis/data_manager.py` | Save/load/rebuild for all three caches |
| `src/gui/callbacks.py` | Deduplicated warm-up chain; rebuild wiring |
| `src/gui/main_window.py` | Menu action + qtpy fix |
| `src/gui/panels/population_panel.py` | qtpy fix |
| `src/gui/widgets/widgets.py` | qtpy fix |
| `requirements.txt` | PyQt5 → PyQt6 |
| `pytest.ini` | NEW — mark registration |
| `tests/performance/conftest.py` | NEW — skip guard for pytest-benchmark |
| `tests/unit/test_physics_cache_persistence.py` | NEW — 13 tests |
| `tests/unit/test_data_manager_spatial_pca.py` | Updated for version stamp |
| `tests/performance/test_ram_usage.py` | importorskip for psutil |
| `tests/performance/test_stress.py` | importorskip for psutil |

## Test results

```
193 passed, 20 failed, 20 skipped — 0 errors
```

All 20 failures predate this work. Zero new regressions (verified against
baseline at `9e46998` via worktree diff).

## How to verify in-app

1. Open a run. Let the warm-up finish.
2. Confirm three `.pkl` files exist in the run's kilosort dir.
3. Close and reopen the same run. It should load near-instantly.
4. File → Rebuild Physics Cache. Confirm it clears and rebuilds.
5. Check that UMAP, Population RF, and grating panels look the same.

## What is next

- **Phase 2 (memory):** Add `DataManager.close()` so old data managers
  release `vision_eis` (>500 MB) on run switch. Fix the spike-array
  mmap comment/behaviour mismatch.
- **Phase 3 (launcher):** Double-click scripts for macOS/Windows/Linux.
  `update.sh` wrapping `git pull`.
- **Design pass (deferred):** Bauhaus design principles. All plots match
  the color scheme in light and dark mode.
