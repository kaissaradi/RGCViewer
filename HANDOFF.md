# Handoff — Physics Cache, Memory, then Load Speed

**Branch:** `claude/physics-cache-loading-98b5op`  
**Base:** `anushka_dev` + `dev-testing` test harness  
**Status:** Pushed, ready for in-app testing

Phase 1 makes a reopened run load from cache instead of recomputing.
Phase 2 stops the app from holding memory it no longer needs.
Phase 1b makes opening one run after another fast.

---

# Phase 1 — Physics cache

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

## How to verify in-app

1. Open a run. Let the warm-up finish.
2. Confirm three `.pkl` files exist in the run's kilosort dir.
3. Close and reopen the same run. It should load near-instantly.
4. File → Rebuild Physics Cache. Confirm it clears and rebuilds.
5. Check that UMAP, Population RF, and grating panels look the same.

---

# Phase 2 — Memory

Three fixes. Together they cut steady-state memory for one loaded run by
about 40%, and stop a second run from stacking on top of the first.

**Fix 1 — The spike arrays were copied into RAM.**  
The comment said `mmap_mode="r"` saved 200–800 MB. The call below it did
a full `np.load`.

**Fix:** Map them. `spike_times` and `amplitudes` map read-only.
`spike_clusters` maps copy-on-write, because `update_after_refinement()`
writes to it — copy-on-write keeps the change private, so the Kilosort
file on disk is never touched. If mapping fails (network mount,
unmappable format) `_load_array` falls back to a normal read.

**Fix 2 — The sort scratch was never freed.**  
`_spk_sorted_cls` and `_spk_sorted_t` are two more full-length spike
arrays. `build_cluster_dataframe()` is their only consumer, and it made
four more full-length temporaries of its own.

**Fix:** Release all six once the ISI pass is done. The existing fallback
branch recomputes the sort if anything ever needs it again.

**Fix 3 — Old datasets stayed resident on a run change.**  
Assigning a new `DataManager` over the old one frees nothing: the panels
and the workers still hold it. The Vision EI table alone is >500 MB.

**Fix:** `DataManager.close()` releases the spike arrays, the Vision
tables, the stimulus analyses and every cache.
`_release_previous_dataset()` calls it on both load paths, after stopping
the workers. When a load is still in flight it waits for that thread —
freeing arrays under a running worker ends the process rather than
raising.

Stopping the workers is a correctness fix too: a worker left running
against the old dataset kept emitting results keyed by the old run's
cluster IDs, and the panels applied them to the new run. Cell IDs do not
carry between runs.

## Measured

Synthetic 20-million-spike run (realistic for one hour on the 512 array):

| | Peak at load | Steady state |
|---|---|---|
| Before | 722 MB | 763 MB |
| After | 643 MB | 444 MB |

The Vision EI release on a run change is on top of this.

## How to verify in-app

1. Open a run, then open a different run. Watch RSS — the first run's
   memory should come back, not stack.
2. Refine a cluster (splits write to `spike_clusters`). Confirm it works
   and that `spike_clusters.npy` on disk is unchanged.
3. Switch runs while a load is still in progress. It must not crash.
4. Confirm ISI violation percentages are unchanged from before.

---

## Commits (Phase 2)

| Hash | Summary |
|------|---------|
| `1cdd0d1` | Map the Kilosort spike arrays into memory |
| `5e92239` | Free the sort scratch after the ISI pass |
| `cbbb39b` | Add `DataManager.close()` to release a dataset |
| `cd9686e` | Release the previous dataset on a run change |
| `5d3765c` | Add tests for dataset memory and release |

## Files touched (Phase 2)

| File | What |
|------|------|
| `src/analysis/data_manager.py` | `_load_array`, scratch release, `close()` |
| `src/gui/callbacks.py` | `_release_previous_dataset`; retire returns parked threads |
| `tests/unit/test_dataset_release.py` | NEW — 24 tests |

## Test results

```
217 passed, 20 failed, 20 skipped — 0 errors
```

All 20 failures predate this work. Zero new regressions (verified against
baseline at `9e46998` via worktree diff).

---

# Phase 1b — Load speed

Phase 1 fixed the caches a warm reload *consults*. It did not touch the
work that runs *before* any cache is consulted, and that is what the
remaining wait was made of. Three fixes.

**Fix 1 — The EI correlation cache was read after the EI arrays.**  
`_compute_ei_correlations_if_needed` sanitized every EI in the dataset,
then looked for `ei_corr_dict.pkl`. On a warm run the pickle was there,
so several hundred MB of arrays were touched and thrown away.

**Fix:** Look for the pickle first. Sanitize only on a path that computes.

The pickle now also stores the cell id of each matrix row. This closed a
silent fault: the row order used to be re-derived from the *current* EI
data, so a regenerated `.ei` file was read against the old matrix and gave
plausible, wrong duplicate marks. A cache that does not fit the dataset is
now refused. Files without ids are upgraded in place.

**Fix 2 — The whole `.ei` file was read at every load.**  
One cell is 513 electrodes x 201 samples x 2 floats x 4 bytes = 825 kB, so
900 cells is 742 MB. All of it was read every time, including the tenth
load of the same run. Half was never wanted: only the EI panel reads the
error array, and only for the cell on screen.

**Fix:** `LazyEIDict` reads a cell when a caller asks for it. No change to
`EIReader` was needed — it already builds a byte-offset table and already
has a single-cell read. Mirrors `LazySTADict`, with one reader per thread.
Its cache holds 32 cells, not the 500 `LazySTADict` holds, because 500 EI
cells would be 412 MB.

**Fix 3 — ISI violations were recomputed at every load.**  
The pass puts 20 M spike times in cluster order and runs a diff/mask over
all of them, for a result that does not change.

**Fix:** Persist the percentages in `isi_violations_cache.pkl`, keyed on a
CRC32 of the spike array *contents*. A file-mtime key would be wrong here:
refinement rewrites `spike_clusters` in memory only (copy-on-write, so the
Kilosort file is never touched), so the file is byte-identical after a
split and a mtime key would serve pre-split percentages forever. The CRC32
costs 25 ms and saves about 4 s.

## Measured

Synthetic `.ei`, 512-electrode shape:

| | Open time | Resident |
|---|---|---|
| Before | 0.5 s (local SSD, warm page cache) | 745 MB |
| After | 0.00 s | 0.0 MB |

The file size prediction was exact: 742 MB at 900 cells. The **time** here
is not representative — this machine has a local SSD and no lab mount. On
a network mount at ~100 MB/s the 742 MB is roughly 7 s per load, and that
is the number that matters, but it has to be measured on your machine.

Synthetic Kilosort run, 20 M spikes / 900 clusters:

| | `load_kilosort_data` | `build_cluster_dataframe` | Total |
|---|---|---|---|
| Cold | 6.09 s | 1.87 s | 7.95 s |
| Warm | 4.08 s | 0.01 s | 4.09 s |

Percentages are bit-identical between the two.

## What was investigated and deliberately not done

- **Overlapping the Vision read with the Kilosort load.** This was the
  fourth planned fix. Fix 2 removed its premise — the Vision open is now a
  seek table costing ~0 s, so there is nothing substantial left to run in
  parallel. Adding a second in-flight thread would have interacted with
  Phase 2's load-parking logic for no measured gain.
- **Vectorizing the per-cluster violation loop.** Measured: the Python loop
  is 0.019 s and `np.add.reduceat` is 0.147 s. The "obvious" optimization
  is 8x slower. Left alone.
- **Caching the whole `cluster_df`.** Only the ISI percentages are worth
  caching; the rest of the build is cheap, and pickling a dataframe holding
  Python `set` objects invites column drift against the chirp/contrast
  columns attached later.
- **The queue gate racing the cache load.** It looks like `start_worker`
  reads `standard_plot_cache` before `StandardPlotsWorker.run()` restores
  it. It does not — `build_cluster_dataframe` already calls
  `load_persisted_caches()` on the worker thread first.
- **`StandardPlotsWorker`'s 20 ms per-cluster sleep** costs ~18 s over 900
  clusters but only on a *cold* load, since warm clusters are never queued.
  Worth trimming for first loads; not a warm-reload cost.

## How to verify in-app

1. Open a run, then a second run, then go back to the first. The return
   visit should be clearly faster than the first visit.
2. Check the EI panel, its error shading, and Cell Tracer. Those are the
   real users of the lazy EI reader.
3. Refine a cluster, then reload the run. The ISI percentages must show
   the refinement, not the values from before it.
4. File ▸ Rebuild Physics Cache must still clear everything, including the
   new `isi_violations_cache.pkl`.

## Commits (Phase 1b)

| Hash | Summary |
|------|---------|
| `c6f4de7` | Read the EI correlation cache before the EI arrays |
| `b1fc19a` | Read the .ei file one cell at a time |
| `fb58626` | Keep the ISI violation results between loads |

## Files touched (Phase 1b)

| File | What |
|------|------|
| `src/analysis/vision_integration.py` | `LazyEIDict`; `load_ei_data` returns it; `LazySTADict.close()` |
| `src/analysis/data_manager.py` | EI-corr cache order + row ids; ISI cache; `close()` closes the readers |
| `tests/unit/test_ei_corr_cache.py` | NEW — 18 tests |
| `tests/unit/test_lazy_ei.py` | NEW — 23 tests |
| `tests/unit/test_isi_cache.py` | NEW — 19 tests |

## Test results

```
277 passed, 20 failed, 20 skipped — 0 errors
```

The same 20 failures as before this work. No new ones.

---

## What is next

- **Phase 3 (launcher):** Double-click scripts for macOS/Windows/Linux.
  `update.sh` wrapping `git pull`. Planned but not started — the plan also
  found that `torch` is a ~2 GB dependency used for one median, and that
  `hdbscan` is not imported anywhere in `src/`.
- **Design pass (deferred):** Bauhaus design principles. All plots match
  the color scheme in light and dark mode.
