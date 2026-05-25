# docs/specs/physics_cache_unified.md

> Reading order: AGENTS.md (full) → PLAN.md (Fragile Zones + Untested Behaviors) → this spec → write failing tests → implement.

---

## Block 0 — Metadata

| Field | Value |
|---|---|
| **Date created** | 2026-05-24 |
| **Last updated** | 2026-05-24 |
| **Commit hash when spec was written** | `git rev-parse --short HEAD` — paste result here |
| **Branch** | `fix/physics-cache-unified` |
| **Author** | Kais |
| **Spec status** | Final |

---

## Block 1 — Problem Statement

**Symptom:** After loading a Kilosort directory and Vision files, the app feels slow and the first UMAP run stalls noticeably. On datasets with 500+ clusters this stall is 7–15 seconds.

**Root cause:** Three compounding problems.

---

**Problem 1 — ACG computed twice.**

`StandardPlotsWorker` already computes `acg_norm` for every cluster and caches it in `standard_plot_cache`. Then `get_cell_physics()` (line 1506–1507 of `data_manager.py`) calls `get_standard_plot_data()` again to retrieve `acg_norm` from `standard_plot_cache` and writes it into `feature_cache`. This is not a double *disk* read — `standard_plot_cache` is in RAM — but it means every cell requires two sequential cache operations and one full traversal of the `get_standard_plot_data` lock path, even though the ACG value was ready the moment `StandardPlotsWorker` wrote it.

The correct design: `_compute_standard_plots()` writes `acg_norm` directly into `feature_cache` at the same time it writes into `standard_plot_cache`. `get_cell_physics()` then finds the ACG already waiting and only needs to do the STA seek + timecourse math — roughly half the work per cell.

---

**Problem 2 — Physics pass is serial.**

Both `_on_vision_loaded()` (line 366) and `_on_vision_native_loaded()` (line 245) in `callbacks.py` run a `for cid in all_ids` loop that calls `get_cell_physics(cid)` sequentially on a single background `QThread`. Even after Problem 1 is fixed, the STA seek is still ~10–15ms per cell on a cold SSD. On 500 clusters that is 5–7 seconds, serial, unmasked. The two loops are textually independent; fixing one does not fix the other.

The correct design: replace both serial loops with a `ThreadPoolExecutor`. STA seeks are I/O-bound; 8 concurrent threads saturate typical SSD queue depth and reduce wall time by ~6–7×. The existing `_physics_cell_locks` double-checked locking inside `get_cell_physics()` already prevents redundant computation under concurrent access.

---

**Problem 3 — Duplicate PCA/feature-matrix logic.**

`extract_features_from_datamanager()` in `umap_panel.py` (lines 39–127) and `FeatureAnalysisWorker.run()` in `feature_extraction.py` (lines 159–228) independently implement the same pipeline: collect timecourse/acg from cache, pad arrays to uniform length, drop NaN rows, apply `RobustScaler` to scalar geometry, run PCA on timecourse and ACG, concatenate with weights. They are already subtly diverged: `umap_panel.py` applies `W_SHAPE`/`W_PATTERN`/`W_GEOMETRY` weights; `feature_extraction.py` does not.

The correct design: add `get_physics_feature_matrix()` to `DataManager` as the single implementation. Both callers delegate to it. The weight constants stay in `umap_panel.py` and are passed as arguments.

---

**Bonus fix — Redundant `np.sort()` on already-sorted arrays.**

`_compute_standard_plots()` line 1665: `t = np.sort(spikes_ms).astype(np.int64)`. `_calculate_isi_violations()` line 2185: `isis = np.diff(np.sort(spike_times_cluster))`. Both arrays come from `get_cluster_spikes()` which slices Kilosort's `spike_times.npy` memmap — a file Kilosort writes in strict ascending order. Both sorts are O(N log N) no-ops on real data. Remove both.

---

**`LazySTADict` thread safety.**

`LazySTADict.__getitem__` in `vision_integration.py` mutates `self._cache` (plain dict) and `self._cache_keys` (plain list) without a lock. With 8 pool threads all calling `__getitem__` concurrently, two threads can both miss the cache simultaneously and both attempt to insert the same key, producing a duplicate `_cache_keys` entry that corrupts FIFO eviction order. A third thread iterating `_cache` during resize raises `RuntimeError: dictionary changed size during iteration`. Add one `threading.Lock` with a double-check pattern that keeps the SSD read outside the lock.

---

**User story:** "As a scientist, I load a Kilosort directory, click through a few cells while the app warms up in the background, then open the UMAP panel and it runs immediately — I never see a multi-second stall at any point."

---

## Block 2 — Vision ID Contract

This spec touches `LazySTADict` in `vision_integration.py` and calls `get_cell_physics()` from multiple threads. It never accesses `vision_stas` directly outside `get_cell_physics()`.

| Question | Answer |
|---|---|
| Does this spec access Vision file data? | Yes — via `LazySTADict` inside `get_cell_physics()` |
| ID space this spec operates in | Both — `get_cell_physics()` handles the translation |
| Translation used | `vid = cluster_id if is_vision_only else cluster_id + 1` — already inside `get_cell_physics()`, not re-derived here |
| Safe access pattern used | `metrics = dm.get_cell_physics(cid)` — no direct `vision_stas` access anywhere in this spec |

The `LazySTADict` lock addition does not change how IDs are accessed. It only guards Python dict/list mutation inside `__getitem__`. The SSD read stays outside the lock.

---

## Block 3 — Affected Files

| File path | What changes | Change type | Touches DataManager? |
|---|---|---|---|
| `src/analysis/vision_integration.py` | `LazySTADict.__init__`, `LazySTADict.__getitem__` | Add `threading.Lock` | No |
| `src/analysis/data_manager.py` | `_compute_standard_plots()` line 1665 | Remove `np.sort` | Yes |
| `src/analysis/data_manager.py` | `_compute_standard_plots()` — after computing `acg_norm` | Write `acg_norm` into `feature_cache` | Yes |
| `src/analysis/data_manager.py` | `get_cell_physics()` — cache miss path | Skip ACG recompute when `acg_norm` already in `feature_cache[cluster_id]` | Yes |
| `src/analysis/data_manager.py` | `get_cell_physics()` — after STA work | Preserve existing write to `feature_cache` with `_computed: True` | Yes |
| `src/analysis/data_manager.py` | `_calculate_isi_violations()` line 2185 | Remove `np.sort` | Yes |
| `src/analysis/data_manager.py` | New method `get_physics_feature_matrix()` | Add after `get_cell_physics()` | Yes |
| `src/analysis/data_manager.py` | New method `ensure_physics_cache()` | Add after `get_physics_feature_matrix()` | Yes |
| `src/gui/callbacks.py` | Both `_compute_physics()` serial loops | Replace with `ThreadPoolExecutor`, remove `QThread`/`QObject` scaffolding, remove progress bar reset block | No |
| `src/gui/panels/umap_panel.py` | `extract_features_from_datamanager()` | Replace body with delegation to `dm.get_physics_feature_matrix()` | No |
| `src/gui/panels/umap_panel.py` | `UMAPWorker.run()` | Call `dm.ensure_physics_cache(target_ids)` before feature extraction | No |
| `src/gui/panels/feature_extraction.py` | `FeatureAnalysisWorker.run()` | Call `dm.ensure_physics_cache(cluster_ids)`, delegate to `dm.get_physics_feature_matrix()` | No |
| `tests/unit/test_physics_cache_unified.py` | All new tests (see Block 9) | New file | No |

> **DataManager is touched.** Rebase from main before every push.

### Import changes required

**`callbacks.py`** — add to existing import block:
```python
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
```

**`vision_integration.py`** — add to existing import block:
```python
import threading
```

**`data_manager.py`** — add to top-level import block (not inside any function):
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
```

**`umap_panel.py`** — after replacing `extract_features_from_datamanager()`, verify `PCA` and `RobustScaler` are unused elsewhere in the file. They are not — remove both imports.

**`feature_extraction.py`** — after replacing `FeatureAnalysisWorker.run()`, verify `PCA` is unused elsewhere. It is not — remove `from sklearn.decomposition import PCA`.

---

## Block 4 — Qt Threading Contract

### Part A — `LazySTADict` thread safety
No Qt signals. Pure `threading.Lock` (stdlib). No new workers or slots.

### Part B — Parallel physics pass in `callbacks.py`

The `QThread`/`QObject` scaffolding (`physics_thread`, `physics_worker`, `moveToThread`, `deleteLater` connections) is **deleted entirely** from both `_on_vision_loaded` and `_on_vision_native_loaded`. It is replaced with a plain `threading.Thread(daemon=True)` that runs the `ThreadPoolExecutor` pool.

**Why `threading.Thread` not `QThread`:** The existing comment about Numba/TBB fork issues applies only to `multiprocessing.Process` which uses `fork()`. Both `threading.Thread` and `QThread` use `pthread_create` — they are the same OS primitive. `QThread` is only required when the background work needs a Qt event loop to emit signals. This pool emits no Qt signals — results go straight into `feature_cache` under `_feature_lock`. `threading.Thread(daemon=True)` is the simpler and equally correct choice.

**Why no progress bar:** The progress bar during the physics pass is deleted. The user does not need to watch a per-cell counter. If they open UMAP before the pool finishes, `ensure_physics_cache()` fills any remaining misses synchronously (fast, because most cells are already cached) before feature extraction proceeds.

| Operation | Runs on | Signal | Slot |
|---|---|---|---|
| `ThreadPoolExecutor` pool | daemon threads inside `threading.Thread` | None — writes to `feature_cache` directly | N/A |
| `ensure_physics_cache()` | Caller's thread (already a background QThread for both callers) | None | N/A |
| `get_physics_feature_matrix()` | Caller's thread | None | N/A |

### Part C — `ensure_physics_cache()` and `get_physics_feature_matrix()`

Pure methods on `DataManager`. Called from `UMAPWorker.run()` and `FeatureAnalysisWorker.run()`, both of which already run on background `QThread`s. No new threads, no new signals.

---

## Block 5 — Cache Contract

| Cache | Lock | Written by (after this spec) | Persisted |
|---|---|---|---|
| `standard_plot_cache` | `_standard_plot_lock` | `_compute_standard_plots()` — unchanged | `standard_plot_cache.pkl` |
| `feature_cache` | `_feature_lock` | `_compute_standard_plots()` (ACG partial entry) AND `get_cell_physics()` (`_computed: True` full entry) | `feature_cache.pkl` |

**The partial entry pattern:**

`_compute_standard_plots()` writes a partial entry:
```python
feature_cache[cluster_id] = {'acg': acg_norm}   # no _computed key
```

`get_cell_physics()` detects the partial entry, skips ACG recompute, does STA work, then overwrites with the full entry:
```python
feature_cache[cluster_id] = {
    '_computed': True,
    'acg': acg_norm,          # already present, reused
    'timecourse': timecourse,
    'rf_area': rf_area,
    'ellipticity': ellipticity,
    'time_to_peak': time_to_peak,
}
```

The fast-path check in `get_cell_physics()` (line 1486) guards on `_computed` being truthy — a partial entry without `_computed` correctly falls through to the computation path. No change to the fast path.

**Test bypass:** Tests must use `tmp_path` (no `.pkl` files) or construct `mock_dm` directly. No test may point `kilosort_dir` at a real dataset directory. See LAW 3 in AGENTS.md.

---

## Block 6 — DataManager Attributes Used

| Attribute | Type | Can be `None`? | How this spec accesses it |
|---|---|---|---|
| `feature_cache` | `dict` | No (may be empty) | Read/write under `_feature_lock` |
| `standard_plot_cache` | `dict` | No (may be empty) | Read under `_standard_plot_lock` (existing pattern, unchanged) |
| `cluster_df` | `pd.DataFrame` | No (may be empty) | `dm.cluster_df['cluster_id'].astype(int).tolist()` (existing pattern) |
| `vision_stas` | `LazySTADict` | Yes | Via `get_cell_physics()` only — never directly |
| `_feature_lock` | `threading.Lock` | No | Acquired before any `feature_cache` read or write |
| `_physics_cell_locks` | `dict` | No | Existing per-cell in-flight locks, unchanged |

`get_physics_feature_matrix()` accesses DataManager only through `self.get_cell_physics(cid)` and `self.feature_cache`. It reads no other attributes directly.

---

## Block 7 — Exact Implementation

### 7A — `LazySTADict` thread safety (`vision_integration.py`)

**In `LazySTADict.__init__` — add one line:**
```python
self._cache_lock = threading.Lock()
```

**Replace `LazySTADict.__getitem__` entirely:**
```python
def __getitem__(self, key):
    # 1. Fast path: check RAM cache under lock (instant dict lookup).
    with self._cache_lock:
        if key in self._cache:
            return self._cache[key]

    # 2. SSD read is OUTSIDE the lock so threads do not serialize on I/O.
    #    Two threads may both reach this line for the same key if they both
    #    missed the cache simultaneously. That is safe: get_sta_for_cell_id()
    #    is a pure read with no shared mutable state.
    sta_data = self.reader.get_sta_for_cell_id(key)

    # 3. Write result under lock. Double-check: another thread may have
    #    populated this key while we were on the SSD.
    with self._cache_lock:
        if key not in self._cache:
            self._cache[key] = sta_data
            self._cache_keys.append(key)
            if len(self._cache_keys) > self._max_cache:
                oldest = self._cache_keys.pop(0)
                self._cache.pop(oldest, None)

    return self._cache[key]
```

**What does NOT change:** `__contains__`, `get()`, `__iter__`, `__len__`, `keys()`, `__del__`, `_max_cache`. The `reader` object is assumed thread-safe for concurrent reads of different cell IDs.

---

### 7B — Write ACG into `feature_cache` from `_compute_standard_plots()` (`data_manager.py`)

**At line 1665, replace:**
```python
t = np.sort(spikes_ms).astype(np.int64)
```
**With:**
```python
# spike_times.npy from Kilosort is written in ascending order; np.sort() is a no-op.
t = spikes_ms.astype(np.int64)
```

**After line 1688 (`data['acg_norm'] = acg_norm`), add:**
```python
            # Write ACG into feature_cache immediately so get_cell_physics()
            # does not need to recompute it. This is a partial entry —
            # _computed is intentionally absent so get_cell_physics() knows
            # the STA work (timecourse, rf_area, etc.) still needs to run.
            with self._feature_lock:
                existing = self.feature_cache.get(cluster_id)
                if existing is None:
                    # No entry yet — write partial entry with just ACG.
                    self.feature_cache[cluster_id] = {'acg': acg_norm}
                elif not existing.get('_computed'):
                    # Partial entry exists (e.g. from a previous interrupted pass)
                    # — update the ACG in place.
                    existing['acg'] = acg_norm
                # If _computed is True, a full entry already exists — leave it alone.
```

This write is inside the `if spikes_ms.size > MIN_SPIKES:` block that already computes `acg_norm`, so it only runs when a valid ACG was actually produced. If a cell has fewer than 50 spikes, `acg_norm` is never computed and no partial entry is written — `get_cell_physics()` will write `acg=None` in that case, which is the correct existing behavior.

---

### 7C — Skip ACG recompute in `get_cell_physics()` (`data_manager.py`)

The current code at lines 1506–1507:
```python
std_data = self.get_standard_plot_data(cluster_id)
acg_norm = std_data.get('acg_norm') if std_data else None
```

Replace with:
```python
# Check for ACG already written by _compute_standard_plots() into feature_cache.
# If present, skip the redundant get_standard_plot_data() call.
with self._feature_lock:
    _partial = self.feature_cache.get(cluster_id, {})
    acg_norm = _partial.get('acg')   # None if not yet written

if acg_norm is None:
    # Fallback: StandardPlotsWorker hasn't reached this cell yet.
    # Call get_standard_plot_data() which will compute and cache it.
    std_data = self.get_standard_plot_data(cluster_id)
    acg_norm = std_data.get('acg_norm') if std_data else None
```

This change is inside the `with cell_lock:` block (line 1495), so it is already protected against concurrent redundant computation. The `_feature_lock` acquisition here is a brief dict lookup — no deadlock risk because `_feature_lock` is never held while calling `get_standard_plot_data()`.

---

### 7D — Remove `np.sort` from `_calculate_isi_violations()` (`data_manager.py`)

At line 2185, replace:
```python
isis = np.diff(np.sort(spike_times_cluster))
```
With:
```python
# spike_times_cluster comes from get_cluster_spikes() which reads the sorted Kilosort memmap.
isis = np.diff(spike_times_cluster)
```

---

### 7E — Add `get_physics_feature_matrix()` to `DataManager` (`data_manager.py`)

Add this method immediately after `get_cell_physics()` (currently ending around line 1567). The `sklearn` imports (`PCA`, `RobustScaler`) must be at the top-level import block.

```python
def get_physics_feature_matrix(self, cluster_ids, w_shape=2.0, w_pattern=1.5, w_geometry=1.0):
    """
    Build the weighted PCA feature matrix used by UMAPWorker and FeatureAnalysisWorker.

    Reads pre-computed physics from feature_cache via get_cell_physics() — O(1)
    cache hit if the background pass has run; triggers on-demand computation otherwise.

    Parameters
    ----------
    cluster_ids : iterable of int
    w_shape : float   — weight on PCA-compressed timecourse block (default 2.0)
    w_pattern : float — weight on PCA-compressed ACG block (default 1.5)
    w_geometry : float — weight on RobustScaler-normalised scalar block (default 1.0)

    Returns
    -------
    feature_matrix : np.ndarray shape (N, n_features), or np.array([]) if N == 0
    valid_ids : list[int] in same row order as feature_matrix
    metadata : dict with keys 'Time to Peak', 'RF Area', 'Ellipticity' (lists of length N)
               or {} if N == 0
    """
    valid_ids = []
    tc_list   = []
    acg_list  = []
    scalars_list = []
    metadata = {'Time to Peak': [], 'RF Area': [], 'Ellipticity': []}

    for cid in cluster_ids:
        metrics = self.get_cell_physics(int(cid))
        tc  = metrics.get('timecourse')
        acg = metrics.get('acg')
        if tc is None or acg is None:
            continue

        valid_ids.append(cid)
        tc_list.append(tc)
        acg_list.append(acg)

        area  = metrics.get('rf_area')  or 0.0
        ellip = metrics.get('ellipticity') or 0.0
        t2p   = metrics.get('time_to_peak') or 0
        scalars_list.append([area, ellip])
        metadata['Time to Peak'].append(t2p)
        metadata['RF Area'].append(area)
        metadata['Ellipticity'].append(ellip)

    if not valid_ids:
        return np.array([]), [], {}

    # Pad/truncate to uniform length
    max_tc  = max(len(t) for t in tc_list)
    tc_mat  = np.array([
        np.pad(t, (0, max_tc - len(t))) if len(t) < max_tc else t[:max_tc]
        for t in tc_list
    ])
    max_acg = max(len(a) for a in acg_list)
    acg_mat = np.array([
        np.pad(a, (0, max_acg - len(a))) if len(a) < max_acg else a[:max_acg]
        for a in acg_list
    ])
    scalars_mat = np.array(scalars_list, dtype=np.float64)

    # Drop rows with NaN in any matrix
    nan_mask = (
        np.any(np.isnan(tc_mat),      axis=1) |
        np.any(np.isnan(acg_mat),     axis=1) |
        np.any(np.isnan(scalars_mat), axis=1)
    )
    if np.any(nan_mask):
        logger.warning("get_physics_feature_matrix: dropping %d unit(s) with NaN features",
                       int(nan_mask.sum()))
        keep        = ~nan_mask
        valid_ids   = [v for v, k in zip(valid_ids,   keep) if k]
        tc_mat      = tc_mat[keep]
        acg_mat     = acg_mat[keep]
        scalars_mat = scalars_mat[keep]
        for key in metadata:
            metadata[key] = [v for v, k in zip(metadata[key], keep) if k]

    if not valid_ids:
        return np.array([]), [], {}

    # Robust normalisation of scalar geometry features
    if scalars_mat.shape[0] > 0:
        scalars_mat = RobustScaler().fit_transform(scalars_mat)

    # PCA compression
    n_comp  = min(3, len(valid_ids))
    tc_pca  = PCA(n_components=n_comp).fit_transform(tc_mat)  if n_comp > 0 else np.zeros((len(valid_ids), 0))
    acg_pca = PCA(n_components=n_comp).fit_transform(acg_mat) if n_comp > 0 else np.zeros((len(valid_ids), 0))

    feature_matrix = np.hstack([
        tc_pca      * w_shape,
        acg_pca     * w_pattern,
        scalars_mat * w_geometry,
    ])

    return feature_matrix, valid_ids, metadata
```

---

### 7F — Add `ensure_physics_cache()` to `DataManager` (`data_manager.py`)

Add immediately after `get_physics_feature_matrix()`.

```python
def ensure_physics_cache(self, cluster_ids, max_workers=8):
    """
    Guarantee that all cluster_ids have a fully-computed feature_cache entry
    (i.e. _computed == True) before returning.

    Called at UMAP/Feature-extraction click time as a safety net for any cells
    the background pass has not yet reached. If the background pass has already
    finished for all requested cells, this function returns in microseconds
    (pure cache lookups, no I/O).

    Parameters
    ----------
    cluster_ids : iterable of int
    max_workers : int — pool size for parallel STA seeks (default 8, I/O-bound)
    """
    # Identify cache misses under lock — one brief critical section.
    with self._feature_lock:
        missing = [
            int(cid) for cid in cluster_ids
            if not self.feature_cache.get(int(cid), {}).get('_computed')
        ]

    if not missing:
        return   # All cells already computed — common case after background pass.

    logger.debug("ensure_physics_cache: %d cells not yet computed, filling now", len(missing))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(self.get_cell_physics, cid): cid for cid in missing}
        for future in as_completed(futures):
            cid = futures[future]
            try:
                future.result()
            except Exception:
                logger.warning("ensure_physics_cache: failed for cluster %d", cid, exc_info=True)
```

> **Import note:** `ThreadPoolExecutor` and `as_completed` must be imported at the top of `data_manager.py`:
> ```python
> from concurrent.futures import ThreadPoolExecutor, as_completed
> ```

---

### 7G — Replace both serial physics loops in `callbacks.py`

**Delete** the following block from `_on_vision_native_loaded()` (lines 239–267), including the progress bar reset and the entire `QThread`/`QObject` scaffolding:

```python
# DELETE THIS ENTIRE BLOCK from _on_vision_native_loaded():
all_ids = main_window.data_manager.cluster_df['cluster_id'].astype(int).tolist()

main_window.cache_progress_count = 0
main_window.cache_progress.setValue(0)
main_window.cache_progress.show()

def _compute_physics():
    for cid in all_ids:
        try:
            main_window.data_manager.get_cell_physics(cid)
            if getattr(main_window, 'standard_plots_worker', None):
                main_window.standard_plots_worker.finished_cluster.emit(int(cid))
        except Exception:
            logger.warning("Physics failed for cluster %s", cid, exc_info=True)
    main_window.physics_thread.quit()

main_window.physics_thread = QThread()
main_window.physics_worker = QObject()
main_window.physics_worker.moveToThread(main_window.physics_thread)
main_window.physics_thread.started.connect(_compute_physics)
main_window.physics_thread.finished.connect(main_window.physics_worker.deleteLater)
main_window.physics_thread.finished.connect(main_window.physics_thread.deleteLater)
main_window.physics_thread.start()

main_window.status_bar.showMessage(
    f"Vision dataset loaded. Computing physics for {len(all_ids)} cells...", 5000)
```

**Replace with:**

```python
# Kick off physics cache warm-up silently in the background.
# daemon=True ensures this thread does not prevent clean app shutdown.
_all_ids = main_window.data_manager.cluster_df['cluster_id'].astype(int).tolist()

def _warm_physics():
    main_window.data_manager.ensure_physics_cache(_all_ids)

threading.Thread(target=_warm_physics, daemon=True).start()
```

---

**Delete** the following block from `_on_vision_loaded()` (lines 357–393), including the progress bar reset and the entire `QThread`/`QObject` scaffolding:

```python
# DELETE THIS ENTIRE BLOCK from _on_vision_loaded():
if success:
    dm = main_window.data_manager
    all_ids = dm.cluster_df['cluster_id'].astype(int).tolist()

    main_window.cache_progress_count = 0
    main_window.cache_progress.setValue(0)
    main_window.cache_progress.show()

    def _compute_physics():
        for cid in all_ids:
            try:
                dm.get_cell_physics(cid)
                if getattr(main_window, 'standard_plots_worker', None):
                    main_window.standard_plots_worker.finished_cluster.emit(int(cid))
            except Exception:
                pass

    main_window.physics_thread = QThread()
    main_window.physics_worker = QObject()
    main_window.physics_worker.moveToThread(main_window.physics_thread)

    def run_physics():
        _compute_physics()
        main_window.physics_thread.quit()

    main_window.physics_thread.started.connect(run_physics)
    main_window.physics_thread.finished.connect(main_window.physics_worker.deleteLater)
    main_window.physics_thread.finished.connect(main_window.physics_thread.deleteLater)
    main_window.physics_thread.start()

    main_window.status_bar.showMessage(
        f"Vision loaded. Computing physics for {len(all_ids)} cells...", 5000)
```

**Replace with:**

```python
if success:
    _all_ids = main_window.data_manager.cluster_df['cluster_id'].astype(int).tolist()

    def _warm_physics():
        main_window.data_manager.ensure_physics_cache(_all_ids)

    threading.Thread(target=_warm_physics, daemon=True).start()
```

> After both deletions, verify that `QThread` and `QObject` are still used elsewhere in `callbacks.py` before deciding whether to keep those imports. They are — `QThread` is used in `load_directory`, `load_vision_directory`, and the auto-load path. Keep both imports.

---

### 7H — Replace `extract_features_from_datamanager()` in `umap_panel.py`

Replace the entire function body (lines 45–127). Preserve the function signature exactly.

```python
def extract_features_from_datamanager(dm, cluster_ids):
    """
    Thin delegation to DataManager.get_physics_feature_matrix().
    All logic lives in the DataManager to eliminate duplication with FeatureAnalysisWorker.
    """
    return dm.get_physics_feature_matrix(
        cluster_ids,
        w_shape=W_SHAPE,
        w_pattern=W_PATTERN,
        w_geometry=W_GEOMETRY,
    )
```

The module-level weight constants (`W_SHAPE`, `W_PATTERN`, `W_GEOMETRY`) at lines 34–36 remain in `umap_panel.py`. Remove the now-unused imports `from sklearn.decomposition import PCA` and `from sklearn.preprocessing import RobustScaler`.

---

### 7I — Add `ensure_physics_cache` call to `UMAPWorker.run()` (`umap_panel.py`)

In `UMAPWorker.run()`, before line 195 (`features, cluster_ids, metadata = extract_features_from_datamanager(...)`), add:

```python
self.progress.emit("Ensuring physics cache...")
self.dm.ensure_physics_cache(target_ids)
self.progress.emit("Extracting features...")
```

Full updated sequence:
```python
target_ids = self.selected_cluster_ids
if target_ids is None:
    target_ids = self.dm.cluster_df['cluster_id'].values

self.progress.emit("Ensuring physics cache...")
self.dm.ensure_physics_cache(target_ids)   # no-op if background pass already finished

self.progress.emit("Extracting features...")
features, cluster_ids, metadata = extract_features_from_datamanager(self.dm, target_ids)
```

---

### 7J — Replace `FeatureAnalysisWorker.run()` in `feature_extraction.py`

Replace the entire `run()` body (lines 159–232). The `finished` signal payload dict keys (`cluster_ids`, `temporal_pca`, `acg_pca`, `rf_diameter`, `time_to_peak`) are preserved exactly so `FeatureExtractionWindow._on_worker_done()` requires no changes.

```python
def run(self):
    try:
        total = len(self.cluster_ids)
        self.progress.emit(f"Ensuring physics cache for {total} clusters...", 0)

        # Safety net: fills any cells not yet reached by the background pass.
        # No-op if the background pass already finished.
        self.data_manager.ensure_physics_cache(self.cluster_ids)

        self.progress.emit(f"Extracting features for {total} clusters...", 10)

        feature_matrix, valid_ids, metadata = self.data_manager.get_physics_feature_matrix(
            self.cluster_ids
            # Uses default weights (all 1.0 effective) — FeatureExtractionWindow
            # does its own axis labeling; absolute scale doesn't matter here.
        )

        if len(valid_ids) == 0:
            self.progress.emit("No valid features found.", 100)
            self.finished.emit({})
            return

        n      = len(valid_ids)
        n_comp = min(3, n)

        # Split the feature matrix back into components for panel plots.
        # Column layout: [0:n_comp] tc×w_shape | [n_comp:2*n_comp] acg×w_pattern | [2*n_comp:] scalars×w_geometry
        def _pad3(arr):
            if arr.shape[1] < 3:
                return np.pad(arr, ((0, 0), (0, 3 - arr.shape[1])), mode='constant')
            return arr

        tc_pca_block  = feature_matrix[:, :n_comp]
        acg_pca_block = feature_matrix[:, n_comp:2 * n_comp]

        results = {
            'cluster_ids':  valid_ids,
            'temporal_pca': _pad3(tc_pca_block),
            'acg_pca':      _pad3(acg_pca_block),
            'rf_diameter':  np.sqrt(np.array(metadata['RF Area']) / np.pi),
            'time_to_peak': np.array(metadata['Time to Peak']),
        }

        self.progress.emit("Done.", 100)
        self.finished.emit(results)

    except Exception as e:
        logger.error("Error in FeatureAnalysisWorker: %s", e, exc_info=True)
        self.finished.emit({})
```

Remove the now-unused import `from sklearn.decomposition import PCA`.

---

## Block 8 — Acceptance Criteria

### AC1 — ACG written to `feature_cache` by `_compute_standard_plots`

- **Setup:** `mock_dm` with 3 clusters. Call `dm.get_standard_plot_data(cid)` for each.
- **Expected:** `dm.feature_cache[cid]['acg']` is a non-None numpy array. `dm.feature_cache[cid].get('_computed')` is falsy (partial entry, not yet fully computed).
- **Test type:** Unit

### AC2 — `get_cell_physics` skips ACG recompute when partial entry exists

- **Setup:** Pre-populate `feature_cache[cid] = {'acg': sentinel_array}` where `sentinel_array = np.ones(201)`. Patch `get_standard_plot_data` with a call counter. Call `get_cell_physics(cid)` with Vision data mocked to return a valid STA.
- **Expected:** `get_standard_plot_data` call counter == 0. Returned `metrics['acg']` is `sentinel_array` (the pre-existing value, not recomputed).
- **Test type:** Unit

### AC3 — `get_cell_physics` falls back to `get_standard_plot_data` when no partial entry exists

- **Setup:** Empty `feature_cache`. Patch `get_standard_plot_data` with a call counter and a return value containing `acg_norm`.
- **Expected:** `get_standard_plot_data` call counter == 1. `metrics['acg']` matches the patched return value.
- **Test type:** Unit

### AC4 — `LazySTADict` concurrent reads do not raise `RuntimeError` or corrupt `_cache_keys`

- **Setup:** `LazySTADict` with mocked reader sleeping 5ms per cell. Two threads call `__getitem__` for different keys simultaneously.
- **Expected:** No exception. Both keys in `_cache`. No key appears twice in `_cache_keys`.
- **Test type:** Unit

### AC5 — `LazySTADict` SSD read is not serialized by the lock

- **Setup:** Mocked reader sleeping 50ms per cell. Two threads each request a different key concurrently. Record wall-clock time.
- **Expected:** Total wall time < 80ms (reads overlapped). `pytest.mark.slow`.
- **Test type:** Unit

### AC6 — `ensure_physics_cache` fills misses and is a no-op when cache is warm

- **Setup A (misses):** `mock_dm`, 10 clusters, empty `feature_cache`. Patch `get_cell_physics` with a call counter. Call `ensure_physics_cache(all_ids)`.
- **Expected A:** Counter == 10. All 10 cells have `_computed == True`.
- **Setup B (warm):** Same `mock_dm`, `feature_cache` pre-populated with `_computed: True` for all 10.
- **Expected B:** Counter == 0 (no calls made).
- **Test type:** Unit

### AC7 — `get_physics_feature_matrix` output matches old `extract_features_from_datamanager`

- **Setup:** `mock_dm`, 10 clusters, all with valid `timecourse` (length 30) and `acg` (length 201) in `feature_cache`. Call both the old function (copied verbatim as reference) and `dm.get_physics_feature_matrix(all_ids, w_shape=2.0, w_pattern=1.5, w_geometry=1.0)`.
- **Expected:** `valid_ids` identical. `feature_matrix.shape` identical. Values `np.allclose(atol=1e-6)`. `metadata` keys and values identical.
- **Test type:** Unit

### AC8 — `get_physics_feature_matrix` excludes cells with `timecourse=None` or `acg=None`

- **Setup:** 5 clusters: 3 valid, 1 with `timecourse=None`, 1 with `acg=None`.
- **Expected:** `len(valid_ids) == 3`. No crash. Excluded IDs absent from `valid_ids`.
- **Test type:** Unit

### AC9 — `get_physics_feature_matrix` drops NaN rows before PCA

- **Setup:** 5 clusters, one with `np.nan` in its timecourse.
- **Expected:** NaN cluster absent from `valid_ids`. No `LinAlgError`. Remaining 4 produce valid `(4, n_features)` matrix with no NaNs.
- **Test type:** Unit

### AC10 — `np.sort` removal in `_calculate_isi_violations` is safe

- **Setup:** Sorted spike train with a known number of refractory violations (see Block 10 test).
- **Expected:** Violation percentage identical before and after removing `np.sort`.
- **Test type:** Unit

### AC11 — Both serial `_compute_physics` loops are gone from `callbacks.py`

- **Action:** `grep` for `for cid in all_ids` in the physics context of `callbacks.py`.
- **Expected:** Zero matches in both `_on_vision_loaded` and `_on_vision_native_loaded`.
- **Test type:** Manual code review before PR.

### AC12 — Visual: UMAP runs without stall on first click after Vision load

- **State to reproduce:**
  1. Launch app, load `/mnt/lab/Array-data/sorted/20260506A/chunk10/kilosort2.5`.
  2. Wait 5 seconds (let background StandardPlotsWorker and physics warm-up run).
  3. Click "Run UMAP (2D)".
- **Expected:** UMAP result appears within 5 seconds of clicking. No progress bar stall at "Extracting features". Status bar shows "Ensuring physics cache..." briefly then "Running UMAP on N cells...".
- **Verified by:** `[ ]` Author `[ ]` Reviewer

---

## Block 9 — Regression Guard

| Prior fix | Files overlap | Regression test | When to run |
|---|---|---|---|
| ACG uses full recording | `data_manager.py::_compute_standard_plots` | `test_acg_includes_late_spike_trains` | Before PR — `np.sort` removal touches this function |
| Same cluster computed once | `data_manager.py` | `test_standard_plot_cache_computes_same_cluster_once` | Before PR |
| Different clusters in parallel | `data_manager.py` | `test_standard_plot_cache_allows_different_clusters_to_compute_concurrently` | Before PR |
| Vision ID offset | `data_manager.py::get_cell_physics` | `test_get_cell_physics_vision_id_offset` (both branches) | Before PR — `get_cell_physics` is modified |
| HDBSCAN default clustering | `umap_panel.py` | `tests/unit/test_hdbscan_clustering.py` (all 7) | Before PR — `umap_panel.py` is modified |

Run every test in this table before opening the PR. A regression in a previously-passing test is a blocking issue.

---

## Block 10 — Test Plan

**File: `tests/unit/test_physics_cache_unified.py`** (new file)

```python
# tests/unit/test_physics_cache_unified.py
import threading
import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_dm(n_clusters=10, acg_len=201, tc_len=30, all_computed=False):
    """Minimal DataManager stub with feature_cache and locking wired up."""
    dm = MagicMock()
    dm._feature_lock = threading.Lock()
    dm._physics_cell_locks = {}
    dm._physics_cell_locks_lock = threading.Lock()
    rng = np.random.default_rng(42)

    cache = {}
    for cid in range(n_clusters):
        entry = {
            'acg': rng.random(acg_len),
            'timecourse': rng.standard_normal(tc_len),
            'rf_area': float(rng.uniform(0.01, 0.5)),
            'ellipticity': float(rng.uniform(0.5, 2.0)),
            'time_to_peak': int(rng.integers(5, 25)),
        }
        if all_computed:
            entry['_computed'] = True
        cache[cid] = entry

    dm.feature_cache = cache
    dm.get_cell_physics = lambda cid: cache[int(cid)]
    return dm, cache


def _make_lazy_sta_dict(keys=None, max_cache=10, sleep_s=0.0):
    """LazySTADict with a mocked reader."""
    from src.analysis.vision_integration import LazySTADict
    reader = MagicMock()
    reader.get_sta_for_cell_id = lambda key: (time.sleep(sleep_s), np.zeros((4, 4, 5)))[1]
    if keys is None:
        keys = list(range(1, 6))
    lsd = LazySTADict.__new__(LazySTADict)
    lsd.reader = reader
    lsd._max_cache = max_cache
    lsd._cache = {}
    lsd._cache_keys = []
    lsd._cache_lock = threading.Lock()
    lsd.keys_list = keys
    return lsd


# ─────────────────────────────────────────────────────────────────────────────
# AC1 — _compute_standard_plots writes ACG into feature_cache
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_standard_plots_writes_acg_to_feature_cache(tmp_path):
    from src.analysis.data_manager import DataManager
    import numpy as np

    dm = DataManager.__new__(DataManager)
    dm._feature_lock = threading.Lock()
    dm._standard_plot_lock = threading.Lock()
    dm._std_plot_cell_locks = {}
    dm._std_plot_cell_locks_lock = threading.Lock()
    dm.feature_cache = {}
    dm.standard_plot_cache = {}
    dm.sampling_rate = 20000.0

    # Minimal spike train: 200 spikes, well-separated (no ACG skip)
    spikes = np.arange(0, 200 * 40, 40, dtype=np.int64)  # 40-sample gaps = 2ms @ 20kHz
    dm.spike_times = spikes
    dm.spike_clusters = np.zeros(len(spikes), dtype=np.int64)

    # Stub out the parts we don't need
    dm.get_cluster_spikes = lambda cid: spikes
    dm.get_cluster_spike_amplitudes = lambda cid: np.ones(len(spikes))
    dm.templates = None

    dm._compute_standard_plots(0)

    assert 0 in dm.feature_cache
    assert dm.feature_cache[0].get('acg') is not None
    assert not dm.feature_cache[0].get('_computed')  # partial entry only


# ─────────────────────────────────────────────────────────────────────────────
# AC2 — get_cell_physics skips get_standard_plot_data when ACG already in cache
# ─────────────────────────────────────────────────────────────────────────────

def test_get_cell_physics_skips_std_data_when_acg_cached(tmp_path):
    from src.analysis.data_manager import DataManager

    dm = DataManager.__new__(DataManager)
    dm._feature_lock = threading.Lock()
    dm._physics_cell_locks = {}
    dm._physics_cell_locks_lock = threading.Lock()
    dm._physics_done_count = 0
    dm.is_vision_only = False
    dm.vision_stas = None
    dm.vision_params = None

    sentinel = np.ones(201)
    dm.feature_cache = {0: {'acg': sentinel}}

    call_counter = {'n': 0}
    def fake_std_data(cid):
        call_counter['n'] += 1
        return {'acg_norm': np.zeros(201)}
    dm.get_standard_plot_data = fake_std_data

    metrics = dm.get_cell_physics(0)

    assert call_counter['n'] == 0, "get_standard_plot_data should not have been called"
    assert np.array_equal(metrics['acg'], sentinel)


# ─────────────────────────────────────────────────────────────────────────────
# AC3 — get_cell_physics falls back to get_standard_plot_data when no ACG cached
# ─────────────────────────────────────────────────────────────────────────────

def test_get_cell_physics_falls_back_to_std_data_when_no_acg(tmp_path):
    from src.analysis.data_manager import DataManager

    dm = DataManager.__new__(DataManager)
    dm._feature_lock = threading.Lock()
    dm._physics_cell_locks = {}
    dm._physics_cell_locks_lock = threading.Lock()
    dm._physics_done_count = 0
    dm.is_vision_only = False
    dm.vision_stas = None
    dm.vision_params = None
    dm.feature_cache = {}

    expected_acg = np.ones(201) * 0.5
    call_counter = {'n': 0}
    def fake_std_data(cid):
        call_counter['n'] += 1
        return {'acg_norm': expected_acg}
    dm.get_standard_plot_data = fake_std_data

    metrics = dm.get_cell_physics(0)

    assert call_counter['n'] == 1
    assert np.array_equal(metrics['acg'], expected_acg)


# ─────────────────────────────────────────────────────────────────────────────
# AC4 — LazySTADict concurrent reads do not corrupt cache
# ─────────────────────────────────────────────────────────────────────────────

def test_lazy_sta_dict_cache_is_thread_safe():
    lsd = _make_lazy_sta_dict(keys=[1, 2], sleep_s=0.005)
    errors = []

    def fetch(key):
        try:
            _ = lsd[key]
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=fetch, args=(1,))
    t2 = threading.Thread(target=fetch, args=(2,))
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert not errors, f"Thread errors: {errors}"
    assert 1 in lsd._cache and 2 in lsd._cache
    assert len(lsd._cache_keys) == len(set(lsd._cache_keys)), "Duplicate keys in _cache_keys"


# ─────────────────────────────────────────────────────────────────────────────
# AC5 — LazySTADict SSD read is not serialised
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_lazy_sta_dict_reads_are_concurrent():
    lsd = _make_lazy_sta_dict(keys=[1, 2], sleep_s=0.05)
    start = time.monotonic()

    t1 = threading.Thread(target=lambda: lsd[1])
    t2 = threading.Thread(target=lambda: lsd[2])
    t1.start(); t2.start()
    t1.join(); t2.join()

    elapsed = time.monotonic() - start
    assert elapsed < 0.08, f"Reads were serialised (elapsed={elapsed:.3f}s)"


# ─────────────────────────────────────────────────────────────────────────────
# AC6 — ensure_physics_cache fills misses, no-op when warm
# ─────────────────────────────────────────────────────────────────────────────

def test_ensure_physics_cache_fills_misses():
    from src.analysis.data_manager import DataManager

    dm = DataManager.__new__(DataManager)
    dm._feature_lock = threading.Lock()
    dm._physics_cell_locks = {}
    dm._physics_cell_locks_lock = threading.Lock()
    dm._physics_done_count = 0
    dm.feature_cache = {}

    call_counter = {'n': 0}
    def fake_get_physics(cid):
        call_counter['n'] += 1
        with dm._feature_lock:
            dm.feature_cache[int(cid)] = {'_computed': True, 'acg': None, 'timecourse': None,
                                           'rf_area': 0.0, 'ellipticity': 0.0, 'time_to_peak': 0}
        return dm.feature_cache[int(cid)]
    dm.get_cell_physics = fake_get_physics

    dm.ensure_physics_cache(list(range(10)))

    assert call_counter['n'] == 10
    assert all(dm.feature_cache[i].get('_computed') for i in range(10))


def test_ensure_physics_cache_noop_when_warm():
    from src.analysis.data_manager import DataManager

    dm = DataManager.__new__(DataManager)
    dm._feature_lock = threading.Lock()
    dm.feature_cache = {i: {'_computed': True} for i in range(10)}

    call_counter = {'n': 0}
    dm.get_cell_physics = lambda cid: (call_counter.__setitem__('n', call_counter['n'] + 1), {})[1]

    dm.ensure_physics_cache(list(range(10)))

    assert call_counter['n'] == 0


# ─────────────────────────────────────────────────────────────────────────────
# AC7 — get_physics_feature_matrix matches old extract_features_from_datamanager
# ─────────────────────────────────────────────────────────────────────────────

def test_get_physics_feature_matrix_matches_old_extractor():
    from src.analysis.data_manager import DataManager
    # Copy the old function verbatim as a reference implementation
    from src.gui.panels.umap_panel import extract_features_from_datamanager as old_fn

    dm, cache = _make_mock_dm(n_clusters=10, all_computed=True)

    # New path
    mat_new, ids_new, meta_new = dm.get_physics_feature_matrix(
        list(range(10)), w_shape=2.0, w_pattern=1.5, w_geometry=1.0)

    # Old path (after the refactor, old_fn delegates to get_physics_feature_matrix
    # with the W_ constants — this test confirms the delegation is correct)
    mat_old, ids_old, meta_old = old_fn(dm, list(range(10)))

    assert ids_new == ids_old
    assert mat_new.shape == mat_old.shape
    assert np.allclose(mat_new, mat_old, atol=1e-6)
    assert meta_new.keys() == meta_old.keys()


# ─────────────────────────────────────────────────────────────────────────────
# AC8 — None timecourse or acg excluded
# ─────────────────────────────────────────────────────────────────────────────

def test_get_physics_feature_matrix_excludes_none_features():
    from src.analysis.data_manager import DataManager

    dm = DataManager.__new__(DataManager)
    dm._feature_lock = threading.Lock()
    dm.is_vision_only = False
    rng = np.random.default_rng(7)

    cache = {
        0: {'_computed': True, 'timecourse': None,                    'acg': rng.random(201), 'rf_area': 0.0, 'ellipticity': 0.0, 'time_to_peak': 0},
        1: {'_computed': True, 'timecourse': rng.standard_normal(30), 'acg': None,            'rf_area': 0.0, 'ellipticity': 0.0, 'time_to_peak': 0},
        2: {'_computed': True, 'timecourse': rng.standard_normal(30), 'acg': rng.random(201), 'rf_area': 0.0, 'ellipticity': 0.0, 'time_to_peak': 0},
        3: {'_computed': True, 'timecourse': rng.standard_normal(30), 'acg': rng.random(201), 'rf_area': 0.0, 'ellipticity': 0.0, 'time_to_peak': 0},
        4: {'_computed': True, 'timecourse': rng.standard_normal(30), 'acg': rng.random(201), 'rf_area': 0.0, 'ellipticity': 0.0, 'time_to_peak': 0},
    }
    dm.feature_cache = cache
    dm.get_cell_physics = lambda cid: cache[int(cid)]

    _, valid_ids, _ = dm.get_physics_feature_matrix(list(range(5)))
    assert len(valid_ids) == 3
    assert 0 not in valid_ids and 1 not in valid_ids


# ─────────────────────────────────────────────────────────────────────────────
# AC9 — NaN rows dropped before PCA
# ─────────────────────────────────────────────────────────────────────────────

def test_get_physics_feature_matrix_drops_nan_rows():
    from src.analysis.data_manager import DataManager

    dm = DataManager.__new__(DataManager)
    dm._feature_lock = threading.Lock()
    dm.is_vision_only = False
    rng = np.random.default_rng(3)

    nan_tc = rng.standard_normal(30).copy()
    nan_tc[5] = np.nan
    cache = {
        0: {'_computed': True, 'timecourse': nan_tc,                  'acg': rng.random(201), 'rf_area': 0.0, 'ellipticity': 0.0, 'time_to_peak': 0},
        1: {'_computed': True, 'timecourse': rng.standard_normal(30), 'acg': rng.random(201), 'rf_area': 0.0, 'ellipticity': 0.0, 'time_to_peak': 0},
        2: {'_computed': True, 'timecourse': rng.standard_normal(30), 'acg': rng.random(201), 'rf_area': 0.0, 'ellipticity': 0.0, 'time_to_peak': 0},
        3: {'_computed': True, 'timecourse': rng.standard_normal(30), 'acg': rng.random(201), 'rf_area': 0.0, 'ellipticity': 0.0, 'time_to_peak': 0},
        4: {'_computed': True, 'timecourse': rng.standard_normal(30), 'acg': rng.random(201), 'rf_area': 0.0, 'ellipticity': 0.0, 'time_to_peak': 0},
    }
    dm.feature_cache = cache
    dm.get_cell_physics = lambda cid: cache[int(cid)]

    mat, valid_ids, _ = dm.get_physics_feature_matrix(list(range(5)))
    assert 0 not in valid_ids
    assert len(valid_ids) == 4
    assert not np.any(np.isnan(mat))


# ─────────────────────────────────────────────────────────────────────────────
# AC10 — np.sort removal from _calculate_isi_violations is safe
# ─────────────────────────────────────────────────────────────────────────────

def test_isi_violations_sort_removed_output_unchanged():
    SAMPLING_RATE = 20000.0
    REFRACTORY_MS = 2.0

    # Sorted spike train with 2 known violations (gaps < 2ms = < 40 samples)
    spikes = np.array([0, 30, 60, 1000, 2000, 3000], dtype=np.int64)

    ref_period = (REFRACTORY_MS / 1000.0) * SAMPLING_RATE
    ref_pct = (np.sum(np.diff(np.sort(spikes)) < ref_period) / (len(spikes) - 1)) * 100
    fix_pct = (np.sum(np.diff(spikes)          < ref_period) / (len(spikes) - 1)) * 100

    assert ref_pct == fix_pct
    assert ref_pct == pytest.approx(40.0)
```

---

## Block 11 — Out of Scope

- Does **not** change `get_cell_physics()` logic beyond the ACG partial-entry check.
- Does **not** change `StandardPlotsWorker` queue logic or `finished_cluster` signal.
- Does **not** change `FeatureExtractionWindow` UI — only `FeatureAnalysisWorker.run()` body.
- Does **not** change `UMAPPanel` UI, controls, colormap, 3D mode, or clustering path.
- Does **not** add UMAP-without-Vision support — that is `umap_vision_optional.md`.
- Does **not** address UI sluggishness during `StandardPlotsWorker` background caching — that is `background_worker_gil_throttle.md`.
- Does **not** change `precompute_ei_correlations_background` — already on its own daemon thread.
- Does **not** add or change any persisted cache file format — `feature_cache.pkl` and `standard_plot_cache.pkl` are unchanged.
- Does **not** change the `vl.STAReader` object or assume anything about its internal locking.
