# Spec: Population Panel — Folder Navigation Performance

## Metadata

| Field | Value |
|---|---|
| **Date** | 2026-05-23 |
| **Branch** | `feat/population-panel-folder-perf` |
| **Files audited** | `main_window.py`, `callbacks.py`, `population_panel.py`, `vision_integration.py`, `workers.py` |
| **Author** | Kais |
| **Implementation status** | AC-focused implementation committed; timing/integration ACs still need real latency verification |

---

## Implementation Update — 2026-05-24

### Git snapshot

Commands run at session close:

```bash
git rev-parse --abbrev-ref HEAD
# feat/population-panel-folder-perf

git rev-parse HEAD
# 61f093390b521e3631c2ce4ad71dbb6b1d2ff1c5

git merge-base dev-testing HEAD
# 7b7679e6d95ba0ba59c7e932b5036ef5114d86a3

git log --oneline --decorate --reverse dev-testing..HEAD
# 8aa8b93 AC5: invalidate population panel caches on data changes
# bffec4a AC3: debounce population folder selection
# 77e9489 AC4: defer initial population panel draw
# 1ea1b88 AC1: cache population folder revisits
# 61f0933 (HEAD -> feat/population-panel-folder-perf) AC2: size LazySTADict cache for folder-scale visits
```

### Commits landed

| AC | Commit | Status | Notes |
|---|---|---|---|
| AC5 | `8aa8b93` | Implemented + unit verified | Added population cache structures, `invalidate_population_caches()`, Vision-load invalidation, and refinement-result invalidation. |
| AC3 | `bffec4a` | Implemented + unit verified | Added `folder_selection_timer`, `_pending_folder_item`, `_process_folder_selection()`, and moved folder clicks off the synchronous draw path. |
| AC4 | `77e9489` | Implemented + unit verified | Deferred initial population-panel draw with `QTimer.singleShot(0, ...)` and added `_draw_population_panel_initial()`. |
| AC1 | `1ea1b88` | Implemented + unit verified | Added group timecourse/ACG caches and RF background geometry LRU replay. Unit tests verify no repeated data-manager calls on revisits. |
| AC2 | `61f0933` | Support slice implemented + unit verified | Replaced fixed `LazySTADict._max_cache = 40` with dynamic sizing via `MAX_STA_CACHE_CELLS = 500`. End-to-end cold folder latency still needs timing verification. |

### Verification run

```bash
conda run -n rgcviewer python -m pytest tests/unit/test_population_panel_cache.py -v
# 13 passed

conda run -n rgcviewer python -m pytest tests/unit/test_folder_debounce.py -v
# 8 passed

conda run -n rgcviewer python -m pytest tests/unit/test_population_rf_mosaic.py -v
# 7 passed
```

Additional focused tests committed:

| File | Purpose |
|---|---|
| `tests/unit/test_population_panel_cache_invalidation.py` | AC5 cache invalidation and data-change hooks. |
| `tests/unit/test_folder_debounce.py` | AC3 folder debounce and callback branch behavior. |
| `tests/unit/test_population_panel_toggle_open.py` | AC4 deferred initial draw behavior. |
| `tests/unit/test_population_panel_revisit_cache.py` | AC1 revisit cache behavior for timecourse, ACG, and RF mosaic. |
| `tests/unit/test_lazy_sta_cache_size.py` | AC2 `LazySTADict` dynamic cache sizing. |

### Remaining work

- **AC1 latency proof:** Unit tests verify zero repeated `get_cell_physics()`, `get_acg_data()`, and RF background `get_stafit_for_cell()` calls on revisits, but the `< 50ms` end-to-end folder revisit target still needs integration timing on a realistic or real dataset.
- **AC2 latency proof:** The `LazySTADict` cache-size support is implemented and unit-tested, but the `< 200ms` cold first-folder visit target still needs a timing test such as `tests/integration/test_population_panel_perf.py::test_first_folder_latency`.
- **AC4 integration proof:** Unit tests verify the draw is deferred one event-loop tick and the helper draws the panels, but the full blank-on-open acceptance test should still assert real canvas state after toggle-open with a selected folder.
- **Worktree note:** `tests/unit/test_population_panel_cache.py` remains untracked from the broader performance-test draft.

---

## 1. Problem Statement

When the Population Panel is open and the user navigates between folder nodes in the Tree View using arrow keys, the UI freezes for **500ms–1s per keypress**. The freeze happens on every visit — first visit and revisit alike. The target dataset scale is **4–6 folders, up to ~900 cells per folder**.

There is a secondary bug: when the Population Panel is toggled on while a folder is already selected, all three canvases render blank. The user must click away and reselect to get content.

**Target after this fix:**

- Folder revisit: **< 50ms**
- Cold first visit: **< 200ms**
- Blank-on-open: **gone**

---

## 2. Root Cause Analysis

### 2.1 Call chain — folder arrow-key navigation

```
QTreeView::selectionChanged
  → MainWindow.on_view_selection_changed()         [main_window.py]
      → callbacks.on_cluster_selection_changed()   [callbacks.py L381]
          → cluster_id = _get_selected_cluster_id()
          → cluster_id is None  ← folders store None in UserRole
          → [folder branch]
              group_ids = _get_group_cluster_ids(item)  # up to 900 IDs
              draw_population_rfs_plot(...)              # SYNCHRONOUS, main thread
              redraw_population_panels(...)              # SYNCHRONOUS, main thread
                  → draw_population_timecourse_panel(...)
                  → draw_population_acg_panel(...)
```

Everything runs synchronously on the main thread. Each arrow keypress fires a complete redraw before the next keypress is processed.

### 2.2 Bottleneck 1 — Folder selection path has no debounce

The 25ms `self.selection_timer` (main_window.py) only fires for non-`None` `cluster_id` values — i.e. leaf cell nodes. Folder nodes return `None` from `_get_selected_cluster_id()`, so the folder branch calls all three draw functions directly, every single keypress, with no delay. This is the primary cause of the freeze.

### 2.3 Bottleneck 2 — RF mosaic rebuilds O(N_total) patches on every folder switch

`plot_population_rfs_background()` creates and `add_patch`es a fresh `Ellipse` for every cell in the dataset on every call: grey patches for the non-subset cells, coloured patches for the subset. The existing `subset_hash` guard on `canvas._pop_plot_state` prevents a rebuild only when the *exact same* subset is drawn twice in a row with nothing in between. The moment the user switches to a different folder, the canvas state is overwritten. Switching back to a previously visited folder triggers a full O(N_total) rebuild. There is no cross-visit background cache.

With 3000 total cells and a 900-cell active folder, every folder switch creates and renders ~3000 `Ellipse` objects from scratch.

### 2.4 Bottleneck 3 — Timecourse and ACG recompute on every visit, including revisits

`draw_population_timecourse_panel()` loops over all N IDs in `subset_ids`, calls `get_cell_physics(cid)` for each, and `np.vstack`s the results into `arr` on every call. The existing `_timecourse_state` hot-swap only skips `fig.clear()` and reuses matplotlib artist objects — it does **not** skip the data extraction loop. The same loop and vstack run on every visit, including revisits to the same folder.

`draw_population_acg_panel()` has the same structure with `get_acg_data(cid)`.

For 900 cells this is 900 dict lookups + a vstack every keypress that clears the debounce.

### 2.5 Bottleneck 4 — LazySTADict cache undersized for folder scale

`LazySTADict._max_cache` is hardcoded to `40`. With up to 900 cells per folder, navigating to a new folder incurs up to 860 STA disk reads — the cache is constantly evicting entries needed moments ago. Since `get_cell_physics` pulls STA data for timecourse extraction, every cold folder visit pays the full I/O cost for most of its cells.

### 2.6 Bottleneck 5 — Canvas 0×0 race on population panel first open

`toggle_population_split_view()` calls `draw_population_rfs_plot()` and `redraw_population_panels()` immediately after `self.pop_context_widget.show()`. At that moment Qt has not yet completed a layout pass on the newly visible widget — the matplotlib canvas is still 0×0 pixels. `draw_idle()` renders into a zero-size figure and produces a blank canvas. This is why the user must reselect after opening the panel: the reselection fires a second draw after layout is complete and the canvas has real dimensions.

---

## 3. What Already Works — Do Not Touch

| Component | What it does | Location |
|---|---|---|
| 25ms cell-selection debounce | `self.selection_timer` fires `_process_selection` after 25ms quiet period | main_window.py |
| RF mosaic within-session hot-swap | Same subset hash → only highlight patch mutated, no `add_patch` loop | population_panel.py |
| Timecourse hot-swap | Same subset → `_timecourse_state` reuses artists, skips `fig.clear()` | population_panel.py |
| ACG hot-swap | Same as timecourse for ACG panel | population_panel.py |
| Panel-toggle initial draw | `toggle_population_split_view` already calls draw on enable | main_window.py |
| Physics precompute | Inline `run_physics()` loop spawns after Vision load | callbacks.py |

This spec does not modify any of these.

---

## 4. Vision ID Contract

No change. All data access goes through `get_cell_physics(cid)` and `get_acg_data(cid)`, which already apply the correct `vision_id = cid + 1` translation internally. No new direct Vision ID access is introduced.

---

## 5. Proposed Changes

### 5.1 Debounce folder selection — `callbacks.py` + `main_window.py`

**The problem:** The folder branch in `on_cluster_selection_changed` fires synchronous draw calls immediately on every keypress.

**The fix:** Add a dedicated `folder_selection_timer` — a separate `QTimer` using the same 25ms single-shot pattern as `selection_timer`. Folder selection stores the pending item and starts the timer. The timer fires `_process_folder_selection()` which performs the actual draw.

A dedicated timer is preferred over routing folder selection through `update_cluster_views` because folder selection never needs `FeatureWorker`, EI, or waveform snippets. Keeping the two paths separate avoids guard conditions in `_process_selection` and contains the diff.

**`main_window.py` — `__init__`:**

```python
self._pending_folder_item = None
self.folder_selection_timer = QTimer(self)
self.folder_selection_timer.setSingleShot(True)
self.folder_selection_timer.setInterval(25)
self.folder_selection_timer.timeout.connect(self._process_folder_selection)
```

**`main_window.py` — new method:**

```python
def _process_folder_selection(self):
    item = self._pending_folder_item
    if item is None or not self.population_view_enabled:
        return
    group_ids = self._get_group_cluster_ids(item)
    if not group_ids:
        return
    draw_population_rfs_plot(
        main_window=self,
        subset_cell_ids=group_ids,
        canvas=self.pop_mosaic_canvas)
    callbacks.redraw_population_panels(self)
```

**`callbacks.py` — folder branch in `on_cluster_selection_changed`:**

```python
# Before (fires synchronous draws immediately):
draw_population_rfs_plot(...)
redraw_population_panels(...)

# After (debounced):
main_window._pending_folder_item = item
main_window.folder_selection_timer.start()
return
```

---

### 5.2 Group-level numpy cache for timecourse and ACG — `population_panel.py`

**The problem:** Both draw functions recompute from scratch on every visit, including revisits.

**The fix:** Add two module-level cache dicts keyed by `frozenset(subset_ids)`. On a cache hit, skip the `get_cell_physics` / `get_acg_data` loop entirely and use the stored arrays.

**Module-level additions:**

```python
_group_timecourse_cache: dict = {}
# frozenset(ids) -> {'arr': np.ndarray, 'mean_tc': np.ndarray,
#                    't_axis': np.ndarray, 'peak_idx': int}

_group_acg_cache: dict = {}
# frozenset(ids) -> {'arr': np.ndarray, 'mean_acg': np.ndarray,
#                    't_axis': np.ndarray}
```

**In `draw_population_timecourse_panel`**, before the data loop:

```python
cache_key = frozenset(subset_ids)
if cache_key in _group_timecourse_cache:
    cached = _group_timecourse_cache[cache_key]
    arr, mean_tc, t_axis, peak_idx = (
        cached['arr'], cached['mean_tc'],
        cached['t_axis'], cached['peak_idx'])
else:
    # existing loop + vstack
    ...
    _group_timecourse_cache[cache_key] = {
        'arr': arr, 'mean_tc': mean_tc,
        't_axis': t_axis, 'peak_idx': peak_idx}
```

Apply the same pattern in `draw_population_acg_panel`.

**Cache invalidation** — add a module-level function called whenever underlying data changes:

```python
def invalidate_population_caches():
    _group_timecourse_cache.clear()
    _group_acg_cache.clear()
    _rf_background_cache.clear()        # see §5.3
    _rf_background_cache_order.clear()  # see §5.3
```

Call `invalidate_population_caches()` from:

- `callbacks._on_vision_loaded` — new Vision data changes timecourses
- `callbacks.handle_refinement_results` — cluster membership changed

**Memory:** A 900-cell timecourse cache entry is `900 × T_frames × 4 bytes` float32. For T=40 frames: ~144 KB. Six folders: ~870 KB. Negligible.

---

### 5.3 LRU background cache for RF mosaic — `population_panel.py`

**The problem:** `canvas._pop_plot_state` holds only the most recently rendered subset. A→B→A always rebuilds A's 3000 ellipses from scratch on the third visit.

**The fix:** Move the background state off the canvas and into a module-level LRU dict keyed by `subset_hash`. On a cache hit, rebuild patches from stored geometry tuples rather than re-calling `get_stafit_for_cell`. Live `Ellipse` artist objects are not stored — only the raw `(cx, cy, w, h, angle, color, alpha)` tuples needed to recreate them. This avoids matplotlib artist re-parenting issues and is still far faster than re-calling Vision params for 3000 cells.

**Module-level additions:**

```python
_rf_background_cache: dict = {}
# subset_hash -> list of (cx, cy, w, h, angle, edgecolor, facecolor, alpha) tuples

_rf_background_cache_order: list = []  # LRU insertion order
_RF_CACHE_MAX = 10                     # keep last 10 folder backgrounds
```

**In `draw_population_rfs_plot`**, before `plot_population_rfs_background`:

```python
if current_subset_hash in _rf_background_cache:
    ax.clear()
    for (cx, cy, w, h, angle, ec, fc, alpha) in _rf_background_cache[current_subset_hash]:
        ax.add_patch(Ellipse((cx, cy), w, h, angle=angle,
                             edgecolor=ec, facecolor=fc, alpha=alpha))
    # then draw highlight patch only — same as existing hot-swap path
else:
    plot_population_rfs_background(...)   # existing code, extracts geometry
    # after drawing, extract and store geometry for next visit:
    patch_tuples = [
        (p.center[0], p.center[1], p.width, p.height, p.angle,
         p.get_edgecolor(), p.get_facecolor(), p.get_alpha())
        for p in ax.patches
    ]
    _rf_background_cache[current_subset_hash] = patch_tuples
    _rf_background_cache_order.append(current_subset_hash)
    if len(_rf_background_cache_order) > _RF_CACHE_MAX:
        evict = _rf_background_cache_order.pop(0)
        _rf_background_cache.pop(evict, None)
```

---

### 5.4 Dynamic LazySTADict cache size — `vision_integration.py`

**The problem:** `_max_cache = 40` causes ~860 STA disk reads per 900-cell folder visit.

**The fix:** Set the cache size dynamically at init time, capped by a tunable constant to bound RAM usage.

```python
# Module-level constant — tune per machine
MAX_STA_CACHE_CELLS = 500

class LazySTADict:
    def __init__(self, vision_dir, dataset_name):
        ...
        self._max_cache = min(MAX_STA_CACHE_CELLS, max(200, len(self.keys_list)))
```

**Memory:** A typical STA is `T × H × W × 3` float32. For 30 frames × 30×20 stixels: ~216 KB per cell. 500 cells: ~108 MB. Acceptable on a lab workstation.

With 500 cached entries and a 900-cell folder, the second visit to that folder will hit ~500 entries and miss ~400 — a meaningful improvement over 40. `MAX_STA_CACHE_CELLS` can be raised if RAM permits.

---

### 5.5 Fix blank canvas on population panel first open — `main_window.py`

**The problem:** Draw calls in `toggle_population_split_view` execute before Qt has completed layout, so the canvas is still 0×0.

**The fix:** Defer the initial draw by one event-loop tick with `QTimer.singleShot(0, ...)`.

```python
def toggle_population_split_view(self, checked: bool):
    self.population_view_enabled = bool(checked)
    if checked:
        self.pop_context_widget.show()
        total = sum(self.right_splitter.sizes()) or 1400
        left_size = max(int(total * 0.75), 400)
        self.right_splitter.setSizes([left_size, total - left_size])
        # Defer draw one tick — canvas has real dimensions after Qt layout pass
        QTimer.singleShot(0, self._draw_population_panel_initial)
    else:
        self.pop_context_widget.hide()
        self.right_splitter.setSizes([sum(self.right_splitter.sizes()), 0])

def _draw_population_panel_initial(self):
    draw_population_rfs_plot(main_window=self, canvas=self.pop_mosaic_canvas)
    callbacks.redraw_population_panels(self)
```

`draw_population_rfs_plot` called with no `subset_cell_ids` already calls `_get_pop_subset_ids()` internally, which correctly handles both folder and cell selections.

---

## 6. Affected Files Summary

| File | Change |
|---|---|
| `src/gui/callbacks.py` | Folder branch replaced with `_pending_folder_item` assignment + `folder_selection_timer.start()` |
| `src/gui/main_window.py` | Add `folder_selection_timer`, `_pending_folder_item`, `_process_folder_selection()`, `_draw_population_panel_initial()`; refactor `toggle_population_split_view` |
| `src/gui/panels/population_panel.py` | Add `_group_timecourse_cache`, `_group_acg_cache`, `_rf_background_cache`, `_rf_background_cache_order`; add `invalidate_population_caches()`; add cache check + store to both draw functions and RF mosaic draw |
| `src/analysis/vision_integration.py` | Replace `_max_cache = 40` with `min(MAX_STA_CACHE_CELLS, max(200, len(self.keys_list)))` |
| `tests/unit/test_population_panel_cache_invalidation.py` | Focused AC5 regression tests |
| `tests/unit/test_folder_debounce.py` | Focused AC3 regression tests |
| `tests/unit/test_population_panel_toggle_open.py` | Focused AC4 regression tests |
| `tests/unit/test_population_panel_revisit_cache.py` | Focused AC1 regression tests |
| `tests/unit/test_lazy_sta_cache_size.py` | Focused AC2 cache-size regression tests |

**No changes to `data_manager.py` or `workers.py`.**

---

## 7. Threading and Cache Contract

| Concern | Detail |
|---|---|
| `folder_selection_timer` | Main thread. `QTimer` owned by `MainWindow`. |
| `_process_folder_selection` | Main thread. Calls draw functions synchronously. |
| Module-level caches | Written and read on main thread only. Draw functions are always called from the main thread. No locks needed. |
| `invalidate_population_caches()` | Must be called from main thread. Both `_on_vision_loaded` and `handle_refinement_results` already execute on main thread. |
| `LazySTADict` thread safety | Unchanged from before this spec. `__getitem__` is not thread-safe; the background `PhysicsWorker` reads STA data concurrently. This pre-existing condition is out of scope here. |

---

## 8. Acceptance Criteria

### AC1 — Folder revisit is instantaneous

**Setup:** All Vision data loaded. User navigates A → B → A with population panel open.  
**Expected:** Second visit to A completes in < 50ms from keypress to `draw_idle`. No `get_cell_physics` calls. No `get_stafit_for_cell` calls. No disk I/O.  
**Test:** Populate caches via first A visit. Time second A visit. Assert mock call counts are zero.

### AC2 — First folder visit completes within 200ms

**Setup:** Cold group cache. Physics `feature_cache` already populated by precompute worker.  
**Expected:** All three panels update in < 200ms. Dominated by numpy ops, not STA disk reads.  
**Test:** `qtbot` integration test, timing from `selectionChanged` to last `draw_idle`.

### AC3 — Rapid arrow-key through folders draws only the final destination

**Setup:** Population panel open. User presses down-arrow 5 times within 20ms.  
**Expected:** Exactly 1 draw call fires — for the 5th folder. All prior keypresses debounced.  
**Test:** Spy on `draw_population_rfs_plot`. Simulate 5 rapid selections. Advance `folder_selection_timer` by 25ms. Assert spy called once.

### AC4 — Population panel not blank on toggle-open with folder selected

**Setup:** A folder node is selected. User toggles the Population Panel button on.  
**Expected:** All three canvases render correct content without a reselect.  
**Test:** Select folder, toggle panel, `qtbot.wait(50)`. Assert `canvas._pop_plot_state` and `canvas._timecourse_state` are set.

### AC5 — Group caches cleared after Vision reload or refinement

**Setup:** Populate all caches. Fire `_on_vision_loaded` or `handle_refinement_results`.  
**Expected:** `_group_timecourse_cache`, `_group_acg_cache`, and `_rf_background_cache` are all empty.  
**Test:** Unit test — populate, call `invalidate_population_caches()`, assert all dicts empty.

---

## 9. Test Plan

### Unit — `tests/unit/test_population_panel_cache.py`

| Test | Description |
|---|---|
| `test_group_timecourse_cache_hit` | Call `draw_population_timecourse_panel` twice, same `subset_ids`. Mock `get_cell_physics`. Assert mock called only on first invocation. |
| `test_group_acg_cache_hit` | Same for `draw_population_acg_panel` with `get_acg_data`. |
| `test_rf_background_cache_hit_on_revisit` | Draw A, draw B, draw A. Assert `get_stafit_for_cell` call count on third draw equals 1 (highlight only), not N_total. |
| `test_invalidate_clears_all_caches` | Populate all module-level caches. Call `invalidate_population_caches()`. Assert all four structures empty. |
| `test_lazy_sta_cache_size_scales_with_dataset` | Mock reader with 800 keys. Assert `_max_cache == min(MAX_STA_CACHE_CELLS, max(200, 800))`. |

### Unit — `tests/unit/test_folder_debounce.py`

| Test | Description |
|---|---|
| `test_rapid_folder_selection_fires_once` | Assign `_pending_folder_item` 5 times within 10ms. Advance timer 25ms. Assert `draw_population_rfs_plot` called exactly once. |
| `test_folder_and_cell_timers_are_independent` | Assert `folder_selection_timer` and `selection_timer` are distinct `QTimer` instances. Starting one does not reset the other. |

### Integration — `tests/integration/test_population_panel_perf.py`

| Test | Pass condition |
|---|---|
| `test_first_folder_latency` | < 200ms, `mock_dm` with 900 pre-populated `feature_cache` entries |
| `test_revisit_folder_latency` | < 50ms, cache pre-warmed from first visit, zero `get_cell_physics` calls |
| `test_blank_panel_bug_fixed` | After toggle + `qtbot.wait(50)`, both `_pop_plot_state` and `_timecourse_state` set |

---

## 10. Out of Scope

- **`PhysicsWorker` refactor and progress bar accuracy** — the inline `run_physics()` closure works. A proper class with per-cell progress signals is a separate improvement that does not affect navigation speed.
- **Drag-and-drop as a separate concern** — drag/drop fires the same `on_cluster_selection_changed` path. The folder debounce fix (§5.1) resolves it for free.
- **`LazySTADict` thread safety** — pre-existing condition, separate ticket.
- **Replacing Matplotlib with pyqtgraph in the population panel** — out of scope; performance gains here come from caching, not renderer replacement.
- **`_get_group_cluster_ids` performance** — O(tree nodes), not O(cells). Not a bottleneck.
- **Extending group caching to other panels** (similarity, standard plots) — not needed for this use case.
