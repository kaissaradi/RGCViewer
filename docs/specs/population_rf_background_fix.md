# Specification: Population RF Background Fix

## Block 0 — Metadata

| Field | Value |
|---|---|
| **Date created** | 2026-05-29 |
| **Last updated** | 2026-05-29 |
| **Commit hash when spec was written** | `874b4c0` |
| **Branch** | `fix/population-rf-background` |
| **Author** | Kais + Agent |
| **Spec status** | Completed |

---

## Block 1 — Problem Statement

**Symptom:** The population RF mosaic shows an empty or nearly-empty plot. Ellipses are drawn off-screen (no axis limits set), the y-flip is missing so the highlight ellipse is misaligned with the background, axes have no title/styling, and the `_snapshot_rf_background` cache captures zero `EllipseCollection`s — making the LRU cache useless even when it hits.

**Root cause:** `plot_population_rfs_background()` (line 482 of `population_panel.py`) was introduced as a replacement for the drawing portion of `plot_population_rfs()`, but it:

1. Draws individual `Ellipse` patches via `ax.add_patch()` → `_snapshot_rf_background` iterates only `ax.collections` looking for `EllipseCollection` instances → snapshots zero collections → cache entries have empty `collections` lists → cache "hits" replay nothing.
2. Never calls `ax.set_xlim()` / `ax.set_ylim()` → ellipses are drawn but axis stays at default `(0, 1)` → all content is off-screen.
3. Does not apply the y-flip (`sta_height - center_y`) → background ellipses use raw `center_y` while `_update_highlight_patch` (line 445) flips the highlight → highlight and background are in different coordinate systems.
4. Does not call `_apply_rf_axes_style()` → no title, no aspect ratio lock, no tick/spine styling.

The old `plot_population_rfs()` (line 534) is dead code — it is defined and exported in `__init__.py` but never called from any source file.

**User story:** "As a scientist reviewing population receptive fields, I want the RF mosaic to render all ellipses within visible axis bounds, with the selected cell's highlight correctly overlaid on the background, and the cached background to replay correctly on revisit, so that I can rapidly scroll between cells and groups without stale or empty plots."

---

## Block 2 — Vision ID Contract

| Question | Answer |
|---|---|
| Does this spec access Vision file data? | Yes — reads `vision_params.get_cell_ids()` and `vision_params.get_stafit_for_cell()` |
| ID space this spec operates in | Both — Vision 1-indexed IDs from `get_cell_ids()`, Kilosort 0-indexed IDs in `subset_cell_ids` |
| Reads `is_vision_only` flag? | Yes — to translate between ID spaces |
| Translation used | `cid = cell_id if is_vision_only else cell_id - 1` (existing pattern, line 497) |
| Safe access pattern used | `try: stafit = vision_params.get_stafit_for_cell(cell_id)` / `except KeyError: continue` |

The existing ID translation in `plot_population_rfs_background` (line 497) is correct and unchanged.
The y-flip fix applies `sta_height - stafit.center_y` to match the convention used by `_update_highlight_patch` (line 445).

---

## Block 3 — Affected Files

### Pass 1 — Core Fix

| File path | Function(s) modified | Change type | Touches DataManager? |
|---|---|---|---|
| `src/gui/panels/population_panel.py` | `plot_population_rfs_background()` | **Rewrite** | No |
| `src/gui/panels/population_panel.py` | `plot_population_rfs()` | **Delete** | No |
| `src/gui/panels/__init__.py` | Imports of `plot_population_rfs` | **Remove dead import** | No |
| `tests/unit/test_population_rf_background_fix.py` | 8 new tests | **Add** | No |

### Pass 2 — Caller Cleanup

| File path | Function(s) modified | Change type | Touches DataManager? |
|---|---|---|---|
| `src/gui/main_window.py` | `_process_folder_selection()` | **Modify** — pass `group_ids` directly to `redraw_population_panels` | No |
| `src/gui/callbacks.py` | `redraw_population_panels()` | **Modify** — accept optional `subset` argument | No |

> No rows touch DataManager. No rebase obligation.

---

## Block 4 — Qt Threading Contract

No new threading introduced. All modified functions run on the main thread.

| Operation | Runs on thread | Worker class | Signal | Slot | Tier |
|---|---|---|---|---|---|
| `plot_population_rfs_background()` | Main thread | N/A | N/A | Called from `draw_population_rfs_plot()` | Tier 2 (inside debounced full rebuild) |
| `_update_highlight_patch()` | Main thread | N/A | N/A | Called from `draw_population_rfs_plot()` | Tier 1 (hot-swap) |

Stale result guard: Not applicable — these are pure rendering functions that don't process async results.

**Tier 1 safety:** `draw_population_rfs_plot` is called from Tier 1 (`update_cluster_views`, line 622) only for the hot-swap path (`can_hot_swap=True`), which just moves the highlight patch and calls `draw_idle()`. The full rebuild path with `plot_population_rfs_background` only fires from Tier 2. This spec does not change that contract.

---

## Block 5 — Cache Contract

| Question | Answer |
|---|---|
| Cache read | `_rf_background_cache` keyed by `hash(tuple(sorted(subset_cell_ids)))` |
| Cache written | `_rf_background_cache` keyed by same |
| Invalidation trigger | `invalidate_population_caches()` on refinement, Vision reload, or dataset change |
| Persisted to disk | No — module-level dict, session only |
| Lock required | None — main thread only |
| Must tests bypass cache? | **Yes** — tests must call `invalidate_population_caches()` in setup |

**Critical fix:** The current cache is functionally broken because `_snapshot_rf_background` looks for `EllipseCollection` instances in `ax.collections`, but `plot_population_rfs_background` only adds individual `Ellipse` patches. The fix converts to `EllipseCollection`s so the snapshot actually captures geometry.

---

## Block 6 — DataManager Attributes Used

| Attribute | Type | Can be `None`? | This spec reads / writes | Safe access pattern |
|---|---|---|---|---|
| `vision_params` | `VisionCellDataTable` | **Yes** | Reads | `if not vision_params:` guard at line 365 in caller |
| `vision_sta_height` | `int` or `None` | **Yes** | Reads | `if sta_height is not None: adjusted_y = sta_height - y` else `adjusted_y = y` |
| `is_vision_only` | `bool` | No | Reads | `getattr(dm, 'is_vision_only', False)` |

---

## Block 7 — Acceptance Criteria

### AC1 — Ellipses render within visible axis bounds

- **Setup:** Mock `vision_params` with 6 cells at known `(center_x, center_y)` coordinates. Call `plot_population_rfs_background()` with `subset_cell_ids=[0, 1, 2]`, `sta_height=100`.
- **Action:** Inspect `ax.get_xlim()` and `ax.get_ylim()`.
- **Expected:** All 6 ellipse centers fall within the x-limits and y-limits (with ≥10 stixel margin on each side).
- **Test type:** Unit

### AC2 — Y-flip is applied to all background ellipses

- **Setup:** Same 6-cell mock. Cell with `center_y=30`, `sta_height=100`.
- **Action:** Inspect the ellipse offsets on the `EllipseCollection`s or individual patches.
- **Expected:** The plotted y-coordinate is `100 - 30 = 70`, not `30`. Every ellipse's y-offset equals `sta_height - center_y`.
- **Test type:** Unit

### AC3 — Y-flip matches between background and highlight

- **Setup:** Mock `vision_params` with 1 cell. Call `draw_population_rfs_plot()` with that cell as both the selected cell and the only subset member. `sta_height=100`.
- **Action:** Extract the background ellipse center and the highlight patch center.
- **Expected:** Both have the same `(x, y)` coordinates (both y-flipped).
- **Test type:** Unit

### AC4 — `_snapshot_rf_background` captures non-empty collections

- **Setup:** Call `plot_population_rfs_background()`, then call `_snapshot_rf_background(ax, colors, show_ids)`.
- **Action:** Inspect the returned dict.
- **Expected:** `cache_entry['collections']` has length ≥ 1. Each entry has non-empty `offsets`, `widths`, `heights`, `angles` arrays.
- **Test type:** Unit

### AC5 — Cache replay produces identical ellipses

- **Setup:** Draw background, snapshot it, then clear the axes and replay via `_draw_cached_rf_background()`.
- **Action:** Compare ellipse count and geometry between original and replayed axes.
- **Expected:** Same number of collections, same offset arrays (within float tolerance).
- **Test type:** Unit

### AC6 — `_apply_rf_axes_style` is called (title, aspect, ticks)

- **Setup:** Call `plot_population_rfs_background()` with 6 cells.
- **Action:** Inspect `ax.get_title()`, `ax.get_aspect()`.
- **Expected:** Title matches `"Population Receptive Fields (n=X)"` where X is the target count. Aspect is `'equal'`.
- **Test type:** Unit

### AC7 — Existing alpha/linewidth spec values preserved

- **Setup:** Same as AC1.
- **Action:** Inspect the `EllipseCollection` properties.
- **Expected:** Background ellipses: `alpha ∈ [0.12, 0.18]`, `lw ∈ [0.6, 0.9]`. Target ellipses: `alpha ∈ [0.45, 0.65]`, `lw ∈ [0.9, 1.2]`. (Matching existing mosaic spec ranges.)
- **Test type:** Unit

### AC8 — Dead code `plot_population_rfs` is removed

- **Setup:** Attempt `from src.gui.panels.population_panel import plot_population_rfs`.
- **Action:** Import.
- **Expected:** `ImportError` is raised. Function does not exist in the module.
- **Test type:** Unit

### AC9 — (Pass 2) `redraw_population_panels` uses passed subset, not re-derived

- **Setup:** Mock `main_window` with `_get_pop_subset_ids` returning `[0, 1]`. Call `redraw_population_panels(mw, subset=[3, 4])`.
- **Action:** Inspect the `subset_ids` argument received by `draw_population_timecourse_panel` and `draw_population_acg_panel`.
- **Expected:** Both receive `[3, 4]`, not `[0, 1]`. `_get_pop_subset_ids` is never called.
- **Test type:** Unit

---

## Block 8 — Regression Guard

| Prior fix | Files overlap | Regression test | When |
|---|---|---|---|
| Population mosaic gridlines, zoom/pan, Show IDs cache | `population_panel.py` | `test_population_rf_mosaic.py` (7 tests) | Before opening PR |
| Population panel caching | `population_panel.py` | `test_population_panel_cache.py` (10 tests) | Before opening PR |
| Population panel cache invalidation | `population_panel.py` | `test_population_panel_cache_invalidation.py` (3 tests) | Before opening PR |
| Population panel revisit cache | `population_panel.py` | `test_population_panel_revisit_cache.py` (3 tests) | Before opening PR |

**Pre-PR regression suite (all 26 must pass):**
```bash
conda run -n rgcviewer python -m pytest \
  tests/unit/test_population_rf_mosaic.py \
  tests/unit/test_population_panel_cache.py \
  tests/unit/test_population_panel_cache_invalidation.py \
  tests/unit/test_population_panel_revisit_cache.py \
  -v
```

---

## Block 9 — Test Plan

### Unit tests

File: `tests/unit/test_population_rf_background_fix.py`

| Test function name | Fixture | What it asserts | Cache bypass? |
|---|---|---|---|
| `test_axis_limits_contain_all_ellipses` | `invalidate_population_caches` | All 6 ellipse centers within `xlim`/`ylim` (with margin) | Yes |
| `test_y_flip_applied_to_all_ellipses` | `invalidate_population_caches` | Every plotted y == `sta_height - center_y` | Yes |
| `test_y_flip_matches_highlight_and_background` | `invalidate_population_caches` | Background and highlight patch share same `(x, y)` for same cell | Yes |
| `test_snapshot_captures_nonempty_collections` | `invalidate_population_caches` | `cache_entry['collections']` has ≥1 entry with non-empty arrays | Yes |
| `test_cache_replay_produces_identical_geometry` | `invalidate_population_caches` | Replayed offsets match original offsets within `atol=1e-6` | Yes |
| `test_axes_style_applied_title_and_aspect` | `invalidate_population_caches` | Title matches pattern, aspect is `'equal'` | Yes |
| `test_ellipse_alpha_lw_ranges_preserved` | `invalidate_population_caches` | Background and target alpha/lw within spec ranges | Yes |
| `test_dead_code_plot_population_rfs_removed` | None | `ImportError` on `from ... import plot_population_rfs` | N/A |
| `test_redraw_population_panels_uses_passed_subset` | None (mock) | Passed subset forwarded, `_get_pop_subset_ids` not called | N/A |

### Integration tests

No new integration tests required. The existing `test_population_rf_ids_toggle` (in `tests/test_population_panel.py`) covers the real-data end-to-end flow. The caller changes (Pass 2) are covered by the `test_redraw_population_panels_uses_passed_subset` unit test.

### Visual regression tests

Not added by this spec. A manual visual check is required (see below).

### Manual verification (you said you'll do this)

1. Load a real dataset with Vision `.params` file
2. Open population split view
3. Verify: RF mosaic shows all ellipses within the viewport — not clipped, not empty
4. Select a cell → highlight ellipse overlaps its background ellipse perfectly (no y-offset mismatch)
5. Click a folder → background redraws with correct subset, title updates
6. Navigate A → B → A → verify cache replay shows identical plot (not empty)
7. Toggle "Show IDs" → labels appear/disappear correctly
8. Verify both light and dark mode

---

## Block 10 — Out of Scope

- Does **not** modify `draw_population_rfs_plot()` — its hot-swap/caching logic is correct once the background renders properly.
- Does **not** modify `_update_highlight_patch()` — it already applies y-flip correctly.
- Does **not** modify `_snapshot_rf_background()` or `_draw_cached_rf_background()` — they are designed for `EllipseCollection`s and will work once the background uses them.
- Does **not** modify `_build_ellipse_collection()` — it is the existing helper that the rewrite will call.
- Does **not** touch `draw_population_timecourse_panel()` or `draw_population_acg_panel()`.
- Does **not** touch `DataManager`, `feature_cache`, `standard_plot_cache`, or any analysis code.
- Does **not** modify alpha/linewidth values — preserves the ranges from the population-panel-mosaic spec.
- Does **not** change the hot-swap path in `draw_population_rfs_plot()`.

---

## Appendix A — Detailed Implementation Notes

### A.1 — Rewritten `plot_population_rfs_background`

The function will be restructured as follows:

```python
def plot_population_rfs_background(ax, vision_params, main_window, sta_height, subset_cell_ids, colors):
    ax.clear()
    show_labels = main_window.pop_show_ids_checkbox.isChecked()
    is_vision_only = getattr(main_window.data_manager, 'is_vision_only', False)

    bg_ellipses = []     # (x, y, w, h, angle_deg) for non-subset cells
    target_ellipses = [] # (x, y, w, h, angle_deg) for subset cells
    x_coords = []
    y_coords = []

    for cell_id in vision_params.get_cell_ids():
        try:
            stafit = vision_params.get_stafit_for_cell(cell_id)
        except KeyError:
            continue

        cid = cell_id if is_vision_only else cell_id - 1
        adjusted_y = sta_height - stafit.center_y if sta_height is not None else stafit.center_y

        x_coords.append(stafit.center_x)
        y_coords.append(adjusted_y)

        entry = (stafit.center_x, adjusted_y,
                 stafit.std_x * 2, stafit.std_y * 2,
                 np.degrees(stafit.rot))

        if cid in subset_cell_ids:
            target_ellipses.append(entry)
            if show_labels:
                ax.text(stafit.center_x, adjusted_y, str(cell_id),
                        color=colors.get('text_secondary', '#9B9DA6'),
                        fontsize=8, ha='center', va='center',
                        alpha=0.8)
        else:
            bg_ellipses.append(entry)

    # Build EllipseCollections (so _snapshot_rf_background can capture them)
    bg_coll = _build_ellipse_collection(
        bg_ellipses,
        edgecolor=colors.get('border_subtle', '#2E3038'),
        alpha=0.15, lw=0.75, zorder=1)
    if bg_coll is not None:
        ax.add_collection(bg_coll)
        bg_coll.set_offset_transform(ax.transData)

    target_coll = _build_ellipse_collection(
        target_ellipses,
        edgecolor=colors.get('plot_highlight', '#00FFFF'),
        alpha=0.55, lw=1.0, zorder=2)
    if target_coll is not None:
        ax.add_collection(target_coll)
        target_coll.set_offset_transform(ax.transData)

    # Set axis limits from collected coordinates
    if x_coords:
        margin = 20
        ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
        ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)

    n_target = len(target_ellipses)
    _apply_rf_axes_style(ax, colors,
                         title=f"Population Receptive Fields (n={n_target})")
```

### A.2 — Delete `plot_population_rfs` (lines 534–647)

This function is dead code. It was the original implementation before `plot_population_rfs_background` was introduced. No source file calls it. Remove it and its import from `__init__.py`.

### A.3 — Pass 2: `redraw_population_panels` accepts optional subset

```python
# callbacks.py — before
def redraw_population_panels(main_window):
    subset = main_window._get_pop_subset_ids()
    ...

# callbacks.py — after
def redraw_population_panels(main_window, subset=None):
    if subset is None:
        subset = main_window._get_pop_subset_ids()
    ...
```

### A.4 — Pass 2: `_process_folder_selection` passes group_ids directly

```python
# main_window.py — before
callbacks.redraw_population_panels(self)

# main_window.py — after
callbacks.redraw_population_panels(self, subset=group_ids)
```

This avoids `redraw_population_panels` re-deriving the subset via `_get_pop_subset_ids()`, which walks the tree model again and may return stale or mismatched results if the selection state has changed between the timer firing and the panels redrawing.

---

## Appendix B — Risk Analysis

### Tests that will break intentionally

| Test | Why | Fix |
|---|---|---|
| `test_rf_background_cache_hit_on_revisit` | Cache now contains real `EllipseCollection` data instead of empty lists. The mock `ax` won't have real `ax.collections`. | The test uses `MagicMock()` for `ax`, so `ax.collections` is already a mock list — `_snapshot_rf_background` iterates it and finds nothing. This test may still pass because it tests the *cache lookup path*, not the cache *content*. If it breaks, the fix is to use a real `Figure()/ax` instead of a `MagicMock`. |
| `test_rf_background_a_b_a_revisit_uses_cached_geometry` | Same issue — mock canvas prevents real geometry capture. | Same fix: use real matplotlib `Figure()`. |

### Tests that must NOT break

All 7 tests in `test_population_rf_mosaic.py` must continue to pass. These test the alpha/lw ranges and Show IDs labels. Since the rewrite preserves the same alpha/lw values and label logic, they should pass — but the rewrite changes from individual `Ellipse` patches to `EllipseCollection`, so the tests that inspect `ax.patches` for individual `Ellipse` objects will need to be updated to inspect `ax.collections` instead. This is a **known test adaptation**, not a regression.

**Decision:** Update the mosaic tests to inspect `EllipseCollection` properties (widths, heights, alpha, linewidths) rather than individual `Ellipse` patch properties. This is a mechanical change that preserves the same assertions.
