# PLAN.md — current snapshot

Read `docs/AGENTS.md` before this file. This file is the pickup point.

Last updated: 2026-08-12. Branch: `feat/bauhaus-redesign` (off `dev-testing`).

This file lists standing decisions, fragile code, and open defects. It is
not a roadmap. The full UX redesign spec is still parked except the
color / layout / plot-refresh pass now on this branch. Chrome is blue.
Light-mode plots use a paper field and solid Bauhaus ink (black / blue /
yellow), not translucent cyan or gray. See `docs/specs/ux_ui_redesign.md`.

Do not push unless the user asks.

User check 2026-08-12: load, UMAP defaults, no-STA cells, and EI View combo
are good. Grating, Contrast, and the DS/OS slider were not checked — no
suitable direction/contrast run on hand.

Tree sidebar (2026-08-12): tree is a 3-column ID / Spikes / Ch table.
Folder/file icons and `#3C3C3C` fills are gone. Window open size is
clamped to the usable screen. Sidebar auto-collapse on a narrow window
is still parked.

## 0. Standing decisions

These are also Laws 4–5 and invariant 8 in `docs/AGENTS.md`.

1. Do not reopen the last run at application start. File dialogs still
   remember the last folder.
2. Broken EIs on older kilosort4 conversions are accepted. Do not remap
   a 519-channel EI onto a 512-electrode plot.
3. Do not rewrite the application as HTML.

## 1. Fragile zones

Read the failure column before you change the file.

| What | Failure | Before you change it |
|---|---|---|
| `data_manager.py` | Every panel and cache depends on it. Concurrent writes are timing-dependent. | Rebase. Run the unit suite. |
| `get_cell_physics()` | Vision ID offset lives here. Wrong key returns the previous cell's STA. | Run `test_get_cell_physics_vision_id_offset` (both branches). |
| `build_cluster_dataframe()` | One `np.unique` scan builds `_spk_unique_cls` and `_counts`. A second scan breaks counts and doubles load time. | Do not add `np.unique` or `np.argsort` on the full spike arrays here. |
| `update_cluster_views()` / `_process_selection()` | Tier 1 / Tier 2 boundary. Heavy work in Tier 1 freezes keypress scroll. | Classify each new call as Tier 1 or Tier 2. See AGENTS.md Law 2. |
| `LazySTADict` | Thread-local readers. Handle leaks if extra instances are created. | Run `test_lazy_sta_dict_reads_are_concurrent` and `test_lazy_sta_dict_cache_is_thread_safe`. |
| `_save_pickle_with_fallback()` | Atomic write via tempfile + `os.replace`. A direct dump can leave a truncated file. | Do not simplify. |
| `_compute_ei_correlations_if_needed()` | `is_vision_only` guard stops a RAM-exhausting correlation matrix. | Do not remove the guard. |
| `get_cluster_spike_indices()` | Callers assume O(1). `np.where` on the full arrays is O(N). | Do not bypass. |
| `theme.py` | QSS and `restyle_plots` use the token dict. A renamed key breaks theme toggle. | Toggle theme. Check every tab. |
| `_setup_style(colors)` | Builds the stylesheet from tokens. Hard-coded hex in panels breaks light mode. | Grep `panels/` for hex literals. |
| `live_selectors._axes_ready` | A 0×0 hidden canvas makes `RectangleSelector` raise `ValueError`. | Run `tests/unit/test_live_selectors.py`. |
| `visionloader.EIReader` | Payload width comes from the `.ei` file. A globals-based stride invents cell IDs. | Run `tests/unit/test_vision_load_robustness.py`. |
| `_apply_ei_updates()` | `cluster_df` is main-thread only. `max_dup_r` is float64. | Run `test_apply_ei_updates_keeps_max_dup_r_as_float`. |
| `EIPanel._redraw_current_view()` | Shared handlers that call `_draw_heatmap_frame` steal the View combo. | Run `tests/unit/test_ei_panel_view.py`. |
| `build_feature_matrix()` | Missing STA/grating/chirp/RF rows are NaN. Filling them with 0 rebuilds the fake "no STA" cluster. | Run `tests/unit/test_dynamic_clustering.py`. Do not switch UMAP to sklearn `nan_euclidean` (MCAR scale-up). |

## 2. Tests to run after a change

| If you change | Run |
|---|---|
| Vision ID / `get_cell_physics` | `test_get_cell_physics_vision_id_offset` |
| ACG / standard plots | `test_acg_includes_late_spike_trains` |
| Caches | `tests/unit/test_data_manager_cache.py` |
| Lazy STA | `test_lazy_sta_dict_*` in `test_physics_cache_unified.py` |
| EI load / stride | `tests/unit/test_vision_load_robustness.py` |
| EI View / Overlay combo | `tests/unit/test_ei_panel_view.py` |
| Selectors / UMAP first paint | `tests/unit/test_live_selectors.py` |
| Dataset switch / memory | `tests/unit/test_dataset_release.py` |
| Theme tokens | `test_theme_keys_match` |
| Feature blocks / prefilter | `tests/unit/test_raw_feature_blocks.py` `tests/unit/test_dynamic_clustering.py` |
| Tree sidebar rows | `tests/unit/test_tree_rows.py` `tests/integration/test_tree_operations.py` |

Use `tmp_path` or `cache_cleared_data_manager` for any math test. A real
run folder can hold a warm `.pkl` and skip the code under test (Law 3).

The full suite still has older failures. Do not mark them skipped.

## 3. Open defects

| Defect | Effect | Status |
|---|---|---|
| Mixed no-STA cells share one temporal PCA point | Fake tight UMAP cluster | Fixed 2026-08-12. PCA fits only cells that have the block; missing rows are NaN. UMAP uses observed Euclidean (shared features only, no MCAR scale-up). Cells stay in the map. |
| Default weights: STA block ~400 vs ACG ~4 | Embedding is almost STA-only | Fixed 2026-08-12. Defaults are Temporal + ACG + RF diameter, each 10/10, grating and chirp off. Euclidean share is still `n_columns × weight²`. |
| EI View combo stolen by heatmap redraws | Combo said Waveform; canvas showed Heatmap | Fixed 2026-08-12. Shared handlers go through `_redraw_current_view`. Wheel on a closed combo is ignored. |
| Stale `feature_cache.pkl` with `_computed: True` and `timecourse=None` | Population panel reports no timecourses | Fixed 2026-08-12. `_physics_entry_is_fresh` recomputes once when an STA source appears (`_sta_checked`). |
| DS/OS slider writes `dsos_threshold` but grating panel uses 0.3 | Slider does not change the grating label | Fixed 2026-08-12 in code. Not user-checked (no grating/contrast run on hand). |
| `get_cell_physics()` indexes the full STA cube | Slow scroll with a cold cache | Fixed 2026-08-12. Params timecourse first. Cube only on a miss. |
| `_draw_plots()` redraws population panels on every selection | Chirp-view scroll is slow until cache is warm | Fixed 2026-08-12. Skip when the group timecourse and ACG caches already hold the subset. First visit of a group still draws. |
| Older pytest failures | Suite is not a clean gate | Open. Do not skip. `test_raw_feature_blocks` now matches the prefilter and scalar-column contracts. `test_gui_polish` still ERRORs: `qtbot` is missing because `pytest-qt` is not installed in `rgcviewer` (it is listed in `requirements-dev.txt`). |

## 4. Expected messages (not defects)

Leave these. Do not treat them as crashes.

| Message | Meaning | Action |
|---|---|---|
| STA provenance dialog; "N of M cells in the .sta do not exist in this sort" | The `.sta` is from an older sort | Use the noise-run STA or Map Reference |
| `ei=519, positions=512` in the EI panel | Converter wrote a mismatched EI | Leave the plot blank |
| `standard_plot_cache.pkl` discarded (too large / unreadable) | Cache file is stale | Next load rebuilds it |
| `retinanalysis` import skipped | Optional package is absent | Ignore |
| `PeakPropertyWindow` warning | scipy peak finder | Ignore |

## 5. Parked work

Do not start these unless the user asks.

| Item | Spec | Notes |
|---|---|---|
| UX / UI redesign | `docs/specs/ux_ui_redesign.md` | Spec only for later phases (browser, command palette, undo, auto-collapse, min 1100×650). This branch uses the mockup's flat Swiss structure but a blue/amber palette (not Bauhaus red). Table light-mode text, UMAP first-show layout, and plot Home/reset bars are in. Do not add the 1100×650 minimum. |
| Cross-run stimulus bridge lab acceptance | `docs/specs/cross_run_stimulus_bridge.md` | Code is in the tree. Lab AC is open. |
| Vision-only remaining gaps | `docs/specs/vision_standalone.md` | Missing `.sta` / `.params` no longer crash. |
| EI panel waveform view | none | View combo exists. Further waveform work is not started. |
| Desktop launcher / `update.sh` | none | Not started. |
| Firing-rate / burstiness embedding feature | none | Do not dump mean rate as a scalar. Needs a construction that separates high-baseline RGCs and bursty vs tonic firing (retina and, later, brain neurons) without letting spike count dominate as QC. ACG already carries some of this. Not started. |
| 3D UMAP view | none | The current 3D UMAP display is poor UX. A 3D embedding (or another embedding) may still be better *for clustering* than the 2D view; do not throw the extra dimension away without checking that. Not started. |
| Stimulus / epoch rasters | none | Chirp already has trial rasters. Grating and Contrast do not (grating spec chose a preferred-direction PSTH instead). Want rasters in more places: per-stimulus tabs and/or a dedicated raster surface. One proposal is to put a spike raster in the Raw tab so a run with no `.bin` (Vision-only, or raw disabled) still has a time view. Open design: keep rasters next to the stimulus they align to, vs one Raster page, vs a Raw fallback when voltage is missing. Do not start until that is decided. |

## 6. Commands

```bash
conda activate rgcviewer
python main.py
python -m pytest tests/unit/ -v
python -m pytest tests/unit/test_live_selectors.py tests/unit/test_vision_load_robustness.py tests/unit/test_ei_panel_view.py -v
```

Lab data (tests skip if unmounted):

```
Raw Litke:     /mnt/lab/Array-data/raw/20260506A/data009
Sorted/Vision: /mnt/lab/Array-data/sorted/20260506A/chunk10/kilosort2.5
```
