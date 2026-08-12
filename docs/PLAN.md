# PLAN.md — current snapshot

Read `docs/AGENTS.md` and `HANDOFF.md` before this file.

Last updated: 2026-08-12. Branch: `dev-testing`.

This file lists fragile code and open defects. It is not a roadmap.
The UX redesign spec is parked. See `docs/specs/ux_ui_redesign.md`.

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

## 2. Tests to run after a change

| If you change | Run |
|---|---|
| Vision ID / `get_cell_physics` | `test_get_cell_physics_vision_id_offset` |
| ACG / standard plots | `test_acg_includes_late_spike_trains` |
| Caches | `tests/unit/test_data_manager_cache.py` |
| Lazy STA | `test_lazy_sta_dict_*` in `test_physics_cache_unified.py` |
| EI load / stride | `tests/unit/test_vision_load_robustness.py` |
| Selectors / UMAP first paint | `tests/unit/test_live_selectors.py` |
| Dataset switch / memory | `tests/unit/test_dataset_release.py` |
| Theme tokens | `test_theme_keys_match` |

Use `tmp_path` or `cache_cleared_data_manager` for any math test. A real
run folder can hold a warm `.pkl` and skip the code under test (Law 3).

The full suite still has older failures. Do not mark them skipped.

## 3. Open defects

| Defect | Effect | Status |
|---|---|---|
| Mixed no-STA cells share one temporal PCA point | Fake tight UMAP cluster | Open. Needs a product call: exclude from UMAP, or change default weights. Do not start. |
| Default weights: STA block ~400 vs ACG ~4 | Embedding is almost STA-only | Open. Scientific. Do not change defaults without the user. |
| Stale `feature_cache.pkl` with `_computed: True` and `timecourse=None` | Population panel reports no timecourses | Fixed 2026-08-12. `_physics_entry_is_fresh` recomputes once when an STA source appears (`_sta_checked`). |
| DS/OS slider writes `dsos_threshold` but grating panel uses 0.3 | Slider does not change the grating label | Fixed 2026-08-12. `select_dsos_for_display` plus `update_all` on slider move. |
| `get_cell_physics()` indexes the full STA cube | Slow scroll with a cold cache | Fixed 2026-08-12. Params timecourse first. Cube only on a miss. |
| `_draw_plots()` redraws population panels on every selection | Chirp-view scroll is slow until cache is warm | Fixed 2026-08-12. Skip when the group timecourse and ACG caches already hold the subset. First visit of a group still draws. |
| Older pytest failures | Suite is not a clean gate | Open. Do not skip. `__new__` physics tests that hit `getattr` on QObject now pass via `_optional_attr`. Remaining failures (qtbot, raw_feature_blocks column set) are older. |

## 4. Parked work

Do not start these unless the user asks.

| Item | Spec | Notes |
|---|---|---|
| UX / UI redesign | `docs/specs/ux_ui_redesign.md` | Spec only. This branch name refers to it. |
| Cross-run stimulus bridge lab acceptance | `docs/specs/cross_run_stimulus_bridge.md` | Code is in the tree. Lab AC is open. |
| Vision-only remaining gaps | `docs/specs/vision_standalone.md` | Missing `.sta` / `.params` no longer crash. |
| EI panel waveform view | none | Not started. |
| Desktop launcher / `update.sh` | none | Not started. |

## 5. Commands

```bash
conda activate rgcviewer
python main.py
python -m pytest tests/unit/ -v
python -m pytest tests/unit/test_live_selectors.py tests/unit/test_vision_load_robustness.py -v
```

Lab data (tests skip if unmounted):

```
Raw Litke:     /mnt/lab/Array-data/raw/20260506A/data009
Sorted/Vision: /mnt/lab/Array-data/sorted/20260506A/chunk10/kilosort2.5
```
