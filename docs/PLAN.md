# PLAN.md — current snapshot

Read `docs/AGENTS.md` before this file. This file is the pickup point.

Last updated: 2026-09-24. Branch: `dev-testing`.

## Active queue (2026-09 sweep)

This is the current work. Start here.

Sources: reports from two testers, the 2026-08-10 lab meeting,
and a code audit on 2026-09-24. `main` is production: `install.sh` updates
every user with `git pull --ff-only` on `main`. Work on `dev-testing`. Commit
each fix separately. Do not push or merge to `main` unless the user asks.

Verify each GUI fix on real data with `tools/gui_harness.py` (see §6). The
unit suite is weak (see defect "Older pytest failures"). A green suite is not
proof. Add one focused test per fix that fails on the old code.

Status: `done` = fixed and verified; `open` = not started; `wip` = started.

| # | Item | Source | Status | Notes |
|---|---|---|---|---|
| Q1 | STA tab does not redraw on first select of an uncached cell | Tester | done | Cause: with a raw file loaded, `_process_selection` waited for `FeatureWorker`, and `on_features_ready` redraws only EI/Waveforms/Standard. Harness `sta_refresh` with `HARNESS_DAT`: 0/12 drew before, 12/12 after. Test `test_selection_draws_active_tab.py`. |
| Q2 | STA image rescales every frame; gray must stay gray | Lab meeting | done | One symmetric scale per movie (±max|STA|), 0 = mid-gray, ImageItem levels fixed. Harness `sta_modes`: image median 0.50 on 3 cells. Test `test_sta_display.py`. |
| Q3 | STA heatmap (spatial) and space-time view | User | done | Display combo: Stimulus / Heatmap (dominant channel, CET-D1, colorbar) / Space–time (x–t and y–t through the fit centre, else the peak pixel). Harness `sta_modes` screenshots. |
| Q4 | STA panel keeps the previous cell's movie after a failed or missing STA | Audit | done | `_clear_all` drops the movie and stops the timer. Harness `sta_modes`: movie dropped, timer stopped. |
| Q5 | STA read blocks the GUI thread up to 8 s on a cold CIFS read | Audit | open | `STAPanel.update_view` reads `vision_stas[vid]` synchronously. 1–2 s seen during physics warm-up. |
| Q6 | Grating polar plot: no units, no error bars (SD across trials) | Lab meeting | done | Polar title and histogram axis carry the unit (spikes/s); ring values labelled; ±1 SD whiskers on the polar spokes and histogram bars; footnote states n. Falls back to SEM, labelled as SEM, for files without `sd_response`. Harness `grating`. |
| Q7 | DSI/OSI missing from the cluster table | Lab meeting | done | Never existed. New columns DS/OS, DSI, OSI (`attach_grating_columns`), same Auto pick as the Grating tab. Refreshed when the grating batch finishes and 400 ms after the DS/OS slider settles. Harness `grating_table`: 64 DS / 46 OS at 0.30, 47 / 7 at 0.60. Test `test_grating_table_columns.py`. |
| Q8 | Per-direction rasters around the polar plot (replace the 8-slot PSTH grid) | User, lab | done | `polar_raster_view.py`: one raster per direction at its angle, preferred direction highlighted, stimulus shaded. Condition combo (Auto / each condition) picks what stats, rasters and error bars show. Rasters need raw trials (`grating_status == "raw_only"`); analysed-only files show a note. Tests `test_polar_raster_view.py`. |
| Q9 | Grating math: timing from trial 0 only; p-value has no +1 correction; 0°/360° not merged; negative delta-rate enters vector sum; F1 window overruns stimTime | Audit | done | Fixed in `grating_calc.py`; see its Conventions block. `GRATING_SCHEMA_VERSION = 2` recomputes old cache rows once. Old vs new on all 201 cells of 20251212A/data018: no DS/OS change. Tests `test_grating_math.py`. Open question for the user: no multiple-comparison correction across conditions and DSI/OSI. |
| Q10 | First load: grating DS/OS batch is slow | User | wip | Math vectorized: 25 → 6.5 ms/cell. Still to measure: the real first-load timeline in the app (physics warm-up over CIFS, cache saves every 100 cells in `GratingBatchWorker`). |
| Q11 | Kilosort + Vision from separate directories builds the physics cache twice or freezes | Tester | done | Freeze: File ▸ Load Vision disabled the window and only the first reveal re-enabled it; `_finalize_dataset_load` now unlocks on every pass. Stale data: physics entries and `ei_corr_dict.pkl` carry `_vision_source`; a different Vision folder drops Vision caches, closes old readers and recomputes (untagged legacy entries are kept). A running warm-up is stopped before a manual attach. Harness `attach_vision` (and with `HARNESS_VISION=.../data017`): window enabled; 324/324 physics rows rebuilt from the new source. Test `test_vision_source_switch.py`. Not done: joining the old warm-up thread (it stops at its next cell). |
| Q12 | `retinanalysis` ignored even when installed | Tester | done; follow-up open | Cause: `install.sh` makes an isolated venv (`include-system-site-packages = false`), so a conda or editable `retinanalysis` is invisible. Also: `load_stim_timing` sets `block_id` / `d_timing` and nothing reads them, so the package has no effect today. Warning lowered to info; README "Optional packages" shows how to install it into `~/.encore/.venv`. User 2026-09-24: `retinanalysis` will be wired to DataJoint later. To do (not started): use DataJoint epoch timing in Encore. |
| Q13 | State from the previous dataset survives a dataset switch | Audit | open | Stale guards check cluster ID only. `_PCA_CACHE`, `EIPanel._ei_map_cache`, array image. Needs a dataset-generation token. |
| Q14 | Feature Extraction default scatters | User | done | `feature_catalog.DEFAULT_PANELS`: Temporal STA PC1 v PC2, ACG PC1 v PC2, Temporal STA PC1 v ACG PC1, Temporal STA PC1 v RF area (user: area, not diameter), then two `RANDOM_PAIR`s that repeat no other panel. The old empirically ranked defaults are now the first `FALLBACK_PANELS` (used when a run lacks a feature, e.g. no STA). The no-catalogue `_PLOT_META` path is unchanged. Harness `feature_defaults`. Test `test_feature_default_panels.py`. Needed Q34 to show the ACG pairs. |
| Q15 | Editable, remembered plot presets | Tester | open | |
| Q16 | Light mode has dark leftover panels | Tester | open | |
| Q17 | Window width cannot shrink on laptops; drag-resize only in full screen | Lab meeting, tester | open | |
| Q18 | Feature-extraction scatters have more contrast than the population view | Tester | open | |
| Q19 | Loading indicator wrong on dataset reload | Lab meeting | open | |
| Q20 | RF y-axis flip vs stimulus | Tester | open | `main` looks consistent after `5903177`. Confirm the reporter's version, then check on real data. |
| Q21 | Mosaic click-to-select fails when the table view is active | Audit | done | `_select_cluster_in_table` and the tree→table sync read a `_data` attribute the model does not have; both now use `_select_table_cluster_id`. `_is_syncing` is released in `finally`. Harness `selection_sync`. |
| Q22 | Small bugs | Audit | done | `summary_tab` slot → now redraws the EI panel; EI panel says why it is empty with no Vision EI and no raw file; `electrode_map` enum `or` fixed (set is unused today); classification save/load go through `get_vision_id_for_cluster` / `get_cluster_id_for_vision`; a failed CCG says so in the plot title. NEW: `src/gui/crash_guard.py` — PyQt6 aborts (SIGABRT, exit 134) on any exception in a slot; the guard logs to `~/.encore/logs/errors.log` and the status bar instead. Tests `test_crash_guard_and_small_fixes.py`. |
| Q23 | Population spike-rate plots like Vision | Tester | open | |
| Q24 | Classification file structure matches Vision | Tester | wip | File ▸ Save wrote `id path/` (no `All/`, one space); both exports now share `vision_classification_lines` → `id  All/path/`, the format of `data017.classificationYT.txt`. Ctrl+S now writes the class into the `.params` itself (Q31), which is what Vision shows. The `.txt` loader cut names at the first space (`All/ON/brisk transient/` → `ON/brisk`); fixed (`parse_classification_text`). Still ask tester what else "match Vision" means. |
| Q25 | Tooltip for "RF short vs long" | Tester | open | |
| Q26 | Vision fits at the lower bound (σx = σy = 1.00) show as real fits | Audit | open | 7.6% of cells on 20251212A/data018. Flag them. |
| Q27 | Infra: CI, pinned dependencies, installer preflight (dirty tree, branch, venv Python), prune the unit suite | User | open | No `.github/workflows`. Deps unpinned. Docs name a missing `CLAUDE.md`, `environment.yml`, `requirements-dev.txt`. |
| Q28 | UMAP does not separate RGC types well | User | open | Evaluate against labelled cells (Vision classification) before tuning. Check feature blocks, weights, missing-data handling. Q34: on a warm open the ACG block was empty for every cell, so the UMAP ran without ACG; re-evaluate after Q34. On 20251212A/data018 the STA/RF blocks also belong to other cells (Q32); use a matched run. |
| Q29 | Windows / plots overlay each other when switching tabs | User | done | Cause: `_finalize_dataset_load` (and the Vision-only load) called `sta_panel.show()` on a `QTabWidget` page; a page shown by hand stays visible over the current tab. Harness `tab_overlay` (27 switches, raw attached): 310 widgets of the STA page visible on other tabs before, 0 after. Now `callbacks.set_tab_available` enables/disables the tab instead; never `show()`/`hide()` a tab page. Tests `test_tab_overlay.py`; `test_vision_loaded_no_modal_gate` asserted the buggy call and now asserts the tab is enabled. |
| Q30 | Opening a dataset "messes up" the `.params`; Vision cannot open it | User | investigated; needs an example | Encore never wrote `.params` (no writer in any commit; `ParametersFileReader` opens `rb`). `data018.params` kept its 2026-03-26 mtime through ~20 opens today. Scan of all 413 `/mnt/lab/Array-data/sorted/*/*/data*/*.params` with a Vision-equivalent parser: 411 good; 2 damaged, neither with Encore's `*.ei_index.pkl` marker: `20260213A/kilosort25/data017` (zero bytes at row 598) and `20260305A/kilosort2_5/data011` (seek table disagrees with the data from row 11: half-saved). Vision's own save (`ParametersFile.flush`) rewrites the file in place, no temp file, no truncate, seek table last, and Vision opens `.params` read-write even to read it. An interrupted or concurrent Vision save gives exactly this damage. Ask the user: which dataset, what does Vision say? |
| Q31 | Ctrl+S saves the classification into the `.params` | User | done | `src/analysis/params_classification.py`: strict Vision-equivalent parse; rewrites only `classID` cells; temp file + fsync + read-back + `<name>.params.bak` + atomic rename; refuses damaged files and files changed since review (stamp). File ▸ Save Classification to Vision .params (Ctrl+S) and File ▸ Load Classification from Vision .params. Dialog on the first save per session, after an outside save, when cells lose a class, or when Q32 flags the files. Noisy moved Ctrl+S → Ctrl+Shift+N (`ux_ui_redesign.md` AC18). Checked: 411/411 lab files rebuild byte-identical with no change; Vision.jar reads written files; harness `params_save` on data018 and 20260220A/data022 (scratch copy; lab file stamp unchanged). Tests `test_params_classification.py`, `test_params_save_gui.py`. Not tested: writing on the CIFS mount itself (same mkstemp + `os.replace` path as `ei_index_cache`, which works there). |
| Q32 | Vision `.sta`/`.params` made from a different sort than the loaded Kilosort run | Found 2026-09-24 | done | Law 1 pairing is only right if the Vision files came from this sort. `src/analysis/vision_sort_check.py` fits RF centre (x0, y0) against array position (`x_um`, `y_um`) over paired cells; robust R² < 0.15 = mismatch. Matched folders 0.38–0.86; ids shuffled < 0.01; **20251212A/kilosort40/data018: 0.03** (its `.sta` is dated 2025-12-18, before the 2026-01-27 sort; `.neurons` matches the sort 208/208). So data018's STAs, RF fits and Vision classes are on other cells. Shown as a permanent status-bar label (no modal; see `_on_vision_loaded` note), in the Ctrl+S dialog, and as a warning strip on the STA panel (user 2026-09-24: "do whatever's better" → mark, do not hide; the check is a heuristic). Open for the lab: regenerate data018's Vision files from its sort. |
| Q34 | ACG block empty on a warm open (Feature Extraction, UMAP) | Found 2026-09-24 | done | ACG reached `feature_cache` only from `_compute_standard_plots`. A warm open restores `standard_plot_cache.pkl` and computes nothing, and a physics row saved without `acg` counted as fresh forever. 20251212A/data018: `acg_norm` in the standard cache for 324/324 cells, `acg` in physics for 0/324; the catalogue had no ACG PCs. `get_cell_physics` now copies it across (`_backfill_acg`, locks not nested). Harness `feature_defaults`: 18 → 21 features. Test `test_physics_acg_backfill.py`. |
| Q33 | DSI/OSI significance: no multiple-comparison correction across conditions | Audit, Q9 | done | User 2026-09-24: one test across conditions. `grating_calc._fill_pvalues`: every DSOS condition is shuffled when any reaches the floor; `family_pvalues` = max statistic over conditions per shuffle → `DSI_pvalue_fw` / `OSI_pvalue_fw`; `gate_pvalue` feeds `select_best_dsos_condition` (files without trials: per-condition p × n conditions). `GRATING_SCHEMA_VERSION = 3` recomputes cached rows once. Simulated untuned cells, 2 conditions: 9.75 % → 5 % pass. 20251212A/data018 (208 cells): DS 64 / OS 47 / none 97 before and after; 2 cells swap OS↔DS; 6.6 ms/cell. Grating tab shows the gate p. Test `test_grating_family_pvalue.py`. |

Parked, low priority: two datasets side by side; cluster matching in Encore.

This file lists standing decisions, fragile code, and open defects. It is
not a roadmap. The full UX redesign spec is still parked except the
color / Swiss-header / plot-refresh pass now on this branch. The locked
palette is in `docs/design/palette.md` (warm paper, warm ink never
`#000`, red / yellow / blue). Chrome accent is Bauhaus blue. Plots use
ink + blue + yellow on `surface`. See `docs/specs/ux_ui_redesign.md`.

Do not push unless the user asks.

User check 2026-08-12: load, UMAP defaults, no-STA cells, and EI View combo
are good. Contrast was not checked.

User check 2026-08-24: grating DS/OS on `1212A/data018-` (6 dirs per
`(bw, tf)`, not a 12-dir crossed grid). Grouping, best-run pick, and
population polar without STA are in. Spec:
`docs/specs/grating_dsos_flexible_conditions.md`.

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
| `pop_mosaic_canvas._pop_plot_state` | A `None` sentinel makes `hasattr` true and `.get` throw. Nested `fig.clear` / `add_axes` during selection paints recursively. | Use `pop_canvas_can_hot_swap`. Never assign `None`. Hot-swap highlight only. Defer full redraw with `QTimer.singleShot(0, ...)`. Run `tests/unit/test_dsos_population.py`. |
| `group_grating_conditions()` / `select_best_dsos_condition()` | An 8-dir cutoff tags 6-dir protocols as SF. Ranking by max `|DSI|` or a 2 Hz veto hides real DS/OS. | Plot the `(bw, tf)` pairs that ran. Rank classified pairs by peak response. Run `tests/unit/test_grating_calc.py` and `tests/unit/test_dsos_threshold.py`. |
| `grating_computed_cache.pkl` | Old DSOS/SF tags survive a protocol change. | `grating_entry_needs_recompute()` must drop the row. Do not reuse it. |

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
| Grating grouping / DSI/OSI pick | `tests/unit/test_grating_calc.py` `tests/unit/test_dsos_threshold.py` |
| Population DS/OS polar / hot-swap | `tests/unit/test_dsos_population.py` |

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
| DS/OS slider writes `dsos_threshold` but grating panel uses 0.3 | Slider does not change the grating label | Fixed 2026-08-12. User-checked 2026-08-24 on a 6-dir-per-condition grating run. |
| 6-dir `(bw, tf)` tagged SF; best run was max `|DSI|` | OSI/DSI missing; wrong condition shown | Fixed 2026-08-24. `MIN_DIRECTIONS_FOR_DSOS = 4`. Rank by peak response. |
| Grating batch after physics; 1000 shuffles | Second load wait | Fixed 2026-08-24. Batch starts with physics. 200 shuffles. Skip shuffle below 0.10. |
| `_pop_plot_state = None` plus nested mosaic draw | `AttributeError` and recursive QWidget repaint on cluster click | Fixed 2026-08-24. State is a dict or absent. Highlight-only hot-swap. |
| `get_cell_physics()` indexes the full STA cube | Slow scroll with a cold cache | Fixed 2026-08-12. Params timecourse first. Cube only on a miss. |
| `_draw_plots()` redraws population panels on every selection | Chirp-view scroll is slow until cache is warm | Fixed 2026-08-12. Skip when the group timecourse and ACG caches already hold the subset. First visit of a group still draws. |
| Older pytest failures | Suite is not a clean gate | Open. Do not skip. 2026-09-24 in `rgcviewer`, offscreen: 335 pass, 3 fail, all in `test_gui_polish.py` (PyQt6 `QMouseEvent` wants `QPointF`, test passes `QPoint`). `pytest-qt` and `pytest-mock` are not declared in `pyproject.toml`. `requirements-dev.txt` does not exist. The suite tests cache plumbing, not dataset switches or grating numerics. The user does not trust it (Q27). |

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
| UX / UI redesign | `docs/specs/ux_ui_redesign.md` | Spec only for later phases (browser, command palette, undo, auto-collapse, min 1100×650). Colors are the locked palette in `docs/design/palette.md`. Swiss 40px header (ENCORE + run meta + inline tabs) is in. Do not add the 1100×650 minimum. |
| Cross-run stimulus bridge lab acceptance | `docs/specs/cross_run_stimulus_bridge.md` | Code is in the tree. Lab AC is open. |
| Vision-only remaining gaps | `docs/specs/vision_standalone.md` | Missing `.sta` / `.params` no longer crash. |
| EI panel waveform view | none | View combo exists. Further waveform work is not started. |
| Desktop launcher / `update.sh` | none | Not started. |
| Firing-rate / burstiness embedding feature | none | Do not dump mean rate as a scalar. Needs a construction that separates high-baseline RGCs and bursty vs tonic firing (retina and, later, brain neurons) without letting spike count dominate as QC. ACG already carries some of this. Not started. |
| 3D UMAP view | none | The current 3D UMAP display is poor UX. A 3D embedding (or another embedding) may still be better *for clustering* than the 2D view; do not throw the extra dimension away without checking that. Not started. |
| Stimulus / epoch rasters | none | Chirp already has trial rasters. Grating and Contrast do not (grating spec chose a preferred-direction PSTH instead). Want rasters in more places: per-stimulus tabs and/or a dedicated raster surface. One proposal is to put a spike raster in the Raw tab so a run with no `.bin` (Vision-only, or raw disabled) still has a time view. Open design: keep rasters next to the stimulus they align to, vs one Raster page, vs a Raw fallback when voltage is missing. Do not start until that is decided. |

## 6. Commands

```bash
conda activate encore
python main.py
python -m pytest tests/unit/ -v
python -m pytest tests/unit/test_live_selectors.py tests/unit/test_vision_load_robustness.py tests/unit/test_ei_panel_view.py -v
```

On the lab workstation the working environment is conda `rgcviewer`.
User installs run from `~/.encore/.venv`.

```bash
# Unit suite, headless
QT_QPA_PLATFORM=offscreen conda run -n rgcviewer python -m pytest tests/unit -q

# Drive the real GUI offscreen on a real run; screenshots in $HARNESS_OUT
HARNESS_OUT=/tmp/encore_harness conda run --no-capture-output -n rgcviewer \
    python tools/gui_harness.py sta_refresh
# Same, with the raw .bin attached (FeatureWorker path)
HARNESS_DAT=/mnt/lab/Array-data/raw/20251212A/data018 \
    conda run --no-capture-output -n rgcviewer python tools/gui_harness.py sta_refresh
```

Harness dataset: `/mnt/lab/Array-data/sorted/20251212A/kilosort40/data018/ksfiles`
(Kilosort; Vision files one level up; DSOS grating file). Raw:
`/mnt/lab/Array-data/raw/20251212A/data018`. Its `.sta`/`.params` come from
a different sort (Q32): use it to test mechanics, not what an STA or RF
shows. Matched run (70 s cold load): pass
`/mnt/lab/Array-data/sorted/20260220A/kilosort25/data022/ksfiles` as the
second argument.

Scenario `params_save` copies the `.params` to `$HARNESS_OUT/params_save`
and points Encore at the copy. It never writes the lab file and checks
that its stamp did not change. It checks the result with Vision.jar when
one is found (`VISION_JAR`, default
`~/Documents/Development/MEA-fieldlab/src/vision7_symphony/Vision.jar`).

Lab data (tests skip if unmounted):

```
Raw Litke:     /mnt/lab/Array-data/raw/20260506A/data009
Sorted/Vision: /mnt/lab/Array-data/sorted/20260506A/chunk10/kilosort2.5
```
