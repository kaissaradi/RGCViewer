# PLAN.md — RGCViewer Master Development Plan

> Read AGENTS.md before this document.
> This is a **snapshot of the current codebase state**, not a roadmap narrative.
> Update this file every time a spec completes or a new untested behavior is discovered.
> Last updated: 2026-08-12 | Branch: claude/plan-ux-ui-design-62g5jl (merged from claude/physics-cache-loading-98b5op)

---

## 1. Fragile Zones

These files and functions will silently break something if modified carelessly.
Read the "failure mechanism" column before touching anything in the "what" column.

| What | Failure mechanism | Required action before modifying |
|---|---|---|
| `data_manager.py` (whole file) | Every panel and cache depends on it. Concurrent write bugs are timing-dependent and hard to reproduce. | `git fetch && git rebase origin/main` before every push. Run full test suite after. |
| `get_cell_physics()` | Vision ID offset lives here. `vision_id = cluster_id + 1` in hybrid mode. Accessing wrong key returns wrong cell's STA silently. | Run `test_get_cell_physics_vision_id_offset` — both parametrize branches — after any change. |
| `build_cluster_dataframe()` | Runs a single `np.unique` scan that produces `_spk_unique_cls`, `_counts`. Three downstream functions consume these. A second scan breaks count assumptions and doubles load time. | Never add a second `np.unique` or `np.argsort` on the full spike arrays inside this function. |
| `update_cluster_views()` / `_process_selection()` | The Tier 1 / Tier 2 boundary. Any heavy operation added to Tier 1 freezes the UI during keypress scrolling. | Classify every new operation as Tier 1 or Tier 2 explicitly before writing code. See AGENTS.md §1 Law 2. |
| `LazySTADict` in `vision_integration.py` | Thread-local STAReader instances are spawned for background threads to ensure thread-safe, lock-free parallel reads from the SSD. Worker readers are tracked in `_all_readers` and closed in `__del__` to prevent handle leaks. | Run `test_lazy_sta_dict_reads_are_concurrent` and `test_lazy_sta_dict_cache_is_thread_safe` after any refactoring. |
| `_save_pickle_with_fallback()` | Uses `tempfile + os.replace()` for atomicity. Replacing with a direct `pickle.dump(open(path))` would leave a corrupt truncated file if the process crashes mid-write. | Never simplify this function. The verbosity is intentional. |
| `_compute_ei_correlations_if_needed()` | The `is_vision_only` guard prevents building a 512×512 correlation matrix that exhausts RAM on large Vision-native datasets. | The guard must never be removed or conditioned on anything else. |
| `get_cluster_spike_indices()` | Returns pre-built index arrays from `_cluster_spike_indices` dict. Callers assume O(1). Replacing with `np.where(spike_clusters == id)` inside any loop causes O(N × n_clusters) runtime. | Never bypass this method. Never call `np.where` on the full spike arrays in a hot path. |
| `theme.py` | All panels and the QSS stylesheet depend on the semantic token dictionary. Renaming a key or removing a color breaks the stylesheet and every `restyle_plots` call. | After any token change, toggle theme in-app and verify every tab in both modes. |
| `_setup_style(colors)` in `main_window.py` | Generates the ~300-line QSS stylesheet from the theme dictionary. Hard-coded colors in any panel bypass it and break on theme toggle. | Grep for hex literals in `panels/` after any style work. All colors must come from the `colors` dict. |

---

## 2. Test Coverage Map

### Tested — do not regress

| Behavior | Test function | File |
|---|---|---|
| ACG uses full recording, not first 2 min | `test_acg_includes_late_spike_trains` | `tests/unit/test_autocorrelation.py` |
| Too-few-spike clusters return `None` ACG | `test_acg_not_computed_for_too_few_spikes` | `tests/unit/test_autocorrelation.py` |
| Same cluster computed exactly once under concurrency | `test_standard_plot_cache_computes_same_cluster_once` | `tests/unit/test_data_manager_cache.py` |
| Different clusters can compute in parallel (no cross-lock) | `test_standard_plot_cache_allows_different_clusters_to_compute_concurrently` | `tests/unit/test_data_manager_cache.py` |
| Disk `.pkl` cache bypasses `_compute_standard_plots()` | `test_disk_cache_bypasses_computation` | `tests/unit/test_data_manager_cache.py` |
| Cell without Vision STA marked `_computed` with safe defaults | `test_cell_physics_marks_cluster_computed_without_vision_sta` | `tests/unit/test_data_manager_cache.py` |
| Raw trace snippet skips Litke TTL row | `test_raw_trace_snippet_skips_litke_ttl_row` | `tests/unit/test_data_manager_cache.py` |
| HDBSCAN runs as default, K-means as fallback (7 tests) | `tests/unit/test_hdbscan_clustering.py` | `tests/unit/test_hdbscan_clustering.py` |
| LazySTADict concurrent reads do not corrupt cache | `test_lazy_sta_dict_cache_is_thread_safe` | `tests/unit/test_physics_cache_unified.py` |
| LazySTADict SSD reads are concurrent (not serialised) | `test_lazy_sta_dict_reads_are_concurrent` | `tests/unit/test_physics_cache_unified.py` |
| Light/dark theme dictionaries have identical keys | `test_theme_keys_match` | `tests/unit/test_theme.py` |
| Sidebar search filters tree and table | `TestSidebarSearch` (7 tests) | `tests/unit/test_gui_polish.py` |
| UMAP layout no overlap on first visit | `TestUmapLayoutFix` (2 tests) | `tests/unit/test_gui_polish.py` |
| Tree branch CSS contains SVG triangles | `TestTreeBranchStyling` (2 tests) | `tests/unit/test_gui_polish.py` |

### Untested — add these before touching the corresponding code paths

| Behavior | Where to add | Priority | Notes |
|---|---|---|---|
| Vision ID offset — hybrid branch (`cluster_id + 1`) | `tests/unit/test_data_manager_cache.py` | **HIGH** | Parametrize over `is_vision_only=True/False` |
| Vision ID offset — vision-only branch (`cluster_id`) | `tests/unit/test_data_manager_cache.py` | **HIGH** | Same test, both branches |
| `on_features_ready()` discards result when cluster changed | `tests/integration/test_main_window.py` | **HIGH** | Use `qtbot` + `threading.Event` |
| Stale `standard_plot_cache` entries pruned after cluster refinement | `tests/unit/test_data_manager_cache.py` | **HIGH** | Inject stale key, call rebuild, assert key gone |
| Atomic pkl write: failure leaves original file intact | `tests/unit/test_data_manager_cache.py` | **HIGH** | `patch('os.replace', side_effect=OSError)` |
| `ei_corr()` with zero-std EI returns zeros, not `NaN` | `tests/unit/test_data_manager_cache.py` | MEDIUM | `np.ones((512, 201))` input |
| `_apply_ei_updates()` is called via signal, never directly | `tests/integration/test_data_manager_signals.py` | MEDIUM | Emit signal, assert `cluster_df` updated on main thread |
| Population mosaic `Show IDs` toggle invalidates hot-swap cache | `tests/integration/test_population_panel.py` | LOW | |
| Auto-save writes sidecar, never overwrites user file | `tests/unit/test_autosave.py` | MEDIUM | Mock `QTimer`, assert `.autosave` path |
| Theme toggle updates all plot backgrounds | `tests/integration/test_theme_toggle.py` | MEDIUM | Toggle, assert every `PlotWidget` bg matches `colors['bg_panel']` |
| Keyboard shortcut registry is complete | `tests/unit/test_shortcuts.py` | LOW | Assert every command-palette entry has a working callable |

---

## 3. Completed Fix Registry

Every completed fix is listed here with the exact test that would catch a regression.
Before modifying any file in the "changed files" column, run the corresponding test first.

| Fix | Changed files | Regression test | Regression risk |
|---|---|---|---|
| ACG uses full recording (not first 2 min) | `data_manager.py::_compute_standard_plots()` | `test_acg_includes_late_spike_trains` | HIGH — any change to `_compute_standard_plots` |
| Physics cache double-load on init | `data_manager.py::__init__`, `_load_standard_plot_cache_from_disk()` | `test_standard_plot_cache_computes_same_cluster_once` | MEDIUM |
| Vision-only subset-of-cells UMAP bug | `data_manager.py::load_vision_native_data()` | `real_data_manager` fixture — assert all cells in `cluster_df` | HIGH — any change to vision loading |
| `vision_channel_positions` unguarded assignment (full + partial load paths) | `data_manager.py::load_vision_data()`, `_partial_vision_reload()` | Add: assert `vision_channel_positions` is None when `electrode_map` contains NaN or values >100 000 µm | MEDIUM |
| `vision_channel_positions` never set in vision-only path | `data_manager.py::load_vision_native_data()` | Add: assert `vision_channel_positions is not None` after a vision-only load with valid `.globals` | MEDIUM |
| `plot_ei_waveforms` added to `analysis_core.py` | `analysis_core.py` | Smoke test: call with synthetic (519, 60) EI + positions, assert 519 artists returned | LOW |
| Cell Tracer EI waveform overlay on single-click | `cell_tracer_dialog.py` | Manual AC: open tracer, draw lasso, single-click row → waveforms appear on canvas; alpha slider → waveforms persist; Clear lasso → waveforms removed | MEDIUM |
| UMAP toolbar overlap on first render | `panels/umap_panel.py` | Visual AC — navigate directly to UMAP tab on cold launch | LOW |
| UMAP lasso/rect selector conflict | `panels/umap_panel.py` | Manual — activate both selectors in sequence | LOW |
| HDBSCAN as default clustering | `panels/umap_panel.py`, `workers/workers.py` | `tests/unit/test_hdbscan_clustering.py` (7 tests) | LOW |
| Population mosaic gridlines, zoom/pan, Show IDs cache | `main_window.py`, `panels/population_panel.py` | Integration tests on real dataset | MEDIUM |
| Light mode theme system | `src/gui/theme.py`, `main_window.py::_setup_style()` | Visual AC — toggle theme, check all panels | LOW |
| Sidebar live search (`Ctrl+F`) | `main_window.py` | Manual | LOW |
| Physics cache warm-up freeze on large datasets | `src/analysis/vision_integration.py` | `test_lazy_sta_dict_cache_is_thread_safe`, `test_lazy_sta_dict_reads_are_concurrent` | HIGH — any changes to LazySTADict concurrency, dict caching, or contains checking |
| Chirp PSTH PCA added as a UMAP feature block | `constants.py`, `analysis_core.py::build_feature_matrix`, `data_manager.py::get_raw_feature_blocks`, `panels/umap_panel.py` | `TestBuildFeatureMatrixChirp` (5, `test_dynamic_clustering.py`), `TestGetRawFeatureBlocksChirp` (5, `test_raw_feature_blocks.py`) | MEDIUM — spec `docs/specs/chirp_umap_feature_spec.md`. Block is additive/self-guarding (width-0 → skipped when no chirp file). |

---

## 4. Active Work

### Priority 0 — Cross-Run Stimulus Bridge (map any run → physics / RF mosaic / UMAP)
- **Spec:** `docs/specs/cross_run_stimulus_bridge.md`
- **Branch:** `feat/cross-run-stimulus-bridge`
- **Status:** Stages 1–4 implemented (unit tests green). Manual lab AC + optional panel borrow still open.
- **Goal:** EI-map any other run; load STA/RF + chirp + grating from reference when present; fill-gap into physics/UMAP; dashed borrowed RF mosaic; per-UI-id `match_caveats`.
- **What landed:**
  - `ReferenceBridge`: chirp/grating load, `CellMatchCaveat`, `build_ui_caveats`, stimulus inventory
  - `DataManager.install_reference_bridge` / `invalidate_physics_for_reference_bridge` / `effective_chirp_available` / `effective_grating_available`
  - `get_cell_physics` borrows with **ref_id** for params timecourse; provenance + match fields
  - `get_raw_feature_blocks` chirp/grating fill-gap
  - Population mosaic draws dashed borrowed ellipses; map Accept re-gates UMAP
  - Tests: `tests/unit/test_reference_bridge.py`, `tests/unit/test_cross_run_stimulus_bridge.py`
- **What remains:**
  - Manual AC on lab data (sibling runs with mixed stimuli)
  - Optional: ChirpPanel/GratingPanel show borrowed curves; second file dialog if npy not in Vision dir
- **Fragile zone overlap:** Touches `data_manager.py`. Rebase before every push.

### Priority 1 — Standalone Vision Integration
- **Spec:** `docs/specs/vision_standalone.md`
- **Branch:** `feat/vision-standalone`
- **Status:** In progress
- **What is done:** Basic `load_vision_native_data()` path exists. `is_vision_only` flag set correctly.
- **What remains:**
  - Subset-of-cells UMAP bug — not all Vision cells appearing. Root cause not yet confirmed.
  - EI waveform templates from `.ei` + `.globals` not yet integrated into Standard Plots panel.
  - Graceful fallback when `.sta` is missing — `STAPanel` currently crashes.
  - Graceful fallback when `.ei` is missing — `EIPanel` currently crashes.
- **Open architectural question:** When Vision-only, `cluster_id` is used directly as `vision_id`. The `cluster_df` index must be built from `neurons_data['spikes_by_id'].keys()` — confirm this is happening in `build_cluster_dataframe()` before adding any panel code.
- **Fragile zone overlap:** Touches `data_manager.py`. Rebase from main before every push.

---

### Priority 2 — EI Panel Waveform View Mode
- **Status:** Planned — not started
- **What it is:** A third view option in the EI panel (`View: Heatmap | Photo | Waveform`) that replaces the heatmap `ImageItem` with a matplotlib canvas rendering `plot_ei_waveforms` co-registered to electrode positions. Triggered via the existing `View:` toggle pattern. Photo underlay optional (toggle). Scrolling to a new cluster re-renders via the existing `_update_ei` path.
- **What remains (all of it):**
  - Add `"Waveform"` to the `View:` button group in `ei_panel.py`
  - Add a `FigureCanvas` widget that is shown/hidden based on mode (hides the `pyqtgraph ImageItem`, shows the mpl canvas)
  - Wire cluster selection → `_draw_ei_waveform_mode()` which calls `plot_ei_waveforms` onto the panel's axes
  - Implement artist cleanup between cluster changes (same `_ei_waveform_artists` list pattern as `CellTracerDialog`)
  - Optional: expose `Photo α` slider in the panel when waveform mode is active, using `ei_panel._overlay_image_rgba` if available
- **Key constraint:** Must not touch the existing heatmap render path — mode switch is purely additive. `_load_vision_ei()` and `_load_ks_ei()` stay unchanged.
- **Uses:** `plot_ei_waveforms` from `analysis_core.py` and `_resolve_channel_positions()` already in `ei_panel.py`.
- **Fragile zone overlap:** Touches `ei_panel.py` render path. Classify any new operation as Tier 1 or Tier 2 before writing (LAW 2).

---

### Priority 3 — Known issues found 2026-07-08, not yet fixed

- **Stale `feature_cache.pkl` is permanently sticky.** `get_cell_physics()` returns any cached entry flagged `_computed: True` without recomputation. If the cache was written while the Vision STA load was incomplete, entries land with `timecourse=None` and `rf_area=0` and are never repaired — the Population Dynamics panel then reports "No valid timecourses" for most cells. Observed on `20260623A-1`: 593 of 894 entries were poisoned this way. Workaround today is deleting the pickle. Fix would be to refuse to mark an entry `_computed` when the Vision block was skipped, or to version the cache.
- **DS/OS threshold slider does not affect the grating panel.** The slider writes `main_window.dsos_threshold`, which is only read by `population_panel.py:910` for the population RF markers. `grating_panel.py:367` calls `select_best_dsos_condition(data)` with no threshold argument, so that panel's `[not significant]` label always uses the hardcoded `DSI_THRESHOLD = 0.3`. Additionally the `pvalue < ALPHA (0.05)` gate runs before any DSI comparison, so lowering the slider could not rescue a borderline cell even if it were wired through. Either thread the threshold into `select_best_dsos_condition` or relabel the slider "Population DS/OS threshold".
- **`get_cell_physics()` reads the full STA cube it does not need.** It indexes `self.vision_stas[vid]`, forcing a full-movie disk read behind the 8 s `LazySTADict` timeout, even though the timecourse it ultimately uses comes from `vision_params['RedTimeCourse']`. Reading the cube should be a fallback, not the default path. This is the dominant cost when scrolling cells with a cold physics cache.
- **`_draw_plots()` redraws the population panels on every selection.** `main_window.py:886` calls `callbacks.redraw_population_panels(...)` whenever `population_view_enabled`, regardless of which tab is visible, invoking `get_cell_physics()` for every cell in the group. Combined with the cube read above, this makes chirp-view scrolling slow until the cache warms.
- **~19 pre-existing test failures on `tsting`, unrelated to any current feature work.** They fall into: a stale `HDBSCAN_AVAILABLE` import, calls to a `get_physics_feature_matrix` method that no longer exists, a precondition fixture yielding 0 valid cells, an `N=1` PCA edge case, and RF-mosaic/layout/debounce GUI tests whose functionality may no longer be wanted. These need triage into "update" vs "delete" before the suite can be trusted as a gate. Per AGENTS.md Rule 5 they have not been silenced.

---

### Priority 4 — UX / UI Redesign (Bauhaus Design Pass)

- **Spec:** `docs/specs/ux_ui_redesign.md`
- **Status:** Spec written. No code changes yet.
- **Goal:** Unified Bauhaus-informed visual and interaction redesign. Every plot, widget, and panel draws from one token palette. Both themes feel native. Classification workflow drops to ≤3 interactions per cell.
- **Scope:** `src/gui/` only — no analysis code changes.

**What the spec covers (8 phases):**

1. **Theme foundations** — extend `DARK_COLORS`/`LIGHT_COLORS` with new tokens (`accent_pressed`, `border_focus`, `bg_overlay`, shadows, tooltips), 12-color CVD-safe categorical palette for cell populations, spacing scale (4px grid), type scale.
2. **Layout** — three-column layout (sidebar / tabs / collapsible context panel), responsive minimum width for laptop screens (down to 1100px), invisible splitter handles, minimal panel headers. Mosaic drag-to-select (merge Anushka's prototype).
3. **Plot theming** — unified `apply_plot_theme()` for both pyqtgraph and matplotlib. Per-element token mapping for every plot. STA fixed symmetric color scale (gray = zero). Grating polar plot: axis units, SD error bars, per-direction rasters (Maria's request).
4. **Tree & table** — theme-aware colors (remove hard-coded `#3C3C3C`), geometric icons, cell count badges, status dot column, DSI/OSI columns restored, dimming inline search.
5. **Experiment browser** — sidebar panel scanning a user-configurable home directory, protocol detection from `<prep>.json`, filter field, recent paths, drag-to-load.
6. **Command palette & shortcuts** — `Ctrl+K` command palette, `Ctrl+1`–`Ctrl+8` tab switching, `Ctrl+S` save, `Ctrl+G` group, `Ctrl+M` quick move-to-group picker, `F2` rename, `Ctrl+Z`/`Ctrl+Shift+Z` undo/redo, status marking bar.
7. **Auto-save & session persistence** — configurable interval (off/2/5/10 min), sidecar `.autosave` file, session state save/restore, save indicator in title bar, loading indicator fix.
8. **Accessibility & polish** — focus rings, contrast audit, tooltip sweep, toast notifications, theme transition animation.

**Lab meeting feedback incorporated:**
- GUI width not reducible for laptop screens → responsive minimum width spec
- STA rescales to input range, gray drifts → symmetric fixed scale centered at zero
- Grating polar plot missing units and error bars → units label + SD whiskers
- Per-direction rasters requested (Maria) → raster subplots around polar plot
- DSI/OSI removed from table → restore as toggleable columns
- Mosaic drag-to-select not in main (Anushka has it) → merge into context panel
- Loading indicator buggy on reload → progress bar reset on every load path
- Export CSV in progress → completion spec with column list
- Save figure only on mosaic → extend to all tabs (PNG/SVG/PDF, publication white bg option)

**Files this priority will create:**

| Path | Purpose |
|---|---|
| `src/gui/panels/experiment_browser.py` | Experiment browser sidebar panel |
| `src/gui/command_palette.py` | Command palette dialog and action registry |
| `src/gui/undo.py` | Undo/redo stack for tree operations |
| `src/gui/toast.py` | Toast notification widget |

**Files this priority will modify heavily:**

| Path | What changes |
|---|---|
| `src/gui/theme.py` | New tokens, categorical palette, `apply_plot_theme()`, spacing/type constants |
| `src/gui/shortcuts.py` | Full shortcut registry replacing `KeyForwarder` |
| `src/gui/main_window.py` | Layout restructure, session persistence, status bar, auto-save |
| `src/gui/callbacks.py` | Auto-save, experiment browser integration, export completion |
| `src/gui/widgets/widgets.py` | Tree delegate update (geometric icons, count badges), table styling |
| `src/gui/panels/*.py` | Each panel: replace hard-coded colors with theme calls via `apply_plot_theme()` |

**Fragile zone overlap:** Touches `main_window.py`, `theme.py`, `_setup_style()`. Must not break existing `restyle_plots` calls. Run theme toggle verification on every tab after any change.

**Key constraint:** No analysis code changes. No `data_manager.py` changes. All work is in `src/gui/`.

---

## 5. UX / UI Design Specification

> Full spec: `docs/specs/ux_ui_redesign.md`
>
> This section summarizes the design decisions and acts as a quick reference.
> The spec has the implementation details. Read the spec before writing code.

### 5.1 Design Philosophy — Bauhaus Principles

Three tenets:

- **Form follows function.** Every visual element earns its space by serving the classification workflow. Decorative chrome is removed; whitespace and alignment do the organizing.
- **Reduction to essentials.** Controls used once per session (Load, Save, Export) live in the menu bar or command palette — not as permanent toolbar buttons. Secondary panels slide in on demand.
- **Unity of design.** Every surface draws from one token palette. A switch from dark to light mode changes the tokens, not the structure.

Design target: classify a cell in ≤3 interactions (select → read plots → drag to group).

### 5.2 Color System

#### Token Hierarchy

```
Tier 1 — Surface        bg_base, bg_panel, bg_surface, bg_elevated
Tier 2 — Content        text_primary, text_secondary, text_tertiary, text_disabled
Tier 3 — Interactive    accent, accent_hover, accent_pressed, accent_muted
                         border_subtle, border_default, border_focus
                         status_good_*, status_mua_*, status_noise_*, status_unsort_*
```

#### New Tokens

| Token | Dark | Light | Purpose |
|---|---|---|---|
| `accent_pressed` | `#1E4FA0` | `#3D7DE8` | Active-state feedback |
| `accent_muted` | `rgba(46,109,212,0.10)` | `rgba(46,109,212,0.08)` | Hover highlight |
| `border_focus` | `#4A8BEF` | `#2E6DD4` | Keyboard-focus ring (2px) |
| `bg_overlay` | `rgba(0,0,0,0.50)` | `rgba(0,0,0,0.25)` | Modal scrim |
| `bg_tooltip` | `#282A30` | `#FFFFFF` | Tooltip background |
| `text_tooltip` | `#F0F0F2` | `#111214` | Tooltip text |

#### Palette Adjustments

- Dark: `bg_panel` shifts `#18191C` → `#1A1B1F` (warmer, +2% contrast).
- Light: `bg_base` shifts `#F0F2F5` → `#F5F6F8` (softer). Plot backgrounds use `bg_surface`.

#### Categorical Palette (12 colors, CVD-safe)

```python
PLOT_CATEGORICAL = [
    ("#4FC3F7", "#0277BD"),   # sky
    ("#81C784", "#2E7D32"),   # green
    ("#FF8A65", "#D84315"),   # coral
    ("#BA68C8", "#7B1FA2"),   # violet
    ("#FFD54F", "#F9A825"),   # gold
    ("#4DD0E1", "#00838F"),   # teal
    ("#F06292", "#C2185B"),   # rose
    ("#A1887F", "#4E342E"),   # brown
    ("#AED581", "#558B2F"),   # lime
    ("#FF8A80", "#C62828"),   # red
    ("#CE93D8", "#6A1B9A"),   # lavender
    ("#80DEEA", "#006064"),   # cyan
]
```

Validation: every pair must pass WCAG AA (4.5:1) against its theme's `bg_panel`. No two adjacent colors confusable under deuteranopia/protanopia.

### 5.3 Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  Menu Bar                                            [☀] [⌘K]    │
├──────────┬──────────────────────────────────┬────────────────────────┤
│ Sidebar  │  Analysis Tabs                   │  Context Panel        │
│ (240px)  │  (flexible)                      │  (300px, collapsible) │
│          │                                  │                       │
│ ┌──────┐ │  ┌────────────────────────────┐  │  RF Mosaic            │
│ │Browse│ │  │                            │  │  ────────────         │
│ │ Tree │ │  │   Active Plot              │  │  [mosaic canvas]      │
│ │ /    │ │  │                            │  │                       │
│ │Table │ │  │                            │  │  Population           │
│ │      │ │  │                            │  │  ────────────         │
│ │      │ │  └────────────────────────────┘  │  [timecourse]         │
│ │      │ │                                  │  [acg]                │
│ │      │ │                                  │  [fr bar]             │
│ ├──────┤ │                                  │                       │
│ │Simil-│ │                                  │                       │
│ │arity │ │                                  │                       │
│ └──────┘ │                                  │                       │
├──────────┴──────────────────────────────────┴────────────────────────┤
│  Status Bar     [progress]     [auto-save ✓]     [cells: 342]      │
└─────────────────────────────────────────────────────────────────────┘
```

- **Responsive:** `MainWindow.setMinimumSize(1100, 650)`. Sidebar auto-collapses below 1300px. Context panel auto-collapses below 1400px. Tab content uses `QScrollArea` wrappers.
- **Test on:** 1280×800, 1366×768, 1440×900.
- **Splitters:** invisible handles, 1px `border_subtle`, 4px drag zone.
- **Context panel:** dedicated right column, collapsible via `Ctrl+\`. RF mosaic supports drag-to-select (merge from `anushka_dev`).

### 5.4 Typography & Spacing

| Role | Size | Weight | Use |
|---|---|---|---|
| `type_heading` | 13px | 600 | Panel titles, group names |
| `type_body` | 12px | 400 | Labels, controls, table cells |
| `type_caption` | 11px | 400 | Status text, axis labels, tooltips |
| `type_mono` | 11px | mono | Cluster IDs, numeric readouts |

Spacing: 4px grid (`sp_1`=4, `sp_2`=8, `sp_3`=12, `sp_4`=16, `sp_5`=24).
Radius: `radius_sm`=4px (buttons, inputs), `radius_md`=6px (panels, dialogs).

### 5.5 Keyboard Shortcuts

| Category | Shortcut | Action |
|---|---|---|
| **File** | `Ctrl+O` | Open experiment browser / filter |
| | `Ctrl+S` | Save classification |
| | `Ctrl+Shift+S` | Save classification as… |
| | `Ctrl+Shift+L` | Load classification file |
| **Navigation** | `Ctrl+1`–`Ctrl+8` | Switch to tab 1–8 (Standard…Raw) |
| | `Ctrl+Tab` / `Ctrl+Shift+Tab` | Cycle tabs |
| | `Ctrl+F` | Focus search / filter |
| | `Ctrl+L` | Toggle tree / table view |
| | `Ctrl+\` | Toggle context panel |
| | `↑` / `↓` | Move selection |
| | `←` / `→` | EI frame navigation |
| **Editing** | `Ctrl+G` | Group selected cells |
| | `Ctrl+M` | Move to group (quick picker) |
| | `F2` | Rename selected group |
| | `Delete` | Move to Trash |
| | `Ctrl+Z` / `Ctrl+Shift+Z` | Undo / Redo |
| **Status** | `Ctrl+D` | Mark Duplicate |
| | `Ctrl+C` | Mark Clean |
| | `Ctrl+E` | Mark Edge |
| | `Ctrl+W` | Mark Unsure |
| | `Ctrl+Shift+N` | Mark Noisy (moved from `Ctrl+S`) |
| | `Ctrl+X` | Mark Contaminated |
| | `Ctrl+A` | Mark Off Array |
| **View** | `Ctrl+Shift+T` | Toggle theme |
| | `Ctrl+K` | Command palette |
| | `Space` | Similarity panel action |

### 5.6 Experiment Browser

Collapsible sidebar panel above the tree/table. Scans a user-configured home directory (`QSettings: experiment/home_dir`) for `<prep>/kilosort25/` patterns. Shows run protocols from `<prep>.json`. Single-click loads a run. Filter field, recent paths section, drag-to-load on `MainWindow`.

### 5.7 Plot Theming

Unified `apply_plot_theme(widget, colors)` in `theme.py`. Applied on creation and on theme toggle.

**Pyqtgraph:** bg=`bg_panel`, axes=`border_default`, labels=`text_secondary`, grid=`border_subtle` 0.3α. No outer border.

**Matplotlib:** `figure.facecolor`=`bg_panel`, `axes.facecolor`=`bg_surface`, spines: left+bottom only, 0.5px, `border_subtle`. Legend: `bg_elevated`/`border_subtle`.

**STA:** Symmetric `clim` centered at zero (`(-absmax, +absmax)`) so gray = zero. Toggle: `[Symmetric] / [Full range]`.

**Grating polar:** Radial axis label (`spikes/s`). SD error bars as radial whiskers. Per-direction raster subplots (toggle).

**UMAP scatter:** `PLOT_CATEGORICAL` colors. Unassigned=`text_disabled` 30%α. Selected=full α + `plot_highlight` ring. Lasso=`accent` outline + `accent_muted` fill.

### 5.8 Tree & Table

- Remove hard-coded `#3C3C3C` folder bg → `bg_elevated`.
- Geometric icons: filled circle (group, `accent`), hollow circle (cell, `text_tertiary`), × (trash).
- Cell count badge: `OnP (12)` in `text_tertiary`.
- Table: row height 24px, alternating `bg_panel`/`bg_surface`, status dot (6px circle), restore DSI/OSI as toggleable columns.
- Search: dimming (non-matches → `text_disabled`) instead of hiding.

### 5.9 Auto-Save & Session

- **Auto-save:** configurable interval (off/2/5/10 min, default 5). Writes to `.autosave` sidecar. On load, prompts to restore if sidecar is newer.
- **Session persistence:** save splitter positions, active tab, theme, sidebar state, selected IDs on quit. Restore on launch.
- **Title bar:** `RGC Viewer — 20260721A / data006 ●` (● when unsaved).
- **Loading indicator fix:** reset progress bar to 0 at start of every load. Set to 100% or hide on completion/error.

### 5.10 Export & Save

- **CSV export:** one row per cell. Columns: `cluster_id`, `vision_id`, `group_path`, `status`, `spikes`, `channel`, `firing_rate`, `isi_violations`, `contamination_pct`, `amplitude`, `dsi`, `osi`, `sta_polarity`, `rf_x`, `rf_y`, `rf_diameter_long`, `rf_diameter_short`.
- **Save figure:** per-tab button. PNG 300 DPI (default), SVG, PDF. White-background option for publications.

---

### Running tests

```bash
# Full suite — always run before pushing
conda run -n rgcviewer python -m pytest tests/ -v

# Unit tests only — fast, no real data
conda run -n rgcviewer python -m pytest tests/unit/ -v

# Single test
conda run -n rgcviewer python -m pytest tests/unit/test_autocorrelation.py::test_acg_includes_late_spike_trains -v

# Stop on first failure
conda run -n rgcviewer python -m pytest tests/ -x -v
```

### Real data paths

```
Raw Litke:     /mnt/lab/Array-data/raw/20260506A/data009
Sorted/Vision: /mnt/lab/Array-data/sorted/20260506A/chunk10/kilosort2.5
```

Tests using `real_data_manager` automatically skip if these paths are unmounted.

### Cache invalidation rule

Any test that verifies computation logic (ACG, ISI, physics, EI correlation) must use `tmp_path` or `cache_cleared_data_manager`. Using a real data path risks loading a warm `.pkl` and skipping all math. See AGENTS.md §1 Law 3.

### Screenshot storage

| Type | Location | Git-tracked? |
|---|---|---|
| Visual AC verification | `tests/screenshots/` | No — gitignored, auto-deleted |
| Visual regression baselines | `tests/baseline_images/` | Yes — generate with `--mpl-generate-path` |

Filename format: `ac{N}_{feature_name}_{dark|light}.png`
Default capture window size: 1800 × 1000px
Always capture both light and dark mode for any visual AC.
