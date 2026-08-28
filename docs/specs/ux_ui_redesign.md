# Specification: UX / UI Redesign — Bauhaus Design Pass

**Status (2026-08-12):** Phase 1 tokens + Swiss header are in progress
on `feat/bauhaus-redesign`. Colors are the locked eight-token palette in
`docs/design/palette.md` (paper / ink / rule / red / yellow / blue). The
encore mockup is a layout idea only — do not copy its older `#e30613` or
cool grays. Later phases (browser, command palette, undo, auto-collapse,
1100×650 minimum) stay parked.

## Block 0 — Metadata

| Field | Value |
|---|---|
| **Date created** | 2026-08-12 |
| **Last updated** | 2026-08-12 |
| **Commit hash when spec was written** | `9d53e57` |
| **Branch** | `claude/plan-ux-ui-design-62g5jl` |
| **Author** | Kais / Claude |
| **Spec status** | Parked |

---

## Block 1 — Problem Statement

**Symptom:** The GUI has accumulated hard-coded colors, inconsistent spacing, and
ad-hoc keyboard handling across panels. Theme toggle breaks some plots. Laptop
screens (1280×768) cannot shrink the window enough. Classification workflow
requires too many clicks. There is no way to browse experiments without
File → Open. Lab meeting feedback (STA gray drift, grating plot missing units,
DSI/OSI removed from table, mosaic drag-to-select not merged, loading indicator
bugs) remains unaddressed.

**Root cause:** No unified design system. `theme.py` defines tokens but panels
hard-code hex values (grep finds `#3C3C3C` in `widgets.py`, hex literals in
several panel stylesheets). `KeyForwarder` handles shortcuts but there is no
discoverable shortcut registry or command palette. Layout is a single QSplitter
with hard-coded 220px sidebar — no responsive behavior, no collapsible context
panel.

**User story:** "As a scientist classifying hundreds of RGC types in a session,
I want every surface, plot, and control to feel like one coherent tool so that
I spend my attention on the biology, not the interface."

---

## Block 2 — Vision ID Contract

`N/A — this spec does not access Vision data.`

All changes are in `src/gui/`. No panel code will read `vision_stas`,
`vision_eis`, or `vision_params` differently. Plot theming applies styling
to existing plot widgets without altering the data they display.

---

## Block 3 — Affected Files

This spec is large. It is organized into 8 implementation phases. Each phase
is a standalone commit (or series). No phase depends on a later phase.

### Phase 1 — Theme Foundations

| File path | Function(s) added or modified | Change type | Touches DataManager? |
|---|---|---|---|
| `src/gui/theme.py` | Add tokens, `PLOT_CATEGORICAL`, `apply_plot_theme()`, spacing/type constants | Modify | No |
| `tests/unit/test_theme.py` | `test_new_tokens_in_both_themes`, `test_categorical_contrast`, `test_apply_plot_theme_*` | Add | No |

### Phase 2 — Layout Restructure

| File path | Function(s) added or modified | Change type | Touches DataManager? |
|---|---|---|---|
| `src/gui/main_window.py` | `__init__` layout (three-column splitter), `setMinimumSize`, responsive collapse logic | Modify | No |
| `src/gui/main_window.py` | `_setup_style()` QSS overhaul | Modify | No |
| `tests/integration/test_layout.py` | `test_minimum_size`, `test_sidebar_collapse`, `test_context_panel_toggle` | Add | No |

### Phase 3 — Plot Theming

| File path | Function(s) added or modified | Change type | Touches DataManager? |
|---|---|---|---|
| `src/gui/theme.py` | `apply_plot_theme()` pyqtgraph + matplotlib paths | Modify | No |
| `src/gui/panels/chirp_panel.py` | Replace hard-coded colors with `apply_plot_theme()` call | Modify | No |
| `src/gui/panels/population_panel.py` | Replace hard-coded colors, STA symmetric scale | Modify | No |
| `src/gui/panels/umap_panel.py` | `PLOT_CATEGORICAL` colors, selection ring | Modify | No |
| `src/gui/panels/rf_map_widget.py` | Theme-aware mosaic, drag-to-select (merge from `anushka_dev`) | Modify | No |
| `src/gui/panels/feature_extraction.py` | Replace hard-coded colors | Modify | No |
| `src/gui/plot_export.py` | White-background publication option, per-tab save | Modify | No |
| `tests/integration/test_plot_theming.py` | `test_theme_toggle_updates_all_plots`, `test_sta_symmetric_scale` | Add | No |

### Phase 4 — Tree & Table

| File path | Function(s) added or modified | Change type | Touches DataManager? |
|---|---|---|---|
| `src/gui/widgets/widgets.py` | Geometric icons, count badges, status dot, DSI/OSI columns, dimming search | Modify | No |
| `src/gui/main_window.py` | Table model updates for new columns | Modify | No |
| `tests/unit/test_tree_table.py` | `test_geometric_icons`, `test_dsi_osi_columns`, `test_dimming_search` | Add | No |

### Phase 5 — Experiment Browser

| File path | Function(s) added or modified | Change type | Touches DataManager? |
|---|---|---|---|
| `src/gui/panels/experiment_browser.py` | `ExperimentBrowser` widget (NEW) | Add | No |
| `src/gui/main_window.py` | Integrate browser into sidebar, drag-to-load | Modify | No |
| `src/gui/recent_paths.py` | Add `experiment_home_dir` QSettings key | Modify | No |
| `tests/unit/test_experiment_browser.py` | `test_scan_home_dir`, `test_filter`, `test_protocol_detection` | Add | No |

### Phase 6 — Command Palette & Shortcuts

| File path | Function(s) added or modified | Change type | Touches DataManager? |
|---|---|---|---|
| `src/gui/command_palette.py` | `CommandPalette` dialog, action registry (NEW) | Add | No |
| `src/gui/shortcuts.py` | Full shortcut registry replacing `KeyForwarder` | Modify | No |
| `src/gui/undo.py` | `UndoStack` for tree operations (NEW) | Add | No |
| `src/gui/main_window.py` | Wire shortcuts, install palette | Modify | No |
| `tests/unit/test_shortcuts.py` | `test_shortcut_registry_complete`, `test_command_palette_filter` | Add | No |
| `tests/unit/test_undo.py` | `test_undo_group`, `test_redo_rename`, `test_stack_limit` | Add | No |

### Phase 7 — Auto-Save & Session Persistence

| File path | Function(s) added or modified | Change type | Touches DataManager? |
|---|---|---|---|
| `src/gui/main_window.py` | Auto-save timer, session save/restore, title bar indicator | Modify | No |
| `src/gui/callbacks.py` | Auto-save write, sidecar restore prompt | Modify | No |
| `tests/unit/test_autosave.py` | `test_sidecar_path`, `test_interval_config`, `test_no_overwrite_user_file` | Add | No |

### Phase 8 — Accessibility & Polish

| File path | Function(s) added or modified | Change type | Touches DataManager? |
|---|---|---|---|
| `src/gui/toast.py` | `ToastWidget` notification (NEW) | Add | No |
| `src/gui/main_window.py` | Focus rings, toast integration, status bar | Modify | No |
| `src/gui/theme.py` | Tooltip tokens, transition timing | Modify | No |
| `tests/unit/test_toast.py` | `test_toast_shows_and_fades`, `test_toast_queue` | Add | No |

### Grating Panel Improvements (Phase 3 sub-task)

| File path | Function(s) added or modified | Change type | Touches DataManager? |
|---|---|---|---|
| `src/gui/panels/grating_panel.py` | Polar plot axis units (`spikes/s`), SD error bars, per-direction rasters | Modify | No |
| `tests/integration/test_grating_visual.py` | `test_polar_units_label`, `test_error_bars_visible` | Add | No |

> **No row touches DataManager.** This entire spec is gui-only.

---

## Block 4 — Qt Threading Contract

### New operations

| Operation | Runs on thread | Worker class | Signal name + signature | Receiving slot | Tier |
|---|---|---|---|---|---|
| Experiment browser directory scan | Background | `QRunnable` (one-shot) | `scan_complete = Signal(list)` | `ExperimentBrowser._on_scan_complete(entries)` | N/A (not in selection path) |
| Auto-save write | Main thread (QTimer) | N/A | N/A | `callbacks._autosave_tick()` | N/A (timer, not selection) |

### Existing operations — no changes

All existing Tier 1 / Tier 2 operations in `update_cluster_views()` and
`_process_selection()` are unchanged. This spec adds no code to either path.

### Tier classification for new interactive operations

| Operation | Tier | Justification |
|---|---|---|
| Theme toggle (restyle all plots) | Tier 2 | Already debounced via menu action; no selection path involvement |
| Command palette open/search | N/A | Modal dialog, blocks main thread by design (standard Qt) |
| Undo/redo tree operation | Tier 2 | Triggered by shortcut, rebuilds tree view; not called during selection |
| Sidebar collapse/expand | N/A | Splitter resize, instant Qt geometry operation |

**Stale result guard:** The experiment browser scan is the only new async
operation. Its slot checks `self._scan_generation` counter before applying
results — a newer scan invalidates an in-flight one.

---

## Block 5 — Cache Contract

| Question | Answer |
|---|---|
| Cache read | None. This spec does not read any DataManager caches. |
| Cache written | None. |
| Invalidation trigger | N/A |
| Persisted to disk | Session state persisted via `QSettings` (splitter sizes, active tab, theme, auto-save interval). Auto-save writes a `.autosave` sidecar classification file — not a DataManager cache. |
| Lock required | None |
| Must tests bypass cache? | **No** — no DataManager caches are involved |

---

## Block 6 — DataManager Attributes Used

`N/A — this spec does not read or write any DataManager attributes.`

All data displayed in plots is already loaded and rendered by existing panel
code. This spec changes how that data is styled, not what data is shown.
The one exception is the experiment browser, which reads the filesystem
directly (not through DataManager).

---

## Block 7 — Acceptance Criteria

### Phase 1 — Theme Foundations

#### AC1 — New tokens exist in both theme dictionaries

- **Setup:** Import `DARK_COLORS` and `LIGHT_COLORS` from `theme.py`.
- **Action:** Check for `accent_pressed`, `accent_muted`, `border_focus`, `bg_overlay`, `bg_tooltip`, `text_tooltip` in both dicts.
- **Expected:** All six keys present in both. Values are valid CSS color strings.
- **Test type:** Unit

#### AC2 — Categorical palette has 12 entries with sufficient contrast

- **Setup:** Import `PLOT_CATEGORICAL` from `theme.py`.
- **Action:** For each `(dark_variant, light_variant)` pair, compute contrast ratio against the respective theme's `bg_panel`.
- **Expected:** 12 pairs. Every pair passes WCAG AA (≥4.5:1 contrast ratio against its theme's `bg_panel`).
- **Test type:** Unit

#### AC3 — `apply_plot_theme()` sets pyqtgraph widget colors

- **Setup:** Create a `pyqtgraph.PlotWidget`. Call `apply_plot_theme(widget, dark_colors)`.
- **Action:** Read back `widget.getViewBox().background` and axis pen colors.
- **Expected:** Background matches `bg_panel`. Axis pen color matches `border_default`.
- **Test type:** Unit

### Phase 2 — Layout

#### AC4 — Minimum window size is 1100×650

- **Setup:** Create `MainWindow`.
- **Action:** Read `minimumSize()`.
- **Expected:** Width ≥ 1100, height ≥ 650.
- **Test type:** Unit

#### AC5 — Sidebar auto-collapses below 1300px window width

- **Setup:** Create `MainWindow` at 1400px width. Sidebar is visible.
- **Action:** Resize to 1200px.
- **Expected:** Sidebar splitter size is 0 (collapsed). Resize back to 1400px — sidebar reappears.
- **Test type:** Integration

#### AC6 — Context panel toggles with `Ctrl+\`

- **Setup:** Create `MainWindow`. Context panel is visible.
- **Action:** Simulate `Ctrl+\` keypress.
- **Expected:** Context panel width becomes 0. Second `Ctrl+\` restores it.
- **Test type:** Integration

### Phase 3 — Plot Theming

#### AC7 — Theme toggle updates all plot backgrounds

- **Setup:** Create `MainWindow` in dark mode. Navigate to each tab.
- **Action:** Toggle to light mode.
- **Expected:** Every `PlotWidget` background matches `LIGHT_COLORS['bg_panel']`. Every matplotlib figure facecolor matches `LIGHT_COLORS['bg_panel']`.
- **Test type:** Integration

#### AC8 — STA display uses symmetric color scale

- **Setup:** Load a dataset with STA data. Select a cell.
- **Action:** Navigate to STA tab. Read the colorbar limits.
- **Expected:** `clim = (-absmax, +absmax)` where `absmax = max(abs(sta_data))`. Gray (mid-scale) corresponds to zero. Toggle button `[Symmetric] / [Full range]` switches behavior.
- **Test type:** Integration / Manual

#### AC9 — Grating polar plot shows axis units and SD error bars (Visual)

- **State to reproduce:**
  1. Load a dataset that has `*_GratingDSOS.npy`.
  2. Select a cell with significant direction selectivity.
  3. Navigate to Grating tab.
- **Expected appearance:** Radial axis is labeled `spikes/s`. Each direction has an SD whisker (radial error bar). Per-direction rasters visible when toggle is on.
- **Must verify:** Dark mode AND light mode.
- **Screenshot filenames:**
  - `tests/screenshots/ac9_grating_polar_dark.png`
  - `tests/screenshots/ac9_grating_polar_light.png`
- **Verified by:** `[ ]` Author `[ ]` Reviewer

#### AC10 — UMAP scatter uses categorical palette

- **Setup:** Load dataset, run UMAP clustering.
- **Action:** Verify scatter point colors.
- **Expected:** Each cluster group uses a color from `PLOT_CATEGORICAL` (dark variant in dark mode, light variant in light mode). Unassigned cells use `text_disabled` at 30% alpha.
- **Test type:** Manual

### Phase 4 — Tree & Table

#### AC11 — No hard-coded hex colors in tree/table rendering

- **Setup:** Grep `src/gui/widgets/widgets.py` for hex color literals.
- **Action:** Count matches excluding `theme.py` imports.
- **Expected:** Zero hex literals. All colors reference the `colors` dict.
- **Test type:** Unit (grep-based or AST scan)

#### AC12 — DSI/OSI columns present in table view

- **Setup:** Load a dataset with grating data.
- **Action:** Switch to table view. Check column headers.
- **Expected:** `DSI` and `OSI` columns are present and populated. Togglable via right-click header context menu.
- **Test type:** Integration / Manual

#### AC13 — Search uses dimming instead of hiding

- **Setup:** Load dataset. Switch to tree view. Type a search query.
- **Action:** Observe non-matching items.
- **Expected:** Non-matching items remain visible but rendered in `text_disabled` color. Matching items retain normal `text_primary`. Clearing search restores all items.
- **Test type:** Unit

### Phase 5 — Experiment Browser

#### AC14 — Browser scans home directory and lists preparations

- **Setup:** Create a temp directory with `prepA/kilosort25/data006/` and `prepB/kilosort25/data007/` structures.
- **Action:** Set `experiment_home_dir` in QSettings. Open experiment browser.
- **Expected:** Both preparations appear. Each shows its run list. Protocol labels populated from `<prep>.json` if present, `[unknown]` otherwise.
- **Test type:** Unit

#### AC15 — Single-click loads a run

- **Setup:** Experiment browser populated with valid runs.
- **Action:** Single-click a run entry.
- **Expected:** `callbacks.load_data()` is called with the correct path. Loading indicator appears.
- **Test type:** Integration

### Phase 6 — Command Palette & Shortcuts

#### AC16 — `Ctrl+K` opens command palette

- **Setup:** `MainWindow` is focused.
- **Action:** Press `Ctrl+K`.
- **Expected:** Floating dialog appears with search field focused. Typing filters the action list. `Escape` closes without action. Selecting an entry executes it.
- **Test type:** Integration

#### AC17 — `Ctrl+1` through `Ctrl+8` switch tabs

- **Setup:** `MainWindow` with all tabs available.
- **Action:** Press `Ctrl+3`.
- **Expected:** Tab index 2 (Grating) is now active. Works for all 1–8.
- **Test type:** Unit

#### AC18 — `Ctrl+S` saves, `Ctrl+Shift+N` replaces `Ctrl+S` for Noisy status

- **Setup:** `MainWindow` with a classification loaded and a cell selected.
- **Action:** Press `Ctrl+S`.
- **Expected:** Classification file is saved (not status-marked as Noisy). `Ctrl+Shift+N` marks Noisy.
- **Test type:** Unit

#### AC19 — Undo/redo works for tree operations

- **Setup:** Load dataset. Create a group. Move cells into it.
- **Action:** Press `Ctrl+Z`.
- **Expected:** Cells return to their previous group. `Ctrl+Shift+Z` re-applies the move.
- **Test type:** Unit

### Phase 7 — Auto-Save & Session

#### AC20 — Auto-save writes sidecar, never overwrites user file

- **Setup:** Load a dataset. Classification file is `test.classification_MC.txt`.
- **Action:** Wait for auto-save interval.
- **Expected:** `.autosave` file exists alongside the classification. Original file is byte-identical to before. Sidecar contains the current state.
- **Test type:** Unit

#### AC21 — Session state persists across launches

- **Setup:** Set splitter positions, active tab, theme. Quit.
- **Action:** Relaunch.
- **Expected:** Splitter positions, active tab, theme, sidebar state restored from QSettings.
- **Test type:** Integration / Manual

#### AC22 — Loading indicator resets properly on reload

- **Setup:** Load a dataset. Load completed.
- **Action:** Load a second dataset (or reload the same one).
- **Expected:** Progress bar resets to 0% at the start of the load. Reaches 100% or hides on completion. No stuck-at-50% state.
- **Test type:** Manual

### Phase 8 — Accessibility & Polish

#### AC23 — Focus rings visible on keyboard navigation

- **Setup:** `MainWindow` in dark mode.
- **Action:** Tab through interactive controls.
- **Expected:** A 2px `border_focus` ring appears around the focused widget. Ring disappears on blur.
- **Test type:** Manual

#### AC24 — Toast notifications appear and fade

- **Setup:** Trigger a save action.
- **Action:** Observe.
- **Expected:** Toast appears at bottom-right: "Classification saved". Fades after 3 seconds. Multiple toasts queue vertically.
- **Test type:** Unit

---

## Block 8 — Regression Guard

| Prior fix | Files overlap | Regression test to run | When to run it |
|---|---|---|---|
| Light mode theme system | `theme.py`, `main_window.py::_setup_style()` | Visual AC — toggle theme, check all panels | After every Phase 1–3 commit |
| Sidebar live search (`Ctrl+F`) | `main_window.py` | `TestSidebarSearch` (7 tests in `test_gui_polish.py`) | After Phase 4, Phase 6 |
| UMAP toolbar overlap on first render | `panels/umap_panel.py` | Visual AC — navigate directly to UMAP tab on cold launch | After Phase 2, Phase 3 |
| Population mosaic gridlines, zoom/pan | `panels/population_panel.py` | Integration tests on real dataset | After Phase 3 |
| HDBSCAN as default clustering | `panels/umap_panel.py` | `tests/unit/test_hdbscan_clustering.py` (7 tests) | After Phase 3 |
| Chirp PSTH PCA feature block | `panels/umap_panel.py`, `constants.py` | `TestBuildFeatureMatrixChirp` (5 tests) | After Phase 3 |
| Physics cache warm-up freeze | `vision_integration.py` (not touched by this spec) | N/A — no overlap | N/A |

> Run every test in this table before opening any PR for the corresponding phase.

---

## Block 9 — Test Plan

### Unit tests

File: `tests/unit/test_theme.py` (extend existing)

| Test function name | Fixture | What it asserts | Cache bypass? |
|---|---|---|---|
| `test_new_tokens_in_both_themes` | None | All 6 new token keys exist in `DARK_COLORS` and `LIGHT_COLORS` | No |
| `test_categorical_palette_length` | None | `len(PLOT_CATEGORICAL) == 12` | No |
| `test_categorical_contrast_ratio` | None | Each pair passes WCAG AA (≥4.5:1) against respective `bg_panel` | No |
| `test_apply_plot_theme_pyqtgraph` | `qtbot` | PlotWidget bg and axis colors match theme dict | No |
| `test_apply_plot_theme_matplotlib` | None | Figure/axes colors match theme dict | No |
| `test_spacing_constants` | None | `SP_1=4, SP_2=8, SP_3=12, SP_4=16, SP_5=24` | No |

File: `tests/unit/test_shortcuts.py` (new)

| Test function name | Fixture | What it asserts | Cache bypass? |
|---|---|---|---|
| `test_shortcut_registry_complete` | None | Every entry in shortcut table has a registered `QShortcut` with a connected callable | No |
| `test_tab_switching_shortcuts` | `qtbot` | `Ctrl+1` through `Ctrl+8` activate correct tab index | No |
| `test_ctrl_s_saves_not_marks_noisy` | `qtbot` | `Ctrl+S` calls save, not status mark | No |
| `test_ctrl_shift_n_marks_noisy` | `qtbot` | `Ctrl+Shift+N` calls `_mark_status("Noisy")` | No |
| `test_command_palette_filter` | `qtbot` | Typing in palette filters action list by substring | No |

File: `tests/unit/test_undo.py` (new)

| Test function name | Fixture | What it asserts | Cache bypass? |
|---|---|---|---|
| `test_undo_group_creation` | `qtbot` | Undo reverses group creation | No |
| `test_redo_after_undo` | `qtbot` | Redo re-applies undone operation | No |
| `test_undo_stack_limit` | None | Stack holds ≤100 operations, oldest dropped | No |
| `test_undo_move_cells` | `qtbot` | Undo returns cells to previous group | No |

File: `tests/unit/test_autosave.py` (new)

| Test function name | Fixture | What it asserts | Cache bypass? |
|---|---|---|---|
| `test_sidecar_path_derivation` | `tmp_path` | `.autosave` path is alongside classification file | No |
| `test_autosave_interval_config` | None | QSettings stores/retrieves interval (off/2/5/10) | No |
| `test_no_overwrite_user_file` | `tmp_path` | Original file unchanged after auto-save | No |
| `test_autosave_disabled` | `tmp_path` | Interval=0 means no timer fires | No |

File: `tests/unit/test_experiment_browser.py` (new)

| Test function name | Fixture | What it asserts | Cache bypass? |
|---|---|---|---|
| `test_scan_finds_preparations` | `tmp_path` | Detects `<prep>/kilosort25/` pattern | No |
| `test_scan_reads_protocol_from_json` | `tmp_path` | Protocol label matches `<prep>.json` content | No |
| `test_filter_narrows_entries` | `tmp_path` | Typing "data006" hides entries without that run | No |
| `test_empty_home_dir_shows_placeholder` | `tmp_path` | No crash, shows "Set experiment directory" prompt | No |

File: `tests/unit/test_toast.py` (new)

| Test function name | Fixture | What it asserts | Cache bypass? |
|---|---|---|---|
| `test_toast_shows_and_fades` | `qtbot` | Toast widget visible on show, hidden after timeout | No |
| `test_toast_queue_stacks` | `qtbot` | Two toasts show without overlap, second below first | No |

File: `tests/unit/test_tree_table.py` (new)

| Test function name | Fixture | What it asserts | Cache bypass? |
|---|---|---|---|
| `test_no_hardcoded_hex_in_widgets` | None | AST/grep scan of `widgets.py` finds zero hex color literals outside theme imports | No |
| `test_dsi_osi_columns_populated` | `qtbot` | Table model has DSI/OSI columns with numeric values | No |
| `test_dimming_search_nonmatch_style` | `qtbot` | Non-matching items rendered in `text_disabled` color | No |
| `test_geometric_icons_rendered` | `qtbot` | Group items have circle icon, not default Qt icon | No |

### Integration tests

File: `tests/integration/test_layout.py` (new)

| Test function name | Fixture | What it exercises |
|---|---|---|
| `test_minimum_size_enforced` | `make_main_window + qtbot` | `minimumSize()` returns ≥1100×650 |
| `test_sidebar_collapse_on_narrow` | `make_main_window + qtbot` | Resize to 1200px, sidebar collapses; resize to 1400px, restores |
| `test_context_panel_toggle` | `make_main_window + qtbot` | `Ctrl+\` toggles context panel visibility |

File: `tests/integration/test_plot_theming.py` (new)

| Test function name | Fixture | What it exercises |
|---|---|---|
| `test_theme_toggle_updates_all_plot_backgrounds` | `make_main_window + qtbot` | Toggle dark→light, assert all PlotWidget bg matches `LIGHT_COLORS['bg_panel']` |
| `test_sta_symmetric_scale` | `make_main_window + qtbot` | STA colorbar limits are `(-absmax, +absmax)` |

File: `tests/integration/test_grating_visual.py` (new)

| Test function name | Fixture | What it exercises |
|---|---|---|
| `test_polar_units_label` | `make_main_window + qtbot` | Radial axis label reads `spikes/s` |
| `test_error_bars_visible` | `make_main_window + qtbot` | SD whiskers rendered on polar plot |

### Visual regression tests

Tool: `pytest-mpl`
Baseline location: `tests/baseline_images/ux_ui_redesign/`

| Baseline | What it captures |
|---|---|
| `test_dark_standard_tab.png` | Standard plots tab, dark mode, full layout |
| `test_light_standard_tab.png` | Standard plots tab, light mode, full layout |
| `test_dark_grating_polar.png` | Grating polar with error bars, dark mode |
| `test_light_grating_polar.png` | Grating polar with error bars, light mode |

Generate with:
```bash
conda run -n encore python -m pytest --mpl-generate-path tests/baseline_images/ tests/integration/test_plot_theming.py -v
```

---

## Block 10 — Out of Scope

- Does **not** modify `src/analysis/data_manager.py` or any analysis code.
- Does **not** change `src/analysis/analysis_core.py` — plot functions produce
  the same data; this spec only changes how it is rendered.
- Does **not** change how `vision_integration.py` reads `.ei`, `.sta`, or
  `.neurons` files.
- Does **not** change `CrossRunMatcher` or `ReferenceBridge`.
- Does **not** modify `src/gui/workers/workers.py` worker logic (except for
  potential experiment browser `QRunnable`, which is a new, separate worker).
- Does **not** change the UMAP embedding or clustering algorithms.
- Does **not** modify `constants.py` feature weights.
- Does **not** address the stale `feature_cache.pkl` bug (Priority 3).
- Does **not** implement the Phase 3 launcher scripts (parked in `docs/PLAN.md`).
- Does **not** add a second experiment browser file dialog for npy files
  (that belongs to the cross-run stimulus bridge spec).

---

## Appendix A — Color Token Reference

Complete list of tokens after Phase 1 is applied. Tokens marked `[NEW]` are
added by this spec. Existing tokens are listed for cross-reference.

### Surface Tier

| Token | Dark | Light | Status |
|---|---|---|---|
| `bg_base` | `#111214` | `#F5F6F8` | Existing (light adjusted) |
| `bg_panel` | `#1A1B1F` | `#FFFFFF` | Existing (dark adjusted) |
| `bg_surface` | `#1E2025` | `#F8F9FA` | Existing |
| `bg_elevated` | `#282A30` | `#E9ECEF` | Existing |
| `bg_overlay` | `rgba(0,0,0,0.50)` | `rgba(0,0,0,0.25)` | [NEW] |
| `bg_tooltip` | `#282A30` | `#FFFFFF` | [NEW] |

### Content Tier

| Token | Dark | Light | Status |
|---|---|---|---|
| `text_primary` | `#F0F0F2` | `#111214` | Existing |
| `text_secondary` | `#9B9DA6` | `#495057` | Existing |
| `text_tertiary` | `#5A5C65` | `#6C757D` | Existing |
| `text_disabled` | `#3A3C44` | `#ADB5BD` | Existing |
| `text_tooltip` | `#F0F0F2` | `#111214` | [NEW] |

### Interactive Tier

| Token | Dark | Light | Status |
|---|---|---|---|
| `accent` | `#2E6DD4` | `#2E6DD4` | Existing |
| `accent_hover` | `#4A8BEF` | `#1A4A9E` | Existing |
| `accent_pressed` | `#1E4FA0` | `#3D7DE8` | [NEW] |
| `accent_muted` | `rgba(46,109,212,0.10)` | `rgba(46,109,212,0.08)` | [NEW] |
| `border_subtle` | `#2E3038` | `#DEE2E6` | Existing |
| `border_default` | `#3D3F48` | `#CED4DA` | Existing |
| `border_strong` | `#5A5C65` | `#ADB5BD` | Existing |
| `border_focus` | `#4A8BEF` | `#2E6DD4` | [NEW] |

### Spacing Scale

| Constant | Value | Use |
|---|---|---|
| `SP_1` | 4px | Minimum padding, icon margins |
| `SP_2` | 8px | Default panel padding (replaces `PANEL_PADDING`) |
| `SP_3` | 12px | Control group spacing |
| `SP_4` | 16px | Section spacing |
| `SP_5` | 24px | Major section breaks |

### Typography Scale

| Constant | Size | Weight | Use |
|---|---|---|---|
| `TYPE_HEADING` | 13px | 600 | Panel titles, group names |
| `TYPE_BODY` | 12px | 400 | Labels, controls, table cells |
| `TYPE_CAPTION` | 11px | 400 | Status text, axis labels, tooltips |
| `TYPE_MONO` | 11px | mono | Cluster IDs, numeric readouts |

---

## Appendix B — Shortcut Conflict Resolution

The current `KeyForwarder` binds `Ctrl+S` to "Mark Noisy". This conflicts with
the universal save shortcut. Resolution:

| Old binding | New binding | Reason |
|---|---|---|
| `Ctrl+S` → Mark Noisy | `Ctrl+Shift+N` → Mark Noisy | Free `Ctrl+S` for save |
| (none) | `Ctrl+S` → Save classification | Universal convention |
| (none) | `Ctrl+K` → Command palette | VS Code convention, non-conflicting |
| (none) | `Ctrl+G` → Group selected | Non-conflicting |
| (none) | `Ctrl+M` → Move to group | Non-conflicting |
| (none) | `F2` → Rename group | OS convention for rename |
| (none) | `Ctrl+Z` / `Ctrl+Shift+Z` → Undo/Redo | Universal convention |
| (none) | `Ctrl+\` → Toggle context panel | VS Code convention |

All existing `Ctrl+{D,C,E,W,X,A}` status shortcuts are retained unchanged.

---

## Appendix C — Lab Meeting Feedback Traceability

| Feedback item | Where addressed | Phase |
|---|---|---|
| GUI width not reducible for laptop screens | AC4, AC5 — responsive minimum width, sidebar collapse | 2 |
| STA rescales to input range, gray drifts | AC8 — symmetric fixed scale | 3 |
| Grating polar plot missing units and SD error bars | AC9 — axis label + whiskers | 3 |
| Per-direction rasters requested (Maria) | AC9 — raster subplots toggle | 3 |
| DSI/OSI removed from table | AC12 — restore as toggleable columns | 4 |
| Mosaic drag-to-select not in main (Anushka has it) | Phase 3 — merge from `anushka_dev` into context panel | 3 |
| Loading indicator buggy on reload | AC22 — progress bar reset | 7 |
| Export CSV in progress | Block 10 — CSV column list specified in PLAN.md §5.10 | 7 |
| Save figure only on mosaic | Phase 3 — extend to all tabs via `plot_export.py` | 3 |
| UMAP crashes on Maya's laptop | Out of scope — resolved by `pip install PyQt6 + git pull` | N/A |
