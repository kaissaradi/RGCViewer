# Handoff — current state

**Date:** 2026-08-12
**Branch:** `dev-testing`
**Uncommitted work:** no (as of the PLAN pass below). Do not push unless
the user asks.

This file is the pickup point. Read `README.md`, `CLAUDE.md`, and
`docs/AGENTS.md` first.

## What this branch is for

`dev-testing` is the active lab branch. The last completed work is load
speed plus the PLAN defects that did not need a user check. Do not start
the UX redesign unless the user asks.

## Standing decisions

1. Do not reopen the last run at application start. File dialogs still
   remember the last folder.
2. Broken EIs on older kilosort4 conversions are accepted. Do not remap
   519-channel EIs onto a 512-electrode plot.
3. Do not rewrite the application as HTML.
4. Do not commit unless the user asks.

## Done — load robustness

Older kilosort4 Vision folders can miss `.sta` / `.params`, or can hold
a 519-wide `.ei` next to a 512-row `.globals`.

| Change | Where |
|---|---|
| EI record stride comes from the `.ei` file, not from the `.globals` map | `src/analysis/visionloader.py` |
| Bogus cell IDs from a wrong stride are dropped | `src/analysis/visionloader.py` |
| Electrode map is replaced only when Litke coords match payload length | `src/analysis/visionloader.py` |
| Missing `.sta` / `.params` does not crash the load | `src/analysis/vision_integration.py` |
| STA quality is prepared off the GUI thread; `cluster_df` is written on the main thread | `data_manager.py`, `workers.py` |
| Start does not call `last_dataset()` | `main.py`, `main_window.py` |
| Rectangle/lasso tools are not created on a 0×0 figure | `live_selectors.py` |
| UMAP / RF / trace widgets retry selectors after layout | `umap_panel.py`, `rf_map_widget.py`, `trace_stack_widget.py` |
| `max_dup_r` stays float64. Do not write format strings into it | `data_manager._apply_ei_updates` |

## Expected messages (not defects)

Leave these. Do not "fix" them as crashes.

| Message | Meaning | Action |
|---|---|---|
| STA provenance dialog; "N of M cells in the .sta do not exist in this sort" | The `.sta` is from an older sort | Use the noise-run STA or Map Reference |
| `ei=519, positions=512` in the EI panel | Converter wrote a mismatched EI | Leave the plot blank |
| `standard_plot_cache.pkl` discarded (too large / unreadable) | Cache file is stale | Next load rebuilds it |
| `retinanalysis` import skipped | Optional package is absent | Ignore |
| `PeakPropertyWindow` warning | scipy peak finder | Ignore |

## Tests for this work

Run in environment `rgcviewer`:

```bash
conda activate rgcviewer
python -m pytest \
  tests/unit/test_live_selectors.py \
  tests/unit/test_vision_load_robustness.py \
  tests/unit/test_data_manager_cache.py::test_apply_ei_updates_keeps_max_dup_r_as_float \
  -v
```

Last run: 11 passed.

## How to verify in the application

1. Start with `python main.py`. Confirm an empty window. No run loads.
2. Open a modern kilosort25 run. Confirm UMAP finishes. Confirm no
   `fig_aspect` error.
3. Open an older kilosort4 run with a 519/512 EI. Confirm the run loads.
   Confirm the EI plot can stay empty.
4. Open 20251204. Confirm the STA dialog. Confirm spike analyses still work.

## Done — PLAN defects that did not need a user check (2026-08-12)

| Defect | Change |
|---|---|
| Stale `_computed` + `timecourse=None` | `DataManager._physics_entry_is_fresh` / `_sta_checked` |
| DS/OS slider ignored by grating panel | `select_dsos_for_display`; slider also calls `grating_panel.update_all` |
| `get_cell_physics` always loaded the STA cube | Params timecourse first |
| Population panels redrawn on every cell in a group | `population_group_plots_cached` |

Load-speed work from the same day is in `1c601c0` (cache bar + table unlock).

## Open defects (need the user, or not sure)

See `docs/PLAN.md`. Still open:

- Cells with no STA form a fake UMAP cluster. Product call.
- Default feature weights make the embedding almost STA-only. Scientific.
- The full pytest suite still has older failures. Do not skip them.

## Next

No next task is assigned. If the user wants more PLAN work, the remaining
open rows need a decision (UMAP / weights) or a user-visible check.
If the user reports a UMAP crash on first open, start at
`src/gui/panels/live_selectors.py` (`_axes_ready`).
If the user reports a start freeze, confirm `MainWindow` does not call
`last_dataset()`.
