# Grating DS/OS: actual conditions, best-run pick, population polar

## Metadata

| Field | Value |
|---|---|
| **Date created** | 2026-08-24 |
| **Last updated** | 2026-08-25 |
| **Branch** | `dev-testing` |
| **Spec status** | Done |
| **Supersedes** | Stale constants and load-time policy in `grating_panel_spec.md` |

---

## Problem

Raw grating files do not always use a 12-direction crossed bar-width × TF grid.

Example: `1212A/data018-/ksfiles` presents 12 orientations, but even 60° steps run at 100 µm / 2 Hz and odd 60° steps run at 400 µm / 4 Hz. Each `(barWidth, temporalFrequency)` therefore has 6 directions.

The old path treated a condition as DSOS only when it had 8 or more directions. Those 6-dir runs were tagged SF, so DSI/OSI never ran. The panel also assumed a full crossed grid instead of plotting the orientations that were actually presented.

Best-condition pick used max `|DSI|` and a 2 Hz peak-rate veto. A noisy high-DSI condition could beat a real DS/OS run at another bar width or TF. Sparse but significant cells were dropped.

Population DS/OS markers required STA RF ellipses. A grating-only run had no population view of preferred directions.

Grating compute waited until physics cache finished and used 1000 shuffles per condition. Opening a grating run felt like a second load.

Cluster clicks then crashed: `QWidget::repaint: Recursive repaint detected` and `AttributeError` on `canvas._pop_plot_state.get` when `_pop_plot_state` was `None`.

---

## Current behavior

### Grouping

`group_grating_conditions()` partitions trials by the `(barWidth, temporalFrequency)` pairs in `trial_parameters`.

| Unique orientations at that pair | Tag | What the GUI shows |
|---|---|---|
| ≥ 4 (`MIN_DIRECTIONS_FOR_DSOS`) | `dsos` | Polar DSI/OSI for that pair |
| 1–3 | `sf` | Bar-width tuning |

Do not merge complementary halves into one 12-dir curve. Plot each pair that ran.

### Best condition

`select_best_dsos_condition()` is the single selector for GratingPanel, the RF overlay, and the preferred-orientation polar.

Per `(bw, tf)` that is tagged `dsos`:

1. Gate on shuffle p-value `< 0.05`. A missing p-value does not veto (analyzed files).
2. Classify that pair: DS if `|DSI|` exceeds the slider; else OS if `|OSI|` exceeds the slider. DS-first applies only inside the same pair.
3. Rank classified pairs by peak `mean_response`. The strongest response wins.

`MIN_RESPONSE_HZ` is 0. Amplitude ranks conditions. It does not drop sparse cells.

The population DS/OS slider (default 0.3, range 0.10–0.90) is the `|DSI|` / `|OSI|` cutoff after the p-value gate.

### Compute cost

| Constant | Value | Why |
|---|---|---|
| `N_SHUFFLES` | 200 | Resolves p to 0.005. Enough for a 0.05 gate. |
| `SHUFFLE_INDEX_FLOOR` | 0.10 | Slider floor. Below that, skip the permutation test and store p = 1.0. |
| `MIN_DIRECTIONS_FOR_DSOS` | 4 | Vector-sum needs a circular set. Four is every 90°. |

`GratingBatchWorker` starts with physics warm-up, not after it. Raw grating trials are already in RAM. The worker uses a thread pool (up to 8). Disk saves happen every 100 cells, not every 25.

### Cache

Results live in `dm.grating_computed_cache`, persisted as `grating_computed_cache.pkl`.

`grating_entry_needs_recompute()` drops entries whose DSOS/SF tags do not match the direction counts that were actually run. Load rewrites or unlinks the pkl so a 6-dir run tagged SF is not reused.

### Population view

If STA RF ellipses exist, DS arrows and OS bars sit on those ellipses. A polar inset and a title `N DS, M OS` show preferred directions.

If there are no STA RFs, the mosaic is a preferred-direction polar: DS arrows and OS bars from the origin. No white-noise STA is required.

### Paint safety

`pop_mosaic_canvas._pop_plot_state` is a dict or absent. Never set it to `None`. A None sentinel made `hasattr` true and `.get` throw on the next click.

`pop_canvas_can_hot_swap(canvas)` is the only hot-swap check. Selection hot-swap updates the highlight only. Do not rebuild DS/OS markers or call `fig.add_axes` / `fig.clear` from Tier 1.

When the grating batch finishes, delete `_pop_plot_state` and call `_draw_population_panel_initial` through `QTimer.singleShot(0, ...)`. Do not nest a matplotlib draw inside a Qt paint or a selection handler.

---

## Files

| File | Change |
|---|---|
| `src/analysis/grating_calc.py` | Group by actual `(bw, tf)`. Constants above. Best-run ranking. |
| `src/analysis/data_manager.py` | Drop stale DSOS/SF cache entries. Treat them as missing. |
| `src/gui/workers/workers.py` | Thread-pool batch. Save less often. Recompute stale tags. |
| `src/gui/callbacks.py` | Start grating with physics. Defer population redraw. |
| `src/gui/main_window.py` | `pop_canvas_can_hot_swap`. Slider tooltip. |
| `src/gui/panels/grating_panel.py` | Condition labels include direction count. |
| `src/gui/panels/population_panel.py` | Polar-only DS/OS. Inset on RF plot. Highlight-only hot-swap. |
| `tests/unit/test_grating_calc.py` | Grouping, stale cache, 4-dir / 6-dir / 12-dir layouts. |
| `tests/unit/test_dsos_threshold.py` | Strongest response wins. Sparse cells. Missing p. |
| `tests/unit/test_dsos_population.py` | Polar without STA. Hot-swap guard. |

---

## Tests

```bash
python -m pytest tests/unit/test_grating_calc.py tests/unit/test_dsos_threshold.py tests/unit/test_dsos_population.py tests/unit/test_population_rf_mosaic.py -v
```

---

## Out of scope

- Merging complementary 6-dir halves into one 12-dir curve.
- Changing UMAP grating features beyond the existing pooled curve.
- Contrast-panel work.
