# SPEC.md — Grating DSOS / SF Tuning Panel

## Block 0 — Metadata

| Field | Value |
|---|---|
| **Date created** | 2026-07-01 |
| **Last updated** | 2026-08-25 |
| **Branch** | `feat/grating-panel` |
| **Author** | Kais |
| **Spec status** | Done. Current grouping, ranking, batch, and population polar: `grating_dsos_flexible_conditions.md`. |

The numbers below are the original design. Do not copy them into new code.

| Original | Current |
|---|---|
| `min_directions_for_dsos=8` | `MIN_DIRECTIONS_FOR_DSOS = 4` |
| `n_shuffles=1000` | `N_SHUFFLES = 200` |
| On-demand only; do not batch at load | `GratingBatchWorker` starts with physics warm-up |
| Assume a typical 12-dir × 2 TF × 2 bar-width grid | Plot the `(bw, tf, orientation)` combinations that ran |

---

## Block 1 — Problem Statement

**Symptom:** Grating-run output (direction/orientation selectivity, and
sometimes a coarse bar-width/SF sweep mixed into the same file) has no
visualization in RGCViewer. Protocol is variable — typically 12 directions x
2 temporal frequencies x 2 bar widths, but this is not fixed and sometimes
includes extra coarse-direction bar-width conditions in the same recording.

**Root cause:** No panel or DataManager hook exists for grating analysis
output, and unlike chirp, there is no single fixed-shape array — condition
count and type varies run to run.

**User story:** "As a scientist, I want to see a cell's direction tuning
(polar plot, DSI/OSI, preferred direction) for whichever grating conditions
were actually run, and its bar-width tuning if that was also collected,
without caring what combination of conditions this particular experiment
used."

---

## Block 2 — Vision ID Contract

| Question | Answer |
|---|---|
| ID space | Kilosort space — same convention as the chirp panel (`cluster_id` used directly, no `+1` translation), per lab convention confirmed for this pipeline. |
| Translation used | None — direct `cluster_id` lookup, same as chirp. |

---

## Block 3 — File Detection + In-GUI Calculation

There is no reliable filename convention — confirmed directly from a sample
raw file (`kilosort2_5_GratingDSOS.npy`) that a grating-named `.npy` can be
either raw (pre-analysis) or analyzed, and analyzed files may or may not
carry a `_combined` suffix depending on which script produced them.

**Since only the raw npy is guaranteed to exist, RGCViewer needs its own
DSI/OSI calculation path** — it can no longer assume an offline analysis
script has already run. This is a real change from chirp: chirp is always
precomputed offline; grating sometimes will be, sometimes won't.

**Detection is schema-based, not name-based:**

```python
def _classify_grating_npy(mdic: dict) -> str:
    """Returns 'analyzed', 'raw', or 'unknown'."""
    if _looks_analyzed(mdic):
        return 'analyzed'
    if 'trial_parameters' in mdic and 'spike_times_by_trial' in mdic:
        return 'raw'
    return 'unknown'

def _looks_analyzed(mdic: dict) -> bool:
    # Analyzed files are per-cluster dicts keyed by cluster_id (int),
    # each value a dict keyed by (barWidth, temporalFrequency) tuples
    # containing 'condition_type'.
    sample_vals = [v for k, v in mdic.items() if isinstance(k, (int, np.integer))]
    if not sample_vals:
        return False
    first = sample_vals[0]
    if not isinstance(first, dict):
        return False
    return any(
        isinstance(k, tuple) and isinstance(v, dict) and 'condition_type' in v
        for k, v in first.items()
    )
```

**Load procedure** (`DataManager.load_grating_data()`):
1. Glob `kilosort_dir` for `*.npy` matching a broad grating pattern
   (`*Grating*.npy`, `*DSOS*.npy`) — cast wide since naming isn't reliable.
2. Load and classify each candidate.
3. Among `'analyzed'` candidates, prefer one with `_combined` in the
   filename; otherwise take the first.
4. **If only `'raw'` candidates found → don't give up.** Store the raw
   dict (`grating_raw_data`) and set `grating_status = 'raw_only'`. DSI/OSI
   get computed **on demand, per cluster, in the GUI** the first time that
   cluster is viewed — not batch-precomputed for all 890 clusters at load
   time, since the shuffle-test is the expensive part and most clusters
   will never be viewed in a given session.
5. If nothing matches at all → `grating_available = False`,
   `grating_status = 'missing'`.

**Calculation module — `grating_calc.py` (new, shared/reusable, not
panel-specific):** a straight port of the response-extraction + vector-sum +
permutation-test logic from `combined_grating_analysis.py` (`f1_amplitude`,
`vector_sum_index`, `shuffle_pvalue`), scoped down to run for ONE cluster at
a time instead of the whole file. Same math, not reimplemented from scratch,
so results match what the offline script would have produced.

```python
def compute_grating_response(cluster_id, spike_times_by_trial, trial_parameters,
                              n_shuffles=1000, min_directions_for_dsos=8,
                              response_metric='f1'):
    """
    Returns the same per-condition dict shape as combined_grating_analysis.py's
    results[cluster_id] — one entry per (barWidth, temporalFrequency) condition,
    each tagged condition_type='dsos'|'sf', so the panel's rendering code
    doesn't care whether the data came from disk or was computed live.
    """
```

**Threading — this is the part that differs from chirp.** Computing DSI/OSI
with a 1000-shuffle permutation test for one cluster is not free (roughly
comparable in cost to a single `StandardPlotsWorker` cluster computation).
This must run on a **background worker**, not inline in `update_all()`:

| Operation | Thread | Worker | Tier |
|---|---|---|---|
| Load raw/analyzed npy at dataset load | Background | `KilosortLoadWorker` (extended) | one-time |
| Compute DSI/OSI for one cluster from raw data | Background | **new `GratingComputeWorker`**, mirrors `StandardPlotsWorker`'s single-cluster-job pattern | Triggered from Tier 2, result delivered via `finished = Signal(int, dict)` (cluster_id, result) |
| Render already-computed/cached result | Main thread | direct call | Tier 2 |

Computed results get cached per-cluster in `dm.grating_computed_cache: dict[int, dict]`
(same shape as `grating_data[cluster_id]` would be for a pre-analyzed file) so
revisiting a cluster in the same session doesn't recompute. This is the same
shape of cache-with-lock pattern as `standard_plot_cache` — needs its own
`_grating_cache_lock`, not the chirp precedent of "no lock needed," because
now there IS a background writer running per-cluster during the session.

---

## Block 4 — DataManager Attributes

| Attribute | Type | Notes |
|---|---|---|
| `grating_data` | `dict[int, dict]` or `None` | Pre-analyzed dict, if an analyzed file was found on disk |
| `grating_raw_data` | `dict` or `None` | Raw `spike_times_by_trial` + `trial_parameters`, if only a raw file was found |
| `grating_available` | `bool` | True if EITHER analyzed data exists OR raw data exists (raw enables on-demand compute) |
| `grating_status` | `str` | `'ok'` (analyzed file loaded) \| `'raw_only'` (will compute on demand) \| `'missing'` |
| `grating_conditions` | `list[tuple[float, float]]` or `None` | All `(barWidth, temporalFrequency)` pairs, sorted, for the condition dropdown — read from whichever of `grating_data`/`grating_raw_data` is present |
| `grating_computed_cache` | `dict[int, dict]` | Per-cluster on-demand compute results, keyed by cluster_id |
| `_grating_cache_lock` | `threading.Lock` | Guards `grating_computed_cache` writes — same pattern as `_standard_plot_lock` |

**New methods:**

```python
def load_grating_data(self, grating_path=None) -> tuple[bool, str]:
    """Never raises. Populates grating_data XOR grating_raw_data depending
    on what's on disk, sets grating_status accordingly."""

def get_grating_data_for_cluster(self, cluster_id) -> dict | None:
    """Tier 1-safe dict lookup ONLY. Checks grating_data first, then
    grating_computed_cache. Returns None (never computes) if the cluster
    isn't in either — the caller (panel) is responsible for kicking off
    GratingComputeWorker when this returns None but grating_status == 'raw_only'.
    """

def try_get_grating_data_for_cluster(self, cluster_id) -> dict | None:
    """Same as above — name kept consistent with try_get_standard_plot_data's
    non-blocking-read convention."""
```

Return shape (same whether pre-analyzed or on-demand-computed, per condition
key) — unchanged from the original draft:

```python
{
    (100.0, 2.0): {
        'condition_type': 'dsos',
        'directions_deg': np.array([...]),
        'mean_response': np.array([...]),
        'sem_response': np.array([...]),
        'DSI': 0.34, 'preferred_direction_deg': 142.0, 'DSI_pvalue': 0.002,
        'OSI': 0.12, 'preferred_orientation_deg': 62.0, 'OSI_pvalue': 0.41,
    },
    (50.0, 2.0): {
        'condition_type': 'sf',
        'directions_deg': np.array([...]),
        'mean_response': np.array([...]),
        'sem_response': np.array([...]),
        'DSI': nan, 'OSI': nan, ...,
        'bw_tuning_point': 18.4,
        'bw_tuning_point_sem': 2.1,
    },
    'sf_bar_widths': np.array([50, 100, 200, 400, 800]),   # optional
    'sf_tuning_curve': np.array([12.1, 18.4, 22.0, 15.3, 9.8]),  # optional
}
```

---

## Block 5 — Panel Design

**What we're showing, and why each piece is there:**

1. **Condition dropdown** — populated once at dataset load from
   `dm.grating_conditions`, labeled e.g. `"bw=100  tf=2.0 Hz  (DSOS)"` or
   `"bw=50  tf=2.0 Hz  (SF probe)"`.

2. **Direction tuning — polar plot, primary view.** `mean_response` vs
   `directions_deg` for the selected `dsos` condition, `sem_response` as
   error bars, a preferred-direction arrow/line overlaid. DSI/OSI/p-values
   shown as a text block beside it. This is the main scientific payload —
   answers "does this cell prefer a direction, and how strongly."

3. **Sanity-check strip below the polar plot — firing rate trace, not
   raster.** For the selected condition, the single direction closest to
   `preferred_direction_deg` gets a small mean-PSTH-style firing-rate trace
   across the stim window (same rendering approach as `ChirpPanel`'s PSTH
   curve — reuse that pattern, not a new plot type). This is a sanity check
   the polar plot alone can't give you: whether the "preferred direction"
   the vector-sum math picked out actually corresponds to a real, clean
   response, or whether it's DSI noise dressed up as a strong preference on
   a nearly-silent cell. A raster would show the same thing with more visual
   noise for a sanity check that just needs "is there a real bump here" —
   firing rate is the more legible choice for a quick look, and it's the
   representation we're already rendering elsewhere in the app (chirp,
   standard plots), so it stays visually consistent rather than introducing
   a raster-plotting code path used nowhere else in the codebase.

4. **Bar-width tuning curve** (only rendered if `sf_tuning_curve` key
   present) — `sf_tuning_curve` vs `sf_bar_widths`, always visible
   regardless of dropdown selection, since it's a single aggregate view
   across all `sf` conditions, not tied to one.

5. **Placeholder page** (same `QStackedLayout` pattern as `ChirpPanel`) for:
   `grating_status == 'missing'` → "No grating data"; cluster not yet
   computed and worker in flight → "Computing DSI/OSI..." (distinct from
   chirp, which never has a loading state at the panel level); cluster
   absent entirely → "No grating response for cluster N".

**Threading for panel updates:** `update_all(cluster_id)` stays a Tier 2
direct call for the READ path (dict lookup). If the lookup misses and
`grating_status == 'raw_only'`, `update_all()` shows the "Computing..."
placeholder and kicks off `GratingComputeWorker` for that cluster — this one
new case is the one place this panel spawns a worker, unlike chirp.

---

## Block 6 — Affected Files

| File | Change |
|---|---|
| `data_manager.py` | Add `grating_data`/`grating_raw_data`/`grating_available`/`grating_status`/`grating_conditions`/`grating_computed_cache`/`_grating_cache_lock`; add `load_grating_data()`, `get_grating_data_for_cluster()`, `try_get_grating_data_for_cluster()`, `_classify_grating_npy()` |
| `grating_calc.py` (new) | Ported single-cluster DSI/OSI/permutation-test logic from `combined_grating_analysis.py` |
| `workers.py` | `KilosortLoadWorker.run()` — add `load_grating_data()` call; add new `GratingComputeWorker` class (mirrors `StandardPlotsWorker`'s single-job pattern) |
| `grating_panel.py` (new) | `GratingPanel` class — dropdown, polar plot, firing-rate sanity strip, bar-width tuning curve, placeholder states incl. "Computing..." |
| `main_window.py` | Register tab, wire into theme restyle / `on_tab_changed` / `_draw_plots` (5 touch points), plus connect `GratingComputeWorker.finished` to a cache-write + repaint slot |
| `tests/unit/test_grating_data.py` (new) | Schema classification, cache lookup, and `grating_calc.py` numerical tests (compare against known `combined_grating_analysis.py` output for the same raw input, to confirm the ported math matches) |

---

## Block 7 — Open Items / Assumptions To Confirm Before Merge

- **Cluster ID space** is assumed Kilosort, matching chirp's confirmed
  convention. Worth a quick sanity check the first time a real analyzed
  grating file is loaded, since this wasn't independently re-verified for
  the grating pipeline specifically.
- **`_classify_grating_npy` schema detection** is built from
  `combined_grating_analysis.py`'s exact output shape. If you write your own
  analysis script with a differently-shaped analyzed dict, the classifier
  needs updating — flag this explicitly rather than silently misclassifying
  a valid analyzed file as `'unknown'`.
- Polar plot is hand-built in pyqtgraph (no native polar plot widget) —
  reasonable given the codebase's existing pyqtgraph usage throughout, but
  worth 10 minutes of visual sanity-check once real data flows through it.

---

## Block 8 — Out of Scope

- Does not run `combined_grating_analysis.py`, `dsos_analysis.py`, or
  `sf_tuning_analysis.py` from inside RGCViewer — offline-only, same as chirp.
- Does not support analyzed files with a schema other than
  `combined_grating_analysis.py`'s (e.g. a standalone `dsos_analysis.py`
  output with a flat-CSV-style shape) — flagged in Block 7, not handled in v1.
- Does not add grating features to the UMAP/GMM classification pipeline.
