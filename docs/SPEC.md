# SPEC.md — Encore Spec Template

> Copy this file to `docs/specs/your_feature_name.md` before filling it out.
> Every section is required. If a section does not apply, write N/A and state why.
> An incomplete spec will be sent back. A spec is not ready for implementation
> until every block below has been filled out and reviewed.
>
> Reading order before writing a spec:
> `docs/AGENTS.md` → `docs/PLAN.md` (fragile zones) → this template.

---

## Block 0 — Metadata

| Field | Value |
|---|---|
| **Date created** | YYYY-MM-DD |
| **Last updated** | YYYY-MM-DD |
| **Commit hash when spec was written** | `git rev-parse --short HEAD` — paste result here |
| **Branch** | `feat/` or `fix/` — name it before starting |
| **Author** | Name or agent ID |
| **Spec status** | Draft / Ready for Dev / In Progress / Done |

---

## Block 1 — Problem Statement

**Symptom** (what the user actually sees or experiences):
> One or two sentences. Describe the observable problem, not the internal cause.
> Example: "The ACG plot is flat for cells that only fire in the second half of a recording."

**Root cause** (what in the code produces it):
> One sentence. Name the function and the specific mechanism.
> Example: "`_compute_standard_plots()` builds a dense boolean array of length `max(spike_times)`
> which hits a memory limit around 2 minutes of recording and is silently truncated."

**User story:**
> "As a [scientist / developer], I want [action] so that [outcome]."

---

## Block 2 — Vision ID Contract

**Required if this spec touches any of:**
`vision_integration.py`, `get_cell_physics()`, `vision_stas`, `vision_eis`,
`vision_params`, any panel that reads STA or EI data.

If none of the above apply, write: `N/A — this spec does not access Vision data.`

| Question | Answer |
|---|---|
| Does this spec access Vision file data? | Yes / No |
| ID space this spec operates in | Kilosort 0-indexed / Vision 1-indexed / Both |
| Reads `is_vision_only` flag? | Yes / No |
| Translation used | `vid = cluster_id + 1` (hybrid) / `vid = cluster_id` (vision-only) / calls `get_cell_physics()` |
| Safe access pattern used | Paste the exact lines of code here |

**Reminder:** Never access `vision_stas[cluster_id]` directly. Always translate first.
The canonical translation lives in `get_cell_physics()` — call it, don't copy it.

---

## Block 3 — Affected Files

List every file this spec touches. Be exact. Vague entries will be rejected.

| File path | Function(s) added or modified | Change type | Touches DataManager? |
|---|---|---|---|
| `src/analysis/data_manager.py` | `_compute_standard_plots()` | Modify | Yes |
| `src/gui/workers/workers.py` | `StandardPlotsWorker.run()` | Modify | No |
| `tests/unit/test_feature.py` | `test_name_one`, `test_name_two` | Add | No |

> **If any row in "Touches DataManager?" is Yes:**
> This spec is subject to the DataManager bottleneck rule.
> You must `git fetch && git rebase origin/main` before every push.
> Tag your PR clearly so other agents know to expect rebase conflicts.

> **Do not add files to this list mid-implementation without updating the spec.**
> Scope creep starts here.

---

## Block 4 — Qt Threading Contract

Every piece of work this spec adds must be classified. Fill out one row per new
operation. If the spec adds no background work, write: `N/A — no new threading.`

| Operation | Runs on thread | Worker class | Signal name + signature | Receiving slot | Tier 1 or Tier 2? |
|---|---|---|---|---|---|
| Compute ACG | Background | `StandardPlotsWorker` | `result_ready = Signal(int, dict)` | `on_standard_plots_ready(cluster_id, data)` | Tier 2 |
| Update ACG plot | Main thread | N/A — direct call | N/A | `_draw_standard_plots()` | Tier 2 |

**Tier reminder:**

- Tier 1 = runs synchronously inside `update_cluster_views()` on the main thread.
  Only allowed if touching already-cached in-RAM data. Forbidden: disk I/O, locks, worker spawns.
- Tier 2 = runs inside `_process_selection()`, debounced 25ms. Workers and I/O are fine here.

If you cannot classify an operation as Tier 1 or Tier 2, you are not ready to implement.

**Stale result guard:** Every new result slot must include this guard as its first line:

```python
if cluster_id != self._get_selected_cluster_id():
    return
```

---

## Block 5 — Cache Contract

| Question | Answer |
|---|---|
| Which cache(s) does this spec read? | e.g. `standard_plot_cache` keyed by `cluster_id` |
| Which cache(s) does this spec write? | e.g. `feature_cache` keyed by `cluster_id` |
| What triggers cache invalidation? | e.g. "when kilosort directory changes" / "never" |
| Is data persisted to disk? | Yes — `standard_plot_cache.pkl` / No |
| Which lock must be held? | e.g. `_standard_plot_lock` / None (main thread only) |
| Must tests bypass the cache? | **Yes / No** |

> **If "Must tests bypass the cache?" is Yes:**
> Every test that verifies computation logic in this spec must use `tmp_path` or
> `cache_cleared_data_manager`. Using a real data path risks loading a warm `.pkl`
> and silently skipping all math — the test will pass while proving nothing.
> This is the False Positive Trap. See AGENTS.md §1 Law 3.

---

## Block 6 — DataManager Attributes Used

List every `DataManager` attribute this spec reads or writes.
Check the nullability table in AGENTS.md §8 before writing access code.

| Attribute | Type | Can be `None`? | This spec reads / writes | Safe access pattern |
|---|---|---|---|---|
| `vision_stas` | `LazySTADict` | Yes | Reads | `if self.vision_stas and vid in self.vision_stas:` |
| `standard_plot_cache` | `dict` | No (may be empty) | Reads + Writes | Acquire `_standard_plot_lock` first |
| `spike_times` | `np.ndarray` memmap | No (after load) | Reads | Direct — treat as immutable |

---

## Block 7 — Acceptance Criteria

Each AC is strictly Pass / Fail. No partial credit.
Functional ACs first. Edge cases second. Visual ACs last.

### AC1 — [Name the behavior]

- **Setup:** Describe the exact state to put the system in.
- **Action:** What is triggered.
- **Expected:** The precise observable outcome. Numbers where possible.
- **Test type:** Unit / Integration / Manual

### AC2 — [Edge case]

- **Setup:** e.g. "`.sta` file is missing from the Vision directory"
- **Action:** User loads the dataset and selects any cluster.
- **Expected:** `STAPanel` shows placeholder text "No STA data". No crash. No error dialog.
- **Test type:** Unit

### AC3 — [Visual — screenshot required]

> Visual ACs require a screenshot. Prose descriptions of visual state are not sufficient.

- **State to reproduce:**
  1. Step one — exact action.
  2. Step two — exact action.
  3. Step three — navigate to [Panel Name] tab.
- **Expected appearance:** Describe layout precisely. Name widgets. Note absence of artifacts.
- **Must verify:** Dark mode AND light mode.
- **Screenshot filenames:**
  - `tests/screenshots/ac3_[feature_name]_dark.png`
  - `tests/screenshots/ac3_[feature_name]_light.png`
- **Verified by:** `[ ]` Author `[ ]` Reviewer

---

## Block 8 — Regression Guard

Look up every file in Block 3 in PLAN.md's "Completed Fix Registry."
For each completed fix that touched the same files, list it here.
If this spec cannot regress any prior fix, write: `N/A — no overlap with completed fixes.`

| Prior fix | Files overlap | Regression test to run | When to run it |
|---|---|---|---|
| ACG full-recording fix | `data_manager.py::_compute_standard_plots()` | `test_acg_includes_late_spike_trains` | Before opening PR |
| Physics cache double-load | `data_manager.py::__init__` | `test_standard_plot_cache_computes_same_cluster_once` | Before opening PR |

> Run every test in this table before opening your PR.
> A regression in a previously-passing test is a blocking issue.

---

## Block 9 — Test Plan

Do not write "add unit tests." Name the exact test functions, their inputs,
their fixtures, and their assertions. Write these names before writing any
implementation code — that is what TDD means.

### Unit tests

File: `tests/unit/test_[feature_name].py`

| Test function name | Fixture | What it asserts | Cache bypass needed? |
|---|---|---|---|
| `test_[name]_[condition]_[expected]` | `tmp_path` | `data['acg_norm'][110] > 0.5` | Yes |
| `test_[name]_too_few_spikes_returns_none` | `tmp_path` | `data['acg_norm'] is None` | Yes |

### Integration tests

File: `tests/integration/test_[feature_name].py`

| Test function name | Fixture | What it exercises |
|---|---|---|
| `test_[name]_signal_emitted_on_completion` | `make_main_window + qtbot` | Qt signal fires, slot receives correct cluster_id |
| `test_[name]_stale_result_discarded` | `make_main_window + qtbot` | Result for old cluster_id does not update panel |

### Visual regression tests (if applicable)

Tool: `pytest-mpl`
Baseline location: `tests/baseline_images/[feature_name]/`
Generate with: `conda run -n encore python -m pytest --mpl-generate-path tests/baseline_images/ tests/`

---

## Block 10 — Out of Scope

Be specific. Name panels and functions. Vague scope statements are not useful.

- Does **not** modify `[specific panel or function]`.
- Does **not** change the ACG bin width or `acg_time_lags` output shape.
- Does **not** touch `UMAPPanel` or `feature_cache`.

---
---

# Worked Example — `docs/specs/autocorrelation_fix.md`

> This is a fully completed spec using the template above.
> Use it as the reference for what "done" looks like.

---

## Block 0 — Metadata

| Field | Value |
|---|---|
| **Date created** | 2026-04-10 |
| **Last updated** | 2026-04-14 |
| **Commit hash when spec was written** | `a3f92c1` |
| **Branch** | `fix/acg-full-recording` |
| **Author** | Kais |
| **Spec status** | Done |

---

## Block 1 — Problem Statement

**Symptom:** The ACG plot appears flat or near-zero for cells that fire predominantly
in the second half of a long recording. Late bursts and late drift are invisible.

**Root cause:** `_compute_standard_plots()` built a dense boolean array of length
`max(spike_times)` to compute autocorrelation. For recordings longer than ~2 minutes
at 30kHz, this array exceeds a soft memory threshold and is silently truncated,
discarding all spikes after ~120 seconds.

**User story:** "As a scientist reviewing long recordings, I want the ACG to reflect
the entire session so that I can detect late drift and burst changes without missing them."

---

## Block 2 — Vision ID Contract

`N/A — this spec does not access Vision data.`
`_compute_standard_plots()` uses `spike_times` and `spike_clusters` only (Kilosort arrays).

---

## Block 3 — Affected Files

| File path | Function(s) modified | Change type | Touches DataManager? |
|---|---|---|---|
| `src/analysis/data_manager.py` | `_compute_standard_plots()` | Modify | Yes |
| `tests/unit/test_autocorrelation.py` | `test_acg_includes_late_spike_trains`, `test_acg_not_computed_for_too_few_spikes` | Add | No |

> DataManager is touched. Rebase from main before every push.

---

## Block 4 — Qt Threading Contract

`_compute_standard_plots()` is a pure computation function. It is called from inside
`get_standard_plot_data()` which is invoked by `StandardPlotsWorker` on a background thread.
No new signals or slots are added by this spec.

| Operation | Runs on thread | Worker class | Signal | Slot | Tier |
|---|---|---|---|---|---|
| ACG computation | Background | `StandardPlotsWorker` (existing) | `result_ready = Signal(int, dict)` (existing) | `on_standard_plots_ready()` (existing) | Tier 2 |

Stale result guard already present in `on_standard_plots_ready()`. No changes needed.

---

## Block 5 — Cache Contract

| Question | Answer |
|---|---|
| Cache read | `standard_plot_cache` keyed by `cluster_id` |
| Cache written | `standard_plot_cache` keyed by `cluster_id` |
| Invalidation trigger | Never — persists for the session |
| Persisted to disk | Yes — `standard_plot_cache.pkl` |
| Lock required | `_standard_plot_lock` |
| Must tests bypass cache? | **Yes** |

Tests use `tmp_path` to guarantee an empty directory with no `.pkl` file.

---

## Block 6 — DataManager Attributes Used

| Attribute | Type | Can be `None`? | Used | Safe access |
|---|---|---|---|---|
| `spike_times` | `np.ndarray` memmap | No (after load) | Read | Direct — treat as immutable |
| `spike_clusters` | `np.ndarray` memmap | No (after load) | Read | Direct — treat as immutable |
| `standard_plot_cache` | `dict` | No (may be empty) | Read + Write | Acquire `_standard_plot_lock` |
| `sampling_rate` | `float` | No | Read | Direct |

---

## Block 7 — Acceptance Criteria

### AC1 — Spikes after 120 seconds contribute to the ACG

- **Setup:** Synthetic spike train with all spikes between t=130s and t=131s at 10ms ISI.
- **Action:** Call `dm._compute_standard_plots(cluster_id)`.
- **Expected:** `data['acg_norm']` is not `None`. `len(data['acg_norm']) == 201`.
  `data['acg_norm'][110] > 0.5` (the +10ms bin is populated).
- **Test type:** Unit

### AC2 — Mixed early and late spikes both appear in ACG

- **Setup:** Spike train with spikes at t=10s (sparse) and t=300s (dense, 10ms ISI).
- **Action:** Call `dm._compute_standard_plots(cluster_id)`.
- **Expected:** `data['acg_norm'][110] > 0.5`.
- **Test type:** Unit (same parametrize block as AC1)

### AC3 — Too-few-spike clusters return `None`, not bogus data

- **Setup:** Spike train with fewer than 50 spikes total.
- **Action:** Call `dm._compute_standard_plots(cluster_id)`.
- **Expected:** `data['acg_time_lags'] is None` and `data['acg_norm'] is None`.
- **Test type:** Unit

### AC4 — Visual: ACG plot shows non-flat bars for late-burst cell (Visual)

- **State to reproduce:**
  1. Load the real dataset at `/mnt/lab/Array-data/sorted/20260506A/chunk10/kilosort2.5`
  2. Select a cluster known to have a late burst (confirm via raw traces first)
  3. Navigate to the Standard Plots tab
- **Expected appearance:** ACG bar chart shows a visible peak near the 0ms center.
  Bars are not uniformly flat. Y-axis is not stuck at 0.
- **Must verify:** Dark mode AND light mode.
- **Screenshot filenames:**
  - `tests/screenshots/ac4_acg_full_recording_dark.png`
  - `tests/screenshots/ac4_acg_full_recording_light.png`
- **Verified by:** `[x]` Author `[ ]` Reviewer

---

## Block 8 — Regression Guard

| Prior fix | Files overlap | Regression test | When |
|---|---|---|---|
| Physics cache double-load | `data_manager.py::__init__` | `test_standard_plot_cache_computes_same_cluster_once` | Before PR |

---

## Block 9 — Test Plan

### Unit tests

File: `tests/unit/test_autocorrelation.py`

| Test function | Fixture | Assertion | Cache bypass? |
|---|---|---|---|
| `test_acg_includes_late_spike_trains` | `tmp_path` (via `mock_dm`) | `acg_norm[110] > 0.5` for late-only and mixed spike trains (parametrized) | Yes |
| `test_acg_not_computed_for_too_few_spikes` | `tmp_path` (via `mock_dm`) | `acg_norm is None`, `acg_time_lags is None` | Yes |

### Integration tests

None required — the computation path through `StandardPlotsWorker` is already
covered by `test_standard_plot_cache_computes_same_cluster_once`.

### Visual regression

Not added for this spec. ACG rendering style is out of scope.

---

## Block 10 — Out of Scope

- Does **not** change the ACG bin width (fixed at 1ms) or the lag range (fixed at ±100ms).
- Does **not** modify the ACG rendering code in `StandardPlotsPanel`.
- Does **not** change `feature_cache` or anything in `get_cell_physics()`.
- Does **not** modify the ISI or firing rate calculations in `_compute_standard_plots()`.
