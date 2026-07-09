# SPEC.md — Chirp PSTH Panel

> Copied from `SPEC.md` template. Block 3 (main_window.py hook point) and
> Block 8 (Regression Guard) are marked INCOMPLETE — `main_window.py` and
> `PLAN.md` were not available when this draft was written. Do not move to
> "Ready for Dev" until both are filled in.

---

## Block 0 — Metadata

| Field | Value |
|---|---|
| **Date created** | 2026-07-01 |
| **Last updated** | 2026-07-01 |
| **Commit hash when spec was written** | *(paste `git rev-parse --short HEAD`)* |
| **Branch** | `feat/chirp-panel` |
| **Author** | Kais |
| **Spec status** | Draft |

---

## Block 1 — Problem Statement

**Symptom:** `chirp_analysis.py` produces a per-chunk `.npy`/`.mat` file containing
mean PSTHs, per-trial binned spikes, and a quality index for every recorded cell,
but RGCViewer has no way to view this data. A scientist scrolling through clusters
has no visual feedback on chirp responsiveness and must load the `.npy` manually
in a separate script to sanity-check a cell.

**Root cause:** No panel or `DataManager` hook exists for chirp output. The file
sits on disk next to `standard_plot_cache.pkl` in the same `kilosort_dir`, unused.

**User story:** "As a scientist reviewing clusters, I want to see each cell's chirp
PSTH (with phase boundaries marked) next to its ACG/ISI/STA so I can judge chirp
responsiveness without leaving the GUI."

---

## Block 2 — Vision ID Contract

| Question | Answer |
|---|---|
| Does this spec access Vision data? | **Yes — indirectly.** |
| ID space this spec operates in | **Both.** UI/`DataManager` cluster selection is Kilosort space. The chirp `.npy`'s `cluster_id` array is **Vision space (1-indexed)** — confirmed by tracing `chirp_analysis.py` → `Dataset.get_spike_times_and_parameters()` → `get_spike_times_dict()`, which iterates `self.cell_ids = sorted(self.vcd.get_cell_ids())`, i.e. Vision cell IDs from the `.params`/`.neurons` table, **not** `spike_clusters.npy`. |
| Reads `is_vision_only` flag? | Yes — indirectly, via `get_vision_id_for_cluster()`. |
| Translation used | `dm.get_vision_id_for_cluster(cluster_id)` — **reuse the existing method** (`data_manager.py`, already implements `vid = cluster_id if is_vision_only else cluster_id + 1`). Do not reimplement. |
| Safe access pattern used | ```python\nvid = self.get_vision_id_for_cluster(cluster_id)\nrow = self.chirp_id_to_row.get(vid)\nif row is None:\n    return None\n``` |

**Reminder:** The chirp `.npy`'s `cluster_id` field is a trap — its name suggests
Kilosort space but it is not. Any code that indexes into `chirp_data['psth_mean']`
using a raw UI `cluster_id` without translation is a Law 1 violation.

---

## Block 3 — Affected Files

| File path | Function(s) added or modified | Change type | Touches DataManager? |
|---|---|---|---|
| `src/analysis/data_manager.py` | `__init__` (new attrs), `load_chirp_data()`, `get_chirp_data_for_cluster()` | Modify | Yes |
| `src/gui/workers/workers.py` | `KilosortLoadWorker.run()` | Modify | No |
| `src/gui/panels/chirp_panel.py` | `ChirpPanel` (new class) | Add | No |
| `src/gui/main_window.py` | panel registration, Tier 2 dispatch, theme restyle hook | Modify | No |
| `tests/unit/test_chirp_data.py` | `test_load_chirp_data_*`, `test_get_chirp_data_for_cluster_*` | Add | No |
| `tests/integration/test_chirp_panel.py` | `test_chirp_panel_*` | Add | No |

> **DataManager is touched.** Rebase from `origin/main` before every push.

> ⚠️ **OPEN ITEM:** `main_window.py` was not available while writing this spec.
> The exact line(s) inside `update_cluster_views()` / `_process_selection()`
> where `ChirpPanel.update_all(cluster_id)` gets called, and wherever panel tabs
> are registered/restyled on theme switch, need to be confirmed against the real
> file before this spec is "Ready for Dev." The pattern should mirror
> `StandardPlotsPanel` exactly (same tier, same call site).

---

## Block 4 — Qt Threading Contract

| Operation | Runs on thread | Worker class | Signal name + signature | Receiving slot | Tier 1 or Tier 2? |
|---|---|---|---|---|---|
| Load chirp `.npy` from disk | Background | `KilosortLoadWorker` (existing — extended) | Reuses existing `finished = Signal(bool, str)` | Existing Kilosort-load completion slot | One-time, at dataset load (not per-selection) |
| Render Chirp PSTH plot for selected cluster | Main thread | N/A — direct call | N/A | `ChirpPanel.update_all(cluster_id)` | **Tier 2** |

**Why the panel update is Tier 2 and not Tier 1:** the underlying lookup
(`get_chirp_data_for_cluster()`) is a cheap in-RAM dict/row read with no locks and
no disk I/O — it *would* qualify for Tier 1 by data-access cost alone. But AGENTS.md
§1 Law 2 forbids `panel.update_all()` in Tier 1 unconditionally, regardless of how
cheap the call underneath is. `StandardPlotsPanel.update_all()` follows the same
rule despite also reading from an already-warm cache — this spec matches that
precedent rather than carving out an exception.

**No new worker for per-cluster chirp lookups.** Unlike `StandardPlotsWorker`,
chirp data is fully precomputed offline by `chirp_analysis.py`; there is nothing
to compute per-cluster at view time, only a row lookup. A dedicated queue-worker
would be pure overhead.

**Stale result guard:** N/A — no async result slot is added. `update_all()` is a
synchronous direct call like `StandardPlotsPanel.update_all()`, not a worker
signal receiver, so there is no stale-cluster race to guard against.

---

## Block 5 — Cache Contract

| Question | Answer |
|---|---|
| Which cache(s) does this spec read? | `DataManager.chirp_data` (raw loaded dict) + `DataManager.chirp_id_to_row` (vision_id → row index), both new |
| Which cache(s) does this spec write? | Same two, written once by `load_chirp_data()` at dataset-load time |
| What triggers cache invalidation? | Never during a session — same lifetime as `vision_stas`. Re-loading a dataset re-runs `KilosortLoadWorker`, which reassigns both attributes fresh. |
| Is data persisted to disk? | No — the `.npy` on disk *is* the persistence; RGCViewer only reads it, never writes it. |
| Which lock must be held? | **None.** Matches the existing precedent for `vision_stas`/`vision_eis`: written once by the loader thread before `finished` fires, read-only afterward, no concurrent writer ever exists. This is an explicit deviation from the `_standard_plot_lock`-style pattern, justified because — unlike `standard_plot_cache` — this cache is never incrementally populated per-cluster during scrolling. |
| Must tests bypass a `.pkl` cache? | **No** — this data has no `.pkl` persistence layer. `Must tests bypass cache?` from Law 3 is N/A here. Tests instead need a real (or synthetic) `.npy` file on disk, since `load_chirp_data()` reads from `kilosort_dir` directly. |

---

## Block 6 — DataManager Attributes Used

| Attribute | Type | Can be `None`? | This spec reads / writes | Safe access pattern |
|---|---|---|---|---|
| `chirp_data` | `dict` (raw loaded mdic) | **Yes** — `None` until loaded, or if load failed | Reads + Writes (new) | `if self.chirp_available and self.chirp_data is not None:` |
| `chirp_id_to_row` | `dict[int, int]` | **Yes** | Reads + Writes (new) | `row = self.chirp_id_to_row.get(vid)` — never index directly |
| `chirp_available` | `bool` | No — defaults `False` | Reads + Writes (new) | Direct |
| `kilosort_dir` | `Path` | No (after init) | Reads (existing) | Direct — `self.kilosort_dir.glob('*Chirp*.npy')` |
| `is_vision_only` | `bool` | No | Reads (existing, via `get_vision_id_for_cluster`) | Never access directly — go through `get_vision_id_for_cluster()` |

---

## Block 7 — Acceptance Criteria

### AC1 — Chirp file loads and populates the id index

- **Setup:** Synthetic `.npy` at `tmp_path / "kilosort4_ChirpStimulus.npy"` containing
  `cluster_id = [3, 7, 12]` (Vision-space ids), `psth_mean` shape `(3, 1100)`,
  `quality_index`, `bin_size_ms=20.0`, and all four `phase_*` keys.
- **Action:** `dm = DataManager(kilosort_dir=tmp_path)`; call `dm.load_chirp_data()`.
- **Expected:** Returns `(True, ...)`. `dm.chirp_available is True`.
  `dm.chirp_id_to_row == {3: 0, 7: 1, 12: 2}`.
- **Test type:** Unit

### AC2 — Missing chirp file does not crash and disables the feature cleanly

- **Setup:** Empty `tmp_path`, no `.npy` files present.
- **Action:** Call `dm.load_chirp_data()`.
- **Expected:** Returns `(False, "No chirp analysis file found.")`.
  `dm.chirp_available is False`. `dm.chirp_data is None`. No exception raised.
- **Test type:** Unit

### AC3 — Vision ID offset is applied correctly (hybrid dataset)

- **Setup:** `dm.is_vision_only = False`. Chirp file has `cluster_id = [6]`
  (Vision-space). UI selects Kilosort `cluster_id = 5`.
- **Action:** Call `dm.get_chirp_data_for_cluster(5)`.
- **Expected:** Returns non-`None` dict — `5 + 1 = 6` resolves to row 0.
  Calling with raw `cluster_id = 6` directly (no translation) must return `None`,
  proving the lookup isn't accidentally working in the wrong ID space by luck.
- **Test type:** Unit — parametrize both branches per Law 1 convention
  (`test_get_cell_physics_vision_id_offset` is the existing template to mirror).

### AC4 — Vision-only dataset uses identity mapping

- **Setup:** `dm.is_vision_only = True`. Chirp file has `cluster_id = [6]`.
- **Action:** Call `dm.get_chirp_data_for_cluster(6)`.
- **Expected:** Returns non-`None` dict — no `+1` applied.
- **Test type:** Unit (same parametrize block as AC3)

### AC5 — Cell present in dataset but absent from chirp file

- **Setup:** Chirp file loaded successfully; UI selects a valid `cluster_id` whose
  translated vision id is not in `chirp_id_to_row`.
- **Action:** Call `dm.get_chirp_data_for_cluster(cluster_id)`.
- **Expected:** Returns `None`. No exception.
- **Test type:** Unit

### AC6 — Panel shows placeholder when chirp data unavailable

- **Setup:** `dm.chirp_available = False`.
- **Action:** `ChirpPanel.update_all(cluster_id)` for any cluster.
- **Expected:** Panel displays placeholder text "No chirp data" (mirrors
  `STAPanel`'s "No STA data" pattern from `SPEC.md`'s own AC2 example). No crash,
  no exception dialog.
- **Test type:** Integration (`make_main_window + qtbot`)

### AC7 — Visual: PSTH renders with phase boundaries marked

- **State to reproduce:**
  1. Load a real dataset with a chirp `.npy` present in `kilosort_dir`.
  2. Select a cluster known to have chirp responses.
  3. Navigate to the Chirp tab.
- **Expected appearance:** PSTH line plot (Hz vs. time), with four vertical
  boundary markers (or shaded `LinearRegionItem`s) separating step-on, step-off,
  frequency sweep, and contrast sweep phases. Quality index shown as a label.
  Y-axis is not flat/stuck at 0 for a known-responsive cell.
- **Must verify:** Dark mode AND light mode.
- **Screenshot filenames:**
  - `tests/screenshots/ac7_chirp_panel_dark.png`
  - `tests/screenshots/ac7_chirp_panel_light.png`
- **Verified by:** `[ ]` Author `[ ]` Reviewer

---

## Block 8 — Regression Guard

> ⚠️ **INCOMPLETE — `PLAN.md` was not available while writing this spec.**
> Before this spec is "Ready for Dev," look up `data_manager.py`,
> `workers.py`, and `main_window.py` in PLAN.md's "Completed Fix Registry" and
> list any overlapping prior fixes here (e.g. anything touching
> `KilosortLoadWorker.run()` or the vision-auto-detection block it currently
> contains). Do not skip this step.

| Prior fix | Files overlap | Regression test to run | When to run it |
|---|---|---|---|
| *(TBD from PLAN.md)* | `workers.py::KilosortLoadWorker.run()` | *(TBD)* | Before opening PR |

---

## Block 9 — Test Plan

### Unit tests

File: `tests/unit/test_chirp_data.py`

| Test function name | Fixture | What it asserts | Cache bypass needed? |
|---|---|---|---|
| `test_load_chirp_data_populates_index` | `tmp_path` | AC1 | N/A — no `.pkl` cache involved |
| `test_load_chirp_data_missing_file_returns_false` | `tmp_path` | AC2 | N/A |
| `test_load_chirp_data_missing_required_keys_returns_false` | `tmp_path` | Malformed `.npy` (missing `psth_mean`) → `(False, ...)`, no exception | N/A |
| `test_get_chirp_data_hybrid_applies_vision_offset` | `tmp_path` | AC3 | N/A |
| `test_get_chirp_data_vision_only_identity_mapping` | `tmp_path` | AC4 | N/A |
| `test_get_chirp_data_cluster_not_in_chirp_file_returns_none` | `tmp_path` | AC5 | N/A |
| `test_get_chirp_data_before_load_returns_none` | `tmp_path` | Calling before `load_chirp_data()` ever ran → `None`, not `AttributeError` | N/A |

### Integration tests

File: `tests/integration/test_chirp_panel.py`

| Test function name | Fixture | What it exercises |
|---|---|---|
| `test_chirp_panel_shows_placeholder_when_unavailable` | `make_main_window + qtbot` | AC6 |
| `test_chirp_panel_renders_psth_for_available_cell` | `make_main_window + qtbot` | PSTH curve populated with non-empty data for a synthetic chirp-loaded cell |
| `test_chirp_panel_no_cross_panel_reference` | `make_main_window + qtbot` (code review assertion) | `chirp_panel.py` imports nothing from `panels/` — enforces AGENTS.md "panels never reference other panels" |

### Visual regression tests

Tool: `pytest-mpl`
Baseline location: `tests/baseline_images/chirp_panel/`
Generate with: `conda run -n rgcviewer python -m pytest --mpl-generate-path tests/baseline_images/ tests/`

---

## Block 10 — Out of Scope

- Does **not** add per-trial raster or `spikes_binned` visualization — only
  `psth_mean` and `quality_index` are surfaced in v1. (`spikes_binned` is loaded
  into `chirp_data` for future use but not rendered.)
- Does **not** add chirp data to the UMAP/GMM classification feature set —
  that's `joint_sta.py`/`dog_rf_refinement.py` territory, not this GUI spec.
- Does **not** modify `chirp_analysis.py` or its output schema in any way.
- Does **not** add a `ChirpLoadWorker`/QThread — loading happens inline inside
  the existing `KilosortLoadWorker.run()`.
- Does **not** touch `standard_plot_cache`, `feature_cache`, or any existing
  panel's rendering code.
- Does **not** handle `.mat`-format chirp files — `.npy` only, since that's
  what `DataManager` already reads elsewhere in this codebase (Vision-adjacent
  files aside).
