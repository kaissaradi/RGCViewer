# SPEC.md — Fix the Dead Chirp Checkbox in the UMAP Panel

## Block 0 — Metadata

| Field | Value |
|---|---|
| **Date created** | 2026-07-13 |
| **Branch** | `fix/umap-chirp-checkbox` |
| **Author** | Kais |
| **Spec status** | Ready to implement |
| **Predecessor** | `docs/specs/chirp_umap_feature_spec.md` (Stages 1–4 landed; this fixes the Stage 3 defect) |
| **Touches `data_manager.py`?** | **No** — GUI-only. No rebase-before-push obligation (Concurrency Rule 3 does not apply). |

---

## Progress Log (resume point)

> Update after every stage so a fresh session can pick up here.

- [ ] **Stage 1 — failing test.** Integration test proving the chirp checkbox is
  enabled+checked after a chirp-bearing dataset finishes loading.
- [ ] **Stage 2 — `UMAPPanel.refresh_feature_availability()`.** Extract the gate,
  make it bidirectional, call it from `__init__` and from the load-finished slot.
- [ ] **Stage 3 — wire the hook in `callbacks._on_kilosort_loaded`.**
- [ ] **Stage 4 — verify end-to-end on real chirp data + update the predecessor spec's status.**

---

## Block 1 — Problem Statement

**Symptom:** The "Chirp PSTH (response shape)" checkbox in the UMAP panel is
permanently greyed out and unchecked. A scientist with chirp data loaded cannot
include it in the embedding. The feature is unreachable from the GUI.

**Root cause — the gate is evaluated against a `DataManager` that does not exist yet.**

The availability gate lives in `UMAPPanel.__init__` (`src/gui/panels/umap_panel.py:261-268`):

```python
if use_key == "use_chirp":
    dm = getattr(self.main_window, "data_manager", None)
    if not getattr(dm, "chirp_available", False):
        chk.setChecked(False)
        chk.setEnabled(False)
        ...
```

The construction order makes this unconditionally false:

1. `MainWindow.__init__` sets `self.data_manager = None` (`main_window.py:99`).
2. `MainWindow.__init__` calls `_setup_ui()` (`main_window.py:135`), which constructs
   `UMAPPanel(self)` (`main_window.py:1266`). **The gate runs here.** `dm` is `None`,
   so `getattr(None, "chirp_available", False)` → `False` → box disabled + unchecked.
3. Only later, when the user picks a directory, does `callbacks.py:106` create the real
   `DataManager`, and `KilosortLoadWorker.run()` call `load_chirp_data()`
   (`workers.py:117`), which sets `chirp_available = True`.

Nothing re-runs the gate after step 3. The checkbox is therefore dead **on every
dataset, whether or not chirp data exists** — the disabled state is not reporting
"no chirp data," it is reporting "no dataset had loaded when this widget was built."

**Everything downstream of the checkbox already works** and is covered by passing
tests — this is the *only* break in the chain:

| Layer | Location | Status |
|---|---|---|
| Checkbox → config dict | `UMAPPanel.get_feature_config()` (`umap_panel.py:323`) | ✅ reads `use_chirp` / `w_chirp` |
| Config → worker | `umap_panel.py:538` → `UMAPWorker` (`workers.py:542`) | ✅ passed through |
| Raw chirp block | `DataManager.get_raw_feature_blocks()` (`data_manager.py:2233-2258`) | ✅ `TestGetRawFeatureBlocksChirp` (5 tests) |
| PCA block | `analysis_core.build_feature_matrix()` (`analysis_core.py:834-853`) | ✅ `TestBuildFeatureMatrixChirp` (5 tests) |
| **UI gate** | **`umap_panel.py:261-268`** | ❌ **dead — this spec** |

**User story:** "As a scientist, when I load a dataset that has chirp data, the
Chirp PSTH checkbox should be live so I can include chirp response shape in the
UMAP. When I load one that doesn't, it should be greyed out and tell me why."

---

## Block 2 — Vision ID Contract

| Question | Answer |
|---|---|
| ID space | **N/A — this spec introduces no ID translation.** |
| Translation used | None. The Vision→Kilosort offset is baked into `chirp_id_to_row` at load time by `load_chirp_data()`. This spec touches only widget enable/check state and reads one boolean (`chirp_available`). **Do not add an ID translation here** (Law 1: one translation point, already implemented). |

---

## Block 3 — Design Decisions

### 3.1 Where the re-gate fires — `_on_kilosort_loaded`

`load_chirp_data()` is called inside `KilosortLoadWorker.run()` (`workers.py:117`),
so by the time that worker's `finished` signal reaches the main thread,
`chirp_available` is final. The slot is
`callbacks._on_kilosort_loaded()` (`callbacks.py:133`) — already the post-load
UI-enabling slot, and already the precedent for exactly this pattern:

```python
# callbacks.py:193-197 — existing precedent
if hasattr(main_window, "similarity_panel") and main_window.data_manager.vision_available:
    main_window.similarity_panel.on_vision_loaded()
```

We add the same shape of hook for the UMAP panel. **Do not** re-gate from
`update_cluster_views()` or any per-selection path — that is Tier 1 and this would
be a needless per-keypress widget write (Law 2). A load-finished slot is Tier 2 by
construction; this is trivially compliant.

### 3.2 The gate must be **bidirectional**

Panels are constructed once and persist; `main_window.data_manager` is **replaced**
on every load (`callbacks.py:106`, `callbacks.py:427`). So loading a chirp-bearing
dataset A and then a chirp-less dataset B must *re-disable* the box. The current
code only ever disables — it has no enable path at all. The refresh method must set
enabled/checked state from `chirp_available` in **both** directions, and clear the
tooltip when the feature becomes available.

### 3.3 Single source of truth for the gate

`__init__` and the post-load refresh must not carry two copies of the gate logic
(that is how the current bug survives a re-read). Extract one private helper and
call it from both. `__init__` calls it with `data_manager is None` and correctly
lands on "disabled" — the startup state stays exactly as it is today, which is the
right state *before any dataset is loaded*.

### 3.4 Default state when chirp IS available — **checked**

Every other feature block defaults to `setChecked(True)` (`umap_panel.py:222`).
Chirp gets parity: when a chirp-bearing dataset loads, the box comes up **checked**,
so chirp participates in the next embedding without the user hunting for a toggle.
The weight slider keeps `DEFAULT_WEIGHT_CHIRP` (3.0).

Re-gating on dataset load resets any manual uncheck the user made against the
*previous* dataset. That is correct: a new dataset is a new feature-config context,
and the reset is visible (the box is on screen).

### 3.5 What "available" means — unchanged

`DataManager.chirp_available` remains the single boolean. This spec does not change
how it is computed, when it is set, or what it implies. It only makes the GUI *read
it at the right time*.

---

## Block 4 — Implementation Plan

> AGENTS.md order: **failing test first, then implement.**
> GUI-only. Does not touch `data_manager.py`, `analysis_core.py`, or `workers.py`.

### 4.1 `src/gui/panels/umap_panel.py`

**(a)** In the `features_info` loop, replace the inline `if use_key == "use_chirp":`
block (lines 261-268) with a call to the new helper, so the loop just registers
widgets:

```python
self.feature_widgets[use_key] = (chk, slider, w_key)
```

**(b)** After the loop in `__init__`, call the public refresh once so the startup
(no-dataset) state is set by the same code path that the post-load hook uses:

```python
self.refresh_feature_availability()
```

**(c)** Add the two new methods:

```python
def refresh_feature_availability(self):
    """Re-gate data-dependent feature checkboxes against the CURRENT DataManager.

    Called once at construction (when data_manager is still None → chirp
    disabled) and again from callbacks._on_kilosort_loaded once a dataset has
    finished loading and chirp_available is final. Must be bidirectional:
    panels persist across loads while data_manager is replaced, so a
    chirp-less dataset following a chirp-bearing one has to re-disable the row.
    """
    dm = getattr(self.main_window, "data_manager", None)
    self._set_feature_enabled(
        "use_chirp",
        enabled=bool(getattr(dm, "chirp_available", False)),
        disabled_tooltip="No chirp data loaded for this dataset",
    )

def _set_feature_enabled(self, use_key, enabled, disabled_tooltip=""):
    """Enable/disable one feature row (checkbox + weight slider + readout)."""
    entry = self.feature_widgets.get(use_key)
    if entry is None:
        return
    chk, slider, _w_key = entry
    chk.setEnabled(enabled)
    chk.setChecked(enabled)          # see spec 3.4 — parity with other blocks
    slider.setEnabled(enabled)
    chk.setToolTip("" if enabled else disabled_tooltip)
    # The readout QLabel is driven by chk.toggled (wired at construction), so
    # setChecked() above already propagates to it. Do not reach for it here —
    # keep the widget refs in feature_widgets as the only handle.
```

> **Note on the readout label:** the current loop connects
> `chk.toggled → value_label.setEnabled` but does **not** store `value_label` in
> `self.feature_widgets`. `setChecked()` fires `toggled` only on an actual state
> *change*, so the label's enabled state stays in sync for every real transition.
> If Stage 2 finds an edge case where it desyncs (e.g. re-gating to the same
> state), extend `feature_widgets` to a 4-tuple `(chk, slider, w_key, value_label)`
> and set it explicitly — and update `get_feature_config()`'s unpack accordingly.

### 4.2 `src/gui/callbacks.py`

In `_on_kilosort_loaded`, in the "Handle Vision specific UI updates" region
(alongside the existing `similarity_panel.on_vision_loaded()` precedent, ~line 197),
add:

```python
# Chirp is loaded inside KilosortLoadWorker.run(), so chirp_available is final
# by the time this slot runs. Re-gate the UMAP feature checkboxes against the
# now-loaded DataManager — at UMAPPanel construction time data_manager was
# still None, so the chirp row was disabled regardless of the dataset.
if hasattr(main_window, "umap_panel"):
    main_window.umap_panel.refresh_feature_availability()
```

---

## Block 5 — Test Plan (write these first)

Qt integration — the bug is *construction-order*, so a pure-logic unit test cannot
catch it. Use `make_main_window + qtbot` (AGENTS.md §9).
File: `tests/integration/test_umap_chirp_checkbox.py`.

1. **`test_chirp_checkbox_disabled_before_any_load`** *(guards the startup state)*
   Fresh `MainWindow`, no dataset. `feature_widgets["use_chirp"]` checkbox is
   disabled and unchecked. (This passes today — it locks in the correct pre-load
   behavior so the fix doesn't over-correct into "enabled by default".)

2. **`test_chirp_checkbox_enabled_after_load_when_available`** ← **the failing test**
   Set `main_window.data_manager` to a mock/DM with `chirp_available = True`, call
   `umap_panel.refresh_feature_availability()`, assert the checkbox is **enabled and
   checked**, the slider is enabled, and the tooltip is empty. **Fails today**
   (method does not exist; the box is frozen disabled).

3. **`test_chirp_checkbox_redisabled_when_next_dataset_has_no_chirp`**
   Refresh with `chirp_available = True`, then swap in a DM with
   `chirp_available = False` and refresh again. Assert disabled + unchecked + the
   "No chirp data loaded" tooltip is back. Guards Decision 3.2 (bidirectionality).

4. **`test_chirp_checkbox_reaches_feature_config`** *(proves "actually used")*
   With `chirp_available = True` and post-refresh, `get_feature_config()` returns
   `use_chirp is True` and `w_chirp == DEFAULT_WEIGHT_CHIRP`. This is the seam
   between the fixed widget and the already-working embedding pipeline.

5. **`test_on_kilosort_loaded_refreshes_umap_panel`**
   Monkeypatch `umap_panel.refresh_feature_availability` with a spy; drive
   `callbacks._on_kilosort_loaded(main_window, success=True, ...)` and assert the
   spy fired. Guards the wiring in 4.2 against a future refactor of the slot.

6. **Regression:** `tests/integration/test_umap_panel_clustering.py`,
   `test_umap_selection.py`, `test_umap_worker.py`, `tests/unit/test_dynamic_clustering.py`,
   `tests/unit/test_raw_feature_blocks.py` must all still pass unchanged.

**Known pre-existing failures — do not fix here, do not skip** (inherited from the
predecessor spec, AGENTS.md rule 5):
`tests/unit/test_raw_feature_blocks.py::TestGetRawFeatureBlocks::test_all_filtered_returns_empty`
and `::test_scalars_all_columns_present` (stale `scalars` column set, pre-dates the
grating refactor).

---

## Block 6 — Manual Verification (Stage 4)

The unit/integration tests use a mocked `chirp_available`; confirm once against real data:

```bash
conda run -n rgcviewer python -m src.gui.main_window
# Load: /mnt/lab/Array-data/sorted/20260506A/chunk10/kilosort2.5
```

1. UMAP tab → "Chirp PSTH (response shape)" is **enabled and checked**, slider live at 3.0.
2. Run UMAP → status/log shows a `chirp_pc*` contribution, **not**
   `"build_feature_matrix: skipping chirp block"` (`analysis_core.py:852`).
3. Uncheck chirp → re-run → embedding visibly changes. This is the actual
   acceptance criterion: the toggle *moves the points*.

---

## Block 7 — Out of Scope

- **No change to the embedding math, weights, PCA component count, or the QI gate.**
  `CHIRP_PCA_COMPONENTS`, `CHIRP_MIN_QI`, `DEFAULT_WEIGHT_CHIRP` are untouched.
- **No change to `data_manager.py`, `analysis_core.py`, or `workers.py`.**
  All three already do the right thing.
- **No change to the chirp panel** or to `get_chirp_data_for_cluster()`.
- **Vision-only datasets still cannot use chirp.** `load_chirp_data()` is only
  called from `KilosortLoadWorker.run()` (`workers.py:117`); the standalone Vision
  path (`callbacks.py:427`) never calls it, so `chirp_available` stays `False` and
  the box stays (correctly) greyed. This is a *real* second gap with a *different*
  cause, and a scientist on a vision-only dataset will still see a dead checkbox —
  but it is a data-loading change, not the construction-order bug, and it is
  deliberately excluded here. **Flag as a follow-up spec if it matters.**
- No per-phase chirp sub-blocks (still deferred from the predecessor spec).
