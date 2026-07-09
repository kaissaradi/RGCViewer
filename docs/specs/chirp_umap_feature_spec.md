# SPEC.md — Chirp PSTH PCA as a UMAP Feature Block

## Block 0 — Metadata

| Field | Value |
|---|---|
| **Date created** | 2026-07-08 |
| **Branch** | `feat/chirp-umap-feature` |
| **Author** | Kais |
| **Spec status** | In progress |

---

## Progress Log (resume point)

> Update after every stage so a fresh session can pick up here.

- [x] **Stage 1 — constants + `build_feature_matrix` chirp block.** Added
  `CHIRP_PCA_COMPONENTS`, `CHIRP_MIN_QI`, `DEFAULT_WEIGHT_CHIRP` to
  `constants.py`; added the chirp PCA block to `analysis_core.build_feature_matrix`
  (after grating, before scalars). Tests: `TestBuildFeatureMatrixChirp` (5 tests)
  in `tests/unit/test_dynamic_clustering.py` — **passing**.
- [x] **Stage 2 — `DataManager.get_raw_feature_blocks`.** Emits the `'chirp'`
  block: L2-normalized PSTH shape, QI gate (`CHIRP_MIN_QI`), zero sentinel for
  missing/low-QI cells, `'chirp'` key in both the normal and empty-blocks return
  dicts. **Deviation from spec Block 5.2:** accesses `chirp_id_to_row` +
  `chirp_data['psth_mean' / 'quality_index']` directly rather than
  `get_chirp_data_for_cluster()` — equally Law-1-safe (the row map has the offset
  baked in at load) and avoids depending on the full phase-region schema for a
  path that only needs two arrays. Tests: `TestGetRawFeatureBlocksChirp` (5) in
  `tests/unit/test_raw_feature_blocks.py` — **passing**.
- [ ] **Stage 3 — `UMAPPanel` UI.** `features_info` entry + disable-when-no-chirp
  gate in `umap_panel.py`. Run full suite.
- [ ] **Stage 4 — docs.** Delete `rgcviewer_stabilization_plan.md`, update
  `docs/PLAN.md`.

**Pre-existing unrelated failures** (not caused by this work, do not fix here):
`tests/unit/test_raw_feature_blocks.py::TestGetRawFeatureBlocks::test_all_filtered_returns_empty`
and `::test_scalars_all_columns_present` assert an old `scalars` column set from
before the grating refactor. Flagged per AGENTS.md rule 5.

---

## Block 1 — Problem Statement

**Goal:** Add the chirp response PSTH to the UMAP feature space as its own
PCA-reduced block, gated on availability — so that when a dataset has chirp
data loaded, a cell's chirp response *shape* contributes to how it clusters,
alongside the existing Temporal STA, ACG, RF diameter, and Grating tuning-shape
blocks.

**Why the shape, not a scalar:** the chirp panel already surfaces a scalar
quality index (`QI`), but QI collapses the entire response (ON/OFF step, frequency
sweep, contrast sweep) into a single "how reliable" number. Two cells with the
same QI can have completely different chirp responses (e.g. an ON-transient cell
vs. a sustained-OFF cell). This mirrors the exact reasoning already documented for
the grating block: PCA the response curve itself so differently-shaped responses
that share a summary scalar stay distinguishable in the embedding, rather than
collapsing to a scalar the way DSI/OSI once were.

**User story:** "As a scientist, when chirp data is loaded I want cells that
respond to the chirp stimulus in similar ways to land near each other in the
UMAP, without me having to eyeball every PSTH — and when chirp data is *not*
loaded, the feature should silently disappear rather than break the embedding."

---

## Block 2 — Vision ID Contract

| Question | Answer |
|---|---|
| ID space | **Kilosort space** — `cluster_id` used directly, no `+1` translation. Chirp data is keyed the same way the chirp panel already reads it. |
| Translation used | None at the feature layer. `DataManager.get_chirp_data_for_cluster(cluster_id)` handles the Vision→Kilosort translation internally at load time (`load_chirp_data` builds `chirp_id_to_row` with `cid - 1` unless `is_vision_only`). **Do not re-translate.** |

This block must call `get_chirp_data_for_cluster(cid)` and never touch
`chirp_data` / `chirp_id_to_row` directly (Law 1 discipline — one translation
point, already implemented).

---

## Block 3 — Data Source & Availability

Chirp data is **always precomputed offline** by `chirp_analysis.py` and loaded
once into `DataManager.chirp_data` via `load_chirp_data()`. Unlike grating, there
is **no in-GUI calculation path** — either the `*Chirp*.npy` file exists and
`chirp_available` is `True`, or the feature is absent.

Per-cell access (the only supported path):

```python
d = dm.get_chirp_data_for_cluster(cid)   # dict or None
# d['psth_mean']  -> (n_bins,) ndarray, Hz
# d['quality_index'] -> float (NaN if silent)
# d['bin_size_ms'] -> float
```

- `psth_mean` has a **fixed `n_bins` across all cells in a dataset** (it comes
  from one precomputed array `chirp_data['psth_mean']` of shape `(n_cells,
  n_bins)`). This is what makes a fixed-width feature block possible without an
  interpolation step (unlike grating, whose directions vary per condition).
- `get_chirp_data_for_cluster` returns `None` for a cell not present in the chirp
  file → that cell gets a zero sentinel row (same pattern as temporal/grating).

---

## Block 4 — Design Decisions (resolve before implementing)

### 4.1 Amplitude normalization — **shape, not firing rate**

`psth_mean` is in Hz and its magnitude scales with a cell's firing rate. Firing
rate was **deliberately removed** from the embedding (see `constants.py`
`DEFAULT_WEIGHT_*` comments), so the chirp block must not smuggle it back in via
raw PSTH amplitude.

**Recommendation:** L2-normalize each cell's PSTH row (`psth / (norm + eps)`)
before PCA, so the block encodes response *shape* only. This matches the
grating block's per-condition peak normalization intent. Do **not** peak-align —
the chirp stimulus is identical and time-locked for every cell, so temporal
position is meaningful and must be preserved (this is the key difference from the
Temporal STA block, which peak-aligns because STA peak latency is per-cell).

### 4.2 Quality-index gate — **sentinel out the noise cells**

A cell with no real chirp response (low/NaN QI) contributes only noise to the
PCA. The grating block already gates analogously ("no condition with usable peak
response → zero sentinel").

**Recommendation:** add `CHIRP_MIN_QI` to `constants.py`. In
`get_raw_feature_blocks`, if `quality_index` is NaN or `< CHIRP_MIN_QI`, append
the zero sentinel instead of the (normalized) PSTH. Start conservative
(`CHIRP_MIN_QI = 0.0`, i.e. only gate out NaN/silent) and let the user raise it;
document that this only affects the *embedding*, not the chirp panel display.

### 4.3 Sentinel width

The zero sentinel must match the real block width. Derive `n_bins` from the
loaded data (`dm.chirp_data['psth_mean'].shape[1]`) inside
`get_raw_feature_blocks`, captured once before the per-cell loop. If
`chirp_available` is `False`, append nothing and emit an **empty** `'chirp'`
block — `build_feature_matrix` then skips it via the `std == 0` / empty guard,
exactly like grating.

### 4.4 Component count & weight

- `CHIRP_PCA_COMPONENTS = 4` (match temporal / ACG / grating).
- `DEFAULT_WEIGHT_CHIRP` — start at `3.0` (same as grating; tune per data). Chirp
  and grating are both "stimulus response shape" blocks, so parity is a sane
  default. Temporal STA stays the dominant block (weight 10).

---

## Block 5 — Implementation Plan

> Follow AGENTS.md order: **write the failing test first, then implement.**
> This touches `data_manager.py` — rebase `origin/main` before every push and
> flag the PR (Concurrency Rule 3).

### 5.1 `src/analysis/constants.py`

```python
CHIRP_PCA_COMPONENTS = 4        # PCA on the L2-normalized chirp PSTH shape
CHIRP_MIN_QI = 0.0              # embedding-only QI gate; sentinel below this
DEFAULT_WEIGHT_CHIRP = 3.0      # parity with grating; tune per data
```

### 5.2 `DataManager.get_raw_feature_blocks` (`data_manager.py`)

Mirror the grating block exactly:

1. Add `chirp_list = []` alongside `grating_list`.
2. Capture width once: `chirp_n_bins = dm.chirp_data['psth_mean'].shape[1]` if
   `chirp_available` else `0`.
3. In the per-`cid` loop:
   ```python
   d = self.get_chirp_data_for_cluster(cid)   # Law-1-safe accessor
   qi = d.get("quality_index") if d else None
   if (
       d is not None
       and chirp_n_bins > 0
       and qi is not None and np.isfinite(qi) and qi >= CHIRP_MIN_QI
   ):
       psth = np.asarray(d["psth_mean"], dtype=np.float64).copy()
       norm = np.linalg.norm(psth)
       chirp_list.append(psth / norm if norm > 0 else np.zeros(chirp_n_bins))
   else:
       chirp_list.append(np.zeros(chirp_n_bins, dtype=np.float64))
   ```
4. Add `'chirp'` to the returned `raw_blocks` dict:
   `np.vstack(chirp_list)` when `chirp_n_bins > 0`, else `np.empty((len(valid_ids), 0))`.
5. Add `'chirp': np.empty((0, 0))` to the early-return `empty_blocks` dict so the
   key is always present.

### 5.3 `analysis_core.build_feature_matrix` (`analysis_core.py`)

Add a chirp block, copy-paste-adapted from the grating block (lines ~790–819):

```python
if feature_config.get("use_chirp", True):
    w = feature_config.get("w_chirp", 3.0)
    chirp_matrix = raw_blocks.get("chirp")
    if chirp_matrix is None or chirp_matrix.size == 0:
        chirp_available = False
    else:
        chirp_matrix = chirp_matrix.copy()
        chirp_available = np.std(chirp_matrix) > 0
    if chirp_available:
        n_comp = min(CHIRP_PCA_COMPONENTS, N - 1, chirp_matrix.shape[1])
        if n_comp >= 1:
            chirp_pca = PCA(n_components=n_comp).fit_transform(chirp_matrix)
            chirp_pca = StandardScaler().fit_transform(chirp_pca)
            blocks.append(chirp_pca * w)
            labels.extend([f"chirp_pc{j}" for j in range(n_comp)])
    else:
        logger.debug("build_feature_matrix: skipping chirp block — "
                     "no cells with usable chirp responses (all-zero matrix).")
```

Add `CHIRP_PCA_COMPONENTS` to the `from .constants import (...)` block.

### 5.4 `UMAPPanel` feature UI (`umap_panel.py`)

Append to `features_info` (after grating):

```python
("Chirp PSTH (response shape)", "use_chirp", "w_chirp", DEFAULT_WEIGHT_CHIRP),
```

**Availability gating ("if avail"):** after building the checkbox, disable and
uncheck it when chirp isn't loaded, so the user can't enable a dead feature:

```python
if use_key == "use_chirp" and not getattr(
    self.main_window.data_manager, "chirp_available", False
):
    chk.setChecked(False)
    chk.setEnabled(False)
    chk.setToolTip("No chirp data loaded for this dataset")
```

Import `DEFAULT_WEIGHT_CHIRP` from constants. The existing all-disabled guard
(`umap_panel.py:480`) and the `build_feature_matrix` ValueError guard
(`workers.py`, Stabilization Issue D) already handle the degenerate case, so no
new error path is needed.

---

## Block 6 — Test Plan (write these first)

Use `tmp_path` / `mock_dm` fixtures — pure logic, no Qt, no real disk (Law 3).

1. **`test_build_feature_matrix_includes_chirp_block`**
   Given `raw_blocks` with a nonzero `'chirp'` matrix and `use_chirp=True`,
   assert `chirp_pc0..3` appear in `col_labels` and the matrix width grows by
   `min(CHIRP_PCA_COMPONENTS, N-1, n_bins)`.

2. **`test_build_feature_matrix_skips_chirp_when_all_zero`**
   All-zero `'chirp'` block → no `chirp_pc*` labels, no NaNs in the output
   (the `std == 0` guard fires — same contract as grating/temporal).

3. **`test_build_feature_matrix_chirp_disabled`**
   `use_chirp=False` → block absent even when data is present.

4. **`test_get_raw_feature_blocks_chirp_width_and_sentinel`**
   With a mocked `chirp_data`, assert every returned `'chirp'` row has width
   `n_bins`, present cells are L2-normalized (`‖row‖ ≈ 1`), and a cell missing
   from `chirp_id_to_row` (or `QI < CHIRP_MIN_QI`) is an all-zero row.

5. **`test_get_raw_feature_blocks_chirp_absent`**
   `chirp_available=False` → `raw_blocks['chirp']` is empty (width 0) and
   `build_feature_matrix` runs without error, producing the same embedding it
   would have without the block.

6. **Regression:** existing UMAP integration tests
   (`tests/integration/test_umap_*`) must still pass unchanged — the chirp block
   is additive and defaults to skipped when data is absent.

---

## Block 7 — Out of Scope

- No new offline computation. If `*Chirp*.npy` is missing, the feature is simply
  unavailable — no in-GUI PSTH computation path (this is the deliberate
  difference from grating).
- No change to the chirp *panel* rendering or to `get_chirp_data_for_cluster`'s
  return shape.
- No per-phase (ON-step / OFF-step / freq-sweep / contrast) sub-blocks in v1 —
  the whole PSTH shape is PCA'd as one vector. Splitting per phase is a possible
  future refinement, not part of this spec.
