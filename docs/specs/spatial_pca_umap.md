# docs/specs/spatial_pca_umap.md

> Reading order: AGENTS.md (full) → PLAN.md (Fragile Zones) → this spec → write failing tests → implement.

---

## Block 0 — Metadata

| Field | Value |
|---|---|
| **Date created** | 2026-05-27 |
| **Last updated** | 2026-05-27 |
| **Commit hash when spec was written** | `712b1eb` |
| **Branch** | `feat/spatial-pca-umap` |
| **Author** | Antigravity (AI Core Developer) |
| **Spec status** | Ready for Dev |

---

## Block 1 — Problem Statement

**Symptom:**
UMAP classification is suboptimal because spatial RF characteristics (size, shape, eccentricity) are represented by only two fragile and lossy Gaussian-fit scalars (`rf_area` and `ellipticity`). Additionally, if Vision STA/parameter files are missing, the UMAP tab is entirely non-functional, even though Kilosort spikes provide high-quality kinetic and temporal features (ACG, Firing Rate, ISI). Finally, scientists cannot balance the relative influence of kinetic vs. spatial characteristics of ganglion cell classes.

**Root cause:**
1. `get_physics_feature_matrix()` currently relies on `rf_area` and `ellipticity` which silently default to `0.0` if `stafit` is unavailable.
2. The data collection in `get_physics_feature_matrix()` uses a hard gate: `if tc is None or acg is None: continue`. When Vision data is missing, `tc` is always `None`, excluding all cells.
3. Feature weights `W_SHAPE`, `W_PATTERN`, and `W_GEOMETRY` are hardcoded constants in `umap_panel.py`.

**User story:**
> "As a scientist analyzing multi-electrode array recordings, I want a robust spatial PCA classification system that works even without Vision files (falling back to Kilosort-only features), and gives me adjustable sliders so I can interactively tune how much weight is placed on kinetics vs. spatial receptive fields."

---

## Block 2 — Vision ID Contract

This spec touches `analysis_core.py`, `data_manager.py`, and `umap_panel.py`. It accesses STA movies through the DataManager which acts as the translation layer.

| Question | Answer |
|---|---|
| Does this spec access Vision file data? | Yes |
| ID space this spec operates in | Both — Translated appropriately via `get_cell_physics()` |
| Reads `is_vision_only` flag? | Yes |
| Translation used | `vid = cluster_id if is_vision_only else cluster_id + 1` inside `get_cell_physics()` |
| Safe access pattern used | `metrics = self.get_cell_physics(int(cid))` |

---

## Block 3 — Affected Files

| File path | Function(s) added or modified | Change type | Touches DataManager? |
|---|---|---|---|
| `src/analysis/constants.py` | Add grid size and threshold constants | Modify | No |
| `src/analysis/analysis_core.py` | Add `extract_spatial_footprint()` | Add | No |
| `src/analysis/data_manager.py` | Update `get_cell_physics()` to extract and cache footprint | Modify | Yes |
| `src/analysis/data_manager.py` | Update `get_physics_feature_matrix()` to run PCA on footprint, handle missing blocks, and support Kilosort-only features | Modify | Yes |
| `src/analysis/data_manager.py` | Add cache format migration check to `load_persisted_caches()` | Modify | Yes |
| `src/gui/panels/umap_panel.py` | Update UI controls (add Row 3 sliders and checkboxes) | Modify | No |
| `src/gui/panels/umap_panel.py` | Update `UMAPWorker` and delegation to pass sliders / features | Modify | No |
| `tests/unit/test_spatial_footprint.py` | Implement Phase 1 footprint extraction tests | Add | No |
| `tests/unit/test_data_manager_spatial_pca.py`| Implement Phase 2 & 4 tests | Add | Yes |

> **DataManager is touched.** Rebase from main before every push.

---

## Block 4 — Qt Threading Contract

| Operation | Runs on thread | Worker class | Signal name + signature | Receiving slot | Tier 1 or Tier 2? |
|---|---|---|---|---|---|
| Run UMAP Projection | Background | `UMAPWorker` | `finished = Signal(object, object, object)` | `on_processing_finished()` | Tier 2 |
| Extract features & matrix | Background | `UMAPWorker` | Direct call inside `run()` | N/A | Tier 2 |
| Render fresh projection | Main thread | N/A | N/A | `update_plot()` | Tier 2 |

**Stale result guard:** Already implemented in `on_processing_finished()`.

---

## Block 5 — Cache Contract

| Question | Answer |
|---|---|
| Which cache(s) does this spec read? | `feature_cache` (to fetch standard plot/spatial/tc keys) |
| Which cache(s) does this spec write? | `feature_cache` (to store computed `spatial_footprint`) |
| What triggers cache invalidation? | Loading a new directory or finding a pre-spatial-footprint schema in `feature_cache.pkl` |
| Is data persisted to disk? | Yes — `feature_cache.pkl` |
| Which lock must be held? | `_feature_lock` during reads/writes of feature_cache |
| Must tests bypass the cache? | **Yes** |

---

## Block 6 — DataManager Attributes Used

| Attribute | Type | Can be `None`? | This spec reads / writes | Safe access pattern |
|---|---|---|---|---|
| `feature_cache` | `dict` | No | Reads + Writes | Wrap in `_feature_lock` |
| `cluster_df` | `pd.DataFrame` | No | Reads | Direct read of `firing_rate_hz` and `isi_violations_pct` columns |
| `vision_stas` | `LazySTADict` | Yes | Reads | `if self.vision_stas and vid in self.vision_stas:` |

---

## Block 7 — Acceptance Criteria

### AC1 — Robust Spatial Footprint Extraction
- **Setup:** A synthetic STA movie with a clear localized bright blob (e.g. 2D Gaussian) at the peak frame.
- **Action:** Call `extract_spatial_footprint(sta_data, peak_idx, grid_size=32)`.
- **Expected:**
  - Returns a flattened `1024`-element vector.
  - Zero-mean and unit-variance normalized.
  - Correctly crops the active footprint using the 3× MAD threshold.
- **Test type:** Unit

### AC2 — Graceful Fallback for Low SNR
- **Setup:** A purely uniform noise STA movie.
- **Action:** Call `extract_spatial_footprint(sta_data, peak_idx, grid_size=32)`.
- **Expected:** Returns `None`. `get_physics_feature_matrix()` fills with a 1024-element zero vector. No crash.
- **Test type:** Unit

### AC3 — UMAP without Vision (Kilosort-Only Mode)
- **Setup:** Load a Kilosort dataset without any `.sta` or `.ei` files.
- **Action:** Select UMAP tab, ensure "Timecourse" and "Spatial RF" are disabled (unchecked/grayed out). Click "Run UMAP".
- **Expected:**
  - Feature matrix is successfully built using **ACG PCA** (3 components) and a Kilosort scalar block containing `firing_rate_hz` and `isi_violations_pct` (scaled via RobustScaler).
  - UMAP runs successfully and displays a valid 2D/3D clustering projection.
- **Test type:** Integration/Manual

### AC4 — Cache Migration Compatibility
- **Setup:** Load a legacy `feature_cache.pkl` from disk that contains `_computed: True` but has NO `spatial_footprint` key.
- **Action:** `load_persisted_caches()` runs on initialization.
- **Expected:**
  - Detects legacy schema.
  - Discards the cache to force recalculation of the high-fidelity spatial footprints.
- **Test type:** Unit

### AC5 — Interactive Feature Weights and Toggles UI
- **State to reproduce:**
  1. Open UMAP panel.
  2. Locate Row 3 containing toggles and weight sliders for [Timecourse], [ACG], and [Spatial RF].
- **Expected appearance:**
  - Sliders show current values (defaults: TC=2.0, ACG=1.5, Spatial=0.8).
  - Checkboxes enable/disable their respective blocks.
  - Click "Reset Defaults" restores original values.
- **Screenshot filenames:**
  - `tests/screenshots/ac5_umap_sliders_dark.png`
  - `tests/screenshots/ac5_umap_sliders_light.png`
- **Verified by:** `[ ]` Author `[ ]` Reviewer

---

## Block 8 — Regression Guard

| Prior fix | Files overlap | Regression test to run | When to run it |
|---|---|---|---|
| Physics cache warm-up freeze | `data_manager.py` | `test_lazy_sta_dict_cache_is_thread_safe` | Before opening PR |
| Physics cache double-load | `data_manager.py` | `test_standard_plot_cache_computes_same_cluster_once` | Before opening PR |

---

## Block 9 — Test Plan

### Unit tests

File: `tests/unit/test_spatial_footprint.py`

| Test function name | Fixture | What it asserts | Cache bypass needed? |
|---|---|---|---|
| `test_extract_spatial_footprint_gaussian` | `None` | Standard 32x32 output shape and value ranges | No |
| `test_extract_spatial_footprint_fallback` | `None` | Low SNR returns `None` safely | No |
| `test_extract_spatial_footprint_bw` | `None` | Single-channel (B/W) STA handles successfully | No |

File: `tests/unit/test_data_manager_spatial_pca.py`

| Test function name | Fixture | What it asserts | Cache bypass needed? |
|---|---|---|---|
| `test_feature_matrix_spatial_pca` | `cache_cleared_data_manager` | Dimensionality includes spatial PCA (9 columns total: 3 for Timecourse, 3 for ACG, 3 for Spatial RF) | Yes |
| `test_feature_matrix_kilosort_only` | `mock_dm` | Valid feature matrix output containing only ACG PCA and Kilosort scalars (FR, ISI) | Yes |
| `test_cache_migration_prunes_legacy` | `tmp_path` | Legacy feature cache is pruned on startup | Yes |

---

## Block 10 — Out of Scope

- Does **not** change standard STA receptive field visualizations in the `STAPanel`.
- Does **not** alter the population RF mosaic rendering algorithm.
- Does **not** introduce new dependencies outside `numpy`, `scipy`, and `scikit-learn` (already installed).
- Does **not** persist weight slider preferences between separate app sessions.
