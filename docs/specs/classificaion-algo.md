# SPEC.md — High-D Manifold Clustering, Dynamic Features & Artifact Pre-Filtering

## Block 0 — Metadata

| Field | Value |
|---|---|
| **Date created** | 2026-06-09 |
| **Last updated** | 2026-06-09 |
| **Branch** | `feat/umap-dynamic-clustering` |
| **Author** | Kais |
| **Spec status** | READY FOR DEV (Strict Mode) |

---

## Block 1 — Engineering Objective

**Problem:** The current UMAP classification pipeline has four fundamental flaws:

1. **Clustering on the UMAP embedding.** `ClusterWorker` (in `umap_panel.py:47–82`) runs HDBSCAN/K-Means on the 2D UMAP coordinates. UMAP is a non-linear projection that distorts inter-point distances — clustering its output is mathematically unsound.
2. **Narrow feature vector.** `get_physics_feature_matrix()` (in `data_manager.py:1660–1743`) produces only 8 dimensions: `tc_pca(3) + acg_pca(3) + rf_area + ellipticity`. Firing rate, ISI violations, and polarity are absent. The UMAP panel's color combo lists 9 modes but only 3 are populated in the metadata — the rest (`Polarity`, `Firing Rate`, `ISI Violations`, `Color Opponency`) are dead code.
3. **No artifact pre-filtering.** Kilosort noise artifacts (flat STAs, massive RFs) enter the PCA, stretching the feature space and destroying biological cluster resolution.
4. **No temporal peak alignment.** PCA wastes components modeling phase shifts between cells instead of waveform shape differences.

**Solution:** Implement a strict sequential pipeline:

1. **Pre-Filter (The Bouncer):** Reject cells with flatline STAs or massive RF areas before any math runs.
2. **Feature Extraction:** Extract raw temporal STAs, ACGs, and scalar features (firing rate, ISI violations, time-to-peak, rf_area, ellipticity).
3. **Align & Transform:** Peak-align temporal STAs, PCA-reduce temporal and ACG blocks, RobustScale scalars.
4. **Weight & Assemble:** Apply user-defined weights per feature block, `hstack` into the High-D matrix.
5. **Classify:** Run HDBSCAN exclusively on the High-D matrix (never on the UMAP embedding).
6. **Visualize:** Run UMAP purely for 2D/3D scatter coordinates, coloring by HDBSCAN labels. Discarded artifacts are omitted from the plot entirely.

---

## Block 2 — Architectural & Law Contracts

### LAW 1: Vision ID Offset (CRITICAL)

Any iteration over `cluster_ids` to fetch physics/STA data MUST use the translation already implemented inside `data_manager.get_cell_physics()`. **Do not re-derive the offset.** The new `get_raw_feature_blocks()` method calls `get_cell_physics()` per cluster, which handles the offset internally.

```python
# Inside get_cell_physics() — already correct, do not duplicate:
vid = cluster_id if getattr(self, 'is_vision_only', False) else cluster_id + 1
```

### LAW 2: Threading & Main Loop

* **Constraint:** HDBSCAN, PCA, RobustScaler, and UMAP on matrices with N > 1000 rows will cause UI frame drops.
* **Enforcement:** All numpy/sklearn/umap/hdbscan computation MUST execute exclusively inside worker `.run()` methods on background QThreads. The main thread only reads results from emitted signals.
* **Workers live in `workers.py`** — the current placement of `UMAPWorker` and `ClusterWorker` inside `umap_panel.py` is an architecture violation. This spec moves them to `src/gui/workers/workers.py`.

### LAW 3: Cache Invalidation & Isolation

* **Constraint:** Do NOT mutate arrays returned by `self.feature_cache`.
* **Enforcement:** `get_raw_feature_blocks()` must `.copy()` all temporal and ACG arrays before returning them. Peak alignment uses `np.roll()` on the copies, never on cached originals.
* **Constraint:** The existing `get_physics_feature_matrix()` is NOT modified. `FeatureAnalysisWorker` in `feature_extraction.py` depends on it. The new method `get_raw_feature_blocks()` is additive.

---

## Block 3 — Mathematical & Edge Case Tolerances

| Scenario | Handled Behavior |
|---|---|
| **Pre-Filter: flat STA** | If `timecourse` is `None` OR `np.std(tc) < 1e-5`, add `cluster_id` to `discarded_ids`. Do NOT include in the High-D matrix. |
| **Pre-Filter: massive RF** | If `rf_area > filter_config['max_rf_area']`, add to `discarded_ids`. |
| **Weight set to `0.0`** | That feature block is completely omitted — no PCA, no columns in the matrix. If ALL weights are 0.0, the "Run UMAP" button is disabled with a tooltip. |
| **Temporal alignment on flat STA** | Handled by pre-filter. If threshold is 0.0 and a flatline sneaks through, `peak_align_timecourse()` checks `np.std(tc) < 1e-5` and returns the array unchanged (no `np.roll`). |
| **Missing `firing_rate`** | If not in `standard_plot_cache` for a cell, impute `0.0` before scaling. |
| **Missing `isi_violations`** | If not in `standard_plot_cache` / `isi_cache`, impute `0.0`. |
| **All cells filtered out** | Worker emits an `error` signal with a user-friendly message. Panel shows status bar warning. No crash. |
| **< `min_cluster_size` cells pass filter** | Worker skips HDBSCAN, emits warning. UMAP still runs. All points colored as unclustered. |
| **PCA components > n_samples** | Clamp: `n_components = min(configured, n_samples - 1, n_features)`. |
| **Matrix row alignment** | The final `N × D` matrix has exactly `len(valid_ids)` rows. Row `i` maps to `valid_ids[i]`. This invariant is asserted in tests. |

---

## Block 4 — Affected Files & Strict Signatures

### Layer Separation Contract

| Layer | File | Responsibility |
|---|---|---|
| **Constants** | `src/analysis/constants.py` | Default thresholds, PCA component counts, feature weights |
| **Pure math** | `src/analysis/analysis_core.py` | `peak_align_timecourse()`, `apply_prefilter()`, `build_feature_matrix()` — no Qt, no I/O, no DataManager |
| **Data access** | `src/analysis/data_manager.py` | `get_raw_feature_blocks()` — calls `get_cell_physics()`, collects raw arrays, delegates filtering to `analysis_core` |
| **Workers** | `src/gui/workers/workers.py` | `UMAPWorker` (moved here), `ClusterWorker` (moved here) — orchestrate the pipeline on background threads |
| **UI** | `src/gui/panels/umap_panel.py` | Feature weight/filter controls, plot rendering, signal connections |

---

### 1. `src/analysis/constants.py` — [MODIFY]

Add clustering/feature constants:

```python
# --- Feature Matrix Defaults ---
TEMPORAL_PCA_COMPONENTS = 5
ACG_PCA_COMPONENTS = 3

# --- Pre-Filter Defaults ---
DEFAULT_MIN_STA_STD = 1e-5       # Below this, STA is considered flat noise
DEFAULT_MAX_RF_AREA = 300.0      # Above this, RF is artifact-scale

# --- HDBSCAN ---
DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE = 5

# --- Feature Weights ---
DEFAULT_WEIGHT_TEMPORAL = 3.0
DEFAULT_WEIGHT_ACG = 2.0
DEFAULT_WEIGHT_FIRING_RATE = 1.5
DEFAULT_WEIGHT_ISI_VIOLATIONS = 1.0
DEFAULT_WEIGHT_TIME_TO_PEAK = 1.0
DEFAULT_WEIGHT_RF_AREA = 1.0
DEFAULT_WEIGHT_ELLIPTICITY = 1.0
```

---

### 2. `src/analysis/analysis_core.py` — [MODIFY]

Add three pure functions. These have **no Qt dependency, no I/O, no DataManager reference** and are independently unit-testable.

#### `peak_align_timecourse(tc: np.ndarray) -> np.ndarray`

```python
def peak_align_timecourse(tc: np.ndarray) -> np.ndarray:
    """
    Shift timecourse so absolute peak is at center index.
    
    If the timecourse is flat (std < DEFAULT_MIN_STA_STD), returns a copy
    unchanged — np.roll is NOT applied.
    
    Args:
        tc: 1D array, the temporal STA trace for one cell.
        
    Returns:
        Aligned copy of tc. Original is never mutated.
    """
```

#### `apply_prefilter(physics_entries: dict, filter_config: dict) -> Tuple[list, list]`

```python
def apply_prefilter(
    physics_entries: dict,   # {cluster_id: physics_dict from get_cell_physics()}
    filter_config: dict,     # {'min_sta_std': float, 'max_rf_area': float}
) -> Tuple[list, list]:
    """
    Partition cluster IDs into valid and discarded based on filter thresholds.
    
    A cell is discarded if:
      - physics['timecourse'] is None
      - np.std(physics['timecourse']) < filter_config['min_sta_std']
      - physics['rf_area'] > filter_config['max_rf_area']
    
    Returns:
        (valid_ids, discarded_ids) — both sorted for deterministic ordering.
    """
```

#### `build_feature_matrix(raw_blocks: dict, feature_config: dict) -> Tuple[np.ndarray, list]`

```python
def build_feature_matrix(
    raw_blocks: dict,
    feature_config: dict,
) -> Tuple[np.ndarray, list]:
    """
    Transform raw feature arrays into a weighted, PCA-reduced feature matrix.
    
    Args:
        raw_blocks: {
            'temporal': np.ndarray (N, T) — raw timecourses, NOT yet aligned,
            'acg': np.ndarray (N, A) — raw ACG arrays,
            'scalars': pd.DataFrame with columns subset of:
                ['firing_rate', 'isi_violations', 'time_to_peak', 'rf_area', 'ellipticity']
        }
        feature_config: {
            'use_temporal': bool, 'w_temporal': float,
            'use_acg': bool, 'w_acg': float,
            'use_firing_rate': bool, 'w_firing_rate': float,
            'use_isi_violations': bool, 'w_isi_violations': float,
            'use_time_to_peak': bool, 'w_time_to_peak': float,
            'use_rf_area': bool, 'w_rf_area': float,
            'use_ellipticity': bool, 'w_ellipticity': float,
        }
        
    Pipeline:
        1. If use_temporal: peak_align each row, PCA to TEMPORAL_PCA_COMPONENTS, multiply by w_temporal
        2. If use_acg: PCA to ACG_PCA_COMPONENTS, multiply by w_acg
        3. For each enabled scalar: RobustScaler, multiply by weight
        4. hstack all enabled blocks
        
    Returns:
        (matrix: np.ndarray (N, D), column_labels: list[str])
        
    Raises:
        ValueError if all features are disabled (D would be 0).
    """
```

---

### 3. `src/analysis/data_manager.py` — [MODIFY] (Fragile Zone)

*Must `git fetch && git rebase origin/main` before touching this file.*

**The existing `get_physics_feature_matrix()` is NOT modified.** It continues to serve `FeatureAnalysisWorker` in `feature_extraction.py` unchanged.

#### Add `get_raw_feature_blocks(self, cluster_ids, filter_config) -> Tuple[dict, list, list]`

```python
def get_raw_feature_blocks(
    self,
    cluster_ids: list,
    filter_config: dict,
) -> Tuple[dict, list, list]:
    """
    Extract raw feature arrays for clustering, with pre-filtering.
    
    Steps:
        1. Call get_cell_physics(cid) for each cluster_id (Law 1 handled internally).
        2. Delegate to analysis_core.apply_prefilter() to partition valid/discarded.
        3. For valid cells, collect:
           - timecourse arrays (copied, not cached originals)
           - ACG arrays (copied)
           - Scalar values: firing_rate, isi_violations, time_to_peak, rf_area, ellipticity
        4. Pad/truncate temporal and ACG arrays to uniform length.
        5. Firing rate: sourced from standard_plot_cache[cid]['firing_rate'] if available, else 0.0
        6. ISI violations: sourced from standard_plot_cache[cid].get('isi_violations', 0.0)
    
    Returns:
        raw_blocks: {
            'temporal': np.ndarray (N, T),
            'acg': np.ndarray (N, A),
            'scalars': pd.DataFrame (N rows, columns = scalar feature names)
        }
        valid_ids: list[int] — cluster IDs that passed pre-filter
        discarded_ids: list[int] — cluster IDs rejected by pre-filter
        
    Thread safety:
        Acquires _feature_lock for cache reads. Safe to call from worker threads.
    """
```

---

### 4. `src/gui/workers/workers.py` — [MODIFY]

Move `UMAPWorker` and `ClusterWorker` from `umap_panel.py` to this file, alongside the existing workers.

#### `UMAPWorker` (moved from umap_panel.py, refactored)

```python
class UMAPWorker(QObject):
    """
    Background worker: raw feature extraction → build matrix → UMAP embedding.
    
    Signals:
        progress(str)           — status messages for the progress bar
        finished(              
            embedding,          — np.ndarray (N, 2 or 3)
            feature_matrix,     — np.ndarray (N, D) the high-D matrix for clustering
            valid_ids,          — list[int]
            discarded_ids,      — list[int]
            metadata_df         — pd.DataFrame with columns for all color modes
        )
        error(str)              — error message
    """
    
    def __init__(self, data_manager, cluster_ids, feature_config, filter_config,
                 n_components=2):
        ...
    
    def run(self):
        """
        1. dm.get_raw_feature_blocks(cluster_ids, filter_config)
           → raw_blocks, valid_ids, discarded_ids
        2. Guard: if len(valid_ids) < 2 → emit error, return
        3. analysis_core.build_feature_matrix(raw_blocks, feature_config)
           → matrix, col_labels
        4. umap.UMAP(n_neighbors=min(15, N-1), min_dist=0.1, 
                      metric='euclidean', n_components=self.n_components)
           → embedding
        5. Build metadata_df from raw_blocks['scalars'] + valid_ids
           Populate ALL color mode columns: firing_rate, isi_violations,
           time_to_peak, rf_area, ellipticity, polarity (from timecourse sign)
        6. self.finished.emit(embedding, matrix, valid_ids, discarded_ids, metadata_df)
        """
```

#### `ClusterWorker` (moved from umap_panel.py, refactored)

```python
class ClusterWorker(QObject):
    """
    Background worker: run HDBSCAN or K-Means on the HIGH-D feature matrix.
    
    KEY CHANGE: Previously clustered on the 2D UMAP embedding.
    Now clusters on the full feature matrix passed from UMAPWorker output.
    
    Signals:
        finished(labels: np.ndarray, method: str)
        error(str)
    """
    
    def __init__(self, feature_matrix, method='HDBSCAN', param=5):
        # param = min_cluster_size for HDBSCAN, n_clusters for K-Means
        ...
    
    def run(self):
        """
        - HDBSCAN: hdbscan.HDBSCAN(min_cluster_size=self.param).fit_predict(self.matrix)
        - K-Means: sklearn.cluster.KMeans(n_clusters=self.param).fit_predict(self.matrix)
        """
```

---

### 5. `src/gui/panels/umap_panel.py` — [MODIFY]

#### Removals

* **Delete** `UMAPWorker` class (lines 85–144) — moved to `workers.py`
* **Delete** `ClusterWorker` class (lines 47–82) — moved to `workers.py`
* **Delete** module-level `W_SHAPE`, `W_PATTERN`, `W_GEOMETRY` (lines 28–32) — replaced by `constants.py`
* **Delete** `extract_features_from_datamanager()` helper (lines 35–45) — no longer needed

#### Additions to `_build_control_panel()`

Add a collapsible "Feature Weights" section with the following controls:

| Control | Type | Default | Range |
|---|---|---|---|
| Temporal STA | `QCheckBox` + `QDoubleSpinBox` | ✓ enabled, weight `3.0` | 0.0 – 5.0, step 0.5 |
| ACG | `QCheckBox` + `QDoubleSpinBox` | ✓ enabled, weight `2.0` | 0.0 – 5.0, step 0.5 |
| Firing Rate | `QCheckBox` + `QDoubleSpinBox` | ✓ enabled, weight `1.5` | 0.0 – 5.0, step 0.5 |
| ISI Violations | `QCheckBox` + `QDoubleSpinBox` | ✓ enabled, weight `1.0` | 0.0 – 5.0, step 0.5 |
| Time to Peak | `QCheckBox` + `QDoubleSpinBox` | ✓ enabled, weight `1.0` | 0.0 – 5.0, step 0.5 |
| RF Area | `QCheckBox` + `QDoubleSpinBox` | ✓ enabled, weight `1.0` | 0.0 – 5.0, step 0.5 |
| Ellipticity | `QCheckBox` + `QDoubleSpinBox` | ✓ enabled, weight `1.0` | 0.0 – 5.0, step 0.5 |

Add a "Pre-Filter" section:

| Control | Type | Default | Range |
|---|---|---|---|
| Min STA Std Dev | `QDoubleSpinBox` | `1e-5` | 0.0 – 1.0, step 0.001 |
| Max RF Area | `QDoubleSpinBox` | `300.0` | 0.0 – 10000.0, step 50.0 |

Modify the HDBSCAN parameter spinner:

| Control | Type | Default | Range |
|---|---|---|---|
| Min Cluster Size | `QSpinBox` | `5` | 2 – 100, step 1 |

#### Modifications to `run_umap()` / `run_umap_3d()`

1. Read UI states into `feature_config` and `filter_config` dicts.
2. Validate: if ALL feature checkboxes are unchecked, show tooltip and return.
3. Create `UMAPWorker(dm, cluster_ids, feature_config, filter_config, n_components)`.
4. Connect signals: `finished` → `on_processing_finished`, `error` → status bar.

#### Modifications to `on_processing_finished()`

New signature receives `(embedding, feature_matrix, valid_ids, discarded_ids, metadata_df)`:
1. Store `self.feature_matrix = feature_matrix` for later use by `ClusterWorker`.
2. Store `self.valid_ids = valid_ids`, `self.discarded_ids = discarded_ids`.
3. Store `self.metadata_df = metadata_df`.
4. Show status: `"{len(valid_ids)} cells mapped, {len(discarded_ids)} filtered out"`.

#### Modifications to `run_clustering()`

1. Pass `self.feature_matrix` to `ClusterWorker`, NOT `self.embedding`.
2. Read `min_cluster_size` from the UI spinner.

#### Modifications to `update_plot()`

1. **Discarded cells:** Omitted from scatter entirely (no points plotted).
2. **HDBSCAN noise (label = -1):** Colored dark gray (`#404040`) with 50% opacity.
3. **Color modes:** All modes now functional using populated `metadata_df` columns.

---

## Block 5 — Acceptance Criteria

**AC1: Artifact Pre-Filtering (The Bouncer)**

* Create 3 mock clusters: C1 is biological (valid STA, rf_area=50). C2 has rf_area=500. C3 has flat STA (std < 1e-5).
* Call `get_raw_feature_blocks([C1, C2, C3], filter_config)`.
* **Validation:** `valid_ids == [C1]`. `discarded_ids` contains C2 and C3. `raw_blocks['temporal']` has 1 row.

**AC2: Zero-Weight Computation Bypass**

* Call `build_feature_matrix(raw_blocks, config)` with `use_acg=False`.
* **Validation:** Matrix column count is exactly `ACG_PCA_COMPONENTS` fewer than when `use_acg=True`. No ACG PCA is computed.

**AC3: Temporal Peak Alignment Fidelity**

* Create two synthetic timecourses with identical shape but shifted by 4 frames.
* Call `peak_align_timecourse()` on each.
* **Validation:** Both arrays peak at exactly index `len(tc) // 2`.

**AC4: Clustering on High-D (Not Embedding)**

* Run the full pipeline. Inspect `ClusterWorker.__init__` arguments.
* **Validation:** The matrix passed to `ClusterWorker` has dimensionality D > 2 (the high-D feature matrix, not the 2D/3D UMAP embedding).

**AC5: Background UI Responsiveness**

* Execute "Run UMAP" on >1,000 valid clusters.
* **Validation:** Main UI remains responsive — hold down-arrow in cluster list during computation, no frame drops.

**AC6: Edge Case — All Cells Filtered**

* Set `max_rf_area=0.0` so all cells are discarded.
* **Validation:** Worker emits error signal with user-friendly message. Panel shows warning. No crash.

**AC7: Edge Case — PCA Component Clamping**

* Pass 3 valid cells with `TEMPORAL_PCA_COMPONENTS=5`.
* **Validation:** PCA automatically clamps to `min(5, 3-1, n_features) = 2` components. No crash.

---

## Block 6 — Test Plan (Isolated & Pytest Native)

*All tests MUST use `mock_dm` (with `tmp_path`) to prevent polluting real data caches.*

### File: `tests/unit/test_dynamic_clustering.py`

Tests for `analysis_core` pure functions — no Qt, no DataManager.

| Test Name | Assertion Target | Notes |
|---|---|---|
| `test_peak_align_centers_peak` | Two shifted TCs both peak at `len//2` after alignment | Pure numpy |
| `test_peak_align_skips_flatline` | TC with `std < 1e-5` returned unchanged, no `np.roll` | Pure numpy |
| `test_prefilter_rejects_large_rf` | Cell with `rf_area=500` is in `discarded_ids` | Pure dict input |
| `test_prefilter_rejects_flat_sta` | Cell with `timecourse=None` is in `discarded_ids` | Pure dict input |
| `test_prefilter_passes_good_cell` | Normal cell is in `valid_ids` | Pure dict input |
| `test_build_matrix_zero_weight_omits_block` | `use_acg=False` → matrix width shrinks by `ACG_PCA_COMPONENTS` | Pure numpy |
| `test_build_matrix_all_disabled_raises` | All features disabled → `ValueError` | Pure numpy |
| `test_build_matrix_row_alignment` | `matrix[i]` corresponds to `valid_ids[i]` in input | Pure numpy |
| `test_pca_clamps_components` | 3 samples, 5 configured components → clamped to 2 | Pure numpy |
| `test_robust_scaler_handles_zero_imputation` | Missing `firing_rate` → imputed as 0.0, no crash | Pure numpy |

### File: `tests/unit/test_raw_feature_blocks.py`

Tests for `data_manager.get_raw_feature_blocks()` — uses `mock_dm`.

| Test Name | Assertion Target | Notes |
|---|---|---|
| `test_raw_blocks_vision_id_offset` | Hybrid mode: `get_cell_physics` called with correct IDs | Mock verification |
| `test_raw_blocks_copies_arrays` | Returned arrays are copies, not cache references | `assert arr is not cached_arr` |
| `test_raw_blocks_uniform_padding` | All TC rows same length, all ACG rows same length | Shape assertions |
| `test_raw_blocks_includes_firing_rate` | `scalars` DataFrame has `firing_rate` column | Column check |
| `test_raw_blocks_includes_isi_violations` | `scalars` DataFrame has `isi_violations` column | Column check |

### File: `tests/integration/test_umap_worker.py`

Tests for full worker pipeline — uses `qtbot` + `mock_dm`.

| Test Name | Assertion Target |
|---|---|
| `test_worker_emits_five_args` | `finished` signal emits `(embedding, matrix, valid_ids, discarded_ids, metadata_df)` |
| `test_worker_valid_ids_match_embedding_rows` | `len(embedding) == len(valid_ids)` |
| `test_worker_all_filtered_emits_error` | All cells fail filter → `error` signal emitted, `finished` NOT emitted |
| `test_worker_single_cell_emits_error` | Only 1 cell passes → `error` signal with descriptive message |
| `test_cluster_worker_receives_highd_matrix` | `ClusterWorker` receives matrix with `D > 2` columns |
| `test_cluster_worker_hdbscan_labels_shape` | `len(labels) == matrix.shape[0]` |

### Existing tests — must still pass:

```bash
conda run -n rgcviewer python -m pytest tests/ -v
```

Specifically verify no regressions in:
- `tests/unit/test_physics_cache_unified.py`
- `tests/unit/test_hdbscan_clustering.py`
- `tests/integration/test_umap_panel_clustering.py`
- `tests/integration/test_umap_selection.py`

---

## Block 7 — Out of Scope

* Does **not** modify `get_physics_feature_matrix()` — `FeatureAnalysisWorker` in `feature_extraction.py` depends on it.
* Does **not** modify the standard K-Means clustering in the main UI (which remains for manual low-D operations).
* Does **not** modify the `FWHM` computation logic inside `analysis_core.py`.
* Does **not** implement `Color Opponency` computation (future spec).
* Does **not** add caching of raw feature blocks (potential future optimization for iterative weight tuning).
