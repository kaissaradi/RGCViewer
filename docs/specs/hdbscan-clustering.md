# Specification: HDBSCAN Clustering in UMAP Panel

## Metadata

* **Status:** Ready for Dev
* **Target Release:** v1.1
* **Primary Developer/Agent:** TBD
* **Branch:** `feat/hdbscan-clustering`

---

## Objective

Replace the fixed-k KMeans clustering in the UMAP panel with HDBSCAN as the default
method, while retaining KMeans as a fallback. KMeans' requirement to pre-specify `k`
makes it a poor fit for exploratory RGC type discovery — HDBSCAN discovers cluster
count from data density, handles noise natively, and is better suited to the
non-convex manifold structure that UMAP typically produces.

---

## User Story

"As a researcher, I want to run density-based clustering on my UMAP embedding without
guessing the number of cell types in advance, so that cluster boundaries reflect actual
data structure rather than an arbitrary `k`."

---

## Acceptance Criteria (Definition of Done)

* **AC1:** The clustering UI row contains a method dropdown (HDBSCAN / K-Means), a
  single parameter spinbox whose label and range update to match the selected method,
  and a single "Run Clustering" button. The old `k_spin` and `kmeans_btn` widgets are
  removed.

* **AC2:** Running HDBSCAN produces a `labels` array stored in
  `metadata_df['HDBSCAN']`. Points with label `-1` (noise) are rendered as grey dots
  (`#888888`) in the scatter plot and are excluded from any auto-group tree operation.

* **AC3:** Running KMeans still produces a `labels` array stored in
  `metadata_df['K-Means']` and behaves identically to the current implementation.

* **AC4:** The `color_combo` dropdown contains an "HDBSCAN" entry that colorizes
  points by `metadata_df['HDBSCAN']` using `tab20`. Noise points (`label == -1`) are
  overridden to grey regardless of colormap.

* **AC5:** `min_cluster_size` spinbox range is 2–200, default 15 when HDBSCAN is
  selected. When KMeans is selected, the same spinbox shows prefix `k=`, range 2–100,
  default 5. No other UI controls are added.

* **AC6:** HDBSCAN runs inside a renamed `ClusterWorker` (replacing `KMeansWorker`)
  in a background `QThread`. The main thread is never blocked.

* **AC7:** If `hdbscan` is not installed, the method dropdown disables the HDBSCAN
  entry and logs a warning. The app does not crash.

* **AC8 (Regression):** All existing KMeans behavior — auto-group tree, Show IDs
  dialog, `show_group_ids()` — continues to work unchanged when KMeans is selected.

---

## Architecture & Technical Constraints

### Files Modified

* `src/gui/panels/umap_panel.py` — all changes are isolated here.

### Widgets Removed

* `self.k_spin` (QSpinBox, range 2–20)
* `self.kmeans_btn` (QPushButton "Run K-Means")

### Widgets Added / Renamed

```
self.cluster_method_combo  QComboBox       ["HDBSCAN", "K-Means"]
self.cluster_param_spin    QSpinBox        label/range swaps on method change
self.cluster_btn           QPushButton     "Run Clustering"
```

`cluster_layout` row becomes:
```
[Label: "Clustering:"] [cluster_method_combo] [cluster_param_spin] [cluster_btn]
[auto_group_chk] [show_ids_btn] [project_3d_chk] [stretch]
```

### ClusterWorker (replaces KMeansWorker)

```python
class ClusterWorker(QObject):
    finished = Signal(object, str)  # (labels_array, method_name)
    error = Signal(str)

    def __init__(self, embedding, method, param):
        # method: "HDBSCAN" | "K-Means"
        # param:  min_cluster_size (HDBSCAN) | k (K-Means)
        ...

    def run(self):
        if self.method == "HDBSCAN":
            import hdbscan
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.param,
                min_samples=None,          # defaults to min_cluster_size
                cluster_selection_method='eom',
                core_dist_n_jobs=-1
            )
            labels = clusterer.fit_predict(self.embedding)
            self.finished.emit(labels, "HDBSCAN")
        else:
            # existing KMeans path unchanged
            ...
            self.finished.emit(labels, "K-Means")
```

### on_cluster_finished handler

Receives `(labels, method_name)`. Stores result in `metadata_df[method_name]`.
Sets `color_combo` to `method_name`. Calls `update_plot()`. If `auto_group_chk` is
checked, skips cells where `label == -1` before creating tree groups.

### update_plot noise override

In the `"HDBSCAN"` branch of `update_plot()`:

```python
elif mode == "HDBSCAN":
    if 'HDBSCAN' in self.metadata_df:
        raw_labels = self.metadata_df['HDBSCAN'].values
        # Build per-point color array; noise → grey
        unique_non_noise = np.unique(raw_labels[raw_labels >= 0])
        n_types = max(len(unique_non_noise), 1)
        cmap_fn = plt.cm.get_cmap('tab20', n_types)
        color_array = []
        for lbl in raw_labels:
            if lbl == -1:
                color_array.append('#888888')
            else:
                idx = np.searchsorted(unique_non_noise, lbl)
                color_array.append(cmap_fn(idx % n_types))
        c = color_array
        is_discrete = True
```

Pass `c` as a list of RGBA/hex values directly to `scatter(..., c=c)`. No colorbar
for discrete modes (consistent with existing KSLabel / K-Means behavior).

### show_group_ids compatibility

`show_group_ids()` already checks `mode not in ["KSLabel", "K-Means", "Polarity"]`
and returns early. Extend this guard to also allow `"HDBSCAN"`. The groupby logic is
identical; noise cells (label -1) will naturally form a group named `-1` in the
dialog — this is acceptable and informative.

### Threading rules

* Identical to the existing `KMeansWorker` pattern: `QThread` + `moveToThread`.
* Worker refs renamed: `self.cluster_worker`, `self.cluster_worker_thread`.
* `_reset_workers()` updated to clean up the new refs.

### Dependency guard

```python
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logger.warning("hdbscan not installed; HDBSCAN clustering disabled")
```

On `__init__`, if `not HDBSCAN_AVAILABLE`, set
`self.cluster_method_combo.model().item(0).setEnabled(False)` (index 0 = HDBSCAN)
and default selection to K-Means.

---

## Test Plan

### Unit — `tests/unit/test_hdbscan_clustering.py`

* **T1:** `ClusterWorker` with method=`"HDBSCAN"`, param=15, synthetic 2D embedding
  (300 pts, 3 obvious blobs) → `labels` contains ≥ 2 unique non-noise clusters,
  no exception raised.
* **T2:** `ClusterWorker` with method=`"K-Means"`, param=5 → labels shape matches
  input, values in `[0, 4]`, no label `-1` present.
* **T3:** Noise point handling — embed with one obvious outlier cluster; confirm at
  least one point gets label `-1` under tight `min_cluster_size`.
* **T4:** `ClusterWorker` with `HDBSCAN_AVAILABLE=False` patched → `error` signal
  emitted, no crash.

### Integration — `tests/integration/test_umap_panel_clustering.py`

* **T5 (qtbot):** Instantiate `UMAPPanel`, inject a synthetic embedding + metadata_df,
  select HDBSCAN in `cluster_method_combo`, click `cluster_btn` → `metadata_df`
  contains `'HDBSCAN'` column after worker finishes.
* **T6 (qtbot):** Switch method combo K-Means → HDBSCAN → verify `cluster_param_spin`
  prefix, range, and default value update correctly.
* **T7 (regression):** Run K-Means via the new UI path → `metadata_df['K-Means']`
  populated, auto-group tree still works.

### Screenshot Verification

1. Launch app, load any dataset, run UMAP 2D.
2. Select HDBSCAN, `min_cluster_size=15`, click "Run Clustering".
3. Verify: scatter recolors, noise points appear grey, status bar shows cluster count.
4. Open Show IDs dialog → confirm `-1` group is present and non-noise groups are
   numbered sequentially from 0.
5. Switch `color_combo` back to KSLabel → HDBSCAN coloring clears without error.

---

## Out Of Scope

* Changing feature extraction weights or PCA components.
* Running HDBSCAN on the raw feature space instead of the UMAP embedding.
* Exposing `min_samples`, `cluster_selection_epsilon`, or `alpha` to the user.
* Any changes to `data_manager.py`, `analysis_core.py`, or any panel outside
  `umap_panel.py`.
* Soft clustering / membership probabilities.