# Specification: HDBSCAN Clustering in UMAP Panel

## Metadata

* **Status:** Completed
* **Target Release:** v1.1
* **Branch:** `feat/hdbscan-clustering` (merged to `main`)

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

## Implementation Summary

### Files Modified

* `src/gui/panels/umap_panel.py` — all changes isolated here
* `tests/unit/test_hdbscan_clustering.py` — unit tests (T1–T4)
* `tests/integration/test_umap_panel_clustering.py` — integration tests (T5–T7)
* `requirements.txt` — `hdbscan>=0.8.0`

### Key Components

* `ClusterWorker` — unified background worker for HDBSCAN and K-Means; emits
  `(labels, method_name)` on `finished`
* `HDBSCAN_AVAILABLE` — module-level import guard; disables HDBSCAN combo entry when false
* `run_clustering` / `on_cluster_finished` — QThread wiring; noise labels skipped in auto-group
* `update_plot` — HDBSCAN branch assigns `#888888` to label `-1`, `tab20` for clusters

---

## Test Plan

### Unit — `tests/unit/test_hdbscan_clustering.py`

* **T1:** HDBSCAN on 3-blob synthetic data → ≥ 2 non-noise clusters
* **T2:** K-Means → labels in `[0, k-1]`, no `-1`
* **T3:** Outliers → labeled `-1`
* **T4:** `HDBSCAN_AVAILABLE=False` patched → `error` signal, no crash

### Integration — `tests/integration/test_umap_panel_clustering.py`

* **T5:** HDBSCAN button click → `metadata_df['HDBSCAN']` populated, color combo updated
* **T6:** Method combo swap → spinbox prefix, range, default update
* **T7:** K-Means + auto-group regression

All tests pass via `conda run -n rgcviewer python -m pytest tests/unit/test_hdbscan_clustering.py tests/integration/test_umap_panel_clustering.py`.

---

## Out Of Scope

* Changing feature extraction weights or PCA components.
* Running HDBSCAN on the raw feature space instead of the UMAP embedding.
* Exposing `min_samples`, `cluster_selection_epsilon`, or `alpha` to the user.
* Any changes to `data_manager.py`, `analysis_core.py`, or any panel outside
  `umap_panel.py`.
* Soft clustering / membership probabilities.
