# RGCViewer Master Development Plan

## Project Vision

RGCViewer is a high-performance, PyQt/pyqtgraph-based GUI tailored for the analysis, curation, and visualization of Retinal Ganglion Cell (RGC) data. It supports dual pipelines: hybrid Kilosort + Vision datasets, and Standalone Vision-native datasets. [cite_start]This document outlines the strategic roadmap and active priorities for development, adhering strictly to Spec-Driven Development [SDD](cite: 9).

---

## Current Milestone: v1.1 - Reliability, Vision Native, & Accessibility

[cite_start]*The focus of this milestone is solidifying the existing feature set, expanding accessibility for new users, stripping out obsolete UI clutter, and ensuring our caching and test suites handle edge cases robustly[cite: 9].*

### 1. Active Priorities (In Order of Execution)

**Priority 1: Standalone Vision Integration**

* [cite_start]**Goal:** Solidify the ability to load purely Vision-sorted datasets without requiring any Kilosort data[cite: 9, 21].
* [cite_start]**Spec:** `docs/specs/vision_standalone.md` [cite: 21]
* **Key Tasks:**
  * [cite_start]Resolve the bug causing only a random subset of cells to appear in UMAP[cite: 21].
  * [cite_start]Integrate array geometry and waveform templates from `.ei` and `.globals` files into the Standard Plots panel[cite: 21].
  * [cite_start]Ensure the `DataManager` correctly maps Vision IDs to `cluster_id` without metadata crashes[cite: 21].
  * [cite_start]Implement graceful fallbacks for missing `.sta` or `.ei` files[cite: 21].

---

### 2. Completed Priorities (v1.1)

**Light Mode Polish & UI Cleanup**

* [cite_start]**Summary:** Implemented a centralized "first-principles" color theme architecture and removed legacy UI elements[cite: 10, 19].
* [cite_start]**Outcome:** Created `src/gui/theme.py` to manage semantic color roles[cite: 19]. [cite_start]Toggling Light Mode now correctly updates stylesheets and plots (including the Population Panel) for universal legibility[cite: 19]. [cite_start]The "Good" view toggle and "Refine Selected Cluster" buttons have been removed, and the cluster table now supports click-to-sort functionality[cite: 19].

**Physics Cache Fix & Loading Optimization**

* [cite_start]**Goal:** Fix the cache invalidation bug where physics precomputations were double-loading[cite: 9, 11].
* [cite_start]**Outcome:** `DataManager` now performs a non-blocking check for existing `.pkl` caches on initialization[cite: 9, 11]. [cite_start]Background workers log failures without hanging the queue, and cells without Vision STA data are marked as `_computed` with safe defaults to ensure UMAP readiness reaches 100%[cite: 9, 11].

**UMAP Selection Fix**

* [cite_start]**Goal:** Resolve tool conflicts between the Lasso and Rectangle selector tools[cite: 9, 13].
* [cite_start]**Outcome:** Ensured only one selector is active at a time and fixed selection logic to correctly identify clusters in both 2D and 3D projections[cite: 13].

**Autocorrelation (ACG) Fix**

* [cite_start]**Goal:** Ensure ACG computations utilize the entire recording duration rather than just the first two minutes[cite: 9, 12].
* [cite_start]**Outcome:** Refactored the algorithm to include spikes across the full session while maintaining memory efficiency and avoiding dense arrays[cite: 12].

**Population Mosaic UI Refinements**

* **Goal:** Resolve UI responsiveness bugs, relocate "Show IDs" to population panel, remove gridlines, and implement mouse-wheel zoom / click-drag panning on the RF mosaic.
* **Outcome:** The `Show IDs` checkbox now instantly invalidates the caching to trigger redraws while preserving the selected group subset context. Gridlines were removed for a cleaner aesthetic, and interactive zoom and panning have been added via `NavigationToolbar2QT`. Verified with integration tests on real-world datasets.

**HDBSCAN Clustering in UMAP Panel**

* **Goal:** Replace fixed-k K-Means with density-based HDBSCAN as the default UMAP clustering method, while retaining K-Means as a fallback.
* **Spec:** `docs/specs/hdbscan-clustering.md`
* **Outcome:** Unified `ClusterWorker` runs HDBSCAN or K-Means in a background thread. New clustering UI (method combo, parameter spinbox, Run Clustering button). Noise points (`-1`) render grey and are excluded from auto-group. Seven unit/integration tests added. Dependency: `hdbscan>=0.8.0`.

---

## Testing & Infrastructure Initiatives

* **Real Data Integration:** Shift away from pure mock testing where possible. [cite_start]Utilize existing datasets in `/mnt/lab/Array-data/` to test edge cases in real-world scenarios[cite: 9].
* [cite_start]**Strict Cache Invalidation in CI:** Ensure the test suite explicitly clears or bypasses the `.pkl` cache when verifying math/physics logic to prevent false positives[cite: 9].
* [cite_start]**Advanced Performance Testing:** Implement stress tests for UI responsiveness, visual regression testing for standard plots using `pytest-mpl`, and memory leak protection[cite: 14].
