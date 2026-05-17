# RGCViewer Master Development Plan

## Project Vision

RGCViewer is a high-performance, PyQt/pyqtgraph-based GUI tailored for the analysis, curation, and visualization of Retinal Ganglion Cell (RGC) data. It supports dual pipelines: hybrid Kilosort + Vision datasets, and Standalone Vision-native datasets. This document outlines the strategic roadmap and active priorities for development, adhering strictly to Spec-Driven Development [SDD].

---

## Current Milestone: v1.1 - Reliability, Vision Native, & Accessibility

*The focus of this milestone is solidifying the existing feature set, expanding accessibility for new users, stripping out obsolete UI clutter, and ensuring our caching and test suites handle edge cases robustly.*

### 1. Active Priorities (In Order of Execution)

**Priority 1: Standalone Vision Integration**

- **Goal:** Solidify the ability to load purely Vision-sorted datasets without requiring any Kilosort data.
- **Spec:** `docs/specs/vision_standalone.md`
- **Key Tasks:**
  - Resolve the bug causing only a random subset of cells to appear in UMAP.
  - Integrate array geometry and waveform templates from `.ei` and `.globals` files into the Standard Plots panel.
  - Ensure the `DataManager` correctly maps Vision IDs to `cluster_id` without metadata crashes.
  - Implement graceful fallbacks for missing `.sta` or `.ei` files.

---

### 2. Completed Priorities (v1.1)

**Light Mode Polish & UI Cleanup**

- **Summary:** Implemented a centralized "first-principles" color theme architecture and removed legacy UI elements.
- **Outcome:** Created `src/gui/theme.py` to manage semantic color roles. Toggling Light Mode now correctly updates stylesheets and plots (including the Population Panel) for universal legibility. The "Good" view toggle and "Refine Selected Cluster" buttons have been removed, and the cluster table now supports click-to-sort functionality.

**UX/UI Polish – UMAP Layout, Sidebar Search & Tree Branch Styling**

- **Summary:** Fixed a classic Qt geometry bug causing UMAP toolbar overlap on first render, added a persistent live search bar to the cluster sidebar (filtering both Tree and Table views), and replaced the default branch indicators with clean inline SVG triangles.
- **Outcome:** UMAP panel now renders correctly on first visit without requiring a tab switch. Users can type a cluster ID or label to instantly filter the left panel (case‑insensitive substring). The tree view uses modern `▶`/`▼` arrows that are theme‑aware (light/dark mode). `Ctrl+F` focuses the search bar. All changes are purely cosmetic with zero data or analysis logic modifications.

**Physics Cache Fix & Loading Optimization**

- **Goal:** Fix the cache invalidation bug where physics precomputations were double-loading.
- **Outcome:** `DataManager` now performs a non-blocking check for existing `.pkl` caches on initialization. Background workers log failures without hanging the queue, and cells without Vision STA data are marked as `_computed` with safe defaults to ensure UMAP readiness reaches 100%.

**UMAP Selection Fix**

- **Goal:** Resolve tool conflicts between the Lasso and Rectangle selector tools.
- **Outcome:** Ensured only one selector is active at a time and fixed selection logic to correctly identify clusters in both 2D and 3D projections.

**Autocorrelation (ACG) Fix**

- **Goal:** Ensure ACG computations utilize the entire recording duration rather than just the first two minutes.
- **Outcome:** Refactored the algorithm to include spikes across the full session while maintaining memory efficiency and avoiding dense arrays.

**Population Mosaic UI Refinements**

- **Goal:** Resolve UI responsiveness bugs, relocate "Show IDs" to population panel, remove gridlines, and implement mouse-wheel zoom / click-drag panning on the RF mosaic.
- **Outcome:** The `Show IDs` checkbox now instantly invalidates the caching to trigger redraws while preserving the selected group subset context. Gridlines were removed for a cleaner aesthetic, and interactive zoom and panning have been added via `NavigationToolbar2QT`. Verified with integration tests on real-world datasets.

**HDBSCAN Clustering in UMAP Panel**

- **Goal:** Replace fixed-k K-Means with density-based HDBSCAN as the default UMAP clustering method, while retaining K-Means as a fallback.
- **Spec:** `docs/specs/hdbscan-clustering.md`
- **Outcome:** Unified `ClusterWorker` runs HDBSCAN or K-Means in a background thread. New clustering UI (method combo, parameter spinbox, Run Clustering button). Noise points (`-1`) render grey and are excluded from auto-group. Seven unit/integration tests added. Dependency: `hdbscan>=0.8.0`.

---

## Testing & Infrastructure Initiatives

- **Real Data Integration:** Shift away from pure mock testing where possible. Utilize existing datasets in `/mnt/lab/Array-data/` to test edge cases in real-world scenarios.
- **Strict Cache Invalidation in CI:** Ensure the test suite explicitly clears or bypasses the `.pkl` cache when verifying math/physics logic to prevent false positives.
- **Advanced Performance Testing:** Implement stress tests for UI responsiveness, visual regression testing for standard plots using `pytest-mpl`, and memory leak protection.
