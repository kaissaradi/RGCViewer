# RGCViewer Master Development Plan

## Project Vision

RGCViewer is a high-performance, PyQt/pyqtgraph-based GUI tailored for the analysis, curation, and visualization of Retinal Ganglion Cell (RGC) data. It supports dual pipelines: hybrid Kilosort + Vision datasets, and Standalone Vision-native datasets. This document outlines the strategic roadmap and active priorities for development, adhering strictly to Spec-Driven Development (SDD).

---

## Current Milestone: v1.1 - Reliability, Vision Native, & Accessibility

*The focus of this milestone is solidifying the existing feature set, expanding accessibility for new users, stripping out obsolete UI clutter, and ensuring our caching and test suites handle edge cases robustly.*

### 1. Active Priorities (In Order of Execution)

**Priority 1: Light Mode Polish & UI Cleanup**

* **Goal:** Ensure a flawless Light Mode, fix core table interactions, and remove dead UI weight.
* **Spec needed:** `docs/specs/ui_ux_polish.md`
* **Architecture Change:** Extract hardcoded colors (`DARK_COLORS` / `LIGHT_COLORS`) out of `main_window.py` and into a dedicated `src/gui/theme.py` file.
* **Fixes:** Enable click-to-sort on Table View; remove "Good" view toggle and "Refine Selected Cluster" button.

**Priority 2: Standalone Vision Integration**

* **Goal:** Solidify the new feature allowing users to load purely Vision files without requiring Kilosort data.
* **Spec needed:** `docs/specs/vision_standalone_integration.md`

---

### 2. Completed Priorities (v1.1)

**Physics Cache Fix & Loading Optimization**

* **Goal:** Fix the cache invalidation bug where physics precomputations are double-loading.
* **Spec:** `docs/specs/physics_optimization.md`
* **Focus:** Audit `DataManager.get_cell_physics` to ensure disk cache is properly checked *and* utilized before the background queue starts FFT/signal calculations.
* **Notes:**
  * Standard plot cache is restored on `DataManager` init (with corrupt-cache fallback).
  * Background standard-plot worker logs per-cluster failures and still emits progress signals so the queue can't hang.
  * `get_cell_physics()` always writes a `_computed` cache entry (safe defaults when Vision STA is missing) so UMAP readiness can reach `N/N`.

**UMAP Selection Fix**

* **Goal:** Resolve conflicts between the Lasso and Rectangle selector tools.
* **Spec:** `docs/specs/umap_selection_fix.md`

**Autocorrelation (ACG) Fix**

* **Goal:** Ensure ACG computations utilize the entire recording duration.
* **Spec:** `docs/specs/autocorrelation_fix.md`

---

## Testing & Infrastructure Initiatives

* **Real Data Integration:** Shift away from pure mock testing where possible. Utilize the existing datasets in `/mnt/lab/Array-data/` to test edge cases in real-world scenarios.
* **Strict Cache Invalidation in CI:** Ensure the test suite explicitly clears or bypasses the `.pkl` cache when verifying math/physics logic to prevent false positives.
