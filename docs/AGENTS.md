# AI Developer Rulebook, Architecture, & Philosophy

Welcome to the RGCViewer repository. You are an AI acting as a core developer on this project. This project strictly follows **Spec-Driven Development (SDD)** and **Test-Driven Development (TDD)**.

Read this document in its entirety before modifying any code.

---

## 1. Guiding UX Philosophy

RGCViewer is a tool for scientists who need to sift through thousands of cells rapidly.

* **Snappy & Quick:** The main thread must *never* be blocked. UI interactions must feel instantaneous.
* **Separation of Concerns:** Data panels are completely isolated. The `UMAPPanel` knows nothing about the `STAPanel`. They only communicate by reacting to cluster selection changes via the `DataManager`.
* **Debouncing is Law:** Wait for scrolling to stop before firing heavy UI workers.

---

## 2. The Architecture & Data Pipeline

* **`src/analysis/vision_integration.py` & Kilosort logic:** Parses raw files.
* **`src/analysis/data_manager.py`:** The single source of truth.
* **`src/gui/workers/`:** All FFTs, file I/O, and signal processing via `QThread`.
* **`src/gui/panels/`:** Thin UI layers using `pyqtgraph`. No heavy lifting.
* **`src/gui/theme.py`:** (Target Architecture) Centralized repository for UI styling, Light/Dark mode toggles, and layout constants.

---

## 3. Environment, Data, & Execution Rules

* **The Conda Environment:** Every terminal command you run MUST use the `rgcviewer` conda environment (e.g., `conda run -n rgcviewer python -m pytest tests/`).
* **Real Testing Data Locations:** Do not generate massive fake datasets. Use the existing real datasets on the machine for integration testing:
  * *Raw Litke Data:* `/mnt/lab/Array-data/raw/20260506A/data009`
  * *Sorted/Vision Data:* `/mnt/lab/Array-data/sorted/20260506A/chunk10/kilosort2.5`
* **CACHE INVALIDATION RULE (CRITICAL):** When writing tests to verify calculation logic (like ACG, ISI, or Physics), you MUST use a temporary directory or explicitly clear/bypass the `.pkl` cache. Otherwise, the test will pass by loading old data without executing your new code.

---

## 4. Git Protocol & Branching

Use descriptive, atomic commit messages and standard branch prefixes:

* `feat/` (e.g., `feat/light-mode-theme`)
* `fix/` (e.g., `fix/umap-selection-bug`)
* `test/` (e.g., `test/physics-cache-invalidation`)
* `chore/` (e.g., `chore/linting-cleanup`)

---

## 5. Multi-Agent Concurrency Rules

If you are operating as part of a multi-agent team, you MUST obey these isolation rules to prevent merge conflicts and race conditions:

1. **One Spec = One Branch = One Agent:** You may only work on the specific `SPEC.md` assigned to you. You must create and stay on your dedicated branch.
2. **Domain Isolation:** If Agent A is working on `UMAPPanel`, Agent B working on `STAPanel` must not touch `UMAPPanel`.
3. **The `DataManager` Bottleneck:** If your spec requires modifying `data_manager.py`, you must execute careful `git pull --rebase main` commands frequently to ensure you are not overwriting another agent's state logic.

---

## 6. The Prime Directives (Workflow)

1. **Never write implementation code without reading the corresponding spec in `docs/specs/` first.**
2. **Always write the failing test in `tests/` before modifying `src/`.**
3. **Qt Threading Rule:** Never update UI elements directly from a background thread. You must use Qt Signals.
4. **Plotting Rule:** Always use `pyqtgraph` for dynamic UI data.
