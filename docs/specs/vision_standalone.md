# Specification: Standalone Vision Integration & Bug Fixes

**Status (2026-08-12):** Partial. Missing `.sta` / `.params` no longer crash.
A 519-wide `.ei` next to a 512-row `.globals` loads; the EI plot may stay
blank (accepted). This spec is not the active work queue. See `docs/PLAN.md`.

## Metadata

* **Status:** Partial — not the active queue
* **Target Release:** v1.1
* **Primary Developer/Agent:** [Agent Name]

## Objective

Fully implement and stabilize the "Standalone Vision" data pipeline. The application must successfully load purely Vision-sorted datasets without requiring any Kilosort data, correctly populating all cells into the UMAP panel, and utilizing the `.ei` and `.globals` files to render array geometry and waveform templates.

## User Story

"As a researcher analyzing older or purely Vision-native datasets, I want to load a Vision directory and have the application behave exactly as it does with Kilosort data. I need to see all of my cells in the UMAP space, and I need the Standard Plots panel to correctly display the array geometry and waveform templates."

## Acceptance Criteria (Definition of Done)

* **AC1 (Data Loading):** The `DataManager` and `vision_integration.py` successfully parse `.params`, `.neurons`, `.sta`, `.ei`, and `.globals` files when a standalone Vision dataset is loaded.
* **AC2 (UMAP Population Bug Fix):** The bug causing only a random subset of cells (e.g., 12 cells) to appear in the UMAP is resolved. The UMAP algorithm must receive and process *all* valid Vision cell IDs.
* **AC3 (Array Geometry Integration):** The Standard Plots panel (and any associated spatial plots) successfully extracts and renders the array geometry and waveform templates using data derived from the `.ei` and `.globals` files.
* **AC4 (State Management):** The `DataManager` correctly sets its `is_vision_only` flag, mapping Vision IDs directly to `cluster_id` in the `cluster_df` table without crashing or requiring Kilosort-specific metadata.
* **AC5 (Graceful Fallback):** If `.sta` or `.ei` files are missing from the directory, the application logs a warning and falls back to safe defaults (e.g., empty canvases for those specific plots) rather than crashing the main thread.

## Architecture & Technical Constraints

* **Files Modified:** * `src/analysis/data_manager.py` (Complete the `is_vision_only` initialization path; fix the UMAP subset logic).
    * `src/analysis/vision_integration.py` (Ensure `visionloader` correctly requests `include_ei=True` and `include_globals=True`).
    * `src/gui/panels/standard_plots_panel.py` (Update array plotting logic to accept Vision-native geometry coordinates).
* **Data Contracts:** In Vision-only mode, the `cluster_df` must use Vision IDs as the primary index. The `channel_positions` array must be populated from the Vision `.globals` file.
* **UI/Threading Rules:** File parsing and UMAP dimensionality reduction must occur on background worker threads (`VisionLoadWorker` / `FeatureWorker`) to prevent UI freezing.

## Test Plan (TDD Requirements)

* **Unit:** Add a test in `test_vision_integration.py` that mocks a Vision directory containing the 5 required files. Assert that `DataManager` correctly maps `len(valid_cells)` to the `cluster_df`, proving no cells are dropped.
* **Integration:** Write a test asserting that when `is_vision_only=True`, calling `DataManager.get_standard_plot_data()` successfully returns array geometry coordinates derived from `.globals`.
* **Visual Check:** Load a known Vision-only dataset. Verify that the UMAP panel is fully populated and that clicking a cluster displays its template on the array geometry in the Standard Plots panel.

## Out Of Scope

* Modifying or writing data *back* to Vision `.params` or `.neurons` files (read-only for now).
* Integrating Kilosort-specific metrics (like Contamination Rate) into Vision-only tables.