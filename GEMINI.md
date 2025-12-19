This `GEMINI.md` is designed to be the "System Context" for your agent. It defines the **Tech Stack**, **Coding Standards**, **Critical Files**, and a **Strategic Execution Plan** to ensure the agent builds the feature safely without breaking existing logic.

Save this file as `GEMINI.md` in your project root.

---

# GEMINI.md - Axolotl STA Split-View Feature Context

## 1. Project Overview

**Application:** Axolotl (Neuroscience Analysis GUI for MEA Data)
**Framework:** Python 3.9+, PyQt5, NumPy, Matplotlib, PyQtGraph.
**Goal:** Refactor the **STA (Spike-Triggered Average) Panel** from a static 4-quadrant layout to a **Split-Pane "Focus + Context" View**.

* **Left Pane (Focus):** Individual Unit Analysis (RF Movie, Timecourse).
* **Right Pane (Context):** Population Analysis (Mosaic of all RF fits in the group, Aggregate Timecourse).

## 2. Coding Standards & Conventions

* **UI Framework:** Use `qtpy` wrappers (not raw `PyQt5`).
* **Plotting:** Use `gui.widgets.MplCanvas` for Matplotlib plots.
* **Data Structures:**
* `cluster_ids` are **0-indexed** (Kilosort convention).
* `vision_ids` are **1-indexed** (Vision convention). *Always convert before querying `DataManager`.*


* **Threading:** Heavy calculations must go to `gui.workers`. **Do not** block the main thread.
* **Error Handling:** Use `try/except` blocks in UI callbacks to prevent crashing. Log errors to `logger`.

## 3. Critical File Map

*Read these files to understand the current architecture. Do not modify files outside this list unless necessary.*

| File | Purpose |
| --- | --- |
| `gui/main_window.py` | **UI Skeleton.** Defines the layout `_setup_ui` and `on_tab_changed`. |
| `analysis/analysis_core.py` | **Math Brain.** Contains `plot_population_rfs`. Needs modification for subsets. |
| `gui/plotting.py` | **Rendering Logic.** Connects data to canvases. Needs `draw_sta_split_view`. |
| `gui/callbacks.py` | **Controller.** Handles `on_cluster_selection_changed` events. |
| `analysis/data_manager.py` | **Data Source.** Provides `vision_params`, `vision_stas`, and `cluster_df`. |

## 4. Strategic Implementation Plan

*Follow these phases in order to maintain stability.*

### Phase 1: The "Backend" (Math & Core Logic)

**Objective:** Update plotting functions to support "Subsets" (Groups) instead of just "All".

* **File:** `analysis/analysis_core.py`
* **Task:** Modify `plot_population_rfs(..., subset_cell_ids=None)`.
* If `subset_cell_ids` is provided, only draw ellipses for those IDs.
* Draw the `selected_cell_id` in **Cyan/Red** (Highlight).
* (Optional) Draw remaining cells in faint gray (alpha=0.1) for context.



### Phase 2: The "Skeleton" (GUI Layout)

**Objective:** Physically split the STA panel into two vertical columns.

* **File:** `gui/main_window.py`
* **Task:** Refactor `_setup_ui` (STA Panel Section).
* Remove the old 4-quadrant splitter.
* Create `self.sta_main_splitter` (Horizontal).
* **Left Widget:** Container for `rf_canvas` (Top) & `timecourse_canvas` (Bottom).
* **Right Widget:** Container for new `pop_mosaic_canvas` (Top) & `pop_dynamics_canvas` (Bottom).



### Phase 3: The "Controller" (Selection Logic)

**Objective:** Detect when a user selects a **Group** vs. a **Unit**.

* **File:** `gui/callbacks.py`
* **Task:** Update `on_cluster_selection_changed`.
1. Get `selected_cluster_id`.
2. Query `data_manager.cluster_df` to find the `KSLabel` or `group` for this cluster.
3. Extract all `cluster_ids` belonging to that group.
4. Call the new plotter: `draw_sta_split_view(selected_id, group_ids)`.



### Phase 4: The "Bridge" (Wiring)

**Objective:** Connect the GUI signals to the new backend logic.

* **File:** `gui/plotting.py`
* **Task:** Create `draw_sta_split_view(main_window, unit_id, group_ids)`.
* **Left Pane:** Call existing `draw_sta_plot(unit_id)`.
* **Right Pane:** Call updated `analysis_core.plot_population_rfs(..., subset_cell_ids=group_ids)`.



## 5. Verification Checklist

* [ ] **Unit Mode:** Clicking a unit shows its specific RF on the left.
* [ ] **Population Mode:** The right pane automatically updates to show *only* the ellipses of the unit's group.
* [ ] **Highlighting:** The selected unit is clearly visible (bold/colored) within the population mosaic on the right.
* [ ] **Stability:** Clicking between groups does not crash the app (handle empty groups/missing Vision data).