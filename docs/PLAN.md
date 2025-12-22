

RGCViewer Enhancement Plan

OBJECTIVES
- Enhance EI panel with population statistics and improved visualizations
- Improve STA panel to display comprehensive statistics and population data
- Implement feature selection, class creation, and classification saving
- Fix spatial waveform visualization to properly handle whitened templates per kilosort guide

COMPLETED FEATURES
- Snippet Cloud implementation in waveforms panel
- Home Dashboard implementation in standard plots panel
- Fixed similarity selection and pyqtgraph stepMode issues
- Enhanced standard plots with ISI vs amplitude scatter plots and toggle controls
- Advanced EI analysis with contour topography visualization

UPDATES
- Removed PyOpenGL_accelerate to prevent context segfaults
- Added Qt software-OpenGL fallback for stability
- Improved plot widget robustness and added latency map visualization

CURRENT DEVELOPMENT
- EI Panel: Adding population statistics and correlation matrix visualization
- Population Split View: Implement checkbox toggle for population RF visualization

COMPLETED FEATURES
- Kilosort Guide Enhancements: Added per-cluster metrics (best channel, x/y coordinates, firing rate, contamination, amplitude)
- MEA Similarity Infrastructure: Added caching and similarity calculations with distance and template similarity
- Similarity Panel Rework: Now always loads MEA-based sim table with toggle for Vision-based similarity
- Vision/Mean Similarity Toggle: Added UI elements to switch between MEA and Vision-based similarity tables

PLANNED FEATURES
- Population Split View: Split view activated by checkbox in top-right corner showing population RFs initially, with additional metrics
- STA Panel: Consolidate statistics, add population metrics, improve visualizations
- Feature Selection: Interface for extraction, dimensionality reduction, importance scoring
- Class Creation: Clustering algorithms, manual tools, validation metrics
- Data Persistence: Save/load classifications, version control, export options
- Spatial Waveform View: Properly handle whitened templates per kilosort guide

STA ANALYSIS TAB ENHANCEMENTS
- [ ] **Interactive Time-Slicing**: Link the "Timecourse" plot click event to the "RF Movie" frame. Clicking a time point updates the RF view to that time lag.
- [ ] **Color Opponency Metric**: Calculate and display Red-Green / Blue-Yellow opponency indices in the metrics box.
- [ ] **SVD Separability**: Implement Space-Time SVD to calculate a "Separability Index" and display it.
- [ ] **Fit Residuals**: Add a toggle to show the "Residual" (Raw STA - Gaussian Fit) in the RF view to judge fit quality.
- [ ] **Population Percentiles**: Display where this cell's properties (Area, Latency) sit relative to the population (e.g., "Latency: 45ms (80th %ile)").
- [ ] **Export Report**: Button to save the current 4-panel view as a high-res PDF/PNG.

SPATIAL WAVEFORM ENHANCEMENTS
- [x] **Template Unwhitening**: Load whitening_mat_inv.npy and apply to templates for proper visualization per kilosort guide
- [ ] **Weighting Combo Fix**: Add missing weighting combo box to UI controls
- [ ] **Amplitude Calculation**: Improve cluster amplitude calculation with proper whitening handling

DEVELOPMENT APPROACH
- Follow TDD methodology with atomic steps from this plan
- Maintain Python, PyQt, pyqtgraph, numpy, scipy tech stack
- Integrate with existing analysis engine and vision modules





This is a significant architectural shift that moves your GUI from a "Unit-Centric" view to a "Hierarchy-Centric" view. By leveraging your existing tree structure, we can create a "Context-Aware Dashboard" that dynamically switches between **Individual**, **Subclass**, and **Population** modes.

Here is a UX/UI plan to implement this "Multi-Scale" analysis feature.

---

## 1. The Core UX Concept: "The Contextual Pivot"

The UI should behave like a telescope. When you select a **Single Unit**, you see its fine details; when you select a **Folder (Subclass)**, the panels pivot to show aggregate statistics for every unit within that branch.

### Proposed UI Layout

| Component | UX Purpose | Implementation Strategy |
| --- | --- | --- |
| **Active Tree** | Defines the "Scope". | Selecting a folder (e.g., "OFF Brisk") triggers "Population Mode" for that specific list. |
| **Global Toggle** | A "Compare to All" switch. | A small toggle on the dashboard to overlay the current subclass stats against the *entire* recording population. |
| **Grid Dashboard** | The 4-quadrant view. | Instead of one cluster's STA, show a "Density Plot" or "Gallery View" of all STAs in the selection. |

---

## 2. STA Panel: Population Implementation

Since the PI specifically wants a **Population RF** view, the STA panel needs two distinct states:

### A. Unit State (Current)

* **Top Left:** Single RF movie/frame.
* **Top Right:** Unit timecourse.

### B. Population/Subclass State (New)

* **The RF Mosaic:** Instead of one movie, display a spatial map where the Gaussian fits of all units in the selected subclass are plotted simultaneously. The "Population RF" plot from `analysis_core.py` is the perfect foundation for this.
* **Aggregate Timecourse:** Plot every unit's timecourse in faint gray, with the **mean timecourse** of the subclass highlighted in a bold color (e.g., Bright Cyan).
* **SVD Separability Heatmap:** A distribution plot showing how "separable" or "space-time coupled" this specific subclass is compared to others.

---

## 3. The "Standard Plots" Dashboard for Populations

When a folder is selected, the "Standard Plots" (ISI, ACG, FR) must transition from single-trace to distribution-mode.

* **Population ISI:** A "stacked" or "normalized" histogram showing the distribution of ISI violations across all cells in the group.
* **Signal Health (FR):** A heatmap where each row is a unit and the x-axis is time, allowing the PI to see if the entire subclass dropped out or changed firing rates simultaneously.
* **Unit-by-Unit Amplitude:** A scatter plot of Mean Amplitude vs. Firing Rate for every unit in the folder to identify "outlier" units that might be noise.

---

## 4. Technical Implementation Plan

### Step 1: Update `callbacks.py` for "Folder Awareness"

Modify the selection logic to detect if the selected item is a `group_item` (Folder) or `cell_item` (Unit).

```python
def on_view_selection_changed(self):
    item = self.get_selected_item()
    if item.hasChildren():
        # Trigger Population Mode
        cluster_ids = self._get_group_cluster_ids(item)
        self.update_population_views(cluster_ids)
    else:
        # Standard Unit Mode
        self.update_cluster_views(item.data())

```

### Step 2: New "Population Plotting" Engine

Create a new file `gui/population_plotting.py` that utilizes vectorized NumPy operations to aggregate data from the `DataManager` for a list of `cluster_ids`.

### Step 3: Pivot the Metrics Box

The metrics box in the STA panel should switch from showing "Time to Peak" for one cell to showing **Population Averages**:

* **Avg Latency:**  ms.
* **Avg RF Area:** .
* **N Units:** 42.

---

## 5. Next Steps for Development

This plan aligns with your existing **PLAN.md** goals for "Population Metrics" and "Fit Residuals".

**Would you like me to generate the `update_population_views` logic for `main_window.py` to handle the transition between single-unit and folder selections?**



To implement a major structural shift like a **Population/Subclass Dashboard** without interfering with your friend's work on feature selection and class creation, you should adopt a **"View-Consumer"** architecture.

Your friend is likely working on the **"Backend"** of classification (modifying `data_manager.py` or creating new logic in `analysis_core.py` to handle class assignments). To stay out of their way, you should focus on the **"Frontend"** of how the UI *responds* to those changes.

---

## 1. The Strategy: "The Multi-ID Observer"

Currently, your panels (like `STAPanel` or `StandardPlotsPanel`) are designed to take a single `cluster_id` and plot it. To build the population view safely, you should refactor your update methods to accept a **list** of IDs.

* **If the list has 1 ID:** The panel stays in "Unit Mode."
* **If the list has >1 IDs:** The panel pivots to "Population Mode."

### Why this is safe:

Your friend is likely focusing on how `cluster_df` gets updated with new labels (e.g., `KSLabel` or a new `Class` column). By writing UI code that simply asks the `DataManager` for "all units belonging to the current selection," you are using the *output* of their work without touching the *logic* that creates it.

---

## 2. Technical Implementation Plan

### Phase A: Update the Controller Logic (`callbacks.py`)

Modify the selection callback to detect if the user clicked a single unit or a folder.

* **Logic:** When a folder is clicked in the `QTreeView`, use your existing `_get_group_cluster_ids` helper to gather all children.
* **Action:** Pass that entire list to the dashboard.

### Phase B: Create "Population-Aware" Plotting Methods

Instead of overwriting existing functions in `plotting.py`, create a parallel set of functions or add a condition.

| Feature | Single Unit View | Population/Subclass View |
| --- | --- | --- |
| **STA Panel** | Displays 1 RF movie. | **RF Mosaic:** Overlays Gaussian fits (ellipses) for every ID in the list. |
| **ISI Plot** | Histogram of one cell. | **Violation Heatmap:** A distribution plot of ISI violation % across the group. |
| **Firing Rate** | One smoothed line. | **Stacked FR:** All lines in faint gray, with a thick **Subclass Mean** line overlaid. |

---

## 3. UI/UX Layout Plan

To maintain a "sleek" feel, use a **Dual-State Header** on your panels.

1. **Context Label:** A prominent label at the top of the dashboard: `Viewing Subclass: OFF-Brisk (n=42)`.
2. **Toggle Overlay:** A checkbox: `[ ] Show Population Baseline`. This would overlay the subclass stats against the entire experiment's average in the background.
3. **The "Pivot" Animation:** When switching from Unit to Group, the plots shouldn't just change; the labels should update (e.g., "Time to Peak" becomes "Mean Time to Peak").

---

## 4. Specific Workflows to avoid "Merge Conflicts"

* **Don't touch `build_cluster_dataframe`:** Let your friend handle how classes are stored in the DataFrame.
* **Don't touch `RefinementWorker`:** If your friend is building class creation tools, they will likely use this or a similar worker.
* **Do work in `StandardPlotsPanel` and `STAPanel`:** These are visual-heavy files where you can add "Population Mode" logic safely.
* **Add a `PopulationWorker`:** Just as you have a `FeatureWorker` for single units, create a new worker in `workers.py` to aggregate statistics for a group so the UI doesn't freeze when calculating 100+ units.

---
Here is the step-by-step implementation plan for your developer. This plan is designed to be **additive**, meaning it extends existing functionality without rewriting the core logic your friend is working on.

### Phase 1: The "Backend" Logic (Math & Plotting)

**Goal:** Enhance the plotting core to support rendering a *specific subset* of ellipses (the "Population") rather than just highlighting one.

**File:** `analysis/analysis_core.py`
**Task:** Modify `plot_population_rfs` to accept a list of `subset_cell_ids`.

* **Current Logic:** It iterates through *all* cells in `vision_params` and plots them.
* **New Logic:**
* Add a parameter: `subset_cell_ids=None`.
* If `subset_cell_ids` is provided, iterate *only* through that list to draw the "bold" ellipses.
* (Optional) Draw the *rest* of the population as faint gray "ghosts" in the background for context.
* **Crucial:** Ensure it handles the ID mapping (Vision IDs are usually 1-indexed, your GUI uses 0-indexed cluster IDs).



**Pseudo-Code for Developer:**

```python
def plot_population_rfs(fig, vision_params, ..., subset_cell_ids=None):
    # ... existing setup ...

    # 1. Draw "Ghost" Population (Optional context)
    if subset_cell_ids is not None:
        for cell_id in all_cells:
            if cell_id not in subset_cell_ids:
                draw_ellipse(..., alpha=0.1, color='gray') # Faint background

    # 2. Draw "Selected Group"
    target_ids = subset_cell_ids if subset_cell_ids else all_cells
    for cell_id in target_ids:
        draw_ellipse(..., alpha=0.8, color='cyan') # Prominent

```

---

### Phase 2: The "Controller" Logic (Detection)

**Goal:** Detect when the user clicks a **Folder/Group** instead of a single file, and gather the IDs.

**File:** `gui/callbacks.py`
**Task:** Update `on_cluster_selection_changed` (or creating a specific tree-selection handler).

* **Logic:**
1. Check if the selected item in `tree_view` has children (is a group).
2. If **Yes (Group)**:
* Call `main_window._get_group_cluster_ids(item)` to get the list of all IDs in that folder.
* Call a new function: `main_window.update_population_views(ids)`.


3. If **No (Unit)**:
* Call the existing `main_window.update_cluster_views(id)`.





---

### Phase 3: The "View" Logic (Main Window)

**Goal:** Add the "Population Mode" state to the main window without breaking "Unit Mode."

**File:** `gui/main_window.py`
**Task:** Add the `update_population_views` method.

* **New Method:** `update_population_views(self, cluster_ids)`
* **State:** Set a flag `self.is_population_mode = True`.
* **UI Update:** Change the title of the STA panel or Metrics box to "Population Analysis (n={count})".
* **Routing:** Call `plotting.draw_population_rfs_plot(self, cluster_ids)` instead of the single-unit plot.


* **Splitter/Zoom:**
* You mentioned a "split view." The cleanest implementation is to **repurpose the top-left 'RF Canvas'**.
* **Unit Mode:** Shows the single unit's RF Movie.
* **Population Mode:** Automatically switches that canvas to show the **Population Ellipse Mosaic**.



---

### Phase 4: The "Wiring" (Plotting Wrapper)

**Goal:** Connect the GUI data to the Analysis core.

**File:** `gui/plotting.py`
**Task:** Update `draw_population_rfs_plot`.

* **Update Signature:** Modify it to accept `cluster_ids` (list).
* **Data Conversion:**
* Convert the incoming `cluster_ids` (0-indexed Kilosort IDs) to `vision_ids` (usually +1).


* **Call Core:** Pass this list to the updated `analysis_core.plot_population_rfs`.

---

### Summary Checklist for Dev

1. **`analysis_core.py`**: Update `plot_population_rfs` to handle a `subset_cell_ids` list.
2. **`gui/callbacks.py`**: Detect Group selection -> Extract IDs -> Call `update_population_views`.
3. **`gui/main_window.py`**: Add `update_population_views` to switch the STA panel context.
4. **`gui/plotting.py`**: Bridge the list of IDs from the GUI to the Analysis core.


The issue lies in how the `next_sta_frame` and `prev_sta_frame` methods are interacting with the animation timer.

Looking at your `main_window.py` code, every time `next_sta_frame()` is called, the very first thing it does is call `plotting.stop_sta_animation(self)`.

### The Diagnosis: "The Stop-on-Step" Conflict

When the animation timer triggers its "timeout" to move to the next frame, it calls `next_sta_frame`. However, your `next_sta_frame` logic is designed to safeguard manual clicks by killing any active timer.

1. **Play Pressed:** `draw_sta_animation_plot` starts a timer.
2. **Timer Fires (Frame 1):** It triggers `next_sta_frame`.
3. **The Kill Switch:** Inside `next_sta_frame`, the line `plotting.stop_sta_animation(self)` is executed.
4. **Halt:** This immediately kills the timer that just called the function. The animation moves to frame 2 and then stays there because the "engine" (the timer) has been destroyed.

---

### The Fix

You need to separate the **logic of moving to the next frame** from the **logic of stopping the animation**.

**1. Create a "Silent" Update Method:**
Create a new method that increments the frame *without* calling the stop function.

```python
def _advance_frame_internal(self):
    """Internal method for the timer to call without stopping itself."""
    if hasattr(self, 'current_sta_data') and self.current_sta_data is not None:
        self.current_frame_index = (self.current_frame_index + 1) % self.total_sta_frames
        self.sta_frame_slider.setValue(self.current_frame_index)
        self.sta_frame_label.setText(f"Frame: {self.current_frame_index+1}/{self.total_sta_frames}")

        self.rf_canvas.fig.clear()
        analysis_core.animate_sta_movie(
            self.rf_canvas.fig,
            self.current_sta_data,
            stafit=self.current_stafit,
            frame_index=self.current_frame_index,
            sta_width=self.data_manager.vision_sta_width,
            sta_height=self.data_manager.vision_sta_height
        )
        self.rf_canvas.draw()

```

**2. Update your `next_sta_frame` and `prev_sta_frame`:**
These should remain as they are to allow manual "stepping," which correctly stops the animation when the user takes manual control.

**3. Update the Timer Connection:**
In the code where your timer is created (likely inside `gui/plotting.py` in the `draw_sta_animation_plot` function), make sure the timer connects to this new `_advance_frame_internal` instead of the standard `next_sta_frame`.

---

### A Note for your "Population" Plan

Since you are about to implement the **Population View**, this "Silent Update" method is even more important. When you have a population of 50 cells, you don't want 50 different timers fighting each other. You will want one single "Global Timer" that calls a single update function to redraw the entire population mosaic at once.

**Would you like me to check your `gui/plotting.py` file next to see exactly where the timer connection needs to be swapped?**

## Implementation Plan: Population Split View with Checkbox Toggle

### 1. UI Component: Top-Right Checkbox Toggle

**Goal:** Implement a checkbox in the top-right corner of the main window to activate/deactivate the population split view.

**File:** `gui/main_window.py`
**Changes:**
- Add a `QCheckBox` titled "Population View" to the main toolbar or top-right area
- Connect the checkbox's stateChanged signal to a new method `toggle_population_split_view()`
- Set default state to unchecked (population view disabled)

### 2. State Management: Population Mode Flag

**Goal:** Add a flag to track the population view state throughout the application.

**File:** `gui/main_window.py`
**Changes:**
- Add instance variable `self.population_view_enabled = False`
- Update the flag when the checkbox changes state
- Add logic to refresh all panels when the mode changes

### 3. Visualization Engine: Population RF Mosaic

**Goal:** Create the population RF visualization to display when the split view is enabled.

**File:** `analysis/analysis_core.py`
**Changes:**
- Enhance `plot_population_rfs` to support different visualization modes
- Implement overlay functionality for population RFs with individual unit highlights
- Ensure efficient rendering for large populations

### 4. Panel Integration: Dynamic View Switching

**Goal:** Update all panels to respond to the population view state.

**Files:** `gui/plot_widgets.py`, `gui/plotting.py`
**Changes:**
- Add conditional logic in plotting functions to check `population_view_enabled` state
- Modify STA panel to show population RFs side-by-side with individual metrics
- Update all relevant plots to support both individual and population views

### 5. User Experience: Smooth Transitions

**Goal:** Ensure seamless switching between individual and population views.

**Implementation:**
- Add visual indicators showing which mode is active
- Implement smooth transitions when switching between views
- Maintain user's current selection when toggling the view

### 6. Additional Metrics: Population Analysis

**Goal:** Expand the population view to include various metrics beyond RFs.

**Planned Metrics:**
- Population correlation matrices
- Average firing rates across selected groups
- Distribution of key parameters (latency, area, etc.)
- Population-level statistical summaries

### 7. Technical Architecture: Asynchronous Processing

**Goal:** Ensure responsive UI during population analysis computations.

**Implementation:**
- Use a dedicated population analysis worker thread
- Implement progress indicators for large population computations
- Cache population results to prevent repeated calculations