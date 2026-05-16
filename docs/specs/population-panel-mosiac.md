Here is the updated spec with your refinements applied:

---

# Specification: Population Receptive Field Panel Polish

## Metadata

* **Status:** Completed
* **Target Release:** v1.1
* **Primary Developer/Agent:** [Agent Name]

## Objective

Improve the legibility and interactivity of the Population Receptive Field (RF) mosaic. Ellipses must be more visible against both Light and Dark themes without becoming visually heavy, the "Show IDs" control must live in the population panel where it is contextually relevant, and the RF canvas must support zoom and pan so users can inspect dense spatial layouts.

## User Story

"As a researcher reviewing population receptive fields, I want to zoom into a crowded region of the RF mosaic, clearly see every ellipse without thick distracting borders, and toggle modest cell ID labels from the population panel itself so that I can inspect spatial patterns without switching contexts or hunting for controls."

## Acceptance Criteria (Definition of Done)

* **AC1 (Ellipse Visibility — Subtle Contrast):** In both Light and Dark modes:
  * Background (non-target) ellipses: `alpha = 0.12–0.18`, `linewidth = 0.6–0.9 px`. Color may shift *only slightly* toward the theme's `text_secondary` to increase contrast against the panel background.
  * Target (in-subset) ellipses: `alpha = 0.45–0.65`, `linewidth = 0.9–1.2 px`. No "bold" or heavy outlines.
  * Selected-cell highlight ellipse fill: `alpha = 0.35–0.50`. Edge remains `plot_highlight` at `linewidth = 1.5–2.0 px`.
  * No other plot elements change color or opacity.
* **AC2 (Show IDs Relocation):** The `QCheckBox("Show IDs")` is removed from `StandardPlotsPanel`'s top control bar. A new `QCheckBox("Show IDs")` is added to the population panel's top control bar (`pop_ctrl_layout`).
* **AC3 (Show IDs Wiring — Modest Labels):** When the population-panel "Show IDs" checkbox is checked, cell ID labels appear at the center of each target ellipse in the RF mosaic using a font size of **7–8 px** and `color: text_secondary` (not `text_primary`, to avoid visual shouting). When unchecked, labels are hidden. The `StandardPlotsPanel` template grid **no longer** displays channel-number labels. The checkbox state is read from `main_window.pop_show_ids_checkbox`.
* **AC4 (Zoom & Pan):** The population RF mosaic canvas (`pop_mosaic_canvas`) supports mouse-wheel zoom and click-drag pan via a compact `NavigationToolbar2QT` attached to the mosaic canvas widget. The toolbar may be rendered icon-only or with hidden text labels, but zoom/pan/home/save tools must be active.
* **AC5 (No Gridlines):** The population RF mosaic plot must not display any background gridlines to maintain a clean aesthetic.
* **AC6 (Context-Aware Checkbox Update):** Toggling the "Show IDs" checkbox must instantly update the plot using the currently selected group subset, rather than resetting to show all cells in the recording when a folder is selected instead of a single cell.
* **AC7 (No Side Effects):** `StandardPlotsPanel` no longer references `show_ids_checkbox` or draws channel labels. All other standard plots (ACG, ISI, FR, Template Grid layout, channel modes) remain unchanged. `population_panel.py` functions other than `plot_population_rfs_background`, `draw_population_rfs_plot`, and `_update_highlight_patch` are untouched.

## Architecture & Technical Constraints

* **Files Modified:**
  * `src/gui/main_window.py` — Add `pop_show_ids_checkbox` to `pop_ctrl_layout`; attach navigation toolbar to `pop_mosaic_canvas`.
  * `src/gui/panels/population_panel.py` — Tune ellipse `alpha`/`lw` values within the ranges above; read ID-toggle state from `main_window.pop_show_ids_checkbox`; ensure canvas supports interactive zoom/pan.
  * `src/gui/panels/standard_plots_panel.py` — Remove `self.show_ids_checkbox` and all logic that adds `pg.TextItem` channel labels to the template grid.
* **Data Contracts:** No changes to `DataManager`, `cluster_df`, or any analysis outputs. Pure UI/UX change.
* **UI/Threading Rules:** All changes execute on the main Qt thread. Matplotlib toolbar initialization must occur during `_setup_ui()` after `pop_mosaic_canvas` is instantiated.

## Test Plan (TDD Requirements)

* **Unit:** Add `test_population_rf_alpha_values()` that mocks `vision_params`, calls `plot_population_rfs_background`, and asserts every added `Ellipse` patch has `alpha` between `0.12` and `0.65` and `linewidth` between `0.6` and `1.2`.
* **Unit:** Add `test_show_ids_checkbox_moved()` that instantiates `MainWindow` and asserts `StandardPlotsPanel` has no `show_ids_checkbox` attribute while `main_window` has `pop_show_ids_checkbox`.
* **Integration:** Add a `qtbot` test that toggles `pop_show_ids_checkbox`, triggers a population RF redraw, and verifies the number of text artists on the canvas matches the number of target cells when checked and is zero when unchecked. Also assert font size ≤ 8.
* **Visual Check:** Launch the GUI, load a dataset with Vision params, open the population split view, and:
  1. Confirm ellipses are clearly visible but not visually heavy in both Light and Dark mode.
  2. Check and uncheck the new "Show IDs" checkbox in the population panel; confirm cell IDs appear/disappear on the RF mosaic at a modest size and do **not** appear in the Standard Plots template grid.
  3. Use mouse wheel to zoom in on a dense region and click-drag to pan; confirm the view updates smoothly.

## Out Of Scope

* Redesigning the ellipse color palette beyond the subtle contrast tweak permitted in AC1.
* Adding zoom/pan to the timecourse or ACG population panels.
* Any changes to the single-cell STA/RF panels.
* Persisting the "Show IDs" checkbox state to disk.
