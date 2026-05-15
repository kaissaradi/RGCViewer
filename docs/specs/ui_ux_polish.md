Specification: Light Mode Polish & UI Cleanup

## Metadata

* **Status:** Ready for Dev
* **Target Release:** v1.1
* **Primary Developer/Agent:** [Agent Name]

## Objective

Implement a "first-principles" color theme architecture to centralize UI styling and resolve legibility issues in Light Mode. The new architecture must enforce semantic color roles (e.g., `text_primary`, `text_secondary`, `bg_panel`) across all Qt widgets, sidebars, tree views, and every data panel. Additionally, clean up deprecated UI elements to streamline the user experience and enable click-to-sort functionality on the cluster table.

## User Story

"As a researcher using the application in a brightly lit lab, I want to switch to Light Mode and read all text and plots clearly—including the sidebar, tree view, and every data panel (especially the Population Panel). Additionally, I want to sort my clusters by clicking the table headers and not be distracted by dead, non-functional UI buttons."

## Acceptance Criteria (Definition of Done)

* **AC1 (First-Principles Theming):** A new file `src/gui/theme.py` is created. It defines an abstract, semantic color dictionary (e.g., `text_primary`, `text_secondary`, `bg_main`, `bg_panel`, `plot_line`, `plot_scatter`) for both Dark and Light modes.
* **AC2 (Universal Legibility):** Toggling Light Mode successfully updates the global stylesheet, pyqtgraph configurations, and matplotlib figures.
    * *Requirement:* All text in the sidebar and tree view must invert correctly.
    * *Requirement:* All plots in the Standard Plots, EI Panel, UMAP Panel, and **Population Panel** must dynamically consume the theme colors. For the Population Panel specifically, the shadow traces, mean lines, and RF ellipses must maintain high contrast against the chosen background.
* **AC3 (Table Sorting):** Clicking a column header in the cluster Table View successfully sorts the table by that column (ascending/descending toggle).
* **AC4 (Cleanup):** The "Good" view toggle button and the "Refine Selected Cluster" button are completely removed from the UI and codebase.
* **AC5 (Visual Check):** Launch the GUI, load a dataset, and toggle Light Mode. Verify the sidebar, tree view, UMAP panel, standard plots, EI panel, and Population panel adapt to the light semantic palette without any hidden, low-contrast, or unreadable text.

## Architecture & Technical Constraints

* **Files Modified:** * `src/gui/main_window.py` (Remove hardcoded colors, remove dead buttons, implement global theme swap).
    * `src/gui/theme.py` (New file for semantic theme state and dictionaries).
    * `src/gui/widgets.py` (Add sorting capability to `PandasModel` or `CustomTableView`).
    * Plotting modules (e.g., `population_panel.py`, `standard_plots_panel.py`, `ei_panel.py`, `umap_panel.py`) must be updated to request colors via a central method like `main_window.get_current_colors()` instead of hardcoding hex values.
* **Data Contracts:** `theme.py` must expose a consistent dictionary structure so that adding new themes in the future requires zero changes to the panel modules.
* **UI/Threading Rules:** Theme switching must execute synchronously on the main thread and trigger a universal `.restyle()` or redraw event across all active pyqtgraph/matplotlib canvases.

## Test Plan (TDD Requirements)

* **Unit:** Add a test verifying `theme.py` exposes identical keys for both Light and Dark dictionaries to prevent `KeyError` crashes in the plotting logic.
* **Integration:** Add a test using `qtbot` to instantiate `MainWindow`, toggle the theme state, and assert that the application's base font color and the pyqtgraph global background setting correctly update.
* **Integration:** Add a test verifying that invoking the sort method on the table header correctly reorders the underlying Pandas dataframe.

## Out Of Scope

* Redesigning the physical layout or sizing of the UI panels.
* Implementing new UI themes beyond Light/Dark at this time.