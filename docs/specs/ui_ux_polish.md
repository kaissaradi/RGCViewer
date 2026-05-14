Specification: Light Mode Polish & UI Cleanup

## Metadata

* **Status:** Draft
* **Target Release:** v1.1
* **Primary Developer/Agent:** [Agent Name]

## Objective

Centralize the application's color theme architecture to resolve legibility issues in Light Mode (e.g., white/grey text disappearing). Additionally, clean up deprecated UI elements to streamline the user experience and enable click-to-sort functionality on the cluster table.

## User Story

"As a researcher using the application in a brightly lit lab, I want to switch to Light Mode and read all text and plots clearly. Additionally, I want to sort my Kilosort clusters by clicking the table headers and not be distracted by dead, non-functional UI buttons."

## Acceptance Criteria (Definition of Done)

* **AC1:** A new file `src/gui/theme.py` is created, containing all hardcoded color definitions previously housed in `main_window.py` (e.g., `DARK_COLORS`, `LIGHT_COLORS`).
* **AC2:** Toggling Light Mode successfully updates the application's stylesheet and pyqtgraph configurations so that no text is unreadable (e.g., no white text on light backgrounds).
* **AC3:** Clicking a column header in the cluster Table View successfully sorts the table by that column (ascending/descending toggle).
* **AC4:** The "Good" view toggle button and the "Refine Selected Cluster" button are completely removed from the UI and codebase.
* **AC5 (Visual Check):** Launch the GUI, load a dataset, toggle Light Mode, and verify the main canvas, UMAP panel, and table text are clearly legible.

## Architecture & Technical Constraints

* **Files Modified:** * `src/gui/main_window.py` (Remove hardcoded colors, remove dead buttons, import theme).
  * `src/gui/theme.py` (New file for theme state and color dictionaries).
  * `src/gui/widgets.py` (Add sorting capability to `PandasModel` or `CustomTableView`).
* **Data Contracts:** `theme.py` must expose a clear dictionary or class structure that other panels can query when restyling.
* **UI/Threading Rules:** Theme switching must execute synchronously on the main thread and trigger a refresh of visible pyqtgraph canvases.

## Test Plan (TDD Requirements)

* **Unit/Integration:** Add a test using `qtbot` to instantiate `MainWindow`, toggle the theme state, and assert that the main background color property correctly updates to the Light Mode hex value.
* **Integration:** Add a test verifying that invoking the sort method on the table header correctly reorders the underlying Pandas dataframe.

## Out Of Scope

* Redesigning the physical layout or sizing of the UI panels.
* Implementing new UI themes beyond Light/Dark.
