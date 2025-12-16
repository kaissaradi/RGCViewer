# Refactoring Plan

This document outlines the plan to address critical bugs and improve usability in the RGCViewer application.

## Priority 1: Fix Core Application Crash (`TypeError`) and Unblock Plot Updates (Completed)

*   **Files:** `gui/panels/ei_panel.py`, `gui/main_window.py`
*   **Problem:** A `TypeError` in `_load_and_draw_ks_ei` was crashing the main plotting function (`_draw_plots`), preventing subsequent plot updates, including the STA plot.
*   **Impact:** The crash was the root cause of the "STA not auto-updating" issue.
*   **Fix:** Modified `_load_and_draw_ks_ei` to correctly handle the `cluster_ids` list. This prevents the crash, allowing `_draw_plots` to complete and the STA plot to auto-update.

## Priority 2: Fix EI Plotting Failure (`ValueError`) (Completed)

*   **File:** `gui/panels/ei_panel.py`
*   **Function:** `compute_ei_map`
*   **Problem:** A `ValueError` occurred during interpolation if the EI data's channel count mismatched the position data, crashing the multi-cluster plotting process.
*   **Fix:** Added validation to `compute_ei_map` to detect the mismatch and return `None`. The calling function, `_load_and_draw_vision_ei`, was enhanced to handle this `None` return, allowing it to gracefully skip plotting a single corrupted EI without stopping the entire process.

## Priority 3: Improve EI Fallback Clarity (Completed)

*   **File:** `gui/panels/ei_panel.py`
*   **Problem:** When Vision EI data is not available for a selected cluster, the application correctly falls back to displaying the Kilosort-based spatial analysis. However, this fallback is silent, causing confusion.
*   **Fix:** Enhanced the user feedback by modifying the Kilosort EI plot title. When the fallback occurs, the title now explicitly states **"(Vision EI not found)"**. This makes the GUI's behavior transparent.
