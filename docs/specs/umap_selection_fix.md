# Specification: UMAP Selection Fix

## Objective
Fix the UMAP panel selection logic where multiple selectors might be active simultaneously, and ensure the Rectangle Selection tool correctly identifies clusters within its bounds.

## User Story
"As a user, when I select the Rectangle Tool in the UMAP panel and drag a box over clusters, I want a single dialog to appear asking if I want to group the clusters I just drew a box around, and I want all clusters within that box to be included."

## Technical Constraints
- The `UMAPPanel` uses `LassoSelector` and `RectangleSelector` from `matplotlib.widgets`.
- Only one selector should be active at any given time.
- The `on_select` logic should correctly handle vertices provided by either tool.
- Selection must work in both 2D and 3D (projected) modes.

## Bug Analysis
- `on_processing_finished` hardcodes a `LassoSelector` into `self.selector`.
- `update_plot` calls `update_selector`, which creates another selector in `self.current_selector`.
- This leads to "double selection" behavior or conflicting tools.
- Selection logic in `on_select_rect` might have issues if `xdata`/`ydata` are `None`.
