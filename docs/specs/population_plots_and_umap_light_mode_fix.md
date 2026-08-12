# Specification: Population Plots & UMAP Light Mode Fixes

## Block 0 — Metadata

| Field | Value |
|---|---|
| **Date created** | 2026-05-29 |
| **Last updated** | 2026-05-29 |
| **Commit hash when spec was written** | `874b4c0` |
| **Branch** | `fix/population-plots-umap-lightmode` |
| **Author** | Antigravity (AI Core Developer) |
| **Spec status** | Ready for Dev |

---

## Block 1 — Problem Statement

There are two distinct user experience and functionality bugs remaining in the population-level views of RGCViewer:

### 1. Population average Timecourse and ACG plots do not update on single cell selection
* **Symptom:** When clicking on an individual cell in the Tree View, the **Population Receptive Field** mosaic updates immediately and highlights the cell. However, the **Population Average Timecourse** and **Population ACG** plots on the right-hand population context pane remain completely static (or blank/empty) until the user explicitly clicks back on the folder/group itself.
* **Root cause:** 
  - Selecting an individual cell triggers `MainWindow.update_cluster_views(cluster_id)`.
  - This method updates the population RF plot via `draw_population_rfs_plot(selected_cell_id=cluster_id)`.
  - However, neither `update_cluster_views()` nor the debounced feature-loading method `_draw_plots()` ever calls `callbacks.redraw_population_panels()`.
  - These two panels are only drawn inside `_process_folder_selection()`, which is triggered exclusively when the user selects a group/folder node in the Tree View.

### 2. Light mode styling is unpolished (especially for the UMAP projection and RF Background)
* **Symptom A (UMAP Spines & Grid):** In Light Mode, the UMAP 2D axes are rendered with a harsh, solid-black box border (default matplotlib spines) and completely lack axis labels. This looks generic, basic, and inconsistent with the rest of the application's clean design system.
* **Symptom B (RF Background Invisible):** In Light Mode, the background receptive fields are completely invisible. This occurs because they are plotted using `colors['border_subtle']` (which is `#DEE2E6` in light mode) with `alpha=0.15`. A light gray line with `15%` opacity on a solid `#FFFFFF` background is visually imperceptible.

---

## Block 2 — Affected Files

| File path | Function(s) modified | Change type | Touches DataManager? |
|---|---|---|---|
| `src/gui/main_window.py` | `_draw_plots()` | **Modify** — trigger population panel redraw when population pane is open | No |
| `src/gui/panels/umap_panel.py` | `update_plot()` | **Modify** — apply polished axis borders, grids, tick styles, and labels in both 2D and 3D modes | No |
| `src/gui/panels/population_panel.py` | `plot_population_rfs_background()` | **Modify** — dynamically adjust background RF colors/alpha based on theme mode | No |

---

## Block 3 — Proposed Solutions

### 1. Fixing the Population Timecourse and ACG updates

To ensure the population context plots are always alive and accurate when the population view is enabled:
* Modify the debounced Tier 2 plotting method `_draw_plots()` in `src/gui/main_window.py`.
* If `self.population_view_enabled` is active, retrieve the active population cluster IDs using the smart tree helper `self._get_pop_subset_ids()`.
* Delegate to `callbacks.redraw_population_panels(self, subset=subset)` to update the canvases.

Since `redraw_population_panels()` uses the highly optimized `_group_timecourse_cache` and `_group_acg_cache`, navigating between different cells within the same parent folder will hit the cache instantly (O(1) lookup), guaranteeing ultra-smooth, zero-lag tree navigation!

#### Proposed code change in `src/gui/main_window.py`:
```python
        # --- TIER 2: POPULATION PLOTS ---
        if self.population_view_enabled:
            # 1. RF Mosaic (existing rebuild logic)
            canvas = self.pop_mosaic_canvas
            can_hot_swap = (
                hasattr(canvas, '_pop_plot_state') and
                canvas._pop_plot_state.get('ax') in canvas.fig.axes
            )
            if not can_hot_swap:
                try:
                    draw_population_rfs_plot(
                        main_window=self,
                        selected_cell_id=cluster_id,
                        canvas=self.pop_mosaic_canvas
                    )
                except Exception as e:
                    logger.error(f"Tier 2 Pop Split rebuild failed: {e}")

            # 2. NEW: Average Timecourse & ACG Panels
            try:
                subset = self._get_pop_subset_ids()
                callbacks.redraw_population_panels(self, subset=subset)
            except Exception as e:
                logger.error(f"Failed to update population panels on cell selection: {e}")
```

---

### 2. Polishing Light Mode Visuals

#### A. UMAP Axes & Spine Styling
Add clean gridlines, custom axis labels, and subtle border spines to the UMAP plot inside `update_plot()` in `src/gui/panels/umap_panel.py`.

```python
        # In src/gui/panels/umap_panel.py inside update_plot() (2D Mode):
        if not self.is_3d:
            # Apply premium styling matching design system
            self.ax.set_xlabel("UMAP Dimension 1", color=colors['text_secondary'], fontsize=9)
            self.ax.set_ylabel("UMAP Dimension 2", color=colors['text_secondary'], fontsize=9)
            self.ax.tick_params(colors=colors['text_secondary'], labelsize=8)
            
            # Hide harsh top/right borders
            self.ax.spines["top"].set_visible(False)
            self.ax.spines["right"].set_visible(False)
            
            # Make remaining borders subtle
            self.ax.spines["left"].set_edgecolor(colors['border_subtle'])
            self.ax.spines["left"].set_linewidth(0.8)
            self.ax.spines["bottom"].set_edgecolor(colors['border_subtle'])
            self.ax.spines["bottom"].set_linewidth(0.8)
            
            # Enable soft, clean gridlines
            self.ax.grid(True, color=colors['border_subtle'], linestyle=':', alpha=0.5, zorder=0)
```

#### B. RF Background Visibility in Light Mode
Dynamically adjust the color and transparency of background ellipses in `plot_population_rfs_background()` in `src/gui/panels/population_panel.py` depending on the active theme (detected by examining the panel background color).

```python
    # In src/gui/panels/population_panel.py inside plot_population_rfs_background():
    # Detect if light mode is active (bg_panel is white/light)
    is_light = (colors.get('bg_panel', '').upper() == '#FFFFFF')
    
    # In light mode, default border_subtle is too light. Use text_tertiary or a medium gray, with higher opacity.
    bg_color = colors.get('text_tertiary', '#ADB5BD') if is_light else colors.get('border_subtle', '#2E3038')
    bg_alpha = 0.35 if is_light else 0.15

    # Build EllipseCollection using the adapted colors
    bg_coll = _build_ellipse_collection(
        bg_ellipses,
        edgecolor=bg_color,
        alpha=bg_alpha, 
        lw=0.75, 
        zorder=1
    )
```

---

## Block 4 — Verification Plan

### Manual Verification Steps
1. **Selection Sync:**
   - Turn on "Population" split view.
   - Click a folder (e.g. `Type_1`) -> Population Timecourse, ACG, and RF plots should render.
   - Click an individual cell *inside* `Type_1` -> The RF mosaic highlights the cell, and the Timecourse/ACG panels should **not** go blank; they must continue displaying the `Type_1` population traces seamlessly.
   - Double-click a cell in a different folder -> The population plots must immediately update to represent the new parent folder's population.

2. **Light Mode Styling Check:**
   - Switch the application to **Light Theme**.
   - Open **UMAP** tab. Run UMAP.
   - Check that the axes have labels ("UMAP Dimension 1 / 2"), subtle gray borders, and soft gridlines. The default black outline box must be gone.
   - Open **Population RF Mosaic**.
   - Confirm that the background receptive field ellipses are beautifully visible as soft gray contours rather than being completely transparent/invisible.
