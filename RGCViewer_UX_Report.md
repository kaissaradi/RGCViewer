# RGCViewer — UI/UX Improvement Report
> Generated: 2026-05-26 | Based on: codebase review, AGENTS.md, PLAN.md, live screenshots, and user interview
> Status: Brainstorming / Pre-spec. Each section ends with a concrete implementation note.

---

## 0. Framing — The Curation Workflow

The actual user journey through a session is linear and fairly fixed:

```
Load dataset → (physics cache warms) → Standard plots → UMAP / feature extraction
→ Cluster → Scroll tree → Per-cell inspection (EI, STA, Waveforms) → Mark & export
```

Every improvement below is evaluated against this flow. Friction at any step compounds — a slow tree collapse after lassoing a group means the user loses their place at exactly the moment they've done something meaningful.

---

## 1. Standard Plots Panel

### Current state
Four-quadrant 2×2 layout: spatial template (top-left), autocorrelogram (top-right), ISI histogram (bottom-left), signal health / firing rate (bottom-right). Functional but reads as a raw data dump rather than a diagnostic tool.

### Problems identified

**1.1 No at-a-glance classification signal**
The ACG decay shape is the single most diagnostic feature for transient vs. sustained cells, but it requires the user to visually estimate the time constant on every scroll. There is no computed summary.

**1.2 ISI contamination is eyeballed**
The red refractory line exists but % contamination is not displayed numerically. The user has to mentally integrate the histogram area to the left of the line.

**1.3 Signal health plot has no legend**
Two traces (yellow = smoothed FR, green = instantaneous FR) are shown with no labels. A new user or collaborator has no idea what they represent.

**1.4 Spatial template panel proportionally oversized**
For transient/sustained discrimination the spatial template is the least informative of the four panels, yet it occupies 25% of the space. The ACG and signal health plots are doing most of the classification work.

### Proposed improvements

**ACG — fit and display decay constant τ**
After computing the ACG, fit a simple single-exponential decay to the falling edge and display τ as a badge on the plot (e.g. "τ = 12 ms"). Over time curators read τ rather than shape. Color-code the badge: fast (< 15 ms) = warm, slow (> 40 ms) = cool. This directly encodes transient vs. sustained.

**ISI — display % refractory violations as a colored badge**
The ISI violation % is already computed and cached in `isi_cache`. Surface it as a number on the histogram: green < 0.5%, yellow 0.5–2%, red > 2%. One glance tells you noise contamination without mental arithmetic.

**Signal health plot — add legend and clean up**
Label the two traces. Consider adding a thin horizontal line at the mean FR. The adaptation shape (is the cell adapting over the recording?) is actually useful for classification and currently invisible.

**Layout — make splitters unequal by default**
Give ACG and signal health 60% of vertical space, spatial template 40%. The user can still drag splitters. This is a one-line change to `setSizes()`.

### Implementation notes
- τ fitting: scipy `curve_fit` with `f(t) = A * exp(-t/τ) + C` on the ACG array, already available in `standard_plot_cache`. Add to `_compute_standard_plots()`, store in cache dict.
- ISI % badge: already available from `isi_cache[(cluster_id, refractory_ms)]`. Just render it.
- Both additions are Tier 2 (debounced) — safe to add without touching Tier 1.

---

## 2. UMAP Panel — Full Redesign

### Current state
Static matplotlib scatter. User runs embedding, colors by label, lassos to create groups, done. Feature weights are hardcoded module-level constants (`W_SHAPE = 2.0`, `W_PATTERN = 1.5`, `W_GEOMETRY = 1.0`). No hover. No live interaction. 3D mode exists but is also static matplotlib.

### Problems identified

**2.1 Matplotlib = no hover**
The single biggest missed opportunity. The whole point of a UMAP is to explore the embedding space by inspecting cells. Right now you have to mentally note a point's position, find it in the tree, and navigate to it. This kills exploration speed.

**2.2 Feature weights are invisible to the user**
The embedding shape depends entirely on W_SHAPE, W_PATTERN, W_GEOMETRY. A user studying EI-heavy separation (e.g. ON vs OFF midget) vs. ACG-heavy separation (transient vs. sustained) needs different weights. There is no way to adjust this without editing source code.

**2.3 Coloring is post-hoc, not exploratory**
Color options exist (Firing Rate, ISI Violations, RF Area, etc.) but they're applied after the fact to a static image. The relationship between embedding geometry and feature values is hard to read.

**2.4 No clustering without Vision/STA**
`get_physics_feature_matrix` depends on `feature_cache` which is populated by `get_cell_physics`. If Vision `.sta` is missing, STA-derived features (RF area, ellipticity, timecourse polarity) return NaN and the affected cells are dropped silently from the UMAP. For Kilosort-only datasets this is a blocker.

**2.5 Lasso → group creation collapses the entire tree**
`group_clusters_in_tree` in `callbacks.py` is doing a full model rebuild on group creation. The user loses their tree state at the exact moment they've created a meaningful group.

### Proposed improvements

**2A — Switch 2D scatter to pyqtgraph**
`pg.ScatterPlotItem` supports hover natively and renders faster than matplotlib for >500 points. On hover: show a mini popup with the cell's ACG thumbnail and STA (if available). On click: select that cluster in the tree and sync all other panels. This is the single highest-impact change in the whole application.

**2B — Feature weight sliders**
Three `QSlider` widgets for Shape, Pattern, Geometry weights in the control bar. Changing a slider marks the embedding as stale (grey out the plot, show "Re-run UMAP to apply"). Do not auto-rerun — UMAP takes time. But the user sees immediately that their weight change will affect the result.

**2C — Kilosort-only fallback feature set**
When Vision data is absent, build the feature matrix from Kilosort-only features: template peak-to-trough amplitude, template spatial spread (which channels fire), ACG τ (once §1 is implemented), mean firing rate, ISI violation %. This is enough to separate cell types meaningfully. Guard with `if self.vision_stas is None` in `get_physics_feature_matrix`.

**2D — Tree collapse fix**
`group_clusters_in_tree` should do a targeted subtree insert: find the parent node, insert the new group node beneath it, expand *only* that node. All sibling nodes stay in their current expanded/collapsed state. Feature extraction already handles this correctly — copy that pattern.

**2E — 3D: switch to Plotly via QWebEngineView or vispy**
Matplotlib 3D is not interactive in any meaningful way. Plotly scatter3d embedded in a `QWebEngineView` gives full orbit/zoom/hover in the same Qt window. vispy is lower latency but harder to set up. Either is dramatically better than the current static surface.

**2F — Color by continuous features with live update**
When the user changes the color combo, recolor the scatter *without* re-running UMAP. All feature values are already in `metadata_df`. This is already partially implemented but can be made instant with pyqtgraph's `setData(brush=...)`.

### Implementation notes
- pyqtgraph `ScatterPlotItem` hover: connect `sigHovered` signal, render ACG thumbnail in a `QLabel` overlay or a small `pg.PlotWidget` floating widget.
- Feature weight sliders: add to the second control row in `UMAPPanel.__init__`. Store weights as instance attributes, pass to `extract_features_from_datamanager` on run.
- Kilosort-only features: add a `_build_fallback_features(cluster_id)` method to `DataManager` that returns a dict with the KS-only feature subset.
- Tree collapse: audit `group_clusters_in_tree` in `callbacks.py`. Replace full model reset with `model.insertRow()` + `model.dataChanged.emit()` on the target parent index only.

---

## 3. EI Panel — Full Rebuild

### Current state
Left pane: stacked widget with three views — 2D heatmap (interpolated energy grid), 3D mountain plot (same data as surface), latency map (static dot scatter). Right pane: temporal waveform traces for 3 hardcoded top electrodes. The `ei_corr_dict` with three full 512×512 correlation matrices (full, space, power) is computed and pickled to disk by `DataManager` but **never surfaced in the UI at all**.

### Problems identified

**3.1 The 3D mountain plot is redundant**
It displays `max(abs(ei))` per channel — identical information to the 2D heatmap, just rendered as a surface. It occupies a full slot in the view dropdown while providing no additional insight.

**3.2 The latency map is static**
A scatter of colored dots showing time-to-threshold-crossing is a poor approximation of what the data actually contains: a real spatiotemporal propagating wavefront across 512 electrodes over ~200 time samples. The animation *is* the information.

**3.3 Temporal waveforms have no spatial context**
Three traces are shown for the top electrodes, labeled only by channel index. There is no way to know where those channels are on the array, and no way to click a different electrode to inspect its trace.

**3.4 ei_corr_dict is fully computed but invisible**
Three correlation matrices (full waveform, spatial footprint, power spectrum) are already computed, cached, and pickled. This is the backbone for duplicate detection and nearest-neighbor cell browsing. It is not connected to any UI element.

**3.5 No deduplication workflow**
Duplicate units (same neuron captured by two clusters) are one of the main quality control targets of this application. The EI is the most reliable signal for identifying them. There is no panel or workflow for presenting likely duplicates to the user for confirmation and merging.

### Proposed redesign — three-pane layout

```
[ Array Map (clickable) ] | [ Wavefront Animation ] | [ Similarity / Dedup ]
```

**Pane 1 — Array map (pyqtgraph ImageItem + ScatterPlotItem overlay)**
- 2D heatmap of EI amplitude on the physical electrode grid (this already exists, just move it to a permanent pane rather than a dropdown view)
- All 512 electrode positions rendered as dots, sized by amplitude
- Click any electrode → updates Pane 2 to show that electrode's waveform
- Current top-3 electrodes highlighted with colored rings
- Replaces both the 3D mountain plot and the latency map dropdown options

**Pane 2 — Wavefront animation**
- `QTimer`-driven frame loop over the 201 EI time samples
- Each frame: rerender the amplitude heatmap for that time slice
- Play / Pause / Step buttons + a scrubber `QSlider` for manual frame navigation
- Peak frame auto-detected (frame of global minimum across all channels) and marked with a distinct color ring on the array map
- Speed control (e.g. 0.5×, 1×, 5×, 10× — 10× plays all 201 frames in ~200ms which feels natural)
- When an electrode is clicked in Pane 1, Pane 2 shows a static trace for that electrode instead, with the peak marked

**Pane 3 — Similarity & dedup**
- Pull top-N similar cells from `ei_corr_dict["full"][cluster_idx, :]`, sort descending
- Render as a ranked list: each row shows cluster ID, correlation score, and a mini EI amplitude thumbnail
- Color-code by correlation: r > 0.95 = red (likely duplicate), 0.8–0.95 = orange (similar), < 0.8 = grey
- Click a row → selects that cell in the tree (same as clicking in UMAP)
- "Mark as Duplicate" button on each row → calls `update_and_export_status` with `dup` status
- Three correlation mode tabs at top: Full | Spatial | Power — toggle which `ei_corr_dict` matrix is used for ranking

This gives EI deduplication as a first-class workflow, not an afterthought.

### Implementation notes
- Frame animation: `QTimer` at ~50ms interval calling `image_item.setImage(ei_data[:, :, frame])` where `ei_data` is reshaped to the spatial grid. pyqtgraph `ImageItem` can redraw at this rate without issue.
- `ei_corr_dict` access: add a `get_ei_similarity(cluster_id, method='full', top_n=10)` method to `DataManager` that translates cluster_id → vision_id, slices the correlation row, returns sorted (id, score) pairs. This isolates the ID offset translation in one place (Law 1).
- The `is_vision_only` guard on `_compute_ei_correlations_if_needed` must be preserved — this prevents a 512×512 matrix build on large Vision-native datasets.

---

## 4. Population Panel — Promote to Dedicated Tab

### Current state
A narrow right sidebar (~250px) showing: population RF mosaic (matplotlib), population dynamics (timecourse), population autocorrelation. Always visible regardless of which tab is active. The selected cell is not highlighted in any of these plots during scrolling.

### Problems identified

**4.1 Sidebar is too narrow for the content**
The RF mosaic with 300+ cells is illegible at 250px wide. The timecourse and population ACG are both cropped. This is premium information rendered at unusable size.

**4.2 Context-insensitive**
The sidebar shows group-level summaries while you're looking at a single cell's EI or raw trace. The information is irrelevant to the current task in most tabs.

**4.3 Selected cell not highlighted**
When you scroll through the tree, the population plots don't indicate which cell you're on. This breaks the mental link between individual cell inspection and population context.

### Proposed redesign — dedicated Population tab

Promote population views to a full-width tab alongside Standard, EI, STA, UMAP, Waveforms, Raw.

**Layout within the tab:**

```
[ Population RF Mosaic (full width, zoomable) ]
[ Population Timecourse ] [ Population ACG ]
```

- RF mosaic: full panel width, matplotlib with proper zoom/pan. Selected cell's RF ellipse highlighted in a distinct color (e.g. cyan ring) and redrawn on every cluster selection without full replot.
- Population timecourse: group-mean STA timecourse with individual cell traces in low alpha, selected cell's trace in full brightness.
- Population ACG: group-mean ACG with selected cell's ACG overlaid in a contrasting color.
- "Show IDs" toggle remains, controlling text labels on the mosaic.
- All three plots update the selected-cell highlight on Tier 1 (immediate, hot-swap) — only the group-level recomputation goes to Tier 2.

**Right sidebar behavior after this change**
Once population views move to their own tab, the right sidebar can be repurposed or collapsed by default. Options:
- Show a compact "cell card" summary: cluster ID, spike count, FR, ISI %, τ — a quick stats panel that's always visible.
- Or collapse it entirely and reclaim ~250px of horizontal space for the main panels.

### Implementation notes
- Selected cell highlight in RF mosaic: store the currently-highlighted ellipse artist, update its color/linewidth on selection change without calling `draw_population_rfs_plot` again. Use `artist.set_edgecolor()` + `canvas.draw_idle()`.
- The existing `_group_timecourse_cache` and `_group_acg_cache` in `population_panel.py` are already structured correctly for this — no cache changes needed.
- The sidebar removal is a layout change in `main_window.py` only. No DataManager changes.

---

## 5. Summary — Priority Order

| # | Change | Impact | Effort | Touches data_manager? |
|---|---|---|---|---|
| 1 | Tree collapse fix (targeted subtree insert) | HIGH — unblocks UMAP workflow | Low | No |
| 2 | EI Pane 3 — similarity ranked list from `ei_corr_dict` | HIGH — near-zero backend work, infra already built | Low | No |
| 3 | ISI % badge + ACG τ badge on Standard plots | HIGH — daily use, instant classification | Medium | Yes (add τ to `_compute_standard_plots`) |
| 4 | UMAP hover (pyqtgraph scatter + mini popup) | HIGH — transforms exploration speed | Medium | No |
| 5 | Feature weight sliders in UMAP | MEDIUM — improves embedding quality | Low | No |
| 6 | Kilosort-only UMAP fallback (no Vision/STA) | MEDIUM — unblocks KS-only datasets | Medium | Yes (`get_physics_feature_matrix`) |
| 7 | EI wavefront animation (replace latency map) | HIGH — scientifically beautiful and useful | Medium | No |
| 8 | EI array map → clickable electrode → trace | MEDIUM | Medium | No |
| 9 | Population panel → dedicated tab | MEDIUM — cleaner layout | Medium | No |
| 10 | UMAP 3D → Plotly/vispy | MEDIUM — better 3D exploration | High | No |

---

## 6. Things Deliberately Not in This Report

- **Physics cache load time on first open**: already addressed in prior work sessions. Not regressed.
- **Raw panel**: not discussed — no user feedback indicating problems.
- **STA panel crash on missing `.sta`**: already listed as open in PLAN.md §4 Active Work. Out of scope here.
- **EI panel crash on missing `.ei`**: same, already tracked.
- **LazySTADict concurrency**: already fixed, tests exist, do not touch.
