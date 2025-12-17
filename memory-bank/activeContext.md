# Active Context: EI 3D Visualization Enhancement

## Current Task: Stable EI Spatial Visualizations (Matplotlib-based)

### Date: December 17, 2025
### Status: COMPLETED (engine switch, integration, and robustness fixes)

### Task Summary
We replaced an unstable OpenGL-based 3D renderer with a stable Matplotlib-based visualization suite and applied several robustness fixes to prevent GL-context crashes. The result is a working interactive analysis UI that provides:
- A rotatable Matplotlib 3D Max-Projection surface (EI mountain)
- A 2D Latency Map scatter view (time-to-peak colored map)
- Stable 2D contour (animated) visualization as a fallback

### Files Modified
- `gui/plot_widgets.py`: Implemented Matplotlib 3D `EIMountainPlotWidget` (max-projection surface) and removed the OpenGL-dependent variant.
- `gui/panels/ei_panel.py`: Added `plot_latency_map` helper and integrated "Latency Map" into the view dropdown.
- `main.py`: Added a conservative `QSurfaceFormat` and software-OpenGL attribute to reduce driver-related crashes.
- `requirements.txt`: Removed `PyOpenGL_accelerate` to avoid context/driver issues.

### Key Features Implemented
1. Matplotlib 3D Max-Projection surface (rotatable, stable across platforms)
2. 2D Latency Map showing per-channel time-to-peak (ms) with colorbar
3. Preserved voltage inversion and Z-scaling semantics (mountain metaphor)
4. Defensive updates: deferred first GL-like draw, retry on transient errors, and guarded updates
5. Dropdown integration for switching between 2D heatmap, 3D mountain, and latency map

### Technical Details
- Uses `scipy.interpolate.griddata` for spatial interpolation (same as prior approach)
- Uses `gui.widgets.MplCanvas` as the Matplotlib wrapper for embedding canvases
- Preserves existing EI data workflows for Vision and Kilosort formats

### Integration Status
- Visualization integrated into `EIPanel`; calls to `mountain_plot_widget.plot_ei_3d(...)` and `plot_latency_map(...)` are wired where EI is loaded.
- OpenGL-accelerator-related crashes have been addressed by removing `PyOpenGL_accelerate` and adding Qt fallbacks; the project now prefers Matplotlib for 3D.