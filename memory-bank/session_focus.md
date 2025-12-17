# Session Focus: Fix EI 3D Visualization OpenGL Context Issues

## Date: December 17, 2025
## Focus: Provide a stable, cross-platform EI spatial visualization stack (Matplotlib 3D + Latency Map)

### Objective
Replace unstable OpenGL-based 3D rendering with a robust Matplotlib-based visualization suite while adding analytical views to aid interpretation:
- Provide a rotatable Matplotlib 3D Max-Projection surface for EI footprint visualization
- Provide a 2D Latency Map scatter view to visualize spike propagation timing
- Preserve voltage inversion and z-scaling semantics
- Improve robustness against driver/context failures (remove problematic accelerator, add Qt fallbacks)

### Completed Tasks
1. Implemented `EIMountainPlotWidget` using Matplotlib 3D `plot_surface` (rotatable max-projection).
2. Added `plot_latency_map(...)` helper and integrated it into `EIPanel` view dropdown.
3. Removed `PyOpenGL_accelerate` from `requirements.txt` and recommended uninstall in environments to avoid segfaults.
4. Added defensive code in `gui/plot_widgets.py`: guarded updates, deferred initial draw, retry-on-failure.
5. Added a conservative `QSurfaceFormat` and software-OpenGL attribute in `main.py` to mitigate driver issues.

### Next Focus
- Add vector-field overlays (propagation vectors) on the Latency Map
- Add isochrone/wavefront contours for arrival-time visualization
- Cache interpolated grids for smoother interaction and faster redraws
- Offer an optional higher-performance 3D path (reintroduce OpenGL or use a GPU-backed library) behind a user toggle

### Status
COMPLETED (engine switch and integration). Next enhancements planned as listed above.