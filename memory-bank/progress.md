# Progress Tracking: EI Visualization — Matplotlib 3D & Latency Map

## Date: December 17, 2025

### Progress Made
✅ COMPLETED: Replaced unstable OpenGL 3D rendering with Matplotlib-based visualizations and added a latency map.

### Tasks Completed
1. ✅ Diagnosed OpenGL context segfaults and race conditions caused by `PyOpenGL_accelerate` and driver issues.
2. ✅ Removed `PyOpenGL_accelerate` from `requirements.txt` and recommended uninstalling it from environments.
3. ✅ Added Qt software-OpenGL fallback and a conservative `QSurfaceFormat` in `main.py` to reduce driver-related crashes.
4. ✅ Implemented a Matplotlib 3D Max-Projection `EIMountainPlotWidget` in `gui/plot_widgets.py` (rotatable surface).
5. ✅ Added `plot_latency_map(...)` helper and integrated a "Latency Map" option in `gui/panels/ei_panel.py`.
6. ✅ Added defensive updates: guarded `update_frame`, deferred initial draw, and retry-on-failure for transient GL-like calls.
7. ✅ Preserved EI data pipeline compatibility for Vision and Kilosort datasets.

### Technical Improvements Achieved
- Stable 3D visualization via Matplotlib that works across platforms (no OpenGL driver dependency).
- Latency Map provides a complementary view showing spike propagation timing across channels.
- Robustness improvements guard against race conditions and context-related exceptions.

### Status
Visualizations are integrated and functional. Next steps include adding propagation vectors, isochrone/wavefront overlays, caching/interpolation optimizations, and optional higher-performance 3D rework if desired.