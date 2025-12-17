The last session focused on implementing a 3D "mountain plot" for EI (Extracellular Impulse) visualization in the `EIPanel`.

- Created a new `EIMountainPlotWidget` in `gui/plot_widgets.py` using pyqtgraph's OpenGL for 3D visualization.
- Updated the `EIPanel` in `gui/panels/ei_panel.py` to use a `QStackedWidget` allowing users to switch between 2D heatmap and 3D mountain plot views.
- Added a dropdown menu to control the visualization type selection.
- Modified both `_load_and_draw_vision_ei` and `_load_and_draw_ks_ei` methods to update the 3D visualization when data is loaded.
- Fixed an issue with the 3D visualization not handling the `EIContainer` format from Vision data files.
- Resolved the `setColorMap` error by removing it since `GLSurfacePlotItem` doesn't support this method.

The 3D visualization now properly handles both Kilosort and Vision EI data formats and provides a dropdown menu for switching between 2D and 3D views.