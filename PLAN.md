# Feature Plan: Implement New Panel Designs and Bug Fixes

## ✅ COMPLETED
- [x] **Phase 1: Implement New Panels**
    - [x] Update `gui/panels/waveforms_panel.py` with the new "Snippet Cloud" implementation.
    - [x] Update `gui/panels/standard_plots_panel.py` with the new "Home Dashboard" implementation.

- [x] **Phase 2: Integration and Verification**
    - [x] Review `gui/main_window.py` to ensure the new panels are correctly integrated.
    - [x] Fix: Adjusted `on_similarity_selection_changed` in `main_window.py` to pass single cluster ID to `waveforms_panel`.
    - [x] Fix: Corrected `pyqtgraph` `stepMode=True` error in `standard_plots_panel.py` by adjusting x-axis data for ACG plot.
    - [x] Verified changes.

- [x] **Phase 3: Enhance Standard Plots Panel**
    - [x] Remove the orange amplitude drift line from the firing rate plot as it's not informative
    - [x] Add option to show only the main/most dominant channel waveform in the spatial template section
    - [x] Add ISI vs amplitude scatter plot functionality (added as toggle in ISI Distribution panel)
    - [x] Add toggle controls to switch between different visualization modes in the standard plots panel (ISI histogram vs ISI vs amplitude, show all channels vs main channel only)
    - [x] Update the 2x2 grid layout to accommodate the new ISI vs amplitude plot instead of firing rate with amplitude overlay (PARTIALLY COMPLETED - added toggle instead of replacing layout)
    - [x] Implement channel selection algorithm to determine the "main" channel for the dominant waveform view (NEEDS FINALIZATION)

- [x] **Phase 4: Advanced EI Analysis Tab**
    - [x] Implement EI Contour "Topography" (3D surface plot) visualization.
        - Implemented as a switchable view in the EI Panel, preserving the original 2D heatmap.
        - Added `PyOpenGL` and `PyOpenGL_accelerate` to `requirements.txt` for 3D functionality.
    - [x] Add dropdown menu to switch between visualization types.
    - [x] Ensure 3D visualization properly handles Vision EI data format.

## 🔄 PENDING
- [ ] **Phase 4 Continued: Advanced EI Analysis Tab**
    - [ ] Add Propagation Vector Field to show direction and speed of spike propagation
    - [ ] Create Temporal-Spatial EI Movie for animation of EI propagation over time
    - [ ] Develop Heat Diffusion Visualization to show how EI spreads from initiation site
    - [ ] Add Wavefront Isochrone Lines showing equal arrival times
    - [ ] Integrate new visualization types with dropdown menu for visualization selection
    - [ ] Add animation controls (play/pause) for temporal visualizations
    - [ ] Implement 3D interactive view functionality
    - [ ] Update cache mechanisms for interpolated grids for smooth interaction
    - [ ] Add export functionality for EI analysis visualizations
