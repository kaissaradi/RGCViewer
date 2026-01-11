RGCViewer Enhancement Plan

OBJECTIVES
- Enhance EI panel with population statistics and improved visualizations
- Improve STA panel to display comprehensive statistics and population data
- Implement feature selection, class creation, and classification saving
- Fix spatial waveform visualization to properly handle whitened templates per kilosort guide

COMPLETED FEATURES
- Snippet Cloud implementation in waveforms panel
- Home Dashboard implementation in standard plots panel
- Fixed similarity selection and pyqtgraph stepMode issues
- Enhanced standard plots with ISI vs amplitude scatter plots and toggle controls
- Advanced EI analysis with contour topography visualization
- Zero-latency scrolling implementation with Tier 1/Tier 2 architecture
- Hot-swap rendering for population RF visualizations
- Persistent PyQtGraph objects for 60fps interaction
- Refactored plotting functions to dedicated panel modules
- Made retinanalysis module optional with graceful degradation
- Suppressed peak width calculation warnings

CURRENT DEVELOPMENT
- UMAP Panel: Enhancing clustering and dimensionality reduction features
- Feature Selection: Developing interface for extraction and importance scoring
- Population Analysis Dashboard: Multi-scale analysis with individual, subclass, and population views

PLANNED FEATURES
- Population Split View: Split view activated by checkbox showing population RFs initially
- UMAP Panel Enhancement: Advanced clustering algorithms and dimensionality reduction techniques
- Class Creation: Clustering algorithms, manual tools, validation metrics
- Data Persistence: Save/load classifications, version control, export options
- Spatial Waveform View: Properly handle whitened templates per kilosort guide

STA ANALYSIS TAB ENHANCEMENTS
- [ ] **Interactive Time-Slicing**: Link timecourse plots to RF movie frames
- [ ] **Color Opponency Metric**: Calculate and display opponency indices
- [ ] **SVD Separability**: Implement Space-Time SVD separability index
- [ ] **Fit Residuals**: Show residuals (Raw STA - Gaussian Fit) for fit quality
- [ ] **Population Percentiles**: Display where cell properties sit relative to population
- [ ] **Export Report**: Save current 4-panel view as high-res PDF/PNG

SPATIAL WAVEFORM ENHANCEMENTS
- [x] **Template Unwhitening**: Load whitening_mat_inv.npy for proper visualization
- [ ] **Weighting Combo Fix**: Add missing weighting combo box to UI controls
- [ ] **Amplitude Calculation**: Improve cluster amplitude calculation

DEVELOPMENT APPROACH
- Follow TDD methodology with atomic steps from this plan
- Maintain Python, PyQt, pyqtgraph, numpy, scipy tech stack
- Integrate with existing analysis engine and vision modules

ARCHITECTURE SUMMARY
The application now uses a **"Stateful Update"** model with Tier 1/Tier 2 architecture:
- **Tier 1 (Immediate):** Fast UI updates (<16ms) for population maps, histograms, and cached data
- **Tier 2 (Delayed):** Heavy computations (150ms delay) for raw data loading and new feature calculations
- **Hot-Swap Rendering:** Population RFs update in <1ms instead of 100ms by preserving canvas state
- **Persistent Objects:** PyQtGraph items are created once and updated with setData() for 60fps performance