RGCViewer Enhancement Plan

OBJECTIVES
- Enhance EI panel with population statistics and improved visualizations
- Improve STA panel to display comprehensive statistics and population data
- Implement feature selection, class creation, and classification saving

COMPLETED FEATURES
- Snippet Cloud implementation in waveforms panel
- Home Dashboard implementation in standard plots panel
- Fixed similarity selection and pyqtgraph stepMode issues
- Enhanced standard plots with ISI vs amplitude scatter plots and toggle controls
- Advanced EI analysis with contour topography visualization

UPDATES
- Removed PyOpenGL_accelerate to prevent context segfaults
- Added Qt software-OpenGL fallback for stability
- Improved plot widget robustness and added latency map visualization

CURRENT DEVELOPMENT
- EI Panel: Adding population statistics and correlation matrix visualization

PLANNED FEATURES
- STA Panel: Consolidate statistics, add population metrics, improve visualizations
- Feature Selection: Interface for extraction, dimensionality reduction, importance scoring
- Class Creation: Clustering algorithms, manual tools, validation metrics
- Data Persistence: Save/load classifications, version control, export options

STA ANALYSIS TAB ENHANCEMENTS
- [ ] **Interactive Time-Slicing**: Link the "Timecourse" plot click event to the "RF Movie" frame. Clicking a time point updates the RF view to that time lag.
- [ ] **Color Opponency Metric**: Calculate and display Red-Green / Blue-Yellow opponency indices in the metrics box.
- [ ] **SVD Separability**: Implement Space-Time SVD to calculate a "Separability Index" and display it.
- [ ] **Fit Residuals**: Add a toggle to show the "Residual" (Raw STA - Gaussian Fit) in the RF view to judge fit quality.
- [ ] **Population Percentiles**: Display where this cell's properties (Area, Latency) sit relative to the population (e.g., "Latency: 45ms (80th %ile)").
- [ ] **Export Report**: Button to save the current 4-panel view as a high-res PDF/PNG.

DEVELOPMENT APPROACH
- Follow TDD methodology with atomic steps from this plan
- Maintain Python, PyQt, pyqtgraph, numpy, scipy tech stack
- Integrate with existing analysis engine and vision modules
