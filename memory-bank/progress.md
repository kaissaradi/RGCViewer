Implemented new `WaveformPanel` and `StandardPlotsPanel` designs. Resolved an integration bug in `main_window.py` and a plotting error in `standard_plots_panel.py` related to `pyqtgraph`'s `stepMode`.

Successfully completed Phase 3 enhancements to the Standard Plots Panel:
- Removed uninformative orange amplitude drift line from firing rate plot
- Added toggle to show main channel waveform only vs all channels
- Added ISI vs amplitude scatter plot as an option in the ISI Distribution panel (accessible via toggle)
- Added multiple visualization toggle controls (ISI histogram vs ISI vs amplitude, all channels vs main channel only)
- Fixed syntax errors and completed the channel selection algorithm

Remaining work: Continue with Phase 4 - implementing advanced EI analysis features including contour topography, propagation vector fields, temporal animations, and heat diffusion visualizations as specified in the PLAN.md.