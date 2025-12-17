Session completed: Successfully enhanced the Standard Plots Panel with the following features:
1. Removed the uninformative orange amplitude drift line from the firing rate plot
2. Added option to show the main/most dominant channel waveform using a toggle checkbox
3. Added ISI vs amplitude scatter plot functionality accessible via a toggle in the ISI Distribution panel
4. Added toggle controls for switching between different visualization modes (ISI histogram vs ISI vs amplitude, all channels vs main channel only)

The StandardPlotsPanel.py file now includes proper syntax and handles all visualization modes correctly. The implementation uses a channel selection algorithm to determine the main channel based on peak-to-peak amplitude.

Next steps: Continue with Phase 4 to implement advanced EI analysis features including contour topography, propagation vector fields, and temporal animations as outlined in the PLAN.md.