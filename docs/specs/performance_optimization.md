# Specification: Advanced Testing & Performance Optimization

## Objective
Implement a multi-tiered testing strategy to ensure GUI responsiveness, visual correctness, and memory stability.

## 1. Frantic User Stress Test (Concurrency)
- **Goal**: Prevent UI freezes during heavy background calculation.
- **Scenario**: Simulate rapid cell switching (50 cells in 2 seconds) while UMAP and Physics workers are active.
- **Metric**: Assert that the main event loop never blocks for more than 100ms.

## 2. Visual Regression (Standard Plots)
- **Goal**: Ensure plots (ACG, ISI, RF) remain visually correct across updates.
- **Tool**: `pytest-mpl` for image comparison.
- **Scope**: ACG Histograms, ISI Histograms, and RF Heatmaps.

## 3. Memory Leak Protection
- **Goal**: Prevent the app from consuming unbounded RAM during long analysis sessions.
- **Method**: Use `tracemalloc` or `objgraph` to monitor object counts (especially Matplotlib Figures and Qt Widgets) during repeated panel opening/closing.

## 4. Hardware Simulation (Trace Viewer)
- **Goal**: Test `RawPanel` with synthetic data.
- **Method**: Implement a `MockBinReader` that generates sine waves or random noise to simulate a .dat file.
