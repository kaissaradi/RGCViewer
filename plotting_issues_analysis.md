# RGCViewer Plotting Issues Analysis

## Overview
Analysis of various plotting issues in the RGCViewer application, including missing ISI plots, firing rate plots, EI plots, and buggy STA RF plots.

## Critical Issues

### 1. Cluster ID Mismatch in Similarity Panel (RESOLVED)
**Original Error:** `IndexError: index 0 is out of bounds for axis 0 with size 0` in `similarity_panel.py:175`

**Original Root Cause:**
- The similarity panel expects Vision cluster IDs (1-indexed) to match with selected Kilosort cluster IDs (0-indexed)
- Code converts Vision IDs: `cluster_ids = np.array(list(self.main_window.data_manager.vision_eis.keys())) - 1`
- However, selected cluster IDs may not exist in the Vision data, causing `np.where(cluster_ids == cluster_id)[0]` to return an empty array
- Accessing `[0]` on an empty array throws the IndexError

**Resolution:**
- Added validation in similarity panel to check if cluster_id exists in Vision data before accessing
- Now properly handles cases where selected cluster ID is not available in Vision data

**Impact of Original Issue:**
- Caused cascade of errors throughout the application
- Prevented similarity panel from displaying
- Blocked other plots from updating properly

### 2. EI Visualization Error (ACTIVE)
**Error:** `ValueError: different number of values and points` in `ei_panel.py:35`

**Root Cause:**
- The `compute_ei_map` function uses `griddata` which expects matching number of points and values
- When Vision EI data has different dimensions or invalid shape, the interpolation fails
- The function assumes EI data has 512 channels but may receive different channel count
- Now occurs during `on_similarity_selection_changed` when trying to plot multiple clusters

**Current Evidence from Logs:**
```
[DEBUG] on_similarity_selection_changed: main_cluster = 1
[DEBUG] on_similarity_selection_changed: clusters_to_plot = [1, 1038638581]
Traceback (most recent call last):
  File "/home/fieldlab/Documents/RGCViewer/gui/main_window.py", line 570, in on_similarity_selection_changed
    self.ei_panel.update_ei(clusters_to_plot)
  File "/home/fieldlab/Documents/RGCViewer/gui/panels/ei_panel.py", line 137, in update_ei
    self._load_and_draw_vision_ei(cluster_ids)
  File "/home/fieldlab/Documents/RGCViewer/gui/panels/ei_panel.py", line 164, in _load_and_draw_vision_ei
    ei_map = compute_ei_map(
  File "/home/fieldlab/Documents/RGCViewer/gui/panels/ei_panel.py", line 35, in compute_ei_map
    ei_energy_grid = griddata(
  File "/home/fieldlab/miniconda3/envs/rgcviewer/lib/python3.9/site-packages/scipy/interpolate/_ndgriddata.py", line 323, in griddata
    ip = LinearNDInterpolator(points, values, fill_value=fill_value,
  File "interpnd.pyx", line 326, in scipy.interpolate.interpnd.LinearNDInterpolator.__init__
  File "interpnd.pyx", line 95, in scipy.interpolate.interpnd.NDInterpolatorBase.__init__
  File "interpnd.pyx", line 104, in scipy.interpolate.interpnd.NDInterpolatorBase._set_values
  File "interpnd.pyx", line 232, in scipy.interpolate.interpnd._check_init_shape
ValueError: different number of values and points
```

**Impact:**
- EI plots still fail to display
- Multiple cluster EI comparisons fail
- Affects spatial visualization of EI data for selected clusters

### 3. Missing Plots (RESOLVED)
**Status:** Waveform, ISI, and firing rate plots are now working after uncommenting the code in `_draw_plots`

**Evidence:** The plots should now appear in the waveforms panel

### 4. Similarity Panel ID Mismatch (RESOLVED)
**Status:** Fixed by adding proper validation in similarity panel

## Data Flow Issues

### 1. Vision vs Kilosort ID Mapping (RESOLVED)
**Problem:** Vision data uses 1-indexed cluster IDs, Kilosort data uses 0-indexed cluster IDs

**Resolution:** Added validation to ensure selected cluster IDs exist in Vision data before accessing

### 2. Channel Position Mapping (ACTIVE)
**Problem:**
- `compute_ei_map` function assumes 512 channels
- May fail if actual channel count differs
- No validation of channel position array dimensions
- The `griddata` function expects matching number of points and values

## Feature-Specific Issues

### 1. STA RF Plot Not Auto Updating (POTENTIAL)
**Issue Status:** May still have issues with auto-updating when selecting different clusters
**Code Location:** `raw_panel.py` and related STA handling in `main_window.py`

### 2. Multiple Cluster Plotting (ACTIVE)
**Problem:**
- When selecting similar clusters, EI plots fail with dimension mismatch
- The `on_similarity_selection_changed` method passes cluster IDs [1, 1038638581] where 1038638581 appears to be invalid or has incompatible dimensions
- Need to validate each cluster ID before attempting to plot EI data

## Questions for Clarification

1. **EI Data Dimensions:**
   - What are the expected dimensions for EI data (channels x timepoints)?
   - How should the system handle EI arrays with different dimensions when plotting multiple clusters?

2. **Invalid Cluster IDs:**
   - What is the significance of large cluster IDs like 1038638581 and 3186463974 in the error logs?
   - Are these invalid values from corrupted data or incorrect conversions?

3. **EI Data Validation:**
   - Should the system validate EI data dimensions before attempting interpolation?
   - How should the system handle clusters with malformed or inconsistent EI data?

4. **Griddata Parameters:**
   - Are the channel positions and EI values being passed correctly to the `griddata` function?
   - Should there be validation of the input shapes before calling interpolation?

## Recommended Fixes

### Immediate (Active Issues):
1. Add validation in `ei_panel.py` to check EI data dimensions before interpolation
2. Filter out invalid cluster IDs in similarity selection before plotting
3. Add error handling in `compute_ei_map` to handle dimension mismatches gracefully

### Completed:
- Uncommented waveform, ISI, and firing rate plotting code
- Fixed cluster ID validation in similarity panel
- Added proper error handling for missing Vision data

### Medium-term:
1. Add comprehensive validation for EI data dimensions across all clusters before attempting multi-cluster plots
2. Ensure consistent handling of cluster ID indexing throughout the application
3. Implement graceful degradation when EI data is missing or malformed for specific clusters