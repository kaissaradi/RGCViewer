# Session Plan: Refactoring GUI Structure - Move plotting functions to panels

## Original Problem Identified
- The `src/gui/plotting.py` file contained population plotting functions that would be better organized in a dedicated panel
- This would eliminate an unnecessary directory and consolidate related functionality

## Proposed Plan
1. Create a new file `population_panel.py` in the `src/gui/panels/` directory
2. Move all functions from `src/gui/plotting.py` to the new file
3. Update imports in main_window.py to reference the new location
4. Remove the `src/gui/plotting.py` file
5. Update the main window to call the population plotting functions from the new location

## Actions Taken

### 1. Created population_panel.py with plotting functions
- Moved all population plotting functions from plotting/plotting.py to the new file
- Ensured proper imports are included

### 2. Updated main_window.py imports
- Changed import statements to reference the new location
- Updated function calls to remove the "plotting." prefix

### 3. Removed the old plotting directory
- Deleted the entire `src/gui/plotting/` directory and its contents

### 4. Updated the panels/__init__.py file
- Added imports for the population plotting functions
- Updated the __all__ list to include the new functions

### 5. Verified functionality
- Confirmed all population plotting functionality still works correctly
- Checked that all function calls have been properly updated

## Result
The population plotting functionality has been successfully moved to a dedicated panel file in the panels directory, eliminating the separate plotting directory and improving code organization. The structure is now cleaner and more maintainable.
=======

## Additional Fixes Discovered During Refactoring
- Fixed remaining references to the old plotting module in ei_panel.py and sta_panel.py
- Updated imports in ei_panel.py to use plot_rich_ei from the new population_panel location
- Updated imports in sta_panel.py to use draw_population_rfs_plot from the new population_panel location
- Ensured all cross-panel references now point to the correct location in the panels module

## Next Focus: UMAP Panel Enhancement

- Turning attention to the UMAP panel for clustering and dimensionality reduction features

- Planning to implement feature selection, class creation, and classification saving capabilities
<<<<<<< HEAD
- Will focus on the umap_panel.py and related components
>>>>>>> f328839 (save before fixing feature rextratction):docs/session-plan.md
=======

- Will focus on the umap_panel.py and related components



## Feature Extraction Panel 2.0 (Completed)



**Goal:** "10x" the Feature Extraction window with lazy loading, modern UX, and better plotting.







### 1. Performance & Lazy Loading



- **Implemented:** `FeatureAnalysisWorker` (QThread) handles all heavy computation.



- **Implemented:** **Caching** in `DataManager`. Computed features (PCA, Traces, ACG) are stored by `cluster_id`.



    - *Result:* Re-opening the window or creating groups is now near-instantaneous after the first load.







### 2. Plotting Improvements



- **Top Right Plot (Subplot 2):**



    - *Old:* Histogram of RF Diameters.



    - *New:* **Time to Peak vs RF Diameter**.



    - *Why:* This provides excellent separation between **Parasol** (Fast + Large RF) and **Midget** (Slow + Small RF) cells.



- **Aesthetics:**



    - Despined plots, alpha blending, improved labels ("Time to Peak (frames)", "RF Diameter (µm)").







### 3. UX: Linked Brushing & Selection



- **Implemented:** Selecting points in one plot highlights the corresponding cells in **all 6 plots** instantly.



- **Implemented:** "Create Group" context menu works seamlessly with the selection.







## Next Focus: UMAP Panel Enhancement



- Turning attention to the UMAP panel for clustering and dimensionality reduction features



- Planning to implement feature selection, class creation, and classification saving capabilities



- Will focus on the umap_panel.py and related components



## UMAP Panel Enhancement Plan

### Goal
Enhance the UMAP panel with the following features:
1. Remove all IAN-related code
2. Add 3D UMAP visualization option
3. Implement HDBSCAN clustering instead of K-Means
4. Prioritize specific important features for RGC classification
5. Add iterative clustering capability based on selected groups

### Phase 1: Clean Up and Removal
#### 1.1 Remove All IAN-Related Code
- Completely remove the IAN import section and related code
- Remove the IANWorker class and its associated functionality
- Remove the IAN button from the UI
- Clean up any IAN-related variables and methods
- Remove unused imports related to 3D projections that are not needed

#### 1.2 Update Dependencies
- Ensure only umap-learn, hdbscan, and necessary scikit-learn packages are imported
- Update requirements.txt if necessary with HDBSCAN

### Phase 2: Core Feature Enhancement
#### 2.1 Feature Prioritization Implementation
Implement feature extraction focusing on the most important RGC features:
- Timecourse PCA 1 and 2 (extracted from dominant channel traces)
- ACG PCA 1 and 2 (auto-correlation function principal components)
- Time course statistics:
  - Time to peak
  - FWHM (Full Width Half Maximum)
  - Biphasic index
  - Energy
  - Zero crossing
- RF (Receptive Field) properties:
  - RF diameter
  - RF size
  - RF angle
  - X vs Y sigma (spatial properties)

#### 2.2 Improved Feature Extraction Function
Modify extract_features_from_datamanager to prioritize the specified features:
- Add ACG PCA features extraction
- Add enhanced time course statistics
- Add comprehensive RF property metrics
- Maintain backward compatibility with existing features

### Phase 3: Visualization Enhancement
#### 3.1 3D UMAP Integration
- Add a 3D UMAP button and functionality
- Modify the UMAPWorker to support both 2D and 3D embeddings
- Implement proper 3D axes initialization
- Add 3D rotation and interaction capabilities
- Add toggle between 2D/3D views

#### 3.2 Enhanced Visualization Features
- Improve plot aesthetics and interactivity
- Add better labeling and tooltips
- Implement proper 3D projection handling for lasso selection

### Phase 4: Clustering Enhancement
#### 4.1 HDBSCAN Integration
- Replace K-means with HDBSCAN clustering
- Add HDBSCAN controls (min_cluster_size, min_samples)
- Implement HDBSCANWorker class similar to KMeansWorker
- Update UI to reflect HDBSCAN parameters
- Add option to show noise points differently

#### 4.2 Improved Clustering UI
- Update clustering interface with HDBSCAN parameters
- Add slider controls for min_cluster_size and min_samples
- Allow dynamic adjustment of clustering parameters
- Add option to show cluster probabilities

### Phase 5: Iterative Selection and Re-embedding
#### 5.1 Selection-Based Re-embedding
- Implement functionality to select a cluster/group in the current view
- Add option to re-embed only selected subset of points
- Add "Generate" button to trigger re-embedding on selected clusters
- Support iterative refinement of clusters based on selection

#### 5.2 Group Management
- Allow users to save current selections as named groups
- Integrate with the main application's group management system
- Enable round-trip between UMAP selection and main GUI
- Add buttons for "Save Selection" and "Re-embed Selected"

### Phase 6: UI/UX Improvements
#### 6.1 Enhanced Control Layout
- Restructure the control layout to accommodate new features
- Add tabs or collapsible sections for different functionalities
- Organize controls logically for better user experience
- Add progress indicators for long-running operations

#### 6.2 Performance Optimizations
- Implement caching for computed embeddings
- Add lazy loading for large datasets
- Optimize feature computation pipeline
- Add cancellation support for long-running tasks

### Implementation Steps

#### Step 1: Code Cleanup
1. Remove all IAN-related code from umap_panel.py
2. Update import statements
3. Remove IAN button and related UI elements
4. Update requirements.txt to reflect new dependencies

#### Step 2: Feature Extraction Enhancement
1. Modify extract_features_from_datamanager() function
2. Add ACG PCA and time course statistics computation
3. Add RF property extraction
4. Ensure feature prioritization is reflected in feature weights

#### Step 3: 3D Visualization
1. Add 3D UMAP functionality to UMAPWorker
2. Implement toggle for 2D/3D visualization
3. Handle 3D lasso selection properly
4. Add 3D rotation controls

#### Step 4: HDBSCAN Integration
1. Create HDBSCANWorker class
2. Add HDBSCAN controls to UI
3. Update clustering workflow to use HDBSCAN
4. Add parameter tuning options

#### Step 5: Iterative Functionality
1. Implement selection-based re-embedding
2. Add "Generate" button for re-embedding
3. Connect selection to group management system
4. Test iterative workflow thoroughly

#### Step 6: Testing and Validation
1. Validate all new features work as expected
2. Ensure backward compatibility
3. Test with various dataset sizes
4. Optimize performance for large datasets

### Technical Implementation Notes

#### Feature Extraction
- Timecourse PCA: Compute PCA on dominant channel traces
- ACG PCA: Extract auto-correlation functions and compute PCA
- Time course stats: Compute statistics from normalized traces
- RF properties: Extract from STA fitting parameters

#### 3D Visualization
- Use Axes3D for 3D plotting
- Implement proper mouse interaction in 3D space
- Consider performance implications of 3D rendering

#### HDBSCAN Configuration
- Default min_cluster_size: 15
- Default min_samples: 10
- Add option to adjust these parameters
- Handle noise points appropriately

#### Iterative Approach
- Use cluster_ids to determine which points to re-embed
- Temporarily mask other data points during re-embedding
- Provide option to return to full dataset view

This plan addresses all the requirements specified: removing IAN, adding 3D visualization, implementing HDBSCAN clustering, prioritizing important features for RGC classification, and creating an iterative selection/re-embedding capability.

>>>>>>> b42ffbe (fixing feature rextratction)