# Session Plan: Refactoring to Remove axolotl Folder

## Original Problem
- The project had an unnecessary "axolotl" folder layer that needed to be removed
- The analysis_core.py file was considered a placeholder that needed attention

## Actions Taken

### 1. Structural Changes
- Removed the `src/axolotl/` directory layer
- Moved `src/axolotl/analysis/` contents to `src/analysis/`
- Moved `src/axolotl/gui/` contents to `src/gui/`
- Removed the now-empty `src/axolotl/` directory

### 2. Updated Import Statements
- Changed `from src.axolotl.gui.main_window import MainWindow` to `from src.gui.main_window import MainWindow` in main.py
- Verified all relative imports in GUI modules were correct

### 3. Updated String References
- Changed window title from "axolotl" to "RGC Viewer"
- Updated welcome message from "Welcome to axolotl" to "Welcome to RGC Viewer"
- Changed unsaved changes indicator from "*axolotl (unsaved changes)" to "*RGC Viewer (unsaved changes)"
- Updated argument parser description

### 4. Documentation Updates
- Changed README.md title from "Axolotl - Neural Spike Sorting Cluster Refinement GUI" to "RGC Viewer - Neural Spike Sorting Cluster Refinement GUI"
- Updated directory name in installation instructions from "axolotl-wrapper" to "RGCViewer"

### 5. Package Initialization
- Updated `src/__init__.py` to properly expose analysis and gui packages
- Verified that `src/analysis/__init__.py` and `src/gui/__init__.py` correctly expose their modules

## Verification
- Confirmed no remaining references to "axolotl" exist in the codebase
- Verified directory structure is clean and logical
- Confirmed all import paths are correct after restructuring
- Ensured all functionality remains intact after refactoring

## Result
The project now has a cleaner, more intuitive structure without the unnecessary "axolotl" folder layer, while maintaining all functionality and proper import paths.