# Axolotl - Neural Spike Sorting Cluster Refinement GUI

A high-performance GUI for refining and analyzing neural spike sorting clusters from Kilosort output.

## Installation and Usage

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd axolotl-wrapper
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    conda create --name rcg-viewer python=3.9
    conda activate rcg-viewer
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python main.py
    ```

## Project Log

### ✅ Completed

* **Phase 1: Code Modularization**
    * The original `gui.py` and `cleaning_utils_cpu.py` files have been successfully refactored into a more organized and scalable structure.
    * The new structure includes: `main.py`, `data_manager.py`, `analysis_core.py`, and a `gui/` directory for UI components.

* **Phase 2: Vision File Integration (Foundation)**
    * Created `vision_integration.py` to handle loading of `.ei`, `.sta`, and `.params` files.
    * Updated the `DataManager` to store loaded Vision data.
    * Added a "Load Vision Files" menu option to the UI.
    * Refactored the entire `gui` module into smaller, single-responsibility files (`main_window.py`, `workers.py`, `plotting.py`, `callbacks.py`) to prepare for new features.
    * Added a new plotting function `plot_vision_rf` to `analysis_core.py` for visualizing receptive fields.

### 📝 To-Do

* **Phase 2: Vision File Integration (Visualization)**
    * Connect the loaded Vision data to the UI so that the "Spatial Analysis" tab displays the receptive field plot when available.

* **Phase 3: UI/UX Enhancements**
    * Implement a top-level tabbed interface to manage multiple datasets.
    * Make the left-hand cluster list panel collapsible.
    * Enable multi-cell selection in the cluster list.
    * Update plotting views to display data for multiple selected cells (overlaid waveforms, grid of RFs).

* **Phase 4: Future Features**
    * Add a hierarchical `QTreeView` for advanced cell classification.
