# **Complete Development Plan: Raw Data & Waveform Analysis GUI**

Date: December 15, 2025
Focus: Raw .bin/.dat file interaction, performance, architecture, and GUI redesign.

## **1\. Critical Performance Issues**

### **The "Method B" File Handle Problem**

Severity: High
Location: analysis/analysis\_core.py \-\> extract\_snippets
The function extract\_snippets instantiates a **new** np.memmap every time it is called. This acts as a file open operation. Since this function is called inside worker threads for every cluster selected or refined, it creates significant and unnecessary I/O overhead.

\# analysis\_core.py
def extract\_snippets(dat\_path, spike\_times, ...):
    \# This line opens the file from disk every single time
    raw\_data \= np.memmap(dat\_path, dtype=dtype, mode='r').reshape(-1, n\_channels)

**Impact:**

* **FeatureWorker (workers.py):** Calls extract\_snippets(str(self.data\_manager.dat\_path), ...) for every cluster selection. This makes browsing clusters sluggish.
* **RefinementWorker (workers.py):** Calls refine\_cluster\_v2, which internally calls extract\_snippets, again reopening the file.

## **2\. Architectural Inconsistencies**

### **Dual Raw Data Access Patterns**

There are currently two competing ways the app accesses raw data. This splits the logic and prevents centralized optimization.

1. **The "Good" Way (DataManager):**
   * DataManager opens the file *once* in set\_dat\_path.
   * It stores the handle in self.raw\_data\_memmap.
   * Used by RawTraceWorker (for the "Raw Trace" tab).
2. **The "Bad" Way (analysis\_core):**
   * Functions accept a str path to the file.
   * They open their own temporary handles.
   * Used by FeatureWorker (Waveforms) and RefinementWorker.

### **refine\_cluster\_v2 Signature Confusion**

**Location:** analysis/analysis\_core.py

The refine\_cluster\_v2 function has a confusing parameter signature regarding dat\_path.

def refine\_cluster\_v2(spike\_times, dat\_path, ...):
    if isinstance(dat\_path, np.ndarray):
        snips \= dat\_path  \# \<--- LOGIC GAP
    elif isinstance(dat\_path, str):
        snips \= extract\_snippets(dat\_path, spike\_times, window)

**The Inconsistency:**

* If you pass a str (filepath), it acts as a path to the **raw binary file**.
* If you pass an np.ndarray (memmap), the code assumes it is **already extracted snippets** (snips), not the raw data trace.
* **Consequence:** You cannot pass the cached DataManager.raw\_data\_memmap to this function without refactoring, forcing you to pass the string path and incur the I/O penalty.

## **3\. Redundant & Dead Code**

### **Dead Plotting Stub**

**Location:** gui/plotting.py

The function update\_raw\_trace\_plot is defined but explicitly does nothing. It should be removed to avoid confusion.

def update\_raw\_trace\_plot(main\_window, cluster\_id):
    """
    This stub is maintained for compatibility but does nothing
    """
    pass

### **Redundant Logic**

**Location:** data\_manager.py vs analysis\_core.py

* DataManager.get\_raw\_trace\_snippet: Fetches continuous raw data.
* analysis\_core.extract\_snippets: Fetches discontinuous raw data (spikes).
* **Issue:** Both implement the logic for slicing the memmap. DataManager should likely own the logic for *all* raw data access to ensure the single file handle is used.

## **4\. Code Quality & Maintenance**

### **Hardcoded Constants**

**Location:** data\_manager.py

The conversion factor for microvolts is hardcoded. This will break if you use a probe with different gain settings or bit depth.

self.uV\_per\_bit \= 0.195  \# Hardcoded in \_\_init\_\_

* **Recommendation:** Load this from params.py (Kilosort params) or constants.py.

### **Excessive Debug Printing**

**Location:** All files, specifically data\_manager.py

The code is littered with print("\[DEBUG\] ...") statements.

* **Example:** print(f"\[DEBUG\] About to call vision\_integration.load\_vision\_data")
* **Recommendation:** Replace with Python's standard logging module or remove them for production cleanup.

### **Data Loading Inefficiency**

**Location:** data\_manager.py \-\> build\_cluster\_dataframe

This function calculates ISI violations for **every single cluster** in a loop during startup.

for i, cluster\_id in enumerate(cluster\_ids):
    isi\_value \= self.\_calculate\_isi\_violations(cluster\_id, ...)
    \# ...

* **Impact:** This linear loop significantly slows down the initial loading of the Kilosort directory, especially for datasets with thousands of clusters.

## **5\. Summary of Recommended Actions**

1. **Consolidate Data Access:** Move extract\_snippets logic into DataManager. Make it use self.raw\_data\_memmap.
2. **Fix Workers:** Update FeatureWorker and RefinementWorker to call DataManager methods instead of analysis\_core file-opening methods.
3. **Refactor Refinement:** Change refine\_cluster\_v2 to accept a raw data source (memmap) OR require DataManager to extract the snippets before calling it.
4. **Cleanup:** Remove the dead plotting code and consolidate uV\_per\_bit into constants.py.

---

## **Status update (actions applied in codebase)**

- `analysis/analysis_core.py`: `extract_snippets` now accepts either a file path or an existing memmap/ndarray and safely handles out-of-bounds spikes. This reduces repeated file opens.
- `analysis/data_manager.py`: added thread-safe `clear_caches()` and a `threading.Lock` protecting `heavyweight_cache`; `get_heavyweight_features` now uses the lock for safe concurrent access.
- `gui/workers.py`: `FeatureWorker` and `RefinementWorker` now prefer `DataManager.raw_data_memmap` (when available) and fall back to the dat file path, avoiding reopening the file on every worker run.
- `gui/main_window.py`: feature-worker stop logic updated to call `.wait()` after `.quit()` to avoid overlapping threads during rapid selection.
- `gui/panels/raw_panel.py`: Implemented reuse of `PlotDataItem` objects to avoid repeated creation/destruction on redraws.

**Panel and UI Refactoring:**
- **Created `gui/panels/standard_plots_panel.py` and `gui/panels/waveforms_panel.py`** as per the development plan.
- **Refactored `gui/main_window.py` UI** to use a `QTabWidget` for analysis panels, with the order: "Standard Plots", "EI Analysis", "STA Analysis", "Raw Waveforms", "Raw Trace".
- **Optimized UI updates**: Implemented `on_tab_changed` logic in `main_window.py` to ensure only the visible panel is updated with new data upon selection, improving responsiveness.
- **Fixed `AttributeError` in `StandardPlotsPanel`**: Corrected `QSplitter` initialization to use `qtpy.QtCore.Qt.Vertical/Horizontal` instead of `pyqtgraph.Qt.Orientation.Vertical/Horizontal`.

**Phase 3: Backend & Data Verification**
- **Verified `analysis/data_manager.py`**: Confirmed `load_kilosort_data` correctly loads `templates.npy` into `self.templates` and `sampling_rate` is available from `_load_kilosort_params`.
- **Verified `gui/workers.py`**: Confirmed `FeatureWorker.run` extracts `raw_snippets` and includes them in the results dictionary emitted to `main_window` for caching.


These changes address the highest-impact issues from the review (#1 and #2): they consolidate data access patterns and remove the repeated memmap reopen overhead. They also implement the new panel-based UI design.

## **Next recommended low-risk steps**

1.  **[In Progress]** Replace verbose `print()` debug lines with `logging` calls and trim noisy debug output.
2.  Refactor `refine_cluster_v2` signature or add a small adapter in `DataManager` so it can supply pre-extracted snippets (longer-term but reduces confusion).
3.  **Refine `WaveformPanel.update_all`**: Currently `update_all` in `WaveformPanel` expects a single `cluster_id`. However, `MainWindow.on_similarity_selection_changed` passes a list of cluster IDs to `self.waveforms_panel.update_all(clusters_to_plot)`. Update `WaveformPanel.update_all` to handle a list of cluster IDs or ensure only a single cluster ID is passed.

---

## **Detailed Development Plan: New Panel Implementation**

### **Phase 1: New Panel Implementation**

#### **1. Create `gui/panels/standard_plots_panel.py**`
This is the new "Home" tab containing the quick diagnostic plots.

*   **Class:** `StandardPlotsPanel(QWidget)`
*   **Layout:** Vertical Splitter containing two Horizontal Splitters (2x2 grid).
*   **Plots:**
    1.  **Template (Top Left):** Plots the Kilosort template (`data_manager.templates`) for the cluster's dominant channel.
    2.  **Autocorrelation (Top Right):** New histogram-based ACG plot.
    3.  **ISI Distribution (Bottom Left):**
        *   **Logic:** Replicate `plotting-old.py`. Use `np.histogram` on ISI differences.
        *   **Aesthetic (The "Old" Look):**
            *   Use `pyqtgraph.plot()` with `stepMode="center"` (or `True`).
            *   Set `fillLevel=0` to fill from the line to the bottom.
            *   Set `brush=(0, 163, 224, 150)` (The semi-transparent blue from your old code).
            *   Add the vertical dashed red line at 2.0ms (or 1.5ms).
    4.  **Firing Rate (Bottom Right):**
        *   **Logic:** Replicate `plotting-old.py`. Bin spikes into 1s bins.
        *   **Aesthetic:**
            *   Apply `scipy.ndimage.gaussian_filter1d` with `sigma=5` (smoothness from old code).
            *   Use `pen='y'` (Yellow) or `#ffeb3b` for high contrast on dark background.

#### **2. Create `gui/panels/waveforms_panel.py**`
This panel is now dedicated purely to the raw snippet visualization (the "cloud").

*   **Class:** `WaveformPanel(QWidget)`
*   **Layout:** Single `pg.PlotWidget`.
*   **Logic (`update_all`):**
    *   Retrieve `raw_snippets` from `data_manager.ei_cache` (computed by `FeatureWorker`).
    *   **Optimization:** Use the `np.nan` connection trick to plot all snippets as a single line item for performance.
    *   **Aesthetic:**
        *   Snippets: White with very low alpha (e.g., `(255, 255, 255, 15)`).
        *   Mean: Thick Teal/Cyan line on top (`width=3`).

---

### **Phase 2: Main Window Integration**

#### **3. Modify `gui/main_window.py**`
*   **Imports:** Import the two new panel classes.
*   **`_setup_ui` Method:**
    *   Remove the old splitter layout in the right pane.
    *   Initialize a `QTabWidget` for the right pane.
*   **Tab Order:**
    1.  "Standard Plots" (`StandardPlotsPanel`)
    2.  "EI Analysis" (`EIPanel` - existing)
    3.  "STA Analysis" (Existing STA logic wrapped in a widget)
    4.  "Raw Waveforms" (`WaveformPanel`)
    5.  "Raw Trace" (`RawPanel` - existing)
*   **`_draw_plots` Method:**
    *   Call `self.standard_panel.update_all(cluster_id)`.
    *   Call `self.waveforms_panel.update_all(cluster_id)`.
    *   Ensure other panels (`ei_panel`, `raw_panel`) update only if their tab is active (optimization).

---

### **Phase 3: Backend & Data**

#### **4. Verify `analysis/data_manager.py**`
*   Ensure `load_kilosort_data` correctly loads `templates.npy` into `self.templates`.
*   Ensure `sampling_rate` is available (needed for ms/Hz conversion).

#### **5. Verify `gui/workers.py**`
*   Ensure `FeatureWorker.run` extracts `raw_snippets` and includes them in the results dictionary emitted to `main_window`.

---

### **Aesthetic Replication Guide (For Developer)**

To exactly match the **ISI** and **Firing Rate** from `plotting-old.py`, use these snippet configurations in `StandardPlotsPanel`:

**ISI Plot (Blue Filled Step):**

```python
# Calculate
isis_ms = np.diff(np.sort(spikes)) / sr * 1000
y, x = np.histogram(isis_ms, bins=np.linspace(0, 50, 101))

# Plot
self.isi_plot.plot(x, y, stepMode="center", fillLevel=0,
                   brush=(0, 163, 224, 150),  # The specific "Old" Blue
                   pen=pg.mkPen(color='#33b5e5', width=2))
```

**Firing Rate (Yellow Smooth Line):**

```python
# Calculate
bins = np.arange(0, total_duration + 1, 1)
counts, _ = np.histogram(spikes_sec, bins=bins)
rate = gaussian_filter1d(counts.astype(float), sigma=5) # sigma=5 for "Old" smoothness

# Plot
self.fr_plot.plot(bins[:-1], rate, pen=pg.mkPen('y', width=2)) # 'y' = Yellow
```