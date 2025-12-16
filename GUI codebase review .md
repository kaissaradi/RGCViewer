# **Codebase Review: Raw Data & Waveform Analysis**

Date: December 15, 2025  
Focus: Raw .bin/.dat file interaction, performance, and architecture.

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