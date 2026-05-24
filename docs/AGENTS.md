# AGENTS.md — RGCViewer Developer Rulebook

> You are an AI agent acting as a core developer on this project.
> Read this document **in full** before touching any code.
> Reading order: **AGENTS.md → PLAN.md (Fragile Zones) → your assigned spec → write the failing test → implement.**

---

## 0. What This App Is

RGCViewer is a **PyQt5/pyqtgraph desktop GUI for spike sorting quality control** of retinal ganglion cell (RGC) electrophysiology data. A 512-electrode Litke MEA records from hundreds of neurons simultaneously. The spike sorter Kilosort assigns voltage events to putative neurons called *clusters*. RGCViewer lets a scientist scroll through thousands of clusters rapidly, inspect autocorrelograms, receptive fields (STAs), electrical images (EIs), and raw waveforms, and mark cells as good / duplicate / noise.

Data comes from **two coexisting pipelines in the same dataset**:

- **Kilosort** — produces `spike_times.npy`, `spike_clusters.npy`, `templates.npy`. IDs are **0-indexed integers**.
- **Vision** — produces `.neurons`, `.ei`, `.sta`, `.params` files. IDs are **1-indexed integers**.

Both ID spaces coexist in every hybrid dataset. The flag `DataManager.is_vision_only` is `True` only when no Kilosort data was loaded.

---

## 1. The Three Laws

These three failure modes are **silent** — they raise no exception and produce no visible error. They are the most common source of bugs in this codebase.

---

### LAW 1 — The Vision/Kilosort ID Offset

**Vision IDs are 1-indexed. Kilosort cluster IDs are 0-indexed.**

In a hybrid dataset, `cluster_id = 5` in Kilosort corresponds to `vision_id = 6` in Vision files.

```python
# THE ONLY SAFE TRANSLATION — copy this exactly, never re-derive it:
vid = cluster_id if getattr(self, 'is_vision_only', False) else cluster_id + 1

# Then access Vision data using vid, not cluster_id:
if self.vision_stas and vid in self.vision_stas:
    sta = self.vision_stas[vid]
```

This translation is already implemented in `DataManager.get_cell_physics()`. **Call that method. Do not re-implement the offset anywhere else.**

**What breaks without it:** `vision_stas[cluster_id]` silently returns the *previous* cell's STA. The panel renders incorrect data with no error.

**Regression test:** `test_get_cell_physics_vision_id_offset` — both parametrize branches (hybrid and vision-only) must pass after any change to `get_cell_physics()` or any code that accesses `vision_stas`, `vision_eis`, or `vision_params`.

---

### LAW 2 — The Main Thread Must Never Block

`MainWindow.update_cluster_views()` is called on **every keypress** during rapid scrolling. It has two tiers with hard rules.

**Tier 1 — Immediate (runs synchronously on the main thread):**

| Allowed | Forbidden |
|---|---|
| Reading from `ei_cache` (already in RAM) | Any disk I/O |
| Population plot hot-swap if state already exists | Any scan of full spike arrays |
| Updating `_pending_cluster_id` | Any `panel.update_all()` |
| Restarting `selection_timer` | Any worker spawn |
| | Any `DataManager` method that acquires a lock |

**Tier 2 — Debounced (fires 25ms after scrolling stops, via `_process_selection()`):**

Everything is allowed here. This is where workers are spawned, panels are rebuilt, and expensive `DataManager` calls are made.

**What breaks without it:** Adding any forbidden operation to Tier 1 freezes the UI during rapid scrolling. Reproducible by holding the down arrow key in the cluster list.

---

### LAW 3 — Cache Bypass in Tests

`DataManager.__init__` calls `_load_standard_plot_cache_from_disk()` immediately. If `standard_plot_cache.pkl` exists in the directory passed to `DataManager(kilosort_dir=...)`, the cache is loaded and **`_compute_standard_plots()` will never be called during the test**. The test passes a green checkmark while proving nothing.

```python
# WRONG — may load a stale .pkl and silently skip all math:
def test_acg_logic():
    dm = DataManager(kilosort_dir="/real/lab/path")

# CORRECT — empty tmp_path has no .pkl, math is forced to run:
def test_acg_logic(tmp_path):
    dm = DataManager(kilosort_dir=str(tmp_path))
```

Use the `cache_cleared_data_manager` fixture when you need real data files but must guarantee math re-runs.

---

## 2. Architecture & Data Pipeline

```
src/
├── analysis/
│   ├── data_manager.py        # Single source of truth. All data, caches, locks.
│   ├── vision_integration.py  # Vision file I/O. LazySTADict lives here.
│   ├── analysis_core.py       # Pure numpy. No Qt. No I/O.
│   └── constants.py           # ISI_REFRACTORY_PERIOD_MS, EI_CORR_THRESHOLD, etc.
├── gui/
│   ├── main_window.py         # Tier 1/2 dispatch, QThread lifecycle, menus.
│   ├── theme.py               # All colors, spacing, light/dark mode constants.
│   ├── panels/                # Thin UI layers. Read DataManager. No cross-panel refs.
│   └── workers/
│       └── workers.py         # All background computation. Emits Qt Signals.
tests/
├── conftest.py                # All shared fixtures. Read before writing any test.
├── unit/                      # Pure logic, no Qt, no disk.
└── integration/               # Qt signal/slot, panel rendering with qtbot.
```

**The layer contract:**
- `vision_integration.py` produces raw data objects → consumed only by `data_manager.py`
- `data_manager.py` is the only data source for all panels and workers
- `workers/` reads from `DataManager`, emits Qt Signals → received by `main_window.py` slots
- `panels/` reads from `DataManager` via slot arguments → renders with `pyqtgraph`
- **Panels never reference other panels. Ever.**

---

## 3. The Five Caches

| Cache | Attribute | Key | Persisted to | Lock | Written by |
|---|---|---|---|---|---|
| Standard plots | `standard_plot_cache` | `cluster_id` | `standard_plot_cache.pkl` | `_standard_plot_lock` | `_compute_standard_plots()` |
| Physics / UMAP features | `feature_cache` | `cluster_id` | `feature_cache.pkl` | `_feature_lock` | `get_cell_physics()` |
| Waveform / EI snippets | `ei_cache` | `cluster_id` | Not persisted | Main thread only | `on_features_ready()` slot |
| ISI violations | `isi_cache` | `(cluster_id, refractory_ms)` | Not persisted | None | `_calculate_isi_violations()` |
| Spatial / heavyweight | `heavyweight_cache` | `cluster_id` | Not persisted | `_heavyweight_lock` | `get_heavyweight_features()` |

Physics features (`timecourse`, `rf_area`, `ellipticity`, `acg`) → `feature_cache`.
ISI/ACG/FR dicts → `standard_plot_cache`.
Do not mix them.

---

## 4. QThread Lifecycle — The Canonical Pattern

```python
# Step 1: Always clean up the PREVIOUS thread first.
# Skipping this causes double-draw bugs where stale results appear over fresh ones.
self._cleanup_thread('my_worker_thread')

# Step 2: Create thread and worker.
self.my_worker_thread = QThread()
self.my_worker = MyWorker(self.data_manager, cluster_id)

# Step 3: Move worker to thread BEFORE connecting any signals.
self.my_worker.moveToThread(self.my_worker_thread)

# Step 4: Connect signals, then start.
self.my_worker.result_ready.connect(self.on_result_ready)
self.my_worker.error.connect(lambda msg: self.status_bar.showMessage(msg, 4000))
self.my_worker_thread.started.connect(self.my_worker.run)
self.my_worker_thread.start()
```

### Stale Result Guard — Required in Every Result Slot

```python
def on_result_ready(self, cluster_id: int, result: dict):
    # NON-NEGOTIABLE. Copy exactly into every result slot.
    # Without this, stale data renders over the correct cell's panel.
    if cluster_id != self._get_selected_cluster_id():
        logger.debug("Discarding stale result for cluster %d", cluster_id)
        return
    # safe to update UI from here
    self.data_manager.ei_cache[cluster_id] = result
    self._draw_plots(cluster_id, result)
```

---

## 5. Thread Ownership of DataManager Attributes

| Attribute | Owner | Rule |
|---|---|---|
| `cluster_df` | Main thread only | Background workers emit `ei_updates_ready` signal; `_apply_ei_updates()` slot writes it |
| `spike_times`, `spike_clusters` | Read-only after load | Immutable memmaps. Never assign or sort in place. |
| `standard_plot_cache` | Any thread | Always acquire `_standard_plot_lock` |
| `feature_cache` | Any thread | Always acquire `_feature_lock` |
| `heavyweight_cache` | Any thread | Always acquire `_heavyweight_lock` |
| `ei_cache` | Main thread only | Written only inside `on_features_ready()` slot |
| `vision_stas`, `vision_eis` | Read-only after load | Set once by loader, then never reassigned |

---

## 6. Architecture Invariants

Things that look like implementation details but are actually load-bearing contracts.

1. **`spike_times` is monotonically sorted.** Kilosort writes spikes in time order. Do not sort it again.

2. **`cluster_spike_indices` is the O(1) lookup path.** Always use `self.get_cluster_spike_indices(cluster_id)`. Never use `np.where(self.spike_clusters == cluster_id)` in any hot path — it is O(N) over millions of spikes.

3. **One `np.unique` scan per load.** `load_kilosort_data()` produces `_spk_unique_cls`, `_counts`, `_spk_sorted_cls`, `_spk_t` in a single pass. Do not add a second scan — it doubles load time and breaks count assumptions downstream.

4. **`LazySTADict` is one instance per dataset.** It holds an open file handle. Never instantiate more than one per Vision directory.

5. **EI correlation is skipped in vision-only mode.** The guard in `_compute_ei_correlations_if_needed()` prevents RAM exhaustion on 512-electrode datasets. Never remove it.

6. **Atomic pkl write.** `_save_pickle_with_fallback()` uses `tempfile + os.replace()`. Never replace with `pickle.dump(open(path, 'wb'))` — a crash mid-write would corrupt the cache file silently.

---

## 7. Data Formats Reference

| Format | ID space | Shape / access | Gotcha |
|---|---|---|---|
| `spike_times.npy` | **0-indexed** | `(N,)` int64, `np.load(mmap_mode='r')` | Monotonically sorted — do not sort again |
| `spike_clusters.npy` | **0-indexed** | `(N,)` int64, `np.load(mmap_mode='r')` | Parallel to `spike_times` — same index = same spike |
| `templates.npy` | **0-indexed** | `(n_clusters, n_time, n_ch)`, memmapped | May not exist in all KS versions |
| `.neurons` (Vision) | **1-indexed** | `dict[vid → spike_sample_nums]` via `vl.NeuronsReader` | Seed electrodes are also 1-indexed |
| `.ei` (Vision) | **1-indexed** | `(512 electrodes, 201 time)` per cell via `vl.EIReader` | `ei_corr()` expects exactly this shape |
| `.sta` (Vision) | **1-indexed** | `LazySTADict[vid]` → obj with `.red/.green/.blue` | Shape: `(height, width, n_frames)` |
| `.params` (Vision) | **1-indexed** | `VisionCellDataTable` via `vl.ParametersFileReader` | `get_stafit_for_cell(vid)` may return `None` |
| `.globals` (Vision) | N/A | `(N_electrodes, 2)` xy positions | `ch = seed_electrode - 1` to convert to 0-indexed |
| Litke `.bin` | N/A | Channels-major via `PyBinFileReader(dir, is_row_major=True)` | Row 0 is TTL — `litke_idx = kilosort_ch + 1` |
| Flat `.dat` | N/A | `(n_samples, n_ch)` int16 memmap | Legacy fallback only |

---

## 8. DataManager Attribute Nullability

| Attribute | Type | Can be `None`? | Safe access pattern |
|---|---|---|---|
| `spike_times` | `np.ndarray` memmap | No (after load) | Direct access |
| `cluster_df` | `pd.DataFrame` | No (may be empty) | Check `.empty` before use |
| `vision_stas` | `LazySTADict` | **Yes** | `if self.vision_stas and vid in self.vision_stas:` |
| `vision_eis` | `dict` | **Yes** | Check `self.vision_available` first |
| `vision_params` | `VisionCellDataTable` | **Yes** | `if self.vision_params:` |
| `raw_reader` | `PyBinFileReader` | **Yes** | `if self.raw_reader is not None:` |
| `channel_positions` | `np.ndarray (N, 2)` | **Yes** | `if self.channel_positions is not None:` |
| `templates` | `np.ndarray` memmap | **Yes** | `if hasattr(self, 'templates') and self.templates is not None:` |
| `is_vision_only` | `bool` | No | Defaults to `False` |
| `sampling_rate` | `float` | No | Defaults to `30000.0` |

---

## 9. Test Fixture Selection

| Situation | Fixture | Why |
|---|---|---|
| Testing pure logic, no Qt, no disk | `mock_dm` | Fastest — no event loop |
| Testing Qt signal/slot wiring | `make_main_window + qtbot` | Exercises actual Qt plumbing |
| Verifying math actually runs (not cached) | `cache_cleared_data_manager` | Copies real data, deletes all `.pkl` files |
| Real-format edge cases | `real_data_manager` | Auto-skips if `/mnt/lab/Array-data/` unmounted |
| Disk cache round-trip | `tmp_path` + manual `.pkl` write | Isolated, no shared state |
| Concurrency / race condition | `threading.Event` barriers | Deterministic thread interleaving |

Do not use `real_data_manager` to verify math. It may already have a warm cache. Use `cache_cleared_data_manager` instead.

---

## 10. Environment & Essential Commands

```bash
# --- TESTING ---

# Full test suite (always run before pushing)
conda run -n rgcviewer python -m pytest tests/ -v

# Fast unit tests only (no real data, no Qt)
conda run -n rgcviewer python -m pytest tests/unit/ -v

# Single test function
conda run -n rgcviewer python -m pytest tests/unit/test_autocorrelation.py::test_acg_includes_late_spike_trains -v

# Tests matching a keyword
conda run -n rgcviewer python -m pytest tests/ -k "cache" -v

# With print output visible
conda run -n rgcviewer python -m pytest tests/ -s -v

# Stop after first failure
conda run -n rgcviewer python -m pytest tests/ -x -v

# --- VISUAL REGRESSION ---

# Generate new baselines (run once after intentional visual change)
conda run -n rgcviewer python -m pytest --mpl-generate-path tests/baseline_images/ tests/

# Compare against baselines
conda run -n rgcviewer python -m pytest --mpl tests/

# --- APP ---

# Launch the application
conda run -n rgcviewer python -m src.gui.main_window

# Check installed packages
conda run -n rgcviewer pip list | grep -E "pyqtgraph|hdbscan|visionloader|qtpy"

# --- ENVIRONMENT SETUP (if environment needs rebuilding) ---
conda env create -f environment.yml
conda activate rgcviewer
```

**Real data paths:**

```
Raw Litke:    /mnt/lab/Array-data/raw/20260506A/data009
Sorted/Vision: /mnt/lab/Array-data/sorted/20260506A/chunk10/kilosort2.5
```

---

## 11. Git Protocol & Etiquette

```bash
# Always branch from a fresh main
git checkout main && git pull origin main
git checkout -b feat/your-feature-name

# If your spec touches data_manager.py — rebase frequently:
git fetch origin && git rebase origin/main

# Stage and commit atomically (one logical change per commit)
git add src/analysis/data_manager.py tests/unit/test_feature.py
git commit -m "fix(data_manager): apply +1 offset when translating cluster_id to vision_id"

# Run tests before every push — no exceptions
conda run -n rgcviewer python -m pytest tests/ -v
git push -u origin feat/your-feature-name
```

**Branch prefixes:**

| Prefix | When to use | Example |
|---|---|---|
| `feat/` | New features | `feat/vision-standalone` |
| `fix/` | Bug fixes | `fix/acg-full-recording` |
| `test/` | Tests only, no src changes | `test/vision-id-offset` |
| `chore/` | Docs, deps, linting | `chore/update-agents-doc` |

**Commit message format:** `type(scope): what changed and why`

```bash
# Good
git commit -m "fix(data_manager): apply +1 offset when translating cluster_id to vision_id"
git commit -m "test(acg): add parametrized late-spike regression covering full recording"
git commit -m "feat(umap): add HDBSCAN clustering with K-means fallback"

# Not acceptable
git commit -m "fix bug"
git commit -m "wip"
git commit -m "update"
```

**PR etiquette:**
- One spec per PR. No bundled unrelated changes.
- PRs touching `data_manager.py` must be flagged — other agents may be mid-rebase.
- Do not merge your own PR.
- If a test was failing before your change and is still failing, document it explicitly. Do not `pytest.mark.skip` it to make your PR green.
- Tests must pass locally before you open the PR. CI is not a test runner.

---

## 12. Multi-Agent Concurrency Rules

1. **One Spec = One Branch = One Agent.** Work only on the spec assigned to you.
2. **Domain isolation.** If Agent A owns `UMAPPanel`, Agent B must not touch `umap_panel.py` for any reason.
3. **`DataManager` is the bottleneck.** Any spec touching `data_manager.py` must `git rebase origin/main` before every push.
4. **Test isolation.** Every test must be self-contained. No shared mutable state between tests. No ordering dependencies.
5. **Never silence a failing test.** If you didn't break it, file a separate issue. Do not skip it to make CI green.

---

## 13. Import Discipline

```python
# CORRECT — all Qt imports through qtpy:
from qtpy.QtWidgets import QWidget, QVBoxLayout
from qtpy.QtCore import QThread, Signal, Qt
from qtpy.QtGui import QColor

# WRONG — known violation exists in main_window.py, do not add more:
# from PyQt5.QtGui import QColor
```

Plotting rule: `pyqtgraph` for all dynamic live-data panels. `matplotlib` only for the population RF mosaic and static file exports.

---

## 14. Prime Directives (Summary)

In order, non-negotiable:

1. Read the spec before writing any code.
2. Write the failing test before writing any implementation.
3. Never update UI from a background thread. Use Qt Signals.
4. Never access `vision_stas[cluster_id]` directly. Translate the ID first (Law 1).
5. Never add heavy operations to Tier 1 of `update_cluster_views()`. (Law 2)
6. Never test math logic without bypassing the `.pkl` cache. (Law 3)
7. Never commit code that breaks a currently-passing test.
