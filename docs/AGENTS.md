# AGENTS.md — developer rules

Read this document before you change code.

Reading order: `README.md` → `CLAUDE.md` → this file → `HANDOFF.md` →
`docs/PLAN.md` (fragile zones). Then write the failing test. Then implement.

RGCViewer is a PyQt5 / pyqtgraph desktop GUI. A scientist uses it to inspect
spike-sorted retinal ganglion cell (RGC) recordings and to assign clusters
to types.

Two ID spaces exist in a hybrid dataset:

- **Kilosort** — `spike_times.npy`, `spike_clusters.npy`, `templates.npy`.
  IDs are 0-indexed.
- **Vision** — `.neurons`, `.ei`, `.sta`, `.params`. IDs are 1-indexed.

`DataManager.is_vision_only` is `True` only when no Kilosort data was loaded.

Start the app with `python main.py` from environment `rgcviewer`. The window
opens empty. Do not reopen the last run at start.

---

## 1. The Laws

Laws 1–3 are silent. They raise no exception. They are the most common
source of bugs. Laws 4–5 are standing decisions from 2026-08-12.

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

### LAW 4 — Do not invent geometry for a broken EI

Some older kilosort4 conversions write a 519-wide `.ei` next to a 512-row
`.globals`. `EIReader` takes payload width from the `.ei` file. The plot map
is replaced only when Litke coordinates match that width.

If the EI panel still logs `ei=519, positions=512`, leave the plot blank.
Do not invent or stretch an electrode map to force a draw.

**Regression test:** `tests/unit/test_vision_load_robustness.py`.

---

### LAW 5 — Do not build a selector on a 0×0 figure

A hidden or collapsed matplotlib canvas reports figure size 0×0.
`RectangleSelector.__init__` then raises
`ValueError: 'box_aspect' and 'fig_aspect' must be positive`.

Call `_axes_ready(ax)` first. If it is false, return `None` and retry in
`resizeEvent`. Do not catch the error by creating a dummy 1×1 map.

**Regression test:** `tests/unit/test_live_selectors.py`.

---

## 2. Architecture & Data Pipeline

```
src/
├── analysis/
│   ├── data_manager.py        # Single source of truth. All data, caches, locks.
│   ├── vision_integration.py  # Vision file I/O. LazySTADict / LazyEIDict.
│   ├── visionloader.py        # Vision readers. EI stride from the .ei file.
│   ├── analysis_core.py       # Pure numpy. No Qt. No I/O.
│   └── constants.py           # ISI_REFRACTORY_PERIOD_MS, EI_CORR_THRESHOLD, etc.
├── gui/
│   ├── main_window.py         # Tier 1/2 dispatch, QThread lifecycle, menus.
│   ├── theme.py               # All colors, spacing, light/dark mode constants.
│   ├── panels/                # Thin UI layers. Read DataManager. No cross-panel refs.
│   ├── panels/live_selectors.py  # Rectangle / lasso. Require _axes_ready.
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
| `cluster_df` | Main thread only | Workers emit signals. `_apply_ei_updates()` and `attach_sta_quality_column()` write it. `max_dup_r` is float64 — never write format strings into it. Slice with `.copy()` before column writes. |
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

7. **`plot_ei_waveforms` is the single EI spatial rendering primitive.** Lives in `analysis_core.py`. Both `CellTracerDialog` and the EI panel waveform mode call it. Do not copy-paste waveform rendering logic into panels — extend the function instead. It returns a list of `Line2D` artists; the caller is responsible for removing them (call `.remove()` on each, never `fig.clear()`). Box geometry is auto-derived from electrode spacing — do not hardcode `box_height` or `box_width` in callers.

8. **Start does not load a dataset.** `MainWindow` opens a run only when the caller passes `--kilosort-dir` or a test calls `load_directory`. `recent_paths.last_dataset()` is not used at start. File dialogs still remember the last folder.

---

## 7. Data Formats Reference

| Format | ID space | Shape / access | Gotcha |
|---|---|---|---|
| `spike_times.npy` | **0-indexed** | `(N,)` int64, `np.load(mmap_mode='r')` | Monotonically sorted — do not sort again |
| `spike_clusters.npy` | **0-indexed** | `(N,)` int64, `np.load(mmap_mode='r')` | Parallel to `spike_times` — same index = same spike |
| `templates.npy` | **0-indexed** | `(n_clusters, n_time, n_ch)`, memmapped | May not exist in all KS versions |
| `.neurons` (Vision) | **1-indexed** | `dict[vid → spike_sample_nums]` via `vl.NeuronsReader` | Seed electrodes are also 1-indexed |
| `.ei` (Vision) | **1-indexed** | `(N_electrodes, T)` per cell via `vl.EIReader`. Width comes from the `.ei` file, not from `.globals`. A true 30 µm Litke is 519. A kilosort4 converter can write 519 samples next to a 512-row `.globals` — that pair is a broken file, not a 30 µm array. Do not hardcode 512. Do not invent a map to force a plot (Law 4). |
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
| `vision_eis` | `LazyEIDict` or `dict` | **Yes** | Check `self.vision_available` first. One reader. Do not copy the table. |
| `vision_params` | `VisionCellDataTable` | **Yes** | `if self.vision_params:` |
| `raw_reader` | `PyBinFileReader` | **Yes** | `if self.raw_reader is not None:` |
| `channel_positions` | `np.ndarray (N, 2)` | **Yes** | `if self.channel_positions is not None:` |
| `vision_channel_positions` | `np.ndarray (N, 2)` | **Yes** | Never access directly in panels — use `_resolve_channel_positions()` in `ei_panel` or `_ch_pos()` in `cell_tracer_dialog`. Always `None` in vision-only mode (intentional — `.globals` positions are in `channel_positions` instead, and `_ch_pos()` falls through correctly). Shape is 519 on a 30 µm array (Vision keeps reference channels), 512 on a 60 µm array — **never hardcode either value**. |
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

# Launch the application (empty window; File → Open loads a run)
conda run -n rgcviewer python main.py
# or, after `conda activate rgcviewer`:
python main.py --debug

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

## 14. Prime directives

In order:

1. Read `HANDOFF.md` and the assigned spec before you write code.
2. Write the failing test before you write the implementation.
3. Do not update the UI from a background thread. Use Qt Signals.
4. Do not access `vision_stas[cluster_id]` directly. Translate the ID first (Law 1).
5. Do not add heavy work to Tier 1 of `update_cluster_views()` (Law 2).
6. Do not test math against a warm `.pkl` cache (Law 3).
7. Do not invent EI geometry for a 519/512 mismatch (Law 4).
8. Do not build a selector on a 0×0 figure (Law 5).
9. Do not reopen the last run at start.
10. Do not commit unless the user asks.
11. Do not skip a failing test to make a change look green.
