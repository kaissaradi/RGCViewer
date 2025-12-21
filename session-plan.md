Nice — below is a precise, actionable checklist + file-by-file plan describing **what to add, change, or remove**, and why. I’ll show the exact places to modify (function names / classes / variables) and include the code snippets you can paste. This will let you implement the global right-side population split pane so the **Population Split View** checkbox controls a pane that appears to the right of the entire tab area (not only the STA tab).

---

# Summary of changes (high level)

1. **Add a global right-hand population context widget** (contains `pop_mosaic_canvas`) and place it in a `QSplitter` together with `analysis_tabs`.
   *File:* `main_window.py` (`_setup_ui()` / constructor).
2. **Hook the existing Population Split View checkbox** to toggle that global pane (show/hide and set splitter sizes).
   *File:* `main_window.py` (`toggle_population_split_view()`).
3. **Ensure plotting draws into the new canvas** when the global population view is enabled. If needed, make `plotting.draw_population_rfs_plot` accept a `canvas` parameter and call `.draw()` on it.
   *File:* `plotting.py`.
4. **Trigger population redraws on selection changes** (tree selection / cluster selection) so when a user selects a group the right pane updates.
   *File:* `callbacks.py` (or wherever selection change handlers live).
5. (Optional) **Remove or stop relying on STA-only context** if you no longer want an STA-only split; otherwise leave STA split intact and independent.

---

# Files to edit

* `main_window.py` ← main changes (create widgets, toggle function)
* `plotting.py` ← ensure plotting routine supports external canvas
* `callbacks.py` ← make selection-change events update the new pop canvas
* (Optional) `sta-related` code locations — likely in `main_window.py` already — small adjustments if you want to deprecate STA-only split

---

# Detailed edits (copy/paste ready)

> **Before you begin:** back up the files or use version control.

---

## 1) `main_window.py` — Add global right-side splitter and population widget

### Where

* Inside your `MainWindow.__init__()` make sure `self.population_view_enabled = False` (if not already).
* In `_setup_ui()` (or wherever you build the right pane / tabs), create the new widgets and the `QSplitter` that will hold `self.analysis_tabs` and `self.pop_context_widget`.

### Add / Replace snippet

Find where you create the `analysis_tabs` and the right pane. Replace that portion with the block below (or adapt it into the existing structure). This keeps your existing tabs but nests them into `self.right_splitter` so the right-side pop widget is available for *all tabs*:

```python
# --- ensure attribute exists in __init__ ---
self.population_view_enabled = False

# --- imports required at top of file ---
# from PyQt6.QtWidgets import QSplitter, QVBoxLayout, QWidget, QCheckBox
# from PyQt6.QtCore import Qt
# from gui.widgets import MplCanvas   # or wherever your canvas class lives

# --- in _setup_ui() where you build the right pane / tabs ---

# ---- existing tab widget ----
self.analysis_tabs = QTabWidget()
# ... add your tabs here as before ...
# e.g. self.analysis_tabs.addTab(self.standard_tab, "Standard Plots")

# Put the checkbox in the corner (if you already have one, reuse it):
self.pop_view_checkbox = QCheckBox("Population Split View")
self.pop_view_checkbox.setChecked(False)
self.pop_view_checkbox.toggled.connect(self.toggle_population_split_view)
self.analysis_tabs.setCornerWidget(self.pop_view_checkbox, Qt.Corner.TopRightCorner)

# --- NEW: population context widget (right side) ---
self.pop_context_widget = QWidget()
pop_layout = QVBoxLayout(self.pop_context_widget)
pop_layout.setContentsMargins(4, 4, 4, 4)
pop_layout.setSpacing(4)

# Make a plotting canvas for population mosaic
self.pop_mosaic_canvas = MplCanvas(width=6, height=6, dpi=100)
pop_layout.addWidget(self.pop_mosaic_canvas)

# (Optional) attach controls below the canvas e.g. filter dropdowns:
# control_row = QHBoxLayout()
# pop_layout.addLayout(control_row)

# --- NEW: right-side splitter containing tabs and pop widget ---
self.right_splitter = QSplitter(Qt.Orientation.Horizontal)
self.right_splitter.addWidget(self.analysis_tabs)
self.right_splitter.addWidget(self.pop_context_widget)

# Start hidden by default
self.pop_context_widget.hide()
self.right_splitter.setSizes([1200, 0])  # initial ratio: all space to tabs

# Now insert right_splitter into the main layout in place of analysis_tabs
# e.g., right_pane_layout.addWidget(self.right_splitter)
```

**Notes:**

* Replace `MplCanvas` import path with your actual canvas class. The canvas must have `.fig` and `.draw()` like your other canvases.
* If you already created a checkbox elsewhere (you mentioned you have one), rewire it to `self.pop_view_checkbox` or ensure its `.toggled` signal calls `toggle_population_split_view`.

---

## 2) `main_window.py` — Replace / create `toggle_population_split_view()` function

### Where

Replace the old function that only showed the STA split. Add this global toggler:

```python
def toggle_population_split_view(self, checked: bool):
    """Toggles the global population context pane (right side)."""
    self.population_view_enabled = bool(checked)

    if checked:
        # show the right-hand population widget
        self.pop_context_widget.show()

        # Expand it to a sensible size (give about 20-30% to the right pane)
        total = sum(self.right_splitter.sizes()) or 1400
        left_size = max(int(total * 0.75), 400)
        right_size = total - left_size
        self.right_splitter.setSizes([left_size, right_size])

        # If a cluster/cell is selected, draw its population mosaic immediately
        selected = None
        try:
            selected = self._get_selected_cluster_id()  # adapt to your selector fun
        except Exception:
            selected = None

        # Call plotting routine with explicit canvas
        import plotting
        plotting.draw_population_rfs_plot(main_window=self, selected_cell_id=selected,
                                         canvas=self.pop_mosaic_canvas)
    else:
        # hide it
        self.pop_context_widget.hide()
        # collapse the right column completely
        self.right_splitter.setSizes([sum(self.right_splitter.sizes()), 0])
```

**Notes:**

* Replace `_get_selected_cluster_id()` with whatever method you already have to get the currently-selected group/cell in the tree. If you don't have such method, later in `callbacks.py` we will show how to call the plotting function on selection change.
* Import path for `plotting` depends on your package structure. Use existing references in your project.

---

## 3) `plotting.py` — Accept a canvas parameter & use it

### Goal

Let the population plotting function draw into `self.pop_mosaic_canvas`. Ensure it can accept an explicit `canvas` argument or use `main_window.pop_mosaic_canvas` when available.

### Changes

Open `plotting.py`. Find `draw_population_rfs_plot` (or whatever function draws population mosaics). Modify signature to accept `canvas=None` and to prefer the passed `canvas`. Example:

```python
def draw_population_rfs_plot(main_window, selected_cell_id=None, canvas=None, **kwargs):
    """
    Draw population receptive fields / mosaics into the provided canvas.
    If canvas is None, fall back to main_window.pop_mosaic_canvas or main_window.rf_canvas.
    """
    if canvas is None:
        canvas = getattr(main_window, 'pop_mosaic_canvas', None) or getattr(main_window, 'rf_canvas', None)

    if canvas is None:
        # no canvas available — bail gracefully
        return

    fig = canvas.fig
    fig.clear()
    ax = fig.add_subplot(111)
    # ... produce plot on ax using your existing plotting logic ...
    # e.g., ax.imshow(...)

    canvas.draw()
```

**Notes:**

* Use the same plotting internals you already have, just route output to `canvas.fig` and call `canvas.draw()` at the end.
* Keep previous function names and parameters, just add `canvas` so other calls remain compatible.
* If multiple subplots are needed, create them on `fig`.

---

## 4) `callbacks.py` — Redraw population when selection changes

### Goal

When user selects different groups in the tree or the cell list, the global pop pane should update.

### Changes

Find the selection-change callback (e.g. `on_tree_selection_changed`, `on_cluster_selected`, or similar). Add a call to the plotting function when `population_view_enabled` is true:

```python
def on_cluster_selected(self, cluster_id, *args, **kwargs):
    # existing logic...
    # now update population pane if enabled
    if getattr(self, 'population_view_enabled', False):
        import plotting
        plotting.draw_population_rfs_plot(main_window=self, selected_cell_id=cluster_id,
                                         canvas=self.pop_mosaic_canvas)
```

If the selection callback lives in a controller class rather than `MainWindow`, adapt to call `main_window.pop_mosaic_canvas`.

**Notes:**

* If you already have code that updates STA-only context on selection, keep both but avoid duplicate canvases: prefer updating global pop canvas when `population_view_enabled`.

---

## 5) Optional: migrate STA-only context to global pop pane (if desired)

If you want to remove duplicated canvases and only use the global population pane:

* Remove the `sta_context_widget` or stop creating `sta_mosaic_canvas`.
* Update STA tab to call `draw_population_rfs_plot(..., canvas=self.pop_mosaic_canvas)` when users click the STA tab or request population visualization.
* Otherwise keep STA split intact for tab-scoped context.

---

# Minor housekeeping / imports

At the top of `main_window.py` add (if not present):

```python
from PyQt6.QtWidgets import QSplitter, QVBoxLayout, QWidget, QCheckBox
from PyQt6.QtCore import Qt
# import your canvas class
from gui.widgets import MplCanvas   # adapt path to your project
```

# Testing checklist (run after edits)

1. Launch the GUI.
2. Confirm the checkbox "Population Split View" sits in the top-right corner of the tab widget.
3. Click the checkbox:

   * The right pane should appear (populated with the empty canvas) and space should be reallocated.
   * If a cluster/cell is selected, the population mosaic plot should be rendered into the right canvas.
4. Switch to other tabs — the right pane remains visible and populated.
5. Change selection in the tree/list — the right pane updates.
6. Uncheck the box — the right pane hides and tabs reclaim space.
7. Verify STA tab still works as before (if you kept STA-specific context).

---

# Troubleshooting notes & tips

* If the right pane flashes but no plot appears: ensure `plotting.draw_population_rfs_plot` calls `canvas.draw()` and that the canvas uses the same Matplotlib backend as your other canvases.
* If sizes don't settle nicely, call `QApplication.processEvents()` after `self.right_splitter.setSizes(...)` to force layout update.
* If you see duplicated canvases or race conditions when switching tabs quickly, consider guarding redraws with `if self.pop_context_widget.isVisible(): ...`.
* Keep `self.population_view_enabled` attribute available (e.g., set in `__init__`) to avoid attribute errors.

---

# What to remove (if you choose to centralize)

* `sta_context_widget` and related STA-only canvas code (only if you no longer want STA-only pane).
* Any code that toggles STA-only splitter when the population checkbox is clicked (we want checkbox → global pane only).

---

# Final checklist before commit

* [ ] main_window.py: create `self.pop_context_widget`, `self.pop_mosaic_canvas`, `self.right_splitter`; update `_setup_ui()` and `toggle_population_split_view`.
* [ ] plotting.py: ensure `draw_population_rfs_plot` takes optional `canvas` and calls `canvas.draw()`.
* [ ] callbacks.py: call the plotting function on selection change when `population_view_enabled` is True.
* [ ] run app and test the 6-step testing checklist above.
* [ ] (optional) remove or keep STA-only UI as per your design decision.

---

If you want, I can now:

* produce a concrete patch/diff for `main_window.py` (I have the file you uploaded), showing exact lines to replace, or
* directly generate the full updated `draw_population_rfs_plot` body matching the style of your existing plotting code if you paste the current function here.

Which of those would you like?
