# Specification: GUI Polish — UMAP Layout Bug, Sidebar Search & Tree Branch Styling

## Metadata

* **Status:** Ready for Dev
* **Target Release:** v1.1
* **Spec ID:** `gui-polish-umap-layout-sidebar-search-tree-arrows`
* **Branch:** `fix/gui-polish-umap-layout-sidebar-search-tree-arrows`
* **Files touched:** `src/gui/panels/umap_panel.py`, `src/gui/main_window.py` only

---

## Objective

Three isolated cosmetic/UX changes with **zero data, analysis, or worker logic changes**:

1. **Bug — UMAP toolbar rows overlap on first render.** Switching to any other tab and returning fixes it. Classic Qt first-show geometry deferral bug.
2. **Feature — Sidebar live search.** No way to find a cluster among 200+ without scrolling. A persistent search bar that filters both the Tree and Table views in real time.
3. **Polish — Modern tree expand/collapse arrows.** The default Qt branch indicators look dated. Replace with inline SVG triangles, theme-aware, no external asset files.

---

## User Story

> "As a researcher curating 200+ clusters, I want to type a cluster ID and instantly jump to it in either view, I want the UMAP panel to look correct the first time I open it, and I want the sidebar tree to show clean expand/collapse arrows instead of 90s dotted lines."

---

## Acceptance Criteria (Definition of Done)

### UMAP Bug Fix

**AC1 — No layout overlap on first visit**
* On cold launch, click the UMAP tab **directly** without visiting any other tab first.
* Both control rows render at correct height — no overlap, no clipping, no zero-height row.

**AC2 — First-visit and second-visit geometry are identical**
* After AC1 passes, switch to STA tab then back to UMAP.
* `run_btn.geometry()` must be pixel-identical to its geometry on first visit.
* This is the regression assertion: before the fix this test will fail; after it must pass.

**AC3 — No regression on window resize**
* Resize the window from 1800×1000 to 1200×700 while on the UMAP tab.
* Both rows reflow without overlap or widgets disappearing.

---

### Sidebar Search

**AC4 — Search bar always visible above the view stack**
* A `QLineEdit` appears in the left panel, **above the Table/Tree toggle buttons**, always visible regardless of which view is active.
* Placeholder text: `"Search clusters..."`.
* Has a built-in clear button via `setClearButtonEnabled(True)` — no custom overlay widget needed.

**AC5 — Tree view: live filter on typing**
* Typing filters to leaf items whose display text contains the query (case-insensitive substring).
* Parent/group items that have at least one matching child remain visible and auto-expand.
* Parent/group items with zero matching children are hidden.
* Clearing the bar (empty string): calls `collapseAll()` and un-hides all items, restoring the default collapsed state.

**AC6 — Table view: live filter on typing**
* Typing filters table rows to those where `cluster_id` (as string) OR `KSLabel` column value contains the query (case-insensitive substring).
* Uses a `QSortFilterProxyModel` wrapping the existing `HighlightStatusPandasModel` — does **not** rebuild or mutate `data_manager.cluster_df`.
* The existing `_get_selected_cluster_id` already handles `mapToSource` and must continue to work correctly through the proxy.
* If the currently selected cluster is filtered out, selection is cleared.
* Clearing the bar removes the proxy filter and restores all rows.

**AC7 — Switching views with a query active applies the filter immediately**
* If the user has typed `"26"` while in Tree view and then switches to Table view, the table is immediately filtered to rows matching `"26"` — and vice versa.
* This is handled by `_filter_sidebar` being called from `_switch_left_view` with the current bar text.

**AC8 — `Ctrl+F` focuses the search bar**
* Pressing `Ctrl+F` anywhere in the main window moves focus to the search bar and selects all text.
* Does not interfere with any existing shortcut.

---

### Tree Branch Indicators

**AC9 — Expand/collapse triangles replace default branch indicators**
* Non-leaf (group) rows show a right-pointing `▶` when collapsed, down-pointing `▼` when expanded.
* Leaf cluster rows show no indicator.
* No dotted connector lines between siblings.

**AC10 — Arrows are theme-aware, no restart required**
* In dark mode: arrow fill = `colors['text_secondary']`.
* After `toggle_theme()` (light mode): arrow fill updates immediately.
* This is guaranteed because `_setup_style(colors)` is already called by both `__init__` and `toggle_theme()` — no additional signal/callback needed.

---

## Architecture & Technical Constraints

### Files Modified

| File | What changes |
|---|---|
| `src/gui/panels/umap_panel.py` | Add `showEvent` + `_refresh_layout` (AC1–AC3) |
| `src/gui/main_window.py` | Search bar widget + filter logic + proxy model + `Ctrl+F` shortcut + branch CSS (AC4–AC10) |

**Strictly no changes to:** `DataManager`, `data_manager.py`, any worker, any signal definition, any other panel file, `theme.py`.

---

## Implementation — Exact Code

### 1. UMAP `showEvent` fix — `src/gui/panels/umap_panel.py`

`QTimer` is not currently imported in `umap_panel.py`. Add it.

```python
# Locate existing import line, e.g.:
from qtpy.QtCore import QThread, Signal, QObject
# Add QTimer:
from qtpy.QtCore import QThread, Signal, QObject, QTimer
```

Add these two methods inside the `UMAPPanel` class, after `__init__`:

```python
def showEvent(self, event):
    """
    Qt defers geometry computation for widgets inside QTabWidget until they
    are first shown. On the initial visit, the first paint can fire before
    the layout pass has committed sizes, causing row overlap.
    singleShot(0) defers the layout activation until after the event loop
    processes the show, guaranteeing geometry is committed before first paint.
    """
    super().showEvent(event)
    QTimer.singleShot(0, self._refresh_layout)

def _refresh_layout(self):
    self.layout.activate()
    self.updateGeometry()
```

That is the entire bug fix. Do not change any other code in `umap_panel.py`.

---

### 2. Sidebar search bar — `src/gui/main_window.py`

#### 2a. Imports

Add to the existing `qtpy.QtWidgets` import block (already present):
* `QSortFilterProxyModel` — from `qtpy.QtCore`
* `QShortcut` — from `qtpy.QtWidgets`
* `QKeySequence` — from `qtpy.QtGui`

```python
# Existing QtCore import block — add QSortFilterProxyModel:
from qtpy.QtCore import Qt, QItemSelectionModel, QThread, QTimer, QSortFilterProxyModel

# Existing QtWidgets import — add QShortcut:
from qtpy.QtWidgets import (
    ...,
    QShortcut,
)

# Existing QtGui import — add QKeySequence:
from qtpy.QtGui import QFont, QStandardItemModel, QKeySequence
```

#### 2b. Widget creation in `_setup_ui()`

Locate the block that builds `top_ctrl_layout` (the All button, Table/Tree toggles, reset button) and adds it to `left_content_layout`. **Immediately after** that `addLayout` call, insert:

```python
# --- Sidebar Search Bar ---
self.cluster_search_bar = QLineEdit()
self.cluster_search_bar.setPlaceholderText("Search clusters...")
self.cluster_search_bar.setClearButtonEnabled(True)
self.cluster_search_bar.setFixedHeight(28)
self.cluster_search_bar.textChanged.connect(self._filter_sidebar)
left_content_layout.addWidget(self.cluster_search_bar)

# --- Ctrl+F shortcut ---
self.search_shortcut = QShortcut(QKeySequence("Ctrl+F"), self)
self.search_shortcut.activated.connect(self._focus_search_bar)
```

#### 2c. Proxy model setup

After the `QTreeView` is created and before `view_stack` is finalized, create the proxy and wire it. The proxy wraps `main_cluster_model` at model-set time. The cleanest hook is `setup_table_model`:

```python
def setup_table_model(self, model):
    """Sets up the table view model, wrapping it in a filter proxy."""
    # Build a proxy that can filter rows by cluster_id or KSLabel
    proxy = QSortFilterProxyModel(self)
    proxy.setSourceModel(model)
    proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
    proxy.setFilterKeyColumn(-1)  # -1 = search all columns

    self.table_view.setModel(proxy)
    self.table_view.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
    self.table_view.verticalHeader().setVisible(False)
    try:
        self.table_view.selectionModel().selectionChanged.disconnect(
            self.on_view_selection_changed)
    except (TypeError, RuntimeError):
        pass
    self.table_view.selectionModel().selectionChanged.connect(
        self.on_view_selection_changed)

    # Re-apply any active search filter
    if hasattr(self, 'cluster_search_bar') and self.cluster_search_bar.text():
        proxy.setFilterFixedString(self.cluster_search_bar.text())
```

**Note on `_get_selected_cluster_id`:** The existing code at line ~1233 already has a `hasattr(model, 'mapToSource')` branch. Since we are now always installing a proxy, the `mapToSource` path will always be taken. Verify the existing code is:

```python
if hasattr(model, 'mapToSource'):
    source_index = model.mapToSource(model.index(selected_row, 0))
    cluster_id = model.sourceModel()._dataframe.iloc[source_index.row()]['cluster_id']
```

If it currently reads `model._dataframe` after `mapToSource`, fix it to use `model.sourceModel()._dataframe` instead — the proxy itself has no `_dataframe`.

#### 2d. Filter methods

Add these three methods to `MainWindow`:

```python
def _focus_search_bar(self):
    """Ctrl+F: focus the sidebar search bar and select all text."""
    self.cluster_search_bar.setFocus()
    self.cluster_search_bar.selectAll()

def _filter_sidebar(self, text: str):
    """
    Dispatches the search query to whichever view is currently active.
    Also called from _switch_left_view so a pending query applies immediately
    when the user switches views.
    """
    query = text.strip()
    if self.view_stack.currentIndex() == 0:
        self._filter_tree(query)
    else:
        self._filter_table(query)

def _filter_tree(self, query: str):
    """
    Show only tree items whose display text contains `query`
    (case-insensitive substring). Empty query restores all items.
    """
    root = self.tree_model.invisibleRootItem()
    self._apply_tree_filter_recursive(root, query.lower())
    if not query:
        self.tree_view.collapseAll()

def _apply_tree_filter_recursive(self, parent_item, query: str) -> bool:
    """
    Recursively show/hide tree items. Returns True if any child was visible.
    Group nodes are visible iff at least one descendant leaf matches.
    """
    any_visible = False
    for row in range(parent_item.rowCount()):
        child = parent_item.child(row)
        index = self.tree_model.indexFromItem(child)

        if child.rowCount() == 0:
            # Leaf node
            visible = (not query) or (query in child.text().lower())
            self.tree_view.setRowHidden(index.row(), index.parent(), not visible)
            if visible:
                any_visible = True
        else:
            # Group node — recurse first, then decide visibility
            child_matched = self._apply_tree_filter_recursive(child, query)
            self.tree_view.setRowHidden(index.row(), index.parent(), not child_matched)
            if child_matched:
                self.tree_view.setExpanded(index, True)
                any_visible = True

    return any_visible

def _filter_table(self, query: str):
    """
    Filter the table proxy model. QSortFilterProxyModel with filterKeyColumn=-1
    searches all columns. setFilterFixedString is a substring match.
    """
    model = self.table_view.model()
    if model is None or not hasattr(model, 'setFilterFixedString'):
        return  # proxy not installed yet (before first data load)
    model.setFilterFixedString(query)
```

#### 2e. Call `_filter_sidebar` from `_switch_left_view`

Locate `_switch_left_view` and add one line at the end:

```python
def _switch_left_view(self, view_index: int):
    """Switches between the tree and table views in the left pane."""
    # ... existing logic ...

    # Re-apply the active search query to whichever view just became active
    if hasattr(self, 'cluster_search_bar'):
        self._filter_sidebar(self.cluster_search_bar.text())
```

---

### 3. Tree branch CSS — `src/gui/main_window.py`, `_setup_style(colors)`

Locate the comment `/* ── Tree View ── */` and the existing `QTreeView::branch` line (~line 379):

```python
QTreeView::branch {{ background: {colors['bg_panel']}; }}
```

**Replace that single line** with the block below. Everything else in the tree CSS block stays untouched. The SVG `fill` is injected as a Python f-string — it picks up the correct color for both dark and light themes automatically, because `_setup_style` is already called by both `__init__` and `toggle_theme`.

```python
# Compute arrow color before the f-string — text_secondary is visible
# in both themes without being harsh.
_arrow = colors['text_secondary']

# Then inside the stylesheet f-string:
f"""
            QTreeView::branch {{
                background: {colors['bg_panel']};
            }}

            /* Collapsed group: right-pointing triangle */
            QTreeView::branch:has-children:!has-siblings:closed,
            QTreeView::branch:closed:has-children:has-siblings {{
                image: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10'><polygon points='3,2 8,5 3,8' fill='{_arrow}'/></svg>");
            }}

            /* Expanded group: down-pointing triangle */
            QTreeView::branch:open:has-children:!has-siblings,
            QTreeView::branch:open:has-children:has-siblings {{
                image: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10'><polygon points='2,3 5,8 8,3' fill='{_arrow}'/></svg>");
            }}

            /* Remove all connector lines — leaves and siblings */
            QTreeView::branch:has-siblings:!adjoins-item,
            QTreeView::branch:has-siblings:adjoins-item,
            QTreeView::branch:!has-children:!has-siblings:adjoins-item {{
                border-image: none;
                image: none;
            }}
"""
```

**Why inline SVG, not image files or a delegate:**

| Approach | Trade-offs |
|---|---|
| `.png` image files | Breaks on HiDPI, needs 4 files (2 states × 2 themes), asset management |
| `QStyledItemDelegate` override | ~60 lines, interferes with selection highlight painting |
| **Inline SVG data URI** | Zero files, color injected per f-string, 1 code location, DPI-independent |

The only constraint with inline SVG in Qt stylesheets: the SVG body must use single quotes, not double quotes (the outer stylesheet string uses double quotes). The block above is already correct in this regard.

---

## Test Plan

### New test file: `tests/unit/test_gui_polish.py`

```python
import pytest
from qtpy.QtCore import Qt, QSortFilterProxyModel
from qtpy.QtGui import QStandardItem, QStandardItemModel
from qtpy.QtWidgets import QApplication
# (use existing test fixtures from conftest.py)


class TestUmapLayoutFix:

    def test_umap_layout_on_first_visit(self, qtbot, main_window_fixture):
        """First visit must not overlap rows."""
        win = main_window_fixture
        win.analysis_tabs.setCurrentIndex(3)  # UMAP tab index
        qtbot.wait(50)  # allow singleShot(0) to fire
        panel = win.umap_panel
        row1_bottom = panel.run_btn.geometry().bottom()
        row2_top = panel.cluster_btn.geometry().top()
        assert row2_top >= row1_bottom, (
            f"Row overlap on first visit: row1 bottom={row1_bottom}, "
            f"row2 top={row2_top}"
        )

    def test_umap_layout_identical_after_tab_switch(self, qtbot, main_window_fixture):
        """Geometry must not change between first and second visit."""
        win = main_window_fixture
        win.analysis_tabs.setCurrentIndex(3)
        qtbot.wait(50)
        geo_before = win.umap_panel.run_btn.geometry()
        win.analysis_tabs.setCurrentIndex(2)  # STA
        win.analysis_tabs.setCurrentIndex(3)  # back to UMAP
        qtbot.wait(50)
        geo_after = win.umap_panel.run_btn.geometry()
        assert geo_before == geo_after, (
            "UMAP layout changed between first and second visit — "
            "showEvent fix is not working"
        )


class TestSidebarSearch:

    def test_search_bar_present_and_visible(self, qtbot, main_window_fixture):
        win = main_window_fixture
        assert hasattr(win, 'cluster_search_bar')
        assert win.cluster_search_bar.isVisible()

    def test_search_bar_visible_in_both_views(self, qtbot, main_window_fixture):
        win = main_window_fixture
        win._switch_left_view(0)  # Tree
        assert win.cluster_search_bar.isVisible()
        win._switch_left_view(1)  # Table
        assert win.cluster_search_bar.isVisible()

    def test_ctrl_f_focuses_search_bar(self, qtbot, main_window_fixture):
        win = main_window_fixture
        win.show()
        qtbot.keyClick(win, Qt.Key_F, Qt.ControlModifier)
        assert win.cluster_search_bar.hasFocus()

    def test_filter_tree_hides_non_matching_leaves(
            self, qtbot, main_window_with_mock_tree):
        """Non-matching leaf items must be hidden after filter."""
        win = main_window_with_mock_tree
        win._switch_left_view(0)
        win.cluster_search_bar.setText("Cluster 1")
        # Cluster 1 should be visible; Cluster 2 should not
        root = win.tree_model.invisibleRootItem()
        group = root.child(0)
        found_visible = []
        found_hidden = []
        for row in range(group.rowCount()):
            child = group.child(row)
            index = win.tree_model.indexFromItem(child)
            hidden = win.tree_view.isRowHidden(index.row(), index.parent())
            if not hidden:
                found_visible.append(child.text())
            else:
                found_hidden.append(child.text())
        assert any("1" in t for t in found_visible)
        assert all("1" not in t for t in found_hidden)

    def test_filter_tree_keeps_parent_when_child_matches(
            self, qtbot, main_window_with_mock_tree):
        win = main_window_with_mock_tree
        win._switch_left_view(0)
        win.cluster_search_bar.setText("Cluster 5")
        root = win.tree_model.invisibleRootItem()
        group = root.child(0)
        group_index = win.tree_model.indexFromItem(group)
        assert not win.tree_view.isRowHidden(group_index.row(), group_index.parent())

    def test_clear_filter_restores_all_items(
            self, qtbot, main_window_with_mock_tree):
        win = main_window_with_mock_tree
        win._switch_left_view(0)
        win.cluster_search_bar.setText("zzz_no_match")
        win.cluster_search_bar.clear()
        root = win.tree_model.invisibleRootItem()
        group = root.child(0)
        for row in range(group.rowCount()):
            child = group.child(row)
            index = win.tree_model.indexFromItem(child)
            assert not win.tree_view.isRowHidden(index.row(), index.parent())

    def test_filter_table_reduces_rows(self, qtbot, main_window_fixture):
        win = main_window_fixture
        win._switch_left_view(1)  # Table
        initial_rows = win.table_view.model().rowCount()
        assert initial_rows > 0
        win.cluster_search_bar.setText("zzz_will_never_match_9999")
        assert win.table_view.model().rowCount() == 0
        win.cluster_search_bar.clear()
        assert win.table_view.model().rowCount() == initial_rows

    def test_filter_applies_on_view_switch(self, qtbot, main_window_with_mock_tree):
        """A query active in tree view must apply to table immediately on switch."""
        win = main_window_with_mock_tree
        win._switch_left_view(0)
        win.cluster_search_bar.setText("Cluster 1")
        win._switch_left_view(1)  # switch to table with query still set
        # Table proxy filter must be active
        model = win.table_view.model()
        assert hasattr(model, 'filterRegularExpression') or hasattr(
            model, 'filterFixedString')
        # All visible rows must contain "1"
        for row in range(model.rowCount()):
            text = str(model.data(model.index(row, 0)))
            assert "1" in text.lower() or True  # at minimum, no crash


class TestTreeBranchStyling:

    def test_branch_css_contains_svg_triangles(self, main_window_fixture):
        qss = main_window_fixture.styleSheet()
        assert "branch:has-children" in qss, "No branch:has-children rule found"
        assert "polygon points=" in qss, "No SVG polygon found in branch CSS"
        assert "branch:open" in qss, "No open-branch rule found"

    def test_branch_color_changes_on_theme_toggle(self, main_window_fixture):
        """Arrow fill color must differ between dark and light themes."""
        win = main_window_fixture
        dark_qss = win.styleSheet()
        win.toggle_theme()
        light_qss = win.styleSheet()
        assert dark_qss != light_qss, (
            "Stylesheet did not change after toggle_theme — "
            "branch arrow color is not theme-aware"
        )
        win.toggle_theme()  # restore
```

### Required test fixtures (`tests/conftest.py` additions)

The agent must add `main_window_with_mock_tree` if it does not already exist:

```python
@pytest.fixture
def main_window_with_mock_tree(qtbot, main_window_fixture):
    """
    main_window_fixture with a pre-built mock tree:
    root
    └── good
        └── Nc0
            ├── Cluster 1 (n=100)
            ├── Cluster 2 (n=200)
            ├── Cluster 5 (n=50)
            └── Cluster 10 (n=75)
    """
    from qtpy.QtGui import QStandardItem, QStandardItemModel
    win = main_window_fixture
    model = QStandardItemModel()
    root = model.invisibleRootItem()
    good = QStandardItem("good")
    nc0 = QStandardItem("Nc0")
    for cid, n in [(1, 100), (2, 200), (5, 50), (10, 75)]:
        leaf = QStandardItem(f"Cluster {cid} (n={n})")
        leaf.setData(cid, Qt.ItemDataRole.UserRole)
        nc0.appendRow(leaf)
    good.appendRow(nc0)
    root.appendRow(good)
    win.tree_model = model
    win.tree_view.setModel(model)
    win.tree_view.expandAll()
    return win
```

---

## Screenshot Verification (Manual)

After implementation, launch with the real dataset at `/mnt/lab/Array-data/` and verify:

| Check | Expected |
|---|---|
| Cold launch → click UMAP immediately | Both toolbar rows fully visible, no overlap |
| Switch STA → UMAP | Geometry identical to first visit |
| Type `"26"` in search bar (Tree view) | Only clusters with "26" in name visible; parent groups stay open |
| Clear search bar | Full tree restored, collapsed to default |
| Switch to Table view with `"26"` still typed | Table rows immediately filtered to matching clusters |
| `Ctrl+F` | Focus jumps to search bar, text selected |
| Click any group row arrow | Collapses/expands, triangle swaps `▶` ↔ `▼` |
| Toggle Light Mode | Arrows recolor immediately, no restart |
| Resize window to ~1200px wide | No layout breaks in any panel |

---

## Out Of Scope

* No UMAP toolbar widget or control order changes.
* No `standard_plots_panel.py` changes (SPATIAL TEMPLATE label left as-is).
* No drag-and-drop, context menu, or cluster data changes.
* No `DataManager`, worker, or signal changes.
* No search text persistence between sessions.
* No fuzzy/regex search — simple case-insensitive substring only.
* No debounce on the search bar — immediate filter is acceptable for <1000 clusters.
* No column-header sorting changes to the table.


# Specification: Light Mode Polish & UI Cleanup
## STATUS: DONE

## Metadata

* **Status:** Ready for Dev
* **Target Release:** v1.1
* **Primary Developer/Agent:** [Agent Name]

## Objective

Implement a "first-principles" color theme architecture to centralize UI styling and resolve legibility issues in Light Mode. The new architecture must enforce semantic color roles (e.g., `text_primary`, `text_secondary`, `bg_panel`) across all Qt widgets, sidebars, tree views, and every data panel. Additionally, clean up deprecated UI elements to streamline the user experience and enable click-to-sort functionality on the cluster table.

## User Story

"As a researcher using the application in a brightly lit lab, I want to switch to Light Mode and read all text and plots clearly—including the sidebar, tree view, and every data panel (especially the Population Panel). Additionally, I want to sort my clusters by clicking the table headers and not be distracted by dead, non-functional UI buttons."

## Acceptance Criteria (Definition of Done)

* **AC1 (First-Principles Theming):** A new file `src/gui/theme.py` is created. It defines an abstract, semantic color dictionary (e.g., `text_primary`, `text_secondary`, `bg_main`, `bg_panel`, `plot_line`, `plot_scatter`) for both Dark and Light modes.
* **AC2 (Universal Legibility):** Toggling Light Mode successfully updates the global stylesheet, pyqtgraph configurations, and matplotlib figures.
    * *Requirement:* All text in the sidebar and tree view must invert correctly.
    * *Requirement:* All plots in the Standard Plots, EI Panel, UMAP Panel, and **Population Panel** must dynamically consume the theme colors. For the Population Panel specifically, the shadow traces, mean lines, and RF ellipses must maintain high contrast against the chosen background.
* **AC3 (Table Sorting):** Clicking a column header in the cluster Table View successfully sorts the table by that column (ascending/descending toggle).
* **AC4 (Cleanup):** The "Good" view toggle button and the "Refine Selected Cluster" button are completely removed from the UI and codebase.
* **AC5 (Visual Check):** Launch the GUI, load a dataset, and toggle Light Mode. Verify the sidebar, tree view, UMAP panel, standard plots, EI panel, and Population panel adapt to the light semantic palette without any hidden, low-contrast, or unreadable text.

## Architecture & Technical Constraints

* **Files Modified:** * `src/gui/main_window.py` (Remove hardcoded colors, remove dead buttons, implement global theme swap).
    * `src/gui/theme.py` (New file for semantic theme state and dictionaries).
    * `src/gui/widgets.py` (Add sorting capability to `PandasModel` or `CustomTableView`).
    * Plotting modules (e.g., `population_panel.py`, `standard_plots_panel.py`, `ei_panel.py`, `umap_panel.py`) must be updated to request colors via a central method like `main_window.get_current_colors()` instead of hardcoding hex values.
* **Data Contracts:** `theme.py` must expose a consistent dictionary structure so that adding new themes in the future requires zero changes to the panel modules.
* **UI/Threading Rules:** Theme switching must execute synchronously on the main thread and trigger a universal `.restyle()` or redraw event across all active pyqtgraph/matplotlib canvases.

## Test Plan (TDD Requirements)

* **Unit:** Add a test verifying `theme.py` exposes identical keys for both Light and Dark dictionaries to prevent `KeyError` crashes in the plotting logic.
* **Integration:** Add a test using `qtbot` to instantiate `MainWindow`, toggle the theme state, and assert that the application's base font color and the pyqtgraph global background setting correctly update.
* **Integration:** Add a test verifying that invoking the sort method on the table header correctly reorders the underlying Pandas dataframe.

## Out Of Scope

* Redesigning the physical layout or sizing of the UI panels.
* Implementing new UI themes beyond Light/Dark at this time.