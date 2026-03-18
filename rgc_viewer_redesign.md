# RGC Viewer — UI/UX Redesign Specification

**Version:** 1.0  
**Scope:** Aesthetic overhaul — color system, layout, spacing, components, and QSS  
**Files covered:** `main_window.py`, `standard_plots_panel.py`, `similarity_panel.py`

---

## Table of Contents

1. [Problems Identified](#1-problems-identified)
2. [Color System](#2-color-system)
3. [Typography](#3-typography)
4. [Layout & Proportions](#4-layout--proportions)
5. [Component Redesigns](#5-component-redesigns)
   - 5.1 [Left Panel Controls](#51-left-panel-controls)
   - 5.2 [Data Tables](#52-data-tables)
   - 5.3 [Tab Bar](#53-tab-bar)
   - 5.4 [Refine Button](#54-refine-button)
   - 5.5 [Sidebar Toggle](#55-sidebar-toggle)
   - 5.6 [Similarity Panel](#56-similarity-panel)
   - 5.7 [Standard Plots Panel Controls](#57-standard-plots-panel-controls)
6. [Plot Styling (pyqtgraph)](#6-plot-styling-pyqtgraph)
7. [Drop-in QSS Stylesheet](#7-drop-in-qss-stylesheet)
8. [Prioritized Implementation Roadmap](#8-prioritized-implementation-roadmap)

---

## 1. Problems Identified

### 1.1 Tab bar overflow / overlap
**File:** `main_window.py` — `_setup_ui()`, line ~438  
Six tabs (`Standard Plots`, `EI Analysis`, `STA Analysis`, `Class Discovery (UMAP)`, `Raw Waveforms`, `Raw Trace`) plus the `Population Split View` checkbox are crammed into a single tab bar row using `setCornerWidget`. At 1800px they nearly overlap — at any smaller window width they will visibly collide.

**Fix:** Shorten tab labels + move Population Split View to an icon-style toggle button (see §5.3).

---

### 1.2 Left panel button clutter
**File:** `main_window.py` — `_setup_ui()`, lines ~363–417  
Four full-width buttons take up ~90px of vertical space above the cluster table:
- `Filter 'Good'` and `Reset View` — rendered with identical visual weight
- `Tree View` and `Table View` — two separate `QPushButton`s instead of a segmented control

`Reset View` is rarely used but occupies the same visual prominence as the primary filter action.

**Fix:** Collapse all four into a single compact row (see §5.1).

---

### 1.3 Data tables feel dense and heavy
**Files:** `main_window.py`, `similarity_panel.py`  
Both the main cluster table and the similarity table use:
- Full gridlines on every cell (horizontal + vertical)
- Alternating row colors that are too close in value to meaningfully separate rows
- Column headers that look identical in weight to data rows
- Raw text status values (`"Original"`, `"Clean"`) — not scannable at a glance

**Fix:** Remove vertical gridlines, increase header contrast, add status badge chips (see §5.2).

---

### 1.4 Inconsistent accent colors
**File:** `main_window.py` — `_setup_style()` and inline `setStyleSheet()` calls  
Every component has its own palette:
- Tab selected: `#4282DA`
- Refine button bg: `#005230`, text: `#aeffe3`
- Pop expand button: `#4282DA` (normal), `#2D6A4F` (active)
- pyqtgraph background: `#1f1f1f`
- Main window background: `#2D2D2D`

These never share tokens, so changing the theme requires touching 10+ places.

**Fix:** Unified CSS token system (see §2).

---

### 1.5 Sidebar toggle is a 20px sliver
**File:** `main_window.py` — `_setup_ui()`, line ~354  
The `◀` button is `setFixedWidth(20)` — barely clickable and easy to miss. It also sits at the very top of `left_layout`, orphaned from the panel content it controls.

**Fix:** Replace with a styled `QSplitter` handle (see §5.5).

---

### 1.6 No visual hierarchy in the left panel
**File:** `main_window.py` — `_setup_ui()`  
The cluster table, Refine button, Similar Clusters section header, similarity table, and Mark Status dropdown all render at the same visual weight. There is no clear primary/secondary/tertiary distinction.

**Fix:** Size and color hierarchy (see §5.1, §5.4, §5.6).

---

### 1.7 ISI control bar is overcrowded
**File:** `standard_plots_panel.py` — `__init__()`, lines ~113–147  
The ISI control bar contains 6 widgets on a single row (`View`, `Refr line` checkbox, `Ref (ms)` label + spinbox + `Set` button, `Plot` combo, `X` range combo) with no grouping or visual separation. At narrow window widths these overlap.

**Fix:** Group related controls with subtle separators and reduce label verbosity (see §5.7).

---

### 1.8 MEA / Vision radio buttons are clunky
**File:** `similarity_panel.py` — `__init__()`, lines ~28–40  
Two `QRadioButton`s in a plain `QHBoxLayout` serve as a source toggle. Radio buttons are the wrong widget for a two-option mutually exclusive toggle — a segmented control is more compact and clearer.

**Fix:** Replace with a segmented QPushButton pair (see §5.6).

---

### 1.9 Commented-out dead code in similarity panel
**File:** `similarity_panel.py` — lines ~51–67  
A large block of 4 individual mark buttons (`Mark Clean`, `Mark Edge`, `Mark as Duplicates`, `Mark Unsure`) is commented out and replaced with the combo+button approach below it. The dead code should be removed entirely to keep the file clean.

---

## 2. Color System

Replace all hardcoded hex values in `_setup_style()` and all inline `setStyleSheet()` calls with these tokens. Define them as Python constants at the top of `main_window.py` so they can be referenced everywhere.

```python
# main_window.py — add near top of file
COLORS = {
    # Surfaces
    "bg_base":     "#111214",   # Window / app background
    "bg_panel":    "#18191C",   # Left and right panes, plot backgrounds
    "bg_surface":  "#1E2025",   # Table rows, cards
    "bg_elevated": "#282A30",   # Hover states, alternating table rows

    # Accents
    "accent":          "#2E6DD4",   # Active tabs, selected rows, links
    "accent_hover":    "#4A8BEF",   # Hover on accent elements
    "accent_positive": "#1A5C3A",   # Refine button, "good" status
    "accent_pos_text": "#6EE7B7",   # Text on positive background

    # Text hierarchy
    "text_primary":   "#F0F0F2",   # Main data, labels
    "text_secondary": "#9B9DA6",   # Supporting text, axis labels
    "text_tertiary":  "#5A5C65",   # Placeholder text, column headers
    "text_disabled":  "#3A3C44",   # Disabled controls

    # Borders
    "border_subtle":   "#2E3038",   # Table dividers, panel edges
    "border_default":  "#3D3F48",   # Button borders, input outlines
    "border_strong":   "#5A5C65",   # Focused inputs, hover borders

    # Status badges
    "status_good_bg":   "rgba(26,  92,  58,  0.20)",
    "status_good_text": "#6EE7B7",
    "status_mua_bg":    "rgba(186, 117,  23, 0.20)",
    "status_mua_text":  "#F0C060",
    "status_noise_bg":  "rgba(163,  45,  45, 0.20)",
    "status_noise_text":"#F08080",
    "status_unsort_bg": "rgba(46,  109, 212, 0.20)",
    "status_unsort_text":"#7EB8F7",
}
```

---

## 3. Typography

**Current:** `Segoe UI 9pt` applied uniformly via `self.setFont(QFont("Segoe UI", 9))`  
**Recommendation:** Use `"Inter"` or `"IBM Plex Sans"` (both free via Google Fonts / bundled) at 11pt base. These render more consistently across platforms than Segoe UI.

### Scale

| Use | Size | Weight | Color token |
|---|---|---|---|
| Panel section headers | 12px | 500 (medium) | `text_primary` |
| Table data | 12px | 400 | `text_secondary` |
| Table column headers | 10px | 500 | `text_tertiary` — ALL CAPS |
| Button labels | 12px | 400 | `text_primary` |
| Status bar messages | 11px | 400 | `text_tertiary` |
| Plot axis labels | 10px | 400 | `text_secondary` |
| Plot titles | 11px | 500 | `text_tertiary` — UPPERCASE |

```python
# In _setup_style():
self.setFont(QFont("Inter", 11))
# Fallback: QFont("Segoe UI", 11) on Windows, QFont("SF Pro Text", 11) on macOS
```

---

## 4. Layout & Proportions

### 4.1 Left panel width
**Current:** Default splitter position leaves the left panel at ~360px — too wide, it steals from the plot area.  
**Fix:** Set initial splitter sizes in `_setup_ui()`:

```python
# After creating the main_splitter, set initial widths:
self.main_splitter.setSizes([220, self.width() - 220])
self.main_splitter.setStretchFactor(0, 0)  # Left panel doesn't stretch
self.main_splitter.setStretchFactor(1, 1)  # Right panel takes remaining space
```

### 4.2 Spacing constants
Apply these padding values consistently. Define at top of `main_window.py`:

```python
PANEL_PADDING  = 8   # px — inner padding on all panels
CTRL_SPACING   = 6   # px — gap between controls in a row
ROW_HEIGHT     = 28  # px — standard table row height (was ~32px)
HEADER_HEIGHT  = 36  # px — tab bar and control bar height
STATUS_HEIGHT  = 24  # px — status bar
```

### 4.3 Apply panel margins
```python
# In _setup_ui(), for left_content_layout:
left_content_layout.setContentsMargins(PANEL_PADDING, PANEL_PADDING, PANEL_PADDING, PANEL_PADDING)
left_content_layout.setSpacing(CTRL_SPACING)

# For similarity panel layout (similarity_panel.py):
layout.setContentsMargins(0, PANEL_PADDING, 0, 0)
layout.setSpacing(CTRL_SPACING)
```

### 4.4 Plot area breathing room
```python
# In standard_plots_panel.py — __init__():
self.vert_splitter.setContentsMargins(4, 4, 4, 4)
self.top_splitter.setHandleWidth(4)
self.bottom_splitter.setHandleWidth(4)
self.vert_splitter.setHandleWidth(4)
```

---

## 5. Component Redesigns

### 5.1 Left Panel Controls

**Current (4 buttons stacked):**
```python
self.filter_button = QPushButton("Filter 'Good'")
self.reset_button  = QPushButton("Reset View")
self.tree_view_button  = QPushButton("Tree View")
self.table_view_button = QPushButton("Table View")
```

**Proposed (single compact row):**

Replace the `filter_box` and `view_switch_layout` with one `QHBoxLayout`:

```python
# --- Filter + View Toggle Row ---
top_ctrl_layout = QHBoxLayout()
top_ctrl_layout.setSpacing(4)

# Segmented filter toggle
self.filter_all_btn  = QPushButton("All")
self.filter_good_btn = QPushButton("Good")
for btn in (self.filter_all_btn, self.filter_good_btn):
    btn.setCheckable(True)
    btn.setFixedHeight(26)
    btn.setStyleSheet("QPushButton { border-radius: 0; }")
self.filter_all_btn.setChecked(True)
self.filter_all_btn.setStyleSheet(
    "border-radius: 0; border-top-left-radius: 5px; border-bottom-left-radius: 5px;"
)
self.filter_good_btn.setStyleSheet(
    "border-radius: 0; border-top-right-radius: 5px; border-bottom-right-radius: 5px;"
)

# Segmented view toggle
self.table_view_button = QPushButton("Table")
self.tree_view_button  = QPushButton("Tree")
for btn in (self.table_view_button, self.tree_view_button):
    btn.setCheckable(True)
    btn.setFixedHeight(26)
self.table_view_button.setChecked(True)

# Reset as ghost link
self.reset_button = QPushButton("↺")
self.reset_button.setToolTip("Reset View")
self.reset_button.setFixedSize(26, 26)
self.reset_button.setStyleSheet(
    "QPushButton { border: none; color: #5A5C65; font-size: 14px; }"
    "QPushButton:hover { color: #F0F0F2; }"
)

top_ctrl_layout.addWidget(self.filter_all_btn)
top_ctrl_layout.addWidget(self.filter_good_btn)
top_ctrl_layout.addSpacing(8)
top_ctrl_layout.addWidget(self.table_view_button)
top_ctrl_layout.addWidget(self.tree_view_button)
top_ctrl_layout.addStretch()
top_ctrl_layout.addWidget(self.reset_button)
```

---

### 5.2 Data Tables

**Apply to both `self.table_view` (main) and `self.table` in `SimilarityPanel`.**

#### QSS changes (in the main stylesheet):
```css
QTableView {
    background-color: #18191C;
    alternate-background-color: #1E2025;
    gridline-color: transparent;          /* Remove all gridlines */
    border: none;
    selection-background-color: rgba(46, 109, 212, 0.18);
    selection-color: #F0F0F2;
    show-decoration-selected: 1;
}
QTableView::item {
    border-bottom: 1px solid #2E3038;    /* Horizontal only */
    padding: 0 8px;
}
QHeaderView::section {
    background-color: #18191C;
    color: #5A5C65;
    padding: 4px 8px;
    border: none;
    border-bottom: 1px solid #3D3F48;
    font-size: 10px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
```

#### Row height:
```python
# After setting the model on both table views:
self.table_view.verticalHeader().setDefaultSectionSize(28)
self.table.verticalHeader().setDefaultSectionSize(28)  # in similarity_panel.py
self.table_view.verticalHeader().setVisible(False)      # Hide row index numbers
self.table.verticalHeader().setVisible(False)
```

#### Status badges (in `HighlightStatusPandasModel` or a custom delegate):
Instead of raw text in the `status` column, consider a `QStyledItemDelegate` that draws a colored rounded rect. At minimum, use `foreground` role in the model to color-code status text:

```python
# In HighlightStatusPandasModel.data(), for the status column:
STATUS_COLORS = {
    "Clean":    "#6EE7B7",
    "Edge":     "#F0C060",
    "Duplicate":"#F08080",
    "Noise":    "#F08080",
    "Unsure":   "#7EB8F7",
    "Original": "#9B9DA6",
}
if role == Qt.ForegroundRole:
    col_name = self._dataframe.columns[index.column()]
    if col_name == 'status':
        status = str(self._dataframe.iloc[index.row()][col_name])
        color = STATUS_COLORS.get(status, "#9B9DA6")
        return QColor(color)
```

---

### 5.3 Tab Bar

**Current:** 6 long labels + corner widget checkbox.  
**Proposed:** Short labels + Population as a subtle toggle button.

```python
# In _setup_ui() — replace tab labels:
TAB_LABELS = {
    "Standard Plots":          "Standard",
    "EI Analysis":             "EI",
    "STA Analysis":            "STA",
    "Class Discovery (UMAP)":  "UMAP",
    "Raw Waveforms":           "Waveforms",
    "Raw Trace":               "Raw",
}
# When adding panels to analysis_tabs, use short labels:
# self.analysis_tabs.addTab(self.standard_plots_panel, "Standard")
# etc.

# Replace corner widget checkbox with a compact icon button:
self.pop_view_btn = QPushButton("⊞  Population")
self.pop_view_btn.setCheckable(True)
self.pop_view_btn.setFixedHeight(28)
self.pop_view_btn.setStyleSheet("""
    QPushButton {
        font-size: 11px;
        padding: 0 10px;
        border: 0.5px solid #3D3F48;
        border-radius: 5px;
        color: #9B9DA6;
        background: transparent;
    }
    QPushButton:checked {
        background: rgba(46, 109, 212, 0.20);
        border-color: #2E6DD4;
        color: #4A8BEF;
    }
    QPushButton:hover:!checked {
        background: #1E2025;
        color: #F0F0F2;
    }
""")
self.pop_view_btn.toggled.connect(self.toggle_population_split_view)
self.analysis_tabs.setCornerWidget(self.pop_view_btn, Qt.TopRightCorner)
```

Also add a `QScrollArea` fallback so tabs never overlap at small window widths:
```python
# After creating analysis_tabs:
self.analysis_tabs.tabBar().setUsesScrollButtons(True)
self.analysis_tabs.tabBar().setElideMode(Qt.ElideNone)
```

---

### 5.4 Refine Button

**Current:**
```python
self.refine_button.setStyleSheet(
    "font-size: 14px; font-weight: bold; color: #aeffe3; background-color: #005230;"
)
self.refine_button.setFixedHeight(40)
```

**Proposed:** Keep the green accent (it correctly signals primary action) but reduce height and align to the token system:
```python
self.refine_button.setFixedHeight(32)
self.refine_button.setStyleSheet("""
    QPushButton {
        font-size: 12px;
        font-weight: 500;
        color: #6EE7B7;
        background-color: #1A5C3A;
        border: none;
        border-radius: 6px;
        padding: 0 12px;
    }
    QPushButton:hover  { background-color: #226B46; }
    QPushButton:pressed { background-color: #14452C; }
    QPushButton:disabled {
        background-color: #1E2025;
        color: #3A3C44;
    }
""")
```

---

### 5.5 Sidebar Toggle

**Current:** A `QPushButton("◀")` set to `setFixedWidth(20)`.

**Proposed:** Remove this button entirely and replace with a styled `QSplitter` handle that users can double-click to collapse, which is standard OS behavior:

```python
# In _setup_ui(), remove the sidebar_toggle_button widget entirely.
# Replace toggle_sidebar() functionality with splitter double-click:

self.main_splitter.setHandleWidth(5)
# Apply via QSS in _setup_style():
# QSplitter::handle:horizontal { background: #2E3038; width: 5px; }
# QSplitter::handle:horizontal:hover { background: #4A8BEF; cursor: col-resize; }

# For double-click collapse (optional):
self.main_splitter.handle(1).mouseDoubleClickEvent = lambda e: self.toggle_sidebar()
```

If a visible collapse button is still desired, embed it inside the splitter handle or at the bottom of the left pane as a 100%-width minimal button:
```python
self.collapse_btn = QPushButton("‹ Hide")
self.collapse_btn.setFixedHeight(20)
self.collapse_btn.setStyleSheet("""
    QPushButton {
        font-size: 10px;
        color: #5A5C65;
        background: #18191C;
        border: none;
        border-top: 0.5px solid #2E3038;
    }
    QPushButton:hover { color: #F0F0F2; background: #1E2025; }
""")
```

---

### 5.6 Similarity Panel

**File:** `similarity_panel.py`

#### Replace radio buttons with a segmented control:
```python
# Remove: self.mea_radio, self.vision_radio, self.source_button_group

# Add in their place:
source_row = QHBoxLayout()
source_row.setSpacing(0)

self.mea_btn    = QPushButton("MEA")
self.vision_btn = QPushButton("Vision")

for i, btn in enumerate([self.mea_btn, self.vision_btn]):
    btn.setCheckable(True)
    btn.setFixedHeight(24)
    btn.setStyleSheet(f"""
        QPushButton {{
            font-size: 11px;
            padding: 0 10px;
            border: 0.5px solid #3D3F48;
            border-{'right' if i == 0 else 'left'}-width: 0;
            border-radius: 0;
            {'border-top-left-radius: 4px; border-bottom-left-radius: 4px;' if i == 0 else
             'border-top-right-radius: 4px; border-bottom-right-radius: 4px; border-left: none;'}
            color: #9B9DA6;
            background: transparent;
        }}
        QPushButton:checked {{
            background: rgba(46, 109, 212, 0.20);
            color: #4A8BEF;
        }}
    """)

self.mea_btn.setChecked(True)
self.mea_btn.clicked.connect(lambda: self._set_source("MEA"))
self.vision_btn.clicked.connect(lambda: self._set_source("vision"))

source_row.addWidget(self.mea_btn)
source_row.addWidget(self.vision_btn)
source_row.addStretch()
layout.addLayout(source_row)
```

#### Move Mark Status inline with the table header:
The `status_combo` + `mark_button` row at the bottom should stay, but reduce the combo width and button to a compact style:
```python
self.status_combo.setFixedHeight(26)
self.status_combo.setFixedWidth(110)
self.mark_button.setFixedHeight(26)
self.mark_button.setText("Mark")  # Shorter label
```

#### Remove the commented-out dead code (lines ~51–67). Clean file.

---

### 5.7 Standard Plots Panel Controls

**File:** `standard_plots_panel.py`

**Current:** All 6 ISI controls crammed into one `QHBoxLayout` with verbose labels.

**Proposed:** Group into two logical segments with a thin separator:

```python
# Replace the ISI controls section with:
isi_controls = QHBoxLayout()
isi_controls.setSpacing(4)

# Group 1: View type
self.isi_view_combo = QComboBox()
self.isi_view_combo.addItems(['ISI Histogram', 'ISI vs Amplitude'])
self.isi_view_combo.setFixedHeight(24)
isi_controls.addWidget(self.isi_view_combo)

# Separator
sep1 = QFrame(); sep1.setFrameShape(QFrame.VLine)
sep1.setStyleSheet("color: #2E3038;"); sep1.setFixedWidth(1)
isi_controls.addWidget(sep1)

# Group 2: Refractory line
self.show_refractory_line_checkbox = QCheckBox('Refr.')
self.show_refractory_line_checkbox.setChecked(True)
self.refractory_spinbox = QDoubleSpinBox()
self.refractory_spinbox.setRange(0.1, 10.0)
self.refractory_spinbox.setDecimals(1)  # 1 decimal is sufficient
self.refractory_spinbox.setSingleStep(0.1)
self.refractory_spinbox.setValue(1.0)
self.refractory_spinbox.setFixedWidth(52)
self.refractory_spinbox.setFixedHeight(24)
self.refractory_spinbox.setSuffix(' ms')  # Replaces the separate QLabel
self.update_refractory_btn = QPushButton('Set')
self.update_refractory_btn.setFixedHeight(24)
self.update_refractory_btn.setFixedWidth(32)

isi_controls.addWidget(self.show_refractory_line_checkbox)
isi_controls.addWidget(self.refractory_spinbox)
isi_controls.addWidget(self.update_refractory_btn)

# Separator
sep2 = QFrame(); sep2.setFrameShape(QFrame.VLine)
sep2.setStyleSheet("color: #2E3038;"); sep2.setFixedWidth(1)
isi_controls.addWidget(sep2)

# Group 3: Display and range
self.isi_display_combo = QComboBox()
self.isi_display_combo.addItems(['Scatter', 'Density'])
self.isi_display_combo.setFixedHeight(24)
self.isi_range_combo = QComboBox()
self.isi_range_combo.addItems(['0–50 ms', '0–500 ms', '0–1000 ms', 'Full'])
self.isi_range_combo.setFixedHeight(24)

isi_controls.addWidget(QLabel('Plot:'))
isi_controls.addWidget(self.isi_display_combo)
isi_controls.addWidget(QLabel('X:'))
isi_controls.addWidget(self.isi_range_combo)
isi_controls.addStretch()
```

Also fix the top channel display control bar height. `ctrl_bar_widget.setMaximumHeight(35)` → `setFixedHeight(32)` for a tighter fit.

---

## 6. Plot Styling (pyqtgraph)

**File:** `main_window.py` (global config) + `standard_plots_panel.py` (`_style_plot()`)

### 6.1 Global config
```python
# main_window.py — replace existing pg.setConfigOption calls:
pg.setConfigOption('background', '#18191C')   # Matches --bg-panel
pg.setConfigOption('foreground', '#9B9DA6')   # Matches --text-secondary
pg.setConfigOptions(antialias=True)
```

### 6.2 `_style_plot()` in StandardPlotsPanel
```python
def _style_plot(self, plot_widget):
    plot_widget.getPlotItem().getAxis('bottom').setPen(pg.mkPen('#3D3F48'))
    plot_widget.getPlotItem().getAxis('left').setPen(pg.mkPen('#3D3F48'))
    plot_widget.getPlotItem().getAxis('bottom').setTextPen(pg.mkPen('#9B9DA6'))
    plot_widget.getPlotItem().getAxis('left').setTextPen(pg.mkPen('#9B9DA6'))

    # Hide top and right spines
    plot_widget.showAxis('top', False)
    plot_widget.showAxis('right', False)

    # Subtle grid
    plot_widget.showGrid(x=True, y=True, alpha=0.08)

    # Remove the default blue border pyqtgraph adds
    plot_widget.getPlotItem().setContentsMargins(8, 8, 8, 8)
    plot_widget.setBackground('#18191C')
```

### 6.3 Plot title style
Update all `pg.PlotWidget(title=...)` calls to use uppercase small labels matching the typography scale:
```python
# Example — autocorrelation:
self.acg_plot = pg.PlotWidget()
self.acg_plot.setTitle(
    "<span style='color:#5A5C65; font-size:10px; letter-spacing:0.06em;'>AUTOCORRELATION</span>"
)
```

### 6.4 ACG bar colors (keep the purple — it's distinctive)
The purple autocorrelation bars are a good visual identity. Just update brush opacity for the new darker background:
```python
self._acg_bar = pg.BarGraphItem(
    x=[], height=[], width=0.8,
    brush=pg.mkBrush(170, 0, 255, 130),    # Slightly more opaque
    pen=pg.mkPen('#9933FF', width=0.5)      # Thinner pen
)
```

---

## 7. Drop-in QSS Stylesheet

Replace the entire body of `_setup_style()` in `main_window.py` with the following. This is the single biggest win — it touches every widget at once.

```python
def _setup_style(self):
    self.setFont(QFont("Inter", 11))

    self.setStyleSheet("""
        /* ── Base ───────────────────────────── */
        QWidget {
            color: #F0F0F2;
            background-color: #111214;
            font-family: 'Inter', 'Segoe UI', sans-serif;
            font-size: 12px;
        }
        QMainWindow, QDialog {
            background-color: #111214;
        }

        /* ── Splitter handles ────────────────── */
        QSplitter::handle {
            background: #2E3038;
        }
        QSplitter::handle:horizontal {
            width: 5px;
        }
        QSplitter::handle:vertical {
            height: 5px;
        }
        QSplitter::handle:horizontal:hover,
        QSplitter::handle:vertical:hover {
            background: #4A8BEF;
        }

        /* ── Tables ──────────────────────────── */
        QTableView {
            background-color: #18191C;
            alternate-background-color: #1E2025;
            gridline-color: transparent;
            border: none;
            selection-background-color: rgba(46, 109, 212, 0.18);
            selection-color: #F0F0F2;
        }
        QTableView::item {
            border-bottom: 1px solid #2E3038;
            padding: 0 8px;
        }
        QTableView::item:selected {
            background-color: rgba(46, 109, 212, 0.18);
        }
        QHeaderView::section {
            background-color: #18191C;
            color: #5A5C65;
            padding: 4px 8px;
            border: none;
            border-bottom: 1px solid #3D3F48;
            font-size: 10px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        QHeaderView::section:hover {
            background-color: #1E2025;
            color: #9B9DA6;
        }
        QHeaderView {
            background-color: #18191C;
        }

        /* ── Buttons ─────────────────────────── */
        QPushButton {
            background-color: transparent;
            border: 0.5px solid #3D3F48;
            color: #9B9DA6;
            padding: 4px 10px;
            border-radius: 5px;
            font-size: 12px;
        }
        QPushButton:hover {
            background-color: #1E2025;
            border-color: #5A5C65;
            color: #F0F0F2;
        }
        QPushButton:pressed {
            background-color: #282A30;
        }
        QPushButton:checked {
            background-color: rgba(46, 109, 212, 0.20);
            border-color: #2E6DD4;
            color: #4A8BEF;
        }
        QPushButton:disabled {
            color: #3A3C44;
            border-color: #2E3038;
        }

        /* ── Tabs ────────────────────────────── */
        QTabWidget::pane {
            border: none;
            border-top: 1px solid #2E3038;
        }
        QTabBar::tab {
            color: #9B9DA6;
            background: transparent;
            padding: 6px 16px;
            font-size: 12px;
            border-bottom: 2px solid transparent;
            margin-right: 2px;
            min-width: 40px;
        }
        QTabBar::tab:selected {
            color: #F0F0F2;
            border-bottom: 2px solid #4A8BEF;
        }
        QTabBar::tab:hover:!selected {
            color: #F0F0F2;
            background: #1E2025;
        }
        QTabBar::scroller {
            width: 24px;
        }

        /* ── Inputs ──────────────────────────── */
        QComboBox {
            background-color: #18191C;
            border: 0.5px solid #3D3F48;
            border-radius: 4px;
            padding: 3px 8px;
            color: #F0F0F2;
            font-size: 12px;
            min-height: 22px;
        }
        QComboBox:hover { border-color: #5A5C65; }
        QComboBox::drop-down {
            border: none;
            width: 18px;
        }
        QComboBox QAbstractItemView {
            background-color: #1E2025;
            border: 0.5px solid #3D3F48;
            selection-background-color: rgba(46, 109, 212, 0.25);
            color: #F0F0F2;
        }
        QDoubleSpinBox, QSpinBox {
            background-color: #18191C;
            border: 0.5px solid #3D3F48;
            border-radius: 4px;
            padding: 3px 6px;
            color: #F0F0F2;
            font-size: 12px;
        }
        QDoubleSpinBox:hover, QSpinBox:hover {
            border-color: #5A5C65;
        }

        /* ── Checkboxes ──────────────────────── */
        QCheckBox {
            color: #9B9DA6;
            spacing: 5px;
            font-size: 12px;
        }
        QCheckBox:hover { color: #F0F0F2; }
        QCheckBox::indicator {
            width: 14px;
            height: 14px;
            border: 0.5px solid #3D3F48;
            border-radius: 3px;
            background: #18191C;
        }
        QCheckBox::indicator:checked {
            background: #2E6DD4;
            border-color: #2E6DD4;
        }

        /* ── Radio buttons ───────────────────── */
        QRadioButton {
            color: #9B9DA6;
            spacing: 5px;
            font-size: 12px;
        }
        QRadioButton:hover { color: #F0F0F2; }

        /* ── Labels ──────────────────────────── */
        QLabel {
            color: #9B9DA6;
            font-size: 12px;
        }

        /* ── Scrollbars ──────────────────────── */
        QScrollBar:vertical {
            background: #18191C;
            width: 6px;
            border-radius: 3px;
            margin: 0;
        }
        QScrollBar::handle:vertical {
            background: #3D3F48;
            border-radius: 3px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover { background: #5A5C65; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        QScrollBar:horizontal {
            background: #18191C;
            height: 6px;
            border-radius: 3px;
            margin: 0;
        }
        QScrollBar::handle:horizontal {
            background: #3D3F48;
            border-radius: 3px;
            min-width: 20px;
        }
        QScrollBar::handle:horizontal:hover { background: #5A5C65; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

        /* ── Tree View ───────────────────────── */
        QTreeView {
            background-color: #18191C;
            border: none;
            alternate-background-color: #1E2025;
            selection-background-color: rgba(46, 109, 212, 0.18);
        }
        QTreeView::item:hover { background: #1E2025; }
        QTreeView::item:selected { background: rgba(46, 109, 212, 0.18); }
        QTreeView::branch { background: #18191C; }

        /* ── Status bar ──────────────────────── */
        QStatusBar {
            color: #5A5C65;
            font-size: 11px;
            border-top: 0.5px solid #2E3038;
            background: #111214;
            padding: 2px 8px;
        }

        /* ── Menu bar ────────────────────────── */
        QMenuBar {
            background-color: #111214;
            color: #9B9DA6;
            border-bottom: 0.5px solid #2E3038;
            font-size: 12px;
        }
        QMenuBar::item:selected { background: #1E2025; color: #F0F0F2; }
        QMenu {
            background-color: #1E2025;
            border: 0.5px solid #3D3F48;
            color: #F0F0F2;
            font-size: 12px;
        }
        QMenu::item:selected { background: rgba(46, 109, 212, 0.25); }
        QMenu::separator {
            height: 1px;
            background: #2E3038;
            margin: 3px 0;
        }

        /* ── Progress bar ────────────────────── */
        QProgressBar {
            background-color: #18191C;
            border: 0.5px solid #3D3F48;
            border-radius: 4px;
            text-align: center;
            color: #9B9DA6;
            font-size: 11px;
            height: 8px;
        }
        QProgressBar::chunk {
            background-color: #2E6DD4;
            border-radius: 3px;
        }

        /* ── Tooltip ─────────────────────────── */
        QToolTip {
            background-color: #1E2025;
            border: 0.5px solid #3D3F48;
            color: #F0F0F2;
            font-size: 11px;
            padding: 4px 8px;
            border-radius: 4px;
        }
    """)
```

---

## 8. Prioritized Implementation Roadmap

Ordered by **impact ÷ effort**. Items 1–5 can be done in under a day and produce a dramatically improved result.

| # | Task | File(s) | Effort | Impact |
|---|---|---|---|---|
| 1 | **Replace `_setup_style()` with full QSS above** | `main_window.py` | ~30 min | ★★★★★ |
| 2 | **Update pyqtgraph global config tokens** | `main_window.py` | ~10 min | ★★★★☆ |
| 3 | **Shorten tab labels + replace Population checkbox with toggle button** | `main_window.py` | ~20 min | ★★★★☆ |
| 4 | **Merge 4 left panel buttons into compact segmented row** | `main_window.py` | ~30 min | ★★★★☆ |
| 5 | **Set initial splitter width to 220px left / remainder right** | `main_window.py` | ~5 min | ★★★★☆ |
| 6 | **Color-code status column text using `ForegroundRole` in model** | `widgets.py` | ~20 min | ★★★★☆ |
| 7 | **Replace MEA/Vision radio buttons with segmented control** | `similarity_panel.py` | ~20 min | ★★★☆☆ |
| 8 | **Remove vertical gridlines, set row height to 28px, hide row index** | `main_window.py`, `similarity_panel.py` | ~15 min | ★★★☆☆ |
| 9 | **Restyle Refine button with new token colors, reduce height to 32px** | `main_window.py` | ~10 min | ★★★☆☆ |
| 10 | **Remove sidebar toggle button, style QSplitter handle** | `main_window.py` | ~15 min | ★★★☆☆ |
| 11 | **Group ISI controls with separators, add suffix to spinbox** | `standard_plots_panel.py` | ~20 min | ★★☆☆☆ |
| 12 | **Update `_style_plot()` with new axis/grid/title tokens** | `standard_plots_panel.py` | ~20 min | ★★★☆☆ |
| 13 | **Delete commented-out dead code in similarity_panel.py** | `similarity_panel.py` | ~2 min | ★☆☆☆☆ |
| 14 | **Light mode toggle** (invert surface token mapping, add menu item) | `main_window.py` | ~2 hrs | ★★☆☆☆ |

---

*End of specification.*
