# RGCViewer UX / UI Redesign Plan

*Last updated: 2025-08-12*

This document is the single specification for the visual and interaction
redesign of RGCViewer. It covers design language, color system, layout
changes, keyboard workflows, file browsing, auto-save, and plot theming.
No core analysis code changes — only `src/gui/`, `src/gui/panels/`,
`src/gui/theme.py`, `src/gui/shortcuts.py`, and new assets.

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Color System & Theming](#2-color-system--theming)
3. [Typography & Spacing](#3-typography--spacing)
4. [Layout Overhaul](#4-layout-overhaul)
5. [Experiment Browser](#5-experiment-browser)
6. [Keyboard Shortcuts & Quick Actions](#6-keyboard-shortcuts--quick-actions)
7. [Auto-Save & Session Persistence](#7-auto-save--session-persistence)
8. [Plot Theming](#8-plot-theming)
9. [Tree & Table Refinements](#9-tree--table-refinements)
10. [Tab Navigation](#10-tab-navigation)
11. [Status Bar & Notifications](#11-status-bar--notifications)
12. [Accessibility](#12-accessibility)
13. [Implementation Phases](#13-implementation-phases)

---

## 1. Design Philosophy

### Bauhaus Principles Applied to Scientific Software

The redesign follows three Bauhaus tenets:

- **Form follows function.** Every visual element earns its space by serving
  the classification workflow. Decorative chrome is removed; whitespace and
  alignment do the organizing. Borders become thinner or disappear — panels
  are separated by spacing, not lines.

- **Reduction to essentials.** Controls that are used once per session (Load,
  Save, Export) live in the menu bar or a command palette — not as permanent
  toolbar buttons. The primary viewport maximizes the data. Secondary panels
  (similarity, population) slide in on demand rather than claiming permanent
  real estate.

- **Unity of design.** Every surface — plots, tree nodes, table rows, tab
  headers, dialogs — draws from one token palette. A switch from dark to light
  mode changes the tokens, not the structure. Plots, widgets, and chrome feel
  like they belong to the same application.

### Design Goals

| Goal | Measure |
|---|---|
| Classify a cell in ≤ 3 interactions | Select cell → read plots → drag to group |
| Switch between analysis views without losing context | Tab hotkeys, persistent selection |
| Distinguish 8+ cell populations on any plot | Categorical palette with ≥ 4:1 contrast in both themes |
| Onboard a new lab member in one sitting | Self-documenting UI: labels, tooltips, consistent icons |

---

## 2. Color System & Theming

### Current State

`theme.py` defines `DARK_COLORS` and `LIGHT_COLORS` with 40+ semantic keys.
The system is already well-structured. The redesign extends it, not replaces it.

### Changes

#### 2.1 Token Hierarchy

Organize tokens into three tiers to make the palette systematic:

```
Tier 1 — Surface        bg_base, bg_panel, bg_surface, bg_elevated
Tier 2 — Content        text_primary, text_secondary, text_tertiary, text_disabled
Tier 3 — Interactive    accent, accent_hover, accent_pressed, accent_muted
                         border_subtle, border_default, border_focus
                         status_good_*, status_mua_*, status_noise_*, status_unsort_*
```

New tokens to add:

| Token | Dark | Light | Purpose |
|---|---|---|---|
| `accent_pressed` | `#1E4FA0` | `#3D7DE8` | Active-state feedback on buttons |
| `accent_muted` | `rgba(46,109,212,0.10)` | `rgba(46,109,212,0.08)` | Hover highlight on sidebar rows |
| `border_focus` | `#4A8BEF` | `#2E6DD4` | Keyboard-focus ring (2px) |
| `bg_overlay` | `rgba(0,0,0,0.50)` | `rgba(0,0,0,0.25)` | Modal / dialog scrim |
| `bg_tooltip` | `#282A30` | `#FFFFFF` | Tooltip background |
| `text_tooltip` | `#F0F0F2` | `#111214` | Tooltip text |
| `shadow_sm` | `0 1px 2px rgba(0,0,0,0.3)` | `0 1px 2px rgba(0,0,0,0.08)` | Elevated panels |
| `shadow_md` | `0 4px 12px rgba(0,0,0,0.4)` | `0 4px 12px rgba(0,0,0,0.12)` | Dialogs, dropdowns |

#### 2.2 Dark Mode Palette Refinement

The current dark palette is good. Adjustments:

- `bg_base` stays `#111214` — true near-black, comfortable for long sessions.
- `bg_panel` shifts from `#18191C` → `#1A1B1F` — slightly warmer, 2% more
  contrast against `bg_base`.
- All plot backgrounds use `bg_panel` (not transparent) so they sit on the
  same plane as their containing panel.

#### 2.3 Light Mode Palette Refinement

- `bg_base` shifts from `#F0F2F5` → `#F5F6F8` — softer, less blue-grey.
- `bg_panel` stays `#FFFFFF`.
- Plot backgrounds use `bg_surface` (`#F8F9FA`) to set them apart from the
  white panel without introducing a border.
- All status badges get slightly more saturated backgrounds for readability
  against white.

#### 2.4 Plot Categorical Palette

A fixed 12-color categorical palette for cell populations, tuned for both
themes and for deuteranopia/protanopia. Each color is defined as a
`(dark_variant, light_variant)` pair:

```python
PLOT_CATEGORICAL = [
    ("#4FC3F7", "#0277BD"),   # sky
    ("#81C784", "#2E7D32"),   # green
    ("#FF8A65", "#D84315"),   # coral
    ("#BA68C8", "#7B1FA2"),   # violet
    ("#FFD54F", "#F9A825"),   # gold
    ("#4DD0E1", "#00838F"),   # teal
    ("#F06292", "#C2185B"),   # rose
    ("#A1887F", "#4E342E"),   # brown
    ("#AED581", "#558B2F"),   # lime
    ("#FF8A80", "#C62828"),   # red
    ("#CE93D8", "#6A1B9A"),   # lavender
    ("#80DEEA", "#006064"),   # cyan
]
```

Validation criteria:
- Every pair must pass WCAG AA (4.5:1) against its theme's `bg_panel`.
- No two adjacent colors should be confusable under simulated CVD
  (check with `colorspacious` or Coblis).

#### 2.5 Theme Toggle

- Current: View → Toggle Light/Dark Mode.
- Add: A small sun/moon icon button in the top-right corner of the menu bar
  area or status bar for quick toggling.
- Shortcut: `Ctrl+Shift+T`.
- Transition: 150ms fade on `bg_base` and `bg_panel` using `QPropertyAnimation`
  on a custom property, applied via a thin overlay widget. No jarring flash.

---

## 3. Typography & Spacing

### Current State

Font is set ad-hoc per widget. `PANEL_PADDING = 8`, `CTRL_SPACING = 6`,
`ROW_HEIGHT = 28`.

### Changes

#### 3.1 Type Scale

Define a four-step scale using the system monospace stack:

| Role | Size | Weight | Use |
|---|---|---|---|
| `type_heading` | 13px | 600 (DemiBold) | Panel titles, group names |
| `type_body` | 12px | 400 (Normal) | Labels, controls, table cells |
| `type_caption` | 11px | 400 | Status text, axis labels, tooltips |
| `type_mono` | 11px | 400, monospace | Cluster IDs, numeric readouts |

Font family: `"Inter", "SF Pro Text", "Segoe UI", system-ui, sans-serif` for
UI chrome. Plots keep their current font but inherit `type_caption` size.

#### 3.2 Spacing Scale

Replace magic numbers with a 4px base grid:

| Token | Value | Use |
|---|---|---|
| `sp_1` | 4px | Inline padding (icon to label) |
| `sp_2` | 8px | Intra-component padding (= current `PANEL_PADDING`) |
| `sp_3` | 12px | Between controls in a group |
| `sp_4` | 16px | Between sections |
| `sp_5` | 24px | Panel margins, major separations |

#### 3.3 Border Radius

Uniform `radius_sm = 4px` for buttons, inputs, tags. `radius_md = 6px` for
panels, cards, dialogs. No rounded-rectangle overuse — sharp corners on
tab headers and tree items to maintain Bauhaus geometry.

---

## 4. Layout Overhaul

### Current State

Horizontal `QSplitter` → left sidebar (220px) + right pane.
Right pane: `QTabWidget` (8 tabs) + population context panel.
Sidebar: filter, tree/table toggle, search, stacked tree/table, similarity.

### Changes

#### 4.1 Three-Column Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  Menu Bar                                            [☀] [⌘K]    │
├──────────┬──────────────────────────────────┬────────────────────────┤
│ Sidebar  │  Analysis Tabs                   │  Context Panel        │
│ (240px)  │  (flexible)                      │  (300px, collapsible) │
│          │                                  │                       │
│ ┌──────┐ │  ┌────────────────────────────┐  │  RF Mosaic            │
│ │Browse│ │  │                            │  │  ────────────         │
│ │ Tree │ │  │   Active Plot              │  │  [mosaic canvas]      │
│ │ /    │ │  │                            │  │                       │
│ │Table │ │  │                            │  │  Population           │
│ │      │ │  │                            │  │  ────────────         │
│ │      │ │  └────────────────────────────┘  │  [timecourse]         │
│ │      │ │                                  │  [acg]                │
│ │      │ │                                  │  [fr bar]             │
│ ├──────┤ │                                  │                       │
│ │Simil-│ │                                  │                       │
│ │arity │ │                                  │                       │
│ └──────┘ │                                  │                       │
├──────────┴──────────────────────────────────┴────────────────────────┤
│  Status Bar     [progress]     [auto-save ✓]     [cells: 342]      │
└─────────────────────────────────────────────────────────────────────┘
```

Key changes:
- **Context panel** (RF mosaic + population plots) becomes a dedicated
  right-side column rather than a toggle-in overlay. It collapses with a
  single click or `Ctrl+\` and remembers its state.
- **Sidebar** width increases from 220 → 240px to fit longer group paths.
- **Experiment browser** (section 5) replaces the top of the sidebar when
  active.

#### 4.2 Splitter Styling

- Remove visible splitter handles. Replace with a 1px `border_subtle` line
  and a 4px invisible drag zone on hover.
- Cursor changes to resize on hover over the drag zone.
- Double-click a splitter edge to reset to default proportions.

#### 4.3 Panel Headers

Each panel (sidebar sections, context panel sections) gets a minimal header:

```
  RF Mosaic                                    [⊟]
  ─────────────────────────────────────────────────
```

- 13px `type_heading` in `text_primary`.
- Collapse toggle `[⊟]`/`[⊞]` aligned right, `text_tertiary`, no border.
- 1px `border_subtle` separator below.
- No background fill — the header is part of the panel surface.

---

## 5. Experiment Browser

### Current State

File → Load Kilosort Directory opens a `QFileDialog`. `recent_paths.py`
remembers the last-used directory per category. No way to quickly switch
between preparations without going through the system file dialog.

### Changes

#### 5.1 Home Directory

A user-configurable **home directory** for experiments, persisted via
`QSettings` under key `experiment/home_dir`. Set it once through:
- File → Set Experiment Home…
- Or by right-clicking a loaded directory → "Set as Home"

Example: `/data/retina/` or `/Volumes/Array/preps/`

#### 5.2 Experiment Browser Panel

A new collapsible panel at the top of the sidebar, above the tree/table:

```
┌─ Experiments ─────────────────── [⌂] [↻] ─┐
│ 🔍 Filter...                               │
│                                             │
│ ▸ 20260721A                                 │
│   ▸ kilosort25/                             │
│     data006  SpatialNoise    ✓ loaded       │
│     data007  ChirpStimulus                  │
│     data008  GratingDSOS_ks                 │
│     data010-013  (concat)                   │
│                                             │
│ ▸ 20260715B                                 │
│ ▸ 20260710A                                 │
│                                             │
│ ── Recent ──────────────────────────────── │
│   /data/retina/20260701C/kilosort25/data003│
│   /data/retina/20260628A/kilosort25/data011│
└─────────────────────────────────────────────┘
```

Behavior:
- On launch, scans `home_dir` for subdirectories matching the `<prep>/kilosort25/`
  pattern.
- Each prep expands to show its sorted runs. Run protocol names are read from
  `<prep>.json` if it exists in the `stimuli/` subfolder.
- **Single-click** a run to load it (replaces current data). If unsaved
  changes exist, prompt to save first (or auto-save if enabled).
- **Double-click** a prep to load its concatenated sort if one exists.
- `[⌂]` button opens a dialog to change the home directory.
- `[↻]` button rescans the home directory.
- A `Filter...` text field filters preps and runs by name, protocol, or date.
- The `Recent` section shows the last 8 loaded paths (from `QSettings`),
  regardless of whether they're inside the home directory.
- The browser is collapsible to one line (`Experiments ▸`) and remembers
  its expanded/collapsed state.

#### 5.3 Drag-to-Load

Dragging a folder from the OS file manager onto the main window loads it as
a kilosort directory (same as File → Load Kilosort Directory). Implemented
via `dragEnterEvent` / `dropEvent` on `MainWindow`.

#### 5.4 Quick-Switch Shortcut

`Ctrl+O` opens the experiment browser's filter field in focus, ready for
typing. If the browser panel is collapsed, it expands first.

---

## 6. Keyboard Shortcuts & Quick Actions

### Current State

`KeyForwarder` handles: Space (similarity), Left/Right (EI frames),
Up/Down (selection), Delete (trash), Ctrl+F (search), Ctrl+{D,C,E,W,S,X,A}
(status marking).

### Changes

#### 6.1 Command Palette

`Ctrl+K` (or `Ctrl+Shift+P`) opens a floating command palette — a
search-as-you-type dialog listing every available action:

```
┌──────────────────────────────────────────────┐
│ 🔍 Type a command...                         │
│                                              │
│   Load Kilosort Directory       Ctrl+O       │
│   Load Classification           Ctrl+Shift+L │
│   Save Classification           Ctrl+S       │
│   Save Classification As...     Ctrl+Shift+S │
│   Export Results                              │
│   ──────────────────────────────────────────  │
│   Group Selected Cells          Ctrl+G       │
│   Move to Group...              Ctrl+M       │
│   Rename Group                  F2           │
│   Flatten Group                              │
│   ──────────────────────────────────────────  │
│   Toggle Theme                  Ctrl+Shift+T │
│   Toggle Population Panel       Ctrl+\       │
│   Feature Extraction...                      │
│   Map Reference Run...                       │
└──────────────────────────────────────────────┘
```

Implementation:
- `QDialog` with `Qt.FramelessWindowHint | Qt.Popup`, anchored top-center.
- `QLineEdit` at top, `QListWidget` below, filtered on every keystroke.
- Enter activates the selected action, Escape closes.
- Actions are registered as `(name, shortcut, callable)` tuples; the palette
  is generated from the registry, so it is always complete and consistent.

#### 6.2 New Shortcuts

| Shortcut | Action | Context |
|---|---|---|
| `Ctrl+S` | Save classification (to current file, or Save As if none) | Global |
| `Ctrl+Shift+S` | Save classification as… | Global |
| `Ctrl+O` | Focus experiment browser filter | Global |
| `Ctrl+Shift+L` | Load classification file | Global |
| `Ctrl+G` | Group selected cells into a new named group | Tree view |
| `Ctrl+M` | Move selected cells to group (opens quick picker) | Tree/Table |
| `F2` | Rename selected group | Tree view |
| `Ctrl+Z` | Undo last tree operation | Global |
| `Ctrl+Shift+Z` | Redo | Global |
| `Ctrl+\` | Toggle context/population panel | Global |
| `Ctrl+Shift+T` | Toggle light/dark theme | Global |
| `Ctrl+K` | Open command palette | Global |
| `Ctrl+1`…`Ctrl+8` | Switch to tab 1–8 (Standard…Raw) | Global |
| `Ctrl+Tab` | Next tab | Global |
| `Ctrl+Shift+Tab` | Previous tab | Global |
| `Ctrl+L` | Toggle tree/table view | Sidebar |
| `Escape` | Clear selection / close palette / collapse browser | Context-dependent |

#### 6.3 Quick Group Picker

When the user presses `Ctrl+M` with cells selected, a small popup appears
at the cursor position:

```
┌─ Move to ────────────────────────┐
│ 🔍 Filter groups...              │
│                                  │
│   All/OnP/                       │
│   All/OffP/                      │
│   All/SBC/                       │
│   All/DSGCs/OnOff/               │
│   ── New Group ──                │
│   + Create "..."                 │
└──────────────────────────────────┘
```

- Lists all existing tree groups, filterable by typing.
- "New Group" option at the bottom creates a new group and moves in one step.
- Enter to confirm, Escape to cancel.

#### 6.4 Status Marking Bar

When cells are selected, a thin horizontal bar appears below the tab widget
showing the available status marks as clickable chips:

```
  Mark as:  [Clean] [Duplicate] [Edge] [Unsure] [Noisy] [Contaminated] [Off Array]
```

Each chip shows its `Ctrl+` shortcut as a subscript. Clicking a chip or
pressing the shortcut applies the status immediately. The bar fades in/out
with selection changes (100ms transition).

#### 6.5 Multi-Select Drag

Selecting multiple cells (Shift+Click range, Ctrl+Click toggle) and
dragging them in the tree view moves them as a batch. Currently only
single-item drag-and-drop is fluid; this extends it to multi-select.

#### 6.6 Undo / Redo Stack

Tree operations (move, group, rename, delete, flatten, status change) push
onto an undo stack. `Ctrl+Z` / `Ctrl+Shift+Z` navigate it. Stack depth: 50
operations. Implementation: store `(operation_type, before_state, after_state)`
tuples. The before/after state is a lightweight snapshot of the affected items'
parent paths and properties, not a full model clone.

---

## 7. Auto-Save & Session Persistence

### Current State

Save is manual: File → Save Classification. No auto-save, no session
restoration.

### Changes

#### 7.1 Auto-Save Classification

A new setting in File → Preferences (or a menu toggle):

- **Auto-save interval**: Off / 2 min / 5 min / 10 min (default: 5 min).
- Auto-save writes to `<current_classification_path>.autosave` — a sidecar
  file, never overwriting the user's explicit save.
- On load, if a `.autosave` file is newer than the `.classification_MC.txt`,
  prompt: *"An auto-saved version exists from [timestamp]. Restore it?"*
- Auto-save status shown in the status bar: `Auto-saved 2m ago ✓` or
  `Auto-save: off`.

#### 7.2 Session State Persistence

On quit, persist to `QSettings`:
- Current classification file path.
- Active tab index.
- Sidebar collapsed/expanded state.
- Context panel collapsed/expanded state.
- Splitter positions.
- Theme (already done, verify).
- Experiment browser expanded/collapsed state and last filter.
- Selected cluster IDs.

On launch with no arguments, offer to restore the last session:
*"Restore previous session? [20260721A/kilosort25/data006]"* — a non-modal
banner at the top of the window for 10 seconds, dismissible.

#### 7.3 Save Indicator

The window title reflects save state:

```
RGC Viewer — 20260721A / data006          # saved
RGC Viewer — 20260721A / data006 ●        # unsaved changes
```

The `●` is `text_secondary` when auto-save is on (changes are safe) and
`status_noise_text` (red/orange) when auto-save is off.

---

## 8. Plot Theming

### Current State

`configure_pyqtgraph_theme` sets global `bg` and `fg`. Individual panels
call `restyle_plots(colors)` on theme toggle. Matplotlib canvases
(`MplCanvas`) set their own colors independently.

### Changes

#### 8.1 Unified Plot Style Function

A single function in `theme.py` that configures any matplotlib `Figure`
or pyqtgraph `PlotWidget` to match the current theme:

```python
def apply_plot_theme(widget, colors: dict) -> None:
    """Style a PlotWidget or MplCanvas to match the current palette."""
    ...
```

Called on creation and on theme toggle.

#### 8.2 Pyqtgraph Plot Style

All `pg.PlotWidget` instances:

| Element | Token |
|---|---|
| Background | `bg_panel` |
| Axis lines | `border_default` |
| Axis labels, tick labels | `text_secondary`, `type_caption` size |
| Title | `text_primary`, `type_body` size |
| Grid (if shown) | `border_subtle`, 0.3 opacity |
| Data line (single) | `plot_line` |
| Data scatter | `plot_scatter` |
| Crosshair / hover | `plot_highlight` |

No outer border on plot widgets — they sit flush against their panel
background.

#### 8.3 Matplotlib Plot Style

All `MplCanvas` instances (EI, STA, population plots, UMAP):

| Element | Token |
|---|---|
| `figure.facecolor` | `bg_panel` |
| `axes.facecolor` | `bg_surface` (subtle differentiation) |
| `axes.edgecolor` | `border_subtle` |
| `axes.labelcolor` | `text_secondary` |
| `xtick.color`, `ytick.color` | `text_tertiary` |
| `text.color` | `text_primary` |
| `legend.facecolor` | `bg_elevated` |
| `legend.edgecolor` | `border_subtle` |
| Colormap (sequential) | Custom: `bg_panel` → `accent` (dark) or `bg_surface` → `accent` (light) |

Axes spines: only left and bottom, weight 0.5px, color `border_subtle`.
Remove top and right spines globally.

#### 8.4 UMAP Scatter Theme

The UMAP scatter plot is the most visually prominent. Special treatment:

- Cluster colors use `PLOT_CATEGORICAL` (section 2.4).
- Unassigned cells: `text_disabled` at 30% opacity.
- Selected cells: full opacity + `plot_highlight` outline ring (2px).
- Background: `bg_surface`.
- Point size: 6px default, 8px on hover, 10px when selected.
- Lasso selection: `accent` outline, `accent_muted` fill.

#### 8.5 EI Electrode Map Theme

- Electrode dots: `text_disabled` (inactive), scaled by amplitude.
- Active electrodes: use a sequential colormap from `accent_muted` → `accent`.
- Retina photo overlay: 40% opacity over `bg_surface`.
- Mountain plot: line color `plot_line`, fill `accent_muted`.

#### 8.6 STA RF Image Theme

- Colormap: `RdBu_r` for ON/OFF — but verify it works on both bg colors.
  If not, build a custom diverging map anchored to `bg_surface` at center.
- RF ellipse overlay: `plot_highlight`, 1.5px, dashed.
- Temporal filter plot: `plot_line` for the filter, `plot_shadow` for
  confidence interval fill.

---

## 9. Tree & Table Refinements

### Current State

Tree: `QStandardItemModel` with bold folder items (hard-coded `#3C3C3C` bg)
and leaf items. Table: `HighlightStatusPandasModel` with per-status row
colors.

### Changes

#### 9.1 Tree Visual Cleanup

- **Remove hard-coded folder background.** Use `bg_elevated` from the theme
  palette instead of `#3C3C3C`.
- **Indentation**: 16px per level (down from Qt default 20px) to save space.
- **Icons**: Replace folder/file icons with minimal geometric shapes:
  - Group: small filled circle in `accent`, 6px diameter.
  - Cell: hollow circle in `text_tertiary`, 5px diameter.
  - Trash: small `×` in `status_noise_text`.
- **Drag indicator**: A 2px `accent` line between items during drag, instead
  of the default highlight rectangle.
- **Cell count badge**: Each group node shows a small count `(n)` in
  `text_tertiary` after its name: `OnP (12)`.

#### 9.2 Table Visual Cleanup

- **Row height**: Reduce from 28px → 24px to show more cells.
- **Header**: Sticky, `bg_elevated`, `type_caption`, uppercase, `text_tertiary`.
  1px `border_subtle` below.
- **Alternating rows**: `bg_panel` / `bg_surface` — subtle alternation, not
  striping.
- **Status column**: Replace text with a small colored dot (6px circle) using
  the status color tokens. Tooltip shows the full status name.
- **Sortable columns**: Click header to sort; small ▲/▼ indicator in
  `text_tertiary`. Current sort column header text in `text_primary`.

#### 9.3 Inline Search

`Ctrl+F` search field behavior:
- As the user types, matching cells/groups are highlighted in-place (both
  tree and table).
- Non-matching items dim to `text_disabled` rather than being hidden — context
  is preserved.
- Enter jumps to the next match, Shift+Enter to the previous.
- Escape clears the search and restores full opacity.

---

## 10. Tab Navigation

### Current State

8 tabs (Standard, Chirp, Grating, EI, STA, UMAP, Waveforms, Raw) in a
`QTabWidget`. No keyboard shortcuts for switching.

### Changes

#### 10.1 Tab Bar Styling

- Tab shape: rectangular, no rounded corners — Bauhaus geometry.
- Active tab: `text_primary` label, 2px `accent` underline at the bottom edge.
  No fill change — the tab content area already has `bg_panel` background.
- Inactive tabs: `text_secondary` label, no underline.
- Hover: `text_primary` label, 1px `border_subtle` underline.
- Tab height: 32px. Padding: `sp_2` horizontal, `sp_1` vertical.
- No tab close buttons — tabs are fixed.

#### 10.2 Tab Shortcuts

`Ctrl+1` through `Ctrl+8` switch to tabs 1–8:

| Shortcut | Tab |
|---|---|
| `Ctrl+1` | Standard |
| `Ctrl+2` | Chirp |
| `Ctrl+3` | Grating |
| `Ctrl+4` | EI |
| `Ctrl+5` | STA |
| `Ctrl+6` | UMAP |
| `Ctrl+7` | Waveforms |
| `Ctrl+8` | Raw |

`Ctrl+Tab` / `Ctrl+Shift+Tab` cycle forward/backward.

#### 10.3 Tab Memory

Each tab remembers its scroll position and zoom level. Switching away and
back restores the exact view state. Implementation: each panel stores its
view state in instance variables; the `QTabWidget.currentChanged` signal
triggers save/restore.

---

## 11. Status Bar & Notifications

### Current State

`QStatusBar` with a permanent `QProgressBar` for cache progress.

### Changes

#### 11.1 Status Bar Layout

```
│ Ready                 │ ████░░░░ 47%  │ Auto-saved 2m ago ✓  │ Cells: 342 │ Dark │
│                       │ [progress]    │ [auto-save status]   │ [count]    │[theme]│
```

Sections:
1. **Message area** (left, stretches): Transient messages ("Loaded data006",
   "Classification saved"). Messages auto-clear after 5 seconds.
2. **Progress bar**: Only visible during long operations. Width: 120px.
   Color: `accent`. Track color: `bg_elevated`.
3. **Auto-save indicator**: Persistent. Shows time since last auto-save and
   a checkmark. Clickable to force an immediate save.
4. **Cell count**: Shows total loaded cells and selected count:
   `342 cells (5 selected)`.
5. **Theme toggle**: Small sun/moon icon, clickable.

#### 11.2 Toast Notifications

For important but non-blocking messages (e.g., "Auto-save failed",
"Reference mapping complete"), show a toast notification:

- Appears in the bottom-right corner, above the status bar.
- 300px wide, `bg_elevated`, `shadow_md`, `radius_md`.
- Auto-dismisses after 4 seconds. Click to dismiss immediately.
- Stacks vertically if multiple toasts appear.
- Types: info (`accent`), success (`status_good_text`), warning
  (`status_mua_text`), error (`status_noise_text`) — a 3px left border.

---

## 12. Accessibility

#### 12.1 Focus Indicators

Every interactive element gets a visible focus ring on keyboard navigation:
- 2px `border_focus` ring, 2px offset from the element.
- Applied via `:focus` pseudo-state in QSS.
- Tree/table rows, buttons, tabs, inputs, sliders — all covered.

#### 12.2 Minimum Contrast

All text meets WCAG AA contrast ratio:
- `text_primary` on `bg_panel`: ≥ 7:1 (AAA).
- `text_secondary` on `bg_panel`: ≥ 4.5:1 (AA).
- `text_tertiary` on `bg_panel`: ≥ 3:1 (AA for large text / non-text).

#### 12.3 Keyboard Navigability

Every action reachable by mouse is also reachable by keyboard:
- Menu bar: `Alt+F`, `Alt+A`, `Alt+V`.
- Tabs: `Ctrl+1`–`Ctrl+8`.
- Command palette: `Ctrl+K`.
- Tree/table: Arrow keys, Enter to expand, Delete to trash.
- Dialogs: Tab between fields, Enter to confirm, Escape to cancel.

#### 12.4 Tooltips

Every icon-only button gets a tooltip describing its action and shortcut:
- Format: `"Toggle theme (Ctrl+Shift+T)"`.
- Delay: 400ms. Duration: 4 seconds.
- Styled with `bg_tooltip` / `text_tooltip`, `radius_sm`, `shadow_sm`.

---

## 13. Implementation Phases

### Phase 1 — Foundations (Theme + Spacing + Shortcuts)

**Files touched:** `theme.py`, `shortcuts.py`, `main_window.py`

1. Add new tokens to `DARK_COLORS` / `LIGHT_COLORS`.
2. Define `PLOT_CATEGORICAL` palette.
3. Create spacing and typography constants.
4. Implement `apply_plot_theme()` for both pyqtgraph and matplotlib.
5. Register all new keyboard shortcuts (tab switching, save, command palette
   placeholder).
6. Update the QSS stylesheet generator to use the new tokens.
7. Add theme toggle button to status bar / menu bar corner.

Estimated scope: ~400 lines changed across 3 files.

### Phase 2 — Layout + Tab Bar

**Files touched:** `main_window.py`, panel modules

1. Restructure to three-column layout with collapsible context panel.
2. Restyle tab bar (underline active, remove chrome).
3. Restyle splitters (invisible handles, 1px borders).
4. Add panel headers with collapse toggles.
5. Implement tab memory (save/restore view state per tab).

Estimated scope: ~600 lines changed, mostly `main_window.py`.

### Phase 3 — Plot Theming

**Files touched:** all `panels/*.py`, `theme.py`, `widgets.py`

1. Apply `apply_plot_theme()` to every `PlotWidget` and `MplCanvas`.
2. Remove hard-coded colors from panel modules.
3. Implement `PLOT_CATEGORICAL` in UMAP and population plots.
4. Theme the EI electrode map, STA RF image, and mountain plot.
5. Theme the chirp PSTH, grating polar plot, and ACG/ISI/FR plots.

Estimated scope: ~300 lines changed across 10 files.

### Phase 4 — Tree & Table Polish

**Files touched:** `main_window.py`, `widgets.py`, `callbacks.py`

1. Replace hard-coded tree colors with theme tokens.
2. Implement geometric icons (circles, ×) via `ClusterTreeDelegate`.
3. Add cell count badges to group nodes.
4. Restyle table headers and alternating rows.
5. Status dot column in table.
6. Enhanced inline search with dimming.

Estimated scope: ~250 lines changed across 3 files.

### Phase 5 — Experiment Browser

**Files touched:** new `panels/experiment_browser.py`, `main_window.py`,
`callbacks.py`, `recent_paths.py`

1. Build `ExperimentBrowser` widget.
2. Home directory setting (QSettings, File menu action).
3. Directory scanning and protocol detection.
4. Filter field.
5. Recent paths section.
6. Integration into sidebar.
7. Drag-to-load on `MainWindow`.

Estimated scope: ~500 lines new, ~100 lines changed.

### Phase 6 — Quick Actions & Command Palette

**Files touched:** new `gui/command_palette.py`, `shortcuts.py`,
`main_window.py`, `callbacks.py`

1. Build action registry.
2. Build `CommandPalette` dialog.
3. Build quick group picker (`Ctrl+M`).
4. Build status marking bar.
5. Multi-select drag in tree.
6. Undo/redo stack for tree operations.

Estimated scope: ~600 lines new, ~200 lines changed.

### Phase 7 — Auto-Save & Session Persistence

**Files touched:** `callbacks.py`, `main_window.py`, `recent_paths.py`

1. Auto-save timer and sidecar file logic.
2. Auto-save settings UI.
3. Session state save on quit.
4. Session restore prompt on launch.
5. Save indicator in window title.
6. Status bar auto-save display.

Estimated scope: ~300 lines new, ~100 lines changed.

### Phase 8 — Accessibility & Polish

**Files touched:** `theme.py`, `main_window.py`, all panels

1. Focus ring styles in QSS.
2. Contrast audit — adjust any failing tokens.
3. Tooltip sweep — every icon-only button.
4. Toast notification widget.
5. Final QSS pass — remove any remaining hard-coded values.
6. Theme transition animation.

Estimated scope: ~200 lines new, ~150 lines changed.

---

## Appendix A: File Map

New files created by this plan:

| Path | Purpose |
|---|---|
| `src/gui/panels/experiment_browser.py` | Experiment browser sidebar panel |
| `src/gui/command_palette.py` | Command palette dialog and action registry |
| `src/gui/undo.py` | Undo/redo stack for tree operations |
| `src/gui/toast.py` | Toast notification widget |
| `docs/PLAN.md` | This document |

Files with major changes:

| Path | What changes |
|---|---|
| `src/gui/theme.py` | New tokens, categorical palette, `apply_plot_theme()` |
| `src/gui/shortcuts.py` | Full shortcut registry, new bindings |
| `src/gui/main_window.py` | Layout restructure, session persistence, status bar |
| `src/gui/callbacks.py` | Auto-save, experiment browser integration |
| `src/gui/widgets/widgets.py` | Tree delegate update, table styling |
| `src/gui/panels/*.py` | Each panel: replace hard-coded colors with theme calls |

## Appendix B: Shortcut Reference Card

| Category | Shortcut | Action |
|---|---|---|
| **File** | `Ctrl+O` | Open experiment browser / filter |
| | `Ctrl+S` | Save classification |
| | `Ctrl+Shift+S` | Save classification as… |
| | `Ctrl+Shift+L` | Load classification file |
| **Navigation** | `Ctrl+1`–`Ctrl+8` | Switch to tab 1–8 |
| | `Ctrl+Tab` | Next tab |
| | `Ctrl+Shift+Tab` | Previous tab |
| | `Ctrl+F` | Focus search / filter |
| | `Ctrl+L` | Toggle tree / table view |
| | `Ctrl+\` | Toggle context panel |
| | `↑` / `↓` | Move selection in tree / table |
| | `←` / `→` | EI frame navigation |
| **Editing** | `Ctrl+G` | Group selected cells |
| | `Ctrl+M` | Move to group (quick picker) |
| | `F2` | Rename selected group |
| | `Delete` | Move to Trash |
| | `Ctrl+Z` | Undo |
| | `Ctrl+Shift+Z` | Redo |
| **Status** | `Ctrl+D` | Mark Duplicate |
| | `Ctrl+C` | Mark Clean |
| | `Ctrl+E` | Mark Edge |
| | `Ctrl+W` | Mark Unsure |
| | `Ctrl+Shift+S` | Mark Noisy |
| | `Ctrl+X` | Mark Contaminated |
| | `Ctrl+A` | Mark Off Array |
| **View** | `Ctrl+Shift+T` | Toggle light / dark theme |
| | `Ctrl+K` | Command palette |
| | `Space` | Similarity panel action |

> **Note on shortcut conflicts:** `Ctrl+S` is reassigned from "Mark Noisy" to
> "Save" (the universally expected binding). "Mark Noisy" moves to
> `Ctrl+Shift+N`. `Ctrl+A` for "Mark Off Array" conflicts with Select All —
> since Select All is less critical in this app (multi-select is via
> Shift/Ctrl+Click), the status binding takes priority, but revisit if users
> report friction.
