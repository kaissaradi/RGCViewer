# Encore

Encore is a PyQt desktop application. Use it to inspect spike-sorted
multi-electrode array recordings from mouse retina and to assign units to
retinal ganglion cell (RGC) types.

## Quick install

Requires **Python 3.10+** and **git**.

**macOS / Linux:**

```bash
curl -fsSL https://raw.githubusercontent.com/kaissaradi/RGCViewer/main/install.sh | bash
```

**Windows (PowerShell):**

```powershell
irm https://raw.githubusercontent.com/kaissaradi/RGCViewer/main/install.ps1 | iex
```

This clones the repo to `~/.encore`, creates a virtual environment,
installs all dependencies, and adds the `encore` command to your PATH.

Run again to update an existing install.

### Beta channel (testers)

The `beta` channel gets changes before `main`. To move an install to it:

```bash
curl -fsSL https://raw.githubusercontent.com/kaissaradi/RGCViewer/beta/install.sh | ENCORE_BRANCH=beta bash
```

```powershell
$env:ENCORE_BRANCH = "beta"; irm https://raw.githubusercontent.com/kaissaradi/RGCViewer/beta/install.ps1 | iex
```

The install then stays on `beta`. Run the same command again to update
it. To go back, run the command with `ENCORE_BRANCH=main`. File ▸ About
Encore shows the channel and commit; put that line in a bug report.

### After install

```bash
encore
```

| Argument | Effect |
|---|---|
| `--debug` | Write DEBUG logs to the console |
| `--kilosort-dir PATH` | Load this run at start |
| `--dat-file PATH` | Attach a raw `.bin` / `.dat` file for the Raw tab |

### Optional packages (retinanalysis)

The installer puts Encore in its own virtual environment,
`~/.encore/.venv`. That environment does not see packages from conda or
from your system Python. To use a package there, install it into that
environment:

```bash
~/.encore/.venv/bin/pip install -e /path/to/retinanalysis
```

Encore does not need `retinanalysis` today. It only looks up stimulus
timing metadata with it, and no view uses that metadata yet.

### Uninstall

```bash
rm -rf ~/.encore ~/.local/bin/encore
```

On Windows:

```powershell
Remove-Item -Recurse -Force $HOME\.encore, $HOME\.local\bin\encore.*
```

## Developer setup

If you prefer to manage your own environment (or need to run tests):

```bash
git clone https://github.com/kaissaradi/RGCViewer.git
cd RGCViewer
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -e .
```

### Running tests

Install dev dependencies first:

```bash
pip install -r requirements-dev.txt
```

Unit tests (headless):

```bash
QT_QPA_PLATFORM=offscreen python -m pytest tests/unit/ -q
```

CI (`.github/workflows/tests.yml`) runs the unit suite on Python 3.10 and
3.13 for every push to `main` / `dev-testing` and every pull request.

Full suite (slow; some tests need lab mounts):

```bash
python -m pytest tests/ -v
```

## Start without installing

You can also run directly from the repo without `pip install`:

```bash
python main.py
```

The window opens empty. Use **File → Open** to load a run.

## Load a run

A run is a folder such as:

```
<prep>/kilosort25/data006/
```

Example: `20260721A/kilosort25/data006/`.

The folder must contain Vision files (`.neurons`, and usually `.ei`, `.params`).
Kilosort files may sit in `ksfiles/`. Stimulus files are `.npy` in the same
folder. Chirp is precomputed. Grating may be a raw
`spike_times_by_trial` + `trial_parameters` file (`*Grating*.npy` or
`*DSOS*.npy`). The GUI then computes DSI/OSI for each `(bar width, TF)`
that was actually run and stores `grating_computed_cache.pkl`.

Without a raw voltage file the Raw tab shows spike rasters instead: one row
per cell, the selected cell on top, the whole recording, with stimulus
blocks shaded. Zoom in below a few seconds to see every spike. File → Load
Raw Data File switches it to voltage traces.

## Save the classification for Vision

**Ctrl+S** (File → Save Classification to Vision .params) writes the tree
into the `classID` column of the run's `.params` file. Vision shows that
column. A group path becomes `All/<group>/<subgroup>`; a cell in the root
"Unclassified" group becomes `All`.

- Encore changes only the `classID` cells. It keeps the previous file as
  `<name>.params.bak`.
- Encore asks first on the first save of a session, after the file was saved
  elsewhere, when a cell would lose its class, and when the Vision files look
  like they come from another sort.
- Close the run in Vision before you save. Vision saves the same file in place,
  and one of the two saves can be lost.
- File → Load Classification from Vision .params replaces the tree with the
  file's classes.

## Check a classification

Put cells in groups (Ctrl+M, Ctrl+G), or load a Vision classification
(File → Load Classification from Vision .params). Then:

- **Types tab: barcode.** One row per cell, one band per group. Pick the row
  type: STA time course, autocorrelation, or chirp. The most typical cell of
  a group is at the top of its band. A red tick at the right edge marks a
  cell that fits another type's average better than its own group's. Hover
  a row for the numbers; click it to select the cell.
- **Types tab: mosaic atlas.** One small RF mosaic per type, all at the same
  scale. A real type tiles the retina, so neighbours sit about one RF apart
  (NNND ≈ 2). Red outlines: two cells of one type overlap too much (a split
  unit, a duplicate, or a mixed group). Grey rings: unclassified cells that
  look like the type and sit in a gap of its mosaic. Click one to select it.
- **Suggested classes.** Types tab → Suggest classes. Encore learns the 5 types
  the lab names most (ON/OFF brisk sustained, ON/OFF brisk transient, OFF
  transient) from every classified run on the lab share, leaving out the run
  you are checking, and suggests a class for each cell. The first time it
  reads the lab's .params files (~1–2 min); later it uses a cache in
  `~/.encore`. The line under the cell list shows the selected cell's
  suggestion from any tab. Ctrl+J goes to the next cell to review (least
  confident first, and classified cells that a confident suggestion
  disagrees with); Ctrl+Enter accepts. "Accept confident" moves every
  unclassified cell with a suggestion ≥ 80 % — except one whose RF would sit
  on top of a cell already in that class (a type should tile). Cells unlike
  any labelled cell get no suggestion. Nothing is saved until Ctrl+S.
- **Type atlas.** Types tab → Type atlas shows what each named type looks like
  across the lab (mean ± 1 SD of every classified cell), with this run's
  cells of that type on top.
- **Compare a cell with its population.** Ctrl+P opens the population pane.
  The selected cell is drawn in red over its group's time courses, ACGs and
  firing rates. Ctrl+K pins up to four more cells for comparison.
- **Borrow responses from another run.** File → Map Reference Run matches the
  cells of this run to another run of the same retina by their EIs. A cell
  with no chirp here then shows its matched cell's chirp, with a note saying
  where it comes from.
- **Optic disc direction.** Array → Find the Optic Disc fits each cell's axon in
  its EI and shows where the axons converge: a disc point when the fit is
  clear, else a direction only, else nothing. It works best on the 512 array.
  It does not tell dorsal from ventral.
- **Compare DS runs.** Array → Compare DS Runs (or the button on the Grating
  tab) lists every DS grating run of this prep, or of every prep on the
  share: its DS cells (the Grating tab's test), the direction its axons point
  (toward the optic disc), and the array → screen turn, measured from a
  white-noise run of the same prep. The roses pool the ticked runs (one run
  per prep by default: two runs of one piece record the same cells), either
  as the bars moved on the screen or as an angle from the direction to the
  optic disc, which means the same in every prep. The angles are the way the
  bars moved: the Grating tab's angle + 180° (see `docs/PLAN.md`, open
  defects). The first look at a run reads it once (10–40 s); later looks
  read a small summary in its `ksfiles/`.

What the lab's type names mean, with references: `docs/design/rgc_types.md`.

## Keyboard shortcuts

Press F1 (or ?) in Encore for this list, and Shift+F1 (or the ? button in the
header) for what the current tab shows. Shortcuts do not fire while you
type in a text field; Esc leaves the search bar.

| Keys | Action |
|---|---|
| **Cells** | |
| ↑ / ↓ | Previous / next cell in the list |
| Delete / Backspace | Move the selected cells to Trash |
| Ctrl+M | Move the selection to a group (type to filter) |
| Ctrl+Shift+M | Move the selection to the last group used |
| Ctrl+G | Put the selected cells in a new group |
| Ctrl+D / Ctrl+C / Ctrl+E / Ctrl+W / Ctrl+X / Ctrl+A | Mark Duplicate / Clean / Edge / Unsure / Contaminated / Off Array |
| Ctrl+Shift+N | Mark Noisy |
| Space | Next row of the similarity table |
| Ctrl+Return / Ctrl+Enter | Accept the suggested class, go to the next cell to review |
| Ctrl+J | Next cell to review (suggested classes) |
| **Groups** | |
| F2 | Rename the selected group |
| Ctrl+Shift+F | Feature Extraction on the selection |
| **Views** | |
| Ctrl+1 … Ctrl+9 | Go to analysis tab 1 … 9 (Ctrl+0: tab 10) |
| Ctrl+Tab / Ctrl+Shift+Tab | Next / previous analysis tab |
| Ctrl+T | Switch the cell list between tree and table |
| Ctrl+P | Show / hide the population pane beside the cell |
| Ctrl+K | Pin / unpin the cell to compare it with others (up to 4) |
| Ctrl+Shift+K | Clear the pinned cells |
| Ctrl+F | Search the cell list (Esc clears) |
| ← / → | Previous / next EI overlay cell |
| **File** | |
| Ctrl+O | Open a Kilosort run |
| Ctrl+S | Save the classification to the Vision .params |
| F1 / ? | Show these shortcuts |
| Shift+F1 | Explain the current tab |

## Documents

Read documents in this order:

1. This file — install and start
2. `CLAUDE.md` — experiment, files, analysis traps
3. `docs/AGENTS.md` — developer rules
4. `docs/PLAN.md` — pickup, fragile zones, open defects

The document map is `docs/README.md`.
