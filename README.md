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

## Keyboard shortcuts

| Keys | Action |
|---|---|
| Ctrl+S | Save the classification to the Vision `.params` |
| Ctrl+F | Search the cell list |
| Ctrl+D / Ctrl+C / Ctrl+E / Ctrl+W / Ctrl+X / Ctrl+A | Mark Duplicate / Clean / Edge / Unsure / Contaminated / Off Array |
| Ctrl+Shift+N | Mark Noisy (was Ctrl+S) |
| Delete | Move the selected cells to Trash |
| Space | Select the next row of the similarity table |

## Documents

Read documents in this order:

1. This file — install and start
2. `CLAUDE.md` — experiment, files, analysis traps
3. `docs/AGENTS.md` — developer rules
4. `docs/PLAN.md` — pickup, fragile zones, open defects

The document map is `docs/README.md`.
