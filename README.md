# RGCViewer (Axolotl)

RGCViewer is a PyQt desktop application. Use it to inspect spike-sorted
multi-electrode array recordings from mouse retina and to assign units to
retinal ganglion cell (RGC) types.

The internal name is Axolotl. The repository name is RGCViewer.

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

This clones the repo to `~/.rgcviewer`, creates a virtual environment,
installs all dependencies, and adds the `rgcviewer` command to your PATH.

Run again to update an existing install.

### After install

```bash
rgcviewer
```

| Argument | Effect |
|---|---|
| `--debug` | Write DEBUG logs to the console |
| `--kilosort-dir PATH` | Load this run at start |
| `--dat-file PATH` | Attach a raw `.bin` / `.dat` file for the Raw tab |

### Uninstall

```bash
rm -rf ~/.rgcviewer ~/.local/bin/rgcviewer
```

On Windows:

```powershell
Remove-Item -Recurse -Force $HOME\.rgcviewer, $HOME\.local\bin\rgcviewer.*
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

Unit tests:

```bash
python -m pytest tests/unit/ -v
```

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
Kilosort files may sit in `ksfiles/`. Stimulus analyses are precomputed
`.npy` files in the same folder. The application does not create those files.

## Documents

Read documents in this order:

1. This file — install and start
2. `CLAUDE.md` — experiment, files, analysis traps
3. `docs/AGENTS.md` — developer rules
4. `HANDOFF.md` — current work state
5. `docs/PLAN.md` — fragile zones and open defects

The document map is `docs/README.md`.
