# RGCViewer (Encore)

RGCViewer is a PyQt desktop application. Use it to inspect spike-sorted
multi-electrode array recordings from mouse retina and to assign units to
retinal ganglion cell (RGC) types.

The internal name is Encore. The repository name is RGCViewer.

## Requirements

- Conda environment name: `rgcviewer`
- Python 3.10
- Qt through `qtpy` (PyQt5)

## Install

1. Clone the repository.
2. Create the environment:

```bash
conda create --name rgcviewer python=3.10
conda activate rgcviewer
pip install -r requirements.txt
```

3. For tests, also install:

```bash
pip install -r requirements-dev.txt
```

## Start the application

1. Activate the environment:

```bash
conda activate rgcviewer
```

2. Start from the repository root:

```bash
python main.py
```

The window opens empty. Use **File → Open** to load a run.

Optional arguments:

| Argument | Effect |
|---|---|
| `--debug` | Write DEBUG logs to the console |
| `--kilosort-dir PATH` | Load this run at start |
| `--dat-file PATH` | Attach a raw `.bin` / `.dat` file for the Raw tab |

The application does **not** reopen the last run at start. File dialogs still
open in the last folder you used.

## Load a run

A run is a folder such as:

```
<prep>/kilosort25/data006/
```

Example: `20260721A/kilosort25/data006/`.

The folder must contain Vision files (`.neurons`, and usually `.ei`, `.params`).
Kilosort files may sit in `ksfiles/`. Stimulus analyses are precomputed
`.npy` files in the same folder. The application does not create those files.

## Tests

Run tests in the `rgcviewer` environment. The base environment does not have
`pytest`.

```bash
conda activate rgcviewer
python -m pytest tests/unit/ -v
```

Full suite (slow; some tests need lab mounts):

```bash
python -m pytest tests/ -v
```

## Documents

Read documents in this order:

1. This file — install and start
2. `CLAUDE.md` — experiment, files, analysis traps
3. `docs/AGENTS.md` — developer rules
4. `docs/PLAN.md` — pickup, fragile zones, open defects

The document map is `docs/README.md`.
