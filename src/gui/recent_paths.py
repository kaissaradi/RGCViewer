"""Remember where the user last browsed, so file dialogs open somewhere useful.

Every file dialog in the app funnels its starting location through
:func:`last_dir` and reports the result back through :func:`remember_dir`.
Locations persist across sessions via QSettings, so reopening the app still
lands on the dataset you were working in rather than your home folder.

Directories are tracked per *category* (kilosort, vision, raw, …) with a
shared most-recent fallback: the first time you open a dialog you have never
used, it starts wherever you last were, and from then on that dialog
remembers its own place.
"""

from __future__ import annotations

import logging
from pathlib import Path

from qtpy.QtCore import QSettings

logger = logging.getLogger(__name__)

__all__ = [
    "last_dir", "remember_dir", "suggested_path",
    "remember_dataset", "last_dataset", "forget_dataset",
]

_ORG = "RGCViewer"
_APP = "RGCViewer"
_GLOBAL_KEY = "paths/_last"
# The dataset itself, not just a dialog's starting folder — this is what lets
# the app reopen where you left off instead of asking every launch.
_DATASET_KEY = "dataset/last_kilosort_dir"
_DATASET_DAT_KEY = "dataset/last_dat_file"


def _settings() -> QSettings:
    return QSettings(_ORG, _APP)


def _as_existing_dir(value) -> Path | None:
    """Coerce *value* to a directory that exists right now, else None.

    Remembered paths go stale — an external drive gets unplugged, a folder is
    renamed. Handing a dead path to QFileDialog silently drops the user
    somewhere arbitrary, so every candidate is checked before use.
    """
    if not value:
        return None
    try:
        path = Path(str(value))
    except (TypeError, ValueError):
        return None
    if path.is_file():
        path = path.parent
    try:
        return path if path.is_dir() else None
    except OSError:
        # Unreachable network path, permission error on a disconnected drive.
        return None


def remember_dir(path, key: str = None) -> None:
    """Record *path* as the most recent location, globally and for *key*.

    Accepts a file or a directory; files are stored as their parent, which is
    what a dialog should reopen to.
    """
    resolved = _as_existing_dir(path)
    if resolved is None:
        return
    try:
        settings = _settings()
        settings.setValue(_GLOBAL_KEY, str(resolved))
        if key:
            settings.setValue(f"paths/{key}", str(resolved))
    except Exception:
        # A broken settings backend must never take a file dialog down with it.
        logger.debug("remember_dir: could not persist %s", resolved, exc_info=True)


def last_dir(main_window=None, key: str = None) -> str:
    """Best starting directory for a dialog, as a string for QFileDialog.

    Preference order: this dialog's own last location, then wherever the user
    most recently browsed, then the directory of the loaded dataset, then home.
    """
    candidates = []

    try:
        settings = _settings()
        if key:
            candidates.append(settings.value(f"paths/{key}"))
        candidates.append(settings.value(_GLOBAL_KEY))
    except Exception:
        logger.debug("last_dir: could not read settings", exc_info=True)

    # The currently loaded dataset is a good answer before anything is
    # remembered — a fresh install should still open next to your data.
    dm = getattr(main_window, "data_manager", None) if main_window else None
    for attr in ("kilosort_dir", "vision_dir"):
        value = getattr(dm, attr, None) if dm else None
        if value:
            candidates.append(value)

    for candidate in candidates:
        resolved = _as_existing_dir(candidate)
        if resolved is not None:
            return str(resolved)

    return str(Path.home())


def remember_dataset(kilosort_dir, dat_file=None) -> None:
    """Record the dataset that just loaded, so the next launch can reopen it.

    Distinct from :func:`remember_dir`, which only steers file dialogs. Only
    called after a load actually succeeds — remembering a directory that failed
    to open would make the app fail the same way on every subsequent start.
    """
    resolved = _as_existing_dir(kilosort_dir)
    if resolved is None:
        return
    try:
        settings = _settings()
        settings.setValue(_DATASET_KEY, str(resolved))
        # dat_file is optional and is a file, not a directory, so it cannot go
        # through _as_existing_dir.
        if dat_file and Path(str(dat_file)).is_file():
            settings.setValue(_DATASET_DAT_KEY, str(dat_file))
        else:
            settings.remove(_DATASET_DAT_KEY)
    except Exception:
        logger.debug("remember_dataset: could not persist", exc_info=True)


def last_dataset():
    """``(kilosort_dir, dat_file)`` from the last successful load.

    Both are ``None`` when nothing is remembered or the remembered location is
    gone — an unplugged drive or a renamed folder must not stop the app
    starting.
    """
    try:
        settings = _settings()
        ks = _as_existing_dir(settings.value(_DATASET_KEY))
        dat = settings.value(_DATASET_DAT_KEY)
    except Exception:
        logger.debug("last_dataset: could not read settings", exc_info=True)
        return None, None
    if ks is None:
        return None, None
    try:
        dat = str(dat) if dat and Path(str(dat)).is_file() else None
    except (TypeError, ValueError, OSError):
        dat = None
    return str(ks), dat


def forget_dataset() -> None:
    """Drop the remembered dataset (used when reopening it fails)."""
    try:
        settings = _settings()
        settings.remove(_DATASET_KEY)
        settings.remove(_DATASET_DAT_KEY)
    except Exception:
        logger.debug("forget_dataset: could not clear settings", exc_info=True)


def suggested_path(main_window, key: str, filename: str, preferred_dir=None) -> str:
    """A full path to pre-fill a save dialog with.

    *preferred_dir* wins when supplied and real — callers pass the loaded
    dataset directory so exports default beside the data they describe.
    """
    base = _as_existing_dir(preferred_dir)
    if base is None:
        base = Path(last_dir(main_window, key))
    return str(base / filename)
