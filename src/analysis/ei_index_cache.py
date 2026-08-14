"""Persist the .ei seek table so opening a run does not rebuild it.

``EIReader`` locates each cell by walking the .ei file record by record: seek to
a record start, read the 8-byte ``(cell_id, n_spikes)`` header, skip a fixed
stride, repeat. The records are one EI apart — about 825 KB on a 512 array — so
this is ~700 tiny reads scattered across a 600 MB file.

On local NVMe that is free. Over the lab's CIFS mount every one of those reads
is its own SMB round trip at ~9 ms, and the scan alone costs 6.5 s of a 19.5 s
open — the single largest item in it. It is not readahead: forcing
``POSIX_FADV_RANDOM`` only moved it from 6.54 s to 5.73 s. It is the round
trips, and the only way to stop paying them is to not make them twice.

The table is small (a few hundred int pairs, ~25 KB pickled) and depends on
nothing but the .ei file, so it caches cleanly next to it and is keyed on that
file's size and mtime. Anything unexpected — missing file, stale key, bad
pickle, unwritable directory — falls back to the scan, which is always correct.
"""

import logging
import os
import pickle
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Bump when the payload's shape changes in a way that makes older files wrong.
INDEX_VERSION = 1

VERSION_KEY = "__index_version__"
SUFFIX = ".ei_index.pkl"


def cache_path(ei_path):
    """Where the index for *ei_path* lives — beside the file it describes."""
    return Path(str(ei_path) + SUFFIX)


def _identity(ei_path):
    """(size, mtime_ns) for *ei_path*, or None if it cannot be stat'd."""
    try:
        st = os.stat(ei_path)
    except OSError:
        return None
    return st.st_size, st.st_mtime_ns


def load(ei_path):
    """Return ``(cell_id_to_offset, cell_id_to_nspikes)`` or None.

    None means "scan the file" — for a missing cache, a version bump, a .ei
    that has changed since, or anything unreadable.
    """
    path = cache_path(ei_path)
    identity = _identity(ei_path)
    if identity is None:
        return None
    try:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
    except FileNotFoundError:
        return None
    except Exception:
        logger.debug("unreadable EI index at %s; rescanning", path, exc_info=True)
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get(VERSION_KEY) != INDEX_VERSION:
        return None
    if tuple(payload.get("ei_identity") or ()) != identity:
        # The .ei was rewritten. Its offsets mean nothing now.
        logger.debug("EI index at %s is stale; rescanning", path)
        return None

    offsets = payload.get("cell_id_to_offset")
    nspikes = payload.get("cell_id_to_nspikes")
    if not isinstance(offsets, dict) or not isinstance(nspikes, dict):
        return None
    if not offsets:
        return None
    return offsets, nspikes


def save(ei_path, cell_id_to_offset, cell_id_to_nspikes):
    """Write the index beside *ei_path*. Never raises.

    Written to a temporary file in the same directory and renamed, so a crash
    or a second process cannot leave a half-written index behind — the rename
    is atomic and readers therefore see either the old file or the whole new
    one. Failure is fine and silent-ish: the only cost is rescanning next time,
    which is exactly what happens today.
    """
    if not cell_id_to_offset:
        return False
    identity = _identity(ei_path)
    if identity is None:
        return False

    path = cache_path(ei_path)
    payload = {
        VERSION_KEY: INDEX_VERSION,
        "ei_identity": identity,
        "cell_id_to_offset": dict(cell_id_to_offset),
        "cell_id_to_nspikes": dict(cell_id_to_nspikes),
    }
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=str(path.parent), prefix=path.name, suffix=".tmp", delete=False
        ) as tmp:
            tmp_name = tmp.name
            pickle.dump(payload, tmp, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_name, path)
        tmp_name = None
        logger.debug("wrote EI index for %d cells to %s",
                     len(cell_id_to_offset), path)
        return True
    except Exception:
        # A read-only share is an ordinary situation, not an error.
        logger.debug("could not write EI index to %s", path, exc_info=True)
        return False
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
