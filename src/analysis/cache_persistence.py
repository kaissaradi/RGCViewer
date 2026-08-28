"""Pure, Qt-free helpers for physics-cache persistence.

These rules decide *what is allowed to reach disk* and *what is allowed to come
back off it*. They live outside DataManager so they can be unit-tested without
Qt, a real dataset, or any file I/O (see AGENTS.md §9 on fixture selection).

The bug they exist to fix: the ACG warm-up pass writes half-built
``feature_cache`` rows — ``{"acg": ...}`` with no ``_computed`` flag — and those
rows used to be pickled along with the finished ones. On the next launch the
load-time pruner in ``build_cluster_dataframe`` drops every row lacking
``_computed``, so the persisted cache was guaranteed to be thrown away and the
whole physics pass re-ran. Filtering on the way *out* means only rows that
survive the pruner are ever written.
"""

from typing import Any, Dict, Optional, Tuple

# Bump when the shape of a persisted entry changes in a way that makes older
# pickles wrong rather than merely incomplete. Unversioned (legacy) payloads are
# always discarded — they predate the _computed filter and may be full of the
# partial rows described above.
CACHE_VERSION = 1

VERSION_KEY = "__cache_version__"


def filter_computed_entries(feature_cache: Dict[Any, Any]) -> Dict[Any, Any]:
    """Keep only fully-computed physics rows.

    A row is complete when ``get_cell_physics`` finished it and set
    ``_computed``. Partial rows (ACG-only) are dropped so they never reach disk.

    Rows flagged ``_timed_out`` are dropped too, and that flag is the whole
    reason this function is not just a ``_computed`` test. When a cell's STA
    read blows its deadline, ``ensure_physics_cache`` still writes a row —
    ``_computed`` set, physics zeroed — so the progress bar and the UMAP gate
    can move on instead of hanging on one slow cell. That placeholder is fine
    in RAM for one session and *ruinous* on disk: it is indistinguishable from
    a real result, so persisting it bakes ``rf_area=0.0, ellipticity=0.0,
    time_to_peak=0`` into the run's feature_cache.pkl and every later session
    reads those zeros back as truth. Keeping it out of the pickle means a
    timed-out cell is simply recomputed next time.
    """
    return {
        k: v
        for k, v in feature_cache.items()
        if isinstance(v, dict) and v.get("_computed") and not v.get("_timed_out")
    }


def add_version(entries: Dict[Any, Any]) -> Dict[Any, Any]:
    """Return a copy of *entries* stamped with the current cache version."""
    payload = dict(entries)
    payload[VERSION_KEY] = CACHE_VERSION
    return payload


def strip_version(payload: Any) -> Tuple[Dict[Any, Any], Optional[int]]:
    """Split a loaded payload into (entries, version).

    A non-dict payload — or one with no version stamp — yields ``None`` for the
    version, which :func:`is_current_version` rejects.
    """
    if not isinstance(payload, dict):
        return {}, None
    version = payload.get(VERSION_KEY)
    entries = {k: v for k, v in payload.items() if k != VERSION_KEY}
    return entries, version


def is_current_version(version: Optional[int]) -> bool:
    return version == CACHE_VERSION


def load_versioned_entries(payload: Any) -> Dict[Any, Any]:
    """Unwrap a versioned payload, returning ``{}`` for legacy or stale files."""
    entries, version = strip_version(payload)
    if not is_current_version(version):
        return {}
    return entries
