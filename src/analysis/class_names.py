"""One name per RGC type across the lab's hand-typed Vision classes (PLAN.md Q44).

Class IDs in the lab's .params files are typed by hand. A scan of all 413
files (2026-09-25) found 465 distinct strings for a handful of named types:
case (``OFF/BRISK TRANSIENT``), hyphens (``on/brisk-sustained``),
sub-clusters (``off/brisk sustained/nc21/nc22``), question marks
(``off/transient?``) and abbreviations (``ON/maybe b-sus``).

``canonical_type`` maps one classID to (type, uncertain), or None when it
names no type (unclassified, numbered clusters only, junk bins such as
``weak`` or ``huge``, or an ambiguous label).
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# Checked in this order: "brisk transient" before "transient".
_TYPES = (
    ("brisk sustained", re.compile(r"\bbrisk ?sus(tained)?\b|\bb ?sus(tained)?\b|\bbs\b")),
    ("brisk transient", re.compile(r"\bbrisk ?trans(ient)?\b|\bb ?trans(ient)?\b|\bbt\b")),
    ("small transient", re.compile(r"\bsmall ?trans(ient)?\b")),
    ("transient", re.compile(r"\btrans(ient)?\b")),
)
_UNSURE = re.compile(r"\?|\bmaybe\b|\bunclassified\b")
# Labels that say the cell is NOT the type, or cannot be read either way.
_REJECT = re.compile(r"\bremoved\b|\bnot\b|\bcenter\b|\bcentre\b")


def _clean(part: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[-_]", " ", part.strip().lower())).strip()


def canonical_type(class_id: str) -> Optional[Tuple[str, bool]]:
    """('ON brisk sustained', uncertain) for a Vision classID, or None."""
    parts = [_clean(p) for p in (class_id or "").strip("/").split("/")]
    if parts and parts[0] == "all":
        parts = parts[1:]
    if not parts:
        return None
    polarity = parts[0].rstrip("?").strip()
    if polarity not in ("on", "off"):
        return None
    rest = " ".join(parts[1:])
    if not rest or _REJECT.search(rest):
        return None
    for name, pattern in _TYPES:
        if pattern.search(rest):
            uncertain = bool(_UNSURE.search(rest) or parts[0].endswith("?"))
            return f"{polarity.upper()} {name}", uncertain
    return None
