"""Named sets of the six Feature Extraction axis pairs (PLAN.md Q15).

Stored in Encore's QSettings, so they survive a restart. The preset picked
last is remembered and applied when the window opens. "Default" is always
there and means feature_catalog.DEFAULT_PANELS (four fixed pairs and two
random ones).
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

DEFAULT_NAME = "Default"
_KEY_PRESETS = "feature_extraction/presets"      # JSON {name: [[x, y], ...]}
_KEY_CURRENT = "feature_extraction/current_preset"


def _settings(settings=None):
    if settings is not None:
        return settings
    from .. import recent_paths
    return recent_paths._settings()


def load_presets(settings=None) -> dict:
    """{name: [(x, y), ...]} of the saved presets (without "Default")."""
    raw = _settings(settings).value(_KEY_PRESETS, "")
    try:
        data = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        logger.warning("feature presets in the settings are not valid JSON; ignored")
        return {}
    out = {}
    for name, pairs in (data.items() if isinstance(data, dict) else []):
        try:
            out[str(name)] = [(str(x), str(y)) for x, y in pairs]
        except (TypeError, ValueError):
            continue
    return out


def save_preset(name, panels, settings=None) -> None:
    name = str(name).strip()
    if not name or name == DEFAULT_NAME:
        raise ValueError(f"a preset needs a name other than {DEFAULT_NAME!r}")
    s = _settings(settings)
    presets = load_presets(s)
    presets[name] = [(str(x), str(y)) for x, y in panels]
    s.setValue(_KEY_PRESETS, json.dumps({k: [list(p) for p in v] for k, v in presets.items()}))
    set_current(name, s)


def delete_preset(name, settings=None) -> None:
    s = _settings(settings)
    presets = load_presets(s)
    presets.pop(name, None)
    s.setValue(_KEY_PRESETS, json.dumps({k: [list(p) for p in v] for k, v in presets.items()}))
    if current(s) == name:
        set_current(DEFAULT_NAME, s)


def current(settings=None) -> str:
    """The preset picked last; "Default" if it no longer exists."""
    s = _settings(settings)
    name = s.value(_KEY_CURRENT, DEFAULT_NAME) or DEFAULT_NAME
    return name if name == DEFAULT_NAME or name in load_presets(s) else DEFAULT_NAME


def set_current(name, settings=None) -> None:
    _settings(settings).setValue(_KEY_CURRENT, str(name))


def panels_for(name, settings=None):
    """The pairs of preset ``name``; None for "Default" (use DEFAULT_PANELS)."""
    if name == DEFAULT_NAME:
        return None
    return load_presets(settings).get(name)
