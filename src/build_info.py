"""Which Encore build is running (PLAN.md Q35).

An install is a git checkout (install.sh / install.ps1 use `pip install -e`),
so the channel and commit come from git. Testers on the beta channel and
users on main then report the exact build with a bug.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                          timeout=3, check=True).stdout.strip()


@lru_cache(maxsize=1)
def describe() -> str:
    """'beta @ 3d3263f (2026-09-25)'; '+ local changes' when the tree is dirty.

    Without git (a copied folder, no git on PATH) the package version.
    """
    try:
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        commit, date = _git("log", "-1", "--format=%h %cs").split()
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, ValueError, subprocess.SubprocessError):
        try:
            from importlib.metadata import version
            return f"version {version('encore')}"
        except Exception:
            return "unknown build"
    return f"{branch} @ {commit} ({date})" + (" + local changes" if dirty else "")
