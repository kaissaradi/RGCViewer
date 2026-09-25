"""The running build is named in About and in errors.log (PLAN.md Q35)."""

import re
import shutil
import subprocess

import pytest

from src import build_info
from src.gui import crash_guard


@pytest.fixture(autouse=True)
def _fresh():
    build_info.describe.cache_clear()
    yield
    build_info.describe.cache_clear()


@pytest.mark.skipif(not shutil.which("git") or not (build_info.ROOT / ".git").exists(),
                    reason="needs the git checkout")
def test_names_branch_and_commit():
    branch = subprocess.run(["git", "-C", str(build_info.ROOT), "rev-parse", "--abbrev-ref",
                             "HEAD"], capture_output=True, text=True).stdout.strip()
    assert re.match(rf"{re.escape(branch)} @ [0-9a-f]{{7,}} \(\d{{4}}-\d\d-\d\d\)",
                    build_info.describe())


def test_without_git_falls_back_to_the_version(monkeypatch):
    def no_git(*_a, **_k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(build_info.subprocess, "run", no_git)
    assert build_info.describe().startswith(("version ", "unknown build"))


def test_error_log_entry_names_the_build(tmp_path, monkeypatch):
    monkeypatch.setattr(build_info, "describe", lambda: "beta @ abc1234 (2026-09-25)")
    log = tmp_path / "errors.log"
    crash_guard.make_hook(log)(ValueError, ValueError("boom"), None)
    assert "beta @ abc1234" in log.read_text() and "boom" in log.read_text()
