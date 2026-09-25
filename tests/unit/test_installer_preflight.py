"""install.sh stops with a clear message instead of a bare git error (PLAN.md Q27).

The functions are called from a bash that sources install.sh with
ENCORE_INSTALL_NO_MAIN=1; nothing is cloned or installed.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not (shutil.which("bash") and shutil.which("git")),
                                reason="needs bash and git")


def _call(snippet):
    script = f'set -euo pipefail; ENCORE_INSTALL_NO_MAIN=1 source "{REPO / "install.sh"}"; {snippet}'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)


def _repo(tmp_path, branch="main"):
    d = tmp_path / "enc"
    d.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(d), *a], check=True,  # noqa: E731
                                    capture_output=True)
    run("init", "-q", "-b", "main")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    (d / "a.txt").write_text("1")
    run("add", "a.txt")
    run("commit", "-q", "-m", "c")
    if branch != "main":
        run("checkout", "-q", "-b", branch)
    return d


def test_clean_main_passes(tmp_path):
    d = _repo(tmp_path)
    (d / "untracked.pkl").write_text("x")          # untracked files do not block
    assert _call(f'preflight_checkout "{d}"').returncode == 0


def test_local_changes_stop_the_update_with_advice(tmp_path):
    d = _repo(tmp_path)
    (d / "a.txt").write_text("edited")
    r = _call(f'preflight_checkout "{d}"')
    assert r.returncode != 0 and "local changes" in r.stderr and "stash" in r.stderr


def test_another_branch_stops_the_update_with_advice(tmp_path):
    d = _repo(tmp_path, branch="dev-testing")
    r = _call(f'preflight_checkout "{d}"')
    assert r.returncode != 0 and "'dev-testing'" in r.stderr and "checkout main" in r.stderr


def test_broken_venv_is_detected(tmp_path):
    fake = tmp_path / "venv"
    (fake / "bin").mkdir(parents=True)
    (fake / "bin" / "python").symlink_to(tmp_path / "gone")
    assert _call(f'venv_is_usable "{fake}"').returncode != 0


def test_working_venv_is_usable(tmp_path):
    venv = tmp_path / "ok"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    assert _call(f'venv_is_usable "{venv}"').returncode == 0
