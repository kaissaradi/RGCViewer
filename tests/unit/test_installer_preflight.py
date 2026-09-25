"""install.sh stops with a clear message instead of a bare git error (PLAN.md Q27).

The functions are called from a bash that sources install.sh with
ENCORE_INSTALL_NO_MAIN=1; nothing is cloned or installed.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not (shutil.which("bash") and shutil.which("git")),
                                reason="needs bash and git")


def _call(snippet, **env):
    script = f'set -euo pipefail; ENCORE_INSTALL_NO_MAIN=1 source "{REPO / "install.sh"}"; {snippet}'
    environ = {k: v for k, v in os.environ.items() if k != "ENCORE_BRANCH"}
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                          env={**environ, **env})


def _git(d, *args):
    return subprocess.run(["git", "-C", str(d), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


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
    r = _call(f'resolve_branch "{d}"')
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


# --- update channels (PLAN.md Q35) -------------------------------------------

def _origin_and_install(tmp_path):
    """A bare 'GitHub' with main and beta (beta one commit ahead) and a clone of main."""
    src = _repo(tmp_path)
    _git(src, "checkout", "-q", "-b", "beta")
    (src / "b.txt").write_text("beta")
    _git(src, "add", "b.txt")
    _git(src, "commit", "-q", "-m", "beta work")
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(src), str(origin)], check=True)
    install = tmp_path / "install"
    subprocess.run(["git", "clone", "-q", "--branch", "main", str(origin), str(install)],
                   check=True)
    return src, origin, install


def test_channel_defaults(tmp_path):
    _src, _origin, install = _origin_and_install(tmp_path)
    assert _call(f'resolve_branch "{tmp_path / "none"}"').stdout.strip() == "main"
    assert _call(f'resolve_branch "{install}"').stdout.strip() == "main"
    assert _call(f'resolve_branch "{install}"', ENCORE_BRANCH="beta").stdout.strip() == "beta"


def test_switch_to_beta_update_it_and_switch_back(tmp_path):
    src, origin, install = _origin_and_install(tmp_path)
    (install / "cache.pkl").write_text("x")      # untracked caches survive a switch

    r = _call(f'update_checkout "{install}" beta')
    assert r.returncode == 0, r.stderr
    assert _git(install, "rev-parse", "--abbrev-ref", "HEAD") == "beta"
    assert (install / "b.txt").exists()
    # Without ENCORE_BRANCH the install now stays on beta.
    assert _call(f'resolve_branch "{install}"').stdout.strip() == "beta"

    (src / "c.txt").write_text("more")           # a newer beta is pushed
    _git(src, "add", "c.txt")
    _git(src, "commit", "-q", "-m", "more beta")
    _git(src, "push", "-q", str(origin), "beta")
    assert _call(f'update_checkout "{install}" beta').returncode == 0
    assert (install / "c.txt").exists()

    r = _call(f'update_checkout "{install}" main')
    assert r.returncode == 0, r.stderr
    assert _git(install, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert not (install / "b.txt").exists() and (install / "cache.pkl").exists()


def test_unknown_channel_stops_with_advice(tmp_path):
    _src, _origin, install = _origin_and_install(tmp_path)
    r = _call(f'update_checkout "{install}" nightly')
    assert r.returncode != 0 and "no 'nightly' channel" in r.stderr
    assert _git(install, "rev-parse", "--abbrev-ref", "HEAD") == "main"


# --- the same channel functions in install.ps1 (Windows) ---------------------

PWSH = shutil.which("pwsh")


def _ps(snippet, **env):
    """Define install.ps1's functions (not its body) in pwsh, then run ``snippet``."""
    script = (
        '$e = $null; '
        f'$ast = [System.Management.Automation.Language.Parser]::ParseFile("{REPO / "install.ps1"}", [ref]$null, [ref]$e); '
        '$ast.FindAll({ $args[0] -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $false) '
        '| ForEach-Object { . ([scriptblock]::Create($_.Extent.Text)) }; '
        '$Channels = @("main", "beta"); $Repo = "origin"; '
        f'{snippet}')
    environ = {k: v for k, v in os.environ.items() if k != "ENCORE_BRANCH"}
    return subprocess.run([PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
                          capture_output=True, text=True, timeout=120, env={**environ, **env})


@pytest.mark.skipif(PWSH is None, reason="needs PowerShell (pwsh)")
def test_ps1_channels(tmp_path):
    src, origin, install = _origin_and_install(tmp_path)
    assert _ps(f'Resolve-Branch "{install}"').stdout.strip() == "main"
    assert _ps(f'Resolve-Branch "{install}"', ENCORE_BRANCH="beta").stdout.strip() == "beta"

    r = _ps(f'Update-Checkout "{install}" beta')
    assert r.returncode == 0, r.stdout + r.stderr
    assert _git(install, "rev-parse", "--abbrev-ref", "HEAD") == "beta"
    assert _ps(f'Resolve-Branch "{install}"').stdout.strip() == "beta"

    r = _ps(f'Update-Checkout "{install}" nightly')
    assert r.returncode != 0 and "no 'nightly' channel" in r.stdout

    assert _ps(f'Update-Checkout "{install}" main').returncode == 0
    assert _git(install, "rev-parse", "--abbrev-ref", "HEAD") == "main"


@pytest.mark.skipif(PWSH is None, reason="needs PowerShell (pwsh)")
def test_ps1_stops_on_local_changes_and_hand_moved_branch(tmp_path):
    d = _repo(tmp_path)
    (d / "a.txt").write_text("edited")
    r = _ps(f'Assert-CleanCheckout "{d}"')
    assert r.returncode != 0 and "local changes" in r.stdout
    _git(d, "checkout", "-q", "--", ".")
    _git(d, "checkout", "-q", "-b", "dev-testing")
    r = _ps(f'Resolve-Branch "{d}"')
    assert r.returncode != 0 and "'dev-testing'" in r.stdout
