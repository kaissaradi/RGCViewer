#!/usr/bin/env bash
set -euo pipefail

REPO="${ENCORE_REPO:-https://github.com/kaissaradi/RGCViewer.git}"
INSTALL_DIR="${ENCORE_HOME:-$HOME/.encore}"
# Update channels. main is what every lab machine runs; beta gets the same
# changes earlier, for testers. ENCORE_BRANCH=beta (or main) picks one.
CHANNELS="main beta"
BIN_DIR="${ENCORE_BIN:-$HOME/.local/bin}"
MIN_PYTHON="3.10"

info()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33mWARN:\033[0m %s\n' "$*"; }
fail()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# --- locate python --------------------------------------------------------
find_python() {
    for cmd in python3 python; do
        if command -v "$cmd" &>/dev/null; then
            local ver
            ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null) || continue
            local major minor
            major=${ver%%.*}
            minor=${ver#*.}
            if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
                echo "$cmd"
                return
            fi
        fi
    done
    return 1
}

# --- preflight (PLAN.md Q27) -------------------------------------------------
# An update is a fast-forward under `set -e`: local edits or another branch
# made it die with a bare git error. Say what is wrong instead.
preflight_checkout() {
    local dir="$1"
    [ -d "$dir/.git" ] || return 0
    if [ -n "$(git -C "$dir" status --porcelain --untracked-files=no)" ]; then
        fail "$dir has local changes, so it cannot be updated.
  Keep them:    git -C \"$dir\" stash
  Drop them:    git -C \"$dir\" checkout -- .
  Then run the installer again."
    fi
}

# The channel to install: ENCORE_BRANCH when set; else the channel the
# install is already on; else main. A checkout moved by hand to any other
# branch stops the update (PLAN.md Q35).
resolve_branch() {
    local dir="$1"
    if [ -n "${ENCORE_BRANCH:-}" ]; then
        echo "$ENCORE_BRANCH"
        return
    fi
    if [ ! -d "$dir/.git" ]; then
        echo main
        return
    fi
    local current
    current=$(git -C "$dir" rev-parse --abbrev-ref HEAD)
    case " $CHANNELS " in
        *" $current "*) echo "$current" ;;
        *) fail "$dir is on branch '$current', not 'main'. Encore updates from 'main' ('beta' for testers).
  Switch back:  git -C \"$dir\" checkout main
  Then run the installer again." ;;
    esac
}

# Put the checkout on origin's $branch: switch when needed, then fast-forward.
update_checkout() {
    local dir="$1" branch="$2"
    git -C "$dir" fetch --quiet origin || fail "git fetch failed in $dir. Is the network reachable?"
    git -C "$dir" rev-parse --verify --quiet "refs/remotes/origin/$branch" >/dev/null ||
        fail "There is no '$branch' channel in $REPO. Use ENCORE_BRANCH=main or ENCORE_BRANCH=beta."
    if [ "$(git -C "$dir" rev-parse --abbrev-ref HEAD)" != "$branch" ]; then
        info "Switching $dir to the '$branch' channel"
        if git -C "$dir" show-ref --verify --quiet "refs/heads/$branch"; then
            git -C "$dir" checkout --quiet "$branch"
        else
            git -C "$dir" checkout --quiet -b "$branch" --track "origin/$branch"
        fi
    fi
    git -C "$dir" merge --ff-only --quiet "origin/$branch" ||
        fail "$dir cannot be fast-forwarded to origin/$branch.
  Start this channel fresh:  git -C \"$dir\" reset --hard origin/$branch
  Then run the installer again."
}

# A venv whose Python was removed or upgraded away cannot run pip.
venv_is_usable() {
    local venv="$1"
    [ -x "$venv/bin/python" ] &&
        "$venv/bin/python" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null
}

main() {
PYTHON=$(find_python) || fail "Python >= $MIN_PYTHON is required but not found. Install it first."
info "Using $($PYTHON --version) at $(command -v "$PYTHON")"

# --- clone or update -------------------------------------------------------
preflight_checkout "$INSTALL_DIR"
BRANCH=$(resolve_branch "$INSTALL_DIR")
if [ -d "$INSTALL_DIR/.git" ]; then
    info "Updating existing installation in $INSTALL_DIR (channel: $BRANCH)"
    update_checkout "$INSTALL_DIR" "$BRANCH"
else
    info "Cloning Encore into $INSTALL_DIR (channel: $BRANCH)"
    git clone --branch "$BRANCH" "$REPO" "$INSTALL_DIR"
fi

# --- virtual environment ---------------------------------------------------
VENV="$INSTALL_DIR/.venv"
if [ -d "$VENV" ] && ! venv_is_usable "$VENV"; then
    warn "The virtual environment in $VENV cannot run (its Python is missing or too old). Rebuilding it."
    [ -n "$INSTALL_DIR" ] && rm -rf "$VENV"
fi
if [ ! -d "$VENV" ]; then
    info "Creating virtual environment"
    "$PYTHON" -m venv "$VENV"
fi

info "Installing dependencies (this may take a few minutes on first run)"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -e "$INSTALL_DIR"

# --- create launcher shim --------------------------------------------------
mkdir -p "$BIN_DIR"
SHIM="$BIN_DIR/encore"

cat > "$SHIM" << 'LAUNCHER'
#!/usr/bin/env bash
INSTALL_DIR="${ENCORE_HOME:-$HOME/.encore}"
exec "$INSTALL_DIR/.venv/bin/encore" "$@"
LAUNCHER
chmod +x "$SHIM"

# --- PATH advice -----------------------------------------------------------
if ! echo "$PATH" | tr ':' '\n' | grep -qx "$BIN_DIR"; then
    warn "$BIN_DIR is not in your PATH."
    echo ""
    echo "  Add it by appending one of these to your shell config:"
    echo ""
    echo "    # bash (~/.bashrc)"
    echo "    export PATH=\"$BIN_DIR:\$PATH\""
    echo ""
    echo "    # zsh (~/.zshrc)"
    echo "    export PATH=\"$BIN_DIR:\$PATH\""
    echo ""
    echo "    # fish (~/.config/fish/config.fish)"
    echo "    fish_add_path $BIN_DIR"
    echo ""
fi

info "Encore installed! Run it with:"
echo ""
echo "    encore"
echo ""
echo "  Options:"
echo "    encore --debug"
echo "    encore --kilosort-dir /path/to/run"
echo "    encore --dat-file /path/to/raw.dat"
echo ""
echo "  Channel: $BRANCH. Run the installer again to update this channel."
echo ""
}

# Tests source this file with ENCORE_INSTALL_NO_MAIN=1 to call the functions.
# (BASH_SOURCE cannot be used: `curl ... | bash` reads the script from stdin.)
if [ "${ENCORE_INSTALL_NO_MAIN:-0}" != "1" ]; then
    main "$@"
fi
