#!/usr/bin/env bash
set -euo pipefail

REPO="https://github.com/kaissaradi/RGCViewer.git"
INSTALL_DIR="${RGCVIEWER_HOME:-$HOME/.rgcviewer}"
BIN_DIR="${RGCVIEWER_BIN:-$HOME/.local/bin}"
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

PYTHON=$(find_python) || fail "Python >= $MIN_PYTHON is required but not found. Install it first."
info "Using $($PYTHON --version) at $(command -v "$PYTHON")"

# --- clone or update -------------------------------------------------------
if [ -d "$INSTALL_DIR/.git" ]; then
    info "Updating existing installation in $INSTALL_DIR"
    git -C "$INSTALL_DIR" pull --ff-only
else
    info "Cloning RGCViewer into $INSTALL_DIR"
    git clone "$REPO" "$INSTALL_DIR"
fi

# --- virtual environment ---------------------------------------------------
VENV="$INSTALL_DIR/.venv"
if [ ! -d "$VENV" ]; then
    info "Creating virtual environment"
    "$PYTHON" -m venv "$VENV"
fi

info "Installing dependencies (this may take a few minutes on first run)"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -e "$INSTALL_DIR"

# --- create launcher shim --------------------------------------------------
mkdir -p "$BIN_DIR"
SHIM="$BIN_DIR/rgcviewer"

cat > "$SHIM" << 'LAUNCHER'
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0" 2>/dev/null || realpath "$0" 2>/dev/null || echo "$0")")" && pwd)"
INSTALL_DIR="${RGCVIEWER_HOME:-$HOME/.rgcviewer}"
exec "$INSTALL_DIR/.venv/bin/rgcviewer" "$@"
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

info "RGCViewer installed! Run it with:"
echo ""
echo "    rgcviewer"
echo ""
echo "  Options:"
echo "    rgcviewer --debug"
echo "    rgcviewer --kilosort-dir /path/to/run"
echo "    rgcviewer --dat-file /path/to/raw.dat"
echo ""
