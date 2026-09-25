"""Keep one bad slot from killing the whole application.

PyQt5.5+ and PyQt6 call qFatal() — SIGABRT, exit code 134 — when a Python
exception escapes a slot while sys.excepthook is the default. The window
then vanishes with no traceback and no chance to save a classification.
Checked on the lab workstation: an AttributeError in a connected slot
aborted the process.

install() replaces the hook. An unhandled exception is logged, appended to
ERROR_LOG, and reported in the main window's status bar; the event loop
keeps running.
"""

import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("encore.crash_guard")

ERROR_LOG = Path.home() / ".encore" / "logs" / "errors.log"


def _write_log(text, path):
    try:
        from src.build_info import describe
        build = describe()
    except Exception:
        build = "unknown build"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n=== {datetime.now().isoformat(timespec='seconds')} · {build} ===\n{text}")
    except OSError:
        pass


def _tell_the_user(exc):
    try:
        from qtpy.QtWidgets import QApplication
    except Exception:
        return
    app = QApplication.instance()
    if app is None:
        return
    for widget in app.topLevelWidgets():
        bar = getattr(widget, "status_bar", None)
        if bar is not None and hasattr(bar, "showMessage"):
            bar.showMessage(
                f"Internal error: {type(exc).__name__}: {exc} "
                f"(details in {ERROR_LOG})", 15000)
            return


def make_hook(log_path=None):
    path = Path(log_path) if log_path is not None else ERROR_LOG

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        logger.critical("Unhandled exception (kept running):\n%s", text)
        _write_log(text, path)
        _tell_the_user(exc)

    return hook


def install(log_path=None):
    """Replace sys.excepthook. Call once, after QApplication exists."""
    sys.excepthook = make_hook(log_path)
