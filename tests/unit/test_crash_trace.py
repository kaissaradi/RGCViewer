"""crash_guard.install_crash_trace: Qt warnings and native crashes reach crash.log."""

import faulthandler
import subprocess
import sys
import textwrap

from qtpy.QtCore import qInstallMessageHandler

from src.gui import crash_guard


def test_qt_warning_is_written(qapp, tmp_path):
    log = tmp_path / "crash.log"
    try:
        crash_guard.install_crash_trace(log)
        from qtpy.QtGui import QPainter
        from qtpy.QtWidgets import QWidget
        w = QWidget()
        p = QPainter()
        p.begin(w)  # outside paintEvent: "Paint device returned engine == 0"
    finally:
        qInstallMessageHandler(None)
        faulthandler.enable()
    text = log.read_text()
    assert "session start" in text
    assert "engine == 0" in text


def test_segfault_leaves_a_stack(tmp_path):
    log = tmp_path / "crash.log"
    code = textwrap.dedent(f"""
        import faulthandler, sys
        sys.path.insert(0, {str(crash_guard.__file__.rsplit('/src/', 1)[0])!r})
        from src.gui import crash_guard
        crash_guard.install_crash_trace({str(log)!r})
        def drag_folder():
            faulthandler._sigsegv()
        drag_folder()
    """)
    subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=120)
    text = log.read_text()
    assert "Segmentation fault" in text
    assert "drag_folder" in text
