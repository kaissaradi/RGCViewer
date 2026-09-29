"""Suite-wide test setup."""

import gc


def pytest_sessionfinish(session, exitstatus):
    """Free every window and pending Qt deletion before Python shuts down.

    Tests close their windows with deleteLater(), and the last ones never see
    an event loop again. PyQt's exit cleanup then walked a wrapper that was
    already freed and segfaulted Python 3.13 after every test had passed
    (CI, empty HOME; gdb: sip cleanup_on_exit → cleanup_qobject, 2026-09-28).
    """
    try:
        from qtpy.QtCore import QCoreApplication, QEvent
        from qtpy.QtWidgets import QApplication
    except Exception:
        return
    app = QApplication.instance()
    if app is None:
        return
    app.closeAllWindows()
    for _ in range(3):
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
        gc.collect()
