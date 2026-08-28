import sys
import argparse
import logging


_qt_dll_dir = None


def _prefer_bundled_qt():
    """Make PyQt6's own Qt DLLs win over any others already on PATH.

    On Windows an Anaconda base environment puts its Library/bin (which ships a
    different Qt6Core.dll) ahead of us in the DLL search order, and PyQt6 then
    fails with "DLL load failed while importing QtCore: The specified procedure
    could not be found." Prepending the wheel's own Qt bin directory fixes it.
    """
    if sys.platform != "win32":
        return
    import os
    import importlib.util

    spec = importlib.util.find_spec("PyQt6")
    if spec is None or not spec.submodule_search_locations:
        return
    qt_bin = os.path.join(spec.submodule_search_locations[0], "Qt6", "bin")
    if os.path.isdir(qt_bin):
        os.environ["PATH"] = qt_bin + os.pathsep + os.environ.get("PATH", "")
        # Keep the handle alive; closing it would undo the search-path entry.
        global _qt_dll_dir
        _qt_dll_dir = os.add_dll_directory(qt_bin)


_prefer_bundled_qt()

from qtpy.QtCore import QCoreApplication, Qt
from qtpy.QtGui import QSurfaceFormat
from qtpy.QtWidgets import QApplication
from src.gui.main_window import MainWindow

def setup_logging(debug_mode):
    # If not in debug mode, set level to WARNING to hide INFO and DEBUG statements
    log_level = logging.DEBUG if debug_mode else logging.WARNING

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

    # Optional: Silencing specific heavy libraries
    if not debug_mode:
        logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
        logging.getLogger('OpenGL').setLevel(logging.CRITICAL)

def main():
    parser = argparse.ArgumentParser(description="Encore - Retinal Ganglion Cell Analysis Tool")
    parser.add_argument('--debug', action='store_true', help="Enable debug logging to console")
    parser.add_argument('--kilosort-dir', default=None,
                        help="Open this Kilosort directory instead of the last one used")
    parser.add_argument('--dat-file', default=None,
                        help="Raw .bin/.dat file to enable the Raw trace tab")
    args = parser.parse_args()

    setup_logging(args.debug)

    try:
        QCoreApplication.setAttribute(Qt.AA_UseSoftwareOpenGL)
    except Exception:
        pass

    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.NoProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    window = MainWindow(args.kilosort_dir, args.dat_file)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
