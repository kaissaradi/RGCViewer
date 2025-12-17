import sys
import os
import argparse
import logging
from qtpy.QtCore import QCoreApplication, Qt
from qtpy.QtGui import QSurfaceFormat
from qtpy.QtWidgets import QApplication
from gui.main_window import MainWindow

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="axolotl - RGC Viewer")
    parser.add_argument('--debug', action='store_true', help="Enable debug logging to console")
    args = parser.parse_args()

    setup_logging(args.debug)

    # Try to avoid driver-related GL context segfaults by preferring
    # software OpenGL or a conservative default surface format.
    # This must be set before creating the QApplication.
    try:
        # Prefer Qt's software OpenGL backend to avoid buggy drivers on some Linux systems
        QCoreApplication.setAttribute(Qt.AA_UseSoftwareOpenGL)
    except Exception:
        pass

    # Set a conservative default surface format (OpenGL 2.1, no profile)
    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.NoProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    # Set your default paths here for testing
    # DEFAULT_KILOSORT_DIR = "/Volumes/Vyom MEA/analysis/20250306C/data026/kilosort2.5"
    # DEFAULT_KILOSORT_DIR = "/Volumes/Vyom MEA/analysis/20250306C/chunk4/kilosort2.5"
    # DEFAULT_KILOSORT_DIR = "/Users/riekelabbackup/Desktop/Vyom/data/analysis/20250306C/data026/kilosort2.5"
    DEFAULT_KILOSORT_DIR = "/Volumes/Vyom MEA/analysis/20250917C/data007/kilosort2.5"

    # DEFAULT_DAT_FILE = "/Volumes/Vyom MEA/data/raw/20250306C/data026.bin"
    DEFAULT_DAT_FILE = None
    app = QApplication(sys.argv)
    window = MainWindow(DEFAULT_KILOSORT_DIR, DEFAULT_DAT_FILE)
    window.show()
    sys.exit(app.exec())
