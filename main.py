import sys
import argparse
import logging
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
    parser = argparse.ArgumentParser(description="RGC Viewer - Retinal Ganglion Cell Analysis Tool")
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
