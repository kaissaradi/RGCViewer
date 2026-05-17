#!/usr/bin/env python3
"""Capture UMAP and sidebar screenshots for manual GUI polish verification."""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication
from qtpy.QtCore import QTimer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.gui.main_window import MainWindow  # noqa: E402

DEFAULT_DATA = os.path.expanduser(
    "~/Documents/Development/data/sorted/20251015A/chunk20/kilosort4"
)
OUT_DIR = REPO_ROOT / "tests" / "visual_snapshots" / "gui_polish"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1600, 1000)
    win.show()

    def after_show():
        if os.path.isdir(DEFAULT_DATA):
            win.load_directory(DEFAULT_DATA)
        win.analysis_tabs.setCurrentIndex(3)
        QTimer.singleShot(150, save_umap_first_visit)

    def save_umap_first_visit():
        path = OUT_DIR / "umap_first_visit.png"
        win.grab().save(str(path))
        print(f"Saved {path}")
        win.sidebar_toggle_btn.click()
        QTimer.singleShot(200, save_sidebar_collapsed)

    def save_sidebar_collapsed():
        path = OUT_DIR / "sidebar_collapsed.png"
        win.grab().save(str(path))
        print(f"Saved {path}")
        win.sidebar_toggle_btn.click()
        QTimer.singleShot(200, save_sidebar_expanded)

    def save_sidebar_expanded():
        path = OUT_DIR / "sidebar_expanded.png"
        win.grab().save(str(path))
        print(f"Saved {path}")
        app.quit()

    QTimer.singleShot(0, after_show)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
