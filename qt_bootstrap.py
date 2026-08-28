"""Ensure PyQt6's bundled Qt DLLs are found before any other Qt on the system.

Import this before importing qtpy or PyQt6.
"""
import sys

_qt_dll_dir = None


def prefer_bundled_qt():
    """Make PyQt6's own Qt DLLs win over any others already on PATH.

    On Windows an Anaconda base environment puts its Library/bin (which ships a
    different Qt6Core.dll) ahead of us in the DLL search order, and PyQt6 then
    fails with "DLL load failed while importing QtCore: The specified procedure
    could not be found." Prepending the wheel's own Qt bin directory fixes it.

    No-op on non-Windows platforms.
    """
    global _qt_dll_dir
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
        _qt_dll_dir = os.add_dll_directory(qt_bin)
