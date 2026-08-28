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


def explain_qt_import_failure(stream=None):
    """Print why the Qt binding failed to load.

    qtpy raises QtBindingsNotFoundError no matter *why* a binding was
    unimportable, which hides the real error. Re-import PyQt6 directly to
    surface it, plus the environment details that usually explain it.
    """
    import os
    import traceback

    out = stream or sys.stderr

    print("\n--- Encore Qt diagnostics ---", file=out)
    print(f"python      : {sys.executable}", file=out)
    print(f"version     : {sys.version.split()[0]}", file=out)

    try:
        import PyQt6
        print(f"PyQt6       : {os.path.dirname(PyQt6.__file__)}", file=out)
        qt_bin = os.path.join(os.path.dirname(PyQt6.__file__), "Qt6", "bin")
        print(f"Qt6/bin     : {qt_bin} (exists={os.path.isdir(qt_bin)})", file=out)
    except Exception:
        print("PyQt6       : not importable", file=out)

    print("\nThe underlying import error qtpy hid:", file=out)
    try:
        import PyQt6.QtCore  # noqa: F401
        print("  (none - PyQt6.QtCore imported fine on retry)", file=out)
    except BaseException:
        traceback.print_exc(file=out)

    # Any Qt6Core.dll or MSVC runtime ahead of ours on PATH is the usual cause.
    if sys.platform == "win32":
        for dll in ("Qt6Core.dll", "msvcp140.dll", "vcruntime140.dll"):
            hits = [
                os.path.join(d, dll)
                for d in os.environ.get("PATH", "").split(os.pathsep)
                if d and os.path.isfile(os.path.join(d, dll))
            ]
            print(f"\n{dll} on PATH:", file=out)
            for h in hits or ["  (none)"]:
                print(f"  {h}", file=out)

    print("\nPlease send this whole block to whoever maintains Encore.", file=out)
    print("--- end diagnostics ---\n", file=out)
