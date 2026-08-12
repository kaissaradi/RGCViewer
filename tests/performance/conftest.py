"""Guards for the performance suite's optional dependencies.

These tests need plugins that live in requirements-dev.txt, not
requirements.txt. Without the guard a plain `pytest tests/` reports them as
setup ERRORs, which reads as breakage rather than "the tool is not installed"
and buries the real results of the rest of the tree.
"""

import pytest

_HAS_BENCHMARK = True
try:  # pragma: no cover - trivial availability probe
    import pytest_benchmark  # noqa: F401
except ImportError:
    _HAS_BENCHMARK = False


def pytest_collection_modifyitems(items):
    """Skip anything that asks for the `benchmark` fixture when it is absent."""
    if _HAS_BENCHMARK:
        return

    skip = pytest.mark.skip(reason="pytest-benchmark is not installed")
    for item in items:
        if "benchmark" in getattr(item, "fixturenames", ()):
            item.add_marker(skip)
