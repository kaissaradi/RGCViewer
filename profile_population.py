"""
Drop this into draw_population_timecourse_panel and draw_population_acg_panel
temporarily to find where the time is actually going.

Usage: add   `from .profile_population import time_section` at the top of
population_panel.py, then wrap sections as shown below.

OR just run the standalone benchmark at the bottom against your real DataManager.
"""

import time
import numpy as np
from contextlib import contextmanager

_timings = {}

@contextmanager
def time_section(label):
    t0 = time.perf_counter()
    yield
    dt = (time.perf_counter() - t0) * 1000
    _timings.setdefault(label, []).append(dt)
    print(f"  [{label}]: {dt:.1f} ms")


def report():
    print("\n=== Population Panel Timing Report ===")
    for label, samples in _timings.items():
        print(f"  {label}: mean={np.mean(samples):.1f}ms  max={np.max(samples):.1f}ms  n={len(samples)}")


# ---------------------------------------------------------------------------
# Standalone benchmark — run this directly:
#   python profile_population.py
# Point it at a real DataManager instance or mock one.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    N = 900  # cells per folder — worst case

    # --- Simulate the data that would come out of the cache ---
    T = 30       # timecourse frames
    ACG = 200    # acg bins

    arr_tc  = np.random.randn(N, T).astype(np.float32)
    arr_acg = np.random.randn(N, ACG).astype(np.float32)
    t_tc    = np.arange(T)
    t_acg   = np.linspace(-100, 100, ACG)

    print(f"Benchmarking with N={N} cells, T={T} timecourse frames, {ACG} ACG bins")
    print("=" * 60)

    REPS = 20

    # 1. segments construction — the fix we applied
    with time_section(f"segments construction x{REPS} (timecourse)"):
        for _ in range(REPS):
            segments = [np.column_stack([t_tc, row]) for row in arr_tc]

    with time_section(f"segments construction x{REPS} (ACG)"):
        for _ in range(REPS):
            segments = [np.column_stack([t_acg, row]) for row in arr_acg]

    # 2. set_segments vs full LineCollection rebuild
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    segs_tc  = [np.column_stack([t_tc,  row]) for row in arr_tc]
    segs_acg = [np.column_stack([t_acg, row]) for row in arr_acg]

    fig, ax = plt.subplots()
    lc = LineCollection(segs_tc, linewidth=0.8, alpha=0.15)
    ax.add_collection(lc)

    with time_section(f"LineCollection.set_segments x{REPS}"):
        for _ in range(REPS):
            lc.set_segments(segs_tc)

    with time_section(f"Full LineCollection rebuild x{REPS}"):
        for _ in range(REPS):
            lc2 = LineCollection(segs_tc, linewidth=0.8, alpha=0.15)

    # 3. fig.clear() + add_subplot vs patch removal
    with time_section(f"fig.clear() + add_subplot x{REPS}"):
        for _ in range(REPS):
            fig2, ax2 = plt.subplots()
            fig2.clear()
            ax2 = fig2.add_subplot(111)
            plt.close(fig2)

    # 4. add_patch loop (what RF mosaic does on cache replay)
    from matplotlib.patches import Ellipse
    fig3, ax3 = plt.subplots()
    with time_section(f"add_patch x{N} ellipses (single run)"):
        for i in range(N):
            e = Ellipse(xy=(i, i), width=2, height=1, angle=0)
            ax3.add_patch(e)

    with time_section(f"patch.remove() x{N} (single run)"):
        for p in list(ax3.patches):
            p.remove()

    # 5. canvas.draw_idle() — this is often the hidden killer
    with time_section(f"fig.canvas.draw() x{REPS} (Agg backend)"):
        for _ in range(REPS):
            fig.canvas.draw()

    print()
    report()
    print()
    print("KEY QUESTION: if segments construction and set_segments are fast")
    print("but you're still seeing freezes, the bottleneck is draw_idle() itself")
    print("— i.e. matplotlib re-rasterizing N=900 line segments every frame.")
    print("That requires a different fix (blitting or pyqtgraph).")
