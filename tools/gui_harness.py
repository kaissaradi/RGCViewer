"""Drive the real Encore MainWindow offscreen against a real dataset.

Use this to verify a GUI fix on real data. The unit suite does not load a
real run, so it cannot show what a user sees.

Usage (from the repo root):

    conda run --no-capture-output -n rgcviewer python tools/gui_harness.py <scenario> [ks_dir]

Environment:

    HARNESS_DAT   Raw .bin directory to attach (enables the FeatureWorker path).
    HARNESS_OUT   Output directory. Default: /tmp/encore_harness.

Screenshots go to $HARNESS_OUT/shots/<scenario>/. Each scenario prints one
JSON line per check. Add a scenario as a function named scenario_<name>(s).

The default dataset (20251212A/data018) has Kilosort, Vision (.ei/.sta/
.params/.neurons) and a DSOS grating file. It sits on the CIFS mount, so a
cold load takes ~15 s. Closing the window writes the normal in-place cache
.pkl files next to the data.
"""
import os
import sys
import time
import json
import logging
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("HARNESS_OUT", "/tmp/encore_harness")
DEFAULT_KS = "/mnt/lab/Array-data/sorted/20251212A/kilosort40/data018/ksfiles"

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, REPO)
os.chdir(REPO)

from qt_bootstrap import prefer_bundled_qt  # noqa: E402

prefer_bundled_qt()
from qtpy.QtWidgets import QApplication  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(relativeCreated)8.0fms %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
for noisy in ("matplotlib", "PIL", "numba", "OpenGL"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

app = QApplication.instance() or QApplication(sys.argv)
from src.gui.main_window import MainWindow  # noqa: E402

T0 = time.time()


def log(msg):
    print(f"[harness {time.time() - T0:7.2f}s] {msg}", flush=True)


def pump(sec):
    end = time.time() + sec
    while time.time() < end:
        app.processEvents()
        time.sleep(0.005)


def wait_until(pred, timeout, step=0.05):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        try:
            if pred():
                return True
        except Exception:
            pass
        time.sleep(step)
    return False


class Session:
    def __init__(self, scenario, ks_dir=DEFAULT_KS, size=(1800, 1000)):
        self.shot_dir = os.path.join(OUT, "shots", scenario)
        os.makedirs(self.shot_dir, exist_ok=True)
        self.w = MainWindow()
        self.w.resize(*size)
        self.w.show()
        pump(0.3)
        self.ks_dir = ks_dir

    def load(self, timeout=600):
        t = time.time()
        self.w.load_directory(self.ks_dir, os.environ.get("HARNESS_DAT") or None)
        ok = wait_until(lambda: getattr(self.w, "_dataset_revealed", False), timeout)
        log(f"dataset revealed={ok} in {time.time() - t:.1f}s; "
            f"clusters={len(self.w.data_manager.cluster_df)} "
            f"vision_stas={bool(self.w.data_manager.vision_stas)}")
        return ok

    def tab(self, name):
        tabs = self.w.analysis_tabs
        for i in range(tabs.count()):
            if tabs.tabText(i) == name:
                tabs.setCurrentIndex(i)
                pump(0.2)
                return
        raise KeyError(name)

    def select(self, cid, settle=1.0):
        ok = self.w._select_cluster_in_tree(int(cid))
        pump(settle)
        return ok

    def shot(self, name, widget=None):
        widget = widget or self.w
        path = os.path.join(self.shot_dir, f"{name}.png")
        widget.grab().save(path)
        return path

    def dm(self):
        return self.w.data_manager


def sta_state(s, cid):
    p = s.w.sta_panel
    img = p._pg_image_item.image
    return {
        "cid": int(cid),
        "panel_cid": p.current_sta_cluster_id,
        "has_img": img is not None,
        "metrics": bool(p._current_metrics),
    }


# ─────────────────────────── scenarios ────────────────────────────

def scenario_sta_refresh(s):
    """Tester report: first select of an uncached STA doesn't draw until a tab switch."""
    s.load()
    dm = s.dm()
    s.tab("STA")
    ids = [int(c) for c in dm.cluster_df["cluster_id"].values]
    with_sta = [c for c in ids if dm.get_vision_id_for_cluster(c) in dm.vision_stas]
    log(f"{len(with_sta)} of {len(ids)} clusters have an STA")
    picks = with_sta[len(with_sta) // 3: len(with_sta) // 3 + 12]
    stale = 0
    for c in picks:
        t = time.time()
        s.select(c, settle=0.0)
        drew = wait_until(lambda: s.w.sta_panel.current_sta_cluster_id == c, 12)
        st = sta_state(s, c)
        st["t"] = round(time.time() - t, 2)
        st["drew"] = drew
        if not drew:
            stale += 1
        log(json.dumps(st))
        s.shot(f"sta_{c}", s.w.sta_panel)
    log(f"STA not redrawn for {stale}/{len(picks)} selections")


def scenario_sta_modes(s):
    """Q2/Q3/Q4: fixed scale, heatmap and space–time views, stale-state clear."""
    import numpy as np
    s.load()
    dm = s.dm()
    p = s.w.sta_panel
    s.tab("STA")
    ids = [int(c) for c in dm.cluster_df["cluster_id"].values]
    with_sta = [c for c in ids if dm.get_vision_id_for_cluster(c) in dm.vision_stas]
    without = [c for c in ids if c not in set(with_sta)]
    picks = with_sta[len(with_sta) // 3: len(with_sta) // 3 + 3]
    for c in picks:
        s.select(c, settle=0.0)
        wait_until(lambda: p.current_sta_cluster_id == c, 12)
        for mode in ("Stimulus", "Heatmap", "Space–time"):
            p.sta_mode_combo.setCurrentText(mode)
            pump(0.3)
            s.shot(f"{c}_{mode.replace('–', '-')}", p)
        p.sta_mode_combo.setCurrentText("Stimulus")
        pump(0.1)
        img = p._pg_image_item.image
        # The noise mean must sit at mid-gray (0.5), not wherever the frame min/max put it.
        log(json.dumps({"cid": c, "img_median": round(float(np.median(img)), 3),
                        "absmax": round(p._absmax_all, 3), "dom": p._dom_idx,
                        "slice": p._slice_rc, "slice_src": p._slice_source}))

    # Stale state: animate a cell, then select a cell with no STA.
    if without:
        s.select(picks[0], settle=0.0)
        wait_until(lambda: p.current_sta_cluster_id == picks[0], 12)
        p._start_animation_timer()
        pump(0.3)
        s.select(without[0], settle=0.5)
        timer = p.sta_animation_timer
        log(json.dumps({"no_sta_cid": without[0],
                        "movie_dropped": p.current_sta_data is None,
                        "timer_stopped": not (timer and timer.isActive()),
                        "img_cleared": p._pg_image_item.image is None}))
        s.shot("no_sta_after_animation", p)


SCENARIOS = {k[len("scenario_"):]: v for k, v in globals().items()
             if k.startswith("scenario_")}

if __name__ == "__main__":
    name = sys.argv[1]
    ks = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_KS
    s = Session(name, ks)
    try:
        SCENARIOS[name](s)
    except Exception:
        traceback.print_exc()
    finally:
        log("closing")
        s.w.close()
        pump(0.5)
        os._exit(0)
