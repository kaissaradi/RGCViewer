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
.pkl files next to the data. Its .sta/.params come from another sort
(PLAN.md Q32): good for mechanics, not for what an STA shows. A matched run:
/mnt/lab/Array-data/sorted/20260220A/kilosort25/data022/ksfiles (~70 s).

Scenario params_save works on a copy of the .params in $HARNESS_OUT and
never writes the lab file.
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
from qtpy.QtCore import Qt  # noqa: E402
import numpy as np  # noqa: E402

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


def scenario_grating(s):
    """Q6/Q8/Q10: batch timing, rasters, error bars, units, condition picker."""
    from src.analysis import grating_calc
    t_load = time.time()
    s.load()
    dm = s.dm()
    ids = [int(c) for c in dm.cluster_df["cluster_id"].values]
    log(f"grating_status={dm.grating_status}")

    def n_current():
        cache = dm.grating_computed_cache or {}
        return sum(1 for c in ids if c in cache
                   and not grating_calc.grating_entry_needs_recompute(cache[c]))
    n_raw = len((dm.grating_raw_data or {}).get("spike_times_by_trial", {}))
    wait_until(lambda: n_current() >= min(n_raw, len(ids)) or
               getattr(s.w, "_grating_batch_thread", None) is None, 300)
    log(f"grating batch: {n_current()} current rows, "
        f"{time.time() - t_load:.1f}s after load start; "
        f"physics done {getattr(dm, '_physics_done_count', 0)}/{len(ids)}")

    by_cls = {}
    for c in ids:
        e = dm.get_grating_data_for_cluster(c)
        if not e:
            continue
        sel = grating_calc.select_best_dsos_condition(e)
        if sel:
            by_cls.setdefault(sel["classification"], []).append(c)
    log(json.dumps({k: len(v) for k, v in by_cls.items()}))

    s.tab("Grating")
    p = s.w.grating_panel
    for cls in ("DS", "OS", "none"):
        if not by_cls.get(cls):
            continue
        c = by_cls[cls][0]
        s.select(c, settle=1.0)
        n_rasters = sum(1 for plot, _r, _t in p.raster_view._pool if plot.isVisible())
        log(json.dumps({"cls": cls, "cid": c, "stats": p.stats_label.text()[:120],
                        "rasters": n_rasters,
                        "combo": [p.condition_combo.itemText(i)
                                  for i in range(p.condition_combo.count())]}))
        s.shot(f"{cls}_{c}", p)
    # Pick the non-best condition by hand, and check it sticks to the next cell.
    if p.condition_combo.count() > 2:
        p.condition_combo.setCurrentIndex(2)
        p.condition_combo.activated.emit(2)
        pump(0.5)
        s.shot("manual_condition", p)
        log("manual pick -> " + p.stats_label.text()[:120])


def scenario_grating_table(s):
    """Q7: DS/OS, DSI, OSI columns fill after the batch and follow the slider."""
    from collections import Counter
    s.load()
    dm = s.dm()
    wait_until(lambda: getattr(s.w, "_grating_batch_thread", None) is None, 300)
    pump(1.0)
    df = dm.cluster_df
    log(json.dumps({"has_cols": all(c in df.columns for c in ("dsos", "dsi", "osi")),
                    "calls": Counter(df["dsos"]) if "dsos" in df else None,
                    "dsi_filled": int(np.isfinite(df["dsi"]).sum()) if "dsi" in df else 0}))
    model = s.w.table_view.model()
    headers = [model.headerData(i, Qt.Horizontal, Qt.DisplayRole)
               for i in range(model.columnCount())]
    log("headers: " + ", ".join(str(h) for h in headers))
    s.w.pop_dsos_threshold_slider.setValue(60)
    pump(1.0)
    log(json.dumps({"calls_at_0.60": Counter(dm.cluster_df["dsos"])}))
    s.w._switch_left_view(1)
    pump(0.5)
    s.shot("table", s.w)


def scenario_attach_vision(s):
    """Q11: File > Load Vision on an open Kilosort run (the reported freeze)."""
    import threading
    from src.gui import callbacks
    s.load()
    wait_until(lambda: s.w.central_widget.isEnabled(), 30)
    vision_dir = os.environ.get("HARNESS_VISION") or os.path.dirname(s.ks_dir)
    orig = callbacks.QFileDialog.getExistingDirectory
    callbacks.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: vision_dir)
    try:
        t = time.time()
        s.w.load_vision_directory()
        done = wait_until(lambda: getattr(s.w, "vision_load_thread", None) is None, 120)
        pump(2.0)
    finally:
        callbacks.QFileDialog.getExistingDirectory = orig
    warm = [th.name for th in threading.enumerate() if th.name == "physics-warm"]
    dm = s.dm()
    log(json.dumps({"vision_load_done": done, "secs": round(time.time() - t, 1),
                    "central_enabled": s.w.central_widget.isEnabled(),
                    "physics_warm_threads": len(warm),
                    "vision_source": getattr(dm, "_vision_source", None)}))
    # Physics must be rebuilt from the attached files, not reused.
    wait_until(lambda: not any(th.name == "physics-warm" and th.is_alive()
                               for th in threading.enumerate()), 300)
    srcs = [e.get("_vision_source") for e in dm.feature_cache.values()
            if isinstance(e, dict) and e.get("_computed")]
    from collections import Counter
    log(json.dumps({"physics_entries_by_source": Counter(srcs)}))
    s.shot("after_attach", s.w)


def scenario_selection_sync(s):
    """Q21: tree -> table sync, table-view select (mosaic click path)."""
    s.load()
    w = s.w
    ids = [int(c) for c in s.dm().cluster_df["cluster_id"].values]
    a, b = ids[10], ids[20]
    s.select(a, settle=0.5)
    log(json.dumps({"tree_selected": a, "table_follows": w._selected_table_cluster_id()}))
    w._switch_left_view(1)                       # table view active
    ok = w.focus_cluster(b)                      # what a mosaic RF click calls
    pump(0.5)
    log(json.dumps({"focus_cluster_table_view": ok,
                    "selected_now": w._get_selected_cluster_id(), "wanted": b}))


def scenario_feature_defaults(s):
    """Q14: Feature Extraction opens on the user's four pairs plus two random ones."""
    from src.gui import callbacks
    s.load()
    ids = [int(c) for c in s.dm().cluster_df["cluster_id"].values]
    shown = {}
    def keep_open(dlg):          # exec() would block the harness
        shown["dlg"] = dlg
        return 0

    orig = callbacks.FeatureExtractionWindow.exec
    callbacks.FeatureExtractionWindow.exec = keep_open
    try:
        callbacks.feature_extraction(s.w, ids)
    finally:
        callbacks.FeatureExtractionWindow.exec = orig
    dlg = shown["dlg"]
    dlg.show()
    ok = wait_until(lambda: bool(dlg.catalog) and bool(dlg._panels), 300)
    pump(1.0)
    log(json.dumps({"catalog_ready": ok, "n_features": len(dlg.catalog),
                    "panels": dlg._panels,
                    "combos": [(cx.currentText(), cy.currentText()) for cx, cy in dlg.axis_combos]}))
    s.shot("feature_window", dlg)
    dlg.close()


def scenario_tab_overlay(s):
    """Q29: switch every tab while changing cells; find widgets from other tabs."""
    from qtpy.QtWidgets import QWidget
    s.load()
    w, tabs = s.w, s.w.analysis_tabs
    ids = [int(c) for c in s.dm().cluster_df["cluster_id"].values]
    pages = [tabs.widget(j) for j in range(tabs.count())]
    found = []
    for rep in range(3):
        for i in range(tabs.count()):
            tabs.setCurrentIndex(i)
            s.select(ids[(rep * 11 + i * 7) % len(ids)], settle=0.6)
            for wid in w.findChildren(QWidget):
                if not wid.isVisible():
                    continue
                for j, page in enumerate(pages):
                    if j != i and (wid is page or page.isAncestorOf(wid)):
                        found.append({"tab": tabs.tabText(i), "from": tabs.tabText(j),
                                      "widget": type(wid).__name__})
            tops = [type(t).__name__ for t in QApplication.topLevelWidgets()
                    if t.isVisible() and t is not w and t.isWindow()
                    and type(t).__name__ not in ("QMenu", "QToolTip")]
            if tops:
                found.append({"tab": tabs.tabText(i), "top_level": tops})
            if rep == 0:
                s.shot(f"{i}_{tabs.tabText(i)}", w)
    log(json.dumps({"switches": 3 * tabs.count(), "problems": found[:20],
                    "n_problems": len(found)}))


def scenario_reload_indicator(s):
    """Q19: the load / cache progress bar on a reload and a dataset switch."""
    w = s.w
    bar = w.cache_progress
    second = os.environ.get("HARNESS_KS2") or s.ks_dir

    def sample(tag, secs):
        out, last = [], None
        end = time.time() + secs
        t0 = time.time()
        while time.time() < end:
            app.processEvents()
            st = (bar.isVisible(), bar.minimum(), bar.maximum(), bar.value(),
                  bar.format(), w.status_bar.currentMessage()[:60],
                  bool(getattr(w, "_dataset_revealed", False)),
                  w.central_widget.isEnabled())
            if st != last:
                out.append([round(time.time() - t0, 2), *st])
                last = st
            time.sleep(0.05)
        log(json.dumps({"phase": tag, "states": out}))

    for tag, ks in (("first", s.ks_dir), ("reload_same", s.ks_dir), ("switch", second)):
        w.load_directory(ks, os.environ.get("HARNESS_DAT") or None)
        sample(tag, 45)


def scenario_switch_mid_warmup(s):
    """Q19: open a cold run, switch to HARNESS_KS2 while its caches still build."""
    w = s.w
    bar = w.cache_progress
    w.load_directory(s.ks_dir, None)
    wait_until(lambda: getattr(w, "_dataset_revealed", False), 300)
    pump(1.5)
    log(json.dumps({"cold_run_bar": [bar.isVisible(), bar.value(), bar.format()]}))
    w.load_directory(os.environ["HARNESS_KS2"], None)
    out, last, t0 = [], None, time.time()
    while time.time() - t0 < 60:
        app.processEvents()
        st = (bar.isVisible(), bar.minimum(), bar.maximum(), bar.value(), bar.format(),
              w.status_bar.currentMessage()[:50], bool(getattr(w, "_dataset_revealed", False)))
        if st != last:
            out.append([round(time.time() - t0, 2), *st])
            last = st
        time.sleep(0.05)
    log(json.dumps({"after_switch": out}))


def scenario_switch_state(s):
    """Q13: after a run switch, cell N's Waveforms PCA is this run's, not the last."""
    from src.gui.panels import waveforms_panel as wp
    w = s.w
    out = {}
    runs = (("first", s.ks_dir, os.environ.get("HARNESS_DAT")),
            ("second", os.environ["HARNESS_KS2"], os.environ.get("HARNESS_DAT2")))
    for tag, ks, dat in runs:
        w.load_directory(ks, dat or None)
        wait_until(lambda: getattr(w, "_dataset_revealed", False), 300)
        pump(1.0)
        s.tab("Waveforms")
        cid = 5
        s.select(cid, settle=0.2)
        wait_until(lambda: (w.waveforms_panel._last_pca_payload or {}).get("cluster_id") == cid, 90)
        pay = w.waveforms_panel._last_pca_payload or {}
        # pca_generation must equal dm_generation (or be None if no PCA came:
        # the PCA needs the raw file; see "isolation").
        out[tag] = {"dm_generation": w.data_manager.generation,
                    "pca_generation": pay.get("_generation"),
                    "isolation": w.waveforms_panel._isolation_label.text(),
                    "cache_generations": sorted({k[0] for k in wp._PCA_CACHE}, key=str)}
        s.shot(f"waveforms_{tag}", w)
    log(json.dumps(out))


def scenario_light_mode(s):
    """Q16: light theme on every tab; report the share of near-black pixels."""
    from qtpy.QtGui import QImage
    s.load()
    w = s.w
    cid = next(int(c) for c in s.dm().cluster_df["cluster_id"]
               if s.dm().get_vision_id_for_cluster(int(c)) in (s.dm().vision_stas or {}))
    s.select(cid, settle=0.5)
    w.toggle_theme()
    pump(0.8)
    tabs = w.analysis_tabs
    report = {}
    for i in range(tabs.count()):
        if not tabs.isTabEnabled(i):
            continue
        tabs.setCurrentIndex(i)
        pump(2.0)
        img = tabs.currentWidget().grab().toImage().convertToFormat(QImage.Format.Format_RGB32)
        ptr = img.constBits()
        ptr.setsize(img.sizeInBytes())
        a = np.frombuffer(ptr, dtype=np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)[:, :img.width(), :3]
        lum = a.mean(axis=2)
        report[tabs.tabText(i)] = round(float((lum < 70).mean()), 3)
        s.shot(f"{i}_{tabs.tabText(i)}", w)
    for name, widget in (("sidebar", w.left_content if hasattr(w, "left_content") else None),
                         ("status_bar", w.status_bar)):
        if widget is not None:
            s.shot(name, widget)
    log(json.dumps({"dark_pixel_share": report}))


def scenario_narrow_window(s):
    """Q17: the window shrinks to laptop widths; tabs still lay out."""
    s.load()
    w = s.w
    out = {"min_width": w.minimumSizeHint().width()}
    for width in (1800, 1050):
        w.resize(width, 700)
        pump(0.5)
        out[f"asked_{width}"] = w.width()
        for name in ("UMAP", "Contrast", "STA", "Raw"):
            try:
                s.tab(name)
            except KeyError:
                continue
            pump(0.8)
            if name == "UMAP":
                sc = w.umap_panel._controls_scroll
                out[f"umap_bar_{width}"] = sc.horizontalScrollBar().isVisible()
                out[f"umap_scroll_h_{width}"] = sc.height()
            s.shot(f"{name}_{width}", w)
    log(json.dumps(out))


def scenario_first_load_timeline(s):
    """Q10: time every phase of a load until the physics and grating caches are done."""
    w = s.w
    bar = w.cache_progress
    t0 = time.time()
    w.load_directory(s.ks_dir, None)
    events, last = [], None
    grating_done = physics_done = False
    while time.time() - t0 < 1500:
        app.processEvents()
        msg = w.status_bar.currentMessage()
        st = (bar.isVisible(), bar.format() if bar.isVisible() else "", msg[:70])
        if st != last:
            dm = w.data_manager
            events.append([round(time.time() - t0, 1), *st,
                           len(getattr(dm, "grating_computed_cache", {}) or {}),
                           int(getattr(dm, "_physics_done_count", 0) or 0)])
            last = st
        grating_done = grating_done or "Grating DS/OS tuning computed" in msg
        physics_done = physics_done or "Physics Cache Ready" in msg
        if grating_done and physics_done:
            break
        time.sleep(0.05)
    dm = w.data_manager
    log(json.dumps({"total_s": round(time.time() - t0, 1), "clusters": len(dm.cluster_df),
                    "grating_cells": len(getattr(dm, "grating_computed_cache", {}) or {}),
                    "events": events}))


VISION_JAR = os.environ.get("VISION_JAR", os.path.expanduser(
    "~/Documents/Development/MEA-fieldlab/src/vision7_symphony/Vision.jar"))
_CHECK_PARAMS_JAVA = """
import edu.ucsc.neurobiology.vision.io.ParametersFile;
import java.util.*;
public class CheckParams {
    public static void main(String[] a) throws Exception {
        ParametersFile p = new ParametersFile(a[0]);
        LinkedHashMap<Integer, ? extends Object> c = p.getClassIDs();
        for (int id : p.getIDList()) System.out.println(id + "\\t" + c.get(id));
        p.close(false);
    }
}
"""


def vision_reads_classes(params_path, work_dir):
    """{id: classID} as Vision's own ParametersFile reads it, or None without Java/Vision.jar."""
    import shutil
    import subprocess
    java_dir = next((os.path.dirname(x) for x in (
        shutil.which("javac"), os.path.expanduser("~/miniconda3/bin/javac"))
        if x and os.path.isfile(x)), None)
    if not (os.path.isfile(VISION_JAR) and java_dir):
        return None
    javac, java = os.path.join(java_dir, "javac"), os.path.join(java_dir, "java")
    src = os.path.join(work_dir, "CheckParams.java")
    with open(src, "w") as f:
        f.write(_CHECK_PARAMS_JAVA)
    subprocess.run([javac, "-cp", VISION_JAR, "-d", work_dir, src], check=True)
    out = subprocess.run([java, "-Djava.awt.headless=true", "-cp",
                          f"{VISION_JAR}{os.pathsep}{work_dir}", "CheckParams", params_path],
                         check=True, capture_output=True, text=True).stdout
    return {int(line.split("\t")[0]): line.split("\t")[1]
            for line in out.splitlines() if line.strip()}


def scenario_params_save(s):
    """Ctrl+S writes the tree into the .params — on a scratch copy, never the lab file."""
    import shutil
    from pathlib import Path
    from qtpy.QtTest import QTest
    from qtpy.QtWidgets import QMessageBox
    from src.gui import callbacks
    from src.analysis import params_classification as pc

    s.load()
    w, dm = s.w, s.dm()
    chk = dm.vision_sort_check()
    log(json.dumps({"step": "sort_check", "n_cells": chk.n_cells, "r2": round(chk.r2, 3),
                    "r2_robust": round(chk.r2_robust, 3), "mismatch": chk.mismatch,
                    "status": w.status_bar.currentMessage(),
                    "warning_label": (w.vision_sort_warning_label.isVisible()
                                      if hasattr(w, "vision_sort_warning_label") else None)}))
    s.shot("status_bar", w.status_bar)
    real = dm.vision_params_path
    real_stamp = pc.file_stamp(real)
    work = os.path.join(OUT, "params_save")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    copy = Path(work) / real.name
    shutil.copy2(real, copy)
    dm.vision_params_path = copy

    def idle():
        return wait_until(lambda: not getattr(w, "_params_busy", False), 60)

    # 1. File > Load Classification from Vision .params
    orig_q = callbacks.QMessageBox.question
    callbacks.QMessageBox.question = staticmethod(
        lambda *a, **k: QMessageBox.StandardButton.Yes)
    try:
        w.load_params_classification_action.trigger()
        idle()
    finally:
        callbacks.QMessageBox.question = orig_q
    file_classes = pc.read_classes(copy)
    tree = callbacks.vision_class_ids(w)
    shared = [v for v in file_classes if v in tree]
    log(json.dumps({"step": "import", "rows": len(file_classes), "rows_with_a_cell": len(shared),
                    "tree_matches_file": all(tree[v] == file_classes[v] for v in shared),
                    "top_groups": [w.tree_model.item(i).text()
                                   for i in range(w.tree_model.rowCount())]}))

    asked = []
    orig_c = callbacks._confirm_params_save
    callbacks._confirm_params_save = lambda mw, path, diff, check=callbacks.NO_SORT_CHECK: (
        asked.append(callbacks.params_save_summary(path, diff, check)), True)[1]

    def ctrl_s():
        w.activateWindow()
        pump(0.1)
        n_noisy = int((dm.status_df["status"] == "Noisy").sum()) if len(dm.status_df) else 0
        before = pc.file_stamp(copy)
        QTest.keyClick(w, Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier)
        started = getattr(w, "_params_busy", False) or pc.file_stamp(copy) != before
        via = "shortcut"
        if not started and "classification" not in w.status_bar.currentMessage():
            via = "action"          # the offscreen platform did not route the key
            w.save_params_action.trigger()
        idle()
        pump(0.3)
        noisy_now = int((dm.status_df["status"] == "Noisy").sum()) if len(dm.status_df) else 0
        return via, w.status_bar.currentMessage(), noisy_now != n_noisy

    try:
        # 2. Ctrl+S with no edits
        via, msg, marked = ctrl_s()
        log(json.dumps({"step": "save_unchanged", "via": via, "status": msg,
                        "marked_noisy": marked, "asked": len(asked),
                        "file_unchanged": pc.file_stamp(copy) == pc.file_stamp(real)}))

        # 3. move 5 classified cells to a new group, Ctrl+S
        groups = {dm.get_cluster_id_for_vision(v): g for v, g in callbacks.tree_vision_groups(w)}
        moved = [c for c, g in groups.items() if g][:5]
        for c in moved:
            groups[c] = ["Encore harness", "moved"]
        callbacks.apply_classification(w, groups)
        via, msg, marked = ctrl_s()
        after = pc.read_classes(copy)
        vids = [dm.get_vision_id_for_cluster(c) for c in moved]
        vision = vision_reads_classes(str(copy), work)
        log(json.dumps({
            "step": "save_edit", "via": via, "status": msg, "marked_noisy": marked,
            "asked": len(asked), "summary": asked[-1] if asked else None,
            "moved_now": [after[v] for v in vids],
            "others_unchanged": all(after[v] == c for v, c in file_classes.items() if v not in vids),
            "backup_is_original": (Path(str(copy) + ".bak").read_bytes() == real.read_bytes()),
            "vision_jar_agrees": None if vision is None else vision == after,
            "tmp_files_left": [p.name for p in Path(work).glob("*.tmp")]}))

        # 4. another edit, same session, file unchanged since our save: no dialog
        n_asked = len(asked)
        groups[moved[0]] = ["Encore harness", "second"]
        callbacks.apply_classification(w, groups)
        via, msg, _ = ctrl_s()
        log(json.dumps({"step": "save_again", "status": msg, "asked_again": len(asked) > n_asked}))

        # 5. someone else saves the file (as Vision would): the dialog comes back
        pc.write_classes(copy, {vids[1]: "All/saved elsewhere"}, keep_backup=False)
        groups[moved[2]] = ["Encore harness", "third"]
        callbacks.apply_classification(w, groups)
        n_asked = len(asked)
        via, msg, _ = ctrl_s()
        log(json.dumps({"step": "save_after_outside_edit", "status": msg,
                        "asked_again": len(asked) > n_asked}))
    finally:
        callbacks._confirm_params_save = orig_c
        dm.vision_params_path = real
    log(json.dumps({"step": "lab_file_untouched", "ok": pc.file_stamp(real) == real_stamp}))
    s.shot("tree_after", w)


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
