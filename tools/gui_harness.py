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
    # The population pane (RF mosaic, dynamics, ACG, firing rate), light mode.
    tabs.setCurrentIndex(0)
    w.toggle_population_split_view(True)
    pump(1.0)
    s.select(cid, settle=2.5)
    s.shot("population_pane", w.pop_context_widget)
    s.shot("window_with_population", w)
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


def scenario_population_fr(s):
    """Q23: the population pane shows the group's firing rate over the recording."""
    s.load()
    w = s.w
    w.toggle_population_split_view(True)
    pump(1.0)
    ids = [int(c) for c in s.dm().cluster_df["cluster_id"].values]
    s.select(ids[3], settle=2.0)
    state = getattr(w.pop_fr_canvas, "_fr_state", None)
    log(json.dumps({"fr_drawn": state is not None,
                    "n_traces": (len(state["shadow_lines"].get_segments()) if state else 0),
                    "summary": w.pop_fr_summary.text()}))
    s.shot("population_pane", w.pop_context_widget)


def scenario_keyboard(s):
    """Q39: the keyboard workflow on a loaded run; how long each key takes.

    Tree edits stay in memory (only File > Save Results writes a tree), so
    nothing reaches the lab folder.
    """
    from qtpy.QtCore import Qt
    from qtpy.QtTest import QTest
    from qtpy.QtWidgets import QInputDialog
    from src.gui import callbacks, keymap
    s.load()
    w = s.w
    w.activateWindow()
    pump(0.5)
    ids = [int(c) for c in s.dm().cluster_df["cluster_id"].values][:6]
    ctrl, shift = Qt.KeyboardModifier.ControlModifier, Qt.KeyboardModifier.ShiftModifier
    timings = {}

    def press(name, key, mods=Qt.KeyboardModifier.NoModifier, target=None, settle=0.3):
        t = time.time()
        QTest.keyClick(target or w.tree_view, key, mods)
        timings[name] = round((time.time() - t) * 1000)
        pump(settle)

    def group_of(cid):
        parent = callbacks.find_items_by_cluster_ids(w, [cid])[0].parent()
        return parent.text() if parent is not None else None

    def select(*cids):
        w._select_cluster_in_tree(cids[0])
        pump(0.5)
        if len(cids) > 1:
            from qtpy.QtCore import QItemSelectionModel
            sel = w.tree_view.selectionModel()
            for cid in cids[1:]:
                item = callbacks.find_items_by_cluster_ids(w, [cid])[0]
                sel.select(w.tree_model.indexFromItem(item),
                           QItemSelectionModel.SelectionFlag.Select
                           | QItemSelectionModel.SelectionFlag.Rows)
        w.tree_view.setFocus()

    result = {}
    for n in (2, 5, 1):
        press(f"ctrl_{n}", getattr(Qt.Key, f"Key_{n}"), ctrl, settle=1.0)
        result[f"tab_after_ctrl_{n}"] = w.analysis_tabs.tabText(w.analysis_tabs.currentIndex())
    press("ctrl_tab", Qt.Key.Key_Tab, ctrl, settle=1.0)
    result["tab_after_ctrl_tab"] = w.analysis_tabs.tabText(w.analysis_tabs.currentIndex())

    QInputDialog.getText = staticmethod(lambda *a, **k: ("Harness group", True))
    select(ids[0], ids[1])
    press("ctrl_g", Qt.Key.Key_G, ctrl)
    result["ctrl_g"] = [group_of(ids[0]), group_of(ids[1])]
    target = next(item for path, item in callbacks.collect_group_items(w)
                  if path.endswith("Harness group"))
    keymap.pick_group = lambda parent, groups, preselect=None: target
    select(ids[2])
    press("ctrl_m", Qt.Key.Key_M, ctrl)
    select(ids[3])
    press("ctrl_shift_m", Qt.Key.Key_M, ctrl | shift)
    result["ctrl_m_then_repeat"] = [group_of(ids[2]), group_of(ids[3])]
    select(ids[4])
    w.analysis_tabs.currentWidget().setFocus()
    press("delete_from_plot", Qt.Key.Key_Delete, target=w.analysis_tabs.currentWidget())
    result["delete_from_plot"] = group_of(ids[4])
    press("ctrl_t", Qt.Key.Key_T, ctrl)
    result["ctrl_t_view"] = w.view_stack.currentIndex()
    press("ctrl_t_back", Qt.Key.Key_T, ctrl, target=w.table_view)
    press("ctrl_p", Qt.Key.Key_P, ctrl, settle=2.0)
    result["ctrl_p_population_pane"] = w.pop_context_widget.isVisible()
    s.shot("population_split", w)
    press("ctrl_p_off", Qt.Key.Key_P, ctrl)
    bar = w.cluster_search_bar
    bar.setFocus()
    QTest.keyClicks(bar, "12 ")
    result["search_text"] = bar.text()
    QTest.keyClick(bar, Qt.Key.Key_Escape)
    result["search_after_esc"] = bar.text()
    log(json.dumps({"result": result, "ms_per_key": timings}))


def scenario_population_compare(s):
    """Q45: selected + pinned cells over their Vision-class population, both themes.

    Loads the run's own Vision classes into the tree (in memory only).
    """
    from qtpy.QtWidgets import QMessageBox
    from src.gui import callbacks, keymap
    from src.analysis.class_names import canonical_type
    s.load()
    w = s.w
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    callbacks.load_classification_from_params(w)
    wait_until(lambda: any("brisk" in p.lower() for p, _ in callbacks.collect_group_items(w)), 120)
    groups = callbacks.collect_group_items(w)
    path, group = next((p, g) for p, g in groups
                       if (canonical_type(p) or ("",))[0] == "ON brisk transient"
                       or p.lower().endswith("brisk transient"))
    cells = callbacks.iter_cluster_ids(group)
    log(f"group {path}: {len(cells)} cells")
    w.toggle_population_split_view(True)
    pump(1.0)
    s.select(cells[0], settle=3.0)
    t = time.time()
    s.select(cells[1], settle=0.0)
    ok = wait_until(lambda: any(k.get_visible() and k.get_text() == f"Cell {cells[1]}"
                                for k in w.pop_fr_canvas._fr_state.get("compare_keys", [])), 10)
    log(f"overlay follows the selection: {ok} in {time.time() - t:.2f}s (incl. 150 ms debounce)")
    w._select_cluster_in_tree(cells[2]); pump(1.0)
    keymap.toggle_pin(w)
    w._select_cluster_in_tree(cells[3]); pump(1.0)
    keymap.toggle_pin(w)
    s.select(cells[1], settle=3.0)
    keys = {pane: [k.get_text() for k in getattr(getattr(w, attr), state).get("compare_keys", [])
                   if k.get_visible()]
            for pane, (attr, state) in __import__(
                "src.gui.panels.population_compare", fromlist=["PANES"]).PANES.items()}
    log(json.dumps({"pins": w._pinned_cells, "keys": keys}))
    cost = []
    for cid in cells[5:15]:
        t = time.time()
        callbacks.refresh_population_overlays(w, cid)
        w.pop_fr_canvas.repaint()
        cost.append((time.time() - t) * 1000)
    log(f"overlay update alone (3 panes + one repaint), 10 cells: "
        f"median {np.median(cost):.0f} ms, max {max(cost):.0f} ms")
    s.shot("compare_dark", w.pop_context_widget)
    w.toggle_theme()
    pump(2.0)
    s.select(cells[4], settle=0.5)
    s.select(cells[1], settle=3.0)
    s.shot("compare_light", w.pop_context_widget)
    s.shot("window_light", w)
    w.toggle_theme()
    pump(1.0)


def scenario_types_tab(s):
    """Q40/Q42: the Types tab (barcode + mosaic atlas) on the run's own Vision classes."""
    from qtpy.QtWidgets import QMessageBox
    from src.gui import callbacks
    s.load()
    w = s.w
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    callbacks.load_classification_from_params(w)
    wait_until(lambda: any("brisk" in p.lower() for p, _ in callbacks.collect_group_items(w)), 120)
    wait_until(lambda: getattr(w.data_manager, "physics_cache_complete", True), 5)
    pump(8.0)                                   # let the physics / standard caches fill
    t = time.time()
    s.tab("Types")
    pump(0.2)
    tp = w.types_panel
    log(f"types tab built in {time.time() - t:.2f}s: {tp.status.text()}")
    code = tp._barcode
    log(json.dumps({
        "bands": [(g, e - b) for g, b, e in code.bands] if code else [],
        "misfits": len(code.misfits()) if code else 0,
        "tiles": [(tl["group"], tl["stats"].verdict, len(tl["holes"])) for tl in tp._tiles]}))
    t = time.time()
    if os.environ.get("HARNESS_PROFILE"):
        import cProfile, pstats, io
        prof = cProfile.Profile()
        prof.enable()
        tp.refresh(force=True)
        prof.disable()
        buf = io.StringIO()
        pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(18)
        log(buf.getvalue())
    else:
        tp.refresh(force=True)
    log(f"forced rebuild {1000 * (time.time() - t):.0f} ms")
    if code and code.cells:
        cid = code.cells[len(code.cells) // 2]
        tp._select(cid)
        pump(1.0)
        log(f"row click selects cell {cid}: {w._get_selected_cluster_id() == cid}")
    s.shot("types_dark", w)
    for feature in ("Autocorrelation", "Chirp response"):
        tp.feature_combo.setCurrentText(feature)
        pump(0.5)
        log(f"{feature}: {tp.status.text()}")
    tp.feature_combo.setCurrentText("Autocorrelation")
    pump(0.3)
    s.shot("types_acg_dark", w)
    tp.feature_combo.setCurrentText("STA time course")
    w.toggle_theme()
    pump(2.0)
    s.shot("types_light", w)
    w.toggle_theme()
    pump(0.5)


def scenario_suggest(s):
    """Q36: suggested classes end to end on a labelled run (its own prep left out of training).

    HARNESS_TYPE_LIBRARY points at a scratch library cache, so ~/.encore is not written.
    """
    from pathlib import Path
    from qtpy.QtCore import Qt
    from qtpy.QtTest import QTest
    from qtpy.QtWidgets import QMessageBox
    from src.analysis import type_library as tl
    from src.gui import callbacks, suggestions
    if os.environ.get("HARNESS_TYPE_LIBRARY"):
        tl.CACHE_PATH = Path(os.environ["HARNESS_TYPE_LIBRARY"])
    else:
        tl.CACHE_PATH = Path(OUT) / "type_library.npz"
    shown = []
    QMessageBox.information = staticmethod(lambda *a, **k: shown.append(a[2] if len(a) > 2 else ""))
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    s.load()
    w = s.w
    callbacks.load_classification_from_params(w)
    wait_until(lambda: any("brisk" in p.lower() for p, _ in callbacks.collect_group_items(w)), 120)
    s.tab("Types")
    t = time.time()
    w.types_panel.suggest_btn.click()
    ok = wait_until(lambda: getattr(w, "_suggestions", None) is not None, 600)
    log(f"suggestions ready={ok} in {time.time() - t:.1f}s")
    st = w._suggestions
    log(shown[-1] if shown else "(no summary)")
    classes = suggestions.folder_classes(w)
    named = [(c, classes[c], st.by_cell[c]) for c in st.by_cell if classes.get(c)]
    agree = [sg.best is not None and sg.best[0] == here for _c, here, sg in named]
    from src.gui.panels.types_panel import tree_groups
    group_of, _ = tree_groups(w)
    weak = [st.by_cell[c] for c, g in group_of.items() if g.startswith("weak") and c in st.by_cell]
    log(json.dumps({
        "cells": len(st.by_cell), "trained_on": st.n_train,
        "named_cells": len(named), "agree_with_label": round(sum(agree) / max(len(named), 1), 3),
        "weak_cells": len(weak), "weak_novel": round(sum(x.novel for x in weak) / max(len(weak), 1), 3),
        "weak_confident": round(sum(x.confident for x in weak) / max(len(weak), 1), 3),
        "review_queue": len(suggestions.review_queue(w))}))
    w.tree_view.setFocus()
    QTest.keyClick(w.tree_view, Qt.Key.Key_J, Qt.KeyboardModifier.ControlModifier)
    pump(1.5)
    cid = w._get_selected_cluster_id()
    log(f"Ctrl+J -> cell {cid}: {w.suggestion_label.text()}")
    before = suggestions.folder_class(w, cid)
    QTest.keyClick(w.tree_view, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    pump(1.0)
    log(f"Ctrl+Enter: cell {cid} {before} -> {suggestions.folder_class(w, cid)}; "
        f"now at {w._get_selected_cluster_id()}")
    s.shot("suggest_sidebar", w)
    n = suggestions.accept_confident(w)
    pump(0.5)
    log(f"accept confident moved {n} cells")
    w.types_panel.refresh(force=True)
    pump(0.5)
    s.shot("suggest_types_after", w)


def scenario_feature_presets(s):
    """Q15: save a preset, reopen the window, the preset is back. Temp settings only."""
    from qtpy.QtCore import QSettings
    from src.gui import callbacks
    from src.gui.panels import feature_presets as fp
    tmp = QSettings(os.path.join(OUT, "presets_test.ini"), QSettings.Format.IniFormat)
    tmp.clear()
    fp._settings = lambda settings=None, _t=tmp: settings if settings is not None else _t
    s.load()
    ids = [int(c) for c in s.dm().cluster_df["cluster_id"].values]
    shown = {}

    def keep_open(dlg):
        shown["dlg"] = dlg
        return 0

    orig = callbacks.FeatureExtractionWindow.exec
    callbacks.FeatureExtractionWindow.exec = keep_open

    def open_window():
        callbacks.feature_extraction(s.w, ids)
        dlg = shown["dlg"]
        dlg.show()
        wait_until(lambda: bool(dlg.catalog) and bool(dlg._panels), 300)
        pump(0.5)
        return dlg

    try:
        dlg = open_window()
        first = [tuple(p) for p in dlg._panels]
        mine = [first[1], first[0], first[3], first[2], first[5], first[4]]
        fp.save_preset("harness", mine)
        dlg._refresh_preset_combo("harness")
        dlg._on_preset_chosen(0)
        chosen = [tuple(p) for p in dlg._panels]
        dlg.close()
        dlg2 = open_window()
        log(json.dumps({"default_first4": first[:4], "applied": chosen == mine,
                        "reopened_with": dlg2.preset_combo.currentText(),
                        "reopened_panels_match": [tuple(p) for p in dlg2._panels] == mine}))
        dlg2.close()
    finally:
        callbacks.FeatureExtractionWindow.exec = orig


def scenario_umap_eval(s):
    """Q28: how well do the UMAP features separate the run's Vision cell types?

    Labels: the named types (>= 8 cells) in the run's .params classID.
    Score: leave-one-out 5-nearest-neighbour accuracy among labelled cells,
    in the feature space (the observed-Euclidean distance the app uses) and
    in a 2D UMAP of all cells. Chance = the largest class share.
    """
    import umap
    from collections import Counter
    from src.analysis import analysis_core
    from src.analysis import params_classification as pc

    s.load()
    dm, panel = s.dm(), s.w.umap_panel
    classes = pc.read_classes(dm.vision_params_path)
    raw = {dm.get_cluster_id_for_vision(v): c.lower() for v, c in classes.items()}
    counts = Counter(raw.values())
    named = {c for c, n in counts.items() if n >= 8 and c != "all"
             and not any(t in c for t in ("unclass", "/nc", "weak", "trash"))}
    labels = {cid: c for cid, c in raw.items() if c in named}
    ids = [int(c) for c in dm.cluster_df["cluster_id"]]
    dm.ensure_physics_cache(ids)
    # The ACG comes from the standard-plots pass; evaluate only once it is done.
    wait_until(lambda: len(dm.standard_plot_cache) >= len(ids), 900)
    base = panel.get_feature_config()
    blocks = [k for k in base if k.startswith("use_")]

    def knn(dist, lab, k=5):
        hits = 0
        for i in range(len(lab)):
            d = dist[i].copy()
            d[i] = np.inf
            nn = np.argsort(d)[:k]
            vote = Counter(lab[j] for j in nn).most_common(1)[0][0]
            hits += vote == lab[i]
        return hits / len(lab)

    def evaluate(cfg):
        rb, vids, disc = dm.get_raw_feature_blocks(ids, panel.get_filter_config())
        m, _cols = analysis_core.build_feature_matrix(rb, cfg)
        m, vids, disc, rb = analysis_core.drop_empty_feature_rows(m, vids, disc, rb)
        vids = [int(v) for v in vids]
        dist = analysis_core.observed_euclidean_distances(m)
        dist = np.where(np.isfinite(dist), dist, np.nanmax(dist[np.isfinite(dist)]) * 2)
        emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="precomputed",
                        random_state=0).fit_transform(dist)
        idx = [i for i, v in enumerate(vids) if v in labels]
        lab = [labels[vids[i]] for i in idx]
        sub = dist[np.ix_(idx, idx)]
        e = emb[idx]
        edist = np.linalg.norm(e[:, None] - e[None], axis=2)
        pol = [l.split("/")[1] for l in lab]
        return {"n_cells": len(vids), "n_labelled": len(idx), "cols": int(m.shape[1]),
                "knn_features": round(knn(sub, lab), 3), "knn_umap": round(knn(edist, lab), 3),
                "polarity_knn_umap": round(knn(edist, pol), 3)}

    out = {"types": dict(Counter(labels.values())),
           "chance": round(max(Counter(labels.values()).values()) / max(len(labels), 1), 3),
           "default_config": {k: v for k, v in base.items()}}
    out["default"] = evaluate(base)
    for b in blocks:
        if base.get(b):
            out["without_" + b[4:]] = evaluate({**base, b: False})
        only = {k: (v if not k.startswith("use_") else k == b) for k, v in base.items()}
        try:
            out["only_" + b[4:]] = evaluate(only)
        except Exception as exc:          # a block this run lacks
            out["only_" + b[4:]] = f"n/a ({type(exc).__name__})"
    log(json.dumps(out))


def scenario_orientation_check(s):
    """Q20: does a cell sit at the same place in its STA movie and in the mosaic?"""
    from src.analysis import rf_geometry
    s.load()
    w, dm = s.w, s.dm()
    vp = dm.vision_params
    fits = {}
    for cid in [int(c) for c in dm.cluster_df["cluster_id"]]:
        vid = dm.get_vision_id_for_cluster(cid)
        f = rf_geometry.raw_rf_fit(vp, vid)
        if f is not None and vid in dm.vision_stas:
            fits[cid] = f
    left = min(fits, key=lambda c: fits[c].x0)
    top = max(fits, key=lambda c: fits[c].y0)
    w.toggle_population_split_view(True)
    s.tab("STA")
    out = {"stimulus_w_h": [dm.vision_sta_width, dm.vision_sta_height]}
    for tag, cid in (("leftmost", left), ("topmost", top)):
        s.select(cid, settle=0.3)
        wait_until(lambda: w.sta_panel.current_sta_cluster_id == cid, 30)
        pump(1.5)
        ax = w.pop_mosaic_canvas.fig.axes[0] if w.pop_mosaic_canvas.fig.axes else None
        out[tag] = {"cid": cid, "x0": round(fits[cid].x0, 1), "y0": round(fits[cid].y0, 1),
                    "mosaic_xlim": [round(v, 1) for v in ax.get_xlim()] if ax else None,
                    "mosaic_ylim": [round(v, 1) for v in ax.get_ylim()] if ax else None}
        s.shot(f"{tag}_sta", w.sta_panel.rf_canvas)
        s.shot(f"{tag}_mosaic", w.pop_mosaic_canvas)
    log(json.dumps(out))


def scenario_array_alignment(s):
    """Q20: array views turned to the screen; numbers + EI screenshots off/on."""
    from src.analysis import rf_geometry, vision_sort_check as vsc
    from src.gui import array_orientation as ao
    s.load()
    w, dm = s.w, s.dm()
    check = dm.vision_sort_check()
    df = dm.cluster_df
    rf, pos = [], []
    for c, x, y in zip(df["cluster_id"], df["x_um"], df["y_um"]):
        f = rf_geometry.raw_rf_fit(dm.vision_params, dm.get_vision_id_for_cluster(int(c)))
        if f is not None and np.isfinite(x) and np.isfinite(y):
            rf.append((f.x0, f.y0))
            pos.append((x, y))
    rf, pos = np.array(rf), np.array(pos)
    shown = ao.to_display(pos, check.screen_turn or ao.IDENTITY)
    coef, *_ = np.linalg.lstsq(np.c_[shown, np.ones(len(shown))], rf, rcond=None)
    log(json.dumps({"screen_turn": check.screen_turn, "described": ao.describe(check.screen_turn or ao.IDENTITY),
                    "turn_left_after_display": vsc.nearest_quarter_turn(coef[:2].T),
                    "n_cells": len(rf)}))
    cid = max((int(c) for c in df["cluster_id"]
               if rf_geometry.raw_rf_fit(dm.vision_params, dm.get_vision_id_for_cluster(int(c)))
               and dm.get_vision_id_for_cluster(int(c)) in dm.vision_stas),
              key=lambda c: rf_geometry.raw_rf_fit(dm.vision_params, dm.get_vision_id_for_cluster(c)).x0)
    w.toggle_population_split_view(True)
    for state in (False, True):
        w.align_array_action.setChecked(state)
        s.tab("EI")
        s.select(cid, settle=3.0)
        s.shot(f"ei_align_{'on' if state else 'off'}", w.ei_panel)
    s.tab("STA")
    s.select(cid, settle=2.0)
    s.shot("sta_same_cell", w.sta_panel.rf_canvas)
    s.shot("mosaic_same_cell", w.pop_mosaic_canvas)
    log(json.dumps({"cell": cid, "rf_x0": round(rf_geometry.raw_rf_fit(dm.vision_params, dm.get_vision_id_for_cluster(cid)).x0, 1)}))


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
