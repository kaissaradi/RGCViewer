"""Suggested classes in the GUI: build, show, accept, review (PLAN.md Q36).

Types tab ▸ "Suggest classes" builds the lab library (``type_library``,
cached in ~/.encore), trains the suggester and scores every cell of the
loaded run, all in the background. Then:

* the line under the cell list shows the selected cell's suggestion;
* Ctrl+Enter accepts it and jumps to the next cell to review;
* Ctrl+J jumps to the next cell to review without accepting;
* "Accept confident" (Types tab) accepts every suggestion ≥ REVIEW_BELOW
  for cells not yet in a type folder. It never moves a cell you classified.

A label check runs with it: cells whose folder disagrees with a confident
suggestion, and runs whose ON/OFF labels contradict the STA.

Accepted cells go to a folder named the lab's way (``ON`` ▸ ``brisk
sustained``), so Ctrl+S writes ``All/ON/brisk sustained`` into the .params.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QMessageBox

from ..analysis.class_names import canonical_type
from ..analysis.type_suggest import REVIEW_BELOW, Suggestion

logger = logging.getLogger(__name__)


@dataclass
class RunSuggestions:
    by_cell: Dict[int, Suggestion]                 # cluster id -> suggestion
    n_train: int
    library_notes: List[str] = field(default_factory=list)
    swapped_polarity: Optional[float] = None       # share of labelled cells with the opposite ON/OFF
    reviewed: set = field(default_factory=set)     # cells the user has accepted or skipped
    library: object = None                         # the full lab library (the type atlas uses it)
    run_features: Dict[int, object] = field(default_factory=dict)   # cluster id -> feature row


# --- background work ------------------------------------------------------------

def _compute(dm, progress):
    from ..analysis import type_features as tf
    from ..analysis import type_library as tl
    from ..analysis.type_suggest import Suggester
    full = tl.library(progress=progress)
    # Never learn from the run being checked (its own labels would agree).
    own = tl.prep_of(getattr(dm, "vision_params_path", None) or getattr(dm, "kilosort_dir", "") or "")
    lib = full.without_prep(own)
    sg = Suggester(lib)
    vp = dm.vision_params
    stas = getattr(dm, "vision_stas", None)
    stixel = float(getattr(getattr(stas, "reader", None), "stixel_size", 0) or 0) or 1.0
    vids = list(vp.get_cell_ids())
    cells = tf.cells_from_vision(vp, vids, stixel)
    order = list(cells)
    X, pol = tf.run_features([cells[v] for v in order])
    out, rows = {}, {}
    for i, (vid, s) in enumerate(zip(order, sg.suggest(X, pol))):
        try:
            cid = int(dm.get_cluster_id_for_vision(int(vid)))
        except Exception:
            continue
        out[cid] = s
        rows[cid] = X[i]
    return RunSuggestions(out, sg.n_train, list(lib.notes), library=full, run_features=rows)


def start(main_window):
    dm = getattr(main_window, "data_manager", None)
    if dm is None or getattr(dm, "vision_params", None) is None:
        QMessageBox.information(main_window, "Suggest classes",
                                "Suggestions need the run's Vision .params (STA time course "
                                "and autocorrelation). Load Vision files first.")
        return
    from . import callbacks
    state = {"i": 0, "n": 0}
    timer = QTimer(main_window)
    timer.setInterval(500)
    timer.timeout.connect(lambda: main_window.status_bar.showMessage(
        f"Suggest classes: reading the lab's labelled runs "
        f"({state['i']}/{state['n'] or '?'})…"))
    timer.start()
    panel = getattr(main_window, "types_panel", None)
    if panel is not None:
        panel.suggest_btn.setEnabled(False)

    def progress(i, n):
        state["i"], state["n"] = i, n

    generation = getattr(dm, "generation", None)

    def done(result: RunSuggestions):
        timer.stop()
        if panel is not None:
            panel.suggest_btn.setEnabled(True)
        if getattr(main_window.data_manager, "generation", None) != generation:
            return                                   # a different run is open now
        result.swapped_polarity = polarity_mismatch(main_window, result)
        main_window._suggestions = result
        if getattr(main_window, "_open_atlas_after_suggest", False):
            main_window._open_atlas_after_suggest = False
            refresh_line(main_window)
            from .panels.type_atlas import open_atlas
            open_atlas(main_window)
            return
        show_summary(main_window, result)
        refresh_line(main_window)

    def failed(exc):
        timer.stop()
        if panel is not None:
            panel.suggest_btn.setEnabled(True)
        logger.warning("suggest classes failed", exc_info=exc)
        QMessageBox.warning(main_window, "Suggest classes",
                            f"Could not build suggestions: {exc}")

    callbacks._run_in_background(main_window, lambda: _compute(dm, progress), done, failed)


# --- tree helpers ------------------------------------------------------------------

def folder_classes(main_window) -> Dict[int, Optional[str]]:
    """{cluster id: named type of its folder, or None} — one walk of the tree."""
    from .panels.types_panel import tree_groups
    group_of, _order = tree_groups(main_window)
    out = {}
    for cid, g in group_of.items():
        ct = canonical_type(g)
        out[cid] = ct[0] if ct else None
    return out


def folder_class(main_window, cluster_id) -> Optional[str]:
    """The named type of the folder a cell sits in, or None."""
    return folder_classes(main_window).get(int(cluster_id))


def class_folder(main_window, class_name: str):
    """A folder for ``class_name``: an existing one, else ``ON`` ▸ ``brisk sustained``."""
    from . import callbacks
    for path, item in callbacks.collect_group_items(main_window):
        ct = canonical_type(path)
        if ct is not None and ct[0] == class_name and not ct[1]:
            return item
    polarity, sub = class_name.split(" ", 1)
    root = main_window.tree_model.invisibleRootItem()
    parent = None
    for i in range(root.rowCount()):
        child = root.child(i)
        if child is not None and callbacks.is_group_item(child) and child.text().lower() == "all":
            root = child
            break
    for i in range(root.rowCount()):
        child = root.child(i)
        if child is not None and callbacks.is_group_item(child) \
                and child.text().strip().lower() == polarity.lower():
            parent = child
            break
    if parent is None:
        callbacks.add_new_group(main_window, polarity,
                                parent_item=None if root is main_window.tree_model.invisibleRootItem() else root)
        parent = root.child(0)
    callbacks.add_new_group(main_window, sub, parent_item=parent)
    return parent.child(0)


# --- actions -------------------------------------------------------------------------

def _state(main_window) -> Optional[RunSuggestions]:
    return getattr(main_window, "_suggestions", None)


def accept(main_window, cluster_id=None, rank=0, advance=True):
    from . import callbacks
    st = _state(main_window)
    if st is None:
        main_window.status_bar.showMessage("No suggestions yet: Types tab ▸ Suggest classes.", 4000)
        return
    cid = cluster_id if cluster_id is not None else main_window._get_selected_cluster_id()
    s = st.by_cell.get(int(cid)) if cid is not None else None
    if s is None or s.novel or len(s.ranked) <= rank:
        main_window.status_bar.showMessage("No suggestion for this cell.", 3000)
        return
    name = s.ranked[rank][0]
    callbacks.move_cluster_ids_to_group(main_window, [int(cid)], class_folder(main_window, name))
    st.reviewed.add(int(cid))
    main_window.status_bar.showMessage(f"Cell {cid} → {name}.", 3000)
    if advance:
        next_to_review(main_window)


def review_queue(main_window) -> List[int]:
    """Cells to look at, least confident first: not yet in a type folder, or
    in one that a confident suggestion disagrees with."""
    st = _state(main_window)
    if st is None:
        return []
    from .panels.types_panel import tree_groups
    group_of, _ = tree_groups(main_window)
    queue = []
    for cid, s in st.by_cell.items():
        if cid in st.reviewed or s.best is None:
            continue
        g = group_of.get(cid)
        ct = canonical_type(g) if g else None
        if ct is None:
            queue.append((s.best[1], cid))
        elif ct[0] != s.best[0] and s.confident:
            queue.append((0.0, cid))                 # a labelled cell to double-check
    return [cid for _p, cid in sorted(queue)]


def next_to_review(main_window):
    queue = review_queue(main_window)
    st = _state(main_window)
    if not queue:
        main_window.status_bar.showMessage(
            "Nothing left to review." if st else "No suggestions yet: Types tab ▸ Suggest classes.", 4000)
        return
    cid = queue[0]
    if st is not None:
        st.reviewed.add(cid)                         # Ctrl+J moves on; the cell stays as it is
    main_window._select_cluster_in_tree(cid)
    main_window.status_bar.showMessage(f"{len(queue) - 1} more to review.", 3000)


def mosaic_screen(main_window, todo: Dict[str, List[int]], here) -> Tuple[Dict[str, List[int]], List[int]]:
    """Hold back suggestions that would break the class's mosaic.

    A type tiles the retina, so two of its cells should not sit on top of
    each other (NNND < CLOSE_NNND, docs/design/rgc_types.md). Candidates go
    in by falling confidence; one that overlaps a member already there (or
    one accepted just before it) is left for review instead.
    """
    from ..analysis import mosaic_stats as ms
    from .panels.types_panel import rf_fits
    dm = main_window.data_manager
    st = _state(main_window)
    kept, held = {}, []
    for name, cids in todo.items():
        members = [c for c, cls in here.items() if cls == name]
        fits = rf_fits(dm, members + cids)
        placed = [fits[c] for c in members if fits.get(c) is not None]
        order = sorted(cids, key=lambda c: -st.by_cell[c].best[1])
        for c in order:
            f = fits.get(c)
            if f is not None and placed and \
                    float(ms.nnnd_matrix([f], placed).min()) < ms.CLOSE_NNND:
                held.append(c)
                continue
            kept.setdefault(name, []).append(c)
            if f is not None:
                placed.append(f)
    return kept, held


def accept_confident(main_window) -> int:
    from . import callbacks
    st = _state(main_window)
    if st is None:
        return 0
    todo: Dict[str, List[int]] = {}
    here = folder_classes(main_window)
    for cid, s in st.by_cell.items():
        if s.confident and here.get(cid) is None:
            todo.setdefault(s.best[0], []).append(cid)
    todo, held = mosaic_screen(main_window, todo, here)
    if not todo:
        QMessageBox.information(main_window, "Accept confident suggestions",
                                "No unclassified cell has a confident suggestion.")
        return 0
    lines = "\n".join(f"  {name}: {len(c)}" for name, c in sorted(todo.items()))
    n = sum(len(c) for c in todo.values())
    held_note = (f"\n\n{len(held)} more are held back for review (Ctrl+J): their RF would sit on "
                 f"top of a cell already in that class, which one type should not do."
                 if held else "")
    if QMessageBox.question(
            main_window, "Accept confident suggestions",
            f"Move {n} unclassified cells to their suggested class "
            f"(confidence ≥ {REVIEW_BELOW:.0%})?\n\n{lines}{held_note}\n\nCells you already "
            f"classified stay where they are. Nothing is saved until Ctrl+S.") \
            != QMessageBox.StandardButton.Yes:
        return 0
    for name, cids in todo.items():
        callbacks.move_cluster_ids_to_group(main_window, cids, class_folder(main_window, name))
        st.reviewed.update(cids)
    main_window.status_bar.showMessage(f"Moved {n} cells to their suggested classes.", 5000)
    return n


# --- display --------------------------------------------------------------------------------

def line_for(main_window, cluster_id) -> str:
    st = _state(main_window)
    if st is None or cluster_id is None:
        return ""
    s = st.by_cell.get(int(cluster_id))
    if s is None:
        return "No suggestion (no Vision time course)."
    if s.novel:
        return "No suggestion: unlike any labelled cell (maybe a type the lab has not named)."
    (name, p), rest = s.ranked[0], s.ranked[1:3]
    other = "  ·  ".join(f"{n} {q:.0%}" for n, q in rest if q >= 0.05)
    here = folder_class(main_window, cluster_id)
    note = ""
    if here is not None and here != name:
        note = f"   (in {here})"
    return f"Suggested: {name} {p:.0%}" + (f"  ·  {other}" if other else "") + note + \
        "   —  Ctrl+Enter accept · Ctrl+J next"


def refresh_line(main_window, cluster_id=None):
    label = getattr(main_window, "suggestion_label", None)
    if label is None:
        return
    if cluster_id is None:
        cluster_id = main_window._get_selected_cluster_id()
    text = line_for(main_window, cluster_id)
    label.setText(text)
    label.setVisible(bool(text))


def polarity_mismatch(main_window, st: RunSuggestions) -> Optional[float]:
    """Share of cells in type folders whose folder ON/OFF contradicts the STA."""
    n = bad = 0
    classes = folder_classes(main_window)
    for cid, s in st.by_cell.items():
        here = classes.get(cid)
        if here is None or s.polarity == 0:
            continue
        n += 1
        bad += (here.startswith("ON")) != (s.polarity > 0)
    return bad / n if n >= 10 else None


def show_summary(main_window, st: RunSuggestions):
    vals = list(st.by_cell.values())
    confident = sum(s.confident for s in vals)
    novel = sum(s.novel for s in vals)
    review = len(review_queue(main_window))
    classes = folder_classes(main_window)
    disagree = sum(1 for cid, s in st.by_cell.items()
                   if s.confident and classes.get(cid) not in (None, s.best[0]))
    text = (f"Suggestions for {len(vals)} cells, learned from {st.n_train} cells the lab "
            f"classified.\n\n"
            f"  Confident (≥ {REVIEW_BELOW:.0%}): {confident}\n"
            f"  To review (Ctrl+J, least confident first): {review}\n"
            f"  No suggestion — unlike any labelled cell: {novel}\n"
            f"  Classified cells that a confident suggestion disagrees with: {disagree}\n")
    if st.swapped_polarity is not None and st.swapped_polarity > 0.5:
        text += (f"\nWARNING: {st.swapped_polarity:.0%} of the classified cells have the "
                 f"opposite ON/OFF of their STA. The ON and OFF labels of this run may be "
                 f"swapped.\n")
    text += ("\nThe suggester knows 5 types (ON/OFF brisk sustained, ON/OFF brisk transient, "
             "OFF transient). Suggestions are a starting point: check them.")
    panel = getattr(main_window, "types_panel", None)
    if panel is not None:
        panel.suggest_summary.setText(
            f"{len(vals)} cells: {confident} confident · {review} to review (Ctrl+J) · "
            f"{novel} no suggestion · {disagree} disagree with their folder")
        panel.suggest_summary.setToolTip(text)
        panel.accept_btn.setEnabled(confident > 0)
    QMessageBox.information(main_window, "Suggested classes", text)
