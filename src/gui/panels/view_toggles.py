"""A checkbox row deciding which population views are on screen.

The RF mosaic and the per-feature trace overlays compete for the same narrow
side panel, and which of them is worth the space depends entirely on what you
are doing: the mosaic answers "does this tile?", the temporal STA answers "do
these share a timecourse?", and the autocorrelogram mostly does not answer
anything you are asking while curating — which is why it starts closed.

Visibility is the AND of two things, and keeping them separate matters:

* **available** — this run actually has data for that block. Set by the panel.
  An unavailable view is greyed out with a reason rather than silently missing,
  so "the chirp overlay is not there" is distinguishable from "this prep has no
  chirp analysis", which is a question that otherwise costs a trip to the files.
* **checked** — the user wants to see it.

Unchecking is therefore never confused with having no data, and a view that the
user closed comes back with its data intact when they reopen it.
"""

from __future__ import annotations

import logging

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QWidget

logger = logging.getLogger(__name__)

__all__ = ["PopulationViewToggles"]


class PopulationViewToggles(QWidget):
    """A compact row of checkboxes owning the visibility of population views."""

    #: Emitted as ``(key, is_open)`` whenever a view opens or closes.
    view_toggled = Signal(str, bool)

    def __init__(self, parent=None, colors: dict = None, heading: str = "Show:"):
        super().__init__(parent)
        self._entries: dict = {}
        colors = colors or {}
        muted = colors.get("text_muted") or colors.get("text_secondary") or "#6b7280"

        row = QHBoxLayout(self)
        row.setContentsMargins(2, 0, 2, 0)
        row.setSpacing(8)

        caption = QLabel(heading)
        caption.setStyleSheet(f"color: {muted}; font-size: 11px;")
        row.addWidget(caption)
        self._row = row
        self._row.addStretch()

    # ── construction ─────────────────────────────────────────────────────────

    def add_view(
        self,
        key: str,
        label: str,
        widget: QWidget,
        checked: bool = True,
        tooltip: str = "",
    ):
        """Register *widget* under *key* with its own checkbox."""
        chk = QCheckBox(label)
        chk.setChecked(bool(checked))
        chk.setToolTip(tooltip)
        chk.toggled.connect(lambda _on, k=key: self._on_toggled(k))

        # Before the trailing stretch, so the checkboxes stay left-aligned.
        self._row.insertWidget(self._row.count() - 1, chk)

        self._entries[key] = {
            "widget": widget,
            "checkbox": chk,
            "available": True,
            "tooltip": tooltip,
        }
        self._apply(key)

    # ── state ────────────────────────────────────────────────────────────────

    def set_available(self, key: str, available: bool, reason: str = ""):
        """Say whether this run has data for *key*.

        An unavailable view is greyed out and hidden but keeps the user's
        checkbox state, so it reappears as they left it on a run that does have
        the data.
        """
        entry = self._entries.get(key)
        if entry is None:
            return
        entry["available"] = bool(available)
        entry["checkbox"].setEnabled(bool(available))
        entry["checkbox"].setToolTip(
            reason if (not available and reason) else entry["tooltip"]
        )
        self._apply(key)

    def is_open(self, key: str) -> bool:
        entry = self._entries.get(key)
        if entry is None:
            return False
        return bool(entry["available"] and entry["checkbox"].isChecked())

    def open_keys(self) -> set:
        return {k for k in self._entries if self.is_open(k)}

    def widget_for(self, key: str):
        entry = self._entries.get(key)
        return entry["widget"] if entry else None

    # ── internals ────────────────────────────────────────────────────────────

    def _apply(self, key: str):
        entry = self._entries.get(key)
        if entry is None:
            return
        widget = entry["widget"]
        if widget is not None:
            widget.setVisible(self.is_open(key))

    def _on_toggled(self, key: str):
        self._apply(key)
        self.view_toggled.emit(key, self.is_open(key))
