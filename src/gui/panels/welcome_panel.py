"""What a new user sees before a run is open (PLAN.md Q49).

Encore used to open on a disabled, empty window. This page says what the
tool is for, how to start, and lists recent runs as one-click buttons (a
run is never reopened on its own — PLAN.md standing decision 1).
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

STEPS = (
    ("Open a run",
     "Pick a Kilosort folder (ksfiles). Vision files next to it (.sta, .ei, .params) "
     "and stimulus analyses are found by themselves."),
    ("Look at a cell",
     "Pick a cell in the list on the left, or step with ↑ / ↓. Each tab shows one view "
     "of it: STA, EI, waveforms, chirp, gratings. Ctrl+P puts its population beside it."),
    ("Classify",
     "Put cells in groups (Ctrl+G new group, Ctrl+M move). The Types tab shows every "
     "class at once and suggests classes from the lab's past work. Ctrl+S saves the "
     "classes into the Vision .params."),
)


def _run_label(path: str) -> str:
    parts = Path(path).parts
    tail = [p for p in parts[-4:] if p != "ksfiles"]
    return " / ".join(tail[-3:])


class WelcomePanel(QWidget):
    def __init__(self, main_window, recent=()):
        super().__init__()
        self.main_window = main_window
        outer = QVBoxLayout(self)
        outer.setContentsMargins(48, 40, 48, 40)
        outer.addStretch(1)

        body = QVBoxLayout()
        body.setSpacing(14)
        self.title = QLabel("Encore")
        body.addWidget(self.title)
        self.lead = QLabel("Inspect spike-sorted retina recordings and classify ganglion cells.")
        self.lead.setWordWrap(True)
        body.addWidget(self.lead)

        open_btn = QPushButton("Open a run…   (Ctrl+O)")
        open_btn.setObjectName("primaryButton")
        open_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        open_btn.clicked.connect(lambda: main_window.load_directory())
        self.open_btn = open_btn
        body.addWidget(open_btn)

        self.recent_buttons = []
        self.heads = []
        if recent:
            head = QLabel("RECENT RUNS")
            self.heads.append(head)
            body.addWidget(head)
            for path in recent:
                b = QPushButton(_run_label(path))
                b.setToolTip(path)
                b.setFlat(True)
                b.setCursor(Qt.CursorShape.PointingHandCursor)
                b.clicked.connect(lambda _c=False, p=path: main_window.load_directory(p))
                body.addWidget(b)
                self.recent_buttons.append(b)

        self.rule = QFrame()
        self.rule.setFixedHeight(1)
        body.addWidget(self.rule)
        steps = QHBoxLayout()
        steps.setSpacing(28)
        self.step_heads, self.muted = [], []
        for i, (name, text) in enumerate(STEPS, start=1):
            col = QVBoxLayout()
            h = QLabel(f"{i}  {name}")
            self.step_heads.append(h)
            col.addWidget(h)
            t = QLabel(text)
            t.setWordWrap(True)
            self.muted.append(t)
            col.addWidget(t)
            col.addStretch(1)
            steps.addLayout(col, 1)
        body.addLayout(steps)
        tip = QLabel("F1 (or ?) shows every keyboard shortcut. "
                     "File ▸ About Encore shows the version to put in a bug report.")
        tip.setWordWrap(True)
        self.muted.append(tip)
        body.addWidget(tip)

        wrap = QHBoxLayout()
        wrap.addStretch(1)
        holder = QWidget()
        holder.setLayout(body)
        holder.setMaximumWidth(860)
        wrap.addWidget(holder, 4)
        wrap.addStretch(1)
        outer.addLayout(wrap)
        outer.addStretch(2)
        self.restyle_plots(main_window.get_current_colors())

    def restyle_plots(self, colors):
        """Theme colours (the app stylesheet would otherwise flatten this page)."""
        from ..theme import resolve_theme_colors
        c = resolve_theme_colors(colors)
        self.title.setStyleSheet(f"font-size: 30px; font-weight: 700; color: {c['text_primary']};")
        self.lead.setStyleSheet(f"font-size: 15px; color: {c['text_secondary']};")
        self.open_btn.setStyleSheet(
            f"QPushButton {{ background: {c['accent']}; color: {c['accent_text']}; border: none;"
            f" border-radius: 4px; padding: 8px 18px; font-size: 13px; font-weight: 600; }}"
            f"QPushButton:hover {{ background: {c.get('accent_hover', c['accent'])}; }}")
        for h in self.heads:
            h.setStyleSheet(f"font-size: 10px; letter-spacing: 0.08em; color: {c['text_tertiary']};")
        # The plot blue: lighter than the chrome accent, so it reads on dark paper.
        link = c.get("plot_highlight", c["accent"])
        for b in self.recent_buttons:
            b.setStyleSheet(
                f"QPushButton {{ text-align: left; padding: 3px 0; border: none; background: transparent;"
                f" color: {link}; font-size: 13px; }}"
                f"QPushButton:hover {{ text-decoration: underline; }}")
        self.rule.setStyleSheet(f"background: {c['border_subtle']};")
        for h in self.step_heads:
            h.setStyleSheet(f"font-weight: 700; font-size: 13px; color: {c['text_primary']};")
        for t in self.muted:
            t.setStyleSheet(f"color: {c['text_secondary']}; font-size: 12px;")
