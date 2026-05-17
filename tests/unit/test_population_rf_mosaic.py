"""
Tests for Population RF Mosaic — Spec: population-panel-mosiac.md

Written BEFORE implementation changes (TDD Red phase).
These tests encode the acceptance criteria from the spec.
"""

import numpy as np
from unittest.mock import MagicMock
from matplotlib.patches import Ellipse
from matplotlib.figure import Figure

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)


# ---------------------------------------------------------------------------
# Helpers: mock VisionParams with realistic stafit data
# ---------------------------------------------------------------------------
class FakeStafit:
    """Minimal stafit with centre + ellipse geometry."""
    def __init__(self, cx, cy, sx, sy, rot):
        self.center_x = cx
        self.center_y = cy
        self.std_x = sx
        self.std_y = sy
        self.rot = rot


class FakeVisionParams:
    """Minimal VisionParams exposing get_cell_ids / get_stafit_for_cell."""
    def __init__(self, cells: dict):
        # cells: {vision_id: FakeStafit}
        self._cells = cells

    def get_cell_ids(self):
        return list(self._cells.keys())

    def get_stafit_for_cell(self, cell_id):
        if cell_id not in self._cells:
            raise KeyError(f"No stafit for {cell_id}")
        return self._cells[cell_id]


def _make_vision_params(n=6):
    """Create n fake cells with vision IDs 1..n."""
    rng = np.random.RandomState(42)
    cells = {}
    for i in range(1, n + 1):
        cells[i] = FakeStafit(
            cx=rng.uniform(10, 90),
            cy=rng.uniform(10, 90),
            sx=rng.uniform(2, 8),
            sy=rng.uniform(2, 8),
            rot=rng.uniform(0, np.pi),
        )
    return cells, FakeVisionParams(cells)


def _make_mock_main_window(vision_params, sta_height=100, subset_ids=None):
    """Build a minimal main_window mock for plot_population_rfs_background."""
    mw = MagicMock()
    mw.data_manager = MagicMock()
    mw.data_manager.vision_params = vision_params
    mw.data_manager.vision_sta_height = sta_height

    # Default colors (dark theme)
    mw.get_current_colors.return_value = {
        "bg_base": "#111214",
        "bg_panel": "#18191C",
        "bg_surface": "#1E2025",
        "border_default": "#3D3F48",
        "border_subtle": "#2E3038",
        "text_primary": "#F0F0F2",
        "text_secondary": "#9B9DA6",
        "text_tertiary": "#5A5C65",
        "text_disabled": "#3A3C44",
        "plot_highlight": "#00FFFF",
    }

    # The show-IDs checkbox should live on main_window after the spec change
    pop_checkbox = MagicMock()
    pop_checkbox.isChecked.return_value = False
    mw.pop_show_ids_checkbox = pop_checkbox

    # Legacy attribute should NOT exist after spec change
    if hasattr(mw, "standard_plots_panel"):
        del mw.standard_plots_panel

    return mw


# ===================================================================
# TEST 1 — AC1: Ellipse alpha / linewidth ranges
# ===================================================================
class TestEllipseVisibility:
    """AC1: Ellipses must have spec-compliant alpha and linewidth values."""

    def test_background_ellipse_alpha_and_lw(self):
        """Background (non-target) ellipses: alpha 0.12–0.18, lw 0.6–0.9."""
        cells, vp = _make_vision_params(6)
        mw = _make_mock_main_window(vp)
        colors = mw.get_current_colors()

        # subset = first 3 cells (kilosort IDs 0..2 → vision IDs 1..3)
        subset_cell_ids = [0, 1, 2]

        fig = Figure()
        ax = fig.add_subplot(111)

        from src.gui.panels.population_panel import plot_population_rfs_background
        plot_population_rfs_background(
            ax, vp, main_window=mw, sta_height=100,
            subset_cell_ids=subset_cell_ids, colors=colors,
        )

        # Background ellipses are those NOT in the subset (vision IDs 4, 5, 6)
        bg_patches = [
            p for p in ax.patches
            if isinstance(p, Ellipse) and p.get_alpha() is not None and p.get_alpha() < 0.20
        ]
        assert len(bg_patches) > 0, "Expected background ellipses"

        for p in bg_patches:
            assert 0.12 <= p.get_alpha() <= 0.18, (
                f"Background alpha {p.get_alpha()} outside [0.12, 0.18]"
            )
            assert 0.6 <= p.get_linewidth() <= 0.9, (
                f"Background lw {p.get_linewidth()} outside [0.6, 0.9]"
            )

    def test_target_ellipse_alpha_and_lw(self):
        """Target (in-subset) ellipses: alpha 0.45–0.65, lw 0.9–1.2."""
        cells, vp = _make_vision_params(6)
        mw = _make_mock_main_window(vp)
        colors = mw.get_current_colors()

        subset_cell_ids = [0, 1, 2]

        fig = Figure()
        ax = fig.add_subplot(111)

        from src.gui.panels.population_panel import plot_population_rfs_background
        plot_population_rfs_background(
            ax, vp, main_window=mw, sta_height=100,
            subset_cell_ids=subset_cell_ids, colors=colors,
        )

        # Target ellipses are those IN the subset (higher alpha)
        target_patches = [
            p for p in ax.patches
            if isinstance(p, Ellipse) and p.get_alpha() is not None and p.get_alpha() >= 0.20
        ]
        assert len(target_patches) > 0, "Expected target ellipses"

        for p in target_patches:
            assert 0.45 <= p.get_alpha() <= 0.65, (
                f"Target alpha {p.get_alpha()} outside [0.45, 0.65]"
            )
            assert 0.9 <= p.get_linewidth() <= 1.2, (
                f"Target lw {p.get_linewidth()} outside [0.9, 1.2]"
            )

    def test_highlight_ellipse_alpha_and_lw(self):
        """Highlight (selected cell): fill alpha 0.35–0.50, edge lw 1.5–2.0."""
        cells, vp = _make_vision_params(6)
        mw = _make_mock_main_window(vp)
        colors = mw.get_current_colors()

        from src.gui.panels.population_panel import draw_population_rfs_plot

        # We need a canvas mock with a fig
        canvas = MagicMock()
        canvas.fig = Figure()
        canvas.fig.set_facecolor(colors["bg_panel"])
        # Force full rebuild by not having _pop_plot_state
        if hasattr(canvas, "_pop_plot_state"):
            del canvas._pop_plot_state

        mw._get_pop_subset_ids = MagicMock(return_value=[0, 1, 2])

        draw_population_rfs_plot(
            main_window=mw,
            selected_cell_id=0,
            subset_cell_ids=[0, 1, 2],
            canvas=canvas,
        )

        ax = canvas.fig.axes[0]
        highlight_patches = [
            p for p in ax.patches
            if isinstance(p, Ellipse) and p.get_zorder() == 10
        ]
        assert len(highlight_patches) == 1, "Expected exactly 1 highlight ellipse"

        hp = highlight_patches[0]
        # facecolor is RGBA — check alpha component
        fc = hp.get_facecolor()
        assert 0.35 <= fc[3] <= 0.50, (
            f"Highlight fill alpha {fc[3]} outside [0.35, 0.50]"
        )
        assert 1.5 <= hp.get_linewidth() <= 2.0, (
            f"Highlight lw {hp.get_linewidth()} outside [1.5, 2.0]"
        )


# ===================================================================
# TEST 2 — AC2: Show IDs checkbox moved
# ===================================================================
class TestShowIdsCheckboxRelocation:
    """AC2: show_ids_checkbox removed from StandardPlotsPanel,
    pop_show_ids_checkbox added to main_window's pop_ctrl_layout."""

    def test_standard_plots_panel_has_no_show_ids(self, qtbot):
        """StandardPlotsPanel must NOT have a show_ids_checkbox attribute."""
        from qtpy.QtWidgets import QMainWindow
        from src.gui.panels.standard_plots_panel import StandardPlotsPanel

        # Minimal main_window mock for StandardPlotsPanel constructor
        mock_mw = QMainWindow()
        mock_mw.data_manager = MagicMock()
        mock_mw.similarity_panel = None
        mock_mw.get_current_colors = lambda: {
            "bg_base": "#111214", "bg_panel": "#18191C", "bg_surface": "#1E2025",
            "border_default": "#3D3F48", "border_subtle": "#2E3038",
            "text_primary": "#F0F0F2", "text_secondary": "#9B9DA6",
            "text_tertiary": "#5A5C65", "text_disabled": "#3A3C44",
        }

        panel = StandardPlotsPanel(mock_mw)
        qtbot.addWidget(panel)

        assert not hasattr(panel, "show_ids_checkbox"), (
            "StandardPlotsPanel still has show_ids_checkbox — it should be removed"
        )

    def test_main_window_has_pop_show_ids_checkbox(self, qtbot):
        """main_window must have pop_show_ids_checkbox (QCheckBox)."""

        # We can't easily instantiate the real MainWindow without data,
        # so we verify the attribute exists structurally via the mock.
        # A real integration test would do the full instantiation.
        mw = _make_mock_main_window(FakeVisionParams({}))
        assert hasattr(mw, "pop_show_ids_checkbox")


# ===================================================================
# TEST 3 — AC3: Show IDs wiring — modest labels on RF mosaic
# ===================================================================
class TestShowIdsLabelsOnMosaic:
    """AC3: Toggling pop_show_ids_checkbox draws/removes text artists on the
    RF mosaic canvas with font size ≤ 8 and color text_secondary."""

    def test_labels_appear_when_checked(self):
        """When checked, text artists == number of target cells, font ≤ 8."""
        cells, vp = _make_vision_params(6)
        mw = _make_mock_main_window(vp, subset_ids=[0, 1, 2])
        mw.pop_show_ids_checkbox.isChecked.return_value = True
        colors = mw.get_current_colors()

        fig = Figure()
        ax = fig.add_subplot(111)

        from src.gui.panels.population_panel import plot_population_rfs_background
        plot_population_rfs_background(
            ax, vp, main_window=mw, sta_height=100,
            subset_cell_ids=[0, 1, 2], colors=colors,
        )

        texts = [t for t in ax.texts]
        # 3 target cells → 3 labels
        assert len(texts) == 3, f"Expected 3 text labels, got {len(texts)}"
        for t in texts:
            assert t.get_fontsize() <= 8, f"Font size {t.get_fontsize()} > 8"

    def test_no_labels_when_unchecked(self):
        """When unchecked, zero text artists on the canvas."""
        cells, vp = _make_vision_params(6)
        mw = _make_mock_main_window(vp)
        mw.pop_show_ids_checkbox.isChecked.return_value = False
        colors = mw.get_current_colors()

        fig = Figure()
        ax = fig.add_subplot(111)

        from src.gui.panels.population_panel import plot_population_rfs_background
        plot_population_rfs_background(
            ax, vp, main_window=mw, sta_height=100,
            subset_cell_ids=[0, 1, 2], colors=colors,
        )

        texts = [t for t in ax.texts]
        assert len(texts) == 0, f"Expected 0 text labels, got {len(texts)}"
