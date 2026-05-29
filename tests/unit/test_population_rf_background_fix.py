"""
Tests for Receptive Field background rendering and cache optimization.
Encoding acceptance criteria from fix/population-rf-background specification.
"""

import numpy as np
from unittest.mock import MagicMock, patch
import pytest

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib.figure import Figure
from matplotlib.collections import EllipseCollection
from matplotlib.patches import Ellipse

from src.gui.panels.population_panel import (
    plot_population_rfs_background,
    _snapshot_rf_background,
    _draw_cached_rf_background,
    invalidate_population_caches,
)


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
    """Build a minimal main_window mock."""
    mw = MagicMock()
    mw.data_manager = MagicMock()
    mw.data_manager.vision_params = vision_params
    mw.data_manager.vision_sta_height = sta_height
    mw.data_manager.is_vision_only = False

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

    pop_checkbox = MagicMock()
    pop_checkbox.isChecked.return_value = False
    mw.pop_show_ids_checkbox = pop_checkbox
    return mw


def test_axis_limits_contain_all_ellipses():
    """AC1: All ellipse centers fall within the x-limits and y-limits with margin."""
    invalidate_population_caches()
    cells, vp = _make_vision_params(6)
    mw = _make_mock_main_window(vp)
    colors = mw.get_current_colors()

    fig = Figure()
    ax = fig.add_subplot(111)

    plot_population_rfs_background(
        ax, vp, main_window=mw, sta_height=100,
        subset_cell_ids=[0, 1, 2], colors=colors,
    )

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    for cell_id, stafit in cells.items():
        adj_y = 100 - stafit.center_y
        assert xlim[0] <= stafit.center_x <= xlim[1], f"Cell {cell_id} center_x outside xlim"
        assert ylim[0] <= adj_y <= ylim[1], f"Cell {cell_id} center_y outside ylim"


def test_y_flip_applied_to_all_ellipses():
    """AC2: Plotted y-coordinate is sta_height - center_y."""
    invalidate_population_caches()
    cells, vp = _make_vision_params(6)
    mw = _make_mock_main_window(vp)
    colors = mw.get_current_colors()

    fig = Figure()
    ax = fig.add_subplot(111)

    plot_population_rfs_background(
        ax, vp, main_window=mw, sta_height=100,
        subset_cell_ids=[0, 1, 2], colors=colors,
    )

    assert len(ax.collections) >= 1
    all_offsets = []
    for coll in ax.collections:
        if isinstance(coll, EllipseCollection):
            all_offsets.extend(coll.get_offsets())

    assert len(all_offsets) == 6

    for cell_id, stafit in cells.items():
        expected_x = stafit.center_x
        expected_y = 100 - stafit.center_y
        found = False
        for ox, oy in all_offsets:
            if np.isclose(ox, expected_x) and np.isclose(oy, expected_y):
                found = True
                break
        assert found, f"Cell {cell_id} center {(expected_x, expected_y)} not found in plotted offsets"


def test_y_flip_matches_highlight_and_background():
    """AC3: Background and highlight patches share same center coords (both y-flipped)."""
    invalidate_population_caches()
    cells, vp = _make_vision_params(6)
    mw = _make_mock_main_window(vp)
    colors = mw.get_current_colors()

    selected_cell_id = 0
    vision_id = selected_cell_id + 1
    stafit = vp.get_stafit_for_cell(vision_id)

    fig = Figure()
    ax = fig.add_subplot(111)

    plot_population_rfs_background(
        ax, vp, main_window=mw, sta_height=100,
        subset_cell_ids=[selected_cell_id], colors=colors,
    )

    target_coll = None
    for coll in ax.collections:
        if isinstance(coll, EllipseCollection) and coll.get_zorder() == 2:
            target_coll = coll
            break

    assert target_coll is not None, "Target collection not found"
    target_offsets = target_coll.get_offsets()
    assert len(target_offsets) == 1
    bg_x, bg_y = target_offsets[0]

    highlight_patch = Ellipse(
        xy=(0, 0), width=1, height=1, angle=0,
        edgecolor=colors['plot_highlight'], facecolor=(0, 1, 1, 0.42),
        lw=1.75, zorder=10, visible=False
    )
    from src.gui.panels.population_panel import _update_highlight_patch
    _update_highlight_patch(highlight_patch, vp, selected_cell_id, 100)

    hl_x, hl_y = highlight_patch.center

    assert np.isclose(bg_x, hl_x)
    assert np.isclose(bg_y, hl_y)
    assert np.isclose(bg_y, 100 - stafit.center_y)


def test_snapshot_captures_nonempty_collections():
    """AC4: _snapshot_rf_background captures non-empty EllipseCollections."""
    invalidate_population_caches()
    cells, vp = _make_vision_params(6)
    mw = _make_mock_main_window(vp)
    colors = mw.get_current_colors()

    fig = Figure()
    ax = fig.add_subplot(111)

    plot_population_rfs_background(
        ax, vp, main_window=mw, sta_height=100,
        subset_cell_ids=[0, 1, 2], colors=colors,
    )

    snapshot = _snapshot_rf_background(ax, colors, show_ids=False)

    assert 'collections' in snapshot
    assert len(snapshot['collections']) >= 1

    for coll_data in snapshot['collections']:
        assert len(coll_data['offsets']) > 0
        assert len(coll_data['widths']) > 0
        assert len(coll_data['heights']) > 0
        assert len(coll_data['angles']) > 0


def test_cache_replay_produces_identical_geometry():
    """AC5: Cache replay reproduces identical collections and offsets."""
    invalidate_population_caches()
    cells, vp = _make_vision_params(6)
    mw = _make_mock_main_window(vp)
    colors = mw.get_current_colors()

    fig1 = Figure()
    ax1 = fig1.add_subplot(111)

    plot_population_rfs_background(
        ax1, vp, main_window=mw, sta_height=100,
        subset_cell_ids=[0, 1, 2], colors=colors,
    )

    snapshot = _snapshot_rf_background(ax1, colors, show_ids=False)

    fig2 = Figure()
    ax2 = fig2.add_subplot(111)

    _draw_cached_rf_background(ax2, snapshot, colors)

    assert len(ax1.collections) == len(ax2.collections)
    for c1, c2 in zip(ax1.collections, ax2.collections):
        assert isinstance(c1, EllipseCollection)
        assert isinstance(c2, EllipseCollection)
        np.testing.assert_allclose(c1.get_offsets(), c2.get_offsets(), atol=1e-6)
        np.testing.assert_allclose(c1._widths, c2._widths, atol=1e-6)
        np.testing.assert_allclose(c1._heights, c2._heights, atol=1e-6)
        np.testing.assert_allclose(c1._angles, c2._angles, atol=1e-6)
        assert c1.get_alpha() == c2.get_alpha()
        assert c1.get_zorder() == c2.get_zorder()


def test_axes_style_applied_title_and_aspect():
    """AC6: _apply_rf_axes_style is called (title, aspect equal)."""
    invalidate_population_caches()
    cells, vp = _make_vision_params(6)
    mw = _make_mock_main_window(vp)
    colors = mw.get_current_colors()

    fig = Figure()
    ax = fig.add_subplot(111)

    plot_population_rfs_background(
        ax, vp, main_window=mw, sta_height=100,
        subset_cell_ids=[0, 1, 2], colors=colors,
    )

    assert ax.get_title() == "Population Receptive Fields (n=3)"
    assert ax.get_aspect() == "equal" or ax.get_aspect() == 1.0


def test_ellipse_alpha_lw_ranges_preserved():
    """AC7: Spec compliance for alpha and lw on background and target collection."""
    invalidate_population_caches()
    cells, vp = _make_vision_params(6)
    mw = _make_mock_main_window(vp)
    colors = mw.get_current_colors()

    fig = Figure()
    ax = fig.add_subplot(111)

    plot_population_rfs_background(
        ax, vp, main_window=mw, sta_height=100,
        subset_cell_ids=[0, 1, 2], colors=colors,
    )

    bg_coll = None
    target_coll = None
    for coll in ax.collections:
        if not isinstance(coll, EllipseCollection):
            continue
        if coll.get_zorder() == 1:
            bg_coll = coll
        elif coll.get_zorder() == 2:
            target_coll = coll

    assert bg_coll is not None, "Background collection not found"
    assert target_coll is not None, "Target collection not found"

    assert 0.12 <= bg_coll.get_alpha() <= 0.18
    lw = bg_coll.get_linewidth()[0] if hasattr(bg_coll.get_linewidth(), '__len__') else bg_coll.get_linewidth()
    assert 0.6 <= lw <= 0.9

    assert 0.45 <= target_coll.get_alpha() <= 0.65
    lw_target = target_coll.get_linewidth()[0] if hasattr(target_coll.get_linewidth(), '__len__') else target_coll.get_linewidth()
    assert 0.9 <= lw_target <= 1.2


def test_dead_code_plot_population_rfs_removed():
    """AC8: Dead code plot_population_rfs does not exist anymore in population_panel."""
    with pytest.raises(ImportError):
        from src.gui.panels.population_panel import plot_population_rfs


def test_redraw_population_panels_uses_passed_subset():
    """AC9: redraw_population_panels uses passed subset when supplied, avoiding _get_pop_subset_ids."""
    from src.gui.callbacks import redraw_population_panels

    main_window = MagicMock()
    main_window._get_pop_subset_ids = MagicMock(return_value=[0, 1])

    with patch('src.gui.callbacks.draw_population_timecourse_panel') as mock_tc, \
         patch('src.gui.panels.population_panel.draw_population_acg_panel') as mock_acg:

        # Case 1: called with subset=None (or omitted) -> calls _get_pop_subset_ids
        redraw_population_panels(main_window)
        main_window._get_pop_subset_ids.assert_called_once()
        mock_tc.assert_called_with(main_window, subset_ids=[0, 1])
        mock_acg.assert_called_with(main_window, subset_ids=[0, 1])

        main_window._get_pop_subset_ids.reset_mock()
        mock_tc.reset_mock()
        mock_acg.reset_mock()

        # Case 2: called with explicit subset -> skips _get_pop_subset_ids
        redraw_population_panels(main_window, subset=[3, 4])
        main_window._get_pop_subset_ids.assert_not_called()
        mock_tc.assert_called_with(main_window, subset_ids=[3, 4])
        mock_acg.assert_called_with(main_window, subset_ids=[3, 4])
