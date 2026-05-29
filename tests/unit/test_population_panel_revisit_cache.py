from unittest.mock import MagicMock

import numpy as np
from matplotlib.figure import Figure


def _make_canvas():
    canvas = MagicMock()
    canvas.fig = Figure()
    return canvas


def _make_group_main_window(subset_ids):
    t_len = 40
    timecourse_data = {
        cid: np.sin(np.linspace(0, 2 * np.pi, t_len)).astype(np.float32)
        for cid in subset_ids
    }
    t_axis = np.linspace(-50, 50, 50)
    acg_data = {
        cid: (t_axis, np.cos(np.linspace(0, np.pi, 50)).astype(np.float32))
        for cid in subset_ids
    }

    dm = MagicMock()
    dm.get_cell_physics.side_effect = lambda cid: {'timecourse': timecourse_data.get(cid)}
    dm.get_acg_data.side_effect = lambda cid: acg_data.get(cid, (None, None))

    mw = MagicMock()
    mw.data_manager = dm
    mw.pop_timecourse_canvas = _make_canvas()
    mw.pop_acg_canvas = _make_canvas()
    mw.pop_timecourse_summary = MagicMock()
    mw.pop_acg_summary = MagicMock()
    mw._get_pop_subset_ids.return_value = subset_ids
    mw.get_current_colors.return_value = {
        'bg_panel': '#18191C',
        'text_primary': '#F0F0F2',
        'text_secondary': '#9B9DA6',
        'accent': '#2E6DD4',
        'plot_mean': '#7EB8F7',
        'plot_peak': '#FF6B6B',
        'plot_acg': '#6BCB77',
        'plot_compare': '#FFD93D',
        'border_subtle': '#2E3038',
    }
    return mw


def _make_rf_main_window(subset_ids, all_vision_ids):
    def _get_stafit(vision_id):
        stafit = MagicMock()
        stafit.center_x = float(vision_id) * 3.0
        stafit.center_y = float(vision_id) * 2.0
        stafit.std_x = 5.0
        stafit.std_y = 4.0
        stafit.rot = 0.0
        return stafit

    vision_params = MagicMock()
    vision_params.get_cell_ids.return_value = all_vision_ids
    vision_params.get_stafit_for_cell.side_effect = _get_stafit

    mw = MagicMock()
    mw.data_manager.vision_params = vision_params
    mw.data_manager.vision_sta_height = 100
    mw.get_current_colors.return_value = {
        'bg_panel': '#18191C',
        'text_primary': '#F0F0F2',
        'text_secondary': '#9B9DA6',
        'plot_highlight': '#FFD93D',
        'border_subtle': '#2E3038',
    }
    mw.pop_show_ids_checkbox.isChecked.return_value = False
    mw.pop_mosaic_canvas = _make_canvas()
    mw._get_pop_subset_ids.return_value = subset_ids
    return mw


def test_group_timecourse_revisit_uses_cached_arrays():
    from src.gui.panels.population_panel import (
        draw_population_timecourse_panel,
        invalidate_population_caches,
    )

    invalidate_population_caches()
    subset_ids = [0, 1, 2]
    mw = _make_group_main_window(subset_ids)

    draw_population_timecourse_panel(mw, subset_ids=subset_ids)
    assert mw.data_manager.get_cell_physics.call_count == len(subset_ids)

    mw.data_manager.get_cell_physics.reset_mock()
    draw_population_timecourse_panel(mw, subset_ids=subset_ids)

    mw.data_manager.get_cell_physics.assert_not_called()


def test_group_acg_revisit_uses_cached_arrays():
    from src.gui.panels.population_panel import draw_population_acg_panel, invalidate_population_caches

    invalidate_population_caches()
    subset_ids = [0, 1, 2]
    mw = _make_group_main_window(subset_ids)

    draw_population_acg_panel(mw, subset_ids=subset_ids)
    assert mw.data_manager.get_acg_data.call_count == len(subset_ids)

    mw.data_manager.get_acg_data.reset_mock()
    draw_population_acg_panel(mw, subset_ids=subset_ids)

    mw.data_manager.get_acg_data.assert_not_called()


def test_rf_background_a_b_a_revisit_uses_cached_geometry():
    from src.gui.panels.population_panel import draw_population_rfs_plot, invalidate_population_caches

    invalidate_population_caches()
    all_vision_ids = list(range(1, 21))
    subset_a = list(range(0, 5))
    subset_b = list(range(5, 10))
    mw = _make_rf_main_window(subset_a, all_vision_ids)

    draw_population_rfs_plot(main_window=mw, subset_cell_ids=subset_a, canvas=mw.pop_mosaic_canvas)
    assert mw.data_manager.vision_params.get_stafit_for_cell.call_count > 0

    draw_population_rfs_plot(main_window=mw, subset_cell_ids=subset_b, canvas=mw.pop_mosaic_canvas)

    mw.data_manager.vision_params.get_stafit_for_cell.reset_mock()
    draw_population_rfs_plot(main_window=mw, subset_cell_ids=subset_a, canvas=mw.pop_mosaic_canvas)

    mw.data_manager.vision_params.get_stafit_for_cell.assert_not_called()
