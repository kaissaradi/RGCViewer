from unittest.mock import MagicMock, patch

import numpy as np


def _seed_population_caches():
    import src.gui.panels.population_panel as pp

    pp._group_timecourse_cache[frozenset([1, 2])] = {'arr': np.zeros((2, 40))}
    pp._group_acg_cache[frozenset([3, 4])] = {'arr': np.zeros((2, 50))}
    pp._rf_background_cache[999] = [(0, 0, 10, 8, 0, 'white', 'none', 0.5)]
    pp._rf_background_cache_order.append(999)
    return pp


def _assert_population_caches_empty(pp):
    assert len(pp._group_timecourse_cache) == 0
    assert len(pp._group_acg_cache) == 0
    assert len(pp._rf_background_cache) == 0
    assert len(pp._rf_background_cache_order) == 0


def test_invalidate_population_caches_clears_all_structures():
    from src.gui.panels.population_panel import invalidate_population_caches

    pp = _seed_population_caches()

    invalidate_population_caches()

    _assert_population_caches_empty(pp)


def test_refinement_results_invalidate_population_caches():
    from src.gui import callbacks

    pp = _seed_population_caches()
    main_window = MagicMock()
    main_window.refine_thread = MagicMock()

    with patch('src.gui.callbacks.populate_tree_view') as mock_populate:
        callbacks.handle_refinement_results(main_window, 7, [70, 71])

    main_window.data_manager.update_after_refinement.assert_called_once_with(7, [70, 71])
    mock_populate.assert_called_once_with(main_window)
    _assert_population_caches_empty(pp)


def test_partial_vision_loaded_invalidate_population_caches():
    from src.gui import callbacks

    pp = _seed_population_caches()
    main_window = MagicMock()
    main_window.vision_load_thread = None
    main_window.vision_load_worker = None
    main_window.data_manager.vision_stas = None
    main_window.data_manager.vision_eis = None

    with patch('src.gui.callbacks.QMessageBox.warning'):
        callbacks._on_vision_loaded(main_window, success=False, message="partial", is_partial=True)

    _assert_population_caches_empty(pp)
