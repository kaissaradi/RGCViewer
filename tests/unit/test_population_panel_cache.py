"""
Unit tests for population panel group-level caches (§5.2 and §5.3).

Tests:
- test_group_timecourse_cache_hit          — get_cell_physics called only on first visit
- test_group_acg_cache_hit                 — get_acg_data called only on first visit
- test_rf_background_cache_hit_on_revisit  — get_stafit_for_cell NOT called on A→B→A revisit
- test_invalidate_clears_all_caches        — invalidate_population_caches() empties all dicts
- test_lazy_sta_cache_size_scales_with_dataset — _max_cache respects MAX_STA_CACHE_CELLS

All tests are pure unit tests: no QApplication, no disk I/O, no real DataManager.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_main_window(subset_ids, timecourse_data=None, acg_data=None):
    """
    Build a minimal mock main_window that satisfies the draw function contracts.

    subset_ids:      list[int]  — the cluster IDs the panel should iterate over
    timecourse_data: dict mapping cid -> 1-D np.ndarray (timecourse).
                     Defaults to a 40-sample sine wave for every ID.
    acg_data:        dict mapping cid -> (time_lags, acg_norm) tuples.
                     Defaults to synthetic 50-bin ACG data.
    """
    n = len(subset_ids)
    t_len = 40

    if timecourse_data is None:
        timecourse_data = {
            cid: np.sin(np.linspace(0, 2 * np.pi, t_len)).astype(np.float32)
            for cid in subset_ids
        }

    if acg_data is None:
        t_axis = np.linspace(-50, 50, 50)
        acg_norm = np.random.rand(50).astype(np.float32)
        acg_data = {cid: (t_axis, acg_norm) for cid in subset_ids}

    def _get_cell_physics(cid):
        return {'timecourse': timecourse_data.get(cid)}

    def _get_acg_data(cid):
        return acg_data.get(cid, (None, None))

    dm = MagicMock()
    dm.get_cell_physics.side_effect = _get_cell_physics
    dm.get_acg_data.side_effect = _get_acg_data

    # Minimal MplCanvas mock — each canvas holds state dicts as attributes
    def _make_canvas():
        canvas = MagicMock()
        canvas.fig = MagicMock()
        canvas.fig.axes = []
        canvas.fig.clear = MagicMock()
        return canvas

    mw = MagicMock()
    mw.data_manager = dm
    mw.pop_timecourse_canvas = _make_canvas()
    mw.pop_acg_canvas = _make_canvas()
    mw.pop_timecourse_summary = MagicMock()
    mw.pop_acg_summary = MagicMock()
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

    mw._get_pop_subset_ids.return_value = subset_ids
    return mw


def _clear_module_caches():
    """Reset all module-level caches before each test that touches them."""
    import src.gui.panels.population_panel as pp
    # These attributes are added by the implementation; we clear them if present.
    for attr in ('_group_timecourse_cache', '_group_acg_cache',
                 '_rf_background_cache', '_rf_background_cache_order'):
        cache = getattr(pp, attr, None)
        if isinstance(cache, dict):
            cache.clear()
        elif isinstance(cache, list):
            cache.clear()


# ---------------------------------------------------------------------------
# §5.2 — Group timecourse cache
# ---------------------------------------------------------------------------

class TestGroupTimecourseCache:

    def setup_method(self):
        _clear_module_caches()

    def test_group_timecourse_cache_hit(self):
        """
        Calling draw_population_timecourse_panel twice with the same subset_ids
        should call get_cell_physics only on the first invocation.
        """
        from src.gui.panels.population_panel import draw_population_timecourse_panel

        subset_ids = [0, 1, 2]
        mw = _make_mock_main_window(subset_ids)

        draw_population_timecourse_panel(mw, subset_ids=subset_ids)
        first_call_count = mw.data_manager.get_cell_physics.call_count
        assert first_call_count == len(subset_ids), (
            f"Expected {len(subset_ids)} get_cell_physics calls on cold draw, "
            f"got {first_call_count}"
        )

        # Reset the mock counter without changing side_effect
        mw.data_manager.get_cell_physics.reset_mock()

        draw_population_timecourse_panel(mw, subset_ids=subset_ids)
        second_call_count = mw.data_manager.get_cell_physics.call_count
        assert second_call_count == 0, (
            f"Expected 0 get_cell_physics calls on cache hit, got {second_call_count}"
        )

    def test_different_subsets_use_separate_cache_entries(self):
        """Two different subset_ids lists each populate their own cache entry."""
        from src.gui.panels.population_panel import (
            draw_population_timecourse_panel,
            _group_timecourse_cache,
        )

        ids_a = [0, 1, 2]
        ids_b = [3, 4, 5]
        mw_a = _make_mock_main_window(ids_a)
        mw_b = _make_mock_main_window(ids_b)

        draw_population_timecourse_panel(mw_a, subset_ids=ids_a)
        draw_population_timecourse_panel(mw_b, subset_ids=ids_b)

        assert frozenset(ids_a) in _group_timecourse_cache
        assert frozenset(ids_b) in _group_timecourse_cache
        assert len(_group_timecourse_cache) == 2

    def test_cache_stores_correct_array_shape(self):
        """Cached array has shape (n_cells, t_len)."""
        from src.gui.panels.population_panel import (
            draw_population_timecourse_panel,
            _group_timecourse_cache,
        )

        subset_ids = [10, 11, 12]
        t_len = 40
        mw = _make_mock_main_window(subset_ids)

        draw_population_timecourse_panel(mw, subset_ids=subset_ids)

        key = frozenset(subset_ids)
        assert key in _group_timecourse_cache
        cached = _group_timecourse_cache[key]
        assert cached['arr'].shape == (len(subset_ids), t_len), (
            f"Unexpected cached array shape: {cached['arr'].shape}"
        )


# ---------------------------------------------------------------------------
# §5.2 — Group ACG cache
# ---------------------------------------------------------------------------

class TestGroupACGCache:

    def setup_method(self):
        _clear_module_caches()

    def test_group_acg_cache_hit(self):
        """
        Calling draw_population_acg_panel twice with the same subset_ids
        should call get_acg_data only on the first invocation.
        """
        from src.gui.panels.population_panel import draw_population_acg_panel

        subset_ids = [0, 1, 2]
        mw = _make_mock_main_window(subset_ids)

        draw_population_acg_panel(mw, subset_ids=subset_ids)
        first_call_count = mw.data_manager.get_acg_data.call_count
        assert first_call_count == len(subset_ids)

        mw.data_manager.get_acg_data.reset_mock()

        draw_population_acg_panel(mw, subset_ids=subset_ids)
        second_call_count = mw.data_manager.get_acg_data.call_count
        assert second_call_count == 0, (
            f"Expected 0 get_acg_data calls on cache hit, got {second_call_count}"
        )

    def test_acg_cache_stores_correct_array_shape(self):
        """Cached ACG array has shape (n_cells, n_bins)."""
        from src.gui.panels.population_panel import (
            draw_population_acg_panel,
            _group_acg_cache,
        )

        n_bins = 50
        subset_ids = [20, 21]
        t_axis = np.linspace(-50, 50, n_bins)
        acg_data = {cid: (t_axis, np.random.rand(n_bins).astype(np.float32))
                    for cid in subset_ids}
        mw = _make_mock_main_window(subset_ids, acg_data=acg_data)

        draw_population_acg_panel(mw, subset_ids=subset_ids)

        key = frozenset(subset_ids)
        assert key in _group_acg_cache
        assert _group_acg_cache[key]['arr'].shape == (len(subset_ids), n_bins)


# ---------------------------------------------------------------------------
# §5.3 — RF background LRU cache
# ---------------------------------------------------------------------------

def _make_mock_vision_params(cell_ids):
    """Build a vision_params mock that returns simple stafit objects."""
    def _get_stafit(vision_id):
        sf = MagicMock()
        sf.center_x = float(vision_id) * 3.0
        sf.center_y = float(vision_id) * 2.0
        sf.std_x = 5.0
        sf.std_y = 4.0
        sf.rot = 0.0
        return sf

    vp = MagicMock()
    vp.get_cell_ids.return_value = cell_ids
    vp.get_stafit_for_cell.side_effect = _get_stafit
    return vp


def _make_rf_main_window(subset_ids, all_cell_ids):
    """Main window mock for RF mosaic tests."""
    mw = MagicMock()
    mw.data_manager.vision_params = _make_mock_vision_params(all_cell_ids)
    mw.data_manager.vision_sta_height = 100
    mw.get_current_colors.return_value = {
        'bg_panel': '#18191C',
        'text_primary': '#F0F0F2',
        'text_secondary': '#9B9DA6',
        'plot_highlight': '#FFD93D',
        'border_subtle': '#2E3038',
    }
    mw.pop_show_ids_checkbox = MagicMock()
    mw.pop_show_ids_checkbox.isChecked.return_value = False

    canvas = MagicMock()
    canvas.fig = MagicMock()
    canvas.fig.axes = []

    # Simulate add_patch collecting patches on a real list so we can count them
    patches = []
    ax = MagicMock()
    ax.patches = patches
    ax.add_patch.side_effect = patches.append
    canvas.fig.add_subplot.return_value = ax

    mw.pop_mosaic_canvas = canvas
    mw._get_pop_subset_ids.return_value = subset_ids
    return mw, ax


class TestRFBackgroundLRUCache:

    def setup_method(self):
        _clear_module_caches()

    def test_rf_background_cache_hit_on_revisit(self):
        """
        Draw order A → B → A.
        On the third draw (A revisit) get_stafit_for_cell should NOT be called
        for the background ellipses — only at most once for the highlight patch.
        """
        from src.gui.panels.population_panel import draw_population_rfs_plot

        all_ids = list(range(1, 21))   # 20 Vision IDs (1-indexed)
        # Kilosort IDs are Vision IDs minus 1
        subset_a = list(range(0, 5))   # maps to vision 1–5
        subset_b = list(range(5, 10))  # maps to vision 6–10

        mw_a, ax_a = _make_rf_main_window(subset_a, all_ids)
        mw_b, ax_b = _make_rf_main_window(subset_b, all_ids)

        # Draw A (cold — must call get_stafit for all cells)
        draw_population_rfs_plot(main_window=mw_a, subset_cell_ids=subset_a,
                                 canvas=mw_a.pop_mosaic_canvas)
        calls_after_a = mw_a.data_manager.vision_params.get_stafit_for_cell.call_count
        assert calls_after_a > 0, "Expected stafit calls on cold draw A"

        # Draw B (cold — different canvas mock, own call counter)
        draw_population_rfs_plot(main_window=mw_b, subset_cell_ids=subset_b,
                                 canvas=mw_b.pop_mosaic_canvas)

        # Reset counter on mw_a to count ONLY the revisit calls
        mw_a.data_manager.vision_params.get_stafit_for_cell.reset_mock()

        # Revisit A — should hit the cache; get_stafit calls should be 0 (or 1 for highlight)
        draw_population_rfs_plot(main_window=mw_a, subset_cell_ids=subset_a,
                                 canvas=mw_a.pop_mosaic_canvas)
        revisit_calls = mw_a.data_manager.vision_params.get_stafit_for_cell.call_count

        # The cache should serve background geometry; at most 1 call is allowed
        # for the selected-cell highlight patch (which is always fresh).
        assert revisit_calls <= 1, (
            f"Expected ≤1 get_stafit call on revisit (highlight only), got {revisit_calls}"
        )

    def test_rf_lru_eviction(self):
        """
        After _RF_CACHE_MAX unique subsets have been drawn, the oldest entry
        is evicted when a new one is added.
        """
        from src.gui.panels.population_panel import (
            draw_population_rfs_plot,
            _rf_background_cache,
            _rf_background_cache_order,
            _RF_CACHE_MAX,
        )

        all_ids = list(range(1, 50))

        # Fill the cache to capacity
        first_subset_hash = None
        for i in range(_RF_CACHE_MAX):
            subset = [i]  # Each is a unique 1-element subset → unique hash
            mw, _ = _make_rf_main_window(subset, all_ids)
            draw_population_rfs_plot(main_window=mw, subset_cell_ids=subset,
                                     canvas=mw.pop_mosaic_canvas)
            if i == 0:
                first_subset_hash = hash(tuple(sorted(subset)))

        assert len(_rf_background_cache) == _RF_CACHE_MAX

        # Drawing one more unique subset should evict the oldest
        new_subset = [999]
        mw_new, _ = _make_rf_main_window(new_subset, all_ids)
        draw_population_rfs_plot(main_window=mw_new, subset_cell_ids=new_subset,
                                 canvas=mw_new.pop_mosaic_canvas)

        assert len(_rf_background_cache) == _RF_CACHE_MAX, (
            "Cache size should stay at _RF_CACHE_MAX after eviction"
        )
        assert first_subset_hash not in _rf_background_cache, (
            "Oldest entry should have been evicted"
        )


# ---------------------------------------------------------------------------
# §5.2 + §5.3 — Cache invalidation
# ---------------------------------------------------------------------------

class TestInvalidatePopulationCaches:

    def setup_method(self):
        _clear_module_caches()

    def test_invalidate_clears_all_caches(self):
        """
        After invalidate_population_caches() all four module-level structures
        must be empty.
        """
        import src.gui.panels.population_panel as pp
        from src.gui.panels.population_panel import invalidate_population_caches

        # Seed all caches with dummy data
        pp._group_timecourse_cache[frozenset([1, 2])] = {'arr': np.zeros((2, 40))}
        pp._group_acg_cache[frozenset([3, 4])] = {'arr': np.zeros((2, 50))}
        pp._rf_background_cache[999] = [(0, 0, 10, 8, 0, 'white', 'none', 0.5)]
        pp._rf_background_cache_order.append(999)

        invalidate_population_caches()

        assert len(pp._group_timecourse_cache) == 0
        assert len(pp._group_acg_cache) == 0
        assert len(pp._rf_background_cache) == 0
        assert len(pp._rf_background_cache_order) == 0

    def test_invalidate_is_idempotent(self):
        """Calling invalidate_population_caches() on empty caches should not raise."""
        from src.gui.panels.population_panel import invalidate_population_caches

        _clear_module_caches()
        invalidate_population_caches()  # should not raise
        invalidate_population_caches()  # second call also safe

    def test_refinement_results_invalidate_population_caches(self):
        """Cluster refinement changes group membership, so population caches clear."""
        import src.gui.panels.population_panel as pp
        from src.gui import callbacks

        pp._group_timecourse_cache[frozenset([1, 2])] = {'arr': np.zeros((2, 40))}
        pp._group_acg_cache[frozenset([3, 4])] = {'arr': np.zeros((2, 50))}
        pp._rf_background_cache[999] = [(0, 0, 10, 8, 0, 'white', 'none', 0.5)]
        pp._rf_background_cache_order.append(999)

        main_window = MagicMock()
        main_window.refine_thread = MagicMock()

        with patch('src.gui.callbacks.populate_tree_view') as mock_populate:
            callbacks.handle_refinement_results(main_window, 7, [70, 71])

        main_window.data_manager.update_after_refinement.assert_called_once_with(7, [70, 71])
        mock_populate.assert_called_once_with(main_window)
        assert len(pp._group_timecourse_cache) == 0
        assert len(pp._group_acg_cache) == 0
        assert len(pp._rf_background_cache) == 0
        assert len(pp._rf_background_cache_order) == 0

    def test_partial_vision_loaded_invalidate_population_caches(self):
        """A Vision reload can change timecourses/RFs, so population caches clear."""
        import src.gui.panels.population_panel as pp
        from src.gui import callbacks

        pp._group_timecourse_cache[frozenset([1, 2])] = {'arr': np.zeros((2, 40))}
        pp._group_acg_cache[frozenset([3, 4])] = {'arr': np.zeros((2, 50))}
        pp._rf_background_cache[999] = [(0, 0, 10, 8, 0, 'white', 'none', 0.5)]
        pp._rf_background_cache_order.append(999)

        main_window = MagicMock()
        main_window.vision_load_thread = None
        main_window.vision_load_worker = None
        main_window.data_manager.vision_stas = None
        main_window.data_manager.vision_eis = None

        with patch('src.gui.callbacks.QMessageBox.warning'):
            callbacks._on_vision_loaded(main_window, success=False, message="partial", is_partial=True)

        assert len(pp._group_timecourse_cache) == 0
        assert len(pp._group_acg_cache) == 0
        assert len(pp._rf_background_cache) == 0
        assert len(pp._rf_background_cache_order) == 0


def test_population_group_plots_cached_requires_both_caches():
    import src.gui.panels.population_panel as pp
    from src.gui.panels.population_panel import population_group_plots_cached

    _clear_module_caches()
    ids = [1, 2, 3]
    assert population_group_plots_cached(ids) is False
    assert population_group_plots_cached([]) is False

    pp._group_timecourse_cache[frozenset(ids)] = {"arr": np.zeros((3, 8))}
    assert population_group_plots_cached(ids) is False

    pp._group_acg_cache[frozenset(ids)] = {"arr": np.zeros((3, 8))}
    assert population_group_plots_cached(ids) is True
    assert population_group_plots_cached([1, 2]) is False


# ---------------------------------------------------------------------------
# §5.4 — LazySTADict dynamic cache size
# ---------------------------------------------------------------------------

class TestLazySTACacheSize:

    def test_lazy_sta_cache_size_scales_with_dataset(self, tmp_path):
        """
        With 800 keys and MAX_STA_CACHE_CELLS=500, _max_cache should be 500.
        With 150 keys and MAX_STA_CACHE_CELLS=500, _max_cache should be 200 (floor).
        """
        from src.analysis.vision_integration import LazySTADict, MAX_STA_CACHE_CELLS

        # Mock the STAReader so no real files are needed
        mock_reader = MagicMock()

        # Case 1: large dataset — capped at MAX_STA_CACHE_CELLS
        mock_reader.cell_id_to_byte_offset = {i: i * 100 for i in range(800)}
        with patch('src.analysis.vision_integration.vl') as mock_vl:
            mock_vl.STAReader.return_value = mock_reader
            lazy = LazySTADict(tmp_path, 'test_dataset')

        expected_large = min(MAX_STA_CACHE_CELLS, max(200, 800))
        assert lazy._max_cache == expected_large, (
            f"Expected _max_cache={expected_large} for 800-key dataset, "
            f"got {lazy._max_cache}"
        )

        # Case 2: small dataset — floor at 200
        mock_reader.cell_id_to_byte_offset = {i: i * 100 for i in range(150)}
        with patch('src.analysis.vision_integration.vl') as mock_vl:
            mock_vl.STAReader.return_value = mock_reader
            lazy_small = LazySTADict(tmp_path, 'test_dataset')

        expected_small = min(MAX_STA_CACHE_CELLS, max(200, 150))
        assert lazy_small._max_cache == expected_small, (
            f"Expected _max_cache={expected_small} for 150-key dataset, "
            f"got {lazy_small._max_cache}"
        )
        assert lazy_small._max_cache >= 200, "_max_cache must never fall below 200"

    def test_max_sta_cache_cells_constant_exists(self):
        """MAX_STA_CACHE_CELLS must be a module-level constant >= 200."""
        from src.analysis.vision_integration import MAX_STA_CACHE_CELLS

        assert isinstance(MAX_STA_CACHE_CELLS, int)
        assert MAX_STA_CACHE_CELLS >= 200, (
            f"MAX_STA_CACHE_CELLS={MAX_STA_CACHE_CELLS} is too small; "
            "must be >= 200 to be useful for large folders"
        )
