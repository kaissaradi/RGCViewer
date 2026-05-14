# Unused Code and Documentation Scan Report

## Python module analysis
This report is generated from a static import scan across the repository. It is a candidate list of modules that are not explicitly imported by other Python files in the repo. Dynamic imports, test-only imports, and plugin entrypoints are not captured.

### Candidate unused Python modules
- `src.__init__`
- `src.analysis.__init__`
- `src.analysis.analysis_core`
- `src.analysis.constants`
- `src.analysis.vision_integration`
- `tests.conftest`
- `tests.integration.test_cell_id_toggle`
- `tests.integration.test_gui_sanity`
- `tests.integration.test_raw_panel_synthetic`
- `tests.integration.test_tree_operations`
- `tests.integration.test_umap_selection`
- `tests.integration.test_visual_regression`
- `tests.performance.test_concurrency_fix`
- `tests.performance.test_first_click`
- `tests.performance.test_lock_contention`
- `tests.performance.test_physics_perf`
- `tests.performance.test_profiling`
- `tests.performance.test_stress`
- `tests.performance.test_tree_view_perf`
- `tests.unit.test_autocorrelation`
- `tests.unit.test_autocorrelation_verify`
- `tests.unit.test_data_manager_cache`
- `tests.unit.test_sanity`

## Documentation files
- Total Markdown files found: 9
- Current root Markdown pages remaining: ['README.md']
- Specification files now in `docs/specs/`: ['docs/specs/README.md', 'docs/specs/autocorrelation_fix.md', 'docs/specs/performance_optimization.md', 'docs/specs/physics_optimization.md', 'docs/specs/umap_selection_fix.md']

## Notes & next steps
- Review the candidate unused modules before deleting. Some are only referenced dynamically or by test discovery.
- Fix documentation references from `specs/` to `docs/specs/` if any remain.
- Keep `README.md` in the repo root as the main GitHub landing page.