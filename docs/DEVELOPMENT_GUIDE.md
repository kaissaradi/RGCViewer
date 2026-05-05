# Developer Guide: Workflow & Testing

Welcome to the RGCViewer development team. This project follows a strict **Spec-Driven Development (SDD)** and **Test-Driven Development (TDD)** workflow to ensure the stability and performance of our complex GUI.

---
 use the conda env rgcviewer

 
## 1. The Workflow

For every new feature or bug fix, follow these four steps:

### Phase 1: Specification (SDD)
Before writing any code, define the feature's behavior in a Markdown file within the `docs/specs/` directory (create it if it doesn't exist).
- **Objective**: What problem are we solving?
- **User Story**: "As a user, I want to [action] so that [result]."
- **Technical Constraints**: Any specific UI components or data structures involved.

### Phase 2: Write Failing Tests (TDD)
Create a test file in the `tests/` directory that targets your new specification.
- Use `tests/unit/` for pure logic (no GUI).
- Use `tests/integration/` for GUI components (using `qtbot`).
- **Goal**: Run `python -m pytest tests/` and watch it fail.

### Phase 3: Implementation
Implement the minimum code necessary to make the tests pass. Keep the GUI layer (`src/gui/`) thin and delegate heavy lifting to `DataManager` or background `workers`.

### Phase 4: Refactor & Verify
Once the tests are green, refactor for performance and readability. Run the full suite again to ensure no regressions.

---

## 2. Testing Tools

We use a "Next-Level" testing stack:
- **pytest**: Our core test runner.
- **pytest-qt**: Provides the `qtbot` fixture to simulate user interaction (clicks, typing, drag-and-drop).
- **pytest-mock**: Used to inject "dummy" data and mock the `DataManager` so tests run in milliseconds without loading real 50GB data files.
- **pytest-benchmark**: Used to track and assert performance (e.g., "Tree view must render 1k items in <100ms").
- **pytest-mpl**: Used for deterministic Matplotlib visual comparisons.
- **psutil/objgraph**: Used by memory and stress tests.

---

## 3. How to Test Before Committing

Always run the full test suite before pushing changes:

```bash
# Run all tests
python -m pytest tests/

# Run tests and show benchmark results
python -m pytest tests/ --benchmark-only

# Run tests with coverage report
python -m pytest tests/ --cov=src
```

For headless machines, the test suite sets Qt to offscreen mode and Matplotlib
to the Agg backend from `tests/conftest.py`.

---

## 4. Advanced Testing

### 1. Visual Regression Testing
We use `pytest-mpl` to ensure plots don't change unexpectedly.
- **Run comparison**: `python -m pytest tests/integration/test_visual_regression.py --mpl`
- **Generate new baselines**: `python -m pytest tests/integration/test_visual_regression.py --mpl-generate-path=tests/integration/baselines`

### 2. Stress & Memory Testing
To ensure the app doesn't leak memory or freeze under load:
- **Run stress tests**: `python -m pytest tests/performance/test_stress.py -s`
- These tests use `objgraph` and `psutil` to track memory growth and object counts.

### 3. Synthetic Data Simulation
The `MockBinFileReader` in `src/analysis/analysis_core.py` allows testing the Trace Viewer (RawPanel) without large `.bin` files.
- See `tests/integration/test_raw_panel_synthetic.py` for usage.

---

## 5. Best Practices for GUI Testing

1.  **Mock Heavy Data**: Never load a real `.dat` or `.bin` file in a test. Use `unittest.mock.MagicMock` to simulate the `DataManager`.
2.  **Isolate Panels**: Test one panel at a time (e.g., `STAPanel`) by providing it with a `MockMainWindow`.
3.  **Check Z-Order**: When testing plotting features, verify that items (like labels) are added to the plot and have the correct `ZValue`.
4.  **Simulate, Don't Invoke**: Use `qtbot.mouseClick(button)` instead of `button.click()` to ensure the event loop and signals are actually firing as they would for a user.
5.  **Wait for Conditions**: Prefer `qtbot.waitUntil(...)` over fixed sleeps so tests are faster and failures point at the missing state.

---

## 5. Directory Structure
- `src/`: Source code.
- `docs/specs/`: Feature specifications (The "What").
- `tests/`: Automated tests (The "Check").
    - `unit/`: Logic tests.
    - `integration/`: GUI component tests.
    - `e2e/`: Full application workflows.
    - `performance/`: Benchmarks.

---

## 6. Roadmap & Ongoing Work
### Completed Features
- **Cell ID Toggle**: Toggle visibility of ID numbers in plots.
- **Testing Infrastructure**: `pytest`, `pytest-qt`, and `pytest-benchmark` setup.
- **Developer Documentation**: Workflow guide established.

### Next Phase: Performance & UX Polish
- **Performance Benchmarking**: Measure and optimize the most laggy parts of the UI.
  - Tree view population time.
  - Scatter plot refresh rate with large subsets.
  - Splitter resizing responsiveness.
- **UX/UI Improvements**:
  - Tree View: implement drag-and-drop for folder organization.
  - Dynamic Layouts: improve how panels resize when the "Population Split View" is toggled.
  - Theme Consistency: ensure all custom widgets react instantly to Light/Dark mode toggles.
- **Split View for Cell Table**:
  - Secondary table on the right for side-by-side comparison.
  - Status: research synchronization logic.

### Ongoing Workflow
1. **Spec** -> 2. **Test** -> 3. **Implement** -> 4. **Bench**
