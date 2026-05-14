# Specification: [Short Feature Or Bug Name]

## Metadata

* **Status:** [Draft | Ready for Dev | In Progress | Done]
* **Target Release:** [e.g., v1.1]
* **Primary Developer/Agent:** [Name]

## Objective

[What specific problem are we solving? Provide context on why this matters.]

## User Story

"As a [user persona], I want to [action] so that [result/benefit]."

## Acceptance Criteria (Definition of Done)

*Must be strictly binary (Pass/Fail) and testable.*

* **AC1:** [Observable behavior that must be true]
* **AC2:** [Edge case or regression that must stay fixed]
* **AC3 (Visual Check):** [e.g., "The Right Panel split view renders without overlapping the Main Canvas when resized."]

## Architecture & Technical Constraints

* **Files Modified:** [Explicitly list where the logic should live, e.g., `src/analysis/data_manager.py`]
* **Data Contracts:** [Expected inputs/outputs]
* **UI/Threading Rules:** [e.g., "This must execute in `StandardPlotsWorker` and emit a Qt Signal; it cannot block the main thread."]

## Test Plan (TDD Requirements)

*Tests must be written to fail BEFORE implementation begins.*

* **Unit:** [Pure logic and data contracts, e.g., `tests/unit/test_feature.py`]
* **Integration:** [Panel, worker, or Qt signal behavior using `qtbot`]
* **Performance/Visual:** [Required if the spec has speed, memory, or rendering risks using `pytest-benchmark` or `pytest-mpl`]
* **Screenshot Verification:** [Instructions for the AI or User to launch the GUI, trigger the state, and verify visually via screenshot/MCP tools.]

## Out Of Scope

* [Things this spec deliberately does NOT change to prevent scope creep.]
