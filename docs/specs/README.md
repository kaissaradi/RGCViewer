# Specs

Every behavior change starts here before implementation. Specs should stay
small, testable, and tied directly to one or more tests.

## Template

```markdown
# Specification: Short Feature Or Bug Name

## Objective
What problem are we solving?

## User Story
As a user, I want to ... so that ...

## Acceptance Criteria
- AC1: Observable behavior that must be true.
- AC2: Edge case or regression that must stay fixed.

## Technical Constraints
- Important modules, widgets, data contracts, threading constraints, or
  performance budgets.

## Test Plan
- Unit: pure logic and data contracts.
- Integration: panel, worker, or Qt signal behavior.
- Performance/Visual: only when the spec has speed, memory, or rendering risk.

## Out Of Scope
- Things this spec deliberately does not change.
```

Tests should reference the acceptance criteria in their names or docstrings
when that makes the intent clearer.
