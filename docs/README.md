# Document map

Read in this order. Do not start from a historical spec.

| Order | File | Audience | Content |
|---|---|---|---|
| 1 | `README.md` | Human | Install, start, tests |
| 2 | `CLAUDE.md` | Agent and human | Experiment, file layout, analysis traps |
| 3 | `docs/AGENTS.md` | Agent | Laws, caches, threads, test rules |
| 4 | `docs/PLAN.md` | Agent and human | Pickup, standing decisions, fragile zones, open defects |
| 5 | `docs/design/palette.md` | Agent and human | Locked Bauhaus color tokens |
| 6 | `docs/SPEC.md` | Agent | Template for a new feature spec |

## Historical specs

`docs/specs/` holds completed or parked feature specs. Treat them as
history unless `docs/PLAN.md` names one as active.

| Spec | Status |
|---|---|
| `ux_ui_redesign.md` | Parked design. Not the current work. |
| `vision_standalone.md` | Partial. Missing `.sta` / `.params` no longer crash. |
| `cross_run_stimulus_bridge.md` | Implemented. Lab acceptance is open. |
| `grating_dsos_flexible_conditions.md` | Done. Actual `(bw, tf)` grouping, best-run pick, population polar, parallel grating batch. |
| `grating_panel_spec.md` | Done. Original panel. Constants superseded by the spec above. |
| Other files in `docs/specs/` | Done or superseded. Do not reopen unless asked. |

## Writing rules

Write project documents in ASD-STE100 style:

- One fact or one instruction per sentence.
- Use the same word for the same thing.
- Use active voice and present tense for descriptions.
- Use numbered steps for procedures.
- Remove session narrative, commit tables, and finished-work logs.
