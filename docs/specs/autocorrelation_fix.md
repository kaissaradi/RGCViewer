# Specification: Autocorrelation Uses Full Recording

## STATUS: DONE

## Objective
Ensure autocorrelation data is computed from all spikes in a cluster, including
spikes that occur late in long recordings.

## User Story
As a user reviewing firing patterns, I want the ACG to include the entire
recording so that late bursts or late drift are visible in the standard plots.

## Acceptance Criteria
- AC1: Spikes after the first two minutes can contribute to the ACG.
- AC2: A burst late in the recording produces the expected lag peak.
- AC3: Clusters with too few spikes return an empty ACG instead of bogus data.

## Technical Constraints
- The computation lives in `DataManager._compute_standard_plots`.
- The output ACG uses millisecond lags from -100 ms to +100 ms.
- The algorithm must avoid dense arrays proportional to recording duration.

## Test Plan
- Unit tests build synthetic spike trains with late-only and mixed early/late
  spikes and assert the +10 ms ACG bin is populated.
- Edge tests cover too-few-spikes behavior and output shape.

## Out Of Scope
- Changing the ACG bin width or visual rendering style.
