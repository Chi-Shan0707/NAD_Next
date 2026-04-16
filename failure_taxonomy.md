# Reasoning Trace Error Taxonomy

## Purpose

This taxonomy is for **reasoning-trace evaluation**, not just final-answer grading.

## Core error types

| Code | Error Type | Operational definition | Typical symptom |
|---|---|---|---|
| F1 | wrong variable update | A variable is assigned the wrong value after an executed step | `acc` or `total` drifts from gold state |
| F2 | wrong branch selected | The trace enters `then` vs `else` incorrectly | condition evaluated or applied incorrectly |
| F3 | loop off-by-one | The trace uses the wrong loop bounds or iteration count | misses first/last iteration or adds one extra |
| F4 | boundary transition failure | The model mishandles a salient boundary such as `pivot` vs `pivot+1` | phase switch happens too early or too late |
| F5 | scope confusion | A local variable is confused with a caller variable, or call/return propagation is wrong | helper local leaks into caller or return value is misapplied |
| F6 | invariant drift | The running summary stops matching the true loop semantics | early steps correct, later accumulator logic inconsistent |
| F7 | hallucinated state | The trace invents variables, values, or steps not licensed by the program | mentions nonexistent temp vars or impossible states |
| F8 | trace-format invalid | The submitted trace cannot be parsed or is structurally incomplete | missing required fields, malformed table, broken JSON-like output |

## Boundary-focused subtype tags

These are not separate root errors; they are location tags for analysis.

- `B1 first_iteration_error`
- `B2 last_iteration_error`
- `B3 branch_entry_error`
- `B4 phase_switch_error`
- `B5 caller_resume_error`

## How to assign errors

### Single-step rule

Assign the **first semantically primary error** at the step where divergence begins.

Example:

- branch chosen incorrectly
- all later state updates wrong because of that branch

Primary error = `F2 wrong branch selected`, not repeated `F1` for every downstream line.

### Cascading rule

After the primary error, later consequences may be logged as:

- `derived_state_error`
- or ignored for root-cause counting

For the pilot, prefer **root-cause counting** plus a separate per-step metric.

## Family-specific expectations

### Sequential baseline

Common:

- `F1 wrong variable update`
- `F7 hallucinated state`

### Single branch

Common:

- `F2 wrong branch selected`
- `F1 wrong variable update`

### Simple loop

Common:

- `F3 loop off-by-one`
- `F1 wrong variable update`
- `F6 invariant drift`

### Loop + branch

Common:

- `F2 wrong branch selected`
- `F3 loop off-by-one`
- `F6 invariant drift`

### Phase-switch loop

Common:

- `F4 boundary transition failure`
- `F2 wrong branch selected`
- `F3 loop off-by-one`

### Nested loop

Common:

- `F3 loop off-by-one`
- `F6 invariant drift`
- `F7 hallucinated state`

### Function scope

Common:

- `F5 scope confusion`
- `F1 wrong variable update`
- `F7 hallucinated state`

## Suggested Scoring Outputs

For each model × prompt format × family:

- total errors by type
- root-cause error rate
- boundary-only error rate
- parse-failure rate
- examples of 3-5 representative failures

## Minimal Annotation Guidance

When manually checking traces:

1. verify final output
2. find first divergence from gold trace
3. assign one primary taxonomy code
4. add a boundary tag if divergence starts on a boundary event

This keeps annotation lightweight enough for a pilot while still diagnostic.
