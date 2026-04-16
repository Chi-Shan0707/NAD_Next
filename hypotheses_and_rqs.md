# Research Questions and Hypotheses

## Framing

This project studies **code reasoning / code execution**, not code generation.  
The core claim is about **reasoning representation**:

- `free-form natural-language CoT`
- vs `structured/stateful representations`

And the main outcome is **not only final answer accuracy**, but also **reasoning trace quality**.

## Research Questions

### RQ1 — Scaling with semantic load

When execution-semantic load increases, does free-form CoT degrade faster than structured/stateful formats?

- **Semantic load dimensions**:
  - `branch_depth`
  - `loop_nesting`
  - `live_var_count`
  - `scope_depth`
  - `phase_switch_count`
  - `boundary_case_count`

### RQ2 — Boundary sensitivity

Are free-form CoTs disproportionately error-prone at boundary points such as:

- first iteration
- last iteration
- branch entry
- `pivot` vs `pivot + 1`
- function return to caller

### RQ3 — Representation effect on trace quality

Do state-externalizing formats improve:

- per-step state accuracy
- branch-choice accuracy
- trace factuality
- trace validity

even when final answer accuracy is similar?

### RQ4 — Phase-switch specificity

Are “phase-switch” programs especially diagnostic of representation mismatch, because they require maintaining both iteration state and a transition rule over time?

### RQ5 — Final answer vs reasoning trace dissociation

How often do models get the final output correct while producing a factually wrong trace, and does this dissociation vary by representation format?

## Testable Hypotheses

### H1 — Load-sensitivity hypothesis

As `control-flow depth` and `live-state size` increase, **free-form CoT performance declines faster** than structured/stateful formats.

- **Primary DVs**:
  - final answer accuracy
  - per-step state accuracy
  - trace factuality

### H2 — Externalized-state hypothesis

Prompt formats that explicitly externalize state (`state table`, `execution trace`) significantly reduce:

- wrong variable update
- hallucinated state
- scope confusion
- branch-choice error

relative to free-form CoT.

### H3 — Control-structure hypothesis

`Structured CoT` with explicit `SEQ / BRANCH / LOOP / CALL / RETURN` labels is more stable than free-form CoT, especially on programs with loops and nested control flow.

### H4 — Phase-switch hypothesis

Programs with a phase transition inside a loop produce a **larger representation gap** than sequential, single-branch, or simple-loop baselines.

### H5 — Trace dissociation hypothesis

Free-form CoT will show a larger gap between:

- `final answer accuracy`
- and `trace factuality / validity`

than structured/stateful formats.

## Independent Variables

### Primary manipulated variables

- **Prompt format**
  - A: free-form natural-language CoT
  - B: structured CoT
  - C: state-table reasoning
  - D: execution-trace reasoning
  - E: invariant + boundary-first reasoning (optional)
- **Program family**
  - sequential baseline
  - single branch
  - simple loop
  - loop + branch
  - phase-switch loop
  - nested loop
  - function call / local scope
- **Difficulty / semantic-load controls**
  - `branch_depth`
  - `loop_nesting`
  - `live_var_count`
  - `phase_switch_count`
  - `boundary_case_count`
  - `scope_depth`
  - `invariant_needed`

### Secondary variables for later experiments

- model family / model size
- decoding temperature
- token budget
- few-shot vs zero-shot

## Dependent Variables

- **Final answer accuracy**: exact match on output
- **Trace factuality**: whether reported trace statements match gold execution facts
- **Trace validity**: whether the trace forms a coherent executable sequence
- **Per-step state accuracy**: variable/value correctness at aligned steps
- **Branch-choice accuracy**: correct branch selected at each branch point
- **Boundary-case error rate**: error rate restricted to boundary-tagged events
- **Token cost / verbosity**: total output length or token count
- **Failure-type distribution**: taxonomy counts per format / family

## Main Confounds to Control

### Task-side confounds

- code length
- identifier difficulty
- arithmetic difficulty
- unusual Python constructs
- hidden library knowledge

### Prompt-side confounds

- prompt length
- output-format strictness
- whether one format is easier to parse automatically
- whether one format implicitly gives more computation budget

### Model-side confounds

- decoding randomness
- model-specific instruction-following bias
- prior exposure to trace-like formats

## Operational Notes

- `live_var_count` is treated as runtime-visible active state, not just number of identifiers in source.
- `trace factuality` and `trace validity` must be scored separately:
  - a trace can be well-formed but false
  - a trace can contain true fragments but be structurally invalid
- `boundary-case error rate` should be computed on boundary-tagged gold events only.

## What Falsifies the Main Claim

The representation-mismatch thesis weakens if:

- free-form CoT matches structured/stateful formats on trace quality under matched token budget, or
- any observed gain disappears once formatting and verbosity are controlled.

## Expected Pattern

The predicted ordering is:

`Execution Trace / State Table` ≥ `Structured CoT` > `Free-form CoT`

with the largest gaps on:

- `phase_switch_loop`
- `nested_loop`
- `function_scope`
- high `live_var_count`
- high `boundary_case_count`
