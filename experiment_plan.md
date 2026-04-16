# Experiment Plan

## Study Goal

Build a **small but rigorous pilot** that can test whether code-execution reasoning is sensitive to the representation used for intermediate reasoning, especially under explicit state/control-flow pressure.

## Pilot Scope

### In scope

- synthetic Python execution tasks
- gold outputs
- gold execution traces
- prompt-format ablations
- trace-quality evaluation design
- threats to validity

### Out of scope for v1

- model API integration
- leaderboard-scale benchmark release
- multilingual support
- broad software-engineering tasks

## Benchmark Design

### Program families

The pilot benchmark includes seven families:

1. **Sequential baseline**
2. **Single branch**
3. **Simple loop**
4. **Loop + branch**
5. **Phase-switch loop**
6. **Nested loop**
7. **Function call / local scope**

### Program constraints

- Python 3.9+ only
- deterministic
- no imports
- no random state
- no file I/O
- no network
- no recursion in v1
- short functions only

### Difficulty ladder

Each family is generated in `low / medium / high` variants by increasing semantic bookkeeping burden rather than syntax novelty.

Examples:

- more simultaneously live variables
- longer loops
- added temporary state
- more salient boundaries
- extra scope transitions

## Dataset Schema

Each JSONL task contains:

- `task_id`
- `family`
- `difficulty`
- `code`
- `entry_fn`
- `input`
- `entry_call`
- `question`
- `gold_output`
- `gold_trace`
- `attributes`
- `boundary_points`
- `expected_failure_modes`
- `trace_summary`

### Gold trace schema

Each event records at least:

- `step`
- `event_type` (`call`, `line`, `return`)
- `line`
- `line_text`
- `scope_name`
- `scope_depth`
- `call_id`
- `locals_before`
- `locals_after`
- `branch_taken`
- `branch_line`
- `loop_iter`
- `loop_line`
- `phase`
- `boundary_tags`

This schema is deliberately designed so later scorers can evaluate:

- branch choices
- state updates
- loop iteration counts
- phase transitions
- call/return scope behavior

## Experimental Matrix

### Minimal pilot

- **Tasks**: the generated synthetic set from `pilot_benchmark.py`
- **Representations**: A-D mandatory, E optional
- **Models**: not bundled in this repo; recommended 2-3 models later

### Recommended first external run

- 1 strong frontier reasoning model
- 1 capable code model
- 1 smaller/open model
- 3 decoding seeds if API cost allows

## Evaluation Plan

### Primary metrics

- **Final answer accuracy**
  - exact match of final output
- **Trace factuality**
  - fraction of predicted trace assertions that match gold facts
- **Trace validity**
  - whether the trace is structurally coherent
- **Per-step state accuracy**
  - aligned state comparison on executed steps
- **Branch-choice accuracy**
  - correct branch at each branch event
- **Boundary-case error rate**
  - errors restricted to boundary-tagged events
- **Token cost / verbosity**
  - output tokens or character-count proxy

### Suggested scoring rules

- Align predicted steps to gold by `line + scope + event order`.
- Score state entries as exact key/value matches.
- Penalize invented variables as `hallucinated state`.
- Penalize missing required steps as trace omissions.
- Report both macro-average and family-level breakdowns.

### Key analysis slices

- by `family`
- by `difficulty`
- by `loop_nesting`
- by `phase_switch_count`
- by `boundary_case_count`
- by `scope_depth`

## Threats to Validity

### Internal validity

- **Format advantage confound**: structured prompts may help because they constrain output, not because they better match semantics.
- **Parser confound**: structured outputs are easier to evaluate automatically.
- **Token-budget confound**: one format may effectively grant more reasoning space.

### Construct validity

- Synthetic tasks may not capture real-world debugging or comprehension.
- Trace quality metrics may still imperfectly reflect “reasoning faithfulness.”
- Gold traces reflect one execution view; a model may reason correctly with a different but equivalent abstraction.

### External validity

- Effects on short Python snippets may not transfer to larger programs.
- Results may differ for generation, repair, debugging, and explanation tasks.
- Closed models and open models may respond differently to output formatting constraints.

## Recommended Controls

- Keep code length narrow within each family.
- Match token budget in a follow-up condition.
- Use the same final-answer channel across prompts.
- Evaluate both raw traces and normalized traces.
- Separate “parse failed” from “reasoning failed.”

## Acceptance Criteria for This Pilot

The pilot is considered complete when:

- `pilot_benchmark.py` generates JSONL successfully
- all seven families are present
- every task has `gold_output` and `gold_trace`
- sampled traces are consistent with execution
- prompt ablations are written and standardized
- evaluation metrics and threats to validity are explicit

## Highest-Value Next Experiments

### 1. Prompt-only representation ablation

- Same model, same tasks, same token budget where possible
- Compare A/B/C/D directly

### 2. Boundary-focused analysis

- Restrict scoring to `boundary_tags`
- Test whether the gap grows sharply at phase transitions and scope returns

### 3. Transfer to external benchmarks

- Small CRUXEval-style output-prediction subset
- Small LiveCodeBench output-prediction subset

### 4. Training-time trace exposure

- Compare prompt-only structured reasoning vs models already trained/tuned on traces

## Expected Interpretation

If stateful formats mainly help on state, branch, and boundary metrics—and not just by a small uniform margin—then the result is more consistent with **representation mismatch** than with generic prompt engineering.
