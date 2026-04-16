# Code Reasoning vs Natural-Language CoT Pilot (2026-04-16)

## Summary

This work turns a vague intuition into a runnable pilot:

- **Hypothesis**: for code execution reasoning, free-form natural-language CoT may be a mismatched carrier because it flattens execution semantics into linear prose.
- **Alternative representations**: structured CoT, state tables, execution traces, and invariant/boundary-first reasoning may be more stable because they externalize state and control flow.
- **Scope of this drop**: research framing, literature map, falsifiable hypotheses, pilot benchmark generator, prompt ablations, evaluation plan, and sample tasks.

The target is **code reasoning / execution**, not code generation.

## Deliverables

- Root docs:
  - `research_summary.md`
  - `literature_map.md`
  - `hypotheses_and_rqs.md`
  - `experiment_plan.md`
  - `prompt_ablations.md`
  - `failure_taxonomy.md`
- Runnable artifact:
  - `pilot_benchmark.py`
- Sample dataset:
  - `sample_tasks.jsonl`
- Entry-point docs:
  - `README.md`
  - `docs/README.md`

## What the pilot benchmark includes

Seven synthetic Python program families:

1. `sequential_baseline`
2. `single_branch`
3. `simple_loop`
4. `loop_plus_branch`
5. `phase_switch_loop`
6. `nested_loop`
7. `function_scope`

Each task records controllable properties:

- `branch_depth`
- `loop_nesting`
- `live_var_count`
- `phase_switch_count`
- `boundary_case_count`
- `scope_depth`
- `invariant_needed`

Each task also includes:

- executable `code`
- `entry_call`
- `gold_output`
- `gold_trace`
- `boundary_points`
- `expected_failure_modes`

## Gold trace design

`pilot_benchmark.py` executes the synthesized program and emits a gold trace with:

- `call`, `line`, and `return` events
- `locals_before` and `locals_after`
- `branch_taken`
- `loop_iter`
- `phase`
- `boundary_tags`

Important boundary tags already supported:

- `first_iteration`
- `last_iteration`
- `iteration_at_pivot`
- `phase_switch`
- `first_post_switch_iteration`
- `helper_call_entry`
- `helper_return`
- `caller_resume`

This is enough for later evaluation of:

- final answer accuracy
- trace factuality
- trace validity
- per-step state accuracy
- branch-choice accuracy
- boundary-case error rate

## Validation run

The following command was run successfully:

```bash
python3 pilot_benchmark.py --num-per-family 3 --seed 0 --out sample_tasks.jsonl --pretty-sample 5
```

Observed result:

- `21` tasks generated
- all `7` families present
- all `3` difficulty levels present
- `gold_output` matches the entry-function return value on checked tasks
- loop iteration sequences are consistent
- `phase_switch` and `caller_resume` tags are present where expected

The repository verification script was also run:

```bash
bash cookbook/00_setup/verify.sh
```

Current environment note:

- benchmark generation works
- some broader repo dependencies are missing locally (`pyroaring`, `flask`, `plotly`, `hmmlearn`, `tokenizers`, `transformers`), so `verify.sh` reports environment failures unrelated to this pilot generator

## Main conclusions from this round

### What is established

- The literature supports the broader claim that program tracing depends on explicit state maintenance and structured execution models.
- Existing LLM work supports execution-aware or structured intermediates as promising tools for code reasoning.

### What remains speculative

- The sharper claim that **free-form NL CoT is intrinsically mismatched** for code execution reasoning is still unproven.
- The current repo drop creates the artifacts needed to test that claim cleanly.

### Best next experiment

Run the same task set under prompt formats:

- `A` free-form CoT
- `B` structured CoT
- `C` state-table reasoning
- `D` execution-trace reasoning

Then compare not only final answers, but also:

- trace factuality
- branch-choice accuracy
- boundary-case error rate
- failure taxonomy distribution

## Recommendation

The highest-value next implementation is a **model-output scorer** that:

- parses A/B/C/D prompt outputs
- aligns them against `gold_trace`
- computes the metrics defined in `experiment_plan.md`
- emits per-family and per-failure-mode summaries
