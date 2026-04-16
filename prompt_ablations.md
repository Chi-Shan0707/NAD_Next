# Prompt Ablations / Representation Ablations

## Common setup

Use the same task content for every condition:

- same `code`
- same `entry_call`
- same final-answer requirement
- no code execution tools
- no external interpreter

Common system instruction:

> You are solving a code execution reasoning task. Do not rewrite or run the code. Reason from the given Python code and input only.

Common task wrapper:

````text
Code:
```python
{{CODE}}
```

Question:
What is the exact return value of `{{ENTRY_CALL}}`?
````

## A. Free-form Natural-Language CoT

### Intent

Baseline free narrative reasoning with minimal structure.

### Template

```text
You may think step by step in natural language, but do not use tables or explicit trace schemas unless they arise naturally.

Return exactly this format:

FINAL_OUTPUT: <exact value>
REASONING:
<free-form natural-language reasoning>
```

### What it tests

- free prose as the reasoning carrier
- vulnerability to linearized narrative drift

## B. Structured CoT with explicit control labels

### Intent

Force the model to preserve program structure explicitly.

### Template

```text
Reason using explicit control-flow labels.

Return exactly this format:

FINAL_OUTPUT: <exact value>
STRUCTURED_COT:
[SEQ line=<line>] ...
[BRANCH line=<line> choice=<then|else>] ...
[LOOP line=<line> iter=<k>] ...
[CALL line=<line> fn=<name>] ...
[RETURN line=<line> value=<value>] ...
```

### What it tests

- whether control-flow externalization alone helps
- whether explicit structural labels reduce branch/loop confusion

## C. State-table reasoning

### Intent

Force explicit state snapshots after each executed step.

### Template

```text
Reason by maintaining a state table. Record only executed steps.

Return exactly this format:

FINAL_OUTPUT: <exact value>
STATE_TABLE:
step | line | scope | vars
0 | <line> | <scope> | <json dict of variables after the step>
1 | ...
```

### Rules

- `vars` should contain the variables currently in scope after that step.
- Only include executed steps.
- Do not skip branch decisions or loop iterations.

### What it tests

- whether explicit state externalization reduces state drift
- whether loop and scope bookkeeping improve

## D. Execution-trace reasoning

### Intent

Match the task to an execution-style representation as closely as possible.

### Template

```text
Reason using an execution trace. Record only executed events.

Return exactly this format:

FINAL_OUTPUT: <exact value>
TRACE:
- step: 0
  event: call
  line: <line>
  scope: <scope>
  locals: <json dict>
- step: 1
  event: line
  line: <line>
  scope: <scope>
  locals: <json dict>
  branch_taken: <then|else|null>
  loop_iter: <int|null>
- step: ...
```

### What it tests

- closest prompt-level proxy to explicit program execution
- best expected performance on branch/state/boundary metrics

## E. Optional: invariant + boundary-first reasoning

### Intent

Probe whether an abstraction-first format helps especially on loops and phase switches.

### Template

```text
Before solving, state:
1. the loop invariant or running summary,
2. the important boundary points,
3. the phase-switch rule if one exists.

Then compute the answer.

Return exactly this format:

FINAL_OUTPUT: <exact value>
INVARIANTS:
- ...
BOUNDARY_POINTS:
- ...
REASONING:
<concise structured reasoning>
```

### What it tests

- whether explicit boundary awareness helps on `phase_switch_loop`
- whether invariants help without a full trace

## Recommended Evaluation Notes

- For condition A, parse only:
  - `FINAL_OUTPUT`
  - free-text reasoning length
- For B-E, parse both final output and trace-like fields.
- Do **not** treat longer reasoning as automatically better.
- Report parse-failure rate separately from reasoning-failure rate.

## Recommended Pilot Order

1. Run `A vs B vs C vs D`
2. Add `E` only if loop/phase results are especially interesting
3. Later add token-budget-matched variants of A-D
