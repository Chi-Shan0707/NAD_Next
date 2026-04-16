# Code Reasoning vs Natural-Language CoT — Research Summary

## Thesis

核心判断不是“代码任务不能推理”，而是：**当任务本质上要求维护执行语义（state / control flow / scope / phase transition）时，free-form natural-language chain-of-thought 可能是一个有 representation mismatch 的载体**。  
更具体地说，代码执行需要显式保持：

- `state`: variable values, scope, call stack, path conditions
- `control flow`: sequence / branch / loop
- `time`: iteration-by-iteration evolution
- `boundaries`: first/last iteration, `m` vs `m+1`, phase switch, call/return

自由叙事式 CoT 会把这些结构压平成线性 prose；结构化 CoT、state table、execution trace、invariant-first reasoning 则更接近程序语义本身。

## What Is Established

- **Human tracing depends on explicit state maintenance.**  
  Classic program-comprehension work already distinguishes control-flow understanding from richer procedural understanding; people do not simply “read code like text” ([Pennington 1987](https://www.cs.kent.edu/~jmaletic/cs69995-PC/papers/pennington87.pdf)).
- **Working memory is a real bottleneck in tracing.**  
  Program tracing requires holding variable/value pairs and stack state; people make swap and overload errors when WM load grows ([Crichton et al. 2021](https://arxiv.org/abs/2101.06305)).
- **Externalized tracing helps humans.**  
  A lightweight explicit tracing strategy with a memory table improved novice tracing accuracy, which is direct evidence that external representation matters for execution reasoning ([Xie, Nelson, Ko 2018](https://faculty.washington.edu/ajko/papers/Xie2018TracingStrategies.pdf)).
- **Code comprehension is not primarily ordinary language processing.**  
  Neuro/cognitive evidence suggests code comprehension leans on domain-general problem-solving systems more than language centers, weakening any naive assumption that free-form linguistic narration is automatically the right medium ([Srikant et al. 2023](https://arxiv.org/abs/2304.12373)).
- **LLM code reasoning benefits from execution-aware or structured intermediates.**  
  NExT shows execution-aware rationales help code-execution reasoning; SCoT shows structured intermediate representations help code generation; newer trace-centric work reinforces that dynamic trace representations matter ([Ni et al. 2024](https://proceedings.mlr.press/v235/ni24a.html), [Li et al. 2023](https://arxiv.org/abs/2305.06599), [Armengol-Estapé et al. 2025](https://arxiv.org/abs/2503.05703)).

## What Is Still Speculative

- **The sharp claim** that *free-form NL CoT is specifically a poor carrier for code execution reasoning* is **not yet established directly**.
- Existing papers usually show one of three weaker claims:
  - humans need external state support;
  - structured intermediates help some code tasks;
  - execution traces improve model training or prompting.
- What remains to test is whether the failure mode is truly a **representation mismatch**, rather than:
  - prompt-length differences,
  - parser/evaluation artifacts,
  - generic task difficulty,
  - or the fact that structured formats simply force more careful output.

## Research Position

这个方向最像一个 **LLM evaluation question**，但受到 **cognitive science** 强启发，并且有明显 **software-engineering relevance**：

- **Cognitive-science angle**: explicit state externalization vs working-memory load
- **LLM-eval angle**: does representation choice change trace fidelity under controlled semantic load?
- **SE angle**: what explanation / debugging interfaces should code LLMs expose?

## What Would Support the Representation-Mismatch Claim

- Free-form CoT and structured/stateful formats have similar final-answer accuracy on easy tasks, but **free-form trace factuality collapses faster** as:
  - `loop_nesting` rises,
  - `live_var_count` rises,
  - `scope_depth` rises,
  - `phase_switch_count` rises,
  - `boundary_case_count` rises.
- The gap is largest on:
  - branch boundaries,
  - loop first/last iterations,
  - phase-switch transitions,
  - function call / local-scope transitions.
- Structured/stateful formats reduce errors that are specifically semantic bookkeeping errors:
  - wrong variable update,
  - wrong branch selected,
  - loop off-by-one,
  - boundary transition failure,
  - scope confusion,
  - hallucinated state.

## What Would Weaken or Refute It

- Under matched token budget and matched output constraints, free-form CoT matches structured/stateful formats on:
  - final answer accuracy,
  - per-step state accuracy,
  - branch-choice accuracy,
  - boundary-case error rate.
- Or the gains vanish once we control for output-format strictness, meaning the effect is mostly “format regularization” rather than semantic representation.

## What To Test Next

### Priority 1

- Run the implemented synthetic pilot benchmark across prompt ablations `A-D`.
- Compare **trace quality**, not just final answers.
- Start with 1-2 strong reasoning models plus 1 smaller open model for contrast.

### Priority 2

- Add a **token-budget-matched** condition:
  - free-form CoT capped to the same budget as structured trace
  - structured trace capped to the same total output budget

### Priority 3

- Add **real benchmark transfer**:
  - CRUXEval-style short execution tasks
  - a small LiveCodeBench output-prediction subset

### Priority 4

- Test whether **dynamic scratchpads** outperform accumulated historical traces on long loops or long nested traces, following the trace-modeling literature.

## What Is Established / Speculative / Next

### Established

- State maintenance matters in program tracing.
- External representations can help tracing.
- Code comprehension is not merely natural-language comprehension.
- Execution-aware / structured intermediates often help code-related tasks.

### Speculative

- Free-form NL CoT is intrinsically mismatched for code execution reasoning.
- Phase-switch and boundary transitions disproportionately break free-form CoT.
- Structured gains persist after tight control for verbosity and formatting effects.

### Next

- Use `pilot_benchmark.py` + `sample_tasks.jsonl` to run the first prompt-format ablation.
- Score `trace factuality`, `trace validity`, and `boundary-case error rate` before drawing strong conclusions.
