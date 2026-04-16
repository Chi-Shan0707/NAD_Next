# Literature Map: Code Reasoning, Mental Models, and Structured Traces

## How to read this map

This map distinguishes:

- **Direct support**: directly bears on the hypothesis that explicit state/control-flow representation matters for code reasoning.
- **Near neighbor**: conceptually relevant, but does not directly test the exact hypothesis.
- **Benchmark/context**: useful for experimental setup or evaluation framing.

Also keep four separations explicit:

- `code generation` ≠ `code reasoning / execution`
- `final answer accuracy` ≠ `reasoning trace quality`
- `free-form CoT` ≠ `structured/stateful representation`
- `human code comprehension` ≠ `LLM code reasoning`

## A. Program tracing / working memory / mental models / notional machines

### [Pennington 1987 — *Stimulus Structures and Mental Representations in Expert Comprehension of Computer Programs*](https://www.cs.kent.edu/~jmaletic/cs69995-PC/papers/pennington87.pdf)

- **Research question**: how do programmers mentally represent programs during comprehension?
- **Method**: behavioral experiments on expert programmers; compares understanding of control flow, data flow, functions, and overall goals.
- **Relation to this project**: foundational evidence that program comprehension involves structured mental representations rather than flat text narration.
- **Why it matters here**: supports the idea that execution reasoning depends on keeping track of program structure and state relations.
- **Limitations**: human experts, not LLMs; predates current code benchmarks; not a direct comparison of prose vs structured traces.
- **Support tag**: **Direct support**.

### [Sorva 2013 — *Notional Machines and Introductory Programming Education*](https://research.aalto.fi/en/publications/notional-machines-and-introductory-programming-education/)

- **Research question**: what abstract machine model must learners acquire in order to reason correctly about program execution?
- **Method**: literature synthesis in computing education around misconceptions, mental models, and instructional representations.
- **Relation to this project**: directly motivates the “representation mismatch” framing; the notional machine is exactly an explicit execution model.
- **Why it matters here**: suggests that a good representation of code reasoning is one that preserves executable semantics, not merely verbal description.
- **Limitations**: education-facing and human-centered; not about LLM prompting or trace scoring.
- **Support tag**: **Direct support**.

### [Crichton, Agrawala, Hanrahan 2021 — *The Role of Working Memory in Program Tracing*](https://arxiv.org/abs/2101.06305)

- **Research question**: how much does short-term working memory constrain tracing performance?
- **Method**: controlled experiments on tracing strategies, restricted-focus interfaces, and WM load.
- **Relation to this project**: very strong support for the claim that tracing is a state-maintenance problem; directly mentions variable/value pairs and call stack.
- **Why it matters here**: predicts that any reasoning representation that externalizes state should help when `live_var_count` or scope pressure rises.
- **Limitations**: studies humans, not models; compares tracing strategies, not prompt formats.
- **Support tag**: **Direct support**.

### [Xie, Nelson, Ko 2018 — *An Explicit Strategy to Scaffold Novice Program Tracing*](https://faculty.washington.edu/ajko/papers/Xie2018TracingStrategies.pdf)

- **Research question**: can explicitly teaching line-by-line tracing with an external memory representation improve tracing?
- **Method**: randomized educational intervention; trace prediction tasks; think-aloud + performance analysis.
- **Relation to this project**: highly relevant because the intervention explicitly combines sequential tracing with a memory table.
- **Why it matters here**: closest human-side evidence that **representation choice** changes execution reasoning quality.
- **Limitations**: novice-focused; does not isolate which component helps most (line-by-line discipline vs external table).
- **Support tag**: **Direct support**.

### [Heinonen et al. 2023 — *Synthesizing Research on Programmers' Mental Models of Programs, Tasks and Concepts*](https://arxiv.org/abs/2212.07763)

- **Research question**: what is known about programmers’ mental models across programs, tasks, and concepts?
- **Method**: systematic literature review across 84 selected studies.
- **Relation to this project**: useful umbrella map of the human literature; helps avoid overclaiming from isolated classic papers.
- **Why it matters here**: supports the broad claim that mental models are central, but does **not** directly validate our specific LLM representation hypothesis.
- **Limitations**: synthesis, not a new execution experiment; broader than tracing alone.
- **Support tag**: **Near neighbor**.

## B. Code comprehension cognitive science / neuroscience

### [Srikant et al. 2023 — *Program Comprehension Does Not Primarily Rely On the Language Centers of the Human Brain*](https://arxiv.org/abs/2304.12373)

- **Research question**: is code comprehension mainly supported by language systems or by domain-general problem-solving systems?
- **Method**: fMRI with Python and ScratchJr tasks, contrasting code with language and control tasks.
- **Relation to this project**: supports the caution that “because CoT is language, it should be a natural code-reasoning medium” is a weak assumption.
- **Why it matters here**: strengthens the argument that code execution reasoning may need representations aligned with structured problem solving, not ordinary prose.
- **Limitations**: no claim about prompting or reasoning formats; brain activation ≠ representation prescription.
- **Support tag**: **Near neighbor**.

### [Decker et al. 2023 — *Developers' Visuo-spatial Mental Model and Program Comprehension*](https://arxiv.org/abs/2304.09301)

- **Research question**: how does visuo-spatial mental organization relate to comprehension and navigation in code?
- **Method**: empirical study of developers’ spatial organization / code understanding.
- **Relation to this project**: relevant to the idea that programmers often reason with structured, spatialized representations instead of pure narrative text.
- **Why it matters here**: suggests code understanding may benefit from external forms that preserve structure and locality.
- **Limitations**: about human spatial organization, not LLM traces; not execution-specific.
- **Support tag**: **Near neighbor**.

### [Sharafi et al. 2021 — *Toward an Objective Measure of Developers’ Cognitive Activities*](https://web.eecs.umich.edu/~weimerw/p/weimer-tosem2021-cognitive.pdf)

- **Research question**: can neural + eye-tracking measures distinguish code comprehension from prose review?
- **Method**: fNIRS / eye-tracking style objective measurements across developer tasks.
- **Relation to this project**: helps separate code comprehension from ordinary reading and prose processing.
- **Why it matters here**: useful contextual support for why code reasoning may need specialized representational scaffolds.
- **Limitations**: not about execution traces or prompt formats.
- **Support tag**: **Near neighbor**.

## C. LLM code reasoning, structured CoT, execution traces

### [Li et al. 2023 — *Structured Chain-of-Thought Prompting for Code Generation (SCoT)*](https://arxiv.org/abs/2305.06599)

- **Research question**: does structuring intermediate reasoning by `sequence / branch / loop` help code generation?
- **Method**: prompting study on code-generation benchmarks.
- **Relation to this project**: strong adjacent evidence for our representation claim because it operationalizes program structure explicitly.
- **Why it matters here**: supports the idea that free-form NL CoT underuses code structure.
- **Limitations**: **code generation**, not code execution reasoning; better code does not prove better runtime reasoning traces.
- **Support tag**: **Near neighbor, very relevant**.

### [Ni et al. 2024 — *NExT: Teaching Large Language Models to Reason about Code Execution*](https://proceedings.mlr.press/v235/ni24a.html)

- **Research question**: can execution-aware rationales improve LLM reasoning about runtime behavior?
- **Method**: synthetic execution-aware rationale generation + self-training; evaluates repair tasks and rationale quality.
- **Relation to this project**: the single most relevant LLM paper in the current map.
- **Why it matters here**: directly argues that models benefit from reasoning over execution traces and variable states, not just surface text.
- **Limitations**: still not a clean free-form-vs-structured ablation on matched tasks; evaluated via repair/fix rate and rationale quality, not detailed boundary-error taxonomy.
- **Support tag**: **Direct support**.

### [Armengol-Estapé et al. 2025 — *What I cannot execute, I do not understand: Training and Evaluating LLMs on Program Execution Traces*](https://arxiv.org/abs/2503.05703)

- **Research question**: what happens if we train/evaluate LLMs explicitly on execution traces, and which trace granularity works best?
- **Method**: execution tuning with line-level and instruction-level traces; output-prediction evaluation; dynamic scratchpads.
- **Relation to this project**: very close to the pilot idea, especially on trace representations and long executions.
- **Why it matters here**: supports the premise that dynamic trace representations are useful and that long executions stress weaker representations.
- **Limitations**: training-focused rather than prompt-only; does not isolate free-form NL CoT as the comparison baseline we care about most.
- **Support tag**: **Direct support**.

### [Gao et al. 2022 — *PAL: Program-aided Language Models*](https://arxiv.org/abs/2211.10435)

- **Research question**: can LLMs do better when intermediate reasoning is executable code rather than free-form text?
- **Method**: generate programs as intermediate reasoning and offload execution to an interpreter.
- **Relation to this project**: broader evidence that symbolic/executable intermediates can dominate pure language reasoning.
- **Why it matters here**: supports the more general “structured or executable scratchpads beat prose” intuition.
- **Limitations**: mainly math/word-problem style tasks, not program execution explanation.
- **Support tag**: **Near neighbor**.

### [Lam et al. 2025 — *CODECRASH: Stress Testing LLM Reasoning under Structural and Semantic Perturbations*](https://arxiv.org/abs/2504.14119)

- **Research question**: how robust are code-reasoning models to structural noise and misleading semantic cues?
- **Method**: perturbation benchmark built on CRUXEval and LiveCodeBench.
- **Relation to this project**: relevant because it shows code reasoning can be fragile to structural perturbations and natural-language distractions.
- **Why it matters here**: indirect evidence that models may over-rely on textual cues rather than preserving executable semantics.
- **Limitations**: stress test, not a representation-ablation study; robustness loss does not by itself prove representation mismatch.
- **Support tag**: **Near neighbor**.

## D. Code reasoning benchmarks

### [Gu et al. 2024 — *CRUXEval: A Benchmark for Code Reasoning, Understanding and Execution*](https://arxiv.org/abs/2401.03065)

- **Research question**: can models predict outputs or infer inputs for short Python functions?
- **Method**: 800 short Python functions with input/output prediction tasks.
- **Relation to this project**: important because it is explicitly **code reasoning/execution**, not generation.
- **Why it matters here**: a natural external benchmark for later transfer beyond the synthetic pilot.
- **Limitations**: focuses on final task success; does not provide rich gold reasoning traces or boundary-error annotations.
- **Support tag**: **Benchmark/context**.

### [Jain et al. 2024 — *LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code*](https://arxiv.org/abs/2403.07974)

- **Research question**: how do models perform on up-to-date, contamination-resistant coding tasks across multiple abilities?
- **Method**: continuously refreshed contest-style benchmark; includes code execution and output prediction settings.
- **Relation to this project**: useful for realism and future external validation.
- **Why it matters here**: lets us test whether pilot findings survive on less synthetic tasks.
- **Limitations**: broad benchmark; not designed to score reasoning-trace factuality.
- **Support tag**: **Benchmark/context**.

### [Xu et al. 2024 — *CRUXEval-X: A Benchmark for Multilingual Code Reasoning, Understanding and Execution*](https://arxiv.org/abs/2408.13001)

- **Research question**: how well do code-reasoning abilities transfer across programming languages?
- **Method**: multilingual extension of CRUXEval.
- **Relation to this project**: useful later if the representation-mismatch effect appears language-agnostic.
- **Why it matters here**: suggests a path beyond Python-only pilot tasks.
- **Limitations**: multilingual breadth is not needed for the first pilot; still not a trace-quality benchmark.
- **Support tag**: **Benchmark/context**.

## Which papers actually support the hypothesis?

### Strongest direct supports

- [Pennington 1987](https://www.cs.kent.edu/~jmaletic/cs69995-PC/papers/pennington87.pdf)
- [Sorva 2013](https://research.aalto.fi/en/publications/notional-machines-and-introductory-programming-education/)
- [Crichton et al. 2021](https://arxiv.org/abs/2101.06305)
- [Xie et al. 2018](https://faculty.washington.edu/ajko/papers/Xie2018TracingStrategies.pdf)
- [Ni et al. 2024 / NExT](https://proceedings.mlr.press/v235/ni24a.html)
- [Armengol-Estapé et al. 2025](https://arxiv.org/abs/2503.05703)

These are the papers that most directly support some version of: **execution reasoning needs explicit state and/or trace-aware representations**.

### Important but only near neighbors

- [SCoT](https://arxiv.org/abs/2305.06599)
- [PAL](https://arxiv.org/abs/2211.10435)
- [Srikant et al. 2023](https://arxiv.org/abs/2304.12373)
- [Decker et al. 2023](https://arxiv.org/abs/2304.09301)
- [CODECRASH](https://arxiv.org/abs/2504.14119)

These strengthen the intuition, but they do **not** directly prove that free-form NL CoT is the wrong reasoning carrier for code execution.

## Literature Gap This Project Targets

What is missing in the literature is a clean experiment that:

- holds code tasks fixed,
- varies only the **reasoning representation**,
- measures **trace quality** rather than only final answers,
- and explicitly probes:
  - state growth,
  - loop nesting,
  - boundary transitions,
  - phase switches,
  - local scope / call-stack handling.

That is the gap the pilot benchmark in this repo is designed to fill.
