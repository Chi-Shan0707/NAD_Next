# 14 — Coding Improvement V2

**Date**: 2026-04-13  **Problems**: 167  **Status**: full-run completed

## Verdict

- **Recoverability**: Tier B — code text and prompt text are locally recoverable; execution-style fields are absent.
- **Overall winner**: `code_v2_baseline` remains best on the real full-set metric with top1 `0.6168` over all `167` problems.
- **Best non-baseline challenger**: `static_code_chars` reaches top1 `0.6108` (`-0.0060` vs `code_v2`).
- **Baseline-aware rerankers**: adding static or neuron summaries on top of `code_v2` does **not** help; all tested meta-rerankers regress.
- **Distance-native caveat**: selectors like `knn_k7` look strong at `0.7246`, but that number is on the `138` oracle-solvable problems only. On the same denominator, `code_v2_baseline` is `0.7464`, so the selector still loses.
- **Conclusion**: V2 finds useful evidence and better diagnostics, but no deployable verifier / reranker in the current Tier-B setting beats the existing `code_v2` reranker on real full-data evaluation.

## Input Audit

| Field | Count |
|---|---:|
| Prompt non-empty | 10688 |
| Generated text non-empty | 10688 |
| Extracted answer non-empty | 10367 |
| Recovered code non-empty | 10688 |
| Selected report kind | full |

## Full Grouped Results

All metrics below are on the real full dataset (`167` problems, including `29` unsolvable problems where every candidate must fail).

| Candidate | Top1 | Pass@1 uplift | Pairwise AUC | Pooled AUROC | Corr(output_tokens) | Corr(code_chars) |
|---|---:|---:|---:|---:|---:|---:|
| code_v2_baseline | 0.6168 | — | 0.5211 | 0.5030 | +0.1589 | +0.1534 |
| static_code_chars | 0.6108 | -0.0060 | 0.4892 | 0.5426 | +0.2940 | +1.0000 |
| hybrid_logreg | 0.5988 | -0.0180 | 0.4984 | 0.5120 | +0.1149 | -0.0832 |
| neuron_n_active_total | 0.5988 | -0.0180 | 0.4821 | 0.5400 | +0.9756 | +0.2285 |
| neuron_wmax_mean_global | 0.5928 | -0.0240 | 0.5253 | 0.4517 | -0.5734 | -0.0321 |
| static_logreg | 0.5868 | -0.0299 | 0.4902 | 0.5422 | +0.2266 | -0.0646 |
| static_pairwise_logreg | 0.5389 | -0.0778 | 0.4887 | 0.4988 | -0.0118 | -0.1298 |

## Baseline-Aware Reranker Check

These follow-up experiments explicitly treat `code_v2_baseline` as the base score and learn a reranker on top of it. They are saved in `results/validation/coding_v2_meta_rerankers.full.json`.

| Meta Reranker | Top1 | Pass@1 uplift vs `code_v2` | Pairwise AUC | Head-to-head |
|---|---:|---:|---:|---|
| meta_pointwise_base_hybrid | 0.5988 | -0.0180 | 0.4992 | wins=9 losses=12 ties=146 |
| meta_pointwise_base_static | 0.5928 | -0.0240 | 0.4901 | wins=13 losses=17 ties=137 |
| meta_pointwise_base_neuron | 0.5928 | -0.0240 | 0.4805 | wins=11 losses=15 ties=141 |
| meta_pairwise_base_static | 0.5509 | -0.0659 | 0.4868 | wins=6 losses=17 ties=144 |

**Takeaway**: even when the model is allowed to start from the current best reranker, simple static / neuron residualization still hurts top1.

## Neuron-Native Evidence

- **Random top1**: `0.5869`
- **Oracle-solvable rate**: `0.8263` (`138 / 167`)
- **Jaccard CC–CI gap**: `+0.00048`
- **Medoid top1**: `0.5784`
- **Best layer**: layer `32`, pairwise AUC `0.5248`
- **w_max mean pairwise AUC**: `0.5098`
- **Consensus voting top1**: `0.7216` *(not deployable; it uses other correct solutions from the same problem as a reference set)*

## Distance-Native Selectors, Apples-to-Apples

The raw selector table from the full run only reports the `138` solvable problems, so it is **not** directly comparable to the all-problem `0.6168` headline. The fair comparison is against `code_v2_baseline` on the same solvable subset. Those aligned numbers are saved in `results/validation/coding_v2_distance_selectors_solvable.full.json`.

| Method on solvable subset (`138` problems) | Top1 | Pass@1 uplift vs solvable `code_v2` | Pairwise AUC |
|---|---:|---:|---:|
| code_v2_baseline | 0.7464 | — | 0.5211 |
| knn_k7 | 0.7246 | -0.0217 | 0.5014 |
| max_active | 0.7246 | -0.0217 | 0.4821 |
| knn_k3 | 0.7101 | -0.0362 | 0.4974 |
| wmax_mean | 0.7029 | -0.0435 | 0.5098 |
| wmax_top10pct | 0.7029 | -0.0435 | 0.5083 |
| min_active | 0.6957 | -0.0507 | 0.5179 |
| medoid | 0.6884 | -0.0580 | 0.4899 |

**Takeaway**: distance-native selectors are genuinely strong heuristics, but they still do not beat the current `code_v2` reranker once the denominator is aligned.

## Failure Diagnosis

### 1) Root cause: statistical smoothing of semantics

The dominant failure mode is consistent with a **statistical smoothing effect**:

- most code tokens / activations are syntax, scaffolding, imports, loop skeletons, container operations, and other high-frequency boilerplate;
- the tiny subset of semantically decisive choices — e.g. `<` vs `<=`, boundary initialization, off-by-one index movement, missing corner-case branches — contributes too little mass to global token-confidence or bag-of-neuron summaries;
- as a result, pooled confidence, structural counts, activation totals, and Jaccard-style overlap mostly measure “looks like a normal solution” rather than “is semantically correct.”

### 2) Evidence from the real run

- **Near-zero class manifold gap**: CC–CI gap is only `+0.00048`, so correct and incorrect code are not cleanly separated by neuron-set similarity.
- **Weak layer signal**: the best single layer reaches only pairwise AUC `0.5248`.
- **Shortcut leakage**: `neuron_n_active_total` has correlation `+0.9756` with output tokens, showing that some flashy neuron signals are mostly response-size proxies.
- **Static AST-lite features do not rescue ranking**: the strongest non-baseline static challenger is just code length (`static_code_chars`), which still loses to `code_v2`.
- **Meta-reranking also fails**: even after feeding `code_v2` into learned rerankers, performance drops rather than improves.

### 3) What this means scientifically

This is not “just needs another 0.5 AUROC point of tuning.” It is a boundary case where:

- surface-form plausibility is high,
- execution-semantic errors are sparse and localized,
- label-defining evidence is much lower-entropy than the surrounding trace,
- and bagged confidence / activation summaries systematically under-represent the decisive signal.

## Option A / Option B

### Option A — pursue a truly stronger verifier

The next credible improvement path is **not** more pooled confidence tuning. It is to add signals that directly target semantic correctness:

1. **AST / operator topology features (`CodeCircuit`-style)**  
   Move beyond counts like “how many loops” into localized structure:
   - comparison-operator fingerprints,
   - boundary-update motifs,
   - control-flow branch topology,
   - dataflow around loop indices and accumulators,
   - prompt-conditioned operator expectations.

2. **Execution-state probes (`Gnosis`-style)**  
   The current local artifacts are only Tier B, so there is no existing pass/fail trace, stdout/stderr, or hidden-test execution record. To get a real jump, we likely need to reconstruct or generate execution probes:
   - run recovered code on public/sample tests,
   - synthesize adversarial edge cases from the prompt,
   - record failure states and disagreement patterns,
   - use those as verifier features rather than raw confidence.

3. **Dynamic coverage / path signals (`SAGA`-style)**  
   If we can obtain runnable test cases, path diversity and branch coverage are much closer to the semantic boundary than token confidence or neuron counts.

### Option B — frame this as a valuable boundary case

If execution-style supervision cannot be recovered, the current result is still scientifically useful:

- the dataset is Tier B rather than Tier A;
- code text and prompt text are available, but execution evidence is missing;
- multiple independent feature families fail in the same way;
- even baseline-aware rerankers regress;
- the problem cleanly demonstrates where confidence-style selection breaks on code.

That is a publishable-style negative result: **coding correctness is a sparse-semantic property that can remain invisible to aggregate confidence and aggregate neuron activity, even when those signals are strong in other domains.**
