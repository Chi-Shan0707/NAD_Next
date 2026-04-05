# Code-Oriented Selector Validation

Date: `2026-04-05`

> Follow-up note:
> - phase-2 leave-one-out, grid, transfer, and disagreement results are recorded in:
>   - `docs/CODE_BASELINE_V1_PHASE2_20260405.md`
> - `code_baseline_v1` remains the frozen baseline
> - the best guarded phase-2 candidate is `thr=0.30 / lb=16 / pf=0.30`

## Summary

This note records the completion of the code-oriented selector slice that was previously only in handoff state.

Implemented selector files:

- `nad/core/selectors/code_dynamic_impl.py`
- `plugins/prefix_saturation_selector.py`

Committed implementation:

- `e725687` — `selectors: optimize code-oriented dynamic plugin`

Final decision:

- keep the plugin as a file-loaded selector for now
- freeze the current coding-specialized implementation as `code_baseline_v1`
- keep the coding selector scope separate from `earlystop_v3`
- keep the default coding reflection threshold at `0.30`
- treat the new plugin as the current best coding-oriented zero-training selector on this cache when the objective is `SelAcc@10` with `Pairwise` / `Hit@1` guardrails

## Implementation Notes

The implementation stayed within the intended selector-first slice:

- shared code-dynamic feature extraction lives in `nad/core/selectors/code_dynamic_impl.py`
- plugin scoring lives in `plugins/prefix_saturation_selector.py`
- no graph-topology expansion was added
- no `earlystop_v3` refactor was attempted

The most important implementation fix during finalization was performance-related:

- the original reflection feature path effectively scanned too much slice history on long coding traces
- final code uses a bounded local reflection scan with early exit
- this preserves the intended “revisiting earlier structure” signal while making full-cache validation practical

This bounded local scan is also the intended coding inductive bias:

- coding quality is better captured by local dynamic convergence shape than by long-range graph topology
- the useful structure is concentrated in prefix saturation, head-tail settling, reflection density, tail variance, and post-event recovery
- the implementation now encodes that hypothesis directly instead of treating reflection as unbounded full-history matching

Feature directions in the final implementation:

- `prefix_best_window_quality`: lower is better
- `head_tail_gap`: higher is better
- `reflection_density`: lower is better
- `tail_variance`: lower is better
- `post_reflection_recovery`: higher is better

## Validation Setup

Target cache:

- `MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/livecodebench_v5/cache_neuron_output_1_act_no_rms_20251127_032808`

Required selectors compared:

- `min-confidence`
- `tournament-copeland`
- `file:/home/jovyan/work/NAD_Next/plugins/prefix_saturation_selector.py:PrefixSaturationSelector`

Primary ranking validation config:

- distance: `ja`
- cut: `mass:1.0`
- distance threads: `12`
- problems: `167`
- samples: `10,688`

Saved artifacts:

- `result/tmp_selector_validation/lcb_code_selector_metrics_ja_full.json`
- `result/tmp_selector_validation/lcb_code_selector_analyze_ja_full.json`
- `result/tmp_selector_validation/lcb_code_selector_accuracy_ja_full.json`

Regression rerun after the shared scorer freeze:

- plugin selection stayed identical on all `167/167` problems versus the earlier validated `nad.cli analyze` output
- rerun `nad.cli accuracy` stayed at:
  - `min-confidence`: `59.88%` (`100/167`)
  - `tournament-copeland`: `62.28%` (`104/167`)
  - `PrefixSaturationSelector`: `59.28%` (`99/167`)

## Main Results

### Deterministic ranking metrics

These are the metrics used for the actual selector decision, because they directly evaluate ranking quality over all runs in each problem.

| Selector | Hit@1 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|
| `min-confidence` | `59.88%` | `49.51%` | `52.15%` |
| `tournament-copeland` | `58.08%` | `49.92%` | `59.27%` |
| `PrefixSaturationSelector` | `59.28%` | `51.27%` | `61.70%` |

### Required CLI pass

The required end-to-end pipeline pass was also run with:

- `nad.cli analyze`
- `nad.cli accuracy`

`nad.cli accuracy` summary:

- `min-confidence`: `59.88%` (`100/167`)
- `tournament-copeland`: `62.28%` (`104/167`)
- `PrefixSaturationSelector`: `59.28%` (`99/167`)

## Interpretation

### Why the plugin is still the recommended winner

The target objective for this slice was explicitly:

- prioritize `SelAcc@10`
- use `Pairwise` / `Hit@1` as guardrails

Under that objective, the plugin is the best of the three compared selectors:

- best `SelAcc@10` by `+2.77pp` over `tournament-copeland`
- best `Pairwise`, and the only selector clearly above `0.50`
- `Hit@1` remains close to the best baseline, down only `0.60pp` versus `min-confidence`

### Why CLI accuracy differs for `tournament-copeland`

The built-in `TournamentCopelandSelector` does not behave like a pure deterministic argmax scorer:

- it samples from Copeland wins through a softmax policy
- therefore `nad.cli accuracy` reflects its sampled top-choice behavior
- the ranking report above uses deterministic Copeland win scores as a ranking signal

So the two views are not contradictory; they are measuring different objects:

- `nad.cli accuracy`: selected answer correctness
- ranking metrics: how well each selector orders runs for `SelAcc@10` / `Pairwise`

For this slice, the ranking metrics are the actual decision gate.

## Threshold Check

I also compared the plugin’s coding reflection threshold choices:

- `reflection_threshold = 0.30`
- `reflection_threshold = 0.20`

Result:

- `0.30` gave the better overall tradeoff and the better `SelAcc@10`
- `0.20` slightly improved `Pairwise` but lost too much on `Hit@1` and `SelAcc@10`

So the final default remains:

- `reflection_threshold = 0.30`

## `code_baseline_v1` Submission Patch

The selector is now treated as a specialized coding module rather than a candidate global replacement.

Frozen `code_baseline_v1` defaults:

- `reflection_threshold = 0.30`
- `reflection_lookback_slices = 16`
- `prefix_fraction = 0.20`
- `prefix_window_tokens = 128`
- weights remain the current prefix/recovery-heavy defaults

Submission export scope:

- base submission:
  - `submission/BestofN/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank.json`
- patched submission:
  - `submission/BestofN/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_baseline_v1_lcb_patch.json`
- replacement scope:
  - only `DS-R1/lcb_v5`
  - only `Qwen3-4B/lcb_v5`
- all other cache keys remain unchanged from the base submission

Patch method:

- compute the same group-internal primary score now used by the plugin
- sort each coding problem by that primary score
- break exact score ties with the same medoid-style local `D`-sum rule used by the selector
- rank-normalize each problem back into the existing `1..100` submission scale

This is intentionally a coding-only artifact:

- it does **not** claim the selector should replace the global submission stack
- it exists to inject the validated code-dynamic ranking signal into the `lcb_v5` slice without broadening scope

## Next Phase

The next experiments should stay on the same coding-first axis:

1. freeze this version as `code_baseline_v1` and run 5-feature leave-one-out
2. run a small coding-only hyperparameter grid:
   - `reflection_threshold ∈ {0.20, 0.25, 0.30, 0.35}`
   - `lookback ∈ {8, 16, 24}`
   - `prefix_fraction ∈ {0.10, 0.20, 0.30}`
3. run a `2×2` transfer gate:
   - code-tuned selector on code vs math/science/global
   - global or non-code selector on code
4. keep graph features out of scope unless the new coding-first line is exhausted first

## Bottom Line

The code-oriented prefix/tail/reflection plugin is now:

- implemented
- validated on the intended `livecodebench_v5` cache
- committed
- frozen as `code_baseline_v1`
- exported into a coding-only patched submission artifact

It beats both required baselines on the main objective:

- `SelAcc@10`

without failing the guardrails:

- `Pairwise`
- `Hit@1`

Phase-2 follow-up keeps that validation intact while clarifying the next step:

- keep `code_baseline_v1` frozen
- use phase-2 only to identify the next coding-specific feature move
- keep graph out of scope until the current coding line is exhausted
