# Code-Oriented Selector Validation

Date: `2026-04-05`

## Summary

This note records the completion of the code-oriented selector slice that was previously only in handoff state.

Implemented selector files:

- `nad/core/selectors/code_dynamic_impl.py`
- `plugins/prefix_saturation_selector.py`

Committed implementation:

- `e725687` — `selectors: optimize code-oriented dynamic plugin`

Final decision:

- keep the plugin as a file-loaded selector for now
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

## Bottom Line

The code-oriented prefix/tail/reflection plugin is now:

- implemented
- validated on the intended `livecodebench_v5` cache
- committed

It beats both required baselines on the main objective:

- `SelAcc@10`

without failing the guardrails:

- `Pairwise`
- `Hit@1`
