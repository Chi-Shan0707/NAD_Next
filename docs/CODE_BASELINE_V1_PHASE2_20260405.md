# Code Baseline v1 Phase 2

Date: `2026-04-05`

## Summary

This note records the phase-2 follow-up after freezing the coding-specialized selector as `code_baseline_v1`.

Source artifacts:

- `result/code_baseline_v1_phase2_20260405_080141/loo_summary.md`
- `result/code_baseline_v1_phase2_20260405_080141/grid_summary.md`
- `result/code_baseline_v1_phase2_20260405_080141/disagreement_summary.md`
- `result/code_baseline_v1_phase2_20260405_080141/transfer_gate.md`

Phase-2 decisions:

- keep `code_baseline_v1` frozen as the current validated coding baseline
- do **not** reopen graph features
- do **not** merge this coding line into `earlystop_v3`
- treat the best grid point as an experimental submission candidate, not yet a promoted new baseline
- treat `last-block instability` as the next feature family to test

## Frozen Baseline

Frozen baseline metrics on the target coding cache:

| Selector | AUROC | Hit@1 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| `code_baseline_v1` | `50.27%` | `59.28%` | `51.27%` | `61.74%` |

Frozen feature family:

- `prefix_best_window_quality_r`
- `head_tail_gap_r`
- `reflection_density_r`
- `tail_variance_r`
- `post_reflection_recovery_r`

Frozen defaults:

- `reflection_threshold = 0.30`
- `reflection_lookback_slices = 16`
- `prefix_fraction = 0.20`
- `prefix_window_tokens = 128`

## Leave-One-Out

The frozen selector was evaluated with five leave-one-out ablations.

| Ablation | Hit@1 | Pairwise | SelAcc@10 | Delta vs baseline |
|---|---:|---:|---:|---:|
| `code_baseline_v1` | `59.28%` | `51.27%` | `61.74%` | `0.00pp` |
| `- prefix_best_window_quality` | `59.28%` | `50.37%` | `58.56%` | `-3.18pp` |
| `- post_reflection_recovery` | `59.28%` | `51.17%` | `60.99%` | `-0.75pp` |
| `- tail_variance` | `62.28%` | `51.40%` | `61.09%` | `-0.65pp` |
| `- head_tail_gap` | `61.68%` | `51.43%` | `61.93%` | `+0.19pp` |
| `- reflection_density` | `59.28%` | `51.10%` | `62.30%` | `+0.56pp` |

Interpretation:

- `prefix_best_window_quality` is the dominant feature and should stay central in any v2 line
- `post_reflection_recovery` is the clearest secondary contributor
- `tail_variance` still helps `SelAcc@10`, but trades against `Hit@1`
- `head_tail_gap` looks weak enough to demote or re-check before promotion
- `reflection_density` looks weakest and is the first removal candidate if the feature set needs pruning

## Small Grid

The coding-only structure grid scanned:

- `reflection_threshold ∈ {0.20, 0.25, 0.30, 0.35}`
- `lookback ∈ {8, 16, 24}`
- `prefix_fraction ∈ {0.10, 0.20, 0.30}`

Guardrail:

- `Pairwise >= 50%`
- `Hit@1` drop <= `1pp` versus frozen baseline

Best guarded config:

| Config | AUROC | Hit@1 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| `thr=0.30, lb=16, pf=0.30` | `50.16%` | `61.08%` | `50.67%` | `63.33%` |

Relative to the frozen baseline:

- `SelAcc@10`: `+1.59pp`
- `Hit@1`: `+1.80pp`
- `Pairwise`: `-0.60pp`

Interpretation:

- the coding line still prefers `reflection_threshold = 0.30`
- increasing `prefix_fraction` from `0.20` to `0.30` is the main structural gain in this grid
- this is good enough to keep as a **phase-2 candidate patch**
- this is **not** enough by itself to rename the baseline or declare a new final selector

## Disagreement Mining

Win/loss buckets were mined against the coding baselines.

Results:

- candidate feature family: `last-block instability`
- win bucket size: `39`
- loss bucket size: `50`
- `post_prefix_settle_shape` separation: `0.0018`
- `last_block_instability` separation: `0.4652`

Interpretation:

- `post-prefix settle` does not currently separate wins from losses enough to justify leading the next iteration
- `last-block instability` is the clearest next candidate family for a focused v2 experiment

## Transfer Gate

Ground-truth transfer summary:

| Selector | Code SelAcc@10 | Code Pairwise | Noncode Hit@1 | Noncode SelAcc@10 |
|---|---:|---:|---:|---:|
| `code_baseline_v1` | `61.74%` | `51.27%` | `47.43%` | `48.59%` |
| `tournament-copeland` | `59.21%` | `49.92%` | `68.73%` | `70.19%` |

Interpretation:

- the coding-tuned selector is better on code
- it transfers badly to non-code
- `tournament-copeland` remains the stronger non-code/default selector family
- the correct architecture is still a **coding-specialized module**, not a global replacement

## Blind DS / Qwen Checks

Blind coding-shape checks were run on:

- `DS-R1/lcb_v5`
- `Qwen3-4B/lcb_v5`

Top-1 agreement remained very low versus the generic baselines:

- `DS-R1/lcb_v5`: `11.98%` vs `min-confidence`, `0.00%` vs `tournament-copeland`
- `Qwen3-4B/lcb_v5`: `7.19%` vs `min-confidence`, `1.20%` vs `tournament-copeland`

Most important blind-shift finding:

- `Qwen3-4B/lcb_v5` shows a much stronger distribution shift than `DS-R1/lcb_v5`
- the strongest shifts are in:
  - `prefix_best_window_quality`
  - `head_tail_gap`
  - `tail_variance`
  - `last_block_instability_score`

This reinforces two points:

- DS and Qwen coding distributions should not be assumed identical
- the next coding feature iteration should explicitly check cross-model transfer before promotion

## Submission Artifacts

Submission artifacts after phase 2:

- frozen baseline patch:
  - `submission/BestofN/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_baseline_v1_lcb_patch.json`
- best-guarded experimental patch:
  - `submission/BestofN/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_phase2_candidate_thr030_lb16_pf030_lcb_patch.json`

The phase-2 experimental patch only replaces:

- `DS-R1/lcb_v5`
- `Qwen3-4B/lcb_v5`

All other submission slices stay unchanged from the base file.

## Exact Next Step

The next iteration should stay narrow:

1. build one `last-block instability` candidate family
2. rerun leave-one-out with that candidate replacing the weakest current feature
3. rerun the same small coding-only grid
4. rerun DS/Qwen transfer checks before any promotion decision

Do **not** reopen graph before this line is exhausted.
