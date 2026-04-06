# Science Baseline v1 Round 1

Date: `2026-04-05`

## Summary

This note records the first focused `GPQA/science` implementation round after the
coding line was frozen as `code_baseline_v1`.

Implemented files:

- `nad/core/selectors/science_dynamic_impl.py`
- `plugins/science_commitment_selector.py`
- `scripts/run_science_baseline_v1_round1.py`
- `scripts/patch_bestofn_submission_with_science_baseline_v1.py`

Primary result directory:

- `result/science_baseline_v1_round1_20260405_153510`

Round-1 decision:

- freeze `science_baseline_v1` as the current science-specialized baseline
- do **not** promote the late-window experimental candidate
- keep the science line separate from `earlystop_v3`
- keep the coding and science slices independently specialized

## Frozen Baseline

`science_baseline_v1` is intentionally control-like and recency-dominant.

Frozen feature family:

- `prefix_conf_mean_r`
- `recency_conf_mean_r`
- `late_worst_window_r`
- `late_recovery_r`

Frozen default weights:

- `prefix_conf_mean = 0.00`
- `recency_conf_mean = 1.00`
- `late_worst_window = 0.00`
- `late_recovery = 0.00`

Frozen structure defaults:

- `prefix_fraction = 0.40`
- `tail_fraction = 0.25`
- `recency_exp = 0.30`
- `window_tokens = 128`

Interpretation:

- the baseline is a dedicated science selector wrapper around the strongest
  current signal, not a claim that the late-window structure family is already
  validated enough to dominate the score
- the implementation still exposes the late-window feature family so a later
  round can promote it without redesigning the interface

## Main Comparison

On:

- `MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/gpqa/cache_neuron_output_1_act_no_rms_20251126_111853`

Round-1 metrics were:

| Selector | AUROC | Hit@1 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| `science_prefix_control` | `51.96%` | `64.65%` | `54.43%` | `62.85%` |
| `science_recency_control` | `53.86%` | `66.16%` | `58.71%` | `64.35%` |
| `science_baseline_v1` | `53.86%` | `66.16%` | `58.71%` | `64.35%` |
| `science_candidate_round1` | `54.01%` | `65.66%` | `58.79%` | `63.09%` |
| `tournament-copeland` | `52.20%` | `63.64%` | `54.69%` | `64.51%` |

Interpretation:

- `recency_conf_mean` is still the dominant science signal
- the late-window experimental candidate mildly improved `AUROC` and `Pairwise`
  but lost too much on `SelAcc@10`
- `tournament-copeland` stayed competitive on `SelAcc@10`, but the science
  baseline now leads it on `Hit@1` and `Pairwise`

## Candidate Leave-One-Out

Experimental candidate weights:

- `prefix_conf_mean = 0.16`
- `recency_conf_mean = 0.56`
- `late_worst_window = 0.16`
- `late_recovery = 0.12`

LOO result highlights:

- removing `recency_conf_mean` hurts most:
  - `SelAcc@10 -1.10pp`
  - `Hit@1 -2.02pp`
- removing `late_recovery` slightly hurts the candidate
- removing `late_worst_window` or `prefix_conf_mean` improves `Hit@1` but does
  not recover the `SelAcc@10` loss versus the frozen baseline

Interpretation:

- the new feature family is not useless, but it is not yet strong enough to
  displace the recency control as the promoted baseline

## Transfer Gate

GT transfer summary:

| Selector | Science Hit@1 | Science Pairwise | Science SelAcc@10 | Non-science SelAcc@10 |
|---|---:|---:|---:|---:|
| `science_baseline_v1` | `66.16%` | `58.71%` | `64.35%` | `69.10%` |
| `science_candidate_round1` | `65.66%` | `58.79%` | `63.09%` | `70.61%` |
| `tournament-copeland` | `63.64%` | `54.69%` | `64.51%` | `69.13%` |

Blind GPQA checks on `cache_test`:

- `science_candidate_round1` remained moderately close to the frozen baseline:
  - `DS-R1/gpqa` top-1 agreement: `46.97%`
  - `Qwen3-4B/gpqa` top-1 agreement: `42.93%`
- `tournament-copeland` stayed almost disjoint from the science baseline:
  - `DS-R1/gpqa` top-1 agreement: `0.51%`
  - `Qwen3-4B/gpqa` top-1 agreement: `0.51%`

Interpretation:

- the science line is meaningfully distinct from the generic pairwise selector
- the experimental late-window candidate does not look wildly unstable across
  DS/Qwen shape checks
- the promotion gate still fails because `SelAcc@10` regressed on the main GT
  science slice

## Exact Next Step

The next science round should stay narrow:

1. keep `science_baseline_v1` frozen as the recency-dominant baseline
2. treat the late-window family as an experimental branch only
3. rework the experimental family before promotion, likely by:
   - reducing or replacing `late_worst_window`
   - rechecking whether recovery should be computed over a larger final band
   - testing whether the candidate should mix with, not replace, the recency control
4. do **not** route this into `earlystop_v3` yet
5. do **not** patch submission slices with the experimental candidate

## Bottom Line

The repo now has a real, file-loaded `science_baseline_v1` path.

The important status change is:

- GPQA is no longer only a diagnosis target
- it now has a dedicated selector implementation
- the baseline is frozen
- the experimental structural science branch is implemented, evaluated, and
  explicitly kept out of promotion for now
