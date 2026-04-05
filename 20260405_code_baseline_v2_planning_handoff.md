# 2026-04-05 Code Baseline v2 Planning Handoff

## 1. Current status

The coding line now has two distinct states:

- frozen baseline:
  - `code_baseline_v1`
- experimental next candidate:
  - best guarded phase-2 config `thr=0.30 / lb=16 / pf=0.30`

Do not collapse these two states together.

Current interpretation:

- `code_baseline_v1` is the stable coding baseline
- the phase-2 grid winner is only a candidate patch
- the next real model-design step is feature revision, not another broad sweep

## 2. Source of truth

Primary result directory:

- `result/code_baseline_v1_phase2_20260405_080141`

Primary docs:

- `docs/CODE_SELECTOR_VALIDATION_20260405.md`
- `docs/CODE_BASELINE_V1_PHASE2_20260405.md`
- `20260405_code_baseline_v1_submission_handoff.md`

Relevant implementation files:

- `nad/core/selectors/code_dynamic_impl.py`
- `plugins/prefix_saturation_selector.py`
- `scripts/run_code_baseline_v1_phase2.py`
- `scripts/patch_bestofn_submission_with_code_baseline_v1.py`

## 3. What phase 2 established

### Frozen baseline

`code_baseline_v1` on the target coding cache:

- `AUROC 0.5027`
- `Hit@1 0.5928`
- `Pairwise 0.5127`
- `SelAcc@10 0.6174`

### 5-feature leave-one-out

Main takeaways:

- removing `prefix_best_window_quality` hurts most:
  - `SelAcc@10 -3.18pp`
- removing `post_reflection_recovery` mildly hurts:
  - `SelAcc@10 -0.75pp`
- removing `tail_variance` mildly hurts `SelAcc@10`, but helps `Hit@1`
- removing `head_tail_gap` slightly helps `SelAcc@10`
- removing `reflection_density` also slightly helps `SelAcc@10`

Conclusion:

- keep `prefix_best_window_quality`
- keep `post_reflection_recovery`
- probably keep `tail_variance`
- treat `head_tail_gap` as optional / weak
- treat `reflection_density` as the weakest current feature

### Small coding-only grid

Best guarded config:

- `reflection_threshold = 0.30`
- `reflection_lookback_slices = 16`
- `prefix_fraction = 0.30`

Metrics:

- `AUROC 0.5016`
- `Hit@1 0.6108`
- `Pairwise 0.5067`
- `SelAcc@10 0.6333`

Interpretation:

- keep `0.30` as the coding reflection threshold
- the main grid gain is from a larger prefix slice, not from changing the reflection line
- this config is strong enough for an experimental submission patch
- it is not enough to skip the next feature-design step

### Disagreement mining

Chosen next feature family:

- `last-block instability`

Reason:

- `post_prefix_settle_shape` separation is effectively zero
- `last_block_instability` separation is large enough to justify a direct follow-up feature experiment

### Transfer gate

2×2 summary:

- `code_baseline_v1` wins on code
- `tournament-copeland` wins decisively on non-code

Blind coding-shape checks:

- DS/Qwen agree very little with generic baselines
- `Qwen3-4B/lcb_v5` shows much larger feature shift than `DS-R1/lcb_v5`

Conclusion:

- keep the coding selector family specialized
- do not reuse it as a global selector
- explicitly test DS/Qwen transfer on the next iteration

## 4. Exact next experiment

The next experiment should be a focused `code_v2_candidate`, not a wide refactor.

Target feature set:

- keep `prefix_best_window_quality`
- keep `post_reflection_recovery`
- keep `tail_variance`
- add one `last-block instability` feature family
- consider dropping or zeroing `reflection_density`
- consider demoting `head_tail_gap` unless the new feature makes it useful again

Recommended procedure:

1. implement exactly one `last-block instability` family
2. compare against frozen `code_baseline_v1`
3. rerun 5-feature-style leave-one-out or the equivalent reduced ablation
4. rerun the same grid:
   - `reflection_threshold ∈ {0.20, 0.25, 0.30, 0.35}`
   - `lookback ∈ {8, 16, 24}`
   - `prefix_fraction ∈ {0.10, 0.20, 0.30}`
5. rerun transfer checks on:
   - GT code vs non-code
   - blind `DS-R1/lcb_v5`
   - blind `Qwen3-4B/lcb_v5`

## 5. Promotion gate

Do not promote a new coding selector unless all of these hold:

- at least one of `DS-R1/lcb_v5` or `Qwen3-4B/lcb_v5` gets a clear `SelAcc@10` gain
- the other coding slice does not regress materially
- `Pairwise` stays at or above `0.50`
- `Hit@1` does not show a systematic drop larger than `1pp`

If these conditions are not met:

- keep `code_baseline_v1` as the only promoted coding baseline
- keep any new result as an experimental patch only

## 6. What not to do

- do **not** reopen graph features
- do **not** merge this into `earlystop_v3`
- do **not** sweep in unrelated worktree files during commit
- do **not** replace the global submission stack with the coding selector

## 7. Submission files

Frozen coding patch:

- `submission/BestofN/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_baseline_v1_lcb_patch.json`

Experimental phase-2 coding patch:

- `submission/BestofN/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_phase2_candidate_thr030_lb16_pf030_lcb_patch.json`

The experimental patch is the correct starting point if someone wants to inspect coding-only submission deltas without changing the frozen baseline definition.
