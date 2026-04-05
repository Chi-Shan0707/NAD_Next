# 2026-04-05 Code Baseline v1 Submission Handoff

## 1. Current state

The coding-specialized selector is no longer just a plugin experiment.

For current repo purposes, treat it as:

- `code_baseline_v1`

The underlying selector remains file-loaded:

- `file:/home/jovyan/work/NAD_Next/plugins/prefix_saturation_selector.py:PrefixSaturationSelector`

But the implementation is now frozen enough to support a submission artifact for the coding slice.

## 2. Frozen baseline definition

Core files:

- `nad/core/selectors/code_dynamic_impl.py`
- `plugins/prefix_saturation_selector.py`

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

Interpretation:

- this is a **coding-only specialized baseline**
- it is **not** the global default selector
- it is **not** merged into `earlystop_v3`
- it is **not** a graph-topology branch

## 3. Validation status

Target validation cache:

- `MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/livecodebench_v5/cache_neuron_output_1_act_no_rms_20251127_032808`

Main deterministic ranking table:

| Selector | Hit@1 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|
| `min-confidence` | `59.88%` | `49.51%` | `52.15%` |
| `tournament-copeland` | `58.08%` | `49.92%` | `59.27%` |
| `PrefixSaturationSelector` | `59.28%` | `51.27%` | `61.70%` |

Why this matters:

- it wins on the actual slice objective: `SelAcc@10`
- it also has the best `Pairwise`
- it only gives up `0.60pp` on `Hit@1` versus the best baseline

Regression after freezing the shared scorer:

- rerun `nad.cli analyze` stayed identical for plugin choices on all `167/167` problems
- rerun `nad.cli accuracy` stayed:
  - `min-confidence`: `100/167`
  - `tournament-copeland`: `104/167`
  - `PrefixSaturationSelector`: `99/167`

## 4. Submission artifact

Base submission:

- `submission/BestofN/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank.json`

Patched coding submission:

- `submission/BestofN/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_baseline_v1_lcb_patch.json`

Patch scope:

- replace only `DS-R1/lcb_v5`
- replace only `Qwen3-4B/lcb_v5`
- keep all other cache keys unchanged

Exporter script:

- `scripts/patch_bestofn_submission_with_code_baseline_v1.py`

Patch logic:

- score each coding problem with the same code-baseline primary score used by the selector
- order runs by that score
- break exact ties with the same local medoid-style `D`-sum rule
- convert each problem back into the repo’s `1..100` rank scale

## 5. What not to do

- do **not** interpret this as evidence that the selector should replace the whole global stack
- do **not** reopen graph features in the next step
- do **not** merge this into `earlystop_v3` unless scope is intentionally expanded
- do **not** let follow-up commits sweep in unrelated worktree files

## 6. Exact next experiments

The next work should be limited to these four items:

1. **5-feature leave-one-out**
   - freeze the current baseline and measure drop from removing each feature one at a time
2. **small coding-only structure grid**
   - `reflection_threshold ∈ {0.20, 0.25, 0.30, 0.35}`
   - `lookback ∈ {8, 16, 24}`
   - `prefix_fraction ∈ {0.10, 0.20, 0.30}`
   - do not retune weights before this grid is understood
3. **cross-domain transfer gate**
   - code-tuned selector on code
   - code-tuned selector on math/science/global
   - non-code/global selector on code
   - summarize as a `2×2` transfer table
4. **keep graph out**
   - only revisit graph if the prefix/tail/reflection line stops producing useful signal

## 7. Bottom line

This is the point where the repo first has a credible, operational coding structure baseline.

The important status change is:

- coding structure is no longer just a hypothesis write-up
- it is now a validated selector
- it is now frozen as `code_baseline_v1`
- it now has a submission-ready `lcb_v5` patch path
