# EarlyStop SSL / Semisup Round1 Executive Summary

**Date:** 2026-04-16  
**Repo:** `/home/jovyan/work/NAD_Next`

This note is the shortest complete handoff for the EarlyStop SSL / semisupervised round1 work. It is meant to answer:

1. what was built,
2. what was actually run,
3. what was learned,
4. what was submitted,
5. what is worth doing next.

For full details, see:

- `docs/EARLYSTOP_SSL_SEMISUP_ROUND1_20260416.md`
- `docs/EARLYSTOP_SSL_SEMISUP_PAPERSTYLE_20260416.md`
- `docs/EARLYSTOP_SSL_SUBMISSION_PATCH_20260416.md`

## 1. Scope

The central question was:

> Can EarlyStop benefit from first learning a shared low-rank self-supervised basis, then fitting a lightweight discriminative head with limited labels?

This was evaluated under the repository’s canonical constraints:

- grouped `85/15` holdout by `dataset + problem_id`
- `split_seed=42`
- prefix-safe features only
- CPU-first only
- math/science prioritized
- coding treated as cheap confirmation only

## 2. What was implemented

### New code

- `nad/ops/earlystop_ssl.py`
  - basis fitting helpers
  - split-safe subsetting
  - linear heads
  - pseudo-label helpers

- `scripts/train_earlystop_ssl_basis.py`
  - basis study harness
  - shared vs task-specific vs no-basis comparisons
  - same-run vs same-problem ablations

- `scripts/train_earlystop_semisup.py`
  - hidden-label simulation
  - label-efficiency study
  - pseudo-label / agreement filtering sweeps

- `scripts/export_earlystop_ssl_submission.py`
  - export experimental EarlyStop JSON from SSL bundles

- `scripts/patch_earlystop_submission_with_overrides.py`
  - splice cache-level overrides from one EarlyStop submission into another
  - now also writes a manifest for patched outputs

### New documentation

- recon memo
- baseline repro docs
- round1 report
- paper-style report
- submission comparison / patch note
- this executive summary

## 3. What was actually trained

The round1 SSL line did **not** train ten separate slot models.

It trained **four anchor-specific models**:

- `10%`
- `40%`
- `70%`
- `100%`

Those four anchors were then mapped onto the public ten-slot EarlyStop output:

- `10 / 20 / 30` → `10%`
- `40 / 50 / 60` → `40%`
- `70 / 80 / 90` → `70%`
- `100` → `100%`

So the precise summary is:

> **trained anchors = 4; exported slots = 10**

## 4. What was run

### Baseline reproduction

The following lines were rechecked or reproduced:

- canonical `es_svd_ms_rr_r1`
- strong-feature noncoding reference
- cross-anchor frozen vs task-specific vs no-SVD comparisons
- simple low-complexity supervised baselines such as `raw+rank LR`

### SSL basis study

Views tested included:

- `raw ↔ rank`
- `token_only ↔ token_plus_traj`
- adjacent-anchor multiview pairing
- denoising / masking on full `raw+rank`

Methods tested included:

- ridge CCA
- reduced-rank regression / cross-SVD
- denoising SVD

### Semisupervised study

Budgets tested:

- `5%`
- `10%`
- `25%`
- `50%`
- `100%`

Pseudo-label thresholds tested:

- `0.90`
- `0.95`

### Live submission follow-up

Compared:

- live best `#279`
- SSL science patch `#289`

Then generated three hybrid patch candidates on top of `#279`.

## 5. Main findings

### Finding 1: science has a real low-label SSL opportunity

The strongest positive result of the entire round is:

> On science, SSL / semisupervised basis learning helps in the low-label regime (`5% / 10% / 25%`).

Representative improvements:

- `5%` labels: science baseline `0.7895 ± 0.0092`
- best SSL/semisup: `0.8128 ± 0.0094`

- `10%` labels: science baseline `0.7985 ± 0.0098`
- best SSL/semisup: `0.8136 ± 0.0129`

- `25%` labels: science baseline `0.8258 ± 0.0029`
- best SSL/semisup: `0.8296 ± 0.0078`

The best science families were:

- `token_only ↔ token_plus_traj` with `rrr`
- `denoise_full`

### Finding 2: math does not justify replacing simple supervised baselines

Math remained the cleanest negative result.

- best supervised baseline: `raw+rank LR = 0.9593`
- best SSL family: `denoise_full task rank16 = 0.9584`
- shared `denoise_full rank16 = 0.9580`

Interpretation:

- shared basis reuse is structurally plausible,
- but math does not reward this SSL route enough to justify replacing simple supervised training.

### Finding 3: coding is still a dead end in this feature space

Cheap confirmation stayed near random:

- `rank_only LR = 0.4958`
- `adjacent_cca same_problem rank16 = 0.4944`
- `raw+rank LR = 0.4747`

Interpretation:

> Do not spend significant additional CPU on coding SSL within the current prefix-safe token/trajectory feature family.

## 6. Important failure modes

### Raw↔rank-only SSL is weak

`raw_rank_cca` and related variants underperformed consistently and never became a serious candidate.

### Same-problem positives are risky

They can make science numbers look better in isolated ablations, but they are unstable and likely to flatten problem identity rather than learning robust correctness structure.

### Pseudo-label drift is real

In low-label science, `0.95` is a safer default threshold than `0.90`.

The semisupervised gain depends on:

- teacher quality,
- threshold choice,
- domain,
- label budget.

It is not a monotonic “more pseudo-labels is better” story.

## 7. Submission outcome

### Experimental SSL exports generated

- `submission/EarlyStop/earlystop_ssl_round1_shared_ms_experimental.json`
- `submission/EarlyStop/earlystop_ssl_round1_science_tokenpair16_patch.json`

The more defensible experimental export was:

- `earlystop_ssl_round1_science_tokenpair16_patch.json`

because only science had positive evidence.

### Live leaderboard result

The SSL science patch was submitted as:

- **#289** `earlystop_ssl_round1_science_tokenpair16_patch`

Live best remained:

- **#279** `es_svd_ms_rr_r1__coding_rotation_adapter_aggressive_tokenonly_late70100`

The key interpretation is:

- `#289` improved `gpqa` AUROC,
- but did not win the combined live ranking objective,
- so the correct next move is **hybrid patching**, not full replacement.

## 8. Hybrid patch outputs

Three patch candidates were generated using `#279` as the base:

- `...__ssl_gpqa_qwen_patch.json`
- `...__ssl_gpqa_both_patch.json`
- `...__ssl_gpqa_ds_patch.json`

Recommended order to test:

1. `Qwen gpqa` only
2. both `gpqa`
3. `DS gpqa` only

Reason:

- `Qwen3-4B/gpqa` had the strongest AUROC lift with the smallest SelAcc penalty.

## 9. What is worth continuing

### Continue

- science-focused low-label SSL
- frozen shared basis + light head
- `tokenpair_rrr@16`
- `denoise_full@16`
- high-confidence pseudo-labeling
- agreement filtering
- threshold near `0.95`
- cache-level hybrid patching instead of whole-submission replacement

### Stop or deprioritize

- raw↔rank SSL as a main research direction
- same-problem positives as the default relation
- coding SSL in the current feature family
- replacing the live best end-to-end with the current SSL science patch

## 10. Recommended next step

The single best next step is:

> submit the `Qwen3-4B/gpqa`-only hybrid patch on top of live best `#279`, then decide whether science SSL has operational value as a selective override source rather than a new global backbone.

If that hybrid also fails to improve the live primary score, the correct conclusion is not “SSL never helped”, but:

> the science SSL gain is real offline and locally live-relevant, but still too narrow or too poorly calibrated to justify production promotion.
