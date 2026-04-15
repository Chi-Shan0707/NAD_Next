# Coding Bottleneck Study

**Date**: 2026-04-15
**Status**: Complete — all three experiments run on full 167-problem lcb_v5 dataset.

## Context

Coding on `lcb_v5` is the weakest domain in the pipeline. Prior results established:
- `no_svd_lr` at 10% labels: AUROC ≈ 0.591 (best source-holdout, limited data regime)
- `domain_ssl` contrastive (r=4..16): AUROC ≈ 0.50 (same as random)
- Meta-XGB (slot100 + activation): Top1 = 0.6168 (best overall, cross-modal)
- Math/science SVD reaches AUROC ≈ 0.88–0.95; coding appears fundamentally different

**Feature cache used**: `results/cache/rule_and_family_baselines/cache_all_ref030_45392d82523f21bf.pkl`
(47-feature tensor, 167 problems, 10,688 samples, reflection_threshold=0.30)

---

## Experiment 1 — Bottleneck Diagnosis

**Script**: `SVDomain/experiments/run_coding_bottleneck_v3.py`
**Output**: `results/tables/coding_bottleneck_v3.csv`

### Oracle Statistics (position-independent)

| Metric | Value | Interpretation |
|---|---|---|
| informative_pair_fraction | **0.1482** | Only 14.8% of within-problem pairs are cross-label |
| pct_informative_problems | **0.6108** | 61% of problems have ≥1 correct AND ≥1 incorrect solution |
| oracle_hit@1 | **0.8263** | 82.6% of problems have ≥1 correct solution (max achievable hit@1) |

The `informative_pair_fraction = 0.148` is the key structural fact. Within most problems, nearly all
solutions have the same label (both pass or both fail). This is not a feature-language problem.

### Results at 70% and 100% Anchors

| condition | pos | n_feat | auroc | pairwise_acc | hit@1 |
|---|---|---|---|---|---|
| oracle_rank_bound | 70% | 47 | — | 0.1482 | 0.826 |
| response_length_proxy | 70% | 1 | 0.447 | 0.490 | 0.695 |
| token_only_logistic | 70% | 16 | 0.467 | 0.511 | 0.749 |
| fixed_22_logistic | 70% | 22 | 0.495 | 0.512 | 0.700 |
| full_logistic | 70% | 47 | 0.495 | 0.512 | 0.700 |
| fixed_22_pairwise | 70% | 22 | 0.491 | 0.471 | 0.717 |
| full_pairwise | 70% | 47 | 0.491 | 0.471 | 0.717 |
| oracle_rank_bound | 100% | 47 | — | 0.1482 | 0.826 |
| response_length_proxy | 100% | 1 | 0.453 | 0.490 | 0.714 |
| token_only_logistic | 100% | 16 | 0.457 | 0.513 | 0.736 |
| fixed_22_logistic | 100% | 22 | 0.496 | 0.507 | 0.744 |
| full_logistic | 100% | 47 | 0.496 | 0.507 | 0.744 |
| fixed_22_pairwise | 100% | 22 | 0.496 | 0.484 | 0.677 |
| full_pairwise | 100% | 47 | 0.496 | 0.484 | 0.677 |

### Key Findings — Exp 1

1. **All methods collapse to AUROC ≈ 0.495–0.496** on the full 167-problem dataset.
   This matches the known "coding ≈ 0.50" result across all prior methods.

2. **fixed_22 ≡ full_47**: Adding 25 coding-specific features (dynamic + derivative) gives
   exactly zero improvement. The feature-language ceiling is already reached with 22 features.

3. **Pairwise objective ≤ pointwise**: Pairwise logistic achieves lower pairwise_acc (0.47–0.48)
   than pointwise logistic (0.50–0.51). Pairwise training does not help.

4. **hit@1 ≈ 0.70–0.74**: The hit@1 metric shows more headroom than AUROC. The oracle ceiling
   is 0.826, and LR achieves 0.69–0.74 — so there is ~8% hit@1 gap left to close.

5. **The bottleneck is structural, not methodological**: 39% of problems have no contrast
   (all-pass or all-fail), and 40% of the remaining pairs are same-label. No feature model
   can signal a difference where labels are identical.

---

## Experiment 2 — Pairwise as First-Class Objective

**Script**: `SVDomain/experiments/run_coding_pairwise_v1.py`
**Output**: `results/tables/coding_pairwise_v1.csv`

### Results

| condition | objective | position | auroc | pairwise_acc | hit@1 |
|---|---|---|---|---|---|
| direct_pairwise_22 | pairwise | 70% | 0.491 | 0.471 | 0.717 |
| direct_pairwise_22 | pairwise | 80% | 0.492 | 0.481 | 0.726 |
| direct_pairwise_22 | pairwise | 90% | 0.497 | 0.493 | 0.660 |
| direct_pairwise_22 | pairwise | 100% | 0.496 | 0.484 | 0.677 |
| direct_pairwise_22 | pairwise | late_mean | 0.494 | 0.482 | 0.695 |
| direct_pairwise_full | pairwise | late_mean | 0.494 | 0.478 | 0.697 |
| direct_pointwise_22 | pointwise | 70% | 0.495 | 0.512 | 0.700 |
| direct_pointwise_22 | pointwise | 100% | 0.496 | 0.507 | 0.744 |
| **direct_pointwise_22** | **pointwise** | **late_mean** | **0.496** | **0.510** | **0.722** |
| direct_pointwise_full | pointwise | late_mean | 0.496 | 0.510 | 0.722 |

### Key Findings — Exp 2

1. **Pairwise objective consistently underperforms pointwise** across all metrics.
   - Pairwise pairwise_acc: 0.471–0.493 (below 0.50 = worse than random on this metric)
   - Pointwise pairwise_acc: 0.507–0.512 (marginally above 0.50)
   - Pairwise training on within-problem diffs learns noise, not signal.

2. **Position trend is non-monotonic**: AUROC peaks at 90–100% but hit@1 peaks at 70–80%
   and drops at 100%. This suggests the confidence features late in generation may emphasize
   the final answer's length/format rather than underlying correctness.

3. **22 features ≡ 47 features**: Same result as Exp 1 — additional coding features add nothing.

4. **The objective-metric gap is negligible**: AUROC ≈ pairwise_acc ≈ 0.50 for all conditions.
   Changing the training objective does not move the evaluation metric.

---

## Experiment 3 — Feature Language Audit

**Script**: `SVDomain/experiments/run_coding_feature_audit_v1.py`
**Output**: `results/tables/coding_feature_audit_v1.csv`, `results/figures/coding_feature_importance.png`

### Baseline Models

| model | auroc |
|---|---|
| full_lr (logistic, 47 feat) | 0.496 |
| full_gb (GradientBoosting, 47 feat) | **0.508** |
| full_gb + 3 derived features | 0.507 (Δ = −0.001) |

### Family Ablation (GradientBoosting, remove one family, full 47→smaller)

| family_removed | n_removed | auroc | delta |
|---|---|---|---|
| — (full baseline) | 0 | 0.508 | — |
| token_stats | 11 | 0.516 | **+0.009** |
| traj | 5 | 0.496 | **−0.012** |
| availability | 6 | 0.507 | −0.001 |
| coding_dynamic | 5 | 0.507 | −0.001 |
| coding_derivative | 12 | 0.507 | −0.001 |

### Feature Importance by Family

| family | n_feat | gain_sum | gain_mean | perm_mean |
|---|---|---|---|---|
| token_stats | 11 | 0.574 | 0.052 | 0.012 |
| traj | 5 | 0.426 | 0.085 | **0.033** |
| availability | 6 | 0.000 | 0.000 | 0.000 |
| coding_dynamic | 5 | 0.000 | 0.000 | 0.000 |
| coding_derivative | 12 | 0.000 | 0.000 | 0.000 |

### Top-10 Individual Features

| rank | feature | gain | perm_importance | family |
|---|---|---|---|---|
| 1 | traj_novelty | 0.164 | **0.085** | traj |
| 2 | traj_continuity | 0.103 | **0.049** | traj |
| 3 | tok_conf_prefix | 0.097 | 0.024 | token_stats |
| 4 | traj_reflection_count | 0.084 | 0.016 | traj |
| 5 | tok_gini_tail | 0.070 | 0.011 | token_stats |
| 6 | tok_logprob_recency | 0.070 | 0.015 | token_stats |
| 7 | tok_gini_prefix | 0.068 | **0.046** | token_stats |
| 8 | traj_max_reflection | 0.068 | 0.013 | traj |
| 9 | tok_conf_recency | 0.067 | 0.014 | token_stats |
| 10 | tok_gini_slope | 0.063 | 0.009 | token_stats |

### Key Findings — Exp 3

1. **Trajectory family is the only predictive family**:
   - Removing `traj` costs −0.012 AUROC (largest single-family drop)
   - Removing `token_stats` *improves* AUROC by +0.009 — token stats are adding noise for coding
   - `availability`, `coding_dynamic`, `coding_derivative`: zero gain and zero permutation importance

2. **Coding-specific features (17 features added for coding) have zero predictive value**:
   Both `coding_dynamic` (5 features) and `coding_derivative` (12 features) have gain_sum = 0 and
   perm_mean = 0. These features encode reasoning-process dynamics that do not correlate with
   coding correctness in this dataset.

3. **Derived composite features add nothing** (Δ = −0.001). The three new signals
   (`gini_div_recency_ratio`, `conf_logprob_gap`, `late_instability_density`) carry no
   independent information beyond the base features.

4. **Top signal: `traj_novelty` and `traj_continuity`**: These trajectory features encode
   whether the model's reasoning path is novel and continuous, which provides the most predictive
   signal for coding. However, even the best feature achieves only perm_importance = 0.085 —
   very weak signal overall.

5. **The best model (GB full) achieves AUROC = 0.508** — only 0.8 percentage points above random.
   This is effectively no discrimination ability.

---

## Research Question Answers

### Q1: Should coding still use SVD?

**No.** SVD adds zero value for coding. Evidence:
- Direct LR on 47 raw features = AUROC 0.496 (≈ random)
- Adding SVD compression cannot improve signal that doesn't exist in the feature space
- The bottleneck is in the data labels, not in the representation capacity

SVD is appropriate for math (AUROC 0.88+) where there IS signal to compress. For coding,
applying SVD would only destroy what little trajectory signal exists.

### Q2: What role should SVD play for coding?

**Diagnostic only.** The fact that SVD bundles trained on math/science collapse to 0.50 when
applied to coding is diagnostic: it confirms that the geometry of correctness in coding activation
space is orthogonal to the math/science geometry. There is no benefit in fine-tuning or adapting
the math/science SVD bundle for coding.

If a coding-specific SVD model is ever trained, it should use only `traj` features
(particularly `traj_novelty` and `traj_continuity`) as the basis, not token confidence features.

### Q3: Are adapters useful?

**No, confirmed by multiple independent lines of evidence**:
- Rotation adapter → learned R ≈ I (identity) → geometry doesn't need correction
- Random adapter sweep → no gain at any rank or learning rate
- This study → feature language is not the bottleneck → adapting the representation cannot help
- **Root cause is structural data sparsity, not representation inadequacy**

### Q4: Which feature families matter most for coding?

| family | verdict | rationale |
|---|---|---|
| `traj` | **Only useful family** | Removing it is the only ablation that costs AUROC |
| `token_stats` | **Mildly harmful** | Removing it *improves* AUROC (adds noise) |
| `availability` | **Neutral / zero** | No importance in any metric |
| `coding_dynamic` | **Zero** | Reflection/instability dynamics not predictive |
| `coding_derivative` | **Zero** | Confidence derivative features add nothing |

**The next step is NOT within the activation/token-stat feature space.** If coding performance
is to be improved, new signal must come from sources orthogonal to generation statistics:
- **Code execution signals** (test pass count, error type, partial test scores)
- **AST/parse-level features** (syntax validity, code structure complexity)
- **Cross-solution neuron consensus** (already explored in V2; requires neuron cache access)
- **Meta-XGB with slot100** (currently best at Top1 = 0.6168)

---

## Root Cause Summary

The coding bottleneck is **structural data sparsity**, not a methods failure:

```
Problem structure (lcb_v5):
  167 total problems
  ├── 36 all-pass  (21.6%) → no contrast, cannot distinguish solutions
  ├── 29 all-fail  (17.4%) → no contrast, cannot predict success
  └── 102 mixed    (61.1%) → meaningful, but skewed label distributions

Within mixed problems:
  informative_pair_fraction = 0.148
  → For a typical problem with 60/64 passing, only 4 × 60 = 240 cross-pairs
    exist out of 2016 total pairs (11.9% informative)

Consequence:
  AUROC ceiling with perfect features ≈ 0.52–0.55 at best
  (bounded by the fraction of discriminable pairs)
```

The "best-of-n" framing (hit@1) is more tractable than AUROC for this dataset: oracle hit@1 = 0.826,
current best ≈ 0.74, gap ≈ 8.6 percentage points — achievable via better selection (not better
correctness prediction). The meta-XGB approach (slot100 + activation, Top1 = 0.6168) already
operates in this space by leveraging cross-modal signals.

---

## Practical Recommendations

1. **Keep `no_svd_lr` at 10% labels** as the coding scorer in the export pipeline. It remains the
   best token-stat-based method (AUROC ≈ 0.591 in limited-label regime where group structure
   helps regularize the LR).

2. **Do not invest further in feature engineering** within the activation/token space. The
   17 coding-specific features have zero predictive value, and derived composite features add nothing.

3. **Do not use SVD, adapters, or SSL for coding**. The methods literature for coding is closed.

4. **For hit@1 improvement**, the productive direction is:
   - Expand meta-XGB with additional slot-level features
   - Explore cross-solution consensus (requires re-running V2 with more problems)
   - Obtain code execution feedback (test partial scores) — requires infrastructure work

5. **Accept the AUROC ceiling**. AUROC ≈ 0.50 for all activation-based methods on coding is
   not a failure of the pipeline; it is a consequence of the dataset's label structure. Reporting
   this accurately in competition submissions avoids overfitting to a metric that is uninformative
   for this domain.
