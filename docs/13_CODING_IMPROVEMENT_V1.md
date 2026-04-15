# 13 — Coding Domain Improvement v1

**Date**: 2026-04-13
**Status**: Phase 1A + Phase 2 complete. Stop criterion NOT met (best AUROC 53.74% < 65% target).

---

## Setup

- **Cache**: `MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/livecodebench_v5/cache_neuron_output_1_act_no_rms_20251127_032808`
- **Samples**: 10,688 · **Problems**: 167 · **Pos rate**: 58.7%
- **Split**: 5-fold GroupKFold by `problem_id` (no holdout — 167 problems too small)
- **Position**: 1.0 (full response)
- **Scripts**: `scripts/baselines/train_coding_dummies.py`, `SVDomain/experiments/run_coding_improvement_v1.py`

---

## Phase 1A — Single-Feature Rule Baselines

> _Do any individual LM stats discriminate code correctness?_

| Signal | AUROC | AUPRC | Brier | Logloss |
|--------|------:|------:|------:|--------:|
| `tok_gini_prefix` | 0.521 | 0.598 | 0.317 | 0.866 |
| `response_token_count` | **0.535** | 0.605 | 0.293 | 0.918 |
| `tok_logprob_prefix` | 0.512 | 0.597 | 0.308 | 0.841 |
| `tok_neg_entropy_prefix` | 0.512 | 0.595 | 0.308 | 0.843 |
| `tok_conf_early_vs_late_gap` | 0.508 | 0.613 | 0.284 | 0.783 |
| `tok_conf_prefix` | 0.495 | 0.586 | 0.350 | 0.964 |

**Leakage check**: `Pearson(response_token_count, label) = 0.064` (p=4.6e-11). Below 0.15 threshold → **SAFE** (not a shortcut exploit).

**Finding**: All single features are near-chance (≤ 53.5% AUROC). `response_token_count` is the strongest single predictor but is a weak structural proxy, not a leakage artifact.

---

## Phase 2 — Novel Feature Engineering

### Branch A: Code-Structural Tier-1 Features

New features derived from token IDs only (no tokenizer required):
- `response_token_count`, `at_max_tokens`, `trigram_repetition_rate`, `unique_token_fraction`
- `tok_conf_early_vs_late_gap`, `tok_conf_section_{early,mid,late}`

| Model | Features | AUROC | AUPRC | Brier |
|-------|----------|------:|------:|------:|
| XGBoost | Tier-1 only | **0.537** | **0.630** | 0.262 |
| XGBoost | base-30 + Tier-1 | 0.510 | 0.593 | 0.272 |
| SVD-LR | base-30 + Tier-1 | 0.488 | 0.592 | 0.249 |
| XGBoost | base-30 (reference) | 0.504 | 0.587 | 0.270 |
| XGBoost | base-30 within-group-rank | 0.489 | 0.581 | 0.252 |
| XGBoost | base-30 + Tier-1 within-group-z | 0.404 | 0.553 | 0.310 |

**Finding**: Tier-1 features alone (XGBoost 53.7%) slightly beat base-30 (50.4%), but concatenation hurts. Within-group normalisation is catastrophic — the inter-problem variance seems to be the only signal, which is destroyed by z-scoring within problem.

### Branch B: Confidence-Curve Derivative Features

12 derivative features: `{conf,gini,entropy}_d{1,abs_d1,abs_d2,abs_d1_full_minus_tail}_tail_mean`.

| Model | Features | AUROC | AUPRC | Brier |
|-------|----------|------:|------:|------:|
| XGBoost | derivative only | 0.504 | 0.589 | 0.256 |
| XGBoost | base-30 + deriv | 0.498 | 0.583 | 0.272 |
| XGBoost | base-30 + Tier-1 + deriv | 0.506 | 0.589 | 0.274 |
| SVD-LR | base-30 + deriv | 0.474 | 0.571 | 0.251 |

**Finding**: Derivative features add no signal. All combinations regress toward base-30 (50.4%) or worse.

### Branch C: Pairwise Hard-Negative Contrastive SVM

Within-group `(correct − incorrect)` feature diffs, mirrored; LinearSVC.

| Model | Features | AUROC | AUPRC | Brier |
|-------|----------|------:|------:|------:|
| Pairwise SVM | base-30 + Tier-1 | **0.513** | 0.609 | 0.274 |
| Pairwise SVM | all features | 0.501 | 0.596 | 0.286 |
| Pairwise SVM | Tier-1 only | 0.494 | 0.588 | 0.328 |

**Finding**: Pairwise contrastive objective yields marginal gain (51.3% vs 50.4% pointwise) on base+Tier-1 but doesn't provide transformative improvement. The pairwise formulation confirms the within-group signal is weak regardless of loss function.

---

## Consolidated Ranking (5-fold GroupKFold, position 1.0)

| Rank | Method | AUROC | Δ vs base-30 |
|-----:|--------|------:|-------------:|
| 1 | XGBoost Tier-1 only | **0.537** | +3.3pp |
| 2 | Pairwise SVM base-30 + Tier-1 | 0.513 | +0.9pp |
| 3 | XGBoost base-30 + Tier-1 | 0.510 | +0.6pp |
| 4 | XGBoost all (base+tier1+deriv) | 0.506 | +0.2pp |
| 5 | XGBoost base-30 (reference) | 0.504 | — |

---

## Protocol Note: GroupKFold vs 85/15 Split

The tree_baselines.csv reference (55.58%) used `GroupShuffleSplit(0.85, 0.15)` — a single 85/15 holdout. Our 5-fold GroupKFold gives smaller training sets (~80%/fold) and more conservative estimates. The base-30 XGBoost in the same GroupKFold evaluation yields **50.4%** vs **55.6%** under the 85/15 split.
The comparison is **within this study** (all methods evaluated consistently under GroupKFold).

---

## Key Conclusions

1. **Feature-label mismatch confirmed** for all tested feature types. None of the three branches achieve breakthrough improvement.
2. **Response length** (token count) is the single strongest predictor (AUROC 53.5%), with verified low leakage (Pearson r=0.064). It likely proxies problem-level difficulty, not sample-level correctness quality.
3. **Derivative features are useless** for coding domain — they add noise, not signal.
4. **Within-group normalisation destroys signal** — the inter-problem variation is the only detectable structure.
5. **Pairwise contrastive objective** provides marginal improvement (+0.9pp) confirming the signal is weak, not just the loss.
6. **Stop criterion not met**: Best AUROC 53.7% < 65% target. Brier 0.262 > 0.22 target.

---

## Stop Criteria Assessment

| Criterion | Target | Best Achieved | Status |
|-----------|--------|---------------|--------|
| AUROC | > 65% | 53.7% | ❌ FAIL |
| AUROC σ < 3% | — | measured, OK | — |
| Brier < 0.22 | < 0.22 | 0.262 | ❌ FAIL |
| Length Pearson < 0.15 | < 0.15 | 0.064 | ✅ PASS |

**Verdict**: Phase 3 (ensemble) is **not warranted** — no individual branch reached a level where ensembling could plausibly close the gap to 65%.

---

## Failure Taxonomy

| Failure | Observation | Interpretation |
|---------|-------------|----------------|
| LM confidence near-random | All confidence metrics ≤ 52% AUROC | Model equally confident on correct and incorrect code |
| Derivative features useless | Branch B adds no signal | Trajectory dynamics not discriminative for code |
| Structural features weak | Token count 53.5%, others <52% | Length is a problem difficulty proxy, not correctness detector |
| Pairwise objective marginal | +0.9pp max | The limitation is the feature space, not the loss function |
| Within-group z-score catastrophic | 40.4% AUROC | Inter-problem variation is the only exploitable structure |

---

## Next Steps

The offline judge approach with LM activation/token statistics is fundamentally limited for LCBv5. Potential directions not yet explored:
- **Execution-based signals**: Does the code run? Partial scoring from test case outputs.
- **Test-coverage-aware prompting**: Prompt model to self-verify, extract confidence from that.
- **Cross-problem calibration using difficulty priors**: Estimate problem difficulty from historical pass rates, use as prior.
- **Model-specific reasoning signals**: Use reasoning trace structure (code blocks, comments, iterations) as features.
