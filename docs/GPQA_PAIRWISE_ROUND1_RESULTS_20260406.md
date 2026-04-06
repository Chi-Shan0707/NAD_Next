# GPQA Group Pairwise Round 1 — Results

Date: `2026-04-06`
Cache: `DeepSeek-R1-0528-Qwen3-8B/gpqa/cache_neuron_output_1_act_no_rms_20251126_111853`
Script: `scripts/run_gpqa_pairwise_round1.py`
Results JSON: `result/gpqa_pairwise_round1_20260406.json`

---

## Decision: NO-PROMOTE

`gpqa_pairwise_round1` does not advance to `gpqa_pairwise_v1`.

Failed thresholds:
- **SelAcc@10**: 62.38% ≤ threshold 64.35% (−1.97pp vs baseline)
- **Hit@1**: 63.64% < guardrail 65.16% (−2.52pp vs baseline)

Passed guardrails:
- **AUROC**: 55.66% > 53.86% (+1.80pp vs baseline) ✓
- **Pairwise**: 61.15% > 58.21% (+2.44pp vs baseline) ✓

---

## Metrics Summary

| Selector              | AUROC   | Hit@1   | Pairwise | SelAcc@10 |
|-----------------------|---------|---------|----------|-----------|
| `science_baseline_v1` | 53.86%  | 66.16%  | 58.71%   | 64.35%    |
| `tournament-copeland` | 52.20%  | 63.64%  | 54.69%   | 64.51%    |
| `gpqa_pairwise_round1`| **55.66%** | 63.64%  | **61.15%** | 62.38%    |

N problems: 198 · N samples: 12,672 · Protocol: LOPO (leave-one-problem-out)

---

## Interpretation

### What went right

**AUROC +1.80pp** (53.86% → 55.66%): The pairwise model discriminates correct/wrong runs
better in a global ranking sense.  The Bradley-Terry framework with 6-dim difference
features captures a signal that the simple weighted-rank baseline misses.

**Pairwise accuracy +2.44pp** (58.71% → 61.15%): On a per-problem basis, the model more
reliably assigns a higher score to correct runs than to wrong runs.  This is the metric
the model is directly optimised for.

### What went wrong

**Hit@1 −2.52pp** (66.16% → 63.64%): The model makes worse top-1 selections despite
better pairwise ranking.  The Bradley-Terry mean-probability score averages votes from
all N−1 opponents.  In GPQA groups where *most* runs are wrong (typical), the correct
runs may accumulate only moderately higher scores than the best-performing wrong runs,
and a different wrong run may edge out due to noise.

**SelAcc@10 −1.97pp** (64.35% → 62.38%): The top-10% window also degrades, consistent
with the same averaging effect spreading pairwise confidence too smoothly.

### Root cause hypothesis

The 6-dim feature set (`dc_z`, `dc_r`, `reflection_count_r`, `prefix_conf_mean_r`,
`recency_conf_mean_r`, `late_recovery_r`) provides a useful *global ranking signal*
(AUROC, pairwise) but the linear Bradley-Terry classifier cannot learn the
non-linear threshold behaviour that matters for the *top slot*.

A pure recency-dominant score (`recency_conf_mean`, weight=1) is a hard, decisive
rule that empirically finds the correct run at rank-1 more often, even if its
global AUC is lower.

---

## Feature Analysis

All 6 features are group-normalised within each problem before differencing.
The pairwise model effectively learns a linear combination of feature-difference weights.

| Feature            | Role                                         |
|--------------------|----------------------------------------------|
| `dc_z`             | DeepConf quality z-score — absolute position |
| `dc_r`             | DeepConf quality rank — relative ordering    |
| `reflection_count_r` | Proxy for deliberation depth               |
| `prefix_conf_mean_r` | Early-token confidence (prefix quality)    |
| `recency_conf_mean_r`| Recency-weighted confidence (decisive!)    |
| `late_recovery_r`  | Late-window recovery (tail quality)          |

`recency_conf_mean` is the dominant feature in `science_baseline_v1` (weight=1, all
others=0), so including it alongside other features in a logistic regression model is
not guaranteed to preserve its dominant contribution at the top slot.

---

## Compute Notes

- **Extraction**: 198 problems × 64 runs, 4 processes × 4 dist-threads = 16 total threads
- **LOPO**: 198 folds, each fitting `StandardScaler + LogisticRegression(C=1.0)` on
  ~2,000 training pairs — negligible CPU
- **Inference**: O(N²) = 64 × 63 = 4,032 `predict_proba` calls per problem, vectorised
- **Total wall time**: ~25 minutes (extraction dominated)

---

## Next Steps

### Option A — Feature engineering (recommended first)

The Bradley-Terry model's averaging effect is hurting top-1.  Before going to
DeepSets, try adding a **separability-focused feature** that amplifies the signal
right at the decision boundary.  Candidates:

1. **Score margin feature**: `max(recency_conf_mean_r) - recency_conf_mean_r[i]` —
   penalises runs far from the leader on the dominant feature.
2. **Dominance count**: for each run i, count j where `recency_conf_mean[i] > recency_conf_mean[j]`
   (i.e., treat `recency` alone as a hard prior and blend it with the pairwise soft scores).
3. **Calibration**: replace `LogisticRegression(C=1.0)` with `C=0.1` to increase
   regularisation and prevent the model from over-fitting to noisy features.

### Option B — Hybrid scoring

Keep `science_baseline_v1` for the top-1 decision but use `gpqa_pairwise_round1`
for top-k selection (where pairwise accuracy is higher).  This avoids the Hit@1
regression while capturing the pairwise accuracy gain.

### Option C — Deeper architecture (only after A/B pass gate)

Pursue a 2-layer DeepSets / Set Transformer on the same 6-dim run-level features.
This retains the full-group context while removing the Bradley-Terry mean-aggregation
limitation.  Only justified if A/B fail.

---

## Model Artefact

Final model (trained on all 198 GPQA problems):
`models/ml_selectors/gpqa_pairwise_round1.pkl`

This model is NOT promoted but is saved for future ablation / hybrid experiments.
