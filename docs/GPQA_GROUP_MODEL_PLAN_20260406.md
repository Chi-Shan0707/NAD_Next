# GPQA Group Pairwise Model — Round 1

Date: `2026-04-06`

## Summary

This note describes the first round of the GPQA group pairwise model, a minimum viable
full-group pairwise scorer intended to determine whether explicit pairwise group context
improves over `science_baseline_v1` before investing in heavier set-transformer work.

Implemented files:

- `docs/GPQA_GROUP_MODEL_PLAN_20260406.md` (this file)
- `nad/core/selectors/gpqa_pairwise_impl.py`
- `plugins/gpqa_pairwise_selector.py`
- `scripts/run_gpqa_pairwise_round1.py`

Model artifact (produced by the evaluation script):

- `models/ml_selectors/gpqa_pairwise_round1.pkl`

---

## Why Extreme8/9/10 ≠ Full-Set Contextual

`Extreme8/9/10` are tuple-sampling models.  At inference time they draw 1024 random
k-tuples (k=8) from the N runs and accumulate per-run linear rank scores.  Each run
therefore appears in at most ~128 out of 4032 possible pairs; the model never
simultaneously observes all N runs.

The group pairwise model implemented here is **architecturally distinct**:

- **Inference**: exhaustively scans all N(N-1) ordered pairs in a single vectorised
  pass (`diffs = X[:,None,:] - X[None,:,:]`).
- **Training**: learns "does run i beat run j?" from pairwise feature differences using
  a Bradley-Terry logistic regression.
- **Normalisation**: all features are group-normalised *before* the difference is taken,
  so the model sees within-group comparisons regardless of problem difficulty.

---

## Frozen Baseline

`science_baseline_v1` is frozen.  Frozen weights:

- `prefix_conf_mean  = 0.00`
- `recency_conf_mean = 1.00`
- `late_worst_window = 0.00`
- `late_recovery     = 0.00`

Frozen performance on GPQA:

| Metric      | Value   |
|-------------|---------|
| AUROC       | 53.86%  |
| Hit@1       | 66.16%  |
| Pairwise    | 58.71%  |
| SelAcc@10   | 64.35%  |

---

## Why GPQA First

- Science (GPQA) is the domain where `science_baseline_v1` is specialised.
- Cross-domain mixing (code/math) is excluded from round 1.
- GPQA has a well-understood ground-truth format and a stable cache key.
- The LOPO protocol (leave-one-problem-out) gives an unbiased estimate with ~N_problems
  iterations each fitting a small logistic regression — negligible CPU cost.

---

## Feature Vector Design (6-dim)

All features are group-normalised before pairwise difference computation.

| Dim | Name                 | Source                                      |
|-----|----------------------|---------------------------------------------|
| 0   | `dc_z`               | `_zscore(dc_raw)` from `extract_extreme8_raw_values()` |
| 1   | `dc_r`               | `_rank01(dc_raw)` same source               |
| 2   | `reflection_count_r` | `_rank01(reflection_count)` same source     |
| 3   | `prefix_conf_mean_r` | `_rank01(prefix_conf_mean)` from `extract_science_dynamic_raw_matrix()` |
| 4   | `recency_conf_mean_r`| `_rank01(recency_conf_mean)` same           |
| 5   | `late_recovery_r`    | `_rank01(late_recovery)` same               |

`late_worst_window` is excluded from the pairwise feature vector (can be added later).

---

## Model: Bradley-Terry Logistic Regression

**Training pairs** (per problem with labels):

```
for each (i=correct, j=wrong):
    X_pairs <- [X[i] - X[j], label=1]   # correct beats wrong
    X_pairs <- [X[j] - X[i], label=0]   # anti-pair
```

**Pipeline**: `sklearn.Pipeline([StandardScaler, LogisticRegression(C=1.0)])`

**Inference** (full-group scan, O(N²)):

```python
diffs = X[:, None, :] - X[None, :, :]      # (N, N, 6)
mask  = ~np.eye(N, dtype=bool)              # off-diagonal
probs = model.predict_proba(diffs[mask])[:, 1]   # (N*(N-1),)
scores = probs.reshape(N, N-1).mean(axis=1)      # (N,) group scores
select argmax(scores)
```

**Training protocol**: Leave-one-problem-out (LOPO) within GPQA cache for evaluation.
Final model saved to `models/ml_selectors/gpqa_pairwise_round1.pkl` is trained on ALL
GPQA problems.

---

## Promote / No-Promote Gate

Thresholds vs `science_baseline_v1` on GPQA:

| Metric    | Threshold          | Baseline  |
|-----------|--------------------|-----------|
| SelAcc@10 | must exceed 64.35% | 64.35%    |
| AUROC     | must exceed 53.86% | 53.86%    |
| Hit@1     | must be ≥ 65.16%   | 66.16%    |
| Pairwise  | must be ≥ 58.21%   | 58.71%    |

All four must pass → promote to `gpqa_pairwise_v1`.
If SelAcc@10 or AUROC fail → no promote; document which metric failed.

If promoted → optionally pursue a small DeepSets follow-up (inputs still run-level
features, ≤2 layers, no raw slices).

---

## What This Plan Does NOT Include

- No graph topology (Extreme10 path closed)
- No DeepSets / Set Transformer (only after round-1 promotion)
- No changes to `science_baseline_v1` frozen weights
- No raw neuron / token-level inputs
- No cross-domain model (code/math excluded from round 1)
- No changes to existing selectors

---

## Compute Budget

- LOPO loop: ~N_problems iterations, each fitting a small logistic regression.  Negligible CPU.
- Feature extraction: one `extract_extreme8_raw_values` + `extract_science_dynamic_raw_matrix`
  pass per problem.  Same cost as existing scripts.
- Inference: O(N²) = 64×63 = 4032 `predict_proba` calls per problem.  Fast with vectorised numpy.
- Total runtime: comparable to `run_science_baseline_v1_round1.py`.  No parallelism needed.

---

## Verification

```bash
# From repo root (NAD_Next/)
python scripts/run_gpqa_pairwise_round1.py \
  --cache-root MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/gpqa/<latest_cache> \
  --out result/gpqa_pairwise_round1_$(date +%Y%m%d).json
```

Expected output:

```
Selector              | AUROC   | Hit@1   | Pairwise | SelAcc@10
science_baseline_v1   | 53.86%  | 66.16%  | 58.71%   | 64.35%
tournament-copeland   | xx.xx%  | xx.xx%  | xx.xx%   | xx.xx%
gpqa_pairwise_round1  | xx.xx%  | xx.xx%  | xx.xx%   | xx.xx%
```

Model saved at `models/ml_selectors/gpqa_pairwise_round1.pkl`.
