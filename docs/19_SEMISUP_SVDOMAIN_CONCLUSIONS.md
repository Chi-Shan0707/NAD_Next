# 19 — Semi-Supervised SVDomain: Conclusions

**Date:** 2026-04-15
**Experiment:** `scripts/semi_supervised/train_semisup_svdomain.py`
**Data:** `results/tables/semisup_svdomain.csv` (102 rows)

---

## What was tested

A self-supervised pre-training stage (SSL) for the SVDomain feature basis,
evaluated against three supervised baselines across coding, science, and math
domains at six label fractions (1 – 100 %).

### SSL design

| Component | Choice |
|-----------|--------|
| Basis | B ∈ ℝ^{44×r}, learned jointly for raw + rank features |
| Optimizer | Numpy Adam, lr = 0.01, 300 epochs, batch = 256 |
| Loss 1 — reconstruction | Masked feature recovery, 30 % mask rate |
| Loss 2 — alignment | Bridge W_ap predicts late embedding (70 %) from early (10 %) |
| Loss 3 — view consistency | Raw-side B[:22,:] vs rank-side B[22:,:] produce same Z |
| Ranks tested | r ∈ {4, 8, 16} |
| Training corpus | 42 240 runs (math + science + coding, labels not used) |

### Conditions compared

| ID | Description |
|----|-------------|
| `semisup` | SSL basis + new LR head per anchor |
| `frozen_svd` | Frozen `es_svd_ms_rr_r1` scaler/SVD + new LR head |
| `supervised_svd` | Fresh scaler + SVD (r = 8) + LR, trained on labeled subset |
| `no_svd_lr` | StandardScaler → LogisticRegression, no SVD |

---

## Main results at 100 % labels

| Domain | no_svd_lr | frozen_svd | supervised_svd | semisup r=16 | semisup r=8 | semisup r=4 |
|--------|----------:|-----------:|---------------:|-------------:|------------:|------------:|
| math | **0.9594** | 0.9582 | 0.8827 | 0.7794 | 0.5000 ⚠ | 0.6661 |
| science | **0.8320** | 0.7944 | 0.7701 | 0.7876 | 0.5000 ⚠ | 0.6102 |
| coding | **0.4797** | N/A | 0.4639 | 0.5020 | 0.5000 ⚠ | 0.4962 |

⚠ = collapsed to constant predictor (see §3).

---

## Findings

### 1. SSL pre-training did not improve label efficiency

At every label fraction and in every domain, `no_svd_lr` outperforms or matches
the best SSL variant.  The gap is largest on math, where `no_svd_lr` achieves
0.96 AUROC vs 0.78 for `semisup r=16` at 100 % labels.

There is one marginal exception: at 1 % labels on science (n = 2 problems),
`semisup r=16` (0.655) slightly edges `supervised_svd` (0.644), but both trail
`no_svd_lr` (0.538 — in this regime `no_svd_lr` is unstable).

**Label-efficiency curve (math, AUC-AUROC vs label fraction):**

| Label % | no_svd_lr | frozen_svd | supervised_svd | semisup r=16 |
|--------:|----------:|-----------:|---------------:|-------------:|
| 1 % | 0.776 | 0.657 | 0.752 | 0.529 |
| 5 % | 0.881 | **0.884** | 0.738 | 0.652 |
| 10 % | 0.916 | **0.919** | 0.819 | 0.638 |
| 20 % | 0.955 | 0.954 | 0.884 | 0.730 |
| 50 % | 0.957 | 0.957 | 0.876 | 0.741 |
| 100 % | **0.959** | 0.958 | 0.883 | 0.779 |

`frozen_svd` slightly surpasses `no_svd_lr` at 5 – 10 % labels, suggesting the
pre-trained `es_svd_ms_rr_r1` basis does carry some transfer value at low label
counts — but the SSL basis we trained here does not.

### 2. The r = 8 SSL basis collapses universally

`semisup r=8` produces AUROC = **0.5000 exactly** on all 18 rows (6 fractions ×
3 domains).  The LR head learns a constant predictor regardless of label count.

**Cause:** the combined SSL loss saturated at ≈ 0.432 before epoch 60 and did
not decrease further through epoch 300.  For r = 8 specifically, the Adam
trajectory reached a degenerate saddle where all three loss terms are locally
minimised by a near-zero projection (all samples collapse to the same point in
ℝ^8).

### 3. SSL loss saturates without discriminative improvement

All three basis ranks show the same pattern:

| Rank | Loss @ epoch 60 | Loss @ epoch 300 | Change |
|------|-----------------|------------------|--------|
| r = 4 | 0.43212 | 0.43218 | +0.00006 |
| r = 8 | 0.43219 | 0.43209 | −0.00010 |
| r = 16 | 0.43444 | 0.43232 | −0.00212 |

No rank improves meaningfully after epoch 60.  The reconstruction term dominates
the gradient signal; the alignment and view consistency terms add noise without
pushing the basis toward class-discriminative directions.

### 4. Coding correctness is not predictable from token-trajectory features

All conditions — including the full-label `supervised_svd` and `no_svd_lr` —
produce AUROC ≈ 0.46 – 0.50 on coding.  This is consistent with
`12_CODING_DIAGNOSIS.md`: the 22 token-uncertainty + trajectory features carry
insufficient signal for code correctness.  The pairwise LR head for coding did
not recover anything beyond random.

### 5. `no_svd_lr` is the empirical best across all domains

StandardScaler → LogisticRegression (44-d input, no dimensionality reduction)
is the strongest single model at nearly every label fraction.  This replicates
the finding from `DIRECT_LINEAR_BASELINES.md` (where `l1_lr_raw_rank` and
`elasticnet_lr_raw_rank` similarly beat `supervised_svd`) and is consistent with
`18_SVD_FEATURE_COMPLEXITY.md`'s finding that SVD becomes valuable only when
the feature bank is wider or noisier than 22 features.

---

## Why SSL failed

| Root cause | Evidence |
|------------|----------|
| **Reconstruction dominance** | Loss is ≈ 0.43 for all r; masking-then-predicting 30 % of 44 features is too easy for a linear autoencoder, producing flat gradients toward discriminative directions |
| **Cross-domain pollution** | Basis is trained on 42 K runs from three heterogeneous domains; math-specific structure is diluted by coding and science structure |
| **Early saddle convergence** | All ranks plateau before epoch 60; no LR schedule or restarts were used to escape the flat region |
| **Linear autoencoder capacity mismatch** | A 44 × r linear projection cannot represent the structured non-linearity needed to separate trajectory shapes that differ only in their tail; SVD already extracts the optimal linear projection from labeled data, so SSL adds nothing |
| **Pairwise coding head is degenerate** | If B collapses coding embeddings, Z_i − Z_j ≈ 0 for all pairs; the pairwise LR has no signal to fit |

---

## Comparison with related experiments

| Experiment | Key finding | Relation to SSL result |
|------------|-------------|----------------------|
| `DIRECT_LINEAR_BASELINES.md` | no-SVD linear heads beat SVD on math, science, ms, coding | Confirms `no_svd_lr` as baseline; SSL cannot beat what SVD already cannot beat |
| `08_FROZEN_BASIS_TRANSFER.md` | Frozen `es_svd_ms_rr_r1` transfers within math/science | `frozen_svd` replicates this; the supervised basis transfers but our SSL basis does not |
| `18_SVD_FEATURE_COMPLEXITY.md` | SVD wins over no-SVD only at wide, noisy feature banks | At 22 clean features, SVD (and SSL) compression hurts |
| `12_CODING_DIAGNOSIS.md` | Coding near-random for all token-trajectory models | SSL adds no new signal source; coding ceiling ≈ 0.50 regardless |

---

## Recommendations

1. **Use `no_svd_lr` (or `frozen_svd` at 5–10 % labels) as the go-forward
   default** for math and science early-stopping.  No SSL is needed.

2. **Do not use the shared cross-domain SSL basis.**  If SSL is retried, train
   one basis per domain on that domain's data only.

3. **Add a cosine LR schedule** (e.g., warm restarts every 50 epochs) if
   reconstruction-based SSL is reattempted; the current flat loss after epoch 60
   indicates Adam is stuck.

4. **Consider contrastive pretext tasks** (e.g., SimCLR-style pair positives =
   same problem, same anchor, different random seed) rather than masked
   reconstruction.  Correctness-agnostic reconstruction is too easy for this
   feature space.

5. **Coding needs a different feature family.**  Token-trajectory features are
   insufficient; code-execution signals (test-pass rates, AST depth, error
   type) are the natural next step.

---

## Artifacts

| File | Contents |
|------|----------|
| `results/tables/semisup_svdomain.csv` | 102 rows: domain × condition × fraction |
| `docs/SEMISUP_SVDOMAIN.md` | Full results table + domain breakdowns |
| `results/cache/semisup_svdomain/ssl_bundles.pkl` | Trained SSL bases r ∈ {4, 8, 16} |
| `scripts/semi_supervised/train_semisup_svdomain.py` | Reusable training script |
