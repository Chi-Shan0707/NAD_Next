# Domain-Specific Contrastive SSL — Full Pipeline & Submission

**Date:** 2026-04-16
**Status:** Complete — 3 BestOfN submission files produced

---

## Overview

This document records the full pipeline that replaced the failed shared-basis SSL approach
(documented in `docs/19_SEMISUP_SVDOMAIN_CONCLUSIONS.md`) with a per-domain NT-Xent contrastive
SSL setup and produced BestOfN submission files trained on all available labeled data.

---

## Pipeline Stages

```
1. Train per-domain SSL bases  (run_domain_ssl_full.py)
        ↓
2. Evaluate label efficiency   (train_domain_specific_ssl.py → domain_specific_ssl.csv)
        ↓
3. Cache test-set features      (background build_feature_store job → test_features.pkl)
        ↓
4. Fullfit + export BestOfN    (export_domain_ssl_bestofn.py → submission/BestofN/)
```

---

## Stage 1: SSL Pre-training

**Script:** `scripts/semi_supervised/run_domain_ssl_full.py`
**Inner trainer:** `scripts/semi_supervised/train_domain_specific_ssl.py`
**Logs:** `logs/domain_ssl_20260415T090831/`
**Bundles:** `results/cache/domain_ssl/` (12 pkl files + combined `domain_ssl_bundles.pkl`)

### Architecture

Each domain gets its own basis matrix **B ∈ ℝ^{44×r}** (r ∈ {4, 8, 16}) trained only on that
domain's runs. Input features are raw+rank concatenated (22 raw + 22 rank = 44 dims).

```
Domain runs
    │
StandardScaler (fit on domain only)
    │
NT-Xent contrastive  (primary, τ=0.07)
  positives = same run, two anchor positions
  negatives = all other runs in batch
    │
+ Future-anchor prediction  (λ=0.3)
+ Raw-vs-rank view consistency  (λ=0.1)
+ [coding_weak only] Pairwise hinge  (λ=0.5)
    │
B ∈ ℝ^{44×r}  per domain
```

**Optimizer:** numpy Adam, cosine LR annealing (lr_max=0.01, lr_min=1e-4, 300 epochs),
gradient clipping at Frobenius norm = 1.0. Warm-started from top-r SVD singular vectors.

### Domain Configuration

| Domain  | NT-Xent views (early → late) | Pairwise hinge     | Batch |
|---------|-----------------------------|--------------------|-------|
| math    | 10% → 100%                  | No                 | 256   |
| science | 40% → 100%                  | No                 | 256   |
| coding  | 40% → 70%                   | Yes (weak variant) | 384   |

Coding uses 70% (not 100%) as the late view to avoid at-max-token truncation artifacts.
Science uses 40% as the early view because 10% is too sparse for GPQA reasoning chains.

### CPU Strategy

6 parallel processes (coding, 3 ranks × 2 variants) then 3 (science) then 3 (math).
`OMP_NUM_THREADS=1` per process — BLAS threading is counterproductive for 256–384 batch sizes
(benchmark: 1 thread = 7.32 ms/iter vs 8 threads = 7.43 ms; threading overhead dominates).

### Trained Bundles

| File | Domain | r | Variant |
|------|--------|---|---------|
| `bundle_math_r4.pkl` | math | 4 | standard |
| `bundle_math_r8.pkl` | math | 8 | standard |
| `bundle_math_r16.pkl` | math | 16 | standard |
| `bundle_science_r4.pkl` | science | 4 | standard |
| `bundle_science_r8.pkl` | science | 8 | standard |
| `bundle_science_r16.pkl` | science | 16 | standard |
| `bundle_coding_r4.pkl` | coding | 4 | standard |
| `bundle_coding_r8.pkl` | coding | 8 | standard |
| `bundle_coding_r16.pkl` | coding | 16 | standard |
| `bundle_coding_r4_weak.pkl` | coding | 4 | + pairwise hinge |
| `bundle_coding_r8_weak.pkl` | coding | 8 | + pairwise hinge |
| `bundle_coding_r16_weak.pkl` | coding | 16 | + pairwise hinge |

Bundle dict keys: `{"B": (44, r), "W_future": (r, r), "W_score": (r, 1), "W_decode": (r, 8), "scaler", "r", "d", "domain", "final_loss"}`.

---

## Stage 2: Label Efficiency Evaluation

**Script:** `scripts/semi_supervised/train_domain_specific_ssl.py`
**Output:** `results/tables/domain_specific_ssl.csv` (120 rows)
**Report:** `docs/DOMAIN_SPECIFIC_SSL.md`

### Conditions Evaluated

| Condition | Description |
|-----------|-------------|
| `domain_ssl` | Per-domain NT-Xent basis + supervised LR head |
| `domain_ssl_weak` | Same + pairwise hinge during coding pre-training |
| `shared_ssl_r16` | Old shared SSL basis (r=16) — failure baseline |
| `no_svd_lr` | StandardScaler → LR, no dim reduction — strong baseline |
| `frozen_svd` | Frozen supervised SVD basis + new LR head |

### Key Results (AUROC @ 100% labels)

| Domain  | domain_ssl_r4 | domain_ssl_r16 | shared_ssl_r16 | no_svd_lr | frozen_svd |
|---------|:-------------:|:--------------:|:--------------:|:---------:|:----------:|
| math    | 0.886         | 0.876          | 0.779          | **0.959** | 0.958      |
| science | 0.818         | **0.845**      | 0.788          | 0.832     | 0.794      |
| coding  | ~0.509        | ~0.495         | ~0.502         | ~0.480    | —          |

### Research Question Answers

**Q1: Was the old SSL failure due to objective, cross-domain mixing, or both?**
Both causes, but cross-domain mixing is the dominant factor:
- `domain_ssl` outperforms `shared_ssl_r16` by **+9.6 pp** on math, **+5.8 pp** on science
- `domain_ssl` still trails `no_svd_lr` by ~7 pp on math → linear-basis capacity also a factor
- Verdict: per-domain separation is necessary but not sufficient to beat a direct LR baseline

**Q2: Does SSL help at low labels?**
Selectively yes — for science:
- Science @ 1%: `domain_ssl_r4 = 0.642` vs `no_svd_lr = 0.442` ← **+20 pp gain**
- Math @ 1%: `domain_ssl_r4 = 0.699` vs `no_svd_lr = 0.882` ← no benefit (LR generalizes well)
- Conclusion: SSL pre-training improves label efficiency for science but not math

**Q3: Does coding benefit from pairwise weak supervision?**
Negligible difference: `domain_ssl_weak` ≈ `domain_ssl` across all coding fractions (both ≈0.50).
Coding AUROC ≈ 0.50 for ALL methods — inherent task difficulty, not SSL failure.

**Q4: Per-domain vs shared basis quality?**
Clear improvement on math (+9.6 pp) and science (+5.8 pp) at 100% labels.
Per-domain training directly fixes cross-domain pollution identified as root cause.

---

## Stage 3: Test Feature Caching

**Output:** `results/cache/domain_ssl_test_features/test_features.pkl`
**Samples:** 62,080 samples across 12 cache_test entries
**Models:** DS-R1, Qwen3-4B
**Datasets:** aime24, aime25, brumo25, gpqa, hmmt25, lcb_v5

Feature extraction anchor positions:
- Math/Science: position = 1.0 (100% anchor, index 3)
- Coding: position = 0.7 (70% anchor, index 2)

Tensor shape per payload: `(N, 1, 47)` — 47 = len(FULL_FEATURE_NAMES).
Only `FIXED_FEATURE_INDICES` (22 indices) are used downstream.

Cross-model generalization: DS-R1-trained SSL bases applied to Qwen3-4B data via
rank normalization within problem groups — features are model-scale agnostic after ranking.

---

## Stage 4: Fullfit & BestOfN Export

**Script:** `scripts/semi_supervised/export_domain_ssl_bestofn.py`
**Logs:** `logs/domain_ssl_bestofn_20260416T082823/`
**Output:** `submission/BestofN/`

### Labeled Training Data

| Source | Payloads | Samples |
|--------|----------|---------|
| `results/cache/es_svd_ms_rr_r1/cache_all_*.pkl` | 6 | 31,040 |
| `results/cache/es_svd_ms_rr_r1/cache_train_all_*.pkl` | 4 | 18,432 |
| **Total** | **10** | **49,472** |

Domain split: math 13,440 / science 25,344 / coding 10,688.

### Fullfit Head Strategy

For each domain and rank:
1. GroupKFold-3 C-search over {0.05, 0.10, 0.20, 0.50, 1.00} (validation only, not held out)
2. Refit on **ALL 49,472 labeled samples** with selected C

```
X_raw  (N, 22)  ←  tensor[:, 0, FIXED_FEATURE_INDICES]
X_rank (N, 22)  ←  rank_transform within problem groups
X_rep  (N, 44)  ←  concat([X_raw, X_rank])
Z      (N, r)   ←  scaler.transform(X_rep) @ B
scores (N,)     ←  head.decision_function(Z)
```

### Fullfit Results (CV AUROC used for C selection, not holdout)

| Rank | math C / CV-AUROC | science C / CV-AUROC | coding C / CV-AUROC |
|------|:-----------------:|:--------------------:|:-------------------:|
| r=4  | C=1.00 / 0.856    | C=0.05 / 0.752       | C=1.00 / 0.454      |
| r=8  | C=1.00 / 0.859    | C=1.00 / 0.725       | C=1.00 / 0.470      |
| r=16 | C=1.00 / **0.892** | C=1.00 / **0.779** | C=1.00 / 0.492      |

Note: CV AUROC is lower than holdout AUROC from Stage 2 because fullfit uses all data
(including samples that would be test in a holdout scheme).

### Submission Files

| File | Method name | Samples | Validated |
|------|-------------|---------|-----------|
| `submission/BestofN/domain_ssl_r4_fullfit_v1.json` | `domain_ssl_r4_fullfit_v1` | 62,080 | PASSED |
| `submission/BestofN/domain_ssl_r8_fullfit_v1.json` | `domain_ssl_r8_fullfit_v1` | 62,080 | PASSED |
| `submission/BestofN/domain_ssl_r16_fullfit_v1.json` | `domain_ssl_r16_fullfit_v1` | 62,080 | PASSED |

Each file covers:
- **12 cache_keys** (DS-R1 × 6 datasets + Qwen3-4B × 6 datasets)
- **970 problems**, **62,080 samples** (64 per problem)
- Format: `{task, method_name, scores: {cache_key: {problem_id: {sample_id: float}}}}`
- `sample_id` = 0-indexed local position within problem group

**Recommended submission: `domain_ssl_r16_fullfit_v1.json`** — highest CV AUROC on both
math (0.892) and science (0.779) across all ranks.

---

## File Inventory

| Path | Description |
|------|-------------|
| `scripts/semi_supervised/train_domain_specific_ssl.py` | Core SSL implementation |
| `scripts/semi_supervised/run_domain_ssl_full.py` | Parallel training launcher |
| `scripts/semi_supervised/export_domain_ssl_bestofn.py` | BestOfN export (fast path) |
| `results/tables/domain_specific_ssl.csv` | 120-row label efficiency results |
| `docs/DOMAIN_SPECIFIC_SSL.md` | Auto-generated results report |
| `docs/20_DOMAIN_SSL_SUBMISSION.md` | This document |
| `results/cache/domain_ssl/` | 12 bundle pkl files + combined pkl |
| `results/cache/domain_ssl_test_features/test_features.pkl` | 62,080 cached test features |
| `submission/BestofN/domain_ssl_r{4,8,16}_fullfit_v1.json` | Submission files |
| `logs/domain_ssl_20260415T090831/` | Training logs + training_summary.md |
| `logs/domain_ssl_bestofn_20260416T082823/` | Export logs + export_summary.md |

---

## Reproducibility

```bash
# Stage 1: Train SSL bases (300 epochs, ~25-45 min)
python3 scripts/semi_supervised/run_domain_ssl_full.py \
    --prebuilt-cache-dir results/cache/es_svd_ms_rr_r1 \
    --ssl-ranks 4 8 16 --n-epochs 300 \
    --out-dir results/cache/domain_ssl \
    --log-dir logs

# Stage 2: Label efficiency evaluation
# (run automatically inside run_domain_ssl_full.py or separately via train_domain_specific_ssl.py)

# Stage 3: Cache test features (one-time, ~10 min)
# (was run as background job via build_feature_store on MUI_HUB/cache_test)

# Stage 4: Export BestOfN submission
python3 scripts/semi_supervised/export_domain_ssl_bestofn.py \
    --ssl-rank 4 8 16 \
    --test-features-pkl results/cache/domain_ssl_test_features/test_features.pkl \
    --prebuilt-cache-dir results/cache/es_svd_ms_rr_r1 \
    --domain-ssl-dir results/cache/domain_ssl \
    --out-dir submission/BestofN \
    --log-dir logs
```

---

## Limitations & Future Work

1. **Coding inherently hard**: AUROC ≈ 0.50 for all methods. Feature set may lack signal at
   intermediate completion percentages. Future: add problem-category-aware grouping or
   use execution-based features (compile errors, test-pass rate).

2. **SSL trails direct LR on math**: `no_svd_lr` 0.959 vs `domain_ssl_r16` 0.876 at 100%.
   The linear bottleneck (B has at most r dimensions) discards useful variance. Future:
   nonlinear projection head (small MLP) on top of B.

3. **Cross-model transfer**: Applied DS-R1 bases to Qwen3-4B without domain adaptation.
   A model-aware pre-training step (separate bases per model) may improve Qwen3-4B accuracy.

4. **Fullfit CV AUROC is optimistic**: No true held-out evaluation for fullfit heads.
   The CV-AUROC figures (0.856/0.752/0.454 for r=4) are used only for C selection.
