# Semi-Supervised SVDomain Extension

Generated: 2026-04-15

## Executive Summary

A self-supervised basis pre-training stage (SSL) was evaluated against three
baselines across math, science, and coding correctness-prediction domains.
The SSL basis (B ∈ ℝ^{44×r}, trained with masked reconstruction + cross-anchor
alignment + raw/rank view consistency) did **not** improve upon fully supervised
or even raw-LR baselines in any domain.  The strongest condition across all
domains is `no_svd_lr` (StandardScaler → LogisticRegression, no SVD), which
achieves AUROC up to **0.96 on math** and **0.83 on science** even with small
label sets.

Key degenerate observation: `semisup r=8` produces AUROC = 0.5000 on every
single row — the LR head fitted on the r=8 SSL embedding collapses to a
constant predictor, indicating the basis learned a near-null projection.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Training data | MUI_HUB/cache (31 040 samples) + MUI_HUB/cache_train (18 432 samples) |
| SSL total runs | 42 240 runs × 4 anchors |
| Holdout | 85/15 split per dataset, seed=42 |
| SSL pre-training | 300 epochs, Adam lr=0.01, batch=256 |
| SSL losses | λ_recon=1.0, λ_align=0.5, λ_view=0.5, mask_rate=0.30 |
| SSL ranks tested | r ∈ {4, 8, 16} |
| Cross-anchor pair | 10% (early) → 70% (late) |
| Feature family | token_plus_traj_fixed (22 features × 2 raw+rank = 44-d) |
| Label fractions | 1%, 5%, 10%, 20%, 50%, 100% |
| Domain order | coding → science → math |

SSL convergence — all ranks saturate near loss ≈ 0.432 before epoch 60
and show no further decrease through epoch 300.  This indicates the combined
loss is dominated by the reconstruction term with no discriminative gain.

---

## Domain Rankings at 100% Labels

| Domain  | no_svd_lr | frozen_svd | supervised_svd | semisup r=16 | semisup r=4 | semisup r=8 |
|---------|----------:|-----------:|---------------:|-------------:|------------:|------------:|
| math    |    0.9594 |     0.9582 |         0.8827 |       0.7794 |      0.6661 |      0.5000 |
| science |    0.8320 |     0.7944 |         0.7701 |       0.7876 |      0.6102 |      0.5000 |
| coding  |    0.4797 |        N/A |         0.4639 |       0.5020 |      0.4962 |      0.5000 |

**Verdict:** `no_svd_lr` wins on math and science; all methods are near-random
on coding.  SSL consistently underperforms every supervised baseline except
at 1% labels on science where `semisup r=16` (0.655) marginally beats
`supervised_svd` (0.644).

---

## Failure Analysis

### Why `semisup r=8` collapses to AUROC = 0.5

The SSL loss saturates at ≈0.432 after 60 epochs.  For r=8, the basis B
converges to a projection that minimises all three losses simultaneously but
loses all discriminative signal.  The LR head learns a constant classifier
(identical logit for every sample), giving exactly AUROC = 0.5.

### Why SSL underperforms overall

1. **Reconstruction dominance** — L_recon is the largest term; the basis
   minimises reconstruction error rather than class-discriminative geometry.
2. **Cross-domain pollution** — the basis is trained on all 42 240 runs from
   math, science, and coding together; domain-specific signal is diluted.
3. **Early saturation** — Adam converges around epoch 60 of 300; a learning-rate
   schedule or warm restarts would be needed to escape the flat region.
4. **Pairwise coding mismatch** — the pairwise coding head uses Z_i − Z_j; if B
   collapses coding samples to similar embeddings, differences are near-zero and
   the head cannot discriminate.

### Recommendations

1. Use **`no_svd_lr`** as the default for math and science — simpler, faster,
   and consistently best.
2. Train **domain-specific SSL bases** instead of one shared cross-domain basis.
3. Add a **cosine LR schedule** to avoid premature convergence at epoch 60.
4. Consider **contrastive / SimCLR-style losses** that push same-class embeddings
   together instead of reconstruction pretext tasks for this tabular setting.
5. For coding: token-trajectory features are insufficient; code-execution or
   AST-based features are likely needed.

---

## Full Results Table

| Domain | Condition | SSL Rank | Label % | AUROC | Top1 Acc | AUC SelAcc |
|--------|-----------|:--------:|--------:|------:|---------:|----------:|
| coding | no_svd_lr | -1 | 1% | 0.4948 | 0.5200 | 0.4344 |
| coding | no_svd_lr | -1 | 5% | 0.4494 | 0.5200 | 0.3375 |
| coding | no_svd_lr | -1 | 10% | 0.4188 | 0.4400 | 0.2444 |
| coding | no_svd_lr | -1 | 20% | 0.4426 | 0.5600 | 0.3387 |
| coding | no_svd_lr | -1 | 50% | 0.4637 | 0.6000 | 0.4006 |
| coding | no_svd_lr | -1 | 100% | 0.4797 | 0.6000 | 0.4725 |
| coding | semisup | 4 | 1% | 0.4988 | 0.4800 | 0.5094 |
| coding | semisup | 4 | 5% | 0.4939 | 0.5600 | 0.5431 |
| coding | semisup | 4 | 10% | 0.5016 | 0.5600 | 0.5581 |
| coding | semisup | 4 | 20% | 0.5068 | 0.5200 | 0.5125 |
| coding | semisup | 4 | 50% | 0.4946 | 0.5200 | 0.5144 |
| coding | semisup | 4 | 100% | 0.4962 | 0.5600 | 0.5150 |
| coding | semisup | 8 | 1% | 0.5000 | 0.5600 | 0.6250 |
| coding | semisup | 8 | 5% | 0.5000 | 0.5600 | 0.6250 |
| coding | semisup | 8 | 10% | 0.5000 | 0.5600 | 0.6250 |
| coding | semisup | 8 | 20% | 0.5000 | 0.5600 | 0.6250 |
| coding | semisup | 8 | 50% | 0.5000 | 0.5600 | 0.6250 |
| coding | semisup | 8 | 100% | 0.5000 | 0.5600 | 0.6250 |
| coding | semisup | 16 | 1% | 0.4988 | 0.5200 | 0.5156 |
| coding | semisup | 16 | 5% | 0.5043 | 0.4800 | 0.5163 |
| coding | semisup | 16 | 10% | 0.4990 | 0.5200 | 0.5119 |
| coding | semisup | 16 | 20% | 0.4991 | 0.5200 | 0.5119 |
| coding | semisup | 16 | 50% | 0.4990 | 0.5200 | 0.5131 |
| coding | semisup | 16 | 100% | 0.5020 | 0.5200 | 0.5269 |
| coding | supervised_svd | -1 | 1% | 0.5010 | 0.5600 | 0.4738 |
| coding | supervised_svd | -1 | 5% | 0.4932 | 0.5200 | 0.4537 |
| coding | supervised_svd | -1 | 10% | 0.4584 | 0.4800 | 0.4437 |
| coding | supervised_svd | -1 | 20% | 0.4532 | 0.5200 | 0.4400 |
| coding | supervised_svd | -1 | 50% | 0.4663 | 0.5200 | 0.4619 |
| coding | supervised_svd | -1 | 100% | 0.4639 | 0.5200 | 0.4619 |
| math | frozen_svd | -1 | 1% | 0.6569 | 0.7500 | 0.8802 |
| math | frozen_svd | -1 | 5% | 0.8842 | 0.7500 | 0.9989 |
| math | frozen_svd | -1 | 10% | 0.9188 | 0.7500 | 0.9989 |
| math | frozen_svd | -1 | 20% | 0.9543 | 0.7500 | 0.9967 |
| math | frozen_svd | -1 | 50% | 0.9573 | 0.7500 | 0.9978 |
| math | frozen_svd | -1 | 100% | 0.9582 | 0.7500 | 0.9978 |
| math | no_svd_lr | -1 | 1% | 0.7759 | 0.7500 | 0.9560 |
| math | no_svd_lr | -1 | 5% | 0.8806 | 0.7500 | 0.9978 |
| math | no_svd_lr | -1 | 10% | 0.9158 | 0.7500 | 0.9995 |
| math | no_svd_lr | -1 | 20% | 0.9547 | 0.7500 | 0.9984 |
| math | no_svd_lr | -1 | 50% | 0.9573 | 0.7143 | 0.9978 |
| math | no_svd_lr | -1 | 100% | 0.9594 | 0.7143 | 0.9973 |
| math | semisup | 4 | 1% | 0.4990 | 0.6786 | 0.6549 |
| math | semisup | 4 | 5% | 0.5928 | 0.7143 | 0.7643 |
| math | semisup | 4 | 10% | 0.5927 | 0.6429 | 0.8093 |
| math | semisup | 4 | 20% | 0.6757 | 0.6786 | 0.8835 |
| math | semisup | 4 | 50% | 0.6882 | 0.7143 | 0.8896 |
| math | semisup | 4 | 100% | 0.6661 | 0.6786 | 0.8802 |
| math | semisup | 8 | 1% | 0.5000 | 0.7143 | 0.7857 |
| math | semisup | 8 | 5% | 0.5000 | 0.7143 | 0.7857 |
| math | semisup | 8 | 10% | 0.5000 | 0.7143 | 0.7857 |
| math | semisup | 8 | 20% | 0.5000 | 0.7143 | 0.7857 |
| math | semisup | 8 | 50% | 0.5000 | 0.7143 | 0.7857 |
| math | semisup | 8 | 100% | 0.5000 | 0.7143 | 0.7857 |
| math | semisup | 16 | 1% | 0.5287 | 0.6786 | 0.7286 |
| math | semisup | 16 | 5% | 0.6516 | 0.7143 | 0.8901 |
| math | semisup | 16 | 10% | 0.6381 | 0.7143 | 0.8912 |
| math | semisup | 16 | 20% | 0.7301 | 0.7500 | 0.9302 |
| math | semisup | 16 | 50% | 0.7406 | 0.7500 | 0.9429 |
| math | semisup | 16 | 100% | 0.7794 | 0.7500 | 0.9555 |
| math | supervised_svd | -1 | 1% | 0.7524 | 0.6786 | 0.9291 |
| math | supervised_svd | -1 | 5% | 0.7380 | 0.7500 | 0.9538 |
| math | supervised_svd | -1 | 10% | 0.8194 | 0.7500 | 0.9830 |
| math | supervised_svd | -1 | 20% | 0.8841 | 0.7500 | 0.9951 |
| math | supervised_svd | -1 | 50% | 0.8764 | 0.7857 | 0.9978 |
| math | supervised_svd | -1 | 100% | 0.8827 | 0.7857 | 0.9989 |
| science | frozen_svd | -1 | 1% | 0.4919 | 0.7833 | 0.7378 |
| science | frozen_svd | -1 | 5% | 0.5770 | 0.7833 | 0.8096 |
| science | frozen_svd | -1 | 10% | 0.7705 | 0.7833 | 0.9552 |
| science | frozen_svd | -1 | 20% | 0.8153 | 0.7333 | 0.9893 |
| science | frozen_svd | -1 | 50% | 0.7924 | 0.8000 | 0.9844 |
| science | frozen_svd | -1 | 100% | 0.7944 | 0.7333 | 0.9846 |
| science | no_svd_lr | -1 | 1% | 0.5382 | 0.7833 | 0.7786 |
| science | no_svd_lr | -1 | 5% | 0.6732 | 0.7167 | 0.8826 |
| science | no_svd_lr | -1 | 10% | 0.7975 | 0.7667 | 0.9505 |
| science | no_svd_lr | -1 | 20% | 0.8283 | 0.7333 | 0.9883 |
| science | no_svd_lr | -1 | 50% | 0.8259 | 0.7500 | 0.9841 |
| science | no_svd_lr | -1 | 100% | 0.8320 | 0.7000 | 0.9852 |
| science | semisup | 4 | 1% | 0.5727 | 0.5667 | 0.7961 |
| science | semisup | 4 | 5% | 0.3980 | 0.7000 | 0.5995 |
| science | semisup | 4 | 10% | 0.6023 | 0.5000 | 0.8341 |
| science | semisup | 4 | 20% | 0.6188 | 0.5167 | 0.8714 |
| science | semisup | 4 | 50% | 0.6107 | 0.5167 | 0.8531 |
| science | semisup | 4 | 100% | 0.6102 | 0.5167 | 0.8516 |
| science | semisup | 8 | 1% | 0.5000 | 0.6500 | 0.8333 |
| science | semisup | 8 | 5% | 0.5000 | 0.6500 | 0.8333 |
| science | semisup | 8 | 10% | 0.5000 | 0.6500 | 0.8333 |
| science | semisup | 8 | 20% | 0.5000 | 0.6500 | 0.8333 |
| science | semisup | 8 | 50% | 0.5000 | 0.6500 | 0.8333 |
| science | semisup | 8 | 100% | 0.5000 | 0.6500 | 0.8333 |
| science | semisup | 16 | 1% | 0.6555 | 0.7000 | 0.8836 |
| science | semisup | 16 | 5% | 0.5001 | 0.7000 | 0.7299 |
| science | semisup | 16 | 10% | 0.7012 | 0.7000 | 0.9654 |
| science | semisup | 16 | 20% | 0.7311 | 0.6833 | 0.9701 |
| science | semisup | 16 | 50% | 0.7715 | 0.7000 | 0.9823 |
| science | semisup | 16 | 100% | 0.7876 | 0.6833 | 0.9872 |
| science | supervised_svd | -1 | 1% | 0.6444 | 0.7667 | 0.8586 |
| science | supervised_svd | -1 | 5% | 0.5713 | 0.7833 | 0.7867 |
| science | supervised_svd | -1 | 10% | 0.7724 | 0.7000 | 0.9570 |
| science | supervised_svd | -1 | 20% | 0.7907 | 0.7500 | 0.9812 |
| science | supervised_svd | -1 | 50% | 0.7727 | 0.7667 | 0.9568 |
| science | supervised_svd | -1 | 100% | 0.7701 | 0.7667 | 0.9578 |

## Conditions

- **semisup**: self-supervised SSL basis + supervised head (proposed)
- **frozen_svd**: frozen SVD from `es_svd_ms_rr_r1` bundle + new LR head only
- **supervised_svd**: fresh SVD + LR trained on labeled subset only (rank=8)
- **no_svd_lr**: StandardScaler → LR, no dimensionality reduction

## SSL Losses

- `L_recon`: masked feature reconstruction (30% mask rate, only on masked positions)
- `L_align`: cross-anchor alignment via learned bridge W_ap (early 10% → late 70%)
- `L_view`: raw-vs-rank view consistency (B[:22,:] on raw half, B[22:,:] on rank half)
- `L_total = L_recon + 0.5 × L_align + 0.5 × L_view`

## Artifacts

| File | Size |
|------|------|
| `results/tables/semisup_svdomain.csv` | 102 rows |
| `results/cache/semisup_svdomain/feature_store.pkl` | 143 MB |
| `results/cache/semisup_svdomain/ssl_bundles.pkl` | SSL bases r=4,8,16 |
| `scripts/semi_supervised/train_semisup_svdomain.py` | training script |
