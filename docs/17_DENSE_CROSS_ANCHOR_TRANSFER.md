# 17. Dense Cross-Anchor Transfer

**Experiment script**: `SVDomain/experiments/run_dense_cross_anchor_transfer.py`
**Source bundle**: `models/ml_selectors/es_svd_ms_rr_r2.pkl`
**Anchors**: `10, 20, 30, 40, 50, 60, 70, 80, 90, 100`
**Domains**: `math, science`
**Date**: 2026-04-12

---

## 1. Claim Under Test

> Cross-anchor transfer is not only a sparse `10/40/70/100` phenomenon, but can be traced across the full dense `10/20/.../100` reasoning trajectory.

这轮实验的目的，是把原本四个 anchor 的 frozen-basis transfer 扩展成 **全 10×10 all-to-all dense grid**，直接回答：共享 low-rank basis 在整条 trajectory 上到底能迁移多远、在哪些方向上开始失真。

## 2. Protocol

- 条件对比仍保持不变：`frozen_basis` vs `task_specific` vs `no_svd`。
- `frozen_basis`：固定 **source anchor** 的 `scaler + SVD`，只在 **target anchor** 上重训 LR head。
- `task_specific`：在 **target anchor** 上重训 `scaler + SVD + LR head`。
- `no_svd`：在 **target anchor** 上直接训练 `StandardScaler + LR`。
- bundle 切换为 dense `r2`：`es_svd_ms_rr_r2.pkl`，因此 source / target anchors 覆盖 `10/20/.../100` 全部 10 个位置。
- 评估协议与原 `11_CROSS_ANCHOR_TRANSFER.md` 一致：沿用 `GroupKFold` offline protocol，不改 split；主指标关注 `AUROC`，并同步导出 `SelAcc@10%` / `StopAcc`。

## 3. Headline Summary

- `math` diagonal mean: frozen=94.43% vs task-specific=94.56% (Δ=-0.13 pts); all off-diagonal mean: frozen=94.22% vs task-specific=94.56% (Δ=-0.34 pts). Near-gap (`10/20`) mean gap = -0.16 pts; far-gap (`50–90`) mean gap = -0.62 pts. Best reusable source anchor is `30%` (mean Δ=-0.14 pts).
- `science` diagonal mean: frozen=74.03% vs task-specific=74.12% (Δ=-0.09 pts); all off-diagonal mean: frozen=73.58% vs task-specific=74.12% (Δ=-0.54 pts). Near-gap (`10/20`) mean gap = -0.23 pts; far-gap (`50–90`) mean gap = -0.97 pts. Best reusable source anchor is `50%` (mean Δ=-0.02 pts).

## 4. Dense Stability Slices

| Domain | Slice | Frozen | Task-specific | No-SVD | Δ(Frozen−Task) | Δ(Frozen−NoSVD) |
|---|---|---:|---:|---:|---:|---:|
| math | diagonal | 94.43% | 94.56% | 94.68% | -0.13 pts | -0.25 pts |
| science | diagonal | 74.03% | 74.12% | 74.43% | -0.09 pts | -0.40 pts |
| math | offdiag all | 94.22% | 94.56% | 94.68% | -0.34 pts | -0.45 pts |
| science | offdiag all | 73.58% | 74.12% | 74.43% | -0.54 pts | -0.85 pts |
| math | offdiag near (10/20) | 94.59% | 94.75% | 94.87% | -0.16 pts | -0.28 pts |
| science | offdiag near (10/20) | 74.16% | 74.39% | 74.65% | -0.23 pts | -0.50 pts |
| math | offdiag far (50–90) | 93.60% | 94.21% | 94.31% | -0.62 pts | -0.71 pts |
| science | offdiag far (50–90) | 72.59% | 73.57% | 73.97% | -0.97 pts | -1.38 pts |
| math | forward all | 95.63% | 95.74% | 95.85% | -0.11 pts | -0.22 pts |
| science | forward all | 74.57% | 75.55% | 76.16% | -0.98 pts | -1.59 pts |
| math | backward all | 92.82% | 93.38% | 93.50% | -0.56 pts | -0.68 pts |
| science | backward all | 72.58% | 72.69% | 72.69% | -0.10 pts | -0.11 pts |

## 5. Gap Profile by Anchor Distance

| Domain | |Δanchor| | Combined Δ(Frozen−Task) | Forward Δ | Backward Δ |
|---|---:|---:|---:|---:|
| math | 10 | -0.15 pts | -0.12 pts | -0.18 pts |
| math | 20 | -0.17 pts | -0.11 pts | -0.24 pts |
| math | 30 | -0.21 pts | -0.09 pts | -0.33 pts |
| math | 40 | -0.29 pts | -0.10 pts | -0.47 pts |
| math | 50 | -0.38 pts | -0.12 pts | -0.65 pts |
| math | 60 | -0.51 pts | -0.11 pts | -0.90 pts |
| math | 70 | -0.68 pts | -0.11 pts | -1.26 pts |
| math | 80 | -0.96 pts | -0.14 pts | -1.78 pts |
| math | 90 | -1.35 pts | -0.19 pts | -2.51 pts |
| science | 10 | -0.20 pts | -0.36 pts | -0.03 pts |
| science | 20 | -0.27 pts | -0.49 pts | -0.05 pts |
| science | 30 | -0.40 pts | -0.68 pts | -0.11 pts |
| science | 40 | -0.51 pts | -0.88 pts | -0.15 pts |
| science | 50 | -0.63 pts | -1.09 pts | -0.17 pts |
| science | 60 | -0.84 pts | -1.52 pts | -0.17 pts |
| science | 70 | -1.15 pts | -2.12 pts | -0.18 pts |
| science | 80 | -1.66 pts | -3.17 pts | -0.14 pts |
| science | 90 | -1.25 pts | -2.57 pts | +0.06 pts |

## 6. Which Anchors Reuse Best?

- `math` source-anchor ranking by mean off-diagonal Δ(frozen−task_specific): 30% (-0.14 pts), 40% (-0.17 pts), 20% (-0.18 pts), 50% (-0.22 pts), 60% (-0.26 pts), 10% (-0.27 pts), 70% (-0.34 pts), 80% (-0.41 pts), 90% (-0.52 pts), 100% (-0.87 pts).
- `math` hardest target anchor by incoming mean transfer gap is `10%` (mean Δ=-1.06 pts).
- `science` source-anchor ranking by mean off-diagonal Δ(frozen−task_specific): 50% (-0.02 pts), 60% (-0.02 pts), 70% (-0.03 pts), 40% (-0.03 pts), 80% (-0.05 pts), 100% (-0.09 pts), 90% (-0.10 pts), 30% (-0.10 pts), 20% (-1.37 pts), 10% (-3.60 pts).
- `science` hardest target anchor by incoming mean transfer gap is `50%` (mean Δ=-0.74 pts).

## 7. Highlighted Pairs

| Domain | Best off-diagonal pair | Δ(Frozen−Task) | Worst off-diagonal pair | Δ(Frozen−Task) |
|---|---|---:|---|---:|
| math | 60->100 | +0.01 pts | 100->10 | -2.51 pts |
| science | 100->90 | +0.26 pts | 10->50 | -4.17 pts |

## 8. Direct Answers

- **Dense transfer 是否说明 math 只在少数 anchors 上可复用？** 不是。`math` 的 dense all-to-all 仍然整体接近 task-specific：diagonal mean gap 为 -0.13 pts，far-gap (`50–90`) mean gap 也只有 -0.62 pts。 最可迁移的 source anchor 是 `30%`。 单个最差 pair 是 `100->10`，gap=-2.51 pts。
- **Dense transfer 是否说明 science 的 basis 也能均匀复用？** 不能。`science` 的 gap 随 distance 明显放大：all off-diagonal mean gap 为 -0.54 pts，far-gap (`50–90`) mean gap 为 -0.97 pts。 最可迁移的 source anchor 是 `50%`。 单个最差 pair 是 `10->50`，gap=-4.17 pts。
- **与 `16_DENSE_ANCHOR_EARLYSTOP.md` 是否一致？** 是。`math` 在 dense timing 里 `10%` 就达到 95%-of-final，约 `50%` 进入 plateau；这与它在 dense cross-anchor transfer 中的弱距离衰减一致。`science` 在 timing 里 `20%` 达到 95%-of-final、约 `40%` 进入 plateau，但 dense transfer 仍显示 early→late basis reuse 不均匀；这更支持“early coarse signal + late refinement”，而不是“only-at-completion onset”。

## 9. Artifacts

- Root matrix CSV: `results/tables/dense_cross_anchor_transfer_matrix.csv`
- Root delta CSV: `results/tables/dense_cross_anchor_transfer_deltas.csv`
- Root summary CSV: `results/tables/dense_cross_anchor_transfer_summary.csv`
- Paper-package matrix CSV: `SVDomain/results/tables/dense_cross_anchor_transfer_matrix.csv`
- Paper-package delta CSV: `SVDomain/results/tables/dense_cross_anchor_transfer_deltas.csv`
- Paper-package summary CSV: `SVDomain/results/tables/dense_cross_anchor_transfer_summary.csv`
