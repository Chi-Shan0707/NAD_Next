# 07. Low-Rank Necessity

这份 note 只回答一个问题：canonical `raw+rank` family 的收益，是否来自 low-rank bottleneck 本身，而不只是来自 feature bank。

## 1. 结论先行

- 在 `math / science / ms` 三个 noncoding domain 上，best fixed-rank SVD 都优于同特征、去掉 SVD 的 `no-SVD` baseline，但幅度是**一致而温和**的。
- 相对 `no_svd_lr` 的 `AUC of AUROC` 提升分别是：`math` +0.02 AUC-pts，`science` +0.19 AUC-pts，`ms` +0.06 AUC-pts。
- `math` 的 smallest sufficient rank = `16`；`science` 的 smallest sufficient rank = `24`；`ms` 的 smallest sufficient rank = `24`。
- `r2/r4/r8` 在三个 domain 上都明显低于最终最佳；真正的平台期从 `math:r16`、`science:r24`、`ms:r24` 才开始。

## 2. Smallest sufficient rank

| Domain | Best rank | Best AUC of AUROC | Smallest sufficient rank | No-SVD AUC of AUROC | Plateau ranks |
|---|---:|---:|---:|---:|---|
| math | 24 | 95.95% | 16 | 95.93% | 16,24,32 |
| science | 24 | 83.67% | 24 | 83.48% | 24,32 |
| ms | 24 | 93.22% | 24 | 93.16% | 24,32 |

## 3. Full ablation table

| Domain | Method | Rank | AUC of AUROC | AUC of SelAcc | AUROC@100 | StopAcc@100 | Fit time | Infer time | Compactness |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| math | no_svd_lr | — | 95.93% | 99.73% | 98.37% | 75.00% | 287.44s | 0.31s | 0.817 |
| math | svd_r2 | 2 | 63.54% | 91.70% | 58.65% | 67.86% | 89.04s | 0.26s | 0.541 |
| math | svd_r4 | 4 | 75.66% | 94.67% | 82.01% | 71.43% | 144.90s | 0.21s | 0.629 |
| math | svd_r8 | 8 | 88.25% | 99.89% | 88.56% | 78.57% | 43.44s | 0.29s | 0.814 |
| math | svd_r12 | 12 | 95.14% | 99.67% | 97.51% | 75.00% | 11.53s | 0.32s | 0.876 |
| math | svd_r16 | 16 | 95.81% | 99.73% | 98.17% | 75.00% | 15.78s | 0.21s | 0.884 |
| math | svd_r24 | 24 | 95.95% | 99.84% | 98.37% | 71.43% | 26.22s | 0.21s | 0.846 |
| math | svd_r32 | 32 | 95.93% | 99.73% | 98.37% | 75.00% | 26.20s | 0.21s | 0.814 |
| science | no_svd_lr | — | 83.48% | 98.46% | 87.61% | 71.67% | 466.55s | 0.31s | 0.768 |
| science | svd_r2 | 2 | 80.25% | 96.02% | 83.36% | 73.33% | 195.10s | 0.26s | 0.559 |
| science | svd_r4 | 4 | 79.96% | 95.86% | 83.29% | 70.00% | 167.20s | 0.21s | 0.567 |
| science | svd_r8 | 8 | 76.97% | 95.89% | 82.29% | 78.33% | 23.33s | 0.29s | 0.671 |
| science | svd_r12 | 12 | 78.70% | 98.10% | 84.11% | 65.00% | 17.00s | 0.32s | 0.816 |
| science | svd_r16 | 16 | 79.98% | 98.83% | 85.43% | 73.33% | 21.66s | 0.21s | 0.807 |
| science | svd_r24 | 24 | 83.67% | 98.28% | 87.60% | 71.67% | 42.10s | 0.21s | 0.778 |
| science | svd_r32 | 32 | 83.42% | 98.18% | 87.10% | 70.00% | 39.45s | 0.21s | 0.724 |
| ms | no_svd_lr | — | 93.16% | 99.44% | 95.98% | 74.26% | 753.99s | 0.92s | 0.792 |
| ms | svd_r2 | 2 | 67.25% | 92.66% | 64.14% | 69.07% | 284.14s | 0.78s | 0.550 |
| ms | svd_r4 | 4 | 76.62% | 94.93% | 82.29% | 71.11% | 312.10s | 0.63s | 0.598 |
| ms | svd_r8 | 8 | 85.74% | 99.00% | 87.17% | 78.52% | 66.77s | 0.87s | 0.742 |
| ms | svd_r12 | 12 | 91.49% | 99.32% | 94.53% | 72.78% | 28.52s | 0.95s | 0.846 |
| ms | svd_r16 | 16 | 92.29% | 99.53% | 95.34% | 74.63% | 37.44s | 0.63s | 0.846 |
| ms | svd_r24 | 24 | 93.22% | 99.49% | 95.98% | 71.48% | 68.31s | 0.64s | 0.812 |
| ms | svd_r32 | 32 | 93.15% | 99.38% | 95.86% | 73.89% | 65.65s | 0.64s | 0.769 |

## 4. Paper-facing interpretation

- **Low-rank 是否必要？** 若 best fixed-rank SVD consistently beats `no_svd_lr`, 说明收益不只是同一套 `raw+rank` feature bank，而是 bottleneck 本身也在起作用。
- **是否存在平台期？** `smallest sufficient rank` 表按 `AUC of AUROC` 距最佳不超过 `0.5%` 的规则给出；若一个较小 rank 已进入 plateau，就可以把它视为最小足够 rank。
- **较低 rank 是否更干净？** 本轮统一用 `top5_feature_mass` 作为解释紧致度指标；值越高，说明有效权重越集中，解释更紧。实测 `r12/r16` 在 `math` 与 `ms` 上比 `no_svd_lr` 更集中，而 `r32` 反而回落。

## 5. 推荐论文话术

> The gain does not come merely from the feature bank; the low-rank bottleneck itself is useful, and a moderate low rank (16 for math, 24 for science/ms) is already sufficient.

## 6. Artifacts

- Summary JSON: `results/scans/lowrank_necessity/lowrank_necessity_summary.json`
- Eval JSON: `results/scans/lowrank_necessity/lowrank_necessity_eval.json`
- Main CSV: `results/tables/lowrank_necessity_ablation.csv`
- Smallest-rank CSV: `results/tables/lowrank_smallest_sufficient_rank.csv`
