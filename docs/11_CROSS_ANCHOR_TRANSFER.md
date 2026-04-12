# 11. Cross-Anchor Transfer

**Experiment script**: `SVDomain/run_cross_anchor_transfer.py`
**Source bundle**: `models/ml_selectors/es_svd_ms_rr_r1.pkl`
**Anchors**: `10, 40, 70, 100`
**Domains**: `math, science`
**Date**: 2026-04-12

---

## 1. Claim Under Test

> The shared low-rank basis is not only reusable at slot-100, but remains competitive across multiple trajectory anchors with lightweight anchor-specific heads.

这里的核心问题不是再补更多 `100%` 数字，而是测试：同一个 low-rank basis 在整条 reasoning trajectory 上是否仍然可复用。

## 2. Protocol

- 条件对比：`frozen_basis` vs `task_specific` vs `no_svd`。
- `frozen_basis`：使用 **source anchor** 的 `scaler + SVD`，只在 **target anchor** 上重训 LR head。
- `task_specific`：在 **target anchor** 上重训 `scaler + SVD + LR head`。
- `no_svd`：在 **target anchor** 上直接训练 `StandardScaler + LR`。
- 评估：沿用现有 `GroupKFold` offline protocol，不改 split；本表默认关注 `AUROC`，并同步导出 `SelAcc@10%` / `StopAcc`。

## 3. Headline Summary

- `math` diagonal mean: frozen=94.02% vs task-specific=94.13% (Δ=-0.11 pts); focus off-diagonal mean: frozen=95.56% vs task-specific=95.89% (Δ=-0.32 pts).
- `science` diagonal mean: frozen=73.34% vs task-specific=74.11% (Δ=-0.77 pts); focus off-diagonal mean: frozen=74.38% vs task-specific=76.35% (Δ=-1.98 pts).

## 4. Which source anchors transfer best?

- `math` off-diagonal transferability ranking by mean Δ(frozen−task_specific): 40% (-0.19 pts), 10% (-0.26 pts), 70% (-0.49 pts), 100% (-1.20 pts).
- `math` best reusable source anchor is `40%` (mean Δ=-0.19 pts).
- `science` off-diagonal transferability ranking by mean Δ(frozen−task_specific): 100% (-0.11 pts), 70% (-1.21 pts), 40% (-1.54 pts), 10% (-4.22 pts).
- `science` best reusable source anchor is `100%` (mean Δ=-0.11 pts).

## 5. Diagonal / Off-Diagonal / Adjacent Stability

| Domain | Slice | Frozen | Task-specific | No-SVD | Δ(Frozen−Task) | Δ(Frozen−NoSVD) |
|---|---|---:|---:|---:|---:|---:|
| math | diagonal | 94.02% | 94.13% | 94.25% | -0.11 pts | -0.23 pts |
| science | diagonal | 73.34% | 74.11% | 74.51% | -0.77 pts | -1.17 pts |
| math | offdiag focus | 95.56% | 95.89% | 96.01% | -0.32 pts | -0.45 pts |
| science | offdiag focus | 74.38% | 76.35% | 76.59% | -1.98 pts | -2.22 pts |
| math | adjacent forward | 95.55% | 95.66% | 95.79% | -0.10 pts | -0.23 pts |
| science | adjacent forward | 74.11% | 76.03% | 76.13% | -1.93 pts | -2.02 pts |
| math | adjacent backward | 92.97% | 93.31% | 93.44% | -0.34 pts | -0.47 pts |
| science | adjacent backward | 72.40% | 73.05% | 73.36% | -0.64 pts | -0.95 pts |

### Highlighted off-diagonal pairs

| Domain | Pair | Frozen | Task-specific | No-SVD | Δ(Frozen−Task) | Verdict |
|---|---|---:|---:|---:|---:|---|
| math | 100->40 | 93.69% | 94.56% | 94.77% | -0.88 pts | tie |
| math | 100->70 | 95.61% | 95.83% | 95.91% | -0.21 pts | tie |
| math | 40->100 | 96.57% | 96.58% | 96.68% | -0.01 pts | tie |
| math | 10->100 | 96.39% | 96.58% | 96.68% | -0.19 pts | tie |
| science | 100->40 | 74.61% | 74.68% | 74.35% | -0.07 pts | tie |
| science | 100->70 | 76.03% | 76.11% | 76.05% | -0.09 pts | tie |
| science | 40->100 | 75.16% | 77.31% | 77.98% | -2.14 pts | loss |
| science | 10->100 | 71.71% | 77.31% | 77.98% | -5.60 pts | loss |

## 6. Direct Answers

- **Shared basis 是否只在 slot-100 可复用？** `math` 上答案是 **不是**：diagonal mean 仅比 task-specific 低 `-0.11 pts`，关键 off-diagonal 也只低 `-0.32 pts`，而且所有 math anchor-pairs 都仍是 `tie/win`。
- `science` 上答案是 **部分成立**：late-anchor basis 仍可复用（`100→40` 只差 -0.07 pts，`100→70` 只差 -0.09 pts），但 `10→40` 已掉到 -2.53 pts，`10→100` 更掉到 -5.60 pts。这说明 cross-anchor transfer 在 science 上并不均匀。
- **哪些 anchor 最 transferable？** `math` 上最可迁移的 source anchor 是 `40%`，其 off-diagonal mean Δ(frozen−task_specific) = `-0.19 pts`。
- **哪些 anchor 最 transferable？** `science` 上最可迁移的 source anchor 是 `100%`，其 off-diagonal mean Δ(frozen−task_specific) = `-0.11 pts`。
- **早期 anchor 是否只学到 coarse signal？** `math` 不是：`10→100` 仍只差 `-0.19 pts`，说明早期 basis 已经捕获到大部分稳定决策边界。`science` 更像是 **yes**：`10→100` 掉到 `-5.60 pts`，而 `70→100` 只差 `-1.54 pts`，说明早期 science anchor 更像 coarse / under-formed signal，late anchor 才带有 completion-heavy information。
- **论文标题里的 transferable 要不要加 cross-anchor 限定？** 建议写成 **`task- and cross-anchor-transferable`**，但正文要补一句限定：该结论在 `math` 上几乎贯穿全 trajectory，在 `science` 上则主要由 `70/100` late-anchor basis 支撑。

## 7. Artifacts

- Matrix CSV: `results/tables/cross_anchor_transfer_matrix.csv`
- Delta CSV: `results/tables/cross_anchor_transfer_deltas.csv`
- Summary CSV: `results/tables/cross_anchor_transfer_summary.csv`
