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

- `math` diagonal mean: frozen=95.44% vs task-specific=95.62% (Δ=-0.18 pts); focus off-diagonal mean: frozen=95.05% vs task-specific=95.62% (Δ=-0.56 pts).
- `science` diagonal mean: frozen=75.04% vs task-specific=76.11% (Δ=-1.07 pts); focus off-diagonal mean: frozen=74.43% vs task-specific=76.11% (Δ=-1.68 pts).

## 4. Which source anchors transfer best?

- `math` off-diagonal transferability ranking by mean Δ(frozen−task_specific): 40% (+0.10 pts), 100% (-1.23 pts).
- `math` best reusable source anchor is `40%` (mean Δ=+0.10 pts).
- `science` off-diagonal transferability ranking by mean Δ(frozen−task_specific): 100% (-0.14 pts), 40% (-3.23 pts).
- `science` best reusable source anchor is `100%` (mean Δ=-0.14 pts).

## 5. Diagonal / Off-Diagonal / Adjacent Stability

| Domain | Slice | Frozen | Task-specific | No-SVD | Δ(Frozen−Task) | Δ(Frozen−NoSVD) |
|---|---|---:|---:|---:|---:|---:|
| math | diagonal | 95.44% | 95.62% | 95.61% | -0.18 pts | -0.17 pts |
| science | diagonal | 75.04% | 76.11% | 75.82% | -1.07 pts | -0.78 pts |
| math | offdiag focus | 95.05% | 95.62% | 95.61% | -0.56 pts | -0.55 pts |
| science | offdiag focus | 74.43% | 76.11% | 75.82% | -1.68 pts | -1.39 pts |

## 6. Paper-Facing Interpretation

- **Shared basis 是否只在 slot-100 可复用？** 如果 diagonal 与关键 off-diagonal 的 `Δ(frozen−task_specific)` 都接近 0，说明它不只在 slot-100 有效，而是在多 anchor 上仍具竞争力。
- **哪些 anchor 最 transferable？** 以上按 source-anchor 汇总的 mean Δ 排名，给出每个 domain 最可迁移的 basis 来源。
- **早期 / 晚期 anchor 学到的信号是否不同？** 若早期 basis 在 `10→40 / 10→70` 上保持竞争力、但在 `10→100` 上退化，而晚期 basis 更擅长 `40→100 / 70→100 / 100→100`，则支持“early = coarse reusable signal；late = completion-heavy signal”的叙述。
- **标题是否需要加 cross-anchor 限定？** 如果 off-diagonal 总体仍是 tie / mild loss，而不是只在 100→100 成立，那么建议在论文里把 transferable 明确写成 `task- and cross-anchor-transferable`，会更稳妥。

## 7. Artifacts

- Matrix CSV: `results/tables_smoke/cross_anchor_transfer_matrix.csv`
- Delta CSV: `results/tables_smoke/cross_anchor_transfer_deltas.csv`
- Summary CSV: `results/tables_smoke/cross_anchor_transfer_summary.csv`
