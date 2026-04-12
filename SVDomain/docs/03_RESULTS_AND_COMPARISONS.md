# Results and Comparisons

这份文档整理当前与论文最相关的结果，包括：

- holdout 对比
- blind leaderboard
- best-of-n slot100 对照
- checkpoint ranking side task

---

## 1. 主结论先说

### 1.1 成功结论

当前最重要的成功结论是：

- `es_svd_ms_rr_r1__coding_from_round1c` 在 blind leaderboard 上优于 previous best `earlystop_prefix10_svd_round1`

对应改进：

- `primary score`: `4.0000 -> 3.8125`
- `auc_of_auroc`: `0.7379 -> 0.7428`
- `auc_of_selacc`: `0.8311 -> 0.8317`

### 1.2 负结果同样重要

另外两个结果也很重要：

- `es_svd_coding_rr_r1` 没有带来整体收益
- `slot100` 直抽 best-of-n 不是当前 best

这两条结果说明：

- domain-aware 建模是必要的
- simple final-slot extraction 不足以解释整体 best-of-n 最优解

---

## 2. Holdout：math / science / combined / coding

### 2.1 Math holdout

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | 68.52% | 94.23% | 10% | 68.38% | 75.00% |
| earlystop_svd_lowrank_lr_v1 | 77.69% | 95.60% | 10% | 90.47% | 71.43% |
| earlystop_prefix10_svd_round1 | 93.48% | 99.95% | 10% | 96.30% | 78.57% |
| es_svd_math_rr_r1 | **95.81%** | 99.73% | 10% | **98.17%** | 75.00% |

结论：

- `es_svd_math_rr_r1` 是 math 域最强证据之一
- 它在 `AUC of AUROC` 上明确超过旧主线
- `Stop Acc@100%` 不一定处处占优，但总体曲线更强

### 2.2 Science holdout

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | **80.89%** | 96.30% | 10% | **85.61%** | 70.00% |
| earlystop_svd_lowrank_lr_v1 | 77.19% | 96.59% | 10% | 79.37% | **76.67%** |
| earlystop_prefix10_svd_round1 | 77.30% | 97.97% | 10% | 79.84% | 73.33% |
| es_svd_science_rr_r1 | 79.85% | **98.80%** | 10% | 84.11% | 65.00% |

结论：

- science 没有 math 那样“一边倒”
- `es_svd_science_rr_r1` 的主要优势在 selective-accuracy 曲线与 canonical clean design
- 这更适合作为“稳定化 / 结构化”的证据，而不是绝对压倒性增益

### 2.3 Combined noncoding holdout

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | 71.27% | 94.69% | 10% | 72.21% | 73.89% |
| earlystop_svd_lowrank_lr_v1 | 77.58% | 95.82% | 10% | 88.01% | 72.59% |
| earlystop_prefix10_svd_round1 | 89.88% | 99.51% | 10% | 92.64% | **77.41%** |
| es_svd_ms_rr_r1 | **92.26%** | **99.52%** | 10% | **95.05%** | 72.78% |

结论：

- 这张表是最适合放正文的 holdout 主表
- 它说明 canonical multi-domain bundle 的总体收益是真实存在的

### 2.4 Coding holdout

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | **50.62%** | **45.81%** | N/A | **49.04%** | 48.00% |
| earlystop_svd_lowrank_lr_v1 | 47.62% | 38.56% | N/A | 47.33% | **56.00%** |
| earlystop_prefix10_svd_round1 | 47.62% | 38.56% | N/A | 47.33% | **56.00%** |
| es_svd_coding_rr_r1 | 43.42% | 24.00% | N/A | 40.68% | 52.00% |

结论：

- coding 结果明确不支持“把同一 canonical family 直接推广到 coding”
- 这是需要在 discussion 里诚实报告的点

---

## 3. Blind leaderboard：主 EarlyStop 提交

### 3.1 当前主提交

- `submission id`: `#144`
- `method_name`: `es_svd_ms_rr_r1__coding_from_round1c`
- `status`: `CURRENT BEST`
- `primary score`: `3.8125`
- `auc_of_auroc`: `0.7428`
- `auc_of_selacc`: `0.8317`

### 3.2 对比 previous best

| Metric | Previous Best | Current | Delta |
|---|---:|---:|---:|
| Primary Score | 4.0000 | **3.8125** | **-0.1875** |
| AUC of AUROC | 0.7379 | **0.7428** | **+0.0049** |
| AUC of SelAcc | 0.8311 | **0.8317** | **+0.0006** |
| AUROC@100% | **0.8492** | 0.8468 | -0.0024 |
| Stop Acc@100% | **0.7504** | 0.7299 | -0.0205 |

解读：

- 这是一个“曲线整体变好，但 100% 终点不一定更强”的结果
- 对 early-stop selection 来说，这种模式是完全合理的
- 因为我们优化的是整个决策过程，而不是只优化 final slot

---

## 4. Coding blind merge：不建议替换主提交

### 4.1 结果

- `submission id`: `#147`
- `method_name`: `es_svd_ms_rr_r1__coding_rr_r1`
- `status`: `Not best`
- `primary score`: `3.8125`
- `auc_of_auroc`: `0.7427`
- `auc_of_selacc`: `0.8276`

### 4.2 对 base submission 的变化

| Metric | Base | Coding Merge | Delta |
|---|---:|---:|---:|
| Primary Score | 3.8125 | 3.8125 | 0.0000 |
| AUC of AUROC | **0.7428** | 0.7427 | -0.0001 |
| AUC of SelAcc | **0.8317** | 0.8276 | -0.0041 |
| AUROC@100% | 0.8468 | **0.8475** | +0.0007 |
| Stop Acc@100% | **0.7299** | 0.7279 | -0.0020 |

结论：

- coding merge 没有提供足够收益
- 不应在论文中把它包装成“主线升级”
- 更适合作为 negative result / appendix

---

## 5. Best-of-N：slot100 直抽

### 5.1 当前结果

- `method_name`: `es_svd_ms_rr_r1__coding_from_round1c__slot100`
- `status`: `Not best`
- `avg rank`: `3.6000`
- `auroc`: `0.8468`
- `hit@1`: `0.7299`
- `hit@3`: `0.8010`
- `selacc@10%`: `0.9313`
- `pairwise acc`: `0.7396`

### 5.2 对比历史最佳 best-of-n

| Metric | Historical Best | Current Slot100 | Delta |
|---|---:|---:|---:|
| Avg Rank | **2.4000** | 3.6000 | +1.2000 |
| Hit@1 | **0.7504** | 0.7299 | -0.0205 |
| Hit@3 | **0.8049** | 0.8010 | -0.0039 |
| SelAcc@10% | 0.9286 | **0.9313** | +0.0027 |
| Pairwise Acc | **0.7510** | 0.7396 | -0.0114 |

结论：

- `slot100` 直抽不是最优 best-of-n 策略
- 它只在 `SelAcc@10%` 上略有优势
- 说明 best-of-n 需要的不只是 final-slot score

---

## 6. RL checkpoint ranking side task

当前有一个 side task：

- `es_svd_math_rr_r1__math5000rl_slot100_meanconf`

在线结果：

- `rank = 3`
- `Spearman ρ = 0.7364`
- `Pearson r = 0.8398`
- `Kendall τ = 0.6000`
- `Top-1 hit = 0`
- `Top-3 hit = 0`

这个任务更适合放在：

- appendix
- transfer / auxiliary evaluation
- “SVD family can generalize to another ranking task” 的补充证据

不建议与主 early-stop 结果抢正文篇幅。

---

## 7. 建议在论文中怎么组织结果

### 正文主表

最推荐放进正文的表有三张：

1. holdout 主表：`combined noncoding`
2. math / science 分域表
3. blind leaderboard 对比表：`#115 vs #144`

### appendix 表

建议放进 appendix 的表：

1. coding holdout
2. coding blind merge
3. slot100 best-of-n
4. checkpoint ranking

---

## 8. 结果叙事上的最重要信息

整套结果最值得强调的不是“每个指标都赢”，而是：

1. canonical `r1` 在 noncoding 主线确实更强
2. 它的收益主要体现在 **整体 early-stop 决策曲线**
3. coding 和纯 slot100 抽取都说明：
   - domain mismatch 是真实问题
   - final-slot-only 不是完整答案

---

## 9. Dense trajectory transfer

The `r2` line answers a more specific representation-level question: is shared low-rank transfer limited to sparse `10/40/70/100` anchors, or does it persist across the full dense `10/20/.../100` trajectory?

### 9.1 Dense cross-anchor 主结论

| Domain | Diagonal Δ(Frozen−Task) | Offdiag-all Δ | Near-gap Δ (`10/20`) | Far-gap Δ (`50–90`) | Best source anchor |
|---|---:|---:|---:|---:|---:|
| `math` | -0.13 pts | -0.34 pts | -0.16 pts | -0.62 pts | `30%` |
| `science` | -0.09 pts | -0.54 pts | -0.23 pts | -0.97 pts | `50%` |

Readout:

- In `math`, basis reuse is not a slot-100-only phenomenon; it remains competitive across almost the entire trajectory.
- In `science`, transfer is also real, but it depends much more strongly on anchor maturity, especially for **early-to-late** forward reuse.

### 9.2 方向性很重要

| Domain | Forward all Δ | Backward all Δ | Worst pair |
|---|---:|---:|---|
| `math` | -0.11 pts | -0.56 pts | `100→10` = -2.51 pts |
| `science` | -0.98 pts | -0.10 pts | `10→50` = -4.17 pts |

Interpretation:

- In `math`, the main failure mode is **late-to-early backward** transfer, while forward reuse into later targets remains very stable.
- In `science`, the main failure mode is **early-to-late forward** transfer, suggesting that early-anchor bases have not yet matured into late-stage reusable representations.

### 9.3 与 dense timing 一起读

This result should be read together with `docs/16_DENSE_ANCHOR_EARLYSTOP.md`:

- `math`: reaches 95%-of-final at `10%` and plateaus around `50%`, so weak distance-decay under dense transfer is exactly what we would expect.
- `science`: reaches 95%-of-final at `20%` and plateaus around `40%`, so the denser picture is better described as “early coarse signal + late refinement” than “only-at-completion onset”.

Dense transfer is therefore a key representation-level result for the canonical SVD narrative: **the representation is genuinely shared, but the ease of reuse depends on domain, direction, and anchor distance.**
