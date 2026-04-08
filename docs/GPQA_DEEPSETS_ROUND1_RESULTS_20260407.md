# GPQA DeepSets Round 1 — Results

Date: `2026-04-07`  
Finalised: `2026-04-08` after one extra tiny auxiliary probe  
Script: `scripts/run_gpqa_deepsets_round1.py`  
Official result payload: `result/gpqa_deepsets_round1_20260407_run2/gpqa_deepsets_round1.json`

## 1. Confirmed Current Repo State

Before running `gpqa_deepsets_round1`, the current repo state is:

1. `code_v2` is already the promoted coding default.
2. `science_baseline_v1` is still the frozen science baseline.
3. `gpqa_pairwise_round2` is `NO-PROMOTE`.
4. `science_hybrid_round3` already has a narrow full-system promote.
5. The repo is no longer pursuing graph-heavy extensions or a new monotonic recency feature family.
6. The real new research line should move toward the smallest viable full-group contextual model.

## 2. Implemented Minimal DeepSets Design

This round stayed inside the requested narrow scope:

- Domain: `GPQA` only
- Inputs: existing run-level structured features only
- Features:
  - `dc_z`
  - `dc_r`
  - `reflection_count_r`
  - `prefix_conf_mean_r`
  - `recency_conf_mean_r`
  - `late_recovery_r`
- Model:
  - per-run encoder MLP: `6 -> 16 -> 8`
  - pooling: `mean` or `max`
  - final score head on `[run_embedding ; pooled_group_embedding]`: `16 -> 8 -> 1`
- Training:
  - leave-one-problem-out evaluation
  - pointwise `BCE-with-logits` as the main objective
  - train-fold-only feature standardisation

Very small auxiliary extension, still within scope:

- after the minimal pointwise `mean` / `max` models were run, I tried a tiny `max + pairwise auxiliary loss` variant
- tested only two tiny auxiliary settings:
  - `pairwise_aux_weight = 0.25`
  - `pairwise_aux_weight = 0.50`
- no attention
- no Set Transformer
- no raw neuron rows
- no broad hyperparameter sweep

## 3. GPQA Single-Domain Results

### Main compare table

| Selector | AUROC | Hit@1 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| `science_baseline_v1` | 53.86% | 66.16% | 58.71% | 64.35% |
| `gpqa_pairwise_round1` | 55.66% | 63.64% | 61.15% | 62.38% |
| `science_hybrid_round3` | 53.86% | **68.18%** | 58.72% | 64.35% |
| `gpqa_deepsets_round1` = `max_pairaux0p25` | **65.38%** | 66.67% | 59.60% | **72.32%** |

Read:

- the best DeepSets round-1 candidate is `gpqa_deepsets_round1_max_pairaux0p25`
- it strongly improves global discrimination and shortlist selection:
  - `AUROC`: `+11.52pp` vs `science_hybrid_round3`
  - `Pairwise`: `+0.88pp`
  - `SelAcc@10`: `+7.97pp`
- but it still loses the current top-slot guardrail:
  - `Hit@1`: `66.67% < 68.18%`

### Small auxiliary ablation trail

| Variant | AUROC | Hit@1 | Pairwise | SelAcc@10 | Outcome |
|---|---:|---:|---:|---:|---|
| `gpqa_deepsets_round1_mean` | 65.62% | 61.62% | 58.61% | 85.73% | massive SelAcc jump, top-slot too weak |
| `gpqa_deepsets_round1_max` | 65.61% | 63.13% | 58.10% | 66.56% | better than `mean` on top-slot, still below gate |
| `gpqa_deepsets_round1_max_pairaux0p25` | 65.38% | **66.67%** | **59.60%** | 72.32% | best overall round-1 candidate |
| `gpqa_deepsets_round1_max_pairaux0p50` | 61.44% | 65.66% | 59.33% | 67.59% | regressed vs `0.25`; not adopted |

Conclusion for the single-domain slice:

- the minimal full-group contextual model is directionally real
- however, round-1 still trades away too much top-slot accuracy for promotion

## 4. Current System Proxy vs Patched System Proxy

Current promoted stack used for patching:

- generic extreme / baseline12 frozen
- coding = promoted `code_v2`
- science = current `science_hybrid_round3`

Patched candidate stack:

- only replace the science slice with `gpqa_deepsets_round1_max_pairaux0p25`
- everything else stays unchanged

### Sample-weighted proxy

| System | Hit@1 | Hit@3 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| current = `code_v2 + science_hybrid_round3` | **67.22%** | 74.02% | 61.29% | 66.15% |
| patched = `code_v2 + gpqa_deepsets_round1_max_pairaux0p25` | 66.60% | **74.43%** | **61.65%** | **69.40%** |

Sample-weighted delta vs current:

- `Hit@1`: `-0.62pp`
- `Hit@3`: `+0.41pp`
- `Pairwise`: `+0.36pp`
- `SelAcc@10`: `+3.25pp`
- `AvgRank proxy`: `-0.0165` (better)

### Equal-cache-mean proxy

| System | Hit@1 | Hit@3 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| current = `code_v2 + science_hybrid_round3` | **70.53%** | 77.71% | 71.53% | 70.90% |
| patched = `code_v2 + gpqa_deepsets_round1_max_pairaux0p25` | 70.28% | **77.88%** | **71.68%** | **72.23%** |

Equal-cache-mean delta vs current:

- `Hit@1`: `-0.25pp`
- `Hit@3`: `+0.17pp`
- `Pairwise`: `+0.15pp`
- `SelAcc@10`: `+1.33pp`
- `AvgRank proxy`: `-0.0067` (better)

System judgement:

- this patch is **not truly better than the current promoted system**
- it improves shortlist-oriented metrics
- but it still lowers the system `Hit@1`, which remains the current promote-sensitive guardrail

## 5. Offline–Online Alignment Mini Analysis

The repo does not currently contain a trustworthy actual leaderboard delta table.
So the analysis below is limited to proxy movement and historical local patch direction.

| Patch | sample-weighted Hit@1 | sample-weighted SelAcc@10 | sample-weighted AvgRank proxy | equal-cache Hit@1 | equal-cache SelAcc@10 | Actual score change |
|---|---:|---:|---:|---:|---:|---|
| `code_v2` | +0.83pp | +1.36pp | +0.0000 | +0.40pp | +0.66pp | unavailable in repo |
| `science_baseline_v1` | +2.27pp | +0.56pp | +0.0000 | +0.93pp | +0.23pp | unavailable in repo |
| `science_hybrid_round3` | +0.82pp | +0.00pp | -0.0082 | +0.34pp | +0.00pp | unavailable in repo |
| `gpqa_deepsets_round1` | -0.62pp | +3.25pp | -0.0165 | -0.25pp | +1.33pp | unavailable in repo |

Interpretation:

- inside the current repo evidence, the proxy direction that looks most like the promote-sensitive direction is:
  - sample-weighted `Hit@1`
  - with `avg_rank_proxy` as a supporting read
- why:
  - `code_v2` and `science_hybrid_round3` both preserved or improved sample-weighted `Hit@1`
  - `gpqa_deepsets_round1` mainly improves shortlist metrics while still lowering `Hit@1`
- important warning:
  - the repo does **not** contain reliable actual public/private leaderboard deltas
  - so proxy-to-leaderboard alignment is still **unverified**
  - I cannot honestly claim that proxy and leaderboard are often consistent or often inconsistent from the local repo alone

## 6. Promote / No-Promote Decision

Decision: **NO-PROMOTE**

Reason:

1. GPQA single-domain top-slot gate is not passed:
   - current `science_hybrid_round3` `Hit@1` = `68.18%`
   - best `gpqa_deepsets_round1` `Hit@1` = `66.67%`
2. Patched full-system proxy is not a guarded improvement:
   - sample-weighted `Hit@1` goes from `67.22%` to `66.60%`
   - equal-cache-mean `Hit@1` goes from `70.53%` to `70.28%`
3. This round mostly improves `AUROC` / `Pairwise` / `SelAcc@10`, not the decisive top-slot metric

Operational consequence:

- keep `science_hybrid_round3` as the current promoted science slice
- do **not** replace the current promoted stack with `gpqa_deepsets_round1`

## 7. Changed Files

Implemented / added:

- `nad/core/selectors/gpqa_deepsets_impl.py`
- `plugins/gpqa_deepsets_selector.py`
- `scripts/run_gpqa_deepsets_round1.py`
- `docs/GPQA_DEEPSETS_ROUND1_PLAN_20260407.md`
- `docs/GPQA_DEEPSETS_ROUND1_RESULTS_20260407.md`

Generated experiment artefacts:

- `models/ml_selectors/gpqa_deepsets_round1_mean.pkl`
- `models/ml_selectors/gpqa_deepsets_round1_max.pkl`
- `models/ml_selectors/gpqa_deepsets_round1_max_pairaux0p25.pkl`
- `models/ml_selectors/gpqa_deepsets_round1_max_pairaux0p50.pkl`
- `models/ml_selectors/gpqa_deepsets_round1.pkl` (reset to the best round-1 artefact: `max_pairaux0p25`)
- `result/gpqa_deepsets_round1_20260407_run1/`
- `result/gpqa_deepsets_round1_20260407_run2/`
- `result/gpqa_deepsets_round1_20260408_run3/`

## 8. How To Re-run

Recommended official rerun:

```bash
source .venv/bin/activate
python scripts/run_gpqa_deepsets_round1.py \
  --out-dir result/gpqa_deepsets_round1_$(date -u +%Y%m%d_%H%M%S) \
  --workers 8 \
  --distance-threads 2 \
  --torch-threads 8 \
  --pairwise-aux-weight 0.25
```

Optional tiny extra probe only:

```bash
source .venv/bin/activate
python scripts/run_gpqa_deepsets_round1.py \
  --out-dir result/gpqa_deepsets_round1_aux050_$(date -u +%Y%m%d_%H%M%S) \
  --workers 8 \
  --distance-threads 2 \
  --torch-threads 8 \
  --pairwise-aux-weight 0.50
```

## 9. Is It Worth Continuing To Cross-Run Attention?

Not yet.

Round-1 already shows that minimal group context can move GPQA strongly on
`AUROC` / `SelAcc@10`, but it still misses the top-slot gate that matters for
promotion. So the next step should **not** automatically jump to cross-run
attention.

If this line is continued later, the cleaner next move is:

- keep the architecture small
- keep the feature family fixed
- target top-slot calibration directly with a narrow ranking-aware objective
- only revisit attention if a small non-attention contextual model first proves it can clear the top-slot gate

---

# GPQA DeepSets Round 1 — 结果（中文）

日期：`2026-04-07`  
定稿：`2026-04-08`，在补做一个极小辅助探针后完成  
脚本：`scripts/run_gpqa_deepsets_round1.py`  
正式结果载荷：`result/gpqa_deepsets_round1_20260407_run2/gpqa_deepsets_round1.json`

## 1. 当前仓库状态确认

在运行 `gpqa_deepsets_round1` 之前，当前仓库状态如下：

1. `code_v2` 已经是 promoted coding default。
2. `science_baseline_v1` 仍然是 frozen science baseline。
3. `gpqa_pairwise_round2` 的结论是 `NO-PROMOTE`。
4. `science_hybrid_round3` 已经拿到一个 narrow full-system promote。
5. 当前主线已经不再继续 graph-heavy 扩展，也不再新增 monotonic recency feature family。
6. 真正的新研究线应转向最小可行的 full-group contextual model。

## 2. 这次实现的最小 DeepSets 设计

本轮严格保持在用户要求的窄范围内：

- 领域：只做 `GPQA`
- 输入：只用现有 run-level structured features
- 特征：
  - `dc_z`
  - `dc_r`
  - `reflection_count_r`
  - `prefix_conf_mean_r`
  - `recency_conf_mean_r`
  - `late_recovery_r`
- 模型：
  - 每条 run 的 encoder MLP：`6 -> 16 -> 8`
  - pooling：`mean` 或 `max`
  - 最终打分 head 使用 `[run_embedding ; pooled_group_embedding]`：`16 -> 8 -> 1`
- 训练：
  - leave-one-problem-out 评估
  - 主目标是 pointwise `BCE-with-logits`
  - 标准化只在 train fold 上拟合

非常小的辅助扩展，仍在允许范围内：

- 先把最小 pointwise `mean` / `max` 跑通
- 然后只补了一个极小的 `max + pairwise auxiliary loss` 版本
- 只试了两个极小辅助强度：
  - `pairwise_aux_weight = 0.25`
  - `pairwise_aux_weight = 0.50`
- 不用 attention
- 不用 Set Transformer
- 不用 raw neuron rows
- 不做大范围超参扫描

## 3. GPQA 单域结果

### 主对比表

| Selector | AUROC | Hit@1 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| `science_baseline_v1` | 53.86% | 66.16% | 58.71% | 64.35% |
| `gpqa_pairwise_round1` | 55.66% | 63.64% | 61.15% | 62.38% |
| `science_hybrid_round3` | 53.86% | **68.18%** | 58.72% | 64.35% |
| `gpqa_deepsets_round1` = `max_pairaux0p25` | **65.38%** | 66.67% | 59.60% | **72.32%** |

解读：

- 本轮最佳 DeepSets 候选是 `gpqa_deepsets_round1_max_pairaux0p25`
- 它在整体判别和 shortlist 选择上提升明显：
  - `AUROC`: 相比 `science_hybrid_round3` 提升 `+11.52pp`
  - `Pairwise`: `+0.88pp`
  - `SelAcc@10`: `+7.97pp`
- 但它仍然输掉当前最关键的 top-slot guardrail：
  - `Hit@1`: `66.67% < 68.18%`

### 小型辅助消融轨迹

| Variant | AUROC | Hit@1 | Pairwise | SelAcc@10 | 结论 |
|---|---:|---:|---:|---:|---|
| `gpqa_deepsets_round1_mean` | 65.62% | 61.62% | 58.61% | 85.73% | SelAcc 暴涨，但 top-slot 太弱 |
| `gpqa_deepsets_round1_max` | 65.61% | 63.13% | 58.10% | 66.56% | 比 `mean` 更接近 top-slot，但仍未过门 |
| `gpqa_deepsets_round1_max_pairaux0p25` | 65.38% | **66.67%** | **59.60%** | 72.32% | 本轮最优候选 |
| `gpqa_deepsets_round1_max_pairaux0p50` | 61.44% | 65.66% | 59.33% | 67.59% | 相比 `0.25` 回退，不采用 |

单域结论：

- 最小 full-group contextual model 这个方向本身是成立的
- 但 round-1 仍然为 promotion 付出了过多的 top-slot 损失

## 4. 当前系统 proxy vs patched system proxy

当前 promoted stack：

- generic extreme / baseline12 frozen
- coding = promoted `code_v2`
- science = 当前 `science_hybrid_round3`

新的 patched 候选系统：

- 只把 science slice 替换成 `gpqa_deepsets_round1_max_pairaux0p25`
- 其余全部保持不变

### Sample-weighted proxy

| System | Hit@1 | Hit@3 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| current = `code_v2 + science_hybrid_round3` | **67.22%** | 74.02% | 61.29% | 66.15% |
| patched = `code_v2 + gpqa_deepsets_round1_max_pairaux0p25` | 66.60% | **74.43%** | **61.65%** | **69.40%** |

相对当前系统的 sample-weighted 变化：

- `Hit@1`: `-0.62pp`
- `Hit@3`: `+0.41pp`
- `Pairwise`: `+0.36pp`
- `SelAcc@10`: `+3.25pp`
- `AvgRank proxy`: `-0.0165`（更好）

### Equal-cache-mean proxy

| System | Hit@1 | Hit@3 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| current = `code_v2 + science_hybrid_round3` | **70.53%** | 77.71% | 71.53% | 70.90% |
| patched = `code_v2 + gpqa_deepsets_round1_max_pairaux0p25` | 70.28% | **77.88%** | **71.68%** | **72.23%** |

相对当前系统的 equal-cache-mean 变化：

- `Hit@1`: `-0.25pp`
- `Hit@3`: `+0.17pp`
- `Pairwise`: `+0.15pp`
- `SelAcc@10`: `+1.33pp`
- `AvgRank proxy`: `-0.0067`（更好）

系统判断：

- 这个 patch **不能算真正优于当前 promoted system**
- 它提高了 shortlist 指标
- 但它仍然降低了系统 `Hit@1`，而这仍是当前 promotion 最敏感的 guardrail

## 5. offline–online alignment 小分析

当前仓库里并没有一张可信的 actual leaderboard delta 对账表。
因此下面的分析只能基于 proxy 变化和历史本地 patch 方向。

| Patch | sample-weighted Hit@1 | sample-weighted SelAcc@10 | sample-weighted AvgRank proxy | equal-cache Hit@1 | equal-cache SelAcc@10 | 实际分数变化 |
|---|---:|---:|---:|---:|---:|---|
| `code_v2` | +0.83pp | +1.36pp | +0.0000 | +0.40pp | +0.66pp | 仓库中无记录 |
| `science_baseline_v1` | +2.27pp | +0.56pp | +0.0000 | +0.93pp | +0.23pp | 仓库中无记录 |
| `science_hybrid_round3` | +0.82pp | +0.00pp | -0.0082 | +0.34pp | +0.00pp | 仓库中无记录 |
| `gpqa_deepsets_round1` | -0.62pp | +3.25pp | -0.0165 | -0.25pp | +1.33pp | 仓库中无记录 |

解读：

- 在当前仓库能看到的证据里，最像 promote-sensitive direction 的 proxy 是：
  - sample-weighted `Hit@1`
  - 再配合 `avg_rank_proxy` 辅助观察
- 原因：
  - `code_v2` 和 `science_hybrid_round3` 都保持或提升了 sample-weighted `Hit@1`
  - `gpqa_deepsets_round1` 则主要提升 shortlist 指标，但仍然拉低 `Hit@1`
- 重要警告：
  - 仓库中**没有**可靠的 public/private leaderboard delta
  - 因此 proxy 到 leaderboard 的映射仍然是**未验证**的
  - 仅凭本地仓库，我不能诚实地声称 proxy 和 leaderboard 经常一致，或经常不一致

## 6. Promote / No-Promote 决策

结论：**NO-PROMOTE**

原因：

1. GPQA 单域 top-slot gate 没过：
   - 当前 `science_hybrid_round3` 的 `Hit@1` = `68.18%`
   - 最佳 `gpqa_deepsets_round1` 的 `Hit@1` = `66.67%`
2. patched full-system proxy 不是 guarded improvement：
   - sample-weighted `Hit@1` 从 `67.22%` 降到 `66.60%`
   - equal-cache-mean `Hit@1` 从 `70.53%` 降到 `70.28%`
3. 本轮主要提升的是 `AUROC` / `Pairwise` / `SelAcc@10`，而不是决定 promotion 的 top-slot 指标

实际操作结论：

- 继续保留 `science_hybrid_round3` 作为当前 promoted science slice
- **不要**用 `gpqa_deepsets_round1` 替换当前 promoted stack

## 7. 改动文件

实现 / 新增：

- `nad/core/selectors/gpqa_deepsets_impl.py`
- `plugins/gpqa_deepsets_selector.py`
- `scripts/run_gpqa_deepsets_round1.py`
- `docs/GPQA_DEEPSETS_ROUND1_PLAN_20260407.md`
- `docs/GPQA_DEEPSETS_ROUND1_RESULTS_20260407.md`

生成的实验产物：

- `models/ml_selectors/gpqa_deepsets_round1_mean.pkl`
- `models/ml_selectors/gpqa_deepsets_round1_max.pkl`
- `models/ml_selectors/gpqa_deepsets_round1_max_pairaux0p25.pkl`
- `models/ml_selectors/gpqa_deepsets_round1_max_pairaux0p50.pkl`
- `models/ml_selectors/gpqa_deepsets_round1.pkl`（已重新指回本轮最佳 artefact：`max_pairaux0p25`）
- `result/gpqa_deepsets_round1_20260407_run1/`
- `result/gpqa_deepsets_round1_20260407_run2/`
- `result/gpqa_deepsets_round1_20260408_run3/`

## 8. 如何复跑

建议使用的正式复跑命令：

```bash
source .venv/bin/activate
python scripts/run_gpqa_deepsets_round1.py \
  --out-dir result/gpqa_deepsets_round1_$(date -u +%Y%m%d_%H%M%S) \
  --workers 8 \
  --distance-threads 2 \
  --torch-threads 8 \
  --pairwise-aux-weight 0.25
```

仅用于补一个极小辅助探针时：

```bash
source .venv/bin/activate
python scripts/run_gpqa_deepsets_round1.py \
  --out-dir result/gpqa_deepsets_round1_aux050_$(date -u +%Y%m%d_%H%M%S) \
  --workers 8 \
  --distance-threads 2 \
  --torch-threads 8 \
  --pairwise-aux-weight 0.50
```

## 9. 是否值得继续到 cross-run attention？

现在还不值得。

Round-1 已经说明：最小 group context 确实能显著推动 `AUROC` / `SelAcc@10`，
但它仍然没有过掉 promotion 真正在意的 top-slot gate。所以这条线的下一步
**不应该**自动跳到 cross-run attention。

如果未来继续这条线，更干净的下一步应该是：

- 保持架构小
- 保持 feature family 不变
- 直接针对 top-slot calibration 做一个窄的 ranking-aware objective
- 只有在小型 non-attention contextual model 先证明自己能过 top-slot gate 之后，再考虑 attention
