# Code DeepSets Round 1 — Results

Date: `2026-04-08`  
Script: `scripts/run_code_deepsets_round1.py`  
Official result payload: `result/code_deepsets_round1_20260408_run2/code_deepsets_round1.json`

## Current Repo State

Before this run:

1. `code_v2` is already the promoted coding default.
2. `science_hybrid_round3` is already the promoted science patch.
3. `gpqa_deepsets_round1` completed the first minimal science-contextual DeepSets study but remained `NO-PROMOTE`.
4. The current DeepSets extension question is whether the same minimal full-group contextual idea transfers to coding and math.

## Minimal DeepSets Design

Coding DeepSets round-1 stays deliberately small:

- domain: `DS-R1/lcb_v5`
- inputs: only the current `code_v2` structured signals
- features:
  - `prefix_best_window_quality_r`
  - `head_tail_gap_r`
  - `tail_variance_r`
  - `post_reflection_recovery_r`
  - `last_block_instability_r`
- model:
  - per-run encoder MLP
  - tiny hidden sizes
  - pooled group context with `mean` / `max`
  - final score from `run embedding + pooled group embedding`
- tiny extra probe:
  - `max + pairwise_aux_weight=0.25`

No attention, no Set Transformer, no raw neuron rows, and no broad hyperparameter search.

## Coding Single-Domain Results

| Selector | AUROC | Hit@1 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| `code_baseline_v1` | 50.27% | 59.28% | 51.27% | 61.74% |
| `code_v2` | 50.29% | 61.68% | 51.00% | **62.11%** |
| `code_deepsets_round1_mean` | 48.79% | 58.68% | 50.12% | 58.65% |
| `code_deepsets_round1_max` | **52.47%** | **62.28%** | 50.29% | 55.75% |
| `code_deepsets_round1_max_pairaux0p25` | 51.67% | 59.28% | 49.08% | 59.96% |

Selected candidate:

- `code_deepsets_round1_max`

Read:

- minimal group context does give a small top-slot lift over `code_v2`
  - `Hit@1`: `61.68% -> 62.28%`
  - `AUROC`: `50.29% -> 52.47%`
- but the candidate sharply hurts shortlist quality
  - `SelAcc@10`: `62.11% -> 55.75%`
- the pairwise-aux probe does not repair this

So coding DeepSets round-1 improves top-1 a bit, but the operating point is not good enough.

## Current System Proxy vs Patched System Proxy

Current promoted stack:

- generic extreme / baseline12 frozen
- coding = promoted `code_v2`
- science = promoted `science_hybrid_round3`

Patched candidate stack:

- replace coding only with `code_deepsets_round1_max`

### Sample-weighted proxy

| System | Hit@1 | Hit@3 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| current = `code_v2 + science_hybrid_round3` | 67.22% | 74.02% | **61.29%** | **66.15%** |
| patched = `code_deepsets_round1_max + science_hybrid_round3` | **67.42%** | **74.23%** | 61.05% | 63.96% |

Delta vs current:

- `Hit@1`: `+0.21pp`
- `Hit@3`: `+0.21pp`
- `Pairwise`: `-0.24pp`
- `SelAcc@10`: `-2.19pp`

### Equal-cache-mean proxy

| System | Hit@1 | Hit@3 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| current = `code_v2 + science_hybrid_round3` | 70.53% | 77.71% | **71.53%** | **70.90%** |
| patched = `code_deepsets_round1_max + science_hybrid_round3` | **70.63%** | **77.81%** | 71.42% | 69.84% |

Delta vs current:

- `Hit@1`: `+0.10pp`
- `Hit@3`: `+0.10pp`
- `Pairwise`: `-0.12pp`
- `SelAcc@10`: `-1.06pp`

Interpretation:

- the candidate raises top-slot system proxy slightly
- but it loses too much shortlist quality, so the current guarded system gate still rejects it

## Decision

Decision: **NO-PROMOTE**

Reason:

1. Coding gate fails:
   - `SelAcc@10` is below current `code_v2`
2. System gate fails:
   - sample-weighted `Hit@1` improves, but `SelAcc@10` drops materially
   - this is not a guarded improvement under the current proxy rule

Operational read:

- coding DeepSets round-1 does show non-trivial contextual signal
- but this first minimal version is not ready to replace `code_v2`

## Files

Added / changed for this line:

- `nad/core/selectors/deepsets_core.py`
- `nad/core/selectors/code_deepsets_impl.py`
- `plugins/code_deepsets_selector.py`
- `scripts/run_code_deepsets_round1.py`
- `docs/CODE_DEEPSETS_ROUND1_PLAN_20260408.md`
- `docs/CODE_DEEPSETS_ROUND1_RESULTS_20260408.md`

Generated artifacts:

- `models/ml_selectors/code_deepsets_round1.pkl`
- `models/ml_selectors/code_deepsets_round1_mean.pkl`
- `models/ml_selectors/code_deepsets_round1_max.pkl`
- `models/ml_selectors/code_deepsets_round1_max_pairaux0p25.pkl`
- `result/code_deepsets_round1_20260408_run2/`

## Re-run

```bash
source .venv/bin/activate
python scripts/run_code_deepsets_round1.py \
  --out-dir result/code_deepsets_round1_$(date -u +%Y%m%d_%H%M%S) \
  --distance-threads 8 \
  --torch-threads 8
```

## Next Read

If this line is continued, the more plausible next move is not bigger context,
but better top-slot vs shortlist calibration inside the existing small feature family.

---

# Code DeepSets Round 1 — 结果（中文）

日期：`2026-04-08`  
脚本：`scripts/run_code_deepsets_round1.py`  
正式结果载荷：`result/code_deepsets_round1_20260408_run2/code_deepsets_round1.json`

## 当前仓库状态

在这次运行之前：

1. `code_v2` 已经是 promoted coding default。
2. `science_hybrid_round3` 已经是 promoted science patch。
3. `gpqa_deepsets_round1` 已经完成第一轮最小科学域 DeepSets 实验，但结论仍是 `NO-PROMOTE`。
4. 当前问题变成：这条最小 full-group contextual DeepSets 思路，能不能迁移到 coding 和 math。

## 最小 DeepSets 设计

coding DeepSets round-1 严格保持很小：

- 领域：`DS-R1/lcb_v5`
- 输入：只用当前 `code_v2` 的结构化信号
- 特征：
  - `prefix_best_window_quality_r`
  - `head_tail_gap_r`
  - `tail_variance_r`
  - `post_reflection_recovery_r`
  - `last_block_instability_r`
- 模型：
  - 每条 run 先过一个小 MLP encoder
  - hidden size 保持很小
  - group context pooling 只试 `mean / max`
  - 最终分数来自 `run embedding + pooled group embedding`
- 唯一的小扩展：
  - `max + pairwise_aux_weight=0.25`

不用 attention，不用 Set Transformer，不用 raw neuron rows，也不做大范围超参搜索。

## Coding 单域结果

| Selector | AUROC | Hit@1 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| `code_baseline_v1` | 50.27% | 59.28% | 51.27% | 61.74% |
| `code_v2` | 50.29% | 61.68% | 51.00% | **62.11%** |
| `code_deepsets_round1_mean` | 48.79% | 58.68% | 50.12% | 58.65% |
| `code_deepsets_round1_max` | **52.47%** | **62.28%** | 50.29% | 55.75% |
| `code_deepsets_round1_max_pairaux0p25` | 51.67% | 59.28% | 49.08% | 59.96% |

本轮选中的候选：

- `code_deepsets_round1_max`

解读：

- 最小 group context 的确给了 `code_v2` 一点 top-slot 提升
  - `Hit@1`: `61.68% -> 62.28%`
  - `AUROC`: `50.29% -> 52.47%`
- 但它显著伤害了 shortlist quality
  - `SelAcc@10`: `62.11% -> 55.75%`
- `pairwise_aux` 小探针也没有把这个问题修回来

所以 coding DeepSets round-1 确实让 top-1 有一点提升，但整体 operating point 还不够好。

## 当前系统 proxy vs patched system proxy

当前 promoted stack：

- generic extreme / baseline12 frozen
- coding = promoted `code_v2`
- science = promoted `science_hybrid_round3`

patched 候选系统：

- 只把 coding 替换成 `code_deepsets_round1_max`

### Sample-weighted proxy

| System | Hit@1 | Hit@3 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| current = `code_v2 + science_hybrid_round3` | 67.22% | 74.02% | **61.29%** | **66.15%** |
| patched = `code_deepsets_round1_max + science_hybrid_round3` | **67.42%** | **74.23%** | 61.05% | 63.96% |

相对当前系统的变化：

- `Hit@1`: `+0.21pp`
- `Hit@3`: `+0.21pp`
- `Pairwise`: `-0.24pp`
- `SelAcc@10`: `-2.19pp`

### Equal-cache-mean proxy

| System | Hit@1 | Hit@3 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| current = `code_v2 + science_hybrid_round3` | 70.53% | 77.71% | **71.53%** | **70.90%** |
| patched = `code_deepsets_round1_max + science_hybrid_round3` | **70.63%** | **77.81%** | 71.42% | 69.84% |

相对当前系统的变化：

- `Hit@1`: `+0.10pp`
- `Hit@3`: `+0.10pp`
- `Pairwise`: `-0.12pp`
- `SelAcc@10`: `-1.06pp`

解读：

- 这个候选确实把系统 top-slot proxy 稍微抬高了一点
- 但 shortlist quality 掉得太多，因此当前 guarded system gate 仍然拒绝它

## 决策

结论：**NO-PROMOTE**

原因：

1. coding gate 没过：
   - `SelAcc@10` 低于当前 `code_v2`
2. system gate 没过：
   - sample-weighted `Hit@1` 虽然提升
   - 但 `SelAcc@10` 明显下滑
   - 在当前 proxy 规则下，这不算 guarded improvement

操作性结论：

- coding DeepSets round-1 说明 contextual signal 是存在的
- 但这个第一版还不能替换 `code_v2`

## 文件

本条线新增 / 修改：

- `nad/core/selectors/deepsets_core.py`
- `nad/core/selectors/code_deepsets_impl.py`
- `plugins/code_deepsets_selector.py`
- `scripts/run_code_deepsets_round1.py`
- `docs/CODE_DEEPSETS_ROUND1_PLAN_20260408.md`
- `docs/CODE_DEEPSETS_ROUND1_RESULTS_20260408.md`

生成产物：

- `models/ml_selectors/code_deepsets_round1.pkl`
- `models/ml_selectors/code_deepsets_round1_mean.pkl`
- `models/ml_selectors/code_deepsets_round1_max.pkl`
- `models/ml_selectors/code_deepsets_round1_max_pairaux0p25.pkl`
- `result/code_deepsets_round1_20260408_run2/`

## 复跑

```bash
source .venv/bin/activate
python scripts/run_code_deepsets_round1.py \
  --out-dir result/code_deepsets_round1_$(date -u +%Y%m%d_%H%M%S) \
  --distance-threads 8 \
  --torch-threads 8
```

## 下一步判断

如果继续这条线，更可能有价值的下一步不是更大的 context，而是在当前小特征家族里继续修正 top-slot 和 shortlist 之间的校准关系。
