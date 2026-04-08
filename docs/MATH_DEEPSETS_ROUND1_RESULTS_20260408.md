# Math DeepSets Round 1 — Results

Date: `2026-04-08`  
Script: `scripts/run_math_deepsets_round1.py`  
Official result payload: `result/math_deepsets_round1_main_20260408_run1/math_deepsets_round1.json`

## Current Repo State

Before this run:

1. `code_v2` is already the promoted coding default.
2. `science_hybrid_round3` is already the promoted science patch.
3. `gpqa_deepsets_round1` completed the first minimal science-contextual DeepSets study but remained `NO-PROMOTE`.
4. `code_deepsets_round1` has now also been tested and remained `NO-PROMOTE`.
5. The remaining open question is whether the same minimal full-group contextual idea is more useful on the math slice.

## Minimal DeepSets Design

Math DeepSets round-1 stays deliberately small:

- domain: `math`
- profile: `main`
- datasets:
  - `aime24`
  - `aime25`
  - `brumo25`
  - `hmmt25`
- inputs: only the current structured `all_aug` math features
- feature stack:
  - base 12 `ml_features`
  - augmented 8 consensus / disagreement features
- model:
  - per-run encoder MLP
  - tiny hidden sizes
  - pooled group context with `mean` / `max`
  - final score from `run embedding + pooled group embedding`
- tiny extra probe:
  - `max + pairwise_aux_weight=0.25`

No attention, no Set Transformer, no raw neuron rows, and no broad hyperparameter sweep.

## Math Single-Domain Results

| Selector | AUROC | Hit@1 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| `medoid` | 56.04% | 70.83% | 68.52% | 72.27% |
| `knn-medoid` | 57.21% | 73.33% | 72.11% | 71.88% |
| `runwise__all_aug__squared_hinge__C0p10__bias__balanced` | 85.46% | 74.17% | 73.27% | 96.88% |
| `ranksvm__no_logs__squared_hinge__C0p10__bias__mean_margin` | 58.90% | 74.17% | 74.49% | 75.26% |
| `math_deepsets_round1_mean` | 86.01% | 71.67% | 74.05% | 98.96% |
| `math_deepsets_round1_max` | 86.32% | 75.83% | 72.97% | **99.09%** |
| `math_deepsets_round1_max_pairaux0p25` | 86.26% | **77.50%** | 74.42% | 98.70% |

Selected candidate:

- `math_deepsets_round1_max_pairaux0p25`

Read:

- minimal group context works much better on math than it did on GPQA or coding
- the best candidate clears the current top-slot read:
  - `Hit@1`: `73.33% -> 77.50%` vs `knn-medoid`
- it also preserves / improves shortlist quality at a very high level:
  - `Pairwise`: `72.11% -> 74.42%`
  - `SelAcc@10`: `71.88% -> 98.70%`

Operationally, this is the first DeepSets extension in this line that looks truly deployable.

## Current System Proxy vs Patched System Proxy

Current promoted stack:

- generic extreme / baseline12 frozen
- coding = promoted `code_v2`
- science = promoted `science_hybrid_round3`
- math = generic math slice from the current stack

Patched candidate stack:

- replace math only with `math_deepsets_round1_max_pairaux0p25`

### Sample-weighted proxy

| System | Hit@1 | Hit@3 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| current = `generic math + code_v2 + science_hybrid_round3` | 67.22% | **74.02%** | **61.29%** | 66.15% |
| patched = `math_deepsets_round1_max_pairaux0p25 + code_v2 + science_hybrid_round3` | **68.25%** | 73.61% | 60.10% | **72.37%** |

Delta vs current:

- `Hit@1`: `+1.03pp`
- `Hit@3`: `-0.41pp`
- `Pairwise`: `-1.19pp`
- `SelAcc@10`: `+6.22pp`

### Equal-cache-mean proxy

| System | Hit@1 | Hit@3 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| current = `generic math + code_v2 + science_hybrid_round3` | 70.53% | **77.71%** | **71.53%** | 70.90% |
| patched = `math_deepsets_round1_max_pairaux0p25 + code_v2 + science_hybrid_round3` | **73.31%** | 76.60% | 68.33% | **87.66%** |

Delta vs current:

- `Hit@1`: `+2.78pp`
- `Hit@3`: `-1.11pp`
- `Pairwise`: `-3.20pp`
- `SelAcc@10`: `+16.76pp`

Interpretation:

- this patch improves the promote-sensitive top-slot read
- it also produces a very large shortlist-quality gain
- although `Hit@3` / `Pairwise` soften somewhat, the current guarded system read still accepts the patch

## Decision

Decision: **Promote**

Reason:

1. Math gate passes:
   - `Hit@1` is above `knn-medoid`
   - the candidate does not lose both `SelAcc@10` and `Pairwise`
2. System gate passes:
   - sample-weighted `Hit@1` improves
   - sample-weighted `SelAcc@10` improves materially
3. This is the first minimal DeepSets extension that improves the patched full-system proxy strongly enough to justify export

Operational read:

- keep `code_v2`
- keep `science_hybrid_round3`
- promote `math_deepsets_round1_max_pairaux0p25` as the new math-side patch for BestofN

## BestofN Submission Export

Export helper:

- `scripts/patch_bestofn_submission_with_math_deepsets_round1.py`

Generated patched BestofN submission:

- `submission/BestofN/extreme12/patches/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa__math_deepsets_round1_patch.json`

This export keeps the current promoted stack and only replaces the math slice with the trained `math_deepsets_round1` winner.

## Files

Added / changed for this line:

- `nad/core/selectors/deepsets_core.py`
- `nad/core/selectors/math_deepsets_impl.py`
- `plugins/math_deepsets_selector.py`
- `scripts/run_math_deepsets_round1.py`
- `scripts/patch_bestofn_submission_with_math_deepsets_round1.py`
- `docs/MATH_DEEPSETS_ROUND1_PLAN_20260408.md`
- `docs/MATH_DEEPSETS_ROUND1_RESULTS_20260408.md`

Generated artifacts:

- `models/ml_selectors/math_deepsets_round1.pkl`
- `models/ml_selectors/math_deepsets_round1_mean.pkl`
- `models/ml_selectors/math_deepsets_round1_max.pkl`
- `models/ml_selectors/math_deepsets_round1_max_pairaux0p25.pkl`
- `result/math_deepsets_round1_main_20260408_run1/`

## Re-run

```bash
source .venv/bin/activate
python scripts/run_math_deepsets_round1.py \
  --profile main \
  --out-dir result/math_deepsets_round1_main_$(date -u +%Y%m%d_%H%M%S) \
  --distance-threads 8 \
  --torch-threads 8
```

Export the promoted BestofN patch:

```bash
source .venv/bin/activate
python scripts/patch_bestofn_submission_with_math_deepsets_round1.py \
  --distance-threads 8
```

## Next Read

This line is good enough to export, so the next step is not an automatic jump to a bigger architecture.

If continued later, the cleaner next move is:

- validate the exported BestofN patch externally
- check whether the math contextual gain survives on real leaderboard feedback
- only then consider whether a larger contextual family is worth the complexity

---

# Math DeepSets Round 1 — 结果（中文）

日期：`2026-04-08`  
脚本：`scripts/run_math_deepsets_round1.py`  
正式结果载荷：`result/math_deepsets_round1_main_20260408_run1/math_deepsets_round1.json`

## 当前仓库状态

在这次运行之前：

1. `code_v2` 已经是 promoted coding default。
2. `science_hybrid_round3` 已经是 promoted science patch。
3. `gpqa_deepsets_round1` 已完成第一轮最小 science-contextual DeepSets 试验，但结论仍是 `NO-PROMOTE`。
4. `code_deepsets_round1` 也已经测试过，结论同样是 `NO-PROMOTE`。
5. 剩下真正开放的问题，就是这条最小 full-group contextual 思路能否在 math slice 上更有效。

## 最小 DeepSets 设计

math DeepSets round-1 严格保持很小：

- 领域：`math`
- profile：`main`
- 数据集：
  - `aime24`
  - `aime25`
  - `brumo25`
  - `hmmt25`
- 输入：只用当前结构化 `all_aug` math 特征
- 特征栈：
  - 12 维基础 `ml_features`
  - 8 维 consensus / disagreement 派生特征
- 模型：
  - 每条 run 先过一个小 MLP encoder
  - hidden size 保持很小
  - group context pooling 只试 `mean / max`
  - 最终分数来自 `run embedding + pooled group embedding`
- 唯一的小扩展：
  - `max + pairwise_aux_weight=0.25`

不用 attention，不用 Set Transformer，不用 raw neuron rows，也不做大范围超参搜索。

## Math 单域结果

| Selector | AUROC | Hit@1 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| `medoid` | 56.04% | 70.83% | 68.52% | 72.27% |
| `knn-medoid` | 57.21% | 73.33% | 72.11% | 71.88% |
| `runwise__all_aug__squared_hinge__C0p10__bias__balanced` | 85.46% | 74.17% | 73.27% | 96.88% |
| `ranksvm__no_logs__squared_hinge__C0p10__bias__mean_margin` | 58.90% | 74.17% | 74.49% | 75.26% |
| `math_deepsets_round1_mean` | 86.01% | 71.67% | 74.05% | 98.96% |
| `math_deepsets_round1_max` | 86.32% | 75.83% | 72.97% | **99.09%** |
| `math_deepsets_round1_max_pairaux0p25` | 86.26% | **77.50%** | 74.42% | 98.70% |

本轮选中的候选：

- `math_deepsets_round1_max_pairaux0p25`

解读：

- 最小 group context 在 math 上明显比在 GPQA / coding 上更有效
- 最优候选已经越过当前 top-slot 门槛：
  - 相对 `knn-medoid`，`Hit@1`: `73.33% -> 77.50%`
- 同时它还保持并提升了 shortlist quality：
  - `Pairwise`: `72.11% -> 74.42%`
  - `SelAcc@10`: `71.88% -> 98.70%`

从操作判断看，这是这条 DeepSets 扩展线里第一个真正像是可以部署的版本。

## 当前系统 proxy vs patched system proxy

当前 promoted stack：

- generic extreme / baseline12 frozen
- coding = promoted `code_v2`
- science = promoted `science_hybrid_round3`
- math = 当前系统里的 generic math slice

patched 候选系统：

- 只把 math 替换成 `math_deepsets_round1_max_pairaux0p25`

### Sample-weighted proxy

| System | Hit@1 | Hit@3 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| current = `generic math + code_v2 + science_hybrid_round3` | 67.22% | **74.02%** | **61.29%** | 66.15% |
| patched = `math_deepsets_round1_max_pairaux0p25 + code_v2 + science_hybrid_round3` | **68.25%** | 73.61% | 60.10% | **72.37%** |

相对当前系统的变化：

- `Hit@1`: `+1.03pp`
- `Hit@3`: `-0.41pp`
- `Pairwise`: `-1.19pp`
- `SelAcc@10`: `+6.22pp`

### Equal-cache-mean proxy

| System | Hit@1 | Hit@3 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| current = `generic math + code_v2 + science_hybrid_round3` | 70.53% | **77.71%** | **71.53%** | 70.90% |
| patched = `math_deepsets_round1_max_pairaux0p25 + code_v2 + science_hybrid_round3` | **73.31%** | 76.60% | 68.33% | **87.66%** |

相对当前系统的变化：

- `Hit@1`: `+2.78pp`
- `Hit@3`: `-1.11pp`
- `Pairwise`: `-3.20pp`
- `SelAcc@10`: `+16.76pp`

解读：

- 这个 patch 明确提升了 promote-sensitive top-slot 读数
- 它同时还带来了非常大的 shortlist-quality 提升
- 虽然 `Hit@3` / `Pairwise` 有一些回落，但按当前 guarded system read，依然应该接受这个 patch

## 结论

结论：**Promote**

原因：

1. Math gate 通过：
   - `Hit@1` 高于 `knn-medoid`
   - 也没有同时输掉 `SelAcc@10` 和 `Pairwise`
2. System gate 通过：
   - sample-weighted `Hit@1` 提升
   - sample-weighted `SelAcc@10` 也有实质提升
3. 这是第一条最小 DeepSets 扩展线中，full-system proxy 提升强到值得导出的版本

操作结论：

- 保留 `code_v2`
- 保留 `science_hybrid_round3`
- 将 `math_deepsets_round1_max_pairaux0p25` promote 为新的 BestofN math-side patch

## BestofN 提交导出

导出脚本：

- `scripts/patch_bestofn_submission_with_math_deepsets_round1.py`

生成的 BestofN patched submission：

- `submission/BestofN/extreme12/patches/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa__math_deepsets_round1_patch.json`

这个导出保留当前 promoted stack，只把 math slice 替换成训练好的 `math_deepsets_round1` winner。

## 文件

本线新增 / 修改：

- `nad/core/selectors/deepsets_core.py`
- `nad/core/selectors/math_deepsets_impl.py`
- `plugins/math_deepsets_selector.py`
- `scripts/run_math_deepsets_round1.py`
- `scripts/patch_bestofn_submission_with_math_deepsets_round1.py`
- `docs/MATH_DEEPSETS_ROUND1_PLAN_20260408.md`
- `docs/MATH_DEEPSETS_ROUND1_RESULTS_20260408.md`

生成产物：

- `models/ml_selectors/math_deepsets_round1.pkl`
- `models/ml_selectors/math_deepsets_round1_mean.pkl`
- `models/ml_selectors/math_deepsets_round1_max.pkl`
- `models/ml_selectors/math_deepsets_round1_max_pairaux0p25.pkl`
- `result/math_deepsets_round1_main_20260408_run1/`

## 复跑

```bash
source .venv/bin/activate
python scripts/run_math_deepsets_round1.py \
  --profile main \
  --out-dir result/math_deepsets_round1_main_$(date -u +%Y%m%d_%H%M%S) \
  --distance-threads 8 \
  --torch-threads 8
```

导出 promoted BestofN patch：

```bash
source .venv/bin/activate
python scripts/patch_bestofn_submission_with_math_deepsets_round1.py \
  --distance-threads 8
```

## 下一步判断

这条线已经足够好，可以先导出，所以接下来不应该自动跳到更大的架构。

如果后续继续推进，更干净的下一步应该是：

- 先做外部 BestofN 验证
- 再看 math contextual gain 在真实 leaderboard 上是否成立
- 只有这一步确认后，再决定要不要付出更高复杂度去尝试更大的 contextual family
