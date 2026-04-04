# Extreme12 V2 实验：Aggregated SelAcc@10% vs Baseline / Aggregated Objective Comparison

> 日期 / Date: 2026-04-03
> 实验目录 / Directory: `results/extreme12_v2_experiments/20260403_105944/`
> 提交 / Submission: `submission/BestofN/extreme12_baseline12_pointwise_*_scale100_rank.json`

## 实验动机 / Motivation

**问题:** Baseline12_pointwise 在训练集上 SelAcc@10% = 75.99%，但这个指标是逐点目标（label 为 0/1）优化的结果。能否设计聚合目标来直接优化 SelAcc@10%？

**假设:** 通过 aggregated SelAcc@10% 目标（在 inner tune set 上搜索最优超参），使得验证集上的 SelAcc@10% 更高。

---

## 实验设置 / Configuration

| 参数 / Parameter | 值 / Value |
|---|---|
| **数据集 / Datasets** | aime24, aime25, brumo25, gpqa, hmmt25, livecodebench_v5 (all 6) |
| **缓存 / Cache** | MUI_HUB/cache (train), cache_train (val) |
| **样本统计 / Samples** | 276 train problems, 58 val problems |
| **Tuple 配置 / Tuple Config** | |
| - tuple_size | 12 |
| - tuple_rule | 2-10 correct (mixed) |
| - num_tuples (train) | 256 |
| - num_tuples (val) | 1024 |
| - num_tuples (inner tune) | 256 |
| **特征 / Features** | 3-D Extreme8 (dc_z, dc_r, reflection_count_r) |
| **目标函数 / Objectives** | |
| - Baseline | pointwise (label-wise logistic regression) |
| - Aggregated | aggregated_selacc10 (inner tune SelAcc@10% 搜索) |
| **聚合搜索参数 / Aggregated Search** | |
| - 候选种子 / Seeds | 42, 43, 44 |
| - 内部调优分割 / Inner tune split | 20% |
| - 内部调优种子 / Inner tune seed | seed + offset |
| **模型超参 / Model Hyperparams** | |
| - alpha | 1.0 (band 1 weight) |
| - beta | 0.35 (band 2 weight) |
| - gamma | -0.50 (band 3 weight) |
| **Guardrail 约束 / Guardrails** | |
| - hit@1 drop | ≤ 1.0% |
| - pairwise drop | ≤ 0.5% |

---

## 实验结果 / Results

| 配置 / Config | Hit@1 | Hit@3 | Pairwise | SelAcc@10 | Inner Tune SelAcc | 通过 Guardrail | 推荐 |
|---|---|---|---|---|---|---|---|
| **baseline12_pointwise** | **67.65%** | 97.06% | **74.59%** | **75.99%** | - | n/a | **✓ yes** |
| aggregated_s042 | 69.12% | 97.06% | 73.68% | 76.91% | 83.66% | ✗ (pairwise 74.09% < 74.59% - 0.5%) | no |
| aggregated_s043 | 66.18% | 97.06% | 73.93% | 75.99% | 81.54% | ✗ (hit1 66.65% < 67.65% - 1%) | no |
| aggregated_s044 | 69.12% | 98.53% | 74.44% | 72.65% | 87.00% | ✓ pass | no |

---

## 详细分析 / Detailed Analysis

### Baseline 指标 / Baseline Metrics

- **Hit@1 = 67.65%** — 排名第一的 sample 正确的概率
- **Hit@3 = 97.06%** — top-3 中至少有一个正确的概率
- **Pairwise = 74.59%** — 随机抽两个 (correct, incorrect)，我们排序正确的概率
- **SelAcc@10% = 75.99%** — top-10% 的正确率（本次优化的目标）

### Aggregated Seed 42 (s042)

**指标:**
- Hit@1 = 69.12% (up 1.47pp)
- Pairwise = 73.68% (down 0.91pp) → **失败 guardrail**
- SelAcc@10% = 76.91% (up 0.92pp)
- Inner tune SelAcc = 83.66% (极高 → 过拟合 inner tune set)

**诊断:**
- 在 inner tune set 上搜索到了很好的超参配置
- 但这个配置在 validation set 上的 pairwise 下降超过阈值
- Pairwise 和 SelAcc@10% 的优化方向冲突（一个增大了对 top-ranked pairs，另一个只关心 top-10%）

### Aggregated Seed 43 (s043)

**指标:**
- Hit@1 = 66.18% (down 1.47pp) → **失败 guardrail**
- Pairwise = 73.93% (down 0.66pp)
- SelAcc@10% = 75.99% (unchanged, 没有改进)
- Inner tune SelAcc = 81.54%

**诊断:**
- 种子 43 的优化不稳定
- Hit@1 下降超过 guardrail
- SelAcc@10% 无改进（等于 baseline）

### Aggregated Seed 44 (s044)

**指标:**
- Hit@1 = 69.12% (up 1.47pp)
- Hit@3 = 98.53% (up 1.47pp)
- Pairwise = 74.44% (down 0.15pp) → **通过 guardrail** ✓
- SelAcc@10% = 72.65% (down 3.34pp) → **更差**
- Inner tune SelAcc = 87.00% (最高的过拟合)

**诊断:**
- 虽然通过了 guardrail，但 SelAcc@10% 反而下降了 3.34pp
- 说明 inner tune set 上的 SelAcc@10% 和 validation set 上的有很大 gap
- Inner tune SelAcc (87%) vs validation SelAcc (72.65%) 的差距说明优化目标完全过拟合了

---

## 关键发现 / Key Findings

### 1. Aggregated 目标在 Guardrail 约束下无法超越 Baseline

三个种子都无法同时满足两个条件：
- 通过 guardrail（Hit@1 drop ≤1%, Pairwise drop ≤0.5%）
- 在 validation 上超越 baseline SelAcc@10% (75.99%)

### 2. Inner Tune 与 Validation 的严重 Mismatch

| 种子 | Inner Tune SelAcc | Validation SelAcc | Gap |
|---|---|---|---|
| s042 | 83.66% | 76.91% | **-6.75pp** |
| s043 | 81.54% | 75.99% | **-5.55pp** |
| s044 | 87.00% | 72.65% | **-14.35pp** ← 极端过拟合 |

**原因:**
- Inner tune set 只有 ~55 problems (20% of train)，统计量太小
- 通过超参搜索优化 inner tune SelAcc 时，容易过拟合到特定样本的特性
- Validation set (~276 problems) 更能代表真实分布，但搜索过程没有 early stopping

### 3. SelAcc@10% 和 Pairwise/Hit@1 的优化冲突

- **s042** 提高了 SelAcc@10% (+0.92pp)，但损害了 pairwise (-0.91pp)
- **s044** 提高了 Hit@1/Hit@3，但 SelAcc@10% 大幅下降 (-3.34pp)

说明：
- 不同指标优化方向不完全一致
- Aggregated 目标（SelAcc@10%）与 guardrail 目标（Hit@1, Pairwise）存在根本冲突
- 简单的内部搜索无法在多个目标间找到 pareto optimal

---

## 结论 / Conclusions

✓ **Recommended: baseline12_pointwise**

**理由 / Rationale:**
1. Aggregated 聚合目标无法同时通过 guardrail 和超越 baseline
2. Inner tune 过拟合导致 validation 指标恶化
3. Baseline 虽然简单，但在全量 validation set 上稳定且平衡

**后续改进方向 / Future Work:**
1. **增大 inner tune set** — 减少过拟合（但需要更多训练数据）
2. **多目标优化** — 不是单纯优化 SelAcc@10%，而是 scalarize 多个指标
3. **提升 Extreme8 特征质量** — 使 base selector 本身更强，减少对超参搜索的依赖
4. **约束搜索空间** — 在搜索过程中实时监测 guardrail，并 early-stop
