# Extreme12 Baseline Test 分析 / Test Analysis & Bottleneck

> 日期 / Date: 2026-04-04
> 模型 / Model: baseline12_pointwise (Extreme8: 3-D features)
> 提交 / Submission: `submission/BestofN/extreme12/base/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank.json`

## Per-Cache 测试指标 / Per-Cache Test Metrics

| Cache | AUROC | Hit@1 | Hit@3 | SelAcc@10% | Pairwise | Samples |
|---|---|---|---|---|---|---|
| DS-R1/aime24 | 0.6207 | 0.9000 | 0.9000 | 0.8594 | 0.8632 | 1,920 |
| DS-R1/aime25 | 0.5612 | 0.6000 | 0.7667 | 0.6771 | 0.6714 | 1,920 |
| DS-R1/brumo25 | 0.6288 | 0.7667 | 0.8667 | 0.8177 | 0.8528 | 1,920 |
| DS-R1/gpqa | 0.5404 | 0.6061 | 0.7222 | 0.6298 | 0.5945 | 12,672 |
| DS-R1/hmmt25 | 0.6177 | 0.6667 | 0.7000 | 0.6354 | 0.8076 | 1,920 |
| DS-R1/lcb_v5 | 0.4966 | 0.5928 | 0.6766 | 0.5815 | 0.4888 | 10,688 |
| Qwen3-4B/aime24 | 0.6232 | 0.9000 | 0.9333 | 0.8854 | 0.8301 | 1,920 |
| Qwen3-4B/aime25 | 0.6090 | 0.7667 | 0.8333 | 0.7812 | 0.8153 | 1,920 |
| Qwen3-4B/brumo25 | 0.5979 | 0.9000 | 0.9667 | 0.8698 | 0.8000 | 1,920 |
| Qwen3-4B/gpqa | 0.5121 | 0.6717 | 0.7475 | 0.6772 | 0.5423 | 12,672 |
| Qwen3-4B/hmmt25 | 0.5825 | 0.6667 | 0.7000 | 0.6042 | 0.7703 | 1,920 |
| Qwen3-4B/lcb_v5 | 0.4967 | 0.6287 | 0.7126 | 0.6077 | 0.4893 | 10,688 |

**汇总 / Summary:**
- 样本加权平均 SelAcc@10%: ~0.664 (gpqa+lcb_v5 占 62,080 中的 46,720 = 75%)
- 等权平均 SelAcc@10%: ~0.719

---

## SelAcc@10% 落后分析 / SelAcc@10% Weakness Diagnosis

### 核心问题 / Core Issues

#### 1. 大 Dataset 上排序能力近乎随机 / Near-Random Ranking on Large Datasets

**lcb_v5 (Live CodeBench):**
- 两个模型 AUROC ≈ 0.497 → **完全随机**
- SelAcc@10% ≈ 0.58-0.61
- Pairwise ≈ 0.49 (两个都低于 0.5)
- 样本数: 10,688 (17% of total)

**gpqa (Google-Proof QA):**
- AUROC ≈ 0.51-0.54 → **几乎随机**
- SelAcc@10% ≈ 0.63-0.68
- Pairwise ≈ 0.54-0.59 (刚好好于随机)
- 样本数: 12,672 (20% of total)

**合计影响:** lcb_v5 + gpqa = **46,720 samples (75% of total)**，而我们的模型在这两个上是随机的

#### 2. 特征对 Coding/Science 问题失效 / Features Fail on Coding & Science Problems

**问题根源 / Root Cause:**
- Extreme8 使用 3 个特征：`dc_z` (DeepConf quality z-score), `dc_r` (DeepConf rank), `reflection_count_r` (reflection count rank)
- **DeepConf quality** = `-token_confidence` 聚合 (token-level confidence)
- **Reflection count** = trajectory 中 reflection 事件的数量

**为什么对 coding/science 无效:**
- **Coding (lcb_v5):** 代码生成的 token confidence 与代码正确性关联度极低
  - 语言模型可以高置信地生成错误的代码
  - Token entropy 分布与数学推理完全不同（完成函数体有高置信但高错误率）
- **Science (GPQA):** 虽然是 Q&A，但：
  - GPQA 题目是多轮推理的科学问题
  - Token confidence 聚合对多步推理中的"关键错误"无感知
  - Reflection count 对科学推理的有效性无直接关联

#### 3. SelAcc@10% 对排序头部极度敏感 / SelAcc@10% is Hypersensitive to Top Rankings

**定义:** SelAcc@10% = 排名前 10% 的 sample 的正确率

**问题:**
- 当 AUROC ≈ 0.5 (随机) 时，top-10% 的正确率 ≈ 全局准确率
- 无法富集正确答案到排名头部 → 指标退化到全局准确率 level

**与其他指标的对比:**
- Hit@1 / Hit@3 / Pairwise: 对排序整体的相关性考虑
- SelAcc@10%: **只看最优的 10% 排名** → 排序错误直接导致指标崩溃

#### 4. 小 Dataset 表现好但权重低 / Good on Small Datasets but Negligible Weight

**aime24, brumo25:** SelAcc@10% > 0.85
- 这些数学竞赛问题上，DeepConf + reflection 特征确实有效
- 每个 dataset 只有 1,920 samples (3% of total each)

**问题:** 无法抵消 gcpqa + lcb_v5 的拖累

---

## 改进方向 / Improvement Directions

### 短期 (1-2 周) / Short-term

1. **引入 local_conf 特征**
   - Extreme8 未使用 8 个 local conf features (tail_2k, lgc_512, bottom_q10, head_tail_gap, etc.)
   - 这些对 token-level 质量有更细粒度的刻画
   - 预期对 coding problem 有帮助

2. **Per-dataset 特征权重**
   - 允许 math/coding 问题使用不同的特征权重或模型
   - 不改变模型，只改变融合权重

3. **提高 Extreme8 到 Extreme9/10**
   - Extreme9: 加入 8 个 local_conf features (11-D total)
   - Extreme10: 加入 graph topology (degree, clustering) + error-mass features (17-D total)
   - 这些特征可能对 coding problem 有更强信号

### 中期 (2-4 周) / Medium-term

4. **Coding-specific 特征工程**
   - Code structure complexity (AST depth, branching)
   - Test-case confidence (对 generated code 的多个 test case 的通过率)
   - Variable/function scope consistency checks

5. **Ensemble 多套模型**
   - 分别训练 math-specialized 和 coding-specialized selectors
   - 加权融合或 problem-type adaptive selection

6. **增加 tuple 复杂度**
   - tuple_size = 12 → 16-20
   - num_tuples = 1024 → 2048
   - 提高统计稳定性和特征可区分性

### 长期 (1+ 月) / Long-term

7. **Semantic-aware 特征**
   - 利用 token 的 semantic content，不只是统计量
   - 例如：代码中 critical tokens (branching, return, variable assignment) 的 confidence
   - 例如：自然语言中 named entities / key concepts 的信心度

8. **跨模型迁移**
   - DS-R1-0528 vs Qwen3-4B 的 activation pattern 差异很大
   - 需要模型自适应的选择器（或 model-specific calibration）

9. **动态 reflection threshold**
   - 当前固定 threshold=0.30
   - 对不同 dataset/model 可能需要不同 threshold
   - 可学习的 per-problem threshold

---

## 验证要点 / Validation Checklist

- [x] Per-cache AUROC, Hit@1/3, SelAcc@10%, Pairwise 已计算
- [x] 75% 样本集中在 lcb_v5+gpqa 且 AUROC≈0.5 确认
- [x] 小 dataset (aime24/brumo25) SelAcc@10% > 0.85 确认
- [ ] Extreme9/10 on cache_test 需要运行
- [ ] Coding-specific 特征工程需要设计与验证
