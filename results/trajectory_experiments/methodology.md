# Trajectory Analysis Methodology | 轨迹分析方法论

[English](#english) | [中文](#中文)

---

## 中文

### 概述

轨迹分析选择器（实验 7-9）利用神经元激活模式的**时序演变**来预测推理正确性。核心思想：重要的不是**哪些**神经元被激活，而是激活模式**如何随时间变化**。

数据来源：
- `rows/` bank（v4.1+）：每 32 个 token 一个"切片"，记录该切片内所有激活神经元的 key 集合
- neuron key 编码：`uint32 = layer << 16 | neuron_id`，可解码出层信息

最新产物快照（UTC）：
- `2026-03-30`：评估报告位于 `results/trajectory_experiments/accuracy_summary_20260330_112435.json`、`results/trajectory_experiments/trajectory_20260330_112435.json`、`results/trajectory_experiments/layer_stratified_20260330_112435.json`
- `2026-03-31 01:56:52`：22-D 轨迹融合训练统计位于 `models/ml_selectors/trajectory_stats.json`，包含 `31,040` 个带标注样本对、`18,873` 个正确样本、`22` 个特征、`6` 个数据集

### 反思（Reflection）的定义与分类

**反思**是轨迹分析中最重要的概念。直觉上，正确推理在展开论证后会"回头看"，重新审视早期的激活模式。

**数学定义：**

对于第 t 个切片，**反思分数** R_t 定义为与所有**非相邻**先前切片的最大 Jaccard 相似度：

```
R_t = max_{s < t-1} Jaccard(slice_t, slice_s)
```

其中 Jaccard(A, B) = |A ∩ B| / |A ∪ B|，A 和 B 是两个切片的已排序 uint32 neuron key 集合。

**关键细节：**
- 排除了相邻切片（slice_{t-1}），因为相邻切片的高相似度只反映"连续性"，不是"反思"
- 阈值 0.3：当 R_t > 0.3 时，认为该切片发生了一次"反思事件"
- `reflection_count`：一个 run 中反思事件的总次数
- `reflection_count_r`：组内 reflection_count 的 rank 归一化到 [0, 1]

**为什么 reflection_count_r 是最强单特征（71.1%）？**
- rank 归一化消除了不同序列长度的影响（长序列天然有更多反思事件）
- 适度的反思（不太少也不太多）与正确推理高度相关

### 轨迹特征详解（5 维）

| 特征 | 计算方法 | 含义 |
|------|---------|------|
| `mean_continuity` | 相邻切片 Jaccard 均值 C_t = J(slice_t, slice_{t-1}) | 骨干连贯性：高 = 推理稳定前进 |
| `mean_novelty` | 每切片新颖度均值 N_t = 1 - max_{s<t} J(slice_t, slice_s) | 探索程度：高 = 不断探索新模式 |
| `max_reflection` | 非相邻切片最大 Jaccard max_{s<t-1} J(slice_t, slice_s) | 最强回溯：高 = 曾经深度回顾 |
| `reflection_count_r` | R_t > 0.3 的切片数，rank 归一化 | 反思频率（组内相对排名）|
| `late_convergence` | 末尾 25% 的连续性是否 > 前 75% | 末尾收敛：正确推理往往在末尾趋于稳定 |

### 层特征详解（5 维）

层信息通过解码 neuron key 获得：`layer_id = key >> 16`。

| 特征 | 计算方法 | 含义 |
|------|---------|------|
| `deep_shallow_ratio_z` | (top 25% 层激活数) / (bottom 25% 层激活数), z-score | 深浅比：正确推理倾向于更多深层激活 |
| `layer_entropy` | 层激活计数的 Shannon 熵（归一化） | 层分布均匀度：高 = 激活分散在多层 |
| `layer_gini` | 层激活计数的 Gini 系数 | 层集中度：高 = 激活集中在少数层 |
| `deep_frac_z` | 深层（top 25%）激活占总激活比例, z-score | 深层激活比：高 = 更多深层参与 |
| `n_active_layers_z` | 激活层数, z-score | 激活广度：高 = 更多层参与计算 |

**层定义：** "深层"和"浅层"按层 ID 的四分位划分。层 ID 越大 = transformer 越靠后 = 越"深"。

### 机器学习方法

#### 1. 基础 ML 选择器（12 维特征）

从距离矩阵和 token 统计中提取 12 维特征向量，每个特征在组内归一化（z-score 或 rank）：

| 特征类型 | 维度 | 来源 |
|---------|------|------|
| 距离特征 | mean_dist_{z,r}, knn3_{z,r} | N×N Jaccard 距离矩阵 |
| 长度特征 | length_{z,r}, log_length | 激活 key 集合大小 |
| 置信度特征 | dc_{z,r} | DeepConf token 置信度质量分 |
| 投票特征 | copeland_{z,r} | Copeland 两两比较胜场数 |
| 上下文特征 | log_n | 组大小 |

**模型：**
- **logistic**：LogisticRegression(C=1.0, balanced) → P(correct) → argmax
- **linear-probe**：Ridge(alpha=1.0) → score → argmax
- **isotonic-medoid/deepconf**：单特征等渗回归 → 单调映射 → P(correct)

#### 2. 轨迹融合选择器（22 维特征）

拼接 12-D 基础特征 + 10-D 轨迹特征 = 22-D：
- LogisticRegression(C=1.0, balanced, StandardScaler)
- 留一数据集交叉验证（6 折 LOO CV）

#### 3. 单特征消融

每个特征单独训练一个 LogisticRegression → LOO CV → 评估该特征的独立预测能力。

### 评估方法：留一数据集交叉验证（LOO CV）

- 6 个数据集：aime24, aime25, brumo25, gpqa, hmmt25, livecodebench_v5
- 每次留出 1 个数据集作为测试集，用其余 5 个训练
- 测试时按"题目组"评估：每个题目组内选 P(correct) 最高的 run，检查是否正确
- 报告每个数据集的准确率和 6 个数据集均值

### 归一化方法

| 后缀 | 方法 | 说明 |
|------|------|------|
| `_z` | z-score | (x - μ) / σ，组内标准化 |
| `_r` | rank | 组内排名映射到 [0, 1] |
| 无后缀 | 原始值 | 本身就在 [0, 1] 范围（如 Jaccard 相似度、熵、Gini） |

### 启示

1. **反思是正确推理的标志**：`reflection_count_r`（71.1%）超越了所有传统特征
2. **层分布比轨迹结构更重要**：层特征（entropy, gini, deep_frac）全部在 69-70%，而原始轨迹特征（continuity, novelty）只有 57-62%
3. **更多特征 ≠ 更好**：22-D 融合（68.3%）不如 12-D 基础（69.7%），6 个数据集的 LOO CV 不足以支撑高维模型
4. **无 ML 的 layer-stratified（69.4%）几乎等于有 ML 的 logistic（69.7%）**：层激活分布是一个鲁棒、可泛化的信号

---

## English

### Overview

Trajectory analysis selectors (Exp 7-9) predict reasoning correctness by analysing the **temporal evolution** of neuron activation patterns. The key insight: what matters is not **which** neurons fire, but **how** activation patterns change over time.

Data sources:
- `rows/` bank (v4.1+): each 32-token "slice" records the set of activated neuron keys
- Neuron key encoding: `uint32 = layer << 16 | neuron_id`, allowing layer decomposition

Latest artifact snapshot (UTC):
- `2026-03-30`: evaluation reports live in `results/trajectory_experiments/accuracy_summary_20260330_112435.json`, `results/trajectory_experiments/trajectory_20260330_112435.json`, and `results/trajectory_experiments/layer_stratified_20260330_112435.json`
- `2026-03-31 01:56:52`: 22-D trajectory-fusion training stats were refreshed in `models/ml_selectors/trajectory_stats.json`, covering `31,040` labelled pairs, `18,873` correct samples, `22` features, and `6` datasets

### Reflection: Definition and Classification

**Reflection** is the most important concept in trajectory analysis. Intuitively, correct reasoning "looks back" at earlier activation patterns after developing an argument.

**Mathematical definition:**

For slice t, the **reflection score** R_t is the maximum Jaccard similarity to any **non-adjacent** prior slice:

```
R_t = max_{s < t-1} Jaccard(slice_t, slice_s)
```

where Jaccard(A, B) = |A ∩ B| / |A ∪ B|, with A and B being sorted uint32 neuron key sets.

**Key details:**
- Adjacent slice (slice_{t-1}) is excluded — high similarity to the immediately preceding slice reflects "continuity", not "reflection"
- Threshold 0.3: when R_t > 0.3, the slice is counted as a "reflection event"
- `reflection_count`: total number of reflection events in a run
- `reflection_count_r`: rank-normalised reflection_count within the problem group, mapped to [0, 1]

**Why is reflection_count_r the best single feature (71.1%)?**
- Rank normalisation removes the effect of sequence length (longer sequences naturally have more reflection events)
- Moderate reflection (neither too little nor too much) strongly correlates with correct reasoning

### Trajectory Features (5 dimensions)

| Feature | Computation | Meaning |
|---------|-------------|---------|
| `mean_continuity` | Mean Jaccard of consecutive slices: C_t = J(slice_t, slice_{t-1}) | Backbone coherence: high = reasoning progresses steadily |
| `mean_novelty` | Mean per-slice novelty: N_t = 1 - max_{s<t} J(slice_t, slice_s) | Exploration degree: high = continuously exploring new patterns |
| `max_reflection` | Max non-adjacent Jaccard: max_{s<t-1} J(slice_t, slice_s) | Deepest look-back: high = strong retrospection occurred |
| `reflection_count_r` | Count of slices where R_t > 0.3, rank-normalised | Reflection frequency (relative rank within group) |
| `late_convergence` | Whether continuity in final 25% exceeds first 75% | Late convergence: correct reasoning often stabilises at the end |

### Layer Features (5 dimensions)

Layer information is decoded from neuron keys: `layer_id = key >> 16`.

| Feature | Computation | Meaning |
|---------|-------------|---------|
| `deep_shallow_ratio_z` | (top 25% layer activations) / (bottom 25%), z-score | Deep-shallow ratio: correct reasoning tends to activate deeper layers more |
| `layer_entropy` | Shannon entropy of layer-wise counts (normalised) | Layer uniformity: high = activations spread across many layers |
| `layer_gini` | Gini coefficient of layer-wise counts | Layer concentration: high = activations concentrated in few layers |
| `deep_frac_z` | Fraction of activations in deepest 25% layers, z-score | Deep layer fraction: high = more deep-layer participation |
| `n_active_layers_z` | Number of distinct active layers, z-score | Activation breadth: high = more layers participate |

**Layer definition:** "deep" and "shallow" are defined by quartiles of layer IDs. Higher layer ID = later in the transformer = "deeper".

### Machine Learning Approaches

#### 1. Base ML Selectors (12-D features)

12 features extracted from the distance matrix and token statistics, each group-normalised:

| Feature type | Dims | Source |
|-------------|------|--------|
| Distance | mean_dist_{z,r}, knn3_{z,r} | N×N Jaccard distance matrix |
| Length | length_{z,r}, log_length | Activation key set size |
| Confidence | dc_{z,r} | DeepConf token confidence quality score |
| Voting | copeland_{z,r} | Copeland pairwise comparison win count |
| Context | log_n | Group size |

**Models:**
- **logistic**: LogisticRegression(C=1.0, balanced) → P(correct) → argmax
- **linear-probe**: Ridge(alpha=1.0) → score → argmax
- **isotonic-medoid/deepconf**: single-feature isotonic regression → monotonic mapping → P(correct)

#### 2. Trajectory Fusion Selector (22-D features)

Concatenates 12-D base + 10-D trajectory = 22-D:
- LogisticRegression(C=1.0, balanced, StandardScaler)
- Leave-one-dataset-out cross-validation (6-fold LOO CV)

#### 3. Single-Feature Ablation

Each feature independently trains a LogisticRegression → LOO CV → evaluates that feature's standalone predictive power.

### Evaluation: Leave-One-Dataset-Out CV (LOO CV)

- 6 datasets: aime24, aime25, brumo25, gpqa, hmmt25, livecodebench_v5
- Hold out 1 dataset as test, train on remaining 5
- Test evaluation is per problem group: select the run with highest P(correct), check if it is correct
- Report per-dataset accuracy and 6-dataset mean

### Normalisation Methods

| Suffix | Method | Description |
|--------|--------|-------------|
| `_z` | z-score | (x - μ) / σ, within-group standardisation |
| `_r` | rank | Within-group rank mapped to [0, 1] |
| none | raw value | Already in [0, 1] range (e.g., Jaccard similarity, entropy, Gini) |

### Key Findings

1. **Reflection is a signature of correct reasoning**: `reflection_count_r` (71.1%) surpasses all traditional features
2. **Layer distribution matters more than trajectory structure**: layer features (entropy, gini, deep_frac) all at 69-70%, while raw trajectory features (continuity, novelty) only 57-62%
3. **More features ≠ better**: 22-D fusion (68.3%) underperforms 12-D base (69.7%) — 6 datasets insufficient for high-dimensional LOO CV
4. **No-ML `layer-stratified` (69.4%) nearly matches ML `logistic` (69.7%)**: layer activation distribution is a robust, generalisable signal
