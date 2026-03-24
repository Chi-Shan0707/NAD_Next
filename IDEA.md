# IDEA

## 20260323 — Smart Slice 与 Best-of-K 打分

### 背景

`app.py` 的 `_build_smart_slices()` 直接调用 `nad.ops.smart_slice.smart_slice_grouping`，无重复实现。

**核心问题**：用了更好的划分方式（smart slice），是否也能弄出更好的打分来选 Best-of-K？

### 灵感：为什么更好的边界 → 可能更好的打分

核心逻辑：每个 slice 代表一个语义单元（一句话、一个推理步骤），而不是任意切断的 32 个 token。神经元激活模式在这个单元内有了"意义"——它代表模型在做一件完整的事时的内部状态。

#### 方向一：逐步共识打分

现有方法本质是全局相似度（整个回答的 Jaccard 距离），隐含假设是"和其他回答最相似的那个最好"。

换成语义 slice 之后，可以做**局部共识**：

> 对于 Run A 的第 i 个 smart slice，在其他 K-1 个 run 里找激活最相似的 slice（不要求位置对齐）。如果大多数 run 都有一个"对应步骤"，说明这步是共识推理；如果这步在其他 run 里找不到对应物，说明这步是异类。

这样每个 run 可以得到一个**逐步共识得分**，而不是一个全局数字。

#### 方向二：Fork Point 检测

Wrong answer 往往不是全错，而是在某个决策点走偏了。Smart slice 对齐到自然语言边界，恰好和这些决策点重合（因为模型通常在句子结尾做"转向"）。

可以画出 K 个 run 的激活轨迹，找出它们开始分叉的那个 slice index——那个位置大概率就是错误发生的地方。这对 Best-of-K 的可解释性帮助很大。

### 批评

**最大的问题：对齐问题没解决。**

不同 run 的 smart slice 数量不同，顺序也可能不同。Run A 的 slice 5 和 Run B 的 slice 5 不一定在讨论同一件事。现在的固定 32-token 方案反而有一个"隐形优点"：位置天然对齐（token 0-31 vs 0-31）。

用了 smart slice 之后，如果还沿用"按 index 对比"的方式，反而可能更差。必须引入某种动态规划或 DTW（动态时间规整）才能真正做跨-run 的步骤对齐。

**第二个问题：更长的 slice = 更多的平均。**

一个 64-token 的 smart slice 其实把两个 32-token 的原始 slice 合并了。激活是 max/sum 聚合的，细粒度信息会被稀释。"语义更纯"的好处能不能抵消"分辨率降低"的坏处，并不显然。

### 结论

Smart slice 的真正价值在于让 fork point 可检测、让共识分析有意义，而不是简单地"更好的 slice → 更好的现有打分"。如果还是跑现有的 selector（greedy/medoid 等），可能收益有限。**需要配套一个新的"步骤对齐"机制**才能充分发挥潜力。

---

## 20260324 — MoSeq 启发的 CoT 阶段分类

### 一、Keypoint-MoSeq 算法详解

#### 1.1 它要解决什么问题

原始场景：用摄像头拍摄小鼠，通过 DeepLabCut 等工具追踪身体关键点（鼻子、尾巴、四肢等）的 2D 坐标。目标是**无监督地**把连续的运动流切成一个个离散的"行为音节"（behavioral syllables）——比如"转身"、"站立"、"探索"等。

核心难点：关键点追踪有高频抖动（jitter）。传统聚类算法会把抖动误判为行为切换，导致状态序列疯狂闪烁（flickering），一秒内切换几十次，完全不符合真实行为节奏。

#### 1.2 模型架构：三层 SLDS

Keypoint-MoSeq 采用**切换线性动力系统（SLDS）**，从上到下三层：

```
第 1 层（顶）：离散音节序列 z₁, z₂, ..., zₜ       ← "在做什么行为？"
                     ↓ 控制
第 2 层（中）：低维姿态轨迹 x₁, x₂, ..., xₜ       ← "身体实际在怎么动？"
                     ↓ 投影 + 噪声
第 3 层（底）：观测到的关键点坐标 y₁, y₂, ..., yₜ   ← "摄像头看到了什么？"
```

**关键直觉**：观测关键点 = 真实姿态 + 噪声。模型联合推断三层，把噪声和真实动态分开。

#### 1.3 每层的机制

**第 1 层：AR-HMM（自回归隐马尔可夫模型）**

离散状态 `z_t` = 当前音节编号，转移由 HMM 控制：

- **转移矩阵 π**：`P(z_t=j | z_{t-1}=i)` 从音节 i 切换到 j 的概率
- **粘性参数 κ（kappa）**：给"保持当前状态"加额外权重。κ 越大 → 切换越少 → 音节越长
  - 小鼠推荐：中位音节时长 ≈ 400ms（12 帧 @ 30fps）
  - 典型值：AR-HMM 阶段 κ=10⁵，全模型阶段 κ=2×10⁴
- **音节数量自动学习**：层次 Dirichlet 先验（HDP），无需预设

**第 2 层：自回归动态**

每个音节 `z_t = k` 定义一组参数 `(Aₖ, bₖ, Qₖ)`：

```
x_t = Aₖ · x_{t-1} + bₖ + ε_t,    ε_t ~ N(0, Qₖ)
```

- `Aₖ`：状态转移矩阵（"做音节 k 时姿态怎么演变"）
- `bₖ`：偏移量
- `Qₖ`：过程噪声协方差

不同音节有不同的 `(A, b, Q)` → 每种行为有独特的**动力学签名**。

降维：原始关键点（如 12 点 × 2 = 24 维）PCA 降到解释 90% 方差的维度（通常 4-10 维）。

**第 3 层：观测模型（Keypoint-MoSeq 的创新）**

```
y_t = C · x_t + d_t + noise
```

- `C`：低维→关键点的投影矩阵
- `d_t`：质心位置 `v_t` + 朝向角 `h_t`（全局平移/旋转对齐）
- 逐关键点噪声 `σ²_k` + 逐帧缩放 `s_{t,k}`

**抗噪核心**：某个关键点突然跳动 → 模型归因于噪声 `s_{t,k}`，而非改变姿态 `x_t` 或切换音节 `z_t`。

#### 1.4 推断：两阶段 Gibbs 采样

所有变量通过 Gibbs 采样（MCMC）联合推断：固定其他变量 → 按条件后验采样更新一个 → 循环。

| 阶段 | 做什么 | 更新变量 |
|---|---|---|
| Stage 1: AR-HMM 初始化 | 去质心、旋转对齐、PCA+白化；假设无观测噪声 | `z, A, b, Q, π, β` |
| Stage 2: 完整 SLDS | 在 Stage 1 基础上精细拟合，~500 次迭代 | 额外推断 `v, h, x, σ², s` |

**输出**：每帧一个音节标签 + 每种音节的运动模板。

#### 1.5 一句话流程

```
视频帧 → 关键点追踪 → 去质心/旋转/PCA
  → Stage 1: AR-HMM 粗切分（忽略噪声）
  → Stage 2: SLDS 精细拟合（分离噪声 vs 真实动态）
  → 输出：每帧音节标签 + 音节运动模板
```

### 二、MoSeq ↔ NAD Token 场景的适配性分析

#### 2.1 类比映射

| Keypoint-MoSeq | NAD Token 场景 |
|---|---|
| 每帧的关键点坐标 | 每个 token 的神经元激活向量 |
| 行为音节（转身、探索…） | 推理阶段（思考、计算、回顾…） |
| 关键点抖动/噪声 | 激活中的随机波动/非语义变化 |
| 音节边界 = 动态不连续点 | smart slice 边界 = 语义/激活转折点 |
| 多帧组成一个音节 | 多 token 组成一个 slice |

#### 2.2 适合借鉴的方面

1. **粘性 HMM 天然适合切片**：κ 控制"停留倾向"，对应直觉——模型做一件事时内部状态应保持稳定，直到真正切换。比启发式规则（标点/token 数阈值）更有原则性。
2. **音节数量自动发现**：HDP 先验让模型自己决定推理状态种数，无需预设。

#### 2.3 不适合 / 需要大改的方面

1. **维度灾难**：小鼠 24 维 → PCA 4-10 维；神经元激活几千到几万维。Gibbs 采样高维收敛极慢，需先降维到 ~50 维以下。
2. **时间尺度不同**：小鼠 30fps / 音节 ~12 帧；LLM 一个推理阶段可能 10-200 tokens。κ 需重新标定，且 AR 线性假设在 token 粒度未必成立。
3. **计算成本太高**：每 run 500 次 Gibbs × K runs × 多 problem groups，16 核机器上很慢。
4. **不需要"噪声分离"**：激活数据是精确的（无测量误差），所谓"噪声"是结构性的"非语义激活变化"，SLDS 的高斯独立噪声模型未必能捕捉。

#### 2.4 判断

**直接套用不合适**，但两个核心思想值得借鉴：

1. **粘性 HMM 做序列分割**：单独用 AR-HMM + sticky prior，在降维激活空间上跑，可得到比启发式更有原理性的 slice 边界。
2. **自回归动力学签名**：每种推理状态有自己的 `(A, b, Q)`，可给 slice 打特征标签，进而做跨 run 的步骤对齐。

更轻量的替代：**HMM-GLM** 或 **Sticky HDP-HMM**（不带 SLDS 观测层），配合 PCA 降维后的激活向量。

> **参考文献**：
> - [Keypoint-MoSeq | Nature Methods (2024)](https://www.nature.com/articles/s41592-024-02318-2)
> - [bioRxiv Preprint](https://www.biorxiv.org/content/10.1101/2023.03.16.532307v1.full)
> - [官方文档](https://keypoint-moseq.readthedocs.io/) / [GitHub](https://github.com/dattalab/keypoint-moseq)

---

### 三、实际方案：CoT 三阶段分类（Exploration / Reflection / Exploitation）

不需要实现完整 MoSeq，用聚类 + MoSeq 借鉴即可。

#### 3.1 三阶段的特征预期

| 阶段 | 语义 | entropy | confidence | 激活模式 |
|---|---|---|---|---|
| **Exploration** | 试探方向、发散思路 | 高 | 低 | 稀疏、多变、跨 slice 差异大 |
| **Reflection** | 回顾检查、自我纠错 | 中等 | 中等 | 与之前某些 slice 相似（重访） |
| **Exploitation** | 确信地执行推导 | 低 | 高 | 密集、稳定、相邻 slice 相似 |

#### 3.2 Pipeline

**Step 1：切 Slice** — `smart_slice_grouping()` 已有。

**Step 2：每 Slice 提取特征向量** — 两类特征拼接：

A. **激活特征**（"哪些神经元亮了"）

```python
# 方案 1：Jaccard 距离矩阵 + 直接聚类（复用 DistanceEngine，不需降维）
# 方案 2：统计特征代替原始向量（轻量）
features_activation = [
    num_active_neurons,          # 激活神经元数量
    active_neuron_overlap_ratio, # 与前一个 slice 共享激活的比例
    activation_density_change,   # 相对前一 slice 的密度变化率
]
```

B. **Token 级统计特征**（"模型有多确信"）— `token_data/` 已有：

```python
features_token = [
    mean_tok_conf,        # slice 内平均 confidence（低 = 更确信）
    mean_tok_neg_entropy, # slice 内平均负熵（接近 0 = 更确定）
    std_tok_conf,         # slice 内 confidence 波动
    mean_tok_gini,        # Gini 系数
    mean_tok_logprob,     # 平均 log probability
]
```

拼接后每个 slice → ~5-8 维特征向量。

**Step 3：聚类（K=3）**

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = km.fit_predict(features)  # features: (num_slices, num_features), 已标准化
```

替代方案：Jaccard 距离矩阵 + 谱聚类（Spectral Clustering），对稀疏集合更自然。

**Step 4：语义标签赋予** — 按 cluster 中心的统计特征自动标注：

```python
# Exploration:  highest conf (least confident), lowest overlap
# Exploitation: lowest conf (most confident), highest overlap
# Reflection:   中间，或 overlap 高但 conf 也高（回看旧内容但不确定）
```

#### 3.3 与 Best-of-K 选择的关系

分出三阶段后可构造新的 selector 特征：

1. **阶段比例**：Exploitation 比例越高 → 更多时间在笃定执行 → 可能更好
2. **E→X 转折点**：越早从探索转入执行 → 方向找对了
3. **Reflection 位置**：出现在中后段并紧接 Exploitation → "检查后确认" → 高质量信号
4. **末段状态**：Exploitation 结尾（自信收尾）vs Exploration 结尾（试探未完就断了）→ 强烈质量信号

#### 3.4 评价

**值得做，但控制预期。**

好处：
- 实现成本低，`CacheReader` + `token_data` + `smart_slice` 覆盖 80% 数据管线
- 高度可解释，可在 `cot_viewer` 里可视化 slice 阶段颜色
- "阶段比例"作为额外 feature 喂给 selector 排名不会更差

风险：
- 三阶段是人的直觉，模型内部未必如此组织。聚类结果可能按内容类型分（数学推导/自然语言/代码），而非按认知阶段分
- 验证困难——无 ground truth 标注，只能看聚类结果是否"看起来合理"

**建议的第一步**：在 `cot_viewer` 对一题多 run 做可视化——每个 slice 一个方块，颜色编码 `mean_tok_conf`，看人眼能否看出阶段结构。能看出"前段不确定→后段确定"的梯度则路线可行，否则聚类也救不回来。

---

### 四、MoSeq 的核心借鉴：轨迹 > 状态

MoSeq 最精妙的不是 SLDS 的数学，而是一个**建模哲学**：

> 传统聚类问："这个 slice **是什么**？" → 映射到一个点，看它离哪个中心近。
>
> MoSeq 问："这个 slice **在往哪走**？" → 不光看位置，还看动力学方向。

MoSeq 的每个音节不是静态聚类中心，而是一组自回归参数 `(A, b, Q)`——"处于这个状态时系统如何演化"。两个 slice 可能在特征空间同一位置，但一个在加速远离（exploration 要发散），一个在减速收敛（exploitation 要锁定）。**纯静态聚类无法区分，AR 动力学可以。**

#### 4.1 静态 vs 动态特征对比

| 阶段 | 静态特征（快照） | 动态特征（导数，MoSeq 启示） |
|---|---|---|
| Exploration | 高 entropy，低 confidence | entropy **在升**，相邻 slice 差异**在增大** |
| Reflection | 中等 entropy | entropy **突变**，激活模式**回访**之前某 slice |
| Exploitation | 低 entropy，高 confidence | entropy **持续低位稳定**，相邻差异**持续小** |

#### 4.2 轻量实现：一阶差分特征

```python
# 原始 slice 特征（静态）
f_t = [mean_conf, mean_neg_entropy, mean_gini, num_active, overlap_ratio]

# MoSeq 启发的动态特征（一阶差分）
delta_t = f_t - f_{t-1}   # 这些值在涨还是在跌？

# 拼接
feature_t = concat(f_t, delta_t)
```

不需要 AR 参数、Gibbs 采样、SLDS。只是把"行为是轨迹不是状态"用最简单的方式实现。

#### 4.3 轻量实现：粘性先验 → 后处理平滑

K-Means 对时序无感，不应出现 `E→X→E→X→E` 这种抖动。正常 CoT 应是 `E→E→E→R→X→X→X`。

```python
# 聚类后中值滤波消除孤立跳变
from scipy.ndimage import median_filter
smoothed_labels = median_filter(raw_labels, size=3)

# 或：强制最短阶段长度，1-slice 阶段合并到前一阶段
```

粘性先验的平民版——效果 80%，复杂度 1%。

#### 4.4 总结：从 MoSeq 借什么、不借什么

| 借什么 | MoSeq 原始形态 | 轻量实现 |
|---|---|---|
| **轨迹 > 状态** | 自回归参数 `(A, b, Q)` | 特征一阶差分 `Δf_t = f_t - f_{t-1}` |
| **切换有成本** | Sticky HDP-HMM (`κ`) | 聚类后中值滤波 / 最短阶段长度约束 |
| **噪声分离** | SLDS 观测模型 | **不借。** 激活数据无测量噪声，不需要这层 |

> 第三点是关键的"不借"——激活数据是精确的，不存在关键点抖动问题。MoSeq 的整个观测层是为了解决一个我们没有的问题。**知道什么不该借，和知道什么该借一样重要。**
