# CoT Viewer — Chain-of-Thought Browser / 思维链查看器

[English](#english) | [中文](#中文)

---

## 中文

一个轻量级 Web UI，用于浏览 NAD Next 缓存中已解码的推理链，检查逐 Token 指标，分析指标导数变化趋势，以及可视化神经元激活分布。

### 快速启动

```bash
/home/jovyan/work/NAD_Next/.venv/bin/python /home/jovyan/work/NAD_Next/cot_viewer/app.py
```

在浏览器中打开 **http://\<host\>:5002**。

**依赖：**
- Python 包：`flask`、`numpy`、`transformers`（已在 `.venv` 中）
- 分词器模型：`/home/jovyan/public-ro/model/DeepSeek-R1-0528-Qwen3-8B`
- 缓存数据：`/home/jovyan/public-ro/MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/`

### 功能概览

- **浏览** `MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/` 下的所有数据集
- **阅读** 每道题 64 次推理运行的完整解码思维链文本
- **多运行对比**：同时勾选多次运行，面板左右并排显示
- **检查** 每个 Token 的指标（置信度、熵、Gini、自我确定性、对数概率），点击片段即可查看
- **导数面板**：展示各指标随推理步进的变化趋势（一阶/二阶/三阶导数）
- **神经元激活面板**：Layer × Slice 热力图、片段间 Jaccard 相似度曲线

### 界面布局

```
┌─────────────────────────────────────────────────────────────┐
│  [数据集 ▼]  [题目 ▼]  [切片模式 ▼]            状态信息    │
├─────────────────────────────────────────────────────────────┤
│  运行选择条: [☑Run 0 ✓] [☐Run 1 ✗] [☐Run 2 ✓] ...        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─── Run 0 ✓ ───┐  ┌─── Run 3 ✗ ───┐                     │
│  │ 思维链文本      │  │ 思维链文本      │  ← 可并排多个面板  │
│  │ (可点击片段)    │  │ (可点击片段)    │                     │
│  └────────────────┘  └────────────────┘                     │
├─────────────────────────────────────────────────────────────┤
│  ▸ Derivatives（可折叠）                                     │
│    [指标勾选] [导数阶数] [布局: Overlay / Split]             │
│    ┌ SVG 折线图：指标导数随 slice 的变化趋势 ┐               │
├─────────────────────────────────────────────────────────────┤
│  ▸ Neuron Activation（可折叠）                               │
│    [指标: Count / w_sum / w_max]  [曲线: entropy, conf]     │
│    [布局: Side-by-side / Difference]                         │
│    ┌ 热力图: Layer(Y) × Slice(X) ─────────────────── ┐     │
│    │                                                   │     │
│    ├ Jaccard 折线: 相邻片段相似度 ────────────────── ┤     │
│    ├ Token 指标曲线: entropy / conf ─────────────── ┤     │
│    └───────────────────────────────────────────────── ┘     │
├─────────────────────────────────────────────────────────────┤
│  Token Detail（点击片段后展开）                               │
│  ┌──────┬──────┬───────┬───┬──────┬────┬─────┐             │
│  │ 位置 │Token │ 置信度│ H │ Gini │ SC │ LP  │             │
│  └──────┴──────┴───────┴───┴──────┴────┴─────┘             │
└─────────────────────────────────────────────────────────────┘
```

### 交互流程

1. **页面加载** → 数据集下拉框自动填充（aime24、aime25、gpqa 等）
2. **选择数据集** → 题目列表显示题目 ID、运行次数和准确率 %
3. **选择题目** → 运行芯片条显示每次运行的 `Run N ✓/✗`，默认选中第一个
4. **勾选运行** → 点击芯片或勾选框切换选中；可同时选中多个运行
5. **查看思维链** → 每个选中的运行展示为一个独立面板，左右并排
6. **点击片段** → 底部 Token Detail 表格显示该片段内每个 Token 的完整指标
7. **导数面板** → 自动展开，显示指标随推理步进的导数变化
8. **神经元面板** → 自动展开，显示热力图和曲线

### 切片模式

- **Fixed (32 tok)**：每个原始行（约 32 个 token）为一个片段
- **Smart Slice**：根据语言边界（换行、句号等）智能合并相邻行为"超级片段"

---

### 如何阅读各面板

#### 1. 思维链文本面板

每个选中的运行以独立面板显示。文本被分割为若干片段（slice），以虚线分隔。点击某个片段会在底部展开 Token Detail 表格。

**运行标记**：✓ = 最终答案正确，✗ = 最终答案错误。

#### 2. Token Detail 表格

点击任意片段后显示。每行对应一个 token，列含义：

| 列 | 含义 | 怎么看 |
|----|------|--------|
| Pos | token 在响应中的绝对位置 | — |
| Token | 解码后的文本 | — |
| Conf | 置信度 (confidence) | 值越大 → 模型越确信这个 token |
| H (Entropy) | Shannon 熵 | 值越大 → 预测越分散/不确定 |
| Gini | Gini 杂质度 | 越大 → 概率分布越均匀 |
| SC (SelfCert) | 自我确定性 | 越大 → 模型自评越确信 |
| LP (LogProb) | 对数概率 | 越接近 0 → 概率越高 |

**颜色**：单元格**绿色** = 熵 < 0.3（高确定性），**红色** = 熵 > 2.0（高不确定性）。

#### 3. Derivatives 导数面板

展示 token 级指标（entropy、conf、gini、selfcert、logprob）**按片段平均后**的变化趋势。

**控件：**
- **Metrics 勾选**：选择要显示哪些指标（默认 entropy + conf）
- **Order 勾选**：选择显示哪些阶导数
  - `avg`：原始平均值，看绝对水平
  - `d1`（一阶导数）：**变化速率**——突然跳升/下降意味着模型确信程度剧变
  - `d2`（二阶导数）：**变化加速度**——峰值处对应 d1 拐点
  - `d3`（三阶导数）：更高阶震荡，通常只在特殊分析时需要
- **Layout**（多运行时出现）：
  - `Overlay`：所有运行叠加在同一张图上，第一个运行用粗线、其他用半透明细线
  - `Split`：每个运行单独一张子图，便于独立观察

**怎么看：**
- d1 曲线从正骤变到负（或反之） → 模型在这个推理步骤发生了"思路转折"
- entropy 的 avg 在某段持续走高 → 模型对该段推理缺乏信心
- 多运行 Overlay 时，正确运行（✓）和错误运行（✗）在某个 slice 附近出现明显分叉 → 该位置是"关键决策点"

**线型说明：**

| 线型 | 含义 |
|------|------|
| 实线 | avg（原始平均值） |
| 长虚线 (— — —) | d1（一阶导数） |
| 短点线 (· · ·) | d2（二阶导数） |
| 点划线 (—·—·) | d3（三阶导数） |

**颜色含义：**

| 颜色 | 指标 |
|------|------|
| 红 | entropy（熵） |
| 蓝 | conf（置信度） |
| 绿 | gini |
| 橙 | selfcert |
| 紫 | logprob |

#### 4. Neuron Activation 神经元激活面板

这是信息密度最高的面板，展示推理过程中**神经元在各层的激活模式**。

##### 4a. 热力图（上半部分）

一张 **Layer（Y 轴）× Slice（X 轴）** 的二维热力图。

- **每个格子**代表"在该层(layer)的该片段(slice)中，有多少（或多强的）神经元被激活"
- **颜色深浅**：白色 → 浅蓝 → 深藏蓝
  - **白色/浅色** = 该层在该片段几乎无激活
  - **深蓝色** = 该层在该片段有大量/强烈的激活
- **Y 轴标签** `L16, L17, ... L35` 是模型的层编号
- **鼠标悬停**在格子上可看到精确数值（tooltip）
- **点击格子**会跳转到对应的思维链片段

**三种指标（通过顶部 Metric 单选框切换）：**

| 指标 | 含义 | 适用场景 |
|------|------|----------|
| **Count** | 每层每片段中**活跃神经元数量** | 看哪些层参与最多——"谁在干活" |
| **w_sum** | 每层每片段中激活权重的**总和** | 看每层的**总体贡献强度** |
| **w_max** | 每层每片段中激活权重的**最大值** | 看是否有**单个特别强的神经元**在主导 |

**典型观察模式：**

- **横向一条深色带**（某层始终深色） → 该层是"常驻工作层"，整个推理过程持续参与
- **纵向一列突然变深**（某个 slice 所有层都变深） → 该推理步骤触发了全面的神经网络响应，通常对应**关键推理步骤**或**思路转折**
- **深色区域从低层扩展到高层**（或反之） → 信息在模型中的流动方向可见
- **某段连续的 slice 整体偏白** → 模型在"惯性推理"，没有太多新的神经元被动员

##### 4b. Jaccard 相似度曲线（中间折线）

标注为 **Jacc** 的折线图，展示**相邻片段之间的神经元重叠度**。

- **Jaccard 系数** = |A ∩ B| / |A ∪ B|，值域 [0, 1]
  - **1** = 两个相邻片段激活了完全相同的神经元
  - **0** = 两个相邻片段没有任何共同激活的神经元
- **高值区间**（线靠近顶部）→ 模型在**稳定推理**，连续 slice 的激活模式相似
- **突然下降**（线急剧跌落）→ 发生了**结构性转换/相变**——模型的内部计算模式突然改变

**怎么用：**
- 找 Jaccard 曲线的**低谷**，这些位置是推理链中的**自然分界点**（相当于"模型换了一种思路"）
- 对比正确/错误运行的 Jaccard 曲线：正确运行的结构转换通常更有规律，而错误运行可能出现频繁无序的震荡

##### 4c. Token 指标曲线（底部折线）

可选的 **entropy（红色）** 和 **conf（蓝色）** 折线，与热力图共享 X 轴。

- 作用是将**神经元层面的结构变化**和**token 层面的确信变化**放在同一张图中对照观察
- 例如：热力图某列变深 + entropy 同时上升 → 该推理步骤不仅激活了更多神经元，模型也更不确定——可能是在"尝试新方向"

##### 4d. 多运行模式（选中 2+ 个运行时出现）

**Side-by-side（并排）：**
- 每个运行各自一张完整的热力图，上下堆叠
- 共享 X 轴（slice 编号），可直接肉眼对比
- 看"正确运行 vs 错误运行"在哪些层/哪些推理阶段有结构差异

**Difference（差异）：**
- 显示 `Run[0] − Run[i]` 的逐格差值
- 使用**红-白-蓝发散色标**：
  - **红色** = Run 0 在此格的值**更高**（Run 0 激活更多/更强）
  - **蓝色** = Run i 在此格的值**更高**（对比运行激活更多/更强）
  - **白色** = 两者相近
- 差异图能一眼看出：
  - 两次运行的**主要分歧**在哪些层、哪些推理阶段
  - 正确运行是否在某些层有独特的激活模式

##### 综合阅读示例

> 选中一个正确运行（Run 0 ✓）和一个错误运行（Run 3 ✗），切换到 Difference 模式。
> 在 slice 80-90 附近看到一片明显的红色区域（集中在 L20-L28），说明正确运行在这些层激活了更多神经元。
> 对照 Jaccard 曲线，Run 0 在 slice 78 处有一个低谷（结构转换），而 Run 3 没有。
> 回到文本面板，点击 slice 78 查看文本内容——发现这是 Run 0 "回头检验"的起点，而 Run 3 跳过了这一步。
> 结论：L20-L28 层在 slice 78 附近的差异性激活与正确运行的自我纠错行为相关。

---

### API 端点

除 `/api/datasets` 外，所有端点均需 `?cache=<path>` 查询参数。

| 端点 | 返回 |
|------|------|
| `GET /api/datasets` | `{名称: 缓存路径, ...}` |
| `GET /api/problems?cache=` | `[{problem_id, num_runs, accuracy}, ...]` |
| `GET /api/runs/<problem_id>?cache=` | `[{sample_id, run_index, is_correct}, ...]` |
| `GET /api/chain/<sample_id>?cache=&mode=` | `{num_slices, slices: [{idx, text, tok_start, tok_end}]}` |
| `GET /api/derivatives/<sample_id>?cache=&mode=` | `{num_slices, metrics, averages, d1, d2, d3}` |
| `GET /api/slice/<sample_id>/<slice_idx>?cache=` | `{tokens: [{pos, token_id, text, conf, entropy, ...}]}` |
| `GET /api/neuron_heatmap/<sample_id>?cache=&mode=` | `{num_slices, layers, heatmap, jaccard, token_metrics}` |

### 架构说明

- **分词器**：启动时从本地模型目录加载一次
- **数据访问**：使用 `CacheReader.get_token_view()` 获取每个样本的 token 数组，用 `rows_token_row_ptr` 确定 32-token 片段边界；用 `rows_sample_row_ptr` / `rows_row_ptr` / `rows_keys` 访问神经元激活数据
- **熵约定**：缓存中存储的是负熵（`tok_neg_entropy`），API 层取反后返回正值
- **提示词 token**：行库中第 0 行是提示词（跳过），响应片段从第 1 行开始
- **懒加载**：`NadNextLoader` 和 `CacheReader` 实例在首次访问时创建，后续复用
- **神经元 key 编码**：`(layer << 16) | neuron_id`，uint32 格式

### 文件结构

```
cot_viewer/
  app.py              # Flask 后端
  templates/
    index.html        # 单页纯 JS 前端
  README.md           # 本文件
```

---

## English

A lightweight web UI for browsing decoded reasoning chains, inspecting per-token metrics, analyzing metric derivative trends, and visualizing neuron activation patterns from NAD Next caches.

## Quick Start

```bash
/home/jovyan/work/NAD_Next/.venv/bin/python /home/jovyan/work/NAD_Next/cot_viewer/app.py
```

Open **http://\<host\>:5002** in your browser.

### Requirements

- Python packages: `flask`, `numpy`, `transformers` (already in `.venv`)
- Tokenizer model at `/home/jovyan/public-ro/model/DeepSeek-R1-0528-Qwen3-8B`
- Cache data at `/home/jovyan/public-ro/MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/`

## Features

- **Browse** all datasets under `MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/`
- **Read** the full decoded chain-of-thought text for each of the 64 reasoning runs per problem
- **Multi-run comparison**: check multiple runs to display panels side by side
- **Inspect** per-token metrics (confidence, entropy, Gini, self-certainty, log-probability) by clicking on slices
- **Derivatives panel**: metric trends over reasoning steps (1st/2nd/3rd order derivatives)
- **Neuron Activation panel**: Layer × Slice heatmap, inter-slice Jaccard similarity curve

## UI Layout

```
┌────────────────────────────────────────────────────────────┐
│  [Dataset ▼]  [Problem ▼]  [Slice mode ▼]         Status  │
├────────────────────────────────────────────────────────────┤
│  Run chips: [☑Run 0 ✓] [☐Run 1 ✗] [☐Run 2 ✓] ...        │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌── Run 0 ✓ ──┐  ┌── Run 3 ✗ ──┐                        │
│  │ CoT text     │  │ CoT text     │   ← side-by-side      │
│  │ (clickable)  │  │ (clickable)  │                        │
│  └──────────────┘  └──────────────┘                        │
├────────────────────────────────────────────────────────────┤
│  ▸ Derivatives (collapsible)                               │
│    [Metric checkboxes] [Order] [Layout: Overlay / Split]   │
│    ┌ SVG line chart ─────────────────────────────── ┐     │
├────────────────────────────────────────────────────────────┤
│  ▸ Neuron Activation (collapsible)                         │
│    [Metric: Count/w_sum/w_max] [Curves] [Layout]          │
│    ┌ Heatmap: Layer(Y) × Slice(X) ──────────────── ┐     │
│    ├ Jaccard curve ─────────────────────────────── ┤     │
│    ├ Token metric curves ──────────────────────── ┤     │
│    └───────────────────────────────────────────── ┘     │
├────────────────────────────────────────────────────────────┤
│  Token Detail (shown after clicking a slice)               │
│  ┌─────┬───────┬──────┬────┬─────┬────┬──────┐           │
│  │ Pos │ Token │ Conf │ H  │Gini │SC  │  LP  │           │
│  └─────┴───────┴──────┴────┴─────┴────┴──────┘           │
└────────────────────────────────────────────────────────────┘
```

## Interaction Flow

1. **Page loads** → datasets dropdown auto-populates (aime24, aime25, gpqa, etc.)
2. **Select a dataset** → problems list shows problem ID, run count, and accuracy %
3. **Select a problem** → run chip bar shows `Run N ✓/✗` for each run; first is auto-selected
4. **Toggle runs** → click chips or checkboxes to select/deselect; multiple runs can be active
5. **View CoT** → each selected run gets its own panel, displayed side by side
6. **Click a slice** → Token Detail table shows per-token metrics for that slice
7. **Derivatives panel** → auto-expands, shows metric derivative trends over slices
8. **Neuron Activation panel** → auto-expands, shows heatmap and curves

### Slice Modes

- **Fixed (32 tok)**: each raw row (~32 tokens) is one slice
- **Smart Slice**: merges adjacent rows at language boundaries (newlines, periods, etc.) into "super-slices"

---

## How to Read Each Panel

### 1. Chain-of-Thought Text Panel

Each selected run is shown as a separate panel. Text is split into slices (dashed separators). Click any slice to inspect its tokens.

**Run marks**: ✓ = final answer correct, ✗ = final answer incorrect.

### 2. Token Detail Table

Shown after clicking any slice. Each row is one token:

| Column | Meaning | Interpretation |
|--------|---------|----------------|
| Pos | Absolute position in response | — |
| Token | Decoded text | — |
| Conf | Confidence | Higher → model is more sure about this token |
| H (Entropy) | Shannon entropy | Higher → prediction is more spread out / uncertain |
| Gini | Gini impurity | Higher → probability distribution is more uniform |
| SC (SelfCert) | Self-certainty | Higher → model self-rates as more confident |
| LP (LogProb) | Log-probability | Closer to 0 → higher probability |

**Cell colors**: **green** = entropy < 0.3 (high certainty), **red** = entropy > 2.0 (high uncertainty).

### 3. Derivatives Panel

Shows token-level metrics (entropy, conf, gini, selfcert, logprob) **averaged per slice**, then differentiated.

**Controls:**
- **Metrics checkboxes**: which metrics to display (default: entropy + conf)
- **Order checkboxes**: which derivative orders to show
  - `avg`: raw per-slice average — shows absolute level
  - `d1` (1st derivative): **rate of change** — sudden jumps/drops mean the model's certainty shifted rapidly
  - `d2` (2nd derivative): **acceleration of change** — peaks correspond to inflection points in d1
  - `d3` (3rd derivative): higher-order oscillation, usually for specialized analysis only
- **Layout** (multi-run only):
  - `Overlay`: all runs superimposed on one chart; first run is bold, others are translucent
  - `Split`: each run gets its own sub-chart for independent inspection

**What to look for:**
- d1 flipping from positive to negative (or vice versa) → the model changed direction at that reasoning step
- avg entropy rising over a stretch of slices → the model is losing confidence during that phase
- In multi-run Overlay: if correct (✓) and incorrect (✗) runs diverge sharply near a slice → that's a **critical decision point**

**Line styles:**

| Style | Meaning |
|-------|---------|
| Solid | avg (raw average) |
| Long dash (— — —) | d1 (1st derivative) |
| Short dot (· · ·) | d2 (2nd derivative) |
| Dash-dot (—·—·) | d3 (3rd derivative) |

**Colors:**

| Color | Metric |
|-------|--------|
| Red | entropy |
| Blue | conf |
| Green | gini |
| Orange | selfcert |
| Purple | logprob |

### 4. Neuron Activation Panel

The highest information-density panel. Shows how **neurons across layers fire during reasoning**.

#### 4a. Heatmap (top region)

A **Layer (Y-axis) × Slice (X-axis)** 2D grid.

- Each cell represents "how many (or how strongly) neurons fired in that layer during that slice"
- **Color scale**: white → light blue → dark navy
  - **White / light** = very few or no activations in that layer for that slice
  - **Dark blue** = many or strong activations
- **Y-axis labels** `L16, L17, ... L35` are model layer numbers
- **Hover** over a cell to see the exact value (tooltip)
- **Click** a cell to jump to the corresponding CoT slice

**Three metrics (switch via radio buttons):**

| Metric | Meaning | Use case |
|--------|---------|----------|
| **Count** | Number of **active neurons** per layer per slice | See which layers are most involved — "who's working" |
| **w_sum** | **Sum** of activation weights per layer per slice | See each layer's **total contribution intensity** |
| **w_max** | **Maximum** activation weight per layer per slice | See if a **single dominant neuron** is driving the response |

**Typical patterns:**

- **Horizontal dark band** (one layer is consistently dark) → that layer is an "always-on worker" throughout reasoning
- **Vertical dark column** (all layers spike at one slice) → that reasoning step triggered a broad neural response — often a **key reasoning step** or **direction change**
- **Dark region spreading from lower to upper layers** (or vice versa) → information flow direction is visible
- **A stretch of mostly white slices** → the model is in "inertial reasoning", not recruiting many new neurons

#### 4b. Jaccard Similarity Curve (middle line)

Labeled **Jacc**, this polyline shows **neuron overlap between consecutive slices**.

- **Jaccard coefficient** = |A ∩ B| / |A ∪ B|, range [0, 1]
  - **1** = two adjacent slices activated exactly the same neurons
  - **0** = two adjacent slices share no activated neurons at all
- **High values** (line near the top) → the model is in **stable reasoning**, consecutive slices have similar activation patterns
- **Sudden drops** (sharp dips) → a **structural transition / phase shift** — the model's internal computation pattern changed abruptly

**How to use:**
- Find **valleys** in the Jaccard curve — these are natural **segmentation points** in the reasoning chain ("the model switched to a different approach")
- Compare Jaccard curves between correct/incorrect runs: correct runs often have more regular transitions, while incorrect runs may show frequent erratic oscillations

#### 4c. Token Metric Curves (bottom lines)

Optional **entropy (red)** and **conf (blue)** polylines, sharing the X-axis with the heatmap.

- Purpose: juxtapose **neuron-level structural changes** with **token-level confidence changes** in one view
- Example: if a heatmap column turns dark AND entropy rises at the same slice → that step recruited more neurons while the model was also more uncertain — likely "exploring a new direction"

#### 4d. Multi-Run Modes (visible when 2+ runs selected)

**Side-by-side:**
- One full heatmap per run, stacked vertically
- Shared X-axis (slice numbers) for direct visual comparison
- Look at "correct vs. incorrect run" to find structural differences by layer and reasoning phase

**Difference:**
- Shows `Run[0] − Run[i]` cell by cell
- Uses a **red-white-blue diverging color scale**:
  - **Red** = Run 0's value is **higher** at that cell (Run 0 has more/stronger activation)
  - **Blue** = Run i's value is **higher** (the comparison run has more/stronger activation)
  - **White** = roughly equal
- The difference view instantly reveals:
  - Where the **main divergence** occurs between runs (which layers, which reasoning phases)
  - Whether the correct run has unique activation patterns in certain layers

#### Putting It All Together — Example Workflow

> Select a correct run (Run 0 ✓) and an incorrect run (Run 3 ✗). Switch to Difference mode.
> Around slices 80–90, you see a prominent red region (concentrated in L20–L28), meaning Run 0 activated more neurons in those layers.
> Checking the Jaccard curve, Run 0 has a valley at slice 78 (structural transition), while Run 3 does not.
> Back in the text panel, click slice 78 to read the text — it turns out this is where Run 0 begins "checking its work", while Run 3 skipped this step.
> Conclusion: The differential activation in L20–L28 around slice 78 correlates with the correct run's self-correction behavior.

---

## API Endpoints

All endpoints (except `/api/datasets`) require a `?cache=<path>` query parameter.

| Endpoint | Returns |
|----------|---------|
| `GET /api/datasets` | `{name: cache_path, ...}` for all datasets |
| `GET /api/problems?cache=` | `[{problem_id, num_runs, accuracy}, ...]` |
| `GET /api/runs/<problem_id>?cache=` | `[{sample_id, run_index, is_correct}, ...]` |
| `GET /api/chain/<sample_id>?cache=&mode=` | `{num_slices, slices: [{idx, text, tok_start, tok_end}]}` |
| `GET /api/derivatives/<sample_id>?cache=&mode=` | `{num_slices, metrics, averages, d1, d2, d3}` |
| `GET /api/slice/<sample_id>/<slice_idx>?cache=` | `{tokens: [{pos, token_id, text, conf, entropy, ...}]}` |
| `GET /api/neuron_heatmap/<sample_id>?cache=&mode=` | `{num_slices, layers, heatmap, jaccard, token_metrics}` |

## Architecture Notes

- **Tokenizer**: loaded once at startup from the local model directory
- **Data access**: uses `CacheReader.get_token_view()` for per-sample token arrays and `rows_token_row_ptr` for 32-token slice boundaries; uses `rows_sample_row_ptr` / `rows_row_ptr` / `rows_keys` for neuron activation data
- **Entropy convention**: the cache stores negative entropy (`tok_neg_entropy`); the API negates it to return positive values
- **Prompt tokens**: row 0 in the rows bank is the prompt (skipped); response slices start at row 1
- **Lazy loading**: `NadNextLoader` and `CacheReader` instances are created on first access per cache and reused
- **Neuron key encoding**: `(layer << 16) | neuron_id`, stored as uint32

## Files

```
cot_viewer/
  app.py              # Flask backend
  templates/
    index.html        # Single-page vanilla JS frontend
  README.md           # This file
```
