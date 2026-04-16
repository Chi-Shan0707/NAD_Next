# NAD Next — Neuron Activation Distribution

[English](#english) | [中文](#中文)

---

## 中文

### Workspace 导航

- 仓库整体布局见 `WORKSPACE_LAYOUT.md`
- 文档索引见 `docs/README.md`
- 实验结果索引见 `results/README.md`
- 提交 JSON 索引见 `submission/README.md`
- 脚本入口索引见 `scripts/README.md`
- 模型产物索引见 `models/README.md`

### Code reasoning pilot / 代码推理 pilot

- `research_summary.md`: hypothesis framing, established vs speculative claims, and next experiments
- `literature_map.md`: literature clusters with direct-support vs near-neighbor distinctions
- `hypotheses_and_rqs.md`: falsifiable RQs/Hs, IVs/DVs/confounds
- `experiment_plan.md`: benchmark + evaluation + threats to validity
- `prompt_ablations.md`: prompt templates for free-form vs structured/stateful reasoning
- `failure_taxonomy.md`: trace error taxonomy
- `pilot_benchmark.py`: runnable synthetic benchmark generator
- `sample_tasks.jsonl`: generated sample tasks with gold outputs and traces

Run from repo root:

```bash
python3 pilot_benchmark.py --num-per-family 3 --seed 0 --out sample_tasks.jsonl --pretty-sample 5
```

Each JSONL record contains:

- `code`, `entry_call`, `gold_output`
- `gold_trace` with per-step state / branch / loop / scope metadata
- controllable attributes such as `branch_depth`, `loop_nesting`, `phase_switch_count`, `boundary_case_count`

This pilot targets **code reasoning / execution**, not code generation.

### 当前研究主线（2026-04-08）

- `code_v2` 已是当前 promoted coding default，详见 `docs/CODE_V2_EXHAUSTIVE_20260406.md`
- `science_hybrid_round3` 是当前 promoted science patch，详见 `docs/SCIENCE_HYBRID_ROUND3_RESULTS_20260406.md`
- `math_deepsets_round1` 现已成为当前 promoted math patch，详见 `docs/MATH_DEEPSETS_ROUND1_RESULTS_20260408.md`
- `gpqa_pairwise_round2` 结论仍为 `NO-PROMOTE`，详见 `docs/GPQA_PAIRWISE_ROUND2_RESULTS_20260406.md`
- `gpqa_deepsets_round1` 已完成最小 full-group contextual 试验，但结论仍为 `NO-PROMOTE`，详见 `docs/GPQA_DEEPSETS_ROUND1_RESULTS_20260407.md`
- `code_deepsets_round1` 已完成第一轮 coding 扩展，但结论仍为 `NO-PROMOTE`，详见 `docs/CODE_DEEPSETS_ROUND1_RESULTS_20260408.md`
- 当前 science 新研究线优先级：小型 contextual model / top-slot calibration，而不是 graph-heavy 扩展或新的 monotonic recency feature family
- `SVDomain` 论文线新增了 cross-anchor transfer 证据：见 `docs/11_CROSS_ANCHOR_TRANSFER.md` 与 `results/tables/cross_anchor_transfer_summary.csv`

神经元激活分布（NAD Next）是一个用于分析神经网络激活的框架，通过二进制 CSR 缓存、选择器算法和可复现的实验手册进行分析。NAD Next 将原始 NPZ 激活分片转换为高效的内存映射缓存（CSR 格式，带 Roaring Bitmap 索引），应用 24 种选择算法（含 ML、时序折扣和轨迹分析）为每道题目挑选最具代表性的样本，并跨模型和数据集评估选择器精度。

### 快速开始

```bash
bash cookbook/00_setup/install.sh                          # 安装依赖
bash cookbook/00_setup/verify.sh                            # 所有检查应显示绿色
bash cookbook/01_cache_browser/cache_browser.sh --background  # 在 :5003 浏览缓存
bash cookbook/02_visualization/visualization.sh \
  MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20250902_025610 \
  --background                                             # 在 :5002 探索单个缓存
bash cookbook/03_batch_analyze/batch_analyze.sh -y          # 分析所有缓存
bash cookbook/05_deepconf_analysis/deepconf_analysis.sh standard --compare-all  # deepconf 分析
```

### 实验手册（Cookbook）

所有脚本位于 [`cookbook/`](cookbook/)，需从**仓库根目录**运行。

| 章节 | 说明 |
|------|------|
| [00 — 环境配置](cookbook/00_setup/README.md) | 安装 Python 依赖，验证环境（Python 3.9+、包、MUI_HUB 软链接） |
| [01 — 缓存浏览器](cookbook/01_cache_browser/README.md) | Web UI，端口 5003，展示所有缓存的数据集名、样本数、准确率、温度、构建日期 |
| [02 — 可视化](cookbook/02_visualization/README.md) | 交互式单缓存探索器，端口 5002（Flask + Plotly），浏览激活模式、token 置信度、选择器表现 |
| [03 — 批量分析](cookbook/03_batch_analyze/README.md) | 对 `MUI_HUB/cache/` 下所有缓存并行分析，生成各选择器的精度结果 |
| [04 — 位置消融](cookbook/04_position_ablation/README.md) | 跨 6 个位置窗口（0-1、0-2、0-8、0-32、0-128、0-512）消融分析 |
| [05 — DeepConf 分析](cookbook/05_deepconf_analysis/README.md) | 基于 `tok_conf` 和 `tok_neg_entropy` 的 token 置信度选择器，四种模式 |

### CoT 查看器 — 思维链浏览器

轻量级 Web UI，端口 5002，用于阅读解码后的推理链，查看每个 token 的置信度、熵、Gini 系数、自我确定性和对数概率。

```bash
/home/jovyan/work/NAD_Next/.venv/bin/python /home/jovyan/work/NAD_Next/cot_viewer/app.py
```

无需参数，自动发现 `MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/` 下的所有数据集。

### 架构

**数据流水线：**

```
NPZ 分片 --> cache-build-fast --> 二进制缓存（CSR + mmap）
                                       |
                                  nad.cli analyze --> 选择结果 JSON
                                       |
                                  nad.cli accuracy --> 精度报告
                                       |
                                  rank_selectors.py --> 跨任务排名
```

### 选择器

内置 35 种选择器算法，从每个题目组中挑选最具代表性的样本：

**基础选择器**

| 选择器 | 类型 | 说明 |
|--------|------|------|
| `min-activation` | [S] | 总神经元激活数最少 |
| `max-activation` | [S] | 总神经元激活数最多 |
| `min-confidence` | [S] | 最低均值 tok_conf（最自信） |
| `medoid` | [P] | 组的几何中心（Jaccard 距离） |
| `knn-medoid` | [P] | 限于 K 个最近邻的 medoid |
| `dbscan-medoid` | [P] | 最密 DBSCAN 簇的 medoid |
| `consensus-min` | [P] | 共识候选集中激活数最少者 |
| `consensus-max` | [P] | 共识候选集中激活数最多者 |
| `deepconf` | [S] | 基于 token 置信度滑窗（需要 `token_data/`） |
| `con64@` / `avg64@` | [O] | 共识 / 均值基线（oracle，仅完整序列） |

类型：`[S]` 单样本评分，无需距离矩阵；`[P]` 需要 N×N Jaccard 矩阵；`[O]` 使用全部标注的 oracle 基线。

**分组 Ensemble 选择器**（随机分组，逐轮淘汰）

| 选择器 | 说明 |
|--------|------|
| `ensemble-medoid` | 随机 8 人一组 → 每组 medoid → 胜者再 medoid |
| `ensemble-deepconf` | 随机 8 人一组 → 每组 DeepConf 最优 → 胜者再最优 |

**Tournament 选择器**（两两比较 + softmax）

| 选择器 | 说明 |
|--------|------|
| `tournament-copeland` | Copeland 投票：对每对 (i,j) 统计多数第三方更近者得分 → softmax |
| `tournament-deepconf` | DeepConf quality 两两比较 → 累计胜场 → softmax |

**两阶段选择器**（分组 Top-K → 决赛）

| 选择器 | 说明 |
|--------|------|
| `twostage-medoid` | 16 人一组 → 每组取组内距离前 4 → 16 人决赛 medoid |
| `twostage-tournament` | 16 人一组 → 每组 Copeland 取前 4 → 16 人决赛 Copeland + softmax |

**ML 选择器**（需要预训练模型，运行 `python scripts/train_ml_selectors.py` 生成）

从 6 个数据集的 31,040 个带标注 (题目, run) 对中训练，特征为 12 维组内归一化向量。

| 选择器 | 模型 | 说明 |
|--------|------|------|
| `linear-probe` | Ridge 回归 | 预测 is_correct 实数分数，取 argmax |
| `logistic` | 逻辑回归 | 预测 P(正确)，取 argmax |
| `isotonic-medoid` | 等渗回归（单特征） | 对 medoid rank 分数做单调校准 → P(正确) |
| `isotonic-deepconf` | 等渗回归（单特征） | 对 DeepConf rank 分数做单调校准 → P(正确) |

类型 `[ML]`。模型保存于 `models/ml_selectors/`，推理时懒加载。
留一数据集 CV：`logistic` 69.7%，`linear-probe` 69.8%——泛化能力与 `knn-medoid` 相当。
单特征消融显示 `dc_z`（DeepConf quality）以 70.9% 成为最强单特征，超过全部 12 特征模型。

**时序折扣切片选择器**（按 slice 分段，末端加权，无需距离矩阵）

| 选择器 | 类型 | 说明 |
|--------|------|------|
| `temporal-slice` | [S] | token 序列按 32 一段分段，γ^(2k) 指数折扣，加权质量分最高者胜出 |

类型 `[S]`，默认参数由网格搜索确定：`tok_neg_entropy`、γ=0.7、T=0.1。最优均值准确率 60.3%。
参数保存于 `models/ml_selectors/temporal_best_params.json`。

**轨迹分析选择器**（基于神经元激活模式的时序轨迹，需要 `rows/` bank v4.1+）

| 选择器 | 类型 | 说明 |
|--------|------|------|
| `trajectory` | [S] | 轨迹结构评分：α·连续性 - β·新颖度 + γ·末尾收敛 + δ·适度反思 |
| `layer-stratified` | [S] | 分层激活分布：α·深层占比 + β·层熵 - γ·层 Gini 系数 |
| `trajectory-fusion` | [ML] | 轨迹 + 层 + 现有 12-D 特征融合为 22-D，用 LogisticRegression 预测 |

类型 `[S]` / `[ML]`。轨迹特征利用 `rows/` bank 的逐切片（32 token）激活 key 集合计算切片间 Jaccard 相似度。
层特征通过 neuron key 编码（`layer<<16|neuron_id`）解码层信息。
实验结果：`layer-stratified` 69.4%，`trajectory` 58.7%，`trajectory-fusion` 68.3%。
单特征消融发现 `reflection_count_r`（反思次数 rank）以 71.1% 超越 `dc_z`（70.9%）成为新最佳单特征。
训练脚本：`python scripts/train_trajectory_selectors.py`。
最新产物快照（UTC）：评估报告生成于 `2026-03-30`，见 `results/trajectory_experiments/accuracy_summary_20260330_112435.json`、`results/trajectory_experiments/trajectory_20260330_112435.json`、`results/trajectory_experiments/layer_stratified_20260330_112435.json`；22 维轨迹融合训练统计更新于 `2026-03-31 01:56:52`，见 `models/ml_selectors/trajectory_stats.json`（`31,040` 个带标注样本对、`18,873` 个正确样本、`22` 个特征、`6` 个数据集）。
`2026-04-02 17:13:06` UTC 的 reflection dynamics follow-up 写入 `results/reflection_dynamics/summary.md`、`results/reflection_dynamics/threshold_sweep_summary.json`：将 reflection event 阈值从 `0.30` 调到 `0.20` 可把 `reflection_count_r` 的单特征 LOO 均值从 `71.1%` 提升到 `71.7%`。同一批分析还显示：reflection 与平均 `gini` 正相关、与平均 `entropy` 负相关，而 reflection event slice 上的 `confidence` 一阶/二阶离散变化幅度整体更低。
新增 `extreme8-best` / `extreme8-worst` / `extreme8-mixed` 三个 pooled subset 选择器：训练阶段仅使用 `dc_z`、`dc_r`、`reflection_count_r` 三个强特征，保留 problem accuracy ∈ `[10%, 90%]` 的题目，每题随机采样 `256` 个 mixed 8-tuples；`2026-04-02 17:56:57` UTC 完成的 blind 64-run 评估对每题使用 `512` 个随机 8-tuples（构造 tuple 时不看 run 对错），`best-only` / `best+worst` / `worst-avoid` 的 6 数据集均值均为 `72.5%`，`worst-only` 的错误命中率为 `56.1%`。对应产物见 `models/ml_selectors/extreme8_best.pkl`、`models/ml_selectors/extreme8_worst.pkl`、`models/ml_selectors/extreme8_stats.json`、`results/extreme8_experiments/20260402_112323/summary_20260402_112323.json`。注意：这轮 Extreme8 模型使用 `reflection_threshold=0.30` 训练；`0.20` 是随后由 follow-up threshold sweep 找到的更优单特征阈值，尚未回灌到本轮模型。

**Extreme9 局部置信度扩展选择器**（11 维，需要训练，`extreme9_impl.py`）

`extreme9-best` / `extreme9-worst` / `extreme9-mixed` 在 Extreme8 的 3 维强特征（`dc_z`、`dc_r`、`reflection_count_r`）基础上，加入 8 个来自 DeepConf 论文的局部 `tok_conf` 聚合特征，共 **11 维**：

| 特征 | 计算方式 | 质量方向 |
|------|---------|---------|
| `tail_2k_r` | 末尾 min(2000,T) 个 token 的均值 tok_conf 排名 | 越低越好 |
| `tail_q10_r` | 末尾 10% token 的均值 tok_conf 排名 | 越低越好 |
| `lgc_512_r` | 最差滑窗均值（窗口 512）排名 | 越低越好 |
| `lgc_2k_r` | 最差滑窗均值（窗口 2000）排名 | 越低越好 |
| `bottom_q10_r` | tok_conf 第 10 百分位排名 | 越低越好 |
| `head_tail_gap_r` | 首 10% 均值 − 尾 10% 均值排名（正值 = 尾部更自信） | 越高越好 |
| `last_event_tail_conf_r` | 最后一次 reflection event 之后所有 token 的均值 tok_conf 排名 | 越低越好 |
| `event_nonevent_gap_r` | event slice 均值 − non-event slice 均值排名 | 越低越好 |

训练脚本：`python scripts/train_extreme9_selectors.py`。产物：`models/ml_selectors/extreme9_{best,worst}.pkl`。
零训练基线 `local-conf-tail` 选最小 `tail_2k` 值的 run，可独立验证局部置信度信号方向。

**图拓扑零训练基线**（`graph-degree`，`graph_topo_impl.py`）

`graph-degree` 从 64-run Jaccard 距离矩阵构建相似图，以自适应 eps（off-diagonal 距离 30th 百分位）建立邻接关系，取**归一化度**（`norm_degree = degree / (n-1)`）最高的 run。高度 = 与最多其他 run 激活相似 = 倾向于处于共识正确簇中。用于在不训练模型的情况下验证图拓扑信号强度，与 `dc_r` / `dbscan-medoid` 对比。

**Extreme10 图拓扑 + 误差质量扩展选择器**（17 维，需要训练，`extreme10_impl.py`）

`extreme10-best` / `extreme10-worst` / `extreme10-mixed` 在 Extreme9 的 11 维基础上，再加入 **3 个图拓扑特征**（从 D 矩阵在 `select()` 中懒计算）和 **3 个误差质量特征**（在 `bind()` 中从 `tok_conf` 时序计算），共 **17 维**：

**图拓扑特征（3 维，来自距离矩阵 D）**

| 特征 | 计算方式 | 质量方向 |
|------|---------|---------|
| `local_cc_r` | 局部聚类系数 = 邻居间互为邻居的比例：`diag(A·A) / (d·(d−1))` | 越高越好 |
| `norm_degree_r` | 归一化度 = `adj.sum(axis=1) / (n-1)` | 越高越好 |
| `cluster_size_r` | DBSCAN 簇大小 / n；噪声点 = 1/n | 越高越好 |

**误差质量与末尾稳定性特征（3 维，来自 tok_conf 时序）**

| 特征 | 计算方式 | 质量方向 |
|------|---------|---------|
| `instability_mass_r` | `mean(arr > μ + 0.5σ)` — 高置信度波动 token 占比 | 越低越好 |
| `tail_variance_r` | `var(arr[-max(1,T//10):])` — 末尾 10% token 方差 | 越低越好 |
| `event_pre_post_delta_r` | `mean(event 前 2 slices) − mean(event 后所有 token)` — 反思后置信度恢复幅度 | 越高越好 |

架构亮点：图拓扑 3 维在 `select(D, ...)` 时**懒计算并缓存**（每个问题组只算一次），避免 `bind()` 阶段无 D 可用的问题；训练脚本 `train_extreme10_selectors.py` 在 worker 内部用 `DistanceEngine(DistanceSpec("ja", num_threads=1)).dense_matrix(views)` 计算 D 并提前提取 `graph_raw`，通过 `payload["graph_raw"]` 传递到训练主进程（Option C 架构）。

消融顺序：① `graph-degree` 零训练验证图信号 → ② Extreme9+图（14维）→ ③ Extreme9+误差质量（14维）→ ④ 完整 Extreme10（17维）。
成功判据：Extreme10 在所有数据集上 Hit@1 ≥ Extreme9、SelAcc@10% ≥ Extreme9，无任何数据集回退。
训练脚本：`python scripts/train_extreme10_selectors.py`。产物：`models/ml_selectors/extreme10_{best,worst}.pkl`。

完整跨数据集精度对比见 [`results/selector_comparison/selector_comparison.md`](results/selector_comparison/selector_comparison.md)。

### CLI 参考

```bash
# 分析
python3 -m nad.cli analyze \
  --cache-root <cache_path> \
  --distance ja --selectors all \
  --distance-threads 16 \
  --out result.json

# 精度评估
python3 -m nad.cli accuracy \
  --selection result.json \
  --cache-root <cache_path> \
  --out accuracy.json

# 跨任务排名
python3 scripts/rank_selectors.py \
  --results-dir ./result/all_model_TIMESTAMP \
  --no-bootstrap --csv --json
```

### 常见问题

| 问题 | 解决方法 |
|------|---------|
| 找不到 `meta.json`（回退 n=100） | 确保每个缓存中存在 `meta.json` |
| 距离计算慢 | 对大集合使用 `roaring` 后端 |
| 缺少 Row-CSR Bank | 用 `--row-bank` 重建缓存 |
| DeepConf 指标未找到（ValueError） | 检查 `token_data/` 是否存在；使用与可用键匹配的 `--metric` |
| 端口 5002 已占用 | `bash cookbook/02_visualization/visualization.sh --kill` |
| 端口 5003 已占用 | `bash cookbook/01_cache_browser/cache_browser.sh --kill` |

---

## English

### Workspace Navigation

- Workspace map: `WORKSPACE_LAYOUT.md`
- Documentation index: `docs/README.md`
- Experiment/result index: `results/README.md`
- Submission JSON index: `submission/README.md`
- Script entrypoint index: `scripts/README.md`
- Model artifact index: `models/README.md`

### Current Research Snapshot (2026-04-08)

- `code_v2` is the current promoted coding default; see `docs/CODE_V2_EXHAUSTIVE_20260406.md`
- `science_hybrid_round3` is the current promoted science patch; see `docs/SCIENCE_HYBRID_ROUND3_RESULTS_20260406.md`
- `gpqa_pairwise_round2` remains `NO-PROMOTE`; see `docs/GPQA_PAIRWISE_ROUND2_RESULTS_20260406.md`
- `gpqa_deepsets_round1` completed the first minimal full-group contextual study, but also remains `NO-PROMOTE`; see `docs/GPQA_DEEPSETS_ROUND1_RESULTS_20260407.md`
- The current science research priority is small contextual models / top-slot calibration, not graph-heavy expansion and not a new monotonic recency feature family

### Code reasoning pilot

- `research_summary.md`: hypothesis framing and interpretation guide
- `literature_map.md`: primary literature map with support-strength labels
- `hypotheses_and_rqs.md`: falsifiable research questions and hypotheses
- `experiment_plan.md`: benchmark, evaluation, and validity threats
- `prompt_ablations.md`: standard prompt templates for A/B/C/D/(E)
- `failure_taxonomy.md`: reasoning-trace error taxonomy
- `pilot_benchmark.py`: runnable synthetic benchmark generator
- `sample_tasks.jsonl`: generated sample tasks with gold outputs and traces

Run from the repo root:

```bash
python3 pilot_benchmark.py --num-per-family 3 --seed 0 --out sample_tasks.jsonl --pretty-sample 5
```

Each record includes executable code, an entry call, gold output, gold trace, and controllable semantic-load attributes. This pilot is for **code reasoning / execution**, not code generation.

A framework for analyzing neural network activations via binary CSR caches, selector algorithms, and a cookbook of reproducible experiments. NAD Next processes raw NPZ activation shards into efficient memory-mapped caches (CSR format with Roaring Bitmap indexing), applies a broad selector suite (including ML-based, temporal discount, and trajectory-based selectors) to pick the most representative sample per problem, and evaluates selector accuracy across models and datasets.

---

## Quick Start

```bash
bash cookbook/00_setup/install.sh                          # install dependencies
bash cookbook/00_setup/verify.sh                            # all checks should be green
bash cookbook/01_cache_browser/cache_browser.sh --background  # browse caches at :5003
bash cookbook/02_visualization/visualization.sh \
  MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20250902_025610 \
  --background                                             # explore one cache at :5002
bash cookbook/03_batch_analyze/batch_analyze.sh -y          # analyze all caches
bash cookbook/05_deepconf_analysis/deepconf_analysis.sh standard --compare-all  # deepconf
```

See each cookbook chapter below for full options.

---

## Cookbook

All scripts live in [`cookbook/`](cookbook/). Run from the **repo root**.

### 00 — Setup

Install Python dependencies and verify the environment.

```bash
bash cookbook/00_setup/install.sh    # numpy, pyroaring, flask, plotly, ...
bash cookbook/00_setup/verify.sh     # 3 groups: Python version, packages, MUI_HUB symlink
```

The verification script checks Python 3.9+, all required/optional packages, and that the `MUI_HUB` symlink points to accessible cache storage. Re-run `install.sh` if any check fails.

> [Chapter README](cookbook/00_setup/README.md) for full details.

### 01 — Cache Browser

Web UI to browse all available caches at **port 5003**. Shows dataset name, sample count, model accuracy, temperature, and build date for every cache under `MUI_HUB/cache/`.

```bash
bash cookbook/01_cache_browser/cache_browser.sh --background   # start
bash cookbook/01_cache_browser/cache_browser.sh --kill          # stop
```

Also exposes a JSON API at `/api/caches` for scripting.

> [Chapter README](cookbook/01_cache_browser/README.md) for all options and API docs.

### 02 — Visualization

Interactive single-cache explorer at **port 5002** (Flask + Plotly). Browse neuron activation patterns per problem, inspect token-level entropy/confidence, view selector performance, and decode token IDs.

```bash
bash cookbook/02_visualization/visualization.sh \
  MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20250902_025610 \
  --background
bash cookbook/02_visualization/visualization.sh --kill
```

Always pass cache paths relative to the repo root via the `MUI_HUB` symlink.

> [Chapter README](cookbook/02_visualization/README.md) for tokenizer setup and all options.

### 03 — Batch Analysis

Parallel analysis across all caches under `MUI_HUB/cache/`. Produces per-selector accuracy results for every model/dataset combination.

```bash
bash cookbook/03_batch_analyze/batch_analyze.sh --dry-run       # preview task list
bash cookbook/03_batch_analyze/batch_analyze.sh --limit 3 -y    # quick test (3 caches)
bash cookbook/03_batch_analyze/batch_analyze.sh -y              # full run
```

Results land in `./result/all_model_TIMESTAMP/`. Pass that directory to `scripts/rank_selectors.py` for cross-task ranking.

> [Chapter README](cookbook/03_batch_analyze/README.md) for modes, parallelism, and output structure.

### 04 — Position Ablation

Ablation across 6 position windows (0-1, 0-2, 0-8, 0-32, 0-128, 0-512) to measure how selector accuracy changes as more tokens are included.

```bash
bash cookbook/04_position_ablation/position_ablation.sh --quick   # 3 windows: 0-1, 0-8, 0-128
bash cookbook/04_position_ablation/position_ablation.sh           # all 6 windows
bash cookbook/04_position_ablation/position_ablation.sh --datasets aime24,aime25  # subset
```

> [Chapter README](cookbook/04_position_ablation/README.md) for all options and output structure.

### 05 — DeepConf Analysis

Token-confidence selector using `tok_conf` and `tok_neg_entropy` metrics from `token_data/`. Four modes: quick (1 config), standard (4 variants), full (standard + position windows), custom.

```bash
bash cookbook/05_deepconf_analysis/deepconf_analysis.sh quick
bash cookbook/05_deepconf_analysis/deepconf_analysis.sh standard --compare-all
bash cookbook/05_deepconf_analysis/deepconf_analysis.sh full
bash cookbook/05_deepconf_analysis/deepconf_analysis.sh custom \
  --metric tok_neg_entropy --reduction mean --group-size 10
```

> [Chapter README](cookbook/05_deepconf_analysis/README.md) for metric semantics and all options.

### CoT Viewer — Chain-of-Thought Browser

Lightweight web UI at **port 5002** for reading decoded reasoning chains and inspecting per-token metrics (confidence, entropy, Gini, self-certainty, log-probability). Browse all datasets, select a problem and run, read the full chain-of-thought, and click on 32-token slices to see token-level details.

```bash
/home/jovyan/work/NAD_Next/.venv/bin/python /home/jovyan/work/NAD_Next/cot_viewer/app.py
```

No arguments needed — auto-discovers all datasets under `MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/`.

> [Full documentation](cot_viewer/README.md) for API endpoints and architecture details.

---

## Architecture

### Project Layout

```
NAD_Next/
  nad/                         # Core Python package
    api.py                     # High-level API: open_cache(), load_correctness_map()
    cli/                       # CLI: python3 -m nad.cli {analyze, accuracy, cache-build, cache-build-fast}
    core/
      adapters/                # NPZ shard -> CSR conversion (shard_reader, batch_processor)
      distance/                # Jaccard distance engine (roaring / numpy, auto-switch at 4096)
      schema/                  # Cache manifest (v4.0+)
      selectors/               # selector algorithms (base, ensemble, ML, temporal, trajectory) + plugin loader
      storage/                 # Binary cache I/O (mmap)
      views/                   # CacheReader (lazy mmap access)
    io/                        # NadNextLoader (256 MB LRU), index, viz catalog
    ops/                       # Accuracy scoring, selector ranking (RNS / Copeland), uniques
    pipeline/                  # Analysis orchestration, 2-stage Map-Reduce cache builder
  cookbook/                     # 6 experiment chapters (00-05)
  scripts/                     # rank_selectors.py, plot_*.py
  plugins/                     # kink_selector.py (reference custom selector)
  tools/                       # Ground truth generation, cache browser server
  cot_viewer/                  # Chain-of-thought browser (Flask, port 5002)
  minimal_visualization_next/  # Flask + Plotly visualization server
```

### Data Pipeline

```
NPZ Shards --> cache-build-fast --> Binary Cache (CSR + mmap)
                                         |
                                    nad.cli analyze --> Selection JSON
                                         |
                                    nad.cli accuracy --> Accuracy Report
                                         |
                                    rank_selectors.py --> Cross-task Ranking
```

**Stage 1 — Build**: Raw NPZ activation shards are converted into a binary cache via a two-stage Map-Reduce process (`cache-build-fast`). The output is a set of memory-mapped CSR files with sorted indices and optional token-level statistics.

**Stage 2 — Analyze**: `nad.cli analyze` reads a cache, groups samples by problem, computes pairwise Jaccard distances, and runs each selector algorithm to pick one representative per problem.

**Stage 3 — Accuracy**: `nad.cli accuracy` evaluates each selector's pick against ground truth from `evaluation_report_compact.json`.

**Stage 4 — Rank**: `scripts/rank_selectors.py` aggregates accuracy across multiple tasks and produces cross-task rankings using RNS, regret, and Copeland scoring with optional Bootstrap CI.

### Cache Format

Caches are stored under `MUI_HUB/cache/{model}/{dataset}/cache_neuron_output_1_*`.

```
cache_neuron_output_1_*/
  meta.json                      # Sample -> problem mapping (auto-parsed by analysis)
  manifest.json                  # Schema version + checksums
  evaluation_report_compact.json # Ground truth, 99% compressed (0.8 MB vs 82 MB)
  base/                          # CSR neuron activations: row_ptr, keys, w_max, w_sum
  index/                         # Sorted indices for fast lookups
  token_data/                    # Per-token LM stats: tok_conf, tok_neg_entropy
  rows/                          # Row-CSR Bank for position-window queries
```

**Two independent data dimensions share the same row structure:**
- **Neuron data** (`base/`, `rows/`): which neurons fired at each token position
- **Token data** (`token_data/`): LM output distribution statistics per token

Cache type `cache_neuron_output_1_*` = 1 token/row. Type `cache_neuron_output_2_*` = ~32 tokens/row.

---

## Selectors

Thirty-five built-in selector algorithms pick the most representative sample from each problem group.

**Base selectors**

| Selector | Type | Description |
|----------|------|-------------|
| `min-activation` | [S] | Fewest total neuron activations |
| `max-activation` | [S] | Most total neuron activations |
| `min-confidence` | [S] | Lowest mean tok_conf (most confident) |
| `medoid` | [P] | Geometric centre of the group (Jaccard distance) |
| `knn-medoid` | [P] | Medoid restricted to K nearest neighbours |
| `dbscan-medoid` | [P] | Medoid of the densest DBSCAN cluster |
| `consensus-min` | [P] | Fewest activations among consensus candidates |
| `consensus-max` | [P] | Most activations among consensus candidates |
| `deepconf` | [S] | Token-confidence sliding window (requires `token_data/`) |
| `con64@` / `avg64@` | [O] | Consensus / average oracle baselines (full sequence only) |

Type: `[S]` = independent per-run score, no distance matrix needed; `[P]` = requires N×N Jaccard matrix; `[O]` = oracle baseline using all ground-truth labels.

**Group Ensemble selectors** (random groups, elimination rounds)

| Selector | Description |
|----------|-------------|
| `ensemble-medoid` | Random groups of 8 → medoid per group → medoid of winners |
| `ensemble-deepconf` | Random groups of 8 → best DeepConf per group → best of winners |

**Tournament selectors** (pairwise comparison + softmax)

| Selector | Description |
|----------|-------------|
| `tournament-copeland` | Copeland voting: for each pair (i,j) count how many third parties are closer to i vs j → softmax |
| `tournament-deepconf` | Pairwise DeepConf quality comparison → win count → softmax |

**Two-stage selectors** (group top-k → final round)

| Selector | Description |
|----------|-------------|
| `twostage-medoid` | Groups of 16 → top-4 by mean distance → 16 finalists → medoid |
| `twostage-tournament` | Groups of 16 → top-4 by Copeland → 16 finalists → Copeland + softmax |

For all softmax-based selectors the default temperature is 0.2 and seed is 42.

**ML selectors** (require pre-trained models — run `python scripts/train_ml_selectors.py`)

Trained on 31,040 labelled (problem, run) pairs from 6 datasets. Features: 12-dim group-normalised vector (mean_dist, knn3, length, deepconf, copeland × z-score+rank; log_n, log_length).

| Selector | Model | Description |
|----------|-------|-------------|
| `linear-probe` | Ridge regression | Predicts is_correct score, selects argmax |
| `logistic` | Logistic regression | Predicts P(correct), selects argmax |
| `isotonic-medoid` | Isotonic regression (1 feature) | Calibrates medoid rank → P(correct) |
| `isotonic-deepconf` | Isotonic regression (1 feature) | Calibrates DeepConf rank → P(correct) |

Type `[ML]`. Models stored in `models/ml_selectors/`, lazy-loaded at inference.
Leave-one-out CV: `logistic` 69.7%, `linear-probe` 69.8% — on par with `knn-medoid` on held-out data.
Single-feature ablation shows `dc_z` (DeepConf quality) at 70.9% as the strongest individual feature, outperforming the full 12-feature model.

**Temporal discount slice selector** (slice-based weighting, no distance matrix needed)

| Selector | Type | Description |
|----------|------|-------------|
| `temporal-slice` | [S] | Splits tokens into 32-token slices, weights later slices with γ^(2k), picks highest weighted quality |

Type `[S]`. Default params from grid search: `tok_neg_entropy`, γ=0.7, T=0.1. Best mean accuracy: 60.3%.
Params saved to `models/ml_selectors/temporal_best_params.json`.

**Trajectory analysis selectors** (neuron activation trajectory over token positions; requires `rows/` bank v4.1+)

| Selector | Type | Description |
|----------|------|-------------|
| `trajectory` | [S] | Trajectory structure score: α·continuity - β·novelty + γ·late_convergence + δ·bounded_reflection |
| `layer-stratified` | [S] | Layer-wise activation distribution: α·deep_frac_z + β·layer_entropy - γ·layer_gini |
| `trajectory-fusion` | [ML] | 22-D fusion of trajectory (10-D) + existing (12-D) features with LogisticRegression |

Type `[S]` / `[ML]`. Trajectory features use per-slice (32-token) activation key sets from the `rows/` bank to compute inter-slice Jaccard similarities.
Layer features decode layer info from neuron key encoding (`layer<<16|neuron_id`).
Results: `layer-stratified` 69.4%, `trajectory` 58.7%, `trajectory-fusion` 68.3%.
Single-feature ablation found `reflection_count_r` (reflection count rank) at 71.1% surpassing `dc_z` (70.9%) as the new best single feature.
Training script: `python scripts/train_trajectory_selectors.py`.
Latest artifact snapshot (UTC): evaluation reports were generated on `2026-03-30` in `results/trajectory_experiments/accuracy_summary_20260330_112435.json`, `results/trajectory_experiments/trajectory_20260330_112435.json`, and `results/trajectory_experiments/layer_stratified_20260330_112435.json`; the 22-D trajectory-fusion training stats were refreshed on `2026-03-31 01:56:52` in `models/ml_selectors/trajectory_stats.json` (`31,040` labelled pairs, `18,873` correct, `22` features, `6` datasets).
The `2026-04-02 17:13:06` UTC reflection-dynamics follow-up was written to `results/reflection_dynamics/summary.md` and `results/reflection_dynamics/threshold_sweep_summary.json`: lowering the reflection-event threshold from `0.30` to `0.20` improves the single-feature LOO mean of `reflection_count_r` from `71.1%` to `71.7%`. The same analysis also shows that reflection correlates positively with average `gini`, negatively with average `entropy`, and that reflection-event slices exhibit smaller first/second-order confidence changes overall.
New pooled subset selectors `extreme8-best` / `extreme8-worst` / `extreme8-mixed` were added: training uses only the three strong features `dc_z`, `dc_r`, and `reflection_count_r`, keeps problems with empirical accuracy in `[10%, 90%]`, and samples `256` mixed 8-tuples per eligible problem. The `2026-04-02 17:56:57` UTC blind 64-run evaluation then samples `512` random 8-tuples per problem without looking at run correctness during tuple construction, yielding a 6-dataset mean of `72.5%` for `best-only` / `best+worst` / `worst-avoid`, while `worst-only` reaches `56.1%` error-hit rate. Artifacts: `models/ml_selectors/extreme8_best.pkl`, `models/ml_selectors/extreme8_worst.pkl`, `models/ml_selectors/extreme8_stats.json`, `results/extreme8_experiments/20260402_112323/summary_20260402_112323.json`. Note that this Extreme8 run still used `reflection_threshold=0.30`; the improved `0.20` threshold was discovered afterwards by the follow-up sweep and has not yet been propagated into this model snapshot.

**Extreme9 local-confidence expansion selectors** (11-dim, requires training, `extreme9_impl.py`)

`extreme9-best` / `extreme9-worst` / `extreme9-mixed` extend Extreme8's 3-dim core (`dc_z`, `dc_r`, `reflection_count_r`) with 8 local `tok_conf` aggregation features derived from the DeepConf paper, totalling **11 dimensions**:

| Feature | Formula | Quality direction |
|---------|---------|-------------------|
| `tail_2k_r` | Mean tok_conf of last min(2000,T) tokens, rank | Lower = better |
| `tail_q10_r` | Mean tok_conf of last 10% tokens, rank | Lower = better |
| `lgc_512_r` | Least-grouped-confidence, window=512, rank | Lower = better |
| `lgc_2k_r` | Least-grouped-confidence, window=2000, rank | Lower = better |
| `bottom_q10_r` | 10th-percentile tok_conf, rank | Lower = better |
| `head_tail_gap_r` | mean(head 10%) − mean(tail 10%), rank (positive = tail more confident) | Higher = better |
| `last_event_tail_conf_r` | Mean tok_conf after last reflection event slice, rank | Lower = better |
| `event_nonevent_gap_r` | Event-slice mean − non-event-slice mean tok_conf, rank | Lower = better |

Training script: `python scripts/train_extreme9_selectors.py`. Artifacts: `models/ml_selectors/extreme9_{best,worst}.pkl`.
Zero-training baseline `local-conf-tail` selects the run with minimum `tail_2k` to validate the local confidence signal direction independently.

**Graph topology zero-training baseline** (`graph-degree`, `graph_topo_impl.py`)

`graph-degree` builds a 64-run similarity graph from the Jaccard distance matrix using an adaptive eps (30th percentile of off-diagonal distances) and selects the run with the highest **normalised degree** (`norm_degree = degree / (n-1)`). High degree means activation-similar to many other runs, which correlates with being in the consensus correct cluster. Used to validate graph topology signal strength before training Extreme10, benchmarked against `dc_r` and `dbscan-medoid`.

**Extreme10 graph topology + error-mass expansion selectors** (17-dim, requires training, `extreme10_impl.py`)

`extreme10-best` / `extreme10-worst` / `extreme10-mixed` extend Extreme9's 11 dimensions with **3 graph topology features** (lazy-computed from D in `select()`) and **3 error-mass / late-stage stability features** (computed from `tok_conf` time-series in `bind()`), totalling **17 dimensions**:

**Graph topology features (3 dims, from distance matrix D)**

| Feature | Formula | Quality direction |
|---------|---------|-------------------|
| `local_cc_r` | Local clustering coefficient = `diag(A·A) / (d·(d−1))` — fraction of neighbours that are mutual neighbours | Higher = better |
| `norm_degree_r` | Normalised degree = `adj.sum(axis=1) / (n-1)` | Higher = better |
| `cluster_size_r` | DBSCAN cluster size / n; noise points receive 1/n | Higher = better |

**Error-mass + late-stage stability features (3 dims, from tok_conf time-series)**

| Feature | Formula | Quality direction |
|---------|---------|-------------------|
| `instability_mass_r` | `mean(arr > μ + 0.5σ)` — fraction of high-volatility tokens | Lower = better |
| `tail_variance_r` | `var(arr[-max(1,T//10):])` — variance of the last 10% of tokens | Lower = better |
| `event_pre_post_delta_r` | `mean(2 slices before last event) − mean(after last event)` — confidence recovery after reflection | Higher = better |

Architecture highlight: the 3 graph topology dimensions are **lazily computed and cached per group** inside `select(D, ...)` (D is not available at `bind()` time). The training script `train_extreme10_selectors.py` precomputes D inside each worker process using `DistanceEngine(DistanceSpec("ja", num_threads=1)).dense_matrix(views)` and extracts `graph_raw` upfront, passing it as `payload["graph_raw"]` to the training master process (Option C architecture — no extra inter-process round trips).

Ablation order: ① `graph-degree` zero-training validates graph signal → ② Extreme9+graph (14-dim) → ③ Extreme9+error-mass (14-dim) → ④ Full Extreme10 (17-dim).
Success criterion: Extreme10 ≥ Extreme9 on Hit@1 and SelAcc@10% across all datasets with no regression on any individual dataset.
Training script: `python scripts/train_extreme10_selectors.py`. Artifacts: `models/ml_selectors/extreme10_{best,worst}.pkl`.

Full cross-dataset accuracy results are in [`results/selector_comparison/selector_comparison.md`](results/selector_comparison/selector_comparison.md).

<details>
<summary><strong>Custom Selectors</strong></summary>

Inherit from `Selector`, implement `bind(context)` and `select(D, run_stats) -> int`:

```python
from nad.core.selectors.base import Selector, SelectorContext
from nad.ops.uniques import extract_tokenwise_counts

class MySelector(Selector):
    def bind(self, context: SelectorContext):
        self._context = context

    def select(self, D, run_stats) -> int:
        # D: pairwise distance matrix
        # self._context.cache: CacheReader
        return selected_index
```

Load with `file:path/to/selector.py:ClassName`. Always filter `inf` values before `argmax`/`argmin` (occurs when a denominator is zero). See [`plugins/kink_selector.py`](plugins/kink_selector.py) for a complete example.

</details>

---

## CLI Reference

### analyze — Run selector algorithms on a cache

```bash
python3 -m nad.cli analyze \
  --cache-root MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20250902_025610 \
  --distance ja --selectors all \
  --group-topk-policy legacy-min \
  --distance-threads 16 \
  --out result.json
```

### analyze — Position window mode

```bash
python3 -m nad.cli analyze \
  --cache-root <cache_path> \
  --pos-window 0-8 --pos-size 32 \
  --out window_0-8_result.json
```

### accuracy — Evaluate selector picks against ground truth

```bash
python3 -m nad.cli accuracy \
  --selection result.json \
  --cache-root <cache_path> \
  --out accuracy.json
```

### rank — Cross-task selector ranking

```bash
python3 scripts/rank_selectors.py \
  --results-dir ./result/all_model_TIMESTAMP \
  --no-bootstrap --csv --json
```

### Custom selector plugin

```bash
python3 -m nad.cli analyze \
  --cache-root <cache_path> \
  --selectors 'all,file:./plugins/kink_selector.py:KinkSelector'
```

### Jaccard Backend

| Backend | When faster |
|---------|-------------|
| `roaring` | Set size >= 4096 elements (default) |
| `numpy` | Small sets |
| `auto` | Switches at the 4096 threshold |

### DeepConf Metric Semantics

| Metric | Formula | Quality direction |
|--------|---------|-------------------|
| `tok_conf` | -mean(log p_topk) | lower -> more confident -> `quality = -tok_conf` |
| `tok_neg_entropy` | sum(p log p) <= 0 | closer to 0 -> more certain -> `quality = tok_neg_entropy` (identity) |

Sliding window uses strict trailing windows with no padding. Sequences shorter than `window_size` return their global mean.

---

## Configuration

### Dependencies

```bash
pip install .             # recommended: installs from pyproject.toml
# or install core deps directly:
pip install numpy>=1.20.0 pyroaring>=0.4.5
```

### Environment Variables

```bash
export NAD_JA_BACKEND=roaring   # Jaccard backend: roaring / numpy / auto
export OMP_NUM_THREADS=1        # prevent NumPy thread oversubscription
export MKL_NUM_THREADS=1
```

### Thread Limits

This machine has **16 cores**. All scripts must satisfy `THREADS x PARALLEL_JOBS <= 16`.

| PARALLEL_JOBS | THREADS |
|---------------|---------|
| 1 | 16 |
| 4 | 4 |
| 8 | 2 |

Current cookbook defaults:
- `03_batch_analyze`: THREADS=4, PARALLEL_JOBS=4
- `04_position_ablation`: THREADS=4, PARALLEL_JOBS=4
- `05_deepconf_analysis`: THREADS=16, PARALLEL_JOBS=1

### nad_config.json

`nad_config.json` at repo root specifies `model_search_dirs` for tokenizer lookup (used by the visualization server).

```json
{
  "model_search_dirs": [
    "/home/jovyan/public-ro/model"
  ]
}
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `meta.json` not found (fallback n=100) | Ensure `meta.json` is present in each cache |
| Slow distance computation | Use `roaring` backend for large sets |
| Missing Row-CSR Bank | Rebuild cache with `--row-bank` |
| DeepConf metric not found (ValueError) | Check `token_data/` exists; use `--metric` matching available keys |
| Port 5002 already in use | `bash cookbook/02_visualization/visualization.sh --kill` |
| Port 5003 already in use | `bash cookbook/01_cache_browser/cache_browser.sh --kill` |

---

## License

Internal use only.
