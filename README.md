# NAD Next — Neuron Activation Distribution

[English](#english) | [中文](#中文)

---

## 中文

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

内置 21 种选择器算法，从每个题目组中挑选最具代表性的样本：

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

A framework for analyzing neural network activations via binary CSR caches, selector algorithms, and a cookbook of reproducible experiments. NAD Next processes raw NPZ activation shards into efficient memory-mapped caches (CSR format with Roaring Bitmap indexing), applies 24 selection algorithms (including ML-based, temporal discount, and trajectory-based selectors) to pick the most representative sample per problem, and evaluates selector accuracy across models and datasets.

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
      selectors/               # 24 selector algorithms (base, ensemble, ML, temporal, trajectory) + plugin loader
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

Twenty-one built-in selector algorithms pick the most representative sample from each problem group.

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
