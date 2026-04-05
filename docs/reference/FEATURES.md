# NAD Next — Comprehensive Feature & Module Breakdown

This document provides a step-by-step breakdown of **everything implemented** in this repository, organised by topic area. It focuses on the full scope beyond the three core concepts you may already know (chain of thought, neuron activation, confidence/entropy).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Data Pipeline: NPZ → Binary Cache](#2-data-pipeline-npz--binary-cache)
3. [Cache Format & Storage Layout](#3-cache-format--storage-layout)
4. [Distance Computation](#4-distance-computation)
5. [Sample Selection Algorithms (Core Selectors + Extended Families)](#5-sample-selection-algorithms-core-selectors--extended-families)
6. [Plugin System: Custom Selectors](#6-plugin-system-custom-selectors)
7. [Evaluation & Accuracy Scoring](#7-evaluation--accuracy-scoring)
8. [Multi-Task Selector Ranking](#8-multi-task-selector-ranking)
9. [Position-Window Ablation](#9-position-window-ablation)
10. [CLI Interface](#10-cli-interface)
11. [Visualization & Web Servers](#11-visualization--web-servers)
12. [Cookbook: Reproducible Experiment Workflows](#12-cookbook-reproducible-experiment-workflows)
13. [Scripts & Plotting Utilities](#13-scripts--plotting-utilities)
14. [Tools & Standalone Utilities](#14-tools--standalone-utilities)
15. [Public Python API](#15-public-python-api)
16. [Unique-Neuron-Count Operator](#16-unique-neuron-count-operator)
17. [Kink Detection (Breakpoint Analysis)](#17-kink-detection-breakpoint-analysis)
18. [Configuration & Environment](#18-configuration--environment)
19. [Feature Summary Table](#19-feature-summary-table)

---

## 1. Project Overview

NAD Next is a production framework for **post-hoc analysis of language-model neural activations**. It does **not** train or fine-tune models; it processes activation recordings that are captured during inference.

The four-stage pipeline is:

```
NPZ Shards  ──▶  Binary Cache  ──▶  Selection JSON  ──▶  Accuracy Report  ──▶  Cross-task Ranking
            build               analyze              accuracy             rank_selectors.py
```

**Key stats (codebase):**
- ~51 Python files, ~4,600 lines
- Python 3.9+, NumPy, PyRoaring, Flask, Plotly
- Includes ML selector training/evaluation scripts and cached experiment artifacts

---

## 2. Data Pipeline: NPZ → Binary Cache

**Module:** `nad/pipeline/build_cache_fast.py`  
**Command:** `python3 -m nad.cli cache-build-fast`

### What it does

Raw inference results are stored as NPZ shards (one per token slice). The cache builder converts them into a compact binary format (CSR + mmap) that supports fast random access.

### Architecture: Two-stage Map-Reduce

| Stage | Workers | Input | Output |
|-------|---------|-------|--------|
| **Map** | 32–60 parallel workers (configurable) | 1 NPZ shard each | `partial_{id}.npz` temp files |
| **Reduce** | Main process | All partial NPZ files | Final binary cache |

### Map stage (per-shard worker)

- Reads one NPZ shard via `read_shard_grouped_by_sample()`
- Groups activations by `sample_id` (not `slice_id`)
- Extracts token-level metadata via `read_shard_tokens_by_sample()`
- Optionally builds per-row (per-token-position) CSR if `--row-bank` is given
- Writes `partial_{id}.npz` (never large returns across process boundary)

### Reduce stage (K-way merge)

- K-way merge of all partial NPZs
- Aggregates duplicate neuron keys via `np.add.reduceat()` (fast vectorised dedup)
- Builds sorted index arrays (`perm_max`, `perm_sum`) and cumulative prefix sums
- Writes binary cache (`base/`, `index/`, `token_data/`, optionally `rows/`)
- Generates `manifest.json` with SHA-256 integrity checksums

### Engineering details

| Technique | Purpose |
|-----------|---------|
| `spawn` multiprocessing context | Avoids forking deadlocks |
| `OMP_NUM_THREADS=1` per worker | Prevents BLAS thread oversubscription |
| Atomic rename (tmp → final) | No partial-write corruption |
| Memory-mapped output | Workers write directly to disk, no large IPC |
| `np.add.reduceat()` dedup | Fast unique-key aggregation without Python loops |

---

## 3. Cache Format & Storage Layout

**Modules:** `nad/core/storage/binary_io.py`, `nad/core/views/reader.py`, `nad/core/schema/manifest.py`

Every cache lives under `MUI_HUB/cache/{model}/{dataset}/cache_neuron_output_1_*`.

```
cache_root/
├── meta.json                        # sample_id → problem_id mapping (parsed by analysis)
├── manifest.json                    # schema v4.0, SHA-256 checksums
├── evaluation_report_compact.json   # ground truth, 99% compressed vs raw
│
├── base/                            # CSR neuron activations (per sample)
│   ├── row_ptr.int64               # sample pointers  [n_samples+1]
│   ├── keys.uint32                 # packed neuron IDs: layer<<16 | neuron
│   ├── w_max.float16               # max-aggregation weights
│   └── w_sum.float16               # sum-aggregation weights
│
├── index/                           # Sort indices for fast top-k queries
│   ├── perm_max.int32              # descending sort permutation (max agg)
│   ├── perm_sum.int32              # descending sort permutation (sum agg)
│   ├── prefix_max.float16          # cumulative weight prefix (max)
│   └── prefix_sum.float16          # cumulative weight prefix (sum)
│
├── token_data/                      # Per-token LM output statistics (v4.0+)
│   ├── token_row_ptr.int64         # token pointers [n_samples+1]
│   ├── token_ids.int32             # token IDs
│   ├── tok_conf.float32            # confidence: −mean(log p_topk)
│   ├── tok_neg_entropy.float32     # negative entropy: Σ p log p  (≤ 0)
│   ├── tok_gini.float32            # Gini coefficient
│   ├── tok_selfcert.float32        # self-certainty: KL(p ∥ Uniform)
│   └── tok_logprob.float32         # log probability
│
├── rows/                            # Row-CSR Bank for position-window queries (v4.1+)
│   ├── sample_row_ptr.int64        # maps run_id → [row_start, row_end)
│   ├── row_ptr.int64               # per-row CSR pointers
│   ├── keys.uint32                 # row-level neuron keys
│   ├── w_max.float16               # row-level max weights
│   ├── w_sum.float16               # row-level sum weights
│   ├── slice_ids.int32             # per-row slice ID (token position)
│   └── token_row_ptr.int64         # global token cumsum
│
└── window_cache/                    # Pre-aggregated position-window caches (lazy, v4.1)
    └── pos_0_8/
        ├── row_ptr.int64
        ├── keys.uint32
        ├── w_max.float16
        └── w_sum.float16
```

### CacheReader — lazy mmap access

`CacheReader` memory-maps every file; nothing is read into RAM until accessed. It exposes:

- `get_run_view(run_id, ViewSpec)` → `RunView(keys, weights)` — neuron key/weight arrays for one sample
- `get_token_view(run_id)` → `TokenView(token_ids, tok_conf, tok_neg_entropy, …)` — token metrics

### ViewSpec — aggregation parameters

| Field | Options | Description |
|-------|---------|-------------|
| `agg` | `"max"` / `"sum"` | Which weight column to use |
| `cut` | `"topk:8192"` / `"mass:0.98"` | How many neurons to include |
| `order` | `"by_key"` / `"by_weight"` | Sort order of the returned array |

### Manifest v4.0

Stores dataset ID, model ID, run count, aggregations, dtypes, total keys, token metadata flags, and SHA-256 hashes of every binary file for integrity verification.

---

## 4. Distance Computation

**Module:** `nad/core/distance/engine.py`

### Algorithms

| Name | Flag | Formula |
|------|------|---------|
| **Jaccard** | `ja` | `1 − |A∩B| / |A∪B|` |
| **Weighted Jaccard** | `wj` | `1 − Σᵢ min(wᴬᵢ, wᴮᵢ) / (Σ wᴬ + Σ wᴮ − Σᵢ min(wᴬᵢ, wᴮᵢ))` where wᴬᵢ / wᴮᵢ are the weights of key i in set A / B |

### Backends

| Backend | When fastest | How selected |
|---------|-------------|-------------|
| **NumPy** | Small sets (<4 096 elements) | `np.intersect1d(assume_unique=True)` |
| **Roaring Bitmap** | Large sets (≥4 096 elements) | PyRoaring C++ library, 3–5× faster |
| **Auto** | Switches at the 4 096 threshold | Default; overrideable via `NAD_JA_BACKEND` |

### `dense_matrix(views)` — implementation

1. Pre-fetch all key arrays; ensure contiguous C layout
2. Detect backend per-row based on set sizes (auto mode)
3. Convert to bitmaps if using Roaring
4. Generate upper-triangle pairs: `(i, j)` where `i < j`
5. Parallel worker computes one distance per pair (`ThreadPoolExecutor`)
6. Symmetrically fill matrix: `D[i,j] = D[j,i]`

Parallelisation is skipped for fewer than 256 pairs (overhead > savings).

---

## 5. Sample Selection Algorithms (Core Selectors + Extended Families)

**Module:** `nad/core/selectors/impl.py`

This section focuses on the 10 core selectors implemented in `nad/core/selectors/impl.py`. The full built-in selector surface also includes ensemble, tournament, two-stage, ML, temporal, trajectory, and oracle-style baselines documented in `README.md` and `results/selector_comparison/selector_comparison.md`.

Latest experiment snapshot (UTC):
- Evaluation reports generated on `2026-03-30`: `results/trajectory_experiments/accuracy_summary_20260330_112435.json`, `results/trajectory_experiments/trajectory_20260330_112435.json`, `results/trajectory_experiments/layer_stratified_20260330_112435.json`
- 22-D trajectory-fusion training stats refreshed on `2026-03-31 01:56:52`: `models/ml_selectors/trajectory_stats.json` (`31,040` labelled pairs, `18,873` correct, `22` features across `6` datasets)
- Reflection-dynamics follow-up generated on `2026-04-02 17:13:06`: `results/reflection_dynamics/summary.md`, `results/reflection_dynamics/threshold_sweep_summary.json` (best single-feature reflection threshold `0.20`, LOO `71.7%`, vs `71.1%` at `0.30`)
- Extreme8 pooled-selector artifacts generated on `2026-04-02 17:56:57`: `models/ml_selectors/extreme8_best.pkl`, `models/ml_selectors/extreme8_worst.pkl`, `models/ml_selectors/extreme8_stats.json`, `results/extreme8_experiments/20260402_112323/summary_20260402_112323.json` (train-time features only `dc_z`, `dc_r`, `reflection_count_r`; blind `512 × 8` evaluation mean `72.5%` for `best-only` / `best+worst`)

All selectors share a common contract: given a pairwise distance matrix `D` (n × n) and per-run statistics, return a **group-local index** (0 to n−1). The pipeline maps that index to the global `run_id`.

### Selector catalogue

| # | Name | Key idea |
|---|------|---------|
| 1 | `min-activation` | Fewest total neuron activations — minimal activity baseline |
| 2 | `max-activation` | Most total neuron activations — maximal activity baseline |
| 3 | `medoid` | `argmin mean(D[i,:])` — geometric centre of the group |
| 4 | `knn-medoid` | Medoid restricted to K nearest neighbours (default k=3) using similarity matrix |
| 5 | `dbscan-medoid` | Medoid of the largest DBSCAN cluster; auto-eps = 30th-percentile of off-diagonal distances |
| 6 | `consensus-min` | Candidate set = {KNN-medoid, Medoid, DBSCAN-medoid}; pick the one with **minimum** activation length |
| 7 | `consensus-max` | Same candidates; pick the one with **maximum** activation length |
| 8 | `deepconf` | Token-confidence metric (tok_conf / tok_neg_entropy / tok_selfcert), `mean` or `min_group` reduction |
| 9 | `avg64@` | Virtual baseline: average accuracy of top-64 runs (full-sequence only) |
| 10 | `con64@` | Virtual baseline: consensus of top-64 runs (full-sequence only) |

### Algorithm details

**Medoid**
```
selected = argmin_i { mean_j D[i, j] }
```

**KNN-Medoid**
```
S = 1 − D                          # similarity matrix
score[i] = mean of top-k S[i, :]   # exclude self
selected = argmax(score)
```

**DBSCAN-Medoid**
```
eps_auto = percentile(off-diagonal distances, 30)
labels   = DBSCAN(D, eps=eps_auto, min_samples=3)
cluster  = largest cluster with label ≥ 0
medoid   = argmin mean distance within cluster
```
Falls back to standard Medoid if no cluster is found.

**Consensus (min / max)**
```
candidates = { knn-medoid result, medoid result, dbscan-medoid result }
selected   = candidate with min (or max) activation length
tie-break  = smallest index (deterministic)
```

**DeepConf**
```
quality[run] = aggregate( token_metric[run] )

metric → quality direction:
  tok_conf        → quality = −tok_conf         (tok_conf = −mean(log p_topk); lower value means higher probability mass → more confident → higher quality)
  tok_neg_entropy → quality =  tok_neg_entropy   (closer to 0 = more certain = higher quality)
  tok_selfcert    → quality =  tok_selfcert       (higher KL from Uniform = more peaked)

reduction:
  mean      → simple mean over all tokens
  min_group → minimum of sliding-window means (window size = group_size)
```

---

## 5a. Extended ML Selector Families: Extreme8 / Extreme9 / Extreme10 + Graph Topology

> **双语说明 | Bilingual note**
> 本节同时提供中文与英文描述，以便中英文读者均可快速定位关键细节。
> This section is provided in both Chinese and English for accessibility.

---

### 5a-1. Shared Architecture (共享架构)

All Extreme selectors follow a unified **tuple-sampling + linear ranking** pattern:

```
bind(context)
  → extract raw per-run scalar features (no D available here)
  → self._raw_values = dict of np.ndarray, one value per run

select(D, run_stats)
  → [if graph features needed] lazy-compute graph_raw from D, cache per group
  → sample N random k-tuples from the n runs
  → for each tuple: build feature matrix (n_sub, dim) and score with model
  → accumulate per-run scores, return argmax(score_best)
```

所有 Extreme 系选择器使用相同的**随机 k-tuple 采样 + 线性排名**框架：`bind()` 阶段提取无需距离矩阵的标量特征；`select(D, ...)` 阶段对随机子集打分并汇总。

**Selector contract** (`base.py`):
- `bind(context: SelectorContext) → None` — called once per problem group; D unavailable
- `select(D: np.ndarray, run_stats: dict) → int` — returns **group-local** index (0 to n-1)

---

### 5a-2. Extreme8 (3-dim) — `extreme8_impl.py`

**Features (3 dimensions)**:

| Dim | Name | Formula | Direction |
|-----|------|---------|-----------|
| 0 | `dc_z` | z-score of per-run DeepConf quality (`−mean(tok_conf)`) within the group | Higher = better |
| 1 | `dc_r` | rank of per-run DeepConf quality, normalised to [0,1] | Higher = better |
| 2 | `reflection_count_r` | rank of reflection event count (inter-slice Jaccard > threshold), normalised to [0,1] | Higher = better |

**Key implementation files:**
- `nad/core/selectors/extreme8_impl.py` — `extract_extreme8_raw_values()`, `build_extreme8_features()`, `LinearRankModel`, `Extreme8{Best,Worst,Mixed}Selector`
- `scripts/train_extreme8_selectors.py` — training with pointwise / band-reward / aggregated_selacc10 objectives

**Performance:** Blind 64-run evaluation (512 tuples/problem) → **72.5%** mean across 6 datasets.
Baseline metrics (Extreme8): Hit@1 ≈ 72.5%, SelAcc@10% ≈ 75.99%.

**中文说明：** Extreme8 仅用三个强特征：DeepConf z-score（`dc_z`）、DeepConf rank（`dc_r`）、反思事件次数 rank（`reflection_count_r`）。其中 `dc_z` 和 `dc_r` 衡量 run 整体的 token 置信度；`reflection_count_r` 衡量推理过程中发生的反思跳转次数，是迄今最强单特征（71.1% LOO 均值）。

---

### 5a-3. Extreme9 (11-dim) — `extreme9_impl.py` + `local_conf_impl.py`

**Feature list (11 dimensions)**:

```
Extreme8 base (3)  +  local tok_conf aggregations (8)
```

| Dim | Name | Source module | Formula | Direction |
|-----|------|--------------|---------|-----------|
| 0–2 | `dc_z`, `dc_r`, `reflection_count_r` | `extreme8_impl` | (see Extreme8) | ↑ better |
| 3 | `tail_2k_r` | `local_conf_impl` | rank of `mean(arr[-min(2000,T):])` | ↑ (lower tok_conf = better) |
| 4 | `tail_q10_r` | `local_conf_impl` | rank of `mean(arr[-T//10:])` | ↑ |
| 5 | `lgc_512_r` | `local_conf_impl` | rank of least sliding-window mean, window=512 | ↑ |
| 6 | `lgc_2k_r` | `local_conf_impl` | rank of least sliding-window mean, window=2000 | ↑ |
| 7 | `bottom_q10_r` | `local_conf_impl` | rank of 10th-percentile tok_conf | ↑ |
| 8 | `head_tail_gap_r` | `local_conf_impl` | rank of `mean(head 10%) − mean(tail 10%)` | ↑ (positive = tail confident) |
| 9 | `last_event_tail_conf_r` | `local_conf_impl` | rank of mean tok_conf after last reflection event | ↑ |
| 10 | `event_nonevent_gap_r` | `local_conf_impl` | rank of `event_mean − nonevent_mean` tok_conf | ↑ |

**Feature builder internals:**
- tok_conf direction: lower value = more confident → apply `_rank01(-arr)` so higher rank = better
- `head_tail_gap`, `event_nonevent_gap`: sign already meaningful; apply `_rank01(arr)`
- `_impute_finite(arr, fill)`: replaces NaN/±inf with fill before ranking
- LGC (Least-Grouped Confidence): `min(sliding-window means)`, mirrors DeepConf paper's operator

**Zero-training baseline:** `local-conf-tail` (`LocalConfTailSelector`) — selects `argmin(tail_2k)` without any trained model. Validates that the local confidence direction is correct before training Extreme9.

**Training script:** `scripts/train_extreme9_selectors.py`
**Model artifacts:** `models/ml_selectors/extreme9_{best,worst}.pkl`

**中文说明：** Extreme9 在 Extreme8 基础上加入 8 个局部 `tok_conf` 聚合特征。核心思路来自 DeepConf 论文：全局均值信号（dc\_r）已经很强，但局部算子（尾部均值、最差滑窗、底部百分位）能捕捉到全局均值遮盖的细节。`last_event_tail_conf_r` 和 `event_nonevent_gap_r` 还利用了 reflection event 的切片边界信息，衡量反思后 token 置信度是否收敛。所有 tok_conf 特征均以"越低越好"方向进行 rank 归一化。

---

### 5a-4. Graph Topology Module — `graph_topo_impl.py`

**Motivation:** Inspired by CodeCircuit (attribution graph topology → correctness prediction). Correct-answer runs tend to form **denser, more connected clusters** in the activation similarity graph built from the 64-run Jaccard distance matrix.

**灵感来源（中文）：** 受 CodeCircuit 研究启发，正确 run 倾向于在激活相似图中形成更密集、连通性更高的簇。图拓扑特征从 64-run Jaccard 距离矩阵 D 中提取，无需额外训练即可作为零训练基线信号。

#### `extract_graph_topo_raw(D, eps=None, min_samples=3) → dict`

```python
# Adaptive eps = 30th percentile of upper-triangular distances
eps = np.quantile(D[np.triu_indices(n, k=1)], 0.30)

adj = (D <= eps).astype(float)   # adjacency matrix (diagonal zeroed)
degree = adj.sum(axis=1)         # raw degree per run

# local_cc: fraction of neighbours that are mutual neighbours
AA = adj @ adj                   # closed walks of length 2
safe_denom = where(d*(d-1) > 0, d*(d-1), 1.0)
local_cc = where(d*(d-1) > 0, diag(AA) / safe_denom, 0.0)

norm_degree = degree / (n - 1)   # normalised to [0, 1]

# DBSCAN cluster labels (same BFS kernel as DBSCANMedoidSelector in impl.py)
labels = _dbscan_cluster_labels(D, eps, min_samples)
cluster_size_frac[i] = size_of_run_i_cluster / n  # noise → 1/n
```

#### `GraphDegreeSelector` — zero-training baseline

Registered as **`graph-degree`**. Selects `argmax(norm_degree)` without any trained model. Used at ablation step ① to confirm graph topology signal before adding it to Extreme10.

**Graph feature quality direction:** All three features — `local_cc`, `norm_degree`, `cluster_size_frac` — are "higher = better" (denser graph neighbourhood → more likely in correct cluster). In the feature builder they use `_rank01(sub)`.

---

### 5a-5. Extreme10 (17-dim) — `extreme10_impl.py`

**Complete feature list (17 dimensions)**:

```
Extreme9 base (11)  +  Graph topology (3, from D)  +  Error-mass (3, from tok_conf)
```

| Dim | Name | Source | Formula | Direction |
|-----|------|--------|---------|-----------|
| 0–10 | (Extreme9 features) | `extreme9_impl` + `local_conf_impl` | (see §5a-3) | — |
| 11 | `local_cc_r` | `graph_topo_impl` | `_rank01(local_cc[idx])` | ↑ |
| 12 | `norm_degree_r` | `graph_topo_impl` | `_rank01(norm_degree[idx])` | ↑ |
| 13 | `cluster_size_r` | `graph_topo_impl` | `_rank01(cluster_size_frac[idx])` | ↑ |
| 14 | `instability_mass_r` | `local_conf_impl` | `_rank01(-mean(arr > μ+0.5σ))` | ↑ (lower mass = better) |
| 15 | `tail_variance_r` | `local_conf_impl` | `_rank01(-var(arr[-T//10:]))` | ↑ (lower var = better) |
| 16 | `event_pre_post_delta_r` | `local_conf_impl` | `_rank01(mean(pre_2slices) − mean(post_event))` | ↑ (recovery = better) |

#### Error-mass feature details (`extract_error_mass_raw` in `local_conf_impl.py`)

**`instability_mass`**: fraction of tokens where `tok_conf > μ_i + 0.5σ_i`. Captures bursty confidence spikes — a run with many such tokens is unstable. Default fallback = 0.5.

**`tail_variance`**: `np.var(arr[-max(1,T//10):])` over the final 10% of tokens. High variance at the end indicates the model has not settled into a confident generation mode. Default fallback = imputed finite mean.

**`event_pre_post_delta`**: `mean(arr[pre_lo:pre_hi]) − mean(arr[post_start:])` where:
- `pre_window` = tokens in the 2 slices immediately preceding the **last** reflection event
- `post_window` = all tokens after the last reflection event slice

Positive delta means the run became more confident (lower tok_conf) after reflecting — a hallmark of productive reflection. Default = 0.0 if no reflection event exists.

**中文说明 — 误差质量特征三要素：**
- `instability_mass`：置信度不稳定 token 占比（tok\_conf 超过均值 + 0.5标准差）；越低越好，说明 run 整体平稳。
- `tail_variance`：末尾 10% token 的方差；越低越好，说明末尾生成进入稳定的自信状态。
- `event_pre_post_delta`：最后一次反思事件前后置信度恢复幅度（正值 = 反思有效，tok\_conf 下降 = 更自信）；越高越好。

#### Key architectural decision: lazy D-computation in `select()`

Graph features require D, which is **not available at `bind()` time**. The base class `_Extreme10BaseSelector` caches `self._graph_raw = None` and populates it on the first call to `_score_payload(D, ...)`:

```python
def _score_payload(self, D, best_model, worst_model):
    if self._graph_raw is None:               # lazy, once per group
        self._graph_raw = extract_graph_topo_raw(D)
    return accumulate_extreme10_scores(
        ..., graph_raw=self._graph_raw, ...
    )
```

`bind()` resets `_graph_raw = None` for each new problem group, ensuring no cross-group contamination.

**关键架构决策（中文）：** 图拓扑特征需要 D 矩阵，而 `bind()` 阶段没有 D。因此在 `_score_payload()` 内部**惰性计算并缓存** `graph_raw`：每个问题组第一次调用 `select()` 时计算，后续同组调用直接复用。`bind()` 会将 `_graph_raw` 重置为 `None`，确保组间隔离。

#### Training script architecture: Option C (precompute D in worker)

`train_extreme10_selectors.py` extends `train_extreme9_selectors.py` with D computation inside each worker:

```python
def _extract_problem_payloads(batch, reflection_threshold):
    for spec in batch:
        # Step 1: extract 14-key raw values (no D needed)
        raw_values = extract_extreme10_raw_values(ctx, reflection_threshold)

        # Step 2: build RunViews and compute D (1 thread per worker)
        default_vspec = ViewSpec(Agg.MAX, CutSpec(CutType.MASS, 1.0), Order.BY_KEY)
        views = [reader.get_run_view(rid, default_vspec) for rid in run_ids]
        D = DistanceEngine(DistanceSpec("ja", num_threads=1)).dense_matrix(views)

        # Step 3: extract graph topology features
        graph_raw = extract_graph_topo_raw(D)

        payload["raw_values"] = {k: np.asarray(v, float64) for k,v in raw_values.items()}
        payload["graph_raw"]  = {k: np.asarray(v, float64) for k,v in graph_raw.items()}
```

Thread constraint: `--workers 4` × `1 DistanceEngine thread` = 4 threads total ≤ 16-core limit.

**训练脚本架构（中文）：** 采用 Option C 方案：在 worker 进程内部直接计算 D（`DistanceEngine(DistanceSpec("ja", num_threads=1))`），同步提取 `graph_raw`，以 `payload["graph_raw"]` 形式传递回主进程。每个 worker 只用 1 个 DistanceEngine 线程，`--workers 4` 时共 4 线程，满足 16 核限制。回退保护：若 D 计算失败，`graph_raw` 自动填充为零（`norm_degree=0, cluster_size_frac=1/n`），训练不中断。

---

### 5a-6. Ablation Strategy & Verification Commands (消融策略与验证命令)

```
消融顺序 | Ablation order:

① graph-degree (零训练，~5 min | zero-training, ~5 min)
   → confirm: graph topology signal vs dc_r / dbscan-medoid

② Extreme9 + graph only (14-dim, train)
   → isolate graph contribution

③ Extreme9 + error-mass only (14-dim, train)
   → isolate error-mass contribution

④ Full Extreme10 (17-dim, train only if ② or ③ shows gain)
   → check synergy vs individual contributions

⑤ Zero-ablation (optional): remove each of the 6 new features one at a time
   → prune features that do not contribute
```

```bash
# Step 1: zero-training graph baseline (5 min)
# 步骤一：零训练图拓扑基线
python -m nad.cli analyze \
  --cache-root MUI_HUB/cache/.../cache_... \
  --selectors graph-degree dc_r dbscan-medoid \
  --distance ja --distance-threads 16 \
  --out /tmp/gtopo_test.json

# Step 2: train full Extreme10
# 步骤二：训练 Extreme10
source .venv/bin/activate
python scripts/train_extreme10_selectors.py \
  --objective aggregated_selacc10 \
  --workers 4

# Step 3: evaluate
# 步骤三：评估
python -m nad.cli analyze \
  --cache-root MUI_HUB/cache/.../cache_... \
  --selectors extreme10-best extreme10-mixed extreme9-best dc_r \
  --distance ja --distance-threads 16 \
  --out /tmp/extreme10_eval.json
```

---

### 5a-7. File Map (文件清单)

| File | Role |
|------|------|
| `nad/core/selectors/extreme8_impl.py` | Extreme8: 3-dim, base reusables (`_impute_finite`, `normalize_weight_direction`, `sample_tuple_indices`, `LinearRankModel`, `ZeroRankModel`, band-reward utilities) |
| `nad/core/selectors/local_conf_impl.py` | Local tok_conf features (8 dims for Extreme9) + error-mass features (3 dims for Extreme10): `extract_local_conf_raw()`, `extract_error_mass_raw()`, `LocalConfTailSelector` |
| `nad/core/selectors/graph_topo_impl.py` | Graph topology features: `extract_graph_topo_raw()`, `_dbscan_cluster_labels()`, `GraphDegreeSelector` |
| `nad/core/selectors/extreme9_impl.py` | Extreme9: 11-dim, `LinearRankModel9`, `extract_extreme9_raw_values()`, `build_extreme9_features()`, `accumulate_extreme9_scores()`, `Extreme9{Best,Worst,Mixed}Selector` |
| `nad/core/selectors/extreme10_impl.py` | Extreme10: 17-dim, `LinearRankModel10`, `extract_extreme10_raw_values()`, `build_extreme10_features()`, `accumulate_extreme10_scores()`, `Extreme10{Best,Worst,Mixed}Selector` |
| `nad/core/selectors/registry.py` | Dispatcher: `build_selector()` routes `graph-degree`, `local-conf-tail`, `extreme9-*`, `extreme10-*` |
| `scripts/train_extreme9_selectors.py` | Extreme9 training (pointwise / band-reward / aggregated_selacc10) |
| `scripts/train_extreme10_selectors.py` | Extreme10 training (same objectives + D computation per worker via `DistanceEngine`) |
| `models/ml_selectors/extreme9_{best,worst}.pkl` | Trained Extreme9 model artifacts |
| `models/ml_selectors/extreme10_{best,worst}.pkl` | Trained Extreme10 model artifacts (after training) |

---

## 6. Plugin System: Custom Selectors

**Module:** `nad/core/selectors/registry.py`
**Example:** `plugins/kink_selector.py`

### Loading custom selectors

```bash
# File-based (recommended)
--selectors 'all,file:./plugins/kink_selector.py:KinkSelector'

# Module path
--selectors 'all,py:nad.core.selectors.impl:MedoidSelector'
```

### Selector contract

```python
from nad.core.selectors.base import Selector, SelectorContext
import numpy as np

class MySelector(Selector):
    def bind(self, context: SelectorContext):
        # context.cache      — CacheReader
        # context.problem_id — current problem
        # context.run_ids    — List[int] global run IDs
        # context.views      — List[RunView] aligned with run_ids
        # context.pos_window — Optional[Tuple[lo, hi]]
        self._ctx = context

    def select(self, D: np.ndarray, run_stats) -> int:
        # D: [n, n] pairwise distance matrix (group-local)
        # run_stats["lengths"]: activation lengths per run
        # MUST return an integer in [0, n-1]  (group-local index)
        # ALWAYS filter inf before argmin/argmax
        return int(np.argmin(D.sum(axis=1)))
```

### Built-in plugin: KinkSelector (`plugins/kink_selector.py`)

Detects **kinks** (breakpoints) in the cumulative unique-neuron-count curve:
1. Calls `extract_tokenwise_counts()` to get cumulative unique count vs token position
2. Computes per-token slope of the curve
3. Applies MAD (Median Absolute Deviation) anomaly detection on slope changes
4. Counts kinks per run
5. Selects run with the **fewest kinks** (most stable activation growth)

---

## 7. Evaluation & Accuracy Scoring

**Modules:** `nad/ops/accuracy.py`, `tools/calculate_accuracy.py`  
**Command:** `python3 -m nad.cli accuracy`

### Process

1. Load selection JSON (`analyze` output)
2. Load ground truth from `evaluation_report_compact.json` (or uncompressed fallback)
3. Build `sample_id → is_correct` map via `meta.json` + ground truth
4. For each `(problem, selector)` pair:
   - Retrieve selected `run_id`
   - Look up correctness
   - Accumulate correct / total counts
5. Emit `AccuracyReport`

### Ground truth schema support

Two schemas are auto-detected:

| Schema | Structure |
|--------|-----------|
| Schema 1 | `{"results": [{"problem_id": …, "runs": […]}]}` |
| Schema 2 | `{"sample_breakdown": {"correct_samples": […], "incorrect_samples": [...]}}` |

### `AccuracyReport` dataclass

```python
selector_accuracy  : Dict[str, float]             # selector → accuracy %
selector_counts    : Dict[str, Tuple[int, int]]    # selector → (correct, total)
per_problem        : Dict[str, Dict]               # per-problem breakdown
```

### Ground truth auto-generation (`tools/auto_generate_ground_truth.py`)

Automatically discovers evaluation reports by traversing from the cache root up to the `MUI_Public` root (which contains `neuron/` and `infer_results/`). Reads `report_path` from `meta.json` and resolves it relative to `infer_results/`.

---

## 8. Multi-Task Selector Ranking

**Module:** `nad/ops/selector_ranking.py`  
**Command:** `python3 scripts/rank_selectors.py`

Aggregates per-task accuracy results across models and datasets into a single cross-task ranking.

### Scoring algorithms

| Algorithm | Formula | Purpose |
|-----------|---------|---------|
| **RNS** (Rank-Normalized Score) | `1 − rank / num_selectors` | Normalises per task; prevents high-accuracy tasks from dominating |
| **Relative Regret** | `(max_acc − selector_acc) / max_acc` | How far from the best selector |
| **Copeland score** | `# selectors that this selector beats` | Pairwise tournament ranking |

### Cross-task aggregation

- **Task weights:** uniform or proportional to problem count
- **Category filters:** `all`, `programming` (mbpp, humaneval, livecodebench), `math_science` (aime24, aime25, gpqa)
- **Bootstrap CI** (optional): resample tasks with replacement, compute 95% CI via `np.percentile([2.5, 97.5])`

### Output formats

- Console table (human-readable)
- CSV (`--csv`)
- JSON (`--json`)
- Per-category breakdown

---

## 9. Position-Window Ablation

**Modules:** `nad/pipeline/window_cache.py`, `nad/pipeline/analysis.py`  
**Cookbook chapter:** `cookbook/04_position_ablation/`

### What it studies

How selector accuracy changes as more tokens (positions) are included in the activation representation.

### How it works

1. Cache is built with `--row-bank` to store per-row (per-token-position) activations in `rows/`
2. At analysis time, `--pos-window 0-8` restricts activations to positions [0, 8) (= 0–256 tokens at `pos_size=32`)
3. The pipeline lazily builds a `window_cache/pos_0_8/` sub-cache on first use and re-uses it subsequently
4. Each window is analysed independently with the full selector suite

### Supported windows (cookbook defaults)

| Window | Token range (pos_size=32) |
|--------|--------------------------|
| `0-1`  | 0–32 tokens |
| `0-2`  | 0–64 tokens |
| `0-8`  | 0–256 tokens |
| `0-32` | 0–1 024 tokens |
| `0-128`| 0–4 096 tokens |
| `0-512`| 0–16 384 tokens |

`--pos-window all` sweeps every window automatically up to the maximum observed length.

---

## 10. CLI Interface

**Module:** `nad/cli/__main__.py`  
**Entry point:** `python3 -m nad.cli`

### Commands

| Command | Purpose | Key flags |
|---------|---------|-----------|
| `cache-build-fast` | Two-stage Map-Reduce cache builder | `--raw-dir`, `--meta-json`, `--workers`, `--row-bank` |
| `analyze` | Run all selectors on a cache | `--cache-root`, `--selectors`, `--distance`, `--pos-window`, `--distance-threads` |
| `accuracy` | Score selector picks against ground truth | `--selection`, `--cache-root`, `--out` |
| `cache-build` | Legacy single-threaded builder | (deprecated) |

### Global flags

- `--log-level DEBUG|INFO|WARNING|ERROR|CRITICAL`

### `analyze` flags of note

| Flag | Default | Description |
|------|---------|-------------|
| `--selectors` | `all` | Selector list, JSON array, or `file:path.py:Class` |
| `--distance` | `ja` | `ja` (Jaccard) or `wj` (Weighted Jaccard) |
| `--agg` | `max` | `max` or `sum` activation aggregation |
| `--cut` | `topk:8192` | `topk:N` or `mass:0.98` |
| `--group-topk-policy` | `none` | Per-group pre-filter: `none`, `legacy-min`, `min`, `max`, `fixed:<K>` |
| `--pos-window` | *(none)* | `0-8` or `all` |
| `--pos-size` | `32` | Tokens per position unit |
| `--distance-threads` | `1` | Parallelism for distance matrix |

---

## 11. Visualization & Web Servers

### Cache Browser (`tools/cache_browser.py` + `cookbook/01_cache_browser/`)

**Port:** 5003

A lightweight Flask server that lists every cache under `MUI_HUB/cache/`. For each cache it shows:
model name, dataset, sample count, problem count, accuracy %, schema version, and build date.

**Endpoints:**

| Endpoint | Response |
|----------|---------|
| `GET /` | HTML table of all caches |
| `GET /api/caches` | JSON array of cache metadata (scriptable) |

### Visualization Server (`minimal_visualization_next/app.py` + `cookbook/02_visualization/`)

**Port:** 5002  
**Stack:** Flask + Plotly

An interactive single-cache explorer with:

| Feature | Description |
|---------|-------------|
| Problem browser | Lists problems; filters by correct / incorrect / all |
| Run comparison | Side-by-side view of multiple runs |
| Neuron activation heatmap | Per-token unique-neuron-count plotted as heatmap |
| Token-level metrics | tok_conf, tok_neg_entropy, tok_gini, tok_selfcert, tok_logprob per token |
| Token decoding | Integrates HuggingFace `AutoTokenizer` or `tokenizers.Tokenizer` to show token text |
| LRU cache | 256 MB in-process cache for array reads |
| Correctness overlay | Highlights correct vs incorrect runs |

**NadNextLoader** (`nad/io/loader.py`) backs the visualization server:
- 256 MB LRU cache with hit-rate and eviction stats
- Lazy loading of binary arrays
- Problem catalogue grouping
- Exposes `extract_tokenwise_counts()` for per-problem analysis

---

## 12. Cookbook: Reproducible Experiment Workflows

Each chapter is a self-contained shell script + README under `cookbook/`.

| Chapter | Name | What it runs |
|---------|------|-------------|
| `00_setup` | Environment setup | `install.sh` (pip), `verify.sh` (12-point checklist) |
| `01_cache_browser` | Cache Browser | Flask server on port 5003 |
| `02_visualization` | Interactive Explorer | Flask+Plotly server on port 5002 |
| `03_batch_analyze` | Batch Analysis | Parallel `analyze` + `accuracy` across all caches; writes `summary.json` |
| `04_position_ablation` | Position Ablation | Sweep 6 token-position windows; generate ablation curves |
| `05_deepconf_analysis` | DeepConf Analysis | 4 metric×reduction variants of DeepConf selector; optional comparison vs all selectors |

### Cookbook 03 — Batch Analysis modes

| Mode | What it does |
|------|-------------|
| `full` | Full-sequence analysis (default) |
| `positions` | Position-window analysis only |
| `all` | Both full and position analysis |

Results land in `./result/all_model_TIMESTAMP/` and feed directly into `rank_selectors.py`.

### Cookbook 05 — DeepConf modes

| Mode | Variants run |
|------|-------------|
| `quick` | 1 config (tok_neg_entropy + mean) |
| `standard` | 4 configs (2 metrics × 2 reductions) |
| `full` | Standard + position-window sweep |
| `custom` | User-specified metric + reduction + group-size |

---

## 13. Scripts & Plotting Utilities

All scripts live in `scripts/`. Run from the repo root.

| Script | Purpose |
|--------|---------|
| `rank_selectors.py` | Multi-task ranking (RNS / Regret / Copeland + Bootstrap CI); outputs CSV / JSON table |
| `plot_position_ablation.py` | Accuracy-vs-token-consumption plots for each cache × selector; exports PNG + CSV summary |
| `plot_aime_combined.py` | NeurIPS-style figure: 18 subplots (3 models × 6 datasets), accuracy vs k, 5 selectors with 95% CI shading |
| `plot_downsample_ablation.py` | Extended downsample ablation visualization |
| `plot_extended_downsample.py` | Extended variant of the downsample ablation plot |
| `compare_token_per_kink_outputs.sh` | Shell script to diff token-per-kink analysis outputs |
| `test_legacy_adapter.sh` | Smoke-test for backward-compat adapter (`nad/compat/`) |

### `plot_position_ablation.py` details

- Loads `rows/token_row_ptr` + `rows/sample_row_ptr` to get per-sample token counts
- Normalises accuracy by token consumption for fair comparison
- Writes one PNG per cache + one aggregate CSV with Avg@64 baseline column

### `plot_aime_combined.py` details

- NeurIPS-style matplotlib settings (serif font, 300 DPI, inward ticks)
- X-axis: k values (2, 4, 8, 16, 32, 64) on log scale
- Y-axis: accuracy %
- 5 selector lines, 95% CI shading, legend, title per subplot

---

## 14. Tools & Standalone Utilities

| Tool | File | Purpose |
|------|------|---------|
| Cache Browser server | `tools/cache_browser.py` | Flask server for browsing all caches |
| Ground-truth generator | `tools/auto_generate_ground_truth.py` | Auto-discovers evaluation reports from `MUI_Public` layout and generates `evaluation_report_compact.json` |
| Accuracy calculator | `tools/calculate_accuracy.py` | Standalone accuracy report: computes selector accuracy %, per-problem breakdown, formatted table, saves JSON |
| Window cache cleaner | `tools/clean_window_cache.sh` | Removes stale `window_cache/` directories |

### `auto_generate_ground_truth.py`

- Walks up the directory tree from a cache or neuron-output directory to find `MUI_Public/`
- Resolves `report_path` from `meta.json` relative to `infer_results/`
- Reads raw evaluation JSON and converts to the compact binary format
- Handles multi-schema evaluation report formats

### `calculate_accuracy.py`

- Standalone version of `nad.cli accuracy` with richer console output
- Prints a formatted accuracy table to stdout
- Saves full per-problem breakdown to JSON
- Useful for ad-hoc analysis without the full CLI pipeline

---

## 15. Public Python API

**Module:** `nad/api.py`

Intended for external use and custom scripting:

```python
from nad.api import open_cache, load_correctness_map, extract_tokenwise_counts

# Open a cache (lazy mmap)
cache = open_cache('./path/to/cache')
print(cache.num_runs)  # total samples

# Load correctness labels
correct_map = load_correctness_map('./path/to/cache')
# -> Dict[int, bool]  (run_id → is_correct)

# Compute cumulative unique neuron count per token position
tokens, counts = extract_tokenwise_counts(
    run_id=42,
    rows_srp=cache.rows_sample_row_ptr,
    rows_rp=cache.rows_row_ptr,
    rows_keys=cache.rows_keys,
)
```

### `CacheReader` properties (via `nad/core/views/reader.py`)

| Property | Type | Description |
|----------|------|-------------|
| `num_runs` | `int` | Total number of samples |
| `paths` | `CachePaths` | All file paths |
| `manifest` | `Manifest` | Schema version, checksums, metadata |
| `get_run_view(id, spec)` | `RunView` | Neuron keys + weights for one sample |
| `get_token_view(id)` | `TokenView` | Token-level metrics for one sample |

---

## 16. Unique-Neuron-Count Operator

**Module:** `nad/ops/uniques.py`  
**Function:** `extract_tokenwise_counts(run_id, rows_srp, rows_rp, rows_keys, …)`

### Purpose

Given a run stored in the Row-CSR Bank, compute the **cumulative count of unique neurons** as tokens accumulate. This answers: "how many distinct neurons have fired by token position t?"

### Algorithm (fully vectorised, no Python loops)

```
1. Extract run's contiguous slice from rows/ arrays
2. If rows_slice_ids provided:
   - Stable sort rows by slice_id (token position order)
   - Reorder keys accordingly
3. np.unique(all_keys, return_index=True)  → find first occurrence of each key
4. np.searchsorted(row_boundaries, first_occurrence_indices)  → map to row (position)
5. np.bincount(row_positions)  → count new unique keys per row
6. np.cumsum(new_per_row)  → cumulative unique count
```

### Output axes

| `token_axis` | X-axis | Y-axis |
|-------------|--------|--------|
| `"row"` | Row index (0, 1, …) | Cumulative unique count |
| `"tokens"` | Actual slice IDs (token positions) | Cumulative unique count |

**Used by:** `KinkSelector` (Plugin), `k_token_per_kink.py` (analysis tool), visualization server heatmap

---

## 17. Kink Detection (Breakpoint Analysis)

**Files:** `k_token_per_kink.py` (standalone script), `plugins/kink_selector.py`

### Purpose

Detect **kinks** — sudden changes in the slope of the cumulative unique-neuron curve. A kink indicates a burst of new neuron activations at a particular token position.

### Algorithm

```
1. Compute cumulative unique count C[t] vs token position t
2. Compute slope: slope[t] = C[t] - C[t-1]
3. Compute slope change: Δ[t] = slope[t] - slope[t-1]
4. MAD (Median Absolute Deviation) anomaly detection:
   threshold = median(|Δ|) + k × MAD(Δ)
5. Kinks = positions t where |Δ[t]| > threshold
6. KinkSelector: select run with fewest kinks (most stable growth)
```

### `k_token_per_kink.py` standalone script

- Processes an entire cache directory
- Reports average tokens-per-kink ratio per problem
- Useful for characterising activation stability of different problems or models

---

## 18. Configuration & Environment

### `pyproject.toml`

Package metadata and dependencies. Install with:
```bash
pip install .          # installs numpy>=1.20.0, pyroaring>=0.4.5 and optional extras
```

### `nad_config.json`

Specifies `model_search_dirs` — directories where the visualization server searches for HuggingFace tokenizer files (model weight directories with `tokenizer.json`).

```json
{
  "model_search_dirs": ["/home/jovyan/public-ro/model"]
}
```

### Environment variables

| Variable | Values | Effect |
|----------|--------|--------|
| `NAD_JA_BACKEND` | `roaring` / `numpy` / `auto` | Override Jaccard backend |
| `OMP_NUM_THREADS` | `1` | Prevent NumPy BLAS thread oversubscription |
| `MKL_NUM_THREADS` | `1` | Same for MKL |

### Thread / parallelism constraint

This machine has **16 cores**. All scripts satisfy `THREADS × PARALLEL_JOBS ≤ 16`.

| PARALLEL_JOBS | THREADS |
|---------------|---------|
| 1 | 16 |
| 4 | 4 |
| 8 | 2 |

### Backward compatibility (`nad/compat/`)

A compatibility shim for older analysis scripts. Tested via `scripts/test_legacy_adapter.sh`.

---

## 19. Feature Summary Table

This table shows everything implemented **beyond** chain-of-thought, neuron activation, and confidence/entropy:

| Area | Implemented features |
|------|---------------------|
| **Data ingestion** | Two-stage Map-Reduce NPZ→CSR cache builder; atomic writes; 32–60 parallel workers |
| **Cache format** | CSR binary cache (base, index, token_data, rows); manifest v4.0 with SHA-256 integrity; lazy mmap access |
| **Distance metrics** | Jaccard (unweighted); Weighted Jaccard; NumPy and Roaring Bitmap backends; auto-switching at 4 096 elements; thread-parallel dense matrix |
| **Selection algorithms** | 35 selectors: 10 core (min/max-activation, Medoid, KNN-Medoid, DBSCAN-Medoid, Consensus-Min, Consensus-Max, DeepConf, avg64@, con64@); 4 Ensemble/Tournament; 2 Two-stage; 4 classic ML; 1 Temporal; 3 Trajectory; 3 Extreme8; 1 LocalConfTail; 3 Extreme9; 1 GraphDegree; 3 Extreme10 |
| **Plugin system** | File-based and module-path plugin loading; full SelectorContext API; KinkSelector reference implementation |
| **Accuracy evaluation** | Ground truth loading (2 schemas); per-problem and per-selector accuracy; compressing/decompressing evaluation reports |
| **Ground truth generation** | Auto-discovery from MUI_Public directory layout; multi-schema conversion; compact binary format |
| **Multi-task ranking** | RNS (Rank-Normalized Score); Relative Regret; Copeland tournament; task category filters (programming / math_science); Bootstrap 95% CI |
| **Position-window ablation** | Row-CSR Bank for per-token queries; lazy window cache generation; 6 pre-defined windows (0-1 to 0-512); sweep-all mode |
| **Unique-neuron counting** | Vectorised cumulative unique count operator; token-axis or row-axis output |
| **Kink detection** | MAD-based slope-change anomaly detection on cumulative neuron curves; tokens-per-kink ratio analysis |
| **CLI** | 4 subcommands (cache-build-fast, analyze, accuracy, cache-build); global log-level; group-topk-policy pre-filter |
| **Cache Browser** | Flask server (port 5003); lists all caches with metadata; JSON API |
| **Visualization server** | Flask+Plotly (port 5002); problem browser; neuron heatmap; token-metric overlay; tokenizer integration; 256 MB LRU cache |
| **Batch workflows** | 6 cookbook chapters (setup, browser, viz, batch-analyze, position-ablation, deepconf); configurable parallelism |
| **Plotting** | Position-ablation accuracy-vs-tokens curves; NeurIPS-style 18-subplot downsample figure; downsample ablation with 95% CI shading |
| **Public API** | `open_cache()`, `load_correctness_map()`, `extract_tokenwise_counts()` |
| **Backward compat** | Legacy adapter shim in `nad/compat/`; smoke-test script |
| **Configuration** | `nad_config.json` for tokenizer paths; `NAD_JA_BACKEND` env var; thread constraint documentation |
