# CoT Viewer — Chain-of-Thought Browser / 思维链查看器

[English](#english) | [中文](#中文)

---

## 中文

一个轻量级 Web UI，用于浏览 NAD Next 缓存中已解码的推理链，并逐 Token 检查各项指标。

### 快速启动

```bash
/home/jovyan/work/NAD_Next/.venv/bin/python /home/jovyan/work/NAD_Next/cot_viewer/app.py
```

在浏览器中打开 **http://\<host\>:5002**。

**依赖：**
- Python 包：`flask`、`numpy`、`transformers`（已在 `.venv` 中）
- 分词器模型：`/home/jovyan/public-ro/model/DeepSeek-R1-0528-Qwen3-8B`
- 缓存数据：`/home/jovyan/public-ro/MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/`

### 功能

- **浏览** `MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/` 下的所有数据集
- **阅读** 每道题 64 次推理运行的完整解码思维链文本
- **检查** 每个 Token 的指标（置信度、熵、Gini、自我确定性、对数概率），点击 32-token 片段即可查看

### 界面布局

```
┌─────────────────────────────────────────────┐
│  [数据集 ▼]  [题目 ▼]  [运行 ▼]            │
├─────────────────────────────────────────────┤
│                                             │
│  思维链文本                                  │
│  （每个约 32-token 的片段可点击，            │
│   以虚线分隔）                               │
│                                             │
│  最大高度 55vh，可滚动                       │
├─────────────────────────────────────────────┤
│  第 N 片段详情（点击上方片段后显示）          │
│  ┌────┬──────┬──────┬───┬────┬───┬────┐    │
│  │位置│Token │置信度│ H │Gini│SC │LP  │    │
│  ├────┼──────┼──────┼───┼────┼───┼────┤    │
│  │160 │ the  │ 0.95 │.42│0.88│.91│-.1 │    │
│  └────┴──────┴──────┴───┴────┴───┴────┘    │
│  最大高度 35vh，可滚动                       │
└─────────────────────────────────────────────┘
```

### 交互流程

1. **页面加载** → 数据集下拉框自动填充（aime24、aime25、gpqa 等）
2. **选择数据集** → 题目列表显示题目 ID、运行次数和准确率 %
3. **选择题目** → 运行下拉框显示每次运行的 `运行 N ✓/✗`
4. **选择运行** → 思维链文本渲染为可点击的 32-token 片段
5. **点击片段** → Token 详情表格显示全部 6 项指标，并对熵值进行颜色标注

**颜色说明：**
- **绿色**：熵 < 0.3（高确定性）
- **红色**：熵 > 2.0（高不确定性）

### API 端点

除 `/api/datasets` 外，所有端点均需 `?cache=<path>` 查询参数。

| 端点 | 返回 |
|------|------|
| `GET /api/datasets` | `{名称: 缓存路径, ...}` |
| `GET /api/problems?cache=` | `[{problem_id, num_runs, accuracy}, ...]` |
| `GET /api/runs/<problem_id>?cache=` | `[{sample_id, run_index, is_correct}, ...]` |
| `GET /api/chain/<sample_id>?cache=` | `{num_slices, slices: [{idx, text, tok_start, tok_end}]}` |
| `GET /api/slice/<sample_id>/<slice_idx>?cache=` | `{tokens: [{pos, token_id, text, conf, entropy, gini, selfcert, logprob}]}` |

### 架构说明

- **分词器**：启动时从本地模型目录加载一次
- **数据访问**：使用 `CacheReader.get_token_view()` 获取每个样本的 token 数组，用 `rows_token_row_ptr` 确定 32-token 片段边界
- **熵约定**：缓存中存储的是负熵（`tok_neg_entropy`），API 层取反后返回正值
- **提示词 token**：行库中第 0 行是提示词（跳过），响应片段从第 1 行开始
- **懒加载**：`NadNextLoader` 和 `CacheReader` 实例在首次访问时创建，后续复用

### 文件结构

```
cot_viewer/
  app.py              # Flask 后端（约 230 行）
  templates/
    index.html        # 单页纯 JS 前端（约 200 行）
  README.md           # 本文件
```

---

## English

A lightweight web UI for browsing decoded reasoning chains and inspecting per-token metrics from NAD Next caches.

## What It Does

- **Browse** all datasets under `MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/`
- **Read** the full decoded chain-of-thought text for each of the 64 reasoning runs per problem
- **Inspect** per-token metrics (confidence, entropy, Gini, self-certainty, log-probability) by clicking on 32-token slices

## Quick Start

```bash
/home/jovyan/work/NAD_Next/.venv/bin/python /home/jovyan/work/NAD_Next/cot_viewer/app.py
```

Open **http://\<host\>:5002** in your browser.

### Requirements

- Python packages: `flask`, `numpy`, `transformers` (already in `.venv`)
- Tokenizer model at `/home/jovyan/public-ro/model/DeepSeek-R1-0528-Qwen3-8B`
- Cache data at `/home/jovyan/public-ro/MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/`

## UI Layout

```
┌─────────────────────────────────────────────┐
│  [Dataset ▼]  [Problem ▼]  [Run ▼]         │
├─────────────────────────────────────────────┤
│                                             │
│  Chain-of-Thought text                      │
│  (each ~32-token slice is a clickable       │
│   <span> with dashed underline)             │
│                                             │
│  max-height: 55vh, scrollable               │
├─────────────────────────────────────────────┤
│  Slice N Detail (click a slice above)       │
│  ┌─────┬───────┬──────┬────┬─────┬────┬───┐│
│  │ Pos │ Token │ Conf │ H  │Gini │SC  │LP ││
│  ├─────┼───────┼──────┼────┼─────┼────┼───┤│
│  │ 160 │ the   │ 0.95 │0.42│0.88 │0.91│-.1││
│  └─────┴───────┴──────┴────┴─────┴────┴───┘│
│  max-height: 35vh, scrollable               │
└─────────────────────────────────────────────┘
```

## Interaction Flow

1. **Page loads** → datasets dropdown auto-populates (aime24, aime25, gpqa, etc.)
2. **Select a dataset** → problems list shows problem ID, run count, and accuracy %
3. **Select a problem** → run dropdown shows `Run N ✓/✗` for each of the 64 runs
4. **Select a run** → chain-of-thought text renders as clickable 32-token slices
5. **Click a slice** → token detail table shows all 6 metrics with color-coded entropy

### Color Coding

- **Green** cells: entropy < 0.3 (high certainty)
- **Red** cells: entropy > 2.0 (high uncertainty)

## API Endpoints

All endpoints (except `/api/datasets`) require a `?cache=<path>` query parameter.

| Endpoint | Returns |
|----------|---------|
| `GET /api/datasets` | `{name: cache_path, ...}` for all datasets |
| `GET /api/problems?cache=` | `[{problem_id, num_runs, accuracy}, ...]` |
| `GET /api/runs/<problem_id>?cache=` | `[{sample_id, run_index, is_correct}, ...]` |
| `GET /api/chain/<sample_id>?cache=` | `{num_slices, slices: [{idx, text, tok_start, tok_end}]}` |
| `GET /api/slice/<sample_id>/<slice_idx>?cache=` | `{tokens: [{pos, token_id, text, conf, entropy, gini, selfcert, logprob}]}` |

## Architecture Notes

- **Tokenizer**: loaded once at startup from the local model directory
- **Data access**: uses `CacheReader.get_token_view()` for per-sample token arrays and `rows_token_row_ptr` for 32-token slice boundaries
- **Entropy convention**: the cache stores negative entropy (`tok_neg_entropy`); the API negates it to return positive values
- **Prompt tokens**: row 0 in the rows bank is the prompt (skipped); response slices start at row 1
- **Lazy loading**: `NadNextLoader` and `CacheReader` instances are created on first access per cache and reused

## Files

```
cot_viewer/
  app.py              # Flask backend (~230 lines)
  templates/
    index.html         # Single-page vanilla JS frontend (~200 lines)
  README.md            # This file
```
