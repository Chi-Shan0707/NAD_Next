# Viewer and Explain APIs

这份文档整理 explainability core 与 viewer 相关的 Python / HTTP 接口。

---

## 1. Explain core：`nad/explain/svd_explain.py`

### 作用

为 canonical SVD models 提供统一解释接口。

### 当前支持

- `es_svd_math_rr_r1`
- `es_svd_science_rr_r1`
- `es_svd_ms_rr_r1`

### 关键能力

1. effective linear weight 回投影
2. 单样本 feature contribution
3. family contribution
4. top1 vs top2 delta
5. wrong-top1 vs best-correct-run 对比
6. reconstruction sanity check

### 为什么重要

这是当前 interpretability 线的核心，不只是 viewer 的辅助函数。

---

## 2. 导出接口

### 脚本

- `scripts/export_svd_explanations.py`

### 输出目录

- `results/interpretability/es_svd_math_rr_r1/`
- `results/interpretability/es_svd_science_rr_r1/`
- `results/interpretability/es_svd_ms_rr_r1/`

### 典型用途

- 生成论文表格
- 生成 appendix artifact
- viewer 离线浏览

---

## 3. Flask API：`cot_viewer/app.py`

### Canonical SVD 相关 API

#### `GET /api/svd/explain/model_summary`

query:

- `method`
- `domain`
- `anchor`

返回：

- model-level summary
- family strength
- top positive / negative features

#### `GET /api/svd/explain/problem_top1_vs_top2`

query:

- `cache`
- `problem_id`
- `method`
- `anchor`

返回：

- top runs
- margin
- top feature deltas
- top family deltas
- trajectory data

#### `GET /api/svd/explain/run_contributions`

query:

- `cache`
- `problem_id`
- `sample_id`
- `method`
- `anchor`

返回：

- per-run feature contributions
- family contributions
- reconstructed score

---

## 4. 前端：`cot_viewer/static/app.js`

### 当前支持的行为

- 切换 canonical SVD methods
- 切换 `Anchor`
- 切换 `SVD Domain`
- 点击 `Top1 / Top2 / Top3` 查看对应 run
- trajectory 图 hover 显示 run 是否正确

### 当前 UI 回答的问题

1. 当前方法选了哪个 run？
2. top1 比 top2 强在哪些 family？
3. 当前选中的 top1 / top2 / top3 本身是否正确？
4. 这条 run 的高分来自哪些 feature contribution？

---

## 5. 模板与样式

相关文件：

- `cot_viewer/templates/index.html`
- `cot_viewer/static/styles.css`

它们主要负责：

- dashboard layout
- decision card
- feature panel
- trajectory hero chart

---

## 6. 当前最适合论文截图的内容

推荐截三类图：

1. trajectory hero 图
2. decision + feature panel
3. top1 / top2 / top3 切换后的 per-run explanation

这样最能体现：

- 方法不是黑盒
- 解释不是离线静态表格
- dashboard 可以直接支持 case study

---

## 7. 目前的重要限制

### 7.1 coding cache

canonical SVD explain 当前在 coding cache 上会返回：

- `applicable = false`

这属于设计选择，而不是 bug。

### 7.2 checked-in artifact

仓库当前提交的是 compact smoke export，而不是全量 explain dump。  
如果你要做全量统计图，建议自行重新导出 full artifact。

---

## 8. 推荐如何在论文中描述

建议写成：

> We expose a three-layer explanation interface for canonical SVD models:
> model-level domain-anchor priors, run-level feature contributions,
> and decision-level top1-vs-top2 deltas, all accessible both offline and through an interactive viewer.
