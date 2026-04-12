# Paper Outline

这份文档提供一个适合当前 SVDomain 材料的论文结构模板。

目标不是替你写全文，而是把当前仓库最适合讲清楚的叙事顺序固定下来。

---

## 1. 推荐标题方向

可选标题风格：

### 偏方法

**Domain-Aware Low-Rank Early-Stop Selection with Interpretable SVD Routing**

### 偏系统

**SVDomain: A Reproducible and Interpretable Early-Stop Selector for Best-of-N Reasoning Runs**

### 偏实验

**Simple Linear Routing Still Matters: Domain-Split SVD Selectors for Early-Stop Run Selection**

---

## 2. 建议摘要结构

摘要建议包含五句话：

1. **问题**：best-of-N / early-stop run selection 很重要，但最终 slot-only 或复杂黑盒 reranker 都不够理想。
2. **方法**：我们提出 domain-aware four-anchor SVD routing，使用 `raw+rank` 表示和低秩线性头。
3. **结果**：方法在 blind leaderboard 上优于 previous best early-stop SVD 主提交。
4. **trajectory evidence**：dense-anchor timing 与 dense cross-anchor transfer 表明，该表示共享并不局限于 slot-100，而是沿 trajectory 持续存在，但可复用性依赖 domain 与 anchor maturity。
5. **解释性**：我们进一步构建了 feature-level、family-level、decision-level 的统一解释接口与 viewer。

---

## 3. 正文结构建议

### 3.1 Introduction

建议回答三个问题：

- 为什么 early-stop selection 比 final-slot-only 更重要？
- 为什么需要 domain-aware 方法？
- 为什么简单、可解释的低秩线性模型仍然值得研究？

### 3.2 Problem Setting

定义：

- problem
- candidate runs
- early-stop positions
- group-local selection

这里建议放一张图：

- `SVDomain/figures/pipeline_overview.mmd`

### 3.3 Method

按下面顺序写：

1. extraction positions / anchor positions
2. `raw+rank` representation
3. feature families
4. `StandardScaler -> TruncatedSVD -> LogisticRegression`
5. domain split
6. route selection / anchor mapping

### 3.4 Experimental Setup

写：

- datasets and domains
- holdout split
- evaluation metrics
- baselines

### 3.5 Main Results

正文建议放三张主表 + 一张 trajectory-supporting table：

1. math / science / combined holdout
2. blind leaderboard 主表
3. dense cross-anchor transfer summary
4. best-of-n or coding 作为附表之一（视篇幅）

### 3.6 Interpretability

建议按三层结构写：

1. model-level
2. sample-level
3. decision-level

### 3.7 Discussion

重点讨论：

- 为什么 coding 不 work
- 为什么 final-slot-only 不够
- 为什么 transfer 不是 slot-100-only，但也不是全域均匀
- 为什么 simple linear routing 仍有竞争力

### 3.8 Limitations and Future Work

可以诚实写：

- coding domain evidence still weak
- current interpretability artifact checked in as smoke export
- dense `r2` evidence is supporting evidence rather than the primary leaderboard result

---

## 4. 图表建议

### 图 1：系统总览

展示：

- extraction positions
- anchor mapping
- domain split
- SVD route
- final group-local selection

来源：

- `SVDomain/figures/pipeline_overview.mmd`

### 图 2：解释性三层结构

展示：

- model summary
- run contributions
- top1 vs top2 deltas

来源：

- `SVDomain/figures/explainability_stack.mmd`

### 图 3：viewer 截图

建议截：

- trajectory 图
- top1 / top2 / top3
- feature panel
- hover correctness

### 表 1：Holdout 主表

用：

- `SVDomain/results/tables/earlystop_holdout_summary.csv`

### 表 2：Blind leaderboard 主表

用：

- `SVDomain/results/tables/blind_leaderboard_summary.csv`

### 表 3：Interpretability sanity

用：

- `SVDomain/results/tables/interpretability_sanity.csv`

### 表 4：Dense cross-anchor transfer summary

用：

- `SVDomain/results/tables/dense_cross_anchor_transfer_summary.csv`

这张表最适合承接 dense timing 结果，展示：

- diagonal mean gap
- all off-diagonal mean gap
- near-gap vs far-gap
- best reusable source anchor

---

## 5. 最适合写进正文的 claim

建议重点 claim 如下：

### Claim A

**四锚点、低秩线性、域拆分的简单结构，足以在 blind early-stop leaderboard 上优于旧主线。**

### Claim B

**相比只看 final slot，multi-anchor routing 更能提升整体 early-stop 选择曲线。**

### Claim C

**Dense trajectory evidence shows that transfer is not slot-100-only: the learned basis is shared across anchors, but its reusability is domain- and maturity-dependent.**

### Claim D

**SVDomain 的解释性不是事后补充，而是直接来自可回投影的线性结构。**

### Claim E

**coding 负结果说明 domain-specific structure matters。**

---

## 6. 建议放 appendix 的内容

1. coding 全部结果
2. slot100 直抽 best-of-n 对照
3. checkpoint ranking side task
4. per-cache breakdown
5. full dense cross-anchor matrix / gap-by-distance tables
6. full route summary table
7. viewer API / artifact index

---

## 7. 可能的弱点与应对

### 弱点 1

science 提升不如 math 显著。

应对：

- 强调 combined noncoding 的总体收益
- 强调方法的 simplicity / interpretability / reproducibility

### 弱点 2

coding 没有赢。

应对：

- 把它写成 boundary case / negative result
- 说明当前 canonical noncoding feature family 并不自动适配 coding

### 弱点 3

best-of-n slot100 不是最优。

应对：

- 说明 pure final-slot extraction 只是桥接 baseline
- 它恰好支持“early-stop 路径信息很重要”这一论点

### 弱点 4

dense `r2` 线不是 leaderboard 主提交。

应对：

- 明确把 dense timing 和 dense transfer 定位为 **representation-level supporting evidence**
- 强调它们回答的是 “where the signal appears” 与 “how far the basis transfers” 两个机制问题，而不是替代 `r1` blind result

---

## 8. 写作时的建议取舍

如果篇幅有限，优先保留：

1. canonical `r1` 方法
2. holdout + blind leaderboard 主结果
3. dense timing + dense cross-anchor summary
4. interpretability

优先弱化：

1. coding merge
2. slot100 direct best-of-n
3. checkpoint ranking side task

因为这三块更适合作为 supporting evidence，而不是主剧情。
