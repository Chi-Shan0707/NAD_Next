# Math DeepSets Round 1 — Plan

Date: `2026-04-08`

## Goal

Run one narrow new math experiment:

- `math_deepsets_round1`

This extends the minimal DeepSets contextual line into math instead of
continuing blind linear-SVM expansion.

## Scope

- domain: `math`
- profile: `main`
- caches:
  - `aime24`
  - `aime25`
  - `brumo25`
  - `hmmt25`
- inputs: existing run-level structured features only
- no raw neuron rows
- no raw slice tensors
- no attention
- no Set Transformer
- no broad hyperparameter search

## Feature Family

Use the current math structured feature stack:

- base 12 `ml_features`
- augmented 8 derived consensus / disagreement features

Total:

- `all_aug` 20-d structured feature family

## Model Shape

Minimal DeepSets:

- per-run encoder MLP
- tiny hidden sizes
- pooled group context via:
  - `mean`
  - `max`
- final per-run score from:
  - run embedding
  - pooled group embedding

Optional tiny extension:

- one small `pairwise_aux_weight` run after the minimal pointwise pass

## Compare Rows

- `medoid`
- `knn-medoid`
- fixed math runwise SVM recipe
- fixed math RankSVM recipe
- `math_deepsets_round1`

## Promotion Read

Math does not yet have a promoted specialized patch in the same way coding and
science do, so the bar is:

1. do not lose to `knn-medoid` on the current top-slot read
2. improve the patched full-system proxy if replacing the math slice

Otherwise it is `NO-PROMOTE`.

---

# Math DeepSets Round 1 — 计划（中文）

日期：`2026-04-08`

## 目标

只做一个收敛范围很小的数学新实验：

- `math_deepsets_round1`

这次是把最小 DeepSets contextual 线路扩到数学，而不是继续盲目扩张线性 SVM 搜索。

## 范围

- 领域：`math`
- profile：`main`
- 缓存：
  - `aime24`
  - `aime25`
  - `brumo25`
  - `hmmt25`
- 输入：只用现有 run-level structured features
- 不用 raw neuron rows
- 不用 raw slice tensors
- 不用 attention
- 不用 Set Transformer
- 不做大范围超参搜索

## 特征家族

使用当前数学结构化特征栈：

- 12 维基础 `ml_features`
- 8 维派生 consensus / disagreement 特征

合计：

- `all_aug` 20 维 structured feature family

## 模型形态

最小 DeepSets：

- 每条 run 先过一个小 MLP encoder
- hidden size 保持很小
- group context pooling 只试：
  - `mean`
  - `max`
- 每条 run 的最终分数来自：
  - run embedding
  - pooled group embedding

允许的唯一小扩展：

- 在最小 pointwise 跑通后，再补一个很小的 `pairwise_aux_weight` 版本

## 对比对象

- `medoid`
- `knn-medoid`
- 固定的 math runwise SVM recipe
- 固定的 math RankSVM recipe
- `math_deepsets_round1`

## Promote 判断

数学域目前还没有像 coding / science 那样已经 promote 的 specialized patch，因此门槛设为：

1. 当前 top-slot 读数上不能输给 `knn-medoid`
2. 如果替换 math slice，patched full-system proxy 需要真正提升

否则结论就是 `NO-PROMOTE`。
