# Code DeepSets Round 1 — Plan

Date: `2026-04-08`

## Goal

Run one narrow new coding experiment:

- `code_deepsets_round1`

This is a direct extension of the minimal `gpqa_deepsets_round1` line into the
coding domain, not a new large architecture family.

## Scope

- domain: `coding` only
- cache: `DS-R1/lcb_v5` under `MUI_HUB/cache`
- inputs: existing run-level structured coding features only
- no raw neuron rows
- no raw slice tensors
- no attention
- no Set Transformer
- no large hyperparameter search

## Feature Family

Use the current promoted coding structured line:

- `prefix_best_window_quality_r`
- `head_tail_gap_r`
- `tail_variance_r`
- `post_reflection_recovery_r`
- `last_block_instability_r`

These are exactly the current `code_v2` structured signals.

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

- `code_baseline_v1`
- `code_v2`
- `code_deepsets_round1`

## Promotion Read

Coding is already on promoted `code_v2`, so the new candidate should only be
considered promotion-worthy if:

1. it does not break the coding gate relative to current `code_v2`
2. it improves the patched full-system proxy

Otherwise it is `NO-PROMOTE`.

---

# Code DeepSets Round 1 — 计划（中文）

日期：`2026-04-08`

## 目标

只做一个收敛范围很小的 coding 新实验：

- `code_deepsets_round1`

这是把最小 `gpqa_deepsets_round1` 思路直接扩展到 coding 域，不是另起一个大模型家族。

## 范围

- 领域：只做 `coding`
- 缓存：`MUI_HUB/cache` 下的 `DS-R1/lcb_v5`
- 输入：只用现有 run-level structured coding features
- 不用 raw neuron rows
- 不用 raw slice tensors
- 不用 attention
- 不用 Set Transformer
- 不做大范围超参搜索

## 特征家族

直接使用当前 promoted coding 线的结构化信号：

- `prefix_best_window_quality_r`
- `head_tail_gap_r`
- `tail_variance_r`
- `post_reflection_recovery_r`
- `last_block_instability_r`

也就是当前 `code_v2` 的 5 个核心 structured signals。

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

- `code_baseline_v1`
- `code_v2`
- `code_deepsets_round1`

## Promote 判断

当前 coding 已经使用 promoted `code_v2`，因此只有同时满足以下条件，才值得考虑 promote：

1. 相对当前 `code_v2` 不破坏 coding gate
2. patch 后的 full-system proxy 真正提升

否则结论就是 `NO-PROMOTE`。
