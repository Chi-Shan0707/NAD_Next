# GPQA DeepSets Round 1 — Plan

Date: `2026-04-07`

## Repo State Confirmed Before Running

1. `code_v2` is already the promoted coding default.
2. `science_baseline_v1` is still the frozen science baseline.
3. `gpqa_pairwise_round2` is `NO-PROMOTE`.
4. `science_hybrid_round3` already has a narrow full-system promote.
5. The current line is no longer graph-heavy expansion or a new monotonic recency feature family.
6. The new research line should move to the smallest viable full-group contextual model.

## Goal

Run one narrow new experiment:

- `gpqa_deepsets_round1`

Nothing else is promoted to main scope:

- no new coding experiment
- no new math experiment
- no Set Transformer
- no attention
- no LinearSVC hybrid gate
- no graph-heavy branch

## Model Shape

Input is restricted to the existing GPQA run-level structured features:

- `dc_z`
- `dc_r`
- `reflection_count_r`
- `prefix_conf_mean_r`
- `recency_conf_mean_r`
- `late_recovery_r`

Minimal DeepSets design:

- each run goes through a tiny MLP encoder
- group pooling is tested with only:
  - `mean`
  - `max`
- final run score uses:
  - per-run embedding
  - pooled group embedding
- no raw neuron rows
- no raw slices
- no attention
- no wide hyperparameter sweep

## Evaluation

Official GPQA compare rows:

- `science_baseline_v1`
- `gpqa_pairwise_round1`
- `science_hybrid_round3`
- `gpqa_deepsets_round1`

Metrics:

- `AUROC`
- `Hit@1`
- `Pairwise`
- `SelAcc@10`

System-level evaluation must patch only the science slice into the current promoted stack:

- generic extreme stays frozen
- coding stays promoted `code_v2`
- current science is `science_hybrid_round3`
- candidate science is `gpqa_deepsets_round1`

Both must be reported:

- sample-weighted system proxy
- equal-cache-mean system proxy

## Promotion Rule

Promote only if both hold:

1. GPQA top-slot metrics do not regress against current `science_hybrid_round3`
   - `Hit@1`
   - `SelAcc@10`
2. The patched full-system proxy is genuinely better than the current system

If the candidate improves only `AUROC` / `Pairwise`, it is `NO-PROMOTE`.

---

# GPQA DeepSets Round 1 — 计划（中文）

日期：`2026-04-07`

## 运行前确认的仓库状态

1. `code_v2` 已经是 promoted coding default。
2. `science_baseline_v1` 仍然是 frozen science baseline。
3. `gpqa_pairwise_round2` 的结论仍然是 `NO-PROMOTE`。
4. `science_hybrid_round3` 已经拿到了一个 narrow full-system promote。
5. 当前主线已经不再继续 graph-heavy 扩展，也不再新增 monotonic recency feature family。
6. 新的研究方向应转向最小可行的 full-group contextual model。

## 目标

本轮只做一个收敛范围很小的新实验：

- `gpqa_deepsets_round1`

以下方向都不进入本轮主线：

- 不做新的 coding 实验
- 不做新的 math 实验
- 不做 Set Transformer
- 不做 attention
- 不做 LinearSVC hybrid gate
- 不做 graph-heavy 分支

## 模型形态

输入严格限制为现有 GPQA 的 run-level structured features：

- `dc_z`
- `dc_r`
- `reflection_count_r`
- `prefix_conf_mean_r`
- `recency_conf_mean_r`
- `late_recovery_r`

最小 DeepSets 设计：

- 每条 run 先过一个很小的 MLP encoder
- group pooling 只测试：
  - `mean`
  - `max`
- 每条 run 的最终分数使用：
  - per-run embedding
  - pooled group embedding
- 不使用 raw neuron rows
- 不使用 raw slices
- 不使用 attention
- 不做大范围超参搜索

## 评估

GPQA 官方对比对象：

- `science_baseline_v1`
- `gpqa_pairwise_round1`
- `science_hybrid_round3`
- `gpqa_deepsets_round1`

指标：

- `AUROC`
- `Hit@1`
- `Pairwise`
- `SelAcc@10`

系统级评估只允许把 science slice patch 到当前 promoted stack：

- generic extreme 保持 frozen
- coding 保持 promoted `code_v2`
- 当前 science 为 `science_hybrid_round3`
- 候选 science 为 `gpqa_deepsets_round1`

必须同时报告：

- sample-weighted system proxy
- equal-cache-mean system proxy

## Promote 规则

只有同时满足以下两点才允许 promote：

1. GPQA 的 top-slot 指标不低于当前 `science_hybrid_round3`
   - `Hit@1`
   - `SelAcc@10`
2. patch 之后的 full-system proxy 确实优于当前系统

如果候选只提升了 `AUROC` / `Pairwise`，则结论必须是 `NO-PROMOTE`。
