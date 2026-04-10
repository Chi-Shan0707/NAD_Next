# EARLYSTOP SVD PROBLEM-CENTERED ROUND1 PLAN (2026-04-09)

## 核心结论（先验确认）

- round1b 正确协议是：non-coding 使用 `cache + cache_train`，并按 `dataset + problem_id` 做跨 root 一致的 `80/20` holdout；coding 继续 fallback 到旧 v1。
- strong-feature 线已经把 family 收窄到 `tok_conf_prefix / tok_conf_recency / traj_reflection_count` 及少量 tail / recovery 信号。
- 当前 holdout winner `strongfeat_noncoding_anchor4_coding_v1_ref020` 的四个 anchors（10/40/70/100）全部使用 `raw+rank`。
- 因此，本轮任务不是“直接去掉 rank”，而是检验更偏题内排序的表示是否能超过当前 `raw+rank`。

## 目标

- 在 strong-feature family + round1b 正确协议下，明确检验题内中心化表示是否优于 `raw+rank`。
- 不做宽特征回退，不改 coding fallback，不看 `cache_test` 标签做选型。

## 训练协议

- non-coding：`cache + cache_train`，按 `dataset + problem_id` 跨 root 一致 `80/20` holdout。
- coding：固定 fallback 到 `earlystop_svd_lowrank_lr_v1`。
- 结构：`10/40/70/100` anchor4 训练 + official 10 slots 路由。

## 表示与特征

- 表示搜索：`raw`, `rank`, `raw+rank`, `centered_raw`, `centered_raw+rank`, `zscore_within_problem_raw`。
- 特征家族仅：`strong_core3` / `strong_tail5` / `strong_stable6` / `strong_event7` / `strong_recovery8`。
- reflection threshold：`0.20`, `0.30`。

## 评测口径

- A. EarlyStop：`AUC of AUROC`、`AUC of SelAcc`、`Earliest > 0.6`、`AUROC@100%`、`Stop Acc@100%`。
- B. Slot100 -> Best-of-N bridge：`Hit@1`、`Hit@3`、`SelAcc@10`、`Pairwise`。
- bridge 评测范围：`Holdout+Coding`（non-coding holdout + coding fallback）。

