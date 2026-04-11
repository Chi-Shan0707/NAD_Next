# ES SVD CODING RR R1

## Naming

- `es_svd`：EarlyStop SVD family。
- `coding`：只覆盖 coding 域。
- `rr`：只用 `raw+rank` 表示。
- `r1`：当前 coding 分域重训第一版。
- 名称里故意去掉 `prefix` / `p10`，但训练协议仍是四个 anchors：`10/40/70/100`。

## Feature Spec

- `representation`：`raw+rank`。
- `feature family`：`token_plus_traj_fixed`。
- `included features`：`tok_conf_prefix, tok_conf_recency, tok_gini_prefix, tok_gini_tail, tok_gini_slope, tok_neg_entropy_prefix, tok_neg_entropy_recency, tok_selfcert_prefix, tok_selfcert_recency, tok_logprob_prefix, tok_logprob_recency, traj_continuity, traj_reflection_count, traj_novelty, traj_max_reflection, traj_late_convergence, has_tok_conf, has_tok_gini, has_tok_neg_entropy, has_tok_selfcert, has_tok_logprob, has_rows_bank`。
- `excluded features`：`nc_mean, nc_slope, self_similarity, tail_q10, head_tail_gap, tail_variance, last_event_tail_conf, event_pre_post_delta`。
- `row feature`：不使用任何数值 row 特征；只保留 `has_rows_bank` 作为 availability flag。

## Protocol

- `main cache root`：`/home/jovyan/public-ro/MUI_HUB/cache`。
- `extra cache root`：`/home/jovyan/public-ro/MUI_HUB/cache_train`。
- `holdout split`：`85/15`。
- `holdout unit`：按 `dataset + problem_id` 做跨 root 一致切分，`split_seed=42`。
- `anchors`：`10, 40, 70, 100`。
- `routing policy`：训练与最终 bundle 均不允许 baseline / single-feature route；baseline 只做对照。
- `extra cache usage`：`scanned but no coding caches found`。

## Artifacts

- `coding model`：`models/ml_selectors/es_svd_coding_rr_r1.pkl`。
- `summary json`：`results/scans/earlystop/es_svd_coding_rr_r1_summary.json`。
- `eval json`：`results/scans/earlystop/es_svd_coding_rr_r1_eval.json`。
- `blind coding scores`：`results/scans/earlystop/es_svd_coding_rr_r1_blind_coding_scores.json`。
- `merged submission`：`submission/EarlyStop/es_svd_ms_rr_r1__coding_rr_r1.json`。

## Validate

### coding holdout

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | 50.62% | 45.81% | N/A | 49.04% | 48.00% |
| earlystop_svd_lowrank_lr_v1 | 47.62% | 38.56% | N/A | 47.33% | 56.00% |
| earlystop_prefix10_svd_round1 | 47.62% | 38.56% | N/A | 47.33% | 56.00% |
| es_svd_coding_rr_r1 | 43.42% | 24.00% | N/A | 40.68% | 52.00% |

### es_svd_coding_rr_r1 per-cache

| Cache | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| cache/DS-R1/lcb_v5 | 43.42% | 24.00% | N/A | 40.68% | 52.00% |

### coding full-fit anchor routes

| Anchor | CV AUROC | Baseline | Baseline CV | Rank | C | Whiten | Class Weight |
|---|---:|---|---:|---:|---:|---:|---|
| 10% | 0.4844 | traj_novelty | 0.5380 | 16 | 0.20 | yes | none |
| 40% | 0.5010 | traj_novelty | 0.5399 | 12 | 1.00 | no | balanced |
| 70% | 0.5296 | tok_gini_prefix | 0.5345 | 12 | 1.00 | no | balanced |
| 100% | 0.5220 | tok_gini_prefix | 0.5281 | 12 | 1.00 | no | balanced |

## Blind Export

- `base submission`：`submission/EarlyStop/es_svd_ms_rr_r1__coding_from_round1c.json`。
- `merged method_name`：`es_svd_ms_rr_r1__coding_rr_r1`。
- `coding cache keys replaced`：`DS-R1/lcb_v5, Qwen3-4B/lcb_v5`。
- `blind feature cache status`：`loaded_from_pickle`。
- `blind feature cache path`：`results/cache/export_earlystop_svd_submission_strongfeat_20260410/feature_store_all_ref030_18a73b5e30f1a00d.pkl`。
- `submission validation`：`{'total_problems': 970, 'total_samples': 62080}`。

## Leaderboard Result

- `submission id`：`#147`。
- `submitted at`：`2026-04-11 02:55:07 UTC`。
- `submission method_name`：`es_svd_ms_rr_r1__coding_rr_r1`。
- `status`：`Not best`。
- `primary score`：`3.8125`。
- `all-metric auc_of_auroc`：`0.7427`。
- `all-metric auc_of_selacc`：`0.8276`。
- `earliest > 0.6`：`10%`。
- `samples`：`62080`。

### Online Metrics by Position

| Position | AUROC | Selective Acc @10% | Stop Acc (Hit@1) |
|---|---:|---:|---:|
| 10% | 0.7884 | 0.8985 | 0.7071 |
| 20% | 0.7990 | 0.9032 | 0.7017 |
| 30% | 0.7871 | 0.9025 | 0.7220 |
| 40% | 0.8319 | 0.9255 | 0.7154 |
| 50% | 0.8325 | 0.9264 | 0.7369 |
| 60% | 0.8295 | 0.9267 | 0.7375 |
| 70% | 0.8399 | 0.9249 | 0.7240 |
| 80% | 0.8437 | 0.9261 | 0.7414 |
| 90% | 0.8451 | 0.9270 | 0.7365 |
| 100% | 0.8475 | 0.9283 | 0.7279 |

### Online Per-Cache Breakdown

| Cache | AUC-AUROC | AUC-SelAcc | Earliest>0.6 | Samples |
|---|---:|---:|---:|---:|
| DS-R1/aime24 | 0.8302 | 0.9000 | 10% | 1920 |
| DS-R1/aime25 | 0.8280 | 0.8935 | 10% | 1920 |
| DS-R1/brumo25 | 0.8601 | 0.8953 | 10% | 1920 |
| DS-R1/gpqa | 0.6755 | 0.8384 | 10% | 12672 |
| DS-R1/hmmt25 | 0.8281 | 0.8984 | 10% | 1920 |
| DS-R1/lcb_v5 | 0.4597 | 0.4905 | N/A | 10688 |
| Qwen3-4B/aime24 | 0.8247 | 0.8995 | 10% | 1920 |
| Qwen3-4B/aime25 | 0.8385 | 0.9000 | 10% | 1920 |
| Qwen3-4B/brumo25 | 0.8133 | 0.9000 | 10% | 1920 |
| Qwen3-4B/gpqa | 0.7067 | 0.8606 | 10% | 12672 |
| Qwen3-4B/hmmt25 | 0.7890 | 0.8961 | 10% | 1920 |
| Qwen3-4B/lcb_v5 | 0.4583 | 0.5586 | N/A | 10688 |

### Comparison to Base Submission

- `base submission`：`#144 / es_svd_ms_rr_r1__coding_from_round1c`。
- `primary score`：`3.8125 -> 3.8125`，无变化。
- `auc_of_auroc`：`0.7428 -> 0.7427`，下降 `-0.0001`。
- `auc_of_selacc`：`0.8317 -> 0.8276`，下降 `-0.0041`。
- `auroc@100%`：`0.8468 -> 0.8475`，改善 `+0.0007`。
- `stop_acc@100%`：`0.7299 -> 0.7279`，下降 `-0.0020`。
- `coding conclusion`：线上结果与 holdout 判断一致；新的 coding SVD 没有带来整体收益，因此该版本保留为已验证分支，不替代当前主提交。
