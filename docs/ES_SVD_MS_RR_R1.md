# ES SVD MS RR R1

## Naming

- `es_svd`：EarlyStop SVD family。
- `math` / `science` / `ms`：模型覆盖域。
- `rr`：只用 `raw+rank` 表示。
- `r1`：当前分域重训第一版。
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
- `coding`：本轮故意不包含 coding routes，以避免把旧 baseline routing 混进新版本。

## Artifacts

- `math model`：`models/ml_selectors/es_svd_math_rr_r1.pkl`。
- `science model`：`models/ml_selectors/es_svd_science_rr_r1.pkl`。
- `ms model`：`models/ml_selectors/es_svd_ms_rr_r1.pkl`。
- `summary json`：`results/scans/earlystop/es_svd_ms_rr_r1_summary.json`。
- `eval json`：`results/scans/earlystop/es_svd_ms_rr_r1_eval.json`。

## Validate

### math holdout

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | 68.52% | 94.23% | 10% | 68.38% | 75.00% |
| earlystop_svd_lowrank_lr_v1 | 77.69% | 95.60% | 10% | 90.47% | 71.43% |
| earlystop_prefix10_svd_round1 | 93.48% | 99.95% | 10% | 96.30% | 78.57% |
| es_svd_math_rr_r1 | 95.81% | 99.73% | 10% | 98.17% | 75.00% |

### es_svd_math_rr_r1 per-cache

| Cache | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| cache/DS-R1/aime24 | 98.92% | 100.00% | 10% | 99.76% | 75.00% |
| cache/DS-R1/aime25 | 88.69% | 100.00% | 10% | 97.10% | 50.00% |
| cache/DS-R1/brumo25 | 99.11% | 100.00% | 10% | 99.47% | 75.00% |
| cache/DS-R1/hmmt25 | 94.79% | 98.85% | 10% | 96.46% | 75.00% |
| cache_train/DS-R1/aime25 | 94.41% | 100.00% | 10% | 97.44% | 75.00% |
| cache_train/DS-R1/brumo25 | 98.12% | 100.00% | 10% | 99.08% | 100.00% |
| cache_train/DS-R1/hmmt25 | 96.61% | 99.23% | 10% | 97.90% | 75.00% |

### math full-fit anchor routes

| Anchor | CV AUROC | Baseline | Baseline CV | Rank | C | Whiten | Class Weight |
|---|---:|---|---:|---:|---:|---:|---|
| 10% | 0.8950 | traj_continuity | 0.7619 | 16 | 1.00 | no | balanced |
| 40% | 0.9354 | traj_continuity | 0.7804 | 16 | 1.00 | no | none |
| 70% | 0.9450 | traj_reflection_count | 0.7766 | 16 | 1.00 | no | none |
| 100% | 0.9567 | traj_reflection_count | 0.8133 | 16 | 1.00 | no | balanced |

### science holdout

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | 80.89% | 96.30% | 10% | 85.61% | 70.00% |
| earlystop_svd_lowrank_lr_v1 | 77.19% | 96.59% | 10% | 79.37% | 76.67% |
| earlystop_prefix10_svd_round1 | 77.30% | 97.97% | 10% | 79.84% | 73.33% |
| es_svd_science_rr_r1 | 79.85% | 98.80% | 10% | 84.11% | 65.00% |

### es_svd_science_rr_r1 per-cache

| Cache | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| cache/DS-R1/gpqa | 77.52% | 98.65% | 10% | 80.51% | 63.33% |
| cache_train/DS-R1/gpqa | 82.18% | 98.96% | 10% | 87.71% | 66.67% |

### science full-fit anchor routes

| Anchor | CV AUROC | Baseline | Baseline CV | Rank | C | Whiten | Class Weight |
|---|---:|---|---:|---:|---:|---:|---|
| 10% | 0.7128 | tok_conf_recency | 0.6391 | 16 | 0.05 | yes | none |
| 40% | 0.7659 | tok_conf_recency | 0.6838 | 16 | 1.00 | yes | none |
| 70% | 0.7640 | tok_conf_recency | 0.6933 | 16 | 1.00 | yes | none |
| 100% | 0.7781 | tok_conf_recency | 0.7229 | 16 | 0.05 | no | none |

### combined noncoding holdout

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | 71.27% | 94.69% | 10% | 72.21% | 73.89% |
| earlystop_svd_lowrank_lr_v1 | 77.58% | 95.82% | 10% | 88.01% | 72.59% |
| earlystop_prefix10_svd_round1 | 89.88% | 99.51% | 10% | 92.64% | 77.41% |
| es_svd_ms_rr_r1 | 92.26% | 99.52% | 10% | 95.05% | 72.78% |

### es_svd_ms_rr_r1 per-cache

| Cache | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| cache/DS-R1/aime24 | 98.92% | 100.00% | 10% | 99.76% | 75.00% |
| cache/DS-R1/aime25 | 88.69% | 100.00% | 10% | 97.10% | 50.00% |
| cache/DS-R1/brumo25 | 99.11% | 100.00% | 10% | 99.47% | 75.00% |
| cache/DS-R1/hmmt25 | 94.79% | 98.85% | 10% | 96.46% | 75.00% |
| cache_train/DS-R1/aime25 | 94.41% | 100.00% | 10% | 97.44% | 75.00% |
| cache_train/DS-R1/brumo25 | 98.12% | 100.00% | 10% | 99.08% | 100.00% |
| cache_train/DS-R1/hmmt25 | 96.61% | 99.23% | 10% | 97.90% | 75.00% |
| cache/DS-R1/gpqa | 77.52% | 98.65% | 10% | 80.51% | 63.33% |
| cache_train/DS-R1/gpqa | 82.18% | 98.96% | 10% | 87.71% | 66.67% |

## Leaderboard Result

- `submission id`：`#144`。
- `submitted at`：`2026-04-11 01:58:22 UTC`。
- `submission method_name`：`es_svd_ms_rr_r1__coding_from_round1c`。
- `status`：`CURRENT BEST`。
- `primary score`：`3.8125`。
- `all-metric auc_of_auroc`：`0.7428`。
- `all-metric auc_of_selacc`：`0.8317`。
- `earliest > 0.6`：`10%`。
- `samples`：`62080`。

### Online Metrics by Position

| Position | AUROC | Selective Acc @10% | Stop Acc (Hit@1) |
|---|---:|---:|---:|
| 10% | 0.7878 | 0.9028 | 0.7021 |
| 20% | 0.7988 | 0.9177 | 0.7027 |
| 30% | 0.7873 | 0.9149 | 0.7225 |
| 40% | 0.8333 | 0.9263 | 0.7134 |
| 50% | 0.8325 | 0.9277 | 0.7319 |
| 60% | 0.8294 | 0.9281 | 0.7350 |
| 70% | 0.8404 | 0.9275 | 0.7260 |
| 80% | 0.8439 | 0.9286 | 0.7394 |
| 90% | 0.8451 | 0.9295 | 0.7395 |
| 100% | 0.8468 | 0.9313 | 0.7299 |

### Online Per-Cache Breakdown

| Cache | AUC-AUROC | AUC-SelAcc | Earliest>0.6 | AUROC@10% | AUROC@50% | AUROC@100% | Stop@100% | Samples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DS-R1/aime24 | 0.8302 | 0.9000 | 10% | 0.8758 | 0.9278 | 0.9639 | 0.8667 | 1920 |
| DS-R1/aime25 | 0.8280 | 0.8935 | 10% | 0.9074 | 0.9242 | 0.9117 | 0.6333 | 1920 |
| DS-R1/brumo25 | 0.8601 | 0.8953 | 10% | 0.9378 | 0.9634 | 0.9731 | 0.8667 | 1920 |
| DS-R1/gpqa | 0.6755 | 0.8384 | 10% | 0.7061 | 0.7689 | 0.7776 | 0.6465 | 12672 |
| DS-R1/hmmt25 | 0.8281 | 0.8984 | 10% | 0.9188 | 0.9244 | 0.9572 | 0.7333 | 1920 |
| DS-R1/lcb_v5 | 0.4708 | 0.5253 | N/A | 0.5147 | 0.5219 | 0.5272 | 0.5988 | 10688 |
| Qwen3-4B/aime24 | 0.8247 | 0.8995 | 10% | 0.8702 | 0.9236 | 0.9196 | 0.8667 | 1920 |
| Qwen3-4B/aime25 | 0.8385 | 0.9000 | 10% | 0.8839 | 0.9383 | 0.9468 | 0.8000 | 1920 |
| Qwen3-4B/brumo25 | 0.8133 | 0.9000 | 10% | 0.7787 | 0.9290 | 0.9367 | 0.8333 | 1920 |
| Qwen3-4B/gpqa | 0.7067 | 0.8606 | 10% | 0.7519 | 0.7821 | 0.8009 | 0.6818 | 12672 |
| Qwen3-4B/hmmt25 | 0.7890 | 0.8961 | 10% | 0.8251 | 0.8913 | 0.9396 | 0.6333 | 1920 |
| Qwen3-4B/lcb_v5 | 0.4489 | 0.5736 | N/A | 0.4830 | 0.4953 | 0.5079 | 0.5988 | 10688 |

### Comparison to Previous Best

- `previous best`：`Submission #115 / earlystop_prefix10_svd_round1`。
- `primary score`：`4.0000 -> 3.8125`，改善 `-0.1875`。
- `auc_of_auroc`：`0.7379 -> 0.7428`，改善 `+0.0049`。
- `auc_of_selacc`：`0.8311 -> 0.8317`，改善 `+0.0006`。
- `auroc@100%`：`0.8492 -> 0.8468`，下降 `-0.0024`。
- `stop_acc@100%`：`0.7504 -> 0.7299`，下降 `-0.0205`。
- `coding note`：两份 coding cache 直接沿用旧 submission 分数，因此本次提升来自 math/science 非 coding 部分。
