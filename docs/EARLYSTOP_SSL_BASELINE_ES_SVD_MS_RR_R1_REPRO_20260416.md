# ES SVD MS RR R1

## Naming

- `es_svd`: EarlyStop SVD family.
- `math` / `science` / `ms`: covered domain(s).
- `rr`: raw+rank representation only.
- `r1`: first version of the split-domain retraining.
- `prefix` / `p10` are intentionally dropped from the name; training protocol still uses four anchors: `10/40/70/100`.

## Feature Spec

- `representation`：`raw+rank`。
- `feature family`：`token_plus_traj_fixed`。
- `included features`：`tok_conf_prefix, tok_conf_recency, tok_gini_prefix, tok_gini_tail, tok_gini_slope, tok_neg_entropy_prefix, tok_neg_entropy_recency, tok_selfcert_prefix, tok_selfcert_recency, tok_logprob_prefix, tok_logprob_recency, traj_continuity, traj_reflection_count, traj_novelty, traj_max_reflection, traj_late_convergence, has_tok_conf, has_tok_gini, has_tok_neg_entropy, has_tok_selfcert, has_tok_logprob, has_rows_bank`。
- `excluded features`：`nc_mean, nc_slope, self_similarity, tail_q10, head_tail_gap, tail_variance, last_event_tail_conf, event_pre_post_delta, prefix_best_window_quality, post_reflection_recovery, last_block_instability, reflection_density, reflection_count, conf_d1_tail_mean, conf_abs_d1_tail_mean, conf_abs_d2_tail_mean, conf_abs_d1_full_minus_tail, gini_d1_tail_mean, gini_abs_d1_tail_mean, gini_abs_d2_tail_mean, gini_abs_d1_full_minus_tail, entropy_d1_tail_mean, entropy_abs_d1_tail_mean, entropy_abs_d2_tail_mean, entropy_abs_d1_full_minus_tail`。
- `row feature`: no numeric row features are used; only `has_rows_bank` is retained as an availability flag.

## Protocol

- `main cache root`：`/home/jovyan/public-ro/MUI_HUB/cache`。
- `extra cache root`：`/home/jovyan/public-ro/MUI_HUB/cache_train`。
- `holdout split`：`85/15`。
- `holdout unit`：按 `dataset + problem_id` 做跨 root 一致切分，`split_seed=42`。
- `anchors`：`10, 40, 70, 100`。
- `routing policy`：训练与最终 bundle 均不允许 baseline / single-feature route；baseline 只做对照。
- `coding`：本轮故意不包含 coding routes，以避免把旧 baseline routing 混进新版本。

## Artifacts

- `math model`：`models/ml_selectors/earlystop_ssl_baselines/es_svd_math_rr_r1_repro.pkl`。
- `science model`：`models/ml_selectors/earlystop_ssl_baselines/es_svd_science_rr_r1_repro.pkl`。
- `ms model`：`models/ml_selectors/earlystop_ssl_baselines/es_svd_ms_rr_r1_repro.pkl`。
- `summary json`：`results/scans/earlystop_ssl/baseline_es_svd_ms_rr_r1_repro_summary.json`。
- `eval json`：`results/scans/earlystop_ssl/baseline_es_svd_ms_rr_r1_repro_eval.json`。

## Validate

### math holdout

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | 68.52% | 94.23% | 10% | 68.38% | 75.00% |
| earlystop_svd_lowrank_lr_v1 | 77.69% | 95.60% | 10% | 90.47% | 71.43% |
| earlystop_prefix10_svd_round1 | 93.48% | 99.95% | 10% | 96.30% | 78.57% |
| es_svd_math_rr_r1 | 95.69% | 99.78% | 10% | 97.55% | 75.00% |

### es_svd_math_rr_r1 per-cache

| Cache | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| cache/DS-R1/aime24 | 98.61% | 100.00% | 10% | 99.80% | 75.00% |
| cache/DS-R1/aime25 | 88.48% | 100.00% | 10% | 95.66% | 50.00% |
| cache/DS-R1/brumo25 | 99.03% | 100.00% | 10% | 98.94% | 75.00% |
| cache/DS-R1/hmmt25 | 94.87% | 99.23% | 10% | 95.74% | 75.00% |
| cache_train/DS-R1/aime25 | 94.09% | 100.00% | 10% | 96.31% | 75.00% |
| cache_train/DS-R1/brumo25 | 98.01% | 100.00% | 10% | 98.94% | 100.00% |
| cache_train/DS-R1/hmmt25 | 96.73% | 99.23% | 10% | 97.45% | 75.00% |

### math full-fit anchor routes

| Anchor | CV AUROC | Baseline | Baseline CV | Rank | C | Whiten | Class Weight |
|---|---:|---|---:|---:|---:|---:|---|
| 10% | 0.9022 | traj_continuity | 0.7667 | 16 | 1.00 | no | none |
| 40% | 0.9391 | traj_continuity | 0.7856 | 16 | 1.00 | no | none |
| 70% | 0.9488 | traj_reflection_count | 0.7826 | 16 | 1.00 | no | none |
| 100% | 0.9563 | traj_reflection_count | 0.8189 | 16 | 1.00 | no | none |

### science holdout

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | 80.89% | 96.30% | 10% | 85.61% | 70.00% |
| earlystop_svd_lowrank_lr_v1 | 77.19% | 96.59% | 10% | 79.37% | 76.67% |
| earlystop_prefix10_svd_round1 | 77.30% | 97.97% | 10% | 79.84% | 73.33% |
| es_svd_science_rr_r1 | 79.38% | 98.75% | 10% | 84.93% | 68.33% |

### es_svd_science_rr_r1 per-cache

| Cache | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| cache/DS-R1/gpqa | 77.10% | 98.54% | 10% | 81.10% | 70.00% |
| cache_train/DS-R1/gpqa | 81.67% | 98.96% | 10% | 88.75% | 66.67% |

### science full-fit anchor routes

| Anchor | CV AUROC | Baseline | Baseline CV | Rank | C | Whiten | Class Weight |
|---|---:|---|---:|---:|---:|---:|---|
| 10% | 0.7202 | tok_conf_recency | 0.6422 | 16 | 0.05 | yes | none |
| 40% | 0.7699 | tok_conf_recency | 0.6849 | 16 | 1.00 | yes | none |
| 70% | 0.7715 | tok_conf_recency | 0.6935 | 16 | 1.00 | no | balanced |
| 100% | 0.7838 | tok_conf_recency | 0.7225 | 16 | 0.50 | yes | none |

### combined noncoding holdout

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | 71.27% | 94.69% | 10% | 72.21% | 73.89% |
| earlystop_svd_lowrank_lr_v1 | 77.58% | 95.82% | 10% | 88.01% | 72.59% |
| earlystop_prefix10_svd_round1 | 89.88% | 99.51% | 10% | 92.64% | 77.41% |
| es_svd_ms_rr_r1 | 92.07% | 99.55% | 10% | 94.74% | 73.52% |

### es_svd_ms_rr_r1 per-cache

| Cache | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| cache/DS-R1/aime24 | 98.61% | 100.00% | 10% | 99.80% | 75.00% |
| cache/DS-R1/aime25 | 88.48% | 100.00% | 10% | 95.66% | 50.00% |
| cache/DS-R1/brumo25 | 99.03% | 100.00% | 10% | 98.94% | 75.00% |
| cache/DS-R1/hmmt25 | 94.87% | 99.23% | 10% | 95.74% | 75.00% |
| cache_train/DS-R1/aime25 | 94.09% | 100.00% | 10% | 96.31% | 75.00% |
| cache_train/DS-R1/brumo25 | 98.01% | 100.00% | 10% | 98.94% | 100.00% |
| cache_train/DS-R1/hmmt25 | 96.73% | 99.23% | 10% | 97.45% | 75.00% |
| cache/DS-R1/gpqa | 77.10% | 98.54% | 10% | 81.10% | 70.00% |
| cache_train/DS-R1/gpqa | 81.67% | 98.96% | 10% | 88.75% | 66.67% |

