# ES SVD MS RR R2

## Naming

- `es_svd`：EarlyStop SVD family。
- `math` / `science` / `ms`：模型覆盖域。
- `rr`：只用 `raw+rank` 表示。
- `r2`：10 anchor 位置 + per-position feature family search。
- 每个 position 独立训练一个 anchor model，不再路由到最近 anchor。

## Feature Spec

- `representation`：`raw+rank`（固定）。
- `feature family`：per-position CV search over:
  - `token_only`：11 token features + availability flags。
  - `token_plus_traj`：11 token + 5 traj features + availability flags。
  - `all`：all PREFIX_SAFE features (token + traj + meta)。
  - `strong_core3`：tok_conf_prefix, tok_conf_recency, traj_reflection_count。
  - `strong_event7`：strong_core3 + tail_q10, head_tail_gap, last_event_tail_conf, has_rows_bank。
  - `token_plus_traj_global` (100% only)：token_plus_traj + self_similarity + tail/event features。
- `row feature`：只保留 `has_rows_bank` 作为 availability flag（strong families）。

## Protocol

- `main cache root`：`/home/jovyan/public-ro/MUI_HUB/cache`。
- `extra cache root`：`/home/jovyan/public-ro/MUI_HUB/cache_train`。
- `holdout split`：`85/15`。
- `holdout unit`：按 `dataset + problem_id` 做跨 root 一致切分，`split_seed=42`。
- `anchors`：`10, 20, 30, 40, 50, 60, 70, 80, 90, 100`。
- `routing policy`：10 独立 anchor，identity 路由 — 不再有 nearest-anchor proxy。
- `coding`：本轮不包含 coding routes。

## Artifacts

- `math model`：`models/ml_selectors/es_svd_math_rr_r2_20260412.pkl`。
- `science model`：`models/ml_selectors/es_svd_science_rr_r2_20260412.pkl`。
- `ms model`：`models/ml_selectors/es_svd_ms_rr_r2_20260412.pkl`。
- `summary json`：`results/scans/earlystop/es_svd_ms_rr_r2_20260412_summary.json`。
- `eval json`：`results/scans/earlystop/es_svd_ms_rr_r2_20260412_eval.json`。

## Validate

### math holdout

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | 68.52% | 94.23% | 10% | 68.38% | 75.00% |
| es_svd_ms_rr_r1 | 95.86% | 99.73% | 10% | 98.26% | 75.00% |
| es_svd_math_rr_r2 | 96.74% | 99.84% | 10% | 97.55% | 75.00% |

### es_svd_math_rr_r2 per-cache

| Cache | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| cache/DS-R1/aime24 | 98.73% | 100.00% | 10% | 99.80% | 75.00% |
| cache/DS-R1/aime25 | 93.04% | 100.00% | 10% | 95.67% | 50.00% |
| cache/DS-R1/brumo25 | 98.85% | 100.00% | 10% | 98.94% | 75.00% |
| cache/DS-R1/hmmt25 | 95.35% | 99.23% | 10% | 95.74% | 75.00% |
| cache_train/DS-R1/aime25 | 95.70% | 100.00% | 10% | 96.31% | 75.00% |
| cache_train/DS-R1/brumo25 | 98.32% | 100.00% | 10% | 98.94% | 100.00% |
| cache_train/DS-R1/hmmt25 | 97.22% | 99.62% | 10% | 97.44% | 75.00% |

### math full-fit anchor routes

| Anchor | CV AUROC | Family | Baseline | Baseline CV | Rank | C | Whiten | Class Weight |
|---|---:|---|---|---:|---:|---:|---:|---|
| 10% | 0.9022 | token_plus_traj | traj_continuity | 0.7667 | 16 | 1.00 | no | none |
| 20% | 0.9211 | token_plus_traj | traj_continuity | 0.7807 | 16 | 1.00 | no | none |
| 30% | 0.9329 | token_plus_traj | traj_continuity | 0.7868 | 16 | 1.00 | no | none |
| 40% | 0.9391 | token_plus_traj | traj_continuity | 0.7856 | 16 | 1.00 | no | none |
| 50% | 0.9436 | token_plus_traj | traj_continuity | 0.7811 | 16 | 1.00 | no | none |
| 60% | 0.9462 | token_plus_traj | traj_continuity | 0.7765 | 16 | 0.50 | no | none |
| 70% | 0.9488 | token_plus_traj | traj_reflection_count | 0.7826 | 16 | 1.00 | no | none |
| 80% | 0.9519 | token_plus_traj | traj_reflection_count | 0.7932 | 16 | 1.00 | no | none |
| 90% | 0.9547 | token_plus_traj | traj_reflection_count | 0.8035 | 16 | 1.00 | no | none |
| 100% | 0.9563 | token_plus_traj | traj_reflection_count | 0.8189 | 16 | 1.00 | no | none |

### science holdout

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | 80.89% | 96.30% | 10% | 85.61% | 70.00% |
| es_svd_ms_rr_r1 | 80.13% | 98.83% | 10% | 86.80% | 73.33% |
| es_svd_science_rr_r2 | 82.86% | 98.36% | 10% | 85.54% | 71.67% |

### es_svd_science_rr_r2 per-cache

| Cache | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| cache/DS-R1/gpqa | 81.57% | 97.81% | 10% | 83.81% | 73.33% |
| cache_train/DS-R1/gpqa | 84.14% | 98.91% | 10% | 87.27% | 70.00% |

### science full-fit anchor routes

| Anchor | CV AUROC | Family | Baseline | Baseline CV | Rank | C | Whiten | Class Weight |
|---|---:|---|---|---:|---:|---:|---:|---|
| 10% | 0.7302 | token_only | tok_conf_recency | 0.6422 | 16 | 0.05 | yes | none |
| 20% | 0.7492 | all | tok_conf_recency | 0.6593 | 16 | 0.20 | yes | none |
| 30% | 0.7694 | all | tok_conf_recency | 0.6758 | 16 | 1.00 | no | balanced |
| 40% | 0.7731 | all | tok_conf_recency | 0.6849 | 16 | 1.00 | no | balanced |
| 50% | 0.7746 | all | tok_conf_recency | 0.6888 | 16 | 1.00 | no | balanced |
| 60% | 0.7746 | all | tok_conf_recency | 0.6904 | 16 | 1.00 | no | balanced |
| 70% | 0.7748 | all | tok_conf_recency | 0.6935 | 16 | 1.00 | no | balanced |
| 80% | 0.7759 | all | tok_conf_recency | 0.6994 | 16 | 1.00 | no | balanced |
| 90% | 0.7776 | all | tok_conf_recency | 0.7101 | 16 | 1.00 | no | balanced |
| 100% | 0.7869 | all | tok_conf_recency | 0.7225 | 16 | 0.20 | no | balanced |

### combined noncoding holdout

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | 71.27% | 94.69% | 10% | 72.21% | 73.89% |
| es_svd_ms_rr_r1 | 92.37% | 99.53% | 10% | 95.71% | 74.63% |
| es_svd_ms_rr_r2 | 93.66% | 99.51% | 10% | 94.88% | 74.26% |

### es_svd_ms_rr_r2 per-cache

| Cache | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| cache/DS-R1/aime24 | 98.73% | 100.00% | 10% | 99.80% | 75.00% |
| cache/DS-R1/aime25 | 93.04% | 100.00% | 10% | 95.67% | 50.00% |
| cache/DS-R1/brumo25 | 98.85% | 100.00% | 10% | 98.94% | 75.00% |
| cache/DS-R1/hmmt25 | 95.35% | 99.23% | 10% | 95.74% | 75.00% |
| cache_train/DS-R1/aime25 | 95.70% | 100.00% | 10% | 96.31% | 75.00% |
| cache_train/DS-R1/brumo25 | 98.32% | 100.00% | 10% | 98.94% | 100.00% |
| cache_train/DS-R1/hmmt25 | 97.22% | 99.62% | 10% | 97.44% | 75.00% |
| cache/DS-R1/gpqa | 81.57% | 97.81% | 10% | 83.81% | 73.33% |
| cache_train/DS-R1/gpqa | 84.14% | 98.91% | 10% | 87.27% | 70.00% |

