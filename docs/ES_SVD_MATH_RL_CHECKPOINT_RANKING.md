# ES SVD MATH RL CHECKPOINT RANKING

## Scope

- `task`：`checkpoint_ranking`
- `method_name`：`es_svd_math_rr_r1__math5000rl_slot100_meanconf`
- `source model`：`models/ml_selectors/es_svd_math_rr_r1.pkl`
- `domain`：`math` only
- `slot`：`100%` (`slot_index=9`)
- `score aggregation`：`mean_confidence`

## Submission Artifact

- `submission json`：`submission/CheckpointRanking/es_svd_math_rr_r1/es_svd_math_rr_r1__math5000rl_slot100_meanconf.json`
- `local eval json`：`results/scans/checkpoint_ranking/es_svd_math_rr_r1/es_svd_math_rr_r1__math5000rl_slot100_meanconf_eval.json`
- `comparison json`：`results/scans/checkpoint_ranking/es_svd_math_rr_r1/rl_checkpoint_ranking_svd_model_comparison.json`

## Score Meaning

- `scores[checkpoint_name] = float`：该 checkpoint 在 `math` RL cache 上的 `slot100` 样本级 SVD 分数再做 `mean_confidence` 聚合后的结果。
- 分数越高，表示该 checkpoint 被模型判断为整体质量越高。
- 排行任务只使用相对顺序；绝对数值本身不是 accuracy，也不是 calibrated probability。

## Submitted Scores

| Checkpoint | Submitted score |
|---|---:|
| `base` | 21.899924 |
| `step-100` | 23.586100 |
| `step-200` | 23.720322 |
| `step-300` | 23.990241 |
| `step-400` | 24.479607 |
| `step-500` | 24.774351 |
| `step-600` | 25.201527 |
| `step-700` | 25.797658 |
| `step-800` | 25.970956 |
| `step-900` | 26.391530 |
| `step-1000` | 26.286716 |

## Predicted Order

1. `step-900`
2. `step-1000`
3. `step-800`
4. `step-700`
5. `step-600`
6. `step-500`
7. `step-400`
8. `step-300`
9. `step-200`
10. `step-100`
11. `base`

## Offline Local Eval

- `spearman_rho`：`0.5727`
- `pearson_r`：`0.7633`
- `kendall_tau`：`0.4182`
- `top1_hit`：`0`
- `top3_hit`：`0`

说明：

- 这里是本地离线复现实验结果，来源于仓库内 `eval json`。
- 与线上 leaderboard 分数不是同一个来源，不能混为一谈；线上结果以下一节为准。

## Leaderboard Result

- `user`：`drfit`
- `rank`：`3`
- `submitted at`：`2026-04-10 12:45`
- `method_name`：`es_svd_math_rr_r1__math5000rl_slot100_meanconf`
- `spearman_rho`：`0.7364`
- `pearson_r`：`0.8398`
- `kendall_tau`：`0.6000`
- `top1_hit`：`0`
- `top3_hit`：`0`

## Leaderboard Comparison

| Rank | User | Method | Spearman ρ | Pearson r | Kendall τ | Top-1 | Top-3 | Submitted |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `yyc` | `math_bon_checkpoint_v1` | 0.7818 | 0.8338 | 0.6000 | 1 | 2 | 2026-04-10 12:45 |
| 2 | `mean_confidence_baseline` | `mean_confidence` | 0.7364 | 0.8577 | 0.6000 | 0 | 0 | 2026-03-31 03:40 |
| 3 | `drfit` | `es_svd_math_rr_r1__math5000rl_slot100_meanconf` | 0.7364 | 0.8398 | 0.6000 | 0 | 0 | 2026-04-10 12:45 |

## Interpretation

- 这版 `es_svd_math_rr_r1` 在线上 `Checkpoint Ranking` 进入第 `3` 名。
- 它与 `mean_confidence` baseline 的 `Spearman ρ` 和 `Kendall τ` 持平，但 `Pearson r` 更低。
- `Top-1` 与 `Top-3` 都是 `0`，说明排序相关性还可以，但没有准确抓到真实最优 checkpoint。
