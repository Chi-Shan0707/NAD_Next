# Linear Heads for Checkpoint Ranking
## Summary
- Data: `results/cache/export_rl_checkpoint_ranking_from_svd_models/feature_store_all_ref030_05edbff25fbfa65b.pkl` with 100 scenarios × 11 checkpoints × 64 runs.
- OOF protocol: 5-fold GroupKFold by scenario_id; hyperparameters chosen with a grouped inner validation split on the training scenarios.
- Primary decision rule: checkpoint ranking first, calibration second.
- Recommended ensemble head: `smoothness_regularized_linear` (ρ=0.591, τ=0.491, RMSE=0.327).

## OOF Metrics
| Head | Family | Spearman | Kendall | Pearson | Top1 | Top3 | RMSE | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `smoothness_regularized_linear` | trajectory | 0.591 | 0.491 | 0.835 | 0 | 0 | 0.327 | 0.107 | 0.017 |
| `elastic_net` | pointwise | 0.582 | 0.455 | 0.832 | 0 | 0 | 0.324 | 0.105 | 0.025 |
| `weak_monotone_linear` | trajectory | 0.582 | 0.455 | 0.826 | 0 | 0 | 0.324 | 0.105 | 0.024 |
| `multi_objective_linear` | multi_objective | 0.582 | 0.455 | 0.823 | 0 | 0 | 0.325 | 0.105 | 0.017 |
| `pairwise_bce_checkpoint_pairs` | pairwise | 0.582 | 0.455 | 0.803 | 0 | 0 | 0.325 | 0.106 | 0.044 |
| `fused_lasso_tv_linear` | trajectory | 0.582 | 0.455 | 0.836 | 0 | 0 | 0.329 | 0.108 | 0.020 |
| `bradley_terry_pairwise_logistic` | pairwise | 0.582 | 0.455 | 0.796 | 0 | 0 | 0.372 | 0.139 | 0.184 |
| `isotonic_calibrator` | ordinal | 0.573 | 0.418 | 0.777 | 0 | 0 | 0.330 | 0.109 | 0.052 |
| `affine_plus_isotonic` | ordinal | 0.573 | 0.418 | 0.777 | 0 | 0 | 0.330 | 0.109 | 0.052 |
| `pairwise_hinge` | pairwise | 0.573 | 0.418 | 0.779 | 0 | 0 | 0.367 | 0.135 | 0.183 |
| `ranksvm` | pairwise | 0.573 | 0.418 | 0.797 | 0 | 0 | 0.375 | 0.140 | 0.183 |
| `cumulative_link_ordinal_logit` | ordinal | 0.527 | 0.382 | 0.795 | 0 | 0 | 0.327 | 0.107 | 0.026 |
| `logistic_regression` | pointwise | 0.518 | 0.345 | 0.799 | 0 | 0 | 0.323 | 0.104 | 0.030 |
| `linear_svr` | pointwise | 0.055 | 0.055 | -0.190 | 0 | 0 | 0.428 | 0.183 | 0.269 |
| `ridge_regression` | pointwise | 0.045 | 0.091 | 0.719 | 0 | 0 | 0.322 | 0.104 | 0.019 |
| `huber_regression` | pointwise | -0.200 | -0.200 | 0.530 | 0 | 0 | 0.446 | 0.199 | 0.275 |

## Calibration / Ranking Notes
- `smoothness_regularized_linear`: predicted order `step-900 > step-1000 > step-700 > step-800 > step-600 > step-400 > step-500 > step-300 > step-200 > step-100 > base`; representative config `{"reg_lambda": 0.001, "w_smooth": 0.5}`.
- `elastic_net`: predicted order `step-900 > step-1000 > step-800 > step-700 > step-600 > step-400 > step-500 > step-300 > step-200 > step-100 > base`; representative config `{"alpha": 0.001, "l1_ratio": 0.8}`.
- `weak_monotone_linear`: predicted order `step-900 > step-1000 > step-800 > step-700 > step-600 > step-400 > step-500 > step-300 > step-200 > step-100 > base`; representative config `{"reg_lambda": 0.001, "w_smooth": 0.1}`.
- `multi_objective_linear`: predicted order `step-900 > step-1000 > step-800 > step-700 > step-600 > step-400 > step-500 > step-300 > step-200 > step-100 > base`; representative config `{"reg_lambda": 0.001, "w_cal": 0.5, "w_rank": 0.5, "w_run": 1.0, "w_smooth": 0.1}`.
- `pairwise_bce_checkpoint_pairs`: predicted order `step-900 > step-1000 > step-800 > step-700 > step-600 > step-400 > step-500 > step-300 > step-200 > step-100 > base`; representative config `{"reg_lambda": 0.001, "w_cal": 0.0, "w_rank": 1.0, "w_run": 0.25}`.
- `fused_lasso_tv_linear`: predicted order `step-900 > step-1000 > step-700 > step-800 > step-400 > step-600 > step-500 > step-300 > step-200 > step-100 > base`; representative config `{"reg_lambda": 0.001, "tv_eps": 0.0001, "w_smooth": 0.02}`.

## True Checkpoint Accuracy
| Checkpoint | True accuracy |
| --- | ---: |
| `base` | 0.2861 |
| `step-100` | 0.3034 |
| `step-200` | 0.3128 |
| `step-300` | 0.3184 |
| `step-400` | 0.3308 |
| `step-500` | 0.3294 |
| `step-600` | 0.3320 |
| `step-700` | 0.3230 |
| `step-800` | 0.3225 |
| `step-900` | 0.3264 |
| `step-1000` | 0.3200 |

## Why this winner
- `smoothness_regularized_linear` ranks checkpoints best under the primary OOF criterion and stays competitive on calibration.
- It remains lightweight, linear or near-linear, and keeps the feature pipeline frozen.
- It is therefore the safest candidate to add into the final ensemble as the checkpoint-accuracy specialist head.
