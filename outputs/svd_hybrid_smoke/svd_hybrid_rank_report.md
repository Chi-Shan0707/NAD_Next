# SVD Hybrid RL Checkpoint Ranking

## Summary
- Data: `results/cache/export_rl_checkpoint_ranking_from_svd_models/feature_store_all_ref030_05edbff25fbfa65b.pkl` with 100 scenarios × 11 checkpoints × 64 runs.
- Activation SVD bundle: `models/ml_selectors/es_svd_math_rr_r1.pkl` (slot-100 math route latent appended when requested).
- Weight prior frame: `outputs/weight_spectral_feature_frame.csv`.
- OOF protocol: 5-fold GroupKFold by scenario, with one grouped inner split for config selection.
- Local slot-100 mean-confidence baseline: ρ=0.573, τ=0.418.
- Best SVD head: `svd_weight_prior_route_latent_weak_monotone` (ρ=0.673, τ=0.491, RMSE=0.325).

## OOF Metrics
| Candidate | Family | Base | Prior | Spearman | Kendall | Pearson | Top1 | Top3 | RMSE | ECE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `svd_weight_prior_route_latent_weak_monotone` | hybrid | `raw_route_latent` | 1 | 0.673 | 0.491 | 0.866 | 0 | 1 | 0.325 | 0.046 |
| `svd_weight_prior_route_latent_multi_objective` | hybrid | `raw_route_latent` | 1 | 0.645 | 0.491 | 0.887 | 0 | 1 | 0.325 | 0.031 |
| `svd_weight_prior_smooth` | hybrid | `raw` | 1 | 0.609 | 0.455 | 0.804 | 0 | 1 | 0.324 | 0.025 |
| `smooth_route_latent` | activation_svd | `raw_route_latent` | 0 | 0.600 | 0.491 | 0.847 | 0 | 0 | 0.325 | 0.031 |
| `smooth_raw_baseline` | baseline | `raw` | 0 | 0.591 | 0.491 | 0.831 | 0 | 0 | 0.324 | 0.018 |
| `svd_weight_prior_route_latent_smooth` | hybrid | `raw_route_latent` | 1 | 0.573 | 0.418 | 0.806 | 0 | 1 | 0.325 | 0.029 |

## Candidate Notes
- `svd_weight_prior_route_latent_weak_monotone`: Activation-SVD route latent plus weight-SVD checkpoint prior under weak-monotone regularization. Representative config `{"reg_lambda": 0.001, "w_cal": 1.0, "w_rank": 0.0, "w_run": 0.5, "w_smooth": 0.1}`; predicted order `step-400 > step-800 > step-700 > step-900 > step-1000 > step-500 > step-600 > step-300 > step-100 > step-200 > base`; mean selected prior features `101.0`; mean prior blend alpha `0.524`.
- `svd_weight_prior_route_latent_multi_objective`: Activation-SVD route latent plus weight-SVD checkpoint prior with run/rank/calibration multitask loss. Representative config `{"reg_lambda": 0.001, "w_cal": 1.0, "w_rank": 0.5, "w_run": 0.5, "w_smooth": 0.25}`; predicted order `step-800 > step-400 > step-700 > step-900 > step-1000 > step-500 > step-600 > step-300 > step-200 > step-100 > base`; mean selected prior features `101.0`; mean prior blend alpha `0.524`.
- `svd_weight_prior_smooth`: Raw response features plus checkpoint-level SVD weight prior. Representative config `{"reg_lambda": 0.001, "w_cal": 1.0, "w_rank": 0.0, "w_run": 0.5, "w_smooth": 0.25}`; predicted order `step-400 > step-800 > step-700 > step-900 > step-500 > step-1000 > step-100 > step-600 > step-300 > step-200 > base`; mean selected prior features `101.0`; mean prior blend alpha `0.524`.
- `smooth_route_latent`: Raw response features plus slot-100 route score and route latent coordinates. Representative config `{"reg_lambda": 0.001, "w_cal": 0.5, "w_rank": 0.0, "w_run": 1.0, "w_smooth": 0.1}`; predicted order `step-900 > step-700 > step-1000 > step-800 > step-400 > step-600 > step-500 > step-300 > step-200 > step-100 > base`; mean selected prior features `0.0`.
- `smooth_raw_baseline`: Raw response features with trajectory L2 smoothing only. Representative config `{"reg_lambda": 0.001, "w_cal": 0.5, "w_rank": 0.0, "w_run": 1.0, "w_smooth": 0.1}`; predicted order `step-900 > step-1000 > step-700 > step-800 > step-600 > step-400 > step-500 > step-300 > step-200 > step-100 > base`; mean selected prior features `0.0`.
- `svd_weight_prior_route_latent_smooth`: Activation-SVD route latent plus weight-SVD checkpoint prior with smoothness regularization. Representative config `{"reg_lambda": 0.001, "w_cal": 1.0, "w_rank": 0.0, "w_run": 0.5, "w_smooth": 0.25}`; predicted order `step-400 > step-800 > step-700 > step-900 > step-1000 > step-500 > step-100 > step-600 > step-300 > step-200 > base`; mean selected prior features `101.0`; mean prior blend alpha `0.524`.

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

## Recommendation
- Prefer `svd_weight_prior_route_latent_weak_monotone` for the final ensemble: it gives the strongest checkpoint-order OOF among the tested SVD heads.
- The gain comes from combining two complementary SVD views: run-level activation route latent and checkpoint-level weight spectral prior.
- It stays lightweight, linear, and interpretable: every added signal is either a route latent coordinate or a checkpoint-global linear prior from weight drift spectra.
