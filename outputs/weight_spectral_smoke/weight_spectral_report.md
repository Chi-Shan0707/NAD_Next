# Weight Spectral Fallback Report

## Branch Status

- This line is implemented as a **fallback / parallel branch** for checkpoint-centric ranking.
- It does **not** replace the existing response / activation branch; it only adds a weights-only path.
- Scenario: `math5000rl_qwen3_4b`
- Checkpoints: `11` (base, step-100, step-200, step-300, step-400, step-500, step-600, step-700, step-800, step-900, step-1000)
- `lm_head` handling: `tied-to-embedding`

## Modeling Stack

- Weight drift features cover `delta-to-base`, `delta-to-prev`, cosine-to-reference, per-layer Fro summaries, module drift, sparsity, and sign flips.
- Randomized spectral features come from principal-matrix sketches with `randomized_svd`, random probe responses, and Hutchinson-style quadratic summaries.
- Lightweight modeling uses a pointwise linear ridge head plus a within-scenario pairwise logistic ranking head.
- Score fusion uses pointwise/pairwise blend alpha `=0.5677` and a separate 1-D temporal smoothing pass on the fused trajectory.

## Full-Fit Ranking

- Spearman ρ: `0.8364`
- Pearson r: `0.9722`
- Kendall τ: `0.7455`
- Top-1 hit: `1`
- Top-3 overlap: `2`

| Pred Rank | Checkpoint | Pred Score | True Accuracy |
| --- | --- | --- | --- |
| 1 | step-600 | 0.794012 | 0.332031 |
| 2 | step-500 | 0.670759 | 0.329375 |
| 3 | step-900 | 0.648758 | 0.326406 |
| 4 | step-1000 | 0.524877 | 0.320000 |
| 5 | step-700 | 0.483017 | 0.322969 |
| 6 | step-800 | 0.459991 | 0.322500 |
| 7 | step-400 | 0.445499 | 0.330781 |
| 8 | step-300 | -0.062039 | 0.318437 |
| 9 | step-200 | -0.586920 | 0.312812 |
| 10 | step-100 | -1.167735 | 0.303438 |
| 11 | base | -2.210218 | 0.286094 |

## OOF Quality

- Spearman ρ: `0.5455`
- Pearson r: `0.8798`
- Kendall τ: `0.3818`
- Top-1 hit: `0`
- Top-3 overlap: `0`

## Most Predictive Layers

| Layer | Importance |
| --- | --- |
| 12 | 0.0676 |
| 10 | 0.0627 |
| 5 | 0.0554 |
| 34 | 0.0544 |
| 19 | 0.0533 |
| 0 | 0.0522 |
| 23 | 0.0474 |
| 4 | 0.0435 |
| 2 | 0.0424 |
| 16 | 0.0362 |

## Random Probe Layers

- Later-half checkpoints above median accuracy are treated as `late-good`: `step-500, step-600, step-900`.
- The table below sums importance only over probe-response features.

| Layer | Probe Importance |
| --- | --- |
| 19 | 0.0412 |
| 5 | 0.0293 |
| 30 | 0.0210 |
| 31 | 0.0208 |
| 35 | 0.0193 |
| 33 | 0.0182 |
| 16 | 0.0149 |
| 10 | 0.0147 |
| 21 | 0.0138 |
| 7 | 0.0125 |

## Module Dependence (Math-Heavy Scenario)

- Only a math-heavy RL scenario is locally available, so this section is within-scenario rather than cross-scenario.
- Higher module importance means the weight-only fallback leaned more on that module family.

| Module | Importance |
| --- | --- |
| attention | 0.0140 |
| embedding | 0.0101 |
| lm | 0.0101 |
| mlp | 0.0055 |
| norm | 0.0000 |
| other | 0.0000 |

## Delta-to-Prev vs Delta-to-Base

- Subset ablations compare the same lightweight stack under restricted feature families.

| Subset | Selected | Spearman ρ | Pearson r | Kendall τ |
| --- | --- | --- | --- | --- |
| delta_base_only | 65 | 0.5818 | 0.9089 | 0.4182 |
| delta_prev_only | 68 | 0.5455 | 0.8798 | 0.3818 |
| full | 69 | 0.5455 | 0.8798 | 0.3818 |

## Top Features

| Feature | Importance | |Spearman| |
| --- | --- | --- |
| layer_12_delta_prev_sign_flip_ratio | 0.0302 | 0.7727 |
| layer_34_delta_prev_stable_rank | 0.0255 | 0.7364 |
| layer_23_delta_prev_sign_flip_ratio | 0.0254 | 0.6909 |
| layer_24_delta_prev_sign_flip_ratio | 0.0244 | 0.6909 |
| layer_19_delta_prev_probe_std | 0.0230 | 0.7455 |
| layer_23_delta_base_sign_flip_ratio | 0.0220 | 0.6788 |
| layer_30_delta_base_probe_std | 0.0210 | 0.6091 |
| layer_00_delta_base_stable_rank | 0.0209 | 0.8818 |
| layer_10_delta_prev_stable_rank | 0.0208 | 0.7182 |
| layer_31_delta_base_probe_std | 0.0208 | 0.6636 |
| layer_35_delta_base_probe_std | 0.0193 | 0.6636 |
| layer_03_delta_prev_stable_rank | 0.0193 | 0.6455 |
| layer_19_delta_base_probe_std | 0.0183 | 0.7273 |
| layer_33_delta_base_probe_std | 0.0182 | 0.6182 |
| layer_00_delta_prev_sign_flip_ratio | 0.0178 | 0.6545 |

## Artifacts

- OOF CSV: `outputs/weight_spectral_smoke/weight_spectral_oof.csv`
- Feature importance CSV: `outputs/weight_spectral_smoke/weight_spectral_feature_importance.csv`
- Report: `outputs/weight_spectral_smoke/weight_spectral_report.md`
