# Weight Spectral Fallback Report

## Branch Status

- This line is implemented as a **fallback / parallel branch** for checkpoint-centric ranking.
- It does **not** replace the existing response / activation branch; it only adds a weights-only path.
- Scenario: `math5000rl_qwen3_4b`
- Checkpoints: `11` (base, step-100, step-200, step-300, step-400, step-500, step-600, step-700, step-800, step-900, step-1000)
- `lm_head` handling: `tied-to-embedding`
- Drift extraction mode: `sampled drift / exact spectral sketches`

## Modeling Stack

- Weight drift features cover `delta-to-base`, `delta-to-prev`, cosine-to-reference, per-layer Fro summaries, module drift, sparsity, and sign flips.
- Randomized spectral features come from principal-matrix sketches with `randomized_svd`, random probe responses, and Hutchinson-style quadratic summaries.
- Lightweight modeling uses a pointwise linear ridge head plus a within-scenario pairwise logistic ranking head.
- Score fusion uses pointwise/pairwise blend alpha `=0.5956` and a separate 1-D temporal smoothing pass on the fused trajectory.

## Full-Fit Ranking

- Spearman ρ: `0.8364`
- Pearson r: `0.9693`
- Kendall τ: `0.7455`
- Top-1 hit: `1`
- Top-3 overlap: `2`

| Pred Rank | Checkpoint | Pred Score | True Accuracy |
| --- | --- | --- | --- |
| 1 | step-600 | 0.797045 | 0.332031 |
| 2 | step-500 | 0.657147 | 0.329375 |
| 3 | step-900 | 0.654023 | 0.326406 |
| 4 | step-1000 | 0.530633 | 0.320000 |
| 5 | step-700 | 0.504478 | 0.322969 |
| 6 | step-800 | 0.484678 | 0.322500 |
| 7 | step-400 | 0.437051 | 0.330781 |
| 8 | step-300 | -0.093610 | 0.318437 |
| 9 | step-200 | -0.595387 | 0.312812 |
| 10 | step-100 | -1.196938 | 0.303438 |
| 11 | base | -2.179121 | 0.286094 |

## OOF Quality

- Spearman ρ: `0.6273`
- Pearson r: `0.9074`
- Kendall τ: `0.4545`
- Top-1 hit: `0`
- Top-3 overlap: `1`

## Most Predictive Layers

| Layer | Importance |
| --- | --- |
| 12 | 0.0636 |
| 0 | 0.0625 |
| 10 | 0.0575 |
| 19 | 0.0516 |
| 2 | 0.0502 |
| 23 | 0.0485 |
| 34 | 0.0441 |
| 5 | 0.0396 |
| 14 | 0.0359 |
| 4 | 0.0351 |

## Random Probe Layers

- Later-half checkpoints above median accuracy are treated as `late-good`: `step-500, step-600, step-900`.
- The table below sums importance only over probe-response features.

| Layer | Probe Importance |
| --- | --- |
| 19 | 0.0336 |
| 5 | 0.0210 |
| 31 | 0.0161 |
| 30 | 0.0159 |
| 35 | 0.0145 |
| 33 | 0.0133 |
| 6 | 0.0129 |
| 8 | 0.0114 |
| 10 | 0.0111 |
| 16 | 0.0111 |

## Module Dependence (Math-Heavy Scenario)

- Only a math-heavy RL scenario is locally available, so this section is within-scenario rather than cross-scenario.
- Higher module importance means the weight-only fallback leaned more on that module family.

| Module | Importance |
| --- | --- |
| attention | 0.0166 |
| embedding | 0.0076 |
| lm_head | 0.0076 |
| mlp | 0.0038 |
| norm | 0.0000 |
| other | 0.0000 |

## Delta-to-Prev vs Delta-to-Base

- Subset ablations compare the same lightweight stack under restricted feature families.

| Subset | Selected | Spearman ρ | Pearson r | Kendall τ |
| --- | --- | --- | --- | --- |
| delta_base_only | 97 | 0.5818 | 0.8759 | 0.4182 |
| delta_prev_only | 100 | 0.6364 | 0.8941 | 0.4909 |
| full | 101 | 0.6273 | 0.9074 | 0.4545 |

## Top Features

| Feature | Importance | |Spearman| |
| --- | --- | --- |
| layer_12_delta_prev_sign_flip_ratio | 0.0251 | 0.7727 |
| layer_23_delta_prev_sign_flip_ratio | 0.0229 | 0.6909 |
| layer_34_delta_prev_stable_rank | 0.0205 | 0.7364 |
| layer_24_delta_prev_sign_flip_ratio | 0.0200 | 0.6909 |
| layer_19_delta_prev_probe_std | 0.0189 | 0.7455 |
| layer_00_delta_base_stable_rank | 0.0175 | 0.8818 |
| layer_23_delta_base_sign_flip_ratio | 0.0173 | 0.6788 |
| layer_10_delta_prev_stable_rank | 0.0169 | 0.7182 |
| layer_31_delta_base_probe_std | 0.0161 | 0.6636 |
| layer_30_delta_base_probe_std | 0.0159 | 0.6091 |
| layer_03_delta_prev_stable_rank | 0.0156 | 0.6455 |
| layer_02_delta_prev_stable_rank | 0.0149 | 0.5636 |
| layer_00_delta_prev_sign_flip_ratio | 0.0148 | 0.6545 |
| layer_19_delta_base_probe_std | 0.0147 | 0.7273 |
| layer_35_delta_base_probe_std | 0.0145 | 0.6636 |

## Artifacts

- OOF CSV: `/home/jovyan/work/NAD_Next/outputs/weight_spectral_oof.csv`
- Feature importance CSV: `/home/jovyan/work/NAD_Next/outputs/weight_spectral_feature_importance.csv`
- Report: `/home/jovyan/work/NAD_Next/outputs/weight_spectral_report.md`
