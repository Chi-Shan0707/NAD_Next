# Axis Intervention Evidence

This note tests whether the learned SVD axes are operational decision objects, not just arbitrary rotations.

## Setup

- Route selection uses the best holdout anchor for each paper-facing route by pooled holdout AUROC, with SelAcc@10 as the tie-breaker.
- Per-axis contribution is `c_k(x) = alpha_eff,k * z_k(x)`, where `alpha_eff = beta / s` for whitened routes and `alpha_eff = beta` otherwise.
- Random matched-count ablations average over `32` draws per `k`.
- Top/bottom ablations rank axes per run by absolute contribution and then zero those latent contributions before rescoring.
- Perturbation uses small aligned steps in route-input space (`x_rep`) along the sign-aligned SVD direction, then rescoring through the original model.

## Selected Routes

| Route | Method | Best Anchor | Rank Spec | Holdout AUROC | Holdout SelAcc@10 | Top-1 Abs Mass | Top-2 Abs Mass | Top-4 Abs Mass |
|---|---|---:|---|---:|---:|---:|---:|---:|
| math | `es_svd_math_rr_r1` | 100 | `math:16` | 0.9813 | 0.9944 | 0.299 | 0.476 | 0.707 |
| science | `es_svd_science_rr_r1` | 100 | `science:16` | 0.8711 | 1.0000 | 0.302 | 0.486 | 0.706 |
| ms | `es_svd_ms_rr_r1` | 100 | `math:16 / science:16` | 0.9177 | 0.9982 | 0.301 | 0.483 | 0.706 |
| coding | `es_svd_coding_rr_r1` | 10 | `coding:16` | 0.5530 | 0.5500 | 0.262 | 0.437 | 0.666 |

The mass numbers above come directly from per-run axis contributions and show that the selected top run is usually supported by a concentrated subset of latent axes.

## Main Intervention Table (`k=2`)

| Route | Policy | Mean Top1 Score Drop | AUROC Drop | SelAcc Drop | Top1 Flip Rate | Decision Flip Rate | Removed Abs Mass |
|---|---|---:|---:|---:|---:|---:|---:|
| `math@100%` | top | 2.7601 | 0.2748 | 0.0889 | 0.929 | 0.500 | 0.476 |
| `math@100%` | random | 0.3023 | 0.0286 | -0.0014 | 0.837 | 0.455 | 0.131 |
| `math@100%` | bottom | -0.0012 | 0.0000 | 0.0000 | 0.036 | 0.036 | 0.002 |
| `science@100%` | top | 0.8613 | 0.1882 | 0.0599 | 0.967 | 0.533 | 0.486 |
| `science@100%` | random | 0.1128 | 0.0268 | 0.0037 | 0.796 | 0.400 | 0.124 |
| `science@100%` | bottom | 0.0040 | 0.0000 | 0.0000 | 0.017 | 0.017 | 0.003 |
| `ms@100%` | top | 1.4655 | 0.2473 | 0.1348 | 0.955 | 0.523 | 0.483 |
| `ms@100%` | random | 0.1447 | 0.0320 | 0.0083 | 0.809 | 0.413 | 0.123 |
| `ms@100%` | bottom | 0.0023 | 0.0000 | 0.0000 | 0.023 | 0.023 | 0.003 |
| `coding@10%` | top | 0.0076 | -0.0185 | -0.0375 | 1.000 | 0.440 | 0.437 |
| `coding@10%` | random | 0.0012 | 0.0005 | -0.0096 | 0.732 | 0.360 | 0.128 |
| `coding@10%` | bottom | 0.0000 | 0.0001 | -0.0125 | 0.080 | 0.080 | 0.006 |

## Perturbation Sanity

| Route | Selected Axis Subset | Monotonic Fraction | Mean Δscore @ ε=-0.2 | Mean Δscore @ ε=+0.2 |
|---|---|---:|---:|---:|
| `math@100%` | positive contribution axis | 1.000 | -0.2957 | 0.2957 |
| `math@100%` | negative contribution axis | 1.000 | -0.3022 | 0.3022 |
| `science@100%` | positive contribution axis | 1.000 | -0.0603 | 0.0603 |
| `science@100%` | negative contribution axis | 1.000 | -0.0807 | 0.0807 |
| `ms@100%` | positive contribution axis | 1.000 | -0.1352 | 0.1352 |
| `ms@100%` | negative contribution axis | 1.000 | -0.1512 | 0.1512 |
| `coding@10%` | positive contribution axis | 1.000 | -0.0016 | 0.0016 |
| `coding@10%` | negative contribution axis | 1.000 | -0.0011 | 0.0011 |

## Figures

- `results/figures/axis_intervention/math_route.png`
- `results/figures/axis_intervention/science_route.png`
- `results/figures/axis_intervention/ms_route.png`
- `results/figures/axis_intervention/coding_route.png`
- `results/figures/axis_intervention/perturbation_monotonicity.png`

## Takeaways

- **Reading note**: top1-flip can be noisy because random ablations often reshuffle near-tied high-scoring runs; AUROC/SelAcc drops and pairwise decision flips are the cleaner route-level signal.
- **math**: top-2 axis removal drops AUROC by `0.2748` vs random `0.0286` and bottom `0.0000`; removed mass is `0.476` vs `0.131` vs `0.002`.
- **science**: top-2 axis removal drops AUROC by `0.1882` vs random `0.0268` and bottom `0.0000`; removed mass is `0.486` vs `0.124` vs `0.003`.
- **ms**: top-2 axis removal drops AUROC by `0.2473` vs random `0.0320` and bottom `0.0000`; removed mass is `0.483` vs `0.123` vs `0.003`.
- **coding**: this remains a boundary case. The best coding route is weak, perturbations are still monotonic, but top-axis ablation does not yield the same clean AUROC/SelAcc degradation seen in noncoding.
- **Overall**: in `math`, `science`, and `ms`, top-contribution axes matter substantially more than matched random axes on the main metrics, while bottom axes are close to a negative control.
- **Boundary case**: `coding` stays useful as a failure/ablation slice, but it should not be used as the main axis-level evidence because the route itself is weak at holdout.
- **Interpretation**: this does not prove a unique canonical basis, but it does show that the learned axes carry concentrated causal signal for the deployed decision rule.
