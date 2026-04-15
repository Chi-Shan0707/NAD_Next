# Low-Rank Structure Evidence

## Setup

- Uses the saved canonical `r1` bundles: `es_svd_math_rr_r1`, `es_svd_science_rr_r1`, `es_svd_ms_rr_r1`, and `es_svd_coding_rr_r1`.
- Reconstructs the exact canonical `raw+rank` route inputs and then applies the frozen route `StandardScaler`; spectra are measured on this standardized pre-SVD matrix.
- Pools rows from the canonical feature stores under the same data sources as the paper route (`MUI_HUB/cache` and `MUI_HUB/cache_train` where available).
- Uses bootstrap units keyed by `dataset + problem_id` across roots, matching the canonical grouped split unit.
- Bootstrap: `500` grouped replicates. Null controls: `12` replicates per control.

## Main Answers

### 1. Do real feature views show sharper spectral decay than nulls?

Yes against the two permutation controls, but not cleanly against the covariance-matched Gaussian control.

- Across all `16/16` domain-anchor settings, the real matrix has **lower effective rank** and **higher top singular-value share** than both permutation nulls.
- The gap is large for those permutation controls: real effective rank is roughly `7.4–9.3`, versus about `23–27` for within-group permutation and about `38` for fully column-wise permutation.
- The Gaussian covariance-matched synthetic features largely reproduce the same spectra, which is expected because they preserve the empirical covariance structure that defines the spectrum.
- So the defensible claim is: the canonical views show a strong low-rank signature relative to permutation-based destruction of structure, but this evidence is mainly **covariance-level** rather than evidence of additional non-Gaussian spectral structure.

### 2. Is coding meaningfully flatter than math/science?

Not as a broad statement.

- Coding is slightly flatter than `math` at matched anchors: coding effective rank is higher than math at `4/4` anchors, with slightly lower top singular-value share.
- But coding is **not** flatter than `science` or pooled `ms`: science/ms have higher effective rank and lower top singular-value share at every matched anchor.
- The mean effective ranks make this clear: `coding=7.90`, `math=7.63`, `science=8.94`, `ms=8.76`.
- So the conservative read is that coding is not a unique “flattest” domain. It sits between math and science/ms rather than clearly above both.

### 3. Are the low-rank signatures stable across anchors and domains?

The low-rank signatures are fairly stable across anchors within each analyzed domain.

- `math`: effective-rank range 7.44–7.96, top-1 variance share 0.338–0.365.
- `science`: effective-rank range 8.57–9.33, top-1 variance share 0.272–0.304.
- `ms`: effective-rank range 8.49–9.12, top-1 variance share 0.291–0.317.
- `coding`: effective-rank range 7.65–8.13, top-1 variance share 0.336–0.360.

This means the anchor-to-anchor drift is modest relative to the gap to the permutation nulls. The cross-domain ordering is also fairly systematic: `math` is the sharpest, `science` the flattest, `ms` tracks between math and science but closer to science, and `coding` stays close to math rather than science.

## Selected Numbers

| Domain | Anchor | Eff. rank | Participation ratio | Stable rank | Top-1 variance share |
|---|---:|---:|---:|---:|---:|
| math | 10% | 7.46 | 4.94 | 2.74 | 0.365 |
| math | 40% | 7.44 | 4.96 | 2.78 | 0.360 |
| math | 70% | 7.67 | 5.14 | 2.86 | 0.350 |
| math | 100% | 7.96 | 5.35 | 2.96 | 0.338 |
| science | 10% | 8.57 | 5.92 | 3.29 | 0.304 |
| science | 40% | 8.84 | 6.17 | 3.44 | 0.290 |
| science | 70% | 9.04 | 6.35 | 3.54 | 0.282 |
| science | 100% | 9.33 | 6.58 | 3.68 | 0.272 |
| ms | 10% | 8.49 | 5.79 | 3.16 | 0.317 |
| ms | 40% | 8.61 | 5.88 | 3.22 | 0.311 |
| ms | 70% | 8.81 | 6.05 | 3.31 | 0.302 |
| ms | 100% | 9.12 | 6.27 | 3.43 | 0.291 |
| coding | 10% | 7.91 | 5.29 | 2.83 | 0.353 |
| coding | 40% | 7.65 | 5.09 | 2.78 | 0.360 |
| coding | 70% | 7.91 | 5.28 | 2.87 | 0.349 |
| coding | 100% | 8.13 | 5.47 | 2.97 | 0.336 |

## Artifacts

- `results/tables/lowrank_structure_spectra.csv`
- `results/tables/lowrank_structure_nulls.csv`
- `results/figures/lowrank_structure/*.png`

Figure files:
- `results/figures/lowrank_structure/scree_math.png`
- `results/figures/lowrank_structure/cumulative_variance_math.png`
- `results/figures/lowrank_structure/scree_science.png`
- `results/figures/lowrank_structure/cumulative_variance_science.png`
- `results/figures/lowrank_structure/scree_ms.png`
- `results/figures/lowrank_structure/cumulative_variance_ms.png`
- `results/figures/lowrank_structure/scree_coding.png`
- `results/figures/lowrank_structure/cumulative_variance_coding.png`
- `results/figures/lowrank_structure/effective_rank_comparison.png`

## Caveats

- This is descriptive evidence about feature organization, not a new predictive evaluation.
- Coding should still be treated as a boundary case: even if some low-rank signature appears, it does not imply the current coding route is strong downstream.
- The spectra are measured after the frozen route scaler, so they reflect the paper-facing representation actually seen by the trained SVD head.
- Within-group permutation preserves cache-local problem groups (`cache_key::problem_id`), while bootstrap grouping follows the split unit (`dataset + problem_id` across roots).

## Cache Notes

- `cache` feature-store status: `loaded_direct_existing`
- `cache_train` feature-store status: `loaded_direct_existing`
- `cache` feature-store path: `/home/jovyan/work/NAD_Next/results/cache/es_svd_ms_rr_r1/cache_all_547b9060debe139e.pkl`
- `cache_train` feature-store path: `/home/jovyan/work/NAD_Next/results/cache/es_svd_ms_rr_r1/cache_train_all_d429f3b93baed972.pkl`
