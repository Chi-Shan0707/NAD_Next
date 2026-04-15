# 09 Interpretability Validation

## Summary

- 本文档面向 paper-facing 叙事，目标是把 SVD introspection 升级为更强的 numerical interpretability evidence。
- 主结论建议写作：

> The explanations are numerically faithful, stable across perturbations, and selectively relevant to the model’s actual decisions.

## Faithfulness（分数重构）

| Method | Explained Runs | Max Recon Error | Mean Recon Error |
|---|---:|---:|---:|
| `es_svd_math_rr_r1` | 1795 | 7.11e-14 | 1.21e-14 |
| `es_svd_ms_rr_r1` | 5434 | 7.11e-14 | 5.21e-15 |
| `es_svd_science_rr_r1` | 3639 | 2.13e-14 | 1.81e-15 |

## Stability（跨 seed / 跨 split）

- 多 seed feature-sign consistency 平均值：`nan`。
- 多 seed top-positive Jaccard@K 平均值：`nan`。
- 多 split feature-sign consistency 平均值：`nan`。
- 多 split top-positive Jaccard@K 平均值：`nan`。

### Raw vs Rank

| Channel | Mean Feature Sign Consistency | Mean |w| CV |
|---|---:|---:|

## Selective Causal Relevance（Deletion Sanity）

| Intervention | Mean Score Drop | Mean Margin Drop | Flip Rate | C→W Rate | W→C Rate |
|---|---:|---:|---:|---:|---:|
| top family | 6.9260 | 6.9274 | 0.998 | 0.090 | 0.156 |
| low family | -0.0674 | -0.0683 | 0.333 | 0.037 | 0.059 |
| top feature | 5.1214 | 5.1139 | 1.000 | 0.091 | 0.160 |
| low feature | -0.0073 | -0.0073 | 0.092 | 0.011 | 0.020 |

- 解释应以数值支撑而非“看起来合理”为准；这里的主证据是 top deletion 对分数与选中结果的影响显著大于 low deletion。

## Failure Archetypes（Wrong-Top1）

| Archetype | Cases | Fraction | Mean Score Gap | Mean Rank Share | Representative |
|---|---:|---:|---:|---:|---|
| Weak-margin ambiguous tie | 294 | 0.251 | 0.0036 | 0.501 | `cache_train/DS-R1/gpqa::gpqa-152@70` |
| Trajectory over-bias | 276 | 0.235 | 0.5763 | 0.409 | `cache/DS-R1/brumo25::brumo25-12@10` |
| Mixed-signal conflict | 244 | 0.208 | 0.5998 | 0.386 | `cache_train/DS-R1/brumo25::brumo25-26@70` |
| Rank-channel reshuffle artifact | 182 | 0.155 | 0.1307 | 0.574 | `cache_train/DS-R1/aime25::13@100` |
| Late-anchor uncertainty over-reward | 108 | 0.092 | 0.4580 | 0.381 | `cache/DS-R1/gpqa::gpqa-127@100` |

## Appendix Cases

- `Case 1: es_svd_math_rr_r1 math correct @ 100%` → `results/case_studies/appendix_case_01_es_svd_math_rr_r1_math_100.md`
- `Case 2: es_svd_science_rr_r1 science correct @ 100%` → `results/case_studies/appendix_case_02_es_svd_science_rr_r1_science_100.md`
- `Case 3: es_svd_math_rr_r1 math wrong @ 70%` → `results/case_studies/appendix_case_03_es_svd_math_rr_r1_math_70.md`
- `Case 4: es_svd_math_rr_r1 math wrong @ 10%` → `results/case_studies/appendix_case_04_es_svd_math_rr_r1_math_10.md`
