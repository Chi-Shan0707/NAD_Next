# Math DeepSets Round 1

## Math Single-Domain (`profile=main`)

| Selector | AUROC | Hit@1 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| medoid | 56.04% | 70.83% | 68.52% | 72.27% |
| knn-medoid | 57.21% | 73.33% | 72.11% | 71.88% |
| runwise__all_aug__squared_hinge__C0p10__bias__balanced | 85.46% | 74.17% | 73.27% | 96.88% |
| ranksvm__no_logs__squared_hinge__C0p10__bias__mean_margin | 58.90% | 74.17% | 74.49% | 75.26% |
| math_deepsets_round1_mean | 86.01% | 71.67% | 74.05% | 98.96% |
| math_deepsets_round1_max | 86.32% | 75.83% | 72.97% | 99.09% |
| math_deepsets_round1_max_pairaux0p25 | 86.26% | 77.50% | 74.42% | 98.70% |

- Selected DeepSets variant: `math_deepsets_round1_max_pairaux0p25`
- Decision: `Promote`

## Current vs Patched System Proxy

### Sample-weighted

| System | Hit@1 | Hit@3 | Pairwise | SelAcc@10 | AvgRank proxy |
|---|---:|---:|---:|---:|---:|
| current = generic math + code_v2 + science_hybrid_round3 | 67.22% | 74.02% | 61.29% | 66.15% | n/a |
| patched = math_deepsets_round1_max_pairaux0p25 + code_v2 + science_hybrid_round3 | 68.25% | 73.61% | 60.10% | 72.37% | n/a |

### Equal-cache-mean

| System | Hit@1 | Hit@3 | Pairwise | SelAcc@10 | AvgRank proxy |
|---|---:|---:|---:|---:|---:|
| current = generic math + code_v2 + science_hybrid_round3 | 70.53% | 77.71% | 71.53% | 70.90% | n/a |
| patched = math_deepsets_round1_max_pairaux0p25 + code_v2 + science_hybrid_round3 | 73.31% | 76.60% | 68.33% | 87.66% | n/a |

### Gate Read

- Math gate passed: `True`
- Math gate failures: `[]`
- System gate passed: `True`
- System gate failures: `[]`
