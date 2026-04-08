# Code DeepSets Round 1

## Coding Single-Domain

| Selector | AUROC | Hit@1 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| code_baseline_v1 | 50.27% | 59.28% | 51.27% | 61.74% |
| code_v2 | 50.29% | 61.68% | 51.00% | 62.11% |
| code_deepsets_round1_mean | 48.79% | 58.68% | 50.12% | 58.65% |
| code_deepsets_round1_max | 52.47% | 62.28% | 50.29% | 55.75% |
| code_deepsets_round1_max_pairaux0p25 | 51.67% | 59.28% | 49.08% | 59.96% |

- Selected DeepSets variant: `code_deepsets_round1_max`
- Decision: `No-Promote`

## Current vs Patched System Proxy

### Sample-weighted

| System | Hit@1 | Hit@3 | Pairwise | SelAcc@10 | AvgRank proxy |
|---|---:|---:|---:|---:|---:|
| current = code_v2 + science_hybrid_round3 | 67.22% | 74.02% | 61.29% | 66.15% | n/a |
| patched = code_deepsets_round1_max + science_hybrid_round3 | 67.42% | 74.23% | 61.05% | 63.96% | n/a |

### Equal-cache-mean

| System | Hit@1 | Hit@3 | Pairwise | SelAcc@10 | AvgRank proxy |
|---|---:|---:|---:|---:|---:|
| current = code_v2 + science_hybrid_round3 | 70.53% | 77.71% | 71.53% | 70.90% | n/a |
| patched = code_deepsets_round1_max + science_hybrid_round3 | 70.63% | 77.81% | 71.42% | 69.84% | n/a |

### Gate Read

- Coding gate passed: `False`
- Coding gate failures: `['SelAcc@10 below current code_v2']`
- System gate passed: `False`
- System gate failures: `['sample_weighted Hit@1/SelAcc@10 did not produce a guarded improvement']`
