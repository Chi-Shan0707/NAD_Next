# GPQA DeepSets Round 1

## GPQA Single-Domain

| Selector | AUROC | Hit@1 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| science_baseline_v1 | 53.86% | 66.16% | 58.71% | 64.35% |
| gpqa_pairwise_round1 | 55.66% | 63.64% | 61.15% | 62.38% |
| science_hybrid_round3 | 53.86% | 68.18% | 58.72% | 64.35% |
| gpqa_deepsets_round1_mean | 65.62% | 61.62% | 58.61% | 85.73% |
| gpqa_deepsets_round1_max | 65.61% | 63.13% | 58.10% | 66.56% |

- Selected DeepSets variant: `gpqa_deepsets_round1_max`
- Decision: `No-Promote`

## Current vs Patched System Proxy

### Sample-weighted

| System | Hit@1 | Hit@3 | Pairwise | SelAcc@10 | AvgRank proxy |
|---|---:|---:|---:|---:|---:|
| current = code_v2 + science_hybrid_round3 | 67.22% | 74.02% | 61.29% | 66.15% | n/a |
| patched = code_v2 + gpqa_deepsets_round1_max | 65.16% | 75.26% | 61.04% | 67.05% | n/a |

### Equal-cache-mean

| System | Hit@1 | Hit@3 | Pairwise | SelAcc@10 | AvgRank proxy |
|---|---:|---:|---:|---:|---:|
| current = code_v2 + science_hybrid_round3 | 70.53% | 77.71% | 71.53% | 70.90% | n/a |
| patched = code_v2 + gpqa_deepsets_round1_max | 69.69% | 78.21% | 71.43% | 71.27% | n/a |

### Patched delta vs current

| Patch | sample-weighted Hit@1 | sample-weighted SelAcc@10 | sample-weighted AvgRank proxy | equal-cache Hit@1 | equal-cache SelAcc@10 | Actual leaderboard delta |
|---|---:|---:|---:|---:|---:|---|
| gpqa_deepsets_round1_max | -2.06% | 0.90% | -0.0474 | -0.84% | 0.37% | unavailable in repo |

## Offline–Online Alignment

| Patch | sample-weighted Hit@1 | sample-weighted SelAcc@10 | sample-weighted AvgRank proxy | equal-cache Hit@1 | equal-cache SelAcc@10 | Actual leaderboard delta |
|---|---:|---:|---:|---:|---:|---|
| code_v2 | 0.83% | 1.36% | +0.0000 | 0.40% | 0.66% | unavailable in repo |
| science_baseline_v1 | 2.27% | 0.56% | +0.0000 | 0.93% | 0.23% | unavailable in repo |
| science_hybrid_round3 | 0.82% | 0.00% | -0.0082 | 0.34% | 0.00% | unavailable in repo |
| gpqa_deepsets_round1 | -2.06% | 0.90% | -0.0474 | -0.84% | 0.37% | unavailable in repo |

- Read: sample-weighted `Hit@1` plus `avg_rank_proxy` is the closest thing to the repo's current promote-sensitive direction.
- Warning: the repo still does not contain a trustworthy leaderboard delta table, so proxy/leaderboard alignment remains unverified.
