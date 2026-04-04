# Extreme12 Aggregated-Objective Comparison

- Generated: 2026-04-03T11:41:58.609096+00:00
- Recommended: `baseline12_pointwise`
- Baseline12: `baseline12_pointwise`
- Best aggregated candidate: `aggregated_selacc10_s044`

| Name | Obj | Seed | Tuple | Rule | Hit@1 | Hit@3 | Pairwise | SelAcc@10 | Inner Tune SelAcc@10 | Guard | Pick | Stats | Trace |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| baseline12_pointwise | pointwise | 42 | 12 | 2-10 | 67.65% | 97.06% | 74.59% | 75.99% | - | - | yes | `results/extreme12_v2_experiments/20260403_105944/models/baseline12_pointwise_stats.json` | - |
| aggregated_selacc10_s042 | aggregated_selacc10 | 42 | 12 | 2-10 | 69.12% | 97.06% | 73.68% | 76.91% | 83.66% | - | - | `results/extreme12_v2_experiments/20260403_105944/models/aggregated_selacc10_s042_stats.json` | `results/extreme12_v2_experiments/20260403_105944/models/aggregated_selacc10_s042_search_trace.json` |
| aggregated_selacc10_s043 | aggregated_selacc10 | 43 | 12 | 2-10 | 66.18% | 97.06% | 73.93% | 75.99% | 81.54% | - | - | `results/extreme12_v2_experiments/20260403_105944/models/aggregated_selacc10_s043_stats.json` | `results/extreme12_v2_experiments/20260403_105944/models/aggregated_selacc10_s043_search_trace.json` |
| aggregated_selacc10_s044 | aggregated_selacc10 | 44 | 12 | 2-10 | 69.12% | 98.53% | 74.44% | 72.65% | 87.00% | yes | - | `results/extreme12_v2_experiments/20260403_105944/models/aggregated_selacc10_s044_stats.json` | `results/extreme12_v2_experiments/20260403_105944/models/aggregated_selacc10_s044_search_trace.json` |
