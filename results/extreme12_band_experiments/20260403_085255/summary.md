# Extreme12 Band-Reward Comparison

- Generated: 2026-04-03T09:19:08.312672+00:00
- Recommended: `baseline12_pointwise`
- Baseline12: `baseline12_pointwise`
- Best band candidate: `band12_b025_g025m`

| Name | Obj | Tuple | Rule | β | γ | Hit@1 | Hit@3 | Pairwise | SelAcc@10 | Mix SelAcc@10 | Guard | Pick | Stats |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| baseline8_pointwise | pointwise | 8 | blind | 0.35 | -0.50 | 69.12% | 97.06% | 74.59% | 75.99% | 75.99% | - | - | `results/extreme12_band_experiments/20260403_085255/models/baseline8_pointwise_stats.json` |
| baseline12_pointwise | pointwise | 12 | 2-10 | 0.35 | -0.50 | 67.65% | 97.06% | 74.54% | 75.99% | 75.99% | - | yes | `results/extreme12_band_experiments/20260403_085255/models/baseline12_pointwise_stats.json` |
| band12_b025_g025m | band_reward | 12 | 2-10 | 0.25 | -0.25 | 69.12% | 98.53% | 74.22% | 72.88% | 72.88% | yes | - | `results/extreme12_band_experiments/20260403_085255/models/band12_b025_g025m_stats.json` |
| band12_b025_g050m | band_reward | 12 | 2-10 | 0.25 | -0.50 | 69.12% | 98.53% | 74.24% | 72.65% | 72.65% | yes | - | `results/extreme12_band_experiments/20260403_085255/models/band12_b025_g050m_stats.json` |
| band12_b025_g100m | band_reward | 12 | 2-10 | 0.25 | -1.00 | 70.59% | 98.53% | 74.20% | 72.42% | 72.42% | yes | - | `results/extreme12_band_experiments/20260403_085255/models/band12_b025_g100m_stats.json` |
| band12_b035_g025m | band_reward | 12 | 2-10 | 0.35 | -0.25 | 69.12% | 98.53% | 74.24% | 72.65% | 72.65% | yes | - | `results/extreme12_band_experiments/20260403_085255/models/band12_b035_g025m_stats.json` |
| band12_b035_g050m | band_reward | 12 | 2-10 | 0.35 | -0.50 | 69.12% | 98.53% | 74.24% | 72.65% | 72.65% | yes | - | `results/extreme12_band_experiments/20260403_085255/models/band12_b035_g050m_stats.json` |
| band12_b035_g100m | band_reward | 12 | 2-10 | 0.35 | -1.00 | 70.59% | 98.53% | 74.20% | 72.42% | 72.42% | yes | - | `results/extreme12_band_experiments/20260403_085255/models/band12_b035_g100m_stats.json` |
| band12_b050_g025m | band_reward | 12 | 2-10 | 0.50 | -0.25 | 69.12% | 98.53% | 74.24% | 72.65% | 72.65% | yes | - | `results/extreme12_band_experiments/20260403_085255/models/band12_b050_g025m_stats.json` |
| band12_b050_g050m | band_reward | 12 | 2-10 | 0.50 | -0.50 | 70.59% | 98.53% | 74.20% | 72.42% | 72.42% | yes | - | `results/extreme12_band_experiments/20260403_085255/models/band12_b050_g050m_stats.json` |
| band12_b050_g100m | band_reward | 12 | 2-10 | 0.50 | -1.00 | 70.59% | 98.53% | 74.20% | 72.42% | 72.42% | yes | - | `results/extreme12_band_experiments/20260403_085255/models/band12_b050_g100m_stats.json` |
