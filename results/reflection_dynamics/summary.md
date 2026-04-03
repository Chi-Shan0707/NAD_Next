# Reflection Dynamics Summary

- Base threshold: `0.30`
- Threshold grid: `0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60`
- Best threshold by LOO single-feature benchmark: `0.20` (71.7%)

## Threshold Sweep

| Threshold | LOO Mean | Pooled Mean |
|---|---:|---:|
| `0.10` | 71.7% | 64.5% |
| `0.15` | 71.4% | 64.9% |
| `0.20` | 71.7% | 64.5% |
| `0.25` | 70.8% | 65.6% |
| `0.30` | 71.1% | 65.2% |
| `0.35` | 70.2% | 63.1% |
| `0.40` | 67.6% | 61.4% |
| `0.45` | 68.7% | 61.9% |
| `0.50` | 67.0% | 61.2% |
| `0.55` | 67.7% | 63.9% |
| `0.60` | 67.8% | 62.1% |

## Top Correlations

| Dataset | Label | Metric | Series | Mean Spearman | Runs |
|---|---|---|---|---:|---:|
| brumo25 | incorrect | gini | avg | 0.4911 | 587 |
| brumo25 | incorrect | entropy | avg | -0.4895 | 587 |
| hmmt25 | incorrect | gini | avg | 0.4572 | 987 |
| hmmt25 | incorrect | entropy | avg | -0.4555 | 987 |
| brumo25 | correct | gini | avg | 0.4531 | 1333 |
| brumo25 | correct | entropy | avg | -0.4506 | 1333 |
| aime25 | incorrect | gini | avg | 0.4475 | 658 |
| aime25 | incorrect | entropy | avg | -0.4466 | 658 |
| aime24 | correct | gini | avg | 0.4370 | 1447 |
| hmmt25 | correct | gini | avg | 0.4324 | 933 |
| hmmt25 | correct | entropy | avg | -0.4304 | 933 |
| aime24 | correct | entropy | avg | -0.4267 | 1447 |

## Top Event vs Non-Event Gaps

| Dataset | Label | Metric | Series | Mean Gap | Runs |
|---|---|---|---|---:|---:|
| gpqa | correct | conf | abs_d2 | -0.2654 | 7620 |
| brumo25 | correct | conf | abs_d2 | -0.2355 | 1333 |
| gpqa | incorrect | conf | abs_d2 | -0.2155 | 5045 |
| aime24 | correct | conf | abs_d2 | -0.2136 | 1447 |
| hmmt25 | correct | conf | abs_d2 | -0.2106 | 933 |
| aime24 | correct | conf | abs_d1 | -0.2082 | 1447 |
| gpqa | correct | conf | abs_d1 | -0.1986 | 7620 |
| aime25 | incorrect | conf | abs_d2 | -0.1984 | 658 |
| brumo25 | incorrect | conf | abs_d2 | -0.1888 | 587 |
| aime24 | incorrect | conf | abs_d2 | -0.1754 | 473 |
| livecodebench_v5 | correct | conf | abs_d2 | -0.1738 | 6271 |
| brumo25 | correct | conf | abs_d1 | -0.1728 | 1333 |
