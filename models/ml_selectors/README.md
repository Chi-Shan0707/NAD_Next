# `ml_selectors` Layout

## Stable root artifacts
- Core ML probes and calibration models such as `linear_probe.pkl`, `logistic.pkl`, and isotonic calibrators.
- Single-feature baselines `single_feat_*.pkl`.
- Shared stats like `feature_stats.json` and `temporal_best_params.json`.
- Stable selector artifacts currently referenced by code defaults, such as `extreme8_best.pkl`, `extreme8_worst.pkl`, `extreme8_stats.json`, `trajectory_fusion.pkl`, and `trajectory_stats.json`.

## Local variants
- `local_variants/extreme8/`: local domain-specialized Extreme8 artifacts when present.
- `local_variants/extreme9/`: local domain-specialized Extreme9 artifacts when present.

## Why this split
- Root paths stay stable for selector defaults and existing scripts.
- Local exploratory artifacts stop cluttering the root and become easier to scan.
