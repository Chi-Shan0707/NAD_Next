# CODING ROTATION ADAPTER

## Summary

- Goal: test whether the `DS-R1 -> Qwen3-4B` coding gap on `lcb_v5` is better explained by a small latent rotation than by a full basis mismatch.
- Source labeled cache: `cache/DS-R1/lcb_v5` from `results/cache/es_svd_ms_rr_r1/cache_all_547b9060debe139e.pkl`.
- Target unlabeled cache: `Qwen3-4B/lcb_v5` from `results/cache/export_earlystop_svd_submission_strongfeat_20260410/feature_store_all_ref030_18a73b5e30f1a00d.pkl`.
- Extra unlabeled target: `none`.
- Bundles: `token_only`.
- Anchors: `70`.
- Seeds: `42`.

## Important Limitation

- The local repo does **not** expose per-sample correctness labels for `cache_test` `lcb_v5`.
- Because of that, the generated CSV reports **source-holdout metrics** plus **unlabeled Qwen proxy metrics** (alignment gap, score correlation, top-1 flip rate), not true Qwen `hit@1` / `pairwise` / `AUROC`.
- The `qwen_*` columns are intentionally left as `NaN` until an external blind evaluation or labeled target artifact is available.

## Best Rows by Cell

| Bundle | Anchor | Head | Best learned source pairwise | Best learned source Hit@1 | Rotation norm | Target corr vs no-rot | Target top1 agreement | Geometry proxy |
|---|---:|---|---:|---:|---:|---:|---:|---|
| token_only | 70 | pointwise | 0.5246 | 0.5400 | 0.8856 | 0.5106 | 0.0120 | geometry-shift-like |
| token_only | 70 | pairwise | 0.5219 | 0.5400 | 0.8244 | 0.8331 | 0.2156 | geometry-shift-like |

## Readout

- **Q1 small rotation recover target ranking quality?** Offline answer is unresolved here because local target labels are unavailable; the current script only establishes whether a near-identity rotation changes Qwen rankings materially and whether it improves unlabeled alignment.
- **Q2 calibration vs geometry shift?** Use the `target_score_corr_vs_no_rotation` and `target_top1_agreement_vs_no_rotation` columns as the offline proxy. High correlation/high agreement suggests mostly calibration-like behavior; lower agreement suggests a more geometric shift.
- **Q3 pointwise vs pairwise after rotation?** Compare `source_val_pairwise` and `source_val_hit1` between `head=pointwise` and `head=pairwise` within the same `bundle × anchor` rows.
- **Q4 does `token_only` remain friendliest?** Compare the best `token_only` and `canonical_22` learned rows. In this offline version, friendliness means retaining source-holdout ranking while requiring only a small rotation and producing a lower target alignment gap.

## Files

- CSV: `results/tables/coding_rotation_adapter.smoke.csv`
- Script: `scripts/coding_ssl/train_coding_rotation_adapter.py`

