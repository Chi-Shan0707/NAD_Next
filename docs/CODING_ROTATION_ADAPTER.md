# CODING ROTATION ADAPTER

**Date**: 2026-04-15 UTC  
**Status**: completed

## Summary

- Goal: test whether the `DS-R1 -> Qwen3-4B` coding gap on `lcb_v5` is better explained by a small latent rotation than by a full basis mismatch.
- Source labeled cache: `cache/DS-R1/lcb_v5` from `results/cache/es_svd_ms_rr_r1/cache_all_547b9060debe139e.pkl`.
- Target unlabeled cache: `Qwen3-4B/lcb_v5` from `results/cache/export_earlystop_svd_submission_strongfeat_20260410/feature_store_all_ref030_18a73b5e30f1a00d.pkl`.
- Extra unlabeled target: `none`.
- Bundles: `token_only, canonical_22`.
- Anchors: `70, 100`.
- Seeds: `42`.

## Important Limitation

- The local repo does **not** expose per-sample correctness labels for `cache_test` `lcb_v5`.
- Because of that, the generated CSV reports **source-holdout metrics** plus **unlabeled Qwen proxy metrics** (alignment gap, score correlation, top-1 flip rate), not true Qwen `hit@1` / `pairwise` / `AUROC`.
- The `qwen_*` columns are intentionally left as `NaN` until an external blind evaluation or labeled target artifact is available.

## Best Rows by Cell

| Bundle | Anchor | Head | Best learned source pairwise | Best learned source Hit@1 | Rotation norm | Target corr vs no-rot | Target top1 agreement | Geometry proxy |
|---|---:|---|---:|---:|---:|---:|---:|---|
| token_only | 70 | pointwise | 0.5482 | 0.5400 | 4.8119 | -0.7178 | 0.0000 | geometry-shift-like |
| token_only | 70 | pairwise | 0.5225 | 0.5600 | 0.4324 | 0.9361 | 0.4251 | geometry-shift-like |
| token_only | 100 | pointwise | 0.5215 | 0.5200 | 2.3624 | 0.1176 | 0.0060 | geometry-shift-like |
| token_only | 100 | pairwise | 0.5316 | 0.5800 | 5.0714 | -0.7797 | 0.0000 | geometry-shift-like |
| canonical_22 | 70 | pointwise | 0.5323 | 0.6400 | 2.5506 | 0.9528 | 0.3593 | geometry-shift-like |
| canonical_22 | 70 | pairwise | 0.4752 | 0.5800 | 2.4417 | 0.4722 | 0.0060 | geometry-shift-like |
| canonical_22 | 100 | pointwise | 0.5381 | 0.6000 | 0.0000 | 1.0000 | 1.0000 | mostly calibration-like |
| canonical_22 | 100 | pairwise | 0.4883 | 0.5800 | 4.3778 | 0.6465 | 0.0479 | geometry-shift-like |

## Readout

- **Q1 small rotation recover target ranking quality?** Offline answer is unresolved here because local target labels are unavailable; the current script only establishes whether a near-identity rotation changes Qwen rankings materially and whether it improves unlabeled alignment.
- **Q2 calibration vs geometry shift?** Use the `target_score_corr_vs_no_rotation` and `target_top1_agreement_vs_no_rotation` columns as the offline proxy. High correlation/high agreement suggests mostly calibration-like behavior; lower agreement suggests a more geometric shift.
- **Q3 pointwise vs pairwise after rotation?** Compare `source_val_pairwise` and `source_val_hit1` between `head=pointwise` and `head=pairwise` within the same `bundle × anchor` rows.
- **Q4 does `token_only` remain friendliest?** Compare the best `token_only` and `canonical_22` learned rows. In this offline version, friendliness means retaining source-holdout ranking while requiring only a small rotation and producing a lower target alignment gap.

## 2026-04-15 Update

- Main offline positive signal remains concentrated in late anchors.
- `canonical_22 @ 70 @ pointwise` is the strongest conservative learned-rotation route used for export:
  - source holdout `pairwise = 0.5323`
  - source holdout `Hit@1 = 0.6400`
  - target proxy `corr vs no-rotation = 0.9528`
  - target proxy `top1 agreement vs no-rotation = 0.3593`
- `canonical_22 @ 100 @ pointwise` behaves like a calibration-like anchor:
  - learned rotation collapses to `R ≈ I`
  - `rotation_norm = 0.0000`
  - target proxy `corr = 1.0000`
  - target proxy `top1 agreement = 1.0000`

## Submission Candidate

- **Date**: `2026-04-15 UTC`
- **Method name**: `es_svd_ms_rr_r1__coding_rotation_adapter_late70100`
- **Base submission**: `submission/EarlyStop/es_svd_ms_rr_r1__coding_from_round1c.json`
- **Output submission**: `submission/EarlyStop/es_svd_ms_rr_r1__coding_rotation_adapter_late70100.json`
- **Manifest**: `results/scans/earlystop/coding_rotation_adapter_submission_manifest.json`
- **Patch scope**: replace only `DS-R1/lcb_v5` and `Qwen3-4B/lcb_v5`
- **Late-slot policy**:
  - `70/80/90%` use the selected `70%` learned-rotation scorer
  - `100%` use the selected `100%` learned-rotation scorer
  - `10–60%` keep base submission scores unchanged
- **Selected 70% route**: `canonical_22`, `pointwise`, `rank=16`, `C=0.50`, `class_weight=balanced`, learned rotation with `best_step=20`
- **Selected 100% route**: `canonical_22`, `pointwise`, `rank=4`, `C=1.00`, `class_weight=balanced`, learned rotation with `best_step=0` (`R=I`)
- **Validation**: `total_problems=970`, `total_samples=62080`
- **Offline delta summary**:
  - `DS-R1/lcb_v5`: late-slot `mean_abs_delta = 7.2108`, `score_corr vs base = 0.1807`
  - `Qwen3-4B/lcb_v5`: late-slot `mean_abs_delta = 7.3178`, `score_corr vs base = 0.1764`
- **Interpretation**: this is a structure-valid blind-test candidate for external leaderboard validation, not an offline proof of Qwen improvement.

## Aggressive Submission Candidate

- **Date**: `2026-04-15 UTC`
- **Method name**: `es_svd_ms_rr_r1__coding_rotation_adapter_aggressive_tokenonly_late70100`
- **Base submission**: `submission/EarlyStop/es_svd_ms_rr_r1__coding_from_round1c.json`
- **Output submission**: `submission/EarlyStop/es_svd_ms_rr_r1__coding_rotation_adapter_aggressive_tokenonly_late70100.json`
- **Manifest**: `results/scans/earlystop/coding_rotation_adapter_aggressive_tokenonly_submission_manifest.json`
- **Exporter support**: `scripts/coding_ssl/export_coding_rotation_submission.py` now supports `--anchor70-bundle`, `--anchor70-head`, `--anchor100-bundle`, `--anchor100-head` for reproducible anchor-specific forcing.
- **Selected 70% route**: `token_only`, `pointwise`, `rank=16`, `C=1.00`, `class_weight=none`, `best_step=40`
- **Selected 100% route**: `token_only`, `pairwise`, `rank=16`, `C=0.50`, `class_weight=none`, `best_step=30`
- **Selection rationale**: maximize source-holdout ranking within forced `token_only` late-anchor routes, even when Qwen unlabeled proxy correlation strongly diverges from the no-rotation baseline.
- **Validation**: `total_problems=970`, `total_samples=62080`
- **Offline delta summary**:
  - `DS-R1/lcb_v5`: late-slot `mean_abs_delta = 39.6736`, `score_corr vs base = -0.1637`
  - `Qwen3-4B/lcb_v5`: late-slot `mean_abs_delta = 39.1699`, `score_corr vs base = -0.1751`
- **Risk note**:
  - `70%` forced route proxy: `corr vs no-rotation = -0.7178`, `top1 agreement = 0.0000`
  - `100%` forced route proxy: `corr vs no-rotation = -0.7797`, `top1 agreement = 0.0000`
  - this is intentionally a **high-variance blind probe** for latent-geometry-shift behavior, not a conservative deployment candidate.

## Files

- CSV: `results/tables/coding_rotation_adapter.csv`
- Script: `scripts/coding_ssl/train_coding_rotation_adapter.py`
- Export script: `scripts/coding_ssl/export_coding_rotation_submission.py`
- Submission: `submission/EarlyStop/es_svd_ms_rr_r1__coding_rotation_adapter_late70100.json`
- Aggressive submission: `submission/EarlyStop/es_svd_ms_rr_r1__coding_rotation_adapter_aggressive_tokenonly_late70100.json`
