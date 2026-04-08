# BestofN SVM Bridge to EarlyStop — 2026-04-08

## Metadata
- Date (UTC): 2026-04-08
- Time (UTC, hour-level): 05:00
- Objective:
  1. Output BestofN submission first.
  2. Reuse the same BestofN-trained SVM model to produce EarlyStop (no second earlystop-specific retraining).

## What was implemented
- Added SVM bridge operator: `nad/ops/earlystop_svm.py`
  - Supports three route types per domain:
    - `pointwise` (LinearSVC)
    - `ranksvm` (pairwise hinge RankSVM)
    - `baseline` (single-signal fallback)
  - Added BestofN bridge APIs:
    - `train_bestofn_svm_bundle(...)`
    - `score_cache_entry_bestofn_svm(...)`
    - `score_cache_entry_earlystop_from_bestofn_svm(...)`

- Added one-shot export script: `scripts/export_bestofn_then_earlystop_from_svm.py`
  - Trains one BestofN bridge model on labeled cache.
  - Exports BestofN submission (`task = best_of_n`).
  - Exports EarlyStop submission (`task = early_stop`) using the same model.

- Kept additional standalone scripts for the full earlystop-svm line:
  - `scripts/train_earlystop_svm.py`
  - `scripts/export_earlystop_svm_submission.py`

## Final run configuration
- Train root: `MUI_HUB/cache`
- Test root: `/home/jovyan/public-ro/MUI_HUB/cache_test`
- CV: `n_splits=3`
- Feature family: `token_only`
- Representation: `raw, rank`
- C grid: `0.1, 1.0`
- Loss: `squared_hinge`
- RankSVM backends: `utility, win_count`
- Class weight: `none, balanced`

## Model and submissions
- Bridge model:
  - `models/ml_selectors/bestofn_svm_bridge_v1.pkl`
- Training summary:
  - `results/scans/earlystop/bestofn_svm_bridge_v1_summary.json`
- BestofN submission (priority output):
  - `submission/BestofN/extreme12/patches/extreme12_svm_bridge_bestofn_v1.json`
- EarlyStop submission (same model, no retrain):
  - `submission/EarlyStop/earlystop_from_bestofn_svm_bridge_v1.json`

## Validation
- BestofN validation:
  - `cache_keys=12`, `problems=970`, `samples=62080`
- EarlyStop validation:
  - `total_problems=970`, `total_samples=62080`

## Notes
- `submission/EarlyStop/*.json` is git-ignored in this repository, so the generated EarlyStop JSON is not tracked by git by default.
