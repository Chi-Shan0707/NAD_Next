# EarlyStop SVD Implementation Report (2026-04-08)

## Report Metadata
- Date (UTC): 2026-04-08
- Time (UTC, hour-level): 03:00
- Scope: Apply SVD-based method to `early_stop` task and generate submission JSON in required schema.

## Direct Answer: SVD or SVM?
- This implementation is **SVD-based**, not SVM.
- Specifically: `StandardScaler -> TruncatedSVD -> LogisticRegression(decision_function)`.
- There is **no SVM / LinearSVC / RankSVM** in this early-stop implementation.

## Initial State
- Existing early-stop lines were:
  - `nad/ops/earlystop.py` (v1 token confidence prefix mean)
  - `nad/ops/earlystop_v2.py` (domain-aware baseline)
  - `nad/ops/earlystop_v3.py` (domain-aware with trajectory coding signal)
- No existing SVD early-stop implementation, no SVD training/export script for early-stop.

## Core Operations Completed
1. Added SVD early-stop operator module:
   - `nad/ops/earlystop_svd.py`
   - Includes:
     - feature extraction for token/trajectory/meta signals by early-stop positions (10%..100%)
     - group-aware CV search (`GroupKFold`, grouped by `cache_key::problem_id`)
     - mixed route selection per `(domain, position)`: choose SVD or best baseline signal by CV AUROC
     - model save/load and inference scorer for cache entries

2. Added training script:
   - `scripts/train_earlystop_svd.py`
   - Trains on labeled cache (`MUI_HUB/cache`) and writes:
     - model bundle (`.pkl`)
     - training summary (`.json`)

3. Added export script:
   - `scripts/export_earlystop_svd_submission.py`
   - Loads trained bundle and exports early-stop submission from `cache_test`.

4. Updated EarlyStop submission docs:
   - `submission/EarlyStop/README.md`
   - Added the new artifact entry.

## Training Configuration Used (Final Run)
- Cache root: `MUI_HUB/cache` (labeled)
- Folds: `n_splits=3`
- Families: `token_only`
- Representations: `raw, rank, raw+rank`
- Ranks: `2,4,6,8,12,16`
- C values: `0.1,1.0,3.0,10.0`
- Whiten options: `false,true`
- Class weights: `none,balanced`

## Final Outputs
- Model bundle:
  - `models/ml_selectors/earlystop_svd_lowrank_lr_v1.pkl`
- Training summary:
  - `results/scans/earlystop/earlystop_svd_lowrank_lr_v1_summary.json`
- Submission JSON:
  - `submission/EarlyStop/earlystop_svd_lowrank_lr_v1.json`

## Key Summary Results
- Slot totals (domain x position = 30):
  - `svd_slots = 20`
  - `baseline_slots = 10`
- By domain:
  - `math`: 10 SVD / 0 baseline
  - `science`: 10 SVD / 0 baseline
  - `coding`: 0 SVD / 10 baseline
- Export validation:
  - `total_problems = 970`
  - `total_samples = 62080`
  - schema valid (`task=early_stop`, each sample has 10 finite scores)

## Notes
- `submission/EarlyStop/*.json` is git-ignored in this repository, so generated submission artifact is not tracked by default.
