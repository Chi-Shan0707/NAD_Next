# Tree Baselines Full-Train Blind Export

## Purpose

- This note records the blind-export stage that follows the documented holdout study in `docs/TREE_BASELINES.md`.
- Model selection stays frozen from `results/tables/tree_baselines.csv`.
- Final training uses all labeled data from `cache + cache_train` with **no holdout**.
- The resulting bundle is exported on `cache_test` as a standard EarlyStop submission JSON.

## Frozen holdout selections

- `math`: `lightgbm` + `raw+rank` | holdout `AUC of AUROC=96.07%` | vs SVD `96.86%` | Δ `-0.79` pts | config `{"colsample": 0.7, "learning_rate": 0.03, "max_depth": 5, "min_data_in_leaf": 10, "model_family": "lightgbm", "n_estimators": 800, "subsample": 0.7}`
- `science`: `xgboost` + `raw+rank` | holdout `AUC of AUROC=84.36%` | vs SVD `83.98%` | Δ `+0.38` pts | config `{"colsample": 0.7, "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 1, "model_family": "xgboost", "n_estimators": 300, "subsample": 0.7}`
- `coding`: `xgboost` + `raw` | holdout `AUC of AUROC=55.58%` | vs SVD `49.24%` | Δ `+6.34` pts | config `{"colsample": 1.0, "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 5, "model_family": "xgboost", "n_estimators": 300, "subsample": 1.0}`

## Full-train protocol

- Labeled training sources: `cache` and `cache_train`.
- No grouped holdout is carved out at this stage.
- Official positions: `10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%`.
- Seed ensemble: `42, 101, 202`.
- Feature bank stays identical to the paper-facing SVD route because every tree route reuses the reference route feature subset.
- Blind export target: `/home/jovyan/public-ro/MUI_HUB/cache_test`.

## Artifacts

- Selection table: `results/tables/tree_baselines.csv`
- Full-train bundle: `results/models/tree_baselines_fulltrain_v1.pkl`
- Submission JSON: `submission/EarlyStop/earlystop_tree_baselines_fulltrain_v1.json`
- Manifest JSON: `results/tables/tree_baselines_fulltrain_submission_manifest.json`
- Submission validation: `970` problems, `62,080` scored samples

## Submission note

- `method_name`: `earlystop_tree_baselines_fulltrain_v1`
- This export is the right object to submit manually and use for external feedback, because it does not waste labeled training data on a local holdout.
