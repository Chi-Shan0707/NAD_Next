# Math-SVD + Science-Tree Hybrid

## Goal

- Keep the strongest existing `math` EarlyStop line unchanged: `es_svd_math_rr_r2_20260412`.
- Swap the weaker `science` SVD line for the stronger grouped-holdout tree baseline.
- Export a full-train blind EarlyStop payload without retraining the already-frozen math SVD bundle.

## Source Bundles

- `math SVD`: `models/ml_selectors/es_svd_math_rr_r2_20260412.pkl`
- `science/coding tree`: `results/models/tree_baselines_fulltrain_v1.pkl`

## Frozen Science Tree Selection

- `domain`: `science`
- `model_family`: `xgboost`
- `feature_variant`: `raw+rank`
- `holdout AUC of AUROC`: `84.36%`
- `delta vs science SVD`: `+0.38` pts
- `search config`: `{"colsample": 0.7, "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 1, "model_family": "xgboost", "n_estimators": 300, "subsample": 0.7}`

## Holdout Readout

| Slice | AUC of AUROC | AUC of SelAcc | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|
| `math` frozen SVD | 96.74% | 99.84% | 97.55% | 75.00% |
| `science` rerun tree | 84.36% | 97.81% | 87.47% | 76.67% |
| `noncoding` frozen SVD | 93.66% | 99.51% | 94.88% | 74.26% |
| `noncoding` hybrid | 93.99% | 99.39% | 95.31% | 75.37% |

## Noncoding Delta vs Frozen SVD

- `AUC of AUROC`: `+0.33` pts
- `AUC of SelAcc`: `-0.12` pts
- `AUROC@100%`: `+0.43` pts
- `Stop Acc@100%`: `+1.11` pts

## Full-Train Export

- `method_name`: `earlystop_math_svd_science_tree_hybrid_fulltrain_v1`
- `bundle`: `results/models/math_svd_science_tree_hybrid_fulltrain_v1.pkl`
- `submission`: `submission/EarlyStop/earlystop_math_svd_science_tree_hybrid_fulltrain_v1.json`
- `manifest`: `results/tables/math_svd_science_tree_hybrid_manifest.json`

## Reading

- The gain comes from a targeted domain swap rather than a new global model family.
- `math` remains an early-strong linear/SVD domain, so the hybrid leaves it untouched.
- `science` benefits from the stronger non-linear tree route while preserving the existing feature bank and holdout protocol.
