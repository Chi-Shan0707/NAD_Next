# Tree Baselines vs Current SVD Route

## Claim under test

> If we keep the feature bank, grouped holdout, and EarlyStop metrics fixed, do strong non-linear tabular baselines beat the current SVD route?

## Protocol

- Features are built from the same canonical EarlyStop feature stores used by the SVD line.
- Holdout uses the same grouped `85/15` split by `dataset + problem_id`.
- Tree feature variants are limited to `raw` and `raw+rank`.
- Tree families tested here: `XGBoostClassifier` and `LGBMClassifier`.
- Search stays modest: shared deterministic config search over anchors `100%`, then full refit on all official positions with seeds `42, 101, 202`.
- `CatBoost` is intentionally skipped here because it is not already wired in this environment and was optional under the study brief.

## Best model per domain

- `math`: best tree is `xgboost` + `raw` with `AUC of AUROC=95.74%` (vs current SVD `96.86%`, Δ=`-1.12` pts).
- `science`: not run.
- `coding`: not run.
- `ms`: not run.

### Grouped ID holdout

| Domain | Tree | Variant | AUC of AUROC | AUC of SelAcc | AUROC@100% | Stop Acc@100% | ΔAUC vs current SVD |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | xgboost | raw | 95.74% | 99.95% | 97.78% | 75.00% | -1.12 pts |

### Structured OOD

| Domain | Tree | Variant | OOD protocol | OOD macro AUC of AUROC | ID→OOD gap |
| --- | --- | --- | --- | ---: | ---: |

## Direct comparison against the current SVD route

- `math / science / ms` are compared against the current `r2` routes.
- `coding` is compared against the current `es_svd_coding_rr_r1` route.
- The main fairness constraint is preserved throughout: same feature bank, same grouped holdout unit, same EarlyStop metrics, and the same structured OOD split logic where executed.

## Where trees help

- `math`: has no structured OOD run in this note.

## Feature-importance summary

- `math`: `trajectory` 61.2%, `uncertainty` 15.3%, `self_cert_logprob` 12.0%.
- `science`: no stable feature-importance signal exported.
- `coding`: no stable feature-importance signal exported.

## Paper judgment

- On the current evidence, the SVD route should be positioned as **weaker** to tree baselines.
- If trees win only on some domains or mostly on ID, the clean paper line is not that SVD is obsolete, but that low-rank linear routing and non-linear tabular models capture overlapping yet not identical structure.

## Artifact

- Main table: `results/tables/tree_baselines.csv`
