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

- `math`: best tree is `lightgbm` + `raw+rank` with `AUC of AUROC=96.07%` (vs current SVD `96.86%`, Δ=`-0.79` pts).
- `science`: best tree is `xgboost` + `raw+rank` with `AUC of AUROC=84.36%` (vs current SVD `83.98%`, Δ=`+0.38` pts).
- `coding`: best tree is `xgboost` + `raw` with `AUC of AUROC=55.58%` (vs current SVD `49.24%`, Δ=`+6.34` pts).
- `ms`: hybrid best-by-domain tree bundle reaches `AUC of AUROC=93.47%` vs current SVD `es_svd_ms_rr_r2=94.00%` (Δ=`-0.53` pts).

### Grouped ID holdout

| Domain | Tree | Variant | AUC of AUROC | AUC of SelAcc | AUROC@100% | Stop Acc@100% | ΔAUC vs current SVD |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| math | xgboost | raw | 95.75% | 99.89% | 97.76% | 75.00% | -1.11 pts |
| math | xgboost | raw+rank | 95.95% | 99.73% | 97.59% | 78.57% | -0.91 pts |
| math | lightgbm | raw | 95.61% | 99.84% | 97.78% | 75.00% | -1.25 pts |
| math | lightgbm | raw+rank | 96.07% | 99.67% | 97.80% | 78.57% | -0.79 pts |
| science | xgboost | raw | 83.88% | 96.54% | 87.61% | 76.67% | -0.11 pts |
| science | xgboost | raw+rank | 84.36% | 97.81% | 87.47% | 76.67% | +0.38 pts |
| science | lightgbm | raw | 83.58% | 96.54% | 87.27% | 73.33% | -0.41 pts |
| science | lightgbm | raw+rank | 84.02% | 97.63% | 87.39% | 76.67% | +0.04 pts |
| coding | xgboost | raw | 55.58% | 53.44% | 54.21% | 56.00% | +6.34 pts |
| coding | xgboost | raw+rank | 54.78% | 51.38% | 52.84% | 52.00% | +5.55 pts |
| coding | lightgbm | raw | 54.72% | 52.19% | 52.16% | 52.00% | +5.48 pts |
| coding | lightgbm | raw+rank | 54.86% | 52.56% | 52.42% | 52.00% | +5.62 pts |
| ms | hybrid | best_by_domain | 93.47% | 99.26% | 95.50% | 78.15% | -0.53 pts |

### Structured OOD

| Domain | Tree | Variant | OOD protocol | OOD macro AUC of AUROC | ID→OOD gap |
| --- | --- | --- | --- | ---: | ---: |
| math | lightgbm | raw+rank | math_benchmark_withheld | 71.70% | -24.37 pts |
| science | xgboost | raw+rank | cache_root_withheld | 59.97% | -24.39 pts |
| coding | xgboost | raw | cache_root_withheld | N/A | N/A |

## Direct comparison against the current SVD route

- `math / science / ms` are compared against the current `r2` routes.
- `coding` is compared against the current `es_svd_coding_rr_r1` route.
- The main fairness constraint is preserved throughout: same feature bank, same grouped holdout unit, same EarlyStop metrics, and the same structured OOD split logic where executed.

## Where trees help

- `math`: does not beat the current SVD route and also does not open a new OOD advantage.
- `science`: helps mainly in ID; OOD erodes part of the gain.
- `coding`: has inconclusive OOD evidence.

## Feature-importance summary

- `math`: `trajectory` 48.3%, `uncertainty` 23.9%, `confidence` 12.5%.
- `science`: `trajectory` 24.6%, `uncertainty` 21.7%, `confidence` 20.1%.
- `coding`: `trajectory` 32.8%, `uncertainty` 32.7%, `self_cert_logprob` 19.0%.

## Paper judgment

- On the current evidence, the SVD route should be positioned as **complementary** relative to tree baselines.
- If trees win only on some domains or mostly on ID, the clean paper line is not that SVD is obsolete, but that low-rank linear routing and non-linear tabular models capture overlapping yet not identical structure.

## Artifact

- Main table: `results/tables/tree_baselines.csv`
- Full-train blind export note: `docs/TREE_BASELINES_FULLTRAIN_SUBMISSION.md`
- Full-train submission JSON: `submission/EarlyStop/earlystop_tree_baselines_fulltrain_v1.json`
