# Direct Linear Baselines Without SVD

## Summary

This report benchmarks the strongest direct linear baselines on the canonical `token_plus_traj_fixed` 22-feature route **without SVD**. The benchmark keeps the same grouped `85/15` holdout, the same four-anchor training protocol, and the same raw/rank feature construction used by the paper-facing SVDomain route.

The direct family includes:

- `lr_raw`
- `lr_rank`
- `lr_raw_rank`
- `elasticnet_lr_raw_rank`
- `l1_lr_raw_rank`
- `linear_svm_raw_rank`

All comparisons below treat structured OOD as a **blind-proxy robustness** slice: it is not the blind leaderboard itself, but it is the closest labeled robustness proxy available offline.

## Key Findings

- On in-repo grouped holdout, direct no-SVD linear heads beat the current SVD route in `math`, `science`, `ms`, and `coding`, with the clearest relative gains on `science` and `ms`.
- `raw+rank` is the strongest plain LR representation for `math`, `science`, and `ms`; `coding` is the exception, where `rank` alone edges out `raw+rank`, but all absolute coding results remain close to chance.
- Elastic-net improves blind-proxy OOD only marginally over plain `raw+rank` LR: about `+0.0006` on withheld math benchmarks and `+0.0005` on science cache-root withholding.
- The dominant signal families in the strongest direct heads remain `trajectory`, `uncertainty`, and `self_cert_logprob`, with `traj_novelty`, `tok_gini_prefix`, and prefix/recency self-certainty features carrying most weight.

## Protocol

- `feature family`: `token_plus_traj_fixed`
- `feature count`: `22`
- `anchors`: `10, 40, 70, 100`
- `holdout split`: `85/15` by `dataset + problem_id`
- `CV folds`: `3` grouped folds
- `C grid`: `0.01, 0.1, 1, 10`
- `class_weight`: `none, balanced`
- `elastic-net l1_ratio`: `0.1, 0.5, 0.9`
- `random seeds where applicable`: `42, 43, 44`

## Best Direct vs Current SVD Route

| Domain | Best Direct | Direct AUC-AUROC | SVD AUC-AUROC | Δ | Direct AUROC@100 | SVD AUROC@100 | Δ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| math | elasticnet_lr_raw_rank | 0.9593 | 0.9581 | 0.0012 | 0.9829 | 0.9817 | 0.0012 |
| science | l1_lr_raw_rank | 0.8363 | 0.7985 | 0.0378 | 0.8735 | 0.8411 | 0.0324 |
| ms | l1_lr_raw_rank | 0.9319 | 0.9226 | 0.0093 | 0.9584 | 0.9505 | 0.0079 |
| coding | lr_rank | 0.4957 | 0.4342 | 0.0615 | 0.4981 | 0.4068 | 0.0913 |

## Representation Check

| Domain | LR raw | LR rank | LR raw+rank | raw+rank vs best(single) |
| --- | ---: | ---: | ---: | ---: |
| math | 0.9533 | 0.5892 | 0.9592 | 0.0060 |
| science | 0.8266 | 0.5366 | 0.8351 | 0.0085 |
| ms | 0.9251 | 0.5775 | 0.9316 | 0.0065 |
| coding | 0.4653 | 0.4957 | 0.4746 | -0.0211 |

## Structured OOD / Blind-Proxy Robustness

| Domain | OOD Protocol | Model | ID AUC-AUROC | OOD Macro | Gap |
| --- | --- | --- | ---: | ---: | ---: |
| coding | cache_root_withheld | lr_raw | 0.4653 | N/A | N/A |
| coding | cache_root_withheld | lr_rank | 0.4957 | N/A | N/A |
| coding | cache_root_withheld | lr_raw_rank | 0.4746 | N/A | N/A |
| coding | cache_root_withheld | elasticnet_lr_raw_rank | 0.4824 | N/A | N/A |
| coding | cache_root_withheld | l1_lr_raw_rank | 0.4914 | N/A | N/A |
| coding | cache_root_withheld | linear_svm_raw_rank | 0.4782 | N/A | N/A |
| coding | model_family_withheld | lr_raw | 0.4653 | N/A | N/A |
| coding | model_family_withheld | lr_rank | 0.4957 | N/A | N/A |
| coding | model_family_withheld | lr_raw_rank | 0.4746 | N/A | N/A |
| coding | model_family_withheld | elasticnet_lr_raw_rank | 0.4824 | N/A | N/A |
| coding | model_family_withheld | l1_lr_raw_rank | 0.4914 | N/A | N/A |
| coding | model_family_withheld | linear_svm_raw_rank | 0.4782 | N/A | N/A |
| math | math_benchmark_withheld | lr_raw | 0.9533 | 0.8317 | -0.1216 |
| math | math_benchmark_withheld | lr_rank | 0.5892 | 0.5225 | -0.0667 |
| math | math_benchmark_withheld | lr_raw_rank | 0.9592 | 0.7497 | -0.2096 |
| math | math_benchmark_withheld | elasticnet_lr_raw_rank | 0.9593 | 0.7503 | -0.2090 |
| math | math_benchmark_withheld | l1_lr_raw_rank | 0.9593 | 0.7507 | -0.2085 |
| math | math_benchmark_withheld | linear_svm_raw_rank | 0.9590 | 0.7487 | -0.2103 |
| science | cache_root_withheld | lr_raw | 0.8266 | 0.5949 | -0.2317 |
| science | cache_root_withheld | lr_rank | 0.5366 | 0.4944 | -0.0422 |
| science | cache_root_withheld | lr_raw_rank | 0.8351 | 0.6230 | -0.2121 |
| science | cache_root_withheld | elasticnet_lr_raw_rank | 0.8357 | 0.6235 | -0.2122 |
| science | cache_root_withheld | l1_lr_raw_rank | 0.8363 | 0.6235 | -0.2128 |
| science | cache_root_withheld | linear_svm_raw_rank | 0.8357 | 0.6357 | -0.2000 |
| science | model_family_withheld | lr_raw | 0.8266 | N/A | N/A |
| science | model_family_withheld | lr_rank | 0.5366 | N/A | N/A |
| science | model_family_withheld | lr_raw_rank | 0.8351 | N/A | N/A |
| science | model_family_withheld | elasticnet_lr_raw_rank | 0.8357 | N/A | N/A |
| science | model_family_withheld | l1_lr_raw_rank | 0.8363 | N/A | N/A |
| science | model_family_withheld | linear_svm_raw_rank | 0.8357 | N/A | N/A |

`N/A` in the OOD columns means the withheld slice collapsed into a single class or an invalid training split, so macro AUROC is not informative there.

## Coefficient Dominance

| Domain | Best Direct | Top Families | Top Base Features |
| --- | --- | --- | --- |
| math | elasticnet_lr_raw_rank | trajectory:584.4771; self_cert_logprob:438.8272; uncertainty:387.6278; confidence:85.1512; availability_meta:1.7926 | traj_novelty:514.6178; tok_gini_prefix:171.1080; tok_logprob_recency:170.7635; tok_selfcert_prefix:130.1013; tok_neg_entropy_prefix:130.1013 |
| science | l1_lr_raw_rank | uncertainty:269.1323; trajectory:151.3200; self_cert_logprob:128.5378; confidence:80.7722; availability_meta:1.9339 | tok_gini_prefix:147.9759; traj_continuity:91.8850; tok_selfcert_prefix:60.6752; tok_neg_entropy_prefix:60.6752; traj_novelty:47.7987 |
| coding | lr_rank | self_cert_logprob:6.0625; uncertainty:3.1985; confidence:1.8192; trajectory:0.6916; availability_meta:0.0635 | tok_logprob_prefix:2.1084; tok_logprob_recency:1.7713; tok_neg_entropy_recency:1.2310; tok_selfcert_recency:1.2310; tok_conf_prefix:0.9674 |

## Explicit Answers

### 1. Does no-SVD direct LR match or beat the current SVD route?

- `math`: best direct baseline is `elasticnet_lr_raw_rank` with `AUC-of-AUROC=0.9593`; it beats the current SVD route (`Δ=0.0012`).
- `science`: best direct baseline is `l1_lr_raw_rank` with `AUC-of-AUROC=0.8363`; it beats the current SVD route (`Δ=0.0378`).
- `ms`: best direct baseline is `l1_lr_raw_rank` with `AUC-of-AUROC=0.9319`; it beats the current SVD route (`Δ=0.0093`).
- `coding`: best direct baseline is `lr_rank` with `AUC-of-AUROC=0.4957`; it beats the current SVD route (`Δ=0.0615`), but the absolute result is still near chance and should be treated as a weak regime overall.

### 2. Is raw+rank better than raw-only or rank-only?

- `math`: `lr_raw_rank=0.9592`, `lr_raw=0.9533`, `lr_rank=0.5892`; best among the three is `lr_raw_rank`.
- `science`: `lr_raw_rank=0.8351`, `lr_raw=0.8266`, `lr_rank=0.5366`; best among the three is `lr_raw_rank`.
- `ms`: `lr_raw_rank=0.9316`, `lr_raw=0.9251`, `lr_rank=0.5775`; best among the three is `lr_raw_rank`.
- `coding`: `lr_raw_rank=0.4746`, `lr_raw=0.4653`, `lr_rank=0.4957`; best among the three is `lr_rank`.

### 3. Does elastic-net improve OOD or blind-proxy robustness?

- `math` / `math_benchmark_withheld`: elastic-net OOD macro AUROC is `0.7503` vs `0.7497` for plain raw+rank LR (`Δ=0.0006`).
- `science` / `cache_root_withheld`: elastic-net OOD macro AUROC is `0.6235` vs `0.6230` for plain raw+rank LR (`Δ=0.0005`).
- `science` / `model_family_withheld` and both `coding` OOD protocols are class-degenerate in this offline suite, so macro AUROC is not a meaningful robustness comparison there.

### 4. Which coefficients / feature families dominate in the best direct model?

- `math` best direct model: `elasticnet_lr_raw_rank`; dominant families: `trajectory:584.4771; self_cert_logprob:438.8272; uncertainty:387.6278; confidence:85.1512; availability_meta:1.7926`; dominant base features: `traj_novelty:514.6178; tok_gini_prefix:171.1080; tok_logprob_recency:170.7635; tok_selfcert_prefix:130.1013; tok_neg_entropy_prefix:130.1013`.
- `science` best direct model: `l1_lr_raw_rank`; dominant families: `uncertainty:269.1323; trajectory:151.3200; self_cert_logprob:128.5378; confidence:80.7722; availability_meta:1.9339`; dominant base features: `tok_gini_prefix:147.9759; traj_continuity:91.8850; tok_selfcert_prefix:60.6752; tok_neg_entropy_prefix:60.6752; traj_novelty:47.7987`.
- `coding` best direct model: `lr_rank`; dominant families: `self_cert_logprob:6.0625; uncertainty:3.1985; confidence:1.8192; trajectory:0.6916; availability_meta:0.0635`; dominant base features: `tok_logprob_prefix:2.1084; tok_logprob_recency:1.7713; tok_neg_entropy_recency:1.2310; tok_selfcert_recency:1.2310; tok_conf_prefix:0.9674`.

## Interpretation

- `raw+rank` should be interpreted as a direct linear head on the same feature bank used by the SVD route, but without the low-rank bottleneck.
- The comparison is therefore about **whether the low-rank bottleneck is useful**, not about whether the feature bank itself carries signal.
- Structured OOD serves as an offline robustness proxy. When an OOD slice degenerates into a single class, AUROC becomes `N/A`, so `AUROC@100` and grouped selection metrics matter more than the macro AUROC alone.

## Artifacts

- `table`: `results/tables/direct_linear_baselines.csv`
- `report`: `docs/DIRECT_LINEAR_BASELINES.md`
- `SVD holdout reference`: `SVDomain/results/summary_metrics.json`
- `SVD OOD reference`: `SVDomain/results/tables/id_vs_ood_summary.csv`
