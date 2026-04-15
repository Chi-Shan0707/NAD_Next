# SHALLOW MLP BASELINES

## Setup

- `repo root`: `/home/jovyan/work/NAD_Next`
- `protocol`: grouped `85/15` holdout by `dataset + problem_id`, `split_seed=42`
- `domains`: `math`, `science`, `ms`, `coding`
- `representations`: `raw+rank` by default; `raw` remains script-optional but is not part of this artifact
- `seeds`: `42,101,29`
- `features`: token + trajectory + availability (`tok_*`, `traj_*`, `has_*`)
- `feature cache`: reuses the existing legacy 30-feature prefix-safe tensors so runs stay compact and match the current SVD baselines
- `model backend`: `sklearn.neural_network.MLPClassifier`
- `searched`: hidden sizes `[32]`, `[64]`, `[64,32]`, `[128,64]`; weight decay `[0, 1e-4, 1e-3]`; batch size `[64, 128]`; activation `ReLU`
- `early stopping`: validation AUROC with `max_epochs=12`, `patience=3`, `min_delta=0.0001`
- `not searched`: dropout / LayerNorm / GELU, because the current repo environment has no `torch` and the sklearn backend does not expose those options cleanly
- `structured OOD`: not included in this artifact; this file reports ID grouped-holdout only

## Main Table

| domain | best SVD ref | shallow MLP | delta | seed std | seeds > ref | family |
|---|---:|---:|---:|---:|---:|---|
| math | 0.9686 | 0.9627 | -0.0059 | 0.0011 | 0/3 | mlp_2h |
| science | 0.8398 | 0.8454 | 0.0056 | 0.0070 | 2/3 | mlp_2h |
| ms | 0.9400 | 0.9323 | -0.0077 | 0.0019 | 0/3 | mlp_2h |
| coding | 0.4924 | 0.5327 | 0.0403 | 0.0430 | 3/3 | mlp_2h |

## Answers

- **Does light non-linearity materially outperform the SVD route?** Mixed — gains appear in some domains, but they are not broad enough to call a clear overall win over the SVD route.
- **Does MLP mainly help science / coding or not?** Yes — the MLP gains are concentrated in science/coding rather than math.
- **Are gains stable across seeds?** Mixed — some seeds beat the SVD route, but the margin is not fully stable.
- **Is the accuracy gain large enough to justify losing exact linear interpretability?** No — the measured gain is not large or stable enough to outweigh the loss of exact linear interpretability.

## Notes

- `best SVD ref` means the strongest available SVD bundle on the same holdout scope, chosen from the evaluated reference rows in `results/tables/shallow_mlp_baselines.csv`.
- `ms` is the combined non-coding scope (`math + science`), matching the existing repo convention.
- `coding` is included only because this run kept it feasible on the same grouped-holdout pipeline; no structured-OOD claim is made here.

## Reference Rows

| domain | method | kind | auc_of_auroc | auroc@100 | stop_acc@100 |
|---|---|---|---:|---:|---:|
| coding | `tok_conf_prefix_mean_v1` | linear | 0.5062 | 0.4904 | 0.4800 |
| coding | `es_svd_coding_rr_r1` | svd | 0.4924 | 0.4282 | 0.5200 |
| coding | `earlystop_svd_lowrank_lr_v1` | svd | 0.4762 | 0.4733 | 0.5600 |
| coding | `earlystop_prefix10_svd_round1` | svd | 0.4762 | 0.4733 | 0.5600 |
| math | `es_svd_math_rr_r2_20260412` | svd | 0.9686 | 0.9827 | 0.7500 |
| math | `earlystop_prefix10_svd_round1` | svd | 0.9348 | 0.9630 | 0.7857 |
| math | `earlystop_svd_lowrank_lr_v1` | svd | 0.7769 | 0.9047 | 0.7143 |
| math | `tok_conf_prefix_mean_v1` | linear | 0.6852 | 0.6838 | 0.7500 |
| ms | `es_svd_ms_rr_r2_20260412` | svd | 0.9400 | 0.9559 | 0.7463 |
| ms | `earlystop_prefix10_svd_round1` | svd | 0.8988 | 0.9264 | 0.7741 |
| ms | `earlystop_svd_lowrank_lr_v1` | svd | 0.7758 | 0.8801 | 0.7259 |
| ms | `tok_conf_prefix_mean_v1` | linear | 0.7127 | 0.7221 | 0.7389 |
| science | `es_svd_science_rr_r2_20260412` | svd | 0.8398 | 0.8621 | 0.7333 |
| science | `tok_conf_prefix_mean_v1` | linear | 0.8089 | 0.8561 | 0.7000 |
| science | `earlystop_prefix10_svd_round1` | svd | 0.7730 | 0.7984 | 0.7333 |
| science | `earlystop_svd_lowrank_lr_v1` | svd | 0.7719 | 0.7937 | 0.7667 |
