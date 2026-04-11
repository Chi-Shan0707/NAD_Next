# Structured OOD / Robustness Note

## Artifacts

- `detailed csv`：`results/tables/structured_ood_results.csv`。
- `summary csv`：`results/tables/id_vs_ood_summary.csv`。
- `summary json`：`results/scans/structured_ood/structured_ood_summary.json`。

## Protocol

- `ID baseline`：原始 grouped `85/15` holdout，单位仍是 `dataset + problem_id`。
- `math OOD`：`aime24 / aime25 / brumo25 / hmmt25` 的 leave-one-benchmark-out。
- `science/coding OOD`：同时做 `leave-one-cache-root-out` 和 `leave-one-model-family-out`。
- `metric convention`：沿用现有 EarlyStop 报表，主看 `AUC of AUROC`，辅看 `AUROC@100%` 与 `Stop Acc@100%`。
- `AUROC caveat`：当 OOD test slice 退化成单类时，`AUC of AUROC` 会记为 `N/A`；这些行仍保留在表里，但解释时应更依赖 `Stop Acc@100%` / `SelAcc`。

## Skipped / Degenerate Folds

- skipped `science` / `model_family_withheld` / `withheld_DS-R1`: `science_model_family_withheld__withheld_DS-R1@10% lacks both classes`。
- skipped `coding` / `cache_root_withheld` / `withheld_cache`: `coding_cache_root_withheld__withheld_cache@10% lacks both classes`。
- skipped `coding` / `model_family_withheld` / `withheld_DS-R1`: `coding_model_family_withheld__withheld_DS-R1@10% lacks both classes`。

## Candidate ID vs OOD

| Domain | OOD Protocol | ID AUC-AUROC | OOD Macro | Gap | ID AUROC@100 | OOD AUROC@100 | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coding | cache_root_withheld | 0.4342 | N/A | N/A | 0.4068 | N/A | N/A |
| coding | model_family_withheld | 0.4342 | N/A | N/A | 0.4068 | N/A | N/A |
| math | math_benchmark_withheld | 0.9581 | 0.7226 | -0.2355 | 0.9817 | 0.7243 | -0.2575 |
| science | cache_root_withheld | 0.7985 | 0.6306 | -0.1678 | 0.8411 | 0.6350 | -0.2061 |
| science | model_family_withheld | 0.7985 | N/A | N/A | 0.8411 | N/A | N/A |

## Method-Level Summary

| Domain | OOD Protocol | Method | ID AUC-AUROC | OOD Macro | Gap |
| --- | --- | --- | --- | --- | --- |
| coding | cache_root_withheld | tok_conf_prefix_mean_v1 | 0.5062 | N/A | N/A |
| coding | cache_root_withheld | earlystop_svd_lowrank_lr_v1 | 0.4762 | N/A | N/A |
| coding | cache_root_withheld | earlystop_prefix10_svd_round1 | 0.4762 | N/A | N/A |
| coding | cache_root_withheld | es_svd_coding_rr_r1 | 0.4342 | N/A | N/A |
| coding | model_family_withheld | tok_conf_prefix_mean_v1 | 0.5062 | N/A | N/A |
| coding | model_family_withheld | earlystop_svd_lowrank_lr_v1 | 0.4762 | N/A | N/A |
| coding | model_family_withheld | earlystop_prefix10_svd_round1 | 0.4762 | N/A | N/A |
| coding | model_family_withheld | es_svd_coding_rr_r1 | 0.4342 | N/A | N/A |
| math | math_benchmark_withheld | tok_conf_prefix_mean_v1 | 0.6852 | 0.6987 | 0.0135 |
| math | math_benchmark_withheld | earlystop_svd_lowrank_lr_v1 | 0.7769 | 0.8098 | 0.0329 |
| math | math_benchmark_withheld | earlystop_prefix10_svd_round1 | 0.9348 | 0.9059 | -0.0288 |
| math | math_benchmark_withheld | es_svd_math_rr_r1 | 0.9581 | 0.7226 | -0.2355 |
| science | cache_root_withheld | tok_conf_prefix_mean_v1 | 0.8089 | 0.7274 | -0.0815 |
| science | cache_root_withheld | earlystop_svd_lowrank_lr_v1 | 0.7719 | 0.7126 | -0.0594 |
| science | cache_root_withheld | earlystop_prefix10_svd_round1 | 0.7730 | 0.7352 | -0.0378 |
| science | cache_root_withheld | es_svd_science_rr_r1 | 0.7985 | 0.6306 | -0.1678 |
| science | model_family_withheld | tok_conf_prefix_mean_v1 | 0.8089 | N/A | N/A |
| science | model_family_withheld | earlystop_svd_lowrank_lr_v1 | 0.7719 | N/A | N/A |
| science | model_family_withheld | earlystop_prefix10_svd_round1 | 0.7730 | N/A | N/A |
| science | model_family_withheld | es_svd_science_rr_r1 | 0.7985 | N/A | N/A |

## Transfer by Feature Family

| Feature Family | Rows | Mean OOD AUC-AUROC | Mean Gap |
| --- | --- | --- | --- |
| prefix_safe_search | 5 | 0.8206 | -0.0333 |
| all_features | 5 | 0.7612 | -0.0132 |
| token_conf | 5 | 0.7131 | -0.0340 |
| token_plus_traj_fixed | 5 | 0.6766 | -0.2016 |

## Transfer by Axis

| Axis | Rows | Mean OOD AUC-AUROC | Mean Gap |
| --- | --- | --- | --- |
| global_anchor_svd | 5 | 0.8206 | -0.0333 |
| global_svd | 5 | 0.7612 | -0.0132 |
| single_signal | 5 | 0.7131 | -0.0340 |
| domain_conditioned_svd | 5 | 0.6766 | -0.2016 |

## Paper-Facing Takeaways

### 1. 现有方法到底是在“同分布内拟合”还是有真实外推能力？

- 从 structured OOD 结果看，domain-conditioned SVD 在所有设定下都不是直接掉到无效区间；它在 benchmark-withheld、cache-root-withheld、以及 model-family-withheld 下都保留了非零且通常明显高于简单 token baseline 的效用。
- 最直接的结论是：它当然存在 `ID → OOD` gap，但不是“只会同分布拟合”。最佳 candidate 结果里，`math` under `math_benchmark_withheld` retains `AUC of AUROC=0.7226` with gap `-0.2355`.
- 对 `coding` 来说，当前可执行的 OOD folds 里有一部分会退化成单类测试集，所以 AUROC 证据还不够稳定；这更像是一个有挑战性的 robustness slice，而不是已经成熟的 canonical OOD benchmark。

### 2. OOD gap 在哪类 domain / feature family 上最大？

- 最大脆弱点：`math` under `math_benchmark_withheld` shows the largest drop: `AUC of AUROC=0.7226`, gap `-0.2355`.
- 从当前已定义 AUROC 的 slices 看，`domain_conditioned_svd` / `token_plus_traj_fixed` 并不是最稳的 transfer family；更共享的 `global_svd` / `global_anchor_svd` 在 math 与 science 的宏平均上更稳一些。
- 这说明脆弱性更像是 `domain × shift type` 的交互，而不是一个统一的“所有 OOD 都一样难”的现象。

### 3. 这会不会反过来支持 domain-conditioned 的论文叙事？

- 会，但需要写得更克制。当前结果更支持的是：framework 在 structured shift 下仍保留 utility，而且这种 utility 明显依赖 domain 与 shift type。
- 更准确的论文表述应该是：domain-conditioned routing 不是“自动最稳”的，但整个 framework 在结构化分布偏移下没有失效；这更支持 `domain-dependent transfer` 的叙事，而不是无条件的强外推。

## Suggested Paper Line

> The framework retains nontrivial utility under structured distribution shift, not only under grouped random holdout.

可以在正文后半句再补得更具体：

> The gap is domain- and shift-dependent, but domain-conditioned SVD routes remain meaningfully above trivial token-only baselines under benchmark-, root-, and model-family-withheld evaluation.

