# Coding Random Adapter

**Date**: 2026-04-15T08:20:35Z  
**Status**: completed

## Summary

- Domain: `coding` / `livecodebench_v5` only.
- Source labeled: `DeepSeek lcb_v5` from `MUI_HUB/cache`.
- Target unlabeled during adaptation: `DeepSeek lcb_v5 cache_test`, `Qwen lcb_v5 cache_test`.
- Anchors: shared late-anchor training over `70%` and `100%`, with per-anchor evaluation.
- Feature families: `canonical_22` (`token_plus_traj_fixed`) and `token_only`, both with `raw+rank`.
- Frozen basis: `rank=12`, `whiten=False`, mirroring the current late-anchor coding SVD setting.
- Adapter sweep: `rank ∈ {2,4,8}`, `alpha ∈ {0.10,0.30,1.00}`.

## Important Limitation

- The local repo does **not** expose per-sample correctness labels for the coding `cache_test` roots used here.
- Because of that, source metrics are fully computed, but target `AUROC` / `Rank` / `Hit@1` / `Hit@3` / `SelAcc@10%` / `Pairwise Acc` remain unavailable offline.
- The transductive training path is still implemented correctly with unlabeled targets; external blind evaluation or a labeled target artifact is still required for final DS->Qwen claims.

## Selected Configs

| Protocol | Family | Condition | Objective | Adapter Rank | Alpha | Source Holdout AUROC | Source Holdout Rank | Source Holdout Hit@1 | Source Holdout Pairwise |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| source_only_inductive | canonical_22 | frozen_basis_pairwise | pairwise | 0 | 1.00 | 0.4904 | 16.9400 | 42.00% | 49.41% |
| source_only_inductive | canonical_22 | frozen_basis_pointwise | pointwise | 0 | 1.00 | 0.4057 | 18.2000 | 48.00% | 46.29% |
| source_only_inductive | canonical_22 | random_frozen_adapter_pairwise | pairwise | 2 | 1.00 | 0.4904 | 16.9400 | 42.00% | 49.42% |
| source_only_inductive | canonical_22 | random_frozen_adapter_pointwise | pointwise | 8 | 1.00 | 0.4057 | 18.1800 | 48.00% | 46.23% |
| source_only_inductive | canonical_22 | trained_adapter_pairwise | pairwise | 4 | 1.00 | 0.4904 | 16.9467 | 42.00% | 49.44% |
| source_only_inductive | canonical_22 | trained_adapter_pointwise | pointwise | 4 | 0.30 | 0.4071 | 18.2400 | 48.00% | 46.50% |
| source_only_inductive | token_only | frozen_basis_pairwise | pairwise | 0 | 1.00 | 0.4986 | 16.0400 | 54.00% | 49.27% |
| source_only_inductive | token_only | frozen_basis_pointwise | pointwise | 0 | 1.00 | 0.4325 | 16.2800 | 56.00% | 49.56% |
| source_only_inductive | token_only | random_frozen_adapter_pairwise | pairwise | 4 | 0.30 | 0.4986 | 16.0400 | 54.00% | 49.28% |
| source_only_inductive | token_only | random_frozen_adapter_pointwise | pointwise | 8 | 0.10 | 0.4325 | 16.2800 | 56.00% | 49.56% |
| source_only_inductive | token_only | trained_adapter_pairwise | pairwise | 4 | 0.30 | 0.4985 | 16.1000 | 52.00% | 49.34% |
| source_only_inductive | token_only | trained_adapter_pointwise | pointwise | 2 | 1.00 | 0.4338 | 16.2800 | 58.00% | 49.57% |
| transductive_target_unlabeled | canonical_22 | trained_adapter_pairwise | pairwise | 2 | 0.10 | 0.4809 | 17.5400 | 47.33% | 47.78% |
| transductive_target_unlabeled | canonical_22 | trained_adapter_pointwise | pointwise | 2 | 0.10 | 0.4555 | 17.8733 | 42.67% | 46.20% |
| transductive_target_unlabeled | token_only | trained_adapter_pairwise | pairwise | 2 | 0.30 | 0.4950 | 16.6600 | 48.00% | 48.21% |
| transductive_target_unlabeled | token_only | trained_adapter_pointwise | pointwise | 4 | 0.30 | 0.4753 | 17.4467 | 48.00% | 47.06% |

## Target Results

| Protocol | Family | Condition | Objective | Dataset | Anchor | AUROC | Rank | Hit@1 | Hit@3 | SelAcc@10 | Pairwise Acc |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| source_only_inductive | canonical_22 | frozen_basis_pairwise | pairwise | target_ds_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | frozen_basis_pairwise | pairwise | target_ds_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | frozen_basis_pairwise | pairwise | target_ds_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | frozen_basis_pairwise | pairwise | target_qwen_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | frozen_basis_pairwise | pairwise | target_qwen_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | frozen_basis_pairwise | pairwise | target_qwen_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | frozen_basis_pointwise | pointwise | target_ds_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | frozen_basis_pointwise | pointwise | target_ds_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | frozen_basis_pointwise | pointwise | target_ds_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | frozen_basis_pointwise | pointwise | target_qwen_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | frozen_basis_pointwise | pointwise | target_qwen_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | frozen_basis_pointwise | pointwise | target_qwen_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | random_frozen_adapter_pairwise | pairwise | target_ds_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | random_frozen_adapter_pairwise | pairwise | target_ds_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | random_frozen_adapter_pairwise | pairwise | target_ds_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | random_frozen_adapter_pairwise | pairwise | target_qwen_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | random_frozen_adapter_pairwise | pairwise | target_qwen_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | random_frozen_adapter_pairwise | pairwise | target_qwen_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | random_frozen_adapter_pointwise | pointwise | target_ds_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | random_frozen_adapter_pointwise | pointwise | target_ds_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | random_frozen_adapter_pointwise | pointwise | target_ds_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | random_frozen_adapter_pointwise | pointwise | target_qwen_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | random_frozen_adapter_pointwise | pointwise | target_qwen_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | random_frozen_adapter_pointwise | pointwise | target_qwen_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | trained_adapter_pairwise | pairwise | target_ds_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | trained_adapter_pairwise | pairwise | target_ds_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | trained_adapter_pairwise | pairwise | target_ds_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | trained_adapter_pairwise | pairwise | target_qwen_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | trained_adapter_pairwise | pairwise | target_qwen_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | trained_adapter_pairwise | pairwise | target_qwen_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | trained_adapter_pointwise | pointwise | target_ds_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | trained_adapter_pointwise | pointwise | target_ds_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | trained_adapter_pointwise | pointwise | target_ds_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | trained_adapter_pointwise | pointwise | target_qwen_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | trained_adapter_pointwise | pointwise | target_qwen_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | canonical_22 | trained_adapter_pointwise | pointwise | target_qwen_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | frozen_basis_pairwise | pairwise | target_ds_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | frozen_basis_pairwise | pairwise | target_ds_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | frozen_basis_pairwise | pairwise | target_ds_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | frozen_basis_pairwise | pairwise | target_qwen_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | frozen_basis_pairwise | pairwise | target_qwen_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | frozen_basis_pairwise | pairwise | target_qwen_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | frozen_basis_pointwise | pointwise | target_ds_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | frozen_basis_pointwise | pointwise | target_ds_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | frozen_basis_pointwise | pointwise | target_ds_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | frozen_basis_pointwise | pointwise | target_qwen_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | frozen_basis_pointwise | pointwise | target_qwen_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | frozen_basis_pointwise | pointwise | target_qwen_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | random_frozen_adapter_pairwise | pairwise | target_ds_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | random_frozen_adapter_pairwise | pairwise | target_ds_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | random_frozen_adapter_pairwise | pairwise | target_ds_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | random_frozen_adapter_pairwise | pairwise | target_qwen_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | random_frozen_adapter_pairwise | pairwise | target_qwen_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | random_frozen_adapter_pairwise | pairwise | target_qwen_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | random_frozen_adapter_pointwise | pointwise | target_ds_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | random_frozen_adapter_pointwise | pointwise | target_ds_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | random_frozen_adapter_pointwise | pointwise | target_ds_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | random_frozen_adapter_pointwise | pointwise | target_qwen_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | random_frozen_adapter_pointwise | pointwise | target_qwen_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | random_frozen_adapter_pointwise | pointwise | target_qwen_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | trained_adapter_pairwise | pairwise | target_ds_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | trained_adapter_pairwise | pairwise | target_ds_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | trained_adapter_pairwise | pairwise | target_ds_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | trained_adapter_pairwise | pairwise | target_qwen_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | trained_adapter_pairwise | pairwise | target_qwen_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | trained_adapter_pairwise | pairwise | target_qwen_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | trained_adapter_pointwise | pointwise | target_ds_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | trained_adapter_pointwise | pointwise | target_ds_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | trained_adapter_pointwise | pointwise | target_ds_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | trained_adapter_pointwise | pointwise | target_qwen_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | trained_adapter_pointwise | pointwise | target_qwen_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| source_only_inductive | token_only | trained_adapter_pointwise | pointwise | target_qwen_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| transductive_target_unlabeled | canonical_22 | trained_adapter_pairwise | pairwise | target_ds_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| transductive_target_unlabeled | canonical_22 | trained_adapter_pairwise | pairwise | target_ds_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| transductive_target_unlabeled | canonical_22 | trained_adapter_pairwise | pairwise | target_ds_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| transductive_target_unlabeled | canonical_22 | trained_adapter_pairwise | pairwise | target_qwen_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| transductive_target_unlabeled | canonical_22 | trained_adapter_pairwise | pairwise | target_qwen_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| transductive_target_unlabeled | canonical_22 | trained_adapter_pairwise | pairwise | target_qwen_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| transductive_target_unlabeled | canonical_22 | trained_adapter_pointwise | pointwise | target_ds_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| transductive_target_unlabeled | canonical_22 | trained_adapter_pointwise | pointwise | target_ds_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| transductive_target_unlabeled | canonical_22 | trained_adapter_pointwise | pointwise | target_ds_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| transductive_target_unlabeled | canonical_22 | trained_adapter_pointwise | pointwise | target_qwen_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| transductive_target_unlabeled | canonical_22 | trained_adapter_pointwise | pointwise | target_qwen_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| transductive_target_unlabeled | canonical_22 | trained_adapter_pointwise | pointwise | target_qwen_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| transductive_target_unlabeled | token_only | trained_adapter_pairwise | pairwise | target_ds_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| transductive_target_unlabeled | token_only | trained_adapter_pairwise | pairwise | target_ds_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| transductive_target_unlabeled | token_only | trained_adapter_pairwise | pairwise | target_ds_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| transductive_target_unlabeled | token_only | trained_adapter_pairwise | pairwise | target_qwen_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| transductive_target_unlabeled | token_only | trained_adapter_pairwise | pairwise | target_qwen_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| transductive_target_unlabeled | token_only | trained_adapter_pairwise | pairwise | target_qwen_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| transductive_target_unlabeled | token_only | trained_adapter_pointwise | pointwise | target_ds_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| transductive_target_unlabeled | token_only | trained_adapter_pointwise | pointwise | target_ds_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| transductive_target_unlabeled | token_only | trained_adapter_pointwise | pointwise | target_ds_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |
| transductive_target_unlabeled | token_only | trained_adapter_pointwise | pointwise | target_qwen_test | 100 | n/a | n/a | n/a | n/a | n/a | n/a |
| transductive_target_unlabeled | token_only | trained_adapter_pointwise | pointwise | target_qwen_test | 70 | n/a | n/a | n/a | n/a | n/a | n/a |
| transductive_target_unlabeled | token_only | trained_adapter_pointwise | pointwise | target_qwen_test | late_mean | n/a | n/a | n/a | n/a | n/a | n/a |

## Source-Holdout Effect

| Family | Objective | Adapter Rank | Alpha | ΔAUROC | ΔRank | ΔHit@1 | ΔPairwise |
|---|---|---:|---:|---:|---:|---:|---:|
| canonical_22 | pairwise | 2 | 0.10 | -0.0096 | -0.5800 | 5.33% | -1.60% |
| canonical_22 | pointwise | 2 | 0.10 | 0.0497 | 0.3267 | -5.33% | -0.16% |
| token_only | pairwise | 2 | 0.30 | -0.0034 | -0.5933 | -6.00% | -1.12% |
| token_only | pointwise | 4 | 0.30 | 0.0415 | -1.1467 | -10.00% | -2.47% |

## Unlabeled Target Effect

| Family | Objective | Dataset | Adapter Rank | Alpha | Proxy Ref | Score Corr | Top1 Agree | Top1 Flip | Mean Abs Score Δ | ΔAnchor Gap | ΔCov Gap | Adapter Residual |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| canonical_22 | pairwise | target_ds_test | 2 | 0.10 | matched_source_only_trained_adapter_pairwise | 0.8639 | 22.65% | 77.35% | 8.4660 | -0.0221 | -0.6449 | 0.5228 |
| canonical_22 | pairwise | target_qwen_test | 2 | 0.10 | matched_source_only_trained_adapter_pairwise | 0.8898 | 21.46% | 78.54% | 7.7771 | -0.0033 | -0.6457 | 0.5228 |
| canonical_22 | pointwise | target_ds_test | 2 | 0.10 | matched_source_only_trained_adapter_pointwise | 0.3374 | 4.69% | 95.31% | 2.0380 | -0.1285 | -0.5676 | 0.5442 |
| canonical_22 | pointwise | target_qwen_test | 2 | 0.10 | matched_source_only_trained_adapter_pointwise | 0.2186 | 6.79% | 93.21% | 1.6750 | -0.0975 | -0.5781 | 0.5442 |
| token_only | pairwise | target_ds_test | 2 | 0.30 | matched_source_only_trained_adapter_pairwise | -0.1816 | 0.00% | 100.00% | 10.9688 | -0.0079 | -0.5518 | 1.0331 |
| token_only | pairwise | target_qwen_test | 2 | 0.30 | matched_source_only_trained_adapter_pairwise | -0.0855 | 0.00% | 100.00% | 10.8140 | 0.0224 | -0.7984 | 1.0331 |
| token_only | pointwise | target_ds_test | 4 | 0.30 | matched_source_only_trained_adapter_pointwise | 0.1904 | 1.30% | 98.70% | 33.1312 | -0.1502 | -0.7503 | 1.3765 |
| token_only | pointwise | target_qwen_test | 4 | 0.30 | matched_source_only_trained_adapter_pointwise | 0.2278 | 0.00% | 100.00% | 34.6933 | -0.3253 | -1.0415 | 1.3765 |

## Questions

1. **Does a tiny low-rank adapter help DS->Qwen transfer?** unresolved locally.
   - Local target correctness labels are unavailable, so this artifact cannot prove or disprove target transfer offline.
   - Proxy readout: Qwen proxy: mean corr=0.3127, top1 flip=92.94%, Δcov=-0.7659; DS-test proxy: mean corr=0.3025, top1 flip=92.84%, Δcov=-0.6286.
2. **Is the gain larger for pairwise ranking than for pointwise correctness?** proxy-leaning yes.
   - Proxy stability on Qwen: pairwise corr `0.4021` / flip `89.27%` vs pointwise corr `0.2232` / flip `96.61%`.
3. **Does token_only benefit more than canonical_22?** proxy-leaning no.
   - Proxy stability on Qwen: token corr `0.0712` / flip `100.00%` vs canonical corr `0.5542` / flip `85.88%`.
4. **Is the improvement mainly in Rank / Hit@1 / Pairwise Acc rather than AUROC?** source-holdout leaning no.
   - Source-holdout deltas: ranking-focused `-0.0085` vs pointwise AUROC `0.0456`.
5. **Does this support weak reusable relative signal but unstable absolute correctness?** mixed.

## Interpretability

- Final adapted score remains affine in the frozen latent coordinates.
- With column-vector notation `z' = M z`, `M = I + A B^T`, and `score = w^T z' + b`,
  the exact frozen-basis score is `score = (M^T w)^T z + b`.
- With the frozen SVD map `z = V x_scaled` (equivalently `x_scaled @ V^T` in row form),
  the original raw+rank feature score is still linear after back-projection through the fixed scaler + SVD.

| Family | Condition | Top Back-Projected Features |
|---|---|---|
| canonical_22 | trained_adapter_pairwise | `raw::tok_gini_slope` (-2.5483), `raw::traj_novelty` (+2.3883), `raw::tok_gini_prefix` (+2.0525), `raw::traj_continuity` (-1.5365), `raw::tok_logprob_prefix` (+1.1849) |
| canonical_22 | trained_adapter_pointwise | `raw::traj_novelty` (+5.5331), `raw::traj_continuity` (-4.7497), `raw::tok_gini_slope` (-2.6228), `raw::tok_gini_prefix` (+0.9967), `rank::traj_max_reflection` (-0.7932) |
| token_only | trained_adapter_pairwise | `raw::tok_gini_slope` (+9.3641), `raw::tok_logprob_prefix` (+7.7367), `raw::tok_logprob_recency` (+7.7136), `raw::tok_gini_prefix` (+1.6793), `rank::tok_logprob_recency` (-1.6343) |
| token_only | trained_adapter_pointwise | `raw::tok_gini_slope` (+6.6887), `raw::tok_gini_prefix` (+3.1950), `rank::tok_gini_prefix` (+1.0074), `rank::tok_gini_slope` (-0.9588), `rank::tok_logprob_recency` (-0.8637) |

## Notes

- Target labels are never used during training or transductive adaptation.
- The optional full-label target upper reference is intentionally omitted from this artifact.
- Exact hyperparameters are stored row-wise in `results/tables/coding_random_adapter.csv`.

