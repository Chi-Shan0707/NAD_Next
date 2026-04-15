# Coding Random Adapter

**Date**: 2026-04-15T07:46:28Z  
**Status**: completed

> This artifact is a smoke-sized validation run. Interpret answers as provisional.

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
| source_only_inductive | canonical_22 | frozen_basis_pairwise | pairwise | 0 | 1.00 | 0.4755 | 1.5000 | 50.00% | 47.55% |
| source_only_inductive | canonical_22 | frozen_basis_pointwise | pointwise | 0 | 1.00 | 0.5141 | 2.0000 | 50.00% | 51.41% |
| source_only_inductive | canonical_22 | random_frozen_adapter_pairwise | pairwise | 2 | 0.30 | 0.4755 | 1.5000 | 50.00% | 47.55% |
| source_only_inductive | canonical_22 | random_frozen_adapter_pointwise | pointwise | 2 | 0.30 | 0.5146 | 2.0000 | 50.00% | 51.46% |
| source_only_inductive | canonical_22 | trained_adapter_pairwise | pairwise | 2 | 0.30 | 0.4766 | 1.5000 | 50.00% | 47.66% |
| source_only_inductive | canonical_22 | trained_adapter_pointwise | pointwise | 2 | 0.30 | 0.5135 | 2.0000 | 50.00% | 51.35% |
| source_only_inductive | token_only | frozen_basis_pairwise | pairwise | 0 | 1.00 | 0.4854 | 1.0000 | 100.00% | 48.54% |
| source_only_inductive | token_only | frozen_basis_pointwise | pointwise | 0 | 1.00 | 0.5464 | 2.0000 | 0.00% | 54.64% |
| source_only_inductive | token_only | random_frozen_adapter_pairwise | pairwise | 2 | 0.30 | 0.4854 | 1.0000 | 100.00% | 48.54% |
| source_only_inductive | token_only | random_frozen_adapter_pointwise | pointwise | 2 | 0.30 | 0.5458 | 2.0000 | 0.00% | 54.58% |
| source_only_inductive | token_only | trained_adapter_pairwise | pairwise | 2 | 0.30 | 0.4854 | 1.0000 | 100.00% | 48.54% |
| source_only_inductive | token_only | trained_adapter_pointwise | pointwise | 2 | 0.30 | 0.5458 | 2.0000 | 0.00% | 54.58% |
| transductive_target_unlabeled | canonical_22 | trained_adapter_pairwise | pairwise | 2 | 0.30 | 0.5188 | 2.0000 | 50.00% | 51.88% |
| transductive_target_unlabeled | canonical_22 | trained_adapter_pointwise | pointwise | 2 | 0.30 | 0.4771 | 3.5000 | 0.00% | 47.71% |
| transductive_target_unlabeled | token_only | trained_adapter_pairwise | pairwise | 2 | 0.30 | 0.5193 | 2.0000 | 0.00% | 51.93% |
| transductive_target_unlabeled | token_only | trained_adapter_pointwise | pointwise | 2 | 0.30 | 0.4964 | 3.0000 | 0.00% | 49.64% |

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
| canonical_22 | pairwise | 2 | 0.30 | 0.0422 | -0.5000 | 0.00% | 4.22% |
| canonical_22 | pointwise | 2 | 0.30 | -0.0365 | -1.5000 | -50.00% | -3.65% |
| token_only | pairwise | 2 | 0.30 | 0.0339 | -1.0000 | -100.00% | 3.39% |
| token_only | pointwise | 2 | 0.30 | -0.0495 | -1.0000 | 0.00% | -4.95% |

## Unlabeled Target Effect

| Family | Objective | Dataset | Adapter Rank | Alpha | Proxy Ref | Score Corr | Top1 Agree | Top1 Flip | Mean Abs Score Δ | ΔAnchor Gap | ΔCov Gap | Adapter Residual |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| canonical_22 | pairwise | target_ds_test | 2 | 0.30 | matched_source_only_trained_adapter_pairwise | 0.8467 | 37.50% | 62.50% | 0.3118 | 0.0176 | 0.0013 | 0.0470 |
| canonical_22 | pairwise | target_qwen_test | 2 | 0.30 | matched_source_only_trained_adapter_pairwise | 0.7930 | 37.50% | 62.50% | 0.5545 | 0.0216 | -0.0000 | 0.0470 |
| canonical_22 | pointwise | target_ds_test | 2 | 0.30 | matched_source_only_trained_adapter_pointwise | 0.8680 | 37.50% | 62.50% | 0.1498 | -0.0951 | 0.0018 | 0.0123 |
| canonical_22 | pointwise | target_qwen_test | 2 | 0.30 | matched_source_only_trained_adapter_pointwise | 0.9090 | 75.00% | 25.00% | 0.3387 | -0.1031 | -0.0001 | 0.0123 |
| token_only | pairwise | target_ds_test | 2 | 0.30 | matched_source_only_trained_adapter_pairwise | 0.9000 | 25.00% | 75.00% | 0.2163 | -0.0128 | 0.0014 | 0.0245 |
| token_only | pairwise | target_qwen_test | 2 | 0.30 | matched_source_only_trained_adapter_pairwise | 0.8726 | 37.50% | 62.50% | 0.4030 | 0.0184 | -0.0003 | 0.0245 |
| token_only | pointwise | target_ds_test | 2 | 0.30 | matched_source_only_trained_adapter_pointwise | 0.9163 | 37.50% | 62.50% | 0.0814 | -0.0369 | 0.0040 | 0.0273 |
| token_only | pointwise | target_qwen_test | 2 | 0.30 | matched_source_only_trained_adapter_pointwise | 0.9170 | 37.50% | 62.50% | 0.1117 | -0.0607 | -0.0007 | 0.0273 |

## Questions

1. **Does a tiny low-rank adapter help DS->Qwen transfer?** unresolved locally.
   - Local target correctness labels are unavailable, so this artifact cannot prove or disprove target transfer offline.
   - Proxy readout: Qwen proxy: mean corr=0.8729, top1 flip=53.12%, Δcov=-0.0003; DS-test proxy: mean corr=0.8828, top1 flip=65.62%, Δcov=0.0021.
2. **Is the gain larger for pairwise ranking than for pointwise correctness?** proxy-leaning no.
   - Proxy stability on Qwen: pairwise corr `0.8328` / flip `62.50%` vs pointwise corr `0.9130` / flip `43.75%`.
3. **Does token_only benefit more than canonical_22?** proxy-leaning no.
   - Proxy stability on Qwen: token corr `0.8948` / flip `62.50%` vs canonical corr `0.8510` / flip `43.75%`.
4. **Is the improvement mainly in Rank / Hit@1 / Pairwise Acc rather than AUROC?** source-holdout leaning no.
   - Source-holdout deltas: ranking-focused `-0.2310` vs pointwise AUROC `-0.0430`.
5. **Does this support weak reusable relative signal but unstable absolute correctness?** proxy-leaning no.

## Interpretability

- Final adapted score remains affine in the frozen latent coordinates.
- With column-vector notation `z' = M z`, `M = I + A B^T`, and `score = w^T z' + b`,
  the exact frozen-basis score is `score = (M^T w)^T z + b`.
- With the frozen SVD map `z = V x_scaled` (equivalently `x_scaled @ V^T` in row form),
  the original raw+rank feature score is still linear after back-projection through the fixed scaler + SVD.

| Family | Condition | Top Back-Projected Features |
|---|---|---|
| canonical_22 | trained_adapter_pairwise | `raw::tok_gini_slope` (+3.7394), `raw::traj_novelty` (-1.6386), `raw::traj_continuity` (+0.9801), `rank::traj_reflection_count` (+0.6609), `rank::traj_novelty` (-0.4637) |
| canonical_22 | trained_adapter_pointwise | `raw::traj_novelty` (-0.9525), `raw::traj_continuity` (+0.5120), `raw::tok_gini_slope` (+0.4956), `raw::tok_gini_tail` (-0.2782), `rank::traj_novelty` (-0.2456) |
| token_only | trained_adapter_pairwise | `raw::tok_gini_prefix` (-1.7498), `raw::tok_gini_slope` (-1.6977), `raw::tok_gini_tail` (+1.0985), `rank::tok_gini_slope` (+0.7360), `rank::tok_gini_prefix` (-0.6572) |
| token_only | trained_adapter_pointwise | `raw::tok_gini_tail` (-0.6203), `rank::tok_gini_prefix` (-0.3993), `raw::tok_gini_prefix` (-0.3253), `rank::tok_neg_entropy_recency` (+0.2639), `rank::tok_selfcert_recency` (+0.2639) |

## Notes

- Target labels are never used during training or transductive adaptation.
- The optional full-label target upper reference is intentionally omitted from this artifact.
- Exact hyperparameters are stored row-wise in `results/tables/coding_random_adapter.csv`.

