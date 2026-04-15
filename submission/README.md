# Submission Layout

## Families
- `BestofN/`: best-of-n submission JSONs, now grouped into `extreme8/` and `extreme12/` subtrees.
- `EarlyStop/`: early-stop submission JSONs.

## Current notable BestofN patch
- `submission/BestofN/extreme12/patches/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa__math_deepsets_round1_patch.json`: current promoted-stack BestofN export with the new math DeepSets patch.
- `submission/BestofN/extreme12/patches/extreme12_earlystop_prefix10_svd_round1_slot100__svm_bridge_lcb__coding_improvement_v1_tier1_xgb_lcb_patch.json`: LCB-only BestofN patch that keeps the `extreme12_earlystop_prefix10_svd_round1_slot100__svm_bridge_lcb` base and replaces only `DS-R1/lcb_v5` + `Qwen3-4B/lcb_v5` using the `coding_improvement_v1` Tier-1 XGBoost scorer.
- `submission/BestofN/extreme12/patches/extreme12_earlystop_prefix10_svd_round1_slot100__svm_bridge_lcb__coding_improvement_v2_hybrid_bridge_patch.json`: activation-bridge coding patch that reuses the slot100 routes, adds grouped neuron-activation summaries, and restores the coding bridge to the `code_v2` validation ceiling on LCB.

## Naming convention
- Keep stable baselines and experimental patches in the filename.
- Prefer adding new files over overwriting a previously validated export.
