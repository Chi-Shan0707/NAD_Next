# Extreme12 Patches

Patched Extreme12 submissions for coding / science / math overrides.

Current recovery candidates:

- `extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa__no_math_patch.json`
- `extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa__math_patch_ds_aime25_only.json`
- `extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa__math_patch_ds_qwen_aime25_brumo25.json`
- `nogithub_detailed_qwen3_lcb_from_rotation_adapter_late70100.json`: keeps `0415patch.json` / `method_name=nogithub` as the base and replaces only `Qwen3-4B/lcb_v5` with the slot100 export from `es_svd_ms_rr_r1__coding_rotation_adapter_late70100`.
- `es_svd_ms_rr_r1__coding_rotation_adapter_late70100__slot100_bestofn.json`: intermediate Best-of-N export produced by extracting slot `9` (`position=1.0`) from the early-stop rotation-adapter submission.

Research note:

- new LambdaSVM / DeepSets round2 / SetTransformerLite work is offline-only for now
- no new research candidate is auto-promoted into this folder until it passes the relevant Best-of-N gates

Historical broad math patch:

- `extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa__math_deepsets_round1_patch.json`
- `extreme12_svm_bridge_bestofn_v1.json`
