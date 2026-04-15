# 15 — Coding Hybrid Bridge

**Date**: 2026-04-14T00:45:38Z  **Status**: completed

## Verdict

- **Selected scorer**: `meta_xgb_slot100_activation` (meta)
- **Top1**: `0.6168`
- **Uplift vs `code_v2`**: `+0.0000`
- **Patch output**: `submission/BestofN/extreme12/patches/extreme12_earlystop_prefix10_svd_round1_slot100__svm_bridge_lcb__coding_improvement_v2_hybrid_bridge_patch.json`

## Best Fixed Slot100 Routes

| Route | Top1 | Uplift vs `code_v2` | Pairwise |
|---|---:|---:|---:|
| `slot100::slot100_svd_code_domain_r1_focus20__hit1` | 0.5928 | -0.0240 | 0.5112 |
| `slot100::slot100_svd_code_domain_r1_focus20__pairwise` | 0.5928 | -0.0240 | 0.5112 |
| `slot100::slot100_svd_code_domain_r1_cap10__hit1` | 0.5868 | -0.0299 | 0.4890 |
| `slot100::slot100_svd_code_domain_r1_cap10__pairwise` | 0.5868 | -0.0299 | 0.4833 |

## Meta Candidates

| Candidate | Blocks | Top1 | Uplift vs `code_v2` | Pairwise |
|---|---|---:|---:|---:|
| `meta_xgb_slot100_activation` | `slot100, activation` | 0.6168 | +0.0000 | 0.5029 |
| `meta_logreg_slot100_code_v2_activation` | `slot100, code_v2, activation` | 0.6048 | -0.0120 | 0.5042 |
| `meta_xgb_slot100_code_v2_activation` | `slot100, code_v2, activation` | 0.6048 | -0.0120 | 0.4995 |
| `meta_xgb_slot100_code_v2` | `slot100, code_v2` | 0.5749 | -0.0419 | 0.4891 |
| `meta_xgb_slot100` | `slot100` | 0.5629 | -0.0539 | 0.4939 |

## Blind Export Stats

| Cache | Problems | Samples | Score mean | Score std |
|---|---:|---:|---:|---:|
| `DS-R1/lcb_v5` | 167 | 10688 | 0.582256 | 0.162824 |
| `Qwen3-4B/lcb_v5` | 167 | 10688 | 0.703302 | 0.096875 |
