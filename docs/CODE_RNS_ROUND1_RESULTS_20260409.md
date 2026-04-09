# Code RNS Round 1 — Implementation + Smoke Result

Date: `2026-04-09`  
Smoke payload: `result/code_rns_round1_smoke/code_rns_round1.json`  
Runner: `scripts/run_code_rns_round1.py`  
Export helper: `scripts/patch_bestofn_submission_with_code_rns_round1.py`

## 1. Confirmed Current Repo State

Before this line was added:

1. `code_v2` is still the promoted coding default.
2. `science_hybrid_round3` is still the promoted science patch.
3. `gpqa_deepsets_round1` remains `NO-PROMOTE` because the `Hit@1` guardrail did not pass.
4. `code_deepsets_round1` remains `NO-PROMOTE` because `SelAcc@10` regressed materially.
5. So the coding-side continuation should stay inside the existing small feature family and focus on **top-slot vs shortlist calibration**, not bigger context.

## 2. What Was Implemented

This round does **not** train a new neural model family.

Implemented instead:

- a new coding reranker in `nad/core/selectors/code_rns_impl.py`
- a plugin entrypoint in `plugins/code_rns_selector.py`
- a round runner in `scripts/run_code_rns_round1.py`
- a DS-only Best-of-N export helper in `scripts/patch_bestofn_submission_with_code_rns_round1.py`

Core idea:

- stay in the current `code_v2` 5-feature rank space:
  - `prefix_best_window_quality_r`
  - `head_tail_gap_r`
  - `tail_variance_r`
  - `post_reflection_recovery_r`
  - `last_block_instability_r`
- build an external anchor bank from historical labeled coding runs:
  - positive anchors = correct runs
  - hard negative anchors = wrong runs
  - optional small synthetic CF negatives = keep prefix-like features, replace tail-like features from matched hard negatives
- compute `neg_fraction@k` in the joint anchor pool
- rerank **only** the `code_v2` shortlist, instead of replacing the full coding score

Operationally this keeps the work conservative:

- no change to `code_v2` default weights
- no change to `science_hybrid_round3`
- no new DeepSets / Set Transformer / cross-run attention line
- no EarlyStop changes

## 3. Search Grid Implemented

The runner currently compares 8 candidates:

1. `code_v2_baseline`
2. `rns_hard_only__top10__knn5__lam0p15`
3. `rns_hard_cf__top10__knn5__lam0p15`
4. `rns_hard_cf__top10__knn5__lam0p25`
5. `rns_hard_cf__top10__knn3__lam0p15`
6. `rns_hard_cf__top10__knn8__lam0p15`
7. `rns_hard_cf__top5__knn5__lam0p15`
8. `rns_hard_cf__top5__knn5__lam0p25`

Guardrails implemented in the runner:

- coding gate:
  - `SelAcc@10 >= current code_v2`
  - `Pairwise >= 50%`
  - `Hit@1 >= current code_v2 - 0.25pp`
- system gate:
  - sample-weighted full-system `Hit@1` does not regress
  - sample-weighted full-system `SelAcc@10` does not regress

## 4. Smoke Validation

Smoke command used:

```bash
python scripts/run_code_rns_round1.py \
  --max-problems 5 \
  --distance-threads 4 \
  --out-dir result/code_rns_round1_smoke \
  --skip-save-model
```

Smoke read:

- the full runner executes successfully
- the blind Qwen risk read also executes successfully
- the recommendation on the 5-problem smoke stays `code_v2_baseline`
- so the new line is **implemented and runnable**, but the smoke does **not** justify promotion

### Smoke compare table

| Candidate | Hit@1 | Pairwise | SelAcc@10 | SW ΔHit@1 | SW ΔSelAcc@10 | Blind Qwen flips | Coding gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `code_v2_baseline` | 60.00% | 50.23% | **71.88%** | +2.75pp | +2.18pp | `0 / 5` | pass |
| `rns_hard_only__top10__knn5__lam0p15` | 60.00% | 50.27% | 65.62% | +2.75pp | +2.08pp | `0 / 5` | fail |
| `rns_hard_cf__top10__knn5__lam0p15` | 60.00% | 50.23% | 65.62% | +2.75pp | +2.08pp | `1 / 5` | fail |
| `rns_hard_cf__top10__knn5__lam0p25` | 60.00% | 50.19% | 65.62% | +2.75pp | +2.08pp | `2 / 5` | fail |
| `rns_hard_cf__top10__knn3__lam0p15` | 60.00% | **50.44%** | 65.62% | +2.75pp | +2.08pp | `2 / 5` | fail |
| `rns_hard_cf__top10__knn8__lam0p15` | 60.00% | 50.19% | 65.62% | +2.75pp | +2.08pp | `1 / 5` | fail |
| `rns_hard_cf__top5__knn5__lam0p15` | 60.00% | 50.27% | 65.62% | +2.75pp | +2.08pp | `1 / 5` | fail |
| `rns_hard_cf__top5__knn5__lam0p25` | 60.00% | 50.27% | 65.62% | +2.75pp | +2.08pp | `2 / 5` | fail |

Important caution:

- this table is only a **smoke subset**
- it is useful for implementation validation
- it is **not** the basis for a promote / no-promote production decision

## 5. Full Run Status

Full run entrypoint is ready:

```bash
python scripts/run_code_rns_round1.py \
  --distance-threads 8 \
  --out-dir result/code_rns_round1_$(date -u +%Y%m%d_%H%M%S)
```

Current status in this turn:

- full-cache preload was started
- but it is materially slower than the smoke path because it scans all coding problems and all blind Qwen risk problems
- so this implementation handoff stops at a successful smoke validation, not a completed full-round promote decision

## 6. Export Path

Once a full run produces a candidate worth exporting and saves `models/ml_selectors/code_rns_round1.pkl`, the DS-only coding patch can be exported with:

```bash
python scripts/patch_bestofn_submission_with_code_rns_round1.py \
  --model-path models/ml_selectors/code_rns_round1.pkl
```

Default behavior of the export helper:

- base submission:
  `submission/BestofN/extreme12/patches/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa__no_math_patch.json`
- patched cache key:
  `DS-R1/lcb_v5`
- output file:
  `submission/BestofN/extreme12/patches/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa__code_rns_ds_lcb_patch.json`

## 7. Files Added / Changed

Added:

- `nad/core/selectors/code_rns_impl.py`
- `plugins/code_rns_selector.py`
- `scripts/run_code_rns_round1.py`
- `scripts/patch_bestofn_submission_with_code_rns_round1.py`
- `docs/CODE_RNS_ROUND1_RESULTS_20260409.md`

Updated:

- `docs/README.md`
- `scripts/README.md`

## 8. 中文结论

这轮已经把你要的 **coding 上的保守 RNS shortlist 校准线** 真正接进仓库了，而且 smoke 已经跑通。

当前可以确认的只有两点：

1. **实现是通的**：  
   新 scorer、plugin、runner、Best-of-N 导出脚本都已经落地。
2. **还不能宣布 promote**：  
   目前只有 `5` 题 smoke 结果，推荐仍然是 `code_v2_baseline`；完整 `167` 题 coding + blind Qwen 风险读取还需要正式长跑。

所以现在最准确的状态是：

- **implementation complete**
- **smoke validated**
- **full decision pending full run**
