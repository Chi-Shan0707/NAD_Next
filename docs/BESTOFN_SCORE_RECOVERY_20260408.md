# Best-of-N Score Recovery / Score Improvement

Date: `2026-04-08`  
Script: `scripts/run_bestofn_score_recovery_20260408.py`  
Result payload: `result/bestofn_score_recovery_20260408/score_recovery.json`

## 0. Current Repo State Confirmed

- `code_v2` is the current promoted coding default.
- `science_hybrid_round3` is the current promoted science patch.
- `gpqa_deepsets_round1` improved `AUROC / SelAcc@10`, but remains `NO-PROMOTE` because the `Hit@1` guardrail did not pass.
- `code_deepsets_round1` also remains `NO-PROMOTE` because `SelAcc@10` regressed materially.
- Therefore this round does **not** train any new model. The only question is: **how to improve Best-of-N more safely inside the current production stack**.

## 1. How Submission #92 Is Handled

User-provided known fact:

- public Best-of-N `Submission #92`
- method:
  `extreme12_baseline12_pointwise_best_only_ref030_t1024__code_v2_lcb__science_hybrid_round3_gpqa__math_deepsets_round1_math_patch`
- result: `Not best`

Operational consequence in this round:

- `math_deepsets_round1_math_patch` is now treated as a **high-risk production patch**
- the previous “full math patch promote” conclusion is treated as **historical**, not as a current deployment recommendation
- this round starts from **rollback first**, then tests only **very small selective math patch** candidates

## 2. Proxy Definition Used In This Recovery Pass

The repo still does **not** contain a trustworthy leaderboard reproducer for the full 12-cache blind submission.  
So this round uses two explicit views instead of pretending otherwise.

### 2.1 GT-backed proxy

This is the only honest offline full-system proxy currently available in-repo:

- `DS-R1/aime24`
- `DS-R1/aime25`
- `DS-R1/brumo25`
- `DS-R1/gpqa`
- `DS-R1/hmmt25`
- `DS-R1/lcb_v5`

Views reported for each candidate:

- `sample-weighted proxy`
- `equal-cache-mean proxy`
- `math-only proxy`
- `full-system proxy`

Important implementation note:

- the selective math search here replays the **actual exported math patch scoring path** with the frozen `math_deepsets_round1.pkl`
- this is intentionally closer to “current production patch behavior” than the older math-round1 study

### 2.2 Blind risk read

`Qwen3-4B` only exists in `cache_test`, not in `MUI_HUB/cache`, so there is no GT-backed offline proxy for it.  
Therefore `Qwen3-4B` candidates are judged by:

- how many blind math cache keys they touch
- how many blind problems they modify
- how many top-1 picks they flip vs the rollback baseline
- whether they look too similar to the broad `#92` failure pattern

## 3. Rollback Baseline vs Full Math Patch

This is the key recovery check.

### 3.1 No-math production baseline

Current rollback baseline:

- `generic baseline12 + code_v2 + science_hybrid_round3`
- candidate name in this document: `no_math_patch`

GT-backed full-system proxy:

| System | sample-weighted Hit@1 | sample-weighted Pairwise | sample-weighted SelAcc@10 | equal-cache Hit@1 | equal-cache Pairwise | equal-cache SelAcc@10 |
|---|---:|---:|---:|---:|---:|---:|
| `no_math_patch` | 67.22% | 61.29% | 66.15% | 70.53% | 71.53% | 70.90% |

### 3.2 Full math patch replay

Full patch candidate:

- `full_math_patch_all8`
- exact blind patch scope = all 8 math caches

GT-backed delta vs rollback baseline:

- sample-weighted `Hit@1`: `+0.21pp`
- sample-weighted `Pairwise`: `-1.08pp`
- sample-weighted `SelAcc@10`: `-0.58pp`
- equal-cache `Hit@1`: `+0.55pp`
- equal-cache `Pairwise`: `-2.92pp`
- equal-cache `SelAcc@10`: `-1.56pp`

Blind exposure:

- patched cache keys: `8`
- patched math problems: `240`
- blind top-1 changes vs rollback baseline: `184`
- changed rate over patched problems: `76.67%`
- Qwen-only changed problems: `99 / 120`

Read:

- after replaying the **actual exported patch behavior**, the full math patch is **not** a stable guarded improvement anymore
- it now looks much closer to the online `#92` failure signal than to a safe production patch
- therefore the production question is no longer “promote full math patch or not”
- it is “**rollback by default, and only keep a very small selective opt-in patch if it earns that right**”

## 4. Selective Math Patch Search Space

This pass keeps the search small and manual.

Compared candidates:

1. `no_math_patch`
2. `full_math_patch_all8`
3. `ds_only_all4`
4. `qwen_only_all4`
5. `aime_only_both_models`
6. `brumo_only_both_models`
7. `hmmt_only_both_models`
8. `ds_only_aime25`
9. `ds_only_aime25_brumo25`
10. `aime25_only_both_models`
11. `aime25_brumo25_both_models`
12. `all_except_hmmt_both_models`

Dataset-level read from the GT-backed DS-R1 replay:

- `aime25` is the cleanest positive slice
- `brumo25` is the secondary positive slice
- `aime24` does not justify inclusion in this recovery pass
- `hmmt25` behaves like a regression-prone slice and should be excluded

## 5. Candidate Proxy Results And Risk Rating

Notation:

- `SW ΔH1` = sample-weighted full-system `Hit@1` delta vs `no_math_patch`
- `SW ΔSel10` = sample-weighted full-system `SelAcc@10` delta vs `no_math_patch`
- `EQ ΔH1` / `EQ ΔSel10` = equal-cache deltas vs `no_math_patch`
- `Math H1` / `Math Sel10` = GT-backed DS-R1 math-only proxy
- `Blind` = changed top-1 / patched problems on blind math caches

| Candidate | Patch Scope | SW ΔH1 | SW ΔSel10 | EQ ΔH1 | EQ ΔSel10 | Math H1 | Math Sel10 | Blind | Risk |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `no_math_patch` | none | `+0.00pp` | `+0.00pp` | `+0.00pp` | `+0.00pp` | `73.34%` | `74.74%` | `0 / 0` | `low risk` |
| `full_math_patch_all8` | all 8 math caches | `+0.21pp` | `-0.58pp` | `+0.55pp` | `-1.56pp` | `74.17%` | `72.40%` | `184 / 240` | `high risk` |
| `ds_only_all4` | all 4 DS math caches | `+0.21pp` | `-0.58pp` | `+0.55pp` | `-1.56pp` | `74.17%` | `72.40%` | `85 / 120` | `high risk` |
| `qwen_only_all4` | all 4 Qwen math caches | `+0.00pp` | `+0.00pp` | `+0.00pp` | `+0.00pp` | `73.34%` | `74.74%` | `99 / 120` | `high risk` |
| `aime_only_both_models` | `aime24+aime25`, both models | `+0.41pp` | `-0.03pp` | `+1.11pp` | `-0.09pp` | `75.00%` | `74.61%` | `91 / 120` | `high risk` |
| `brumo_only_both_models` | `brumo25`, both models | `+0.21pp` | `-0.29pp` | `+0.56pp` | `-0.78pp` | `74.17%` | `73.57%` | `49 / 60` | `medium risk` |
| `hmmt_only_both_models` | `hmmt25`, both models | `-0.41pp` | `-0.26pp` | `-1.11pp` | `-0.69pp` | `71.67%` | `73.70%` | `44 / 60` | `high risk` |
| `ds_only_aime25` | `DS-R1/aime25` | `+1.03pp` | `+0.32pp` | `+2.78pp` | `+0.87pp` | `77.50%` | `76.04%` | `20 / 30` | `low risk` |
| `ds_only_aime25_brumo25` | `DS-R1/aime25+brumo25` | `+1.24pp` | `+0.03pp` | `+3.33pp` | `+0.09pp` | `78.33%` | `74.87%` | `43 / 60` | `medium risk` |
| `aime25_only_both_models` | `aime25`, both models | `+1.03pp` | `+0.32pp` | `+2.78pp` | `+0.87pp` | `77.50%` | `76.04%` | `43 / 60` | `medium risk` |
| `aime25_brumo25_both_models` | `aime25+brumo25`, both models | `+1.24pp` | `+0.03pp` | `+3.33pp` | `+0.09pp` | `78.33%` | `74.87%` | `92 / 120` | `high risk` |
| `all_except_hmmt_both_models` | all except `hmmt25`, both models | `+0.62pp` | `-0.32pp` | `+1.67pp` | `-0.87pp` | `75.83%` | `73.44%` | `140 / 180` | `high risk` |

## 6. Recommendation

### 6.1 Next formal submission

Recommended next formal submission:

- `math_patch_ds_aime25_only`
- file:
  `submission/BestofN/extreme12/patches/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa__math_patch_ds_aime25_only.json`

Why:

- it is the strongest **guarded** positive point in this recovery pass
- it improves both sample-weighted and equal-cache full-system proxy
- it improves `Pairwise` as well, unlike the broader patch candidates
- it patches only `1` cache key and does **not** touch Qwen blind math
- it is the least similar selective candidate to the broad `#92` failure mode

### 6.2 Conservative backup

Conservative backup:

- `no_math_patch`
- file:
  `submission/BestofN/extreme12/patches/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa__no_math_patch.json`

Use:

- rollback baseline
- safest recovery submission if we want to eliminate math-patch risk entirely

### 6.3 Aggressive score shot

Aggressive but still explainable candidate:

- `math_patch_ds_qwen_aime25_brumo25`
- file:
  `submission/BestofN/extreme12/patches/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa__math_patch_ds_qwen_aime25_brumo25.json`

Use:

- probe whether the positive DS core (`aime25 + brumo25`) also transfers online to Qwen blind math
- this is **not** the safest choice
- it is exported only as a controlled score-shot candidate, not as the default recommendation

## 7. Final Answer To The Production Question

### 7.1 Should math_deepsets remain the production patch?

Recommendation: **No.**

More precisely:

- withdraw `math_deepsets_round1` as the **production default** patch
- do **not** keep the full all-math patch on the production stack
- only keep `math_deepsets_round1` as a **selective opt-in tool**, with `DS-R1/aime25` as the current best recovery point

### 7.2 Why not keep the old promoted full patch?

Because the combined evidence is no longer supportive:

1. `Submission #92` already failed online
2. replaying the actual exported patch behavior no longer gives a guarded full-system proxy improvement
3. the broad patch touches too many blind math caches
4. the selective search finds a smaller and cleaner positive point

## 8. Files Changed

Added:

- `docs/BESTOFN_SCORE_RECOVERY_20260408.md`
- `scripts/run_bestofn_score_recovery_20260408.py`
- `submission/BestofN/extreme12/patches/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa__no_math_patch.json`
- `submission/BestofN/extreme12/patches/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa__math_patch_ds_aime25_only.json`
- `submission/BestofN/extreme12/patches/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa__math_patch_ds_qwen_aime25_brumo25.json`

Updated:

- `scripts/patch_bestofn_submission_with_math_deepsets_round1.py`

Generated:

- `result/bestofn_score_recovery_20260408/score_recovery.json`

## 9. How To Re-run

### 9.1 Re-run the recovery analysis

```bash
source .venv/bin/activate
python scripts/run_bestofn_score_recovery_20260408.py
```

### 9.2 Export a selective math patch manually

Example: only patch `DS-R1/aime25`

```bash
source .venv/bin/activate
python scripts/patch_bestofn_submission_with_math_deepsets_round1.py \
  --base-submission submission/BestofN/extreme12/patches/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa_patch.json \
  --patch-cache-keys DS-R1/aime25 \
  --method-name extreme12_baseline12_pointwise_best_only_ref030_t1024__code_v2_lcb__science_hybrid_round3_gpqa__math_patch_ds_aime25_only \
  --out submission/BestofN/extreme12/patches/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa__math_patch_ds_aime25_only.json
```

### 9.3 Export the aggressive candidate manually

```bash
source .venv/bin/activate
python scripts/patch_bestofn_submission_with_math_deepsets_round1.py \
  --base-submission submission/BestofN/extreme12/patches/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa_patch.json \
  --patch-cache-keys DS-R1/aime25,DS-R1/brumo25,Qwen3-4B/aime25,Qwen3-4B/brumo25 \
  --method-name extreme12_baseline12_pointwise_best_only_ref030_t1024__code_v2_lcb__science_hybrid_round3_gpqa__math_patch_ds_qwen_aime25_brumo25 \
  --out submission/BestofN/extreme12/patches/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa__math_patch_ds_qwen_aime25_brumo25.json
```

## 10. Bottom Line

- The full `math_deepsets_round1` patch should be treated as a **reverted default**.
- The safest score-recovery move is `no_math_patch`.
- The best selective improvement point found here is `DS-R1/aime25` only.
- The only reason to go beyond that is a deliberate blind score-shot, not a production default.
