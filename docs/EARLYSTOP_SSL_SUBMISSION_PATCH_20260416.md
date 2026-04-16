# EarlyStop SSL Round1: Live Submission Comparison and Patch Plan

**Date:** 2026-04-16  
**Context:** follow-up to `docs/EARLYSTOP_SSL_SEMISUP_ROUND1_20260416.md` after live leaderboard submission

## 1. What was actually trained?

The SSL / semisupervised round1 work did **not** train ten independent slot models.

It trained **four anchor-specific heads / basis views** at the canonical EarlyStop anchors:

- `10%`
- `40%`
- `70%`
- `100%`

The public EarlyStop payload still emits scores for the official ten slots:

- `10 / 20 / 30` → reuse the `10%` anchor model
- `40 / 50 / 60` → reuse the `40%` anchor model
- `70 / 80 / 90` → reuse the `70%` anchor model
- `100` → use the `100%` anchor model

This is the exact mapping used in the SSL study code:

- `EARLY_STOP_POSITIONS = [0.1, 0.2, ..., 1.0]`
- `ANCHOR_POSITIONS = [0.1, 0.4, 0.7, 1.0]`
- `OFFICIAL_SLOT_TO_ANCHOR = {0.1:0.1, 0.2:0.1, 0.3:0.1, 0.4:0.4, 0.5:0.4, 0.6:0.4, 0.7:0.7, 0.8:0.7, 0.9:0.7, 1.0:1.0}`

So, when asked “which slots were trained”, the precise answer is:

> **trained anchors = 4 (`10/40/70/100`), exported slots = 10**

## 2. Which domains were newly trained in the SSL export?

For the exported SSL round1 submission family:

- **science**: newly trained and actually used in the live patch
- **math**: trained in offline SSL sweeps, but not justified enough to replace the best live line
- **coding**: not retrained as a promoted SSL line; only cheap confirmation was run offline, and coding remained near-random

This means the submitted file

- `submission/EarlyStop/earlystop_ssl_round1_science_tokenpair16_patch.json`

was never intended as a full “replace everything” candidate. Its point was:

> keep the old base for non-science pieces, and test whether the science SSL override is useful enough to promote

## 3. Live leaderboard outcome: #289 vs #279

The relevant live submissions were:

- **#279**: `es_svd_ms_rr_r1__coding_rotation_adapter_aggressive_tokenonly_late70100`
  - status: `CURRENT BEST`
  - primary score: `4.4375` (lower is better)
- **#289**: `earlystop_ssl_round1_science_tokenpair16_patch`
  - status: `Not best`
  - primary score: `4.6875`

This is an important result:

> The SSL science patch produced **interesting local gains**, but did **not** beat the current live best on the leaderboard’s primary ranking.

### 3.1 Why did the SSL science patch fail to become live best?

The short answer is:

1. it improved some **AUROC** values, especially on `gpqa`,
2. but it lost enough on **AUC of SelAcc / overall rank mix** that the live primary score got worse,
3. and the live best `#279` already includes a stronger coding-side patch than the conservative base used by the SSL science export.

So the live result is **not** evidence that the science SSL idea is useless. It is evidence that:

- the science improvement is **too narrow** to outweigh other parts of the live rank objective when packaged as `#289`,
- and the right next step is **hybrid patching**, not blindly promoting the whole SSL file.

## 4. Cache-level comparison

The cache-level live comparison is recorded in:

- `results/tables/earlystop_ssl/submission_live_compare_289_vs_279.csv`

The most important deltas are:

### AUROC gains from `#289`

- `DS-R1/gpqa`: `+0.0066`
- `Qwen3-4B/gpqa`: `+0.0169`

These are the strongest reasons to keep investigating the science SSL line.

### SelAcc losses from `#289`

- `DS-R1/gpqa`: `-0.0061`
- `Qwen3-4B/gpqa`: `-0.0014`
- `DS-R1/lcb_v5`: `-0.0152`

This helps explain why the live leaderboard did not improve overall despite the `gpqa` AUROC lift.

### Interpretation

The cleanest reading is:

- **science SSL improves ranking quality on `gpqa`**
- but **the specific exported score surface is not yet calibrated enough** to win on the leaderboard’s combined objective

That is a meaningful result, not a contradiction.

## 5. Hybrid patch strategy

Because `#279` is still the current best live file, the correct patch direction is:

> start from `#279`, then selectively splice in only the `gpqa` slices that improved under SSL

I generated three hybrid candidates locally:

### Candidate A: both `gpqa` caches replaced

- JSON: `submission/EarlyStop/es_svd_ms_rr_r1__coding_rotation_adapter_aggressive_tokenonly_late70100__ssl_gpqa_both_patch.json`
- manifest: `submission/EarlyStop/es_svd_ms_rr_r1__coding_rotation_adapter_aggressive_tokenonly_late70100__ssl_gpqa_both_patch.manifest.json`

Overrides:

- `DS-R1/gpqa`
- `Qwen3-4B/gpqa`

### Candidate B: only `Qwen3-4B/gpqa` replaced

- JSON: `submission/EarlyStop/es_svd_ms_rr_r1__coding_rotation_adapter_aggressive_tokenonly_late70100__ssl_gpqa_qwen_patch.json`
- manifest: `submission/EarlyStop/es_svd_ms_rr_r1__coding_rotation_adapter_aggressive_tokenonly_late70100__ssl_gpqa_qwen_patch.manifest.json`

Override:

- `Qwen3-4B/gpqa`

### Candidate C: only `DS-R1/gpqa` replaced

- JSON: `submission/EarlyStop/es_svd_ms_rr_r1__coding_rotation_adapter_aggressive_tokenonly_late70100__ssl_gpqa_ds_patch.json`
- manifest: `submission/EarlyStop/es_svd_ms_rr_r1__coding_rotation_adapter_aggressive_tokenonly_late70100__ssl_gpqa_ds_patch.manifest.json`

Override:

- `DS-R1/gpqa`

## 6. Which hybrid is most worth submitting first?

### Recommended order

1. **`...__ssl_gpqa_qwen_patch.json`**
2. **`...__ssl_gpqa_both_patch.json`**
3. **`...__ssl_gpqa_ds_patch.json`**

### Why `Qwen gpqa` first?

From the live numbers:

- `Qwen3-4B/gpqa` has the **largest AUROC gain** (`+0.0169`)
- while its SelAcc loss is relatively small (`-0.0014`)

That is the most favorable tradeoff among the science overrides we observed.

By contrast:

- `DS-R1/gpqa` improves AUROC too, but pays a larger SelAcc penalty (`-0.0061`)

So the most defensible next live test is:

> keep the full `#279` backbone intact, and patch in **only `Qwen3-4B/gpqa`** from the SSL submission

## 7. What this changes about the research conclusion

The live result sharpens the round1 takeaway:

1. **Science SSL is real but narrow**  
   The `gpqa` AUROC lift survives contact with live evaluation.

2. **The gain is not yet promotion-safe as a full submission**  
   Packaging the science SSL patch as `#289` worsened the leaderboard primary score.

3. **The right operational move is cache-level hybridization**  
   This is now a patching / calibration problem, not a “replace the whole method” problem.

## 8. Bottom line

After the live submission result, the best current operational stance is:

- keep `#279` as the live baseline,
- treat the SSL science line as a **selective `gpqa` override source**,
- test the `Qwen gpqa`-only hybrid first,
- and avoid promoting the full SSL science patch unchanged.
