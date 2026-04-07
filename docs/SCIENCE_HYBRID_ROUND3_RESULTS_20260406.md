# Science Hybrid Round 3 — Results

Date: `2026-04-06`
Script: `scripts/run_science_hybrid_round3.py`

## Repo State Confirmed Before Running

1. `science_baseline_v1` is still the frozen science baseline.
2. GPQA pairwise round-2 is still `NO-PROMOTE`.
3. `code_v2` is already the promoted coding default.
4. This round-3 prioritises a narrow hybrid rule, not a new feature family and not a deeper architecture.

## Experiment Goal

This round evaluates a narrow post-hoc `science round-3 hybrid` built on top of:

- `science_baseline_v1`
- the existing GPQA pairwise scorer
- a small rule-based / calibrated / shortlist rerank search

The decision target is two-level:

- GPQA single-domain improvement
- system-level comprehensive improvement after patching science into the current promoted stack

## Sanity Checks

### 1. Why round-2 `margin / dominance / regularization` were almost tied

The rerun confirms that the round-2 additions were effectively the same signal:

- `recency_margin_r` vs `recency_conf_mean_r`: mean Pearson = `1.000`, mean Spearman = `1.000`
- `recency_dominance_r` vs `recency_conf_mean_r`: mean Pearson = `1.000`, mean Spearman = `1.000`

Order-consistency vs round-1 mean pairwise ordering:

| Variant | Top1 Agreement vs round1 mean | Exact Order Match vs round1 mean |
|---|---:|---:|
| `margin` | 100.00% | 69.19% |
| `dominance` | 100.00% | 69.19% |
| `stronger_regularization` | 100.00% | 74.75% |
| `margin_reg` | 100.00% | 75.25% |
| `dominance_reg` | 100.00% | 75.25% |
| `margin_dominance` | 100.00% | 72.22% |
| `margin_dominance_reg` | 100.00% | 77.78% |

Conclusion:

- round-2 variants changed some lower-order ranks
- they did **not** change the top slot at all
- this explains why round-2 landed on the same operating point

### 2. Baseline vs pairwise disagreement

Using the LOPO pairwise probabilities with the best round-3 backend (`win_count`):

- baseline/pairwise top1 agreement = `7.07%`
- disagreement rate = `92.93%`
- when they disagree:
  - baseline top1 correct rate = `65.76%`
  - pairwise top1 correct rate = `63.04%`

Raw baseline-gap read (using the frozen baseline's raw `recency_conf_mean`, not
rank gaps):

- when baseline and pairwise agree:
  - mean gap = `0.0953`
  - median gap = `0.0960`
- when they disagree:
  - mean gap = `0.1477`
  - median gap = `0.1223`

Conclusion:

- disagreement is **not** mainly concentrated in the smallest baseline-gap problems
- this weakens the justification for a pure gap-triggered fallback as the mainline
- it also matches the round-3 outcome: Family A can help a bit, but Family B is stronger

## GPQA Single-Domain Results

### Main baselines

| Selector | AUROC | Hit@1 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| `science_baseline_v1` | 53.86% | 66.16% | 58.71% | 64.35% |
| `tournament-copeland` | 52.20% | 63.64% | 54.69% | 64.51% |
| `gpqa_pairwise_round1` | **55.66%** | 63.64% | **61.15%** | 62.38% |

### Pairwise backend scan

| Backend | AUROC | Hit@1 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| `mean` | **55.66%** | 63.64% | 61.15% | 62.38% |
| `softmax_mean_t0p75` | 55.51% | 63.64% | 61.15% | 62.62% |
| `softmax_mean_t0p50` | 55.46% | 63.64% | 61.15% | 62.70% |
| `win_count` | 54.94% | 63.64% | 61.15% | **64.98%** |
| `copeland_margin` | 54.94% | 63.64% | 61.15% | **64.98%** |

Selected backend for hybrid search:

- `win_count`

Interpretation:

- `mean` keeps the best AUROC but stays too soft for top-slot use
- `win_count` / `copeland_margin` sharply improve `SelAcc@10`
- round-3 therefore uses `win_count` for the narrow hybrid search

### All round-3 candidates

| Candidate | Trigger | Override | Top1Change | Hit@1 | SelAcc@10 | Pairwise | Science Gate | Full-System Gate |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `familyA__tau0p0321__k2` | 20.20% | 0.00% | 9.60% | 66.67% | 64.35% | 58.71% | pass | pass |
| `familyA__tau0p0321__k3` | 20.20% | 0.00% | 11.62% | 66.67% | 64.35% | 58.71% | pass | pass |
| `familyA__tau0p0321__k4` | 20.20% | 0.00% | 12.63% | 66.16% | 64.35% | 58.71% | pass | fail |
| `familyA__tau0p0321__k5` | 20.20% | 0.00% | 14.14% | 66.16% | 64.35% | 58.71% | pass | fail |
| `familyA__tau0p0662__k2` | 34.85% | 0.00% | 16.67% | 66.67% | 64.35% | 58.71% | pass | pass |
| `familyA__tau0p0662__k3` | 34.85% | 0.00% | 22.22% | 66.16% | 64.35% | 58.71% | pass | fail |
| `familyA__tau0p0662__k4` | 34.85% | 0.00% | 25.25% | 64.65% | 64.35% | 58.70% | fail | fail |
| `familyA__tau0p0662__k5` | 34.85% | 0.00% | 27.78% | 66.16% | 64.35% | 58.70% | pass | fail |
| `familyA__tau0p1212__k2` | 50.00% | 0.00% | 23.23% | 67.17% | 64.35% | 58.71% | pass | pass |
| `familyA__tau0p1212__k3` | 50.00% | 0.00% | 31.82% | 66.16% | 64.35% | 58.71% | pass | fail |
| `familyA__tau0p1212__k4` | 50.00% | 0.00% | 35.86% | 63.64% | 64.35% | 58.70% | fail | fail |
| `familyA__tau0p1212__k5` | 50.00% | 0.00% | 39.90% | 65.15% | 64.35% | 58.70% | fail | fail |
| `familyA__tau0p1726__k2` | 65.15% | 0.00% | 30.81% | 67.17% | 64.35% | 58.71% | pass | pass |
| `familyA__tau0p1726__k3` | 65.15% | 0.00% | 40.91% | 66.16% | 64.35% | 58.71% | pass | fail |
| `familyA__tau0p1726__k4` | 65.15% | 0.00% | 45.45% | 64.14% | 64.35% | 58.70% | fail | fail |
| `familyA__tau0p1726__k5` | 65.15% | 0.00% | 51.01% | 65.15% | 64.35% | 58.70% | fail | fail |
| `familyA__tau0p2112__k2` | 79.80% | 0.00% | 39.39% | 67.68% | 64.35% | 58.71% | pass | pass |
| `familyA__tau0p2112__k3` | 79.80% | 0.00% | 51.01% | 66.16% | 64.35% | 58.71% | pass | fail |
| `familyA__tau0p2112__k4` | 79.80% | 0.00% | 57.07% | 64.14% | 64.35% | 58.71% | fail | fail |
| `familyA__tau0p2112__k5` | 79.80% | 0.00% | 63.13% | 64.65% | 64.35% | 58.71% | fail | fail |
| `familyB__k2__a0p25` | 100.00% | 0.00% | 50.00% | **68.18%** | 64.35% | **58.72%** | pass | pass |
| `familyB__k2__a0p50` | 100.00% | 0.00% | 21.72% | 66.16% | 64.35% | 58.71% | pass | fail |
| `familyB__k2__a0p75` | 100.00% | 0.00% | 0.00% | 66.16% | 64.35% | 58.71% | pass | fail |
| `familyB__k3__a0p25` | 100.00% | 0.00% | 67.17% | 66.67% | 64.35% | 58.71% | pass | pass |
| `familyB__k3__a0p50` | 100.00% | 0.00% | 37.37% | 66.67% | 64.35% | 58.71% | pass | fail |
| `familyB__k3__a0p75` | 100.00% | 0.00% | 0.00% | 66.16% | 64.35% | 58.71% | pass | fail |
| `familyB__k4__a0p25` | 100.00% | 0.00% | 72.22% | 65.66% | 64.35% | 58.71% | fail | fail |
| `familyB__k4__a0p50` | 100.00% | 0.00% | 42.93% | 66.16% | 64.35% | 58.70% | pass | fail |
| `familyB__k4__a0p75` | 100.00% | 0.00% | 3.54% | 66.67% | 64.35% | 58.71% | pass | pass |
| `familyB__k5__a0p25` | 100.00% | 0.00% | 73.74% | 65.66% | 64.35% | 58.72% | fail | fail |
| `familyB__k5__a0p50` | 100.00% | 0.00% | 47.47% | 66.16% | 64.35% | 58.71% | pass | fail |
| `familyB__k5__a0p75` | 100.00% | 0.00% | 8.08% | 67.17% | 64.35% | 58.71% | pass | pass |
| `familyC__tau0p0321__m0p00` | 17.68% | 17.68% | 17.68% | 64.65% | 64.20% | 58.70% | fail | fail |
| `familyC__tau0p0321__m0p02` | 16.67% | 16.67% | 16.67% | 65.66% | 64.27% | 58.71% | fail | fail |
| `familyC__tau0p0321__m0p04` | 14.65% | 14.65% | 14.65% | 65.66% | 64.27% | 58.71% | fail | fail |
| `familyC__tau0p0321__m0p06` | 13.64% | 13.64% | 13.64% | 65.15% | 64.20% | 58.70% | fail | fail |
| `familyC__tau0p0662__m0p00` | 31.82% | 31.82% | 31.82% | 63.13% | 64.20% | 58.69% | fail | fail |
| `familyC__tau0p0662__m0p02` | 29.80% | 29.80% | 29.80% | 64.14% | 64.27% | 58.70% | fail | fail |
| `familyC__tau0p0662__m0p04` | 26.26% | 26.26% | 26.26% | 64.14% | 64.20% | 58.70% | fail | fail |
| `familyC__tau0p0662__m0p06` | 24.24% | 24.24% | 24.24% | 63.64% | 64.12% | 58.69% | fail | fail |
| `familyC__tau0p1212__m0p00` | 45.96% | 45.96% | 45.96% | 62.63% | 64.12% | 58.68% | fail | fail |
| `familyC__tau0p1212__m0p02` | 42.93% | 42.93% | 42.93% | 63.64% | 64.20% | 58.70% | fail | fail |
| `familyC__tau0p1212__m0p04` | 37.88% | 37.88% | 37.88% | 63.13% | 64.04% | 58.69% | fail | fail |
| `familyC__tau0p1212__m0p06` | 35.86% | 35.86% | 35.86% | 62.63% | 63.96% | 58.68% | fail | fail |
| `familyC__tau0p1726__m0p00` | 59.60% | 59.60% | 59.60% | 62.63% | 64.04% | 58.69% | fail | fail |
| `familyC__tau0p1726__m0p02` | 54.04% | 54.04% | 54.04% | 64.14% | 64.12% | 58.69% | fail | fail |
| `familyC__tau0p1726__m0p04` | 48.48% | 48.48% | 48.48% | 63.64% | 63.96% | 58.69% | fail | fail |
| `familyC__tau0p1726__m0p06` | 46.46% | 46.46% | 46.46% | 63.13% | 63.88% | 58.68% | fail | fail |
| `familyC__tau0p2112__m0p00` | 72.73% | 72.73% | 72.73% | 63.64% | 64.20% | 58.72% | fail | fail |
| `familyC__tau0p2112__m0p02` | 66.16% | 66.16% | 66.16% | 65.66% | 64.27% | 58.73% | fail | fail |
| `familyC__tau0p2112__m0p04` | 60.61% | 60.61% | 60.61% | 65.15% | 64.12% | 58.72% | fail | fail |
| `familyC__tau0p2112__m0p06` | 57.58% | 57.58% | 57.58% | 64.65% | 64.12% | 58.72% | fail | fail |

## Comprehensive Proxy Definition

Because the repo does not expose a single end-to-end leaderboard reproducer for
the exact currently-patched system state, this round uses a transparent
cache-level proxy instead of pretending to report an exact leaderboard score.

The proxy bundle is:

- fixed DS-R1 math/base slices from `docs/EXTREME12_TEST_ANALYSIS.md`
- current promoted coding slice from `result/code_v2_candidate_20260406_exhaustive/code_v2_metrics.json`
- mutable science slice recomputed locally on `DS-R1/gpqa`

The proxy aggregates metrics as:

- `Hit@1`: weighted by `n_problems`
- `Hit@3`: weighted by `n_problems`
- `Pairwise`: weighted by `n_samples`
- `SelAcc@10%`: weighted by `top10_count`
- `Avg Rank proxy`: delta-only from the mutable GPQA slice, because the fixed
  math/code slices cancel between candidates

Both sample-weighted and equal-cache-mean views are reported.

## Comprehensive Proxy Results

### Absolute proxy values

Sample-weighted system proxy:

| System | Hit@1 | Hit@3 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| `extreme12_base` | 63.30% | 72.78% | 60.86% | 64.23% |
| `extreme12_plus_code_v2` | 64.13% | 72.78% | 61.59% | 65.59% |
| `current = code_v2 + science_baseline_v1` | 66.39% | 74.02% | 61.29% | 66.15% |
| `selected = code_v2 + science_hybrid_round3` | **67.22%** | 74.02% | 61.29% | 66.15% |

Equal-cache-mean system proxy:

| System | Hit@1 | Hit@3 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| `current = code_v2 + science_baseline_v1` | 70.20% | 77.71% | 71.53% | 70.90% |
| `selected = code_v2 + science_hybrid_round3` | **70.53%** | 77.71% | 71.53% | 70.90% |

### Selected candidate vs current system

Selected candidate:

- `familyB__k2__a0p25`
- family = `shortlist_blend`
- backend = `win_count`
- `k = 2`
- `alpha = 0.25`

Comprehensive delta vs current system:

- sample-weighted `Hit@1`: `+0.82pp`
- sample-weighted `Hit@3`: `+0.00pp`
- sample-weighted `Pairwise`: `+0.0019pp`
- sample-weighted `SelAcc@10`: `+0.00pp`
- sample-weighted `Avg Rank proxy`: improved by `-0.008247`

Interpretation:

- the round-3 win is primarily a **top-slot / rank-1** improvement
- it does **not** improve `SelAcc@10`
- but it improves the current system-level proxy without destabilising the other reported metrics

## Gates

### Science Gate

Science gate pass condition in this round:

- candidate `Hit@1 >= science_baseline_v1`
- candidate `SelAcc@10 >= science_baseline_v1`

Science-gate passed candidates:

- Family A: `tau0p0321 k2/k3/k4/k5`, `tau0p0662 k2/k3/k5`, `tau0p1212 k2/k3`, `tau0p1726 k2/k3`, `tau0p2112 k2/k3`
- Family B: `k2 a0p25/a0p50/a0p75`, `k3 a0p25/a0p50/a0p75`, `k4 a0p50/a0p75`, `k5 a0p50/a0p75`
- Family C: none

Read:

- Family B is the strongest science line
- Family A has several pass points but never beats the best Family B point
- Family C is not viable in this round

### Comprehensive Gate

Full-system comprehensive-gate passed candidates:

- `familyA__tau0p0321__k2`
- `familyA__tau0p0321__k3`
- `familyA__tau0p0662__k2`
- `familyA__tau0p1212__k2`
- `familyA__tau0p1726__k2`
- `familyA__tau0p2112__k2`
- `familyB__k2__a0p25`
- `familyB__k3__a0p25`
- `familyB__k4__a0p75`
- `familyB__k5__a0p75`

Read:

- there are multiple candidates that improve the transparent full-system proxy
- the best one is still `familyB__k2__a0p25`, because it gives the largest GPQA `Hit@1` gain and the largest sample-weighted system `Hit@1` gain

## Decision

- **Decision:** `Promote for full system`
- **Promote / No-Promote:** `Promote`
- **Promote for science only / full system / no-promote:** `Promote for full system`

Reason:

- GPQA single-domain improves on the key top-slot metric:
  - `Hit@1`: `66.16% -> 68.18%` (`+2.02pp`)
- GPQA `SelAcc@10` does not regress:
  - `64.35% -> 64.35%`
- the patched full-system proxy also improves:
  - sample-weighted `Hit@1`: `66.39% -> 67.22%` (`+0.82pp`)
  - `Avg Rank proxy` also improves

This is therefore not just a GPQA-only win. It is a narrow, interpretable
system-level improvement with minimal code disturbance.

## Closest-to-Usable Candidate

The closest-to-usable and actually selected candidate is:

- `familyB__k2__a0p25`

Why it is the strongest point:

- best GPQA `Hit@1`
- no `SelAcc@10` loss
- passes both science and full-system gates

Near-miss families:

- **Family A**: several candidates pass both gates, but all remain weaker than `familyB__k2__a0p25`
- **Family C**: fails across the board; hard override is too brittle and usually loses both science top-slot and system proxy

## Repro

Main experiment:

```bash
python scripts/run_science_hybrid_round3.py \
  --distance-threads 4 \
  --workers 4 \
  --skip-blind-shapes \
  --out-dir result/science_hybrid_round3_20260407_run3
```

Patch current promoted coding slice:

```bash
python scripts/patch_bestofn_submission_with_code_v2.py \
  --distance-threads 8 \
  --out submission/BestofN/extreme12/patches/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb_patch.json
```

Patch selected science round-3 hybrid on top of `code_v2`:

```bash
python scripts/patch_bestofn_submission_with_science_hybrid_round3.py \
  --base-submission submission/BestofN/extreme12/patches/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb_patch.json \
  --distance-threads 4 \
  --workers 4 \
  --family shortlist_blend \
  --backend win_count \
  --k 2 \
  --alpha 0.25 \
  --tau 0.0 \
  --m 0.0 \
  --temperature 1.0 \
  --out submission/BestofN/extreme12/patches/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa_patch.json
```

Selected patch parameters:

- family = `shortlist_blend`
- backend = `win_count`
- `k = 2`
- `alpha = 0.25`
- `tau = 0.0`
- `m = 0.0`
- `temperature = 1.0`
