# Science Hybrid Round 3 — Plan

Date: `2026-04-06`

## Very Short State Summary

After checking the requested docs, the current implementations, and the existing
submission/patch scripts, the repo state is:

1. `science_baseline_v1` is still the frozen science baseline.
2. GPQA pairwise round-2 is still `NO-PROMOTE`.
3. `code_v2` is already the promoted coding default.
4. This round-3 should prioritise a narrow hybrid rule, not a new feature family and not a deeper architecture.

## Scope

This round is intentionally narrow:

- keep `science_baseline_v1` as the baseline science score
- keep the existing GPQA pairwise scorer / saved model family
- do post-hoc hybrid decision rules only
- do not introduce DeepSets / Set Transformer / graph-heavy expansions
- do not change frozen generic/math definitions
- do not change `code_v2` weights or coding defaults

## Round-3 Families

The search space is limited to small, interpretable families:

- **Family A — margin-triggered fallback**
  - baseline-first
  - if baseline top1-top2 gap is large, keep baseline top1
  - otherwise rerank a small shortlist with pairwise aggregation

- **Family B — shortlist blend**
  - take baseline top-k
  - blend baseline shortlist rank with pairwise shortlist rank
  - no new model training

- **Family C — hard override**
  - baseline top1 stays default
  - only override when baseline gap is small and pairwise advantage over baseline top1 is large enough

- **Pairwise backend variants**
  - `mean`
  - `softmax_mean`
  - `win_count`
  - `copeland_margin`

## Small Search Grid

The round-3 search stays intentionally small:

- `tau`: a tiny quantile-based grid from baseline top1-top2 gaps
- `k`: `{2,3,4,5}`
- `alpha`: `{0.25, 0.50, 0.75}`
- `m`: `{0.00, 0.02, 0.04, 0.06}`
- `temperature`: only a very small backend grid for softened pairwise aggregation

For `tau`-based gating, the gap should be measured on the frozen baseline's
underlying raw `recency_conf_mean` signal, not on rank-only baseline scores,
because rank gaps become degenerate when group size is fixed.

## Sanity Checks

Two cheap sanity checks are required before any promotion decision:

1. explain why round-2 `margin / dominance / stronger_regularization` were nearly identical
   - feature correlation vs `recency_conf_mean_r`
   - final ranking agreement vs round-1 mean pairwise ordering

2. inspect baseline/pairwise disagreement
   - top1 agreement rate
   - disagreement winner correctness
   - whether disagreement concentrates in small baseline-gap problems

## Evaluation

### GPQA single-domain

Report at least:

- `AUROC`
- `Hit@1`
- `Pairwise`
- `SelAcc@10`

Against at least:

- `science_baseline_v1`
- `tournament-copeland`
- `gpqa_pairwise_round1`
- all round-3 hybrid candidates

### Comprehensive proxy

The round-3 decision is not GPQA-only. The system view uses:

- frozen DS-R1 base slices from existing `extreme12` analysis
- current promoted `code_v2` coding slice
- mutable science slice replaced by candidate `science_hybrid_round3`

If no full automated leaderboard reproduction tool exists, use a transparent
proxy and state it explicitly in the results note.

## Decision Gates

### Science Gate

- should not lose the key top-slot science metrics against `science_baseline_v1`
- should be closer to usable than pairwise round1/round2

### Comprehensive Gate

- patching the candidate into the current system should improve the overall proxy
- tiny noisy system gains do not justify promotion if science top-slot clearly regresses

## Deliverables

- `docs/SCIENCE_HYBRID_ROUND3_PLAN_20260406.md`
- `docs/SCIENCE_HYBRID_ROUND3_RESULTS_20260406.md`
- `nad/core/selectors/science_hybrid_impl.py`
- `plugins/science_hybrid_selector.py`
- `scripts/run_science_hybrid_round3.py`
- minimal patch scripts only where needed
