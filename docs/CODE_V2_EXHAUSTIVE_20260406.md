# Code v2 Candidate — Exhaustive Pass Result

Date: `2026-04-06`
Script: `scripts/run_code_v2_candidate.py`
Results Dir: `result/code_v2_candidate_20260406_exhaustive`

---

## Decision: PROMOTE

The full exhaustive `code_v2_candidate` pass was completed over the configured
11-candidate search set:

- 1 default candidate
- 10 searched weight combinations

The winning candidate passed the agreed coding gate and is now the promoted
`code_v2` default.

Selected candidate:

- `pfx0p42_gap0p06_tail0p08_rec0p28_inst0p16`

Promoted weights:

- `prefix_best_window_quality = 0.42`
- `head_tail_gap = 0.06`
- `tail_variance = 0.08`
- `post_reflection_recovery = 0.28`
- `last_block_instability = 0.16`

---

## Gate Summary

Gate definition:

- **SelAcc@10** must exceed `code_baseline_v1`
- **Pairwise** must be at least `50%`
- **Hit@1** must be at least baseline minus `1pp`
- blind DS/Qwen checks are report-only

Winning metrics:

| Selector | AUROC | Hit@1 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| `code_baseline_v2_candidate` | 50.29% | 61.68% | 50.99% | **62.11%** |
| `code_baseline_v1` | 50.27% | 59.28% | 51.27% | 61.74% |

Gate outcome:

- **SelAcc@10**: `62.11% > 61.74%` ✓
- **Pairwise**: `50.99% ≥ 50.00%` ✓
- **Hit@1**: `61.68% ≥ 58.28%` ✓

---

## Search Outcome

Passing candidates from the exhaustive run:

| Candidate | SelAcc@10 | Hit@1 | Pairwise |
|---|---:|---:|---:|
| `pfx0p46_gap0p00_tail0p08_rec0p22_inst0p24` | 61.93% | 60.48% | 50.99% |
| `pfx0p42_gap0p06_tail0p08_rec0p28_inst0p16` | **62.11%** | 61.68% | 50.99% |
| `pfx0p40_gap0p06_tail0p08_rec0p22_inst0p24` | 61.83% | 61.68% | 51.02% |

The exhaustive winner improves on the first-passing candidate from the earlier
short run, so the promotion uses the true best passing configuration rather than
the first configuration that cleared gate.

---

## Feature Read

LOO continues to support the narrow coding story:

- `prefix_best_window_quality` remains the dominant driver
- `post_reflection_recovery` is still important
- `tail_variance` helps more at `0.08` than at the older `0.16`
- `head_tail_gap` stays useful as a small auxiliary term
- `last_block_instability` remains part of the promoted score, but the best
  setting keeps it at `0.16` rather than increasing it

---

## Blind Checks

Report-only overlap checks for the promoted winner:

- `DS-R1/lcb_v5`: top-1 agreement vs `code_baseline_v1` = `51.50%`
- `Qwen3-4B/lcb_v5`: top-1 agreement vs `code_baseline_v1` = `37.72%`

These do not affect promotion, but confirm the promoted selector is materially
different from both the old coding baseline and the generic comparators.
