# GPQA Pairwise Round 2 — Results

Date: `2026-04-06`
Cache: `DeepSeek-R1-0528-Qwen3-8B/gpqa/cache_neuron_output_1_act_no_rms_20251126_111853`
Script: `scripts/run_gpqa_pairwise_round2.py`
Results Dir: `result/gpqa_pairwise_round2_20260406_live`

---

## Decision: NO-PROMOTE

The planned round-2 science ablations were run:

- `margin`
- `dominance`
- `stronger_regularization`

The runner also evaluated the natural combinations:

- `margin_reg`
- `dominance_reg`
- `margin_dominance`
- `margin_dominance_reg`

No variant passed the existing GPQA promotion gate, so `science_baseline_v1`
remains the promoted science line and no `gpqa_pairwise_v1.pkl` artefact was
written.

---

## Metrics Summary

| Selector | AUROC | Hit@1 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|---:|
| `science_baseline_v1` | 53.86% | 66.16% | 58.71% | 64.35% |
| `tournament-copeland` | 52.20% | 63.64% | 54.69% | 64.51% |
| `margin` | **55.66%** | 63.64% | **61.15%** | 62.38% |
| `dominance` | **55.66%** | 63.64% | **61.15%** | 62.38% |
| `stronger_regularization` | **55.66%** | 63.64% | **61.15%** | 62.38% |
| `margin_reg` | **55.66%** | 63.64% | **61.15%** | 62.38% |
| `dominance_reg` | **55.66%** | 63.64% | **61.15%** | 62.38% |
| `margin_dominance` | **55.66%** | 63.64% | 61.14% | 62.38% |
| `margin_dominance_reg` | **55.66%** | 63.64% | **61.15%** | 62.38% |

Gate outcome for every round-2 variant:

- **SelAcc@10**: `62.38% ≤ 64.35%` threshold
- **Hit@1**: `63.64% < 65.16%` guardrail

---

## Interpretation

Round-2 did not change the top-slot behavior in any meaningful way.

Observed pattern:

- AUROC and pairwise stay improved versus `science_baseline_v1`
- Hit@1 and SelAcc@10 stay below gate
- `margin`, `dominance`, and `C=0.1` regularization all land on essentially the
  same operating point

Most likely explanation:

- `recency_margin_r` and `recency_dominance_r` are both monotonic transforms of
  the same recency-led ordering signal
- adding them to the Bradley-Terry feature stack does not materially alter the
  induced within-group ranking
- stronger regularization likewise does not solve the top-1 calibration issue

So the failure mode from round-1 still stands: the pairwise model improves
global discrimination, but the mean pairwise aggregation remains too soft for
the decisive GPQA top slot.

---

## Bottom Line

- the requested science round-2 experiments were completed
- all seven evaluated variants failed the current GPQA gate
- `science_baseline_v1` remains unchanged
- the next science iteration should target **top-1 calibration / hybrid
  decision rules**, not more monotonic recency-derived features
