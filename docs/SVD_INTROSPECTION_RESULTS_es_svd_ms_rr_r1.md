# SVD Introspection Results — es_svd_ms_rr_r1

> Domains: math, science | Anchors: 10%, 40%, 70%, 100%
> Routes: 20 | Wrong-top1 cases: 0

---

## Q1: What does the model look at?

Family contribution strength (aggregate |weight| across all anchors/domains):

| Family | Aggregate |w| |
|--------|------------|
| trajectory | 414.3357 |
| uncertainty | 93.7111 |
| self_cert_logprob | 25.3969 |
| confidence | 5.7753 |
| availability_meta | 1.8809 |

## Q2: Top driving features?

Top feature weights per anchor are available in `effective_weights.csv` and `component_table.csv`.

Primary family: **trajectory** accounts for the largest aggregate weight.

## Q3: Why does top1 beat top2?

The dominant family advantage is consistently the primary separator. See `problem_top1_vs_top2.jsonl` for per-problem deltas.

## Q4: Common failure modes?

Cases where model selected a wrong run over the best correct run:

**Top over-boosted families:**

| Family | Top-Driver Count | Mean Δ |
|--------|-----------------|--------|

**Top over-boosted features:**

| Feature | Top-Driver Count | Mean Δ |
|---------|-----------------|--------|

## Q5: Stable vs unstable explanations?

Cross-anchor sign consistency (adjacent anchors, same domain): **95.8%** of feature weights maintain sign direction.

Features with sign flips across anchors (sample):

| Domain | Feature | From Anchor | To Anchor |
|--------|---------|------------|-----------|
| math | has_rows_bank | 10% | 40% |
| math | has_tok_conf | 10% | 40% |
| math | has_tok_gini | 10% | 40% |
| math | has_tok_logprob | 10% | 40% |
| math | has_tok_neg_entropy | 10% | 40% |
| math | has_tok_selfcert | 10% | 40% |
| math | tok_gini_prefix | 10% | 40% |
| math | tok_gini_slope | 10% | 40% |

## Q6: Domain differences?

Cross-domain sign consistency: **46.0%** of shared features maintain the same sign direction across domains.

Full per-feature comparison is in `stability_report.csv` (filter `comparison_type == cross_domain`).

---
*Generated automatically by `scripts/export_svd_introspection.py` for `es_svd_ms_rr_r1`.*

