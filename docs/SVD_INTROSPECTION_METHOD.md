# SVD Introspection Method

This note defines the exact reconstruction used by `nad.explain.svd_introspection` and the artifacts exported by `scripts/export_svd_introspection.py`.

## 1. Back-projected decision function

For a route with representation vector `x_rep`, SVD basis `V`, singular values `s`, and linear classifier weights `alpha`:

```text
x_std = (x_rep - mu) / sigma
z     = x_std @ V.T
z'    = z / s                  # whiten=True only
score = b + alpha @ z'
```

Define the effective latent weights:

```text
alpha_eff[k] = alpha[k] / s[k]   if whiten=True
alpha_eff[k] = alpha[k]          if whiten=False
```

Then the exact original-space linear form is:

```text
w_orig = (V.T @ alpha_eff) / sigma
b_orig = b - dot(mu / sigma, V.T @ alpha_eff)
score  = b_orig + x_rep @ w_orig
```

`recover_effective_weights()` exposes this decomposition directly. `w_orig` is identical to the effective per-representation weight returned by the legacy internal helper.

## 2. Representation channels

Routes operate on representation features rather than raw cache features alone. A single base feature may appear as:

- `feature::raw`: absolute feature value at an anchor
- `feature::rank`: within-problem rank of that feature across runs

`build_problem_rank_matrix()` must always receive the full `n_runs x n_features` matrix for one problem. Ranking a single run in isolation is invalid.

## 3. Feature families

The exporter groups features with `feature_family()` into a small taxonomy used throughout the reports:

- `trajectory`
- `uncertainty`
- `confidence`
- `self_cert_logprob`
- `availability_meta`

Family-level rows aggregate signed and absolute effective weights and also aggregate run-time contributions.

## 4. Three explanation layers

The introspection system exposes three aligned views of the same decision:

- **Feature layer:** `x_rep[j] * w_orig[j]`
- **Family layer:** sum of feature contributions within a family
- **Component layer:** `z_k * alpha_eff[k]`

For component inspection, the sign of each SVD component is aligned so that positive direction always means “more likely correct”:

```text
sign_k            = +1 if alpha_eff[k] >= 0 else -1
aligned_V[k]      = V[k] * sign_k
aligned_alpha[k]  = abs(alpha_eff[k])
```

The exporter labels each component with its dominant family and purity score.

## 5. Problem-level explanation

`explain_problem_decision()` compares the selected top run against the runner-up by:

- scoring all runs with the exact route matrix
- explaining the top runs individually
- subtracting feature, family, and component contributions

This yields an exact decomposition of the top1-vs-top2 margin.

## 6. Sanity criteria

The implementation enforces exactness numerically:

1. Per run: `abs(sum(feature_contrib) + b_orig - score) < 1e-6`
2. Per run: `abs(sum(component_contrib) + b_orig - score) < 1e-6`
3. Per problem: `abs(sum(delta_feature_contrib) - (score_top1 - score_top2)) < 1e-6`
4. Rank transforms are computed group-wise, never per-run
5. Baseline routes use a trivial one-feature decomposition

In smoke verification for `es_svd_ms_rr_r1`, the observed maximum absolute reconstruction error is on the order of `1e-13`.

## 7. Exported artifacts

Per method, the exporter writes:

- `route_inventory.json`
- `effective_weights.csv`
- `component_table.csv`
- `family_summary.csv`
- `failure_modes.csv`
- `stability_report.csv`
- `problem_top1_vs_top2.jsonl`
- `run_contributions/...`
- `sanity_checks.json`

It also generates result summaries under `docs/` for paper-facing interpretation.
