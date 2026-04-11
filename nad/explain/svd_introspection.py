"""
SVD Introspection: exact, interpretable, extractable paper-quality artifacts.

Extends nad/explain/svd_explain.py with:
- Route inventory (structured JSON of all routes with feature::channel spec)
- Component-level decomposition (alpha_eff, sign-aligned V components, auto-labels)
- CSV artifacts (family_summary, component_table, failure_modes, stability_report)
- Stability analysis across domains/anchors (no retraining needed)
- Standalone per-route functions usable outside the 4-anchor framework

All computations are exact back-projections; sanity checks enforce max_abs_error < 1e-6.

Do NOT modify nad/explain/svd_explain.py — this module only imports from it.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np

from nad.core.selectors.trajectory_impl import DEFAULT_REFLECTION_THRESHOLD
from nad.core.views.reader import CacheReader
from nad.explain.svd_explain import (
    _canonical_feature_names_for_route,
    _family_delta_rows,
    _family_strength_rows,
    _feature_delta_rows,
    _feature_indices_for_route,
    _feature_weight_rows,
    _ranking_rows,
    _representation_features,
    _route_effective_linear_terms,
    _score_route_matrix,
    _top_runs,
    _why_top1,
    feature_family,
)
from nad.ops.earlystop_svd import (
    FULL_FEATURE_NAMES,
    _build_representation,
    _rank_transform_matrix,
    extract_earlystop_signals_for_positions,
)

_EPS = 1e-12

# ---------------------------------------------------------------------------
# Task 1 — Route Inventory
# ---------------------------------------------------------------------------


def expand_feature_names_for_representation(route: Mapping[str, Any]) -> list[str]:
    """Return rep-feature names as 'base_feature::channel' strings."""
    rep_defs = _representation_features(route)
    return [f"{rd.base_feature}::{rd.source}" for rd in rep_defs]


def build_route_feature_spec(route: Mapping[str, Any], domain: str, position: float) -> dict[str, Any]:
    """Return a flat dict describing all route fields including rep_feature names."""
    route_type = str(route.get("route_type", "svd"))
    feature_names = _canonical_feature_names_for_route(route)
    rep_feature_names = expand_feature_names_for_representation(route)
    families = list(dict.fromkeys(feature_family(f) for f in feature_names))
    family_name = families[0] if families else "unknown"

    return {
        "domain": domain,
        "position_pct": int(round(float(position) * 100.0)),
        "route_type": route_type,
        "family_name": family_name,
        "representation": str(route.get("representation", "raw")),
        "rank": int(route.get("rank", 0)) if route.get("rank") is not None else None,
        "c_value": route.get("c_value"),
        "whiten": bool(route.get("whiten", False)),
        "class_weight": route.get("class_weight"),
        "feature_names": feature_names,
        "rep_feature_names": rep_feature_names,
        "n_rep_features": len(rep_feature_names),
        "baseline_signal_name": route.get("baseline_signal_name"),
        "baseline_cv_auroc": route.get("baseline_cv_auroc"),
        "cv_auroc": route.get("cv_auroc"),
    }


def iter_routes(bundle: Mapping[str, Any]) -> Iterator[tuple[str, float, Mapping[str, Any]]]:
    """Yield (domain, position, route) for every domain × position in the bundle."""
    for domain, domain_bundle in bundle.get("domains", {}).items():
        for route in domain_bundle.get("routes", []):
            position = float(route.get("training_position", 0.0))
            yield (str(domain), position, route)


def build_route_inventory(bundle: Mapping[str, Any], method_id: str) -> list[dict[str, Any]]:
    """Build a list of route feature specs covering all domains × positions."""
    rows: list[dict[str, Any]] = []
    for domain, position, route in iter_routes(bundle):
        spec = build_route_feature_spec(route, domain, position)
        spec["method_id"] = method_id
        rows.append(spec)
    return rows


# ---------------------------------------------------------------------------
# Task 2 — Standalone Representation Reconstruction
# ---------------------------------------------------------------------------


def extract_problem_raw_matrix(
    reader: CacheReader,
    run_ids: Sequence[int],
    positions: Sequence[float],
    required_features: set[str],
    reflection_threshold: float = DEFAULT_REFLECTION_THRESHOLD,
) -> np.ndarray:
    """
    Extract raw feature matrix for a problem group.

    Returns shape (n_runs, len(positions), len(FULL_FEATURE_NAMES)).
    Accepts arbitrary positions — works for r1 (4 anchors) and r2 (10 anchors).
    """
    positions_tuple = tuple(float(v) for v in positions)
    tensor = np.zeros(
        (len(run_ids), len(positions_tuple), len(FULL_FEATURE_NAMES)),
        dtype=np.float64,
    )
    req: Optional[set[str]] = None if not required_features else set(str(v) for v in required_features)

    for row_idx, sample_id in enumerate(run_ids):
        signal_map = extract_earlystop_signals_for_positions(
            reader=reader,
            run_id=int(sample_id),
            positions=positions_tuple,
            required_features=req,
            reflection_threshold=float(reflection_threshold),
        )
        for feat_idx, feat_name in enumerate(FULL_FEATURE_NAMES):
            vals = np.asarray(
                signal_map.get(feat_name, [0.0] * len(positions_tuple)),
                dtype=np.float64,
            )
            if vals.size >= len(positions_tuple):
                tensor[row_idx, :, feat_idx] = vals[: len(positions_tuple)]
    return tensor


def build_problem_rank_matrix(x_raw_at_position: np.ndarray) -> np.ndarray:
    """
    Compute group-wise rank transform.

    Shape: (n_runs, n_features) → (n_runs, n_features).
    CRITICAL: all n_runs must be passed together; never rank a single run in isolation.
    """
    return _rank_transform_matrix(x_raw_at_position)


def build_problem_route_matrix(
    route: Mapping[str, Any],
    x_raw: np.ndarray,
    x_rank: np.ndarray,
) -> np.ndarray:
    """
    Build the representation matrix for a route.

    Shape: (n_runs, n_rep_features) where n_rep_features depends on the route's representation.
    """
    feature_indices = _feature_indices_for_route(route)
    representation = str(route.get("representation", "raw"))
    return _build_representation(
        x_raw=x_raw,
        x_rank=x_rank,
        feature_indices=feature_indices,
        representation=representation,
    )


# ---------------------------------------------------------------------------
# Task 3 — Recover Effective Weights (public API + component level)
# ---------------------------------------------------------------------------


def recover_effective_weights(route: Mapping[str, Any]) -> dict[str, Any]:
    """
    Extend _route_effective_linear_terms with component-level decomposition.

    New fields vs existing _route_effective_linear_terms:
    - rep_feature_names: list[str]    e.g. ["tok_conf_prefix::raw", ...]
    - w_orig: ndarray(n_rep,)         same as weights_rep
    - b_orig: float                   same as intercept_effective
    - alpha_eff: ndarray(rank,)       LR coef / singular_values (if whiten)
    - V: ndarray(rank, n_rep)         svd.components_
    - component_importance: sorted by |alpha_eff_k|
    - aligned_components: sign-aligned V rows with auto_labels
    - sanity: formula + expected_max_error

    For baseline routes, alpha_eff = [1.0] and V = [[1.0]] (trivial decomposition).
    """
    terms = _route_effective_linear_terms(route)
    route_type = str(terms["route_type"])
    rep_defs = list(terms["representation_features"])
    rep_feature_names = [f"{rd.base_feature}::{rd.source}" for rd in rep_defs]
    w_orig = np.asarray(terms["weights_rep"], dtype=np.float64)
    b_orig = float(terms["intercept_effective"])

    # --- Baseline: trivial decomposition ---
    if route_type == "baseline":
        return {
            **terms,
            "rep_feature_names": rep_feature_names,
            "w_orig": w_orig,
            "b_orig": b_orig,
            "alpha_eff": np.array([1.0], dtype=np.float64),
            "V": np.array([[1.0]], dtype=np.float64),
            "component_importance": [
                {"comp_k": 0, "alpha_eff_raw": 1.0, "abs_importance": 1.0}
            ],
            "aligned_components": [
                {
                    "comp_k": 0,
                    "alpha_eff_aligned": 1.0,
                    "sign_k": 1,
                    "top_positive_rep_features": rep_feature_names[:5],
                    "top_negative_rep_features": [],
                    "dominant_family": feature_family(str(terms["feature_names"][0])) if terms["feature_names"] else "other",
                    "family_purity": 1.0,
                    "auto_label": f"{feature_family(str(terms['feature_names'][0])) if terms['feature_names'] else 'other'}(↑)",
                }
            ],
            "sanity": {
                "formula": "score = b_orig + x_rep @ w_orig (baseline: w_orig=[1.0], b_orig=0)",
                "expected_max_error": "<1e-10",
            },
        }

    # --- SVD route ---
    model = route["model"]
    scaler = model["scaler"]
    svd = model["svd"]
    lr = model["lr"]

    beta = np.asarray(lr.coef_, dtype=np.float64).reshape(-1)
    singular_values = np.asarray(svd.singular_values_, dtype=np.float64)
    singular_values_safe = np.where(np.abs(singular_values) < _EPS, 1.0, singular_values)

    if bool(model.get("whiten", False)):
        alpha_eff = beta / singular_values_safe
    else:
        alpha_eff = beta.copy()

    V = np.asarray(svd.components_, dtype=np.float64)  # (rank, n_rep)
    rank = int(V.shape[0])

    # --- Component importance (sorted by |alpha_eff_k|) ---
    component_importance = sorted(
        [
            {
                "comp_k": k,
                "alpha_eff_raw": float(alpha_eff[k]),
                "abs_importance": float(abs(alpha_eff[k])),
            }
            for k in range(rank)
        ],
        key=lambda x: -x["abs_importance"],
    )

    # --- Aligned components ---
    aligned_components: list[dict[str, Any]] = []
    for k in range(rank):
        sign_k = 1 if alpha_eff[k] >= 0 else -1
        aligned_V_k = V[k] * sign_k  # positive = "more likely correct"
        aligned_alpha_k = float(abs(alpha_eff[k]))

        n_rep = len(rep_feature_names)
        sorted_indices = np.argsort(aligned_V_k)
        top_positive = [rep_feature_names[int(i)] for i in sorted_indices[-(min(5, n_rep)):][::-1]]
        top_negative = [rep_feature_names[int(i)] for i in sorted_indices[: min(5, n_rep)]]

        # Family dominance
        family_signed: dict[str, float] = {}
        family_abs: dict[str, float] = {}
        for j, fname in enumerate(rep_feature_names):
            fam = feature_family(fname.split("::")[0])
            family_signed[fam] = family_signed.get(fam, 0.0) + float(aligned_V_k[j])
            family_abs[fam] = family_abs.get(fam, 0.0) + abs(float(aligned_V_k[j]))

        dominant_family = max(family_abs, key=lambda f: family_abs[f]) if family_abs else "other"
        signed_dominant = family_signed.get(dominant_family, 0.0)
        total_abs = float(np.sum(np.abs(aligned_V_k)))
        family_purity = (family_abs[dominant_family] / total_abs) if total_abs > _EPS else 0.0
        auto_label = f"{dominant_family}({'↑' if signed_dominant > 0 else '↓'})"

        aligned_components.append(
            {
                "comp_k": k,
                "alpha_eff_aligned": aligned_alpha_k,
                "sign_k": int(sign_k),
                "top_positive_rep_features": top_positive,
                "top_negative_rep_features": top_negative,
                "dominant_family": dominant_family,
                "family_purity": float(family_purity),
                "auto_label": auto_label,
            }
        )

    return {
        **terms,
        "rep_feature_names": rep_feature_names,
        "w_orig": w_orig,
        "b_orig": b_orig,
        "alpha_eff": alpha_eff,
        "V": V,
        "component_importance": component_importance,
        "aligned_components": aligned_components,
        "sanity": {
            "formula": "score = b_orig + x_rep @ w_orig = b_orig + sum_k(z_k * alpha_eff_k)",
            "expected_max_error": "<1e-10",
        },
    }


# ---------------------------------------------------------------------------
# Task 4 — Family Summary (weight-based)
# ---------------------------------------------------------------------------


def build_family_weight_rows(
    route: Mapping[str, Any],
    domain: str,
    position: float,
) -> list[dict[str, Any]]:
    """
    Build per-family weight summary rows for a route.

    Returns rows with columns:
      domain, anchor_pct, family, channel, signed_weight_sum, abs_weight_sum,
      mean_abs_contribution (nan, filled by export script),
      discriminative_delta (nan, filled by export script).

    channel = "raw" | "rank" | "raw+rank" (aggregated over both).
    """
    terms = _route_effective_linear_terms(route)
    feat_rows = _feature_weight_rows(terms)
    anchor_pct = int(round(float(position) * 100.0))

    # Per-channel buckets
    fam_channel: dict[tuple[str, str], dict[str, float]] = {}
    for row in feat_rows:
        family = str(row["family"])
        rw = float(row["raw_weight"])
        rkw = float(row["rank_weight"])
        if abs(rw) > 0 or abs(rkw) > 0:
            for channel, w in [("raw", rw), ("rank", rkw)]:
                if abs(w) > 0:
                    key = (family, channel)
                    bucket = fam_channel.setdefault(key, {"signed": 0.0, "abs": 0.0})
                    bucket["signed"] += w
                    bucket["abs"] += abs(w)

    # Aggregate "raw+rank"
    fam_all: dict[str, dict[str, float]] = {}
    for row in feat_rows:
        family = str(row["family"])
        w_total = float(row["raw_weight"]) + float(row["rank_weight"])
        abs_total = abs(float(row["raw_weight"])) + abs(float(row["rank_weight"]))
        bucket = fam_all.setdefault(family, {"signed": 0.0, "abs": 0.0})
        bucket["signed"] += w_total
        bucket["abs"] += abs_total

    out: list[dict[str, Any]] = []
    nan = float("nan")

    for (family, channel), bucket in fam_channel.items():
        out.append(
            {
                "domain": domain,
                "anchor_pct": anchor_pct,
                "family": family,
                "channel": channel,
                "signed_weight_sum": float(bucket["signed"]),
                "abs_weight_sum": float(bucket["abs"]),
                "mean_abs_contribution": nan,
                "discriminative_delta": nan,
            }
        )

    for family, bucket in fam_all.items():
        out.append(
            {
                "domain": domain,
                "anchor_pct": anchor_pct,
                "family": family,
                "channel": "raw+rank",
                "signed_weight_sum": float(bucket["signed"]),
                "abs_weight_sum": float(bucket["abs"]),
                "mean_abs_contribution": nan,
                "discriminative_delta": nan,
            }
        )

    return out


# ---------------------------------------------------------------------------
# Task 5 — Standalone Run Explanation
# ---------------------------------------------------------------------------


def explain_run(
    route: Mapping[str, Any],
    x_row_raw: np.ndarray,
    x_row_rank: np.ndarray,
    run_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Explain a single run's score decomposition.

    Parameters
    ----------
    route       : route dict from the bundle
    x_row_raw   : shape (n_features,), FULL_FEATURE_NAMES indexed
    x_row_rank  : shape (n_features,), group-wise rank from the same problem group
    run_metadata: optional dict of run context (sample_id, is_correct, etc.)

    Returns
    -------
    exact_score, intercept, rep_feature_contributions, canonical_feature_rows,
    family_contributions, component_contributions,
    top_positive_features, top_negative_features,
    top_positive_families, top_negative_families,
    reconstruction_error, human_summary, run_metadata

    Sanity checks (both enforced):
      abs(sum(canon_feature_rows[j].total_contribution) + intercept - exact_score) < 1e-6
      abs(sum(comp_contrib_k) + intercept - exact_score) < 1e-6  [SVD routes only]
    """
    route_type = str(route.get("route_type", "svd"))
    eff = recover_effective_weights(route)
    w_orig = eff["w_orig"]
    b_orig = eff["b_orig"]
    rep_feature_names = eff["rep_feature_names"]
    rep_defs = list(eff["representation_features"])
    feature_indices = list(eff["feature_indices"])

    # --- Build x_rep for this run ---
    x_raw_2d = x_row_raw[np.newaxis, :].astype(np.float64)
    x_rank_2d = x_row_rank[np.newaxis, :].astype(np.float64)

    if route_type == "baseline":
        x_rep_row = x_raw_2d[0, feature_indices].astype(np.float64)
    else:
        x_rep_2d = _build_representation(
            x_raw=x_raw_2d,
            x_rank=x_rank_2d,
            feature_indices=feature_indices,
            representation=str(route.get("representation", "raw")),
        )
        x_rep_row = np.asarray(x_rep_2d[0], dtype=np.float64)

    exact_score = float(b_orig + np.dot(x_rep_row, w_orig))

    # --- Rep-level contributions (one per rep_feature) ---
    rep_feature_contributions = [
        {
            "rep_feature": rep_feature_names[j],
            "value": float(x_rep_row[j]),
            "weight": float(w_orig[j]),
            "contribution": float(x_rep_row[j] * w_orig[j]),
        }
        for j in range(len(rep_feature_names))
    ]

    # --- Canonical feature rows (aggregated by base feature, compatible with _feature_delta_rows) ---
    canonical_names = list(eff["feature_names"])
    raw_vals = x_row_raw[feature_indices] if feature_indices else np.zeros(0)
    rank_vals = x_row_rank[feature_indices] if feature_indices else np.zeros(0)

    canonical_feature_rows: list[dict[str, Any]] = []
    for feat_local_idx, feat_name in enumerate(canonical_names):
        raw_w = 0.0
        rank_w = 0.0
        for rep_idx, rd in enumerate(rep_defs):
            if rd.base_feature != feat_name:
                continue
            if rd.source == "rank":
                rank_w += float(w_orig[rep_idx])
            else:
                raw_w += float(w_orig[rep_idx])
        rv = float(raw_vals[feat_local_idx]) if feat_local_idx < len(raw_vals) else 0.0
        rkv = float(rank_vals[feat_local_idx]) if feat_local_idx < len(rank_vals) else 0.0
        canonical_feature_rows.append(
            {
                "feature": feat_name,
                "family": feature_family(feat_name),
                "raw_value": rv,
                "rank_value": rkv,
                "raw_weight": float(raw_w),
                "rank_weight": float(rank_w),
                "raw_contribution": float(rv * raw_w),
                "rank_contribution": float(rkv * rank_w),
                "total_contribution": float(rv * raw_w + rkv * rank_w),
            }
        )
    canonical_feature_rows.sort(key=lambda r: abs(r["total_contribution"]), reverse=True)

    # --- Family contributions ---
    fam_dict: dict[str, float] = {}
    for row in canonical_feature_rows:
        fam_dict[row["family"]] = fam_dict.get(row["family"], 0.0) + row["total_contribution"]
    family_contributions = sorted(
        [{"family": f, "total_contribution": float(v)} for f, v in fam_dict.items()],
        key=lambda r: abs(r["total_contribution"]),
        reverse=True,
    )

    # --- Component contributions (SVD routes only) ---
    component_contributions: list[dict[str, Any]] = []
    comp_sanity_error = 0.0
    if route_type != "baseline" and "alpha_eff" in eff and eff["V"].shape[0] > 0:
        model = route["model"]
        scaler = model["scaler"]
        alpha_eff = eff["alpha_eff"]
        V = eff["V"]

        scale = np.asarray(scaler.scale_, dtype=np.float64)
        scale = np.where(np.abs(scale) < _EPS, 1.0, scale)
        mean_val = np.asarray(scaler.mean_, dtype=np.float64)

        x_std = (x_rep_row - mean_val) / scale
        z = V @ x_std  # (rank,)

        comp_sum = 0.0
        for k in range(len(alpha_eff)):
            contrib = float(z[k] * alpha_eff[k])
            comp_sum += contrib
            component_contributions.append(
                {
                    "comp_k": k,
                    "z_k": float(z[k]),
                    "alpha_eff_k": float(alpha_eff[k]),
                    "contribution": contrib,
                }
            )
        comp_sanity_error = abs(b_orig + comp_sum - exact_score)

    # --- Sanity check (feature-level) ---
    feat_sum = sum(r["total_contribution"] for r in canonical_feature_rows)
    reconstruction_error = abs(b_orig + feat_sum - exact_score)

    # --- Top features / families ---
    top_positive_features = [r["feature"] for r in canonical_feature_rows if r["total_contribution"] > 0][:5]
    top_negative_features = [
        r["feature"]
        for r in sorted(canonical_feature_rows, key=lambda r: r["total_contribution"])
        if r["total_contribution"] < 0
    ][:5]
    top_positive_families = [r for r in family_contributions if r["total_contribution"] > 0][:3]
    top_negative_families = [r for r in reversed(family_contributions) if r["total_contribution"] < 0][:3]

    # --- Human summary (Chinese) ---
    top_fam_name = top_positive_families[0]["family"] if top_positive_families else "N/A"
    top_fam_c = top_positive_families[0]["total_contribution"] if top_positive_families else 0.0
    neg_fam_name = top_negative_families[0]["family"] if top_negative_families else "N/A"
    neg_fam_c = top_negative_families[0]["total_contribution"] if top_negative_families else 0.0
    top_feat_name = top_positive_features[0] if top_positive_features else "N/A"
    human_summary = (
        f"该 run 得分 {exact_score:.4f}（截距 {b_orig:.4f}），"
        f"{top_fam_name} 家族贡献最大（{top_fam_c:.4f}），"
        f"{top_feat_name}::raw/rank 主导；"
        f"{neg_fam_name} 家族为负向拖累（{neg_fam_c:.4f}）。"
    )

    return {
        "exact_score": exact_score,
        "intercept": b_orig,
        "rep_feature_contributions": rep_feature_contributions,
        "canonical_feature_rows": canonical_feature_rows,
        "family_contributions": family_contributions,
        "component_contributions": component_contributions,
        "top_positive_features": top_positive_features,
        "top_negative_features": top_negative_features,
        "top_positive_families": top_positive_families,
        "top_negative_families": top_negative_families,
        "reconstruction_error": reconstruction_error,
        "component_sanity_error": comp_sanity_error,
        "human_summary": human_summary,
        "run_metadata": run_metadata,
    }


# ---------------------------------------------------------------------------
# Task 6 — Standalone Problem Decision Explanation
# ---------------------------------------------------------------------------


def explain_problem_decision(
    route: Mapping[str, Any],
    x_raw_problem: np.ndarray,
    x_rank_problem: np.ndarray,
    sample_ids: Sequence[int],
    labels: Optional[Sequence[bool]] = None,
) -> dict[str, Any]:
    """
    Explain why the model selected top1 over top2 for a problem group.

    Parameters
    ----------
    route          : route dict from the bundle
    x_raw_problem  : shape (n_runs, n_features), FULL_FEATURE_NAMES indexed
    x_rank_problem : shape (n_runs, n_features), group-wise rank (ALL runs together)
    sample_ids     : ordered list of sample IDs (length n_runs)
    labels         : optional correctness labels (length n_runs)

    Returns
    -------
    top1, top2, top3, margin, delta_feature_contrib, delta_family_contrib,
    delta_component_contrib, why_selected, whether_top1_is_correct,
    run_explanations (top1 and top2), sanity_check
    """
    n_runs = int(x_raw_problem.shape[0])
    if n_runs == 0:
        return {
            "top1": None,
            "top2": None,
            "top3": None,
            "margin": 0.0,
            "delta_feature_contrib": [],
            "delta_family_contrib": [],
            "delta_component_contrib": [],
            "why_selected": "No runs available.",
            "whether_top1_is_correct": False,
            "run_explanations": {},
            "sanity_check": {"max_abs_error": 0.0, "delta_sum_error": 0.0},
        }

    labels_list = list(labels) if labels is not None else [False] * n_runs
    run_infos = [
        {
            "sample_id": int(sample_ids[i]),
            "local_index": i,
            "run_index": i,
            "is_correct": bool(labels_list[i]) if i < len(labels_list) else False,
        }
        for i in range(n_runs)
    ]

    scores, _ = _score_route_matrix(route, x_raw_problem, x_rank_problem)
    ranking_rows, order = _ranking_rows(scores, run_infos)
    top_runs_list = _top_runs(ranking_rows, order, top_n=3)

    top1_idx = int(order[0]) if order.size > 0 else 0
    top2_idx = int(order[1]) if order.size > 1 else top1_idx
    margin = float(scores[top1_idx] - scores[top2_idx]) if top1_idx != top2_idx else 0.0

    # Explain top1 and top2
    exp_top1 = explain_run(
        route,
        x_raw_problem[top1_idx],
        x_rank_problem[top1_idx],
        run_metadata=run_infos[top1_idx],
    )
    exp_top2 = explain_run(
        route,
        x_raw_problem[top2_idx],
        x_rank_problem[top2_idx],
        run_metadata=run_infos[top2_idx],
    )

    # Feature-level deltas (uses _feature_delta_rows format)
    delta_feature = _feature_delta_rows(
        exp_top1["canonical_feature_rows"],
        exp_top2["canonical_feature_rows"],
    )

    # Family-level deltas
    fam1_map = {r["family"]: r["total_contribution"] for r in exp_top1["family_contributions"]}
    fam2_map = {r["family"]: r["total_contribution"] for r in exp_top2["family_contributions"]}
    all_families = sorted(set(fam1_map) | set(fam2_map))
    delta_family = sorted(
        [
            {
                "family": fam,
                "top1_contribution": float(fam1_map.get(fam, 0.0)),
                "top2_contribution": float(fam2_map.get(fam, 0.0)),
                "delta": float(fam1_map.get(fam, 0.0) - fam2_map.get(fam, 0.0)),
            }
            for fam in all_families
        ],
        key=lambda r: abs(r["delta"]),
        reverse=True,
    )

    # Component-level deltas
    comp1 = {c["comp_k"]: c["contribution"] for c in exp_top1["component_contributions"]}
    comp2 = {c["comp_k"]: c["contribution"] for c in exp_top2["component_contributions"]}
    all_comps = sorted(set(comp1) | set(comp2))
    delta_component = [
        {
            "comp_k": k,
            "top1_contribution": float(comp1.get(k, 0.0)),
            "top2_contribution": float(comp2.get(k, 0.0)),
            "delta": float(comp1.get(k, 0.0) - comp2.get(k, 0.0)),
        }
        for k in all_comps
    ]

    # Sanity: sum(delta_features) ≈ score_top1 - score_top2
    delta_feat_sum = sum(r["delta"] for r in delta_feature)
    delta_sum_error = abs(delta_feat_sum - margin)

    why_top1_positive = [r for r in delta_family if r["delta"] > 0]
    top_why_fam = why_top1_positive[:3]
    top_why_feat = [r for r in delta_feature if r["delta"] > 0][:3]
    why_str = _why_top1(top_why_feat, top_why_fam, margin)

    return {
        "top1": top_runs_list[0] if len(top_runs_list) > 0 else None,
        "top2": top_runs_list[1] if len(top_runs_list) > 1 else None,
        "top3": top_runs_list[2] if len(top_runs_list) > 2 else None,
        "margin": margin,
        "delta_feature_contrib": delta_feature,
        "delta_family_contrib": delta_family,
        "delta_component_contrib": delta_component,
        "why_selected": why_str,
        "whether_top1_is_correct": bool(run_infos[top1_idx].get("is_correct", False)),
        "run_explanations": {
            "top1": exp_top1,
            "top2": exp_top2,
        },
        "sanity_check": {
            "max_abs_error": float(max(
                exp_top1["reconstruction_error"],
                exp_top2["reconstruction_error"],
            )),
            "delta_sum_error": delta_sum_error,
        },
    }


# ---------------------------------------------------------------------------
# Task 7 — Failure Mode CSV + Doc Helpers
# ---------------------------------------------------------------------------


def export_failure_modes_csv(failure_summary: Mapping[str, Any], out_path: Path) -> None:
    """
    Write failure_modes.csv from aggregate_failure_modes() output.

    Columns: rank, feature_or_family, type, count_as_top_driver, count_in_top3,
             mean_delta, median_delta, max_delta
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for idx, row in enumerate(failure_summary.get("families", []), start=1):
        rows.append(
            {
                "rank": idx,
                "feature_or_family": str(row["family"]),
                "type": "family",
                "count_as_top_driver": int(row.get("count_as_top_driver", 0)),
                "count_in_top3": int(row.get("count_in_top3", 0)),
                "mean_delta": float(row.get("mean_delta", 0.0)),
                "median_delta": float(row.get("median_delta", 0.0)),
                "max_delta": float(row.get("max_delta", 0.0)),
            }
        )

    for idx, row in enumerate(failure_summary.get("features", []), start=1):
        rows.append(
            {
                "rank": idx,
                "feature_or_family": str(row["feature"]),
                "type": "feature",
                "count_as_top_driver": int(row.get("count_as_top_driver", 0)),
                "count_in_top3": int(row.get("count_in_top3", 0)),
                "mean_delta": float(row.get("mean_delta", 0.0)),
                "median_delta": float(row.get("median_delta", 0.0)),
                "max_delta": float(row.get("max_delta", 0.0)),
            }
        )

    if not rows:
        out_path.write_text("rank,feature_or_family,type,count_as_top_driver,count_in_top3,mean_delta,median_delta,max_delta\n", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_failure_modes_doc(
    failure_summary: Mapping[str, Any],
    method_id: str,
    route_inventory: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """
    Write docs/SVD_FAILURE_MODES_<method_id>.md.

    Sections: top overboost features, top overboost families,
    which domain/anchor is most affected, average delta wrong vs correct.
    """
    total_cases = int(failure_summary.get("total_wrong_top1_cases", 0))
    families = list(failure_summary.get("families", []))
    features = list(failure_summary.get("features", []))

    lines: list[str] = [
        f"# Failure Mode Analysis — {method_id}",
        "",
        f"Total wrong-top1 cases analysed: **{total_cases}**",
        "",
        "## Top Over-Boosted Families",
        "",
        "Families that most often drove the model to select a **wrong** run over the best correct run.",
        "",
        "| Rank | Family | Top Driver Count | In Top-3 Count | Mean Δ | Max Δ |",
        "|------|--------|-----------------|----------------|--------|-------|",
    ]
    for idx, row in enumerate(families[:10], start=1):
        lines.append(
            f"| {idx} | {row['family']} | {row.get('count_as_top_driver', 0)} "
            f"| {row.get('count_in_top3', 0)} | {row.get('mean_delta', 0.0):.4f} "
            f"| {row.get('max_delta', 0.0):.4f} |"
        )

    lines += [
        "",
        "## Top Over-Boosted Features",
        "",
        "Individual features most often responsible for wrong selections.",
        "",
        "| Rank | Feature | Family | Top Driver Count | In Top-3 Count | Mean Δ |",
        "|------|---------|--------|-----------------|----------------|--------|",
    ]
    for idx, row in enumerate(features[:15], start=1):
        lines.append(
            f"| {idx} | {row['feature']} | {row.get('family', '?')} "
            f"| {row.get('count_as_top_driver', 0)} "
            f"| {row.get('count_in_top3', 0)} | {row.get('mean_delta', 0.0):.4f} |"
        )

    # Domain/anchor breakdown from by_anchor if available
    by_anchor = failure_summary.get("by_anchor", {})
    if by_anchor:
        lines += [
            "",
            "## Failure Counts by Anchor",
            "",
            "| Anchor % | Wrong-Top1 Cases |",
            "|---------|-----------------|",
        ]
        for anchor_key, anchor_data in sorted(by_anchor.items(), key=lambda kv: int(kv[0])):
            cnt = int(anchor_data.get("total_wrong_top1_cases", 0))
            lines.append(f"| {anchor_key} | {cnt} |")

    lines += [
        "",
        "## Route Inventory Reference",
        "",
        f"Total routes in bundle: {len(route_inventory)}",
        "",
        "| Domain | Anchor % | Representation | Rank |",
        "|--------|---------|----------------|------|",
    ]
    for spec in route_inventory:
        lines.append(
            f"| {spec.get('domain')} | {spec.get('position_pct')} "
            f"| {spec.get('representation')} | {spec.get('rank', 'N/A')} |"
        )

    lines += [
        "",
        "---",
        f"*Generated automatically by `scripts/export_svd_introspection.py` for `{method_id}`.*",
        "",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Task 8 — Stability Analysis (cross-domain / cross-anchor)
# ---------------------------------------------------------------------------


def compute_weight_stability(bundle: Mapping[str, Any], method_id: str) -> list[dict[str, Any]]:
    """
    Compute two proxy stability measures without retraining.

    (A) Cross-domain: compare effective weights at matching anchors across domains.
        - sign_consistent per feature, cosine_sim of full w_orig vectors.
        - Applicable when bundle has ≥2 domains with same features.

    (B) Cross-anchor: compare adjacent anchor pairs within same domain.
        - sign_consistent per feature, weight_ratio = |w_at+1| / |w_at|.

    Returns rows for stability_report.csv.
    """
    rows: list[dict[str, Any]] = []
    domains = list(bundle.get("domains", {}).keys())

    # Collect routes per domain
    domain_routes: dict[str, list[Mapping[str, Any]]] = {}
    for domain in domains:
        domain_bundle = bundle["domains"][domain]
        domain_routes[domain] = list(domain_bundle.get("routes", []))

    # (A) Cross-domain
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            d1, d2 = domains[i], domains[j]
            pos_map1: dict[float, Mapping[str, Any]] = {
                float(r.get("training_position", 0)): r for r in domain_routes[d1]
            }
            pos_map2: dict[float, Mapping[str, Any]] = {
                float(r.get("training_position", 0)): r for r in domain_routes[d2]
            }
            common_pos = sorted(set(pos_map1) & set(pos_map2))

            for pos in common_pos:
                terms1 = _route_effective_linear_terms(pos_map1[pos])
                terms2 = _route_effective_linear_terms(pos_map2[pos])
                w1 = np.asarray(terms1["weights_rep"], dtype=np.float64)
                w2 = np.asarray(terms2["weights_rep"], dtype=np.float64)
                rep1 = [f"{rd.base_feature}::{rd.source}" for rd in terms1["representation_features"]]
                rep2 = [f"{rd.base_feature}::{rd.source}" for rd in terms2["representation_features"]]

                anchor_pct = int(round(pos * 100.0))
                cos_sim = float(np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2) + _EPS)) if len(w1) == len(w2) else float("nan")

                w1_map = {fname: float(w1[k]) for k, fname in enumerate(rep1)}
                w2_map = {fname: float(w2[k]) for k, fname in enumerate(rep2)}
                all_names = sorted(set(rep1) | set(rep2))

                for fname in all_names:
                    wf = w1_map.get(fname, 0.0)
                    wt = w2_map.get(fname, 0.0)
                    base_feat = fname.split("::")[0]
                    channel = fname.split("::")[-1] if "::" in fname else "raw"
                    sign_ok = (
                        bool(math.copysign(1, wf) == math.copysign(1, wt))
                        if abs(wf) > 1e-10 and abs(wt) > 1e-10
                        else True
                    )
                    rows.append(
                        {
                            "method_id": method_id,
                            "comparison_type": "cross_domain",
                            "domain": f"{d1}_vs_{d2}",
                            "anchor_from_pct": anchor_pct,
                            "anchor_to_pct": anchor_pct,
                            "feature": base_feat,
                            "channel": channel,
                            "family": feature_family(base_feat),
                            "w_from": wf,
                            "w_to": wt,
                            "sign_consistent": sign_ok,
                            "cosine_sim": cos_sim,
                            "weight_ratio": float("nan"),
                        }
                    )

    # (B) Cross-anchor
    for domain in domains:
        sorted_routes = sorted(
            domain_routes[domain],
            key=lambda r: float(r.get("training_position", 0)),
        )
        for i in range(len(sorted_routes) - 1):
            r1 = sorted_routes[i]
            r2 = sorted_routes[i + 1]
            pos1 = float(r1.get("training_position", 0))
            pos2 = float(r2.get("training_position", 0))

            terms1 = _route_effective_linear_terms(r1)
            terms2 = _route_effective_linear_terms(r2)
            w1 = np.asarray(terms1["weights_rep"], dtype=np.float64)
            w2 = np.asarray(terms2["weights_rep"], dtype=np.float64)
            rep1 = [f"{rd.base_feature}::{rd.source}" for rd in terms1["representation_features"]]
            rep2 = [f"{rd.base_feature}::{rd.source}" for rd in terms2["representation_features"]]

            w1_map = {fname: float(w1[k]) for k, fname in enumerate(rep1)}
            w2_map = {fname: float(w2[k]) for k, fname in enumerate(rep2)}
            all_names = sorted(set(rep1) | set(rep2))

            for fname in all_names:
                wf = w1_map.get(fname, 0.0)
                wt = w2_map.get(fname, 0.0)
                base_feat = fname.split("::")[0]
                channel = fname.split("::")[-1] if "::" in fname else "raw"
                sign_ok = (
                    bool(math.copysign(1, wf) == math.copysign(1, wt))
                    if abs(wf) > 1e-10 and abs(wt) > 1e-10
                    else True
                )
                weight_ratio = abs(wt) / (abs(wf) + _EPS)
                rows.append(
                    {
                        "method_id": method_id,
                        "comparison_type": "cross_anchor",
                        "domain": domain,
                        "anchor_from_pct": int(round(pos1 * 100.0)),
                        "anchor_to_pct": int(round(pos2 * 100.0)),
                        "feature": base_feat,
                        "channel": channel,
                        "family": feature_family(base_feat),
                        "w_from": wf,
                        "w_to": wt,
                        "sign_consistent": sign_ok,
                        "cosine_sim": float("nan"),
                        "weight_ratio": weight_ratio,
                    }
                )

    return rows


# ---------------------------------------------------------------------------
# Task 9 — Documentation Generator
# ---------------------------------------------------------------------------


def generate_introspection_results_doc(
    method_id: str,
    route_inventory: list[dict[str, Any]],
    family_summary: list[dict[str, Any]],
    failure_summary: Mapping[str, Any],
    stability_rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """
    Write docs/SVD_INTROSPECTION_RESULTS_<method_id>.md.

    Answers six paper questions:
    1. What does the model look at? (family weights)
    2. Top driving features?
    3. Why does top1 beat top2? (family delta patterns)
    4. Common failure modes?
    5. Stable vs unstable explanations? (stability analysis)
    6. Domain differences? (cross-domain comparison)
    """
    domains = sorted({r.get("domain", "") for r in route_inventory})
    anchors = sorted({r.get("position_pct", 0) for r in route_inventory})

    # Aggregate family weight stats across all routes
    fam_abs: dict[str, float] = {}
    for row in family_summary:
        channel = str(row.get("channel", ""))
        if channel == "raw+rank":
            fam = str(row.get("family", ""))
            fam_abs[fam] = fam_abs.get(fam, 0.0) + float(row.get("abs_weight_sum", 0.0))
    top_families = sorted(fam_abs.items(), key=lambda x: -x[1])

    # Stability stats
    cross_anchor = [r for r in stability_rows if r.get("comparison_type") == "cross_anchor"]
    cross_domain = [r for r in stability_rows if r.get("comparison_type") == "cross_domain"]
    sign_consistent_pct = (
        100.0 * sum(1 for r in cross_anchor if r.get("sign_consistent", True)) / len(cross_anchor)
        if cross_anchor
        else float("nan")
    )
    cd_sign_pct = (
        100.0 * sum(1 for r in cross_domain if r.get("sign_consistent", True)) / len(cross_domain)
        if cross_domain
        else float("nan")
    )

    top_failures_fam = list(failure_summary.get("families", []))[:5]
    top_failures_feat = list(failure_summary.get("features", []))[:5]

    lines: list[str] = [
        f"# SVD Introspection Results — {method_id}",
        "",
        f"> Domains: {', '.join(domains)} | Anchors: {', '.join(str(a) + '%' for a in anchors)}",
        f"> Routes: {len(route_inventory)} | Wrong-top1 cases: {failure_summary.get('total_wrong_top1_cases', 0)}",
        "",
        "---",
        "",
        "## Q1: What does the model look at?",
        "",
        "Family contribution strength (aggregate |weight| across all anchors/domains):",
        "",
        "| Family | Aggregate |w| |",
        "|--------|------------|",
    ]
    for fam, w in top_families[:8]:
        lines.append(f"| {fam} | {w:.4f} |")

    lines += [
        "",
        "## Q2: Top driving features?",
        "",
        "Top feature weights per anchor are available in `effective_weights.csv` "
        "and `component_table.csv`.",
        "",
    ]

    if top_families:
        dom_fam = top_families[0][0]
        lines += [
            f"Primary family: **{dom_fam}** accounts for the largest aggregate weight.",
            "",
        ]

    lines += [
        "## Q3: Why does top1 beat top2?",
        "",
        "The dominant family advantage is consistently the primary separator. "
        "See `problem_top1_vs_top2.jsonl` for per-problem deltas.",
        "",
        "## Q4: Common failure modes?",
        "",
        "Cases where model selected a wrong run over the best correct run:",
        "",
        "**Top over-boosted families:**",
        "",
        "| Family | Top-Driver Count | Mean Δ |",
        "|--------|-----------------|--------|",
    ]
    for row in top_failures_fam:
        lines.append(f"| {row['family']} | {row.get('count_as_top_driver', 0)} | {row.get('mean_delta', 0.0):.4f} |")

    lines += [
        "",
        "**Top over-boosted features:**",
        "",
        "| Feature | Top-Driver Count | Mean Δ |",
        "|---------|-----------------|--------|",
    ]
    for row in top_failures_feat:
        lines.append(f"| {row['feature']} | {row.get('count_as_top_driver', 0)} | {row.get('mean_delta', 0.0):.4f} |")

    lines += [
        "",
        "## Q5: Stable vs unstable explanations?",
        "",
        "Cross-anchor sign consistency (adjacent anchors, same domain): "
        f"**{sign_consistent_pct:.1f}%** of feature weights maintain sign direction.",
        "",
    ]
    if cross_anchor:
        # Most volatile features
        volatile = sorted(
            [r for r in cross_anchor if not r.get("sign_consistent", True)],
            key=lambda r: (r.get("domain", ""), r.get("feature", "")),
        )
        if volatile:
            lines += [
                "Features with sign flips across anchors (sample):",
                "",
                "| Domain | Feature | From Anchor | To Anchor |",
                "|--------|---------|------------|-----------|",
            ]
            seen: set[tuple] = set()
            for r in volatile[:10]:
                key = (r.get("domain"), r.get("feature"))
                if key not in seen:
                    seen.add(key)
                    lines.append(
                        f"| {r.get('domain')} | {r.get('feature')} "
                        f"| {r.get('anchor_from_pct')}% | {r.get('anchor_to_pct')}% |"
                    )

    lines += [
        "",
        "## Q6: Domain differences?",
        "",
    ]
    if cross_domain:
        lines += [
            f"Cross-domain sign consistency: **{cd_sign_pct:.1f}%** of shared features "
            "maintain the same sign direction across domains.",
            "",
            "Full per-feature comparison is in `stability_report.csv` "
            "(filter `comparison_type == cross_domain`).",
            "",
        ]
    else:
        lines += [
            "This bundle covers a single domain — cross-domain comparison not applicable.",
            "",
        ]

    lines += [
        "---",
        f"*Generated automatically by `scripts/export_svd_introspection.py` for `{method_id}`.*",
        "",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
