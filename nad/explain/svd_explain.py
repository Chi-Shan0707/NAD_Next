from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from nad.core.selectors.trajectory_impl import DEFAULT_REFLECTION_THRESHOLD
from nad.core.views.reader import CacheReader
from nad.ops.earlystop_svd import (
    FULL_FEATURE_NAMES,
    _build_representation,
    _predict_svd_lr,
    _rank_transform_matrix,
    extract_earlystop_signals_for_positions,
)


EXPLAIN_ANCHORS: tuple[float, ...] = (0.1, 0.4, 0.7, 1.0)
ANCHOR_TO_SLOT_INDEX: dict[float, int] = {
    0.1: 0,
    0.4: 3,
    0.7: 6,
    1.0: 9,
}

_EPS = 1e-12
_FULL_FEATURE_TO_INDEX = {name: idx for idx, name in enumerate(FULL_FEATURE_NAMES)}


def normalize_anchor(anchor: float | int | str) -> float:
    value = float(anchor)
    if value > 1.0:
        value = value / 100.0
    for candidate in EXPLAIN_ANCHORS:
        if abs(float(candidate) - value) < 1e-9:
            return float(candidate)
    raise ValueError(f"Unsupported anchor: {anchor}")


def anchor_pct(anchor: float | int | str) -> int:
    return int(round(normalize_anchor(anchor) * 100.0))


def feature_family(feature_name: str) -> str:
    name = str(feature_name)
    if name.startswith("tok_conf_"):
        return "confidence"
    if name.startswith("tok_gini_") or name.startswith("tok_neg_entropy_"):
        return "uncertainty"
    if name.startswith("tok_selfcert_") or name.startswith("tok_logprob_"):
        return "self_cert_logprob"
    if name.startswith("traj_"):
        return "trajectory"
    if (
        name.startswith("has_")
        or name.startswith("nc_")
        or name == "self_similarity"
    ):
        return "availability_meta"
    if (
        name in {
            "tail_q10",
            "head_tail_gap",
            "tail_variance",
            "last_event_tail_conf",
            "event_pre_post_delta",
        }
        or name.startswith("terminal_")
    ):
        return "terminal_tail"
    return "other"


def collect_bundle_required_features(bundle: Mapping[str, Any]) -> set[str]:
    required: set[str] = set()
    for domain_bundle in bundle.get("domains", {}).values():
        for route in domain_bundle.get("routes", []):
            route_type = str(route.get("route_type", "svd"))
            if route_type == "baseline":
                required.add(str(route["signal_name"]))
            else:
                required.update(str(name) for name in route.get("feature_names", []))
    return required


def build_problem_anchor_tensor(
    reader: CacheReader,
    run_ids: Sequence[int],
    *,
    positions: Sequence[float] = EXPLAIN_ANCHORS,
    required_features: Optional[set[str]] = None,
    reflection_threshold: float = DEFAULT_REFLECTION_THRESHOLD,
) -> np.ndarray:
    positions_tuple = tuple(float(v) for v in positions)
    tensor = np.zeros((len(run_ids), len(positions_tuple), len(FULL_FEATURE_NAMES)), dtype=np.float64)
    req = None if required_features is None else set(str(v) for v in required_features)

    for row_idx, sample_id in enumerate(run_ids):
        signal_map = extract_earlystop_signals_for_positions(
            reader=reader,
            run_id=int(sample_id),
            positions=positions_tuple,
            required_features=req,
            reflection_threshold=float(reflection_threshold),
        )
        for feat_idx, feat_name in enumerate(FULL_FEATURE_NAMES):
            vals = np.asarray(signal_map.get(feat_name, [0.0] * len(positions_tuple)), dtype=np.float64)
            if vals.size >= len(positions_tuple):
                tensor[row_idx, :, feat_idx] = vals[: len(positions_tuple)]
    return tensor


def get_anchor_route(bundle: Mapping[str, Any], domain: str, anchor: float | int | str) -> Mapping[str, Any]:
    anchor_value = normalize_anchor(anchor)
    domain_bundle = bundle.get("domains", {}).get(str(domain))
    if not domain_bundle:
        raise KeyError(f"Domain not present in bundle: {domain}")
    routes = list(domain_bundle.get("routes", []))
    if not routes:
        raise KeyError(f"No routes for domain: {domain}")
    for route in routes:
        route_anchor = route.get("training_position")
        if route_anchor is not None and abs(float(route_anchor) - anchor_value) < 1e-9:
            return route
    slot_idx = ANCHOR_TO_SLOT_INDEX[anchor_value]
    if slot_idx >= len(routes):
        raise KeyError(f"Route index out of range for anchor={anchor_value}")
    return routes[slot_idx]


def _canonical_feature_names_for_route(route: Mapping[str, Any]) -> list[str]:
    if str(route.get("route_type", "svd")) == "baseline":
        return [str(route["signal_name"])]
    return [str(v) for v in route.get("feature_names", [])]


def _feature_indices_for_route(route: Mapping[str, Any]) -> list[int]:
    route_type = str(route.get("route_type", "svd"))
    if route_type == "baseline":
        return [_FULL_FEATURE_TO_INDEX[str(route["signal_name"])]]
    return [int(v) for v in route.get("feature_indices", [])]


@dataclass(frozen=True)
class _RepresentationFeature:
    base_feature: str
    source: str


def _representation_features(route: Mapping[str, Any]) -> list[_RepresentationFeature]:
    route_type = str(route.get("route_type", "svd"))
    if route_type == "baseline":
        return [_RepresentationFeature(base_feature=str(route["signal_name"]), source="raw")]

    feature_names = _canonical_feature_names_for_route(route)
    representation = str(route.get("representation", "raw"))
    if representation in {"raw", "centered_raw", "zscore_within_problem_raw"}:
        return [_RepresentationFeature(base_feature=name, source="raw") for name in feature_names]
    if representation == "rank":
        return [_RepresentationFeature(base_feature=name, source="rank") for name in feature_names]
    if representation in {"raw+rank", "centered_raw+rank"}:
        return (
            [_RepresentationFeature(base_feature=name, source="raw") for name in feature_names]
            + [_RepresentationFeature(base_feature=name, source="rank") for name in feature_names]
        )
    raise ValueError(f"Unsupported representation for explanation: {representation}")


def _route_effective_linear_terms(route: Mapping[str, Any]) -> dict[str, Any]:
    route_type = str(route.get("route_type", "svd"))
    feature_names = _canonical_feature_names_for_route(route)
    feature_indices = _feature_indices_for_route(route)
    rep_defs = _representation_features(route)

    if route_type == "baseline":
        return {
            "route_type": route_type,
            "feature_names": feature_names,
            "feature_indices": feature_indices,
            "representation_features": rep_defs,
            "weights_rep": np.asarray([1.0], dtype=np.float64),
            "intercept_effective": 0.0,
        }

    model = route["model"]
    scaler = model["scaler"]
    svd = model["svd"]
    lr = model["lr"]

    beta = np.asarray(lr.coef_, dtype=np.float64).reshape(-1)
    if beta.size != int(svd.components_.shape[0]):
        raise ValueError("Unexpected latent coefficient shape for SVD LR route")

    if bool(model.get("whiten", False)):
        singular_values = np.asarray(svd.singular_values_, dtype=np.float64)
        singular_values = np.where(np.abs(singular_values) < _EPS, 1.0, singular_values)
        beta_latent = beta / singular_values
    else:
        beta_latent = beta

    w_scaled = np.asarray(svd.components_.T, dtype=np.float64) @ beta_latent
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale = np.where(np.abs(scale) < _EPS, 1.0, scale)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    weights_rep = w_scaled / scale
    intercept_effective = float(np.asarray(lr.intercept_, dtype=np.float64).reshape(-1)[0] - np.dot(mean / scale, w_scaled))

    return {
        "route_type": route_type,
        "feature_names": feature_names,
        "feature_indices": feature_indices,
        "representation_features": rep_defs,
        "weights_rep": np.asarray(weights_rep, dtype=np.float64),
        "intercept_effective": intercept_effective,
    }


def _feature_weight_rows(route_terms: Mapping[str, Any]) -> list[dict[str, Any]]:
    feature_names = [str(v) for v in route_terms["feature_names"]]
    rep_defs = list(route_terms["representation_features"])
    weights_rep = np.asarray(route_terms["weights_rep"], dtype=np.float64)

    rows = []
    for feat_idx, feat_name in enumerate(feature_names):
        raw_weight = 0.0
        rank_weight = 0.0
        for rep_idx, rep_def in enumerate(rep_defs):
            if rep_def.base_feature != feat_name:
                continue
            if rep_def.source == "rank":
                rank_weight += float(weights_rep[rep_idx])
            else:
                raw_weight += float(weights_rep[rep_idx])
        rows.append(
            {
                "feature": feat_name,
                "family": feature_family(feat_name),
                "raw_weight": float(raw_weight),
                "rank_weight": float(rank_weight),
                "signed_weight": float(raw_weight + rank_weight),
                "strength": float(abs(raw_weight) + abs(rank_weight)),
            }
        )
    return rows


def _family_strength_rows(feature_weight_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, dict[str, float]] = {}
    for row in feature_weight_rows:
        family = str(row["family"])
        bucket = by_family.setdefault(
            family,
            {
                "raw_weight_abs_sum": 0.0,
                "rank_weight_abs_sum": 0.0,
                "strength": 0.0,
                "signed_weight_sum": 0.0,
            },
        )
        bucket["raw_weight_abs_sum"] += abs(float(row["raw_weight"]))
        bucket["rank_weight_abs_sum"] += abs(float(row["rank_weight"]))
        bucket["strength"] += float(row["strength"])
        bucket["signed_weight_sum"] += float(row["signed_weight"])

    out = []
    for family, bucket in by_family.items():
        out.append(
            {
                "family": family,
                "raw_weight_abs_sum": float(bucket["raw_weight_abs_sum"]),
                "rank_weight_abs_sum": float(bucket["rank_weight_abs_sum"]),
                "strength": float(bucket["strength"]),
                "signed_weight_sum": float(bucket["signed_weight_sum"]),
            }
        )
    out.sort(key=lambda row: float(row["strength"]), reverse=True)
    return out


def _format_run_info(run_info: Optional[Mapping[str, Any]], row_idx: int) -> dict[str, Any]:
    if run_info is None:
        return {
            "local_index": int(row_idx),
            "sample_id": int(row_idx),
            "run_index": int(row_idx),
            "is_correct": False,
        }
    out = dict(run_info)
    out.setdefault("local_index", int(row_idx))
    out["local_index"] = int(out["local_index"])
    if "sample_id" in out:
        out["sample_id"] = int(out["sample_id"])
    if "run_index" in out:
        out["run_index"] = int(out["run_index"])
    if "is_correct" in out:
        out["is_correct"] = bool(out["is_correct"])
    return out


def _score_route_matrix(route: Mapping[str, Any], x_raw: np.ndarray, x_rank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    route_type = str(route.get("route_type", "svd"))
    if route_type == "baseline":
        feature_indices = _feature_indices_for_route(route)
        values = np.asarray(x_raw[:, feature_indices], dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        scores = values[:, 0]
        return scores.astype(np.float64, copy=False), values.astype(np.float64, copy=False)

    x_rep = _build_representation(
        x_raw=x_raw,
        x_rank=x_rank,
        feature_indices=_feature_indices_for_route(route),
        representation=str(route.get("representation", "raw")),
    )
    scores = np.asarray(_predict_svd_lr(route["model"], x_rep), dtype=np.float64)
    return scores, np.asarray(x_rep, dtype=np.float64)


def _ranking_rows(scores: np.ndarray, run_infos: Sequence[Optional[Mapping[str, Any]]]) -> tuple[list[dict[str, Any]], np.ndarray]:
    n_rows = int(scores.shape[0])
    infos = [_format_run_info(run_infos[idx] if idx < len(run_infos) else None, idx) for idx in range(n_rows)]
    tie_values = np.asarray(
        [
            int(info.get("run_index", info.get("sample_id", idx)))
            for idx, info in enumerate(infos)
        ],
        dtype=np.int64,
    )
    order = np.lexsort((tie_values, -np.asarray(scores, dtype=np.float64)))
    ranks = np.empty(n_rows, dtype=np.int64)
    ranks[order] = np.arange(1, n_rows + 1, dtype=np.int64)

    rows: list[dict[str, Any]] = []
    for idx, info in enumerate(infos):
        pct = 1.0 if n_rows <= 1 else float((n_rows - int(ranks[idx])) / float(n_rows - 1))
        rows.append(
            {
                **info,
                "score": float(scores[idx]),
                "rank": int(ranks[idx]),
                "percentile": round(float(100.0 * pct), 2),
            }
        )
    return rows, order


def _family_contribution_rows(feature_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, dict[str, float]] = {}
    for row in feature_rows:
        family = str(row["family"])
        bucket = by_family.setdefault(
            family,
            {
                "raw_contribution": 0.0,
                "rank_contribution": 0.0,
                "total_contribution": 0.0,
            },
        )
        bucket["raw_contribution"] += float(row["raw_contribution"])
        bucket["rank_contribution"] += float(row["rank_contribution"])
        bucket["total_contribution"] += float(row["total_contribution"])
    out = []
    for family, bucket in by_family.items():
        out.append(
            {
                "family": family,
                "raw_contribution": float(bucket["raw_contribution"]),
                "rank_contribution": float(bucket["rank_contribution"]),
                "total_contribution": float(bucket["total_contribution"]),
                "strength": float(abs(bucket["total_contribution"])),
            }
        )
    out.sort(key=lambda row: abs(float(row["total_contribution"])), reverse=True)
    return out


def _feature_delta_rows(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    right_map = {str(row["feature"]): row for row in right_rows}
    out = []
    for left in left_rows:
        feature_name = str(left["feature"])
        right = right_map.get(feature_name, {})
        delta = float(left["total_contribution"]) - float(right.get("total_contribution", 0.0))
        out.append(
            {
                "feature": feature_name,
                "family": str(left["family"]),
                "top1_contribution": float(left["total_contribution"]),
                "top2_contribution": float(right.get("total_contribution", 0.0)),
                "delta": delta,
                "advantage": delta,
                "raw_weight": float(left["raw_weight"]),
                "rank_weight": float(left["rank_weight"]),
            }
        )
    out.sort(key=lambda row: abs(float(row["delta"])), reverse=True)
    return out


def _family_delta_rows(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    right_map = {str(row["family"]): row for row in right_rows}
    out = []
    for left in left_rows:
        family = str(left["family"])
        right = right_map.get(family, {})
        delta = float(left["total_contribution"]) - float(right.get("total_contribution", 0.0))
        out.append(
            {
                "family": family,
                "top1_contribution": float(left["total_contribution"]),
                "top2_contribution": float(right.get("total_contribution", 0.0)),
                "delta": delta,
                "advantage": delta,
            }
        )
    out.sort(key=lambda row: abs(float(row["delta"])), reverse=True)
    return out


def _top_runs(rows: Sequence[Mapping[str, Any]], order: np.ndarray, top_n: int = 3) -> list[dict[str, Any]]:
    out = []
    for rank_pos, row_idx in enumerate(order[: min(len(order), int(top_n))].tolist()):
        row = dict(rows[int(row_idx)])
        next_score = None
        if rank_pos + 1 < min(len(order), int(top_n)):
            next_score = float(rows[int(order[rank_pos + 1])]["score"])
        row["rank"] = int(rank_pos + 1)
        row["correctness_mark"] = "✓" if bool(row.get("is_correct", False)) else "✗"
        row["margin_vs_next"] = None if next_score is None else float(float(row["score"]) - next_score)
        out.append(row)
    return out


def _score_stats(scores: np.ndarray) -> dict[str, float]:
    finite = np.asarray(scores, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
    }


def _why_top1(
    top_feature_deltas: Sequence[Mapping[str, Any]],
    top_family_deltas: Sequence[Mapping[str, Any]],
    margin: float,
) -> str:
    family_bits = [str(row["family"]) for row in top_family_deltas if float(row["delta"]) > 0][:2]
    feature_bits = [str(row["feature"]) for row in top_feature_deltas if float(row["delta"]) > 0][:2]
    family_text = " / ".join(family_bits) if family_bits else "整体贡献"
    feature_text = " / ".join(feature_bits) if feature_bits else "特征差异"
    return (
        f"Top1 在当前 anchor 上以 margin={margin:.4f} 领先，"
        f"主要赢在 {family_text} family；关键特征差异来自 {feature_text}。"
    )


def model_summary_from_bundle(
    bundle: Mapping[str, Any],
    *,
    method_id: str,
    domain: str,
    anchor: float | int | str,
    top_k: int = 8,
) -> dict[str, Any]:
    anchor_value = normalize_anchor(anchor)
    route = get_anchor_route(bundle, domain, anchor_value)
    route_terms = _route_effective_linear_terms(route)
    feature_rows = _feature_weight_rows(route_terms)
    family_rows = _family_strength_rows(feature_rows)
    sorted_by_signed = sorted(feature_rows, key=lambda row: float(row["signed_weight"]))
    route_meta = {
        "route_type": str(route.get("route_type", "svd")),
        "representation": str(route.get("representation", "raw")),
        "rank": int(route.get("rank", 0)) if route.get("rank") is not None else None,
        "whiten": bool(route.get("whiten", False)),
        "class_weight": route.get("class_weight"),
        "training_position": float(route.get("training_position", anchor_value)),
        "feature_names": list(_canonical_feature_names_for_route(route)),
        "baseline_signal_name": route.get("baseline_signal_name"),
        "baseline_cv_auroc": route.get("baseline_cv_auroc"),
        "cv_auroc": route.get("cv_auroc"),
    }
    return {
        "success": True,
        "method_id": str(method_id),
        "domain": str(domain),
        "anchor": float(anchor_value),
        "anchor_pct": int(round(anchor_value * 100.0)),
        "route_meta": route_meta,
        "family_strengths": family_rows,
        "top_positive_features": list(reversed(sorted_by_signed[-int(top_k) :])),
        "top_negative_features": sorted_by_signed[: int(top_k)],
        "all_feature_weights": feature_rows,
    }


def explain_problem_from_anchor_tensor(
    bundle: Mapping[str, Any],
    *,
    method_id: str,
    domain: str,
    anchor: float | int | str,
    anchor_tensor: np.ndarray,
    run_infos: Optional[Sequence[Mapping[str, Any]]] = None,
    problem_id: Optional[str] = None,
    cache_key: Optional[str] = None,
) -> dict[str, Any]:
    anchor_value = normalize_anchor(anchor)
    if anchor_tensor.ndim != 3 or anchor_tensor.shape[1] != len(EXPLAIN_ANCHORS):
        raise ValueError("anchor_tensor must have shape [n_runs, 4, n_features]")

    if anchor_tensor.shape[0] == 0:
        return {
            "success": True,
            "method_id": str(method_id),
            "domain": str(domain),
            "anchor": float(anchor_value),
            "anchor_pct": int(round(anchor_value * 100.0)),
            "problem_id": None if problem_id is None else str(problem_id),
            "cache_key": cache_key,
            "route_meta": {},
            "top_runs": [],
            "selected_sample_id": None,
            "selected_run_index": None,
            "why_selected": "No runs available for this problem.",
            "group_context": {"runs": [], "n_runs": 0, "head_count": 0},
            "compare_bars": {"top1_vs_top2": [], "top1_vs_median": []},
            "top_feature_deltas": [],
            "top_family_deltas": [],
            "score_stats": {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0},
            "run_explanations": [],
            "model_summary": {"family_strengths": [], "top_positive_features": [], "top_negative_features": []},
            "anchor_scores": np.zeros((0, len(EXPLAIN_ANCHORS)), dtype=np.float64),
            "anchor_positions": [float(v) for v in EXPLAIN_ANCHORS],
            "sanity_check": {"max_abs_error": 0.0, "mean_abs_error": 0.0},
        }

    route = get_anchor_route(bundle, domain, anchor_value)
    route_terms = _route_effective_linear_terms(route)
    feature_weight_rows = _feature_weight_rows(route_terms)
    family_strength_rows = _family_strength_rows(feature_weight_rows)

    anchor_scores_list = []
    for anchor_idx, anchor_pos in enumerate(EXPLAIN_ANCHORS):
        local_route = get_anchor_route(bundle, domain, anchor_pos)
        x_raw_local = np.asarray(anchor_tensor[:, anchor_idx, :], dtype=np.float64)
        x_rank_local = _rank_transform_matrix(x_raw_local)
        local_scores, _ = _score_route_matrix(local_route, x_raw_local, x_rank_local)
        anchor_scores_list.append(np.asarray(local_scores, dtype=np.float64))
    anchor_scores = np.stack(anchor_scores_list, axis=1)

    active_anchor_idx = EXPLAIN_ANCHORS.index(anchor_value)
    x_raw = np.asarray(anchor_tensor[:, active_anchor_idx, :], dtype=np.float64)
    x_rank = _rank_transform_matrix(x_raw)
    scores, x_rep = _score_route_matrix(route, x_raw, x_rank)

    rep_defs = list(route_terms["representation_features"])
    weights_rep = np.asarray(route_terms["weights_rep"], dtype=np.float64)
    canonical_names = [str(v) for v in route_terms["feature_names"]]
    feature_indices = [int(v) for v in route_terms["feature_indices"]]

    full_feature_values = x_raw[:, feature_indices]
    full_rank_values = x_rank[:, feature_indices]
    feature_rows_per_run: list[list[dict[str, Any]]] = []
    family_rows_per_run: list[list[dict[str, Any]]] = []
    reconstruction_errors = []

    for row_idx in range(x_raw.shape[0]):
        feature_rows = []
        for feat_local_idx, feat_name in enumerate(canonical_names):
            raw_weight = 0.0
            rank_weight = 0.0
            for rep_idx, rep_def in enumerate(rep_defs):
                if rep_def.base_feature != feat_name:
                    continue
                if rep_def.source == "rank":
                    rank_weight += float(weights_rep[rep_idx])
                else:
                    raw_weight += float(weights_rep[rep_idx])
            raw_value = float(full_feature_values[row_idx, feat_local_idx]) if feat_local_idx < full_feature_values.shape[1] else 0.0
            rank_value = float(full_rank_values[row_idx, feat_local_idx]) if feat_local_idx < full_rank_values.shape[1] else 0.0
            raw_contribution = raw_value * raw_weight
            rank_contribution = rank_value * rank_weight
            feature_rows.append(
                {
                    "feature": feat_name,
                    "family": feature_family(feat_name),
                    "raw_value": raw_value,
                    "rank_value": rank_value,
                    "raw_weight": float(raw_weight),
                    "rank_weight": float(rank_weight),
                    "raw_contribution": float(raw_contribution),
                    "rank_contribution": float(rank_contribution),
                    "total_contribution": float(raw_contribution + rank_contribution),
                }
            )
        feature_rows.sort(key=lambda row: abs(float(row["total_contribution"])), reverse=True)
        family_rows = _family_contribution_rows(feature_rows)
        reconstructed = float(route_terms["intercept_effective"] + sum(float(row["total_contribution"]) for row in feature_rows))
        reconstruction_error = float(abs(reconstructed - float(scores[row_idx])))
        reconstruction_errors.append(reconstruction_error)
        feature_rows_per_run.append(feature_rows)
        family_rows_per_run.append(family_rows)

    run_infos_list = list(run_infos or [])
    ranking_rows, order = _ranking_rows(scores, run_infos_list)
    top_runs = _top_runs(ranking_rows, order, top_n=3)

    top1_idx = int(order[0]) if order.size else 0
    top2_idx = int(order[1]) if order.size > 1 else top1_idx
    feature_delta_rows = _feature_delta_rows(feature_rows_per_run[top1_idx], feature_rows_per_run[top2_idx])
    family_delta_rows = _family_delta_rows(family_rows_per_run[top1_idx], family_rows_per_run[top2_idx])
    margin = 0.0 if top1_idx == top2_idx else float(scores[top1_idx] - scores[top2_idx])

    run_explanations = []
    for row_idx, ranking_row in enumerate(ranking_rows):
        run_explanations.append(
            {
                "run": ranking_row,
                "score": float(scores[row_idx]),
                "anchor_scores": {
                    str(int(round(anchor_pos * 100.0))): float(anchor_scores[row_idx, anchor_idx])
                    for anchor_idx, anchor_pos in enumerate(EXPLAIN_ANCHORS)
                },
                "intercept_effective": float(route_terms["intercept_effective"]),
                "feature_contributions": feature_rows_per_run[row_idx],
                "family_contributions": family_rows_per_run[row_idx],
                "score_reconstructed": float(route_terms["intercept_effective"] + sum(float(row["total_contribution"]) for row in feature_rows_per_run[row_idx])),
                "reconstruction_error": float(reconstruction_errors[row_idx]),
            }
        )

    median_feature_rows = []
    if ranking_rows:
        median_idx = int(order[len(order) // 2])
        median_feature_rows = _feature_delta_rows(feature_rows_per_run[top1_idx], feature_rows_per_run[median_idx])

    route_meta = {
        "route_type": str(route.get("route_type", "svd")),
        "representation": str(route.get("representation", "raw")),
        "rank": int(route.get("rank", 0)) if route.get("rank") is not None else None,
        "whiten": bool(route.get("whiten", False)),
        "class_weight": route.get("class_weight"),
        "training_position": float(route.get("training_position", anchor_value)),
        "feature_names": canonical_names,
        "baseline_signal_name": route.get("baseline_signal_name"),
        "baseline_cv_auroc": route.get("baseline_cv_auroc"),
        "cv_auroc": route.get("cv_auroc"),
    }
    model_summary = {
        "family_strengths": family_strength_rows,
        "top_positive_features": sorted(feature_weight_rows, key=lambda row: float(row["signed_weight"]), reverse=True)[:8],
        "top_negative_features": sorted(feature_weight_rows, key=lambda row: float(row["signed_weight"]))[:8],
        "feature_weights": feature_weight_rows,
    }

    return {
        "success": True,
        "method_id": str(method_id),
        "domain": str(domain),
        "anchor": float(anchor_value),
        "anchor_pct": int(round(anchor_value * 100.0)),
        "problem_id": None if problem_id is None else str(problem_id),
        "cache_key": cache_key,
        "route_meta": route_meta,
        "top_runs": top_runs,
        "selected_sample_id": int(top_runs[0]["sample_id"]) if top_runs else None,
        "selected_run_index": int(top_runs[0]["run_index"]) if top_runs else None,
        "why_selected": _why_top1(feature_delta_rows[:3], family_delta_rows[:3], margin),
        "group_context": {
            "runs": ranking_rows,
            "n_runs": int(len(ranking_rows)),
            "head_count": max(1, int(np.ceil(0.10 * max(1, len(ranking_rows))))) if ranking_rows else 0,
        },
        "compare_bars": {
            "top1_vs_top2": feature_delta_rows[:12],
            "top1_vs_median": median_feature_rows[:12],
        },
        "top_feature_deltas": feature_delta_rows,
        "top_family_deltas": family_delta_rows,
        "score_stats": _score_stats(scores),
        "run_explanations": run_explanations,
        "model_summary": model_summary,
        "anchor_scores": anchor_scores,
        "anchor_positions": [float(v) for v in EXPLAIN_ANCHORS],
        "sanity_check": {
            "max_abs_error": float(max(reconstruction_errors) if reconstruction_errors else 0.0),
            "mean_abs_error": float(np.mean(reconstruction_errors) if reconstruction_errors else 0.0),
        },
    }


def explain_problem_from_reader(
    bundle: Mapping[str, Any],
    *,
    method_id: str,
    domain: str,
    anchor: float | int | str,
    reader: CacheReader,
    run_ids: Sequence[int],
    run_infos: Optional[Sequence[Mapping[str, Any]]] = None,
    problem_id: Optional[str] = None,
    cache_key: Optional[str] = None,
    required_features: Optional[set[str]] = None,
    reflection_threshold: float = DEFAULT_REFLECTION_THRESHOLD,
) -> dict[str, Any]:
    required = collect_bundle_required_features(bundle) if required_features is None else set(required_features)
    anchor_tensor = build_problem_anchor_tensor(
        reader,
        run_ids,
        positions=EXPLAIN_ANCHORS,
        required_features=required,
        reflection_threshold=float(reflection_threshold),
    )
    return explain_problem_from_anchor_tensor(
        bundle,
        method_id=method_id,
        domain=domain,
        anchor=anchor,
        anchor_tensor=anchor_tensor,
        run_infos=run_infos,
        problem_id=problem_id,
        cache_key=cache_key,
    )


def summarize_wrong_top1_case(problem_payload: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    run_explanations = list(problem_payload.get("run_explanations", []))
    if not run_explanations:
        return None

    top1 = run_explanations[0]
    top1_run = dict(top1.get("run", {}))
    if bool(top1_run.get("is_correct", False)):
        return None

    correct_candidates = [row for row in run_explanations if bool(row.get("run", {}).get("is_correct", False))]
    if not correct_candidates:
        return None
    best_correct = max(correct_candidates, key=lambda row: float(row.get("score", float("-inf"))))

    feature_deltas = _feature_delta_rows(
        top1.get("feature_contributions", []),
        best_correct.get("feature_contributions", []),
    )
    family_deltas = _family_delta_rows(
        top1.get("family_contributions", []),
        best_correct.get("family_contributions", []),
    )
    return {
        "problem_id": problem_payload.get("problem_id"),
        "cache_key": problem_payload.get("cache_key"),
        "method_id": problem_payload.get("method_id"),
        "domain": problem_payload.get("domain"),
        "anchor": problem_payload.get("anchor"),
        "anchor_pct": problem_payload.get("anchor_pct"),
        "predicted_top1": top1_run,
        "best_correct_run": dict(best_correct.get("run", {})),
        "predicted_top1_score": float(top1.get("score", 0.0)),
        "best_correct_score": float(best_correct.get("score", 0.0)),
        "score_gap": float(float(top1.get("score", 0.0)) - float(best_correct.get("score", 0.0))),
        "top_feature_deltas": feature_deltas[:12],
        "top_family_deltas": family_deltas[:8],
    }


def aggregate_failure_modes(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    family_stats: dict[str, dict[str, Any]] = {}
    feature_stats: dict[str, dict[str, Any]] = {}
    total_cases = 0

    for case in cases:
        family_rows = [row for row in case.get("top_family_deltas", []) if float(row.get("delta", 0.0)) > 0]
        feature_rows = [row for row in case.get("top_feature_deltas", []) if float(row.get("delta", 0.0)) > 0]
        if not family_rows and not feature_rows:
            continue
        total_cases += 1

        for idx, row in enumerate(family_rows[:3]):
            family = str(row["family"])
            bucket = family_stats.setdefault(
                family,
                {
                    "family": family,
                    "count_as_top_driver": 0,
                    "count_in_top3": 0,
                    "deltas": [],
                },
            )
            if idx == 0:
                bucket["count_as_top_driver"] += 1
            bucket["count_in_top3"] += 1
            bucket["deltas"].append(float(row["delta"]))

        for idx, row in enumerate(feature_rows[:3]):
            feature_name = str(row["feature"])
            bucket = feature_stats.setdefault(
                feature_name,
                {
                    "feature": feature_name,
                    "family": str(row["family"]),
                    "count_as_top_driver": 0,
                    "count_in_top3": 0,
                    "deltas": [],
                },
            )
            if idx == 0:
                bucket["count_as_top_driver"] += 1
            bucket["count_in_top3"] += 1
            bucket["deltas"].append(float(row["delta"]))

    def _finalize(rows: dict[str, dict[str, Any]], key_name: str) -> list[dict[str, Any]]:
        out = []
        for bucket in rows.values():
            deltas = np.asarray(bucket.pop("deltas", []), dtype=np.float64)
            bucket["mean_delta"] = float(np.mean(deltas)) if deltas.size else 0.0
            bucket["median_delta"] = float(np.median(deltas)) if deltas.size else 0.0
            bucket["max_delta"] = float(np.max(deltas)) if deltas.size else 0.0
            out.append(bucket)
        out.sort(
            key=lambda row: (
                -int(row["count_as_top_driver"]),
                -int(row["count_in_top3"]),
                -float(row["mean_delta"]),
                str(row[key_name]),
            )
        )
        return out

    return {
        "total_wrong_top1_cases": int(total_cases),
        "families": _finalize(family_stats, "family"),
        "features": _finalize(feature_stats, "feature"),
    }
