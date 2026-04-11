from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from nad.explain.svd_explain import (
    EXPLAIN_ANCHORS,
    _family_delta_rows,
    _feature_delta_rows,
    _ranking_rows,
    _score_route_matrix,
    feature_family,
    get_anchor_route,
    model_summary_from_bundle,
)
from nad.explain.svd_introspection import explain_run
from nad.ops.earlystop_svd import FULL_FEATURE_NAMES, _rank_transform_matrix, load_earlystop_svd_bundle


_EPS = 1e-12
_FULL_FEATURE_TO_INDEX = {name: idx for idx, name in enumerate(FULL_FEATURE_NAMES)}


@dataclass(frozen=True)
class BundleRecord:
    record_id: str
    method_group: str
    bundle_path: Path
    axis: str
    random_state: int
    split_seed: int
    note: str = ""


def load_bundle_records(records: Sequence[BundleRecord]) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    for record in records:
        bundle = load_earlystop_svd_bundle(Path(record.bundle_path))
        loaded.append(
            {
                "record_id": str(record.record_id),
                "method_group": str(record.method_group),
                "bundle_path": str(record.bundle_path),
                "axis": str(record.axis),
                "random_state": int(record.random_state),
                "split_seed": int(record.split_seed),
                "note": str(record.note),
                "bundle": bundle,
            }
        )
    return loaded


def _safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _safe_std(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr)) if arr.size else float("nan")


def _safe_median(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _ci95(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(1.96 * np.std(arr) / math.sqrt(arr.size))


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = {str(v) for v in left}
    right_set = {str(v) for v in right}
    union = left_set | right_set
    if not union:
        return 1.0
    return float(len(left_set & right_set) / len(union))


def _overlap_count(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = {str(v) for v in left}
    right_set = {str(v) for v in right}
    return float(len(left_set & right_set))


def _descending_rank_map(values: Mapping[str, float]) -> dict[str, int]:
    ordered = sorted(
        ((str(k), float(v)) for k, v in values.items()),
        key=lambda kv: (-kv[1], kv[0]),
    )
    return {name: idx + 1 for idx, (name, _) in enumerate(ordered)}


def _pearson_corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size != right.size or left.size <= 1:
        return float("nan")
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - np.mean(left)
    right = right - np.mean(right)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= _EPS:
        return float("nan")
    return float(np.dot(left, right) / denom)


def _spearman_rho(values_left: Mapping[str, float], values_right: Mapping[str, float]) -> float:
    keys = sorted(set(values_left) | set(values_right))
    if len(keys) <= 1:
        return float("nan")
    rank_left = _descending_rank_map({key: float(values_left.get(key, 0.0)) for key in keys})
    rank_right = _descending_rank_map({key: float(values_right.get(key, 0.0)) for key in keys})
    vec_left = np.asarray([float(rank_left[key]) for key in keys], dtype=np.float64)
    vec_right = np.asarray([float(rank_right[key]) for key in keys], dtype=np.float64)
    return _pearson_corr(vec_left, vec_right)


def _kendall_tau(values_left: Mapping[str, float], values_right: Mapping[str, float]) -> float:
    keys = sorted(set(values_left) | set(values_right))
    if len(keys) <= 1:
        return float("nan")
    concordant = 0
    discordant = 0
    for left_idx in range(len(keys)):
        for right_idx in range(left_idx + 1, len(keys)):
            key_i = keys[left_idx]
            key_j = keys[right_idx]
            delta_left = float(values_left.get(key_i, 0.0)) - float(values_left.get(key_j, 0.0))
            delta_right = float(values_right.get(key_i, 0.0)) - float(values_right.get(key_j, 0.0))
            if abs(delta_left) <= _EPS or abs(delta_right) <= _EPS:
                continue
            if delta_left * delta_right > 0:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    if total <= 0:
        return float("nan")
    return float((concordant - discordant) / total)


def _pairwise_sign_consistency(
    values_left: Mapping[str, float],
    values_right: Mapping[str, float],
) -> float:
    keys = sorted(set(values_left) | set(values_right))
    if not keys:
        return float("nan")
    matches: list[float] = []
    for key in keys:
        left_val = float(values_left.get(key, 0.0))
        right_val = float(values_right.get(key, 0.0))
        if abs(left_val) <= _EPS or abs(right_val) <= _EPS:
            matches.append(1.0)
            continue
        matches.append(1.0 if math.copysign(1.0, left_val) == math.copysign(1.0, right_val) else 0.0)
    return float(np.mean(matches)) if matches else float("nan")


def _mean_abs_weight_cv(weight_maps: Sequence[Mapping[str, float]]) -> float:
    keys = sorted({str(key) for weight_map in weight_maps for key in weight_map.keys()})
    if not keys:
        return float("nan")
    rows = []
    for key in keys:
        arr = np.asarray([abs(float(weight_map.get(key, 0.0))) for weight_map in weight_maps], dtype=np.float64)
        mean_val = float(np.mean(arr))
        if mean_val <= _EPS:
            continue
        rows.append(float(np.std(arr) / mean_val))
    return float(np.mean(rows)) if rows else float("nan")


def _family_channel_maps(feature_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    out = {
        "raw": {},
        "rank": {},
        "raw+rank": {},
    }
    for row in feature_rows:
        family = str(row["family"])
        raw_weight = float(row["raw_weight"])
        rank_weight = float(row["rank_weight"])
        out["raw"][family] = out["raw"].get(family, 0.0) + raw_weight
        out["rank"][family] = out["rank"].get(family, 0.0) + rank_weight
        out["raw+rank"][family] = out["raw+rank"].get(family, 0.0) + raw_weight + rank_weight
    return out


def route_signature_from_bundle(
    bundle: Mapping[str, Any],
    *,
    method_group: str,
    domain: str,
    anchor: float,
    top_k: int,
) -> dict[str, Any]:
    summary = model_summary_from_bundle(
        bundle,
        method_id=method_group,
        domain=domain,
        anchor=anchor,
        top_k=top_k,
    )
    feature_rows = list(summary["all_feature_weights"])
    family_rows = list(summary["family_strengths"])
    top_positive_features = [str(row["feature"]) for row in summary["top_positive_features"][:top_k]]
    top_negative_features = [str(row["feature"]) for row in summary["top_negative_features"][:top_k]]
    feature_weight_maps = {
        "raw": {str(row["feature"]): float(row["raw_weight"]) for row in feature_rows},
        "rank": {str(row["feature"]): float(row["rank_weight"]) for row in feature_rows},
        "raw+rank": {str(row["feature"]): float(row["signed_weight"]) for row in feature_rows},
    }
    family_strengths = {str(row["family"]): float(row["strength"]) for row in family_rows}
    family_channel_maps = _family_channel_maps(feature_rows)
    return {
        "method_group": str(method_group),
        "domain": str(domain),
        "anchor_pct": int(round(float(anchor) * 100.0)),
        "top_positive_features": top_positive_features,
        "top_negative_features": top_negative_features,
        "feature_weight_maps": feature_weight_maps,
        "family_strengths": family_strengths,
        "family_channel_maps": family_channel_maps,
    }


def compute_stability_rows(
    loaded_records: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pairwise_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    raw_rank_rows: list[dict[str, Any]] = []

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in loaded_records:
        key = (str(record["axis"]), str(record["method_group"]))
        groups.setdefault(key, []).append(record)

    for (axis, method_group), records in sorted(groups.items()):
        if len(records) < 2:
            continue

        shared_domains = set(str(v) for v in records[0]["bundle"].get("domains", {}).keys())
        for record in records[1:]:
            shared_domains &= set(str(v) for v in record["bundle"].get("domains", {}).keys())
        for domain in sorted(shared_domains):
            for anchor in EXPLAIN_ANCHORS:
                route_signatures = []
                for record in records:
                    route_signatures.append(
                        {
                            "record_id": str(record["record_id"]),
                            "signature": route_signature_from_bundle(
                                record["bundle"],
                                method_group=method_group,
                                domain=domain,
                                anchor=float(anchor),
                                top_k=int(top_k),
                            ),
                        }
                    )

                for left_idx in range(len(route_signatures)):
                    for right_idx in range(left_idx + 1, len(route_signatures)):
                        left = route_signatures[left_idx]
                        right = route_signatures[right_idx]
                        left_sig = left["signature"]
                        right_sig = right["signature"]
                        anchor_pct = int(left_sig["anchor_pct"])

                        metric_rows = [
                            ("top_positive_jaccard", "feature", "raw+rank", _jaccard(left_sig["top_positive_features"], right_sig["top_positive_features"])),
                            ("top_positive_overlap_count", "feature", "raw+rank", _overlap_count(left_sig["top_positive_features"], right_sig["top_positive_features"])),
                            ("top_negative_jaccard", "feature", "raw+rank", _jaccard(left_sig["top_negative_features"], right_sig["top_negative_features"])),
                            ("top_negative_overlap_count", "feature", "raw+rank", _overlap_count(left_sig["top_negative_features"], right_sig["top_negative_features"])),
                            ("family_rank_spearman", "family", "raw+rank", _spearman_rho(left_sig["family_strengths"], right_sig["family_strengths"])),
                            ("family_rank_kendall", "family", "raw+rank", _kendall_tau(left_sig["family_strengths"], right_sig["family_strengths"])),
                        ]

                        for channel in ("raw", "rank", "raw+rank"):
                            metric_rows.append(
                                (
                                    "feature_sign_consistency",
                                    "feature",
                                    channel,
                                    _pairwise_sign_consistency(
                                        left_sig["feature_weight_maps"][channel],
                                        right_sig["feature_weight_maps"][channel],
                                    ),
                                )
                            )
                            metric_rows.append(
                                (
                                    "family_sign_consistency",
                                    "family",
                                    channel,
                                    _pairwise_sign_consistency(
                                        left_sig["family_channel_maps"][channel],
                                        right_sig["family_channel_maps"][channel],
                                    ),
                                )
                            )

                        for metric_name, target, channel, metric_value in metric_rows:
                            pairwise_rows.append(
                                {
                                    "axis": axis,
                                    "method_group": method_group,
                                    "domain": domain,
                                    "anchor_pct": anchor_pct,
                                    "metric": metric_name,
                                    "target": target,
                                    "channel": channel,
                                    "value": float(metric_value),
                                    "left_record_id": str(left["record_id"]),
                                    "right_record_id": str(right["record_id"]),
                                    "top_k": int(top_k),
                                }
                            )

                metric_groups: dict[tuple[str, str, str], list[float]] = {}
                for row in pairwise_rows:
                    if (
                        row["axis"] != axis
                        or row["method_group"] != method_group
                        or row["domain"] != domain
                        or int(row["anchor_pct"]) != int(round(float(anchor) * 100.0))
                    ):
                        continue
                    metric_key = (str(row["metric"]), str(row["target"]), str(row["channel"]))
                    metric_groups.setdefault(metric_key, []).append(float(row["value"]))

                for (metric_name, target, channel), values in sorted(metric_groups.items()):
                    summary_rows.append(
                        {
                            "axis": axis,
                            "method_group": method_group,
                            "domain": domain,
                            "anchor_pct": int(round(float(anchor) * 100.0)),
                            "metric": metric_name,
                            "target": target,
                            "channel": channel,
                            "mean_value": _safe_mean(values),
                            "std_value": _safe_std(values),
                            "median_value": _safe_median(values),
                            "ci95": _ci95(values),
                            "min_value": float(np.min(values)) if values else float("nan"),
                            "max_value": float(np.max(values)) if values else float("nan"),
                            "n_pairs": int(len(values)),
                            "n_models": int(len(records)),
                            "top_k": int(top_k),
                        }
                    )

                for channel in ("raw", "rank", "raw+rank"):
                    feature_weight_maps = [
                        route_signature["signature"]["feature_weight_maps"][channel]
                        for route_signature in route_signatures
                    ]
                    raw_rank_rows.append(
                        {
                            "axis": axis,
                            "method_group": method_group,
                            "domain": domain,
                            "anchor_pct": int(round(float(anchor) * 100.0)),
                            "channel": channel,
                            "metric": "abs_weight_cv",
                            "value": _mean_abs_weight_cv(feature_weight_maps),
                            "n_models": int(len(records)),
                        }
                    )

    return pairwise_rows, summary_rows, raw_rank_rows


def summarize_raw_vs_rank_stability(
    stability_summary_rows: Sequence[Mapping[str, Any]],
    raw_rank_cv_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for axis in sorted({str(row["axis"]) for row in stability_summary_rows}):
        for channel in ("raw", "rank"):
            sign_rows = [
                row
                for row in stability_summary_rows
                if str(row["axis"]) == axis
                and str(row["metric"]) == "feature_sign_consistency"
                and str(row["channel"]) == channel
            ]
            cv_rows = [
                row
                for row in raw_rank_cv_rows
                if str(row["axis"]) == axis
                and str(row["channel"]) == channel
                and str(row["metric"]) == "abs_weight_cv"
            ]
            rows.append(
                {
                    "axis": axis,
                    "channel": channel,
                    "mean_feature_sign_consistency": _safe_mean([float(row["mean_value"]) for row in sign_rows]),
                    "median_feature_sign_consistency": _safe_median([float(row["mean_value"]) for row in sign_rows]),
                    "mean_abs_weight_cv": _safe_mean([float(row["value"]) for row in cv_rows]),
                    "median_abs_weight_cv": _safe_median([float(row["value"]) for row in cv_rows]),
                    "n_anchor_rows": int(len(sign_rows)),
                }
            )
    return rows


def iter_problem_slices(
    feature_store: Sequence[Mapping[str, Any]],
    run_index_maps: Mapping[str, Mapping[int, int]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for payload in feature_store:
        cache_key = str(payload["cache_key"])
        sample_to_run_index = run_index_maps.get(cache_key, {})
        tensor = np.asarray(payload["tensor"], dtype=np.float64)
        labels = np.asarray(payload["labels"], dtype=np.int32)
        sample_ids = np.asarray(payload["sample_ids"], dtype=np.int32)
        offsets = [int(v) for v in payload["problem_offsets"]]
        problem_ids = [str(v) for v in payload["problem_ids"]]
        for problem_idx, problem_id in enumerate(problem_ids):
            start = offsets[problem_idx]
            end = offsets[problem_idx + 1]
            local_sample_ids = sample_ids[start:end]
            run_infos = [
                {
                    "sample_id": int(sample_id),
                    "run_index": int(sample_to_run_index.get(int(sample_id), int(sample_id))),
                    "is_correct": bool(labels[start + row_idx]),
                }
                for row_idx, sample_id in enumerate(local_sample_ids.tolist())
            ]
            run_infos.sort(key=lambda row: int(row["run_index"]))
            sorted_sample_ids = [int(row["sample_id"]) for row in run_infos]
            order = [int(np.where(local_sample_ids == sample_id)[0][0]) for sample_id in sorted_sample_ids]
            out.append(
                {
                    "cache_key": cache_key,
                    "base_cache_key": str(payload.get("base_cache_key", cache_key)),
                    "dataset_name": str(payload["dataset_name"]),
                    "domain": str(payload["domain"]),
                    "problem_id": str(problem_id),
                    "anchor_tensor": tensor[start:end][order],
                    "labels": labels[start:end][order],
                    "sample_ids": np.asarray(sorted_sample_ids, dtype=np.int32),
                    "run_infos": run_infos,
                }
            )
    return out


def _select_target_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
    target_key: str,
) -> dict[str, Optional[Mapping[str, Any]]]:
    rows_list = list(rows)
    positive_rows = [row for row in rows_list if float(row.get(key, 0.0)) > 0]
    positive_rows = sorted(positive_rows, key=lambda row: float(row.get(key, 0.0)), reverse=True)

    if positive_rows:
        top_row = positive_rows[0]
    else:
        fallback_rows = sorted(rows_list, key=lambda row: abs(float(row.get(key, 0.0))), reverse=True)
        top_row = fallback_rows[0] if fallback_rows else None

    remaining_rows = []
    top_name = None if top_row is None else str(top_row.get(target_key, ""))
    for row in rows_list:
        row_name = str(row.get(target_key, ""))
        if top_name is not None and row_name == top_name:
            continue
        remaining_rows.append(row)

    low_row = min(
        remaining_rows,
        key=lambda row: abs(float(row.get(key, 0.0))),
        default=None,
    )
    random_candidates = sorted(
        remaining_rows,
        key=lambda row: abs(float(row.get(key, 0.0))),
    )
    random_row = random_candidates[len(random_candidates) // 2] if random_candidates else low_row
    return {
        "top": top_row,
        "low": low_row,
        "random": random_row,
    }


def _mutate_problem_row(
    x_raw_problem: np.ndarray,
    *,
    row_idx: int,
    feature_names: Sequence[str],
) -> np.ndarray:
    mutated = np.asarray(x_raw_problem, dtype=np.float64).copy()
    for feature_name in feature_names:
        feat_idx = _FULL_FEATURE_TO_INDEX.get(str(feature_name))
        if feat_idx is None:
            continue
        mutated[int(row_idx), int(feat_idx)] = 0.0
    return mutated


def _run_intervention(
    route: Mapping[str, Any],
    *,
    method_id: str,
    domain: str,
    anchor_pct: int,
    cache_key: str,
    problem_id: str,
    x_raw_problem: np.ndarray,
    labels: np.ndarray,
    run_infos: Sequence[Mapping[str, Any]],
    top1_idx: int,
    top1_score: float,
    top1_margin: float,
    target_level: str,
    target_policy: str,
    target_name: str,
    target_features: Sequence[str],
    target_contribution: float,
) -> dict[str, Any]:
    x_raw_new = _mutate_problem_row(
        x_raw_problem,
        row_idx=int(top1_idx),
        feature_names=target_features,
    )
    x_rank_new = _rank_transform_matrix(x_raw_new) if x_raw_new.shape[0] > 0 else np.zeros_like(x_raw_new)
    scores_new, _ = _score_route_matrix(route, x_raw_new, x_rank_new)
    ranking_rows_new, order_new = _ranking_rows(scores_new, run_infos)
    new_top1_idx = int(order_new[0]) if order_new.size else int(top1_idx)
    same_run_score_new = float(scores_new[int(top1_idx)])
    best_other_new = float(np.max(np.delete(scores_new, int(top1_idx)))) if scores_new.size > 1 else same_run_score_new
    same_run_margin_new = float(same_run_score_new - best_other_new)
    new_rank = int(ranking_rows_new[int(top1_idx)]["rank"]) if ranking_rows_new else 1

    return {
        "method_id": str(method_id),
        "domain": str(domain),
        "anchor_pct": int(anchor_pct),
        "cache_key": str(cache_key),
        "problem_id": str(problem_id),
        "target_level": str(target_level),
        "target_policy": str(target_policy),
        "target_name": str(target_name),
        "target_features": ";".join(sorted(set(str(v) for v in target_features))),
        "target_feature_count": int(len(set(str(v) for v in target_features))),
        "target_contribution": float(target_contribution),
        "selected_sample_id": int(run_infos[int(top1_idx)]["sample_id"]),
        "selected_run_index": int(run_infos[int(top1_idx)]["run_index"]),
        "selected_is_correct": bool(labels[int(top1_idx)]),
        "original_score": float(top1_score),
        "intervened_same_run_score": float(same_run_score_new),
        "score_drop": float(top1_score - same_run_score_new),
        "original_margin": float(top1_margin),
        "intervened_same_run_margin": float(same_run_margin_new),
        "margin_drop": float(top1_margin - same_run_margin_new),
        "selection_flip": bool(new_top1_idx != int(top1_idx)),
        "selected_new_rank": int(new_rank),
        "new_selected_sample_id": int(run_infos[int(new_top1_idx)]["sample_id"]),
        "new_selected_run_index": int(run_infos[int(new_top1_idx)]["run_index"]),
        "new_selected_is_correct": bool(labels[int(new_top1_idx)]),
        "accuracy_delta": int(bool(labels[int(new_top1_idx)])) - int(bool(labels[int(top1_idx)])),
    }


def deletion_intervention_rows(
    *,
    method_id: str,
    domain: str,
    anchor_pct: int,
    cache_key: str,
    problem_id: str,
    route: Mapping[str, Any],
    x_raw_problem: np.ndarray,
    labels: np.ndarray,
    run_infos: Sequence[Mapping[str, Any]],
    top1_idx: int,
    top2_idx: int,
    scores: np.ndarray,
    exp_top1: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    top1_score = float(scores[int(top1_idx)])
    top2_score = float(scores[int(top2_idx)]) if int(top2_idx) != int(top1_idx) else float(top1_score)
    top1_margin = float(top1_score - top2_score)

    family_targets = _select_target_rows(
        exp_top1["family_contributions"],
        key="total_contribution",
        target_key="family",
    )
    feature_targets = _select_target_rows(
        exp_top1["canonical_feature_rows"],
        key="total_contribution",
        target_key="feature",
    )

    canonical_feature_rows = list(exp_top1["canonical_feature_rows"])

    for policy, family_row in family_targets.items():
        if family_row is None:
            continue
        family_name = str(family_row["family"])
        feature_names = [
            str(row["feature"])
            for row in canonical_feature_rows
            if str(row["family"]) == family_name
        ]
        if not feature_names:
            continue
        rows.append(
            _run_intervention(
                route,
                method_id=method_id,
                domain=domain,
                anchor_pct=anchor_pct,
                cache_key=cache_key,
                problem_id=problem_id,
                x_raw_problem=x_raw_problem,
                labels=labels,
                run_infos=run_infos,
                top1_idx=top1_idx,
                top1_score=top1_score,
                top1_margin=top1_margin,
                target_level="family",
                target_policy=policy,
                target_name=family_name,
                target_features=feature_names,
                target_contribution=float(family_row["total_contribution"]),
            )
        )

    for policy, feature_row in feature_targets.items():
        if feature_row is None:
            continue
        feature_name = str(feature_row["feature"])
        rows.append(
            _run_intervention(
                route,
                method_id=method_id,
                domain=domain,
                anchor_pct=anchor_pct,
                cache_key=cache_key,
                problem_id=problem_id,
                x_raw_problem=x_raw_problem,
                labels=labels,
                run_infos=run_infos,
                top1_idx=top1_idx,
                top1_score=top1_score,
                top1_margin=top1_margin,
                target_level="feature",
                target_policy=policy,
                target_name=feature_name,
                target_features=[feature_name],
                target_contribution=float(feature_row["total_contribution"]),
            )
        )

    return rows


def aggregate_deletion_rows(detail_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = {}
    for row in detail_rows:
        key = (
            str(row["method_id"]),
            str(row["domain"]),
            str(row["target_level"]),
            int(row["anchor_pct"]),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (method_id, domain, target_level, anchor_pct), rows in sorted(grouped.items()):
        policy_groups: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            policy_groups.setdefault(str(row["target_policy"]), []).append(row)

        mean_score_drop_by_policy = {
            policy: _safe_mean([float(item["score_drop"]) for item in policy_rows])
            for policy, policy_rows in policy_groups.items()
        }
        mean_flip_rate_by_policy = {
            policy: _safe_mean([1.0 if bool(item["selection_flip"]) else 0.0 for item in policy_rows])
            for policy, policy_rows in policy_groups.items()
        }

        for policy, policy_rows in sorted(policy_groups.items()):
            correct_rows = [row for row in policy_rows if bool(row["selected_is_correct"])]
            wrong_rows = [row for row in policy_rows if not bool(row["selected_is_correct"])]
            correct_to_wrong = [
                row
                for row in correct_rows
                if not bool(row["new_selected_is_correct"])
            ]
            wrong_to_correct = [
                row
                for row in wrong_rows
                if bool(row["new_selected_is_correct"])
            ]
            summary_rows.append(
                {
                    "method_id": method_id,
                    "domain": domain,
                    "anchor_pct": int(anchor_pct),
                    "target_level": target_level,
                    "target_policy": policy,
                    "n_cases": int(len(policy_rows)),
                    "mean_target_contribution": _safe_mean([float(row["target_contribution"]) for row in policy_rows]),
                    "mean_score_drop": _safe_mean([float(row["score_drop"]) for row in policy_rows]),
                    "median_score_drop": _safe_median([float(row["score_drop"]) for row in policy_rows]),
                    "mean_margin_drop": _safe_mean([float(row["margin_drop"]) for row in policy_rows]),
                    "selection_flip_rate": _safe_mean([1.0 if bool(row["selection_flip"]) else 0.0 for row in policy_rows]),
                    "correct_to_wrong_rate": (float(len(correct_to_wrong)) / float(len(correct_rows))) if correct_rows else 0.0,
                    "wrong_to_correct_rate": (float(len(wrong_to_correct)) / float(len(wrong_rows))) if wrong_rows else 0.0,
                    "mean_accuracy_delta": _safe_mean([float(row["accuracy_delta"]) for row in policy_rows]),
                    "score_drop_ratio_vs_low": (
                        float(mean_score_drop_by_policy.get("top", float("nan")) / mean_score_drop_by_policy.get("low"))
                        if policy == "top" and abs(float(mean_score_drop_by_policy.get("low", 0.0))) > _EPS
                        else float("nan")
                    ),
                    "score_drop_ratio_vs_random": (
                        float(mean_score_drop_by_policy.get("top", float("nan")) / mean_score_drop_by_policy.get("random"))
                        if policy == "top" and abs(float(mean_score_drop_by_policy.get("random", 0.0))) > _EPS
                        else float("nan")
                    ),
                    "flip_rate_ratio_vs_low": (
                        float(mean_flip_rate_by_policy.get("top", float("nan")) / mean_flip_rate_by_policy.get("low"))
                        if policy == "top" and abs(float(mean_flip_rate_by_policy.get("low", 0.0))) > _EPS
                        else float("nan")
                    ),
                }
            )
    return summary_rows


def build_wrong_case_record(
    *,
    method_id: str,
    domain: str,
    anchor_pct: int,
    cache_key: str,
    problem_id: str,
    run_infos: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    scores: np.ndarray,
    top1_idx: int,
    best_correct_idx: int,
    exp_top1: Mapping[str, Any],
    exp_best_correct: Mapping[str, Any],
) -> dict[str, Any]:
    feature_deltas = _feature_delta_rows(
        exp_top1["canonical_feature_rows"],
        exp_best_correct["canonical_feature_rows"],
    )
    family_deltas = _family_delta_rows(
        exp_top1["family_contributions"],
        exp_best_correct["family_contributions"],
    )

    raw_delta_abs = 0.0
    rank_delta_abs = 0.0
    right_map = {str(row["feature"]): row for row in exp_best_correct["canonical_feature_rows"]}
    for left_row in exp_top1["canonical_feature_rows"]:
        right_row = right_map.get(str(left_row["feature"]), {})
        raw_delta_abs += abs(float(left_row["raw_contribution"]) - float(right_row.get("raw_contribution", 0.0)))
        rank_delta_abs += abs(float(left_row["rank_contribution"]) - float(right_row.get("rank_contribution", 0.0)))
    total_channel_delta_abs = raw_delta_abs + rank_delta_abs
    rank_share = float(rank_delta_abs / total_channel_delta_abs) if total_channel_delta_abs > _EPS else 0.0

    positive_family_rows = [row for row in family_deltas if float(row["delta"]) > 0]
    positive_feature_rows = [row for row in feature_deltas if float(row["delta"]) > 0]
    positive_family_total = sum(float(row["delta"]) for row in positive_family_rows)
    dominant_family = str(positive_family_rows[0]["family"]) if positive_family_rows else "none"
    dominant_family_share = (
        float(float(positive_family_rows[0]["delta"]) / positive_family_total)
        if positive_family_rows and positive_family_total > _EPS
        else 0.0
    )
    dominant_feature = str(positive_feature_rows[0]["feature"]) if positive_feature_rows else "none"

    return {
        "case_type": "wrong",
        "method_id": str(method_id),
        "domain": str(domain),
        "anchor_pct": int(anchor_pct),
        "cache_key": str(cache_key),
        "problem_id": str(problem_id),
        "predicted_sample_id": int(run_infos[int(top1_idx)]["sample_id"]),
        "predicted_run_index": int(run_infos[int(top1_idx)]["run_index"]),
        "predicted_is_correct": bool(labels[int(top1_idx)]),
        "best_correct_sample_id": int(run_infos[int(best_correct_idx)]["sample_id"]),
        "best_correct_run_index": int(run_infos[int(best_correct_idx)]["run_index"]),
        "top1_sample_id": int(run_infos[int(top1_idx)]["sample_id"]),
        "top1_run_index": int(run_infos[int(top1_idx)]["run_index"]),
        "top1_is_correct": bool(labels[int(top1_idx)]),
        "compare_sample_id": int(run_infos[int(best_correct_idx)]["sample_id"]),
        "compare_run_index": int(run_infos[int(best_correct_idx)]["run_index"]),
        "compare_is_correct": bool(labels[int(best_correct_idx)]),
        "predicted_score": float(scores[int(top1_idx)]),
        "best_correct_score": float(scores[int(best_correct_idx)]),
        "top1_score": float(scores[int(top1_idx)]),
        "compare_score": float(scores[int(best_correct_idx)]),
        "score_gap": float(scores[int(top1_idx)] - scores[int(best_correct_idx)]),
        "margin_or_gap": float(scores[int(top1_idx)] - scores[int(best_correct_idx)]),
        "raw_delta_abs": float(raw_delta_abs),
        "rank_delta_abs": float(rank_delta_abs),
        "rank_delta_share": float(rank_share),
        "dominant_positive_family": dominant_family,
        "dominant_positive_family_share": float(dominant_family_share),
        "dominant_positive_feature": dominant_feature,
        "top_family_deltas": positive_family_rows[:5],
        "top_feature_deltas": positive_feature_rows[:8],
        "top1_summary": str(exp_top1["human_summary"]),
        "best_correct_summary": str(exp_best_correct["human_summary"]),
    }


def assign_failure_archetypes(
    wrong_cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not wrong_cases:
        return []

    score_gaps = np.asarray([float(case["score_gap"]) for case in wrong_cases], dtype=np.float64)
    weak_margin_threshold = float(np.quantile(score_gaps, 0.25)) if score_gaps.size else 0.1

    out: list[dict[str, Any]] = []
    for case in wrong_cases:
        dominant_family = str(case["dominant_positive_family"])
        dominant_feature = str(case["dominant_positive_feature"])
        score_gap = float(case["score_gap"])
        anchor_pct = int(case["anchor_pct"])
        rank_share = float(case["rank_delta_share"])

        if score_gap <= weak_margin_threshold:
            archetype_id = "weak_margin_ambiguous_tie"
            label = "Weak-margin ambiguous tie"
        elif dominant_family == "self_cert_logprob":
            archetype_id = "self_cert_overtrust"
            label = "Self-cert/logprob over-trust"
        elif dominant_family == "uncertainty" and anchor_pct >= 70:
            archetype_id = "late_uncertainty_overreward"
            label = "Late-anchor uncertainty over-reward"
        elif dominant_family == "trajectory" and dominant_feature in {
            "traj_continuity",
            "traj_reflection_count",
            "traj_max_reflection",
        }:
            archetype_id = "trajectory_overbias"
            label = "Trajectory over-bias"
        elif rank_share >= 0.5:
            archetype_id = "rank_channel_reshuffle"
            label = "Rank-channel reshuffle artifact"
        else:
            archetype_id = "mixed_signal_conflict"
            label = "Mixed-signal conflict"

        case_out = dict(case)
        case_out["archetype_id"] = archetype_id
        case_out["archetype_label"] = label
        out.append(case_out)
    return out


def aggregate_failure_archetypes(
    wrong_cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for case in wrong_cases:
        grouped.setdefault(str(case["archetype_id"]), []).append(case)

    out: list[dict[str, Any]] = []
    total_cases = max(1, len(wrong_cases))
    for archetype_id, cases in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        domains: dict[str, int] = {}
        anchors: dict[int, int] = {}
        families: dict[str, int] = {}
        features: dict[str, int] = {}
        for case in cases:
            domains[str(case["domain"])] = domains.get(str(case["domain"]), 0) + 1
            anchors[int(case["anchor_pct"])] = anchors.get(int(case["anchor_pct"]), 0) + 1
            families[str(case["dominant_positive_family"])] = families.get(str(case["dominant_positive_family"]), 0) + 1
            features[str(case["dominant_positive_feature"])] = features.get(str(case["dominant_positive_feature"]), 0) + 1
        representative = max(cases, key=lambda case: float(case["score_gap"]))
        label = str(cases[0]["archetype_label"])
        out.append(
            {
                "archetype_id": archetype_id,
                "label": label,
                "n_cases": int(len(cases)),
                "fraction_of_wrong_cases": float(len(cases) / total_cases),
                "dominant_domains": ";".join(f"{key}:{value}" for key, value in sorted(domains.items(), key=lambda kv: (-kv[1], kv[0]))),
                "dominant_anchors": ";".join(f"{key}:{value}" for key, value in sorted(anchors.items(), key=lambda kv: (-kv[1], kv[0]))),
                "top_positive_families": ";".join(f"{key}:{value}" for key, value in sorted(families.items(), key=lambda kv: (-kv[1], kv[0]))[:3]),
                "top_positive_features": ";".join(f"{key}:{value}" for key, value in sorted(features.items(), key=lambda kv: (-kv[1], kv[0]))[:3]),
                "mean_score_gap": _safe_mean([float(case["score_gap"]) for case in cases]),
                "mean_rank_delta_share": _safe_mean([float(case["rank_delta_share"]) for case in cases]),
                "representative_method_id": str(representative["method_id"]),
                "representative_cache_key": str(representative["cache_key"]),
                "representative_problem_id": str(representative["problem_id"]),
                "representative_anchor_pct": int(representative["anchor_pct"]),
            }
        )
    return out


def select_appendix_cases(
    *,
    correct_candidates: Sequence[Mapping[str, Any]],
    wrong_candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []

    def _best_case(rows: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
        if not rows:
            return None
        return max(
            rows,
            key=lambda row: (
                float(row.get("selection_flip_rate_top_family", 0.0)),
                float(row.get("mean_score_drop_top_family", 0.0)),
                float(row.get("margin", 0.0)),
            ),
        )

    math_correct = [
        row
        for row in correct_candidates
        if str(row["method_id"]) == "es_svd_math_rr_r1" and str(row["domain"]) == "math"
    ]
    science_correct = [
        row
        for row in correct_candidates
        if str(row["method_id"]) == "es_svd_science_rr_r1" and str(row["domain"]) == "science"
    ]
    for candidate in (_best_case(math_correct), _best_case(science_correct)):
        if candidate is not None:
            selected.append(dict(candidate))

    used_keys = {
        (str(row["cache_key"]), str(row["problem_id"]), int(row["anchor_pct"]))
        for row in selected
    }
    wrong_sorted = sorted(
        wrong_candidates,
        key=lambda row: (
            -float(row.get("score_gap", 0.0)),
            str(row.get("archetype_id", "")),
        ),
    )
    seen_archetypes: set[str] = set()
    for row in wrong_sorted:
        key = (str(row["cache_key"]), str(row["problem_id"]), int(row["anchor_pct"]))
        archetype_id = str(row.get("archetype_id", ""))
        if key in used_keys:
            continue
        if archetype_id in seen_archetypes and len(seen_archetypes) < 2:
            continue
        selected.append(dict(row))
        used_keys.add(key)
        seen_archetypes.add(archetype_id)
        if len(selected) >= 4:
            break
    return selected[:4]


def summarize_faithfulness(
    run_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in run_records:
        grouped.setdefault(str(row["method_id"]), []).append(row)

    out = []
    for method_id, rows in sorted(grouped.items()):
        recon_errors = [float(row["reconstruction_error"]) for row in rows]
        component_errors = [float(row["component_sanity_error"]) for row in rows]
        out.append(
            {
                "method_id": method_id,
                "n_explained_runs": int(len(rows)),
                "max_reconstruction_error": float(np.max(recon_errors)) if recon_errors else float("nan"),
                "mean_reconstruction_error": _safe_mean(recon_errors),
                "max_component_error": float(np.max(component_errors)) if component_errors else float("nan"),
                "mean_component_error": _safe_mean(component_errors),
            }
        )
    return out
