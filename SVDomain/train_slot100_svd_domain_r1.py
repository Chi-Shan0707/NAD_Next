#!/usr/bin/env python3
"""Train a coding-only slot100 SVDomain bundle and export Best-of-N patches."""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.distance.engine import DistanceEngine, DistanceSpec
from nad.core.selectors.code_dynamic_impl import order_code_dynamic_group_indices
from nad.core.selectors.code_v2_impl import compute_code_v2_primary_scores_from_raw
from nad.core.selectors.lambda_svm_core import LambdaSVMScorer, build_weighted_pairwise_training_examples
from nad.core.views.reader import CacheReader
from nad.ops.bestofn_extreme8 import discover_cache_entries, validate_submission_payload, write_submission_payload
from nad.ops.earlystop_svd import (
    AVAILABILITY_FEATURES,
    CODING_DERIVATIVE_FEATURES,
    CODING_DYNAMIC_FEATURES,
    FULL_FEATURE_NAMES,
    SVDGroupTransformScorer,
    _auroc,
    _build_representation,
    _fit_svd_lr_model,
    _fit_svd_transform,
    _predict_svd_lr,
    _rank_transform_matrix,
    get_domain,
    save_earlystop_svd_bundle,
)
from scripts.export_earlystop_svd_submission import _load_or_build_feature_store
from scripts.export_bestofn_from_earlystop_svd_model import _problem_scores_from_payload
from scripts.run_bestofn_score_recovery_20260408 import (
    CODE_V2_EXHAUSTIVE_JSON,
    DEFAULT_FULL_MATH_SUBMISSION,
    _load_current_system_bundle,
)
from scripts.run_code_baseline_v1_phase2 import DEFAULT_VIEW
from scripts.run_earlystop_prefix10_svd_round1 import (
    EXTRACTION_POSITION_INDEX,
    EXTRACTION_POSITIONS,
    _display_path,
    _now_utc,
)
from scripts.run_science_hybrid_round3 import _combine_cache_metric_proxy, _system_delta
from SVDomain.train_es_svd_ms_rr_r1 import (
    DEFAULT_FEATURE_CHUNK_PROBLEMS,
    _build_holdout_problem_map,
    _resolve_path,
    _split_feature_store,
    _summarise_feature_store,
)

POSITION_VALUE = 1.0
TRAIN_POSITIONS = (float(POSITION_VALUE),)
DOMAIN_NAME = "coding"
BASE_METHOD_ID = "slot100_svd_code_domain_r1"
PRIMARY_METHOD_ID = f"{BASE_METHOD_ID}__hit1"
PAIRWISE_METHOD_ID = f"{BASE_METHOD_ID}__pairwise"
TARGET_BLIND_CACHE_KEYS = ("DS-R1/lcb_v5", "Qwen3-4B/lcb_v5")
POINTWISE_RANKS = (4, 8, 12, 16, 24)
POINTWISE_C_VALUES = (0.1, 1.0, 3.0, 10.0)
POINTWISE_CLASS_WEIGHT = ("none", "balanced")
PAIRWISE_RANKS = (4, 8, 12, 16, 24)
PAIRWISE_C_VALUES = (0.1, 1.0, 3.0, 10.0)
PAIRWISE_LOSSES = ("hinge", "squared_hinge")
WHITEN_OPTIONS = (False, True)
REPRESENTATION = "raw+rank"


@dataclass
class FamilySpec:
    name: str
    feature_names: list[str]


@dataclass
class CandidateSpec:
    family_name: str
    head_kind: str
    feature_names: list[str]
    rank: int
    whiten: bool
    c_value: float
    class_weight: str | None = None
    loss: str | None = None


@dataclass
class SearchGrid:
    head_kinds: tuple[str, ...]
    pointwise_ranks: tuple[int, ...]
    pointwise_c_values: tuple[float, ...]
    pointwise_class_weights: tuple[str, ...]
    pairwise_ranks: tuple[int, ...]
    pairwise_c_values: tuple[float, ...]
    pairwise_losses: tuple[str, ...]
    whiten_options: tuple[bool, ...]


@dataclass
class ProblemData:
    source_name: str
    cache_key: str
    base_cache_key: str
    dataset_name: str
    problem_id: str
    cv_group_key: str
    run_ids: list[int]
    labels: np.ndarray
    x_raw: np.ndarray
    x_rank: np.ndarray
    D: np.ndarray
    code_v2_scores: np.ndarray
    x_by_family: dict[str, np.ndarray]


@dataclass
class ProblemEvalRow:
    cache_key: str
    dataset_name: str
    problem_id: str
    sample_ids: list[int]
    labels: np.ndarray
    scores: np.ndarray
    order: np.ndarray


def _latest_science_json() -> Path:
    pattern = str(REPO_ROOT / "result" / "science_hybrid_round3_*" / "science_hybrid_round3.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError("No science_hybrid_round3.json found under result/")
    return Path(matches[-1])


def _discover_domain_cache_keys(cache_root: str, domain_name: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            str(entry.cache_key)
            for entry in discover_cache_entries(cache_root)
            if get_domain(str(entry.dataset_name)) == str(domain_name)
        )
    )


def _qualify_feature_store_local(feature_store: list[dict[str, Any]], source_name: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for payload in feature_store:
        item = dict(payload)
        item["source_name"] = str(source_name)
        item["base_cache_key"] = str(payload["cache_key"])
        item["cache_key"] = f"{source_name}/{payload['cache_key']}"
        out.append(item)
    return out


def _load_domain_feature_store(
    *,
    source_name: str,
    cache_root: str,
    domain_name: str,
    positions: tuple[float, ...],
    required_feature_names: set[str],
    max_problems_per_cache: int | None,
    feature_workers: int,
    chunk_problems: int,
    feature_cache_dir: Path | None,
    refresh_feature_cache: bool,
) -> tuple[list[dict[str, Any]], Path | None, str]:
    include_cache_keys = set(_discover_domain_cache_keys(cache_root, domain_name))
    if not include_cache_keys:
        return [], None, "skipped_empty_domain"
    raw_store, cache_path, cache_status = _load_or_build_feature_store(
        cache_root=cache_root,
        positions=positions,
        required_feature_names=required_feature_names,
        max_problems=max_problems_per_cache,
        reflection_threshold=0.30,
        workers=int(feature_workers),
        feature_chunk_problems=int(chunk_problems),
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(refresh_feature_cache),
        include_cache_keys=include_cache_keys,
    )
    return _qualify_feature_store_local(raw_store, source_name), cache_path, cache_status


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _family_specs(selected_names: Optional[set[str]] = None) -> list[FamilySpec]:
    code_v2_core = [
        "prefix_best_window_quality",
        "head_tail_gap",
        "tail_variance",
        "post_reflection_recovery",
        "last_block_instability",
    ]
    code_dyn_aux = [
        "reflection_density",
        "reflection_count",
        "traj_late_convergence",
        "traj_continuity",
        "traj_max_reflection",
    ]
    terminal_aux = [
        "nc_mean",
        "nc_slope",
        "self_similarity",
        "tail_q10",
        "last_event_tail_conf",
        "event_pre_post_delta",
    ]
    avail = list(AVAILABILITY_FEATURES)
    families = [
        FamilySpec(
            name="svd_code_core",
            feature_names=_dedupe(code_v2_core + avail),
        ),
        FamilySpec(
            name="svd_code_dyn",
            feature_names=_dedupe(code_v2_core + code_dyn_aux + avail),
        ),
        FamilySpec(
            name="svd_code_deriv",
            feature_names=_dedupe(code_v2_core + list(CODING_DERIVATIVE_FEATURES) + avail),
        ),
        FamilySpec(
            name="svd_code_messy_all",
            feature_names=_dedupe(code_v2_core + code_dyn_aux + terminal_aux + list(CODING_DERIVATIVE_FEATURES) + avail),
        ),
    ]
    if not selected_names:
        return families
    selected = {str(name) for name in selected_names}
    out = [family for family in families if family.name in selected]
    missing = sorted(selected - {family.name for family in families})
    if missing:
        raise ValueError(f"Unknown family names: {missing}")
    return out


def _feature_to_index() -> dict[str, int]:
    return {name: idx for idx, name in enumerate(FULL_FEATURE_NAMES)}


def _route_feature_indices(feature_names: list[str]) -> list[int]:
    feature_to_idx = _feature_to_index()
    return [int(feature_to_idx[name]) for name in feature_names]


def _candidate_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, str]:
    metrics = row["metrics"]
    return (
        float(metrics.get("hit@1") or 0.0),
        float(metrics.get("pairwise") or -1.0),
        float(metrics.get("selacc@10%") or 0.0),
        float(metrics.get("auroc") or -1.0),
        -float(metrics.get("avg_rank_proxy") or 9999.0),
        str(row["name"]),
    )


def _choose_recommended_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not rows:
        raise ValueError("No holdout rows to choose from")
    hit1_sorted = sorted(
        rows,
        key=lambda row: (
            float(row["metrics"].get("hit@1") or 0.0),
            float(row["metrics"].get("pairwise") or -1.0),
            float(row["metrics"].get("selacc@10%") or 0.0),
            float(row["metrics"].get("auroc") or -1.0),
            str(row["name"]),
        ),
        reverse=True,
    )
    pairwise_sorted = sorted(
        rows,
        key=lambda row: (
            float(row["metrics"].get("pairwise") or -1.0),
            float(row["metrics"].get("hit@1") or 0.0),
            float(row["metrics"].get("selacc@10%") or 0.0),
            float(row["metrics"].get("auroc") or -1.0),
            str(row["name"]),
        ),
        reverse=True,
    )
    hit1_row = hit1_sorted[0]
    pairwise_row = pairwise_sorted[0]
    if hit1_row["name"] == pairwise_row["name"]:
        return {"deploy_mode": "single", "hit1": hit1_row, "pairwise": pairwise_row, "single": hit1_row}
    hit1_delta = abs(float(hit1_row["metrics"].get("hit@1") or 0.0) - float(pairwise_row["metrics"].get("hit@1") or 0.0))
    if hit1_delta < 0.002:
        return {"deploy_mode": "pairwise_first", "hit1": hit1_row, "pairwise": pairwise_row}
    return {"deploy_mode": "dual_head", "hit1": hit1_row, "pairwise": pairwise_row}


def _group_problem_folds(problems: list[ProblemData], n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray([prob.cv_group_key for prob in problems], dtype=object)
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        return []
    splits = min(int(n_splits), int(unique_groups.size))
    if splits < 2:
        return []
    gkf = GroupKFold(n_splits=splits)
    dummy = np.zeros((len(problems), 1), dtype=np.float64)
    return list(gkf.split(dummy, groups=groups))


def _code_v2_required_raw(x_raw: np.ndarray) -> dict[str, np.ndarray]:
    feature_to_idx = _feature_to_index()
    return {
        "prefix_best_window_quality": np.asarray(x_raw[:, feature_to_idx["prefix_best_window_quality"]], dtype=np.float64),
        "head_tail_gap": np.asarray(x_raw[:, feature_to_idx["head_tail_gap"]], dtype=np.float64),
        "tail_variance": np.asarray(x_raw[:, feature_to_idx["tail_variance"]], dtype=np.float64),
        "post_reflection_recovery": np.asarray(x_raw[:, feature_to_idx["post_reflection_recovery"]], dtype=np.float64),
        "last_block_instability": np.asarray(x_raw[:, feature_to_idx["last_block_instability"]], dtype=np.float64),
    }


def _fit_pointwise_route(
    problems: list[ProblemData],
    spec: CandidateSpec,
    *,
    random_state: int,
) -> Optional[dict[str, Any]]:
    if not problems:
        return None
    x_train = np.vstack([prob.x_by_family[spec.family_name] for prob in problems]).astype(np.float64, copy=False)
    y_train = np.concatenate([prob.labels for prob in problems]).astype(np.int32, copy=False)
    model = _fit_svd_lr_model(
        x=x_train,
        y=y_train,
        rank=int(spec.rank),
        c_value=float(spec.c_value),
        whiten=bool(spec.whiten),
        class_weight_name=str(spec.class_weight),
        random_state=int(random_state),
    )
    if model is None:
        return None
    return {
        "route_type": "svd",
        "family_name": str(spec.family_name),
        "representation": REPRESENTATION,
        "feature_names": list(spec.feature_names),
        "feature_indices": _route_feature_indices(spec.feature_names),
        "rank": int(spec.rank),
        "whiten": bool(spec.whiten),
        "c_value": float(spec.c_value),
        "class_weight": str(spec.class_weight),
        "head_kind": "pointwise",
        "model": model,
    }


def _fit_pairwise_route(
    problems: list[ProblemData],
    spec: CandidateSpec,
    *,
    random_state: int,
) -> Optional[dict[str, Any]]:
    if not problems:
        return None
    x_train = np.vstack([prob.x_by_family[spec.family_name] for prob in problems]).astype(np.float64, copy=False)
    transform = _fit_svd_transform(
        x_train,
        rank=int(spec.rank),
        whiten=bool(spec.whiten),
        random_state=int(random_state),
    )
    if transform is None:
        return None

    pair_x: list[np.ndarray] = []
    pair_y: list[np.ndarray] = []
    pair_w: list[np.ndarray] = []
    wrapper = SVDGroupTransformScorer(
        scaler=transform["scaler"],
        svd=transform["svd"],
        scorer=LambdaSVMScorer(
            C=float(spec.c_value),
            loss=str(spec.loss),
            fit_intercept=False,
            backend="utility",
        ),
        whiten=bool(spec.whiten),
    )

    for prob in problems:
        z = wrapper.transform(prob.x_by_family[spec.family_name])
        x_pairs, y_pairs, sample_weight = build_weighted_pairwise_training_examples(
            z,
            prob.labels,
            reference_scores=np.asarray(prob.code_v2_scores, dtype=np.float64),
            pair_weight_mode="dcg_delta",
        )
        if x_pairs.shape[0] <= 0:
            continue
        pair_x.append(x_pairs)
        pair_y.append(y_pairs)
        pair_w.append(sample_weight)

    if not pair_x:
        return None

    wrapper.scorer.fit(
        np.concatenate(pair_x, axis=0),
        np.concatenate(pair_y, axis=0),
        sample_weight=np.concatenate(pair_w, axis=0),
    )
    return {
        "route_type": "ranksvm",
        "family_name": str(spec.family_name),
        "representation": REPRESENTATION,
        "feature_names": list(spec.feature_names),
        "feature_indices": _route_feature_indices(spec.feature_names),
        "rank": int(transform["rank"]),
        "whiten": bool(spec.whiten),
        "c_value": float(spec.c_value),
        "loss": str(spec.loss),
        "head_kind": "pairwise",
        "scorer": wrapper,
    }


def _fit_route(
    problems: list[ProblemData],
    spec: CandidateSpec,
    *,
    random_state: int,
) -> Optional[dict[str, Any]]:
    if spec.head_kind == "pointwise":
        return _fit_pointwise_route(problems, spec, random_state=random_state)
    if spec.head_kind == "pairwise":
        return _fit_pairwise_route(problems, spec, random_state=random_state)
    raise ValueError(f"Unknown head_kind: {spec.head_kind}")


def _score_problem_with_route(problem: ProblemData, route: dict[str, Any]) -> np.ndarray:
    route_type = str(route["route_type"])
    if route_type == "code_v2":
        scores, _ = compute_code_v2_primary_scores_from_raw(_code_v2_required_raw(problem.x_raw))
        return np.asarray(scores, dtype=np.float64)
    if route_type == "baseline":
        feature_to_idx = _feature_to_index()
        return np.asarray(problem.x_raw[:, feature_to_idx[str(route["signal_name"])]], dtype=np.float64)

    x_rep = np.asarray(problem.x_by_family[str(route["family_name"])], dtype=np.float64)
    if route_type == "svd":
        return _predict_svd_lr(route["model"], x_rep)
    if route_type == "ranksvm":
        return np.asarray(route["scorer"].score_group(x_rep), dtype=np.float64)
    raise ValueError(f"Unsupported route_type: {route_type}")


def _problem_eval_rows(
    problems: list[ProblemData],
    route: dict[str, Any],
) -> tuple[list[ProblemEvalRow], dict[str, dict[str, dict[str, float]]]]:
    rows: list[ProblemEvalRow] = []
    problem_scores: dict[str, dict[str, dict[str, float]]] = {}
    for prob in problems:
        scores = np.asarray(_score_problem_with_route(prob, route), dtype=np.float64)
        order = order_code_dynamic_group_indices(scores, prob.D, run_ids=prob.run_ids)
        rows.append(
            ProblemEvalRow(
                cache_key=str(prob.cache_key),
                dataset_name=str(prob.dataset_name),
                problem_id=str(prob.problem_id),
                sample_ids=list(prob.run_ids),
                labels=np.asarray(prob.labels, dtype=np.int32),
                scores=scores,
                order=np.asarray(order, dtype=np.int64),
            )
        )
        cache_scores = problem_scores.setdefault(str(prob.cache_key), {})
        cache_scores[str(prob.problem_id)] = {
            str(sample_id): float(scores[idx])
            for idx, sample_id in enumerate(prob.run_ids)
        }
    return rows, problem_scores


def _metrics_from_eval_rows(rows: list[ProblemEvalRow]) -> dict[str, Any]:
    all_scores: list[float] = []
    all_labels: list[int] = []
    hit1_total = 0.0
    hit3_total = 0.0
    pairwise_num = 0.0
    pairwise_den = 0.0
    avg_rank_proxy_total = 0.0

    for row in rows:
        scores = np.asarray(row.scores, dtype=np.float64)
        labels = np.asarray(row.labels, dtype=np.int32)
        if scores.size <= 0:
            continue
        order = np.asarray(row.order, dtype=np.int64)
        if order.size <= 0:
            order = np.argsort(-scores, kind="mergesort")
        hit1_total += float(labels[int(order[0])] > 0)
        hit3_total += float(np.any(labels[order[: min(3, order.size)]] > 0))
        correct_pos = np.where(labels[order] > 0)[0]
        best_correct_rank = int(correct_pos[0] + 1) if correct_pos.size > 0 else int(order.size + 1)
        avg_rank_proxy_total += float(best_correct_rank)

        pos_scores = scores[labels > 0]
        neg_scores = scores[labels <= 0]
        if pos_scores.size > 0 and neg_scores.size > 0:
            diff = pos_scores[:, None] - neg_scores[None, :]
            pairwise_num += float((diff > 0).sum()) + 0.5 * float((diff == 0).sum())
            pairwise_den += float(diff.size)

        all_scores.extend(scores.tolist())
        all_labels.extend(labels.tolist())

    scores_arr = np.asarray(all_scores, dtype=np.float64)
    labels_arr = np.asarray(all_labels, dtype=np.int32)
    auroc = None
    if labels_arr.size > 0 and np.unique(labels_arr).size >= 2:
        auroc = float(_auroc(scores_arr, labels_arr))

    selacc10 = 0.0
    top10_count = 0
    if labels_arr.size > 0:
        top10_count = max(1, int(math.ceil(0.10 * labels_arr.size)))
        order = np.argsort(-scores_arr, kind="mergesort")
        selacc10 = float(labels_arr[order[:top10_count]].mean())

    return {
        "auroc": auroc,
        "hit@1": float(hit1_total / len(rows)) if rows else 0.0,
        "hit@3": float(hit3_total / len(rows)) if rows else 0.0,
        "pairwise": float(pairwise_num / pairwise_den) if pairwise_den > 0 else None,
        "selacc@10%": float(selacc10),
        "avg_rank_proxy": float(avg_rank_proxy_total / len(rows)) if rows else None,
        "n_problems": int(len(rows)),
        "n_samples": int(labels_arr.size),
        "top10_count": int(top10_count),
    }


def _equal_cache_mean(per_cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not per_cache:
        return {}
    out: dict[str, Any] = {}
    for key in ("auroc", "hit@1", "hit@3", "pairwise", "selacc@10%", "avg_rank_proxy"):
        values = [
            float(metrics[key])
            for metrics in per_cache.values()
            if metrics.get(key) is not None and np.isfinite(float(metrics[key]))
        ]
        out[key] = None if not values else float(np.mean(values))
    return out


def _evaluate_route(
    *,
    method_name: str,
    problems: list[ProblemData],
    route: dict[str, Any],
) -> dict[str, Any]:
    rows, problem_scores = _problem_eval_rows(problems, route)
    per_cache_rows: dict[str, list[ProblemEvalRow]] = {}
    for row in rows:
        per_cache_rows.setdefault(str(row.cache_key), []).append(row)
    per_cache = {
        cache_key: _metrics_from_eval_rows(cache_rows)
        for cache_key, cache_rows in sorted(per_cache_rows.items())
    }
    sample_weighted = _metrics_from_eval_rows(rows)
    equal_cache = _equal_cache_mean(per_cache)
    return {
        "method_name": str(method_name),
        "metrics": sample_weighted,
        "sample_weighted": sample_weighted,
        "equal_cache_mean": equal_cache,
        "per_cache": per_cache,
        "problem_scores": problem_scores,
    }


def _candidate_specs(
    family_specs: list[FamilySpec],
    *,
    search_grid: SearchGrid,
) -> list[CandidateSpec]:
    out: list[CandidateSpec] = []
    for family in family_specs:
        if "pointwise" in search_grid.head_kinds:
            for rank in search_grid.pointwise_ranks:
                for whiten in search_grid.whiten_options:
                    for c_value in search_grid.pointwise_c_values:
                        for class_weight in search_grid.pointwise_class_weights:
                            out.append(
                                CandidateSpec(
                                    family_name=family.name,
                                    head_kind="pointwise",
                                    feature_names=list(family.feature_names),
                                    rank=int(rank),
                                    whiten=bool(whiten),
                                    c_value=float(c_value),
                                    class_weight=str(class_weight),
                                )
                            )
        if "pairwise" in search_grid.head_kinds:
            for rank in search_grid.pairwise_ranks:
                for whiten in search_grid.whiten_options:
                    for c_value in search_grid.pairwise_c_values:
                        for loss in search_grid.pairwise_losses:
                            out.append(
                                CandidateSpec(
                                    family_name=family.name,
                                    head_kind="pairwise",
                                    feature_names=list(family.feature_names),
                                    rank=int(rank),
                                    whiten=bool(whiten),
                                    c_value=float(c_value),
                                    loss=str(loss),
                                )
                            )
    return out


def _parse_csv_items(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(raw).split(",") if item.strip())


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _parse_csv_items(raw))


def _parse_csv_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(item) for item in _parse_csv_items(raw))


def _parse_csv_bools(raw: str) -> tuple[bool, ...]:
    out: list[bool] = []
    for item in _parse_csv_items(raw):
        text = item.lower()
        if text in {"1", "true", "t", "yes", "y", "on"}:
            out.append(True)
        elif text in {"0", "false", "f", "no", "n", "off"}:
            out.append(False)
        else:
            raise ValueError(f"Invalid boolean token: {item}")
    return tuple(out)


def _candidate_name(spec: CandidateSpec) -> str:
    if spec.head_kind == "pointwise":
        return (
            f"{spec.family_name}__pointwise__rank{int(spec.rank)}"
            f"__w{int(bool(spec.whiten))}__C{float(spec.c_value):.2f}__cw{spec.class_weight}"
        ).replace(".", "p")
    return (
        f"{spec.family_name}__pairwise__rank{int(spec.rank)}"
        f"__w{int(bool(spec.whiten))}__C{float(spec.c_value):.2f}__{spec.loss}"
    ).replace(".", "p")


def _crossval_candidate(
    problems: list[ProblemData],
    spec: CandidateSpec,
    *,
    n_splits: int,
    random_state: int,
) -> Optional[dict[str, Any]]:
    folds = _group_problem_folds(problems, n_splits=n_splits)
    if not folds:
        return None

    rows: list[ProblemEvalRow] = []
    per_fold: list[dict[str, Any]] = []
    for train_idx, test_idx in folds:
        train_probs = [problems[int(idx)] for idx in train_idx.tolist()]
        test_probs = [problems[int(idx)] for idx in test_idx.tolist()]
        route = _fit_route(train_probs, spec, random_state=random_state)
        if route is None:
            continue
        fold_rows, _ = _problem_eval_rows(test_probs, route)
        fold_metrics = _metrics_from_eval_rows(fold_rows)
        rows.extend(fold_rows)
        per_fold.append(fold_metrics)

    if not rows:
        return None

    metrics = _metrics_from_eval_rows(rows)
    return {
        "name": _candidate_name(spec),
        "family_name": str(spec.family_name),
        "head_kind": str(spec.head_kind),
        "config": {
            "family_name": str(spec.family_name),
            "head_kind": str(spec.head_kind),
            "feature_names": list(spec.feature_names),
            "rank": int(spec.rank),
            "whiten": bool(spec.whiten),
            "c_value": float(spec.c_value),
            "class_weight": None if spec.class_weight is None else str(spec.class_weight),
            "loss": None if spec.loss is None else str(spec.loss),
        },
        "metrics": metrics,
        "per_fold": per_fold,
        "n_valid_folds": int(len(per_fold)),
    }


def _select_cv_winners(cv_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in cv_rows:
        buckets.setdefault((str(row["family_name"]), str(row["head_kind"])), []).append(row)
    winners: list[dict[str, Any]] = []
    for key, rows in sorted(buckets.items()):
        winner = sorted(rows, key=_candidate_key, reverse=True)[0]
        winners.append(winner)
    return winners


def _build_entry_lookup(*, main_cache_root: str, extra_cache_root: str) -> dict[tuple[str, str], Any]:
    lookup: dict[tuple[str, str], Any] = {}
    for source_name, root in (("cache", main_cache_root), ("cache_train", extra_cache_root)):
        for entry in discover_cache_entries(root):
            lookup[(str(source_name), str(entry.cache_key))] = entry
    return lookup


def _build_problem_dataset(
    *,
    feature_store: list[dict[str, Any]],
    family_specs: list[FamilySpec],
    entry_lookup: dict[tuple[str, str], Any],
    distance_threads: int,
) -> list[ProblemData]:
    feature_to_idx = _feature_to_index()
    family_indices = {
        family.name: [int(feature_to_idx[name]) for name in family.feature_names]
        for family in family_specs
    }
    readers: dict[tuple[str, str], CacheReader] = {}
    problems: list[ProblemData] = []
    engine = DistanceEngine(DistanceSpec("ja", num_threads=int(distance_threads)))

    for payload in feature_store:
        lookup_key = (str(payload["source_name"]), str(payload["base_cache_key"]))
        entry = entry_lookup[lookup_key]
        reader = readers.get(lookup_key)
        if reader is None:
            reader = CacheReader(str(entry.cache_root))
            readers[lookup_key] = reader

        tensor = np.asarray(payload["tensor"], dtype=np.float64)
        labels_all = np.asarray(payload["labels"], dtype=np.int32)
        sample_ids_all = np.asarray(payload["sample_ids"], dtype=np.int32)
        position_to_index = {float(pos): idx for idx, pos in enumerate(payload["positions"])}
        src_pos_idx = int(position_to_index[float(POSITION_VALUE)])
        for problem_idx, problem_id in enumerate(payload["problem_ids"]):
            start = int(payload["problem_offsets"][problem_idx])
            end = int(payload["problem_offsets"][problem_idx + 1])
            x_raw = np.asarray(tensor[start:end, src_pos_idx, :], dtype=np.float64)
            if x_raw.shape[0] <= 0:
                continue
            x_rank = _rank_transform_matrix(x_raw)
            run_ids = [int(v) for v in sample_ids_all[start:end].tolist()]
            labels = np.asarray(labels_all[start:end], dtype=np.int32)
            views = [reader.get_run_view(int(run_id), DEFAULT_VIEW) for run_id in run_ids]
            D = engine.dense_matrix(views)
            code_v2_scores, _ = compute_code_v2_primary_scores_from_raw(_code_v2_required_raw(x_raw))
            x_by_family = {
                family.name: _build_representation(
                    x_raw=x_raw,
                    x_rank=x_rank,
                    feature_indices=list(family_indices[family.name]),
                    representation=REPRESENTATION,
                )
                for family in family_specs
            }
            problems.append(
                ProblemData(
                    source_name=str(payload["source_name"]),
                    cache_key=str(payload["cache_key"]),
                    base_cache_key=str(payload["base_cache_key"]),
                    dataset_name=str(payload["dataset_name"]),
                    problem_id=str(problem_id),
                    cv_group_key=f"{payload['dataset_name']}::{problem_id}",
                    run_ids=run_ids,
                    labels=labels,
                    x_raw=x_raw,
                    x_rank=x_rank,
                    D=np.asarray(D, dtype=np.float64),
                    code_v2_scores=np.asarray(code_v2_scores, dtype=np.float64),
                    x_by_family=x_by_family,
                )
            )
    return problems


def _route_from_current_slot100(bundle_path: Path) -> dict[str, Any]:
    import pickle

    with bundle_path.open("rb") as handle:
        bundle = pickle.load(handle)
    return dict(bundle["domains"][DOMAIN_NAME]["routes"][9])


def _code_v2_baseline_route() -> dict[str, Any]:
    return {
        "route_type": "code_v2",
        "family_name": "code_v2_raw",
        "representation": REPRESENTATION,
        "feature_names": [
            "prefix_best_window_quality",
            "head_tail_gap",
            "tail_variance",
            "post_reflection_recovery",
            "last_block_instability",
        ],
    }


def _build_single_slot_bundle(*, method_id: str, route: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "bundle_version": str(method_id),
        "created_at_utc": _now_utc(),
        "feature_names": list(FULL_FEATURE_NAMES),
        "positions": [float(POSITION_VALUE)],
        "anchor_positions": [float(POSITION_VALUE)],
        "protocol": dict(protocol),
        "domains": {
            DOMAIN_NAME: {
                "routes": [route],
            }
        },
    }


def _patch_base_submission(
    *,
    base_submission_path: Path,
    out_path: Path,
    method_name: str,
    scores: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    payload = json.loads(base_submission_path.read_text(encoding="utf-8"))
    for cache_key in TARGET_BLIND_CACHE_KEYS:
        if cache_key not in payload.get("scores", {}):
            raise RuntimeError(f"Base submission missing coding cache key: {cache_key}")
        if cache_key not in scores:
            raise RuntimeError(f"Patched scores missing coding cache key: {cache_key}")
        payload["scores"][cache_key] = scores[cache_key]
    payload["method_name"] = str(method_name)
    payload.setdefault("score_postprocess", {})["coding_patch"] = {
        "source_method_name": str(method_name),
        "target_cache_keys": list(TARGET_BLIND_CACHE_KEYS),
        "note": "coding caches replaced by slot100 SVDomain route",
    }
    expected_cache_keys = [entry.cache_key for entry in discover_cache_entries("/home/jovyan/public-ro/MUI_HUB/cache_test")]
    summary = validate_submission_payload(payload, expected_cache_keys=expected_cache_keys)
    write_submission_payload(payload, out_path)
    return {
        "path": _display_path(out_path),
        "validation": summary,
    }


def _build_blind_problem_dataset(
    *,
    cache_root: str,
    feature_store: list[dict[str, Any]],
    family_specs: list[FamilySpec],
    distance_threads: int,
) -> list[ProblemData]:
    lookup = {
        ("blind", str(entry.cache_key)): entry
        for entry in discover_cache_entries(cache_root)
        if str(entry.cache_key) in TARGET_BLIND_CACHE_KEYS
    }
    qualified = []
    for payload in feature_store:
        item = dict(payload)
        item["source_name"] = "blind"
        item["base_cache_key"] = str(payload["cache_key"])
        qualified.append(item)
    return _build_problem_dataset(
        feature_store=qualified,
        family_specs=family_specs,
        entry_lookup=lookup,
        distance_threads=distance_threads,
    )


def _render_pct(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        val = float(value)
    except Exception:
        return "N/A"
    if not np.isfinite(val):
        return "N/A"
    return f"{100.0 * val:.2f}%"


def _render_rank(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        val = float(value)
    except Exception:
        return "N/A"
    if not np.isfinite(val):
        return "N/A"
    return f"{val:.3f}"


def _write_doc(path: Path, summary: dict[str, Any]) -> None:
    holdout_rows = summary["holdout"]["candidate_rows"]
    lines = [
        "# SVD Slot100 Coding Domain R1",
        "",
        "## Summary",
        "",
        "- 只覆盖 `coding` 域，只训练 `100% slot`。",
        "- 主模型家族仍然是 `raw+rank + low-rank linear`。",
        "- 允许 `code_v2` 核心特征、reflection/dynamic 辅助、terminal 辅助、slice 导数特征混合搜索。",
        "",
        "## Protocol",
        "",
        f"- `main cache root`：`{summary['protocol']['main_cache_root']}`。",
        f"- `extra cache root`：`{summary['protocol']['extra_cache_root']}`。",
        f"- `blind cache root`：`{summary['protocol']['blind_cache_root']}`。",
        f"- `holdout split`：`{summary['protocol']['holdout_split']}`。",
        f"- `split seed`：`{summary['protocol']['split_seed']}`。",
        f"- `cv folds`：`{summary['protocol']['n_splits']}`。",
        f"- `max_problems_per_cache`：`{summary['run_flags']['max_problems_per_cache']}`。",
        "",
        "## Holdout Baselines",
        "",
        "| Method | AUROC | Hit@1 | Hit@3 | SelAcc@10 | Pairwise | AvgRank proxy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["holdout"]["baselines"].items():
        metrics = row["metrics"]
        lines.append(
            "| {name} | {auroc} | {hit1} | {hit3} | {selacc} | {pairwise} | {avg_rank} |".format(
                name=name,
                auroc=_render_pct(metrics.get("auroc")),
                hit1=_render_pct(metrics.get("hit@1")),
                hit3=_render_pct(metrics.get("hit@3")),
                selacc=_render_pct(metrics.get("selacc@10%")),
                pairwise=_render_pct(metrics.get("pairwise")),
                avg_rank=_render_rank(metrics.get("avg_rank_proxy")),
            )
        )
    lines.extend([
        "",
        "## Holdout Candidates",
        "",
        "| Candidate | Head | Family | AUROC | Hit@1 | Hit@3 | SelAcc@10 | Pairwise | AvgRank proxy |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in holdout_rows:
        metrics = row["metrics"]
        lines.append(
            "| {name} | {head} | {family} | {auroc} | {hit1} | {hit3} | {selacc} | {pairwise} | {avg_rank} |".format(
                name=row["name"],
                head=row["head_kind"],
                family=row["family_name"],
                auroc=_render_pct(metrics.get("auroc")),
                hit1=_render_pct(metrics.get("hit@1")),
                hit3=_render_pct(metrics.get("hit@3")),
                selacc=_render_pct(metrics.get("selacc@10%")),
                pairwise=_render_pct(metrics.get("pairwise")),
                avg_rank=_render_rank(metrics.get("avg_rank_proxy")),
            )
        )
    recommended = summary["recommended"]
    lines.extend([
        "",
        "## Recommendation",
        "",
        f"- `deploy_mode`：`{recommended['deploy_mode']}`。",
        f"- `hit1 route`：`{recommended['hit1']['name']}`。",
        f"- `pairwise route`：`{recommended['pairwise']['name']}`。",
        "",
        "## Full-System Proxy",
        "",
    ])
    for label, row in summary["system_proxy"].items():
        main = row["delta"]["sample_weighted"]
        eq = row["delta"]["equal_cache_mean"]
        lines.extend([
            f"### {label}",
            "",
            f"- `sample_weighted ΔHit@1`：`{_render_pct(main.get('hit@1'))}`。",
            f"- `sample_weighted ΔSelAcc@10`：`{_render_pct(main.get('selacc@10%'))}`。",
            f"- `equal_cache ΔHit@1`：`{_render_pct(eq.get('hit@1'))}`。",
            f"- `equal_cache ΔSelAcc@10`：`{_render_pct(eq.get('selacc@10%'))}`。",
            "",
        ])
    lines.extend([
        "## Artifacts",
        "",
        f"- `summary json`：`{summary['artifacts']['summary_json']}`。",
        f"- `eval json`：`{summary['artifacts']['eval_json']}`。",
        f"- `candidate json`：`{summary['artifacts']['candidate_json']}`。",
        f"- `doc`：`{summary['artifacts']['doc_md']}`。",
    ])
    for label, model_path in summary["artifacts"]["models"].items():
        lines.append(f"- `{label} model`：`{model_path}`。")
    for label, sub_info in summary["artifacts"]["submissions"].items():
        lines.append(f"- `{label} submission`：`{sub_info['path']}`。")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compact_eval_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "method_name": str(row["method_name"]),
        "name": str(row.get("name", row["method_name"])),
        "family_name": None if row.get("family_name") is None else str(row["family_name"]),
        "head_kind": None if row.get("head_kind") is None else str(row["head_kind"]),
        "config": None if row.get("config") is None else dict(row["config"]),
        "metrics": dict(row["metrics"]),
        "sample_weighted": dict(row["sample_weighted"]),
        "equal_cache_mean": dict(row["equal_cache_mean"]),
        "per_cache": dict(row["per_cache"]),
    }


def _write_registry(path: Path, summary: dict[str, Any]) -> None:
    registry = {"family": "es_svd", "methods": []}
    if path.exists():
        registry = json.loads(path.read_text(encoding="utf-8"))
    existing = {
        str(item.get("method_id")): dict(item)
        for item in registry.get("methods", [])
        if isinstance(item, dict) and item.get("method_id")
    }
    for label, model_path in summary["artifacts"]["models"].items():
        method_id = PRIMARY_METHOD_ID if label == "hit1" else PAIRWISE_METHOD_ID
        item = {
            "method_id": method_id,
            "kind": "single_domain_slot100_bundle",
            "domain": DOMAIN_NAME,
            "representation": REPRESENTATION,
            "position": 100,
            "model_path": model_path,
            "summary_path": summary["artifacts"]["summary_json"],
            "eval_path": summary["artifacts"]["eval_json"],
            "doc_path": summary["artifacts"]["doc_md"],
        }
        if label in summary["artifacts"]["submissions"]:
            item["submission_path"] = summary["artifacts"]["submissions"][label]["path"]
        existing[method_id] = item
    registry["family"] = "es_svd"
    registry["updated_at_utc"] = _now_utc()
    registry["methods"] = sorted(existing.values(), key=lambda item: str(item["method_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train coding-only slot100 SVDomain best-of-n candidates")
    ap.add_argument("--main-cache-root", default="MUI_HUB/cache")
    ap.add_argument("--extra-cache-root", default="MUI_HUB/cache_train")
    ap.add_argument("--blind-cache-root", default="/home/jovyan/public-ro/MUI_HUB/cache_test")
    ap.add_argument("--holdout-split", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--feature-workers", type=int, default=8)
    ap.add_argument("--distance-threads", type=int, default=8)
    ap.add_argument("--feature-chunk-problems", type=int, default=DEFAULT_FEATURE_CHUNK_PROBLEMS)
    ap.add_argument("--feature-cache-dir", default="results/cache/slot100_svd_code_domain_r1")
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--max-problems-per-cache", type=int, default=0)
    ap.add_argument("--blind-max-problems-per-cache", type=int, default=0)
    ap.add_argument("--families", default="svd_code_core,svd_code_dyn,svd_code_deriv,svd_code_messy_all")
    ap.add_argument("--heads", default="pointwise,pairwise")
    ap.add_argument("--pointwise-ranks", default="4,8,12,16,24")
    ap.add_argument("--pairwise-ranks", default="4,8,12,16,24")
    ap.add_argument("--pointwise-c-values", default="0.1,1.0,3.0,10.0")
    ap.add_argument("--pairwise-c-values", default="0.1,1.0,3.0,10.0")
    ap.add_argument("--pointwise-class-weights", default="none,balanced")
    ap.add_argument("--pairwise-losses", default="hinge,squared_hinge")
    ap.add_argument("--whiten-options", default="0,1")
    ap.add_argument("--skip-system-proxy", action="store_true")
    ap.add_argument("--skip-blind-export", action="store_true")
    ap.add_argument("--current-slot100-bundle", default="models/ml_selectors/earlystop_prefix10_svd_round1c_fullcache.pkl")
    ap.add_argument("--base-submission", default=str(DEFAULT_FULL_MATH_SUBMISSION))
    ap.add_argument("--out-summary", default="results/scans/bestofn_bridge/slot100_svd_code_domain_r1_summary.json")
    ap.add_argument("--out-eval", default="results/scans/bestofn_bridge/slot100_svd_code_domain_r1_eval.json")
    ap.add_argument("--out-candidates", default="results/scans/bestofn_bridge/slot100_svd_code_domain_r1_candidates.json")
    ap.add_argument("--out-doc", default="docs/SVD_SLOT100_DOMAIN_R1_RESULTS_20260411.md")
    ap.add_argument("--out-model-prefix", default="models/ml_selectors/slot100_svd_code_domain_r1")
    ap.add_argument("--out-submission-prefix", default="submission/BestofN/extreme12/patches/slot100_svd_code_domain_r1")
    ap.add_argument("--registry-path", default="SVDomain/registry.json")
    args = ap.parse_args()

    main_cache_root = _resolve_path(str(args.main_cache_root))
    extra_cache_root = _resolve_path(str(args.extra_cache_root))
    blind_cache_root = _resolve_path(str(args.blind_cache_root))
    base_submission_path = Path(str(args.base_submission))
    if not base_submission_path.is_absolute():
        base_submission_path = (REPO_ROOT / base_submission_path).resolve()
    current_slot100_bundle = Path(str(args.current_slot100_bundle))
    if not current_slot100_bundle.is_absolute():
        current_slot100_bundle = (REPO_ROOT / current_slot100_bundle).resolve()
    current_slot100_route = _route_from_current_slot100(current_slot100_bundle)

    feature_cache_dir = None if str(args.feature_cache_dir).strip().lower() in {"", "none", "off"} else (REPO_ROOT / str(args.feature_cache_dir)).resolve()
    max_problems_per_cache = None if int(args.max_problems_per_cache) <= 0 else int(args.max_problems_per_cache)
    blind_max_problems_per_cache = None if int(args.blind_max_problems_per_cache) <= 0 else int(args.blind_max_problems_per_cache)

    family_specs = _family_specs(set(_parse_csv_items(str(args.families))))
    search_grid = SearchGrid(
        head_kinds=_parse_csv_items(str(args.heads)),
        pointwise_ranks=_parse_csv_ints(str(args.pointwise_ranks)),
        pointwise_c_values=_parse_csv_floats(str(args.pointwise_c_values)),
        pointwise_class_weights=_parse_csv_items(str(args.pointwise_class_weights)),
        pairwise_ranks=_parse_csv_ints(str(args.pairwise_ranks)),
        pairwise_c_values=_parse_csv_floats(str(args.pairwise_c_values)),
        pairwise_losses=_parse_csv_items(str(args.pairwise_losses)),
        whiten_options=_parse_csv_bools(str(args.whiten_options)),
    )
    family_required_features = set()
    for family in family_specs:
        family_required_features.update(str(name) for name in family.feature_names)
    baseline_required_features = set(current_slot100_route.get("feature_names", []))
    if current_slot100_route.get("signal_name"):
        baseline_required_features.add(str(current_slot100_route["signal_name"]))
    required_features = set(family_required_features) | set(baseline_required_features)

    main_store, main_cache_path, main_cache_status = _load_domain_feature_store(
        source_name="cache",
        cache_root=main_cache_root,
        domain_name=DOMAIN_NAME,
        positions=TRAIN_POSITIONS,
        required_feature_names=required_features,
        max_problems_per_cache=max_problems_per_cache,
        feature_workers=int(args.feature_workers),
        chunk_problems=int(args.feature_chunk_problems),
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_feature_cache),
    )
    extra_store, extra_cache_path, extra_cache_status = _load_domain_feature_store(
        source_name="cache_train",
        cache_root=extra_cache_root,
        domain_name=DOMAIN_NAME,
        positions=TRAIN_POSITIONS,
        required_feature_names=required_features,
        max_problems_per_cache=max_problems_per_cache,
        feature_workers=int(args.feature_workers),
        chunk_problems=int(args.feature_chunk_problems),
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_feature_cache),
    )

    coding_full_store = list(main_store) + list(extra_store)
    holdout_problem_map, holdout_split_summary = _build_holdout_problem_map(
        coding_full_store,
        holdout_split=float(args.holdout_split),
        split_seed=int(args.split_seed),
    )
    train_store, holdout_store, full_store = _split_feature_store(
        coding_full_store,
        holdout_problem_map=holdout_problem_map,
    )

    entry_lookup = _build_entry_lookup(main_cache_root=main_cache_root, extra_cache_root=extra_cache_root)
    train_problems = _build_problem_dataset(
        feature_store=train_store,
        family_specs=family_specs,
        entry_lookup=entry_lookup,
        distance_threads=int(args.distance_threads),
    )
    holdout_problems = _build_problem_dataset(
        feature_store=holdout_store,
        family_specs=family_specs,
        entry_lookup=entry_lookup,
        distance_threads=int(args.distance_threads),
    )
    full_problems = _build_problem_dataset(
        feature_store=full_store,
        family_specs=family_specs,
        entry_lookup=entry_lookup,
        distance_threads=int(args.distance_threads),
    )
    main_ds_problems = [
        prob
        for prob in full_problems
        if prob.source_name == "cache" and prob.base_cache_key == "DS-R1/lcb_v5"
    ]
    train_groups = sorted({str(prob.cv_group_key) for prob in train_problems})
    if len(train_groups) < 2:
        raise RuntimeError(
            "Not enough training problems for grouped CV: "
            f"train_problems={len(train_problems)}, unique_groups={len(train_groups)}, "
            f"holdout_problems={len(holdout_problems)}. "
            "Increase --max-problems-per-cache or reduce --holdout-split."
        )

    candidate_specs = _candidate_specs(family_specs, search_grid=search_grid)
    cv_rows: list[dict[str, Any]] = []
    for idx, spec in enumerate(candidate_specs, start=1):
        print(f"[search] {idx}/{len(candidate_specs)} { _candidate_name(spec) }", flush=True)
        row = _crossval_candidate(
            train_problems,
            spec,
            n_splits=int(args.n_splits),
            random_state=int(args.random_state),
        )
        if row is not None:
            cv_rows.append(row)

    if not cv_rows:
        raise RuntimeError("No valid CV candidates were produced")

    selected_cv_rows = _select_cv_winners(cv_rows)
    holdout_rows: list[dict[str, Any]] = []
    holdout_routes: dict[str, dict[str, Any]] = {}
    for row in selected_cv_rows:
        cfg = row["config"]
        spec = CandidateSpec(
            family_name=str(cfg["family_name"]),
            head_kind=str(cfg["head_kind"]),
            feature_names=list(cfg["feature_names"]),
            rank=int(cfg["rank"]),
            whiten=bool(cfg["whiten"]),
            c_value=float(cfg["c_value"]),
            class_weight=None if cfg["class_weight"] is None else str(cfg["class_weight"]),
            loss=None if cfg["loss"] is None else str(cfg["loss"]),
        )
        route = _fit_route(train_problems, spec, random_state=int(args.random_state))
        if route is None:
            continue
        eval_row = _evaluate_route(method_name=row["name"], problems=holdout_problems, route=route)
        eval_row["name"] = row["name"]
        eval_row["family_name"] = row["family_name"]
        eval_row["head_kind"] = row["head_kind"]
        eval_row["config"] = cfg
        holdout_rows.append(eval_row)
        holdout_routes[row["name"]] = route

    if not holdout_rows:
        raise RuntimeError("No holdout candidates were fitted successfully")

    baseline_rows = {
        "code_v2": _evaluate_route(
            method_name="code_v2",
            problems=holdout_problems,
            route=_code_v2_baseline_route(),
        ),
        "current_svd_slot100": _evaluate_route(
            method_name="current_svd_slot100",
            problems=holdout_problems,
            route=current_slot100_route,
        ),
    }

    recommended = _choose_recommended_rows(holdout_rows)
    final_route_configs = {
        "hit1": recommended["hit1"]["config"],
        "pairwise": recommended["pairwise"]["config"],
    }
    final_routes: dict[str, dict[str, Any]] = {}
    final_bundles: dict[str, dict[str, Any]] = {}
    protocol = {
        "main_cache_root": str(main_cache_root),
        "extra_cache_root": str(extra_cache_root),
        "blind_cache_root": str(blind_cache_root),
        "domain": DOMAIN_NAME,
        "position": float(POSITION_VALUE),
        "holdout_split": float(args.holdout_split),
        "split_seed": int(args.split_seed),
        "n_splits": int(args.n_splits),
        "random_state": int(args.random_state),
    }
    for label, cfg in final_route_configs.items():
        spec = CandidateSpec(
            family_name=str(cfg["family_name"]),
            head_kind=str(cfg["head_kind"]),
            feature_names=list(cfg["feature_names"]),
            rank=int(cfg["rank"]),
            whiten=bool(cfg["whiten"]),
            c_value=float(cfg["c_value"]),
            class_weight=None if cfg["class_weight"] is None else str(cfg["class_weight"]),
            loss=None if cfg["loss"] is None else str(cfg["loss"]),
        )
        route = _fit_route(full_problems, spec, random_state=int(args.random_state))
        if route is None:
            raise RuntimeError(f"Full-fit route failed for {label}")
        final_routes[label] = route
        method_id = PRIMARY_METHOD_ID if label == "hit1" else PAIRWISE_METHOD_ID
        final_bundles[label] = _build_single_slot_bundle(method_id=method_id, route=route, protocol=protocol)

    system_proxy: dict[str, Any] = {}
    if not bool(args.skip_system_proxy):
        science_json = _latest_science_json()
        current_bundle, current_cache_metrics, _ = _load_current_system_bundle(
            ds_cache_root=Path(main_cache_root),
            science_json=science_json,
            code_v2_json=Path(CODE_V2_EXHAUSTIVE_JSON),
        )
        for label, cfg in final_route_configs.items():
            spec = CandidateSpec(
                family_name=str(cfg["family_name"]),
                head_kind=str(cfg["head_kind"]),
                feature_names=list(cfg["feature_names"]),
                rank=int(cfg["rank"]),
                whiten=bool(cfg["whiten"]),
                c_value=float(cfg["c_value"]),
                class_weight=None if cfg["class_weight"] is None else str(cfg["class_weight"]),
                loss=None if cfg["loss"] is None else str(cfg["loss"]),
            )
            cv_row = _crossval_candidate(
                main_ds_problems,
                spec,
                n_splits=int(args.n_splits),
                random_state=int(args.random_state),
            )
            candidate_metrics = cv_row["metrics"] if cv_row is not None else _evaluate_route(method_name=label, problems=main_ds_problems, route=final_routes[label])["metrics"]
            candidate_cache_metrics = {cache_key: dict(metrics) for cache_key, metrics in current_cache_metrics.items()}
            candidate_cache_metrics["DS-R1/lcb_v5"] = dict(candidate_metrics)
            candidate_bundle = _combine_cache_metric_proxy(candidate_cache_metrics)
            delta = _system_delta(candidate_bundle, current_bundle)
            system_proxy[label] = {
                "coding_metrics": candidate_metrics,
                "bundle": candidate_bundle,
                "delta": delta,
            }

    blind_feature_cache_path = None
    blind_feature_cache_status = "skipped"
    blind_feature_store: list[dict[str, Any]] = []
    blind_problems: list[ProblemData] = []

    out_model_prefix = (REPO_ROOT / str(args.out_model_prefix)).resolve()
    out_submission_prefix = (REPO_ROOT / str(args.out_submission_prefix)).resolve()
    submissions: dict[str, Any] = {}
    blind_eval: dict[str, Any] = {}
    model_paths: dict[str, str] = {}
    for label, bundle in final_bundles.items():
        model_path = out_model_prefix.with_name(out_model_prefix.name + f"__{label}.pkl")
        save_earlystop_svd_bundle(bundle, model_path)
        model_paths[label] = _display_path(model_path)
        if not bool(args.skip_blind_export):
            if not blind_feature_store:
                blind_required_features = set()
                for _label, _bundle in final_bundles.items():
                    blind_required_features.update(str(name) for name in _bundle["domains"][DOMAIN_NAME]["routes"][0].get("feature_names", []))
                blind_feature_store, blind_feature_cache_path, blind_feature_cache_status = _load_or_build_feature_store(
                    cache_root=blind_cache_root,
                    positions=(float(POSITION_VALUE),),
                    required_feature_names=blind_required_features,
                    max_problems=blind_max_problems_per_cache,
                    reflection_threshold=0.30,
                    workers=int(args.feature_workers),
                    feature_chunk_problems=int(args.feature_chunk_problems),
                    feature_cache_dir=feature_cache_dir,
                    refresh_feature_cache=bool(args.refresh_feature_cache),
                    include_cache_keys=set(TARGET_BLIND_CACHE_KEYS),
                )
                blind_problems = _build_blind_problem_dataset(
                    cache_root=blind_cache_root,
                    feature_store=blind_feature_store,
                    family_specs=family_specs,
                    distance_threads=int(args.distance_threads),
                )
            problem_scores = {}
            for payload in blind_feature_store:
                problem_scores[str(payload["cache_key"])] = _problem_scores_from_payload(
                    payload,
                    slot_index=0,
                    score_fn=lambda domain, position_index, x_raw, _bundle=bundle: _score_problem_with_route(
                        ProblemData(
                            source_name="blind",
                            cache_key=str(payload["cache_key"]),
                            base_cache_key=str(payload["cache_key"]),
                            dataset_name=str(payload["dataset_name"]),
                            problem_id="adhoc",
                            cv_group_key="adhoc",
                            run_ids=[],
                            labels=np.zeros((x_raw.shape[0],), dtype=np.int32),
                            x_raw=np.asarray(x_raw, dtype=np.float64),
                            x_rank=_rank_transform_matrix(np.asarray(x_raw, dtype=np.float64)),
                            D=np.zeros((x_raw.shape[0], x_raw.shape[0]), dtype=np.float64),
                            code_v2_scores=np.zeros((x_raw.shape[0],), dtype=np.float64),
                            x_by_family={
                                str(bundle["domains"][DOMAIN_NAME]["routes"][0]["family_name"]): _build_representation(
                                    x_raw=np.asarray(x_raw, dtype=np.float64),
                                    x_rank=_rank_transform_matrix(np.asarray(x_raw, dtype=np.float64)),
                                    feature_indices=list(bundle["domains"][DOMAIN_NAME]["routes"][0]["feature_indices"]),
                                    representation=REPRESENTATION,
                                )
                            },
                        ),
                        bundle["domains"][DOMAIN_NAME]["routes"][0],
                    ),
                )
            submission_path = out_submission_prefix.with_name(out_submission_prefix.name + f"__{label}.json")
            submissions[label] = _patch_base_submission(
                base_submission_path=base_submission_path,
                out_path=submission_path,
                method_name=PRIMARY_METHOD_ID if label == "hit1" else PAIRWISE_METHOD_ID,
                scores=problem_scores,
            )
            blind_eval[label] = _evaluate_route(
                method_name=label,
                problems=blind_problems,
                route=final_routes[label],
            )

    summary = {
        "created_at_utc": _now_utc(),
        "protocol": protocol,
        "feature_cache": {
            "cache": {
                "status": main_cache_status,
                "path": None if main_cache_path is None else _display_path(main_cache_path),
            },
            "cache_train": {
                "status": extra_cache_status,
                "path": None if extra_cache_path is None else _display_path(extra_cache_path),
            },
            "blind": {
                "status": blind_feature_cache_status,
                "path": None if blind_feature_cache_path is None else _display_path(blind_feature_cache_path),
            },
        },
        "stores": {
            "full": _summarise_feature_store(full_store),
            "train": _summarise_feature_store(train_store),
            "holdout": _summarise_feature_store(holdout_store),
        },
        "run_flags": {
            "max_problems_per_cache": None if max_problems_per_cache is None else int(max_problems_per_cache),
            "blind_max_problems_per_cache": None if blind_max_problems_per_cache is None else int(blind_max_problems_per_cache),
            "families": [family.name for family in family_specs],
            "heads": list(search_grid.head_kinds),
            "pointwise_ranks": list(search_grid.pointwise_ranks),
            "pairwise_ranks": list(search_grid.pairwise_ranks),
            "pointwise_c_values": list(search_grid.pointwise_c_values),
            "pairwise_c_values": list(search_grid.pairwise_c_values),
            "pointwise_class_weights": list(search_grid.pointwise_class_weights),
            "pairwise_losses": list(search_grid.pairwise_losses),
            "whiten_options": [bool(v) for v in search_grid.whiten_options],
            "skip_system_proxy": bool(args.skip_system_proxy),
            "skip_blind_export": bool(args.skip_blind_export),
        },
        "holdout_split": holdout_split_summary,
        "families": [
            {"name": family.name, "feature_names": list(family.feature_names)}
            for family in family_specs
        ],
        "holdout": {
            "baselines": {name: _compact_eval_row(row) for name, row in baseline_rows.items()},
            "candidate_rows": [_compact_eval_row(row) for row in holdout_rows],
        },
        "recommended": {
            "deploy_mode": str(recommended["deploy_mode"]),
            "hit1": _compact_eval_row(recommended["hit1"]),
            "pairwise": _compact_eval_row(recommended["pairwise"]),
            "single": None if recommended.get("single") is None else _compact_eval_row(recommended["single"]),
        },
        "system_proxy": system_proxy,
        "blind_eval": {label: _compact_eval_row(row) for label, row in blind_eval.items()},
        "artifacts": {
            "summary_json": _display_path((REPO_ROOT / str(args.out_summary)).resolve()),
            "eval_json": _display_path((REPO_ROOT / str(args.out_eval)).resolve()),
            "candidate_json": _display_path((REPO_ROOT / str(args.out_candidates)).resolve()),
            "doc_md": _display_path((REPO_ROOT / str(args.out_doc)).resolve()),
            "models": model_paths,
            "submissions": submissions,
        },
    }

    out_candidates = (REPO_ROOT / str(args.out_candidates)).resolve()
    out_candidates.parent.mkdir(parents=True, exist_ok=True)
    out_candidates.write_text(
        json.dumps(
            {
                "cv_rows": cv_rows,
                "selected_cv_rows": selected_cv_rows,
                "holdout_rows": [_compact_eval_row(row) for row in holdout_rows],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    out_eval = (REPO_ROOT / str(args.out_eval)).resolve()
    out_eval.parent.mkdir(parents=True, exist_ok=True)
    out_eval.write_text(
        json.dumps(
            {
                "holdout": summary["holdout"],
                "blind_eval": summary["blind_eval"],
                "system_proxy": system_proxy,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    out_summary = (REPO_ROOT / str(args.out_summary)).resolve()
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    out_doc = (REPO_ROOT / str(args.out_doc)).resolve()
    _write_doc(out_doc, summary)
    _write_registry((REPO_ROOT / str(args.registry_path)).resolve(), summary)

    print(f"Summary written : {_display_path(out_summary)}")
    print(f"Eval written    : {_display_path(out_eval)}")
    print(f"Candidates      : {_display_path(out_candidates)}")
    print(f"Doc written     : {_display_path(out_doc)}")
    for label, info in submissions.items():
        print(f"Submission {label}: {info['path']} | {info['validation']}")


if __name__ == "__main__":
    main()
