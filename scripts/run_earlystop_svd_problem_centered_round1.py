#!/usr/bin/env python3
"""Run EarlyStop strong-feature SVD problem-centered round1 experiments."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.selectors.trajectory_impl import DEFAULT_REFLECTION_THRESHOLD
from nad.ops.earlystop import EARLY_STOP_POSITIONS
from nad.ops.earlystop_svd import (
    FULL_FEATURE_NAMES,
    _auroc,
    _build_representation,
    _cv_auroc_baseline,
    _fit_svd_lr_model,
    _group_folds,
    _rank_transform_matrix,
    load_earlystop_svd_bundle,
    save_earlystop_svd_bundle,
)
from nad.ops.earlystop_svm import load_earlystop_svm_bundle
from scripts.run_earlystop_prefix10_svd_round1 import (
    ANCHOR_POSITIONS,
    DEFAULT_FEATURE_CHUNK_PROBLEMS,
    EXTRACTION_POSITION_INDEX,
    EXTRACTION_POSITIONS,
    OFFICIAL_SLOT_TO_ANCHOR,
    _display_path,
    _fmt_earliest,
    _fmt_pct,
    _metric_ge,
    _now_utc,
    _pct_label,
    build_anchor_bundle,
    build_feature_store,
    choose_best_candidate,
    collect_required_features_for_eval,
    evaluate_method_from_feature_store,
    make_bridge_score_fn,
    make_svd_bundle_score_fn,
    make_tok_conf_score_fn,
    summarise_route,
)
from scripts.run_earlystop_prefix10_svd_round1b import (
    _build_holdout_problem_map,
    _render_cache_table,
    _render_method_table,
    _split_feature_store,
    _summarise_feature_store,
)


STRONG_FEATURE_FAMILY_MAP = {
    "strong_core3": [
        "tok_conf_prefix",
        "tok_conf_recency",
        "traj_reflection_count",
    ],
    "strong_tail5": [
        "tok_conf_prefix",
        "tok_conf_recency",
        "traj_reflection_count",
        "tail_q10",
        "head_tail_gap",
    ],
    "strong_stable6": [
        "tok_conf_prefix",
        "tok_conf_recency",
        "traj_reflection_count",
        "tail_q10",
        "head_tail_gap",
        "tail_variance",
    ],
    "strong_event7": [
        "tok_conf_prefix",
        "tok_conf_recency",
        "traj_reflection_count",
        "tail_q10",
        "head_tail_gap",
        "tail_variance",
        "last_event_tail_conf",
    ],
    "strong_recovery8": [
        "tok_conf_prefix",
        "tok_conf_recency",
        "traj_reflection_count",
        "tail_q10",
        "head_tail_gap",
        "tail_variance",
        "last_event_tail_conf",
        "event_pre_post_delta",
    ],
}

STRONG_BASELINE_SIGNAL_NAMES = (
    "tok_conf_prefix",
    "tok_conf_recency",
    "traj_reflection_count",
    "tail_q10",
    "head_tail_gap",
    "tail_variance",
    "last_event_tail_conf",
    "event_pre_post_delta",
)

CORE_REPRESENTATIONS = (
    "raw",
    "rank",
    "raw+rank",
    "centered_raw",
    "centered_raw+rank",
)
OPTIONAL_REPRESENTATIONS = (
    "zscore_within_problem_raw",
)
SEARCH_RANKS = (2, 4, 6, 8, 12, 16)
SEARCH_C_VALUES = (0.05, 0.10, 0.20, 0.50, 1.00)
SEARCH_WHITEN = (False, True)
SEARCH_CLASS_WEIGHT = ("none", "balanced")
SEARCH_REFLECTION_THRESHOLDS = (0.20, 0.30)

REPO_STATUS_LINES = [
    "round1b 正确协议是：non-coding 使用 `cache + cache_train`，并按 `dataset + problem_id` 做跨 root 一致的 `80/20` holdout；coding 继续 fallback 到旧 v1。",
    "strong-feature 线已经把 family 收窄到 `tok_conf_prefix / tok_conf_recency / traj_reflection_count` 及少量 tail / recovery 信号。",
    "当前 holdout winner `strongfeat_noncoding_anchor4_coding_v1_ref020` 的四个 anchors（10/40/70/100）全部使用 `raw+rank`。",
    "因此，本轮任务不是“直接去掉 rank”，而是检验更偏题内排序的表示是否能超过当前 `raw+rank`。",
]


def _qualify_feature_store(feature_store: list[dict[str, Any]], source_name: str) -> list[dict[str, Any]]:
    qualified: list[dict[str, Any]] = []
    for payload in feature_store:
        item = dict(payload)
        item["source_name"] = str(source_name)
        item["base_cache_key"] = str(payload["cache_key"])
        item["cache_key"] = f"{source_name}/{payload['cache_key']}"
        qualified.append(item)
    return qualified


def _feature_cache_key(
    *,
    source_name: str,
    cache_root: str,
    positions: tuple[float, ...],
    required_feature_names: set[str],
    max_problems_per_cache: Optional[int],
    reflection_threshold: float,
) -> str:
    payload = {
        "version": 1,
        "source_name": str(source_name),
        "cache_root": str(cache_root),
        "positions": [float(p) for p in positions],
        "required_feature_names": sorted(str(v) for v in required_feature_names),
        "max_problems_per_cache": None if max_problems_per_cache is None else int(max_problems_per_cache),
        "reflection_threshold": float(reflection_threshold),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def _feature_cache_path(
    *,
    cache_dir: Path,
    source_name: str,
    cache_root: str,
    positions: tuple[float, ...],
    required_feature_names: set[str],
    max_problems_per_cache: Optional[int],
    reflection_threshold: float,
) -> Path:
    key = _feature_cache_key(
        source_name=source_name,
        cache_root=cache_root,
        positions=positions,
        required_feature_names=required_feature_names,
        max_problems_per_cache=max_problems_per_cache,
        reflection_threshold=reflection_threshold,
    )
    suffix = "all" if max_problems_per_cache is None else f"cap{int(max_problems_per_cache)}"
    thr_tag = f"ref{int(round(float(reflection_threshold) * 100.0)):03d}"
    return cache_dir / f"{source_name}_{suffix}_{thr_tag}_{key}.pkl"


def _load_or_build_qualified_feature_store(
    *,
    source_name: str,
    cache_root: str,
    positions: tuple[float, ...],
    required_feature_names: set[str],
    max_problems_per_cache: Optional[int],
    reflection_threshold: float,
    max_workers: int,
    chunk_problems: int,
    feature_cache_dir: Optional[Path],
    refresh_feature_cache: bool,
) -> tuple[list[dict[str, Any]], Optional[Path], str]:
    cache_path: Optional[Path] = None
    if feature_cache_dir is not None:
        cache_path = _feature_cache_path(
            cache_dir=feature_cache_dir,
            source_name=source_name,
            cache_root=cache_root,
            positions=positions,
            required_feature_names=required_feature_names,
            max_problems_per_cache=max_problems_per_cache,
            reflection_threshold=reflection_threshold,
        )
        if cache_path.exists() and not refresh_feature_cache:
            print(f"[problem-centered] loading feature cache source={source_name} thr={reflection_threshold:.2f} path={cache_path}")
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            return list(payload["feature_store"]), cache_path, "loaded"

    print(f"[problem-centered] building feature store source={source_name} thr={reflection_threshold:.2f} root={cache_root}")
    store = _qualify_feature_store(
        build_feature_store(
            cache_root=cache_root,
            positions=positions,
            required_feature_names=required_feature_names,
            max_problems_per_cache=max_problems_per_cache,
            max_workers=max(1, int(max_workers)),
            chunk_problems=max(1, int(chunk_problems)),
            reflection_threshold=float(reflection_threshold),
        ),
        source_name=source_name,
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(
                {
                    "source_name": str(source_name),
                    "cache_root": str(cache_root),
                    "positions": [float(p) for p in positions],
                    "max_problems_per_cache": None if max_problems_per_cache is None else int(max_problems_per_cache),
                    "reflection_threshold": float(reflection_threshold),
                    "feature_store": store,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        tmp_path.replace(cache_path)
        print(f"[problem-centered] saved feature cache source={source_name} thr={reflection_threshold:.2f} path={cache_path}")
    return store, cache_path, "built"


def _filter_noncoding(feature_store: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [payload for payload in feature_store if payload["domain"] in {"math", "science"}]


def _filter_coding(feature_store: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [payload for payload in feature_store if payload["domain"] == "coding"]


def _rep_tag(representation: str) -> str:
    return (
        str(representation)
        .replace("+", "plus")
        .replace("zscore_within_problem_raw", "zscore_wp_raw")
    )


def _candidate_method_name(representation: str, reflection_threshold: float) -> str:
    thr_tag = f"ref{int(round(float(reflection_threshold) * 100.0)):03d}"
    return f"problem_centered_{_rep_tag(representation)}_noncoding_anchor4_coding_v1_{thr_tag}"


def _build_scope_training_tables_with_rank_groups(
    feature_store: list[dict[str, Any]],
    positions: tuple[float, ...],
) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    rows: dict[str, dict[int, list[np.ndarray]]] = {
        "global": {idx: [] for idx in range(len(positions))},
        "noncoding": {idx: [] for idx in range(len(positions))},
    }
    labels: dict[str, dict[int, list[np.ndarray]]] = {
        "global": {idx: [] for idx in range(len(positions))},
        "noncoding": {idx: [] for idx in range(len(positions))},
    }
    rank_groups: dict[str, dict[int, list[np.ndarray]]] = {
        "global": {idx: [] for idx in range(len(positions))},
        "noncoding": {idx: [] for idx in range(len(positions))},
    }
    cv_groups: dict[str, dict[int, list[np.ndarray]]] = {
        "global": {idx: [] for idx in range(len(positions))},
        "noncoding": {idx: [] for idx in range(len(positions))},
    }

    position_indices = [EXTRACTION_POSITION_INDEX[float(position)] for position in positions]
    for payload in feature_store:
        tensor = payload["tensor"]
        if tensor.shape[0] == 0:
            continue
        y = payload["labels"]
        local_rank_groups = payload["group_keys"]
        local_cv_groups = payload["cv_group_keys"]
        for local_pos_idx, src_pos_idx in enumerate(position_indices):
            x_raw = tensor[:, src_pos_idx, :]
            rows["global"][local_pos_idx].append(x_raw)
            labels["global"][local_pos_idx].append(y)
            rank_groups["global"][local_pos_idx].append(local_rank_groups)
            cv_groups["global"][local_pos_idx].append(local_cv_groups)
            if payload["domain"] in {"math", "science"}:
                rows["noncoding"][local_pos_idx].append(x_raw)
                labels["noncoding"][local_pos_idx].append(y)
                rank_groups["noncoding"][local_pos_idx].append(local_rank_groups)
                cv_groups["noncoding"][local_pos_idx].append(local_cv_groups)

    out: dict[str, dict[int, dict[str, np.ndarray]]] = {"global": {}, "noncoding": {}}
    for scope in ("global", "noncoding"):
        for pos_idx in range(len(positions)):
            if rows[scope][pos_idx]:
                x_raw = np.vstack(rows[scope][pos_idx]).astype(np.float64, copy=False)
                y = np.concatenate(labels[scope][pos_idx]).astype(np.int32, copy=False)
                groups_rank = np.concatenate(rank_groups[scope][pos_idx]).astype(object, copy=False)
                groups_cv = np.concatenate(cv_groups[scope][pos_idx]).astype(object, copy=False)
            else:
                x_raw = np.zeros((0, len(FULL_FEATURE_NAMES)), dtype=np.float64)
                y = np.zeros((0,), dtype=np.int32)
                groups_rank = np.asarray([], dtype=object)
                groups_cv = np.asarray([], dtype=object)

            x_rank = np.zeros_like(x_raw)
            if x_raw.shape[0] > 0:
                by_rank_group: dict[Any, list[int]] = {}
                for row_idx, group_key in enumerate(groups_rank.tolist()):
                    by_rank_group.setdefault(group_key, []).append(row_idx)
                for group_rows in by_rank_group.values():
                    x_rank[group_rows] = _rank_transform_matrix(x_raw[group_rows])

            out[scope][pos_idx] = {
                "x_raw": x_raw,
                "x_rank": x_rank,
                "y": y,
                "groups_cv": groups_cv,
                "groups_rank": groups_rank,
            }
    return out


def _center_within_problem_matrix(x_raw: np.ndarray, groups_rank: np.ndarray) -> np.ndarray:
    if x_raw.shape[0] == 0:
        return np.zeros_like(x_raw, dtype=np.float64)
    out = np.zeros_like(x_raw, dtype=np.float64)
    by_group: dict[Any, list[int]] = {}
    for row_idx, group_key in enumerate(groups_rank.tolist()):
        by_group.setdefault(group_key, []).append(row_idx)
    for group_rows in by_group.values():
        sub = x_raw[group_rows]
        out[group_rows] = sub - np.mean(sub, axis=0, keepdims=True)
    return out


def _zscore_within_problem_matrix(x_raw: np.ndarray, groups_rank: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if x_raw.shape[0] == 0:
        return np.zeros_like(x_raw, dtype=np.float64)
    out = np.zeros_like(x_raw, dtype=np.float64)
    by_group: dict[Any, list[int]] = {}
    for row_idx, group_key in enumerate(groups_rank.tolist()):
        by_group.setdefault(group_key, []).append(row_idx)
    for group_rows in by_group.values():
        sub = x_raw[group_rows]
        mean = np.mean(sub, axis=0, keepdims=True)
        std = np.std(sub, axis=0, keepdims=True)
        std = np.where(std < float(eps), 1.0, std)
        out[group_rows] = (sub - mean) / std
    return out


def _build_representation_groupaware(
    *,
    x_raw: np.ndarray,
    x_rank: np.ndarray,
    groups_rank: np.ndarray,
    feature_indices: list[int],
    representation: str,
) -> np.ndarray:
    if representation in {"raw", "rank", "raw+rank"}:
        return _build_representation(
            x_raw=x_raw,
            x_rank=x_rank,
            feature_indices=feature_indices,
            representation=representation,
        )
    if representation == "centered_raw":
        x_centered = _center_within_problem_matrix(x_raw, groups_rank)
        return x_centered[:, feature_indices]
    if representation == "centered_raw+rank":
        x_centered = _center_within_problem_matrix(x_raw, groups_rank)
        return np.concatenate([
            x_centered[:, feature_indices],
            x_rank[:, feature_indices],
        ], axis=1)
    if representation == "zscore_within_problem_raw":
        x_z = _zscore_within_problem_matrix(x_raw, groups_rank)
        return x_z[:, feature_indices]
    raise ValueError(f"Unknown representation: {representation}")


def _fit_route_for_table_problem_centered(
    *,
    x_raw: np.ndarray,
    x_rank: np.ndarray,
    y: np.ndarray,
    groups_cv: np.ndarray,
    groups_rank: np.ndarray,
    position: float,
    training_scope: str,
    reflection_threshold: float,
    representation: str,
    n_splits: int,
    random_state: int,
) -> dict[str, Any]:
    if x_raw.shape[0] == 0 or np.unique(y).shape[0] < 2:
        return {
            "route_type": "baseline",
            "signal_name": "tok_conf_prefix",
            "cv_auroc": float("nan"),
            "n_valid_folds": 0,
            "training_position": float(position),
            "training_scope": str(training_scope),
            "reflection_threshold": float(reflection_threshold),
            "representation": str(representation),
            "note": "insufficient labeled data",
        }

    baseline_candidates = [name for name in STRONG_BASELINE_SIGNAL_NAMES if name in FULL_FEATURE_NAMES]
    best_baseline = {
        "signal_name": "tok_conf_prefix",
        "cv_auroc": float("-inf"),
        "n_valid_folds": 0,
    }
    feature_to_idx = {name: idx for idx, name in enumerate(FULL_FEATURE_NAMES)}
    for signal_name in baseline_candidates:
        score_col = feature_to_idx[signal_name]
        score_vec = x_raw[:, score_col]
        cv_auc, n_folds = _cv_auroc_baseline(
            scores=score_vec,
            y=y,
            groups=groups_cv,
            n_splits=n_splits,
        )
        if np.isfinite(cv_auc) and cv_auc > float(best_baseline["cv_auroc"]):
            best_baseline = {
                "signal_name": str(signal_name),
                "cv_auroc": float(cv_auc),
                "n_valid_folds": int(n_folds),
            }

    best_svd: dict[str, Any] = {"cv_auroc": float("-inf")}
    folds = _group_folds(groups_cv, n_splits=n_splits)
    for family_name, family_features in STRONG_FEATURE_FAMILY_MAP.items():
        feature_indices = [feature_to_idx[name] for name in family_features]
        x_rep = _build_representation_groupaware(
            x_raw=x_raw,
            x_rank=x_rank,
            groups_rank=groups_rank,
            feature_indices=feature_indices,
            representation=representation,
        )
        if not folds:
            continue

        candidate_scores: dict[tuple[int, float, bool, str], list[float]] = {}
        for train_idx, test_idx in folds:
            y_train = y[train_idx]
            y_test = y[test_idx]
            if np.unique(y_train).shape[0] < 2 or np.unique(y_test).shape[0] < 2:
                continue

            scaler = StandardScaler(with_mean=True, with_std=True)
            x_train_scaled = scaler.fit_transform(x_rep[train_idx])
            x_test_scaled = scaler.transform(x_rep[test_idx])

            max_rank = min(int(max(SEARCH_RANKS)), int(x_train_scaled.shape[1]), int(x_train_scaled.shape[0] - 1))
            if max_rank < 1:
                continue

            svd = TruncatedSVD(n_components=max_rank, random_state=int(random_state))
            z_train_full = svd.fit_transform(x_train_scaled)
            z_test_full = svd.transform(x_test_scaled)
            singular_values = np.asarray(svd.singular_values_, dtype=np.float64)
            singular_values = np.where(np.abs(singular_values) < 1e-8, 1.0, singular_values)

            valid_ranks = [int(rank) for rank in SEARCH_RANKS if int(rank) <= max_rank]
            for rank in valid_ranks:
                z_train = z_train_full[:, :rank]
                z_test = z_test_full[:, :rank]
                for whiten in SEARCH_WHITEN:
                    if whiten:
                        scale = singular_values[:rank]
                        z_train_use = z_train / scale
                        z_test_use = z_test / scale
                    else:
                        z_train_use = z_train
                        z_test_use = z_test
                    for c_value in SEARCH_C_VALUES:
                        for class_weight in SEARCH_CLASS_WEIGHT:
                            clf = LogisticRegression(
                                C=float(c_value),
                                class_weight=None if class_weight == "none" else "balanced",
                                max_iter=2000,
                                random_state=int(random_state),
                            )
                            try:
                                clf.fit(z_train_use, y_train)
                                scores = np.asarray(clf.decision_function(z_test_use), dtype=np.float64)
                            except Exception:
                                continue
                            fold_auc = _auroc(scores, y_test)
                            if np.isfinite(fold_auc):
                                candidate_scores.setdefault(
                                    (int(rank), float(c_value), bool(whiten), str(class_weight)),
                                    [],
                                ).append(float(fold_auc))

        for (rank, c_value, whiten, class_weight), values in candidate_scores.items():
            if not values:
                continue
            cv_auc = float(np.mean(values))
            n_folds = int(len(values))
            if cv_auc > float(best_svd["cv_auroc"]):
                best_svd = {
                    "cv_auroc": float(cv_auc),
                    "n_valid_folds": int(n_folds),
                    "family_name": str(family_name),
                    "representation": str(representation),
                    "rank": int(rank),
                    "c_value": float(c_value),
                    "whiten": bool(whiten),
                    "class_weight": str(class_weight),
                    "feature_names": list(family_features),
                    "feature_indices": list(feature_indices),
                }

    baseline_auc = float(best_baseline["cv_auroc"]) if np.isfinite(best_baseline["cv_auroc"]) else float("-inf")
    svd_auc = float(best_svd["cv_auroc"]) if np.isfinite(best_svd["cv_auroc"]) else float("-inf")

    if svd_auc > baseline_auc and np.isfinite(svd_auc):
        x_rep_full = _build_representation_groupaware(
            x_raw=x_raw,
            x_rank=x_rank,
            groups_rank=groups_rank,
            feature_indices=best_svd["feature_indices"],
            representation=best_svd["representation"],
        )
        model = _fit_svd_lr_model(
            x=x_rep_full,
            y=y,
            rank=best_svd["rank"],
            c_value=best_svd["c_value"],
            whiten=best_svd["whiten"],
            class_weight_name=best_svd["class_weight"],
            random_state=random_state,
        )
        if model is not None:
            return {
                "route_type": "svd",
                "cv_auroc": float(best_svd["cv_auroc"]),
                "n_valid_folds": int(best_svd["n_valid_folds"]),
                "family_name": str(best_svd["family_name"]),
                "representation": str(best_svd["representation"]),
                "rank": int(best_svd["rank"]),
                "c_value": float(best_svd["c_value"]),
                "whiten": bool(best_svd["whiten"]),
                "class_weight": str(best_svd["class_weight"]),
                "feature_names": list(best_svd["feature_names"]),
                "feature_indices": list(best_svd["feature_indices"]),
                "baseline_signal_name": best_baseline["signal_name"],
                "baseline_cv_auroc": float(best_baseline["cv_auroc"]),
                "training_position": float(position),
                "training_scope": str(training_scope),
                "reflection_threshold": float(reflection_threshold),
                "model": model,
            }

    return {
        "route_type": "baseline",
        "signal_name": best_baseline["signal_name"],
        "cv_auroc": float(best_baseline["cv_auroc"]),
        "n_valid_folds": int(best_baseline["n_valid_folds"]),
        "svd_best_cv_auroc": None if not np.isfinite(svd_auc) else float(svd_auc),
        "training_position": float(position),
        "training_scope": str(training_scope),
        "reflection_threshold": float(reflection_threshold),
        "representation": str(representation),
    }


def _train_routes_for_scope_problem_centered(
    *,
    tables: list[dict[str, np.ndarray]],
    positions: tuple[float, ...],
    scope_name: str,
    representation: str,
    reflection_threshold: float,
    n_splits: int,
    random_state: int,
    max_workers: int,
) -> dict[float, dict[str, Any]]:
    routes: dict[float, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        future_map = {}
        for pos_idx, position in enumerate(positions):
            future = executor.submit(
                _fit_route_for_table_problem_centered,
                x_raw=tables[pos_idx]["x_raw"],
                x_rank=tables[pos_idx]["x_rank"],
                y=tables[pos_idx]["y"],
                groups_cv=tables[pos_idx]["groups_cv"],
                groups_rank=tables[pos_idx]["groups_rank"],
                position=float(position),
                training_scope=str(scope_name),
                reflection_threshold=float(reflection_threshold),
                representation=str(representation),
                n_splits=int(n_splits),
                random_state=int(random_state),
            )
            future_map[future] = float(position)

        for future in as_completed(future_map):
            position = future_map[future]
            route = future.result()
            routes[float(position)] = route
            if route["route_type"] == "svd":
                print(
                    f"[problem-centered] scope={scope_name:<24s} pos={_pct_label(position):>4s} "
                    f"thr={reflection_threshold:.2f} rep={representation:<24s} route=svd auc={route['cv_auroc']:.4f} "
                    f"family={route['family_name']}"
                )
            else:
                print(
                    f"[problem-centered] scope={scope_name:<24s} pos={_pct_label(position):>4s} "
                    f"thr={reflection_threshold:.2f} rep={representation:<24s} route=baseline auc={route['cv_auroc']:.4f} "
                    f"signal={route['signal_name']}"
                )
    return routes


def _annotate_bundle_thresholds(
    bundle: dict[str, Any],
    *,
    noncoding_threshold: float,
    coding_threshold: Optional[float] = None,
) -> dict[str, Any]:
    out = copy.deepcopy(bundle)
    for domain_name, domain_bundle in out["domains"].items():
        if domain_name in {"math", "science"}:
            threshold = float(noncoding_threshold)
        elif coding_threshold is not None:
            threshold = float(coding_threshold)
        else:
            threshold = float(noncoding_threshold)
        for route in domain_bundle["routes"]:
            route["reflection_threshold"] = float(threshold)
    out["reflection_threshold_by_domain"] = {
        "math": float(noncoding_threshold),
        "science": float(noncoding_threshold),
        "coding": float(coding_threshold if coding_threshold is not None else noncoding_threshold),
    }
    return out


def _evaluate_bestofn_cache_slot100(
    *,
    payload: dict[str, Any],
    score_fn,
) -> dict[str, Any]:
    slot100_position = 1.0
    slot100_src_idx = EXTRACTION_POSITION_INDEX[slot100_position]
    slot100_out_idx = list(EARLY_STOP_POSITIONS).index(slot100_position)

    tensor = payload["tensor"]
    labels_all = payload["labels"]
    sample_ids_all = payload["sample_ids"]
    problem_ids = payload["problem_ids"]
    problem_offsets = payload["problem_offsets"]

    all_scores: list[float] = []
    all_labels: list[int] = []
    hit1_total = 0.0
    hit3_total = 0.0
    pairwise_num = 0.0
    pairwise_den = 0.0
    n_problems = 0

    for problem_idx, _problem_id in enumerate(problem_ids):
        start = int(problem_offsets[problem_idx])
        end = int(problem_offsets[problem_idx + 1])
        if end <= start:
            continue

        x_raw = tensor[start:end, slot100_src_idx, :]
        labels = np.asarray(labels_all[start:end], dtype=np.int32)
        sample_ids = np.asarray(sample_ids_all[start:end], dtype=np.int32)
        scores = np.asarray(score_fn(payload["domain"], slot100_out_idx, x_raw), dtype=np.float64)
        order = np.lexsort((sample_ids, -scores))

        hit1_total += float(labels[order[0]] > 0)
        hit3_total += float(np.any(labels[order[: min(3, order.size)]] > 0))

        pos_scores = scores[labels > 0]
        neg_scores = scores[labels <= 0]
        if pos_scores.size > 0 and neg_scores.size > 0:
            diff = pos_scores[:, None] - neg_scores[None, :]
            pairwise_num += float((diff > 0).sum()) + 0.5 * float((diff == 0).sum())
            pairwise_den += float(diff.size)

        all_scores.extend(scores.tolist())
        all_labels.extend(labels.tolist())
        n_problems += 1

    labels_arr = np.asarray(all_labels, dtype=np.int32)
    scores_arr = np.asarray(all_scores, dtype=np.float64)
    top10_count = max(1, int(math.ceil(0.10 * labels_arr.size))) if labels_arr.size > 0 else 0
    selacc10 = 0.0
    if labels_arr.size > 0:
        order = np.argsort(-scores_arr, kind="mergesort")
        selacc10 = float(labels_arr[order[:top10_count]].mean())

    return {
        "cache_key": str(payload["cache_key"]),
        "dataset_name": str(payload["dataset_name"]),
        "domain": str(payload["domain"]),
        "n_problems": int(n_problems),
        "n_samples": int(labels_arr.size),
        "top10_count": int(top10_count),
        "hit@1": float(hit1_total / n_problems) if n_problems else 0.0,
        "hit@3": float(hit3_total / n_problems) if n_problems else 0.0,
        "selacc@10": float(selacc10),
        "pairwise": float(pairwise_num / pairwise_den) if pairwise_den > 0 else None,
    }


def _aggregate_bestofn_bridge(by_cache: list[dict[str, Any]]) -> dict[str, Any]:
    if not by_cache:
        return {
            "num_caches": 0,
            "sample_weighted": {
                "hit@1": 0.0,
                "hit@3": 0.0,
                "selacc@10": 0.0,
                "pairwise": None,
            },
            "equal_cache_mean": {
                "hit@1": 0.0,
                "hit@3": 0.0,
                "selacc@10": 0.0,
                "pairwise": None,
            },
        }

    total_problem_weight = sum(int(row["n_problems"]) for row in by_cache)
    total_sample_weight = sum(int(row["n_samples"]) for row in by_cache)
    total_top10_weight = sum(int(row["top10_count"]) for row in by_cache)

    pairwise_weighted_num = 0.0
    pairwise_weighted_den = 0.0
    pairwise_equal_values: list[float] = []
    for row in by_cache:
        pairwise = row.get("pairwise")
        if pairwise is None or not np.isfinite(float(pairwise)):
            continue
        pairwise_weighted_num += float(pairwise) * float(row["n_samples"])
        pairwise_weighted_den += float(row["n_samples"])
        pairwise_equal_values.append(float(pairwise))

    sample_weighted = {
        "hit@1": float(sum(float(row["hit@1"]) * int(row["n_problems"]) for row in by_cache) / max(total_problem_weight, 1)),
        "hit@3": float(sum(float(row["hit@3"]) * int(row["n_problems"]) for row in by_cache) / max(total_problem_weight, 1)),
        "selacc@10": float(sum(float(row["selacc@10"]) * int(row["top10_count"]) for row in by_cache) / max(total_top10_weight, 1)),
        "pairwise": None if pairwise_weighted_den <= 0 else float(pairwise_weighted_num / pairwise_weighted_den),
        "n_problems": int(total_problem_weight),
        "n_samples": int(total_sample_weight),
        "top10_count": int(total_top10_weight),
    }

    equal_cache_mean = {
        "hit@1": float(np.mean([float(row["hit@1"]) for row in by_cache])),
        "hit@3": float(np.mean([float(row["hit@3"]) for row in by_cache])),
        "selacc@10": float(np.mean([float(row["selacc@10"]) for row in by_cache])),
        "pairwise": None if not pairwise_equal_values else float(np.mean(pairwise_equal_values)),
    }
    return {
        "num_caches": int(len(by_cache)),
        "sample_weighted": sample_weighted,
        "equal_cache_mean": equal_cache_mean,
    }


def evaluate_slot100_bestofn_bridge_from_feature_store(
    *,
    method_name: str,
    feature_store: list[dict[str, Any]],
    score_fn,
) -> dict[str, Any]:
    by_cache: list[dict[str, Any]] = []
    print(f"[bridge] start method={method_name}")
    for payload in feature_store:
        print(f"[bridge]   method={method_name} cache={payload['cache_key']}")
        by_cache.append(_evaluate_bestofn_cache_slot100(payload=payload, score_fn=score_fn))
    print(f"[bridge] done method={method_name}")
    return {
        "method_name": str(method_name),
        "aggregate": _aggregate_bestofn_bridge(by_cache),
        "by_cache": by_cache,
    }


def _route_summary_lines(title: str, route_map: dict[float, dict[str, Any]]) -> list[str]:
    lines = [f"### {title}", "", "| Anchor | Route | Detail |", "|---|---|---|"]
    for position in ANCHOR_POSITIONS:
        route = route_map[float(position)]
        if route["route_type"] == "svd":
            detail = (
                f"{route['family_name']} / {route['representation']} / rank={route['rank']} / "
                f"C={route['c_value']} / whiten={route['whiten']} / thr={route['reflection_threshold']:.2f}"
            )
        else:
            detail = f"{route['signal_name']} / thr={route['reflection_threshold']:.2f}"
        lines.append(f"| {_pct_label(position)} | {route['route_type']} | {detail} |")
    lines.append("")
    return lines


def _render_representation_holdout_table(
    *,
    title: str,
    rows: list[dict[str, Any]],
) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Representation | Threshold | Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        agg = row["aggregate"]
        lines.append(
            "| {rep} | {thr:.2f} | {name} | {auc_auroc} | {auc_selacc} | {earliest} | {auroc100} | {stop100} |".format(
                rep=row["representation"],
                thr=float(row["reflection_threshold"]),
                name=row["method_name"],
                auc_auroc=_fmt_pct(agg["auc_of_auroc"]),
                auc_selacc=_fmt_pct(agg["auc_of_selacc"]),
                earliest=_fmt_earliest(agg.get("earliest_gt_0.6")),
                auroc100=_fmt_pct(agg["auroc@100%"]),
                stop100=_fmt_pct(agg["stop_acc@100%"]),
            )
        )
    lines.append("")
    return lines


def _fmt_bridge_pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if not np.isfinite(float(value)):
        return "N/A"
    return f"{100.0 * float(value):.2f}%"


def _render_bridge_table(
    *,
    title: str,
    rows: list[dict[str, Any]],
) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Method | Hit@1 | Hit@3 | SelAcc@10 | Pairwise |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        agg = row["aggregate"]["sample_weighted"]
        lines.append(
            "| {name} | {hit1} | {hit3} | {selacc10} | {pairwise} |".format(
                name=row["method_name"],
                hit1=_fmt_bridge_pct(agg.get("hit@1")),
                hit3=_fmt_bridge_pct(agg.get("hit@3")),
                selacc10=_fmt_bridge_pct(agg.get("selacc@10")),
                pairwise=_fmt_bridge_pct(agg.get("pairwise")),
            )
        )
    lines.append("")
    return lines


def _write_plan_doc(path: Path, representations: tuple[str, ...]) -> None:
    lines = [
        "# EARLYSTOP SVD PROBLEM-CENTERED ROUND1 PLAN (2026-04-09)",
        "",
        "## 核心结论（先验确认）",
        "",
    ]
    for line in REPO_STATUS_LINES:
        lines.append(f"- {line}")

    lines.extend([
        "",
        "## 目标",
        "",
        "- 在 strong-feature family + round1b 正确协议下，明确检验题内中心化表示是否优于 `raw+rank`。",
        "- 不做宽特征回退，不改 coding fallback，不看 `cache_test` 标签做选型。",
        "",
        "## 训练协议",
        "",
        "- non-coding：`cache + cache_train`，按 `dataset + problem_id` 跨 root 一致 `80/20` holdout。",
        "- coding：固定 fallback 到 `earlystop_svd_lowrank_lr_v1`。",
        "- 结构：`10/40/70/100` anchor4 训练 + official 10 slots 路由。",
        "",
        "## 表示与特征",
        "",
        f"- 表示搜索：{', '.join(f'`{rep}`' for rep in representations)}。",
        "- 特征家族仅：`strong_core3` / `strong_tail5` / `strong_stable6` / `strong_event7` / `strong_recovery8`。",
        "- reflection threshold：`0.20`, `0.30`。",
        "",
        "## 评测口径",
        "",
        "- A. EarlyStop：`AUC of AUROC`、`AUC of SelAcc`、`Earliest > 0.6`、`AUROC@100%`、`Stop Acc@100%`。",
        "- B. Slot100 -> Best-of-N bridge：`Hit@1`、`Hit@3`、`SelAcc@10`、`Pairwise`。",
        "- bridge 评测范围：`Holdout+Coding`（non-coding holdout + coding fallback）。",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_results_doc(path: Path, summary: dict[str, Any], changed_files: list[str]) -> None:
    holdout_rows = list(summary["holdout_eval"]["candidates"].values())
    holdout_rows.sort(
        key=lambda row: (
            -float(row["aggregate"]["auc_of_selacc"]),
            -float(row["aggregate"]["auroc@100%"]),
            row["method_name"],
        )
    )
    bridge_rows = list(summary["bridge_eval"]["candidates"].values())
    bridge_rows.sort(
        key=lambda row: (
            -(float(row["aggregate"]["sample_weighted"]["hit@1"]) if row["aggregate"]["sample_weighted"]["hit@1"] is not None else -1.0),
            -(float(row["aggregate"]["sample_weighted"]["pairwise"]) if row["aggregate"]["sample_weighted"]["pairwise"] is not None else -1.0),
            row["method_name"],
        )
    )

    winner_name = summary["decision"]["winner_method_name"]
    winner_meta = summary["winner_meta"]
    winner_holdout = summary["holdout_eval"]["candidates"][winner_name]
    winner_bridge = summary["bridge_eval"]["candidates"][winner_name]
    baseline_bridge = summary["bridge_eval"]["baselines"]["slot100_strongfeat_current"]
    rep_best_rows = list(summary["holdout_eval"]["best_by_representation"].values())
    rep_best_bridge_rows = list(summary["bridge_eval"]["best_by_representation"].values())
    rep_best_rows.sort(key=lambda row: row["representation"])
    rep_best_bridge_rows.sort(key=lambda row: row["representation"])

    qa = summary["qa_answers"]
    lines: list[str] = [
        "# EARLYSTOP SVD PROBLEM-CENTERED ROUND1 RESULTS (2026-04-09)",
        "",
        "## 开头确认结论",
        "",
    ]
    for line in REPO_STATUS_LINES:
        lines.append(f"- {line}")

    lines.extend([
        "",
        "## 1. 你确认的当前 repo 状态",
        "",
        f"- `main root`：`{summary['protocol']['main_cache_root']}`。",
        f"- `extra root`：`{summary['protocol']['extra_cache_root']}`。",
        f"- `holdout split`：`{int(round((1.0 - float(summary['protocol']['holdout_split'])) * 100))}/{int(round(float(summary['protocol']['holdout_split']) * 100))}` 按 `dataset + problem_id`。",
        f"- `max_problems_per_cache`：`{summary['config']['max_problems_per_cache']}`。",
        "",
        "## 2. 为什么本轮不是“删除 rank”",
        "",
        "- 目标是检验题内排序表示是否有增益，而不是先验假设 rank 一定无效。",
        "- 因此本轮做并行对照：`raw / rank / raw+rank / centered_raw / centered_raw+rank`（可选再加 `zscore_within_problem_raw`）。",
        "",
        "## 3. 你新增了哪些表示",
        "",
        f"- 本轮实际表示集合：{', '.join(f'`{rep}`' for rep in summary['config']['representations'])}。",
        "- `centered_raw` 定义：每题（64 runs）每个 slot 下按特征减去该题均值。",
        "",
        "## 4. EarlyStop 结果",
        "",
    ])

    holdout_baselines = [
        summary["holdout_eval"]["baselines"]["tok_conf_prefix_mean_v1"],
        summary["holdout_eval"]["baselines"]["earlystop_from_bestofn_svm_bridge_v1"],
        summary["holdout_eval"]["baselines"]["earlystop_svd_lowrank_lr_v1"],
        summary["holdout_eval"]["baselines"]["earlystop_prefix10_svd_round1b_cap8"],
        summary["holdout_eval"]["baselines"]["earlystop_strongfeat_round1_cap8"],
    ]
    lines.extend(_render_method_table("holdout baselines", holdout_baselines))
    lines.extend(_render_representation_holdout_table(title="holdout candidates (all representation x threshold)", rows=holdout_rows))
    lines.extend(_render_representation_holdout_table(title="holdout best-by-representation", rows=rep_best_rows))
    lines.extend(_render_cache_table("holdout winner per-cache", winner_holdout))

    lines.extend([
        "## 5. Slot100 -> BestofN bridge 结果",
        "",
    ])
    bridge_baselines = [
        summary["bridge_eval"]["baselines"]["slot100_strongfeat_current"],
    ]
    lines.extend(_render_bridge_table(title="slot100 bridge baseline", rows=bridge_baselines))
    lines.extend(_render_bridge_table(title="slot100 bridge candidates (all representation x threshold)", rows=bridge_rows))
    lines.extend(_render_bridge_table(title="slot100 bridge best-by-representation", rows=rep_best_bridge_rows))

    lines.extend([
        "## 6. 最终结论：最值得保留的表示是什么",
        "",
        f"- holdout winner：`{winner_name}`（representation=`{winner_meta['representation']}`，thr=`{winner_meta['reflection_threshold']:.2f}`）。",
        f"- 对应 bridge（sample-weighted）：Hit@1={_fmt_bridge_pct(winner_bridge['aggregate']['sample_weighted']['hit@1'])}，Hit@3={_fmt_bridge_pct(winner_bridge['aggregate']['sample_weighted']['hit@3'])}，SelAcc@10={_fmt_bridge_pct(winner_bridge['aggregate']['sample_weighted']['selacc@10'])}，Pairwise={_fmt_bridge_pct(winner_bridge['aggregate']['sample_weighted']['pairwise'])}。",
        "",
        "## 7. 如果没有胜过当前 strongest winner，失败原因是什么",
        "",
        f"- 是否在 EarlyStop holdout 上超过 `earlystop_strongfeat_round1_cap8`：`{'YES' if summary['decision']['holdout_beats_strongfeat'] else 'NO'}`。",
        f"- 是否在 slot100 bridge 上超过当前 baseline：`{'YES' if summary['decision']['bridge_beats_slot100_baseline'] else 'NO'}`。",
        f"- 判定说明：{summary['decision']['reason']}",
        "",
        "## 8. 必答问题（Q1~Q4）",
        "",
        f"1. 直接去掉 rank 是否有帮助？**{qa['q1_remove_rank_helpful']}**",
        f"2. `problem-centered raw` 是否比 `raw+rank` 更好？**{qa['q2_centered_raw_vs_rawplusrank']}**",
        f"3. strongest setting 更偏 global correctness 还是 within-problem ranking？**{qa['q3_global_vs_within']}**",
        f"4. 新表示是否改善 `slot100` bridge 的 Best-of-N 表现？**{qa['q4_bridge_improvement']}**",
        "",
        "## 9. 改了哪些文件",
        "",
    ])
    for file_path in changed_files:
        lines.append(f"- `{file_path}`")

    lines.extend([
        "",
        "## 10. 如何复跑",
        "",
        "```bash",
        "bash cookbook/00_setup/verify.sh",
        "python3 scripts/run_earlystop_svd_problem_centered_round1.py \\",
        "  --main-cache-root MUI_HUB/cache \\",
        "  --extra-cache-root /home/jovyan/public-ro/MUI_HUB/cache_train \\",
        "  --max-problems-per-cache 8",
        "```",
        "",
        "```bash",
        "python3 scripts/run_earlystop_svd_problem_centered_round1.py \\",
        "  --main-cache-root MUI_HUB/cache \\",
        "  --extra-cache-root /home/jovyan/public-ro/MUI_HUB/cache_train \\",
        "  --max-problems-per-cache 0 \\",
        "  --run-full-pass",
        "```",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _qa_answers(summary: dict[str, Any]) -> dict[str, str]:
    rep_best_holdout = summary["holdout_eval"]["best_by_representation"]
    rep_best_bridge = summary["bridge_eval"]["best_by_representation"]
    raw = rep_best_holdout.get("raw")
    raw_rank = rep_best_holdout.get("raw+rank")
    centered_raw = rep_best_holdout.get("centered_raw")
    centered_raw_rank = rep_best_holdout.get("centered_raw+rank")
    baseline_bridge = summary["bridge_eval"]["baselines"]["slot100_strongfeat_current"]["aggregate"]["sample_weighted"]

    raw_sel = float(raw["aggregate"]["auc_of_selacc"]) if raw is not None else float("-inf")
    raw_rank_sel = float(raw_rank["aggregate"]["auc_of_selacc"]) if raw_rank is not None else float("-inf")
    centered_raw_sel = float(centered_raw["aggregate"]["auc_of_selacc"]) if centered_raw is not None else float("-inf")
    centered_best = centered_raw
    if centered_raw_rank is not None and centered_raw is not None:
        if float(centered_raw_rank["aggregate"]["auc_of_selacc"]) > float(centered_raw["aggregate"]["auc_of_selacc"]):
            centered_best = centered_raw_rank
    elif centered_raw_rank is not None:
        centered_best = centered_raw_rank

    centered_best_sel = float(centered_best["aggregate"]["auc_of_selacc"]) if centered_best is not None else float("-inf")
    centered_best_bridge = rep_best_bridge.get(centered_best["representation"]) if centered_best is not None else None

    q1 = "YES，去掉 rank 在 holdout 上更好。" if raw_sel > raw_rank_sel else "NO，去掉 rank 未带来稳定收益。"
    q2 = "YES，`centered_raw` 超过了 `raw+rank`。" if centered_raw_sel > raw_rank_sel else "NO，`centered_raw` 未超过 `raw+rank`。"

    q3 = (
        "当前 strongest setting 仍更偏 global correctness，within-problem 显式编码增益有限。"
        if centered_best_sel <= raw_rank_sel
        else "当前 strongest setting 已出现更强 within-problem ranking 信号。"
    )

    if centered_best_bridge is not None:
        centered_hit1 = float(centered_best_bridge["aggregate"]["sample_weighted"]["hit@1"])
        baseline_hit1 = float(baseline_bridge["hit@1"])
        centered_pair = centered_best_bridge["aggregate"]["sample_weighted"]["pairwise"]
        baseline_pair = baseline_bridge["pairwise"]
        improved = (
            centered_hit1 > baseline_hit1
            and _metric_ge(float(centered_pair) if centered_pair is not None else float("nan"), float(baseline_pair) if baseline_pair is not None else float("nan"))
        )
        q4 = "YES，slot100 bridge 指标整体优于当前 baseline。" if improved else "NO，slot100 bridge 未稳定超过当前 baseline。"
    else:
        q4 = "N/A（未得到可比较的 centered bridge 结果）。"

    return {
        "q1_remove_rank_helpful": q1,
        "q2_centered_raw_vs_rawplusrank": q2,
        "q3_global_vs_within": q3,
        "q4_bridge_improvement": q4,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run EarlyStop SVD problem-centered round1")
    ap.add_argument("--main-cache-root", default="MUI_HUB/cache")
    ap.add_argument("--extra-cache-root", default="/home/jovyan/public-ro/MUI_HUB/cache_train")
    ap.add_argument("--out-model", default="models/ml_selectors/earlystop_svd_problem_centered_round1_cap8.pkl")
    ap.add_argument("--out-summary", default="results/scans/earlystop/earlystop_svd_problem_centered_round1_cap8_summary.json")
    ap.add_argument("--out-eval", default="results/scans/earlystop/earlystop_svd_problem_centered_round1_cap8_eval.json")
    ap.add_argument("--out-plan-doc", default="docs/EARLYSTOP_SVD_PROBLEM_CENTERED_ROUND1_PLAN_20260409.md")
    ap.add_argument("--out-doc", default="docs/EARLYSTOP_SVD_PROBLEM_CENTERED_ROUND1_RESULTS_20260409.md")
    ap.add_argument("--holdout-split", type=float, default=0.20)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--feature-chunk-problems", type=int, default=DEFAULT_FEATURE_CHUNK_PROBLEMS)
    ap.add_argument("--feature-cache-dir", default="results/cache/earlystop_svd_problem_centered_round1")
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--max-problems-per-cache", type=int, default=8, help="0 means all problems")
    ap.add_argument("--enable-zscore-within-problem-raw", action="store_true")
    ap.add_argument("--run-full-pass", action="store_true", help="Run an additional full non-coding pass (max_problems_per_cache=0) for winner representation")
    args = ap.parse_args()

    main_cache_root = str((REPO_ROOT / args.main_cache_root).resolve()) if not Path(args.main_cache_root).is_absolute() else str(Path(args.main_cache_root).resolve())
    extra_cache_root = str(Path(args.extra_cache_root).resolve())
    feature_cache_dir = None if str(args.feature_cache_dir).strip().lower() in {"", "none", "off"} else (REPO_ROOT / str(args.feature_cache_dir)).resolve()
    max_problems_per_cache = None if int(args.max_problems_per_cache) <= 0 else int(args.max_problems_per_cache)
    representations = tuple(list(CORE_REPRESENTATIONS) + (list(OPTIONAL_REPRESENTATIONS) if args.enable_zscore_within_problem_raw else []))

    v1_bundle = load_earlystop_svd_bundle(REPO_ROOT / "models/ml_selectors/earlystop_svd_lowrank_lr_v1.pkl")
    round1b_bundle = load_earlystop_svd_bundle(REPO_ROOT / "models/ml_selectors/earlystop_prefix10_svd_round1b_cap8.pkl")
    strongfeat_bundle = load_earlystop_svd_bundle(REPO_ROOT / "models/ml_selectors/earlystop_strongfeat_round1_cap8.pkl")
    bridge_bundle = load_earlystop_svm_bundle(REPO_ROOT / "models/ml_selectors/bestofn_svm_bridge_v1.pkl")

    required_features = collect_required_features_for_eval(
        v1_bundle=v1_bundle,
        bridge_bundle=bridge_bundle,
    )
    for features in STRONG_FEATURE_FAMILY_MAP.values():
        required_features.update(features)

    stores_by_threshold: dict[float, dict[str, Any]] = {}
    holdout_problem_map: Optional[dict[str, set[str]]] = None
    holdout_problem_summary: dict[str, Any] = {}
    for reflection_threshold in SEARCH_REFLECTION_THRESHOLDS:
        main_store, main_cache_path, main_cache_status = _load_or_build_qualified_feature_store(
            source_name="cache",
            cache_root=main_cache_root,
            positions=EXTRACTION_POSITIONS,
            required_feature_names=required_features,
            max_problems_per_cache=max_problems_per_cache,
            reflection_threshold=float(reflection_threshold),
            max_workers=int(args.workers),
            chunk_problems=int(args.feature_chunk_problems),
            feature_cache_dir=feature_cache_dir,
            refresh_feature_cache=bool(args.refresh_feature_cache),
        )
        extra_store, extra_cache_path, extra_cache_status = _load_or_build_qualified_feature_store(
            source_name="cache_train",
            cache_root=extra_cache_root,
            positions=EXTRACTION_POSITIONS,
            required_feature_names=required_features,
            max_problems_per_cache=max_problems_per_cache,
            reflection_threshold=float(reflection_threshold),
            max_workers=int(args.workers),
            chunk_problems=int(args.feature_chunk_problems),
            feature_cache_dir=feature_cache_dir,
            refresh_feature_cache=bool(args.refresh_feature_cache),
        )
        combined_store = main_store + extra_store
        if holdout_problem_map is None:
            holdout_problem_map, holdout_problem_summary = _build_holdout_problem_map(
                combined_store,
                holdout_split=float(args.holdout_split),
                split_seed=int(args.split_seed),
            )
        train_store, holdout_store, full_store = _split_feature_store(
            combined_store,
            holdout_problem_map=holdout_problem_map,
        )
        stores_by_threshold[float(reflection_threshold)] = {
            "train_store": _filter_noncoding(train_store),
            "train_store_with_coding": train_store,
            "holdout_store": holdout_store,
            "full_store": _filter_noncoding(full_store),
            "coding_store": _filter_coding(train_store),
            "main_cache_status": str(main_cache_status),
            "extra_cache_status": str(extra_cache_status),
            "main_cache_path": None if main_cache_path is None else str(main_cache_path),
            "extra_cache_path": None if extra_cache_path is None else str(extra_cache_path),
        }

    baseline_threshold = 0.20
    baseline_store = stores_by_threshold[baseline_threshold]

    holdout_baselines = {
        "tok_conf_prefix_mean_v1": evaluate_method_from_feature_store(
            method_name="tok_conf_prefix_mean_v1",
            feature_store=baseline_store["holdout_store"],
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_tok_conf_score_fn(),
        ),
        "earlystop_from_bestofn_svm_bridge_v1": evaluate_method_from_feature_store(
            method_name="earlystop_from_bestofn_svm_bridge_v1",
            feature_store=baseline_store["holdout_store"],
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_bridge_score_fn(bridge_bundle),
        ),
        "earlystop_svd_lowrank_lr_v1": evaluate_method_from_feature_store(
            method_name="earlystop_svd_lowrank_lr_v1",
            feature_store=baseline_store["holdout_store"],
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_svd_bundle_score_fn(v1_bundle),
        ),
        "earlystop_prefix10_svd_round1b_cap8": evaluate_method_from_feature_store(
            method_name="earlystop_prefix10_svd_round1b_cap8",
            feature_store=baseline_store["holdout_store"],
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_svd_bundle_score_fn(round1b_bundle),
        ),
        "earlystop_strongfeat_round1_cap8": evaluate_method_from_feature_store(
            method_name="earlystop_strongfeat_round1_cap8",
            feature_store=baseline_store["holdout_store"],
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_svd_bundle_score_fn(strongfeat_bundle),
        ),
    }

    bridge_baseline_store = baseline_store["holdout_store"] + baseline_store["coding_store"]
    bridge_baselines = {
        "slot100_strongfeat_current": evaluate_slot100_bestofn_bridge_from_feature_store(
            method_name="slot100_strongfeat_current",
            feature_store=bridge_baseline_store,
            score_fn=make_svd_bundle_score_fn(strongfeat_bundle),
        ),
    }

    holdout_candidates: dict[str, dict[str, Any]] = {}
    bridge_candidates: dict[str, dict[str, Any]] = {}
    candidate_meta: dict[str, dict[str, Any]] = {}
    splitfit_route_summary: dict[str, dict[float, dict[str, Any]]] = {}

    for reflection_threshold in SEARCH_REFLECTION_THRESHOLDS:
        store_pack = stores_by_threshold[float(reflection_threshold)]
        train_tables = _build_scope_training_tables_with_rank_groups(
            store_pack["train_store"],
            positions=ANCHOR_POSITIONS,
        )
        bridge_store = store_pack["holdout_store"] + store_pack["coding_store"]

        for representation in representations:
            splitfit_noncoding_routes = _train_routes_for_scope_problem_centered(
                tables=[train_tables["noncoding"][idx] for idx in range(len(ANCHOR_POSITIONS))],
                positions=ANCHOR_POSITIONS,
                scope_name=f"noncoding_{_rep_tag(representation)}_ref{int(round(reflection_threshold * 100)):03d}",
                representation=representation,
                reflection_threshold=float(reflection_threshold),
                n_splits=int(args.n_splits),
                random_state=int(args.random_state),
                max_workers=int(args.workers),
            )

            noncoding_bundle = _annotate_bundle_thresholds(
                build_anchor_bundle(
                    bundle_version=f"earlystop_svd_problem_centered_round1_{_rep_tag(representation)}_ref{int(round(reflection_threshold * 100)):03d}_splitfit",
                    anchor_routes=splitfit_noncoding_routes,
                    coding_routes=copy.deepcopy(v1_bundle["domains"]["coding"]["routes"]),
                ),
                noncoding_threshold=float(reflection_threshold),
                coding_threshold=float(DEFAULT_REFLECTION_THRESHOLD),
            )

            method_name = _candidate_method_name(representation, float(reflection_threshold))
            holdout_eval = evaluate_method_from_feature_store(
                method_name=method_name,
                feature_store=store_pack["holdout_store"],
                position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
                score_fn=make_svd_bundle_score_fn(noncoding_bundle),
            )
            holdout_eval["representation"] = str(representation)
            holdout_eval["reflection_threshold"] = float(reflection_threshold)
            bridge_eval = evaluate_slot100_bestofn_bridge_from_feature_store(
                method_name=method_name,
                feature_store=bridge_store,
                score_fn=make_svd_bundle_score_fn(noncoding_bundle),
            )
            bridge_eval["representation"] = str(representation)
            bridge_eval["reflection_threshold"] = float(reflection_threshold)
            holdout_candidates[method_name] = holdout_eval
            bridge_candidates[method_name] = bridge_eval
            candidate_meta[method_name] = {
                "bundle_type": "noncoding_anchor4_coding_v1",
                "representation": str(representation),
                "reflection_threshold": float(reflection_threshold),
                "splitfit_bundle": noncoding_bundle,
            }
            splitfit_route_summary[method_name] = {
                float(position): summarise_route(route)
                for position, route in splitfit_noncoding_routes.items()
            }

    holdout_best = choose_best_candidate(list(holdout_candidates.values()))
    winner_name = str(holdout_best["method_name"])
    winner_meta = candidate_meta[winner_name]
    winner_threshold = float(winner_meta["reflection_threshold"])
    winner_representation = str(winner_meta["representation"])

    winner_store = stores_by_threshold[winner_threshold]
    full_tables = _build_scope_training_tables_with_rank_groups(
        winner_store["full_store"],
        positions=ANCHOR_POSITIONS,
    )
    fullfit_routes = _train_routes_for_scope_problem_centered(
        tables=[full_tables["noncoding"][idx] for idx in range(len(ANCHOR_POSITIONS))],
        positions=ANCHOR_POSITIONS,
        scope_name=f"noncoding_fullfit_{_rep_tag(winner_representation)}_ref{int(round(winner_threshold * 100)):03d}",
        representation=winner_representation,
        reflection_threshold=float(winner_threshold),
        n_splits=int(args.n_splits),
        random_state=int(args.random_state),
        max_workers=int(args.workers),
    )
    fullfit_bundle = _annotate_bundle_thresholds(
        build_anchor_bundle(
            bundle_version=f"earlystop_svd_problem_centered_round1_{_rep_tag(winner_representation)}_ref{int(round(winner_threshold * 100)):03d}_fullfit",
            anchor_routes=fullfit_routes,
            coding_routes=copy.deepcopy(v1_bundle["domains"]["coding"]["routes"]),
        ),
        noncoding_threshold=float(winner_threshold),
        coding_threshold=float(DEFAULT_REFLECTION_THRESHOLD),
    )

    out_model = REPO_ROOT / args.out_model
    save_earlystop_svd_bundle(fullfit_bundle, out_model)

    if bool(args.run_full_pass):
        print("[problem-centered] optional full-pass flag enabled: rerunning winner representation with max_problems_per_cache=0")
        fullpass_main_store, _, _ = _load_or_build_qualified_feature_store(
            source_name="cache",
            cache_root=main_cache_root,
            positions=EXTRACTION_POSITIONS,
            required_feature_names=required_features,
            max_problems_per_cache=None,
            reflection_threshold=float(winner_threshold),
            max_workers=int(args.workers),
            chunk_problems=int(args.feature_chunk_problems),
            feature_cache_dir=feature_cache_dir,
            refresh_feature_cache=bool(args.refresh_feature_cache),
        )
        fullpass_extra_store, _, _ = _load_or_build_qualified_feature_store(
            source_name="cache_train",
            cache_root=extra_cache_root,
            positions=EXTRACTION_POSITIONS,
            required_feature_names=required_features,
            max_problems_per_cache=None,
            reflection_threshold=float(winner_threshold),
            max_workers=int(args.workers),
            chunk_problems=int(args.feature_chunk_problems),
            feature_cache_dir=feature_cache_dir,
            refresh_feature_cache=bool(args.refresh_feature_cache),
        )
        fullpass_combined = fullpass_main_store + fullpass_extra_store
        fullpass_train, _, fullpass_full = _split_feature_store(
            fullpass_combined,
            holdout_problem_map=holdout_problem_map or {},
        )
        fullpass_tables = _build_scope_training_tables_with_rank_groups(
            _filter_noncoding(fullpass_full),
            positions=ANCHOR_POSITIONS,
        )
        fullpass_routes = _train_routes_for_scope_problem_centered(
            tables=[fullpass_tables["noncoding"][idx] for idx in range(len(ANCHOR_POSITIONS))],
            positions=ANCHOR_POSITIONS,
            scope_name=f"noncoding_fullpass_{_rep_tag(winner_representation)}_ref{int(round(winner_threshold * 100)):03d}",
            representation=winner_representation,
            reflection_threshold=float(winner_threshold),
            n_splits=int(args.n_splits),
            random_state=int(args.random_state),
            max_workers=int(args.workers),
        )
        fullfit_bundle = _annotate_bundle_thresholds(
            build_anchor_bundle(
                bundle_version=f"earlystop_svd_problem_centered_round1_{_rep_tag(winner_representation)}_ref{int(round(winner_threshold * 100)):03d}_fullpass",
                anchor_routes=fullpass_routes,
                coding_routes=copy.deepcopy(v1_bundle["domains"]["coding"]["routes"]),
            ),
            noncoding_threshold=float(winner_threshold),
            coding_threshold=float(DEFAULT_REFLECTION_THRESHOLD),
        )
        save_earlystop_svd_bundle(fullfit_bundle, out_model)
        _ = fullpass_train

    best_by_representation_holdout: dict[str, dict[str, Any]] = {}
    best_by_representation_bridge: dict[str, dict[str, Any]] = {}
    for representation in representations:
        rep_methods = [
            method_name
            for method_name, meta in candidate_meta.items()
            if str(meta["representation"]) == str(representation)
        ]
        rep_holdout_rows = [holdout_candidates[method_name] for method_name in rep_methods]
        rep_bridge_rows = [bridge_candidates[method_name] for method_name in rep_methods]
        if rep_holdout_rows:
            best_holdout = choose_best_candidate(rep_holdout_rows)
            best_name = str(best_holdout["method_name"])
            best_row = copy.deepcopy(best_holdout)
            best_row["representation"] = str(representation)
            best_row["reflection_threshold"] = float(candidate_meta[best_name]["reflection_threshold"])
            best_by_representation_holdout[str(representation)] = best_row
        if rep_bridge_rows:
            best_bridge = max(
                rep_bridge_rows,
                key=lambda item: (
                    float(item["aggregate"]["sample_weighted"]["hit@1"]),
                    float(item["aggregate"]["sample_weighted"]["pairwise"] or 0.0),
                    str(item["method_name"]),
                ),
            )
            best_name = str(best_bridge["method_name"])
            best_row = copy.deepcopy(best_bridge)
            best_row["representation"] = str(representation)
            best_row["reflection_threshold"] = float(candidate_meta[best_name]["reflection_threshold"])
            best_by_representation_bridge[str(representation)] = best_row

    winner_bridge = bridge_candidates[winner_name]
    baseline_strongfeat_holdout = holdout_baselines["earlystop_strongfeat_round1_cap8"]["aggregate"]
    baseline_slot100_bridge = bridge_baselines["slot100_strongfeat_current"]["aggregate"]["sample_weighted"]
    cand_holdout = holdout_candidates[winner_name]["aggregate"]
    cand_bridge = winner_bridge["aggregate"]["sample_weighted"]

    holdout_beats_strongfeat = (
        float(cand_holdout["auc_of_selacc"]) > float(baseline_strongfeat_holdout["auc_of_selacc"])
        and _metric_ge(float(cand_holdout["auroc@100%"]), float(baseline_strongfeat_holdout["auroc@100%"]))
        and float(cand_holdout["stop_acc@100%"]) >= float(baseline_strongfeat_holdout["stop_acc@100%"])
    )
    bridge_beats_slot100_baseline = (
        float(cand_bridge["hit@1"]) > float(baseline_slot100_bridge["hit@1"])
        and _metric_ge(float(cand_bridge["pairwise"] or 0.0), float(baseline_slot100_bridge["pairwise"] or 0.0))
        and _metric_ge(float(cand_bridge["selacc@10"]), float(baseline_slot100_bridge["selacc@10"]))
    )
    reason_parts = []
    if not holdout_beats_strongfeat:
        reason_parts.append("EarlyStop holdout 未稳定超过 strongfeat 当前 winner")
    if not bridge_beats_slot100_baseline:
        reason_parts.append("slot100 bridge 未稳定超过当前 slot100 baseline")
    reason = "；".join(reason_parts) if reason_parts else "EarlyStop 与 slot100 bridge 均达到超越当前 baseline 的条件"

    summary = {
        "created_at_utc": _now_utc(),
        "repo_status": list(REPO_STATUS_LINES),
        "protocol": {
            "main_cache_root": str(main_cache_root),
            "extra_cache_root": str(extra_cache_root),
            "feature_cache_dir": None if feature_cache_dir is None else str(feature_cache_dir),
            "holdout_split": float(args.holdout_split),
            "split_seed": int(args.split_seed),
            "train_store_by_threshold": {
                f"{thr:.2f}": _summarise_feature_store(stores_by_threshold[float(thr)]["train_store"])
                for thr in SEARCH_REFLECTION_THRESHOLDS
            },
            "holdout_store_by_threshold": {
                f"{thr:.2f}": _summarise_feature_store(stores_by_threshold[float(thr)]["holdout_store"])
                for thr in SEARCH_REFLECTION_THRESHOLDS
            },
            "bridge_store_by_threshold": {
                f"{thr:.2f}": _summarise_feature_store(stores_by_threshold[float(thr)]["holdout_store"] + stores_by_threshold[float(thr)]["coding_store"])
                for thr in SEARCH_REFLECTION_THRESHOLDS
            },
            "winner_full_store": _summarise_feature_store(winner_store["full_store"]),
            "holdout_problem_summary": holdout_problem_summary,
            "feature_cache_status_by_threshold": {
                f"{thr:.2f}": {
                    "main": stores_by_threshold[float(thr)]["main_cache_status"],
                    "extra": stores_by_threshold[float(thr)]["extra_cache_status"],
                    "main_path": stores_by_threshold[float(thr)]["main_cache_path"],
                    "extra_path": stores_by_threshold[float(thr)]["extra_cache_path"],
                }
                for thr in SEARCH_REFLECTION_THRESHOLDS
            },
        },
        "config": {
            "anchor_positions": list(ANCHOR_POSITIONS),
            "official_slot_to_anchor": {str(k): float(v) for k, v in OFFICIAL_SLOT_TO_ANCHOR.items()},
            "representation_priority": "problem-centered",
            "representations": list(representations),
            "reflection_thresholds": list(SEARCH_REFLECTION_THRESHOLDS),
            "family_map": STRONG_FEATURE_FAMILY_MAP,
            "baseline_signal_names": list(STRONG_BASELINE_SIGNAL_NAMES),
            "svd_dims": list(SEARCH_RANKS),
            "c_values": list(SEARCH_C_VALUES),
            "class_weight": list(SEARCH_CLASS_WEIGHT),
            "whiten": list(SEARCH_WHITEN),
            "workers": int(args.workers),
            "feature_chunk_problems": int(args.feature_chunk_problems),
            "max_problems_per_cache": 0 if max_problems_per_cache is None else int(max_problems_per_cache),
            "bridge_eval_scope": "holdout_noncoding_plus_coding_fallback",
            "run_full_pass": bool(args.run_full_pass),
        },
        "holdout_eval": {
            "baselines": holdout_baselines,
            "candidates": holdout_candidates,
            "best_by_representation": best_by_representation_holdout,
        },
        "bridge_eval": {
            "baselines": bridge_baselines,
            "candidates": bridge_candidates,
            "best_by_representation": best_by_representation_bridge,
        },
        "splitfit_route_summary": splitfit_route_summary,
        "fullfit_route_summary": {
            winner_name: {
                float(position): summarise_route(route)
                for position, route in fullfit_routes.items()
            }
        },
        "winner_meta": {
            "bundle_type": "noncoding_anchor4_coding_v1",
            "representation": winner_representation,
            "reflection_threshold": float(winner_threshold),
        },
        "fullfit": {
            "saved_model": _display_path(out_model),
            "winner_method_name": winner_name,
            "winner_representation": winner_representation,
            "winner_reflection_threshold": float(winner_threshold),
            "winner_bundle_version": str(fullfit_bundle["bundle_version"]),
        },
        "decision": {
            "winner_method_name": winner_name,
            "holdout_beats_strongfeat": bool(holdout_beats_strongfeat),
            "bridge_beats_slot100_baseline": bool(bridge_beats_slot100_baseline),
            "reason": str(reason),
        },
    }
    summary["qa_answers"] = _qa_answers(summary)

    eval_payload = {
        "holdout_eval": summary["holdout_eval"],
        "bridge_eval": summary["bridge_eval"],
        "winner_meta": summary["winner_meta"],
        "decision": summary["decision"],
        "fullfit": summary["fullfit"],
        "qa_answers": summary["qa_answers"],
    }

    out_summary = REPO_ROOT / args.out_summary
    out_eval = REPO_ROOT / args.out_eval
    out_plan_doc = REPO_ROOT / args.out_plan_doc
    out_doc = REPO_ROOT / args.out_doc
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    out_eval.write_text(json.dumps(eval_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    changed_files = [
        "nad/ops/earlystop_svd.py",
        "scripts/run_earlystop_svd_problem_centered_round1.py",
        str(Path(args.out_plan_doc)),
        str(Path(args.out_doc)),
        str(Path(args.out_summary)),
        str(Path(args.out_eval)),
        str(Path(args.out_model)),
    ]
    _write_plan_doc(out_plan_doc, representations=representations)
    _write_results_doc(out_doc, summary=summary, changed_files=changed_files)

    print(f"[saved] summary: {_display_path(out_summary)}")
    print(f"[saved] eval   : {_display_path(out_eval)}")
    print(f"[saved] model  : {_display_path(out_model)}")
    print(f"[saved] plan   : {_display_path(out_plan_doc)}")
    print(f"[saved] doc    : {_display_path(out_doc)}")

if __name__ == "__main__":
    main()
