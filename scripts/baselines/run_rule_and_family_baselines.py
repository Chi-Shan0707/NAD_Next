#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from nad.core.selectors.trajectory_impl import DEFAULT_REFLECTION_THRESHOLD
from nad.ops.earlystop import EARLY_STOP_POSITIONS, _problem_sort_key
from nad.ops.earlystop_svd import FULL_FEATURE_NAMES, _auroc, _group_folds, _rank_transform_matrix
from scripts.run_earlystop_prefix10_svd_round1 import (
    EXTRACTION_POSITION_INDEX,
    EXTRACTION_POSITIONS,
    build_feature_store,
    evaluate_method_from_feature_store,
)
from scripts.run_earlystop_prefix10_svd_round1b import (
    _annotate_full_payload,
    _subset_payload_by_problem_ids,
    _summarise_feature_store,
)


R2_ANCHOR_POSITIONS = tuple(float(p) for p in EARLY_STOP_POSITIONS)
FEATURE_TO_INDEX = {name: idx for idx, name in enumerate(FULL_FEATURE_NAMES)}

UNCERTAINTY_FEATURES = (
    "tok_conf_prefix",
    "tok_conf_recency",
    "tok_gini_prefix",
    "tok_gini_tail",
    "tok_gini_slope",
    "tok_neg_entropy_prefix",
    "tok_neg_entropy_recency",
    "tok_selfcert_prefix",
    "tok_selfcert_recency",
)
LOGPROB_FEATURES = (
    "tok_logprob_prefix",
    "tok_logprob_recency",
)
TRAJECTORY_FEATURES = (
    "traj_continuity",
    "traj_reflection_count",
    "traj_novelty",
    "traj_max_reflection",
    "traj_late_convergence",
)
FAMILY_FEATURES: dict[str, tuple[str, ...]] = {
    "uncertainty": UNCERTAINTY_FEATURES,
    "logprob": LOGPROB_FEATURES,
    "trajectory": TRAJECTORY_FEATURES,
}
DOMAIN_SPECS: dict[str, dict[str, Any]] = {
    "math": {
        "payload_domains": {"math"},
        "reference_key": "math",
        "label": "math",
    },
    "science": {
        "payload_domains": {"science"},
        "reference_key": "science",
        "label": "science",
    },
    "ms": {
        "payload_domains": {"math", "science"},
        "reference_key": "combined_noncoding",
        "label": "ms",
    },
    "coding": {
        "payload_domains": {"coding"},
        "reference_key": "coding",
        "label": "coding",
    },
}
CSV_COLUMNS = [
    "row_scope",
    "domain",
    "anchor_pct",
    "family",
    "method_type",
    "method_name",
    "feature_names",
    "representation",
    "cv_auroc",
    "n_valid_folds",
    "c_value",
    "class_weight",
    "holdout_auroc",
    "holdout_selacc",
    "holdout_stop_acc",
    "holdout_auc_of_auroc",
    "holdout_auc_of_selacc",
    "holdout_auroc_100",
    "holdout_stop_acc_100",
    "earliest_gt_0.6",
    "reference_method",
    "reference_auc_of_auroc",
    "gap_to_reference_auc_of_auroc",
    "is_best_single",
    "is_best_family_rule",
    "is_best_family_lr",
    "is_family_winner",
    "is_domain_winner",
]


@dataclass(frozen=True)
class ComboRuleSpec:
    name: str
    family: str
    feature_names: tuple[str, ...]
    mode: str


COMBO_RULE_SPECS = (
    ComboRuleSpec(
        name="uncertainty_conf_mean",
        family="uncertainty",
        feature_names=("tok_conf_prefix", "tok_conf_recency"),
        mode="raw_mean",
    ),
    ComboRuleSpec(
        name="uncertainty_conf_selfcert_zmean",
        family="uncertainty",
        feature_names=("tok_conf_prefix", "tok_conf_recency", "tok_selfcert_prefix", "tok_selfcert_recency"),
        mode="zscore_mean",
    ),
    ComboRuleSpec(
        name="uncertainty_dispersion_zmean",
        family="uncertainty",
        feature_names=("tok_gini_prefix", "tok_gini_tail", "tok_neg_entropy_prefix", "tok_neg_entropy_recency"),
        mode="zscore_mean",
    ),
    ComboRuleSpec(
        name="logprob_mean",
        family="logprob",
        feature_names=("tok_logprob_prefix", "tok_logprob_recency"),
        mode="raw_mean",
    ),
    ComboRuleSpec(
        name="logprob_zmean",
        family="logprob",
        feature_names=("tok_logprob_prefix", "tok_logprob_recency"),
        mode="zscore_mean",
    ),
    ComboRuleSpec(
        name="trajectory_backbone_zmean",
        family="trajectory",
        feature_names=("traj_continuity", "traj_reflection_count", "traj_max_reflection", "traj_late_convergence"),
        mode="zscore_mean",
    ),
    ComboRuleSpec(
        name="trajectory_all_zmean",
        family="trajectory",
        feature_names=TRAJECTORY_FEATURES,
        mode="zscore_mean",
    ),
)


def _stable_hash(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if not np.isfinite(float(value)):
        return "N/A"
    return f"{100.0 * float(value):.2f}%"


def _fmt_gap(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if not np.isfinite(float(value)):
        return "N/A"
    sign = "+" if float(value) >= 0.0 else ""
    return f"{sign}{100.0 * float(value):.2f}pp"


def _fmt_earliest(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{int(round(float(value) * 100.0))}%"


def _score_sort_key_overall(row: dict[str, Any]) -> tuple[float, float, float, float, str]:
    return (
        float(row.get("holdout_auc_of_auroc", float("-inf"))),
        float(row.get("holdout_auc_of_selacc", float("-inf"))),
        float(row.get("holdout_auroc_100", float("-inf"))),
        float(row.get("holdout_stop_acc_100", float("-inf"))),
        str(row.get("method_name", "")),
    )


def _score_sort_key_anchor(row: dict[str, Any]) -> tuple[float, float, float, str]:
    return (
        float(row.get("holdout_auroc", float("-inf"))),
        float(row.get("holdout_selacc", float("-inf"))),
        float(row.get("holdout_stop_acc", float("-inf"))),
        str(row.get("method_name", "")),
    )


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


def _load_or_build_feature_store(
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
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            return list(payload["feature_store"]), cache_path, "loaded"

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
    return store, cache_path, "built"


def _build_holdout_problem_map(
    feature_store: list[dict[str, Any]],
    *,
    holdout_split: float,
    split_seed: int,
    include_domains: set[str],
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    datasets: dict[str, set[str]] = {}
    for payload in feature_store:
        if str(payload["domain"]) not in include_domains:
            continue
        datasets.setdefault(str(payload["dataset_name"]), set()).update(str(v) for v in payload["problem_ids"])

    holdout_map: dict[str, set[str]] = {}
    summary: dict[str, Any] = {}
    for dataset_name in sorted(datasets.keys()):
        ordered_problem_ids = sorted(datasets[dataset_name], key=_problem_sort_key)
        if len(ordered_problem_ids) < 2:
            holdout_ids: set[str] = set()
        else:
            rng = np.random.RandomState(int(split_seed) + _stable_hash(dataset_name))
            order = rng.permutation(len(ordered_problem_ids))
            n_holdout = int(round(len(ordered_problem_ids) * float(holdout_split)))
            n_holdout = max(1, n_holdout)
            n_holdout = min(len(ordered_problem_ids) - 1, n_holdout)
            holdout_ids = {ordered_problem_ids[int(idx)] for idx in order[:n_holdout].tolist()}
        holdout_map[dataset_name] = holdout_ids
        summary[dataset_name] = {
            "total_unique_problem_ids": int(len(ordered_problem_ids)),
            "holdout_unique_problem_ids": int(len(holdout_ids)),
            "train_unique_problem_ids": int(len(ordered_problem_ids) - len(holdout_ids)),
            "holdout_problem_ids": sorted(holdout_ids, key=_problem_sort_key),
        }
    return holdout_map, summary


def _split_feature_store(
    feature_store: list[dict[str, Any]],
    *,
    holdout_problem_map: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    full_payloads: list[dict[str, Any]] = []
    train_payloads: list[dict[str, Any]] = []
    holdout_payloads: list[dict[str, Any]] = []

    for payload in feature_store:
        full_payload = _annotate_full_payload(payload)
        if full_payload["samples"] > 0:
            full_payloads.append(full_payload)

        holdout_ids = set(holdout_problem_map.get(str(payload["dataset_name"]), set()))
        train_ids = {str(v) for v in payload["problem_ids"] if str(v) not in holdout_ids}

        train_payload = _subset_payload_by_problem_ids(payload, train_ids)
        if train_payload is not None and train_payload["samples"] > 0:
            train_payloads.append(train_payload)

        holdout_payload = _subset_payload_by_problem_ids(payload, holdout_ids)
        if holdout_payload is not None and holdout_payload["samples"] > 0:
            holdout_payloads.append(holdout_payload)

    return train_payloads, holdout_payloads, full_payloads


def _filter_store(feature_store: list[dict[str, Any]], payload_domains: set[str]) -> list[dict[str, Any]]:
    return [payload for payload in feature_store if str(payload["domain"]) in payload_domains]


def _build_training_tables(
    feature_store: list[dict[str, Any]],
    positions: tuple[float, ...],
) -> dict[int, dict[str, np.ndarray]]:
    rows: dict[int, list[np.ndarray]] = {idx: [] for idx in range(len(positions))}
    labels: dict[int, list[np.ndarray]] = {idx: [] for idx in range(len(positions))}
    groups_rank: dict[int, list[np.ndarray]] = {idx: [] for idx in range(len(positions))}
    groups_cv: dict[int, list[np.ndarray]] = {idx: [] for idx in range(len(positions))}

    position_indices = [EXTRACTION_POSITION_INDEX[float(position)] for position in positions]
    for payload in feature_store:
        tensor = np.asarray(payload["tensor"], dtype=np.float64)
        if tensor.shape[0] == 0:
            continue
        y = np.asarray(payload["labels"], dtype=np.int32)
        local_rank_groups = np.asarray(payload["group_keys"], dtype=object)
        local_cv_groups = np.asarray(payload.get("cv_group_keys", payload["group_keys"]), dtype=object)
        for local_pos_idx, src_pos_idx in enumerate(position_indices):
            rows[local_pos_idx].append(np.asarray(tensor[:, src_pos_idx, :], dtype=np.float64))
            labels[local_pos_idx].append(y)
            groups_rank[local_pos_idx].append(local_rank_groups)
            groups_cv[local_pos_idx].append(local_cv_groups)

    out: dict[int, dict[str, np.ndarray]] = {}
    for pos_idx in range(len(positions)):
        if rows[pos_idx]:
            x_raw = np.vstack(rows[pos_idx]).astype(np.float64, copy=False)
            y = np.concatenate(labels[pos_idx]).astype(np.int32, copy=False)
            rank_group_keys = np.concatenate(groups_rank[pos_idx]).astype(object, copy=False)
            cv_group_keys = np.concatenate(groups_cv[pos_idx]).astype(object, copy=False)
        else:
            x_raw = np.zeros((0, len(FULL_FEATURE_NAMES)), dtype=np.float64)
            y = np.zeros((0,), dtype=np.int32)
            rank_group_keys = np.asarray([], dtype=object)
            cv_group_keys = np.asarray([], dtype=object)

        x_rank = np.zeros_like(x_raw)
        if x_raw.shape[0] > 0:
            by_rank_group: dict[Any, list[int]] = {}
            for row_idx, group_key in enumerate(rank_group_keys.tolist()):
                by_rank_group.setdefault(group_key, []).append(row_idx)
            for group_rows in by_rank_group.values():
                x_rank[group_rows] = _rank_transform_matrix(x_raw[group_rows])

        out[pos_idx] = {
            "x_raw": x_raw,
            "x_rank": x_rank,
            "y": y,
            "groups": cv_group_keys,
        }
    return out


def _zscore_transform_matrix(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.float64)
    mean = np.mean(arr, axis=0, keepdims=True)
    std = np.std(arr, axis=0, keepdims=True)
    std = np.where(std < float(eps), 1.0, std)
    return (arr - mean) / std


def _make_feature_score_fn(feature_name: str) -> Callable[[str, int, np.ndarray], np.ndarray]:
    feature_idx = FEATURE_TO_INDEX[feature_name]

    def _score(_domain: str, _position_index: int, x_raw: np.ndarray) -> np.ndarray:
        return np.asarray(x_raw[:, feature_idx], dtype=np.float64)

    return _score


def _make_combo_score_fn(spec: ComboRuleSpec) -> Callable[[str, int, np.ndarray], np.ndarray]:
    feature_indices = [FEATURE_TO_INDEX[name] for name in spec.feature_names]

    def _score(_domain: str, _position_index: int, x_raw: np.ndarray) -> np.ndarray:
        x_sel = np.asarray(x_raw[:, feature_indices], dtype=np.float64)
        if spec.mode == "raw_mean":
            return np.asarray(np.mean(x_sel, axis=1), dtype=np.float64)
        if spec.mode == "rank_mean":
            ranks = _rank_transform_matrix(x_sel)
            return np.asarray(np.mean(ranks, axis=1), dtype=np.float64)
        if spec.mode == "zscore_mean":
            z = _zscore_transform_matrix(x_sel)
            return np.asarray(np.mean(z, axis=1), dtype=np.float64)
        raise ValueError(f"Unknown combo mode: {spec.mode}")

    return _score


def _build_raw_rank_representation(
    x_raw: np.ndarray,
    x_rank: np.ndarray,
    feature_indices: list[int],
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(x_raw[:, feature_indices], dtype=np.float64),
            np.asarray(x_rank[:, feature_indices], dtype=np.float64),
        ],
        axis=1,
    )


def _fit_no_svd_lr(
    x: np.ndarray,
    y: np.ndarray,
    *,
    c_value: float,
    class_weight_name: str,
    random_state: int,
) -> Optional[dict[str, Any]]:
    if x.shape[0] < 4 or x.shape[1] < 1:
        return None
    if np.unique(y).shape[0] < 2:
        return None

    scaler = StandardScaler(with_mean=True, with_std=True)
    x_scaled = scaler.fit_transform(np.asarray(x, dtype=np.float64))
    clf = LogisticRegression(
        C=float(c_value),
        class_weight=None if class_weight_name == "none" else "balanced",
        max_iter=2000,
        random_state=int(random_state),
    )
    clf.fit(x_scaled, y)
    return {
        "scaler": scaler,
        "lr": clf,
    }


def _predict_no_svd_lr(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    x_scaled = model["scaler"].transform(np.asarray(x, dtype=np.float64))
    return np.asarray(model["lr"].decision_function(x_scaled), dtype=np.float64)


def _train_family_lr_models(
    *,
    family_name: str,
    feature_names: tuple[str, ...],
    train_tables: dict[int, dict[str, np.ndarray]],
    positions: tuple[float, ...],
    c_values: tuple[float, ...],
    class_weight_options: tuple[str, ...],
    n_splits: int,
    random_state: int,
) -> dict[str, Any]:
    feature_indices = [FEATURE_TO_INDEX[name] for name in feature_names]
    position_models: list[dict[str, Any]] = []

    for pos_idx, position in enumerate(positions):
        table = train_tables[pos_idx]
        x_raw = np.asarray(table["x_raw"], dtype=np.float64)
        x_rank = np.asarray(table["x_rank"], dtype=np.float64)
        y = np.asarray(table["y"], dtype=np.int32)
        groups = np.asarray(table["groups"], dtype=object)
        x_rep = _build_raw_rank_representation(
            x_raw=x_raw,
            x_rank=x_rank,
            feature_indices=feature_indices,
        )

        best = {
            "cv_auroc": float("-inf"),
            "n_valid_folds": 0,
            "c_value": None,
            "class_weight": None,
            "model": None,
        }
        folds = _group_folds(groups, n_splits=n_splits)
        candidate_scores: dict[tuple[float, str], list[float]] = {}

        for train_idx, test_idx in folds:
            y_train = y[train_idx]
            y_test = y[test_idx]
            if np.unique(y_train).shape[0] < 2 or np.unique(y_test).shape[0] < 2:
                continue
            for c_value in c_values:
                for class_weight_name in class_weight_options:
                    model = _fit_no_svd_lr(
                        x=x_rep[train_idx],
                        y=y_train,
                        c_value=float(c_value),
                        class_weight_name=str(class_weight_name),
                        random_state=int(random_state),
                    )
                    if model is None:
                        continue
                    scores = _predict_no_svd_lr(model, x_rep[test_idx])
                    fold_auc = _auroc(scores, y_test)
                    if np.isfinite(fold_auc):
                        candidate_scores.setdefault((float(c_value), str(class_weight_name)), []).append(float(fold_auc))

        for (c_value, class_weight_name), values in candidate_scores.items():
            if not values:
                continue
            cv_auc = float(np.mean(values))
            n_valid_folds = int(len(values))
            if cv_auc > float(best["cv_auroc"]):
                best = {
                    "cv_auroc": float(cv_auc),
                    "n_valid_folds": int(n_valid_folds),
                    "c_value": float(c_value),
                    "class_weight": str(class_weight_name),
                    "model": None,
                }

        if np.isfinite(float(best["cv_auroc"])) and best["c_value"] is not None:
            final_model = _fit_no_svd_lr(
                x=x_rep,
                y=y,
                c_value=float(best["c_value"]),
                class_weight_name=str(best["class_weight"]),
                random_state=int(random_state),
            )
            best["model"] = final_model

        position_models.append(
            {
                "position": float(position),
                "family_name": str(family_name),
                "feature_names": list(feature_names),
                "feature_indices": list(feature_indices),
                "representation": "raw+rank",
                "cv_auroc": float(best["cv_auroc"]),
                "n_valid_folds": int(best["n_valid_folds"]),
                "c_value": None if best["c_value"] is None else float(best["c_value"]),
                "class_weight": None if best["class_weight"] is None else str(best["class_weight"]),
                "model": best["model"],
            }
        )

    return {
        "family_name": str(family_name),
        "feature_names": list(feature_names),
        "feature_indices": list(feature_indices),
        "representation": "raw+rank",
        "position_models": position_models,
    }


def _make_family_lr_score_fn(model_bundle: dict[str, Any]) -> Callable[[str, int, np.ndarray], np.ndarray]:
    position_models = list(model_bundle["position_models"])

    def _score(_domain: str, position_index: int, x_raw: np.ndarray) -> np.ndarray:
        pos_model = position_models[int(position_index)]
        model = pos_model["model"]
        if model is None:
            return np.zeros((x_raw.shape[0],), dtype=np.float64)
        x_rank = _rank_transform_matrix(np.asarray(x_raw, dtype=np.float64))
        x_rep = _build_raw_rank_representation(
            x_raw=np.asarray(x_raw, dtype=np.float64),
            x_rank=x_rank,
            feature_indices=list(pos_model["feature_indices"]),
        )
        return _predict_no_svd_lr(model, x_rep)

    return _score


def _evaluate_domain_methods(
    *,
    domain_key: str,
    train_store: list[dict[str, Any]],
    holdout_store: list[dict[str, Any]],
    c_values: tuple[float, ...],
    class_weight_options: tuple[str, ...],
    n_splits: int,
    random_state: int,
) -> dict[str, Any]:
    train_tables = _build_training_tables(train_store, R2_ANCHOR_POSITIONS)
    methods: list[dict[str, Any]] = []

    for family_name, feature_names in FAMILY_FEATURES.items():
        for feature_name in feature_names:
            eval_result = evaluate_method_from_feature_store(
                method_name=feature_name,
                feature_store=holdout_store,
                position_values=R2_ANCHOR_POSITIONS,
                score_fn=_make_feature_score_fn(feature_name),
            )
            methods.append(
                {
                    "domain": str(domain_key),
                    "family": str(family_name),
                    "method_type": "single_rule",
                    "method_name": str(feature_name),
                    "feature_names": [str(feature_name)],
                    "representation": "raw",
                    "eval": eval_result,
                    "train": None,
                }
            )

    for spec in COMBO_RULE_SPECS:
        eval_result = evaluate_method_from_feature_store(
            method_name=spec.name,
            feature_store=holdout_store,
            position_values=R2_ANCHOR_POSITIONS,
            score_fn=_make_combo_score_fn(spec),
        )
        methods.append(
            {
                "domain": str(domain_key),
                "family": str(spec.family),
                "method_type": "combo_rule",
                "method_name": str(spec.name),
                "feature_names": list(spec.feature_names),
                "representation": str(spec.mode),
                "eval": eval_result,
                "train": None,
            }
        )

    for family_name, feature_names in FAMILY_FEATURES.items():
        train_bundle = _train_family_lr_models(
            family_name=family_name,
            feature_names=feature_names,
            train_tables=train_tables,
            positions=R2_ANCHOR_POSITIONS,
            c_values=c_values,
            class_weight_options=class_weight_options,
            n_splits=n_splits,
            random_state=random_state,
        )
        eval_result = evaluate_method_from_feature_store(
            method_name=f"{family_name}_lr",
            feature_store=holdout_store,
            position_values=R2_ANCHOR_POSITIONS,
            score_fn=_make_family_lr_score_fn(train_bundle),
        )
        methods.append(
            {
                "domain": str(domain_key),
                "family": str(family_name),
                "method_type": "family_lr",
                "method_name": f"{family_name}_lr",
                "feature_names": list(feature_names),
                "representation": "raw+rank",
                "eval": eval_result,
                "train": train_bundle,
            }
        )

    return {
        "domain": str(domain_key),
        "train_summary": _summarise_feature_store(train_store),
        "holdout_summary": _summarise_feature_store(holdout_store),
        "methods": methods,
    }


def _load_reference_metrics() -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    ms_summary_path = REPO_ROOT / "results" / "scans" / "earlystop" / "es_svd_ms_rr_r2_20260412_summary.json"
    if ms_summary_path.exists():
        payload = json.loads(ms_summary_path.read_text(encoding="utf-8"))
        refs["math"] = {
            "method_name": payload["validate"]["math"]["candidate"]["method_name"],
            "aggregate": payload["validate"]["math"]["candidate"]["aggregate"],
        }
        refs["science"] = {
            "method_name": payload["validate"]["science"]["candidate"]["method_name"],
            "aggregate": payload["validate"]["science"]["candidate"]["aggregate"],
        }
        refs["ms"] = {
            "method_name": payload["validate"]["combined_noncoding"]["candidate"]["method_name"],
            "aggregate": payload["validate"]["combined_noncoding"]["candidate"]["aggregate"],
        }

    coding_summary_path = REPO_ROOT / "results" / "scans" / "earlystop" / "es_svd_coding_rr_r1_summary.json"
    if coding_summary_path.exists():
        payload = json.loads(coding_summary_path.read_text(encoding="utf-8"))
        refs["coding"] = {
            "method_name": payload["validate"]["candidate"]["method_name"],
            "aggregate": payload["validate"]["candidate"]["aggregate"],
        }
    return refs


def _build_csv_rows(
    *,
    domain_results: dict[str, dict[str, Any]],
    references: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for domain_key, payload in domain_results.items():
        reference = references.get(domain_key)
        reference_method = None if reference is None else str(reference["method_name"])
        reference_auc = None if reference is None else float(reference["aggregate"]["auc_of_auroc"])
        for method in payload["methods"]:
            eval_result = method["eval"]
            aggregate = eval_result["aggregate"]
            train_bundle = method["train"]
            by_position = list(aggregate["by_position"])

            overall_row = {
                "row_scope": "overall",
                "domain": str(domain_key),
                "anchor_pct": "ALL",
                "family": str(method["family"]),
                "method_type": str(method["method_type"]),
                "method_name": str(method["method_name"]),
                "feature_names": "|".join(str(name) for name in method["feature_names"]),
                "representation": str(method["representation"]),
                "cv_auroc": "",
                "n_valid_folds": "",
                "c_value": "",
                "class_weight": "",
                "holdout_auroc": "",
                "holdout_selacc": "",
                "holdout_stop_acc": "",
                "holdout_auc_of_auroc": float(aggregate["auc_of_auroc"]),
                "holdout_auc_of_selacc": float(aggregate["auc_of_selacc"]),
                "holdout_auroc_100": float(aggregate["auroc@100%"]),
                "holdout_stop_acc_100": float(aggregate["stop_acc@100%"]),
                "earliest_gt_0.6": "" if aggregate["earliest_gt_0.6"] is None else float(aggregate["earliest_gt_0.6"]),
                "reference_method": "" if reference_method is None else str(reference_method),
                "reference_auc_of_auroc": "" if reference_auc is None else float(reference_auc),
                "gap_to_reference_auc_of_auroc": "" if reference_auc is None else float(aggregate["auc_of_auroc"] - reference_auc),
                "is_best_single": False,
                "is_best_family_rule": False,
                "is_best_family_lr": False,
                "is_family_winner": False,
                "is_domain_winner": False,
            }
            if train_bundle is not None:
                cv_values = [
                    float(item["cv_auroc"])
                    for item in train_bundle["position_models"]
                    if np.isfinite(float(item["cv_auroc"]))
                ]
                fold_values = [int(item["n_valid_folds"]) for item in train_bundle["position_models"]]
                overall_row["cv_auroc"] = "" if not cv_values else float(np.mean(cv_values))
                overall_row["n_valid_folds"] = int(np.sum(fold_values))
            rows.append(overall_row)

            for pos_idx, pos_metrics in enumerate(by_position):
                anchor_pct = int(round(float(pos_metrics["position"]) * 100.0))
                anchor_row = {
                    "row_scope": "anchor",
                    "domain": str(domain_key),
                    "anchor_pct": int(anchor_pct),
                    "family": str(method["family"]),
                    "method_type": str(method["method_type"]),
                    "method_name": str(method["method_name"]),
                    "feature_names": "|".join(str(name) for name in method["feature_names"]),
                    "representation": str(method["representation"]),
                    "cv_auroc": "",
                    "n_valid_folds": "",
                    "c_value": "",
                    "class_weight": "",
                    "holdout_auroc": float(pos_metrics["auroc"]),
                    "holdout_selacc": float(pos_metrics["selacc@10%"]),
                    "holdout_stop_acc": float(pos_metrics["stop_acc"]),
                    "holdout_auc_of_auroc": "",
                    "holdout_auc_of_selacc": "",
                    "holdout_auroc_100": "",
                    "holdout_stop_acc_100": "",
                    "earliest_gt_0.6": "",
                    "reference_method": "" if reference_method is None else str(reference_method),
                    "reference_auc_of_auroc": "",
                    "gap_to_reference_auc_of_auroc": "",
                    "is_best_single": False,
                    "is_best_family_rule": False,
                    "is_best_family_lr": False,
                    "is_family_winner": False,
                    "is_domain_winner": False,
                }
                if train_bundle is not None:
                    pos_train = train_bundle["position_models"][pos_idx]
                    anchor_row["cv_auroc"] = float(pos_train["cv_auroc"]) if np.isfinite(float(pos_train["cv_auroc"])) else ""
                    anchor_row["n_valid_folds"] = int(pos_train["n_valid_folds"])
                    anchor_row["c_value"] = "" if pos_train["c_value"] is None else float(pos_train["c_value"])
                    anchor_row["class_weight"] = "" if pos_train["class_weight"] is None else str(pos_train["class_weight"])
                rows.append(anchor_row)

    _apply_winner_flags(rows)
    return rows


def _apply_winner_flags(rows: list[dict[str, Any]]) -> None:
    overall_rows = [row for row in rows if row["row_scope"] == "overall"]
    anchor_rows = [row for row in rows if row["row_scope"] == "anchor"]

    for domain_key in sorted({str(row["domain"]) for row in rows}):
        domain_overall = [row for row in overall_rows if str(row["domain"]) == domain_key]
        domain_anchor = [row for row in anchor_rows if str(row["domain"]) == domain_key]

        if domain_overall:
            best_domain_row = max(domain_overall, key=_score_sort_key_overall)
            best_domain_row["is_domain_winner"] = True

        for family_name in sorted({str(row["family"]) for row in domain_overall}):
            family_rows = [row for row in domain_overall if str(row["family"]) == family_name]
            if family_rows:
                best_family_row = max(family_rows, key=_score_sort_key_overall)
                best_family_row["is_family_winner"] = True

            single_overall = [
                row for row in family_rows if str(row["method_type"]) == "single_rule"
            ]
            if single_overall:
                max(single_overall, key=_score_sort_key_overall)["is_best_single"] = True

            family_rule_overall = [
                row for row in family_rows if str(row["method_type"]) in {"single_rule", "combo_rule"}
            ]
            if family_rule_overall:
                max(family_rule_overall, key=_score_sort_key_overall)["is_best_family_rule"] = True

            family_lr_overall = [
                row for row in family_rows if str(row["method_type"]) == "family_lr"
            ]
            if family_lr_overall:
                max(family_lr_overall, key=_score_sort_key_overall)["is_best_family_lr"] = True

        for anchor_pct in sorted({int(row["anchor_pct"]) for row in domain_anchor}):
            anchor_subset = [
                row for row in domain_anchor if int(row["anchor_pct"]) == int(anchor_pct)
            ]
            for family_name in sorted({str(row["family"]) for row in anchor_subset}):
                family_anchor_rows = [
                    row for row in anchor_subset if str(row["family"]) == family_name
                ]
                single_anchor = [
                    row for row in family_anchor_rows if str(row["method_type"]) == "single_rule"
                ]
                if single_anchor:
                    max(single_anchor, key=_score_sort_key_anchor)["is_best_single"] = True

                family_rule_anchor = [
                    row for row in family_anchor_rows if str(row["method_type"]) in {"single_rule", "combo_rule"}
                ]
                if family_rule_anchor:
                    max(family_rule_anchor, key=_score_sort_key_anchor)["is_best_family_rule"] = True

                family_lr_anchor = [
                    row for row in family_anchor_rows if str(row["method_type"]) == "family_lr"
                ]
                if family_lr_anchor:
                    max(family_lr_anchor, key=_score_sort_key_anchor)["is_best_family_lr"] = True


def _write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})


def _render_markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    return out


def _best_overall_method(rows: list[dict[str, Any]], *, domain: str, family: Optional[str] = None, method_types: Optional[set[str]] = None) -> Optional[dict[str, Any]]:
    subset = [
        row
        for row in rows
        if row["row_scope"] == "overall" and str(row["domain"]) == str(domain)
    ]
    if family is not None:
        subset = [row for row in subset if str(row["family"]) == str(family)]
    if method_types is not None:
        subset = [row for row in subset if str(row["method_type"]) in method_types]
    if not subset:
        return None
    return max(subset, key=_score_sort_key_overall)


def _load_doc_references() -> dict[str, dict[str, Any]]:
    return _load_reference_metrics()


def _write_markdown(
    *,
    out_path: Path,
    rows: list[dict[str, Any]],
    domain_results: dict[str, dict[str, Any]],
    references: dict[str, dict[str, Any]],
    protocol_summary: dict[str, Any],
) -> None:
    lines: list[str] = [
        "# Rule and Family Baselines",
        "",
        "Interpretability-first baselines that isolate uncertainty-only, logprob-only, and trajectory-only signals under the grouped Early-Stop holdout protocol.",
        "",
        "## Protocol",
        "",
        f"- Repository: `NAD_Next` only.",
        f"- Holdout: `{int(round((1.0 - float(protocol_summary['holdout_split'])) * 100))}/{int(round(float(protocol_summary['holdout_split']) * 100))}` grouped split by `dataset + problem_id`, seed `{int(protocol_summary['split_seed'])}`.",
        f"- Anchors: `{', '.join(str(int(round(float(p) * 100.0))) for p in R2_ANCHOR_POSITIONS)}`.",
        f"- Main cache root: `{protocol_summary['main_cache_root']}`.",
        f"- Extra cache root: `{protocol_summary['extra_cache_root']}`.",
        f"- Reflection threshold: `{float(protocol_summary['reflection_threshold']):.2f}`.",
        "",
        "## Data Coverage",
        "",
    ]

    coverage_rows: list[list[str]] = []
    for domain_key in ("math", "science", "ms", "coding"):
        if domain_key not in domain_results:
            continue
        payload = domain_results[domain_key]
        coverage_rows.append(
            [
                domain_key,
                str(payload["train_summary"]["total_problems"]),
                str(payload["train_summary"]["total_samples"]),
                str(payload["holdout_summary"]["total_problems"]),
                str(payload["holdout_summary"]["total_samples"]),
            ]
        )
    lines.extend(_render_markdown_table(
        ["Domain", "Train Problems", "Train Samples", "Holdout Problems", "Holdout Samples"],
        coverage_rows,
    ))

    lines.extend([
        "## Domain Summary",
        "",
    ])
    domain_table_rows: list[list[str]] = []
    for domain_key in ("math", "science", "ms", "coding"):
        if domain_key not in domain_results:
            continue
        best_single = _best_overall_method(rows, domain=domain_key, method_types={"single_rule"})
        best_rule = _best_overall_method(rows, domain=domain_key, method_types={"single_rule", "combo_rule"})
        best_lr = _best_overall_method(rows, domain=domain_key, method_types={"family_lr"})
        ref = references.get(domain_key)
        domain_table_rows.append(
            [
                domain_key,
                "N/A" if best_single is None else f"{best_single['method_name']} ({_fmt_pct(best_single['holdout_auc_of_auroc'])})",
                "N/A" if best_rule is None else f"{best_rule['method_name']} ({_fmt_pct(best_rule['holdout_auc_of_auroc'])})",
                "N/A" if best_lr is None else f"{best_lr['method_name']} ({_fmt_pct(best_lr['holdout_auc_of_auroc'])})",
                "N/A" if ref is None else f"{ref['method_name']} ({_fmt_pct(ref['aggregate']['auc_of_auroc'])})",
            ]
        )
    lines.extend(_render_markdown_table(
        ["Domain", "Best Single Feature", "Best Simple Rule", "Best Family LR", "Reference SVDomain"],
        domain_table_rows,
    ))

    lines.extend([
        "## Answers",
        "",
    ])

    best_rules_by_domain = {
        domain_key: _best_overall_method(rows, domain=domain_key, method_types={"single_rule", "combo_rule"})
        for domain_key in domain_results.keys()
    }
    best_family_lr_by_domain = {
        domain_key: _best_overall_method(rows, domain=domain_key, method_types={"family_lr"})
        for domain_key in domain_results.keys()
    }

    best_rule_parts = []
    for domain_key in ("math", "science", "ms", "coding"):
        row = best_rules_by_domain.get(domain_key)
        if row is None:
            continue
        best_rule_parts.append(f"`{domain_key}`: `{row['method_name']}` at `{_fmt_pct(row['holdout_auc_of_auroc'])}`")
    lines.append(f"- How strong are the best simple rules? {'; '.join(best_rule_parts) if best_rule_parts else 'No rule results were generated.'}")

    math_traj = _best_overall_method(rows, domain="math", family="trajectory")
    math_traj_rule = _best_overall_method(rows, domain="math", family="trajectory", method_types={"single_rule", "combo_rule"})
    math_ref = references.get("math")
    if math_traj is None:
        lines.append("- Is trajectory-only enough in math? Not evaluated.")
    else:
        math_traj_gap = None
        if math_ref is not None:
            math_traj_gap = float(math_traj["holdout_auc_of_auroc"]) - float(math_ref["aggregate"]["auc_of_auroc"])
        if math_traj_gap is not None and math_traj_gap >= 0.0:
            rule_clause = ""
            if math_traj_rule is not None:
                rule_clause = (
                    f" The best fixed trajectory rule is `{math_traj_rule['method_name']}` at `{_fmt_pct(math_traj_rule['holdout_auc_of_auroc'])}`."
                )
            lines.append(
                "- Is trajectory-only enough in math? "
                + (
                    f"Yes, with a caveat: trajectory-only reaches SVDomain-level math performance once you allow the family LR. "
                    f"`{math_traj['method_name']}` gets `{_fmt_pct(math_traj['holdout_auc_of_auroc'])}`, "
                    f"`{_fmt_gap(math_traj_gap)}` versus `{math_ref['method_name']}`.{rule_clause}"
                )
            )
        elif math_traj_rule is not None:
            lines.append(
                "- Is trajectory-only enough in math? "
                + (
                    f"Partly: trajectory-only owns the best transparent rule in math with `{math_traj_rule['method_name']}` at "
                    f"`{_fmt_pct(math_traj_rule['holdout_auc_of_auroc'])}`, but the full trajectory-only ceiling is "
                    f"`{math_traj['method_name']}` at `{_fmt_pct(math_traj['holdout_auc_of_auroc'])}`."
                )
            )
        else:
            lines.append(
                "- Is trajectory-only enough in math? "
                + f"Trajectory-only tops out at `{math_traj['method_name']}` / `{_fmt_pct(math_traj['holdout_auc_of_auroc'])}`."
            )

    sci_unc = _best_overall_method(rows, domain="science", family="uncertainty")
    sci_log = _best_overall_method(rows, domain="science", family="logprob")
    sci_traj = _best_overall_method(rows, domain="science", family="trajectory")
    if sci_unc is None:
        lines.append("- Is uncertainty-only stronger in science? Not evaluated.")
    else:
        stronger = True
        for other in (sci_log, sci_traj):
            if other is not None and float(other["holdout_auc_of_auroc"]) > float(sci_unc["holdout_auc_of_auroc"]):
                stronger = False
        lines.append(
            "- Is uncertainty-only stronger in science? "
            + (
                f"Yes: uncertainty-only wins science with `{sci_unc['method_name']}` at `{_fmt_pct(sci_unc['holdout_auc_of_auroc'])}`."
                if stronger
                else f"No: uncertainty-only reaches `{_fmt_pct(sci_unc['holdout_auc_of_auroc'])}`, but another family is higher."
            )
        )

    coding_rows = [row for row in rows if row["row_scope"] == "overall" and str(row["domain"]) == "coding"]
    if not coding_rows:
        lines.append("- Is coding weak because every family is weak, or because only the combined route fails? Coding was not evaluated in this run.")
    else:
        coding_best = max(coding_rows, key=_score_sort_key_overall)
        coding_unc = _best_overall_method(rows, domain="coding", family="uncertainty")
        coding_log = _best_overall_method(rows, domain="coding", family="logprob")
        coding_traj = _best_overall_method(rows, domain="coding", family="trajectory")
        coding_ref = references.get("coding")
        weak_threshold = float(coding_best["holdout_auc_of_auroc"]) < 0.60
        if coding_ref is not None:
            ref_gap = float(coding_best["holdout_auc_of_auroc"]) - float(coding_ref["aggregate"]["auc_of_auroc"])
            lines.append(
                "- Is coding weak because every family is weak, or because only the combined route fails? "
                + (
                    f"Both signals show up: every family is fairly weak in absolute terms, but the combined route fails even harder. "
                    f"Coding tops out at `{coding_best['method_name']}` / `{_fmt_pct(coding_best['holdout_auc_of_auroc'])}`; "
                    f"the best uncertainty/logprob/trajectory baselines are "
                    f"`{_fmt_pct(coding_unc['holdout_auc_of_auroc']) if coding_unc is not None else 'N/A'}` / "
                    f"`{_fmt_pct(coding_log['holdout_auc_of_auroc']) if coding_log is not None else 'N/A'}` / "
                    f"`{_fmt_pct(coding_traj['holdout_auc_of_auroc']) if coding_traj is not None else 'N/A'}`; "
                    f"`{coding_ref['method_name']}` is lower at `{_fmt_pct(coding_ref['aggregate']['auc_of_auroc'])}` "
                    f"(`{_fmt_gap(ref_gap)}` behind the best family baseline)."
                    if weak_threshold
                    else f"It is not just a combined-route failure: `{coding_best['method_name']}` already reaches `{_fmt_pct(coding_best['holdout_auc_of_auroc'])}` on coding."
                )
            )
        else:
            lines.append(
                "- Is coding weak because every family is weak, or because only the combined route fails? "
                + (
                    f"Every family is weak: the best coding-only baseline is only `{_fmt_pct(coding_best['holdout_auc_of_auroc'])}`."
                    if weak_threshold
                    else f"The best coding-only baseline reaches `{_fmt_pct(coding_best['holdout_auc_of_auroc'])}`, so the failure is not purely from combining families."
                )
            )

    overall_rows = [row for row in rows if row["row_scope"] == "overall"]
    rows_with_ref = [row for row in overall_rows if row.get("reference_auc_of_auroc") not in (None, "", "nan")]
    if not rows_with_ref:
        lines.append("- Which simple baseline is the most serious competitor to SVDomain? Not enough results to judge.")
    else:
        best_gap_row = max(rows_with_ref, key=lambda row: float(row["holdout_auc_of_auroc"]) - float(row["reference_auc_of_auroc"]))
        best_gap = float(best_gap_row["holdout_auc_of_auroc"]) - float(best_gap_row["reference_auc_of_auroc"])
        noncoding_rows = [row for row in rows_with_ref if str(row["domain"]) != "coding"]
        if best_gap_row["domain"] == "coding" and noncoding_rows:
            noncoding_best = max(noncoding_rows, key=lambda row: float(row["holdout_auc_of_auroc"]) - float(row["reference_auc_of_auroc"]))
            noncoding_gap = float(noncoding_best["holdout_auc_of_auroc"]) - float(noncoding_best["reference_auc_of_auroc"])
            lines.append(
                f"- Which simple baseline is the most serious competitor to SVDomain? The biggest anomaly is `{best_gap_row['method_name']}` on `coding` at `{_fmt_pct(best_gap_row['holdout_auc_of_auroc'])}` (`{_fmt_gap(best_gap)}` versus `{best_gap_row['reference_method']}`), but on the main noncoding benchmark the strongest competitor is `{noncoding_best['method_name']}` on `{noncoding_best['domain']}` at `{_fmt_pct(noncoding_best['holdout_auc_of_auroc'])}` (`{_fmt_gap(noncoding_gap)}` versus `{noncoding_best['reference_method']}`)."
            )
        else:
            lines.append(
                f"- Which simple baseline is the most serious competitor to SVDomain? `{best_gap_row['method_name']}` on `{best_gap_row['domain']}` is closest at `{_fmt_pct(best_gap_row['holdout_auc_of_auroc'])}`, a gap of `{_fmt_gap(best_gap)}` versus `{best_gap_row['reference_method']}`."
            )

    lines.extend([
        "",
        "## Notes",
        "",
        "- Rule baselines are fixed, transparent formulas only; there is no learned weighting in `single_rule` or `combo_rule` rows.",
        "- Family LRs use only same-family features with `raw+rank`, standard scaling, and no SVD.",
        "- All files are generated under `NAD_Next` outputs: `scripts/baselines/`, `results/tables/`, and `docs/`.",
        "",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run rule-only and family-restricted baseline study.")
    ap.add_argument("--main-cache-root", default="MUI_HUB/cache")
    ap.add_argument("--extra-cache-root", default="MUI_HUB/cache_train")
    ap.add_argument("--holdout-split", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--reflection-threshold", type=float, default=float(DEFAULT_REFLECTION_THRESHOLD))
    ap.add_argument("--feature-workers", type=int, default=8)
    ap.add_argument("--feature-chunk-problems", type=int, default=24)
    ap.add_argument("--max-problems-per-cache", type=int, default=0, help="0 means all problems")
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--feature-cache-dir", default="results/cache/rule_and_family_baselines")
    ap.add_argument("--csv-out", default="results/tables/rule_and_family_baselines.csv")
    ap.add_argument("--doc-out", default="docs/RULE_AND_FAMILY_BASELINES.md")
    args = ap.parse_args()

    max_problems_per_cache = None if int(args.max_problems_per_cache) <= 0 else int(args.max_problems_per_cache)
    feature_cache_dir = Path(args.feature_cache_dir)

    required_feature_names = set(UNCERTAINTY_FEATURES) | set(LOGPROB_FEATURES) | set(TRAJECTORY_FEATURES)

    cache_store, cache_store_path, cache_status = _load_or_build_feature_store(
        source_name="cache",
        cache_root=str(args.main_cache_root),
        positions=EXTRACTION_POSITIONS,
        required_feature_names=required_feature_names,
        max_problems_per_cache=max_problems_per_cache,
        reflection_threshold=float(args.reflection_threshold),
        max_workers=int(args.feature_workers),
        chunk_problems=int(args.feature_chunk_problems),
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_feature_cache),
    )
    extra_store, extra_store_path, extra_status = _load_or_build_feature_store(
        source_name="cache_train",
        cache_root=str(args.extra_cache_root),
        positions=EXTRACTION_POSITIONS,
        required_feature_names=required_feature_names,
        max_problems_per_cache=max_problems_per_cache,
        reflection_threshold=float(args.reflection_threshold),
        max_workers=int(args.feature_workers),
        chunk_problems=int(args.feature_chunk_problems),
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_feature_cache),
    )
    feature_store = list(cache_store) + list(extra_store)

    holdout_problem_map, holdout_problem_summary = _build_holdout_problem_map(
        feature_store,
        holdout_split=float(args.holdout_split),
        split_seed=int(args.split_seed),
        include_domains={"math", "science", "coding"},
    )
    train_store, holdout_store, full_store = _split_feature_store(
        feature_store,
        holdout_problem_map=holdout_problem_map,
    )

    print(
        "[rule-family] feature caches:",
        f"cache={cache_status} ({'none' if cache_store_path is None else _display_path(cache_store_path)})",
        f"cache_train={extra_status} ({'none' if extra_store_path is None else _display_path(extra_store_path)})",
        flush=True,
    )
    print(
        "[rule-family] store summary:",
        json.dumps(
            {
                "full": _summarise_feature_store(full_store),
                "train": _summarise_feature_store(train_store),
                "holdout": _summarise_feature_store(holdout_store),
            },
            indent=2,
        ),
        flush=True,
    )

    domain_results: dict[str, dict[str, Any]] = {}
    for domain_key in ("math", "science", "ms", "coding"):
        payload_domains = set(DOMAIN_SPECS[domain_key]["payload_domains"])
        domain_train_store = _filter_store(train_store, payload_domains)
        domain_holdout_store = _filter_store(holdout_store, payload_domains)
        if not domain_train_store or not domain_holdout_store:
            print(f"[rule-family] skip domain={domain_key} train={len(domain_train_store)} holdout={len(domain_holdout_store)}", flush=True)
            continue
        print(f"[rule-family] evaluating domain={domain_key}", flush=True)
        domain_results[domain_key] = _evaluate_domain_methods(
            domain_key=domain_key,
            train_store=domain_train_store,
            holdout_store=domain_holdout_store,
            c_values=(0.05, 0.10, 0.20, 0.50, 1.00),
            class_weight_options=("none", "balanced"),
            n_splits=int(args.n_splits),
            random_state=int(args.random_state),
        )

    references = _load_doc_references()
    csv_rows = _build_csv_rows(
        domain_results=domain_results,
        references=references,
    )
    csv_out = Path(args.csv_out)
    doc_out = Path(args.doc_out)
    _write_csv(csv_rows, csv_out)
    _write_markdown(
        out_path=doc_out,
        rows=csv_rows,
        domain_results=domain_results,
        references=references,
        protocol_summary={
            "holdout_split": float(args.holdout_split),
            "split_seed": int(args.split_seed),
            "main_cache_root": str(args.main_cache_root),
            "extra_cache_root": str(args.extra_cache_root),
            "reflection_threshold": float(args.reflection_threshold),
            "holdout_problem_summary": holdout_problem_summary,
        },
    )
    print(f"[rule-family] wrote csv={_display_path(csv_out)}", flush=True)
    print(f"[rule-family] wrote doc={_display_path(doc_out)}", flush=True)


if __name__ == "__main__":
    main()
