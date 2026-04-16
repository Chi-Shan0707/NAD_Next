#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

for env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(env_name, "1")

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings(
    "ignore",
    message=r"'penalty' was deprecated in version 1\.8 and will be removed in 1\.10\..*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Inconsistent values: penalty=l1 with l1_ratio=0\.0\..*",
    category=UserWarning,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nad.ops.earlystop import discover_cache_entries, validate_earlystop_payload, write_earlystop_payload
from nad.ops.earlystop_svd import (
    DEFAULT_REFLECTION_THRESHOLD,
    FULL_FEATURE_NAMES,
    TOKEN_FEATURES,
    TRAJ_FEATURES,
    _auroc,
    _build_representation,
    _fit_svd_lr_model,
    _predict_svd_lr,
    _rank_transform_matrix,
    get_domain,
    load_earlystop_svd_bundle,
)
from scripts.export_earlystop_svd_submission import (
    _load_or_build_feature_store as _load_or_build_blind_feature_store,
    _problem_scores_from_payload,
)
from scripts.run_earlystop_prefix10_svd_round1 import (
    DEFAULT_FEATURE_CHUNK_PROBLEMS,
    EARLY_STOP_POSITIONS,
    _display_path,
    _pct_label,
    evaluate_method_from_feature_store,
    make_svd_bundle_score_fn,
)
from scripts.run_structured_ood_suite import _ensure_training_payload
from SVDomain.train_es_svd_ms_rr_r1 import FIXED_FEATURE_INDICES, FIXED_FEATURE_NAMES
from SVDomain.train_es_svd_ms_rr_r2 import (
    _build_domain_training_tables,
    _build_holdout_problem_map,
    _load_or_build_qualified_feature_store,
    _resolve_path,
    _split_feature_store,
)


POSITIONS = tuple(float(v) for v in EARLY_STOP_POSITIONS)
POSITION_TO_INDEX = {float(v): idx for idx, v in enumerate(POSITIONS)}
SCIENCE_DOMAIN = "science"
SCIENCE_DATASET = "gpqa"
BASE_AGGRESSIVE_JSON = (
    REPO_ROOT
    / "submission"
    / "EarlyStop"
    / "es_svd_ms_rr_r1__coding_rotation_adapter_aggressive_tokenonly_late70100.json"
)
CURRENT_SCIENCE_BUNDLE = REPO_ROOT / "models" / "ml_selectors" / "es_svd_science_rr_r2_20260412.pkl"
CURRENT_SCIENCE_EVAL = REPO_ROOT / "results" / "scans" / "earlystop" / "es_svd_ms_rr_r2_20260412_eval.json"

FEATURE_BANK_47 = list(FULL_FEATURE_NAMES)
FEATURE_TO_INDEX = {str(name): idx for idx, name in enumerate(FEATURE_BANK_47)}
AVAILABILITY_FEATURES = [
    "has_tok_conf",
    "has_tok_gini",
    "has_tok_neg_entropy",
    "has_tok_selfcert",
    "has_tok_logprob",
    "has_rows_bank",
]
PREFIX_META_FEATURES = ["nc_mean", "nc_slope"]
WIDE46_FEATURES = [name for name in FEATURE_BANK_47 if name != "self_similarity"]
WINDOW5_FEATURES = [
    "prefix_best_window_quality",
    "post_reflection_recovery",
    "last_block_instability",
    "reflection_density",
    "reflection_count",
]
DELTA12_FEATURES = [
    "conf_d1_tail_mean",
    "conf_abs_d1_tail_mean",
    "conf_abs_d2_tail_mean",
    "conf_abs_d1_full_minus_tail",
    "gini_d1_tail_mean",
    "gini_abs_d1_tail_mean",
    "gini_abs_d2_tail_mean",
    "gini_abs_d1_full_minus_tail",
    "entropy_d1_tail_mean",
    "entropy_abs_d1_tail_mean",
    "entropy_abs_d2_tail_mean",
    "entropy_abs_d1_full_minus_tail",
]
DYNAMIC17_FEATURES = WINDOW5_FEATURES + DELTA12_FEATURES
SCIENCE_FRONT7_FEATURES = [
    "tok_conf_prefix",
    "tok_conf_recency",
    "prefix_best_window_quality",
    "tail_q10",
    "traj_reflection_count",
    "tok_selfcert_prefix",
    "tok_selfcert_recency",
]
SCIENCE_FRONT9_FEATURES = SCIENCE_FRONT7_FEATURES + [
    "tok_gini_prefix",
    "traj_continuity",
]
SCIENCE_FRONT11_FEATURES = SCIENCE_FRONT9_FEATURES + [
    "nc_mean",
    "last_block_instability",
]
SCIENCE_FRONT13_FEATURES = SCIENCE_FRONT11_FEATURES + [
    "tok_gini_tail",
    "nc_slope",
]
UNCERTAINTY9_FEATURES = [
    "tok_conf_prefix",
    "tok_conf_recency",
    "tok_gini_prefix",
    "tok_gini_tail",
    "tok_gini_slope",
    "tok_neg_entropy_prefix",
    "tok_neg_entropy_recency",
    "tok_selfcert_prefix",
    "tok_selfcert_recency",
]
TRAJECTORY5_FEATURES = list(TRAJ_FEATURES)

SEARCH_FAMILY_MAP: dict[str, list[str]] = {
    "fixed22": list(FIXED_FEATURE_NAMES),
    "all24": list(TOKEN_FEATURES) + list(TRAJ_FEATURES) + PREFIX_META_FEATURES + AVAILABILITY_FEATURES,
    "strong_core3": [
        "tok_conf_prefix",
        "tok_conf_recency",
        "traj_reflection_count",
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
    "uncertainty9": list(UNCERTAINTY9_FEATURES),
    "trajectory5": list(TRAJECTORY5_FEATURES),
    "window5": list(WINDOW5_FEATURES),
    "delta12": list(DELTA12_FEATURES),
    "dynamic17": list(DYNAMIC17_FEATURES),
    "science_front7": list(SCIENCE_FRONT7_FEATURES),
    "science_front9": list(SCIENCE_FRONT9_FEATURES),
    "science_front11": list(SCIENCE_FRONT11_FEATURES),
    "science_front13": list(SCIENCE_FRONT13_FEATURES),
    "wide46": list(WIDE46_FEATURES),
}
SVD_FAMILIES = {"fixed22", "wide46"}
XGB_FAMILIES = {"fixed22", "science_front11", "science_front13", "wide46"}
LINEAR_RAW_FAMILIES = ("fixed22",)
LINEAR_RR_FAMILIES = ("fixed22", "dynamic17", "science_front11", "science_front13", "wide46")

LINEAR_C_VALUES = (0.01, 0.10, 1.0, 10.0)
LINEAR_CLASS_WEIGHT = ("none", "balanced")
ELASTICNET_L1_RATIOS = (0.2, 0.5, 0.8)
SVD_C_VALUES = (0.01, 0.05, 0.20, 1.0)
SVD_RANKS = (4, 8, 12, 16, 24)
SVD_WHITEN = (False, True)
XGB_CONFIGS = (
    {
        "tag": "d3_lr003_n300_mc1_s07",
        "max_depth": 3,
        "learning_rate": 0.03,
        "n_estimators": 300,
        "min_child_weight": 1,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
    },
    {
        "tag": "d3_lr01_n100_mc5_s10",
        "max_depth": 3,
        "learning_rate": 0.10,
        "n_estimators": 100,
        "min_child_weight": 5,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    },
)


@dataclass(frozen=True)
class LinearSpec:
    model_name: str
    estimator_kind: str
    penalty: str
    solver: str
    use_random_seeds: bool
    l1_ratio: Optional[float] = None


LINEAR_SPECS = (
    LinearSpec("lr_l2", "logreg", "l2", "lbfgs", False),
    LinearSpec("lr_l1", "logreg", "l1", "saga", True),
    LinearSpec("elasticnet_lr", "logreg", "elasticnet", "saga", True, 0.5),
    LinearSpec("linear_svm", "linear_svm", "l2", "linear_svc", False),
)
LINEAR_SPEC_BY_NAME = {spec.model_name: spec for spec in LINEAR_SPECS}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(raw).split(",") if item.strip())


def _parse_anchor_csv(raw: str) -> tuple[int, ...]:
    out = []
    for item in _parse_csv(raw):
        value = int(float(item))
        if value not in {10, 20, 30, 40, 50, 60, 70, 80, 90, 100}:
            raise ValueError(f"Unsupported anchor percentage: {value}")
        out.append(value)
    return tuple(out)


def _mean_ignore_nan(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _safe_pct(value: Any) -> str:
    try:
        value_f = float(value)
    except Exception:
        return "N/A"
    if not math.isfinite(value_f):
        return "N/A"
    return f"{100.0 * value_f:.2f}%"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _class_weight_value(name: str) -> Optional[str]:
    return None if str(name) == "none" else "balanced"


def _group_folds(groups: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold

    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        return []
    splits = min(int(n_splits), int(unique_groups.size))
    if splits < 2:
        return []
    dummy_x = np.zeros((len(groups), 1), dtype=np.float64)
    gkf = GroupKFold(n_splits=splits)
    return list(gkf.split(dummy_x, groups=groups))


def _group_top1(scores: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    by_group: dict[str, list[int]] = {}
    for idx, group in enumerate(groups.tolist()):
        by_group.setdefault(str(group), []).append(int(idx))
    hits: list[float] = []
    for idxs in by_group.values():
        local_scores = np.asarray(scores[idxs], dtype=np.float64)
        local_y = np.asarray(y[idxs], dtype=np.int32)
        if local_scores.size == 0:
            continue
        best_idx = int(idxs[int(np.argmax(local_scores))])
        hits.append(float(y[best_idx]))
    if not hits:
        return float("nan")
    return float(np.mean(hits))


def _fit_scaled_linear_model(
    *,
    spec: LinearSpec,
    x: np.ndarray,
    y: np.ndarray,
    c_value: float,
    class_weight_name: str,
    seed: int,
) -> Optional[dict[str, Any]]:
    if x.shape[0] < 4 or x.shape[1] < 1:
        return None
    if np.unique(y).shape[0] < 2:
        return None

    scaler = StandardScaler(with_mean=True, with_std=True)
    x_scaled = scaler.fit_transform(x)
    if spec.estimator_kind == "logreg":
        kwargs: dict[str, Any] = {
            "C": float(c_value),
            "penalty": str(spec.penalty),
            "class_weight": _class_weight_value(str(class_weight_name)),
            "solver": str(spec.solver),
            "fit_intercept": True,
            "max_iter": 5000,
        }
        if spec.l1_ratio is not None:
            kwargs["l1_ratio"] = float(spec.l1_ratio)
        if spec.use_random_seeds:
            kwargs["random_state"] = int(seed)
        estimator = LogisticRegression(**kwargs)
    elif spec.estimator_kind == "linear_svm":
        estimator = LinearSVC(
            C=float(c_value),
            class_weight=_class_weight_value(str(class_weight_name)),
            dual=False,
            max_iter=5000,
            fit_intercept=True,
        )
    else:
        raise ValueError(f"Unsupported estimator kind: {spec.estimator_kind}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        estimator.fit(x_scaled, y)
    return {"scaler": scaler, "estimator": estimator}


def _predict_scaled_linear_model(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    x_scaled = model["scaler"].transform(x)
    return np.asarray(model["estimator"].decision_function(x_scaled), dtype=np.float64).reshape(-1)


def _seed_sequence(spec: LinearSpec, seeds: tuple[int, ...]) -> tuple[int, ...]:
    if spec.use_random_seeds:
        return tuple(int(v) for v in seeds)
    return (int(seeds[0]),)


def _linear_l1_ratio_grid(spec: LinearSpec) -> tuple[Optional[float], ...]:
    if str(spec.penalty) == "elasticnet":
        return tuple(float(v) for v in ELASTICNET_L1_RATIOS)
    return (None,)


def _search_linear_candidate(
    *,
    spec: LinearSpec,
    representation: str,
    x_rep: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seeds: tuple[int, ...],
) -> Optional[dict[str, Any]]:
    best: dict[str, Any] = {"cv_mean_auroc": float("-inf")}
    search_seeds = (int(seeds[0]),)
    for c_value in LINEAR_C_VALUES:
        for class_weight in LINEAR_CLASS_WEIGHT:
            for l1_ratio in _linear_l1_ratio_grid(spec):
                local_spec = spec if l1_ratio is None else LinearSpec(
                    model_name=spec.model_name,
                    estimator_kind=spec.estimator_kind,
                    penalty=spec.penalty,
                    solver=spec.solver,
                    use_random_seeds=spec.use_random_seeds,
                    l1_ratio=float(l1_ratio),
                )
                scores: list[float] = []
                for train_idx, test_idx in folds:
                    y_train = y[train_idx]
                    y_test = y[test_idx]
                    if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
                        continue
                    x_train = x_rep[train_idx]
                    x_test = x_rep[test_idx]
                    for seed in _seed_sequence(local_spec, search_seeds):
                        model = _fit_scaled_linear_model(
                            spec=local_spec,
                            x=x_train,
                            y=y_train,
                            c_value=float(c_value),
                            class_weight_name=str(class_weight),
                            seed=int(seed),
                        )
                        if model is None:
                            continue
                        try:
                            est_scores = _predict_scaled_linear_model(model, x_test)
                        except Exception:
                            continue
                        fold_auc = _auroc(est_scores, y_test)
                        if np.isfinite(fold_auc):
                            scores.append(float(fold_auc))
                cv_mean = _mean_ignore_nan(scores)
                if not np.isfinite(cv_mean):
                    continue
                tie_key = (
                    float(cv_mean),
                    -float(c_value),
                    0 if str(class_weight) == "none" else -1,
                    -1.0 if l1_ratio is None else -float(l1_ratio),
                )
                best_key = (
                    float(best["cv_mean_auroc"]),
                    -float(best.get("c_value", 1e9)),
                    0 if str(best.get("class_weight", "none")) == "none" else -1,
                    -1.0 if best.get("l1_ratio") is None else -float(best.get("l1_ratio", 1.0)),
                )
                if tie_key > best_key:
                    best = {
                        "route_family": "linear",
                        "model_name": str(local_spec.model_name),
                        "representation": str(representation),
                        "cv_mean_auroc": float(cv_mean),
                        "n_valid_scores": int(len(scores)),
                        "c_value": float(c_value),
                        "class_weight": str(class_weight),
                        "seed_values": [int(v) for v in _seed_sequence(local_spec, seeds)],
                    }
                    if l1_ratio is not None:
                        best["l1_ratio"] = float(l1_ratio)
    return best if np.isfinite(float(best["cv_mean_auroc"])) else None


def _search_svd_candidate(
    *,
    x_rep: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> Optional[dict[str, Any]]:
    from nad.ops.earlystop_svd import _cv_auroc_svd

    best: dict[str, Any] = {"cv_mean_auroc": float("-inf")}
    for rank in SVD_RANKS:
        for c_value in SVD_C_VALUES:
            for whiten in SVD_WHITEN:
                for class_weight in LINEAR_CLASS_WEIGHT:
                    cv_mean, n_valid = _cv_auroc_svd(
                        x=x_rep,
                        y=y,
                        groups=groups,
                        n_splits=int(n_splits),
                        rank=int(rank),
                        c_value=float(c_value),
                        whiten=bool(whiten),
                        class_weight_name=str(class_weight),
                        random_state=42,
                    )
                    if not np.isfinite(cv_mean):
                        continue
                    tie_key = (
                        float(cv_mean),
                        int(n_valid),
                        -int(rank),
                        -float(c_value),
                        0 if str(class_weight) == "none" else -1,
                        0 if not bool(whiten) else -1,
                    )
                    best_key = (
                        float(best["cv_mean_auroc"]),
                        int(best.get("n_valid_scores", 0)),
                        -int(best.get("rank", 10**9)),
                        -float(best.get("c_value", 1e9)),
                        0 if str(best.get("class_weight", "none")) == "none" else -1,
                        0 if not bool(best.get("whiten", False)) else -1,
                    )
                    if tie_key > best_key:
                        best = {
                            "route_family": "svd_lr",
                            "model_name": "svd_lr",
                            "representation": "raw+rank",
                            "cv_mean_auroc": float(cv_mean),
                            "n_valid_scores": int(n_valid),
                            "rank": int(rank),
                            "c_value": float(c_value),
                            "whiten": bool(whiten),
                            "class_weight": str(class_weight),
                            "seed_values": [42],
                        }
    return best if np.isfinite(float(best["cv_mean_auroc"])) else None


def _build_xgb_estimator(config: dict[str, Any], seed: int, n_jobs: int):
    from xgboost import XGBClassifier

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        max_depth=int(config["max_depth"]),
        learning_rate=float(config["learning_rate"]),
        n_estimators=int(config["n_estimators"]),
        min_child_weight=float(config["min_child_weight"]),
        subsample=float(config["subsample"]),
        colsample_bytree=float(config["colsample_bytree"]),
        random_state=int(seed),
        n_jobs=max(1, int(n_jobs)),
        verbosity=0,
    )


def _predict_positive_scores(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x), dtype=np.float64)
    return np.asarray(model.predict(x), dtype=np.float64)


def _search_xgb_candidate(
    *,
    representation: str,
    x_rep: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> Optional[dict[str, Any]]:
    best: dict[str, Any] = {"cv_mean_auroc": float("-inf")}
    for config in XGB_CONFIGS:
        scores: list[float] = []
        for train_idx, test_idx in folds:
            y_train = y[train_idx]
            y_test = y[test_idx]
            if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
                continue
            model = _build_xgb_estimator(config, seed=42, n_jobs=1)
            try:
                model.fit(x_rep[train_idx], y_train)
                est_scores = _predict_positive_scores(model, x_rep[test_idx])
            except Exception:
                continue
            fold_auc = _auroc(est_scores, y_test)
            if np.isfinite(fold_auc):
                scores.append(float(fold_auc))
        cv_mean = _mean_ignore_nan(scores)
        if not np.isfinite(cv_mean):
            continue
        tie_key = (
            float(cv_mean),
            -int(config["max_depth"]),
            -int(config["n_estimators"]),
            -float(config["learning_rate"]),
        )
        best_key = (
            float(best["cv_mean_auroc"]),
            -int(best.get("max_depth", 10**9)),
            -int(best.get("n_estimators", 10**9)),
            -float(best.get("learning_rate", 1e9)),
        )
        if tie_key > best_key:
            best = {
                "route_family": "xgboost",
                "model_name": "xgboost",
                "representation": str(representation),
                "cv_mean_auroc": float(cv_mean),
                "n_valid_scores": int(len(scores)),
                "config_tag": str(config["tag"]),
                "max_depth": int(config["max_depth"]),
                "learning_rate": float(config["learning_rate"]),
                "n_estimators": int(config["n_estimators"]),
                "min_child_weight": float(config["min_child_weight"]),
                "subsample": float(config["subsample"]),
                "colsample_bytree": float(config["colsample_bytree"]),
                "seed_values": [42, 101, 202],
            }
    return best if np.isfinite(float(best["cv_mean_auroc"])) else None


def _family_indices(family_name: str) -> list[int]:
    features = SEARCH_FAMILY_MAP[str(family_name)]
    return [int(FEATURE_TO_INDEX[name]) for name in features]


def _search_source_anchor(job: dict[str, Any]) -> list[dict[str, Any]]:
    source_anchor_pct = int(job["source_anchor_pct"])
    source_idx = POSITION_TO_INDEX[float(source_anchor_pct) / 100.0]
    table = job["train_tables"][source_idx]
    x_raw = np.asarray(table["x_raw"], dtype=np.float64)
    x_rank = np.asarray(table["x_rank"], dtype=np.float64)
    y = np.asarray(table["y"], dtype=np.int32)
    groups = np.asarray(table["groups"], dtype=object)
    folds = _group_folds(groups, int(job["n_splits"]))
    if not folds:
        raise ValueError(f"science@{source_anchor_pct}% has insufficient CV folds")

    seeds = tuple(int(v) for v in job["seed_values"])
    out: list[dict[str, Any]] = []
    rep_cache: dict[tuple[str, str], np.ndarray] = {}

    def rep_for(family_name: str, representation: str) -> np.ndarray:
        key = (str(family_name), str(representation))
        if key not in rep_cache:
            rep_cache[key] = _build_representation(
                x_raw=x_raw,
                x_rank=x_rank,
                feature_indices=_family_indices(str(family_name)),
                representation=str(representation),
            )
        return rep_cache[key]

    for family_name in LINEAR_RAW_FAMILIES:
        x_rep_raw = rep_for(family_name, "raw")
        linear_raw = _search_linear_candidate(
            spec=LINEAR_SPEC_BY_NAME["lr_l2"],
            representation="raw",
            x_rep=x_rep_raw,
            y=y,
            folds=folds,
            seeds=seeds,
        )
        if linear_raw is not None:
            linear_raw.update(
                {
                    "source_anchor_pct": int(source_anchor_pct),
                    "family_name": str(family_name),
                    "feature_count": int(len(SEARCH_FAMILY_MAP[str(family_name)])),
                    "feature_names": list(SEARCH_FAMILY_MAP[str(family_name)]),
                }
            )
            out.append(linear_raw)

    for family_name in LINEAR_RR_FAMILIES:
        x_rep_rr = rep_for(family_name, "raw+rank")
        for spec in LINEAR_SPECS:
            linear_rr = _search_linear_candidate(
                spec=spec,
                representation="raw+rank",
                x_rep=x_rep_rr,
                y=y,
                folds=folds,
                seeds=seeds,
            )
            if linear_rr is not None:
                linear_rr.update(
                    {
                        "source_anchor_pct": int(source_anchor_pct),
                        "family_name": str(family_name),
                        "feature_count": int(len(SEARCH_FAMILY_MAP[str(family_name)])),
                        "feature_names": list(SEARCH_FAMILY_MAP[str(family_name)]),
                    }
                )
                out.append(linear_rr)

    for family_name in sorted(SVD_FAMILIES):
        x_rep_rr = rep_for(family_name, "raw+rank")
        svd_candidate = _search_svd_candidate(
            x_rep=x_rep_rr,
            y=y,
            groups=groups,
            n_splits=int(job["n_splits"]),
        )
        if svd_candidate is not None:
            svd_candidate.update(
                {
                    "source_anchor_pct": int(source_anchor_pct),
                    "family_name": str(family_name),
                    "feature_count": int(len(SEARCH_FAMILY_MAP[str(family_name)])),
                    "feature_names": list(SEARCH_FAMILY_MAP[str(family_name)]),
                }
            )
            out.append(svd_candidate)

    for family_name in sorted(XGB_FAMILIES):
        for representation in ("raw", "raw+rank"):
            x_rep = rep_for(family_name, representation)
            xgb_candidate = _search_xgb_candidate(
                representation=representation,
                x_rep=x_rep,
                y=y,
                folds=folds,
            )
            if xgb_candidate is not None:
                xgb_candidate.update(
                    {
                        "source_anchor_pct": int(source_anchor_pct),
                        "family_name": str(family_name),
                        "feature_count": int(len(SEARCH_FAMILY_MAP[str(family_name)])),
                        "feature_names": list(SEARCH_FAMILY_MAP[str(family_name)]),
                    }
                )
                out.append(xgb_candidate)

    for idx, row in enumerate(out):
        row["candidate_id"] = (
            f"src{int(source_anchor_pct):03d}__{row['route_family']}__{row['model_name']}"
            f"__{row['family_name']}__{row['representation'].replace('+', 'p')}__{idx:03d}"
        )
    return out


def _fit_candidate_route(
    *,
    candidate: dict[str, Any],
    table: dict[str, np.ndarray],
) -> dict[str, Any]:
    x_raw = np.asarray(table["x_raw"], dtype=np.float64)
    x_rank = np.asarray(table["x_rank"], dtype=np.float64)
    y = np.asarray(table["y"], dtype=np.int32)
    feature_indices = _family_indices(str(candidate["family_name"]))
    x_rep = _build_representation(
        x_raw=x_raw,
        x_rank=x_rank,
        feature_indices=feature_indices,
        representation=str(candidate["representation"]),
    )

    route = dict(candidate)
    route["feature_indices"] = list(feature_indices)
    route["models"] = []

    if str(candidate["route_family"]) == "linear":
        base_spec = LINEAR_SPEC_BY_NAME[str(candidate["model_name"])]
        spec = LinearSpec(
            model_name=base_spec.model_name,
            estimator_kind=base_spec.estimator_kind,
            penalty=base_spec.penalty,
            solver=base_spec.solver,
            use_random_seeds=base_spec.use_random_seeds,
            l1_ratio=(
                None
                if candidate.get("l1_ratio") is None
                else float(candidate["l1_ratio"])
            ),
        )
        for seed in _seed_sequence(spec, tuple(int(v) for v in candidate["seed_values"])):
            model = _fit_scaled_linear_model(
                spec=spec,
                x=x_rep,
                y=y,
                c_value=float(candidate["c_value"]),
                class_weight_name=str(candidate["class_weight"]),
                seed=int(seed),
            )
            if model is not None:
                route["models"].append(model)
        if not route["models"]:
            raise RuntimeError(f"Failed full-fit for {candidate['candidate_id']}")
        return route

    if str(candidate["route_family"]) == "svd_lr":
        model = _fit_svd_lr_model(
            x=x_rep,
            y=y,
            rank=int(candidate["rank"]),
            c_value=float(candidate["c_value"]),
            whiten=bool(candidate["whiten"]),
            class_weight_name=str(candidate["class_weight"]),
            random_state=42,
        )
        if model is None:
            raise RuntimeError(f"Failed SVD full-fit for {candidate['candidate_id']}")
        route["models"] = [model]
        return route

    if str(candidate["route_family"]) == "xgboost":
        config = {
            "max_depth": int(candidate["max_depth"]),
            "learning_rate": float(candidate["learning_rate"]),
            "n_estimators": int(candidate["n_estimators"]),
            "min_child_weight": float(candidate["min_child_weight"]),
            "subsample": float(candidate["subsample"]),
            "colsample_bytree": float(candidate["colsample_bytree"]),
        }
        for seed in tuple(int(v) for v in candidate["seed_values"]):
            model = _build_xgb_estimator(config, seed=int(seed), n_jobs=1)
            model.fit(x_rep, y)
            route["models"].append(model)
        return route

    raise ValueError(f"Unsupported route family: {candidate['route_family']}")


def _score_route(route: dict[str, Any], x_raw: np.ndarray) -> np.ndarray:
    representation = str(route["representation"])
    x_rank = np.zeros_like(x_raw, dtype=np.float64)
    if representation == "raw+rank":
        x_rank = _rank_transform_matrix(x_raw)
    x_rep = _build_representation(
        x_raw=x_raw,
        x_rank=x_rank,
        feature_indices=[int(v) for v in route["feature_indices"]],
        representation=representation,
    )
    if str(route["route_family"]) == "linear":
        scores = [_predict_scaled_linear_model(model, x_rep) for model in route["models"]]
        return np.mean(np.vstack(scores), axis=0, dtype=np.float64)
    if str(route["route_family"]) == "svd_lr":
        scores = [_predict_svd_lr(model, x_rep) for model in route["models"]]
        return np.mean(np.vstack(scores), axis=0, dtype=np.float64)
    if str(route["route_family"]) == "xgboost":
        scores = [_predict_positive_scores(model, x_rep) for model in route["models"]]
        return np.mean(np.vstack(scores), axis=0, dtype=np.float64)
    raise ValueError(f"Unsupported route family: {route['route_family']}")


def _evaluate_candidates_on_holdout(
    *,
    routes_by_id: dict[str, dict[str, Any]],
    holdout_tables: list[dict[str, np.ndarray]],
    target_anchor_pcts: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_anchor_pct in target_anchor_pcts:
        target_idx = POSITION_TO_INDEX[float(target_anchor_pct) / 100.0]
        table = holdout_tables[target_idx]
        x_raw = np.asarray(table["x_raw"], dtype=np.float64)
        y = np.asarray(table["y"], dtype=np.int32)
        groups = np.asarray(table["groups"], dtype=object)
        for candidate_id, route in routes_by_id.items():
            scores = _score_route(route, x_raw)
            rows.append(
                {
                    "candidate_id": str(candidate_id),
                    "target_anchor_pct": int(target_anchor_pct),
                    "source_anchor_pct": int(route["source_anchor_pct"]),
                    "route_family": str(route["route_family"]),
                    "model_name": str(route["model_name"]),
                    "family_name": str(route["family_name"]),
                    "representation": str(route["representation"]),
                    "feature_count": int(route["feature_count"]),
                    "cv_mean_auroc": float(route["cv_mean_auroc"]),
                    "holdout_auroc": float(_auroc(scores, y)),
                    "holdout_group_top1": float(_group_top1(scores, y, groups)),
                    "is_task_specific": bool(int(route["source_anchor_pct"]) == int(target_anchor_pct)),
                }
            )
    return rows


def _candidate_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        float(row["holdout_auroc"]),
        float(row["holdout_group_top1"]),
        -abs(int(row["source_anchor_pct"]) - int(row["target_anchor_pct"])),
        1 if str(row["route_family"]) != "xgboost" else 0,
        -int(row["feature_count"]),
        str(row["candidate_id"]),
    )


def _best_row(rows: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not rows:
        return None
    return max(rows, key=_candidate_sort_key)


def _select_rows(
    *,
    holdout_rows: list[dict[str, Any]],
    target_anchor_pcts: tuple[int, ...],
    transfer_margin: float,
    tree_margin: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_specific_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for target_anchor_pct in target_anchor_pcts:
        target_rows = [row for row in holdout_rows if int(row["target_anchor_pct"]) == int(target_anchor_pct)]
        task_best = _best_row([row for row in target_rows if bool(row["is_task_specific"])])
        best_linear = _best_row([row for row in target_rows if str(row["route_family"]) != "xgboost"])
        best_any = _best_row(target_rows)
        if task_best is None or best_any is None:
            raise ValueError(f"Missing candidate rows for target={target_anchor_pct}")

        chosen = dict(best_any)
        selection_notes: list[str] = []
        if best_linear is not None and str(chosen["route_family"]) == "xgboost":
            if float(chosen["holdout_auroc"]) <= float(best_linear["holdout_auroc"]) + float(tree_margin):
                chosen = dict(best_linear)
                selection_notes.append(f"tree_margin->{best_linear['candidate_id']}")
        if int(chosen["source_anchor_pct"]) != int(target_anchor_pct):
            if float(chosen["holdout_auroc"]) <= float(task_best["holdout_auroc"]) + float(transfer_margin):
                chosen = dict(task_best)
                selection_notes.append(f"transfer_margin->{task_best['candidate_id']}")

        task_row = dict(task_best)
        task_row["selection_type"] = "task_specific"
        task_row["selection_notes"] = "task_specific_best"
        task_specific_rows.append(task_row)

        chosen["selection_type"] = "dense_final"
        chosen["selection_notes"] = "|".join(selection_notes) if selection_notes else "best_holdout"
        final_rows.append(chosen)
    return task_specific_rows, final_rows


def _build_selected_score_fn(routes: list[dict[str, Any]]):
    route_map = {
        POSITION_TO_INDEX[float(int(route["target_anchor_pct"]) / 100.0)]: route
        for route in routes
    }

    def _score(domain: str, position_index: int, x_raw: np.ndarray) -> np.ndarray:
        if str(domain) != SCIENCE_DOMAIN:
            raise ValueError(f"Unexpected domain for science dense search: {domain}")
        route = route_map[int(position_index)]
        return _score_route(route, x_raw)

    return _score


def _read_science_reference_auc() -> float:
    payload = _load_json(CURRENT_SCIENCE_EVAL)
    return float(payload["validate"]["science"]["candidate"]["aggregate"]["auc_of_auroc"])


def _evaluate_reference_science_bundle(
    holdout_store: list[dict[str, Any]],
    position_values: tuple[float, ...],
) -> dict[str, Any]:
    bundle = load_earlystop_svd_bundle(CURRENT_SCIENCE_BUNDLE)
    return evaluate_method_from_feature_store(
        method_name="es_svd_science_rr_r2_20260412",
        feature_store=holdout_store,
        position_values=position_values,
        score_fn=make_svd_bundle_score_fn(bundle),
    )


def _collect_required_features() -> set[str]:
    required: set[str] = set()
    for features in SEARCH_FAMILY_MAP.values():
        required.update(str(name) for name in features)
    return required


def _science_cache_keys(cache_root: str) -> set[str]:
    return {
        str(entry.cache_key)
        for entry in discover_cache_entries(str(cache_root))
        if str(get_domain(entry.dataset_name)) == SCIENCE_DOMAIN
    }


def _load_labeled_science_store(
    *,
    main_cache_root: str,
    extra_cache_root: str,
    feature_cache_dir: Optional[Path],
    refresh_feature_cache: bool,
    feature_workers: int,
    feature_chunk_problems: int,
    required_feature_names: set[str],
    max_problems_per_cache: Optional[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stores: list[dict[str, Any]] = []
    state: dict[str, Any] = {}
    for source_name, cache_root in (("cache", main_cache_root), ("cache_train", extra_cache_root)):
        include_cache_keys = {"DS-R1/gpqa"}
        store, cache_path, cache_status = _load_or_build_qualified_feature_store(
            source_name=str(source_name),
            cache_root=str(cache_root),
            positions=POSITIONS,
            required_feature_names=required_feature_names,
            max_problems_per_cache=max_problems_per_cache,
            feature_workers=int(feature_workers),
            chunk_problems=int(feature_chunk_problems),
            feature_cache_dir=feature_cache_dir,
            refresh_feature_cache=bool(refresh_feature_cache),
            include_cache_keys=include_cache_keys,
        )
        fixed = [_ensure_training_payload(payload, source_name=str(source_name)) for payload in list(store)]
        science_only = [payload for payload in fixed if str(payload.get("domain")) == SCIENCE_DOMAIN]
        stores.extend(science_only)
        state[source_name] = {
            "status": str(cache_status),
            "path": None if cache_path is None else _display_path(cache_path),
            "num_payloads": int(len(science_only)),
            "cache_root": str(cache_root),
        }
    return stores, state


def _load_blind_science_feature_store(
    *,
    blind_cache_root: str,
    required_feature_names: set[str],
    feature_workers: int,
    feature_chunk_problems: int,
    feature_cache_dir: Optional[Path],
    refresh_feature_cache: bool,
    max_problems: Optional[int],
) -> tuple[list[dict[str, Any]], Path | None, str]:
    include_cache_keys = _science_cache_keys(blind_cache_root)
    store, cache_path, cache_status = _load_or_build_blind_feature_store(
        cache_root=str(blind_cache_root),
        positions=POSITIONS,
        required_feature_names=required_feature_names,
        max_problems=max_problems,
        reflection_threshold=float(DEFAULT_REFLECTION_THRESHOLD),
        workers=int(feature_workers),
        feature_chunk_problems=int(feature_chunk_problems),
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(refresh_feature_cache),
        include_cache_keys=include_cache_keys,
        exclude_cache_keys=None,
    )
    science_only = [payload for payload in list(store) if str(payload.get("domain")) == SCIENCE_DOMAIN]
    return science_only, cache_path, str(cache_status)


def _problem_scores_from_payload_partial(
    *,
    payload: dict[str, Any],
    score_fn,
    base_problem_scores: dict[str, dict[str, list[float]]],
    target_position_indices: set[int],
) -> dict[str, dict[str, list[float]]]:
    problem_scores: dict[str, dict[str, list[float]]] = {}
    tensor = payload["tensor"]
    sample_ids_all = payload["sample_ids"]
    problem_ids = payload["problem_ids"]
    problem_offsets = payload["problem_offsets"]
    n_positions = len(EARLY_STOP_POSITIONS)

    for problem_idx, problem_id in enumerate(problem_ids):
        problem_id_str = str(problem_id)
        start = int(problem_offsets[problem_idx])
        end = int(problem_offsets[problem_idx + 1])
        problem_tensor = tensor[start:end]
        sample_ids = sample_ids_all[start:end]
        base_run_map = base_problem_scores[problem_id_str]
        run_scores = {
            str(sample_id): [float(v) for v in base_run_map[str(sample_id)]]
            for sample_id in sample_ids.tolist()
        }
        for values in run_scores.values():
            if len(values) != n_positions:
                raise ValueError(f"Unexpected base score length for problem={problem_id_str}")

        for pos_idx in sorted(int(v) for v in target_position_indices):
            x_raw = problem_tensor[:, pos_idx, :]
            scores = score_fn(payload["domain"], pos_idx, x_raw)
            for row_idx, sample_id in enumerate(sample_ids.tolist()):
                run_scores[str(sample_id)][pos_idx] = float(scores[row_idx])

        problem_scores[problem_id_str] = run_scores
    return problem_scores


def _patch_base_submission(
    *,
    base_json: Path,
    science_score_map: dict[str, dict[str, dict[str, list[float]]]],
    out_path: Path,
    method_name: str,
) -> dict[str, Any]:
    payload = _load_json(base_json)
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        raise ValueError(f"Invalid base payload: {base_json}")
    for cache_key, problem_scores in science_score_map.items():
        if cache_key not in scores:
            raise KeyError(f"Base payload missing science cache key: {cache_key}")
        scores[cache_key] = problem_scores
    payload["method_name"] = str(method_name)
    validation = validate_earlystop_payload(payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_earlystop_payload(payload, out_path)
    return {"validation": validation, "path": _display_path(out_path)}


def _compare_science_delta(
    *,
    base_json: Path,
    science_score_map: dict[str, dict[str, dict[str, list[float]]]],
) -> dict[str, Any]:
    payload = _load_json(base_json)
    out: dict[str, Any] = {}
    for cache_key, problem_scores in science_score_map.items():
        base_scores = payload["scores"][cache_key]
        deltas: list[float] = []
        for problem_id, run_map in problem_scores.items():
            base_run_map = base_scores[problem_id]
            for sample_id, values in run_map.items():
                base_vals = base_run_map[sample_id]
                for lhs, rhs in zip(values, base_vals):
                    deltas.append(abs(float(lhs) - float(rhs)))
        arr = np.asarray(deltas, dtype=np.float64)
        out[cache_key] = {
            "mean_abs_delta": float(np.mean(arr)) if arr.size else 0.0,
            "max_abs_delta": float(np.max(arr)) if arr.size else 0.0,
            "num_values": int(arr.size),
        }
    return out


def _render_doc(
    *,
    path: Path,
    manifest: dict[str, Any],
) -> None:
    holdout = manifest["holdout"]
    selected_rows = manifest["selected_rows"]
    final_anchor_pct = max(int(v) for v in manifest["protocol"]["target_anchor_pcts"])
    final_auroc_key = f"auroc@{final_anchor_pct}%"
    final_stop_key = f"stop_acc@{final_anchor_pct}%"
    lines = [
        "# Science Dense Slot Search (2026-04-16)",
        "",
        "## Goal",
        "",
        "- Keep the user-selected aggressive coding routes intact.",
        "- Re-search `science/gpqa` on the user-targeted slots instead of reusing the frozen science patch.",
        "- Search over `source anchor × feature family × model family`, then patch only the targeted blind science slots back into the aggressive submission.",
        "",
        "## Why this run differs",
        "",
        "- Labeled science features are rebuilt from current code, not the old `r2` prebuilt cache.",
        "- The rebuilt bank uses the current `47`-dimensional EarlyStop features, so the search can test post-`r2` window / instability / tail-delta signals.",
        "- The candidate grid includes science front-half families centered on `tok_conf_*`, `prefix_best_window_quality`, `tail_q10`, `traj_continuity`, `nc_mean`, and `last_block_instability`.",
        "- Direct linear search stays focused on `raw+rank` linear heads over the strongest front-half families, keeping the search targeted without falling back to the older frozen route.",
        "- Selection is dense: each target slot can choose a different training source anchor.",
        "- Untouched science slots inherit the base aggressive submission values.",
        "",
        "## Protocol",
        "",
        f"- `holdout split`: `85/15`, grouped by `dataset + problem_id`, `split_seed={manifest['protocol']['split_seed']}`.",
        f"- `reflection threshold`: `{manifest['protocol']['reflection_threshold']:.2f}`.",
        f"- `source anchors`: `{', '.join(str(v) for v in manifest['protocol']['source_anchor_pcts'])}`.",
        f"- `target anchors`: `{', '.join(str(v) for v in manifest['protocol']['target_anchor_pcts'])}`.",
        f"- `candidate families`: `{', '.join(manifest['protocol']['family_names'])}`.",
        f"- `transfer margin`: `{manifest['protocol']['transfer_margin']:.4f}` AUROC.",
        f"- `tree margin`: `{manifest['protocol']['tree_margin']:.4f}` AUROC.",
        "",
        "## Holdout Readout",
        "",
        f"| Slice | AUC of AUROC | AUC of SelAcc | AUROC@{final_anchor_pct}% | Stop Acc@{final_anchor_pct}% |",
        "|---|---:|---:|---:|---:|",
        "| `science r2` | {auc} | {sel} | {a100} | {s100} |".format(
            auc=_safe_pct(holdout["reference_r2"]["aggregate"]["auc_of_auroc"]),
            sel=_safe_pct(holdout["reference_r2"]["aggregate"]["auc_of_selacc"]),
            a100=_safe_pct(holdout["reference_r2"]["aggregate"].get(final_auroc_key)),
            s100=_safe_pct(holdout["reference_r2"]["aggregate"].get(final_stop_key)),
        ),
        "| `dense task-specific` | {auc} | {sel} | {a100} | {s100} |".format(
            auc=_safe_pct(holdout["task_specific"]["aggregate"]["auc_of_auroc"]),
            sel=_safe_pct(holdout["task_specific"]["aggregate"]["auc_of_selacc"]),
            a100=_safe_pct(holdout["task_specific"]["aggregate"].get(final_auroc_key)),
            s100=_safe_pct(holdout["task_specific"]["aggregate"].get(final_stop_key)),
        ),
        "| `dense final` | {auc} | {sel} | {a100} | {s100} |".format(
            auc=_safe_pct(holdout["dense_final"]["aggregate"]["auc_of_auroc"]),
            sel=_safe_pct(holdout["dense_final"]["aggregate"]["auc_of_selacc"]),
            a100=_safe_pct(holdout["dense_final"]["aggregate"].get(final_auroc_key)),
            s100=_safe_pct(holdout["dense_final"]["aggregate"].get(final_stop_key)),
        ),
        "",
        "## Selected Routes",
        "",
        "| Target | Source | Route | Family | Rep | Holdout AUROC | SelAcc | Note |",
        "|---|---:|---|---|---|---:|---:|---|",
    ]
    for row in selected_rows:
        lines.append(
            "| {target}% | {source}% | {route} | {family} | {rep} | {auroc:.4f} | {selacc:.4f} | {note} |".format(
                target=int(row["target_anchor_pct"]),
                source=int(row["source_anchor_pct"]),
                route=str(row["model_name"]),
                family=str(row["family_name"]),
                rep=str(row["representation"]),
                auroc=float(row["holdout_auroc"]),
                selacc=float(row["holdout_group_top1"]),
                note=str(row["selection_notes"]),
            )
        )
    lines.extend(
        [
            "",
            "## Front-Half Readout",
            "",
        ]
    )
    early_rows = [row for row in selected_rows if int(row["target_anchor_pct"]) <= 50]
    for row in early_rows:
        lines.append(
            "- `{target}%`: source `{source}%`, `{route}` on `{family}` (`{rep}`), holdout AUROC `{auroc:.4f}`.".format(
                target=int(row["target_anchor_pct"]),
                source=int(row["source_anchor_pct"]),
                route=str(row["model_name"]),
                family=str(row["family_name"]),
                rep=str(row["representation"]),
                auroc=float(row["holdout_auroc"]),
            )
        )
    lines.extend(
        [
            "",
            "## Export",
            "",
            f"- `patched submission`: `{manifest['artifacts']['patched_submission']}`",
            f"- `candidate csv`: `{manifest['artifacts']['candidate_csv']}`",
            f"- `selected csv`: `{manifest['artifacts']['selected_csv']}`",
            f"- `eval json`: `{manifest['artifacts']['eval_json']}`",
            "",
            "## Reading",
            "",
            "- If `dense final` beats `dense task-specific`, the gain comes from cross-anchor transfer and/or wider feature families, not from simply densifying the old route.",
            "- If early slots choose `science_front11`, `science_front13`, or `wide46`, that is direct evidence that the newer front-half science features matter.",
            "- The patched JSON is the file to score externally; all other artifacts are there to justify how it was chosen.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a science-only dense 10-slot early-stop search and patch blind science caches")
    ap.add_argument("--main-cache-root", default="MUI_HUB/cache")
    ap.add_argument("--extra-cache-root", default="MUI_HUB/cache_train")
    ap.add_argument("--blind-cache-root", default="/home/jovyan/public-ro/MUI_HUB/cache_test")
    ap.add_argument("--feature-cache-dir", default="results/cache/science_dense_slot_search/labeled")
    ap.add_argument("--blind-feature-cache-dir", default="results/cache/science_dense_slot_search/blind")
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--refresh-blind-feature-cache", action="store_true")
    ap.add_argument("--feature-workers", type=int, default=8)
    ap.add_argument("--feature-chunk-problems", type=int, default=DEFAULT_FEATURE_CHUNK_PROBLEMS)
    ap.add_argument("--max-problems-per-cache", type=int, default=0, help="0 means all problems")
    ap.add_argument("--search-workers", type=int, default=10)
    ap.add_argument("--holdout-split", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed-values", default="42,101,202")
    ap.add_argument("--source-anchors", default="10,20,30,40,50,60,70,80,90,100")
    ap.add_argument("--target-anchors", default="10,20,30,40,50,60,70,80,90,100")
    ap.add_argument("--transfer-margin", type=float, default=0.0010)
    ap.add_argument("--tree-margin", type=float, default=0.0025)
    ap.add_argument("--base-json", default=str(BASE_AGGRESSIVE_JSON))
    ap.add_argument(
        "--out-json",
        default="submission/EarlyStop/es_svd_ms_rr_r1__coding_rotation_adapter_aggressive_tokenonly_late70100__science_dense_slot_search_20260416.json",
    )
    ap.add_argument(
        "--method-name",
        default="es_svd_ms_rr_r1__coding_rotation_adapter_aggressive_tokenonly_late70100__science_dense_slot_search_20260416",
    )
    ap.add_argument(
        "--candidate-csv",
        default="results/tables/science_dense_slot_search_candidates_20260416.csv",
    )
    ap.add_argument(
        "--selected-csv",
        default="results/tables/science_dense_slot_search_selected_20260416.csv",
    )
    ap.add_argument(
        "--eval-json",
        default="results/scans/earlystop/science_dense_slot_search_20260416_eval.json",
    )
    ap.add_argument(
        "--manifest-json",
        default="results/tables/science_dense_slot_search_manifest_20260416.json",
    )
    ap.add_argument(
        "--doc-out",
        default="docs/SCIENCE_DENSE_SLOT_SEARCH_20260416.md",
    )
    args = ap.parse_args()

    source_anchor_pcts = _parse_anchor_csv(args.source_anchors)
    target_anchor_pcts = _parse_anchor_csv(args.target_anchors)
    eval_positions = tuple(float(int(v)) / 100.0 for v in target_anchor_pcts)
    target_position_indices = {
        POSITION_TO_INDEX[float(int(v)) / 100.0]
        for v in target_anchor_pcts
    }
    seed_values = tuple(int(v) for v in _parse_csv(args.seed_values)) or (42, 101, 202)
    required_feature_names = _collect_required_features()
    max_problems_per_cache = None if int(args.max_problems_per_cache) <= 0 else int(args.max_problems_per_cache)

    labeled_feature_cache_dir = None
    if str(args.feature_cache_dir).strip().lower() not in {"", "none", "off"}:
        labeled_feature_cache_dir = (REPO_ROOT / str(args.feature_cache_dir)).resolve()

    blind_feature_cache_dir = None
    if str(args.blind_feature_cache_dir).strip().lower() not in {"", "none", "off"}:
        blind_feature_cache_dir = (REPO_ROOT / str(args.blind_feature_cache_dir)).resolve()

    labeled_store, labeled_cache_state = _load_labeled_science_store(
        main_cache_root=_resolve_path(str(args.main_cache_root)),
        extra_cache_root=_resolve_path(str(args.extra_cache_root)),
        feature_cache_dir=labeled_feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_feature_cache),
        feature_workers=int(args.feature_workers),
        feature_chunk_problems=int(args.feature_chunk_problems),
        required_feature_names=required_feature_names,
        max_problems_per_cache=max_problems_per_cache,
    )
    if not labeled_store:
        raise ValueError("No labeled science feature-store payloads were loaded")

    holdout_problem_map, holdout_problem_summary = _build_holdout_problem_map(
        labeled_store,
        holdout_split=float(args.holdout_split),
        split_seed=int(args.split_seed),
    )
    train_store, holdout_store, full_store = _split_feature_store(
        labeled_store,
        holdout_problem_map=holdout_problem_map,
    )

    train_tables = _build_domain_training_tables(train_store, POSITIONS)
    holdout_tables = _build_domain_training_tables(holdout_store, POSITIONS)
    full_tables = _build_domain_training_tables(full_store, POSITIONS)

    print(
        f"[data] labeled science payloads={len(labeled_store)} train_payloads={len(train_store)} holdout_payloads={len(holdout_store)}",
        flush=True,
    )
    print(
        f"[data] train_groups={len(np.unique(train_tables[0]['groups']))} holdout_groups={len(np.unique(holdout_tables[0]['groups']))}",
        flush=True,
    )

    search_jobs = [
        {
            "source_anchor_pct": int(anchor_pct),
            "train_tables": train_tables,
            "n_splits": int(args.n_splits),
            "seed_values": list(int(v) for v in seed_values),
        }
        for anchor_pct in source_anchor_pcts
    ]

    candidate_specs: list[dict[str, Any]] = []
    worker_count = max(1, min(int(args.search_workers), len(search_jobs), 16))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(_search_source_anchor, job): int(job["source_anchor_pct"])
            for job in search_jobs
        }
        for future in as_completed(future_map):
            source_anchor_pct = future_map[future]
            rows = future.result()
            candidate_specs.extend(rows)
            print(
                f"[search] source={source_anchor_pct}% candidates={len(rows)}",
                flush=True,
            )

    if not candidate_specs:
        raise ValueError("Dense search produced no candidate specifications")

    routes_by_id: dict[str, dict[str, Any]] = {}
    for candidate in candidate_specs:
        source_idx = POSITION_TO_INDEX[float(int(candidate["source_anchor_pct"])) / 100.0]
        route = _fit_candidate_route(candidate=candidate, table=train_tables[source_idx])
        routes_by_id[str(candidate["candidate_id"])] = route
    print(f"[fit] train-split candidate routes={len(routes_by_id)}", flush=True)

    candidate_holdout_rows = _evaluate_candidates_on_holdout(
        routes_by_id=routes_by_id,
        holdout_tables=holdout_tables,
        target_anchor_pcts=target_anchor_pcts,
    )
    task_specific_rows, selected_rows = _select_rows(
        holdout_rows=candidate_holdout_rows,
        target_anchor_pcts=target_anchor_pcts,
        transfer_margin=float(args.transfer_margin),
        tree_margin=float(args.tree_margin),
    )

    selected_route_ids = {str(row["candidate_id"]) for row in selected_rows}
    selected_holdout_rows = [row for row in candidate_holdout_rows if str(row["candidate_id"]) in selected_route_ids]
    candidate_csv = REPO_ROOT / str(args.candidate_csv)
    selected_csv = REPO_ROOT / str(args.selected_csv)
    _write_csv(candidate_csv, candidate_holdout_rows)
    _write_csv(selected_csv, selected_rows)

    task_selected_routes = []
    for row in task_specific_rows:
        route = dict(routes_by_id[str(row["candidate_id"])])
        route["target_anchor_pct"] = int(row["target_anchor_pct"])
        task_selected_routes.append(route)
    dense_selected_routes = []
    for row in selected_rows:
        route = dict(routes_by_id[str(row["candidate_id"])])
        route["target_anchor_pct"] = int(row["target_anchor_pct"])
        dense_selected_routes.append(route)
    task_score_fn = _build_selected_score_fn(task_selected_routes)
    dense_score_fn = _build_selected_score_fn(dense_selected_routes)

    reference_r2_eval = _evaluate_reference_science_bundle(
        holdout_store,
        position_values=eval_positions,
    )
    task_eval = evaluate_method_from_feature_store(
        method_name="science_dense_task_specific_proxy",
        feature_store=holdout_store,
        position_values=eval_positions,
        score_fn=task_score_fn,
    )
    dense_eval = evaluate_method_from_feature_store(
        method_name="science_dense_final_proxy",
        feature_store=holdout_store,
        position_values=eval_positions,
        score_fn=dense_score_fn,
    )

    fullfit_routes_by_target: list[dict[str, Any]] = []
    for selected in selected_rows:
        selected_candidate = next(row for row in candidate_specs if str(row["candidate_id"]) == str(selected["candidate_id"]))
        source_idx = POSITION_TO_INDEX[float(int(selected_candidate["source_anchor_pct"])) / 100.0]
        full_route = _fit_candidate_route(candidate=selected_candidate, table=full_tables[source_idx])
        full_route["target_anchor_pct"] = int(selected["target_anchor_pct"])
        fullfit_routes_by_target.append(full_route)
    blind_score_fn = _build_selected_score_fn(fullfit_routes_by_target)

    blind_feature_store, blind_cache_path, blind_cache_status = _load_blind_science_feature_store(
        blind_cache_root=str(args.blind_cache_root),
        required_feature_names=required_feature_names,
        feature_workers=int(args.feature_workers),
        feature_chunk_problems=int(args.feature_chunk_problems),
        feature_cache_dir=blind_feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_blind_feature_cache),
        max_problems=max_problems_per_cache,
    )
    base_payload = _load_json(Path(args.base_json))
    base_scores = base_payload["scores"]
    science_score_map: dict[str, dict[str, dict[str, list[float]]]] = {}
    for payload in blind_feature_store:
        cache_key = str(payload.get("cache_key"))
        problem_scores = _problem_scores_from_payload_partial(
            payload=payload,
            score_fn=blind_score_fn,
            base_problem_scores=base_scores[cache_key],
            target_position_indices=target_position_indices,
        )
        science_score_map[cache_key] = problem_scores
        print(
            f"[blind] cache={cache_key} problems={len(problem_scores)} samples={sum(len(v) for v in problem_scores.values())}",
            flush=True,
        )

    patched_out = REPO_ROOT / str(args.out_json)
    patch_result = _patch_base_submission(
        base_json=Path(args.base_json),
        science_score_map=science_score_map,
        out_path=patched_out,
        method_name=str(args.method_name),
    )
    delta_summary = _compare_science_delta(
        base_json=Path(args.base_json),
        science_score_map=science_score_map,
    )

    eval_json = REPO_ROOT / str(args.eval_json)
    eval_payload = {
        "created_at_utc": _now_utc(),
        "method_name": str(args.method_name),
        "holdout_problem_summary": holdout_problem_summary,
        "reference_r2": reference_r2_eval,
        "task_specific": task_eval,
        "dense_final": dense_eval,
        "selected_rows": selected_rows,
    }
    _write_json(eval_json, eval_payload)

    manifest = {
        "created_at_utc": _now_utc(),
        "method_name": str(args.method_name),
        "protocol": {
            "holdout_split": float(args.holdout_split),
            "split_seed": int(args.split_seed),
            "n_splits": int(args.n_splits),
            "reflection_threshold": float(DEFAULT_REFLECTION_THRESHOLD),
            "source_anchor_pcts": [int(v) for v in source_anchor_pcts],
            "target_anchor_pcts": [int(v) for v in target_anchor_pcts],
            "family_names": list(SEARCH_FAMILY_MAP.keys()),
            "transfer_margin": float(args.transfer_margin),
            "tree_margin": float(args.tree_margin),
            "seed_values": [int(v) for v in seed_values],
            "max_problems_per_cache": None if max_problems_per_cache is None else int(max_problems_per_cache),
        },
        "cache_state": {
            "labeled": labeled_cache_state,
            "blind": {
                "status": str(blind_cache_status),
                "path": None if blind_cache_path is None else _display_path(blind_cache_path),
                "num_payloads": int(len(blind_feature_store)),
            },
        },
        "artifacts": {
            "patched_submission": _display_path(patched_out),
            "candidate_csv": _display_path(candidate_csv),
            "selected_csv": _display_path(selected_csv),
            "eval_json": _display_path(eval_json),
        },
        "holdout": {
            "reference_r2": reference_r2_eval,
            "task_specific": task_eval,
            "dense_final": dense_eval,
        },
        "selected_rows": selected_rows,
        "science_delta_vs_base": delta_summary,
        "patch_validation": patch_result["validation"],
    }
    manifest_json = REPO_ROOT / str(args.manifest_json)
    _write_json(manifest_json, manifest)

    doc_out = REPO_ROOT / str(args.doc_out)
    _render_doc(path=doc_out, manifest=manifest)

    print(
        "[holdout] dense_final auc_of_auroc={dense:.4f} vs task_specific={task:.4f} vs r2={r2:.4f}".format(
            dense=float(dense_eval["aggregate"]["auc_of_auroc"]),
            task=float(task_eval["aggregate"]["auc_of_auroc"]),
            r2=float(reference_r2_eval["aggregate"]["auc_of_auroc"]),
        ),
        flush=True,
    )
    print(f"[artifact] patched_submission={_display_path(patched_out)}", flush=True)
    print(f"[artifact] candidate_csv={_display_path(candidate_csv)}", flush=True)
    print(f"[artifact] selected_csv={_display_path(selected_csv)}", flush=True)
    print(f"[artifact] eval_json={_display_path(eval_json)}", flush=True)
    print(f"[artifact] manifest_json={_display_path(manifest_json)}", flush=True)
    print(f"[artifact] doc={_display_path(doc_out)}", flush=True)
    print(f"[validate] {patch_result['validation']}", flush=True)


if __name__ == "__main__":
    main()
