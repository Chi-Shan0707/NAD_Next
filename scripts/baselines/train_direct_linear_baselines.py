#!/usr/bin/env python3
"""Train and evaluate direct no-SVD linear baselines for the SVDomain protocol.

This script intentionally lives in `work/NAD_Next` and adds a clean baseline
track without mutating any paper-facing SVD routes. It reuses the canonical:

- `token_plus_traj_fixed` 22-feature family
- `raw`, `rank`, and `raw+rank` representations
- grouped `dataset + problem_id` 85/15 holdout
- four-anchor route protocol `10 / 40 / 70 / 100`

Required outputs
----------------
- `scripts/baselines/train_direct_linear_baselines.py`
- `results/tables/direct_linear_baselines.csv`
- `docs/DIRECT_LINEAR_BASELINES.md`

The benchmark is intentionally explicit about dependencies and protocol
contracts. It is not meant to be universally plug-and-play across arbitrary
cache layouts; instead it aims to be complete, readable, and auditable for the
paper-facing open-source release.
"""
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
from pathlib import Path
from typing import Any, Iterable, Optional

for env_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(env_name, "1")

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

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

from nad.explain.svd_explain import feature_family
from nad.ops.earlystop_svd import (
    _auroc,
    _build_representation,
    FULL_FEATURE_NAMES,
)
from scripts.run_earlystop_prefix10_svd_round1 import (
    ANCHOR_POSITIONS,
    DEFAULT_FEATURE_CHUNK_PROBLEMS,
    EARLY_STOP_POSITIONS,
    EXTRACTION_POSITIONS,
    FEATURE_TO_INDEX,
    OFFICIAL_SLOT_TO_ANCHOR,
    _display_path,
    _now_utc,
    _pct_label,
    evaluate_method_from_feature_store,
)
from scripts.run_structured_ood_suite import (
    AUTO_PREBUILT_FEATURE_STORE_GLOBS,
    MATH_BENCHMARKS,
    ROOT_ORDER,
    _ensure_training_payload,
    _filter_feature_store,
    _load_prebuilt_feature_store,
    _payload_model_family,
    _payload_root,
    _store_summary,
)
from SVDomain.train_es_svd_ms_rr_r1 import (
    FIXED_FAMILY_NAME,
    FIXED_FEATURE_INDICES,
    FIXED_FEATURE_NAMES,
    _build_domain_training_tables,
    _build_holdout_problem_map,
    _group_folds,
    _load_or_build_qualified_feature_store,
    _resolve_path,
    _split_feature_store,
    _summarise_feature_store,
)


HOLDOUT_DOMAINS = ("math", "science", "coding")
NONCODING_DOMAINS = ("math", "science")
OUTPUT_DOMAIN_ORDER = ("math", "science", "ms", "coding")
SEARCH_C_VALUES = (0.01, 0.10, 1.0, 10.0)
SEARCH_CLASS_WEIGHT = ("none", "balanced")
SEARCH_L1_RATIOS = (0.1, 0.5, 0.9)
DEFAULT_RANDOM_SEEDS = (42, 43, 44)
DEFAULT_OUT_CSV = REPO_ROOT / "results" / "tables" / "direct_linear_baselines.csv"
DEFAULT_OUT_DOC = REPO_ROOT / "docs" / "DIRECT_LINEAR_BASELINES.md"
DEFAULT_SVD_HOLDOUT = REPO_ROOT / "SVDomain" / "results" / "summary_metrics.json"
DEFAULT_SVD_OOD = REPO_ROOT / "SVDomain" / "results" / "tables" / "id_vs_ood_summary.csv"
METHOD_DISPLAY_ORDER = (
    "lr_raw",
    "lr_rank",
    "lr_raw_rank",
    "elasticnet_lr_raw_rank",
    "l1_lr_raw_rank",
    "linear_svm_raw_rank",
)
CURRENT_SVD_METHOD = {
    "math": "es_svd_math_rr_r1",
    "science": "es_svd_science_rr_r1",
    "ms": "es_svd_ms_rr_r1",
    "coding": "es_svd_coding_rr_r1",
}
CURRENT_SVD_HOLDOUT_KEY = {
    "math": "math",
    "science": "science",
    "ms": "combined_noncoding",
    "coding": "coding",
}


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    display_name: str
    representation: str
    estimator_kind: str
    penalty: str
    solver: str
    use_random_seeds: bool


MODEL_SPECS = (
    ModelSpec(
        model_name="lr_raw",
        display_name="Logistic Regression (raw)",
        representation="raw",
        estimator_kind="logreg",
        penalty="l2",
        solver="lbfgs",
        use_random_seeds=False,
    ),
    ModelSpec(
        model_name="lr_rank",
        display_name="Logistic Regression (rank)",
        representation="rank",
        estimator_kind="logreg",
        penalty="l2",
        solver="lbfgs",
        use_random_seeds=False,
    ),
    ModelSpec(
        model_name="lr_raw_rank",
        display_name="Logistic Regression (raw+rank)",
        representation="raw+rank",
        estimator_kind="logreg",
        penalty="l2",
        solver="lbfgs",
        use_random_seeds=False,
    ),
    ModelSpec(
        model_name="elasticnet_lr_raw_rank",
        display_name="Elastic-Net Logistic Regression (raw+rank)",
        representation="raw+rank",
        estimator_kind="logreg",
        penalty="elasticnet",
        solver="saga",
        use_random_seeds=True,
    ),
    ModelSpec(
        model_name="l1_lr_raw_rank",
        display_name="L1 Logistic Regression (raw+rank)",
        representation="raw+rank",
        estimator_kind="logreg",
        penalty="l1",
        solver="saga",
        use_random_seeds=True,
    ),
    ModelSpec(
        model_name="linear_svm_raw_rank",
        display_name="Linear SVM (raw+rank)",
        representation="raw+rank",
        estimator_kind="linear_svm",
        penalty="l2",
        solver="linear_svc",
        use_random_seeds=False,
    ),
)
MODEL_SPEC_BY_NAME = {spec.model_name: spec for spec in MODEL_SPECS}


def _parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(raw).split(",") if item.strip())


def _safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _fmt_float(value: Any, digits: int = 4) -> str:
    try:
        value_f = float(value)
    except Exception:
        return "N/A"
    if not math.isfinite(value_f):
        return "N/A"
    return f"{value_f:.{digits}f}"


def _fmt_pct(value: Any) -> str:
    try:
        value_f = float(value)
    except Exception:
        return "N/A"
    if not math.isfinite(value_f):
        return "N/A"
    return f"{100.0 * value_f:.2f}%"


def _max_workers_default() -> int:
    cpu = max(1, int(os.cpu_count() or 1))
    return max(1, min(8, cpu, 16))


def _feature_cache_dir(raw: str) -> Optional[Path]:
    text = str(raw).strip().lower()
    if text in {"", "none", "off"}:
        return None
    return (REPO_ROOT / str(raw)).resolve()


def _make_rep_feature_names(representation: str) -> list[str]:
    if representation == "raw":
        return [f"{name}::raw" for name in FIXED_FEATURE_NAMES]
    if representation == "rank":
        return [f"{name}::rank" for name in FIXED_FEATURE_NAMES]
    if representation == "raw+rank":
        return [f"{name}::raw" for name in FIXED_FEATURE_NAMES] + [f"{name}::rank" for name in FIXED_FEATURE_NAMES]
    raise ValueError(f"Unsupported representation: {representation}")


def _base_feature_name(rep_feature_name: str) -> str:
    return str(rep_feature_name).split("::")[0]


def _class_weight_value(name: str) -> Optional[str]:
    return None if str(name) == "none" else "balanced"


def _group_top1(scores: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    unique_groups = np.unique(groups)
    hits: list[float] = []
    for group in unique_groups:
        mask = groups == group
        if int(mask.sum()) <= 0:
            continue
        local_scores = scores[mask]
        local_labels = y[mask]
        best_idx = int(np.argmax(local_scores))
        hits.append(float(local_labels[best_idx]))
    if not hits:
        return float("nan")
    return float(np.mean(hits))


def _evaluate_scores(scores: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(_auroc(np.asarray(scores, dtype=np.float64), np.asarray(y, dtype=np.int32))),
        "group_top1": float(_group_top1(np.asarray(scores, dtype=np.float64), np.asarray(y, dtype=np.int32), np.asarray(groups, dtype=object))),
    }


def _recover_effective_weights(
    *,
    scaler: StandardScaler,
    estimator: Any,
    representation: str,
) -> tuple[np.ndarray, float, list[str]]:
    rep_feature_names = _make_rep_feature_names(representation)
    coef_scaled = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)
    intercept_scaled = float(np.asarray(estimator.intercept_, dtype=np.float64).reshape(-1)[0])
    scale = np.asarray(getattr(scaler, "scale_", np.ones_like(coef_scaled)), dtype=np.float64).reshape(-1)
    mean = np.asarray(getattr(scaler, "mean_", np.zeros_like(coef_scaled)), dtype=np.float64).reshape(-1)
    safe_scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    coef_orig = coef_scaled / safe_scale
    intercept_orig = intercept_scaled - float(np.dot(mean, coef_orig))
    return coef_orig.astype(np.float64, copy=False), float(intercept_orig), rep_feature_names


def _top_signed_features(rep_feature_names: list[str], weights: np.ndarray, positive: bool, k: int = 5) -> str:
    pairs = list(zip(rep_feature_names, weights.tolist()))
    if positive:
        ordered = sorted((pair for pair in pairs if float(pair[1]) > 0.0), key=lambda item: (-float(item[1]), item[0]))
    else:
        ordered = sorted((pair for pair in pairs if float(pair[1]) < 0.0), key=lambda item: (float(item[1]), item[0]))
    top = ordered[:k]
    if not top:
        return ""
    return "; ".join(f"{name}:{value:.4f}" for name, value in top)


def _collapse_base_feature_mass(rep_feature_names: list[str], weights: np.ndarray, k: int = 5) -> str:
    mass: dict[str, float] = {}
    for rep_name, weight in zip(rep_feature_names, weights.tolist()):
        base = _base_feature_name(rep_name)
        mass[base] = mass.get(base, 0.0) + abs(float(weight))
    ordered = sorted(mass.items(), key=lambda item: (-float(item[1]), item[0]))[:k]
    return "; ".join(f"{name}:{value:.4f}" for name, value in ordered)


def _collapse_family_mass(rep_feature_names: list[str], weights: np.ndarray, k: int = 5) -> str:
    mass: dict[str, float] = {}
    for rep_name, weight in zip(rep_feature_names, weights.tolist()):
        family = feature_family(_base_feature_name(rep_name))
        mass[family] = mass.get(family, 0.0) + abs(float(weight))
    ordered = sorted(mass.items(), key=lambda item: (-float(item[1]), item[0]))[:k]
    return "; ".join(f"{name}:{value:.4f}" for name, value in ordered)


def _family_mass_map(rep_feature_names: list[str], weights: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for rep_name, weight in zip(rep_feature_names, weights.tolist()):
        family = feature_family(_base_feature_name(rep_name))
        out[family] = out.get(family, 0.0) + abs(float(weight))
    return out


def _base_feature_mass_map(rep_feature_names: list[str], weights: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for rep_name, weight in zip(rep_feature_names, weights.tolist()):
        base = _base_feature_name(rep_name)
        out[base] = out.get(base, 0.0) + abs(float(weight))
    return out


def _active_dims(weights: np.ndarray, tol: float = 1e-8) -> int:
    return int(np.sum(np.abs(np.asarray(weights, dtype=np.float64)) > float(tol)))


def _seed_sequence(spec: ModelSpec, random_seeds: tuple[int, ...]) -> tuple[int, ...]:
    if spec.use_random_seeds:
        return tuple(int(seed) for seed in random_seeds)
    return (int(random_seeds[0]),)


def _iter_candidate_params(spec: ModelSpec) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c_value in SEARCH_C_VALUES:
        for class_weight_name in SEARCH_CLASS_WEIGHT:
            if spec.estimator_kind == "linear_svm":
                out.append(
                    {
                        "c_value": float(c_value),
                        "class_weight": str(class_weight_name),
                        "l1_ratio": None,
                    }
                )
                continue
            if spec.penalty == "elasticnet":
                for l1_ratio in SEARCH_L1_RATIOS:
                    out.append(
                        {
                            "c_value": float(c_value),
                            "class_weight": str(class_weight_name),
                            "l1_ratio": float(l1_ratio),
                        }
                    )
            else:
                out.append(
                    {
                        "c_value": float(c_value),
                        "class_weight": str(class_weight_name),
                        "l1_ratio": None,
                    }
                )
    return out


def _build_estimator(
    *,
    spec: ModelSpec,
    c_value: float,
    class_weight_name: str,
    seed: int,
    l1_ratio: Optional[float],
) -> Any:
    if spec.estimator_kind == "logreg":
        kwargs: dict[str, Any] = {
            "C": float(c_value),
            "penalty": str(spec.penalty),
            "class_weight": _class_weight_value(class_weight_name),
            "solver": str(spec.solver),
            "fit_intercept": True,
            "max_iter": 5000,
        }
        if spec.use_random_seeds:
            kwargs["random_state"] = int(seed)
        if spec.penalty == "elasticnet":
            kwargs["l1_ratio"] = float(l1_ratio) if l1_ratio is not None else 0.5
        elif spec.penalty == "l1":
            kwargs["l1_ratio"] = 1.0
        return LogisticRegression(**kwargs)

    if spec.estimator_kind == "linear_svm":
        return LinearSVC(
            C=float(c_value),
            class_weight=_class_weight_value(class_weight_name),
            dual=False,
            max_iter=5000,
            fit_intercept=True,
        )

    raise ValueError(f"Unsupported estimator kind: {spec.estimator_kind}")


def _fit_scaled_linear_model(
    *,
    spec: ModelSpec,
    x: np.ndarray,
    y: np.ndarray,
    c_value: float,
    class_weight_name: str,
    seed: int,
    l1_ratio: Optional[float],
) -> Optional[dict[str, Any]]:
    if x.shape[0] < 4 or x.shape[1] < 1:
        return None
    if np.unique(y).shape[0] < 2:
        return None

    scaler = StandardScaler(with_mean=True, with_std=True)
    x_scaled = scaler.fit_transform(x)
    estimator = _build_estimator(
        spec=spec,
        c_value=float(c_value),
        class_weight_name=str(class_weight_name),
        seed=int(seed),
        l1_ratio=l1_ratio,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        estimator.fit(x_scaled, y)
    return {
        "scaler": scaler,
        "estimator": estimator,
    }


def _predict_scaled_linear_model(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    scaler = model["scaler"]
    estimator = model["estimator"]
    x_scaled = scaler.transform(x)
    return np.asarray(estimator.decision_function(x_scaled), dtype=np.float64).reshape(-1)


def _cv_candidate_score(
    *,
    spec: ModelSpec,
    x_rep: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    random_seeds: tuple[int, ...],
    c_value: float,
    class_weight_name: str,
    l1_ratio: Optional[float],
) -> dict[str, Any]:
    values: list[float] = []
    for train_idx, test_idx in folds:
        y_train = y[train_idx]
        y_test = y[test_idx]
        if np.unique(y_train).shape[0] < 2 or np.unique(y_test).shape[0] < 2:
            continue

        x_train = x_rep[train_idx]
        x_test = x_rep[test_idx]
        scaler = StandardScaler(with_mean=True, with_std=True)
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        for seed in _seed_sequence(spec, random_seeds):
            estimator = _build_estimator(
                spec=spec,
                c_value=float(c_value),
                class_weight_name=str(class_weight_name),
                seed=int(seed),
                l1_ratio=l1_ratio,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                try:
                    estimator.fit(x_train_scaled, y_train)
                    scores = np.asarray(estimator.decision_function(x_test_scaled), dtype=np.float64).reshape(-1)
                except Exception:
                    continue
            fold_auc = _auroc(scores, y_test)
            if np.isfinite(fold_auc):
                values.append(float(fold_auc))

    return {
        "cv_mean_auroc": _safe_mean(values),
        "n_valid_scores": int(len(values)),
    }


def _fit_direct_route(
    *,
    spec: ModelSpec,
    x_raw: np.ndarray,
    x_rank: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    position: float,
    training_scope: str,
    random_seeds: tuple[int, ...],
    n_splits: int,
) -> dict[str, Any]:
    if x_raw.shape[0] == 0:
        raise ValueError(f"{training_scope}@{_pct_label(position)} has no rows")
    if np.unique(y).shape[0] < 2:
        raise ValueError(f"{training_scope}@{_pct_label(position)} lacks both classes")

    folds = _group_folds(groups, n_splits=int(n_splits))
    if not folds:
        raise ValueError(f"{training_scope}@{_pct_label(position)} has insufficient grouped folds")

    x_rep = _build_representation(
        x_raw=x_raw,
        x_rank=x_rank,
        feature_indices=FIXED_FEATURE_INDICES,
        representation=str(spec.representation),
    )

    best: dict[str, Any] = {
        "cv_mean_auroc": float("-inf"),
        "n_valid_scores": 0,
        "c_value": None,
        "class_weight": None,
        "l1_ratio": None,
    }
    for candidate in _iter_candidate_params(spec):
        score = _cv_candidate_score(
            spec=spec,
            x_rep=x_rep,
            y=y,
            folds=folds,
            random_seeds=random_seeds,
            c_value=float(candidate["c_value"]),
            class_weight_name=str(candidate["class_weight"]),
            l1_ratio=candidate["l1_ratio"],
        )
        cv_mean_auroc = float(score["cv_mean_auroc"])
        if not np.isfinite(cv_mean_auroc):
            continue
        tie_key = (
            cv_mean_auroc,
            -float(candidate["c_value"]),
            0 if str(candidate["class_weight"]) == "none" else -1,
            -1.0 if candidate["l1_ratio"] is None else -float(candidate["l1_ratio"]),
        )
        best_key = (
            float(best["cv_mean_auroc"]),
            -float(best["c_value"] or 1e9),
            0 if str(best["class_weight"]) == "none" else -1,
            -1.0 if best["l1_ratio"] is None else -float(best["l1_ratio"]),
        )
        if tie_key > best_key:
            best = {
                "cv_mean_auroc": float(cv_mean_auroc),
                "n_valid_scores": int(score["n_valid_scores"]),
                "c_value": float(candidate["c_value"]),
                "class_weight": str(candidate["class_weight"]),
                "l1_ratio": None if candidate["l1_ratio"] is None else float(candidate["l1_ratio"]),
            }

    if not np.isfinite(float(best["cv_mean_auroc"])):
        raise RuntimeError(f"{training_scope}@{_pct_label(position)} found no valid candidate for {spec.model_name}")

    fit_seed = int(_seed_sequence(spec, random_seeds)[0])
    model = _fit_scaled_linear_model(
        spec=spec,
        x=x_rep,
        y=y,
        c_value=float(best["c_value"]),
        class_weight_name=str(best["class_weight"]),
        seed=fit_seed,
        l1_ratio=best["l1_ratio"],
    )
    if model is None:
        raise RuntimeError(f"{training_scope}@{_pct_label(position)} full-fit failed for {spec.model_name}")

    weights, intercept, rep_feature_names = _recover_effective_weights(
        scaler=model["scaler"],
        estimator=model["estimator"],
        representation=str(spec.representation),
    )
    return {
        "route_type": "direct_linear",
        "model_name": str(spec.model_name),
        "display_name": str(spec.display_name),
        "representation": str(spec.representation),
        "estimator_kind": str(spec.estimator_kind),
        "penalty": str(spec.penalty),
        "solver": str(spec.solver),
        "cv_mean_auroc": float(best["cv_mean_auroc"]),
        "n_valid_scores": int(best["n_valid_scores"]),
        "c_value": float(best["c_value"]),
        "class_weight": str(best["class_weight"]),
        "l1_ratio": None if best["l1_ratio"] is None else float(best["l1_ratio"]),
        "fit_seed": int(fit_seed),
        "seed_count_used": int(len(_seed_sequence(spec, random_seeds))),
        "feature_names": list(FIXED_FEATURE_NAMES),
        "feature_indices": list(FIXED_FEATURE_INDICES),
        "rep_feature_names": list(rep_feature_names),
        "effective_weights": weights,
        "effective_intercept": float(intercept),
        "active_dims": int(_active_dims(weights)),
        "top_positive_features": _top_signed_features(rep_feature_names, weights, positive=True),
        "top_negative_features": _top_signed_features(rep_feature_names, weights, positive=False),
        "top_base_features": _collapse_base_feature_mass(rep_feature_names, weights),
        "top_families": _collapse_family_mass(rep_feature_names, weights),
        "family_mass": _family_mass_map(rep_feature_names, weights),
        "base_feature_mass": _base_feature_mass_map(rep_feature_names, weights),
        "training_position": float(position),
        "training_scope": str(training_scope),
        "n_train_samples": int(x_rep.shape[0]),
        "model": model,
    }


def _fit_route_job(payload: dict[str, Any]) -> dict[str, Any]:
    spec = MODEL_SPEC_BY_NAME[str(payload["model_name"])]
    return _fit_direct_route(
        spec=spec,
        x_raw=np.asarray(payload["x_raw"], dtype=np.float64),
        x_rank=np.asarray(payload["x_rank"], dtype=np.float64),
        y=np.asarray(payload["y"], dtype=np.int32),
        groups=np.asarray(payload["groups"], dtype=object),
        position=float(payload["position"]),
        training_scope=str(payload["training_scope"]),
        random_seeds=tuple(int(seed) for seed in payload["random_seeds"]),
        n_splits=int(payload["n_splits"]),
    )


def _train_routes_for_domain_store(
    *,
    domain_name: str,
    tables: list[dict[str, np.ndarray]],
    model_specs: tuple[ModelSpec, ...],
    random_seeds: tuple[int, ...],
    n_splits: int,
    max_workers: int,
) -> dict[str, dict[float, dict[str, Any]]]:
    routes_by_model: dict[str, dict[float, dict[str, Any]]] = {spec.model_name: {} for spec in model_specs}
    jobs: list[dict[str, Any]] = []
    for spec in model_specs:
        for pos_idx, position in enumerate(ANCHOR_POSITIONS):
            jobs.append(
                {
                    "model_name": str(spec.model_name),
                    "x_raw": tables[pos_idx]["x_raw"],
                    "x_rank": tables[pos_idx]["x_rank"],
                    "y": tables[pos_idx]["y"],
                    "groups": tables[pos_idx]["groups"],
                    "position": float(position),
                    "training_scope": str(domain_name),
                    "random_seeds": list(int(seed) for seed in random_seeds),
                    "n_splits": int(n_splits),
                }
            )

    worker_count = max(1, min(int(max_workers), int(len(jobs)) if jobs else 1))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_map = {executor.submit(_fit_route_job, job): (job["model_name"], float(job["position"])) for job in jobs}
        for future in as_completed(future_map):
            model_name, position = future_map[future]
            route = future.result()
            routes_by_model[str(model_name)][float(position)] = route
            print(
                "[train] domain={domain:<7s} model={model:<24s} anchor={anchor:>4s} "
                "cv={cv:.4f} rep={rep:<8s} C={c:.2f} cw={cw} l1={l1}".format(
                    domain=str(domain_name),
                    model=str(model_name),
                    anchor=_pct_label(position),
                    cv=float(route["cv_mean_auroc"]),
                    rep=str(route["representation"]),
                    c=float(route["c_value"]),
                    cw=str(route["class_weight"]),
                    l1="-" if route["l1_ratio"] is None else f"{float(route['l1_ratio']):.1f}",
                ),
                flush=True,
            )
    return routes_by_model


def _predict_direct_route(route: dict[str, Any], x_raw: np.ndarray) -> np.ndarray:
    representation = str(route["representation"])
    feature_indices = [int(idx) for idx in route["feature_indices"]]
    if representation == "raw":
        x_rank = np.zeros_like(x_raw, dtype=np.float64)
    else:
        from nad.ops.earlystop_svd import _rank_transform_matrix

        x_rank = _rank_transform_matrix(x_raw)
    x_rep = _build_representation(
        x_raw=x_raw,
        x_rank=x_rank,
        feature_indices=feature_indices,
        representation=representation,
    )
    return _predict_scaled_linear_model(route["model"], x_rep)


def _make_routes_score_fn(routes_by_domain: dict[str, dict[str, dict[float, dict[str, Any]]]], model_name: str):
    official_to_anchor = {
        int(pos_idx): float(OFFICIAL_SLOT_TO_ANCHOR[float(position)])
        for pos_idx, position in enumerate(EARLY_STOP_POSITIONS)
    }

    def _score(domain: str, position_index: int, x_raw: np.ndarray) -> np.ndarray:
        domain_routes = routes_by_domain.get(str(domain))
        if domain_routes is None:
            raise KeyError(f"Missing routes for domain={domain}")
        route = domain_routes[str(model_name)][official_to_anchor[int(position_index)]]
        return _predict_direct_route(route, np.asarray(x_raw, dtype=np.float64))

    return _score


def _anchor_route_rows(
    *,
    domain: str,
    model_name: str,
    routes: dict[float, dict[str, Any]],
    holdout_tables: list[dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pos_idx, position in enumerate(ANCHOR_POSITIONS):
        route = routes[float(position)]
        x_holdout = np.asarray(holdout_tables[pos_idx]["x_raw"], dtype=np.float64)
        y_holdout = np.asarray(holdout_tables[pos_idx]["y"], dtype=np.int32)
        groups_holdout = np.asarray(holdout_tables[pos_idx]["groups"], dtype=object)
        scores = _predict_direct_route(route, x_holdout) if x_holdout.shape[0] > 0 else np.asarray([], dtype=np.float64)
        metrics = _evaluate_scores(scores, y_holdout, groups_holdout) if x_holdout.shape[0] > 0 else {"auroc": float("nan"), "group_top1": float("nan")}
        rows.append(
            {
                "row_kind": "anchor_route",
                "domain": str(domain),
                "model_name": str(model_name),
                "display_name": str(route["display_name"]),
                "representation": str(route["representation"]),
                "anchor_pct": int(round(float(position) * 100.0)),
                "id_auc_of_auroc": float("nan"),
                "id_auc_of_selacc": float("nan"),
                "id_auroc_at_100": float("nan"),
                "id_stop_acc_at_100": float("nan"),
                "holdout_anchor_auroc": float(metrics["auroc"]),
                "holdout_anchor_group_top1": float(metrics["group_top1"]),
                "cv_mean_auroc": float(route["cv_mean_auroc"]),
                "n_valid_scores": int(route["n_valid_scores"]),
                "best_c": float(route["c_value"]),
                "best_class_weight": str(route["class_weight"]),
                "best_penalty": str(route["penalty"]),
                "best_l1_ratio": "" if route["l1_ratio"] is None else float(route["l1_ratio"]),
                "solver": str(route["solver"]),
                "fit_seed": int(route["fit_seed"]),
                "seed_count_used": int(route["seed_count_used"]),
                "n_train_samples": int(route.get("n_train_samples", 0)),
                "n_holdout_samples": int(x_holdout.shape[0]),
                "n_holdout_groups": int(np.unique(groups_holdout).shape[0]),
                "active_dims": int(route["active_dims"]),
                "top_positive_features": str(route["top_positive_features"]),
                "top_negative_features": str(route["top_negative_features"]),
                "top_base_features": str(route["top_base_features"]),
                "top_families": str(route["top_families"]),
                "sparse_zero_fraction": float(1.0 - (float(route["active_dims"]) / max(1.0, float(len(route["effective_weights"]))))),
                "ood_protocol": "",
                "ood_fold_count": "",
                "ood_macro_auc_of_auroc": float("nan"),
                "ood_macro_auc_of_selacc": float("nan"),
                "ood_macro_auroc_at_100": float("nan"),
                "ood_macro_stop_acc_at_100": float("nan"),
                "abs_gap_auc_of_auroc": float("nan"),
                "abs_gap_auroc_at_100": float("nan"),
                "svd_reference_method": "",
                "svd_reference_id_auc_of_auroc": float("nan"),
                "svd_reference_id_auroc_at_100": float("nan"),
                "delta_vs_svd_id_auc_of_auroc": float("nan"),
                "delta_vs_svd_id_auroc_at_100": float("nan"),
            }
        )
    return rows


def _aggregate_route_signature(routes: dict[float, dict[str, Any]], top_k: int = 5) -> dict[str, str]:
    family_mass: dict[str, float] = {}
    feature_mass: dict[str, float] = {}
    positive_counter: dict[str, int] = {}
    negative_counter: dict[str, int] = {}
    for route in routes.values():
        for name, value in route["family_mass"].items():
            family_mass[str(name)] = family_mass.get(str(name), 0.0) + float(value)
        for name, value in route["base_feature_mass"].items():
            feature_mass[str(name)] = feature_mass.get(str(name), 0.0) + float(value)
        for entry in str(route["top_positive_features"]).split(";"):
            token = entry.strip().split(":")[0]
            if token:
                positive_counter[token] = positive_counter.get(token, 0) + 1
        for entry in str(route["top_negative_features"]).split(";"):
            token = entry.strip().split(":")[0]
            if token:
                negative_counter[token] = negative_counter.get(token, 0) + 1

    top_families = "; ".join(
        f"{name}:{value:.4f}"
        for name, value in sorted(family_mass.items(), key=lambda item: (-float(item[1]), item[0]))[:top_k]
    )
    top_features = "; ".join(
        f"{name}:{value:.4f}"
        for name, value in sorted(feature_mass.items(), key=lambda item: (-float(item[1]), item[0]))[:top_k]
    )
    top_positive = "; ".join(
        f"{name}:{count}"
        for name, count in sorted(positive_counter.items(), key=lambda item: (-int(item[1]), item[0]))[:top_k]
    )
    top_negative = "; ".join(
        f"{name}:{count}"
        for name, count in sorted(negative_counter.items(), key=lambda item: (-int(item[1]), item[0]))[:top_k]
    )
    return {
        "top_families": top_families,
        "top_base_features": top_features,
        "top_positive_recurrent": top_positive,
        "top_negative_recurrent": top_negative,
    }


def _summarise_route_sample_counts(routes: dict[float, dict[str, Any]]) -> str:
    parts: list[str] = []
    for position, route in sorted(routes.items()):
        count = route.get("n_train_samples", "")
        if count in {"", None}:
            continue
        parts.append(f"{_pct_label(position)}:{int(count)}")
    return "; ".join(parts)


def _load_svd_holdout_reference(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for domain_name, key in CURRENT_SVD_HOLDOUT_KEY.items():
        rows = list(payload.get("holdout", {}).get(key, []))
        method_id = CURRENT_SVD_METHOD[domain_name]
        matched = next((row for row in rows if str(row.get("method")) == method_id), None)
        if matched is None:
            out[domain_name] = {}
        else:
            out[domain_name] = dict(matched)
    return out


def _load_svd_ood_reference(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        domain = str(row["domain"])
        protocol = str(row["ood_protocol"])
        if domain not in CURRENT_SVD_METHOD:
            continue
        if str(row["method"]) != CURRENT_SVD_METHOD[domain]:
            continue
        out[(domain, protocol)] = dict(row)
    return out


def _domain_summary_row(
    *,
    domain: str,
    spec: ModelSpec,
    routes: dict[float, dict[str, Any]],
    eval_payload: dict[str, Any],
    svd_reference: dict[str, Any],
) -> dict[str, Any]:
    aggregate = dict(eval_payload["aggregate"])
    signature = _aggregate_route_signature(routes)
    id_auc = float(aggregate.get("auc_of_auroc", float("nan")))
    id_auc_100 = float(aggregate.get("auroc@100%", float("nan")))
    svd_auc = float(svd_reference.get("auc_of_auroc", float("nan"))) if svd_reference else float("nan")
    svd_auc_100 = float(svd_reference.get("auroc_at_100", float("nan"))) if svd_reference else float("nan")
    active_dims_mean = _safe_mean(float(route["active_dims"]) for route in routes.values())
    sparse_zero_mean = _safe_mean(
        1.0 - (float(route["active_dims"]) / max(1.0, float(len(route["effective_weights"]))))
        for route in routes.values()
    )
    return {
        "row_kind": "id_summary",
        "domain": str(domain),
        "model_name": str(spec.model_name),
        "display_name": str(spec.display_name),
        "representation": str(spec.representation),
        "anchor_pct": "",
        "id_auc_of_auroc": float(id_auc),
        "id_auc_of_selacc": float(aggregate.get("auc_of_selacc", float("nan"))),
        "id_auroc_at_100": float(id_auc_100),
        "id_stop_acc_at_100": float(aggregate.get("stop_acc@100%", float("nan"))),
        "holdout_anchor_auroc": float("nan"),
        "holdout_anchor_group_top1": float("nan"),
        "cv_mean_auroc": _safe_mean(float(route["cv_mean_auroc"]) for route in routes.values()),
        "n_valid_scores": int(sum(int(route["n_valid_scores"]) for route in routes.values())),
        "best_c": "; ".join(f"{_pct_label(position)}:{float(route['c_value']):.2f}" for position, route in sorted(routes.items())),
        "best_class_weight": "; ".join(f"{_pct_label(position)}:{route['class_weight']}" for position, route in sorted(routes.items())),
        "best_penalty": str(spec.penalty),
        "best_l1_ratio": "; ".join(
            f"{_pct_label(position)}:{float(route['l1_ratio']):.1f}" for position, route in sorted(routes.items()) if route["l1_ratio"] is not None
        ),
        "solver": str(spec.solver),
        "fit_seed": "; ".join(f"{_pct_label(position)}:{int(route['fit_seed'])}" for position, route in sorted(routes.items())),
        "seed_count_used": int(max(int(route["seed_count_used"]) for route in routes.values())),
        "n_train_samples": _summarise_route_sample_counts(routes),
        "n_holdout_samples": int(sum(int(cache.get("n_samples", 0)) for cache in eval_payload.get("by_cache", []))),
        "n_holdout_groups": int(sum(int(cache.get("n_problems", 0)) for cache in eval_payload.get("by_cache", []))),
        "active_dims": int(round(active_dims_mean)) if np.isfinite(active_dims_mean) else "",
        "top_positive_features": str(signature["top_positive_recurrent"]),
        "top_negative_features": str(signature["top_negative_recurrent"]),
        "top_base_features": str(signature["top_base_features"]),
        "top_families": str(signature["top_families"]),
        "sparse_zero_fraction": float(sparse_zero_mean),
        "ood_protocol": "",
        "ood_fold_count": "",
        "ood_macro_auc_of_auroc": float("nan"),
        "ood_macro_auc_of_selacc": float("nan"),
        "ood_macro_auroc_at_100": float("nan"),
        "ood_macro_stop_acc_at_100": float("nan"),
        "abs_gap_auc_of_auroc": float("nan"),
        "abs_gap_auroc_at_100": float("nan"),
        "svd_reference_method": "" if not svd_reference else str(svd_reference.get("method", CURRENT_SVD_METHOD.get(domain, ""))),
        "svd_reference_id_auc_of_auroc": float(svd_auc),
        "svd_reference_id_auroc_at_100": float(svd_auc_100),
        "delta_vs_svd_id_auc_of_auroc": float(id_auc - svd_auc) if np.isfinite(id_auc) and np.isfinite(svd_auc) else float("nan"),
        "delta_vs_svd_id_auroc_at_100": float(id_auc_100 - svd_auc_100) if np.isfinite(id_auc_100) and np.isfinite(svd_auc_100) else float("nan"),
    }


def _ood_summary_row(
    *,
    domain: str,
    spec: ModelSpec,
    id_row: dict[str, Any],
    protocol: str,
    ood_rows: list[dict[str, Any]],
    svd_reference: Optional[dict[str, Any]],
) -> dict[str, Any]:
    ood_macro_auc = _safe_mean(float(row["auc_of_auroc"]) for row in ood_rows)
    ood_macro_sel = _safe_mean(float(row["auc_of_selacc"]) for row in ood_rows)
    ood_macro_auc100 = _safe_mean(float(row["auroc_at_100"]) for row in ood_rows)
    ood_macro_stop100 = _safe_mean(float(row["stop_acc_at_100"]) for row in ood_rows)
    svd_ood_auc = float("nan")
    svd_ood_auc100 = float("nan")
    if svd_reference is not None:
        try:
            svd_ood_auc = float(svd_reference.get("ood_macro_auc_of_auroc", float("nan")))
            svd_ood_auc100 = float(svd_reference.get("ood_macro_auroc_at_100", float("nan")))
        except Exception:
            pass
    return {
        "row_kind": "ood_summary",
        "domain": str(domain),
        "model_name": str(spec.model_name),
        "display_name": str(spec.display_name),
        "representation": str(spec.representation),
        "anchor_pct": "",
        "id_auc_of_auroc": float(id_row["id_auc_of_auroc"]),
        "id_auc_of_selacc": float(id_row["id_auc_of_selacc"]),
        "id_auroc_at_100": float(id_row["id_auroc_at_100"]),
        "id_stop_acc_at_100": float(id_row["id_stop_acc_at_100"]),
        "holdout_anchor_auroc": float("nan"),
        "holdout_anchor_group_top1": float("nan"),
        "cv_mean_auroc": float("nan"),
        "n_valid_scores": "",
        "best_c": "",
        "best_class_weight": "",
        "best_penalty": str(spec.penalty),
        "best_l1_ratio": "",
        "solver": str(spec.solver),
        "fit_seed": "",
        "seed_count_used": "",
        "n_train_samples": "",
        "n_holdout_samples": "",
        "n_holdout_groups": "",
        "active_dims": "",
        "top_positive_features": "",
        "top_negative_features": "",
        "top_base_features": "",
        "top_families": "",
        "sparse_zero_fraction": float("nan"),
        "ood_protocol": str(protocol),
        "ood_fold_count": int(len(ood_rows)),
        "ood_macro_auc_of_auroc": float(ood_macro_auc),
        "ood_macro_auc_of_selacc": float(ood_macro_sel),
        "ood_macro_auroc_at_100": float(ood_macro_auc100),
        "ood_macro_stop_acc_at_100": float(ood_macro_stop100),
        "abs_gap_auc_of_auroc": float(ood_macro_auc - float(id_row["id_auc_of_auroc"])) if np.isfinite(ood_macro_auc) and np.isfinite(float(id_row["id_auc_of_auroc"])) else float("nan"),
        "abs_gap_auroc_at_100": float(ood_macro_auc100 - float(id_row["id_auroc_at_100"])) if np.isfinite(ood_macro_auc100) and np.isfinite(float(id_row["id_auroc_at_100"])) else float("nan"),
        "svd_reference_method": CURRENT_SVD_METHOD.get(domain, ""),
        "svd_reference_id_auc_of_auroc": float(svd_ood_auc),
        "svd_reference_id_auroc_at_100": float(svd_ood_auc100),
        "delta_vs_svd_id_auc_of_auroc": float(ood_macro_auc - svd_ood_auc) if np.isfinite(ood_macro_auc) and np.isfinite(svd_ood_auc) else float("nan"),
        "delta_vs_svd_id_auroc_at_100": float(ood_macro_auc100 - svd_ood_auc100) if np.isfinite(ood_macro_auc100) and np.isfinite(svd_ood_auc100) else float("nan"),
    }


def _load_root_store(
    *,
    source_name: str,
    cache_root: str,
    required_feature_names: set[str],
    feature_cache_dir: Optional[Path],
    refresh_feature_cache: bool,
    feature_workers: int,
    feature_chunk_problems: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not refresh_feature_cache:
        prebuilt_store, prebuilt_paths = _load_prebuilt_feature_store(str(source_name))
        if prebuilt_store:
            return (
                [_ensure_training_payload(payload, source_name=str(source_name)) for payload in prebuilt_store],
                {
                    "status": "loaded_prebuilt",
                    "path": ",".join(prebuilt_paths),
                    "cache_root": str(cache_root),
                    "patterns": list(AUTO_PREBUILT_FEATURE_STORE_GLOBS.get(str(source_name), [])),
                },
            )

    store, cache_path, cache_status = _load_or_build_qualified_feature_store(
        source_name=str(source_name),
        cache_root=str(cache_root),
        positions=tuple(float(v) for v in EXTRACTION_POSITIONS),
        required_feature_names=required_feature_names,
        max_problems_per_cache=None,
        feature_workers=int(feature_workers),
        chunk_problems=int(feature_chunk_problems),
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(refresh_feature_cache),
    )
    return (
        list(store),
        {
            "status": str(cache_status),
            "path": None if cache_path is None else _display_path(cache_path),
            "cache_root": str(cache_root),
        },
    )


def _build_structured_ood_splits(all_feature_store: list[dict[str, Any]]) -> dict[str, dict[str, list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]]]]:
    splits: dict[str, dict[str, list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]]]] = {
        "math": {"math_benchmark_withheld": []},
        "science": {"cache_root_withheld": [], "model_family_withheld": []},
        "coding": {"cache_root_withheld": [], "model_family_withheld": []},
    }

    domain_stores_all = {
        "math": _filter_feature_store(all_feature_store, domain="math"),
        "science": _filter_feature_store(all_feature_store, domain="science"),
        "coding": _filter_feature_store(all_feature_store, domain="coding"),
    }

    math_store = domain_stores_all["math"]
    for benchmark in MATH_BENCHMARKS:
        train_store = _filter_feature_store(math_store, include_datasets=set(MATH_BENCHMARKS) - {benchmark})
        test_store = _filter_feature_store(math_store, include_datasets={benchmark})
        if train_store and test_store:
            splits["math"]["math_benchmark_withheld"].append((f"withheld_{benchmark}", train_store, test_store))

    for domain in ("science", "coding"):
        domain_store = domain_stores_all[domain]
        roots = sorted({_payload_root(payload) for payload in domain_store}, key=lambda item: ROOT_ORDER.index(item) if item in ROOT_ORDER else 999)
        for root_name in roots:
            train_store = _filter_feature_store(domain_store, exclude_roots={root_name})
            test_store = _filter_feature_store(domain_store, include_roots={root_name})
            if train_store and test_store:
                splits[domain]["cache_root_withheld"].append((f"withheld_{root_name}", train_store, test_store))

        model_families = sorted({_payload_model_family(payload) for payload in domain_store})
        for model_family in model_families:
            train_store = _filter_feature_store(domain_store, exclude_model_families={model_family})
            test_store = _filter_feature_store(domain_store, include_model_families={model_family})
            if train_store and test_store:
                splits[domain]["model_family_withheld"].append((f"withheld_{model_family}", train_store, test_store))

    return splits


def _eval_rows_from_feature_store(
    *,
    domain: str,
    model_specs: tuple[ModelSpec, ...],
    routes_by_model: dict[str, dict[float, dict[str, Any]]],
    test_store: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    routes_by_domain = {
        str(domain): {
            str(spec.model_name): routes_by_model[str(spec.model_name)]
            for spec in model_specs
        }
    }
    out: dict[str, dict[str, Any]] = {}
    for spec in model_specs:
        score_fn = _make_routes_score_fn(routes_by_domain, str(spec.model_name))
        out[str(spec.model_name)] = evaluate_method_from_feature_store(
            method_name=str(spec.model_name),
            feature_store=test_store,
            position_values=tuple(float(v) for v in EARLY_STOP_POSITIONS),
            score_fn=score_fn,
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _best_id_summary(rows: list[dict[str, Any]], domain: str) -> Optional[dict[str, Any]]:
    candidates = [row for row in rows if str(row["row_kind"]) == "id_summary" and str(row["domain"]) == str(domain)]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            float(row["id_auc_of_auroc"]) if np.isfinite(float(row["id_auc_of_auroc"])) else float("-inf"),
            float(row["id_auroc_at_100"]) if np.isfinite(float(row["id_auroc_at_100"])) else float("-inf"),
            -METHOD_DISPLAY_ORDER.index(str(row["model_name"])) if str(row["model_name"]) in METHOD_DISPLAY_ORDER else -999,
        ),
    )


def _select_row(rows: list[dict[str, Any]], *, domain: str, model_name: str, row_kind: str, protocol: str = "") -> Optional[dict[str, Any]]:
    for row in rows:
        if str(row["row_kind"]) != str(row_kind):
            continue
        if str(row["domain"]) != str(domain):
            continue
        if str(row["model_name"]) != str(model_name):
            continue
        if protocol and str(row["ood_protocol"]) != str(protocol):
            continue
        return row
    return None


def _question_1_answer(rows: list[dict[str, Any]]) -> str:
    lines = []
    for domain in OUTPUT_DOMAIN_ORDER:
        best_row = _best_id_summary(rows, domain)
        if best_row is None:
            continue
        delta = float(best_row["delta_vs_svd_id_auc_of_auroc"])
        relation = "beats" if np.isfinite(delta) and delta > 0.0 else "does not beat"
        lines.append(
            f"- `{domain}`: best direct baseline is `{best_row['model_name']}` with "
            f"`AUC-of-AUROC={_fmt_float(best_row['id_auc_of_auroc'])}`; it {relation} "
            f"the current SVD route (`Δ={_fmt_float(delta)}`)."
        )
    if not lines:
        return "- No valid direct-vs-SVD comparison was produced."
    return "\n".join(lines)


def _question_2_answer(rows: list[dict[str, Any]]) -> str:
    lines = []
    for domain in OUTPUT_DOMAIN_ORDER:
        raw_row = _select_row(rows, domain=domain, model_name="lr_raw", row_kind="id_summary")
        rank_row = _select_row(rows, domain=domain, model_name="lr_rank", row_kind="id_summary")
        rr_row = _select_row(rows, domain=domain, model_name="lr_raw_rank", row_kind="id_summary")
        if raw_row is None or rank_row is None or rr_row is None:
            continue
        best_name = max(
            [raw_row, rank_row, rr_row],
            key=lambda row: (
                float(row["id_auc_of_auroc"]) if np.isfinite(float(row["id_auc_of_auroc"])) else float("-inf"),
                float(row["id_auroc_at_100"]) if np.isfinite(float(row["id_auroc_at_100"])) else float("-inf"),
            ),
        )["model_name"]
        lines.append(
            f"- `{domain}`: `lr_raw_rank={_fmt_float(rr_row['id_auc_of_auroc'])}`, "
            f"`lr_raw={_fmt_float(raw_row['id_auc_of_auroc'])}`, "
            f"`lr_rank={_fmt_float(rank_row['id_auc_of_auroc'])}`; best among the three is `{best_name}`."
        )
    if not lines:
        return "- Raw-vs-rank representation comparison is unavailable."
    return "\n".join(lines)


def _question_3_answer(rows: list[dict[str, Any]]) -> str:
    lines = []
    for domain in ("math", "science", "coding"):
        id_row = _select_row(rows, domain=domain, model_name="elasticnet_lr_raw_rank", row_kind="id_summary")
        if id_row is None:
            continue
        protocols = sorted({str(row["ood_protocol"]) for row in rows if str(row["row_kind"]) == "ood_summary" and str(row["domain"]) == domain})
        for protocol in protocols:
            enet_row = _select_row(rows, domain=domain, model_name="elasticnet_lr_raw_rank", row_kind="ood_summary", protocol=protocol)
            l2_row = _select_row(rows, domain=domain, model_name="lr_raw_rank", row_kind="ood_summary", protocol=protocol)
            if enet_row is None or l2_row is None:
                continue
            delta = float(enet_row["ood_macro_auc_of_auroc"]) - float(l2_row["ood_macro_auc_of_auroc"])
            lines.append(
                f"- `{domain}` / `{protocol}`: elastic-net OOD macro AUROC is "
                f"`{_fmt_float(enet_row['ood_macro_auc_of_auroc'])}` vs "
                f"`{_fmt_float(l2_row['ood_macro_auc_of_auroc'])}` for plain raw+rank LR "
                f"(`Δ={_fmt_float(delta)}`)."
            )
    if not lines:
        return "- Structured OOD rows are unavailable, so elastic-net robustness cannot be assessed."
    return "\n".join(lines)


def _question_4_answer(rows: list[dict[str, Any]]) -> str:
    lines = []
    for domain in ("math", "science", "coding"):
        best_row = _best_id_summary(rows, domain)
        if best_row is None:
            continue
        lines.append(
            f"- `{domain}` best direct model: `{best_row['model_name']}`; "
            f"dominant families: `{best_row['top_families'] or 'N/A'}`; "
            f"dominant base features: `{best_row['top_base_features'] or 'N/A'}`."
        )
    if not lines:
        return "- Coefficient dominance analysis is unavailable."
    return "\n".join(lines)


def _write_report(
    *,
    path: Path,
    rows: list[dict[str, Any]],
    protocol_summary: dict[str, Any],
    svd_holdout_reference: dict[str, dict[str, Any]],
) -> None:
    id_rows = [row for row in rows if str(row["row_kind"]) == "id_summary"]
    ood_rows = [row for row in rows if str(row["row_kind"]) == "ood_summary"]

    main_table_rows = []
    for domain in OUTPUT_DOMAIN_ORDER:
        best_row = _best_id_summary(rows, domain)
        if best_row is None:
            continue
        svd_row = svd_holdout_reference.get(domain, {})
        main_table_rows.append(
            [
                str(domain),
                str(best_row["model_name"]),
                _fmt_float(best_row["id_auc_of_auroc"]),
                _fmt_float(best_row["svd_reference_id_auc_of_auroc"]),
                _fmt_float(best_row["delta_vs_svd_id_auc_of_auroc"]),
                _fmt_float(best_row["id_auroc_at_100"]),
                _fmt_float(best_row["svd_reference_id_auroc_at_100"]),
                _fmt_float(best_row["delta_vs_svd_id_auroc_at_100"]),
            ]
        )

    repr_table_rows = []
    for domain in OUTPUT_DOMAIN_ORDER:
        raw_row = _select_row(rows, domain=domain, model_name="lr_raw", row_kind="id_summary")
        rank_row = _select_row(rows, domain=domain, model_name="lr_rank", row_kind="id_summary")
        rr_row = _select_row(rows, domain=domain, model_name="lr_raw_rank", row_kind="id_summary")
        if raw_row is None or rank_row is None or rr_row is None:
            continue
        repr_table_rows.append(
            [
                str(domain),
                _fmt_float(raw_row["id_auc_of_auroc"]),
                _fmt_float(rank_row["id_auc_of_auroc"]),
                _fmt_float(rr_row["id_auc_of_auroc"]),
                _fmt_float(float(rr_row["id_auc_of_auroc"]) - max(float(raw_row["id_auc_of_auroc"]), float(rank_row["id_auc_of_auroc"]))),
            ]
        )

    ood_table_rows = []
    for row in sorted(ood_rows, key=lambda item: (item["domain"], item["ood_protocol"], METHOD_DISPLAY_ORDER.index(item["model_name"]) if item["model_name"] in METHOD_DISPLAY_ORDER else 999)):
        ood_table_rows.append(
            [
                str(row["domain"]),
                str(row["ood_protocol"]),
                str(row["model_name"]),
                _fmt_float(row["id_auc_of_auroc"]),
                _fmt_float(row["ood_macro_auc_of_auroc"]),
                _fmt_float(row["abs_gap_auc_of_auroc"]),
            ]
        )

    coeff_rows = []
    for domain in ("math", "science", "coding"):
        best_row = _best_id_summary(rows, domain)
        if best_row is None:
            continue
        coeff_rows.append(
            [
                str(domain),
                str(best_row["model_name"]),
                str(best_row["top_families"] or "N/A"),
                str(best_row["top_base_features"] or "N/A"),
            ]
        )

    lines = [
        "# Direct Linear Baselines Without SVD",
        "",
        "## Summary",
        "",
        "This report benchmarks the strongest direct linear baselines on the canonical "
        "`token_plus_traj_fixed` 22-feature route **without SVD**. The benchmark keeps the "
        "same grouped `85/15` holdout, the same four-anchor training protocol, and the same "
        "raw/rank feature construction used by the paper-facing SVDomain route.",
        "",
        "The direct family includes:",
        "",
        "- `lr_raw`",
        "- `lr_rank`",
        "- `lr_raw_rank`",
        "- `elasticnet_lr_raw_rank`",
        "- `l1_lr_raw_rank`",
        "- `linear_svm_raw_rank`",
        "",
        "All comparisons below treat structured OOD as a **blind-proxy robustness** slice: it is not the blind leaderboard itself, but it is the closest labeled robustness proxy available offline.",
        "",
        "## Protocol",
        "",
        f"- `feature family`: `{protocol_summary['feature_family']}`",
        f"- `feature count`: `{protocol_summary['feature_count']}`",
        f"- `anchors`: `{', '.join(str(v) for v in protocol_summary['anchor_positions_pct'])}`",
        f"- `holdout split`: `{protocol_summary['holdout_split_label']}` by `dataset + problem_id`",
        f"- `CV folds`: `{protocol_summary['n_splits']}` grouped folds",
        f"- `C grid`: `{', '.join(f'{value:g}' for value in SEARCH_C_VALUES)}`",
        f"- `class_weight`: `{', '.join(SEARCH_CLASS_WEIGHT)}`",
        f"- `elastic-net l1_ratio`: `{', '.join(f'{value:.1f}' for value in SEARCH_L1_RATIOS)}`",
        f"- `random seeds where applicable`: `{', '.join(str(seed) for seed in protocol_summary['random_seeds'])}`",
        "",
        "## Best Direct vs Current SVD Route",
        "",
        "| Domain | Best Direct | Direct AUC-AUROC | SVD AUC-AUROC | Δ | Direct AUROC@100 | SVD AUROC@100 | Δ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in main_table_rows:
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "## Representation Check",
            "",
            "| Domain | LR raw | LR rank | LR raw+rank | raw+rank vs best(single) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in repr_table_rows:
        lines.append("| " + " | ".join(row) + " |")

    if ood_table_rows:
        lines.extend(
            [
                "",
                "## Structured OOD / Blind-Proxy Robustness",
                "",
                "| Domain | OOD Protocol | Model | ID AUC-AUROC | OOD Macro | Gap |",
                "| --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in ood_table_rows:
            lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "## Coefficient Dominance",
            "",
            "| Domain | Best Direct | Top Families | Top Base Features |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in coeff_rows:
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "## Explicit Answers",
            "",
            "### 1. Does no-SVD direct LR match or beat the current SVD route?",
            "",
            _question_1_answer(rows),
            "",
            "### 2. Is raw+rank better than raw-only or rank-only?",
            "",
            _question_2_answer(rows),
            "",
            "### 3. Does elastic-net improve OOD or blind-proxy robustness?",
            "",
            _question_3_answer(rows),
            "",
            "### 4. Which coefficients / feature families dominate in the best direct model?",
            "",
            _question_4_answer(rows),
            "",
            "## Interpretation",
            "",
            "- `raw+rank` should be interpreted as a direct linear head on the same feature bank used by the SVD route, but without the low-rank bottleneck.",
            "- The comparison is therefore about **whether the low-rank bottleneck is useful**, not about whether the feature bank itself carries signal.",
            "- Structured OOD serves as an offline robustness proxy. When an OOD slice degenerates into a single class, AUROC becomes `N/A`, so `AUROC@100` and grouped selection metrics matter more than the macro AUROC alone.",
            "",
            "## Artifacts",
            "",
            f"- `table`: `{_display_path(DEFAULT_OUT_CSV)}`",
            f"- `report`: `{_display_path(path)}`",
            f"- `SVD holdout reference`: `{_display_path(DEFAULT_SVD_HOLDOUT)}`",
            f"- `SVD OOD reference`: `{_display_path(DEFAULT_SVD_OOD)}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train direct no-SVD linear baselines on the SVDomain protocol")
    ap.add_argument("--main-cache-root", default="MUI_HUB/cache")
    ap.add_argument("--extra-cache-root", default="MUI_HUB/cache_train")
    ap.add_argument("--test-cache-root", default="MUI_HUB/cache_test")
    ap.add_argument("--domains", default="math,science,coding")
    ap.add_argument("--models", default=",".join(spec.model_name for spec in MODEL_SPECS))
    ap.add_argument("--holdout-split", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--random-seeds", default="42,43,44")
    ap.add_argument("--feature-workers", type=int, default=_max_workers_default())
    ap.add_argument("--feature-chunk-problems", type=int, default=DEFAULT_FEATURE_CHUNK_PROBLEMS)
    ap.add_argument("--max-workers", type=int, default=_max_workers_default())
    ap.add_argument("--feature-cache-dir", default="results/cache/direct_linear_baselines")
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--disable-structured-ood", action="store_true")
    ap.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV.relative_to(REPO_ROOT)))
    ap.add_argument("--out-doc", default=str(DEFAULT_OUT_DOC.relative_to(REPO_ROOT)))
    ap.add_argument("--svd-holdout-reference", default=str(DEFAULT_SVD_HOLDOUT.relative_to(REPO_ROOT)))
    ap.add_argument("--svd-ood-reference", default=str(DEFAULT_SVD_OOD.relative_to(REPO_ROOT)))
    args = ap.parse_args()

    selected_domains = tuple(domain for domain in _parse_csv(args.domains) if domain in HOLDOUT_DOMAINS)
    if not selected_domains:
        raise ValueError("Need at least one domain from: math,science,coding")

    selected_model_specs = tuple(
        MODEL_SPEC_BY_NAME[name]
        for name in _parse_csv(args.models)
        if name in MODEL_SPEC_BY_NAME
    )
    if not selected_model_specs:
        raise ValueError("Need at least one valid model from the required direct baseline family")

    random_seeds = tuple(int(token) for token in _parse_csv(args.random_seeds))
    if not random_seeds:
        raise ValueError("Need at least one random seed")

    main_cache_root = _resolve_path(str(args.main_cache_root))
    extra_cache_root = _resolve_path(str(args.extra_cache_root))
    test_cache_root = _resolve_path(str(args.test_cache_root))
    feature_cache_dir = _feature_cache_dir(str(args.feature_cache_dir))
    out_csv = (REPO_ROOT / str(args.out_csv)).resolve()
    out_doc = (REPO_ROOT / str(args.out_doc)).resolve()
    svd_holdout_reference_path = (REPO_ROOT / str(args.svd_holdout_reference)).resolve()
    svd_ood_reference_path = (REPO_ROOT / str(args.svd_ood_reference)).resolve()

    required_feature_names = set(str(name) for name in FIXED_FEATURE_NAMES)
    main_store, main_store_meta = _load_root_store(
        source_name="cache",
        cache_root=main_cache_root,
        required_feature_names=required_feature_names,
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_feature_cache),
        feature_workers=int(args.feature_workers),
        feature_chunk_problems=int(args.feature_chunk_problems),
    )
    extra_store, extra_store_meta = _load_root_store(
        source_name="cache_train",
        cache_root=extra_cache_root,
        required_feature_names=required_feature_names,
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_feature_cache),
        feature_workers=int(args.feature_workers),
        feature_chunk_problems=int(args.feature_chunk_problems),
    )
    test_store = []
    test_store_meta: dict[str, Any] = {"status": "disabled", "path": None, "cache_root": str(test_cache_root)}
    if not bool(args.disable_structured_ood):
        test_store, test_store_meta = _load_root_store(
            source_name="cache_test",
            cache_root=test_cache_root,
            required_feature_names=required_feature_names,
            feature_cache_dir=feature_cache_dir,
            refresh_feature_cache=bool(args.refresh_feature_cache),
            feature_workers=int(args.feature_workers),
            feature_chunk_problems=int(args.feature_chunk_problems),
        )

    full_id_store = list(main_store) + list(extra_store)
    domain_stores = {
        domain_name: [payload for payload in full_id_store if str(payload["domain"]) == domain_name]
        for domain_name in HOLDOUT_DOMAINS
    }

    split_packs: dict[str, dict[str, Any]] = {}
    train_tables_by_domain: dict[str, list[dict[str, np.ndarray]]] = {}
    holdout_tables_by_domain: dict[str, list[dict[str, np.ndarray]]] = {}
    routes_by_domain_model: dict[str, dict[str, dict[float, dict[str, Any]]]] = {}
    id_eval_by_domain_model: dict[str, dict[str, dict[str, Any]]] = {}
    csv_rows: list[dict[str, Any]] = []

    for domain_name in selected_domains:
        feature_store = domain_stores.get(domain_name, [])
        if not feature_store:
            print(f"[skip] domain={domain_name} has no feature_store payloads", flush=True)
            continue
        holdout_problem_map, holdout_problem_summary = _build_holdout_problem_map(
            feature_store,
            holdout_split=float(args.holdout_split),
            split_seed=int(args.split_seed),
        )
        train_store, holdout_store, full_store = _split_feature_store(
            feature_store,
            holdout_problem_map=holdout_problem_map,
        )
        if not train_store or not holdout_store:
            print(f"[skip] domain={domain_name} has empty train or holdout split", flush=True)
            continue
        split_packs[domain_name] = {
            "train_store": train_store,
            "holdout_store": holdout_store,
            "full_store": full_store,
            "holdout_problem_summary": holdout_problem_summary,
        }
        train_tables = _build_domain_training_tables(train_store, ANCHOR_POSITIONS)
        holdout_tables = _build_domain_training_tables(holdout_store, ANCHOR_POSITIONS)
        train_tables_by_domain[domain_name] = train_tables
        holdout_tables_by_domain[domain_name] = holdout_tables

        routes_by_model = _train_routes_for_domain_store(
            domain_name=str(domain_name),
            tables=train_tables,
            model_specs=selected_model_specs,
            random_seeds=random_seeds,
            n_splits=int(args.n_splits),
            max_workers=int(args.max_workers),
        )
        routes_by_domain_model[domain_name] = routes_by_model

        for spec in selected_model_specs:
            routes_by_domain = {
                str(domain_name): {
                    str(spec.model_name): routes_by_model[str(spec.model_name)]
                }
            }
            score_fn = _make_routes_score_fn(routes_by_domain, str(spec.model_name))
            eval_payload = evaluate_method_from_feature_store(
                method_name=str(spec.model_name),
                feature_store=holdout_store,
                position_values=tuple(float(v) for v in EARLY_STOP_POSITIONS),
                score_fn=score_fn,
            )
            id_eval_by_domain_model.setdefault(domain_name, {})[str(spec.model_name)] = eval_payload
            csv_rows.extend(
                _anchor_route_rows(
                    domain=str(domain_name),
                    model_name=str(spec.model_name),
                    routes=routes_by_model[str(spec.model_name)],
                    holdout_tables=holdout_tables,
                )
            )

    svd_holdout_reference = _load_svd_holdout_reference(svd_holdout_reference_path)
    svd_ood_reference = _load_svd_ood_reference(svd_ood_reference_path) if svd_ood_reference_path.exists() else {}

    for domain_name, eval_by_model in sorted(id_eval_by_domain_model.items(), key=lambda item: OUTPUT_DOMAIN_ORDER.index(item[0]) if item[0] in OUTPUT_DOMAIN_ORDER else 999):
        for spec in selected_model_specs:
            if str(spec.model_name) not in eval_by_model:
                continue
            csv_rows.append(
                _domain_summary_row(
                    domain=str(domain_name),
                    spec=spec,
                    routes=routes_by_domain_model[domain_name][str(spec.model_name)],
                    eval_payload=eval_by_model[str(spec.model_name)],
                    svd_reference=svd_holdout_reference.get(domain_name, {}),
                )
            )

    if "math" in id_eval_by_domain_model and "science" in id_eval_by_domain_model:
        combined_holdout_store = list(split_packs["math"]["holdout_store"]) + list(split_packs["science"]["holdout_store"])
        for spec in selected_model_specs:
            routes_by_domain = {
                "math": {str(spec.model_name): routes_by_domain_model["math"][str(spec.model_name)]},
                "science": {str(spec.model_name): routes_by_domain_model["science"][str(spec.model_name)]},
            }
            eval_payload = evaluate_method_from_feature_store(
                method_name=str(spec.model_name),
                feature_store=combined_holdout_store,
                position_values=tuple(float(v) for v in EARLY_STOP_POSITIONS),
                score_fn=_make_routes_score_fn(routes_by_domain, str(spec.model_name)),
            )
            csv_rows.append(
                _domain_summary_row(
                    domain="ms",
                    spec=spec,
                    routes=routes_by_domain_model["math"][str(spec.model_name)],
                    eval_payload=eval_payload,
                    svd_reference=svd_holdout_reference.get("ms", {}),
                )
            )

    if not bool(args.disable_structured_ood) and test_store:
        all_feature_store = list(main_store) + list(extra_store) + list(test_store)
        ood_splits = _build_structured_ood_splits(all_feature_store)
        for domain_name in ("math", "science", "coding"):
            if domain_name not in selected_domains:
                continue
            if domain_name not in ood_splits:
                continue
            for protocol, split_defs in ood_splits[domain_name].items():
                protocol_rows_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for split_name, train_store, test_store_fold in split_defs:
                    print(f"[ood] domain={domain_name} protocol={protocol} split={split_name}", flush=True)
                    train_tables = _build_domain_training_tables(train_store, ANCHOR_POSITIONS)
                    try:
                        split_routes = _train_routes_for_domain_store(
                            domain_name=f"{domain_name}_{protocol}_{split_name}",
                            tables=train_tables,
                            model_specs=selected_model_specs,
                            random_seeds=random_seeds,
                            n_splits=int(args.n_splits),
                            max_workers=int(args.max_workers),
                        )
                    except Exception as exc:
                        print(f"[skip] ood domain={domain_name} protocol={protocol} split={split_name} reason={exc}", flush=True)
                        continue
                    eval_by_model = _eval_rows_from_feature_store(
                        domain=str(domain_name),
                        model_specs=selected_model_specs,
                        routes_by_model=split_routes,
                        test_store=test_store_fold,
                    )
                    for spec in selected_model_specs:
                        aggregate = dict(eval_by_model[str(spec.model_name)]["aggregate"])
                        protocol_rows_by_model[str(spec.model_name)].append(
                            {
                                "split_name": str(split_name),
                                "auc_of_auroc": float(aggregate.get("auc_of_auroc", float("nan"))),
                                "auc_of_selacc": float(aggregate.get("auc_of_selacc", float("nan"))),
                                "auroc_at_100": float(aggregate.get("auroc@100%", float("nan"))),
                                "stop_acc_at_100": float(aggregate.get("stop_acc@100%", float("nan"))),
                            }
                        )
                for spec in selected_model_specs:
                    id_row = _select_row(csv_rows, domain=domain_name, model_name=str(spec.model_name), row_kind="id_summary")
                    if id_row is None:
                        continue
                    csv_rows.append(
                        _ood_summary_row(
                            domain=str(domain_name),
                            spec=spec,
                            id_row=id_row,
                            protocol=str(protocol),
                            ood_rows=list(protocol_rows_by_model.get(str(spec.model_name), [])),
                            svd_reference=svd_ood_reference.get((str(domain_name), str(protocol))),
                        )
                    )

    order_map = {name: idx for idx, name in enumerate(METHOD_DISPLAY_ORDER)}
    row_kind_order = {"id_summary": 10, "anchor_route": 20, "ood_summary": 30}
    csv_rows.sort(
        key=lambda row: (
            OUTPUT_DOMAIN_ORDER.index(str(row["domain"])) if str(row["domain"]) in OUTPUT_DOMAIN_ORDER else 999,
            row_kind_order.get(str(row["row_kind"]), 999),
            order_map.get(str(row["model_name"]), 999),
            999 if row["anchor_pct"] == "" else int(row["anchor_pct"]),
            str(row["ood_protocol"]),
        )
    )

    protocol_summary = {
        "feature_family": FIXED_FAMILY_NAME,
        "feature_count": int(len(FIXED_FEATURE_NAMES)),
        "anchor_positions_pct": [int(round(float(v) * 100.0)) for v in ANCHOR_POSITIONS],
        "holdout_split_label": f"{int(round((1.0 - float(args.holdout_split)) * 100.0))}/{int(round(float(args.holdout_split) * 100.0))}",
        "n_splits": int(args.n_splits),
        "random_seeds": list(int(seed) for seed in random_seeds),
        "cache_roots": {
            "cache": str(main_cache_root),
            "cache_train": str(extra_cache_root),
            "cache_test": str(test_cache_root),
        },
        "feature_store_state": {
            "cache": main_store_meta,
            "cache_train": extra_store_meta,
            "cache_test": test_store_meta,
        },
        "data_summary": {
            domain_name: {
                "full": _summarise_feature_store(domain_stores.get(domain_name, [])),
                "train": _summarise_feature_store(split_packs.get(domain_name, {}).get("train_store", [])),
                "holdout": _summarise_feature_store(split_packs.get(domain_name, {}).get("holdout_store", [])),
                "holdout_problem_summary": split_packs.get(domain_name, {}).get("holdout_problem_summary", {}),
            }
            for domain_name in selected_domains
        },
        "structured_ood_enabled": not bool(args.disable_structured_ood),
        "structured_ood_store_summary": _store_summary(list(main_store) + list(extra_store) + list(test_store)) if not bool(args.disable_structured_ood) else {},
    }

    _write_csv(out_csv, csv_rows)
    _write_report(
        path=out_doc,
        rows=csv_rows,
        protocol_summary=protocol_summary,
        svd_holdout_reference=svd_holdout_reference,
    )

    print("[done] direct linear baselines", flush=True)
    print(f"  table : {_display_path(out_csv)}", flush=True)
    print(f"  doc   : {_display_path(out_doc)}", flush=True)


if __name__ == "__main__":
    main()
