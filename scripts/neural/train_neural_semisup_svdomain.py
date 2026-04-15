#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from nad.ops.earlystop import EARLY_STOP_POSITIONS
from nad.ops.earlystop_svd import (
    _auroc,
    _fit_svd_transform,
    get_domain,
    load_earlystop_svd_bundle,
)
from nad.ops.neural_semisup_svdomain import (
    ANCHOR_POSITIONS,
    CANONICAL_FEATURE_INDICES,
    CANONICAL_FEATURE_NAMES,
    DOMAIN_TO_ID,
    FrozenNeuralFeatureTransform,
    IdentityFeatureTransform,
    LowRankAdapterScorer,
    NeuralPretrainConfig,
    ProjectedMLPScorer,
    ProjectedPairwiseScorer,
    ProjectedPointwiseScorer,
    build_anchor_route_bundle,
    build_run_matrix_from_feature_store,
    compute_latent_reuse_matrix,
    decoder_feature_strengths,
    extract_anchor_examples,
    filter_run_matrix,
    fit_frozen_svd_basis,
    fit_lowrank_latent_scorer,
    pretrain_neural_encoder,
    require_torch,
    subsample_run_matrix_by_group_fraction,
)
from scripts.export_earlystop_svd_submission import _load_or_build_feature_store
from scripts.run_earlystop_prefix10_svd_round1 import (
    EXTRACTION_POSITIONS,
    OFFICIAL_POSITION_INDEX,
    _display_path,
    _now_utc,
    _pct_label,
    evaluate_method_from_feature_store,
    evaluate_single_position_route_from_feature_store,
    make_svd_bundle_score_fn,
)
from SVDomain.train_es_svd_ms_rr_r1 import (
    FIXED_FAMILY_NAME,
    FIXED_REPRESENTATION,
    _build_holdout_problem_map,
    _qualify_feature_store,
    _split_feature_store,
)


DEFAULT_MAIN_CACHE_ROOT = "MUI_HUB/cache"
DEFAULT_EXTRA_CACHE_ROOT = "MUI_HUB/cache_train"
DEFAULT_PRETRAIN_EXTRA_ROOTS = ""
DEFAULT_FEATURE_CACHE_DIR = "results/cache/neural_semisup_svdomain"
DEFAULT_OUT_CSV = "results/tables/neural_semisup_svdomain.csv"
DEFAULT_OUT_DOC = "docs/NEURAL_SEMISUP_SVDOMAIN.md"
DEFAULT_OUT_FIG_DIR = "results/figures/neural_semisup_svdomain"
DEFAULT_OUT_MODEL_DIR = "models/ml_selectors/neural_semisup_svdomain"
DEFAULT_CURRENT_BUNDLES = {
    "math": "models/ml_selectors/es_svd_math_rr_r1.pkl",
    "science": "models/ml_selectors/es_svd_science_rr_r1.pkl",
    "coding": "models/ml_selectors/es_svd_coding_rr_r1.pkl",
}
DEFAULT_CURRENT_CODING_PAIRWISE = "models/ml_selectors/slot100_svd_code_domain_r1_cap10__pairwise.pkl"
DEFAULT_LATENT_DIMS = (8, 16, 24)
DEFAULT_LABEL_FRACTIONS = (1, 5, 10, 20, 50, 100)
DEFAULT_C_VALUES = (0.10, 1.0)
DEFAULT_CLASS_WEIGHTS = ("none", "balanced")
DEFAULT_SVD_RANKS = (4, 8, 16)
DEFAULT_SVD_WHITEN = (False, True)
DEFAULT_MLP_HIDDEN = ((32,), (64,))
DEFAULT_MLP_ALPHA = (1e-4, 1e-3)
DEFAULT_ADAPTER_RANKS = (0, 2, 4)
DEFAULT_ADAPTER_PAIRWISE = (0.25, 0.50, 1.00)
DEFAULT_ADAPTER_PENALTY = (0.01, 0.03)
DEFAULT_TRAIN_METHODS = (
    "neural_semisup_freeze",
    "neural_semisup_coding_adapter",
    "no_svd_logreg",
    "frozen_svd_linear_head",
    "shallow_mlp_no_ssl",
)
DEFAULT_BATCH_SIZE = 256
DEFAULT_PRETRAIN_EPOCHS = 16
DEFAULT_RANDOM_STATE = 42


def _parse_csv(raw: str, *, cast: Callable[[str], Any]) -> tuple[Any, ...]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    return tuple(cast(item) for item in values)


def _safe_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return float("nan")
    return float(np.mean(finite))


def _safe_float_cell(value: Any) -> float:
    if value in {"", None}:
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def _best_metric_by_fraction(
    rows: list[dict[str, Any]],
    *,
    method_name: str,
    domain_scope: str,
    metric_scope: str,
    metric_key: str,
) -> list[tuple[float, float]]:
    best_by_fraction: dict[int, float] = {}
    for row in rows:
        if str(row.get("method_name")) != str(method_name):
            continue
        if str(row.get("domain_scope")) != str(domain_scope):
            continue
        if str(row.get("metric_scope")) != str(metric_scope):
            continue
        fraction = int(row["label_fraction_pct"])
        metric = _safe_float_cell(row.get(metric_key))
        if not np.isfinite(metric):
            continue
        current = best_by_fraction.get(fraction, float("-inf"))
        if metric > current:
            best_by_fraction[fraction] = metric
    return [(float(fraction), float(best_by_fraction[fraction])) for fraction in sorted(best_by_fraction)]


def _group_folds(groups: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups_arr = np.asarray(groups, dtype=object)
    unique = np.unique(groups_arr)
    if unique.shape[0] < 2:
        return []
    splits = min(int(n_splits), int(unique.shape[0]))
    if splits < 2:
        return []
    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=splits)
    dummy = np.zeros((groups_arr.shape[0], 1), dtype=np.float64)
    return list(gkf.split(dummy, groups=groups_arr))


def _build_pairwise_logistic_examples(
    x: np.ndarray,
    y: np.ndarray,
    rank_groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int32).reshape(-1)
    g_arr = np.asarray(rank_groups, dtype=object).reshape(-1)
    if x_arr.ndim != 2:
        raise ValueError(f"Expected 2D x matrix, got shape={x_arr.shape}")
    n_feat = int(x_arr.shape[1])
    by_group: dict[str, list[int]] = {}
    for idx, group in enumerate(g_arr.tolist()):
        by_group.setdefault(str(group), []).append(int(idx))

    pair_x_parts: list[np.ndarray] = []
    pair_y_parts: list[np.ndarray] = []
    for idxs in by_group.values():
        x_g = x_arr[idxs]
        y_g = y_arr[idxs]
        pos_idx = np.where(y_g > 0)[0]
        neg_idx = np.where(y_g <= 0)[0]
        if pos_idx.size == 0 or neg_idx.size == 0:
            continue
        diffs_pos = (x_g[pos_idx][:, None, :] - x_g[neg_idx][None, :, :]).reshape(-1, n_feat)
        diffs_neg = (x_g[neg_idx][None, :, :] - x_g[pos_idx][:, None, :]).reshape(-1, n_feat)
        pair_x_parts.append(np.concatenate([diffs_pos, diffs_neg], axis=0))
        pair_y_parts.append(
            np.concatenate(
                [
                    np.ones(diffs_pos.shape[0], dtype=np.int32),
                    np.zeros(diffs_neg.shape[0], dtype=np.int32),
                ]
            )
        )
    if not pair_x_parts:
        return np.zeros((0, n_feat), dtype=np.float64), np.zeros((0,), dtype=np.int32)
    return (
        np.concatenate(pair_x_parts, axis=0).astype(np.float64, copy=False),
        np.concatenate(pair_y_parts, axis=0).astype(np.int32, copy=False),
    )


def _discover_all_cache_keys(cache_root: str) -> tuple[str, ...]:
    from nad.ops.earlystop import discover_cache_entries

    return tuple(sorted(str(entry.cache_key) for entry in discover_cache_entries(cache_root)))


def _load_root_feature_store(
    *,
    source_name: str,
    cache_root: str,
    positions: tuple[float, ...],
    required_feature_names: set[str],
    max_problems_per_cache: int | None,
    feature_workers: int,
    feature_chunk_problems: int,
    feature_cache_dir: Path | None,
    refresh_feature_cache: bool,
) -> tuple[list[dict[str, Any]], Optional[Path], str]:
    include_cache_keys = set(_discover_all_cache_keys(cache_root))
    if not include_cache_keys:
        return [], None, "skipped_empty_root"
    raw_store, cache_path, cache_status = _load_or_build_feature_store(
        cache_root=cache_root,
        positions=positions,
        required_feature_names=required_feature_names,
        max_problems=max_problems_per_cache,
        reflection_threshold=0.30,
        workers=int(feature_workers),
        feature_chunk_problems=int(feature_chunk_problems),
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(refresh_feature_cache),
        include_cache_keys=include_cache_keys,
    )
    return _qualify_feature_store(raw_store, source_name), cache_path, cache_status


def _load_feature_store_from_pickle(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    feature_store = obj["feature_store"] if isinstance(obj, dict) and "feature_store" in obj else obj
    return list(feature_store)


def _best_single_feature_baseline(
    x_raw: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
) -> dict[str, Any]:
    best = {"signal_name": str(CANONICAL_FEATURE_NAMES[0]), "cv_auroc": float("-inf")}
    folds = _group_folds(groups, n_splits=n_splits)
    if not folds:
        return {"signal_name": str(CANONICAL_FEATURE_NAMES[0]), "cv_auroc": float("nan")}
    for feature_idx, feature_name in enumerate(CANONICAL_FEATURE_NAMES):
        vals: list[float] = []
        scores = np.asarray(x_raw[:, feature_idx], dtype=np.float64)
        for _, test_idx in folds:
            y_test = np.asarray(y[test_idx], dtype=np.int32)
            if np.unique(y_test).shape[0] < 2:
                continue
            auc = _auroc(scores[test_idx], y_test)
            if np.isfinite(auc):
                vals.append(float(auc))
        cv_auc = _safe_mean(vals)
        if np.isfinite(cv_auc) and cv_auc > float(best["cv_auroc"]):
            best = {"signal_name": str(feature_name), "cv_auroc": float(cv_auc)}
    return best


def _cv_pointwise_logreg(
    x_rep: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    basis_fit: Callable[[np.ndarray, dict[str, Any]], Any | None],
    basis_grid: list[dict[str, Any]],
    c_values: tuple[float, ...],
    class_weights: tuple[str, ...],
    n_splits: int,
    random_state: int,
) -> dict[str, Any]:
    folds = _group_folds(groups, n_splits=n_splits)
    best: dict[str, Any] = {"cv_auroc": float("-inf")}
    if not folds:
        return best

    for basis_params in basis_grid:
        for c_value in c_values:
            for class_weight in class_weights:
                vals: list[float] = []
                for train_idx, test_idx in folds:
                    y_train = np.asarray(y[train_idx], dtype=np.int32)
                    y_test = np.asarray(y[test_idx], dtype=np.int32)
                    if np.unique(y_train).shape[0] < 2 or np.unique(y_test).shape[0] < 2:
                        continue
                    basis = basis_fit(np.asarray(x_rep[train_idx], dtype=np.float64), basis_params)
                    if basis is None:
                        continue
                    z_train = basis.transform(np.asarray(x_rep[train_idx], dtype=np.float64))
                    z_test = basis.transform(np.asarray(x_rep[test_idx], dtype=np.float64))
                    scaler = StandardScaler()
                    z_train_scaled = scaler.fit_transform(z_train)
                    z_test_scaled = scaler.transform(z_test)
                    clf = LogisticRegression(
                        C=float(c_value),
                        class_weight=None if str(class_weight) == "none" else "balanced",
                        max_iter=4000,
                        random_state=int(random_state),
                    )
                    try:
                        clf.fit(z_train_scaled, y_train)
                        scores = np.asarray(clf.decision_function(z_test_scaled), dtype=np.float64)
                    except Exception:
                        continue
                    auc = _auroc(scores, y_test)
                    if np.isfinite(auc):
                        vals.append(float(auc))
                cv_auc = _safe_mean(vals)
                if np.isfinite(cv_auc) and cv_auc > float(best["cv_auroc"]):
                    best = {
                        "cv_auroc": float(cv_auc),
                        "basis_params": dict(basis_params),
                        "c_value": float(c_value),
                        "class_weight": str(class_weight),
                    }
    return best


def _fit_pointwise_logreg(
    x_rep: np.ndarray,
    y: np.ndarray,
    *,
    basis_fit: Callable[[np.ndarray, dict[str, Any]], Any | None],
    basis_params: dict[str, Any],
    c_value: float,
    class_weight: str,
    random_state: int,
) -> ProjectedPointwiseScorer:
    basis = basis_fit(np.asarray(x_rep, dtype=np.float64), dict(basis_params))
    if basis is None:
        raise ValueError(f"Basis fit failed with params={basis_params}")
    z = basis.transform(np.asarray(x_rep, dtype=np.float64))
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z)
    clf = LogisticRegression(
        C=float(c_value),
        class_weight=None if str(class_weight) == "none" else "balanced",
        max_iter=4000,
        random_state=int(random_state),
    )
    clf.fit(z_scaled, np.asarray(y, dtype=np.int32))
    return ProjectedPointwiseScorer(basis=basis, scaler=scaler, clf=clf)


def _cv_pairwise_logreg(
    x_rep: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    basis_fit: Callable[[np.ndarray, dict[str, Any]], Any | None],
    basis_grid: list[dict[str, Any]],
    c_values: tuple[float, ...],
    n_splits: int,
    random_state: int,
) -> dict[str, Any]:
    folds = _group_folds(groups, n_splits=n_splits)
    best: dict[str, Any] = {"cv_auroc": float("-inf")}
    if not folds:
        return best
    for basis_params in basis_grid:
        for c_value in c_values:
            vals: list[float] = []
            for train_idx, test_idx in folds:
                y_train = np.asarray(y[train_idx], dtype=np.int32)
                y_test = np.asarray(y[test_idx], dtype=np.int32)
                if np.unique(y_train).shape[0] < 2 or np.unique(y_test).shape[0] < 2:
                    continue
                basis = basis_fit(np.asarray(x_rep[train_idx], dtype=np.float64), basis_params)
                if basis is None:
                    continue
                z_train = basis.transform(np.asarray(x_rep[train_idx], dtype=np.float64))
                z_test = basis.transform(np.asarray(x_rep[test_idx], dtype=np.float64))
                x_pairs, y_pairs = _build_pairwise_logistic_examples(z_train, y_train, groups[train_idx])
                if x_pairs.shape[0] <= 0:
                    continue
                scaler = StandardScaler(with_mean=False)
                x_pairs_scaled = scaler.fit_transform(x_pairs)
                clf = LogisticRegression(
                    C=float(c_value),
                    fit_intercept=False,
                    max_iter=4000,
                    random_state=int(random_state),
                )
                try:
                    clf.fit(x_pairs_scaled, y_pairs)
                    scores = np.asarray(clf.decision_function(scaler.transform(z_test)), dtype=np.float64)
                except Exception:
                    continue
                auc = _auroc(scores, y_test)
                if np.isfinite(auc):
                    vals.append(float(auc))
            cv_auc = _safe_mean(vals)
            if np.isfinite(cv_auc) and cv_auc > float(best["cv_auroc"]):
                best = {
                    "cv_auroc": float(cv_auc),
                    "basis_params": dict(basis_params),
                    "c_value": float(c_value),
                }
    return best


def _fit_pairwise_logreg(
    x_rep: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    basis_fit: Callable[[np.ndarray, dict[str, Any]], Any | None],
    basis_params: dict[str, Any],
    c_value: float,
    random_state: int,
) -> ProjectedPairwiseScorer:
    basis = basis_fit(np.asarray(x_rep, dtype=np.float64), dict(basis_params))
    if basis is None:
        raise ValueError(f"Basis fit failed with params={basis_params}")
    z = basis.transform(np.asarray(x_rep, dtype=np.float64))
    x_pairs, y_pairs = _build_pairwise_logreg_examples(z, y, groups)
    scaler = StandardScaler(with_mean=False)
    x_pairs_scaled = scaler.fit_transform(x_pairs)
    clf = LogisticRegression(
        C=float(c_value),
        fit_intercept=False,
        max_iter=4000,
        random_state=int(random_state),
    )
    clf.fit(x_pairs_scaled, y_pairs)
    return ProjectedPairwiseScorer(basis=basis, scaler=scaler, clf=clf)


def _cv_lowrank_adapter(
    x_rep: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    snapshot: Any,
    domain_name: str,
    anchor_position: float,
    adapter_ranks: tuple[int, ...],
    pairwise_weights: tuple[float, ...],
    adapter_penalties: tuple[float, ...],
    n_splits: int,
    random_state: int,
) -> dict[str, Any]:
    folds = _group_folds(groups, n_splits=n_splits)
    best: dict[str, Any] = {"cv_auroc": float("-inf")}
    if not folds:
        return best

    for adapter_rank in adapter_ranks:
        for pairwise_weight in pairwise_weights:
            for adapter_penalty in adapter_penalties:
                vals: list[float] = []
                for train_idx, test_idx in folds:
                    y_train = np.asarray(y[train_idx], dtype=np.int32)
                    y_test = np.asarray(y[test_idx], dtype=np.int32)
                    if np.unique(y_train).shape[0] < 2 or np.unique(y_test).shape[0] < 2:
                        continue
                    try:
                        scorer = fit_lowrank_latent_scorer(
                            np.asarray(x_rep[train_idx], dtype=np.float64),
                            y_train,
                            np.asarray(groups[train_idx], dtype=object),
                            snapshot=snapshot,
                            domain_name=str(domain_name),
                            anchor_position=float(anchor_position),
                            adapter_rank=int(adapter_rank),
                            pairwise_weight=float(pairwise_weight),
                            adapter_penalty=float(adapter_penalty),
                            epochs=90,
                            random_state=int(random_state),
                        )
                        scores = np.asarray(
                            scorer.score_group(np.asarray(x_rep[test_idx], dtype=np.float64)),
                            dtype=np.float64,
                        )
                    except Exception:
                        continue
                    auc = _auroc(scores, y_test)
                    if np.isfinite(auc):
                        vals.append(float(auc))
                cv_auc = _safe_mean(vals)
                if np.isfinite(cv_auc) and cv_auc > float(best["cv_auroc"]):
                    best = {
                        "cv_auroc": float(cv_auc),
                        "adapter_rank": int(adapter_rank),
                        "pairwise_weight": float(pairwise_weight),
                        "adapter_penalty": float(adapter_penalty),
                    }
    return best


def _fit_lowrank_adapter_route(
    x_rep: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    snapshot: Any,
    domain_name: str,
    anchor_position: float,
    adapter_rank: int,
    pairwise_weight: float,
    adapter_penalty: float,
    random_state: int,
) -> LowRankAdapterScorer:
    return fit_lowrank_latent_scorer(
        np.asarray(x_rep, dtype=np.float64),
        np.asarray(y, dtype=np.int32),
        np.asarray(groups, dtype=object),
        snapshot=snapshot,
        domain_name=str(domain_name),
        anchor_position=float(anchor_position),
        adapter_rank=int(adapter_rank),
        pairwise_weight=float(pairwise_weight),
        adapter_penalty=float(adapter_penalty),
        epochs=140,
        random_state=int(random_state),
    )


def _build_pairwise_logreg_examples(
    z: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_pairs, y_pairs = _build_pairwise_logistic_examples(
        np.asarray(z, dtype=np.float64),
        np.asarray(y, dtype=np.int32),
        np.asarray(groups, dtype=object),
    )
    if x_pairs.shape[0] <= 0:
        raise ValueError("No valid positive-vs-negative pairs found")
    return x_pairs, y_pairs


def _cv_mlp(
    x_rep: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    hidden_options: tuple[tuple[int, ...], ...],
    alpha_options: tuple[float, ...],
    n_splits: int,
    random_state: int,
) -> dict[str, Any]:
    folds = _group_folds(groups, n_splits=n_splits)
    best: dict[str, Any] = {"cv_auroc": float("-inf")}
    if not folds:
        return best
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for hidden_layers in hidden_options:
            for alpha in alpha_options:
                vals: list[float] = []
                for train_idx, test_idx in folds:
                    y_train = np.asarray(y[train_idx], dtype=np.int32)
                    y_test = np.asarray(y[test_idx], dtype=np.int32)
                    if np.unique(y_train).shape[0] < 2 or np.unique(y_test).shape[0] < 2:
                        continue
                    scaler = StandardScaler()
                    x_train = scaler.fit_transform(np.asarray(x_rep[train_idx], dtype=np.float64))
                    x_test = scaler.transform(np.asarray(x_rep[test_idx], dtype=np.float64))
                    clf = MLPClassifier(
                        hidden_layer_sizes=tuple(int(v) for v in hidden_layers),
                        activation="relu",
                        alpha=float(alpha),
                        random_state=int(random_state),
                        max_iter=300,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=12,
                    )
                    try:
                        clf.fit(x_train, y_train)
                        scores = np.asarray(clf.predict_proba(x_test)[:, 1], dtype=np.float64)
                    except Exception:
                        continue
                    auc = _auroc(scores, y_test)
                    if np.isfinite(auc):
                        vals.append(float(auc))
                cv_auc = _safe_mean(vals)
                if np.isfinite(cv_auc) and cv_auc > float(best["cv_auroc"]):
                    best = {
                        "cv_auroc": float(cv_auc),
                        "hidden_layers": tuple(int(v) for v in hidden_layers),
                        "alpha": float(alpha),
                    }
    return best


def _fit_mlp(
    x_rep: np.ndarray,
    y: np.ndarray,
    *,
    hidden_layers: tuple[int, ...],
    alpha: float,
    random_state: int,
) -> ProjectedMLPScorer:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(np.asarray(x_rep, dtype=np.float64))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf = MLPClassifier(
            hidden_layer_sizes=tuple(int(v) for v in hidden_layers),
            activation="relu",
            alpha=float(alpha),
            random_state=int(random_state),
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=12,
        )
        clf.fit(x_scaled, np.asarray(y, dtype=np.int32))
    return ProjectedMLPScorer(basis=IdentityFeatureTransform(), scaler=scaler, clf=clf)


def _identity_basis_fit(_x: np.ndarray, _params: dict[str, Any]) -> IdentityFeatureTransform:
    return IdentityFeatureTransform()


def _svd_basis_fit(x: np.ndarray, params: dict[str, Any]) -> Any | None:
    return fit_frozen_svd_basis(
        np.asarray(x, dtype=np.float64),
        rank=int(params["rank"]),
        whiten=bool(params["whiten"]),
        random_state=int(params["random_state"]),
    )


def _neural_basis_fit_builder(snapshot, *, domain_name: str, anchor_position: float):
    def _fit(_x: np.ndarray, _params: dict[str, Any]) -> FrozenNeuralFeatureTransform:
        return FrozenNeuralFeatureTransform(
            snapshot=snapshot,
            domain_name=str(domain_name),
            anchor_position=float(anchor_position),
        )

    return _fit


def _has_valid_cv_result(best: dict[str, Any]) -> bool:
    return bool(np.isfinite(_safe_float_cell(best.get("cv_auroc"))))


def _route_metric_summary(
    *,
    feature_store: list[dict[str, Any]],
    bundle: dict[str, Any],
    domains: tuple[str, ...],
) -> dict[str, Any]:
    subset = [payload for payload in feature_store if str(payload["domain"]) in set(domains)]
    if not subset:
        return {}
    result = evaluate_method_from_feature_store(
        method_name=str(bundle["bundle_version"]),
        feature_store=subset,
        position_values=tuple(float(v) for v in EARLY_STOP_POSITIONS),
        score_fn=make_svd_bundle_score_fn(bundle),
    )
    return dict(result["aggregate"])


def _anchor_metric_summary(
    *,
    feature_store: list[dict[str, Any]],
    bundle: dict[str, Any],
    position: float,
    domains: tuple[str, ...],
) -> dict[str, Any]:
    subset = [payload for payload in feature_store if str(payload["domain"]) in set(domains)]
    if not subset:
        return {}

    def _route_resolver(domain_name: str) -> Optional[dict[str, Any]]:
        if str(domain_name) not in bundle["domains"]:
            return None
        routes = bundle["domains"][str(domain_name)]["routes"]
        if len(routes) == 1:
            return routes[0]
        route_idx = int(OFFICIAL_POSITION_INDEX[float(position)])
        return routes[route_idx]

    result = evaluate_single_position_route_from_feature_store(
        feature_store=subset,
        position=float(position),
        route_resolver=_route_resolver,
    )
    return dict(result["aggregate"])


def _fit_route_for_method(
    *,
    method_name: str,
    domain_name: str,
    anchor_position: float,
    examples: Any,
    n_splits: int,
    random_state: int,
    neural_snapshot: Any | None = None,
) -> dict[str, Any]:
    x_raw = np.asarray(examples.x_raw, dtype=np.float64)
    x_rep = np.asarray(examples.x_rep, dtype=np.float64)
    y = np.asarray(examples.y, dtype=np.int32)
    groups = np.asarray(examples.groups, dtype=object)
    best_baseline = _best_single_feature_baseline(x_raw, y, groups, n_splits=n_splits)

    if x_rep.shape[0] <= 0 or np.unique(y).shape[0] < 2:
        raise ValueError(f"{method_name}:{domain_name}@{_pct_label(anchor_position)} has insufficient data")

    if str(method_name) == "neural_semisup_freeze":
        basis_grid = [{"kind": "neural"}]
        basis_fit = _neural_basis_fit_builder(neural_snapshot, domain_name=domain_name, anchor_position=anchor_position)
        is_pairwise_route = str(domain_name) == "coding" and float(anchor_position) in {0.70, 1.00}
        best = _cv_pairwise_logreg(
            x_rep,
            y,
            groups,
            basis_fit=basis_fit,
            basis_grid=basis_grid,
            c_values=DEFAULT_C_VALUES,
            n_splits=n_splits,
            random_state=random_state,
        ) if is_pairwise_route else _cv_pointwise_logreg(
            x_rep,
            y,
            groups,
            basis_fit=basis_fit,
            basis_grid=basis_grid,
            c_values=DEFAULT_C_VALUES,
            class_weights=DEFAULT_CLASS_WEIGHTS,
            n_splits=n_splits,
            random_state=random_state,
        )
        used_fallback = not _has_valid_cv_result(best)
        if used_fallback:
            best = {
                "cv_auroc": float("nan"),
                "basis_params": dict(basis_grid[0]),
                "c_value": float(DEFAULT_C_VALUES[0]),
                "class_weight": "balanced",
            }
        if is_pairwise_route:
            try:
                scorer = _fit_pairwise_logreg(
                    x_rep,
                    y,
                    groups,
                    basis_fit=basis_fit,
                    basis_params=best["basis_params"],
                    c_value=float(best["c_value"]),
                    random_state=random_state,
                )
                route_type = "ranksvm"
            except Exception:
                scorer = _fit_pointwise_logreg(
                    x_rep,
                    y,
                    basis_fit=basis_fit,
                    basis_params=best["basis_params"],
                    c_value=float(best["c_value"]),
                    class_weight="balanced",
                    random_state=random_state,
                )
                route_type = "pointwise"
        else:
            scorer = _fit_pointwise_logreg(
                x_rep,
                y,
                basis_fit=basis_fit,
                basis_params=best["basis_params"],
                c_value=float(best["c_value"]),
                class_weight=str(best["class_weight"]),
                random_state=random_state,
            )
            route_type = "pointwise"
        return {
            "route_type": route_type,
            "family_name": FIXED_FAMILY_NAME,
            "representation": FIXED_REPRESENTATION,
            "feature_names": list(CANONICAL_FEATURE_NAMES),
            "feature_indices": list(CANONICAL_FEATURE_INDICES),
            "training_position": float(anchor_position),
            "training_scope": str(domain_name),
            "baseline_signal_name": str(best_baseline["signal_name"]),
            "baseline_cv_auroc": float(best_baseline["cv_auroc"]),
            "cv_auroc": float(best["cv_auroc"]),
            "c_value": float(best["c_value"]),
            "class_weight": None if route_type == "ranksvm" else str(best["class_weight"]),
            "latent_dim": int(neural_snapshot.latent_dim),
            "scorer": scorer,
            "used_cv_fallback": bool(used_fallback),
        }

    if str(method_name) == "neural_semisup_coding_adapter":
        if str(domain_name) != "coding":
            return _fit_route_for_method(
                method_name="neural_semisup_freeze",
                domain_name=str(domain_name),
                anchor_position=float(anchor_position),
                examples=examples,
                n_splits=int(n_splits),
                random_state=int(random_state),
                neural_snapshot=neural_snapshot,
            )

        is_pairwise_route = float(anchor_position) in {0.70, 1.00}
        pairwise_weights = DEFAULT_ADAPTER_PAIRWISE if is_pairwise_route else (0.0,)
        best = _cv_lowrank_adapter(
            x_rep,
            y,
            groups,
            snapshot=neural_snapshot,
            domain_name=str(domain_name),
            anchor_position=float(anchor_position),
            adapter_ranks=DEFAULT_ADAPTER_RANKS,
            pairwise_weights=pairwise_weights,
            adapter_penalties=DEFAULT_ADAPTER_PENALTY,
            n_splits=n_splits,
            random_state=random_state,
        )
        used_fallback = not _has_valid_cv_result(best)
        if used_fallback:
            best = {
                "cv_auroc": float("nan"),
                "adapter_rank": 0,
                "pairwise_weight": float(pairwise_weights[0]),
                "adapter_penalty": float(DEFAULT_ADAPTER_PENALTY[0]),
            }
        scorer = _fit_lowrank_adapter_route(
            x_rep,
            y,
            groups,
            snapshot=neural_snapshot,
            domain_name=str(domain_name),
            anchor_position=float(anchor_position),
            adapter_rank=int(best["adapter_rank"]),
            pairwise_weight=float(best["pairwise_weight"]),
            adapter_penalty=float(best["adapter_penalty"]),
            random_state=random_state,
        )
        return {
            "route_type": "ranksvm" if is_pairwise_route and float(best["pairwise_weight"]) > 0.0 else "pointwise",
            "family_name": FIXED_FAMILY_NAME,
            "representation": FIXED_REPRESENTATION,
            "feature_names": list(CANONICAL_FEATURE_NAMES),
            "feature_indices": list(CANONICAL_FEATURE_INDICES),
            "training_position": float(anchor_position),
            "training_scope": str(domain_name),
            "baseline_signal_name": str(best_baseline["signal_name"]),
            "baseline_cv_auroc": float(best_baseline["cv_auroc"]),
            "cv_auroc": float(best["cv_auroc"]),
            "latent_dim": int(neural_snapshot.latent_dim),
            "adapter_rank": int(best["adapter_rank"]),
            "pairwise_weight": float(best["pairwise_weight"]),
            "adapter_penalty": float(best["adapter_penalty"]),
            "scorer": scorer,
            "used_cv_fallback": bool(used_fallback),
        }

    if str(method_name) == "neural_semisup_coding_hybrid":
        if str(domain_name) != "coding":
            return _fit_route_for_method(
                method_name="neural_semisup_freeze",
                domain_name=str(domain_name),
                anchor_position=float(anchor_position),
                examples=examples,
                n_splits=int(n_splits),
                random_state=int(random_state),
                neural_snapshot=neural_snapshot,
            )
        freeze_route = _fit_route_for_method(
            method_name="neural_semisup_freeze",
            domain_name=str(domain_name),
            anchor_position=float(anchor_position),
            examples=examples,
            n_splits=int(n_splits),
            random_state=int(random_state),
            neural_snapshot=neural_snapshot,
        )
        adapter_route = _fit_route_for_method(
            method_name="neural_semisup_coding_adapter",
            domain_name=str(domain_name),
            anchor_position=float(anchor_position),
            examples=examples,
            n_splits=int(n_splits),
            random_state=int(random_state),
            neural_snapshot=neural_snapshot,
        )
        freeze_auc = _safe_float_cell(freeze_route.get("cv_auroc"))
        adapter_auc = _safe_float_cell(adapter_route.get("cv_auroc"))
        if np.isfinite(adapter_auc) and (not np.isfinite(freeze_auc) or adapter_auc > freeze_auc + 1e-8):
            chosen = dict(adapter_route)
            chosen["hybrid_choice"] = "adapter"
            return chosen
        chosen = dict(freeze_route)
        chosen["hybrid_choice"] = "freeze"
        return chosen

    if str(method_name) == "no_svd_logreg":
        best = _cv_pointwise_logreg(
            x_rep,
            y,
            groups,
            basis_fit=_identity_basis_fit,
            basis_grid=[{"kind": "identity"}],
            c_values=DEFAULT_C_VALUES,
            class_weights=DEFAULT_CLASS_WEIGHTS,
            n_splits=n_splits,
            random_state=random_state,
        )
        if not _has_valid_cv_result(best):
            best = {
                "cv_auroc": float("nan"),
                "basis_params": {"kind": "identity"},
                "c_value": float(DEFAULT_C_VALUES[0]),
                "class_weight": "balanced",
            }
        scorer = _fit_pointwise_logreg(
            x_rep,
            y,
            basis_fit=_identity_basis_fit,
            basis_params=best["basis_params"],
            c_value=float(best["c_value"]),
            class_weight=str(best["class_weight"]),
            random_state=random_state,
        )
        return {
            "route_type": "pointwise",
            "family_name": FIXED_FAMILY_NAME,
            "representation": FIXED_REPRESENTATION,
            "feature_names": list(CANONICAL_FEATURE_NAMES),
            "feature_indices": list(CANONICAL_FEATURE_INDICES),
            "training_position": float(anchor_position),
            "training_scope": str(domain_name),
            "baseline_signal_name": str(best_baseline["signal_name"]),
            "baseline_cv_auroc": float(best_baseline["cv_auroc"]),
            "cv_auroc": float(best["cv_auroc"]),
            "c_value": float(best["c_value"]),
            "class_weight": str(best["class_weight"]),
            "scorer": scorer,
        }

    if str(method_name) == "frozen_svd_linear_head":
        basis_grid = [
            {"rank": int(rank), "whiten": bool(whiten), "random_state": int(random_state)}
            for rank in DEFAULT_SVD_RANKS
            for whiten in DEFAULT_SVD_WHITEN
        ]
        is_pairwise_route = str(domain_name) == "coding" and float(anchor_position) in {0.70, 1.00}
        best = _cv_pairwise_logreg(
            x_rep,
            y,
            groups,
            basis_fit=_svd_basis_fit,
            basis_grid=basis_grid,
            c_values=DEFAULT_C_VALUES,
            n_splits=n_splits,
            random_state=random_state,
        ) if is_pairwise_route else _cv_pointwise_logreg(
            x_rep,
            y,
            groups,
            basis_fit=_svd_basis_fit,
            basis_grid=basis_grid,
            c_values=DEFAULT_C_VALUES,
            class_weights=DEFAULT_CLASS_WEIGHTS,
            n_splits=n_splits,
            random_state=random_state,
        )
        used_fallback = not _has_valid_cv_result(best)
        if used_fallback:
            best = {
                "cv_auroc": float("nan"),
                "basis_params": dict(basis_grid[0]),
                "c_value": float(DEFAULT_C_VALUES[0]),
                "class_weight": "balanced",
            }
        if is_pairwise_route:
            try:
                scorer = _fit_pairwise_logreg(
                    x_rep,
                    y,
                    groups,
                    basis_fit=_svd_basis_fit,
                    basis_params=best["basis_params"],
                    c_value=float(best["c_value"]),
                    random_state=random_state,
                )
                route_type = "ranksvm"
            except Exception:
                scorer = _fit_pointwise_logreg(
                    x_rep,
                    y,
                    basis_fit=_svd_basis_fit,
                    basis_params=best["basis_params"],
                    c_value=float(best["c_value"]),
                    class_weight="balanced",
                    random_state=random_state,
                )
                route_type = "pointwise"
        else:
            scorer = _fit_pointwise_logreg(
                x_rep,
                y,
                basis_fit=_svd_basis_fit,
                basis_params=best["basis_params"],
                c_value=float(best["c_value"]),
                class_weight=str(best["class_weight"]),
                random_state=random_state,
            )
            route_type = "pointwise"
        return {
            "route_type": route_type,
            "family_name": FIXED_FAMILY_NAME,
            "representation": FIXED_REPRESENTATION,
            "feature_names": list(CANONICAL_FEATURE_NAMES),
            "feature_indices": list(CANONICAL_FEATURE_INDICES),
            "training_position": float(anchor_position),
            "training_scope": str(domain_name),
            "baseline_signal_name": str(best_baseline["signal_name"]),
            "baseline_cv_auroc": float(best_baseline["cv_auroc"]),
            "cv_auroc": float(best["cv_auroc"]),
            "c_value": float(best["c_value"]),
            "rank": int(best["basis_params"]["rank"]),
            "whiten": bool(best["basis_params"]["whiten"]),
            "scorer": scorer,
            "used_cv_fallback": bool(used_fallback),
        }

    if str(method_name) == "shallow_mlp_no_ssl":
        best = _cv_mlp(
            x_rep,
            y,
            groups,
            hidden_options=DEFAULT_MLP_HIDDEN,
            alpha_options=DEFAULT_MLP_ALPHA,
            n_splits=n_splits,
            random_state=random_state,
        )
        if not _has_valid_cv_result(best):
            best = {
                "cv_auroc": float("nan"),
                "hidden_layers": tuple(int(v) for v in DEFAULT_MLP_HIDDEN[0]),
                "alpha": float(DEFAULT_MLP_ALPHA[0]),
            }
        scorer = _fit_mlp(
            x_rep,
            y,
            hidden_layers=tuple(best["hidden_layers"]),
            alpha=float(best["alpha"]),
            random_state=random_state,
        )
        return {
            "route_type": "pointwise",
            "family_name": FIXED_FAMILY_NAME,
            "representation": FIXED_REPRESENTATION,
            "feature_names": list(CANONICAL_FEATURE_NAMES),
            "feature_indices": list(CANONICAL_FEATURE_INDICES),
            "training_position": float(anchor_position),
            "training_scope": str(domain_name),
            "baseline_signal_name": str(best_baseline["signal_name"]),
            "baseline_cv_auroc": float(best_baseline["cv_auroc"]),
            "cv_auroc": float(best["cv_auroc"]),
            "hidden_layers": list(best["hidden_layers"]),
            "alpha": float(best["alpha"]),
            "scorer": scorer,
        }

    raise ValueError(f"Unknown method_name={method_name}")


def _train_method_bundle(
    *,
    method_name: str,
    train_matrix: Any,
    label_fraction_pct: int,
    random_state: int,
    n_splits: int,
    neural_snapshot: Any | None,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    fraction = float(label_fraction_pct) / 100.0
    sampled = subsample_run_matrix_by_group_fraction(
        train_matrix,
        fraction=fraction,
        random_state=int(random_state) + int(label_fraction_pct),
    )
    routes_by_domain: dict[str, dict[float, dict[str, Any]]] = {}
    for domain_name in ("math", "science", "coding"):
        routes_by_domain[domain_name] = {}
        for anchor_position in ANCHOR_POSITIONS:
            examples = extract_anchor_examples(
                sampled,
                anchor_position=float(anchor_position),
                domains=(domain_name,),
            )
            route = _fit_route_for_method(
                method_name=method_name,
                domain_name=domain_name,
                anchor_position=float(anchor_position),
                examples=examples,
                n_splits=n_splits,
                random_state=random_state,
                neural_snapshot=neural_snapshot,
            )
            routes_by_domain[domain_name][float(anchor_position)] = route
    bundle_id = f"{method_name}__lf{int(label_fraction_pct)}"
    return build_anchor_route_bundle(
        method_id=bundle_id,
        routes_by_domain=routes_by_domain,
        protocol=dict(protocol),
    )


def _add_aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    method_name: str,
    bundle: dict[str, Any],
    holdout_store: list[dict[str, Any]],
    label_fraction_pct: int,
    latent_dim: int | None,
    pretrain_summary: dict[str, Any] | None,
    notes: str,
) -> None:
    scopes = {
        "math": ("math",),
        "science": ("science",),
        "coding": ("coding",),
        "ms": ("math", "science"),
    }
    for scope_name, domains in scopes.items():
        metrics = _route_metric_summary(
            feature_store=holdout_store,
            bundle=bundle,
            domains=tuple(domains),
        )
        if not metrics:
            continue
        rows.append(
            {
                "method_name": str(method_name),
                "metric_scope": "aggregate",
                "domain_scope": str(scope_name),
                "anchor_pct": "",
                "label_fraction_pct": int(label_fraction_pct),
                "latent_dim": "" if latent_dim is None else int(latent_dim),
                "auc_of_auroc": float(metrics.get("auc_of_auroc", float("nan"))),
                "auc_of_selacc": float(metrics.get("auc_of_selacc", float("nan"))),
                "earliest_gt_0_6": metrics.get("earliest_gt_0.6"),
                "auroc_at_100": float(metrics.get("auroc@100%", float("nan"))),
                "stop_acc_at_100": float(metrics.get("stop_acc@100%", float("nan"))),
                "auroc": "",
                "selacc_at_10": "",
                "stop_acc": "",
                "pretrain_best_loss": None if pretrain_summary is None else float(pretrain_summary["best_loss"]),
                "pretrain_best_epoch": None if pretrain_summary is None else int(pretrain_summary["best_epoch"]),
                "notes": str(notes),
            }
        )

    for anchor_position in (0.70, 1.00):
        metrics = _anchor_metric_summary(
            feature_store=holdout_store,
            bundle=bundle,
            position=float(anchor_position),
            domains=("coding",),
        )
        if not metrics:
            continue
        rows.append(
            {
                "method_name": str(method_name),
                "metric_scope": "anchor",
                "domain_scope": "coding",
                "anchor_pct": int(round(float(anchor_position) * 100.0)),
                "label_fraction_pct": int(label_fraction_pct),
                "latent_dim": "" if latent_dim is None else int(latent_dim),
                "auc_of_auroc": "",
                "auc_of_selacc": "",
                "earliest_gt_0_6": "",
                "auroc_at_100": "",
                "stop_acc_at_100": "",
                "auroc": float(metrics.get("auroc", float("nan"))),
                "selacc_at_10": float(metrics.get("selacc@10%", float("nan"))),
                "stop_acc": float(metrics.get("stop_acc", float("nan"))),
                "pretrain_best_loss": None if pretrain_summary is None else float(pretrain_summary["best_loss"]),
                "pretrain_best_epoch": None if pretrain_summary is None else int(pretrain_summary["best_epoch"]),
                "notes": str(notes),
            }
        )


def _evaluate_current_baselines(
    *,
    holdout_store: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    bundle_map = {
        "math": load_earlystop_svd_bundle(REPO_ROOT / DEFAULT_CURRENT_BUNDLES["math"]),
        "science": load_earlystop_svd_bundle(REPO_ROOT / DEFAULT_CURRENT_BUNDLES["science"]),
        "coding": load_earlystop_svd_bundle(REPO_ROOT / DEFAULT_CURRENT_BUNDLES["coding"]),
    }
    for scope_name, domains in {
        "math": ("math",),
        "science": ("science",),
        "coding": ("coding",),
    }.items():
        subset = [payload for payload in holdout_store if str(payload["domain"]) in set(domains)]
        if not subset:
            continue
        bundle = bundle_map[scope_name]
        result = evaluate_method_from_feature_store(
            method_name=f"current_svdomain_{scope_name}",
            feature_store=subset,
            position_values=tuple(float(v) for v in EARLY_STOP_POSITIONS),
            score_fn=make_svd_bundle_score_fn(bundle),
        )
        metrics = result["aggregate"]
        rows.append(
            {
                "method_name": f"current_svdomain_{scope_name}",
                "metric_scope": "aggregate",
                "domain_scope": str(scope_name),
                "anchor_pct": "",
                "label_fraction_pct": 100,
                "latent_dim": "",
                "auc_of_auroc": float(metrics.get("auc_of_auroc", float("nan"))),
                "auc_of_selacc": float(metrics.get("auc_of_selacc", float("nan"))),
                "earliest_gt_0_6": metrics.get("earliest_gt_0.6"),
                "auroc_at_100": float(metrics.get("auroc@100%", float("nan"))),
                "stop_acc_at_100": float(metrics.get("stop_acc@100%", float("nan"))),
                "auroc": "",
                "selacc_at_10": "",
                "stop_acc": "",
                "pretrain_best_loss": "",
                "pretrain_best_epoch": "",
                "notes": "existing r1 domain bundle",
            }
        )
    ms_subset = [payload for payload in holdout_store if str(payload["domain"]) in {"math", "science"}]
    if ms_subset:
        ms_bundle = load_earlystop_svd_bundle(REPO_ROOT / "models/ml_selectors/es_svd_ms_rr_r1.pkl")
        ms_result = evaluate_method_from_feature_store(
            method_name="current_svdomain_ms",
            feature_store=ms_subset,
            position_values=tuple(float(v) for v in EARLY_STOP_POSITIONS),
            score_fn=make_svd_bundle_score_fn(ms_bundle),
        )
        metrics = ms_result["aggregate"]
        rows.append(
            {
                "method_name": "current_svdomain_ms",
                "metric_scope": "aggregate",
                "domain_scope": "ms",
                "anchor_pct": "",
                "label_fraction_pct": 100,
                "latent_dim": "",
                "auc_of_auroc": float(metrics.get("auc_of_auroc", float("nan"))),
                "auc_of_selacc": float(metrics.get("auc_of_selacc", float("nan"))),
                "earliest_gt_0_6": metrics.get("earliest_gt_0.6"),
                "auroc_at_100": float(metrics.get("auroc@100%", float("nan"))),
                "stop_acc_at_100": float(metrics.get("stop_acc@100%", float("nan"))),
                "auroc": "",
                "selacc_at_10": "",
                "stop_acc": "",
                "pretrain_best_loss": "",
                "pretrain_best_epoch": "",
                "notes": "existing r1 ms bundle",
            }
        )
    pairwise_path = REPO_ROOT / DEFAULT_CURRENT_CODING_PAIRWISE
    if pairwise_path.exists():
        try:
            bundle = load_earlystop_svd_bundle(pairwise_path)
            metrics = _anchor_metric_summary(
                feature_store=[payload for payload in holdout_store if str(payload["domain"]) == "coding"],
                bundle=bundle,
                position=1.00,
                domains=("coding",),
            )
            rows.append(
                {
                    "method_name": "current_coding_pairwise_slot100",
                    "metric_scope": "anchor",
                    "domain_scope": "coding",
                    "anchor_pct": 100,
                    "label_fraction_pct": 100,
                    "latent_dim": "",
                    "auc_of_auroc": "",
                    "auc_of_selacc": "",
                    "earliest_gt_0_6": "",
                    "auroc_at_100": "",
                    "stop_acc_at_100": "",
                    "auroc": float(metrics.get("auroc", float("nan"))),
                    "selacc_at_10": float(metrics.get("selacc@10%", float("nan"))),
                    "stop_acc": float(metrics.get("stop_acc", float("nan"))),
                    "pretrain_best_loss": "",
                    "pretrain_best_epoch": "",
                    "notes": "existing slot100 coding pairwise baseline",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "method_name": "current_coding_pairwise_slot100",
                    "metric_scope": "anchor",
                    "domain_scope": "coding",
                    "anchor_pct": 100,
                    "label_fraction_pct": 100,
                    "latent_dim": "",
                    "auc_of_auroc": "",
                    "auc_of_selacc": "",
                    "earliest_gt_0_6": "",
                    "auroc_at_100": "",
                    "stop_acc_at_100": "",
                    "auroc": "",
                    "selacc_at_10": "",
                    "stop_acc": "",
                    "pretrain_best_loss": "",
                    "pretrain_best_epoch": "",
                    "notes": f"slot100 pairwise baseline skipped: {exc}",
                }
            )


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method_name",
        "metric_scope",
        "domain_scope",
        "anchor_pct",
        "label_fraction_pct",
        "latent_dim",
        "auc_of_auroc",
        "auc_of_selacc",
        "earliest_gt_0_6",
        "auroc_at_100",
        "stop_acc_at_100",
        "auroc",
        "selacc_at_10",
        "stop_acc",
        "pretrain_best_loss",
        "pretrain_best_epoch",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_label_efficiency(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [
        "neural_semisup_freeze",
        "neural_semisup_coding_adapter",
        "neural_semisup_coding_hybrid",
        "no_svd_logreg",
        "frozen_svd_linear_head",
        "shallow_mlp_no_ssl",
    ]
    metric_rows = [
        row for row in rows
        if row["metric_scope"] == "aggregate"
        and row["domain_scope"] == "coding"
        and str(row["method_name"]) in set(methods)
    ]
    if not metric_rows:
        return
    plt.figure(figsize=(7.5, 4.5))
    for method_name in methods:
        points = _best_metric_by_fraction(
            metric_rows,
            method_name=method_name,
            domain_scope="coding",
            metric_scope="aggregate",
            metric_key="auc_of_auroc",
        )
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        if xs:
            plt.plot(xs, ys, marker="o", label=method_name)
    plt.xscale("log")
    plt.xticks([1, 5, 10, 20, 50, 100], ["1", "5", "10", "20", "50", "100"])
    plt.xlabel("Labeled training groups (%)")
    plt.ylabel("Coding AUC of AUROC")
    plt.title("Label efficiency on coding holdout")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "label_efficiency_coding_auc_of_auroc.png", dpi=180)
    plt.close()


def _plot_coding_late_anchors(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    anchor_rows = [
        row for row in rows
        if row["metric_scope"] == "anchor"
        and row["domain_scope"] == "coding"
        and int(row["label_fraction_pct"]) == 100
        and str(row["anchor_pct"]) in {"70", "100"}
    ]
    if not anchor_rows:
        return
    methods = []
    for method_name in [
        "neural_semisup_freeze",
        "neural_semisup_coding_adapter",
        "neural_semisup_coding_hybrid",
        "frozen_svd_linear_head",
        "no_svd_logreg",
        "shallow_mlp_no_ssl",
        "current_coding_pairwise_slot100",
    ]:
        if any(str(row["method_name"]) == method_name for row in anchor_rows):
            methods.append(method_name)
    anchor_labels = ["70", "100"]
    x = np.arange(len(anchor_labels))
    width = 0.14 if methods else 0.25
    plt.figure(figsize=(8.0, 4.8))
    for idx, method_name in enumerate(methods):
        values = []
        for anchor_label in anchor_labels:
            candidates = [
                _safe_float_cell(row["auroc"])
                for row in anchor_rows
                if str(row["method_name"]) == method_name and str(row["anchor_pct"]) == anchor_label
            ]
            finite = [value for value in candidates if np.isfinite(value)]
            values.append(max(finite) if finite else float("nan"))
        plt.bar(x + idx * width - ((len(methods) - 1) * width / 2.0), values, width=width, label=method_name)
    plt.xticks(x, [f"{label}%" for label in anchor_labels])
    plt.ylabel("AUROC")
    plt.title("Coding late-anchor AUROC (100% labels)")
    plt.ylim(bottom=max(0.0, plt.ylim()[0]))
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "coding_late_anchor_auroc.png", dpi=180)
    plt.close()


def _plot_latent_reuse(names: list[str], matrix: np.ndarray, out_dir: Path) -> None:
    if matrix.size <= 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(4.0, 0.55 * len(names)), max(4.0, 0.50 * len(names))))
    im = plt.imshow(matrix, cmap="viridis", vmin=-1.0, vmax=1.0)
    plt.xticks(np.arange(len(names)), names, rotation=45, ha="right", fontsize=8)
    plt.yticks(np.arange(len(names)), names, fontsize=8)
    plt.title("Latent reuse across domains and anchors")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_dir / "latent_reuse_heatmap.png", dpi=180)
    plt.close()


def _plot_decoder_strengths(strengths: dict[str, float], out_dir: Path) -> None:
    if not strengths:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(strengths.items(), key=lambda item: float(item[1]), reverse=True)
    names = [item[0] for item in ordered]
    values = [float(item[1]) for item in ordered]
    plt.figure(figsize=(9.0, 5.0))
    plt.bar(np.arange(len(names)), values)
    plt.xticks(np.arange(len(names)), names, rotation=60, ha="right", fontsize=8)
    plt.ylabel("Decoder latent-weight norm")
    plt.title("Neural decoder feature strength")
    plt.tight_layout()
    plt.savefig(out_dir / "decoder_feature_strength.png", dpi=180)
    plt.close()


def _pick_row(
    rows: list[dict[str, Any]],
    *,
    method_name: str,
    domain_scope: str,
    metric_scope: str,
    label_fraction_pct: int,
    anchor_pct: str | int = "",
) -> Optional[dict[str, Any]]:
    for row in rows:
        if str(row["method_name"]) != str(method_name):
            continue
        if str(row["domain_scope"]) != str(domain_scope):
            continue
        if str(row["metric_scope"]) != str(metric_scope):
            continue
        if int(row["label_fraction_pct"]) != int(label_fraction_pct):
            continue
        if str(row["anchor_pct"]) != str(anchor_pct):
            continue
        return row
    return None


def _pick_best_row(
    rows: list[dict[str, Any]],
    *,
    method_name: str,
    domain_scope: str,
    metric_scope: str,
    label_fraction_pct: int,
    anchor_pct: str | int = "",
    metric_key: str = "auc_of_auroc",
) -> Optional[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if str(row["method_name"]) == str(method_name)
        and str(row["domain_scope"]) == str(domain_scope)
        and str(row["metric_scope"]) == str(metric_scope)
        and int(row["label_fraction_pct"]) == int(label_fraction_pct)
        and str(row["anchor_pct"]) == str(anchor_pct)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: float(row[metric_key]) if row.get(metric_key) not in {"", None} else float("-inf"),
    )


def _write_report(
    *,
    rows: list[dict[str, Any]],
    out_path: Path,
    figure_dir: Path,
    csv_path: Path,
    max_problems_per_cache: int | None,
    latent_reuse_names: list[str],
    latent_reuse_matrix: np.ndarray,
    decoder_strengths: dict[str, float],
) -> None:
    neural_candidates = [
        _pick_best_row(
            rows,
            method_name=method_name,
            domain_scope="coding",
            metric_scope="aggregate",
            label_fraction_pct=100,
        )
        for method_name in ("neural_semisup_freeze", "neural_semisup_coding_adapter")
        + ("neural_semisup_coding_hybrid",)
    ]
    neural_candidates = [row for row in neural_candidates if row is not None]
    neural_coding = (
        max(neural_candidates, key=lambda row: _safe_float_cell(row.get("auc_of_auroc")))
        if neural_candidates
        else None
    )
    current_coding = _pick_best_row(
        rows,
        method_name="current_svdomain_coding",
        domain_scope="coding",
        metric_scope="aggregate",
        label_fraction_pct=100,
    )
    frozen_coding = _pick_best_row(
        rows,
        method_name="frozen_svd_linear_head",
        domain_scope="coding",
        metric_scope="aggregate",
        label_fraction_pct=100,
    )
    mlp_coding = _pick_best_row(
        rows,
        method_name="shallow_mlp_no_ssl",
        domain_scope="coding",
        metric_scope="aggregate",
        label_fraction_pct=100,
    )

    def _delta(lhs: Optional[dict[str, Any]], rhs: Optional[dict[str, Any]], key: str) -> float:
        if lhs is None or rhs is None:
            return float("nan")
        try:
            return float(lhs[key]) - float(rhs[key])
        except Exception:
            return float("nan")

    coding_delta_vs_current = _delta(neural_coding, current_coding, "auc_of_auroc")
    coding_delta_vs_frozen = _delta(neural_coding, frozen_coding, "auc_of_auroc")
    coding_delta_vs_mlp = _delta(neural_coding, mlp_coding, "auc_of_auroc")

    neural_label_curve = [
        point[1]
        for point in _best_metric_by_fraction(
            rows,
            method_name=(
                str(neural_coding["method_name"])
                if neural_coding is not None
                else "neural_semisup_freeze"
            ),
            domain_scope="coding",
            metric_scope="aggregate",
            metric_key="auc_of_auroc",
        )
    ]
    frozen_label_curve = [
        point[1]
        for point in _best_metric_by_fraction(
            rows,
            method_name="frozen_svd_linear_head",
            domain_scope="coding",
            metric_scope="aggregate",
            metric_key="auc_of_auroc",
        )
    ]
    label_eff_delta = _safe_mean(neural_label_curve) - _safe_mean(frozen_label_curve)

    top_decoder = sorted(decoder_strengths.items(), key=lambda item: float(item[1]), reverse=True)[:5]
    smoke_note = (
        f"This is a smoke-scale run (`max_problems_per_cache={max_problems_per_cache}`), so treat all claims as provisional."
        if max_problems_per_cache is not None
        else "This report uses the configured full protocol for the chosen run."
    )

    def _answer_bool(delta: float, threshold: float = 0.005) -> str:
        if not np.isfinite(delta):
            return "Insufficient evidence in the current run."
        if delta > threshold:
            return f"Yes, but modestly (`Δ AUC of AUROC = {delta:+.4f}`)."
        if delta < -threshold:
            return f"No (`Δ AUC of AUROC = {delta:+.4f}`)."
        return f"Not reliably; the difference is small (`Δ AUC of AUROC = {delta:+.4f}`)."

    if np.isfinite(coding_delta_vs_current) and np.isfinite(coding_delta_vs_frozen):
        if coding_delta_vs_current > 0.005 and coding_delta_vs_frozen > 0.005:
            coding_focus = "The gains appear concentrated in coding more than in the existing fixed-SVD baseline."
        elif coding_delta_vs_current <= 0.0 and coding_delta_vs_frozen <= 0.0:
            coding_focus = "The current run does not show a coding-specific gain."
        else:
            coding_focus = "The coding effect is mixed and should be treated as tentative."
    else:
        coding_focus = "Coding-specific effect is inconclusive in the current run."

    if np.isfinite(label_eff_delta):
        label_eff_answer = (
            f"Neural pretraining improves average coding label efficiency versus frozen SVD by `{label_eff_delta:+.4f}` AUC-of-AUROC."
            if label_eff_delta > 0.005
            else (
                f"Neural pretraining does not clearly improve label efficiency (`Δ={label_eff_delta:+.4f}`)."
                if label_eff_delta < -0.005
                else f"Label-efficiency differences are small (`Δ={label_eff_delta:+.4f}`)."
            )
        )
    else:
        label_eff_answer = "Label-efficiency comparison is inconclusive in the current run."

    if top_decoder:
        interp_answer = (
            "Partially yes. The model keeps linear decoders back to the canonical features, "
            "so feature-level saliency remains inspectable, but the latent axes are less directly named than SVD components."
        )
    else:
        interp_answer = "Interpretability was not fully assessed in the current run."

    if np.isfinite(coding_delta_vs_mlp) and np.isfinite(coding_delta_vs_frozen):
        if coding_delta_vs_mlp > 0.005 and coding_delta_vs_frozen > 0.005:
            source_answer = "The current evidence points more toward representation learning than a simple objective swap."
        elif coding_delta_vs_mlp <= 0.0 and coding_delta_vs_frozen > 0.0:
            source_answer = "The gain appears mixed: objective choice matters, but representation learning is not cleanly separated."
        else:
            source_answer = "The current run does not cleanly separate representation learning from objective choice."
    else:
        source_answer = "The gain source is inconclusive in the current run."

    lines = [
        "# Neural Semi-Supervised SVDomain",
        "",
        "## Summary",
        "",
        f"- {smoke_note}",
        f"- Canonical feature family: `{FIXED_FAMILY_NAME}` with `{FIXED_REPRESENTATION}` and anchors `10/40/70/100`.",
        f"- Results table: `{_display_path(csv_path)}`.",
        f"- Figures directory: `{_display_path(figure_dir)}`.",
        "",
        "## Requested Answers",
        "",
        f"1. **Does self-supervised bottleneck learning help beyond fixed SVD?** { _answer_bool(coding_delta_vs_current) }",
        f"2. **Does it mainly help coding or not?** {coding_focus}",
        f"3. **Does it improve label efficiency?** {label_eff_answer}",
        f"4. **Does it preserve enough interpretability to be worth the extra complexity?** {interp_answer}",
        f"5. **Is the gain coming from representation learning, objective change, or both?** {source_answer}",
        "",
        "## Main Comparisons",
        "",
    ]
    for method_name in [
        "neural_semisup_freeze",
        "neural_semisup_coding_adapter",
        "frozen_svd_linear_head",
        "no_svd_logreg",
        "shallow_mlp_no_ssl",
        "current_svdomain_coding",
    ]:
        row = _pick_best_row(
            rows,
            method_name=method_name,
            domain_scope="coding",
            metric_scope="aggregate",
            label_fraction_pct=100,
        )
        if row is None:
            continue
        lines.append(
            "- `{}`: coding `AUC of AUROC={:.4f}`, `AUC of SelAcc={:.4f}`, `AUROC@100%={:.4f}`, `Stop Acc@100%={:.4f}`.".format(
                method_name,
                float(row["auc_of_auroc"]),
                float(row["auc_of_selacc"]),
                float(row["auroc_at_100"]),
                float(row["stop_acc_at_100"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretability Notes",
            "",
            "- The bottleneck remains small (`latent dim ∈ {8,16,24}` in the implemented sweep) and decodes linearly back to the canonical features.",
            "- Top decoder-linked features in the current run: "
            + ", ".join(f"`{name}` ({value:.3f})" for name, value in top_decoder)
            + ".",
        ]
    )
    if latent_reuse_names and latent_reuse_matrix.size > 0:
        lines.extend(
            [
                f"- Latent reuse heatmap covers `{len(latent_reuse_names)}` domain-anchor cells.",
            ]
        )
    lines.extend(
        [
            "",
            "## Cautions",
            "",
            "- The purpose of this experiment is to test whether a neural bottleneck adds value beyond fixed low-rank linear routing, not to overclaim a new mainline.",
            "- Pairwise supervision is intentionally limited to coding late anchors (`70%` / `100%`).",
            "- The current implementation stays fully inside the structured feature space and does not use raw CoT text or hidden states.",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train neural semi-supervised SVDomain extensions")
    ap.add_argument("--main-cache-root", default=DEFAULT_MAIN_CACHE_ROOT)
    ap.add_argument("--extra-cache-root", default=DEFAULT_EXTRA_CACHE_ROOT)
    ap.add_argument("--pretrain-extra-roots", default=DEFAULT_PRETRAIN_EXTRA_ROOTS, help="Comma-separated extra roots for unlabeled pretraining")
    ap.add_argument("--feature-cache-dir", default=DEFAULT_FEATURE_CACHE_DIR)
    ap.add_argument("--main-feature-store-pkl", default="none")
    ap.add_argument("--extra-feature-store-pkl", default="none")
    ap.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    ap.add_argument("--out-doc", default=DEFAULT_OUT_DOC)
    ap.add_argument("--out-fig-dir", default=DEFAULT_OUT_FIG_DIR)
    ap.add_argument("--out-model-dir", default=DEFAULT_OUT_MODEL_DIR)
    ap.add_argument("--methods", default=",".join(DEFAULT_TRAIN_METHODS))
    ap.add_argument("--label-fractions", default="1,5,10,20,50,100")
    ap.add_argument("--latent-dims", default="8,16,24")
    ap.add_argument("--feature-workers", type=int, default=8)
    ap.add_argument("--feature-chunk-problems", type=int, default=24)
    ap.add_argument("--holdout-split", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--pretrain-epochs", type=int, default=DEFAULT_PRETRAIN_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--align-weight", type=float, default=0.25)
    ap.add_argument("--future-weight", type=float, default=0.25)
    ap.add_argument("--consistency-weight", type=float, default=0.10)
    ap.add_argument("--contrastive-weight", type=float, default=0.0)
    ap.add_argument("--max-problems-per-cache", type=int, default=0)
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch-threads", type=int, default=8)
    args = ap.parse_args()

    require_torch()
    import torch

    torch.set_num_threads(max(1, int(args.torch_threads)))

    feature_cache_dir = Path(args.feature_cache_dir)
    if not feature_cache_dir.is_absolute():
        feature_cache_dir = REPO_ROOT / feature_cache_dir
    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = REPO_ROOT / out_csv
    out_doc = Path(args.out_doc)
    if not out_doc.is_absolute():
        out_doc = REPO_ROOT / out_doc
    out_fig_dir = Path(args.out_fig_dir)
    if not out_fig_dir.is_absolute():
        out_fig_dir = REPO_ROOT / out_fig_dir
    out_model_dir = Path(args.out_model_dir)
    if not out_model_dir.is_absolute():
        out_model_dir = REPO_ROOT / out_model_dir

    label_fractions = tuple(int(v) for v in _parse_csv(args.label_fractions, cast=int))
    latent_dims = tuple(int(v) for v in _parse_csv(args.latent_dims, cast=int))
    train_methods = tuple(str(v) for v in _parse_csv(args.methods, cast=str))
    max_problems_per_cache = None if int(args.max_problems_per_cache) <= 0 else int(args.max_problems_per_cache)
    required_feature_names = set(str(name) for name in CANONICAL_FEATURE_NAMES)
    main_feature_store_pkl = None if str(args.main_feature_store_pkl).strip().lower() in {"", "none", "off"} else (REPO_ROOT / str(args.main_feature_store_pkl)).resolve()
    extra_feature_store_pkl = None if str(args.extra_feature_store_pkl).strip().lower() in {"", "none", "off"} else (REPO_ROOT / str(args.extra_feature_store_pkl)).resolve()

    if main_feature_store_pkl is not None:
        main_store = _load_feature_store_from_pickle(main_feature_store_pkl)
    else:
        main_store, _, _ = _load_root_feature_store(
            source_name="cache",
            cache_root=str(args.main_cache_root),
            positions=tuple(float(v) for v in EXTRACTION_POSITIONS),
            required_feature_names=required_feature_names,
            max_problems_per_cache=max_problems_per_cache,
            feature_workers=int(args.feature_workers),
            feature_chunk_problems=int(args.feature_chunk_problems),
            feature_cache_dir=feature_cache_dir,
            refresh_feature_cache=bool(args.refresh_feature_cache),
        )
    if extra_feature_store_pkl is not None:
        extra_store = _load_feature_store_from_pickle(extra_feature_store_pkl)
    else:
        extra_store, _, _ = _load_root_feature_store(
            source_name="cache_train",
            cache_root=str(args.extra_cache_root),
            positions=tuple(float(v) for v in EXTRACTION_POSITIONS),
            required_feature_names=required_feature_names,
            max_problems_per_cache=max_problems_per_cache,
            feature_workers=int(args.feature_workers),
            feature_chunk_problems=int(args.feature_chunk_problems),
            feature_cache_dir=feature_cache_dir,
            refresh_feature_cache=bool(args.refresh_feature_cache),
        )
    full_labeled_store = list(main_store) + list(extra_store)
    if not full_labeled_store:
        raise ValueError("No labeled feature-store payloads were found")

    holdout_problem_map, _ = _build_holdout_problem_map(
        full_labeled_store,
        holdout_split=float(args.holdout_split),
        split_seed=int(args.split_seed),
    )
    train_store, holdout_store, full_store = _split_feature_store(
        full_labeled_store,
        holdout_problem_map=holdout_problem_map,
    )
    if not train_store or not holdout_store:
        raise ValueError("Train/holdout split produced an empty side")

    pretrain_store = list(full_store)
    extra_pretrain_roots = [root.strip() for root in str(args.pretrain_extra_roots).split(",") if root.strip()]
    for idx, root in enumerate(extra_pretrain_roots, start=1):
        additional_store, _, _ = _load_root_feature_store(
            source_name=f"pretrain_extra_{idx}",
            cache_root=str(root),
            positions=tuple(float(v) for v in EXTRACTION_POSITIONS),
            required_feature_names=required_feature_names,
            max_problems_per_cache=max_problems_per_cache,
            feature_workers=int(args.feature_workers),
            feature_chunk_problems=int(args.feature_chunk_problems),
            feature_cache_dir=feature_cache_dir,
            refresh_feature_cache=bool(args.refresh_feature_cache),
        )
        pretrain_store.extend(additional_store)

    pretrain_matrix = build_run_matrix_from_feature_store(
        pretrain_store,
        anchor_positions=ANCHOR_POSITIONS,
        feature_indices=CANONICAL_FEATURE_INDICES,
    )
    train_matrix = build_run_matrix_from_feature_store(
        train_store,
        anchor_positions=ANCHOR_POSITIONS,
        feature_indices=CANONICAL_FEATURE_INDICES,
    )
    holdout_matrix = build_run_matrix_from_feature_store(
        holdout_store,
        anchor_positions=ANCHOR_POSITIONS,
        feature_indices=CANONICAL_FEATURE_INDICES,
    )
    if pretrain_matrix.raw.shape[0] <= 0 or train_matrix.raw.shape[0] <= 0 or holdout_matrix.raw.shape[0] <= 0:
        raise ValueError("One of pretrain/train/holdout matrices is empty")

    rows: list[dict[str, Any]] = []
    _evaluate_current_baselines(holdout_store=holdout_store, rows=rows)

    protocol_core = {
        "main_cache_root": str(args.main_cache_root),
        "extra_cache_root": str(args.extra_cache_root),
        "pretrain_extra_roots": list(extra_pretrain_roots),
        "holdout_split": float(args.holdout_split),
        "split_seed": int(args.split_seed),
        "n_splits": int(args.n_splits),
        "feature_family": FIXED_FAMILY_NAME,
        "representation": FIXED_REPRESENTATION,
        "anchor_positions": [float(v) for v in ANCHOR_POSITIONS],
        "align_weight": float(args.align_weight),
        "future_weight": float(args.future_weight),
        "consistency_weight": float(args.consistency_weight),
        "contrastive_weight": float(args.contrastive_weight),
        "methods": list(train_methods),
        "label_fractions": [int(v) for v in label_fractions],
        "created_at_utc": _now_utc(),
    }

    best_snapshot = None
    best_snapshot_summary = None
    best_snapshot_score = float("-inf")

    first_latent_dim = None if not latent_dims else int(latent_dims[0])
    neural_methods = tuple(
        method_name
        for method_name in train_methods
        if method_name in {
            "neural_semisup_freeze",
            "neural_semisup_coding_adapter",
            "neural_semisup_coding_hybrid",
        }
    )
    for latent_dim in (latent_dims if neural_methods else ()):
        if first_latent_dim is None:
            first_latent_dim = int(latent_dim)
        snapshot, pretrain_summary = pretrain_neural_encoder(
            pretrain_matrix,
            config=NeuralPretrainConfig(
                latent_dim=int(latent_dim),
                batch_size=int(args.batch_size),
                epochs=int(args.pretrain_epochs),
                learning_rate=float(args.learning_rate),
                weight_decay=float(args.weight_decay),
                align_weight=float(args.align_weight),
                future_weight=float(args.future_weight),
                consistency_weight=float(args.consistency_weight),
                contrastive_weight=float(args.contrastive_weight),
                random_state=int(args.random_state),
                device=str(args.device),
            ),
        )
        out_model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "latent_dim": int(latent_dim),
                "summary": dict(pretrain_summary),
                "snapshot": snapshot,
            },
            out_model_dir / f"neural_semisup_latent{int(latent_dim)}.pt",
        )

        for method_name in neural_methods:
            for label_fraction_pct in label_fractions:
                neural_bundle = _train_method_bundle(
                    method_name=str(method_name),
                    train_matrix=train_matrix,
                    label_fraction_pct=int(label_fraction_pct),
                    random_state=int(args.random_state),
                    n_splits=int(args.n_splits),
                    neural_snapshot=snapshot,
                    protocol={
                        **protocol_core,
                        "latent_dim": int(latent_dim),
                        "label_fraction_pct": int(label_fraction_pct),
                        "method_name": str(method_name),
                    },
                )
                _add_aggregate_rows(
                    rows,
                    method_name=str(method_name),
                    bundle=neural_bundle,
                    holdout_store=holdout_store,
                    label_fraction_pct=int(label_fraction_pct),
                    latent_dim=int(latent_dim),
                    pretrain_summary=pretrain_summary,
                    notes=(
                        "self-supervised neural bottleneck + frozen linear heads"
                        if str(method_name) == "neural_semisup_freeze"
                        else "self-supervised neural bottleneck + coding low-rank latent adapter"
                    ),
                )

                if int(label_fraction_pct) == 100:
                    coding_row = _pick_best_row(
                        rows,
                        method_name=str(method_name),
                        domain_scope="coding",
                        metric_scope="aggregate",
                        label_fraction_pct=100,
                    )
                    if coding_row is not None and np.isfinite(float(coding_row["auc_of_auroc"])) and float(coding_row["auc_of_auroc"]) > best_snapshot_score:
                        best_snapshot_score = float(coding_row["auc_of_auroc"])
                        best_snapshot = snapshot
                        best_snapshot_summary = dict(pretrain_summary)

    for baseline_method in ("no_svd_logreg", "frozen_svd_linear_head", "shallow_mlp_no_ssl"):
        if str(baseline_method) not in set(train_methods):
            continue
        for label_fraction_pct in label_fractions:
            bundle = _train_method_bundle(
                method_name=baseline_method,
                train_matrix=train_matrix,
                label_fraction_pct=int(label_fraction_pct),
                random_state=int(args.random_state),
                n_splits=int(args.n_splits),
                neural_snapshot=None,
                protocol={
                    **protocol_core,
                    "latent_dim": first_latent_dim,
                    "label_fraction_pct": int(label_fraction_pct),
                    "method_name": baseline_method,
                },
            )
            _add_aggregate_rows(
                rows,
                method_name=baseline_method,
                bundle=bundle,
                holdout_store=holdout_store,
                label_fraction_pct=int(label_fraction_pct),
                latent_dim=None,
                pretrain_summary=None,
                notes=baseline_method.replace("_", " "),
            )

    _write_csv(rows, out_csv)

    latent_reuse_names: list[str] = []
    latent_reuse_matrix = np.zeros((0, 0), dtype=np.float64)
    decoder_strength = {}
    if best_snapshot is not None:
        latent_reuse_names, latent_reuse_matrix = compute_latent_reuse_matrix(best_snapshot, holdout_matrix)
        decoder_strength = decoder_feature_strengths(best_snapshot)

    _plot_label_efficiency(rows, out_fig_dir)
    _plot_coding_late_anchors(rows, out_fig_dir)
    _plot_latent_reuse(latent_reuse_names, latent_reuse_matrix, out_fig_dir)
    _plot_decoder_strengths(decoder_strength, out_fig_dir)
    _write_report(
        rows=rows,
        out_path=out_doc,
        figure_dir=out_fig_dir,
        csv_path=out_csv,
        max_problems_per_cache=max_problems_per_cache,
        latent_reuse_names=latent_reuse_names,
        latent_reuse_matrix=latent_reuse_matrix,
        decoder_strengths=decoder_strength,
    )
    print(f"Wrote: {_display_path(out_csv)}")
    print(f"Wrote: {_display_path(out_doc)}")
    print(f"Figures: {_display_path(out_fig_dir)}")


if __name__ == "__main__":
    main()
