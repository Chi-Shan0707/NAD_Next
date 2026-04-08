from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.model_selection import GroupKFold

from nad.core.views.reader import CacheReader
from nad.ops.accuracy import load_correctness_map
from nad.ops.earlystop import (
    CacheEntry,
    EARLY_STOP_POSITIONS,
    N_POSITIONS,
    build_problem_groups,
    discover_cache_entries,
    _problem_sort_key,
)
from nad.ops.earlystop_svd import (
    BASELINE_SIGNAL_NAMES,
    FEATURE_FAMILY_MAP,
    FULL_FEATURE_NAMES,
    get_domain,
    extract_earlystop_signals_for_sample,
)
from nad.core.selectors.math_svm_impl import (
    MathLinearSVMScorer,
    MathRankSVMScorer,
    build_math_pairwise_hinge_training_examples,
)


@dataclass(frozen=True)
class SVMEarlyStopConfig:
    n_splits: int = 3
    family_names: tuple[str, ...] = ("token_only",)
    representations: tuple[str, ...] = ("raw", "rank", "raw+rank")
    c_values: tuple[float, ...] = (0.1, 1.0, 3.0, 10.0)
    losses: tuple[str, ...] = ("hinge", "squared_hinge")
    ranksvm_backends: tuple[str, ...] = ("utility", "mean_margin", "win_count")
    class_weight_options: tuple[str, ...] = ("none", "balanced")
    random_state: int = 42
    max_problems_per_cache: int = 0


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    try:
        from scipy.stats import mannwhitneyu

        stat, _ = mannwhitneyu(scores[pos], scores[neg], alternative="greater")
        return float(stat) / float(n_pos * n_neg)
    except Exception:
        ranks = np.argsort(np.argsort(scores)) + 1
        u = float(ranks[pos].sum()) - n_pos * (n_pos + 1) / 2.0
        return u / float(n_pos * n_neg)


def _rank_transform_matrix(x: np.ndarray) -> np.ndarray:
    n, m = x.shape
    out = np.zeros_like(x, dtype=np.float64)
    if n <= 1:
        return out
    for col_i in range(m):
        col = x[:, col_i]
        order = np.argsort(col, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        out[:, col_i] = ranks / float(n - 1)
    return out


def _group_folds(groups: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return []
    splits = min(int(n_splits), int(len(unique_groups)))
    if splits < 2:
        return []
    gkf = GroupKFold(n_splits=splits)
    dummy_x = np.zeros((len(groups), 1), dtype=np.float64)
    return list(gkf.split(dummy_x, groups=groups))


def _build_representation(
    x_raw: np.ndarray,
    x_rank: np.ndarray,
    feature_indices: list[int],
    representation: str,
) -> np.ndarray:
    if representation == "raw":
        return x_raw[:, feature_indices]
    if representation == "rank":
        return x_rank[:, feature_indices]
    if representation == "raw+rank":
        return np.concatenate([
            x_raw[:, feature_indices],
            x_rank[:, feature_indices],
        ], axis=1)
    raise ValueError(f"Unknown representation: {representation}")


def _parse_meta_groups(entry: CacheEntry) -> list[tuple[str, list[int]]]:
    meta = json.loads((entry.cache_root / "meta.json").read_text(encoding="utf-8"))
    groups = build_problem_groups(meta)
    return sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0]))


def _build_domain_training_tables(
    cache_root: str | Path,
    required_feature_names: Optional[set[str]] = None,
    max_problems_per_cache: Optional[int] = None,
) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    entries = discover_cache_entries(cache_root)
    req = set(FULL_FEATURE_NAMES) if required_feature_names is None else set(required_feature_names)

    rows: dict[str, dict[int, list[np.ndarray]]] = {
        "math": {i: [] for i in range(N_POSITIONS)},
        "science": {i: [] for i in range(N_POSITIONS)},
        "coding": {i: [] for i in range(N_POSITIONS)},
    }
    labels: dict[str, dict[int, list[int]]] = {
        "math": {i: [] for i in range(N_POSITIONS)},
        "science": {i: [] for i in range(N_POSITIONS)},
        "coding": {i: [] for i in range(N_POSITIONS)},
    }
    group_keys: dict[str, dict[int, list[str]]] = {
        "math": {i: [] for i in range(N_POSITIONS)},
        "science": {i: [] for i in range(N_POSITIONS)},
        "coding": {i: [] for i in range(N_POSITIONS)},
    }

    for entry in entries:
        domain = get_domain(entry.dataset_name)
        correctness = load_correctness_map(str(entry.cache_root))
        reader = CacheReader(str(entry.cache_root))

        for problem_idx, (problem_id, sample_ids) in enumerate(_parse_meta_groups(entry)):
            if max_problems_per_cache is not None and problem_idx >= int(max_problems_per_cache):
                break
            group_id = f"{entry.cache_key}::{problem_id}"
            for sample_id in sample_ids:
                y = int(bool(correctness.get(int(sample_id), False)))
                signal_map = extract_earlystop_signals_for_sample(
                    reader,
                    int(sample_id),
                    required_features=req,
                )
                sample_mat = np.zeros((N_POSITIONS, len(FULL_FEATURE_NAMES)), dtype=np.float64)
                for f_i, f_name in enumerate(FULL_FEATURE_NAMES):
                    sample_mat[:, f_i] = np.asarray(signal_map[f_name], dtype=np.float64)
                for pos_i in range(N_POSITIONS):
                    rows[domain][pos_i].append(sample_mat[pos_i])
                    labels[domain][pos_i].append(y)
                    group_keys[domain][pos_i].append(group_id)

    out: dict[str, dict[int, dict[str, np.ndarray]]] = {
        "math": {},
        "science": {},
        "coding": {},
    }
    for domain in out.keys():
        for pos_i in range(N_POSITIONS):
            if rows[domain][pos_i]:
                x_raw = np.vstack(rows[domain][pos_i]).astype(np.float64, copy=False)
                y = np.asarray(labels[domain][pos_i], dtype=np.int32)
                groups = np.asarray(group_keys[domain][pos_i], dtype=object)
            else:
                x_raw = np.zeros((0, len(FULL_FEATURE_NAMES)), dtype=np.float64)
                y = np.zeros((0,), dtype=np.int32)
                groups = np.asarray([], dtype=object)

            x_rank = np.zeros_like(x_raw)
            if x_raw.shape[0] > 0:
                by_group: dict[Any, list[int]] = {}
                for idx, g in enumerate(groups.tolist()):
                    by_group.setdefault(g, []).append(idx)
                for idxs in by_group.values():
                    x_rank[idxs] = _rank_transform_matrix(x_raw[idxs])

            out[domain][pos_i] = {
                "x_raw": x_raw,
                "x_rank": x_rank,
                "y": y,
                "groups": groups,
            }
    return out


def _cv_auroc_baseline(
    scores: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> tuple[float, int]:
    folds = _group_folds(groups, n_splits=n_splits)
    if not folds:
        return float("nan"), 0
    vals: list[float] = []
    for _, test_idx in folds:
        y_test = y[test_idx]
        if np.unique(y_test).shape[0] < 2:
            continue
        v = _auroc(scores[test_idx], y_test)
        if np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), len(vals)


def _cv_auroc_pointwise(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    c_value: float,
    loss: str,
    class_weight: str,
) -> tuple[float, int]:
    folds = _group_folds(groups, n_splits=n_splits)
    if not folds:
        return float("nan"), 0
    vals: list[float] = []
    for train_idx, test_idx in folds:
        y_train = y[train_idx]
        y_test = y[test_idx]
        if np.unique(y_train).shape[0] < 2 or np.unique(y_test).shape[0] < 2:
            continue
        scorer = MathLinearSVMScorer(
            C=float(c_value),
            loss=str(loss),
            fit_intercept=True,
            class_weight=None if class_weight == "none" else class_weight,
            dual="auto",
            max_iter=100000,
            tol=1e-4,
        )
        try:
            scorer.fit(x[train_idx], y_train)
        except Exception:
            continue
        scores = scorer.score_group(x[test_idx])
        v = _auroc(scores, y_test)
        if np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), len(vals)


def _build_pairs_from_groups(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    by_group: dict[Any, list[int]] = {}
    for idx, g in enumerate(groups.tolist()):
        by_group.setdefault(g, []).append(idx)

    pair_x_parts: list[np.ndarray] = []
    pair_y_parts: list[np.ndarray] = []
    for idxs in by_group.values():
        x_g = x[idxs]
        y_g = y[idxs]
        px, py = build_math_pairwise_hinge_training_examples(x_g, y_g)
        if px.shape[0] > 0:
            pair_x_parts.append(px)
            pair_y_parts.append(py)
    if not pair_x_parts:
        return np.zeros((0, x.shape[1]), dtype=np.float64), np.zeros((0,), dtype=np.int32)
    return (
        np.concatenate(pair_x_parts, axis=0).astype(np.float64, copy=False),
        np.concatenate(pair_y_parts, axis=0).astype(np.int32, copy=False),
    )


def _cv_auroc_ranksvm(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    c_value: float,
    loss: str,
    backend: str,
) -> tuple[float, int]:
    folds = _group_folds(groups, n_splits=n_splits)
    if not folds:
        return float("nan"), 0
    vals: list[float] = []
    for train_idx, test_idx in folds:
        y_train = y[train_idx]
        y_test = y[test_idx]
        g_train = groups[train_idx]
        g_test = groups[test_idx]
        if np.unique(y_train).shape[0] < 2 or np.unique(y_test).shape[0] < 2:
            continue

        pair_x, pair_y = _build_pairs_from_groups(x[train_idx], y_train, g_train)
        if pair_x.shape[0] <= 0:
            continue

        fit_intercept = backend != "utility"
        scorer = MathRankSVMScorer(
            C=float(c_value),
            loss=str(loss),
            fit_intercept=fit_intercept,
            backend=str(backend),
            dual="auto",
            max_iter=100000,
            tol=1e-4,
        )
        try:
            scorer.fit(pair_x, pair_y)
        except Exception:
            continue

        by_group_test: dict[Any, list[int]] = {}
        for local_idx, g in enumerate(g_test.tolist()):
            by_group_test.setdefault(g, []).append(local_idx)

        y_collect: list[np.ndarray] = []
        s_collect: list[np.ndarray] = []
        for idxs in by_group_test.values():
            if len(idxs) <= 1:
                continue
            y_g = y_test[idxs]
            if np.unique(y_g).shape[0] < 2:
                continue
            s_g = scorer.score_group(x[test_idx][idxs])
            y_collect.append(np.asarray(y_g, dtype=np.int32))
            s_collect.append(np.asarray(s_g, dtype=np.float64))

        if not s_collect:
            continue
        y_all = np.concatenate(y_collect, axis=0)
        s_all = np.concatenate(s_collect, axis=0)
        v = _auroc(s_all, y_all)
        if np.isfinite(v):
            vals.append(float(v))

    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), len(vals)


def train_earlystop_svm_bundle(
    cache_root: str | Path,
    config: SVMEarlyStopConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    required_feature_names: set[str] = set()
    for family_name in config.family_names:
        required_feature_names.update(FEATURE_FAMILY_MAP[family_name])
    baseline_candidates = [s for s in BASELINE_SIGNAL_NAMES if s in required_feature_names]
    if not baseline_candidates:
        baseline_candidates = ["tok_conf_prefix"]
        required_feature_names.add("tok_conf_prefix")

    max_problems_per_cache = None
    if int(config.max_problems_per_cache) > 0:
        max_problems_per_cache = int(config.max_problems_per_cache)

    tables = _build_domain_training_tables(
        cache_root,
        required_feature_names=required_feature_names,
        max_problems_per_cache=max_problems_per_cache,
    )
    feature_to_idx = {name: i for i, name in enumerate(FULL_FEATURE_NAMES)}

    bundle: dict[str, Any] = {
        "bundle_version": "earlystop_svm_hybrid_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_names": list(FULL_FEATURE_NAMES),
        "positions": list(EARLY_STOP_POSITIONS),
        "domains": {},
    }
    summary: dict[str, Any] = {
        "bundle_version": "earlystop_svm_hybrid_v1",
        "created_at_utc": bundle["created_at_utc"],
        "cache_root": str(cache_root),
        "config": {
            "n_splits": int(config.n_splits),
            "families": list(config.family_names),
            "representations": list(config.representations),
            "c_values": list(config.c_values),
            "losses": list(config.losses),
            "ranksvm_backends": list(config.ranksvm_backends),
            "class_weight_options": list(config.class_weight_options),
            "random_state": int(config.random_state),
            "max_problems_per_cache": int(config.max_problems_per_cache),
        },
        "domains": {},
    }

    route_counts = {"pointwise": 0, "ranksvm": 0, "baseline": 0}
    for domain in ("math", "science", "coding"):
        domain_bundle: dict[str, Any] = {"routes": []}
        domain_summary: dict[str, Any] = {"positions": []}

        for pos_i in range(N_POSITIONS):
            tbl = tables[domain][pos_i]
            x_raw = tbl["x_raw"]
            x_rank = tbl["x_rank"]
            y = tbl["y"]
            groups = tbl["groups"]

            pos_desc = f"{domain}@{EARLY_STOP_POSITIONS[pos_i]:.1f}"
            print(f"[train-svm] {pos_desc} samples={x_raw.shape[0]} groups={len(np.unique(groups))}")

            if x_raw.shape[0] == 0 or np.unique(y).shape[0] < 2:
                route = {
                    "route_type": "baseline",
                    "signal_name": "tok_conf_prefix",
                    "cv_auroc": float("nan"),
                    "note": "insufficient labeled data",
                }
                domain_bundle["routes"].append(route)
                domain_summary["positions"].append({
                    "position": float(EARLY_STOP_POSITIONS[pos_i]),
                    "route": route,
                })
                route_counts["baseline"] += 1
                continue

            best_baseline = {"signal_name": None, "cv_auroc": float("-inf"), "n_valid_folds": 0}
            for signal_name in baseline_candidates:
                score_col = feature_to_idx[signal_name]
                score_vec = x_raw[:, score_col]
                cv_auc, n_folds = _cv_auroc_baseline(score_vec, y, groups, config.n_splits)
                if np.isfinite(cv_auc) and cv_auc > float(best_baseline["cv_auroc"]):
                    best_baseline = {
                        "signal_name": signal_name,
                        "cv_auroc": float(cv_auc),
                        "n_valid_folds": int(n_folds),
                    }

            best_pointwise: dict[str, Any] = {"cv_auroc": float("-inf")}
            best_ranksvm: dict[str, Any] = {"cv_auroc": float("-inf")}

            for family_name in config.family_names:
                feat_indices = [feature_to_idx[n] for n in FEATURE_FAMILY_MAP[family_name]]
                for rep in config.representations:
                    x_rep = _build_representation(x_raw, x_rank, feat_indices, rep)

                    for c_value in config.c_values:
                        for loss in config.losses:
                            for class_weight in config.class_weight_options:
                                cv_auc, n_folds = _cv_auroc_pointwise(
                                    x_rep, y, groups, config.n_splits, c_value, loss, class_weight
                                )
                                if np.isfinite(cv_auc) and cv_auc > float(best_pointwise["cv_auroc"]):
                                    best_pointwise = {
                                        "cv_auroc": float(cv_auc),
                                        "n_valid_folds": int(n_folds),
                                        "family_name": family_name,
                                        "representation": rep,
                                        "c_value": float(c_value),
                                        "loss": str(loss),
                                        "class_weight": str(class_weight),
                                        "feature_indices": list(feat_indices),
                                        "feature_names": list(FEATURE_FAMILY_MAP[family_name]),
                                    }
                            for backend in config.ranksvm_backends:
                                cv_auc, n_folds = _cv_auroc_ranksvm(
                                    x_rep, y, groups, config.n_splits, c_value, loss, backend
                                )
                                if np.isfinite(cv_auc) and cv_auc > float(best_ranksvm["cv_auroc"]):
                                    best_ranksvm = {
                                        "cv_auroc": float(cv_auc),
                                        "n_valid_folds": int(n_folds),
                                        "family_name": family_name,
                                        "representation": rep,
                                        "c_value": float(c_value),
                                        "loss": str(loss),
                                        "backend": str(backend),
                                        "feature_indices": list(feat_indices),
                                        "feature_names": list(FEATURE_FAMILY_MAP[family_name]),
                                    }

            candidates = [
                ("baseline", float(best_baseline["cv_auroc"])),
                ("pointwise", float(best_pointwise["cv_auroc"]) if np.isfinite(best_pointwise["cv_auroc"]) else float("-inf")),
                ("ranksvm", float(best_ranksvm["cv_auroc"]) if np.isfinite(best_ranksvm["cv_auroc"]) else float("-inf")),
            ]
            winner = max(candidates, key=lambda kv: kv[1])[0]

            if winner == "pointwise":
                x_rep = _build_representation(
                    x_raw, x_rank, best_pointwise["feature_indices"], best_pointwise["representation"]
                )
                scorer = MathLinearSVMScorer(
                    C=float(best_pointwise["c_value"]),
                    loss=str(best_pointwise["loss"]),
                    fit_intercept=True,
                    class_weight=None if best_pointwise["class_weight"] == "none" else best_pointwise["class_weight"],
                    dual="auto",
                    max_iter=100000,
                    tol=1e-4,
                )
                scorer.fit(x_rep, y)
                route = {
                    "route_type": "pointwise",
                    "cv_auroc": float(best_pointwise["cv_auroc"]),
                    "n_valid_folds": int(best_pointwise["n_valid_folds"]),
                    "family_name": str(best_pointwise["family_name"]),
                    "representation": str(best_pointwise["representation"]),
                    "c_value": float(best_pointwise["c_value"]),
                    "loss": str(best_pointwise["loss"]),
                    "class_weight": str(best_pointwise["class_weight"]),
                    "feature_indices": list(best_pointwise["feature_indices"]),
                    "feature_names": list(best_pointwise["feature_names"]),
                    "baseline_signal_name": best_baseline["signal_name"],
                    "baseline_cv_auroc": float(best_baseline["cv_auroc"]),
                    "scorer": scorer,
                }
                route_counts["pointwise"] += 1
            elif winner == "ranksvm":
                x_rep = _build_representation(
                    x_raw, x_rank, best_ranksvm["feature_indices"], best_ranksvm["representation"]
                )
                pair_x, pair_y = _build_pairs_from_groups(x_rep, y, groups)
                scorer = MathRankSVMScorer(
                    C=float(best_ranksvm["c_value"]),
                    loss=str(best_ranksvm["loss"]),
                    fit_intercept=(best_ranksvm["backend"] != "utility"),
                    backend=str(best_ranksvm["backend"]),
                    dual="auto",
                    max_iter=100000,
                    tol=1e-4,
                )
                scorer.fit(pair_x, pair_y)
                route = {
                    "route_type": "ranksvm",
                    "cv_auroc": float(best_ranksvm["cv_auroc"]),
                    "n_valid_folds": int(best_ranksvm["n_valid_folds"]),
                    "family_name": str(best_ranksvm["family_name"]),
                    "representation": str(best_ranksvm["representation"]),
                    "c_value": float(best_ranksvm["c_value"]),
                    "loss": str(best_ranksvm["loss"]),
                    "backend": str(best_ranksvm["backend"]),
                    "feature_indices": list(best_ranksvm["feature_indices"]),
                    "feature_names": list(best_ranksvm["feature_names"]),
                    "baseline_signal_name": best_baseline["signal_name"],
                    "baseline_cv_auroc": float(best_baseline["cv_auroc"]),
                    "scorer": scorer,
                }
                route_counts["ranksvm"] += 1
            else:
                route = {
                    "route_type": "baseline",
                    "signal_name": best_baseline["signal_name"],
                    "cv_auroc": float(best_baseline["cv_auroc"]),
                    "n_valid_folds": int(best_baseline["n_valid_folds"]),
                    "pointwise_best_cv_auroc": None if not np.isfinite(best_pointwise["cv_auroc"]) else float(best_pointwise["cv_auroc"]),
                    "ranksvm_best_cv_auroc": None if not np.isfinite(best_ranksvm["cv_auroc"]) else float(best_ranksvm["cv_auroc"]),
                }
                route_counts["baseline"] += 1

            domain_bundle["routes"].append(route)
            domain_summary["positions"].append({
                "position": float(EARLY_STOP_POSITIONS[pos_i]),
                "route": {k: v for k, v in route.items() if k != "scorer"},
                "num_samples": int(x_raw.shape[0]),
                "num_groups": int(len(np.unique(groups))),
                "positive_rate": float(np.mean(y)) if y.size else float("nan"),
            })

            if route["route_type"] == "baseline":
                print(f"  -> baseline keep auc={route['cv_auroc']:.4f} signal={route['signal_name']}")
            elif route["route_type"] == "pointwise":
                print(f"  -> pointwise SVM win auc={route['cv_auroc']:.4f}")
            else:
                print(f"  -> rankSVM win auc={route['cv_auroc']:.4f} backend={route['backend']}")

        bundle["domains"][domain] = domain_bundle
        summary["domains"][domain] = domain_summary

    summary["totals"] = {
        "pointwise_slots": int(route_counts["pointwise"]),
        "ranksvm_slots": int(route_counts["ranksvm"]),
        "baseline_slots": int(route_counts["baseline"]),
        "total_slots": int(route_counts["pointwise"] + route_counts["ranksvm"] + route_counts["baseline"]),
    }
    return bundle, summary


def save_earlystop_svm_bundle(bundle: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(bundle, f)


def load_earlystop_svm_bundle(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("rb") as f:
        return pickle.load(f)


def _problem_tensor(
    reader: CacheReader,
    sample_ids: list[int],
    required_feature_names: Optional[set[str]] = None,
) -> np.ndarray:
    n_runs = len(sample_ids)
    tensor = np.zeros((n_runs, N_POSITIONS, len(FULL_FEATURE_NAMES)), dtype=np.float64)
    for row_i, sample_id in enumerate(sample_ids):
        signal_map = extract_earlystop_signals_for_sample(
            reader,
            int(sample_id),
            required_features=required_feature_names,
        )
        for f_i, f_name in enumerate(FULL_FEATURE_NAMES):
            tensor[row_i, :, f_i] = np.asarray(signal_map[f_name], dtype=np.float64)
    return tensor


def score_cache_entry_earlystop_svm(
    entry: CacheEntry,
    bundle: dict[str, Any],
    max_problems: int | None = None,
) -> dict[str, dict[str, list[float]]]:
    domain = get_domain(entry.dataset_name)
    domain_bundle = bundle["domains"][domain]
    feature_to_idx = {name: i for i, name in enumerate(bundle["feature_names"])}

    required_features: set[str] = set()
    for route in domain_bundle["routes"]:
        if route["route_type"] == "baseline":
            required_features.add(str(route["signal_name"]))
        else:
            required_features.update(str(v) for v in route["feature_names"])

    reader = CacheReader(str(entry.cache_root))
    groups = _parse_meta_groups(entry)

    out: dict[str, dict[str, list[float]]] = {}
    for problem_i, (problem_id, sample_ids_raw) in enumerate(groups):
        if max_problems is not None and problem_i >= max_problems:
            break
        sample_ids = [int(sid) for sid in sample_ids_raw]
        tensor = _problem_tensor(reader, sample_ids, required_feature_names=required_features)
        run_scores = {str(sid): [0.0] * N_POSITIONS for sid in sample_ids}

        for pos_i in range(N_POSITIONS):
            route = domain_bundle["routes"][pos_i]
            x_raw = tensor[:, pos_i, :]
            x_rank = _rank_transform_matrix(x_raw)

            if route["route_type"] == "baseline":
                score_col = feature_to_idx[str(route["signal_name"])]
                scores = x_raw[:, score_col]
            else:
                feat_indices = [int(v) for v in route["feature_indices"]]
                rep = str(route["representation"])
                x_rep = _build_representation(x_raw, x_rank, feat_indices, rep)
                scores = route["scorer"].score_group(x_rep)

            for run_idx, sample_id in enumerate(sample_ids):
                run_scores[str(sample_id)][pos_i] = float(scores[run_idx])

        out[str(problem_id)] = run_scores
    return out


def _fit_domain_route_for_position(
    *,
    x_raw: np.ndarray,
    x_rank: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_to_idx: dict[str, int],
    baseline_candidates: list[str],
    config: SVMEarlyStopConfig,
) -> dict[str, Any]:
    best_baseline = {"signal_name": None, "cv_auroc": float("-inf"), "n_valid_folds": 0}
    for signal_name in baseline_candidates:
        score_col = feature_to_idx[signal_name]
        score_vec = x_raw[:, score_col]
        cv_auc, n_folds = _cv_auroc_baseline(score_vec, y, groups, config.n_splits)
        if np.isfinite(cv_auc) and cv_auc > float(best_baseline["cv_auroc"]):
            best_baseline = {
                "signal_name": signal_name,
                "cv_auroc": float(cv_auc),
                "n_valid_folds": int(n_folds),
            }

    best_pointwise: dict[str, Any] = {"cv_auroc": float("-inf")}
    best_ranksvm: dict[str, Any] = {"cv_auroc": float("-inf")}

    for family_name in config.family_names:
        feat_indices = [feature_to_idx[n] for n in FEATURE_FAMILY_MAP[family_name]]
        for rep in config.representations:
            x_rep = _build_representation(x_raw, x_rank, feat_indices, rep)
            for c_value in config.c_values:
                for loss in config.losses:
                    for class_weight in config.class_weight_options:
                        cv_auc, n_folds = _cv_auroc_pointwise(
                            x_rep, y, groups, config.n_splits, c_value, loss, class_weight
                        )
                        if np.isfinite(cv_auc) and cv_auc > float(best_pointwise["cv_auroc"]):
                            best_pointwise = {
                                "cv_auroc": float(cv_auc),
                                "n_valid_folds": int(n_folds),
                                "family_name": family_name,
                                "representation": rep,
                                "c_value": float(c_value),
                                "loss": str(loss),
                                "class_weight": str(class_weight),
                                "feature_indices": list(feat_indices),
                                "feature_names": list(FEATURE_FAMILY_MAP[family_name]),
                            }
                    for backend in config.ranksvm_backends:
                        cv_auc, n_folds = _cv_auroc_ranksvm(
                            x_rep, y, groups, config.n_splits, c_value, loss, backend
                        )
                        if np.isfinite(cv_auc) and cv_auc > float(best_ranksvm["cv_auroc"]):
                            best_ranksvm = {
                                "cv_auroc": float(cv_auc),
                                "n_valid_folds": int(n_folds),
                                "family_name": family_name,
                                "representation": rep,
                                "c_value": float(c_value),
                                "loss": str(loss),
                                "backend": str(backend),
                                "feature_indices": list(feat_indices),
                                "feature_names": list(FEATURE_FAMILY_MAP[family_name]),
                            }

    candidates = [
        ("baseline", float(best_baseline["cv_auroc"])),
        ("pointwise", float(best_pointwise["cv_auroc"]) if np.isfinite(best_pointwise["cv_auroc"]) else float("-inf")),
        ("ranksvm", float(best_ranksvm["cv_auroc"]) if np.isfinite(best_ranksvm["cv_auroc"]) else float("-inf")),
    ]
    winner = max(candidates, key=lambda kv: kv[1])[0]

    if winner == "pointwise":
        x_rep = _build_representation(
            x_raw, x_rank, best_pointwise["feature_indices"], best_pointwise["representation"]
        )
        scorer = MathLinearSVMScorer(
            C=float(best_pointwise["c_value"]),
            loss=str(best_pointwise["loss"]),
            fit_intercept=True,
            class_weight=None if best_pointwise["class_weight"] == "none" else best_pointwise["class_weight"],
            dual="auto",
            max_iter=100000,
            tol=1e-4,
        )
        scorer.fit(x_rep, y)
        return {
            "route_type": "pointwise",
            "cv_auroc": float(best_pointwise["cv_auroc"]),
            "n_valid_folds": int(best_pointwise["n_valid_folds"]),
            "family_name": str(best_pointwise["family_name"]),
            "representation": str(best_pointwise["representation"]),
            "c_value": float(best_pointwise["c_value"]),
            "loss": str(best_pointwise["loss"]),
            "class_weight": str(best_pointwise["class_weight"]),
            "feature_indices": list(best_pointwise["feature_indices"]),
            "feature_names": list(best_pointwise["feature_names"]),
            "baseline_signal_name": best_baseline["signal_name"],
            "baseline_cv_auroc": float(best_baseline["cv_auroc"]),
            "scorer": scorer,
        }

    if winner == "ranksvm":
        x_rep = _build_representation(
            x_raw, x_rank, best_ranksvm["feature_indices"], best_ranksvm["representation"]
        )
        pair_x, pair_y = _build_pairs_from_groups(x_rep, y, groups)
        scorer = MathRankSVMScorer(
            C=float(best_ranksvm["c_value"]),
            loss=str(best_ranksvm["loss"]),
            fit_intercept=(best_ranksvm["backend"] != "utility"),
            backend=str(best_ranksvm["backend"]),
            dual="auto",
            max_iter=100000,
            tol=1e-4,
        )
        scorer.fit(pair_x, pair_y)
        return {
            "route_type": "ranksvm",
            "cv_auroc": float(best_ranksvm["cv_auroc"]),
            "n_valid_folds": int(best_ranksvm["n_valid_folds"]),
            "family_name": str(best_ranksvm["family_name"]),
            "representation": str(best_ranksvm["representation"]),
            "c_value": float(best_ranksvm["c_value"]),
            "loss": str(best_ranksvm["loss"]),
            "backend": str(best_ranksvm["backend"]),
            "feature_indices": list(best_ranksvm["feature_indices"]),
            "feature_names": list(best_ranksvm["feature_names"]),
            "baseline_signal_name": best_baseline["signal_name"],
            "baseline_cv_auroc": float(best_baseline["cv_auroc"]),
            "scorer": scorer,
        }

    return {
        "route_type": "baseline",
        "signal_name": best_baseline["signal_name"],
        "cv_auroc": float(best_baseline["cv_auroc"]),
        "n_valid_folds": int(best_baseline["n_valid_folds"]),
        "pointwise_best_cv_auroc": None if not np.isfinite(best_pointwise["cv_auroc"]) else float(best_pointwise["cv_auroc"]),
        "ranksvm_best_cv_auroc": None if not np.isfinite(best_ranksvm["cv_auroc"]) else float(best_ranksvm["cv_auroc"]),
    }


def train_bestofn_svm_bundle(
    cache_root: str | Path,
    config: SVMEarlyStopConfig,
    position_index: int = (N_POSITIONS - 1),
) -> tuple[dict[str, Any], dict[str, Any]]:
    if int(position_index) < 0 or int(position_index) >= N_POSITIONS:
        raise ValueError(f"position_index out of range: {position_index}")

    required_feature_names: set[str] = set()
    for family_name in config.family_names:
        required_feature_names.update(FEATURE_FAMILY_MAP[family_name])
    baseline_candidates = [s for s in BASELINE_SIGNAL_NAMES if s in required_feature_names]
    if not baseline_candidates:
        baseline_candidates = ["tok_conf_prefix"]
        required_feature_names.add("tok_conf_prefix")

    max_problems_per_cache = None
    if int(config.max_problems_per_cache) > 0:
        max_problems_per_cache = int(config.max_problems_per_cache)

    tables = _build_domain_training_tables(
        cache_root,
        required_feature_names=required_feature_names,
        max_problems_per_cache=max_problems_per_cache,
    )
    feature_to_idx = {name: i for i, name in enumerate(FULL_FEATURE_NAMES)}

    bundle: dict[str, Any] = {
        "bundle_version": "bestofn_svm_bridge_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_names": list(FULL_FEATURE_NAMES),
        "position_index": int(position_index),
        "position_value": float(EARLY_STOP_POSITIONS[int(position_index)]),
        "domains": {},
    }
    summary: dict[str, Any] = {
        "bundle_version": "bestofn_svm_bridge_v1",
        "created_at_utc": bundle["created_at_utc"],
        "cache_root": str(cache_root),
        "position_index": int(position_index),
        "position_value": float(EARLY_STOP_POSITIONS[int(position_index)]),
        "config": {
            "n_splits": int(config.n_splits),
            "families": list(config.family_names),
            "representations": list(config.representations),
            "c_values": list(config.c_values),
            "losses": list(config.losses),
            "ranksvm_backends": list(config.ranksvm_backends),
            "class_weight_options": list(config.class_weight_options),
            "random_state": int(config.random_state),
            "max_problems_per_cache": int(config.max_problems_per_cache),
        },
        "domains": {},
    }

    route_counts = {"pointwise": 0, "ranksvm": 0, "baseline": 0}
    for domain in ("math", "science", "coding"):
        tbl = tables[domain][int(position_index)]
        x_raw = tbl["x_raw"]
        x_rank = tbl["x_rank"]
        y = tbl["y"]
        groups = tbl["groups"]
        if x_raw.shape[0] == 0 or np.unique(y).shape[0] < 2:
            route = {
                "route_type": "baseline",
                "signal_name": "tok_conf_prefix",
                "cv_auroc": float("nan"),
                "note": "insufficient labeled data",
            }
            route_counts["baseline"] += 1
        else:
            route = _fit_domain_route_for_position(
                x_raw=x_raw,
                x_rank=x_rank,
                y=y,
                groups=groups,
                feature_to_idx=feature_to_idx,
                baseline_candidates=baseline_candidates,
                config=config,
            )
            route_counts[str(route["route_type"])] += 1

        bundle["domains"][domain] = {"route": route}
        summary["domains"][domain] = {
            "route": {k: v for k, v in route.items() if k != "scorer"},
            "num_samples": int(x_raw.shape[0]),
            "num_groups": int(len(np.unique(groups))),
            "positive_rate": float(np.mean(y)) if y.size else float("nan"),
        }

        if route["route_type"] == "baseline":
            print(f"[bestofn-train] {domain}: baseline keep auc={route['cv_auroc']:.4f} signal={route['signal_name']}")
        elif route["route_type"] == "pointwise":
            print(f"[bestofn-train] {domain}: pointwise SVM win auc={route['cv_auroc']:.4f}")
        else:
            print(f"[bestofn-train] {domain}: rankSVM win auc={route['cv_auroc']:.4f} backend={route['backend']}")

    summary["totals"] = {
        "pointwise_slots": int(route_counts["pointwise"]),
        "ranksvm_slots": int(route_counts["ranksvm"]),
        "baseline_slots": int(route_counts["baseline"]),
        "total_slots": int(route_counts["pointwise"] + route_counts["ranksvm"] + route_counts["baseline"]),
    }
    return bundle, summary


def _score_problem_with_route(
    x_raw: np.ndarray,
    route: dict[str, Any],
    feature_to_idx: dict[str, int],
) -> np.ndarray:
    x_rank = _rank_transform_matrix(x_raw)
    if route["route_type"] == "baseline":
        score_col = feature_to_idx[str(route["signal_name"])]
        return np.asarray(x_raw[:, score_col], dtype=np.float64)
    feat_indices = [int(v) for v in route["feature_indices"]]
    rep = str(route["representation"])
    x_rep = _build_representation(x_raw, x_rank, feat_indices, rep)
    return np.asarray(route["scorer"].score_group(x_rep), dtype=np.float64)


def score_cache_entry_bestofn_svm(
    entry: CacheEntry,
    bundle: dict[str, Any],
    max_problems: int | None = None,
    rank_scale_1_100: bool = True,
) -> dict[str, dict[str, float]]:
    domain = get_domain(entry.dataset_name)
    route = bundle["domains"][domain]["route"]
    feature_to_idx = {name: i for i, name in enumerate(bundle["feature_names"])}
    pos_i = int(bundle["position_index"])

    required_features: set[str] = set()
    if route["route_type"] == "baseline":
        required_features.add(str(route["signal_name"]))
    else:
        required_features.update(str(v) for v in route["feature_names"])

    reader = CacheReader(str(entry.cache_root))
    groups = _parse_meta_groups(entry)
    scores_out: dict[str, dict[str, float]] = {}

    for problem_i, (problem_id, sample_ids_raw) in enumerate(groups):
        if max_problems is not None and problem_i >= max_problems:
            break
        sample_ids = [int(sid) for sid in sample_ids_raw]
        tensor = _problem_tensor(reader, sample_ids, required_feature_names=required_features)
        x_raw = tensor[:, pos_i, :]
        raw_scores = _score_problem_with_route(x_raw, route, feature_to_idx)

        final_scores = np.asarray(raw_scores, dtype=np.float64)
        if rank_scale_1_100 and final_scores.size > 1:
            order = np.argsort(final_scores, kind="mergesort")
            ranks = np.empty(final_scores.size, dtype=np.float64)
            ranks[order] = np.arange(final_scores.size, dtype=np.float64)
            final_scores = 1.0 + (ranks * 99.0 / float(final_scores.size - 1))

        scores_out[str(problem_id)] = {
            str(sample_id): float(final_scores[idx])
            for idx, sample_id in enumerate(sample_ids)
        }
    return scores_out


def score_cache_entry_earlystop_from_bestofn_svm(
    entry: CacheEntry,
    bundle: dict[str, Any],
    max_problems: int | None = None,
) -> dict[str, dict[str, list[float]]]:
    domain = get_domain(entry.dataset_name)
    route = bundle["domains"][domain]["route"]
    feature_to_idx = {name: i for i, name in enumerate(bundle["feature_names"])}

    required_features: set[str] = set()
    if route["route_type"] == "baseline":
        required_features.add(str(route["signal_name"]))
    else:
        required_features.update(str(v) for v in route["feature_names"])

    reader = CacheReader(str(entry.cache_root))
    groups = _parse_meta_groups(entry)
    out: dict[str, dict[str, list[float]]] = {}

    for problem_i, (problem_id, sample_ids_raw) in enumerate(groups):
        if max_problems is not None and problem_i >= max_problems:
            break
        sample_ids = [int(sid) for sid in sample_ids_raw]
        tensor = _problem_tensor(reader, sample_ids, required_feature_names=required_features)
        run_scores = {str(sid): [0.0] * N_POSITIONS for sid in sample_ids}

        for pos_i in range(N_POSITIONS):
            x_raw = tensor[:, pos_i, :]
            scores = _score_problem_with_route(x_raw, route, feature_to_idx)
            for idx, sid in enumerate(sample_ids):
                run_scores[str(sid)][pos_i] = float(scores[idx])
        out[str(problem_id)] = run_scores
    return out
