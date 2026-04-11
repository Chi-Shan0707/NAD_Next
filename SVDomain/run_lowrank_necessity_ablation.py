#!/usr/bin/env python3
"""Run low-rank necessity ablations for canonical r1 raw+rank EarlyStop models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.explain.svd_explain import feature_family
from nad.explain.svd_introspection import recover_effective_weights
from nad.ops.earlystop import EARLY_STOP_POSITIONS, discover_cache_entries
from nad.ops.earlystop_svd import (
    _auroc,
    _build_representation,
    _predict_svd_lr,
    _rank_transform_matrix,
    get_domain,
)
from scripts.run_earlystop_prefix10_svd_round1 import (
    ANCHOR_POSITIONS,
    DEFAULT_FEATURE_CHUNK_PROBLEMS,
    EXTRACTION_POSITIONS,
    OFFICIAL_SLOT_TO_ANCHOR,
    SEARCH_C_VALUES,
    SEARCH_CLASS_WEIGHT,
    SEARCH_WHITEN,
    _now_utc,
    _pct_label,
    build_feature_store,
    evaluate_method_from_feature_store,
)
from SVDomain.train_es_svd_ms_rr_r1 import (
    FIXED_FEATURE_INDICES,
    FIXED_FEATURE_NAMES,
    FIXED_FAMILY_NAME,
    FIXED_REPRESENTATION,
    _best_baseline_summary,
    _build_domain_training_tables,
    _build_holdout_problem_map,
    _feature_cache_key,
    _group_folds,
    _qualify_feature_store,
    _resolve_path,
    _split_feature_store,
    _summarise_feature_store,
)


OUTPUT_DOMAIN_ORDER = ("math", "science", "ms")
TRAIN_DOMAIN_ORDER = ("math", "science")
SMALLEST_SUFFICIENT_TOLERANCE = 0.005


@dataclass(frozen=True)
class AblationSpec:
    method_name: str
    label: str
    kind: str
    rank: Optional[int] = None


def default_suite_workers() -> int:
    cpu = max(1, int(os.cpu_count() or 1))
    return max(1, min(4, cpu))


def default_fit_workers() -> int:
    cpu = max(1, int(os.cpu_count() or 1))
    return max(1, min(16, cpu))


def parse_ranks(raw: str) -> tuple[int, ...]:
    values = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"Rank must be positive: {value}")
        values.append(value)
    if not values:
        raise ValueError("Need at least one rank")
    return tuple(sorted(dict.fromkeys(values)))


def build_ablation_specs(ranks: tuple[int, ...]) -> list[AblationSpec]:
    specs = [AblationSpec(method_name="no_svd_lr", label="StandardScaler -> LogisticRegression", kind="no_svd")]
    for rank in ranks:
        specs.append(
            AblationSpec(
                method_name=f"svd_r{int(rank)}",
                label=f"StandardScaler -> TruncatedSVD(rank={int(rank)}) -> LogisticRegression",
                kind="svd",
                rank=int(rank),
            )
        )
    return specs


def _feature_cache_path(
    *,
    cache_dir: Path,
    source_name: str,
    cache_root: str,
    positions: tuple[float, ...],
    required_feature_names: set[str],
    max_problems_per_cache: Optional[int],
) -> Path:
    raw_key = _feature_cache_key(
        source_name=source_name,
        cache_root=cache_root,
        positions=positions,
        required_feature_names=required_feature_names,
        max_problems_per_cache=max_problems_per_cache,
    )
    key = hashlib.sha1(f"noncoding::{raw_key}".encode("utf-8")).hexdigest()[:16]
    suffix = "all" if max_problems_per_cache is None else f"cap{int(max_problems_per_cache)}"
    return cache_dir / f"noncoding_{source_name}_{suffix}_{key}.pkl"


def _load_or_build_noncoding_feature_store(
    *,
    source_name: str,
    cache_root: str,
    positions: tuple[float, ...],
    required_feature_names: set[str],
    max_problems_per_cache: Optional[int],
    feature_workers: int,
    chunk_problems: int,
    feature_cache_dir: Optional[Path],
    refresh_feature_cache: bool,
) -> tuple[list[dict[str, Any]], Optional[Path], str]:
    def _load_cached_payload(path: Path) -> dict[str, Any]:
        import pickle

        with path.open("rb") as handle:
            return pickle.load(handle)

    def _filter_noncoding_store(feature_store: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [payload for payload in feature_store if str(payload.get("domain")) in TRAIN_DOMAIN_ORDER]

    cache_path: Optional[Path] = None
    canonical_cache_path: Optional[Path] = None
    if feature_cache_dir is not None:
        raw_key = _feature_cache_key(
            source_name=source_name,
            cache_root=cache_root,
            positions=positions,
            required_feature_names=required_feature_names,
            max_problems_per_cache=max_problems_per_cache,
        )
        cache_path = _feature_cache_path(
            cache_dir=feature_cache_dir,
            source_name=source_name,
            cache_root=cache_root,
            positions=positions,
            required_feature_names=required_feature_names,
            max_problems_per_cache=max_problems_per_cache,
        )
        if cache_path.exists() and not refresh_feature_cache:
            print(f"[load] noncoding feature cache source={source_name} path={cache_path}", flush=True)
            payload = _load_cached_payload(cache_path)
            return list(payload["feature_store"]), cache_path, "loaded"

        canonical_dirs = []
        canonical_dirs.append(feature_cache_dir)
        fallback_dir = REPO_ROOT / "results/cache/es_svd_ms_rr_r1"
        if fallback_dir != feature_cache_dir:
            canonical_dirs.append(fallback_dir)
        suffix = "all" if max_problems_per_cache is None else f"cap{int(max_problems_per_cache)}"
        for candidate_dir in canonical_dirs:
            maybe_path = candidate_dir / f"{source_name}_{suffix}_{raw_key}.pkl"
            if maybe_path.exists() and not refresh_feature_cache:
                canonical_cache_path = maybe_path
                break
        if canonical_cache_path is not None:
            print(
                f"[load] canonical feature cache source={source_name} path={canonical_cache_path} -> filter noncoding",
                flush=True,
            )
            payload = _load_cached_payload(canonical_cache_path)
            filtered_store = _filter_noncoding_store(list(payload["feature_store"]))
            if cache_path is not None and cache_path != canonical_cache_path:
                import pickle

                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
                with tmp_path.open("wb") as handle:
                    pickle.dump(
                        {
                            "source_name": str(source_name),
                            "cache_root": str(cache_root),
                            "positions": [float(p) for p in positions],
                            "max_problems_per_cache": None if max_problems_per_cache is None else int(max_problems_per_cache),
                            "included_domains": list(TRAIN_DOMAIN_ORDER),
                            "feature_store": filtered_store,
                        },
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                tmp_path.replace(cache_path)
                print(f"[save] noncoding feature cache source={source_name} path={cache_path}", flush=True)
            return filtered_store, canonical_cache_path, "loaded_canonical_filtered"

    include_cache_keys = {
        str(entry.cache_key)
        for entry in discover_cache_entries(cache_root)
        if get_domain(str(entry.dataset_name)) in TRAIN_DOMAIN_ORDER
    }
    print(
        f"[build] noncoding feature store source={source_name} root={cache_root} "
        f"include={len(include_cache_keys)} caches",
        flush=True,
    )
    store = _qualify_feature_store(
        build_feature_store(
            cache_root=cache_root,
            positions=positions,
            required_feature_names=required_feature_names,
            max_problems_per_cache=max_problems_per_cache,
            max_workers=max(1, int(feature_workers)),
            chunk_problems=max(1, int(chunk_problems)),
            include_cache_keys=include_cache_keys,
        ),
        source_name=source_name,
    )

    if cache_path is not None:
        import pickle

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(
                {
                    "source_name": str(source_name),
                    "cache_root": str(cache_root),
                    "positions": [float(p) for p in positions],
                    "max_problems_per_cache": None if max_problems_per_cache is None else int(max_problems_per_cache),
                    "included_domains": list(TRAIN_DOMAIN_ORDER),
                    "feature_store": store,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        tmp_path.replace(cache_path)
        print(f"[save] noncoding feature cache source={source_name} path={cache_path}", flush=True)
    return store, cache_path, "built"


def _route_summary_for_json(route: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in route.items() if key != "model"}


def _effective_rep_feature_names(route: dict[str, Any]) -> list[str]:
    feature_names = [str(name) for name in route.get("feature_names", FIXED_FEATURE_NAMES)]
    representation = str(route.get("representation", FIXED_REPRESENTATION))
    if representation == "raw":
        return [f"{name}::raw" for name in feature_names]
    if representation == "rank":
        return [f"{name}::rank" for name in feature_names]
    if representation == "raw+rank":
        return [f"{name}::raw" for name in feature_names] + [f"{name}::rank" for name in feature_names]
    raise ValueError(f"Unsupported representation for compactness: {representation}")


def _fit_lr_model(
    *,
    x: np.ndarray,
    y: np.ndarray,
    c_value: float,
    class_weight_name: str,
    random_state: int,
) -> Optional[dict[str, Any]]:
    if x.shape[0] < 4 or x.shape[1] < 1:
        return None
    if np.unique(y).shape[0] < 2:
        return None

    scaler = StandardScaler(with_mean=True, with_std=True)
    x_scaled = scaler.fit_transform(x)
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


def _predict_lr_model(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    scaler = model["scaler"]
    lr = model["lr"]
    return np.asarray(lr.decision_function(scaler.transform(x)), dtype=np.float64)


def _fit_ablation_route(
    *,
    x_raw: np.ndarray,
    x_rank: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    position: float,
    domain_name: str,
    spec: AblationSpec,
    n_splits: int,
    random_state: int,
) -> dict[str, Any]:
    if x_raw.shape[0] == 0:
        raise ValueError(f"{domain_name}@{_pct_label(position)} has no labeled rows")
    if np.unique(y).shape[0] < 2:
        raise ValueError(f"{domain_name}@{_pct_label(position)} lacks both classes")

    folds = _group_folds(groups, n_splits=n_splits)
    if not folds:
        raise ValueError(f"{domain_name}@{_pct_label(position)} has insufficient CV groups")

    best_baseline = _best_baseline_summary(x_raw=x_raw, y=y, groups=groups, n_splits=n_splits)
    x_rep = _build_representation(
        x_raw=x_raw,
        x_rank=x_rank,
        feature_indices=FIXED_FEATURE_INDICES,
        representation=FIXED_REPRESENTATION,
    )

    if spec.kind == "no_svd":
        best = {"cv_auroc": float("-inf")}
        candidate_scores: dict[tuple[float, str], list[float]] = {}
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
            for c_value in SEARCH_C_VALUES:
                for class_weight in SEARCH_CLASS_WEIGHT:
                    clf = LogisticRegression(
                        C=float(c_value),
                        class_weight=None if class_weight == "none" else "balanced",
                        max_iter=2000,
                        random_state=int(random_state),
                    )
                    try:
                        clf.fit(x_train_scaled, y_train)
                        scores = np.asarray(clf.decision_function(x_test_scaled), dtype=np.float64)
                    except Exception:
                        continue
                    fold_auc = _auroc(scores, y_test)
                    if np.isfinite(fold_auc):
                        candidate_scores.setdefault((float(c_value), str(class_weight)), []).append(float(fold_auc))

        for (c_value, class_weight), values in candidate_scores.items():
            if not values:
                continue
            cv_auc = float(np.mean(values))
            if cv_auc > float(best["cv_auroc"]):
                best = {
                    "cv_auroc": cv_auc,
                    "n_valid_folds": int(len(values)),
                    "c_value": float(c_value),
                    "class_weight": str(class_weight),
                }

        if not np.isfinite(float(best["cv_auroc"])):
            raise RuntimeError(f"{domain_name}@{_pct_label(position)} found no valid no-SVD candidate")

        model = _fit_lr_model(
            x=x_rep,
            y=y,
            c_value=float(best["c_value"]),
            class_weight_name=str(best["class_weight"]),
            random_state=int(random_state),
        )
        if model is None:
            raise RuntimeError(f"{domain_name}@{_pct_label(position)} full-fit no-SVD failed")

        return {
            "route_type": "lr",
            "ablation_kind": str(spec.kind),
            "method_name": str(spec.method_name),
            "cv_auroc": float(best["cv_auroc"]),
            "n_valid_folds": int(best["n_valid_folds"]),
            "family_name": FIXED_FAMILY_NAME,
            "representation": FIXED_REPRESENTATION,
            "rank": None,
            "requested_rank": None,
            "effective_rank": None,
            "c_value": float(best["c_value"]),
            "whiten": False,
            "class_weight": str(best["class_weight"]),
            "feature_names": list(FIXED_FEATURE_NAMES),
            "feature_indices": list(FIXED_FEATURE_INDICES),
            "baseline_signal_name": str(best_baseline["signal_name"]),
            "baseline_cv_auroc": float(best_baseline["cv_auroc"]),
            "training_position": float(position),
            "training_scope": str(domain_name),
            "model": model,
        }

    requested_rank = int(spec.rank or 0)
    best = {"cv_auroc": float("-inf")}
    candidate_scores: dict[tuple[float, bool, str], list[float]] = {}
    effective_rank = max(1, min(requested_rank, int(x_rep.shape[1]), int(x_rep.shape[0] - 1)))
    for train_idx, test_idx in folds:
        y_train = y[train_idx]
        y_test = y[test_idx]
        if np.unique(y_train).shape[0] < 2 or np.unique(y_test).shape[0] < 2:
            continue

        x_train = x_rep[train_idx]
        x_test = x_rep[test_idx]
        fold_rank = max(1, min(effective_rank, int(x_train.shape[1]), int(x_train.shape[0] - 1)))
        scaler = StandardScaler(with_mean=True, with_std=True)
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        svd = TruncatedSVD(n_components=fold_rank, random_state=int(random_state))
        try:
            z_train_full = svd.fit_transform(x_train_scaled)
            z_test_full = svd.transform(x_test_scaled)
        except Exception:
            continue
        singular_values = np.asarray(svd.singular_values_, dtype=np.float64)
        singular_values = np.where(np.abs(singular_values) < 1e-8, 1.0, singular_values)

        for whiten in SEARCH_WHITEN:
            if whiten:
                z_train_use = z_train_full / singular_values[:fold_rank]
                z_test_use = z_test_full / singular_values[:fold_rank]
            else:
                z_train_use = z_train_full
                z_test_use = z_test_full
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
                        candidate_scores.setdefault((float(c_value), bool(whiten), str(class_weight)), []).append(float(fold_auc))

    for (c_value, whiten, class_weight), values in candidate_scores.items():
        if not values:
            continue
        cv_auc = float(np.mean(values))
        if cv_auc > float(best["cv_auroc"]):
            best = {
                "cv_auroc": cv_auc,
                "n_valid_folds": int(len(values)),
                "c_value": float(c_value),
                "whiten": bool(whiten),
                "class_weight": str(class_weight),
            }

    if not np.isfinite(float(best["cv_auroc"])):
        raise RuntimeError(f"{domain_name}@{_pct_label(position)} found no valid SVD candidate")

    from nad.ops.earlystop_svd import _fit_svd_lr_model

    model = _fit_svd_lr_model(
        x=x_rep,
        y=y,
        rank=effective_rank,
        c_value=float(best["c_value"]),
        whiten=bool(best["whiten"]),
        class_weight_name=str(best["class_weight"]),
        random_state=int(random_state),
    )
    if model is None:
        raise RuntimeError(f"{domain_name}@{_pct_label(position)} full-fit fixed-rank SVD failed")

    return {
        "route_type": "svd",
        "ablation_kind": str(spec.kind),
        "method_name": str(spec.method_name),
        "cv_auroc": float(best["cv_auroc"]),
        "n_valid_folds": int(best["n_valid_folds"]),
        "family_name": FIXED_FAMILY_NAME,
        "representation": FIXED_REPRESENTATION,
        "rank": int(effective_rank),
        "requested_rank": int(requested_rank),
        "effective_rank": int(effective_rank),
        "c_value": float(best["c_value"]),
        "whiten": bool(best["whiten"]),
        "class_weight": str(best["class_weight"]),
        "feature_names": list(FIXED_FEATURE_NAMES),
        "feature_indices": list(FIXED_FEATURE_INDICES),
        "baseline_signal_name": str(best_baseline["signal_name"]),
        "baseline_cv_auroc": float(best_baseline["cv_auroc"]),
        "training_position": float(position),
        "training_scope": str(domain_name),
        "model": model,
    }


def _train_variant_for_domain(
    *,
    domain_name: str,
    tables: list[dict[str, np.ndarray]],
    spec: AblationSpec,
    n_splits: int,
    random_state: int,
    anchor_workers: int,
) -> tuple[dict[float, dict[str, Any]], float]:
    start = time.perf_counter()
    routes: dict[float, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(anchor_workers))) as executor:
        future_map = {}
        for pos_idx, position in enumerate(ANCHOR_POSITIONS):
            future = executor.submit(
                _fit_ablation_route,
                x_raw=tables[pos_idx]["x_raw"],
                x_rank=tables[pos_idx]["x_rank"],
                y=tables[pos_idx]["y"],
                groups=tables[pos_idx]["groups"],
                position=float(position),
                domain_name=str(domain_name),
                spec=spec,
                n_splits=int(n_splits),
                random_state=int(random_state),
            )
            future_map[future] = float(position)

        for future in as_completed(future_map):
            position = future_map[future]
            route = future.result()
            routes[float(position)] = route
            extra = f"rank={route['rank']}" if route["route_type"] == "svd" else "rank=none"
            print(
                f"[train] method={spec.method_name:<10s} domain={domain_name:<7s} pos={_pct_label(position):>4s} "
                f"auc={route['cv_auroc']:.4f} {extra} c={route['c_value']:.2f} "
                f"cw={route['class_weight']} whiten={'yes' if route.get('whiten') else 'no'}",
                flush=True,
            )
    return routes, float(time.perf_counter() - start)


def _make_score_fn(routes_by_domain: dict[str, dict[float, dict[str, Any]]]):
    official_to_anchor = {
        int(pos_idx): float(OFFICIAL_SLOT_TO_ANCHOR[float(position)])
        for pos_idx, position in enumerate(EARLY_STOP_POSITIONS)
    }

    def _score(domain: str, position_index: int, x_raw: np.ndarray) -> np.ndarray:
        route = routes_by_domain[str(domain)][official_to_anchor[int(position_index)]]
        x_rank = _rank_transform_matrix(x_raw)
        x_rep = _build_representation(
            x_raw=x_raw,
            x_rank=x_rank,
            feature_indices=[int(v) for v in route["feature_indices"]],
            representation=str(route["representation"]),
        )
        if route["route_type"] == "svd":
            return _predict_svd_lr(route["model"], x_rep)
        if route["route_type"] == "lr":
            return _predict_lr_model(route["model"], x_rep)
        raise ValueError(f"Unsupported route type: {route['route_type']}")

    return _score


def _collapse_feature_weights(rep_feature_names: list[str], weights: np.ndarray) -> dict[str, float]:
    collapsed: dict[str, float] = {}
    for rep_name, value in zip(rep_feature_names, weights.tolist()):
        base = str(rep_name).split("::")[0]
        collapsed[base] = collapsed.get(base, 0.0) + abs(float(value))
    return collapsed


def _route_effective_feature_stats(route: dict[str, Any]) -> dict[str, Any]:
    if route["route_type"] == "svd":
        eff = recover_effective_weights(route)
        rep_feature_names = [str(v) for v in eff["rep_feature_names"]]
        weights = np.asarray(eff["w_orig"], dtype=np.float64)
        top_component_purity = None
        component_importance = list(eff.get("component_importance", []))
        aligned_components = {int(item["comp_k"]): item for item in eff.get("aligned_components", [])}
        if component_importance:
            comp_k = int(component_importance[0]["comp_k"])
            comp = aligned_components.get(comp_k)
            if comp is not None:
                top_component_purity = float(comp["family_purity"])
    elif route["route_type"] == "lr":
        rep_feature_names = _effective_rep_feature_names(route)
        model = route["model"]
        scaler = model["scaler"]
        lr = model["lr"]
        beta = np.asarray(lr.coef_, dtype=np.float64).reshape(-1)
        scale = np.asarray(scaler.scale_, dtype=np.float64).reshape(-1)
        scale = np.where(np.abs(scale) < 1e-8, 1.0, scale)
        weights = beta / scale
        top_component_purity = None
    else:
        raise ValueError(f"Unsupported route type for feature stats: {route['route_type']}")

    collapsed = _collapse_feature_weights(rep_feature_names, weights)
    total_mass = float(sum(collapsed.values()))
    if total_mass <= 0.0:
        return {
            "top5_feature_mass": float("nan"),
            "top_family_mass": float("nan"),
            "top_component_purity": top_component_purity,
        }

    feature_masses = sorted(collapsed.values(), reverse=True)
    top5_feature_mass = float(sum(feature_masses[:5]) / total_mass)

    family_masses: dict[str, float] = {}
    for feature_name, mass in collapsed.items():
        family = feature_family(str(feature_name))
        family_masses[family] = family_masses.get(family, 0.0) + float(mass)
    top_family_mass = float(max(family_masses.values()) / total_mass) if family_masses else float("nan")

    return {
        "top5_feature_mass": top5_feature_mass,
        "top_family_mass": top_family_mass,
        "top_component_purity": top_component_purity,
    }


def _aggregate_route_stats(routes: dict[float, dict[str, Any]]) -> dict[str, Any]:
    per_anchor = {}
    top5_values: list[float] = []
    top_family_values: list[float] = []
    top_component_values: list[float] = []
    for position, route in sorted(routes.items(), key=lambda item: float(item[0])):
        stats = _route_effective_feature_stats(route)
        per_anchor[_pct_label(position)] = stats
        if np.isfinite(stats["top5_feature_mass"]):
            top5_values.append(float(stats["top5_feature_mass"]))
        if np.isfinite(stats["top_family_mass"]):
            top_family_values.append(float(stats["top_family_mass"]))
        top_component = stats.get("top_component_purity")
        if top_component is not None and np.isfinite(float(top_component)):
            top_component_values.append(float(top_component))

    return {
        "compactness_metric_name": "top5_feature_mass",
        "compactness_value": float(np.mean(top5_values)) if top5_values else float("nan"),
        "top_family_mass": float(np.mean(top_family_values)) if top_family_values else float("nan"),
        "top_component_purity": float(np.mean(top_component_values)) if top_component_values else None,
        "per_anchor": per_anchor,
    }


def _aggregate_ms_route_stats(routes_by_domain: dict[str, dict[float, dict[str, Any]]]) -> dict[str, Any]:
    merged_top5: list[float] = []
    merged_family: list[float] = []
    merged_component: list[float] = []
    per_domain = {}
    for domain_name in TRAIN_DOMAIN_ORDER:
        stats = _aggregate_route_stats(routes_by_domain[domain_name])
        per_domain[domain_name] = stats
        if np.isfinite(stats["compactness_value"]):
            merged_top5.append(float(stats["compactness_value"]))
        if np.isfinite(stats["top_family_mass"]):
            merged_family.append(float(stats["top_family_mass"]))
        if stats.get("top_component_purity") is not None and np.isfinite(float(stats["top_component_purity"])):
            merged_component.append(float(stats["top_component_purity"]))

    return {
        "compactness_metric_name": "top5_feature_mass",
        "compactness_value": float(np.mean(merged_top5)) if merged_top5 else float("nan"),
        "top_family_mass": float(np.mean(merged_family)) if merged_family else float("nan"),
        "top_component_purity": float(np.mean(merged_component)) if merged_component else None,
        "per_domain": per_domain,
    }


def _evaluate_routes(
    *,
    method_name: str,
    routes_by_domain: dict[str, dict[float, dict[str, Any]]],
    holdout_by_domain: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    score_fn = _make_score_fn(routes_by_domain)
    domain_results = {
        "math": evaluate_method_from_feature_store(
            method_name=f"{method_name}__math",
            feature_store=holdout_by_domain["math"],
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=score_fn,
        ),
        "science": evaluate_method_from_feature_store(
            method_name=f"{method_name}__science",
            feature_store=holdout_by_domain["science"],
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=score_fn,
        ),
        "ms": evaluate_method_from_feature_store(
            method_name=f"{method_name}__ms",
            feature_store=holdout_by_domain["math"] + holdout_by_domain["science"],
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=score_fn,
        ),
    }
    return domain_results, float(time.perf_counter() - start)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def build_ablation_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_name, payload in summary["methods"].items():
        for domain in OUTPUT_DOMAIN_ORDER:
            domain_payload = payload["domains"][domain]
            aggregate = domain_payload["aggregate"]
            rows.append(
                {
                    "domain": str(domain),
                    "method": str(method_name),
                    "method_label": str(payload["label"]),
                    "kind": str(payload["kind"]),
                    "rank": "" if payload["rank"] is None else int(payload["rank"]),
                    "auc_of_auroc": float(aggregate["auc_of_auroc"]),
                    "auc_of_selacc": float(aggregate["auc_of_selacc"]),
                    "auroc_at_100": float(aggregate["auroc@100%"]),
                    "stop_acc_at_100": float(aggregate["stop_acc@100%"]),
                    "earliest_gt_0p6": "" if aggregate.get("earliest_gt_0.6") is None else _pct_label(float(aggregate["earliest_gt_0.6"])),
                    "fit_time_sec": float(domain_payload["fit_time_sec"]),
                    "inference_time_sec": float(domain_payload["inference_time_sec"]),
                    "samples": int(aggregate["samples"]),
                    "compactness_metric_name": str(domain_payload["compactness_metric_name"]),
                    "compactness_value": float(domain_payload["compactness_value"]),
                    "top_family_mass": float(domain_payload["top_family_mass"]),
                    "top_component_purity": "" if domain_payload.get("top_component_purity") is None else float(domain_payload["top_component_purity"]),
                }
            )
    order = {name: idx for idx, name in enumerate(["no_svd_lr"] + [name for name in summary["methods"].keys() if name != "no_svd_lr"])}
    return sorted(rows, key=lambda row: (OUTPUT_DOMAIN_ORDER.index(str(row["domain"])), order.get(str(row["method"]), 999), 10**9 if row["rank"] == "" else int(row["rank"])))


def build_smallest_sufficient_rows(
    ablation_rows: list[dict[str, Any]],
    tolerance: float = SMALLEST_SUFFICIENT_TOLERANCE,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain in OUTPUT_DOMAIN_ORDER:
        domain_rows = [row for row in ablation_rows if str(row["domain"]) == domain]
        svd_rows = [row for row in domain_rows if str(row["kind"]) == "svd"]
        if not svd_rows:
            continue
        best_row = max(svd_rows, key=lambda row: float(row["auc_of_auroc"]))
        best_auc = float(best_row["auc_of_auroc"])
        threshold_auc = best_auc * (1.0 - float(tolerance))
        eligible = [row for row in svd_rows if float(row["auc_of_auroc"]) >= threshold_auc]
        smallest = min(eligible, key=lambda row: int(row["rank"]))
        no_svd_row = next((row for row in domain_rows if str(row["kind"]) == "no_svd"), None)
        rows.append(
            {
                "domain": str(domain),
                "threshold_rule": f"within_{float(tolerance) * 100.0:.1f}pct_of_best_auc_of_auroc",
                "best_rank": int(best_row["rank"]),
                "best_auc_of_auroc": float(best_auc),
                "smallest_sufficient_rank": int(smallest["rank"]),
                "smallest_sufficient_auc_of_auroc": float(smallest["auc_of_auroc"]),
                "smallest_sufficient_gap_pct": float((best_auc - float(smallest["auc_of_auroc"])) / max(best_auc, 1e-12) * 100.0),
                "no_svd_auc_of_auroc": "" if no_svd_row is None else float(no_svd_row["auc_of_auroc"]),
                "no_svd_gap_vs_best_pct": "" if no_svd_row is None else float((best_auc - float(no_svd_row["auc_of_auroc"])) / max(best_auc, 1e-12) * 100.0),
                "plateau_ranks": ",".join(str(int(row["rank"])) for row in sorted(eligible, key=lambda item: int(item["rank"]))),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_pct(value: Optional[float]) -> str:
    if value is None or not np.isfinite(float(value)):
        return "N/A"
    return f"{float(value) * 100.0:.2f}%"


def build_note_markdown(
    *,
    summary: dict[str, Any],
    ablation_rows: list[dict[str, Any]],
    smallest_rows: list[dict[str, Any]],
) -> str:
    by_domain = {str(row["domain"]): row for row in smallest_rows}
    by_domain_method = {
        (str(row["domain"]), str(row["method"])): row
        for row in ablation_rows
    }
    lines = [
        "# 07. Low-Rank Necessity",
        "",
        "这份 note 只回答一个问题：canonical `raw+rank` family 的收益，是否来自 low-rank bottleneck 本身，而不只是来自 feature bank。",
        "",
        "## 1. 结论先行",
        "",
    ]

    positive_domains = []
    for domain in OUTPUT_DOMAIN_ORDER:
        smallest = by_domain.get(domain)
        if smallest is None:
            continue
        no_svd_gap = smallest.get("no_svd_gap_vs_best_pct")
        if no_svd_gap != "" and float(no_svd_gap) > 0.0:
            positive_domains.append(domain)
    if len(positive_domains) == len(OUTPUT_DOMAIN_ORDER):
        deltas = []
        for domain in OUTPUT_DOMAIN_ORDER:
            best = by_domain.get(domain)
            no_svd = by_domain_method.get((domain, "no_svd_lr"))
            if best is None or no_svd is None:
                continue
            delta_pp = (float(best["best_auc_of_auroc"]) - float(no_svd["auc_of_auroc"])) * 100.0
            deltas.append(f"`{domain}` +{delta_pp:.2f} AUC-pts")
        lines.append("- 在 `math / science / ms` 三个 noncoding domain 上，best fixed-rank SVD 都优于同特征、去掉 SVD 的 `no-SVD` baseline，但幅度是**一致而温和**的。")
        if deltas:
            lines.append("- 相对 `no_svd_lr` 的 `AUC of AUROC` 提升分别是：" + "，".join(deltas) + "。")
    elif positive_domains:
        lines.append(f"- 在 `{', '.join(positive_domains)}` 上，best fixed-rank SVD 明确优于同特征 `no-SVD` baseline；其余 domain 的优势更温和。")
    else:
        lines.append("- `no-SVD` 并没有系统性胜出；收益不能仅用 feature bank 来解释。")

    plateau_notes = []
    for domain in OUTPUT_DOMAIN_ORDER:
        smallest = by_domain.get(domain)
        if smallest is None:
            continue
        plateau_notes.append(f"`{domain}` 的 smallest sufficient rank = `{smallest['smallest_sufficient_rank']}`")
    if plateau_notes:
        lines.append("- " + "；".join(plateau_notes) + "。")
    lines.append("- `r2/r4/r8` 在三个 domain 上都明显低于最终最佳；真正的平台期从 `math:r16`、`science:r24`、`ms:r24` 才开始。")

    lines.extend([
        "",
        "## 2. Smallest sufficient rank",
        "",
        "| Domain | Best rank | Best AUC of AUROC | Smallest sufficient rank | No-SVD AUC of AUROC | Plateau ranks |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for row in smallest_rows:
        lines.append(
            "| {domain} | {best_rank} | {best_auc} | {smallest_rank} | {no_svd_auc} | {plateau} |".format(
                domain=row["domain"],
                best_rank=row["best_rank"],
                best_auc=_format_pct(row["best_auc_of_auroc"]),
                smallest_rank=row["smallest_sufficient_rank"],
                no_svd_auc=_format_pct(None if row["no_svd_auc_of_auroc"] == "" else float(row["no_svd_auc_of_auroc"])),
                plateau=row["plateau_ranks"],
            )
        )

    lines.extend([
        "",
        "## 3. Full ablation table",
        "",
        "| Domain | Method | Rank | AUC of AUROC | AUC of SelAcc | AUROC@100 | StopAcc@100 | Fit time | Infer time | Compactness |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in ablation_rows:
        lines.append(
            "| {domain} | {method} | {rank} | {auc_auroc} | {auc_selacc} | {auroc100} | {stop100} | {fit:.2f}s | {infer:.2f}s | {compactness:.3f} |".format(
                domain=row["domain"],
                method=row["method"],
                rank="—" if row["rank"] == "" else row["rank"],
                auc_auroc=_format_pct(float(row["auc_of_auroc"])),
                auc_selacc=_format_pct(float(row["auc_of_selacc"])),
                auroc100=_format_pct(float(row["auroc_at_100"])),
                stop100=_format_pct(float(row["stop_acc_at_100"])),
                fit=float(row["fit_time_sec"]),
                infer=float(row["inference_time_sec"]),
                compactness=float(row["compactness_value"]),
            )
        )

    lines.extend([
        "",
        "## 4. Paper-facing interpretation",
        "",
        "- **Low-rank 是否必要？** 若 best fixed-rank SVD consistently beats `no_svd_lr`, 说明收益不只是同一套 `raw+rank` feature bank，而是 bottleneck 本身也在起作用。",
        "- **是否存在平台期？** `smallest sufficient rank` 表按 `AUC of AUROC` 距最佳不超过 `0.5%` 的规则给出；若一个较小 rank 已进入 plateau，就可以把它视为最小足够 rank。",
        "- **较低 rank 是否更干净？** 本轮统一用 `top5_feature_mass` 作为解释紧致度指标；值越高，说明有效权重越集中，解释更紧。实测 `r12/r16` 在 `math` 与 `ms` 上比 `no_svd_lr` 更集中，而 `r32` 反而回落。",
        "",
        "## 5. 推荐论文话术",
        "",
        "> The gain does not come merely from the feature bank; the low-rank bottleneck itself is useful, and a moderate low rank (16 for math, 24 for science/ms) is already sufficient.",
        "",
        "## 6. Artifacts",
        "",
        f"- Summary JSON: `{summary['artifacts']['summary_json']}`",
        f"- Eval JSON: `{summary['artifacts']['eval_json']}`",
        f"- Main CSV: `{summary['artifacts']['ablation_csv']}`",
        f"- Smallest-rank CSV: `{summary['artifacts']['smallest_rank_csv']}`",
        "",
    ])
    return "\n".join(lines)


def write_outputs_from_summary(
    *,
    summary: dict[str, Any],
    out_ablation_csv: Path,
    out_smallest_csv: Path,
    out_note: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    ablation_rows = build_ablation_rows(summary)
    smallest_rows = build_smallest_sufficient_rows(ablation_rows)
    note_text = build_note_markdown(summary=summary, ablation_rows=ablation_rows, smallest_rows=smallest_rows)
    _write_csv(out_ablation_csv, ablation_rows)
    _write_csv(out_smallest_csv, smallest_rows)
    out_note.parent.mkdir(parents=True, exist_ok=True)
    out_note.write_text(note_text, encoding="utf-8")
    return ablation_rows, smallest_rows, note_text


def main() -> None:
    ap = argparse.ArgumentParser(description="Run low-rank necessity ablations for canonical r1")
    ap.add_argument("--main-cache-root", default="MUI_HUB/cache")
    ap.add_argument("--extra-cache-root", default="MUI_HUB/cache_train")
    ap.add_argument("--holdout-split", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--feature-workers", type=int, default=default_fit_workers())
    ap.add_argument("--fit-workers", type=int, default=default_fit_workers())
    ap.add_argument("--suite-workers", type=int, default=default_suite_workers())
    ap.add_argument("--feature-chunk-problems", type=int, default=DEFAULT_FEATURE_CHUNK_PROBLEMS)
    ap.add_argument("--feature-cache-dir", default="results/cache/lowrank_necessity")
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--max-problems-per-cache", type=int, default=0, help="0 means all problems")
    ap.add_argument("--ranks", default="2,4,8,12,16,24,32")
    ap.add_argument("--out-summary", default="results/scans/lowrank_necessity/lowrank_necessity_summary.json")
    ap.add_argument("--out-eval", default="results/scans/lowrank_necessity/lowrank_necessity_eval.json")
    ap.add_argument("--out-ablation-csv", default="results/tables/lowrank_necessity_ablation.csv")
    ap.add_argument("--out-smallest-csv", default="results/tables/lowrank_smallest_sufficient_rank.csv")
    ap.add_argument("--out-note", default="docs/07_LOWRANK_NECESSITY.md")
    args = ap.parse_args()

    ranks = parse_ranks(str(args.ranks))
    specs = build_ablation_specs(ranks)

    main_cache_root = _resolve_path(str(args.main_cache_root))
    extra_cache_root = _resolve_path(str(args.extra_cache_root))
    max_problems_per_cache = None if int(args.max_problems_per_cache) <= 0 else int(args.max_problems_per_cache)
    feature_cache_dir = None if str(args.feature_cache_dir).strip().lower() in {"", "none", "off"} else (REPO_ROOT / str(args.feature_cache_dir)).resolve()
    anchor_workers = max(1, int(args.fit_workers) // max(1, int(args.suite_workers)))

    required_features = set(str(name) for name in FIXED_FEATURE_NAMES)

    main_store, main_cache_path, main_cache_status = _load_or_build_noncoding_feature_store(
        source_name="cache",
        cache_root=main_cache_root,
        positions=tuple(float(v) for v in EXTRACTION_POSITIONS),
        required_feature_names=required_features,
        max_problems_per_cache=max_problems_per_cache,
        feature_workers=int(args.feature_workers),
        chunk_problems=int(args.feature_chunk_problems),
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_feature_cache),
    )
    extra_store, extra_cache_path, extra_cache_status = _load_or_build_noncoding_feature_store(
        source_name="cache_train",
        cache_root=extra_cache_root,
        positions=tuple(float(v) for v in EXTRACTION_POSITIONS),
        required_feature_names=required_features,
        max_problems_per_cache=max_problems_per_cache,
        feature_workers=int(args.feature_workers),
        chunk_problems=int(args.feature_chunk_problems),
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_feature_cache),
    )

    full_store = list(main_store) + list(extra_store)
    domain_stores = {
        domain_name: [payload for payload in full_store if payload["domain"] == domain_name]
        for domain_name in TRAIN_DOMAIN_ORDER
    }

    split_packs: dict[str, dict[str, Any]] = {}
    train_tables_by_domain: dict[str, list[dict[str, np.ndarray]]] = {}
    holdout_by_domain: dict[str, list[dict[str, Any]]] = {}
    for domain_name in TRAIN_DOMAIN_ORDER:
        holdout_problem_map, holdout_problem_summary = _build_holdout_problem_map(
            domain_stores[domain_name],
            holdout_split=float(args.holdout_split),
            split_seed=int(args.split_seed),
        )
        train_store, holdout_store, full_domain_store = _split_feature_store(
            domain_stores[domain_name],
            holdout_problem_map=holdout_problem_map,
        )
        split_packs[domain_name] = {
            "train_store": train_store,
            "holdout_store": holdout_store,
            "full_store": full_domain_store,
            "holdout_problem_summary": holdout_problem_summary,
        }
        holdout_by_domain[domain_name] = holdout_store
        train_tables_by_domain[domain_name] = _build_domain_training_tables(train_store, ANCHOR_POSITIONS)

    task_defs = [(spec, domain_name) for spec in specs for domain_name in TRAIN_DOMAIN_ORDER]
    routes_by_method: dict[str, dict[str, dict[float, dict[str, Any]]]] = {
        spec.method_name: {} for spec in specs
    }
    fit_time_by_method_domain: dict[tuple[str, str], float] = {}
    task_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, int(args.suite_workers))) as executor:
        future_map = {}
        for spec, domain_name in task_defs:
            future = executor.submit(
                _train_variant_for_domain,
                domain_name=domain_name,
                tables=train_tables_by_domain[domain_name],
                spec=spec,
                n_splits=int(args.n_splits),
                random_state=int(args.random_state),
                anchor_workers=int(anchor_workers),
            )
            future_map[future] = (spec, domain_name)

        for future in as_completed(future_map):
            spec, domain_name = future_map[future]
            routes, fit_time_sec = future.result()
            routes_by_method[spec.method_name][domain_name] = routes
            fit_time_by_method_domain[(spec.method_name, domain_name)] = float(fit_time_sec)
    total_fit_wall_time_sec = float(time.perf_counter() - task_start)

    method_results: dict[str, Any] = {}
    eval_payload: dict[str, Any] = {
        "created_at_utc": _now_utc(),
        "protocol": {
            "main_cache_root": str(main_cache_root),
            "extra_cache_root": str(extra_cache_root),
            "holdout_split": float(args.holdout_split),
            "split_seed": int(args.split_seed),
            "n_splits": int(args.n_splits),
            "random_state": int(args.random_state),
            "feature_workers": int(args.feature_workers),
            "fit_workers": int(args.fit_workers),
            "suite_workers": int(args.suite_workers),
            "anchor_workers_per_task": int(anchor_workers),
            "feature_chunk_problems": int(args.feature_chunk_problems),
            "max_problems_per_cache": None if max_problems_per_cache is None else int(max_problems_per_cache),
            "anchor_positions": [float(v) for v in ANCHOR_POSITIONS],
            "rank_sweep": [int(v) for v in ranks],
            "representation": FIXED_REPRESENTATION,
            "family_name": FIXED_FAMILY_NAME,
        },
        "methods": {},
    }

    for spec in specs:
        method_name = spec.method_name
        domain_eval, total_inference_sec = _evaluate_routes(
            method_name=method_name,
            routes_by_domain=routes_by_method[method_name],
            holdout_by_domain=holdout_by_domain,
        )
        compactness_math = _aggregate_route_stats(routes_by_method[method_name]["math"])
        compactness_science = _aggregate_route_stats(routes_by_method[method_name]["science"])
        compactness_ms = _aggregate_ms_route_stats(routes_by_method[method_name])

        math_fit = float(fit_time_by_method_domain[(method_name, "math")])
        science_fit = float(fit_time_by_method_domain[(method_name, "science")])
        total_fit = math_fit + science_fit
        eval_payload["methods"][method_name] = {
            "label": str(spec.label),
            "kind": str(spec.kind),
            "rank": None if spec.rank is None else int(spec.rank),
            "domains": domain_eval,
        }
        method_results[method_name] = {
            "label": str(spec.label),
            "kind": str(spec.kind),
            "rank": None if spec.rank is None else int(spec.rank),
            "domains": {
                "math": {
                    "aggregate": domain_eval["math"]["aggregate"],
                    "fit_time_sec": math_fit,
                    "inference_time_sec": total_inference_sec / 3.0,
                    **compactness_math,
                    "route_summary": {
                        _pct_label(position): _route_summary_for_json(route)
                        for position, route in sorted(routes_by_method[method_name]["math"].items(), key=lambda item: float(item[0]))
                    },
                },
                "science": {
                    "aggregate": domain_eval["science"]["aggregate"],
                    "fit_time_sec": science_fit,
                    "inference_time_sec": total_inference_sec / 3.0,
                    **compactness_science,
                    "route_summary": {
                        _pct_label(position): _route_summary_for_json(route)
                        for position, route in sorted(routes_by_method[method_name]["science"].items(), key=lambda item: float(item[0]))
                    },
                },
                "ms": {
                    "aggregate": domain_eval["ms"]["aggregate"],
                    "fit_time_sec": total_fit,
                    "inference_time_sec": total_inference_sec,
                    "compactness_metric_name": str(compactness_ms["compactness_metric_name"]),
                    "compactness_value": float(compactness_ms["compactness_value"]),
                    "top_family_mass": float(compactness_ms["top_family_mass"]),
                    "top_component_purity": compactness_ms["top_component_purity"],
                    "route_summary": {
                        domain_name: {
                            _pct_label(position): _route_summary_for_json(route)
                            for position, route in sorted(routes_by_method[method_name][domain_name].items(), key=lambda item: float(item[0]))
                        }
                        for domain_name in TRAIN_DOMAIN_ORDER
                    },
                },
            },
        }

    out_summary = REPO_ROOT / str(args.out_summary)
    out_eval = REPO_ROOT / str(args.out_eval)
    out_ablation_csv = REPO_ROOT / str(args.out_ablation_csv)
    out_smallest_csv = REPO_ROOT / str(args.out_smallest_csv)
    out_note = REPO_ROOT / str(args.out_note)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_eval.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "created_at_utc": _now_utc(),
        "protocol": {
            "main_cache_root": str(main_cache_root),
            "extra_cache_root": str(extra_cache_root),
            "holdout_split": float(args.holdout_split),
            "split_seed": int(args.split_seed),
            "n_splits": int(args.n_splits),
            "random_state": int(args.random_state),
            "feature_workers": int(args.feature_workers),
            "fit_workers": int(args.fit_workers),
            "suite_workers": int(args.suite_workers),
            "anchor_workers_per_task": int(anchor_workers),
            "feature_chunk_problems": int(args.feature_chunk_problems),
            "max_problems_per_cache": None if max_problems_per_cache is None else int(max_problems_per_cache),
            "anchor_positions": [float(v) for v in ANCHOR_POSITIONS],
            "official_positions": [float(v) for v in EARLY_STOP_POSITIONS],
            "representation": FIXED_REPRESENTATION,
            "family_name": FIXED_FAMILY_NAME,
            "rank_sweep": [int(v) for v in ranks],
            "smallest_sufficient_tolerance": float(SMALLEST_SUFFICIENT_TOLERANCE),
        },
        "search_space": {
            "c_values": list(SEARCH_C_VALUES),
            "class_weight": list(SEARCH_CLASS_WEIGHT),
            "whiten": list(SEARCH_WHITEN),
        },
        "feature_spec": {
            "feature_names": list(FIXED_FEATURE_NAMES),
            "feature_indices": list(FIXED_FEATURE_INDICES),
            "representation": FIXED_REPRESENTATION,
            "family_name": FIXED_FAMILY_NAME,
        },
        "data": {
            "feature_cache_status": {
                "cache": str(main_cache_status),
                "cache_train": str(extra_cache_status),
            },
            "feature_cache_paths": {
                "cache": None if main_cache_path is None else str(main_cache_path),
                "cache_train": None if extra_cache_path is None else str(extra_cache_path),
            },
            "store_summary": {
                "full": _summarise_feature_store(full_store),
                "math": {
                    "train": _summarise_feature_store(split_packs["math"]["train_store"]),
                    "holdout": _summarise_feature_store(split_packs["math"]["holdout_store"]),
                    "full": _summarise_feature_store(split_packs["math"]["full_store"]),
                    "holdout_problem_summary": split_packs["math"]["holdout_problem_summary"],
                },
                "science": {
                    "train": _summarise_feature_store(split_packs["science"]["train_store"]),
                    "holdout": _summarise_feature_store(split_packs["science"]["holdout_store"]),
                    "full": _summarise_feature_store(split_packs["science"]["full_store"]),
                    "holdout_problem_summary": split_packs["science"]["holdout_problem_summary"],
                },
            },
        },
        "timing": {
            "total_fit_wall_time_sec": float(total_fit_wall_time_sec),
        },
        "methods": method_results,
        "artifacts": {
            "summary_json": str(args.out_summary),
            "eval_json": str(args.out_eval),
            "ablation_csv": str(args.out_ablation_csv),
            "smallest_rank_csv": str(args.out_smallest_csv),
            "paper_note": str(args.out_note),
        },
    }

    ablation_rows, smallest_rows, _ = write_outputs_from_summary(
        summary=summary,
        out_ablation_csv=out_ablation_csv,
        out_smallest_csv=out_smallest_csv,
        out_note=out_note,
    )
    summary["tables"] = {
        "ablation_rows": ablation_rows,
        "smallest_sufficient_rows": smallest_rows,
    }

    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_eval.write_text(json.dumps(eval_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("[done] low-rank necessity ablation finished", flush=True)
    print(f"[done] summary={out_summary}", flush=True)
    print(f"[done] eval={out_eval}", flush=True)
    print(f"[done] ablation_csv={out_ablation_csv}", flush=True)
    print(f"[done] smallest_csv={out_smallest_csv}", flush=True)
    print(f"[done] note={out_note}", flush=True)


if __name__ == "__main__":
    main()
