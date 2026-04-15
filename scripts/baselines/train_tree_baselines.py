#!/usr/bin/env python3
"""Train paper-aligned tree baselines for EarlyStop SVD feature stores.

This script answers a narrow question:

Can modern non-linear tabular classifiers beat the current SVD route when they
see the same features, the same grouped 85/15 holdout, and the same EarlyStop
metrics?

Design principles
-----------------
- Reuse the canonical feature-store construction already used by the SVD line.
- Reuse the grouped `dataset + problem_id` holdout logic.
- Compare against the current repo baselines:
  - `es_svd_*_rr_r2` for noncoding (`math`, `science`, `ms`)
  - `es_svd_coding_rr_r1` for `coding`
- Keep the search modest and reproducible:
  - XGBoost + LightGBM only
  - `raw` vs `raw+rank`
  - deterministic shared config search on user-selected anchors (default `100`)
  - final route = refit all official positions (`10..100`) and average 3 seeds

Outputs
-------
- `results/tables/tree_baselines.csv`
- `docs/TREE_BASELINES.md`
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.explain.svd_explain import feature_family
from nad.ops.earlystop_svd import FULL_FEATURE_NAMES, _auroc, _build_representation, load_earlystop_svd_bundle
from scripts.run_earlystop_prefix10_svd_round1 import (
    ANCHOR_POSITIONS,
    EARLY_STOP_POSITIONS,
    _display_path,
    _pct_label,
    aggregate_cache_metrics,
    evaluate_method_from_feature_store,
    make_svd_bundle_score_fn,
)
from scripts.run_structured_ood_suite import (
    CODING_BENCHMARKS,
    MATH_BENCHMARKS,
    ROOT_ORDER,
    _ensure_training_payload,
    _filter_feature_store,
    _load_prebuilt_feature_store,
    _payload_model_family,
    _payload_root,
    _store_summary,
)
from SVDomain.train_es_svd_coding_rr_r1 import DOMAIN_NAME as CODING_DOMAIN_NAME
from SVDomain.train_es_svd_ms_rr_r1 import (
    _build_domain_bundle as _build_domain_bundle_r1,
    _build_domain_training_tables,
    _build_holdout_problem_map,
    _load_or_build_qualified_feature_store,
    _resolve_path,
    _split_feature_store,
)

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)


TREE_MODELS = ("xgboost", "lightgbm")
FEATURE_VARIANTS = ("raw", "raw+rank")
ALL_DOMAINS = ("math", "science", "coding")
SEARCH_ANCHORS = (1.0,)
DEFAULT_SEEDS = (42, 101, 202)
NONCODING_DOMAINS = ("math", "science")


@dataclass(frozen=True)
class TreeSearchConfig:
    model_family: str
    max_depth: int
    learning_rate: float
    n_estimators: int
    min_child_value: int
    subsample: float
    colsample: float

    def to_dict(self) -> dict[str, Any]:
        key = "min_child_weight" if self.model_family == "xgboost" else "min_data_in_leaf"
        return {
            "model_family": self.model_family,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            key: self.min_child_value,
            "subsample": self.subsample,
            "colsample": self.colsample,
        }


def _mean_ignore_nan(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(raw).split(",") if item.strip())


def _parse_pct_csv(raw: str) -> tuple[float, ...]:
    out = []
    for item in _parse_csv(raw):
        value = float(item)
        if value > 1.0:
            value = value / 100.0
        out.append(float(value))
    return tuple(out)


def _route_at(bundle: dict[str, Any], domain: str, position_index: int) -> dict[str, Any]:
    return dict(bundle["domains"][domain]["routes"][position_index])


def _domain_method_name(domain: str) -> str:
    return {
        "math": "es_svd_math_rr_r2_20260412",
        "science": "es_svd_science_rr_r2_20260412",
        "ms": "es_svd_ms_rr_r2_20260412",
        "coding": "es_svd_coding_rr_r1",
    }[str(domain)]


def _current_bundle_paths() -> dict[str, Path]:
    model_dir = REPO_ROOT / "models" / "ml_selectors"
    return {
        "math": model_dir / "es_svd_math_rr_r2_20260412.pkl",
        "science": model_dir / "es_svd_science_rr_r2_20260412.pkl",
        "ms": model_dir / "es_svd_ms_rr_r2_20260412.pkl",
        "coding": model_dir / "es_svd_coding_rr_r1.pkl",
    }


def _build_tree_config_grid(model_family: str) -> list[TreeSearchConfig]:
    if model_family not in TREE_MODELS:
        raise ValueError(f"Unsupported tree model: {model_family}")
    child_grid = (1, 5) if model_family == "xgboost" else (10, 30)
    configs: list[TreeSearchConfig] = []
    for max_depth in (3, 5, 7):
        for learning_rate in (0.03, 0.1):
            for n_estimators in (100, 300, 800):
                for min_child_value in child_grid:
                    for sample in (0.7, 1.0):
                        configs.append(
                            TreeSearchConfig(
                                model_family=str(model_family),
                                max_depth=int(max_depth),
                                learning_rate=float(learning_rate),
                                n_estimators=int(n_estimators),
                                min_child_value=int(min_child_value),
                                subsample=float(sample),
                                colsample=float(sample),
                            )
                        )
    return configs


def _required_features_from_bundles(*bundles: dict[str, Any]) -> set[str]:
    required: set[str] = set()
    for bundle in bundles:
        for domain_bundle in bundle.get("domains", {}).values():
            for route in domain_bundle.get("routes", []):
                if str(route.get("route_type", "svd")) == "baseline":
                    required.add(str(route["signal_name"]))
                else:
                    required.update(str(name) for name in route.get("feature_names", []))
    return required


def _load_all_feature_stores(
    *,
    main_cache_root: str,
    extra_cache_root: str,
    test_cache_root: str,
    feature_cache_dir: Optional[Path],
    refresh_feature_cache: bool,
    feature_workers: int,
    feature_chunk_problems: int,
    required_feature_names: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root_specs = [
        ("cache", main_cache_root),
        ("cache_train", extra_cache_root),
        ("cache_test", test_cache_root),
    ]
    stores: list[dict[str, Any]] = []
    state: dict[str, Any] = {}
    for source_name, cache_root in root_specs:
        prebuilt_store, prebuilt_paths = ([], [])
        if not refresh_feature_cache:
            prebuilt_store, prebuilt_paths = _load_prebuilt_feature_store(str(source_name))
        if prebuilt_store:
            stores.extend(list(prebuilt_store))
            state[source_name] = {
                "status": "loaded_prebuilt",
                "paths": list(prebuilt_paths),
                "cache_root": str(cache_root),
            }
            continue
        store, cache_path, cache_status = _load_or_build_qualified_feature_store(
            source_name=str(source_name),
            cache_root=str(cache_root),
            positions=tuple(float(v) for v in EARLY_STOP_POSITIONS),
            required_feature_names=required_feature_names,
            max_problems_per_cache=None,
            feature_workers=int(feature_workers),
            chunk_problems=int(feature_chunk_problems),
            feature_cache_dir=feature_cache_dir,
            refresh_feature_cache=bool(refresh_feature_cache),
        )
        fixed_store = [_ensure_training_payload(payload, source_name=str(source_name)) for payload in list(store)]
        stores.extend(fixed_store)
        state[source_name] = {
            "status": str(cache_status),
            "path": None if cache_path is None else _display_path(cache_path),
            "cache_root": str(cache_root),
        }
    return stores, state


def _build_tree_estimator(
    *,
    model_family: str,
    config: TreeSearchConfig,
    seed: int,
    n_jobs: int,
):
    if model_family == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            max_depth=int(config.max_depth),
            learning_rate=float(config.learning_rate),
            n_estimators=int(config.n_estimators),
            min_child_weight=float(config.min_child_value),
            subsample=float(config.subsample),
            colsample_bytree=float(config.colsample),
            random_state=int(seed),
            n_jobs=max(1, int(n_jobs)),
            verbosity=0,
        )

    if model_family == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            max_depth=int(config.max_depth),
            num_leaves=max(3, min(255, (2 ** int(config.max_depth)) - 1)),
            learning_rate=float(config.learning_rate),
            n_estimators=int(config.n_estimators),
            min_data_in_leaf=int(config.min_child_value),
            subsample=float(config.subsample),
            feature_fraction=float(config.colsample),
            random_state=int(seed),
            n_jobs=max(1, int(n_jobs)),
            verbose=-1,
        )
    raise ValueError(f"Unsupported model_family: {model_family}")


def _predict_positive_scores(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x), dtype=np.float64)
    return np.asarray(model.predict(x), dtype=np.float64)


def _group_folds(groups: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        return []
    splits = min(int(n_splits), int(unique_groups.size))
    if splits < 2:
        return []
    dummy_x = np.zeros((len(groups), 1), dtype=np.float64)
    gkf = GroupKFold(n_splits=splits)
    return list(gkf.split(dummy_x, groups=groups))


def _fit_cv_for_table(
    *,
    table: dict[str, np.ndarray],
    feature_indices: list[int],
    feature_variant: str,
    model_family: str,
    config: TreeSearchConfig,
    n_splits: int,
    seed: int,
    n_jobs: int,
) -> tuple[float, int]:
    x = _build_representation(
        x_raw=table["x_raw"],
        x_rank=table["x_rank"],
        feature_indices=feature_indices,
        representation=feature_variant,
    )
    y = np.asarray(table["y"], dtype=np.int32)
    groups = np.asarray(table["groups"], dtype=object)
    if x.shape[0] == 0 or np.unique(y).size < 2:
        return float("nan"), 0

    fold_scores: list[float] = []
    for train_idx, test_idx in _group_folds(groups, n_splits=n_splits):
        y_train = y[train_idx]
        y_test = y[test_idx]
        if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
            continue
        model = _build_tree_estimator(
            model_family=model_family,
            config=config,
            seed=seed,
            n_jobs=n_jobs,
        )
        model.fit(x[train_idx], y_train)
        scores = _predict_positive_scores(model, x[test_idx])
        fold_auc = _auroc(scores, y_test)
        if np.isfinite(fold_auc):
            fold_scores.append(float(fold_auc))
    return _mean_ignore_nan(fold_scores), int(len(fold_scores))


def _search_domain_candidate(
    *,
    domain: str,
    train_tables: list[dict[str, np.ndarray]],
    reference_bundle: dict[str, Any],
    model_family: str,
    feature_variant: str,
    search_positions: tuple[float, ...],
    n_splits: int,
    seed: int,
    n_jobs: int,
) -> dict[str, Any]:
    pos_to_idx = {float(position): idx for idx, position in enumerate(EARLY_STOP_POSITIONS)}
    configs = _build_tree_config_grid(model_family)
    best: dict[str, Any] = {"cv_auroc": float("-inf")}
    for config in configs:
        position_scores: list[float] = []
        valid_folds = 0
        for position in search_positions:
            pos_idx = pos_to_idx[float(position)]
            route_template = _route_at(reference_bundle, domain, pos_idx)
            cv_auc, n_valid = _fit_cv_for_table(
                table=train_tables[pos_idx],
                feature_indices=[int(v) for v in route_template["feature_indices"]],
                feature_variant=feature_variant,
                model_family=model_family,
                config=config,
                n_splits=n_splits,
                seed=seed,
                n_jobs=n_jobs,
            )
            if np.isfinite(cv_auc):
                position_scores.append(float(cv_auc))
                valid_folds += int(n_valid)
        mean_cv = _mean_ignore_nan(position_scores)
        if np.isfinite(mean_cv) and mean_cv > float(best["cv_auroc"]):
            best = {
                "cv_auroc": float(mean_cv),
                "n_valid_folds": int(valid_folds),
                "search_positions": [int(round(float(v) * 100.0)) for v in search_positions],
                "config": config.to_dict(),
            }
    if not np.isfinite(float(best["cv_auroc"])):
        raise RuntimeError(f"{domain}/{model_family}/{feature_variant} found no valid tree config")
    return best


def _fit_route_models(
    *,
    table: dict[str, np.ndarray],
    route_template: dict[str, Any],
    feature_variant: str,
    model_family: str,
    config: TreeSearchConfig,
    seeds: tuple[int, ...],
    n_jobs: int,
) -> list[Any]:
    x = _build_representation(
        x_raw=table["x_raw"],
        x_rank=table["x_rank"],
        feature_indices=[int(v) for v in route_template["feature_indices"]],
        representation=feature_variant,
    )
    y = np.asarray(table["y"], dtype=np.int32)
    if x.shape[0] == 0 or np.unique(y).size < 2:
        raise ValueError(f"{route_template.get('training_scope', 'tree_route')} lacks both classes")

    models = []
    for seed in seeds:
        model = _build_tree_estimator(
            model_family=model_family,
            config=config,
            seed=int(seed),
            n_jobs=n_jobs,
        )
        model.fit(x, y)
        models.append(model)
    return models


def _fit_domain_tree_routes(
    *,
    domain: str,
    train_tables: list[dict[str, np.ndarray]],
    reference_bundle: dict[str, Any],
    model_family: str,
    feature_variant: str,
    best_config: dict[str, Any],
    seeds: tuple[int, ...],
    n_jobs: int,
) -> dict[float, dict[str, Any]]:
    routes: dict[float, dict[str, Any]] = {}
    cfg = TreeSearchConfig(
        model_family=model_family,
        max_depth=int(best_config["config"]["max_depth"]),
        learning_rate=float(best_config["config"]["learning_rate"]),
        n_estimators=int(best_config["config"]["n_estimators"]),
        min_child_value=int(
            best_config["config"].get("min_child_weight", best_config["config"].get("min_data_in_leaf"))
        ),
        subsample=float(best_config["config"]["subsample"]),
        colsample=float(best_config["config"]["colsample"]),
    )
    for pos_idx, position in enumerate(EARLY_STOP_POSITIONS):
        route_template = _route_at(reference_bundle, domain, pos_idx)
        models = _fit_route_models(
            table=train_tables[pos_idx],
            route_template=route_template,
            feature_variant=feature_variant,
            model_family=model_family,
            config=cfg,
            seeds=seeds,
            n_jobs=n_jobs,
        )
        routes[float(position)] = {
            "route_type": "tree_ensemble",
            "model_family": str(model_family),
            "feature_variant": str(feature_variant),
            "feature_names": [str(v) for v in route_template["feature_names"]],
            "feature_indices": [int(v) for v in route_template["feature_indices"]],
            "reference_family_name": str(route_template.get("family_name", "?")),
            "reference_representation": str(route_template.get("representation", "?")),
            "training_position": float(position),
            "search_cv_auroc": float(best_config["cv_auroc"]),
            "search_positions_pct": list(best_config["search_positions"]),
            "config": dict(best_config["config"]),
            "seed_values": [int(v) for v in seeds],
            "models": models,
        }
    return routes


def _build_tree_bundle(
    *,
    method_id: str,
    domain_routes: dict[str, dict[float, dict[str, Any]]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    return {
        "bundle_version": str(method_id),
        "created_at_utc": None,
        "feature_names": list(FULL_FEATURE_NAMES),
        "positions": list(EARLY_STOP_POSITIONS),
        "domains": {
            str(domain): {
                "routes": [routes[float(position)] for position in EARLY_STOP_POSITIONS]
            }
            for domain, routes in domain_routes.items()
        },
        "protocol": dict(protocol),
    }


def _score_xraw_with_tree_route(*, x_raw: np.ndarray, route: dict[str, Any]) -> np.ndarray:
    x_rank = np.zeros_like(x_raw, dtype=np.float64)
    if str(route["feature_variant"]) == "raw+rank":
        from nad.ops.earlystop_svd import _rank_transform_matrix

        x_rank = _rank_transform_matrix(x_raw)
    x = _build_representation(
        x_raw=x_raw,
        x_rank=x_rank,
        feature_indices=[int(v) for v in route["feature_indices"]],
        representation=str(route["feature_variant"]),
    )
    seed_scores = []
    for model in route["models"]:
        seed_scores.append(_predict_positive_scores(model, x))
    if not seed_scores:
        return np.zeros((x.shape[0],), dtype=np.float64)
    return np.mean(np.vstack(seed_scores), axis=0, dtype=np.float64)


def make_tree_bundle_score_fn(bundle: dict[str, Any]) -> Callable[[str, int, np.ndarray], np.ndarray]:
    def _score(domain: str, position_index: int, x_raw: np.ndarray) -> np.ndarray:
        route = bundle["domains"][domain]["routes"][position_index]
        return _score_xraw_with_tree_route(x_raw=x_raw, route=route)

    return _score


def _evaluate_bundle(
    *,
    method_name: str,
    bundle: dict[str, Any],
    feature_store: list[dict[str, Any]],
) -> dict[str, Any]:
    return evaluate_method_from_feature_store(
        method_name=str(method_name),
        feature_store=feature_store,
        position_values=tuple(float(v) for v in EARLY_STOP_POSITIONS),
        score_fn=make_tree_bundle_score_fn(bundle),
    )


def _summarise_feature_importance(bundle: dict[str, Any]) -> dict[str, float]:
    totals: defaultdict[str, float] = defaultdict(float)
    for domain_bundle in bundle.get("domains", {}).values():
        for route in domain_bundle.get("routes", []):
            feature_names = [str(v) for v in route.get("feature_names", [])]
            models = list(route.get("models", []))
            if not feature_names or not models:
                continue
            n_feat = len(feature_names)
            family_weights: defaultdict[str, float] = defaultdict(float)
            for model in models:
                if hasattr(model, "feature_importances_"):
                    raw_importance = np.asarray(model.feature_importances_, dtype=np.float64)
                else:
                    continue
                if raw_importance.size == 0:
                    continue
                if raw_importance.size == n_feat:
                    rep_feature_names = feature_names
                elif raw_importance.size == (2 * n_feat):
                    rep_feature_names = feature_names + feature_names
                else:
                    continue
                for feat_name, imp in zip(rep_feature_names, raw_importance):
                    family_weights[feature_family(feat_name)] += float(max(0.0, imp))
            total = sum(family_weights.values())
            if total <= 0.0:
                continue
            for family_name, value in family_weights.items():
                totals[family_name] += float(value / total)
    grand_total = sum(totals.values())
    if grand_total <= 0.0:
        return {}
    return {
        family_name: float(value / grand_total)
        for family_name, value in sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    }


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            float(row["auc_of_auroc"]),
            float(row["auc_of_selacc"]),
            float(row["auroc_at_100"]),
            float(row["stop_acc_at_100"]),
            str(row["model_family"]),
            str(row["feature_variant"]),
        ),
    )


def _judgment_text(best_id_rows: dict[str, dict[str, Any]], ood_rows: dict[str, dict[str, Any]]) -> str:
    gains = []
    for domain in ("math", "science", "coding"):
        row = best_id_rows.get(domain)
        if not row:
            continue
        gains.append(float(row.get("delta_auc_of_auroc_vs_svd", 0.0)))
    ood_gaps = []
    for row in ood_rows.values():
        if np.isfinite(float(row.get("ood_gap_auc_of_auroc", float("nan")))):
            ood_gaps.append(float(row["ood_gap_auc_of_auroc"]))
    mean_gain = _mean_ignore_nan(gains)
    mean_ood_gap = _mean_ignore_nan(ood_gaps)
    if np.isfinite(mean_gain) and mean_gain > 0.01 and (not np.isfinite(mean_ood_gap) or mean_ood_gap > -0.05):
        return "weaker"
    if np.isfinite(mean_gain) and mean_gain < -0.01:
        return "stronger"
    return "complementary"


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


def _render_metrics_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Domain | Tree | Variant | AUC of AUROC | AUC of SelAcc | AUROC@100% | Stop Acc@100% | ΔAUC vs current SVD |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {domain} | {model_family} | {feature_variant} | {auc:.2%} | {selacc:.2%} | {auroc100:.2%} | {stop100:.2%} | {delta:+.2f} pts |".format(
                domain=str(row["domain"]),
                model_family=str(row["model_family"]),
                feature_variant=str(row["feature_variant"]),
                auc=float(row["auc_of_auroc"]),
                selacc=float(row["auc_of_selacc"]),
                auroc100=float(row["auroc_at_100"]),
                stop100=float(row["stop_acc_at_100"]),
                delta=100.0 * float(row.get("delta_auc_of_auroc_vs_svd", 0.0)),
            )
        )
    lines.append("")
    return lines


def _render_ood_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "### Structured OOD",
        "",
        "| Domain | Tree | Variant | OOD protocol | OOD macro AUC of AUROC | ID→OOD gap |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {domain} | {model_family} | {feature_variant} | {protocol} | {ood_auc} | {gap} |".format(
                domain=str(row["domain"]),
                model_family=str(row["model_family"]),
                feature_variant=str(row["feature_variant"]),
                protocol=str(row["protocol"]),
                ood_auc="N/A" if not np.isfinite(float(row["auc_of_auroc"])) else f"{float(row['auc_of_auroc']):.2%}",
                gap="N/A" if not np.isfinite(float(row.get("ood_gap_auc_of_auroc", float("nan")))) else f"{100.0 * float(row['ood_gap_auc_of_auroc']):+.2f} pts",
            )
        )
    lines.append("")
    return lines


def _write_doc(
    *,
    path: Path,
    out_csv_path: Path,
    id_rows: list[dict[str, Any]],
    best_id_rows: dict[str, dict[str, Any]],
    ms_row: Optional[dict[str, Any]],
    ood_rows: dict[str, dict[str, Any]],
    importance_rows: dict[str, dict[str, float]],
    catboost_note: str,
    search_positions_text: str,
) -> None:
    math_best = best_id_rows.get("math")
    science_best = best_id_rows.get("science")
    coding_best = best_id_rows.get("coding")
    ood_list = list(ood_rows.values())
    judgment = _judgment_text(best_id_rows, ood_rows)

    def _best_line(domain: str, row: Optional[dict[str, Any]]) -> str:
        if row is None:
            return f"- `{domain}`: not run."
        return (
            f"- `{domain}`: best tree is `{row['model_family']}` + `{row['feature_variant']}` "
            f"with `AUC of AUROC={float(row['auc_of_auroc']):.2%}` "
            f"(vs current SVD `{float(row['svd_auc_of_auroc']):.2%}`, "
            f"Δ=`{100.0 * float(row['delta_auc_of_auroc_vs_svd']):+.2f}` pts)."
        )

    def _importance_line(domain: str) -> str:
        fams = importance_rows.get(domain, {})
        if not fams:
            return f"- `{domain}`: no stable feature-importance signal exported."
        top = list(fams.items())[:3]
        return "- `{}`: {}.".format(
            domain,
            ", ".join(f"`{name}` {100.0 * float(value):.1f}%" for name, value in top),
        )

    lines = [
        "# Tree Baselines vs Current SVD Route",
        "",
        "## Claim under test",
        "",
        "> If we keep the feature bank, grouped holdout, and EarlyStop metrics fixed, do strong non-linear tabular baselines beat the current SVD route?",
        "",
        "## Protocol",
        "",
        "- Features are built from the same canonical EarlyStop feature stores used by the SVD line.",
        "- Holdout uses the same grouped `85/15` split by `dataset + problem_id`.",
        "- Tree feature variants are limited to `raw` and `raw+rank`.",
        "- Tree families tested here: `XGBoostClassifier` and `LGBMClassifier`.",
        f"- Search stays modest: shared deterministic config search over anchors `{search_positions_text}`, then full refit on all official positions with seeds `{', '.join(str(v) for v in DEFAULT_SEEDS)}`.",
        f"- {catboost_note}",
        "",
        "## Best model per domain",
        "",
        _best_line("math", math_best),
        _best_line("science", science_best),
        _best_line("coding", coding_best),
        (
            "- `ms`: not run."
            if not ms_row
            else (
                f"- `ms`: hybrid best-by-domain tree bundle reaches `AUC of AUROC={float(ms_row['auc_of_auroc']):.2%}` "
                f"vs current SVD `es_svd_ms_rr_r2={float(ms_row['svd_auc_of_auroc']):.2%}` "
                f"(Δ=`{100.0 * float(ms_row['delta_auc_of_auroc_vs_svd']):+.2f}` pts)."
            )
        ),
        "",
    ]
    render_rows = list(id_rows)
    if ms_row:
        render_rows.append(ms_row)
    lines.extend(_render_metrics_table("Grouped ID holdout", render_rows))
    lines.extend(_render_ood_table(ood_list))
    lines.extend(
        [
            "## Direct comparison against the current SVD route",
            "",
            "- `math / science / ms` are compared against the current `r2` routes.",
            "- `coding` is compared against the current `es_svd_coding_rr_r1` route.",
            "- The main fairness constraint is preserved throughout: same feature bank, same grouped holdout unit, same EarlyStop metrics, and the same structured OOD split logic where executed.",
            "",
            "## Where trees help",
            "",
        ]
    )

    def _where_help_text() -> list[str]:
        texts = []
        for domain in ("math", "science", "coding"):
            row = best_id_rows.get(domain)
            if row is None:
                continue
            gain = 100.0 * float(row["delta_auc_of_auroc_vs_svd"])
            ood_domain_rows = [r for r in ood_list if str(r["domain"]) == domain]
            if ood_domain_rows:
                ood_best = max(
                    ood_domain_rows,
                    key=lambda item: (
                        float(item["auc_of_auroc"]) if np.isfinite(float(item["auc_of_auroc"])) else -1.0
                    ),
                )
                gap = float(ood_best.get("ood_gap_auc_of_auroc", float("nan")))
                if gain > 0.0 and np.isfinite(gap) and gap > -0.05:
                    verdict = "helps in ID and remains reasonably stable under OOD"
                elif gain > 0.0 and np.isfinite(gap):
                    verdict = "helps mainly in ID; OOD erodes part of the gain"
                elif gain <= 0.0 and np.isfinite(gap):
                    verdict = "does not beat the current SVD route and also does not open a new OOD advantage"
                else:
                    verdict = "has inconclusive OOD evidence"
            else:
                verdict = "has no structured OOD run in this note"
            texts.append(f"- `{domain}`: {verdict}.")
        return texts

    lines.extend(_where_help_text())
    lines.extend(
        [
            "",
            "## Feature-importance summary",
            "",
            _importance_line("math"),
            _importance_line("science"),
            _importance_line("coding"),
            "",
            "## Paper judgment",
            "",
            f"- On the current evidence, the SVD route should be positioned as **{judgment}** relative to tree baselines.",
            "- If trees win only on some domains or mostly on ID, the clean paper line is not that SVD is obsolete, but that low-rank linear routing and non-linear tabular models capture overlapping yet not identical structure.",
            "",
            "## Artifact",
            "",
            f"- Main table: `{_display_path(out_csv_path)}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _flush_artifacts(
    *,
    out_csv: Path,
    out_doc: Path,
    id_rows: list[dict[str, Any]],
    ms_row: Optional[dict[str, Any]],
    best_id_rows: dict[str, dict[str, Any]],
    ood_rows: dict[str, dict[str, Any]],
    importance_rows: dict[str, dict[str, float]],
    search_positions: tuple[float, ...],
) -> None:
    csv_rows = list(id_rows)
    if ms_row is not None:
        csv_rows.append(ms_row)
    _write_csv(out_csv, csv_rows)
    _write_doc(
        path=out_doc,
        out_csv_path=out_csv,
        id_rows=[row for row in id_rows if row["protocol"] == "id_grouped_85_15"],
        best_id_rows=best_id_rows,
        ms_row=ms_row,
        ood_rows=ood_rows,
        importance_rows=importance_rows,
        catboost_note="`CatBoost` is intentionally skipped here because it is not already wired in this environment and was optional under the study brief.",
        search_positions_text=",".join(str(int(round(100.0 * float(v)))) + "%" for v in search_positions),
    )


def _metric_row(
    *,
    domain: str,
    protocol: str,
    model_family: str,
    feature_variant: str,
    aggregate: dict[str, Any],
    search_summary: Optional[dict[str, Any]],
    svd_method: Optional[str],
    svd_aggregate: Optional[dict[str, Any]],
    note: str = "",
) -> dict[str, Any]:
    row = {
        "domain": str(domain),
        "protocol": str(protocol),
        "model_family": str(model_family),
        "feature_variant": str(feature_variant),
        "auc_of_auroc": float(aggregate.get("auc_of_auroc", float("nan"))),
        "auc_of_selacc": float(aggregate.get("auc_of_selacc", float("nan"))),
        "auroc_at_100": float(aggregate.get("auroc@100%", float("nan"))),
        "stop_acc_at_100": float(aggregate.get("stop_acc@100%", float("nan"))),
        "samples": int(aggregate.get("samples", 0)),
        "num_caches": int(aggregate.get("num_caches", 0)),
        "svd_method": "" if svd_method is None else str(svd_method),
        "svd_auc_of_auroc": float("nan") if svd_aggregate is None else float(svd_aggregate.get("auc_of_auroc", float("nan"))),
        "svd_auc_of_selacc": float("nan") if svd_aggregate is None else float(svd_aggregate.get("auc_of_selacc", float("nan"))),
        "svd_auroc_at_100": float("nan") if svd_aggregate is None else float(svd_aggregate.get("auroc@100%", float("nan"))),
        "svd_stop_acc_at_100": float("nan") if svd_aggregate is None else float(svd_aggregate.get("stop_acc@100%", float("nan"))),
        "delta_auc_of_auroc_vs_svd": float("nan") if svd_aggregate is None else float(aggregate.get("auc_of_auroc", float("nan")) - svd_aggregate.get("auc_of_auroc", float("nan"))),
        "delta_auc_of_selacc_vs_svd": float("nan") if svd_aggregate is None else float(aggregate.get("auc_of_selacc", float("nan")) - svd_aggregate.get("auc_of_selacc", float("nan"))),
        "delta_auroc_at_100_vs_svd": float("nan") if svd_aggregate is None else float(aggregate.get("auroc@100%", float("nan")) - svd_aggregate.get("auroc@100%", float("nan"))),
        "delta_stop_acc_at_100_vs_svd": float("nan") if svd_aggregate is None else float(aggregate.get("stop_acc@100%", float("nan")) - svd_aggregate.get("stop_acc@100%", float("nan"))),
        "search_cv_auroc": float("nan") if search_summary is None else float(search_summary.get("cv_auroc", float("nan"))),
        "search_positions_pct": "" if search_summary is None else ",".join(str(v) for v in search_summary.get("search_positions", [])),
        "search_config_json": "" if search_summary is None else json.dumps(search_summary.get("config", {}), sort_keys=True),
        "note": str(note),
    }
    return row


def _run_ood_for_best_candidate(
    *,
    domain: str,
    best_row: dict[str, Any],
    all_feature_store: list[dict[str, Any]],
    reference_bundle: dict[str, Any],
    seeds: tuple[int, ...],
    n_jobs: int,
    n_splits: int,
) -> list[dict[str, Any]]:
    domain_store = _filter_feature_store(all_feature_store, domain=domain)
    rows: list[dict[str, Any]] = []

    def _run_fold(protocol: str, split_name: str, train_store: list[dict[str, Any]], test_store: list[dict[str, Any]]) -> None:
        if not train_store or not test_store:
            return
        train_tables = _build_domain_training_tables(train_store, tuple(float(v) for v in EARLY_STOP_POSITIONS))
        best_config = {
            "cv_auroc": float(best_row["search_cv_auroc"]),
            "search_positions": [int(v) for v in str(best_row["search_positions_pct"]).split(",") if str(v)],
            "config": json.loads(str(best_row["search_config_json"])),
        }
        try:
            routes = _fit_domain_tree_routes(
                domain=domain,
                train_tables=train_tables,
                reference_bundle=reference_bundle,
                model_family=str(best_row["model_family"]),
                feature_variant=str(best_row["feature_variant"]),
                best_config=best_config,
                seeds=seeds,
                n_jobs=n_jobs,
            )
        except ValueError as exc:
            print(
                f"[ood] skip domain={domain} protocol={protocol} split={split_name}: {exc}",
                flush=True,
            )
            return
        bundle = _build_tree_bundle(
            method_id=f"tree_{domain}_{protocol}_{split_name}",
            domain_routes={domain: routes},
            protocol={"protocol": str(protocol), "split_name": str(split_name)},
        )
        method_eval = _evaluate_bundle(
            method_name=f"tree_{domain}",
            bundle=bundle,
            feature_store=test_store,
        )
        rows.append(
            _metric_row(
                domain=domain,
                protocol=protocol,
                model_family=str(best_row["model_family"]),
                feature_variant=str(best_row["feature_variant"]),
                aggregate=method_eval["aggregate"],
                search_summary=None,
                svd_method=None,
                svd_aggregate=None,
                note=str(split_name),
            )
        )

    if domain == "math":
        for benchmark in MATH_BENCHMARKS:
            _run_fold(
                protocol="math_benchmark_withheld",
                split_name=f"withheld_{benchmark}",
                train_store=_filter_feature_store(domain_store, include_datasets=set(MATH_BENCHMARKS) - {benchmark}),
                test_store=_filter_feature_store(domain_store, include_datasets={benchmark}),
            )
    else:
        roots = sorted(
            {_payload_root(payload) for payload in domain_store},
            key=lambda item: ROOT_ORDER.index(item) if item in ROOT_ORDER else 999,
        )
        for root_name in roots:
            _run_fold(
                protocol="cache_root_withheld",
                split_name=f"withheld_{root_name}",
                train_store=_filter_feature_store(domain_store, exclude_roots={root_name}),
                test_store=_filter_feature_store(domain_store, include_roots={root_name}),
            )
        model_families = sorted({_payload_model_family(payload) for payload in domain_store})
        for model_family in model_families:
            _run_fold(
                protocol="model_family_withheld",
                split_name=f"withheld_{model_family}",
                train_store=_filter_feature_store(domain_store, exclude_model_families={model_family}),
                test_store=_filter_feature_store(domain_store, include_model_families={model_family}),
            )
    if not rows:
        return []

    by_protocol: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_protocol[str(row["protocol"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    id_auc = float(best_row["auc_of_auroc"])
    for protocol, proto_rows in sorted(by_protocol.items()):
        aggregate = {
            "auc_of_auroc": _mean_ignore_nan(row["auc_of_auroc"] for row in proto_rows),
            "auc_of_selacc": _mean_ignore_nan(row["auc_of_selacc"] for row in proto_rows),
            "auroc@100%": _mean_ignore_nan(row["auroc_at_100"] for row in proto_rows),
            "stop_acc@100%": _mean_ignore_nan(row["stop_acc_at_100"] for row in proto_rows),
            "samples": int(sum(int(row["samples"]) for row in proto_rows)),
            "num_caches": int(sum(int(row["num_caches"]) for row in proto_rows)),
        }
        summary_row = _metric_row(
            domain=domain,
            protocol=protocol,
            model_family=str(best_row["model_family"]),
            feature_variant=str(best_row["feature_variant"]),
            aggregate=aggregate,
            search_summary=None,
            svd_method=None,
            svd_aggregate=None,
            note=f"macro_mean_over_{len(proto_rows)}_folds",
        )
        summary_row["ood_gap_auc_of_auroc"] = float(aggregate["auc_of_auroc"] - id_auc)
        summary_rows.append(summary_row)
    return summary_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Train XGBoost/LightGBM baselines on canonical EarlyStop feature stores")
    ap.add_argument("--main-cache-root", default="MUI_HUB/cache")
    ap.add_argument("--extra-cache-root", default="MUI_HUB/cache_train")
    ap.add_argument("--test-cache-root", default="/home/jovyan/public-ro/MUI_HUB/cache_test")
    ap.add_argument("--feature-cache-dir", default="results/cache/tree_baselines")
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--feature-workers", type=int, default=8)
    ap.add_argument("--feature-chunk-problems", type=int, default=8)
    ap.add_argument("--holdout-split", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--search-seed", type=int, default=42)
    ap.add_argument("--search-positions", default="100")
    ap.add_argument("--tree-threads", type=int, default=8)
    ap.add_argument("--seeds", default="42,101,202")
    ap.add_argument("--domains", default="math,science", help="Subset of domains: math,science,coding")
    ap.add_argument("--models", default="xgboost,lightgbm", help="Subset of models: xgboost,lightgbm")
    ap.add_argument("--feature-variants", default="raw,raw+rank", help="Subset of feature variants: raw,raw+rank")
    ap.add_argument("--run-coding", action="store_true", help="Also run the coding ID/OOD slice")
    ap.add_argument("--run-ood", action="store_true", help="Run structured OOD for the best ID tree per domain")
    ap.add_argument("--out-csv", default="results/tables/tree_baselines.csv")
    ap.add_argument("--out-doc", default="docs/TREE_BASELINES.md")
    args = ap.parse_args()

    main_cache_root = _resolve_path(str(args.main_cache_root))
    extra_cache_root = _resolve_path(str(args.extra_cache_root))
    test_cache_root = _resolve_path(str(args.test_cache_root))
    feature_cache_dir = None
    if str(args.feature_cache_dir).strip().lower() not in {"", "none", "off"}:
        feature_cache_dir = (REPO_ROOT / str(args.feature_cache_dir)).resolve()

    seeds = tuple(int(v) for v in _parse_csv(args.seeds))
    if not seeds:
        seeds = DEFAULT_SEEDS
    search_positions = _parse_pct_csv(args.search_positions)
    if not search_positions:
        search_positions = SEARCH_ANCHORS
    requested_domains = tuple(v for v in _parse_csv(args.domains) if v in ALL_DOMAINS)
    if not requested_domains:
        requested_domains = ("math", "science")
    requested_models = tuple(v for v in _parse_csv(args.models) if v in TREE_MODELS)
    if not requested_models:
        requested_models = TREE_MODELS
    requested_feature_variants = tuple(v for v in _parse_csv(args.feature_variants) if v in FEATURE_VARIANTS)
    if not requested_feature_variants:
        requested_feature_variants = FEATURE_VARIANTS

    bundle_paths = _current_bundle_paths()
    reference_bundles = {
        name: load_earlystop_svd_bundle(path)
        for name, path in bundle_paths.items()
    }
    required_feature_names = _required_features_from_bundles(
        reference_bundles["math"],
        reference_bundles["science"],
        reference_bundles["ms"],
        reference_bundles["coding"],
    )
    all_feature_store, feature_cache_state = _load_all_feature_stores(
        main_cache_root=main_cache_root,
        extra_cache_root=extra_cache_root,
        test_cache_root=test_cache_root,
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_feature_cache),
        feature_workers=int(args.feature_workers),
        feature_chunk_problems=int(args.feature_chunk_problems),
        required_feature_names=required_feature_names,
    )

    noncoding_source = _filter_feature_store(
        all_feature_store,
        include_roots={"cache", "cache_train"},
        exclude_datasets=set(CODING_BENCHMARKS),
    )
    coding_source = _filter_feature_store(
        all_feature_store,
        domain=CODING_DOMAIN_NAME,
        include_roots={"cache", "cache_train"},
    )

    noncoding_holdout_map, _ = _build_holdout_problem_map(
        noncoding_source,
        holdout_split=float(args.holdout_split),
        split_seed=int(args.split_seed),
    )
    noncoding_train_store, noncoding_holdout_store, _ = _split_feature_store(
        noncoding_source,
        holdout_problem_map=noncoding_holdout_map,
    )
    math_train_store = _filter_feature_store(noncoding_train_store, domain="math")
    math_holdout_store = _filter_feature_store(noncoding_holdout_store, domain="math")
    science_train_store = _filter_feature_store(noncoding_train_store, domain="science")
    science_holdout_store = _filter_feature_store(noncoding_holdout_store, domain="science")

    coding_train_store: list[dict[str, Any]] = []
    coding_holdout_store: list[dict[str, Any]] = []
    if args.run_coding and coding_source:
        coding_holdout_map, _ = _build_holdout_problem_map(
            coding_source,
            holdout_split=float(args.holdout_split),
            split_seed=int(args.split_seed),
        )
        coding_train_store, coding_holdout_store, _ = _split_feature_store(
            coding_source,
            holdout_problem_map=coding_holdout_map,
        )

    domain_specs: dict[str, dict[str, Any]] = {
        "math": {
            "train_store": math_train_store,
            "holdout_store": math_holdout_store,
            "reference_bundle": reference_bundles["math"],
            "svd_method": "es_svd_math_rr_r2_20260412",
            "search_n_splits": 5,
        },
        "science": {
            "train_store": science_train_store,
            "holdout_store": science_holdout_store,
            "reference_bundle": reference_bundles["science"],
            "svd_method": "es_svd_science_rr_r2_20260412",
            "search_n_splits": 5,
        },
    }
    if args.run_coding and coding_train_store and coding_holdout_store:
        domain_specs["coding"] = {
            "train_store": coding_train_store,
            "holdout_store": coding_holdout_store,
            "reference_bundle": reference_bundles["coding"],
            "svd_method": "es_svd_coding_rr_r1",
            "search_n_splits": 3,
        }
    domain_specs = {
        domain: spec
        for domain, spec in domain_specs.items()
        if domain in set(requested_domains)
    }

    id_rows: list[dict[str, Any]] = []
    best_id_rows: dict[str, dict[str, Any]] = {}
    best_bundles: dict[str, dict[str, Any]] = {}
    importance_rows: dict[str, dict[str, float]] = {}

    for domain, spec in domain_specs.items():
        print(f"[domain] start {domain}", flush=True)
        train_tables = _build_domain_training_tables(
            spec["train_store"],
            tuple(float(v) for v in EARLY_STOP_POSITIONS),
        )
        svd_eval = evaluate_method_from_feature_store(
            method_name=str(spec["svd_method"]),
            feature_store=spec["holdout_store"],
            position_values=tuple(float(v) for v in EARLY_STOP_POSITIONS),
            score_fn=make_svd_bundle_score_fn(spec["reference_bundle"]),
        )
        domain_candidate_rows: list[dict[str, Any]] = []
        for model_family in requested_models:
            for feature_variant in requested_feature_variants:
                print(
                    f"[search] domain={domain:<7s} model={model_family:<8s} variant={feature_variant}",
                    flush=True,
                )
                search_summary = _search_domain_candidate(
                    domain=domain,
                    train_tables=train_tables,
                    reference_bundle=spec["reference_bundle"],
                    model_family=model_family,
                    feature_variant=feature_variant,
                    search_positions=search_positions,
                    n_splits=int(spec["search_n_splits"]),
                    seed=int(args.search_seed),
                    n_jobs=int(args.tree_threads),
                )
                routes = _fit_domain_tree_routes(
                    domain=domain,
                    train_tables=train_tables,
                    reference_bundle=spec["reference_bundle"],
                    model_family=model_family,
                    feature_variant=feature_variant,
                    best_config=search_summary,
                    seeds=seeds,
                    n_jobs=int(args.tree_threads),
                )
                bundle = _build_tree_bundle(
                    method_id=f"tree_{domain}_{model_family}_{feature_variant}",
                    domain_routes={domain: routes},
                    protocol={
                        "holdout_split": float(args.holdout_split),
                        "split_seed": int(args.split_seed),
                        "search_positions_pct": [int(round(100.0 * float(v))) for v in search_positions],
                        "search_seed": int(args.search_seed),
                    },
                )
                tree_eval = _evaluate_bundle(
                    method_name=f"tree_{domain}_{model_family}_{feature_variant}",
                    bundle=bundle,
                    feature_store=spec["holdout_store"],
                )
                row = _metric_row(
                    domain=domain,
                    protocol="id_grouped_85_15",
                    model_family=model_family,
                    feature_variant=feature_variant,
                    aggregate=tree_eval["aggregate"],
                    search_summary=search_summary,
                    svd_method=str(spec["svd_method"]),
                    svd_aggregate=svd_eval["aggregate"],
                )
                domain_candidate_rows.append(row)
                id_rows.append(row)
                best_bundles[f"{domain}::{model_family}::{feature_variant}"] = bundle

        best_row = _best_row(domain_candidate_rows)
        best_id_rows[domain] = best_row
        importance_rows[domain] = _summarise_feature_importance(
            best_bundles[f"{domain}::{best_row['model_family']}::{best_row['feature_variant']}"]
        )
        print(
            f"[domain] best {domain}: {best_row['model_family']} + {best_row['feature_variant']} "
            f"AUC={float(best_row['auc_of_auroc']):.4f} "
            f"delta_vs_svd={100.0 * float(best_row['delta_auc_of_auroc_vs_svd']):+.2f} pts",
            flush=True,
        )

    # Combined noncoding (`ms`) uses the best per-domain tree routes.
    ms_row: Optional[dict[str, Any]] = None
    if "math" in best_id_rows and "science" in best_id_rows:
        ms_bundle = _build_tree_bundle(
            method_id="tree_ms_best_by_domain",
            domain_routes={
                "math": {
                    float(position): route
                    for position, route in zip(
                        EARLY_STOP_POSITIONS,
                        best_bundles[
                            f"math::{best_id_rows['math']['model_family']}::{best_id_rows['math']['feature_variant']}"
                        ]["domains"]["math"]["routes"],
                    )
                },
                "science": {
                    float(position): route
                    for position, route in zip(
                        EARLY_STOP_POSITIONS,
                        best_bundles[
                            f"science::{best_id_rows['science']['model_family']}::{best_id_rows['science']['feature_variant']}"
                        ]["domains"]["science"]["routes"],
                    )
                },
            },
            protocol={
                "composition": "best_by_domain",
                "holdout_split": float(args.holdout_split),
                "split_seed": int(args.split_seed),
            },
        )
        ms_tree_eval = _evaluate_bundle(
            method_name="tree_ms_best_by_domain",
            bundle=ms_bundle,
            feature_store=noncoding_holdout_store,
        )
        ms_svd_eval = evaluate_method_from_feature_store(
            method_name="es_svd_ms_rr_r2_20260412",
            feature_store=noncoding_holdout_store,
            position_values=tuple(float(v) for v in EARLY_STOP_POSITIONS),
            score_fn=make_svd_bundle_score_fn(reference_bundles["ms"]),
        )
        ms_row = _metric_row(
            domain="ms",
            protocol="id_grouped_85_15",
            model_family="hybrid",
            feature_variant="best_by_domain",
            aggregate=ms_tree_eval["aggregate"],
            search_summary=None,
            svd_method="es_svd_ms_rr_r2_20260412",
            svd_aggregate=ms_svd_eval["aggregate"],
            note="math/science best routes composed into one noncoding bundle",
        )

    out_csv = REPO_ROOT / str(args.out_csv)
    out_doc = REPO_ROOT / str(args.out_doc)
    _flush_artifacts(
        out_csv=out_csv,
        out_doc=out_doc,
        id_rows=id_rows,
        ms_row=ms_row,
        best_id_rows=best_id_rows,
        ood_rows={},
        importance_rows=importance_rows,
        search_positions=search_positions,
    )

    ood_rows: dict[str, dict[str, Any]] = {}
    if args.run_ood:
        for domain in domain_specs:
            print(f"[ood] start {domain}", flush=True)
            summaries = _run_ood_for_best_candidate(
                domain=domain,
                best_row=best_id_rows[domain],
                all_feature_store=all_feature_store,
                reference_bundle=domain_specs[domain]["reference_bundle"],
                seeds=seeds,
                n_jobs=int(args.tree_threads),
                n_splits=3,
            )
            if summaries:
                best_ood = max(
                    summaries,
                    key=lambda row: (
                        float(row["auc_of_auroc"]) if np.isfinite(float(row["auc_of_auroc"])) else -1.0
                    ),
                )
                ood_rows[domain] = best_ood
                id_rows.extend(summaries)

    _flush_artifacts(
        out_csv=out_csv,
        out_doc=out_doc,
        id_rows=id_rows,
        ms_row=ms_row,
        best_id_rows=best_id_rows,
        ood_rows=ood_rows,
        importance_rows=importance_rows,
        search_positions=search_positions,
    )

    print("[done] tree baselines", flush=True)
    print(f"[artifact] csv={_display_path(out_csv)}", flush=True)
    print(f"[artifact] doc={_display_path(out_doc)}", flush=True)
    print(f"[cache] {json.dumps(feature_cache_state, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
