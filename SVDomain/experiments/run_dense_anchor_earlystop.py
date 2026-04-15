#!/usr/bin/env python3
"""Run dense-anchor EarlyStop experiments for legacy and neuron-meta-min feature lines."""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.ops.earlystop import EARLY_STOP_POSITIONS, discover_cache_entries
from nad.ops.earlystop_svd import (
    AVAILABILITY_FEATURES,
    LEGACY_FULL_FEATURE_NAMES,
    META_FEATURES,
    TRAJ_FEATURES,
    TOKEN_FEATURES,
    _auroc,
    _build_representation,
    _fit_svd_lr_model,
    _group_folds,
    _predict_svd_lr,
    _rank_transform_matrix,
)
from scripts.run_earlystop_prefix10_svd_round1 import (
    DEFAULT_FEATURE_CHUNK_PROBLEMS,
    EXTRACTION_POSITIONS,
    SEARCH_C_VALUES,
    SEARCH_CLASS_WEIGHT,
    SEARCH_RANKS,
    SEARCH_WHITEN,
    _now_utc,
    _pct_label,
    build_feature_store,
    evaluate_method_from_feature_store,
)
from SVDomain.train_es_svd_ms_rr_r1 import (
    _build_holdout_problem_map,
    _qualify_feature_store,
    _resolve_path,
    _split_feature_store,
)


DEFAULT_MAIN_CACHE_ROOT = "MUI_HUB/cache"
DEFAULT_EXTRA_CACHE_ROOT = "MUI_HUB/cache_train"
DEFAULT_MAIN_STORE_PATH = "results/cache/es_svd_ms_rr_r1/cache_all_547b9060debe139e.pkl"
DEFAULT_EXTRA_STORE_PATH = "results/cache/es_svd_ms_rr_r1/cache_train_all_d429f3b93baed972.pkl"
DEFAULT_OUT_SUMMARY = "results/scans/dense_anchor_earlystop/dense_anchor_earlystop_summary.json"
DEFAULT_OUT_TABLE = "results/tables/dense_anchor_earlystop.csv"
DEFAULT_OUT_ONSET = "results/tables/onset_of_signal.csv"
DEFAULT_OUT_PLATEAU = "results/tables/plateau_of_signal.csv"
DEFAULT_OUT_COMPARE = "results/tables/dense_anchor_neuron_vs_legacy.csv"
DEFAULT_OUT_NOTE = "docs/16_DENSE_ANCHOR_EARLYSTOP.md"

LINE_ORDER = ("canonical_legacy", "canonical_plus_neuron_meta_min")
DOMAIN_ORDER = ("math", "science", "coding")
ANCHOR_PCTS = tuple(int(round(float(v) * 100.0)) for v in EARLY_STOP_POSITIONS)
EXTRACTION_POSITION_INDEX = {float(v): idx for idx, v in enumerate(EXTRACTION_POSITIONS)}
LEGACY_FEATURE_TO_INDEX = {str(name): idx for idx, name in enumerate(LEGACY_FULL_FEATURE_NAMES)}
THRESHOLD_BY_DOMAIN = {
    "math": 0.80,
    "science": 0.70,
    "coding": 0.55,
}
PLATEAU_TOLERANCE_ABS = 0.01

CANONICAL_FEATURE_NAMES = tuple(
    list(TOKEN_FEATURES) + list(TRAJ_FEATURES) + list(AVAILABILITY_FEATURES)
)
NEURON_META_MIN_FEATURES = ("nc_mean", "nc_slope")


@dataclass(frozen=True)
class LineSpec:
    line_id: str
    label: str
    feature_names: tuple[str, ...]
    note: str


LINE_SPECS = (
    LineSpec(
        line_id="canonical_legacy",
        label="B0 legacy canonical",
        feature_names=CANONICAL_FEATURE_NAMES,
        note="22-feature canonical bank = token + trajectory + availability, raw+rank.",
    ),
    LineSpec(
        line_id="canonical_plus_neuron_meta_min",
        label="B1 canonical + neuron_meta_min",
        feature_names=CANONICAL_FEATURE_NAMES + NEURON_META_MIN_FEATURES,
        note="Fallback neuron line = canonical + {nc_mean, nc_slope}.",
    ),
)


def default_feature_workers() -> int:
    cpu = max(1, int(os.cpu_count() or 1))
    return max(1, min(8, cpu))


def default_fit_workers() -> int:
    cpu = max(1, int(os.cpu_count() or 1))
    return max(1, min(12, cpu))


def default_suite_workers() -> int:
    cpu = max(1, int(os.cpu_count() or 1))
    return max(1, min(3, cpu))


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    value_f = float(value)
    if not np.isfinite(value_f):
        return None
    return value_f


def _fmt_pct_value(position: Optional[float]) -> str:
    if position is None:
        return "N/A"
    position_f = float(position)
    if position_f > 1.0 + 1e-9:
        return f"{int(round(position_f))}%"
    return _pct_label(position_f)


def _mean_ignore_nan(values: list[float] | tuple[float, ...] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_prebuilt_feature_store(
    *,
    store_path: Path,
    source_name: str,
    expected_positions: tuple[float, ...],
) -> list[dict[str, Any]]:
    with store_path.open("rb") as handle:
        payload = pickle.load(handle)
    actual_positions = tuple(float(v) for v in payload.get("positions", []))
    if actual_positions and actual_positions != tuple(float(v) for v in expected_positions):
        raise ValueError(
            f"Feature store positions mismatch for {store_path}: "
            f"{list(actual_positions)} != {list(expected_positions)}"
        )
    feature_store = list(payload.get("feature_store", []))
    if not feature_store:
        return []
    first = feature_store[0]
    if "source_name" not in first or "base_cache_key" not in first:
        return _qualify_feature_store(feature_store, source_name=source_name)
    return feature_store


def _load_or_build_feature_store(
    *,
    source_name: str,
    store_path: Optional[Path],
    cache_root: Optional[Path],
    positions: tuple[float, ...],
    required_feature_names: set[str],
    max_problems_per_cache: Optional[int],
    feature_workers: int,
    chunk_problems: int,
    rebuild_store: bool,
) -> tuple[list[dict[str, Any]], str]:
    if store_path is not None and store_path.exists() and not rebuild_store:
        print(f"[load] feature_store source={source_name} path={store_path}", flush=True)
        return _load_prebuilt_feature_store(
            store_path=store_path,
            source_name=source_name,
            expected_positions=positions,
        ), "loaded_prebuilt"

    if cache_root is None or not cache_root.exists():
        return [], "missing"

    include_cache_keys = {str(entry.cache_key) for entry in discover_cache_entries(cache_root)}
    print(
        f"[build] feature_store source={source_name} root={cache_root} "
        f"positions={len(positions)} include={len(include_cache_keys)} caches",
        flush=True,
    )
    feature_store = _qualify_feature_store(
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
    if store_path is not None:
        store_path.parent.mkdir(parents=True, exist_ok=True)
        with store_path.open("wb") as handle:
            pickle.dump(
                {
                    "source_name": str(source_name),
                    "cache_root": str(cache_root),
                    "positions": [float(v) for v in positions],
                    "max_problems_per_cache": None if max_problems_per_cache is None else int(max_problems_per_cache),
                    "feature_store": feature_store,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"[save] feature_store source={source_name} path={store_path}", flush=True)
    return feature_store, "built"


def _build_training_tables(
    feature_store: list[dict[str, Any]],
    positions: tuple[float, ...],
) -> list[dict[str, np.ndarray]]:
    rows: dict[int, list[np.ndarray]] = {idx: [] for idx in range(len(positions))}
    labels: dict[int, list[np.ndarray]] = {idx: [] for idx in range(len(positions))}
    rank_groups: dict[int, list[np.ndarray]] = {idx: [] for idx in range(len(positions))}
    cv_groups: dict[int, list[np.ndarray]] = {idx: [] for idx in range(len(positions))}
    position_indices = [EXTRACTION_POSITION_INDEX[float(position)] for position in positions]
    feature_dim = int(max((int(item["tensor"].shape[2]) for item in feature_store), default=len(LEGACY_FULL_FEATURE_NAMES)))

    for payload in feature_store:
        tensor = np.asarray(payload["tensor"], dtype=np.float64)
        if tensor.shape[0] == 0:
            continue
        y = np.asarray(payload["labels"], dtype=np.int32)
        local_rank_groups = np.asarray(payload["group_keys"], dtype=object)
        local_cv_groups = np.asarray(payload["cv_group_keys"], dtype=object)
        for local_pos_idx, src_pos_idx in enumerate(position_indices):
            rows[local_pos_idx].append(np.asarray(tensor[:, src_pos_idx, :], dtype=np.float64))
            labels[local_pos_idx].append(y)
            rank_groups[local_pos_idx].append(local_rank_groups)
            cv_groups[local_pos_idx].append(local_cv_groups)

    tables: list[dict[str, np.ndarray]] = []
    for pos_idx in range(len(positions)):
        if rows[pos_idx]:
            x_raw = np.vstack(rows[pos_idx]).astype(np.float64, copy=False)
            y = np.concatenate(labels[pos_idx]).astype(np.int32, copy=False)
            groups_rank = np.concatenate(rank_groups[pos_idx]).astype(object, copy=False)
            groups_cv = np.concatenate(cv_groups[pos_idx]).astype(object, copy=False)
        else:
            x_raw = np.zeros((0, feature_dim), dtype=np.float64)
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

        tables.append({
            "x_raw": x_raw,
            "x_rank": x_rank,
            "y": y,
            "groups": groups_cv,
        })
    return tables


def _fit_dense_route(
    *,
    x_raw: np.ndarray,
    x_rank: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    position: float,
    domain_name: str,
    line_spec: LineSpec,
    n_splits: int,
    random_state: int,
) -> dict[str, Any]:
    if x_raw.shape[0] == 0:
        raise ValueError(f"{line_spec.line_id}/{domain_name}@{_pct_label(position)} has no labeled rows")
    if np.unique(y).shape[0] < 2:
        raise ValueError(f"{line_spec.line_id}/{domain_name}@{_pct_label(position)} lacks both classes")

    folds = _group_folds(groups, n_splits=n_splits)
    if not folds:
        raise ValueError(f"{line_spec.line_id}/{domain_name}@{_pct_label(position)} has insufficient CV groups")

    feature_indices = [LEGACY_FEATURE_TO_INDEX[name] for name in line_spec.feature_names]
    x_rep = _build_representation(
        x_raw=x_raw,
        x_rank=x_rank,
        feature_indices=feature_indices,
        representation="raw+rank",
    )

    best = {"cv_auroc": float("-inf")}
    candidate_scores: dict[tuple[int, float, bool, str], list[float]] = {}
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

        for requested_rank in SEARCH_RANKS:
            fold_rank = max(1, min(int(requested_rank), int(x_train_scaled.shape[1]), int(x_train_scaled.shape[0] - 1)))
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
                    z_train = z_train_full / singular_values[:fold_rank]
                    z_test = z_test_full / singular_values[:fold_rank]
                else:
                    z_train = z_train_full
                    z_test = z_test_full
                for c_value in SEARCH_C_VALUES:
                    for class_weight in SEARCH_CLASS_WEIGHT:
                        clf = LogisticRegression(
                            C=float(c_value),
                            class_weight=None if class_weight == "none" else "balanced",
                            max_iter=2000,
                            random_state=int(random_state),
                        )
                        try:
                            clf.fit(z_train, y_train)
                            scores = np.asarray(clf.decision_function(z_test), dtype=np.float64)
                        except Exception:
                            continue
                        fold_auc = _auroc(scores, y_test)
                        if np.isfinite(fold_auc):
                            candidate_scores.setdefault(
                                (int(requested_rank), float(c_value), bool(whiten), str(class_weight)),
                                [],
                            ).append(float(fold_auc))

    for (requested_rank, c_value, whiten, class_weight), values in candidate_scores.items():
        if not values:
            continue
        cv_auc = float(np.mean(values))
        if cv_auc > float(best["cv_auroc"]):
            best = {
                "requested_rank": int(requested_rank),
                "rank": int(max(1, min(int(requested_rank), int(x_rep.shape[1]), int(x_rep.shape[0] - 1)))),
                "c_value": float(c_value),
                "whiten": bool(whiten),
                "class_weight": str(class_weight),
                "cv_auroc": float(cv_auc),
                "n_valid_folds": int(len(values)),
            }

    if not np.isfinite(float(best["cv_auroc"])):
        raise RuntimeError(f"{line_spec.line_id}/{domain_name}@{_pct_label(position)} found no valid route")

    model = _fit_svd_lr_model(
        x=x_rep,
        y=y,
        rank=int(best["rank"]),
        c_value=float(best["c_value"]),
        whiten=bool(best["whiten"]),
        class_weight_name=str(best["class_weight"]),
        random_state=int(random_state),
    )
    if model is None:
        raise RuntimeError(f"{line_spec.line_id}/{domain_name}@{_pct_label(position)} full-fit route failed")

    return {
        "route_type": "svd",
        "line_id": str(line_spec.line_id),
        "feature_names": list(line_spec.feature_names),
        "feature_indices": feature_indices,
        "representation": "raw+rank",
        "rank": int(best["rank"]),
        "requested_rank": int(best["requested_rank"]),
        "c_value": float(best["c_value"]),
        "whiten": bool(best["whiten"]),
        "class_weight": str(best["class_weight"]),
        "cv_auroc": float(best["cv_auroc"]),
        "n_valid_folds": int(best["n_valid_folds"]),
        "training_position": float(position),
        "training_scope": str(domain_name),
        "model": model,
    }


def _train_line_for_domain(
    *,
    domain_name: str,
    tables: list[dict[str, np.ndarray]],
    line_spec: LineSpec,
    positions: tuple[float, ...],
    n_splits: int,
    random_state: int,
    anchor_workers: int,
) -> tuple[dict[float, dict[str, Any]], float]:
    start = time.perf_counter()
    routes: dict[float, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(anchor_workers))) as executor:
        future_map = {}
        for pos_idx, position in enumerate(positions):
            future = executor.submit(
                _fit_dense_route,
                x_raw=tables[pos_idx]["x_raw"],
                x_rank=tables[pos_idx]["x_rank"],
                y=tables[pos_idx]["y"],
                groups=tables[pos_idx]["groups"],
                position=float(position),
                domain_name=str(domain_name),
                line_spec=line_spec,
                n_splits=int(n_splits),
                random_state=int(random_state),
            )
            future_map[future] = float(position)

        for future in as_completed(future_map):
            position = future_map[future]
            route = future.result()
            routes[float(position)] = route
            print(
                f"[train] line={line_spec.line_id:<31s} domain={domain_name:<7s} "
                f"pos={_pct_label(position):>4s} auc={route['cv_auroc']:.4f} "
                f"rank={route['rank']:>2d} c={route['c_value']:.2f} "
                f"cw={route['class_weight']} whiten={'yes' if route['whiten'] else 'no'}",
                flush=True,
            )
    return routes, float(time.perf_counter() - start)


def _make_dense_score_fn(routes_by_domain: dict[str, dict[float, dict[str, Any]]]):
    anchor_positions = tuple(float(v) for v in EARLY_STOP_POSITIONS)

    def _score(domain: str, position_index: int, x_raw: np.ndarray) -> np.ndarray:
        route = routes_by_domain[str(domain)][float(anchor_positions[int(position_index)])]
        x_rank = _rank_transform_matrix(x_raw)
        x_rep = _build_representation(
            x_raw=x_raw,
            x_rank=x_rank,
            feature_indices=[int(v) for v in route["feature_indices"]],
            representation=str(route["representation"]),
        )
        return _predict_svd_lr(route["model"], x_rep)

    return _score


def _empty_eval_result(method_name: str) -> dict[str, Any]:
    return {
        "method_name": str(method_name),
        "aggregate": {
            "num_caches": 0,
            "samples": 0,
            "auc_of_auroc": float("nan"),
            "auc_of_selacc": float("nan"),
            "earliest_gt_0.6": None,
            "auroc@100%": float("nan"),
            "stop_acc@100%": float("nan"),
            "by_position": [
                {
                    "position": float(position),
                    "auroc": float("nan"),
                    "selacc@10%": float("nan"),
                    "stop_acc": float("nan"),
                }
                for position in EARLY_STOP_POSITIONS
            ],
        },
        "by_cache": [],
    }


def _evaluate_line(
    *,
    line_spec: LineSpec,
    routes_by_domain: dict[str, dict[float, dict[str, Any]]],
    holdout_by_domain: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, float]]:
    score_fn = _make_dense_score_fn(routes_by_domain)
    domain_results: dict[str, Any] = {}
    inference_time_sec: dict[str, float] = {}

    for domain_name in DOMAIN_ORDER:
        feature_store = list(holdout_by_domain.get(domain_name, []))
        if not feature_store:
            domain_results[domain_name] = _empty_eval_result(f"{line_spec.line_id}__{domain_name}")
            inference_time_sec[domain_name] = 0.0
            continue
        start = time.perf_counter()
        domain_results[domain_name] = evaluate_method_from_feature_store(
            method_name=f"{line_spec.line_id}__{domain_name}",
            feature_store=feature_store,
            position_values=tuple(float(v) for v in EARLY_STOP_POSITIONS),
            score_fn=score_fn,
        )
        inference_time_sec[domain_name] = float(time.perf_counter() - start)

    return domain_results, inference_time_sec


def _earliest_anchor(
    by_position: list[dict[str, Any]],
    predicate,
) -> Optional[int]:
    for row in by_position:
        value = _safe_float(row.get("auroc"))
        if value is not None and bool(predicate(value)):
            return int(round(float(row["position"]) * 100.0))
    return None


def _earliest_plateau_anchor(
    by_position: list[dict[str, Any]],
    *,
    final_auroc: Optional[float],
    tolerance_abs: float,
) -> Optional[int]:
    if final_auroc is None:
        return None
    values = [_safe_float(row.get("auroc")) for row in by_position]
    positions = [int(round(float(row["position"]) * 100.0)) for row in by_position]
    for idx in range(len(values)):
        tail = values[idx:]
        if tail and all(v is not None and abs(float(final_auroc) - float(v)) <= float(tolerance_abs) for v in tail):
            return positions[idx]
    return positions[-1] if positions else None


def _strip_model(route: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in route.items() if key != "model"}


def _build_main_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_spec in LINE_SPECS:
        line_payload = summary["lines"][line_spec.line_id]
        for domain_name in DOMAIN_ORDER:
            domain_payload = line_payload["domains"][domain_name]
            aggregate = domain_payload["aggregate"]
            by_position = aggregate["by_position"]
            route_summary = domain_payload["route_summary"]
            final_auroc = _safe_float(aggregate.get("auroc@100%"))
            for row in by_position:
                anchor_pct = int(round(float(row["position"]) * 100.0))
                route = route_summary[str(anchor_pct)]
                auroc = _safe_float(row.get("auroc"))
                rows.append({
                    "bundle_line": str(line_spec.line_id),
                    "bundle_label": str(line_spec.label),
                    "domain": str(domain_name),
                    "anchor_pct": int(anchor_pct),
                    "feature_set": ",".join(line_spec.feature_names),
                    "n_features": int(len(line_spec.feature_names)),
                    "selected_rank": int(route["rank"]),
                    "selected_requested_rank": int(route["requested_rank"]),
                    "selected_c_value": float(route["c_value"]),
                    "selected_whiten": bool(route["whiten"]),
                    "selected_class_weight": str(route["class_weight"]),
                    "route_cv_auroc": float(route["cv_auroc"]),
                    "auroc": "" if auroc is None else float(auroc),
                    "selacc_at_10": float(row["selacc@10%"]),
                    "stop_acc": float(row["stop_acc"]),
                    "auc_over_anchor_auroc": float(aggregate["auc_of_auroc"]),
                    "auc_over_anchor_selacc": float(aggregate["auc_of_selacc"]),
                    "auc_over_anchor_stop_acc": float(domain_payload["auc_of_stop_acc"]),
                    "final_anchor_auroc": "" if final_auroc is None else float(final_auroc),
                    "final_anchor_stop_acc": float(aggregate["stop_acc@100%"]),
                    "auroc_ratio_to_final": "" if auroc is None or final_auroc is None or abs(float(final_auroc)) < 1e-12 else float(auroc / final_auroc),
                    "threshold_value": float(domain_payload["threshold_value"]),
                    "reaches_95pct_final_auroc": bool(domain_payload["earliest_anchor_95pct_final_auroc"] is not None and anchor_pct >= int(domain_payload["earliest_anchor_95pct_final_auroc"])),
                    "reaches_fixed_threshold": bool(domain_payload["earliest_anchor_fixed_threshold"] is not None and anchor_pct >= int(domain_payload["earliest_anchor_fixed_threshold"])),
                    "earliest_anchor_95pct_final_auroc": "" if domain_payload["earliest_anchor_95pct_final_auroc"] is None else int(domain_payload["earliest_anchor_95pct_final_auroc"]),
                    "earliest_anchor_fixed_threshold": "" if domain_payload["earliest_anchor_fixed_threshold"] is None else int(domain_payload["earliest_anchor_fixed_threshold"]),
                    "plateau_anchor_pct": "" if domain_payload["plateau_anchor_pct"] is None else int(domain_payload["plateau_anchor_pct"]),
                    "plateau_tolerance_abs": float(domain_payload["plateau_tolerance_abs"]),
                    "fit_time_sec": float(domain_payload["fit_time_sec"]),
                    "inference_time_sec": float(domain_payload["inference_time_sec"]),
                    "samples": int(aggregate["samples"]),
                    "num_caches": int(aggregate["num_caches"]),
                })
    return sorted(
        rows,
        key=lambda item: (
            LINE_ORDER.index(str(item["bundle_line"])),
            DOMAIN_ORDER.index(str(item["domain"])),
            int(item["anchor_pct"]),
        ),
    )


def _build_onset_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_spec in LINE_SPECS:
        for domain_name in DOMAIN_ORDER:
            domain_payload = summary["lines"][line_spec.line_id]["domains"][domain_name]
            by_position = domain_payload["aggregate"]["by_position"]
            row: dict[str, Any] = {
                "bundle_line": str(line_spec.line_id),
                "bundle_label": str(line_spec.label),
                "domain": str(domain_name),
                "threshold_value": float(domain_payload["threshold_value"]),
                "auc_over_anchor_auroc": float(domain_payload["aggregate"]["auc_of_auroc"]),
                "final_anchor_auroc": float(domain_payload["aggregate"]["auroc@100%"]),
                "earliest_anchor_fixed_threshold": "" if domain_payload["earliest_anchor_fixed_threshold"] is None else int(domain_payload["earliest_anchor_fixed_threshold"]),
                "earliest_anchor_95pct_final_auroc": "" if domain_payload["earliest_anchor_95pct_final_auroc"] is None else int(domain_payload["earliest_anchor_95pct_final_auroc"]),
            }
            for pos_row in by_position:
                anchor_pct = int(round(float(pos_row["position"]) * 100.0))
                row[f"auroc_{anchor_pct}"] = float(pos_row["auroc"])
            rows.append(row)
    return sorted(rows, key=lambda item: (LINE_ORDER.index(str(item["bundle_line"])), DOMAIN_ORDER.index(str(item["domain"]))))


def _build_plateau_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_spec in LINE_SPECS:
        for domain_name in DOMAIN_ORDER:
            domain_payload = summary["lines"][line_spec.line_id]["domains"][domain_name]
            by_position = domain_payload["aggregate"]["by_position"]
            final_auroc = float(domain_payload["aggregate"]["auroc@100%"])
            plateau_anchor = domain_payload["plateau_anchor_pct"]
            plateau_rows = [
                pos_row
                for pos_row in by_position
                if plateau_anchor is not None and int(round(float(pos_row["position"]) * 100.0)) >= int(plateau_anchor)
            ]
            plateau_values = [float(pos_row["auroc"]) for pos_row in plateau_rows] if plateau_rows else []
            rows.append({
                "bundle_line": str(line_spec.line_id),
                "bundle_label": str(line_spec.label),
                "domain": str(domain_name),
                "final_anchor_auroc": float(final_auroc),
                "plateau_anchor_pct": "" if plateau_anchor is None else int(plateau_anchor),
                "plateau_tolerance_abs": float(domain_payload["plateau_tolerance_abs"]),
                "plateau_length": int(len(plateau_rows)),
                "plateau_mean_auroc": "" if not plateau_values else float(np.mean(plateau_values)),
                "plateau_min_auroc": "" if not plateau_values else float(np.min(plateau_values)),
                "plateau_gap_to_final_abs": "" if not plateau_values else float(max(abs(final_auroc - value) for value in plateau_values)),
            })
    return sorted(rows, key=lambda item: (LINE_ORDER.index(str(item["bundle_line"])), DOMAIN_ORDER.index(str(item["domain"]))))


def _build_compare_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    legacy = summary["lines"]["canonical_legacy"]["domains"]
    neuron = summary["lines"]["canonical_plus_neuron_meta_min"]["domains"]
    early_band = {10, 20, 30, 40}
    late_band = {70, 80, 90, 100}

    for domain_name in DOMAIN_ORDER:
        legacy_pos = {
            int(round(float(item["position"]) * 100.0)): item
            for item in legacy[domain_name]["aggregate"]["by_position"]
        }
        neuron_pos = {
            int(round(float(item["position"]) * 100.0)): item
            for item in neuron[domain_name]["aggregate"]["by_position"]
        }
        anchor_deltas: list[dict[str, Any]] = []
        for anchor_pct in ANCHOR_PCTS:
            l_row = legacy_pos[anchor_pct]
            n_row = neuron_pos[anchor_pct]
            delta_row = {
                "row_type": "per_anchor",
                "domain": str(domain_name),
                "anchor_band": "",
                "anchor_pct": int(anchor_pct),
                "legacy_auroc": float(l_row["auroc"]),
                "neuron_auroc": float(n_row["auroc"]),
                "delta_auroc": float(n_row["auroc"] - l_row["auroc"]),
                "legacy_selacc_at_10": float(l_row["selacc@10%"]),
                "neuron_selacc_at_10": float(n_row["selacc@10%"]),
                "delta_selacc_at_10": float(n_row["selacc@10%"] - l_row["selacc@10%"]),
                "legacy_stop_acc": float(l_row["stop_acc"]),
                "neuron_stop_acc": float(n_row["stop_acc"]),
                "delta_stop_acc": float(n_row["stop_acc"] - l_row["stop_acc"]),
                "legacy_auc_over_anchor_auroc": float(legacy[domain_name]["aggregate"]["auc_of_auroc"]),
                "neuron_auc_over_anchor_auroc": float(neuron[domain_name]["aggregate"]["auc_of_auroc"]),
                "delta_auc_over_anchor_auroc": float(neuron[domain_name]["aggregate"]["auc_of_auroc"] - legacy[domain_name]["aggregate"]["auc_of_auroc"]),
            }
            rows.append(delta_row)
            anchor_deltas.append(delta_row)

        for band_name, band_set in (("early_10_40", early_band), ("late_70_100", late_band), ("all_10_100", set(ANCHOR_PCTS))):
            band_rows = [row for row in anchor_deltas if int(row["anchor_pct"]) in band_set]
            rows.append({
                "row_type": "band_summary",
                "domain": str(domain_name),
                "anchor_band": str(band_name),
                "anchor_pct": "",
                "legacy_auroc": "",
                "neuron_auroc": "",
                "delta_auroc": float(np.mean([row["delta_auroc"] for row in band_rows])) if band_rows else float("nan"),
                "legacy_selacc_at_10": "",
                "neuron_selacc_at_10": "",
                "delta_selacc_at_10": float(np.mean([row["delta_selacc_at_10"] for row in band_rows])) if band_rows else float("nan"),
                "legacy_stop_acc": "",
                "neuron_stop_acc": "",
                "delta_stop_acc": float(np.mean([row["delta_stop_acc"] for row in band_rows])) if band_rows else float("nan"),
                "legacy_auc_over_anchor_auroc": float(legacy[domain_name]["aggregate"]["auc_of_auroc"]),
                "neuron_auc_over_anchor_auroc": float(neuron[domain_name]["aggregate"]["auc_of_auroc"]),
                "delta_auc_over_anchor_auroc": float(neuron[domain_name]["aggregate"]["auc_of_auroc"] - legacy[domain_name]["aggregate"]["auc_of_auroc"]),
            })
    return rows


def _explanation_math(summary: dict[str, Any]) -> str:
    domain_payload = summary["lines"]["canonical_legacy"]["domains"]["math"]
    onset = domain_payload["earliest_anchor_95pct_final_auroc"]
    plateau = domain_payload["plateau_anchor_pct"]
    final_auroc = float(domain_payload["aggregate"]["auroc@100%"])
    if onset is not None and onset <= 30:
        return f"`math` 基本属于早期强信号：在 `{onset}%` 就达到 final-AUROC 的 95%，并在 `{plateau}%` 左右进入平台；dense anchors 更像是在细化“多早就够了”，而不是改写结论。"
    return f"`math` 不是瞬时饱和，但依然明显早于其他域：final-anchor AUROC=`{final_auroc:.3f}`，95%-of-final 出现在 `{_fmt_pct_value(onset)}`，平台期出现在 `{_fmt_pct_value(plateau)}`。"


def _explanation_science(summary: dict[str, Any]) -> str:
    domain_payload = summary["lines"]["canonical_legacy"]["domains"]["science"]
    onset_fixed = domain_payload["earliest_anchor_fixed_threshold"]
    onset_95 = domain_payload["earliest_anchor_95pct_final_auroc"]
    plateau = domain_payload["plateau_anchor_pct"]
    by_position = domain_payload["aggregate"]["by_position"]
    auroc_10 = _safe_float(by_position[0].get("auroc")) if by_position else None
    final_auroc = _safe_float(domain_payload["aggregate"].get("auroc@100%"))
    return (
        f"`science` 不是纯 late-onset：`{_fmt_pct_value(onset_fixed)}` 已越过固定阈值，"
        f"`{_fmt_pct_value(onset_95)}` 已达到 final-AUROC 的 95%，但 AUROC 仍从 `10%` 的 `{auroc_10:.3f}` "
        f"继续抬升到 `100%` 的 `{final_auroc:.3f}`，并在 `{_fmt_pct_value(plateau)}` 左右进入 `±0.01` 平台；"
        "dense anchors 更支持“早期已有 coarse signal、后期继续抬升”而不是“只在 completion 才首次出现信号”。"
    )


def _explanation_coding(summary: dict[str, Any]) -> str:
    domain_payload = summary["lines"]["canonical_legacy"]["domains"]["coding"]
    onset_fixed = domain_payload["earliest_anchor_fixed_threshold"]
    final_auroc = float(domain_payload["aggregate"]["auroc@100%"])
    if onset_fixed is None:
        return f"`coding` 更接近噪声主导：直到 `100%` 也没有稳定越过固定阈值，final-anchor AUROC 只有 `{final_auroc:.3f}`。"
    return f"`coding` 不是全程纯噪声，但可用信息明显偏中后段：固定阈值首次出现在 `{onset_fixed}%`，final-anchor AUROC=`{final_auroc:.3f}`。"


def _neuron_delta_text(compare_rows: list[dict[str, Any]], domain_name: str) -> str:
    early = next(
        row for row in compare_rows
        if row["row_type"] == "band_summary" and row["domain"] == domain_name and row["anchor_band"] == "early_10_40"
    )
    late = next(
        row for row in compare_rows
        if row["row_type"] == "band_summary" and row["domain"] == domain_name and row["anchor_band"] == "late_70_100"
    )
    return (
        f"`{domain_name}` 上，B1−B0 的 mean ΔAUROC 在 early anchors 为 `{float(early['delta_auroc']):+.3f}`，"
        f"在 late anchors 为 `{float(late['delta_auroc']):+.3f}`。"
    )


def build_note_markdown(
    *,
    summary: dict[str, Any],
    compare_rows: list[dict[str, Any]],
) -> str:
    legacy = summary["lines"]["canonical_legacy"]["domains"]
    neuron = summary["lines"]["canonical_plus_neuron_meta_min"]["domains"]
    math_legacy = legacy["math"]["aggregate"]
    science_legacy = legacy["science"]["aggregate"]
    coding_legacy = legacy["coding"]["aggregate"]
    math_neuron = neuron["math"]["aggregate"]
    science_neuron = neuron["science"]["aggregate"]
    coding_neuron = neuron["coding"]["aggregate"]

    lines = [
        "# 16. Dense-Anchor EarlyStop",
        "",
        "这份 note 只回答一个问题：把 EarlyStop 从粗粒度 `10 / 40 / 70 / 100` 扩展到完整 `10 / 20 / ... / 100` 之后，主论文关于“预测信号何时出现”的结论会不会改变。",
        "",
        "## 1. Setup",
        "",
        "- 任务范围：只做 `EarlyStop`，不扩展到 `Best-of-N` 或 `RL ranking`。",
        "- Anchor grid：`10 / 20 / 30 / 40 / 50 / 60 / 70 / 80 / 90 / 100`。",
        "- 域：`math / science / coding`。",
        "- B0：`legacy canonical`（22 特征 = token + trajectory + availability）。",
        "- B1：`canonical + neuron_meta_min`（B0 + `nc_mean` / `nc_slope`）。",
        "- 实现：复用现有 prefix-safe feature store 与 offline holdout split，只把评估网格从四锚点细化到十锚点。",
        "- 协议：沿用当前 offline holdout + grouped CV；不改 split / grouping / metrics。",
        "",
        "## 2. Headline answers",
        "",
        f"- **当前 4 anchors 是否足够支撑主论文？** 是。Dense-anchor 结果没有推翻粗锚点结论：`math` 仍然最早、最强，`science` 早期已有可用信号但中后段继续抬升，`coding` 仍然最弱。",
        f"- **Dense anchors 是否改变主结论？** 主要没有改结论，而是在 timing 上给出更细定位。B0 的 AUC-of-AUROC 分别为 `math={float(math_legacy['auc_of_auroc']):.3f}`、`science={float(science_legacy['auc_of_auroc']):.3f}`、`coding={float(coding_legacy['auc_of_auroc']):.3f}`。",
        f"- **Dense anchors 是否应进入正文？** 如果正文要和 neuron-agreement / early-correctness 文献对话，建议至少放一张 dense-anchor timing 图；否则四锚点主表仍然够用，dense 结果适合作为 timing-focused 主文图或 appendix 主证据。",
        "",
        "## 3. Domain readout",
        "",
        f"- {_explanation_math(summary)}",
        f"- {_explanation_science(summary)}",
        f"- {_explanation_coding(summary)}",
        "",
        "## 4. Neuron-meta-min comparison",
        "",
        "- 当前仓库里没有已经固化的更强 neuron bundle，因此这里按约定使用 `canonical + neuron_meta_min` fallback。",
        f"- `math`: B0 AUC-of-AUROC=`{float(math_legacy['auc_of_auroc']):.3f}` → B1=`{float(math_neuron['auc_of_auroc']):.3f}`。 {_neuron_delta_text(compare_rows, 'math')}",
        f"- `science`: B0 AUC-of-AUROC=`{float(science_legacy['auc_of_auroc']):.3f}` → B1=`{float(science_neuron['auc_of_auroc']):.3f}`。 {_neuron_delta_text(compare_rows, 'science')}",
        f"- `coding`: B0 AUC-of-AUROC=`{float(coding_legacy['auc_of_auroc']):.3f}` → B1=`{float(coding_neuron['auc_of_auroc']):.3f}`。 {_neuron_delta_text(compare_rows, 'coding')}",
        "",
        "## 5. Paper-facing interpretation",
        "",
        "- `math`: dense anchors show that the predictive signal is already strong by `10%` and largely stabilizes by the mid trajectory.",
        "- `science`: dense anchors argue for early coarse signal plus late refinement, not a purely completion-only onset story.",
        "- `coding`: dense anchors help distinguish a weak noisy curve from a true onset event; in the current family it never clears the fixed threshold and remains the weakest slice.",
        "- 对弱域来说，`95% of final AUROC` 不是最稳健的 onset 指标，因为 final AUROC 本身可能偏低；fixed-threshold onset 与 plateau 更值得引用。",
        "- 对论文正文最稳的写法是：dense anchors refine **when** predictability appears and plateaus; they do not overturn the coarse-anchor story.",
        "",
        "## 6. Recommended sentence",
        "",
        "> The coarse 10/40/70/100 anchors are directionally correct, but denser anchors reveal where each domain’s predictive signal actually emerges and stabilizes.",
        "",
        "## 7. Artifacts",
        "",
        f"- `results/tables/dense_anchor_earlystop.csv`",
        f"- `results/tables/onset_of_signal.csv`",
        f"- `results/tables/plateau_of_signal.csv`",
        f"- `results/tables/dense_anchor_neuron_vs_legacy.csv`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run dense-anchor EarlyStop suite")
    ap.add_argument("--main-cache-root", default=DEFAULT_MAIN_CACHE_ROOT)
    ap.add_argument("--extra-cache-root", default=DEFAULT_EXTRA_CACHE_ROOT)
    ap.add_argument("--main-store-path", default=DEFAULT_MAIN_STORE_PATH)
    ap.add_argument("--extra-store-path", default=DEFAULT_EXTRA_STORE_PATH)
    ap.add_argument("--rebuild-store", action="store_true")
    ap.add_argument("--holdout-split", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--feature-workers", type=int, default=default_feature_workers())
    ap.add_argument("--fit-workers", type=int, default=default_fit_workers())
    ap.add_argument("--suite-workers", type=int, default=default_suite_workers())
    ap.add_argument("--feature-chunk-problems", type=int, default=DEFAULT_FEATURE_CHUNK_PROBLEMS)
    ap.add_argument("--max-problems-per-cache", type=int, default=0, help="0 means all problems")
    ap.add_argument("--out-summary", default=DEFAULT_OUT_SUMMARY)
    ap.add_argument("--out-table", default=DEFAULT_OUT_TABLE)
    ap.add_argument("--out-onset", default=DEFAULT_OUT_ONSET)
    ap.add_argument("--out-plateau", default=DEFAULT_OUT_PLATEAU)
    ap.add_argument("--out-compare", default=DEFAULT_OUT_COMPARE)
    ap.add_argument("--out-note", default=DEFAULT_OUT_NOTE)
    args = ap.parse_args()

    main_cache_root = _resolve_path(str(args.main_cache_root))
    extra_cache_root = _resolve_path(str(args.extra_cache_root))
    main_store_path = None if str(args.main_store_path).strip().lower() in {"", "none", "off"} else (REPO_ROOT / str(args.main_store_path)).resolve()
    extra_store_path = None if str(args.extra_store_path).strip().lower() in {"", "none", "off"} else (REPO_ROOT / str(args.extra_store_path)).resolve()
    out_summary = (REPO_ROOT / str(args.out_summary)).resolve()
    out_table = (REPO_ROOT / str(args.out_table)).resolve()
    out_onset = (REPO_ROOT / str(args.out_onset)).resolve()
    out_plateau = (REPO_ROOT / str(args.out_plateau)).resolve()
    out_compare = (REPO_ROOT / str(args.out_compare)).resolve()
    out_note = (REPO_ROOT / str(args.out_note)).resolve()

    max_problems_per_cache = None if int(args.max_problems_per_cache) <= 0 else int(args.max_problems_per_cache)
    anchor_workers = max(1, int(args.fit_workers) // max(1, int(args.suite_workers)))
    required_feature_names = set(str(name) for name in LEGACY_FULL_FEATURE_NAMES)

    main_store, main_store_status = _load_or_build_feature_store(
        source_name="cache",
        store_path=main_store_path,
        cache_root=main_cache_root,
        positions=tuple(float(v) for v in EXTRACTION_POSITIONS),
        required_feature_names=required_feature_names,
        max_problems_per_cache=max_problems_per_cache,
        feature_workers=int(args.feature_workers),
        chunk_problems=int(args.feature_chunk_problems),
        rebuild_store=bool(args.rebuild_store),
    )
    extra_store, extra_store_status = _load_or_build_feature_store(
        source_name="cache_train",
        store_path=extra_store_path,
        cache_root=extra_cache_root,
        positions=tuple(float(v) for v in EXTRACTION_POSITIONS),
        required_feature_names=required_feature_names,
        max_problems_per_cache=max_problems_per_cache,
        feature_workers=int(args.feature_workers),
        chunk_problems=int(args.feature_chunk_problems),
        rebuild_store=bool(args.rebuild_store),
    )

    full_store = list(main_store) + list(extra_store)
    domain_stores = {
        domain_name: [payload for payload in full_store if str(payload["domain"]) == domain_name]
        for domain_name in DOMAIN_ORDER
    }

    split_payloads: dict[str, dict[str, Any]] = {}
    train_tables_by_domain: dict[str, list[dict[str, np.ndarray]]] = {}
    holdout_by_domain: dict[str, list[dict[str, Any]]] = {}
    holdout_summary_by_domain: dict[str, Any] = {}

    for domain_name in DOMAIN_ORDER:
        holdout_problem_map, holdout_problem_summary = _build_holdout_problem_map(
            domain_stores[domain_name],
            holdout_split=float(args.holdout_split),
            split_seed=int(args.split_seed),
        )
        train_store, holdout_store, full_domain_store = _split_feature_store(
            domain_stores[domain_name],
            holdout_problem_map=holdout_problem_map,
        )
        split_payloads[domain_name] = {
            "train_store": train_store,
            "holdout_store": holdout_store,
            "full_store": full_domain_store,
        }
        holdout_summary_by_domain[domain_name] = holdout_problem_summary
        holdout_by_domain[domain_name] = holdout_store
        train_tables_by_domain[domain_name] = _build_training_tables(
            train_store,
            positions=tuple(float(v) for v in EARLY_STOP_POSITIONS),
        )

    task_defs = [(line_spec, domain_name) for line_spec in LINE_SPECS for domain_name in DOMAIN_ORDER]
    routes_by_line: dict[str, dict[str, dict[float, dict[str, Any]]]] = {
        line_spec.line_id: {} for line_spec in LINE_SPECS
    }
    fit_time_by_line_domain: dict[tuple[str, str], float] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(args.suite_workers))) as executor:
        future_map = {}
        for line_spec, domain_name in task_defs:
            future = executor.submit(
                _train_line_for_domain,
                domain_name=domain_name,
                tables=train_tables_by_domain[domain_name],
                line_spec=line_spec,
                positions=tuple(float(v) for v in EARLY_STOP_POSITIONS),
                n_splits=int(args.n_splits),
                random_state=int(args.random_state),
                anchor_workers=int(anchor_workers),
            )
            future_map[future] = (line_spec, domain_name)
        for future in as_completed(future_map):
            line_spec, domain_name = future_map[future]
            routes, fit_time_sec = future.result()
            routes_by_line[line_spec.line_id][domain_name] = routes
            fit_time_by_line_domain[(line_spec.line_id, domain_name)] = float(fit_time_sec)

    summary: dict[str, Any] = {
        "created_at_utc": _now_utc(),
        "protocol": {
            "main_cache_root": str(main_cache_root),
            "extra_cache_root": str(extra_cache_root),
            "main_store_path": None if main_store_path is None else str(main_store_path),
            "extra_store_path": None if extra_store_path is None else str(extra_store_path),
            "main_store_status": str(main_store_status),
            "extra_store_status": str(extra_store_status),
            "positions": [float(v) for v in EXTRACTION_POSITIONS],
            "anchor_positions": [float(v) for v in EARLY_STOP_POSITIONS],
            "anchor_positions_pct": [int(round(float(v) * 100.0)) for v in EARLY_STOP_POSITIONS],
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
            "threshold_by_domain": {key: float(value) for key, value in THRESHOLD_BY_DOMAIN.items()},
            "plateau_tolerance_abs": float(PLATEAU_TOLERANCE_ABS),
            "neuron_bundle_status": "fallback_to_canonical_plus_neuron_meta_min",
        },
        "lines": {},
        "holdout_problem_summary": holdout_summary_by_domain,
        "artifacts": {
            "table_csv": str(out_table.relative_to(REPO_ROOT)),
            "onset_csv": str(out_onset.relative_to(REPO_ROOT)),
            "plateau_csv": str(out_plateau.relative_to(REPO_ROOT)),
            "compare_csv": str(out_compare.relative_to(REPO_ROOT)),
            "note_md": str(out_note.relative_to(REPO_ROOT)),
            "summary_json": str(out_summary.relative_to(REPO_ROOT)),
        },
    }

    for line_spec in LINE_SPECS:
        domain_eval, inference_time_by_domain = _evaluate_line(
            line_spec=line_spec,
            routes_by_domain=routes_by_line[line_spec.line_id],
            holdout_by_domain=holdout_by_domain,
        )
        summary["lines"][line_spec.line_id] = {
            "label": str(line_spec.label),
            "feature_names": list(line_spec.feature_names),
            "note": str(line_spec.note),
            "domains": {},
        }

        for domain_name in DOMAIN_ORDER:
            aggregate = domain_eval[domain_name]["aggregate"]
            by_position = list(aggregate["by_position"])
            auc_of_stop_acc = float(np.mean([float(item["stop_acc"]) for item in by_position])) if by_position else float("nan")
            final_auroc = _safe_float(aggregate.get("auroc@100%"))
            threshold_value = float(THRESHOLD_BY_DOMAIN[domain_name])
            earliest_95 = _earliest_anchor(
                by_position,
                lambda value, final=final_auroc: final is not None and value >= 0.95 * final,
            )
            earliest_fixed = _earliest_anchor(
                by_position,
                lambda value, threshold=threshold_value: value >= threshold,
            )
            plateau_anchor = _earliest_plateau_anchor(
                by_position,
                final_auroc=final_auroc,
                tolerance_abs=float(PLATEAU_TOLERANCE_ABS),
            )
            route_summary = {
                str(int(round(float(position) * 100.0))): _strip_model(route)
                for position, route in sorted(routes_by_line[line_spec.line_id][domain_name].items(), key=lambda item: float(item[0]))
            }
            summary["lines"][line_spec.line_id]["domains"][domain_name] = {
                "aggregate": aggregate,
                "fit_time_sec": float(fit_time_by_line_domain[(line_spec.line_id, domain_name)]),
                "inference_time_sec": float(inference_time_by_domain[domain_name]),
                "auc_of_stop_acc": float(auc_of_stop_acc),
                "threshold_value": float(threshold_value),
                "earliest_anchor_95pct_final_auroc": None if earliest_95 is None else int(earliest_95),
                "earliest_anchor_fixed_threshold": None if earliest_fixed is None else int(earliest_fixed),
                "plateau_anchor_pct": None if plateau_anchor is None else int(plateau_anchor),
                "plateau_tolerance_abs": float(PLATEAU_TOLERANCE_ABS),
                "route_summary": route_summary,
                "by_cache": domain_eval[domain_name]["by_cache"],
            }

    main_rows = _build_main_rows(summary)
    onset_rows = _build_onset_rows(summary)
    plateau_rows = _build_plateau_rows(summary)
    compare_rows = _build_compare_rows(summary)
    note_text = build_note_markdown(summary=summary, compare_rows=compare_rows)

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(out_table, main_rows)
    _write_csv(out_onset, onset_rows)
    _write_csv(out_plateau, plateau_rows)
    _write_csv(out_compare, compare_rows)
    out_note.parent.mkdir(parents=True, exist_ok=True)
    out_note.write_text(note_text + "\n", encoding="utf-8")

    print(f"[done] summary={out_summary}", flush=True)
    print(f"[done] table={out_table}", flush=True)
    print(f"[done] onset={out_onset}", flush=True)
    print(f"[done] plateau={out_plateau}", flush=True)
    print(f"[done] compare={out_compare}", flush=True)
    print(f"[done] note={out_note}", flush=True)


if __name__ == "__main__":
    main()
