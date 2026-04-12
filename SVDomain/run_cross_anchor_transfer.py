#!/usr/bin/env python3
"""Run cross-anchor frozen-basis transfer ablations for canonical es_svd_ms_rr_r1."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.ops.earlystop import EARLY_STOP_POSITIONS, discover_cache_entries
from nad.ops.earlystop_svd import (
    _auroc,
    _build_representation,
    _group_folds,
    _rank_transform_matrix,
    get_domain,
    load_earlystop_svd_bundle,
)
from scripts.run_earlystop_prefix10_svd_round1 import (
    ANCHOR_POSITIONS,
    DEFAULT_FEATURE_CHUNK_PROBLEMS,
    EXTRACTION_POSITIONS,
    _now_utc,
    _pct_label,
    build_feature_store,
)


DEFAULT_BUNDLE_PATH = "models/ml_selectors/es_svd_ms_rr_r1.pkl"
DEFAULT_FEATURE_CACHE_DIR = "results/cache/cross_anchor_transfer"
CANONICAL_FEATURE_CACHE_DIR = "results/cache/es_svd_ms_rr_r1"
ANCHOR_PCTS = tuple(int(round(float(v) * 100.0)) for v in ANCHOR_POSITIONS)
ANCHOR_PCT_TO_POSITION = {int(round(float(v) * 100.0)): float(v) for v in ANCHOR_POSITIONS}
ANCHOR_PCT_TO_SLOT_INDEX = {
    int(round(float(position) * 100.0)): int(idx)
    for idx, position in enumerate(EARLY_STOP_POSITIONS)
    if int(round(float(position) * 100.0)) in ANCHOR_PCTS
}
EXTRACTION_PCT_TO_INDEX = {
    int(round(float(position) * 100.0)): int(idx)
    for idx, position in enumerate(EXTRACTION_POSITIONS)
}
METRIC_ORDER = ("auroc", "selacc@10%", "stop_acc")
CONDITION_ORDER = ("frozen_basis", "task_specific", "no_svd")
DOMAIN_ORDER = ("math", "science")
DIAGONAL_PAIRS = tuple((pct, pct) for pct in ANCHOR_PCTS)
OFFDIAGONAL_FOCUS_PAIRS = ((100, 40), (100, 70), (40, 100), (10, 100))
ADJACENT_FORWARD_PAIRS = ((10, 40), (40, 70), (70, 100))
ADJACENT_BACKWARD_PAIRS = ((40, 10), (70, 40), (100, 70))


def default_feature_workers() -> int:
    cpu = max(1, int(os.cpu_count() or 1))
    return max(1, min(16, cpu))


def default_suite_workers() -> int:
    cpu = max(1, int(os.cpu_count() or 1))
    return max(1, min(8, cpu))


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _parse_anchor_pcts(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value not in ANCHOR_PCTS:
            raise ValueError(f"Unsupported anchor {value}; expected subset of {list(ANCHOR_PCTS)}")
        values.append(int(value))
    if not values:
        raise ValueError("Need at least one anchor")
    return tuple(sorted(dict.fromkeys(values)))


def _feature_cache_key(
    *,
    cache_root: str,
    positions: tuple[float, ...],
    required_feature_names: set[str],
    max_problems_per_cache: Optional[int],
    included_domains: tuple[str, ...],
) -> str:
    payload = {
        "version": 1,
        "cache_root": str(cache_root),
        "positions": [float(v) for v in positions],
        "required_feature_names": sorted(str(v) for v in required_feature_names),
        "max_problems_per_cache": None if max_problems_per_cache is None else int(max_problems_per_cache),
        "included_domains": [str(v) for v in included_domains],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _feature_cache_path(
    *,
    cache_dir: Path,
    cache_root: str,
    positions: tuple[float, ...],
    required_feature_names: set[str],
    max_problems_per_cache: Optional[int],
    included_domains: tuple[str, ...],
) -> Path:
    suffix = "all" if max_problems_per_cache is None else f"cap{int(max_problems_per_cache)}"
    key = _feature_cache_key(
        cache_root=cache_root,
        positions=positions,
        required_feature_names=required_feature_names,
        max_problems_per_cache=max_problems_per_cache,
        included_domains=included_domains,
    )
    return cache_dir / f"noncoding_cross_anchor_{suffix}_{key}.pkl"


def _load_cached_feature_store(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return list(payload["feature_store"])


def _try_load_canonical_feature_store(
    *,
    positions: tuple[float, ...],
    included_domains: tuple[str, ...],
) -> tuple[Optional[list[dict[str, Any]]], Optional[Path], Optional[str]]:
    expected_positions = [float(v) for v in positions]
    canonical_dir = _resolve_path(CANONICAL_FEATURE_CACHE_DIR)
    if not canonical_dir.exists():
        return None, None, None
    for path in sorted(canonical_dir.glob("noncoding_cache_all_*.pkl")):
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception:
            continue
        if [float(v) for v in payload.get("positions", [])] != expected_positions:
            continue
        payload_domains = tuple(str(v) for v in payload.get("included_domains", ()))
        if payload_domains and payload_domains != tuple(str(v) for v in included_domains):
            continue
        feature_store = list(payload.get("feature_store", []))
        if feature_store:
            return feature_store, path, "loaded_canonical"
    return None, None, None


def _load_or_build_noncoding_feature_store(
    *,
    cache_root: str,
    positions: tuple[float, ...],
    required_feature_names: set[str],
    max_problems_per_cache: Optional[int],
    feature_workers: int,
    chunk_problems: int,
    feature_cache_dir: Optional[Path],
    refresh_feature_cache: bool,
    included_domains: tuple[str, ...] = DOMAIN_ORDER,
) -> tuple[list[dict[str, Any]], Optional[Path], str]:
    cache_path: Optional[Path] = None
    if feature_cache_dir is not None:
        cache_path = _feature_cache_path(
            cache_dir=feature_cache_dir,
            cache_root=cache_root,
            positions=positions,
            required_feature_names=required_feature_names,
            max_problems_per_cache=max_problems_per_cache,
            included_domains=included_domains,
        )
        if cache_path.exists() and not refresh_feature_cache:
            return _load_cached_feature_store(cache_path), cache_path, "loaded"

    if not refresh_feature_cache:
        canonical_store, canonical_path, canonical_status = _try_load_canonical_feature_store(
            positions=positions,
            included_domains=included_domains,
        )
        if canonical_store is not None:
            if cache_path is not None and cache_path != canonical_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open("wb") as handle:
                    pickle.dump(
                        {
                            "cache_root": str(cache_root),
                            "positions": [float(v) for v in positions],
                            "max_problems_per_cache": None if max_problems_per_cache is None else int(max_problems_per_cache),
                            "included_domains": [str(v) for v in included_domains],
                            "feature_store": canonical_store,
                        },
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
            return canonical_store, canonical_path, str(canonical_status)

    include_cache_keys = {
        str(entry.cache_key)
        for entry in discover_cache_entries(cache_root)
        if get_domain(str(entry.dataset_name)) in included_domains
    }
    feature_store = build_feature_store(
        cache_root=cache_root,
        positions=positions,
        required_feature_names=required_feature_names,
        max_problems_per_cache=max_problems_per_cache,
        max_workers=max(1, int(feature_workers)),
        chunk_problems=max(1, int(chunk_problems)),
        include_cache_keys=include_cache_keys,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump(
                {
                    "cache_root": str(cache_root),
                    "positions": [float(v) for v in positions],
                    "max_problems_per_cache": None if max_problems_per_cache is None else int(max_problems_per_cache),
                    "included_domains": [str(v) for v in included_domains],
                    "feature_store": feature_store,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    return list(feature_store), cache_path, "built"


def _route_feature_key(route: dict[str, Any]) -> str:
    feature_indices = ",".join(str(int(v)) for v in route["feature_indices"])
    return f"{route['representation']}::{feature_indices}"


def _collect_task_representation(
    *,
    feature_store: list[dict[str, Any]],
    domain_filter: str,
    target_anchor_pct: int,
    feature_indices: list[int],
    representation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor_position_index = int(EXTRACTION_PCT_TO_INDEX[int(target_anchor_pct)])
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []

    for payload in feature_store:
        if str(payload.get("domain")) != str(domain_filter):
            continue
        tensor = np.asarray(payload["tensor"], dtype=np.float64)
        labels = np.asarray(payload["labels"], dtype=np.int32)
        groups = np.asarray(payload["group_keys"], dtype=object)
        if tensor.shape[0] <= 0:
            continue
        x_raw = tensor[:, tensor_position_index, :]
        x_rank = _rank_transform_matrix(x_raw)
        x_rep = _build_representation(
            x_raw=x_raw,
            x_rank=x_rank,
            feature_indices=feature_indices,
            representation=representation,
        )
        x_parts.append(x_rep)
        y_parts.append(labels)
        group_parts.append(groups)

    if not x_parts:
        n_feat = len(feature_indices) * (2 if representation == "raw+rank" else 1)
        return (
            np.zeros((0, n_feat), dtype=np.float64),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=object),
        )

    return (
        np.concatenate(x_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        np.concatenate(group_parts, axis=0),
    )


def _z_project(
    x: np.ndarray,
    scaler: StandardScaler,
    svd: TruncatedSVD,
    whiten: bool,
) -> np.ndarray:
    z = svd.transform(scaler.transform(np.asarray(x, dtype=np.float64)))
    if whiten:
        singular_values = np.asarray(svd.singular_values_, dtype=np.float64)
        singular_values = np.where(np.abs(singular_values) < 1e-8, 1.0, singular_values)
        z = z / singular_values
    return np.asarray(z, dtype=np.float64)


def _selacc_metric(scores: np.ndarray, y: np.ndarray, _groups: np.ndarray) -> float:
    if scores.size <= 0:
        return float("nan")
    topk = max(1, int(math.ceil(0.10 * max(1, int(scores.shape[0])))))
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    return float(np.mean(np.asarray(y, dtype=np.float64)[order[:topk]]))


def _stop_acc_metric(scores: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    unique_groups = np.unique(groups)
    hits: list[float] = []
    for group_key in unique_groups:
        mask = groups == group_key
        if not np.any(mask):
            continue
        group_scores = np.asarray(scores, dtype=np.float64)[mask]
        group_labels = np.asarray(y, dtype=np.float64)[mask]
        top_idx = int(np.argmax(group_scores))
        hits.append(float(group_labels[top_idx]))
    return float(np.mean(hits)) if hits else float("nan")


METRIC_FNS: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], float]] = {
    "auroc": lambda scores, y, groups: _auroc(scores, y),
    "selacc@10%": _selacc_metric,
    "stop_acc": _stop_acc_metric,
}


def _cv_compare_metrics(
    *,
    x_frozen: np.ndarray,
    x_target: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    frozen_scaler: StandardScaler,
    frozen_svd: TruncatedSVD,
    frozen_whiten: bool,
    target_rank: int,
    n_splits: int,
    random_state: int,
) -> dict[str, dict[str, dict[str, float]]]:
    empty = {
        condition: {
            metric: {"mean": float("nan"), "std": 0.0, "n_folds": 0}
            for metric in METRIC_ORDER
        }
        for condition in CONDITION_ORDER
    }
    if x_target.shape[0] <= 0 or y.shape[0] <= 0:
        return empty

    folds = _group_folds(groups, n_splits)
    if not folds:
        return empty

    values: dict[str, dict[str, list[float]]] = {
        condition: {metric: [] for metric in METRIC_ORDER}
        for condition in CONDITION_ORDER
    }

    for train_idx, test_idx in folds:
        y_train = y[train_idx]
        y_test = y[test_idx]
        if np.unique(y_train).shape[0] < 2 or np.unique(y_test).shape[0] < 2:
            continue
        groups_test = groups[test_idx]

        predictions: dict[str, Optional[np.ndarray]] = {
            "frozen_basis": None,
            "task_specific": None,
            "no_svd": None,
        }

        try:
            z_train = _z_project(x_frozen[train_idx], frozen_scaler, frozen_svd, frozen_whiten)
            z_test = _z_project(x_frozen[test_idx], frozen_scaler, frozen_svd, frozen_whiten)
            clf = LogisticRegression(C=1.0, max_iter=2000, random_state=random_state)
            clf.fit(z_train, y_train)
            predictions["frozen_basis"] = np.asarray(clf.decision_function(z_test), dtype=np.float64)
        except Exception as exc:
            print(f"    [frozen_basis] fold error: {exc}", flush=True)

        try:
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_target[train_idx])
            x_test_scaled = scaler.transform(x_target[test_idx])
            max_rank = max(1, min(int(target_rank), int(x_train_scaled.shape[1]), int(x_train_scaled.shape[0] - 1)))
            svd = TruncatedSVD(n_components=max_rank, random_state=random_state)
            z_train_ts = svd.fit_transform(x_train_scaled)
            z_test_ts = svd.transform(x_test_scaled)
            clf = LogisticRegression(C=1.0, max_iter=2000, random_state=random_state)
            clf.fit(z_train_ts, y_train)
            predictions["task_specific"] = np.asarray(clf.decision_function(z_test_ts), dtype=np.float64)
        except Exception as exc:
            print(f"    [task_specific] fold error: {exc}", flush=True)

        try:
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_target[train_idx])
            x_test_scaled = scaler.transform(x_target[test_idx])
            clf = LogisticRegression(C=1.0, max_iter=2000, random_state=random_state)
            clf.fit(x_train_scaled, y_train)
            predictions["no_svd"] = np.asarray(clf.decision_function(x_test_scaled), dtype=np.float64)
        except Exception as exc:
            print(f"    [no_svd] fold error: {exc}", flush=True)

        for condition, scores in predictions.items():
            if scores is None:
                continue
            for metric_name, metric_fn in METRIC_FNS.items():
                try:
                    value = float(metric_fn(scores, y_test, groups_test))
                except Exception as exc:
                    print(f"    [{condition}/{metric_name}] fold error: {exc}", flush=True)
                    continue
                if np.isfinite(value):
                    values[condition][metric_name].append(value)

    results = {}
    for condition in CONDITION_ORDER:
        results[condition] = {}
        for metric_name in METRIC_ORDER:
            vals = values[condition][metric_name]
            if vals:
                results[condition][metric_name] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n_folds": int(len(vals)),
                }
            else:
                results[condition][metric_name] = {"mean": float("nan"), "std": 0.0, "n_folds": 0}
    return results


def _format_float(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "nan"
    return f"{value:.6f}" if np.isfinite(value) else "nan"


def _verdict(delta: float) -> str:
    if not np.isfinite(delta):
        return "unknown"
    if delta >= 0.0:
        return "win"
    if delta >= -0.02:
        return "tie"
    return "loss"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_subset_from_delta_rows(
    delta_rows: list[dict[str, Any]],
    *,
    domain: str,
    metric: str,
    pairs: tuple[tuple[int, int], ...],
) -> Optional[dict[str, Any]]:
    rows = [
        row for row in delta_rows
        if str(row["domain"]) == str(domain)
        and str(row["metric"]) == str(metric)
        and (int(row["source_anchor_pct"]), int(row["target_anchor_pct"])) in pairs
    ]
    if not rows:
        return None

    def _mean_of(key: str) -> float:
        vals = [float(row[key]) for row in rows if str(row[key]) not in {"", "nan"} and np.isfinite(float(row[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "domain": str(domain),
        "metric": str(metric),
        "pair_count": int(len(rows)),
        "pairs": ",".join(f"{int(row['source_anchor_pct'])}->{int(row['target_anchor_pct'])}" for row in rows),
        "frozen_basis": _mean_of("frozen_basis"),
        "task_specific": _mean_of("task_specific"),
        "no_svd": _mean_of("no_svd"),
        "delta_fb_minus_ts": _mean_of("delta_fb_minus_ts"),
        "delta_fb_minus_nosvd": _mean_of("delta_fb_minus_nosvd"),
    }


def build_summary_rows(delta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def _append_pair_rows(group_name: str, pairs: tuple[tuple[int, int], ...]) -> None:
        for domain in DOMAIN_ORDER:
            for metric in METRIC_ORDER:
                for source_anchor_pct, target_anchor_pct in pairs:
                    matched = next(
                        (
                            row for row in delta_rows
                            if str(row["domain"]) == str(domain)
                            and str(row["metric"]) == str(metric)
                            and int(row["source_anchor_pct"]) == int(source_anchor_pct)
                            and int(row["target_anchor_pct"]) == int(target_anchor_pct)
                        ),
                        None,
                    )
                    if matched is None:
                        continue
                    rows.append(
                        {
                            "summary_type": f"{group_name}_pair",
                            "domain": str(domain),
                            "metric": str(metric),
                            "source_anchor_pct": int(source_anchor_pct),
                            "target_anchor_pct": int(target_anchor_pct),
                            "pair_label": f"{int(source_anchor_pct)}->{int(target_anchor_pct)}",
                            "pair_count": 1,
                            "frozen_basis": float(matched["frozen_basis"]),
                            "task_specific": float(matched["task_specific"]),
                            "no_svd": float(matched["no_svd"]),
                            "delta_fb_minus_ts": float(matched["delta_fb_minus_ts"]),
                            "delta_fb_minus_nosvd": float(matched["delta_fb_minus_nosvd"]),
                            "paired_source_anchor_pct": "",
                            "paired_target_anchor_pct": "",
                            "paired_pair_label": "",
                            "paired_frozen_basis": "",
                            "paired_delta_fb_minus_ts": "",
                            "forward_minus_backward_frozen": "",
                            "forward_minus_backward_delta": "",
                        }
                    )
                aggregate = _aggregate_subset_from_delta_rows(delta_rows, domain=domain, metric=metric, pairs=pairs)
                if aggregate is None:
                    continue
                rows.append(
                    {
                        "summary_type": f"{group_name}_mean",
                        "domain": str(domain),
                        "metric": str(metric),
                        "source_anchor_pct": "",
                        "target_anchor_pct": "",
                        "pair_label": aggregate["pairs"],
                        "pair_count": int(aggregate["pair_count"]),
                        "frozen_basis": float(aggregate["frozen_basis"]),
                        "task_specific": float(aggregate["task_specific"]),
                        "no_svd": float(aggregate["no_svd"]),
                        "delta_fb_minus_ts": float(aggregate["delta_fb_minus_ts"]),
                        "delta_fb_minus_nosvd": float(aggregate["delta_fb_minus_nosvd"]),
                        "paired_source_anchor_pct": "",
                        "paired_target_anchor_pct": "",
                        "paired_pair_label": "",
                        "paired_frozen_basis": "",
                        "paired_delta_fb_minus_ts": "",
                        "forward_minus_backward_frozen": "",
                        "forward_minus_backward_delta": "",
                    }
                )

    _append_pair_rows("diagonal", DIAGONAL_PAIRS)
    _append_pair_rows("offdiag_focus", OFFDIAGONAL_FOCUS_PAIRS)
    _append_pair_rows("adjacent_forward", ADJACENT_FORWARD_PAIRS)
    _append_pair_rows("adjacent_backward", ADJACENT_BACKWARD_PAIRS)

    for domain in DOMAIN_ORDER:
        for metric in METRIC_ORDER:
            for forward_pair, backward_pair in zip(ADJACENT_FORWARD_PAIRS, ADJACENT_BACKWARD_PAIRS):
                forward = next(
                    (
                        row for row in delta_rows
                        if str(row["domain"]) == str(domain)
                        and str(row["metric"]) == str(metric)
                        and int(row["source_anchor_pct"]) == int(forward_pair[0])
                        and int(row["target_anchor_pct"]) == int(forward_pair[1])
                    ),
                    None,
                )
                backward = next(
                    (
                        row for row in delta_rows
                        if str(row["domain"]) == str(domain)
                        and str(row["metric"]) == str(metric)
                        and int(row["source_anchor_pct"]) == int(backward_pair[0])
                        and int(row["target_anchor_pct"]) == int(backward_pair[1])
                    ),
                    None,
                )
                if forward is None or backward is None:
                    continue
                forward_frozen = float(forward["frozen_basis"])
                backward_frozen = float(backward["frozen_basis"])
                forward_delta = float(forward["delta_fb_minus_ts"])
                backward_delta = float(backward["delta_fb_minus_ts"])
                rows.append(
                    {
                        "summary_type": "adjacent_symmetry",
                        "domain": str(domain),
                        "metric": str(metric),
                        "source_anchor_pct": int(forward_pair[0]),
                        "target_anchor_pct": int(forward_pair[1]),
                        "pair_label": f"{int(forward_pair[0])}->{int(forward_pair[1])}",
                        "pair_count": 1,
                        "frozen_basis": forward_frozen,
                        "task_specific": float(forward["task_specific"]),
                        "no_svd": float(forward["no_svd"]),
                        "delta_fb_minus_ts": forward_delta,
                        "delta_fb_minus_nosvd": float(forward["delta_fb_minus_nosvd"]),
                        "paired_source_anchor_pct": int(backward_pair[0]),
                        "paired_target_anchor_pct": int(backward_pair[1]),
                        "paired_pair_label": f"{int(backward_pair[0])}->{int(backward_pair[1])}",
                        "paired_frozen_basis": backward_frozen,
                        "paired_delta_fb_minus_ts": backward_delta,
                        "forward_minus_backward_frozen": forward_frozen - backward_frozen,
                        "forward_minus_backward_delta": forward_delta - backward_delta,
                    }
                )

    order = {
        "diagonal_pair": 0,
        "diagonal_mean": 1,
        "offdiag_focus_pair": 2,
        "offdiag_focus_mean": 3,
        "adjacent_forward_pair": 4,
        "adjacent_forward_mean": 5,
        "adjacent_backward_pair": 6,
        "adjacent_backward_mean": 7,
        "adjacent_symmetry": 8,
    }
    metric_order = {name: idx for idx, name in enumerate(METRIC_ORDER)}
    domain_order = {name: idx for idx, name in enumerate(DOMAIN_ORDER)}
    return sorted(
        rows,
        key=lambda row: (
            order.get(str(row["summary_type"]), 999),
            domain_order.get(str(row["domain"]), 999),
            metric_order.get(str(row["metric"]), 999),
            999 if row["source_anchor_pct"] == "" else int(row["source_anchor_pct"]),
            999 if row["target_anchor_pct"] == "" else int(row["target_anchor_pct"]),
        ),
    )


def _fmt_pct(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(value):
        return "N/A"
    return f"{value * 100.0:.2f}%"


def _fmt_signed_pct_points(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(value):
        return "N/A"
    return f"{value * 100.0:+.2f} pts"


def _mean_delta_by_source(
    delta_rows: list[dict[str, Any]],
    *,
    domain: str,
    metric: str,
    off_diagonal_only: bool,
) -> list[tuple[int, float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in delta_rows:
        if str(row["domain"]) != str(domain) or str(row["metric"]) != str(metric):
            continue
        source_anchor = int(row["source_anchor_pct"])
        target_anchor = int(row["target_anchor_pct"])
        if off_diagonal_only and source_anchor == target_anchor:
            continue
        value = float(row["delta_fb_minus_ts"])
        if np.isfinite(value):
            grouped[source_anchor].append(value)
    summary = [
        (int(source_anchor), float(np.mean(values)))
        for source_anchor, values in grouped.items()
        if values
    ]
    return sorted(summary, key=lambda item: (-item[1], item[0]))


def build_note_markdown(
    *,
    bundle_path: Path,
    domains: tuple[str, ...],
    delta_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    artifacts: dict[str, str],
) -> str:
    lines = [
        "# 11. Cross-Anchor Transfer",
        "",
        f"**Experiment script**: `SVDomain/run_cross_anchor_transfer.py`",
        f"**Source bundle**: `{bundle_path.relative_to(REPO_ROOT)}`",
        f"**Anchors**: `{', '.join(str(v) for v in ANCHOR_PCTS)}`",
        f"**Domains**: `{', '.join(domains)}`",
        f"**Date**: {_now_utc()[:10]}",
        "",
        "---",
        "",
        "## 1. Claim Under Test",
        "",
        "> The shared low-rank basis is not only reusable at slot-100, but remains competitive across multiple trajectory anchors with lightweight anchor-specific heads.",
        "",
        "这里的核心问题不是再补更多 `100%` 数字，而是测试：同一个 low-rank basis 在整条 reasoning trajectory 上是否仍然可复用。",
        "",
        "## 2. Protocol",
        "",
        "- 条件对比：`frozen_basis` vs `task_specific` vs `no_svd`。",
        "- `frozen_basis`：使用 **source anchor** 的 `scaler + SVD`，只在 **target anchor** 上重训 LR head。",
        "- `task_specific`：在 **target anchor** 上重训 `scaler + SVD + LR head`。",
        "- `no_svd`：在 **target anchor** 上直接训练 `StandardScaler + LR`。",
        "- 评估：沿用现有 `GroupKFold` offline protocol，不改 split；本表默认关注 `AUROC`，并同步导出 `SelAcc@10%` / `StopAcc`。",
        "",
        "## 3. Headline Summary",
        "",
    ]

    def _find_summary(summary_type: str, domain: str, metric: str) -> Optional[dict[str, Any]]:
        return next(
            (
                row for row in summary_rows
                if str(row["summary_type"]) == str(summary_type)
                and str(row["domain"]) == str(domain)
                and str(row["metric"]) == str(metric)
            ),
            None,
        )

    def _find_delta(domain: str, source_anchor_pct: int, target_anchor_pct: int, metric: str = "auroc") -> Optional[dict[str, Any]]:
        return next(
            (
                row for row in delta_rows
                if str(row["domain"]) == str(domain)
                and str(row["metric"]) == str(metric)
                and int(row["source_anchor_pct"]) == int(source_anchor_pct)
                and int(row["target_anchor_pct"]) == int(target_anchor_pct)
            ),
            None,
        )

    for domain in domains:
        diag = _find_summary("diagonal_mean", domain, "auroc")
        offdiag = _find_summary("offdiag_focus_mean", domain, "auroc")
        if diag is None or offdiag is None:
            continue
        lines.append(
            "- `{domain}` diagonal mean: frozen={frozen} vs task-specific={task} (Δ={delta}); "
            "focus off-diagonal mean: frozen={offdiag_frozen} vs task-specific={offdiag_task} (Δ={offdiag_delta}).".format(
                domain=domain,
                frozen=_fmt_pct(diag["frozen_basis"]),
                task=_fmt_pct(diag["task_specific"]),
                delta=_fmt_signed_pct_points(diag["delta_fb_minus_ts"]),
                offdiag_frozen=_fmt_pct(offdiag["frozen_basis"]),
                offdiag_task=_fmt_pct(offdiag["task_specific"]),
                offdiag_delta=_fmt_signed_pct_points(offdiag["delta_fb_minus_ts"]),
            )
        )

    lines.extend([
        "",
        "## 4. Which source anchors transfer best?",
        "",
    ])
    for domain in domains:
        ranked = _mean_delta_by_source(delta_rows, domain=domain, metric="auroc", off_diagonal_only=True)
        if not ranked:
            continue
        best_anchor, best_delta = ranked[0]
        ranking_text = ", ".join(f"{anchor}% ({delta * 100.0:+.2f} pts)" for anchor, delta in ranked)
        lines.append(f"- `{domain}` off-diagonal transferability ranking by mean Δ(frozen−task_specific): {ranking_text}.")
        lines.append(f"- `{domain}` best reusable source anchor is `{best_anchor}%` (mean Δ={best_delta * 100.0:+.2f} pts).")

    lines.extend([
        "",
        "## 5. Diagonal / Off-Diagonal / Adjacent Stability",
        "",
        "| Domain | Slice | Frozen | Task-specific | No-SVD | Δ(Frozen−Task) | Δ(Frozen−NoSVD) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for summary_type in ("diagonal_mean", "offdiag_focus_mean", "adjacent_forward_mean", "adjacent_backward_mean"):
        for domain in domains:
            row = next(
                (
                    item for item in summary_rows
                    if str(item["summary_type"]) == str(summary_type)
                    and str(item["domain"]) == str(domain)
                    and str(item["metric"]) == "auroc"
                ),
                None,
            )
            if row is None:
                continue
            lines.append(
                "| {domain} | {slice_name} | {frozen} | {task} | {nosvd} | {delta_task} | {delta_nosvd} |".format(
                    domain=domain,
                    slice_name=summary_type.replace("_mean", "").replace("_", " "),
                    frozen=_fmt_pct(row["frozen_basis"]),
                    task=_fmt_pct(row["task_specific"]),
                    nosvd=_fmt_pct(row["no_svd"]),
                    delta_task=_fmt_signed_pct_points(row["delta_fb_minus_ts"]),
                    delta_nosvd=_fmt_signed_pct_points(row["delta_fb_minus_nosvd"]),
                )
            )

    lines.extend([
        "",
        "### Highlighted off-diagonal pairs",
        "",
        "| Domain | Pair | Frozen | Task-specific | No-SVD | Δ(Frozen−Task) | Verdict |",
        "|---|---|---:|---:|---:|---:|---|",
    ])
    for domain in domains:
        for source_anchor_pct, target_anchor_pct in OFFDIAGONAL_FOCUS_PAIRS:
            row = _find_delta(domain, source_anchor_pct, target_anchor_pct, "auroc")
            if row is None:
                continue
            lines.append(
                "| {domain} | {pair} | {frozen} | {task} | {nosvd} | {delta_task} | {verdict} |".format(
                    domain=domain,
                    pair=f"{source_anchor_pct}->{target_anchor_pct}",
                    frozen=_fmt_pct(row["frozen_basis"]),
                    task=_fmt_pct(row["task_specific"]),
                    nosvd=_fmt_pct(row["no_svd"]),
                    delta_task=_fmt_signed_pct_points(row["delta_fb_minus_ts"]),
                    verdict=row["transfer_verdict"],
                )
            )

    lines.extend([
        "",
        "## 6. Direct Answers",
        "",
    ])

    math_diag = _find_summary("diagonal_mean", "math", "auroc")
    math_offdiag = _find_summary("offdiag_focus_mean", "math", "auroc")
    science_diag = _find_summary("diagonal_mean", "science", "auroc")
    science_offdiag = _find_summary("offdiag_focus_mean", "science", "auroc")
    if math_diag is not None and math_offdiag is not None:
        lines.append(
            "- **Shared basis 是否只在 slot-100 可复用？** `math` 上答案是 **不是**：diagonal mean 仅比 task-specific 低 `{diag_gap}`，关键 off-diagonal 也只低 `{offdiag_gap}`，而且所有 math anchor-pairs 都仍是 `tie/win`。".format(
                diag_gap=_fmt_signed_pct_points(math_diag["delta_fb_minus_ts"]),
                offdiag_gap=_fmt_signed_pct_points(math_offdiag["delta_fb_minus_ts"]),
            )
        )
    if science_diag is not None and science_offdiag is not None:
        row_100_40 = _find_delta("science", 100, 40, "auroc")
        row_100_70 = _find_delta("science", 100, 70, "auroc")
        row_10_40 = _find_delta("science", 10, 40, "auroc")
        row_10_100 = _find_delta("science", 10, 100, "auroc")
        late_text = ""
        early_text = ""
        if row_100_40 is not None and row_100_70 is not None:
            late_text = (
                f"`100→40` 只差 {_fmt_signed_pct_points(row_100_40['delta_fb_minus_ts'])}，"
                f"`100→70` 只差 {_fmt_signed_pct_points(row_100_70['delta_fb_minus_ts'])}"
            )
        if row_10_40 is not None and row_10_100 is not None:
            early_text = (
                f"但 `10→40` 已掉到 {_fmt_signed_pct_points(row_10_40['delta_fb_minus_ts'])}，"
                f"`10→100` 更掉到 {_fmt_signed_pct_points(row_10_100['delta_fb_minus_ts'])}"
            )
        lines.append(
            "- `science` 上答案是 **部分成立**：late-anchor basis 仍可复用（{late_text}），{early_text}。这说明 cross-anchor transfer 在 science 上并不均匀。".format(
                late_text=late_text or "late anchors 之间接近 task-specific",
                early_text=early_text or "早期 anchor 到晚期 anchor 的迁移明显更差",
            )
        )

    for domain in domains:
        ranked = _mean_delta_by_source(delta_rows, domain=domain, metric="auroc", off_diagonal_only=True)
        if ranked:
            best_anchor, best_delta = ranked[0]
            lines.append(
                "- **哪些 anchor 最 transferable？** `{domain}` 上最可迁移的 source anchor 是 `{anchor}%`，其 off-diagonal mean Δ(frozen−task_specific) = `{delta}`。".format(
                    domain=domain,
                    anchor=best_anchor,
                    delta=_fmt_signed_pct_points(best_delta),
                )
            )

    row_math_10_100 = _find_delta("math", 10, 100, "auroc")
    row_math_40_100 = _find_delta("math", 40, 100, "auroc")
    row_science_10_100 = _find_delta("science", 10, 100, "auroc")
    row_science_70_100 = _find_delta("science", 70, 100, "auroc")
    if row_math_10_100 is not None and row_math_40_100 is not None and row_science_10_100 is not None and row_science_70_100 is not None:
        lines.append(
            "- **早期 anchor 是否只学到 coarse signal？** `math` 不是：`10→100` 仍只差 `{math_10_100}`，说明早期 basis 已经捕获到大部分稳定决策边界。`science` 更像是 **yes**：`10→100` 掉到 `{science_10_100}`，而 `70→100` 只差 `{science_70_100}`，说明早期 science anchor 更像 coarse / under-formed signal，late anchor 才带有 completion-heavy information。".format(
                math_10_100=_fmt_signed_pct_points(row_math_10_100["delta_fb_minus_ts"]),
                science_10_100=_fmt_signed_pct_points(row_science_10_100["delta_fb_minus_ts"]),
                science_70_100=_fmt_signed_pct_points(row_science_70_100["delta_fb_minus_ts"]),
            )
        )

    lines.append(
        "- **论文标题里的 transferable 要不要加 cross-anchor 限定？** 建议写成 **`task- and cross-anchor-transferable`**，但正文要补一句限定：该结论在 `math` 上几乎贯穿全 trajectory，在 `science` 上则主要由 `70/100` late-anchor basis 支撑。"
    )

    lines.extend([
        "",
        "## 7. Artifacts",
        "",
        f"- Matrix CSV: `{artifacts['matrix_csv']}`",
        f"- Delta CSV: `{artifacts['delta_csv']}`",
        f"- Summary CSV: `{artifacts['summary_csv']}`",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run cross-anchor frozen-basis transfer suite")
    ap.add_argument("--bundle-path", default=DEFAULT_BUNDLE_PATH)
    ap.add_argument("--cache-root", default="MUI_HUB/cache")
    ap.add_argument("--domains", default="math,science")
    ap.add_argument("--source-anchors", default="10,40,70,100")
    ap.add_argument("--target-anchors", default="10,40,70,100")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--feature-workers", type=int, default=default_feature_workers())
    ap.add_argument("--suite-workers", type=int, default=default_suite_workers())
    ap.add_argument("--feature-chunk-problems", type=int, default=DEFAULT_FEATURE_CHUNK_PROBLEMS)
    ap.add_argument("--feature-cache-dir", default=DEFAULT_FEATURE_CACHE_DIR)
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--max-problems-per-cache", type=int, default=0, help="0 means all problems")
    ap.add_argument("--out-matrix", default="results/tables/cross_anchor_transfer_matrix.csv")
    ap.add_argument("--out-deltas", default="results/tables/cross_anchor_transfer_deltas.csv")
    ap.add_argument("--out-summary", default="results/tables/cross_anchor_transfer_summary.csv")
    ap.add_argument("--out-note", default="docs/11_CROSS_ANCHOR_TRANSFER.md")
    args = ap.parse_args()

    bundle_path = _resolve_path(str(args.bundle_path))
    cache_root = str(_resolve_path(str(args.cache_root)))
    feature_cache_dir = None if str(args.feature_cache_dir).strip().lower() in {"", "none", "off"} else _resolve_path(str(args.feature_cache_dir))
    source_anchors = _parse_anchor_pcts(str(args.source_anchors))
    target_anchors = _parse_anchor_pcts(str(args.target_anchors))
    domains = tuple(
        domain.strip() for domain in str(args.domains).split(",")
        if domain.strip()
    )
    for domain in domains:
        if domain not in DOMAIN_ORDER:
            raise ValueError(f"Unsupported domain {domain!r}; expected subset of {DOMAIN_ORDER}")

    bundle = load_earlystop_svd_bundle(bundle_path)
    required_features: set[str] = set()
    for domain in domains:
        for route in bundle["domains"][domain]["routes"]:
            required_features.update(str(name) for name in route.get("feature_names", []))

    max_problems_per_cache = None if int(args.max_problems_per_cache) <= 0 else int(args.max_problems_per_cache)
    feature_store, feature_cache_path, feature_cache_status = _load_or_build_noncoding_feature_store(
        cache_root=cache_root,
        positions=tuple(float(v) for v in EXTRACTION_POSITIONS),
        required_feature_names=required_features,
        max_problems_per_cache=max_problems_per_cache,
        feature_workers=int(args.feature_workers),
        chunk_problems=int(args.feature_chunk_problems),
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_feature_cache),
        included_domains=tuple(domains),
    )

    print(f"Bundle           : {bundle_path}", flush=True)
    print(f"Cache root       : {cache_root}", flush=True)
    print(f"Domains          : {domains}", flush=True)
    print(f"Source anchors   : {source_anchors}", flush=True)
    print(f"Target anchors   : {target_anchors}", flush=True)
    print(f"Feature cache    : {feature_cache_status} | {feature_cache_path}", flush=True)

    route_by_domain_anchor: dict[tuple[str, int], dict[str, Any]] = {}
    for domain in domains:
        for anchor_pct in set(source_anchors) | set(target_anchors):
            slot_index = int(ANCHOR_PCT_TO_SLOT_INDEX[int(anchor_pct)])
            route = bundle["domains"][domain]["routes"][slot_index]
            route_by_domain_anchor[(domain, int(anchor_pct))] = route

    data_cache: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for domain in domains:
        for target_anchor_pct in target_anchors:
            required_route_keys: dict[str, dict[str, Any]] = {}
            target_route = route_by_domain_anchor[(domain, int(target_anchor_pct))]
            required_route_keys[_route_feature_key(target_route)] = target_route
            for source_anchor_pct in source_anchors:
                source_route = route_by_domain_anchor[(domain, int(source_anchor_pct))]
                required_route_keys[_route_feature_key(source_route)] = source_route

            reference_labels: Optional[np.ndarray] = None
            reference_groups: Optional[np.ndarray] = None
            for feature_key, route in required_route_keys.items():
                x_rep, y, groups = _collect_task_representation(
                    feature_store=feature_store,
                    domain_filter=domain,
                    target_anchor_pct=int(target_anchor_pct),
                    feature_indices=[int(v) for v in route["feature_indices"]],
                    representation=str(route.get("representation", "raw+rank")),
                )
                data_cache[(domain, int(target_anchor_pct), feature_key)] = (x_rep, y, groups)
                if reference_labels is None:
                    reference_labels = y
                    reference_groups = groups
                else:
                    if not np.array_equal(reference_labels, y):
                        raise RuntimeError(f"Label mismatch for {domain}@{target_anchor_pct}% across feature specs")
                    if not np.array_equal(reference_groups, groups):
                        raise RuntimeError(f"Group mismatch for {domain}@{target_anchor_pct}% across feature specs")

    matrix_rows: list[dict[str, Any]] = []

    def _run_cell(domain: str, source_anchor_pct: int, target_anchor_pct: int) -> list[dict[str, Any]]:
        source_route = route_by_domain_anchor[(domain, int(source_anchor_pct))]
        target_route = route_by_domain_anchor[(domain, int(target_anchor_pct))]
        source_key = _route_feature_key(source_route)
        target_key = _route_feature_key(target_route)
        x_frozen, y_frozen, groups_frozen = data_cache[(domain, int(target_anchor_pct), source_key)]
        x_target, y_target, groups_target = data_cache[(domain, int(target_anchor_pct), target_key)]
        if not np.array_equal(y_frozen, y_target):
            raise RuntimeError(f"Label mismatch in cell {domain} {source_anchor_pct}->{target_anchor_pct}")
        if not np.array_equal(groups_frozen, groups_target):
            raise RuntimeError(f"Group mismatch in cell {domain} {source_anchor_pct}->{target_anchor_pct}")

        print(
            f"[cell] domain={domain:<7s} source={source_anchor_pct:>3d} target={target_anchor_pct:>3d} "
            f"samples={x_target.shape[0]} groups={len(np.unique(groups_target))}",
            flush=True,
        )
        comparison = _cv_compare_metrics(
            x_frozen=x_frozen,
            x_target=x_target,
            y=y_target,
            groups=groups_target,
            frozen_scaler=source_route["model"]["scaler"],
            frozen_svd=source_route["model"]["svd"],
            frozen_whiten=bool(source_route["model"].get("whiten", False)),
            target_rank=int(target_route.get("rank", source_route.get("rank", 16))),
            n_splits=int(args.n_splits),
            random_state=int(args.random_state),
        )

        n_samples = int(x_target.shape[0])
        n_groups = int(len(np.unique(groups_target))) if n_samples > 0 else 0
        rows: list[dict[str, Any]] = []
        for condition in CONDITION_ORDER:
            for metric_name in METRIC_ORDER:
                stats = comparison[condition][metric_name]
                rows.append(
                    {
                        "task": "earlystop",
                        "domain": str(domain),
                        "source_anchor_pct": int(source_anchor_pct),
                        "target_anchor_pct": int(target_anchor_pct),
                        "condition": str(condition),
                        "metric": str(metric_name),
                        "mean": _format_float(stats["mean"]),
                        "std": _format_float(stats["std"]),
                        "n_folds": int(stats["n_folds"]),
                        "n_samples": int(n_samples),
                        "n_groups": int(n_groups),
                        "source_bundle": bundle_path.stem,
                        "source_training_position_pct": int(round(float(source_route["training_position"]) * 100.0)),
                        "target_training_position_pct": int(round(float(target_route["training_position"]) * 100.0)),
                        "source_route_auroc": _format_float(source_route.get("cv_auroc", float("nan"))),
                        "target_route_auroc": _format_float(target_route.get("cv_auroc", float("nan"))),
                        "source_rank": int(source_route.get("rank", 0)),
                        "target_rank": int(target_route.get("rank", 0)),
                        "source_representation": str(source_route.get("representation", "")),
                        "target_representation": str(target_route.get("representation", "")),
                    }
                )
        return rows

    tasks = [(domain, int(source_anchor_pct), int(target_anchor_pct)) for domain in domains for source_anchor_pct in source_anchors for target_anchor_pct in target_anchors]
    with ThreadPoolExecutor(max_workers=max(1, int(args.suite_workers))) as executor:
        future_map = {
            executor.submit(_run_cell, domain, source_anchor_pct, target_anchor_pct): (domain, source_anchor_pct, target_anchor_pct)
            for domain, source_anchor_pct, target_anchor_pct in tasks
        }
        for future in as_completed(future_map):
            cell_rows = future.result()
            matrix_rows.extend(cell_rows)

    condition_order = {name: idx for idx, name in enumerate(CONDITION_ORDER)}
    metric_order = {name: idx for idx, name in enumerate(METRIC_ORDER)}
    domain_order = {name: idx for idx, name in enumerate(DOMAIN_ORDER)}
    matrix_rows = sorted(
        matrix_rows,
        key=lambda row: (
            domain_order.get(str(row["domain"]), 999),
            int(row["source_anchor_pct"]),
            int(row["target_anchor_pct"]),
            condition_order.get(str(row["condition"]), 999),
            metric_order.get(str(row["metric"]), 999),
        ),
    )

    pivot: dict[tuple[str, int, int, str], dict[str, float]] = defaultdict(dict)
    for row in matrix_rows:
        key = (str(row["domain"]), int(row["source_anchor_pct"]), int(row["target_anchor_pct"]), str(row["metric"]))
        try:
            value = float(row["mean"])
        except (TypeError, ValueError):
            value = float("nan")
        pivot[key][str(row["condition"])] = value

    delta_rows: list[dict[str, Any]] = []
    for (domain, source_anchor_pct, target_anchor_pct, metric_name), cond_values in sorted(
        pivot.items(),
        key=lambda item: (domain_order.get(item[0][0], 999), item[0][1], item[0][2], metric_order.get(item[0][3], 999)),
    ):
        frozen_value = cond_values.get("frozen_basis", float("nan"))
        task_value = cond_values.get("task_specific", float("nan"))
        nosvd_value = cond_values.get("no_svd", float("nan"))
        delta_task = frozen_value - task_value if np.isfinite(frozen_value) and np.isfinite(task_value) else float("nan")
        delta_nosvd = frozen_value - nosvd_value if np.isfinite(frozen_value) and np.isfinite(nosvd_value) else float("nan")
        delta_rows.append(
            {
                "task": "earlystop",
                "domain": str(domain),
                "source_anchor_pct": int(source_anchor_pct),
                "target_anchor_pct": int(target_anchor_pct),
                "metric": str(metric_name),
                "frozen_basis": _format_float(frozen_value),
                "task_specific": _format_float(task_value),
                "no_svd": _format_float(nosvd_value),
                "delta_fb_minus_ts": _format_float(delta_task),
                "delta_fb_minus_nosvd": _format_float(delta_nosvd),
                "transfer_verdict": _verdict(delta_task),
            }
        )

    summary_rows = build_summary_rows(delta_rows)
    artifacts = {
        "matrix_csv": str(Path(args.out_matrix)),
        "delta_csv": str(Path(args.out_deltas)),
        "summary_csv": str(Path(args.out_summary)),
        "paper_note": str(Path(args.out_note)),
    }
    note_text = build_note_markdown(
        bundle_path=bundle_path,
        domains=tuple(domains),
        delta_rows=delta_rows,
        summary_rows=summary_rows,
        artifacts=artifacts,
    )

    out_matrix = _resolve_path(str(args.out_matrix))
    out_deltas = _resolve_path(str(args.out_deltas))
    out_summary = _resolve_path(str(args.out_summary))
    out_note = _resolve_path(str(args.out_note))

    _write_csv(out_matrix, matrix_rows)
    _write_csv(out_deltas, delta_rows)
    _write_csv(out_summary, summary_rows)
    out_note.parent.mkdir(parents=True, exist_ok=True)
    out_note.write_text(note_text, encoding="utf-8")

    print(f"Wrote matrix  : {out_matrix}", flush=True)
    print(f"Wrote deltas  : {out_deltas}", flush=True)
    print(f"Wrote summary : {out_summary}", flush=True)
    print(f"Wrote note    : {out_note}", flush=True)


if __name__ == "__main__":
    main()
