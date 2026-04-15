#!/usr/bin/env python3
"""Train shallow MLP baselines for EarlyStop domains inside NAD_Next."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from nad.ops.earlystop import EARLY_STOP_POSITIONS, _problem_sort_key
from nad.ops.earlystop_svd import (
    AVAILABILITY_FEATURES,
    LEGACY_FULL_FEATURE_NAMES,
    TOKEN_FEATURES,
    TRAJ_FEATURES,
    _auroc,
    _build_representation,
    _rank_transform_matrix,
    load_earlystop_svd_bundle,
)
from scripts.run_earlystop_prefix10_svd_round1 import (
    EXTRACTION_POSITION_INDEX,
    EXTRACTION_POSITIONS,
    build_feature_store,
    evaluate_method_from_feature_store,
    make_svd_bundle_score_fn,
    make_tok_conf_score_fn,
)


DEFAULT_MAIN_CACHE_ROOT = "MUI_HUB/cache"
DEFAULT_EXTRA_CACHE_ROOT = "/home/jovyan/public-ro/MUI_HUB/cache_train"
DEFAULT_FEATURE_CACHE_DIR = "results/cache/shallow_mlp_baselines"
DEFAULT_OUT_CSV = "results/tables/shallow_mlp_baselines.csv"
DEFAULT_OUT_DOC = "docs/SHALLOW_MLP_BASELINES.md"

DEFAULT_SEEDS = (42, 101, 29)
DEFAULT_REPRESENTATIONS = ("raw+rank",)
DEFAULT_FAMILIES = ("mlp_1h", "mlp_2h")
DEFAULT_HIDDEN_OPTIONS = {
    "mlp_1h": ((32,), (64,)),
    "mlp_2h": ((64, 32), (128, 64)),
}
DEFAULT_ALPHA_OPTIONS = (0.0, 1e-4, 1e-3)
DEFAULT_BATCH_SIZE_OPTIONS = (64, 128)
DEFAULT_ACTIVATION = "relu"
DEFAULT_HOLDOUT_SPLIT = 0.15
DEFAULT_SPLIT_SEED = 42
DEFAULT_MAX_EPOCHS = 12
DEFAULT_PATIENCE = 3
DEFAULT_VAL_SPLIT = 0.15
DEFAULT_MIN_DELTA = 1e-4
DEFAULT_MAX_WORKERS = 4
DEFAULT_FEATURE_CHUNK_PROBLEMS = 24

EARLIEST_LABEL_THRESHOLD = 0.60
ARTIFACT_FEATURE_NAMES = tuple(LEGACY_FULL_FEATURE_NAMES)
MLP_FEATURE_NAMES = tuple(TOKEN_FEATURES + TRAJ_FEATURES + AVAILABILITY_FEATURES)
FEATURE_TO_INDEX = {name: idx for idx, name in enumerate(ARTIFACT_FEATURE_NAMES)}
MLP_FEATURE_INDICES = [FEATURE_TO_INDEX[name] for name in MLP_FEATURE_NAMES]
HOLDOUT_DOMAIN_ORDER = ("math", "science", "ms", "coding")
PREBUILT_FEATURE_CACHE_DIRS = (
    REPO_ROOT / "results" / "cache" / "es_svd_ms_rr_r2",
    REPO_ROOT / "results" / "cache" / "earlystop_prefix10_svd_round1c_fullcache",
)

REFERENCE_BUNDLES: dict[str, dict[str, Any]] = {
    "earlystop_svd_lowrank_lr_v1": {
        "type": "bundle",
        "path": "models/ml_selectors/earlystop_svd_lowrank_lr_v1.pkl",
        "scopes": {"math", "science", "ms", "coding"},
        "kind": "svd",
    },
    "earlystop_prefix10_svd_round1": {
        "type": "bundle",
        "path": "models/ml_selectors/earlystop_prefix10_svd_round1.pkl",
        "scopes": {"math", "science", "ms", "coding"},
        "kind": "svd",
    },
    "es_svd_math_rr_r2_20260412": {
        "type": "bundle",
        "path": "models/ml_selectors/es_svd_math_rr_r2_20260412.pkl",
        "scopes": {"math"},
        "kind": "svd",
    },
    "es_svd_science_rr_r2_20260412": {
        "type": "bundle",
        "path": "models/ml_selectors/es_svd_science_rr_r2_20260412.pkl",
        "scopes": {"science"},
        "kind": "svd",
    },
    "es_svd_ms_rr_r2_20260412": {
        "type": "bundle",
        "path": "models/ml_selectors/es_svd_ms_rr_r2_20260412.pkl",
        "scopes": {"ms"},
        "kind": "svd",
    },
    "es_svd_coding_rr_r1": {
        "type": "bundle",
        "path": "models/ml_selectors/es_svd_coding_rr_r1.pkl",
        "scopes": {"coding"},
        "kind": "svd",
    },
    "tok_conf_prefix_mean_v1": {
        "type": "signal",
        "scopes": {"math", "science", "ms", "coding"},
        "kind": "linear",
    },
}


@dataclass(frozen=True)
class SearchConfig:
    hidden_layers: tuple[int, ...]
    alpha: float
    batch_size: int
    activation: str = DEFAULT_ACTIVATION

    def label(self) -> str:
        hidden = "x".join(str(v) for v in self.hidden_layers)
        return f"h={hidden}|alpha={self.alpha:g}|bs={self.batch_size}|act={self.activation}"


@dataclass
class FitResult:
    scaler: Optional[StandardScaler]
    model: Optional[MLPClassifier]
    best_epoch: int
    best_val_auroc: float
    status: str
    note: str = ""


def _parse_int_list(value: str) -> tuple[int, ...]:
    values = [int(v.strip()) for v in str(value).split(",") if v.strip()]
    if not values:
        raise ValueError("Expected at least one integer")
    return tuple(values)


def _parse_str_list(value: str) -> tuple[str, ...]:
    values = [str(v.strip()) for v in str(value).split(",") if v.strip()]
    if not values:
        raise ValueError("Expected at least one token")
    return tuple(values)


def _stable_hash(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _pct_label(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{int(round(float(value) * 100.0))}%"


def _mean_ignore_nan(values: list[float] | tuple[float, ...] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _std_ignore_nan(values: list[float] | tuple[float, ...] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size <= 1:
        return 0.0 if finite.size == 1 else float("nan")
    return float(np.std(finite))


def _feature_cache_key(
    *,
    source_name: str,
    cache_root: str,
    positions: tuple[float, ...],
    required_feature_names: tuple[str, ...],
    max_problems_per_cache: Optional[int],
) -> str:
    payload = {
        "version": 1,
        "source_name": str(source_name),
        "cache_root": str(cache_root),
        "positions": [float(v) for v in positions],
        "required_feature_names": list(required_feature_names),
        "max_problems_per_cache": None if max_problems_per_cache is None else int(max_problems_per_cache),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def _feature_cache_path(
    *,
    cache_dir: Path,
    source_name: str,
    cache_root: str,
    positions: tuple[float, ...],
    required_feature_names: tuple[str, ...],
    max_problems_per_cache: Optional[int],
) -> Path:
    key = _feature_cache_key(
        source_name=source_name,
        cache_root=cache_root,
        positions=positions,
        required_feature_names=required_feature_names,
        max_problems_per_cache=max_problems_per_cache,
    )
    suffix = "all" if max_problems_per_cache is None else f"cap{int(max_problems_per_cache)}"
    return cache_dir / f"{source_name}_{suffix}_{key}.pkl"


def _positions_match(lhs: list[float] | tuple[float, ...], rhs: list[float] | tuple[float, ...]) -> bool:
    lhs_arr = np.asarray(lhs, dtype=np.float64)
    rhs_arr = np.asarray(rhs, dtype=np.float64)
    return lhs_arr.shape == rhs_arr.shape and bool(np.allclose(lhs_arr, rhs_arr, atol=1e-12, rtol=0.0))


def _extract_compatible_feature_store(
    cache_payload: Any,
    *,
    source_name: str,
    positions: tuple[float, ...],
    required_feature_names: tuple[str, ...],
    max_problems_per_cache: Optional[int],
) -> Optional[list[dict[str, Any]]]:
    if not isinstance(cache_payload, dict):
        return None
    feature_store = cache_payload.get("feature_store")
    if not isinstance(feature_store, list) or not feature_store:
        return None
    if str(cache_payload.get("source_name", "")) != str(source_name):
        return None
    cached_max = cache_payload.get("max_problems_per_cache")
    if max_problems_per_cache is None:
        if cached_max not in (None, 0):
            return None
    elif int(cached_max) != int(max_problems_per_cache):
        return None
    if not _positions_match(cache_payload.get("positions", ()), positions):
        return None

    feature_dim: Optional[int] = None
    for payload in feature_store:
        tensor = np.asarray(payload.get("tensor"))
        if tensor.ndim == 3:
            feature_dim = int(tensor.shape[-1])
            break
    if feature_dim is None or feature_dim != len(required_feature_names):
        return None
    return list(feature_store)


def _save_feature_store_cache(
    *,
    cache_path: Path,
    source_name: str,
    cache_root: str,
    positions: tuple[float, ...],
    max_problems_per_cache: Optional[int],
    feature_store: list[dict[str, Any]],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
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
    tmp_path.replace(cache_path)


def _load_prebuilt_feature_store(
    *,
    source_name: str,
    positions: tuple[float, ...],
    required_feature_names: tuple[str, ...],
    max_problems_per_cache: Optional[int],
) -> tuple[Optional[list[dict[str, Any]]], Optional[Path]]:
    if max_problems_per_cache is not None:
        return None, None
    for cache_dir in PREBUILT_FEATURE_CACHE_DIRS:
        if not cache_dir.exists():
            continue
        for candidate in sorted(cache_dir.glob(f"{source_name}_all_*.pkl")):
            try:
                with candidate.open("rb") as handle:
                    payload = pickle.load(handle)
            except Exception:
                continue
            feature_store = _extract_compatible_feature_store(
                payload,
                source_name=source_name,
                positions=positions,
                required_feature_names=required_feature_names,
                max_problems_per_cache=max_problems_per_cache,
            )
            if feature_store is not None:
                return feature_store, candidate
    return None, None


def _qualify_feature_store(feature_store: list[dict[str, Any]], source_name: str) -> list[dict[str, Any]]:
    qualified: list[dict[str, Any]] = []
    for payload in feature_store:
        item = dict(payload)
        item["source_name"] = str(source_name)
        item["base_cache_key"] = str(payload["cache_key"])
        item["cache_key"] = f"{source_name}/{payload['cache_key']}"
        qualified.append(item)
    return qualified


def _load_or_build_qualified_feature_store(
    *,
    source_name: str,
    cache_root: str,
    positions: tuple[float, ...],
    required_feature_names: tuple[str, ...],
    max_problems_per_cache: Optional[int],
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
        )
        if cache_path.exists() and not refresh_feature_cache:
            print(f"[shallow-mlp] loading feature cache source={source_name} path={cache_path}")
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            feature_store = _extract_compatible_feature_store(
                payload,
                source_name=source_name,
                positions=positions,
                required_feature_names=required_feature_names,
                max_problems_per_cache=max_problems_per_cache,
            )
            if feature_store is not None:
                return feature_store, cache_path, "loaded"
            print(f"[shallow-mlp] ignoring incompatible feature cache source={source_name} path={cache_path}")

    if not refresh_feature_cache:
        prebuilt_store, prebuilt_path = _load_prebuilt_feature_store(
            source_name=source_name,
            positions=positions,
            required_feature_names=required_feature_names,
            max_problems_per_cache=max_problems_per_cache,
        )
        if prebuilt_store is not None:
            print(f"[shallow-mlp] reusing prebuilt feature cache source={source_name} path={prebuilt_path}")
            if cache_path is not None and prebuilt_path is not None and cache_path != prebuilt_path and not cache_path.exists():
                _save_feature_store_cache(
                    cache_path=cache_path,
                    source_name=source_name,
                    cache_root=cache_root,
                    positions=positions,
                    max_problems_per_cache=max_problems_per_cache,
                    feature_store=prebuilt_store,
                )
                print(f"[shallow-mlp] aliased prebuilt cache source={source_name} path={cache_path}")
            return prebuilt_store, prebuilt_path, "prebuilt"

    print(f"[shallow-mlp] building feature store source={source_name} root={cache_root}")
    feature_store = _qualify_feature_store(
        build_feature_store(
            cache_root=cache_root,
            positions=positions,
            required_feature_names=set(required_feature_names),
            max_problems_per_cache=max_problems_per_cache,
            max_workers=max(1, int(max_workers)),
            chunk_problems=max(1, int(chunk_problems)),
        ),
        source_name=source_name,
    )

    if cache_path is not None:
        _save_feature_store_cache(
            cache_path=cache_path,
            source_name=source_name,
            cache_root=cache_root,
            positions=positions,
            max_problems_per_cache=max_problems_per_cache,
            feature_store=feature_store,
        )
        print(f"[shallow-mlp] saved feature cache source={source_name} path={cache_path}")
    return feature_store, cache_path, "built"


def _subset_payload_by_problem_ids(
    payload: dict[str, Any],
    selected_problem_ids: set[str],
) -> Optional[dict[str, Any]]:
    if not selected_problem_ids:
        return None

    tensor_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    sample_parts: list[np.ndarray] = []
    rank_group_parts: list[np.ndarray] = []
    cv_group_parts: list[np.ndarray] = []
    problem_ids: list[str] = []
    problem_offsets = [0]

    offsets = [int(v) for v in payload["problem_offsets"]]
    all_problem_ids = [str(v) for v in payload["problem_ids"]]
    total_samples = 0

    for problem_idx, problem_id in enumerate(all_problem_ids):
        if problem_id not in selected_problem_ids:
            continue
        start = offsets[problem_idx]
        end = offsets[problem_idx + 1]
        width = max(0, end - start)
        if width <= 0:
            continue

        tensor_parts.append(np.asarray(payload["tensor"][start:end], dtype=np.float64))
        label_parts.append(np.asarray(payload["labels"][start:end], dtype=np.int32))
        sample_parts.append(np.asarray(payload["sample_ids"][start:end], dtype=np.int32))
        rank_group_parts.append(np.asarray([f"{payload['cache_key']}::{problem_id}"] * width, dtype=object))
        cv_group_parts.append(np.asarray([f"{payload['dataset_name']}::{problem_id}"] * width, dtype=object))
        problem_ids.append(str(problem_id))
        total_samples += int(width)
        problem_offsets.append(problem_offsets[-1] + int(width))

    if not tensor_parts:
        return None

    return {
        "cache_key": str(payload["cache_key"]),
        "base_cache_key": str(payload["base_cache_key"]),
        "source_name": str(payload["source_name"]),
        "dataset_name": str(payload["dataset_name"]),
        "domain": str(payload["domain"]),
        "positions": list(payload["positions"]),
        "tensor": np.concatenate(tensor_parts, axis=0).astype(np.float64, copy=False),
        "labels": np.concatenate(label_parts).astype(np.int32, copy=False),
        "sample_ids": np.concatenate(sample_parts).astype(np.int32, copy=False),
        "group_keys": np.concatenate(rank_group_parts).astype(object, copy=False),
        "cv_group_keys": np.concatenate(cv_group_parts).astype(object, copy=False),
        "problem_ids": problem_ids,
        "problem_offsets": problem_offsets,
        "samples": int(total_samples),
    }


def _build_holdout_problem_map(
    feature_store: list[dict[str, Any]],
    *,
    holdout_split: float,
    split_seed: int,
    holdout_domains: set[str],
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    datasets: dict[str, set[str]] = {}
    for payload in feature_store:
        if str(payload["domain"]) not in holdout_domains:
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
    holdout_domains: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_payloads: list[dict[str, Any]] = []
    holdout_payloads: list[dict[str, Any]] = []

    for payload in feature_store:
        payload_domain = str(payload["domain"])
        payload_problem_ids = {str(v) for v in payload["problem_ids"]}
        if payload_domain not in holdout_domains:
            train_payload = _subset_payload_by_problem_ids(payload, payload_problem_ids)
            if train_payload is not None and int(train_payload["samples"]) > 0:
                train_payloads.append(train_payload)
            continue

        holdout_ids = set(holdout_problem_map.get(str(payload["dataset_name"]), set()))
        train_ids = {pid for pid in payload_problem_ids if pid not in holdout_ids}

        train_payload = _subset_payload_by_problem_ids(payload, train_ids)
        if train_payload is not None and int(train_payload["samples"]) > 0:
            train_payloads.append(train_payload)

        holdout_payload = _subset_payload_by_problem_ids(payload, holdout_ids)
        if holdout_payload is not None and int(holdout_payload["samples"]) > 0:
            holdout_payloads.append(holdout_payload)

    return train_payloads, holdout_payloads


def _filter_feature_store(feature_store: list[dict[str, Any]], domains: set[str]) -> list[dict[str, Any]]:
    return [payload for payload in feature_store if str(payload["domain"]) in domains]


def _summarise_feature_store(feature_store: list[dict[str, Any]]) -> dict[str, Any]:
    per_cache = []
    total_samples = 0
    total_problems = 0
    for payload in sorted(feature_store, key=lambda item: str(item["cache_key"])):
        problems = int(len(payload["problem_ids"]))
        samples = int(payload["samples"])
        total_samples += samples
        total_problems += problems
        per_cache.append(
            {
                "cache_key": str(payload["cache_key"]),
                "source_name": str(payload["source_name"]),
                "dataset_name": str(payload["dataset_name"]),
                "domain": str(payload["domain"]),
                "n_problems": problems,
                "n_samples": samples,
            }
        )
    return {
        "num_caches": int(len(per_cache)),
        "total_problems": int(total_problems),
        "total_samples": int(total_samples),
        "per_cache": per_cache,
    }


def _build_domain_training_tables(
    feature_store: list[dict[str, Any]],
    positions: tuple[float, ...],
) -> dict[int, dict[str, np.ndarray]]:
    feature_dim = len(ARTIFACT_FEATURE_NAMES)
    for payload in feature_store:
        tensor = np.asarray(payload["tensor"])
        if tensor.ndim == 3:
            feature_dim = int(tensor.shape[-1])
            break
    rows: dict[int, list[np.ndarray]] = {idx: [] for idx in range(len(positions))}
    labels: dict[int, list[np.ndarray]] = {idx: [] for idx in range(len(positions))}
    rank_groups: dict[int, list[np.ndarray]] = {idx: [] for idx in range(len(positions))}
    cv_groups: dict[int, list[np.ndarray]] = {idx: [] for idx in range(len(positions))}

    position_indices = [EXTRACTION_POSITION_INDEX[float(position)] for position in positions]
    for payload in feature_store:
        tensor = np.asarray(payload["tensor"], dtype=np.float64)
        if tensor.shape[0] == 0:
            continue
        payload_labels = np.asarray(payload["labels"], dtype=np.int32)
        payload_rank_groups = np.asarray(payload["group_keys"], dtype=object)
        payload_cv_groups = np.asarray(payload["cv_group_keys"], dtype=object)
        for local_pos_idx, src_pos_idx in enumerate(position_indices):
            rows[local_pos_idx].append(tensor[:, src_pos_idx, :])
            labels[local_pos_idx].append(payload_labels)
            rank_groups[local_pos_idx].append(payload_rank_groups)
            cv_groups[local_pos_idx].append(payload_cv_groups)

    out: dict[int, dict[str, np.ndarray]] = {}
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
                by_rank_group.setdefault(group_key, []).append(int(row_idx))
            for group_rows in by_rank_group.values():
                x_rank[group_rows] = _rank_transform_matrix(x_raw[group_rows])

        out[pos_idx] = {
            "x_raw": x_raw,
            "x_rank": x_rank,
            "y": y,
            "groups": groups_cv,
        }
    return out


def _grouped_train_val_split(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    val_fraction: float,
    seed: int,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        return None

    n_groups = int(unique_groups.size)
    n_val_groups = int(round(float(val_fraction) * n_groups))
    n_val_groups = max(1, n_val_groups)
    n_val_groups = min(n_groups - 1, n_val_groups)
    test_size = float(n_val_groups) / float(n_groups)

    for offset in range(16):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=int(seed) + offset)
        train_idx, val_idx = next(gss.split(np.zeros((len(y), 1)), y, groups))
        if train_idx.size == 0 or val_idx.size == 0:
            continue
        if np.unique(y[train_idx]).size < 2:
            continue
        if np.unique(y[val_idx]).size < 2:
            continue
        return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)

    return None


def _predict_positive_proba(model: MLPClassifier, x: np.ndarray) -> np.ndarray:
    probs = np.asarray(model.predict_proba(x), dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] == 0:
        return np.zeros((x.shape[0],), dtype=np.float64)
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        return probs[:, classes.index(1)]
    return probs[:, -1]


def _fit_epochs(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    search: SearchConfig,
    seed: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
) -> FitResult:
    if x_train.shape[0] < 2 or np.unique(y_train).size < 2:
        return FitResult(None, None, 0, float("nan"), "train_single_class", "training split lacks both classes")
    if x_val.shape[0] < 2 or np.unique(y_val).size < 2:
        return FitResult(None, None, 0, float("nan"), "val_single_class", "validation split lacks both classes")

    scaler = StandardScaler(with_mean=True, with_std=True)
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    model = MLPClassifier(
        hidden_layer_sizes=tuple(int(v) for v in search.hidden_layers),
        activation=str(search.activation),
        alpha=float(search.alpha),
        batch_size=int(search.batch_size),
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=1,
        warm_start=True,
        shuffle=True,
        random_state=int(seed),
        tol=0.0,
        early_stopping=False,
    )

    best_snapshot: Optional[tuple[list[np.ndarray], list[np.ndarray], int, float]] = None
    best_val = float("-inf")
    bad_epochs = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for epoch in range(1, int(max_epochs) + 1):
            model.fit(x_train_scaled, y_train)
            val_scores = _predict_positive_proba(model, x_val_scaled)
            val_auc = _auroc(val_scores, y_val)
            if math.isfinite(float(val_auc)) and float(val_auc) > float(best_val) + float(min_delta):
                best_val = float(val_auc)
                best_snapshot = (
                    [coef.copy() for coef in model.coefs_],
                    [inter.copy() for inter in model.intercepts_],
                    int(epoch),
                    float(val_auc),
                )
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= int(patience):
                break

    if best_snapshot is None:
        return FitResult(None, None, 0, float("nan"), "no_improving_epoch", "validation AUROC never became finite")

    model.coefs_ = [coef.copy() for coef in best_snapshot[0]]
    model.intercepts_ = [inter.copy() for inter in best_snapshot[1]]
    return FitResult(scaler, model, int(best_snapshot[2]), float(best_snapshot[3]), "ok")


def _fit_final_model(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    search: SearchConfig,
    seed: int,
    epochs: int,
) -> tuple[Optional[StandardScaler], Optional[MLPClassifier], str]:
    if x_train.shape[0] < 2 or np.unique(y_train).size < 2:
        return None, None, "train_single_class"

    scaler = StandardScaler(with_mean=True, with_std=True)
    x_train_scaled = scaler.fit_transform(x_train)
    model = MLPClassifier(
        hidden_layer_sizes=tuple(int(v) for v in search.hidden_layers),
        activation=str(search.activation),
        alpha=float(search.alpha),
        batch_size=int(search.batch_size),
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=1,
        warm_start=True,
        shuffle=True,
        random_state=int(seed),
        tol=0.0,
        early_stopping=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for _ in range(max(1, int(epochs))):
            model.fit(x_train_scaled, y_train)
    return scaler, model, "ok"


def _choose_fallback_signal(
    x_raw: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_indices: list[int],
) -> dict[str, Any]:
    best = {
        "signal_name": "tok_conf_prefix",
        "feature_index": int(FEATURE_TO_INDEX["tok_conf_prefix"]),
        "sign": 1.0,
        "cv_auroc": float("-inf"),
    }
    if x_raw.shape[0] == 0 or np.unique(y).size < 2:
        return best

    from nad.ops.earlystop_svd import _cv_auroc_baseline

    for feature_index in feature_indices:
        raw_scores = np.asarray(x_raw[:, int(feature_index)], dtype=np.float64)
        for sign in (1.0, -1.0):
            cv_auc, _ = _cv_auroc_baseline(sign * raw_scores, y, groups, n_splits=3)
            if math.isfinite(float(cv_auc)) and float(cv_auc) > float(best["cv_auroc"]):
                best = {
                    "signal_name": str(ARTIFACT_FEATURE_NAMES[int(feature_index)]),
                    "feature_index": int(feature_index),
                    "sign": float(sign),
                    "cv_auroc": float(cv_auc),
                }
    return best


def _build_search_space(family_name: str) -> list[SearchConfig]:
    hidden_options = DEFAULT_HIDDEN_OPTIONS[str(family_name)]
    out = []
    for hidden in hidden_options:
        for alpha in DEFAULT_ALPHA_OPTIONS:
            for batch_size in DEFAULT_BATCH_SIZE_OPTIONS:
                out.append(
                    SearchConfig(
                        hidden_layers=tuple(int(v) for v in hidden),
                        alpha=float(alpha),
                        batch_size=int(batch_size),
                        activation=DEFAULT_ACTIVATION,
                    )
                )
    return out


def _search_position_config(
    *,
    x_raw: np.ndarray,
    x_rank: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    representation: str,
    family_name: str,
    tuning_seed: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
    val_fraction: float,
) -> dict[str, Any]:
    split = _grouped_train_val_split(y, groups, val_fraction=val_fraction, seed=tuning_seed)
    if split is None:
        fallback = _choose_fallback_signal(x_raw, y, groups, MLP_FEATURE_INDICES)
        return {
            "route_type": "fallback_signal",
            "representation": "raw",
            "feature_indices": [int(fallback["feature_index"])],
            "feature_names": [str(fallback["signal_name"])],
            "signal_name": str(fallback["signal_name"]),
            "sign": float(fallback["sign"]),
            "status": "fallback_no_grouped_val",
            "val_auroc": float(fallback["cv_auroc"]),
            "best_epoch": 0,
        }

    train_idx, val_idx = split
    x_rep = _build_representation(
        x_raw=x_raw,
        x_rank=x_rank,
        feature_indices=MLP_FEATURE_INDICES,
        representation=representation,
    )
    best_choice: Optional[dict[str, Any]] = None

    for search in _build_search_space(family_name):
        fit = _fit_epochs(
            x_train=x_rep[train_idx],
            y_train=y[train_idx],
            x_val=x_rep[val_idx],
            y_val=y[val_idx],
            search=search,
            seed=tuning_seed,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
        )
        if fit.model is None or not math.isfinite(float(fit.best_val_auroc)):
            continue
        candidate = {
            "route_type": "mlp",
            "representation": str(representation),
            "feature_indices": list(MLP_FEATURE_INDICES),
            "feature_names": list(MLP_FEATURE_NAMES),
            "hidden_layers": list(search.hidden_layers),
            "activation": str(search.activation),
            "alpha": float(search.alpha),
            "batch_size": int(search.batch_size),
            "best_epoch": int(fit.best_epoch),
            "val_auroc": float(fit.best_val_auroc),
            "status": "ok",
        }
        if best_choice is None or float(candidate["val_auroc"]) > float(best_choice["val_auroc"]):
            best_choice = candidate

    if best_choice is not None:
        return best_choice

    fallback = _choose_fallback_signal(x_raw, y, groups, MLP_FEATURE_INDICES)
    return {
        "route_type": "fallback_signal",
        "representation": "raw",
        "feature_indices": [int(fallback["feature_index"])],
        "feature_names": [str(fallback["signal_name"])],
        "signal_name": str(fallback["signal_name"]),
        "sign": float(fallback["sign"]),
        "status": "fallback_search_failed",
        "val_auroc": float(fallback["cv_auroc"]),
        "best_epoch": 0,
    }


def _train_seeded_route(
    *,
    position_table: dict[str, np.ndarray],
    selected_route: dict[str, Any],
    seed: int,
    val_fraction: float,
    max_epochs: int,
    patience: int,
    min_delta: float,
) -> dict[str, Any]:
    if selected_route["route_type"] != "mlp":
        return dict(selected_route)

    x_raw = np.asarray(position_table["x_raw"], dtype=np.float64)
    x_rank = np.asarray(position_table["x_rank"], dtype=np.float64)
    y = np.asarray(position_table["y"], dtype=np.int32)
    groups = np.asarray(position_table["groups"], dtype=object)

    split = _grouped_train_val_split(y, groups, val_fraction=val_fraction, seed=seed)
    if split is None:
        fallback = _choose_fallback_signal(x_raw, y, groups, MLP_FEATURE_INDICES)
        return {
            "route_type": "fallback_signal",
            "representation": "raw",
            "feature_indices": [int(fallback["feature_index"])],
            "feature_names": [str(fallback["signal_name"])],
            "signal_name": str(fallback["signal_name"]),
            "sign": float(fallback["sign"]),
            "status": "fallback_seed_no_grouped_val",
            "seed": int(seed),
            "val_auroc": float(fallback["cv_auroc"]),
            "best_epoch": 0,
        }

    train_idx, val_idx = split
    x_rep = _build_representation(
        x_raw=x_raw,
        x_rank=x_rank,
        feature_indices=[int(v) for v in selected_route["feature_indices"]],
        representation=str(selected_route["representation"]),
    )
    search = SearchConfig(
        hidden_layers=tuple(int(v) for v in selected_route["hidden_layers"]),
        alpha=float(selected_route["alpha"]),
        batch_size=int(selected_route["batch_size"]),
        activation=str(selected_route["activation"]),
    )
    fit = _fit_epochs(
        x_train=x_rep[train_idx],
        y_train=y[train_idx],
        x_val=x_rep[val_idx],
        y_val=y[val_idx],
        search=search,
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=min_delta,
    )
    if fit.model is None or fit.scaler is None or fit.best_epoch <= 0:
        fallback = _choose_fallback_signal(x_raw, y, groups, MLP_FEATURE_INDICES)
        return {
            "route_type": "fallback_signal",
            "representation": "raw",
            "feature_indices": [int(fallback["feature_index"])],
            "feature_names": [str(fallback["signal_name"])],
            "signal_name": str(fallback["signal_name"]),
            "sign": float(fallback["sign"]),
            "status": f"fallback_seed_fit_{fit.status}",
            "seed": int(seed),
            "val_auroc": float(fallback["cv_auroc"]),
            "best_epoch": 0,
        }

    final_scaler, final_model, final_status = _fit_final_model(
        x_train=x_rep,
        y_train=y,
        search=search,
        seed=seed,
        epochs=int(fit.best_epoch),
    )
    if final_model is None or final_scaler is None:
        fallback = _choose_fallback_signal(x_raw, y, groups, MLP_FEATURE_INDICES)
        return {
            "route_type": "fallback_signal",
            "representation": "raw",
            "feature_indices": [int(fallback["feature_index"])],
            "feature_names": [str(fallback["signal_name"])],
            "signal_name": str(fallback["signal_name"]),
            "sign": float(fallback["sign"]),
            "status": f"fallback_final_{final_status}",
            "seed": int(seed),
            "val_auroc": float(fallback["cv_auroc"]),
            "best_epoch": 0,
        }

    route = dict(selected_route)
    route.update(
        {
            "route_type": "mlp",
            "model": final_model,
            "scaler": final_scaler,
            "seed": int(seed),
            "best_epoch": int(fit.best_epoch),
            "val_auroc": float(fit.best_val_auroc),
            "status": "ok",
        }
    )
    return route


def _score_xraw_with_route(x_raw: np.ndarray, route: dict[str, Any]) -> np.ndarray:
    if route["route_type"] == "fallback_signal":
        feature_index = int(route["feature_indices"][0])
        return float(route.get("sign", 1.0)) * np.asarray(x_raw[:, feature_index], dtype=np.float64)

    x_rank = _rank_transform_matrix(x_raw)
    x_rep = _build_representation(
        x_raw=x_raw,
        x_rank=x_rank,
        feature_indices=[int(v) for v in route["feature_indices"]],
        representation=str(route["representation"]),
    )
    scaler = route["scaler"]
    model = route["model"]
    x_scaled = scaler.transform(x_rep)
    return np.asarray(_predict_positive_proba(model, x_scaled), dtype=np.float64)


def _make_mlp_score_fn(bundle: dict[str, Any]) -> Callable[[str, int, np.ndarray], np.ndarray]:
    def _score(domain: str, position_index: int, x_raw: np.ndarray) -> np.ndarray:
        route = bundle["domains"][domain]["routes"][position_index]
        return _score_xraw_with_route(x_raw=x_raw, route=route)

    return _score


def _train_scope_family_seed_bundle(
    *,
    scope_name: str,
    family_name: str,
    representation: str,
    seed: int,
    tuning_routes: list[dict[str, Any]],
    position_tables: dict[int, dict[str, np.ndarray]],
    max_epochs: int,
    patience: int,
    min_delta: float,
    val_fraction: float,
) -> dict[str, Any]:
    routes = []
    for pos_idx, _position in enumerate(EARLY_STOP_POSITIONS):
        route = _train_seeded_route(
            position_table=position_tables[pos_idx],
            selected_route=tuning_routes[pos_idx],
            seed=seed,
            val_fraction=val_fraction,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
        )
        routes.append(route)

    domain_keys = ("math", "science") if scope_name == "ms" else (scope_name,)
    domains = {domain_key: {"routes": routes} for domain_key in domain_keys}
    return {
        "bundle_version": "shallow_mlp_baselines_v1",
        "scope_name": str(scope_name),
        "family_name": str(family_name),
        "representation": str(representation),
        "seed": int(seed),
        "feature_names": list(ARTIFACT_FEATURE_NAMES),
        "mlp_feature_names": list(MLP_FEATURE_NAMES),
        "domains": domains,
    }


def _anchor_config_summary(routes: list[dict[str, Any]]) -> str:
    summary: dict[str, Any] = {}
    for pos_idx, position in enumerate(EARLY_STOP_POSITIONS):
        route = routes[pos_idx]
        key = _pct_label(float(position))
        if route["route_type"] == "mlp":
            summary[key] = {
                "route_type": "mlp",
                "hidden_layers": list(route["hidden_layers"]),
                "alpha": float(route["alpha"]),
                "batch_size": int(route["batch_size"]),
                "best_epoch": int(route["best_epoch"]),
                "val_auroc": float(route["val_auroc"]),
            }
        else:
            summary[key] = {
                "route_type": str(route["route_type"]),
                "signal_name": str(route["signal_name"]),
                "sign": float(route["sign"]),
                "val_auroc": float(route["val_auroc"]),
            }
    return json.dumps(summary, ensure_ascii=False, sort_keys=True)


def _evaluate_reference_method(
    *,
    method_name: str,
    feature_store: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    spec = REFERENCE_BUNDLES[method_name]
    if spec["type"] == "signal":
        score_fn = make_tok_conf_score_fn()
    else:
        bundle_path = REPO_ROOT / str(spec["path"])
        if not bundle_path.exists():
            return None
        bundle = load_earlystop_svd_bundle(bundle_path)
        score_fn = make_svd_bundle_score_fn(bundle)
    return evaluate_method_from_feature_store(
        method_name=method_name,
        feature_store=feature_store,
        position_values=tuple(float(v) for v in EARLY_STOP_POSITIONS),
        score_fn=score_fn,
    )


def _pick_best_reference(
    rows: list[dict[str, Any]],
    *,
    kind_filter: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    candidates = []
    for row in rows:
        if row.get("row_kind") != "reference":
            continue
        if kind_filter is not None and str(row.get("reference_kind")) != str(kind_filter):
            continue
        auc = float(row.get("auc_of_auroc", float("nan")))
        if math.isfinite(auc):
            candidates.append(row)
    if not candidates:
        return None
    candidates.sort(key=lambda item: float(item["auc_of_auroc"]), reverse=True)
    return candidates[0]


def _format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        value_f = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value_f):
        return "N/A"
    return f"{value_f:.4f}"


def _write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_kind",
        "domain",
        "protocol",
        "split_name",
        "representation",
        "method_name",
        "family_name",
        "seed",
        "n_seeds",
        "reference_kind",
        "status",
        "auc_of_auroc",
        "auc_of_selacc",
        "auroc_at_100",
        "stop_acc_at_100",
        "earliest_gt_0p6",
        "val_auroc_mean",
        "best_svd_reference_method",
        "best_svd_reference_auc_of_auroc",
        "delta_auc_of_auroc_vs_best_svd",
        "best_linear_reference_method",
        "best_linear_reference_auc_of_auroc",
        "delta_auc_of_auroc_vs_best_linear",
        "anchor_config_summary",
        "notes",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _verdict_nonlinearity(best_rows: list[dict[str, Any]]) -> str:
    deltas = [float(row["delta_auc_of_auroc_vs_best_svd"]) for row in best_rows if math.isfinite(float(row["delta_auc_of_auroc_vs_best_svd"]))]
    if not deltas:
        return "Not evaluated."
    mean_delta = float(np.mean(deltas))
    positive_domains = sum(1 for delta in deltas if delta > 0.01)
    if mean_delta > 0.01 and positive_domains >= max(1, len(deltas) - 1):
        return "Yes — the best shallow MLP materially beats the best SVD reference overall."
    if max(deltas) > 0.01:
        return "Mixed — gains appear in some domains, but they are not broad enough to call a clear overall win over the SVD route."
    return "No — the gains versus the best SVD reference are small or inconsistent."


def _verdict_science_coding(best_rows: list[dict[str, Any]]) -> str:
    by_domain = {str(row["domain"]): float(row["delta_auc_of_auroc_vs_best_svd"]) for row in best_rows if math.isfinite(float(row["delta_auc_of_auroc_vs_best_svd"]))}
    science_delta = by_domain.get("science")
    coding_delta = by_domain.get("coding")
    math_delta = by_domain.get("math")
    if science_delta is None and coding_delta is None:
        return "Not evaluated."
    positives = []
    if science_delta is not None:
        positives.append(("science", science_delta))
    if coding_delta is not None:
        positives.append(("coding", coding_delta))
    lead = max(positives, key=lambda item: item[1]) if positives else None
    if lead is not None and lead[1] > 0.01 and (math_delta is None or lead[1] > math_delta + 0.005):
        if coding_delta is not None and science_delta is not None and max(science_delta, coding_delta) > 0.01:
            return "Yes — the MLP gains are concentrated in science/coding rather than math."
        if lead[0] == "science":
            return "Mostly science — that is where the clearest non-linear gain shows up."
        return "Mostly coding — that is where the clearest non-linear gain shows up."
    return "No — the gains are not clearly concentrated in science/coding."


def _verdict_seed_stability(seed_rows: list[dict[str, Any]]) -> str:
    deltas = [float(row["delta_auc_of_auroc_vs_best_svd"]) for row in seed_rows if math.isfinite(float(row["delta_auc_of_auroc_vs_best_svd"]))]
    if not deltas:
        return "Not evaluated."
    wins = sum(1 for delta in deltas if delta > 0.0)
    std_delta = _std_ignore_nan(deltas)
    if wins >= math.ceil(len(deltas) * 2.0 / 3.0) and std_delta < 0.01:
        return "Yes — seed-to-seed variance is modest and most seeds stay on the same side of the SVD baseline."
    if wins > 0:
        return "Mixed — some seeds beat the SVD route, but the margin is not fully stable."
    return "No — the gain does not survive seed variation reliably."


def _verdict_interpretability(best_rows: list[dict[str, Any]], seed_rows: list[dict[str, Any]]) -> str:
    deltas = [float(row["delta_auc_of_auroc_vs_best_svd"]) for row in best_rows if math.isfinite(float(row["delta_auc_of_auroc_vs_best_svd"]))]
    mean_delta = float(np.mean(deltas)) if deltas else float("nan")
    if math.isfinite(mean_delta) and mean_delta > 0.015:
        stability_text = _verdict_seed_stability(seed_rows)
        if stability_text.startswith("Yes"):
            return "Probably yes — the average gain looks large enough to justify a small loss of exact linear interpretability."
    return "No — the measured gain is not large or stable enough to outweigh the loss of exact linear interpretability."


def _render_markdown(
    *,
    rows: list[dict[str, Any]],
    out_path: Path,
    include_coding: bool,
    seeds: tuple[int, ...],
) -> None:
    reference_rows = [row for row in rows if row.get("row_kind") == "reference"]
    best_mean_rows = [row for row in rows if row.get("row_kind") == "mlp_best_mean"]
    seed_rows = [row for row in rows if row.get("row_kind") == "mlp_seed_best_family"]

    best_mean_rows = sorted(best_mean_rows, key=lambda item: HOLDOUT_DOMAIN_ORDER.index(item["domain"]))
    main_table = [
        "| domain | best SVD ref | shallow MLP | delta | seed std | seeds > ref | family |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in best_mean_rows:
        domain = str(row["domain"])
        domain_seed_rows = [item for item in seed_rows if str(item["domain"]) == domain]
        deltas = [float(item["delta_auc_of_auroc_vs_best_svd"]) for item in domain_seed_rows if math.isfinite(float(item["delta_auc_of_auroc_vs_best_svd"]))]
        wins = sum(1 for delta in deltas if delta > 0.0)
        seed_std = _std_ignore_nan([float(item["auc_of_auroc"]) for item in domain_seed_rows]) if domain_seed_rows else float("nan")
        main_table.append(
            "| {domain} | {svd_auc} | {mlp_auc} | {delta} | {seed_std} | {wins}/{total} | {family} |".format(
                domain=domain,
                svd_auc=_format_metric(row.get("best_svd_reference_auc_of_auroc")),
                mlp_auc=_format_metric(row.get("auc_of_auroc")),
                delta=_format_metric(row.get("delta_auc_of_auroc_vs_best_svd")),
                seed_std=_format_metric(seed_std),
                wins=int(wins),
                total=int(len(domain_seed_rows)),
                family=str(row.get("family_name", "")),
            )
        )

    lines = [
        "# SHALLOW MLP BASELINES",
        "",
        "## Setup",
        "",
        f"- `repo root`: `{REPO_ROOT}`",
        f"- `protocol`: grouped `85/15` holdout by `dataset + problem_id`, `split_seed={DEFAULT_SPLIT_SEED}`",
        f"- `domains`: `math`, `science`, `ms`{', `coding`' if include_coding else ''}",
        f"- `representations`: `raw+rank` by default; `raw` remains script-optional but is not part of this artifact",
        f"- `seeds`: `{','.join(str(v) for v in seeds)}`",
        "- `features`: token + trajectory + availability (`tok_*`, `traj_*`, `has_*`)",
        "- `feature cache`: reuses the existing legacy 30-feature prefix-safe tensors so runs stay compact and match the current SVD baselines",
        "- `model backend`: `sklearn.neural_network.MLPClassifier`",
        "- `searched`: hidden sizes `[32]`, `[64]`, `[64,32]`, `[128,64]`; weight decay `[0, 1e-4, 1e-3]`; batch size `[64, 128]`; activation `ReLU`",
        f"- `early stopping`: validation AUROC with `max_epochs={DEFAULT_MAX_EPOCHS}`, `patience={DEFAULT_PATIENCE}`, `min_delta={DEFAULT_MIN_DELTA}`",
        "- `not searched`: dropout / LayerNorm / GELU, because the current repo environment has no `torch` and the sklearn backend does not expose those options cleanly",
        "- `structured OOD`: not included in this artifact; this file reports ID grouped-holdout only",
        "",
        "## Main Table",
        "",
        *main_table,
        "",
        "## Answers",
        "",
        f"- **Does light non-linearity materially outperform the SVD route?** {_verdict_nonlinearity(best_mean_rows)}",
        f"- **Does MLP mainly help science / coding or not?** {_verdict_science_coding(best_mean_rows)}",
        f"- **Are gains stable across seeds?** {_verdict_seed_stability(seed_rows)}",
        f"- **Is the accuracy gain large enough to justify losing exact linear interpretability?** {_verdict_interpretability(best_mean_rows, seed_rows)}",
        "",
        "## Notes",
        "",
        "- `best SVD ref` means the strongest available SVD bundle on the same holdout scope, chosen from the evaluated reference rows in `results/tables/shallow_mlp_baselines.csv`.",
        "- `ms` is the combined non-coding scope (`math + science`), matching the existing repo convention.",
        "- `coding` is included only because this run kept it feasible on the same grouped-holdout pipeline; no structured-OOD claim is made here.",
        "",
        "## Reference Rows",
        "",
    ]

    ref_table = [
        "| domain | method | kind | auc_of_auroc | auroc@100 | stop_acc@100 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in sorted(reference_rows, key=lambda item: (str(item["domain"]), -float(item["auc_of_auroc"]) if math.isfinite(float(item["auc_of_auroc"])) else float("inf"))):
        ref_table.append(
            "| {domain} | `{method}` | {kind} | {auc} | {auroc100} | {stop100} |".format(
                domain=str(row["domain"]),
                method=str(row["method_name"]),
                kind=str(row["reference_kind"]),
                auc=_format_metric(row.get("auc_of_auroc")),
                auroc100=_format_metric(row.get("auroc_at_100")),
                stop100=_format_metric(row.get("stop_acc_at_100")),
            )
        )
    lines.extend(ref_table)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train shallow MLP baselines for EarlyStop domains")
    ap.add_argument("--cache-root", default=DEFAULT_MAIN_CACHE_ROOT)
    ap.add_argument("--extra-cache-root", default=DEFAULT_EXTRA_CACHE_ROOT)
    ap.add_argument("--feature-cache-dir", default=DEFAULT_FEATURE_CACHE_DIR)
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--max-problems-per-cache", type=int, default=0, help="0 means all problems")
    ap.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS)
    ap.add_argument("--feature-chunk-problems", type=int, default=DEFAULT_FEATURE_CHUNK_PROBLEMS)
    ap.add_argument("--holdout-split", type=float, default=DEFAULT_HOLDOUT_SPLIT)
    ap.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    ap.add_argument("--seeds", default="42,101,29")
    ap.add_argument("--families", default="mlp_1h,mlp_2h")
    ap.add_argument("--representations", default="raw+rank")
    ap.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    ap.add_argument("--val-split", type=float, default=DEFAULT_VAL_SPLIT)
    ap.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    ap.add_argument("--include-coding", action="store_true")
    ap.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    ap.add_argument("--out-doc", default=DEFAULT_OUT_DOC)
    args = ap.parse_args()

    main_cache_root = str((REPO_ROOT / args.cache_root).resolve()) if not Path(args.cache_root).is_absolute() else str(Path(args.cache_root).resolve())
    extra_cache_root = str((REPO_ROOT / args.extra_cache_root).resolve()) if not Path(args.extra_cache_root).is_absolute() else str(Path(args.extra_cache_root).resolve())
    feature_cache_dir = Path(args.feature_cache_dir)
    if not feature_cache_dir.is_absolute():
        feature_cache_dir = REPO_ROOT / feature_cache_dir

    max_problems_per_cache = None if int(args.max_problems_per_cache) <= 0 else int(args.max_problems_per_cache)
    seeds = _parse_int_list(args.seeds)
    families = _parse_str_list(args.families)
    representations = _parse_str_list(args.representations)

    required_feature_names = tuple(str(name) for name in ARTIFACT_FEATURE_NAMES)
    main_store, _, _ = _load_or_build_qualified_feature_store(
        source_name="cache",
        cache_root=main_cache_root,
        positions=tuple(float(v) for v in EXTRACTION_POSITIONS),
        required_feature_names=required_feature_names,
        max_problems_per_cache=max_problems_per_cache,
        max_workers=int(args.workers),
        chunk_problems=int(args.feature_chunk_problems),
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_feature_cache),
    )
    extra_store, _, _ = _load_or_build_qualified_feature_store(
        source_name="cache_train",
        cache_root=extra_cache_root,
        positions=tuple(float(v) for v in EXTRACTION_POSITIONS),
        required_feature_names=required_feature_names,
        max_problems_per_cache=max_problems_per_cache,
        max_workers=int(args.workers),
        chunk_problems=int(args.feature_chunk_problems),
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_feature_cache),
    )
    combined_store = list(main_store) + list(extra_store)
    holdout_domains = {"math", "science"}
    if bool(args.include_coding):
        holdout_domains.add("coding")

    holdout_problem_map, holdout_summary = _build_holdout_problem_map(
        combined_store,
        holdout_split=float(args.holdout_split),
        split_seed=int(args.split_seed),
        holdout_domains=holdout_domains,
    )
    train_store, holdout_store = _split_feature_store(
        combined_store,
        holdout_problem_map=holdout_problem_map,
        holdout_domains=holdout_domains,
    )

    scope_domains = {
        "math": {"math"},
        "science": {"science"},
        "ms": {"math", "science"},
    }
    if bool(args.include_coding):
        scope_domains["coding"] = {"coding"}

    train_store_by_scope = {scope: _filter_feature_store(train_store, domains) for scope, domains in scope_domains.items()}
    holdout_store_by_scope = {scope: _filter_feature_store(holdout_store, domains) for scope, domains in scope_domains.items()}

    rows: list[dict[str, Any]] = []
    tuning_routes_by_scope_family_repr: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    best_seed_rows: list[dict[str, Any]] = []

    print("[shallow-mlp] evaluating reference methods")
    for scope_name, test_store in holdout_store_by_scope.items():
        if not test_store:
            continue
        for method_name, spec in REFERENCE_BUNDLES.items():
            if scope_name not in set(spec["scopes"]):
                continue
            ref_result = _evaluate_reference_method(
                method_name=method_name,
                feature_store=test_store,
            )
            if ref_result is None:
                continue
            aggregate = ref_result["aggregate"]
            rows.append(
                {
                    "row_kind": "reference",
                    "domain": str(scope_name),
                    "protocol": "id_grouped_85_15",
                    "split_name": f"seed{int(args.split_seed)}_grouped_holdout",
                    "representation": "n/a",
                    "method_name": str(method_name),
                    "family_name": "",
                    "seed": "",
                    "n_seeds": 0,
                    "reference_kind": str(spec["kind"]),
                    "status": "ok",
                    "auc_of_auroc": float(aggregate["auc_of_auroc"]),
                    "auc_of_selacc": float(aggregate["auc_of_selacc"]),
                    "auroc_at_100": float(aggregate["auroc@100%"]),
                    "stop_acc_at_100": float(aggregate["stop_acc@100%"]),
                    "earliest_gt_0p6": _pct_label(aggregate.get("earliest_gt_0.6")),
                    "val_auroc_mean": "",
                    "best_svd_reference_method": "",
                    "best_svd_reference_auc_of_auroc": "",
                    "delta_auc_of_auroc_vs_best_svd": "",
                    "best_linear_reference_method": "",
                    "best_linear_reference_auc_of_auroc": "",
                    "delta_auc_of_auroc_vs_best_linear": "",
                    "anchor_config_summary": "",
                    "notes": "",
                }
            )

    print("[shallow-mlp] searching per-position shallow MLP configs")
    for scope_name, scope_train_store in train_store_by_scope.items():
        if not scope_train_store:
            continue
        position_tables = _build_domain_training_tables(scope_train_store, tuple(float(v) for v in EARLY_STOP_POSITIONS))
        for representation in representations:
            for family_name in families:
                tuning_routes = []
                for pos_idx, position in enumerate(EARLY_STOP_POSITIONS):
                    table = position_tables[pos_idx]
                    route = _search_position_config(
                        x_raw=np.asarray(table["x_raw"], dtype=np.float64),
                        x_rank=np.asarray(table["x_rank"], dtype=np.float64),
                        y=np.asarray(table["y"], dtype=np.int32),
                        groups=np.asarray(table["groups"], dtype=object),
                        representation=str(representation),
                        family_name=str(family_name),
                        tuning_seed=int(args.split_seed),
                        max_epochs=int(args.max_epochs),
                        patience=int(args.patience),
                        min_delta=float(args.min_delta),
                        val_fraction=float(args.val_split),
                    )
                    tuning_routes.append(route)
                    print(
                        f"[shallow-mlp] tuned scope={scope_name} family={family_name} rep={representation} "
                        f"pos={_pct_label(position)} route={route['route_type']} val_auc={route.get('val_auroc', float('nan')):.4f}"
                    )
                tuning_routes_by_scope_family_repr[(scope_name, family_name, representation)] = tuning_routes

    print("[shallow-mlp] training seeded bundles and evaluating holdout")
    for scope_name, scope_train_store in train_store_by_scope.items():
        scope_holdout_store = holdout_store_by_scope.get(scope_name, [])
        if not scope_train_store or not scope_holdout_store:
            continue
        position_tables = _build_domain_training_tables(scope_train_store, tuple(float(v) for v in EARLY_STOP_POSITIONS))
        domain_reference_rows = [row for row in rows if row.get("row_kind") == "reference" and str(row.get("domain")) == str(scope_name)]
        best_svd_reference = _pick_best_reference(domain_reference_rows, kind_filter="svd")
        best_linear_reference = _pick_best_reference(domain_reference_rows, kind_filter="linear")

        for representation in representations:
            family_mean_rows: list[dict[str, Any]] = []
            seed_metric_rows_by_family: dict[str, list[dict[str, Any]]] = {}
            for family_name in families:
                tuning_routes = tuning_routes_by_scope_family_repr[(scope_name, family_name, representation)]
                seed_rows_for_family: list[dict[str, Any]] = []
                for seed in seeds:
                    bundle = _train_scope_family_seed_bundle(
                        scope_name=scope_name,
                        family_name=family_name,
                        representation=representation,
                        seed=int(seed),
                        tuning_routes=tuning_routes,
                        position_tables=position_tables,
                        max_epochs=int(args.max_epochs),
                        patience=int(args.patience),
                        min_delta=float(args.min_delta),
                        val_fraction=float(args.val_split),
                    )
                    eval_result = evaluate_method_from_feature_store(
                        method_name=f"{family_name}_seed{seed}",
                        feature_store=scope_holdout_store,
                        position_values=tuple(float(v) for v in EARLY_STOP_POSITIONS),
                        score_fn=_make_mlp_score_fn(bundle),
                    )
                    aggregate = eval_result["aggregate"]
                    row = {
                        "row_kind": "mlp_seed",
                        "domain": str(scope_name),
                        "protocol": "id_grouped_85_15",
                        "split_name": f"seed{int(args.split_seed)}_grouped_holdout",
                        "representation": str(representation),
                        "method_name": f"{family_name}_seed{seed}",
                        "family_name": str(family_name),
                        "seed": int(seed),
                        "n_seeds": int(len(seeds)),
                        "reference_kind": "mlp",
                        "status": "ok",
                        "auc_of_auroc": float(aggregate["auc_of_auroc"]),
                        "auc_of_selacc": float(aggregate["auc_of_selacc"]),
                        "auroc_at_100": float(aggregate["auroc@100%"]),
                        "stop_acc_at_100": float(aggregate["stop_acc@100%"]),
                        "earliest_gt_0p6": _pct_label(aggregate.get("earliest_gt_0.6")),
                        "val_auroc_mean": _mean_ignore_nan([float(route.get("val_auroc", float("nan"))) for route in bundle["domains"][next(iter(bundle["domains"]))]["routes"]]),
                        "best_svd_reference_method": best_svd_reference["method_name"] if best_svd_reference is not None else "",
                        "best_svd_reference_auc_of_auroc": best_svd_reference["auc_of_auroc"] if best_svd_reference is not None else "",
                        "delta_auc_of_auroc_vs_best_svd": (
                            float(aggregate["auc_of_auroc"]) - float(best_svd_reference["auc_of_auroc"])
                            if best_svd_reference is not None and math.isfinite(float(best_svd_reference["auc_of_auroc"]))
                            else ""
                        ),
                        "best_linear_reference_method": best_linear_reference["method_name"] if best_linear_reference is not None else "",
                        "best_linear_reference_auc_of_auroc": best_linear_reference["auc_of_auroc"] if best_linear_reference is not None else "",
                        "delta_auc_of_auroc_vs_best_linear": (
                            float(aggregate["auc_of_auroc"]) - float(best_linear_reference["auc_of_auroc"])
                            if best_linear_reference is not None and math.isfinite(float(best_linear_reference["auc_of_auroc"]))
                            else ""
                        ),
                        "anchor_config_summary": _anchor_config_summary(bundle["domains"][next(iter(bundle["domains"]))]["routes"]),
                        "notes": "",
                    }
                    rows.append(row)
                    seed_rows_for_family.append(row)
                    print(
                        f"[shallow-mlp] holdout scope={scope_name} family={family_name} rep={representation} seed={seed} "
                        f"auc_of_auroc={float(aggregate['auc_of_auroc']):.4f}"
                    )

                seed_metric_rows_by_family[family_name] = seed_rows_for_family
                if seed_rows_for_family:
                    family_mean_row = {
                        "row_kind": "mlp_family_mean",
                        "domain": str(scope_name),
                        "protocol": "id_grouped_85_15",
                        "split_name": f"seed{int(args.split_seed)}_grouped_holdout",
                        "representation": str(representation),
                        "method_name": f"{family_name}_mean",
                        "family_name": str(family_name),
                        "seed": "mean",
                        "n_seeds": int(len(seed_rows_for_family)),
                        "reference_kind": "mlp",
                        "status": "ok",
                        "auc_of_auroc": _mean_ignore_nan([float(row["auc_of_auroc"]) for row in seed_rows_for_family]),
                        "auc_of_selacc": _mean_ignore_nan([float(row["auc_of_selacc"]) for row in seed_rows_for_family]),
                        "auroc_at_100": _mean_ignore_nan([float(row["auroc_at_100"]) for row in seed_rows_for_family]),
                        "stop_acc_at_100": _mean_ignore_nan([float(row["stop_acc_at_100"]) for row in seed_rows_for_family]),
                        "earliest_gt_0p6": min(
                            (row["earliest_gt_0p6"] for row in seed_rows_for_family if row["earliest_gt_0p6"] != "N/A"),
                            default="N/A",
                        ),
                        "val_auroc_mean": _mean_ignore_nan([float(row["val_auroc_mean"]) for row in seed_rows_for_family]),
                        "best_svd_reference_method": best_svd_reference["method_name"] if best_svd_reference is not None else "",
                        "best_svd_reference_auc_of_auroc": best_svd_reference["auc_of_auroc"] if best_svd_reference is not None else "",
                        "delta_auc_of_auroc_vs_best_svd": (
                            _mean_ignore_nan([float(row["auc_of_auroc"]) for row in seed_rows_for_family]) - float(best_svd_reference["auc_of_auroc"])
                            if best_svd_reference is not None and math.isfinite(float(best_svd_reference["auc_of_auroc"]))
                            else ""
                        ),
                        "best_linear_reference_method": best_linear_reference["method_name"] if best_linear_reference is not None else "",
                        "best_linear_reference_auc_of_auroc": best_linear_reference["auc_of_auroc"] if best_linear_reference is not None else "",
                        "delta_auc_of_auroc_vs_best_linear": (
                            _mean_ignore_nan([float(row["auc_of_auroc"]) for row in seed_rows_for_family]) - float(best_linear_reference["auc_of_auroc"])
                            if best_linear_reference is not None and math.isfinite(float(best_linear_reference["auc_of_auroc"]))
                            else ""
                        ),
                        "anchor_config_summary": "",
                        "notes": "",
                    }
                    rows.append(family_mean_row)
                    family_mean_rows.append(family_mean_row)

            if family_mean_rows:
                family_mean_rows.sort(key=lambda item: float(item["auc_of_auroc"]), reverse=True)
                best_family_row = dict(family_mean_rows[0])
                best_family = str(best_family_row["family_name"])
                best_family_row["row_kind"] = "mlp_best_mean"
                best_family_row["method_name"] = f"{best_family}_best_mean"
                rows.append(best_family_row)

                for seed_row in seed_metric_rows_by_family.get(best_family, []):
                    best_seed_row = dict(seed_row)
                    best_seed_row["row_kind"] = "mlp_seed_best_family"
                    rows.append(best_seed_row)
                    best_seed_rows.append(best_seed_row)

    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = REPO_ROOT / out_csv
    _write_csv(rows, out_csv)

    out_doc = Path(args.out_doc)
    if not out_doc.is_absolute():
        out_doc = REPO_ROOT / out_doc
    _render_markdown(
        rows=rows,
        out_path=out_doc,
        include_coding=bool(args.include_coding),
        seeds=seeds,
    )

    print("[shallow-mlp] done")
    print(f"  train_store:  {_summarise_feature_store(train_store)}")
    print(f"  holdout_store:{_summarise_feature_store(holdout_store)}")
    print(f"  holdout_map datasets: {sorted(holdout_summary.keys())}")
    print(f"  csv: {_display_path(out_csv)}")
    print(f"  doc: {_display_path(out_doc)}")


if __name__ == "__main__":
    main()
