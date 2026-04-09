#!/usr/bin/env python3
"""Run EarlyStop prefix/anchor low-rank SVD round1 experiments."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.selectors.trajectory_impl import (
    DEFAULT_REFLECTION_THRESHOLD,
    _compute_trajectory_scores_for_prefix_counts,
    _compute_trajectory_scores,
)
from nad.core.views.reader import CacheReader
from nad.ops.accuracy import load_correctness_map
from nad.ops.earlystop import (
    CacheEntry,
    EARLY_STOP_POSITIONS,
    N_POSITIONS,
    build_earlystop_payload,
    build_problem_groups,
    discover_cache_entries,
    score_cache_entry_earlystop,
    validate_earlystop_payload,
    write_earlystop_payload,
    _problem_sort_key,
)
from nad.ops.earlystop_svd import (
    AVAILABILITY_FEATURES,
    FULL_FEATURE_NAMES,
    TOKEN_FEATURES,
    TRAJ_FEATURES,
    _auroc,
    _build_representation,
    _cv_auroc_baseline,
    extract_earlystop_signals_for_positions,
    _fit_svd_lr_model,
    _predict_svd_lr,
    _rank_transform_matrix,
    get_domain,
    load_earlystop_svd_bundle,
    save_earlystop_svd_bundle,
    score_cache_entry_earlystop_svd,
)
from nad.ops.earlystop_svm import (
    load_earlystop_svm_bundle,
    score_cache_entry_earlystop_from_bestofn_svm,
)


CONTROL_POSITIONS = (0.05, 0.10, 0.15, 0.20)
ANCHOR_POSITIONS = (0.10, 0.40, 0.70, 1.00)
OFFICIAL_SLOT_TO_ANCHOR = {
    0.10: 0.10,
    0.20: 0.10,
    0.30: 0.10,
    0.40: 0.40,
    0.50: 0.40,
    0.60: 0.40,
    0.70: 0.70,
    0.80: 0.70,
    0.90: 0.70,
    1.00: 1.00,
}

PREFIX_SAFE_META_FEATURES = ("nc_mean", "nc_slope")
PREFIX_SAFE_SIGNAL_NAMES = list(TOKEN_FEATURES) + list(TRAJ_FEATURES) + list(PREFIX_SAFE_META_FEATURES)
PREFIX_SAFE_FEATURE_FAMILY_MAP = {
    "token_only": list(TOKEN_FEATURES) + [
        "has_tok_conf",
        "has_tok_gini",
        "has_tok_neg_entropy",
        "has_tok_selfcert",
        "has_tok_logprob",
    ],
    "token_plus_traj": list(TOKEN_FEATURES) + list(TRAJ_FEATURES) + [
        "has_tok_conf",
        "has_tok_gini",
        "has_tok_neg_entropy",
        "has_tok_selfcert",
        "has_tok_logprob",
        "has_rows_bank",
    ],
    "all": list(TOKEN_FEATURES) + list(TRAJ_FEATURES) + list(PREFIX_SAFE_META_FEATURES) + list(AVAILABILITY_FEATURES),
}

SEARCH_FAMILIES = ("token_only", "token_plus_traj", "all")
SEARCH_REPRESENTATIONS = ("raw", "rank", "raw+rank")
SEARCH_RANKS = (2, 4, 6, 8, 12, 16)
SEARCH_C_VALUES = (0.05, 0.10, 0.20, 0.50, 1.00)
SEARCH_WHITEN = (False, True)
SEARCH_CLASS_WEIGHT = ("none", "balanced")
DEFAULT_FEATURE_CHUNK_PROBLEMS = 24

OFFICIAL_POSITION_INDEX = {float(p): idx for idx, p in enumerate(EARLY_STOP_POSITIONS)}
EXTRACTION_POSITIONS = tuple(sorted({float(p) for p in tuple(EARLY_STOP_POSITIONS) + CONTROL_POSITIONS}))
EXTRACTION_POSITION_INDEX = {float(p): idx for idx, p in enumerate(EXTRACTION_POSITIONS)}
FEATURE_TO_INDEX = {name: idx for idx, name in enumerate(FULL_FEATURE_NAMES)}

REPO_STATUS_LINES = [
    "repo 已经有 `earlystop_svd_lowrank_lr_v1` 的导出链路",
    "这次任务只做 Early-Stop",
    "目标是提高公开/综合 Early-Stop 分数，不是扩平台",
    "本轮核心假设：prefix 前 10% 已包含足够强的早停信号，值得单独训练一个 prefix-10 专用 low-rank 模型",
]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _float_key(p: float) -> str:
    return f"{float(p):.4f}"


def _pct_label(p: float) -> str:
    return f"{int(round(float(p) * 100.0))}%"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _empty_signal_map(positions: tuple[float, ...]) -> dict[str, list[float]]:
    return {name: [0.0] * len(positions) for name in FULL_FEATURE_NAMES}


def _prefix_mean(arr: Optional[np.ndarray], p: float) -> float:
    if arr is None:
        return 0.0
    t = int(arr.shape[0])
    if t == 0:
        return 0.0
    cut = max(1, int(float(p) * t))
    return float(np.mean(arr[:cut]))


def _prefix_recency(arr: Optional[np.ndarray], p: float, lam: float = 0.3) -> float:
    if arr is None:
        return 0.0
    t = int(arr.shape[0])
    if t == 0:
        return 0.0
    cut = max(1, int(float(p) * t))
    seg = arr[:cut]
    w = np.exp(lam * np.arange(cut, dtype=np.float64) / max(1, cut))
    return float(np.average(seg, weights=w))


def _prefix_tail_mean(arr: Optional[np.ndarray], p: float, tail_frac: float = 0.1) -> float:
    if arr is None:
        return 0.0
    t = int(arr.shape[0])
    if t == 0:
        return 0.0
    cut = max(1, int(float(p) * t))
    tail_w = max(1, int(tail_frac * cut))
    return float(np.mean(arr[max(0, cut - tail_w):cut]))


def _prefix_half_slope(arr: Optional[np.ndarray], p: float) -> float:
    if arr is None:
        return 0.0
    t = int(arr.shape[0])
    if t < 2:
        return 0.0
    cut = max(1, int(float(p) * t))
    if cut < 2:
        return 0.0
    half = cut // 2
    return float(np.mean(arr[half:cut]) - np.mean(arr[:half]))


def _prefix_count_slope(arr: Optional[np.ndarray], k: int) -> float:
    if arr is None:
        return 0.0
    if int(k) < 2:
        return 0.0
    cut = min(int(k), int(len(arr)))
    if cut < 2:
        return 0.0
    half = cut // 2
    if half <= 0 or half >= cut:
        return 0.0
    return float(np.mean(arr[half:cut]) - np.mean(arr[:half]))


def _parse_meta_groups(entry: CacheEntry) -> list[tuple[str, list[int]]]:
    meta = json.loads((entry.cache_root / "meta.json").read_text(encoding="utf-8"))
    groups = build_problem_groups(meta)
    return sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0]))


def _extract_slice_keysets_fast(reader: CacheReader, run_id: int) -> list[np.ndarray]:
    rows_srp = reader.rows_sample_row_ptr
    rows_rp = reader.rows_row_ptr
    rows_keys = reader.rows_keys
    if rows_srp is None or rows_rp is None or rows_keys is None:
        return []
    if run_id < 0 or run_id >= len(rows_srp) - 1:
        return []

    row_start = int(rows_srp[run_id])
    row_end = int(rows_srp[run_id + 1])
    slices: list[np.ndarray] = []
    for row_idx in range(row_start, row_end):
        key_start = int(rows_rp[row_idx])
        key_end = int(rows_rp[row_idx + 1])
        if key_end > key_start:
            slices.append(np.asarray(rows_keys[key_start:key_end], dtype=np.uint32))
        else:
            slices.append(np.empty(0, dtype=np.uint32))
    return slices


def _group_folds(groups: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return []
    splits = min(int(n_splits), int(len(unique_groups)))
    if splits < 2:
        return []
    dummy_x = np.zeros((len(groups), 1), dtype=np.float64)
    gkf = GroupKFold(n_splits=splits)
    return list(gkf.split(dummy_x, groups=groups))


def extract_signals_for_sample_at_positions(
    reader: CacheReader,
    run_id: int,
    positions: tuple[float, ...],
    required_features: Optional[set[str]] = None,
    reflection_threshold: float = DEFAULT_REFLECTION_THRESHOLD,
) -> dict[str, list[float]]:
    return extract_earlystop_signals_for_positions(
        reader=reader,
        run_id=int(run_id),
        positions=tuple(float(p) for p in positions),
        required_features=required_features,
        reflection_threshold=float(reflection_threshold),
    )


def _empty_feature_tensor(n_runs: int, positions: tuple[float, ...]) -> np.ndarray:
    return np.zeros((int(n_runs), len(positions), len(FULL_FEATURE_NAMES)), dtype=np.float64)


def _extract_entry_feature_payload(
    entry: CacheEntry,
    positions: tuple[float, ...],
    required_feature_names: set[str],
    max_problems_per_cache: Optional[int],
    reflection_threshold: float = DEFAULT_REFLECTION_THRESHOLD,
    problem_start: int = 0,
    problem_end: Optional[int] = None,
) -> dict[str, Any]:
    domain = get_domain(entry.dataset_name)
    try:
        correctness = load_correctness_map(str(entry.cache_root))
    except Exception:
        correctness = {}
    reader = CacheReader(str(entry.cache_root))

    tensor_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    sample_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    problem_ids: list[str] = []
    problem_offsets = [0]
    processed_samples = 0

    parsed_groups = _parse_meta_groups(entry)
    capped_total = len(parsed_groups)
    if max_problems_per_cache is not None:
        capped_total = min(capped_total, int(max_problems_per_cache))
    start_idx = max(0, int(problem_start))
    end_idx = capped_total if problem_end is None else min(capped_total, int(problem_end))

    for problem_idx, (problem_id, sample_ids_raw) in enumerate(parsed_groups[start_idx:end_idx], start=start_idx):
        if max_problems_per_cache is not None and problem_idx >= int(max_problems_per_cache):
            break
        sample_ids = [int(sample_id) for sample_id in sample_ids_raw]
        problem_tensor = _empty_feature_tensor(len(sample_ids), positions)
        problem_labels = np.zeros((len(sample_ids),), dtype=np.int32)

        for row_idx, sample_id in enumerate(sample_ids):
            problem_labels[row_idx] = int(bool(correctness.get(int(sample_id), False)))
            signal_map = extract_signals_for_sample_at_positions(
                reader,
                int(sample_id),
                positions=positions,
                required_features=required_feature_names,
                reflection_threshold=float(reflection_threshold),
            )
            for f_idx, feature_name in enumerate(FULL_FEATURE_NAMES):
                problem_tensor[row_idx, :, f_idx] = np.asarray(signal_map[feature_name], dtype=np.float64)

        problem_ids.append(str(problem_id))
        problem_offsets.append(problem_offsets[-1] + len(sample_ids))
        tensor_parts.append(problem_tensor)
        label_parts.append(problem_labels)
        sample_parts.append(np.asarray(sample_ids, dtype=np.int32))
        group_parts.append(np.asarray([f"{entry.cache_key}::{problem_id}"] * len(sample_ids), dtype=object))
        processed_samples += len(sample_ids)

    if tensor_parts:
        tensor = np.concatenate(tensor_parts, axis=0).astype(np.float64, copy=False)
        labels = np.concatenate(label_parts).astype(np.int32, copy=False)
        sample_ids = np.concatenate(sample_parts).astype(np.int32, copy=False)
        group_keys = np.concatenate(group_parts).astype(object, copy=False)
    else:
        tensor = _empty_feature_tensor(0, positions)
        labels = np.zeros((0,), dtype=np.int32)
        sample_ids = np.zeros((0,), dtype=np.int32)
        group_keys = np.asarray([], dtype=object)

    return {
        "cache_key": str(entry.cache_key),
        "dataset_name": str(entry.dataset_name),
        "domain": str(domain),
        "positions": [float(p) for p in positions],
        "tensor": tensor,
        "labels": labels,
        "sample_ids": sample_ids,
        "group_keys": group_keys,
        "problem_ids": problem_ids,
        "problem_offsets": problem_offsets,
        "samples": int(processed_samples),
        "problem_start": int(start_idx),
    }


def build_feature_store(
    cache_root: str | Path,
    positions: tuple[float, ...],
    required_feature_names: set[str],
    max_problems_per_cache: Optional[int] = None,
    max_workers: int = 1,
    chunk_problems: int = DEFAULT_FEATURE_CHUNK_PROBLEMS,
    include_cache_keys: Optional[set[str]] = None,
    exclude_cache_keys: Optional[set[str]] = None,
    reflection_threshold: float = DEFAULT_REFLECTION_THRESHOLD,
) -> list[dict[str, Any]]:
    entries = discover_cache_entries(cache_root)
    payload_chunks: list[dict[str, Any]] = []
    chunk_specs: list[tuple[CacheEntry, int, int, int, int]] = []
    for entry in entries:
        cache_key = str(entry.cache_key)
        if include_cache_keys is not None and cache_key not in include_cache_keys:
            continue
        if exclude_cache_keys is not None and cache_key in exclude_cache_keys:
            continue
        n_problems_total = len(_parse_meta_groups(entry))
        if max_problems_per_cache is not None:
            n_problems_total = min(n_problems_total, int(max_problems_per_cache))
        chunk = max(1, int(chunk_problems))
        n_chunks = max(1, int(math.ceil(n_problems_total / float(chunk))))
        for chunk_idx, start in enumerate(range(0, max(1, n_problems_total), chunk), start=1):
            end = min(n_problems_total, start + chunk)
            chunk_specs.append((entry, int(chunk_idx), int(n_chunks), int(start), int(end)))

    if int(max_workers) <= 1:
        for entry, chunk_idx, n_chunks, start, end in chunk_specs:
            print(f"[features] run   cache={entry.cache_key} chunk={chunk_idx}/{n_chunks} problems={start}:{end}")
            payload = _extract_entry_feature_payload(
                entry,
                positions,
                required_feature_names,
                max_problems_per_cache,
                float(reflection_threshold),
                start,
                end,
            )
            print(f"[features] done  cache={entry.cache_key} chunk={chunk_idx}/{n_chunks} samples={payload['samples']}")
            payload_chunks.append(payload)
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
            future_map = {}
            for entry, chunk_idx, n_chunks, start, end in chunk_specs:
                print(f"[features] queue cache={entry.cache_key} chunk={chunk_idx}/{n_chunks} problems={start}:{end}")
                future = executor.submit(
                    _extract_entry_feature_payload,
                    entry,
                    positions,
                    required_feature_names,
                    max_problems_per_cache,
                    float(reflection_threshold),
                    start,
                    end,
                )
                future_map[future] = (str(entry.cache_key), int(chunk_idx), int(n_chunks))

            for future in as_completed(future_map):
                payload = future.result()
                cache_key, chunk_idx, n_chunks = future_map[future]
                print(f"[features] done cache={cache_key} chunk={chunk_idx}/{n_chunks} samples={payload['samples']}")
                payload_chunks.append(payload)

    merged: dict[str, list[dict[str, Any]]] = {}
    for payload in payload_chunks:
        merged.setdefault(payload["cache_key"], []).append(payload)

    payloads: list[dict[str, Any]] = []
    for cache_key in sorted(merged.keys()):
        parts = sorted(merged[cache_key], key=lambda item: int(item["problem_start"]))
        base = parts[0]
        if len(parts) == 1:
            payloads.append(base)
            continue

        tensors = [part["tensor"] for part in parts]
        labels = [part["labels"] for part in parts]
        sample_ids = [part["sample_ids"] for part in parts]
        group_keys = [part["group_keys"] for part in parts]
        problem_ids: list[str] = []
        problem_offsets = [0]
        total_samples = 0
        for part in parts:
            problem_ids.extend(part["problem_ids"])
            local_widths = np.diff(np.asarray(part["problem_offsets"], dtype=np.int64))
            for width in local_widths.tolist():
                problem_offsets.append(problem_offsets[-1] + int(width))
            total_samples += int(part["samples"])

        payloads.append({
            "cache_key": str(base["cache_key"]),
            "dataset_name": str(base["dataset_name"]),
            "domain": str(base["domain"]),
            "positions": list(base["positions"]),
            "tensor": np.concatenate(tensors, axis=0).astype(np.float64, copy=False),
            "labels": np.concatenate(labels).astype(np.int32, copy=False),
            "sample_ids": np.concatenate(sample_ids).astype(np.int32, copy=False),
            "group_keys": np.concatenate(group_keys).astype(object, copy=False),
            "problem_ids": problem_ids,
            "problem_offsets": problem_offsets,
            "samples": int(total_samples),
        })

    return payloads


def build_scope_training_tables_from_feature_store(
    feature_store: list[dict[str, Any]],
    positions: tuple[float, ...],
) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    rows: dict[str, dict[int, list[np.ndarray]]] = {
        "global": {idx: [] for idx in range(len(positions))},
        "noncoding": {idx: [] for idx in range(len(positions))},
    }
    labels: dict[str, dict[int, list[np.ndarray]]] = {
        "global": {idx: [] for idx in range(len(positions))},
        "noncoding": {idx: [] for idx in range(len(positions))},
    }
    group_keys: dict[str, dict[int, list[np.ndarray]]] = {
        "global": {idx: [] for idx in range(len(positions))},
        "noncoding": {idx: [] for idx in range(len(positions))},
    }

    position_indices = [EXTRACTION_POSITION_INDEX[float(position)] for position in positions]
    for payload in feature_store:
        tensor = payload["tensor"]
        y = payload["labels"]
        groups = payload["group_keys"]
        if tensor.shape[0] == 0:
            continue
        for local_pos_idx, src_pos_idx in enumerate(position_indices):
            x_raw = tensor[:, src_pos_idx, :]
            rows["global"][local_pos_idx].append(x_raw)
            labels["global"][local_pos_idx].append(y)
            group_keys["global"][local_pos_idx].append(groups)
            if payload["domain"] in {"math", "science"}:
                rows["noncoding"][local_pos_idx].append(x_raw)
                labels["noncoding"][local_pos_idx].append(y)
                group_keys["noncoding"][local_pos_idx].append(groups)

    out: dict[str, dict[int, dict[str, np.ndarray]]] = {"global": {}, "noncoding": {}}
    for scope in ("global", "noncoding"):
        for pos_idx in range(len(positions)):
            if rows[scope][pos_idx]:
                x_raw = np.vstack(rows[scope][pos_idx]).astype(np.float64, copy=False)
                y = np.concatenate(labels[scope][pos_idx]).astype(np.int32, copy=False)
                groups = np.concatenate(group_keys[scope][pos_idx]).astype(object, copy=False)
            else:
                x_raw = np.zeros((0, len(FULL_FEATURE_NAMES)), dtype=np.float64)
                y = np.zeros((0,), dtype=np.int32)
                groups = np.asarray([], dtype=object)

            x_rank = np.zeros_like(x_raw)
            if x_raw.shape[0] > 0:
                by_group: dict[Any, list[int]] = {}
                for row_idx, group_key in enumerate(groups.tolist()):
                    by_group.setdefault(group_key, []).append(row_idx)
                for group_rows in by_group.values():
                    sub = x_raw[group_rows]
                    x_rank[group_rows] = _rank_transform_matrix(sub)

            out[scope][pos_idx] = {
                "x_raw": x_raw,
                "x_rank": x_rank,
                "y": y,
                "groups": groups,
            }
    return out


def _extract_entry_scope_payload(
    entry: CacheEntry,
    positions: tuple[float, ...],
    required_feature_names: set[str],
    max_problems_per_cache: Optional[int],
) -> dict[str, Any]:
    domain = get_domain(entry.dataset_name)
    correctness = load_correctness_map(str(entry.cache_root))
    reader = CacheReader(str(entry.cache_root))

    rows: dict[str, dict[int, list[np.ndarray]]] = {
        "global": {idx: [] for idx in range(len(positions))},
        "noncoding": {idx: [] for idx in range(len(positions))},
    }
    labels: dict[str, dict[int, list[int]]] = {
        "global": {idx: [] for idx in range(len(positions))},
        "noncoding": {idx: [] for idx in range(len(positions))},
    }
    groups: dict[str, dict[int, list[str]]] = {
        "global": {idx: [] for idx in range(len(positions))},
        "noncoding": {idx: [] for idx in range(len(positions))},
    }
    processed_samples = 0

    for problem_idx, (problem_id, sample_ids) in enumerate(_parse_meta_groups(entry)):
        if max_problems_per_cache is not None and problem_idx >= int(max_problems_per_cache):
            break
        group_id = f"{entry.cache_key}::{problem_id}"
        for sample_id in sample_ids:
            label = int(bool(correctness.get(int(sample_id), False)))
            signal_map = extract_signals_for_sample_at_positions(
                reader,
                int(sample_id),
                positions=positions,
                required_features=required_feature_names,
            )
            sample_mat = np.zeros((len(positions), len(FULL_FEATURE_NAMES)), dtype=np.float64)
            for f_idx, feature_name in enumerate(FULL_FEATURE_NAMES):
                sample_mat[:, f_idx] = np.asarray(signal_map[feature_name], dtype=np.float64)

            for pos_idx in range(len(positions)):
                rows["global"][pos_idx].append(sample_mat[pos_idx])
                labels["global"][pos_idx].append(label)
                groups["global"][pos_idx].append(group_id)
                if domain in {"math", "science"}:
                    rows["noncoding"][pos_idx].append(sample_mat[pos_idx])
                    labels["noncoding"][pos_idx].append(label)
                    groups["noncoding"][pos_idx].append(group_id)
            processed_samples += 1

    packed: dict[str, dict[int, dict[str, np.ndarray]]] = {"global": {}, "noncoding": {}}
    for scope in ("global", "noncoding"):
        for pos_idx in range(len(positions)):
            if rows[scope][pos_idx]:
                packed[scope][pos_idx] = {
                    "x_raw": np.vstack(rows[scope][pos_idx]).astype(np.float64, copy=False),
                    "y": np.asarray(labels[scope][pos_idx], dtype=np.int32),
                    "groups": np.asarray(groups[scope][pos_idx], dtype=object),
                }
            else:
                packed[scope][pos_idx] = {
                    "x_raw": np.zeros((0, len(FULL_FEATURE_NAMES)), dtype=np.float64),
                    "y": np.zeros((0,), dtype=np.int32),
                    "groups": np.asarray([], dtype=object),
                }

    return {
        "entry": str(entry.cache_key),
        "samples": int(processed_samples),
        "scopes": packed,
    }


def build_scope_training_tables(
    cache_root: str | Path,
    positions: tuple[float, ...],
    required_feature_names: set[str],
    max_problems_per_cache: Optional[int] = None,
    max_workers: int = 1,
) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    entries = discover_cache_entries(cache_root)
    rows: dict[str, dict[int, list[np.ndarray]]] = {
        "global": {idx: [] for idx in range(len(positions))},
        "noncoding": {idx: [] for idx in range(len(positions))},
    }
    labels: dict[str, dict[int, list[np.ndarray]]] = {
        "global": {idx: [] for idx in range(len(positions))},
        "noncoding": {idx: [] for idx in range(len(positions))},
    }
    group_keys: dict[str, dict[int, list[np.ndarray]]] = {
        "global": {idx: [] for idx in range(len(positions))},
        "noncoding": {idx: [] for idx in range(len(positions))},
    }

    with ProcessPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        future_map = {}
        for entry in entries:
            print(f"[table] queue cache={entry.cache_key}")
            future = executor.submit(
                _extract_entry_scope_payload,
                entry,
                positions,
                required_feature_names,
                max_problems_per_cache,
            )
            future_map[future] = str(entry.cache_key)

        for future in as_completed(future_map):
            payload = future.result()
            print(f"[table] done cache={payload['entry']} samples={payload['samples']}")
            for scope in ("global", "noncoding"):
                for pos_idx in range(len(positions)):
                    chunk = payload["scopes"][scope][pos_idx]
                    if chunk["x_raw"].shape[0] == 0:
                        continue
                    rows[scope][pos_idx].append(chunk["x_raw"])
                    labels[scope][pos_idx].append(chunk["y"])
                    group_keys[scope][pos_idx].append(chunk["groups"])

    out: dict[str, dict[int, dict[str, np.ndarray]]] = {"global": {}, "noncoding": {}}
    for scope in ("global", "noncoding"):
        for pos_idx in range(len(positions)):
            if rows[scope][pos_idx]:
                x_raw = np.vstack(rows[scope][pos_idx]).astype(np.float64, copy=False)
                y = np.concatenate(labels[scope][pos_idx]).astype(np.int32, copy=False)
                groups = np.concatenate(group_keys[scope][pos_idx]).astype(object, copy=False)
            else:
                x_raw = np.zeros((0, len(FULL_FEATURE_NAMES)), dtype=np.float64)
                y = np.zeros((0,), dtype=np.int32)
                groups = np.asarray([], dtype=object)

            x_rank = np.zeros_like(x_raw)
            if x_raw.shape[0] > 0:
                by_group: dict[Any, list[int]] = {}
                for row_idx, group_key in enumerate(groups.tolist()):
                    by_group.setdefault(group_key, []).append(row_idx)
                for group_rows in by_group.values():
                    sub = x_raw[group_rows]
                    x_rank[group_rows] = _rank_transform_matrix(sub)

            out[scope][pos_idx] = {
                "x_raw": x_raw,
                "x_rank": x_rank,
                "y": y,
                "groups": groups,
            }
    return out


def fit_route_for_table(
    *,
    x_raw: np.ndarray,
    x_rank: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    position: float,
    training_scope: str,
    n_splits: int,
    random_state: int,
) -> dict[str, Any]:
    if x_raw.shape[0] == 0 or np.unique(y).shape[0] < 2:
        return {
            "route_type": "baseline",
            "signal_name": "tok_conf_prefix",
            "cv_auroc": float("nan"),
            "n_valid_folds": 0,
            "training_position": float(position),
            "training_scope": str(training_scope),
            "note": "insufficient labeled data",
        }

    baseline_candidates = [name for name in PREFIX_SAFE_SIGNAL_NAMES if name in FEATURE_TO_INDEX]
    best_baseline = {
        "signal_name": "tok_conf_prefix",
        "cv_auroc": float("-inf"),
        "n_valid_folds": 0,
    }
    for signal_name in baseline_candidates:
        score_col = FEATURE_TO_INDEX[signal_name]
        score_vec = x_raw[:, score_col]
        cv_auc, n_folds = _cv_auroc_baseline(
            scores=score_vec,
            y=y,
            groups=groups,
            n_splits=n_splits,
        )
        if np.isfinite(cv_auc) and cv_auc > float(best_baseline["cv_auroc"]):
            best_baseline = {
                "signal_name": signal_name,
                "cv_auroc": float(cv_auc),
                "n_valid_folds": int(n_folds),
            }

    best_svd: dict[str, Any] = {"cv_auroc": float("-inf")}
    folds = _group_folds(groups, n_splits=n_splits)

    for family_name in SEARCH_FAMILIES:
        family_features = PREFIX_SAFE_FEATURE_FAMILY_MAP[family_name]
        feature_indices = [FEATURE_TO_INDEX[name] for name in family_features]
        for representation in SEARCH_REPRESENTATIONS:
            x_rep = _build_representation(
                x_raw=x_raw,
                x_rank=x_rank,
                feature_indices=feature_indices,
                representation=representation,
            )
            if not folds:
                continue

            candidate_scores: dict[tuple[int, float, bool, str], list[float]] = {}
            for train_idx, test_idx in folds:
                y_train = y[train_idx]
                y_test = y[test_idx]
                if np.unique(y_train).shape[0] < 2 or np.unique(y_test).shape[0] < 2:
                    continue

                scaler = StandardScaler(with_mean=True, with_std=True)
                x_train_scaled = scaler.fit_transform(x_rep[train_idx])
                x_test_scaled = scaler.transform(x_rep[test_idx])

                max_rank = min(int(max(SEARCH_RANKS)), int(x_train_scaled.shape[1]), int(x_train_scaled.shape[0] - 1))
                if max_rank < 1:
                    continue

                svd = TruncatedSVD(n_components=max_rank, random_state=int(random_state))
                z_train_full = svd.fit_transform(x_train_scaled)
                z_test_full = svd.transform(x_test_scaled)
                singular_values = np.asarray(svd.singular_values_, dtype=np.float64)
                singular_values = np.where(np.abs(singular_values) < 1e-8, 1.0, singular_values)

                valid_ranks = [int(rank) for rank in SEARCH_RANKS if int(rank) <= max_rank]
                for rank in valid_ranks:
                    z_train = z_train_full[:, :rank]
                    z_test = z_test_full[:, :rank]
                    for whiten in SEARCH_WHITEN:
                        if whiten:
                            scale = singular_values[:rank]
                            z_train_use = z_train / scale
                            z_test_use = z_test / scale
                        else:
                            z_train_use = z_train
                            z_test_use = z_test
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
                                    candidate_scores.setdefault(
                                        (int(rank), float(c_value), bool(whiten), str(class_weight)),
                                        [],
                                    ).append(float(fold_auc))

            for (rank, c_value, whiten, class_weight), values in candidate_scores.items():
                if not values:
                    continue
                cv_auc = float(np.mean(values))
                n_folds = int(len(values))
                if cv_auc > float(best_svd["cv_auroc"]):
                    best_svd = {
                        "cv_auroc": float(cv_auc),
                        "n_valid_folds": int(n_folds),
                        "family_name": str(family_name),
                        "representation": str(representation),
                        "rank": int(rank),
                        "c_value": float(c_value),
                        "whiten": bool(whiten),
                        "class_weight": str(class_weight),
                        "feature_names": list(family_features),
                        "feature_indices": list(feature_indices),
                    }

    baseline_auc = float(best_baseline["cv_auroc"]) if np.isfinite(best_baseline["cv_auroc"]) else float("-inf")
    svd_auc = float(best_svd["cv_auroc"]) if np.isfinite(best_svd["cv_auroc"]) else float("-inf")

    if svd_auc > baseline_auc and np.isfinite(svd_auc):
        x_rep_full = _build_representation(
            x_raw=x_raw,
            x_rank=x_rank,
            feature_indices=best_svd["feature_indices"],
            representation=best_svd["representation"],
        )
        model = _fit_svd_lr_model(
            x=x_rep_full,
            y=y,
            rank=best_svd["rank"],
            c_value=best_svd["c_value"],
            whiten=best_svd["whiten"],
            class_weight_name=best_svd["class_weight"],
            random_state=random_state,
        )
        if model is not None:
            return {
                "route_type": "svd",
                "cv_auroc": float(best_svd["cv_auroc"]),
                "n_valid_folds": int(best_svd["n_valid_folds"]),
                "family_name": str(best_svd["family_name"]),
                "representation": str(best_svd["representation"]),
                "rank": int(best_svd["rank"]),
                "c_value": float(best_svd["c_value"]),
                "whiten": bool(best_svd["whiten"]),
                "class_weight": str(best_svd["class_weight"]),
                "feature_names": list(best_svd["feature_names"]),
                "feature_indices": list(best_svd["feature_indices"]),
                "baseline_signal_name": best_baseline["signal_name"],
                "baseline_cv_auroc": float(best_baseline["cv_auroc"]),
                "training_position": float(position),
                "training_scope": str(training_scope),
                "model": model,
            }

    return {
        "route_type": "baseline",
        "signal_name": best_baseline["signal_name"],
        "cv_auroc": float(best_baseline["cv_auroc"]),
        "n_valid_folds": int(best_baseline["n_valid_folds"]),
        "svd_best_cv_auroc": None if not np.isfinite(svd_auc) else float(svd_auc),
        "training_position": float(position),
        "training_scope": str(training_scope),
    }


def summarise_route(route: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in route.items() if key != "model"}


def train_routes_for_scope(
    tables: list[dict[str, np.ndarray]],
    positions: tuple[float, ...],
    scope_name: str,
    n_splits: int,
    random_state: int,
    max_workers: int = 1,
) -> dict[float, dict[str, Any]]:
    routes: dict[float, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        future_map = {}
        for pos_idx, position in enumerate(positions):
            table = tables[pos_idx]
            future = executor.submit(
                fit_route_for_table,
                x_raw=table["x_raw"],
                x_rank=table["x_rank"],
                y=table["y"],
                groups=table["groups"],
                position=position,
                training_scope=scope_name,
                n_splits=n_splits,
                random_state=random_state,
            )
            future_map[future] = float(position)

        for future in as_completed(future_map):
            position = future_map[future]
            route = future.result()
            routes[float(position)] = route
            if route["route_type"] == "svd":
                print(
                    f"[train] scope={scope_name:<9s} pos={_pct_label(position):>4s} "
                    f"route=svd auc={route['cv_auroc']:.4f} family={route['family_name']} rep={route['representation']}"
                )
            else:
                print(
                    f"[train] scope={scope_name:<9s} pos={_pct_label(position):>4s} "
                    f"route=baseline auc={route['cv_auroc']:.4f} signal={route['signal_name']}"
                )
    return routes


def build_anchor_bundle(
    *,
    bundle_version: str,
    anchor_routes: dict[float, dict[str, Any]],
    coding_routes: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    bundle = {
        "bundle_version": str(bundle_version),
        "created_at_utc": _now_utc(),
        "feature_names": list(FULL_FEATURE_NAMES),
        "positions": list(EARLY_STOP_POSITIONS),
        "domains": {},
    }
    shared_routes = [anchor_routes[float(OFFICIAL_SLOT_TO_ANCHOR[float(position)])] for position in EARLY_STOP_POSITIONS]
    for domain in ("math", "science"):
        bundle["domains"][domain] = {"routes": shared_routes}
    if coding_routes is None:
        bundle["domains"]["coding"] = {"routes": shared_routes}
    else:
        bundle["domains"]["coding"] = {"routes": coding_routes}
    return bundle


def score_cache_entry_custom_anchor_bundle(
    entry: CacheEntry,
    bundle: dict[str, Any],
    max_problems: Optional[int] = None,
) -> dict[str, dict[str, list[float]]]:
    return score_cache_entry_earlystop_svd(entry, bundle, max_problems=max_problems)


def evaluate_problem_scores(
    entry: CacheEntry,
    problem_scores: dict[str, dict[str, list[float]]],
) -> dict[str, Any]:
    correctness = load_correctness_map(str(entry.cache_root))
    per_position_auroc: list[float] = []
    per_position_selacc: list[float] = []
    per_position_stop_acc: list[float] = []

    n_samples = 0
    for position_index in range(N_POSITIONS):
        score_rows: list[tuple[float, int, str, int]] = []
        hit_rows: list[int] = []
        for problem_id, sample_map in sorted(problem_scores.items(), key=lambda kv: _problem_sort_key(kv[0])):
            ordered_samples = sorted(
                ((int(sample_id), float(score_list[position_index])) for sample_id, score_list in sample_map.items()),
                key=lambda item: (-item[1], item[0]),
            )
            if ordered_samples:
                best_sample_id = ordered_samples[0][0]
                hit_rows.append(int(bool(correctness.get(int(best_sample_id), False))))
            for sample_id, score in ordered_samples:
                label = int(bool(correctness.get(int(sample_id), False)))
                score_rows.append((float(score), label, str(problem_id), int(sample_id)))

        n_samples = max(n_samples, len(score_rows))
        scores = np.asarray([row[0] for row in score_rows], dtype=np.float64)
        labels = np.asarray([row[1] for row in score_rows], dtype=np.int32)
        auroc = _auroc(scores, labels)
        per_position_auroc.append(float(auroc))

        topk = max(1, int(math.ceil(0.10 * max(1, len(score_rows)))))
        top_rows = sorted(score_rows, key=lambda row: (-row[0], row[3]))[:topk]
        per_position_selacc.append(float(np.mean([row[1] for row in top_rows])) if top_rows else 0.0)
        per_position_stop_acc.append(float(np.mean(hit_rows)) if hit_rows else 0.0)

    earliest = None
    for position, auroc in zip(EARLY_STOP_POSITIONS, per_position_auroc):
        if float(auroc) > 0.60:
            earliest = float(position)
            break

    return {
        "cache_key": str(entry.cache_key),
        "dataset_name": str(entry.dataset_name),
        "domain": str(get_domain(entry.dataset_name)),
        "n_problems": int(len(problem_scores)),
        "n_samples": int(n_samples),
        "position_metrics": [
            {
                "position": float(position),
                "auroc": float(per_position_auroc[idx]),
                "selacc@10%": float(per_position_selacc[idx]),
                "stop_acc": float(per_position_stop_acc[idx]),
            }
            for idx, position in enumerate(EARLY_STOP_POSITIONS)
        ],
        "auc_of_auroc": float(np.mean(per_position_auroc)) if per_position_auroc else float("nan"),
        "auc_of_selacc": float(np.mean(per_position_selacc)) if per_position_selacc else float("nan"),
        "earliest_gt_0.6": None if earliest is None else float(earliest),
        "auroc@100%": float(per_position_auroc[-1]) if per_position_auroc else float("nan"),
        "stop_acc@100%": float(per_position_stop_acc[-1]) if per_position_stop_acc else float("nan"),
    }


def aggregate_cache_metrics(cache_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not cache_metrics:
        return {
            "num_caches": 0,
            "samples": 0,
            "auc_of_auroc": float("nan"),
            "auc_of_selacc": float("nan"),
            "auroc@100%": float("nan"),
            "stop_acc@100%": float("nan"),
            "by_position": [],
        }

    by_position = []
    for pos_idx, position in enumerate(EARLY_STOP_POSITIONS):
        by_position.append({
            "position": float(position),
            "auroc": float(np.mean([cache["position_metrics"][pos_idx]["auroc"] for cache in cache_metrics])),
            "selacc@10%": float(np.mean([cache["position_metrics"][pos_idx]["selacc@10%"] for cache in cache_metrics])),
            "stop_acc": float(np.mean([cache["position_metrics"][pos_idx]["stop_acc"] for cache in cache_metrics])),
        })

    aggregate_earliest = None
    for row in by_position:
        if float(row["auroc"]) > 0.60:
            aggregate_earliest = float(row["position"])
            break

    return {
        "num_caches": int(len(cache_metrics)),
        "samples": int(sum(int(cache["n_samples"]) for cache in cache_metrics)),
        "auc_of_auroc": float(np.mean([cache["auc_of_auroc"] for cache in cache_metrics])),
        "auc_of_selacc": float(np.mean([cache["auc_of_selacc"] for cache in cache_metrics])),
        "earliest_gt_0.6": None if aggregate_earliest is None else float(aggregate_earliest),
        "auroc@100%": float(np.mean([cache["auroc@100%"] for cache in cache_metrics])),
        "stop_acc@100%": float(np.mean([cache["stop_acc@100%"] for cache in cache_metrics])),
        "by_position": by_position,
    }


def _compute_cache_metrics_from_rows(
    *,
    cache_key: str,
    dataset_name: str,
    domain: str,
    position_values: tuple[float, ...],
    score_rows_by_pos: list[list[tuple[float, int, str, int]]],
    hit_rows_by_pos: list[list[int]],
    n_samples: int,
    n_problems: int,
) -> dict[str, Any]:
    per_position_auroc: list[float] = []
    per_position_selacc: list[float] = []
    per_position_stop_acc: list[float] = []

    for pos_idx in range(len(position_values)):
        score_rows = score_rows_by_pos[pos_idx]
        hit_rows = hit_rows_by_pos[pos_idx]
        scores = np.asarray([row[0] for row in score_rows], dtype=np.float64)
        labels = np.asarray([row[1] for row in score_rows], dtype=np.int32)
        per_position_auroc.append(float(_auroc(scores, labels)))

        topk = max(1, int(math.ceil(0.10 * max(1, len(score_rows)))))
        top_rows = sorted(score_rows, key=lambda row: (-row[0], row[3]))[:topk]
        per_position_selacc.append(float(np.mean([row[1] for row in top_rows])) if top_rows else 0.0)
        per_position_stop_acc.append(float(np.mean(hit_rows)) if hit_rows else 0.0)

    earliest = None
    for position, auroc in zip(position_values, per_position_auroc):
        if float(auroc) > 0.60:
            earliest = float(position)
            break

    return {
        "cache_key": str(cache_key),
        "dataset_name": str(dataset_name),
        "domain": str(domain),
        "n_problems": int(n_problems),
        "n_samples": int(n_samples),
        "position_metrics": [
            {
                "position": float(position),
                "auroc": float(per_position_auroc[idx]),
                "selacc@10%": float(per_position_selacc[idx]),
                "stop_acc": float(per_position_stop_acc[idx]),
            }
            for idx, position in enumerate(position_values)
        ],
        "auc_of_auroc": float(np.mean(per_position_auroc)) if per_position_auroc else float("nan"),
        "auc_of_selacc": float(np.mean(per_position_selacc)) if per_position_selacc else float("nan"),
        "earliest_gt_0.6": None if earliest is None else float(earliest),
        "auroc@100%": float(per_position_auroc[-1]) if per_position_auroc else float("nan"),
        "stop_acc@100%": float(per_position_stop_acc[-1]) if per_position_stop_acc else float("nan"),
    }


def _score_xraw_with_route(
    *,
    x_raw: np.ndarray,
    route: dict[str, Any],
    feature_to_idx: dict[str, int],
) -> np.ndarray:
    if route["route_type"] == "baseline":
        score_col = feature_to_idx[str(route["signal_name"])]
        return np.asarray(x_raw[:, score_col], dtype=np.float64)

    x_rank = _rank_transform_matrix(x_raw)
    feat_indices = [int(v) for v in route["feature_indices"]]
    representation = str(route["representation"])
    x_rep = _build_representation(
        x_raw=x_raw,
        x_rank=x_rank,
        feature_indices=feat_indices,
        representation=representation,
    )

    if route["route_type"] == "svd":
        return _predict_svd_lr(route["model"], x_rep)
    if route["route_type"] in {"pointwise", "ranksvm"}:
        return np.asarray(route["scorer"].score_group(x_rep), dtype=np.float64)
    raise ValueError(f"Unknown route type: {route['route_type']}")


def _evaluate_entry_from_feature_store(
    payload: dict[str, Any],
    position_values: tuple[float, ...],
    score_fn: Callable[[str, int, np.ndarray], np.ndarray],
) -> dict[str, Any]:
    src_position_indices = [EXTRACTION_POSITION_INDEX[float(position)] for position in position_values]
    score_rows_by_pos: list[list[tuple[float, int, str, int]]] = [[] for _ in position_values]
    hit_rows_by_pos: list[list[int]] = [[] for _ in position_values]

    tensor = payload["tensor"]
    labels_all = payload["labels"]
    sample_ids_all = payload["sample_ids"]
    problem_ids = payload["problem_ids"]
    problem_offsets = payload["problem_offsets"]

    for problem_idx, problem_id in enumerate(problem_ids):
        start = int(problem_offsets[problem_idx])
        end = int(problem_offsets[problem_idx + 1])
        sample_ids = sample_ids_all[start:end]
        labels = labels_all[start:end]
        problem_tensor = tensor[start:end]

        for out_pos_idx, src_pos_idx in enumerate(src_position_indices):
            x_raw = problem_tensor[:, src_pos_idx, :]
            scores = np.asarray(score_fn(payload["domain"], out_pos_idx, x_raw), dtype=np.float64)
            order = np.lexsort((sample_ids, -scores))
            ordered_scores = scores[order]
            ordered_labels = labels[order]
            ordered_ids = sample_ids[order]
            if ordered_labels.size > 0:
                hit_rows_by_pos[out_pos_idx].append(int(ordered_labels[0]))
            for score, label, sample_id in zip(ordered_scores, ordered_labels, ordered_ids):
                score_rows_by_pos[out_pos_idx].append((float(score), int(label), str(problem_id), int(sample_id)))

    return _compute_cache_metrics_from_rows(
        cache_key=payload["cache_key"],
        dataset_name=payload["dataset_name"],
        domain=payload["domain"],
        position_values=position_values,
        score_rows_by_pos=score_rows_by_pos,
        hit_rows_by_pos=hit_rows_by_pos,
        n_samples=int(sample_ids_all.shape[0]),
        n_problems=int(len(problem_ids)),
    )


def evaluate_method_from_feature_store(
    *,
    method_name: str,
    feature_store: list[dict[str, Any]],
    position_values: tuple[float, ...],
    score_fn: Callable[[str, int, np.ndarray], np.ndarray],
) -> dict[str, Any]:
    caches = []
    print(f"[eval] start method={method_name}")
    for payload in feature_store:
        print(f"[eval]   method={method_name} cache={payload['cache_key']}")
        caches.append(_evaluate_entry_from_feature_store(payload, position_values, score_fn))
    print(f"[eval] done method={method_name}")
    return {
        "method_name": str(method_name),
        "aggregate": aggregate_cache_metrics(caches),
        "by_cache": caches,
    }


def make_tok_conf_score_fn() -> Callable[[str, int, np.ndarray], np.ndarray]:
    score_col = FEATURE_TO_INDEX["tok_conf_prefix"]

    def _score(_domain: str, _position_index: int, x_raw: np.ndarray) -> np.ndarray:
        return np.asarray(x_raw[:, score_col], dtype=np.float64)

    return _score


def make_svd_bundle_score_fn(bundle: dict[str, Any]) -> Callable[[str, int, np.ndarray], np.ndarray]:
    feature_to_idx = {name: i for i, name in enumerate(bundle["feature_names"])}

    def _score(domain: str, position_index: int, x_raw: np.ndarray) -> np.ndarray:
        route = bundle["domains"][domain]["routes"][position_index]
        return _score_xraw_with_route(x_raw=x_raw, route=route, feature_to_idx=feature_to_idx)

    return _score


def make_bridge_score_fn(bundle: dict[str, Any]) -> Callable[[str, int, np.ndarray], np.ndarray]:
    feature_to_idx = {name: i for i, name in enumerate(bundle["feature_names"])}

    def _score(domain: str, _position_index: int, x_raw: np.ndarray) -> np.ndarray:
        route = bundle["domains"][domain]["route"]
        return _score_xraw_with_route(x_raw=x_raw, route=route, feature_to_idx=feature_to_idx)

    return _score


def make_single_route_score_fn(
    route_resolver: Callable[[str], Optional[dict[str, Any]]],
) -> Callable[[str, int, np.ndarray], np.ndarray]:
    def _score(domain: str, _position_index: int, x_raw: np.ndarray) -> np.ndarray:
        route = route_resolver(domain)
        if route is None:
            raise ValueError(f"No route for domain={domain}")
        return _score_xraw_with_route(x_raw=x_raw, route=route, feature_to_idx=FEATURE_TO_INDEX)

    return _score


def evaluate_single_position_route_from_feature_store(
    *,
    feature_store: list[dict[str, Any]],
    position: float,
    route_resolver: Callable[[str], Optional[dict[str, Any]]],
) -> dict[str, Any]:
    cache_rows = []
    position_value = (float(position),)
    score_fn = make_single_route_score_fn(route_resolver)

    for payload in feature_store:
        route = route_resolver(payload["domain"])
        if route is None:
            continue
        metrics = _evaluate_entry_from_feature_store(payload, position_value, score_fn)
        pos_metrics = metrics["position_metrics"][0]
        cache_rows.append({
            "cache_key": str(payload["cache_key"]),
            "dataset_name": str(payload["dataset_name"]),
            "domain": str(payload["domain"]),
            "n_samples": int(metrics["n_samples"]),
            "n_problems": int(metrics["n_problems"]),
            "auroc": float(pos_metrics["auroc"]),
            "selacc@10%": float(pos_metrics["selacc@10%"]),
            "stop_acc": float(pos_metrics["stop_acc"]),
        })

    aggregate = {
        "position": float(position),
        "num_caches": int(len(cache_rows)),
        "auroc": float(np.mean([row["auroc"] for row in cache_rows])) if cache_rows else float("nan"),
        "selacc@10%": float(np.mean([row["selacc@10%"] for row in cache_rows])) if cache_rows else float("nan"),
        "stop_acc": float(np.mean([row["stop_acc"] for row in cache_rows])) if cache_rows else float("nan"),
    }
    return {
        "position": float(position),
        "aggregate": aggregate,
        "by_cache": cache_rows,
    }


def evaluate_full_method(
    *,
    method_name: str,
    entries: list[CacheEntry],
    scorer: Callable[[CacheEntry], dict[str, dict[str, list[float]]]],
) -> dict[str, Any]:
    caches = []
    print(f"[eval] start method={method_name}")
    for entry in entries:
        print(f"[eval]   method={method_name} cache={entry.cache_key}")
        problem_scores = scorer(entry)
        caches.append(evaluate_problem_scores(entry, problem_scores))
    print(f"[eval] done method={method_name}")
    return {
        "method_name": str(method_name),
        "aggregate": aggregate_cache_metrics(caches),
        "by_cache": caches,
    }


def evaluate_single_position_route(
    *,
    entries: list[CacheEntry],
    position: float,
    route_resolver: Callable[[str], Optional[dict[str, Any]]],
    max_problems: Optional[int] = None,
) -> dict[str, Any]:
    cache_rows = []
    required_features: set[str] = set()
    for domain in ("math", "science", "coding"):
        route = route_resolver(domain)
        if route is None:
            continue
        if route["route_type"] == "baseline":
            required_features.add(str(route["signal_name"]))
        else:
            required_features.update(str(name) for name in route["feature_names"])

    for entry in entries:
        domain = get_domain(entry.dataset_name)
        route = route_resolver(domain)
        if route is None:
            continue
        correctness = load_correctness_map(str(entry.cache_root))
        reader = CacheReader(str(entry.cache_root))
        per_problem = _parse_meta_groups(entry)
        score_rows: list[tuple[float, int, str, int]] = []
        hit_rows: list[int] = []
        for problem_idx, (problem_id, sample_ids_raw) in enumerate(per_problem):
            if max_problems is not None and problem_idx >= int(max_problems):
                break
            sample_ids = [int(sample_id) for sample_id in sample_ids_raw]
            tensor = np.zeros((len(sample_ids), len(FULL_FEATURE_NAMES)), dtype=np.float64)
            for row_idx, sample_id in enumerate(sample_ids):
                signal_map = extract_signals_for_sample_at_positions(
                    reader,
                    sample_id,
                    positions=(float(position),),
                    required_features=required_features,
                )
                for feature_idx, feature_name in enumerate(FULL_FEATURE_NAMES):
                    tensor[row_idx, feature_idx] = float(signal_map[feature_name][0])

            x_rank = _rank_transform_matrix(tensor)
            if route["route_type"] == "baseline":
                scores = tensor[:, FEATURE_TO_INDEX[str(route["signal_name"])]]
            else:
                x_rep = _build_representation(
                    x_raw=tensor,
                    x_rank=x_rank,
                    feature_indices=[int(idx) for idx in route["feature_indices"]],
                    representation=str(route["representation"]),
                )
                scores = _predict_svd_lr(route["model"], x_rep)

            ordered = sorted(
                ((sample_ids[idx], float(scores[idx])) for idx in range(len(sample_ids))),
                key=lambda item: (-item[1], item[0]),
            )
            if ordered:
                hit_rows.append(int(bool(correctness.get(int(ordered[0][0]), False))))
            for sample_id, score in ordered:
                score_rows.append((float(score), int(bool(correctness.get(int(sample_id), False))), str(problem_id), int(sample_id)))

        scores_arr = np.asarray([row[0] for row in score_rows], dtype=np.float64)
        labels_arr = np.asarray([row[1] for row in score_rows], dtype=np.int32)
        topk = max(1, int(math.ceil(0.10 * max(1, len(score_rows)))))
        top_rows = sorted(score_rows, key=lambda row: (-row[0], row[3]))[:topk]
        cache_rows.append({
            "cache_key": str(entry.cache_key),
            "dataset_name": str(entry.dataset_name),
            "domain": str(domain),
            "n_samples": int(len(score_rows)),
            "n_problems": int(len(per_problem)),
            "auroc": float(_auroc(scores_arr, labels_arr)),
            "selacc@10%": float(np.mean([row[1] for row in top_rows])) if top_rows else 0.0,
            "stop_acc": float(np.mean(hit_rows)) if hit_rows else 0.0,
        })

    aggregate = {
        "position": float(position),
        "num_caches": int(len(cache_rows)),
        "auroc": float(np.mean([row["auroc"] for row in cache_rows])) if cache_rows else float("nan"),
        "selacc@10%": float(np.mean([row["selacc@10%"] for row in cache_rows])) if cache_rows else float("nan"),
        "stop_acc": float(np.mean([row["stop_acc"] for row in cache_rows])) if cache_rows else float("nan"),
    }
    return {
        "position": float(position),
        "aggregate": aggregate,
        "by_cache": cache_rows,
    }


def choose_best_candidate(new_methods: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        new_methods,
        key=lambda item: (
            float(item["aggregate"]["auc_of_selacc"]),
            float(item["aggregate"]["auc_of_auroc"]),
            float(item["aggregate"]["auroc@100%"]),
            float(item["aggregate"]["stop_acc@100%"]),
            str(item["method_name"]),
        ),
    )


def decide_export(best_new: dict[str, Any], baseline_v1: dict[str, Any]) -> dict[str, Any]:
    cand = best_new["aggregate"]
    base = baseline_v1["aggregate"]
    beats = (
        float(cand["auc_of_selacc"]) > float(base["auc_of_selacc"])
        and float(cand["auc_of_auroc"]) >= float(base["auc_of_auroc"])
        and float(cand["auroc@100%"]) >= float(base["auroc@100%"])
        and float(cand["stop_acc@100%"]) >= float(base["stop_acc@100%"])
    )
    if beats:
        reason = "offline direct-eval strict dominance over earlystop_svd_lowrank_lr_v1"
    else:
        failures = []
        if float(cand["auc_of_selacc"]) <= float(base["auc_of_selacc"]):
            failures.append("AUC of SelAcc 未超过 v1")
        if float(cand["auc_of_auroc"]) < float(base["auc_of_auroc"]):
            failures.append("AUC of AUROC 低于 v1")
        if float(cand["auroc@100%"]) < float(base["auroc@100%"]):
            failures.append("AUROC@100% 低于 v1")
        if float(cand["stop_acc@100%"]) < float(base["stop_acc@100%"]):
            failures.append("Stop Acc@100% 低于 v1")
        reason = "；".join(failures) if failures else "未满足严格胜出条件"
    return {
        "winner_method_name": str(best_new["method_name"]),
        "export_recommended": bool(beats),
        "reason": str(reason),
    }


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if not np.isfinite(float(value)):
        return "N/A"
    return f"{100.0 * float(value):.2f}%"


def _fmt_earliest(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return _pct_label(float(value))


def _render_control_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Checkpoint | AUROC | SelAcc@10 | StopAcc |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        agg = row["aggregate"]
        lines.append(
            "| {position} | {auroc} | {selacc} | {stop} |".format(
                position=_pct_label(row["position"]),
                auroc=_fmt_pct(agg["auroc"]),
                selacc=_fmt_pct(agg["selacc@10%"]),
                stop=_fmt_pct(agg["stop_acc"]),
            )
        )
    lines.append("")
    return lines


def _render_method_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        agg = row["aggregate"]
        lines.append(
            "| {name} | {auc_auroc} | {auc_selacc} | {earliest} | {auroc100} | {stop100} |".format(
                name=row["method_name"],
                auc_auroc=_fmt_pct(agg["auc_of_auroc"]),
                auc_selacc=_fmt_pct(agg["auc_of_selacc"]),
                earliest=_fmt_earliest(agg.get("earliest_gt_0.6")),
                auroc100=_fmt_pct(agg["auroc@100%"]),
                stop100=_fmt_pct(agg["stop_acc@100%"]),
            )
        )
    lines.append("")
    return lines


def _render_cache_table(method_eval: dict[str, Any]) -> list[str]:
    lines = [
        f"### {method_eval['method_name']} per-cache",
        "",
        "| Cache | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in method_eval["by_cache"]:
        lines.append(
            "| {cache_key} | {auc_auroc} | {auc_selacc} | {earliest} | {auroc100} | {stop100} |".format(
                cache_key=row["cache_key"],
                auc_auroc=_fmt_pct(row["auc_of_auroc"]),
                auc_selacc=_fmt_pct(row["auc_of_selacc"]),
                earliest=_fmt_earliest(row["earliest_gt_0.6"]),
                auroc100=_fmt_pct(row["auroc@100%"]),
                stop100=_fmt_pct(row["stop_acc@100%"]),
            )
        )
    lines.append("")
    return lines


def write_results_doc(
    *,
    path: Path,
    summary: dict[str, Any],
    changed_files: list[str],
) -> None:
    best_new_name = summary["decision"]["winner_method_name"]
    best_new = summary["candidates"][best_new_name]
    baseline_v1 = summary["baselines"]["earlystop_svd_lowrank_lr_v1"]
    global_controls = summary["controls"]["global"]
    noncoding_controls = summary["controls"]["noncoding"]
    global_best_control = max(global_controls, key=lambda row: float(row["aggregate"]["auroc"]))
    noncoding_best_control = max(noncoding_controls, key=lambda row: float(row["aggregate"]["auroc"]))
    control_note_lines = [
        "- 这些对照只用于验证早段 checkpoint 信号强弱，不直接作为最终十槽 submission。",
    ]
    if float(global_best_control["position"]) != 0.10 or float(noncoding_best_control["position"]) != 0.10:
        control_note_lines.extend([
            f"- 离线 control 结果并不支持“10% 单点最优”：全域最好是 `{_pct_label(global_best_control['position'])}`，非 coding 最好是 `{_pct_label(noncoding_best_control['position'])}`。",
            "- 因此本轮最终胜出并不是靠“纯 10% 单 checkpoint”，而是靠 prefix-safe 训练 + 非 coding 专用 anchor4 + coding 回退到 v1。",
        ])
    lines: list[str] = [
        "# EARLYSTOP PREFIX10 / ANCHOR4 SVD ROUND1 RESULTS (2026-04-08)",
        "",
        "## 开头确认",
        "",
    ]
    for line in REPO_STATUS_LINES:
        lines.append(f"- {line}")
    lines.extend([
        "",
        "## 1. 我确认的当前 repo 状态",
        "",
        "- 已读 `docs/README.md`、`docs/WORK_SUMMARY_0408_06.md`、`nad/ops/earlystop_svd.py`、`scripts/export_earlystop_svd_submission.py`。",
        "- `scripts/export_earlystop_svd_submission.py` 已可直接加载 `models/ml_selectors/earlystop_svd_lowrank_lr_v1.pkl` 导出 Early-Stop submission。",
        "- 当前 repo 内可直接复跑的 Early-Stop 导出线至少包括：`tok_conf_prefix_mean_v1`、`earlystop_svd_lowrank_lr_v1`、`earlystop_from_bestofn_svm_bridge_v1`。",
        "- 从现有反馈文本看，repo 内已导出线里 `earlystop_svd_lowrank_lr_v1` 强于 `earlystop_from_bestofn_svm_bridge_v1`；`benchmark_early_stop_v1` 只在反馈文本中出现，不是当前仓库内可直接复跑的实现。",
        "",
        "## 2. prefix-10 特征定义",
        "",
        "### 保留特征",
        "",
        "- `token_only`: `tok_conf_*`、`tok_gini_*`、`tok_neg_entropy_*`、`tok_selfcert_*`、`tok_logprob_*` 与对应 token availability flags。",
        "- `token_plus_traj`: 上述 token 特征 + `traj_continuity`、`traj_reflection_count`、`traj_novelty`、`traj_max_reflection`、`traj_late_convergence` + `has_rows_bank`。",
        "- `all`: `token_plus_traj` + `nc_mean`、`nc_slope` + 全部 availability flags。",
        "",
        "### 删除特征",
        "",
        "- `self_similarity`：现有实现按全序列前半/后半计算，会把 50%/100% 后验信息泄露进前缀视角；本轮明确删除。",
        "",
        "## 3. 5/10/15/20 checkpoint 对照",
        "",
    ])

    lines.extend(_render_control_table("全域统一单 checkpoint", summary["controls"]["global"]))
    lines.extend(_render_control_table("非 coding 单 checkpoint", summary["controls"]["noncoding"]))
    lines.extend(control_note_lines)
    lines.extend(["", "## 4. 与 `earlystop_svd_lowrank_lr_v1` 的对比", ""])

    compare_rows = [
        summary["baselines"]["tok_conf_prefix_mean_v1"],
        summary["baselines"]["earlystop_from_bestofn_svm_bridge_v1"],
        baseline_v1,
        summary["candidates"]["global_anchor4"],
        summary["candidates"]["noncoding_anchor4_coding_v1"],
    ]
    lines.extend(_render_method_table("整体对比", compare_rows))
    lines.extend(_render_cache_table(best_new))
    lines.extend([
        "### Anchor 映射",
        "",
        "- 正式 bundle 采用四个锚点模型：`10/40/70/100`。",
        "- 十槽映射：`10/20/30→10`，`40/50/60→40`，`70/80/90→70`，`100→100`。",
        "- `global_anchor4`：math/science/coding 全域统一训练。",
        "- `noncoding_anchor4_coding_v1`：math+science 统一训练，coding 直接复用 `earlystop_svd_lowrank_lr_v1` 原生十槽路由。",
        "",
        "## 5. 是否导出新 submission",
        "",
        f"- 结论：`{'YES' if summary['decision']['export_recommended'] else 'NO'}`。",
        f"- 判定理由：{summary['decision']['reason']}。",
        "",
        "## 6. 如果没有胜出，失败原因是什么",
        "",
        f"- {'不适用：本轮已严格胜出 `earlystop_svd_lowrank_lr_v1` 并完成导出。' if summary['decision']['export_recommended'] else summary['decision']['reason'] + '.'}",
        "",
        "## 7. 改了哪些文件",
        "",
    ])
    for file_path in changed_files:
        lines.append(f"- `{file_path}`")
    lines.extend([
        "",
        "## 8. 如何复跑",
        "",
        "```bash",
        "bash cookbook/00_setup/verify.sh",
        "python3 -m nad.cli --help",
        "python3 scripts/run_earlystop_prefix10_svd_round1.py \\",
        "  --cache-root MUI_HUB/cache \\",
        "  --test-cache-root /home/jovyan/public-ro/MUI_HUB/cache_test",
        "```",
    ])
    if summary["decision"]["export_recommended"]:
        lines.extend([
            "",
            "```bash",
            "python3 scripts/export_earlystop_svd_submission.py \\",
            "  --cache-root /home/jovyan/public-ro/MUI_HUB/cache_test \\",
            "  --model-path models/ml_selectors/earlystop_prefix10_svd_round1.pkl \\",
            "  --method-name earlystop_prefix10_svd_round1 \\",
            "  --filename earlystop_prefix10_svd_round1.json",
            "```",
        ])
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_export_if_needed(
    *,
    should_export: bool,
    model_path: Path,
    cache_root: str,
) -> Optional[Path]:
    if not should_export:
        return None
    out_path = REPO_ROOT / "submission" / "EarlyStop" / "earlystop_prefix10_svd_round1.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "export_earlystop_svd_submission.py"),
            "--cache-root",
            str(cache_root),
            "--model-path",
            str(model_path),
            "--method-name",
            "earlystop_prefix10_svd_round1",
            "--filename",
            out_path.name,
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    return out_path


def collect_required_features_for_eval(
    *,
    v1_bundle: dict[str, Any],
    bridge_bundle: dict[str, Any],
) -> set[str]:
    required = set(PREFIX_SAFE_FEATURE_FAMILY_MAP["all"])

    for bundle in (v1_bundle,):
        for domain_bundle in bundle["domains"].values():
            for route in domain_bundle["routes"]:
                if route["route_type"] == "baseline":
                    required.add(str(route["signal_name"]))
                else:
                    required.update(str(name) for name in route["feature_names"])

    for domain_bundle in bridge_bundle["domains"].values():
        route = domain_bundle["route"]
        if route["route_type"] == "baseline":
            required.add(str(route["signal_name"]))
        else:
            required.update(str(name) for name in route["feature_names"])

    return required


def main() -> None:
    ap = argparse.ArgumentParser(description="Run EarlyStop prefix10/anchor4 SVD round1 experiments")
    ap.add_argument("--cache-root", default="MUI_HUB/cache", help="Labeled training/eval cache root")
    ap.add_argument("--test-cache-root", default="/home/jovyan/public-ro/MUI_HUB/cache_test", help="Blind cache root for optional export")
    ap.add_argument("--out-model", default="models/ml_selectors/earlystop_prefix10_svd_round1.pkl")
    ap.add_argument("--out-summary", default="results/scans/earlystop/earlystop_prefix10_svd_round1_summary.json")
    ap.add_argument("--out-eval", default="results/scans/earlystop/earlystop_prefix10_svd_round1_eval.json")
    ap.add_argument("--out-doc", default="docs/EARLYSTOP_PREFIX10_SVD_ROUND1_RESULTS_20260408.md")
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4, help="Parallel workers for independent position searches")
    ap.add_argument("--feature-chunk-problems", type=int, default=DEFAULT_FEATURE_CHUNK_PROBLEMS, help="Problems per feature-extraction chunk")
    ap.add_argument("--max-problems-per-cache", type=int, default=0, help="0 means all problems")
    ap.add_argument("--no-export", action="store_true", help="Skip export even if new candidate wins")
    args = ap.parse_args()

    train_cache_root = args.cache_root
    if not Path(train_cache_root).is_absolute():
        train_cache_root = str((REPO_ROOT / train_cache_root).resolve())

    max_problems_per_cache = None if int(args.max_problems_per_cache) <= 0 else int(args.max_problems_per_cache)

    positions_needed = tuple(sorted(set(CONTROL_POSITIONS + ANCHOR_POSITIONS)))
    v1_bundle = load_earlystop_svd_bundle(REPO_ROOT / "models/ml_selectors/earlystop_svd_lowrank_lr_v1.pkl")
    bridge_bundle = load_earlystop_svm_bundle(REPO_ROOT / "models/ml_selectors/bestofn_svm_bridge_v1.pkl")
    required_features = collect_required_features_for_eval(
        v1_bundle=v1_bundle,
        bridge_bundle=bridge_bundle,
    )

    print(f"Building feature store for positions={','.join(_pct_label(p) for p in EXTRACTION_POSITIONS)}")
    feature_store = build_feature_store(
        cache_root=train_cache_root,
        positions=EXTRACTION_POSITIONS,
        required_feature_names=required_features,
        max_problems_per_cache=max_problems_per_cache,
        max_workers=int(args.workers),
        chunk_problems=int(args.feature_chunk_problems),
    )
    scope_tables = build_scope_training_tables_from_feature_store(
        feature_store=feature_store,
        positions=positions_needed,
    )

    global_anchor_indices = tuple(positions_needed.index(p) for p in ANCHOR_POSITIONS)
    global_control_indices = tuple(positions_needed.index(p) for p in CONTROL_POSITIONS)

    global_anchor_routes = train_routes_for_scope(
        tables=[scope_tables["global"][i] for i in global_anchor_indices],
        positions=ANCHOR_POSITIONS,
        scope_name="global",
        n_splits=int(args.n_splits),
        random_state=int(args.random_state),
        max_workers=int(args.workers),
    )
    noncoding_anchor_routes = train_routes_for_scope(
        tables=[scope_tables["noncoding"][i] for i in global_anchor_indices],
        positions=ANCHOR_POSITIONS,
        scope_name="noncoding",
        n_splits=int(args.n_splits),
        random_state=int(args.random_state),
        max_workers=int(args.workers),
    )
    global_control_routes = train_routes_for_scope(
        tables=[scope_tables["global"][i] for i in global_control_indices],
        positions=CONTROL_POSITIONS,
        scope_name="global",
        n_splits=int(args.n_splits),
        random_state=int(args.random_state),
        max_workers=int(args.workers),
    )
    noncoding_control_routes = train_routes_for_scope(
        tables=[scope_tables["noncoding"][i] for i in global_control_indices],
        positions=CONTROL_POSITIONS,
        scope_name="noncoding",
        n_splits=int(args.n_splits),
        random_state=int(args.random_state),
        max_workers=int(args.workers),
    )

    global_bundle = build_anchor_bundle(
        bundle_version="earlystop_prefix10_svd_round1_global_anchor4",
        anchor_routes=global_anchor_routes,
    )
    noncoding_bundle = build_anchor_bundle(
        bundle_version="earlystop_prefix10_svd_round1_noncoding_anchor4_coding_v1",
        anchor_routes=noncoding_anchor_routes,
        coding_routes=v1_bundle["domains"]["coding"]["routes"],
    )

    print("[main] training finished, starting full-method evaluation")

    baseline_results = {
        "tok_conf_prefix_mean_v1": evaluate_method_from_feature_store(
            method_name="tok_conf_prefix_mean_v1",
            feature_store=feature_store,
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_tok_conf_score_fn(),
        ),
        "earlystop_from_bestofn_svm_bridge_v1": evaluate_method_from_feature_store(
            method_name="earlystop_from_bestofn_svm_bridge_v1",
            feature_store=feature_store,
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_bridge_score_fn(bridge_bundle),
        ),
        "earlystop_svd_lowrank_lr_v1": evaluate_method_from_feature_store(
            method_name="earlystop_svd_lowrank_lr_v1",
            feature_store=feature_store,
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_svd_bundle_score_fn(v1_bundle),
        ),
    }
    print("[main] baselines evaluated, starting candidate evaluation")

    candidate_results = {
        "global_anchor4": evaluate_method_from_feature_store(
            method_name="global_anchor4",
            feature_store=feature_store,
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_svd_bundle_score_fn(global_bundle),
        ),
        "noncoding_anchor4_coding_v1": evaluate_method_from_feature_store(
            method_name="noncoding_anchor4_coding_v1",
            feature_store=feature_store,
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_svd_bundle_score_fn(noncoding_bundle),
        ),
    }
    print("[main] candidates evaluated, starting checkpoint controls")

    global_controls = [
        {
            "position": float(position),
            **evaluate_single_position_route_from_feature_store(
                feature_store=feature_store,
                position=float(position),
                route_resolver=lambda _domain, p=float(position): global_control_routes[p],
            ),
        }
        for position in CONTROL_POSITIONS
    ]
    print("[main] global controls evaluated")
    noncoding_controls = [
        {
            "position": float(position),
            **evaluate_single_position_route_from_feature_store(
                feature_store=[payload for payload in feature_store if payload["domain"] in {"math", "science"}],
                position=float(position),
                route_resolver=lambda _domain, p=float(position): noncoding_control_routes[p],
            ),
        }
        for position in CONTROL_POSITIONS
    ]
    print("[main] noncoding controls evaluated")

    new_methods = list(candidate_results.values())
    best_new = choose_best_candidate(new_methods)
    decision = decide_export(best_new, baseline_results["earlystop_svd_lowrank_lr_v1"])

    best_bundle = global_bundle if best_new["method_name"] == "global_anchor4" else noncoding_bundle
    out_model = REPO_ROOT / args.out_model
    save_earlystop_svd_bundle(best_bundle, out_model)

    route_summary = {
        "global_anchor4": {
            _pct_label(position): summarise_route(route)
            for position, route in global_anchor_routes.items()
        },
        "noncoding_anchor4_coding_v1": {
            "noncoding_anchor_routes": {
                _pct_label(position): summarise_route(route)
                for position, route in noncoding_anchor_routes.items()
            },
            "coding_routes_source": "earlystop_svd_lowrank_lr_v1",
        },
    }

    summary = {
        "created_at_utc": _now_utc(),
        "repo_status": REPO_STATUS_LINES,
        "cache_root": str(train_cache_root),
        "config": {
            "n_splits": int(args.n_splits),
            "random_state": int(args.random_state),
            "workers": int(args.workers),
            "feature_chunk_problems": int(args.feature_chunk_problems),
            "max_problems_per_cache": 0 if max_problems_per_cache is None else int(max_problems_per_cache),
            "control_positions": list(CONTROL_POSITIONS),
            "anchor_positions": list(ANCHOR_POSITIONS),
            "official_slot_to_anchor": {str(k): float(v) for k, v in OFFICIAL_SLOT_TO_ANCHOR.items()},
            "families": list(SEARCH_FAMILIES),
            "representations": list(SEARCH_REPRESENTATIONS),
            "svd_dims": list(SEARCH_RANKS),
            "c_values": list(SEARCH_C_VALUES),
            "class_weight": list(SEARCH_CLASS_WEIGHT),
            "whiten": [bool(v) for v in SEARCH_WHITEN],
        },
        "feature_definition": {
            "kept_features": list(PREFIX_SAFE_FEATURE_FAMILY_MAP["all"]),
            "dropped_features": ["self_similarity"],
            "family_map": PREFIX_SAFE_FEATURE_FAMILY_MAP,
        },
        "controls": {
            "global": global_controls,
            "noncoding": noncoding_controls,
        },
        "route_summary": route_summary,
        "baselines": baseline_results,
        "candidates": candidate_results,
        "decision": decision,
        "best_new_bundle_version": str(best_bundle["bundle_version"]),
        "best_new_saved_model": _display_path(out_model),
    }

    out_summary = REPO_ROOT / args.out_summary
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    out_eval = REPO_ROOT / args.out_eval
    out_eval.parent.mkdir(parents=True, exist_ok=True)
    out_eval.write_text(json.dumps({
        "baselines": baseline_results,
        "candidates": candidate_results,
        "controls": {
            "global": global_controls,
            "noncoding": noncoding_controls,
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    export_path = run_export_if_needed(
        should_export=(not args.no_export) and bool(decision["export_recommended"]),
        model_path=out_model,
        cache_root=str(args.test_cache_root),
    )
    summary["export_path"] = None if export_path is None else _display_path(export_path)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    changed_files = [
        "docs/EARLYSTOP_PREFIX10_SVD_ROUND1_PLAN_20260408.md",
        "docs/EARLYSTOP_PREFIX10_SVD_ROUND1_RESULTS_20260408.md",
        "scripts/run_earlystop_prefix10_svd_round1.py",
    ]
    if export_path is not None:
        changed_files.append(_display_path(export_path))
    write_results_doc(
        path=REPO_ROOT / args.out_doc,
        summary=summary,
        changed_files=changed_files,
    )

    print("\nRound1 complete")
    print(f"  Saved best new bundle : {out_model}")
    print(f"  Summary JSON          : {out_summary}")
    print(f"  Eval JSON             : {out_eval}")
    print(f"  Results doc           : {REPO_ROOT / args.out_doc}")
    if export_path is not None:
        print(f"  Exported submission   : {export_path}")
    else:
        print("  Exported submission   : skipped")


if __name__ == "__main__":
    main()
