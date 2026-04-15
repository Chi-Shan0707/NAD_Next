#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env, "1")

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nad.ops.earlystop_svd import (
    AVAILABILITY_FEATURES,
    LEGACY_FULL_FEATURE_NAMES,
    TOKEN_FEATURES,
    TRAJ_FEATURES,
    _build_representation,
    _rank_transform_matrix,
)
from nad.ops.grouped_ranking import evaluate_grouped_scores


DEFAULT_SOURCE_FEATURE_STORE = "results/cache/es_svd_ms_rr_r1/cache_all_547b9060debe139e.pkl"
DEFAULT_TARGET_FEATURE_STORE = (
    "results/cache/export_earlystop_svd_submission_strongfeat_20260410/"
    "feature_store_all_ref030_18a73b5e30f1a00d.pkl"
)
DEFAULT_SOURCE_CACHE_KEY = "cache/DS-R1/lcb_v5"
DEFAULT_TARGET_CACHE_KEY = "Qwen3-4B/lcb_v5"
DEFAULT_EXTRA_TARGET_CACHE_KEY = "none"
DEFAULT_OUT_CSV = "results/tables/coding_rotation_adapter.csv"
DEFAULT_OUT_DOC = "docs/CODING_ROTATION_ADAPTER.md"

ANCHOR_PCT_TO_POS = {
    10: 0.10,
    20: 0.20,
    30: 0.30,
    40: 0.40,
    50: 0.50,
    60: 0.60,
    70: 0.70,
    80: 0.80,
    90: 0.90,
    100: 1.00,
}

FEATURE_TO_INDEX = {str(name): idx for idx, name in enumerate(LEGACY_FULL_FEATURE_NAMES)}
BUNDLE_TO_FEATURES = {
    "token_only": tuple(str(name) for name in TOKEN_FEATURES),
    "canonical_22": tuple(str(name) for name in list(TOKEN_FEATURES) + list(TRAJ_FEATURES) + list(AVAILABILITY_FEATURES)),
}
DEFAULT_BUNDLES = ("token_only", "canonical_22")
DEFAULT_ANCHORS = (70, 100)
DEFAULT_RANKS = (4, 8, 16)
DEFAULT_POINTWISE_C = (0.10, 0.50, 1.00)
DEFAULT_PAIRWISE_C = (0.10, 0.50, 1.00)
DEFAULT_SEEDS = (42,)
DEFAULT_CLASS_WEIGHTS = ("none", "balanced")
DEFAULT_ALIGN_WEIGHTS = (0.05, 0.10, 0.25)
DEFAULT_IDENTITY_WEIGHTS = (0.10, 0.30, 1.00)


@dataclass(frozen=True)
class HeadSpec:
    head_kind: str
    rank: int
    c_value: float
    class_weight: str


@dataclass
class LinearHead:
    head_kind: str
    rank: int
    scaler: StandardScaler
    svd: TruncatedSVD
    weight: np.ndarray
    intercept: float
    c_value: float
    class_weight: str
    source_val_metrics: dict[str, Any]


@dataclass(frozen=True)
class RotationSpec:
    align_weight: float
    identity_weight: float
    lr: float
    steps: int
    eval_every: int


@dataclass
class RotationResult:
    method: str
    head_kind: str
    rank: int
    c_value: float
    class_weight: str
    rotation: np.ndarray
    rotation_norm: float
    orth_error: float
    source_val_metrics: dict[str, Any]
    target_proxy_metrics: dict[str, Any]
    alignment_gap: float
    fit_details: dict[str, Any]


def _parse_csv_str(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(raw).split(",") if item.strip())


def _parse_csv_int(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in str(raw).split(",") if item.strip())


def _parse_csv_float(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in str(raw).split(",") if item.strip())


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_feature_store(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    feature_store = payload["feature_store"] if isinstance(payload, dict) and "feature_store" in payload else payload
    return list(feature_store)


def _find_payload(feature_store: list[dict[str, Any]], cache_key: str) -> dict[str, Any]:
    for payload in feature_store:
        if str(payload["cache_key"]) == str(cache_key):
            return payload
    available = ", ".join(sorted(str(item["cache_key"]) for item in feature_store))
    raise KeyError(f"cache_key={cache_key!r} not found; available={available}")


def _problem_groups_from_keys(group_keys: np.ndarray) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for idx, key in enumerate(np.asarray(group_keys, dtype=object).tolist()):
        out.setdefault(str(key), []).append(int(idx))
    return out


def _anchor_index(payload: dict[str, Any], anchor_pct: int) -> int:
    pos = ANCHOR_PCT_TO_POS[int(anchor_pct)]
    positions = [float(v) for v in payload["positions"]]
    for idx, value in enumerate(positions):
        if abs(float(value) - float(pos)) < 1e-8:
            return int(idx)
    raise KeyError(f"anchor_pct={anchor_pct} not found in positions={positions}")


def _rank_per_problem(x_raw: np.ndarray, problem_offsets: list[int]) -> np.ndarray:
    x = np.asarray(x_raw, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64)
    offsets = [int(v) for v in problem_offsets]
    for start, end in zip(offsets[:-1], offsets[1:]):
        if end - start <= 0:
            continue
        out[start:end] = _rank_transform_matrix(x[start:end])
    return out


def _build_payload_matrix(
    payload: dict[str, Any],
    *,
    anchor_pct: int,
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    pos_idx = _anchor_index(payload, anchor_pct)
    x_raw_full = np.asarray(payload["tensor"][:, pos_idx, : len(LEGACY_FULL_FEATURE_NAMES)], dtype=np.float64)
    x_rank_full = _rank_per_problem(x_raw_full, [int(v) for v in payload["problem_offsets"]])
    feature_indices = [FEATURE_TO_INDEX[name] for name in feature_names]
    x_rep = _build_representation(x_raw_full, x_rank_full, feature_indices, "raw+rank")
    labels = np.asarray(payload["labels"], dtype=np.int32).reshape(-1)
    sample_ids = np.asarray(payload["sample_ids"], dtype=np.int64).reshape(-1)
    group_keys = np.asarray(payload["group_keys"], dtype=object).reshape(-1)
    return {
        "x_raw": x_raw_full,
        "x_rank": x_rank_full,
        "x": x_rep,
        "labels": labels,
        "sample_ids": sample_ids,
        "group_keys": group_keys,
        "problem_ids": [str(v) for v in payload["problem_ids"]],
        "problem_offsets": [int(v) for v in payload["problem_offsets"]],
        "problem_groups": _problem_groups_from_keys(group_keys),
    }


def _split_problem_ids(problem_ids: list[str], val_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    if not problem_ids:
        return set(), set()
    rng = np.random.RandomState(int(seed))
    ordered = [str(pid) for pid in problem_ids]
    perm = rng.permutation(len(ordered))
    n_val = max(1, int(round(len(ordered) * float(val_fraction))))
    n_val = min(max(1, n_val), max(1, len(ordered) - 1))
    val_ids = {ordered[int(idx)] for idx in perm[:n_val].tolist()}
    train_ids = {pid for pid in ordered if pid not in val_ids}
    return train_ids, val_ids


def _row_mask_for_problem_ids(group_keys: np.ndarray, problem_ids: set[str]) -> np.ndarray:
    keys = np.asarray(group_keys, dtype=object).reshape(-1)
    mask = np.zeros(keys.shape[0], dtype=bool)
    for idx, key in enumerate(keys.tolist()):
        problem_id = str(key).split("::", 1)[-1]
        if problem_id in problem_ids:
            mask[idx] = True
    return mask


def _subset_matrix(matrix: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    x = np.asarray(matrix["x"], dtype=np.float64)[mask]
    labels = np.asarray(matrix["labels"], dtype=np.int32)[mask]
    sample_ids = np.asarray(matrix["sample_ids"], dtype=np.int64)[mask]
    group_keys = np.asarray(matrix["group_keys"], dtype=object)[mask]
    return {
        "x": x,
        "labels": labels,
        "sample_ids": sample_ids,
        "group_keys": group_keys,
        "problem_groups": _problem_groups_from_keys(group_keys),
    }


def _z_transform(x: np.ndarray, scaler: StandardScaler, svd: TruncatedSVD) -> np.ndarray:
    return np.asarray(svd.transform(scaler.transform(np.asarray(x, dtype=np.float64))), dtype=np.float64)


def _fit_svd_basis(x_train: np.ndarray, rank: int, seed: int) -> tuple[StandardScaler, TruncatedSVD]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    x_scaled = scaler.fit_transform(np.asarray(x_train, dtype=np.float64))
    max_rank = max(1, min(int(rank), int(x_scaled.shape[1]), int(x_scaled.shape[0] - 1)))
    svd = TruncatedSVD(n_components=max_rank, random_state=int(seed))
    svd.fit(x_scaled)
    return scaler, svd


def _metric_key(value: Any) -> float:
    if value is None:
        return float("-inf")
    value_f = float(value)
    return value_f if np.isfinite(value_f) else float("-inf")


def _score_linear(z: np.ndarray, weight: np.ndarray, intercept: float) -> np.ndarray:
    return np.asarray(np.asarray(z, dtype=np.float64) @ np.asarray(weight, dtype=np.float64) + float(intercept), dtype=np.float64)


def _build_pairwise_diffs(z: np.ndarray, y: np.ndarray, group_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_arr = np.asarray(z, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int32).reshape(-1)
    g_arr = np.asarray(group_keys, dtype=object).reshape(-1)
    n_feat = int(z_arr.shape[1])
    pair_x_parts: list[np.ndarray] = []
    pair_y_parts: list[np.ndarray] = []
    by_group = _problem_groups_from_keys(g_arr)
    for indices in by_group.values():
        idx = np.asarray(indices, dtype=np.int64)
        y_g = y_arr[idx]
        pos = idx[y_g > 0]
        neg = idx[y_g <= 0]
        if pos.size <= 0 or neg.size <= 0:
            continue
        diffs_pos = (z_arr[pos][:, None, :] - z_arr[neg][None, :, :]).reshape(-1, n_feat)
        diffs_neg = -diffs_pos
        pair_x_parts.append(np.concatenate([diffs_pos, diffs_neg], axis=0))
        pair_y_parts.append(
            np.concatenate([
                np.ones(diffs_pos.shape[0], dtype=np.int32),
                np.zeros(diffs_neg.shape[0], dtype=np.int32),
            ])
        )
    if not pair_x_parts:
        return np.zeros((0, n_feat), dtype=np.float64), np.zeros((0,), dtype=np.int32)
    return (
        np.concatenate(pair_x_parts, axis=0).astype(np.float64, copy=False),
        np.concatenate(pair_y_parts, axis=0).astype(np.int32, copy=False),
    )


def _fit_pointwise_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    groups_val: np.ndarray,
    sample_ids_val: np.ndarray,
    rank: int,
    c_value: float,
    class_weight: str,
    seed: int,
) -> Optional[LinearHead]:
    if np.unique(np.asarray(y_train, dtype=np.int32)).shape[0] < 2:
        return None
    scaler, svd = _fit_svd_basis(x_train, rank=rank, seed=seed)
    z_train = _z_transform(x_train, scaler, svd)
    z_val = _z_transform(x_val, scaler, svd)
    clf = LogisticRegression(
        C=float(c_value),
        class_weight=None if class_weight == "none" else "balanced",
        max_iter=2000,
        random_state=int(seed),
    )
    clf.fit(z_train, np.asarray(y_train, dtype=np.int32))
    scores_val = _score_linear(z_val, clf.coef_[0], float(clf.intercept_[0]))
    metrics_val = evaluate_grouped_scores(
        _problem_groups_from_keys(groups_val),
        np.asarray(y_val, dtype=np.int32),
        scores_val,
        sample_ids=np.asarray(sample_ids_val, dtype=np.int64),
    )
    return LinearHead(
        head_kind="pointwise",
        rank=int(z_train.shape[1]),
        scaler=scaler,
        svd=svd,
        weight=np.asarray(clf.coef_[0], dtype=np.float64),
        intercept=float(clf.intercept_[0]),
        c_value=float(c_value),
        class_weight=str(class_weight),
        source_val_metrics=metrics_val,
    )


def _fit_pairwise_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    groups_val: np.ndarray,
    sample_ids_val: np.ndarray,
    rank: int,
    c_value: float,
    seed: int,
) -> Optional[LinearHead]:
    if np.unique(np.asarray(y_train, dtype=np.int32)).shape[0] < 2:
        return None
    scaler, svd = _fit_svd_basis(x_train, rank=rank, seed=seed)
    z_train = _z_transform(x_train, scaler, svd)
    z_val = _z_transform(x_val, scaler, svd)
    x_pairs, y_pairs = _build_pairwise_diffs(z_train, y_train, groups_train)
    if x_pairs.shape[0] <= 0 or np.unique(y_pairs).shape[0] < 2:
        return None
    clf = LogisticRegression(
        C=float(c_value),
        class_weight=None,
        max_iter=2000,
        random_state=int(seed),
    )
    clf.fit(x_pairs, y_pairs)
    scores_val = _score_linear(z_val, clf.coef_[0], float(clf.intercept_[0]))
    metrics_val = evaluate_grouped_scores(
        _problem_groups_from_keys(groups_val),
        np.asarray(y_val, dtype=np.int32),
        scores_val,
        sample_ids=np.asarray(sample_ids_val, dtype=np.int64),
    )
    return LinearHead(
        head_kind="pairwise",
        rank=int(z_train.shape[1]),
        scaler=scaler,
        svd=svd,
        weight=np.asarray(clf.coef_[0], dtype=np.float64),
        intercept=float(clf.intercept_[0]),
        c_value=float(c_value),
        class_weight="none",
        source_val_metrics=metrics_val,
    )


def _choose_best_head(candidates: list[LinearHead]) -> LinearHead:
    if not candidates:
        raise ValueError("No valid head candidates found")
    return max(
        candidates,
        key=lambda head: (
            _metric_key(head.source_val_metrics.get("pairwise_auc")),
            _metric_key(head.source_val_metrics.get("top1_accuracy")),
            _metric_key(head.source_val_metrics.get("pooled_pointwise_auroc")),
            -int(head.rank),
        ),
    )


def _covariance_torch(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape={tuple(x.shape)}")
    if x.shape[0] <= 1:
        return torch.zeros((x.shape[1], x.shape[1]), dtype=x.dtype, device=x.device)
    centered = x - x.mean(dim=0, keepdim=True)
    return centered.T @ centered / float(x.shape[0])


def _alignment_gap_numpy(z_source: np.ndarray, z_target_rot: np.ndarray) -> float:
    src = np.asarray(z_source, dtype=np.float64)
    tgt = np.asarray(z_target_rot, dtype=np.float64)
    if src.shape[1] != tgt.shape[1]:
        raise ValueError(f"latent dim mismatch: {src.shape} vs {tgt.shape}")
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    src_cov = np.cov(src, rowvar=False, bias=True)
    tgt_cov = np.cov(tgt, rowvar=False, bias=True)
    mean_term = float(np.mean((src_mean - tgt_mean) ** 2))
    cov_term = float(np.mean((src_cov - tgt_cov) ** 2))
    return mean_term + cov_term


def _pair_indices_for_source(labels: np.ndarray, group_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    g = np.asarray(group_keys, dtype=object).reshape(-1)
    pos_rows: list[int] = []
    neg_rows: list[int] = []
    for indices in _problem_groups_from_keys(g).values():
        idx = np.asarray(indices, dtype=np.int64)
        y_g = y[idx]
        pos = idx[y_g > 0]
        neg = idx[y_g <= 0]
        if pos.size <= 0 or neg.size <= 0:
            continue
        pos_grid = np.repeat(pos, neg.size)
        neg_grid = np.tile(neg, pos.size)
        pos_rows.extend(pos_grid.tolist())
        neg_rows.extend(neg_grid.tolist())
    return np.asarray(pos_rows, dtype=np.int64), np.asarray(neg_rows, dtype=np.int64)


def _to_problem_top1_map(scores: np.ndarray, group_keys: np.ndarray, sample_ids: np.ndarray) -> dict[str, int]:
    scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    groups_arr = np.asarray(group_keys, dtype=object).reshape(-1)
    sample_arr = np.asarray(sample_ids, dtype=np.int64).reshape(-1)
    out: dict[str, int] = {}
    by_group = _problem_groups_from_keys(groups_arr)
    for group, indices in by_group.items():
        idx = np.asarray(indices, dtype=np.int64)
        local_scores = scores_arr[idx]
        local_samples = sample_arr[idx]
        order = sorted(range(idx.size), key=lambda j: (-float(local_scores[j]), int(local_samples[j])))
        out[str(group)] = int(local_samples[int(order[0])]) if order else int(local_samples[0])
    return out


def _target_proxy_metrics(
    scores_base: np.ndarray,
    scores_new: np.ndarray,
    group_keys: np.ndarray,
    sample_ids: np.ndarray,
    alignment_gap_before: float,
    alignment_gap_after: float,
) -> dict[str, Any]:
    base = np.asarray(scores_base, dtype=np.float64).reshape(-1)
    new = np.asarray(scores_new, dtype=np.float64).reshape(-1)
    corr = float(np.corrcoef(base, new)[0, 1]) if base.size > 1 else float("nan")
    top1_base = _to_problem_top1_map(base, group_keys, sample_ids)
    top1_new = _to_problem_top1_map(new, group_keys, sample_ids)
    common = sorted(set(top1_base.keys()) & set(top1_new.keys()))
    top1_agreement = (
        float(np.mean([int(top1_base[key] == top1_new[key]) for key in common])) if common else float("nan")
    )
    return {
        "target_label_status": "unlabeled_proxy_only",
        "target_score_mean": float(np.mean(new)) if new.size else float("nan"),
        "target_score_std": float(np.std(new)) if new.size else float("nan"),
        "target_score_corr_vs_no_rotation": corr,
        "target_top1_agreement_vs_no_rotation": top1_agreement,
        "target_top1_flip_rate_vs_no_rotation": (
            float(1.0 - top1_agreement) if np.isfinite(top1_agreement) else float("nan")
        ),
        "target_mean_abs_score_delta_vs_no_rotation": float(np.mean(np.abs(new - base))) if new.size else float("nan"),
        "alignment_gap_before": float(alignment_gap_before),
        "alignment_gap_after": float(alignment_gap_after),
        "alignment_gap_delta": float(alignment_gap_after - alignment_gap_before),
    }


def _rotation_matrix_torch(raw_param: torch.Tensor) -> torch.Tensor:
    skew = raw_param - raw_param.T
    eye = torch.eye(raw_param.shape[0], dtype=raw_param.dtype, device=raw_param.device)
    return torch.linalg.solve(eye - skew, eye + skew)


def _rotation_metrics(rotation: np.ndarray) -> tuple[float, float]:
    rot = np.asarray(rotation, dtype=np.float64)
    eye = np.eye(rot.shape[0], dtype=np.float64)
    return (
        float(np.linalg.norm(rot - eye, ord="fro")),
        float(np.linalg.norm(rot.T @ rot - eye, ord="fro")),
    )


def _evaluate_rotated_scores(
    *,
    head: LinearHead,
    rotation: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    groups_val: np.ndarray,
    sample_ids_val: np.ndarray,
    x_target: np.ndarray,
    groups_target: np.ndarray,
    sample_ids_target: np.ndarray,
    z_source_train: np.ndarray,
    target_base_scores: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    z_val = _z_transform(x_val, head.scaler, head.svd) @ rotation.T
    z_target = _z_transform(x_target, head.scaler, head.svd)
    z_target_rot = z_target @ rotation.T
    val_scores = _score_linear(z_val, head.weight, head.intercept)
    target_scores = _score_linear(z_target_rot, head.weight, head.intercept)
    source_metrics = evaluate_grouped_scores(
        _problem_groups_from_keys(groups_val),
        np.asarray(y_val, dtype=np.int32),
        val_scores,
        sample_ids=np.asarray(sample_ids_val, dtype=np.int64),
    )
    alignment_before = _alignment_gap_numpy(z_source_train, z_target)
    alignment_after = _alignment_gap_numpy(z_source_train, z_target_rot)
    target_metrics = _target_proxy_metrics(
        target_base_scores,
        target_scores,
        groups_target,
        sample_ids_target,
        alignment_before,
        alignment_after,
    )
    return source_metrics, target_metrics, alignment_after


def _fit_learned_rotation(
    *,
    head: LinearHead,
    x_source_train: np.ndarray,
    y_source_train: np.ndarray,
    groups_source_train: np.ndarray,
    x_source_val: np.ndarray,
    y_source_val: np.ndarray,
    groups_source_val: np.ndarray,
    sample_ids_source_val: np.ndarray,
    x_target_pool: np.ndarray,
    groups_target: np.ndarray,
    sample_ids_target: np.ndarray,
    target_base_scores: np.ndarray,
    spec: RotationSpec,
    seed: int,
    torch_threads: int,
) -> RotationResult:
    torch.set_num_threads(max(1, int(torch_threads)))

    z_source_train = _z_transform(x_source_train, head.scaler, head.svd)
    z_source_val = _z_transform(x_source_val, head.scaler, head.svd)
    z_target_pool = _z_transform(x_target_pool, head.scaler, head.svd)

    z_src_train_t = torch.as_tensor(z_source_train, dtype=torch.float64)
    y_src_train_t = torch.as_tensor(np.asarray(y_source_train, dtype=np.float64), dtype=torch.float64)
    z_tgt_t = torch.as_tensor(z_target_pool, dtype=torch.float64)
    weight_t = torch.as_tensor(np.asarray(head.weight, dtype=np.float64), dtype=torch.float64)
    intercept_t = torch.tensor(float(head.intercept), dtype=torch.float64)

    src_mean_t = z_src_train_t.mean(dim=0)
    src_cov_t = _covariance_torch(z_src_train_t)

    raw_param = torch.zeros((head.rank, head.rank), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([raw_param], lr=float(spec.lr))

    pair_pos_idx: Optional[torch.Tensor] = None
    pair_neg_idx: Optional[torch.Tensor] = None
    if head.head_kind == "pairwise":
        pos_idx, neg_idx = _pair_indices_for_source(y_source_train, groups_source_train)
        pair_pos_idx = torch.as_tensor(pos_idx, dtype=torch.long)
        pair_neg_idx = torch.as_tensor(neg_idx, dtype=torch.long)

    best_rotation = np.eye(head.rank, dtype=np.float64)
    best_source_metrics, best_target_metrics, best_alignment = _evaluate_rotated_scores(
        head=head,
        rotation=best_rotation,
        x_val=x_source_val,
        y_val=y_source_val,
        groups_val=groups_source_val,
        sample_ids_val=sample_ids_source_val,
        x_target=x_target_pool,
        groups_target=groups_target,
        sample_ids_target=sample_ids_target,
        z_source_train=z_source_train,
        target_base_scores=target_base_scores,
    )
    best_key = (
        _metric_key(best_source_metrics.get("pairwise_auc")),
        _metric_key(best_source_metrics.get("top1_accuracy")),
        _metric_key(best_source_metrics.get("pooled_pointwise_auroc")),
        -float(best_alignment),
    )
    best_step = 0

    for step in range(1, int(spec.steps) + 1):
        rotation_t = _rotation_matrix_torch(raw_param)
        z_src_rot = z_src_train_t @ rotation_t.T

        if head.head_kind == "pointwise":
            logits = z_src_rot @ weight_t + intercept_t
            sup_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_src_train_t)
        else:
            if pair_pos_idx is None or pair_neg_idx is None or pair_pos_idx.numel() <= 0:
                sup_loss = torch.tensor(0.0, dtype=torch.float64)
            else:
                pair_scores = z_src_rot @ weight_t
                margins = pair_scores[pair_pos_idx] - pair_scores[pair_neg_idx]
                sup_loss = torch.nn.functional.softplus(-margins).mean()

        z_tgt_rot = z_tgt_t @ rotation_t.T
        mean_loss = torch.mean((z_tgt_rot.mean(dim=0) - src_mean_t) ** 2)
        cov_loss = torch.mean((_covariance_torch(z_tgt_rot) - src_cov_t) ** 2)
        align_loss = mean_loss + cov_loss

        eye_t = torch.eye(head.rank, dtype=torch.float64)
        identity_loss = torch.mean((rotation_t - eye_t) ** 2)
        orth_loss = torch.mean((rotation_t.T @ rotation_t - eye_t) ** 2)
        total_loss = sup_loss + float(spec.align_weight) * align_loss + float(spec.identity_weight) * identity_loss + 0.10 * orth_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % int(spec.eval_every) != 0 and step != int(spec.steps):
            continue

        rotation_np = rotation_t.detach().cpu().numpy().astype(np.float64, copy=False)
        source_metrics, target_metrics, alignment_after = _evaluate_rotated_scores(
            head=head,
            rotation=rotation_np,
            x_val=x_source_val,
            y_val=y_source_val,
            groups_val=groups_source_val,
            sample_ids_val=sample_ids_source_val,
            x_target=x_target_pool,
            groups_target=groups_target,
            sample_ids_target=sample_ids_target,
            z_source_train=z_source_train,
            target_base_scores=target_base_scores,
        )
        key = (
            _metric_key(source_metrics.get("pairwise_auc")),
            _metric_key(source_metrics.get("top1_accuracy")),
            _metric_key(source_metrics.get("pooled_pointwise_auroc")),
            -float(alignment_after),
        )
        if key > best_key:
            best_rotation = rotation_np.copy()
            best_source_metrics = source_metrics
            best_target_metrics = target_metrics
            best_alignment = float(alignment_after)
            best_key = key
            best_step = int(step)

    rotation_norm, orth_error = _rotation_metrics(best_rotation)
    return RotationResult(
        method="learned_rotation",
        head_kind=head.head_kind,
        rank=int(head.rank),
        c_value=float(head.c_value),
        class_weight=str(head.class_weight),
        rotation=best_rotation,
        rotation_norm=rotation_norm,
        orth_error=orth_error,
        source_val_metrics=best_source_metrics,
        target_proxy_metrics=best_target_metrics,
        alignment_gap=float(best_alignment),
        fit_details={
            "align_weight": float(spec.align_weight),
            "identity_weight": float(spec.identity_weight),
            "lr": float(spec.lr),
            "steps": int(spec.steps),
            "eval_every": int(spec.eval_every),
            "best_step": int(best_step),
        },
    )


def _sample_random_matched_rotation(dim: int, target_norm: float, seed: int) -> np.ndarray:
    eye = np.eye(dim, dtype=np.float64)
    if target_norm <= 1e-10:
        return eye
    rng = np.random.RandomState(int(seed))
    raw = rng.normal(size=(dim, dim))
    skew = raw - raw.T
    skew_norm = np.linalg.norm(skew, ord="fro")
    if skew_norm <= 1e-12:
        return eye
    skew = skew / skew_norm

    def _rotation(scale: float) -> np.ndarray:
        a = float(scale) * skew
        return np.linalg.solve(eye - a, eye + a)

    lo, hi = 0.0, 1.0
    while np.linalg.norm(_rotation(hi) - eye, ord="fro") < target_norm and hi < 1e6:
        hi *= 2.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        value = np.linalg.norm(_rotation(mid) - eye, ord="fro")
        if value < target_norm:
            lo = mid
        else:
            hi = mid
    return _rotation(hi)


def _row_from_result(
    *,
    seed: int,
    bundle_name: str,
    anchor_pct: int,
    target_pool: str,
    result: RotationResult,
) -> dict[str, Any]:
    row = {
        "seed": int(seed),
        "bundle": str(bundle_name),
        "anchor_pct": int(anchor_pct),
        "target_pool": str(target_pool),
        "method": str(result.method),
        "head": str(result.head_kind),
        "rank": int(result.rank),
        "c_value": float(result.c_value),
        "class_weight": str(result.class_weight),
        "source_val_pairwise": result.source_val_metrics.get("pairwise_auc"),
        "source_val_hit1": result.source_val_metrics.get("top1_accuracy"),
        "source_val_auroc": result.source_val_metrics.get("pooled_pointwise_auroc"),
        "source_val_selacc10": result.source_val_metrics.get("local_selacc@10%"),
        "rotation_norm": float(result.rotation_norm),
        "orth_error": float(result.orth_error),
        "alignment_gap": float(result.alignment_gap),
        "target_label_status": result.target_proxy_metrics.get("target_label_status"),
        "target_score_mean": result.target_proxy_metrics.get("target_score_mean"),
        "target_score_std": result.target_proxy_metrics.get("target_score_std"),
        "target_score_corr_vs_no_rotation": result.target_proxy_metrics.get("target_score_corr_vs_no_rotation"),
        "target_top1_agreement_vs_no_rotation": result.target_proxy_metrics.get("target_top1_agreement_vs_no_rotation"),
        "target_top1_flip_rate_vs_no_rotation": result.target_proxy_metrics.get("target_top1_flip_rate_vs_no_rotation"),
        "target_mean_abs_score_delta_vs_no_rotation": result.target_proxy_metrics.get("target_mean_abs_score_delta_vs_no_rotation"),
        "alignment_gap_before": result.target_proxy_metrics.get("alignment_gap_before"),
        "alignment_gap_after": result.target_proxy_metrics.get("alignment_gap_after"),
        "alignment_gap_delta": result.target_proxy_metrics.get("alignment_gap_delta"),
        "qwen_pairwise": float("nan"),
        "qwen_hit1": float("nan"),
        "qwen_auroc": float("nan"),
    }
    for key, value in sorted(result.fit_details.items()):
        row[f"fit__{key}"] = value
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value_f = float(value)
    except Exception:
        return None
    return value_f if np.isfinite(value_f) else None


def _fmt(value: Any, digits: int = 4) -> str:
    value_f = _safe_float(value)
    if value_f is None:
        return "N/A"
    return f"{value_f:.{digits}f}"


def _fmt_pct(value: Any) -> str:
    value_f = _safe_float(value)
    if value_f is None:
        return "N/A"
    return f"{value_f * 100.0:.2f}%"


def _top_row(rows: list[dict[str, Any]], *, head_kind: str, bundle_name: str, anchor_pct: int, method: str) -> Optional[dict[str, Any]]:
    subset = [
        row for row in rows
        if str(row["head"]) == str(head_kind)
        and str(row["bundle"]) == str(bundle_name)
        and int(row["anchor_pct"]) == int(anchor_pct)
        and str(row["method"]) == str(method)
    ]
    if not subset:
        return None
    return max(
        subset,
        key=lambda row: (
            _metric_key(row.get("source_val_pairwise")),
            _metric_key(row.get("source_val_hit1")),
            _metric_key(row.get("source_val_auroc")),
        ),
    )


def _geometry_proxy_label(row: Optional[dict[str, Any]]) -> str:
    if row is None:
        return "N/A"
    corr = _safe_float(row.get("target_score_corr_vs_no_rotation"))
    agree = _safe_float(row.get("target_top1_agreement_vs_no_rotation"))
    if corr is None or agree is None:
        return "unavailable"
    if corr >= 0.99 and agree >= 0.95:
        return "mostly calibration-like"
    if corr <= 0.97 or agree <= 0.85:
        return "geometry-shift-like"
    return "mixed / weakly geometric"


def _write_doc(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    out_csv_path = (REPO_ROOT / str(args.out_csv)).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CODING ROTATION ADAPTER",
        "",
        "## Summary",
        "",
        "- Goal: test whether the `DS-R1 -> Qwen3-4B` coding gap on `lcb_v5` is better explained by a small latent rotation than by a full basis mismatch.",
        f"- Source labeled cache: `{args.source_cache_key}` from `{args.source_feature_store}`.",
        f"- Target unlabeled cache: `{args.target_cache_key}` from `{args.target_feature_store}`.",
        f"- Extra unlabeled target: `{args.extra_target_cache_key}`.",
        f"- Bundles: `{', '.join(args.bundles)}`.",
        f"- Anchors: `{', '.join(str(v) for v in args.anchors)}`.",
        f"- Seeds: `{', '.join(str(v) for v in args.seeds)}`.",
        "",
        "## Important Limitation",
        "",
        "- The local repo does **not** expose per-sample correctness labels for `cache_test` `lcb_v5`.",
        "- Because of that, the generated CSV reports **source-holdout metrics** plus **unlabeled Qwen proxy metrics** (alignment gap, score correlation, top-1 flip rate), not true Qwen `hit@1` / `pairwise` / `AUROC`.",
        "- The `qwen_*` columns are intentionally left as `NaN` until an external blind evaluation or labeled target artifact is available.",
        "",
        "## Best Rows by Cell",
        "",
        "| Bundle | Anchor | Head | Best learned source pairwise | Best learned source Hit@1 | Rotation norm | Target corr vs no-rot | Target top1 agreement | Geometry proxy |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for bundle_name in args.bundles:
        for anchor_pct in args.anchors:
            for head_kind in ("pointwise", "pairwise"):
                row = _top_row(rows, head_kind=head_kind, bundle_name=bundle_name, anchor_pct=anchor_pct, method="learned_rotation")
                if row is None:
                    continue
                lines.append(
                    "| {bundle} | {anchor} | {head} | {pairwise} | {hit1} | {norm} | {corr} | {agree} | {label} |".format(
                        bundle=bundle_name,
                        anchor=anchor_pct,
                        head=head_kind,
                        pairwise=_fmt(row.get("source_val_pairwise")),
                        hit1=_fmt(row.get("source_val_hit1")),
                        norm=_fmt(row.get("rotation_norm")),
                        corr=_fmt(row.get("target_score_corr_vs_no_rotation")),
                        agree=_fmt(row.get("target_top1_agreement_vs_no_rotation")),
                        label=_geometry_proxy_label(row),
                    )
                )
    lines.extend([
        "",
        "## Readout",
        "",
        "- **Q1 small rotation recover target ranking quality?** Offline answer is unresolved here because local target labels are unavailable; the current script only establishes whether a near-identity rotation changes Qwen rankings materially and whether it improves unlabeled alignment.",
        "- **Q2 calibration vs geometry shift?** Use the `target_score_corr_vs_no_rotation` and `target_top1_agreement_vs_no_rotation` columns as the offline proxy. High correlation/high agreement suggests mostly calibration-like behavior; lower agreement suggests a more geometric shift.",
        "- **Q3 pointwise vs pairwise after rotation?** Compare `source_val_pairwise` and `source_val_hit1` between `head=pointwise` and `head=pairwise` within the same `bundle × anchor` rows.",
        "- **Q4 does `token_only` remain friendliest?** Compare the best `token_only` and `canonical_22` learned rows. In this offline version, friendliness means retaining source-holdout ranking while requiring only a small rotation and producing a lower target alignment gap.",
        "",
        "## Files",
        "",
        f"- CSV: `{_display_path(out_csv_path)}`",
        f"- Script: `scripts/coding_ssl/train_coding_rotation_adapter.py`",
        "",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a near-identity orthogonal coding rotation adapter")
    ap.add_argument("--source-feature-store", default=DEFAULT_SOURCE_FEATURE_STORE)
    ap.add_argument("--target-feature-store", default=DEFAULT_TARGET_FEATURE_STORE)
    ap.add_argument("--source-cache-key", default=DEFAULT_SOURCE_CACHE_KEY)
    ap.add_argument("--target-cache-key", default=DEFAULT_TARGET_CACHE_KEY)
    ap.add_argument("--extra-target-cache-key", default=DEFAULT_EXTRA_TARGET_CACHE_KEY)
    ap.add_argument("--bundles", type=_parse_csv_str, default=DEFAULT_BUNDLES)
    ap.add_argument("--anchors", type=_parse_csv_int, default=DEFAULT_ANCHORS)
    ap.add_argument("--ranks", type=_parse_csv_int, default=DEFAULT_RANKS)
    ap.add_argument("--pointwise-c", type=_parse_csv_float, default=DEFAULT_POINTWISE_C)
    ap.add_argument("--pairwise-c", type=_parse_csv_float, default=DEFAULT_PAIRWISE_C)
    ap.add_argument("--class-weights", type=_parse_csv_str, default=DEFAULT_CLASS_WEIGHTS)
    ap.add_argument("--align-weights", type=_parse_csv_float, default=DEFAULT_ALIGN_WEIGHTS)
    ap.add_argument("--identity-weights", type=_parse_csv_float, default=DEFAULT_IDENTITY_WEIGHTS)
    ap.add_argument("--seeds", type=_parse_csv_int, default=DEFAULT_SEEDS)
    ap.add_argument("--val-problem-fraction", type=float, default=0.30)
    ap.add_argument("--rotation-lr", type=float, default=0.05)
    ap.add_argument("--rotation-steps", type=int, default=200)
    ap.add_argument("--rotation-eval-every", type=int, default=20)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    ap.add_argument("--out-doc", default=DEFAULT_OUT_DOC)
    args = ap.parse_args()

    source_feature_store_path = (REPO_ROOT / str(args.source_feature_store)).resolve()
    target_feature_store_path = (REPO_ROOT / str(args.target_feature_store)).resolve()
    out_csv = (REPO_ROOT / str(args.out_csv)).resolve()
    out_doc = (REPO_ROOT / str(args.out_doc)).resolve()

    print(f"[load] source feature store={_display_path(source_feature_store_path)}", flush=True)
    source_store = _load_feature_store(source_feature_store_path)
    print(f"[load] target feature store={_display_path(target_feature_store_path)}", flush=True)
    target_store = _load_feature_store(target_feature_store_path)

    source_payload = _find_payload(source_store, args.source_cache_key)
    target_payload = _find_payload(target_store, args.target_cache_key)
    extra_target_payload = None if str(args.extra_target_cache_key).lower() in {"", "none", "off"} else _find_payload(target_store, args.extra_target_cache_key)

    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        train_problem_ids, val_problem_ids = _split_problem_ids(source_payload["problem_ids"], args.val_problem_fraction, seed)
        print(
            f"[split] seed={seed} train_problems={len(train_problem_ids)} val_problems={len(val_problem_ids)}",
            flush=True,
        )
        for bundle_name in args.bundles:
            if bundle_name not in BUNDLE_TO_FEATURES:
                raise KeyError(f"Unknown bundle={bundle_name!r}; expected one of {sorted(BUNDLE_TO_FEATURES.keys())}")
            feature_names = BUNDLE_TO_FEATURES[bundle_name]
            for anchor_pct in args.anchors:
                print(
                    f"[cell] seed={seed} bundle={bundle_name} anchor={anchor_pct}%",
                    flush=True,
                )
                source_matrix = _build_payload_matrix(source_payload, anchor_pct=anchor_pct, feature_names=feature_names)
                target_matrix = _build_payload_matrix(target_payload, anchor_pct=anchor_pct, feature_names=feature_names)
                if extra_target_payload is not None:
                    extra_target_matrix = _build_payload_matrix(extra_target_payload, anchor_pct=anchor_pct, feature_names=feature_names)
                    x_target_pool = np.concatenate([target_matrix["x"], extra_target_matrix["x"]], axis=0)
                    groups_target_pool = np.concatenate([target_matrix["group_keys"], extra_target_matrix["group_keys"]], axis=0)
                    sample_ids_target_pool = np.concatenate([target_matrix["sample_ids"], extra_target_matrix["sample_ids"]], axis=0)
                    target_pool_name = f"{args.target_cache_key}+{args.extra_target_cache_key}"
                else:
                    x_target_pool = np.asarray(target_matrix["x"], dtype=np.float64)
                    groups_target_pool = np.asarray(target_matrix["group_keys"], dtype=object)
                    sample_ids_target_pool = np.asarray(target_matrix["sample_ids"], dtype=np.int64)
                    target_pool_name = str(args.target_cache_key)

                train_mask = _row_mask_for_problem_ids(source_matrix["group_keys"], train_problem_ids)
                val_mask = _row_mask_for_problem_ids(source_matrix["group_keys"], val_problem_ids)
                source_train = _subset_matrix(source_matrix, train_mask)
                source_val = _subset_matrix(source_matrix, val_mask)

                for head_kind in ("pointwise", "pairwise"):
                    candidates: list[LinearHead] = []
                    for rank in args.ranks:
                        if head_kind == "pointwise":
                            for c_value in args.pointwise_c:
                                for class_weight in args.class_weights:
                                    head = _fit_pointwise_head(
                                        source_train["x"],
                                        source_train["labels"],
                                        source_val["x"],
                                        source_val["labels"],
                                        source_val["group_keys"],
                                        source_val["sample_ids"],
                                        rank=rank,
                                        c_value=c_value,
                                        class_weight=class_weight,
                                        seed=seed,
                                    )
                                    if head is not None:
                                        candidates.append(head)
                        else:
                            for c_value in args.pairwise_c:
                                head = _fit_pairwise_head(
                                    source_train["x"],
                                    source_train["labels"],
                                    source_train["group_keys"],
                                    source_val["x"],
                                    source_val["labels"],
                                    source_val["group_keys"],
                                    source_val["sample_ids"],
                                    rank=rank,
                                    c_value=c_value,
                                    seed=seed,
                                )
                                if head is not None:
                                    candidates.append(head)
                    head = _choose_best_head(candidates)
                    print(
                        f"  [head] kind={head_kind} rank={head.rank} C={head.c_value:.2f} cw={head.class_weight} "
                        f"pairwise={_fmt(head.source_val_metrics.get('pairwise_auc'))} "
                        f"hit1={_fmt(head.source_val_metrics.get('top1_accuracy'))}",
                        flush=True,
                    )

                    z_target = _z_transform(target_matrix["x"], head.scaler, head.svd)
                    target_base_scores = _score_linear(z_target, head.weight, head.intercept)
                    no_rotation = np.eye(head.rank, dtype=np.float64)
                    no_source_metrics, no_target_metrics, no_alignment = _evaluate_rotated_scores(
                        head=head,
                        rotation=no_rotation,
                        x_val=source_val["x"],
                        y_val=source_val["labels"],
                        groups_val=source_val["group_keys"],
                        sample_ids_val=source_val["sample_ids"],
                        x_target=target_matrix["x"],
                        groups_target=target_matrix["group_keys"],
                        sample_ids_target=target_matrix["sample_ids"],
                        z_source_train=_z_transform(source_train["x"], head.scaler, head.svd),
                        target_base_scores=target_base_scores,
                    )
                    no_result = RotationResult(
                        method="no_rotation",
                        head_kind=head.head_kind,
                        rank=head.rank,
                        c_value=head.c_value,
                        class_weight=head.class_weight,
                        rotation=no_rotation,
                        rotation_norm=0.0,
                        orth_error=0.0,
                        source_val_metrics=no_source_metrics,
                        target_proxy_metrics=no_target_metrics,
                        alignment_gap=float(no_alignment),
                        fit_details={},
                    )

                    best_learned: Optional[RotationResult] = None
                    for align_weight in args.align_weights:
                        for identity_weight in args.identity_weights:
                            spec = RotationSpec(
                                align_weight=float(align_weight),
                                identity_weight=float(identity_weight),
                                lr=float(args.rotation_lr),
                                steps=int(args.rotation_steps),
                                eval_every=int(args.rotation_eval_every),
                            )
                            result = _fit_learned_rotation(
                                head=head,
                                x_source_train=source_train["x"],
                                y_source_train=source_train["labels"],
                                groups_source_train=source_train["group_keys"],
                                x_source_val=source_val["x"],
                                y_source_val=source_val["labels"],
                                groups_source_val=source_val["group_keys"],
                                sample_ids_source_val=source_val["sample_ids"],
                                x_target_pool=x_target_pool,
                                groups_target=target_matrix["group_keys"],
                                sample_ids_target=target_matrix["sample_ids"],
                                target_base_scores=target_base_scores,
                                spec=spec,
                                seed=seed,
                                torch_threads=int(args.torch_threads),
                            )
                            if best_learned is None:
                                best_learned = result
                                continue
                            lhs = (
                                _metric_key(result.source_val_metrics.get("pairwise_auc")),
                                _metric_key(result.source_val_metrics.get("top1_accuracy")),
                                _metric_key(result.source_val_metrics.get("pooled_pointwise_auroc")),
                                -float(result.alignment_gap),
                            )
                            rhs = (
                                _metric_key(best_learned.source_val_metrics.get("pairwise_auc")),
                                _metric_key(best_learned.source_val_metrics.get("top1_accuracy")),
                                _metric_key(best_learned.source_val_metrics.get("pooled_pointwise_auroc")),
                                -float(best_learned.alignment_gap),
                            )
                            if lhs > rhs:
                                best_learned = result

                    if best_learned is None:
                        raise RuntimeError("Failed to fit any learned rotation candidate")

                    random_rotation = _sample_random_matched_rotation(
                        head.rank,
                        best_learned.rotation_norm,
                        seed=seed + 1000 + int(anchor_pct),
                    )
                    random_source_metrics, random_target_metrics, random_alignment = _evaluate_rotated_scores(
                        head=head,
                        rotation=random_rotation,
                        x_val=source_val["x"],
                        y_val=source_val["labels"],
                        groups_val=source_val["group_keys"],
                        sample_ids_val=source_val["sample_ids"],
                        x_target=target_matrix["x"],
                        groups_target=target_matrix["group_keys"],
                        sample_ids_target=target_matrix["sample_ids"],
                        z_source_train=_z_transform(source_train["x"], head.scaler, head.svd),
                        target_base_scores=target_base_scores,
                    )
                    random_rotation_norm, random_orth_error = _rotation_metrics(random_rotation)
                    random_result = RotationResult(
                        method="random_rotation_control",
                        head_kind=head.head_kind,
                        rank=head.rank,
                        c_value=head.c_value,
                        class_weight=head.class_weight,
                        rotation=random_rotation,
                        rotation_norm=random_rotation_norm,
                        orth_error=random_orth_error,
                        source_val_metrics=random_source_metrics,
                        target_proxy_metrics=random_target_metrics,
                        alignment_gap=float(random_alignment),
                        fit_details={"matched_to_learned_norm": float(best_learned.rotation_norm)},
                    )

                    for result in (no_result, random_result, best_learned):
                        rows.append(
                            _row_from_result(
                                seed=seed,
                                bundle_name=bundle_name,
                                anchor_pct=anchor_pct,
                                target_pool=target_pool_name,
                                result=result,
                            )
                        )

                    print(
                        f"  [best learned] head={head_kind} pairwise={_fmt(best_learned.source_val_metrics.get('pairwise_auc'))} "
                        f"hit1={_fmt(best_learned.source_val_metrics.get('top1_accuracy'))} "
                        f"rot_norm={_fmt(best_learned.rotation_norm)} "
                        f"target_corr={_fmt(best_learned.target_proxy_metrics.get('target_score_corr_vs_no_rotation'))}",
                        flush=True,
                    )

    _write_csv(out_csv, rows)
    _write_doc(out_doc, rows=rows, args=args)
    print(f"[done] wrote csv={_display_path(out_csv)}", flush=True)
    print(f"[done] wrote doc={_display_path(out_doc)}", flush=True)


if __name__ == "__main__":
    main()
