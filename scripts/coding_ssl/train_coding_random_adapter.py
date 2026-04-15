#!/usr/bin/env python3
"""Coding-only transductive random low-rank adapter experiment.

This script tests whether a tiny trainable low-rank residual adapter around a
frozen coding SVD basis improves DS -> Qwen transfer on livecodebench_v5 while
preserving exact linear interpretability.

Protocol summary
----------------
- Source labeled:      DeepSeek `lcb_v5` from `MUI_HUB/cache`
- Target unlabeled #1: DeepSeek `lcb_v5` from `MUI_HUB/cache_test`
- Target unlabeled #2: Qwen `lcb_v5` from `MUI_HUB/cache_test`
- Anchors:             70% and 100%, trained as a shared late-anchor model
- Feature families:    canonical_22 (= token_plus_traj_fixed), token_only
- Representation:      raw+rank only

Baselines / conditions
----------------------
- frozen basis + pointwise head
- frozen basis + pairwise head
- frozen basis + random frozen adapter control (+ pointwise / pairwise head)
- frozen basis + trained low-rank adapter + pointwise head
- frozen basis + trained low-rank adapter + pairwise head

Only the trained low-rank adapter receives the target-unlabeled transductive
adaptation phase. Frozen baselines are retained as inductive references.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env, "1")

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
    from torch import nn
    from torch.nn import functional as F

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - fail loudly in main()
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

from nad.ops.earlystop import _problem_sort_key, discover_cache_entries
from nad.ops.earlystop_svd import FULL_FEATURE_NAMES, _fit_svd_transform
from nad.ops.neural_semisup_svdomain import RunMatrix, build_run_matrix_from_feature_store
from scripts.run_earlystop_prefix10_svd_round1 import (
    DEFAULT_FEATURE_CHUNK_PROBLEMS,
    FEATURE_TO_INDEX,
    PREFIX_SAFE_FEATURE_FAMILY_MAP,
    build_feature_store,
)
from SVDomain.train_es_svd_ms_rr_r1 import (
    FIXED_FEATURE_NAMES,
    _build_holdout_problem_map,
    _split_feature_store,
)


DEFAULT_SOURCE_MAIN_ROOT = REPO_ROOT / "MUI_HUB" / "cache"
DEFAULT_TARGET_TEST_ROOT = REPO_ROOT / "MUI_HUB" / "cache_test"
DEFAULT_FEATURE_CACHE_DIR = REPO_ROOT / "results" / "cache" / "coding_random_adapter"
DEFAULT_OUT_CSV = REPO_ROOT / "results" / "tables" / "coding_random_adapter.csv"
DEFAULT_OUT_DOC = REPO_ROOT / "docs" / "CODING_RANDOM_ADAPTER.md"

SOURCE_CACHE_KEY = "DS-R1/lcb_v5"
TARGET_DS_CACHE_KEY = "DS-R1/lcb_v5"
TARGET_QWEN_CACHE_KEY = "Qwen3-4B/lcb_v5"

LATE_ANCHORS = (0.70, 1.00)
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_ADAPTER_RANKS = (2, 4, 8)
DEFAULT_ALPHAS = (0.10, 0.30, 1.00)

DEFAULT_BASIS_RANK = 12
DEFAULT_BASIS_WHITEN = False
DEFAULT_BASIS_RANDOM_STATE = 42

DEFAULT_SOURCE_EPOCHS = 60
DEFAULT_TRANSDUCTIVE_EPOCHS = 35
DEFAULT_LEARNING_RATE = 2e-2
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_ADAPTER_INIT_STD = 0.03

DEFAULT_PSEUDO_PAIR_MARGIN = 0.50
DEFAULT_PSEUDO_PAIR_MAX_PER_GROUP = 256
DEFAULT_PSEUDO_PAIR_WEIGHT = 0.35
DEFAULT_ANCHOR_CONSISTENCY_WEIGHT = 0.20
DEFAULT_COVARIANCE_ALIGNMENT_WEIGHT = 0.10
DEFAULT_PSEUDO_START_EPOCH = 5

SMOKE_SOURCE_EPOCHS = 4
SMOKE_TRANSDUCTIVE_EPOCHS = 3
SMOKE_MAX_PROBLEMS = 4
SMOKE_ADAPTER_RANKS = (2,)
SMOKE_ALPHAS = (0.30,)
SMOKE_SEEDS = (42,)


def require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for scripts/coding_ssl/train_coding_random_adapter.py."
        )


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stable_hash(text: str) -> int:
    digest = hashlib.sha1(str(text).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _parse_int_csv(text: str) -> tuple[int, ...]:
    values = [int(chunk.strip()) for chunk in str(text).split(",") if chunk.strip()]
    if not values:
        raise ValueError(f"Expected at least one integer, got: {text!r}")
    return tuple(values)


def _parse_float_csv(text: str) -> tuple[float, ...]:
    values = [float(chunk.strip()) for chunk in str(text).split(",") if chunk.strip()]
    if not values:
        raise ValueError(f"Expected at least one float, got: {text!r}")
    return tuple(values)


def _qualify_feature_store(feature_store: list[dict[str, Any]], source_name: str) -> list[dict[str, Any]]:
    qualified: list[dict[str, Any]] = []
    for payload in feature_store:
        item = dict(payload)
        item["source_name"] = str(source_name)
        item["base_cache_key"] = str(payload["cache_key"])
        item["cache_key"] = f"{source_name}/{payload['cache_key']}"
        qualified.append(item)
    return qualified


def _feature_cache_path(
    *,
    cache_dir: Path,
    source_name: str,
    base_root: str | Path,
    include_cache_keys: Iterable[str],
    positions: tuple[float, ...],
    required_feature_names: set[str],
    max_problems_per_cache: Optional[int],
) -> Path:
    payload = {
        "version": 1,
        "source_name": str(source_name),
        "base_root": str(base_root),
        "include_cache_keys": sorted(str(v) for v in include_cache_keys),
        "positions": [float(v) for v in positions],
        "required_feature_names": sorted(str(v) for v in required_feature_names),
        "max_problems_per_cache": None if max_problems_per_cache is None else int(max_problems_per_cache),
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    suffix = "all" if max_problems_per_cache is None else f"cap{int(max_problems_per_cache)}"
    return cache_dir / f"{source_name}_{suffix}_{digest}.pkl"


def _load_or_build_feature_store(
    *,
    source_name: str,
    base_root: str | Path,
    include_cache_keys: Iterable[str],
    positions: tuple[float, ...],
    required_feature_names: set[str],
    feature_cache_dir: Path,
    refresh_feature_cache: bool,
    max_problems_per_cache: Optional[int],
    feature_workers: int,
    chunk_problems: int,
) -> tuple[list[dict[str, Any]], Path, str]:
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _feature_cache_path(
        cache_dir=feature_cache_dir,
        source_name=source_name,
        base_root=base_root,
        include_cache_keys=include_cache_keys,
        positions=positions,
        required_feature_names=required_feature_names,
        max_problems_per_cache=max_problems_per_cache,
    )
    if cache_path.exists() and not refresh_feature_cache:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        return list(payload["feature_store"]), cache_path, "loaded"

    store = build_feature_store(
        cache_root=base_root,
        positions=positions,
        required_feature_names=required_feature_names,
        include_cache_keys=set(str(v) for v in include_cache_keys),
        max_problems_per_cache=max_problems_per_cache,
        max_workers=max(1, int(feature_workers)),
        chunk_problems=max(1, int(chunk_problems)),
    )
    qualified = _qualify_feature_store(store, source_name=source_name)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(
            {
                "source_name": str(source_name),
                "base_root": str(base_root),
                "include_cache_keys": sorted(str(v) for v in include_cache_keys),
                "positions": [float(v) for v in positions],
                "required_feature_names": sorted(str(v) for v in required_feature_names),
                "max_problems_per_cache": None if max_problems_per_cache is None else int(max_problems_per_cache),
                "feature_store": qualified,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    tmp_path.replace(cache_path)
    return qualified, cache_path, "built"


def _resolve_cache_root(base_root: str | Path, cache_key: str) -> Path:
    for entry in discover_cache_entries(base_root):
        if str(entry.cache_key) == str(cache_key):
            return Path(entry.cache_root)
    raise FileNotFoundError(f"Unable to resolve cache_key={cache_key} under base_root={base_root}")


def _has_ground_truth(cache_root: Path) -> bool:
    root = Path(cache_root)
    return (root / "evaluation_report_compact.json").exists() or (root / "evaluation_report.json").exists()


def _select_single_coding_payload(feature_store: list[dict[str, Any]], *, split_name: str) -> list[dict[str, Any]]:
    coding_payloads = [payload for payload in feature_store if str(payload.get("domain")) == "coding"]
    if not coding_payloads:
        raise ValueError(f"No coding payload found in {split_name}")
    if len(coding_payloads) > 1:
        keys = ", ".join(sorted(str(payload["cache_key"]) for payload in coding_payloads))
        raise ValueError(f"Expected exactly one coding payload in {split_name}, found {len(coding_payloads)}: {keys}")
    return coding_payloads


def _family_feature_names(family_name: str) -> tuple[str, ...]:
    family = str(family_name)
    if family == "canonical_22":
        return tuple(FIXED_FEATURE_NAMES)
    if family == "token_only":
        return tuple(PREFIX_SAFE_FEATURE_FAMILY_MAP["token_only"])
    raise ValueError(f"Unknown family_name: {family_name}")


def _family_feature_indices(family_name: str) -> tuple[int, ...]:
    return tuple(int(FEATURE_TO_INDEX[name]) for name in _family_feature_names(family_name))


def _build_family_matrix(
    feature_store: list[dict[str, Any]],
    *,
    family_name: str,
    anchors: tuple[float, ...],
) -> RunMatrix:
    return build_run_matrix_from_feature_store(
        feature_store,
        anchor_positions=anchors,
        feature_indices=_family_feature_indices(family_name),
    )


def _rep_tensor(matrix: RunMatrix) -> np.ndarray:
    return np.concatenate(
        [np.asarray(matrix.raw, dtype=np.float64), np.asarray(matrix.rank, dtype=np.float64)],
        axis=2,
    ).astype(np.float64, copy=False)


def _flatten_rep_tensor(rep: np.ndarray) -> np.ndarray:
    rep_arr = np.asarray(rep, dtype=np.float64)
    if rep_arr.ndim != 3:
        raise ValueError(f"Expected 3D rep tensor, got shape={rep_arr.shape}")
    return rep_arr.reshape(rep_arr.shape[0] * rep_arr.shape[1], rep_arr.shape[2]).astype(np.float64, copy=False)


def _fit_basis_from_matrix(
    matrix: RunMatrix,
    *,
    basis_rank: int,
    basis_whiten: bool,
    basis_random_state: int,
) -> dict[str, Any]:
    rep_flat = _flatten_rep_tensor(_rep_tensor(matrix))
    transform = _fit_svd_transform(
        rep_flat,
        rank=int(basis_rank),
        whiten=bool(basis_whiten),
        random_state=int(basis_random_state),
    )
    if transform is None:
        raise ValueError(
            f"Unable to fit frozen SVD basis on matrix shape={rep_flat.shape} "
            f"rank={basis_rank} whiten={basis_whiten}"
        )
    return transform


def _transform_matrix_to_latent(matrix: RunMatrix, basis: dict[str, Any]) -> np.ndarray:
    rep = _rep_tensor(matrix)
    flat = _flatten_rep_tensor(rep)
    scaled = basis["scaler"].transform(flat)
    z = basis["svd"].transform(scaled)
    if bool(basis.get("whiten", False)):
        singular_values = np.asarray(basis["svd"].singular_values_, dtype=np.float64)
        singular_values = np.where(np.abs(singular_values) < 1e-8, 1.0, singular_values)
        z = z / singular_values[: z.shape[1]]
    n_runs, n_anchors, _ = rep.shape
    return np.asarray(z, dtype=np.float64).reshape(n_runs, n_anchors, -1)


def _flatten_latent_with_metadata(
    z_tensor: np.ndarray,
    matrix: RunMatrix,
) -> dict[str, np.ndarray]:
    z_arr = np.asarray(z_tensor, dtype=np.float64)
    if z_arr.ndim != 3:
        raise ValueError(f"Expected 3D latent tensor, got shape={z_arr.shape}")
    if z_arr.shape[0] != matrix.labels.shape[0]:
        raise ValueError("z_tensor run count must match RunMatrix row count")
    n_runs, n_anchors, latent_dim = z_arr.shape
    flat_z = z_arr.reshape(n_runs * n_anchors, latent_dim).astype(np.float64, copy=False)
    y = np.asarray(
        [int(matrix.labels[i]) for i in range(n_runs) for _ in range(n_anchors)],
        dtype=np.int32,
    )
    anchor_values = np.asarray(
        [float(matrix.anchor_positions[a_idx]) for _ in range(n_runs) for a_idx in range(n_anchors)],
        dtype=np.float64,
    )
    anchor_pct = np.asarray([int(round(100.0 * float(v))) for v in anchor_values.tolist()], dtype=np.int32)
    pair_groups = np.asarray(
        [
            f"{matrix.rank_groups[i]}@@{anchor_pct[a_row]}"
            for i in range(n_runs)
            for a_row in range(i * n_anchors, (i + 1) * n_anchors)
        ],
        dtype=object,
    )
    eval_groups = np.asarray(
        [str(matrix.rank_groups[i]) for i in range(n_runs) for _ in range(n_anchors)],
        dtype=object,
    )
    sample_keys = np.asarray(
        [
            f"{matrix.cache_keys[i]}::{int(matrix.sample_ids[i])}"
            for i in range(n_runs)
            for _ in range(n_anchors)
        ],
        dtype=object,
    )
    run_ids = np.asarray(
        [int(matrix.sample_ids[i]) for i in range(n_runs) for _ in range(n_anchors)],
        dtype=np.int32,
    )
    problem_keys = np.asarray(
        [f"{matrix.cache_keys[i]}::{matrix.problem_ids[i]}" for i in range(n_runs) for _ in range(n_anchors)],
        dtype=object,
    )
    return {
        "z": flat_z,
        "y": y,
        "pair_groups": pair_groups,
        "eval_groups": eval_groups,
        "anchor_pct": anchor_pct,
        "sample_keys": sample_keys,
        "run_ids": run_ids,
        "problem_keys": problem_keys,
    }


def _build_pair_indices(
    labels: np.ndarray,
    group_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    groups = np.asarray(group_keys, dtype=object).reshape(-1)
    if y.shape[0] != groups.shape[0]:
        raise ValueError("labels and group_keys must have the same length")
    pos_parts: list[np.ndarray] = []
    neg_parts: list[np.ndarray] = []
    by_group: dict[str, list[int]] = {}
    for idx, group_key in enumerate(groups.tolist()):
        by_group.setdefault(str(group_key), []).append(int(idx))
    for idxs in by_group.values():
        idx_arr = np.asarray(idxs, dtype=np.int64)
        y_g = y[idx_arr]
        pos_idx = idx_arr[np.where(y_g > 0)[0]]
        neg_idx = idx_arr[np.where(y_g <= 0)[0]]
        if pos_idx.size == 0 or neg_idx.size == 0:
            continue
        pos_parts.append(np.repeat(pos_idx, neg_idx.size))
        neg_parts.append(np.tile(neg_idx, pos_idx.size))
    if not pos_parts:
        raise ValueError("No valid positive-vs-negative within-group pairs found")
    return (
        np.concatenate(pos_parts).astype(np.int64, copy=False),
        np.concatenate(neg_parts).astype(np.int64, copy=False),
    )


def _build_pseudo_pair_indices(
    scores: np.ndarray,
    group_keys: np.ndarray,
    *,
    margin: float,
    max_pairs_per_group: int,
) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    groups = np.asarray(group_keys, dtype=object).reshape(-1)
    by_group: dict[str, list[int]] = {}
    for idx, group_key in enumerate(groups.tolist()):
        by_group.setdefault(str(group_key), []).append(int(idx))

    win_parts: list[np.ndarray] = []
    lose_parts: list[np.ndarray] = []
    for idxs in by_group.values():
        idx_arr = np.asarray(idxs, dtype=np.int64)
        if idx_arr.size < 2:
            continue
        sub_scores = s[idx_arr]
        diff = sub_scores[:, None] - sub_scores[None, :]
        iu, ju = np.triu_indices(idx_arr.size, k=1)
        sub_diff = diff[iu, ju]
        keep = np.where(np.abs(sub_diff) >= float(margin))[0]
        if keep.size <= 0:
            continue
        if int(max_pairs_per_group) > 0 and keep.size > int(max_pairs_per_group):
            keep = keep[np.argsort(-np.abs(sub_diff[keep]), kind="mergesort")[: int(max_pairs_per_group)]]
        left = idx_arr[iu[keep]]
        right = idx_arr[ju[keep]]
        winner = np.where(s[left] >= s[right], left, right)
        loser = np.where(s[left] >= s[right], right, left)
        win_parts.append(np.asarray(winner, dtype=np.int64))
        lose_parts.append(np.asarray(loser, dtype=np.int64))

    if not win_parts:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    return (
        np.concatenate(win_parts).astype(np.int64, copy=False),
        np.concatenate(lose_parts).astype(np.int64, copy=False),
    )


@dataclass(frozen=True)
class TrainConfig:
    source_epochs: int
    transductive_epochs: int
    learning_rate: float
    weight_decay: float
    adapter_init_std: float
    pseudo_pair_margin: float
    pseudo_pair_max_per_group: int
    pseudo_pair_weight: float
    anchor_consistency_weight: float
    covariance_alignment_weight: float
    pseudo_start_epoch: int
    device: str


class LinearRandomAdapterModel(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        adapter_mode: str,
        adapter_rank: int,
        alpha: float,
        init_std: float,
        seed: int,
    ) -> None:
        require_torch()
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.adapter_mode = str(adapter_mode)
        self.adapter_rank = int(adapter_rank)
        self.alpha = float(alpha)
        self.init_std = float(init_std)
        self.seed = int(seed)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))

        self.head_weight = nn.Parameter(torch.zeros(self.latent_dim, dtype=torch.float32))
        self.head_bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))

        if self.adapter_mode == "none":
            self.A = None
            self.B = None
        elif self.adapter_mode == "trained":
            if self.adapter_rank < 1:
                raise ValueError("adapter_rank must be >=1 for trained adapter mode")
            A_init = torch.randn(self.latent_dim, self.adapter_rank, generator=generator, dtype=torch.float32)
            A_init = A_init * float(self.init_std)
            B_init = torch.zeros(self.latent_dim, self.adapter_rank, dtype=torch.float32)
            self.A = nn.Parameter(A_init)
            self.B = nn.Parameter(B_init)
        elif self.adapter_mode == "random_frozen":
            if self.adapter_rank < 1:
                raise ValueError("adapter_rank must be >=1 for random_frozen mode")
            A_val = torch.randn(self.latent_dim, self.adapter_rank, generator=generator, dtype=torch.float32)
            B_val = torch.randn(self.latent_dim, self.adapter_rank, generator=generator, dtype=torch.float32)
            A_val = A_val * float(self.init_std)
            B_val = B_val * float(self.init_std)
            self.register_buffer("A", A_val)
            self.register_buffer("B", B_val)
        else:
            raise ValueError(f"Unknown adapter_mode: {self.adapter_mode}")

    def adapted_latent(self, z: torch.Tensor) -> torch.Tensor:
        z_in = z.to(dtype=torch.float32)
        if self.adapter_mode == "none":
            return z_in
        if self.A is None or self.B is None:
            raise RuntimeError("Adapter parameters are not initialized")
        return z_in + float(self.alpha) * ((z_in @ self.B) @ self.A.T)

    def score(self, z: torch.Tensor) -> torch.Tensor:
        adapted = self.adapted_latent(z)
        return adapted @ self.head_weight + self.head_bias

    def effective_latent_weight(self) -> np.ndarray:
        w = self.head_weight.detach().cpu().numpy().astype(np.float64).reshape(-1)
        if self.adapter_mode == "none":
            return w
        if self.A is None or self.B is None:
            raise RuntimeError("Adapter parameters are not initialized")
        A = self.A.detach().cpu().numpy().astype(np.float64)
        B = self.B.detach().cpu().numpy().astype(np.float64)
        m_t = np.eye(self.latent_dim, dtype=np.float64) + float(self.alpha) * (B @ A.T)
        return np.asarray(m_t @ w, dtype=np.float64)

    def head_bias_value(self) -> float:
        return float(self.head_bias.detach().cpu().numpy().reshape(-1)[0])

    def adapter_residual_norm(self) -> float:
        if self.adapter_mode == "none":
            return 0.0
        if self.A is None or self.B is None:
            raise RuntimeError("Adapter parameters are not initialized")
        A = self.A.detach().cpu().numpy().astype(np.float64)
        B = self.B.detach().cpu().numpy().astype(np.float64)
        residual = float(self.alpha) * (A @ B.T)
        return float(np.linalg.norm(residual, ord="fro"))


def _device(device_name: str) -> torch.device:
    name = str(device_name).strip().lower()
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_tensor(x: np.ndarray, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


def _pairwise_logistic_loss(
    scores: torch.Tensor,
    pos_idx: torch.Tensor,
    neg_idx: torch.Tensor,
) -> torch.Tensor:
    if pos_idx.numel() <= 0:
        raise ValueError("Pairwise loss requires at least one pair")
    margins = scores[pos_idx] - scores[neg_idx]
    return F.softplus(-margins).mean()


def _pointwise_logistic_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(scores, labels)


def _covariance_matrix(z: torch.Tensor) -> torch.Tensor:
    if z.ndim != 2:
        raise ValueError(f"Expected 2D tensor for covariance, got shape={tuple(z.shape)}")
    centered = z - z.mean(dim=0, keepdim=True)
    denom = max(1, int(centered.shape[0]) - 1)
    return (centered.T @ centered) / float(denom)


def _build_model(
    *,
    latent_dim: int,
    adapter_mode: str,
    adapter_rank: int,
    alpha: float,
    init_std: float,
    seed: int,
    device: torch.device,
) -> LinearRandomAdapterModel:
    model = LinearRandomAdapterModel(
        latent_dim=latent_dim,
        adapter_mode=adapter_mode,
        adapter_rank=adapter_rank,
        alpha=alpha,
        init_std=init_std,
        seed=seed,
    )
    return model.to(device=device)


def _train_source_only(
    *,
    z_source_flat: np.ndarray,
    source_labels: np.ndarray,
    source_pair_groups: np.ndarray,
    objective: str,
    adapter_mode: str,
    adapter_rank: int,
    alpha: float,
    seed: int,
    train_cfg: TrainConfig,
) -> LinearRandomAdapterModel:
    device = _device(train_cfg.device)
    model = _build_model(
        latent_dim=int(z_source_flat.shape[1]),
        adapter_mode=adapter_mode,
        adapter_rank=adapter_rank,
        alpha=alpha,
        init_std=float(train_cfg.adapter_init_std),
        seed=seed,
        device=device,
    )
    model.train()
    z_t = _to_tensor(z_source_flat, device=device)
    y_t = _to_tensor(source_labels, device=device)
    pair_pos_np, pair_neg_np = _build_pair_indices(source_labels, source_pair_groups)
    pair_pos_t = _to_tensor(pair_pos_np, device=device, dtype=torch.long)
    pair_neg_t = _to_tensor(pair_neg_np, device=device, dtype=torch.long)

    params = [model.head_weight, model.head_bias]
    if adapter_mode == "trained":
        params.extend([model.A, model.B])  # type: ignore[arg-type]
    optimizer = torch.optim.Adam(
        params,
        lr=float(train_cfg.learning_rate),
        weight_decay=float(train_cfg.weight_decay),
    )

    for _epoch in range(int(train_cfg.source_epochs)):
        optimizer.zero_grad(set_to_none=True)
        scores = model.score(z_t).reshape(-1)
        if str(objective) == "pointwise":
            loss = _pointwise_logistic_loss(scores, y_t)
        elif str(objective) == "pairwise":
            loss = _pairwise_logistic_loss(scores, pair_pos_t, pair_neg_t)
        else:
            raise ValueError(f"Unknown objective: {objective}")
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _train_transductive(
    *,
    source_model: LinearRandomAdapterModel,
    z_source_flat: np.ndarray,
    source_labels: np.ndarray,
    source_pair_groups: np.ndarray,
    z_target_tensor_list: list[np.ndarray],
    target_pair_groups_list: list[np.ndarray],
    objective: str,
    train_cfg: TrainConfig,
) -> LinearRandomAdapterModel:
    if str(source_model.adapter_mode) != "trained":
        raise ValueError("Transductive phase only supports adapter_mode='trained'")

    device = _device(train_cfg.device)
    model = copy.deepcopy(source_model).to(device=device)
    model.train()

    z_source_t = _to_tensor(z_source_flat, device=device)
    source_y_t = _to_tensor(source_labels, device=device)
    source_pair_pos_np, source_pair_neg_np = _build_pair_indices(source_labels, source_pair_groups)
    source_pair_pos_t = _to_tensor(source_pair_pos_np, device=device, dtype=torch.long)
    source_pair_neg_t = _to_tensor(source_pair_neg_np, device=device, dtype=torch.long)
    source_cov_t = _covariance_matrix(z_source_t).detach()

    target_flat_parts = [arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2]) for arr in z_target_tensor_list]
    target_pair_group_parts = [np.asarray(groups, dtype=object) for groups in target_pair_groups_list]
    target_flat_np = np.concatenate(target_flat_parts, axis=0).astype(np.float64, copy=False)
    target_pair_groups_np = np.concatenate(target_pair_group_parts).astype(object, copy=False)

    target_tensor_parts = [_to_tensor(arr, device=device) for arr in z_target_tensor_list]
    target_flat_t = _to_tensor(target_flat_np, device=device)

    optimizer = torch.optim.Adam(
        [model.head_weight, model.head_bias, model.A, model.B],  # type: ignore[list-item]
        lr=float(train_cfg.learning_rate),
        weight_decay=float(train_cfg.weight_decay),
    )

    for epoch_idx in range(int(train_cfg.transductive_epochs)):
        optimizer.zero_grad(set_to_none=True)

        source_scores = model.score(z_source_t).reshape(-1)
        if str(objective) == "pointwise":
            source_loss = _pointwise_logistic_loss(source_scores, source_y_t)
        elif str(objective) == "pairwise":
            source_loss = _pairwise_logistic_loss(source_scores, source_pair_pos_t, source_pair_neg_t)
        else:
            raise ValueError(f"Unknown objective: {objective}")
        total_loss = source_loss

        if int(epoch_idx) >= int(train_cfg.pseudo_start_epoch):
            with torch.no_grad():
                target_scores_np = model.score(target_flat_t).detach().cpu().numpy().reshape(-1)
            pseudo_win_np, pseudo_lose_np = _build_pseudo_pair_indices(
                target_scores_np,
                target_pair_groups_np,
                margin=float(train_cfg.pseudo_pair_margin),
                max_pairs_per_group=int(train_cfg.pseudo_pair_max_per_group),
            )
            if pseudo_win_np.size > 0:
                pseudo_win_t = _to_tensor(pseudo_win_np, device=device, dtype=torch.long)
                pseudo_lose_t = _to_tensor(pseudo_lose_np, device=device, dtype=torch.long)
                target_scores_t = model.score(target_flat_t).reshape(-1)
                pseudo_loss = _pairwise_logistic_loss(target_scores_t, pseudo_win_t, pseudo_lose_t)
                total_loss = total_loss + float(train_cfg.pseudo_pair_weight) * pseudo_loss

        consistency_terms: list[torch.Tensor] = []
        target_adapted_flat_parts: list[torch.Tensor] = []
        for target_tensor in target_tensor_parts:
            target_scores_2d = model.score(target_tensor).reshape(target_tensor.shape[0], target_tensor.shape[1])
            if target_scores_2d.shape[1] != len(LATE_ANCHORS):
                raise ValueError(
                    f"Expected exactly {len(LATE_ANCHORS)} anchors for target consistency, "
                    f"got {target_scores_2d.shape[1]}"
                )
            consistency_terms.append(torch.mean((target_scores_2d[:, 0] - target_scores_2d[:, 1]) ** 2))
            target_adapted_flat_parts.append(
                model.adapted_latent(target_tensor.reshape(target_tensor.shape[0] * target_tensor.shape[1], -1))
            )

        if consistency_terms:
            anchor_consistency = torch.stack(consistency_terms).mean()
            total_loss = total_loss + float(train_cfg.anchor_consistency_weight) * anchor_consistency

        if target_adapted_flat_parts:
            target_adapted_flat = torch.cat(target_adapted_flat_parts, dim=0)
            cov_loss = torch.mean((_covariance_matrix(target_adapted_flat) - source_cov_t) ** 2)
            total_loss = total_loss + float(train_cfg.covariance_alignment_weight) * cov_loss

        total_loss.backward()
        optimizer.step()

    model.eval()
    return model


def _order_indices(scores: np.ndarray, run_ids: np.ndarray) -> np.ndarray:
    score_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    run_arr = np.asarray(run_ids, dtype=np.int64).reshape(-1)
    if score_arr.shape[0] != run_arr.shape[0]:
        raise ValueError("scores and run_ids must have the same length")
    return np.asarray(np.lexsort((run_arr, -score_arr)), dtype=np.int64)


def _problem_groups_from_keys(group_keys: np.ndarray) -> dict[str, list[int]]:
    groups = np.asarray(group_keys, dtype=object).reshape(-1)
    out: dict[str, list[int]] = {}
    for idx, group_key in enumerate(groups.tolist()):
        out.setdefault(str(group_key), []).append(int(idx))
    return out


def _safe_corr(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    left_arr = np.asarray(left, dtype=np.float64).reshape(-1)
    right_arr = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_arr.size != right_arr.size or left_arr.size < 2:
        return None
    if np.std(left_arr) <= 1e-12 or np.std(right_arr) <= 1e-12:
        return None
    corr = float(np.corrcoef(left_arr, right_arr)[0, 1])
    return corr if np.isfinite(corr) else None


def _to_problem_top1_map(scores: np.ndarray, group_keys: np.ndarray, sample_ids: np.ndarray) -> dict[str, int]:
    score_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    groups_arr = np.asarray(group_keys, dtype=object).reshape(-1)
    sample_arr = np.asarray(sample_ids, dtype=np.int64).reshape(-1)
    if not (score_arr.shape[0] == groups_arr.shape[0] == sample_arr.shape[0]):
        raise ValueError("scores, group_keys, and sample_ids must have matching lengths")
    out: dict[str, int] = {}
    for group_key, indices in _problem_groups_from_keys(groups_arr).items():
        idx = np.asarray(indices, dtype=np.int64)
        order = _order_indices(score_arr[idx], sample_arr[idx])
        local_idx = idx[order]
        out[str(group_key)] = int(sample_arr[int(local_idx[0])]) if local_idx.size else int(sample_arr[int(idx[0])])
    return out


def _cov_gap(source_cov: np.ndarray, z_tensor: np.ndarray) -> float:
    source_cov_arr = np.asarray(source_cov, dtype=np.float64)
    z_arr = np.asarray(z_tensor, dtype=np.float64)
    if z_arr.ndim != 3:
        raise ValueError(f"Expected 3D adapted latent tensor, got shape={z_arr.shape}")
    flat = z_arr.reshape(z_arr.shape[0] * z_arr.shape[1], z_arr.shape[2]).astype(np.float64, copy=False)
    if flat.shape[0] <= 1:
        return float("nan")
    centered = flat - np.mean(flat, axis=0, keepdims=True)
    cov = (centered.T @ centered) / float(max(1, flat.shape[0] - 1))
    return float(np.mean((cov - source_cov_arr) ** 2))


def _numpy_covariance(z_tensor: np.ndarray) -> np.ndarray:
    z_arr = np.asarray(z_tensor, dtype=np.float64)
    if z_arr.ndim != 3:
        raise ValueError(f"Expected 3D latent tensor, got shape={z_arr.shape}")
    flat = z_arr.reshape(z_arr.shape[0] * z_arr.shape[1], z_arr.shape[2]).astype(np.float64, copy=False)
    if flat.shape[0] <= 1:
        return np.zeros((flat.shape[1], flat.shape[1]), dtype=np.float64)
    centered = flat - np.mean(flat, axis=0, keepdims=True)
    return (centered.T @ centered) / float(max(1, flat.shape[0] - 1))


def _score_model_on_tensor(model: LinearRandomAdapterModel, z_tensor: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    with torch.no_grad():
        z_t = _to_tensor(z_tensor, device=device)
        return model.score(z_t).detach().cpu().numpy().astype(np.float64)


def _adapted_latent_tensor(model: LinearRandomAdapterModel, z_tensor: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    with torch.no_grad():
        z_t = _to_tensor(z_tensor, device=device)
        return model.adapted_latent(z_t).detach().cpu().numpy().astype(np.float64)


def _proxy_rows_for_dataset(
    *,
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    matrix: RunMatrix,
    proxy_reference: str,
    source_cov: np.ndarray,
    reference_adapted: np.ndarray,
    current_adapted: np.ndarray,
    adapter_residual_norm: float,
) -> dict[str | int, dict[str, Any]]:
    ref_scores_arr = np.asarray(reference_scores, dtype=np.float64)
    cur_scores_arr = np.asarray(current_scores, dtype=np.float64)
    if ref_scores_arr.shape != cur_scores_arr.shape:
        raise ValueError("reference_scores and current_scores must have the same shape")
    if ref_scores_arr.ndim != 2:
        raise ValueError(f"Expected 2D score arrays, got shape={ref_scores_arr.shape}")

    out: dict[str | int, dict[str, Any]] = {}
    corr_values: list[float] = []
    agree_values: list[float] = []
    flip_values: list[float] = []
    delta_values: list[float] = []

    for anchor_idx, anchor_position in enumerate(matrix.anchor_positions):
        anchor_pct = int(round(100.0 * float(anchor_position)))
        ref = ref_scores_arr[:, anchor_idx]
        cur = cur_scores_arr[:, anchor_idx]
        corr = _safe_corr(ref, cur)
        top1_ref = _to_problem_top1_map(ref, matrix.rank_groups, matrix.sample_ids)
        top1_cur = _to_problem_top1_map(cur, matrix.rank_groups, matrix.sample_ids)
        common = sorted(set(top1_ref.keys()) & set(top1_cur.keys()))
        top1_agreement = (
            float(np.mean([int(top1_ref[key] == top1_cur[key]) for key in common])) if common else None
        )
        top1_flip_rate = None if top1_agreement is None else float(1.0 - top1_agreement)
        mean_abs_delta = float(np.mean(np.abs(cur - ref))) if cur.size else None

        if corr is not None:
            corr_values.append(float(corr))
        if top1_agreement is not None:
            agree_values.append(float(top1_agreement))
        if top1_flip_rate is not None:
            flip_values.append(float(top1_flip_rate))
        if mean_abs_delta is not None:
            delta_values.append(float(mean_abs_delta))

        out[anchor_pct] = {
            "proxy_reference": str(proxy_reference),
            "proxy_score_corr": corr,
            "proxy_top1_agreement": top1_agreement,
            "proxy_top1_flip_rate": top1_flip_rate,
            "proxy_mean_abs_score_delta": mean_abs_delta,
            "proxy_anchor_gap_before": None,
            "proxy_anchor_gap_after": None,
            "proxy_anchor_gap_delta": None,
            "proxy_cov_gap_before": None,
            "proxy_cov_gap_after": None,
            "proxy_cov_gap_delta": None,
            "adapter_residual_norm": float(adapter_residual_norm),
        }

    anchor_gap_before = None
    anchor_gap_after = None
    anchor_gap_delta = None
    if ref_scores_arr.shape[1] >= 2:
        anchor_gap_before = float(np.mean(np.abs(ref_scores_arr[:, 0] - ref_scores_arr[:, 1])))
        anchor_gap_after = float(np.mean(np.abs(cur_scores_arr[:, 0] - cur_scores_arr[:, 1])))
        anchor_gap_delta = float(anchor_gap_after - anchor_gap_before)

    cov_gap_before = _cov_gap(source_cov, reference_adapted)
    cov_gap_after = _cov_gap(source_cov, current_adapted)
    cov_gap_delta = (
        None
        if (not np.isfinite(cov_gap_before) or not np.isfinite(cov_gap_after))
        else float(cov_gap_after - cov_gap_before)
    )

    out["late_mean"] = {
        "proxy_reference": str(proxy_reference),
        "proxy_score_corr": None if not corr_values else float(np.mean(corr_values)),
        "proxy_top1_agreement": None if not agree_values else float(np.mean(agree_values)),
        "proxy_top1_flip_rate": None if not flip_values else float(np.mean(flip_values)),
        "proxy_mean_abs_score_delta": None if not delta_values else float(np.mean(delta_values)),
        "proxy_anchor_gap_before": anchor_gap_before,
        "proxy_anchor_gap_after": anchor_gap_after,
        "proxy_anchor_gap_delta": anchor_gap_delta,
        "proxy_cov_gap_before": cov_gap_before if np.isfinite(cov_gap_before) else None,
        "proxy_cov_gap_after": cov_gap_after if np.isfinite(cov_gap_after) else None,
        "proxy_cov_gap_delta": cov_gap_delta,
        "adapter_residual_norm": float(adapter_residual_norm),
    }
    return out


def _evaluate_group_metrics(
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    group_keys: np.ndarray,
    run_ids: np.ndarray,
    labels_available: bool,
) -> dict[str, Any]:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    g = np.asarray(group_keys, dtype=object).reshape(-1)
    r = np.asarray(run_ids, dtype=np.int64).reshape(-1)
    if not (s.shape[0] == y.shape[0] == g.shape[0] == r.shape[0]):
        raise ValueError("scores, labels, group_keys, run_ids must have matching lengths")

    if not bool(labels_available):
        return {
            "auroc": None,
            "rank": None,
            "hit@1": None,
            "hit@3": None,
            "selacc@10%": None,
            "pairwise": None,
            "n_problems": int(np.unique(g).shape[0]),
            "n_samples": int(s.size),
            "top10_count": max(1, int(math.ceil(0.10 * s.size))) if s.size > 0 else 0,
        }

    auroc: Optional[float] = None
    if y.size > 0 and np.unique(y).size >= 2:
        auroc = float(roc_auc_score(y, s))

    by_group: dict[str, list[int]] = {}
    for idx, group_key in enumerate(g.tolist()):
        by_group.setdefault(str(group_key), []).append(int(idx))

    hit1_total = 0.0
    hit3_total = 0.0
    pairwise_num = 0.0
    pairwise_den = 0.0
    rank_total = 0.0
    n_problems = 0

    for idxs in by_group.values():
        idx_arr = np.asarray(idxs, dtype=np.int64)
        if idx_arr.size <= 0:
            continue
        order_local = _order_indices(s[idx_arr], r[idx_arr])
        ordered_idx = idx_arr[order_local]
        ordered_labels = y[ordered_idx]
        hit1_total += float(ordered_labels[0] > 0)
        hit3_total += float(np.any(ordered_labels[: min(3, ordered_labels.size)] > 0))

        pos_positions = np.where(ordered_labels > 0)[0]
        best_correct_rank = int(pos_positions[0] + 1) if pos_positions.size > 0 else int(ordered_labels.size + 1)
        rank_total += float(best_correct_rank)

        pos_scores = s[idx_arr][y[idx_arr] > 0]
        neg_scores = s[idx_arr][y[idx_arr] <= 0]
        if pos_scores.size > 0 and neg_scores.size > 0:
            diff = pos_scores[:, None] - neg_scores[None, :]
            pairwise_num += float((diff > 0).sum()) + 0.5 * float((diff == 0).sum())
            pairwise_den += float(diff.size)
        n_problems += 1

    selacc10 = 0.0
    top10_count = 0
    if y.size > 0:
        top10_count = max(1, int(math.ceil(0.10 * y.size)))
        order = _order_indices(s, r)
        selacc10 = float(y[order[:top10_count]].mean())

    return {
        "auroc": auroc,
        "rank": float(rank_total / n_problems) if n_problems else None,
        "hit@1": float(hit1_total / n_problems) if n_problems else 0.0,
        "hit@3": float(hit3_total / n_problems) if n_problems else 0.0,
        "selacc@10%": float(selacc10),
        "pairwise": float(pairwise_num / pairwise_den) if pairwise_den > 0 else None,
        "n_problems": int(n_problems),
        "n_samples": int(y.size),
        "top10_count": int(top10_count),
    }


def _evaluate_model_on_matrix(
    *,
    model: LinearRandomAdapterModel,
    z_tensor: np.ndarray,
    matrix: RunMatrix,
    labels_available: bool,
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    z_t = _to_tensor(z_tensor, device=device)
    with torch.no_grad():
        scores_np = model.score(z_t).detach().cpu().numpy().astype(np.float64)
    rows: list[dict[str, Any]] = []
    for anchor_idx, anchor_position in enumerate(matrix.anchor_positions):
        rows.append({
            "anchor_pct": int(round(100.0 * float(anchor_position))),
            **_evaluate_group_metrics(
                scores=scores_np[:, anchor_idx],
                labels=np.asarray(matrix.labels, dtype=np.int32),
                group_keys=np.asarray(matrix.rank_groups, dtype=object),
                run_ids=np.asarray(matrix.sample_ids, dtype=np.int32),
                labels_available=bool(labels_available),
            ),
        })
    return rows


def _aggregate_anchor_rows(anchor_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not anchor_rows:
        return {
            "anchor_pct": "late_mean",
            "auroc": None,
            "rank": None,
            "hit@1": None,
            "hit@3": None,
            "selacc@10%": None,
            "pairwise": None,
            "n_problems": 0,
            "n_samples": 0,
            "top10_count": 0,
        }
    out: dict[str, Any] = {"anchor_pct": "late_mean"}
    for key in ("auroc", "rank", "hit@1", "hit@3", "selacc@10%", "pairwise"):
        values = [row.get(key) for row in anchor_rows]
        finite = [float(v) for v in values if v is not None and np.isfinite(float(v))]
        out[key] = None if not finite else float(np.mean(finite))
    out["n_problems"] = int(sum(int(row.get("n_problems") or 0) for row in anchor_rows))
    out["n_samples"] = int(sum(int(row.get("n_samples") or 0) for row in anchor_rows))
    out["top10_count"] = int(sum(int(row.get("top10_count") or 0) for row in anchor_rows))
    return out


def _mean_finite(values: Iterable[Any]) -> Optional[float]:
    finite: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            numeric = float(value)
        except Exception:
            continue
        if np.isfinite(numeric):
            finite.append(float(numeric))
    return None if not finite else float(np.mean(finite))


def _back_project_weights(
    basis: dict[str, Any],
    effective_latent_weight: np.ndarray,
    head_bias: float,
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    latent_weight = np.asarray(effective_latent_weight, dtype=np.float64).reshape(-1)
    components = np.asarray(basis["svd"].components_, dtype=np.float64)[: latent_weight.shape[0], :]
    if bool(basis.get("whiten", False)):
        singular_values = np.asarray(basis["svd"].singular_values_, dtype=np.float64)[: latent_weight.shape[0]]
        singular_values = np.where(np.abs(singular_values) < 1e-8, 1.0, singular_values)
        latent_weight = latent_weight / singular_values
    scaled_weight = np.asarray(components.T @ latent_weight, dtype=np.float64).reshape(-1)
    scaler = basis["scaler"]
    scale = np.asarray(scaler.scale_, dtype=np.float64).reshape(-1)
    scale = np.where(np.abs(scale) < 1e-8, 1.0, scale)
    mean = np.asarray(scaler.mean_, dtype=np.float64).reshape(-1)
    orig_weight = scaled_weight / scale
    orig_bias = float(head_bias - np.dot(mean, orig_weight))

    rep_feature_names = tuple([f"raw::{name}" for name in feature_names] + [f"rank::{name}" for name in feature_names])
    if len(rep_feature_names) != orig_weight.shape[0]:
        raise ValueError(
            f"Feature-name count mismatch: names={len(rep_feature_names)} weights={orig_weight.shape[0]}"
        )
    order = np.argsort(-np.abs(orig_weight), kind="mergesort")
    top_features = [
        {"feature": str(rep_feature_names[int(idx)]), "weight": float(orig_weight[int(idx)])}
        for idx in order[: min(8, order.size)].tolist()
    ]
    return {
        "effective_latent_weight": latent_weight.tolist(),
        "original_representation_weight": orig_weight.tolist(),
        "original_representation_bias": float(orig_bias),
        "top_features": top_features,
    }


def _fmt_metric(value: Any, *, pct: bool = False, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except Exception:
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    if pct:
        return f"{100.0 * numeric:.2f}%"
    return f"{numeric:.{digits}f}"


def _row_to_config_key(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "protocol": str(row["protocol"]),
            "family": str(row["feature_family"]),
            "condition": str(row["condition_name"]),
            "objective": str(row["objective"]),
            "seed": int(row["seed"]),
            "adapter_rank": int(row["adapter_rank"]),
            "alpha": float(row["alpha"]),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _select_metric_name(condition_name: str, objective: str) -> str:
    if str(objective) == "pairwise":
        return "pairwise"
    if str(condition_name).startswith("trained_adapter"):
        return "auroc"
    return "auroc"


def _aggregate_seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = json.dumps(
            {
                "protocol": row["protocol"],
                "feature_family": row["feature_family"],
                "condition_name": row["condition_name"],
                "objective": row["objective"],
                "dataset_role": row["dataset_role"],
                "anchor_pct": row["anchor_pct"],
                "adapter_rank": row["adapter_rank"],
                "alpha": row["alpha"],
                "basis_rank": row["basis_rank"],
                "basis_whiten": row["basis_whiten"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for key in sorted(grouped.keys()):
        members = grouped[key]
        base = dict(members[0])
        base["seed"] = "mean"
        base["n_seeds"] = int(len({int(row["seed"]) for row in members}))
        for metric_name in ("auroc", "rank", "hit@1", "hit@3", "selacc@10%", "pairwise"):
            base[metric_name] = _mean_finite(row.get(metric_name) for row in members)
        for metric_name in (
            "proxy_score_corr",
            "proxy_top1_agreement",
            "proxy_top1_flip_rate",
            "proxy_mean_abs_score_delta",
            "proxy_anchor_gap_before",
            "proxy_anchor_gap_after",
            "proxy_anchor_gap_delta",
            "proxy_cov_gap_before",
            "proxy_cov_gap_after",
            "proxy_cov_gap_delta",
            "adapter_residual_norm",
        ):
            base[metric_name] = _mean_finite(row.get(metric_name) for row in members)
        proxy_refs = sorted(
            {
                str(row.get("proxy_reference"))
                for row in members
                if row.get("proxy_reference") not in (None, "", "None")
            }
        )
        base["proxy_reference"] = None if not proxy_refs else " / ".join(proxy_refs)
        base["n_problems"] = int(np.mean([int(row["n_problems"]) for row in members])) if members else 0
        base["n_samples"] = int(np.mean([int(row["n_samples"]) for row in members])) if members else 0
        base["top10_count"] = int(np.mean([int(row["top10_count"]) for row in members])) if members else 0
        out.append(base)
    return out


def _representative_model_key(
    rows: list[dict[str, Any]],
    selected_row: dict[str, Any],
) -> Optional[str]:
    metric_name = _select_metric_name(str(selected_row["condition_name"]), str(selected_row["objective"]))
    target_metric = selected_row.get(metric_name)
    matched = [
        row
        for row in rows
        if str(row["protocol"]) == str(selected_row["protocol"])
        and str(row["feature_family"]) == str(selected_row["feature_family"])
        and str(row["condition_name"]) == str(selected_row["condition_name"])
        and str(row["objective"]) == str(selected_row["objective"])
        and str(row["dataset_role"]) == "source_holdout"
        and str(row["anchor_pct"]) == "late_mean"
        and int(row["adapter_rank"]) == int(selected_row["adapter_rank"])
        and abs(float(row["alpha"]) - float(selected_row["alpha"])) < 1e-12
        and int(row["basis_rank"]) == int(selected_row["basis_rank"])
        and bool(row["basis_whiten"]) == bool(selected_row["basis_whiten"])
    ]
    if not matched:
        return None
    if target_metric is None:
        return str(sorted(matched, key=lambda row: int(row["seed"]))[0]["model_key"])
    numeric_target = float(target_metric)
    matched_sorted = sorted(
        matched,
        key=lambda row: (
            abs(float(row.get(metric_name) or 0.0) - numeric_target),
            int(row["seed"]),
        ),
    )
    return str(matched_sorted[0]["model_key"])


def _select_summary_configs(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    seed_mean_rows = _aggregate_seed_rows(rows)
    late_source_rows = [
        row
        for row in seed_mean_rows
        if str(row["dataset_role"]) == "source_holdout" and str(row["anchor_pct"]) == "late_mean"
    ]
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in late_source_rows:
        key = (str(row["protocol"]), str(row["feature_family"]), str(row["condition_name"]))
        grouped.setdefault(key, []).append(row)

    for key, members in grouped.items():
        objective = str(members[0]["objective"])
        metric_name = _select_metric_name(str(members[0]["condition_name"]), objective)
        members_sorted = sorted(
            members,
            key=lambda row: (
                -float(row[metric_name] or float("-inf")),
                -float(row["hit@1"] or float("-inf")),
                float(row["rank"] or float("inf")),
                -float(row["auroc"] or float("-inf")),
                int(row["adapter_rank"]),
                float(row["alpha"]),
            ),
        )
        selected[key] = dict(members_sorted[0])
    return selected


def _matching_rows_for_selected(
    rows: list[dict[str, Any]],
    selected_row: dict[str, Any],
) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    for row in rows:
        if (
            str(row["protocol"]) == str(selected_row["protocol"])
            and str(row["feature_family"]) == str(selected_row["feature_family"])
            and str(row["condition_name"]) == str(selected_row["condition_name"])
            and str(row["objective"]) == str(selected_row["objective"])
            and int(row["adapter_rank"]) == int(selected_row["adapter_rank"])
            and abs(float(row["alpha"]) - float(selected_row["alpha"])) < 1e-12
            and int(row["basis_rank"]) == int(selected_row["basis_rank"])
            and bool(row["basis_whiten"]) == bool(selected_row["basis_whiten"])
        ):
            matched.append(row)
    return matched


def _build_doc(
    *,
    out_path: Path,
    rows: list[dict[str, Any]],
    selected_configs: dict[tuple[str, str, str], dict[str, Any]],
    model_summaries: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    seed_mean_rows = _aggregate_seed_rows(rows)
    lines: list[str] = []
    lines.append("# Coding Random Adapter")
    lines.append("")
    lines.append(f"**Date**: {_now_utc()}  ")
    lines.append("**Status**: completed")
    lines.append("")
    if bool(args.smoke):
        lines.append("> This artifact is a smoke-sized validation run. Interpret answers as provisional.")
        lines.append("")

    target_labels_available = any(
        int(row.get("labels_available", 0)) > 0
        and str(row.get("dataset_role")) in {"target_ds_test", "target_qwen_test"}
        for row in rows
    )

    lines.append("## Summary")
    lines.append("")
    lines.append("- Domain: `coding` / `livecodebench_v5` only.")
    lines.append("- Source labeled: `DeepSeek lcb_v5` from `MUI_HUB/cache`.")
    lines.append("- Target unlabeled during adaptation: `DeepSeek lcb_v5 cache_test`, `Qwen lcb_v5 cache_test`.")
    lines.append("- Anchors: shared late-anchor training over `70%` and `100%`, with per-anchor evaluation.")
    lines.append("- Feature families: `canonical_22` (`token_plus_traj_fixed`) and `token_only`, both with `raw+rank`.")
    lines.append("- Frozen basis: `rank=12`, `whiten=False`, mirroring the current late-anchor coding SVD setting.")
    lines.append("- Adapter sweep: `rank ∈ {"
                 + ",".join(str(int(v)) for v in _parse_int_csv(args.adapter_ranks))
                 + "}`, `alpha ∈ {"
                 + ",".join(f"{float(v):.2f}" for v in _parse_float_csv(args.alphas))
                 + "}`.")
    lines.append("")

    if not bool(target_labels_available):
        lines.append("## Important Limitation")
        lines.append("")
        lines.append("- The local repo does **not** expose per-sample correctness labels for the coding `cache_test` roots used here.")
        lines.append("- Because of that, source metrics are fully computed, but target `AUROC` / `Rank` / `Hit@1` / `Hit@3` / `SelAcc@10%` / `Pairwise Acc` remain unavailable offline.")
        lines.append("- The transductive training path is still implemented correctly with unlabeled targets; external blind evaluation or a labeled target artifact is still required for final DS->Qwen claims.")
        lines.append("")

    lines.append("## Selected Configs")
    lines.append("")
    lines.append("| Protocol | Family | Condition | Objective | Adapter Rank | Alpha | Source Holdout AUROC | Source Holdout Rank | Source Holdout Hit@1 | Source Holdout Pairwise |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for key in sorted(selected_configs.keys()):
        row = selected_configs[key]
        lines.append(
            "| {protocol} | {family} | {condition} | {objective} | {adapter_rank} | {alpha:.2f} | {auroc} | {rank} | {hit1} | {pairwise} |".format(
                protocol=str(row["protocol"]),
                family=str(row["feature_family"]),
                condition=str(row["condition_name"]),
                objective=str(row["objective"]),
                adapter_rank=int(row["adapter_rank"]),
                alpha=float(row["alpha"]),
                auroc=_fmt_metric(row.get("auroc")),
                rank=_fmt_metric(row.get("rank")),
                hit1=_fmt_metric(row.get("hit@1"), pct=True),
                pairwise=_fmt_metric(row.get("pairwise"), pct=True),
            )
        )
    lines.append("")

    lines.append("## Target Results")
    lines.append("")
    lines.append("| Protocol | Family | Condition | Objective | Dataset | Anchor | AUROC | Rank | Hit@1 | Hit@3 | SelAcc@10 | Pairwise Acc |")
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    selected_rows_all: list[dict[str, Any]] = []
    selected_mean_rows: list[dict[str, Any]] = []
    for selected_row in selected_configs.values():
        for row in _matching_rows_for_selected(seed_mean_rows, selected_row):
            selected_rows_all.append(row)
            if str(row["dataset_role"]) not in {"target_ds_test", "target_qwen_test"}:
                continue
            selected_mean_rows.append(row)
    for row in sorted(
        selected_mean_rows,
        key=lambda item: (
            str(item["protocol"]),
            str(item["feature_family"]),
            str(item["condition_name"]),
            str(item["dataset_role"]),
            str(item["anchor_pct"]),
        ),
    ):
        lines.append(
            "| {protocol} | {family} | {condition} | {objective} | {dataset_role} | {anchor_pct} | {auroc} | {rank} | {hit1} | {hit3} | {selacc} | {pairwise} |".format(
                protocol=str(row["protocol"]),
                family=str(row["feature_family"]),
                condition=str(row["condition_name"]),
                objective=str(row["objective"]),
                dataset_role=str(row["dataset_role"]),
                anchor_pct=str(row["anchor_pct"]),
                auroc=_fmt_metric(row.get("auroc")),
                rank=_fmt_metric(row.get("rank")),
                hit1=_fmt_metric(row.get("hit@1"), pct=True),
                hit3=_fmt_metric(row.get("hit@3"), pct=True),
                selacc=_fmt_metric(row.get("selacc@10%"), pct=True),
                pairwise=_fmt_metric(row.get("pairwise"), pct=True),
            )
        )
    lines.append("")

    def _pick(protocol: str, family: str, condition: str, dataset_role: str) -> Optional[dict[str, Any]]:
        for row in selected_rows_all:
            if (
                str(row["protocol"]) == str(protocol)
                and str(row["feature_family"]) == str(family)
                and str(row["condition_name"]) == str(condition)
                and str(row["dataset_role"]) == str(dataset_role)
                and str(row["anchor_pct"]) == "late_mean"
            ):
                return row
        return None

    def _pick_matched_source_only(
        *,
        family: str,
        condition: str,
        objective: str,
        adapter_rank: int,
        alpha: float,
        dataset_role: str,
    ) -> Optional[dict[str, Any]]:
        for row in seed_mean_rows:
            if (
                str(row["protocol"]) == "source_only_inductive"
                and str(row["feature_family"]) == str(family)
                and str(row["condition_name"]) == str(condition)
                and str(row["objective"]) == str(objective)
                and str(row["dataset_role"]) == str(dataset_role)
                and str(row["anchor_pct"]) == "late_mean"
                and int(row["adapter_rank"]) == int(adapter_rank)
                and abs(float(row["alpha"]) - float(alpha)) < 1e-12
            ):
                return row
        return None

    transductive_source_delta_rows: list[dict[str, Any]] = []
    for family in ("canonical_22", "token_only"):
        for objective, condition in (
            ("pointwise", "trained_adapter_pointwise"),
            ("pairwise", "trained_adapter_pairwise"),
        ):
            tx_row = _pick("transductive_target_unlabeled", family, condition, "source_holdout")
            if tx_row is None:
                continue
            src_row = _pick_matched_source_only(
                family=family,
                condition=condition,
                objective=objective,
                adapter_rank=int(tx_row["adapter_rank"]),
                alpha=float(tx_row["alpha"]),
                dataset_role="source_holdout",
            )
            if src_row is None:
                continue
            transductive_source_delta_rows.append(
                {
                    "family": family,
                    "objective": objective,
                    "condition": condition,
                    "adapter_rank": int(tx_row["adapter_rank"]),
                    "alpha": float(tx_row["alpha"]),
                    "delta_auroc": None
                    if tx_row.get("auroc") is None or src_row.get("auroc") is None
                    else float(tx_row["auroc"]) - float(src_row["auroc"]),
                    "delta_rank": None
                    if tx_row.get("rank") is None or src_row.get("rank") is None
                    else float(src_row["rank"]) - float(tx_row["rank"]),
                    "delta_hit1": None
                    if tx_row.get("hit@1") is None or src_row.get("hit@1") is None
                    else float(tx_row["hit@1"]) - float(src_row["hit@1"]),
                    "delta_pairwise": None
                    if tx_row.get("pairwise") is None or src_row.get("pairwise") is None
                    else float(tx_row["pairwise"]) - float(src_row["pairwise"]),
                }
            )

    if transductive_source_delta_rows:
        lines.append("## Source-Holdout Effect")
        lines.append("")
        lines.append("| Family | Objective | Adapter Rank | Alpha | ΔAUROC | ΔRank | ΔHit@1 | ΔPairwise |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for row in sorted(transductive_source_delta_rows, key=lambda item: (str(item["family"]), str(item["objective"]))):
            lines.append(
                "| {family} | {objective} | {adapter_rank} | {alpha:.2f} | {delta_auroc} | {delta_rank} | {delta_hit1} | {delta_pairwise} |".format(
                    family=str(row["family"]),
                    objective=str(row["objective"]),
                    adapter_rank=int(row["adapter_rank"]),
                    alpha=float(row["alpha"]),
                    delta_auroc=_fmt_metric(row.get("delta_auroc")),
                    delta_rank=_fmt_metric(row.get("delta_rank")),
                    delta_hit1=_fmt_metric(row.get("delta_hit1"), pct=True),
                    delta_pairwise=_fmt_metric(row.get("delta_pairwise"), pct=True),
                )
            )
        lines.append("")

    transductive_target_proxy_rows = [
        row
        for row in selected_mean_rows
        if str(row["protocol"]) == "transductive_target_unlabeled"
        and str(row["anchor_pct"]) == "late_mean"
        and row.get("proxy_reference") not in (None, "", "None")
    ]
    if transductive_target_proxy_rows:
        lines.append("## Unlabeled Target Effect")
        lines.append("")
        lines.append(
            "| Family | Objective | Dataset | Adapter Rank | Alpha | Proxy Ref | Score Corr | Top1 Agree | Top1 Flip | Mean Abs Score Δ | ΔAnchor Gap | ΔCov Gap | Adapter Residual |"
        )
        lines.append("|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in sorted(
            transductive_target_proxy_rows,
            key=lambda item: (str(item["feature_family"]), str(item["objective"]), str(item["dataset_role"])),
        ):
            lines.append(
                "| {family} | {objective} | {dataset_role} | {adapter_rank} | {alpha:.2f} | {proxy_reference} | {corr} | {agree} | {flip} | {delta} | {anchor_gap} | {cov_gap} | {resid} |".format(
                    family=str(row["feature_family"]),
                    objective=str(row["objective"]),
                    dataset_role=str(row["dataset_role"]),
                    adapter_rank=int(row["adapter_rank"]),
                    alpha=float(row["alpha"]),
                    proxy_reference=str(row.get("proxy_reference") or "n/a"),
                    corr=_fmt_metric(row.get("proxy_score_corr")),
                    agree=_fmt_metric(row.get("proxy_top1_agreement"), pct=True),
                    flip=_fmt_metric(row.get("proxy_top1_flip_rate"), pct=True),
                    delta=_fmt_metric(row.get("proxy_mean_abs_score_delta")),
                    anchor_gap=_fmt_metric(row.get("proxy_anchor_gap_delta")),
                    cov_gap=_fmt_metric(row.get("proxy_cov_gap_delta")),
                    resid=_fmt_metric(row.get("adapter_residual_norm")),
                )
            )
        lines.append("")

    q1_notes: list[str] = []
    q2_pairwise_deltas: list[float] = []
    q2_pointwise_deltas: list[float] = []
    token_benefits: list[float] = []
    canon_benefits: list[float] = []
    ranking_improvements: list[float] = []
    auroc_improvements: list[float] = []

    q1_answer = "unresolved locally"
    q2_answer = "unresolved locally"
    q3_answer = "unresolved locally"
    q4_answer = "unresolved locally"
    q5_answer = "unresolved locally"

    def _proxy_preference(
        preferred_rows: list[dict[str, Any]],
        other_rows: list[dict[str, Any]],
    ) -> str:
        if not preferred_rows or not other_rows:
            return "unresolved locally"
        votes = 0
        pref_corr = _mean_finite(row.get("proxy_score_corr") for row in preferred_rows)
        other_corr = _mean_finite(row.get("proxy_score_corr") for row in other_rows)
        pref_agree = _mean_finite(row.get("proxy_top1_agreement") for row in preferred_rows)
        other_agree = _mean_finite(row.get("proxy_top1_agreement") for row in other_rows)
        pref_flip = _mean_finite(row.get("proxy_top1_flip_rate") for row in preferred_rows)
        other_flip = _mean_finite(row.get("proxy_top1_flip_rate") for row in other_rows)
        if pref_corr is not None and other_corr is not None:
            votes += 1 if pref_corr > other_corr else -1 if pref_corr < other_corr else 0
        if pref_agree is not None and other_agree is not None:
            votes += 1 if pref_agree > other_agree else -1 if pref_agree < other_agree else 0
        if pref_flip is not None and other_flip is not None:
            votes += 1 if pref_flip < other_flip else -1 if pref_flip > other_flip else 0
        if votes > 0:
            return "proxy-leaning yes"
        if votes < 0:
            return "proxy-leaning no"
        return "mixed"

    if bool(target_labels_available):
        for family in ("canonical_22", "token_only"):
            pair_base = _pick("source_only_inductive", family, "frozen_basis_pairwise", "target_qwen_test")
            pair_tx = _pick("transductive_target_unlabeled", family, "trained_adapter_pairwise", "target_qwen_test")
            if pair_base is not None and pair_tx is not None:
                delta = float(pair_tx.get("pairwise") or 0.0) - float(pair_base.get("pairwise") or 0.0)
                q1_notes.append(f"{family}: pairwise Δ={delta:+.4f}")

            pt_base = _pick("source_only_inductive", family, "frozen_basis_pointwise", "target_qwen_test")
            pt_tx = _pick("transductive_target_unlabeled", family, "trained_adapter_pointwise", "target_qwen_test")
            if pair_base is not None and pair_tx is not None:
                q2_pairwise_deltas.append(float(pair_tx.get("pairwise") or 0.0) - float(pair_base.get("pairwise") or 0.0))
            if pt_base is not None and pt_tx is not None:
                q2_pointwise_deltas.append(float(pt_tx.get("auroc") or 0.0) - float(pt_base.get("auroc") or 0.0))
            if family == "token_only" and pair_base is not None and pair_tx is not None:
                token_benefits.append(float(pair_tx.get("pairwise") or 0.0) - float(pair_base.get("pairwise") or 0.0))
            if family == "canonical_22" and pair_base is not None and pair_tx is not None:
                canon_benefits.append(float(pair_tx.get("pairwise") or 0.0) - float(pair_base.get("pairwise") or 0.0))
            if pair_base is not None and pair_tx is not None:
                rank_delta = float(pair_base.get("rank") or 0.0) - float(pair_tx.get("rank") or 0.0)
                hit1_delta = float(pair_tx.get("hit@1") or 0.0) - float(pair_base.get("hit@1") or 0.0)
                pair_delta = float(pair_tx.get("pairwise") or 0.0) - float(pair_base.get("pairwise") or 0.0)
                ranking_improvements.append(float(np.mean([rank_delta, hit1_delta, pair_delta])))
                auroc_improvements.append(float(pair_tx.get("auroc") or 0.0) - float(pair_base.get("auroc") or 0.0))

        if q1_notes:
            deltas = []
            for note in q1_notes:
                try:
                    deltas.append(float(note.split("Δ=")[-1]))
                except Exception:
                    pass
            q1_answer = "mixed"
            if deltas and max(deltas) > 0.005:
                q1_answer = "yes (at least one family improves)"
            elif deltas and max(deltas) <= 0.0:
                q1_answer = "no"

        q2_answer = "mixed"
        if q2_pairwise_deltas and q2_pointwise_deltas:
            if float(np.mean(q2_pairwise_deltas)) > float(np.mean(q2_pointwise_deltas)):
                q2_answer = "yes"
            elif float(np.mean(q2_pairwise_deltas)) < float(np.mean(q2_pointwise_deltas)):
                q2_answer = "no"

        q3_answer = "mixed"
        if token_benefits and canon_benefits:
            if float(np.mean(token_benefits)) > float(np.mean(canon_benefits)):
                q3_answer = "yes"
            elif float(np.mean(token_benefits)) < float(np.mean(canon_benefits)):
                q3_answer = "no"

        q4_answer = "mixed"
        if ranking_improvements and auroc_improvements:
            if float(np.mean(ranking_improvements)) > float(np.mean(auroc_improvements)):
                q4_answer = "yes"
            elif float(np.mean(ranking_improvements)) < float(np.mean(auroc_improvements)):
                q4_answer = "no"

        q5_answer = "mixed"
        if q2_answer == "yes" and q4_answer == "yes":
            q5_answer = "yes"
        elif q2_answer == "no" and q4_answer == "no":
            q5_answer = "no"
    else:
        q1_qwen_proxy = [
            row
            for row in transductive_target_proxy_rows
            if str(row["dataset_role"]) == "target_qwen_test"
        ]
        q1_ds_proxy = [
            row
            for row in transductive_target_proxy_rows
            if str(row["dataset_role"]) == "target_ds_test"
        ]
        if q1_qwen_proxy:
            q1_notes.append(
                "Qwen proxy: mean corr={}, top1 flip={}, Δcov={}".format(
                    _fmt_metric(_mean_finite(row.get("proxy_score_corr") for row in q1_qwen_proxy)),
                    _fmt_metric(_mean_finite(row.get("proxy_top1_flip_rate") for row in q1_qwen_proxy), pct=True),
                    _fmt_metric(_mean_finite(row.get("proxy_cov_gap_delta") for row in q1_qwen_proxy)),
                )
            )
        if q1_ds_proxy:
            q1_notes.append(
                "DS-test proxy: mean corr={}, top1 flip={}, Δcov={}".format(
                    _fmt_metric(_mean_finite(row.get("proxy_score_corr") for row in q1_ds_proxy)),
                    _fmt_metric(_mean_finite(row.get("proxy_top1_flip_rate") for row in q1_ds_proxy), pct=True),
                    _fmt_metric(_mean_finite(row.get("proxy_cov_gap_delta") for row in q1_ds_proxy)),
                )
            )

        pairwise_qwen_proxy = [
            row
            for row in transductive_target_proxy_rows
            if str(row["dataset_role"]) == "target_qwen_test" and str(row["objective"]) == "pairwise"
        ]
        pointwise_qwen_proxy = [
            row
            for row in transductive_target_proxy_rows
            if str(row["dataset_role"]) == "target_qwen_test" and str(row["objective"]) == "pointwise"
        ]
        q2_answer = _proxy_preference(pairwise_qwen_proxy, pointwise_qwen_proxy)

        token_qwen_proxy = [
            row
            for row in transductive_target_proxy_rows
            if str(row["dataset_role"]) == "target_qwen_test" and str(row["feature_family"]) == "token_only"
        ]
        canon_qwen_proxy = [
            row
            for row in transductive_target_proxy_rows
            if str(row["dataset_role"]) == "target_qwen_test" and str(row["feature_family"]) == "canonical_22"
        ]
        q3_answer = _proxy_preference(token_qwen_proxy, canon_qwen_proxy)

        ranking_effect = _mean_finite(
            [row.get("delta_pairwise") for row in transductive_source_delta_rows if str(row["objective"]) == "pairwise"]
            + [row.get("delta_hit1") for row in transductive_source_delta_rows if str(row["objective"]) == "pairwise"]
        )
        auroc_effect = _mean_finite(
            row.get("delta_auroc") for row in transductive_source_delta_rows if str(row["objective"]) == "pointwise"
        )
        if ranking_effect is not None and auroc_effect is not None:
            if ranking_effect > auroc_effect:
                q4_answer = "source-holdout leaning yes"
            elif ranking_effect < auroc_effect:
                q4_answer = "source-holdout leaning no"
            else:
                q4_answer = "mixed"

        if q2_answer == "proxy-leaning yes" and q4_answer == "source-holdout leaning yes":
            q5_answer = "proxy-leaning yes"
        elif q2_answer == "proxy-leaning no" and q4_answer == "source-holdout leaning no":
            q5_answer = "proxy-leaning no"
        else:
            q5_answer = "mixed"

    lines.append("## Questions")
    lines.append("")
    lines.append(f"1. **Does a tiny low-rank adapter help DS->Qwen transfer?** {q1_answer}.")
    if bool(target_labels_available) and q1_notes:
        lines.append(f"   - Evidence: {', '.join(q1_notes)}.")
    elif not bool(target_labels_available):
        lines.append("   - Local target correctness labels are unavailable, so this artifact cannot prove or disprove target transfer offline.")
        if q1_notes:
            lines.append(f"   - Proxy readout: {'; '.join(q1_notes)}.")
    lines.append(f"2. **Is the gain larger for pairwise ranking than for pointwise correctness?** {q2_answer}.")
    if bool(target_labels_available) and q2_pairwise_deltas and q2_pointwise_deltas:
        lines.append(
            "   - Mean Qwen deltas: pairwise metric `{}` vs pointwise AUROC `{}`.".format(
                _fmt_metric(float(np.mean(q2_pairwise_deltas))),
                _fmt_metric(float(np.mean(q2_pointwise_deltas))),
            )
        )
    elif not bool(target_labels_available) and pairwise_qwen_proxy and pointwise_qwen_proxy:
        lines.append(
            "   - Proxy stability on Qwen: pairwise corr `{}` / flip `{}` vs pointwise corr `{}` / flip `{}`.".format(
                _fmt_metric(_mean_finite(row.get("proxy_score_corr") for row in pairwise_qwen_proxy)),
                _fmt_metric(_mean_finite(row.get("proxy_top1_flip_rate") for row in pairwise_qwen_proxy), pct=True),
                _fmt_metric(_mean_finite(row.get("proxy_score_corr") for row in pointwise_qwen_proxy)),
                _fmt_metric(_mean_finite(row.get("proxy_top1_flip_rate") for row in pointwise_qwen_proxy), pct=True),
            )
        )
    lines.append(f"3. **Does token_only benefit more than canonical_22?** {q3_answer}.")
    if bool(target_labels_available) and token_benefits and canon_benefits:
        lines.append(
            "   - Mean Qwen pairwise deltas: `token_only={}`, `canonical_22={}`.".format(
                _fmt_metric(float(np.mean(token_benefits))),
                _fmt_metric(float(np.mean(canon_benefits))),
            )
        )
    elif not bool(target_labels_available) and token_qwen_proxy and canon_qwen_proxy:
        lines.append(
            "   - Proxy stability on Qwen: token corr `{}` / flip `{}` vs canonical corr `{}` / flip `{}`.".format(
                _fmt_metric(_mean_finite(row.get("proxy_score_corr") for row in token_qwen_proxy)),
                _fmt_metric(_mean_finite(row.get("proxy_top1_flip_rate") for row in token_qwen_proxy), pct=True),
                _fmt_metric(_mean_finite(row.get("proxy_score_corr") for row in canon_qwen_proxy)),
                _fmt_metric(_mean_finite(row.get("proxy_top1_flip_rate") for row in canon_qwen_proxy), pct=True),
            )
        )
    lines.append(f"4. **Is the improvement mainly in Rank / Hit@1 / Pairwise Acc rather than AUROC?** {q4_answer}.")
    if bool(target_labels_available) and ranking_improvements and auroc_improvements:
        lines.append(
            "   - Mean ranking composite `{}` vs AUROC delta `{}`.".format(
                _fmt_metric(float(np.mean(ranking_improvements))),
                _fmt_metric(float(np.mean(auroc_improvements))),
            )
        )
    elif not bool(target_labels_available) and ranking_effect is not None and auroc_effect is not None:
        lines.append(
            "   - Source-holdout deltas: ranking-focused `{}` vs pointwise AUROC `{}`.".format(
                _fmt_metric(ranking_effect),
                _fmt_metric(auroc_effect),
            )
        )
    lines.append(f"5. **Does this support weak reusable relative signal but unstable absolute correctness?** {q5_answer}.")
    lines.append("")

    lines.append("## Interpretability")
    lines.append("")
    lines.append("- Final adapted score remains affine in the frozen latent coordinates.")
    lines.append("- With column-vector notation `z' = M z`, `M = I + A B^T`, and `score = w^T z' + b`,")
    lines.append("  the exact frozen-basis score is `score = (M^T w)^T z + b`.")
    lines.append("- With the frozen SVD map `z = V x_scaled` (equivalently `x_scaled @ V^T` in row form),")
    lines.append("  the original raw+rank feature score is still linear after back-projection through the fixed scaler + SVD.")
    lines.append("")

    interpret_rows = [
        row
        for row in selected_configs.values()
        if str(row["protocol"]) == "transductive_target_unlabeled"
        and str(row["condition_name"]).startswith("trained_adapter")
    ]
    if interpret_rows:
        lines.append("| Family | Condition | Top Back-Projected Features |")
        lines.append("|---|---|---|")
        for row in sorted(interpret_rows, key=lambda item: (str(item["feature_family"]), str(item["condition_name"]))):
            model_key = str(row["model_key"])
            summary = model_summaries.get(model_key, {})
            features = summary.get("top_features", [])
            feature_text = ", ".join(
                f"`{item['feature']}` ({float(item['weight']):+.4f})"
                for item in list(features)[:5]
            ) or "n/a"
            lines.append(
                "| {family} | {condition} | {features} |".format(
                    family=str(row["feature_family"]),
                    condition=str(row["condition_name"]),
                    features=feature_text,
                )
            )
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Target labels are never used during training or transductive adaptation.")
    lines.append("- The optional full-label target upper reference is intentionally omitted from this artifact.")
    lines.append("- Exact hyperparameters are stored row-wise in `results/tables/coding_random_adapter.csv`.")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class DatasetRun:
    dataset_role: str
    matrix: RunMatrix
    z_tensor: np.ndarray
    labels_available: bool


def _run_condition(
    *,
    protocol: str,
    feature_family: str,
    objective: str,
    condition_name: str,
    adapter_mode: str,
    adapter_rank: int,
    alpha: float,
    seed: int,
    basis: dict[str, Any],
    source_train: DatasetRun,
    source_holdout: DatasetRun,
    target_ds_test: DatasetRun,
    target_qwen_test: DatasetRun,
    train_cfg: TrainConfig,
    model_summaries: dict[str, dict[str, Any]],
    reference_model: Optional[LinearRandomAdapterModel] = None,
    proxy_reference_name: Optional[str] = None,
    source_cov: Optional[np.ndarray] = None,
) -> tuple[list[dict[str, Any]], LinearRandomAdapterModel]:
    source_flat = _flatten_latent_with_metadata(source_train.z_tensor, source_train.matrix)
    base_model = _train_source_only(
        z_source_flat=source_flat["z"],
        source_labels=source_flat["y"],
        source_pair_groups=source_flat["pair_groups"],
        objective=objective,
        adapter_mode=adapter_mode,
        adapter_rank=adapter_rank,
        alpha=alpha,
        seed=seed,
        train_cfg=train_cfg,
    )
    if str(protocol) == "transductive_target_unlabeled":
        if str(adapter_mode) != "trained":
            raise ValueError("Only trained adapters support the transductive protocol")
        target_flat_ds = _flatten_latent_with_metadata(target_ds_test.z_tensor, target_ds_test.matrix)
        target_flat_qw = _flatten_latent_with_metadata(target_qwen_test.z_tensor, target_qwen_test.matrix)
        model = _train_transductive(
            source_model=base_model,
            z_source_flat=source_flat["z"],
            source_labels=source_flat["y"],
            source_pair_groups=source_flat["pair_groups"],
            z_target_tensor_list=[target_ds_test.z_tensor, target_qwen_test.z_tensor],
            target_pair_groups_list=[target_flat_ds["pair_groups"], target_flat_qw["pair_groups"]],
            objective=objective,
            train_cfg=train_cfg,
        )
    else:
        model = base_model

    model_key = json.dumps(
        {
            "protocol": str(protocol),
            "family": str(feature_family),
            "condition_name": str(condition_name),
            "objective": str(objective),
            "seed": int(seed),
            "adapter_rank": int(adapter_rank),
            "alpha": float(alpha),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    model_summaries[model_key] = _back_project_weights(
        basis,
        model.effective_latent_weight(),
        model.head_bias_value(),
        tuple(source_train.matrix.feature_names),
    )

    rows: list[dict[str, Any]] = []
    proxy_rows_by_dataset: dict[str, dict[str | int, dict[str, Any]]] = {}
    if reference_model is not None:
        if proxy_reference_name is None:
            raise ValueError("proxy_reference_name is required when reference_model is provided")
        if source_cov is None:
            raise ValueError("source_cov is required when reference_model is provided")
        for dataset_run in (source_train, source_holdout, target_ds_test, target_qwen_test):
            ref_scores = _score_model_on_tensor(reference_model, dataset_run.z_tensor)
            cur_scores = _score_model_on_tensor(model, dataset_run.z_tensor)
            ref_adapted = _adapted_latent_tensor(reference_model, dataset_run.z_tensor)
            cur_adapted = _adapted_latent_tensor(model, dataset_run.z_tensor)
            proxy_rows_by_dataset[str(dataset_run.dataset_role)] = _proxy_rows_for_dataset(
                reference_scores=ref_scores,
                current_scores=cur_scores,
                matrix=dataset_run.matrix,
                proxy_reference=str(proxy_reference_name),
                source_cov=np.asarray(source_cov, dtype=np.float64),
                reference_adapted=ref_adapted,
                current_adapted=cur_adapted,
                adapter_residual_norm=float(model.adapter_residual_norm()),
            )

    for dataset_run in (source_train, source_holdout, target_ds_test, target_qwen_test):
        anchor_rows = _evaluate_model_on_matrix(
            model=model,
            z_tensor=dataset_run.z_tensor,
            matrix=dataset_run.matrix,
            labels_available=bool(dataset_run.labels_available),
        )
        anchor_rows.append(_aggregate_anchor_rows(anchor_rows))
        for metric_row in anchor_rows:
            proxy_row = proxy_rows_by_dataset.get(str(dataset_run.dataset_role), {}).get(metric_row["anchor_pct"], {})
            rows.append({
                "timestamp_utc": _now_utc(),
                "protocol": str(protocol),
                "used_target_unlabeled": int(str(protocol) == "transductive_target_unlabeled"),
                "feature_family": str(feature_family),
                "representation": "raw+rank",
                "training_anchor_scope": "shared_70_100",
                "dataset_role": str(dataset_run.dataset_role),
                "labels_available": int(bool(dataset_run.labels_available)),
                "condition_name": str(condition_name),
                "objective": str(objective),
                "adapter_mode": str(adapter_mode),
                "seed": int(seed),
                "basis_rank": int(basis["rank"]),
                "basis_whiten": bool(basis["whiten"]),
                "basis_random_state": int(DEFAULT_BASIS_RANDOM_STATE),
                "adapter_rank": int(adapter_rank),
                "alpha": float(alpha),
                "adapter_init_std": float(train_cfg.adapter_init_std),
                "source_epochs": int(train_cfg.source_epochs),
                "transductive_epochs": int(train_cfg.transductive_epochs),
                "learning_rate": float(train_cfg.learning_rate),
                "weight_decay": float(train_cfg.weight_decay),
                "pseudo_pair_margin": float(train_cfg.pseudo_pair_margin),
                "pseudo_pair_max_per_group": int(train_cfg.pseudo_pair_max_per_group),
                "pseudo_pair_weight": float(train_cfg.pseudo_pair_weight),
                "anchor_consistency_weight": float(train_cfg.anchor_consistency_weight),
                "covariance_alignment_weight": float(train_cfg.covariance_alignment_weight),
                "pseudo_start_epoch": int(train_cfg.pseudo_start_epoch),
                "anchor_pct": metric_row["anchor_pct"],
                "auroc": metric_row["auroc"],
                "rank": metric_row["rank"],
                "hit@1": metric_row["hit@1"],
                "hit@3": metric_row["hit@3"],
                "selacc@10%": metric_row["selacc@10%"],
                "pairwise": metric_row["pairwise"],
                "n_problems": metric_row["n_problems"],
                "n_samples": metric_row["n_samples"],
                "top10_count": metric_row["top10_count"],
                "proxy_reference": proxy_row.get("proxy_reference"),
                "proxy_score_corr": proxy_row.get("proxy_score_corr"),
                "proxy_top1_agreement": proxy_row.get("proxy_top1_agreement"),
                "proxy_top1_flip_rate": proxy_row.get("proxy_top1_flip_rate"),
                "proxy_mean_abs_score_delta": proxy_row.get("proxy_mean_abs_score_delta"),
                "proxy_anchor_gap_before": proxy_row.get("proxy_anchor_gap_before"),
                "proxy_anchor_gap_after": proxy_row.get("proxy_anchor_gap_after"),
                "proxy_anchor_gap_delta": proxy_row.get("proxy_anchor_gap_delta"),
                "proxy_cov_gap_before": proxy_row.get("proxy_cov_gap_before"),
                "proxy_cov_gap_after": proxy_row.get("proxy_cov_gap_after"),
                "proxy_cov_gap_delta": proxy_row.get("proxy_cov_gap_delta"),
                "adapter_residual_norm": proxy_row.get("adapter_residual_norm", float(model.adapter_residual_norm())),
                "model_key": model_key,
            })
    return rows, model


def _write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Refusing to write an empty CSV")
    fieldnames = [
        "timestamp_utc",
        "protocol",
        "used_target_unlabeled",
        "feature_family",
        "representation",
        "training_anchor_scope",
        "dataset_role",
        "labels_available",
        "condition_name",
        "objective",
        "adapter_mode",
        "seed",
        "basis_rank",
        "basis_whiten",
        "basis_random_state",
        "adapter_rank",
        "alpha",
        "adapter_init_std",
        "source_epochs",
        "transductive_epochs",
        "learning_rate",
        "weight_decay",
        "pseudo_pair_margin",
        "pseudo_pair_max_per_group",
        "pseudo_pair_weight",
        "anchor_consistency_weight",
        "covariance_alignment_weight",
        "pseudo_start_epoch",
        "anchor_pct",
        "auroc",
        "rank",
        "hit@1",
        "hit@3",
        "selacc@10%",
        "pairwise",
        "n_problems",
        "n_samples",
        "top10_count",
        "proxy_reference",
        "proxy_score_corr",
        "proxy_top1_agreement",
        "proxy_top1_flip_rate",
        "proxy_mean_abs_score_delta",
        "proxy_anchor_gap_before",
        "proxy_anchor_gap_after",
        "proxy_anchor_gap_delta",
        "proxy_cov_gap_before",
        "proxy_cov_gap_after",
        "proxy_cov_gap_delta",
        "adapter_residual_norm",
        "model_key",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    ap = argparse.ArgumentParser(description="Train coding random low-rank adapter SSL experiment")
    ap.add_argument("--source-main-root", default=str(DEFAULT_SOURCE_MAIN_ROOT))
    ap.add_argument("--target-test-root", default=str(DEFAULT_TARGET_TEST_ROOT))
    ap.add_argument("--feature-cache-dir", default=str(DEFAULT_FEATURE_CACHE_DIR))
    ap.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV))
    ap.add_argument("--out-doc", default=str(DEFAULT_OUT_DOC))
    ap.add_argument("--feature-workers", type=int, default=8)
    ap.add_argument("--feature-chunk-problems", type=int, default=DEFAULT_FEATURE_CHUNK_PROBLEMS)
    ap.add_argument("--holdout-split", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--families", default="canonical_22,token_only")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--adapter-ranks", default="2,4,8")
    ap.add_argument("--alphas", default="0.1,0.3,1.0")
    ap.add_argument("--basis-rank", type=int, default=DEFAULT_BASIS_RANK)
    ap.add_argument("--basis-whiten", action="store_true", default=DEFAULT_BASIS_WHITEN)
    ap.add_argument("--source-epochs", type=int, default=DEFAULT_SOURCE_EPOCHS)
    ap.add_argument("--transductive-epochs", type=int, default=DEFAULT_TRANSDUCTIVE_EPOCHS)
    ap.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    ap.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    ap.add_argument("--adapter-init-std", type=float, default=DEFAULT_ADAPTER_INIT_STD)
    ap.add_argument("--pseudo-pair-margin", type=float, default=DEFAULT_PSEUDO_PAIR_MARGIN)
    ap.add_argument("--pseudo-pair-max-per-group", type=int, default=DEFAULT_PSEUDO_PAIR_MAX_PER_GROUP)
    ap.add_argument("--pseudo-pair-weight", type=float, default=DEFAULT_PSEUDO_PAIR_WEIGHT)
    ap.add_argument("--anchor-consistency-weight", type=float, default=DEFAULT_ANCHOR_CONSISTENCY_WEIGHT)
    ap.add_argument("--covariance-alignment-weight", type=float, default=DEFAULT_COVARIANCE_ALIGNMENT_WEIGHT)
    ap.add_argument("--pseudo-start-epoch", type=int, default=DEFAULT_PSEUDO_START_EPOCH)
    ap.add_argument("--max-problems-per-cache", type=int, default=0)
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    require_torch()
    torch.set_num_threads(max(1, int(args.torch_threads)))
    np.random.seed(0)
    torch.manual_seed(0)

    feature_cache_dir = Path(args.feature_cache_dir)
    out_csv = Path(args.out_csv)
    out_doc = Path(args.out_doc)

    families = tuple(str(chunk.strip()) for chunk in str(args.families).split(",") if chunk.strip())
    seeds = _parse_int_csv(args.seeds)
    adapter_ranks = _parse_int_csv(args.adapter_ranks)
    alphas = _parse_float_csv(args.alphas)
    max_problems_per_cache = None if int(args.max_problems_per_cache) <= 0 else int(args.max_problems_per_cache)

    if bool(args.smoke):
        seeds = SMOKE_SEEDS
        adapter_ranks = SMOKE_ADAPTER_RANKS
        alphas = SMOKE_ALPHAS
        max_problems_per_cache = SMOKE_MAX_PROBLEMS
        args.source_epochs = SMOKE_SOURCE_EPOCHS
        args.transductive_epochs = SMOKE_TRANSDUCTIVE_EPOCHS

    for family_name in families:
        _family_feature_names(family_name)

    required_feature_names = set(FIXED_FEATURE_NAMES)

    source_store, source_cache_path, source_cache_status = _load_or_build_feature_store(
        source_name="source_main",
        base_root=args.source_main_root,
        include_cache_keys=(SOURCE_CACHE_KEY,),
        positions=LATE_ANCHORS,
        required_feature_names=required_feature_names,
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_feature_cache),
        max_problems_per_cache=max_problems_per_cache,
        feature_workers=int(args.feature_workers),
        chunk_problems=int(args.feature_chunk_problems),
    )
    target_ds_store, target_ds_cache_path, target_ds_cache_status = _load_or_build_feature_store(
        source_name="target_ds_test",
        base_root=args.target_test_root,
        include_cache_keys=(TARGET_DS_CACHE_KEY,),
        positions=LATE_ANCHORS,
        required_feature_names=required_feature_names,
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_feature_cache),
        max_problems_per_cache=max_problems_per_cache,
        feature_workers=int(args.feature_workers),
        chunk_problems=int(args.feature_chunk_problems),
    )
    target_qwen_store, target_qwen_cache_path, target_qwen_cache_status = _load_or_build_feature_store(
        source_name="target_qwen_test",
        base_root=args.target_test_root,
        include_cache_keys=(TARGET_QWEN_CACHE_KEY,),
        positions=LATE_ANCHORS,
        required_feature_names=required_feature_names,
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_feature_cache),
        max_problems_per_cache=max_problems_per_cache,
        feature_workers=int(args.feature_workers),
        chunk_problems=int(args.feature_chunk_problems),
    )

    source_store = _select_single_coding_payload(source_store, split_name="source_main")
    target_ds_store = _select_single_coding_payload(target_ds_store, split_name="target_ds_test")
    target_qwen_store = _select_single_coding_payload(target_qwen_store, split_name="target_qwen_test")

    source_cache_root = _resolve_cache_root(args.source_main_root, SOURCE_CACHE_KEY)
    target_ds_cache_root = _resolve_cache_root(args.target_test_root, TARGET_DS_CACHE_KEY)
    target_qwen_cache_root = _resolve_cache_root(args.target_test_root, TARGET_QWEN_CACHE_KEY)

    source_labels_available = _has_ground_truth(source_cache_root)
    target_ds_labels_available = _has_ground_truth(target_ds_cache_root)
    target_qwen_labels_available = _has_ground_truth(target_qwen_cache_root)

    holdout_map, holdout_summary = _build_holdout_problem_map(
        source_store,
        holdout_split=float(args.holdout_split),
        split_seed=int(args.split_seed),
    )
    source_train_store, source_holdout_store, _source_full_store = _split_feature_store(
        source_store,
        holdout_problem_map=holdout_map,
    )
    if not source_train_store or not source_holdout_store:
        raise ValueError(
            f"Source split failed: train={len(source_train_store)} holdout={len(source_holdout_store)} "
            f"summary={holdout_summary}"
        )

    train_cfg = TrainConfig(
        source_epochs=int(args.source_epochs),
        transductive_epochs=int(args.transductive_epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        adapter_init_std=float(args.adapter_init_std),
        pseudo_pair_margin=float(args.pseudo_pair_margin),
        pseudo_pair_max_per_group=int(args.pseudo_pair_max_per_group),
        pseudo_pair_weight=float(args.pseudo_pair_weight),
        anchor_consistency_weight=float(args.anchor_consistency_weight),
        covariance_alignment_weight=float(args.covariance_alignment_weight),
        pseudo_start_epoch=int(args.pseudo_start_epoch),
        device=str(args.device),
    )

    rows: list[dict[str, Any]] = []
    model_summaries: dict[str, dict[str, Any]] = {}
    for family_name in families:
        print(f"[coding-random-adapter] family={family_name}", flush=True)
        source_train_matrix = _build_family_matrix(source_train_store, family_name=family_name, anchors=LATE_ANCHORS)
        source_holdout_matrix = _build_family_matrix(source_holdout_store, family_name=family_name, anchors=LATE_ANCHORS)
        target_ds_matrix = _build_family_matrix(target_ds_store, family_name=family_name, anchors=LATE_ANCHORS)
        target_qwen_matrix = _build_family_matrix(target_qwen_store, family_name=family_name, anchors=LATE_ANCHORS)

        basis = _fit_basis_from_matrix(
            source_train_matrix,
            basis_rank=int(args.basis_rank),
            basis_whiten=bool(args.basis_whiten),
            basis_random_state=DEFAULT_BASIS_RANDOM_STATE,
        )

        source_train_run = DatasetRun(
            dataset_role="source_train_eval",
            matrix=source_train_matrix,
            z_tensor=_transform_matrix_to_latent(source_train_matrix, basis),
            labels_available=bool(source_labels_available),
        )
        source_holdout_run = DatasetRun(
            dataset_role="source_holdout",
            matrix=source_holdout_matrix,
            z_tensor=_transform_matrix_to_latent(source_holdout_matrix, basis),
            labels_available=bool(source_labels_available),
        )
        target_ds_run = DatasetRun(
            dataset_role="target_ds_test",
            matrix=target_ds_matrix,
            z_tensor=_transform_matrix_to_latent(target_ds_matrix, basis),
            labels_available=bool(target_ds_labels_available),
        )
        target_qwen_run = DatasetRun(
            dataset_role="target_qwen_test",
            matrix=target_qwen_matrix,
            z_tensor=_transform_matrix_to_latent(target_qwen_matrix, basis),
            labels_available=bool(target_qwen_labels_available),
        )
        source_cov = _numpy_covariance(source_train_run.z_tensor)

        for seed in seeds:
            print(f"[coding-random-adapter] family={family_name} seed={seed}", flush=True)

            frozen_point_rows, frozen_point_model = _run_condition(
                protocol="source_only_inductive",
                feature_family=family_name,
                objective="pointwise",
                condition_name="frozen_basis_pointwise",
                adapter_mode="none",
                adapter_rank=0,
                alpha=1.0,
                seed=int(seed),
                basis=basis,
                source_train=source_train_run,
                source_holdout=source_holdout_run,
                target_ds_test=target_ds_run,
                target_qwen_test=target_qwen_run,
                train_cfg=train_cfg,
                model_summaries=model_summaries,
            )
            rows.extend(frozen_point_rows)
            frozen_pair_rows, frozen_pair_model = _run_condition(
                protocol="source_only_inductive",
                feature_family=family_name,
                objective="pairwise",
                condition_name="frozen_basis_pairwise",
                adapter_mode="none",
                adapter_rank=0,
                alpha=1.0,
                seed=int(seed),
                basis=basis,
                source_train=source_train_run,
                source_holdout=source_holdout_run,
                target_ds_test=target_ds_run,
                target_qwen_test=target_qwen_run,
                train_cfg=train_cfg,
                model_summaries=model_summaries,
            )
            rows.extend(frozen_pair_rows)

            for adapter_rank in adapter_ranks:
                for alpha in alphas:
                    random_point_rows, _random_point_model = _run_condition(
                        protocol="source_only_inductive",
                        feature_family=family_name,
                        objective="pointwise",
                        condition_name="random_frozen_adapter_pointwise",
                        adapter_mode="random_frozen",
                        adapter_rank=int(adapter_rank),
                        alpha=float(alpha),
                        seed=int(seed),
                        basis=basis,
                        source_train=source_train_run,
                        source_holdout=source_holdout_run,
                        target_ds_test=target_ds_run,
                        target_qwen_test=target_qwen_run,
                        train_cfg=train_cfg,
                        model_summaries=model_summaries,
                        reference_model=frozen_point_model,
                        proxy_reference_name="frozen_basis_pointwise",
                        source_cov=source_cov,
                    )
                    rows.extend(random_point_rows)
                    random_pair_rows, _random_pair_model = _run_condition(
                        protocol="source_only_inductive",
                        feature_family=family_name,
                        objective="pairwise",
                        condition_name="random_frozen_adapter_pairwise",
                        adapter_mode="random_frozen",
                        adapter_rank=int(adapter_rank),
                        alpha=float(alpha),
                        seed=int(seed),
                        basis=basis,
                        source_train=source_train_run,
                        source_holdout=source_holdout_run,
                        target_ds_test=target_ds_run,
                        target_qwen_test=target_qwen_run,
                        train_cfg=train_cfg,
                        model_summaries=model_summaries,
                        reference_model=frozen_pair_model,
                        proxy_reference_name="frozen_basis_pairwise",
                        source_cov=source_cov,
                    )
                    rows.extend(random_pair_rows)
                    trained_point_rows, trained_point_model = _run_condition(
                        protocol="source_only_inductive",
                        feature_family=family_name,
                        objective="pointwise",
                        condition_name="trained_adapter_pointwise",
                        adapter_mode="trained",
                        adapter_rank=int(adapter_rank),
                        alpha=float(alpha),
                        seed=int(seed),
                        basis=basis,
                        source_train=source_train_run,
                        source_holdout=source_holdout_run,
                        target_ds_test=target_ds_run,
                        target_qwen_test=target_qwen_run,
                        train_cfg=train_cfg,
                        model_summaries=model_summaries,
                        reference_model=frozen_point_model,
                        proxy_reference_name="frozen_basis_pointwise",
                        source_cov=source_cov,
                    )
                    rows.extend(trained_point_rows)
                    trained_pair_rows, trained_pair_model = _run_condition(
                        protocol="source_only_inductive",
                        feature_family=family_name,
                        objective="pairwise",
                        condition_name="trained_adapter_pairwise",
                        adapter_mode="trained",
                        adapter_rank=int(adapter_rank),
                        alpha=float(alpha),
                        seed=int(seed),
                        basis=basis,
                        source_train=source_train_run,
                        source_holdout=source_holdout_run,
                        target_ds_test=target_ds_run,
                        target_qwen_test=target_qwen_run,
                        train_cfg=train_cfg,
                        model_summaries=model_summaries,
                        reference_model=frozen_pair_model,
                        proxy_reference_name="frozen_basis_pairwise",
                        source_cov=source_cov,
                    )
                    rows.extend(trained_pair_rows)
                    transductive_point_rows, _transductive_point_model = _run_condition(
                        protocol="transductive_target_unlabeled",
                        feature_family=family_name,
                        objective="pointwise",
                        condition_name="trained_adapter_pointwise",
                        adapter_mode="trained",
                        adapter_rank=int(adapter_rank),
                        alpha=float(alpha),
                        seed=int(seed),
                        basis=basis,
                        source_train=source_train_run,
                        source_holdout=source_holdout_run,
                        target_ds_test=target_ds_run,
                        target_qwen_test=target_qwen_run,
                        train_cfg=train_cfg,
                        model_summaries=model_summaries,
                        reference_model=trained_point_model,
                        proxy_reference_name="matched_source_only_trained_adapter_pointwise",
                        source_cov=source_cov,
                    )
                    rows.extend(transductive_point_rows)
                    transductive_pair_rows, _transductive_pair_model = _run_condition(
                        protocol="transductive_target_unlabeled",
                        feature_family=family_name,
                        objective="pairwise",
                        condition_name="trained_adapter_pairwise",
                        adapter_mode="trained",
                        adapter_rank=int(adapter_rank),
                        alpha=float(alpha),
                        seed=int(seed),
                        basis=basis,
                        source_train=source_train_run,
                        source_holdout=source_holdout_run,
                        target_ds_test=target_ds_run,
                        target_qwen_test=target_qwen_run,
                        train_cfg=train_cfg,
                        model_summaries=model_summaries,
                        reference_model=trained_pair_model,
                        proxy_reference_name="matched_source_only_trained_adapter_pairwise",
                        source_cov=source_cov,
                    )
                    rows.extend(transductive_pair_rows)

    _write_csv(rows, out_csv)
    selected_configs = _select_summary_configs(rows)
    for key, selected in selected_configs.items():
        representative_model_key = _representative_model_key(rows, selected)
        if representative_model_key is not None:
            selected["model_key"] = str(representative_model_key)
        elif selected.get("model_key") is not None:
            selected["model_key"] = str(selected["model_key"])
    _build_doc(
        out_path=out_doc,
        rows=rows,
        selected_configs=selected_configs,
        model_summaries=model_summaries,
        args=args,
    )

    print(
        json.dumps(
            {
                "rows_written": len(rows),
                "out_csv": str(out_csv),
                "out_doc": str(out_doc),
                "feature_cache": {
                    "source_main": {"path": str(source_cache_path), "status": source_cache_status},
                    "target_ds_test": {"path": str(target_ds_cache_path), "status": target_ds_cache_status},
                    "target_qwen_test": {"path": str(target_qwen_cache_path), "status": target_qwen_cache_status},
                },
                "labels_available": {
                    "source_main": bool(source_labels_available),
                    "target_ds_test": bool(target_ds_labels_available),
                    "target_qwen_test": bool(target_qwen_labels_available),
                },
                "holdout_summary": holdout_summary,
                "smoke": bool(args.smoke),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
