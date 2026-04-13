#!/usr/bin/env python3
"""Three-layer SVD vs no-SVD feature-complexity study for EarlyStop.

This study answers a paper-facing question:

    If no-SVD is already close to SVD on the current canonical route,
    is that because the feature bank is unusually clean?

The protocol is intentionally structured in three layers:

1) canonical curated bundles
   trajectory_only -> uncertainty_only -> token_only -> token_plus_trajectory -> canonical_22

2) real upstream expansion
   canonical_22 -> +neuron_adjacent -> +prefix_tail -> +event_local

3) noise / decoy control
   start from the widest real bank (30 features), then add
   permutation / duplicate-noisy / random-control decoys at low/med/high doses

Outputs:
  - results/scans/feature_complexity/three_layer_summary.json
  - results/tables/feature_complexity_model_rows.csv
  - results/tables/feature_complexity_comparison.csv
  - results/tables/feature_complexity_clean_sweep.csv
  - results/tables/feature_complexity_noise_robustness.csv
  - results/figures/feature_complexity/*.png
  - docs/18_SVD_FEATURE_COMPLEXITY_RESULTS.md
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.ops.earlystop import EARLY_STOP_POSITIONS
from nad.ops.earlystop_svd import (
    AVAILABILITY_FEATURES,
    LEGACY_FULL_FEATURE_NAMES,
    META_FEATURES,
    PREFIX_LOCAL_FEATURES,
    TRAJ_FEATURES,
    TOKEN_FEATURES,
    _auroc,
    _build_representation,
    _predict_svd_lr,
    _rank_transform_matrix,
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
    evaluate_method_from_feature_store,
)
from SVDomain.train_es_svd_ms_rr_r1 import (
    _build_holdout_problem_map,
    _load_or_build_qualified_feature_store,
    _resolve_path,
    _split_feature_store,
    _summarise_feature_store,
)


DOMAIN_ORDER = ("math", "science", "coding")
LAYER_ORDER = ("canonical", "expansion", "noise")
FEATURE_TO_INDEX = {str(name): idx for idx, name in enumerate(LEGACY_FULL_FEATURE_NAMES)}

UNCERTAINTY_ONLY_FEATURES = (
    "tok_gini_prefix",
    "tok_gini_tail",
    "tok_gini_slope",
    "tok_neg_entropy_prefix",
    "tok_neg_entropy_recency",
    "tok_selfcert_prefix",
    "tok_selfcert_recency",
)
CANONICAL_22_FEATURES = tuple(list(TOKEN_FEATURES) + list(TRAJ_FEATURES) + list(AVAILABILITY_FEATURES))
NEURON_ADJACENT_FEATURES = tuple(list(META_FEATURES))
PREFIX_TAIL_FEATURES = ("tail_q10", "head_tail_gap", "tail_variance")
EVENT_LOCAL_FEATURES = ("last_event_tail_conf", "event_pre_post_delta")

NOISE_LEVELS = {
    "low": {"ratio": 0.5, "noise_scale": 0.05},
    "med": {"ratio": 1.0, "noise_scale": 0.10},
    "high": {"ratio": 2.0, "noise_scale": 0.20},
}
CONDITION_ORDER = [
    "trajectory_only",
    "uncertainty_only",
    "token_only",
    "token_plus_trajectory",
    "canonical_22",
    "canonical_plus_neuron_adjacent",
    "canonical_plus_prefix_tail",
    "canonical_plus_event_local",
    "perm_low",
    "perm_med",
    "perm_high",
    "duplicate_low",
    "duplicate_med",
    "duplicate_high",
    "random_low",
    "random_med",
    "random_high",
]


@dataclass(frozen=True)
class ConditionSpec:
    layer: str
    condition_id: str
    label: str
    description: str
    feature_names: tuple[str, ...]
    decoy_family: Optional[str] = None
    decoy_level: Optional[str] = None
    decoy_ratio: float = 0.0
    decoy_noise_scale: float = 0.0

    @property
    def base_feature_count(self) -> int:
        return int(len(self.feature_names))


def default_feature_workers() -> int:
    cpu = max(1, int((__import__("os").cpu_count() or 1)))
    return max(1, min(8, cpu))


def default_fit_workers() -> int:
    cpu = max(1, int((__import__("os").cpu_count() or 1)))
    return max(1, min(4, cpu))


def _parse_csv_ints(raw: str) -> list[int]:
    values: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Need at least one integer")
    return values


def _parse_csv_strings(raw: str) -> list[str]:
    values = [token.strip() for token in str(raw).split(",") if token.strip()]
    if not values:
        raise ValueError("Need at least one token")
    return values


def _hash_seed(*parts: Any) -> int:
    raw = "::".join(str(part) for part in parts)
    return int(hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8], 16)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    value_f = float(value)
    if not np.isfinite(value_f):
        return None
    return value_f


def _fmt_pct(value: Optional[float]) -> str:
    if value is None or not np.isfinite(float(value)):
        return "N/A"
    return f"{float(value) * 100.0:.2f}%"


def _condition_specs() -> list[ConditionSpec]:
    specs: list[ConditionSpec] = [
        ConditionSpec(
            layer="canonical",
            condition_id="trajectory_only",
            label="L1 trajectory-only",
            description="Only the 5 trajectory-shape features.",
            feature_names=tuple(TRAJ_FEATURES),
        ),
        ConditionSpec(
            layer="canonical",
            condition_id="uncertainty_only",
            label="L1 uncertainty-only",
            description="Token uncertainty proxies: gini, neg-entropy, self-certainty.",
            feature_names=UNCERTAINTY_ONLY_FEATURES,
        ),
        ConditionSpec(
            layer="canonical",
            condition_id="token_only",
            label="L1 token-only",
            description="All 11 token-level features, no trajectory, no availability flags.",
            feature_names=tuple(TOKEN_FEATURES),
        ),
        ConditionSpec(
            layer="canonical",
            condition_id="token_plus_trajectory",
            label="L1 token + trajectory",
            description="11 token features + 5 trajectory features, still excluding availability flags.",
            feature_names=tuple(list(TOKEN_FEATURES) + list(TRAJ_FEATURES)),
        ),
        ConditionSpec(
            layer="canonical",
            condition_id="canonical_22",
            label="L1 canonical 22",
            description="Paper-facing canonical 22-feature bank = token + trajectory + availability.",
            feature_names=CANONICAL_22_FEATURES,
        ),
        ConditionSpec(
            layer="expansion",
            condition_id="canonical_plus_neuron_adjacent",
            label="L2 canonical + neuron-adjacent",
            description="Canonical 22 + nc_mean + nc_slope + self_similarity.",
            feature_names=tuple(list(CANONICAL_22_FEATURES) + list(NEURON_ADJACENT_FEATURES)),
        ),
        ConditionSpec(
            layer="expansion",
            condition_id="canonical_plus_prefix_tail",
            label="L2 canonical + prefix-tail",
            description="Previous line + tail_q10 + head_tail_gap + tail_variance.",
            feature_names=tuple(list(CANONICAL_22_FEATURES) + list(NEURON_ADJACENT_FEATURES) + list(PREFIX_TAIL_FEATURES)),
        ),
        ConditionSpec(
            layer="expansion",
            condition_id="canonical_plus_event_local",
            label="L2 canonical + event-local",
            description="Widest real upstream bank: canonical + neuron-adjacent + prefix-tail + event-local (30 features).",
            feature_names=tuple(
                list(CANONICAL_22_FEATURES)
                + list(NEURON_ADJACENT_FEATURES)
                + list(PREFIX_TAIL_FEATURES)
                + list(EVENT_LOCAL_FEATURES)
            ),
        ),
    ]
    widest_real = tuple(
        list(CANONICAL_22_FEATURES)
        + list(NEURON_ADJACENT_FEATURES)
        + list(PREFIX_TAIL_FEATURES)
        + list(EVENT_LOCAL_FEATURES)
    )
    family_id_map = {
        "permutation": "perm",
        "duplicate": "duplicate",
        "random": "random",
    }
    for decoy_family in ("permutation", "duplicate", "random"):
        for level_name, cfg in NOISE_LEVELS.items():
            specs.append(
                ConditionSpec(
                    layer="noise",
                    condition_id=f"{family_id_map[decoy_family]}_{level_name}",
                    label=f"L3 {decoy_family} {level_name}",
                    description=(
                        f"Widest real bank (30 features) + {decoy_family} decoys at {level_name} dose "
                        f"(ratio={cfg['ratio']:.1f})."
                    ),
                    feature_names=widest_real,
                    decoy_family=decoy_family,
                    decoy_level=level_name,
                    decoy_ratio=float(cfg["ratio"]),
                    decoy_noise_scale=float(cfg["noise_scale"]),
                )
            )
    return specs


def _condition_index(condition_id: str) -> int:
    try:
        return CONDITION_ORDER.index(str(condition_id))
    except ValueError:
        return len(CONDITION_ORDER)


def _limit_payload_problems(payload: dict[str, Any], max_problems: int) -> Optional[dict[str, Any]]:
    limit = max(0, int(max_problems))
    if limit <= 0:
        return dict(payload)
    problem_ids = [str(v) for v in payload["problem_ids"]][:limit]
    if not problem_ids:
        return None
    offsets = [int(v) for v in payload["problem_offsets"]]
    width = offsets[len(problem_ids)]
    out = dict(payload)
    out["problem_ids"] = problem_ids
    out["problem_offsets"] = [int(v) for v in offsets[: len(problem_ids) + 1]]
    out["tensor"] = np.asarray(payload["tensor"][:width], dtype=np.float64)
    out["labels"] = np.asarray(payload["labels"][:width], dtype=np.int32)
    out["sample_ids"] = np.asarray(payload["sample_ids"][:width], dtype=np.int32)
    out["group_keys"] = np.asarray(payload["group_keys"][:width], dtype=object)
    if "cv_group_keys" in payload:
        out["cv_group_keys"] = np.asarray(payload["cv_group_keys"][:width], dtype=object)
    out["samples"] = int(width)
    return out


def _limit_feature_store(feature_store: list[dict[str, Any]], max_problems: int) -> list[dict[str, Any]]:
    limited: list[dict[str, Any]] = []
    for payload in feature_store:
        item = _limit_payload_problems(payload, max_problems=int(max_problems))
        if item is not None and int(item["samples"]) > 0:
            limited.append(item)
    return limited


def _load_prebuilt_store(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    return list(payload["feature_store"])


def _load_full_store(
    *,
    main_store_path: Optional[str],
    extra_store_path: Optional[str],
    rebuild_store: bool,
    main_cache_root: str,
    extra_cache_root: Optional[str],
    feature_cache_dir: Optional[Path],
    feature_workers: int,
    feature_chunk_problems: int,
    max_problems_per_cache: Optional[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    required_features = set(str(v) for v in LEGACY_FULL_FEATURE_NAMES)
    meta: dict[str, Any] = {
        "main_store_source": None,
        "extra_store_source": None,
        "main_cache_path": None,
        "extra_cache_path": None,
    }

    if not rebuild_store and main_store_path:
        main_store = _load_prebuilt_store(main_store_path)
        meta["main_store_source"] = str(main_store_path)
    else:
        main_store, main_cache_path, _ = _load_or_build_qualified_feature_store(
            source_name="cache",
            cache_root=str(main_cache_root),
            positions=tuple(float(v) for v in EXTRACTION_POSITIONS),
            required_feature_names=required_features,
            max_problems_per_cache=max_problems_per_cache,
            feature_workers=int(feature_workers),
            chunk_problems=int(feature_chunk_problems),
            feature_cache_dir=feature_cache_dir,
            refresh_feature_cache=False,
        )
        meta["main_store_source"] = "rebuilt_from_cache_root"
        meta["main_cache_path"] = None if main_cache_path is None else str(main_cache_path)

    extra_store: list[dict[str, Any]] = []
    if extra_store_path and str(extra_store_path).strip().lower() not in {"", "none", "off"} and not rebuild_store:
        extra_store = _load_prebuilt_store(extra_store_path)
        meta["extra_store_source"] = str(extra_store_path)
    elif rebuild_store and extra_cache_root:
        extra_store, extra_cache_path, _ = _load_or_build_qualified_feature_store(
            source_name="cache_train",
            cache_root=str(extra_cache_root),
            positions=tuple(float(v) for v in EXTRACTION_POSITIONS),
            required_feature_names=required_features,
            max_problems_per_cache=max_problems_per_cache,
            feature_workers=int(feature_workers),
            chunk_problems=int(feature_chunk_problems),
            feature_cache_dir=feature_cache_dir,
            refresh_feature_cache=False,
        )
        meta["extra_store_source"] = "rebuilt_from_extra_cache_root"
        meta["extra_cache_path"] = None if extra_cache_path is None else str(extra_cache_path)
    else:
        meta["extra_store_source"] = "disabled"

    return list(main_store) + list(extra_store), meta


def _transform_payload(
    payload: dict[str, Any],
    *,
    spec: ConditionSpec,
    decoy_seed: int,
) -> tuple[dict[str, Any], list[str]]:
    feature_indices = [FEATURE_TO_INDEX[str(name)] for name in spec.feature_names]
    base_tensor = np.asarray(payload["tensor"][:, :, feature_indices], dtype=np.float64)
    feature_names = list(spec.feature_names)
    if spec.decoy_family is None:
        out = dict(payload)
        out["tensor"] = base_tensor
        return out, feature_names

    base_count = int(base_tensor.shape[2])
    extra_count = max(1, int(round(base_count * float(spec.decoy_ratio))))
    rng = np.random.default_rng(_hash_seed(decoy_seed, spec.condition_id, payload["cache_key"], payload["source_name"]))
    decoy_tensor = np.zeros((base_tensor.shape[0], base_tensor.shape[1], extra_count), dtype=np.float64)
    decoy_names: list[str] = []

    problem_offsets = [int(v) for v in payload["problem_offsets"]]
    for extra_idx in range(extra_count):
        src_idx = int(extra_idx % base_count)
        src_name = str(feature_names[src_idx])
        decoy_names.append(f"{spec.decoy_family}_{spec.decoy_level}_{extra_idx:03d}__{src_name}")
        for pos_idx in range(base_tensor.shape[1]):
            src_col = np.asarray(base_tensor[:, pos_idx, src_idx], dtype=np.float64)
            if spec.decoy_family == "permutation":
                col = np.zeros_like(src_col)
                for problem_idx in range(len(problem_offsets) - 1):
                    start = problem_offsets[problem_idx]
                    end = problem_offsets[problem_idx + 1]
                    if end <= start:
                        continue
                    values = np.asarray(src_col[start:end], dtype=np.float64)
                    if values.shape[0] <= 1:
                        col[start:end] = values
                    else:
                        perm = rng.permutation(values.shape[0])
                        col[start:end] = values[perm]
                decoy_tensor[:, pos_idx, extra_idx] = col
            elif spec.decoy_family == "duplicate":
                sigma = max(float(np.std(src_col)), 1e-6)
                noise = rng.normal(loc=0.0, scale=sigma * float(spec.decoy_noise_scale), size=src_col.shape[0])
                decoy_tensor[:, pos_idx, extra_idx] = src_col + noise
            elif spec.decoy_family == "random":
                mean = float(np.mean(src_col))
                sigma = max(float(np.std(src_col)), 1e-6)
                decoy_tensor[:, pos_idx, extra_idx] = rng.normal(loc=mean, scale=sigma, size=src_col.shape[0])
            else:
                raise ValueError(f"Unknown decoy family: {spec.decoy_family}")

    out = dict(payload)
    out["tensor"] = np.concatenate([base_tensor, decoy_tensor], axis=2)
    return out, feature_names + decoy_names


def _transform_feature_store(
    feature_store: list[dict[str, Any]],
    *,
    spec: ConditionSpec,
    decoy_seed: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    transformed: list[dict[str, Any]] = []
    feature_names: Optional[list[str]] = None
    for payload in feature_store:
        item, item_feature_names = _transform_payload(payload, spec=spec, decoy_seed=int(decoy_seed))
        transformed.append(item)
        if feature_names is None:
            feature_names = list(item_feature_names)
    return transformed, list(feature_names or list(spec.feature_names))


def _build_training_tables(
    feature_store: list[dict[str, Any]],
    positions: tuple[float, ...],
    n_features: int,
) -> list[dict[str, np.ndarray]]:
    rows: dict[int, list[np.ndarray]] = {idx: [] for idx in range(len(positions))}
    labels: dict[int, list[np.ndarray]] = {idx: [] for idx in range(len(positions))}
    rank_groups: dict[int, list[np.ndarray]] = {idx: [] for idx in range(len(positions))}
    cv_groups: dict[int, list[np.ndarray]] = {idx: [] for idx in range(len(positions))}

    position_to_idx = {float(v): idx for idx, v in enumerate(EXTRACTION_POSITIONS)}
    source_indices = [position_to_idx[float(position)] for position in positions]
    for payload in feature_store:
        tensor = np.asarray(payload["tensor"], dtype=np.float64)
        if tensor.shape[0] == 0:
            continue
        y = np.asarray(payload["labels"], dtype=np.int32)
        local_rank_groups = np.asarray(payload["group_keys"], dtype=object)
        local_cv_groups = np.asarray(payload["cv_group_keys"], dtype=object)
        for local_pos_idx, src_pos_idx in enumerate(source_indices):
            x_raw = tensor[:, src_pos_idx, :]
            rows[local_pos_idx].append(x_raw)
            labels[local_pos_idx].append(y)
            rank_groups[local_pos_idx].append(local_rank_groups)
            cv_groups[local_pos_idx].append(local_cv_groups)

    out: list[dict[str, np.ndarray]] = []
    for pos_idx in range(len(positions)):
        if rows[pos_idx]:
            x_raw = np.vstack(rows[pos_idx]).astype(np.float64, copy=False)
            y = np.concatenate(labels[pos_idx]).astype(np.int32, copy=False)
            groups_rank = np.concatenate(rank_groups[pos_idx]).astype(object, copy=False)
            groups_cv = np.concatenate(cv_groups[pos_idx]).astype(object, copy=False)
        else:
            x_raw = np.zeros((0, int(n_features)), dtype=np.float64)
            y = np.zeros((0,), dtype=np.int32)
            groups_rank = np.asarray([], dtype=object)
            groups_cv = np.asarray([], dtype=object)

        x_rank = np.zeros_like(x_raw)
        if x_raw.shape[0] > 0:
            by_group: dict[Any, list[int]] = {}
            for row_idx, group_key in enumerate(groups_rank.tolist()):
                by_group.setdefault(group_key, []).append(row_idx)
            for row_ids in by_group.values():
                x_rank[row_ids] = _rank_transform_matrix(x_raw[row_ids])

        out.append({"x_raw": x_raw, "x_rank": x_rank, "y": y, "groups": groups_cv})
    return out


def _group_folds(groups: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return []
    splits = min(int(n_splits), int(len(unique_groups)))
    if splits < 2:
        return []
    from sklearn.model_selection import GroupKFold

    dummy = np.zeros((len(groups), 1), dtype=np.float64)
    gkf = GroupKFold(n_splits=splits)
    return list(gkf.split(dummy, groups=groups))


def _fit_lr_model(
    *,
    x: np.ndarray,
    y: np.ndarray,
    c_value: float,
    class_weight_name: str,
    random_state: int,
) -> Optional[dict[str, Any]]:
    if x.shape[0] < 4 or x.shape[1] < 1 or np.unique(y).shape[0] < 2:
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
    return {"scaler": scaler, "lr": clf}


def _predict_lr_model(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    scaler = model["scaler"]
    lr = model["lr"]
    return np.asarray(lr.decision_function(scaler.transform(x)), dtype=np.float64)


def _fit_anchor_model_suite(
    *,
    x_raw: np.ndarray,
    x_rank: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    svd_ranks: list[int],
    n_splits: int,
    random_state: int,
) -> dict[str, dict[str, Any]]:
    if x_raw.shape[0] == 0:
        raise ValueError("No labeled rows for anchor")
    if np.unique(y).shape[0] < 2:
        raise ValueError("Anchor lacks both classes")

    folds = _group_folds(groups, n_splits=n_splits)
    if not folds:
        raise ValueError("Anchor has insufficient CV groups")

    feature_indices = list(range(len(feature_names)))
    x_rep = _build_representation(
        x_raw=x_raw,
        x_rank=x_rank,
        feature_indices=feature_indices,
        representation="raw+rank",
    )

    no_svd_scores: dict[tuple[float, str], list[float]] = {}
    svd_scores: dict[int, dict[tuple[float, bool, str], list[float]]] = {int(rank): {} for rank in svd_ranks}
    max_requested_rank = max(1, min(max(int(v) for v in svd_ranks), int(x_rep.shape[1]), int(x_rep.shape[0] - 1)))

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
                    no_svd_scores.setdefault((float(c_value), str(class_weight)), []).append(float(fold_auc))

        if max_requested_rank < 1:
            continue
        svd = TruncatedSVD(n_components=int(max_requested_rank), random_state=int(random_state))
        try:
            z_train_full = svd.fit_transform(x_train_scaled)
            z_test_full = svd.transform(x_test_scaled)
        except Exception:
            continue
        singular_values = np.asarray(svd.singular_values_, dtype=np.float64)
        singular_values = np.where(np.abs(singular_values) < 1e-8, 1.0, singular_values)

        valid_ranks = [int(rank) for rank in svd_ranks if int(rank) <= int(max_requested_rank)]
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
                            svd_scores[int(rank)].setdefault(
                                (float(c_value), bool(whiten), str(class_weight)),
                                [],
                            ).append(float(fold_auc))

    best_no_svd = {"cv_auroc": float("-inf")}
    for (c_value, class_weight), values in no_svd_scores.items():
        if values:
            cv_auc = float(np.mean(values))
            if cv_auc > float(best_no_svd["cv_auroc"]):
                best_no_svd = {
                    "cv_auroc": cv_auc,
                    "n_valid_folds": int(len(values)),
                    "c_value": float(c_value),
                    "class_weight": str(class_weight),
                }
    if not np.isfinite(float(best_no_svd["cv_auroc"])):
        raise RuntimeError("Failed to fit any valid no-SVD candidate")

    no_svd_model = _fit_lr_model(
        x=x_rep,
        y=y,
        c_value=float(best_no_svd["c_value"]),
        class_weight_name=str(best_no_svd["class_weight"]),
        random_state=int(random_state),
    )
    if no_svd_model is None:
        raise RuntimeError("Failed to fit full no-SVD model")

    suite: dict[str, dict[str, Any]] = {
        "no_svd": {
            "route_type": "lr",
            "model_kind": "no_svd",
            "method_name": "no_svd",
            "cv_auroc": float(best_no_svd["cv_auroc"]),
            "n_valid_folds": int(best_no_svd["n_valid_folds"]),
            "representation": "raw+rank",
            "feature_names": list(feature_names),
            "feature_indices": feature_indices,
            "rank": None,
            "requested_rank": None,
            "effective_rank": None,
            "c_value": float(best_no_svd["c_value"]),
            "whiten": False,
            "class_weight": str(best_no_svd["class_weight"]),
            "model": no_svd_model,
        }
    }

    for rank in svd_ranks:
        candidates = svd_scores.get(int(rank), {})
        best_svd = {"cv_auroc": float("-inf")}
        for (c_value, whiten, class_weight), values in candidates.items():
            if values:
                cv_auc = float(np.mean(values))
                if cv_auc > float(best_svd["cv_auroc"]):
                    best_svd = {
                        "cv_auroc": cv_auc,
                        "n_valid_folds": int(len(values)),
                        "c_value": float(c_value),
                        "whiten": bool(whiten),
                        "class_weight": str(class_weight),
                    }
        if not np.isfinite(float(best_svd["cv_auroc"])):
            continue

        effective_rank = max(1, min(int(rank), int(x_rep.shape[1]), int(x_rep.shape[0] - 1)))
        model = _fit_svd_lr_model_local(
            x=x_rep,
            y=y,
            rank=int(effective_rank),
            c_value=float(best_svd["c_value"]),
            whiten=bool(best_svd["whiten"]),
            class_weight_name=str(best_svd["class_weight"]),
            random_state=int(random_state),
        )
        if model is None:
            continue
        suite[f"svd_r{int(rank)}"] = {
            "route_type": "svd",
            "model_kind": "svd",
            "method_name": f"svd_r{int(rank)}",
            "cv_auroc": float(best_svd["cv_auroc"]),
            "n_valid_folds": int(best_svd["n_valid_folds"]),
            "representation": "raw+rank",
            "feature_names": list(feature_names),
            "feature_indices": feature_indices,
            "rank": int(effective_rank),
            "requested_rank": int(rank),
            "effective_rank": int(effective_rank),
            "c_value": float(best_svd["c_value"]),
            "whiten": bool(best_svd["whiten"]),
            "class_weight": str(best_svd["class_weight"]),
            "model": model,
        }
    return suite


def _fit_svd_lr_model_local(
    *,
    x: np.ndarray,
    y: np.ndarray,
    rank: int,
    c_value: float,
    whiten: bool,
    class_weight_name: str,
    random_state: int,
) -> Optional[dict[str, Any]]:
    if x.shape[0] < 4 or x.shape[1] < 1 or np.unique(y).shape[0] < 2:
        return None
    scaler = StandardScaler(with_mean=True, with_std=True)
    x_scaled = scaler.fit_transform(x)
    max_rank = min(int(rank), int(x_scaled.shape[1]), int(x_scaled.shape[0] - 1))
    if max_rank < 1:
        return None
    svd = TruncatedSVD(n_components=max_rank, random_state=int(random_state))
    z = svd.fit_transform(x_scaled)
    if whiten:
        s = np.asarray(svd.singular_values_, dtype=np.float64)
        s = np.where(np.abs(s) < 1e-8, 1.0, s)
        z = z / s
    clf = LogisticRegression(
        C=float(c_value),
        class_weight=None if class_weight_name == "none" else "balanced",
        max_iter=2000,
        random_state=int(random_state),
    )
    clf.fit(z, y)
    return {"scaler": scaler, "svd": svd, "lr": clf, "whiten": bool(whiten)}


def _train_routes_for_domain(
    *,
    train_tables: list[dict[str, np.ndarray]],
    feature_names: list[str],
    svd_ranks: list[int],
    n_splits: int,
    random_state: int,
) -> dict[str, dict[float, dict[str, Any]]]:
    routes_by_model: dict[str, dict[float, dict[str, Any]]] = {"no_svd": {}}
    for rank in svd_ranks:
        routes_by_model[f"svd_r{int(rank)}"] = {}

    for pos_idx, position in enumerate(ANCHOR_POSITIONS):
        suite = _fit_anchor_model_suite(
            x_raw=train_tables[pos_idx]["x_raw"],
            x_rank=train_tables[pos_idx]["x_rank"],
            y=train_tables[pos_idx]["y"],
            groups=train_tables[pos_idx]["groups"],
            feature_names=feature_names,
            svd_ranks=svd_ranks,
            n_splits=int(n_splits),
            random_state=int(random_state),
        )
        for model_name, route in suite.items():
            route_with_pos = dict(route)
            route_with_pos["training_position"] = float(position)
            routes_by_model.setdefault(model_name, {})[float(position)] = route_with_pos
    return routes_by_model


def _make_score_fn(routes: dict[float, dict[str, Any]]):
    official_to_anchor = {
        int(pos_idx): float(OFFICIAL_SLOT_TO_ANCHOR[float(position)])
        for pos_idx, position in enumerate(EARLY_STOP_POSITIONS)
    }

    def _score(_domain: str, position_index: int, x_raw: np.ndarray) -> np.ndarray:
        route = routes[official_to_anchor[int(position_index)]]
        x_rank = _rank_transform_matrix(x_raw)
        x_rep = _build_representation(
            x_raw=x_raw,
            x_rank=x_rank,
            feature_indices=[int(v) for v in route["feature_indices"]],
            representation=str(route["representation"]),
        )
        if route["route_type"] == "svd":
            return _predict_svd_lr(route["model"], x_rep)
        return _predict_lr_model(route["model"], x_rep)

    return _score


def _evaluate_routes(
    *,
    condition_id: str,
    domain: str,
    seed: int,
    feature_names: list[str],
    holdout_store: list[dict[str, Any]],
    routes_by_model: dict[str, dict[float, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name, routes in routes_by_model.items():
        start = time.perf_counter()
        eval_result = evaluate_method_from_feature_store(
            method_name=f"{condition_id}__{domain}__seed{int(seed)}__{model_name}",
            feature_store=holdout_store,
            position_values=tuple(float(v) for v in EARLY_STOP_POSITIONS),
            score_fn=_make_score_fn(routes),
        )
        agg = eval_result["aggregate"]
        inference_time = float(time.perf_counter() - start)
        route_cv = [float(route["cv_auroc"]) for route in routes.values() if np.isfinite(float(route["cv_auroc"]))]
        rows.append(
            {
                "condition_id": str(condition_id),
                "domain": str(domain),
                "seed": int(seed),
                "model_name": str(model_name),
                "model_kind": "no_svd" if model_name == "no_svd" else "svd",
                "rank": "" if model_name == "no_svd" else int(model_name.split("_r", 1)[1]),
                "feature_count": int(len(feature_names)),
                "representation_dim": int(len(feature_names) * 2),
                "auc_of_auroc": float(agg["auc_of_auroc"]),
                "auc_of_selacc": float(agg["auc_of_selacc"]),
                "auroc_at_100": float(agg["auroc@100%"]),
                "stop_acc_at_100": float(agg["stop_acc@100%"]),
                "earliest_gt_0_6": "" if agg["earliest_gt_0.6"] is None else float(agg["earliest_gt_0.6"]),
                "num_caches": int(agg["num_caches"]),
                "samples": int(agg["samples"]),
                "mean_anchor_cv_auroc": float(np.mean(route_cv)) if route_cv else float("nan"),
                "inference_time_sec": float(inference_time),
            }
        )
    return rows


def _mean_std_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    metrics = (
        "auc_of_auroc",
        "auc_of_selacc",
        "auroc_at_100",
        "stop_acc_at_100",
        "mean_anchor_cv_auroc",
        "inference_time_sec",
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)

    out: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        base = {key: value for key, value in zip(keys, group_key)}
        for metric in metrics:
            values = [float(row[metric]) for row in group_rows if np.isfinite(float(row[metric]))]
            base[f"{metric}_mean"] = float(np.mean(values)) if values else float("nan")
            base[f"{metric}_std"] = float(np.std(values)) if values else float("nan")
        base["n_rows"] = int(len(group_rows))
        if "rank" not in base and group_rows and group_rows[0].get("rank", "") != "":
            base["rank"] = group_rows[0]["rank"]
        out.append(base)
    return out


def _best_svd_seed_rows(detail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in detail_rows:
        if row["model_kind"] != "svd":
            continue
        grouped.setdefault((str(row["condition_id"]), str(row["domain"]), int(row["seed"])), []).append(row)

    best_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        best = max(
            rows,
            key=lambda row: (
                float(row["auc_of_auroc"]),
                float(row["auc_of_selacc"]),
                -int(row["rank"]),
            ),
        )
        item = dict(best)
        item["selected_as"] = "best_svd_by_auc_of_auroc"
        best_rows.append(item)
    return best_rows


def _seed_comparison_rows(
    detail_rows: list[dict[str, Any]],
    best_svd_rows: list[dict[str, Any]],
    spec_by_id: dict[str, ConditionSpec],
) -> list[dict[str, Any]]:
    no_svd_lookup = {
        (str(row["condition_id"]), str(row["domain"]), int(row["seed"])): row
        for row in detail_rows
        if row["model_name"] == "no_svd"
    }
    out: list[dict[str, Any]] = []
    for best in best_svd_rows:
        key = (str(best["condition_id"]), str(best["domain"]), int(best["seed"]))
        no_svd = no_svd_lookup.get(key)
        if no_svd is None:
            continue
        spec = spec_by_id[str(best["condition_id"])]
        out.append(
            {
                "layer": str(spec.layer),
                "condition_id": str(best["condition_id"]),
                "condition_label": str(spec.label),
                "domain": str(best["domain"]),
                "seed": int(best["seed"]),
                "base_feature_count": int(spec.base_feature_count),
                "feature_count": int(best["feature_count"]),
                "decoy_family": "" if spec.decoy_family is None else str(spec.decoy_family),
                "decoy_level": "" if spec.decoy_level is None else str(spec.decoy_level),
                "no_svd_auc_of_auroc": float(no_svd["auc_of_auroc"]),
                "best_svd_auc_of_auroc": float(best["auc_of_auroc"]),
                "delta_auc_of_auroc": float(best["auc_of_auroc"]) - float(no_svd["auc_of_auroc"]),
                "no_svd_auc_of_selacc": float(no_svd["auc_of_selacc"]),
                "best_svd_auc_of_selacc": float(best["auc_of_selacc"]),
                "delta_auc_of_selacc": float(best["auc_of_selacc"]) - float(no_svd["auc_of_selacc"]),
                "no_svd_auroc_at_100": float(no_svd["auroc_at_100"]),
                "best_svd_auroc_at_100": float(best["auroc_at_100"]),
                "delta_auroc_at_100": float(best["auroc_at_100"]) - float(no_svd["auroc_at_100"]),
                "no_svd_stop_acc_at_100": float(no_svd["stop_acc_at_100"]),
                "best_svd_stop_acc_at_100": float(best["stop_acc_at_100"]),
                "delta_stop_acc_at_100": float(best["stop_acc_at_100"]) - float(no_svd["stop_acc_at_100"]),
                "best_svd_rank": int(best["rank"]),
            }
        )
    return out


def _aggregate_seed_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["condition_id"]), str(row["domain"])), []).append(row)

    out: list[dict[str, Any]] = []
    for (condition_id, domain), group_rows in sorted(grouped.items(), key=lambda item: (_condition_index(item[0][0]), item[0][1])):
        base = {
            "condition_id": str(condition_id),
            "domain": str(domain),
            "layer": str(group_rows[0]["layer"]),
            "condition_label": str(group_rows[0]["condition_label"]),
            "base_feature_count": int(group_rows[0]["base_feature_count"]),
            "feature_count": int(group_rows[0]["feature_count"]),
            "decoy_family": str(group_rows[0]["decoy_family"]),
            "decoy_level": str(group_rows[0]["decoy_level"]),
            "n_seeds": int(len(group_rows)),
        }
        for metric in (
            "no_svd_auc_of_auroc",
            "best_svd_auc_of_auroc",
            "delta_auc_of_auroc",
            "no_svd_auc_of_selacc",
            "best_svd_auc_of_selacc",
            "delta_auc_of_selacc",
            "delta_auroc_at_100",
            "delta_stop_acc_at_100",
        ):
            vals = [float(row[metric]) for row in group_rows]
            base[f"{metric}_mean"] = float(np.mean(vals))
            base[f"{metric}_std"] = float(np.std(vals))
        rank_vals = [int(row["best_svd_rank"]) for row in group_rows]
        rank_hist: dict[int, int] = {}
        for value in rank_vals:
            rank_hist[int(value)] = rank_hist.get(int(value), 0) + 1
        best_rank = max(sorted(rank_hist.items()), key=lambda item: (item[1], -item[0]))[0]
        base["best_svd_rank_mode"] = int(best_rank)
        base["best_svd_rank_hist"] = json.dumps(rank_hist, sort_keys=True)
        out.append(base)
    return out


def _macro_seed_comparisons(rows: list[dict[str, Any]], domains: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if str(row["domain"]) not in domains:
            continue
        grouped.setdefault((str(row["condition_id"]), int(row["seed"])), []).append(row)

    out: list[dict[str, Any]] = []
    for (condition_id, seed), group_rows in sorted(grouped.items(), key=lambda item: (_condition_index(item[0][0]), item[0][1])):
        if len(group_rows) != len(domains):
            continue
        base = dict(group_rows[0])
        base["domain"] = "macro"
        for metric in (
            "no_svd_auc_of_auroc",
            "best_svd_auc_of_auroc",
            "delta_auc_of_auroc",
            "no_svd_auc_of_selacc",
            "best_svd_auc_of_selacc",
            "delta_auc_of_selacc",
            "delta_auroc_at_100",
            "delta_stop_acc_at_100",
        ):
            base[metric] = float(np.mean([float(row[metric]) for row in group_rows]))
        base["best_svd_rank"] = int(round(float(np.mean([int(row["best_svd_rank"]) for row in group_rows]))))
        out.append(base)
    return out


def _condition_catalog_rows(specs: list[ConditionSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        feature_count = int(spec.base_feature_count)
        if spec.decoy_family is not None:
            feature_count += max(1, int(round(spec.base_feature_count * float(spec.decoy_ratio))))
        rows.append(
            {
                "layer": str(spec.layer),
                "condition_id": str(spec.condition_id),
                "label": str(spec.label),
                "feature_count": int(feature_count),
                "decoy_family": "" if spec.decoy_family is None else str(spec.decoy_family),
                "decoy_level": "" if spec.decoy_level is None else str(spec.decoy_level),
                "description": str(spec.description),
                "features": ", ".join(spec.feature_names),
            }
        )
    return rows


def _plot_clean_curves(
    *,
    comparison_rows: list[dict[str, Any]],
    out_curve: Path,
    out_gain: Path,
) -> None:
    clean_rows = [
        row for row in comparison_rows
        if row["layer"] in {"canonical", "expansion"} and row["domain"] in {"macro", "math", "science", "coding"}
    ]
    if not clean_rows:
        return
    domain_order = ["macro", "math", "science", "coding"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, domain in zip(axes.flat, domain_order):
        rows = sorted(
            [row for row in clean_rows if row["domain"] == domain],
            key=lambda row: (int(row["feature_count"]), _condition_index(str(row["condition_id"]))),
        )
        x = [int(row["feature_count"]) for row in rows]
        no_svd = [float(row["no_svd_auc_of_auroc_mean"]) for row in rows]
        best_svd = [float(row["best_svd_auc_of_auroc_mean"]) for row in rows]
        ax.plot(x, no_svd, marker="o", label="no-SVD")
        ax.plot(x, best_svd, marker="o", label="best-SVD")
        ax.set_title(domain)
        ax.set_ylabel("AUC of AUROC")
        ax.grid(True, alpha=0.25)
        if domain == "macro":
            ax.legend(loc="best")
    axes[1, 0].set_xlabel("# features")
    axes[1, 1].set_xlabel("# features")
    fig.tight_layout()
    out_curve.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_curve, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, domain in zip(axes.flat, domain_order):
        rows = sorted(
            [row for row in clean_rows if row["domain"] == domain],
            key=lambda row: (int(row["feature_count"]), _condition_index(str(row["condition_id"]))),
        )
        x = [int(row["feature_count"]) for row in rows]
        delta = [float(row["delta_auc_of_auroc_mean"]) for row in rows]
        ax.plot(x, delta, marker="o", color="#8c2d04")
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.4)
        ax.set_title(domain)
        ax.set_ylabel("Δ(best-SVD - no-SVD)")
        ax.grid(True, alpha=0.25)
    axes[1, 0].set_xlabel("# features")
    axes[1, 1].set_xlabel("# features")
    fig.tight_layout()
    fig.savefig(out_gain, dpi=160)
    plt.close(fig)


def _plot_noise_robustness(
    *,
    comparison_rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    noise_rows = [row for row in comparison_rows if row["layer"] == "noise" and row["domain"] == "macro"]
    if not noise_rows:
        return
    families = ["permutation", "duplicate", "random"]
    levels = ["low", "med", "high"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, family in zip(axes, families):
        rows = [row for row in noise_rows if row["decoy_family"] == family]
        rows = sorted(rows, key=lambda row: levels.index(str(row["decoy_level"])))
        x = np.arange(len(rows))
        no_svd = [float(row["no_svd_auc_of_auroc_mean"]) for row in rows]
        best_svd = [float(row["best_svd_auc_of_auroc_mean"]) for row in rows]
        delta = [float(row["delta_auc_of_auroc_mean"]) for row in rows]
        ax.plot(x, no_svd, marker="o", label="no-SVD")
        ax.plot(x, best_svd, marker="o", label="best-SVD")
        ax.bar(x, delta, alpha=0.2, color="#8c2d04", label="Δ")
        ax.set_xticks(x, [str(row["decoy_level"]) for row in rows])
        ax.set_title(family)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("AUC of AUROC")
    axes[0].legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _build_results_doc(
    *,
    protocol: dict[str, Any],
    specs: list[ConditionSpec],
    comparison_rows: list[dict[str, Any]],
    out_curve: Optional[str],
    out_gain: Optional[str],
    out_noise_plot: Optional[str],
) -> str:
    by_key = {
        (str(row["condition_id"]), str(row["domain"])): row
        for row in comparison_rows
    }
    macro_clean = [
        row for row in comparison_rows
        if row["domain"] == "macro" and row["layer"] in {"canonical", "expansion"}
    ]
    macro_clean = sorted(macro_clean, key=lambda row: (int(row["feature_count"]), _condition_index(str(row["condition_id"]))))
    macro_noise = [
        row for row in comparison_rows
        if row["domain"] == "macro" and row["layer"] == "noise"
    ]
    macro_noise = sorted(
        macro_noise,
        key=lambda row: (
            str(row["decoy_family"]),
            {"low": 0, "med": 1, "high": 2}.get(str(row["decoy_level"]), 99),
        ),
    )

    canonical = by_key.get(("canonical_22", "macro"))
    widest_real = by_key.get(("canonical_plus_event_local", "macro"))
    headline_clean = ""
    if canonical is not None and widest_real is not None:
        headline_clean = (
            f"- 在 macro 平均上，`canonical_22` 的 `Δ(best-SVD − no-SVD)` 为 "
            f"`{float(canonical['delta_auc_of_auroc_mean']) * 100.0:+.2f}` AUC-pts；"
            f"扩展到最宽真实 bank (`30` features) 后变为 "
            f"`{float(widest_real['delta_auc_of_auroc_mean']) * 100.0:+.2f}` AUC-pts。"
        )
    best_noise = max(macro_noise, key=lambda row: float(row["delta_auc_of_auroc_mean"])) if macro_noise else None
    headline_noise = ""
    if best_noise is not None:
        headline_noise = (
            f"- 在 decoy control 里，macro 最大 SVD 增益出现在 "
            f"`{best_noise['decoy_family']} / {best_noise['decoy_level']}`，"
            f"`Δ={float(best_noise['delta_auc_of_auroc_mean']) * 100.0:+.2f}` AUC-pts。"
        )

    lines = [
        "# 18. SVD Feature Complexity Results",
        "",
        "这份结果 note 用同一套 grouped holdout / anchor routing / raw+rank 协议，对比 `no-SVD` 与 `best fixed-rank SVD` 在三层 feature-complexity 设计中的相对表现。",
        "",
        "## 1. Protocol Snapshot",
        "",
        f"- `domains`: `{', '.join(protocol['domains'])}`",
        f"- `layers`: `{', '.join(protocol['layers'])}`",
        f"- `seeds`: `{', '.join(str(v) for v in protocol['seeds'])}`",
        f"- `svd ranks`: `{', '.join(str(v) for v in protocol['svd_ranks'])}`",
        f"- `holdout split`: `{protocol['holdout_split']}` with `split_seed={protocol['split_seed']}`",
        f"- `n_splits`: `{protocol['n_splits']}`",
        f"- `mode`: `{'smoke' if protocol['smoke'] else 'full'}`",
        "",
        "## 2. Headline Answers",
        "",
    ]
    if headline_clean:
        lines.append(headline_clean)
    if widest_real is not None:
        lines.append(
            f"- `30`-feature real-upstream bank 的 macro `AUC of AUROC`："
            f"`no-SVD={float(widest_real['no_svd_auc_of_auroc_mean']) * 100.0:.2f}%`，"
            f"`best-SVD={float(widest_real['best_svd_auc_of_auroc_mean']) * 100.0:.2f}%`。"
        )
    if headline_noise:
        lines.append(headline_noise)

    lines.extend([
        "",
        "## 3. Condition Catalog",
        "",
        "| Layer | Condition | #feat | Decoy | Description |",
        "|---|---|---:|---|---|",
    ])
    for spec in specs:
        decoy = "—" if spec.decoy_family is None else f"{spec.decoy_family}/{spec.decoy_level}"
        feature_count = int(spec.base_feature_count)
        if spec.decoy_family is not None:
            feature_count = max(1, int(round(spec.base_feature_count * (1.0 + spec.decoy_ratio))))
        lines.append(
            f"| {spec.layer} | `{spec.condition_id}` | {feature_count} | {decoy} | {spec.description} |"
        )

    lines.extend([
        "",
        "## 4. Clean Feature Sweep (macro)",
        "",
        "| Condition | #feat | no-SVD AUC | best-SVD AUC | ΔAUC | no-SVD SelAcc | best-SVD SelAcc | ΔSelAcc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in macro_clean:
        lines.append(
            "| {condition} | {count} | {no_auc} | {svd_auc} | {delta_auc} | {no_sel} | {svd_sel} | {delta_sel} |".format(
                condition=row["condition_id"],
                count=int(row["feature_count"]),
                no_auc=_fmt_pct(float(row["no_svd_auc_of_auroc_mean"])),
                svd_auc=_fmt_pct(float(row["best_svd_auc_of_auroc_mean"])),
                delta_auc=_fmt_pct(float(row["delta_auc_of_auroc_mean"])),
                no_sel=_fmt_pct(float(row["no_svd_auc_of_selacc_mean"])),
                svd_sel=_fmt_pct(float(row["best_svd_auc_of_selacc_mean"])),
                delta_sel=_fmt_pct(float(row["delta_auc_of_selacc_mean"])),
            )
        )

    lines.extend([
        "",
        "## 5. Noise Robustness (macro)",
        "",
        "| Family | Dose | #feat | no-SVD AUC | best-SVD AUC | ΔAUC |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in macro_noise:
        lines.append(
            "| {family} | {dose} | {count} | {no_auc} | {svd_auc} | {delta_auc} |".format(
                family=row["decoy_family"],
                dose=row["decoy_level"],
                count=int(row["feature_count"]),
                no_auc=_fmt_pct(float(row["no_svd_auc_of_auroc_mean"])),
                svd_auc=_fmt_pct(float(row["best_svd_auc_of_auroc_mean"])),
                delta_auc=_fmt_pct(float(row["delta_auc_of_auroc_mean"])),
            )
        )

    lines.extend([
        "",
        "## 6. Figures",
        "",
    ])
    if out_curve:
        lines.append(f"- Clean sweep curves: `{out_curve}`")
    if out_gain:
        lines.append(f"- Clean sweep SVD-gain curves: `{out_gain}`")
    if out_noise_plot:
        lines.append(f"- Noise robustness plot: `{out_noise_plot}`")

    lines.extend([
        "",
        "## 7. Reproduction",
        "",
        "```bash",
        "bash cookbook/00_setup/verify.sh",
        "python3 SVDomain/experiments/run_svd_feature_complexity_study.py \\",
        "  --domains math,science,coding \\",
        "  --layers canonical,expansion,noise \\",
        "  --seeds 42,43,44 \\",
        "  --svd-ranks 4,8,12,16,24",
        "```",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the three-layer SVD vs no-SVD feature-complexity study")
    ap.add_argument("--main-cache-root", default="MUI_HUB/cache")
    ap.add_argument("--extra-cache-root", default="MUI_HUB/cache_train")
    ap.add_argument("--main-store-path", default="results/cache/es_svd_ms_rr_r1/cache_all_547b9060debe139e.pkl")
    ap.add_argument("--extra-store-path", default="results/cache/es_svd_ms_rr_r1/cache_train_all_d429f3b93baed972.pkl")
    ap.add_argument("--rebuild-store", action="store_true")
    ap.add_argument("--feature-cache-dir", default="results/cache/feature_complexity")
    ap.add_argument("--feature-workers", type=int, default=default_feature_workers())
    ap.add_argument("--feature-chunk-problems", type=int, default=DEFAULT_FEATURE_CHUNK_PROBLEMS)
    ap.add_argument("--max-problems-per-cache", type=int, default=0, help="0 means all problems when rebuilding stores")
    ap.add_argument("--domains", default="math,science,coding")
    ap.add_argument("--layers", default="canonical,expansion,noise")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--svd-ranks", default="4,8,12,16,24")
    ap.add_argument("--holdout-split", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--fit-workers", type=int, default=default_fit_workers())
    ap.add_argument("--decoy-seed", type=int, default=7)
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-max-problems-per-payload", type=int, default=6)
    ap.add_argument("--out-summary", default="results/scans/feature_complexity/three_layer_summary.json")
    ap.add_argument("--out-detail-csv", default="results/tables/feature_complexity_model_rows.csv")
    ap.add_argument("--out-aggregate-csv", default="results/tables/feature_complexity_aggregate_rows.csv")
    ap.add_argument("--out-comparison-csv", default="results/tables/feature_complexity_comparison.csv")
    ap.add_argument("--out-clean-csv", default="results/tables/feature_complexity_clean_sweep.csv")
    ap.add_argument("--out-noise-csv", default="results/tables/feature_complexity_noise_robustness.csv")
    ap.add_argument("--out-condition-csv", default="results/tables/feature_complexity_conditions.csv")
    ap.add_argument("--out-doc", default="docs/18_SVD_FEATURE_COMPLEXITY_RESULTS.md")
    ap.add_argument("--out-curve-fig", default="results/figures/feature_complexity/clean_feature_curves.png")
    ap.add_argument("--out-gain-fig", default="results/figures/feature_complexity/clean_feature_gain.png")
    ap.add_argument("--out-noise-fig", default="results/figures/feature_complexity/noise_robustness.png")
    args = ap.parse_args()

    domains = [domain for domain in _parse_csv_strings(str(args.domains)) if domain in DOMAIN_ORDER]
    if not domains:
        raise ValueError("No supported domains requested")
    layers = [layer for layer in _parse_csv_strings(str(args.layers)) if layer in LAYER_ORDER]
    if not layers:
        raise ValueError("No supported layers requested")
    seeds = _parse_csv_ints(str(args.seeds))
    svd_ranks = sorted(dict.fromkeys(_parse_csv_ints(str(args.svd_ranks))))

    if bool(args.smoke):
        seeds = seeds[:1]
        svd_ranks = [rank for rank in svd_ranks[:2]] or [12]

    main_cache_root = _resolve_path(str(args.main_cache_root))
    extra_cache_root = None
    if str(args.extra_cache_root).strip().lower() not in {"", "none", "off"}:
        extra_cache_root = _resolve_path(str(args.extra_cache_root))
    feature_cache_dir = None if str(args.feature_cache_dir).strip().lower() in {"", "none", "off"} else (REPO_ROOT / str(args.feature_cache_dir)).resolve()
    max_problems_per_cache = None if int(args.max_problems_per_cache) <= 0 else int(args.max_problems_per_cache)

    full_store, store_meta = _load_full_store(
        main_store_path=None if bool(args.rebuild_store) else str(args.main_store_path),
        extra_store_path=None if bool(args.rebuild_store) else str(args.extra_store_path),
        rebuild_store=bool(args.rebuild_store),
        main_cache_root=str(main_cache_root),
        extra_cache_root=extra_cache_root,
        feature_cache_dir=feature_cache_dir,
        feature_workers=int(args.feature_workers),
        feature_chunk_problems=int(args.feature_chunk_problems),
        max_problems_per_cache=max_problems_per_cache,
    )
    if bool(args.smoke):
        full_store = _limit_feature_store(full_store, int(args.smoke_max_problems_per_payload))

    full_store = [payload for payload in full_store if str(payload["domain"]) in domains]
    split_packs: dict[str, dict[str, Any]] = {}
    for domain in domains:
        domain_store = [payload for payload in full_store if str(payload["domain"]) == domain]
        holdout_problem_map, holdout_problem_summary = _build_holdout_problem_map(
            domain_store,
            holdout_split=float(args.holdout_split),
            split_seed=int(args.split_seed),
        )
        train_store, holdout_store, domain_full_store = _split_feature_store(
            domain_store,
            holdout_problem_map=holdout_problem_map,
        )
        split_packs[domain] = {
            "train_store": train_store,
            "holdout_store": holdout_store,
            "full_store": domain_full_store,
            "holdout_problem_summary": holdout_problem_summary,
        }

    specs = [spec for spec in _condition_specs() if spec.layer in layers]
    spec_by_id = {str(spec.condition_id): spec for spec in specs}

    detail_rows: list[dict[str, Any]] = []
    condition_rows = _condition_catalog_rows(specs)
    start_all = time.perf_counter()

    for spec in specs:
        print(f"[condition] start {spec.condition_id} layer={spec.layer}", flush=True)
        for domain in domains:
            train_store, feature_names = _transform_feature_store(
                split_packs[domain]["train_store"],
                spec=spec,
                decoy_seed=int(args.decoy_seed),
            )
            holdout_store, _ = _transform_feature_store(
                split_packs[domain]["holdout_store"],
                spec=spec,
                decoy_seed=int(args.decoy_seed),
            )
            train_tables = _build_training_tables(
                train_store,
                positions=tuple(float(v) for v in ANCHOR_POSITIONS),
                n_features=int(len(feature_names)),
            )
            for seed in seeds:
                print(
                    f"[train] condition={spec.condition_id:<32s} domain={domain:<7s} seed={seed}",
                    flush=True,
                )
                routes_by_model = _train_routes_for_domain(
                    train_tables=train_tables,
                    feature_names=feature_names,
                    svd_ranks=svd_ranks,
                    n_splits=int(args.n_splits),
                    random_state=int(seed),
                )
                rows = _evaluate_routes(
                    condition_id=str(spec.condition_id),
                    domain=str(domain),
                    seed=int(seed),
                    feature_names=feature_names,
                    holdout_store=holdout_store,
                    routes_by_model=routes_by_model,
                )
                for row in rows:
                    row["layer"] = str(spec.layer)
                    row["condition_label"] = str(spec.label)
                    row["description"] = str(spec.description)
                    row["base_feature_count"] = int(spec.base_feature_count)
                    row["decoy_family"] = "" if spec.decoy_family is None else str(spec.decoy_family)
                    row["decoy_level"] = "" if spec.decoy_level is None else str(spec.decoy_level)
                detail_rows.extend(rows)

    total_wall_time = float(time.perf_counter() - start_all)

    aggregate_rows = _mean_std_rows(
        detail_rows,
        keys=("layer", "condition_id", "condition_label", "domain", "model_name", "model_kind", "rank", "feature_count"),
    )
    best_svd_seed_rows = _best_svd_seed_rows(detail_rows)
    seed_comparison_rows = _seed_comparison_rows(detail_rows, best_svd_seed_rows, spec_by_id)
    comparison_rows = _aggregate_seed_comparisons(seed_comparison_rows)
    macro_seed_rows = _macro_seed_comparisons(seed_comparison_rows, domains=domains)
    macro_rows = _aggregate_seed_comparisons(macro_seed_rows)
    comparison_with_macro = comparison_rows + macro_rows

    clean_rows = [
        row for row in comparison_with_macro
        if row["domain"] in set(domains) | {"macro"} and row["layer"] in {"canonical", "expansion"}
    ]
    noise_rows = [
        row for row in comparison_with_macro
        if row["domain"] in set(domains) | {"macro"} and row["layer"] == "noise"
    ]

    out_summary = REPO_ROOT / str(args.out_summary)
    out_detail_csv = REPO_ROOT / str(args.out_detail_csv)
    out_aggregate_csv = REPO_ROOT / str(args.out_aggregate_csv)
    out_comparison_csv = REPO_ROOT / str(args.out_comparison_csv)
    out_clean_csv = REPO_ROOT / str(args.out_clean_csv)
    out_noise_csv = REPO_ROOT / str(args.out_noise_csv)
    out_condition_csv = REPO_ROOT / str(args.out_condition_csv)
    out_doc = REPO_ROOT / str(args.out_doc)
    out_curve_fig = REPO_ROOT / str(args.out_curve_fig)
    out_gain_fig = REPO_ROOT / str(args.out_gain_fig)
    out_noise_fig = REPO_ROOT / str(args.out_noise_fig)

    _write_csv(out_condition_csv, condition_rows)
    _write_csv(out_detail_csv, detail_rows)
    _write_csv(out_aggregate_csv, aggregate_rows)
    _write_csv(out_comparison_csv, comparison_with_macro)
    _write_csv(out_clean_csv, clean_rows)
    _write_csv(out_noise_csv, noise_rows)

    curve_rel: Optional[str] = None
    gain_rel: Optional[str] = None
    noise_rel: Optional[str] = None
    if not bool(args.skip_plots):
        _plot_clean_curves(comparison_rows=comparison_with_macro, out_curve=out_curve_fig, out_gain=out_gain_fig)
        _plot_noise_robustness(comparison_rows=comparison_with_macro, out_path=out_noise_fig)
        curve_rel = str(Path(args.out_curve_fig))
        gain_rel = str(Path(args.out_gain_fig))
        noise_rel = str(Path(args.out_noise_fig))

    protocol = {
        "created_at_utc": _now_utc(),
        "domains": list(domains),
        "layers": list(layers),
        "seeds": [int(v) for v in seeds],
        "svd_ranks": [int(v) for v in svd_ranks],
        "holdout_split": float(args.holdout_split),
        "split_seed": int(args.split_seed),
        "n_splits": int(args.n_splits),
        "smoke": bool(args.smoke),
        "smoke_max_problems_per_payload": int(args.smoke_max_problems_per_payload),
        "decoy_seed": int(args.decoy_seed),
        "main_cache_root": str(main_cache_root),
        "extra_cache_root": None if extra_cache_root is None else str(extra_cache_root),
        "main_store_path": str(args.main_store_path),
        "extra_store_path": str(args.extra_store_path),
        "rebuild_store": bool(args.rebuild_store),
        "store_meta": store_meta,
        "anchor_positions": [float(v) for v in ANCHOR_POSITIONS],
        "official_positions": [float(v) for v in EARLY_STOP_POSITIONS],
    }

    summary = {
        "created_at_utc": _now_utc(),
        "protocol": protocol,
        "data": {
            "full_store_summary": _summarise_feature_store(full_store),
            "by_domain": {
                domain: {
                    "train_store": _summarise_feature_store(split_packs[domain]["train_store"]),
                    "holdout_store": _summarise_feature_store(split_packs[domain]["holdout_store"]),
                    "full_store": _summarise_feature_store(split_packs[domain]["full_store"]),
                    "holdout_problem_summary": split_packs[domain]["holdout_problem_summary"],
                }
                for domain in domains
            },
        },
        "conditions": condition_rows,
        "tables": {
            "detail_rows": detail_rows,
            "aggregate_rows": aggregate_rows,
            "comparison_rows": comparison_with_macro,
            "clean_rows": clean_rows,
            "noise_rows": noise_rows,
        },
        "timing": {
            "total_wall_time_sec": float(total_wall_time),
        },
        "artifacts": {
            "summary_json": str(args.out_summary),
            "detail_csv": str(args.out_detail_csv),
            "aggregate_csv": str(args.out_aggregate_csv),
            "comparison_csv": str(args.out_comparison_csv),
            "clean_csv": str(args.out_clean_csv),
            "noise_csv": str(args.out_noise_csv),
            "condition_csv": str(args.out_condition_csv),
            "doc_md": str(args.out_doc),
            "curve_png": None if curve_rel is None else curve_rel,
            "gain_png": None if gain_rel is None else gain_rel,
            "noise_png": None if noise_rel is None else noise_rel,
        },
    }

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    doc_text = _build_results_doc(
        protocol=protocol,
        specs=specs,
        comparison_rows=comparison_with_macro,
        out_curve=curve_rel,
        out_gain=gain_rel,
        out_noise_plot=noise_rel,
    )
    out_doc.parent.mkdir(parents=True, exist_ok=True)
    out_doc.write_text(doc_text, encoding="utf-8")

    print("[done] feature-complexity study finished", flush=True)
    print(f"[done] summary={out_summary}", flush=True)
    print(f"[done] doc={out_doc}", flush=True)


if __name__ == "__main__":
    main()
