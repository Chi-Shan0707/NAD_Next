#!/usr/bin/env python3
"""Train split-domain EarlyStop SVD models for math/science with raw+rank only.

Purpose
-------
Trains the canonical r1 EarlyStop SVD family: math, science, and combined (ms) domain models.
Each model uses a StandardScaler → TruncatedSVD → LogisticRegression pipeline with raw+rank
feature representation and four training anchors: 10 / 40 / 70 / 100 percent.

Method
------
- Feature family: token_plus_traj_fixed (token uncertainty + trajectory shape + availability flags)
- Representation: raw+rank (each feature appears as both its raw value and within-group rank)
- Anchors: 10 / 40 / 70 / 100 percent of generation length
- Slot routing: nearby official slots map to the nearest anchor (e.g. 20/30% → 10% anchor)
- Search grid: rank ∈ {2,4,6,8,12,16}, C ∈ {0.05,0.1,0.2,0.5,1.0}, whiten ∈ {F,T}
- Holdout: 85/15 split by dataset+problem_id, consistent across cache roots; seed=42

Data Requirements
-----------------
- --main-cache-root: primary labeled NAD cache (default: MUI_HUB/cache)
- --extra-cache-root: additional labeled NAD cache (default: MUI_HUB/cache_train)
Both must contain meta.json, base/, token_data/ as per the NAD cache layout.

CLI Outputs
-----------
- models/ml_selectors/es_svd_math_rr_r1.pkl
- models/ml_selectors/es_svd_science_rr_r1.pkl
- models/ml_selectors/es_svd_ms_rr_r1.pkl  (combined noncoding bundle)
- results/scans/earlystop/es_svd_ms_rr_r1_summary.json
- results/scans/earlystop/es_svd_ms_rr_r1_eval.json
- docs/ES_SVD_MS_RR_R1.md  (auto-generated results document)

Integration Note
----------------
To adapt to a new dataset: (1) build a NAD cache with nad.cli cache-build-fast,
(2) pass --main-cache-root pointing to your cache, (3) the domain assignment in
DEFAULT_DOMAIN_MAP must include your dataset names mapped to "math", "science", or "coding".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.selectors.trajectory_impl import DEFAULT_REFLECTION_THRESHOLD
from nad.ops.earlystop import EARLY_STOP_POSITIONS, _problem_sort_key
from nad.ops.earlystop_svd import (
    FULL_FEATURE_NAMES,
    _auroc,
    _build_representation,
    _cv_auroc_baseline,
    _fit_svd_lr_model,
    _rank_transform_matrix,
    load_earlystop_svd_bundle,
    save_earlystop_svd_bundle,
)
from scripts.run_earlystop_prefix10_svd_round1 import (
    ANCHOR_POSITIONS,
    DEFAULT_FEATURE_CHUNK_PROBLEMS,
    EXTRACTION_POSITION_INDEX,
    EXTRACTION_POSITIONS,
    FEATURE_TO_INDEX,
    OFFICIAL_SLOT_TO_ANCHOR,
    PREFIX_SAFE_FEATURE_FAMILY_MAP,
    PREFIX_SAFE_SIGNAL_NAMES,
    SEARCH_C_VALUES,
    SEARCH_CLASS_WEIGHT,
    SEARCH_RANKS,
    SEARCH_WHITEN,
    _display_path,
    _now_utc,
    _pct_label,
    build_feature_store,
    evaluate_method_from_feature_store,
    make_svd_bundle_score_fn,
    make_tok_conf_score_fn,
    summarise_route,
    _render_cache_table,
    _render_method_table,
)


FIXED_FEATURE_NAMES = list(PREFIX_SAFE_FEATURE_FAMILY_MAP["token_plus_traj"])
FIXED_FEATURE_INDICES = [FEATURE_TO_INDEX[name] for name in FIXED_FEATURE_NAMES]
FIXED_REPRESENTATION = "raw+rank"
FIXED_FAMILY_NAME = "token_plus_traj_fixed"
FIXED_EXCLUDED_FEATURES = [name for name in FULL_FEATURE_NAMES if name not in FIXED_FEATURE_NAMES]
DOMAINS = ("math", "science")
METHOD_IDS = {
    "math": "es_svd_math_rr_r1",
    "science": "es_svd_science_rr_r1",
    "combined": "es_svd_ms_rr_r1",
}


def _stable_hash(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _qualify_feature_store(feature_store: list[dict[str, Any]], source_name: str) -> list[dict[str, Any]]:
    qualified: list[dict[str, Any]] = []
    for payload in feature_store:
        item = dict(payload)
        item["source_name"] = str(source_name)
        item["base_cache_key"] = str(payload["cache_key"])
        item["cache_key"] = f"{source_name}/{payload['cache_key']}"
        qualified.append(item)
    return qualified


def _feature_cache_key(
    *,
    source_name: str,
    cache_root: str,
    positions: tuple[float, ...],
    required_feature_names: set[str],
    max_problems_per_cache: Optional[int],
) -> str:
    payload = {
        "version": 1,
        "source_name": str(source_name),
        "cache_root": str(cache_root),
        "positions": [float(p) for p in positions],
        "required_feature_names": sorted(str(v) for v in required_feature_names),
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
    required_feature_names: set[str],
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


def _load_or_build_qualified_feature_store(
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
            print(f"[load] feature cache source={source_name} path={cache_path}", flush=True)
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            return list(payload["feature_store"]), cache_path, "loaded"

    print(f"[build] feature store source={source_name} root={cache_root}", flush=True)
    store = _qualify_feature_store(
        build_feature_store(
            cache_root=cache_root,
            positions=positions,
            required_feature_names=required_feature_names,
            max_problems_per_cache=max_problems_per_cache,
            max_workers=max(1, int(feature_workers)),
            chunk_problems=max(1, int(chunk_problems)),
            reflection_threshold=float(DEFAULT_REFLECTION_THRESHOLD),
        ),
        source_name=source_name,
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(
                {
                    "source_name": str(source_name),
                    "cache_root": str(cache_root),
                    "positions": [float(p) for p in positions],
                    "max_problems_per_cache": None if max_problems_per_cache is None else int(max_problems_per_cache),
                    "feature_store": store,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        tmp_path.replace(cache_path)
        print(f"[save] feature cache source={source_name} path={cache_path}", flush=True)
    return store, cache_path, "built"


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


def _annotate_full_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return _subset_payload_by_problem_ids(payload, {str(v) for v in payload["problem_ids"]}) or {
        "cache_key": str(payload["cache_key"]),
        "base_cache_key": str(payload["base_cache_key"]),
        "source_name": str(payload["source_name"]),
        "dataset_name": str(payload["dataset_name"]),
        "domain": str(payload["domain"]),
        "positions": list(payload["positions"]),
        "tensor": np.zeros((0, len(EXTRACTION_POSITIONS), len(FULL_FEATURE_NAMES)), dtype=np.float64),
        "labels": np.zeros((0,), dtype=np.int32),
        "sample_ids": np.zeros((0,), dtype=np.int32),
        "group_keys": np.asarray([], dtype=object),
        "cv_group_keys": np.asarray([], dtype=object),
        "problem_ids": [],
        "problem_offsets": [0],
        "samples": 0,
    }


def _build_holdout_problem_map(
    feature_store: list[dict[str, Any]],
    *,
    holdout_split: float,
    split_seed: int,
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    datasets: dict[str, set[str]] = {}
    for payload in feature_store:
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    full_payloads: list[dict[str, Any]] = []
    train_payloads: list[dict[str, Any]] = []
    holdout_payloads: list[dict[str, Any]] = []

    for payload in feature_store:
        full_payload = _annotate_full_payload(payload)
        if full_payload["samples"] > 0:
            full_payloads.append(full_payload)

        holdout_ids = set(holdout_problem_map.get(str(payload["dataset_name"]), set()))
        train_ids = {str(v) for v in payload["problem_ids"] if str(v) not in holdout_ids}

        train_payload = _subset_payload_by_problem_ids(payload, train_ids)
        if train_payload is not None and train_payload["samples"] > 0:
            train_payloads.append(train_payload)

        holdout_payload = _subset_payload_by_problem_ids(payload, holdout_ids)
        if holdout_payload is not None and holdout_payload["samples"] > 0:
            holdout_payloads.append(holdout_payload)

    return train_payloads, holdout_payloads, full_payloads


def _build_domain_training_tables(
    feature_store: list[dict[str, Any]],
    positions: tuple[float, ...],
) -> list[dict[str, np.ndarray]]:
    rows: dict[int, list[np.ndarray]] = {idx: [] for idx in range(len(positions))}
    labels: dict[int, list[np.ndarray]] = {idx: [] for idx in range(len(positions))}
    rank_groups: dict[int, list[np.ndarray]] = {idx: [] for idx in range(len(positions))}
    cv_groups: dict[int, list[np.ndarray]] = {idx: [] for idx in range(len(positions))}

    position_indices = [EXTRACTION_POSITION_INDEX[float(position)] for position in positions]
    for payload in feature_store:
        tensor = payload["tensor"]
        if tensor.shape[0] == 0:
            continue
        y = payload["labels"]
        local_rank_groups = payload["group_keys"]
        local_cv_groups = payload["cv_group_keys"]
        for local_pos_idx, src_pos_idx in enumerate(position_indices):
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
            x_raw = np.zeros((0, len(FULL_FEATURE_NAMES)), dtype=np.float64)
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

        out.append({
            "x_raw": x_raw,
            "x_rank": x_rank,
            "y": y,
            "groups": groups_cv,
        })
    return out


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


def _best_baseline_summary(
    *,
    x_raw: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> dict[str, Any]:
    best = {
        "signal_name": "tok_conf_prefix",
        "cv_auroc": float("-inf"),
        "n_valid_folds": 0,
    }
    for signal_name in PREFIX_SAFE_SIGNAL_NAMES:
        if signal_name not in FIXED_FEATURE_NAMES:
            continue
        score_col = FEATURE_TO_INDEX[signal_name]
        score_vec = x_raw[:, score_col]
        cv_auc, n_folds = _cv_auroc_baseline(
            scores=score_vec,
            y=y,
            groups=groups,
            n_splits=n_splits,
        )
        if np.isfinite(cv_auc) and cv_auc > float(best["cv_auroc"]):
            best = {
                "signal_name": str(signal_name),
                "cv_auroc": float(cv_auc),
                "n_valid_folds": int(n_folds),
            }
    return best


def _fit_fixed_rr_route(
    *,
    x_raw: np.ndarray,
    x_rank: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    position: float,
    domain_name: str,
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

    best_svd: dict[str, Any] = {"cv_auroc": float("-inf")}
    candidate_scores: dict[tuple[int, float, bool, str], list[float]] = {}
    for train_idx, test_idx in folds:
        y_train = y[train_idx]
        y_test = y[test_idx]
        if np.unique(y_train).shape[0] < 2 or np.unique(y_test).shape[0] < 2:
            continue

        x_train = x_rep[train_idx]
        x_test = x_rep[test_idx]
        max_rank = min(int(max(SEARCH_RANKS)), int(x_train.shape[1]), int(x_train.shape[0] - 1))
        if max_rank < 1:
            continue

        from sklearn.decomposition import TruncatedSVD
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler(with_mean=True, with_std=True)
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
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
        n_valid_folds = int(len(values))
        if cv_auc > float(best_svd["cv_auroc"]):
            best_svd = {
                "cv_auroc": float(cv_auc),
                "n_valid_folds": int(n_valid_folds),
                "rank": int(rank),
                "c_value": float(c_value),
                "whiten": bool(whiten),
                "class_weight": str(class_weight),
            }

    if not np.isfinite(float(best_svd["cv_auroc"])):
        raise RuntimeError(f"{domain_name}@{_pct_label(position)} found no valid SVD candidate")

    model = _fit_svd_lr_model(
        x=x_rep,
        y=y,
        rank=int(best_svd["rank"]),
        c_value=float(best_svd["c_value"]),
        whiten=bool(best_svd["whiten"]),
        class_weight_name=str(best_svd["class_weight"]),
        random_state=int(random_state),
    )
    if model is None:
        raise RuntimeError(f"{domain_name}@{_pct_label(position)} full-fit SVD failed")

    return {
        "route_type": "svd",
        "cv_auroc": float(best_svd["cv_auroc"]),
        "n_valid_folds": int(best_svd["n_valid_folds"]),
        "family_name": FIXED_FAMILY_NAME,
        "representation": FIXED_REPRESENTATION,
        "rank": int(best_svd["rank"]),
        "c_value": float(best_svd["c_value"]),
        "whiten": bool(best_svd["whiten"]),
        "class_weight": str(best_svd["class_weight"]),
        "feature_names": list(FIXED_FEATURE_NAMES),
        "feature_indices": list(FIXED_FEATURE_INDICES),
        "baseline_signal_name": str(best_baseline["signal_name"]),
        "baseline_cv_auroc": float(best_baseline["cv_auroc"]),
        "training_position": float(position),
        "training_scope": str(domain_name),
        "model": model,
    }


def _train_domain_anchor_routes(
    *,
    domain_name: str,
    tables: list[dict[str, np.ndarray]],
    positions: tuple[float, ...],
    n_splits: int,
    random_state: int,
    fit_workers: int,
) -> dict[float, dict[str, Any]]:
    routes: dict[float, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(fit_workers))) as executor:
        future_map = {}
        for pos_idx, position in enumerate(positions):
            future = executor.submit(
                _fit_fixed_rr_route,
                x_raw=tables[pos_idx]["x_raw"],
                x_rank=tables[pos_idx]["x_rank"],
                y=tables[pos_idx]["y"],
                groups=tables[pos_idx]["groups"],
                position=float(position),
                domain_name=str(domain_name),
                n_splits=int(n_splits),
                random_state=int(random_state),
            )
            future_map[future] = float(position)

        for future in as_completed(future_map):
            position = future_map[future]
            route = future.result()
            routes[float(position)] = route
            print(
                f"[train] domain={domain_name:<7s} pos={_pct_label(position):>4s} "
                f"route=svd auc={route['cv_auroc']:.4f} rep={route['representation']} "
                f"rank={route['rank']} c={route['c_value']:.2f} "
                f"baseline={route['baseline_signal_name']}:{route['baseline_cv_auroc']:.4f}",
                flush=True,
            )
    return routes


def _expand_anchor_routes(anchor_routes: dict[float, dict[str, Any]]) -> list[dict[str, Any]]:
    return [anchor_routes[float(OFFICIAL_SLOT_TO_ANCHOR[float(position)])] for position in EARLY_STOP_POSITIONS]


def _build_domain_bundle(
    *,
    method_id: str,
    domain_name: str,
    routes: dict[float, dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    return {
        "bundle_version": str(method_id),
        "created_at_utc": _now_utc(),
        "feature_names": list(FULL_FEATURE_NAMES),
        "positions": list(EARLY_STOP_POSITIONS),
        "anchor_positions": list(ANCHOR_POSITIONS),
        "protocol": dict(protocol),
        "domains": {
            str(domain_name): {
                "routes": _expand_anchor_routes(routes),
            }
        },
    }


def _build_ms_bundle(
    *,
    method_id: str,
    math_routes: dict[float, dict[str, Any]],
    science_routes: dict[float, dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    return {
        "bundle_version": str(method_id),
        "created_at_utc": _now_utc(),
        "feature_names": list(FULL_FEATURE_NAMES),
        "positions": list(EARLY_STOP_POSITIONS),
        "anchor_positions": list(ANCHOR_POSITIONS),
        "protocol": dict(protocol),
        "domains": {
            "math": {"routes": _expand_anchor_routes(math_routes)},
            "science": {"routes": _expand_anchor_routes(science_routes)},
        },
    }


def _collect_required_features(*bundles: dict[str, Any]) -> set[str]:
    required = set(FIXED_FEATURE_NAMES)
    required.add("tok_conf_prefix")
    for bundle in bundles:
        for domain_bundle in bundle.get("domains", {}).values():
            for route in domain_bundle.get("routes", []):
                if route["route_type"] == "baseline":
                    required.add(str(route["signal_name"]))
                else:
                    required.update(str(name) for name in route.get("feature_names", []))
    return required


def _summarise_feature_store(feature_store: list[dict[str, Any]]) -> dict[str, Any]:
    per_cache = []
    total_samples = 0
    total_problems = 0
    for payload in sorted(feature_store, key=lambda item: str(item["cache_key"])):
        problems = int(len(payload["problem_ids"]))
        samples = int(payload["samples"])
        total_samples += samples
        total_problems += problems
        per_cache.append({
            "cache_key": str(payload["cache_key"]),
            "source_name": str(payload["source_name"]),
            "dataset_name": str(payload["dataset_name"]),
            "domain": str(payload["domain"]),
            "n_problems": problems,
            "n_samples": samples,
        })
    return {
        "num_caches": int(len(per_cache)),
        "total_problems": int(total_problems),
        "total_samples": int(total_samples),
        "per_cache": per_cache,
    }


def _route_summary(routes: dict[float, dict[str, Any]]) -> dict[str, Any]:
    return {
        _pct_label(position): summarise_route(route)
        for position, route in sorted(routes.items(), key=lambda item: float(item[0]))
    }


def _baseline_bundle_result(
    *,
    method_name: str,
    feature_store: list[dict[str, Any]],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    return evaluate_method_from_feature_store(
        method_name=str(method_name),
        feature_store=feature_store,
        position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
        score_fn=make_svd_bundle_score_fn(bundle),
    )


def _render_route_table(title: str, route_summary: dict[str, Any]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Anchor | CV AUROC | Baseline | Baseline CV | Rank | C | Whiten | Class Weight |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for anchor_label, route in route_summary.items():
        lines.append(
            "| {anchor} | {cv_auc:.4f} | {baseline} | {baseline_cv:.4f} | {rank} | {c_value:.2f} | {whiten} | {class_weight} |".format(
                anchor=anchor_label,
                cv_auc=float(route["cv_auroc"]),
                baseline=str(route["baseline_signal_name"]),
                baseline_cv=float(route["baseline_cv_auroc"]),
                rank=int(route["rank"]),
                c_value=float(route["c_value"]),
                whiten="yes" if bool(route["whiten"]) else "no",
                class_weight=str(route["class_weight"]),
            )
        )
    lines.append("")
    return lines


def _write_registry(path: Path, summary: dict[str, Any]) -> None:
    registry: dict[str, Any]
    if path.exists():
        registry = json.loads(path.read_text(encoding="utf-8"))
    else:
        registry = {"family": "es_svd", "methods": []}

    registry["family"] = "es_svd"
    registry["updated_at_utc"] = _now_utc()
    existing_methods = {
        str(item.get("method_id")): dict(item)
        for item in registry.get("methods", [])
        if isinstance(item, dict) and item.get("method_id")
    }
    upserts = [
        {
            "method_id": METHOD_IDS["math"],
            "kind": "single_domain_bundle",
            "domain": "math",
            "representation": FIXED_REPRESENTATION,
            "anchors": [int(round(float(v) * 100.0)) for v in ANCHOR_POSITIONS],
            "model_path": summary["artifacts"]["models"]["math"],
        },
        {
            "method_id": METHOD_IDS["science"],
            "kind": "single_domain_bundle",
            "domain": "science",
            "representation": FIXED_REPRESENTATION,
            "anchors": [int(round(float(v) * 100.0)) for v in ANCHOR_POSITIONS],
            "model_path": summary["artifacts"]["models"]["science"],
        },
        {
            "method_id": METHOD_IDS["combined"],
            "kind": "multi_domain_bundle",
            "domains": ["math", "science"],
            "representation": FIXED_REPRESENTATION,
            "anchors": [int(round(float(v) * 100.0)) for v in ANCHOR_POSITIONS],
            "model_path": summary["artifacts"]["models"]["combined"],
            "summary_path": summary["artifacts"]["summary_json"],
            "eval_path": summary["artifacts"]["eval_json"],
            "doc_path": summary["artifacts"]["doc_md"],
        },
    ]
    for item in upserts:
        existing_methods[str(item["method_id"])] = item

    order = {
        METHOD_IDS["math"]: 10,
        METHOD_IDS["science"]: 20,
        METHOD_IDS["combined"]: 30,
        "es_svd_coding_rr_r1": 40,
    }
    registry["methods"] = sorted(
        existing_methods.values(),
        key=lambda item: (order.get(str(item.get("method_id")), 999), str(item.get("method_id"))),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_results_doc(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ES SVD MS RR R1",
        "",
        "## Naming",
        "",
        "- `es_svd`: EarlyStop SVD family.",
        "- `math` / `science` / `ms`: covered domain(s).",
        "- `rr`: raw+rank representation only.",
        "- `r1`: first version of the split-domain retraining.",
        "- `prefix` / `p10` are intentionally dropped from the name; training protocol still uses four anchors: `10/40/70/100`.",
        "",
        "## Feature Spec",
        "",
        f"- `representation`：`{summary['feature_spec']['representation']}`。",
        f"- `feature family`：`{summary['feature_spec']['family_name']}`。",
        f"- `included features`：`{', '.join(summary['feature_spec']['feature_names'])}`。",
        f"- `excluded features`：`{', '.join(summary['feature_spec']['excluded_feature_names'])}`。",
        "- `row feature`: no numeric row features are used; only `has_rows_bank` is retained as an availability flag.",
        "",
        "## Protocol",
        "",
        f"- `main cache root`：`{summary['protocol']['main_cache_root']}`。",
        f"- `extra cache root`：`{summary['protocol']['extra_cache_root']}`。",
        f"- `holdout split`：`{summary['protocol']['train_ratio']}/{summary['protocol']['holdout_ratio']}`。",
        f"- `holdout unit`：按 `dataset + problem_id` 做跨 root 一致切分，`split_seed={summary['protocol']['split_seed']}`。",
        f"- `anchors`：`{', '.join(str(v) for v in summary['protocol']['anchor_positions_pct'])}`。",
        "- `routing policy`：训练与最终 bundle 均不允许 baseline / single-feature route；baseline 只做对照。",
        "- `coding`：本轮故意不包含 coding routes，以避免把旧 baseline routing 混进新版本。",
        "",
        "## Artifacts",
        "",
        f"- `math model`：`{summary['artifacts']['models']['math']}`。",
        f"- `science model`：`{summary['artifacts']['models']['science']}`。",
        f"- `ms model`：`{summary['artifacts']['models']['combined']}`。",
        f"- `summary json`：`{summary['artifacts']['summary_json']}`。",
        f"- `eval json`：`{summary['artifacts']['eval_json']}`。",
        "",
        "## Validate",
        "",
    ]

    math_rows = [
        summary["validate"]["math"]["baselines"]["tok_conf_prefix_mean_v1"],
        summary["validate"]["math"]["baselines"]["earlystop_svd_lowrank_lr_v1"],
        summary["validate"]["math"]["baselines"]["earlystop_prefix10_svd_round1"],
        summary["validate"]["math"]["candidate"],
    ]
    science_rows = [
        summary["validate"]["science"]["baselines"]["tok_conf_prefix_mean_v1"],
        summary["validate"]["science"]["baselines"]["earlystop_svd_lowrank_lr_v1"],
        summary["validate"]["science"]["baselines"]["earlystop_prefix10_svd_round1"],
        summary["validate"]["science"]["candidate"],
    ]
    combined_rows = [
        summary["validate"]["combined_noncoding"]["baselines"]["tok_conf_prefix_mean_v1"],
        summary["validate"]["combined_noncoding"]["baselines"]["earlystop_svd_lowrank_lr_v1"],
        summary["validate"]["combined_noncoding"]["baselines"]["earlystop_prefix10_svd_round1"],
        summary["validate"]["combined_noncoding"]["candidate"],
    ]
    lines.extend(_render_method_table("math holdout", math_rows))
    lines.extend(_render_cache_table(summary["validate"]["math"]["candidate"]))
    lines.extend(_render_route_table("math full-fit anchor routes", summary["fullfit"]["math"]["route_summary"]))
    lines.extend(_render_method_table("science holdout", science_rows))
    lines.extend(_render_cache_table(summary["validate"]["science"]["candidate"]))
    lines.extend(_render_route_table("science full-fit anchor routes", summary["fullfit"]["science"]["route_summary"]))
    lines.extend(_render_method_table("combined noncoding holdout", combined_rows))
    lines.extend(_render_cache_table(summary["validate"]["combined_noncoding"]["candidate"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evaluate_domain_block(
    *,
    domain_name: str,
    holdout_store: list[dict[str, Any]],
    candidate_bundle: dict[str, Any],
    v1_bundle: dict[str, Any],
    legacy_prefix10_bundle: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate": evaluate_method_from_feature_store(
            method_name=METHOD_IDS[domain_name],
            feature_store=holdout_store,
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_svd_bundle_score_fn(candidate_bundle),
        ),
        "baselines": {
            "tok_conf_prefix_mean_v1": evaluate_method_from_feature_store(
                method_name="tok_conf_prefix_mean_v1",
                feature_store=holdout_store,
                position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
                score_fn=make_tok_conf_score_fn(),
            ),
            "earlystop_svd_lowrank_lr_v1": _baseline_bundle_result(
                method_name="earlystop_svd_lowrank_lr_v1",
                feature_store=holdout_store,
                bundle=v1_bundle,
            ),
            "earlystop_prefix10_svd_round1": _baseline_bundle_result(
                method_name="earlystop_prefix10_svd_round1",
                feature_store=holdout_store,
                bundle=legacy_prefix10_bundle,
            ),
        },
    }


def _resolve_path(raw_path: str) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def main() -> None:
    ap = argparse.ArgumentParser(description="Train split-domain EarlyStop SVD models without prefix naming")
    ap.add_argument("--main-cache-root", default="MUI_HUB/cache", help="Primary labeled cache root")
    ap.add_argument("--extra-cache-root", default="MUI_HUB/cache_train", help="Extra labeled cache root")
    ap.add_argument("--holdout-split", type=float, default=0.15, help="Holdout ratio; 0.15 means 85/15")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--feature-workers", type=int, default=8)
    ap.add_argument("--fit-workers", type=int, default=8)
    ap.add_argument("--feature-chunk-problems", type=int, default=DEFAULT_FEATURE_CHUNK_PROBLEMS)
    ap.add_argument("--feature-cache-dir", default="results/cache/es_svd_ms_rr_r1")
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--max-problems-per-cache", type=int, default=0, help="0 means all problems")
    ap.add_argument("--out-math-model", default="models/ml_selectors/es_svd_math_rr_r1.pkl")
    ap.add_argument("--out-science-model", default="models/ml_selectors/es_svd_science_rr_r1.pkl")
    ap.add_argument("--out-combined-model", default="models/ml_selectors/es_svd_ms_rr_r1.pkl")
    ap.add_argument("--out-summary", default="results/scans/earlystop/es_svd_ms_rr_r1_summary.json")
    ap.add_argument("--out-eval", default="results/scans/earlystop/es_svd_ms_rr_r1_eval.json")
    ap.add_argument("--out-doc", default="docs/ES_SVD_MS_RR_R1.md")
    ap.add_argument("--registry-path", default="SVDomain/registry.json")
    args = ap.parse_args()

    main_cache_root = _resolve_path(str(args.main_cache_root))
    extra_cache_root = _resolve_path(str(args.extra_cache_root))
    max_problems_per_cache = None if int(args.max_problems_per_cache) <= 0 else int(args.max_problems_per_cache)
    feature_cache_dir = None if str(args.feature_cache_dir).strip().lower() in {"", "none", "off"} else (REPO_ROOT / str(args.feature_cache_dir)).resolve()

    v1_bundle = load_earlystop_svd_bundle(REPO_ROOT / "models/ml_selectors/earlystop_svd_lowrank_lr_v1.pkl")
    legacy_prefix10_bundle = load_earlystop_svd_bundle(REPO_ROOT / "models/ml_selectors/earlystop_prefix10_svd_round1.pkl")
    required_features = _collect_required_features(v1_bundle, legacy_prefix10_bundle)

    main_store, main_cache_path, main_cache_status = _load_or_build_qualified_feature_store(
        source_name="cache",
        cache_root=main_cache_root,
        positions=EXTRACTION_POSITIONS,
        required_feature_names=required_features,
        max_problems_per_cache=max_problems_per_cache,
        feature_workers=int(args.feature_workers),
        chunk_problems=int(args.feature_chunk_problems),
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_feature_cache),
    )
    extra_store, extra_cache_path, extra_cache_status = _load_or_build_qualified_feature_store(
        source_name="cache_train",
        cache_root=extra_cache_root,
        positions=EXTRACTION_POSITIONS,
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
        for domain_name in DOMAINS
    }

    split_packs: dict[str, dict[str, Any]] = {}
    splitfit_routes: dict[str, dict[float, dict[str, Any]]] = {}
    fullfit_routes: dict[str, dict[float, dict[str, Any]]] = {}
    splitfit_bundles: dict[str, dict[str, Any]] = {}
    fullfit_bundles: dict[str, dict[str, Any]] = {}
    protocol_core = {
        "main_cache_root": str(main_cache_root),
        "extra_cache_root": str(extra_cache_root),
        "train_ratio": "85",
        "holdout_ratio": "15",
        "holdout_split": float(args.holdout_split),
        "split_seed": int(args.split_seed),
        "n_splits": int(args.n_splits),
        "random_state": int(args.random_state),
        "feature_workers": int(args.feature_workers),
        "fit_workers": int(args.fit_workers),
        "feature_chunk_problems": int(args.feature_chunk_problems),
        "max_problems_per_cache": None if max_problems_per_cache is None else int(max_problems_per_cache),
        "anchor_positions": [float(v) for v in ANCHOR_POSITIONS],
        "anchor_positions_pct": [int(round(float(v) * 100.0)) for v in ANCHOR_POSITIONS],
        "official_slot_to_anchor": {str(int(round(float(k) * 100.0))): int(round(float(v) * 100.0)) for k, v in OFFICIAL_SLOT_TO_ANCHOR.items()},
        "coding_included": False,
    }

    for domain_name in DOMAINS:
        store = domain_stores[domain_name]
        holdout_problem_map, holdout_problem_summary = _build_holdout_problem_map(
            store,
            holdout_split=float(args.holdout_split),
            split_seed=int(args.split_seed),
        )
        train_store, holdout_store, domain_full_store = _split_feature_store(
            store,
            holdout_problem_map=holdout_problem_map,
        )
        split_packs[domain_name] = {
            "train_store": train_store,
            "holdout_store": holdout_store,
            "full_store": domain_full_store,
            "holdout_problem_summary": holdout_problem_summary,
        }

        train_tables = _build_domain_training_tables(train_store, ANCHOR_POSITIONS)
        splitfit_routes[domain_name] = _train_domain_anchor_routes(
            domain_name=f"{domain_name}_splitfit",
            tables=train_tables,
            positions=ANCHOR_POSITIONS,
            n_splits=int(args.n_splits),
            random_state=int(args.random_state),
            fit_workers=int(args.fit_workers),
        )
        splitfit_bundles[domain_name] = _build_domain_bundle(
            method_id=f"{METHOD_IDS[domain_name]}_splitfit",
            domain_name=domain_name,
            routes=splitfit_routes[domain_name],
            protocol=protocol_core,
        )

        full_tables = _build_domain_training_tables(domain_full_store, ANCHOR_POSITIONS)
        fullfit_routes[domain_name] = _train_domain_anchor_routes(
            domain_name=f"{domain_name}_fullfit",
            tables=full_tables,
            positions=ANCHOR_POSITIONS,
            n_splits=int(args.n_splits),
            random_state=int(args.random_state),
            fit_workers=int(args.fit_workers),
        )
        fullfit_bundles[domain_name] = _build_domain_bundle(
            method_id=METHOD_IDS[domain_name],
            domain_name=domain_name,
            routes=fullfit_routes[domain_name],
            protocol=protocol_core,
        )

    splitfit_ms_bundle = _build_ms_bundle(
        method_id=f"{METHOD_IDS['combined']}_splitfit",
        math_routes=splitfit_routes["math"],
        science_routes=splitfit_routes["science"],
        protocol=protocol_core,
    )
    fullfit_ms_bundle = _build_ms_bundle(
        method_id=METHOD_IDS["combined"],
        math_routes=fullfit_routes["math"],
        science_routes=fullfit_routes["science"],
        protocol=protocol_core,
    )

    validate_math = _evaluate_domain_block(
        domain_name="math",
        holdout_store=split_packs["math"]["holdout_store"],
        candidate_bundle=splitfit_bundles["math"],
        v1_bundle=v1_bundle,
        legacy_prefix10_bundle=legacy_prefix10_bundle,
    )
    validate_science = _evaluate_domain_block(
        domain_name="science",
        holdout_store=split_packs["science"]["holdout_store"],
        candidate_bundle=splitfit_bundles["science"],
        v1_bundle=v1_bundle,
        legacy_prefix10_bundle=legacy_prefix10_bundle,
    )
    combined_holdout_store = split_packs["math"]["holdout_store"] + split_packs["science"]["holdout_store"]
    validate_combined = {
        "candidate": evaluate_method_from_feature_store(
            method_name=METHOD_IDS["combined"],
            feature_store=combined_holdout_store,
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_svd_bundle_score_fn(splitfit_ms_bundle),
        ),
        "baselines": {
            "tok_conf_prefix_mean_v1": evaluate_method_from_feature_store(
                method_name="tok_conf_prefix_mean_v1",
                feature_store=combined_holdout_store,
                position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
                score_fn=make_tok_conf_score_fn(),
            ),
            "earlystop_svd_lowrank_lr_v1": _baseline_bundle_result(
                method_name="earlystop_svd_lowrank_lr_v1",
                feature_store=combined_holdout_store,
                bundle=v1_bundle,
            ),
            "earlystop_prefix10_svd_round1": _baseline_bundle_result(
                method_name="earlystop_prefix10_svd_round1",
                feature_store=combined_holdout_store,
                bundle=legacy_prefix10_bundle,
            ),
        },
    }

    out_math_model = REPO_ROOT / str(args.out_math_model)
    out_science_model = REPO_ROOT / str(args.out_science_model)
    out_combined_model = REPO_ROOT / str(args.out_combined_model)
    out_summary = REPO_ROOT / str(args.out_summary)
    out_eval = REPO_ROOT / str(args.out_eval)
    out_doc = REPO_ROOT / str(args.out_doc)
    registry_path = REPO_ROOT / str(args.registry_path)

    save_earlystop_svd_bundle(fullfit_bundles["math"], out_math_model)
    save_earlystop_svd_bundle(fullfit_bundles["science"], out_science_model)
    save_earlystop_svd_bundle(fullfit_ms_bundle, out_combined_model)

    summary = {
        "method_family": "es_svd",
        "method_ids": dict(METHOD_IDS),
        "created_at_utc": _now_utc(),
        "naming": {
            "removed_prefix_from_name": True,
            "rr_means": FIXED_REPRESENTATION,
            "anchor_protocol_omitted_from_name": [int(round(float(v) * 100.0)) for v in ANCHOR_POSITIONS],
        },
        "protocol": dict(protocol_core),
        "feature_spec": {
            "family_name": FIXED_FAMILY_NAME,
            "representation": FIXED_REPRESENTATION,
            "feature_names": list(FIXED_FEATURE_NAMES),
            "excluded_feature_names": list(FIXED_EXCLUDED_FEATURES),
            "uses_numeric_row_features": False,
            "uses_has_rows_bank_flag": True,
            "reflection_threshold": float(DEFAULT_REFLECTION_THRESHOLD),
        },
        "search_space": {
            "ranks": list(SEARCH_RANKS),
            "c_values": list(SEARCH_C_VALUES),
            "whiten": list(SEARCH_WHITEN),
            "class_weight": list(SEARCH_CLASS_WEIGHT),
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
        "validate": {
            "math": validate_math,
            "science": validate_science,
            "combined_noncoding": validate_combined,
        },
        "fullfit": {
            "math": {
                "saved_model": _display_path(out_math_model),
                "route_summary": _route_summary(fullfit_routes["math"]),
            },
            "science": {
                "saved_model": _display_path(out_science_model),
                "route_summary": _route_summary(fullfit_routes["science"]),
            },
            "combined_noncoding": {
                "saved_model": _display_path(out_combined_model),
            },
        },
        "artifacts": {
            "models": {
                "math": _display_path(out_math_model),
                "science": _display_path(out_science_model),
                "combined": _display_path(out_combined_model),
            },
            "summary_json": _display_path(out_summary),
            "eval_json": _display_path(out_eval),
            "doc_md": _display_path(out_doc),
            "registry_json": _display_path(registry_path),
        },
    }
    eval_payload = {
        "protocol": summary["protocol"],
        "feature_spec": summary["feature_spec"],
        "validate": summary["validate"],
        "artifacts": summary["artifacts"],
    }

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_eval.parent.mkdir(parents=True, exist_ok=True)
    out_eval.write_text(json.dumps(eval_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_results_doc(out_doc, summary)
    _write_registry(registry_path, summary)

    print("[done] artifacts", flush=True)
    print(f"  math model     : {_display_path(out_math_model)}", flush=True)
    print(f"  science model  : {_display_path(out_science_model)}", flush=True)
    print(f"  combined model : {_display_path(out_combined_model)}", flush=True)
    print(f"  summary json   : {_display_path(out_summary)}", flush=True)
    print(f"  eval json      : {_display_path(out_eval)}", flush=True)
    print(f"  doc            : {_display_path(out_doc)}", flush=True)
    print(f"  registry       : {_display_path(registry_path)}", flush=True)


if __name__ == "__main__":
    main()
