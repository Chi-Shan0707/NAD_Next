#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.views.reader import CacheReader
from nad.ops.accuracy import load_correctness_map
from nad.ops.bestofn_extreme8 import (
    discover_cache_entries,
    validate_submission_payload,
    write_submission_payload,
)
from nad.ops.coding_neuron_features import (
    build_activation_hybrid_feature_matrix,
    group_rank_matrix,
    load_or_build_layer_summary_cache,
)
from nad.ops.earlystop_svd import (
    FULL_FEATURE_NAMES,
    LEGACY_FULL_FEATURE_NAMES,
    extract_earlystop_signals_for_positions,
)
from nad.ops.grouped_ranking import evaluate_grouped_scores
from scripts.export_earlystop_svd_submission import _collect_required_features, _load_or_build_feature_store
from scripts.run_earlystop_prefix10_svd_round1 import _score_xraw_with_route
from SVDomain.experiments.run_coding_improvement_v2 import (
    DEFAULT_CACHE_ROOT,
    _load_or_build_code_v2_baseline_scores,
    _load_or_build_dist_matrices,
    _load_problem_groups,
)


DEFAULT_TRAIN_CACHE_ROOT = REPO_ROOT / DEFAULT_CACHE_ROOT
DEFAULT_TRAIN_FEATURE_ROOT = REPO_ROOT / "MUI_HUB/cache"
DEFAULT_BLIND_CACHE_ROOT = Path("/home/jovyan/public-ro/MUI_HUB/cache_test")
DEFAULT_BASE_SUBMISSION = (
    REPO_ROOT
    / "submission/BestofN/extreme12/patches/extreme12_earlystop_prefix10_svd_round1_slot100__svm_bridge_lcb.json"
)
DEFAULT_SLOT100_MODEL_NAMES = (
    "slot100_svd_code_domain_r1_focus20__hit1.pkl",
    "slot100_svd_code_domain_r1_focus20__pairwise.pkl",
    "slot100_svd_code_domain_r1_cap10__hit1.pkl",
    "slot100_svd_code_domain_r1_cap10__pairwise.pkl",
)
DEFAULT_TRAIN_LEGACY_CACHE = REPO_ROOT / "results/cache/es_svd_ms_rr_r1/cache_all_547b9060debe139e.pkl"
DEFAULT_BLIND_LEGACY_CACHE = (
    REPO_ROOT
    / "results/cache/export_earlystop_svd_submission_strongfeat_20260410/feature_store_all_ref030_18a73b5e30f1a00d.pkl"
)
DEFAULT_TARGET_CACHE_KEYS = ("DS-R1/lcb_v5", "Qwen3-4B/lcb_v5")
DEFAULT_FEATURE_CACHE_DIR = REPO_ROOT / "results/cache/coding_hybrid_bridge"
DEFAULT_TRAIN_LAYER_CACHE = REPO_ROOT / "results/cache/coding_v2_layer_features.pkl"
DEFAULT_VALIDATION_JSON = REPO_ROOT / "results/validation/coding_hybrid_bridge.full.json"
DEFAULT_OUT_MODEL = REPO_ROOT / "models/ml_selectors/coding_improvement_v2_hybrid_bridge.pkl"
DEFAULT_OUT_SUMMARY = REPO_ROOT / "results/scans/bestofn_bridge/coding_improvement_v2_hybrid_bridge_summary.json"
DEFAULT_OUT_DOC = REPO_ROOT / "docs/15_CODING_HYBRID_BRIDGE.md"
DEFAULT_OUT = (
    REPO_ROOT
    / "submission/BestofN/extreme12/patches/"
    / "extreme12_earlystop_prefix10_svd_round1_slot100__svm_bridge_lcb__coding_improvement_v2_hybrid_bridge_patch.json"
)
DEFAULT_METHOD_NAME = (
    "extreme12_earlystop_prefix10_svd_round1_slot100__svm_bridge_lcb"
    "__coding_improvement_v2_hybrid_bridge_patch"
)
DEFAULT_SEEDS = (42, 101, 29)
DEFAULT_XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 260,
    "min_child_weight": 6,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "n_jobs": 16,
    "verbosity": 0,
}
DEFAULT_REFLECTION_THRESHOLD = 0.30


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_csv(raw: str) -> tuple[str, ...]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return tuple(values)


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(raw).split(",") if item.strip())
    if not values:
        raise ValueError("Expected at least one integer seed")
    return values


def _resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _sanitize_cache_key(cache_key: str) -> str:
    return str(cache_key).replace("/", "__").replace(":", "_")


def _load_xgb_classifier():
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("xgboost is required for coding_improvement_v2 hybrid export") from exc
    return XGBClassifier


def _write_json(payload: dict[str, Any] | list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _problem_groups_to_array(problem_groups: dict[str, list[int]], *, n_total: int) -> np.ndarray:
    groups = np.empty(int(n_total), dtype=object)
    for problem_id, run_ids in problem_groups.items():
        groups[np.asarray(run_ids, dtype=np.int64)] = str(problem_id)
    return groups


def _load_training_labels(cache_root: Path) -> tuple[dict[str, list[int]], np.ndarray, np.ndarray, np.ndarray]:
    problem_groups = _load_problem_groups(str(cache_root))
    n_total = max(max(map(int, run_ids)) for run_ids in problem_groups.values()) + 1
    correctness = load_correctness_map(str(cache_root))
    labels = np.asarray([int(correctness.get(i, 0)) for i in range(n_total)], dtype=np.int32)
    sample_ids = np.arange(n_total, dtype=np.int64)
    group_labels = _problem_groups_to_array(problem_groups, n_total=n_total)
    return problem_groups, labels, sample_ids, group_labels


def _load_slot100_models(model_names: tuple[str, ...]) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for model_name in model_names:
        model_path = REPO_ROOT / "models/ml_selectors" / str(model_name)
        with model_path.open("rb") as handle:
            bundle = pickle.load(handle)
        route = bundle["domains"]["coding"]["routes"][0]
        feature_names = [str(name) for name in bundle["feature_names"]]
        models.append(
            {
                "name": str(model_name),
                "model_path": str(model_path),
                "route": route,
                "feature_names": feature_names,
                "feature_to_idx": {name: idx for idx, name in enumerate(feature_names)},
            }
        )
    return models


def _collect_slot100_required_features(slot100_models: list[dict[str, Any]]) -> set[str]:
    required: set[str] = set()
    for item in slot100_models:
        model_path = Path(item["model_path"])
        with model_path.open("rb") as handle:
            bundle = pickle.load(handle)
        required.update(_collect_required_features(bundle))
    return required


def _select_coding_payload(feature_store: list[dict[str, Any]]) -> dict[str, Any]:
    coding_payloads = [payload for payload in feature_store if str(payload.get("domain")) == "coding"]
    if len(coding_payloads) != 1:
        raise ValueError(f"Expected exactly one coding payload, found {len(coding_payloads)}")
    return coding_payloads[0]


def _cache_key_matches(found: str, target: str) -> bool:
    found_str = str(found).strip()
    target_str = str(target).strip()
    return found_str == target_str or found_str.endswith(f"/{target_str}") or found_str.endswith(target_str)


def _load_legacy_payload_from_cache(cache_path: Path, cache_key: str) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)
    feature_store = payload.get("feature_store")
    if not isinstance(feature_store, list):
        return None
    matches = [
        item
        for item in feature_store
        if str(item.get("domain")) == "coding" and _cache_key_matches(str(item.get("cache_key", "")), cache_key)
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _position_index(payload: dict[str, Any], target: float = 1.0) -> int:
    positions = [float(value) for value in payload.get("positions", [])]
    for idx, value in enumerate(positions):
        if abs(float(value) - float(target)) < 1e-8:
            return int(idx)
    raise ValueError(f"Position {target} not found in payload positions={positions}")


def _extract_extra_feature_chunk(
    cache_root: str,
    sample_ids: list[int],
    feature_names: tuple[str, ...],
    reflection_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    reader = CacheReader(str(cache_root))
    req = set(feature_names)
    X = np.zeros((len(sample_ids), len(feature_names)), dtype=np.float64)
    for row_idx, sample_id in enumerate(sample_ids):
        signal_map = extract_earlystop_signals_for_positions(
            reader=reader,
            run_id=int(sample_id),
            positions=(1.0,),
            required_features=req,
            reflection_threshold=float(reflection_threshold),
        )
        for col_idx, feature_name in enumerate(feature_names):
            X[row_idx, col_idx] = float(signal_map[feature_name][0])
    return np.asarray(sample_ids, dtype=np.int64), X


def _load_or_build_extra_feature_cache(
    *,
    cache_root: Path,
    sample_ids: np.ndarray,
    feature_names: tuple[str, ...],
    cache_path: Path,
    reflection_threshold: float,
    workers: int,
    refresh: bool,
    chunk_size: int = 512,
) -> np.ndarray:
    sample_ids_arr = np.asarray(sample_ids, dtype=np.int64).reshape(-1)
    if cache_path.exists() and not refresh:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        cached_ids = np.asarray(payload.get("sample_ids", np.zeros(0, dtype=np.int64)), dtype=np.int64)
        cached_names = tuple(str(name) for name in payload.get("feature_names", ()))
        if np.array_equal(cached_ids, sample_ids_arr) and cached_names == tuple(feature_names):
            return np.asarray(payload["X"], dtype=np.float64)

    X = np.zeros((len(sample_ids_arr), len(feature_names)), dtype=np.float64)
    if len(sample_ids_arr) == 0 or len(feature_names) == 0:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump(
                {
                    "sample_ids": sample_ids_arr,
                    "feature_names": list(feature_names),
                    "X": X,
                },
                handle,
                protocol=4,
            )
        return X

    futures = {}
    completed = 0
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        for start in range(0, len(sample_ids_arr), max(1, int(chunk_size))):
            end = min(len(sample_ids_arr), start + max(1, int(chunk_size)))
            chunk_ids = sample_ids_arr[start:end].tolist()
            future = executor.submit(
                _extract_extra_feature_chunk,
                str(cache_root),
                chunk_ids,
                tuple(feature_names),
                float(reflection_threshold),
            )
            futures[future] = len(chunk_ids)

        for future in as_completed(futures):
            chunk_sample_ids, chunk_X = future.result()
            X[np.asarray(chunk_sample_ids, dtype=np.int64)] = np.asarray(chunk_X, dtype=np.float64)
            completed += int(futures[future])
            print(f"[extra] {completed}/{len(sample_ids_arr)}", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(
            {
                "sample_ids": sample_ids_arr,
                "feature_names": list(feature_names),
                "X": X,
            },
            handle,
            protocol=4,
        )
    return X


def _load_fast_slot100_matrix(
    *,
    cache_root: Path,
    cache_key: str,
    problem_groups: dict[str, list[int]],
    required_earlystop_features: set[str],
    legacy_cache_path: Path | None,
    feature_cache_dir: Path,
    reflection_threshold: float,
    workers: int,
    refresh_extra_cache: bool,
) -> np.ndarray | None:
    legacy_payload = None if legacy_cache_path is None else _load_legacy_payload_from_cache(legacy_cache_path, cache_key)
    if legacy_payload is None:
        return None

    n_total = max(max(map(int, run_ids)) for run_ids in problem_groups.values()) + 1
    x_full = np.zeros((int(n_total), len(FULL_FEATURE_NAMES)), dtype=np.float64)

    pos_idx = _position_index(legacy_payload, target=1.0)
    legacy_rows = np.asarray(legacy_payload["tensor"], dtype=np.float64)[:, pos_idx, :]
    legacy_sample_ids = np.asarray(legacy_payload["sample_ids"], dtype=np.int64)
    x_full[legacy_sample_ids, : len(LEGACY_FULL_FEATURE_NAMES)] = legacy_rows

    extra_feature_names = tuple(
        sorted(str(name) for name in required_earlystop_features if str(name) not in set(LEGACY_FULL_FEATURE_NAMES))
    )
    extra_cache_path = feature_cache_dir / f"{_sanitize_cache_key(cache_key)}_slot100_extra.pkl"
    X_extra = _load_or_build_extra_feature_cache(
        cache_root=cache_root,
        sample_ids=np.arange(n_total, dtype=np.int64),
        feature_names=extra_feature_names,
        cache_path=extra_cache_path,
        reflection_threshold=float(reflection_threshold),
        workers=int(workers),
        refresh=bool(refresh_extra_cache),
    )
    for col_idx, feature_name in enumerate(extra_feature_names):
        feature_pos = FULL_FEATURE_NAMES.index(str(feature_name))
        x_full[:, feature_pos] = X_extra[:, col_idx]
    return x_full


def _score_xfull_with_slot100_models(
    x_full: np.ndarray,
    problem_groups: dict[str, list[int]],
    slot100_models: list[dict[str, Any]],
) -> tuple[np.ndarray, list[str]]:
    scores = np.zeros((x_full.shape[0], len(slot100_models)), dtype=np.float64)
    for problem_id in sorted(problem_groups.keys()):
        run_ids = np.asarray(problem_groups[problem_id], dtype=np.int64)
        x_local = np.asarray(x_full[run_ids], dtype=np.float64)
        for model_idx, item in enumerate(slot100_models):
            local_scores = _score_xraw_with_route(
                x_raw=x_local,
                route=item["route"],
                feature_to_idx=item["feature_to_idx"],
            )
            scores[run_ids, model_idx] = np.asarray(local_scores, dtype=np.float64)
    names = [f"slot100::{Path(item['name']).stem}" for item in slot100_models]
    return scores, names


def _score_payload_with_slot100_models(
    payload: dict[str, Any],
    slot100_models: list[dict[str, Any]],
    *,
    n_total: int,
) -> tuple[np.ndarray, list[str]]:
    tensor = np.asarray(payload["tensor"], dtype=np.float64)
    if tensor.ndim != 3 or tensor.shape[1] != 1:
        raise ValueError(f"Expected coding tensor shape (n, 1, d), got {tensor.shape}")
    sample_ids_all = np.asarray(payload["sample_ids"], dtype=np.int64)
    problem_offsets = np.asarray(payload["problem_offsets"], dtype=np.int64)
    x_raw_all = tensor[:, 0, :]

    scores = np.zeros((int(n_total), len(slot100_models)), dtype=np.float64)
    for problem_idx in range(len(problem_offsets) - 1):
        start = int(problem_offsets[problem_idx])
        end = int(problem_offsets[problem_idx + 1])
        local_sample_ids = sample_ids_all[start:end]
        x_raw = x_raw_all[start:end]
        for model_idx, item in enumerate(slot100_models):
            local_scores = _score_xraw_with_route(
                x_raw=x_raw,
                route=item["route"],
                feature_to_idx=item["feature_to_idx"],
            )
            scores[local_sample_ids, model_idx] = np.asarray(local_scores, dtype=np.float64)

    names = [f"slot100::{Path(item['name']).stem}" for item in slot100_models]
    return scores, names


def _build_slot100_feature_block(
    slot100_raw_scores: np.ndarray,
    slot100_names: list[str],
    problem_groups: dict[str, list[int]],
) -> tuple[np.ndarray, list[str]]:
    raw = np.asarray(slot100_raw_scores, dtype=np.float64)
    agg = np.column_stack(
        [
            raw.mean(axis=1),
            raw.std(axis=1),
            raw.max(axis=1),
            raw.min(axis=1),
            raw.max(axis=1) - raw.min(axis=1),
        ]
    ).astype(np.float64)
    agg_names = [
        "slot100::mean",
        "slot100::std",
        "slot100::max",
        "slot100::min",
        "slot100::max_minus_min",
    ]
    raw_rank = group_rank_matrix(raw, problem_groups)
    agg_rank = group_rank_matrix(agg, problem_groups)
    names = [
        *slot100_names,
        *agg_names,
        *[f"{name}_group_rank" for name in slot100_names],
        *[f"{name}_group_rank" for name in agg_names],
    ]
    X = np.column_stack([raw, agg, raw_rank, agg_rank]).astype(np.float64)
    return X, names


def _build_code_v2_feature_block(
    code_v2_scores: np.ndarray,
    problem_groups: dict[str, list[int]],
) -> tuple[np.ndarray, list[str]]:
    raw = np.asarray(code_v2_scores, dtype=np.float64).reshape(-1, 1)
    rank = group_rank_matrix(raw, problem_groups)
    X = np.column_stack([raw, rank]).astype(np.float64)
    return X, ["code_v2::score", "code_v2::score_group_rank"]


def _assemble_candidate_matrix(
    block_map: dict[str, tuple[np.ndarray, list[str]]],
    block_names: tuple[str, ...],
) -> tuple[np.ndarray, list[str]]:
    matrices: list[np.ndarray] = []
    feature_names: list[str] = []
    for block_name in block_names:
        if block_name not in block_map:
            raise KeyError(f"Missing feature block: {block_name}")
        X_block, names = block_map[block_name]
        matrices.append(np.asarray(X_block, dtype=np.float64))
        feature_names.extend(str(name) for name in names)
    X = np.column_stack(matrices).astype(np.float64) if matrices else np.zeros((0, 0), dtype=np.float64)
    return X, feature_names


def _fit_predict_grouped_scores(
    *,
    X: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    model_kind: str,
    seeds: tuple[int, ...],
    xgb_params: dict[str, Any],
) -> np.ndarray:
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(labels, dtype=np.int32)
    g_arr = np.asarray(groups, dtype=object)
    n_splits = min(int(n_splits), int(np.unique(g_arr).size))
    out = np.zeros(len(y_arr), dtype=np.float64)
    if n_splits < 2 or np.unique(y_arr).size < 2:
        return out

    splitter = GroupKFold(n_splits=n_splits)
    if model_kind == "xgb":
        XGBClassifier = _load_xgb_classifier()
    else:
        XGBClassifier = None

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_arr, y_arr, g_arr), start=1):
        y_train = y_arr[train_idx]
        y_test = y_arr[test_idx]
        if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
            continue

        if model_kind == "xgb":
            preds = np.zeros(len(test_idx), dtype=np.float64)
            for seed in seeds:
                clf = XGBClassifier(random_state=int(seed), **xgb_params)
                clf.fit(X_arr[train_idx], y_train)
                preds += np.asarray(clf.predict_proba(X_arr[test_idx])[:, 1], dtype=np.float64)
            preds /= float(len(seeds))
            out[test_idx] = preds
            continue

        if model_kind == "logreg":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_arr[train_idx])
            X_test = scaler.transform(X_arr[test_idx])
            clf = LogisticRegression(
                C=0.3,
                class_weight="balanced",
                max_iter=4000,
                random_state=int(fold_idx),
            )
            clf.fit(X_train, y_train)
            out[test_idx] = np.asarray(clf.predict_proba(X_test)[:, 1], dtype=np.float64)
            continue

        raise ValueError(f"Unknown model_kind={model_kind}")
    return out


def _fit_full_models(
    *,
    X: np.ndarray,
    labels: np.ndarray,
    model_kind: str,
    seeds: tuple[int, ...],
    xgb_params: dict[str, Any],
) -> list[dict[str, Any]]:
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(labels, dtype=np.int32)
    if model_kind == "xgb":
        XGBClassifier = _load_xgb_classifier()
        out: list[dict[str, Any]] = []
        for seed in seeds:
            clf = XGBClassifier(random_state=int(seed), **xgb_params)
            clf.fit(X_arr, y_arr)
            out.append({"model_kind": "xgb", "seed": int(seed), "model": clf})
        return out

    if model_kind == "logreg":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_arr)
        clf = LogisticRegression(C=0.3, class_weight="balanced", max_iter=4000, random_state=0)
        clf.fit(X_scaled, y_arr)
        return [{"model_kind": "logreg", "seed": 0, "scaler": scaler, "model": clf}]

    raise ValueError(f"Unknown model_kind={model_kind}")


def _predict_full_models(models: list[dict[str, Any]], X: np.ndarray) -> np.ndarray:
    X_arr = np.asarray(X, dtype=np.float64)
    out = np.zeros(X_arr.shape[0], dtype=np.float64)
    for item in models:
        if item["model_kind"] == "xgb":
            out += np.asarray(item["model"].predict_proba(X_arr)[:, 1], dtype=np.float64)
            continue
        if item["model_kind"] == "logreg":
            X_scaled = item["scaler"].transform(X_arr)
            out += np.asarray(item["model"].predict_proba(X_scaled)[:, 1], dtype=np.float64)
            continue
        raise ValueError(f"Unknown model_kind={item['model_kind']}")
    return out / float(len(models))


def _metric_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    uplift = row.get("uplift")
    pairwise = row.get("pairwise")
    win_rate = ((row.get("head_to_head") or {}).get("win_rate_non_ties"))
    return (
        float(row.get("top1", float("-inf"))),
        float(uplift) if uplift is not None else float("-inf"),
        float(pairwise) if pairwise is not None else float("-inf"),
        float(win_rate) if win_rate is not None else float("-inf"),
    )


def _build_problem_score_dict(
    problem_groups: dict[str, list[int]],
    scores: np.ndarray,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    scores_arr = np.asarray(scores, dtype=np.float64)
    for problem_id in sorted(problem_groups.keys()):
        run_ids = np.asarray(problem_groups[problem_id], dtype=np.int64)
        out[str(problem_id)] = {
            str(int(run_id)): float(scores_arr[int(run_id)])
            for run_id in run_ids.tolist()
        }
    return out


def _build_entry_artifacts(
    *,
    cache_root: str,
    entry_cache_root: Path,
    cache_key: str,
    slot100_models: list[dict[str, Any]],
    required_earlystop_features: set[str],
    legacy_cache_path: Path | None,
    feature_cache_dir: Path,
    reflection_threshold: float,
    workers: int,
    feature_chunk_problems: int,
    refresh_feature_cache: bool,
    refresh_layer_cache: bool,
    refresh_code_v2_cache: bool,
    include_code_v2: bool,
) -> dict[str, Any]:
    problem_groups = _load_problem_groups(str(entry_cache_root))
    n_total = max(max(map(int, run_ids)) for run_ids in problem_groups.values()) + 1
    sample_ids = np.arange(n_total, dtype=np.int64)
    cache_tag = _sanitize_cache_key(cache_key)

    x_fast = _load_fast_slot100_matrix(
        cache_root=entry_cache_root,
        cache_key=str(cache_key),
        problem_groups=problem_groups,
        required_earlystop_features=required_earlystop_features,
        legacy_cache_path=legacy_cache_path,
        feature_cache_dir=feature_cache_dir,
        reflection_threshold=float(reflection_threshold),
        workers=int(workers),
        refresh_extra_cache=bool(refresh_feature_cache),
    )
    if x_fast is not None:
        feature_cache_path = None if legacy_cache_path is None else str(legacy_cache_path)
        feature_cache_status = "legacy+extra"
        slot100_raw, slot100_names = _score_xfull_with_slot100_models(
            x_fast,
            problem_groups,
            slot100_models,
        )
    else:
        feature_store, feature_cache_path, feature_cache_status = _load_or_build_feature_store(
            cache_root=str(cache_root),
            positions=(1.0,),
            required_feature_names=required_earlystop_features,
            max_problems=None,
            reflection_threshold=float(reflection_threshold),
            workers=int(workers),
            feature_chunk_problems=int(feature_chunk_problems),
            feature_cache_dir=feature_cache_dir,
            refresh_feature_cache=bool(refresh_feature_cache),
            include_cache_keys={str(cache_key)},
            exclude_cache_keys=None,
        )
        payload = _select_coding_payload(feature_store)
        slot100_raw, slot100_names = _score_payload_with_slot100_models(
            payload,
            slot100_models,
            n_total=n_total,
        )
    slot100_block = _build_slot100_feature_block(slot100_raw, slot100_names, problem_groups)

    layer_cache_path = feature_cache_dir / f"{cache_tag}_layer_summary.pkl"
    if str(entry_cache_root.resolve()) == str(DEFAULT_TRAIN_CACHE_ROOT.resolve()):
        layer_cache_path = DEFAULT_TRAIN_LAYER_CACHE
    layer_payload = load_or_build_layer_summary_cache(
        str(entry_cache_root),
        cache_path=str(layer_cache_path),
        sample_ids=sample_ids,
        refresh=bool(refresh_layer_cache),
        verbose=bool(refresh_layer_cache),
    )
    activation_block = build_activation_hybrid_feature_matrix(layer_payload, problem_groups)

    code_v2_scores = None
    code_v2_block = None
    if include_code_v2:
        dist_cache_path = feature_cache_dir / f"{cache_tag}_dist_matrices.pkl"
        code_v2_cache_path = feature_cache_dir / f"{cache_tag}_code_v2_scores.pkl"
        dist_matrices = _load_or_build_dist_matrices(
            cache_root=str(entry_cache_root),
            problem_groups=problem_groups,
            cache_path=str(dist_cache_path),
            refresh=bool(refresh_code_v2_cache),
        )
        code_v2_scores = _load_or_build_code_v2_baseline_scores(
            str(entry_cache_root),
            problem_groups,
            dist_matrices=dist_matrices,
            cache_path=str(code_v2_cache_path),
            refresh=bool(refresh_code_v2_cache),
        )
        code_v2_block = _build_code_v2_feature_block(code_v2_scores, problem_groups)

    return {
        "problem_groups": problem_groups,
        "n_total": int(n_total),
        "sample_ids": sample_ids,
        "slot100_raw": slot100_raw,
        "slot100_names": slot100_names,
        "slot100_block": slot100_block,
        "activation_block": activation_block,
        "code_v2_scores": code_v2_scores,
        "code_v2_block": code_v2_block,
        "feature_cache_path": None if feature_cache_path is None else str(feature_cache_path),
        "feature_cache_status": str(feature_cache_status),
    }


def _build_candidate_rows(
    *,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    group_labels: np.ndarray,
    n_splits: int,
    block_map: dict[str, tuple[np.ndarray, list[str]]],
    slot100_raw: np.ndarray,
    slot100_names: list[str],
    problem_groups: dict[str, list[int]],
    code_v2_scores: np.ndarray | None,
    seeds: tuple[int, ...],
    xgb_params: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_scores = code_v2_scores
    code_v2_row = {
        "name": "code_v2_baseline",
        "kind": "baseline",
        "model_kind": None,
    }
    if baseline_scores is not None:
        ev = evaluate_grouped_scores(
            problem_groups,
            labels,
            baseline_scores,
            sample_ids=sample_ids,
            baseline_scores=None,
        )
        code_v2_row.update(
            {
                "top1": float(ev["top1_accuracy"]),
                "uplift": None,
                "pairwise": None if ev.get("pairwise_auc") is None else float(ev["pairwise_auc"]),
                "head_to_head": None,
            }
        )

    fixed_rows: list[dict[str, Any]] = []
    for model_idx, model_name in enumerate(slot100_names):
        raw_scores = slot100_raw[:, model_idx]
        ev = evaluate_grouped_scores(
            problem_groups,
            labels,
            raw_scores,
            sample_ids=sample_ids,
            baseline_scores=baseline_scores,
        )
        fixed_rows.append(
            {
                "name": str(model_name),
                "kind": "slot100_route",
                "model_kind": "route",
                "selected_slot100_model_name": str(model_name),
                "top1": float(ev["top1_accuracy"]),
                "uplift": None if ev.get("pass@1_uplift_abs") is None else float(ev["pass@1_uplift_abs"]),
                "pairwise": None if ev.get("pairwise_auc") is None else float(ev["pairwise_auc"]),
                "head_to_head": ev["head_to_head"],
            }
        )

    candidate_defs = [
        {"name": "meta_xgb_slot100", "model_kind": "xgb", "blocks": ("slot100",)},
        {"name": "meta_xgb_slot100_code_v2", "model_kind": "xgb", "blocks": ("slot100", "code_v2")},
        {"name": "meta_xgb_slot100_activation", "model_kind": "xgb", "blocks": ("slot100", "activation")},
        {
            "name": "meta_xgb_slot100_code_v2_activation",
            "model_kind": "xgb",
            "blocks": ("slot100", "code_v2", "activation"),
        },
        {
            "name": "meta_logreg_slot100_code_v2_activation",
            "model_kind": "logreg",
            "blocks": ("slot100", "code_v2", "activation"),
        },
    ]

    meta_rows: list[dict[str, Any]] = []
    for candidate in candidate_defs:
        if any(block_name not in block_map for block_name in candidate["blocks"]):
            continue
        X, feature_names = _assemble_candidate_matrix(block_map, tuple(candidate["blocks"]))
        cv_scores = _fit_predict_grouped_scores(
            X=X,
            labels=labels,
            groups=group_labels,
            n_splits=int(n_splits),
            model_kind=str(candidate["model_kind"]),
            seeds=seeds,
            xgb_params=xgb_params,
        )
        ev = evaluate_grouped_scores(
            problem_groups,
            labels,
            cv_scores,
            sample_ids=sample_ids,
            baseline_scores=baseline_scores,
        )
        meta_rows.append(
            {
                "name": str(candidate["name"]),
                "kind": "meta",
                "model_kind": str(candidate["model_kind"]),
                "feature_blocks": list(candidate["blocks"]),
                "feature_count": int(X.shape[1]),
                "feature_names": feature_names,
                "top1": float(ev["top1_accuracy"]),
                "uplift": None if ev.get("pass@1_uplift_abs") is None else float(ev["pass@1_uplift_abs"]),
                "pairwise": None if ev.get("pairwise_auc") is None else float(ev["pairwise_auc"]),
                "head_to_head": ev["head_to_head"],
            }
        )

    return code_v2_row, fixed_rows, meta_rows


def _save_bundle(bundle: dict[str, Any], out_model: Path) -> None:
    out_model.parent.mkdir(parents=True, exist_ok=True)
    with out_model.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=4)


def _load_bundle(model_path: Path) -> dict[str, Any]:
    with model_path.open("rb") as handle:
        bundle = pickle.load(handle)
    if "scorer_type" not in bundle or "training_summary" not in bundle:
        raise ValueError(f"Invalid coding hybrid bundle: {model_path}")
    return bundle


def _fit_selected_bundle(
    *,
    selected_row: dict[str, Any],
    labels: np.ndarray,
    block_map: dict[str, tuple[np.ndarray, list[str]]],
    slot100_models: list[dict[str, Any]],
    required_earlystop_features: set[str],
    seeds: tuple[int, ...],
    xgb_params: dict[str, Any],
    training_summary: dict[str, Any],
) -> dict[str, Any]:
    bundle: dict[str, Any] = {
        "bundle_version": "coding_improvement_v2_hybrid_bridge_v1",
        "created_at_utc": _now_utc(),
        "reflection_threshold": float(DEFAULT_REFLECTION_THRESHOLD),
        "slot100_models": [
            {
                "name": str(item["name"]),
                "model_path": str(item["model_path"]),
                "route": item["route"],
                "feature_names": list(item["feature_names"]),
                "feature_to_idx": dict(item["feature_to_idx"]),
            }
            for item in slot100_models
        ],
        "earlystop_required_feature_names": sorted(str(name) for name in required_earlystop_features),
        "training_summary": training_summary,
    }

    if selected_row["kind"] == "slot100_route":
        bundle["scorer_type"] = "slot100_route"
        bundle["selected_slot100_model_name"] = str(selected_row["selected_slot100_model_name"])
        return bundle

    if selected_row["kind"] != "meta":
        raise ValueError(f"Unsupported selected row kind={selected_row['kind']}")

    X, _ = _assemble_candidate_matrix(block_map, tuple(selected_row["feature_blocks"]))
    models = _fit_full_models(
        X=X,
        labels=labels,
        model_kind=str(selected_row["model_kind"]),
        seeds=seeds,
        xgb_params=xgb_params,
    )
    bundle.update(
        {
            "scorer_type": "meta_ensemble",
            "model_kind": str(selected_row["model_kind"]),
            "feature_blocks": list(selected_row["feature_blocks"]),
            "feature_names": list(selected_row["feature_names"]),
            "models": models,
        }
    )
    return bundle


def _score_entry_with_bundle(
    *,
    cache_root: str,
    entry: Any,
    bundle: dict[str, Any],
    legacy_cache_path: Path | None,
    feature_cache_dir: Path,
    workers: int,
    feature_chunk_problems: int,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    slot100_models = [
        {
            "name": str(item["name"]),
            "model_path": str(item["model_path"]),
            "route": item["route"],
            "feature_names": list(item["feature_names"]),
            "feature_to_idx": dict(item["feature_to_idx"]),
        }
        for item in bundle["slot100_models"]
    ]
    use_code_v2 = bool(bundle.get("scorer_type") == "meta_ensemble" and "code_v2" in bundle.get("feature_blocks", []))
    artifacts = _build_entry_artifacts(
        cache_root=str(cache_root),
        entry_cache_root=Path(entry.cache_root),
        cache_key=str(entry.cache_key),
        slot100_models=slot100_models,
        required_earlystop_features=set(bundle["earlystop_required_feature_names"]),
        legacy_cache_path=legacy_cache_path,
        feature_cache_dir=feature_cache_dir,
        reflection_threshold=float(bundle.get("reflection_threshold", DEFAULT_REFLECTION_THRESHOLD)),
        workers=int(workers),
        feature_chunk_problems=int(feature_chunk_problems),
        refresh_feature_cache=False,
        refresh_layer_cache=False,
        refresh_code_v2_cache=False,
        include_code_v2=use_code_v2,
    )

    slot100_raw = np.asarray(artifacts["slot100_raw"], dtype=np.float64)
    problem_groups = artifacts["problem_groups"]

    if bundle["scorer_type"] == "slot100_route":
        raw_names = artifacts["slot100_names"]
        target_name = str(bundle["selected_slot100_model_name"])
        try:
            model_idx = raw_names.index(target_name)
        except ValueError as exc:
            raise ValueError(f"Selected slot100 model not found in blind artifacts: {target_name}") from exc
        scores = slot100_raw[:, model_idx]
    else:
        block_map = {
            "slot100": artifacts["slot100_block"],
            "activation": artifacts["activation_block"],
        }
        if artifacts["code_v2_block"] is not None:
            block_map["code_v2"] = artifacts["code_v2_block"]
        X, feature_names = _assemble_candidate_matrix(block_map, tuple(bundle["feature_blocks"]))
        if list(feature_names) != list(bundle["feature_names"]):
            raise ValueError("Blind feature assembly mismatch with trained bundle")
        scores = _predict_full_models(bundle["models"], X)

    problem_scores = _build_problem_score_dict(problem_groups, scores)
    scores_arr = np.asarray(scores, dtype=np.float64)
    stats = {
        "cache_key": str(entry.cache_key),
        "cache_root": str(entry.cache_root),
        "n_problems": int(len(problem_groups)),
        "n_samples": int(len(scores_arr)),
        "score_min": float(np.min(scores_arr)) if scores_arr.size else None,
        "score_max": float(np.max(scores_arr)) if scores_arr.size else None,
        "score_mean": float(np.mean(scores_arr)) if scores_arr.size else None,
        "score_std": float(np.std(scores_arr)) if scores_arr.size else None,
    }
    return problem_scores, stats


def _build_patched_payload(
    *,
    base_payload: dict[str, Any],
    base_submission: Path,
    method_name: str,
    target_scores: dict[str, dict[str, dict[str, float]]],
    out_model: Path,
    bundle: dict[str, Any],
    target_cache_keys: tuple[str, ...],
) -> dict[str, Any]:
    patched = json.loads(json.dumps(base_payload))
    for cache_key in target_cache_keys:
        if cache_key not in patched.get("scores", {}):
            raise ValueError(f"Base submission missing target cache key: {cache_key}")
        if cache_key not in target_scores:
            raise ValueError(f"Target scores missing cache key: {cache_key}")
        patched["scores"][cache_key] = target_scores[cache_key]

    patched["method_name"] = str(method_name)
    score_postprocess = dict(patched.get("score_postprocess") or {})
    score_postprocess["override_bestofn_source"] = None
    score_postprocess["override_cache_keys"] = []
    score_postprocess["note"] = (
        "non-lcb caches inherited from slot100 earlystop base; "
        "lcb caches replaced by coding_improvement_v2 hybrid bridge scorer"
    )
    score_postprocess["lcb_override"] = "coding_improvement_v2_hybrid_bridge=slot100_projection+activation_summary(+code_v2)"
    score_postprocess["coding_patch"] = {
        "source_method_name": "coding_improvement_v2_hybrid_bridge",
        "target_cache_keys": list(target_cache_keys),
        "note": "LCB caches replaced by hybrid slot100 bridge scorer with activation-summary reranking.",
    }
    score_postprocess["coding_patch_params"] = {
        "base_submission": str(base_submission),
        "base_method_name": str(base_payload.get("method_name", "")),
        "model_path": str(out_model),
        "selected_row": dict(bundle["training_summary"]["selected_candidate"]),
        "training_summary": dict(bundle["training_summary"]),
    }
    patched["score_postprocess"] = score_postprocess
    return patched


def _changed_cache_keys(
    before: dict[str, dict[str, dict[str, float]]],
    after: dict[str, dict[str, dict[str, float]]],
) -> list[str]:
    changed = []
    for cache_key in sorted(set(before) | set(after)):
        if before.get(cache_key) != after.get(cache_key):
            changed.append(cache_key)
    return changed


def _write_markdown_report(summary: dict[str, Any], out_doc: Path) -> None:
    selected = summary["training_summary"]["selected_candidate"]
    fixed_rows = summary["training_summary"]["fixed_slot100_rows"]
    meta_rows = summary["training_summary"]["meta_candidate_rows"]
    lines = [
        "# 15 — Coding Hybrid Bridge",
        "",
        f"**Date**: {summary['created_at_utc']}  **Status**: completed",
        "",
        "## Verdict",
        "",
        f"- **Selected scorer**: `{selected['name']}` ({selected['kind']})",
        f"- **Top1**: `{selected['top1']:.4f}`",
        f"- **Uplift vs `code_v2`**: "
        + ("`n/a`" if selected.get("uplift") is None else f"`{selected['uplift']:+.4f}`"),
        f"- **Patch output**: `{summary['out_submission']}`",
        "",
        "## Best Fixed Slot100 Routes",
        "",
        "| Route | Top1 | Uplift vs `code_v2` | Pairwise |",
        "|---|---:|---:|---:|",
    ]
    for row in sorted(fixed_rows, key=_metric_sort_key, reverse=True):
        uplift = "—" if row.get("uplift") is None else f"{row['uplift']:+.4f}"
        pairwise = "—" if row.get("pairwise") is None else f"{row['pairwise']:.4f}"
        lines.append(f"| `{row['name']}` | {row['top1']:.4f} | {uplift} | {pairwise} |")

    lines.extend(
        [
            "",
            "## Meta Candidates",
            "",
            "| Candidate | Blocks | Top1 | Uplift vs `code_v2` | Pairwise |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in sorted(meta_rows, key=_metric_sort_key, reverse=True):
        uplift = "—" if row.get("uplift") is None else f"{row['uplift']:+.4f}"
        pairwise = "—" if row.get("pairwise") is None else f"{row['pairwise']:.4f}"
        blocks = ", ".join(row.get("feature_blocks", []))
        lines.append(f"| `{row['name']}` | `{blocks}` | {row['top1']:.4f} | {uplift} | {pairwise} |")

    lines.extend(
        [
            "",
            "## Blind Export Stats",
            "",
            "| Cache | Problems | Samples | Score mean | Score std |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for cache_key, stats in summary["blind_stats"].items():
        lines.append(
            f"| `{cache_key}` | {int(stats['n_problems'])} | {int(stats['n_samples'])} | "
            f"{stats['score_mean']:.6f} | {stats['score_std']:.6f} |"
        )

    out_doc.parent.mkdir(parents=True, exist_ok=True)
    out_doc.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Patch BestofN submission with coding_improvement_v2 hybrid bridge scores"
    )
    ap.add_argument("--train-cache-root", default=str(DEFAULT_TRAIN_CACHE_ROOT))
    ap.add_argument("--train-feature-root", default=str(DEFAULT_TRAIN_FEATURE_ROOT))
    ap.add_argument("--blind-cache-root", default=str(DEFAULT_BLIND_CACHE_ROOT))
    ap.add_argument("--base-submission", default=str(DEFAULT_BASE_SUBMISSION))
    ap.add_argument("--slot100-models", default=",".join(DEFAULT_SLOT100_MODEL_NAMES))
    ap.add_argument("--train-legacy-cache", default=str(DEFAULT_TRAIN_LEGACY_CACHE))
    ap.add_argument("--blind-legacy-cache", default=str(DEFAULT_BLIND_LEGACY_CACHE))
    ap.add_argument("--target-cache-keys", default=",".join(DEFAULT_TARGET_CACHE_KEYS))
    ap.add_argument("--feature-cache-dir", default=str(DEFAULT_FEATURE_CACHE_DIR))
    ap.add_argument("--validation-json", default=str(DEFAULT_VALIDATION_JSON))
    ap.add_argument("--out-model", default=str(DEFAULT_OUT_MODEL))
    ap.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    ap.add_argument("--out-doc", default=str(DEFAULT_OUT_DOC))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--method-name", default=DEFAULT_METHOD_NAME)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--xgb-jobs", type=int, default=int(DEFAULT_XGB_PARAMS["n_jobs"]))
    ap.add_argument("--feature-chunk-problems", type=int, default=8)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--refresh-layer-cache", action="store_true")
    ap.add_argument("--refresh-code-v2-cache", action="store_true")
    ap.add_argument("--load-only", action="store_true", help="Skip retraining and reuse --out-model")
    args = ap.parse_args()

    train_cache_root = _resolve_repo_path(args.train_cache_root)
    train_feature_root = _resolve_repo_path(args.train_feature_root)
    blind_cache_root = Path(args.blind_cache_root)
    base_submission = _resolve_repo_path(args.base_submission)
    feature_cache_dir = _resolve_repo_path(args.feature_cache_dir)
    train_legacy_cache = _resolve_repo_path(args.train_legacy_cache)
    blind_legacy_cache = _resolve_repo_path(args.blind_legacy_cache)
    validation_json = _resolve_repo_path(args.validation_json)
    out_model = _resolve_repo_path(args.out_model)
    out_summary = _resolve_repo_path(args.out_summary)
    out_doc = _resolve_repo_path(args.out_doc)
    out_path = _resolve_repo_path(args.out)
    slot100_model_names = _parse_csv(args.slot100_models)
    target_cache_keys = _parse_csv(args.target_cache_keys)
    seeds = _parse_int_csv(args.seeds)
    xgb_params = dict(DEFAULT_XGB_PARAMS)
    xgb_params["n_jobs"] = int(args.xgb_jobs)

    if args.load_only:
        print(f"[bundle] loading existing bundle={_display_path(out_model)}", flush=True)
        bundle = _load_bundle(out_model)
    else:
        print("[train] loading labels + groups", flush=True)
        problem_groups, labels, sample_ids, group_labels = _load_training_labels(train_cache_root)
        slot100_models = _load_slot100_models(slot100_model_names)
        required_earlystop_features = _collect_slot100_required_features(slot100_models)

        print("[train] building training artifacts", flush=True)
        train_artifacts = _build_entry_artifacts(
            cache_root=str(train_feature_root),
            entry_cache_root=train_cache_root,
            cache_key="DS-R1/lcb_v5",
            slot100_models=slot100_models,
            required_earlystop_features=required_earlystop_features,
            legacy_cache_path=train_legacy_cache,
            feature_cache_dir=feature_cache_dir,
            reflection_threshold=float(DEFAULT_REFLECTION_THRESHOLD),
            workers=int(args.workers),
            feature_chunk_problems=int(args.feature_chunk_problems),
            refresh_feature_cache=bool(args.refresh_feature_cache),
            refresh_layer_cache=bool(args.refresh_layer_cache),
            refresh_code_v2_cache=bool(args.refresh_code_v2_cache),
            include_code_v2=True,
        )

        block_map = {
            "slot100": train_artifacts["slot100_block"],
            "activation": train_artifacts["activation_block"],
        }
        if train_artifacts["code_v2_block"] is not None:
            block_map["code_v2"] = train_artifacts["code_v2_block"]

        print("[train] evaluating fixed and meta candidates", flush=True)
        code_v2_row, fixed_rows, meta_rows = _build_candidate_rows(
            labels=labels,
            sample_ids=sample_ids,
            group_labels=group_labels,
            n_splits=int(args.n_splits),
            block_map=block_map,
            slot100_raw=train_artifacts["slot100_raw"],
            slot100_names=train_artifacts["slot100_names"],
            problem_groups=problem_groups,
            code_v2_scores=train_artifacts["code_v2_scores"],
            seeds=seeds,
            xgb_params=xgb_params,
        )
        candidate_rows = [*fixed_rows, *meta_rows]
        selected_row = max(candidate_rows, key=_metric_sort_key)
        best_fixed = max(fixed_rows, key=_metric_sort_key) if fixed_rows else None
        best_meta = max(meta_rows, key=_metric_sort_key) if meta_rows else None

        validation_payload = {
            "created_at_utc": _now_utc(),
            "train_cache_root": str(train_cache_root),
            "code_v2_baseline": code_v2_row,
            "best_fixed_slot100": best_fixed,
            "best_meta_candidate": best_meta,
            "selected_candidate": selected_row,
            "fixed_slot100_rows": fixed_rows,
            "meta_candidate_rows": meta_rows,
        }
        _write_json(validation_payload, validation_json)
        print(f"[saved] validation={_display_path(validation_json)}", flush=True)

        training_summary = {
            "train_cache_root": str(train_cache_root),
            "train_feature_root": str(train_feature_root),
            "feature_cache_dir": str(feature_cache_dir),
            "n_samples": int(len(labels)),
            "n_problems": int(len(problem_groups)),
            "pos_rate": float(np.mean(labels.astype(np.float64))),
            "slot100_model_names": list(slot100_model_names),
            "required_earlystop_features": sorted(str(name) for name in required_earlystop_features),
            "feature_cache_path": train_artifacts["feature_cache_path"],
            "feature_cache_status": train_artifacts["feature_cache_status"],
            "train_legacy_cache": _display_path(train_legacy_cache),
            "blind_legacy_cache": _display_path(blind_legacy_cache),
            "selected_candidate": selected_row,
            "code_v2_baseline": code_v2_row,
            "best_fixed_slot100": best_fixed,
            "best_meta_candidate": best_meta,
            "fixed_slot100_rows": fixed_rows,
            "meta_candidate_rows": meta_rows,
            "seeds": list(seeds),
            "xgb_params": dict(xgb_params),
            "validation_json": _display_path(validation_json),
        }

        bundle = _fit_selected_bundle(
            selected_row=selected_row,
            labels=labels,
            block_map=block_map,
            slot100_models=slot100_models,
            required_earlystop_features=required_earlystop_features,
            seeds=seeds,
            xgb_params=xgb_params,
            training_summary=training_summary,
        )
        _save_bundle(bundle, out_model)
        print(f"[bundle] saved={_display_path(out_model)}", flush=True)

    entries = [entry for entry in discover_cache_entries(blind_cache_root) if entry.cache_key in target_cache_keys]
    entries.sort(key=lambda item: item.cache_key)
    found = {entry.cache_key for entry in entries}
    missing = sorted(set(target_cache_keys) - found)
    if missing:
        raise ValueError(f"Missing blind cache entries for target cache keys: {missing}")

    target_scores: dict[str, dict[str, dict[str, float]]] = {}
    blind_stats: dict[str, Any] = {}
    for entry in entries:
        print(f"[blind] scoring {entry.cache_key} <- {entry.cache_root}", flush=True)
        scores, stats = _score_entry_with_bundle(
            cache_root=str(blind_cache_root),
            entry=entry,
            bundle=bundle,
            legacy_cache_path=blind_legacy_cache,
            feature_cache_dir=feature_cache_dir,
            workers=int(args.workers),
            feature_chunk_problems=int(args.feature_chunk_problems),
        )
        target_scores[entry.cache_key] = scores
        blind_stats[entry.cache_key] = stats
        print(
            f"[blind] {entry.cache_key}: problems={stats['n_problems']} samples={stats['n_samples']} "
            f"score_mean={stats['score_mean']:.6f}",
            flush=True,
        )

    base_payload = json.loads(base_submission.read_text(encoding="utf-8"))
    expected_cache_keys = [entry.cache_key for entry in discover_cache_entries(blind_cache_root)]
    patched_payload = _build_patched_payload(
        base_payload=base_payload,
        base_submission=base_submission,
        method_name=str(args.method_name),
        target_scores=target_scores,
        out_model=out_model,
        bundle=bundle,
        target_cache_keys=target_cache_keys,
    )
    validation = validate_submission_payload(patched_payload, expected_cache_keys=expected_cache_keys)
    written = write_submission_payload(patched_payload, out_path)
    changed = _changed_cache_keys(base_payload.get("scores", {}), patched_payload.get("scores", {}))

    summary = {
        "created_at_utc": _now_utc(),
        "method_name": str(args.method_name),
        "base_submission": _display_path(base_submission),
        "base_method_name": str(base_payload.get("method_name", "")),
        "out_submission": _display_path(written),
        "out_model": _display_path(out_model),
        "target_cache_keys": list(target_cache_keys),
        "changed_cache_keys": changed,
        "validation": validation,
        "training_summary": dict(bundle["training_summary"]),
        "blind_stats": blind_stats,
    }
    _write_json(summary, out_summary)
    _write_markdown_report(summary, out_doc)

    print(f"[saved] submission={_display_path(written)}", flush=True)
    print(f"[saved] summary={_display_path(out_summary)}", flush=True)
    print(f"[saved] doc={_display_path(out_doc)}", flush=True)
    print(f"[saved] changed_cache_keys={changed}", flush=True)
    print(f"[saved] validation={validation}", flush=True)


if __name__ == "__main__":
    main()
