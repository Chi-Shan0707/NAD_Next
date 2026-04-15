#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.views.reader import CacheReader
from nad.ops.bestofn_extreme8 import (
    build_problem_groups,
    discover_cache_entries,
    validate_submission_payload,
    write_submission_payload,
)
from nad.ops.coding_features import TIER1_FEATURE_NAMES, extract_tier1_feature_matrix


DEFAULT_FEATURE_CACHE = REPO_ROOT / "results/cache/es_svd_ms_rr_r1/cache_all_547b9060debe139e.pkl"
DEFAULT_TRAIN_CACHE_ROOT = (
    REPO_ROOT
    / "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/livecodebench_v5/cache_neuron_output_1_act_no_rms_20251127_032808"
)
DEFAULT_TIER1_CACHE = REPO_ROOT / "results/cache/coding_dummies_tier1.pkl"
DEFAULT_BLIND_CACHE_ROOT = Path("/home/jovyan/public-ro/MUI_HUB/cache_test")
DEFAULT_BASE_SUBMISSION = (
    REPO_ROOT
    / "submission/BestofN/extreme12/patches/extreme12_earlystop_prefix10_svd_round1_slot100__svm_bridge_lcb.json"
)
DEFAULT_OUT_MODEL = REPO_ROOT / "models/ml_selectors/coding_improvement_v1_tier1_xgb_lcb_patch.pkl"
DEFAULT_OUT_SUMMARY = REPO_ROOT / "results/scans/bestofn_bridge/coding_improvement_v1_tier1_xgb_lcb_patch_summary.json"
DEFAULT_OUT = (
    REPO_ROOT
    / "submission/BestofN/extreme12/patches/"
    / "extreme12_earlystop_prefix10_svd_round1_slot100__svm_bridge_lcb__coding_improvement_v1_tier1_xgb_lcb_patch.json"
)
DEFAULT_METHOD_NAME = (
    "extreme12_earlystop_prefix10_svd_round1_slot100__svm_bridge_lcb"
    "__coding_improvement_v1_tier1_xgb_lcb_patch"
)
DEFAULT_TARGET_CACHE_KEYS = ("DS-R1/lcb_v5", "Qwen3-4B/lcb_v5")
DEFAULT_SEEDS = (42, 101, 29)
DEFAULT_XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": 4,
    "verbosity": 0,
}


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


def _load_training_payload(feature_cache: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with feature_cache.open("rb") as handle:
        payload = pickle.load(handle)
    feature_store = payload["feature_store"]
    for item in feature_store:
        if item.get("domain") != "coding":
            continue
        labels = np.asarray(item["labels"], dtype=np.int32)
        groups = np.asarray(item["group_keys"], dtype=object)
        sample_ids = np.asarray(item["sample_ids"], dtype=np.int64)
        return labels, groups, sample_ids, np.asarray(item["positions"], dtype=np.float64)
    raise ValueError(f"No coding domain found in feature store: {feature_cache}")


def _load_or_build_tier1_train(
    *,
    cache_root: Path,
    sample_ids: np.ndarray,
    tier1_cache_path: Path,
    refresh: bool,
) -> np.ndarray:
    if not refresh and tier1_cache_path.exists():
        with tier1_cache_path.open("rb") as handle:
            saved = pickle.load(handle)
        if np.array_equal(np.asarray(saved["sample_ids"], dtype=np.int64), sample_ids):
            print(f"[train] reusing tier1 cache={_display_path(tier1_cache_path)}", flush=True)
            return np.asarray(saved["x_tier1"], dtype=np.float64)
        print("[train] tier1 cache sample_ids mismatch; rebuilding", flush=True)

    print(f"[train] building tier1 features from {_display_path(cache_root)}", flush=True)
    reader = CacheReader(str(cache_root))
    x_tier1 = extract_tier1_feature_matrix(reader, sample_ids, verbose=True)
    tier1_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tier1_cache_path.open("wb") as handle:
        pickle.dump({"sample_ids": sample_ids, "x_tier1": x_tier1}, handle, protocol=4)
    print(f"[train] saved tier1 cache={_display_path(tier1_cache_path)}", flush=True)
    return x_tier1


def _load_xgb_classifier():
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("xgboost is required for coding_improvement_v1 export") from exc
    return XGBClassifier


def _safe_mean(values: list[float]) -> float:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return float(np.mean(arr)) if arr.size else float("nan")


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def _brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels.astype(np.float64)) ** 2))


def _logloss(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-7) -> float:
    clipped = np.clip(probs, eps, 1.0 - eps)
    y = labels.astype(np.float64)
    return float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)))


def _crossval_metrics(
    *,
    x: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    seeds: tuple[int, ...],
    n_splits: int,
    xgb_params: dict[str, Any],
) -> dict[str, float]:
    XGBClassifier = _load_xgb_classifier()
    splitter = GroupKFold(n_splits=int(n_splits))
    aurocs: list[float] = []
    auprcs: list[float] = []
    briers: list[float] = []
    loglosses: list[float] = []

    for train_idx, test_idx in splitter.split(x, labels, groups):
        y_train = labels[train_idx]
        y_test = labels[test_idx]
        if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
            continue
        fold_probs = np.zeros(len(test_idx), dtype=np.float64)
        for seed in seeds:
            clf = XGBClassifier(random_state=int(seed), **xgb_params)
            clf.fit(x[train_idx], y_train)
            fold_probs += np.asarray(clf.predict_proba(x[test_idx])[:, 1], dtype=np.float64)
        fold_probs /= float(len(seeds))
        aurocs.append(_auroc(fold_probs, y_test))
        auprcs.append(_auprc(fold_probs, y_test))
        briers.append(_brier(fold_probs, y_test))
        loglosses.append(_logloss(fold_probs, y_test))

    return {
        "auroc": _safe_mean(aurocs),
        "auprc": _safe_mean(auprcs),
        "brier": _safe_mean(briers),
        "logloss": _safe_mean(loglosses),
        "n_folds": int(len(aurocs)),
    }


def _fit_full_bundle(
    *,
    x_train: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    sample_ids: np.ndarray,
    seeds: tuple[int, ...],
    n_splits: int,
    xgb_params: dict[str, Any],
    feature_cache: Path,
    train_cache_root: Path,
    tier1_cache_path: Path,
    target_cache_keys: tuple[str, ...],
) -> dict[str, Any]:
    XGBClassifier = _load_xgb_classifier()
    crossval = _crossval_metrics(
        x=x_train,
        labels=labels,
        groups=groups,
        seeds=seeds,
        n_splits=n_splits,
        xgb_params=xgb_params,
    )
    models = []
    for seed in seeds:
        clf = XGBClassifier(random_state=int(seed), **xgb_params)
        clf.fit(x_train, labels)
        models.append({"seed": int(seed), "model": clf})

    return {
        "bundle_version": "coding_improvement_v1_tier1_xgb_lcb_patch_v1",
        "created_at_utc": _now_utc(),
        "feature_names": list(TIER1_FEATURE_NAMES),
        "models": models,
        "training_summary": {
            "feature_cache": str(feature_cache),
            "train_cache_root": str(train_cache_root),
            "tier1_cache_path": str(tier1_cache_path),
            "n_samples": int(len(labels)),
            "n_problems": int(len(np.unique(groups))),
            "pos_rate": float(np.mean(labels.astype(np.float64))),
            "sample_id_min": int(sample_ids.min()) if sample_ids.size else None,
            "sample_id_max": int(sample_ids.max()) if sample_ids.size else None,
            "target_cache_keys": list(target_cache_keys),
            "cv_metrics": crossval,
            "seeds": list(seeds),
            "xgb_params": dict(xgb_params),
        },
    }


def _save_bundle(bundle: dict[str, Any], out_model: Path) -> None:
    out_model.parent.mkdir(parents=True, exist_ok=True)
    with out_model.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=4)


def _load_bundle(model_path: Path) -> dict[str, Any]:
    with model_path.open("rb") as handle:
        bundle = pickle.load(handle)
    if "models" not in bundle or "feature_names" not in bundle:
        raise ValueError(f"Invalid coding improvement bundle: {model_path}")
    return bundle


def _predict_mean_probability(models: list[dict[str, Any]], x: np.ndarray) -> np.ndarray:
    probs = np.zeros(x.shape[0], dtype=np.float64)
    for item in models:
        clf = item["model"]
        probs += np.asarray(clf.predict_proba(x)[:, 1], dtype=np.float64)
    probs /= float(len(models))
    return probs


def _load_target_entries(cache_root: Path, target_cache_keys: tuple[str, ...]) -> list[Any]:
    entries = [entry for entry in discover_cache_entries(cache_root) if entry.cache_key in target_cache_keys]
    entries.sort(key=lambda item: item.cache_key)
    found = {entry.cache_key for entry in entries}
    missing = sorted(set(target_cache_keys) - found)
    if missing:
        raise ValueError(f"Missing blind cache entries for target cache keys: {missing}")
    return entries


def _problem_sort_key(problem_id: str) -> tuple[int, str]:
    try:
        return (0, f"{int(problem_id):09d}")
    except (TypeError, ValueError):
        return (1, str(problem_id))


def _score_blind_entry(entry: Any, bundle: dict[str, Any]) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    meta = json.loads((entry.cache_root / "meta.json").read_text(encoding="utf-8"))
    groups = build_problem_groups(meta)
    reader = CacheReader(str(entry.cache_root))
    sample_ids = np.arange(len(meta.get("samples", [])), dtype=np.int64)
    x_cache = extract_tier1_feature_matrix(reader, sample_ids, verbose=True)
    probs = _predict_mean_probability(bundle["models"], x_cache)
    if not np.all(np.isfinite(probs)):
        raise ValueError(f"Non-finite probabilities produced for {entry.cache_key}")

    problem_scores: dict[str, dict[str, float]] = {}
    for problem_id, group_sample_ids in sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0])):
        sample_arr = np.asarray(group_sample_ids, dtype=np.int64)
        problem_scores[str(problem_id)] = {
            str(int(sample_id)): float(prob)
            for sample_id, prob in zip(sample_arr.tolist(), probs[sample_arr].tolist())
        }

    stats = {
        "cache_key": entry.cache_key,
        "cache_root": str(entry.cache_root),
        "n_problems": int(len(problem_scores)),
        "n_samples": int(len(sample_ids)),
        "score_min": float(np.min(probs)) if probs.size else None,
        "score_max": float(np.max(probs)) if probs.size else None,
        "score_mean": float(np.mean(probs)) if probs.size else None,
        "score_std": float(np.std(probs)) if probs.size else None,
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
            raise ValueError(f"Patched scores missing target cache key: {cache_key}")
        patched["scores"][cache_key] = target_scores[cache_key]

    patched["method_name"] = str(method_name)
    score_postprocess = dict(patched.get("score_postprocess") or {})
    score_postprocess["override_bestofn_source"] = None
    score_postprocess["override_cache_keys"] = []
    score_postprocess["note"] = (
        "non-lcb caches inherited from slot100 earlystop base; "
        "lcb caches replaced by coding_improvement_v1 Tier-1 XGBoost probabilities"
    )
    score_postprocess["lcb_override"] = "coding_improvement_v1_tier1_xgb=Tier1XGBoost@raw_prob"
    score_postprocess["coding_patch"] = {
        "source_method_name": "coding_improvement_v1_tier1_xgb",
        "target_cache_keys": list(target_cache_keys),
        "note": "LCB caches replaced by Tier-1-only XGBoost blind scorer from docs/13_CODING_IMPROVEMENT_V1.md",
    }
    score_postprocess["coding_patch_params"] = {
        "base_submission": str(base_submission),
        "base_method_name": str(base_payload.get("method_name", "")),
        "model_path": str(out_model),
        "feature_names": list(bundle["feature_names"]),
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


def _write_summary(summary: dict[str, Any], out_summary: Path) -> None:
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Patch BestofN submission by replacing LCB caches with coding_improvement_v1 Tier-1 XGBoost scores"
    )
    ap.add_argument("--feature-cache", default=str(DEFAULT_FEATURE_CACHE))
    ap.add_argument("--train-cache-root", default=str(DEFAULT_TRAIN_CACHE_ROOT))
    ap.add_argument("--tier1-cache", default=str(DEFAULT_TIER1_CACHE))
    ap.add_argument("--blind-cache-root", default=str(DEFAULT_BLIND_CACHE_ROOT))
    ap.add_argument("--base-submission", default=str(DEFAULT_BASE_SUBMISSION))
    ap.add_argument("--target-cache-keys", default=",".join(DEFAULT_TARGET_CACHE_KEYS))
    ap.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--out-model", default=str(DEFAULT_OUT_MODEL))
    ap.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--method-name", default=DEFAULT_METHOD_NAME)
    ap.add_argument("--refresh-tier1-cache", action="store_true")
    ap.add_argument("--load-only", action="store_true", help="Skip training and reuse --out-model")
    args = ap.parse_args()

    feature_cache = _resolve_repo_path(args.feature_cache)
    train_cache_root = _resolve_repo_path(args.train_cache_root)
    tier1_cache = _resolve_repo_path(args.tier1_cache)
    blind_cache_root = Path(args.blind_cache_root)
    base_submission = _resolve_repo_path(args.base_submission)
    out_model = _resolve_repo_path(args.out_model)
    out_summary = _resolve_repo_path(args.out_summary)
    out_path = _resolve_repo_path(args.out)
    target_cache_keys = _parse_csv(args.target_cache_keys)
    seeds = _parse_int_csv(args.seeds)

    xgb_params = dict(DEFAULT_XGB_PARAMS)

    if args.load_only:
        print(f"[bundle] loading existing bundle={_display_path(out_model)}", flush=True)
        bundle = _load_bundle(out_model)
    else:
        print(f"[train] feature cache={_display_path(feature_cache)}", flush=True)
        labels, groups, sample_ids, positions = _load_training_payload(feature_cache)
        print(
            f"[train] samples={len(labels)} problems={len(np.unique(groups))} "
            f"pos_rate={float(np.mean(labels.astype(np.float64))):.4f} positions={positions.tolist()}",
            flush=True,
        )
        x_train = _load_or_build_tier1_train(
            cache_root=train_cache_root,
            sample_ids=sample_ids,
            tier1_cache_path=tier1_cache,
            refresh=bool(args.refresh_tier1_cache),
        )
        bundle = _fit_full_bundle(
            x_train=x_train,
            labels=labels,
            groups=groups,
            sample_ids=sample_ids,
            seeds=seeds,
            n_splits=int(args.n_splits),
            xgb_params=xgb_params,
            feature_cache=feature_cache,
            train_cache_root=train_cache_root,
            tier1_cache_path=tier1_cache,
            target_cache_keys=target_cache_keys,
        )
        _save_bundle(bundle, out_model)
        print(f"[bundle] saved={_display_path(out_model)}", flush=True)

    entries = _load_target_entries(blind_cache_root, target_cache_keys)
    target_scores: dict[str, dict[str, dict[str, float]]] = {}
    blind_stats: dict[str, Any] = {}
    for entry in entries:
        print(f"[blind] scoring {entry.cache_key} <- {_display_path(entry.cache_root)}", flush=True)
        scores, stats = _score_blind_entry(entry, bundle)
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
    _write_summary(summary, out_summary)

    print(f"[saved] submission={_display_path(written)}", flush=True)
    print(f"[saved] summary={_display_path(out_summary)}", flush=True)
    print(f"[saved] changed_cache_keys={changed}", flush=True)
    print(f"[saved] validation={validation}", flush=True)


if __name__ == "__main__":
    main()
