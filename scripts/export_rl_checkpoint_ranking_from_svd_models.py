#!/usr/bin/env python3
"""Export Task-3 checkpoint-ranking submissions from slot-100 SVD verifiers."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.ops.earlystop import EARLY_STOP_POSITIONS, discover_cache_entries
from nad.ops.earlystop_svd import load_earlystop_svd_bundle
from scripts.export_earlystop_svd_submission import _load_or_build_feature_store
from scripts.run_earlystop_prefix10_svd_round1 import make_svd_bundle_score_fn


DEFAULT_RL_CACHE_ROOT = Path("/home/jovyan/public-ro/NAD_RL/math5000RL_neuron_analysis/cache")
DEFAULT_MODEL_PATHS = (
    "models/ml_selectors/earlystop_svd_lowrank_lr_v1.pkl",
    "models/ml_selectors/earlystop_svd_problem_centered_round1_cap8.pkl",
    "models/ml_selectors/earlystop_prefix10_svd_round1.pkl",
)
OFFICIAL_CHECKPOINTS = (
    "base",
    "step-100",
    "step-200",
    "step-300",
    "step-400",
    "step-500",
    "step-600",
    "step-700",
    "step-800",
    "step-900",
    "step-1000",
)
CHECKPOINT_ORDER = {name: idx for idx, name in enumerate(OFFICIAL_CHECKPOINTS)}


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _parse_csv(raw: str) -> tuple[str, ...]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one CSV token")
    return tuple(values)


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _official_checkpoint_name(model_dir_name: str) -> str:
    if model_dir_name == "Qwen3-4B-Base_base":
        return "base"
    match = re.fullmatch(r"Qwen3-4B-Base_math7500-step-(\d+)", model_dir_name)
    if match is None:
        raise ValueError(f"Unrecognized RL checkpoint directory: {model_dir_name}")
    step = int(match.group(1))
    name = f"step-{step}"
    if name not in CHECKPOINT_ORDER:
        raise ValueError(f"Unexpected checkpoint step: {model_dir_name}")
    return name


def _parse_statistic(result: Any) -> float | None:
    value = getattr(result, "statistic", result)
    if isinstance(value, tuple):
        value = value[0]
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _collect_required_features_for_math_slot(bundle: dict[str, Any], slot_index: int) -> set[str]:
    route = bundle["domains"]["math"]["routes"][int(slot_index)]
    required: set[str] = set()
    if route["route_type"] == "baseline":
        required.add(str(route["signal_name"]))
    else:
        required.update(str(name) for name in route["feature_names"])
    return required


def _slot_threshold_for_math(bundle: dict[str, Any], slot_index: int) -> float:
    route = bundle["domains"]["math"]["routes"][int(slot_index)]
    return float(route.get("reflection_threshold", bundle.get("reflection_threshold", 0.30)))


def _validate_checkpoint_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("task") != "checkpoint_ranking":
        raise ValueError(f"Expected task='checkpoint_ranking', got {payload.get('task')!r}")
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("scores must be an object")
    if set(scores.keys()) != set(OFFICIAL_CHECKPOINTS):
        missing = sorted(set(OFFICIAL_CHECKPOINTS) - set(scores.keys()))
        extra = sorted(set(scores.keys()) - set(OFFICIAL_CHECKPOINTS))
        raise ValueError(f"Checkpoint keys mismatch | missing={missing} extra={extra}")
    for checkpoint_name in OFFICIAL_CHECKPOINTS:
        value = scores[checkpoint_name]
        if not np.isfinite(float(value)):
            raise ValueError(f"Non-finite score for checkpoint: {checkpoint_name}")
    return {
        "num_checkpoints": len(scores),
        "min_score": float(min(float(v) for v in scores.values())),
        "max_score": float(max(float(v) for v in scores.values())),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _rank_checkpoint_names(score_map: dict[str, float]) -> list[str]:
    return sorted(
        OFFICIAL_CHECKPOINTS,
        key=lambda name: (-float(score_map[name]), int(CHECKPOINT_ORDER[name])),
    )


def _checkpoint_metrics(
    *,
    predicted_scores: dict[str, float],
    true_accuracies: dict[str, float],
) -> dict[str, Any]:
    pred_vec = np.asarray([float(predicted_scores[name]) for name in OFFICIAL_CHECKPOINTS], dtype=np.float64)
    true_vec = np.asarray([float(true_accuracies[name]) for name in OFFICIAL_CHECKPOINTS], dtype=np.float64)

    pred_ranked = _rank_checkpoint_names(predicted_scores)
    true_ranked = _rank_checkpoint_names(true_accuracies)

    metrics = {
        "spearman_rho": _parse_statistic(spearmanr(pred_vec, true_vec)),
        "pearson_r": _parse_statistic(pearsonr(pred_vec, true_vec)),
        "kendall_tau": _parse_statistic(kendalltau(pred_vec, true_vec)),
        "top1_hit": int(pred_ranked[0] == true_ranked[0]),
        "top3_hit": int(len(set(pred_ranked[:3]) & set(true_ranked[:3]))),
        "predicted_rank_order": pred_ranked,
        "true_rank_order": true_ranked,
    }
    return metrics


def _evaluate_checkpoint_payload(
    *,
    payload: dict[str, Any],
    slot_index: int,
    score_fn,
) -> dict[str, Any]:
    tensor = np.asarray(payload["tensor"], dtype=np.float64)
    labels = np.asarray(payload["labels"], dtype=np.int32)
    if tensor.ndim != 3 or tensor.shape[1] != 1:
        raise ValueError(
            f"Expected tensor shape [n_runs, 1, n_features], got {tuple(tensor.shape)} "
            f"for cache={payload['cache_key']}"
        )

    score_parts: list[np.ndarray] = []
    problem_offsets = [int(v) for v in payload["problem_offsets"]]
    for problem_idx, _problem_id in enumerate(payload["problem_ids"]):
        start = problem_offsets[problem_idx]
        end = problem_offsets[problem_idx + 1]
        if end <= start:
            continue
        x_raw = tensor[start:end, 0, :]
        problem_scores = np.asarray(score_fn(payload["domain"], int(slot_index), x_raw), dtype=np.float64)
        if problem_scores.shape[0] != max(0, end - start):
            raise ValueError(
                f"Per-problem score/width mismatch for cache={payload['cache_key']} "
                f"problem_idx={problem_idx}: scores={problem_scores.shape[0]} width={max(0, end - start)}"
            )
        score_parts.append(problem_scores)

    scores = np.concatenate(score_parts).astype(np.float64, copy=False) if score_parts else np.zeros((0,), dtype=np.float64)
    if scores.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Score/label shape mismatch for cache={payload['cache_key']}: "
            f"scores={scores.shape[0]} labels={labels.shape[0]}"
        )

    cache_model_name = str(payload["cache_key"]).split("/", 1)[0]
    checkpoint_name = _official_checkpoint_name(cache_model_name)
    return {
        "checkpoint_name": checkpoint_name,
        "cache_key": str(payload["cache_key"]),
        "n_problems": int(len(payload["problem_ids"])),
        "n_samples": int(labels.shape[0]),
        "mean_confidence": float(np.mean(scores)) if scores.size else 0.0,
        "score_std": float(np.std(scores)) if scores.size else 0.0,
        "true_accuracy": float(np.mean(labels)) if labels.size else 0.0,
    }


def _build_submission_payload(*, method_name: str, checkpoint_scores: dict[str, float]) -> dict[str, Any]:
    ordered_scores = {name: float(checkpoint_scores[name]) for name in OFFICIAL_CHECKPOINTS}
    return {
        "task": "checkpoint_ranking",
        "method_name": str(method_name),
        "scores": ordered_scores,
    }


def _discover_rl_entries(rl_cache_root: Path) -> list[Any]:
    entries = discover_cache_entries(rl_cache_root)
    if len(entries) != len(OFFICIAL_CHECKPOINTS):
        raise ValueError(
            f"Expected {len(OFFICIAL_CHECKPOINTS)} RL checkpoints under {rl_cache_root}, "
            f"found {len(entries)}"
        )
    seen: set[str] = set()
    for entry in entries:
        checkpoint_name = _official_checkpoint_name(str(entry.model_name))
        if checkpoint_name in seen:
            raise ValueError(f"Duplicate checkpoint discovered: {checkpoint_name}")
        seen.add(checkpoint_name)
    missing = sorted(set(OFFICIAL_CHECKPOINTS) - seen)
    if missing:
        raise ValueError(f"Missing RL checkpoints: {missing}")
    return entries


def _run_one_model(
    *,
    model_path: Path,
    slot_index: int,
    bundle: dict[str, Any],
    required_features: set[str],
    threshold: float,
    feature_cache_path: Path | None,
    feature_cache_status: str,
    feature_store: list[dict[str, Any]],
    out_dir: Path,
    summary_dir: Path,
) -> dict[str, Any]:
    position_value = float(EARLY_STOP_POSITIONS[int(slot_index)])
    score_fn = make_svd_bundle_score_fn(bundle)

    checkpoint_rows: list[dict[str, Any]] = []
    checkpoint_scores: dict[str, float] = {}
    true_accuracies: dict[str, float] = {}
    for payload in feature_store:
        row = _evaluate_checkpoint_payload(payload=payload, slot_index=slot_index, score_fn=score_fn)
        checkpoint_name = str(row["checkpoint_name"])
        checkpoint_rows.append(row)
        checkpoint_scores[checkpoint_name] = float(row["mean_confidence"])
        true_accuracies[checkpoint_name] = float(row["true_accuracy"])

    if set(checkpoint_scores.keys()) != set(OFFICIAL_CHECKPOINTS):
        missing = sorted(set(OFFICIAL_CHECKPOINTS) - set(checkpoint_scores.keys()))
        extra = sorted(set(checkpoint_scores.keys()) - set(OFFICIAL_CHECKPOINTS))
        raise ValueError(f"Checkpoint evaluation incomplete for {model_path.name} | missing={missing} extra={extra}")

    method_name = f"{model_path.stem}__math5000rl_slot100_meanconf"
    submission_payload = _build_submission_payload(method_name=method_name, checkpoint_scores=checkpoint_scores)
    submission_stats = _validate_checkpoint_payload(submission_payload)

    submission_path = out_dir / f"{method_name}.json"
    eval_path = summary_dir / f"{method_name}_eval.json"
    metrics = _checkpoint_metrics(predicted_scores=checkpoint_scores, true_accuracies=true_accuracies)

    eval_payload = {
        "task": "checkpoint_ranking_eval",
        "method_name": method_name,
        "model_path": str(model_path),
        "slot_index": int(slot_index),
        "position": float(position_value),
        "score_aggregation": "mean_confidence",
        "math_reflection_threshold": float(threshold),
        "required_features": sorted(required_features),
        "feature_cache": {
            "status": str(feature_cache_status),
            "path": None if feature_cache_path is None else str(feature_cache_path),
        },
        "checkpoints": sorted(checkpoint_rows, key=lambda row: CHECKPOINT_ORDER[row["checkpoint_name"]]),
        "predicted_scores": {name: float(checkpoint_scores[name]) for name in OFFICIAL_CHECKPOINTS},
        "true_checkpoint_accuracy": {name: float(true_accuracies[name]) for name in OFFICIAL_CHECKPOINTS},
        "metrics": metrics,
        "submission_path": str(submission_path),
        "submission_validation": submission_stats,
    }

    _write_json(submission_path, submission_payload)
    _write_json(eval_path, eval_payload)
    return {
        "method_name": method_name,
        "model_path": str(model_path),
        "submission_path": str(submission_path),
        "eval_path": str(eval_path),
        "predicted_scores": {name: float(checkpoint_scores[name]) for name in OFFICIAL_CHECKPOINTS},
        "true_checkpoint_accuracy": {name: float(true_accuracies[name]) for name in OFFICIAL_CHECKPOINTS},
        "metrics": metrics,
        "feature_cache": {
            "status": str(feature_cache_status),
            "path": None if feature_cache_path is None else str(feature_cache_path),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export RL checkpoint-ranking submissions from slot-100 SVD models")
    ap.add_argument("--rl-cache-root", default=str(DEFAULT_RL_CACHE_ROOT))
    ap.add_argument("--model-paths", default=",".join(DEFAULT_MODEL_PATHS))
    ap.add_argument("--slot-index", type=int, default=9)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--feature-chunk-problems", type=int, default=8)
    ap.add_argument("--feature-cache-dir", default="results/cache/export_rl_checkpoint_ranking_from_svd_models")
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--out-dir", default="submission/CheckpointRanking")
    ap.add_argument("--summary-dir", default="results/scans/checkpoint_ranking")
    args = ap.parse_args()

    rl_cache_root = _resolve_path(str(args.rl_cache_root))
    if not rl_cache_root.exists():
        raise FileNotFoundError(f"RL cache root not found: {rl_cache_root}")
    if int(args.slot_index) != 9:
        raise ValueError("This export is defined for slot 100 only; use --slot-index 9")

    _ = _discover_rl_entries(rl_cache_root)

    model_paths = tuple(_resolve_path(raw) for raw in _parse_csv(args.model_paths))
    for model_path in model_paths:
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

    feature_cache_dir = None
    if str(args.feature_cache_dir).strip().lower() not in {"", "none", "off"}:
        feature_cache_dir = _resolve_path(str(args.feature_cache_dir))
    out_dir = _resolve_path(str(args.out_dir))
    summary_dir = _resolve_path(str(args.summary_dir))

    print(f"RL cache root   : {_display_path(rl_cache_root)}")
    print(f"Slot index      : {int(args.slot_index)} ({EARLY_STOP_POSITIONS[int(args.slot_index)] * 100:.0f}%)")
    print(f"Model count     : {len(model_paths)}")
    for model_path in model_paths:
        print(f"  - {_display_path(model_path)}")

    model_specs: list[dict[str, Any]] = []
    threshold_to_features: dict[float, set[str]] = {}
    for model_path in model_paths:
        bundle = load_earlystop_svd_bundle(model_path)
        threshold = _slot_threshold_for_math(bundle, int(args.slot_index))
        required_features = _collect_required_features_for_math_slot(bundle, int(args.slot_index))
        model_specs.append({
            "model_path": model_path,
            "bundle": bundle,
            "threshold": float(threshold),
            "required_features": set(required_features),
        })
        threshold_to_features.setdefault(float(threshold), set()).update(required_features)

    feature_store_by_threshold: dict[float, list[dict[str, Any]]] = {}
    feature_cache_info_by_threshold: dict[float, dict[str, Any]] = {}
    for threshold in sorted(threshold_to_features.keys()):
        union_features = threshold_to_features[threshold]
        print(
            f"\n[rl-checkpoint] build feature store threshold={threshold:.2f} "
            f"feature_count={len(union_features)}"
        )
        feature_store, feature_cache_path, feature_cache_status = _load_or_build_feature_store(
            cache_root=str(rl_cache_root),
            positions=(float(EARLY_STOP_POSITIONS[int(args.slot_index)]),),
            required_feature_names=union_features,
            max_problems=None,
            reflection_threshold=float(threshold),
            workers=int(args.workers),
            feature_chunk_problems=int(args.feature_chunk_problems),
            feature_cache_dir=feature_cache_dir,
            refresh_feature_cache=bool(args.refresh_feature_cache),
        )
        feature_store_by_threshold[float(threshold)] = feature_store
        feature_cache_info_by_threshold[float(threshold)] = {
            "path": None if feature_cache_path is None else feature_cache_path,
            "status": str(feature_cache_status),
        }

    results: list[dict[str, Any]] = []
    shared_truth: dict[str, float] | None = None
    for spec in model_specs:
        model_path = spec["model_path"]
        print(f"\n[rl-checkpoint] scoring model={_display_path(model_path)}")
        result = _run_one_model(
            model_path=model_path,
            slot_index=int(args.slot_index),
            bundle=spec["bundle"],
            required_features=spec["required_features"],
            threshold=float(spec["threshold"]),
            feature_cache_path=feature_cache_info_by_threshold[float(spec["threshold"])]["path"],
            feature_cache_status=feature_cache_info_by_threshold[float(spec["threshold"])]["status"],
            feature_store=feature_store_by_threshold[float(spec["threshold"])],
            out_dir=out_dir,
            summary_dir=summary_dir,
        )
        if shared_truth is None:
            shared_truth = dict(result["true_checkpoint_accuracy"])
        else:
            for checkpoint_name in OFFICIAL_CHECKPOINTS:
                prev = float(shared_truth[checkpoint_name])
                cur = float(result["true_checkpoint_accuracy"][checkpoint_name])
                if not np.isclose(prev, cur, atol=1e-12, rtol=0.0):
                    raise ValueError(
                        f"True checkpoint accuracy mismatch for {checkpoint_name}: "
                        f"{prev} vs {cur}"
                    )
        print(
            f"[rl-checkpoint] done  model={_display_path(model_path)} "
            f"spearman={result['metrics']['spearman_rho']} top1={result['metrics']['top1_hit']} "
            f"top3={result['metrics']['top3_hit']}"
        )
        results.append(result)

    comparison_path = summary_dir / "rl_checkpoint_ranking_svd_model_comparison.json"
    comparison_payload = {
        "task": "checkpoint_ranking_model_comparison",
        "rl_cache_root": str(rl_cache_root),
        "slot_index": int(args.slot_index),
        "position": float(EARLY_STOP_POSITIONS[int(args.slot_index)]),
        "score_aggregation": "mean_confidence",
        "official_checkpoints": list(OFFICIAL_CHECKPOINTS),
        "true_checkpoint_accuracy": None if shared_truth is None else {
            name: float(shared_truth[name]) for name in OFFICIAL_CHECKPOINTS
        },
        "models": results,
    }
    _write_json(comparison_path, comparison_payload)

    print(f"\nComparison file : {_display_path(comparison_path)}")
    for result in results:
        print(f"Submission      : {_display_path(Path(result['submission_path']))}")
        print(f"Eval            : {_display_path(Path(result['eval_path']))}")


if __name__ == "__main__":
    main()
