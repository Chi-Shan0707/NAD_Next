#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.ops.bestofn_extreme8 import (  # noqa: E402
    discover_cache_entries,
    validate_submission_payload,
    write_submission_payload,
)
from nad.ops.earlystop import EARLY_STOP_POSITIONS  # noqa: E402
from nad.ops.earlystop_svd import get_domain, load_earlystop_svd_bundle  # noqa: E402
from scripts.export_earlystop_svd_submission import _load_or_build_feature_store  # noqa: E402
from scripts.run_earlystop_prefix10_svd_round1 import make_svd_bundle_score_fn  # noqa: E402


DEFAULT_MODEL_PATH = REPO_ROOT / "models/ml_selectors/earlystop_prefix10_svd_round1b_cap8.pkl"
DEFAULT_OVERRIDE_JSON = REPO_ROOT / "submission/BestofN/extreme12/patches/extreme12_svm_bridge_bestofn_v1.json"
DEFAULT_OUT = (
    REPO_ROOT
    / "submission/BestofN/extreme12/patches/"
    / "extreme12_earlystop_prefix10_svd_round1b_cap8_slot100__svm_bridge_lcb.json"
)
DEFAULT_METHOD_NAME = "extreme12_earlystop_prefix10_svd_round1b_cap8_slot100__svm_bridge_lcb"
DEFAULT_OVERRIDE_CACHE_KEYS = ("DS-R1/lcb_v5", "Qwen3-4B/lcb_v5")
DEFAULT_CACHE_ROOT = Path("/home/jovyan/public-ro/MUI_HUB/cache_test")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _parse_csv(raw: str) -> tuple[str, ...]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    return tuple(values)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_required_features_for_slot(bundle: dict[str, Any], slot_index: int) -> set[str]:
    required: set[str] = set()
    for domain_bundle in bundle["domains"].values():
        route = domain_bundle["routes"][int(slot_index)]
        if route["route_type"] == "baseline":
            required.add(str(route["signal_name"]))
        else:
            required.update(str(name) for name in route["feature_names"])
    return required


def _problem_scores_from_payload(
    payload: dict[str, Any],
    *,
    slot_index: int,
    score_fn,
) -> dict[str, dict[str, float]]:
    problem_scores: dict[str, dict[str, float]] = {}
    tensor = payload["tensor"]
    sample_ids_all = payload["sample_ids"]
    problem_ids = payload["problem_ids"]
    problem_offsets = payload["problem_offsets"]

    for problem_idx, problem_id in enumerate(problem_ids):
        start = int(problem_offsets[problem_idx])
        end = int(problem_offsets[problem_idx + 1])
        x_raw = tensor[start:end, 0, :]
        sample_ids = sample_ids_all[start:end]
        scores = np.asarray(score_fn(payload["domain"], int(slot_index), x_raw), dtype=np.float64)
        problem_scores[str(problem_id)] = {
            str(sample_id): float(scores[row_idx])
            for row_idx, sample_id in enumerate(sample_ids.tolist())
        }
    return problem_scores


def _build_payload(
    scores: dict[str, dict[str, dict[str, float]]],
    *,
    method_name: str,
    source_method_name: str,
    slot_index: int,
    position_value: float,
    override_method_name: str | None,
    override_cache_keys: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "task": "best_of_n",
        "method_name": str(method_name),
        "scores": scores,
        "score_postprocess": {
            "source_task": "early_stop",
            "source_method_name": str(source_method_name),
            "extracted_slot_index": int(slot_index),
            "extracted_position": float(position_value),
            "override_bestofn_source": None if override_method_name is None else str(override_method_name),
            "override_cache_keys": list(override_cache_keys),
            "note": "direct best_of_n export from earlystop SVD model slot",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Best-of-N directly from an EarlyStop SVD bundle slot, with optional cache overrides")
    ap.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    ap.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    ap.add_argument("--slot-index", type=int, default=9, help="EarlyStop slot index to extract (default 9 = 100%%)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--method-name", default=DEFAULT_METHOD_NAME)
    ap.add_argument("--source-method-name", default="earlystop_prefix10_svd_round1b_cap8")
    ap.add_argument("--override-bestofn-json", default=str(DEFAULT_OVERRIDE_JSON), help="Best-of-N JSON used for optional cache overrides; use 'none' to disable")
    ap.add_argument("--override-cache-keys", default=",".join(DEFAULT_OVERRIDE_CACHE_KEYS), help="Comma-separated cache keys copied from --override-bestofn-json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--feature-chunk-problems", type=int, default=8)
    ap.add_argument("--feature-cache-dir", default="results/cache/export_bestofn_from_earlystop_svd_model", help="Directory for cached blind feature stores; use 'none' to disable")
    ap.add_argument("--refresh-feature-cache", action="store_true")
    args = ap.parse_args()

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    bundle = load_earlystop_svd_bundle(model_path)
    slot_index = int(args.slot_index)
    position_value = float(EARLY_STOP_POSITIONS[slot_index])
    required_features = _collect_required_features_for_slot(bundle, slot_index=slot_index)
    score_fn = make_svd_bundle_score_fn(bundle)
    override_cache_keys = _parse_csv(args.override_cache_keys)
    override_path_raw = str(args.override_bestofn_json).strip()
    use_override = override_path_raw.lower() not in {"", "none", "off"} and bool(override_cache_keys)

    feature_cache_dir = None if str(args.feature_cache_dir).strip().lower() in {"", "none", "off"} else (REPO_ROOT / str(args.feature_cache_dir)).resolve()
    feature_store, feature_cache_path, feature_cache_status = _load_or_build_feature_store(
        cache_root=str(args.cache_root),
        positions=(position_value,),
        required_feature_names=required_features,
        max_problems=None,
        workers=int(args.workers),
        feature_chunk_problems=int(args.feature_chunk_problems),
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_feature_cache),
        exclude_cache_keys=set(override_cache_keys) if use_override else None,
    )

    print(f"Loaded bundle      : {_display_path(model_path)}")
    print(f"Feature cache      : {feature_cache_status} | {feature_cache_path}")
    if use_override:
        print(f"Skipped feature extraction for override caches: {list(override_cache_keys)}")

    scores: dict[str, dict[str, dict[str, float]]] = {}
    for payload in feature_store:
        cache_key = str(payload["cache_key"])
        scores[cache_key] = _problem_scores_from_payload(payload, slot_index=slot_index, score_fn=score_fn)
        n_problems = len(scores[cache_key])
        n_samples = sum(len(v) for v in scores[cache_key].values())
        print(f"  [{cache_key}] domain={get_domain(payload['dataset_name'])} problems={n_problems} samples={n_samples}")

    override_method_name: str | None = None
    if use_override:
        override_path = Path(override_path_raw)
        if not override_path.is_absolute():
            override_path = REPO_ROOT / override_path
        override_payload = _load_json(override_path)
        expected_cache_keys = [entry.cache_key for entry in discover_cache_entries(args.cache_root)]
        validate_submission_payload(override_payload, expected_cache_keys=expected_cache_keys)
        override_scores = override_payload.get("scores")
        if not isinstance(override_scores, dict) or not override_scores:
            raise ValueError("override best_of_n payload scores must be a non-empty mapping")
        override_method_name = str(override_payload.get("method_name", ""))
        for cache_key in override_cache_keys:
            if cache_key not in override_scores:
                raise ValueError(f"Override payload missing cache key: {cache_key}")
            scores[cache_key] = override_scores[cache_key]
        print(f"Override caches    : {list(override_cache_keys)} from {_display_path(override_path)}")

    payload = _build_payload(
        scores,
        method_name=str(args.method_name),
        source_method_name=str(args.source_method_name),
        slot_index=slot_index,
        position_value=position_value,
        override_method_name=override_method_name,
        override_cache_keys=override_cache_keys,
    )
    expected_cache_keys = [entry.cache_key for entry in discover_cache_entries(args.cache_root)]
    summary = validate_submission_payload(payload, expected_cache_keys=expected_cache_keys)
    written = write_submission_payload(payload, out_path)
    print(f"Written BestofN    : {_display_path(written)} | {summary}")


if __name__ == "__main__":
    main()
