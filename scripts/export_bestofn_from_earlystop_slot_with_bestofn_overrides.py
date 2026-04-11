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
from nad.ops.earlystop import EARLY_STOP_POSITIONS, validate_earlystop_payload  # noqa: E402


DEFAULT_EARLYSTOP_JSON = REPO_ROOT / "submission/EarlyStop/earlystop_prefix10_svd_round1.json"
DEFAULT_OVERRIDE_JSON = REPO_ROOT / "submission/BestofN/extreme12/patches/extreme12_svm_bridge_bestofn_v1.json"
DEFAULT_OUT = (
    REPO_ROOT
    / "submission/BestofN/extreme12/patches/"
    / "extreme12_earlystop_prefix10_svd_round1_slot100__svm_bridge_lcb.json"
)
DEFAULT_METHOD_NAME = "extreme12_earlystop_prefix10_svd_round1_slot100__svm_bridge_lcb"
DEFAULT_OVERRIDE_CACHE_KEYS = ("DS-R1/lcb_v5", "Qwen3-4B/lcb_v5")
DEFAULT_CACHE_ROOT = Path("/home/jovyan/public-ro/MUI_HUB/cache_test")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _parse_csv(raw: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    text = str(raw).strip()
    if text.lower() in {"", "none", "off"}:
        if allow_empty:
            return ()
        raise ValueError("Expected at least one cache key")

    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        if allow_empty:
            return ()
        raise ValueError("Expected at least one cache key")
    return tuple(values)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_bestofn_scores_from_earlystop(
    earlystop_payload: dict[str, Any],
    *,
    slot_index: int,
) -> dict[str, dict[str, dict[str, float]]]:
    scores = earlystop_payload.get("scores")
    if not isinstance(scores, dict) or not scores:
        raise ValueError("earlystop payload scores must be a non-empty mapping")

    extracted: dict[str, dict[str, dict[str, float]]] = {}
    for cache_key, problem_map in scores.items():
        if not isinstance(problem_map, dict) or not problem_map:
            raise ValueError(f"{cache_key}: expected non-empty problem map")
        extracted[cache_key] = {}
        for problem_id, sample_map in problem_map.items():
            if not isinstance(sample_map, dict) or not sample_map:
                raise ValueError(f"{cache_key}/{problem_id}: expected non-empty sample map")
            extracted_problem: dict[str, float] = {}
            for sample_id, score_list in sample_map.items():
                if not isinstance(score_list, list):
                    raise ValueError(
                        f"{cache_key}/{problem_id}/{sample_id}: expected score list, got {type(score_list).__name__}"
                    )
                if slot_index < 0 or slot_index >= len(score_list):
                    raise ValueError(
                        f"{cache_key}/{problem_id}/{sample_id}: slot_index={slot_index} out of range for len={len(score_list)}"
                    )
                value = float(score_list[slot_index])
                if not np.isfinite(value):
                    raise ValueError(f"{cache_key}/{problem_id}/{sample_id}: extracted score must be finite")
                extracted_problem[str(sample_id)] = value
            extracted[cache_key][str(problem_id)] = extracted_problem
    return extracted


def _build_payload(
    scores: dict[str, dict[str, dict[str, float]]],
    *,
    method_name: str,
    earlystop_method_name: str,
    slot_index: int,
    position_value: float,
    override_method_name: str | None,
    override_cache_keys: tuple[str, ...],
) -> dict[str, Any]:
    if override_method_name is None:
        note = "direct best_of_n export from earlystop slot"
    else:
        note = "non-overridden caches from earlystop slot; selected caches copied from override bestofn"
    return {
        "task": "best_of_n",
        "method_name": str(method_name),
        "scores": scores,
        "score_postprocess": {
            "source_task": "early_stop",
            "source_method_name": str(earlystop_method_name),
            "extracted_slot_index": int(slot_index),
            "extracted_position": float(position_value),
            "override_bestofn_source": None if override_method_name is None else str(override_method_name),
            "override_cache_keys": list(override_cache_keys),
            "note": note,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert one EarlyStop slot into Best-of-N scores and override selected caches from another Best-of-N file"
    )
    ap.add_argument("--earlystop-json", default=str(DEFAULT_EARLYSTOP_JSON), help="Source EarlyStop JSON")
    ap.add_argument("--slot-index", type=int, default=9, help="EarlyStop slot index to extract (default 9 = 100%)")
    ap.add_argument("--override-bestofn-json", default=str(DEFAULT_OVERRIDE_JSON), help="Best-of-N JSON used for cache overrides")
    ap.add_argument(
        "--override-cache-keys",
        default=",".join(DEFAULT_OVERRIDE_CACHE_KEYS),
        help="Comma-separated cache keys to copy from --override-bestofn-json",
    )
    ap.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT), help="Blind cache root used for expected cache-key validation")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output Best-of-N JSON")
    ap.add_argument("--method-name", default=DEFAULT_METHOD_NAME, help="method_name stored in output JSON")
    args = ap.parse_args()

    earlystop_json = Path(args.earlystop_json)
    out_path = Path(args.out)
    override_cache_keys = _parse_csv(args.override_cache_keys, allow_empty=True)
    override_path_raw = str(args.override_bestofn_json).strip()
    use_override = override_path_raw.lower() not in {"", "none", "off"} and bool(override_cache_keys)

    earlystop_payload = _load_json(earlystop_json)
    early_summary = validate_earlystop_payload(earlystop_payload)
    if earlystop_payload.get("task") != "early_stop":
        raise ValueError(f"Expected task='early_stop', got {earlystop_payload.get('task')!r}")

    expected_cache_keys = [entry.cache_key for entry in discover_cache_entries(args.cache_root)]

    output_scores = _extract_bestofn_scores_from_earlystop(
        earlystop_payload,
        slot_index=int(args.slot_index),
    )

    override_summary: dict[str, int] | None = None
    override_method_name: str | None = None
    if use_override:
        override_bestofn_json = Path(override_path_raw)
        override_payload = _load_json(override_bestofn_json)
        override_summary = validate_submission_payload(override_payload, expected_cache_keys=expected_cache_keys)

        override_scores = override_payload.get("scores")
        if not isinstance(override_scores, dict) or not override_scores:
            raise ValueError("override best_of_n payload scores must be a non-empty mapping")

        for cache_key in override_cache_keys:
            if cache_key not in output_scores:
                raise ValueError(f"Source earlystop payload missing override target cache key: {cache_key}")
            if cache_key not in override_scores:
                raise ValueError(f"Override best_of_n payload missing override target cache key: {cache_key}")
            output_scores[cache_key] = override_scores[cache_key]

        override_method_name = str(override_payload.get("method_name", ""))

    payload = _build_payload(
        output_scores,
        method_name=str(args.method_name),
        earlystop_method_name=str(earlystop_payload.get("method_name", "")),
        slot_index=int(args.slot_index),
        position_value=float(EARLY_STOP_POSITIONS[int(args.slot_index)]),
        override_method_name=override_method_name,
        override_cache_keys=override_cache_keys,
    )
    summary = validate_submission_payload(payload, expected_cache_keys=expected_cache_keys)
    written = write_submission_payload(payload, out_path)

    print(f"EarlyStop source : {_display_path(earlystop_json)} | {early_summary}")
    if use_override:
        print(f"Override source  : {_display_path(override_bestofn_json)} | {override_summary}")
        print(f"Override caches  : {list(override_cache_keys)}")
    else:
        print("Override source  : disabled")
        print("Override caches  : []")
    print(
        f"Written BestofN  : {_display_path(written)} "
        f"(cache_keys={summary['cache_keys']}, problems={summary['problems']}, samples={summary['samples']})"
    )


if __name__ == "__main__":
    main()
