#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.ops.accuracy import _load_ground_truth
from nad.ops.bestofn_extreme8 import discover_cache_entries, validate_submission_payload
from scripts.patch_bestofn_submission_with_math_deepsets_round1 import (
    DEFAULT_BASE_SUBMISSION as DEFAULT_NO_MATH_SUBMISSION,
    DEFAULT_MATH_OVERRIDE,
    _compute_problem_scores,
    _problem_sort_key,
    resolve_math_patch_entries,
)
from scripts.run_science_hybrid_round3 import (
    CODE_V2_EXHAUSTIVE_JSON,
    EXTREME12_TEST_ANALYSIS_DOC,
    ProblemScoreRecord,
    _combine_cache_metric_proxy,
    _evaluate_problem_records,
    _load_code_v2_proxy_metrics,
    _load_extreme12_test_metrics,
    _system_delta,
)
from nad.core.selectors.math_deepsets_impl import MathDeepSetsScorer, default_math_deepsets_model_path

DEFAULT_DS_CACHE_ROOT = REPO_ROOT / "MUI_HUB" / "cache"
DEFAULT_BLIND_CACHE_ROOT = Path("/home/jovyan/public-ro/MUI_HUB/cache_test")
DEFAULT_FULL_MATH_SUBMISSION = (
    REPO_ROOT
    / "submission/BestofN/extreme12/patches/"
    / "extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa__math_deepsets_round1_patch.json"
)
DEFAULT_OUT_DIR = REPO_ROOT / "result" / "bestofn_score_recovery_20260408"
DEFAULT_CANDIDATE_EXPORT_DIR = REPO_ROOT / "submission/BestofN/extreme12/patches"
DEFAULT_NO_MATH_EXPORT = (
    DEFAULT_CANDIDATE_EXPORT_DIR
    / "extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa__no_math_patch.json"
)
DEFAULT_RECOMMENDED_EXPORT = (
    DEFAULT_CANDIDATE_EXPORT_DIR
    / "extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa__math_patch_ds_aime25_only.json"
)
DEFAULT_AGGRESSIVE_EXPORT = (
    DEFAULT_CANDIDATE_EXPORT_DIR
    / "extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa__math_patch_ds_qwen_aime25_brumo25.json"
)

NO_MATH_METHOD_NAME = (
    "extreme12_baseline12_pointwise_best_only_ref030_t1024"
    "__code_v2_lcb__science_hybrid_round3_gpqa__no_math_patch"
)
RECOMMENDED_METHOD_NAME = (
    "extreme12_baseline12_pointwise_best_only_ref030_t1024"
    "__code_v2_lcb__science_hybrid_round3_gpqa__math_patch_ds_aime25_only"
)
AGGRESSIVE_METHOD_NAME = (
    "extreme12_baseline12_pointwise_best_only_ref030_t1024"
    "__code_v2_lcb__science_hybrid_round3_gpqa__math_patch_ds_qwen_aime25_brumo25"
)

DS_MATH_KEYS = (
    "DS-R1/aime24",
    "DS-R1/aime25",
    "DS-R1/brumo25",
    "DS-R1/hmmt25",
)
BLIND_MATH_KEYS = (
    "DS-R1/aime24",
    "DS-R1/aime25",
    "DS-R1/brumo25",
    "DS-R1/hmmt25",
    "Qwen3-4B/aime24",
    "Qwen3-4B/aime25",
    "Qwen3-4B/brumo25",
    "Qwen3-4B/hmmt25",
)
FULL_SYSTEM_KEYS = (
    "DS-R1/aime24",
    "DS-R1/aime25",
    "DS-R1/brumo25",
    "DS-R1/gpqa",
    "DS-R1/hmmt25",
    "DS-R1/lcb_v5",
)
DEFAULT_SCIENCE_JSON_GLOB = "result/science_hybrid_round3_*/science_hybrid_round3.json"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    patch_cache_keys: tuple[str, ...]
    risk: str
    rationale: str


CANDIDATE_SPECS: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        name="no_math_patch",
        patch_cache_keys=(),
        risk="low risk",
        rationale="Exact rollback of the current promoted stack after Submission #92 failed online.",
    ),
    CandidateSpec(
        name="full_math_patch_all8",
        patch_cache_keys=BLIND_MATH_KEYS,
        risk="high risk",
        rationale="Matches the broad #92 failure pattern and touches every blind math cache.",
    ),
    CandidateSpec(
        name="ds_only_all4",
        patch_cache_keys=DS_MATH_KEYS,
        risk="high risk",
        rationale="Still broad on DS math and keeps the known hmmt/aime24 regression-prone slices.",
    ),
    CandidateSpec(
        name="qwen_only_all4",
        patch_cache_keys=(
            "Qwen3-4B/aime24",
            "Qwen3-4B/aime25",
            "Qwen3-4B/brumo25",
            "Qwen3-4B/hmmt25",
        ),
        risk="high risk",
        rationale="Pure blind Qwen patch with no GT-backed uplift and large exposure.",
    ),
    CandidateSpec(
        name="aime_only_both_models",
        patch_cache_keys=(
            "DS-R1/aime24",
            "DS-R1/aime25",
            "Qwen3-4B/aime24",
            "Qwen3-4B/aime25",
        ),
        risk="high risk",
        rationale="Includes the regression-prone aime24 slice and adds blind Qwen exposure.",
    ),
    CandidateSpec(
        name="brumo_only_both_models",
        patch_cache_keys=("DS-R1/brumo25", "Qwen3-4B/brumo25"),
        risk="medium risk",
        rationale="Small selective patch, but half of the exposure is blind Qwen.",
    ),
    CandidateSpec(
        name="hmmt_only_both_models",
        patch_cache_keys=("DS-R1/hmmt25", "Qwen3-4B/hmmt25"),
        risk="high risk",
        rationale="Centers the known hmmt regression-prone slice and adds blind Qwen risk.",
    ),
    CandidateSpec(
        name="ds_only_aime25",
        patch_cache_keys=("DS-R1/aime25",),
        risk="low risk",
        rationale="Smallest GT-backed positive patch and avoids all known regression slices.",
    ),
    CandidateSpec(
        name="ds_only_aime25_brumo25",
        patch_cache_keys=("DS-R1/aime25", "DS-R1/brumo25"),
        risk="medium risk",
        rationale="Keeps the two positive DS slices only and still avoids Qwen blind exposure.",
    ),
    CandidateSpec(
        name="aime25_only_both_models",
        patch_cache_keys=("DS-R1/aime25", "Qwen3-4B/aime25"),
        risk="medium risk",
        rationale="Anchors on the strongest DS slice but adds one blind Qwen patch.",
    ),
    CandidateSpec(
        name="aime25_brumo25_both_models",
        patch_cache_keys=(
            "DS-R1/aime25",
            "DS-R1/brumo25",
            "Qwen3-4B/aime25",
            "Qwen3-4B/brumo25",
        ),
        risk="high risk",
        rationale="Reasonable positive DS core, but the added Qwen exposure is large.",
    ),
    CandidateSpec(
        name="all_except_hmmt_both_models",
        patch_cache_keys=(
            "DS-R1/aime24",
            "DS-R1/aime25",
            "DS-R1/brumo25",
            "Qwen3-4B/aime24",
            "Qwen3-4B/aime25",
            "Qwen3-4B/brumo25",
        ),
        risk="high risk",
        rationale="Drops hmmt but still stays too broad and keeps aime24 plus large Qwen exposure.",
    ),
)


def _latest_current_science_json() -> Path:
    matches = sorted(glob.glob(str(REPO_ROOT / DEFAULT_SCIENCE_JSON_GLOB)))
    if not matches:
        raise FileNotFoundError("No science_hybrid_round3.json payload found under result/")
    return Path(matches[-1])


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _json_clone(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


def _problem_counts_by_cache(cache_root: Path, cache_keys: tuple[str, ...]) -> dict[str, int]:
    entry_map = {entry.cache_key: entry for entry in discover_cache_entries(cache_root)}
    out: dict[str, int] = {}
    for cache_key in cache_keys:
        entry = entry_map.get(cache_key)
        if entry is None:
            raise RuntimeError(f"Missing cache entry for {cache_key} under {_display_path(cache_root)}")
        meta = json.loads((entry.cache_root / "meta.json").read_text(encoding="utf-8"))
        problem_ids = {str(sample["problem_id"]) for sample in meta["samples"]}
        out[cache_key] = int(len(problem_ids))
    return out


def _load_current_system_bundle(
    *,
    ds_cache_root: Path,
    science_json: Path,
    code_v2_json: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    base_doc_metrics = _load_extreme12_test_metrics(EXTREME12_TEST_ANALYSIS_DOC)
    problem_counts = _problem_counts_by_cache(ds_cache_root, FULL_SYSTEM_KEYS)
    for cache_key in FULL_SYSTEM_KEYS:
        base_doc_metrics[cache_key]["n_problems"] = int(problem_counts[cache_key])
        base_doc_metrics[cache_key]["top10_count"] = max(
            1,
            int(math.ceil(0.10 * int(base_doc_metrics[cache_key]["n_samples"]))),
        )

    science_payload = json.loads(science_json.read_text(encoding="utf-8"))
    selected_candidate = dict(science_payload["selected_candidate"])
    current_science_metrics = dict(selected_candidate["gpqa_proxy_metrics"])

    code_v2_metrics = _load_code_v2_proxy_metrics(
        code_v2_json,
        fallback_hit3=float(base_doc_metrics["DS-R1/lcb_v5"]["hit@3"]),
    )

    current_cache_metrics = {cache_key: dict(base_doc_metrics[cache_key]) for cache_key in FULL_SYSTEM_KEYS}
    current_cache_metrics["DS-R1/gpqa"] = dict(current_science_metrics)
    current_cache_metrics["DS-R1/lcb_v5"] = dict(code_v2_metrics)
    current_bundle = _combine_cache_metric_proxy(current_cache_metrics)
    return current_bundle, current_cache_metrics, selected_candidate


def _records_from_problem_scores(
    cache_key: str,
    problem_scores: dict[str, dict[str, float]],
    correctness: dict[int, bool],
) -> list[ProblemScoreRecord]:
    records: list[ProblemScoreRecord] = []
    for problem_id, sample_score_map in sorted(problem_scores.items(), key=lambda kv: _problem_sort_key(kv[0])):
        sample_ids = [int(sample_id) for sample_id in sample_score_map.keys()]
        scores = np.asarray([float(sample_score_map[str(sample_id)]) for sample_id in sample_ids], dtype=np.float64)
        labels = np.asarray([int(bool(correctness[int(sample_id)])) for sample_id in sample_ids], dtype=np.int32)
        records.append(
            ProblemScoreRecord(
                cache_key=str(cache_key),
                problem_id=str(problem_id),
                sample_ids=sample_ids,
                labels=labels,
                scores=scores,
            )
        )
    return records


def _compute_ds_math_patch_metrics(
    *,
    ds_cache_root: Path,
    model_path: Path,
    distance_threads: int,
) -> dict[str, dict[str, Any]]:
    scorer = MathDeepSetsScorer.load(model_path)
    entries = resolve_math_patch_entries(
        ds_cache_root,
        allowed_cache_keys=set(FULL_SYSTEM_KEYS),
        patch_models={"DS-R1"},
    )
    out: dict[str, dict[str, Any]] = {}
    for entry in entries:
        correctness = _load_ground_truth(entry.cache_root)
        problem_scores = _compute_problem_scores(
            entry.cache_root,
            scorer=scorer,
            distance_threads=int(distance_threads),
        )
        records = _records_from_problem_scores(entry.cache_key, problem_scores, correctness)
        out[entry.cache_key] = _evaluate_problem_records(records)
    return out


def _top1_sample_id(problem_scores: dict[str, float]) -> int:
    ordered = sorted(
        ((int(sample_id), float(score)) for sample_id, score in problem_scores.items()),
        key=lambda item: (-item[1], item[0]),
    )
    return int(ordered[0][0])


def _blind_exposure_by_cache(
    *,
    base_submission: Path,
    full_math_submission: Path,
) -> dict[str, dict[str, Any]]:
    base_payload = json.loads(base_submission.read_text(encoding="utf-8"))
    full_payload = json.loads(full_math_submission.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for cache_key in BLIND_MATH_KEYS:
        base_scores = base_payload["scores"][cache_key]
        full_scores = full_payload["scores"][cache_key]
        problem_ids = sorted(set(base_scores) | set(full_scores), key=_problem_sort_key)
        changed_problem_ids: list[str] = []
        for problem_id in problem_ids:
            if _top1_sample_id(base_scores[problem_id]) != _top1_sample_id(full_scores[problem_id]):
                changed_problem_ids.append(str(problem_id))
        out[cache_key] = {
            "n_problems": int(len(problem_ids)),
            "changed_top1": int(len(changed_problem_ids)),
            "changed_top1_rate": float(len(changed_problem_ids) / max(len(problem_ids), 1)),
            "model": str(cache_key.split("/", 1)[0]),
            "dataset": str(cache_key.split("/", 1)[1]),
            "changed_problem_ids": changed_problem_ids,
        }
    return out


def _aggregate_blind_exposure(
    candidate_keys: tuple[str, ...],
    exposure_by_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    selected = [dict(exposure_by_cache[cache_key]) | {"cache_key": str(cache_key)} for cache_key in candidate_keys]
    patched_problems = sum(int(item["n_problems"]) for item in selected)
    changed_top1 = sum(int(item["changed_top1"]) for item in selected)
    qwen_changed = sum(int(item["changed_top1"]) for item in selected if str(item["cache_key"]).startswith("Qwen3-4B/"))
    qwen_problems = sum(int(item["n_problems"]) for item in selected if str(item["cache_key"]).startswith("Qwen3-4B/"))
    return {
        "patched_cache_keys": [str(item["cache_key"]) for item in selected],
        "patched_cache_count": int(len(selected)),
        "patched_problems": int(patched_problems),
        "changed_top1": int(changed_top1),
        "changed_top1_rate_over_patched": float(changed_top1 / max(patched_problems, 1)),
        "qwen_patched_problems": int(qwen_problems),
        "qwen_changed_top1": int(qwen_changed),
        "per_cache": selected,
    }


def _candidate_rows(
    *,
    current_bundle: dict[str, Any],
    current_cache_metrics: dict[str, dict[str, Any]],
    ds_math_patch_metrics: dict[str, dict[str, Any]],
    blind_exposure_by_cache: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_math_metrics = {
        cache_key: dict(current_cache_metrics[cache_key])
        for cache_key in DS_MATH_KEYS
    }
    for spec in CANDIDATE_SPECS:
        patched_ds_keys = [cache_key for cache_key in spec.patch_cache_keys if cache_key.startswith("DS-R1/")]
        candidate_cache_metrics = {cache_key: dict(metrics) for cache_key, metrics in current_cache_metrics.items()}
        for cache_key in patched_ds_keys:
            candidate_cache_metrics[cache_key] = dict(ds_math_patch_metrics[cache_key])
        candidate_bundle = _combine_cache_metric_proxy(candidate_cache_metrics)
        candidate_delta = _system_delta(candidate_bundle, current_bundle)

        math_only_cache_metrics = {cache_key: dict(metrics) for cache_key, metrics in baseline_math_metrics.items()}
        for cache_key in patched_ds_keys:
            math_only_cache_metrics[cache_key] = dict(ds_math_patch_metrics[cache_key])
        math_only_bundle = _combine_cache_metric_proxy(math_only_cache_metrics)

        rows.append(
            {
                "name": str(spec.name),
                "risk": str(spec.risk),
                "risk_rationale": str(spec.rationale),
                "patch_cache_keys": list(spec.patch_cache_keys),
                "patched_ds_cache_keys": patched_ds_keys,
                "patched_qwen_cache_keys": [
                    cache_key for cache_key in spec.patch_cache_keys if cache_key.startswith("Qwen3-4B/")
                ],
                "sample_weighted_delta_vs_no_math": dict(candidate_delta["sample_weighted"]),
                "equal_cache_mean_delta_vs_no_math": dict(candidate_delta["equal_cache_mean"]),
                "full_system_proxy": {
                    "sample_weighted": dict(candidate_bundle["sample_weighted"]),
                    "equal_cache_mean": dict(candidate_bundle["equal_cache_mean"]),
                },
                "math_only_proxy": {
                    "sample_weighted": dict(math_only_bundle["sample_weighted"]),
                    "equal_cache_mean": dict(math_only_bundle["equal_cache_mean"]),
                },
                "blind_exposure": _aggregate_blind_exposure(spec.patch_cache_keys, blind_exposure_by_cache),
            }
        )
    return rows


def _candidate_index(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(row for row in rows if row["name"] == name)


def _patch_payload_with_cache_keys(
    *,
    base_payload: dict[str, Any],
    cache_root: Path,
    model_path: Path,
    patch_cache_keys: tuple[str, ...],
    distance_threads: int,
    method_name: str,
    math_override_note: str,
) -> dict[str, Any]:
    scorer = MathDeepSetsScorer.load(model_path)
    entries = resolve_math_patch_entries(
        cache_root,
        allowed_cache_keys=set(base_payload.get("scores", {}).keys()),
        requested_cache_keys=set(patch_cache_keys),
    )
    patched_payload = _json_clone(base_payload)
    for entry in entries:
        patched_payload["scores"][entry.cache_key] = _compute_problem_scores(
            entry.cache_root,
            scorer=scorer,
            distance_threads=int(distance_threads),
        )
    patched_payload["method_name"] = str(method_name)
    patched_payload.setdefault("score_postprocess", {})["math_override"] = str(math_override_note)
    patched_payload["score_postprocess"]["math_patch_params"] = {
        "distance": "ja",
        "distance_threads": int(distance_threads),
        "model_path": str(model_path),
        "view_cut_mass": 0.98,
        "normalize_l1": True,
        "patched_cache_keys": [entry.cache_key for entry in entries],
        "math_datasets": sorted({entry.dataset_name for entry in entries}),
        "selection": {
            "patch_cache_keys": list(map(str, patch_cache_keys)),
            "patch_datasets": sorted({entry.dataset_name for entry in entries}),
            "patch_models": sorted({entry.cache_key.split("/", 1)[0] for entry in entries}),
        },
    }
    return patched_payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _export_submission_candidates(
    *,
    base_submission: Path,
    blind_cache_root: Path,
    model_path: Path,
    distance_threads: int,
) -> dict[str, str]:
    base_payload = json.loads(base_submission.read_text(encoding="utf-8"))

    no_math_payload = _json_clone(base_payload)
    no_math_payload["method_name"] = NO_MATH_METHOD_NAME
    no_math_payload.setdefault("score_postprocess", {})["math_override"] = "none"
    no_math_payload["score_postprocess"]["math_patch_params"] = {
        "distance": "ja",
        "distance_threads": int(distance_threads),
        "model_path": str(model_path),
        "view_cut_mass": 0.98,
        "normalize_l1": True,
        "patched_cache_keys": [],
        "math_datasets": [],
        "selection": {
            "patch_cache_keys": [],
            "patch_datasets": [],
            "patch_models": [],
        },
        "note": "Explicit rollback candidate: no math patch applied.",
    }
    validate_submission_payload(no_math_payload, expected_cache_keys=list(base_payload["scores"].keys()))
    _write_json(DEFAULT_NO_MATH_EXPORT, no_math_payload)

    recommended_payload = _patch_payload_with_cache_keys(
        base_payload=base_payload,
        cache_root=blind_cache_root,
        model_path=model_path,
        patch_cache_keys=("DS-R1/aime25",),
        distance_threads=distance_threads,
        method_name=RECOMMENDED_METHOD_NAME,
        math_override_note=DEFAULT_MATH_OVERRIDE,
    )
    validate_submission_payload(recommended_payload, expected_cache_keys=list(base_payload["scores"].keys()))
    _write_json(DEFAULT_RECOMMENDED_EXPORT, recommended_payload)

    aggressive_payload = _patch_payload_with_cache_keys(
        base_payload=base_payload,
        cache_root=blind_cache_root,
        model_path=model_path,
        patch_cache_keys=(
            "DS-R1/aime25",
            "DS-R1/brumo25",
            "Qwen3-4B/aime25",
            "Qwen3-4B/brumo25",
        ),
        distance_threads=distance_threads,
        method_name=AGGRESSIVE_METHOD_NAME,
        math_override_note=DEFAULT_MATH_OVERRIDE,
    )
    validate_submission_payload(aggressive_payload, expected_cache_keys=list(base_payload["scores"].keys()))
    _write_json(DEFAULT_AGGRESSIVE_EXPORT, aggressive_payload)

    return {
        "no_math_patch": _display_path(DEFAULT_NO_MATH_EXPORT),
        "recommended": _display_path(DEFAULT_RECOMMENDED_EXPORT),
        "aggressive": _display_path(DEFAULT_AGGRESSIVE_EXPORT),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Best-of-N score recovery search for conservative math patch selection")
    ap.add_argument("--ds-cache-root", default=str(DEFAULT_DS_CACHE_ROOT))
    ap.add_argument("--blind-cache-root", default=str(DEFAULT_BLIND_CACHE_ROOT))
    ap.add_argument("--base-submission", default=str(DEFAULT_NO_MATH_SUBMISSION))
    ap.add_argument("--full-math-submission", default=str(DEFAULT_FULL_MATH_SUBMISSION))
    ap.add_argument("--science-json", default="")
    ap.add_argument("--code-v2-json", default=str(CODE_V2_EXHAUSTIVE_JSON))
    ap.add_argument("--model-path", default=str(default_math_deepsets_model_path()))
    ap.add_argument("--distance-threads", type=int, default=8)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    ds_cache_root = Path(args.ds_cache_root)
    blind_cache_root = Path(args.blind_cache_root)
    base_submission = Path(args.base_submission)
    full_math_submission = Path(args.full_math_submission)
    science_json = Path(args.science_json) if args.science_json else _latest_current_science_json()
    code_v2_json = Path(args.code_v2_json)
    model_path = Path(args.model_path)
    out_dir = Path(args.out_dir)

    current_bundle, current_cache_metrics, selected_science = _load_current_system_bundle(
        ds_cache_root=ds_cache_root,
        science_json=science_json,
        code_v2_json=code_v2_json,
    )
    ds_math_patch_metrics = _compute_ds_math_patch_metrics(
        ds_cache_root=ds_cache_root,
        model_path=model_path,
        distance_threads=int(args.distance_threads),
    )
    blind_exposure = _blind_exposure_by_cache(
        base_submission=base_submission,
        full_math_submission=full_math_submission,
    )
    candidate_rows = _candidate_rows(
        current_bundle=current_bundle,
        current_cache_metrics=current_cache_metrics,
        ds_math_patch_metrics=ds_math_patch_metrics,
        blind_exposure_by_cache=blind_exposure,
    )

    exports = {}
    if not args.skip_export:
        exports = _export_submission_candidates(
            base_submission=base_submission,
            blind_cache_root=blind_cache_root,
            model_path=model_path,
            distance_threads=int(args.distance_threads),
        )

    payload = {
        "status_summary": {
            "code_v2_is_promoted_default": True,
            "science_hybrid_round3_is_promoted_science_patch": True,
            "gpqa_deepsets_round1_is_no_promote_due_to_hit1_guardrail": True,
            "code_deepsets_round1_is_no_promote_due_to_selacc_regression": True,
            "math_deepsets_round1_full_patch_is_now_high_risk_due_to_submission_92": True,
        },
        "assumptions": {
            "full_system_proxy_scope": "DS-R1 six-cache GT-backed proxy only",
            "qwen_scope": "blind exposure only; no GT-backed offline proxy available in repo",
            "submission_92_signal": {
                "submission_id": 92,
                "method_name": (
                    "extreme12_baseline12_pointwise_best_only_ref030_t1024"
                    "__code_v2_lcb__science_hybrid_round3_gpqa__math_deepsets_round1_math_patch"
                ),
                "result": "Not best",
                "conclusion": "Treat full math_deepsets patch as high risk unless selective rollback search says otherwise.",
            },
        },
        "inputs": {
            "ds_cache_root": _display_path(ds_cache_root),
            "blind_cache_root": str(blind_cache_root),
            "base_submission": _display_path(base_submission),
            "full_math_submission": _display_path(full_math_submission),
            "science_json": _display_path(science_json),
            "code_v2_json": _display_path(code_v2_json),
            "model_path": _display_path(model_path),
            "distance_threads": int(args.distance_threads),
        },
        "current_system": {
            "full_system_proxy": current_bundle,
            "selected_science_candidate": {
                "name": str(selected_science["name"]),
                "gpqa_proxy_metrics": dict(selected_science["gpqa_proxy_metrics"]),
            },
        },
        "ds_math_patch_metrics": ds_math_patch_metrics,
        "blind_full_patch_exposure_by_cache": blind_exposure,
        "candidate_rows": candidate_rows,
        "recommended_candidates": {
            "conservative_backup": _candidate_index(candidate_rows, "no_math_patch"),
            "recommended_formal_submission": _candidate_index(candidate_rows, "ds_only_aime25"),
            "aggressive_score_shot": _candidate_index(candidate_rows, "aime25_brumo25_both_models"),
        },
        "exports": exports,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "score_recovery.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {_display_path(out_path)}")
    if exports:
        for key, rel_path in exports.items():
            print(f"[export] {key}: {rel_path}")


if __name__ == "__main__":
    main()
