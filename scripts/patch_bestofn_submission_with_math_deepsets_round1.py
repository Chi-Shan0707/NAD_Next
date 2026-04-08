#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.distance.engine import DistanceEngine, DistanceSpec
from nad.core.selectors.base import SelectorContext
from nad.core.selectors.math_deepsets_impl import (
    MathDeepSetsScorer,
    build_math_deepsets_features,
    default_math_deepsets_model_path,
)
from nad.core.views.reader import Agg, CacheReader, CutSpec, CutType, Order, ViewSpec
from nad.ops.bestofn_extreme8 import (
    build_problem_groups,
    discover_cache_entries,
    validate_submission_payload,
)

DEFAULT_BASE_SUBMISSION = (
    REPO_ROOT
    / "submission/BestofN/extreme12/patches/"
    / "extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa_patch.json"
)
DEFAULT_OUT = (
    REPO_ROOT
    / "submission/BestofN/extreme12/patches/"
    / "extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa__math_deepsets_round1_patch.json"
)
DEFAULT_CACHE_ROOT = Path("/home/jovyan/public-ro/MUI_HUB/cache_test")
DEFAULT_METHOD_NAME = (
    "extreme12_baseline12_pointwise_best_only_ref030_t1024"
    "__code_v2_lcb__science_hybrid_round3_gpqa__math_deepsets_round1_math_patch"
)
DEFAULT_MATH_OVERRIDE = "math_deepsets_round1=MathDeepSetsSelector@ja_mass0.98"
MATH_DATASETS = {"aime24", "aime25", "brumo25", "hmmt25"}
MATH_MODELS = {"DS-R1", "Qwen3-4B"}
DEFAULT_VIEW = ViewSpec(
    agg=Agg.MAX,
    cut=CutSpec(CutType.MASS, 0.98),
    order=Order.BY_KEY,
)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _problem_sort_key(problem_id: str) -> tuple[int, str]:
    text = str(problem_id)
    try:
        suffix = text.split("-")[-1]
        return (0, f"{int(suffix):09d}")
    except Exception:
        return (1, text)


def _parse_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _rank_scale_desc(order: np.ndarray, run_ids: list[int], lo: float = 1.0, hi: float = 100.0) -> dict[str, float]:
    n = len(run_ids)
    if n == 0:
        return {}
    if n == 1:
        return {str(run_ids[int(order[0])]): float(hi)}
    values = np.linspace(float(hi), float(lo), num=n, dtype=np.float64)
    return {
        str(run_ids[int(group_idx)]): float(values[rank_pos])
        for rank_pos, group_idx in enumerate(order.tolist())
    }


def resolve_math_patch_entries(
    cache_root: Path,
    *,
    allowed_cache_keys: set[str],
    requested_cache_keys: set[str] | None = None,
    patch_datasets: set[str] | None = None,
    patch_models: set[str] | None = None,
) -> list:
    allowed_math_cache_keys = {
        cache_key
        for cache_key in allowed_cache_keys
        if "/" in str(cache_key) and str(cache_key).split("/", 1)[1] in MATH_DATASETS
    }
    if requested_cache_keys:
        invalid_cache_keys = sorted(set(requested_cache_keys) - allowed_math_cache_keys)
        if invalid_cache_keys:
            raise SystemExit(
                "Requested math patch cache keys are invalid for this base submission: "
                f"{invalid_cache_keys}"
            )

    if patch_datasets:
        invalid_datasets = sorted(set(patch_datasets) - set(MATH_DATASETS))
        if invalid_datasets:
            raise SystemExit(f"Unknown math datasets in --patch-datasets: {invalid_datasets}")

    if patch_models:
        invalid_models = sorted(set(patch_models) - set(MATH_MODELS))
        if invalid_models:
            raise SystemExit(f"Unknown models in --patch-models: {invalid_models}")

    entries = []
    for entry in discover_cache_entries(cache_root):
        if entry.cache_key not in allowed_math_cache_keys:
            continue
        if requested_cache_keys is not None and requested_cache_keys:
            if entry.cache_key not in requested_cache_keys:
                continue
        else:
            if patch_datasets and entry.dataset_name not in patch_datasets:
                continue
            if patch_models and entry.cache_key.split("/", 1)[0] not in patch_models:
                continue
        entries.append(entry)
    entries.sort(key=lambda entry: entry.cache_key)
    if not entries:
        raise SystemExit("No math entries matched the requested patch selection.")
    return entries


def _compute_problem_scores(
    cache_root: Path,
    *,
    scorer: MathDeepSetsScorer,
    distance_threads: int,
) -> dict[str, dict[str, float]]:
    meta = json.loads((cache_root / "meta.json").read_text(encoding="utf-8"))
    groups = build_problem_groups(meta)
    reader = CacheReader(str(cache_root))
    engine = DistanceEngine(
        DistanceSpec(
            name="ja",
            normalize=True,
            num_threads=int(distance_threads),
            assume_unique=True,
        )
    )
    out: dict[str, dict[str, float]] = {}

    for problem_id, run_ids in sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0])):
        run_ids = list(map(int, run_ids))
        if len(run_ids) < 2:
            continue
        views = [reader.get_run_view(int(run_id), DEFAULT_VIEW, normalize_l1=True) for run_id in run_ids]
        lengths = np.asarray([len(view.keys) for view in views], dtype=np.int32)
        D = engine.dense_matrix(views)
        ctx = SelectorContext(
            cache=reader,
            problem_id=str(problem_id),
            run_ids=run_ids,
            views=views,
        )
        X = build_math_deepsets_features(
            D,
            {"lengths": lengths, "views": views},
            context=ctx,
        )
        scores = np.asarray(scorer.score_group(X), dtype=np.float64)
        order = np.argsort(-scores, kind="stable")
        out[str(problem_id)] = _rank_scale_desc(order, run_ids)

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch BestofN submission with math_deepsets_round1 math scores")
    ap.add_argument("--base-submission", default=str(DEFAULT_BASE_SUBMISSION))
    ap.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--model-path", default=str(default_math_deepsets_model_path()))
    ap.add_argument("--distance-threads", type=int, default=8)
    ap.add_argument("--method-name", default=DEFAULT_METHOD_NAME)
    ap.add_argument("--math-override-note", default=DEFAULT_MATH_OVERRIDE)
    ap.add_argument(
        "--patch-cache-keys",
        default="",
        help="Optional comma-separated exact cache keys to patch; overrides dataset/model filters",
    )
    ap.add_argument(
        "--patch-datasets",
        default="",
        help="Optional comma-separated math datasets to patch, e.g. aime25,brumo25",
    )
    ap.add_argument(
        "--patch-models",
        default="",
        help="Optional comma-separated model tags to patch, e.g. DS-R1,Qwen3-4B",
    )
    args = ap.parse_args()

    base_submission = Path(args.base_submission)
    out_path = Path(args.out)
    cache_root = Path(args.cache_root)
    model_path = Path(args.model_path)

    base_payload = json.loads(base_submission.read_text(encoding="utf-8"))
    base_keys = list(base_payload.get("scores", {}).keys())
    if not base_keys:
        raise SystemExit(f"Base submission has no scores: {_display_path(base_submission)}")

    scorer = MathDeepSetsScorer.load(model_path)
    requested_cache_keys = set(_parse_csv(args.patch_cache_keys)) or None
    patch_datasets = set(_parse_csv(args.patch_datasets)) or None
    patch_models = set(_parse_csv(args.patch_models)) or None
    entries = resolve_math_patch_entries(
        cache_root,
        allowed_cache_keys=set(base_keys),
        requested_cache_keys=requested_cache_keys,
        patch_datasets=patch_datasets,
        patch_models=patch_models,
    )
    patched_scores: dict[str, dict[str, dict[str, float]]] = {}
    for entry in entries:
        print(f"[patch] {entry.cache_key} <- {_display_path(entry.cache_root)}")
        patched_scores[entry.cache_key] = _compute_problem_scores(
            entry.cache_root,
            scorer=scorer,
            distance_threads=int(args.distance_threads),
        )
        n_problems = len(patched_scores[entry.cache_key])
        sample_sizes = sorted({len(problem_scores) for problem_scores in patched_scores[entry.cache_key].values()})
        print(f"  problems={n_problems} samples_per_problem={sample_sizes}")

    patched_payload = json.loads(json.dumps(base_payload))
    for cache_key, problem_scores in patched_scores.items():
        patched_payload["scores"][cache_key] = problem_scores

    patched_payload["method_name"] = str(args.method_name)
    patched_payload.setdefault("score_postprocess", {})["math_override"] = str(args.math_override_note)
    patched_payload["score_postprocess"]["math_patch_params"] = {
        "distance": "ja",
        "distance_threads": int(args.distance_threads),
        "model_path": str(model_path),
        "view_cut_mass": 0.98,
        "normalize_l1": True,
        "patched_cache_keys": [entry.cache_key for entry in entries],
        "math_datasets": sorted(MATH_DATASETS),
        "selection": {
            "patch_cache_keys": sorted(requested_cache_keys) if requested_cache_keys else [],
            "patch_datasets": sorted(patch_datasets) if patch_datasets else [],
            "patch_models": sorted(patch_models) if patch_models else [],
        },
    }

    summary = validate_submission_payload(patched_payload, expected_cache_keys=base_keys)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(patched_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[saved] {_display_path(out_path)} "
        f"(cache_keys={summary['cache_keys']}, problems={summary['problems']}, samples={summary['samples']})"
    )


if __name__ == "__main__":
    main()
