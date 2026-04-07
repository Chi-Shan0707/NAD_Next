#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.distance.engine import DistanceEngine, DistanceSpec
from nad.core.selectors.base import SelectorContext
from nad.core.selectors.code_dynamic_impl import order_code_dynamic_group_indices
from nad.core.selectors.science_dynamic_impl import (
    DEFAULT_SCIENCE_DYNAMIC_WEIGHTS,
    DEFAULT_SCIENCE_PREFIX_FRACTION,
    DEFAULT_SCIENCE_RECENCY_EXP,
    DEFAULT_SCIENCE_TAIL_FRACTION,
    DEFAULT_SCIENCE_WINDOW_TOKENS,
    compute_science_dynamic_primary_scores,
)
from nad.core.views.reader import Agg, CacheReader, CutSpec, CutType, Order, ViewSpec
from nad.ops.bestofn_extreme8 import (
    build_problem_groups,
    discover_cache_entries,
    validate_submission_payload,
)

BASE_SUBMISSION = REPO_ROOT / "submission/BestofN/extreme12/base/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank.json"
DEFAULT_OUT = REPO_ROOT / "submission/BestofN/extreme12/patches/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__science_baseline_v1_gpqa_patch.json"
DEFAULT_CACHE_ROOT = Path("/home/jovyan/public-ro/MUI_HUB/cache_test")
TARGET_CACHE_KEYS = ("DS-R1/gpqa", "Qwen3-4B/gpqa")
DEFAULT_METHOD_NAME = "extreme12_baseline12_pointwise_best_only_ref030_t1024__science_baseline_v1_gpqa_patch"
DEFAULT_GPQA_OVERRIDE = "science_baseline_v1=ScienceCommitmentSelector@ja_mass1.0"
DEFAULT_VIEW = ViewSpec(
    agg=Agg.MAX,
    cut=CutSpec(CutType.MASS, 1.0),
    order=Order.BY_KEY,
)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _problem_sort_key(problem_id: str) -> tuple[int, str]:
    try:
        return (0, f"{int(problem_id):09d}")
    except (TypeError, ValueError):
        return (1, str(problem_id))


def _rank_scale_desc(order, run_ids: list[int], lo: float = 1.0, hi: float = 100.0) -> dict[str, float]:
    n = len(run_ids)
    if n == 0:
        return {}
    if n == 1:
        return {str(run_ids[int(order[0])]): float(hi)}
    import numpy as np
    values = np.linspace(float(hi), float(lo), num=n, dtype=np.float64)
    return {
        str(run_ids[int(group_idx)]): float(values[rank_pos])
        for rank_pos, group_idx in enumerate(order.tolist())
    }


def _load_target_entries(cache_root: Path) -> list:
    entries = [entry for entry in discover_cache_entries(cache_root) if entry.cache_key in TARGET_CACHE_KEYS]
    entries.sort(key=lambda entry: entry.cache_key)
    found = {entry.cache_key for entry in entries}
    missing = sorted(set(TARGET_CACHE_KEYS) - found)
    if missing:
        raise SystemExit(f"Missing cache_test entries for target cache keys: {missing}")
    return entries


def _compute_problem_scores(
    cache_root: Path,
    *,
    distance_threads: int,
    prefix_fraction: float,
    tail_fraction: float,
    recency_exp: float,
    window_tokens: int,
) -> dict[str, dict[str, float]]:
    meta = json.loads((cache_root / "meta.json").read_text(encoding="utf-8"))
    groups = build_problem_groups(meta)
    reader = CacheReader(str(cache_root))
    engine = DistanceEngine(DistanceSpec("ja", num_threads=int(distance_threads)))
    out: dict[str, dict[str, float]] = {}

    for problem_id, run_ids in sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0])):
        views = [reader.get_run_view(int(run_id), DEFAULT_VIEW) for run_id in run_ids]
        D = engine.dense_matrix(views)
        ctx = SelectorContext(
            cache=reader,
            problem_id=str(problem_id),
            run_ids=list(map(int, run_ids)),
            views=views,
        )
        scores, _, _ = compute_science_dynamic_primary_scores(
            ctx,
            weights=DEFAULT_SCIENCE_DYNAMIC_WEIGHTS,
            prefix_fraction=prefix_fraction,
            tail_fraction=tail_fraction,
            recency_exp=recency_exp,
            window_tokens=window_tokens,
        )
        order = order_code_dynamic_group_indices(
            scores,
            D,
            run_ids=run_ids,
        )
        out[str(problem_id)] = _rank_scale_desc(order, list(map(int, run_ids)))

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch best_of_n submission with science_baseline_v1 gpqa scores")
    ap.add_argument("--base-submission", default=str(BASE_SUBMISSION))
    ap.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--distance-threads", type=int, default=12)
    ap.add_argument("--prefix-fraction", type=float, default=DEFAULT_SCIENCE_PREFIX_FRACTION)
    ap.add_argument("--tail-fraction", type=float, default=DEFAULT_SCIENCE_TAIL_FRACTION)
    ap.add_argument("--recency-exp", type=float, default=DEFAULT_SCIENCE_RECENCY_EXP)
    ap.add_argument("--window-tokens", type=int, default=DEFAULT_SCIENCE_WINDOW_TOKENS)
    ap.add_argument("--method-name", default=DEFAULT_METHOD_NAME)
    ap.add_argument("--gpqa-override-note", default=DEFAULT_GPQA_OVERRIDE)
    args = ap.parse_args()

    base_submission = Path(args.base_submission)
    out_path = Path(args.out)
    cache_root = Path(args.cache_root)

    base_payload = json.loads(base_submission.read_text(encoding="utf-8"))
    base_keys = list(base_payload.get("scores", {}).keys())
    for cache_key in TARGET_CACHE_KEYS:
        if cache_key not in base_payload.get("scores", {}):
            raise SystemExit(f"Base submission missing target cache key: {cache_key}")

    entries = _load_target_entries(cache_root)
    patched_scores: dict[str, dict[str, dict[str, float]]] = {}
    for entry in entries:
        print(f"[patch] {entry.cache_key} <- {_display_path(entry.cache_root)}")
        patched_scores[entry.cache_key] = _compute_problem_scores(
            entry.cache_root,
            distance_threads=int(args.distance_threads),
            prefix_fraction=float(args.prefix_fraction),
            tail_fraction=float(args.tail_fraction),
            recency_exp=float(args.recency_exp),
            window_tokens=int(args.window_tokens),
        )

    patched_payload = json.loads(json.dumps(base_payload))
    for cache_key, problem_scores in patched_scores.items():
        patched_payload["scores"][cache_key] = problem_scores
    patched_payload["method_name"] = str(args.method_name)
    patched_payload.setdefault("score_postprocess", {})["gpqa_override"] = str(args.gpqa_override_note)
    patched_payload["score_postprocess"]["gpqa_patch_params"] = {
        "distance_threads": int(args.distance_threads),
        "prefix_fraction": float(args.prefix_fraction),
        "tail_fraction": float(args.tail_fraction),
        "recency_exp": float(args.recency_exp),
        "window_tokens": int(args.window_tokens),
        "target_cache_keys": list(TARGET_CACHE_KEYS),
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
