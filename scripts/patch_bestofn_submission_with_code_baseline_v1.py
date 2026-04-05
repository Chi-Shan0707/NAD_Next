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
from nad.core.selectors.code_dynamic_impl import (
    DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK,
    DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD,
    compute_code_dynamic_primary_scores,
    order_code_dynamic_group_indices,
)
from nad.core.views.reader import Agg, CacheReader, CutSpec, CutType, Order, ViewSpec
from nad.ops.bestofn_extreme8 import (
    build_problem_groups,
    discover_cache_entries,
    validate_submission_payload,
)

BASE_SUBMISSION = REPO_ROOT / "submission/BestofN/extreme12/base/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank.json"
DEFAULT_OUT = REPO_ROOT / "submission/BestofN/extreme12/patches/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_baseline_v1_lcb_patch.json"
DEFAULT_CACHE_ROOT = Path("/home/jovyan/public-ro/MUI_HUB/cache_test")
TARGET_CACHE_KEYS = ("DS-R1/lcb_v5", "Qwen3-4B/lcb_v5")
DEFAULT_METHOD_NAME = "extreme12_baseline12_pointwise_best_only_ref030_t1024__code_baseline_v1_lcb_patch"
DEFAULT_LCB_OVERRIDE = "code_baseline_v1=PrefixSaturationSelector@ja_mass1.0"
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
    distance: str,
    distance_threads: int,
    reflection_threshold: float,
    reflection_lookback_slices: int,
    prefix_fraction: float,
    prefix_window_tokens: int,
) -> dict[str, dict[str, float]]:
    meta = json.loads((cache_root / "meta.json").read_text(encoding="utf-8"))
    groups = build_problem_groups(meta)
    reader = CacheReader(str(cache_root))
    engine = DistanceEngine(DistanceSpec(distance, num_threads=int(distance_threads)))
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
        scores, _, _ = compute_code_dynamic_primary_scores(
            ctx,
            reflection_threshold=reflection_threshold,
            reflection_lookback_slices=reflection_lookback_slices,
            prefix_fraction=prefix_fraction,
            prefix_window_tokens=prefix_window_tokens,
        )
        order = order_code_dynamic_group_indices(
            scores,
            D,
            run_ids=run_ids,
        )
        out[str(problem_id)] = _rank_scale_desc(order, list(map(int, run_ids)))

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch best_of_n submission with code_baseline_v1 lcb_v5 scores")
    ap.add_argument("--base-submission", default=str(BASE_SUBMISSION), help="Base submission JSON to patch")
    ap.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT), help="cache_test root")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output submission path")
    ap.add_argument("--distance", default="ja", choices=["ja", "wj"], help="Distance used for run-view ranking")
    ap.add_argument("--distance-threads", type=int, default=12, help="Distance computation threads")
    ap.add_argument("--reflection-threshold", type=float, default=DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD)
    ap.add_argument("--reflection-lookback-slices", type=int, default=DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK)
    ap.add_argument("--prefix-fraction", type=float, default=0.20)
    ap.add_argument("--prefix-window-tokens", type=int, default=128)
    ap.add_argument("--method-name", default=DEFAULT_METHOD_NAME, help="method_name stored in submission JSON")
    ap.add_argument(
        "--lcb-override-note",
        default=DEFAULT_LCB_OVERRIDE,
        help="score_postprocess.lcb_override metadata string",
    )
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
            distance=args.distance,
            distance_threads=args.distance_threads,
            reflection_threshold=args.reflection_threshold,
            reflection_lookback_slices=args.reflection_lookback_slices,
            prefix_fraction=args.prefix_fraction,
            prefix_window_tokens=args.prefix_window_tokens,
        )
        n_problems = len(patched_scores[entry.cache_key])
        sample_sizes = sorted({len(problem_scores) for problem_scores in patched_scores[entry.cache_key].values()})
        print(f"  problems={n_problems} samples_per_problem={sample_sizes}")

    patched_payload = json.loads(json.dumps(base_payload))
    for cache_key, problem_scores in patched_scores.items():
        patched_payload["scores"][cache_key] = problem_scores

    patched_payload["method_name"] = str(args.method_name)
    patched_payload.setdefault("score_postprocess", {})["lcb_override"] = str(args.lcb_override_note)
    patched_payload["score_postprocess"]["lcb_patch_params"] = {
        "distance": str(args.distance),
        "distance_threads": int(args.distance_threads),
        "reflection_threshold": float(args.reflection_threshold),
        "reflection_lookback_slices": int(args.reflection_lookback_slices),
        "prefix_fraction": float(args.prefix_fraction),
        "prefix_window_tokens": int(args.prefix_window_tokens),
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
