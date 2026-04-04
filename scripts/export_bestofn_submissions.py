#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.ops.bestofn_extreme8 import (
    build_submission_payload,
    default_method_name,
    default_submission_filename,
    discover_cache_entries,
    load_model,
    score_cache_entry,
    summarize_cache_scores,
    validate_cache_scores,
    validate_submission_payload,
    write_submission_payload,
)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _parse_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Export blind Extreme8 Best-of-N submissions from cache_test")
    ap.add_argument(
        "--cache-root",
        default="/home/jovyan/public-ro/MUI_HUB/cache_test",
        help="Root directory containing cache_test model/dataset subdirectories",
    )
    ap.add_argument("--best-model", default="models/ml_selectors/extreme8_best.pkl", help="Best-model path")
    ap.add_argument("--worst-model", default="models/ml_selectors/extreme8_worst.pkl", help="Worst-model path")
    ap.add_argument("--out-dir", default="submission/BestofN", help="Output directory")
    ap.add_argument("--tuple-size", type=int, default=8, help="Tuple size")
    ap.add_argument("--num-tuples", type=int, default=1024, help="Random tuples per problem")
    ap.add_argument("--reflection-threshold", type=float, default=0.30, help="Reflection threshold at inference time")
    ap.add_argument("--seed", type=int, default=42, help="Base random seed")
    ap.add_argument("--cache-keys", default="", help="Optional comma-separated cache keys to include")
    ap.add_argument("--max-caches", type=int, default=None, help="Optional max caches for smoke tests")
    ap.add_argument("--max-problems", type=int, default=None, help="Optional max problems per cache for smoke tests")
    ap.add_argument("--expected-samples-per-problem", type=int, default=64, help="Expected sample count per problem")
    ap.add_argument("--best-method-name", default="", help="Optional override for best-only method_name")
    ap.add_argument("--mix-method-name", default="", help="Optional override for mix method_name")
    ap.add_argument("--best-filename", default="", help="Optional override for best-only filename")
    ap.add_argument("--mix-filename", default="", help="Optional override for mix filename")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)

    best_model = load_model(REPO_ROOT / args.best_model)
    worst_model = load_model(REPO_ROOT / args.worst_model)
    entries = discover_cache_entries(args.cache_root)
    requested_cache_keys = set(_parse_csv(args.cache_keys))
    if requested_cache_keys:
        entries = [entry for entry in entries if entry.cache_key in requested_cache_keys]
    if args.max_caches is not None:
        entries = entries[: max(0, int(args.max_caches))]
    if not entries:
        raise SystemExit("No cache_test entries matched the requested filters.")

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scoring {len(entries)} cache entries from {args.cache_root}")
    print(f"Using best model: {_display_path(REPO_ROOT / args.best_model)}")
    print(f"Using worst model: {_display_path(REPO_ROOT / args.worst_model)}")
    print(
        "Config: "
        f"tuple_size={int(args.tuple_size)}, "
        f"num_tuples={int(args.num_tuples)}, "
        f"reflection_threshold={float(args.reflection_threshold):.2f}, "
        f"seed={int(args.seed)}"
    )

    cache_scores_list = []
    for cache_index, entry in enumerate(entries):
        print(f"[{cache_index + 1}/{len(entries)}] {entry.cache_key} <- {entry.cache_root}")
        cache_scores = score_cache_entry(
            entry=entry,
            best_model=best_model,
            worst_model=worst_model,
            reflection_threshold=float(args.reflection_threshold),
            num_tuples=int(args.num_tuples),
            tuple_size=int(args.tuple_size),
            seed=int(args.seed) + cache_index * 100_000,
            max_problems=args.max_problems,
        )
        validate_cache_scores(
            cache_scores,
            expected_samples_per_problem=(
                int(args.expected_samples_per_problem)
                if args.expected_samples_per_problem is not None
                else None
            ),
        )
        stats = summarize_cache_scores(cache_scores)
        print(
            f"  -> problems={stats['n_problems']}, "
            f"samples={stats['n_samples']}, "
            f"tuples/problem={int(args.num_tuples)}"
        )
        cache_scores_list.append(cache_scores)

    expected_cache_keys = [cache_scores.entry.cache_key for cache_scores in cache_scores_list]
    export_specs = {
        "best_only": {
            "method_name": args.best_method_name
            or default_method_name("best_only", float(args.reflection_threshold), int(args.num_tuples)),
            "filename": args.best_filename
            or default_submission_filename("best_only", float(args.reflection_threshold), int(args.num_tuples)),
        },
        "mix": {
            "method_name": args.mix_method_name
            or default_method_name("mix", float(args.reflection_threshold), int(args.num_tuples)),
            "filename": args.mix_filename
            or default_submission_filename("mix", float(args.reflection_threshold), int(args.num_tuples)),
        },
    }

    for score_name, spec in export_specs.items():
        payload = build_submission_payload(
            cache_scores_list=cache_scores_list,
            score_name=score_name,
            method_name=spec["method_name"],
        )
        summary = validate_submission_payload(payload, expected_cache_keys=expected_cache_keys)
        out_path = write_submission_payload(payload, out_dir / spec["filename"])
        print(
            f"Saved {score_name} submission to {_display_path(out_path)} "
            f"(cache_keys={summary['cache_keys']}, problems={summary['problems']}, samples={summary['samples']})"
        )


if __name__ == "__main__":
    main()
