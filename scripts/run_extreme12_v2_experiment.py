#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scripts.train_extreme8_selectors as train

DEFAULT_DATASETS = train.DEFAULT_DATASETS


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def _metric(stats: dict[str, Any], subset: str, score_name: str, metric_name: str) -> float | None:
    return stats.get("validation", {}).get(subset, {}).get("mean", {}).get(score_name, {}).get(metric_name)


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}%"


def _row_from_result(result: dict[str, Any]) -> dict[str, Any]:
    config = result["config"]
    stats = result["stats"]
    objective_stats = stats.get("objective_stats", {})
    selected_candidate = objective_stats.get("selected_candidate", {})
    return {
        "name": str(config["name"]),
        "objective": str(config["objective"]),
        "seed": int(config["seed"]),
        "tuple_size": int(config["tuple_size"]),
        "tuple_rule": "blind" if config.get("tuple_min_correct") is None else f"{config['tuple_min_correct']}-{config['tuple_max_correct']}",
        "best_hit1": _metric(stats, "eligible", "best_only", "hit_at_1"),
        "best_hit3": _metric(stats, "eligible", "best_only", "hit_at_3"),
        "best_pairwise": _metric(stats, "eligible", "best_only", "pairwise_accuracy"),
        "best_selacc10": _metric(stats, "eligible", "best_only", "selective_acc_at_10pct"),
        "mix_selacc10": _metric(stats, "eligible", "mix", "selective_acc_at_10pct"),
        "all_best_selacc10": _metric(stats, "all", "best_only", "selective_acc_at_10pct"),
        "inner_tune_selacc10": selected_candidate.get("metrics", {}).get("selective_acc_at_10pct"),
        "search_trace_path": result.get("search_trace_path"),
        "stats_path": result["stats_path"],
        "log_path": result["log_path"],
    }


def _select_recommended(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline12 = next(row for row in rows if row["name"] == "baseline12_pointwise")
    baseline_hit1 = float(baseline12["best_hit1"] or 0.0)
    baseline_pairwise = float(baseline12["best_pairwise"] or 0.0)

    best_agg = None
    for row in rows:
        row["passes_guardrails"] = False
        if row["objective"] != "aggregated_selacc10":
            continue
        hit1 = float(row["best_hit1"] or 0.0)
        pairwise = float(row["best_pairwise"] or 0.0)
        if hit1 < baseline_hit1 - train.DEFAULT_HIT1_GUARDRAIL_DROP:
            continue
        if pairwise < baseline_pairwise - train.DEFAULT_PAIRWISE_GUARDRAIL_DROP:
            continue
        row["passes_guardrails"] = True
        if best_agg is None:
            best_agg = row
            continue
        current_key = (
            float(row["best_selacc10"] or 0.0),
            float(row["best_hit3"] or 0.0),
            float(row["best_hit1"] or 0.0),
            float(row["best_pairwise"] or 0.0),
        )
        best_key = (
            float(best_agg["best_selacc10"] or 0.0),
            float(best_agg["best_hit3"] or 0.0),
            float(best_agg["best_hit1"] or 0.0),
            float(best_agg["best_pairwise"] or 0.0),
        )
        if current_key > best_key:
            best_agg = row

    recommended = baseline12
    if best_agg is not None and float(best_agg["best_selacc10"] or 0.0) > float(baseline12["best_selacc10"] or 0.0):
        recommended = best_agg

    recommended_name = str(recommended["name"])
    for row in rows:
        row["recommended"] = bool(row["name"] == recommended_name)

    return {
        "baseline12": baseline12,
        "best_aggregated": best_agg,
        "recommended": recommended,
    }


def _write_summary(run_dir: Path, rows: list[dict[str, Any]], selection: dict[str, Any], args: argparse.Namespace) -> None:
    summary_json = run_dir / "summary.json"
    summary_md = run_dir / "summary.md"

    payload = {
        "config": {
            "datasets": args.datasets,
            "cache_root": args.cache_root,
            "train_extra_cache_root": args.train_extra_cache_root,
            "val_cache_root": args.val_cache_root,
            "val_split": float(args.val_split),
            "split_seed": int(args.split_seed),
            "reflection_threshold": str(args.reflection_threshold),
            "num_tuples": int(args.num_tuples),
            "num_tuples_val": int(args.num_tuples_val),
            "num_tuples_tune": int(args.num_tuples_tune),
            "tune_split": float(args.tune_split),
            "tune_seed": int(args.tune_seed),
            "workers": int(args.workers),
            "seed": int(args.seed),
            "aggregate_seeds": _parse_ints(args.aggregate_seeds),
        },
        "runs": rows,
        "selection": {
            "baseline12": selection["baseline12"]["name"],
            "best_aggregated": None if selection["best_aggregated"] is None else selection["best_aggregated"]["name"],
            "recommended": selection["recommended"]["name"],
        },
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Extreme12 Aggregated-Objective Comparison",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Recommended: `{selection['recommended']['name']}`",
        f"- Baseline12: `{selection['baseline12']['name']}`",
        (
            f"- Best aggregated candidate: `{selection['best_aggregated']['name']}`"
            if selection["best_aggregated"] is not None
            else "- Best aggregated candidate: none passed guardrails"
        ),
        "",
        "| Name | Obj | Seed | Tuple | Rule | Hit@1 | Hit@3 | Pairwise | SelAcc@10 | Inner Tune SelAcc@10 | Guard | Pick | Stats | Trace |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {name} | {objective} | {seed} | {tuple_size} | {tuple_rule} | {hit1} | {hit3} | {pairwise} | {selacc} | {inner_selacc} | {guard} | {pick} | `{stats_path}` | {trace} |".format(
                name=row["name"],
                objective=row["objective"],
                seed=int(row["seed"]),
                tuple_size=int(row["tuple_size"]),
                tuple_rule=row["tuple_rule"],
                hit1=_fmt_pct(row["best_hit1"]),
                hit3=_fmt_pct(row["best_hit3"]),
                pairwise=_fmt_pct(row["best_pairwise"]),
                selacc=_fmt_pct(row["best_selacc10"]),
                inner_selacc=_fmt_pct(row["inner_tune_selacc10"]),
                guard="yes" if row.get("passes_guardrails") else "-",
                pick="yes" if row.get("recommended") else "-",
                stats_path=row["stats_path"],
                trace=f"`{row['search_trace_path']}`" if row.get("search_trace_path") else "-",
            )
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run cached Extreme12 baseline vs aggregated-selacc10 comparison")
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS), help="Comma-separated datasets")
    ap.add_argument("--cache-root", default="MUI_HUB/cache", help="Full training cache root")
    ap.add_argument("--train-extra-cache-root", default="", help="Optional additional full training cache root")
    ap.add_argument("--val-cache-root", default="/home/jovyan/public-ro/MUI_HUB/cache_train", help="Split train/validate cache root")
    ap.add_argument("--val-split", type=float, default=0.20, help="Validation fraction")
    ap.add_argument("--split-seed", type=int, default=42, help="Validation split seed")
    ap.add_argument("--out-dir", default="results/extreme12_v2_experiments", help="Output directory")
    ap.add_argument("--num-tuples", type=int, default=256, help="Training tuples per problem")
    ap.add_argument("--num-tuples-val", type=int, default=1024, help="Blind validation tuples per problem")
    ap.add_argument("--num-tuples-tune", type=int, default=train.DEFAULT_NUM_TUPLES_TUNE, help="Inner tuning tuples per problem")
    ap.add_argument("--tune-split", type=float, default=train.DEFAULT_TUNE_SPLIT, help="Inner tuning fraction")
    ap.add_argument("--tune-seed", type=int, default=42, help="Inner tuning split seed")
    ap.add_argument("--min-accuracy", type=float, default=0.10, help="Minimum problem accuracy")
    ap.add_argument("--max-accuracy", type=float, default=0.90, help="Maximum problem accuracy")
    ap.add_argument("--reflection-threshold", default="0.30", help="Reflection threshold")
    ap.add_argument("--workers", type=int, default=4, help="Extraction workers")
    ap.add_argument("--seed", type=int, default=42, help="Baseline seed")
    ap.add_argument("--aggregate-seeds", default="42,43,44", help="Comma-separated seeds for aggregated_selacc10 runs")
    ap.add_argument("--max-problems", type=int, default=None, help="Optional smoke-test cap")
    args = ap.parse_args()

    run_root = REPO_ROOT / args.out_dir
    run_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = run_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO_ROOT)

    reflection_threshold = train._resolve_threshold(str(args.reflection_threshold))
    train_specs, val_all_specs, split_summary = train._collect_specs(args)
    if not train_specs:
        raise SystemExit("No eligible training problems were found.")

    print(f"Collecting raw values once for {len(train_specs)} training problems")
    train_payloads, train_bucket_sizes = train._load_problem_payloads(
        train_specs,
        reflection_threshold=reflection_threshold,
        workers=int(args.workers),
    )
    print(f"Collecting raw values once for {len(val_all_specs)} validation problems")
    val_all_payloads, val_bucket_sizes = train._load_problem_payloads(
        val_all_specs,
        reflection_threshold=reflection_threshold,
        workers=int(args.workers),
    )

    configs: list[dict[str, Any]] = [
        {
            "name": "baseline12_pointwise",
            "objective": "pointwise",
            "tuple_size": 12,
            "tuple_min_correct": 2,
            "tuple_max_correct": 10,
            "alpha": 1.0,
            "beta": 0.35,
            "gamma": -0.50,
            "seed": int(args.seed),
            "tune_seed": int(args.tune_seed),
        },
    ]
    for agg_seed in _parse_ints(args.aggregate_seeds):
        configs.append({
            "name": f"aggregated_selacc10_s{int(agg_seed):03d}",
            "objective": "aggregated_selacc10",
            "tuple_size": 12,
            "tuple_min_correct": 2,
            "tuple_max_correct": 10,
            "alpha": 1.0,
            "beta": 0.35,
            "gamma": -0.50,
            "seed": int(agg_seed),
            "tune_seed": int(args.tune_seed) + int(agg_seed),
        })

    results: list[dict[str, Any]] = []
    datasets = train._parse_datasets(args.datasets)
    for config in configs:
        name = str(config["name"])
        log_path = run_dir / f"{name}.log"
        print(f"Running {name}")
        with log_path.open("w", encoding="utf-8") as log_file:
            def log(message: str) -> None:
                print(message)
                print(message, file=log_file, flush=True)

            log(f"Config: {json.dumps(config, sort_keys=True)}")
            result = train.train_selector_artifacts(
                train_payloads=train_payloads,
                val_all_payloads=val_all_payloads,
                split_summary=split_summary,
                train_bucket_sizes=train_bucket_sizes,
                val_bucket_sizes=val_bucket_sizes,
                reflection_threshold=float(reflection_threshold),
                out_dir=model_dir,
                artifact_prefix=name,
                objective=str(config["objective"]),
                tuple_size=int(config["tuple_size"]),
                num_tuples=int(args.num_tuples),
                num_tuples_val=int(args.num_tuples_val),
                tuple_min_correct=config.get("tuple_min_correct"),
                tuple_max_correct=config.get("tuple_max_correct"),
                datasets=datasets,
                split_seed=int(args.split_seed),
                val_split=float(args.val_split),
                alpha=float(config["alpha"]),
                beta=float(config["beta"]),
                gamma=float(config["gamma"]),
                workers=int(args.workers),
                seed=int(config["seed"]),
                tune_split=float(args.tune_split),
                tune_seed=int(config["tune_seed"]),
                num_tuples_tune=int(args.num_tuples_tune),
            )
            stats = result["stats"]
            mean_best = stats.get("validation", {}).get("eligible", {}).get("mean", {}).get("best_only", {})
            search_trace_path = None
            if str(config["objective"]) == "aggregated_selacc10":
                trace_payload = {
                    "name": name,
                    "seed": int(config["seed"]),
                    "objective": str(config["objective"]),
                    "baseline_inner_tune_metrics": stats.get("objective_stats", {}).get("baseline_inner_tune_metrics"),
                    "selected_candidate": stats.get("objective_stats", {}).get("selected_candidate"),
                    "guardrails": stats.get("objective_stats", {}).get("guardrails"),
                    "search_trace": stats.get("objective_stats", {}).get("search_trace", []),
                }
                trace_path = model_dir / f"{name}_search_trace.json"
                trace_path.write_text(json.dumps(trace_payload, indent=2), encoding="utf-8")
                search_trace_path = _display_path(trace_path)
                log(f"Search trace saved: {trace_path}")
            log(f"Model stats saved: {result['stats_path']}")
            if mean_best:
                log(
                    "Validation eligible best_only: "
                    f"Hit@1={mean_best.get('hit_at_1', 0.0):.4f} "
                    f"Hit@3={mean_best.get('hit_at_3', 0.0):.4f} "
                    f"Pairwise={(mean_best.get('pairwise_accuracy', 0.0) or 0.0):.4f} "
                    f"SelAcc@10={mean_best.get('selective_acc_at_10pct', 0.0):.4f}"
                )

        results.append({
            "config": config,
            "stats": stats,
            "log_path": _display_path(log_path),
            "stats_path": _display_path(result["stats_path"]),
            "search_trace_path": search_trace_path,
        })

    rows = [_row_from_result(result) for result in results]
    selection = _select_recommended(rows)
    _write_summary(run_dir, rows, selection, args)

    print(f"Saved experiment summary to {_display_path(run_dir / 'summary.json')}")
    print(f"Saved experiment markdown to {_display_path(run_dir / 'summary.md')}")
    print(f"Recommended config: {selection['recommended']['name']}")


if __name__ == "__main__":
    main()
