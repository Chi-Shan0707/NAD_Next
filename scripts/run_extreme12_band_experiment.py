#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATASETS = (
    "aime24",
    "aime25",
    "brumo25",
    "gpqa",
    "hmmt25",
    "livecodebench_v5",
)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _parse_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def _metric(stats: dict[str, Any], subset: str, score_name: str, metric_name: str) -> float | None:
    return stats.get("validation", {}).get(subset, {}).get("mean", {}).get(score_name, {}).get(metric_name)


def _run_training(run_dir: Path, config: dict[str, Any], common_args: argparse.Namespace) -> dict[str, Any]:
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    name = str(config["name"])
    log_path = run_dir / f"{name}.log"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_extreme8_selectors.py"),
        "--datasets", common_args.datasets,
        "--cache-root", common_args.cache_root,
        "--train-extra-cache-root", common_args.train_extra_cache_root,
        "--val-cache-root", common_args.val_cache_root,
        "--val-split", str(float(common_args.val_split)),
        "--split-seed", str(int(common_args.split_seed)),
        "--out", str(model_dir),
        "--artifact-prefix", name,
        "--objective", str(config["objective"]),
        "--tuple-size", str(int(config["tuple_size"])),
        "--num-tuples", str(int(common_args.num_tuples)),
        "--num-tuples-val", str(int(common_args.num_tuples_val)),
        "--min-accuracy", str(float(common_args.min_accuracy)),
        "--max-accuracy", str(float(common_args.max_accuracy)),
        "--reflection-threshold", str(common_args.reflection_threshold),
        "--workers", str(int(common_args.workers)),
        "--seed", str(int(common_args.seed)),
        "--alpha", str(float(config.get("alpha", 1.0))),
        "--beta", str(float(config.get("beta", 0.35))),
        "--gamma", str(float(config.get("gamma", -0.5))),
    ]
    if config.get("tuple_min_correct") is not None:
        cmd.extend(["--tuple-min-correct", str(int(config["tuple_min_correct"]))])
    if config.get("tuple_max_correct") is not None:
        cmd.extend(["--tuple-max-correct", str(int(config["tuple_max_correct"]))])
    if common_args.max_problems is not None:
        cmd.extend(["--max-problems", str(int(common_args.max_problems))])

    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise SystemExit(f"Training failed for {name}; see log {log_path}")

    stats_path = model_dir / f"{name}_stats.json"
    if not stats_path.exists():
        raise SystemExit(f"Missing stats file for {name}: {stats_path}")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return {
        "config": config,
        "stats": stats,
        "log_path": _display_path(log_path),
        "stats_path": _display_path(stats_path),
    }


def _row_from_result(result: dict[str, Any]) -> dict[str, Any]:
    config = result["config"]
    stats = result["stats"]
    row = {
        "name": str(config["name"]),
        "objective": str(config["objective"]),
        "tuple_size": int(config["tuple_size"]),
        "tuple_rule": "blind" if config.get("tuple_min_correct") is None else f"{config['tuple_min_correct']}-{config['tuple_max_correct']}",
        "alpha": float(config.get("alpha", 1.0)),
        "beta": float(config.get("beta", 0.35)),
        "gamma": float(config.get("gamma", -0.5)),
        "best_hit1": _metric(stats, "eligible", "best_only", "hit_at_1"),
        "best_hit3": _metric(stats, "eligible", "best_only", "hit_at_3"),
        "best_pairwise": _metric(stats, "eligible", "best_only", "pairwise_accuracy"),
        "best_selacc10": _metric(stats, "eligible", "best_only", "selective_acc_at_10pct"),
        "mix_selacc10": _metric(stats, "eligible", "mix", "selective_acc_at_10pct"),
        "all_best_selacc10": _metric(stats, "all", "best_only", "selective_acc_at_10pct"),
        "stats_path": result["stats_path"],
        "log_path": result["log_path"],
    }
    return row


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}%"


def _select_recommended(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline12 = next(row for row in rows if row["name"] == "baseline12_pointwise")
    baseline_hit1 = float(baseline12["best_hit1"] or 0.0)
    baseline_pairwise = float(baseline12["best_pairwise"] or 0.0)

    best_band = None
    for row in rows:
        row["passes_guardrails"] = False
        if row["objective"] != "band_reward":
            continue
        hit1 = float(row["best_hit1"] or 0.0)
        pairwise = float(row["best_pairwise"] or 0.0)
        if hit1 < baseline_hit1 - 0.01:
            continue
        if pairwise < baseline_pairwise - 0.005:
            continue
        row["passes_guardrails"] = True
        if best_band is None:
            best_band = row
            continue
        current_key = (
            float(row["best_selacc10"] or 0.0),
            float(row["best_hit3"] or 0.0),
        )
        best_key = (
            float(best_band["best_selacc10"] or 0.0),
            float(best_band["best_hit3"] or 0.0),
        )
        if current_key > best_key:
            best_band = row

    recommended = baseline12
    if best_band is not None and float(best_band["best_selacc10"] or 0.0) > float(baseline12["best_selacc10"] or 0.0):
        recommended = best_band
    recommended_name = str(recommended["name"])
    for row in rows:
        row["recommended"] = bool(row["name"] == recommended_name)
    return {
        "baseline12": baseline12,
        "best_band": best_band,
        "recommended": recommended,
    }


def _write_summary(run_dir: Path, rows: list[dict[str, Any]], selection: dict[str, Any], common_args: argparse.Namespace) -> None:
    summary_json = run_dir / "summary.json"
    summary_md = run_dir / "summary.md"

    payload = {
        "config": {
            "datasets": common_args.datasets,
            "cache_root": common_args.cache_root,
            "train_extra_cache_root": common_args.train_extra_cache_root,
            "val_cache_root": common_args.val_cache_root,
            "val_split": float(common_args.val_split),
            "split_seed": int(common_args.split_seed),
            "reflection_threshold": str(common_args.reflection_threshold),
            "num_tuples": int(common_args.num_tuples),
            "num_tuples_val": int(common_args.num_tuples_val),
            "workers": int(common_args.workers),
            "seed": int(common_args.seed),
            "beta_grid": _parse_floats(common_args.beta_grid),
            "gamma_grid": _parse_floats(common_args.gamma_grid),
        },
        "runs": rows,
        "selection": {
            "baseline12": None if selection["baseline12"] is None else selection["baseline12"]["name"],
            "best_band": None if selection["best_band"] is None else selection["best_band"]["name"],
            "recommended": selection["recommended"]["name"],
        },
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Extreme12 Band-Reward Comparison",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Recommended: `{selection['recommended']['name']}`",
        f"- Baseline12: `{selection['baseline12']['name']}`",
        f"- Best band candidate: `{selection['best_band']['name']}`" if selection["best_band"] is not None else "- Best band candidate: none passed guardrails",
        "",
        "| Name | Obj | Tuple | Rule | β | γ | Hit@1 | Hit@3 | Pairwise | SelAcc@10 | Mix SelAcc@10 | Guard | Pick | Stats |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {name} | {objective} | {tuple_size} | {tuple_rule} | {beta:.2f} | {gamma:.2f} | {hit1} | {hit3} | {pairwise} | {selacc} | {mix_selacc} | {guard} | {pick} | `{stats_path}` |".format(
                name=row["name"],
                objective=row["objective"],
                tuple_size=row["tuple_size"],
                tuple_rule=row["tuple_rule"],
                beta=float(row["beta"]),
                gamma=float(row["gamma"]),
                hit1=_fmt_pct(row["best_hit1"]),
                hit3=_fmt_pct(row["best_hit3"]),
                pairwise=_fmt_pct(row["best_pairwise"]),
                selacc=_fmt_pct(row["best_selacc10"]),
                mix_selacc=_fmt_pct(row["mix_selacc10"]),
                guard="yes" if row.get("passes_guardrails") else "-",
                pick="yes" if row.get("recommended") else "-",
                stats_path=row["stats_path"],
            )
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run 8-pointwise / 12-pointwise / 12-band_reward comparison")
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS), help="Comma-separated datasets")
    ap.add_argument("--cache-root", default="MUI_HUB/cache", help="Full training cache root")
    ap.add_argument("--train-extra-cache-root", default="", help="Optional additional full training cache root")
    ap.add_argument("--val-cache-root", default="/home/jovyan/public-ro/MUI_HUB/cache_train", help="Split train/validate cache root")
    ap.add_argument("--val-split", type=float, default=0.20, help="Validation fraction")
    ap.add_argument("--split-seed", type=int, default=42, help="Validation split seed")
    ap.add_argument("--out-dir", default="results/extreme12_band_experiments", help="Output directory")
    ap.add_argument("--num-tuples", type=int, default=256, help="Training tuples per problem")
    ap.add_argument("--num-tuples-val", type=int, default=1024, help="Blind validation tuples per problem")
    ap.add_argument("--min-accuracy", type=float, default=0.10, help="Minimum problem accuracy")
    ap.add_argument("--max-accuracy", type=float, default=0.90, help="Maximum problem accuracy")
    ap.add_argument("--reflection-threshold", default="0.30", help="Reflection threshold")
    ap.add_argument("--workers", type=int, default=4, help="Extraction workers")
    ap.add_argument("--seed", type=int, default=42, help="Base seed")
    ap.add_argument("--beta-grid", default="0.25,0.35,0.50", help="Comma-separated beta values")
    ap.add_argument("--gamma-grid", default="-0.25,-0.50,-1.00", help="Comma-separated gamma values")
    ap.add_argument("--max-problems", type=int, default=None, help="Optional smoke-test cap")
    args = ap.parse_args()

    run_root = REPO_ROOT / args.out_dir
    run_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = run_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO_ROOT)

    configs = [
        {
            "name": "baseline8_pointwise",
            "objective": "pointwise",
            "tuple_size": 8,
            "alpha": 1.0,
            "beta": 0.35,
            "gamma": -0.50,
            "tuple_min_correct": None,
            "tuple_max_correct": None,
        },
        {
            "name": "baseline12_pointwise",
            "objective": "pointwise",
            "tuple_size": 12,
            "alpha": 1.0,
            "beta": 0.35,
            "gamma": -0.50,
            "tuple_min_correct": 2,
            "tuple_max_correct": 10,
        },
    ]
    for beta in _parse_floats(args.beta_grid):
        for gamma in _parse_floats(args.gamma_grid):
            configs.append({
                "name": f"band12_b{int(round(beta * 100)):03d}_g{int(round(abs(gamma) * 100)):03d}{'m' if gamma < 0 else 'p'}",
                "objective": "band_reward",
                "tuple_size": 12,
                "alpha": 1.0,
                "beta": float(beta),
                "gamma": float(gamma),
                "tuple_min_correct": 2,
                "tuple_max_correct": 10,
            })

    results = []
    for config in configs:
        print(f"Running {config['name']}")
        results.append(_run_training(run_dir, config, args))

    rows = [_row_from_result(result) for result in results]
    selection = _select_recommended(rows)
    _write_summary(run_dir, rows, selection, args)

    print(f"Saved experiment summary to {_display_path(run_dir / 'summary.json')}")
    print(f"Saved experiment markdown to {_display_path(run_dir / 'summary.md')}")
    print(f"Recommended config: {selection['recommended']['name']}")


if __name__ == "__main__":
    main()
