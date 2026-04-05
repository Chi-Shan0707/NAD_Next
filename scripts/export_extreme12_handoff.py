#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.ops.bestofn_extreme8 import format_threshold_tag, validate_submission_payload


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _find_run_row(summary: dict[str, Any], artifact_prefix: str) -> dict[str, Any]:
    for row in summary.get("runs", []):
        if str(row.get("name")) == str(artifact_prefix):
            return row
    raise KeyError(f"Run {artifact_prefix!r} not found in summary.json")


def _scaled_path(raw_path: Path, method: str) -> Path:
    return raw_path.with_name(f"{raw_path.stem}_scale100_{method}{raw_path.suffix}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a clean Extreme12 handoff from an experiment run directory")
    ap.add_argument(
        "--experiment-dir",
        default="results/extreme12_band_experiments/20260403_085255",
        help="Experiment directory containing summary.json and models/",
    )
    ap.add_argument(
        "--artifact-prefix",
        default="",
        help="Optional run name inside the experiment; defaults to summary selection.recommended",
    )
    ap.add_argument(
        "--cache-root",
        default="/home/jovyan/public-ro/MUI_HUB/cache_test",
        help="cache_test root used for final blind export",
    )
    ap.add_argument("--out-dir", default="submission/BestofN/extreme12/base", help="Output directory for submission JSONs")
    ap.add_argument("--work-dir", default="/tmp/bestofn_parallel_work_extreme12", help="Temporary shard workspace")
    ap.add_argument("--parallel-jobs", type=int, default=4, help="Parallel shard workers")
    ap.add_argument("--grouping", choices=("balanced_3plus3", "chunked"), default="balanced_3plus3", help="Shard grouping policy")
    ap.add_argument("--scale-method", choices=("rank", "minmax"), default="rank", help="Post-export score rescaling method")
    ap.add_argument("--submission-prefix", default="", help="Optional output/method prefix; defaults to extreme12_<artifact-prefix>")
    ap.add_argument("--max-caches", type=int, default=None, help="Optional cache cap for smoke tests")
    ap.add_argument("--max-problems", type=int, default=None, help="Optional problem cap per cache for smoke tests")
    ap.add_argument("--keep-work-dir", action="store_true", help="Keep temporary shard workspace")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    experiment_dir = REPO_ROOT / args.experiment_dir
    summary_path = experiment_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary.json: {summary_path}")

    summary = _load_json(summary_path)
    artifact_prefix = str(args.artifact_prefix or summary.get("selection", {}).get("recommended") or "").strip()
    if not artifact_prefix:
        raise SystemExit("Could not resolve artifact prefix from --artifact-prefix or summary selection")
    run_row = _find_run_row(summary, artifact_prefix)

    stats_path = REPO_ROOT / str(run_row.get("stats_path"))
    if not stats_path.exists():
        raise SystemExit(f"Missing stats JSON for {artifact_prefix}: {stats_path}")
    stats = _load_json(stats_path)

    best_model_path = REPO_ROOT / str(stats.get("model_paths", {}).get("best", ""))
    worst_model_path = REPO_ROOT / str(stats.get("model_paths", {}).get("worst", ""))
    if not best_model_path.exists():
        raise SystemExit(f"Missing best model: {best_model_path}")
    if not worst_model_path.exists():
        raise SystemExit(f"Missing worst model: {worst_model_path}")

    tuple_size = int(stats["tuple_size"])
    num_tuples = int(stats["num_tuples_validation"])
    reflection_threshold = float(stats["reflection_threshold"])
    tag = format_threshold_tag(reflection_threshold)
    submission_prefix = str(args.submission_prefix or f"extreme12_{artifact_prefix}")
    best_method_name = f"{submission_prefix}_best_only_{tag}_t{num_tuples}"
    mix_method_name = f"{submission_prefix}_mix_{tag}_t{num_tuples}"
    best_filename = f"{best_method_name}.json"
    mix_filename = f"{mix_method_name}.json"

    export_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "export_bestofn_submissions_parallel.py"),
        "--cache-root", str(args.cache_root),
        "--best-model", _display_path(best_model_path),
        "--worst-model", _display_path(worst_model_path),
        "--out-dir", str(args.out_dir),
        "--work-dir", str(args.work_dir),
        "--parallel-jobs", str(int(args.parallel_jobs)),
        "--grouping", str(args.grouping),
        "--tuple-size", str(tuple_size),
        "--num-tuples", str(num_tuples),
        "--reflection-threshold", f"{reflection_threshold:.2f}",
        "--best-method-name", best_method_name,
        "--mix-method-name", mix_method_name,
        "--best-filename", best_filename,
        "--mix-filename", mix_filename,
    ]
    if args.max_caches is not None:
        export_cmd.extend(["--max-caches", str(int(args.max_caches))])
    if args.max_problems is not None:
        export_cmd.extend(["--max-problems", str(int(args.max_problems))])
    if args.keep_work_dir:
        export_cmd.append("--keep-work-dir")
    _run(export_cmd)

    out_dir = REPO_ROOT / args.out_dir
    best_raw = out_dir / best_filename
    mix_raw = out_dir / mix_filename
    if not best_raw.exists() or not mix_raw.exists():
        raise SystemExit("Raw submission export did not produce both best_only and mix JSON files")

    rescale_commands = []
    for raw_path in (best_raw, mix_raw):
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "rescale_bestofn_submission_scores.py"),
            str(raw_path),
            "--method", str(args.scale_method),
        ]
        _run(cmd)
        rescale_commands.append(cmd)

    best_scaled = _scaled_path(best_raw, str(args.scale_method))
    mix_scaled = _scaled_path(mix_raw, str(args.scale_method))
    if not best_scaled.exists() or not mix_scaled.exists():
        raise SystemExit("Scaled submission export did not produce both scale100 JSON files")

    raw_best_summary = validate_submission_payload(_load_json(best_raw))
    raw_mix_summary = validate_submission_payload(_load_json(mix_raw))
    scaled_best_summary = validate_submission_payload(_load_json(best_scaled))
    scaled_mix_summary = validate_submission_payload(_load_json(mix_scaled))

    manifest = {
        "experiment_dir": _display_path(experiment_dir),
        "summary_path": _display_path(summary_path),
        "artifact_prefix": artifact_prefix,
        "selection": summary.get("selection", {}),
        "run_row": run_row,
        "stats_path": _display_path(stats_path),
        "model_paths": {
            "best": _display_path(best_model_path),
            "worst": _display_path(worst_model_path),
        },
        "export_config": {
            "cache_root": str(args.cache_root),
            "tuple_size": int(tuple_size),
            "num_tuples": int(num_tuples),
            "reflection_threshold": float(reflection_threshold),
            "parallel_jobs": int(args.parallel_jobs),
            "grouping": str(args.grouping),
            "scale_method": str(args.scale_method),
            "submission_prefix": submission_prefix,
            "max_caches": None if args.max_caches is None else int(args.max_caches),
            "max_problems": None if args.max_problems is None else int(args.max_problems),
        },
        "artifacts": {
            "best_raw": _display_path(best_raw),
            "mix_raw": _display_path(mix_raw),
            "best_scaled": _display_path(best_scaled),
            "mix_scaled": _display_path(mix_scaled),
            "recommended_submission": _display_path(best_scaled),
        },
        "validation": {
            "raw_best": raw_best_summary,
            "raw_mix": raw_mix_summary,
            "scaled_best": scaled_best_summary,
            "scaled_mix": scaled_mix_summary,
        },
        "commands": {
            "export": export_cmd,
            "rescale": rescale_commands,
        },
    }

    manifest_path = out_dir / f"{submission_prefix}_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved export manifest to {_display_path(manifest_path)}")
    print(f"Recommended submission JSON: {_display_path(best_scaled)}")


if __name__ == "__main__":
    main()
