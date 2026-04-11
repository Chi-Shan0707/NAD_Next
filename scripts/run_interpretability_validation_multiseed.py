#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class TrainJob:
    record_id: str
    axis: str
    method_group: str
    random_state: int
    split_seed: int
    math_model_path: Path
    science_model_path: Path
    ms_model_path: Path
    summary_path: Path
    eval_path: Path
    doc_path: Path
    registry_path: Path


def _parse_int_csv(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def _limit_threads_env() -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    return env


def _make_jobs(
    *,
    out_root: Path,
    seed_values: list[int],
    split_values: list[int],
) -> list[TrainJob]:
    jobs: list[TrainJob] = []
    for seed in seed_values:
        run_dir = out_root / f"seed_rs{seed:03d}_ss042"
        jobs.append(
            TrainJob(
                record_id=f"seed_rs{seed:03d}_ss042",
                axis="seed",
                method_group="bundle",
                random_state=int(seed),
                split_seed=42,
                math_model_path=run_dir / "models/es_svd_math_rr_r1.pkl",
                science_model_path=run_dir / "models/es_svd_science_rr_r1.pkl",
                ms_model_path=run_dir / "models/es_svd_ms_rr_r1.pkl",
                summary_path=run_dir / "scans/es_svd_ms_rr_r1_summary.json",
                eval_path=run_dir / "scans/es_svd_ms_rr_r1_eval.json",
                doc_path=run_dir / "docs/ES_SVD_MS_RR_R1.md",
                registry_path=run_dir / "registry.json",
            )
        )
    for split_seed in split_values:
        run_dir = out_root / f"split_rs042_ss{split_seed:03d}"
        jobs.append(
            TrainJob(
                record_id=f"split_rs042_ss{split_seed:03d}",
                axis="split",
                method_group="bundle",
                random_state=42,
                split_seed=int(split_seed),
                math_model_path=run_dir / "models/es_svd_math_rr_r1.pkl",
                science_model_path=run_dir / "models/es_svd_science_rr_r1.pkl",
                ms_model_path=run_dir / "models/es_svd_ms_rr_r1.pkl",
                summary_path=run_dir / "scans/es_svd_ms_rr_r1_summary.json",
                eval_path=run_dir / "scans/es_svd_ms_rr_r1_eval.json",
                doc_path=run_dir / "docs/ES_SVD_MS_RR_R1.md",
                registry_path=run_dir / "registry.json",
            )
        )
    return jobs


def _job_command(
    job: TrainJob,
    *,
    feature_cache_dir: Path,
    feature_workers: int,
    fit_workers: int,
    holdout_split: float,
    n_splits: int,
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "SVDomain/train_es_svd_ms_rr_r1.py"),
        "--holdout-split",
        str(float(holdout_split)),
        "--split-seed",
        str(int(job.split_seed)),
        "--n-splits",
        str(int(n_splits)),
        "--random-state",
        str(int(job.random_state)),
        "--feature-workers",
        str(int(feature_workers)),
        "--fit-workers",
        str(int(fit_workers)),
        "--feature-cache-dir",
        str(feature_cache_dir),
        "--out-math-model",
        str(job.math_model_path.relative_to(REPO_ROOT)),
        "--out-science-model",
        str(job.science_model_path.relative_to(REPO_ROOT)),
        "--out-combined-model",
        str(job.ms_model_path.relative_to(REPO_ROOT)),
        "--out-summary",
        str(job.summary_path.relative_to(REPO_ROOT)),
        "--out-eval",
        str(job.eval_path.relative_to(REPO_ROOT)),
        "--out-doc",
        str(job.doc_path.relative_to(REPO_ROOT)),
        "--registry-path",
        str(job.registry_path.relative_to(REPO_ROOT)),
    ]


def _run_job(
    job: TrainJob,
    *,
    feature_cache_dir: Path,
    feature_workers: int,
    fit_workers: int,
    holdout_split: float,
    n_splits: int,
) -> dict[str, Any]:
    if (
        job.math_model_path.exists()
        and job.science_model_path.exists()
        and job.ms_model_path.exists()
        and job.summary_path.exists()
        and job.eval_path.exists()
    ):
        return {
            "record_id": str(job.record_id),
            "axis": str(job.axis),
            "random_state": int(job.random_state),
            "split_seed": int(job.split_seed),
            "status": "skipped_existing",
            "artifacts": {
                "math_model_path": str(job.math_model_path),
                "science_model_path": str(job.science_model_path),
                "ms_model_path": str(job.ms_model_path),
                "summary_path": str(job.summary_path),
                "eval_path": str(job.eval_path),
                "doc_path": str(job.doc_path),
                "registry_path": str(job.registry_path),
            },
        }

    for path in [
        job.math_model_path,
        job.science_model_path,
        job.ms_model_path,
        job.summary_path,
        job.eval_path,
        job.doc_path,
        job.registry_path,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    cmd = _job_command(
        job,
        feature_cache_dir=feature_cache_dir,
        feature_workers=feature_workers,
        fit_workers=fit_workers,
        holdout_split=holdout_split,
        n_splits=n_splits,
    )
    env = _limit_threads_env()
    subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
    )
    return {
        "record_id": str(job.record_id),
        "axis": str(job.axis),
        "random_state": int(job.random_state),
        "split_seed": int(job.split_seed),
        "status": "completed",
        "artifacts": {
            "math_model_path": str(job.math_model_path),
            "science_model_path": str(job.science_model_path),
            "ms_model_path": str(job.ms_model_path),
            "summary_path": str(job.summary_path),
            "eval_path": str(job.eval_path),
            "doc_path": str(job.doc_path),
            "registry_path": str(job.registry_path),
        },
    }


def _manifest_records(job_outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for job in job_outputs:
        axis = str(job["axis"])
        random_state = int(job["random_state"])
        split_seed = int(job["split_seed"])
        artifacts = dict(job["artifacts"])
        records.extend(
            [
                {
                    "record_id": f"{job['record_id']}::math",
                    "axis": axis,
                    "method_group": "math",
                    "bundle_path": artifacts["math_model_path"],
                    "random_state": random_state,
                    "split_seed": split_seed,
                    "note": "es_svd_math_rr_r1 retrain",
                },
                {
                    "record_id": f"{job['record_id']}::science",
                    "axis": axis,
                    "method_group": "science",
                    "bundle_path": artifacts["science_model_path"],
                    "random_state": random_state,
                    "split_seed": split_seed,
                    "note": "es_svd_science_rr_r1 retrain",
                },
                {
                    "record_id": f"{job['record_id']}::ms",
                    "axis": axis,
                    "method_group": "ms",
                    "bundle_path": artifacts["ms_model_path"],
                    "random_state": random_state,
                    "split_seed": split_seed,
                    "note": "es_svd_ms_rr_r1 retrain",
                },
            ]
        )
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Run multi-seed interpretability validation training")
    ap.add_argument("--seed-values", default="13,29,42,71,101")
    ap.add_argument("--split-values", default="42,43,44")
    ap.add_argument("--parallel-jobs", type=int, default=4)
    ap.add_argument("--fit-workers", type=int, default=4)
    ap.add_argument("--feature-workers", type=int, default=1)
    ap.add_argument("--holdout-split", type=float, default=0.15)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument(
        "--feature-cache-dir",
        default="results/cache/es_svd_ms_rr_r1",
        help="Reuse existing feature cache to avoid rebuilding tensors",
    )
    ap.add_argument(
        "--out-root",
        default="results/validation/interpretability_multiseed",
    )
    ap.add_argument("--manifest-path", default="")
    ap.add_argument("--run-export", action="store_true")
    ap.add_argument(
        "--export-script",
        default="scripts/export_interpretability_validation.py",
    )
    args = ap.parse_args()

    seed_values = _parse_int_csv(args.seed_values)
    split_values = _parse_int_csv(args.split_values)
    out_root = (REPO_ROOT / str(args.out_root)).resolve()
    feature_cache_dir = (REPO_ROOT / str(args.feature_cache_dir)).resolve()
    manifest_path = (
        (REPO_ROOT / str(args.manifest_path)).resolve()
        if str(args.manifest_path).strip()
        else out_root / "manifest.json"
    )

    parallel_jobs = max(1, int(args.parallel_jobs))
    fit_workers = max(1, int(args.fit_workers))
    if parallel_jobs * fit_workers > 16:
        raise ValueError(
            f"parallel_jobs * fit_workers must be <= 16, got {parallel_jobs} * {fit_workers}"
        )

    jobs = _make_jobs(
        out_root=out_root,
        seed_values=seed_values,
        split_values=split_values,
    )
    out_root.mkdir(parents=True, exist_ok=True)

    outputs: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
        future_map = {
            executor.submit(
                _run_job,
                job,
                feature_cache_dir=feature_cache_dir,
                feature_workers=max(1, int(args.feature_workers)),
                fit_workers=fit_workers,
                holdout_split=float(args.holdout_split),
                n_splits=max(2, int(args.n_splits)),
            ): job
            for job in jobs
        }
        for future in as_completed(future_map):
            job = future_map[future]
            result = future.result()
            outputs.append(result)
            print(
                f"[done] {job.record_id} random_state={job.random_state} split_seed={job.split_seed}",
                flush=True,
            )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "out_root": str(out_root),
        "feature_cache_dir": str(feature_cache_dir),
        "parallel_jobs": parallel_jobs,
        "fit_workers": fit_workers,
        "feature_workers": max(1, int(args.feature_workers)),
        "records": _manifest_records(outputs),
        "job_outputs": sorted(outputs, key=lambda row: str(row["record_id"])),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[manifest] {manifest_path}", flush=True)

    if bool(args.run_export):
        export_cmd = [
            sys.executable,
            str(REPO_ROOT / str(args.export_script)),
            "--multiseed-manifest",
            str(manifest_path.relative_to(REPO_ROOT)),
        ]
        subprocess.run(
            export_cmd,
            cwd=str(REPO_ROOT),
            env=_limit_threads_env(),
            check=True,
        )


if __name__ == "__main__":
    main()
