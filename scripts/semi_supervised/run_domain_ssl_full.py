#!/usr/bin/env python3
"""Full training launcher for domain-specific contrastive SSL.

Domain order: coding → science → math  (user-specified sequential order).

CPU utilization strategy
------------------------
BLAS threading is counterproductive for our batch-sized matrix operations
(768×768 similarity matrices, batch=256-384): benchmarks show 1 thread is
optimal, more threads add overhead.  Maximum CPU utilization is therefore
achieved via PROCESS-level parallelism — running all rank variants for the
current domain simultaneously, one OS process per (rank, weak) combination:

  coding:  6 processes — r∈{4,8,16} × {non-weak, weak}       → 6 cores
  science: 3 processes — r∈{4,8,16}                          → 3 cores
  math:    3 processes — r∈{4,8,16}                          → 3 cores

Each process runs with OMP_NUM_THREADS=1 (benchmark-verified optimal).

Logging
-------
  logs/domain_ssl_<timestamp>/
    run.log              master log (all phases)
    coding_r4.log        per-rank training output
    coding_r8.log
    coding_r16.log
    coding_r4_weak.log
    coding_r8_weak.log
    coding_r16_weak.log
    science_r4.log  ...
    math_r4.log     ...
    study.log            evaluation phase

Outputs (from train_domain_specific_ssl.py)
------------------------------------------
  results/cache/domain_ssl/         per-rank bundle pkl files
  results/tables/domain_specific_ssl.csv
  docs/DOMAIN_SPECIFIC_SSL.md

Usage:
  python3 scripts/semi_supervised/run_domain_ssl_full.py
  python3 scripts/semi_supervised/run_domain_ssl_full.py \\
      --ssl-ranks 4 8 16 --n-epochs 300 \\
      --out-dir results/cache/domain_ssl \\
      --log-dir logs
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import pickle
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ── IMPORTANT: set OMP before any import of numpy / scipy / sklearn ───────────
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ─── Logging setup ────────────────────────────────────────────────────────────

def _setup_logger(log_dir: Path, name: str = "run") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # File handler (DEBUG level — full detail)
    fh = logging.FileHandler(log_dir / f"{name}.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Stream handler (INFO level — progress summary)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# ─── Worker function (module-level — required for spawn context) ──────────────

def _worker_fn(job: dict[str, Any]) -> dict[str, Any]:
    """Train one (domain, rank, use_pairwise) bundle.

    Runs in a subprocess; sets OMP_NUM_THREADS=1 before importing numpy.
    Saves bundle to out_dir/bundle_{domain}_r{rank}[_weak].pkl.
    Returns summary dict.
    """
    # Must override threading limits before any BLAS library is loaded
    for env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[env] = "1"

    repo_root = str(job["repo_root"])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import numpy as np

    # Redirect stdout/stderr to per-rank log file
    log_path = Path(job["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    domain: str = job["domain"]
    rank: int = job["rank"]
    use_pairwise: bool = job["use_pairwise"]
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_weak" if use_pairwise else ""
    bundle_path = out_dir / f"bundle_{domain}_r{rank}{suffix}.pkl"

    # Load domain matrix from temp file (written by main process)
    with open(job["data_file"], "rb") as fh:
        data = pickle.load(fh)

    X_all: np.ndarray = data["X_all"]
    y_all: np.ndarray = data["y_all"]
    groups_all: np.ndarray = data["groups_all"]
    X_proxy = data.get("X_proxy_all")

    from scripts.semi_supervised.train_domain_specific_ssl import pretrain_domain_ssl_basis

    t0 = time.perf_counter()
    label = f"{domain} r={rank}{'(weak)' if use_pairwise else ''}"

    # Tee to log file
    import io
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    buf = io.StringIO()

    class _Tee(io.IOBase):
        def __init__(self, *streams):
            self._streams = streams
        def write(self, data):
            for s in self._streams:
                s.write(data)
                s.flush()
            return len(data)
        def flush(self):
            for s in self._streams:
                s.flush()

    with open(log_path, "w", encoding="utf-8", buffering=1) as log_fh:
        tee = _Tee(log_fh, _orig_stdout)
        sys.stdout = sys.stderr = tee   # type: ignore[assignment]
        try:
            print(f"[{label}] worker start  {datetime.now(timezone.utc).isoformat()}")
            print(f"[{label}] X={X_all.shape}  y_pos={int(np.sum(y_all==1))}  y_neg={int(np.sum(y_all==0))}")

            bundle = pretrain_domain_ssl_basis(
                X_all,
                y_all,
                groups_all,
                domain=domain,
                r=rank,
                use_pairwise=use_pairwise,
                X_proxy_all=X_proxy,
                **job["kwargs"],
            )

            t1 = time.perf_counter()
            elapsed = t1 - t0
            print(f"[{label}] training done in {elapsed:.1f}s  final_loss={bundle['final_loss']:.5f}")

            with open(bundle_path, "wb") as fh:
                pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[{label}] bundle saved → {bundle_path}")

        except Exception as exc:
            import traceback
            traceback.print_exc()
            sys.stdout = _orig_stdout
            sys.stderr = _orig_stderr
            return {
                "domain": domain,
                "rank": rank,
                "use_pairwise": use_pairwise,
                "bundle_path": None,
                "elapsed": time.perf_counter() - t0,
                "error": str(exc),
            }
        finally:
            sys.stdout = _orig_stdout
            sys.stderr = _orig_stderr

    return {
        "domain": domain,
        "rank": rank,
        "use_pairwise": use_pairwise,
        "bundle_path": str(bundle_path),
        "elapsed": elapsed,
        "final_loss": bundle["final_loss"],
        "error": None,
    }


# ─── Main training orchestrator ───────────────────────────────────────────────

def _run_phase(
    domain: str,
    jobs: list[dict[str, Any]],
    log: logging.Logger,
    max_workers: int,
) -> list[dict[str, Any]]:
    """Run all jobs for one domain in parallel.  Returns list of result dicts."""
    log.info(
        f"{'─'*60}\n"
        f"  PHASE: {domain.upper()}\n"
        f"  Jobs: {len(jobs)} parallel  (r × weak variants)\n"
        f"{'─'*60}"
    )
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(max_workers, len(jobs)),
                             mp_context=ctx) as pool:
        futures = {pool.submit(_worker_fn, job): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            label = f"{job['domain']} r={job['rank']}{'(weak)' if job['use_pairwise'] else ''}"
            try:
                res = fut.result()
                if res["error"]:
                    log.error(f"  FAILED {label}: {res['error']}")
                else:
                    log.info(
                        f"  DONE  {label}  elapsed={res['elapsed']:.1f}s  "
                        f"loss={res['final_loss']:.5f}"
                    )
                results.append(res)
            except Exception as exc:
                log.error(f"  EXCEPTION {label}: {exc}")
                results.append({"domain": job["domain"], "rank": job["rank"],
                                 "error": str(exc)})

    elapsed = time.perf_counter() - t0
    ok = sum(1 for r in results if not r.get("error"))
    log.info(f"  Phase {domain.upper()} complete: {ok}/{len(jobs)} OK  total={elapsed:.1f}s\n")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Domain-specific SSL full training launcher"
    )
    ap.add_argument("--prebuilt-cache-dir", default="results/cache/es_svd_ms_rr_r1")
    ap.add_argument("--shared-ssl-pkl",
                    default="results/cache/semisup_svdomain/ssl_bundles.pkl")
    ap.add_argument("--supervised-bundle",
                    default="models/ml_selectors/es_svd_ms_rr_r1.pkl")
    ap.add_argument("--out-dir", default="results/cache/domain_ssl")
    ap.add_argument("--out-csv", default="results/tables/domain_specific_ssl.csv")
    ap.add_argument("--out-doc", default="docs/DOMAIN_SPECIFIC_SSL.md")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--ssl-ranks", nargs="+", type=int, default=[4, 8, 16])
    ap.add_argument("--n-epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--lam-pred", type=float, default=0.3)
    ap.add_argument("--lam-view", type=float, default=0.1)
    ap.add_argument("--lam-pair", type=float, default=0.5)
    ap.add_argument("--lam-proxy", type=float, default=0.1)
    ap.add_argument("--lr-max", type=float, default=0.01)
    ap.add_argument("--lr-min", type=float, default=1e-4)
    ap.add_argument("--holdout-split", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--refresh-ssl-cache", action="store_true")
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="Short smoke run (20 epochs, r=4 8 only)")
    args = ap.parse_args()

    def rp(s: str) -> Path:
        p = Path(s)
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    out_dir = rp(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    log_dir = rp(args.log_dir) / f"domain_ssl_{ts}"
    log = _setup_logger(log_dir, name="run")

    ssl_ranks = tuple(sorted(set(int(r) for r in args.ssl_ranks)))
    if args.smoke:
        ssl_ranks = tuple(r for r in ssl_ranks if r <= 8)
        args.n_epochs = min(args.n_epochs, 20)

    log.info("=" * 70)
    log.info("  Domain-Specific Contrastive SSL — Full Training Run")
    log.info(f"  Started : {datetime.now(timezone.utc).isoformat()}")
    log.info(f"  Ranks   : {ssl_ranks}")
    log.info(f"  Epochs  : {args.n_epochs}")
    log.info(f"  Log dir : {log_dir}")
    log.info(f"  Out dir : {out_dir}")
    log.info(f"  CPUs    : {os.cpu_count()}")
    log.info("=" * 70)

    # ── Load feature stores (main process, once) ──────────────────────────────
    from scripts.semi_supervised.train_semisup_svdomain import _load_prebuilt_stores
    from scripts.semi_supervised.train_domain_specific_ssl import (
        _extract_domain_matrix,
        _run_domain_ssl_study,
        _write_csv,
        _write_markdown_domain,
    )
    from SVDomain.train_es_svd_ms_rr_r1 import (
        _build_holdout_problem_map,
        _split_feature_store,
    )
    from nad.ops.earlystop_svd import load_earlystop_svd_bundle

    prebuilt_dir = rp(args.prebuilt_cache_dir)
    log.info(f"Loading prebuilt feature stores from {prebuilt_dir} ...")
    full_store = _load_prebuilt_stores(prebuilt_dir)
    if not full_store:
        log.error("No feature stores found — aborting.")
        sys.exit(1)
    log.info(f"Loaded {len(full_store)} payloads  "
             f"({sum(p.get('samples',0) for p in full_store)} total samples)")

    # ── Train/holdout split ───────────────────────────────────────────────────
    log.info("Splitting 85/15 train/holdout ...")
    hmap, _ = _build_holdout_problem_map(
        full_store,
        holdout_split=float(args.holdout_split),
        split_seed=int(args.split_seed),
    )
    train_store, holdout_store, _ = _split_feature_store(
        full_store, holdout_problem_map=hmap
    )

    train_by_domain: dict[str, list] = {}
    holdout_by_domain: dict[str, list] = {}
    for dom in ("math", "science", "coding"):
        train_by_domain[dom] = [p for p in train_store if p["domain"] == dom]
        holdout_by_domain[dom] = [p for p in holdout_store if p["domain"] == dom]
        ntr = sum(p.get("samples", 0) for p in train_by_domain[dom])
        nho = sum(p.get("samples", 0) for p in holdout_by_domain[dom])
        log.info(f"  {dom}: train={len(train_by_domain[dom])} payloads ({ntr} samples)  "
                 f"holdout={len(holdout_by_domain[dom])} payloads ({nho} samples)")

    # ── Extract domain matrices and save to temp files ────────────────────────
    log.info("Extracting domain feature matrices ...")
    tmp_dir = Path(tempfile.mkdtemp(prefix="domain_ssl_data_"))
    domain_data_files: dict[str, str] = {}

    for dom in ("coding", "science", "math"):
        X_all, y_all, groups_all = _extract_domain_matrix(
            train_by_domain[dom], dom
        )
        log.info(f"  {dom}: X={X_all.shape}  "
                 f"pos={int((y_all==1).sum())}  neg={int((y_all==0).sum())}")

        data_file = tmp_dir / f"data_{dom}.pkl"
        with data_file.open("wb") as fh:
            pickle.dump(
                {"X_all": X_all, "y_all": y_all, "groups_all": groups_all},
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        domain_data_files[dom] = str(data_file)

    # Shared kwargs for pretrain_domain_ssl_basis
    train_kwargs = dict(
        n_epochs=int(args.n_epochs),
        batch=int(args.batch_size),
        lr_max=float(args.lr_max),
        lr_min=float(args.lr_min),
        seed=int(args.split_seed),
        smoke=bool(args.smoke),
        tau=float(args.tau),
        lam_pred=float(args.lam_pred),
        lam_view=float(args.lam_view),
        lam_pair=float(args.lam_pair),
        lam_proxy=float(args.lam_proxy),
    )

    # ── Phase helper — build job list for one domain ──────────────────────────
    def _build_jobs(domain: str, include_weak: bool) -> list[dict[str, Any]]:
        jobs = []
        for r in ssl_ranks:
            # Check if bundle already exists (skip if not refreshing)
            suffix_std = ""
            bundle_std = out_dir / f"bundle_{domain}_r{r}.pkl"
            if not args.refresh_ssl_cache and bundle_std.exists():
                log.info(f"  skip {domain} r={r} (cached)")
            else:
                jobs.append({
                    "domain": domain,
                    "rank": r,
                    "use_pairwise": False,
                    "data_file": domain_data_files[domain],
                    "out_dir": str(out_dir),
                    "log_file": str(log_dir / f"{domain}_r{r}.log"),
                    "kwargs": train_kwargs,
                    "repo_root": str(REPO_ROOT),
                })

            if include_weak:
                bundle_weak = out_dir / f"bundle_{domain}_r{r}_weak.pkl"
                if not args.refresh_ssl_cache and bundle_weak.exists():
                    log.info(f"  skip {domain} r={r} weak (cached)")
                else:
                    jobs.append({
                        "domain": domain,
                        "rank": r,
                        "use_pairwise": True,
                        "data_file": domain_data_files[domain],
                        "out_dir": str(out_dir),
                        "log_file": str(log_dir / f"{domain}_r{r}_weak.log"),
                        "kwargs": {**train_kwargs, "seed": int(args.split_seed) + 1000},
                        "repo_root": str(REPO_ROOT),
                    })
        return jobs

    # ── Phase 1: CODING ───────────────────────────────────────────────────────
    coding_jobs = _build_jobs("coding", include_weak=True)
    coding_results = _run_phase(
        "coding", coding_jobs, log,
        max_workers=len(ssl_ranks) * 2,  # r=4/8/16 × non-weak/weak
    )

    # ── Phase 2: SCIENCE ──────────────────────────────────────────────────────
    science_jobs = _build_jobs("science", include_weak=False)
    science_results = _run_phase(
        "science", science_jobs, log,
        max_workers=len(ssl_ranks),
    )

    # ── Phase 3: MATH ─────────────────────────────────────────────────────────
    math_jobs = _build_jobs("math", include_weak=False)
    math_results = _run_phase(
        "math", math_jobs, log,
        max_workers=len(ssl_ranks),
    )

    # ── Collect all trained bundles ───────────────────────────────────────────
    log.info("Collecting trained bundles ...")
    domain_bundles: dict[str, dict[int, dict]] = {}
    weak_bundles: dict[str, dict[int, dict]] = {}

    for dom in ("coding", "science", "math"):
        domain_bundles[dom] = {}
        weak_bundles[dom] = {}
        for r in ssl_ranks:
            b_path = out_dir / f"bundle_{dom}_r{r}.pkl"
            bw_path = out_dir / f"bundle_{dom}_r{r}_weak.pkl"
            if b_path.exists():
                with b_path.open("rb") as fh:
                    domain_bundles[dom][r] = pickle.load(fh)
                log.info(f"  loaded bundle  {dom} r={r}")
            else:
                log.warning(f"  missing bundle {dom} r={r} — condition will be skipped")
            if bw_path.exists():
                with bw_path.open("rb") as fh:
                    weak_bundles[dom][r] = pickle.load(fh)
                log.info(f"  loaded bundle  {dom} r={r} (weak)")
            elif dom != "coding":
                # For non-coding domains, reuse domain_ssl as weak (no pairwise)
                if r in domain_bundles[dom]:
                    weak_bundles[dom][r] = domain_bundles[dom][r]

    # Save combined bundle pkl
    combined_pkl = out_dir / "domain_ssl_bundles.pkl"
    with combined_pkl.open("wb") as fh:
        pickle.dump(
            {"domain_bundles": domain_bundles, "weak_bundles": weak_bundles},
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    log.info(f"Saved combined bundles → {combined_pkl}")

    # ── Load shared SSL bundles (shared_ssl_r16 baseline) ────────────────────
    from scripts.semi_supervised.train_domain_specific_ssl import (
        DEFAULT_SHARED_SSL_PKL,
        LABEL_FRACTIONS,
    )
    shared_ssl_bundles: dict[int, dict] = {}
    shared_pkl = rp(args.shared_ssl_pkl)
    if shared_pkl.exists():
        with shared_pkl.open("rb") as fh:
            shared_ssl_bundles = pickle.load(fh)
        log.info(f"Loaded shared SSL bundles ranks={list(shared_ssl_bundles.keys())}")
    else:
        log.warning(f"shared SSL pkl not found: {shared_pkl}")

    # ── Load supervised bundle (frozen_svd baseline) ─────────────────────────
    supervised_bundle = None
    sup_path = rp(args.supervised_bundle)
    if sup_path.exists():
        try:
            supervised_bundle = load_earlystop_svd_bundle(sup_path)
            log.info(f"Loaded supervised bundle ← {sup_path}")
        except Exception as exc:
            log.warning(f"Could not load supervised bundle: {exc}")
    else:
        log.warning(f"Supervised bundle not found: {sup_path}")

    # ── Study: label-efficiency evaluation ────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("  PHASE: LABEL-EFFICIENCY STUDY (all conditions, all fractions)")
    log.info("=" * 70)

    study_log = _setup_logger(log_dir, name="study")
    t_study = time.perf_counter()

    study_fractions = LABEL_FRACTIONS
    if args.smoke:
        study_fractions = (0.10, 0.50, 1.0)

    results = _run_domain_ssl_study(
        train_by_domain=train_by_domain,
        holdout_by_domain=holdout_by_domain,
        domain_bundles=domain_bundles,
        weak_bundles=weak_bundles,
        shared_ssl_bundles=shared_ssl_bundles,
        supervised_bundle=supervised_bundle,
        label_fractions=study_fractions,
        ssl_ranks=ssl_ranks,
        domains=("coding", "science", "math"),
        seed=int(args.split_seed),
        smoke=bool(args.smoke),
    )
    study_elapsed = time.perf_counter() - t_study
    log.info(f"Study complete: {len(results)} result rows  elapsed={study_elapsed:.1f}s")

    # ── Write outputs ─────────────────────────────────────────────────────────
    out_csv = rp(args.out_csv)
    out_doc = rp(args.out_doc)
    _write_csv(results, out_csv)
    _write_markdown_domain(
        results, out_doc,
        n_epochs=args.n_epochs,
        tau=args.tau,
    )

    # ── Write training summary doc ─────────────────────────────────────────────
    _write_training_summary(
        all_results=coding_results + science_results + math_results,
        results=results,
        log_dir=log_dir,
        out_dir=out_dir,
        ssl_ranks=ssl_ranks,
        n_epochs=args.n_epochs,
        tau=args.tau,
    )

    # ── Cleanup temp files ────────────────────────────────────────────────────
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    total_time = time.perf_counter() - t_study + study_elapsed
    log.info("\n" + "=" * 70)
    log.info(f"  ALL DONE")
    log.info(f"  Total training time: see per-domain logs")
    log.info(f"  CSV      → {out_csv}")
    log.info(f"  Markdown → {out_doc}")
    log.info(f"  Log dir  → {log_dir}")
    log.info("=" * 70)


# ─── Training summary document ────────────────────────────────────────────────

def _write_training_summary(
    all_results: list[dict[str, Any]],
    results: list[dict[str, Any]],
    log_dir: Path,
    out_dir: Path,
    ssl_ranks: tuple[int, ...],
    n_epochs: int,
    tau: float,
) -> None:
    """Write a human-readable training summary markdown."""
    import numpy as np

    ts = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Domain-Specific SSL — Training Summary",
        "",
        f"Generated: {ts}",
        "",
        "## Training Configuration",
        "",
        f"- Epochs: {n_epochs}",
        f"- Temperature τ: {tau}",
        f"- SSL ranks: {list(ssl_ranks)}",
        f"- CPU strategy: process-level parallelism (OMP_NUM_THREADS=1/process)",
        f"  (BLAS threading counterproductive for batch-size 256-384 matrices)",
        "",
        "## Training Run Results",
        "",
        "| Domain | Rank | Variant | Elapsed (s) | Final Loss | Status |",
        "|--------|------|---------|-------------|------------|--------|",
    ]
    for r in sorted(all_results, key=lambda x: (x.get("domain",""), x.get("rank",0), x.get("use_pairwise",False))):
        dom = r.get("domain", "?")
        rank = r.get("rank", "?")
        variant = "weak" if r.get("use_pairwise") else "standard"
        elapsed = f"{r.get('elapsed', 0):.1f}" if r.get("elapsed") else "—"
        loss = f"{r.get('final_loss', float('nan')):.5f}" if r.get("final_loss") is not None else "—"
        status = "ERROR: " + r["error"][:40] if r.get("error") else "OK"
        lines.append(f"| {dom} | {rank} | {variant} | {elapsed} | {loss} | {status} |")

    # Per-domain AUROC summary at 100% labels
    lines += [
        "",
        "## Evaluation Summary (100% labels)",
        "",
        "| Domain | Condition | SSL r | AUROC |",
        "|--------|-----------|:-----:|------:|",
    ]
    rows_100 = [r for r in results if abs(r.get("label_fraction", 0) - 1.0) < 0.01]
    for row in sorted(rows_100, key=lambda r: (r["domain"], r["condition"], r["ssl_rank"])):
        auroc = f"{row.get('auroc', float('nan')):.4f}"
        lines.append(
            f"| {row['domain']} | {row['condition']} | {row['ssl_rank']} | {auroc} |"
        )

    lines += [
        "",
        "## Log Files",
        "",
        f"- Master log: `{log_dir}/run.log`",
        f"- Per-rank logs: `{log_dir}/<domain>_r<rank>[_weak].log`",
        f"- Study log: `{log_dir}/study.log`",
        "",
        "## Output Files",
        "",
        f"- Bundles: `{out_dir}/bundle_<domain>_r<rank>[_weak].pkl`",
        f"- Combined bundles: `{out_dir}/domain_ssl_bundles.pkl`",
    ]

    summary_path = log_dir / "training_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[summary] Training summary → {summary_path}")


if __name__ == "__main__":
    # Ensure spawn context works on all platforms
    mp.set_start_method("spawn", force=True)
    main()
