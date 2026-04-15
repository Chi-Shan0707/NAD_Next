#!/usr/bin/env python3
"""
Frozen-Basis Transfer Suite
============================
3-way ablation testing whether the scaler+SVD from a canonical source bundle
("frozen basis") transfers to downstream tasks with only a new LR head.

Conditions compared per task:
  A. frozen_basis  — project with saved scaler+SVD, train new LR head only
  B. task_specific — fresh scaler+SVD+LR trained on target task data (upper bound)
  C. no_svd        — fresh scaler+LR, no SVD (lower bound)

Target tasks:
  earlystop/{math,science,coding}  — binary correctness, metric=AUROC
  bestofn/math                     — binary correctness, metric=hit@1
  rl_ranking/math                  — checkpoint accuracy ranking, metric=Spearman ρ

Outputs:
  results/tables/frozen_basis_transfer_matrix.csv   (~40 rows)
  results/tables/frozen_basis_transfer_deltas.csv   (~12 summary rows)
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.ops.earlystop import EARLY_STOP_POSITIONS
from nad.ops.earlystop_svd import (
    FULL_FEATURE_NAMES,
    _auroc,
    _build_representation,
    _group_folds,
    _rank_transform_matrix,
    load_earlystop_svd_bundle,
)
from scripts.export_earlystop_svd_submission import _load_or_build_feature_store


OFFICIAL_CHECKPOINTS = (
    "base", "step-100", "step-200", "step-300", "step-400",
    "step-500", "step-600", "step-700", "step-800", "step-900", "step-1000",
)
CHECKPOINT_ORDER = {name: i for i, name in enumerate(OFFICIAL_CHECKPOINTS)}
DEFAULT_RL_CACHE_ROOT = Path("/home/jovyan/public-ro/NAD_RL/math5000RL_neuron_analysis/cache")
DEFAULT_BUNDLE_PATH = "models/ml_selectors/es_svd_ms_rr_r1.pkl"


# ─── projection & data helpers ───────────────────────────────────────────────

def _z_project(
    X: np.ndarray,
    scaler: StandardScaler,
    svd: TruncatedSVD,
    whiten: bool,
) -> np.ndarray:
    """Apply frozen scaler+SVD projection without re-fitting."""
    z = svd.transform(scaler.transform(X))
    if whiten:
        s = np.asarray(svd.singular_values_, dtype=np.float64)
        s = np.where(np.abs(s) < 1e-8, 1.0, s)
        z = z / s
    return z


def _collect_task_data(
    feature_store: list[dict[str, Any]],
    feature_indices: list[int],
    representation: str,
    domain_filter: Optional[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten a feature store into (X, y, groups) for a given domain.

    The feature store tensor has shape [n_samples, n_positions, n_features].
    Since we build with a single position, we index tensor[:, 0, :].
    Returns X in the requested representation, binary labels y, and group_keys.
    """
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []

    for payload in feature_store:
        if domain_filter is not None and str(payload["domain"]) != str(domain_filter):
            continue
        tensor = np.asarray(payload["tensor"], dtype=np.float64)
        labels = np.asarray(payload["labels"], dtype=np.int32)
        group_keys = np.asarray(payload["group_keys"], dtype=object)

        if tensor.shape[0] == 0:
            continue

        x_raw = tensor[:, 0, :]  # [n_samples, n_all_features]
        x_rank = _rank_transform_matrix(x_raw)
        x_rep = _build_representation(x_raw, x_rank, feature_indices, representation)

        X_parts.append(x_rep)
        y_parts.append(labels)
        group_parts.append(group_keys)

    if not X_parts:
        n_feat = len(feature_indices) * (2 if representation == "raw+rank" else 1)
        return (
            np.zeros((0, n_feat), dtype=np.float64),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=object),
        )

    return (
        np.concatenate(X_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        np.concatenate(group_parts, axis=0),
    )


# ─── metric functions ─────────────────────────────────────────────────────────

def _auroc_metric(scores: np.ndarray, y: np.ndarray, _groups: np.ndarray) -> float:
    """AUROC wrapper with uniform (scores, y, groups) signature."""
    return _auroc(scores, y)


def _hit_at_1_metric(scores: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """Hit@1: for each problem group, does the top-scored sample have label=1?"""
    unique_groups = np.unique(groups)
    hits: list[int] = []
    for g in unique_groups:
        mask = groups == g
        g_labels = y[mask]
        if g_labels.size == 0:
            continue
        hits.append(int(g_labels[np.argmax(scores[mask])]))
    return float(np.mean(hits)) if hits else float("nan")


# ─── cross-validation ────────────────────────────────────────────────────────

def _cv_compare(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    frozen_scaler: StandardScaler,
    frozen_svd: TruncatedSVD,
    frozen_whiten: bool,
    rank: int,
    metric_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    n_splits: int,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """
    GroupKFold 3-way comparison: frozen_basis / task_specific / no_svd.

    Returns dict with {mean, std, n_folds} per condition.
    """
    _empty: dict[str, dict[str, float]] = {
        c: {"mean": float("nan"), "std": 0.0, "n_folds": 0}
        for c in ("frozen_basis", "task_specific", "no_svd")
    }
    if X.shape[0] == 0:
        return _empty

    folds = _group_folds(groups, n_splits)
    if not folds:
        return _empty

    vals_fb: list[float] = []
    vals_ts: list[float] = []
    vals_ns: list[float] = []

    for train_idx, test_idx in folds:
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        g_te = groups[test_idx]

        if np.unique(y_tr).shape[0] < 2 or np.unique(y_te).shape[0] < 2:
            continue

        # A: frozen_basis — project with saved scaler+SVD, train new LR head only
        try:
            z_tr = _z_project(X_tr, frozen_scaler, frozen_svd, frozen_whiten)
            z_te = _z_project(X_te, frozen_scaler, frozen_svd, frozen_whiten)
            clf = LogisticRegression(C=1.0, max_iter=2000, random_state=random_state)
            clf.fit(z_tr, y_tr)
            v = metric_fn(clf.decision_function(z_te), y_te, g_te)
            if np.isfinite(v):
                vals_fb.append(float(v))
        except Exception as exc:
            print(f"    [frozen_basis] fold error: {exc}")

        # B: task_specific — fresh scaler+SVD+LR on train fold
        try:
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_tr)
            X_te_sc = sc.transform(X_te)
            max_r = max(1, min(rank, X_tr_sc.shape[1], X_tr_sc.shape[0] - 1))
            svd = TruncatedSVD(n_components=max_r, random_state=random_state)
            z_tr_ts = svd.fit_transform(X_tr_sc)
            z_te_ts = svd.transform(X_te_sc)
            clf = LogisticRegression(C=1.0, max_iter=2000, random_state=random_state)
            clf.fit(z_tr_ts, y_tr)
            v = metric_fn(clf.decision_function(z_te_ts), y_te, g_te)
            if np.isfinite(v):
                vals_ts.append(float(v))
        except Exception as exc:
            print(f"    [task_specific] fold error: {exc}")

        # C: no_svd — fresh scaler+LR, no SVD
        try:
            sc = StandardScaler()
            X_tr_ns = sc.fit_transform(X_tr)
            X_te_ns = sc.transform(X_te)
            clf = LogisticRegression(C=1.0, max_iter=2000, random_state=random_state)
            clf.fit(X_tr_ns, y_tr)
            v = metric_fn(clf.decision_function(X_te_ns), y_te, g_te)
            if np.isfinite(v):
                vals_ns.append(float(v))
        except Exception as exc:
            print(f"    [no_svd] fold error: {exc}")

    def _stats(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"mean": float("nan"), "std": 0.0, "n_folds": 0}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n_folds": len(vals)}

    return {"frozen_basis": _stats(vals_fb), "task_specific": _stats(vals_ts), "no_svd": _stats(vals_ns)}


# ─── RL checkpoint ranking ───────────────────────────────────────────────────

def _official_checkpoint_name(model_dir_name: str) -> str:
    if model_dir_name == "Qwen3-4B-Base_base":
        return "base"
    m = re.fullmatch(r"Qwen3-4B-Base_math7500-step-(\d+)", model_dir_name)
    if m is None:
        raise ValueError(f"Unrecognized RL checkpoint directory: {model_dir_name!r}")
    name = f"step-{int(m.group(1))}"
    if name not in CHECKPOINT_ORDER:
        raise ValueError(f"Unexpected checkpoint step in: {model_dir_name!r}")
    return name


def _rl_loo(
    checkpoint_records: list[dict[str, Any]],
    frozen_scaler: StandardScaler,
    frozen_svd: TruncatedSVD,
    frozen_whiten: bool,
    rank: int,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """
    11-fold leave-one-checkpoint-out Spearman evaluation.

    For each fold: pool samples from 10 train checkpoints, train model, predict
    mean score on the left-out checkpoint's samples.  Then Spearman(pred, true_acc).
    """
    n = len(checkpoint_records)
    _empty = {c: {"mean": float("nan"), "std": 0.0, "n_folds": n}
              for c in ("frozen_basis", "task_specific", "no_svd")}
    if n < 3:
        return _empty

    pred_fb = np.full(n, float("nan"))
    pred_ts = np.full(n, float("nan"))
    pred_ns = np.full(n, float("nan"))
    true_acc = np.asarray([r["true_accuracy"] for r in checkpoint_records], dtype=np.float64)

    for loo_i, test_rec in enumerate(checkpoint_records):
        train_recs = [checkpoint_records[j] for j in range(n) if j != loo_i]
        X_te = test_rec["X"]
        X_tr = np.concatenate([r["X"] for r in train_recs], axis=0)
        y_tr = np.concatenate([r["y"] for r in train_recs], axis=0)

        if np.unique(y_tr).shape[0] < 2:
            continue

        # A: frozen_basis
        try:
            z_tr = _z_project(X_tr, frozen_scaler, frozen_svd, frozen_whiten)
            z_te = _z_project(X_te, frozen_scaler, frozen_svd, frozen_whiten)
            clf = LogisticRegression(C=1.0, max_iter=2000, random_state=random_state)
            clf.fit(z_tr, y_tr)
            pred_fb[loo_i] = float(np.mean(clf.decision_function(z_te)))
        except Exception as exc:
            print(f"    [rl/frozen_basis] LOO-{loo_i} error: {exc}")

        # B: task_specific
        try:
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_tr)
            X_te_sc = sc.transform(X_te)
            max_r = max(1, min(rank, X_tr_sc.shape[1], X_tr_sc.shape[0] - 1))
            svd = TruncatedSVD(n_components=max_r, random_state=random_state)
            z_tr_ts = svd.fit_transform(X_tr_sc)
            z_te_ts = svd.transform(X_te_sc)
            clf = LogisticRegression(C=1.0, max_iter=2000, random_state=random_state)
            clf.fit(z_tr_ts, y_tr)
            pred_ts[loo_i] = float(np.mean(clf.decision_function(z_te_ts)))
        except Exception as exc:
            print(f"    [rl/task_specific] LOO-{loo_i} error: {exc}")

        # C: no_svd
        try:
            sc = StandardScaler()
            X_tr_ns = sc.fit_transform(X_tr)
            X_te_ns = sc.transform(X_te)
            clf = LogisticRegression(C=1.0, max_iter=2000, random_state=random_state)
            clf.fit(X_tr_ns, y_tr)
            pred_ns[loo_i] = float(np.mean(clf.decision_function(X_te_ns)))
        except Exception as exc:
            print(f"    [rl/no_svd] LOO-{loo_i} error: {exc}")

    def _spearman(pred: np.ndarray) -> float:
        mask = np.isfinite(pred) & np.isfinite(true_acc)
        if mask.sum() < 3:
            return float("nan")
        result = spearmanr(pred[mask], true_acc[mask])
        val = getattr(result, "statistic", result[0])
        return float(val) if np.isfinite(float(val)) else float("nan")

    return {
        "frozen_basis":  {"mean": _spearman(pred_fb), "std": 0.0, "n_folds": n},
        "task_specific": {"mean": _spearman(pred_ts), "std": 0.0, "n_folds": n},
        "no_svd":        {"mean": _spearman(pred_ns), "std": 0.0, "n_folds": n},
    }


# ─── CSV writers ─────────────────────────────────────────────────────────────

def _write_matrix_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task", "domain", "anchor_pct", "condition", "metric",
        "mean", "std", "n_folds", "n_samples", "n_groups",
        "source_bundle", "source_route_auroc", "rank",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote matrix CSV ({len(rows)} rows): {path}")


def _write_deltas_csv(matrix_rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    pivot: dict[tuple, dict[str, float]] = defaultdict(dict)
    for row in matrix_rows:
        key = (str(row["task"]), str(row["domain"]), str(row["anchor_pct"]), str(row["metric"]))
        try:
            val = float(row["mean"])
        except (TypeError, ValueError):
            val = float("nan")
        pivot[key][str(row["condition"])] = val

    def _verdict(delta: float) -> str:
        if not np.isfinite(delta):
            return "unknown"
        return "win" if delta >= 0.0 else ("tie" if delta >= -0.02 else "loss")

    def _f(v: float) -> str:
        return f"{v:.4f}" if np.isfinite(v) else "nan"

    fieldnames = [
        "task", "domain", "anchor_pct", "metric",
        "frozen_basis", "task_specific", "no_svd",
        "delta_fb_minus_ts", "delta_fb_minus_nosvd", "transfer_verdict",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(pivot):
            task, domain, anchor_pct, metric = key
            conds = pivot[key]
            fb = conds.get("frozen_basis", float("nan"))
            ts = conds.get("task_specific", float("nan"))
            ns = conds.get("no_svd", float("nan"))
            d_ts = fb - ts if np.isfinite(fb) and np.isfinite(ts) else float("nan")
            d_ns = fb - ns if np.isfinite(fb) and np.isfinite(ns) else float("nan")
            writer.writerow({
                "task": task, "domain": domain, "anchor_pct": anchor_pct, "metric": metric,
                "frozen_basis": _f(fb), "task_specific": _f(ts), "no_svd": _f(ns),
                "delta_fb_minus_ts": _f(d_ts), "delta_fb_minus_nosvd": _f(d_ns),
                "transfer_verdict": _verdict(d_ts),
            })
    print(f"Wrote deltas CSV ({len(pivot)} summary rows): {path}")


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Frozen-Basis Transfer Suite")
    ap.add_argument("--bundle-path", default=DEFAULT_BUNDLE_PATH,
                    help="Source SVD bundle (default: %(default)s)")
    ap.add_argument("--cache-root", default="MUI_HUB/cache",
                    help="Cache root for math+science caches")
    ap.add_argument("--coding-cache-root", default=None,
                    help="Cache root for coding (default: same as --cache-root)")
    ap.add_argument("--rl-cache-root", default=str(DEFAULT_RL_CACHE_ROOT),
                    help="RL checkpoint cache root")
    ap.add_argument("--domains", default="math,science,coding",
                    help="Comma-separated domains for earlystop (math/science/coding)")
    ap.add_argument("--anchor-pct", type=int, default=100,
                    help="Anchor position %% (10 | 40 | 70 | 100)")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--feature-chunk-problems", type=int, default=8)
    ap.add_argument("--feature-cache-dir", default="results/cache/frozen_basis_transfer",
                    help="Feature store cache dir ('none' to disable)")
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--out-dir", default="results/tables")
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    def _resolve(raw: str) -> Path:
        p = Path(raw)
        return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()

    bundle_path = _resolve(str(args.bundle_path))
    cache_root = str(_resolve(str(args.cache_root)))
    coding_cache_root = str(_resolve(str(args.coding_cache_root or args.cache_root)))
    rl_cache_root = _resolve(str(args.rl_cache_root))
    out_dir = _resolve(str(args.out_dir))
    feature_cache_dir: Optional[Path] = None
    if str(args.feature_cache_dir).strip().lower() not in {"", "none", "off"}:
        feature_cache_dir = _resolve(str(args.feature_cache_dir))

    domains_to_run = [d.strip() for d in str(args.domains).split(",") if d.strip()]
    anchor_pct = int(args.anchor_pct)

    print(f"Bundle        : {bundle_path}")
    print(f"Cache root    : {cache_root}")
    print(f"RL cache root : {rl_cache_root}")
    print(f"Domains       : {domains_to_run}")
    print(f"Anchor        : {anchor_pct}%  n_splits={args.n_splits}")

    # Load bundle and extract frozen basis from math route
    bundle = load_earlystop_svd_bundle(bundle_path)
    bundle_name = bundle_path.stem
    print(f"\nBundle '{bundle_name}' — domains: {sorted(bundle['domains'].keys())}")

    anchor_pct_to_slot = {int(round(float(p) * 100)): i for i, p in enumerate(EARLY_STOP_POSITIONS)}
    if anchor_pct not in anchor_pct_to_slot:
        raise ValueError(f"--anchor-pct {anchor_pct} not in {sorted(anchor_pct_to_slot)}")
    slot_index = anchor_pct_to_slot[anchor_pct]
    position_value = float(EARLY_STOP_POSITIONS[slot_index])
    print(f"slot_index={slot_index}, position={position_value:.2f}")

    math_route = bundle["domains"]["math"]["routes"][slot_index]
    if math_route["route_type"] != "svd":
        print(f"ERROR: math route at slot {slot_index} is type='{math_route['route_type']}', not 'svd'.")
        sys.exit(1)

    frozen_model = math_route["model"]
    frozen_scaler: StandardScaler = frozen_model["scaler"]
    frozen_svd: TruncatedSVD = frozen_model["svd"]
    frozen_whiten: bool = bool(frozen_model.get("whiten", False))
    feature_indices: list[int] = [int(i) for i in math_route["feature_indices"]]
    representation: str = str(math_route.get("representation", "raw+rank"))
    rank: int = int(math_route.get("rank", 16))
    source_route_auroc: float = float(math_route.get("cv_auroc", float("nan")))
    required_features: set[str] = {FULL_FEATURE_NAMES[i] for i in feature_indices}
    default_threshold = float(bundle.get("reflection_threshold", 0.30))
    math_threshold = float(math_route.get("reflection_threshold", default_threshold))

    print(f"Frozen basis  : representation={representation}, rank={rank}, whiten={frozen_whiten}")
    print(f"Features      : {len(feature_indices)}, threshold={math_threshold}")
    print(f"Source AUROC  : {source_route_auroc:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Shared kwargs for _load_or_build_feature_store
    fs_kwargs: dict[str, Any] = dict(
        positions=(position_value,),
        required_feature_names=required_features,
        max_problems=None,
        reflection_threshold=math_threshold,
        workers=args.workers,
        feature_chunk_problems=args.feature_chunk_problems,
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=args.refresh_feature_cache,
    )

    # ─── Build feature stores ──────────────────────────────────────────────
    main_feature_store: list[dict[str, Any]] = []
    coding_feature_store: list[dict[str, Any]] = []

    if any(d in domains_to_run for d in ("math", "science")):
        print(f"\nBuilding main feature store ({cache_root})...")
        main_feature_store, _, _ = _load_or_build_feature_store(
            cache_root=cache_root, **fs_kwargs
        )
        print(f"  → {len(main_feature_store)} payloads")

    if "coding" in domains_to_run:
        if coding_cache_root != cache_root:
            print(f"\nBuilding coding feature store ({coding_cache_root})...")
            coding_feature_store, _, _ = _load_or_build_feature_store(
                cache_root=coding_cache_root, **fs_kwargs
            )
            print(f"  → {len(coding_feature_store)} payloads")
        else:
            coding_feature_store = main_feature_store

    domain_to_store: dict[str, list[dict[str, Any]]] = {
        "math": main_feature_store,
        "science": main_feature_store,
        "coding": coding_feature_store,
    }

    # ─── EarlyStop + BestofN ──────────────────────────────────────────────
    matrix_rows: list[dict[str, Any]] = []

    task_specs: list[tuple[str, str, Callable, list[str]]] = [
        ("earlystop", "auroc",    _auroc_metric,    domains_to_run),
        ("bestofn",   "hit_at_1", _hit_at_1_metric, [d for d in domains_to_run if d == "math"]),
    ]

    for task, metric_name, metric_fn, task_domains in task_specs:
        for domain in task_domains:
            feature_store = domain_to_store.get(domain, [])
            if not feature_store:
                print(f"\n[{task}/{domain}] No feature store — skipping.")
                continue

            print(f"\n[{task}/{domain}] Collecting data...")
            X, y, groups = _collect_task_data(
                feature_store=feature_store,
                feature_indices=feature_indices,
                representation=representation,
                domain_filter=domain,
            )
            n_samples, n_groups = int(X.shape[0]), 0
            if n_samples > 0:
                n_groups = int(len(np.unique(groups)))
            print(f"  n_samples={n_samples}, n_groups={n_groups}")

            if n_samples == 0:
                print(f"  No samples found — skipping.")
                continue

            print(f"  Running 3-way CV (n_splits={args.n_splits})...")
            cv_results = _cv_compare(
                X=X, y=y, groups=groups,
                frozen_scaler=frozen_scaler, frozen_svd=frozen_svd,
                frozen_whiten=frozen_whiten, rank=rank,
                metric_fn=metric_fn, n_splits=args.n_splits,
                random_state=args.random_state,
            )

            auroc_ref = float("nan")
            if np.isfinite(source_route_auroc):
                auroc_ref = source_route_auroc

            for condition, stats in cv_results.items():
                mean_str = f"{stats['mean']:.6f}" if np.isfinite(stats["mean"]) else "nan"
                print(
                    f"  [{condition:14s}] {metric_name}={stats['mean']:.4f}"
                    f"±{stats['std']:.4f}  n_folds={stats['n_folds']}"
                )
                matrix_rows.append({
                    "task": task, "domain": domain, "anchor_pct": anchor_pct,
                    "condition": condition, "metric": metric_name,
                    "mean": mean_str, "std": f"{stats['std']:.6f}",
                    "n_folds": stats["n_folds"], "n_samples": n_samples,
                    "n_groups": n_groups, "source_bundle": bundle_name,
                    "source_route_auroc": f"{auroc_ref:.6f}" if np.isfinite(auroc_ref) else "nan",
                    "rank": rank,
                })

    # ─── RL Checkpoint Ranking ────────────────────────────────────────────
    if "math" in domains_to_run:
        if not rl_cache_root.exists():
            print(f"\nWARNING: RL cache root not found ({rl_cache_root}) — skipping RL ranking.")
        else:
            print(f"\nBuilding RL feature store ({rl_cache_root})...")
            rl_feature_store, _, _ = _load_or_build_feature_store(
                cache_root=str(rl_cache_root), **fs_kwargs
            )
            print(f"  → {len(rl_feature_store)} payloads")

            checkpoint_records: list[dict[str, Any]] = []
            for payload in rl_feature_store:
                model_dir = str(payload["cache_key"]).split("/", 1)[0]
                try:
                    ckpt_name = _official_checkpoint_name(model_dir)
                except ValueError as exc:
                    print(f"  WARNING: {exc}")
                    continue
                tensor = np.asarray(payload["tensor"], dtype=np.float64)
                labels = np.asarray(payload["labels"], dtype=np.int32)
                x_raw = tensor[:, 0, :]
                x_rank = _rank_transform_matrix(x_raw)
                x_rep = _build_representation(x_raw, x_rank, feature_indices, representation)
                checkpoint_records.append({
                    "checkpoint_name": ckpt_name, "X": x_rep, "y": labels,
                    "true_accuracy": float(np.mean(labels)) if labels.size > 0 else float("nan"),
                })

            checkpoint_records.sort(
                key=lambda r: CHECKPOINT_ORDER.get(r["checkpoint_name"], 999)
            )
            n_ckpts = len(checkpoint_records)
            n_rl_samples = sum(len(r["y"]) for r in checkpoint_records)
            print(f"  {n_ckpts} checkpoints, {n_rl_samples} total samples")
            print("  Running 11-fold LOO...")

            rl_results = _rl_loo(
                checkpoint_records=checkpoint_records,
                frozen_scaler=frozen_scaler, frozen_svd=frozen_svd,
                frozen_whiten=frozen_whiten, rank=rank,
                random_state=args.random_state,
            )

            auroc_ref = source_route_auroc
            for condition, stats in rl_results.items():
                mean_str = f"{stats['mean']:.6f}" if np.isfinite(stats["mean"]) else "nan"
                print(
                    f"  [rl/{condition:14s}] spearman_rho={stats['mean']:.4f}"
                    f"  n_folds={stats['n_folds']}"
                )
                matrix_rows.append({
                    "task": "rl_ranking", "domain": "math", "anchor_pct": anchor_pct,
                    "condition": condition, "metric": "spearman_rho",
                    "mean": mean_str, "std": f"{stats['std']:.6f}",
                    "n_folds": stats["n_folds"], "n_samples": n_rl_samples,
                    "n_groups": n_ckpts, "source_bundle": bundle_name,
                    "source_route_auroc": f"{auroc_ref:.6f}" if np.isfinite(auroc_ref) else "nan",
                    "rank": rank,
                })

    # ─── Write outputs ────────────────────────────────────────────────────
    matrix_path = out_dir / "frozen_basis_transfer_matrix.csv"
    deltas_path = out_dir / "frozen_basis_transfer_deltas.csv"

    _write_matrix_csv(matrix_rows, matrix_path)
    _write_deltas_csv(matrix_rows, deltas_path)

    print(f"\nDone.")
    print(f"  Matrix  : {matrix_path}")
    print(f"  Deltas  : {deltas_path}")


if __name__ == "__main__":
    main()
