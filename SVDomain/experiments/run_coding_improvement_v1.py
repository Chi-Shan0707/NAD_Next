#!/usr/bin/env python3
"""Phase 2: Coding domain improvement experiments.

Tests three hypotheses for beating the 55.58% XGBoost AUROC ceiling on LCBv5:

  Branch A – Code-structural token-ID features (Tier 1 + XGBoost / LR).
  Branch B – Confidence-curve derivative features (per-slice confidence derivatives).
  Branch C – Pairwise hard-negative contrastive objective (within-group SVM).

All experiments use 5-fold GroupKFold by problem_id (no holdout; 167 problems).
Metrics: AUROC, AUPRC, Brier, logloss.

Usage (from repo root):
    python3 SVDomain/experiments/run_coding_improvement_v1.py
    python3 SVDomain/experiments/run_coding_improvement_v1.py --skip-branch-c
    python3 SVDomain/experiments/run_coding_improvement_v1.py --out-csv results/tables/coding_improvement_v1.csv
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.views.reader import CacheReader
from nad.ops.coding_features import TIER1_FEATURE_NAMES, extract_tier1_feature_matrix
from nad.ops.earlystop_svd import (
    CODING_DERIVATIVE_FEATURES,
    FULL_FEATURE_NAMES,
    LEGACY_FULL_FEATURE_NAMES,
    TOKEN_FEATURES,
    TRAJ_FEATURES,
    _auroc,
    _build_representation,
    _fit_svd_transform,
    _group_folds,
    _rank_transform_matrix,
    extract_earlystop_signals_for_positions,
)

# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_FEATURE_CACHE = "results/cache/es_svd_ms_rr_r1/cache_all_547b9060debe139e.pkl"
DEFAULT_CACHE_ROOT = (
    "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/livecodebench_v5/"
    "cache_neuron_output_1_act_no_rms_20251127_032808"
)
DEFAULT_TIER1_CACHE = "results/cache/coding_dummies_tier1.pkl"
DEFAULT_DERIV_CACHE = "results/cache/coding_improvement_v1_deriv.pkl"
DEFAULT_OUT_CSV = "results/tables/coding_improvement_v1.csv"
DEFAULT_N_SPLITS = 5
DEFAULT_SEEDS = (42, 101, 29)

# ── metric helpers ─────────────────────────────────────────────────────────────

def _auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    if int(labels.sum()) == 0 or int((1 - labels).sum()) == 0:
        return float("nan")
    return float(average_precision_score(labels, scores))


def _brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels.astype(np.float64)) ** 2))


def _logloss(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-7) -> float:
    probs = np.clip(probs, eps, 1.0 - eps)
    y = labels.astype(np.float64)
    return float(-np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs)))


def _scores_to_probs(scores: np.ndarray) -> np.ndarray:
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        return np.full_like(scores, 0.5, dtype=np.float64)
    return (scores - lo) / (hi - lo)


# ── data loading ──────────────────────────────────────────────────────────────

def _load_coding_base(feature_cache: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_raw, labels, groups, sample_ids) from existing 30-feature store."""
    with open(feature_cache, "rb") as fh:
        data = pickle.load(fh)
    fs = data["feature_store"]
    for item in fs:
        if item["domain"] == "coding":
            tensor = item["tensor"]  # (n, n_pos, 30)
            pos_idx = int(len(item["positions"]) - 1)  # last position = 1.0
            x_raw = np.asarray(tensor[:, pos_idx, :], dtype=np.float64)
            labels = np.asarray(item["labels"], dtype=np.int32)
            groups = np.asarray(item["group_keys"], dtype=object)
            sample_ids = np.asarray(item["sample_ids"], dtype=np.int64)
            return x_raw, labels, groups, sample_ids
    raise ValueError("No coding domain found in feature store")


def _load_or_build_tier1(
    cache_root: str,
    sample_ids: np.ndarray,
    cache_path: str,
    refresh: bool = False,
) -> np.ndarray:
    cp = Path(cache_path)
    if not refresh and cp.exists():
        with open(cp, "rb") as fh:
            saved = pickle.load(fh)
        if np.array_equal(saved["sample_ids"], sample_ids):
            print(f"  [cache] Tier-1 loaded from {cp}")
            return np.asarray(saved["x_tier1"], dtype=np.float64)
        print("  [cache] Tier-1 mismatch — rebuilding")
    print(f"  Building Tier-1 features ({len(sample_ids)} samples)…", flush=True)
    reader = CacheReader(cache_root)
    x = extract_tier1_feature_matrix(reader, sample_ids, verbose=True)
    cp.parent.mkdir(parents=True, exist_ok=True)
    with open(cp, "wb") as fh:
        pickle.dump({"sample_ids": sample_ids, "x_tier1": x}, fh, protocol=4)
    print(f"  [cache] Tier-1 saved → {cp}")
    return x


def _load_or_build_deriv(
    cache_root: str,
    sample_ids: np.ndarray,
    cache_path: str,
    refresh: bool = False,
) -> np.ndarray:
    """Extract derivative features (12) for all samples at position 1.0."""
    cp = Path(cache_path)
    if not refresh and cp.exists():
        with open(cp, "rb") as fh:
            saved = pickle.load(fh)
        if np.array_equal(saved["sample_ids"], sample_ids):
            print(f"  [cache] Derivative features loaded from {cp}")
            return np.asarray(saved["x_deriv"], dtype=np.float64)
        print("  [cache] Derivative features mismatch — rebuilding")

    print(f"  Building derivative features ({len(sample_ids)} samples)…", flush=True)
    reader = CacheReader(cache_root)
    req = set(CODING_DERIVATIVE_FEATURES)
    full_idx = {name: i for i, name in enumerate(FULL_FEATURE_NAMES)}
    deriv_indices = [full_idx[name] for name in CODING_DERIVATIVE_FEATURES]

    x = np.zeros((len(sample_ids), len(CODING_DERIVATIVE_FEATURES)), dtype=np.float64)
    for i, sid in enumerate(sample_ids):
        sig = extract_earlystop_signals_for_positions(
            reader, int(sid), positions=(1.0,), required_features=req
        )
        for j, name in enumerate(CODING_DERIVATIVE_FEATURES):
            x[i, j] = float(sig[name][0])
        if i > 0 and i % 2000 == 0:
            print(f"    {i}/{len(sample_ids)}", flush=True)

    cp.parent.mkdir(parents=True, exist_ok=True)
    with open(cp, "wb") as fh:
        pickle.dump({"sample_ids": sample_ids, "x_deriv": x}, fh, protocol=4)
    print(f"  [cache] Derivative features saved → {cp}")
    return x


# ── pairwise helpers ──────────────────────────────────────────────────────────

def _build_pairwise_examples(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build within-group (correct - incorrect) pairwise diffs and mirrored negatives."""
    pos_rows, neg_rows = [], []
    for g in np.unique(groups):
        mask = groups == g
        idx = np.where(mask)[0]
        y_g = y[idx]
        pos_idx = idx[y_g == 1]
        neg_idx = idx[y_g == 0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        for pi in pos_idx:
            for ni in neg_idx:
                pos_rows.append(x[pi] - x[ni])
                neg_rows.append(x[ni] - x[pi])
    if not pos_rows:
        return np.zeros((0, x.shape[1]), dtype=np.float64), np.zeros(0, dtype=np.int32)
    x_pairs = np.vstack(pos_rows + neg_rows).astype(np.float64)
    y_pairs = np.concatenate([
        np.ones(len(pos_rows), dtype=np.int32),
        np.zeros(len(neg_rows), dtype=np.int32),
    ])
    return x_pairs, y_pairs


def _pairwise_score_from_lr(
    clf: Any,
    scaler: StandardScaler,
    x: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Convert pairwise model to pointwise scores via within-group averaging."""
    x_scaled = scaler.transform(x)
    scores = np.zeros(len(x), dtype=np.float64)
    for g in np.unique(groups):
        mask = groups == g
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        x_g = x_scaled[idx]
        s_g = np.zeros(len(idx), dtype=np.float64)
        for j in range(len(idx)):
            diffs = x_g[j:j+1] - x_g  # (n, d)
            s_g[j] = float(np.mean(clf.decision_function(diffs)))
        scores[idx] = s_g
    return scores


# ── CV evaluation ─────────────────────────────────────────────────────────────

def _cv_xgb(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seeds: tuple[int, ...],
) -> dict[str, float]:
    """5-fold GroupKFold CV with XGBoost; average over multiple seeds."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return {"auroc": float("nan"), "auprc": float("nan"), "brier": float("nan"), "logloss": float("nan")}

    folds = _group_folds(groups, n_splits=n_splits)
    if not folds:
        return {"auroc": float("nan"), "auprc": float("nan"), "brier": float("nan"), "logloss": float("nan")}

    aurocs, auprcs, briers, loglosses = [], [], [], []
    for train_idx, test_idx in folds:
        y_tr, y_te = y[train_idx], y[test_idx]
        x_tr, x_te = x[train_idx], x[test_idx]
        if np.unique(y_tr).shape[0] < 2 or np.unique(y_te).shape[0] < 2:
            continue
        fold_probs = np.zeros(len(y_te), dtype=np.float64)
        for seed in seeds:
            clf = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                max_depth=5,
                learning_rate=0.05,
                n_estimators=300,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                n_jobs=4,
                verbosity=0,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(x_tr, y_tr)
            fold_probs += clf.predict_proba(x_te)[:, 1]
        fold_probs /= len(seeds)
        aurocs.append(_auroc(fold_probs, y_te))
        auprcs.append(_auprc(fold_probs, y_te))
        briers.append(_brier(fold_probs, y_te))
        loglosses.append(_logloss(fold_probs, y_te))

    def _m(vals: list) -> float:
        arr = np.array([v for v in vals if np.isfinite(v)])
        return float(np.mean(arr)) if arr.size > 0 else float("nan")

    return {"auroc": _m(aurocs), "auprc": _m(auprcs), "brier": _m(briers), "logloss": _m(loglosses)}


def _cv_lr_svd(
    x_raw: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    rank: int = 8,
    c_value: float = 1.0,
    representation: str = "raw+rank",
) -> dict[str, float]:
    """5-fold GroupKFold CV with SVD-LR."""
    folds = _group_folds(groups, n_splits=n_splits)
    if not folds:
        return {"auroc": float("nan"), "auprc": float("nan"), "brier": float("nan"), "logloss": float("nan")}

    feat_idx = list(range(x_raw.shape[1]))
    aurocs, auprcs, briers, loglosses = [], [], [], []

    for train_idx, test_idx in folds:
        y_tr, y_te = y[train_idx], y[test_idx]
        if np.unique(y_tr).shape[0] < 2 or np.unique(y_te).shape[0] < 2:
            continue

        x_rank = _rank_transform_matrix(x_raw)
        x_tr = _build_representation(x_raw[train_idx], x_rank[train_idx], feat_idx, representation)
        x_te = _build_representation(x_raw[test_idx], x_rank[test_idx], feat_idx, representation)

        transform = _fit_svd_transform(x_tr, rank=rank, whiten=False, random_state=42)
        if transform is None:
            continue
        scaler, svd = transform["scaler"], transform["svd"]
        z_tr = svd.transform(scaler.transform(x_tr))
        z_te = svd.transform(scaler.transform(x_te))

        clf = LogisticRegression(C=c_value, max_iter=2000, random_state=42)
        clf.fit(z_tr, y_tr)
        probs_te = clf.predict_proba(z_te)[:, 1]

        aurocs.append(_auroc(probs_te, y_te))
        auprcs.append(_auprc(probs_te, y_te))
        briers.append(_brier(probs_te, y_te))
        loglosses.append(_logloss(probs_te, y_te))

    def _m(vals: list) -> float:
        arr = np.array([v for v in vals if np.isfinite(v)])
        return float(np.mean(arr)) if arr.size > 0 else float("nan")

    return {"auroc": _m(aurocs), "auprc": _m(auprcs), "brier": _m(briers), "logloss": _m(loglosses)}


def _cv_pairwise_svm(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    c_value: float = 1.0,
    max_pairs_per_fold: int = 20000,
) -> dict[str, float]:
    """5-fold GroupKFold CV with within-group pairwise SVM; pointwise AUROC."""
    from sklearn.svm import LinearSVC

    folds = _group_folds(groups, n_splits=n_splits)
    if not folds:
        return {"auroc": float("nan"), "auprc": float("nan"), "brier": float("nan"), "logloss": float("nan")}

    aurocs, auprcs, briers, loglosses = [], [], [], []

    for train_idx, test_idx in folds:
        y_tr, y_te = y[train_idx], y[test_idx]
        grp_tr = groups[train_idx]
        if np.unique(y_tr).shape[0] < 2 or np.unique(y_te).shape[0] < 2:
            continue

        x_tr_raw, x_te_raw = x[train_idx], x[test_idx]
        scaler = StandardScaler().fit(x_tr_raw)
        x_tr, x_te = scaler.transform(x_tr_raw), scaler.transform(x_te_raw)

        x_pairs, y_pairs = _build_pairwise_examples(x_tr, y_tr, grp_tr)
        if x_pairs.shape[0] == 0:
            continue
        # Subsample if too many pairs
        if x_pairs.shape[0] > max_pairs_per_fold:
            rng = np.random.default_rng(42)
            idx_sub = rng.choice(x_pairs.shape[0], max_pairs_per_fold, replace=False)
            x_pairs, y_pairs = x_pairs[idx_sub], y_pairs[idx_sub]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            svm = LinearSVC(C=c_value, max_iter=4000, random_state=42)
            svm.fit(x_pairs, y_pairs)

        # Pointwise score: decision_function(x_i - group_mean)
        grp_te = groups[test_idx]
        scores_te = np.zeros(len(y_te), dtype=np.float64)
        for g in np.unique(grp_te):
            mask = grp_te == g
            if not mask.any():
                continue
            x_g = x_te[mask]
            g_center = x_te.mean(axis=0)  # global mean as reference
            diffs = x_g - g_center
            scores_te[mask] = svm.decision_function(diffs)

        probs_te = _scores_to_probs(scores_te)
        aurocs.append(_auroc(scores_te, y_te))
        auprcs.append(_auprc(scores_te, y_te))
        briers.append(_brier(probs_te, y_te))
        loglosses.append(_logloss(probs_te, y_te))

    def _m(vals: list) -> float:
        arr = np.array([v for v in vals if np.isfinite(v)])
        return float(np.mean(arr)) if arr.size > 0 else float("nan")

    return {"auroc": _m(aurocs), "auprc": _m(auprcs), "brier": _m(briers), "logloss": _m(loglosses)}


# ── within-group relative transforms ──────────────────────────────────────────

def _within_group_zscore(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Standardise each feature within its problem group."""
    out = np.zeros_like(x, dtype=np.float64)
    for g in np.unique(groups):
        mask = groups == g
        sub = x[mask]
        mu = np.mean(sub, axis=0)
        sigma = np.std(sub, axis=0)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        out[mask] = (sub - mu) / sigma
    return out


def _within_group_rank(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Rank-transform each feature within its problem group."""
    out = np.zeros_like(x, dtype=np.float64)
    for g in np.unique(groups):
        mask = groups == g
        sub = x[mask]
        n = sub.shape[0]
        if n <= 1:
            continue
        for col in range(sub.shape[1]):
            order = np.argsort(sub[:, col], kind="mergesort")
            ranks = np.empty(n, dtype=np.float64)
            ranks[order] = np.arange(n, dtype=np.float64) / float(n - 1)
            out[mask, col] = ranks
    return out


# ── experiment runner ─────────────────────────────────────────────────────────

def _print_row(label: str, m: dict[str, float]) -> None:
    auroc = m.get("auroc", float("nan"))
    auprc = m.get("auprc", float("nan"))
    brier = m.get("brier", float("nan"))
    logloss = m.get("logloss", float("nan"))
    flag = " ← BEST" if auroc > 0.56 else ""
    print(
        f"  {label:<45} AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
        f"Brier={brier:.4f}  LL={logloss:.4f}{flag}"
    )


def run_coding_improvement(
    feature_cache: str,
    cache_root: str,
    tier1_cache_path: str,
    deriv_cache_path: str,
    n_splits: int = DEFAULT_N_SPLITS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    out_csv: Optional[str] = None,
    skip_branch_c: bool = False,
    refresh_caches: bool = False,
) -> list[dict]:
    print("=" * 70)
    print("Coding improvement experiment v1")
    print(f"  5-fold GroupKFold  |  target: >56% AUROC (current best: 55.58%)")
    print("=" * 70)

    # ── Load data ───────────────────────────────────────────────────────────
    print("\n[1] Loading feature stores…")
    x_base, labels, groups, sample_ids = _load_coding_base(feature_cache)
    print(f"    Base features: {x_base.shape}  (LEGACY_FULL_FEATURE_NAMES × 30)")
    print(f"    n_samples={len(labels)}  n_problems={len(np.unique(groups))}  pos_rate={labels.mean():.3f}")

    x_tier1 = _load_or_build_tier1(cache_root, sample_ids, tier1_cache_path, refresh=refresh_caches)
    print(f"    Tier-1 features: {x_tier1.shape}  ({TIER1_FEATURE_NAMES})")

    x_deriv = _load_or_build_deriv(cache_root, sample_ids, deriv_cache_path, refresh=refresh_caches)
    print(f"    Derivative features: {x_deriv.shape}  ({CODING_DERIVATIVE_FEATURES})")

    # ── Within-group transforms ─────────────────────────────────────────────
    print("\n[2] Computing within-group relative transforms…")
    x_base_z = _within_group_zscore(x_base, groups)
    x_tier1_z = _within_group_zscore(x_tier1, groups)
    x_base_wrank = _within_group_rank(x_base, groups)

    # ── Combined feature matrices ────────────────────────────────────────────
    x_base_tier1 = np.concatenate([x_base, x_tier1], axis=1)
    x_base_tier1_z = np.concatenate([x_base_z, x_tier1_z], axis=1)
    x_base_deriv = np.concatenate([x_base, x_deriv], axis=1)
    x_all = np.concatenate([x_base, x_tier1, x_deriv], axis=1)

    rows: list[dict] = []

    def _record(label: str, branch: str, model: str, features: str, m: dict) -> None:
        _print_row(label, m)
        rows.append({
            "label": label,
            "branch": branch,
            "model": model,
            "features": features,
            "auroc": m.get("auroc", float("nan")),
            "auprc": m.get("auprc", float("nan")),
            "brier": m.get("brier", float("nan")),
            "logloss": m.get("logloss", float("nan")),
        })

    # ── Reference ────────────────────────────────────────────────────────────
    print("\n── Reference baselines ──────────────────────────────────────────────")
    # tok_conf_prefix
    tc_idx = LEGACY_FULL_FEATURE_NAMES.index("tok_conf_prefix")
    tok_conf_scores = x_base[:, tc_idx]
    m_ref_conf = _cv_metrics_single(tok_conf_scores, labels, groups, n_splits)
    _record("tok_conf_prefix (rule)", "ref", "rule", "tok_conf_prefix", m_ref_conf)

    # XGBoost on base 30 features (reference, matches paper 55.58%)
    print("  [XGB base 30-feat]…", flush=True)
    m_ref_xgb = _cv_xgb(x_base, labels, groups, n_splits, seeds)
    _record("XGBoost base-30 (reference ~55.58%)", "ref", "xgb", "base30", m_ref_xgb)

    # ── Branch A: Tier-1 structural features ─────────────────────────────────
    print("\n── Branch A: Code-structural Tier-1 features ────────────────────────")

    print("  [XGB tier1 only]…", flush=True)
    m_a1 = _cv_xgb(x_tier1, labels, groups, n_splits, seeds)
    _record("XGBoost Tier-1 only", "A", "xgb", "tier1_only", m_a1)

    print("  [XGB base+tier1]…", flush=True)
    m_a2 = _cv_xgb(x_base_tier1, labels, groups, n_splits, seeds)
    _record("XGBoost base-30 + Tier-1", "A", "xgb", "base30+tier1", m_a2)

    print("  [XGB base+tier1 within-group-z]…", flush=True)
    m_a3 = _cv_xgb(x_base_tier1_z, labels, groups, n_splits, seeds)
    _record("XGBoost base-30 + Tier-1 (within-group-z)", "A", "xgb", "base30+tier1_wz", m_a3)

    print("  [LR-SVD tier1 only]…", flush=True)
    m_a4 = _cv_lr_svd(x_tier1, labels, groups, n_splits, rank=6)
    _record("SVD-LR Tier-1 only", "A", "lr_svd", "tier1_only", m_a4)

    print("  [LR-SVD base+tier1]…", flush=True)
    m_a5 = _cv_lr_svd(x_base_tier1, labels, groups, n_splits, rank=8)
    _record("SVD-LR base-30 + Tier-1", "A", "lr_svd", "base30+tier1", m_a5)

    # Within-group relative rank (1C hypothesis: relative > absolute for coding)
    print("  [XGB within-group-rank base-30]…", flush=True)
    m_a6 = _cv_xgb(x_base_wrank, labels, groups, n_splits, seeds)
    _record("XGBoost base-30 within-group-rank", "A", "xgb", "base30_wrank", m_a6)

    print("  [XGB within-group-rank base+tier1]…", flush=True)
    x_base_tier1_wrank = np.concatenate([
        _within_group_rank(x_base_tier1, groups)
    ], axis=1)
    m_a7 = _cv_xgb(x_base_tier1_wrank, labels, groups, n_splits, seeds)
    _record("XGBoost base-30+Tier-1 within-group-rank", "A", "xgb", "base30+tier1_wrank", m_a7)

    # ── Branch B: Derivative features ─────────────────────────────────────────
    print("\n── Branch B: Confidence-curve derivative features ────────────────────")

    print("  [XGB deriv only]…", flush=True)
    m_b1 = _cv_xgb(x_deriv, labels, groups, n_splits, seeds)
    _record("XGBoost derivative only", "B", "xgb", "deriv_only", m_b1)

    print("  [XGB base+deriv]…", flush=True)
    m_b2 = _cv_xgb(x_base_deriv, labels, groups, n_splits, seeds)
    _record("XGBoost base-30 + derivatives", "B", "xgb", "base30+deriv", m_b2)

    print("  [XGB all (base+tier1+deriv)]…", flush=True)
    m_b3 = _cv_xgb(x_all, labels, groups, n_splits, seeds)
    _record("XGBoost all features (base+tier1+deriv)", "B", "xgb", "all50", m_b3)

    print("  [LR-SVD base+deriv]…", flush=True)
    m_b4 = _cv_lr_svd(x_base_deriv, labels, groups, n_splits, rank=8)
    _record("SVD-LR base-30 + derivatives", "B", "lr_svd", "base30+deriv", m_b4)

    # ── Branch C: Pairwise SVM ────────────────────────────────────────────────
    if not skip_branch_c:
        print("\n── Branch C: Pairwise hard-negative contrastive SVM ─────────────────")

        print("  [Pairwise SVM tier1 only]…", flush=True)
        m_c1 = _cv_pairwise_svm(x_tier1, labels, groups, n_splits)
        _record("Pairwise SVM Tier-1 only", "C", "pairwise_svm", "tier1_only", m_c1)

        print("  [Pairwise SVM base+tier1]…", flush=True)
        m_c2 = _cv_pairwise_svm(x_base_tier1, labels, groups, n_splits)
        _record("Pairwise SVM base-30 + Tier-1", "C", "pairwise_svm", "base30+tier1", m_c2)

        print("  [Pairwise SVM all features]…", flush=True)
        m_c3 = _cv_pairwise_svm(x_all, labels, groups, n_splits)
        _record("Pairwise SVM all features", "C", "pairwise_svm", "all50", m_c3)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Summary (sorted by AUROC desc) ───────────────────────────────────")
    sorted_rows = sorted(rows, key=lambda r: r["auroc"] if np.isfinite(r["auroc"]) else -999.0, reverse=True)
    for row in sorted_rows[:10]:
        _print_row(row["label"], row)

    # Leakage check
    from scipy.stats import pearsonr
    length_col = TIER1_FEATURE_NAMES.index("response_token_count")
    corr, pval = pearsonr(x_tier1[:, length_col], labels.astype(np.float64))
    print(f"\n[leakage] Pearson(response_token_count, label) = {corr:.4f}  p={pval:.3g}")
    print(f"  → {'SAFE (|r|<0.15)' if abs(corr) < 0.15 else 'WARNING: length shortcut!'}")

    if out_csv:
        out_p = Path(out_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[saved] {out_p}")

    return rows


def _cv_metrics_single(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> dict[str, float]:
    folds = _group_folds(groups, n_splits=n_splits)
    if not folds:
        return {"auroc": float("nan"), "auprc": float("nan"), "brier": float("nan"), "logloss": float("nan")}
    aurocs, auprcs, briers, loglosses = [], [], [], []
    for _, test_idx in folds:
        y_te = labels[test_idx]
        s_te = scores[test_idx]
        if np.unique(y_te).shape[0] < 2:
            continue
        probs = _scores_to_probs(s_te)
        aurocs.append(_auroc(s_te, y_te))
        auprcs.append(_auprc(s_te, y_te))
        briers.append(_brier(probs, y_te))
        loglosses.append(_logloss(probs, y_te))

    def _m(vals: list) -> float:
        arr = np.array([v for v in vals if np.isfinite(v)])
        return float(np.mean(arr)) if arr.size > 0 else float("nan")

    return {"auroc": _m(aurocs), "auprc": _m(auprcs), "brier": _m(briers), "logloss": _m(loglosses)}


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feature-cache", default=DEFAULT_FEATURE_CACHE)
    p.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    p.add_argument("--tier1-cache", default=DEFAULT_TIER1_CACHE)
    p.add_argument("--deriv-cache", default=DEFAULT_DERIV_CACHE)
    p.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    p.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    p.add_argument("--skip-branch-c", action="store_true", help="Skip pairwise SVM (slow)")
    p.add_argument("--refresh-caches", action="store_true", help="Rebuild all disk caches")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_coding_improvement(
        feature_cache=str(REPO_ROOT / args.feature_cache),
        cache_root=str(REPO_ROOT / args.cache_root),
        tier1_cache_path=str(REPO_ROOT / args.tier1_cache),
        deriv_cache_path=str(REPO_ROOT / args.deriv_cache),
        n_splits=int(args.n_splits),
        seeds=DEFAULT_SEEDS,
        out_csv=str(REPO_ROOT / args.out_csv) if args.out_csv else None,
        skip_branch_c=bool(args.skip_branch_c),
        refresh_caches=bool(args.refresh_caches),
    )
