#!/usr/bin/env python3
"""Phase 1A: Single-feature rule baselines for the coding domain (LCBv5).

Checks whether any single LM statistic predicts code correctness.
Loads the existing feature store (30-feature, all 10688 samples) to avoid
recomputation, and supplements with response_token_count from CacheReader.

Evaluation: 5-fold GroupKFold by problem_id.
Metrics: AUROC, AUPRC, Brier, logloss.

Usage (from repo root):
    python3 scripts/baselines/train_coding_dummies.py
    python3 scripts/baselines/train_coding_dummies.py --refresh-cache
    python3 scripts/baselines/train_coding_dummies.py --out-csv results/tables/coding_dummies.csv
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.views.reader import CacheReader
from nad.ops.coding_features import (
    TIER1_FEATURE_NAMES,
    extract_tier1_feature_matrix,
)
from nad.ops.earlystop_svd import LEGACY_FULL_FEATURE_NAMES, _auroc, _group_folds

# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_FEATURE_CACHE = "results/cache/es_svd_ms_rr_r1/cache_all_547b9060debe139e.pkl"
DEFAULT_CACHE_ROOT = "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/livecodebench_v5/cache_neuron_output_1_act_no_rms_20251127_032808"
DEFAULT_TIER1_CACHE = "results/cache/coding_dummies_tier1.pkl"
DEFAULT_OUT_CSV = "results/tables/coding_dummies.csv"
DEFAULT_N_SPLITS = 5
POSITION_INDEX = 11  # index for position 1.0 in 12-position schema

# ── metric helpers ────────────────────────────────────────────────────────────

def _auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under precision-recall curve (trapezoidal)."""
    from sklearn.metrics import average_precision_score
    if int(labels.sum()) == 0 or int((1 - labels).sum()) == 0:
        return float("nan")
    return float(average_precision_score(labels, scores))


def _brier(scores: np.ndarray, labels: np.ndarray) -> float:
    """Brier score (mean squared error vs binary labels).
    Scores are raw — sigmoid-squash to [0,1] first using min-max normalisation.
    """
    if len(scores) == 0:
        return float("nan")
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        probs = np.full_like(scores, 0.5, dtype=np.float64)
    else:
        probs = (scores - lo) / (hi - lo)
    return float(np.mean((probs - labels.astype(np.float64)) ** 2))


def _logloss(scores: np.ndarray, labels: np.ndarray, eps: float = 1e-7) -> float:
    """Log-loss after min-max normalisation of raw scores to probabilities."""
    if len(scores) == 0:
        return float("nan")
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        probs = np.full_like(scores, 0.5, dtype=np.float64)
    else:
        probs = (scores - lo) / (hi - lo)
    probs = np.clip(probs, eps, 1.0 - eps)
    y = labels.astype(np.float64)
    return float(-np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs)))


def _cv_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> dict[str, float]:
    folds = _group_folds(groups, n_splits=n_splits)
    if not folds:
        return {"auroc": float("nan"), "auprc": float("nan"), "brier": float("nan"), "logloss": float("nan"), "n_folds": 0}
    aurocs, auprcs, briers, loglosses = [], [], [], []
    for _, test_idx in folds:
        y_test = labels[test_idx]
        s_test = scores[test_idx]
        if np.unique(y_test).shape[0] < 2:
            continue
        aurocs.append(_auroc(s_test, y_test))
        auprcs.append(_auprc(s_test, y_test))
        briers.append(_brier(s_test, y_test))
        loglosses.append(_logloss(s_test, y_test))

    def _safe_mean(vals: list[float]) -> float:
        arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
        return float(np.mean(arr)) if arr.size > 0 else float("nan")

    return {
        "auroc": _safe_mean(aurocs),
        "auprc": _safe_mean(auprcs),
        "brier": _safe_mean(briers),
        "logloss": _safe_mean(loglosses),
        "n_folds": len(aurocs),
    }


# ── data loading ──────────────────────────────────────────────────────────────

def _load_coding_data(feature_cache: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_raw, labels, groups, sample_ids) from existing feature store.

    x_raw: (n_samples, 30) features at position=1.0.
    """
    with open(feature_cache, "rb") as fh:
        data = pickle.load(fh)
    fs = data["feature_store"]
    for item in fs:
        if item["domain"] == "coding":
            tensor = item["tensor"]  # (n, 12, 30)
            # Position index 11 = 1.0
            pos_idx = int(len(item["positions"]) - 1)  # last position
            x_raw = np.asarray(tensor[:, pos_idx, :], dtype=np.float64)
            labels = np.asarray(item["labels"], dtype=np.int32)
            groups = np.asarray(item["group_keys"], dtype=object)
            sample_ids = np.asarray(item["sample_ids"], dtype=np.int64)
            return x_raw, labels, groups, sample_ids
    raise ValueError("No coding domain found in feature store")


def _load_or_build_tier1(
    cache_root: str,
    sample_ids: np.ndarray,
    tier1_cache_path: str,
    refresh: bool = False,
) -> np.ndarray:
    """Load Tier-1 features from disk cache or build from CacheReader."""
    cache_p = Path(tier1_cache_path)
    if not refresh and cache_p.exists():
        print(f"[dummies] Loading Tier-1 cache: {cache_p}")
        with open(cache_p, "rb") as fh:
            saved = pickle.load(fh)
        if np.array_equal(saved["sample_ids"], sample_ids):
            return np.asarray(saved["x_tier1"], dtype=np.float64)
        print("[dummies] Tier-1 cache sample_ids mismatch — rebuilding")

    print(f"[dummies] Building Tier-1 features from CacheReader ({len(sample_ids)} samples)...")
    reader = CacheReader(cache_root)
    x_tier1 = extract_tier1_feature_matrix(reader, sample_ids, verbose=True)
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_p, "wb") as fh:
        pickle.dump({"sample_ids": sample_ids, "x_tier1": x_tier1}, fh, protocol=4)
    print(f"[dummies] Saved Tier-1 cache: {cache_p}")
    return x_tier1


# ── main experiment ───────────────────────────────────────────────────────────

def run_coding_dummies(
    feature_cache: str,
    cache_root: str,
    tier1_cache_path: str,
    n_splits: int = DEFAULT_N_SPLITS,
    out_csv: Optional[str] = None,
    refresh_tier1: bool = False,
) -> list[dict]:
    print("[dummies] Loading existing feature store…")
    x_raw, labels, groups, sample_ids = _load_coding_data(feature_cache)
    print(f"[dummies] n_samples={len(labels)}, n_problems={len(np.unique(groups))}, "
          f"pos_rate={labels.mean():.3f}")

    print("[dummies] Loading Tier-1 structural features…")
    x_tier1 = _load_or_build_tier1(
        cache_root=cache_root,
        sample_ids=sample_ids,
        tier1_cache_path=tier1_cache_path,
        refresh=refresh_tier1,
    )

    # Map feature names to arrays
    feat_name_to_idx = {name: i for i, name in enumerate(LEGACY_FULL_FEATURE_NAMES)}
    tier1_name_to_idx = {name: i for i, name in enumerate(TIER1_FEATURE_NAMES)}

    # (signal_name, score_vector, direction_note)
    candidates: list[tuple[str, np.ndarray, str]] = [
        # Single LM stats at position 1.0
        ("tok_conf_prefix", x_raw[:, feat_name_to_idx["tok_conf_prefix"]], "higher=more confident"),
        ("tok_logprob_prefix", x_raw[:, feat_name_to_idx["tok_logprob_prefix"]], "higher=higher prob"),
        ("tok_neg_entropy_prefix", x_raw[:, feat_name_to_idx["tok_neg_entropy_prefix"]], "neg means certain"),
        ("tok_gini_prefix", x_raw[:, feat_name_to_idx["tok_gini_prefix"]], "lower=more concentrated"),
        ("tok_selfcert_prefix", x_raw[:, feat_name_to_idx["tok_selfcert_prefix"]], "higher=more certain"),
        # Tail versions
        ("tok_gini_tail", x_raw[:, feat_name_to_idx["tok_gini_tail"]], "tail=last 10%"),
        ("tok_gini_slope", x_raw[:, feat_name_to_idx["tok_gini_slope"]], "slope: head minus tail"),
        ("head_tail_gap", x_raw[:, feat_name_to_idx["head_tail_gap"]], "conf head minus tail"),
        ("tail_variance", x_raw[:, feat_name_to_idx["tail_variance"]], "conf variance in tail"),
        # Trajectory
        ("traj_continuity", x_raw[:, feat_name_to_idx["traj_continuity"]], "higher=more consistent"),
        ("traj_reflection_count", x_raw[:, feat_name_to_idx["traj_reflection_count"]], "neg of count"),
        # Tier-1 structural
        ("response_token_count", x_tier1[:, tier1_name_to_idx["response_token_count"]], "total tokens"),
        ("at_max_tokens", x_tier1[:, tier1_name_to_idx["at_max_tokens"]], "binary: hit limit"),
        ("trigram_repetition_rate", -x_tier1[:, tier1_name_to_idx["trigram_repetition_rate"]], "lower=less repetitive"),
        ("unique_token_fraction", x_tier1[:, tier1_name_to_idx["unique_token_fraction"]], "higher=less repetitive"),
        ("tok_conf_early_vs_late_gap", x_tier1[:, tier1_name_to_idx["tok_conf_early_vs_late_gap"]], "early minus late conf"),
        ("tok_conf_section_late", x_tier1[:, tier1_name_to_idx["tok_conf_section_late"]], "late third conf mean"),
    ]

    rows: list[dict] = []
    print()
    print(f"{'Signal':<35} {'AUROC':>7} {'AUPRC':>7} {'Brier':>7} {'Logloss':>8}")
    print("-" * 70)

    for signal_name, scores, note in candidates:
        m = _cv_metrics(scores, labels, groups, n_splits)
        row = {
            "signal": signal_name,
            "note": note,
            "auroc": m["auroc"],
            "auprc": m["auprc"],
            "brier": m["brier"],
            "logloss": m["logloss"],
            "n_folds": m["n_folds"],
        }
        rows.append(row)
        print(
            f"{signal_name:<35} "
            f"{m['auroc']:>7.4f} "
            f"{m['auprc']:>7.4f} "
            f"{m['brier']:>7.4f} "
            f"{m['logloss']:>8.4f}"
        )

    # Leakage check: Pearson(length, label)
    lengths = x_tier1[:, tier1_name_to_idx["response_token_count"]]
    from scipy.stats import pearsonr
    corr, pval = pearsonr(lengths, labels.astype(np.float64))
    print()
    print(f"[leakage check] Pearson(length, label) = {corr:.4f}  (p={pval:.3g})")
    print(f"  → {'SAFE' if abs(corr) < 0.15 else 'WARNING: length shortcut detected'}")

    if out_csv:
        out_p = Path(out_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[dummies] Saved → {out_p}")

    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feature-cache", default=DEFAULT_FEATURE_CACHE)
    p.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    p.add_argument("--tier1-cache", default=DEFAULT_TIER1_CACHE)
    p.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    p.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    p.add_argument("--refresh-cache", action="store_true", help="Rebuild Tier-1 feature cache")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_coding_dummies(
        feature_cache=str(REPO_ROOT / args.feature_cache),
        cache_root=str(REPO_ROOT / args.cache_root),
        tier1_cache_path=str(REPO_ROOT / args.tier1_cache),
        n_splits=int(args.n_splits),
        out_csv=str(REPO_ROOT / args.out_csv) if args.out_csv else None,
        refresh_tier1=bool(args.refresh_cache),
    )
