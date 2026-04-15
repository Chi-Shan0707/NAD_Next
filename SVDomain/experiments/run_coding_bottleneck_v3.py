#!/usr/bin/env python3
"""Coding Bottleneck Diagnosis V3 — separates feature poverty, objective mismatch,
structural constraints, and fundamental irreducibility.

Conditions (all at late anchors 70% and 100%):
  response_length_only  — output_tokens proxy (finished_reason_length feature or tok_conf_prefix rank)
  token_only_logistic   — 11 token features + 5 availability flags, LR
  fixed_22_logistic     — token_plus_traj 22 features, LR (current standard)
  full_logistic         — all features in tensor, LR
  fixed_22_pairwise     — 22 features, within-problem pairwise logistic
  full_pairwise         — all features, pairwise logistic
  oracle_rank_bound     — fraction of pairwise pairs that are distinguishable (Bayes ceiling)

Usage (from repo root):
    python3 SVDomain/experiments/run_coding_bottleneck_v3.py
    python3 SVDomain/experiments/run_coding_bottleneck_v3.py --smoke
    python3 SVDomain/experiments/run_coding_bottleneck_v3.py \\
        --feature-cache results/cache/earlystop_strongfeat_round1/cache_cap8_ref030_9494cc35a9202b8d.pkl
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.ops.earlystop_svd import (
    AVAILABILITY_FEATURES,
    CODING_DERIVATIVE_FEATURES,
    CODING_DYNAMIC_FEATURES,
    FULL_FEATURE_NAMES,
    TOKEN_FEATURES,
    TRAJ_FEATURES,
    _auroc,
    _rank_transform_matrix,
)
from scripts.run_earlystop_prefix10_svd_round1 import (
    EXTRACTION_POSITION_INDEX,
    EXTRACTION_POSITIONS,
    FEATURE_TO_INDEX,
    PREFIX_SAFE_FEATURE_FAMILY_MAP,
)

# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_FEATURE_CACHE = (
    "results/cache/rule_and_family_baselines/cache_all_ref030_45392d82523f21bf.pkl"
)
DEFAULT_OUT_CSV = "results/tables/coding_bottleneck_v3.csv"
DEFAULT_N_SPLITS = 5
DEFAULT_SEEDS = (42, 101, 29)

LATE_POSITIONS = (0.70, 1.00)

FIXED_22_NAMES = list(PREFIX_SAFE_FEATURE_FAMILY_MAP["token_plus_traj"])
TOKEN_ONLY_NAMES = list(PREFIX_SAFE_FEATURE_FAMILY_MAP["token_only"])

# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_indices(names: list[str], tensor_n_feat: int) -> list[int]:
    """Return valid indices for feature names, clamped to available tensor cols."""
    full = list(FULL_FEATURE_NAMES)
    idxs = []
    for name in names:
        idx = FEATURE_TO_INDEX.get(name, -1)
        if 0 <= idx < tensor_n_feat:
            idxs.append(idx)
    return idxs


def _auroc_safe(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = labels == 1
    neg = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    return _auroc(scores, labels)


def _pairwise_acc(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> float:
    """Within-problem pairwise ranking accuracy (fraction of cross-label pairs correctly ordered)."""
    n_correct = 0
    n_total = 0
    for g in np.unique(groups):
        mask = groups == g
        s_g = scores[mask]
        y_g = labels[mask]
        pos_idx = np.where(y_g == 1)[0]
        neg_idx = np.where(y_g == 0)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        for pi in pos_idx:
            for ni in neg_idx:
                n_total += 1
                if s_g[pi] > s_g[ni]:
                    n_correct += 1
                elif s_g[pi] == s_g[ni]:
                    n_correct += 0.5
    return float(n_correct) / float(n_total) if n_total > 0 else float("nan")


def _hit_at_k(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray, k: int = 1) -> float:
    """Fraction of problems where top-k contains at least one correct solution."""
    hits = 0
    n_problems = 0
    for g in np.unique(groups):
        mask = groups == g
        s_g = scores[mask]
        y_g = labels[mask]
        if y_g.sum() == 0:
            continue  # skip all-negative problems (not a valid evaluation point)
        n_problems += 1
        top_k = np.argsort(-s_g)[:k]
        if y_g[top_k].sum() > 0:
            hits += 1
    return float(hits) / float(n_problems) if n_problems > 0 else float("nan")


def _oracle_stats(labels: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """Compute oracle ceiling statistics:
    - informative_pair_fraction: cross-label pairs / all pairs (signal density)
    - pct_informative_problems: problems with both pos+neg labels
    - oracle_hit1: problems where at least one solution is correct (max achievable hit@1)
    Note: oracle pairwise_acc (for cross-label pairs only) = 1.0 by definition.
    """
    n_cross = 0
    n_total = 0
    n_informative = 0
    n_achievable_hit1 = 0
    unique_groups = np.unique(groups)
    for g in unique_groups:
        mask = groups == g
        y_g = labels[mask]
        n_pos = int(y_g.sum())
        n_neg = len(y_g) - n_pos
        cross = n_pos * n_neg
        same = n_pos * (n_pos - 1) // 2 + n_neg * (n_neg - 1) // 2
        n_cross += cross
        n_total += cross + same
        if n_pos > 0 and n_neg > 0:
            n_informative += 1
        if n_pos > 0:
            n_achievable_hit1 += 1
    n_problems = len(unique_groups)
    return {
        "informative_pair_fraction": float(n_cross) / float(n_total) if n_total > 0 else float("nan"),
        "pct_informative_problems": float(n_informative) / float(n_problems) if n_problems > 0 else float("nan"),
        "oracle_hit1": float(n_achievable_hit1) / float(n_problems) if n_problems > 0 else float("nan"),
    }


def _build_pairwise_examples(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pair_x, pair_y = [], []
    for g in np.unique(groups):
        mask = groups == g
        idx = np.where(mask)[0]
        y_g = y[idx]
        pos = idx[y_g == 1]
        neg = idx[y_g == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        diffs_pos = (x[pos][:, None, :] - x[neg][None, :, :]).reshape(-1, x.shape[1])
        diffs_neg = -diffs_pos
        pair_x.append(np.vstack([diffs_pos, diffs_neg]))
        pair_y.append(np.concatenate([
            np.ones(diffs_pos.shape[0], dtype=np.int32),
            np.zeros(diffs_neg.shape[0], dtype=np.int32),
        ]))
    if not pair_x:
        return np.zeros((0, x.shape[1]), dtype=np.float64), np.zeros(0, dtype=np.int32)
    return np.vstack(pair_x).astype(np.float64), np.concatenate(pair_y).astype(np.int32)


def _pairwise_scores_from_clf(
    clf: Any, scaler: StandardScaler, x: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    """Convert pairwise LR to pointwise scores via within-group mean decision function."""
    x_sc = scaler.transform(x)
    scores = np.zeros(len(x), dtype=np.float64)
    for g in np.unique(groups):
        mask = groups == g
        idx = np.where(mask)[0]
        x_g = x_sc[idx]
        s_g = np.zeros(len(idx), dtype=np.float64)
        for j in range(len(idx)):
            diffs = x_g[j:j+1] - x_g
            s_g[j] = float(np.mean(clf.decision_function(diffs)))
        scores[idx] = s_g
    return scores


# ── CV evaluation ─────────────────────────────────────────────────────────────

def _cv_pointwise(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray,
    n_splits: int, seeds: tuple[int, ...]
) -> dict[str, float]:
    folds = list(GroupKFold(n_splits=min(n_splits, len(np.unique(groups)))).split(x, groups=groups))
    if not folds:
        return {"auroc": float("nan"), "pairwise_acc": float("nan"), "hit@1": float("nan")}

    aurocs, pw_accs, hit1s = [], [], []
    for train_idx, test_idx in folds:
        y_tr, y_te = y[train_idx], y[test_idx]
        x_tr, x_te = x[train_idx], x[test_idx]
        g_te = groups[test_idx]
        if np.unique(y_tr).shape[0] < 2 or np.unique(y_te).shape[0] < 2:
            continue

        fold_scores = np.zeros(len(y_te), dtype=np.float64)
        for seed in seeds:
            sc = StandardScaler()
            x_tr_sc = sc.fit_transform(x_tr)
            x_te_sc = sc.transform(x_te)
            clf = LogisticRegression(
                C=0.2, max_iter=1000, solver="lbfgs", class_weight="balanced",
                random_state=seed
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(x_tr_sc, y_tr)
            fold_scores += clf.decision_function(x_te_sc)
        fold_scores /= len(seeds)

        aurocs.append(_auroc_safe(fold_scores, y_te))
        pw_accs.append(_pairwise_acc(fold_scores, y_te, g_te))
        hit1s.append(_hit_at_k(fold_scores, y_te, g_te, k=1))

    def _m(v: list) -> float:
        arr = np.array([x for x in v if np.isfinite(x)])
        return float(np.mean(arr)) if arr.size > 0 else float("nan")

    return {"auroc": _m(aurocs), "pairwise_acc": _m(pw_accs), "hit@1": _m(hit1s)}


def _cv_pairwise(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray,
    n_splits: int, seeds: tuple[int, ...]
) -> dict[str, float]:
    folds = list(GroupKFold(n_splits=min(n_splits, len(np.unique(groups)))).split(x, groups=groups))
    if not folds:
        return {"auroc": float("nan"), "pairwise_acc": float("nan"), "hit@1": float("nan")}

    aurocs, pw_accs, hit1s = [], [], []
    for train_idx, test_idx in folds:
        y_tr, y_te = y[train_idx], y[test_idx]
        x_tr, x_te = x[train_idx], x[test_idx]
        g_te = groups[test_idx]
        g_tr = groups[train_idx]
        if np.unique(y_tr).shape[0] < 2 or np.unique(y_te).shape[0] < 2:
            continue

        x_pair_tr, y_pair_tr = _build_pairwise_examples(x_tr, y_tr, g_tr)
        if x_pair_tr.shape[0] < 10:
            continue

        fold_scores = np.zeros(len(y_te), dtype=np.float64)
        for seed in seeds:
            sc = StandardScaler()
            x_pair_sc = sc.fit_transform(x_pair_tr)
            clf = LogisticRegression(
                C=0.2, max_iter=1000, solver="lbfgs",
                random_state=seed
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(x_pair_sc, y_pair_tr)
            fold_scores += _pairwise_scores_from_clf(clf, sc, x_te, g_te)
        fold_scores /= len(seeds)

        aurocs.append(_auroc_safe(fold_scores, y_te))
        pw_accs.append(_pairwise_acc(fold_scores, y_te, g_te))
        hit1s.append(_hit_at_k(fold_scores, y_te, g_te, k=1))

    def _m(v: list) -> float:
        arr = np.array([x for x in v if np.isfinite(x)])
        return float(np.mean(arr)) if arr.size > 0 else float("nan")

    return {"auroc": _m(aurocs), "pairwise_acc": _m(pw_accs), "hit@1": _m(hit1s)}


# ── main ──────────────────────────────────────────────────────────────────────

def _load_coding_tensor(feature_cache: str) -> list[dict[str, Any]]:
    with open(feature_cache, "rb") as fh:
        data = pickle.load(fh)
    fs = data["feature_store"]
    return [item for item in fs if item.get("domain") == "coding"]


def run(args: argparse.Namespace) -> None:
    feature_cache = str(args.feature_cache)
    print(f"[load] feature cache: {feature_cache}", flush=True)
    coding_items = _load_coding_tensor(feature_cache)
    if not coding_items:
        print("ERROR: no coding domain items found in feature cache")
        sys.exit(1)

    print(f"[info] found {len(coding_items)} coding cache(s)")

    # Merge all coding items into one flat array
    all_tensors, all_labels, all_groups = [], [], []
    for item in coding_items:
        all_tensors.append(np.asarray(item["tensor"], dtype=np.float64))
        all_labels.append(np.asarray(item["labels"], dtype=np.int32))
        gk = item.get("group_keys")
        if gk is None:
            # Build from problem_ids + problem_offsets
            offsets = item["problem_offsets"]
            pids = item["problem_ids"]
            gk_arr = np.empty(item["tensor"].shape[0], dtype=object)
            for pi, pid in enumerate(pids):
                gk_arr[offsets[pi]:offsets[pi+1]] = pid
            gk = gk_arr
        all_groups.append(np.asarray(gk, dtype=object))

    full_tensor = np.concatenate(all_tensors, axis=0)   # (N, n_pos, n_feat)
    full_labels = np.concatenate(all_labels, axis=0)    # (N,)
    full_groups = np.concatenate(all_groups, axis=0)    # (N,)

    n_pos_in_tensor = full_tensor.shape[1]
    n_feat = full_tensor.shape[2]
    print(f"[info] tensor shape: {full_tensor.shape}")
    print(f"[info] labels: n_pos={int(full_labels.sum())} n_neg={int((1-full_labels).sum())}")

    # Determine position indices for 0.70 and 1.00
    pos_map = {float(p): i for i, p in enumerate(EXTRACTION_POSITIONS)}

    target_positions = []
    for p in LATE_POSITIONS:
        if p in pos_map and pos_map[p] < n_pos_in_tensor:
            target_positions.append((p, pos_map[p]))
        else:
            print(f"[warn] position {p} not available in tensor (n_pos={n_pos_in_tensor})")

    if not target_positions:
        print("ERROR: no valid target positions found")
        sys.exit(1)

    # Feature index sets
    fixed_22_idx = _safe_indices(FIXED_22_NAMES, n_feat)
    token_only_idx = _safe_indices(TOKEN_ONLY_NAMES, n_feat)
    all_idx = list(range(n_feat))
    # Proxy for response_length_only: use tok_conf_prefix (lower = longer responses in many settings)
    length_proxy_idx = [FEATURE_TO_INDEX.get("tok_conf_prefix", 0)]
    length_proxy_idx = [i for i in length_proxy_idx if i < n_feat]

    # Optionally subsample for smoke mode
    if args.smoke:
        rng = np.random.RandomState(42)
        unique_groups = np.unique(full_groups)
        n_smoke = max(10, len(unique_groups) // 5)
        chosen = rng.choice(unique_groups, size=n_smoke, replace=False)
        chosen_set = set(chosen.tolist())
        mask = np.array([g in chosen_set for g in full_groups.tolist()])
        full_tensor = full_tensor[mask]
        full_labels = full_labels[mask]
        full_groups = full_groups[mask]
        print(f"[smoke] using {mask.sum()} samples / {n_smoke} problems")

    n_groups = len(np.unique(full_groups))
    n_splits = min(args.n_splits, n_groups)

    rows = []

    # Oracle stats (position-independent)
    oracle_s = _oracle_stats(full_labels, full_groups)
    print(
        f"[oracle] informative_pair_fraction={oracle_s['informative_pair_fraction']:.4f}  "
        f"pct_informative_problems={oracle_s['pct_informative_problems']:.4f}  "
        f"oracle_hit1={oracle_s['oracle_hit1']:.4f}"
    )

    for pos_val, pos_idx in target_positions:
        x_raw = full_tensor[:, pos_idx, :]  # (N, n_feat)
        x_rank = _rank_transform_matrix(x_raw)

        # Oracle bound row
        rows.append({
            "condition": "oracle_rank_bound",
            "position": pos_val,
            "n_feat": n_feat,
            "auroc": float("nan"),
            "pairwise_acc": oracle_s["informative_pair_fraction"],
            "hit@1": oracle_s["oracle_hit1"],
        })

        conditions = [
            ("response_length_proxy", length_proxy_idx, "pointwise"),
            ("token_only_logistic", token_only_idx, "pointwise"),
            ("fixed_22_logistic", fixed_22_idx, "pointwise"),
            ("full_logistic", all_idx, "pointwise"),
            ("fixed_22_pairwise", fixed_22_idx, "pairwise"),
            ("full_pairwise", all_idx, "pairwise"),
        ]

        for cond_name, feat_idx, obj in conditions:
            if not feat_idx:
                print(f"[skip] {cond_name} — no valid feature indices")
                continue
            x_sub = x_raw[:, feat_idx]

            print(f"[run] cond={cond_name} pos={pos_val:.0%} n_feat={len(feat_idx)} obj={obj}", flush=True)
            if obj == "pointwise":
                metrics = _cv_pointwise(x_sub, full_labels, full_groups, n_splits, DEFAULT_SEEDS)
            else:
                metrics = _cv_pairwise(x_sub, full_labels, full_groups, n_splits, DEFAULT_SEEDS)

            rows.append({
                "condition": cond_name,
                "position": pos_val,
                "n_feat": len(feat_idx),
                **metrics,
            })
            print(
                f"  auroc={metrics['auroc']:.4f}  pairwise_acc={metrics['pairwise_acc']:.4f}"
                f"  hit@1={metrics['hit@1']:.4f}"
            )

    # Print summary
    print("\n=== SUMMARY ===")
    header = f"{'condition':<28} {'pos':>5} {'n_feat':>7} {'auroc':>8} {'pw_acc':>8} {'hit@1':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['condition']:<28} {r['position']:>5.0%} {r['n_feat']:>7} "
            f"{r['auroc']:>8.4f} {r['pairwise_acc']:>8.4f} {r['hit@1']:>7.4f}"
        )

    # Write CSV
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["condition", "position", "n_feat", "auroc", "pairwise_acc", "hit@1"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[save] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Coding bottleneck diagnosis V3")
    parser.add_argument("--feature-cache", default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--smoke", action="store_true", help="Use 20%% of problems for fast verification")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
