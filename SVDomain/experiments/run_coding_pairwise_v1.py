#!/usr/bin/env python3
"""Coding Pairwise Ranking V1 — makes pairwise ranking the primary objective.

Primary metric: within-problem pairwise accuracy and hit@1 (not AUROC).
Tests whether pairwise training at late anchors beats pointwise correctness prediction.

Conditions:
  direct_pairwise_22    — Raw 22 features (no SVD), pairwise logistic, anchors: 70/80/90/100/%/late_mean
  direct_pairwise_full  — All features, pairwise logistic, 70%/100%/late_mean
  direct_pointwise_22   — Raw 22 features, pointwise LR, 70%/100% (control)
  direct_pointwise_full — All features, pointwise LR, 70%/100% (control)

Usage (from repo root):
    python3 SVDomain/experiments/run_coding_pairwise_v1.py
    python3 SVDomain/experiments/run_coding_pairwise_v1.py --smoke
    python3 SVDomain/experiments/run_coding_pairwise_v1.py \\
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
    FULL_FEATURE_NAMES,
    _auroc,
    _rank_transform_matrix,
)
from scripts.run_earlystop_prefix10_svd_round1 import (
    EXTRACTION_POSITIONS,
    FEATURE_TO_INDEX,
    PREFIX_SAFE_FEATURE_FAMILY_MAP,
)

# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_FEATURE_CACHE = (
    "results/cache/rule_and_family_baselines/cache_all_ref030_45392d82523f21bf.pkl"
)
DEFAULT_OUT_CSV = "results/tables/coding_pairwise_v1.csv"
DEFAULT_N_SPLITS = 5
DEFAULT_SEEDS = (42, 101, 29)

FIXED_22_NAMES = list(PREFIX_SAFE_FEATURE_FAMILY_MAP["token_plus_traj"])

# Positions to evaluate: 70/80/90/100%
EVAL_POSITIONS = (0.70, 0.80, 0.90, 1.00)

# ── metric helpers ────────────────────────────────────────────────────────────

def _auroc_safe(scores: np.ndarray, labels: np.ndarray) -> float:
    if (labels == 1).sum() == 0 or (labels == 0).sum() == 0:
        return float("nan")
    return _auroc(scores, labels)


def _pairwise_acc(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> float:
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


def _hit_at_1(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> float:
    hits, n_problems = 0, 0
    for g in np.unique(groups):
        mask = groups == g
        s_g = scores[mask]
        y_g = labels[mask]
        if y_g.sum() == 0:
            continue
        n_problems += 1
        best = int(np.argmax(s_g))
        if y_g[best] == 1:
            hits += 1
    return float(hits) / float(n_problems) if n_problems > 0 else float("nan")


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
    x_sc = scaler.transform(x)
    scores = np.zeros(len(x), dtype=np.float64)
    for g in np.unique(groups):
        mask = groups == g
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        x_g = x_sc[idx]
        s_g = np.zeros(len(idx), dtype=np.float64)
        for j in range(len(idx)):
            diffs = x_g[j:j+1] - x_g
            s_g[j] = float(np.mean(clf.decision_function(diffs)))
        scores[idx] = s_g
    return scores


# ── CV runners ────────────────────────────────────────────────────────────────

def _cv_run(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray,
    objective: str, n_splits: int, seeds: tuple[int, ...]
) -> dict[str, float]:
    """Run GroupKFold CV for pointwise or pairwise objective."""
    n_groups = len(np.unique(groups))
    folds = list(GroupKFold(n_splits=min(n_splits, n_groups)).split(x, groups=groups))
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

        fold_scores = np.zeros(len(y_te), dtype=np.float64)
        for seed in seeds:
            sc = StandardScaler()
            if objective == "pointwise":
                x_tr_sc = sc.fit_transform(x_tr)
                x_te_sc = sc.transform(x_te)
                clf = LogisticRegression(
                    C=0.2, max_iter=1000, solver="lbfgs",
                    class_weight="balanced", random_state=seed
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf.fit(x_tr_sc, y_tr)
                fold_scores += clf.decision_function(x_te_sc)
            else:  # pairwise
                x_pair_tr, y_pair_tr = _build_pairwise_examples(x_tr, y_tr, g_tr)
                if x_pair_tr.shape[0] < 10:
                    continue
                x_pair_sc = sc.fit_transform(x_pair_tr)
                clf = LogisticRegression(
                    C=0.2, max_iter=1000, solver="lbfgs", random_state=seed
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf.fit(x_pair_sc, y_pair_tr)
                fold_scores += _pairwise_scores_from_clf(clf, sc, x_te, g_te)
        fold_scores /= len(seeds)

        aurocs.append(_auroc_safe(fold_scores, y_te))
        pw_accs.append(_pairwise_acc(fold_scores, y_te, g_te))
        hit1s.append(_hit_at_1(fold_scores, y_te, g_te))

    def _m(v: list) -> float:
        arr = np.array([x for x in v if np.isfinite(x)])
        return float(np.mean(arr)) if arr.size > 0 else float("nan")

    return {"auroc": _m(aurocs), "pairwise_acc": _m(pw_accs), "hit@1": _m(hit1s)}


# ── main ──────────────────────────────────────────────────────────────────────

def _load_coding_items(feature_cache: str) -> list[dict[str, Any]]:
    with open(feature_cache, "rb") as fh:
        data = pickle.load(fh)
    return [item for item in data["feature_store"] if item.get("domain") == "coding"]


def run(args: argparse.Namespace) -> None:
    print(f"[load] {args.feature_cache}", flush=True)
    coding_items = _load_coding_items(args.feature_cache)
    if not coding_items:
        print("ERROR: no coding items found")
        sys.exit(1)

    # Merge all coding items
    all_tensors, all_labels, all_groups = [], [], []
    for item in coding_items:
        t = np.asarray(item["tensor"], dtype=np.float64)
        all_tensors.append(t)
        all_labels.append(np.asarray(item["labels"], dtype=np.int32))
        gk = item.get("group_keys")
        if gk is None:
            offsets = item["problem_offsets"]
            pids = item["problem_ids"]
            gk = np.empty(t.shape[0], dtype=object)
            for pi, pid in enumerate(pids):
                gk[offsets[pi]:offsets[pi+1]] = pid
        all_groups.append(np.asarray(gk, dtype=object))

    full_tensor = np.concatenate(all_tensors, axis=0)
    full_labels = np.concatenate(all_labels, axis=0)
    full_groups = np.concatenate(all_groups, axis=0)

    n_feat = full_tensor.shape[2]
    n_pos_in_tensor = full_tensor.shape[1]
    print(f"[info] tensor {full_tensor.shape}  pos={int(full_labels.sum())}+ neg={int((1-full_labels).sum())}-")

    pos_map = {float(p): i for i, p in enumerate(EXTRACTION_POSITIONS)}

    # Feature subsets
    fixed_22_idx = [FEATURE_TO_INDEX[n] for n in FIXED_22_NAMES if FEATURE_TO_INDEX.get(n, -1) < n_feat]
    all_idx = list(range(n_feat))

    if args.smoke:
        rng = np.random.RandomState(42)
        unique_groups = np.unique(full_groups)
        n_smoke = max(10, len(unique_groups) // 5)
        chosen = set(rng.choice(unique_groups, size=n_smoke, replace=False).tolist())
        mask = np.array([g in chosen for g in full_groups.tolist()])
        full_tensor = full_tensor[mask]
        full_labels = full_labels[mask]
        full_groups = full_groups[mask]
        print(f"[smoke] {mask.sum()} samples / {len(chosen)} problems")

    n_splits = min(args.n_splits, len(np.unique(full_groups)))

    # Build conditions: (condition_name, feat_idx, objective, positions)
    conditions = [
        ("direct_pairwise_22", fixed_22_idx, "pairwise", EVAL_POSITIONS),
        ("direct_pairwise_full", all_idx, "pairwise", (0.70, 1.00)),
        ("direct_pointwise_22", fixed_22_idx, "pointwise", (0.70, 1.00)),
        ("direct_pointwise_full", all_idx, "pointwise", (0.70, 1.00)),
    ]

    rows = []
    for cond_name, feat_idx, objective, positions in conditions:
        if not feat_idx:
            print(f"[skip] {cond_name} — empty feature set")
            continue

        pos_results = {}
        for pos_val in positions:
            if pos_val not in pos_map or pos_map[pos_val] >= n_pos_in_tensor:
                print(f"[warn] {cond_name} pos={pos_val} not in tensor")
                continue
            pos_idx = pos_map[pos_val]
            x_raw = full_tensor[:, pos_idx, :]
            x_sub = x_raw[:, feat_idx]

            print(f"[run] {cond_name} pos={pos_val:.0%} n_feat={len(feat_idx)}", flush=True)
            metrics = _cv_run(x_sub, full_labels, full_groups, objective, n_splits, DEFAULT_SEEDS)
            pos_results[pos_val] = metrics
            rows.append({
                "condition": cond_name,
                "objective": objective,
                "position": pos_val,
                "n_feat": len(feat_idx),
                **metrics,
            })
            print(f"  auroc={metrics['auroc']:.4f}  pairwise_acc={metrics['pairwise_acc']:.4f}  hit@1={metrics['hit@1']:.4f}")

        # Late mean aggregate across available positions >= 0.70
        late_vals = {p: v for p, v in pos_results.items() if p >= 0.70}
        if len(late_vals) >= 2:
            for metric_name in ("auroc", "pairwise_acc", "hit@1"):
                vals = [v[metric_name] for v in late_vals.values() if np.isfinite(v[metric_name])]
                late_mean = float(np.mean(vals)) if vals else float("nan")
                rows.append({
                    "condition": cond_name,
                    "objective": objective,
                    "position": "late_mean",
                    "n_feat": len(feat_idx),
                    "auroc": late_mean if metric_name == "auroc" else float("nan"),
                    "pairwise_acc": late_mean if metric_name == "pairwise_acc" else float("nan"),
                    "hit@1": late_mean if metric_name == "hit@1" else float("nan"),
                })
            # Collapse the three late_mean rows into one
            rows = rows[:-3]  # remove the three partial rows
            late_row = {
                "condition": cond_name,
                "objective": objective,
                "position": "late_mean",
                "n_feat": len(feat_idx),
            }
            for metric_name in ("auroc", "pairwise_acc", "hit@1"):
                vals = [v[metric_name] for v in late_vals.values() if np.isfinite(v[metric_name])]
                late_row[metric_name] = float(np.mean(vals)) if vals else float("nan")
            rows.append(late_row)

    # Summary
    print("\n=== SUMMARY ===")
    header = f"{'condition':<26} {'obj':<12} {'pos':>9} {'n_feat':>7} {'auroc':>8} {'pw_acc':>8} {'hit@1':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        pos_str = f"{r['position']:.0%}" if isinstance(r["position"], float) else str(r["position"])
        print(
            f"{r['condition']:<26} {r['objective']:<12} {pos_str:>9} {r['n_feat']:>7} "
            f"{r.get('auroc', float('nan')):>8.4f} {r.get('pairwise_acc', float('nan')):>8.4f} "
            f"{r.get('hit@1', float('nan')):>7.4f}"
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["condition", "objective", "position", "n_feat", "auroc", "pairwise_acc", "hit@1"]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[save] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Coding pairwise ranking V1")
    parser.add_argument("--feature-cache", default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
