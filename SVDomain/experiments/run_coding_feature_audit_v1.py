#!/usr/bin/env python3
"""Coding Feature Language Audit V1 — identifies which feature families predict coding correctness.

Approach:
  1. Train GradientBoosting on all available features at pos=1.00
  2. Extract gain-based + permutation importance
  3. Group importances by family: token_stats, traj, availability, coding_dynamic, coding_derivative
  4. Family ablation: remove each family and measure AUROC delta
  5. Derived feature test: add 3 composite features computed from existing raw features

New derived features (computed from FULL_FEATURE_NAMES, no cache rebuild needed):
  gini_div_recency_ratio    = tok_gini_slope / (|tok_gini_tail| + eps)
  conf_logprob_gap          = tok_conf_prefix - tok_logprob_prefix   (proxy for overconfidence)
  late_instability_density  = last_block_instability * reflection_density

Usage (from repo root):
    python3 SVDomain/experiments/run_coding_feature_audit_v1.py
    python3 SVDomain/experiments/run_coding_feature_audit_v1.py --smoke
    python3 SVDomain/experiments/run_coding_feature_audit_v1.py \\
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
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
    EXTRACTION_POSITIONS,
    FEATURE_TO_INDEX,
)

# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_FEATURE_CACHE = (
    "results/cache/rule_and_family_baselines/cache_all_ref030_45392d82523f21bf.pkl"
)
DEFAULT_OUT_CSV = "results/tables/coding_feature_audit_v1.csv"
DEFAULT_FIG_PATH = "results/figures/coding_feature_importance.png"
DEFAULT_N_SPLITS = 5

# Position to use for audit (1.00 = full generation)
AUDIT_POSITION = 1.00

# Feature families
FAMILY_MAP: dict[str, list[str]] = {
    "token_stats": TOKEN_FEATURES,
    "traj": TRAJ_FEATURES,
    "availability": AVAILABILITY_FEATURES,
    "coding_dynamic": CODING_DYNAMIC_FEATURES,
    "coding_derivative": CODING_DERIVATIVE_FEATURES,
}

DERIVED_FEATURE_NAMES = [
    "gini_div_recency_ratio",
    "conf_logprob_gap",
    "late_instability_density",
]

# ── helpers ───────────────────────────────────────────────────────────────────

def _auroc_safe(scores: np.ndarray, labels: np.ndarray) -> float:
    if (labels == 1).sum() == 0 or (labels == 0).sum() == 0:
        return float("nan")
    return _auroc(scores, labels)


def _load_coding_items(feature_cache: str) -> list[dict[str, Any]]:
    with open(feature_cache, "rb") as fh:
        data = pickle.load(fh)
    return [item for item in data["feature_store"] if item.get("domain") == "coding"]


def _build_derived_features(x_raw: np.ndarray, n_feat: int) -> np.ndarray:
    """Compute 3 derived composite features from raw feature matrix."""
    eps = 1e-6
    fi = {name: idx for name, idx in FEATURE_TO_INDEX.items() if idx < n_feat}

    def _col(name: str) -> np.ndarray:
        idx = fi.get(name, -1)
        if idx < 0:
            return np.zeros(x_raw.shape[0], dtype=np.float64)
        return x_raw[:, idx].astype(np.float64)

    gini_slope = _col("tok_gini_slope")
    gini_tail = _col("tok_gini_tail")
    gini_div_recency = gini_slope / (np.abs(gini_tail) + eps)

    conf_prefix = _col("tok_conf_prefix")
    logprob_prefix = _col("tok_logprob_prefix")
    conf_logprob_gap = conf_prefix - logprob_prefix

    instability = _col("last_block_instability")
    refl_density = _col("reflection_density")
    late_instability = instability * refl_density

    derived = np.stack([gini_div_recency, conf_logprob_gap, late_instability], axis=1)
    return derived.astype(np.float64)


def _cv_auroc(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray,
    n_splits: int, use_gb: bool = True
) -> float:
    n_groups = len(np.unique(groups))
    folds = list(GroupKFold(n_splits=min(n_splits, n_groups)).split(x, groups=groups))
    if not folds:
        return float("nan")

    aurocs = []
    for train_idx, test_idx in folds:
        y_tr, y_te = y[train_idx], y[test_idx]
        x_tr, x_te = x[train_idx], x[test_idx]
        if np.unique(y_tr).shape[0] < 2 or np.unique(y_te).shape[0] < 2:
            continue

        sc = StandardScaler()
        x_tr_sc = sc.fit_transform(x_tr)
        x_te_sc = sc.transform(x_te)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if use_gb:
                clf = GradientBoostingClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    min_samples_leaf=5, subsample=0.8, random_state=42
                )
            else:
                clf = LogisticRegression(
                    C=0.2, max_iter=1000, solver="lbfgs",
                    class_weight="balanced", random_state=42
                )
            clf.fit(x_tr_sc, y_tr)

        if use_gb:
            scores = clf.predict_proba(x_te_sc)[:, 1]
        else:
            scores = clf.decision_function(x_te_sc)
        aurocs.append(_auroc_safe(scores, y_te))

    arr = np.array([v for v in aurocs if np.isfinite(v)])
    return float(np.mean(arr)) if arr.size > 0 else float("nan")


def _compute_permutation_importance(
    x: np.ndarray, y: np.ndarray, feature_names: list[str], n_splits: int
) -> dict[str, float]:
    """Single-split permutation importance (train on 70%, eval on 30%)."""
    rng = np.random.RandomState(42)
    n = len(y)
    idx = rng.permutation(n)
    split = int(0.7 * n)
    tr_idx, te_idx = idx[:split], idx[split:]

    y_tr, y_te = y[tr_idx], y[te_idx]
    x_tr, x_te = x[tr_idx], x[te_idx]
    if np.unique(y_tr).shape[0] < 2 or np.unique(y_te).shape[0] < 2:
        return {name: float("nan") for name in feature_names}

    sc = StandardScaler()
    x_tr_sc = sc.fit_transform(x_tr)
    x_te_sc = sc.transform(x_te)

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=5, subsample=0.8, random_state=42
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(x_tr_sc, y_tr)

    try:
        result = permutation_importance(
            clf, x_te_sc, y_te, n_repeats=10, random_state=42, scoring="roc_auc"
        )
        return {name: float(result.importances_mean[i]) for i, name in enumerate(feature_names)}
    except Exception:
        return {name: float("nan") for name in feature_names}


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
    n_pos = full_tensor.shape[1]
    print(f"[info] tensor {full_tensor.shape}")

    # Get position index for AUDIT_POSITION
    pos_map = {float(p): i for i, p in enumerate(EXTRACTION_POSITIONS)}
    if AUDIT_POSITION not in pos_map or pos_map[AUDIT_POSITION] >= n_pos:
        # Fall back to last position
        audit_pos_idx = n_pos - 1
        print(f"[warn] position {AUDIT_POSITION} not found, using last ({EXTRACTION_POSITIONS[audit_pos_idx]:.0%})")
    else:
        audit_pos_idx = pos_map[AUDIT_POSITION]

    x_raw = full_tensor[:, audit_pos_idx, :]  # (N, n_feat)

    if args.smoke:
        rng = np.random.RandomState(42)
        unique_groups = np.unique(full_groups)
        n_smoke = max(10, len(unique_groups) // 5)
        chosen = set(rng.choice(unique_groups, size=n_smoke, replace=False).tolist())
        mask = np.array([g in chosen for g in full_groups.tolist()])
        x_raw = x_raw[mask]
        full_labels = full_labels[mask]
        full_groups = full_groups[mask]
        print(f"[smoke] {mask.sum()} samples / {len(chosen)} problems")

    n_splits = min(args.n_splits, len(np.unique(full_groups)))

    # Available feature names (capped to tensor width)
    avail_feature_names = list(FULL_FEATURE_NAMES[:n_feat])
    avail_idx = list(range(n_feat))

    # Append derived features
    x_derived = _build_derived_features(x_raw, n_feat)
    x_augmented = np.concatenate([x_raw, x_derived], axis=1)
    augmented_names = avail_feature_names + DERIVED_FEATURE_NAMES

    rows = []

    # ── 1. Baseline: full features (LR) ──────────────────────────────────────
    print("[run] baseline full_lr", flush=True)
    auroc_lr = _cv_auroc(x_raw[:, avail_idx], full_labels, full_groups, n_splits, use_gb=False)
    rows.append({"experiment": "full_lr", "family_removed": "none", "auroc": auroc_lr, "delta": 0.0})
    print(f"  full_lr auroc={auroc_lr:.4f}")

    # ── 2. Full GradientBoosting ──────────────────────────────────────────────
    print("[run] baseline full_gb", flush=True)
    auroc_gb = _cv_auroc(x_raw[:, avail_idx], full_labels, full_groups, n_splits, use_gb=True)
    rows.append({"experiment": "full_gb", "family_removed": "none", "auroc": auroc_gb, "delta": 0.0})
    print(f"  full_gb auroc={auroc_gb:.4f}")

    # ── 3. GradientBoosting with derived features ─────────────────────────────
    print("[run] full_gb + derived", flush=True)
    auroc_gb_deriv = _cv_auroc(x_augmented, full_labels, full_groups, n_splits, use_gb=True)
    rows.append({"experiment": "full_gb_plus_derived", "family_removed": "none",
                 "auroc": auroc_gb_deriv, "delta": auroc_gb_deriv - auroc_gb})
    print(f"  full_gb_plus_derived auroc={auroc_gb_deriv:.4f}  delta={auroc_gb_deriv - auroc_gb:+.4f}")

    # ── 4. Family ablation ────────────────────────────────────────────────────
    for family_name, family_feat_names in FAMILY_MAP.items():
        keep_idx = [
            i for i, name in enumerate(avail_feature_names)
            if name not in set(family_feat_names)
        ]
        if not keep_idx:
            print(f"[skip] ablation {family_name} — would remove all features")
            continue
        print(f"[run] ablation remove={family_name} ({n_feat - len(keep_idx)} feats removed)", flush=True)
        auroc_abl = _cv_auroc(x_raw[:, keep_idx], full_labels, full_groups, n_splits, use_gb=True)
        delta = auroc_abl - auroc_gb
        rows.append({
            "experiment": f"ablation_remove_{family_name}",
            "family_removed": family_name,
            "auroc": auroc_abl,
            "delta": delta,
        })
        print(f"  auroc={auroc_abl:.4f}  delta={delta:+.4f}")

    # ── 5. Individual feature importance (gain from final GB model) ───────────
    print("[run] computing feature importances on full data", flush=True)
    sc = StandardScaler()
    x_sc = sc.fit_transform(x_raw[:, avail_idx])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gb_full = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=5, subsample=0.8, random_state=42
        )
        gb_full.fit(x_sc, full_labels)

    gain_importances = {avail_feature_names[i]: float(imp)
                        for i, imp in enumerate(gb_full.feature_importances_)}

    # Permutation importance (single split, fast)
    print("[run] permutation importance", flush=True)
    perm_importances = _compute_permutation_importance(
        x_raw[:, avail_idx], full_labels, avail_feature_names, n_splits
    )

    # Aggregate importances by family
    print("\n=== FEATURE IMPORTANCE BY FAMILY ===")
    family_importance: dict[str, dict[str, float]] = {}
    for family_name, family_feat_names in FAMILY_MAP.items():
        gain_vals = [gain_importances.get(n, 0.0) for n in family_feat_names if n in gain_importances]
        perm_vals = [perm_importances.get(n, float("nan")) for n in family_feat_names if n in perm_importances]
        family_importance[family_name] = {
            "gain_sum": float(np.sum(gain_vals)),
            "gain_mean": float(np.mean(gain_vals)) if gain_vals else 0.0,
            "perm_mean": float(np.nanmean(perm_vals)) if perm_vals else float("nan"),
        }
        print(f"  {family_name:<20} gain_sum={family_importance[family_name]['gain_sum']:.4f}  "
              f"gain_mean={family_importance[family_name]['gain_mean']:.4f}  "
              f"perm_mean={family_importance[family_name]['perm_mean']:.4f}")

    # Top-10 individual features
    print("\n=== TOP-10 FEATURES BY GAIN ===")
    sorted_gain = sorted(gain_importances.items(), key=lambda kv: kv[1], reverse=True)
    for name, gain in sorted_gain[:10]:
        perm = perm_importances.get(name, float("nan"))
        family = next((f for f, names in FAMILY_MAP.items() if name in names), "other")
        print(f"  {name:<35} gain={gain:.4f}  perm={perm:.4f}  [{family}]")

    # Add importance rows to output
    for name, gain in gain_importances.items():
        perm = perm_importances.get(name, float("nan"))
        family = next((f for f, names in FAMILY_MAP.items() if name in names), "other")
        rows.append({
            "experiment": "feature_importance",
            "family_removed": family,
            "feature_name": name,
            "gain_importance": gain,
            "perm_importance": perm,
            "auroc": float("nan"),
            "delta": float("nan"),
        })

    # ── 6. Summary table ─────────────────────────────────────────────────────
    ablation_rows = [r for r in rows if r["experiment"].startswith("ablation_") or r["experiment"].startswith("full_")]
    print("\n=== ABLATION SUMMARY ===")
    header = f"{'experiment':<35} {'auroc':>8} {'delta':>8}"
    print(header)
    print("-" * len(header))
    for r in ablation_rows:
        print(f"{r['experiment']:<35} {r['auroc']:>8.4f} {r.get('delta', float('nan')):>+8.4f}")

    # ── 7. Plot (if matplotlib available) ─────────────────────────────────────
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Gain importance by family
            family_names = list(family_importance.keys())
            gain_sums = [family_importance[f]["gain_sum"] for f in family_names]
            axes[0].barh(family_names, gain_sums)
            axes[0].set_xlabel("Gain importance (sum)")
            axes[0].set_title("Feature family importance (GB gain)")

            # Top-15 individual features
            top_n = min(15, len(sorted_gain))
            feat_names_top = [n for n, _ in sorted_gain[:top_n]]
            feat_gains_top = [g for _, g in sorted_gain[:top_n]]
            axes[1].barh(feat_names_top[::-1], feat_gains_top[::-1])
            axes[1].set_xlabel("Gain importance")
            axes[1].set_title(f"Top-{top_n} individual features")

            plt.tight_layout()
            fig_path = Path(args.fig_path)
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(fig_path, dpi=100, bbox_inches="tight")
            plt.close()
            print(f"\n[save] figure → {fig_path}")
        except ImportError:
            print("[warn] matplotlib not available, skipping plot")

    # Write CSV
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys)
    # Ensure key columns appear first
    priority = ["experiment", "family_removed", "feature_name", "auroc", "delta",
                "gain_importance", "perm_importance"]
    fieldnames = [k for k in priority if k in all_keys] + [k for k in fieldnames if k not in priority]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[save] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Coding feature language audit V1")
    parser.add_argument("--feature-cache", default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    parser.add_argument("--fig-path", default=DEFAULT_FIG_PATH)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
