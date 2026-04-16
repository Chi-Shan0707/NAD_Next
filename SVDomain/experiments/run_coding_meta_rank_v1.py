#!/usr/bin/env python3
"""Coding Meta-Ranking V1 — comprehensive multi-source signal fusion for lcb_v5.

Assembles ALL available signal sources for 10,688 lcb_v5 solutions and finds
the optimal within-problem selector via leave-problem-out 5-fold CV.

Signal sources:
  feat_traj22     — token+traj 22 features (pos=1.0), LR score
  feat_full47     — all 47 features (pos=1.0), LR score
  layer_late      — n_active in layers 29-35 (late transformer layers), GBM
  layer_all       — n_active + wmax_mean across all 36 layers, GBM
  static_code     — 29 code structure features (ast_depth, n_conditionals...), GBM
  slot100_r1_ds   — DS-R1 slot100 extra 17 features, LR
  slot100_r1_qwen — Qwen3-4B slot100 extra 17 features, LR
  code_v2_base    — pre-computed code_v2_baseline scores (no CV needed)
  ssl_score       — domain_ssl coding bundle W_score projection
  ssl_medoid      — within-problem centroid distance in SSL latent space
  meta_all        — XGBoost meta-ranker on stacked CV scores (all sources)
  meta_best       — XGBoost meta-ranker on top-4 sources

Primary metric: hit@1 (5-fold leave-problem-out CV)
Secondary: pairwise_acc, AUROC

Usage (from repo root):
    python3 SVDomain/experiments/run_coding_meta_rank_v1.py
    python3 SVDomain/experiments/run_coding_meta_rank_v1.py --smoke
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.ops.earlystop_svd import FULL_FEATURE_NAMES, _auroc, _rank_transform_matrix
from scripts.run_earlystop_prefix10_svd_round1 import (
    EXTRACTION_POSITIONS, FEATURE_TO_INDEX, PREFIX_SAFE_FEATURE_FAMILY_MAP,
)

# ── paths ──────────────────────────────────────────────────────────────────────
FEAT_CACHE      = "results/cache/rule_and_family_baselines/cache_all_ref030_45392d82523f21bf.pkl"
STATIC_CACHE    = "results/cache/coding_v2_static_features.pkl"
LAYER_CACHE     = "results/cache/coding_v2_layer_features.pkl"
CODE_V2_CACHE   = "results/cache/coding_v2_code_v2_scores.pkl"
SLOT100_DS      = "results/cache/coding_hybrid_bridge/DS-R1__lcb_v5_slot100_extra.pkl"
SLOT100_QWEN    = "results/cache/coding_hybrid_bridge/Qwen3-4B__lcb_v5_slot100_extra.pkl"
SSL_BUNDLES     = "results/cache/domain_ssl/domain_ssl_bundles.pkl"
DEFAULT_OUT_CSV = "results/tables/coding_meta_rank_v1.csv"

FIXED_22_NAMES  = list(PREFIX_SAFE_FEATURE_FAMILY_MAP["token_plus_traj"])
FIXED_22_IDX    = [FEATURE_TO_INDEX[n] for n in FIXED_22_NAMES]
N_SPLITS        = 5
SEEDS           = (42, 101, 29)
LATE_LAYERS     = list(range(25, 36))   # layers 25-35 inclusive

# ── metric helpers ─────────────────────────────────────────────────────────────

def _auroc_safe(scores, labels):
    pos = labels == 1; neg = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    return _auroc(scores, labels)


def _pairwise_acc(scores, labels, groups):
    nc, nt = 0, 0
    for g in np.unique(groups):
        m = groups == g
        s, y = scores[m], labels[m]
        pos_i = np.where(y == 1)[0]; neg_i = np.where(y == 0)[0]
        for pi in pos_i:
            for ni in neg_i:
                nt += 1
                nc += (1.0 if s[pi] > s[ni] else 0.5 if s[pi] == s[ni] else 0.0)
    return float(nc) / float(nt) if nt > 0 else float("nan")


def _hit_at_1(scores, labels, groups):
    hits = n_probs = 0
    for g in np.unique(groups):
        m = groups == g
        y = labels[m]
        if y.sum() == 0:
            continue
        n_probs += 1
        if y[np.argmax(scores[m])] == 1:
            hits += 1
    return float(hits) / float(n_probs) if n_probs > 0 else float("nan")


def _mean_finite(vals):
    arr = np.array([v for v in vals if np.isfinite(v)])
    return float(np.mean(arr)) if arr.size > 0 else float("nan")


# ── data loading / alignment ───────────────────────────────────────────────────

def _load_base() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (tensor, labels, groups, sample_ids) from 47-feat feature cache."""
    with open(FEAT_CACHE, "rb") as fh:
        d = pickle.load(fh)
    item = [x for x in d["feature_store"] if x.get("domain") == "coding"][0]
    tensor = np.asarray(item["tensor"], dtype=np.float64)
    labels = np.asarray(item["labels"], dtype=np.int32)
    groups = np.asarray(item.get("group_keys",
        _rebuild_groups(item)), dtype=object)
    sids = np.asarray(item["sample_ids"], dtype=np.int64)
    return tensor, labels, groups, sids


def _rebuild_groups(item):
    offsets = item["problem_offsets"]
    pids = item["problem_ids"]
    gk = np.empty(item["tensor"].shape[0], dtype=object)
    for pi, pid in enumerate(pids):
        gk[offsets[pi]:offsets[pi+1]] = pid
    return gk


def _align(sids_ref: np.ndarray, sids_src: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Reorder X rows to match sids_ref ordering. Missing → zeros."""
    src_map = {int(s): i for i, s in enumerate(sids_src)}
    out = np.zeros((len(sids_ref), X.shape[1] if X.ndim > 1 else 1), dtype=np.float64)
    for j, s in enumerate(sids_ref):
        if int(s) in src_map:
            out[j] = X[src_map[int(s)]] if X.ndim > 1 else X[src_map[int(s)]]
    return out.squeeze() if X.ndim == 1 else out


# ── SSL helpers ────────────────────────────────────────────────────────────────

def _compute_ssl_signals(
    x_raw: np.ndarray, bundle: dict[str, Any], groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SSL W_score projection and within-problem centroid similarity."""
    B = bundle["B"]          # (44, r)
    scaler = bundle["scaler"]
    W_score = bundle["W_score"]  # (r, 1)

    # Build 44-dim raw+rank input from the first 22 features
    x_22 = x_raw[:, :22]
    x_rank_22 = _rank_transform_matrix(x_22)
    x_44 = np.concatenate([x_22, x_rank_22], axis=1)

    x_sc = scaler.transform(x_44)
    z = x_sc @ B            # (n, r)
    ssl_scores = (z @ W_score).squeeze()  # (n,)

    # Medoid: within-problem cosine similarity to group mean
    medoid_scores = np.zeros(len(groups), dtype=np.float64)
    for g in np.unique(groups):
        m = groups == g
        z_g = z[m]
        centroid = z_g.mean(axis=0)
        norm_z = np.linalg.norm(z_g, axis=1, keepdims=True) + 1e-10
        norm_c = np.linalg.norm(centroid) + 1e-10
        medoid_scores[m] = (z_g / norm_z) @ (centroid / norm_c)

    return ssl_scores.astype(np.float64), medoid_scores


# ── CV scorer ─────────────────────────────────────────────────────────────────

def _cv_score_lr(
    X: np.ndarray, labels: np.ndarray, groups: np.ndarray, C: float = 0.2
) -> np.ndarray:
    """Leave-problem-out 5-fold CV with logistic regression. Returns OOF scores."""
    n_g = len(np.unique(groups))
    folds = list(GroupKFold(n_splits=min(N_SPLITS, n_g)).split(X, groups=groups))
    oof = np.zeros(len(labels), dtype=np.float64)
    for tr_idx, te_idx in folds:
        y_tr = labels[tr_idx]
        if np.unique(y_tr).shape[0] < 2:
            continue
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr_idx])
        X_te = sc.transform(X[te_idx])
        fold_scores = np.zeros(len(te_idx), dtype=np.float64)
        for seed in SEEDS:
            clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs",
                                     class_weight="balanced", random_state=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X_tr, y_tr)
            fold_scores += clf.decision_function(X_te)
        oof[te_idx] = fold_scores / len(SEEDS)
    return oof


def _cv_score_gb(
    X: np.ndarray, labels: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    """Leave-problem-out 5-fold CV with GradientBoosting. Returns OOF scores."""
    n_g = len(np.unique(groups))
    folds = list(GroupKFold(n_splits=min(N_SPLITS, n_g)).split(X, groups=groups))
    oof = np.zeros(len(labels), dtype=np.float64)
    for tr_idx, te_idx in folds:
        y_tr = labels[tr_idx]
        if np.unique(y_tr).shape[0] < 2:
            continue
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr_idx])
        X_te = sc.transform(X[te_idx])
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=5, subsample=0.8, random_state=42
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr, y_tr)
        oof[te_idx] = clf.predict_proba(X_te)[:, 1]
    return oof


def _cv_score_xgb(
    X: np.ndarray, labels: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    """Leave-problem-out 5-fold CV with XGBoost. Returns OOF scores."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("[warn] XGBoost not available, falling back to GBM")
        return _cv_score_gb(X, labels, groups)

    n_g = len(np.unique(groups))
    folds = list(GroupKFold(n_splits=min(N_SPLITS, n_g)).split(X, groups=groups))
    oof = np.zeros(len(labels), dtype=np.float64)
    for tr_idx, te_idx in folds:
        y_tr = labels[tr_idx]
        if np.unique(y_tr).shape[0] < 2:
            continue
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr_idx])
        X_te = sc.transform(X[te_idx])
        fold_scores = np.zeros(len(te_idx), dtype=np.float64)
        for seed in SEEDS:
            clf = XGBClassifier(
                objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", max_depth=5, learning_rate=0.05,
                n_estimators=300, min_child_weight=5,
                subsample=0.8, colsample_bytree=0.8,
                random_state=seed, n_jobs=4, verbosity=0,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X_tr, y_tr)
            fold_scores += clf.predict_proba(X_te)[:, 1]
        oof[te_idx] = fold_scores / len(SEEDS)
    return oof


def _eval_oof(oof: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> dict:
    return {
        "auroc": _auroc_safe(oof, labels),
        "pairwise_acc": _pairwise_acc(oof, labels, groups),
        "hit@1": _hit_at_1(oof, labels, groups),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    print("[load] base feature cache …", flush=True)
    tensor, labels, groups, sids = _load_base()
    n_feat = tensor.shape[2]
    n_pos = tensor.shape[1]

    # Position index for 1.00
    pos_map = {float(p): i for i, p in enumerate(EXTRACTION_POSITIONS)}
    pos_100 = pos_map.get(1.0, n_pos - 1)
    pos_070 = pos_map.get(0.7, 8)
    x_raw_100 = tensor[:, pos_100, :]
    x_raw_070 = tensor[:, pos_070, :]

    # Load auxiliary caches
    print("[load] auxiliary caches …", flush=True)
    with open(STATIC_CACHE, "rb") as fh:
        sf = pickle.load(fh)
    with open(LAYER_CACHE, "rb") as fh:
        lf = pickle.load(fh)
    with open(CODE_V2_CACHE, "rb") as fh:
        cv2 = pickle.load(fh)
    with open(SLOT100_DS, "rb") as fh:
        s1 = pickle.load(fh)
    with open(SLOT100_QWEN, "rb") as fh:
        s2 = pickle.load(fh)

    X_static   = _align(sids, sf["sample_ids"], sf["X"])
    X_layer_na = _align(sids, lf["sample_ids"], lf["n_active"])
    X_layer_wm = _align(sids, lf["sample_ids"], lf["wmax_mean"])
    X_layer_wx = _align(sids, lf["sample_ids"], lf["wmax_max"])
    code_v2_scores = _align(sids, cv2["sample_ids"],
                            cv2["scores"].reshape(-1, 1)).squeeze()
    X_slot_ds   = _align(sids, s1["sample_ids"], s1["X"])
    X_slot_qwen = _align(sids, s2["sample_ids"], s2["X"])

    # SSL signals
    ssl_score_oof = np.zeros(len(labels), dtype=np.float64)
    ssl_medoid_oof = np.zeros(len(labels), dtype=np.float64)
    ssl_loaded = False
    if Path(SSL_BUNDLES).exists():
        with open(SSL_BUNDLES, "rb") as fh:
            bundles = pickle.load(fh)
        coding_bundles = bundles.get("domain_bundles", {}).get("coding", {})
        best_r = max(coding_bundles.keys()) if coding_bundles else None
        if best_r is not None:
            bundle = coding_bundles[best_r]
            print(f"[ssl] computing coding domain_ssl signals r={best_r}", flush=True)
            ssl_score_raw, ssl_medoid_raw = _compute_ssl_signals(x_raw_100, bundle, groups)
            ssl_score_oof = ssl_score_raw
            ssl_medoid_oof = ssl_medoid_raw
            ssl_loaded = True

    # Smoke: subsample to 20% of problems
    if args.smoke:
        rng = np.random.RandomState(42)
        ugroups = np.unique(groups)
        chosen = set(rng.choice(ugroups, size=max(20, len(ugroups)//5), replace=False).tolist())
        mask = np.array([g in chosen for g in groups.tolist()])
        def _sm(arr):
            return arr[mask] if arr.ndim == 1 else arr[mask]
        labels, groups, sids = _sm(labels), _sm(groups), _sm(sids)
        x_raw_100, x_raw_070 = _sm(x_raw_100), _sm(x_raw_070)
        X_static, X_layer_na, X_layer_wm, X_layer_wx = _sm(X_static), _sm(X_layer_na), _sm(X_layer_wm), _sm(X_layer_wx)
        code_v2_scores = _sm(code_v2_scores)
        X_slot_ds, X_slot_qwen = _sm(X_slot_ds), _sm(X_slot_qwen)
        ssl_score_oof, ssl_medoid_oof = _sm(ssl_score_oof), _sm(ssl_medoid_oof)
        print(f"[smoke] {mask.sum()} samples / {len(chosen)} problems")

    n_groups = len(np.unique(groups))
    print(f"[info] {len(labels)} samples / {n_groups} problems  pos={labels.sum()}")

    oof_store: dict[str, np.ndarray] = {}
    rows = []

    def _run_signal(name: str, X: np.ndarray, scorer: str = "lr") -> np.ndarray:
        print(f"[cv] {name} ({scorer}, {X.shape[1] if X.ndim>1 else 1} feat) …", flush=True)
        if scorer == "gb":
            oof = _cv_score_gb(X if X.ndim > 1 else X.reshape(-1,1), labels, groups)
        elif scorer == "xgb":
            oof = _cv_score_xgb(X if X.ndim > 1 else X.reshape(-1,1), labels, groups)
        else:
            oof = _cv_score_lr(X if X.ndim > 1 else X.reshape(-1,1), labels, groups)
        m = _eval_oof(oof, labels, groups)
        oof_store[name] = oof
        rows.append({"signal": name, "n_feat": X.shape[1] if X.ndim>1 else 1, "scorer": scorer, **m})
        print(f"  auroc={m['auroc']:.4f}  pairwise_acc={m['pairwise_acc']:.4f}  hit@1={m['hit@1']:.4f}")
        return oof

    # ── Individual signals ──────────────────────────────────────────────────
    # 1. token+traj 22 features at pos=1.0
    _run_signal("feat_traj22_100",     x_raw_100[:, FIXED_22_IDX],      "lr")
    _run_signal("feat_traj22_070",     x_raw_070[:, FIXED_22_IDX],      "lr")

    # 2. All 47 features at pos=1.0
    _run_signal("feat_full47_100",     x_raw_100,                        "gb")

    # 3. Layer n_active: late layers only (25-35)
    late_idx = [l for l in LATE_LAYERS if l < X_layer_na.shape[1]]
    X_late = X_layer_na[:, late_idx]
    _run_signal("layer_late_nactive",  X_late,                           "gb")

    # 4. Layer n_active + wmax: all 36 layers
    X_layer_all = np.concatenate([X_layer_na, X_layer_wm, X_layer_wx], axis=1)
    _run_signal("layer_all_108",       X_layer_all,                      "gb")

    # 5. Static code features
    _run_signal("static_code_29",      X_static,                         "gb")

    # 6. DS-R1 slot100 extra
    _run_signal("slot100_ds_r1",       X_slot_ds,                        "lr")

    # 7. Qwen3-4B slot100 extra
    _run_signal("slot100_qwen3",       X_slot_qwen,                      "lr")

    # 8. Cross-model combined slot100
    X_cross = np.concatenate([X_slot_ds, X_slot_qwen], axis=1)
    _run_signal("slot100_cross_model", X_cross,                          "lr")

    # 9. code_v2_baseline (no CV, use directly)
    m_cv2 = _eval_oof(code_v2_scores, labels, groups)
    oof_store["code_v2_base"] = code_v2_scores
    rows.append({"signal": "code_v2_base", "n_feat": 1, "scorer": "precomp", **m_cv2})
    print(f"[precomp] code_v2_base  auroc={m_cv2['auroc']:.4f}  hit@1={m_cv2['hit@1']:.4f}")

    # 10. SSL signals
    if ssl_loaded:
        m_ssl = _eval_oof(ssl_score_oof, labels, groups)
        oof_store["ssl_score"] = ssl_score_oof
        rows.append({"signal": "ssl_score", "n_feat": 1, "scorer": "ssl", **m_ssl})
        print(f"[ssl] ssl_score  auroc={m_ssl['auroc']:.4f}  hit@1={m_ssl['hit@1']:.4f}")

        m_med = _eval_oof(ssl_medoid_oof, labels, groups)
        oof_store["ssl_medoid"] = ssl_medoid_oof
        rows.append({"signal": "ssl_medoid", "n_feat": 1, "scorer": "ssl", **m_med})
        print(f"[ssl] ssl_medoid  auroc={m_med['auroc']:.4f}  hit@1={m_med['hit@1']:.4f}")

    # ── Meta-ranking: stack OOF scores and re-rank ──────────────────────────
    # Meta-all: stack all signal OOF scores
    oof_keys_all = [k for k in oof_store]
    X_meta_all = np.stack([oof_store[k] for k in oof_keys_all], axis=1)
    print(f"\n[meta] training meta-XGB on {X_meta_all.shape[1]} stacked signals …", flush=True)
    _run_signal("meta_xgb_all", X_meta_all, "xgb")

    # Meta-best: top signals by hit@1
    sorted_rows = sorted([r for r in rows if r["signal"] != "meta_xgb_all"],
                         key=lambda r: r.get("hit@1", 0), reverse=True)
    top4_names = [r["signal"] for r in sorted_rows[:4]]
    print(f"[meta] top-4 signals: {top4_names}")
    X_meta_best = np.stack([oof_store[k] for k in top4_names if k in oof_store], axis=1)
    _run_signal("meta_xgb_top4", X_meta_best, "xgb")

    # Meta: late_layer + static + code_v2 (hypothesis: independent signals)
    best3 = ["layer_late_nactive", "static_code_29", "code_v2_base"]
    available3 = [k for k in best3 if k in oof_store]
    if len(available3) >= 2:
        X_meta_best3 = np.stack([oof_store[k] for k in available3], axis=1)
        _run_signal("meta_layer_static_cv2", X_meta_best3, "xgb")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n=== SUMMARY (sorted by hit@1) ===")
    sorted_rows_all = sorted(rows, key=lambda r: r.get("hit@1", 0), reverse=True)
    header = f"{'signal':<28} {'scorer':<8} {'n_feat':>6} {'auroc':>8} {'pw_acc':>8} {'hit@1':>7}"
    print(header)
    print("-" * len(header))
    for r in sorted_rows_all:
        print(
            f"{r['signal']:<28} {r['scorer']:<8} {r['n_feat']:>6} "
            f"{r.get('auroc',float('nan')):>8.4f} "
            f"{r.get('pairwise_acc',float('nan')):>8.4f} "
            f"{r.get('hit@1',float('nan')):>7.4f}"
        )

    print(f"\nBaseline (random): ~0.5869")
    print(f"Baseline (code_v2): {m_cv2['hit@1']:.4f}")
    print(f"Oracle hit@1: ~0.8263")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["signal", "scorer", "n_feat", "auroc", "pairwise_acc", "hit@1"]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[save] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
