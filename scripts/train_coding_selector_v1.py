#!/usr/bin/env python3
"""Train and save the coding selector V1 model (allpos-XGB).

Model: XGBoost ensemble on all-positions × all-features tensor.
Input: (n_samples, 12, 47) → flatten → (n_samples, 564)
Output: per-sample correctness probability
Training: full-fit on ALL 167 labeled lcb_v5 problems, 64-seed ensemble.

CV estimate (5-fold leave-problem-out): hit@1 = 0.746 ± 0.013
Baseline (random selection):           hit@1 = 0.587
Oracle (always picks correct):         hit@1 = 0.826

Usage (from repo root):
    python3 scripts/train_coding_selector_v1.py
    python3 scripts/train_coding_selector_v1.py --cv-only  # CV eval, no save
    python3 scripts/train_coding_selector_v1.py --n-seeds 8  # faster, less stable
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_earlystop_prefix10_svd_round1 import EXTRACTION_POSITIONS

# ── paths ──────────────────────────────────────────────────────────────────────
FEATURE_CACHE   = "results/cache/rule_and_family_baselines/cache_all_ref030_45392d82523f21bf.pkl"
MODEL_OUT_DIR   = "models/ml_selectors"
MODEL_OUT_PATH  = "models/ml_selectors/coding_allpos_xgb_v1.pkl"
CV_RESULT_PATH  = "results/tables/coding_selector_v1_cv.json"
DEFAULT_N_SEEDS = 64

# Optimal XGB hyperparameters from extensive 64-seed sweep on 167-problem CV
XGB_PARAMS = dict(
    objective="binary:logistic",
    tree_method="hist",
    max_depth=5,
    learning_rate=0.05,
    n_estimators=300,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    verbosity=0,
    n_jobs=1,         # parallelism handled at seed level
)


def _load_coding_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(FEATURE_CACHE, "rb") as fh:
        d = pickle.load(fh)
    item = [x for x in d["feature_store"] if x.get("domain") == "coding"][0]
    tensor = np.asarray(item["tensor"], dtype=np.float64)
    labels = np.asarray(item["labels"], dtype=np.int32)
    gk = item.get("group_keys")
    if gk is None:
        offsets = item["problem_offsets"]
        pids = item["problem_ids"]
        gk = np.empty(tensor.shape[0], dtype=object)
        for pi, pid in enumerate(pids):
            gk[offsets[pi]:offsets[pi + 1]] = pid
    groups = np.asarray(gk, dtype=object)
    sids = np.asarray(item["sample_ids"], dtype=np.int64)
    return tensor, labels, groups, sids


def _hit_at_1(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> float:
    h = n = 0
    for g in np.unique(groups):
        m = groups == g
        y = labels[m]
        if y.sum() == 0:
            continue
        n += 1
        if y[np.argmax(scores[m])] == 1:
            h += 1
    return float(h) / float(n) if n > 0 else float("nan")


def _pairwise_acc(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> float:
    nc = nt = 0
    for g in np.unique(groups):
        m = groups == g
        s, y = scores[m], labels[m]
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        for pi in pos:
            for ni in neg:
                nt += 1
                nc += 1.0 if s[pi] > s[ni] else 0.5 if s[pi] == s[ni] else 0.0
    return float(nc) / float(nt) if nt > 0 else float("nan")


def _train_single_seed(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, seed: int
) -> np.ndarray:
    clf = XGBClassifier(**XGB_PARAMS, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def run_cv(
    tensor: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_seeds: int = DEFAULT_N_SEEDS,
    n_workers: int = 16,
) -> dict[str, Any]:
    """5-fold leave-problem-out CV with parallel seed ensemble."""
    X_flat = tensor.reshape(len(labels), -1)
    folds = list(GroupKFold(n_splits=5).split(X_flat, groups=groups))
    oof = np.zeros(len(labels), dtype=np.float64)

    print(f"[cv] 5-fold GroupKFold × {n_seeds} seeds × {n_workers} threads", flush=True)
    for fold_i, (tr_idx, te_idx) in enumerate(folds):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_flat[tr_idx])
        X_te = sc.transform(X_flat[te_idx])
        y_tr = labels[tr_idx]
        print(f"  fold {fold_i+1}/5 (n_tr={len(tr_idx)}, n_te={len(te_idx)}) …", flush=True)

        fold_scores = np.zeros(len(te_idx), dtype=np.float64)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_train_single_seed, X_tr, y_tr, X_te, s): s
                       for s in range(1, n_seeds + 1)}
            for fut in as_completed(futures):
                fold_scores += fut.result()
        oof[te_idx] = fold_scores / n_seeds

    h1 = _hit_at_1(oof, labels, groups)
    pa = _pairwise_acc(oof, labels, groups)
    print(f"\n[cv] hit@1={h1:.4f}  pairwise_acc={pa:.4f}")
    return {"hit@1": h1, "pairwise_acc": pa, "n_seeds": n_seeds, "n_problems": int(len(np.unique(groups)))}


def train_full_model(
    tensor: np.ndarray,
    labels: np.ndarray,
    n_seeds: int = DEFAULT_N_SEEDS,
    n_workers: int = 16,
) -> dict[str, Any]:
    """Full-fit: train on ALL data, return ensemble of scalers + models."""
    X_flat = tensor.reshape(len(labels), -1)
    sc = StandardScaler()
    X_sc = sc.fit_transform(X_flat)
    y = labels.astype(np.int32)

    print(f"[fullfit] training {n_seeds}-seed ensemble on {len(labels)} samples …", flush=True)
    models = []

    def _fit_seed(seed: int) -> XGBClassifier:
        clf = XGBClassifier(**XGB_PARAMS, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_sc, y)
        return clf

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_fit_seed, s): s for s in range(1, n_seeds + 1)}
        for fut in as_completed(futures):
            models.append(fut.result())
    print(f"[fullfit] done — {len(models)} models trained")

    return {
        "scaler": sc,
        "models": models,
        "n_seeds": n_seeds,
        "feature_shape": (12, 47),
        "input_shape_flat": X_flat.shape[1],
        "extraction_positions": list(EXTRACTION_POSITIONS),
        "model_type": "allpos_xgb_v1",
        "description": "XGBoost on all-positions × all-features (12×47=564) token/traj tensor",
    }


def score_new_solutions(bundle: dict[str, Any], tensor: np.ndarray) -> np.ndarray:
    """Score new solutions. tensor shape: (n_solutions, 12, 47)."""
    n = tensor.shape[0]
    X_flat = tensor.reshape(n, -1).astype(np.float64)
    X_sc = bundle["scaler"].transform(X_flat)
    scores = np.zeros(n, dtype=np.float64)
    for clf in bundle["models"]:
        scores += clf.predict_proba(X_sc)[:, 1]
    return scores / len(bundle["models"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train coding selector V1")
    parser.add_argument("--cv-only", action="store_true", help="Only run CV, do not save model")
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--n-workers", type=int, default=16)
    parser.add_argument("--model-out", default=MODEL_OUT_PATH)
    parser.add_argument("--cv-out", default=CV_RESULT_PATH)
    args = parser.parse_args()

    print(f"[load] {FEATURE_CACHE}", flush=True)
    tensor, labels, groups, sids = _load_coding_data()
    print(f"[info] tensor {tensor.shape}, labels pos={labels.sum()} neg={(1-labels).sum()}, "
          f"problems={len(np.unique(groups))}")

    # CV evaluation
    cv_results = run_cv(tensor, labels, groups,
                        n_seeds=args.n_seeds, n_workers=args.n_workers)
    cv_out = Path(args.cv_out)
    cv_out.parent.mkdir(parents=True, exist_ok=True)
    with cv_out.open("w") as fh:
        json.dump(cv_results, fh, indent=2)
    print(f"[save] CV results → {cv_out}")

    if args.cv_only:
        return

    # Full-fit model
    bundle = train_full_model(tensor, labels,
                               n_seeds=args.n_seeds, n_workers=args.n_workers)

    # Verify: apply full model to training data
    full_scores = score_new_solutions(bundle, tensor)
    full_h1 = _hit_at_1(full_scores, labels, groups)
    print(f"[verify] full-fit training hit@1={full_h1:.4f} (expected ~1.0 — sanity check)")

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    with model_out.open("wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[save] model → {model_out}  ({model_out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
