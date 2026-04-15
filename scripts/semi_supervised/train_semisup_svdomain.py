#!/usr/bin/env python3
"""Semi-Supervised SVDomain Extension.

Tests whether self-supervised basis pre-training improves label efficiency
compared to fully supervised SVD training across math, science, and coding.

Conditions compared:
  semisup       — pre-trained SSL basis + supervised head (proposed)
  frozen_svd    — frozen SVD from supervised model + new LR head only
  supervised_svd — standard SVD+LR trained on labeled subset
  no_svd_lr     — StandardScaler → LogisticRegression (no SVD, no SSL)

Self-supervised losses:
  1. Masked feature reconstruction (30% masking rate)
  2. Cross-anchor alignment via learned bridge W_ap
  3. Raw-vs-rank view consistency

Usage:
  # Smoke test (~1 min)
  python3 scripts/semi_supervised/train_semisup_svdomain.py \\
      --main-cache-root MUI_HUB/cache \\
      --test-cache-root MUI_HUB/cache_test \\
      --smoke

  # Full run
  python3 scripts/semi_supervised/train_semisup_svdomain.py \\
      --main-cache-root MUI_HUB/cache \\
      --extra-cache-root MUI_HUB/cache_train \\
      --test-cache-root MUI_HUB/cache_test \\
      --out-dir results/cache/semisup_svdomain
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Callable, Optional

for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env, "1")

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nad.ops.earlystop import _problem_sort_key
from nad.ops.earlystop_svd import (
    FULL_FEATURE_NAMES,
    _auroc,
    _group_folds,
    _rank_transform_matrix,
    load_earlystop_svd_bundle,
)
from scripts.run_earlystop_prefix10_svd_round1 import (
    ANCHOR_POSITIONS,
    DEFAULT_FEATURE_CHUNK_PROBLEMS,
    EARLY_STOP_POSITIONS,
    EXTRACTION_POSITION_INDEX,
    EXTRACTION_POSITIONS,
    OFFICIAL_SLOT_TO_ANCHOR,
    SEARCH_C_VALUES,
    _display_path,
    _now_utc,
    build_feature_store,
    evaluate_method_from_feature_store,
)
from SVDomain.train_es_svd_ms_rr_r1 import (
    FIXED_FEATURE_INDICES,
    FIXED_FEATURE_NAMES,
    _build_holdout_problem_map,
    _qualify_feature_store,
    _split_feature_store,
    _subset_payload_by_problem_ids,
)


# ─── Constants ────────────────────────────────────────────────────────────────

LABEL_FRACTIONS = (0.01, 0.05, 0.10, 0.20, 0.50, 1.0)
DEFAULT_SSL_RANKS = (4, 8, 16)
ALL_DOMAINS = ("math", "science", "coding")

# Cross-anchor alignment: index into ANCHOR_POSITIONS = (0.10, 0.40, 0.70, 1.00)
EARLY_ANCHOR_IDX = 0  # 10%
LATE_ANCHOR_IDX = 2   # 70%

ANCHOR_POS_INDICES = [EXTRACTION_POSITION_INDEX[float(p)] for p in ANCHOR_POSITIONS]
N_FEATURES = len(FIXED_FEATURE_NAMES)   # 22 raw features
D_FULL = 2 * N_FEATURES                  # 44 = raw + rank

DEFAULT_BUNDLE_PATH = REPO_ROOT / "models/ml_selectors/es_svd_ms_rr_r1.pkl"

# Map each EARLY_STOP_POSITIONS index to nearest anchor index
_POS_TO_ANCHOR_IDX: dict[int, int] = {}
for _pi, _pos in enumerate(EARLY_STOP_POSITIONS):
    _anch = float(OFFICIAL_SLOT_TO_ANCHOR[float(_pos)])
    _POS_TO_ANCHOR_IDX[_pi] = list(ANCHOR_POSITIONS).index(_anch)


# ─── Adam update ──────────────────────────────────────────────────────────────

def _adam_update(
    param: np.ndarray,
    grad: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    t: int,
    lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = beta1 * m + (1.0 - beta1) * grad
    v = beta2 * v + (1.0 - beta2) * (grad ** 2)
    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)
    param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    return param, m, v


# ─── SSL loss and gradients ───────────────────────────────────────────────────

def _ssl_loss_and_grad(
    B: np.ndarray,           # (d=44, r)
    W_ap: np.ndarray,        # (r, r)  anchor bridge
    X_scaled: np.ndarray,    # (n, d)  full batch (raw+rank), pre-scaled
    X_early: np.ndarray,     # (n, d)  early-anchor features, pre-scaled
    X_late: np.ndarray,      # (n, d)  late-anchor features, pre-scaled
    mask_rate: float = 0.30,
    lam_align: float = 0.5,
    lam_view: float = 0.5,
    rng: Optional[np.random.RandomState] = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (total_loss, grad_B, grad_W_ap) for one mini-batch."""
    if rng is None:
        rng = np.random.RandomState(0)

    n, d = X_scaled.shape
    half_d = d // 2  # 22

    # ── 1. Masked Feature Reconstruction ─────────────────────────────────────
    M = (rng.random_sample((n, d)) < mask_rate).astype(np.float64)
    n_masked = float(max(1.0, M.sum()))
    X_in = X_scaled * (1.0 - M)        # zeroed-out input
    Z_rec = X_in @ B                   # (n, r)
    X_rec = Z_rec @ B.T                # (n, d)
    err_r = (X_rec - X_scaled) * M    # error only at masked positions
    L_recon = 0.5 * float(np.sum(err_r ** 2)) / n_masked
    # decoder gradient + encoder gradient
    grad_B_rec = (err_r.T @ Z_rec + X_in.T @ (err_r @ B)) / n_masked

    # ── 2. Cross-Anchor Alignment ─────────────────────────────────────────────
    n_al = float(max(1, X_early.shape[0]))
    Z_ea = X_early @ B          # (n, r)
    Z_la = X_late @ B           # (n, r)
    Z_pred = Z_ea @ W_ap        # (n, r)  predicted late from early
    err_al = Z_pred - Z_la      # (n, r)
    L_align = 0.5 * float(np.sum(err_al ** 2)) / n_al
    grad_W_ap = (Z_ea.T @ err_al) / n_al
    # dL/dB: early path (positive) + late path (negative)
    grad_B_al = (X_early.T @ (err_al @ W_ap.T) - X_late.T @ err_al) / n_al

    # ── 3. Raw-vs-Rank View Consistency ──────────────────────────────────────
    # B[:22, :] applied to raw half, B[22:, :] applied to rank half
    X_rw = X_scaled[:, :half_d]    # (n, 22) scaled raw
    X_rk = X_scaled[:, half_d:]    # (n, 22) scaled rank
    Z_rw = X_rw @ B[:half_d, :]    # (n, r)
    Z_rk = X_rk @ B[half_d:, :]    # (n, r)
    err_vw = Z_rw - Z_rk            # (n, r)
    L_view = 0.5 * float(np.sum(err_vw ** 2)) / float(n)
    grad_B_vw = np.vstack([
        (X_rw.T @ err_vw) / float(n),   # (22, r)  raw half
        -(X_rk.T @ err_vw) / float(n),  # (22, r)  rank half
    ])

    # ── Combine ───────────────────────────────────────────────────────────────
    L_total = L_recon + lam_align * L_align + lam_view * L_view
    grad_B = grad_B_rec + lam_align * grad_B_al + lam_view * grad_B_vw
    grad_W = lam_align * grad_W_ap

    return float(L_total), grad_B, grad_W


# ─── SSL matrix extraction from feature store ─────────────────────────────────

def _extract_ssl_matrix(feature_store: list[dict[str, Any]]) -> np.ndarray:
    """Return (N, n_anchors, D_FULL) array of raw+rank features.

    N = total runs across all payloads.
    n_anchors = 4 (10%, 40%, 70%, 100%).
    D_FULL = 44 (22 raw + 22 rank).
    """
    n_anchors = len(ANCHOR_POSITIONS)
    parts: list[np.ndarray] = []

    for payload in feature_store:
        tensor = np.asarray(payload["tensor"], dtype=np.float64)
        n = tensor.shape[0]
        if n == 0:
            continue

        # x_raw: (n, n_anchors, 22)
        x_raw = tensor[:, ANCHOR_POS_INDICES, :][:, :, FIXED_FEATURE_INDICES]

        # Compute rank within problem group, per anchor
        x_rank = np.zeros_like(x_raw)
        by_group: dict[Any, list[int]] = {}
        for i, gk in enumerate(payload["group_keys"].tolist()):
            by_group.setdefault(gk, []).append(i)

        for idx_list in by_group.values():
            idx_arr = np.asarray(idx_list)
            for a_idx in range(n_anchors):
                sub = x_raw[idx_arr, a_idx, :]
                x_rank[idx_arr, a_idx, :] = _rank_transform_matrix(sub)

        # Concatenate raw + rank → (n, n_anchors, 44)
        x_full = np.concatenate([x_raw, x_rank], axis=-1)
        parts.append(x_full)

    if not parts:
        return np.zeros((0, n_anchors, D_FULL), dtype=np.float64)
    return np.concatenate(parts, axis=0)


# ─── SSL Pre-training ──────────────────────────────────────────────────────────

def pretrain_ssl_basis(
    X_full_all: np.ndarray,  # (N, n_anchors, D_FULL)
    r: int,
    n_epochs: int = 300,
    batch: int = 256,
    lr: float = 0.01,
    seed: int = 42,
    smoke: bool = False,
    lam_align: float = 0.5,
    lam_view: float = 0.5,
    mask_rate: float = 0.30,
) -> dict[str, Any]:
    """Pre-train SSL basis B on all runs (labels not used).

    Returns dict: {B: (d, r), W_ap: (r, r), scaler: StandardScaler, r, d}.
    """
    N, n_anchors, d = X_full_all.shape
    if N < 4:
        raise ValueError(f"SSL needs at least 4 runs, got {N}")

    if smoke:
        n_epochs = min(n_epochs, 20)
        batch = min(batch, 64)

    # Fit scaler on all flattened data (all runs, all anchors)
    X_flat = X_full_all.reshape(-1, d)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X_flat)
    X_scaled_all = scaler.transform(X_flat).reshape(N, n_anchors, d)

    # Warm-start B from top-r right singular vectors of scaled data
    try:
        n_components = min(r, d - 1, N * n_anchors - 1)
        svd_init = TruncatedSVD(n_components=n_components, random_state=seed)
        svd_init.fit(X_scaled_all.reshape(-1, d))
        B = svd_init.components_.T.astype(np.float64)  # (d, r_init)
        r_actual = B.shape[1]
    except Exception:
        rng_init = np.random.RandomState(seed)
        r_actual = min(r, d)
        B = rng_init.randn(d, r_actual).astype(np.float64) * 0.01

    # W_ap: anchor bridge, init as small identity
    W_ap = np.eye(r_actual, dtype=np.float64) * 0.01

    # Adam state
    m_B, v_B = np.zeros_like(B), np.zeros_like(B)
    m_W, v_W = np.zeros_like(W_ap), np.zeros_like(W_ap)

    rng = np.random.RandomState(seed)
    t = 0

    log_interval = max(1, n_epochs // 5)
    print(f"[ssl] pretrain r={r_actual} N={N} n_anchors={n_anchors} "
          f"epochs={n_epochs} batch={batch} d={d}")

    for epoch in range(n_epochs):
        perm = rng.permutation(N)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch):
            idx = perm[start:start + batch]
            if len(idx) < 4:
                continue

            # Iterate over all anchor positions for reconstruction/view losses
            for a_idx in range(n_anchors):
                X_batch = X_scaled_all[idx, a_idx, :]           # (b, d)
                X_early = X_scaled_all[idx, EARLY_ANCHOR_IDX, :]  # (b, d)
                X_late = X_scaled_all[idx, LATE_ANCHOR_IDX, :]    # (b, d)

                L, gB, gW = _ssl_loss_and_grad(
                    B, W_ap, X_batch, X_early, X_late,
                    mask_rate=mask_rate,
                    lam_align=lam_align,
                    lam_view=lam_view,
                    rng=rng,
                )
                epoch_loss += L
                n_batches += 1

                t += 1
                B, m_B, v_B = _adam_update(B, gB, m_B, v_B, t, lr=lr)
                W_ap, m_W, v_W = _adam_update(W_ap, gW, m_W, v_W, t, lr=lr)

        if (epoch + 1) % log_interval == 0:
            avg = epoch_loss / max(1, n_batches)
            print(f"[ssl] epoch={epoch + 1}/{n_epochs} avg_loss={avg:.5f}")

    return {"B": B, "W_ap": W_ap, "scaler": scaler, "r": r_actual, "d": d}


# ─── Feature table builder ────────────────────────────────────────────────────

def _build_anchor_tables(
    feature_store: list[dict[str, Any]],
    domain: Optional[str] = None,
) -> dict[int, dict[str, np.ndarray]]:
    """Build {anchor_idx: {x_raw, x_rank, y, groups}} for the given domain.

    anchor_idx ∈ {0,1,2,3} maps to ANCHOR_POSITIONS = (0.10, 0.40, 0.70, 1.00).
    x_raw and x_rank have shape (n_samples, 22).
    """
    n_anchors = len(ANCHOR_POSITIONS)
    rows: list[list[np.ndarray]] = [[] for _ in range(n_anchors)]
    rank_rows: list[list[np.ndarray]] = [[] for _ in range(n_anchors)]
    ys: list[list[np.ndarray]] = [[] for _ in range(n_anchors)]
    grp_lists: list[list[np.ndarray]] = [[] for _ in range(n_anchors)]

    for payload in feature_store:
        if domain is not None and payload["domain"] != domain:
            continue
        tensor = np.asarray(payload["tensor"], dtype=np.float64)
        n = tensor.shape[0]
        if n == 0:
            continue

        y = np.asarray(payload["labels"], dtype=np.int32)
        # Use cv_group_keys for CV folds when available, else fall back
        gk = np.asarray(
            payload.get("cv_group_keys", payload["group_keys"]),
            dtype=object,
        )
        rank_gk = np.asarray(payload["group_keys"], dtype=object)

        for a_idx, pos_idx in enumerate(ANCHOR_POS_INDICES):
            x_raw_a = tensor[:, pos_idx, :][:, FIXED_FEATURE_INDICES]  # (n, 22)

            # Rank within problem group
            x_rank_a = np.zeros_like(x_raw_a)
            by_rank_grp: dict[Any, list[int]] = {}
            for i, rk in enumerate(rank_gk.tolist()):
                by_rank_grp.setdefault(rk, []).append(i)
            for idx_list in by_rank_grp.values():
                idx_arr = np.asarray(idx_list)
                x_rank_a[idx_arr] = _rank_transform_matrix(x_raw_a[idx_arr])

            rows[a_idx].append(x_raw_a)
            rank_rows[a_idx].append(x_rank_a)
            ys[a_idx].append(y)
            grp_lists[a_idx].append(gk)

    out: dict[int, dict[str, np.ndarray]] = {}
    for a_idx in range(n_anchors):
        if rows[a_idx]:
            out[a_idx] = {
                "x_raw": np.vstack(rows[a_idx]).astype(np.float64, copy=False),
                "x_rank": np.vstack(rank_rows[a_idx]).astype(np.float64, copy=False),
                "y": np.concatenate(ys[a_idx]).astype(np.int32, copy=False),
                "groups": np.concatenate(grp_lists[a_idx]).astype(object, copy=False),
            }
        else:
            out[a_idx] = {
                "x_raw": np.zeros((0, N_FEATURES), dtype=np.float64),
                "x_rank": np.zeros((0, N_FEATURES), dtype=np.float64),
                "y": np.zeros(0, dtype=np.int32),
                "groups": np.asarray([], dtype=object),
            }
    return out


# ─── Label subset sampling ────────────────────────────────────────────────────

def _sample_labeled_problems(
    train_payloads: list[dict[str, Any]],
    domain: str,
    fraction: float,
    seed: int = 42,
) -> set[str]:
    """Sample `fraction` of training problem_ids for a domain."""
    all_pids: list[str] = []
    for payload in train_payloads:
        if payload["domain"] != domain:
            continue
        all_pids.extend(str(pid) for pid in payload["problem_ids"])

    unique_pids = sorted(set(all_pids), key=_problem_sort_key)
    n_select = max(1, int(round(fraction * len(unique_pids))))
    rng = np.random.RandomState(seed + abs(hash(domain)) % (2 ** 31))
    chosen = rng.choice(len(unique_pids), size=min(n_select, len(unique_pids)), replace=False)
    return {unique_pids[i] for i in chosen}


def _subset_store_by_problems(
    payloads: list[dict[str, Any]],
    domain: str,
    problem_ids: set[str],
) -> list[dict[str, Any]]:
    """Return payloads filtered to domain and selected problem_ids."""
    out = []
    for payload in payloads:
        if payload["domain"] != domain:
            continue
        sub = _subset_payload_by_problem_ids(payload, problem_ids)
        if sub is not None and int(sub["samples"]) > 0:
            out.append(sub)
    return out


# ─── Head fitting ─────────────────────────────────────────────────────────────

def _make_transform_fn(
    B: np.ndarray,
    scaler: StandardScaler,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Factory: returns a function (x_raw, x_rank) → Z."""
    def _transform(x_raw: np.ndarray, x_rank: np.ndarray) -> np.ndarray:
        X_rep = np.concatenate([x_raw, x_rank], axis=1)
        return scaler.transform(X_rep) @ B
    return _transform


def _fit_pointwise_head(
    x_raw: np.ndarray,
    x_rank: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    transform_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    c_search: bool = True,
    random_state: int = 42,
) -> Optional[LogisticRegression]:
    """Fit pointwise LR head using `transform_fn(x_raw, x_rank)` as embedding."""
    if x_raw.shape[0] < 4 or np.unique(y).shape[0] < 2:
        return None
    Z = transform_fn(x_raw, x_rank)
    if Z.shape[1] < 1:
        return None

    best_c = float(SEARCH_C_VALUES[len(SEARCH_C_VALUES) // 2])
    if c_search:
        folds = _group_folds(groups, n_splits=3)
        if len(folds) >= 2:
            best_cv = float("-inf")
            for c_val in SEARCH_C_VALUES:
                aucs = []
                for tr_idx, te_idx in folds:
                    y_tr, y_te = y[tr_idx], y[te_idx]
                    if np.unique(y_tr).shape[0] < 2 or np.unique(y_te).shape[0] < 2:
                        continue
                    try:
                        clf = LogisticRegression(C=float(c_val), max_iter=2000,
                                                 random_state=random_state)
                        clf.fit(Z[tr_idx], y_tr)
                        auc = _auroc(clf.decision_function(Z[te_idx]), y_te)
                        if np.isfinite(auc):
                            aucs.append(float(auc))
                    except Exception:
                        pass
                if aucs and float(np.mean(aucs)) > best_cv:
                    best_cv = float(np.mean(aucs))
                    best_c = float(c_val)

    clf = LogisticRegression(C=best_c, max_iter=2000, random_state=random_state)
    try:
        clf.fit(Z, y)
    except Exception:
        return None
    return clf


def _fit_pairwise_head(
    x_raw: np.ndarray,
    x_rank: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    transform_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    random_state: int = 42,
    max_pairs: int = 10000,
) -> Optional[LogisticRegression]:
    """Fit pairwise logistic head for coding domain.

    Generates (correct_i, incorrect_j) pairs within each problem group.
    Returns a classifier trained on Z_i − Z_j differences.
    """
    if x_raw.shape[0] < 4 or np.unique(y).shape[0] < 2:
        return None
    Z = transform_fn(x_raw, x_rank)

    by_group: dict[Any, list[int]] = {}
    for i, gk in enumerate(groups.tolist()):
        by_group.setdefault(gk, []).append(i)

    diffs: list[np.ndarray] = []
    lbls: list[int] = []

    for idx_list in by_group.values():
        idx_arr = np.asarray(idx_list)
        y_g = y[idx_arr]
        pos = idx_arr[y_g == 1]
        neg = idx_arr[y_g == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        for pi in pos:
            for ni in neg:
                diffs.append(Z[pi] - Z[ni])
                lbls.append(1)
                diffs.append(Z[ni] - Z[pi])
                lbls.append(0)

    if len(diffs) < 4:
        return None

    X_pairs = np.vstack(diffs)
    y_pairs = np.asarray(lbls, dtype=np.int32)

    if X_pairs.shape[0] > max_pairs:
        rng_p = np.random.RandomState(random_state)
        sel = rng_p.choice(X_pairs.shape[0], size=max_pairs, replace=False)
        X_pairs, y_pairs = X_pairs[sel], y_pairs[sel]

    clf = LogisticRegression(C=0.1, max_iter=2000, random_state=random_state)
    try:
        clf.fit(X_pairs, y_pairs)
    except Exception:
        return None
    return clf


# ─── Score functions for evaluate_method_from_feature_store ──────────────────

def _make_ssl_score_fn(
    ssl_bundle: dict[str, Any],
    heads: dict[int, LogisticRegression],  # anchor_idx → clf
    domain: str,
    is_pairwise: bool = False,
) -> Callable[[str, int, np.ndarray], np.ndarray]:
    B = ssl_bundle["B"]
    scaler = ssl_bundle["scaler"]
    feat_idx = FIXED_FEATURE_INDICES

    def _score(dom: str, pos_idx: int, x_raw_all: np.ndarray) -> np.ndarray:
        n = x_raw_all.shape[0]
        if dom != domain:
            return np.zeros(n, dtype=np.float64)
        a_idx = _POS_TO_ANCHOR_IDX.get(pos_idx, 0)
        clf = heads.get(a_idx)
        if clf is None:
            return np.zeros(n, dtype=np.float64)
        x_f = x_raw_all[:, feat_idx]
        x_rk = _rank_transform_matrix(x_f)
        X_rep = np.concatenate([x_f, x_rk], axis=1)
        Z = scaler.transform(X_rep) @ B
        if is_pairwise:
            Z_c = Z - Z.mean(axis=0, keepdims=True)
            return np.asarray(clf.decision_function(Z_c), dtype=np.float64)
        return np.asarray(clf.decision_function(Z), dtype=np.float64)

    return _score


def _make_frozen_svd_score_fn(
    route_models: dict[int, dict[str, Any]],  # anchor_idx → {scaler, svd, lr, whiten}
    heads: dict[int, LogisticRegression],
    domain: str,
) -> Callable[[str, int, np.ndarray], np.ndarray]:
    feat_idx = FIXED_FEATURE_INDICES

    def _score(dom: str, pos_idx: int, x_raw_all: np.ndarray) -> np.ndarray:
        n = x_raw_all.shape[0]
        if dom != domain:
            return np.zeros(n, dtype=np.float64)
        a_idx = _POS_TO_ANCHOR_IDX.get(pos_idx, 0)
        clf = heads.get(a_idx)
        model = route_models.get(a_idx)
        if clf is None or model is None:
            return np.zeros(n, dtype=np.float64)
        x_f = x_raw_all[:, feat_idx]
        x_rk = _rank_transform_matrix(x_f)
        X_rep = np.concatenate([x_f, x_rk], axis=1)
        sc: StandardScaler = model["scaler"]
        svd: TruncatedSVD = model["svd"]
        whiten = bool(model.get("whiten", False))
        Z = svd.transform(sc.transform(X_rep))
        if whiten:
            sv = np.asarray(svd.singular_values_, dtype=np.float64)
            sv = np.where(np.abs(sv) < 1e-8, 1.0, sv)
            Z = Z / sv
        return np.asarray(clf.decision_function(Z), dtype=np.float64)

    return _score


def _make_supervised_svd_score_fn(
    models: dict[int, dict[str, Any]],  # anchor_idx → {scaler, svd, lr, whiten}
    domain: str,
) -> Callable[[str, int, np.ndarray], np.ndarray]:
    feat_idx = FIXED_FEATURE_INDICES

    def _score(dom: str, pos_idx: int, x_raw_all: np.ndarray) -> np.ndarray:
        n = x_raw_all.shape[0]
        if dom != domain:
            return np.zeros(n, dtype=np.float64)
        a_idx = _POS_TO_ANCHOR_IDX.get(pos_idx, 0)
        m = models.get(a_idx)
        if m is None:
            return np.zeros(n, dtype=np.float64)
        x_f = x_raw_all[:, feat_idx]
        x_rk = _rank_transform_matrix(x_f)
        X_rep = np.concatenate([x_f, x_rk], axis=1)
        Z = m["svd"].transform(m["scaler"].transform(X_rep))
        if bool(m.get("whiten", False)):
            sv = np.asarray(m["svd"].singular_values_, dtype=np.float64)
            sv = np.where(np.abs(sv) < 1e-8, 1.0, sv)
            Z = Z / sv
        return np.asarray(m["lr"].decision_function(Z), dtype=np.float64)

    return _score


def _make_no_svd_score_fn(
    scalers: dict[int, StandardScaler],
    heads: dict[int, LogisticRegression],
    domain: str,
) -> Callable[[str, int, np.ndarray], np.ndarray]:
    feat_idx = FIXED_FEATURE_INDICES

    def _score(dom: str, pos_idx: int, x_raw_all: np.ndarray) -> np.ndarray:
        n = x_raw_all.shape[0]
        if dom != domain:
            return np.zeros(n, dtype=np.float64)
        a_idx = _POS_TO_ANCHOR_IDX.get(pos_idx, 0)
        clf = heads.get(a_idx)
        sc = scalers.get(a_idx)
        if clf is None or sc is None:
            return np.zeros(n, dtype=np.float64)
        x_f = x_raw_all[:, feat_idx]
        x_rk = _rank_transform_matrix(x_f)
        X_rep = np.concatenate([x_f, x_rk], axis=1)
        return np.asarray(clf.decision_function(sc.transform(X_rep)), dtype=np.float64)

    return _score


# ─── Bundle → per-anchor route models ────────────────────────────────────────

def _extract_route_models(
    bundle: dict[str, Any],
    domain: str,
) -> dict[int, dict[str, Any]]:
    """Extract {anchor_idx → model} from a supervised bundle."""
    routes = bundle.get("domains", {}).get(domain, {}).get("routes", [])
    out: dict[int, dict[str, Any]] = {}
    for pos_i, pos in enumerate(EARLY_STOP_POSITIONS):
        a_idx = _POS_TO_ANCHOR_IDX[pos_i]
        if a_idx in out:
            continue
        if pos_i < len(routes):
            route = routes[pos_i]
            if route.get("route_type") == "svd":
                model = route.get("model")
                if model is not None:
                    out[a_idx] = model
    return out


# ─── Single condition evaluation ──────────────────────────────────────────────

def _eval(
    holdout_store: list[dict[str, Any]],
    score_fn: Callable,
    method_name: str,
) -> dict[str, Any]:
    return evaluate_method_from_feature_store(
        method_name=method_name,
        feature_store=holdout_store,
        position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
        score_fn=score_fn,
    )


def _agg_to_row(
    agg: dict[str, Any],
    condition: str,
    domain: str,
    fraction: float,
    n_labeled: int,
    ssl_rank: int,
) -> dict[str, Any]:
    return {
        "condition": condition,
        "domain": domain,
        "label_fraction": float(fraction),
        "n_labeled": int(n_labeled),
        "ssl_rank": int(ssl_rank),
        "auroc": float(agg.get("auc_of_auroc", float("nan"))),
        "top1_acc": float(agg.get("stop_acc@100%", float("nan"))),
        "auc_of_selacc": float(agg.get("auc_of_selacc", float("nan"))),
    }


# ─── Label-efficiency study ───────────────────────────────────────────────────

def _run_label_efficiency_study(
    train_by_domain: dict[str, list[dict[str, Any]]],
    holdout_by_domain: dict[str, list[dict[str, Any]]],
    ssl_bundles: dict[int, dict[str, Any]],
    supervised_bundle: Optional[dict[str, Any]],
    label_fractions: tuple[float, ...],
    ssl_ranks: tuple[int, ...],
    domains: tuple[str, ...],
    seed: int = 42,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    active_fractions = (0.10, 0.50, 1.0) if smoke else label_fractions

    for domain in domains:
        train_store = train_by_domain.get(domain, [])
        holdout_store = holdout_by_domain.get(domain, [])
        if not train_store or not holdout_store:
            print(f"[study] skip domain={domain}: missing train or holdout data")
            continue

        all_pids = sorted(
            {str(pid) for p in train_store for pid in p["problem_ids"]},
            key=_problem_sort_key,
        )
        n_total = len(all_pids)
        use_pairwise = (domain == "coding")
        print(f"\n[study] domain={domain} n_train_problems={n_total} "
              f"pairwise={use_pairwise} n_holdout_payloads={len(holdout_store)}")

        for fraction in active_fractions:
            selected = _sample_labeled_problems(train_store, domain, fraction, seed=seed)
            labeled_store = _subset_store_by_problems(train_store, domain, selected)
            if not labeled_store:
                print(f"[study] skip domain={domain} frac={fraction:.0%}: empty labeled store")
                continue

            n_labeled = len(selected)
            tables = _build_anchor_tables(labeled_store, domain=domain)
            print(f"[study] domain={domain} frac={fraction:.0%} n_labeled={n_labeled}")

            # ── supervised_svd ────────────────────────────────────────────
            svd_models: dict[int, dict[str, Any]] = {}
            for a_idx, tbl in tables.items():
                x_rw, x_rk, y, grp = tbl["x_raw"], tbl["x_rank"], tbl["y"], tbl["groups"]
                if x_rw.shape[0] < 4 or np.unique(y).shape[0] < 2:
                    continue
                X_rep = np.concatenate([x_rw, x_rk], axis=1)
                r_svd = min(8, X_rep.shape[1] - 1, X_rep.shape[0] - 1)
                if r_svd < 1:
                    continue
                try:
                    sc = StandardScaler()
                    svd = TruncatedSVD(n_components=r_svd, random_state=seed)
                    Z = svd.fit_transform(sc.fit_transform(X_rep))
                    clf = LogisticRegression(C=0.1, max_iter=2000, random_state=seed)
                    clf.fit(Z, y)
                    svd_models[a_idx] = {"scaler": sc, "svd": svd, "lr": clf, "whiten": False}
                except Exception as e:
                    print(f"[study] supervised_svd a={a_idx} err: {e}")

            if svd_models:
                score_fn = _make_supervised_svd_score_fn(svd_models, domain)
                res = _eval(holdout_store, score_fn, f"supervised_svd/{domain}")
                row = _agg_to_row(res["aggregate"], "supervised_svd", domain, fraction, n_labeled, -1)
                results.append(row)
                print(f"[result] supervised_svd domain={domain} frac={fraction:.0%} "
                      f"auroc={row['auroc']:.4f}")

            # ── no_svd_lr ─────────────────────────────────────────────────
            no_svd_heads: dict[int, LogisticRegression] = {}
            no_svd_scalers: dict[int, StandardScaler] = {}
            for a_idx, tbl in tables.items():
                x_rw, x_rk, y = tbl["x_raw"], tbl["x_rank"], tbl["y"]
                if x_rw.shape[0] < 4 or np.unique(y).shape[0] < 2:
                    continue
                X_rep = np.concatenate([x_rw, x_rk], axis=1)
                sc = StandardScaler()
                X_sc = sc.fit_transform(X_rep)
                clf = LogisticRegression(C=0.1, max_iter=2000, random_state=seed)
                try:
                    clf.fit(X_sc, y)
                    no_svd_heads[a_idx] = clf
                    no_svd_scalers[a_idx] = sc
                except Exception as e:
                    print(f"[study] no_svd_lr a={a_idx} err: {e}")

            if no_svd_heads:
                score_fn = _make_no_svd_score_fn(no_svd_scalers, no_svd_heads, domain)
                res = _eval(holdout_store, score_fn, f"no_svd_lr/{domain}")
                row = _agg_to_row(res["aggregate"], "no_svd_lr", domain, fraction, n_labeled, -1)
                results.append(row)
                print(f"[result] no_svd_lr domain={domain} frac={fraction:.0%} "
                      f"auroc={row['auroc']:.4f}")

            # ── frozen_svd ────────────────────────────────────────────────
            if supervised_bundle is not None:
                frozen_route_models = _extract_route_models(supervised_bundle, domain)
                frozen_heads: dict[int, LogisticRegression] = {}
                for a_idx, model in frozen_route_models.items():
                    tbl = tables.get(a_idx, {})
                    x_rw, x_rk, y = tbl.get("x_raw", np.zeros((0, N_FEATURES))), \
                                     tbl.get("x_rank", np.zeros((0, N_FEATURES))), \
                                     tbl.get("y", np.zeros(0, dtype=np.int32))
                    if x_rw.shape[0] < 4 or np.unique(y).shape[0] < 2:
                        continue
                    X_rep = np.concatenate([x_rw, x_rk], axis=1)
                    sc_fr: StandardScaler = model["scaler"]
                    svd_fr: TruncatedSVD = model["svd"]
                    whiten = bool(model.get("whiten", False))
                    Z_fr = svd_fr.transform(sc_fr.transform(X_rep))
                    if whiten:
                        sv = np.asarray(svd_fr.singular_values_, dtype=np.float64)
                        sv = np.where(np.abs(sv) < 1e-8, 1.0, sv)
                        Z_fr = Z_fr / sv
                    clf = LogisticRegression(C=0.1, max_iter=2000, random_state=seed)
                    try:
                        clf.fit(Z_fr, y)
                        frozen_heads[a_idx] = clf
                    except Exception:
                        pass

                if frozen_heads:
                    score_fn = _make_frozen_svd_score_fn(frozen_route_models, frozen_heads, domain)
                    res = _eval(holdout_store, score_fn, f"frozen_svd/{domain}")
                    row = _agg_to_row(res["aggregate"], "frozen_svd", domain, fraction, n_labeled, -1)
                    results.append(row)
                    print(f"[result] frozen_svd domain={domain} frac={fraction:.0%} "
                          f"auroc={row['auroc']:.4f}")

            # ── semisup (one per rank) ────────────────────────────────────
            for r_ssl in ssl_ranks:
                ssl_bundle = ssl_bundles.get(r_ssl)
                if ssl_bundle is None:
                    continue

                transform_fn = _make_transform_fn(ssl_bundle["B"], ssl_bundle["scaler"])
                ssl_heads: dict[int, LogisticRegression] = {}

                for a_idx, tbl in tables.items():
                    x_rw, x_rk, y, grp = tbl["x_raw"], tbl["x_rank"], tbl["y"], tbl["groups"]
                    if x_rw.shape[0] < 4 or np.unique(y).shape[0] < 2:
                        continue
                    c_search = fraction >= 0.10
                    anchor_pct = float(ANCHOR_POSITIONS[a_idx])

                    if use_pairwise and anchor_pct >= 0.70:
                        clf = _fit_pairwise_head(
                            x_rw, x_rk, y, grp, transform_fn, random_state=seed,
                        )
                    else:
                        clf = _fit_pointwise_head(
                            x_rw, x_rk, y, grp, transform_fn,
                            c_search=c_search, random_state=seed,
                        )
                    if clf is not None:
                        ssl_heads[a_idx] = clf

                if ssl_heads:
                    score_fn = _make_ssl_score_fn(ssl_bundle, ssl_heads, domain,
                                                  is_pairwise=use_pairwise)
                    res = _eval(holdout_store, score_fn, f"semisup_r{r_ssl}/{domain}")
                    row = _agg_to_row(res["aggregate"], "semisup", domain, fraction, n_labeled, r_ssl)
                    results.append(row)
                    print(f"[result] semisup r={r_ssl} domain={domain} frac={fraction:.0%} "
                          f"auroc={row['auroc']:.4f}")

    return results


# ─── Outputs ─────────────────────────────────────────────────────────────────

def _write_csv(results: list[dict[str, Any]], out_path: Path) -> None:
    if not results:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["domain", "condition", "ssl_rank", "label_fraction",
              "n_labeled", "auroc", "top1_acc", "auc_of_selacc"]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in results:
            w.writerow(row)
    print(f"[out] CSV → {_display_path(out_path)}")


def _write_markdown(results: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Semi-Supervised SVDomain Extension",
        "",
        f"Generated: {_now_utc()}",
        "",
        "## Summary",
        "",
        "| Domain | Condition | SSL Rank | Label % | AUROC | Top1 Acc | AUC SelAcc |",
        "|--------|-----------|:--------:|--------:|------:|---------:|----------:|",
    ]
    for row in sorted(results, key=lambda r: (
        r["domain"], r["condition"], r["ssl_rank"], r["label_fraction"]
    )):
        lines.append(
            f"| {row['domain']} | {row['condition']} | {row['ssl_rank']} "
            f"| {row['label_fraction']:.0%} "
            f"| {row.get('auroc', float('nan')):.4f} "
            f"| {row.get('top1_acc', float('nan')):.4f} "
            f"| {row.get('auc_of_selacc', float('nan')):.4f} |"
        )
    lines += [
        "",
        "## Conditions",
        "",
        "- **semisup**: self-supervised SSL basis + supervised head",
        "- **frozen_svd**: frozen SVD from supervised bundle + new LR head",
        "- **supervised_svd**: fresh SVD + LR trained on labeled subset only",
        "- **no_svd_lr**: StandardScaler → LR, no dimensionality reduction",
        "",
        "## SSL Losses",
        "",
        "- `L_recon`: masked feature reconstruction (30% mask rate)",
        "- `L_align`: cross-anchor alignment via bridge W_ap (early 10% → late 70%)",
        "- `L_view`: raw-vs-rank view consistency",
        "- `L_total = L_recon + 0.5 * L_align + 0.5 * L_view`",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[out] markdown → {_display_path(out_path)}")


# ─── Pre-built cache loader ───────────────────────────────────────────────────

def _load_prebuilt_stores(cache_dir: Path) -> list[dict[str, Any]]:
    """Load all qualified feature-store pkl files from an existing cache directory.

    Expects files with format {"feature_store": [...], "source_name": ..., ...}
    as produced by _load_or_build_qualified_feature_store in train_es_svd_ms_rr_r1.
    Files whose names contain "noncoding" or "cap" (capped) are skipped in favour
    of the full ("all") versions.
    """
    combined: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    pkl_files = sorted(cache_dir.glob("*.pkl"))
    # Prefer the "all" (uncapped) files; skip noncoding-filtered copies
    priority = [p for p in pkl_files if "_all_" in p.name and "noncoding" not in p.name]
    fallback = [p for p in pkl_files if p not in priority]

    for pkl in priority + fallback:
        try:
            with pkl.open("rb") as fh:
                data = pickle.load(fh)
        except Exception as e:
            print(f"[prebuilt] skip {pkl.name}: {e}")
            continue
        if not isinstance(data, dict) or "feature_store" not in data:
            continue
        source = str(data.get("source_name", pkl.stem))
        if source in seen_sources:
            continue  # already loaded a full version for this source
        seen_sources.add(source)
        store = list(data["feature_store"])
        n_samples = sum(int(p.get("samples", 0)) for p in store)
        print(f"[prebuilt] loaded source={source} payloads={len(store)} "
              f"samples={n_samples} ← {pkl.name}")
        combined.extend(store)

    return combined


# ─── Main ────────────────────────────────────────────────────────────────────

def _resolve_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Semi-supervised SVDomain label efficiency study"
    )
    ap.add_argument("--main-cache-root", default="MUI_HUB/cache")
    ap.add_argument("--extra-cache-root", default=None)
    ap.add_argument("--test-cache-root", default="MUI_HUB/cache_test",
                    help="Used as holdout evaluation set")
    ap.add_argument("--prebuilt-cache-dir", default=None,
                    help="Directory with pre-built feature cache pkl files "
                         "(e.g. results/cache/es_svd_ms_rr_r1). When set, "
                         "skips build_feature_store and uses those caches directly; "
                         "holdout is then taken as a 85/15 split of the combined store.")
    ap.add_argument("--out-dir", default="results/cache/semisup_svdomain")
    ap.add_argument("--out-csv", default="results/tables/semisup_svdomain.csv")
    ap.add_argument("--out-doc", default="docs/SEMISUP_SVDOMAIN.md")
    ap.add_argument("--ssl-ranks", nargs="+", type=int, default=list(DEFAULT_SSL_RANKS))
    ap.add_argument("--n-epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--mask-rate", type=float, default=0.30)
    ap.add_argument("--lam-align", type=float, default=0.5)
    ap.add_argument("--lam-view", type=float, default=0.5)
    ap.add_argument("--holdout-split", type=float, default=0.15,
                    help="Holdout ratio for 85/15 split (used when no separate test cache)")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--feature-workers", type=int, default=8)
    ap.add_argument("--feature-chunk-problems", type=int, default=DEFAULT_FEATURE_CHUNK_PROBLEMS)
    ap.add_argument("--max-problems-per-cache", type=int, default=0)
    ap.add_argument("--supervised-bundle", default=str(DEFAULT_BUNDLE_PATH))
    ap.add_argument("--label-fractions", nargs="+", type=float,
                    default=list(LABEL_FRACTIONS))
    ap.add_argument("--domains", nargs="+", default=["math", "science"])
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--refresh-ssl-cache", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="Quick smoke test: caps problems, epochs, fractions")
    args = ap.parse_args()

    out_dir = _resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_problems: Optional[int] = (
        None if int(args.max_problems_per_cache) <= 0
        else int(args.max_problems_per_cache)
    )
    if args.smoke:
        max_problems = max_problems or 10
        args.n_epochs = min(args.n_epochs, 30)
        args.batch_size = min(args.batch_size, 64)

    ssl_ranks = tuple(sorted(set(int(r) for r in args.ssl_ranks)))
    domains = tuple(d for d in args.domains if d in ALL_DOMAINS)
    label_fractions = tuple(float(f) for f in args.label_fractions)
    required_features = set(FIXED_FEATURE_NAMES)

    # ── Feature stores ────────────────────────────────────────────────────────
    feat_cache = out_dir / "feature_store.pkl"

    if args.prebuilt_cache_dir:
        # Fast path: reuse pre-built feature pkl files (avoids slow NAD cache reads)
        prebuilt_dir = _resolve_path(args.prebuilt_cache_dir)
        print(f"[prebuilt] loading feature stores from {_display_path(prebuilt_dir)}")
        full_store: list[dict[str, Any]] = _load_prebuilt_stores(prebuilt_dir)
        if not full_store:
            raise RuntimeError(f"No usable feature stores found in {prebuilt_dir}")
        # Save combined for caching
        with feat_cache.open("wb") as fh:
            pickle.dump(full_store, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[save] combined store → {_display_path(feat_cache)}")
        holdout_store: list[dict[str, Any]] = []  # will split below

    elif feat_cache.exists() and not args.refresh_feature_cache:
        print(f"[load] feature store ← {_display_path(feat_cache)}")
        with feat_cache.open("rb") as fh:
            full_store = pickle.load(fh)
        holdout_store = []

    else:
        main_root = _resolve_path(args.main_cache_root)
        print(f"[build] feature store main={main_root}")
        raw = build_feature_store(
            cache_root=main_root,
            positions=EXTRACTION_POSITIONS,
            required_feature_names=required_features,
            max_problems_per_cache=max_problems,
            max_workers=int(args.feature_workers),
            chunk_problems=int(args.feature_chunk_problems),
        )
        full_store = _qualify_feature_store(raw, "cache")

        if args.extra_cache_root:
            extra_root = _resolve_path(args.extra_cache_root)
            print(f"[build] feature store extra={extra_root}")
            extra_raw = build_feature_store(
                cache_root=extra_root,
                positions=EXTRACTION_POSITIONS,
                required_feature_names=required_features,
                max_problems_per_cache=max_problems,
                max_workers=int(args.feature_workers),
                chunk_problems=int(args.feature_chunk_problems),
            )
            full_store += _qualify_feature_store(extra_raw, "cache_train")

        with feat_cache.open("wb") as fh:
            pickle.dump(full_store, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[save] feature store → {_display_path(feat_cache)}")
        holdout_store = []

    # ── Holdout: try test-cache pkl, else fall back to 85/15 split ───────────
    if not holdout_store and not args.prebuilt_cache_dir:
        holdout_cache_pkl = out_dir / "holdout_store.pkl"
        test_root = _resolve_path(args.test_cache_root)
        if holdout_cache_pkl.exists() and not args.refresh_feature_cache:
            print(f"[load] holdout store ← {_display_path(holdout_cache_pkl)}")
            with holdout_cache_pkl.open("rb") as fh:
                holdout_store = pickle.load(fh)
        elif test_root.exists():
            print(f"[build] holdout store test={test_root}")
            h_raw = build_feature_store(
                cache_root=test_root,
                positions=EXTRACTION_POSITIONS,
                required_feature_names=required_features,
                max_problems_per_cache=max_problems,
                max_workers=int(args.feature_workers),
                chunk_problems=int(args.feature_chunk_problems),
            )
            holdout_store = _qualify_feature_store(h_raw, "cache_test")
            with holdout_cache_pkl.open("wb") as fh:
                pickle.dump(holdout_store, fh, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[save] holdout store → {_display_path(holdout_cache_pkl)}")

    # ── Train / holdout split ─────────────────────────────────────────────────
    if holdout_store:
        train_store = full_store
        print(f"[split] using separate holdout store ({len(holdout_store)} payloads)")
    else:
        print("[split] no separate holdout — splitting train 85/15")
        hmap, _ = _build_holdout_problem_map(
            full_store,
            holdout_split=float(args.holdout_split),
            split_seed=int(args.split_seed),
        )
        train_store, holdout_store, _ = _split_feature_store(
            full_store, holdout_problem_map=hmap,
        )

    # ── Domain partition ──────────────────────────────────────────────────────
    train_by_domain: dict[str, list[dict[str, Any]]] = {}
    holdout_by_domain: dict[str, list[dict[str, Any]]] = {}
    for dom in ALL_DOMAINS:
        train_by_domain[dom] = [p for p in train_store if p["domain"] == dom]
        holdout_by_domain[dom] = [p for p in holdout_store if p["domain"] == dom]
        print(f"[split] domain={dom} train={len(train_by_domain[dom])} "
              f"holdout={len(holdout_by_domain[dom])}")

    # ── SSL pre-training ──────────────────────────────────────────────────────
    ssl_cache = out_dir / "ssl_bundles.pkl"
    ssl_bundles: dict[int, dict[str, Any]] = {}
    if ssl_cache.exists() and not args.refresh_ssl_cache:
        print(f"[load] SSL bundles ← {_display_path(ssl_cache)}")
        with ssl_cache.open("rb") as fh:
            ssl_bundles = pickle.load(fh)
    else:
        print("[ssl] extracting feature matrices from all training data...")
        X_ssl = _extract_ssl_matrix(train_store)
        print(f"[ssl] X_ssl shape: {X_ssl.shape}")

        if X_ssl.shape[0] < 4:
            print("[warn] not enough training runs for SSL; skipping SSL conditions")
        else:
            for r_ssl in ssl_ranks:
                ssl_bundles[r_ssl] = pretrain_ssl_basis(
                    X_full_all=X_ssl,
                    r=r_ssl,
                    n_epochs=int(args.n_epochs),
                    batch=int(args.batch_size),
                    lr=float(args.lr),
                    seed=int(args.split_seed),
                    smoke=bool(args.smoke),
                    lam_align=float(args.lam_align),
                    lam_view=float(args.lam_view),
                    mask_rate=float(args.mask_rate),
                )

        with ssl_cache.open("wb") as fh:
            pickle.dump(ssl_bundles, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[save] SSL bundles → {_display_path(ssl_cache)}")

    # ── Load supervised bundle (for frozen_svd) ───────────────────────────────
    supervised_bundle: Optional[dict[str, Any]] = None
    bundle_path = _resolve_path(args.supervised_bundle)
    if bundle_path.exists():
        try:
            supervised_bundle = load_earlystop_svd_bundle(bundle_path)
            print(f"[load] supervised bundle ← {_display_path(bundle_path)}")
        except Exception as e:
            print(f"[warn] could not load supervised bundle: {e}")
    else:
        print(f"[warn] supervised bundle not found: {bundle_path} — skipping frozen_svd")

    # ── Label-efficiency study ────────────────────────────────────────────────
    results = _run_label_efficiency_study(
        train_by_domain=train_by_domain,
        holdout_by_domain=holdout_by_domain,
        ssl_bundles=ssl_bundles,
        supervised_bundle=supervised_bundle,
        label_fractions=label_fractions,
        ssl_ranks=tuple(r for r in ssl_ranks if r in ssl_bundles),
        domains=domains,
        seed=int(args.split_seed),
        smoke=bool(args.smoke),
    )

    # ── Write outputs ─────────────────────────────────────────────────────────
    out_csv = _resolve_path(args.out_csv)
    out_doc = _resolve_path(args.out_doc)
    _write_csv(results, out_csv)
    _write_markdown(results, out_doc)

    print(f"\n[done] {len(results)} result rows")
    print(f"  CSV:      {_display_path(out_csv)}")
    print(f"  Markdown: {_display_path(out_doc)}")


if __name__ == "__main__":
    main()
