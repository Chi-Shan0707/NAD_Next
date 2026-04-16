#!/usr/bin/env python3
"""Domain-Specific Contrastive SSL for SVDomain.

Addresses five root causes of failure in the shared-basis SSL experiment:
  1. Reconstruction dominance   → replaced with NT-Xent contrastive objective
  2. Cross-domain pollution      → per-domain separate bases
  3. Early saddle                → cosine LR annealing
  4. Linear autoencoder mismatch → SVD warm-start + contrastive objective
  5. Pairwise coding collapse    → pairwise hinge during pre-training (weak sup.)

Conditions compared:
  domain_ssl       — per-domain NT-Xent basis + supervised head
  domain_ssl_weak  — same but coding pre-training adds pairwise hinge
  shared_ssl_r16   — old shared SSL basis (r=16) for blame attribution
  no_svd_lr        — StandardScaler → LR (prior winner)
  frozen_svd       — frozen supervised SVD + new LR head

Usage:
  # Smoke test (~3-5 min)
  python3 scripts/semi_supervised/train_domain_specific_ssl.py \\
      --prebuilt-cache-dir results/cache/es_svd_ms_rr_r1 \\
      --shared-ssl-pkl results/cache/semisup_svdomain/ssl_bundles.pkl \\
      --ssl-ranks 4 8 --n-epochs 20 --batch-size 64 \\
      --domains math science coding --smoke \\
      --out-dir results/cache/domain_ssl_smoke \\
      --out-csv results/tables/domain_specific_ssl.smoke.csv \\
      --out-doc docs/DOMAIN_SPECIFIC_SSL_SMOKE.md

  # Full run (~25-45 min)
  python3 scripts/semi_supervised/train_domain_specific_ssl.py \\
      --prebuilt-cache-dir results/cache/es_svd_ms_rr_r1 \\
      --shared-ssl-pkl results/cache/semisup_svdomain/ssl_bundles.pkl \\
      --ssl-ranks 4 8 16 --n-epochs 300 --batch-size 256 \\
      --domains math science coding \\
      --out-dir results/cache/domain_ssl \\
      --out-csv results/tables/domain_specific_ssl.csv \\
      --out-doc docs/DOMAIN_SPECIFIC_SSL.md
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

# Reuse helpers from the previous experiment script
from scripts.semi_supervised.train_semisup_svdomain import (
    _adam_update,
    _build_anchor_tables,
    _extract_route_models,
    _fit_pairwise_head,
    _fit_pointwise_head,
    _load_prebuilt_stores,
    _make_frozen_svd_score_fn,
    _make_no_svd_score_fn,
    _make_ssl_score_fn,
    _make_transform_fn,
    _sample_labeled_problems,
    _subset_store_by_problems,
    _eval,
    _agg_to_row,
    _write_csv,
)


# ─── Constants ────────────────────────────────────────────────────────────────

LABEL_FRACTIONS = (0.01, 0.05, 0.10, 0.20, 0.50, 1.0)
DEFAULT_SSL_RANKS = (4, 8, 16)
ALL_DOMAINS = ("math", "science", "coding")

ANCHOR_POS_INDICES = [EXTRACTION_POSITION_INDEX[float(p)] for p in ANCHOR_POSITIONS]
N_FEATURES = len(FIXED_FEATURE_NAMES)   # 22 raw features
D_FULL = 2 * N_FEATURES                  # 44 = raw + rank

DEFAULT_BUNDLE_PATH = REPO_ROOT / "models/ml_selectors/es_svd_ms_rr_r1.pkl"
DEFAULT_SHARED_SSL_PKL = REPO_ROOT / "results/cache/semisup_svdomain/ssl_bundles.pkl"

# Map each EARLY_STOP_POSITIONS index to nearest anchor index
_POS_TO_ANCHOR_IDX: dict[int, int] = {}
for _pi, _pos in enumerate(EARLY_STOP_POSITIONS):
    _anch = float(OFFICIAL_SLOT_TO_ANCHOR[float(_pos)])
    _POS_TO_ANCHOR_IDX[_pi] = list(ANCHOR_POSITIONS).index(_anch)

# ── Domain-specific configuration ─────────────────────────────────────────────
# NT-Xent view pair and future-prediction anchor pair use the same anchor indices.
# early_anchor / late_anchor: indices into ANCHOR_POSITIONS = (0.10, 0.40, 0.70, 1.00)
DOMAIN_CFG: dict[str, dict[str, Any]] = {
    "math": {
        "early_anchor": 0,   # 10%
        "late_anchor": 3,    # 100%
        "batch_size": 256,
        "pairwise_in_pretrain": False,
        "proxy_reconstruction": False,
    },
    "science": {
        "early_anchor": 1,   # 40%
        "late_anchor": 3,    # 100%
        "batch_size": 256,
        "pairwise_in_pretrain": False,
        "proxy_reconstruction": False,
    },
    "coding": {
        "early_anchor": 1,   # 40%
        "late_anchor": 2,    # 70% (avoids at-max-token distortion of 100%)
        "batch_size": 384,
        "pairwise_in_pretrain": True,   # active only in domain_ssl_weak
        "proxy_reconstruction": True,   # only if cache roots available
    },
}


# ─── LR schedule ──────────────────────────────────────────────────────────────

def _cosine_lr(epoch: int, n_epochs: int, lr_max: float, lr_min: float = 1e-4) -> float:
    """Cosine annealing: lr_min + 0.5*(lr_max-lr_min)*(1 + cos(π*epoch/n_epochs))."""
    return float(lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos(np.pi * epoch / max(1, n_epochs))))


# ─── L2 normalization ─────────────────────────────────────────────────────────

def _l2_normalize_rows(Z: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise L2 normalize Z.  Returns (H, norms) where H = Z / norms."""
    norms = np.sqrt(np.sum(Z ** 2, axis=1, keepdims=True) + eps)
    return Z / norms, norms


# ─── NT-Xent contrastive loss and gradient ────────────────────────────────────

def _nt_xent_loss_grad(
    X_a: np.ndarray,   # (n, d)  early-anchor features, pre-scaled
    X_b: np.ndarray,   # (n, d)  late-anchor features, pre-scaled
    B: np.ndarray,     # (d, r)
    tau: float = 0.07,
) -> tuple[float, np.ndarray]:
    """Full symmetric NT-Xent loss and gradient dL/dB.

    Positives: (X_a[i], X_b[i]) same-run pair at two different anchor positions.
    Negatives: all other runs in the batch.

    Returns (loss, grad_B) where grad_B has shape (d, r).
    """
    n = X_a.shape[0]
    if n < 2:
        return 0.0, np.zeros_like(B)

    # Project
    Z_a = X_a @ B  # (n, r)
    Z_b = X_b @ B  # (n, r)

    # L2 normalize
    H_a, norms_a = _l2_normalize_rows(Z_a)
    H_b, norms_b = _l2_normalize_rows(Z_b)

    # Build 2n × 2n similarity matrix / tau
    S_aa = H_a @ H_a.T / tau   # (n, n)
    S_ab = H_a @ H_b.T / tau   # (n, n)
    S_bb = H_b @ H_b.T / tau   # (n, n)

    S = np.empty((2 * n, 2 * n), dtype=np.float64)
    S[:n, :n] = S_aa
    S[:n, n:] = S_ab
    S[n:, :n] = S_ab.T   # S_ba
    S[n:, n:] = S_bb
    np.fill_diagonal(S, -np.inf)

    # Numerically stable softmax per row
    S_max = np.max(S, axis=1, keepdims=True)   # (2n, 1)
    exp_S = np.exp(S - S_max)
    sum_exp = np.sum(exp_S, axis=1, keepdims=True)   # (2n, 1)
    P = exp_S / sum_exp   # (2n, 2n)

    # Loss: -½ * (1/n) * Σ_i [ log P[i, n+i] + log P[n+i, i] ]
    # S_ab[i,i] = S[i, n+i] = S[n+i, i]  (positive pair logit)
    diag_ab = np.diag(S_ab)   # (n,)
    log_p_top = diag_ab - S_max[:n, 0] - np.log(sum_exp[:n, 0])
    log_p_bot = diag_ab - S_max[n:, 0] - np.log(sum_exp[n:, 0])
    L = float(-0.5 * np.mean(log_p_top + log_p_bot))

    # Gradient w.r.t. S: G[i,j] = (1/(2n)) * (P[i,j] - delta_{j, pos_i})
    G = P / (2.0 * n)
    idx = np.arange(n)
    G[idx, n + idx] -= 1.0 / (2.0 * n)
    G[n + idx, idx] -= 1.0 / (2.0 * n)

    # Split into blocks
    G_aa = G[:n, :n]   # (n, n)
    G_ab = G[:n, n:]   # (n, n)
    G_ba = G[n:, :n]   # (n, n)
    G_bb = G[n:, n:]   # (n, n)

    # Gradient w.r.t. H_a and H_b (via chain rule through S = H H.T / tau)
    # dL/dH_a = (1/tau) * [(G_aa + G_aa.T) @ H_a + (G_ab + G_ba.T) @ H_b]
    dH_a = ((G_aa + G_aa.T) @ H_a + (G_ab + G_ba.T) @ H_b) / tau   # (n, r)
    dH_b = ((G_bb + G_bb.T) @ H_b + (G_ba + G_ab.T) @ H_a) / tau   # (n, r)

    # Gradient through L2 normalization: dL/dZ[i] = (u[i] - H[i]*(H[i]·u[i])) / ||Z[i]||
    proj_a = np.sum(H_a * dH_a, axis=1, keepdims=True)   # (n, 1)
    dZ_a = (dH_a - H_a * proj_a) / norms_a               # (n, r)
    proj_b = np.sum(H_b * dH_b, axis=1, keepdims=True)
    dZ_b = (dH_b - H_b * proj_b) / norms_b               # (n, r)

    # Gradient w.r.t. B
    grad_B = X_a.T @ dZ_a + X_b.T @ dZ_b   # (d, r)

    return L, grad_B


# ─── Future-anchor prediction loss and gradient ───────────────────────────────

def _future_prediction_loss_grad(
    B: np.ndarray,        # (d, r)
    W_future: np.ndarray, # (r, r)
    X_early: np.ndarray,  # (n, d)
    X_late: np.ndarray,   # (n, d)
) -> tuple[float, np.ndarray, np.ndarray]:
    """MSE prediction: Z_early @ W_future ≈ Z_late.

    Returns (loss, grad_B, grad_W_future).
    """
    n = float(max(1, X_early.shape[0]))
    Z_early = X_early @ B    # (n, r)
    Z_late = X_late @ B      # (n, r)
    pred = Z_early @ W_future  # (n, r)
    err = pred - Z_late          # (n, r)
    L = 0.5 * float(np.sum(err ** 2)) / n

    grad_W_future = (Z_early.T @ err) / n                             # (r, r)
    grad_B = (X_early.T @ (err @ W_future.T) - X_late.T @ err) / n  # (d, r)

    return L, grad_B, grad_W_future


# ─── Pairwise hinge loss and gradient (coding domain) ─────────────────────────

def _pairwise_hinge_loss_grad(
    B: np.ndarray,         # (d, r)
    W_score: np.ndarray,   # (r, 1)
    X_batch: np.ndarray,   # (n, d)
    y_batch: np.ndarray,   # (n,)
    group_keys: np.ndarray,  # (n,)
    margin: float = 1.0,
    max_pairs: int = 2048,
    rng: Optional[np.random.RandomState] = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Max-margin hinge on (correct_i, wrong_j) pairs within problem groups.

    Returns (loss, grad_B, grad_W_score).
    """
    n = X_batch.shape[0]
    Z = X_batch @ B            # (n, r)
    scores = Z @ W_score       # (n, 1)

    # Build (pos, neg) pairs within each problem group
    by_group: dict[Any, list[int]] = {}
    for i, gk in enumerate(group_keys.tolist()):
        by_group.setdefault(gk, []).append(i)

    pos_list: list[int] = []
    neg_list: list[int] = []
    for idx_list in by_group.values():
        idx_arr = np.asarray(idx_list)
        y_g = y_batch[idx_arr]
        pos = idx_arr[y_g == 1]
        neg = idx_arr[y_g == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        for pi in pos:
            for ni in neg:
                pos_list.append(int(pi))
                neg_list.append(int(ni))

    if not pos_list:
        return 0.0, np.zeros_like(B), np.zeros_like(W_score)

    pos_idxs = np.asarray(pos_list, dtype=np.int64)
    neg_idxs = np.asarray(neg_list, dtype=np.int64)
    n_pairs = len(pos_idxs)

    if n_pairs > max_pairs and rng is not None:
        sel = rng.choice(n_pairs, size=max_pairs, replace=False)
        pos_idxs = pos_idxs[sel]
        neg_idxs = neg_idxs[sel]
        n_pairs = max_pairs

    s_pos = scores[pos_idxs, 0]   # (n_pairs,)
    s_neg = scores[neg_idxs, 0]   # (n_pairs,)
    margins = margin - (s_pos - s_neg)
    violated = margins > 0.0

    L = float(np.mean(margins[violated])) if np.any(violated) else 0.0

    # Accumulate score gradients: d_scores[pos] -= 1/n_pairs (violated only)
    d_scores = np.zeros((n, 1), dtype=np.float64)
    if np.any(violated):
        vp = pos_idxs[violated]
        vn = neg_idxs[violated]
        np.add.at(d_scores[:, 0], vp, -1.0 / n_pairs)
        np.add.at(d_scores[:, 0], vn, +1.0 / n_pairs)

    # Chain rule: s = Z @ W_score
    dZ = d_scores @ W_score.T             # (n, r)
    grad_B = X_batch.T @ dZ               # (d, r)
    grad_W_score = Z.T @ d_scores         # (r, 1)

    return L, grad_B, grad_W_score


# ─── Proxy reconstruction loss and gradient (coding domain, optional) ─────────

def _proxy_reconstruction_loss_grad(
    B: np.ndarray,              # (d, r)
    W_decode: np.ndarray,       # (r, 8)
    X_raw_batch: np.ndarray,    # (n, d)
    X_tier1_targets: np.ndarray,  # (n, 8)
) -> tuple[float, np.ndarray, np.ndarray]:
    """MSE decoder: Z @ W_decode ≈ tier-1 coding features.

    Returns (loss, grad_B, grad_W_decode).
    """
    n = float(max(1, X_raw_batch.shape[0]))
    Z = X_raw_batch @ B               # (n, r)
    pred = Z @ W_decode               # (n, 8)
    err = pred - X_tier1_targets      # (n, 8)
    L = 0.5 * float(np.sum(err ** 2)) / n

    grad_W_decode = (Z.T @ err) / n                # (r, 8)
    grad_B = (X_raw_batch.T @ (err @ W_decode.T)) / n  # (d, r)

    return L, grad_B, grad_W_decode


# ─── Combined domain SSL loss dispatcher ──────────────────────────────────────

def _domain_ssl_loss_and_grad(
    B: np.ndarray,         # (d, r)
    W_future: np.ndarray,  # (r, r)
    W_score: np.ndarray,   # (r, 1)
    W_decode: Optional[np.ndarray],  # (r, 8) or None
    X_scaled_batch: np.ndarray,      # (batch, 4, d)  all anchors, pre-scaled
    y_batch: np.ndarray,             # (batch,)
    group_keys: np.ndarray,          # (batch,)
    X_proxy_targets: Optional[np.ndarray],  # (batch, 8) or None
    domain: str,
    tau: float,
    lam_pred: float,
    lam_view: float,
    lam_pair: float,
    lam_proxy: float,
    domain_cfg: dict[str, Any],
    use_pairwise: bool,
    rng: np.random.RandomState,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute all domain-specific SSL losses and combine gradients.

    Returns (L_total, grad_B, grad_W_future, grad_W_score, grad_W_decode).
    """
    early_a: int = domain_cfg["early_anchor"]
    late_a: int = domain_cfg["late_anchor"]
    half_d = B.shape[0] // 2   # 22

    X_early = X_scaled_batch[:, early_a, :]   # (batch, d)
    X_late = X_scaled_batch[:, late_a, :]     # (batch, d)

    # ── Loss 1: NT-Xent contrastive (primary) ─────────────────────────────────
    L1, gB1 = _nt_xent_loss_grad(X_early, X_late, B, tau)

    # ── Loss 2: Future-anchor prediction (λ=0.3) ──────────────────────────────
    L2, gB2, gWf = _future_prediction_loss_grad(B, W_future, X_early, X_late)

    # ── Loss 3: Raw-vs-Rank view consistency (λ=0.1) ──────────────────────────
    X_rw = X_early[:, :half_d]    # (batch, 22) scaled raw half
    X_rk = X_early[:, half_d:]    # (batch, 22) scaled rank half
    Z_rw = X_rw @ B[:half_d, :]   # (batch, r)
    Z_rk = X_rk @ B[half_d:, :]   # (batch, r)
    err_vw = Z_rw - Z_rk
    n_b = float(max(1, X_early.shape[0]))
    L3 = 0.5 * float(np.sum(err_vw ** 2)) / n_b
    gB3 = np.vstack([
        (X_rw.T @ err_vw) / n_b,     # (22, r) raw half
        -(X_rk.T @ err_vw) / n_b,    # (22, r) rank half
    ])   # (44, r)

    # Combine primary + auxiliary
    L_total = L1 + lam_pred * L2 + lam_view * L3
    grad_B = gB1 + lam_pred * gB2 + lam_view * gB3
    grad_W_future = lam_pred * gWf
    grad_W_score = np.zeros_like(W_score)
    grad_W_decode = np.zeros_like(W_decode) if W_decode is not None else np.zeros((B.shape[1], 8))

    # ── Loss 4: Pairwise hinge (coding domain, domain_ssl_weak only) ──────────
    if use_pairwise:
        L4, gB4, gWs = _pairwise_hinge_loss_grad(
            B, W_score, X_early, y_batch, group_keys,
            margin=1.0, max_pairs=2048, rng=rng,
        )
        if L4 > 0.0:
            L_total += lam_pair * L4
            grad_B += lam_pair * gB4
            grad_W_score = lam_pair * gWs

    # ── Loss 5: Proxy reconstruction (coding, when tier-1 targets available) ──
    if X_proxy_targets is not None and W_decode is not None:
        L5, gB5, gWd = _proxy_reconstruction_loss_grad(
            B, W_decode, X_early, X_proxy_targets,
        )
        if L5 > 0.0:
            L_total += lam_proxy * L5
            grad_B += lam_proxy * gB5
            grad_W_decode = lam_proxy * gWd

    return float(L_total), grad_B, grad_W_future, grad_W_score, grad_W_decode


# ─── Domain matrix extraction ─────────────────────────────────────────────────

def _extract_domain_matrix(
    feature_store: list[dict[str, Any]],
    domain: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (X_full[N,4,44], y[N], groups[N]) for the given domain.

    Applies rank-within-group transform per anchor position (same logic as
    _build_anchor_tables). Uses cv_group_keys for group assignment where available.
    """
    n_anchors = len(ANCHOR_POSITIONS)
    parts_x: list[np.ndarray] = []
    parts_y: list[np.ndarray] = []
    parts_g: list[np.ndarray] = []

    for payload in feature_store:
        if payload.get("domain") != domain:
            continue
        tensor = np.asarray(payload["tensor"], dtype=np.float64)
        n = tensor.shape[0]
        if n == 0:
            continue

        y = np.asarray(payload["labels"], dtype=np.int32)
        rank_gk = np.asarray(payload["group_keys"], dtype=object)
        cv_gk = np.asarray(
            payload.get("cv_group_keys", payload["group_keys"]), dtype=object
        )

        # x_raw: (n, n_anchors, 22)
        x_raw = tensor[:, ANCHOR_POS_INDICES, :][:, :, FIXED_FEATURE_INDICES]

        # Rank transform within problem group at each anchor
        x_rank = np.zeros_like(x_raw)
        by_group: dict[Any, list[int]] = {}
        for i, gk in enumerate(rank_gk.tolist()):
            by_group.setdefault(gk, []).append(i)
        for idx_list in by_group.values():
            idx_arr = np.asarray(idx_list)
            for a_idx in range(n_anchors):
                sub = x_raw[idx_arr, a_idx, :]
                x_rank[idx_arr, a_idx, :] = _rank_transform_matrix(sub)

        # raw + rank → (n, 4, 44)
        x_full = np.concatenate([x_raw, x_rank], axis=-1)
        parts_x.append(x_full)
        parts_y.append(y)
        parts_g.append(cv_gk)

    if not parts_x:
        return (
            np.zeros((0, n_anchors, D_FULL), dtype=np.float64),
            np.zeros(0, dtype=np.int32),
            np.asarray([], dtype=object),
        )

    return (
        np.concatenate(parts_x, axis=0),
        np.concatenate(parts_y, axis=0),
        np.concatenate(parts_g, axis=0),
    )


# ─── Optional tier-1 proxy matrix ─────────────────────────────────────────────

def _load_tier1_proxy_matrix(
    feature_store_coding: list[dict[str, Any]],
    cache_roots: Optional[list[str]],
) -> Optional[np.ndarray]:
    """Pre-extract 8 tier-1 coding features aligned to the feature store order.

    Returns (N_coding, 8) float64 array or None if cache roots are unavailable.
    """
    if not cache_roots:
        return None
    try:
        from nad.core.views.reader import CacheReader
        from nad.ops.coding_features import (
            TIER1_FEATURE_NAMES,
            extract_tier1_feature_matrix,
        )
    except ImportError:
        return None

    all_matrices: list[np.ndarray] = []
    for payload in feature_store_coding:
        if payload.get("domain") != "coding":
            continue
        # Find matching cache root by source_name or base_cache_key
        source = str(payload.get("base_cache_key", payload.get("source_name", "")))
        matched_root: Optional[str] = None
        for cr in cache_roots:
            if Path(cr).name in source or source in Path(cr).name:
                matched_root = cr
                break
        if matched_root is None and cache_roots:
            matched_root = cache_roots[0]
        try:
            reader = CacheReader(matched_root)
            # sample run_ids from payload's group_keys (they are run_ids in this context)
            run_ids = np.asarray(payload.get("group_keys", []), dtype=np.int64)
            if len(run_ids) == 0:
                n = int(payload.get("samples", 0))
                all_matrices.append(np.zeros((n, len(TIER1_FEATURE_NAMES))))
                continue
            mat = extract_tier1_feature_matrix(reader, run_ids, verbose=False)
            all_matrices.append(mat)
        except Exception:
            n = int(payload.get("samples", 0))
            all_matrices.append(np.zeros((n, 8)))

    if not all_matrices:
        return None
    result = np.concatenate(all_matrices, axis=0)
    # Standardize proxy targets
    mu = result.mean(axis=0)
    std = result.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (result - mu) / std


# ─── Domain-specific SSL pre-training ─────────────────────────────────────────

def pretrain_domain_ssl_basis(
    X_all: np.ndarray,      # (N, 4, 44) raw+rank features all anchors
    y_all: np.ndarray,      # (N,)  labels (used only if use_pairwise)
    groups_all: np.ndarray, # (N,)  group keys
    domain: str,
    r: int,
    n_epochs: int = 300,
    batch: int = 256,
    lr_max: float = 0.01,
    lr_min: float = 1e-4,
    seed: int = 42,
    smoke: bool = False,
    tau: float = 0.07,
    lam_pred: float = 0.3,
    lam_view: float = 0.1,
    lam_pair: float = 0.5,
    lam_proxy: float = 0.1,
    X_proxy_all: Optional[np.ndarray] = None,  # (N, 8)
    use_pairwise: bool = False,
) -> dict[str, Any]:
    """Pre-train per-domain contrastive SSL basis.

    Returns dict with keys: B, W_future, W_score, W_decode, scaler, r, d, domain.
    """
    N, n_anchors, d = X_all.shape
    if N < 4:
        raise ValueError(f"[ssl/{domain}] need ≥4 runs, got {N}")

    cfg = DOMAIN_CFG[domain]
    if smoke:
        n_epochs = min(n_epochs, 20)
        batch = min(batch, 64)

    # Fit StandardScaler on all flattened data (all runs × all anchors)
    X_flat = X_all.reshape(-1, d)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X_flat)
    X_scaled_all = scaler.transform(X_flat).reshape(N, n_anchors, d)

    # Warm-start B from top-r SVD singular vectors
    try:
        n_components = min(r, d - 1, N * n_anchors - 1)
        svd_init = TruncatedSVD(n_components=n_components, random_state=seed)
        svd_init.fit(X_scaled_all.reshape(-1, d))
        B = svd_init.components_.T.astype(np.float64)   # (d, r_actual)
        r_actual = B.shape[1]
    except Exception:
        r_actual = min(r, d)
        B = np.random.RandomState(seed).randn(d, r_actual).astype(np.float64) * 0.01

    # Auxiliary parameters
    W_future = np.eye(r_actual, dtype=np.float64) * 0.01
    W_score = np.random.RandomState(seed + 1).randn(r_actual, 1).astype(np.float64) * 0.01
    W_decode: Optional[np.ndarray] = None
    if X_proxy_all is not None:
        W_decode = np.random.RandomState(seed + 2).randn(r_actual, 8).astype(np.float64) * 0.01

    # Adam state dicts
    m_B, v_B = np.zeros_like(B), np.zeros_like(B)
    m_Wf, v_Wf = np.zeros_like(W_future), np.zeros_like(W_future)
    m_Ws, v_Ws = np.zeros_like(W_score), np.zeros_like(W_score)
    m_Wd: Optional[np.ndarray] = None
    v_Wd: Optional[np.ndarray] = None
    if W_decode is not None:
        m_Wd, v_Wd = np.zeros_like(W_decode), np.zeros_like(W_decode)

    rng = np.random.RandomState(seed)
    t = 0
    log_interval = max(1, n_epochs // 5)
    batch_size = cfg["batch_size"] if not smoke else min(cfg["batch_size"], 64)
    batch_size = min(batch_size, batch)

    print(
        f"[ssl/{domain}] pretrain r={r_actual} N={N} epochs={n_epochs} "
        f"batch={batch_size} tau={tau} pairwise={use_pairwise} "
        f"proxy={X_proxy_all is not None}"
    )

    for epoch in range(n_epochs):
        lr = _cosine_lr(epoch, n_epochs, lr_max, lr_min)
        perm = rng.permutation(N)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            idx = perm[start: start + batch_size]
            if len(idx) < 4:
                continue

            X_b = X_scaled_all[idx]            # (b, 4, d)
            y_b = y_all[idx]
            g_b = groups_all[idx]
            Xp_b = X_proxy_all[idx] if X_proxy_all is not None else None

            L, gB, gWf, gWs, gWd = _domain_ssl_loss_and_grad(
                B, W_future, W_score, W_decode,
                X_b, y_b, g_b, Xp_b,
                domain, tau, lam_pred, lam_view, lam_pair, lam_proxy,
                cfg, use_pairwise, rng,
            )
            epoch_loss += L
            n_batches += 1

            # Gradient clipping at Frobenius norm = 1.0
            for g_arr in (gB, gWf, gWs):
                fn = float(np.sqrt(np.sum(g_arr ** 2)))
                if fn > 1.0:
                    g_arr *= 1.0 / fn

            t += 1
            B, m_B, v_B = _adam_update(B, gB, m_B, v_B, t, lr=lr)
            W_future, m_Wf, v_Wf = _adam_update(W_future, gWf, m_Wf, v_Wf, t, lr=lr)
            W_score, m_Ws, v_Ws = _adam_update(W_score, gWs, m_Ws, v_Ws, t, lr=lr)
            if W_decode is not None and m_Wd is not None and v_Wd is not None:
                fn_d = float(np.sqrt(np.sum(gWd ** 2)))
                if fn_d > 1.0:
                    gWd *= 1.0 / fn_d
                W_decode, m_Wd, v_Wd = _adam_update(W_decode, gWd, m_Wd, v_Wd, t, lr=lr)

        if (epoch + 1) % log_interval == 0:
            avg = epoch_loss / max(1, n_batches)
            print(f"[ssl/{domain}] epoch={epoch + 1}/{n_epochs} "
                  f"avg_loss={avg:.5f} lr={lr:.6f}")

    # Collapse check: crude discriminative AUROC using mean-difference direction at 70%
    a70 = 2   # ANCHOR_POSITIONS index for 70%
    if N > 10 and np.unique(y_all).shape[0] >= 2:
        X_70 = X_scaled_all[:, a70, :]   # (N, d)
        Z_70 = X_70 @ B                   # (N, r)
        pos_mask = y_all == 1
        neg_mask = y_all == 0
        if pos_mask.any() and neg_mask.any():
            z_pos_mean = Z_70[pos_mask].mean(axis=0)
            z_neg_mean = Z_70[neg_mask].mean(axis=0)
            direction = z_pos_mean - z_neg_mean
            dn = float(np.sqrt(np.sum(direction ** 2))) + 1e-8
            train_scores = Z_70 @ (direction / dn)
            auc = _auroc(train_scores, y_all)
            if auc < 0.52:
                print(f"[warn/{domain}] collapse: train AUROC at 70% = {auc:.4f} < 0.52")
            else:
                print(f"[ssl/{domain}] train AUROC at 70% = {auc:.4f} (ok)")

    return {
        "B": B,
        "W_future": W_future,
        "W_score": W_score,
        "W_decode": W_decode,
        "scaler": scaler,
        "r": r_actual,
        "d": d,
        "domain": domain,
        "final_loss": epoch_loss / max(1, n_batches),
    }


# ─── Domain SSL study ──────────────────────────────────────────────────────────

def _run_domain_ssl_study(
    train_by_domain: dict[str, list[dict[str, Any]]],
    holdout_by_domain: dict[str, list[dict[str, Any]]],
    domain_bundles: dict[str, dict[int, dict[str, Any]]],   # domain → r → bundle
    weak_bundles: dict[str, dict[int, dict[str, Any]]],     # domain → r → bundle (weak)
    shared_ssl_bundles: dict[int, dict[str, Any]],          # r → old shared bundle
    supervised_bundle: Optional[dict[str, Any]],
    label_fractions: tuple[float, ...],
    ssl_ranks: tuple[int, ...],
    domains: tuple[str, ...],
    seed: int = 42,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    """Run the 5-condition label-efficiency study per domain."""
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
              f"pairwise={use_pairwise} n_holdout={len(holdout_store)}")

        for fraction in active_fractions:
            selected = _sample_labeled_problems(train_store, domain, fraction, seed=seed)
            labeled_store = _subset_store_by_problems(train_store, domain, selected)
            if not labeled_store:
                print(f"[study] skip domain={domain} frac={fraction:.0%}: empty")
                continue

            n_labeled = len(selected)
            tables = _build_anchor_tables(labeled_store, domain=domain)
            print(f"[study] domain={domain} frac={fraction:.0%} n_labeled={n_labeled}")

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
                except Exception:
                    pass

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
                    x_rw = tbl.get("x_raw", np.zeros((0, N_FEATURES)))
                    x_rk = tbl.get("x_rank", np.zeros((0, N_FEATURES)))
                    y = tbl.get("y", np.zeros(0, dtype=np.int32))
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
                    score_fn = _make_frozen_svd_score_fn(
                        frozen_route_models, frozen_heads, domain
                    )
                    res = _eval(holdout_store, score_fn, f"frozen_svd/{domain}")
                    row = _agg_to_row(
                        res["aggregate"], "frozen_svd", domain, fraction, n_labeled, -1
                    )
                    results.append(row)
                    print(f"[result] frozen_svd domain={domain} frac={fraction:.0%} "
                          f"auroc={row['auroc']:.4f}")

            # ── shared_ssl_r16 (old shared basis) ─────────────────────────
            if 16 in shared_ssl_bundles:
                bundle_s16 = shared_ssl_bundles[16]
                transform_fn_s16 = _make_transform_fn(bundle_s16["B"], bundle_s16["scaler"])
                ssl_heads_s16: dict[int, LogisticRegression] = {}
                for a_idx, tbl in tables.items():
                    x_rw, x_rk, y, grp = (
                        tbl["x_raw"], tbl["x_rank"], tbl["y"], tbl["groups"]
                    )
                    if x_rw.shape[0] < 4 or np.unique(y).shape[0] < 2:
                        continue
                    c_search = fraction >= 0.10
                    anchor_pct = float(ANCHOR_POSITIONS[a_idx])
                    if use_pairwise and anchor_pct >= 0.70:
                        clf = _fit_pairwise_head(
                            x_rw, x_rk, y, grp, transform_fn_s16, random_state=seed
                        )
                    else:
                        clf = _fit_pointwise_head(
                            x_rw, x_rk, y, grp, transform_fn_s16,
                            c_search=c_search, random_state=seed,
                        )
                    if clf is not None:
                        ssl_heads_s16[a_idx] = clf

                if ssl_heads_s16:
                    score_fn = _make_ssl_score_fn(
                        bundle_s16, ssl_heads_s16, domain, is_pairwise=use_pairwise
                    )
                    res = _eval(holdout_store, score_fn, f"shared_ssl_r16/{domain}")
                    row = _agg_to_row(
                        res["aggregate"], "shared_ssl_r16", domain, fraction, n_labeled, 16
                    )
                    results.append(row)
                    print(f"[result] shared_ssl_r16 domain={domain} frac={fraction:.0%} "
                          f"auroc={row['auroc']:.4f}")

            # ── domain_ssl and domain_ssl_weak (per r) ────────────────────
            for r_ssl in ssl_ranks:
                for condition, bundles_map in [
                    ("domain_ssl", domain_bundles),
                    ("domain_ssl_weak", weak_bundles),
                ]:
                    # domain_ssl_weak only differs for coding; reuse domain_ssl for others
                    if condition == "domain_ssl_weak" and domain != "coding":
                        continue  # already covered by domain_ssl for non-coding

                    dom_r_bundles = bundles_map.get(domain, {})
                    bundle = dom_r_bundles.get(r_ssl)
                    if bundle is None:
                        continue

                    transform_fn = _make_transform_fn(bundle["B"], bundle["scaler"])
                    ssl_heads: dict[int, LogisticRegression] = {}

                    for a_idx, tbl in tables.items():
                        x_rw, x_rk, y, grp = (
                            tbl["x_raw"], tbl["x_rank"], tbl["y"], tbl["groups"]
                        )
                        if x_rw.shape[0] < 4 or np.unique(y).shape[0] < 2:
                            continue
                        c_search = fraction >= 0.10
                        anchor_pct = float(ANCHOR_POSITIONS[a_idx])

                        if use_pairwise and anchor_pct >= 0.70:
                            clf = _fit_pairwise_head(
                                x_rw, x_rk, y, grp, transform_fn, random_state=seed
                            )
                        else:
                            clf = _fit_pointwise_head(
                                x_rw, x_rk, y, grp, transform_fn,
                                c_search=c_search, random_state=seed,
                            )
                        if clf is not None:
                            ssl_heads[a_idx] = clf

                    if ssl_heads:
                        score_fn = _make_ssl_score_fn(
                            bundle, ssl_heads, domain, is_pairwise=use_pairwise
                        )
                        res = _eval(
                            holdout_store, score_fn, f"{condition}_r{r_ssl}/{domain}"
                        )
                        row = _agg_to_row(
                            res["aggregate"], condition, domain, fraction, n_labeled, r_ssl
                        )
                        results.append(row)
                        print(f"[result] {condition} r={r_ssl} domain={domain} "
                              f"frac={fraction:.0%} auroc={row['auroc']:.4f}")

    return results


# ─── Markdown report ──────────────────────────────────────────────────────────

def _write_markdown_domain(
    results: list[dict[str, Any]],
    out_path: Path,
    n_epochs: int,
    tau: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Domain-Specific Contrastive SSL for SVDomain",
        "",
        f"Generated: {_now_utc()}",
        "",
        "## Training Configuration",
        "",
        f"- Epochs: {n_epochs}  |  Temperature τ: {tau}",
        "- Optimizer: Adam with cosine LR annealing (lr_max=0.01, lr_min=1e-4)",
        "- Gradient clipping: Frobenius norm = 1.0",
        "- Warm-start: top-r SVD singular vectors per domain",
        "",
        "## Results",
        "",
        "| Domain | Condition | SSL r | Label % | AUROC | Top1 Acc | AUC SelAcc |",
        "|--------|-----------|:-----:|--------:|------:|---------:|-----------:|",
    ]

    sorted_rows = sorted(
        results,
        key=lambda r: (
            r["domain"], r["condition"], r["ssl_rank"], r["label_fraction"]
        ),
    )
    for row in sorted_rows:
        lines.append(
            f"| {row['domain']} | {row['condition']} | {row['ssl_rank']} "
            f"| {row['label_fraction']:.0%} "
            f"| {row.get('auroc', float('nan')):.4f} "
            f"| {row.get('top1_acc', float('nan')):.4f} "
            f"| {row.get('auc_of_selacc', float('nan')):.4f} |"
        )

    lines += [
        "",
        "## Research Questions",
        "",
        "### Q1: Was the failure due to objective, cross-domain mixing, or both?",
        "Compare `domain_ssl` vs `shared_ssl_r16` at 100% labels per domain.",
        "If `domain_ssl` substantially outperforms → cross-domain mixing was primary cause.",
        "If both still trail `no_svd_lr` → linear-basis capacity is the primary cause.",
        "",
        "### Q2: Does contrastive/domain-specific SSL help at low labels?",
        "Compare `domain_ssl_r{4,8,16}` vs `no_svd_lr` vs `frozen_svd` at 1–10%.",
        "If `domain_ssl` ≥ `frozen_svd` at ≤5% labels → SSL improves label efficiency.",
        "",
        "### Q3: Does coding benefit from pairwise weak supervision?",
        "Compare `domain_ssl_weak` vs `domain_ssl` on coding across all label fractions.",
        "Positive delta → pairwise hinge during pre-training helps.",
        "",
        "### Q4: Is the per-domain basis more useful than the shared basis?",
        "AUROC gap `domain_ssl_r16 − shared_ssl_r16` at 100% labels per domain.",
        "Positive gap → per-domain training improves basis quality.",
        "",
        "## Conditions",
        "",
        "| Condition | Description |",
        "|-----------|-------------|",
        "| `domain_ssl` | Per-domain NT-Xent basis + supervised head |",
        "| `domain_ssl_weak` | Same + pairwise hinge during coding pre-training |",
        "| `shared_ssl_r16` | Old shared SSL basis (r=16) loaded from pkl |",
        "| `no_svd_lr` | StandardScaler → LR, no dimensionality reduction |",
        "| `frozen_svd` | Frozen supervised SVD + new LR head |",
        "",
        "## Domain Configuration",
        "",
        "| Domain | NT-Xent views (early, late) | Pairwise hinge | Batch |",
        "|--------|----------------------------|----------------|-------|",
        "| math    | 10%, 100% | No  | 256 |",
        "| science | 40%, 100% | No  | 256 |",
        "| coding  | 40%, 70%  | Yes (domain_ssl_weak only) | 384 |",
        "",
        "## SSL Losses",
        "",
        "1. `L_ntxent` (primary): symmetric NT-Xent, τ=0.07, positives = same run two anchors",
        "2. `L_pred` (λ=0.3): MSE future-anchor prediction Z_early @ W_future ≈ Z_late",
        "3. `L_view` (λ=0.1): raw-vs-rank view consistency ||X_rw@B[:22] − X_rk@B[22:]||²",
        "4. `L_pair` (λ=0.5, coding weak): pairwise hinge on 40% anchor within problem groups",
        "5. `L_proxy` (λ=0.1, coding, optional): MSE decode tier-1 features from Z",
    ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[out] markdown → {_display_path(out_path)}")


# ─── Path utilities ────────────────────────────────────────────────────────────

def _resolve_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Domain-specific contrastive SSL label-efficiency study"
    )
    # Data sources
    ap.add_argument("--main-cache-root", default="MUI_HUB/cache")
    ap.add_argument("--extra-cache-root", default=None)
    ap.add_argument("--test-cache-root", default="MUI_HUB/cache_test")
    ap.add_argument("--prebuilt-cache-dir", default=None,
                    help="Pre-built feature pkl files (fast path)")
    ap.add_argument("--shared-ssl-pkl", default=str(DEFAULT_SHARED_SSL_PKL),
                    help="Old shared SSL bundles pkl (for shared_ssl_r16 baseline)")
    ap.add_argument("--supervised-bundle", default=str(DEFAULT_BUNDLE_PATH),
                    help="Supervised SVD bundle for frozen_svd condition")
    ap.add_argument("--coding-cache-roots", nargs="*", default=None,
                    help="Cache roots for tier-1 proxy feature extraction (optional)")

    # Output
    ap.add_argument("--out-dir", default="results/cache/domain_ssl")
    ap.add_argument("--out-csv", default="results/tables/domain_specific_ssl.csv")
    ap.add_argument("--out-doc", default="docs/DOMAIN_SPECIFIC_SSL.md")

    # SSL hyperparameters
    ap.add_argument("--ssl-ranks", nargs="+", type=int, default=list(DEFAULT_SSL_RANKS))
    ap.add_argument("--n-epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--tau", type=float, default=0.07, help="NT-Xent temperature")
    ap.add_argument("--lam-pred", type=float, default=0.3, help="Future-prediction weight")
    ap.add_argument("--lam-view", type=float, default=0.1, help="View-consistency weight")
    ap.add_argument("--lam-pair", type=float, default=0.5, help="Pairwise hinge weight")
    ap.add_argument("--lam-proxy", type=float, default=0.1, help="Proxy reconstruction weight")
    ap.add_argument("--lr-max", type=float, default=0.01, help="Peak cosine LR")
    ap.add_argument("--lr-min", type=float, default=1e-4, help="Floor cosine LR")
    ap.add_argument("--hinge-margin", type=float, default=1.0)
    ap.add_argument("--max-pairs-per-batch", type=int, default=2048)
    ap.add_argument("--no-proxy", action="store_true", help="Disable proxy reconstruction")

    # Study configuration
    ap.add_argument("--label-fractions", nargs="+", type=float,
                    default=list(LABEL_FRACTIONS))
    ap.add_argument("--domains", nargs="+", default=["math", "science", "coding"])
    ap.add_argument("--holdout-split", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--feature-workers", type=int, default=8)
    ap.add_argument("--feature-chunk-problems", type=int,
                    default=DEFAULT_FEATURE_CHUNK_PROBLEMS)
    ap.add_argument("--max-problems-per-cache", type=int, default=0)

    # Cache refresh
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--refresh-ssl-cache", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="Quick smoke test: fewer epochs/problems/fractions")

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
        prebuilt_dir = _resolve_path(args.prebuilt_cache_dir)
        print(f"[prebuilt] loading from {_display_path(prebuilt_dir)}")
        full_store: list[dict[str, Any]] = _load_prebuilt_stores(prebuilt_dir)
        if not full_store:
            raise RuntimeError(f"No usable feature stores in {prebuilt_dir}")
        with feat_cache.open("wb") as fh:
            pickle.dump(full_store, fh, protocol=pickle.HIGHEST_PROTOCOL)
        holdout_store: list[dict[str, Any]] = []

    elif feat_cache.exists() and not args.refresh_feature_cache:
        print(f"[load] feature store ← {_display_path(feat_cache)}")
        with feat_cache.open("rb") as fh:
            full_store = pickle.load(fh)
        holdout_store = []

    else:
        main_root = _resolve_path(args.main_cache_root)
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
        holdout_store = []

    # ── Holdout ───────────────────────────────────────────────────────────────
    if not holdout_store and not args.prebuilt_cache_dir:
        holdout_cache_pkl = out_dir / "holdout_store.pkl"
        test_root = _resolve_path(args.test_cache_root)
        if holdout_cache_pkl.exists() and not args.refresh_feature_cache:
            with holdout_cache_pkl.open("rb") as fh:
                holdout_store = pickle.load(fh)
        elif test_root.exists():
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

    # ── Train/holdout split ───────────────────────────────────────────────────
    if holdout_store:
        train_store = full_store
        print(f"[split] using separate holdout ({len(holdout_store)} payloads)")
    else:
        print("[split] no separate holdout — splitting 85/15")
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
        n_tr_s = sum(int(p.get("samples", 0)) for p in train_by_domain[dom])
        n_ho_s = sum(int(p.get("samples", 0)) for p in holdout_by_domain[dom])
        print(f"[split] domain={dom} train={len(train_by_domain[dom])} payloads "
              f"({n_tr_s} samples) holdout={len(holdout_by_domain[dom])} ({n_ho_s} samples)")

    # ── Optional tier-1 proxy extraction ──────────────────────────────────────
    coding_proxy_matrix: Optional[np.ndarray] = None
    if not args.no_proxy and args.coding_cache_roots:
        print("[proxy] extracting tier-1 coding features...")
        coding_proxy_matrix = _load_tier1_proxy_matrix(
            train_by_domain.get("coding", []),
            args.coding_cache_roots,
        )
        if coding_proxy_matrix is not None:
            print(f"[proxy] extracted shape={coding_proxy_matrix.shape}")

    # ── Domain-specific SSL pre-training ──────────────────────────────────────
    ssl_cache = out_dir / "domain_ssl_bundles.pkl"
    domain_bundles: dict[str, dict[int, dict[str, Any]]] = {}
    weak_bundles: dict[str, dict[int, dict[str, Any]]] = {}

    if ssl_cache.exists() and not args.refresh_ssl_cache:
        print(f"[load] domain SSL bundles ← {_display_path(ssl_cache)}")
        with ssl_cache.open("rb") as fh:
            saved = pickle.load(fh)
        domain_bundles = saved.get("domain_bundles", {})
        weak_bundles = saved.get("weak_bundles", {})
    else:
        for domain in domains:
            train_dom = train_by_domain.get(domain, [])
            if not train_dom:
                print(f"[ssl] skip domain={domain}: no training data")
                continue

            X_all, y_all, groups_all = _extract_domain_matrix(train_dom, domain)
            if X_all.shape[0] < 4:
                print(f"[ssl] skip domain={domain}: only {X_all.shape[0]} runs")
                continue

            print(f"\n[ssl] domain={domain} N={X_all.shape[0]} "
                  f"label_pos={np.sum(y_all == 1)} label_neg={np.sum(y_all == 0)}")

            # Proxy matrix slice (coding only, aligned to domain matrix order)
            dom_proxy = coding_proxy_matrix if domain == "coding" else None

            domain_bundles.setdefault(domain, {})
            weak_bundles.setdefault(domain, {})

            for r_ssl in ssl_ranks:
                # ── domain_ssl: no pairwise hinge ─────────────────────────
                bundle_key = out_dir / f"bundle_{domain}_r{r_ssl}.pkl"
                if bundle_key.exists() and not args.refresh_ssl_cache:
                    print(f"[ssl] load existing bundle {bundle_key.name}")
                    with bundle_key.open("rb") as fh:
                        domain_bundles[domain][r_ssl] = pickle.load(fh)
                else:
                    b = pretrain_domain_ssl_basis(
                        X_all, y_all, groups_all,
                        domain=domain, r=r_ssl,
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
                        X_proxy_all=dom_proxy,
                        use_pairwise=False,
                    )
                    domain_bundles[domain][r_ssl] = b
                    with bundle_key.open("wb") as fh:
                        pickle.dump(b, fh, protocol=pickle.HIGHEST_PROTOCOL)

                # ── domain_ssl_weak: pairwise hinge for coding only ────────
                if domain == "coding":
                    weak_key = out_dir / f"bundle_{domain}_r{r_ssl}_weak.pkl"
                    if weak_key.exists() and not args.refresh_ssl_cache:
                        print(f"[ssl] load existing weak bundle {weak_key.name}")
                        with weak_key.open("rb") as fh:
                            weak_bundles[domain][r_ssl] = pickle.load(fh)
                    else:
                        bw = pretrain_domain_ssl_basis(
                            X_all, y_all, groups_all,
                            domain=domain, r=r_ssl,
                            n_epochs=int(args.n_epochs),
                            batch=int(args.batch_size),
                            lr_max=float(args.lr_max),
                            lr_min=float(args.lr_min),
                            seed=int(args.split_seed) + 1000,
                            smoke=bool(args.smoke),
                            tau=float(args.tau),
                            lam_pred=float(args.lam_pred),
                            lam_view=float(args.lam_view),
                            lam_pair=float(args.lam_pair),
                            lam_proxy=float(args.lam_proxy),
                            X_proxy_all=dom_proxy,
                            use_pairwise=True,
                        )
                        weak_bundles[domain][r_ssl] = bw
                        with weak_key.open("wb") as fh:
                            pickle.dump(bw, fh, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    # For math/science, weak == no-weak (no pairwise)
                    weak_bundles.setdefault(domain, {})[r_ssl] = (
                        domain_bundles[domain][r_ssl]
                    )

        # Save combined bundle dict
        with ssl_cache.open("wb") as fh:
            pickle.dump(
                {"domain_bundles": domain_bundles, "weak_bundles": weak_bundles},
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"[save] domain SSL bundles → {_display_path(ssl_cache)}")

    # ── Load shared SSL bundles (old pkl for shared_ssl_r16 baseline) ─────────
    shared_ssl_bundles: dict[int, dict[str, Any]] = {}
    shared_pkl = _resolve_path(str(args.shared_ssl_pkl))
    if shared_pkl.exists():
        try:
            with shared_pkl.open("rb") as fh:
                shared_ssl_bundles = pickle.load(fh)
            print(f"[load] shared SSL bundles ← {_display_path(shared_pkl)} "
                  f"ranks={list(shared_ssl_bundles.keys())}")
        except Exception as e:
            print(f"[warn] could not load shared SSL pkl: {e}")
    else:
        print(f"[warn] shared SSL pkl not found: {shared_pkl} — skipping shared_ssl_r16")

    # ── Load supervised bundle (for frozen_svd) ───────────────────────────────
    supervised_bundle: Optional[dict[str, Any]] = None
    bundle_path = _resolve_path(str(args.supervised_bundle))
    if bundle_path.exists():
        try:
            supervised_bundle = load_earlystop_svd_bundle(bundle_path)
            print(f"[load] supervised bundle ← {_display_path(bundle_path)}")
        except Exception as e:
            print(f"[warn] could not load supervised bundle: {e}")
    else:
        print(f"[warn] supervised bundle not found: {bundle_path} — skipping frozen_svd")

    # ── Label-efficiency study ────────────────────────────────────────────────
    results = _run_domain_ssl_study(
        train_by_domain=train_by_domain,
        holdout_by_domain=holdout_by_domain,
        domain_bundles=domain_bundles,
        weak_bundles=weak_bundles,
        shared_ssl_bundles=shared_ssl_bundles,
        supervised_bundle=supervised_bundle,
        label_fractions=label_fractions,
        ssl_ranks=ssl_ranks,
        domains=domains,
        seed=int(args.split_seed),
        smoke=bool(args.smoke),
    )

    # ── Write outputs ─────────────────────────────────────────────────────────
    out_csv = _resolve_path(args.out_csv)
    out_doc = _resolve_path(args.out_doc)
    _write_csv(results, out_csv)
    _write_markdown_domain(results, out_doc, n_epochs=args.n_epochs, tau=args.tau)

    print(f"\n[done] {len(results)} result rows")
    print(f"  CSV:      {_display_path(out_csv)}")
    print(f"  Markdown: {_display_path(out_doc)}")


if __name__ == "__main__":
    main()
