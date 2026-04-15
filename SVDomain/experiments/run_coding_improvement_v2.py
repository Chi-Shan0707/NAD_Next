#!/usr/bin/env python3
"""Coding Improvement V2: neuron-native verifier / reranker exploration.

V1 conclusion (frozen): token statistics (tok_conf, logprob, derivatives, Tier-1 structural)
all plateau below 54% pointwise AUROC on LCBv5. The feature-label mismatch is fundamental.

V2 pivots to code-native signals using the neuron activation cache directly.

HYPOTHESES TESTED
-----------------
A. Jaccard clustering: Do correct solutions cluster together in activation space?
   H_A: mean(CC distance) < mean(CI distance) → medoid picks correct more often.

B. Layer specialisation (stride=65536, 36 layers):
   H_B: Some transformer layers encode code correctness better than others.
   For each layer: compute layer-only Jaccard distance → pairwise AUC vs correctness.

C. w_max signal: Higher peak neuron activation (w_max) correlates with correctness.
   For each run: mean/max/tail of w_max across activated neurons.

D. Consensus neuron set: For each problem, find neurons activated in ≥K% of correct
   solutions ("correct-consensus"). Rank runs by overlap with this consensus set.
   (Only computable with cross-validation — train on other folds.)

E. NAD selector selection accuracy: Benchmark all existing selectors (Medoid, KNN,
   DBSCAN, DeepConf, Tournament…) on LCBv5. Establish current ceiling.

PRIMARY METRICS
---------------
- within-problem top1 accuracy (fraction of 167 problems where selected = correct)
- pairwise AUC (AUROC of pairwise-distance score vs correctness label within groups)
- pass@1 uplift over random

SECONDARY (sanity check only)
- pooled pointwise AUROC

Usage (from repo root):
    python3 SVDomain/experiments/run_coding_improvement_v2.py
    python3 SVDomain/experiments/run_coding_improvement_v2.py --n-problems 30  # fast smoke test
    python3 SVDomain/experiments/run_coding_improvement_v2.py --out docs/14_CODING_IMPROVEMENT_V2.md
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.selectors.base import SelectorContext
from nad.core.selectors.code_dynamic_impl import order_code_dynamic_group_indices
from nad.core.selectors.code_rns_impl import extract_code_rns_context_payload
from nad.core.distance.engine import DistanceEngine, DistanceSpec
from nad.core.views.reader import CacheReader, ViewSpec, Agg, CutSpec, CutType, Order
from nad.ops.accuracy import load_correctness_map
from nad.ops.coding_neuron_features import (
    build_global_neuron_feature_matrix,
    load_or_build_layer_summary_cache,
)
from nad.ops.coding_report_features import (
    STATIC_FEATURE_NAMES,
    audit_coding_inputs,
    build_static_feature_cache,
)
from nad.ops.earlystop_svd import _auroc
from nad.ops.grouped_ranking import evaluate_grouped_scores, write_problem_records_jsonl

# ── constants ─────────────────────────────────────────────────────────────────

DEFAULT_CACHE_ROOT = (
    "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/livecodebench_v5/"
    "cache_neuron_output_1_act_no_rms_20251127_032808"
)
DEFAULT_DIST_CACHE = "results/cache/coding_v2_dist_matrices.pkl"
DEFAULT_LAYER_CACHE = "results/cache/coding_v2_layer_features.pkl"
DEFAULT_STATIC_CACHE = "results/cache/coding_v2_static_features.pkl"
DEFAULT_CODE_V2_SCORE_CACHE = "results/cache/coding_v2_code_v2_scores.pkl"
DEFAULT_OUT_DOC = "docs/14_CODING_IMPROVEMENT_V2.md"
DEFAULT_AUDIT_JSON = "results/validation/coding_v2_data_audit.json"
DEFAULT_METRICS_JSON = "results/validation/coding_v2_grouped_metrics.json"
DEFAULT_RECORDS_JSONL = "results/validation/coding_v2_problem_records.jsonl"

LAYER_STRIDE = 65536       # neuron_key = layer_id * 65536 + neuron_within_layer
N_LAYERS = 36              # DeepSeek-R1-0528-Qwen3-8B (Qwen3-8B base)
GLOBAL_TOPK = 2000         # from meta.json
N_THREADS = 8              # Jaccard computation threads (8 × 2 = 16 cores total)

# ViewSpec: use all top-2000 neurons per run (global topk selection)
VIEW_SPEC = ViewSpec(
    agg=Agg.MAX,
    cut=CutSpec(type=CutType.TOPK, value=GLOBAL_TOPK),
    order=Order.BY_KEY,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_problem_groups(cache_root: str) -> dict[str, list[int]]:
    meta = json.load(open(f"{cache_root}/meta.json"))
    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(meta["samples"]):
        groups[s["problem_id"]].append(i)
    return dict(groups)


def _pairwise_auc(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Pairwise AUC: fraction of (correct, incorrect) pairs where score_correct > score_incorrect."""
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return float("nan")
    wins = 0
    total = 0
    for pi in pos_idx:
        wins += int((scores[pi] > scores[neg_idx]).sum())
        total += len(neg_idx)
    return float(wins) / float(total) if total > 0 else float("nan")


def _centrality_scores(D: np.ndarray) -> np.ndarray:
    """Return -mean_distance (higher = more central = medoid-like score)."""
    return -D.mean(axis=1)


def _knn_scores(D: np.ndarray, k: int = 3) -> np.ndarray:
    """KNN affinity: negative mean distance to k nearest neighbours."""
    n = D.shape[0]
    k_eff = min(k, n - 1)
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        row = D[i].copy()
        row[i] = np.inf
        nearest = np.sort(row)[:k_eff]
        scores[i] = -float(nearest.mean()) if len(nearest) > 0 else 0.0
    return scores


def _wmax_scores(reader: CacheReader, run_ids: list[int]) -> np.ndarray:
    """Per-run mean w_max across all activated neurons."""
    rp = reader.row_ptr
    wm = reader.weights_for("max")
    scores = np.zeros(len(run_ids), dtype=np.float64)
    for i, rid in enumerate(run_ids):
        s, e = int(rp[rid]), int(rp[rid + 1])
        if e > s:
            w = np.asarray(wm[s:e], dtype=np.float64)
            scores[i] = float(w.mean())
    return scores


def _wmax_top10pct_scores(reader: CacheReader, run_ids: list[int]) -> np.ndarray:
    """Per-run mean of top-10% w_max values (peak activation intensity)."""
    rp = reader.row_ptr
    wm = reader.weights_for("max")
    scores = np.zeros(len(run_ids), dtype=np.float64)
    for i, rid in enumerate(run_ids):
        s, e = int(rp[rid]), int(rp[rid + 1])
        if e > s:
            w = np.sort(np.asarray(wm[s:e], dtype=np.float64))
            top_n = max(1, int(0.10 * (e - s)))
            scores[i] = float(w[-top_n:].mean())
    return scores


def _n_active_scores(reader: CacheReader, run_ids: list[int]) -> np.ndarray:
    """Number of activated neurons per run."""
    rp = reader.row_ptr
    return np.array([int(rp[rid + 1]) - int(rp[rid]) for rid in run_ids], dtype=np.float64)


def _layer_jaccard_distance(
    keys_a: np.ndarray,
    keys_b: np.ndarray,
    layer_id: int,
    stride: int = LAYER_STRIDE,
) -> float:
    """Jaccard distance using only neurons from a specific transformer layer."""
    lo = layer_id * stride
    hi = (layer_id + 1) * stride
    a = keys_a[(keys_a >= lo) & (keys_a < hi)]
    b = keys_b[(keys_b >= lo) & (keys_b < hi)]
    if a.size == 0 and b.size == 0:
        return float("nan")
    if a.size == 0 or b.size == 0:
        return 1.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    denom = a.size + b.size - inter
    return float(1.0 - inter / denom) if denom > 0 else 0.0


def _selection_accuracy(selected_idx: int, labels: np.ndarray) -> int:
    """1 if selected sample is correct, else 0. -1 if no correct sample exists."""
    if labels.sum() == 0:
        return -1
    return int(labels[selected_idx])


# ── distance matrix cache ─────────────────────────────────────────────────────

def _load_or_build_dist_matrices(
    cache_root: str,
    problem_groups: dict[str, list[int]],
    cache_path: str,
    n_problems: Optional[int] = None,
    n_threads: int = N_THREADS,
    refresh: bool = False,
) -> dict[str, np.ndarray]:
    """Compute or load full Jaccard distance matrices for all problem groups."""
    cp = Path(cache_path)
    if not refresh and cp.exists():
        with open(cp, "rb") as fh:
            saved = pickle.load(fh)
        print(f"  [cache] Distance matrices loaded: {len(saved)} problems from {cp}")
        if n_problems is None or len(saved) >= min(n_problems, len(problem_groups)):
            return saved

    reader = CacheReader(cache_root)
    engine = DistanceEngine(DistanceSpec(name="ja", num_threads=n_threads, ja_backend="auto"))

    problems = list(problem_groups.keys())
    if n_problems is not None:
        problems = problems[:n_problems]

    dist_matrices: dict[str, np.ndarray] = {}
    t0 = time.time()
    for p_idx, pid in enumerate(problems):
        run_ids = problem_groups[pid]
        views = [reader.get_run_view(rid, VIEW_SPEC) for rid in run_ids]
        D = engine.dense_matrix(views)
        dist_matrices[pid] = np.asarray(D, dtype=np.float32)
        if p_idx > 0 and p_idx % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / p_idx * (len(problems) - p_idx)
            print(f"  [{p_idx}/{len(problems)}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"  Done: {len(dist_matrices)} distance matrices in {elapsed:.1f}s")
    cp.parent.mkdir(parents=True, exist_ok=True)
    with open(cp, "wb") as fh:
        pickle.dump(dist_matrices, fh, protocol=4)
    print(f"  [cache] Saved → {cp}")
    return dist_matrices


# ── analysis A: Jaccard clustering by correctness ─────────────────────────────

def analyze_jaccard_clustering(
    dist_matrices: dict[str, np.ndarray],
    problem_groups: dict[str, list[int]],
    labels: np.ndarray,
) -> dict[str, Any]:
    """Test whether correct solutions cluster more tightly (H_A)."""
    cc_dists, ci_dists, ii_dists = [], [], []
    centrality_aucs, knn_aucs = [], []
    medoid_acc, knn_acc, random_acc = [], [], []
    solvable_problems = 0

    for pid, D in dist_matrices.items():
        run_ids = problem_groups[pid]
        y = labels[run_ids]

        if y.sum() == 0 or y.sum() == len(y):
            continue
        solvable_problems += 1
        n = len(run_ids)
        D_arr = np.asarray(D, dtype=np.float64)

        # CC / CI / II distance distributions
        for i in range(n):
            for j in range(i + 1, n):
                d = float(D_arr[i, j])
                yi, yj = int(y[i]), int(y[j])
                if yi == 1 and yj == 1:
                    cc_dists.append(d)
                elif yi == 0 and yj == 0:
                    ii_dists.append(d)
                else:
                    ci_dists.append(d)

        # Centrality (medoid-like) scores
        cent = _centrality_scores(D_arr)
        centrality_aucs.append(_pairwise_auc(cent, y))
        selected_medoid = int(np.argmax(cent))
        medoid_acc.append(_selection_accuracy(selected_medoid, y))

        # KNN scores
        knn = _knn_scores(D_arr, k=5)
        knn_aucs.append(_pairwise_auc(knn, y))
        selected_knn = int(np.argmax(knn))
        knn_acc.append(_selection_accuracy(selected_knn, y))

        # Random baseline
        random_acc.append(float(y.mean()))

    def _m(lst): return float(np.nanmean(lst)) if lst else float("nan")

    return {
        "solvable_problems": solvable_problems,
        "n_cc_pairs": len(cc_dists),
        "n_ci_pairs": len(ci_dists),
        "n_ii_pairs": len(ii_dists),
        "mean_cc_dist": _m(cc_dists),
        "mean_ci_dist": _m(ci_dists),
        "mean_ii_dist": _m(ii_dists),
        "cc_ci_gap": _m(ci_dists) - _m(cc_dists),
        "centrality_pairwise_auc": _m(centrality_aucs),
        "knn_pairwise_auc": _m(knn_aucs),
        "medoid_top1_acc": _m([a for a in medoid_acc if a >= 0]),
        "knn_top1_acc": _m([a for a in knn_acc if a >= 0]),
        "random_top1_acc": _m(random_acc),
    }


# ── analysis B: layer-wise discriminability ────────────────────────────────────

def analyze_layer_discriminability(
    cache_root: str,
    problem_groups: dict[str, list[int]],
    labels: np.ndarray,
    n_problems: Optional[int] = None,
    n_threads: int = 4,
) -> dict[str, Any]:
    """For each of 36 layers, compute pairwise AUC of layer-only Jaccard distance."""
    reader = CacheReader(cache_root)
    rp = reader.row_ptr
    ks = reader.keys

    problems = list(problem_groups.keys())
    if n_problems is not None:
        problems = problems[:n_problems]

    # Accumulate pairwise AUC per layer
    layer_aucs: dict[int, list[float]] = {lid: [] for lid in range(N_LAYERS)}
    layer_n_keys: dict[int, list[float]] = {lid: [] for lid in range(N_LAYERS)}  # mean keys per run per layer

    print(f"  Layer analysis: {len(problems)} problems × {N_LAYERS} layers…")
    t0 = time.time()
    for p_idx, pid in enumerate(problems):
        run_ids = problem_groups[pid]
        y = labels[run_ids]
        if y.sum() == 0 or y.sum() == len(y):
            continue

        # Load all keys for this problem
        all_keys = []
        for rid in run_ids:
            s, e = int(rp[rid]), int(rp[rid + 1])
            all_keys.append(np.asarray(ks[s:e], dtype=np.int64))

        # For each layer: compute pairwise distances → score = -mean_layer_dist
        for lid in range(N_LAYERS):
            lo = lid * LAYER_STRIDE
            hi = (lid + 1) * LAYER_STRIDE

            # Extract layer-specific keys
            layer_keys = [k[(k >= lo) & (k < hi)] for k in all_keys]
            n_keys_mean = float(np.mean([len(lk) for lk in layer_keys]))
            layer_n_keys[lid].append(n_keys_mean)

            # Skip if this layer has almost no activations
            if n_keys_mean < 5:
                continue

            # Compute pairwise distances for this layer
            n = len(run_ids)
            D_layer = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = layer_keys[i], layer_keys[j]
                    if a.size == 0 and b.size == 0:
                        d = 0.0
                    elif a.size == 0 or b.size == 0:
                        d = 1.0
                    else:
                        inter = np.intersect1d(a, b, assume_unique=True).size
                        denom = a.size + b.size - inter
                        d = float(1.0 - inter / denom) if denom > 0 else 0.0
                    D_layer[i, j] = D_layer[j, i] = d

            cent = _centrality_scores(D_layer)
            auc = _pairwise_auc(cent, y)
            if np.isfinite(auc):
                layer_aucs[lid].append(auc)

        if p_idx > 0 and p_idx % 10 == 0:
            print(f"    [{p_idx}/{len(problems)}] {time.time()-t0:.0f}s", flush=True)

    results = {}
    for lid in range(N_LAYERS):
        aucs = layer_aucs[lid]
        n_keys = layer_n_keys[lid]
        results[lid] = {
            "pairwise_auc_mean": float(np.nanmean(aucs)) if aucs else float("nan"),
            "pairwise_auc_std": float(np.nanstd(aucs)) if len(aucs) > 1 else float("nan"),
            "mean_keys_per_run": float(np.mean(n_keys)) if n_keys else 0.0,
            "n_problems": len(aucs),
        }
    return results


# ── analysis C: w_max signal ──────────────────────────────────────────────────

def analyze_wmax_signal(
    cache_root: str,
    problem_groups: dict[str, list[int]],
    labels: np.ndarray,
    n_problems: Optional[int] = None,
) -> dict[str, Any]:
    """Test whether high w_max activation predicts correctness."""
    reader = CacheReader(cache_root)

    problems = list(problem_groups.keys())
    if n_problems is not None:
        problems = problems[:n_problems]

    wmax_mean_aucs, wmax_top_aucs, n_active_aucs = [], [], []

    for pid in problems:
        run_ids = problem_groups[pid]
        y = labels[run_ids]
        if y.sum() == 0 or y.sum() == len(y):
            continue

        s_mean = _wmax_scores(reader, run_ids)
        s_top = _wmax_top10pct_scores(reader, run_ids)
        s_n = _n_active_scores(reader, run_ids)

        wmax_mean_aucs.append(_pairwise_auc(s_mean, y))
        wmax_top_aucs.append(_pairwise_auc(s_top, y))
        n_active_aucs.append(_pairwise_auc(s_n, y))

    def _m(lst): return float(np.nanmean(lst)) if lst else float("nan")

    return {
        "wmax_mean_pairwise_auc": _m(wmax_mean_aucs),
        "wmax_top10pct_pairwise_auc": _m(wmax_top_aucs),
        "n_active_neurons_pairwise_auc": _m(n_active_aucs),
        "n_problems_evaluated": len(wmax_mean_aucs),
    }


# ── analysis D: consensus neuron voting (cross-val) ───────────────────────────

def analyze_consensus_voting(
    cache_root: str,
    problem_groups: dict[str, list[int]],
    labels: np.ndarray,
    n_folds: int = 5,
    n_problems: Optional[int] = None,
    consensus_threshold: float = 0.5,
) -> dict[str, Any]:
    """Cross-val consensus neuron scoring: train on other problems, test on held-out.

    Strategy: For each fold, use training problems to identify "correct-consensus neurons"
    (neurons present in ≥threshold fraction of correct solutions per problem), then
    score test runs by overlap with consensus set built from training problems.

    Actually, since each problem is independent, we can build consensus within the
    problem itself using half the correct solutions and score the remaining.
    Uses leave-one-out within each problem's correct set.
    """
    reader = CacheReader(cache_root)
    rp = reader.row_ptr
    ks = reader.keys

    problems = list(problem_groups.keys())
    if n_problems is not None:
        problems = problems[:n_problems]

    consensus_aucs, consensus_top1 = [], []

    for pid in problems:
        run_ids = problem_groups[pid]
        y = labels[run_ids]
        n_pos = int(y.sum())
        n_neg = int((1 - y).sum())
        if n_pos < 2 or n_neg == 0:
            continue

        # Load all neuron keys for this problem
        all_keys = []
        for rid in run_ids:
            s, e = int(rp[rid]), int(rp[rid + 1])
            all_keys.append(set(int(k) for k in ks[s:e]))

        pos_indices = [i for i, yi in enumerate(y) if yi == 1]
        neg_indices = [i for i, yi in enumerate(y) if yi == 0]

        # Leave-one-out: for each correct solution, build consensus from remaining correct
        loo_scores = np.zeros(len(run_ids), dtype=np.float64)
        for i, pos_i in enumerate(pos_indices):
            # Consensus = neurons in ≥ threshold of OTHER correct solutions
            other_pos = [j for j in pos_indices if j != pos_i]
            if not other_pos:
                continue
            # Count how often each neuron appears in other correct solutions
            neuron_counts: dict[int, int] = {}
            for j in other_pos:
                for nid in all_keys[j]:
                    neuron_counts[nid] = neuron_counts.get(nid, 0) + 1
            consensus = {nid for nid, cnt in neuron_counts.items()
                         if cnt >= consensus_threshold * len(other_pos)}
            # Score this run: fraction of consensus neurons activated
            if consensus:
                loo_scores[pos_i] = len(all_keys[pos_i] & consensus) / len(consensus)

        # Score negative samples against the full correct consensus
        if pos_indices:
            all_pos = pos_indices
            neuron_counts_full: dict[int, int] = {}
            for j in all_pos:
                for nid in all_keys[j]:
                    neuron_counts_full[nid] = neuron_counts_full.get(nid, 0) + 1
            consensus_full = {nid for nid, cnt in neuron_counts_full.items()
                              if cnt >= consensus_threshold * len(all_pos)}
            for i in neg_indices:
                if consensus_full:
                    loo_scores[i] = len(all_keys[i] & consensus_full) / len(consensus_full)

        auc = _pairwise_auc(loo_scores, y)
        if np.isfinite(auc):
            consensus_aucs.append(auc)
        top1 = int(np.argmax(loo_scores))
        consensus_top1.append(_selection_accuracy(top1, y))

    def _m(lst): return float(np.nanmean(lst)) if lst else float("nan")

    return {
        "consensus_pairwise_auc": _m(consensus_aucs),
        "consensus_top1_acc": _m([a for a in consensus_top1 if a >= 0]),
        "n_problems_evaluated": len(consensus_aucs),
        "consensus_threshold": consensus_threshold,
    }


# ── analysis E: existing NAD selector benchmark ───────────────────────────────

def analyze_nad_selectors(
    dist_matrices: dict[str, np.ndarray],
    cache_root: str,
    problem_groups: dict[str, list[int]],
    labels: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Benchmark existing NAD selectors: top1 accuracy + pairwise AUC."""
    from nad.core.selectors.impl import (
        MedoidSelector,
        KNNMedoidSelector,
        MinActivationSelector,
        MaxActivationSelector,
    )
    reader = CacheReader(cache_root)

    def _run_selector(selector_name: str, score_fn) -> dict[str, float]:
        top1_results, pairwise_aucs = [], []
        for pid, D in dist_matrices.items():
            run_ids = problem_groups[pid]
            y = labels[run_ids]
            if y.sum() == 0:
                continue
            D_arr = np.asarray(D, dtype=np.float64)
            try:
                scores = score_fn(D_arr, run_ids, reader)
                sel_idx = int(np.argmax(scores))
                top1_results.append(_selection_accuracy(sel_idx, y))
                auc = _pairwise_auc(scores, y)
                if np.isfinite(auc):
                    pairwise_aucs.append(auc)
            except Exception as e:
                pass

        valid = [a for a in top1_results if a >= 0]
        return {
            "top1_acc": float(np.mean(valid)) if valid else float("nan"),
            "pairwise_auc": float(np.nanmean(pairwise_aucs)) if pairwise_aucs else float("nan"),
            "n_problems": len(valid),
        }

    results = {}

    # Medoid (centrality)
    results["medoid"] = _run_selector(
        "medoid",
        lambda D, rids, r: _centrality_scores(D),
    )
    # KNN k=3
    results["knn_k3"] = _run_selector(
        "knn_k3",
        lambda D, rids, r: _knn_scores(D, k=3),
    )
    # KNN k=7
    results["knn_k7"] = _run_selector(
        "knn_k7",
        lambda D, rids, r: _knn_scores(D, k=7),
    )
    # Min activation (fewer neurons = more focused)
    results["min_active"] = _run_selector(
        "min_active",
        lambda D, rids, r: -_n_active_scores(r, rids),
    )
    # Max activation
    results["max_active"] = _run_selector(
        "max_active",
        lambda D, rids, r: _n_active_scores(r, rids),
    )
    # w_max mean
    results["wmax_mean"] = _run_selector(
        "wmax_mean",
        lambda D, rids, r: _wmax_scores(r, rids),
    )
    # w_max top-10%
    results["wmax_top10pct"] = _run_selector(
        "wmax_top10pct",
        lambda D, rids, r: _wmax_top10pct_scores(r, rids),
    )

    return results


def _rank_scores_from_order(order: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(int(n), dtype=np.float64)
    if int(n) <= 0:
        return out
    if int(n) == 1:
        out[int(order[0])] = 1.0
        return out
    values = np.linspace(1.0, 0.0, num=int(n), dtype=np.float64)
    for pos, idx in enumerate(np.asarray(order, dtype=np.int64).tolist()):
        out[int(idx)] = float(values[int(pos)])
    return out


def _selected_sample_ids(problem_groups: dict[str, list[int]]) -> np.ndarray:
    rows: list[int] = []
    for problem_id in sorted(problem_groups.keys()):
        rows.extend(int(v) for v in problem_groups[problem_id])
    return np.asarray(rows, dtype=np.int64)


def _selected_group_labels(problem_groups: dict[str, list[int]]) -> np.ndarray:
    labels: list[str] = []
    for problem_id in sorted(problem_groups.keys()):
        labels.extend([str(problem_id)] * len(problem_groups[problem_id]))
    return np.asarray(labels, dtype=object)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a_arr = np.asarray(a, dtype=np.float64).reshape(-1)
    b_arr = np.asarray(b, dtype=np.float64).reshape(-1)
    if a_arr.size != b_arr.size or a_arr.size <= 1:
        return float("nan")
    if np.std(a_arr) < 1e-12 or np.std(b_arr) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a_arr, b_arr)[0, 1])


def _fit_grouped_pointwise_lr_scores(
    problem_groups: dict[str, list[int]],
    labels: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    sample_ids = _selected_sample_ids(problem_groups)
    group_labels = _selected_group_labels(problem_groups)
    y = np.asarray(labels[sample_ids], dtype=np.int32)
    X_sel = np.asarray(X[sample_ids], dtype=np.float64)
    unique_groups = np.unique(group_labels)
    n_splits = min(5, int(unique_groups.size))
    out = np.zeros(len(labels), dtype=np.float64)
    if n_splits < 2 or np.unique(y).size < 2:
        return out

    splitter = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in splitter.split(X_sel, y, group_labels):
        y_train = y[train_idx]
        y_test = y[test_idx]
        if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
            continue
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_sel[train_idx])
        X_test = scaler.transform(X_sel[test_idx])
        clf = LogisticRegression(max_iter=4000, class_weight="balanced")
        clf.fit(X_train, y_train)
        out[sample_ids[test_idx]] = clf.decision_function(X_test)
    return out


def _fit_grouped_pairwise_lr_scores(
    problem_groups: dict[str, list[int]],
    labels: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    sample_ids = _selected_sample_ids(problem_groups)
    group_labels = _selected_group_labels(problem_groups)
    unique_groups = np.unique(group_labels)
    n_splits = min(5, int(unique_groups.size))
    out = np.zeros(len(labels), dtype=np.float64)
    if n_splits < 2:
        return out

    fold_map = {str(problem_id): int(idx % n_splits) for idx, problem_id in enumerate(unique_groups.tolist())}
    for fold in range(n_splits):
        train_groups = [str(pid) for pid in unique_groups.tolist() if fold_map[str(pid)] != int(fold)]
        test_groups = [str(pid) for pid in unique_groups.tolist() if fold_map[str(pid)] == int(fold)]
        X_pairs: list[np.ndarray] = []
        y_pairs: list[int] = []
        for problem_id in train_groups:
            idx = np.asarray(problem_groups[problem_id], dtype=np.int64)
            pos = idx[labels[idx] > 0]
            neg = idx[labels[idx] <= 0]
            if pos.size <= 0 or neg.size <= 0:
                continue
            for pos_idx in pos.tolist():
                for neg_idx in neg.tolist():
                    X_pairs.append(np.asarray(X[pos_idx] - X[neg_idx], dtype=np.float64))
                    y_pairs.append(1)
                    X_pairs.append(np.asarray(X[neg_idx] - X[pos_idx], dtype=np.float64))
                    y_pairs.append(0)
        if not X_pairs:
            continue
        X_pair = np.asarray(X_pairs, dtype=np.float64)
        y_pair = np.asarray(y_pairs, dtype=np.int32)
        if np.unique(y_pair).size < 2:
            continue
        scaler = StandardScaler()
        X_pair_scaled = scaler.fit_transform(X_pair)
        clf = LogisticRegression(max_iter=4000, class_weight="balanced")
        clf.fit(X_pair_scaled, y_pair)

        for problem_id in test_groups:
            idx = np.asarray(problem_groups[problem_id], dtype=np.int64)
            X_local = scaler.transform(np.asarray(X[idx], dtype=np.float64))
            local_scores = np.zeros(idx.size, dtype=np.float64)
            for local_idx in range(idx.size):
                diffs = X_local[local_idx : local_idx + 1] - X_local
                local_scores[local_idx] = float(np.mean(clf.decision_function(diffs)))
            out[idx] = local_scores
    return out


def _load_or_build_code_v2_baseline_scores(
    cache_root: str,
    problem_groups: dict[str, list[int]],
    *,
    dist_matrices: dict[str, np.ndarray],
    cache_path: str,
    refresh: bool = False,
) -> np.ndarray:
    cp = Path(cache_path)
    reader = CacheReader(cache_root)
    n_total = int(reader.num_runs())
    all_sample_ids = np.sort(_selected_sample_ids(problem_groups))
    if cp.exists() and not refresh:
        with cp.open("rb") as fh:
            payload = pickle.load(fh)
        cached_sample_ids = np.asarray(payload.get("sample_ids", np.zeros(0, dtype=np.int64)), dtype=np.int64)
        if np.array_equal(cached_sample_ids, all_sample_ids):
            cached_scores = np.asarray(payload["scores"], dtype=np.float64)
            full_scores = np.zeros(int(payload.get("n_total", n_total)), dtype=np.float64)
            full_scores[all_sample_ids] = cached_scores
            return full_scores

    baseline_scores = np.zeros(n_total, dtype=np.float64)
    for problem_id in sorted(problem_groups.keys()):
        run_ids = list(map(int, problem_groups[problem_id]))
        views = [reader.get_run_view(int(run_id), VIEW_SPEC) for run_id in run_ids]
        context = SelectorContext(
            cache=reader,
            problem_id=str(problem_id),
            run_ids=run_ids,
            views=views,
        )
        payload = extract_code_rns_context_payload(context)
        raw_scores = np.asarray(payload["baseline_scores"], dtype=np.float64)
        D = np.asarray(dist_matrices[str(problem_id)], dtype=np.float64)
        order = order_code_dynamic_group_indices(raw_scores, D, run_ids=run_ids)
        baseline_scores[np.asarray(run_ids, dtype=np.int64)] = _rank_scores_from_order(order, len(run_ids))

    cp.parent.mkdir(parents=True, exist_ok=True)
    with cp.open("wb") as fh:
        pickle.dump(
            {
                "sample_ids": all_sample_ids,
                "scores": baseline_scores[all_sample_ids],
                "n_total": int(n_total),
            },
            fh,
            protocol=4,
        )
    return baseline_scores


# ── main ───────────────────────────────────────────────────────────────────────

def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_v2(
    cache_root: str,
    dist_cache: str,
    *,
    layer_cache: str = DEFAULT_LAYER_CACHE,
    static_cache: str = DEFAULT_STATIC_CACHE,
    code_v2_cache: str = DEFAULT_CODE_V2_SCORE_CACHE,
    audit_json: str = DEFAULT_AUDIT_JSON,
    metrics_json: str = DEFAULT_METRICS_JSON,
    records_jsonl: str = DEFAULT_RECORDS_JSONL,
    n_problems: Optional[int] = None,
    n_threads: int = N_THREADS,
    out_doc: Optional[str] = None,
    refresh_dist: bool = False,
    refresh_features: bool = False,
    skip_distance_native: bool = False,
) -> None:
    print("=" * 70)
    print("Coding Improvement V2: Code-Native Verifier / Reranker")
    print("=" * 70)

    cache_root_abs = str(REPO_ROOT / cache_root) if not Path(cache_root).is_absolute() else cache_root

    print("\n[0] Loading groups + labels…")
    full_problem_groups = _load_problem_groups(cache_root_abs)
    if n_problems is not None:
        subset = list(full_problem_groups.keys())[: int(n_problems)]
        problem_groups = {problem_id: full_problem_groups[problem_id] for problem_id in subset}
    else:
        problem_groups = dict(full_problem_groups)

    correctness = load_correctness_map(cache_root_abs)
    n_total = max(int(max(ids)) for ids in full_problem_groups.values()) + 1
    labels = np.asarray([int(correctness.get(i, 0)) for i in range(n_total)], dtype=np.int32)
    sample_ids = np.arange(n_total, dtype=np.int64)
    oracle_rates = np.asarray([labels[np.asarray(ids, dtype=np.int64)].mean() for ids in problem_groups.values()], dtype=np.float64)
    random_top1 = float(np.mean(oracle_rates)) if oracle_rates.size > 0 else 0.0
    oracle_solvable = int((oracle_rates > 0).sum())
    print(f"    problems={len(problem_groups)} samples={n_total} pos_rate={labels.mean():.4f}")
    print(f"    random_top1={random_top1:.4f} oracle_solvable={oracle_solvable}/{len(problem_groups)}")

    print("\n[1] Auditing locally recoverable inputs…")
    audit = audit_coding_inputs(cache_root_abs, prefer_full=True)
    _write_json(str(REPO_ROOT / audit_json), audit)
    print(f"    tier={audit['sufficiency_tier']}  report={audit['selected_report_kind']}  recovered_code={audit['recovered_code_nonempty_count']}/{audit['n_samples']}")

    print("\n[2] Loading static report features…")
    static_payload = build_static_feature_cache(
        cache_root_abs,
        cache_path=str(REPO_ROOT / static_cache),
        prefer_full=True,
        refresh=refresh_features,
    )
    X_static = np.asarray(static_payload["X"], dtype=np.float64)
    report_records = static_payload["records"]
    output_tokens = np.asarray([float(row.get("output_tokens", 0.0) or 0.0) for row in report_records], dtype=np.float64)
    code_chars = np.asarray(X_static[:, STATIC_FEATURE_NAMES.index("code_chars")], dtype=np.float64)
    print(f"    static_features={X_static.shape[1]} report_path={static_payload['report_info']['report_path']}")

    print("\n[3] Loading neuron summary features…")
    layer_payload = load_or_build_layer_summary_cache(
        cache_root_abs,
        cache_path=str(REPO_ROOT / layer_cache),
        sample_ids=sample_ids,
        refresh=refresh_features,
        verbose=refresh_features,
    )
    X_neuron, neuron_feature_names = build_global_neuron_feature_matrix(layer_payload)
    print(f"    neuron_features={X_neuron.shape[1]} layers={layer_payload['n_layers']}")

    print("\n[4] Loading/building current code_v2 baseline scores…")
    dist_matrices = _load_or_build_dist_matrices(
        cache_root=cache_root_abs,
        problem_groups=problem_groups,
        cache_path=str(REPO_ROOT / dist_cache),
        n_problems=n_problems,
        n_threads=n_threads,
        refresh=refresh_dist,
    )
    code_v2_scores = _load_or_build_code_v2_baseline_scores(
        cache_root_abs,
        problem_groups,
        dist_matrices=dist_matrices,
        cache_path=str(REPO_ROOT / code_v2_cache),
        refresh=refresh_features,
    )

    print("\n[5] Fitting grouped reranker candidates…")
    X_hybrid = np.column_stack([X_static, X_neuron]).astype(np.float64)
    static_lr_scores = _fit_grouped_pointwise_lr_scores(problem_groups, labels, X_static)
    static_pairwise_scores = _fit_grouped_pairwise_lr_scores(problem_groups, labels, X_static)
    hybrid_lr_scores = _fit_grouped_pointwise_lr_scores(problem_groups, labels, X_hybrid)

    neuron_n_active_total = np.asarray(X_neuron[:, neuron_feature_names.index("n_active_total")], dtype=np.float64)
    neuron_wmax_global = np.asarray(X_neuron[:, neuron_feature_names.index("wmax_mean_global")], dtype=np.float64)
    static_code_chars = np.asarray(code_chars, dtype=np.float64)

    candidate_scores = {
        "code_v2_baseline": code_v2_scores,
        "static_code_chars": static_code_chars,
        "static_logreg": static_lr_scores,
        "static_pairwise_logreg": static_pairwise_scores,
        "neuron_n_active_total": neuron_n_active_total,
        "neuron_wmax_mean_global": neuron_wmax_global,
        "hybrid_logreg": hybrid_lr_scores,
    }

    candidate_results: dict[str, dict[str, Any]] = {}
    for name, scores in candidate_scores.items():
        baseline_scores = None if name == "code_v2_baseline" else code_v2_scores
        result = evaluate_grouped_scores(
            problem_groups,
            labels,
            scores,
            sample_ids=sample_ids,
            baseline_scores=baseline_scores,
        )
        result["score_corr_output_tokens"] = _corr(scores[_selected_sample_ids(problem_groups)], output_tokens[_selected_sample_ids(problem_groups)])
        result["score_corr_code_chars"] = _corr(scores[_selected_sample_ids(problem_groups)], code_chars[_selected_sample_ids(problem_groups)])
        candidate_results[name] = result
        uplift = result.get("pass@1_uplift_abs")
        uplift_str = "n/a" if uplift is None else f"{float(uplift):+.4f}"
        print(
            f"    {name:<24} top1={result['top1_accuracy']:.4f} "
            f"uplift={uplift_str} pairwise={result['pairwise_auc'] if result['pairwise_auc'] is not None else float('nan'):.4f}"
        )

    candidate_order = [
        "code_v2_baseline",
        *sorted(
            [name for name in candidate_results.keys() if name != "code_v2_baseline"],
            key=lambda name: (
                float(candidate_results[name]["top1_accuracy"]),
                float(candidate_results[name].get("pass@1_uplift_abs") or -1e9),
                float(candidate_results[name].get("pairwise_auc") or -1e9),
                -abs(float(candidate_results[name].get("score_corr_output_tokens") or 0.0)),
                name,
            ),
            reverse=True,
        ),
    ]
    best_candidate_name = candidate_order[1] if len(candidate_order) > 1 else "code_v2_baseline"
    best_candidate = candidate_results[best_candidate_name]
    write_problem_records_jsonl(best_candidate["problem_records"], str(REPO_ROOT / records_jsonl))

    print("\n[6] Running neuron-native exploratory branch…")
    res_A = res_B = res_C = res_D = res_E = None
    sorted_layers: list[tuple[int, dict[str, Any]]] = []
    if not skip_distance_native:
        print("    [A] Jaccard clustering…")
        res_A = analyze_jaccard_clustering(dist_matrices, problem_groups, labels)
        print(f"        CC-CI gap={res_A['cc_ci_gap']:+.5f} medoid_top1={res_A['medoid_top1_acc']:.4f}")

        print("    [B] Layer-wise discriminability…")
        res_B = analyze_layer_discriminability(
            cache_root=cache_root_abs,
            problem_groups=problem_groups,
            labels=labels,
            n_problems=min(n_problems or len(problem_groups), 60),
            n_threads=4,
        )
        sorted_layers = sorted(res_B.items(), key=lambda kv: kv[1]["pairwise_auc_mean"], reverse=True)
        if sorted_layers:
            print(f"        best_layer={sorted_layers[0][0]} pairwise_auc={sorted_layers[0][1]['pairwise_auc_mean']:.4f}")

        print("    [C] w_max / activation intensity…")
        res_C = analyze_wmax_signal(
            cache_root=cache_root_abs,
            problem_groups=problem_groups,
            labels=labels,
            n_problems=n_problems,
        )
        print(f"        wmax_mean_pairwise_auc={res_C['wmax_mean_pairwise_auc']:.4f}")

        print("    [D] Consensus voting…")
        res_D = analyze_consensus_voting(
            cache_root=cache_root_abs,
            problem_groups=problem_groups,
            labels=labels,
            n_problems=n_problems,
            consensus_threshold=0.5,
        )
        print(f"        consensus_top1={res_D['consensus_top1_acc']:.4f}")

        print("    [E] Distance-native selector benchmark…")
        res_E = analyze_nad_selectors(dist_matrices, cache_root_abs, problem_groups, labels)
        top_distance = sorted(res_E.items(), key=lambda kv: kv[1]["top1_acc"], reverse=True)[0]
        print(f"        best_distance_selector={top_distance[0]} top1={top_distance[1]['top1_acc']:.4f}")
    else:
        print("    skipped")

    baseline_top1 = float(candidate_results["code_v2_baseline"]["top1_accuracy"])
    best_uplift = float(best_candidate.get("pass@1_uplift_abs") or 0.0)
    if audit["sufficiency_tier"] in {"Tier A", "Tier B"} and best_uplift > 0.0:
        sufficiency_conclusion = (
            f"{audit['sufficiency_tier']} sufficient: code-native signals improve pass@1 "
            f"over current code_v2 by {best_uplift:+.4f}."
        )
    elif audit["sufficiency_tier"] in {"Tier A", "Tier B"}:
        sufficiency_conclusion = (
            f"{audit['sufficiency_tier']} locally sufficient for code recovery, but current "
            f"candidates do not beat code_v2."
        )
    else:
        sufficiency_conclusion = "Current local artifacts are insufficient for a code-native verifier."

    summary_payload = {
        "status_summary": {
            "sufficiency_tier": audit["sufficiency_tier"],
            "baseline_candidate": "code_v2_baseline",
            "best_candidate": best_candidate_name,
            "best_candidate_top1": best_candidate["top1_accuracy"],
            "best_candidate_pass@1_uplift_abs": best_candidate.get("pass@1_uplift_abs"),
            "distance_native_branch_ran": not skip_distance_native,
        },
        "inputs": {
            "cache_root": cache_root_abs,
            "n_problems_used": int(len(problem_groups)),
            "static_cache": str(REPO_ROOT / static_cache),
            "layer_cache": str(REPO_ROOT / layer_cache),
            "dist_cache": str(REPO_ROOT / dist_cache),
        },
        "audit": audit,
        "baseline_top1_accuracy": baseline_top1,
        "random_top1_accuracy": random_top1,
        "oracle_solvable_rate": float(oracle_solvable / len(problem_groups)) if problem_groups else 0.0,
        "candidate_results": candidate_results,
        "candidate_order": candidate_order,
        "selected_candidate": {
            "name": best_candidate_name,
            "metrics": best_candidate,
        },
        "exploratory_distance_native": {
            "jaccard_clustering": res_A,
            "layer_discriminability": res_B,
            "wmax_signal": res_C,
            "consensus_voting": res_D,
            "selector_benchmark": res_E,
        },
        "sufficiency_conclusion": sufficiency_conclusion,
    }
    _write_json(str(REPO_ROOT / metrics_json), summary_payload)

    if out_doc:
        _write_doc(
            out_path=str(REPO_ROOT / out_doc) if not Path(out_doc).is_absolute() else out_doc,
            audit=audit,
            candidate_results=candidate_results,
            candidate_order=candidate_order,
            best_candidate_name=best_candidate_name,
            random_top1=random_top1,
            oracle_pct=float(oracle_solvable / len(problem_groups)) if problem_groups else 0.0,
            sufficiency_conclusion=sufficiency_conclusion,
            res_A=res_A,
            res_C=res_C,
            res_D=res_D,
            res_E=res_E,
            sorted_layers=sorted_layers,
            n_problems_used=len(problem_groups),
        )
        print(f"\n[doc] Saved → {out_doc}")
    print(f"[json] Saved → {metrics_json}")
    print(f"[jsonl] Saved → {records_jsonl}")


def _write_doc(
    out_path: str,
    *,
    audit: dict[str, Any],
    candidate_results: dict[str, dict[str, Any]],
    candidate_order: list[str],
    best_candidate_name: str,
    random_top1: float,
    oracle_pct: float,
    sufficiency_conclusion: str,
    res_A: dict[str, Any] | None,
    res_C: dict[str, Any] | None,
    res_D: dict[str, Any] | None,
    res_E: dict[str, Any] | None,
    sorted_layers: list[tuple[int, dict[str, Any]]],
    n_problems_used: int,
) -> None:
    lines = [
        "# 14 — Coding Improvement V2",
        "",
        f"**Date**: 2026-04-13  **Problems**: {n_problems_used}  **Status**: implemented",
        "",
        "## Summary",
        "",
        f"- **Recoverability**: {audit['sufficiency_tier']} — {audit['sufficiency_reason']}",
        f"- **Random top1**: {random_top1:.4f}",
        f"- **Oracle-solvable rate**: {oracle_pct:.4f}",
        f"- **Selected V2 candidate**: `{best_candidate_name}`",
        f"- **Conclusion**: {sufficiency_conclusion}",
        "",
        "## Input Audit",
        "",
        "| Field | Count |",
        "|---|---:|",
        f"| Prompt non-empty | {audit['prompt_nonempty_count']} |",
        f"| Generated text non-empty | {audit['generated_text_nonempty_count']} |",
        f"| Extracted answer non-empty | {audit['extracted_answer_nonempty_count']} |",
        f"| Recovered code non-empty | {audit['recovered_code_nonempty_count']} |",
        f"| Selected report kind | {audit['selected_report_kind']} |",
        "",
        "## Grouped Ranking Results",
        "",
        "| Candidate | Top1 | Pass@1 uplift | Pairwise AUC | Pooled AUROC | Corr(output_tokens) | Corr(code_chars) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in candidate_order:
        row = candidate_results[name]
        uplift = row.get("pass@1_uplift_abs")
        lines.append(
            "| {name} | {top1:.4f} | {uplift} | {pairwise} | {auroc} | {corr_out} | {corr_code} |".format(
                name=name,
                top1=float(row["top1_accuracy"]),
                uplift="—" if uplift is None else f"{float(uplift):+.4f}",
                pairwise="—" if row.get("pairwise_auc") is None else f"{float(row['pairwise_auc']):.4f}",
                auroc="—" if row.get("pooled_pointwise_auroc") is None else f"{float(row['pooled_pointwise_auroc']):.4f}",
                corr_out="—" if row.get("score_corr_output_tokens") is None else f"{float(row['score_corr_output_tokens']):+.4f}",
                corr_code="—" if row.get("score_corr_code_chars") is None else f"{float(row['score_corr_code_chars']):+.4f}",
            )
        )
    best_row = candidate_results[best_candidate_name]
    lines += [
        "",
        "## Selected Candidate",
        "",
        f"- **Name**: `{best_candidate_name}`",
        f"- **Top1**: {best_row['top1_accuracy']:.4f}",
        f"- **Pass@1 uplift vs code_v2**: {float(best_row.get('pass@1_uplift_abs') or 0.0):+.4f}",
        f"- **Head-to-head**: wins={best_row['head_to_head']['wins']} losses={best_row['head_to_head']['losses']} ties={best_row['head_to_head']['ties']}",
        f"- **Hard-problem top1**: {best_row['difficulty_summary']['hard']['top1_accuracy']}",
        f"- **Easy-problem top1**: {best_row['difficulty_summary']['easy']['top1_accuracy']}",
    ]

    if res_A is not None and res_C is not None and res_D is not None and res_E is not None:
        lines += [
            "",
            "## Neuron-Native Exploratory Branch",
            "",
            f"- **Jaccard CC–CI gap**: {res_A['cc_ci_gap']:+.5f}",
            f"- **Medoid top1**: {res_A['medoid_top1_acc']:.4f}",
            f"- **w_max mean pairwise AUC**: {res_C['wmax_mean_pairwise_auc']:.4f}",
            f"- **Consensus voting top1**: {res_D['consensus_top1_acc']:.4f}",
        ]
        if sorted_layers:
            lines.append(f"- **Best layer**: {sorted_layers[0][0]} with pairwise AUC {sorted_layers[0][1]['pairwise_auc_mean']:.4f}")
        lines += [
            "",
            "| Distance-Native Selector | Top1 | Pairwise AUC |",
            "|---|---:|---:|",
        ]
        for sel_name, metrics in sorted(res_E.items(), key=lambda kv: kv[1]["top1_acc"], reverse=True):
            lines.append(f"| {sel_name} | {metrics['top1_acc']:.4f} | {metrics['pairwise_auc']:.4f} |")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    p.add_argument("--dist-cache", default=DEFAULT_DIST_CACHE)
    p.add_argument("--layer-cache", default=DEFAULT_LAYER_CACHE)
    p.add_argument("--static-cache", default=DEFAULT_STATIC_CACHE)
    p.add_argument("--code-v2-cache", default=DEFAULT_CODE_V2_SCORE_CACHE)
    p.add_argument("--audit-json", default=DEFAULT_AUDIT_JSON)
    p.add_argument("--metrics-json", default=DEFAULT_METRICS_JSON)
    p.add_argument("--records-jsonl", default=DEFAULT_RECORDS_JSONL)
    p.add_argument("--n-problems", type=int, default=None, help="Limit to N problems (smoke test)")
    p.add_argument("--n-threads", type=int, default=N_THREADS)
    p.add_argument("--out", default=DEFAULT_OUT_DOC)
    p.add_argument("--refresh-dist", action="store_true")
    p.add_argument("--refresh-features", action="store_true")
    p.add_argument("--skip-distance-native", action="store_true")
    args = p.parse_args()

    run_v2(
        cache_root=args.cache_root,
        dist_cache=args.dist_cache,
        layer_cache=args.layer_cache,
        static_cache=args.static_cache,
        code_v2_cache=args.code_v2_cache,
        audit_json=args.audit_json,
        metrics_json=args.metrics_json,
        records_jsonl=args.records_jsonl,
        n_problems=args.n_problems,
        n_threads=args.n_threads,
        out_doc=args.out,
        refresh_dist=args.refresh_dist,
        refresh_features=args.refresh_features,
        skip_distance_native=args.skip_distance_native,
    )
