#!/usr/bin/env python3
"""
Scan structural selectors on lcb_v5 (MUI_HUB, has ground truth) to find
the best per-run scoring strategy, then apply to cache_test lcb_v5 caches
and patch the submission JSON.

Selectors evaluated (all distance-based / activation-based, no tok_conf):
  medoid            : score = -mean(D[i, :])
  knn(k)            : score = mean of top-k similarities (1-D[i,:]) excl. self
  min_activation    : score = -activation_length[i]
  max_activation    : score = +activation_length[i]
  graph_degree(eps) : score = adj.sum(axis=1) with adaptive eps
  inverse_current   : score = -(current submission score)  [invert anti-signal]

Run from repo root:
  python scripts/scan_lcb_knn_selectors.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.distance.engine import DistanceEngine, DistanceSpec
from nad.core.views.reader import Agg, CacheReader, CutSpec, CutType, Order, ViewSpec
from nad.ops.accuracy import _load_ground_truth

# ── Paths ──────────────────────────────────────────────────────────────────
LCB_MAIN = REPO_ROOT / "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/livecodebench_v5/cache_neuron_output_1_act_no_rms_20251127_032808"
LCB_TEST_DS = Path("/home/jovyan/public-ro/MUI_HUB/cache_test/DeepSeek-R1-0528-Qwen3-8B/livecodebench_v5/cache_neuron_output_1_act_no_rms_20250913_000912")
LCB_TEST_QW = Path("/home/jovyan/public-ro/MUI_HUB/cache_test/Qwen3-4B-Thinking-2507/livecodebench_v5/cache_neuron_output_1_act_no_rms_20250920_094942")

SUBMISSION_IN  = REPO_ROOT / "submission/BestofN/extreme12/base/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank.json"
SUBMISSION_OUT = REPO_ROOT / "submission/BestofN/extreme12/patches/extreme12_lcb_knn_patched.json"

DIST_THREADS = 8
DEFAULT_VSPEC = ViewSpec(agg=Agg.MAX, cut=CutSpec(CutType.MASS, 1.0), order=Order.BY_KEY)

# ── Helpers ─────────────────────────────────────────────────────────────────

def _problem_sort_key(pid: str):
    try:
        return (0, int(pid.split("-")[-1]))
    except Exception:
        return (1, str(pid))


def load_problem_groups(cache_root: Path) -> dict[str, list[int]]:
    meta = json.loads((cache_root / "meta.json").read_text())
    groups: dict[str, list[int]] = {}
    for sid, sample in enumerate(meta["samples"]):
        pid = str(sample["problem_id"])
        groups.setdefault(pid, []).append(int(sid))
    return groups


def compute_D(reader: CacheReader, run_ids: list[int]) -> np.ndarray:
    views = [reader.get_run_view(rid, DEFAULT_VSPEC) for rid in run_ids]
    engine = DistanceEngine(DistanceSpec("ja", num_threads=DIST_THREADS))
    return engine.dense_matrix(views)


def score_medoid(D: np.ndarray) -> np.ndarray:
    return -D.mean(axis=1)


def score_knn(D: np.ndarray, k: int) -> np.ndarray:
    n = D.shape[0]
    sim = 1.0 - D
    np.fill_diagonal(sim, -np.inf)
    top_k = np.sort(sim, axis=1)[:, -k:]
    return top_k.mean(axis=1)


def score_min_activation(reader: CacheReader, run_ids: list[int]) -> np.ndarray:
    lengths = []
    for rid in run_ids:
        rv = reader.get_run_view(rid, DEFAULT_VSPEC)
        lengths.append(len(rv.keys) if rv.keys is not None else 0)
    return -np.array(lengths, dtype=np.float64)


def score_max_activation(reader: CacheReader, run_ids: list[int]) -> np.ndarray:
    return -score_min_activation(reader, run_ids)


def score_graph_degree(D: np.ndarray, eps_pct: float = 0.30) -> np.ndarray:
    n = D.shape[0]
    triu = D[np.triu_indices(n, k=1)]
    eps = float(np.quantile(triu, eps_pct)) if triu.size else 0.5
    adj = (D <= eps).astype(np.float64)
    np.fill_diagonal(adj, 0.0)
    return adj.sum(axis=1) / max(n - 1, 1)


def rank_normalize_scores(scores_per_problem: dict[str, dict[str, float]], scale: tuple = (1, 100)) -> dict[str, dict[str, float]]:
    """Monotonic per-problem rank normalization to [1, 100], same as export postprocess."""
    lo, hi = scale
    out = {}
    for pid, run_scores in scores_per_problem.items():
        sids = list(run_scores.keys())
        vals = np.array([run_scores[s] for s in sids], dtype=np.float64)
        n = len(vals)
        ranks = np.argsort(np.argsort(vals)) + 1  # 1..n
        normed = lo + (ranks - 1) * (hi - lo) / max(n - 1, 1)
        out[pid] = {s: float(v) for s, v in zip(sids, normed)}
    return out


# ── Evaluation (uses ground truth) ─────────────────────────────────────────

def evaluate_scores(
    scores_per_problem: dict[str, dict[str, float]],
    correctness: dict[int, bool],
) -> dict[str, float]:
    hit1_total = pairwise_num = pairwise_den = selacc_num = selacc_den = 0
    n_problems = len(scores_per_problem)

    # Collect all (score, is_correct) for selacc@10%
    all_entries: list[tuple[float, bool]] = []

    for pid, run_scores in sorted(scores_per_problem.items(), key=lambda kv: _problem_sort_key(kv[0])):
        sids = sorted(run_scores.keys(), key=lambda x: int(x))
        vals = np.array([run_scores[s] for s in sids])
        labels = np.array([int(bool(correctness.get(int(s), False))) for s in sids])

        # Hit@1
        best_idx = int(np.argmax(vals))
        hit1_total += int(labels[best_idx])

        # Pairwise
        correct_vals = vals[labels == 1]
        wrong_vals   = vals[labels == 0]
        if correct_vals.size > 0 and wrong_vals.size > 0:
            wins = (correct_vals[:, None] > wrong_vals[None, :]).sum()
            total_pairs = correct_vals.size * wrong_vals.size
            pairwise_num += int(wins)
            pairwise_den += int(total_pairs)

        for s, v in zip(sids, vals):
            is_corr = bool(correctness.get(int(s), False))
            all_entries.append((float(v), is_corr))

    # SelAcc@10%
    all_entries.sort(key=lambda x: -x[0])
    top_n = max(1, len(all_entries) // 10)
    top_entries = all_entries[:top_n]
    selacc = sum(1 for _, c in top_entries if c) / max(len(top_entries), 1)

    return {
        "hit@1": hit1_total / max(n_problems, 1),
        "pairwise": pairwise_num / max(pairwise_den, 1),
        "selacc@10%": selacc,
        "n_problems": n_problems,
    }


# ── Main scan ──────────────────────────────────────────────────────────────

def scan_cache(cache_root: Path, correctness: dict[int, bool] | None, label: str) -> dict[str, dict[str, dict[str, float]]]:
    """
    Compute per-run score dicts for all selector candidates.
    Returns: {selector_name: {problem_id: {sample_id_str: score}}}
    """
    print(f"\n[{label}] Loading cache: {cache_root.name}")
    reader = CacheReader(str(cache_root))
    groups = load_problem_groups(cache_root)
    problems = sorted(groups.keys(), key=_problem_sort_key)
    print(f"  {len(problems)} problems × 64 runs")

    candidates = {
        "medoid":         {},
        "knn3":           {},
        "knn5":           {},
        "knn8":           {},
        "knn12":          {},
        "knn16":          {},
        "knn24":          {},
        "knn32":          {},
        "min_activation": {},
        "max_activation": {},
        "graph_deg25":    {},
        "graph_deg30":    {},
        "graph_deg40":    {},
    }

    for prob_idx, pid in enumerate(problems):
        if prob_idx % 20 == 0:
            print(f"  [{label}] problem {prob_idx+1}/{len(problems)} ...", flush=True)
        run_ids = groups[pid]
        sids = [str(r) for r in run_ids]

        D = compute_D(reader, run_ids)

        s_med  = score_medoid(D)
        s_min  = score_min_activation(reader, run_ids)
        s_max  = score_max_activation(reader, run_ids)
        s_gd25 = score_graph_degree(D, 0.25)
        s_gd30 = score_graph_degree(D, 0.30)
        s_gd40 = score_graph_degree(D, 0.40)

        def _make(scores):
            return {s: float(v) for s, v in zip(sids, scores)}

        candidates["medoid"][pid]         = _make(s_med)
        candidates["min_activation"][pid] = _make(s_min)
        candidates["max_activation"][pid] = _make(s_max)
        candidates["graph_deg25"][pid]    = _make(s_gd25)
        candidates["graph_deg30"][pid]    = _make(s_gd30)
        candidates["graph_deg40"][pid]    = _make(s_gd40)

        for k, name in [(3,"knn3"),(5,"knn5"),(8,"knn8"),(12,"knn12"),(16,"knn16"),(24,"knn24"),(32,"knn32")]:
            candidates[name][pid] = _make(score_knn(D, k))

    return candidates


def main():
    # ── Step 1: scan on MUI_HUB lcb_v5 with ground truth ──────────────────
    print("=" * 60)
    print("STEP 1: Scanning selectors on MUI_HUB lcb_v5 (has ground truth)")
    print("=" * 60)

    correctness = _load_ground_truth(LCB_MAIN)
    print(f"Ground truth loaded: {sum(correctness.values())}/{len(correctness)} correct")

    candidates = scan_cache(LCB_MAIN, correctness, "MUI_HUB")

    print("\n=== Evaluation Results ===")
    results = []
    for name, scores in candidates.items():
        m = evaluate_scores(scores, correctness)
        results.append((name, m))
        print(f"  {name:20s}  Hit@1={m['hit@1']:.4f}  Pairwise={m['pairwise']:.4f}  SelAcc@10%={m['selacc@10%']:.4f}")

    # Sort by pairwise (best surrogate for avoiding anti-signal)
    best_name, best_m = max(results, key=lambda x: (x[1]["pairwise"], x[1]["hit@1"]))
    print(f"\n>>> Best (by pairwise): {best_name}  →  Pairwise={best_m['pairwise']:.4f}  SelAcc@10%={best_m['selacc@10%']:.4f}")

    # Also find best by selacc
    best_selacc_name, best_selacc_m = max(results, key=lambda x: (x[1]["selacc@10%"], x[1]["pairwise"]))
    print(f">>> Best (by selacc): {best_selacc_name}  →  SelAcc@10%={best_selacc_m['selacc@10%']:.4f}  Pairwise={best_selacc_m['pairwise']:.4f}")

    # Save full table
    table_path = REPO_ROOT / "results" / "lcb_knn_scan.json"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(json.dumps(
        {name: m for name, m in results}, indent=2
    ), encoding="utf-8")
    print(f"\nFull table saved: {table_path}")

    # Use best by pairwise (avoids anti-signal problem)
    chosen = best_selacc_name
    print(f"\nUsing: {chosen} for submission patch")

    # ── Step 2: apply to cache_test lcb_v5 ────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Scoring cache_test lcb_v5 caches")
    print("=" * 60)

    test_caches = {
        "DS-R1":    LCB_TEST_DS,
        "Qwen3-4B": LCB_TEST_QW,
    }
    new_lcb_scores: dict[str, dict[str, dict[str, float]]] = {}
    for model_tag, cache_path in test_caches.items():
        cands = scan_cache(cache_path, None, model_tag)
        raw = cands[chosen]
        # Rank-normalize to [1,100] to match submission format
        normed = rank_normalize_scores(raw)
        new_lcb_scores[model_tag] = normed
        hit1 = sum(
            1 for pid, rs in normed.items()
            if max(rs, key=lambda k: rs[k]) == max(rs, key=lambda k: rs[k])  # just count
        )
        print(f"  {model_tag}: {len(normed)} problems scored")

    # ── Step 3: patch submission JSON ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Patching submission JSON")
    print("=" * 60)

    sub = json.loads(SUBMISSION_IN.read_text(encoding="utf-8"))

    for model_tag, lcb_scores in new_lcb_scores.items():
        cache_key = f"{model_tag}/lcb_v5"
        old_problems = set(sub["scores"].get(cache_key, {}).keys())
        new_problems = set(lcb_scores.keys())
        print(f"  {cache_key}: replacing {len(old_problems)} → {len(new_problems)} problems")
        sub["scores"][cache_key] = lcb_scores

    sub["method_name"] = sub["method_name"] + f"__lcb_{chosen}"
    sub.setdefault("score_postprocess", {})["lcb_override"] = f"knn_scan_best={chosen}"

    SUBMISSION_OUT.write_text(json.dumps(sub, ensure_ascii=False), encoding="utf-8")
    print(f"\nPatched submission saved: {SUBMISSION_OUT}")
    print(f"Method name: {sub['method_name']}")

    # ── Step 4: sanity check on train cache ───────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Sanity check — evaluate {chosen} on MUI_HUB")
    print("=" * 60)
    m_best = evaluate_scores(candidates[chosen], correctness)
    m_base_raw = candidates.get("knn3", {})
    print(f"  {chosen}: Hit@1={m_best['hit@1']:.4f}  Pairwise={m_best['pairwise']:.4f}  SelAcc@10%={m_best['selacc@10%']:.4f}")
    print()

    # Compare with current model (anti-signal)
    # Load current submission scores for lcb and check
    curr_sub = json.loads(SUBMISSION_IN.read_text(encoding="utf-8"))
    curr_lcb = curr_sub["scores"].get("DS-R1/lcb_v5", {})
    curr_eval_scores: dict[str, dict[str, float]] = {}
    # Note: MUI_HUB and cache_test have different samples; we can only evaluate MUI_HUB
    # The current submission uses cache_test — can't directly compare apples to apples
    print("  Note: current submission uses cache_test (no GT) so direct AUROC comparison")
    print("  uses MUI_HUB training cache. Relative gain from scan is the best we can do.")
    print()
    print("  All scanner results:")
    for name, m in sorted(results, key=lambda x: -x[1]["selacc@10%"]):
        print(f"    {name:20s}  Hit@1={m['hit@1']:.4f}  Pairwise={m['pairwise']:.4f}  SelAcc@10%={m['selacc@10%']:.4f}")


if __name__ == "__main__":
    main()
