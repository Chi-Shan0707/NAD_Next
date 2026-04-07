#!/usr/bin/env python3
"""
Early Stop Signal Scanner — v3 experiment.

Scans token and neuron-structural signals for gpqa and lcb_v5.
Outputs per-signal AUROC @10%/50%/100% and Spearman rho table.

Usage:
    python scripts/scan_earlystop_signals.py --out results/scans/earlystop/earlystop_signal_scan_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.selectors.trajectory_impl import (
    DEFAULT_REFLECTION_THRESHOLD,
    _compute_trajectory_scores,
    _extract_slice_keysets,
)
from nad.core.views.reader import CacheReader
from nad.ops.accuracy import load_correctness_map
from nad.ops.earlystop import (
    EARLY_STOP_POSITIONS,
    build_problem_groups,
    discover_cache_entries,
    _problem_sort_key,
)

TARGET_DATASETS = {"gpqa", "livecodebench_v5", "lcb_v5"}

# ── AUROC ────────────────────────────────────────────────────────────────────

def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Binary AUROC via rank sum."""
    pos = labels == 1
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    # use scipy if available, else manual
    try:
        from scipy.stats import mannwhitneyu
        stat, _ = mannwhitneyu(scores[pos], scores[neg], alternative="greater")
        return float(stat) / (float(pos.sum()) * float(neg.sum()))
    except ImportError:
        pass
    # manual U-statistic
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    rank = np.argsort(np.argsort(scores)) + 1
    u = float(rank[pos].sum()) - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        r, _ = spearmanr(x, y)
        return float(r)
    except ImportError:
        pass
    # manual rank correlation
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d = rx - ry
    n = len(d)
    return float(1 - 6 * np.dot(d, d) / (n * (n * n - 1)))


# ── Per-sample signal extraction ─────────────────────────────────────────────

def extract_signals_for_sample(reader: CacheReader, run_id: int) -> dict[str, list[float]]:
    """Return a dict: signal_name -> list of 10 floats (one per position 10%..100%)."""
    positions = EARLY_STOP_POSITIONS

    tv = reader.get_token_view(run_id)

    # Pre-load token arrays (may be None)
    tok_conf_arr = np.asarray(tv.tok_conf, dtype=np.float64) if (tv and tv.tok_conf is not None) else None
    tok_gini_arr = np.asarray(tv.tok_gini, dtype=np.float64) if (tv and tv.tok_gini is not None) else None

    # Pre-load neuron slices (may be empty)
    slices = _extract_slice_keysets(reader, run_id)
    n_slices = len(slices)

    # Neuron count per slice (from rows/ bank CSR row_ptr differences)
    nc_all: np.ndarray | None = None
    try:
        rows_srp = reader.rows_sample_row_ptr
        rows_rp = reader.rows_row_ptr
        if rows_srp is not None and rows_rp is not None and run_id < len(rows_srp) - 1:
            row_start = int(rows_srp[run_id])
            row_end = int(rows_srp[run_id + 1])
            if row_end > row_start:
                rp_seg = np.asarray(rows_rp[row_start:row_end + 1], dtype=np.int64)
                nc_all = np.diff(rp_seg).astype(np.float64)
    except Exception:
        pass

    # Self-similarity (full): first-half vs second-half neuron union
    self_sim_full = 0.0
    if n_slices > 1:
        half = n_slices // 2
        try:
            first = set()
            for s in slices[:half]:
                first.update(int(k) for k in s)
            second = set()
            for s in slices[half:]:
                second.update(int(k) for k in s)
            inter = len(first & second)
            union = len(first | second)
            self_sim_full = inter / union if union > 0 else 0.0
        except Exception:
            pass

    signals: dict[str, list[float]] = {}

    for i, p in enumerate(positions):
        T_conf = len(tok_conf_arr) if tok_conf_arr is not None else 0
        T_gini = len(tok_gini_arr) if tok_gini_arr is not None else 0
        cut_conf = max(1, int(p * T_conf))
        cut_gini = max(1, int(p * T_gini))
        k = max(1, int(p * n_slices)) if n_slices > 0 else 0

        # ── token signals ───────────────────────────────────────────────────
        # tok_conf_prefix
        if tok_conf_arr is not None and T_conf > 0:
            v = float(np.mean(tok_conf_arr[:cut_conf]))
            signals.setdefault("tok_conf_prefix", []).append(v)
        else:
            signals.setdefault("tok_conf_prefix", []).append(0.0)

        # tok_gini_prefix
        if tok_gini_arr is not None and T_gini > 0:
            v = float(np.mean(tok_gini_arr[:cut_gini]))
            signals.setdefault("tok_gini_prefix", []).append(v)
        else:
            signals.setdefault("tok_gini_prefix", []).append(0.0)

        # tok_gini_tail (last 10% of prefix)
        if tok_gini_arr is not None and T_gini > 0:
            tail_w = max(1, int(0.1 * cut_gini))
            v = float(np.mean(tok_gini_arr[max(0, cut_gini - tail_w):cut_gini]))
            signals.setdefault("tok_gini_tail", []).append(v)
        else:
            signals.setdefault("tok_gini_tail", []).append(0.0)

        # tok_gini_slope (second-half mean minus first-half mean within prefix)
        if tok_gini_arr is not None and cut_gini >= 2:
            h = cut_gini // 2
            v = float(np.mean(tok_gini_arr[h:cut_gini])) - float(np.mean(tok_gini_arr[:h]))
            signals.setdefault("tok_gini_slope", []).append(v)
        else:
            signals.setdefault("tok_gini_slope", []).append(0.0)

        # tok_conf_recency (exponentially weighted, λ=0.3)
        if tok_conf_arr is not None and T_conf > 0:
            seg = tok_conf_arr[:cut_conf]
            w = np.exp(0.3 * np.arange(len(seg), dtype=np.float64) / len(seg))
            v = float(np.average(seg, weights=w))
            signals.setdefault("tok_conf_recency", []).append(v)
        else:
            signals.setdefault("tok_conf_recency", []).append(0.0)

        # ── neuron/structural signals ────────────────────────────────────────
        if n_slices > 0 and k > 0:
            traj = _compute_trajectory_scores(
                slices[:k], reflection_threshold=DEFAULT_REFLECTION_THRESHOLD
            )
            signals.setdefault("traj_reflection_count", []).append(float(traj["reflection_count"]))
            signals.setdefault("traj_continuity", []).append(float(traj["mean_continuity"]))
            signals.setdefault("traj_novelty", []).append(float(traj["mean_novelty"]))
            signals.setdefault("traj_max_reflection", []).append(float(traj["max_reflection"]))
            signals.setdefault("traj_late_convergence", []).append(float(traj["late_convergence"]))
        else:
            for sname in ["traj_reflection_count", "traj_continuity", "traj_novelty",
                          "traj_max_reflection", "traj_late_convergence"]:
                signals.setdefault(sname, []).append(0.0)

        # neuron count mean (from rows/ bank)
        if nc_all is not None and k > 0:
            v_mean = float(np.mean(nc_all[:k]))
            signals.setdefault("nc_mean", []).append(v_mean)
            h = max(1, k // 2)
            v_slope = float(np.nan_to_num(np.mean(nc_all[h:k]) - np.mean(nc_all[:h])))
            signals.setdefault("nc_slope", []).append(v_slope)
        else:
            signals.setdefault("nc_mean", []).append(0.0)
            signals.setdefault("nc_slope", []).append(0.0)

        # self_similarity: same value at all positions (full sequence property),
        # but only meaningful at p=1.0 — replicate the full-sequence value everywhere
        signals.setdefault("self_similarity", []).append(self_sim_full)

    return signals


# ── Per-cache scanning ────────────────────────────────────────────────────────

def scan_cache(entry, max_problems: int | None) -> dict[str, Any]:
    """
    Scan one cache entry. Returns:
      {
        "cache_key": ...,
        "signal_data": {signal_name: {"scores_per_pos": [[s0..s9], ...], "labels": [0/1, ...]}}
      }
    """
    print(f"  Scanning [{entry.cache_key}] ...")
    meta = json.loads((entry.cache_root / "meta.json").read_text(encoding="utf-8"))
    groups = build_problem_groups(meta)
    reader = CacheReader(str(entry.cache_root))
    correctness = load_correctness_map(str(entry.cache_root))

    # signal_name -> list of (score_at_pos_i, label) for each sample
    # We accumulate per position:  signal_name -> list of 10 score lists
    # Easier: signal_name -> {"pos_{i}": [scores...], "labels": [labels...]}
    accum: dict[str, dict[str, list]] = {}

    n_processed = 0
    for problem_index, (problem_id, sample_ids) in enumerate(
        sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0]))
    ):
        if max_problems is not None and problem_index >= max_problems:
            break
        for sample_id in sample_ids:
            label = int(bool(correctness.get(int(sample_id), False)))
            try:
                sigs = extract_signals_for_sample(reader, int(sample_id))
            except Exception as e:
                print(f"    WARN sample {sample_id}: {e}")
                continue

            for sname, vals10 in sigs.items():
                if sname not in accum:
                    accum[sname] = {f"p{i}": [] for i in range(10)}
                    accum[sname]["labels"] = []
                for i, v in enumerate(vals10):
                    accum[sname][f"p{i}"].append(float(v))
                accum[sname]["labels"].append(label)

            n_processed += 1

    print(f"    processed {n_processed} samples")
    return {"cache_key": entry.cache_key, "signal_data": accum}


# ── Analytics ─────────────────────────────────────────────────────────────────

def analyse_signal_data(accum: dict[str, dict]) -> dict[str, dict]:
    """
    For each signal compute:
      - AUROC at positions 10%, 50%, 100%
      - Best Spearman rho (max over positions)
      - Best position index
    """
    results = {}
    for sname, data in accum.items():
        labels = np.asarray(data["labels"], dtype=np.int32)
        aurocs = []
        rhos = []
        for i in range(10):
            scores = np.asarray(data[f"p{i}"], dtype=np.float64)
            aurocs.append(_auroc(scores, labels))
            rhos.append(_spearman_rho(scores, labels))

        best_rho_i = int(np.nanargmax(np.abs(rhos)))
        results[sname] = {
            "auroc_10pct": aurocs[0],
            "auroc_50pct": aurocs[4],
            "auroc_100pct": aurocs[9],
            "auroc_all": aurocs,
            "best_rho": rhos[best_rho_i],
            "best_rho_pos": EARLY_STOP_POSITIONS[best_rho_i],
            "rho_all": rhos,
        }
    return results


def print_table(cache_key: str, analysis: dict[str, dict]) -> None:
    print(f"\n{'='*80}")
    print(f"  {cache_key}")
    print(f"{'='*80}")
    header = f"{'Signal':<30} {'rho_best':>9} {'pos':>5} {'AUROC@10%':>10} {'AUROC@50%':>10} {'AUROC@100%':>11}"
    print(header)
    print("-" * 80)
    # sort by AUROC@10%
    for sname, d in sorted(analysis.items(), key=lambda x: -x[1]["auroc_10pct"]):
        a10 = d["auroc_10pct"]
        a50 = d["auroc_50pct"]
        a100 = d["auroc_100pct"]
        rho = d["best_rho"]
        pos = d["best_rho_pos"]
        flag = " ★" if a10 > 0.65 else (" ●" if a10 > 0.55 else "")
        print(f"  {sname:<28} {rho:>+9.4f} {pos:>5.1f} {a10:>10.4f} {a50:>10.4f} {a100:>11.4f}{flag}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Scan early-stop signals for gpqa and lcb_v5")
    ap.add_argument("--cache-root", default="MUI_HUB/cache")
    ap.add_argument("--out", default="results/scans/earlystop/earlystop_signal_scan_v1.json")
    ap.add_argument("--max-problems", type=int, default=None,
                    help="Limit number of problems per cache (for quick debug runs)")
    args = ap.parse_args()

    cache_root = REPO_ROOT / args.cache_root if not Path(args.cache_root).is_absolute() else Path(args.cache_root)
    entries = discover_cache_entries(cache_root)
    target_entries = [e for e in entries if e.dataset_name in TARGET_DATASETS]
    print(f"Found {len(entries)} total entries, {len(target_entries)} target (gpqa/lcb_v5)")

    all_results: list[dict] = []
    for entry in target_entries:
        scan_result = scan_cache(entry, max_problems=args.max_problems)
        analysis = analyse_signal_data(scan_result["signal_data"])
        print_table(entry.cache_key, analysis)
        all_results.append({
            "cache_key": entry.cache_key,
            "dataset_name": entry.dataset_name,
            "model_name": entry.model_name,
            "analysis": analysis,
        })

    out_path = REPO_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
