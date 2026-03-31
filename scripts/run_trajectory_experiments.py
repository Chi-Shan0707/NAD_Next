#!/usr/bin/env python3
"""
运行轨迹分析实验（实验 7-9）并生成对比报告。
Run trajectory-based reasoning path analysis experiments (Exp 7-9) and generate
a comparison report against existing selectors.

用法（从仓库根目录运行）| Usage (from repo root):
    python scripts/run_trajectory_experiments.py [--datasets aime24,...] [--threads 16]

输出 | Outputs
-------
results/trajectory_experiments/
    exp7_trajectory.json          实验 7 各数据集选择结果
    exp8_layer_stratified.json    实验 8 各数据集选择结果
    accuracy_summary.json         所有实验准确率汇总
    comparison_table.txt          与现有选择器的对比表
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.views.reader import CacheReader, ViewSpec, CutSpec, Agg, CutType, Order
from nad.core.distance.engine import DistanceEngine, DistanceSpec
from nad.core.selectors.base import SelectorContext
from nad.core.selectors.trajectory_impl import TrajectorySelector, LayerStratifiedSelector
from nad.ops.accuracy import _load_ground_truth

DATASET_CACHES: dict[str, str] = {
    "aime24":          "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20250902_025610",
    "aime25":          "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime25/cache_neuron_output_1_act_no_rms_20251126_114548",
    "brumo25":         "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/brumo25/cache_neuron_output_1_act_no_rms_20251126_071142",
    "gpqa":            "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/gpqa/cache_neuron_output_1_act_no_rms_20251126_111853",
    "hmmt25":          "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/hmmt25/cache_neuron_output_1_act_no_rms_20251126_223151",
    "livecodebench_v5":"MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/livecodebench_v5/cache_neuron_output_1_act_no_rms_20251127_032808",
}

_AGG   = Agg("max")
_CS    = CutSpec(CutType.MASS, 0.98)
_VSPEC = ViewSpec(agg=_AGG, cut=_CS, order=Order.BY_KEY)
_DSPEC = DistanceSpec(name="ja", normalize=True, num_threads=16, assume_unique=True)


def run_selector_on_dataset(selector, ds_name: str, cache_path: str) -> tuple[float, dict]:
    """
    在单个数据集上运行选择器，返回准确率和选择详情。
    Run a selector on a single dataset, return (accuracy, details).
    """
    cache_root = Path(cache_path)
    reader = CacheReader(str(cache_root))
    correctness = _load_ground_truth(str(cache_root))
    meta = json.loads((cache_root / "meta.json").read_text())

    groups: dict[str, list[int]] = {}
    for sid, sample in enumerate(meta["samples"]):
        pid = str(sample["problem_id"])
        groups.setdefault(pid, []).append(sid)

    correct = total = 0
    selections = {}

    for pid, run_ids in groups.items():
        run_ids = list(run_ids)
        if len(run_ids) < 2:
            continue

        views = [reader.get_run_view(rid, _VSPEC, normalize_l1=True) for rid in run_ids]
        lengths = np.array([len(v.keys) for v in views], dtype=np.int32)
        D = DistanceEngine(_DSPEC).dense_matrix(views)

        ctx = SelectorContext(
            cache=reader, problem_id=pid, run_ids=run_ids, views=views,
        )
        selector.bind(ctx)
        run_stats = {"lengths": lengths, "views": views}

        chosen_idx = selector.select(D, run_stats)
        chosen_rid = run_ids[chosen_idx]
        is_correct = bool(correctness.get(chosen_rid, False))
        correct += int(is_correct)
        total += 1
        selections[pid] = {
            "chosen_run_id": chosen_rid,
            "chosen_idx": chosen_idx,
            "is_correct": is_correct,
            "group_size": len(run_ids),
        }

    acc = correct / total if total else 0.0
    return acc, selections


def main():
    ap = argparse.ArgumentParser(description="Run trajectory experiments (Exp 7-9)")
    ap.add_argument("--datasets", default=",".join(DATASET_CACHES.keys()),
                    help="Comma-separated datasets")
    ap.add_argument("--threads", type=int, default=16,
                    help="Distance computation threads")
    args = ap.parse_args()

    global _DSPEC
    _DSPEC = DistanceSpec(name="ja", normalize=True, num_threads=args.threads, assume_unique=True)

    requested = [d.strip() for d in args.datasets.split(",") if d.strip()]
    available = [d for d in requested if d in DATASET_CACHES]

    os.chdir(REPO_ROOT)
    out_dir = REPO_ROOT / "results" / "trajectory_experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    selectors = {
        "trajectory":       TrajectorySelector(),
        "layer-stratified": LayerStratifiedSelector(),
    }

    # ── Run experiments ──────────────────────────────────────────────────────
    all_results = {}

    for sel_name, selector in selectors.items():
        print(f"\n=== {sel_name} ===")
        ds_results = {}
        for ds in available:
            print(f"  {ds} … ", end="", flush=True)
            acc, selections = run_selector_on_dataset(
                selector, ds, DATASET_CACHES[ds]
            )
            print(f"{acc*100:.1f}%")
            ds_results[ds] = {
                "accuracy": acc,
                "n_problems": len(selections),
                "selections": selections,
            }
        all_results[sel_name] = ds_results

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    header = f"{'Selector':<25s}" + "".join(f"  {ds[:8]:>8s}" for ds in available) + "   Mean"
    print(header)
    print("-" * len(header))

    summary = {}
    for sel_name, ds_results in all_results.items():
        accs = [ds_results[ds]["accuracy"] for ds in available]
        row = f"{sel_name:<25s}" + "".join(f"  {a*100:7.1f}%" for a in accs)
        row += f"   {np.mean(accs)*100:.1f}%"
        print(row)
        summary[sel_name] = {
            "per_dataset": {ds: ds_results[ds]["accuracy"] for ds in available},
            "mean": float(np.mean(accs)),
        }

    # ── Save results ─────────────────────────────────────────────────────────
    summary_path = out_dir / f"accuracy_summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nResults saved to {summary_path}")

    # Save detailed results
    for sel_name, ds_results in all_results.items():
        fname = sel_name.replace("-", "_")
        detail_path = out_dir / f"{fname}_{timestamp}.json"
        detail_path.write_text(json.dumps(ds_results, indent=2, default=str), encoding="utf-8")

    # ── Comparison table (text) ──────────────────────────────────────────────
    table_path = out_dir / f"comparison_table_{timestamp}.txt"
    lines = [
        f"Trajectory Experiments — {timestamp}",
        "=" * 70,
        header,
        "-" * len(header),
    ]
    for sel_name, ds_results in all_results.items():
        accs = [ds_results[ds]["accuracy"] for ds in available]
        row = f"{sel_name:<25s}" + "".join(f"  {a*100:7.1f}%" for a in accs)
        row += f"   {np.mean(accs)*100:.1f}%"
        lines.append(row)
    lines.append("")
    lines.append("Reference baselines (from previous experiments):")
    lines.append("  medoid:          ~50-60%")
    lines.append("  deepconf:        ~55-65%")
    lines.append("  logistic (12-D): ~70.6%")
    lines.append("  dc_z (single):   ~70.9%")
    table_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Comparison table saved to {table_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
