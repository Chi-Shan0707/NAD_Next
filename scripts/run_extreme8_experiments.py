#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.selectors.base import SelectorContext
from nad.core.selectors.extreme8_impl import accumulate_extreme8_scores, extract_extreme8_raw_values
from nad.core.selectors.trajectory_impl import DEFAULT_REFLECTION_THRESHOLD
from nad.core.views.reader import CacheReader
from nad.ops.accuracy import _load_ground_truth

DATASET_CACHES: dict[str, str] = {
    "aime24": "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20250902_025610",
    "aime25": "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime25/cache_neuron_output_1_act_no_rms_20251126_114548",
    "brumo25": "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/brumo25/cache_neuron_output_1_act_no_rms_20251126_071142",
    "gpqa": "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/gpqa/cache_neuron_output_1_act_no_rms_20251126_111853",
    "hmmt25": "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/hmmt25/cache_neuron_output_1_act_no_rms_20251126_223151",
    "livecodebench_v5": "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/livecodebench_v5/cache_neuron_output_1_act_no_rms_20251127_032808",
}


def _load_model(path: Path):
    import joblib
    return joblib.load(path)


def _resolve_threshold(raw: str) -> float:
    if raw.lower() != "auto":
        return float(raw)
    summary_path = REPO_ROOT / "results" / "reflection_dynamics" / "threshold_sweep_summary.json"
    if not summary_path.exists():
        print(f"[WARN] {summary_path} not found; fallback to {DEFAULT_REFLECTION_THRESHOLD:.2f}")
        return float(DEFAULT_REFLECTION_THRESHOLD)
    data = json.loads(summary_path.read_text())
    value = data.get("best_threshold_loo")
    if value is None:
        print(f"[WARN] best_threshold_loo missing in {summary_path}; fallback to {DEFAULT_REFLECTION_THRESHOLD:.2f}")
        return float(DEFAULT_REFLECTION_THRESHOLD)
    return float(value)


def _build_groups(meta: dict) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for sid, sample in enumerate(meta["samples"]):
        pid = str(sample["problem_id"])
        groups.setdefault(pid, []).append(int(sid))
    return groups


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main():
    ap = argparse.ArgumentParser(description="Run Extreme8 best/worst/mixed experiments on full problems")
    ap.add_argument("--datasets", default=",".join(DATASET_CACHES.keys()), help="Comma-separated datasets")
    ap.add_argument("--best-model", default="models/ml_selectors/extreme8_best.pkl", help="Best-model path")
    ap.add_argument("--worst-model", default="models/ml_selectors/extreme8_worst.pkl", help="Worst-model path")
    ap.add_argument("--out", default="results/extreme8_experiments", help="Output directory")
    ap.add_argument("--tuple-size", type=int, default=8, help="Tuple size")
    ap.add_argument("--num-tuples", type=int, default=1024, help="Random tuples per full problem")
    ap.add_argument("--tuple-sampling", choices=("blind", "mixed_if_possible"), default="blind", help="Tuple sampling policy during evaluation")
    ap.add_argument("--reflection-threshold", default=f"{DEFAULT_REFLECTION_THRESHOLD:.2f}", help="Float threshold or 'auto'")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--max-problems", type=int, default=None, help="Optional max problems per dataset for smoke tests")
    args = ap.parse_args()

    requested = [d.strip() for d in args.datasets.split(",") if d.strip()]
    best_model = _load_model(REPO_ROOT / args.best_model)
    worst_model = _load_model(REPO_ROOT / args.worst_model)
    reflection_threshold = _resolve_threshold(str(args.reflection_threshold))
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO_ROOT)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed = {}
    summary = {
        "config": {
            "tuple_size": int(args.tuple_size),
            "num_tuples": int(args.num_tuples),
            "tuple_sampling": str(args.tuple_sampling),
            "seed": int(args.seed),
            "reflection_threshold": float(reflection_threshold),
            "best_model": args.best_model,
            "worst_model": args.worst_model,
        },
        "best_only": {"per_dataset": {}},
        "worst_only": {"per_dataset": {}},
        "best_plus_worst": {"per_dataset": {}},
        "worst_avoid": {"per_dataset": {}},
    }

    for ds_idx, ds in enumerate(requested):
        if ds not in DATASET_CACHES:
            print(f"[WARN] Unknown dataset '{ds}', skipping.")
            continue

        cache_root = REPO_ROOT / DATASET_CACHES[ds]
        reader = CacheReader(str(cache_root))
        correctness = _load_ground_truth(cache_root)
        meta = json.loads((cache_root / "meta.json").read_text())
        groups = _build_groups(meta)
        detailed[ds] = {}

        best_correct = 0
        worst_error = 0
        mix_correct = 0
        avoid_correct = 0
        total = 0

        for prob_idx, (pid, run_ids) in enumerate(sorted(groups.items(), key=lambda kv: kv[0])):
            if args.max_problems is not None and prob_idx >= args.max_problems:
                break
            labels = np.asarray([int(bool(correctness.get(rid, False))) for rid in run_ids], dtype=np.int32)
            has_mixed = bool(np.any(labels > 0) and np.any(labels <= 0))
            sampling_labels = None
            require_mixed = False
            if args.tuple_sampling == "mixed_if_possible" and has_mixed:
                sampling_labels = labels
                require_mixed = True

            ctx = SelectorContext(cache=reader, problem_id=pid, run_ids=list(map(int, run_ids)), views=[])
            raw_values = extract_extreme8_raw_values(ctx, reflection_threshold=reflection_threshold)
            payload = accumulate_extreme8_scores(
                best_model=best_model,
                worst_model=worst_model,
                raw_values=raw_values,
                tuple_size=args.tuple_size,
                num_tuples=args.num_tuples,
                seed=int(args.seed) + ds_idx * 100_000 + prob_idx,
                labels=sampling_labels,
                require_mixed=require_mixed,
            )

            score_best = np.asarray(payload["score_best"], dtype=np.float64)
            score_worst = np.asarray(payload["score_worst"], dtype=np.float64)
            score_mix = np.asarray(payload["score_mix"], dtype=np.float64)

            idx_best = int(np.argmax(score_best))
            idx_worst = int(np.argmax(score_worst))
            idx_mix = int(np.argmax(score_mix))
            idx_avoid = int(np.argmin(score_worst))

            best_ok = bool(labels[idx_best])
            worst_bad = not bool(labels[idx_worst])
            mix_ok = bool(labels[idx_mix])
            avoid_ok = bool(labels[idx_avoid])

            best_correct += int(best_ok)
            worst_error += int(worst_bad)
            mix_correct += int(mix_ok)
            avoid_correct += int(avoid_ok)
            total += 1

            detailed[ds][pid] = {
                "num_runs": len(run_ids),
                "has_mixed_labels": has_mixed,
                "tuple_sampling": str(args.tuple_sampling),
                "num_sampled_tuples": int(payload["num_tuples"][0]),
                "mean_tuple_appearances_per_run": float(np.mean(payload["counts"])) if len(payload["counts"]) else 0.0,
                "min_tuple_appearances_per_run": int(np.min(payload["counts"])) if len(payload["counts"]) else 0,
                "max_tuple_appearances_per_run": int(np.max(payload["counts"])) if len(payload["counts"]) else 0,
                "chosen_best_run_id": int(run_ids[idx_best]),
                "chosen_worst_run_id": int(run_ids[idx_worst]),
                "chosen_mix_run_id": int(run_ids[idx_mix]),
                "chosen_worst_avoid_run_id": int(run_ids[idx_avoid]),
                "best_is_correct": best_ok,
                "worst_is_incorrect": worst_bad,
                "mix_is_correct": mix_ok,
                "worst_avoid_is_correct": avoid_ok,
            }

        summary["best_only"]["per_dataset"][ds] = best_correct / total if total else 0.0
        summary["worst_only"]["per_dataset"][ds] = worst_error / total if total else 0.0
        summary["best_plus_worst"]["per_dataset"][ds] = mix_correct / total if total else 0.0
        summary["worst_avoid"]["per_dataset"][ds] = avoid_correct / total if total else 0.0

    for key in ("best_only", "worst_only", "best_plus_worst", "worst_avoid"):
        vals = list(summary[key]["per_dataset"].values())
        summary[key]["mean"] = float(np.mean(vals)) if vals else 0.0

    summary_path = out_dir / f"summary_{timestamp}.json"
    detail_path = out_dir / f"details_{timestamp}.json"
    table_path = out_dir / f"comparison_table_{timestamp}.txt"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    detail_path.write_text(json.dumps(detailed, indent=2), encoding="utf-8")

    datasets = list(summary["best_only"]["per_dataset"].keys())
    header = f"{'Method':<20s}" + "".join(f"  {ds[:8]:>8s}" for ds in datasets) + "   Mean"
    lines = [
        f"Extreme8 Experiments — {timestamp}",
        "=" * 72,
        header,
        "-" * len(header),
    ]
    for key, label in (
        ("best_only", "best-only"),
        ("worst_only", "worst-only"),
        ("best_plus_worst", "best+worst"),
        ("worst_avoid", "worst-avoid"),
    ):
        row = f"{label:<20s}" + "".join(
            f"  {summary[key]['per_dataset'][ds] * 100:7.1f}%" for ds in datasets
        )
        row += f"   {summary[key]['mean'] * 100:.1f}%"
        lines.append(row)
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"Saved summary to {summary_path}")
    print(f"Saved details to {detail_path}")
    print(f"Saved comparison table to {table_path}")


if __name__ == "__main__":
    main()
