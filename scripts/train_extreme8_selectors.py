#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.selectors.base import SelectorContext
from nad.core.selectors.extreme8_impl import (
    EXTREME8_FEATURE_NAMES,
    build_extreme8_features,
    extract_extreme8_raw_values,
    sample_tuple_indices,
)
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


def _fit_logistic(X: np.ndarray, y: np.ndarray):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", class_weight="balanced")),
    ])
    pipe.fit(X, y)
    return pipe


def _selector_accuracy(model, groups: list[tuple[np.ndarray, np.ndarray]]) -> float:
    correct = total = 0
    for X_g, y_g in groups:
        probs = model.predict_proba(X_g)[:, 1]
        chosen = int(np.argmax(probs))
        correct += int(y_g[chosen])
        total += 1
    return correct / total if total else 0.0


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


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main():
    ap = argparse.ArgumentParser(description="Train pooled Extreme8 best/worst selectors")
    ap.add_argument("--datasets", default=",".join(DATASET_CACHES.keys()), help="Comma-separated datasets")
    ap.add_argument("--out", default="models/ml_selectors", help="Output directory")
    ap.add_argument("--tuple-size", type=int, default=8, help="Tuple size")
    ap.add_argument("--num-tuples", type=int, default=256, help="Random mixed tuples per eligible problem")
    ap.add_argument("--min-accuracy", type=float, default=0.10, help="Minimum problem accuracy to keep")
    ap.add_argument("--max-accuracy", type=float, default=0.90, help="Maximum problem accuracy to keep")
    ap.add_argument("--reflection-threshold", default=f"{DEFAULT_REFLECTION_THRESHOLD:.2f}", help="Float threshold or 'auto'")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--max-problems", type=int, default=None, help="Optional max eligible problems per dataset for smoke tests")
    args = ap.parse_args()

    requested = [d.strip() for d in args.datasets.split(",") if d.strip()]
    reflection_threshold = _resolve_threshold(str(args.reflection_threshold))
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO_ROOT)

    X_best_rows = []
    y_best_rows = []
    X_worst_rows = []
    y_worst_rows = []
    best_groups: list[tuple[np.ndarray, np.ndarray]] = []
    worst_groups: list[tuple[np.ndarray, np.ndarray]] = []
    dataset_problem_counts: dict[str, int] = {}

    base_seed = int(args.seed)

    for ds_idx, ds in enumerate(requested):
        if ds not in DATASET_CACHES:
            print(f"[WARN] Unknown dataset '{ds}', skipping.")
            continue

        cache_root = REPO_ROOT / DATASET_CACHES[ds]
        reader = CacheReader(str(cache_root))
        correctness = _load_ground_truth(cache_root)
        meta = json.loads((cache_root / "meta.json").read_text())
        groups = _build_groups(meta)

        kept = 0
        for pid, run_ids in tqdm(sorted(groups.items(), key=lambda kv: kv[0]), desc=f"{ds}"):
            labels = np.asarray([int(bool(correctness.get(rid, False))) for rid in run_ids], dtype=np.int32)
            acc = float(labels.mean()) if labels.size else 0.0
            if acc < args.min_accuracy or acc > args.max_accuracy:
                continue
            if labels.sum() == 0 or labels.sum() == labels.size:
                continue
            if args.max_problems is not None and kept >= args.max_problems:
                break

            ctx = SelectorContext(cache=reader, problem_id=pid, run_ids=list(map(int, run_ids)), views=[])
            raw_values = extract_extreme8_raw_values(ctx, reflection_threshold=reflection_threshold)
            rng = np.random.RandomState(base_seed + ds_idx * 100_000 + kept)
            tuples = sample_tuple_indices(
                n_runs=len(run_ids),
                tuple_size=args.tuple_size,
                num_tuples=args.num_tuples,
                rng=rng,
                labels=labels,
                require_mixed=True,
            )

            for idx in tuples:
                feat = build_extreme8_features(raw_values, idx)
                y_best = labels[idx].astype(np.int32, copy=False)
                y_worst = (1 - y_best).astype(np.int32, copy=False)
                X_best_rows.append(feat)
                y_best_rows.append(y_best)
                X_worst_rows.append(feat)
                y_worst_rows.append(y_worst)
                best_groups.append((feat, y_best))
                worst_groups.append((feat, y_worst))

            kept += 1

        dataset_problem_counts[ds] = kept

    if not X_best_rows:
        raise SystemExit("No eligible mixed 8-run problems were collected for training.")

    X_best = np.vstack(X_best_rows)
    y_best = np.concatenate(y_best_rows)
    X_worst = np.vstack(X_worst_rows)
    y_worst = np.concatenate(y_worst_rows)

    print(f"Training best-model on {len(y_best)} run samples from {len(best_groups)} tuples")
    best_model = _fit_logistic(X_best, y_best)
    print(f"Training worst-model on {len(y_worst)} run samples from {len(worst_groups)} tuples")
    worst_model = _fit_logistic(X_worst, y_worst)

    from joblib import dump

    best_path = out_dir / "extreme8_best.pkl"
    worst_path = out_dir / "extreme8_worst.pkl"
    dump(best_model, best_path)
    dump(worst_model, worst_path)

    stats = {
        "feature_names": EXTREME8_FEATURE_NAMES,
        "reflection_threshold": float(reflection_threshold),
        "tuple_size": int(args.tuple_size),
        "num_tuples_per_problem": int(args.num_tuples),
        "seed": int(args.seed),
        "datasets": requested,
        "eligible_problem_counts": dataset_problem_counts,
        "n_best_samples": int(len(y_best)),
        "n_worst_samples": int(len(y_worst)),
        "best_positive_rate": float(y_best.mean()),
        "worst_positive_rate": float(y_worst.mean()),
        "best_training_tuple_accuracy": float(_selector_accuracy(best_model, best_groups)),
        "worst_training_error_hit": float(_selector_accuracy(worst_model, worst_groups)),
        "model_paths": {
            "best": _display_path(best_path),
            "worst": _display_path(worst_path),
        },
    }
    stats_path = out_dir / "extreme8_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Saved best model to {best_path}")
    print(f"Saved worst model to {worst_path}")
    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
