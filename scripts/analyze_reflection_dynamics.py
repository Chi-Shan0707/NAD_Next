#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.selectors.ml_features import _rank01
from nad.core.selectors.trajectory_impl import (
    DEFAULT_REFLECTION_THRESHOLD,
    extract_run_dynamics,
)
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


def _parse_thresholds(raw: str) -> list[float]:
    if not raw:
        return [round(x, 2) for x in np.arange(0.10, 0.601, 0.05)]
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(round(float(part), 4))
    return vals


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return None
    xv = x[valid]
    yv = y[valid]
    if np.allclose(xv, xv[0]) or np.allclose(yv, yv[0]):
        return None
    xr = _rank01(xv)
    yr = _rank01(yv)
    corr = np.corrcoef(xr, yr)[0, 1]
    if not np.isfinite(corr):
        return None
    return float(corr)


def _positive_probs(model, X: np.ndarray) -> np.ndarray:
    probs = np.asarray(model.predict_proba(X), dtype=np.float64)
    classes = np.asarray(getattr(model, "classes_", []))
    if probs.ndim == 2 and probs.shape[1] > 1:
        if classes.size:
            hit = np.where(classes == 1)[0]
            if hit.size:
                return probs[:, int(hit[0])]
        return probs[:, -1]
    if classes.size == 1 and int(classes[0]) == 1:
        return np.ones(len(X), dtype=np.float64)
    return np.zeros(len(X), dtype=np.float64)


def _selector_accuracy(model, groups: list[tuple[np.ndarray, np.ndarray]]) -> float:
    correct = total = 0
    for X_g, y_g in groups:
        probs = _positive_probs(model, X_g)
        chosen = int(np.argmax(probs))
        correct += int(y_g[chosen])
        total += 1
    return correct / total if total else 0.0


def _fit_logistic(X: np.ndarray, y: np.ndarray):
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    if np.unique(y).size < 2:
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X, y)
        return dummy

    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", class_weight="balanced")),
    ])
    pipe.fit(X, y)
    return pipe


def _as_mean(values: list[float]) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    return float(arr.mean())


def _label_name(is_correct: bool) -> str:
    return "correct" if is_correct else "incorrect"


def _build_groups(meta: dict) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for sid, sample in enumerate(meta["samples"]):
        pid = str(sample["problem_id"])
        groups.setdefault(pid, []).append(int(sid))
    return groups


def _aligned_metric_series(run_dyn: dict, metric: str) -> dict[str, np.ndarray]:
    refl = np.asarray(run_dyn["reflection_scores"], dtype=np.float64)
    metric_avg = np.asarray(run_dyn["slice_metrics"][metric], dtype=np.float64)
    deriv = run_dyn["derivatives"][metric]
    out = {
        "reflection": refl,
        "avg": metric_avg[1:] if metric_avg.size > 1 else np.zeros(0, dtype=np.float64),
        "d1": np.asarray(deriv["d1"][1:], dtype=np.float64) if len(deriv["d1"]) > 1 else np.zeros(0, dtype=np.float64),
        "d2": np.asarray(deriv["d2"][1:], dtype=np.float64) if len(deriv["d2"]) > 1 else np.zeros(0, dtype=np.float64),
    }
    out["abs_d1"] = np.abs(out["d1"])
    out["abs_d2"] = np.abs(out["d2"])
    m = min(len(out["reflection"]), *(len(v) for k, v in out.items() if k != "reflection")) if out["reflection"].size else 0
    if m <= 0:
        return {k: np.zeros(0, dtype=np.float64) for k in out}
    return {k: np.asarray(v[:m], dtype=np.float64) for k, v in out.items()}


def _collect_dataset_run_data(ds: str, cache_root: Path, base_threshold: float, max_problems: int | None):
    reader = CacheReader(str(cache_root))
    correctness = _load_ground_truth(cache_root)
    meta = json.loads((cache_root / "meta.json").read_text())
    groups = _build_groups(meta)

    result = []
    for idx, (pid, run_ids) in enumerate(sorted(groups.items(), key=lambda kv: kv[0])):
        if max_problems is not None and idx >= max_problems:
            break
        labels = np.asarray([int(bool(correctness.get(rid, False))) for rid in run_ids], dtype=np.int32)
        run_rows = []
        for rid, label in zip(run_ids, labels.tolist()):
            dyn = extract_run_dynamics(reader, int(rid), reflection_threshold=base_threshold)
            run_rows.append({
                "run_id": int(rid),
                "is_correct": bool(label),
                "reflection_scores": np.asarray(dyn["reflection_scores"], dtype=np.float64),
                "slice_metrics": {k: np.asarray(v, dtype=np.float64) for k, v in dyn["slice_metrics"].items()},
                "derivatives": {k: {kk: np.asarray(vv, dtype=np.float64) for kk, vv in dv.items()} for k, dv in dyn["derivatives"].items()},
            })
        result.append({
            "problem_id": pid,
            "run_ids": list(map(int, run_ids)),
            "labels": labels,
            "runs": run_rows,
        })
    return result


def _threshold_groups(dataset_data: dict[str, list[dict]], threshold: float) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for ds, problems in dataset_data.items():
        groups = []
        for prob in problems:
            counts = []
            for run in prob["runs"]:
                refl = np.asarray(run["reflection_scores"], dtype=np.float64)
                counts.append(float((refl > threshold).sum()))
            X_g = _rank01(np.asarray(counts, dtype=np.float64)).reshape(-1, 1)
            y_g = np.asarray(prob["labels"], dtype=np.int32)
            groups.append((X_g, y_g))
        out[ds] = groups
    return out


def _run_threshold_sweep(dataset_data: dict[str, list[dict]], thresholds: list[float]) -> list[dict]:
    rows = []
    datasets = list(dataset_data.keys())
    for threshold in thresholds:
        groups_by_ds = _threshold_groups(dataset_data, threshold)

        X_all = np.vstack([X for d in datasets for X, _ in groups_by_ds[d]])
        y_all = np.concatenate([y for d in datasets for _, y in groups_by_ds[d]])
        pooled_model = _fit_logistic(X_all, y_all)
        pooled_acc = _selector_accuracy(pooled_model, [g for d in datasets for g in groups_by_ds[d]])

        loo_per_dataset = {}
        loo_scores = []
        if len(datasets) >= 2:
            for test_ds in datasets:
                train_ds = [d for d in datasets if d != test_ds]
                X_train = np.vstack([X for d in train_ds for X, _ in groups_by_ds[d]])
                y_train = np.concatenate([y for d in train_ds for _, y in groups_by_ds[d]])
                model = _fit_logistic(X_train, y_train)
                acc = _selector_accuracy(model, groups_by_ds[test_ds])
                loo_per_dataset[test_ds] = acc
                loo_scores.append(acc)
        elif datasets:
            loo_per_dataset[datasets[0]] = pooled_acc
            loo_scores.append(pooled_acc)

        rows.append({
            "threshold": float(threshold),
            "loo_mean": float(np.mean(loo_scores)) if loo_scores else 0.0,
            "pooled_mean": float(pooled_acc),
            "loo_per_dataset": {k: float(v) for k, v in loo_per_dataset.items()},
        })
    return rows


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt_pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def main():
    ap = argparse.ArgumentParser(description="Analyze reflection dynamics and threshold sweeps")
    ap.add_argument("--datasets", default=",".join(DATASET_CACHES.keys()), help="Comma-separated datasets")
    ap.add_argument("--base-threshold", type=float, default=DEFAULT_REFLECTION_THRESHOLD, help="Base reflection threshold for dynamics/event-gap analysis")
    ap.add_argument("--thresholds", default="", help="Comma-separated threshold grid; default 0.10..0.60 step 0.05")
    ap.add_argument("--max-problems", type=int, default=None, help="Optional max problems per dataset for smoke tests")
    ap.add_argument("--out", default="results/reflection_dynamics", help="Output directory")
    args = ap.parse_args()

    requested = [d.strip() for d in args.datasets.split(",") if d.strip()]
    thresholds = _parse_thresholds(args.thresholds)
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO_ROOT)

    dataset_data: dict[str, list[dict]] = {}
    corr_bucket: dict[tuple[str, str, str, str], list[float]] = {}
    gap_bucket: dict[tuple[str, str, str, str], list[float]] = {}

    for ds in requested:
        if ds not in DATASET_CACHES:
            print(f"[WARN] Unknown dataset '{ds}', skipping.")
            continue
        cache_root = REPO_ROOT / DATASET_CACHES[ds]
        print(f"Collecting dynamics for {ds} …")
        problems = _collect_dataset_run_data(ds, cache_root, args.base_threshold, args.max_problems)
        dataset_data[ds] = problems

        for prob in tqdm(problems, desc=f"  {ds}", leave=False):
            for run in prob["runs"]:
                label = _label_name(run["is_correct"])
                for metric in ("entropy", "conf", "gini"):
                    aligned = _aligned_metric_series(run, metric)
                    refl = aligned["reflection"]
                    if refl.size == 0:
                        continue
                    for series_name in ("avg", "d1", "d2", "abs_d1", "abs_d2"):
                        corr = _spearmanr(refl, aligned[series_name])
                        if corr is not None:
                            corr_bucket.setdefault((ds, label, metric, series_name), []).append(corr)

                    event_mask = refl > float(args.base_threshold)
                    non_event_mask = ~event_mask
                    for series_name in ("abs_d1", "abs_d2"):
                        values = np.asarray(aligned[series_name], dtype=np.float64)
                        valid = np.isfinite(values)
                        ev = values[event_mask & valid]
                        non_ev = values[non_event_mask & valid]
                        if ev.size == 0 or non_ev.size == 0:
                            continue
                        gap = float(ev.mean() - non_ev.mean())
                        gap_bucket.setdefault((ds, label, metric, series_name), []).append(gap)

    threshold_rows = _run_threshold_sweep(dataset_data, thresholds)
    best_threshold_row = max(threshold_rows, key=lambda row: row["loo_mean"]) if threshold_rows else None

    corr_rows = []
    for (ds, label, metric, series_name), values in sorted(corr_bucket.items()):
        corr_rows.append({
            "dataset": ds,
            "label": label,
            "metric": metric,
            "series": series_name,
            "mean_spearman": _as_mean(values),
            "n_runs": len(values),
        })

    gap_rows = []
    for (ds, label, metric, series_name), values in sorted(gap_bucket.items()):
        gap_rows.append({
            "dataset": ds,
            "label": label,
            "metric": metric,
            "series": series_name,
            "mean_gap": _as_mean(values),
            "n_runs": len(values),
        })

    (out_dir / "correlation_summary.json").write_text(json.dumps(corr_rows, indent=2), encoding="utf-8")
    (out_dir / "event_gap_summary.json").write_text(json.dumps(gap_rows, indent=2), encoding="utf-8")
    (out_dir / "threshold_sweep_summary.json").write_text(json.dumps({
        "base_threshold": float(args.base_threshold),
        "thresholds": threshold_rows,
        "best_threshold_loo": None if best_threshold_row is None else float(best_threshold_row["threshold"]),
        "best_loo_mean": None if best_threshold_row is None else float(best_threshold_row["loo_mean"]),
        "best_pooled_mean": None if best_threshold_row is None else float(best_threshold_row["pooled_mean"]),
    }, indent=2), encoding="utf-8")

    _write_csv(out_dir / "correlation_summary.csv", corr_rows, ["dataset", "label", "metric", "series", "mean_spearman", "n_runs"])
    _write_csv(out_dir / "event_gap_summary.csv", gap_rows, ["dataset", "label", "metric", "series", "mean_gap", "n_runs"])
    _write_csv(out_dir / "threshold_sweep_summary.csv", [
        {
            "threshold": row["threshold"],
            "loo_mean": row["loo_mean"],
            "pooled_mean": row["pooled_mean"],
            **{f"loo_{ds}": row["loo_per_dataset"].get(ds) for ds in dataset_data.keys()},
        }
        for row in threshold_rows
    ], ["threshold", "loo_mean", "pooled_mean", *[f"loo_{ds}" for ds in dataset_data.keys()]])

    top_corr = sorted(
        [row for row in corr_rows if row["mean_spearman"] is not None],
        key=lambda row: abs(row["mean_spearman"]),
        reverse=True,
    )[:12]
    top_gap = sorted(
        [row for row in gap_rows if row["mean_gap"] is not None],
        key=lambda row: abs(row["mean_gap"]),
        reverse=True,
    )[:12]

    md_lines = [
        "# Reflection Dynamics Summary",
        "",
        f"- Base threshold: `{args.base_threshold:.2f}`",
        f"- Threshold grid: `{', '.join(f'{t:.2f}' for t in thresholds)}`",
        f"- Best threshold by LOO single-feature benchmark: `{best_threshold_row['threshold']:.2f}` ({_fmt_pct(best_threshold_row['loo_mean'])})" if best_threshold_row else "- No threshold sweep results",
        "",
        "## Threshold Sweep",
        "",
        "| Threshold | LOO Mean | Pooled Mean |",
        "|---|---:|---:|",
    ]
    for row in threshold_rows:
        md_lines.append(f"| `{row['threshold']:.2f}` | {_fmt_pct(row['loo_mean'])} | {_fmt_pct(row['pooled_mean'])} |")

    md_lines.extend([
        "",
        "## Top Correlations",
        "",
        "| Dataset | Label | Metric | Series | Mean Spearman | Runs |",
        "|---|---|---|---|---:|---:|",
    ])
    for row in top_corr:
        md_lines.append(
            f"| {row['dataset']} | {row['label']} | {row['metric']} | {row['series']} | {row['mean_spearman']:.4f} | {row['n_runs']} |"
        )

    md_lines.extend([
        "",
        "## Top Event vs Non-Event Gaps",
        "",
        "| Dataset | Label | Metric | Series | Mean Gap | Runs |",
        "|---|---|---|---|---:|---:|",
    ])
    for row in top_gap:
        md_lines.append(
            f"| {row['dataset']} | {row['label']} | {row['metric']} | {row['series']} | {row['mean_gap']:.4f} | {row['n_runs']} |"
        )

    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Saved reflection dynamics outputs to {out_dir}")


if __name__ == "__main__":
    main()
