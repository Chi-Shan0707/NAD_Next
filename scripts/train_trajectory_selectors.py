#!/usr/bin/env python3
"""
训练轨迹融合选择器（实验 9）并评估轨迹 / 层选择器（实验 7, 8）。
Train trajectory fusion selector (Exp 9) and evaluate trajectory / layer selectors (Exp 7, 8).

从所有可用缓存中提取轨迹特征（10 维）+ 现有 ML 特征（12 维）= 22 维，
训练 LogisticRegression 融合模型并做留一数据集交叉验证。

Extract trajectory features (10-D) + existing ML features (12-D) = 22-D from all
available caches, train a LogisticRegression fusion model with leave-one-dataset-out CV.

用法（从仓库根目录运行）| Usage (from repo root):
    python scripts/train_trajectory_selectors.py [--datasets aime24,...] [--out models/ml_selectors]

输出 | Outputs
-------
models/ml_selectors/
    trajectory_fusion.pkl     Pipeline(StandardScaler, LogisticRegression)  轨迹融合模型
    trajectory_stats.json     特征统计信息 | feature stats and diagnostics
"""
from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ── repo root on PYTHONPATH ──────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.views.reader import CacheReader, ViewSpec, CutSpec, Agg, CutType, Order
from nad.core.distance.engine import DistanceEngine, DistanceSpec
from nad.core.selectors.base import SelectorContext
from nad.core.selectors.ml_features import extract_run_features, FEATURE_NAMES, N_FEATURES
from nad.core.selectors.trajectory_impl import (
    extract_trajectory_features, _extract_slice_keysets, _compute_trajectory_scores,
    _extract_layer_features, TRAJECTORY_FEATURE_NAMES, N_TRAJECTORY_FEATURES,
)
from nad.ops.accuracy import _load_ground_truth

# ── 数据集 → 缓存路径映射 | dataset → cache path mapping ────────────────────
DATASET_CACHES: dict[str, str] = {
    "aime24":          "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20250902_025610",
    "aime25":          "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime25/cache_neuron_output_1_act_no_rms_20251126_114548",
    "brumo25":         "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/brumo25/cache_neuron_output_1_act_no_rms_20251126_071142",
    "gpqa":            "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/gpqa/cache_neuron_output_1_act_no_rms_20251126_111853",
    "hmmt25":          "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/hmmt25/cache_neuron_output_1_act_no_rms_20251126_223151",
    "livecodebench_v5":"MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/livecodebench_v5/cache_neuron_output_1_act_no_rms_20251127_032808",
}

# ── 视图/距离配置 | view / distance config ───────────────────────────────────
_AGG   = Agg("max")
_CS    = CutSpec(CutType.MASS, 0.98)
_VSPEC = ViewSpec(agg=_AGG, cut=_CS, order=Order.BY_KEY)
_DSPEC = DistanceSpec(name="ja", normalize=True, num_threads=16, assume_unique=True)


# ── data collection ──────────────────────────────────────────────────────────

def _collect_dataset(cache_root: str) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """
    为单个缓存构建融合特征矩阵 X (n_samples, 22) 和标签向量 y (n_samples,)。
    Build fused feature matrix X (n_samples, 22) and label vector y (n_samples,).

    返回 | Returns: (X, y, groups)
      groups: list of (X_group, y_group) per problem — for CV evaluation.
    """
    cache_root = Path(cache_root)
    reader = CacheReader(str(cache_root))
    correctness = _load_ground_truth(str(cache_root))
    meta = json.loads((cache_root / "meta.json").read_text())

    groups: dict[str, list[int]] = {}
    for sid, sample in enumerate(meta["samples"]):
        pid = str(sample["problem_id"])
        groups.setdefault(pid, []).append(sid)

    X_rows, y_rows = [], []
    group_list = []

    for pid, run_ids in tqdm(groups.items(), desc=f"  {cache_root.name[:30]}", leave=False):
        run_ids = list(run_ids)
        n = len(run_ids)
        if n < 2:
            continue

        views = [reader.get_run_view(rid, _VSPEC, normalize_l1=True) for rid in run_ids]
        lengths = np.array([len(v.keys) for v in views], dtype=np.int32)
        D = DistanceEngine(_DSPEC).dense_matrix(views)

        ctx = SelectorContext(
            cache=reader, problem_id=pid, run_ids=run_ids, views=views
        )
        run_stats = {"lengths": lengths, "views": views}

        # Existing 12-D features
        base_feat = extract_run_features(D, run_stats, context=ctx)   # (n, 12)
        # Trajectory 10-D features
        traj_feat = extract_trajectory_features(ctx)                  # (n, 10)
        # Fused 22-D
        feat = np.hstack([base_feat, traj_feat])

        labels = np.array(
            [int(bool(correctness.get(rid, False))) for rid in run_ids],
            dtype=np.int32,
        )

        X_rows.append(feat)
        y_rows.append(labels)
        group_list.append((feat, labels))

    if not X_rows:
        return np.empty((0, N_FEATURES + N_TRAJECTORY_FEATURES)), np.empty((0,), dtype=np.int32), []

    return np.vstack(X_rows), np.concatenate(y_rows), group_list


def collect_all(datasets: list[str]):
    """收集所有数据集 | Collect all datasets. Returns {ds: (X, y, groups)}."""
    os.chdir(REPO_ROOT)
    data = {}
    for ds in datasets:
        if ds not in DATASET_CACHES:
            print(f"[WARN] Unknown dataset '{ds}', skipping.", file=sys.stderr)
            continue
        print(f"Collecting {ds} …")
        X, y, groups = _collect_dataset(DATASET_CACHES[ds])
        print(f"  {len(y)} samples, {y.mean()*100:.1f}% correct, {len(groups)} problems")
        data[ds] = (X, y, groups)
    return data


# ── selector accuracy helpers ────────────────────────────────────────────────

def _selector_acc(model, X_groups, y_groups, use_proba=True):
    correct = total = 0
    for X_g, y_g in zip(X_groups, y_groups):
        if use_proba and hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_g)[:, 1]
        else:
            scores = model.predict(X_g)
        chosen = int(np.argmax(scores))
        correct += int(y_g[chosen])
        total += 1
    return correct / total if total else 0.0


# ── evaluate Exp 7 & Exp 8 (non-ML, direct scoring) ─────────────────────────

def evaluate_trajectory_selector(datasets: list[str]):
    """
    直接评估 TrajectorySelector (Exp 7) 和 LayerStratifiedSelector (Exp 8)。
    Directly evaluate non-ML selectors on all datasets (no training needed).
    """
    from nad.core.selectors.trajectory_impl import TrajectorySelector, LayerStratifiedSelector

    available_ds = [ds for ds in datasets if ds in DATASET_CACHES]
    if not available_ds:
        print("No datasets available for Exp 7/8 evaluation.")
        return

    selectors = {
        "trajectory (Exp 7)":      TrajectorySelector(),
        "layer-stratified (Exp 8)": LayerStratifiedSelector(),
    }

    print("\n=== Exp 7 & 8: Direct Selector Evaluation ===")
    header = f"{'Selector':<30s}" + "".join(f"  {ds[:8]:>8s}" for ds in available_ds) + "   Mean"
    print(header)
    print("-" * len(header))

    for sel_name, selector in selectors.items():
        accs = []
        for ds in available_ds:
            cache_root = Path(DATASET_CACHES[ds])
            reader = CacheReader(str(cache_root))
            correctness = _load_ground_truth(str(cache_root))
            meta = json.loads((cache_root / "meta.json").read_text())

            groups: dict[str, list[int]] = {}
            for sid, sample in enumerate(meta["samples"]):
                pid = str(sample["problem_id"])
                groups.setdefault(pid, []).append(sid)

            correct = total = 0
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
                correct += int(bool(correctness.get(chosen_rid, False)))
                total += 1

            accs.append(correct / total if total else 0.0)

        row = f"{sel_name:<30s}" + "".join(f"  {a*100:7.1f}%" for a in accs)
        row += f"   {np.mean(accs)*100:.1f}%"
        print(row)


# ── leave-one-out CV for fusion model ────────────────────────────────────────

def leave_one_out_cv_fusion(data: dict):
    """
    留一数据集交叉验证融合模型 (Exp 9)。
    Leave-one-dataset-out CV for trajectory fusion model (Exp 9).
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    datasets = list(data.keys())
    if len(datasets) < 2:
        print("Need >= 2 datasets for LOO CV, skipping.")
        return

    print("\n=== Exp 9: Trajectory Fusion LOO CV ===")
    n_feat = N_FEATURES + N_TRAJECTORY_FEATURES
    header = f"{'Model':<30s}" + "".join(f"  {ds[:8]:>8s}" for ds in datasets) + "   Mean"
    print(header)
    print("-" * len(header))

    # Also compare: base 12-D logistic, trajectory-only 10-D logistic, fusion 22-D logistic
    configs = {
        "logistic (base 12-D)":      (0, N_FEATURES),
        "logistic (traj 10-D)":      (N_FEATURES, n_feat),
        "logistic (fusion 22-D)":    (0, n_feat),
    }

    for config_name, (col_lo, col_hi) in configs.items():
        accs = []
        for test_ds in datasets:
            train_dsets = [d for d in datasets if d != test_ds]
            X_train = np.vstack([data[d][0][:, col_lo:col_hi] for d in train_dsets])
            y_train = np.concatenate([data[d][1] for d in train_dsets])

            pipe = Pipeline([
                ("sc", StandardScaler()),
                ("lr", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", class_weight="balanced")),
            ])
            pipe.fit(X_train, y_train)

            _, _, test_groups = data[test_ds]
            X_groups = [g[0][:, col_lo:col_hi] for g in test_groups]
            y_groups = [g[1] for g in test_groups]
            acc = _selector_acc(pipe, X_groups, y_groups)
            accs.append(acc)

        row = f"{config_name:<30s}" + "".join(f"  {a*100:7.1f}%" for a in accs)
        row += f"   {np.mean(accs)*100:.1f}%"
        print(row)


# ── single-feature ablation for trajectory features ──────────────────────────

def trajectory_feature_ablation(data: dict):
    """
    轨迹特征单特征消融。
    Single-feature ablation for trajectory features (10-D) using LOO CV.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    datasets = list(data.keys())
    if len(datasets) < 2:
        print("Need >= 2 datasets for ablation, skipping.")
        return

    print("\n=== Trajectory Feature Ablation (LOO CV) ===")
    all_names = FEATURE_NAMES + TRAJECTORY_FEATURE_NAMES
    header = f"{'Feature':<24s}" + "".join(f"  {ds[:8]:>8s}" for ds in datasets) + "   Mean"
    print(header)
    print("-" * len(header))

    for feat_idx, fname in enumerate(all_names):
        accs = []
        for test_ds in datasets:
            train_dsets = [d for d in datasets if d != test_ds]
            X_train = np.vstack([data[d][0][:, feat_idx:feat_idx+1] for d in train_dsets])
            y_train = np.concatenate([data[d][1] for d in train_dsets])

            pipe = Pipeline([
                ("sc", StandardScaler()),
                ("lr", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", class_weight="balanced")),
            ])
            pipe.fit(X_train, y_train)

            _, _, test_groups = data[test_ds]
            correct = total = 0
            for X_g, y_g in test_groups:
                X_single = X_g[:, feat_idx:feat_idx+1]
                scores = pipe.predict_proba(X_single)[:, 1]
                chosen = int(np.argmax(scores))
                correct += int(y_g[chosen])
                total += 1
            accs.append(correct / total if total else 0.0)

        row = f"{fname:<24s}" + "".join(f"  {a*100:7.1f}%" for a in accs)
        row += f"   {np.mean(accs)*100:.1f}%"
        print(row)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train trajectory fusion selector (Exp 7-9)")
    ap.add_argument("--datasets", default=",".join(DATASET_CACHES.keys()),
                    help="Comma-separated list of datasets")
    ap.add_argument("--out", default="models/ml_selectors",
                    help="Output directory for trained models")
    args = ap.parse_args()

    requested = [d.strip() for d in args.datasets.split(",") if d.strip()]
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(REPO_ROOT)

    # ── 1. Evaluate Exp 7 & 8 (non-ML) ──────────────────────────────────────
    evaluate_trajectory_selector(requested)

    # ── 2. Collect fused features ────────────────────────────────────────────
    print("\n--- Collecting fused features (12-D base + 10-D trajectory) ---")
    data = collect_all(requested)
    if not data:
        print("No data collected. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ── 3. LOO CV for fusion models ─────────────────────────────────────────
    leave_one_out_cv_fusion(data)

    # ── 4. Feature ablation ──────────────────────────────────────────────────
    trajectory_feature_ablation(data)

    # ── 5. Train final fusion model on all data ─────────────────────────────
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    import joblib

    X_all = np.vstack([d[0] for d in data.values()])
    y_all = np.concatenate([d[1] for d in data.values()])
    print(f"\nFinal training: {len(y_all)} samples, {y_all.mean()*100:.1f}% correct")

    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", class_weight="balanced")),
    ])
    pipe.fit(X_all, y_all)

    model_path = out_dir / "trajectory_fusion.pkl"
    joblib.dump(pipe, model_path)
    print(f"Fusion model saved to {model_path}")

    # ── 6. Save diagnostics ──────────────────────────────────────────────────
    all_names = FEATURE_NAMES + TRAJECTORY_FEATURE_NAMES
    stats = {
        "n_samples": int(len(y_all)),
        "n_correct": int(y_all.sum()),
        "pct_correct": float(y_all.mean() * 100),
        "n_features": len(all_names),
        "feature_names": all_names,
        "feature_means": X_all.mean(axis=0).tolist(),
        "feature_stds": X_all.std(axis=0).tolist(),
        "logistic_coefs": (
            pipe.named_steps["lr"].coef_[0].tolist()
            if hasattr(pipe.named_steps["lr"], "coef_") else []
        ),
        "datasets": requested,
    }
    stats_path = out_dir / "trajectory_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Stats saved to {stats_path}")
    print("Done.")


if __name__ == "__main__":
    main()
