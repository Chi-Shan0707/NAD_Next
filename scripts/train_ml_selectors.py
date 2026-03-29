#!/usr/bin/env python3
"""
训练基于机器学习的选择器（线性探针、逻辑回归、等渗校准）。
Train ML-based selectors (linear probe, logistic regression, isotonic calibration)
using labelled data from all available caches.

从所有可用缓存中收集带标注的训练数据（problem, run）对，训练 4 个模型并保存。
同时输出留一数据集交叉验证（leave-one-dataset-out CV）结果，提供诚实的泛化估计。

用法（从仓库根目录运行）| Usage (from repo root):
    python scripts/train_ml_selectors.py [--datasets aime24,aime25,...] [--out models/ml_selectors]

输出 | Outputs
-------
models/ml_selectors/
    linear_probe.pkl        Pipeline(StandardScaler, Ridge)          线性探针
    logistic.pkl            Pipeline(StandardScaler, LogisticRegression)  逻辑回归
    isotonic_medoid.pkl     IsotonicRegression on mean_dist_r feature     等渗校准（medoid）
    isotonic_deepconf.pkl   IsotonicRegression on dc_r feature            等渗校准（deepconf）
    feature_stats.json      类别分布、特征均值/标准差等诊断信息
                            class balance, feature means/stds for diagnostics

留一交叉验证精度会打印到标准输出。
Leave-one-dataset-out cross-validation accuracy is printed to stdout.
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
from nad.ops.accuracy import _load_ground_truth

# ── 数据集 → 缓存路径映射（相对于仓库根目录）| dataset → cache path mapping ──
DATASET_CACHES: dict[str, str] = {
    "aime24":          "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20250902_025610",
    "aime25":          "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime25/cache_neuron_output_1_act_no_rms_20251126_114548",
    "brumo25":         "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/brumo25/cache_neuron_output_1_act_no_rms_20251126_071142",
    "gpqa":            "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/gpqa/cache_neuron_output_1_act_no_rms_20251126_111853",
    "hmmt25":          "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/hmmt25/cache_neuron_output_1_act_no_rms_20251126_223151",
    "livecodebench_v5":"MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/livecodebench_v5/cache_neuron_output_1_act_no_rms_20251127_032808",
}

# ── 视图/距离配置（必须与 nad.cli analyze 默认值一致）| view / distance config ─
_AGG   = Agg("max")
_CS    = CutSpec(CutType.MASS, 0.98)
_VSPEC = ViewSpec(agg=_AGG, cut=_CS, order=Order.BY_KEY)
_DSPEC = DistanceSpec(name="ja", normalize=True, num_threads=16, assume_unique=True)


# ── data collection ──────────────────────────────────────────────────────────

def _collect_dataset(cache_root: str) -> tuple[np.ndarray, np.ndarray]:
    """
    为单个缓存构建特征矩阵 X (n_samples, N_FEATURES) 和标签向量 y (n_samples,)。
    每个样本 = 一个 (题目, run) 对。
    Build feature matrix X (n_samples, N_FEATURES) and label vector y (n_samples,)
    for one cache.  One sample = one (problem, run) pair.
    """
    cache_root = Path(cache_root)
    reader     = CacheReader(str(cache_root))
    correctness = _load_ground_truth(str(cache_root))           # sample_id -> bool
    meta       = json.loads((cache_root / "meta.json").read_text())

    # Group samples by problem_id
    groups: dict[str, list[int]] = {}
    for sid, sample in enumerate(meta["samples"]):
        pid = str(sample["problem_id"])
        groups.setdefault(pid, []).append(sid)

    X_rows, y_rows = [], []

    for pid, run_ids in tqdm(groups.items(), desc=f"  {cache_root.name[:30]}", leave=False):
        run_ids = list(run_ids)
        n = len(run_ids)
        if n < 2:
            continue   # can't compute distance matrix

        # Build RunViews
        views   = [reader.get_run_view(rid, _VSPEC, normalize_l1=True) for rid in run_ids]
        lengths = np.array([len(v.keys) for v in views], dtype=np.int32)

        # Distance matrix
        D = DistanceEngine(_DSPEC).dense_matrix(views)

        # SelectorContext for DeepConf features
        ctx = SelectorContext(
            cache=reader, problem_id=pid, run_ids=run_ids, views=views
        )

        run_stats = {"lengths": lengths, "views": views}
        feat = extract_run_features(D, run_stats, context=ctx)   # (n, N_FEATURES)

        labels = np.array(
            [int(bool(correctness.get(rid, False))) for rid in run_ids],
            dtype=np.int32
        )

        X_rows.append(feat)
        y_rows.append(labels)

    if not X_rows:
        return np.empty((0, N_FEATURES)), np.empty((0,), dtype=np.int32)

    return np.vstack(X_rows), np.concatenate(y_rows)


def collect_all(datasets: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """收集所有请求数据集的特征和标签。Return {dataset_name: (X, y)} for each requested dataset."""
    os.chdir(REPO_ROOT)     # paths are relative to repo root
    data = {}
    for ds in datasets:
        if ds not in DATASET_CACHES:
            print(f"[WARN] Unknown dataset '{ds}', skipping.", file=sys.stderr)
            continue
        print(f"Collecting {ds} …")
        X, y = _collect_dataset(DATASET_CACHES[ds])
        print(f"  {len(y)} samples, {y.mean()*100:.1f}% correct")
        data[ds] = (X, y)
    return data


# ── cross-validation (leave-one-dataset-out) ─────────────────────────────────

def _train_logistic(X, y):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", class_weight="balanced")),
    ])
    pipe.fit(X, y)
    return pipe


def _train_linear_probe(X, y):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    pipe.fit(X, y)
    return pipe


def _train_isotonic(X_col: np.ndarray, y: np.ndarray):
    """Fit IsotonicRegression (increasing) on a single feature column."""
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
    ir.fit(X_col, y)
    return ir


def _selector_accuracy_from_model(model, X_groups, y_groups, model_type="logistic"):
    """
    Given a list of (X_group, y_group) tuples for each problem,
    simulate the selector: pick argmax predicted score per group, check if correct.
    """
    correct = total = 0
    for X_g, y_g in zip(X_groups, y_groups):
        if model_type == "logistic":
            scores = model.predict_proba(X_g)[:, 1]
        elif model_type == "linear":
            scores = model.predict(X_g)
        elif model_type.startswith("isotonic_"):
            col = {"isotonic_medoid": 1, "isotonic_deepconf": 7}[model_type]
            scores = model.predict(X_g[:, col])
        else:
            raise ValueError(model_type)
        chosen = int(np.argmax(scores))
        correct += int(y_g[chosen])
        total += 1
    return correct / total if total else 0.0


def leave_one_out_cv(data: dict[str, tuple[np.ndarray, np.ndarray]],
                     data_groups: dict[str, list[tuple[np.ndarray, np.ndarray]]]):
    """留一数据集交叉验证，打印结果表格。Leave-one-dataset-out CV, print results table."""
    datasets = list(data.keys())
    if len(datasets) < 2:
        print("Need ≥ 2 datasets for leave-one-out CV, skipping.")
        return

    print("\n=== Leave-One-Dataset-Out Cross-Validation ===")
    header = f"{'Selector':<26s}" + "".join(f"  {ds[:8]:>8s}" for ds in datasets) + "   Mean"
    print(header)
    print("-" * len(header))

    results: dict[str, list[float]] = {
        "linear_probe": [], "logistic": [], "isotonic_medoid": [], "isotonic_deepconf": [],
    }

    for test_ds in datasets:
        train_dsets = [d for d in datasets if d != test_ds]
        X_train = np.vstack([data[d][0] for d in train_dsets])
        y_train = np.concatenate([data[d][1] for d in train_dsets])

        test_groups = data_groups[test_ds]

        lp   = _train_linear_probe(X_train, y_train)
        lr   = _train_logistic(X_train, y_train)
        iso_m = _train_isotonic(X_train[:, 1], y_train)
        iso_d = _train_isotonic(X_train[:, 7], y_train)

        results["linear_probe"].append(
            _selector_accuracy_from_model(lp, *zip(*test_groups), model_type="linear"))
        results["logistic"].append(
            _selector_accuracy_from_model(lr, *zip(*test_groups), model_type="logistic"))
        results["isotonic_medoid"].append(
            _selector_accuracy_from_model(iso_m, *zip(*test_groups), model_type="isotonic_medoid"))
        results["isotonic_deepconf"].append(
            _selector_accuracy_from_model(iso_d, *zip(*test_groups), model_type="isotonic_deepconf"))

    for sel, accs in results.items():
        row = f"{sel:<26s}" + "".join(f"  {a*100:7.1f}%" for a in accs)
        row += f"   {np.mean(accs)*100:.1f}%"
        print(row)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train ML-based NAD selectors")
    ap.add_argument("--datasets", default=",".join(DATASET_CACHES.keys()),
                    help="Comma-separated list of datasets to use for training")
    ap.add_argument("--out", default="models/ml_selectors",
                    help="Output directory for trained models")
    args = ap.parse_args()

    requested = [d.strip() for d in args.datasets.split(",") if d.strip()]
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Collect data ──────────────────────────────────────────────────────
    os.chdir(REPO_ROOT)
    data = collect_all(requested)
    if not data:
        print("No data collected. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Keep per-group lists for CV evaluation
    data_groups: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for ds, (X, y) in data.items():
        # We need to re-split into per-problem groups for CV accuracy.
        # We'll rebuild them by re-collecting (cheap via cached reader or just track idx).
        # Simple approach: just store (X, y) slices per problem via group metadata.
        # Because extract_run_features returns contiguous rows per problem,
        # we can split by group size.
        cache_root = Path(DATASET_CACHES[ds])
        meta = json.loads((cache_root / "meta.json").read_text())
        grp_sizes = []
        _groups: dict[str, list[int]] = {}
        for sid, s in enumerate(meta["samples"]):
            pid = str(s["problem_id"])
            _groups.setdefault(pid, []).append(sid)
        for pid, rids in _groups.items():
            if len(rids) >= 2:
                grp_sizes.append(len(rids))

        groups_list = []
        ptr = 0
        for sz in grp_sizes:
            groups_list.append((X[ptr:ptr+sz], y[ptr:ptr+sz]))
            ptr += sz
        data_groups[ds] = groups_list

    # ── 2. Leave-one-out CV ──────────────────────────────────────────────────
    leave_one_out_cv(data, data_groups)

    # ── 3. Train final models on all data ────────────────────────────────────
    X_all = np.vstack([d[0] for d in data.values()])
    y_all = np.concatenate([d[1] for d in data.values()])
    print(f"\nFinal training: {len(y_all)} samples, {y_all.mean()*100:.1f}% correct")

    import joblib

    print("Training linear probe …")
    lp = _train_linear_probe(X_all, y_all)
    joblib.dump(lp, out_dir / "linear_probe.pkl")

    print("Training logistic regression …")
    lr = _train_logistic(X_all, y_all)
    joblib.dump(lr, out_dir / "logistic.pkl")

    print("Training isotonic (medoid base) …")
    iso_m = _train_isotonic(X_all[:, 1], y_all)
    joblib.dump(iso_m, out_dir / "isotonic_medoid.pkl")

    print("Training isotonic (deepconf base) …")
    iso_d = _train_isotonic(X_all[:, 7], y_all)
    joblib.dump(iso_d, out_dir / "isotonic_deepconf.pkl")

    # ── 4. Save diagnostics ──────────────────────────────────────────────────
    stats = {
        "n_samples": int(len(y_all)),
        "n_correct": int(y_all.sum()),
        "pct_correct": float(y_all.mean() * 100),
        "datasets": requested,
        "feature_names": FEATURE_NAMES,
        "feature_means": X_all.mean(axis=0).tolist(),
        "feature_stds":  X_all.std(axis=0).tolist(),
        "logistic_coefs": (
            lr.named_steps["lr"].coef_[0].tolist()
            if hasattr(lr.named_steps["lr"], "coef_") else []
        ),
        "ridge_coefs": (
            lp.named_steps["ridge"].coef_.tolist()
            if hasattr(lp.named_steps["ridge"], "coef_") else []
        ),
    }
    (out_dir / "feature_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    print(f"\nModels saved to {out_dir}/")
    for f in sorted(out_dir.glob("*.pkl")):
        print(f"  {f.name}")
    print("Done.")


if __name__ == "__main__":
    main()
