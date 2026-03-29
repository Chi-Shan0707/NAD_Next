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


# ── single-feature ablation ───────────────────────────────────────────────────

def _selector_accuracy_from_model_single(
    model,
    groups: list[tuple[np.ndarray, np.ndarray]],
    feat_idx: int,
) -> float:
    """
    在单特征 X_g[:, feat_idx:feat_idx+1] 上评估 logistic 模型的选择准确率。
    Evaluate logistic model accuracy using only feature column feat_idx.
    """
    correct = total = 0
    for X_g, y_g in groups:
        X_single = X_g[:, feat_idx:feat_idx+1]
        scores   = model.predict_proba(X_single)[:, 1]
        chosen   = int(np.argmax(scores))
        correct += int(y_g[chosen])
        total   += 1
    return correct / total if total else 0.0


def train_single_feature_ablation(
    data: dict[str, tuple[np.ndarray, np.ndarray]],
    data_groups: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    out_dir: "Path",
):
    """
    单特征消融：逐一用每个特征（12个）单独训练 LogisticRegression，
    做留一数据集交叉验证，打印准确率汇总表，
    并将每个特征的全量训练模型保存到 out_dir/single_feat_<name>.pkl。

    Single-feature ablation: train one LogisticRegression per feature (12 total)
    using leave-one-dataset-out CV, print accuracy summary table,
    and save each full-data model to out_dir/single_feat_<name>.pkl.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    import joblib

    datasets = list(data.keys())
    if len(datasets) < 2:
        print("Need ≥ 2 datasets for single-feature ablation CV, skipping.")
        return

    print("\n=== Single-Feature Ablation (Leave-One-Dataset-Out CV) ===")
    header = f"{'Feature':<20s}" + "".join(f"  {ds[:8]:>8s}" for ds in datasets) + "   Mean"
    print(header)
    print("-" * len(header))

    X_all = np.vstack([d[0] for d in data.values()])
    y_all = np.concatenate([d[1] for d in data.values()])

    for feat_idx, fname in enumerate(FEATURE_NAMES):
        accs = []

        # 留一交叉验证 | leave-one-dataset-out CV
        for test_ds in datasets:
            train_dsets = [d for d in datasets if d != test_ds]
            X_train = np.vstack([data[d][0][:, feat_idx:feat_idx+1] for d in train_dsets])
            y_train = np.concatenate([data[d][1] for d in train_dsets])

            pipe = Pipeline([
                ("sc", StandardScaler()),
                ("lr", LogisticRegression(
                    C=1.0, max_iter=2000, solver="lbfgs", class_weight="balanced"
                )),
            ])
            pipe.fit(X_train, y_train)

            acc = _selector_accuracy_from_model_single(
                pipe, data_groups[test_ds], feat_idx
            )
            accs.append(acc)

        row  = f"{fname:<20s}" + "".join(f"  {a*100:7.1f}%" for a in accs)
        row += f"   {np.mean(accs)*100:.1f}%"
        print(row)

        # 全量训练并保存 | train on all data and save
        full_pipe = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(
                C=1.0, max_iter=2000, solver="lbfgs", class_weight="balanced"
            )),
        ])
        full_pipe.fit(X_all[:, feat_idx:feat_idx+1], y_all)
        joblib.dump(full_pipe, out_dir / f"single_feat_{fname}.pkl")

    print(f"\nSingle-feature models saved to {out_dir}/single_feat_*.pkl")


# ── temporal selector tuning ──────────────────────────────────────────────────

def tune_temporal_selector(datasets: list[str], out_dir: "Path"):
    """
    网格搜索时序折扣切片选择器的超参数 (gamma, threshold, metric)。
    在所有可用数据集上直接评估准确率（时序选择器无需训练）。
    打印结果表格，并将最优参数保存到 out_dir/temporal_best_params.json。

    Grid-search hyperparameters (gamma, threshold, metric) for TemporalSliceSelector.
    Evaluates accuracy directly on all available datasets (no training needed).
    Prints a result table and saves best params to out_dir/temporal_best_params.json.
    """
    import math as _math
    from nad.core.selectors.temporal_impl import TemporalSliceSelector

    gamma_grid     = [0.7, 0.8, 0.9, 0.95]
    threshold_grid = [0.001, 0.01, 0.1]
    metric_grid    = ["tok_conf", "tok_neg_entropy"]
    slice_size     = 32

    available_ds = [ds for ds in datasets if ds in DATASET_CACHES]
    if not available_ds:
        print("No available datasets for temporal tuning, skipping.")
        return

    # ── 预加载数据集元信息 | Preload dataset metadata ────────────────────────
    print("\nPreloading caches for temporal selector tuning …")
    ds_info: dict[str, tuple] = {}
    for ds in available_ds:
        cache_root  = Path(DATASET_CACHES[ds])
        reader      = CacheReader(str(cache_root))
        correctness = _load_ground_truth(str(cache_root))
        meta        = json.loads((cache_root / "meta.json").read_text())
        groups: dict[str, list[int]] = {}
        for sid, sample in enumerate(meta["samples"]):
            pid = str(sample["problem_id"])
            groups.setdefault(pid, []).append(sid)
        ds_info[ds] = (reader, correctness, groups)

    # ── 预计算 token 数组（每 metric × dataset 计算一次）| Precompute token arrays ──
    # token_arrays[ds][metric] = {run_id: np.ndarray | None}
    token_arrays: dict[str, dict[str, dict[int, np.ndarray | None]]] = {}
    for ds in available_ds:
        reader, _, groups = ds_info[ds]
        token_arrays[ds] = {}
        for metric in metric_grid:
            print(f"  Loading {metric} arrays for {ds} …", end="", flush=True)
            arrs: dict[int, np.ndarray | None] = {}
            for run_ids in groups.values():
                for rid in run_ids:
                    if rid in arrs:
                        continue
                    tv = reader.get_token_view(int(rid))
                    if tv is None:
                        arrs[rid] = None
                        continue
                    raw = tv.tok_conf if metric == "tok_conf" else tv.tok_neg_entropy
                    arrs[rid] = (
                        np.asarray(raw, dtype=np.float64)
                        if raw is not None and len(raw) > 0
                        else None
                    )
            token_arrays[ds][metric] = arrs
            total_runs = sum(len(rids) for rids in groups.values())
            print(f" {total_runs} runs done.")

    # ── 计算单个 run 的加权分数 | Compute per-run weighted score ─────────────
    def _score(arr, metric, gamma, threshold):
        """时序折扣切片质量分 | Temporally-discounted slice quality score."""
        if arr is None or len(arr) == 0:
            return -np.inf
        n  = len(arr)
        S  = max(1, (n + slice_size - 1) // slice_size)
        means = np.array([
            float(np.mean(arr[s * slice_size:(s + 1) * slice_size]))
            for s in range(S)
        ], dtype=np.float64)
        quality = -means if metric == "tok_conf" else means

        # K = number of slices with γ^(2k) ≥ threshold
        if gamma >= 1.0 or threshold <= 0.0:
            K = S
        elif gamma <= 0.0:
            K = 1
        else:
            log_g2  = _math.log(gamma ** 2)
            K = max(1, min(S, int(_math.floor(_math.log(threshold) / log_g2)) + 1))

        return sum(
            (gamma ** (2 * k)) * quality[S - 1 - k]
            for k in range(K)
        )

    # ── 网格搜索 | Grid search ───────────────────────────────────────────────
    print("\n=== Temporal Selector Grid Search ===")
    header = (
        f"{'metric':<16s} {'gamma':>6s} {'thresh':>7s}"
        + "".join(f"  {ds[:8]:>8s}" for ds in available_ds)
        + "   Mean"
    )
    print(header)
    print("-" * len(header))

    best_mean   = -1.0
    best_params: dict = {}

    for metric in metric_grid:
        for gamma in gamma_grid:
            for threshold in threshold_grid:
                accs = []
                for ds in available_ds:
                    _, correctness, groups = ds_info[ds]
                    arrs = token_arrays[ds][metric]
                    correct = total = 0
                    for pid, run_ids in groups.items():
                        if len(run_ids) < 2:
                            continue
                        run_scores = np.array(
                            [_score(arrs.get(rid), metric, gamma, threshold)
                             for rid in run_ids],
                            dtype=np.float64,
                        )
                        chosen   = int(np.argmax(run_scores))
                        correct += int(bool(correctness.get(run_ids[chosen], False)))
                        total   += 1
                    accs.append(correct / total if total else 0.0)

                mean_acc = float(np.mean(accs))
                row = (
                    f"{metric:<16s} {gamma:>6.2f} {threshold:>7.3f}"
                    + "".join(f"  {a * 100:7.1f}%" for a in accs)
                    + f"   {mean_acc * 100:.1f}%"
                )
                print(row)

                if mean_acc > best_mean:
                    best_mean   = mean_acc
                    best_params = {
                        "metric":     metric,
                        "gamma":      gamma,
                        "threshold":  threshold,
                        "slice_size": slice_size,
                    }

    print(f"\nBest params: {best_params}  (mean accuracy: {best_mean*100:.1f}%)")
    out_path = out_dir / "temporal_best_params.json"
    out_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    print(f"Saved to {out_path}")


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

    # ── 2b. Single-feature ablation ──────────────────────────────────────────
    train_single_feature_ablation(data, data_groups, out_dir)

    # ── 2c. Temporal selector tuning ─────────────────────────────────────────
    tune_temporal_selector(requested, out_dir)

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
