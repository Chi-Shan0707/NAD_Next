
from __future__ import annotations
from typing import Dict, Any
import re
import importlib
import importlib.util
import sys
import os
from .base import SelectorSpec
from .impl import (MinActivationSelector, MaxActivationSelector, MinConfidenceSelector,
                   MedoidSelector, KNNMedoidSelector, DBSCANMedoidSelector,
                   ConsensusMinSelector, ConsensusMaxSelector, DeepConfSelector, BaselineSelector,
                   GroupEnsembleMedoidSelector, GroupEnsembleDeepConfSelector,
                   TournamentCopelandSelector, TournamentDeepConfSelector,
                   TwoStageMedoidSelector, TwoStageTournamentSelector)
from .ml_impl import LinearProbeSelector, LogisticSelector, IsotonicCalibratedSelector
from .temporal_impl import TemporalSliceSelector
from .trajectory_impl import TrajectorySelector, LayerStratifiedSelector, TrajectoryFusionSelector
from .impl_legacy import (LegacyKNNMedoidSelector, LegacyMedoidSelector,
                         LegacyDBSCANMedoidSelector, LegacyConsensusMinSelector,
                         LegacyConsensusMaxSelector)

def _build_external_selector(name: str, params: dict):
    """支持外部插件选择器：
    语法：
      - py:module.path:ClassName
      - file:/abs/path/to/module.py:ClassName
    插件类需实现 nad.core.selectors.base.Selector 接口（duck-typing：实现 select(); 可选 bind()）。
    """
    # py:module.path:ClassName
    if name.startswith("py:"):
        try:
            mod_name, cls_name = name[3:].split(":", 1)
        except ValueError:
            raise ValueError(f"Invalid external selector spec '{name}'. Expected 'py:module.path:ClassName'")
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
        return cls(**params) if params else cls()
    # file:/abs/path.py:ClassName
    if name.startswith("file:"):
        try:
            path_and_cls = name[5:]
            path_str, cls_name = path_and_cls.rsplit(":", 1)
        except ValueError:
            raise ValueError(f"Invalid external selector spec '{name}'. Expected 'file:/abs/path.py:ClassName'")
        path = os.path.abspath(path_str)
        if not os.path.exists(path):
            raise FileNotFoundError(f"External selector file not found: {path}")
        module_name = f"nad_ext_{hash(path)%10**8}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load spec from {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        cls = getattr(mod, cls_name)
        return cls(**params) if params else cls()
    return None

def build_selector(spec: SelectorSpec):
    name = spec.name
    lowered = name.lower()
    params = spec.params or {}

    # Try external first
    ext = _build_external_selector(name, params)
    if ext is not None:
        return ext

    # Baseline selectors (avgN@, conN@) - virtual selectors for statistical baselines
    # Pattern: avg<number>@ or con<number>@
    if re.match(r'^(avg|con)\d+@$', lowered):
        return BaselineSelector()

    # Legacy selectors (using old algorithm with similarity matrix)
    if lowered in ("legacy-knn-medoid", "legacy_knn_medoid", "legacy-knn"):
        k = int(params.get("k", 3))
        return LegacyKNNMedoidSelector(k=k)
    if lowered in ("legacy-medoid",):
        return LegacyMedoidSelector()
    if lowered in ("legacy-dbscan-medoid", "legacy_dbscan_medoid", "legacy-dbscan"):
        eps = float(params.get("eps", 0.3))
        ms = int(params.get("min_samples", 3))
        # 默认启用legacy-JA（使用去重的keys计算Jaccard距离）
        use_legacy_ja = bool(params.get("use_legacy_ja", True))
        return LegacyDBSCANMedoidSelector(eps=eps, min_samples=ms, use_legacy_ja=use_legacy_ja)
    if lowered in ("legacy-consensus-min", "legacy_consensus_min"):
        k = int(params.get("k", 3))
        eps = float(params.get("eps", 0.3))
        ms = int(params.get("min_samples", 3))
        return LegacyConsensusMinSelector(k=k, eps=eps, min_samples=ms)
    if lowered in ("legacy-consensus-max", "legacy_consensus_max"):
        k = int(params.get("k", 3))
        eps = float(params.get("eps", 0.3))
        ms = int(params.get("min_samples", 3))
        return LegacyConsensusMaxSelector(k=k, eps=eps, min_samples=ms)

    # Current selectors
    if lowered in ("min-activation", "min_activation", "min"):
        return MinActivationSelector()
    if lowered in ("max-activation", "max_activation", "max"):
        return MaxActivationSelector()
    if lowered in ("min-confidence", "min_confidence", "minconf"):
        return MinConfidenceSelector()
    if lowered in ("medoid",):
        return MedoidSelector()
    if lowered in ("knn-medoid", "knn_medoid", "knn"):
        k = int(params.get("k", 3))
        return KNNMedoidSelector(k=k)
    if lowered in ("dbscan-medoid","dbscan_medoid","dbscan"):
        raw_eps = params.get("eps", 'auto')
        eps = None if (raw_eps is None or str(raw_eps).lower()=='auto') else float(raw_eps)
        ms  = int(params.get("min_samples", 3))
        return DBSCANMedoidSelector(eps=eps, min_samples=ms)
    if lowered in ("consensus-min","consensus_min","consensusmin"):
        k = int(params.get("k", 5))
        raw_eps = params.get("eps", 0.25)
        eps = None if (raw_eps is None or str(raw_eps).lower()=='auto') else float(raw_eps)
        ms  = int(params.get("min_samples", 3))
        return ConsensusMinSelector(k=k, eps=eps, min_samples=ms)
    if lowered in ("consensus-max","consensus_max","consensusmax"):
        k = int(params.get("k", 5))
        raw_eps = params.get("eps", 0.25)
        eps = None if (raw_eps is None or str(raw_eps).lower()=='auto') else float(raw_eps)
        ms  = int(params.get("min_samples", 3))
        return ConsensusMaxSelector(k=k, eps=eps, min_samples=ms)

    # DeepConf selector - using token-level confidence metrics
    if lowered in ("deepconf", "deep_conf", "deep-conf", "dc"):
        metric = params.get("metric", "tok_conf")
        reduction = params.get("reduction", "min_group")
        group_size = int(params.get("group_size", 20))
        return DeepConfSelector(metric=metric, reduction=reduction, group_size=group_size)

    # Ensemble selectors (分组淘汰)
    if lowered in ("ensemble-medoid", "ensemble_medoid"):
        gs = int(params.get("group_size", 8))
        seed = int(params.get("seed", 42))
        return GroupEnsembleMedoidSelector(group_size=gs, seed=seed)
    if lowered in ("ensemble-deepconf", "ensemble_deepconf"):
        gs = int(params.get("group_size", 8))
        metric = params.get("metric", "tok_conf")
        seed = int(params.get("seed", 42))
        return GroupEnsembleDeepConfSelector(group_size=gs, metric=metric, seed=seed)

    # Tournament selectors (两两比较 + softmax)
    if lowered in ("tournament-copeland", "tournament_copeland"):
        temp = float(params.get("temperature", 0.2))
        seed = int(params.get("seed", 42))
        return TournamentCopelandSelector(temperature=temp, seed=seed)
    if lowered in ("tournament-deepconf", "tournament_deepconf"):
        metric = params.get("metric", "tok_conf")
        temp = float(params.get("temperature", 0.2))
        seed = int(params.get("seed", 42))
        return TournamentDeepConfSelector(metric=metric, temperature=temp, seed=seed)

    # Two-stage selectors (分组 Top-K → 决赛)
    if lowered in ("twostage-medoid", "twostage_medoid", "2stage-medoid"):
        gs = int(params.get("group_size", 16))
        tk = int(params.get("top_k", 4))
        seed = int(params.get("seed", 42))
        return TwoStageMedoidSelector(group_size=gs, top_k=tk, seed=seed)
    if lowered in ("twostage-tournament", "twostage_tournament", "2stage-tournament"):
        gs = int(params.get("group_size", 16))
        tk = int(params.get("top_k", 4))
        temp = float(params.get("temperature", 0.2))
        seed = int(params.get("seed", 42))
        return TwoStageTournamentSelector(group_size=gs, top_k=tk, temperature=temp, seed=seed)

    # ML selectors (require pre-trained models from scripts/train_ml_selectors.py)
    if lowered in ("linear-probe", "linear_probe", "linearprobe"):
        mp = params.get("model_path", None)
        return LinearProbeSelector(model_path=mp)
    if lowered in ("logistic", "logistic-regression", "logistic_regression"):
        mp = params.get("model_path", None)
        return LogisticSelector(model_path=mp)
    if lowered in ("isotonic-medoid", "isotonic_medoid"):
        mp = params.get("model_path", None)
        return IsotonicCalibratedSelector(base="medoid", model_path=mp)
    if lowered in ("isotonic-deepconf", "isotonic_deepconf"):
        mp = params.get("model_path", None)
        return IsotonicCalibratedSelector(base="deepconf", model_path=mp)

    # Temporal slice selector (时序折扣切片选择器)
    # params: metric, gamma, threshold, slice_size
    if lowered in ("temporal-slice", "temporal_slice"):
        return TemporalSliceSelector(
            metric=params.get("metric", "tok_conf"),
            gamma=float(params.get("gamma", 0.9)),
            threshold=float(params.get("threshold", 0.01)),
            slice_size=int(params.get("slice_size", 32)),
        )

    # Trajectory selectors (轨迹分析选择器) — Exp 7, 8, 9
    if lowered in ("trajectory", "trajectory-structure", "trajectory_structure"):
        return TrajectorySelector(
            alpha=float(params.get("alpha", 1.0)),
            beta=float(params.get("beta", 0.5)),
            gamma=float(params.get("gamma", 0.3)),
            delta=float(params.get("delta", 0.2)),
        )
    if lowered in ("layer-stratified", "layer_stratified"):
        return LayerStratifiedSelector(
            alpha=float(params.get("alpha", 1.0)),
            beta=float(params.get("beta", 0.5)),
            gamma=float(params.get("gamma", 0.3)),
        )
    if lowered in ("trajectory-fusion", "trajectory_fusion"):
        mp = params.get("model_path", None)
        return TrajectoryFusionSelector(model_path=mp)

    raise ValueError(f"Unknown selector: {spec.name}")

def expand_selector_all(cache_root: str = None):
    """
    Expand "all" selector to include all standard selectors plus baselines.

    Args:
        cache_root: Path to cache directory (optional, for auto-detecting N in avgN@/conN@)

    Returns:
        List of selector specifications
    """
    selectors = [
        {"name":"min-activation"},
        {"name":"max-activation"},
        {"name":"min-confidence"},
        {"name":"medoid"},
        {"name":"knn-medoid","params":{"k":3}},
        {"name":"dbscan-medoid","params":{"eps":"auto","min_samples":3}},
        {"name":"consensus-min","params":{"k":3,"eps":"auto","min_samples":3}},
        {"name":"consensus-max","params":{"k":3,"eps":"auto","min_samples":3}},
        {"name":"deepconf"},  # DeepConf selector with default parameters
        {"name":"ensemble-medoid","params":{"group_size":8,"seed":42}},
        {"name":"ensemble-deepconf","params":{"group_size":8,"seed":42}},
        {"name":"tournament-copeland","params":{"temperature":0.2,"seed":42}},
        {"name":"tournament-deepconf","params":{"temperature":0.2,"seed":42}},
        {"name":"twostage-medoid","params":{"group_size":16,"top_k":4,"seed":42}},
        {"name":"twostage-tournament","params":{"group_size":16,"top_k":4,"temperature":0.2,"seed":42}},
    ]

    # Add baseline selectors (avgN@, conN@)
    # Try to auto-detect N from cache, otherwise default to 64
    n_runs = 64  # default

    if cache_root:
        try:
            import json
            from pathlib import Path
            meta_path = Path(cache_root) / "meta.json"
            if meta_path.exists():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    # Count runs per problem to detect N
                    if "samples" in meta and len(meta["samples"]) > 0:
                        problem_runs = {}
                        for sample in meta["samples"]:
                            pid = sample.get("problem_id")
                            if pid not in problem_runs:
                                problem_runs[pid] = 0
                            problem_runs[pid] += 1
                        # Use the most common N (should be same for all problems)
                        if problem_runs:
                            n_runs = max(problem_runs.values())
        except Exception:
            pass  # fallback to default 64

    # Add baseline selectors
    selectors.extend([
        {"name": f"avg{n_runs}@"},
        {"name": f"con{n_runs}@"},
    ])

    return selectors
