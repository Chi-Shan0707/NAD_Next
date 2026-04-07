#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.distance.engine import DistanceEngine, DistanceSpec
from nad.core.selectors.base import SelectorContext
from nad.core.selectors.gpqa_pairwise_impl import (
    GPQAPairwiseScorer,
    build_gpqa_pairwise_features_configurable,
    extract_gpqa_pairwise_raw,
)
from nad.core.selectors.science_hybrid_impl import (
    DEFAULT_SCIENCE_HYBRID_ALPHA,
    DEFAULT_SCIENCE_HYBRID_BACKEND,
    DEFAULT_SCIENCE_HYBRID_FAMILY,
    DEFAULT_SCIENCE_HYBRID_K,
    DEFAULT_SCIENCE_HYBRID_M,
    DEFAULT_SCIENCE_HYBRID_TAU,
    DEFAULT_SCIENCE_HYBRID_TEMPERATURE,
    ScienceHybridConfig,
    compute_science_baseline_scores_from_context,
    compute_science_hybrid_decision_from_feature_matrix,
    default_gpqa_pairwise_model_path,
)
from nad.core.views.reader import Agg, CacheReader, CutSpec, CutType, Order, ViewSpec
from nad.ops.bestofn_extreme8 import (
    build_problem_groups,
    discover_cache_entries,
    validate_submission_payload,
)

BASE_SUBMISSION = REPO_ROOT / "submission/BestofN/extreme12/base/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank.json"
DEFAULT_OUT = REPO_ROOT / "submission/BestofN/extreme12/patches/extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__science_hybrid_round3_gpqa_patch.json"
DEFAULT_CACHE_ROOT = Path("/home/jovyan/public-ro/MUI_HUB/cache_test")
TARGET_CACHE_KEYS = ("DS-R1/gpqa", "Qwen3-4B/gpqa")
DEFAULT_METHOD_NAME = "extreme12_baseline12_pointwise_best_only_ref030_t1024__science_hybrid_round3_gpqa_patch"
DEFAULT_GPQA_OVERRIDE = "science_hybrid_round3=ScienceHybridSelector@ja_mass1.0"
DEFAULT_VIEW = ViewSpec(
    agg=Agg.MAX,
    cut=CutSpec(CutType.MASS, 1.0),
    order=Order.BY_KEY,
)

_worker_reader: CacheReader | None = None
_worker_engine: DistanceEngine | None = None
_worker_scorer: GPQAPairwiseScorer | None = None
_worker_config: ScienceHybridConfig | None = None


def _worker_init(cache_root_str: str, distance_threads: int, model_path_str: str, config_dict: dict[str, float | int | str]) -> None:
    global _worker_reader, _worker_engine, _worker_scorer, _worker_config
    _worker_reader = CacheReader(cache_root_str)
    _worker_engine = DistanceEngine(DistanceSpec("ja", num_threads=int(distance_threads)))
    _worker_scorer = GPQAPairwiseScorer.load(Path(model_path_str))
    _worker_config = ScienceHybridConfig(
        family=str(config_dict["family"]),
        backend=str(config_dict["backend"]),
        tau=float(config_dict["tau"]),
        k=int(config_dict["k"]),
        alpha=float(config_dict["alpha"]),
        m=float(config_dict["m"]),
        temperature=float(config_dict["temperature"]),
    ).validate()


def _compute_problem_scores_worker(task: tuple[str, list[int]]) -> tuple[str, dict[str, float]]:
    problem_id, run_ids = task
    reader = _worker_reader
    engine = _worker_engine
    scorer = _worker_scorer
    worker_config = _worker_config
    views = [reader.get_run_view(int(run_id), DEFAULT_VIEW) for run_id in run_ids]
    D = engine.dense_matrix(views)
    ctx = SelectorContext(
        cache=reader,
        problem_id=str(problem_id),
        run_ids=run_ids,
        views=views,
    )
    baseline_scores = compute_science_baseline_scores_from_context(ctx)
    raw = extract_gpqa_pairwise_raw(ctx)
    X = build_gpqa_pairwise_features_configurable(
        raw,
        include_margin=bool(getattr(scorer, "include_margin", False)),
        include_dominance=bool(getattr(scorer, "include_dominance", False)),
    )
    decision = compute_science_hybrid_decision_from_feature_matrix(
        baseline_scores,
        X,
        scorer,
        D,
        run_ids=run_ids,
        baseline_gate_scores=np.asarray(raw["recency_conf_mean"], dtype=np.float64),
        config=worker_config,
    )
    return str(problem_id), _rank_scale_desc(decision.hybrid_order, run_ids)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _problem_sort_key(problem_id: str) -> tuple[int, str]:
    try:
        return (0, f"{int(problem_id):09d}")
    except (TypeError, ValueError):
        return (1, str(problem_id))


def _rank_scale_desc(order: np.ndarray, run_ids: list[int], lo: float = 1.0, hi: float = 100.0) -> dict[str, float]:
    n = len(run_ids)
    if n == 0:
        return {}
    if n == 1:
        return {str(run_ids[int(order[0])]): float(hi)}
    values = np.linspace(float(hi), float(lo), num=n, dtype=np.float64)
    return {
        str(run_ids[int(group_idx)]): float(values[rank_pos])
        for rank_pos, group_idx in enumerate(order.tolist())
    }


def _load_target_entries(cache_root: Path) -> list:
    entries = [entry for entry in discover_cache_entries(cache_root) if entry.cache_key in TARGET_CACHE_KEYS]
    entries.sort(key=lambda entry: entry.cache_key)
    found = {entry.cache_key for entry in entries}
    missing = sorted(set(TARGET_CACHE_KEYS) - found)
    if missing:
        raise SystemExit(f"Missing cache_test entries for target cache keys: {missing}")
    return entries


def _compute_problem_scores(
    cache_root: Path,
    *,
    distance_threads: int,
    model_path: Path,
    config: ScienceHybridConfig,
    workers: int,
) -> dict[str, dict[str, float]]:
    meta = json.loads((cache_root / "meta.json").read_text(encoding="utf-8"))
    groups = build_problem_groups(meta)
    out: dict[str, dict[str, float]] = {}
    tasks = [(str(problem_id), list(map(int, run_ids))) for problem_id, run_ids in sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0]))]
    n_total = len(tasks)
    chunksize = max(1, n_total // max(int(workers) * 4, 1))

    with Pool(
        processes=int(workers),
        initializer=_worker_init,
        initargs=(str(cache_root), int(distance_threads), str(model_path), config.as_dict()),
    ) as pool:
        completed = 0
        for problem_id, scores in pool.imap_unordered(_compute_problem_scores_worker, tasks, chunksize=chunksize):
            out[str(problem_id)] = scores
            completed += 1
            if completed % 20 == 0 or completed == n_total:
                print(f"  problems={completed}/{n_total}", flush=True)

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch best_of_n submission with science round-3 hybrid GPQA scores")
    ap.add_argument("--base-submission", default=str(BASE_SUBMISSION))
    ap.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--distance-threads", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model-path", default=str(default_gpqa_pairwise_model_path()))
    ap.add_argument("--family", default=DEFAULT_SCIENCE_HYBRID_FAMILY, choices=list(("margin_fallback", "shortlist_blend", "hard_override")))
    ap.add_argument("--backend", default=DEFAULT_SCIENCE_HYBRID_BACKEND, choices=list(("mean", "softmax_mean", "win_count", "copeland_margin")))
    ap.add_argument("--tau", type=float, default=DEFAULT_SCIENCE_HYBRID_TAU)
    ap.add_argument("--k", type=int, default=DEFAULT_SCIENCE_HYBRID_K)
    ap.add_argument("--alpha", type=float, default=DEFAULT_SCIENCE_HYBRID_ALPHA)
    ap.add_argument("--m", type=float, default=DEFAULT_SCIENCE_HYBRID_M)
    ap.add_argument("--temperature", type=float, default=DEFAULT_SCIENCE_HYBRID_TEMPERATURE)
    ap.add_argument("--method-name", default=DEFAULT_METHOD_NAME)
    ap.add_argument("--gpqa-override-note", default=DEFAULT_GPQA_OVERRIDE)
    args = ap.parse_args()

    config = ScienceHybridConfig(
        family=str(args.family),
        backend=str(args.backend),
        tau=float(args.tau),
        k=int(args.k),
        alpha=float(args.alpha),
        m=float(args.m),
        temperature=float(args.temperature),
    ).validate()

    base_submission = Path(args.base_submission)
    out_path = Path(args.out)
    cache_root = Path(args.cache_root)
    model_path = Path(args.model_path)

    base_payload = json.loads(base_submission.read_text(encoding="utf-8"))
    base_keys = list(base_payload.get("scores", {}).keys())
    for cache_key in TARGET_CACHE_KEYS:
        if cache_key not in base_payload.get("scores", {}):
            raise SystemExit(f"Base submission missing target cache key: {cache_key}")

    entries = _load_target_entries(cache_root)
    patched_scores: dict[str, dict[str, dict[str, float]]] = {}
    for entry in entries:
        print(f"[patch] {entry.cache_key} <- {_display_path(entry.cache_root)}")
        patched_scores[entry.cache_key] = _compute_problem_scores(
            entry.cache_root,
            distance_threads=int(args.distance_threads),
            model_path=model_path,
            config=config,
            workers=int(args.workers),
        )

    patched_payload = json.loads(json.dumps(base_payload))
    for cache_key, problem_scores in patched_scores.items():
        patched_payload["scores"][cache_key] = problem_scores
    patched_payload["method_name"] = str(args.method_name)
    patched_payload.setdefault("score_postprocess", {})["gpqa_override"] = str(args.gpqa_override_note)
    patched_payload["score_postprocess"]["gpqa_patch_params"] = {
        "distance_threads": int(args.distance_threads),
        "model_path": str(model_path),
        "family": str(config.family),
        "backend": str(config.backend),
        "tau": float(config.tau),
        "k": int(config.k),
        "alpha": float(config.alpha),
        "m": float(config.m),
        "temperature": float(config.temperature),
        "target_cache_keys": list(TARGET_CACHE_KEYS),
    }

    summary = validate_submission_payload(patched_payload, expected_cache_keys=base_keys)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(patched_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[saved] {_display_path(out_path)} "
        f"(cache_keys={summary['cache_keys']}, problems={summary['problems']}, samples={summary['samples']})"
    )


if __name__ == "__main__":
    main()
