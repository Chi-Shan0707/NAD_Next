#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.distance.engine import DistanceEngine, DistanceSpec
from nad.core.selectors.base import SelectorContext
from nad.core.selectors.code_dynamic_impl import order_code_dynamic_group_indices
from nad.core.selectors.code_rns_impl import CodeRNSScorer, default_code_rns_model_path
from nad.core.views.reader import CacheReader
from nad.ops.bestofn_extreme8 import build_problem_groups, discover_cache_entries, validate_submission_payload
from scripts.run_code_baseline_v1_phase2 import DEFAULT_VIEW, _problem_sort_key

DEFAULT_BASE_SUBMISSION = (
    REPO_ROOT
    / "submission/BestofN/extreme12/patches/"
    / "extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa__no_math_patch.json"
)
DEFAULT_OUT = (
    REPO_ROOT
    / "submission/BestofN/extreme12/patches/"
    / "extreme12_baseline12_pointwise_best_only_ref030_t1024_scale100_rank__code_v2_lcb__science_hybrid_round3_gpqa__code_rns_ds_lcb_patch.json"
)
DEFAULT_CACHE_ROOT = Path("/home/jovyan/public-ro/MUI_HUB/cache_test")
TARGET_CACHE_KEYS = ("DS-R1/lcb_v5",)
DEFAULT_OVERRIDE_NOTE = "code_rns_round1=CodeRNSSelector@ja_mass1.0:ds_only"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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


def _lambda_token(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def _default_method_name(scorer: CodeRNSScorer) -> str:
    bundle = scorer.bundle
    if bundle is None:
        raise RuntimeError("CodeRNSScorer bundle is missing")
    cfg = bundle.config
    return (
        "extreme12_baseline12_pointwise_best_only_ref030_t1024"
        "__code_v2_lcb__science_hybrid_round3_gpqa"
        f"__code_rns_ds_lcb_top{int(cfg.shortlist_size)}_knn{int(cfg.knn_k)}_lam{_lambda_token(float(cfg.lambda_weight))}"
    )


def _compute_problem_scores(
    cache_root: Path,
    *,
    scorer: CodeRNSScorer,
    distance_threads: int,
) -> dict[str, dict[str, float]]:
    meta = json.loads((cache_root / "meta.json").read_text(encoding="utf-8"))
    groups = build_problem_groups(meta)
    reader = CacheReader(str(cache_root))
    engine = DistanceEngine(DistanceSpec("ja", num_threads=int(distance_threads)))
    out: dict[str, dict[str, float]] = {}

    for problem_id, run_ids in sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0])):
        run_ids = list(map(int, run_ids))
        if len(run_ids) < 2:
            continue
        views = [reader.get_run_view(int(run_id), DEFAULT_VIEW) for run_id in run_ids]
        D = engine.dense_matrix(views)
        ctx = SelectorContext(
            cache=reader,
            problem_id=str(problem_id),
            run_ids=run_ids,
            views=views,
        )
        scores = np.asarray(scorer.score_context(ctx, D), dtype=np.float64)
        order = order_code_dynamic_group_indices(scores, D, run_ids=run_ids)
        out[str(problem_id)] = _rank_scale_desc(order, run_ids)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch BestofN submission with code_rns_round1 DS-only coding scores")
    ap.add_argument("--base-submission", default=str(DEFAULT_BASE_SUBMISSION))
    ap.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--model-path", default=str(default_code_rns_model_path()))
    ap.add_argument("--distance-threads", type=int, default=8)
    ap.add_argument("--method-name", default="", help="If empty, derive from the saved code_rns bundle config")
    ap.add_argument("--code-override-note", default=DEFAULT_OVERRIDE_NOTE)
    args = ap.parse_args()

    base_submission = Path(args.base_submission)
    out_path = Path(args.out)
    cache_root = Path(args.cache_root)
    model_path = Path(args.model_path)

    base_payload = json.loads(base_submission.read_text(encoding="utf-8"))
    base_keys = list(base_payload.get("scores", {}).keys())
    for cache_key in TARGET_CACHE_KEYS:
        if cache_key not in base_payload.get("scores", {}):
            raise SystemExit(f"Base submission missing target cache key: {cache_key}")

    scorer = CodeRNSScorer.load(model_path)
    method_name = str(args.method_name).strip() or _default_method_name(scorer)
    entries = _load_target_entries(cache_root)
    patched_scores: dict[str, dict[str, dict[str, float]]] = {}
    for entry in entries:
        print(f"[patch] {entry.cache_key} <- {_display_path(entry.cache_root)}")
        patched_scores[entry.cache_key] = _compute_problem_scores(
            entry.cache_root,
            scorer=scorer,
            distance_threads=int(args.distance_threads),
        )
        n_problems = len(patched_scores[entry.cache_key])
        sample_sizes = sorted({len(problem_scores) for problem_scores in patched_scores[entry.cache_key].values()})
        print(f"  problems={n_problems} samples_per_problem={sample_sizes}")

    patched_payload = json.loads(json.dumps(base_payload))
    for cache_key, problem_scores in patched_scores.items():
        patched_payload["scores"][cache_key] = problem_scores

    bundle = scorer.bundle
    if bundle is None:
        raise RuntimeError("Loaded CodeRNSScorer has no bundle")

    patched_payload["method_name"] = method_name
    patched_payload.setdefault("score_postprocess", {})["code_override"] = str(args.code_override_note)
    patched_payload["score_postprocess"]["code_rns_patch_params"] = {
        "distance": "ja",
        "distance_threads": int(args.distance_threads),
        "model_path": str(model_path),
        "target_cache_keys": list(TARGET_CACHE_KEYS),
        "config": bundle.config.as_dict(),
        "training_summary": dict(bundle.training_summary),
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
