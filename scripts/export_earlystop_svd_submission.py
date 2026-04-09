#!/usr/bin/env python3
"""Export EarlyStop SVD submission from cache_test."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.ops.earlystop import (
    build_earlystop_payload,
    discover_cache_entries,
    EARLY_STOP_POSITIONS,
    validate_earlystop_payload,
    write_earlystop_payload,
)
from nad.ops.earlystop_svd import (
    get_domain,
    load_earlystop_svd_bundle,
)
from scripts.run_earlystop_prefix10_svd_round1 import (
    build_feature_store,
    make_svd_bundle_score_fn,
)


def _collect_required_features(bundle: dict) -> set[str]:
    required: set[str] = set()
    for domain_bundle in bundle["domains"].values():
        for route in domain_bundle["routes"]:
            if route["route_type"] == "baseline":
                required.add(str(route["signal_name"]))
            else:
                required.update(str(name) for name in route["feature_names"])
    return required


def _problem_scores_from_payload(
    payload: dict,
    score_fn,
) -> dict[str, dict[str, list[float]]]:
    problem_scores: dict[str, dict[str, list[float]]] = {}
    tensor = payload["tensor"]
    sample_ids_all = payload["sample_ids"]
    problem_ids = payload["problem_ids"]
    problem_offsets = payload["problem_offsets"]
    n_positions = len(EARLY_STOP_POSITIONS)

    for problem_idx, problem_id in enumerate(problem_ids):
        start = int(problem_offsets[problem_idx])
        end = int(problem_offsets[problem_idx + 1])
        problem_tensor = tensor[start:end]
        sample_ids = sample_ids_all[start:end]
        run_scores = {str(sample_id): [0.0] * n_positions for sample_id in sample_ids.tolist()}

        for pos_idx in range(n_positions):
            x_raw = problem_tensor[:, pos_idx, :]
            scores = score_fn(payload["domain"], pos_idx, x_raw)
            for row_idx, sample_id in enumerate(sample_ids.tolist()):
                run_scores[str(sample_id)][pos_idx] = float(scores[row_idx])

        problem_scores[str(problem_id)] = run_scores
    return problem_scores


def main() -> None:
    ap = argparse.ArgumentParser(description="Export EarlyStop SVD submission")
    ap.add_argument("--cache-root", default="/home/jovyan/public-ro/MUI_HUB/cache_test")
    ap.add_argument("--model-path", default="models/ml_selectors/earlystop_svd_lowrank_lr_v1.pkl")
    ap.add_argument("--out-dir", default="submission/EarlyStop")
    ap.add_argument("--method-name", default="earlystop_svd_lowrank_lr_v1")
    ap.add_argument("--filename", default="earlystop_svd_lowrank_lr_v1.json")
    ap.add_argument("--max-problems", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--feature-chunk-problems", type=int, default=8)
    args = ap.parse_args()

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    bundle = load_earlystop_svd_bundle(model_path)
    required_features = _collect_required_features(bundle)
    score_fn = make_svd_bundle_score_fn(bundle)

    entries = discover_cache_entries(args.cache_root)
    print(f"Loaded bundle from {model_path}")
    print(f"Found {len(entries)} cache entries\n")

    for entry in entries:
        domain = get_domain(entry.dataset_name)
        print(f"  [{entry.cache_key}] domain={domain} queued")

    feature_store = build_feature_store(
        cache_root=args.cache_root,
        positions=tuple(float(p) for p in EARLY_STOP_POSITIONS),
        required_feature_names=required_features,
        max_problems_per_cache=args.max_problems,
        max_workers=int(args.workers),
        chunk_problems=int(args.feature_chunk_problems),
    )

    all_scores: list[tuple[str, dict]] = []
    for payload in feature_store:
        ps = _problem_scores_from_payload(payload, score_fn)
        all_scores.append((payload["cache_key"], ps))
        print(f"  [{payload['cache_key']}] domain={payload['domain']}")
        print(f"    problems : {len(ps)}")
        print(f"    samples  : {sum(len(v) for v in ps.values())}\n")

    payload = build_earlystop_payload(all_scores, method_name=args.method_name)
    stats = validate_earlystop_payload(payload)
    print(f"Validation passed: {stats}\n")

    out_path = REPO_ROOT / args.out_dir / args.filename
    write_earlystop_payload(payload, out_path)
    print(f"Written to {out_path}\n")


if __name__ == "__main__":
    main()
