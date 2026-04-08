#!/usr/bin/env python3
"""Train one BestofN SVM model and export both BestofN + EarlyStop submissions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.ops.bestofn_extreme8 import (
    validate_submission_payload as validate_bestofn_payload,
    write_submission_payload as write_bestofn_payload,
)
from nad.ops.earlystop import (
    build_earlystop_payload,
    discover_cache_entries,
    validate_earlystop_payload,
    write_earlystop_payload,
)
from nad.ops.earlystop_svm import (
    SVMEarlyStopConfig,
    load_earlystop_svm_bundle,
    save_earlystop_svm_bundle,
    score_cache_entry_bestofn_svm,
    score_cache_entry_earlystop_from_bestofn_svm,
    train_bestofn_svm_bundle,
)


def _parse_float_list(value: str) -> tuple[float, ...]:
    vals = [float(v.strip()) for v in str(value).split(",") if v.strip()]
    if not vals:
        raise ValueError("Expected at least one float value")
    return tuple(vals)


def _parse_str_list(value: str) -> tuple[str, ...]:
    vals = [str(v.strip()) for v in str(value).split(",") if v.strip()]
    if not vals:
        raise ValueError("Expected at least one token")
    return tuple(vals)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train BestofN SVM once, then export BestofN and EarlyStop submissions without retraining"
    )
    ap.add_argument("--train-cache-root", default="MUI_HUB/cache", help="Labeled training cache root")
    ap.add_argument("--test-cache-root", default="/home/jovyan/public-ro/MUI_HUB/cache_test", help="Blind cache root")

    ap.add_argument("--out-model", default="models/ml_selectors/bestofn_svm_bridge_v1.pkl")
    ap.add_argument("--out-summary", default="results/scans/earlystop/bestofn_svm_bridge_v1_summary.json")

    ap.add_argument("--bestofn-out", default="submission/BestofN/extreme12/patches/extreme12_svm_bridge_bestofn_v1.json")
    ap.add_argument("--bestofn-method-name", default="extreme12_svm_bridge_bestofn_v1")
    ap.add_argument("--bestofn-rank-scale", action="store_true", help="Rank-rescale each problem to 1..100")

    ap.add_argument("--earlystop-out", default="submission/EarlyStop/earlystop_from_bestofn_svm_bridge_v1.json")
    ap.add_argument("--earlystop-method-name", default="earlystop_from_bestofn_svm_bridge_v1")

    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--families", default="token_only")
    ap.add_argument("--representations", default="raw,rank")
    ap.add_argument("--c-values", default="0.1,1.0")
    ap.add_argument("--losses", default="squared_hinge")
    ap.add_argument("--ranksvm-backends", default="utility,win_count")
    ap.add_argument("--class-weight-options", default="none,balanced")
    ap.add_argument("--max-problems-per-cache", type=int, default=0, help="0 means all problems")
    ap.add_argument("--position-index", type=int, default=9, help="Training position (default 9=100%)")

    ap.add_argument("--load-only", action="store_true", help="Skip training and load --out-model")
    args = ap.parse_args()

    train_cache_root = args.train_cache_root
    if not Path(train_cache_root).is_absolute():
        train_cache_root = str((REPO_ROOT / train_cache_root).resolve())

    if args.load_only:
        bundle = load_earlystop_svm_bundle(REPO_ROOT / args.out_model)
        summary_payload = {
            "loaded_from": str(REPO_ROOT / args.out_model),
            "note": "load-only mode",
        }
    else:
        config = SVMEarlyStopConfig(
            n_splits=int(args.n_splits),
            family_names=_parse_str_list(args.families),
            representations=_parse_str_list(args.representations),
            c_values=_parse_float_list(args.c_values),
            losses=_parse_str_list(args.losses),
            ranksvm_backends=_parse_str_list(args.ranksvm_backends),
            class_weight_options=_parse_str_list(args.class_weight_options),
            max_problems_per_cache=int(args.max_problems_per_cache),
        )
        bundle, summary_payload = train_bestofn_svm_bundle(
            cache_root=train_cache_root,
            config=config,
            position_index=int(args.position_index),
        )
        out_model = REPO_ROOT / args.out_model
        save_earlystop_svm_bundle(bundle, out_model)
        print(f"Saved bridge model to: {out_model}")

    out_summary = REPO_ROOT / args.out_summary
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary to: {out_summary}")

    entries = discover_cache_entries(args.test_cache_root)
    expected_cache_keys = [entry.cache_key for entry in entries]

    bestofn_scores: dict[str, dict[str, dict[str, float]]] = {}
    earlystop_scores: list[tuple[str, dict]] = []
    for entry in entries:
        print(f"[export] {entry.cache_key}")
        best_map = score_cache_entry_bestofn_svm(
            entry,
            bundle,
            rank_scale_1_100=bool(args.bestofn_rank_scale),
        )
        early_map = score_cache_entry_earlystop_from_bestofn_svm(entry, bundle)
        bestofn_scores[entry.cache_key] = best_map
        earlystop_scores.append((entry.cache_key, early_map))

    bestofn_payload = {
        "task": "best_of_n",
        "method_name": str(args.bestofn_method_name),
        "scores": bestofn_scores,
        "score_postprocess": {
            "source": "svm_bridge_single_model",
            "rank_scale_1_100": bool(args.bestofn_rank_scale),
            "note": "same bestofn-trained svm model reused for earlystop export",
        },
    }
    best_summary = validate_bestofn_payload(bestofn_payload, expected_cache_keys=expected_cache_keys)
    bestofn_out = REPO_ROOT / args.bestofn_out
    write_bestofn_payload(bestofn_payload, bestofn_out)
    print(f"BestofN written: {bestofn_out} | {best_summary}")

    early_payload = build_earlystop_payload(earlystop_scores, method_name=str(args.earlystop_method_name))
    early_summary = validate_earlystop_payload(early_payload)
    early_out = REPO_ROOT / args.earlystop_out
    write_earlystop_payload(early_payload, early_out)
    print(f"EarlyStop written: {early_out} | {early_summary}")


if __name__ == "__main__":
    main()
