#!/usr/bin/env python3
"""Train EarlyStop SVD low-rank routing bundle on labeled cache."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.ops.earlystop_svd import (
    SVDSearchConfig,
    save_earlystop_svd_bundle,
    train_earlystop_svd_bundle,
)


def _parse_int_list(value: str) -> tuple[int, ...]:
    vals = [int(v.strip()) for v in str(value).split(",") if v.strip()]
    if not vals:
        raise ValueError("Expected at least one integer value")
    return tuple(vals)


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


def _parse_bool_list(value: str) -> tuple[bool, ...]:
    out: list[bool] = []
    for raw in str(value).split(","):
        v = raw.strip().lower()
        if not v:
            continue
        if v in {"1", "true", "t", "yes", "y", "on"}:
            out.append(True)
        elif v in {"0", "false", "f", "no", "n", "off"}:
            out.append(False)
        else:
            raise ValueError(f"Invalid bool token: {raw!r}")
    if not out:
        raise ValueError("Expected at least one bool value")
    return tuple(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train EarlyStop SVD routing bundle")
    ap.add_argument("--cache-root", default="MUI_HUB/cache", help="Labeled training cache root")
    ap.add_argument("--out-model", default="models/ml_selectors/earlystop_svd_lowrank_lr_v1.pkl")
    ap.add_argument("--out-summary", default="results/scans/earlystop/earlystop_svd_lowrank_lr_v1_summary.json")

    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--families", default="token_only,token_plus_traj,all")
    ap.add_argument("--representations", default="raw,rank,raw+rank")
    ap.add_argument("--ranks", default="2,4,6,8,12,16")
    ap.add_argument("--c-values", default="0.1,1.0,3.0,10.0")
    ap.add_argument("--whiten-options", default="false,true")
    ap.add_argument("--class-weight-options", default="none,balanced")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--max-problems-per-cache", type=int, default=0, help="0 means all problems")

    args = ap.parse_args()

    config = SVDSearchConfig(
        n_splits=int(args.n_splits),
        family_names=_parse_str_list(args.families),
        representations=_parse_str_list(args.representations),
        ranks=_parse_int_list(args.ranks),
        c_values=_parse_float_list(args.c_values),
        whiten_options=_parse_bool_list(args.whiten_options),
        class_weight_options=_parse_str_list(args.class_weight_options),
        random_state=int(args.random_state),
        max_problems_per_cache=int(args.max_problems_per_cache),
    )

    cache_root = args.cache_root
    if not Path(cache_root).is_absolute():
        cache_root = str((REPO_ROOT / cache_root).resolve())

    bundle, summary = train_earlystop_svd_bundle(
        cache_root=cache_root,
        config=config,
    )

    out_model = Path(args.out_model)
    if not out_model.is_absolute():
        out_model = REPO_ROOT / out_model
    save_earlystop_svd_bundle(bundle, out_model)

    out_summary = Path(args.out_summary)
    if not out_summary.is_absolute():
        out_summary = REPO_ROOT / out_summary
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    totals = summary.get("totals", {})
    print("\nTraining finished")
    print(f"  Model   : {out_model}")
    print(f"  Summary : {out_summary}")
    print(f"  SVD slots      : {totals.get('svd_slots')}")
    print(f"  Baseline slots : {totals.get('baseline_slots')}")
    print(f"  Total slots    : {totals.get('total_slots')}")


if __name__ == "__main__":
    main()
