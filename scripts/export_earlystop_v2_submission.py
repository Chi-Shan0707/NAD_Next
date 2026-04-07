#!/usr/bin/env python3
"""Export Early Stop v2 (domain-aware) submission from cache_test."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.ops.earlystop import (
    build_earlystop_payload,
    discover_cache_entries,
    validate_earlystop_payload,
    write_earlystop_payload,
)
from nad.ops.earlystop_v2 import get_domain, score_cache_entry_earlystop_v2


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Early Stop v2 (domain-aware) submission")
    ap.add_argument("--cache-root", default="/home/jovyan/public-ro/MUI_HUB/cache_test")
    ap.add_argument("--out-dir", default="submission/EarlyStop")
    ap.add_argument("--method-name", default="domain_aware_v2")
    ap.add_argument("--filename", default="domain_aware_v2.json")
    ap.add_argument("--max-problems", type=int, default=None)
    args = ap.parse_args()

    entries = discover_cache_entries(args.cache_root)
    print(f"Found {len(entries)} cache entries\n")

    all_scores: list[tuple[str, dict]] = []
    for entry in entries:
        domain = get_domain(entry.dataset_name)
        print(f"  [{entry.cache_key}]  domain={domain}")
        ps = score_cache_entry_earlystop_v2(entry, max_problems=args.max_problems)
        all_scores.append((entry.cache_key, ps))
        n_probs = len(ps)
        n_samps = sum(len(v) for v in ps.values())
        print(f"    problems : {n_probs}")
        print(f"    samples  : {n_samps}\n")

    payload = build_earlystop_payload(all_scores, method_name=args.method_name)
    stats = validate_earlystop_payload(payload)
    print(f"Validation passed: {stats}\n")

    out_path = REPO_ROOT / args.out_dir / args.filename
    write_earlystop_payload(payload, out_path)
    print(f"Written to {out_path}\n")


if __name__ == "__main__":
    main()
