#!/usr/bin/env python3
"""Rewrite low-rank necessity tables and note from the saved summary JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from SVDomain.run_lowrank_necessity_ablation import write_outputs_from_summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize low-rank necessity outputs from saved summary JSON")
    ap.add_argument("--summary-json", default="results/scans/lowrank_necessity/lowrank_necessity_summary.json")
    ap.add_argument("--out-ablation-csv", default="results/tables/lowrank_necessity_ablation.csv")
    ap.add_argument("--out-smallest-csv", default="results/tables/lowrank_smallest_sufficient_rank.csv")
    ap.add_argument("--out-note", default="docs/07_LOWRANK_NECESSITY.md")
    args = ap.parse_args()

    summary_path = REPO_ROOT / str(args.summary_json)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    write_outputs_from_summary(
        summary=summary,
        out_ablation_csv=REPO_ROOT / str(args.out_ablation_csv),
        out_smallest_csv=REPO_ROOT / str(args.out_smallest_csv),
        out_note=REPO_ROOT / str(args.out_note),
    )
    print(f"[done] rewrote outputs from {summary_path}", flush=True)


if __name__ == "__main__":
    main()
